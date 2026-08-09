"""Capture the exact software, model and hardware identity of one grid invocation.

This script never records API keys. It writes one immutable, timestamped JSON document per
invocation so runs made on the control PC, Ollama workstation and H100 cluster can later be
audited separately.

Examples::

    python scripts/capture_run_manifest.py --config configs/grid.json --models GEMMA,QWEN
    python scripts/capture_run_manifest.py --config configs/grid_all_systems.json \
        --models LOCAL-QWEN3-8B
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from tabbench_llm.config import model_keys, resolve_list
from tabbench_llm.llm.client import build_client
from tabbench_llm.llm.registry import LLM_MODELS


def _run(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_head(start: Path) -> str | None:
    """Resolve HEAD without requiring the ``git`` executable to be on PATH."""
    for root in (start, *start.parents):
        git_dir = root / ".git"
        if git_dir.is_file():
            line = git_dir.read_text().strip()
            if line.startswith("gitdir:"):
                candidate = Path(line.split(":", 1)[1].strip())
                git_dir = candidate if candidate.is_absolute() else (root / candidate).resolve()
        if not git_dir.is_dir():
            continue
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref_name = head.split(":", 1)[1].strip()
        loose = git_dir / ref_name
        if loose.is_file():
            return loose.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref_name:
                        return commit
        return None
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _hf_revision(model_id: str) -> str | None:
    """Return the locally resolved Hugging Face ``main`` commit, if cached."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / ("models--" + model_id.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        value = ref.read_text().strip()
        return value or None
    return None


def _ollama_tags() -> tuple[dict[str, dict], str | None]:
    base = os.environ.get("TABBENCH_LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=10) as response:
            payload = json.load(response)
        tags: dict[str, dict] = {}
        for model in payload.get("models", []):
            name = model["name"]
            tags[name] = model
            if name.endswith(":latest"):
                tags[name.removesuffix(":latest")] = model
        return tags, None
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _advertised_models(provider: str) -> tuple[list[str], str | None]:
    try:
        page = build_client(provider).models.list()
        return sorted(str(model.id) for model in page.data), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _model_record(key: str, ollama: dict[str, dict]) -> dict:
    info = dict(LLM_MODELS[key])
    provider = str(info.get("provider", "ml-cloud"))
    api_model = str(info["api_model"])
    record = {"key": key, **info}

    if provider == "ollama":
        tag = ollama.get(api_model)
        if tag:
            record["resolved_revision"] = tag.get("digest")
            record["ollama_details"] = tag.get("details")
            record["identity_quality"] = "immutable_digest"
        else:
            record["resolved_revision"] = None
            record["identity_quality"] = "unresolved_alias"
    elif provider == "local":
        revision = str(info.get("revision") or "") or _hf_revision(api_model)
        record["resolved_revision"] = revision
        record["identity_quality"] = (
            "immutable_huggingface_commit" if revision else "unresolved_huggingface_ref"
        )
    else:
        revision = info.get("revision")
        record["resolved_revision"] = revision
        record["identity_quality"] = "provider_revision" if revision else "provider_alias_only"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="override the config's output root")
    parser.add_argument("--models", help="comma-separated model slice")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    roster = model_keys(resolve_list(config["models"], str(config_path.parent)))
    selected = (
        [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list(roster)
    )
    unknown = [m for m in selected if m not in roster]
    if unknown:
        raise SystemExit(f"models not in config roster: {unknown}")

    llm_keys = [m for m in selected if m in LLM_MODELS]
    providers = sorted({str(LLM_MODELS[m].get("provider", "ml-cloud")) for m in llm_keys})
    ollama, ollama_error = _ollama_tags() if "ollama" in providers else ({}, None)
    advertised = {}
    for provider in providers:
        ids, error = _advertised_models(provider)
        advertised[provider] = {"model_ids": ids, "error": error}

    gpu = _run(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader",
    )
    git_commit = _run("git", "rev-parse", "HEAD") or _git_head(Path.cwd())
    git_status = _run("git", "status", "--short")
    if not git_commit:
        raise SystemExit("Could not resolve the Git commit; refusing an unidentified run.")
    now = datetime.now(UTC)
    output_root = Path(args.output or config["output"]).resolve()
    manifest_dir = output_root / "run_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    safe_slice = "-".join(re.sub(r"[^A-Za-z0-9_.-]", "_", m) for m in selected)
    filename = f"{now.strftime('%Y%m%dT%H%M%SZ')}__{safe_slice or 'none'}.json"

    manifest = {
        "schema_version": 1,
        "created_utc": now.isoformat(),
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "canonical_roster": roster,
            "selected_models": selected,
        },
        "git": {
            "commit": git_commit,
            "dirty": None if git_status is None else bool(git_status),
            "status": git_status,
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "gpu": gpu,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ollama_version": _run("ollama", "--version"),
        },
        "packages": _packages(),
        "models": [_model_record(key, ollama) for key in llm_keys],
        "provider_advertised_models": advertised,
        "ollama_tags_error": ollama_error,
        "identity_note": (
            "immutable_digest / immutable_huggingface_commit are frozen identities; "
            "provider_alias_only and unresolved_* require provider-side revision metadata "
            "before a confirmatory publication claim."
        ),
    }
    path = manifest_dir / filename
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
