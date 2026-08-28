"""Safely merge raw grid outputs copied from another machine.

Only immutable raw artifacts (configs, predictions, probabilities, status/statistics and run
manifests) are merged. Derived metrics, diagnostic logs and aggregate CSVs are deliberately
ignored; recompute them after importing whichever machine slices are currently available.

If the destination already contains a different file at the same path, the script stops rather
than choosing one silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

DERIVED_NAMES = {
    "grid_metrics.csv",
    "feature_grid_metrics.csv",
    "feature_grid_summary.csv",
}
IGNORED_DIRS = {"metrics", "logs"}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_derived_or_log(relative: Path) -> bool:
    """Return whether *relative* can be regenerated and should not be transferred."""
    return bool(IGNORED_DIRS.intersection(relative.parts)) or relative.name in DERIVED_NAMES


def _portable_config(path: Path) -> dict:
    """Load a cell config with platform-specific output separators normalised."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    output_dir = payload.get("output_dir")
    if isinstance(output_dir, str):
        payload["output_dir"] = output_dir.replace("\\", "/")
    return payload


def _files_equivalent(source: Path, dest: Path, relative: Path) -> bool:
    if _digest(source) == _digest(dest):
        return True
    if relative.name == "config.json":
        try:
            source_config = _portable_config(source)
            dest_config = _portable_config(dest)
        except (OSError, ValueError):
            return False
        # Only a roster extension is tolerated. Anything that is not a pair of cell configs
        # carrying a model list is treated as a genuine conflict for the caller to report.
        if "models" not in source_config or "models" not in dest_config:
            return False
        source_models = set(source_config.pop("models"))
        dest_models = set(dest_config.pop("models"))
        return source_config == dest_config and source_models.issubset(dest_models)
    if relative.suffix.lower() == ".csv":
        try:
            source_text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
            dest_text = dest.read_text(encoding="utf-8").replace("\r\n", "\n")
            return source_text == dest_text
        except (OSError, UnicodeDecodeError):
            return False
    if relative.suffix.lower() == ".json" and "stats" in relative.parts:
        try:
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            dest_payload = json.loads(dest.read_text(encoding="utf-8"))
            source_payload.pop("timestamp", None)
            dest_payload.pop("timestamp", None)
            return source_payload == dest_payload
        except (OSError, ValueError, TypeError):
            return False
    return False


def _is_newer_pass_replacing_retry(source: Path, dest: Path, relative: Path) -> bool:
    if relative.suffix.lower() != ".json" or "stats" not in relative.parts:
        return False
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    dest_payload = json.loads(dest.read_text(encoding="utf-8"))
    return (
        source_payload["dataset"] == dest_payload["dataset"]
        and source_payload["model"] == dest_payload["model"]
        and source_payload["status"] == "pass"
        and dest_payload["status"] == "retry"
        and datetime.fromisoformat(source_payload["timestamp"])
        > datetime.fromisoformat(dest_payload["timestamp"])
    )


def _conflicting_ground_truth_units(source: Path, dest: Path) -> set[tuple[Path, str]]:
    """Return ``(seed_dir, key)`` units whose shared truth values genuinely differ."""
    conflicts = set()
    for src in source.rglob("*_ground_truth.csv"):
        relative = src.relative_to(source)
        dst = dest / relative
        if dst.exists() and not _files_equivalent(src, dst, relative):
            key = src.name.removesuffix("_ground_truth.csv")
            conflicts.add((relative.parents[1], key))
    return conflicts


def _belongs_to_unit(relative: Path, units: set[tuple[Path, str]]) -> bool:
    for seed_dir, key in units:
        try:
            below_seed = relative.relative_to(seed_dir)
        except ValueError:
            continue
        if len(below_seed.parts) == 2 and below_seed.parts[0] in {"predictions", "stats"}:
            if below_seed.name == f"{key}_ground_truth.csv" or below_seed.name.startswith(
                f"{key}_"
            ):
                return True
    return False


def merge_results(
    source: Path,
    dest: Path,
    *,
    keep_dest_conflicting_units: bool = False,
    replace_dest_retries: bool = False,
) -> dict[str, int]:
    """Conflict-check and merge one raw result directory into another."""
    source = source.resolve()
    dest = dest.resolve()
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    if source == dest:
        raise ValueError("source and destination are the same directory")

    conflicting_units = (
        _conflicting_ground_truth_units(source, dest) if keep_dest_conflicting_units else set()
    )
    counts = {
        "copied": 0,
        "identical": 0,
        "ignored_derived": 0,
        "skipped_conflicting_unit": 0,
        "replaced_retry": 0,
    }
    for src in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = src.relative_to(source)
        if _belongs_to_unit(relative, conflicting_units):
            counts["skipped_conflicting_unit"] += 1
            continue
        if is_derived_or_log(relative):
            counts["ignored_derived"] += 1
            continue
        dst = dest / relative
        if dst.exists():
            if not _files_equivalent(src, dst, relative):
                if replace_dest_retries and _is_newer_pass_replacing_retry(src, dst, relative):
                    shutil.copy2(src, dst)
                    counts["replaced_retry"] += 1
                    continue
                raise ValueError(
                    f"conflict at {relative}: source and destination differ; "
                    "keep both result roots and investigate the duplicate run"
                )
            counts["identical"] += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        counts["copied"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument(
        "--keep-dest-conflicting-units",
        action="store_true",
        help=(
            "when shared ground truth differs, keep the destination and skip every source "
            "artifact for that dataset/fold unit; all other conflicts still fail"
        ),
    )
    parser.add_argument(
        "--replace-dest-retries",
        action="store_true",
        help="replace an older retry stats record with a newer pass for the same unit",
    )
    args = parser.parse_args()

    try:
        counts = merge_results(
            args.source,
            args.dest,
            keep_dest_conflicting_units=args.keep_dest_conflicting_units,
            replace_dest_retries=args.replace_dest_retries,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    print("Next: python scripts/finalize_grid.py --config configs/grid_all_systems.json")


if __name__ == "__main__":
    main()
