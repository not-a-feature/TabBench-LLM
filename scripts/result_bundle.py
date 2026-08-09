"""Export or import portable, conflict-checked raw benchmark result bundles.

Examples
--------
python scripts/result_bundle.py export --label ollama-pc
python scripts/result_bundle.py import --dest results/grid_all_systems gpu.zip h100.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_grid_results import is_derived_or_log, merge_results  # noqa: E402

SCHEMA_VERSION = 1
PAYLOAD_DIR = "payload"
MANIFEST_NAME = "bundle_manifest.json"


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "results"


def _read_stable(path: Path) -> bytes:
    """Read one file only if it did not change during the read."""
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(data) != after.st_size
    ):
        raise ValueError(
            f"{path} changed while the bundle was being created; "
            "wait for the current write to finish and export again"
        )
    return data


def export_bundle(source: Path, output: Path | None, label: str) -> Path:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")

    files = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and not is_derived_or_log(path.relative_to(source))
    ]
    if not files:
        raise ValueError(f"no raw result artifacts found under {source}")

    now = datetime.now(UTC)
    safe_label = _safe_label(label)
    if output is None:
        output = (
            PROJECT_ROOT / "result-bundles" / f"{safe_label}-{now.strftime('%Y%m%dT%H%M%SZ')}.zip"
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"bundle already exists: {output}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        entries = []
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in files:
                relative = path.relative_to(source).as_posix()
                data = _read_stable(path)
                entries.append(
                    {
                        "path": relative,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                archive.writestr(f"{PAYLOAD_DIR}/{relative}", data)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": now.isoformat(),
                "label": safe_label,
                "hostname": platform.node(),
                "source_name": source.name,
                "file_count": len(entries),
                "files": entries,
            }
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _validate_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")


def import_bundle(
    bundle: Path, dest: Path, *, keep_dest_conflicting_units: bool = False
) -> dict[str, int]:
    bundle = bundle.resolve()
    if not bundle.is_file():
        raise ValueError(f"bundle does not exist: {bundle}")

    with tempfile.TemporaryDirectory(prefix="tabarena-result-bundle-") as temp:
        extracted = Path(temp)
        with zipfile.ZipFile(bundle) as archive:
            for member in archive.namelist():
                _validate_member(member)
            try:
                manifest = json.loads(archive.read(MANIFEST_NAME))
            except KeyError as error:
                raise ValueError(f"{bundle} has no {MANIFEST_NAME}") from error
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"{bundle} uses unsupported schema {manifest.get('schema_version')}"
                )
            archive.extractall(extracted)

        payload = extracted / PAYLOAD_DIR
        entries = manifest.get("files", [])
        if len(entries) != manifest.get("file_count"):
            raise ValueError(f"{bundle} manifest file count is inconsistent")
        for entry in entries:
            _validate_member(entry["path"])
            path = payload / Path(entry["path"])
            if not path.is_file():
                raise ValueError(f"{bundle} is missing {entry['path']}")
            if path.stat().st_size != entry["size"] or _digest(path) != entry["sha256"]:
                raise ValueError(f"{bundle} failed integrity validation at {entry['path']}")
        return merge_results(
            payload,
            dest,
            keep_dest_conflicting_units=keep_dest_conflicting_units,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="create a portable raw-result bundle")
    export_parser.add_argument("--source", type=Path, default=Path("results/grid_all_systems"))
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument("--label", default=platform.node() or "results")

    import_parser = subparsers.add_parser(
        "import", help="merge one or more bundles into a local result directory"
    )
    import_parser.add_argument("--dest", type=Path, default=Path("results/grid_all_systems"))
    import_parser.add_argument("bundles", type=Path, nargs="+")
    import_parser.add_argument(
        "--keep-dest-conflicting-units",
        action="store_true",
        help="keep destination truth and skip source artifacts for incompatible dataset/fold units",
    )
    args = parser.parse_args()

    try:
        if args.command == "export":
            path = export_bundle(args.source, args.output, args.label)
            print(path)
            return

        totals = {
            "copied": 0,
            "identical": 0,
            "ignored_derived": 0,
            "skipped_conflicting_unit": 0,
        }
        for bundle in args.bundles:
            counts = import_bundle(
                bundle,
                args.dest,
                keep_dest_conflicting_units=args.keep_dest_conflicting_units,
            )
            print(f"{bundle}: " + " ".join(f"{key}={value}" for key, value in counts.items()))
            for key, value in counts.items():
                totals[key] += value
        print("total: " + " ".join(f"{key}={value}" for key, value in totals.items()))
    except (ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
