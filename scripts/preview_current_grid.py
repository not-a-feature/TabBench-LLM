"""Build a clearly labelled, observed-results-only preview of the current result directory.

This intentionally does not apply the publication completeness gate and does not impute
failures whose DUMMY baseline may live on another machine. It is for monitoring only.
The final public leaderboard must still be built with ``scripts/finalize_grid.py`` after
all machine slices have been merged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from tabbench_llm.config import load_config
from tabbench_llm.coverage import load_status
from tabbench_llm.evaluation import compute_metrics_from_predictions
from tabbench_llm.site import build_site

AXES = ("reasoning", "labels", "feature_cap", "n_train")
REPORT_METRICS = (
    "balanced_accuracy",
    "matthews_corrcoef",
    "f1_macro",
    "f1_score",
    "roc_auc",
)


def _recompute_available_metrics(output: Path) -> int:
    """Add metric rows for every prediction currently present under *output*.

    Cell configs retain the canonical full roster. The evaluator skips models without a
    prediction file, so this also works after arbitrary machine bundles have been merged.
    """
    recomputed = 0
    for config_path in sorted(output.glob("reason_*/labels_*/feat_*/n_*/config.json")):
        cell = config_path.parent
        if not any(cell.glob("seed_*/predictions/*_predictions.csv")):
            continue
        config = load_config(str(config_path))
        config["output_dir"] = str(cell)
        compute_metrics_from_predictions(config)
        recomputed += 1
    return recomputed


def _read_cells(output: Path) -> tuple[pd.DataFrame, list[tuple[Path, str, int]]]:
    frames: list[pd.DataFrame] = []
    candidates: list[tuple[Path, str, int]] = []
    pattern = "reason_*/labels_*/feat_*/n_*/metrics/classification_metrics.csv"

    for metrics_path in sorted(output.glob(pattern)):
        cell = metrics_path.parents[1]
        relative = cell.relative_to(output)
        reason = relative.parts[0].removeprefix("reason_")
        labels = relative.parts[1].removeprefix("labels_")
        feature_cap = relative.parts[2].removeprefix("feat_")
        n_token = relative.parts[3].removeprefix("n_")
        try:
            n_train = int(n_token)
        except ValueError:
            continue

        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            continue
        metrics.insert(0, "n_train", n_train)
        metrics.insert(0, "feature_cap", feature_cap)
        metrics.insert(0, "labels", labels)
        metrics.insert(0, "reasoning", reason)
        frames.append(metrics)

        if reason == "off" and feature_cap == "full":
            candidates.append((cell, labels, n_train))

    if not frames:
        raise SystemExit(f"No completed classification metrics found under {output}")
    if not candidates:
        raise SystemExit("No completed reason_off / feat_full cell is available for the preview")
    return pd.concat(frames, ignore_index=True), candidates


def _write_grid_data(grid: pd.DataFrame, site_out: Path) -> None:
    data_out = site_out / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(data_out / "feature_grid_metrics.csv", index=False)

    present = [metric for metric in REPORT_METRICS if metric in grid.columns]
    summary = (
        grid.groupby(["dataset", "model", *AXES])[present]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["__".join(column).rstrip("_") for column in summary.columns]
    summary.to_csv(data_out / "feature_grid_summary.csv", index=False)


def _timing_summary(records: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    summaries = []
    for keys, group in records.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        total_samples = int(group["n_test_samples"].sum())
        total_seconds = float(group["inference_time_s"].sum())
        row = dict(zip(group_cols, keys, strict=True))
        if "n_train" in row:
            row["n_train"] = int(row["n_train"])
        row.update(
            {
                "runs": int(len(group)),
                "datasets": int(group["dataset"].nunique()),
                "samples": total_samples,
                "total_inference_time_s": round(total_seconds, 3),
                "weighted_ms_per_sample": round(1000 * total_seconds / total_samples, 4),
                "median_run_ms_per_sample": round(
                    float(group["inference_time_per_sample_ms"].median()), 4
                ),
            }
        )
        summaries.append(row)
    return summaries


def _write_inference_time(output: Path, site_out: Path) -> dict[str, int]:
    records = []
    pattern = "reason_*/labels_*/feat_*/n_*/seed_*/stats/*.json"
    for stats_path in sorted(output.glob(pattern)):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats["status"] != "pass" or stats["inference_time_s"] is None:
            continue
        relative = stats_path.relative_to(output)
        record = {
            "model": str(stats["model"]),
            "reasoning": relative.parts[0].removeprefix("reason_"),
            "labels": relative.parts[1].removeprefix("labels_"),
            "feature_cap": relative.parts[2].removeprefix("feat_"),
            "n_train": int(relative.parts[3].removeprefix("n_")),
            "seed": int(relative.parts[4].removeprefix("seed_")),
            "dataset": str(stats["dataset"]),
            "n_test_samples": int(stats["n_test_samples"]),
            "inference_time_s": float(stats["inference_time_s"]),
            "inference_time_per_sample_ms": float(stats["inference_time_per_sample_ms"]),
            "llm_api_calls": (
                int(stats["llm_api_calls"])
                if "llm_api_calls" in stats and stats["llm_api_calls"] is not None
                else None
            ),
            "timestamp": str(stats["timestamp"]),
        }
        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        overall = []
        by_cell = []
        source_max_timestamp = None
    else:
        overall = _timing_summary(frame, ["model", "reasoning"])
        by_cell = _timing_summary(
            frame, ["model", "reasoning", "labels", "feature_cap", "n_train"]
        )
        source_max_timestamp = max(frame["timestamp"])

    payload = {
        "schema_version": 1,
        "scope": "Passing runs with non-null inference timing.",
        "timing_semantics": (
            "End-to-end model.predict wall time, including endpoint latency, queueing, "
            "retries and generation; not pure model token throughput."
        ),
        "join_keys": [
            "model",
            "reasoning",
            "labels",
            "feature_cap",
            "n_train",
            "seed",
            "dataset",
        ],
        "units": {
            "inference_time_s": "seconds per completed run",
            "inference_time_per_sample_ms": "milliseconds per test sample",
            "weighted_ms_per_sample": "total seconds / total samples, in milliseconds",
        },
        "source_max_timestamp": source_max_timestamp,
        "overall": overall,
        "by_cell": by_cell,
        "records": records,
    }
    data_out = site_out / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    (data_out / "inference_time.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return {"records": len(records), "overall": len(overall), "by_cell": len(by_cell)}


def _current_status(output: Path) -> tuple[dict[str, int], set[str]]:
    frames = []
    for config_path in sorted(output.glob("reason_*/labels_*/feat_*/n_*/config.json")):
        status = load_status(str(config_path.parent))
        if not status.empty:
            frames.append(status)
    if not frames:
        return {}, set()
    combined = pd.concat(frames, ignore_index=True)
    counts = {str(key): int(value) for key, value in combined["status"].value_counts().items()}
    return counts, set(combined["model"].astype(str))


def _choose_reference(candidates: list[tuple[Path, str, int]]) -> Path:
    # Prefer the actual opaque-label n=100 headline. During an early run, fall back to the
    # largest completed opaque full-feature cell, then to a visible-label diagnostic cell.
    ranked = sorted(
        candidates,
        key=lambda item: (
            item[1] == "hidden" and item[2] == 100,
            item[1] == "hidden",
            item[2],
        ),
        reverse=True,
    )
    return ranked[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/grid_all_systems")
    parser.add_argument("--site-out", default="site_data")
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="add metric rows for all raw predictions currently present before previewing",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    site_out = Path(args.site_out).resolve()
    if args.recompute_metrics:
        count = _recompute_available_metrics(output)
        print(f"Recomputed available metrics in {count} cell(s).")
    grid, candidates = _read_cells(output)
    _write_grid_data(grid, site_out)
    timing_counts = _write_inference_time(output, site_out)
    reference = _choose_reference(candidates)
    status_counts, status_models = _current_status(output)

    with tempfile.TemporaryDirectory(prefix="tabbench-partial-preview-") as temp:
        preview_results = Path(temp)
        preview_metrics = preview_results / "metrics"
        preview_metrics.mkdir()
        shutil.copy2(
            reference / "metrics" / "classification_metrics.csv",
            preview_metrics / "classification_metrics.csv",
        )
        model_names = sorted(set(grid["model"].astype(str)) | status_models)
        models = len(model_names)
        datasets = grid["dataset"].nunique()
        status_text = " · ".join(
            f"{name}={status_counts.get(name, 0)}" for name in ("pass", "fail", "skip", "retry")
        )
        build_site(
            str(preview_results),
            str(site_out),
            config_path=None,
            title="TabBench-LLM — CURRENT-STATE PREVIEW",
            subtitle=f"{models} model(s) · {datasets} dataset(s) · {status_text}",
        )

    state = {
        "preview": True,
        "publication_ready": False,
        "models": model_names,
        "models_observed": models,
        "datasets_with_metrics": int(datasets),
        "metric_rows": len(grid),
        "status_counts": status_counts,
        "reference_cell": str(reference.relative_to(output)).replace("\\", "/"),
    }
    (site_out / "data" / "current_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Current-state preview built from {reference}")
    print(f"Observed rows: {len(grid)}; models: {models}; datasets: {datasets}")
    print(
        "Inference timing: "
        f"records={timing_counts['records']}; overall={timing_counts['overall']}; "
        f"by_cell={timing_counts['by_cell']}"
    )
    print("Status: " + (status_text or "no status records"))
    print("Failures are not chance-imputed in this preview; use finalize_grid.py for publication.")


if __name__ == "__main__":
    main()
