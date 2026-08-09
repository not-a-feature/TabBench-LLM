"""Recompute metrics and site data after all machine slices have been merged."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.grid import _build_schedule, _cell_dir, _raw_feature_counts
from tabbench_llm.config import load_config, resolve_list
from tabbench_llm.coverage import assert_complete, load_status
from tabbench_llm.evaluation import compute_metrics_from_predictions
from tabbench_llm.seeds import get_seeds
from tabbench_llm.site import _expected_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_all_systems.json")
    parser.add_argument("--output", help="override the grid output root")
    parser.add_argument("--site-out", default="site_data")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    grid_config = json.loads(config_path.read_text())
    output = Path(args.output or grid_config["output"]).resolve()
    datasets = resolve_list(grid_config["datasets"], str(config_path.parent))
    schedule = _build_schedule(
        grid_config, datasets, _raw_feature_counts(datasets, grid_config["cache_dir"])
    )
    if not schedule:
        raise SystemExit(f"no grid cell configs found under {output}")

    for index, ((reason, labels, cap, samples), cell) in enumerate(schedule.items(), 1):
        path = Path(_cell_dir(str(output), reason, labels, cap, samples)) / "config.json"
        if not path.is_file():
            raise SystemExit(f"scheduled grid cell is missing: {path}")
        print(f"[{index}/{len(schedule)}] metrics: {path.parent}")
        loaded = load_config(str(path))
        # The on-disk config keeps the legacy all-suite cell definition for compatibility with
        # imported partial results. Finalization validates only the currently scheduled subset.
        loaded["dataset_names_classification"] = cell["datasets"]
        compute_metrics_from_predictions(loaded)
        metrics_path = path.parent / "metrics" / "classification_metrics.csv"
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        if not metrics.empty:
            metrics = metrics[metrics["dataset"].isin(cell["datasets"])]
        assert_complete(
            metrics,
            load_status(str(path.parent)),
            keys=_expected_keys(loaded),
            models=loaded["models"],
            seeds=get_seeds(loaded),
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/grid.py",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--skip-run",
        ],
        check=True,
    )

    site_out = Path(args.site_out).resolve()
    data_out = site_out / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    for name in ("feature_grid_metrics.csv", "feature_grid_summary.csv"):
        shutil.copy2(output / name, data_out / name)

    headline = output / "reason_off" / "labels_hidden" / "feat_full" / "n_100"
    if not headline.is_dir():
        raise SystemExit(f"headline cell is missing: {headline}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tabbench_llm.cli",
            "site",
            "--results-dir",
            str(headline),
            "--out",
            str(site_out),
            "--config",
            str(headline / "config.json"),
        ],
        check=True,
    )
    print(f"Site rebuilt at {site_out}")


if __name__ == "__main__":
    main()
