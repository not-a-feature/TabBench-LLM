"""Tests for the observed-results-only current-state preview."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import preview_current_grid
from scripts.preview_current_grid import (
    _choose_reference,
    _current_status,
    _read_cells,
    _write_inference_time,
)


def _write_metrics(output: Path, labels: str, n_train: int, dataset: str) -> Path:
    cell = output / "reason_off" / f"labels_{labels}" / "feat_full" / f"n_{n_train}"
    metrics = cell / "metrics"
    metrics.mkdir(parents=True)
    (cell / "config.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "key": f"{dataset}_fold_0",
                "seed": 0,
                "model": "OLLAMA-QWEN3-8B",
                "f1_macro": 0.75,
                "balanced_accuracy": 0.75,
            }
        ]
    ).to_csv(metrics / "classification_metrics.csv", index=False)
    return cell


def test_partial_preview_prefers_the_actual_headline(tmp_path):
    visible = _write_metrics(tmp_path, "visible", 100, "credit-g")
    hidden_50 = _write_metrics(tmp_path, "hidden", 50, "synthetic_xor")
    headline = _write_metrics(tmp_path, "hidden", 100, "synthetic_linear")

    grid, candidates = _read_cells(tmp_path)

    assert set(grid["labels"]) == {"visible", "hidden"}
    assert set(grid["n_train"]) == {50, 100}
    assert _choose_reference(candidates) == headline
    assert visible != hidden_50

    stats = headline / "seed_0" / "stats"
    stats.mkdir(parents=True)
    (stats / "task_MODEL.json").write_text(
        '{"dataset":"synthetic_linear_fold_0","model":"OLLAMA-QWEN3-8B",'
        '"status":"pass","reason":""}',
        encoding="utf-8",
    )
    counts, models = _current_status(tmp_path)
    assert counts == {"pass": 1}
    assert models == {"OLLAMA-QWEN3-8B"}


def test_recompute_uses_the_selected_result_root(tmp_path, monkeypatch):
    cell = tmp_path / "reason_off" / "labels_hidden" / "feat_full" / "n_100"
    predictions = cell / "seed_0" / "predictions"
    predictions.mkdir(parents=True)
    (cell / "config.json").write_text("{}", encoding="utf-8")
    (predictions / "dataset_MODEL_predictions.csv").write_text("prediction\nA\n")
    observed = []
    monkeypatch.setattr(
        preview_current_grid,
        "load_config",
        lambda _: {"output_dir": "results/from-another-machine"},
    )
    monkeypatch.setattr(
        preview_current_grid,
        "compute_metrics_from_predictions",
        lambda config: observed.append(config["output_dir"]),
    )

    assert preview_current_grid._recompute_available_metrics(tmp_path) == 1
    assert observed == [str(cell)]


def test_inference_time_payload_has_records_and_weighted_summaries(tmp_path):
    output = tmp_path / "results"
    site = tmp_path / "site"
    stats = output / "reason_off" / "labels_hidden" / "feat_full" / "n_10" / "seed_0" / "stats"
    stats.mkdir(parents=True)
    records = [
        {
            "dataset": "synthetic_linear_0",
            "model": "OLLAMA-QWEN3-8B",
            "status": "pass",
            "n_test_samples": 20,
            "inference_time_s": 4.0,
            "inference_time_per_sample_ms": 200.0,
            "llm_api_calls": 20,
            "timestamp": "2026-08-25T10:00:00",
        },
        {
            "dataset": "synthetic_xor_0",
            "model": "OLLAMA-QWEN3-8B",
            "status": "pass",
            "n_test_samples": 80,
            "inference_time_s": 8.0,
            "inference_time_per_sample_ms": 100.0,
            "llm_api_calls": 80,
            "timestamp": "2026-08-25T10:05:00",
        },
    ]
    for index, record in enumerate(records):
        (stats / f"task_{index}.json").write_text(json.dumps(record), encoding="utf-8")

    assert _write_inference_time(output, site) == {"records": 2, "overall": 1, "by_cell": 1}
    payload = json.loads((site / "data" / "inference_time.json").read_text(encoding="utf-8"))

    assert payload["join_keys"][-2:] == ["seed", "dataset"]
    assert payload["overall"][0]["samples"] == 100
    assert payload["overall"][0]["weighted_ms_per_sample"] == 120.0
    assert payload["overall"][0]["median_run_ms_per_sample"] == 150.0
    assert payload["records"][0]["feature_cap"] == "full"
