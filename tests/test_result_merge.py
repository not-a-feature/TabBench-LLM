from __future__ import annotations

import json

import pytest

from scripts.merge_grid_results import merge_results


def test_merge_accepts_only_platform_separator_difference_in_config(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.json").write_text(
        json.dumps({"output_dir": "results/grid/cell", "models": ["A"]}), encoding="utf-8"
    )
    (dest / "config.json").write_text(
        json.dumps({"output_dir": "results\\grid\\cell", "models": ["A"]}), encoding="utf-8"
    )

    assert merge_results(source, dest) == {
        "copied": 0,
        "identical": 1,
        "ignored_derived": 0,
        "skipped_conflicting_unit": 0,
    }


def test_merge_accepts_source_model_roster_subset(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.json").write_text(
        json.dumps({"output_dir": "cell", "models": ["A"]}), encoding="utf-8"
    )
    (dest / "config.json").write_text(
        json.dumps({"output_dir": "cell", "models": ["A", "B"]}), encoding="utf-8"
    )

    assert merge_results(source, dest)["identical"] == 1


def test_merge_rejects_source_model_missing_from_destination(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.json").write_text(
        json.dumps({"output_dir": "cell", "models": ["A", "B"]}), encoding="utf-8"
    )
    (dest / "config.json").write_text(
        json.dumps({"output_dir": "cell", "models": ["A"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="conflict at config.json"):
        merge_results(source, dest)


def test_merge_still_rejects_scientifically_different_config(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "config.json").write_text(
        json.dumps({"output_dir": "results/grid/cell", "models": ["A"]}), encoding="utf-8"
    )
    (dest / "config.json").write_text(
        json.dumps({"output_dir": "results\\grid\\cell", "models": ["B"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="conflict at config.json"):
        merge_results(source, dest)


def test_merge_accepts_csv_line_ending_difference_only(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "truth.csv").write_bytes(b",target\n0,1\n1,0\n")
    (dest / "truth.csv").write_bytes(b",target\r\n0,1\r\n1,0\r\n")

    assert merge_results(source, dest) == {
        "copied": 0,
        "identical": 1,
        "ignored_derived": 0,
        "skipped_conflicting_unit": 0,
    }


def test_merge_rejects_csv_value_difference(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "truth.csv").write_bytes(b",target\n0,1\n")
    (dest / "truth.csv").write_bytes(b",target\r\n0,0\r\n")

    with pytest.raises(ValueError, match="conflict at truth.csv"):
        merge_results(source, dest)


def test_merge_accepts_only_timestamp_difference_in_stats(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    relative = "cell/seed_0/stats/dataset_MODEL.json"
    (source / relative).parent.mkdir(parents=True)
    (dest / relative).parent.mkdir(parents=True)
    payload = {"dataset": "dataset", "model": "MODEL", "status": "skip"}
    (source / relative).write_text(
        json.dumps({**payload, "timestamp": "2026-08-11T00:00:00"}), encoding="utf-8"
    )
    (dest / relative).write_text(
        json.dumps({**payload, "timestamp": "2026-08-09T00:00:00"}), encoding="utf-8"
    )

    assert merge_results(source, dest)["identical"] == 1


def test_merge_rejects_scientifically_different_stats(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    relative = "cell/seed_0/stats/dataset_MODEL.json"
    (source / relative).parent.mkdir(parents=True)
    (dest / relative).parent.mkdir(parents=True)
    (source / relative).write_text(
        json.dumps({"status": "skip", "reason": "context_window", "timestamp": "new"}),
        encoding="utf-8",
    )
    (dest / relative).write_text(
        json.dumps({"status": "skip", "reason": "api_error", "timestamp": "old"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflict at"):
        merge_results(source, dest)


def test_keep_dest_conflicting_unit_skips_all_source_model_artifacts(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source_predictions = source / "cell" / "seed_0" / "predictions"
    source_stats = source / "cell" / "seed_0" / "stats"
    dest_predictions = dest / "cell" / "seed_0" / "predictions"
    source_predictions.mkdir(parents=True)
    source_stats.mkdir(parents=True)
    dest_predictions.mkdir(parents=True)
    (source_predictions / "MIC_0_ground_truth.csv").write_text(",target\n0,A\n")
    (source_predictions / "MIC_0_MODEL_predictions.csv").write_text(",prediction\n0,A\n")
    (source_stats / "MIC_0_MODEL.json").write_text("{}")
    (source_predictions / "other_0_ground_truth.csv").write_text(",target\n0,A\n")
    (dest_predictions / "MIC_0_ground_truth.csv").write_text(",target\n0,B\n")

    counts = merge_results(source, dest, keep_dest_conflicting_units=True)

    assert counts["skipped_conflicting_unit"] == 3
    assert counts["copied"] == 1
    assert not (dest_predictions / "MIC_0_MODEL_predictions.csv").exists()
    assert (dest_predictions / "other_0_ground_truth.csv").exists()
