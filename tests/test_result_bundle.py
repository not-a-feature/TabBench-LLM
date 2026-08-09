"""Tests for portable raw-result bundles."""

from __future__ import annotations

import json
import zipfile

import pytest

from scripts.result_bundle import MANIFEST_NAME, export_bundle, import_bundle


def test_bundle_round_trip_excludes_regenerable_files(tmp_path):
    source = tmp_path / "source"
    raw = source / "reason_off" / "labels_hidden" / "feat_full" / "n_100"
    prediction = raw / "seed_0" / "predictions" / "task_MODEL_predictions.csv"
    statistic = raw / "seed_0" / "stats" / "task_MODEL.json"
    metric = raw / "metrics" / "classification_metrics.csv"
    log = raw / "seed_0" / "logs" / "task_MODEL.log"
    manifest = source / "run_manifests" / "run.json"
    for path, content in (
        (raw / "config.json", "{}"),
        (prediction, "target\n0\n"),
        (statistic, '{"status":"pass"}'),
        (metric, "model,score\nMODEL,1\n"),
        (log, "regenerable diagnostic log"),
        (manifest, '{"schema_version":1}'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (source / "feature_grid_metrics.csv").write_text("derived", encoding="utf-8")

    bundle = export_bundle(source, tmp_path / "bundle.zip", "GPU PC")
    with zipfile.ZipFile(bundle) as archive:
        bundle_manifest = json.loads(archive.read(MANIFEST_NAME))
        names = set(archive.namelist())

    assert bundle_manifest["label"] == "GPU-PC"
    assert "payload/run_manifests/run.json" in names
    assert not any("/metrics/" in name or "/logs/" in name for name in names)
    assert "payload/feature_grid_metrics.csv" not in names

    destination = tmp_path / "destination"
    counts = import_bundle(bundle, destination)
    assert counts["copied"] == bundle_manifest["file_count"]
    assert (destination / prediction.relative_to(source)).is_file()
    assert (destination / statistic.relative_to(source)).is_file()
    assert not (destination / metric.relative_to(source)).exists()
    assert not (destination / log.relative_to(source)).exists()

    repeated = import_bundle(bundle, destination)
    assert repeated["copied"] == 0
    assert repeated["identical"] == bundle_manifest["file_count"]


def test_bundle_import_refuses_conflicting_raw_file(tmp_path):
    source = tmp_path / "source"
    config = source / "cell" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"version":1}', encoding="utf-8")
    bundle = export_bundle(source, tmp_path / "bundle.zip", "source")

    destination = tmp_path / "destination"
    conflicting = destination / "cell" / "config.json"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text('{"version":2}', encoding="utf-8")

    with pytest.raises(ValueError, match="conflict"):
        import_bundle(bundle, destination)
