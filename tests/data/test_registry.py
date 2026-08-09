"""Unit tests for the configurable bio dataset registry (network-free)."""

from __future__ import annotations

import json

import pytest

from tabbench_llm.data import datasets as ds
from tabbench_llm.data.datasets import DatasetSpec, load_specs


def test_bundled_registry_loads_and_validates():
    specs = load_specs()  # the bundled registry/datasets.json (TabArena classification set)
    assert specs, "registry is empty"
    # dataset_id uniqueness is enforced by load_specs; spot-check a known TabArena entry.
    assert "diabetes" in specs
    assert specs["diabetes"].source == "openml"
    assert specs["diabetes"].fetch_id == "46921"


def test_curated_datasets_runnable():
    runnable = ds.runnable_specs()
    # 38 OpenML TabArena datasets (targets resolve from default_target_attribute, so a null
    # target is still curated) + one generated, contamination-free anchor per synthetic recipe.
    from tabbench_llm.data.loaders.synthetic import _RECIPES

    assert {s.source for s in runnable} == {"openml", "synthetic"}
    openml = [s for s in runnable if s.source == "openml"]
    synthetic = [s for s in runnable if s.source == "synthetic"]
    assert len(openml) == 38
    # Every recipe is registered and every registered synthetic dataset has a recipe — a recipe
    # with no registry entry would never run, and an entry with no recipe fails at fetch.
    assert {s.fetch_id for s in synthetic} == set(_RECIPES)
    assert all(s.is_curated for s in runnable)
    assert all(s.target is None for s in openml)  # resolved at fetch, not curated inline
    assert all(s.target == "target" for s in synthetic)


def test_problem_type_filtering():
    assert "diabetes" in ds.dataset_names("binary")  # 2 classes
    assert "website_phishing" in ds.dataset_names("multiclass")  # 3 classes
    assert "diabetes" not in ds.dataset_names("multiclass")


def test_non_openml_source_still_needs_explicit_target():
    # The openml exemption is source-specific: a geo/tcga/... spec with a null target
    # is NOT curated (only openml resolves a default target attribute at fetch).
    geo = DatasetSpec(dataset_id="g", source="geo", fetch_id="GSE1", problem_type="binary")
    assert not geo.is_curated
    openml = DatasetSpec(dataset_id="o", source="openml", fetch_id="1", problem_type="binary")
    assert openml.is_curated


def test_unknown_field_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps([{"dataset_id": "x", "source": "openml", "fetch_id": "1", "bogus": 1}])
    )
    with pytest.raises(ValueError, match="unknown field"):
        load_specs(bad)


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        DatasetSpec(dataset_id="x", source="nope", fetch_id="1")


def test_resolved_eval_metric_defaults_by_problem_type():
    spec = DatasetSpec(
        dataset_id="x", source="openml", fetch_id="1", target="t", problem_type="binary"
    )
    assert spec.resolved_eval_metric() == "roc_auc"
