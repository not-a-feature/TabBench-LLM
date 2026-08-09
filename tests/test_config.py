"""Tests for config loading, validation, and list resolution."""

import json

import pytest

from scripts.grid import _build_schedule, _label_datasets, _write_canonical_config
from tabbench_llm.config import REQUIRED_KEYS, load_config


def _complete_cfg(**overrides):
    """A config declaring every required key (the loader rejects partial configs)."""
    cfg = {k: None for k in REQUIRED_KEYS}
    cfg.update(
        {
            "datasets_classification": [],
            "datasets_regression": [],
            "models": ["RF"],
            "test_size": 0.2,
            "random_state": 42,
            "n_repetitions": 1,
            "min_samples_per_class": 5,
            "group_regression_splits": False,
            "max_features_default": 1000,
            "cache_dir": ".cache",
            "output_dir": "results/test",
            "autogluon_time_limit": 60,
            "autogluon_presets": "medium_quality",
            "optimize": False,
            "ensemble": False,
            "num_hpo_trials": 0,
            "exclude_keys": [],
            "exclude_datasets": [],
            "exclude_targets": [],
        }
    )
    cfg.update(overrides)
    return cfg


def _write(tmp_path, cfg):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_load_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.json"))


def test_load_config_complete(tmp_path):
    loaded = load_config(_write(tmp_path, _complete_cfg()))
    assert loaded["models"] == ["RF"]
    assert loaded["dataset_names_classification"] == []
    assert loaded["dataset_names_regression"] == []


def test_load_config_rejects_missing_key(tmp_path):
    cfg = _complete_cfg()
    del cfg["max_features_default"]
    with pytest.raises(AssertionError, match="missing required config key"):
        load_config(_write(tmp_path, cfg))


def test_load_config_rejects_unknown_key(tmp_path):
    # A typo'd key would otherwise silently default; the loader rejects it.
    cfg = _complete_cfg(group_regresion_splits=True)
    with pytest.raises(AssertionError, match="unknown config key"):
        load_config(_write(tmp_path, cfg))


def test_load_config_allows_comment_keys(tmp_path):
    loaded = load_config(_write(tmp_path, _complete_cfg(_doc="a comment")))
    assert loaded["models"] == ["RF"]


def test_load_config_datasets_null_pass_through(tmp_path):
    # null is the explicit "load every registered dataset" sentinel.
    cfg = _complete_cfg(datasets_classification=None, datasets_regression=None)
    loaded = load_config(_write(tmp_path, cfg))
    assert loaded["dataset_names_classification"] is None
    assert loaded["dataset_names_regression"] is None


def test_load_config_resolves_list_file(tmp_path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "clf.json").write_text(json.dumps(["OpenML-1138"]))
    cfg = _complete_cfg(datasets_classification="datasets/clf.json")
    loaded = load_config(_write(tmp_path, cfg))
    assert loaded["dataset_names_classification"] == ["OpenML-1138"]


def test_canonical_grid_config_refuses_changed_resume(tmp_path):
    path = tmp_path / "config.json"
    original = _complete_cfg(models=["A", "B"])
    _write_canonical_config(str(path), original)
    _write_canonical_config(str(path), original)

    changed = {**original, "llm_settings": {"test_batch_size": 2}}
    with pytest.raises(RuntimeError, match="refusing to mix incompatible results"):
        _write_canonical_config(str(path), changed)
    assert json.loads(path.read_text()) == original


def test_canonical_grid_config_allows_append_only_roster_extension(tmp_path):
    path = tmp_path / "config.json"
    original = _complete_cfg(models=["A", "B"])
    expanded = _complete_cfg(models=["A", "NEW", "B"])
    _write_canonical_config(str(path), original)

    _write_canonical_config(str(path), expanded)
    assert json.loads(path.read_text()) == expanded

    with pytest.raises(RuntimeError, match="refusing to mix incompatible results"):
        _write_canonical_config(str(path), original)


def test_canonical_grid_config_allows_additive_dataset_and_model_extension(tmp_path):
    path = tmp_path / "config.json"
    original = _complete_cfg(datasets_classification=["old_a", "old_b"], models=["A"])
    expanded = _complete_cfg(
        datasets_classification=["old_a", "old_b", "new_wide"], models=["A", "NEW"]
    )
    _write_canonical_config(str(path), original)

    _write_canonical_config(str(path), expanded)
    assert json.loads(path.read_text()) == expanded

    reordered = {**expanded, "datasets_classification": ["old_b", "old_a", "new_wide"]}
    with pytest.raises(RuntimeError, match="refusing to mix incompatible results"):
        _write_canonical_config(str(path), reordered)


def test_canonical_grid_config_ignores_output_path_separator_style(tmp_path):
    path = tmp_path / "config.json"
    original = _complete_cfg(output_dir="results/grid/cell", models=["A"])
    expanded = _complete_cfg(output_dir=r"results\grid\cell", models=["A", "NEW"])
    _write_canonical_config(str(path), original)

    _write_canonical_config(str(path), expanded)
    saved = json.loads(path.read_text())
    assert saved["models"] == ["A", "NEW"]
    assert saved["output_dir"] == "results/grid/cell"


def test_visible_labels_are_only_scheduled_for_real_datasets():
    datasets = ["credit-g", "synthetic_linear", "bank-marketing", "synthetic_xor"]

    assert _label_datasets("visible", datasets) == ["credit-g", "bank-marketing"]
    assert _label_datasets("hidden", datasets) == datasets


def test_real_grid_reduction_keeps_legacy_cell_configs_compatible():
    datasets = ["synthetic_linear", "credit-g", "synthetic_wide"]
    config = {
        "reasoning": ["off"],
        "samples": [10, 50],
        "real_grid": {
            "reasoning": ["off"],
            "labels": ["visible", "hidden"],
            "feature_caps": ["full"],
            "samples": [50],
        },
    }
    schedule = _build_schedule(
        config, datasets, {"synthetic_linear": 2, "credit-g": 3, "synthetic_wide": 64}
    )

    # The old n=10 hidden/full config listed both suites. The new run executes synthetic only
    # there, so partial Windows results can be resumed without rewriting that config.
    hidden_10 = schedule[("off", "hidden", None, 10)]
    assert hidden_10["datasets"] == ["synthetic_linear", "synthetic_wide"]
    assert hidden_10["canonical_datasets"] == datasets

    # Real data is only scheduled in the requested four-axis slice; visible labels never apply
    # to the synthetic suite.
    assert ("off", "visible", None, 10) not in schedule
    assert schedule[("off", "visible", None, 50)]["datasets"] == ["credit-g"]
    assert schedule[("off", "hidden", None, 50)]["datasets"] == datasets
