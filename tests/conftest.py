"""Shared test fixtures for TabBench-LLM."""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_clf_df():
    """Tiny classification DataFrame with 30 samples, 20 features, 3 classes."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 20)).astype(np.float32)
    y = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    df["target"] = y
    return df


@pytest.fixture
def small_reg_df():
    """Tiny regression DataFrame with 40 samples and 20 features."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 20)).astype(np.float32)
    y = rng.standard_normal(40).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    df["target"] = y
    return df


@pytest.fixture
def debug_config(tmp_path):
    """Complete config dict for fast unit tests (already in post-load form, no file I/O)."""
    return {
        "dataset_names_classification": ["OpenML-1138"],
        "dataset_names_regression": [],
        "test_size": 0.2,
        "random_state": 42,
        "n_repetitions": 1,
        "cv_folds": None,
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "results"),
        "models": ["RF"],
        "autogluon_time_limit": 30,
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "num_hpo_trials": 0,
        "min_samples_per_class": 3,
        "group_regression_splits": False,
        "max_features_default": 30000,
        "max_classes": None,
        "train_subsample": None,
        "model_limits": {},
        "nan_policy": None,
        "exclude_keys": [],
        "exclude_datasets": [],
        "exclude_targets": [],
    }


@pytest.fixture
def temp_cache():
    """Create a temporary cache directory and clean it up after the test."""
    temp_dir = tempfile.mkdtemp(prefix="tabbench_llm_test_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
