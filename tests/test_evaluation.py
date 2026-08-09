"""Tests for the metric step's handling of degenerate probabilities and unparsed rows."""

from __future__ import annotations

import json
import os

import pandas as pd

from tabbench_llm.evaluation import _unparsed_frac, compute_metrics_from_predictions


def _write_cell(root, seed, key, model, y_true, y_pred, proba, stats=None):
    """Lay out one finished (seed, dataset, model) cell the way the prediction step does."""
    pred_dir = os.path.join(root, f"seed_{seed}", "predictions")
    stats_dir = os.path.join(root, f"seed_{seed}", "stats")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    idx = list(range(len(y_true)))
    pd.DataFrame({"target": y_true}, index=idx).to_csv(
        os.path.join(pred_dir, f"{key}_ground_truth.csv")
    )
    pd.DataFrame({"target": y_pred}, index=idx).to_csv(
        os.path.join(pred_dir, f"{key}_{model}_predictions.csv")
    )
    pd.DataFrame(proba, index=idx, columns=["0", "1"]).to_csv(
        os.path.join(pred_dir, f"{key}_{model}_proba.csv")
    )
    if stats is not None:
        record = {"dataset": key, "model": model, "status": "pass", "reason": "", "error": ""}
        record.update(stats)
        with open(os.path.join(stats_dir, f"{key}_{model}.json"), "w") as f:
            json.dump(record, f)


def _config(root, models):
    return {
        "datasets_classification": ["synthetic_linear"],
        "datasets_regression": [],
        "dataset_names_classification": ["synthetic_linear"],
        "dataset_names_regression": [],
        "models": models,
        "test_size": 0.2,
        "random_state": 0,
        "n_repetitions": 1,
        "cv_folds": None,
        "min_samples_per_class": 10,
        "group_regression_splits": False,
        "max_features_default": None,
        "max_classes": None,
        "train_subsample": None,
        "test_subsample": None,
        "model_limits": {},
        "llm_settings": {"test_batch_size": 1, "elicit_proba": False},
        "cache_dir": ".cache",
        "output_dir": root,
        "autogluon_time_limit": 60,
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "num_hpo_trials": 0,
        "nan_policy": {"default": "native"},
        "exclude_keys": [],
        "exclude_datasets": [],
        "exclude_targets": [],
    }


def test_one_hot_proba_suppresses_ranking_metrics(tmp_path):
    """A one-hot proba must not yield a ROC-AUC / log-loss that looks like a real one.

    One-hot "probabilities" are the predicted label written as a vector: ROC-AUC computed from
    them collapses onto a thresholded accuracy and log-loss is the clip applied to the wrong
    predictions. Reported as numbers they would sit in the same column as the genuine ranking
    metrics of models that emit a distribution.
    """
    root = str(tmp_path)
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    # ONEHOT mirrors y_pred exactly; SOFT is a real distribution.
    _write_cell(
        root,
        0,
        "synthetic_linear_0",
        "ONEHOT",
        y_true,
        y_pred,
        [[1.0 - p, float(p)] for p in y_pred],
    )
    _write_cell(
        root,
        0,
        "synthetic_linear_0",
        "SOFT",
        y_true,
        y_pred,
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1], [0.55, 0.45]],
    )

    compute_metrics_from_predictions(_config(root, ["ONEHOT", "SOFT"]))
    df = pd.read_csv(os.path.join(root, "metrics", "classification_metrics.csv")).set_index("model")

    assert df.loc["ONEHOT", "proba_degenerate"]
    assert not df.loc["SOFT", "proba_degenerate"]
    # Label metrics are identical (same predictions); only the probability metrics differ.
    assert df.loc["ONEHOT", "balanced_accuracy"] == df.loc["SOFT", "balanced_accuracy"]
    assert pd.isna(df.loc["ONEHOT", "roc_auc"]) and pd.isna(df.loc["ONEHOT", "log_loss"])
    assert not pd.isna(df.loc["SOFT", "roc_auc"]) and not pd.isna(df.loc["SOFT", "log_loss"])


def test_unparsed_fraction_reaches_the_metrics_row(tmp_path):
    # The majority-class fallback rate lives in the per-cell stats; it has to travel to the
    # metrics CSV, otherwise the caveat never reaches whoever reads the leaderboard.
    root = str(tmp_path)
    y_true, y_pred = [0, 0, 1, 1], [0, 0, 1, 1]
    proba = [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]]
    _write_cell(
        root,
        0,
        "synthetic_linear_0",
        "LLM",
        y_true,
        y_pred,
        proba,
        stats={"llm_unparsed": 1, "llm_unparsed_frac": 0.25},
    )
    _write_cell(root, 0, "synthetic_linear_0", "RF", y_true, y_pred, proba, stats={})

    assert _unparsed_frac(root, 0, "synthetic_linear_0", "LLM") == 0.25
    assert _unparsed_frac(root, 0, "synthetic_linear_0", "RF") is None
    assert _unparsed_frac(root, 0, "synthetic_linear_0", "ABSENT") is None

    compute_metrics_from_predictions(_config(root, ["LLM", "RF"]))
    df = pd.read_csv(os.path.join(root, "metrics", "classification_metrics.csv")).set_index("model")
    assert df.loc["LLM", "llm_unparsed_frac"] == 0.25
    assert pd.isna(df.loc["RF", "llm_unparsed_frac"])
