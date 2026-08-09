"""Evaluation metrics for classification and regression tasks."""

from tabbench_llm.metrics.classification import ClassificationMetrics
from tabbench_llm.metrics.regression import RegressionMetrics
from tabbench_llm.metrics.utils import compute_metrics

#: Metric the leaderboard ``Score`` and Elo rank classification models by, and the metric
#: AutoGluon fits and selects on (:mod:`tabbench_llm.model`) — one metric for both, so the
#: published ranking reports what the baselines were optimised for. Single source of truth:
#: change it here to re-rank by a different recorded metric (e.g. ``"balanced_accuracy"``,
#: ``"matthews_corrcoef"``, ``"roc_auc"``). ``f1_macro`` is the unweighted mean of per-class
#: F1: imbalance-robust like balanced accuracy (both weight every class equally regardless of
#: prior), unlike weighted F1, and unlike balanced accuracy it charges a model for false
#: positives as well as missed positives — which matters here because a model that buys recall
#: on a rare class by over-predicting it scores well on balanced accuracy alone, and an
#: in-context classifier's label prior is set by the prompt rather than fitted. The full
#: metric suite is always stored per (seed, dataset, model), so switching this only changes
#: ranking, never what is recorded.
PRIMARY_CLF_METRIC = "f1_macro"
#: Metric the leaderboard ``Score`` and Elo rank regression models by (lower is better).
PRIMARY_REG_METRIC = "rmse"

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "compute_metrics",
    "PRIMARY_CLF_METRIC",
    "PRIMARY_REG_METRIC",
]
