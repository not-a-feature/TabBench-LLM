"""Regression tests for score-definition consistency across the published aggregates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabbench_llm.coverage import (
    DESIGN_SKIPS,
    assert_complete,
    coverage_counts,
    impute_failures,
    load_status,
)
from tabbench_llm.leaderboard import Leaderboard
from tabbench_llm.metrics import PRIMARY_CLF_METRIC
from tabbench_llm.site import _normalized_scores


def test_site_normalization_matches_leaderboard_minmax_definition():
    scores = pd.DataFrame(
        {
            "dataset_0": [0.2, 0.5, 0.8, np.nan],
            "tied_0": [1.0, 1.0, 1.0, np.nan],
        },
        index=["worst", "middle", "best", "not_run"],
    )

    normalized = _normalized_scores(scores)

    assert normalized["dataset_0"].iloc[:3].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert normalized["tied_0"].iloc[:3].tolist() == [1.0, 1.0, 1.0]
    assert normalized.loc["not_run"].isna().all()


def test_ranking_metric_is_the_metric_the_baselines_are_fit_on():
    # The leaderboard reports what the trained baselines were optimised for; letting the two
    # drift apart ranks models on a metric none of them selected against.
    autogluon = pytest.importorskip("tabbench_llm.model")
    from tabbench_llm.dataset import TaskType
    from tabbench_llm.metrics import PRIMARY_REG_METRIC

    clf = autogluon.AutoGluonModel(models=["RF"], task_type=TaskType.Classification, optimize=False)
    reg = autogluon.AutoGluonModel(models=["RF"], task_type=TaskType.Regression, optimize=False)
    assert clf.metric == PRIMARY_CLF_METRIC
    assert reg.metric == PRIMARY_REG_METRIC


# ---------------------------------------------------------------------------
# Absent (model, target) pairings: design exclusion vs. attempted-and-failed
# ---------------------------------------------------------------------------


def _results_dir(tmp_path, records, metric_rows):
    """A minimal results dir: one stats record per unit plus the metrics CSV."""
    stats_dir = tmp_path / "seed_0" / "stats"
    stats_dir.mkdir(parents=True)
    for key, model, status, reason in records:
        (stats_dir / f"{key}_{model}.json").write_text(
            json.dumps({"dataset": key, "model": model, "status": status, "reason": reason})
        )
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(metric_rows, columns=["seed", "key", "model", PRIMARY_CLF_METRIC]).to_csv(
        metrics_dir / "classification_metrics.csv", index=False
    )
    return str(tmp_path)


#: Two targets. LIMITED is excluded by design on the second (its prompt does not fit the
#: model's context window), BROKEN was asked and came back with nothing usable.
_RECORDS = [
    ("ds1_0", "DUMMY", "pass", ""),
    ("ds1_0", "STRONG", "pass", ""),
    ("ds1_0", "LIMITED", "pass", ""),
    ("ds1_0", "BROKEN", "pass", ""),
    ("ds2_0", "DUMMY", "pass", ""),
    ("ds2_0", "STRONG", "pass", ""),
    ("ds2_0", "LIMITED", "skip", "context_window"),
    ("ds2_0", "BROKEN", "fail", "unparseable_responses"),
]
_METRICS = [
    (0, "ds1_0", "DUMMY", 0.20),
    (0, "ds1_0", "STRONG", 0.90),
    (0, "ds1_0", "LIMITED", 0.80),
    (0, "ds1_0", "BROKEN", 0.85),
    (0, "ds2_0", "DUMMY", 0.20),
    (0, "ds2_0", "STRONG", 0.70),
]


def test_design_skip_is_excluded_from_the_score_not_zeroed(tmp_path):
    # A model kept off an input by a declared size limit must not be scored as though it had
    # lost there: that ranks the harness, not the model.
    lb = Leaderboard.from_results_dir(_results_dir(tmp_path, _RECORDS, _METRICS)).rank(
        "classification"
    )
    row = lb.set_index("model_id").loc["LIMITED"]
    assert row["# Targets"] == 1
    assert row["Score"] == pytest.approx((0.80 - 0.20) / (0.90 - 0.20), abs=1e-4)


def test_failed_run_is_scored_at_the_chance_baseline(tmp_path):
    # A run that was attempted and produced nothing usable is a real outcome. Imputing it at
    # the chance baseline stops a model from improving its standing by breaking where it
    # struggles.
    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    lb = Leaderboard.from_results_dir(results_dir).rank("classification")
    row = lb.set_index("model_id").loc["BROKEN"]
    assert row["# Targets"] == 2
    # ds1: (0.85-0.20)/(0.90-0.20); ds2: imputed to DUMMY, i.e. the per-target minimum -> 0.
    assert row["Score"] == pytest.approx(((0.85 - 0.20) / (0.90 - 0.20)) / 2, abs=1e-4)

    counts = coverage_counts(load_status(results_dir)).set_index("model_id")
    assert counts.loc["BROKEN", "# Failed"] == 1
    assert counts.loc["BROKEN", "# Skipped"] == 0
    assert counts.loc["LIMITED", "# Skipped"] == 1
    assert counts.loc["LIMITED", "# Failed"] == 0


def test_failure_imputation_makes_elo_and_score_agree_on_direction(tmp_path):
    # Both aggregates must see the failure. Omitting it would let BROKEN out-rank STRONG on
    # Elo (one unbeaten target) while scoring below it on the mean.
    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    status = load_status(results_dir)
    clf = impute_failures(pd.read_csv(f"{results_dir}/metrics/classification_metrics.csv"), status)
    played = clf[clf["model"] == "BROKEN"]
    assert set(played["key"]) == {"ds1_0", "ds2_0"}
    assert played.set_index("key").loc["ds2_0", PRIMARY_CLF_METRIC] == pytest.approx(0.20)
    # The design exclusion is still absent, so that pairing is simply not played.
    assert set(clf[clf["model"] == "LIMITED"]["key"]) == {"ds1_0"}


def test_unrecorded_unit_blocks_publication(tmp_path):
    # A dataset that silently produced nothing would otherwise renormalise every aggregate
    # over the survivors and report a dataset count nobody chose.
    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    status = load_status(results_dir)
    clf = pd.read_csv(f"{results_dir}/metrics/classification_metrics.csv")
    models = ["DUMMY", "STRONG", "LIMITED", "BROKEN"]

    assert_complete(clf, status, keys=["ds1_0", "ds2_0"], models=models, seeds=[0])
    with pytest.raises(AssertionError, match="incomplete"):
        assert_complete(clf, status, keys=["ds1_0", "ds2_0", "ds3_0"], models=models, seeds=[0])


def test_infrastructure_retry_blocks_publication():
    metrics = pd.DataFrame([{"seed": 0, "key": "ds1_0", "model": "RF", PRIMARY_CLF_METRIC: 0.8}])
    status = pd.DataFrame(
        [
            {
                "seed": 0,
                "key": "ds1_0",
                "model": "LLM",
                "status": "retry",
                "reason": "infrastructure_error",
            }
        ]
    )

    with pytest.raises(AssertionError, match="incomplete"):
        assert_complete(metrics, status, keys=["ds1_0"], models=["RF", "LLM"], seeds=[0])


def test_llm_specific_outcomes_are_on_the_declared_side_of_the_taxonomy():
    # A context window is a property of the model the harness checks before asking, so the
    # unit is excluded; an unusable answer or a blown wall-clock budget are what came back
    # from asking, so they are failures scored at chance.
    assert "context_window" in DESIGN_SKIPS
    assert "unparseable_responses" not in DESIGN_SKIPS
    assert "cell_timeout" not in DESIGN_SKIPS


def test_chance_baseline_is_in_the_roster(tmp_path):
    # impute_failures reads BASELINE_MODEL's score on the failed unit; a roster without it
    # cannot score any failure at all.
    with open("configs/models/all.json") as f:
        roster = json.load(f)
    assert "DUMMY" in {entry["key"] for entry in roster}


def test_a_wholly_absent_dataset_is_caught_from_the_config(tmp_path):
    # The failure this gate exists for: a configured dataset that produced neither a metric
    # row nor a status record is absent from everything on disk, so the expected key set has
    # to come from the config rather than from what happened to land.
    from tabbench_llm.site import _expected_keys

    keys = _expected_keys(
        {
            "dataset_names_classification": ["ds1", "ds2", "vanished"],
            "dataset_names_regression": [],
        }
    )
    assert keys == ["ds1_0", "ds2_0", "vanished_0"]

    results_dir = _results_dir(tmp_path, _RECORDS, _METRICS)
    clf = pd.read_csv(f"{results_dir}/metrics/classification_metrics.csv")
    with pytest.raises(AssertionError, match="vanished_0"):
        assert_complete(
            clf,
            load_status(results_dir),
            keys=keys,
            models=["DUMMY", "STRONG", "LIMITED", "BROKEN"],
            seeds=[0],
        )
