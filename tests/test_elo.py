"""Tests for the Bradley-Terry Elo leaderboard.

The properties pinned here are the ones a silent regression would quietly corrupt: that a
strict dominance order comes back in the right order, that the anchor lands exactly on its
target, that unplayed (model, target) pairings are never imputed into a rating, and that a
rating difference means what the paper says it means (a 400-point gap is 10:1 odds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabbench_llm.elo import (
    _table_to_battles,
    compute_elo,
    compute_elo_online,
    score_table,
    win_counts,
)
from tabbench_llm.metrics import PRIMARY_CLF_METRIC

# Keep the bootstrap small: these tests assert on the point rating and its ordering, not
# on interval width, and TabArena's default of 100 rounds makes the suite crawl.
N_BOOT = 20


def _clf(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(model, key, primary-metric) triples -> the frame score_table expects.

    Named from PRIMARY_CLF_METRIC rather than hard-coded, so re-ranking the benchmark on a
    different recorded metric does not silently turn these tests into no-ops.
    """
    return pd.DataFrame(rows, columns=["model", "key", PRIMARY_CLF_METRIC])


def _dominance_table(n_models: int = 4, n_targets: int = 6) -> pd.DataFrame:
    """Model i beats model j on every target iff i < j, by construction."""
    rows = []
    for t in range(n_targets):
        for i in range(n_models):
            rows.append((f"M{i}", f"task{t}", 1.0 - 0.1 * i))
    return score_table(_clf(rows), None)


class TestScoreTable:
    def test_regression_metric_is_negated_so_higher_is_better(self):
        reg = pd.DataFrame([("A", "t1", 1.0), ("B", "t1", 5.0)], columns=["model", "key", "rmse"])
        table = score_table(None, reg)
        # A has the lower RMSE, so after negation it must hold the larger score.
        assert table.loc["A", "t1"] > table.loc["B", "t1"]

    def test_absent_pairings_stay_nan(self):
        table = score_table(_clf([("A", "t1", 0.9), ("B", "t2", 0.8)]), None)
        assert np.isnan(table.loc["A", "t2"])
        assert np.isnan(table.loc["B", "t1"])

    def test_empty_input_returns_empty(self):
        assert score_table(None, None).empty


class TestComputeElo:
    def test_strict_dominance_recovers_the_true_order(self):
        elo = compute_elo(_dominance_table(), n_boot=N_BOOT)
        assert list(elo["model_id"]) == ["M0", "M1", "M2", "M3"]

    def test_anchor_sits_exactly_on_its_target(self):
        elo = compute_elo(_dominance_table(), anchor="M2", anchor_value=1000, n_boot=N_BOOT)
        assert elo.loc[elo.model_id == "M2", "Elo"].iloc[0] == 1000

        shifted = compute_elo(_dominance_table(), anchor="M2", anchor_value=1500, n_boot=N_BOOT)
        assert shifted.loc[shifted.model_id == "M2", "Elo"].iloc[0] == 1500

    def test_anchor_shift_preserves_rating_differences(self):
        a = compute_elo(_dominance_table(), anchor="M0", anchor_value=1000, n_boot=N_BOOT)
        b = compute_elo(_dominance_table(), anchor="M0", anchor_value=1700, n_boot=N_BOOT)
        merged = a.merge(b, on="model_id", suffixes=("_a", "_b"))
        # Calibration is affine, so every rating must move by the same constant.
        deltas = (merged["Elo_b"] - merged["Elo_a"]).unique()
        assert len(deltas) == 1

    def test_missing_anchor_does_not_raise(self):
        elo = compute_elo(_dominance_table(), anchor="NOPE", n_boot=N_BOOT)
        assert len(elo) == 4

    def test_equal_models_get_equal_ratings(self):
        rows = [(m, f"task{t}", 0.5) for m in ("A", "B", "C") for t in range(4)]
        elo = compute_elo(score_table(_clf(rows), None), anchor="A", n_boot=N_BOOT)
        assert elo["Elo"].nunique() == 1

    def test_n_targets_counts_only_played_pairings(self):
        rows = [("A", f"t{i}", 0.9) for i in range(5)]
        rows += [("B", f"t{i}", 0.8) for i in range(5)]
        rows += [("C", "t0", 0.7)]  # C ran on a single target
        elo = compute_elo(score_table(_clf(rows), None), anchor="A", n_boot=N_BOOT)
        n = dict(zip(elo["model_id"], elo["n_targets"]))
        assert n == {"A": 5, "B": 5, "C": 1}

    def test_rating_difference_is_a_calibrated_log_odds(self):
        """A 400-point gap must imply a 10:1 (90.9%) expected win rate.

        This is the claim the paper makes about the scale; it holds for the BT fit and is
        what the online variant cannot support at an arbitrary K.

        Uses 100 targets rather than 10 for a reason: TabArena's bootstrap falls back to
        an iterative Elo (K=1, a wholly different scale) for any draw in which no pair has
        a win on one side, and with a 9:1 record over 10 targets ~35% of draws miss the
        lone loss and trip that fallback, dragging the median far off the MLE. At 90:10
        the probability is 0.9**100 ~ 3e-5. See test_bootstrap_fallback_is_documented.
        """
        # 90 wins to 10 losses = 9:1 odds => a gap of 400*log10(9) points.
        rows = []
        for t in range(100):
            a, b = (0.9, 0.1) if t < 90 else (0.1, 0.9)
            rows += [("A", f"t{t}", a), ("B", f"t{t}", b)]
        elo = compute_elo(
            score_table(_clf(rows), None), anchor="B", anchor_value=1000, n_boot=N_BOOT
        )
        gap = elo.loc[elo.model_id == "A", "Elo"].iloc[0] - 1000
        expected = 400 * np.log10(9)  # ~382
        assert gap == pytest.approx(expected, abs=25)

    def test_bootstrap_fallback_is_documented(self):
        """Pin the upstream quirk that the calibration test has to route around.

        When a bootstrap draw leaves one side of every pair without a win, TabArena's
        helper abandons the MLE for that draw and returns an iterative Elo at K=1. The
        reported median then mixes two scales. This is upstream behaviour we inherit
        deliberately (we run their estimator verbatim); it only bites when the target pool
        is tiny and one model is near-perfectly dominant, so it is asserted here rather
        than patched, and it is why leaderboard cells over few targets need reading with
        their intervals rather than their point rating.
        """
        rows = []
        for t in range(10):
            a, b = (0.9, 0.1) if t < 9 else (0.1, 0.9)
            rows += [("A", f"t{t}", a), ("B", f"t{t}", b)]
        table = score_table(_clf(rows), None)

        battles, helper = _table_to_battles(table)
        point = helper.compute_mle_elo(battles, calibration_framework="B", calibration_elo=1000)
        # The point fit itself is exact...
        assert (point["A"] - point["B"]) == pytest.approx(400 * np.log10(9), abs=1)

        # ...but the bootstrap median is pulled below it by the fallback draws.
        elo = compute_elo(table, anchor="B", anchor_value=1000, n_boot=100)
        gap = elo.loc[elo.model_id == "A", "Elo"].iloc[0] - 1000
        assert gap < 400 * np.log10(9)

    @pytest.mark.parametrize("table", [pd.DataFrame(), None])
    def test_degenerate_input_returns_empty_frame(self, table):
        out = compute_elo(table, n_boot=N_BOOT)
        assert out.empty
        assert list(out.columns) == ["model_id", "Elo", "Elo_lo", "Elo_hi", "n_targets"]

    def test_single_model_returns_empty(self):
        out = compute_elo(score_table(_clf([("A", "t1", 0.9)]), None), n_boot=N_BOOT)
        assert out.empty

    def test_confidence_interval_brackets_the_rating(self):
        elo = compute_elo(_dominance_table(n_targets=8), n_boot=N_BOOT)
        assert (elo["Elo_lo"] <= elo["Elo"]).all()
        assert (elo["Elo"] <= elo["Elo_hi"]).all()


class TestOnlineEloRemainsAvailable:
    """The online variant is retained only for the estimator comparison in the appendix."""

    def test_agrees_with_bt_on_ordering(self):
        table = _dominance_table()
        bt = compute_elo(table, n_boot=N_BOOT)
        on = compute_elo_online(table, n_boot=N_BOOT)
        assert list(bt["model_id"]) == list(on["model_id"])

    def test_rating_spread_depends_on_k(self):
        """The scale artefact that motivated moving off it: spread is a function of K."""
        table = _dominance_table()
        spread = {}
        for k in (4.0, 64.0):
            e = compute_elo_online(table, k=k, n_boot=5)
            spread[k] = e["Elo"].max() - e["Elo"].min()
        assert spread[64.0] > 2 * spread[4.0]


class TestWinCounts:
    def test_counts_wins_and_splits_ties(self):
        rows = [("A", "t1", 0.9), ("B", "t1", 0.1), ("A", "t2", 0.5), ("B", "t2", 0.5)]
        models, m = win_counts(score_table(_clf(rows), None))
        i, j = models.index("A"), models.index("B")
        assert m[i][j] == 1.5  # one outright win + half a tie
        assert m[j][i] == 0.5  # half a tie

    def test_ignores_targets_only_one_model_ran(self):
        rows = [("A", "t1", 0.9), ("B", "t1", 0.1), ("A", "t2", 0.9)]
        models, m = win_counts(score_table(_clf(rows), None))
        i, j = models.index("A"), models.index("B")
        assert m[i][j] == 1.0  # t2 contributes nothing, having no opponent
