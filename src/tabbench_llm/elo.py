"""Pairwise Elo ratings and head-to-head win counts for the leaderboard.

Elo is derived from per-target pairwise comparisons: for every dataset target, each model
is compared against every other model that also ran on it. The comparison score is
"higher is better" for both task types — the mean primary classification metric
(:data:`~tabbench_llm.metrics.PRIMARY_CLF_METRIC`, macro-F1) for classification targets,
**negated** mean RMSE for regression — so a larger score always means a better model.
Ratings are calibrated so a chosen anchor model (Random Forest) sits at exactly 1000, and
confidence intervals come from bootstrap resampling of the target pool.

Absent ``(model, target)`` pairings are not imputed here: a run that was attempted and
failed has already been imputed at chance level upstream
(:func:`tabbench_llm.coverage.impute_failures`), so what reaches this module as missing is a
unit the benchmark excluded by design and those pairings are simply not played. That is the
same convention the leaderboard ``Score`` and the per-dataset score heatmap use, so the
three aggregates on the site are computed over one definition of the data.

The rating itself is a **Bradley-Terry (BT) maximum-likelihood fit**, computed by the
vendored :mod:`tabbench_llm._vendor.tabarena_elo_utils` — TabArena's own ``EloHelper``,
which in turn follows Chatbot Arena's ``compute_mle_elo``. Using their code verbatim
rather than a reimplementation means our leaderboard is the same estimator as TabArena's,
which is the point of a benchmark named TabBench-LLM.

Why BT rather than the online/sequential Elo update this module used previously
(preserved as :func:`compute_elo_online`): the online update
``R_i += K * (s - E)`` is exactly a stochastic-gradient step on the BT log-likelihood, so
a single sequential pass is one under-converged epoch of SGD on the objective BT solves
exactly. The two agree on *ordering* (Spearman ~0.997 on our results), but the online
variant's rating *scale* is governed by the arbitrary constant ``K`` — the best-to-worst
spread moves from ~220 at K=4 to ~840 at K=64 — and no amount of order-averaging removes
that dependence. Only under the BT fit is a rating difference a calibrated log-odds, which
is what licenses the standard reading that a 400-point gap is a 10:1 expected win rate.
:func:`compute_elo_online` is kept so the two can be compared on real results.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from tabbench_llm._vendor.tabarena_elo_utils import EloHelper
from tabbench_llm.metrics import PRIMARY_CLF_METRIC, PRIMARY_REG_METRIC

logger = logging.getLogger(__name__)

DEFAULT_K = 32.0
DEFAULT_BASE = 1000.0
DEFAULT_ANCHOR = "RF"
DEFAULT_SCALE = 400.0

#: Bootstrap rounds for the Elo confidence intervals. Matches TabArena's default.
DEFAULT_N_BOOT = 100


def score_table(
    clf_df: pd.DataFrame | None,
    reg_df: pd.DataFrame | None,
    clf_metric: str = PRIMARY_CLF_METRIC,
    reg_metric: str = PRIMARY_REG_METRIC,
) -> pd.DataFrame:
    """Build a models x targets table of comparison scores (higher = better; NaN = absent).

    Classification targets use the mean ``clf_metric`` (higher is better); regression
    targets use the **negated** mean ``reg_metric`` (so higher is better there too). The
    two task pools share one column space (targets are dataset ``key``s). Pass ``None`` /
    empty for one task to build a task-specific table (used for per-tab Elo).
    """
    frames = []
    if clf_df is not None and not clf_df.empty and clf_metric in clf_df.columns:
        g = clf_df.groupby(["model", "key"])[clf_metric].mean().reset_index()
        g = g.rename(columns={clf_metric: "score"})
        frames.append(g[["model", "key", "score"]])
    if reg_df is not None and not reg_df.empty and reg_metric in reg_df.columns:
        g = reg_df.groupby(["model", "key"])[reg_metric].mean().reset_index()
        g["score"] = -g[reg_metric]
        frames.append(g[["model", "key", "score"]])
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    return long.pivot_table(index="model", columns="key", values="score", aggfunc="mean")


def _table_to_battles(table: pd.DataFrame) -> tuple[pd.DataFrame, EloHelper]:
    """Score table → TabArena ``battles`` frame, via its own ``convert_results_to_battles``.

    TabArena's helper is written against a *metric error* (lower is better) while our
    table is oriented higher-is-better, so the score is negated back into an error here.
    Absent ``(model, target)`` pairings drop out rather than being imputed, which is what
    stops a model from being rated on targets it never ran.
    """
    helper = EloHelper(method_col="method", task_col="task", error_col="metric_error")
    long = table.stack().reset_index()  # stack() drops NaN, i.e. unplayed pairings
    long.columns = ["method", "task", "score"]
    long["metric_error"] = -long["score"].astype(float)
    return helper.convert_results_to_battles(long[["method", "task", "metric_error"]]), helper


def compute_elo(
    table: pd.DataFrame,
    anchor: str = DEFAULT_ANCHOR,
    anchor_value: float = DEFAULT_BASE,
    n_boot: int = DEFAULT_N_BOOT,
    scale: float = DEFAULT_SCALE,
    random_state: int = 0,
    show_process: bool = False,
) -> pd.DataFrame:
    """Bradley-Terry Elo per model with bootstrap 95% CIs (TabArena's estimator).

    Parameters
    ----------
    table : pd.DataFrame
        Models × targets score table from :func:`score_table` (higher = better).
    anchor, anchor_value : str, float
        Model whose rating is fixed (Random Forest → 1000). When the anchor is absent the
        ratings are returned uncalibrated (BT fixes only differences, not the origin).
    n_boot : int
        Bootstrap resamples of the *target* pool for the confidence interval. As in
        TabArena, the reported rating is the bootstrap **median**, not the point fit.
    scale : float
        Elo scale; 400 gives the conventional "400 points = 10:1 odds" reading.

    Returns
    -------
    pd.DataFrame
        Columns ``model_id``, ``Elo``, ``Elo_lo``, ``Elo_hi``, ``n_targets`` (rounded ints).
    """
    cols = ["model_id", "Elo", "Elo_lo", "Elo_hi", "n_targets"]
    if table is None or table.empty or table.shape[0] < 2:
        return pd.DataFrame(columns=cols)

    battles, helper = _table_to_battles(table)
    if battles.empty:
        return pd.DataFrame(columns=cols)

    calibration = anchor if anchor in table.index else None
    if calibration is None:
        logger.warning(
            "Elo anchor %r absent from the score table; returning uncalibrated ratings.", anchor
        )

    draws = helper.compute_elo_ratings(
        battles=battles,
        seed=random_state,
        calibration_framework=calibration,
        calibration_elo=anchor_value if calibration else None,
        INIT_RATING=anchor_value,
        BOOTSTRAP_ROUNDS=n_boot,
        SCALE=scale,
        show_process=show_process,
    )

    counts = (~table.isna()).sum(axis=1)
    models = [m for m in draws.columns]
    out = pd.DataFrame(
        {
            "model_id": models,
            "Elo": np.round(draws.quantile(0.5)[models].to_numpy()).astype(int),
            "Elo_lo": np.round(draws.quantile(0.025)[models].to_numpy()).astype(int),
            "Elo_hi": np.round(draws.quantile(0.975)[models].to_numpy()).astype(int),
            "n_targets": counts.reindex(models).fillna(0).to_numpy().astype(int),
        }
    )
    return out.sort_values("Elo", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Online (sequential) Elo — retained for the estimator comparison only
# ---------------------------------------------------------------------------


def _target_games(table: pd.DataFrame) -> tuple[list[str], list[list[tuple[int, int, float]]]]:
    """Per-target lists of ``(i, j, s_i)`` games over present model-index pairs.

    ``s_i`` is 1.0 if model ``i`` beats ``j`` on that target, 0.0 if it loses, 0.5 on a tie.
    """
    models = list(table.index)
    arr = table.to_numpy()
    n_models, n_targets = arr.shape
    per_target: list[list[tuple[int, int, float]]] = []
    for t in range(n_targets):
        col = arr[:, t]
        present = [i for i in range(n_models) if not np.isnan(col[i])]
        games: list[tuple[int, int, float]] = []
        for a, b in combinations(present, 2):
            va, vb = col[a], col[b]
            s = 0.5 if va == vb else (1.0 if va > vb else 0.0)
            games.append((a, b, s))
        per_target.append(games)
    return models, per_target


def _elo_pass(
    games: list[tuple[int, int, float]], order: list[int], ratings: list[float], k: float
):
    """One sequential Elo sweep over *games* in *order*, mutating *ratings* in place."""
    for gi in order:
        a, b, s = games[gi]
        ra = ratings[a]
        rb = ratings[b]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) * 0.0025))  # 0.0025 == 1/400
        delta = k * (s - ea)
        ratings[a] = ra + delta
        ratings[b] = rb - delta  # B's update is exactly the negation of A's


def _calibrate(
    ratings: np.ndarray, models: list[str], anchor: str, anchor_value: float
) -> np.ndarray:
    """Shift ratings so *anchor* sits at *anchor_value* (else so the mean does)."""
    if anchor in models:
        shift = anchor_value - ratings[models.index(anchor)]
    else:
        shift = anchor_value - float(np.mean(ratings))
    return ratings + shift


def compute_elo_online(
    table: pd.DataFrame,
    anchor: str = DEFAULT_ANCHOR,
    anchor_value: float = DEFAULT_BASE,
    k: float = DEFAULT_K,
    base: float = DEFAULT_BASE,
    n_orderings: int = 50,
    n_boot: int = 200,
    boot_orderings: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """Online/sequential Elo, averaged over random match orderings.

    **Not** the benchmark's reported rating — :func:`compute_elo` is. This is kept only to
    quantify the difference between the two estimators on real results (see the module
    docstring and the paper's Elo appendix): its ratings depend on ``k``, and averaging
    orderings suppresses the variance of path-dependence without removing its bias.
    """
    cols = ["model_id", "Elo", "Elo_lo", "Elo_hi", "n_targets"]
    if table is None or table.empty or table.shape[0] < 2:
        return pd.DataFrame(columns=cols)

    models, per_target = _target_games(table)
    n_models = len(models)
    n_targets = len(per_target)
    rng = np.random.default_rng(random_state)

    def run(target_indices, n_ord: int) -> np.ndarray:
        games: list[tuple[int, int, float]] = []
        for ti in target_indices:
            games.extend(per_target[ti])
        if not games:
            return np.full(n_models, base, dtype=float)
        acc = np.zeros(n_models, dtype=float)
        ng = len(games)
        for _ in range(n_ord):
            ratings = [base] * n_models
            _elo_pass(games, rng.permutation(ng).tolist(), ratings, k)
            acc += np.asarray(ratings)
        return acc / n_ord

    point = _calibrate(run(range(n_targets), n_orderings), models, anchor, anchor_value)

    boot = np.empty((n_boot, n_models), dtype=float)
    for bi in range(n_boot):
        sel = rng.integers(0, n_targets, size=n_targets)
        boot[bi] = _calibrate(run(sel, boot_orderings), models, anchor, anchor_value)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)

    counts = (~table.isna()).sum(axis=1).reindex(models).to_numpy()
    return pd.DataFrame(
        {
            "model_id": models,
            "Elo": np.round(point).astype(int),
            "Elo_lo": np.round(lo).astype(int),
            "Elo_hi": np.round(hi).astype(int),
            "n_targets": counts.astype(int),
        }
    )


def win_counts(table: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Head-to-head win counts: ``M[i, j]`` = targets where model *i* beats *j* (ties 0.5).

    Only targets on which both models have a result are counted.
    """
    models = list(table.index)
    arr = table.to_numpy()
    n = len(models)
    m = np.zeros((n, n), dtype=float)
    for t in range(arr.shape[1]):
        col = arr[:, t]
        present = [i for i in range(n) if not np.isnan(col[i])]
        for a, b in combinations(present, 2):
            va, vb = col[a], col[b]
            if va > vb:
                m[a, b] += 1.0
            elif va < vb:
                m[b, a] += 1.0
            else:
                m[a, b] += 0.5
                m[b, a] += 0.5
    return models, m
