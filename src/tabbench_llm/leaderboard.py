"""Leaderboard utilities for ranking models from benchmark results.

The :class:`Leaderboard` class normalises per-(seed, dataset, model) metrics into a
ranked leaderboard.  Build one from a results directory produced by the pipeline, or
evaluate a new scikit-learn-compatible model in-process and add it.

Typical workflow
----------------
::

    from tabbench_llm import Leaderboard
    from sklearn.ensemble import RandomForestClassifier

    # 1. Load metrics from a results directory
    lb = Leaderboard.from_results_dir("results/bio_classification")

    # 2. Print current ranking
    print(lb.rank())

    # 3. Evaluate your model and add it to the leaderboard
    results = lb.evaluate_and_add("My-RF", RandomForestClassifier())
    print(lb.rank())

    # 4. Visualise
    lb.plot()
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from tabbench_llm.coverage import impute_failures, load_status
from tabbench_llm.metrics import PRIMARY_CLF_METRIC, PRIMARY_REG_METRIC

logger = logging.getLogger(__name__)

_RANK_COL = "Rank"

# Columns from display metadata kept verbatim (not recomputed from raw metrics)
_META_COLS = [
    "Model",
    "Category",
    "Elo",
    "Train Time s",
    "Infer. s/1K",
    "# Failed",
    "# Skipped",
    "# Retry",
]


class Leaderboard:
    """Manage and extend the TabBench-LLM leaderboard.

    Parameters
    ----------
    reg_metrics : pd.DataFrame
        Raw per-(seed, key, model) regression metrics.  Must contain columns
        ``seed``, ``key``, ``model``, ``rmse`` (and optionally ``mse``,
        ``mae``, ``r2``, …).
    clf_metrics : pd.DataFrame
        Raw per-(seed, key, model) classification metrics.  Must contain
        columns ``seed``, ``key``, ``model``, ``f1_macro`` (the primary
        ranking metric; and optionally ``matthews_corrcoef``, ``f1_macro``,
        ``f1_score``, ``roc_auc``, …).
    display_meta : pd.DataFrame
        Per-model display metadata indexed by ``model_id``.  Columns:
        ``model_id``, ``Model``, ``Category``, ``Elo``,
        ``Train Time s``, ``Infer. s/1K``.  Missing models (e.g. newly
        added) are filled with sensible defaults.

    Notes
    -----
    Use the class method :meth:`from_results_dir` to construct instances from a
    pipeline results directory — do not call ``__init__`` directly.
    """

    def __init__(
        self,
        reg_metrics: pd.DataFrame,
        clf_metrics: pd.DataFrame,
        display_meta: pd.DataFrame | None = None,
    ):
        self._reg_metrics = reg_metrics.copy()
        self._clf_metrics = clf_metrics.copy()
        self._display_meta = display_meta.copy() if display_meta is not None else pd.DataFrame()
        self._added_models: list[str] = []

        # Populated by _rebuild()
        self._overall: pd.DataFrame = pd.DataFrame()
        self._clf: pd.DataFrame = pd.DataFrame()
        self._reg: pd.DataFrame = pd.DataFrame()
        self._rebuild()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_results_dir(cls, results_dir: str) -> Leaderboard:
        """Load leaderboard from a local results directory.

        Expects ``metrics/classification_metrics.csv`` and
        ``metrics/regression_metrics.csv`` inside *results_dir*.

        Parameters
        ----------
        results_dir : str
            Path produced by running the benchmark pipeline
            (``scripts/feature_sweep.py``).

        Returns
        -------
        Leaderboard
        """
        metrics_dir = os.path.join(results_dir, "metrics")
        clf_path = os.path.join(metrics_dir, "classification_metrics.csv")
        reg_path = os.path.join(metrics_dir, "regression_metrics.csv")

        reg_df = pd.read_csv(reg_path) if os.path.exists(reg_path) else pd.DataFrame()
        clf_df = pd.read_csv(clf_path) if os.path.exists(clf_path) else pd.DataFrame()

        # Score failed fits at chance rather than omitting them, so a model cannot improve
        # its standing by crashing on the targets it finds hard.
        status = load_status(results_dir)
        return cls(impute_failures(reg_df, status), impute_failures(clf_df, status))

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self, task: str = "overall") -> pd.DataFrame:
        """Return a ranked leaderboard DataFrame.

        Parameters
        ----------
        task : {"overall", "classification", "regression"}
            Which leaderboard to return.

        Returns
        -------
        pd.DataFrame
            Sorted by Score (descending) with a ``Rank`` column prepended.
        """
        df = self._select_leaderboard(task).copy()
        if "Score" in df.columns:
            df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df[_RANK_COL] = df.index + 1
        cols = [_RANK_COL] + [c for c in df.columns if c != _RANK_COL]
        return df[cols]

    def summary(self) -> str:
        """Return a human-readable summary of the current leaderboard."""
        df = self.rank()
        lines = ["TabBench-LLM Leaderboard", "=" * 40]
        for _, row in df.iterrows():
            model = row.get("Model", row.get("model_id", "?"))
            score = row.get("Score", float("nan"))
            elo = row.get("Elo", float("nan"))
            elo_str = f"  Elo={elo:.0f}" if not (isinstance(elo, float) and np.isnan(elo)) else ""
            lines.append(f"  #{int(row[_RANK_COL]):2d}  {model:<28}  Score={score:.3f}{elo_str}")
        if self._added_models:
            lines.append(f"\nAdded models: {', '.join(self._added_models)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Adding new models
    # ------------------------------------------------------------------

    def add_results(
        self,
        model_name: str,
        metrics_df: pd.DataFrame,
    ) -> None:
        """Add raw per-(seed, key) metrics for a new model to the leaderboard.

        The metrics are stored in the same format as the paper's CSV files and
        the leaderboard scores are recomputed from all models combined
        (including the new one), so the normalization is updated automatically.

        Parameters
        ----------
        model_name : str
            Display name and internal identifier for the new model.
        metrics_df : pd.DataFrame
            DataFrame with one row per (seed, dataset_key) containing metric
            columns.  Must include ``seed`` and ``key``.  Regression rows need
            ``rmse``; classification rows need ``f1_macro``.  A ``model``
            column is added automatically.
        """
        df = metrics_df.copy()
        df["model"] = model_name

        reg_cols = {"rmse", "mse", "mae", "r2"}
        clf_cols = {
            "f1_score",
            "f1_macro",
            "balanced_accuracy",
            "matthews_corrcoef",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
        }

        if reg_cols & set(df.columns):
            keep = ["seed", "key", "model"] + [c for c in df.columns if c in reg_cols]
            self._reg_metrics = pd.concat(
                [self._reg_metrics, df[keep].dropna(subset=[PRIMARY_REG_METRIC])],
                ignore_index=True,
            )

        if clf_cols & set(df.columns):
            keep = ["seed", "key", "model"] + [c for c in df.columns if c in clf_cols]
            self._clf_metrics = pd.concat(
                [self._clf_metrics, df[keep].dropna(subset=[PRIMARY_CLF_METRIC])],
                ignore_index=True,
            )

        if model_name not in self._added_models:
            self._added_models.append(model_name)

        self._rebuild()
        logger.info("Added model '%s' to leaderboard.", model_name)

    def evaluate_and_add(
        self,
        model_name: str,
        model: Any,
        config_path: str,
        seeds: int = 3,
        task: str = "overall",
    ) -> pd.DataFrame:
        """Run a model through the full benchmark and add it to the leaderboard.

        This is a convenience wrapper that:

        1. Loads all benchmark datasets via :class:`~tabbench_llm.benchmark.TabBenchLLM`.
        2. Fits and evaluates *model* on each train/test split.
        3. Computes metrics.
        4. Calls :meth:`add_results` to insert the model into the leaderboard.

        *model* must expose a scikit-learn-compatible API:
        ``fit(X, y)`` and ``predict(X)``.

        Parameters
        ----------
        model_name : str
            Display name for the leaderboard.
        model : object
            A scikit-learn-compatible estimator.
        config_path : str
            Path to a benchmark config JSON describing the datasets to run on.
        seeds : int
            Number of random seeds to average over (default 3).
        task : str
            Filter to only regression or classification datasets when set to
            ``"regression"`` or ``"classification"``; default ``"overall"``
            runs both.

        Returns
        -------
        pd.DataFrame
            Per-(seed, key) metrics for the newly evaluated model, in the same
            format as the raw metrics CSVs.
        """
        import time

        from tabbench_llm.benchmark import configure_benchmark
        from tabbench_llm.config import load_config
        from tabbench_llm.dataset import TaskType
        from tabbench_llm.metrics import compute_metrics

        config = load_config(config_path)

        records = []
        fit_times: list[float] = []
        infer_us_per_sample: list[float] = []

        for seed in range(seeds):
            config["random_state"] = seed
            bench = configure_benchmark(config)
            for train_df, test_df, key, task_type in bench:
                if train_df is None:
                    continue
                if task == "regression" and task_type != TaskType.Regression:
                    continue
                if task == "classification" and task_type != TaskType.Classification:
                    continue
                label_col = train_df.columns[-1]
                X_train = train_df.drop(columns=[label_col]).values
                y_train = train_df[label_col].values
                X_test = test_df.drop(columns=[label_col]).values
                y_test = test_df[label_col].values

                try:
                    t0 = time.perf_counter()
                    model.fit(X_train, y_train)
                    fit_times.append(time.perf_counter() - t0)

                    t1 = time.perf_counter()
                    y_pred = model.predict(X_test)
                    infer_s = time.perf_counter() - t1
                    infer_us_per_sample.append((infer_s / len(X_test)) * 1e6)

                    metrics = compute_metrics(y_test, y_pred, task_type=task_type)
                    records.append(
                        {
                            "seed": seed,
                            "key": key,
                            "task_type": task_type.name,
                            **metrics,
                        }
                    )
                except Exception as e:
                    logger.warning("Model %s failed on %s (seed %d): %s", model_name, key, seed, e)

        metrics_df = pd.DataFrame(records)
        if not metrics_df.empty:
            # µs/sample == ms/1K — same unit as the precomputed "Infer. s/1K" column
            mean_train_s = float(np.mean(fit_times)) if fit_times else float("nan")
            mean_infer_ms_per_1k = (
                float(np.mean(infer_us_per_sample)) if infer_us_per_sample else float("nan")
            )
            self._upsert_display_meta(
                model_name,
                {
                    "Train Time s": mean_train_s,
                    "Infer. s/1K": mean_infer_ms_per_1k,
                },
            )
            self.add_results(model_name, metrics_df)
        return metrics_df

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        task: str = "overall",
        n_top: int = 30,
        figsize: tuple = (10, 8),
    ):
        """Plot a horizontal bar chart of model scores.

        Parameters
        ----------
        task : {"overall", "classification", "regression"}
            Which leaderboard to visualise.
        n_top : int
            Show only the top *n_top* models.
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        df = self.rank(task).head(n_top)
        model_col = "Model" if "Model" in df.columns else "model_id"
        models = df[model_col].tolist()
        scores = df["Score"].tolist() if "Score" in df.columns else [0] * len(df)
        colors = ["#e74c3c" if m in self._added_models else "#3498db" for m in models]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(models)), scores[::-1], color=colors[::-1])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models[::-1])
        ax.set_xlabel("Score")
        ax.axvline(0, color="black", linewidth=0.5)

        if self._added_models:
            from matplotlib.patches import Patch

            legend = [
                Patch(color="#3498db", label="Baseline (v0.1)"),
                Patch(color="#e74c3c", label="New model"),
            ]
            ax.legend(handles=legend, loc="lower right")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal: rebuild leaderboard from raw metrics
    # ------------------------------------------------------------------

    def _upsert_display_meta(self, model_id: str, values: dict) -> None:
        """Insert or update display metadata columns for *model_id*."""
        if self._display_meta.empty or "model_id" not in self._display_meta.columns:
            self._display_meta = pd.DataFrame([{"model_id": model_id, **values}])
            return
        mask = self._display_meta["model_id"] == model_id
        if mask.any():
            for col, val in values.items():
                self._display_meta.loc[mask, col] = val
        else:
            new_row = pd.DataFrame([{"model_id": model_id, **values}])
            self._display_meta = pd.concat([self._display_meta, new_row], ignore_index=True)

    def _rebuild(self) -> None:
        """Recompute Score, Avg Rank, Improvability from current raw metrics.

        Ranks by the configured primary metrics (:data:`PRIMARY_CLF_METRIC` /
        :data:`PRIMARY_REG_METRIC`); the full metric suite stays in the raw CSVs and is
        surfaced for display separately.
        """
        reg_scores = _per_dataset_scores(
            self._reg_metrics, PRIMARY_REG_METRIC, higher_is_better=False
        )
        clf_scores = _per_dataset_scores(
            self._clf_metrics, PRIMARY_CLF_METRIC, higher_is_better=True
        )

        self._reg = self._merge_meta(_aggregate_leaderboard(reg_scores))
        self._clf = self._merge_meta(_aggregate_leaderboard(clf_scores))

        non_empty = [df for df in [reg_scores, clf_scores] if not df.empty]
        all_scores = pd.concat(non_empty, ignore_index=True) if non_empty else reg_scores
        self._overall = self._merge_meta(_aggregate_leaderboard(all_scores))

    def _merge_meta(self, lb: pd.DataFrame) -> pd.DataFrame:
        """Merge display metadata into a computed leaderboard DataFrame."""
        if lb.empty:
            return lb
        if self._display_meta.empty or "model_id" not in self._display_meta.columns:
            lb["Model"] = lb["model_id"]
            return lb
        available = [c for c in _META_COLS if c in self._display_meta.columns]
        meta = self._display_meta[["model_id"] + available]
        merged = lb.merge(meta, on="model_id", how="left")
        if "Model" in merged.columns:
            merged["Model"] = merged["Model"].fillna(merged["model_id"])
        else:
            merged["Model"] = merged["model_id"]
        col_order = ["model_id", "Model"] + [
            c
            for c in [
                "Category",
                "Elo",
                "Score",
                "Avg Rank",
                "Improvability",
                # Coverage sits next to Score on purpose: the two are only interpretable
                # together now that absent-by-design pairings are excluded rather than zeroed.
                "# Targets",
                "# Failed",
                "# Skipped",
                "# Retry",
                "Train Time s",
                "Infer. s/1K",
            ]
            if c in merged.columns
        ]
        extra = [c for c in merged.columns if c not in col_order]
        return merged[col_order + extra]

    def _select_leaderboard(self, task: str) -> pd.DataFrame:
        if task == "overall":
            return self._overall
        if task == "classification":
            return self._clf
        if task == "regression":
            return self._reg
        raise ValueError(
            f"Unknown task {task!r}. Use 'overall', 'classification', or 'regression'."
        )


# ------------------------------------------------------------------
# Helpers for building leaderboard from raw metrics
# ------------------------------------------------------------------


def _per_dataset_scores(
    metrics_df: pd.DataFrame,
    metric_col: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Compute per-(model, dataset) min-max normalized scores and ranks.

    For each dataset key the primary metric is min-max normalized across the whole
    field of models:
    - best model in that dataset  → 1.0
    - worst model in that dataset → 0.0
    - linear in between (no clipping)

    Anchoring the zero at the worst model, not the per-dataset median, keeps the
    bottom half of the field discriminated: a mediocre and a catastrophic model get
    different scores instead of both collapsing to 0.0. A dataset on which every
    model ties (e.g. a near-trivial task where all models are at ceiling) is
    non-discriminative — it scores every model 1.0, so it neither rewards nor
    penalises anyone.

    Returns a DataFrame with columns ``model``, ``key``, ``norm_score``,
    ``rank``.  Only datasets with at least two models are included.
    """
    if metrics_df.empty or metric_col not in metrics_df.columns:
        return pd.DataFrame(columns=["model", "key", "norm_score", "rank"])

    # Average the metric across seeds for each (model, dataset)
    per_ds = (
        metrics_df.groupby(["model", "key"])[metric_col]
        .mean()
        .reset_index()
        .dropna(subset=[metric_col])
    )

    results = []
    for key, group in per_ds.groupby("key"):
        if len(group) < 2:
            continue
        vals = group[metric_col].values
        e_vals = vals if higher_is_better else -vals

        best_e = float(np.max(e_vals))
        worst_e = float(np.min(e_vals))
        denom = best_e - worst_e

        if denom > 0:
            norm = (e_vals - worst_e) / denom
        else:
            # Every model tied on this dataset: indistinguishable, nothing to
            # improve. Score them all 1.0 so a non-discriminative dataset neither
            # rewards nor penalises anyone (0.0 would wrongly imply max Improvability).
            norm = np.ones(len(e_vals))

        ranks = pd.Series(e_vals).rank(ascending=False, method="min").values

        for i, row in enumerate(group.itertuples(index=False)):
            results.append(
                {
                    "model": row.model,
                    "key": key,
                    "norm_score": float(norm[i]),
                    "rank": float(ranks[i]),
                }
            )

    return pd.DataFrame(results)


def _aggregate_leaderboard(per_ds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-dataset scores into a leaderboard row per model.

    Every model is scored on the targets it actually has a result for, and ``# Targets``
    reports how many that is. Absent ``(model, key)`` pairings are **not** imputed here: a
    fit that was attempted and failed has already been imputed at chance level upstream
    (:func:`tabbench_llm.coverage.impute_failures`), so what remains absent is a unit the
    benchmark excluded by design — a declared input-size limit, a prompt that cannot fit the
    model's context window, a degenerate split. Scoring those as zero ranks the benchmark's
    own limits rather than the model, which is why ``# Targets`` must be read alongside
    ``Score`` and why Elo — which compares models through shared opponents instead of a mean
    over unequal pools — is the headline ranking.
    """
    if per_ds.empty:
        return pd.DataFrame(columns=["model_id", "Score", "Avg Rank", "Improvability", "# Targets"])

    agg = (
        per_ds.groupby("model")
        .agg(
            Score=("norm_score", "mean"),
            avg_rank=("rank", "mean"),
            n_targets=("key", "nunique"),
        )
        .reset_index()
        .rename(columns={"model": "model_id", "avg_rank": "Avg Rank", "n_targets": "# Targets"})
    )
    agg["Improvability"] = ((1.0 - agg["Score"]) * 100).round(1)
    agg["Score"] = agg["Score"].round(4)
    agg["Avg Rank"] = agg["Avg Rank"].round(1)
    return agg


def _build_leaderboard_from_metrics(
    clf_df: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall/clf/reg leaderboard DataFrames from raw metrics CSVs.

    Kept for backwards compatibility with :meth:`Leaderboard.from_results_dir`.
    """
    lb = Leaderboard(reg_df, clf_df)
    return lb._overall, lb._clf, lb._reg
