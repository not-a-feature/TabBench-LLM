"""Metric computation step for the benchmark pipeline (Step 2).

Reads prediction CSV files produced by :mod:`tabbench_llm.predictions` and
computes per-(seed, dataset, model) metrics, writing them to
``metrics/classification_metrics.csv`` and ``metrics/regression_metrics.csv``
inside the results directory.

Incremental updates
-------------------
Already-computed (seed, key, model) triples are skipped, so you can safely
re-run this step after adding new models or seeds.

Usage
-----
::

    from tabbench_llm.config import load_config
    from tabbench_llm.evaluation import compute_metrics_from_predictions

    config = load_config("configs/benchmark_v0.1.json")
    compute_metrics_from_predictions(config)
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from tabbench_llm.benchmark import configure_benchmark
from tabbench_llm.coverage import DESIGN_SKIPS, load_status
from tabbench_llm.dataset import TaskType
from tabbench_llm.metrics import compute_metrics
from tabbench_llm.seeds import get_seeds

logger = logging.getLogger(__name__)


def _load_ground_truth(predictions_dir: str, key: str, task_type_lookup: dict):
    """Return ``(data_test, task_type)`` from saved files, or ``(None, None)``."""
    if key not in task_type_lookup:
        return None, None
    path = os.path.join(predictions_dir, f"{key}_ground_truth.csv")
    if not os.path.exists(path):
        return None, None
    return pd.read_csv(path, index_col=0), task_type_lookup[key]


def _unparsed_frac(output_dir: str, seed, key: str, model_name: str) -> float | None:
    """Fraction of this cell's test rows the LLM gave no parseable label for, or ``None``.

    Those rows were answered with the majority class, which scores like any other prediction,
    so the rate belongs next to the score rather than only in the per-cell stats file. ``None``
    for models that do not report it (every trained baseline).
    """
    path = os.path.join(output_dir, f"seed_{seed}", "stats", f"{key}_{model_name}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        stats = json.load(f)
    return stats["llm_unparsed_frac"] if "llm_unparsed_frac" in stats else None


def _build_excluded_keys(config) -> set[str]:
    """Return the set of dataset keys to exclude based on config.

    Three mechanisms (all applied in combination):

    * ``exclude_keys``    — exact keys (e.g. ``"sugar_mixtures_high_snr_4"``)
    * ``exclude_datasets``— all keys for a dataset (e.g. ``"timegate_fermentation"``)
    * ``exclude_targets`` — keys whose target name matches (e.g. ``"time_h"``)
    """
    excluded: set[str] = set(config["exclude_keys"])

    exclude_datasets = set(config["exclude_datasets"])
    exclude_names = set(config["exclude_targets"])

    if exclude_datasets or exclude_names:
        stats_path = os.path.join(config["output_dir"], "dataset_stats.json")
        if not os.path.exists(stats_path):
            logger.warning(
                "exclude_datasets/exclude_targets set but dataset_stats.json not found "
                "— dataset/name-based exclusions skipped."
            )
        else:
            with open(stats_path) as f:
                stats = json.load(f)
            for ds_id, s in stats.items():
                if ds_id in exclude_datasets:
                    n_targets = len((s or {}).get("target_names") or []) or 1
                    for idx in range(n_targets):
                        excluded.add(f"{ds_id}_{idx}")
                if exclude_names and s and s.get("target_names"):
                    for idx, name in enumerate(s["target_names"]):
                        if name in exclude_names:
                            excluded.add(f"{ds_id}_{idx}")

    return excluded


def compute_metrics_from_predictions(config):
    """Compute metrics from saved prediction CSV files.

    Parameters
    ----------
    config : dict
        Loaded benchmark configuration.
    """
    logger.info("=" * 60 + "\nSTEP 2: Computing Metrics")

    output_dir = config["output_dir"]
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    seeds = get_seeds(config)
    model_names = config["models"]
    excluded_keys = _build_excluded_keys(config)
    if excluded_keys:
        logger.info("Excluding %d key(s): %s", len(excluded_keys), sorted(excluded_keys))

    clf_csv = os.path.join(metrics_dir, "classification_metrics.csv")
    reg_csv = os.path.join(metrics_dir, "regression_metrics.csv")
    clf_excl_csv = os.path.join(metrics_dir, "excluded_classification_metrics.csv")
    reg_excl_csv = os.path.join(metrics_dir, "excluded_regression_metrics.csv")

    def _load_existing(path):
        if not os.path.isfile(path):
            return pd.DataFrame(), set()
        df = pd.read_csv(path)
        done = {(int(r.seed), r.key, r.model) for r in df.itertuples()}
        return df, done

    clf_existing, clf_done = _load_existing(clf_csv)
    reg_existing, reg_done = _load_existing(reg_csv)
    clf_excl_existing, clf_excl_done = _load_existing(clf_excl_csv)
    reg_excl_existing, reg_excl_done = _load_existing(reg_excl_csv)
    already_done = clf_done | reg_done | clf_excl_done | reg_excl_done

    clf_rows, reg_rows, clf_excl_rows, reg_excl_rows = [], [], [], []

    def _save(rows, existing, path):
        """Merge freshly computed *rows* into the on-disk CSV and rewrite it (plus its
        summary). Returns the combined frame so the caller carries it forward; *rows* is
        cleared by the caller after. Called after every seed so an interrupted metrics step
        keeps the seeds it already finished instead of discarding them all."""
        if not rows:
            return existing
        new = pd.DataFrame(rows)
        combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
        combined.to_csv(path, index=False)
        _write_summary(combined, path.replace(".csv", "_summary.csv"))
        logger.info("Wrote %d new rows to %s (total %d).", len(new), path, len(combined))
        return combined

    # Units the predictions step excluded by design. A resumed run can leave prediction CSVs
    # behind from before an exclusion applied (e.g. a cell later recognised as exceeding the
    # model's context window); scoring those would put rows in the metrics CSV that the
    # status records say should not exist.
    status = load_status(output_dir)
    excluded = status[(status["status"] == "skip") & status["reason"].isin(DESIGN_SKIPS)]
    excluded_units = {(r.seed, r.key, r.model) for r in excluded.itertuples(index=False)}
    if excluded_units:
        logger.info("Excluding %d unit(s) skipped by design.", len(excluded_units))

    for seed in seeds:
        logger.info("--- Seed %s ---", seed)
        predictions_dir = os.path.join(output_dir, f"seed_{seed}", "predictions")

        # Build (key → task_type) lookup from ground-truth files
        gt_keys = (
            [
                f[: -len("_ground_truth.csv")]
                for f in os.listdir(predictions_dir)
                if f.endswith("_ground_truth.csv")
            ]
            if os.path.isdir(predictions_dir)
            else []
        )

        config["random_state"] = seed
        benchmark = configure_benchmark(config, init_benchmark=not bool(gt_keys))

        if gt_keys:
            # Build task_type_lookup directly from dataset name lists.
            # _index is empty when init_benchmark=False (data loading skipped),
            # so we infer task type from membership in the regression name set.
            reg_names = set(benchmark.dataset_names_regression)
            task_type_lookup = {
                key: (
                    TaskType.Regression
                    if benchmark.split_key(key)[0] in reg_names
                    else TaskType.Classification
                )
                for key in gt_keys
            }

            items = []
            for key in gt_keys:
                data_test, task_type = _load_ground_truth(predictions_dir, key, task_type_lookup)
                if data_test is not None:
                    items.append((data_test, key, task_type))
        else:
            items = [(dt, k, tt) for _, dt, k, tt in benchmark]

        pbar = tqdm(total=len(items) * len(model_names), desc=f"Metrics seed {seed}")

        for data_test, key, task_type in items:
            is_excluded = key in excluded_keys
            dataset_name, target_idx = benchmark.split_key(key)

            for model_name in model_names:
                pbar.set_description(f"Seed {seed} | {key} | {model_name}")
                pred_path = os.path.join(predictions_dir, f"{key}_{model_name}_predictions.csv")

                if not os.path.exists(pred_path) or (seed, key, model_name) in excluded_units:
                    pbar.update(1)
                    continue

                y_pred = pd.read_csv(pred_path, index_col=0).sort_index()
                data_test_ = data_test.sort_index()

                if not np.array_equal(data_test_.index, y_pred.index):
                    logger.warning(
                        "Index mismatch for %s / %s seed %s — deleting stale prediction.",
                        key,
                        model_name,
                        seed,
                    )
                    os.remove(pred_path)
                    pbar.update(1)
                    continue

                if (seed, key, model_name) in already_done:
                    pbar.update(1)
                    continue

                # Guard: classification predictions must not be continuous floats
                if task_type == TaskType.Classification:
                    pred_vals = y_pred["target"]
                    if (
                        pd.api.types.is_float_dtype(pred_vals)
                        and not pred_vals.apply(float.is_integer).all()
                    ):
                        logger.warning(
                            "Skipping %s / %s seed %s: continuous floats for Classification.",
                            key,
                            model_name,
                            seed,
                        )
                        pbar.update(1)
                        continue

                row = {
                    "seed": seed,
                    "key": key,
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "target_idx": target_idx,
                    "model": model_name,
                }
                # A CSV round-trip can infer different dtypes for the ground-truth vs the
                # prediction column (e.g. anneal's 'U' class makes the truth column object
                # while an all-numeric prediction column parses as int), which sklearn rejects
                # with "Mix of label input types (string and number)". Classification labels are
                # categorical identities, so compare them as strings; this also pins the proba
                # column order below to the same (string) class order.
                if task_type == TaskType.Classification:
                    y_true_target = data_test_["target"].astype(str)
                    y_pred_target = y_pred["target"].astype(str)
                else:
                    y_true_target = data_test_["target"]
                    y_pred_target = y_pred["target"]

                # Load probability matrix if available so log_loss / ROC-AUC
                # can be computed alongside the label-based metrics. Missing
                # proba file just means those metrics stay absent — no error.
                y_proba = None
                if task_type == TaskType.Classification:
                    row["llm_unparsed_frac"] = _unparsed_frac(output_dir, seed, key, model_name)
                    proba_path = pred_path.replace("_predictions.csv", "_proba.csv")
                    if os.path.exists(proba_path):
                        proba_df = pd.read_csv(proba_path, index_col=0).sort_index()
                        if np.array_equal(proba_df.index, data_test_.index):
                            # roc_auc / log_loss index the proba columns by sorted class order
                            # (np.unique / LabelBinarizer). Reorder the saved columns to that
                            # order explicitly instead of trusting the on-disk column order, so
                            # the probability metrics stay correct regardless of how the model
                            # serialised its classes.
                            wanted = [str(c) for c in np.unique(y_true_target)]
                            proba_df.columns = [str(c) for c in proba_df.columns]
                            if set(wanted).issubset(proba_df.columns):
                                y_proba = proba_df.reindex(columns=wanted).to_numpy()
                            else:
                                y_proba = proba_df.to_numpy()

                    # A one-hot "probability" is the predicted label written as a vector: it
                    # carries no ranking information, so ROC-AUC computed from it collapses
                    # onto a thresholded accuracy and log-loss is just the clip applied to
                    # wrong predictions. Both would then sit in the same column as the genuine
                    # ranking metrics of models that do emit a distribution, and compare as if
                    # they measured the same thing. Report them as absent instead.
                    row["proba_degenerate"] = y_proba is not None and bool(
                        np.isin(y_proba, (0.0, 1.0)).all()
                    )
                    if row["proba_degenerate"]:
                        logger.debug(
                            "One-hot probabilities for %s / %s seed %s — skipping roc_auc / "
                            "log_loss (see LLMModel.elicit_proba).",
                            key,
                            model_name,
                            seed,
                        )
                        y_proba = None

                row.update(
                    compute_metrics(y_true_target, y_pred_target, task_type, y_proba=y_proba)
                )

                if task_type == TaskType.Classification:
                    (clf_excl_rows if is_excluded else clf_rows).append(row)
                else:
                    (reg_excl_rows if is_excluded else reg_rows).append(row)

                pbar.update(1)
        pbar.close()

        # Persist after each seed so a kill mid-metrics keeps completed seeds (the rows
        # are merged into the on-disk CSV and the in-memory buffers reset).
        clf_existing = _save(clf_rows, clf_existing, clf_csv)
        clf_rows.clear()
        reg_existing = _save(reg_rows, reg_existing, reg_csv)
        reg_rows.clear()
        clf_excl_existing = _save(clf_excl_rows, clf_excl_existing, clf_excl_csv)
        clf_excl_rows.clear()
        reg_excl_existing = _save(reg_excl_rows, reg_excl_existing, reg_excl_csv)
        reg_excl_rows.clear()


def _write_summary(df: pd.DataFrame, path: str):
    """Write mean ± std per (key, model) across seeds."""
    group_cols = [
        c for c in ["key", "dataset", "task_type", "target_idx", "model"] if c in df.columns
    ]
    numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in ("seed", "target_idx")
    ]
    agg = {col: ["mean", "std"] for col in numeric}
    summary = df.groupby(group_cols).agg(agg)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.reset_index().to_csv(path, index=False)
