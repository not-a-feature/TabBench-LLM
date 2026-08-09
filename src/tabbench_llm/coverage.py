"""Per-unit run status: what passed, what the design excluded, and what failed.

The predictions step writes one JSON record per ``(key, model)`` unit under
``seed_<n>/stats/``. That record is the only place the pipeline distinguishes a unit the
benchmark excluded **by design** — a declared per-model input-size limit, a prompt that
cannot fit the model's context window, a degenerate split — from a unit the model was asked
to answer and **failed** on. Neither appears in the metrics CSVs, so any ranking that reads
only those CSVs cannot tell the two apart and ends up measuring the harness.

One convention, applied by every consumer (leaderboard ``Score``, the per-dataset score
heatmap, and Elo alike):

* ``fail``   — imputed at the run's chance-level baseline (:data:`BASELINE_MODEL`) on that
  ``(seed, key)``. A model that could not produce usable predictions delivered chance
  performance; that is a real outcome, and imputing it stops a model from improving its
  standing by breaking on the targets it finds hard. This is where an LLM's unusable answers
  (``unparseable_responses``) and its blown wall-clock budget (``cell_timeout``) land: both
  are outcomes of asking the model, not exclusions applied before asking it.
* ``skip``   — dropped from that model's target pool, and reported as reduced coverage
  (:func:`coverage_counts`). The design excluded the unit, so scoring it as a loss would
  rank the benchmark's own limits rather than the model.
* ``retry``  — a provider/infrastructure failure such as HTTP 429 or 503. It is not model
  performance and remains incomplete until rerun.
* no record — the run is also incomplete. :func:`assert_complete` refuses to publish.
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

#: Model whose result defines chance level on a target; failures are imputed to it.
BASELINE_MODEL = "DUMMY"

#: ``reason`` codes that mark a unit the benchmark excluded by design rather than a model
#: outcome. Kept as an explicit set so a new skip path must declare which side it is on.
#: ``context_window`` is the LLM counterpart of ``model_limit``: an in-context classifier
#: carries the whole training table in every request, so a table that does not fit the
#: model's declared window is an input the benchmark cannot present to it — the same kind of
#: declared size limit that keeps a foundation model off inputs it cannot ingest.
DESIGN_SKIPS = frozenset(
    {
        "model_limit",
        "context_window",
        "empty_split",
        "empty_split_after_filtering",
        "constant_target",
        "classification_only",
    }
)

_STATUS_COLUMNS = ["seed", "key", "model", "status", "reason"]


def load_status(results_dir: str) -> pd.DataFrame:
    """Read every ``seed_*/stats/*.json`` into a ``(seed, key, model, status, reason)`` frame."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "seed_*", "stats", "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        seed = int(os.path.basename(os.path.dirname(os.path.dirname(path))).removeprefix("seed_"))
        rows.append(
            {
                "seed": seed,
                "key": rec["dataset"],
                "model": rec["model"],
                "status": rec["status"],
                # Records written before the reason taxonomy carry only a free-text error.
                "reason": rec["reason"] if "reason" in rec else "",
            }
        )
    return pd.DataFrame(rows, columns=_STATUS_COLUMNS)


def coverage_counts(status: pd.DataFrame) -> pd.DataFrame:
    """Per-model unit counts, including non-terminal ``# Retry`` units.

    Counted over ``(seed, key, model)`` units, so a model skipped on one dataset across all
    five folds shows five skips — the same unit granularity the pipeline schedules on.
    """
    cols = ["model_id", "# Passed", "# Failed", "# Skipped", "# Retry"]
    if status.empty:
        return pd.DataFrame(columns=cols)
    counts = (
        status.assign(n=1)
        .pivot_table(index="model", columns="status", values="n", aggfunc="sum", fill_value=0)
        .reset_index()
        .rename(columns={"model": "model_id"})
    )
    for src, dst in (
        ("pass", "# Passed"),
        ("fail", "# Failed"),
        ("skip", "# Skipped"),
        ("retry", "# Retry"),
    ):
        counts[dst] = counts[src].astype(int) if src in counts.columns else 0
    return counts[cols]


def impute_failures(
    metrics_df: pd.DataFrame, status: pd.DataFrame, baseline_model: str = BASELINE_MODEL
) -> pd.DataFrame:
    """Append a chance-level row for every failed unit that produced no metrics.

    The appended row copies :data:`BASELINE_MODEL`'s metrics on the same ``(seed, key)`` and
    is flagged ``imputed``, so downstream aggregation scores the failure at chance instead
    of silently omitting it. Design skips (:data:`DESIGN_SKIPS`) are left absent. Returns
    *metrics_df* unchanged when there is nothing to impute.
    """
    if metrics_df.empty or status.empty:
        return metrics_df

    tasks = set(zip(metrics_df["seed"], metrics_df["key"]))
    have = set(zip(metrics_df["seed"], metrics_df["key"], metrics_df["model"]))
    baseline = metrics_df[metrics_df["model"] == baseline_model].set_index(["seed", "key"])

    failures = status[status["status"] == "fail"]
    rows = []
    for unit in failures.itertuples(index=False):
        task = (unit.seed, unit.key)
        # Only impute inside this task pool: a (seed, key) with no results at all is an
        # incomplete run, which assert_complete reports rather than papers over.
        if task not in tasks or (unit.seed, unit.key, unit.model) in have:
            continue
        assert task in baseline.index, (
            f"cannot impute {unit.model} failure on {unit.key} (seed {unit.seed}): "
            f"baseline {baseline_model!r} has no result on that unit either."
        )
        row = baseline.loc[task].to_dict()
        row.update({"seed": unit.seed, "key": unit.key, "model": unit.model, "imputed": True})
        rows.append(row)

    if not rows:
        return metrics_df
    out = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)
    out["imputed"] = out["imputed"].astype("boolean").fillna(False).astype(bool)
    return out


def missing_units(
    metrics_df: pd.DataFrame,
    status: pd.DataFrame,
    keys: list[str],
    models: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    """Units in the ``seeds x keys x models`` grid with neither a metric row nor a status record."""
    have = (
        set(zip(metrics_df["seed"], metrics_df["key"], metrics_df["model"]))
        if not metrics_df.empty
        else set()
    )
    terminal = status[status["status"].isin(["pass", "fail", "skip"])]
    recorded = (
        set(zip(terminal["seed"], terminal["key"], terminal["model"]))
        if not terminal.empty
        else set()
    )
    rows = [
        {"seed": s, "key": k, "model": m}
        for s in seeds
        for k in keys
        for m in models
        if (s, k, m) not in have and (s, k, m) not in recorded
    ]
    return pd.DataFrame(rows, columns=["seed", "key", "model"])


def assert_complete(
    metrics_df: pd.DataFrame,
    status: pd.DataFrame,
    keys: list[str],
    models: list[str],
    seeds: list[int],
) -> None:
    """Refuse to proceed unless every scheduled unit produced a metric or terminal status.

    Without this, a dataset that silently produced nothing in a cell is indistinguishable
    from one that was never configured: the metrics CSV simply has fewer rows, every
    aggregate quietly renormalises over the survivors, and the published figure reports a
    dataset count nobody chose.
    """
    missing = missing_units(metrics_df, status, keys, models, seeds)
    if missing.empty:
        return
    per_key = missing.groupby("key")["model"].nunique().sort_values(ascending=False)
    raise AssertionError(
        f"{len(missing)} scheduled unit(s) have neither metrics nor a status record — the run "
        f"is incomplete and must not be published. Affected targets (n models):\n"
        f"{per_key.to_string()}\n"
        f"Rerun the predictions step for these, or remove them from the config."
    )
