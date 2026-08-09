"""Benchmark grid: reasoning-mode x label-mode x feature-cap x sample-size x dataset x model.

For each reasoning mode (off / on = medium effort), applicable label mode (real class names vs.
opaque tokens), feature cap (:data:`FEATURE_CAPS`) and training-set size in ``samples`` this runs the
standard predictions -> metrics pipeline over every dataset x model, then aggregates the
per-cell metrics into one CSV. Cells run sequentially and resume (existing predictions are
skipped), so a re-run only fills gaps.

Usage::

    python scripts/grid.py --config configs/grid.json
    python scripts/grid.py --config configs/grid.json --skip-run          # just re-aggregate
    python scripts/grid.py --config configs/grid.json --models RF         # one model's slice

All knobs live in the config; see ``configs/grid.json``.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from tabbench_llm.config import load_config, model_keys, resolve_list
from tabbench_llm.coverage import impute_failures, load_status
from tabbench_llm.data import load_as_dataset
from tabbench_llm.evaluation import compute_metrics_from_predictions
from tabbench_llm.predictions import compute_predictions


def _cell_config(
    train_subsample,
    *,
    datasets,
    models,
    cv_folds,
    test_size,
    test_subsample,
    min_samples_per_class,
    random_state,
    output_dir,
    cache_dir,
    max_features,
    llm_settings,
):
    """A full run config for one sample-size cell (every dataset x model at this train size).

    ``train_subsample`` is the resolved few-shot size (``None`` = all training rows).

    Uses the resolved ``dataset_names_*`` keys (not the ``datasets_*`` spec keys) because this
    config is handed straight to ``compute_predictions`` / ``configure_benchmark`` rather than
    written to disk and then passed through ``load_config`` (which resolves the spec keys to
    the ``dataset_names_*`` the pipeline consumes and validates the schema), so the config file
    is also loadable by ``tabbench-llm site`` for its dataset feature counts."""
    return {
        "datasets_classification": datasets,
        "datasets_regression": [],
        "models": models,
        "test_size": test_size,
        "random_state": random_state,
        # One pass of the k folds. Under cv_folds this is the CV repeat count, and
        # get_seeds then yields split indices 0..k-1; random_state above is unused there.
        "n_repetitions": 1,
        "cv_folds": cv_folds,
        "min_samples_per_class": min_samples_per_class,
        "group_regression_splits": False,
        "max_features_default": max_features,
        "max_classes": None,
        "train_subsample": train_subsample,
        "test_subsample": test_subsample,
        "model_limits": {},
        "llm_settings": llm_settings,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "autogluon_time_limit": 300,
        "autogluon_presets": "medium_quality",
        "optimize": False,
        "ensemble": False,
        "num_hpo_trials": 0,
        "nan_policy": {"default": "native"},
        "exclude_keys": [],
        "exclude_datasets": [],
        "exclude_targets": [],
    }


#: The label-visibility axis: real class names vs. opaque tokens (see LLMModel.hide_labels).
LABEL_MODES = ["visible", "hidden"]

#: The reasoning axis, as (mode name -> reasoning_effort). "off" = no chain-of-thought;
#: "on" = medium effort (Mistral, which lacks "medium", snaps to its max "high" — see the
#: effort-snapping in LLMModel). The trained baselines are unaffected by either mode.
REASONING_MODES = [("off", "none"), ("on", "medium")]

#: The feature-cap axis: full features vs. random-subset caps (applied train-only, leak-free,
#: in benchmark._fit_apply_features). A cap is a no-op on datasets already narrower than it, so
#: it only bites the wide datasets — where it also lets an LLM fit tables that would otherwise
#: overflow the context window. ``None`` = all features.
#:
#: Redundancy note: a cap of K narrows only datasets with more than K features; on a narrower
#: dataset the cap produces a byte-identical result to the full-feature (None) arm. Running every
#: cap on every dataset would therefore recompute the same result many times — on this benchmark
#: 39/57 datasets are untouched by cap=32 and 49/57 by cap=64 — needlessly tripling the
#: expensive LLM cost. So each cap arm runs only the datasets it genuinely narrows (see
#: :func:`_cap_datasets`); the full-feature arm still covers every dataset.
FEATURE_CAPS: list[int | None] = [None, 32, 64]


def _raw_feature_counts(datasets: list[str], cache_dir: str) -> dict[str, int]:
    """Uncapped feature count per dataset (cached load), for the wide-only cap filter."""
    return {
        d: load_as_dataset(
            d, cache_dir=os.path.join(cache_dir, "datasets"), max_features=None
        ).features.shape[1]
        for d in datasets
    }


def _cap_datasets(cap: int | None, datasets: list[str], raw_counts: dict[str, int]) -> list[str]:
    """Datasets a cap actually affects: all for ``None``, else only those wider than the cap.

    Skips the datasets on which the cap is a no-op (see the redundancy note on
    :data:`FEATURE_CAPS`), so a cap arm never recomputes a result identical to the full-feature
    arm."""
    if cap is None:
        return list(datasets)
    return [d for d in datasets if raw_counts[d] > cap]


def _label_datasets(label_mode: str, datasets: list[str]) -> list[str]:
    """Datasets for which *label_mode* is a meaningful intervention.

    Synthetic targets have no real-world class names to reveal. A visible-label synthetic arm
    therefore adds calls without testing label semantics. Real datasets run both modes;
    synthetic datasets run only ``hidden`` (displayed as "opaque" on the website).
    """
    if label_mode == "visible":
        return [d for d in datasets if not d.startswith("synthetic_")]
    if label_mode == "hidden":
        return list(datasets)
    raise ValueError(f"unknown label mode: {label_mode!r}")


def _cap_name(cap: int | None) -> str:
    """Dir/label token for a feature cap: ``'full'`` (no cap) or the integer as a string."""
    return "full" if cap is None else str(cap)


def _cell_dir(output: str, reason_name: str, label_mode: str, cap: int | None, n) -> str:
    return os.path.join(
        output, f"reason_{reason_name}", f"labels_{label_mode}", f"feat_{_cap_name(cap)}", f"n_{n}"
    )


def _write_canonical_config(path: str, config: dict) -> None:
    """Write a cell's full-roster config once and refuse incompatible resume attempts.

    A model-sliced cluster run mutates ``models`` only in memory; the file remains the
    canonical description of every model scheduled for the cell. Reusing an output directory
    with changed splits, inference settings, or roster would otherwise mix measurements while
    making the last writer's config look authoritative.
    """
    if os.path.isfile(path):
        with open(path) as f:
            previous = json.load(f)
        comparable_previous = dict(previous)
        comparable_config = dict(config)
        for payload in (comparable_previous, comparable_config):
            if isinstance(payload.get("output_dir"), str):
                payload["output_dir"] = payload["output_dir"].replace("\\", "/")
        if comparable_previous != comparable_config:
            changed = sorted(
                key
                for key in set(comparable_previous) | set(comparable_config)
                if comparable_previous.get(key) != comparable_config.get(key)
            )
            extensible = {"models", "datasets_classification"}

            def append_only(old, new) -> bool:
                return (
                    isinstance(old, list)
                    and isinstance(new, list)
                    and old == [item for item in new if item in set(old)]
                    and set(old) < set(new)
                )

            additive_extension = bool(changed) and set(changed).issubset(extensible) and all(
                append_only(previous.get(key), config.get(key)) for key in changed
            )
            if additive_extension:
                # Adding datasets or comparison models does not alter any completed prediction:
                # splits are seeded per dataset and prompts/inference settings are unchanged.
                # Persist the expanded canonical definition so the existing tree can fill only
                # the new dataset/model units. Removing or reordering entries remains forbidden.
                expanded = dict(config)
                # Preserve the platform-neutral path spelling already recorded in a merged
                # result tree. Both slash styles resolve on Windows, and keeping it avoids
                # creating a superficial config conflict when this tree returns to Linux.
                expanded["output_dir"] = previous.get("output_dir", config.get("output_dir"))
                with open(path, "w") as f:
                    json.dump(expanded, f, indent=2)
                return
            raise RuntimeError(
                f"{path} already contains a different run definition (changed: {changed}). "
                "Choose a new output root; refusing to mix incompatible results."
            )
        return
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def _sample_subsample(n):
    """Resolve a ``samples`` entry to a train_subsample value: ``None`` (use all training
    rows, no subsampling) for "all"/"full", else the integer few-shot size."""
    return None if str(n).lower() in ("all", "full") else int(n)


def _feature_caps(config: dict) -> list[int | None]:
    """Resolve optional JSON feature caps (``\"full\"``/``null`` means no cap)."""
    raw_caps = config.get("feature_caps", FEATURE_CAPS)
    caps = [None if cap is None or str(cap).lower() == "full" else int(cap) for cap in raw_caps]
    assert caps and len(caps) == len(set(caps)), f"invalid feature_caps: {raw_caps!r}"
    return caps


def _reasoning_modes(names: list[str]) -> list[tuple[str, str]]:
    """Resolve configured reasoning names and reject misspellings early."""
    modes = [(name, effort) for name, effort in REASONING_MODES if name in names]
    assert modes and len(modes) == len(names), (
        f"reasoning={names!r} matched no/unknown mode of {[name for name, _ in REASONING_MODES]}."
    )
    return modes


def _legacy_cell_datasets(
    label_mode: str,
    cap: int | None,
    datasets: list[str],
    raw_counts: dict[str, int],
) -> list[str]:
    """Dataset list written by the original global-grid scheduler for one cell.

    These canonical lists deliberately remain stable even when a suite-specific schedule narrows
    what is *executed*. That lets a reduced grid resume and merge with results written by the
    old global grid without changing any saved ``config.json`` file.
    """
    cap_datasets = _cap_datasets(cap, datasets, raw_counts)
    label_datasets = set(_label_datasets(label_mode, datasets))
    return [dataset for dataset in cap_datasets if dataset in label_datasets]


def _build_schedule(
    config: dict,
    datasets: list[str],
    raw_counts: dict[str, int],
) -> dict[tuple[str, str, int | None, object], dict[str, list[str]]]:
    """Build a suite-aware plan keyed by the persistent result-cell axes.

    ``real_grid`` optionally narrows only the real-data suite, with ``reasoning``, ``labels``,
    ``feature_caps`` and ``samples`` keys. Each cell keeps the original full canonical dataset
    list on disk while executing only its scheduled subset; that makes old partial result trees
    safe to resume and merge.
    """
    global_reasoning = list(config.get("reasoning", [name for name, _ in REASONING_MODES]))
    global_caps = _feature_caps(config)
    global_samples = list(config["samples"])
    _reasoning_modes(global_reasoning)

    real_grid = config.get("real_grid", {})
    real_reasoning = list(real_grid.get("reasoning", global_reasoning))
    real_labels = list(real_grid.get("labels", LABEL_MODES))
    real_caps = [
        None if cap is None or str(cap).lower() == "full" else int(cap)
        for cap in real_grid.get("feature_caps", global_caps)
    ]
    real_samples = list(real_grid.get("samples", global_samples))
    _reasoning_modes(real_reasoning)
    assert real_labels and set(real_labels).issubset(LABEL_MODES), (
        f"real_grid.labels must be a non-empty subset of {LABEL_MODES}, got {real_labels!r}"
    )
    assert real_caps and len(real_caps) == len(set(real_caps)), (
        f"invalid real_grid.feature_caps: {real_grid.get('feature_caps')!r}"
    )
    assert real_samples, "real_grid.samples must not be empty"

    synthetic = {dataset for dataset in datasets if dataset.startswith("synthetic_")}
    real = set(datasets) - synthetic
    schedule: dict[tuple[str, str, int | None, object], dict[str, list[str]]] = {}

    def add(reason: str, label: str, cap: int | None, n, wanted: set[str]) -> None:
        canonical = _legacy_cell_datasets(label, cap, datasets, raw_counts)
        selected = [dataset for dataset in canonical if dataset in wanted]
        if not selected:
            return
        key = (reason, label, cap, n)
        if key not in schedule:
            schedule[key] = {"datasets": [], "canonical_datasets": canonical}
        elif schedule[key]["canonical_datasets"] != canonical:
            raise AssertionError(f"conflicting canonical datasets for grid cell {key!r}")
        combined = {*schedule[key]["datasets"], *selected}
        schedule[key]["datasets"] = [dataset for dataset in canonical if dataset in combined]

    # Primary synthetic suite stays on the global protocol. Visible labels add no synthetic cell.
    for reason, _ in _reasoning_modes(global_reasoning):
        for label in LABEL_MODES:
            for cap in global_caps:
                for n in global_samples:
                    add(reason, label, cap, n, synthetic)

    # Secondary real suite can be reduced independently to control the number of LLM calls.
    for reason, _ in _reasoning_modes(real_reasoning):
        for label in real_labels:
            for cap in real_caps:
                for n in real_samples:
                    add(reason, label, cap, n, real)

    return schedule


def aggregate_schedule(
    output: str,
    schedule: dict[tuple[str, str, int | None, object], dict[str, list[str]]],
) -> pd.DataFrame:
    """Aggregate only currently scheduled datasets, not legacy extra real-data cells."""
    frames = []
    for (reason_name, mode, cap, n), cell in schedule.items():
        cell_dir = _cell_dir(output, reason_name, mode, cap, n)
        path = os.path.join(cell_dir, "metrics", "classification_metrics.csv")
        if not os.path.isfile(path):
            print(f"  (no metrics at {path}) - skipping")
            continue
        df = impute_failures(pd.read_csv(path), load_status(cell_dir))
        df = df[df["dataset"].isin(cell["datasets"])].copy()
        if df.empty:
            print(f"  (no scheduled metrics at {path}) - skipping")
            continue
        df.insert(0, "n_train", n)
        df.insert(0, "feature_cap", _cap_name(cap))
        df.insert(0, "labels", mode)
        df.insert(0, "reasoning", reason_name)
        frames.append(df)
    if not frames:
        raise SystemExit("No scheduled metrics to aggregate. Run the grid first (omit --skip-run).")
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="grid config JSON (see configs/grid.json)")
    p.add_argument("--skip-run", action="store_true", help="only re-aggregate existing results")
    p.add_argument(
        "--models",
        help="comma-separated model keys, overriding the config's models list for a machine slice.",
    )
    p.add_argument(
        "--datasets",
        help="comma-separated dataset IDs to execute; canonical cell configs remain full-roster.",
    )
    p.add_argument("--output", help="output root, overriding the config's output value")
    args = p.parse_args()

    base = os.path.dirname(os.path.abspath(args.config))
    with open(args.config) as f:
        gc = json.load(f)

    datasets = resolve_list(gc["datasets"], base)
    roster = model_keys(resolve_list(gc["models"], base))
    models = list(roster)
    if args.models:
        selected_models = [model.strip() for model in args.models.split(",") if model.strip()]
        unknown = [model for model in selected_models if model not in roster]
        assert not unknown, f"--models {unknown} not in the config roster {roster}."
        models = selected_models
    selected_datasets = None
    if args.datasets:
        selected_datasets = {
            dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()
        }
        unknown = sorted(selected_datasets - set(datasets))
        assert not unknown, f"--datasets {unknown} not in the config dataset roster."
    output = args.output if args.output else gc["output"]
    os.makedirs(output, exist_ok=True)

    # Feature caps decide which wide datasets have a non-redundant capped arm. The same plan is
    # used for execution and aggregation, so old results in now-unscheduled real-data cells are
    # retained on disk but cannot leak back into the current reduced protocol.
    raw_counts = _raw_feature_counts(datasets, gc["cache_dir"])
    schedule = _build_schedule(gc, datasets, raw_counts)

    if not args.skip_run:
        n_units = sum(
            len(
                cell["datasets"]
                if selected_datasets is None
                else [dataset for dataset in cell["datasets"] if dataset in selected_datasets]
            )
            for cell in schedule.values()
        ) * len(models)
        print(
            "Grid: "
            f"synthetic global axes plus real_grid={gc.get('real_grid', 'global default')} "
            f"x models={len(models)} = {n_units} dataset-model cell(s)."
        )
        for (reason_name, mode, cap, n), cell in schedule.items():
            execution_datasets = cell["datasets"]
            if selected_datasets is not None:
                execution_datasets = [
                    dataset for dataset in execution_datasets if dataset in selected_datasets
                ]
                if not execution_datasets:
                    continue
            effort = dict(REASONING_MODES)[reason_name]
            os.environ["TABBENCH_LLM_REASONING_EFFORT"] = effort
            os.environ["TABBENCH_LLM_HIDE_LABELS"] = "1" if mode == "hidden" else "0"
            out_dir = _cell_dir(output, reason_name, mode, cap, n)
            os.makedirs(out_dir, exist_ok=True)

            # Preserve the original global cell definition on disk. Existing Windows results
            # therefore resume without a config conflict even though this invocation executes a
            # reduced real-data subset from that cell.
            cfg = _cell_config(
                _sample_subsample(n),
                datasets=cell["canonical_datasets"],
                models=roster,
                cv_folds=gc["cv_folds"],
                test_size=gc["test_size"],
                test_subsample=gc["test_subsample"],
                min_samples_per_class=gc["min_samples_per_class"],
                random_state=gc["random_state"],
                output_dir=out_dir,
                cache_dir=gc["cache_dir"],
                max_features=cap,
                llm_settings=gc["llm_settings"],
            )
            cfg_path = os.path.join(out_dir, "config.json")
            _write_canonical_config(cfg_path, cfg)
            print(
                f"\n=== reasoning={reason_name} labels={mode} feat={_cap_name(cap)} "
                f"n_train={n} datasets={len(execution_datasets)} ==="
            )
            loaded = load_config(cfg_path)
            # Model slicing and suite-specific dataset selection are execution-only. The on-disk
            # config remains the canonical legacy-compatible record used by result merging.
            loaded["models"] = models
            loaded["dataset_names_classification"] = execution_datasets
            compute_predictions(loaded)
            compute_metrics_from_predictions(loaded)

    grid = aggregate_schedule(output, schedule)
    grid_path = os.path.join(output, "grid_metrics.csv")
    grid.to_csv(grid_path, index=False)
    grid.to_csv(os.path.join(output, "feature_grid_metrics.csv"), index=False)
    report_metrics = ["balanced_accuracy", "matthews_corrcoef", "f1_macro", "f1_score", "roc_auc"]
    present = [metric for metric in report_metrics if metric in grid.columns]
    summary = (
        grid.groupby(["dataset", "model", "reasoning", "labels", "feature_cap", "n_train"])[present]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["__".join(column).rstrip("_") for column in summary.columns]
    summary_path = os.path.join(output, "feature_grid_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {grid_path}")
    print(f"Wrote {summary_path} (+ feature_grid_metrics.csv) for the site grid.")


if __name__ == "__main__":
    main()
