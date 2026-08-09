"""Configuration loading and validation for the benchmark pipeline."""

import json
import os

#: Every key a benchmark config must declare. The config file is the complete, explicit
#: record of a run: nothing defaults implicitly, so a missing or misspelled key is an
#: error rather than a silent fallback to a code default. Optional features are listed
#: here too and must be set explicitly (``null`` / ``[]`` / ``{}``).
REQUIRED_KEYS = frozenset(
    {
        # data + models
        "datasets_classification",
        "datasets_regression",
        "models",
        # split + repetitions
        "test_size",
        "random_state",
        "n_repetitions",
        "cv_folds",
        "min_samples_per_class",
        "group_regression_splits",
        "max_features_default",
        "max_classes",
        "train_subsample",
        "test_subsample",
        "model_limits",
        "llm_settings",
        # io
        "cache_dir",
        "output_dir",
        # training
        "autogluon_time_limit",
        "autogluon_presets",
        "optimize",
        "ensemble",
        "num_hpo_trials",
        "nan_policy",
        # evaluation
        "exclude_keys",
        "exclude_datasets",
        "exclude_targets",
    }
)


def resolve_list(value, base_dir):
    """Resolve a dataset/model list spec to a Python list.

    A list is returned as-is; a string is a path to a JSON array file, resolved
    relative to *base_dir* unless absolute. ``None`` passes through (the caller's
    "use the default set" sentinel).
    """
    if value is None or isinstance(value, list):
        return value
    path = value if os.path.isabs(value) else os.path.join(base_dir, value)
    with open(path) as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"List config file {path} must contain a JSON array.")
    return items


def model_keys(models):
    """The plain model-key list the pipeline consumes, from either the object form
    (``configs/models/*.json``) or the bare-key lists the grid writes into per-cell configs."""
    return [m["key"] if isinstance(m, dict) else m for m in models]


def model_limits(models):
    """Map model key -> ``{'max_features', 'max_samples'}`` for models declaring a size limit.

    Read from the object form (``configs/models/*.json``): a model may set ``max_features``
    and/or ``max_samples``. The prediction step skips a ``(model, dataset, cell)`` unit whose
    dataset raw feature count exceeds ``max_features`` **and** whose training rows exceed
    ``max_samples`` (both, so the model still runs on narrow data at any N and on wide data at
    small N) — used to keep memory-heavy foundation models off inputs that would OOM. Models
    declaring neither key are omitted (no limit). The grid writes this map into each per-cell
    config as ``model_limits`` (the bare-key model lists there no longer carry the objects)."""
    limits = {}
    for m in models:
        if isinstance(m, dict) and ("max_features" in m or "max_samples" in m):
            limits[m["key"]] = {
                "max_features": m["max_features"] if "max_features" in m else None,
                "max_samples": m["max_samples"] if "max_samples" in m else None,
            }
    return limits


def load_config(config_path):
    """Load, validate, and normalise a benchmark JSON config file.

    The config must declare *every* key in :data:`REQUIRED_KEYS` and nothing else (keys
    prefixed with ``_`` are treated as comments and ignored), so a run is fully described
    by its config — no key silently falls back to a code default, and a typo'd key is
    rejected rather than quietly ignored.

    ``datasets_classification`` / ``datasets_regression`` / ``models`` each accept an
    inline list or a path to a JSON array file (relative to the config's directory); set
    a datasets key to ``null`` to load every registered dataset of that task.
    """
    with open(config_path) as f:
        config = json.load(f)

    keys = {k for k in config if not k.startswith("_")}
    missing = REQUIRED_KEYS - keys
    unknown = keys - REQUIRED_KEYS
    assert not missing, f"{config_path}: missing required config key(s): {sorted(missing)}"
    assert not unknown, f"{config_path}: unknown config key(s): {sorted(unknown)}"

    base_dir = os.path.dirname(os.path.abspath(config_path))
    config["dataset_names_classification"] = resolve_list(
        config["datasets_classification"], base_dir
    )
    config["dataset_names_regression"] = resolve_list(config["datasets_regression"], base_dir)
    # Device tags (if any) are scheduling metadata only; the pipeline consumes plain keys.
    config["models"] = model_keys(resolve_list(config["models"], base_dir))

    return config
