"""HDLSS biological datasets for the TabBench-LLM benchmark.

This subpackage fetches and curates high-dimensional, low-sample-size (HDLSS)
biological datasets (GEO, TCGA, Kaggle, OpenML).  The genuinely source-specific
work lives in :mod:`~tabbench_llm.data.loaders` and the configurable registry
:mod:`~tabbench_llm.data.datasets`; everything downstream (cross-validation,
AutoGluon fitting, metrics, leaderboard) is the standard pipeline, reached via the
:class:`~tabbench_llm.data.loaders.base.RawDataset` ->
:class:`~tabbench_llm.dataset.Dataset` adapter.

Nothing heavy (the per-source fetch libraries) is imported at package import time,
so ``import tabbench_llm.data`` stays cheap and dependency-light.
"""

from __future__ import annotations

from tabbench_llm.data.adapter import (
    DEFAULT_MAX_FEATURES,
    cap_features,
    load_as_dataset,
    load_raw_dataset,
    raw_to_dataset,
)
from tabbench_llm.data.datasets import (
    DATASETS,
    DatasetSpec,
    dataset_names,
    get_spec,
    is_known_dataset,
    reload,
    runnable_specs,
)

__all__ = [
    "DATASETS",
    "DEFAULT_MAX_FEATURES",
    "DatasetSpec",
    "dataset_names",
    "raw_to_dataset",
    "cap_features",
    "get_spec",
    "is_known_dataset",
    "load_as_dataset",
    "load_raw_dataset",
    "reload",
    "runnable_specs",
]
