"""Turn a fetched :class:`RawDataset` into a native :class:`~tabbench_llm.dataset.Dataset`.

A gene-expression / feature matrix is exactly the wide-matrix shape the benchmark
consumes, so a bio dataset becomes a :class:`Dataset` (``features`` = the matrix,
``feature_names`` = the real gene/probe identifiers, ``targets`` = labels) with no
further changes downstream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_llm.data.cache import default_dataset_cache_dir, load_cached_raw, save_cached_raw
from tabbench_llm.data.datasets import get_spec
from tabbench_llm.data.loaders import get_loader
from tabbench_llm.dataset import Dataset, DatasetInfo, TaskType

if TYPE_CHECKING:
    from tabbench_llm.data.loaders.base import RawDataset

logger = logging.getLogger(__name__)

#: Benchmark-wide default feature cap. Datasets wider than this (e.g. TCGA RNA-seq at
#: ~60660 genes) are truncated to a uniform random subset of columns to bound compute time
#: while preserving the HDLSS (high-dim, low-sample) character of the benchmark. The cap
#: is configurable per run (``max_features_default`` in the config) and per dataset
#: (``max_features`` in the registry); set it to ``None`` to keep all features.
DEFAULT_MAX_FEATURES = 30_000


def cap_features(
    X: pd.DataFrame,
    *,
    max_features: int | None = DEFAULT_MAX_FEATURES,
    random_state: int = 0,
) -> pd.DataFrame:
    """Truncate a wide feature matrix to a random subset of ``max_features`` columns.

    No-op when ``max_features`` is ``None`` or ``X`` already has ``<= max_features``
    columns. Otherwise ``max_features`` columns are drawn uniformly at random (without
    replacement, seeded by ``random_state`` for reproducibility); survivors retain their
    original left-to-right order so the matrix stays readable.

    The draw is uniform over columns — it does not rank by variance. On expression data
    (TPM, microarray intensity) variance scales with mean expression, so a variance rank
    would preferentially retain highly-expressed genes and bias the retained set by
    modality; a random subset is unbiased with respect to expression level and, depending
    only on column indices rather than values, carries no risk of leakage.
    """
    if max_features is None or X.shape[1] <= max_features:
        return X
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(X.shape[1], size=max_features, replace=False))
    return X.iloc[:, idx]


def raw_to_dataset(raw: RawDataset, *, max_features: int | None = DEFAULT_MAX_FEATURES) -> Dataset:
    """Adapt a :class:`RawDataset` to a native :class:`~tabbench_llm.dataset.Dataset`.

    Parameters
    ----------
    raw : RawDataset
        The fetched dataset.
    max_features : int | None
        Feature cap (see :func:`cap_features`). ``None`` keeps all features.
    """
    is_classification = raw.problem_type in ("binary", "multiclass")

    # Keep every feature column with its dtype: numeric columns stay numeric and categorical /
    # string columns are preserved, so the LLM sees the real values and AutoGluon encodes them
    # natively. (Regular tabular datasets are mixed-type; forcing an all-numeric matrix would
    # drop categorical features entirely and make all-categorical datasets unusable.)
    X = raw.X
    assert X.shape[1] > 0, f"{raw.dataset_id}: dataset has no feature columns."
    y = raw.y
    # Loader-layer guarantee: drop rows whose target is missing — they cannot be
    # supervised, so no adapted Dataset should carry them. (Done here, the single
    # chokepoint all source loaders flow through, rather than per loader.)
    target_missing = y.isna().to_numpy()
    if target_missing.any():
        keep = ~target_missing
        logger.info(
            "%s: dropped %d row(s) with missing target (%d remain).",
            raw.dataset_id,
            int(target_missing.sum()),
            int(keep.sum()),
        )
        X = X.iloc[keep]
        y = y.iloc[keep]

    capped = cap_features(X, max_features=max_features)
    if capped.shape[1] < X.shape[1]:
        logger.info(
            "%s: capped features %d -> %d (max_features=%s).",
            raw.dataset_id,
            X.shape[1],
            capped.shape[1],
            max_features,
        )

    features = capped.reset_index(drop=True)
    features.columns = [str(c) for c in features.columns]
    feature_names = list(features.columns)

    targets = y.reset_index(drop=True).to_numpy()
    target_name = y.name or raw.metadata.get("target") or "target"
    task_type = TaskType.Classification if is_classification else TaskType.Regression

    info = DatasetInfo(
        id=raw.dataset_id,
        name=raw.dataset_id,
        task_type=task_type,
        metadata=dict(raw.metadata),
    )
    return Dataset(
        features=features,
        targets=targets,
        feature_names=feature_names,
        target_names=[str(target_name)],
        info=info,
    )


def load_raw_dataset(
    dataset_id: str,
    *,
    cache_dir: str | None = None,
    force_refetch: bool = False,
) -> RawDataset:
    """Return the fetched :class:`RawDataset` for ``dataset_id``, using the unified cache.

    On a cache miss the dataset is fetched via its source loader and cached so later
    runs skip both the download and the re-assembly.
    """
    root = cache_dir or str(default_dataset_cache_dir())

    if not force_refetch:
        cached = load_cached_raw(root, dataset_id)
        if cached is not None:
            return cached

    spec = get_spec(dataset_id)
    if not spec.enabled:
        raise ValueError(
            f"{dataset_id}: dataset is disabled in the registry; enable it before running."
        )
    if not spec.is_curated:
        raise ValueError(
            f"{dataset_id}: target/problem_type not curated; set them in the registry before running."
        )

    raw = get_loader(spec, cache_dir=root).fetch(spec)
    save_cached_raw(root, raw)
    return raw


def load_as_dataset(
    dataset_id: str,
    *,
    cache_dir: str | None = None,
    max_features: int | None = DEFAULT_MAX_FEATURES,
) -> Dataset:
    """Fetch (cached) and adapt a bio dataset to a native :class:`~tabbench_llm.dataset.Dataset`.

    The per-dataset registry ``max_features`` override takes precedence over the
    ``max_features`` argument when set.
    """
    raw = load_raw_dataset(dataset_id, cache_dir=cache_dir)
    spec = get_spec(dataset_id)
    cap = spec.max_features if spec.max_features is not None else max_features
    return raw_to_dataset(raw, max_features=cap)
