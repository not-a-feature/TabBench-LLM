"""Caching for fetched bio datasets.

Two layers of caching keep the loaders from re-downloading:

1. **Source caches** (per loader): the TCGA full matrix (``tcga_raw/``) and the GEO SOFT
   file (``geo_raw/``) are cached by the loaders themselves; OpenML and Kaggle reuse
   their own native caches (``~/.cache/openml``, ``~/.cache/kagglehub``).
2. **Unified dataset cache** (this module): the assembled :class:`RawDataset`
   (``datasets/<dataset_id>.pkl``) so a second run skips fetching *and* re-assembling.

All bio caches live under one root (``default_dataset_cache_dir()`` or an explicit
``cache_dir`` passed through from :class:`~tabbench_llm.benchmark.TabBenchLLM`).
Pickle is used to match the split-cache convention (no extra deps).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tabbench_llm.data.loaders.base import RawDataset

#: Env var to override the bio cache root.
_CACHE_ENV = "TABBENCH_LLM_CACHE"


def default_dataset_cache_dir() -> Path:
    """Bio cache root used when no explicit ``cache_dir`` is provided.

    Resolves ``$TABBENCH_LLM_CACHE`` (or ``~/.cache/tabbench_llm``) and appends ``bio``.
    """
    base = os.environ.get(_CACHE_ENV) or str(Path.home() / ".cache" / "tabbench_llm")
    return Path(base) / "datasets"


def _safe(dataset_id: str) -> str:
    """Filesystem-safe form of a ``dataset_id`` for use as a path component."""
    return dataset_id.replace("/", "_").replace("\\", "_")


def dataset_cache_path(root: str | Path, dataset_id: str) -> Path:
    """Path to the unified per-dataset cache file under ``root/datasets``."""
    return Path(root) / "datasets" / f"{_safe(dataset_id)}.pkl"


def load_cached_raw(root: str | Path, dataset_id: str) -> RawDataset | None:
    """Load a cached :class:`RawDataset`, or ``None`` if absent/unreadable."""
    path = dataset_cache_path(root, dataset_id)
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        # A corrupt/partial cache should not be fatal — drop it and re-fetch.
        path.unlink(missing_ok=True)
        return None


def save_cached_raw(root: str | Path, raw: RawDataset) -> Path:
    """Persist a :class:`RawDataset` to the unified cache and return its path."""
    path = dataset_cache_path(root, raw.dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(raw, path)
    return path
