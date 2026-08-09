"""Per-source fetch-by-id loaders for TabBench-LLM.

Each loader turns a :class:`~tabbench_llm.data.datasets.DatasetSpec` into a
:class:`~tabbench_llm.data.loaders.base.RawDataset`. :func:`get_loader` dispatches on
the spec's ``source`` and forwards per-source options (GEO platform via the ``@GPL``
fetch-id, Kaggle ``data_file``, local ``embedding_column``). Heavy/optional source dependencies (GEOparse, biopython,
kagglehub, openml, requests) are imported lazily inside each loader so importing this
package never forces them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabbench_llm.data.loaders.base import DatasetLoader, RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec


def get_loader(spec: DatasetSpec, *, cache_dir: str | None = None) -> DatasetLoader:
    """Return the loader instance for a given spec (fail-fast on unknown source).

    Parameters
    ----------
    spec : DatasetSpec
        The dataset spec; its ``source`` selects the loader and per-source options
        (``data_file`` for Kaggle, ``embedding_column`` for local) are forwarded.
    cache_dir : str | None
        Bio cache root; the per-source caches (TCGA matrix, GEO SOFT) live under it.
        ``None`` uses each loader's default.
    """
    from pathlib import Path

    source = spec.source
    if source == "geo":
        from tabbench_llm.data.loaders.geo import GeoLoader

        destdir = Path(cache_dir) / "geo_raw" if cache_dir else None
        return GeoLoader(destdir=destdir)
    if source == "tcga":
        from tabbench_llm.data.loaders.tcga import TcgaLoader

        tcga_cache = Path(cache_dir) / "tcga_raw" if cache_dir else None
        return TcgaLoader(cache_dir=tcga_cache)
    if source == "kaggle":
        from tabbench_llm.data.loaders.kaggle import KaggleLoader

        return KaggleLoader(data_file=spec.data_file)
    if source == "openml":
        from tabbench_llm.data.loaders.openml import OpenMLLoader

        return OpenMLLoader()
    if source == "local":
        from tabbench_llm.data.loaders.local import LocalLoader

        return LocalLoader(embedding_column=spec.embedding_column)
    if source == "mgnify":
        from tabbench_llm.data.loaders.mgnify import MgnifyLoader

        return MgnifyLoader(cache_dir=Path(cache_dir) / "mgnify_raw" if cache_dir else None)
    if source == "metagenomics":
        from tabbench_llm.data.loaders.metagenomics import MetagenomicsLoader

        return MetagenomicsLoader(
            cache_dir=Path(cache_dir) / "metagenomics_raw" if cache_dir else None
        )
    if source == "synthetic":
        from tabbench_llm.data.loaders.synthetic import SyntheticLoader

        return SyntheticLoader()
    raise ValueError(f"Unknown bio source: {source!r}")


__all__ = ["DatasetLoader", "RawDataset", "get_loader"]
