"""Base types for the fetch-by-id bio loader layer.

Adapted from the ``tabular-benchmark-pipeline`` prototype, stripped of the discovery
machinery: TabBench-LLM fetches a *fixed* set of dataset ids, so we only need
``fetch(spec) -> RawDataset`` per source, not candidate scraping.

A :class:`RawDataset` is the source-agnostic, fetched-but-not-yet-split form of a
dataset. It is later turned into a :class:`~tabbench_llm.dataset.Dataset` (see
:mod:`tabbench_llm.data.adapter`) so the standard pipeline can consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from tabbench_llm.data.datasets import DatasetSpec


@dataclass(frozen=True)
class RawDataset:
    """A fetched-but-not-yet-split dataset, as returned by a loader.

    Attributes
    ----------
    dataset_id : str
        The TabBench-LLM id this was fetched for.
    X : pandas.DataFrame
        Feature matrix (samples x features). Bio feature matrices are numeric
        (gene-expression / spectral-like); the adapter coerces to float32.
    y : pandas.Series
        Target series aligned to ``X`` (length n_samples).
    problem_type : str
        ``"binary" | "multiclass" | "regression"``.
    license : str
        License string captured from the source (provenance).
    source_url : str
        Canonical URL of the original record (provenance).
    citation : str
        Recommended citation / accession (provenance).
    metadata : dict
        Any extra source metadata (platform, organism, transform, ...).
    """

    dataset_id: str
    X: pd.DataFrame
    y: pd.Series
    problem_type: str
    license: str
    source_url: str
    citation: str
    metadata: dict


class DatasetLoader(Protocol):
    """Protocol every per-source loader implements."""

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Download the dataset identified by ``spec`` and return it as a RawDataset.

        Implementations must honor ``spec.target`` (the curated target column /
        strategy) and ``spec.problem_type`` rather than re-deriving them heuristically.
        """
        ...
