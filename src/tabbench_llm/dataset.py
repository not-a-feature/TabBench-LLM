"""Native dataset container and task-type enum for TabBench-LLM.

TabBench-LLM is a self-contained benchmark: a dataset is simply a wide numeric
feature matrix (genes / probes / measured channels as columns, samples as rows)
plus a target vector.  :class:`Dataset` is the lingua franca that every stage of
the pipeline consumes — the bio loaders produce one (via
:func:`tabbench_llm.data.adapter.raw_to_dataset`) and the benchmark turns it
into train/test ``DataFrame`` splits.

The label column produced by :meth:`Dataset.to_dataframe` is always named
``"target"`` so the rest of the pipeline (AutoGluon, metrics, prediction CSVs)
can rely on a fixed name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

#: Fixed name of the label column in every :meth:`Dataset.to_dataframe` result.
TARGET_COLUMN = "target"


class TaskType(IntEnum):
    """The kind of supervised task a dataset defines."""

    Classification = 0
    Regression = 1

    @classmethod
    def from_problem_type(cls, problem_type: str) -> TaskType:
        """Map an AutoGluon-style ``problem_type`` to a :class:`TaskType`."""
        return cls.Classification if problem_type in ("binary", "multiclass") else cls.Regression


@dataclass
class DatasetInfo:
    """Lightweight metadata describing a dataset."""

    id: str
    name: str
    task_type: TaskType
    metadata: dict = field(default_factory=dict)


@dataclass
class Dataset:
    """A wide feature matrix with one or more targets.

    Parameters
    ----------
    features : pd.DataFrame
        ``(n_samples, n_features)`` feature table with per-column dtypes preserved
        (numeric columns numeric, categorical/string columns kept as-is).
    targets : np.ndarray
        ``(n_samples,)`` for a single target or ``(n_samples, n_targets)`` for
        several.
    feature_names : list[str]
        Column names for ``features`` (e.g. gene / probe identifiers).  Real
        identifiers are preserved so downstream artifacts stay interpretable.
    target_names : list[str]
        Human-readable names, one per target column.
    info : DatasetInfo
        Dataset-level metadata.
    """

    features: pd.DataFrame
    targets: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    info: DatasetInfo

    @property
    def n_targets(self) -> int:
        """Number of target columns (``0`` when there are no targets)."""
        if self.targets is None:
            return 0
        return 1 if self.targets.ndim == 1 else self.targets.shape[1]

    def to_dataframe(self, target_idx: int = 0) -> pd.DataFrame:
        """Return a ``DataFrame`` of features with target *target_idx* appended.

        The target is always the last column and is named :data:`TARGET_COLUMN`.
        """
        df = self.features.copy()
        df.columns = [str(c) for c in self.feature_names]
        if self.targets.ndim == 1:
            df[TARGET_COLUMN] = self.targets
        else:
            df[TARGET_COLUMN] = self.targets[:, target_idx]
        return df
