"""Local-file loader, fetch-by-path.

For datasets that are not fetched from a remote source but ship as a curated file in
the repo (or live anywhere on disk). ``spec.fetch_id`` is a path to a tabular file
(``.csv`` / ``.tsv`` / ``.parquet``): an absolute path is used as-is, a relative path is
resolved against the local-data dir — ``$TABBENCH_LLM_LOCAL_DIR`` or the bundled
``data/registry/local/`` shipped with the package.

The file is framed with the **curated** ``spec.target`` (no heuristic target detection):
that column is the label, every other column is a feature. When ``spec.embedding_column``
is set, that single column is expected to hold a per-row embedding as one comma-separated
string of floats (e.g. a DNA/protein language-model embedding) and is exploded into one
numeric feature column per dimension (``"<embedding_column>_<i>"``).

No optional deps: only ``pandas`` / ``numpy`` (already required). Provenance (license,
citation, url) comes from the curated ``spec`` since there is no remote record.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec

#: Env var to override the local-data root (where relative ``fetch_id`` paths resolve).
_LOCAL_ENV = "TABBENCH_LLM_LOCAL_DIR"

#: Bundled local-data dir, shipped as package data alongside the registry JSON.
_BUNDLED_LOCAL = Path(__file__).parents[1] / "registry" / "local"

#: Tabular file extensions we know how to load.
_TABLE_SUFFIXES = (".csv", ".tsv", ".parquet")


def _local_root() -> Path:
    """Root for relative ``fetch_id`` paths ($TABBENCH_LLM_LOCAL_DIR or the bundled dir)."""
    override = os.environ.get(_LOCAL_ENV)
    return Path(override) if override else _BUNDLED_LOCAL


def _resolve_path(fetch_id: str) -> Path:
    """Resolve ``fetch_id`` to a file path (absolute as-is, relative under the local root)."""
    path = Path(fetch_id)
    return path if path.is_absolute() else _local_root() / fetch_id


def _read_table(path: Path) -> pd.DataFrame:
    """Load a tabular file by extension; strip stray whitespace from column names."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # skipinitialspace handles a space after the delimiter; strip below covers the rest.
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _explode_embedding(series: pd.Series, column: str, dataset_id: str) -> pd.DataFrame:
    """Explode a column of comma-separated float strings into a wide numeric DataFrame.

    Each row's string is parsed into a float vector; all rows must share one width.
    Columns are named ``"<column>_<i>"`` and the original row index is preserved.
    """
    if series.isna().any():
        raise ValueError(
            f"{dataset_id}: embedding column {column!r} has {int(series.isna().sum())} missing value(s)."
        )
    vectors = [np.fromstring(str(s), sep=",", dtype=np.float32) for s in series]
    widths = {v.shape[0] for v in vectors}
    if widths == {0}:
        raise ValueError(
            f"{dataset_id}: embedding column {column!r} parsed to empty vectors (wrong separator?)."
        )
    if len(widths) != 1:
        raise ValueError(
            f"{dataset_id}: embedding column {column!r} has ragged widths {sorted(widths)}; expected one."
        )
    matrix = np.vstack(vectors)
    cols = [f"{column}_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=cols, index=series.index)


class LocalLoader:
    """Load a curated tabular file from disk and frame it with the curated target."""

    def __init__(self, *, embedding_column: str | None = None) -> None:
        """Initialize the loader.

        Parameters
        ----------
        embedding_column : str | None
            Name of a column holding a per-row comma-separated embedding string to
            explode into numeric features. ``None`` treats every non-target column as a
            plain feature.
        """
        self.embedding_column = embedding_column

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Load the local file and return it as a :class:`RawDataset`."""
        if spec.target is None:
            raise ValueError(f"{spec.dataset_id}: local loader needs a curated target.")
        path = _resolve_path(spec.fetch_id)
        if not path.exists():
            raise FileNotFoundError(
                f"{spec.dataset_id}: local file {path} not found (fetch_id={spec.fetch_id!r})."
            )
        if path.suffix.lower() not in _TABLE_SUFFIXES:
            raise ValueError(
                f"{spec.dataset_id}: unsupported local file type {path.suffix!r} (expected {_TABLE_SUFFIXES})."
            )

        df = _read_table(path)
        if spec.target not in df.columns:
            raise ValueError(
                f"{spec.dataset_id}: curated target {spec.target!r} not in {path.name} "
                f"(columns: {list(df.columns[:8])}...).",
            )
        y = df[spec.target]
        features = df.drop(columns=[spec.target])

        if self.embedding_column is not None:
            if self.embedding_column not in features.columns:
                raise ValueError(
                    f"{spec.dataset_id}: embedding_column {self.embedding_column!r} not in {path.name} "
                    f"(columns: {list(features.columns[:8])}...).",
                )
            exploded = _explode_embedding(
                features[self.embedding_column], self.embedding_column, spec.dataset_id
            )
            others = features.drop(columns=[self.embedding_column]).reset_index(drop=True)
            X = (
                exploded.reset_index(drop=True)
                if others.empty
                else pd.concat([others, exploded.reset_index(drop=True)], axis=1)
            )
            y = y.reset_index(drop=True)
        else:
            X = features

        problem_type = spec.problem_type or (
            "multiclass" if y.nunique(dropna=True) > 2 else "binary"
        )

        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license=spec.license or "unknown (local file; set spec.license)",
            source_url=str(path),
            citation=f"Local dataset {spec.dataset_id} ({path.name}).",
            metadata={
                "local_path": str(path),
                "table_file": path.name,
                "embedding_column": self.embedding_column,
                "n_features": int(X.shape[1]),
                "target": spec.target,
            },
        )
