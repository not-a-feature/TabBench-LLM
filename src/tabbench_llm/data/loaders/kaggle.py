"""Kaggle loader, fetch-by-``owner/slug``.

``spec.fetch_id`` is the ``owner/slug`` ref. Downloads the dataset via ``kagglehub``
(which authenticates from ``~/.kaggle/access_token`` or ``~/.kaggle/kaggle.json`` and
caches downloads under ``~/.cache/kagglehub``), loads its single tabular file, and frames
it with the **curated** ``spec.target`` (no heuristic target detection). When a dataset
ships more than one table, set ``spec.data_file`` to disambiguate.

Optional dep: ``tabbench-llm[bio]`` (``kagglehub``). Imported lazily inside :meth:`fetch`.
The license is taken from the curated ``spec.license`` (re-confirm the real per-dataset
license on the Kaggle page before publishing — CC-BY-SA implies share-alike on any
re-hosted copy).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec

#: Tabular file extensions we know how to load.
_TABLE_SUFFIXES = (".csv", ".tsv", ".parquet", ".xlsx")


def _read_table(path: Path) -> pd.DataFrame:
    """Load a single Kaggle tabular file into a DataFrame by extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")


class KaggleLoader:
    """Fetch a Kaggle dataset by ``owner/slug`` and frame it with the curated target."""

    def __init__(self, *, data_file: str | None = None) -> None:
        """Initialize the loader.

        Parameters
        ----------
        data_file : str | None
            Relative name of the table to load when the dataset ships more than one
            tabular file. ``None`` requires exactly one table (else error).
        """
        self.data_file = data_file

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Download a Kaggle dataset and return it as a :class:`RawDataset`."""
        import kagglehub

        if spec.target is None:
            raise ValueError(f"{spec.dataset_id}: Kaggle loader needs a curated target.")
        download_dir = Path(kagglehub.dataset_download(spec.fetch_id))
        table_path = self._resolve_table(download_dir, spec)
        df = _read_table(table_path)

        if spec.target not in df.columns:
            raise ValueError(
                f"{spec.dataset_id}: curated target {spec.target!r} not in {table_path.name} "
                f"(columns include: {list(df.columns[:5])}...).",
            )
        y = df[spec.target]
        X = df.drop(columns=[spec.target])
        problem_type = spec.problem_type or (
            "multiclass" if y.nunique(dropna=True) > 2 else "binary"
        )

        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license=spec.license or "unknown (confirm on the Kaggle dataset page)",
            source_url=f"https://www.kaggle.com/datasets/{spec.fetch_id}",
            citation=f"Kaggle dataset {spec.fetch_id}.",
            metadata={
                "kaggle_ref": spec.fetch_id,
                "table_file": table_path.name,
                "n_features": int(X.shape[1]),
                "target": spec.target,
            },
        )

    def _resolve_table(self, download_dir: Path, spec: DatasetSpec) -> Path:
        """Pick the table file: the explicit ``data_file``, else the only known table."""
        if self.data_file is not None:
            path = download_dir / self.data_file
            if not path.exists():
                raise FileNotFoundError(
                    f"{spec.dataset_id}: data_file {self.data_file!r} not found under {download_dir}."
                )
            return path
        tables = sorted(
            p
            for p in download_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _TABLE_SUFFIXES
        )
        if not tables:
            raise ValueError(
                f"{spec.dataset_id}: no tabular file ({_TABLE_SUFFIXES}) under {download_dir}."
            )
        if len(tables) > 1:
            names = ", ".join(p.name for p in tables)
            raise ValueError(
                f"{spec.dataset_id}: {len(tables)} tables found ({names}); set spec.data_file to choose.",
            )
        return tables[0]
