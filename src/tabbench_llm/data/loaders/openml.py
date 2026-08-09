"""OpenML loader, fetch-by-dataset-id.

``spec.fetch_id`` is the OpenML dataset id. The target column is the curated
``spec.target`` when set, otherwise the OpenML ``default_target_attribute``. The
per-dataset ``licence`` / citation / url are captured for provenance.

OpenML caches downloaded datasets under its own cache directory (``~/.cache/openml``
by default), so repeated fetches do not re-download.

Optional dep: ``tabbench-llm[bio]`` (``openml``). Imported lazily inside :meth:`fetch`
so importing this package never forces it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec


def _infer_problem_type(y: pd.Series) -> str:
    """Fallback problem-type inference when a spec leaves ``problem_type`` unset.

    Curated specs should set ``problem_type`` explicitly; this only guards the
    ad-hoc path. Non-numeric or categorical targets are treated as classification
    (binary vs multiclass by class count), numeric otherwise as regression.
    """
    if isinstance(y.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(y):
        return "binary" if y.nunique(dropna=True) <= 2 else "multiclass"
    return "regression"


class OpenMLLoader:
    """Fetch an OpenML dataset by id and frame it with its (curated) target."""

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Download an OpenML dataset and return it as a :class:`RawDataset`."""
        import openml

        dataset = openml.datasets.get_dataset(
            int(spec.fetch_id),
            download_data=True,
            download_qualities=False,
            download_features_meta_data=True,
        )

        target = spec.target or dataset.default_target_attribute
        if not target:
            raise ValueError(
                f"{spec.dataset_id}: no curated target and OpenML dataset {spec.fetch_id} "
                "has no default_target_attribute; set spec.target.",
            )

        X, y, _, _ = dataset.get_data(target=target, dataset_format="dataframe")
        if y is None:
            raise ValueError(
                f"{spec.dataset_id}: target {target!r} not found in OpenML dataset {spec.fetch_id}."
            )

        problem_type = spec.problem_type or _infer_problem_type(y)

        url = dataset.openml_url or dataset.url or ""
        citation = dataset.citation or f"OpenML dataset {spec.fetch_id} ({dataset.name})"
        license_str = spec.license or dataset.licence or "unknown"

        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license=license_str,
            source_url=url,
            citation=citation,
            metadata={
                "openml_dataset_id": int(spec.fetch_id),
                "openml_name": dataset.name,
                "openml_version": dataset.version,
                "target": target,
            },
        )
