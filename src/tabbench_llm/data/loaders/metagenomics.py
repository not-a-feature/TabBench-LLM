"""Human-gut metagenomics loader (MetAML marker table), fetch-by-disease.

``spec.fetch_id`` is a disease code from Pasolli's MetAML marker matrix (e.g.
``"cirrhosis"``). The loader downloads the marker-presence table once (cached under
``<bio-cache>/metagenomics_raw``), keeps markers present in at least ``MIN_PREVALENCE``
of samples, and frames a **binary** task: all healthy subjects vs. that disease's
patients. Features are 0/1 microbe-marker presence calls (~34k markers).

Optional dep: ``tabbench-llm[bio]`` (``requests``). Imported lazily inside the fetch path.
"""

from __future__ import annotations

import bz2
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_llm.data.cache import default_dataset_cache_dir
from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec

MARKER_URL = (
    "https://raw.githubusercontent.com/segatalab/metaml/master/data/marker_presence.txt.bz2"
)
MIN_PREVALENCE = 0.10  # drop markers present in <10% of samples
HEALTHY = frozenset({"n", "nd", "n_relative"})  # disease codes counted as healthy


class MetagenomicsLoader:
    """Fetch the MetAML gut-marker matrix and frame healthy-vs-<disease> binary tasks."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = default_dataset_cache_dir() / "metagenomics_raw"
        self.cache_dir = Path(cache_dir)

    def _marker_matrix(self) -> tuple[pd.DataFrame, pd.Series]:
        """Download (once) and parse the marker table into X + a per-sample disease label."""
        import requests

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw = self.cache_dir / "marker_presence.txt.bz2"
        if not raw.exists():
            resp = requests.get(MARKER_URL, timeout=180)
            resp.raise_for_status()
            raw.write_bytes(resp.content)

        disease, markers, names, n_samples = None, [], [], None
        with bz2.open(raw, "rt") as fh:
            for line in fh:
                name, _, rest = line.partition("\t")
                values = rest.rstrip("\n").split("\t")
                if n_samples is None:
                    n_samples = len(values)
                if name == "disease":
                    disease = values
                    continue
                ones = values.count("1")
                if ones + values.count("0") != n_samples:
                    continue  # a text-metadata row, not a 0/1 marker
                if ones >= MIN_PREVALENCE * n_samples:
                    markers.append(np.array(values, dtype=np.int8))
                    names.append(name)

        X = pd.DataFrame(np.array(markers).T, columns=names)  # rows = samples, cols = markers
        disease = pd.Series([d.strip() for d in disease], name="disease")
        return X, disease

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Assemble the healthy-vs-<disease> binary task named by ``spec.fetch_id``."""
        X, disease = self._marker_matrix()
        target_disease = spec.fetch_id.strip().lower()

        available = set(disease) - HEALTHY
        assert (
            target_disease in available
        ), f"{spec.dataset_id}: disease {target_disease!r} not in MetAML table; available: {sorted(available)}"

        keep = (disease.isin(HEALTHY) | (disease == target_disease)).to_numpy()
        X = X[keep].reset_index(drop=True)
        y = pd.Series(
            ["healthy" if d in HEALTHY else "disease" for d in disease[keep]],
            name="target",
        )

        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=spec.problem_type or "binary",
            license=spec.license or "CC-BY-4.0 (Pasolli et al. 2016, MetAML)",
            source_url="https://github.com/segatalab/metaml",
            citation=(
                "Pasolli E, et al. (2016) Machine Learning Meta-analysis of Large "
                "Metagenomic Datasets: Tools and Biological Insights. PLoS Comput Biol 12(7)."
            ),
            metadata={
                "disease": target_disease,
                "n_markers": int(X.shape[1]),
                "min_prevalence": MIN_PREVALENCE,
            },
        )
