"""MGnify functional-profile loader, fetch-by-study.

``spec.fetch_id`` is an MGnify study accession (e.g. ``"MGYS00005791"``). The loader
downloads the study's aggregated InterPro (IPR) functional-abundance table — the wide
(10k+ feature) matrix — from the MGnify API and orients it samples x features (log1p).

The label is the sample-metadata field named by ``spec.target`` (curated); when
``spec.target`` is unset it auto-picks the first metadata field (skipping date/coordinate
junk) that splits samples into 2..``MAX_CLASSES`` groups of at least ``MIN_CLASS`` samples
each. Classes smaller than ``MIN_CLASS`` are dropped before labelling.

The IPR table maps to sample metadata through two id systems: table columns are
run/assembly accessions, resolved to sample accessions via the study's analyses.

Optional dep: ``tabbench-llm[bio]`` (``requests``). Imported lazily inside the fetch path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_llm.data.cache import default_dataset_cache_dir
from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec

API = "https://www.ebi.ac.uk/metagenomics/api/v1"
MIN_CLASS = 30  # a target class needs at least this many samples
MAX_CLASSES = 10  # skip near-unique fields (coordinates, ids)
SKIP_SUBSTRINGS = ("date", "latitude", "longitude")


class MgnifyLoader:
    """Fetch an MGnify study's IPR functional profile and frame a metadata-field label."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = default_dataset_cache_dir() / "mgnify_raw"
        self.cache_dir = Path(cache_dir)

    def _session(self):
        """A retrying requests Session (EBI's API times out under load)."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        session.headers.update({"Accept": "application/json"})
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(total=4, backoff_factor=1, status_forcelist=(500, 502, 503, 504))
            ),
        )
        return session

    def _ipr_table(self, session, accession: str) -> pd.DataFrame:
        """Download (once) the IPR functional-abundance table, oriented samples x features."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{accession}_IPR_abundances.tsv"
        if not cache_path.exists():
            downloads = session.get(f"{API}/studies/{accession}/downloads", timeout=30).json()
            table_url = None
            for d in downloads["data"]:
                if "IPR_abundances" in d["attributes"]["alias"]:
                    table_url = d["links"]["self"]
                    break
            assert table_url is not None, f"{accession}: no IPR_abundances functional table"
            cache_path.write_text(session.get(table_url, timeout=90).text)

        table = pd.read_csv(cache_path, sep="\t")
        table = table.set_index(table.columns[0]).drop(columns="description", errors="ignore")
        return table.transpose()  # rows = run/assembly accessions, cols = IPR features

    def _sample_of(self, session, accession: str) -> dict[str, str]:
        """Map each run/assembly accession (the table's rows) to its sample accession."""
        sample_of: dict[str, str] = {}
        url = f"{API}/studies/{accession}/analyses?page_size=100"
        while url:
            page = session.get(url, timeout=60).json()
            for analysis in page["data"]:
                rel = analysis["relationships"]
                sample = rel["sample"]["data"]
                if sample is None:
                    continue
                for kind in ("run", "assembly"):
                    ref = rel[kind]["data"]
                    if ref is not None:
                        sample_of[ref["id"]] = sample["id"]
            url = page["links"]["next"]
        return sample_of

    def _metadata_of(self, session, accession: str) -> dict[str, dict[str, str]]:
        """Sample accession -> {metadata field name: value} for every sample in the study."""
        meta_of: dict[str, dict[str, str]] = {}
        url = f"{API}/studies/{accession}/samples?page_size=100"
        while url:
            page = session.get(url, timeout=60).json()
            for s in page["data"]:
                attrs = s["attributes"]
                meta_of[attrs["accession"]] = {
                    m["key"]: m["value"] for m in attrs["sample-metadata"]
                }
            url = page["links"]["next"]
        return meta_of

    def _labels_for_field(self, row_ids, sample_of, meta_of, field) -> dict[str, str]:
        """Per-row label from ``field``, skipping rows that don't map to a value."""
        labels: dict[str, str] = {}
        for row_id in row_ids:
            if row_id not in sample_of:
                continue
            sample = sample_of[row_id]
            if sample not in meta_of or field not in meta_of[sample]:
                continue
            value = meta_of[sample][field]
            if value in (None, ""):
                continue
            labels[row_id] = str(value).strip().lower()
        return labels

    def _pick_field(self, row_ids, sample_of, meta_of, all_fields):
        """Auto-pick the first metadata field yielding 2..MAX_CLASSES big-enough classes."""
        for field in all_fields:
            if any(bad in field.lower() for bad in SKIP_SUBSTRINGS):
                continue
            labels = self._labels_for_field(row_ids, sample_of, meta_of, field)
            big = [g for g, n in Counter(labels.values()).items() if n >= MIN_CLASS]
            if 2 <= len(big) <= MAX_CLASSES:
                return field, labels, big
        return None, None, None

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Assemble the IPR functional matrix + the curated (or auto-picked) metadata label."""
        session = self._session()
        accession = spec.fetch_id

        table = self._ipr_table(session, accession)
        sample_of = self._sample_of(session, accession)
        meta_of = self._metadata_of(session, accession)
        all_fields = sorted({f for fields in meta_of.values() for f in fields})

        if spec.target is not None:
            field = spec.target
            labels = self._labels_for_field(table.index, sample_of, meta_of, field)
            big = [g for g, n in Counter(labels.values()).items() if n >= MIN_CLASS]
            assert 2 <= len(big) <= MAX_CLASSES, (
                f"{spec.dataset_id}: target {field!r} yields {len(big)} usable classes "
                f"(need 2..{MAX_CLASSES}, each with >={MIN_CLASS} samples); fields: {all_fields}"
            )
        else:
            field, labels, big = self._pick_field(table.index, sample_of, meta_of, all_fields)
            assert field is not None, (
                f"{spec.dataset_id}: no metadata field splits samples into 2..{MAX_CLASSES} "
                f"classes of >={MIN_CLASS}; fields: {all_fields}"
            )

        labels = {row: v for row, v in labels.items() if v in big}
        rows = list(labels)
        X = np.log1p(table.loc[rows]).reset_index(drop=True)
        y = pd.Series([labels[r] for r in rows], name="target")

        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=spec.problem_type or ("binary" if len(big) == 2 else "multiclass"),
            license=spec.license or "public-domain (MGnify / EMBL-EBI)",
            source_url=f"https://www.ebi.ac.uk/metagenomics/studies/{accession}",
            citation=f"MGnify study {accession} (EMBL-EBI MGnify metagenomics resource)",
            metadata={
                "mgnify_accession": accession,
                "target_field": field,
                "n_classes": len(big),
                "n_features": int(X.shape[1]),
            },
        )
