"""GEO loader, fetch-by-accession (credential-free for fixed accessions).

``spec.fetch_id`` is the GEO series accession (e.g. ``"GSE10893"``). The series SOFT
file is downloaded with ``GEOparse`` (public NCBI GEO download — the ``ENTREZ`` API key
is only needed for *discovery*, which we don't do here). The expression matrix is built
samples x probes from each sample's ``VALUE`` column, and the **curated** target is the
GEO sample-characteristic *key* named by ``spec.target`` (e.g. ``"subtype"``); samples
lacking that characteristic are dropped.

Multi-platform series (several ``GPL``s with different probe sets) must pick one platform
via the registry ``platform`` field (``GeoLoader(platform="GPLxxxx")`` or the
``"<accession>@<GPL>"`` fetch-id form) — otherwise probes don't align across samples.

The downloaded SOFT file is cached in ``destdir`` so rebuilds skip the download.

Optional dep: ``tabbench-llm[bio]`` (``GEOparse``, ``biopython``). Imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabbench_llm.data.cache import default_dataset_cache_dir
from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec


def build_expression_matrix(gsms: dict, *, value_column: str = "VALUE") -> pd.DataFrame:
    """Assemble a ``samples x probes`` numeric matrix from GEO sample tables.

    Each GSM contributes a row, indexed by its ``ID_REF`` probe ids → ``value_column``.
    Duplicate probe ids keep the first; values are coerced to numeric. Columns align by
    probe id across samples (use a single platform so the probe sets match).
    """
    series: dict[str, pd.Series] = {}
    for gsm_id, gsm in gsms.items():
        table = gsm.table
        if "ID_REF" not in table.columns or value_column not in table.columns:
            continue
        s = table.set_index("ID_REF")[value_column]
        s = s[~s.index.duplicated(keep="first")]
        series[gsm_id] = pd.to_numeric(s, errors="coerce")
    if not series:
        raise ValueError(f"No GEO samples had an 'ID_REF'/{value_column!r} table.")
    return pd.DataFrame(series).T


def extract_target(gsms: dict, *, characteristic_key: str) -> pd.Series:
    """Per-sample value of the ``characteristics_ch1`` entry whose key == ``characteristic_key``.

    GEO characteristics are ``"key: value"`` strings; matching is case-insensitive on the
    key. Samples without that characteristic are omitted from the result.
    """
    key = characteristic_key.lower()
    out: dict[str, str] = {}
    for gsm_id, gsm in gsms.items():
        for ch in gsm.metadata.get("characteristics_ch1", []):
            if ":" in ch and ch.split(":", 1)[0].strip().lower() == key:
                out[gsm_id] = ch.split(":", 1)[1].strip()
                break
    return pd.Series(out, name=characteristic_key)


class GeoLoader:
    """Fetch a GEO series by accession and frame it with the curated target."""

    def __init__(
        self,
        *,
        platform: str | None = None,
        value_column: str = "VALUE",
        destdir: str | Path | None = None,
    ) -> None:
        """Initialize the loader.

        Parameters
        ----------
        platform : str | None
            Restrict to one ``GPL`` platform id (required for multi-platform series so
            probe sets align). ``None`` uses all samples (single-platform).
        value_column : str
            Per-sample table column to use as the expression value.
        destdir : str | pathlib.Path | None
            Download dir for the SOFT file. Defaults to ``<bio-cache>/geo_raw``.
        """
        self.platform = platform
        self.value_column = value_column
        if destdir is None:
            destdir = default_dataset_cache_dir() / "geo_raw"
        self.destdir = Path(destdir)

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Download a GEO series and return it as a :class:`RawDataset`."""
        import GEOparse

        if spec.target is None:
            raise ValueError(
                f"{spec.dataset_id}: GEO loader needs a curated target (a sample-characteristic key)."
            )
        # fetch_id is "<accession>" or "<accession>@<GPL platform>" (platform wins over
        # the constructor default, so the registry can pin the platform per dataset).
        accession, _, fetch_platform = spec.fetch_id.partition("@")
        platform = fetch_platform or self.platform

        self.destdir.mkdir(parents=True, exist_ok=True)
        gse = GEOparse.get_GEO(geo=accession, destdir=str(self.destdir), silent=True)
        gsms = self._select_samples(gse, spec, platform=platform)

        X = build_expression_matrix(gsms, value_column=self.value_column)
        y = extract_target(gsms, characteristic_key=spec.target)
        common = X.index.intersection(y.index)
        if len(common) == 0:
            raise ValueError(f"{spec.dataset_id}: no samples have characteristic {spec.target!r}.")
        X = X.loc[common].dropna(axis=1, how="all")
        y = y.loc[common]

        problem_type = spec.problem_type or ("multiclass" if y.nunique() > 2 else "binary")
        platforms = platform or ",".join(gse.gpls)
        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license=spec.license
            or "GEO (NCBI public-domain US-gov data; per-record terms may apply)",
            source_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={spec.fetch_id}",
            citation=f"NCBI GEO series {spec.fetch_id}: {gse.metadata.get('title', [''])[0]}",
            metadata={
                "geo_accession": spec.fetch_id,
                "platform": platforms,
                "value_column": self.value_column,
                "n_probes": int(X.shape[1]),
                "target_characteristic": spec.target,
            },
        )

    def _select_samples(self, gse, spec: DatasetSpec, *, platform: str | None) -> dict:
        """Return the GSM dict, restricted to ``platform`` for multi-platform series."""
        if platform is not None:
            gsms = {
                gid: g
                for gid, g in gse.gsms.items()
                if g.metadata.get("platform_id", [None])[0] == platform
            }
            if not gsms:
                raise ValueError(f"{spec.dataset_id}: no samples on platform {platform!r}.")
            return gsms
        if len(gse.gpls) > 1:
            raise ValueError(
                f"{spec.dataset_id}: multi-platform series ({list(gse.gpls)}); "
                f"pin one via fetch_id '<accession>@GPLxxxx' or the platform field.",
            )
        return gse.gsms
