"""TCGA loader (GDC open-access), fetch-by-project/data-category.

``spec.fetch_id`` is ``"<project>/<data_category>"`` (e.g.
``"TCGA-BRCA/Gene-Expression-Quantification"``). The matrix is fetched from the GDC
REST API (no credentials, open-access tier only) and framed with the curated target.

For ``Gene-Expression-Quantification`` the features are per-gene ``tpm_unstranded``
values from the STAR-Counts workflow (GENCODE v36), one row per sample (aliquot file),
one column per gene. The curated target is sample type collapsed to **tumor vs normal**
(binary): ``Solid Tissue Normal`` (and other ``* Normal`` types) -> ``"Normal"``,
everything else (Primary Tumor, Metastatic, ...) -> ``"Tumor"``.

The full parsed matrix is disk-cached (``<cache_dir>/<dataset_id>.pkl``) so rebuilds skip
the multi-GB GDC download.

Optional dep: ``tabbench-llm[bio]`` (``requests``). Imported lazily inside :meth:`fetch`.
Provenance: GDC open-access (NIH Genomic Data Sharing Policy); acknowledge TCGA / GDC.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabbench_llm.data.cache import default_dataset_cache_dir
from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    import requests

    from tabbench_llm.data.datasets import DatasetSpec

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

#: Per-``data_category`` GDC filter + parse config, keyed by the label used in
#: ``DatasetSpec.fetch_id`` (``"<project>/<data_category>"``).
_CATEGORY_CONFIG: dict[str, dict] = {
    "Gene-Expression-Quantification": {
        "data_category": "Transcriptome Profiling",
        "data_type": "Gene Expression Quantification",
        "workflow_type": "STAR - Counts",
        "value_column": "tpm_unstranded",
        "index_column": "gene_id",
    },
}

#: STAR-Counts alignment-summary rows (not genes) — dropped from the feature matrix.
_SUMMARY_ROWS = frozenset({"N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"})

#: GDC ``sample_type`` values that map to the "Normal" class; all others -> "Tumor".
_NORMAL_SAMPLE_TYPES = frozenset(
    {"Solid Tissue Normal", "Blood Derived Normal", "Bone Marrow Normal", "Buccal Cell Normal"},
)


def sample_type_to_binary(sample_type: str) -> str:
    """Collapse a GDC ``sample_type`` to the binary tumor/normal target."""
    return "Normal" if sample_type in _NORMAL_SAMPLE_TYPES else "Tumor"


def parse_star_counts(text: str, *, value_column: str, index_column: str) -> pd.Series:
    """Parse one STAR-Counts expression file into a ``gene_id -> value`` Series.

    Drops the leading ``# gene-model`` comment line (via ``comment="#"``) and the four
    ``N_*`` alignment-summary rows, keeping only per-gene rows.
    """
    df = pd.read_csv(io.StringIO(text), sep="\t", comment="#")
    df = df[~df[index_column].isin(_SUMMARY_ROWS)]
    return df.set_index(index_column)[value_column].astype("float32")


class TcgaLoader:
    """Fetch a TCGA project's open-access matrix and frame it with the curated target."""

    def __init__(
        self,
        *,
        max_samples: int | None = None,
        request_timeout: int = 300,
        batch_size: int = 50,
        max_retries: int = 5,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the loader.

        Parameters
        ----------
        max_samples : int | None
            If set, only fetch the first ``max_samples`` files (for fast smoke tests).
            ``None`` fetches the project's full open-access matrix.
        request_timeout : int
            Per-request timeout (seconds) for GDC calls.
        batch_size : int
            Number of files per bulk ``/data`` download request. Smaller is more robust
            to GDC dropping large transfers mid-stream.
        max_retries : int
            Retry attempts (exponential backoff) per GDC request.
        cache_dir : str | pathlib.Path | None
            Directory for the cached full matrix. Defaults to
            ``<bio-cache>/tcga_raw``. Only full fetches (``max_samples=None``) use it.
        """
        self.max_samples = max_samples
        self.request_timeout = request_timeout
        self.batch_size = batch_size
        self.max_retries = max_retries
        if cache_dir is None:
            cache_dir = default_dataset_cache_dir() / "tcga_raw"
        self.cache_dir = Path(cache_dir)

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Download a TCGA project's open-access matrix and return it as a RawDataset."""
        project, category = self._parse_fetch_id(spec)
        cfg = _CATEGORY_CONFIG.get(category)
        if cfg is None:
            raise ValueError(
                f"{spec.dataset_id}: unsupported TCGA data category {category!r}. Supported: {sorted(_CATEGORY_CONFIG)}.",
            )

        cached = self._load_cache(spec) if self.max_samples is None else None
        if cached is not None:
            X, sample_type_by_fid = cached
            print(
                f"[tcga] {spec.dataset_id}: loaded cached matrix X={X.shape} (skip GDC download)."
            )
        else:
            import requests

            session = requests.Session()
            sample_type_by_fid = self._query_files(session, project=project, cfg=cfg)
            if not sample_type_by_fid:
                raise ValueError(
                    f"{spec.dataset_id}: GDC returned no open-access files for {project}/{category}."
                )
            if self.max_samples is not None:
                sample_type_by_fid = dict(list(sample_type_by_fid.items())[: self.max_samples])
            X = self._download_matrix(session, file_ids=list(sample_type_by_fid), cfg=cfg)
            if self.max_samples is None:
                self._save_cache(spec, X, sample_type_by_fid)

        y = pd.Series(
            {fid: sample_type_to_binary(sample_type_by_fid[fid]) for fid in X.index},
            name=spec.target or "sample_type",
        )

        problem_type = spec.problem_type or ("binary" if y.nunique() <= 2 else "multiclass")
        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license=spec.license or "NIH Genomic Data Sharing Policy (GDC open-access)",
            source_url=f"https://portal.gdc.cancer.gov/projects/{project}",
            citation=f"The Cancer Genome Atlas (TCGA), GDC project {project} ({category}).",
            metadata={
                "gdc_project": project,
                "data_category": category,
                "workflow_type": cfg["workflow_type"],
                "value_column": cfg["value_column"],
                "n_genes": int(X.shape[1]),
                "sample_type_counts": pd.Series(sample_type_by_fid).value_counts().to_dict(),
            },
        )

    def _post_with_retries(
        self, session: requests.Session, url: str, **kwargs
    ) -> requests.Response:
        """POST to a GDC endpoint, retrying transient connection/timeout errors with backoff."""
        import requests

        transient = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = session.post(url, timeout=self.request_timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except transient as exc:
                last_exc = exc
                wait = 2**attempt
                print(
                    f"[tcga] {type(exc).__name__} on {url} (attempt {attempt + 1}/{self.max_retries}); retry in {wait}s"
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _parse_fetch_id(spec: DatasetSpec) -> tuple[str, str]:
        """Split ``spec.fetch_id`` into ``(project, data_category)``."""
        project, _, category = spec.fetch_id.partition("/")
        if not project or not category:
            raise ValueError(
                f"{spec.dataset_id}: TCGA fetch_id must be '<project>/<data_category>', got {spec.fetch_id!r}.",
            )
        return project, category

    def _cache_path(self, spec: DatasetSpec) -> Path:
        """Local cache file for a dataset's full (untruncated) matrix + sample types."""
        safe = spec.dataset_id.replace("/", "_")
        return self.cache_dir / f"{safe}.pkl"

    def _load_cache(self, spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, str]] | None:
        """Load the cached ``(X, sample_type_by_fid)`` for a dataset, or ``None`` if absent."""
        path = self._cache_path(spec)
        if not path.exists():
            return None
        blob = pd.read_pickle(path)
        return blob["X"], blob["sample_type_by_fid"]

    def _save_cache(
        self, spec: DatasetSpec, X: pd.DataFrame, sample_type_by_fid: dict[str, str]
    ) -> None:
        """Persist the full matrix + sample types so later (re)builds skip the GDC download."""
        path = self._cache_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"X": X, "sample_type_by_fid": sample_type_by_fid}, path)
        print(f"[tcga] {spec.dataset_id}: cached matrix -> {path}")

    def _query_files(self, session: requests.Session, *, project: str, cfg: dict) -> dict[str, str]:
        """Return an ordered ``file_id -> sample_type`` map for the project's open files."""
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
                {
                    "op": "in",
                    "content": {"field": "data_category", "value": [cfg["data_category"]]},
                },
                {"op": "in", "content": {"field": "data_type", "value": [cfg["data_type"]]}},
                {
                    "op": "in",
                    "content": {"field": "analysis.workflow_type", "value": [cfg["workflow_type"]]},
                },
                {"op": "in", "content": {"field": "access", "value": ["open"]}},
            ],
        }
        page_size = 1000
        sample_type_by_fid: dict[str, str] = {}
        from_ = 0
        while True:
            params = {
                "filters": json.dumps(filters),
                "fields": "file_id,cases.samples.sample_type",
                "format": "JSON",
                "size": str(page_size),
                "from": str(from_),
            }
            resp = self._post_with_retries(session, GDC_FILES_ENDPOINT, json=params)
            data = resp.json()["data"]
            for hit in data["hits"]:
                samples = hit.get("cases", [{}])[0].get("samples", [{}])
                sample_type = samples[0].get("sample_type")
                if sample_type is not None:
                    sample_type_by_fid[hit["file_id"]] = sample_type
            pagination = data["pagination"]
            from_ += page_size
            if from_ >= int(pagination["total"]):
                break
        return sample_type_by_fid

    def _download_matrix(
        self, session: requests.Session, *, file_ids: list[str], cfg: dict
    ) -> pd.DataFrame:
        """Download + parse all files into a ``samples (file_id) x genes`` float matrix."""
        series_by_fid: dict[str, pd.Series] = {}
        for start in range(0, len(file_ids), self.batch_size):
            batch = file_ids[start : start + self.batch_size]
            series_by_fid.update(self._download_batch(session, file_ids=batch, cfg=cfg))
        # from_dict(orient="index") aligns columns by gene_id; all STAR-Counts files
        # share the same GENCODE v36 gene set, so the union equals each file's index.
        # Cast back to float32 (the per-file dtype): aligning Series upcasts to float64.
        return pd.DataFrame.from_dict(series_by_fid, orient="index").astype("float32")

    def _download_batch(
        self, session: requests.Session, *, file_ids: list[str], cfg: dict
    ) -> dict[str, pd.Series]:
        """Bulk-download one batch of files via ``/data`` and parse each to a gene Series.

        A multi-file request returns a gzipped tarball (each member at
        ``<file_id>/<file_name>``); a single-file request returns the raw file text.
        """
        resp = self._post_with_retries(
            session,
            GDC_DATA_ENDPOINT,
            json={"ids": file_ids},
            headers={"Content-Type": "application/json"},
        )
        content = resp.content

        parse_kwargs = {"value_column": cfg["value_column"], "index_column": cfg["index_column"]}
        # gzip magic -> tarball of many files; otherwise a single raw file.
        if content[:2] == b"\x1f\x8b":
            out: dict[str, pd.Series] = {}
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or member.name.endswith("MANIFEST.txt"):
                        continue
                    fid = member.name.split("/")[0]
                    text = tar.extractfile(member).read().decode("utf-8")
                    out[fid] = parse_star_counts(text, **parse_kwargs)
            return out
        return {file_ids[0]: parse_star_counts(content.decode("utf-8"), **parse_kwargs)}
