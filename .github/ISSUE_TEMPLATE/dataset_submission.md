---
name: Dataset submission
about: Propose a new dataset for TabBench-LLM
labels: dataset
---

**dataset_id**
A short, unique, stable identifier (e.g. `TCGA-TCGA-LUAD_Gene-Expression-Quantification`).

**Source**
- [ ] GEO
- [ ] TCGA
- [ ] Kaggle
- [ ] OpenML

**Task type**
- [ ] Classification (binary / multiclass)
- [ ] Regression

**Dataset summary**

| Field | Value |
|---|---|
| Samples (rows) | |
| Features (genes/probes/columns) | |
| Target | |
| Number of classes (if classification) | |
| Modality (RNA-seq / methylation / …) | |

**Fetch details**
- `fetch_id` (accession / dataset id / OpenML id):
- `target` (column or characteristic to predict):
- `data_file` (Kaggle only, if multiple files):

**License & provenance**
- License:
- Source URL:
- Citation (paper / preprint DOI):

**Additional context**
Any other information about the dataset (preprocessing, class balance, known caveats).

---
Adding a dataset is usually a one-line entry in
[`src/tabbench_llm/data/registry/datasets.json`](../../src/tabbench_llm/data/registry/datasets.json).
See [CONTRIBUTING.md](../../CONTRIBUTING.md#adding-a-dataset).
