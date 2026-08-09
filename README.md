# TabBench-LLM

[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A controlled benchmark for few-shot, in-context LLM classification on tabular data.

TabBench-LLM compares LLMs with RandomForest and TabPFN v2 using the same sampled training
rows and held-out examples. Its primary suite contains 19 deterministic synthetic datasets;
38 TabArena-v0.1 OpenML classification datasets form a secondary suite. The default sample
sizes are 10, 20, 50, and 100 rows, and the primary ranking metric is macro-F1.

The predeclared headline setting is **synthetic data, reasoning off, opaque labels, all
features, and 100 training rows**.

## Install

```bash
git clone https://github.com/not-a-feature/TabBench-LLM.git
cd TabBench-LLM
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

For development and grid scripts, install `requirements-dev.txt` instead. The Python package
is named `tabbench_llm`.

## Configure LLMs

Models are registered in
[`src/tabbench_llm/llm/data/llm_models.json`](src/tabbench_llm/llm/data/llm_models.json).
Credentials and endpoints go in a git-ignored `configs/llm_key.json`; copy the structure from
[`configs/llm_key.example.json`](configs/llm_key.example.json):

```json
{
  "ml-cloud": {
    "api_key": "sk-...",
    "base_url": "https://example.com/v1"
  }
}
```

An endpoint can instead be supplied with
`TABBENCH_LLM_<PROVIDER>_BASE_URL` and, optionally,
`TABBENCH_LLM_<PROVIDER>_API_KEY`.

## Run

```bash
# Run the canonical grid or one model from it
python scripts/grid.py --config configs/grid_all_systems.json
python scripts/grid.py --config configs/grid_all_systems.json --models LOCAL-QWEN3-8B

# Build the final leaderboard payload
python scripts/finalize_grid.py --config configs/grid_all_systems.json
```

Machine-specific helpers are also included:

```powershell
# Windows workstation with local Ollama models
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\run_windows_ollama.ps1 -Pilot
.\scripts\run_windows_ollama.ps1

# Hosted MLCloud slice
.\scripts\run_mlcloud.ps1
```

```bash
# Linux/H100 with job-local Ollama
bash scripts/setup_linux_ollama.sh
sbatch run_ollama_grid.sbatch
```

Runs resume per cell when the saved configuration is unchanged. Raw results belong under
`results/` and should not be committed.

## How evaluation works

- Each LLM receives the labeled training table in its prompt and predicts held-out rows.
- LLMs and trained baselines use identical subsamples and cross-validation splits.
- Synthetic targets and the hidden-label real-data arm use opaque class tokens.
- Context-window checks skip oversized cells before sending API requests.
- Unparseable responses above the configured threshold fail the cell.
- Missing attempted results are imputed at the dummy baseline; declared skips reduce coverage.
- One-hot LLM outputs do not produce ROC-AUC or log-loss unless confidence elicitation is
  enabled. Macro-F1 remains the primary metric.
- Leaderboards use per-dataset normalized scores and Bradley-Terry Elo anchored to
  RandomForest = 1000.

The grid also evaluates reasoning mode, label visibility, training size, and feature caps.
Settings that change the measured task live under `llm_settings` in the run configuration and
are recorded with every result.

## Results across machines

Export result bundles on each machine, then merge them into one result tree:

```bash
python scripts/result_bundle.py export --label ollama-pc
python scripts/result_bundle.py export --label mlcloud
python scripts/result_bundle.py export --label h100

python scripts/result_bundle.py import --dest results/grid_all_systems gpu.zip cloud.zip h100.zip
python scripts/preview_current_grid.py --recompute-metrics
```

On Windows, `snapshot_results.ps1` can import bundles and rebuild the diagnostic preview.

## Python API

```python
from tabbench_llm import TabBenchLLM

bench = TabBenchLLM(
    dataset_names_classification=["credit-g", "diabetes"],
    dataset_names_regression=[],
    cache_dir=".cache",
    train_subsample=50,
)
bench.init_datasets()

for train_df, test_df, key, task_type in bench:
    print(key, task_type, train_df.shape, test_df.shape)
```

## Key files

- [`configs/grid_all_systems.json`](configs/grid_all_systems.json): canonical model and dataset grid
- [`configs/datasets/tabarena_classification.json`](configs/datasets/tabarena_classification.json): dataset selection
- [`src/tabbench_llm/data/registry/datasets.json`](src/tabbench_llm/data/registry/datasets.json): dataset registry
- [`src/tabbench_llm/llm/model.py`](src/tabbench_llm/llm/model.py): prompt serialization and prediction parsing
- [`scripts/grid.py`](scripts/grid.py): resumable grid runner
- [`scripts/finalize_grid.py`](scripts/finalize_grid.py): publication payload builder

## License

MIT - see [LICENSE](LICENSE).
