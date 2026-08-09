"""TabBench-LLM: controlled evaluation of LLMs as few-shot tabular classifiers.

The primary suite contains deterministic synthetic classification tasks with opaque labels;
public TabArena-v0.1 datasets form a separate secondary suite. LLMs, Random Forest and TabPFN
receive identical splits through a reproducible fetch/generate → split → fit/prompt → score →
rank pipeline.

Quick Start
-----------
::

    from tabbench_llm import TabBenchLLM

    bench = TabBenchLLM(
        dataset_names_classification=["OpenML-1138"],
        dataset_names_regression=[],
        cache_dir=".cache",
    )
    bench.init_datasets()
    for train_df, test_df, key, task_type in bench:
        print(key, len(train_df), len(test_df))

Or run the canonical multi-system grid and build the site::

    python scripts/grid.py --config configs/grid_all_systems.json --models GEMMA,QWEN
    python scripts/finalize_grid.py --config configs/grid_all_systems.json
"""

import importlib

__version__ = "0.1.0"
__author__ = "Jules Kreuer (Uni Tübingen)"

_public_map = {
    # Core benchmark
    "TabBenchLLM": ("tabbench_llm.benchmark", "TabBenchLLM"),
    "configure_benchmark": ("tabbench_llm.benchmark", "configure_benchmark"),
    # Data layer
    "Dataset": ("tabbench_llm.dataset", "Dataset"),
    "DatasetInfo": ("tabbench_llm.dataset", "DatasetInfo"),
    "TaskType": ("tabbench_llm.dataset", "TaskType"),
    # Models
    "AutoGluonModel": ("tabbench_llm.model", "AutoGluonModel"),
    # Leaderboard — rank models from a results directory
    "Leaderboard": ("tabbench_llm.leaderboard", "Leaderboard"),
    # Leaderboard payload for the public page
    "build_site": ("tabbench_llm.site", "build_site"),
    # Metrics
    "ClassificationMetrics": ("tabbench_llm.metrics", "ClassificationMetrics"),
    "RegressionMetrics": ("tabbench_llm.metrics", "RegressionMetrics"),
    "compute_metrics": ("tabbench_llm.metrics", "compute_metrics"),
    # Pipeline steps
    "compute_predictions": ("tabbench_llm.predictions", "compute_predictions"),
    "compute_metrics_from_predictions": (
        "tabbench_llm.evaluation",
        "compute_metrics_from_predictions",
    ),
    # Bio dataset registry / loaders
    "DATASETS": ("tabbench_llm.data", "DATASETS"),
    "dataset_names": ("tabbench_llm.data", "dataset_names"),
    "load_raw_dataset": ("tabbench_llm.data", "load_raw_dataset"),
    "load_as_dataset": ("tabbench_llm.data", "load_as_dataset"),
    # Config
    "load_config": ("tabbench_llm.config", "load_config"),
}

__all__: list[str] = sorted(_public_map.keys())


def __getattr__(name: str):
    if name in _public_map:
        module_name, attr = _public_map[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_public_map.keys()))
