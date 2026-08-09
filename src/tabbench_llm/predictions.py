"""Prediction generation step for the benchmark pipeline (Step 1).

Trains each (model, dataset, seed) combination and saves predictions as CSV
files.  Supports resume: existing prediction files are skipped unless
``overwrite=True``.  Measures train/inference time, peak memory, GPU power
(pynvml), and CPU energy (Intel RAPL).

Usage
-----
::

    from tabbench_llm.config import load_config
    from tabbench_llm.predictions import compute_predictions

    config = load_config("configs/benchmark_v0.1.json")
    compute_predictions(config)
"""

import gc
import json
import logging
import os
import random
import shutil
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from tabbench_llm.benchmark import configure_benchmark
from tabbench_llm.coverage import DESIGN_SKIPS
from tabbench_llm.dataset import TaskType
from tabbench_llm.llm import CellTimeout, LLMModel, UnparseableResponses, is_llm_model
from tabbench_llm.logging_utils import LOG_FORMAT, run_file_logger
from tabbench_llm.seeds import get_seeds

try:
    from tabbench_llm.model import AutoGluonModel
except ImportError:
    AutoGluonModel = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# No model is classification-only in the bio benchmark; kept as an extension point.
CLASSIFICATION_ONLY_MODELS: set[str] = set()
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False


class _PsutilMemoryTracker:
    _POLL_S = 0.05

    def __init__(self):
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        proc = _psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / (1024**2)
                if rss > self._peak_mb:
                    self._peak_mb = rss
            except Exception:
                break
            self._stop.wait(self._POLL_S)

    def __enter__(self):
        self._stop.clear()
        self._peak_mb = _psutil.Process().memory_info().rss / (1024**2)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_mb, 2)


class _TracemallocMemoryTracker:
    _peak_mb: float = 0.0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *_):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_mb = peak / (1024**2)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_mb, 2)


def _memory_tracker():
    return _PsutilMemoryTracker() if _HAS_PSUTIL else _TracemallocMemoryTracker()


# ---------------------------------------------------------------------------
# Power / energy tracking
# ---------------------------------------------------------------------------

try:
    import pynvml as _pynvml

    _pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _pynvml = None
    _HAS_PYNVML = False

_RAPL_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
_RAPL_MAX_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"
_HAS_RAPL = os.path.isfile(_RAPL_PATH)


def _read_rapl():
    try:
        with open(_RAPL_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return None


class _PowerTracker:
    _POLL_S = 0.1

    def __init__(self):
        self._gpu_samples: list[float] = []
        self._gpu_handle = None
        self._elapsed_s = 0.0
        self._cpu_energy_j: float | None = None
        self._stop = threading.Event()
        self._thread = None

    def _poll_gpu(self):
        while not self._stop.is_set():
            try:
                mw = _pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                self._gpu_samples.append(mw / 1000.0)
            except Exception:
                pass
            self._stop.wait(self._POLL_S)

    def __enter__(self):
        self._start = time.perf_counter()
        self._gpu_samples = []
        if _HAS_PYNVML:
            try:
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                self._stop.clear()
                self._thread = threading.Thread(target=self._poll_gpu, daemon=True)
                self._thread.start()
            except Exception:
                self._gpu_handle = None
        self._cpu_start = _read_rapl() if _HAS_RAPL else None
        return self

    def __exit__(self, *_):
        self._elapsed_s = time.perf_counter() - self._start
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if _HAS_RAPL and self._cpu_start is not None:
            end = _read_rapl()
            if end is not None:
                delta = end - self._cpu_start
                if delta < 0:
                    try:
                        with open(_RAPL_MAX_PATH) as f:
                            delta += int(f.read().strip())
                    except Exception:
                        pass
                self._cpu_energy_j = delta / 1e6

    @property
    def gpu_mean_power_w(self):
        return (
            round(sum(self._gpu_samples) / len(self._gpu_samples), 2) if self._gpu_samples else None
        )

    @property
    def gpu_energy_j(self):
        if self._gpu_samples and self._elapsed_s > 0:
            return round(sum(self._gpu_samples) / len(self._gpu_samples) * self._elapsed_s, 2)
        return None

    @property
    def cpu_energy_j(self):
        return round(self._cpu_energy_j, 2) if self._cpu_energy_j is not None else None


# ---------------------------------------------------------------------------
# Per-model NaN handling
# ---------------------------------------------------------------------------

#: Valid per-model missing-value policies. ``native``/``none`` keep NaNs so the model
#: (or AutoGluon's internal pipeline) handles them; the rest fill from *training*
#: statistics. The benchmark's drop-high-missing + variance cap run upstream and are
#: shared across models; only this fill step is per-model.
_NAN_POLICIES = ("native", "none", "median", "mean", "zero")

#: Policy for models without a ``nan_policy`` entry: keep NaNs for AutoGluon to impute.
DEFAULT_NAN_POLICY = "native"


def _resolve_nan_policy(model_name: str, nan_policy: dict | None) -> str:
    """Pick the NaN policy for *model_name*: explicit entry, else the ``default`` key, else native."""
    if not nan_policy:
        return DEFAULT_NAN_POLICY
    return nan_policy.get(model_name, nan_policy.get("default", DEFAULT_NAN_POLICY))


def _apply_nan_policy(
    train: pd.DataFrame, test: pd.DataFrame, policy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill feature NaNs per *policy*, fitting fill values on TRAIN only (no leakage).

    ``native``/``none`` is a no-op — NaNs are left for the model / AutoGluon to handle
    internally. ``median``/``mean`` fill from the training-column statistic; ``zero``
    fills with 0. The identical per-column fill is applied to *test*. Returns new frames
    (originals untouched); the label column is never imputed.
    """
    if policy in ("native", "none"):
        return train, test
    if policy not in _NAN_POLICIES:
        raise ValueError(f"Unknown nan policy {policy!r}; expected one of {_NAN_POLICIES}.")

    label_col = train.columns[-1]
    feat = [c for c in train.columns if c != label_col]
    X_train = train[feat]
    if policy == "median":
        fill = X_train.median(numeric_only=True)
    elif policy == "mean":
        fill = X_train.mean(numeric_only=True)
    else:  # "zero"
        fill = pd.Series(0.0, index=feat)

    train = train.copy()
    train[feat] = X_train.fillna(fill)
    test = test.copy()
    test_feat = [c for c in feat if c in test.columns]
    test[test_feat] = test[test_feat].fillna(fill[test_feat])
    return train, test


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@contextmanager
def _timed():
    t = [0.0]
    start = time.perf_counter()
    try:
        yield t
    finally:
        t[0] = time.perf_counter() - start


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prepare_splits(config, seed_index: int | None = None):
    """Materialize every train/test split into the on-disk cache (no model fitting).

    Run once, single-threaded, before launching multi-GPU shard workers. The processed-split
    cache is shared by all workers (same seed + param hash) but its pickle writes are not
    concurrency safe, so the workers must only ever *read* it. Iterating the benchmark
    computes and caches each split (see ``TabBenchLLM.__getitem__``); here we exhaust the
    iterator and discard the data, leaving a fully warmed cache for the parallel workers.
    """
    logger.info("=" * 60 + "\nSTEP 0: Preparing (caching) dataset splits")
    seeds = get_seeds(config)
    if seed_index is not None:
        seeds = [seeds[seed_index]]
    for seed in seeds:
        logger.info("--- Seed %s (prepare) ---", seed)
        config["random_state"] = seed
        _set_global_seeds(seed)
        benchmark = configure_benchmark(config)
        n_cached = sum(1 for _ in benchmark)
        logger.info("Cached %d split(s) for seed %s", n_cached, seed)


def _is_context_window_error(exc: Exception) -> bool:
    """Whether an exception is an API context-window overflow.

    The predictive pre-skip (``LLMModel.context_skip_reason``) estimates prompt tokens from
    characters and so can miss a cell that sits just over the boundary; when the request then
    400s, this lets the caller record a clean *skip* instead of a hard failure. The proxies word
    it differently (litellm ``ContextWindowExceededError``, vLLM "maximum context length is N
    tokens"), so match on the message rather than the exception class."""
    msg = str(exc).lower()
    return "context" in msg and ("length" in msg or "window" in msg)


def _is_infrastructure_error(exc: Exception) -> bool:
    """Whether a failed request should be retried instead of scored as model performance."""
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    msg = str(exc).lower()
    markers = (
        "rate limit",
        "too many requests",
        "quota",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "connection error",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
    )
    return any(marker in msg for marker in markers)


def _execute_cell(
    model_name,
    data_train,
    data_test,
    key,
    task_type,
    *,
    seed,
    config,
    predictions_dir,
    logs_dir,
    stats_dir,
    autogluon_path,
    mem_backend,
    truth_lock,
    use_file_logger,
):
    """Fit one (model, dataset) cell, write its predictions/proba/truth/stats, return the record.

    Self-contained so it can run in a worker thread: every output file is per-cell except the
    shared per-dataset ground-truth file, whose write is guarded by ``truth_lock``. Per-cell
    file logging is skipped under concurrency (``use_file_logger=False``) because
    ``run_file_logger`` swaps a process-global root handler.
    """
    pred_path = os.path.join(predictions_dir, f"{key}_{model_name}_predictions.csv")
    proba_path = os.path.join(predictions_dir, f"{key}_{model_name}_proba.csv")
    log_path = os.path.join(logs_dir, f"{key}_{model_name}.log")
    nan_pol = _resolve_nan_policy(model_name, config["nan_policy"])

    record = {
        "dataset": key,
        "model": model_name,
        "nan_policy": nan_pol,
        "n_train_samples": len(data_train),
        "n_test_samples": len(data_test),
        # reason is the machine-readable skip/fail category consumed by
        # tabbench_llm.coverage; empty on a passing run, error carries the detail.
        "status": "pass",
        "reason": "",
        "error": "",
        "timestamp": datetime.now().isoformat(),
        "train_time_s": None,
        "inference_time_s": None,
        "inference_time_per_sample_ms": None,
        "train_peak_memory_mb": None,
        "inference_peak_memory_mb": None,
        "memory_backend": mem_backend,
        "n_models_trained": None,
        "n_base_models": None,
        "ag_total_fit_time_s": None,
        "ag_time_per_model_s": None,
        "train_gpu_power_w": None,
        "train_gpu_energy_j": None,
        "train_cpu_energy_j": None,
        "inference_gpu_power_w": None,
        "inference_gpu_energy_j": None,
        "inference_cpu_energy_j": None,
    }

    with run_file_logger(log_path) if use_file_logger else nullcontext():
        if is_llm_model(model_name):
            # Hold the system prompt fixed across CV folds (override with
            # $TABBENCH_LLM_PROMPT_INDEX for a dedicated prompt-sensitivity sweep) so the
            # across-fold score spread is data variance alone, not a convolution of data
            # and prompt-wording variance.
            model = LLMModel(
                model_name=model_name,
                task_type=task_type,
                prompt_index=int(os.environ.get("TABBENCH_LLM_PROMPT_INDEX", "0")),
                seed=seed,
                **config["llm_settings"],
            )
        else:
            assert (
                AutoGluonModel is not None
            ), "autogluon is required for non-LLM models; install tabbench-llm[autogluon]."
            model = AutoGluonModel(
                ensemble=config["ensemble"],
                optimize=config["optimize"],
                models=[model_name],
                task_type=task_type,
                autogluon_time_limit=config["autogluon_time_limit"],
                autogluon_presets=config["autogluon_presets"],
                autogluon_path=os.path.join(autogluon_path, key),
                num_hpo_trials=config["num_hpo_trials"],
            )

        try:
            data_train_fit, data_test_fit = _apply_nan_policy(data_train, data_test, nan_pol)

            with _timed() as tt, _memory_tracker() as tm, _PowerTracker() as tp:
                model.fit(data_train_fit)
            record["train_time_s"] = round(tt[0], 3)
            record["train_peak_memory_mb"] = tm.peak_mb
            record["train_gpu_power_w"] = tp.gpu_mean_power_w
            record["train_gpu_energy_j"] = tp.gpu_energy_j
            record["train_cpu_energy_j"] = tp.cpu_energy_j

            with _timed() as it, _memory_tracker() as im, _PowerTracker() as ip:
                y_pred = model.predict(data_test_fit)
            record.update(model.get_fit_stats())
            if "llm_unparsed" in record:
                # Carried into the metrics CSV downstream: a cell can sit under the refusal
                # ceiling and still have had, say, 15% of its rows answered by fallback, which
                # is a caveat on that score rather than a reason to drop it.
                record["llm_unparsed_frac"] = round(
                    record["llm_unparsed"] / max(len(data_test), 1), 4
                )
            record["inference_time_s"] = round(it[0], 3)
            record["inference_peak_memory_mb"] = im.peak_mb
            record["inference_time_per_sample_ms"] = round(it[0] / len(data_test) * 1000, 4)
            record["inference_gpu_power_w"] = ip.gpu_mean_power_w
            record["inference_gpu_energy_j"] = ip.gpu_energy_j
            record["inference_cpu_energy_j"] = ip.cpu_energy_j

            y_pred.sort_index().to_csv(pred_path, index=True)

            if task_type == TaskType.Classification:
                try:
                    model.predict_proba(data_test_fit).sort_index().to_csv(proba_path, index=True)
                except Exception as e:
                    logger.warning("predict_proba failed for %s / %s: %s", key, model_name, e)

            truth_path = os.path.join(predictions_dir, f"{key}_ground_truth.csv")
            with truth_lock:
                if not os.path.exists(truth_path):
                    data_test[["target"]].sort_index().to_csv(truth_path, index=True)

        except Exception as e:
            if isinstance(e, ValueError) and "train set will be empty" in str(e):
                logger.info(
                    "Skipping %s / %s: empty split after AutoGluon class filtering.",
                    key,
                    model_name,
                )
                record["status"] = "skip"
                record["reason"] = "empty_split_after_filtering"
                record["error"] = "empty_split_after_filtering"
            elif is_llm_model(model_name) and _is_context_window_error(e):
                # The char-based pre-skip missed a cell just over the window; record it as a
                # clean skip (one wasted request) rather than a hard failure.
                logger.info(
                    "Skipping %s / %s: context window exceeded at request time.", key, model_name
                )
                record["status"] = "skip"
                record["reason"] = "context_window"
                record["error"] = f"context_window_exceeded: {e}"
            elif isinstance(e, UnparseableResponses):
                # A failure, not a skip: the model answered, the answers were unusable. Marking
                # it a skip would file it next to "could not be attempted" and hide a compliance
                # problem that belongs in the results.
                logger.error("Failing %s / %s: %s", key, model_name, e)
                record["status"] = "fail"
                record["reason"] = "unparseable_responses"
                record["error"] = f"unparseable_responses: {e}"
            elif isinstance(e, CellTimeout):
                # The budget was offered and the model did not answer within it, so this is a
                # failure scored at chance — not a design exclusion. The two differ in who
                # imposed the limit: a context window is a property of the model that the
                # harness can check before asking, whereas the wall-clock budget is the same
                # for every entrant and only a model too slow to use it comes back empty.
                # Recording it as a skip would let the slowest models pick which targets
                # count towards their rating. The grid still advances either way.
                logger.error("Failing %s / %s: %s", key, model_name, e)
                record["status"] = "fail"
                record["reason"] = "cell_timeout"
                record["error"] = f"cell_timeout: {e}"
            elif is_llm_model(model_name) and _is_infrastructure_error(e):
                # Quota, rate-limit and service failures say nothing about the model. They are
                # non-terminal: completeness checks treat "retry" like a missing unit.
                logger.error("Retry required for %s / %s: %s", key, model_name, e)
                record["status"] = "retry"
                record["reason"] = "infrastructure_error"
                record["error"] = str(e)
            else:
                logger.error("Error for %s / %s: %s", key, model_name, e, exc_info=True)
                record["status"] = "fail"
                record["reason"] = "fit_error"
                record["error"] = str(e)

        with open(os.path.join(stats_dir, f"{key}_{model_name}.json"), "w") as f:
            json.dump(record, f, indent=2)

    _cleanup_model(model)
    del model
    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return record


def compute_predictions(
    config,
    seed_index: int | None = None,
    overwrite: bool = False,
    reverse: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
):
    """Train all models and save predictions to CSV.

    Parameters
    ----------
    config : dict
        Loaded benchmark configuration.
    seed_index : int | None
        Run only this zero-based seed index (for parallel seed execution).
    overwrite : bool
        Re-run even when a prediction file already exists.
    reverse : bool
        Iterate datasets in reverse order (useful for forward+reverse parallel jobs).
    num_shards, shard_index : int
        Split the (model x dataset) grid across ``num_shards`` workers; this process runs
        only the cells with ``cell_index % num_shards == shard_index``. Used for multi-GPU
        execution — launch one worker per GPU (each pinned with ``CUDA_VISIBLE_DEVICES``)
        with the same ``num_shards`` and a distinct ``shard_index``. The cell ordering is
        deterministic, so workers partition the grid without overlap. Warm the split cache
        first (``--step prepare``) so the workers only read it (the cache is not concurrency
        safe to write). Results aggregate naturally — every worker writes into the same
        ``output_dir``, and a cell already on disk is skipped.
    """
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    logger.info("=" * 60 + "\nSTEP 1: Computing Predictions")

    mem_backend = "psutil" if _HAS_PSUTIL else "tracemalloc"
    output_dir = config["output_dir"]

    cache_dir = config["cache_dir"]
    autogluon_path = (
        os.path.join(cache_dir, "autogluon") if cache_dir else os.path.join(".cache", "autogluon")
    )
    os.makedirs(autogluon_path, exist_ok=True)

    seeds = get_seeds(config)
    if seed_index is not None:
        seeds = [seeds[seed_index]]

    model_names = config["models"]
    model_size_limits = config["model_limits"]

    # Per-model NaN handling: ``nan_policy`` maps model key -> fill policy, with an
    # optional "default" entry; unlisted models keep NaNs ("native") for AutoGluon to
    # impute internally. The fill is fit on each model's training split at predict time.
    nan_policy = config["nan_policy"]
    if nan_policy:
        logger.info("NaN handling policy: %s", nan_policy)

    for seed in seeds:
        logger.info("--- Seed %s ---", seed)
        config["random_state"] = seed
        _set_global_seeds(seed)
        benchmark = configure_benchmark(config)

        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        predictions_dir = os.path.join(seed_dir, "predictions")
        logs_dir = os.path.join(seed_dir, "logs")
        stats_dir = os.path.join(seed_dir, "stats")
        for d in (predictions_dir, logs_dir, stats_dir):
            os.makedirs(d, exist_ok=True)

        if reverse:
            benchmark._key_list = list(reversed(benchmark._key_list))

        pbar = tqdm(total=len(benchmark) * len(model_names), desc=f"Seed {seed}")
        results = []

        # LLM cells are network-bound, so optionally run them concurrently across the grid
        # (TABBENCH_LLM_CELL_WORKERS > 1). Baselines stay inline. The shared per-dataset
        # ground-truth write is guarded; each cell's other outputs are distinct files.
        cell_workers = max(1, int(os.environ.get("TABBENCH_LLM_CELL_WORKERS", "1")))
        truth_lock = threading.Lock()
        pool = ThreadPoolExecutor(max_workers=cell_workers) if cell_workers > 1 else None
        futures: list = []

        # Deterministic grid index, reset per seed, so multi-GPU shard workers (each with
        # the same num_shards but a distinct shard_index) partition the cells with no overlap.
        cell_idx = -1
        for model_name in model_names:
            for data_train, data_test, key, task_type in benchmark:
                cell_idx += 1
                if num_shards > 1 and cell_idx % num_shards != shard_index:
                    pbar.update(1)
                    continue
                pbar.set_description(f"Seed {seed} | {key} | {model_name}")

                if data_train is None or data_test is None:
                    pbar.update(1)
                    continue

                # An empty train or test split can't be fit (AutoGluon's internal
                # holdout split raises on n_samples=0). Record the skip and move on with
                # a single line instead of letting fit() emit a full traceback.
                if len(data_train) == 0 or len(data_test) == 0:
                    logger.warning(
                        "Skipping %s / %s: empty split (train=%d, test=%d).",
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                    )
                    _write_skip_record(
                        stats_dir,
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                        "empty_split",
                        f"empty_split: train={len(data_train)}, test={len(data_test)}",
                    )
                    pbar.update(1)
                    continue

                # Per-model input-size limit (foundation models that OOM on wide, large
                # inputs): skip when the dataset's raw feature count exceeds max_features AND
                # the training split exceeds max_samples. Both must be exceeded, so the model
                # still runs on narrow data at any N and on wide data at small N; the raw
                # feature count is short-circuited behind the (cheap) sample check so its
                # uncapped load only happens on cells large enough to matter.
                if model_name in model_size_limits:
                    lim = model_size_limits[model_name]
                    max_features, max_samples = lim["max_features"], lim["max_samples"]
                    dataset_name = benchmark.split_key(key)[0]
                    if (
                        max_features is not None
                        and max_samples is not None
                        and len(data_train) > max_samples
                        and benchmark.raw_num_features(dataset_name) > max_features
                    ):
                        _write_skip_record(
                            stats_dir,
                            key,
                            model_name,
                            len(data_train),
                            len(data_test),
                            "model_limit",
                            f"exceeds {model_name} input limit: "
                            f"{benchmark.raw_num_features(dataset_name)} features > {max_features} "
                            f"and n_train {len(data_train)} > {max_samples}",
                        )
                        pbar.update(1)
                        continue

                # Predictive context-window skip: an LLM carries the whole training table in
                # every request, so if even a single-row prompt would overflow the model's
                # context window the cell can never succeed — skip it up front (a clean skip
                # record, no wasted API calls) instead of firing requests that 400. No-op when
                # the model has no configured context_window.
                if is_llm_model(model_name):
                    probe = LLMModel(
                        model_name=model_name,
                        task_type=task_type,
                        prompt_index=int(os.environ.get("TABBENCH_LLM_PROMPT_INDEX", "0")),
                        seed=seed,
                        **config["llm_settings"],
                    )
                    probe.fit(data_train)
                    reason = probe.context_skip_reason(data_test)
                    if reason is not None:
                        logger.info("Skipping %s / %s: %s", key, model_name, reason)
                        _write_skip_record(
                            stats_dir,
                            key,
                            model_name,
                            len(data_train),
                            len(data_test),
                            "context_window",
                            reason,
                        )
                        pbar.update(1)
                        continue

                # Skip classification-only models for regression datasets
                if task_type == TaskType.Regression and model_name in CLASSIFICATION_ONLY_MODELS:
                    _write_skip_record(
                        stats_dir,
                        key,
                        model_name,
                        len(data_train),
                        len(data_test),
                        "classification_only",
                        f"{model_name} only supports classification tasks",
                    )
                    pbar.update(1)
                    continue

                # Skip constant-target datasets for regression
                if task_type == TaskType.Regression:
                    label_col = data_train.columns[-1]
                    if data_train[label_col].std() == 0:
                        _write_skip_record(
                            stats_dir,
                            key,
                            model_name,
                            len(data_train),
                            len(data_test),
                            "constant_target",
                            "constant_target: all training target values are equal",
                        )
                        pbar.update(1)
                        continue

                pred_path = os.path.join(predictions_dir, f"{key}_{model_name}_predictions.csv")
                proba_path = os.path.join(predictions_dir, f"{key}_{model_name}_proba.csv")
                # For classification, both files must exist to skip — otherwise
                # rerun so the missing proba file gets written and downstream
                # log_loss / ROC-AUC become available.
                if task_type == TaskType.Classification:
                    already_complete = os.path.exists(pred_path) and os.path.exists(proba_path)
                else:
                    already_complete = os.path.exists(pred_path)
                if already_complete and not overwrite:
                    pbar.update(1)
                    continue

                # Dispatch: LLM cells run concurrently (network-bound) when
                # TABBENCH_LLM_CELL_WORKERS > 1; baselines (CPU-bound) run inline.
                if pool is not None and is_llm_model(model_name):
                    futures.append(
                        pool.submit(
                            _execute_cell,
                            model_name,
                            data_train,
                            data_test,
                            key,
                            task_type,
                            seed=seed,
                            config=config,
                            predictions_dir=predictions_dir,
                            logs_dir=logs_dir,
                            stats_dir=stats_dir,
                            autogluon_path=autogluon_path,
                            mem_backend=mem_backend,
                            truth_lock=truth_lock,
                            use_file_logger=False,
                        )
                    )
                else:
                    results.append(
                        _execute_cell(
                            model_name,
                            data_train,
                            data_test,
                            key,
                            task_type,
                            seed=seed,
                            config=config,
                            predictions_dir=predictions_dir,
                            logs_dir=logs_dir,
                            stats_dir=stats_dir,
                            autogluon_path=autogluon_path,
                            mem_backend=mem_backend,
                            truth_lock=truth_lock,
                            use_file_logger=True,
                        )
                    )
                    pbar.update(1)

        if pool is not None:
            for fut in as_completed(futures):
                results.append(fut.result())
                pbar.update(1)
            pool.shutdown()
        pbar.close()
        _log_seed_summary(seed, results, stats_dir)


def _cleanup_model(model):
    """Clear AutoGluon internals and delete the on-disk predictor dir.

    The in-memory teardown breaks reference cycles so the predictor can be GC'd.
    The on-disk predictor directory is then removed: it is write-once / read-never
    once predictions, probabilities, and stats are saved (resume keys off the result
    CSVs, not this dir, and every new run gets a fresh ``uuid`` path), so leaving it
    behind only accumulates disk. Runs unconditionally after save, so it also sweeps
    the partial directory left by a failed ``fit()``.
    """
    try:
        if model.predictor is not None:
            # _learner / _trainer are AutoGluon internals: navigate defensively.
            learner = getattr(model.predictor, "_learner", None)
            trainer = getattr(learner, "_trainer", None) if learner else None
            if trainer is not None and hasattr(trainer, "models"):
                trainer.models.clear()
            model.predictor = None
    except Exception:
        pass

    path = model.autogluon_path
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _write_skip_record(stats_dir, key, model_name, n_train, n_test, reason, error):
    """Record a unit the benchmark excluded by design.

    ``reason`` is one of :data:`tabbench_llm.coverage.DESIGN_SKIPS` — the machine-readable
    category the leaderboard reads to keep the skip out of the score rather than treating
    it as a loss; ``error`` carries the human-readable detail.
    """
    assert (
        reason in DESIGN_SKIPS
    ), f"undeclared skip reason {reason!r}; add it to coverage.DESIGN_SKIPS"
    record = {
        "dataset": key,
        "model": model_name,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "status": "skip",
        "reason": reason,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(stats_dir, f"{key}_{model_name}.json"), "w") as f:
        json.dump(record, f, indent=2)


def _log_seed_summary(seed, results, stats_dir):
    all_records = []
    for fname in sorted(os.listdir(stats_dir)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(stats_dir, fname)) as f:
                    all_records.append(json.load(f))
            except Exception:
                pass

    n_total = len(all_records)
    n_failed = sum(r.get("status") == "fail" for r in all_records)
    n_skipped = sum(r.get("status") == "skip" for r in all_records)
    n_this = len(results)
    n_this_failed = sum(r.get("status") == "fail" for r in results)
    n_this_skipped = sum(r.get("status") == "skip" for r in results)

    if n_total:
        logger.info(
            "Seed %s: %d this run (%d passed, %d skipped, %d failed) | "
            "%d pre-existing (%d skipped, %d failed). Stats: %s",
            seed,
            n_this,
            n_this - n_this_failed - n_this_skipped,
            n_this_skipped,
            n_this_failed,
            n_total - n_this,
            n_skipped - n_this_skipped,
            n_failed - n_this_failed,
            stats_dir,
        )
