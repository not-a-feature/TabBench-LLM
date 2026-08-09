"""Dataset loading, caching, and iteration for the TabBench-LLM pipeline.

The :class:`TabBenchLLM` class fetches biological datasets through the bio loaders
(GEO / TCGA / Kaggle / OpenML), adapts them to a native
:class:`~tabbench_llm.dataset.Dataset`, applies train/test splits, and caches the
prepared splits to disk for faster subsequent runs.

Datasets and targets
--------------------
Each dataset may have multiple targets.  The benchmark expands every
(dataset, target) pair into a separate "key" of the form
``"{dataset_name}_{target_idx}"``.  Iterating over a :class:`TabBenchLLM` instance
yields these keys, making it easy to run a model on every target independently.

Example
-------
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
"""

import hashlib
import inspect
import json
import logging
import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from tqdm import tqdm

from tabbench_llm.data import (
    DEFAULT_MAX_FEATURES,
    dataset_names,
    is_known_dataset,
    load_as_dataset,
)
from tabbench_llm.dataset import TaskType

logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from dataset downloads
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

#: Features missing in more than this fraction of *training* samples are dropped
#: rather than kept/imputed — too sparse to reconstruct reliably (see
#: ``_fit_apply_features``).
_MAX_FEATURE_MISSING_FRAC = 0.5


def configure_benchmark(config, init_benchmark: bool = True) -> "TabBenchLLM":
    """Instantiate a :class:`TabBenchLLM` from a loaded config dict.

    Every :class:`TabBenchLLM` constructor parameter must be present in *config*
    (``load_config`` guarantees this); a missing key raises rather than silently
    falling back to a constructor default, so the split is fully determined by the
    config. When ``init_benchmark`` is true (default), the returned benchmark has
    already loaded and cached its dataset splits.
    """
    params = inspect.signature(TabBenchLLM).parameters
    benchmark = TabBenchLLM(**{name: config[name] for name in params})
    if init_benchmark:
        benchmark.init_datasets()
    return benchmark


class TabBenchLLM:
    """Manage dataset loading, caching, and iteration for the benchmark.

    Provides an iterator-like container of prepared train/test splits sourced from
    the bio dataset registry.

    Parameters
    ----------
    dataset_names_classification : list[str] | None
        Classification dataset ids.  ``None`` loads every enabled classification
        dataset in the registry.
    dataset_names_regression : list[str] | None
        Regression dataset ids.  ``None`` loads every enabled regression dataset.
    test_size : float
        Fraction reserved for testing (default 0.2).
    random_state : int
        Seed for the train/test split (default 42).
    cache_dir : str | None
        Directory for cached dataset splits.  Defaults to ``".cache"``.
    min_samples_per_class : int
        Classes with fewer samples than this are removed before splitting
        (classification only).  Default 9.
    group_regression_splits : bool
        If ``True`` (default), use :class:`~sklearn.model_selection.GroupShuffleSplit`
        to keep co-measured samples in the same split, preventing leakage in
        multi-measurement datasets.
    cv_folds : int | None
        If set to ``k``, evaluate with (repeated) k-fold cross-validation instead of a
        repeated random holdout: classification uses
        :class:`~sklearn.model_selection.StratifiedKFold`, regression uses
        :class:`~sklearn.model_selection.KFold` (or
        :class:`~sklearn.model_selection.GroupKFold` when ``group_regression_splits``).
        ``random_state`` is then interpreted as the global split index ``g`` and the
        run yields fold ``g % k`` of the shuffle for repeat ``g // k`` — so across the
        ``k`` splits of a repeat every sample is tested exactly once and the across-split
        dispersion is a genuine CV error bar, not the optimistic spread of overlapping
        holdouts. ``test_size`` is ignored when set (the test fraction is ``1/k``). The
        run's total split count is ``k * n_repetitions`` (``n_repetitions`` = CV repeats;
        see :func:`~tabbench_llm.seeds.get_seeds`). ``None`` (default) keeps the holdout.
    max_features_default : int | None
        Feature cap for HDLSS datasets: wide matrices are truncated to a uniform random
        subset of columns (train-only, seeded by the split index) to bound compute time.
        ``None`` keeps all features.  Per-dataset overrides live in the bio registry.
    max_classes : int | None
        If set, classification tasks are capped to their ``max_classes`` most frequent
        classes (samples of rarer classes are dropped).  ``None`` (default) keeps all
        classes.  Useful to fit class-limited models (e.g. TabPFN, ~10 classes) without
        many-class ECOC; complementary to it so both views can be benchmarked.

    Notes
    -----
    Missing values are left in place by the benchmark and imputed per model at predict
    time (``predictions._apply_nan_policy``, config key ``nan_policy``), so each model
    can be benchmarked under its own policy.  Features missing in more than
    :data:`_MAX_FEATURE_MISSING_FRAC` of training rows are dropped first.

    Examples
    --------
    ::

        bench = TabBenchLLM(
            dataset_names_classification=["OpenML-1138"],
            dataset_names_regression=[],
            cache_dir=".cache",
        )
        bench.init_datasets()
        train_df, test_df, key, task_type = bench[0]
    """

    def __init__(
        self,
        dataset_names_classification: list[str] | None = None,
        dataset_names_regression: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cache_dir: str | None = None,
        min_samples_per_class: int = 10,
        group_regression_splits: bool = True,
        max_features_default: int | None = DEFAULT_MAX_FEATURES,
        max_classes: int | None = None,
        train_subsample: int | None = None,
        test_subsample: int | None = None,
        cv_folds: int | None = None,
    ):
        if not cache_dir:
            cache_dir = ".cache"

        self.cache_dir = cache_dir

        # Bio datasets cache under <cache_dir>/bio (full, uncapped — the feature cap is
        # applied train-only after the split, in _fit_apply_features).
        self.dataset_cache_dir = os.path.join(cache_dir, "datasets")
        self.max_features_default = max_features_default

        self.test_size = test_size
        self.random_state = random_state
        self.min_samples_per_class = min_samples_per_class
        self.group_regression_splits = group_regression_splits
        self.max_classes = max_classes
        # (Repeated) k-fold CV toggle. When set, random_state carries the global split
        # index and _split returns one CV fold; None keeps the repeated random holdout.
        self.cv_folds = cv_folds
        # HDLSS sample-size axis: cap the number of TRAINING rows (stratified), applied
        # train-only after the split — the held-out test set is untouched, so metrics stay
        # comparable across sizes (a learning curve). None = use all training rows.
        self.train_subsample = train_subsample
        # Cap the number of TEST rows (stratified), applied test-only after the split. Keeps
        # per-row-cost models (the LLM classifier issues one API call per test row) affordable
        # while leaving every model scored on the *same* held-out rows. None = full test set.
        self.test_subsample = test_subsample

        # The processed-split cache must be keyed on *every* parameter that changes the
        # produced splits, not just the seed — otherwise changing e.g. max_classes
        # silently reuses stale splits.
        # The NaN fill policy is deliberately absent: it is applied per model at predict
        # time (predictions._apply_nan_policy), so the cached split holds NaN-intact
        # features and is independent of the policy.
        split_params = {
            "test_size": test_size,
            "min_samples_per_class": min_samples_per_class,
            "max_classes": max_classes,
            "max_features_default": max_features_default,
            "group_regression_splits": group_regression_splits,
            "train_subsample": train_subsample,
            "test_subsample": test_subsample,
            "cv_folds": cv_folds,
        }
        param_hash = hashlib.md5(json.dumps(split_params, sort_keys=True).encode()).hexdigest()[:8]
        self.cache_dir_processed = os.path.join(
            cache_dir, "datasets_processed", f"seed_{random_state}_{param_hash}"
        )
        os.makedirs(self.cache_dir_processed, exist_ok=True)

        if dataset_names_classification is None:
            self.dataset_names_classification = dataset_names("binary") + dataset_names(
                "multiclass"
            )
        else:
            self.dataset_names_classification = list(dataset_names_classification)

        if dataset_names_regression is None:
            self.dataset_names_regression = dataset_names("regression")
        else:
            self.dataset_names_regression = list(dataset_names_regression)

        logger.info(
            "Datasets: %d classification, %d regression",
            len(self.dataset_names_classification),
            len(self.dataset_names_regression),
        )

        self._key_list: list[str] = []
        self._task_type_list: list[TaskType] = []
        self._index_file = os.path.join(self.cache_dir_processed, "index.json")
        self._index: dict[str, int] = self._load_index()
        self._raw_feature_counts: dict[str, int] = {}
        self.is_initialized = False

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._key_list)

    def __iter__(self):
        if not self.is_initialized:
            self.init_datasets()
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> tuple[DataFrame, DataFrame, str, TaskType]:
        """Return ``(train_df, test_df, key, task_type)`` for index *idx*."""
        if not self.is_initialized:
            self.init_datasets()

        key = self._key_list[idx]
        task_type = self._task_type_list[idx]

        if self._has_dataset_in_cache(key):
            data_train, data_test = self._load_dataset_from_cache(key)
        else:
            data_train, data_test = self._load_dataset_from_key(key)
            if data_train is not None:
                self._save_dataset(key, data_train, data_test)

        return data_train, data_test, key, task_type

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_datasets(self):
        """Populate the key list and ensure all splits are cached."""
        self._load_datasets(self.dataset_names_regression)
        self._load_datasets(self.dataset_names_classification)

        for dataset_name in self.dataset_names_classification + self.dataset_names_regression:
            for target_idx in range(self._index.get(dataset_name, 0)):
                self._key_list.append(self.get_key(dataset_name, target_idx))
                task = (
                    TaskType.Classification
                    if dataset_name in self.dataset_names_classification
                    else TaskType.Regression
                )
                self._task_type_list.append(task)

        self._save_index()
        self.is_initialized = True

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_key(dataset_name: str, target_idx: int) -> str:
        """Return the cache key ``"{dataset_name}_{target_idx}"``."""
        dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
        return f"{dataset_name}_{target_idx}"

    @staticmethod
    def split_key(key: str) -> tuple[str, int]:
        """Reverse :meth:`get_key` → ``(dataset_name, target_idx)``."""
        parts = key.split("_")
        return "_".join(parts[:-1]), int(parts[-1])

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache_paths(self, key: str) -> tuple[str, str]:
        train = f"{self.cache_dir_processed}/{key}_train.pkl"
        test = f"{self.cache_dir_processed}/{key}_test.pkl"
        return train, test

    def _has_dataset_in_cache(self, key: str) -> bool:
        train, test = self._get_cache_paths(key)
        return os.path.exists(train) and os.path.exists(test)

    def _save_dataset(self, key: str, train: DataFrame, test: DataFrame):
        train_path, test_path = self._get_cache_paths(key)
        train.to_pickle(train_path)
        test.to_pickle(test_path)

    def _load_dataset_from_cache(self, key: str) -> tuple[DataFrame | None, DataFrame | None]:
        train_path, test_path = self._get_cache_paths(key)
        try:
            data_train = pd.read_pickle(train_path)
            data_test = pd.read_pickle(test_path)
        except Exception as e:
            logger.warning("Cache for %s unreadable (%s); regenerating.", key, e)
            for f in (train_path, test_path):
                if os.path.exists(f):
                    os.remove(f)
            return self._load_dataset_from_key(key)

        # Re-deriving rare classes from the cached splits only makes sense when they hold the
        # full post-split data. With train/test subsampling the caches are intentionally tiny,
        # so a minority class legitimately falls below min_samples_per_class here; re-filtering
        # would then wrongly drop it (and skip the dataset). Rare classes were already removed
        # on the full data before caching, so trust the cached splits as-is when subsampling.
        if self.train_subsample is not None or self.test_subsample is not None:
            return data_train, data_test

        combined = pd.concat([data_train, data_test], ignore_index=True)
        _, rare = self._filter_rare_classes(combined, key)
        if rare is None:
            return None, None
        data_train = self._drop_classes(data_train, key, rare)
        if data_train is None:
            return None, None
        data_test = self._drop_classes(data_test, key, rare)
        return data_train, data_test

    def _load_index(self) -> dict[str, int]:
        if not os.path.exists(self._index_file):
            return {}
        with open(self._index_file) as f:
            return json.load(f)

    def _save_index(self):
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset_name: str):
        """Fetch + cache a dataset and adapt it to a :class:`~tabbench_llm.dataset.Dataset`."""
        if not is_known_dataset(dataset_name):
            raise ValueError(
                f"Unknown dataset {dataset_name!r}: not in the bio registry. "
                "Add it to datasets.json or $TABBENCH_LLM_DATASETS."
            )
        # Load the full (uncapped) matrix; the feature cap is applied train-only after
        # the split in _fit_apply_features to avoid test-set leakage. The per-dataset
        # registry override is honoured there via _effective_max_features.
        return load_as_dataset(
            dataset_name,
            cache_dir=self.dataset_cache_dir,
            max_features=None,
        )

    def raw_num_features(self, dataset_name: str) -> int:
        """Uncapped feature count of a dataset (memoised), for per-model ``max_features`` skips.

        The processed split cache stores the feature-*capped* matrix, so the width the
        prediction step sees in a cell reflects ``max_features_default``, not the dataset's true
        breadth. This loads the full (registry-capped only) matrix once per process to recover
        it — the number a per-model size limit must gate on so a low-cap cell of a genuinely
        wide dataset (e.g. an 8192-dim embedding capped to 2000) is still recognised as wide."""
        if dataset_name not in self._raw_feature_counts:
            self._raw_feature_counts[dataset_name] = self._load_dataset(
                dataset_name
            ).features.shape[1]
        return self._raw_feature_counts[dataset_name]

    def _effective_max_features(self, dataset_name: str) -> int | None:
        """Resolve the feature cap for a dataset: registry override else the run default."""
        from tabbench_llm.data.datasets import get_spec

        try:
            spec = get_spec(dataset_name)
        except Exception:
            return self.max_features_default
        return spec.max_features if spec.max_features is not None else self.max_features_default

    def _load_datasets(self, dataset_names: list[str]):
        for dataset_name in tqdm(dataset_names, desc="Loading datasets"):
            try:
                if dataset_name in self._index and self._index[dataset_name] > 0:
                    num_targets = self._index[dataset_name]
                else:
                    # Bio datasets are single-target by construction; the actual fetch is
                    # deferred to _load_dataset_from_key below (so we don't fetch twice).
                    num_targets = 1
                    self._index[dataset_name] = num_targets
                    self._save_index()

                for target_idx in range(num_targets):
                    key = self.get_key(dataset_name, target_idx)
                    if not self._has_dataset_in_cache(key):
                        data_train, data_test = self._load_dataset_from_key(key)
                        if data_train is not None:
                            self._save_dataset(key, data_train, data_test)
            except Exception as exc:
                # A dataset that can't be loaded — unregistered id, or a `local-*` whose
                # data file hasn't been generated yet — must not abort the whole run (the
                # benchmark/sweep is meant to cover whatever datasets ARE available). Drop
                # it (index 0 -> it contributes no keys in init_datasets) and continue.
                logger.warning(
                    "Skipping dataset %r: could not load it (%s). The run continues over "
                    "the remaining datasets.",
                    dataset_name,
                    exc,
                )
                self._index[dataset_name] = 0
                self._save_index()

    def _load_dataset_from_key(self, key: str) -> tuple[DataFrame | None, DataFrame | None]:
        dataset_name, target_idx = self.split_key(key)
        dataset = self._load_dataset(dataset_name)

        num_targets = dataset.n_targets
        if target_idx >= num_targets:
            raise ValueError(f"Target index {target_idx} out of range for dataset {dataset_name}")

        data_df = dataset.to_dataframe(target_idx)

        # Defensive target-NA drop. The bio loaders already guarantee complete targets
        # (bio.adapter.raw_to_dataset), but guard non-bio / future paths.
        label_col = data_df.columns[-1]
        data_df = data_df.dropna(subset=[label_col])
        if len(data_df) == 0:
            logger.warning("Dataset %s has 0 samples (all targets missing).", key)
            return None, None

        # Task-level class-space definition, decided pre-split *by design* (not a leak):
        # the class set must be identical across train, test, every seed and every model
        # for the benchmark to compare a fixed task, and it uses only the label marginal
        # (no feature/target relationship). Choosing it per training split would instead
        # let the task drift between seeds and admit test-only classes.
        data_df, _ = self._filter_rare_classes(data_df, key)
        if data_df is None:
            return None, None
        data_df = self._limit_classes(data_df, key)

        # Split FIRST, then fit every data-dependent feature transform on train only and
        # apply it to test — no test-set leakage.
        train, test = self._split(data_df, dataset_name, dataset, num_targets)
        # HDLSS sample-size axis: thin the TRAIN partition only (stratified), before the
        # train-fit feature prep so feature selection sees exactly the rows the model will.
        train = self._subsample_train(train, key)
        # Cap the held-out test partition too (stratified, test-only) to bound per-row LLM
        # cost; same rows for every model, so comparisons stay fair.
        test = self._subsample_test(test, key)
        return self._fit_apply_features(train, test, key)

    def _subsample_train(self, train: DataFrame, key: str) -> DataFrame:
        """Cap training rows to ``train_subsample`` (leak-free, train-only)."""
        return self._subsample_partition(train, key, self.train_subsample)

    def _subsample_test(self, test: DataFrame, key: str) -> DataFrame:
        """Cap test rows to ``test_subsample`` (leak-free, test-only).

        Applied to the held-out partition so every model is scored on the same rows; it
        bounds the per-row API cost of the LLM classifier without changing what any model
        sees relative to the others.
        """
        return self._subsample_partition(test, key, self.test_subsample)

    def _subsample_partition(self, df: DataFrame, key: str, n: int | None) -> DataFrame:
        """Cap a partition to ``n`` rows (leak-free within that partition).

        Classification subsampling is stratified: every present class keeps at least one
        row and the remaining budget is shared in proportion to class frequency, so the
        class marginal is preserved even at tiny ``N`` (when ``N`` is below the class count
        the ``N`` largest classes each contribute one row). Regression takes a plain random
        draw. Deterministic in ``random_state``.
        """
        train = df
        if n is None or len(train) <= n:
            return train

        dataset_name, _ = self.split_key(key)
        label_col = train.columns[-1]
        if dataset_name in self.dataset_names_regression:
            return train.sample(n=n, random_state=self.random_state)

        y = train[label_col]
        counts = y.value_counts()
        classes = counts.index.tolist()
        k = len(classes)
        if n <= k:
            # Too small to cover every class once — take one row from each of the N largest.
            per = {cls: (1 if i < n else 0) for i, cls in enumerate(classes)}
        else:
            # One per class, then distribute the rest proportionally (floored).
            remaining = n - k
            total = int(counts.sum())
            per = {}
            for cls in classes:
                extra = int((counts[cls] / total) * remaining)  # floor
                per[cls] = min(1 + extra, int(counts[cls]))
            # Hand out the rounding deficit to classes that still have spare rows (largest
            # first) until we hit N exactly or run out of headroom.
            deficit = n - sum(per.values())
            i = 0
            while deficit > 0 and i < 10 * k:
                cls = classes[i % k]
                if per[cls] < int(counts[cls]):
                    per[cls] += 1
                    deficit -= 1
                i += 1

        frames = [
            train[y == cls].sample(n=per[cls], random_state=self.random_state)
            for cls in classes
            if per[cls] > 0
        ]
        # Shuffle so row order doesn't encode the per-class grouping.
        return pd.concat(frames).sample(frac=1, random_state=self.random_state)

    def _split(
        self, data_df: DataFrame, dataset_name: str, dataset, num_targets: int
    ) -> tuple[DataFrame, DataFrame]:
        """Train/test split: (repeated) k-fold when ``cv_folds`` is set, else holdout.

        Holdout mode is grouped for regression (leakage-safe) and stratified otherwise.
        """
        is_regression = dataset_name in self.dataset_names_regression

        if self.cv_folds is not None:
            return self._kfold_split(data_df, dataset_name, dataset, num_targets, is_regression)

        if is_regression and self.group_regression_splits:
            group_by_df = self._group_by_df(data_df, dataset, num_targets)
            return self._grouped_train_test_split(data_df, group_by_df=group_by_df)

        label_col = data_df.columns[-1]
        stratify = data_df[label_col] if not is_regression else None
        return train_test_split(
            data_df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def _kfold_split(
        self,
        data_df: DataFrame,
        dataset_name: str,
        dataset,
        num_targets: int,
        is_regression: bool,
    ) -> tuple[DataFrame, DataFrame]:
        """Return one fold of a (repeated) k-fold split (stratified for classification).

        ``random_state`` carries the global split index ``g``; ``repeat = g // k`` seeds
        the shuffle and ``fold = g % k`` selects the held-out fold. Within a repeat the
        ``k`` folds are complementary partitions, so every sample is tested exactly once
        and the across-fold dispersion is a real CV error bar (not the optimistic spread
        of overlapping holdouts).
        """
        k = self.cv_folds
        repeat, fold = divmod(self.random_state, k)
        label_col = data_df.columns[-1]

        if not is_regression:
            y = data_df[label_col]
            min_count = int(y.value_counts().min())
            assert min_count >= k, (
                f"{dataset_name}: smallest class has {min_count} sample(s) < cv_folds={k}; "
                f"raise min_samples_per_class to >= cv_folds or lower cv_folds."
            )
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=repeat)
            splits = list(splitter.split(data_df, y))
        elif self.group_regression_splits:
            groups = self._group_labels(self._group_by_df(data_df, dataset, num_targets))
            if len(np.unique(groups)) >= k:
                # GroupKFold has no shuffle/seed, so repeats reproduce the same folds.
                splits = list(GroupKFold(n_splits=k).split(data_df, groups=groups))
            else:
                splitter = KFold(n_splits=k, shuffle=True, random_state=repeat)
                splits = list(splitter.split(data_df))
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=repeat)
            splits = list(splitter.split(data_df))

        train_idx, test_idx = splits[fold]
        return data_df.iloc[train_idx], data_df.iloc[test_idx]

    def _fit_apply_features(
        self, train: DataFrame, test: DataFrame, key: str
    ) -> tuple[DataFrame, DataFrame]:
        """Fit all feature preprocessing on TRAIN only and apply to TEST (no leakage).

        Runs after the split, so no test row ever informs feature selection or
        imputation. For wide omics matrices such
        as GEO microarrays — where scattered per-probe NaNs mean essentially every
        sample carries a missing value — the steps are:

        1. **Drop high-missingness features** missing in more than
           :data:`_MAX_FEATURE_MISSING_FRAC` of *training* rows (too sparse to use).
        2. **Random feature cap** to the effective ``max_features`` columns (registry
           override else ``max_features_default``), drawn uniformly at random rather than by
           variance; bounds compute while preserving the HDLSS character and without the
           expression-level bias a variance rank would introduce.

        Missing values are intentionally left **in place** here: imputation is applied
        per model at predict time (``predictions._apply_nan_policy``), so models that
        consume NaN natively (LightGBM/CatBoost/TabPFN) and those that need a fill are
        each handled by their configured ``nan_policy``. Both steps above are
        parameterised from ``train`` and the identical column set is applied to ``test``.
        For dense, complete matrices (RNA-seq counts, OpenML tables) under the default
        cap this is a near no-op.
        """
        dataset_name, _ = self.split_key(key)
        label_col = train.columns[-1]
        X_train = train.drop(columns=[label_col])
        X_test = test.drop(columns=[label_col])
        n_start = X_train.shape[1]

        # 1. Drop features mostly missing in the *training* partition.
        keep_mask = X_train.isna().mean() <= _MAX_FEATURE_MISSING_FRAC
        n_dropped = int((~keep_mask).sum())
        if n_dropped:
            cols = X_train.columns[keep_mask]
            X_train, X_test = X_train[cols], X_test[cols]

        # 2. Feature cap by uniform random subsampling (not variance ranking). On
        #    expression data variance scales with mean expression, so a variance rank
        #    would preferentially retain highly-expressed genes and bias the retained set
        #    by modality; a random subset is unbiased with respect to expression level.
        #    The draw depends only on column indices and the seed (self.random_state, i.e.
        #    the split index) — not on any value — so it is identical across models on this
        #    split and carries no leakage. Original column order is preserved.
        cap = self._effective_max_features(dataset_name)
        n_capped = 0
        if cap is not None and X_train.shape[1] > cap:
            rng = np.random.default_rng(self.random_state)
            chosen = np.sort(rng.choice(X_train.shape[1], size=cap, replace=False))
            cols = X_train.columns[chosen]
            n_capped = X_train.shape[1] - len(cols)
            X_train, X_test = X_train[cols], X_test[cols]

        # Missing values are left in place; imputation is per-model at predict time
        # (predictions._apply_nan_policy).

        if n_dropped or n_capped:
            logger.info(
                "Dataset %s: train-fit feature prep — dropped %d high-missing, capped %d "
                "(%d -> %d features); NaNs kept for per-model handling.",
                key,
                n_dropped,
                n_capped,
                n_start,
                X_train.shape[1],
            )

        train = pd.concat([X_train, train[[label_col]]], axis=1)
        test = pd.concat([X_test, test[[label_col]]], axis=1)
        return train, test

    # ------------------------------------------------------------------
    # Rare-class filtering
    # ------------------------------------------------------------------

    def _filter_rare_classes(
        self, data_df: DataFrame, key: str
    ) -> tuple[DataFrame | None, list | None]:
        dataset_name, _ = self.split_key(key)
        if dataset_name not in self.dataset_names_classification:
            return data_df, []
        if self.min_samples_per_class <= 0:
            return data_df, []

        label_col = data_df.columns[-1]
        counts = data_df[label_col].value_counts()
        rare = counts[counts < self.min_samples_per_class].index.tolist()
        if rare:
            logger.warning(
                "Dataset %s: removing %d class(es) with < %d samples",
                key,
                len(rare),
                self.min_samples_per_class,
            )
            data_df = data_df[~data_df[label_col].isin(rare)]

        if data_df[label_col].nunique() < 2:
            logger.warning("Dataset %s: < 2 classes remain. Skipping.", key)
            return None, None

        return data_df, rare

    def _limit_classes(self, data_df: DataFrame, key: str) -> DataFrame:
        """Keep only the ``max_classes`` most frequent classes (classification only).

        High-cardinality targets (e.g. the 50-author OpenML-1457) exceed the native
        class limit of TabPFN-family models (~10). This caps the task to its
        ``max_classes`` largest classes by sample count — the most-populated, most-
        learnable strata — and drops samples of the rarer classes; downstream
        train/test splitting stays stratified over the retained classes. It is an
        alternative to many-class ECOC support (``tabpfn-extensions``): the two are
        complementary so both the subsampled and full-class tasks can be benchmarked.

        No-op when ``max_classes`` is unset/``<=0``, the dataset is regression, or it
        already has ``<= max_classes`` classes.
        """
        dataset_name, _ = self.split_key(key)
        if dataset_name not in self.dataset_names_classification:
            return data_df
        if not self.max_classes or self.max_classes <= 0:
            return data_df

        label_col = data_df.columns[-1]
        counts = data_df[label_col].value_counts()
        if len(counts) <= self.max_classes:
            return data_df

        keep = counts.nlargest(self.max_classes).index
        before = len(data_df)
        data_df = data_df[data_df[label_col].isin(keep)]
        logger.info(
            "Dataset %s: limited to top %d of %d classes (dropped %d class(es), %d sample(s)).",
            key,
            self.max_classes,
            len(counts),
            len(counts) - self.max_classes,
            before - len(data_df),
        )
        return data_df

    def _drop_classes(self, data_df: DataFrame, key: str, classes: list) -> DataFrame | None:
        if not classes:
            return data_df
        dataset_name, _ = self.split_key(key)
        if dataset_name not in self.dataset_names_classification:
            return data_df
        label_col = data_df.columns[-1]
        data_df = data_df[~data_df[label_col].isin(classes)]
        if data_df[label_col].nunique() < 2:
            return None
        return data_df

    # ------------------------------------------------------------------
    # Group-aware train/test split (regression leakage prevention)
    # ------------------------------------------------------------------

    def _group_by_df(self, data_df: DataFrame, dataset, num_targets: int) -> DataFrame:
        """The frame whose identical non-zero rows define co-measured groups.

        Multi-target datasets group on the full target matrix; single-target datasets on
        the one target column.
        """
        if num_targets > 1:
            return pd.DataFrame(dataset.targets, columns=dataset.target_names).loc[data_df.index]
        return data_df[[data_df.columns[-1]]]

    @staticmethod
    def _group_labels(group_by_df: DataFrame) -> np.ndarray:
        """Map each row to a group id.

        Two rows share a group when their non-zero values are identical (the same physical
        measurement); an all-zero row is placed in its own unique group.
        """

        def _row_key(row):
            nonzero = {col: val for col, val in row.items() if val != 0}
            return frozenset(nonzero.items()) if nonzero else None

        keys = group_by_df.apply(_row_key, axis=1)
        unique_counter = 0
        group_labels = []
        seen: dict = {}
        for k in keys:
            if k is None:
                group_labels.append(f"__unique_{unique_counter}")
                unique_counter += 1
            else:
                if k not in seen:
                    seen[k] = str(k)
                group_labels.append(seen[k])
        return np.array(group_labels)

    def _grouped_train_test_split(
        self, data_df: DataFrame, group_by_df: DataFrame | None = None
    ) -> tuple[DataFrame, DataFrame]:
        """Split while keeping co-measured samples in the same partition.

        Rows that share a group key (see :meth:`_group_labels`) always land in the same
        split, preventing train/test leakage from replicate measurements of the same
        sample. Falls back to a plain :func:`~sklearn.model_selection.train_test_split`
        when every row is in its own unique group.
        """
        if group_by_df is None:
            group_by_df = data_df[[data_df.columns[-1]]]
        groups = self._group_labels(group_by_df)
        if len(np.unique(groups)) == len(data_df):
            return train_test_split(
                data_df, test_size=self.test_size, random_state=self.random_state
            )

        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(gss.split(data_df, groups=groups))
        return data_df.iloc[train_idx], data_df.iloc[test_idx]
