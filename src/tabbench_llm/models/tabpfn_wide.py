"""AutoGluon model wrapper for TabPFN-Wide (the wide-dataset TabPFN variant).

TabPFN-Wide (https://github.com/not-a-feature/TabPFN-Wide) extends TabPFN-2.0 for datasets
with many features and few samples (HDLSS) — exactly the TabBench-LLM regime. It ships
as a scikit-learn classifier (``tabpfnwide.classifier.TabPFNWideClassifier``); this module
adapts it to AutoGluon's model API so it can be selected from a config like any built-in
model, via the key ``"TABPFN-WIDE"``.

Requires ``tabpfnwide`` (``pip install tabpfnwide``; v0.3 builds on ``tabpfn==8.0.3``).
Classification only — TabPFN-Wide has no regressor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import TabPFNModel
from sklearn.base import BaseEstimator, ClassifierMixin

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_WIDE_MODEL = "wide-v2-8k"


class _CloneableTabPFNWide(BaseEstimator, ClassifierMixin):
    """sklearn-clone-safe wrapper that builds a ``TabPFNWideClassifier`` lazily in ``fit``.

    ``TabPFNWideClassifier``'s constructor forbids passing both ``model_name`` and
    ``model_path``, but after construction it exposes both as attributes — so
    ``sklearn.clone`` (used by ``ManyClassClassifier``'s ECOC) re-instantiates it with
    both and crashes. This wrapper only carries the simple, clone-safe params and
    constructs the real classifier from ``model_name`` at ``fit`` time, delegating
    ``predict``/``predict_proba``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_WIDE_MODEL,
        device: str = "cuda",
        n_estimators: int = 1,
        features_per_group: int = 1,
        categorical_features_indices: list[int] | None = None,
        ignore_pretraining_limits: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.n_estimators = n_estimators
        self.features_per_group = features_per_group
        self.categorical_features_indices = categorical_features_indices
        self.ignore_pretraining_limits = ignore_pretraining_limits

    def fit(self, X, y):
        from tabpfnwide.classifier import TabPFNWideClassifier

        self.classes_ = np.unique(y)
        self.model_ = TabPFNWideClassifier(
            model_name=self.model_name,
            device=self.device,
            n_estimators=self.n_estimators,
            features_per_group=self.features_per_group,
            categorical_features_indices=self.categorical_features_indices,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class TabPFNWideModel(TabPFNModel):
    """AutoGluon wrapper around ``tabpfnwide.classifier.TabPFNWideClassifier``.

    Inherits :class:`TabPFNModel`'s preprocessing (categorical → numeric label encoding),
    GPU/CPU resource detection, and unified ``predict_proba`` handling; only ``_fit``
    differs — it builds a TabPFN-Wide checkpoint instead of stock TabPFN.
    """

    ag_key = "TABPFN-WIDE"
    ag_name = "TabPFNWide"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        try:
            import tabpfnwide  # noqa: F401  (lazy availability check)
        except ImportError as exc:
            raise ImportError(
                "TabPFNWideModel requires the tabpfnwide package. Install with: pip install tabpfnwide",
            ) from exc

        import torch

        if num_gpus and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            if num_gpus:
                logger.warning(
                    "TabPFN-Wide: GPU requested but CUDA unavailable; running on CPU (slow)."
                )

        # AutoGluon preprocessing: align features, label-encode categoricals to numeric,
        # and populate self._cat_indices (empty for the all-numeric bio matrices).
        X = self.preprocess(X, y=y, is_train=True)

        params = self._get_model_params()
        base_model = _CloneableTabPFNWide(
            model_name=params.get("model_name", _DEFAULT_WIDE_MODEL),
            device=device,
            n_estimators=params.get("n_estimators", 1),
            features_per_group=params.get("features_per_group", 1),
            categorical_features_indices=self._cat_indices,
            ignore_pretraining_limits=True,
        )

        # TabPFN-Wide (like TabPFN) supports ~10 classes natively. For more, wrap with
        # ManyClassClassifier (ECOC, from tabpfn-extensions) — the same fallback AutoGluon's
        # TabPFN models use. (Capping classes instead is available via the config's
        # ``max_classes``; the two approaches are complementary.)
        many_class_threshold = self.params_aux.get("many_class_threshold", 10)
        if self.num_classes is not None and self.num_classes > many_class_threshold:
            try:
                from tabpfn_extensions.many_class import ManyClassClassifier
            except ImportError as exc:
                raise ImportError(
                    f"TabPFN-Wide: {self.num_classes} classes exceeds the native limit "
                    f"({many_class_threshold}); the ManyClassClassifier (ECOC) fallback requires "
                    "tabpfn-extensions (pip install tabpfn-extensions). Alternatively cap classes "
                    "via 'max_classes' in the config.",
                ) from exc
            logger.log(
                20,
                f"\tTabPFN-Wide: {self.num_classes} classes exceeds native limit "
                f"({many_class_threshold}); using ManyClassClassifier (ECOC wrapper).",
            )
            self.model = ManyClassClassifier(
                estimator=base_model, alphabet_size=many_class_threshold
            )
        else:
            self.model = base_model

        self.model.fit(X, y)

    def _set_default_params(self) -> None:
        default_params = {
            "model_name": _DEFAULT_WIDE_MODEL,
            "n_estimators": 1,
            "features_per_group": 1,
            "ignore_pretraining_limits": True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass"]

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        return []

    def _log_license(self, device: str) -> None:
        logger.log(20, "\tBuilt with TabPFN-Wide (PriorLabs-TabPFN derivative)")
