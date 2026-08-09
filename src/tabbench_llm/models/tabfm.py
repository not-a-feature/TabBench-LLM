"""AutoGluon model wrapper for TabFM (Google Research's tabular foundation model).

TabFM (https://github.com/google-research/tabfm) is a zero-shot, in-context tabular
foundation model that supports both classification and regression. It ships as
scikit-learn estimators (``tabfm.TabFMClassifier`` / ``tabfm.TabFMRegressor``) that wrap a
pre-trained backend loaded via ``tabfm.tabfm_v1_0_0_pytorch.load()``; this module adapts
them to AutoGluon's model API so TabFM is selectable from a config like any built-in
model, via the key ``"TABFM"``.

TabFM is not on PyPI; install the PyTorch backend from source
(``pip install 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm'``). The
PyTorch backend shares the torch/CUDA stack already required by the other foundation
models. Both task types are supported (unlike TabPFN-Wide, which is classification only).
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

# TabFM caps features per ensemble member at 500 by default; None lifts that cap for the
# HDLSS omics matrices (the upstream ``max_features_default`` config governs the feature count).
_DEFAULT_MAX_FEATURES = None


def _load_backend(device: str):
    """Load the pre-trained TabFM PyTorch backend on *device*.

    Weights are downloaded from the Hugging Face Hub on first use; ``tabfm``'s ``load``
    caches per ``(model, checkpoint, device)`` so repeated calls (e.g. one per ECOC
    sub-estimator) reuse the loaded model.
    """
    from tabfm import tabfm_v1_0_0_pytorch

    return tabfm_v1_0_0_pytorch.load(device=device)


class _CloneableTabFM(BaseEstimator, ClassifierMixin):
    """sklearn-clone-safe wrapper that builds a ``TabFMClassifier`` lazily in ``fit``.

    ``TabFMClassifier`` carries the heavy pre-trained backend as its ``model`` param, which
    ``sklearn.clone`` (used by ``ManyClassClassifier``'s ECOC) would try to deep-copy. This
    wrapper carries only the lightweight ``device`` / ``max_num_features`` params and
    constructs the real classifier from the (cached) backend at ``fit`` time, delegating
    ``predict``/``predict_proba``.
    """

    def __init__(self, device: str = "cuda", max_num_features: int | None = None) -> None:
        self.device = device
        self.max_num_features = max_num_features

    def fit(self, X, y):
        from tabfm import TabFMClassifier

        self.classes_ = np.unique(y)
        self.model_ = TabFMClassifier(
            model=_load_backend(self.device),
            max_num_features=self.max_num_features,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class TabFMModel(TabPFNModel):
    """AutoGluon wrapper around TabFM's scikit-learn estimators.

    Inherits :class:`TabPFNModel`'s preprocessing (categorical → numeric label encoding),
    GPU/CPU resource detection, and unified predict handling; only ``_fit`` differs — it
    builds a TabFM classifier or regressor (per ``problem_type``) around the pre-trained
    TabFM backend instead of stock TabPFN.
    """

    ag_key = "TABFM"
    ag_name = "TabFM"

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
            import tabfm  # noqa: F401  (lazy availability check)
        except ImportError as exc:
            raise ImportError(
                "TabFMModel requires the tabfm package. Install with: "
                "pip install 'tabfm[pytorch] @ git+https://github.com/google-research/tabfm'",
            ) from exc

        import torch

        if num_gpus and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            if num_gpus:
                logger.warning("TabFM: GPU requested but CUDA unavailable; running on CPU (slow).")

        # AutoGluon preprocessing: align features, label-encode categoricals to numeric,
        # and populate self._cat_indices (empty for the all-numeric bio matrices).
        X = self.preprocess(X, y=y, is_train=True)

        params = self._get_model_params()
        max_num_features = params["max_num_features"]

        if self.problem_type == "regression":
            from tabfm import TabFMRegressor

            self.model = TabFMRegressor(
                model=_load_backend(device),
                max_num_features=max_num_features,
            )
            self.model.fit(X, y)
            return

        base_model = _CloneableTabFM(device=device, max_num_features=max_num_features)

        # TabFM (like TabPFN) supports up to model.max_classes classes natively. For more,
        # wrap with ManyClassClassifier (ECOC, from tabpfn-extensions) — the same fallback
        # AutoGluon's TabPFN models use. (Capping classes instead is available via the
        # config's ``max_classes``; the two approaches are complementary.)
        many_class_threshold = _load_backend(device).max_classes
        if self.num_classes is not None and self.num_classes > many_class_threshold:
            try:
                from tabpfn_extensions.many_class import ManyClassClassifier
            except ImportError as exc:
                raise ImportError(
                    f"TabFM: {self.num_classes} classes exceeds the native limit "
                    f"({many_class_threshold}); the ManyClassClassifier (ECOC) fallback requires "
                    "tabpfn-extensions (pip install tabpfn-extensions). Alternatively cap classes "
                    "via 'max_classes' in the config.",
                ) from exc
            logger.log(
                20,
                f"\tTabFM: {self.num_classes} classes exceeds native limit "
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
            "max_num_features": _DEFAULT_MAX_FEATURES,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        return []

    def _log_license(self, device: str) -> None:
        logger.log(20, "\tBuilt with TabFM (Google Research, Apache-2.0)")
