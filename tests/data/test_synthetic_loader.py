"""Tests for the synthetic (contamination-free) dataset loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabbench_llm.data.datasets import DatasetSpec, get_spec, is_known_dataset
from tabbench_llm.data.loaders import get_loader
from tabbench_llm.data.loaders.synthetic import _RECIPES, SyntheticLoader


def _spec(recipe: str, problem_type: str = "binary", fetch_id: str | None = None) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=f"synthetic_{recipe}",
        source="synthetic",
        fetch_id=fetch_id or recipe,
        target="target",
        problem_type=problem_type,
    )


_MULTICLASS = {"blobs": 3, "hier": 6}

#: Recipes whose attainable score is capped by their own design (see their docstrings).
_LEARNABILITY_FLOOR = {"noisy": 0.58, "imbalanced": 0.62}


@pytest.mark.parametrize("recipe", sorted(_RECIPES))
def test_recipe_shape_balance_and_names(recipe):
    problem_type = "multiclass" if recipe in _MULTICLASS else "binary"
    raw = SyntheticLoader().fetch(_spec(recipe, problem_type))
    assert len(raw.X) == len(raw.y) and raw.X.shape[1] >= 4
    assert raw.problem_type == problem_type
    # Generic column names only — nothing an LLM could recall from a real-world table.
    assert list(raw.X.columns) == [f"x{i + 1}" for i in range(raw.X.shape[1])]
    counts = raw.y.value_counts()
    # Enough per class to survive min_samples_per_class + 5-fold stratified CV.
    assert counts.min() >= 10
    assert counts.index.nunique() == _MULTICLASS.get(recipe, 2)


@pytest.mark.parametrize("recipe", sorted(_RECIPES))
def test_recipe_is_learnable(recipe):
    """Every prior must carry recoverable signal, at any seed.

    A recipe no model can learn cannot discriminate between models — it would contribute a
    column of chance scores and dilute the leaderboard rather than probe anything. The bar is
    deliberately loose (RandomForest on the full table, well above the few-shot sizes the
    benchmark actually runs). Two recipes carry a lower floor because their ceiling is part of
    the design: `noisy` caps balanced accuracy at 0.75 through 25% label noise, and
    `imbalanced` gives the minority class only ~10% of the rows.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    floor = _LEARNABILITY_FLOOR.get(recipe, 0.70)
    for seed in (0, 1):
        raw = SyntheticLoader().fetch(_spec(recipe, fetch_id=f"{recipe}:{seed}"))
        X = pd.get_dummies(raw.X) if raw.X.dtypes.nunique() > 1 else raw.X
        score = cross_val_score(
            RandomForestClassifier(100, random_state=0),
            X,
            raw.y,
            cv=3,
            scoring="balanced_accuracy",
        ).mean()
        assert score > floor, f"{recipe} (seed {seed}): balanced accuracy {score:.3f} <= {floor}"


def test_mixed_recipe_keeps_categorical_dtypes():
    # The one recipe that exercises the categorical path end to end: the LLM must see
    # "x1=c2" rather than a float code, and AutoGluon must encode the level natively.
    raw = SyntheticLoader().fetch(_spec("mixed"))
    cats = [c for c in raw.X.columns if str(raw.X[c].dtype) == "category"]
    assert len(cats) == 2
    assert all(str(v).startswith(("c", "g")) for v in raw.X[cats[0]].unique())


@pytest.mark.parametrize("recipe", ["wide_sparse", "wide_dense", "wide_nonlinear"])
def test_wide_recipes_have_common_context_dimension(recipe):
    raw = SyntheticLoader().fetch(_spec(recipe))
    assert raw.X.shape[1] == 64
    assert len(raw.X) >= 640


def test_deterministic_same_seed():
    a = SyntheticLoader().fetch(_spec("linear"))
    b = SyntheticLoader().fetch(_spec("linear"))
    assert np.array_equal(a.X.to_numpy(), b.X.to_numpy())
    assert np.array_equal(a.y.to_numpy(), b.y.to_numpy())


def test_seed_override_changes_data():
    a = SyntheticLoader().fetch(_spec("xor"))
    b = SyntheticLoader().fetch(_spec("xor", fetch_id="xor:1"))
    assert not np.array_equal(a.X.to_numpy(), b.X.to_numpy())


def test_unknown_recipe_raises():
    with pytest.raises(AssertionError):
        SyntheticLoader().fetch(_spec("does_not_exist"))


def test_registered_and_dispatched():
    for recipe in sorted(_RECIPES):
        dataset_id = f"synthetic_{recipe}"
        assert is_known_dataset(dataset_id)
        spec = get_spec(dataset_id)
        assert spec.source == "synthetic" and spec.is_curated
        assert isinstance(get_loader(spec), SyntheticLoader)
