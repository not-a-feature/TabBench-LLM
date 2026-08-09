"""Network-free unit tests for the feature-cap helper."""

from __future__ import annotations

import pandas as pd

from tabbench_llm.data.adapter import cap_features


def test_cap_features_random_subset_in_original_order():
    X = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in "abcde"})
    out = cap_features(X, max_features=2, random_state=0)
    # Exactly max_features columns, a subset of the originals kept in left-to-right order.
    assert out.shape == (3, 2)
    kept = set(out.columns)
    assert kept <= set(X.columns)
    assert list(out.columns) == [c for c in X.columns if c in kept]


def test_cap_features_random_subset_deterministic_in_seed():
    X = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in "abcdefghij"})
    assert list(cap_features(X, max_features=4, random_state=7).columns) == list(
        cap_features(X, max_features=4, random_state=7).columns
    )


def test_cap_features_noop_when_within_cap():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = cap_features(X, max_features=5)
    assert out is X
    assert list(out.columns) == ["a", "b"]


def test_cap_features_none_keeps_all():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 9.0], "c": [0.0, 1.0, 0.0]})
    out = cap_features(X, max_features=None)
    assert out is X
    assert list(out.columns) == ["a", "b", "c"]
