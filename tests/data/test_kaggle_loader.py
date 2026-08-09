"""Network-free unit tests for the Kaggle loader's table helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tabbench_llm.data.loaders.kaggle import KaggleLoader, _read_table


def test_read_table_csv(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame({"a": [1, 2], "label": ["x", "y"]}).to_csv(p, index=False)
    df = _read_table(p)
    assert list(df.columns) == ["a", "label"]
    assert len(df) == 2


def test_resolve_table_single(tmp_path):
    (tmp_path / "only.csv").write_text("a,b\n1,2\n")
    assert KaggleLoader()._resolve_table(tmp_path, spec=_FakeSpec()).name == "only.csv"


def test_resolve_table_multiple_requires_data_file(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.csv").write_text("y\n1\n")
    with pytest.raises(ValueError, match="data_file"):
        KaggleLoader()._resolve_table(tmp_path, spec=_FakeSpec())
    # explicit data_file disambiguates
    assert (
        KaggleLoader(data_file="b.csv")._resolve_table(tmp_path, spec=_FakeSpec()).name == "b.csv"
    )


class _FakeSpec:
    dataset_id = "kaggle-test"
