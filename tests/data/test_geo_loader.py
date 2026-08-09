"""Network-free unit tests for the GEO loader's matrix/target helpers."""

from __future__ import annotations

import pandas as pd

from tabbench_llm.data.loaders.geo import build_expression_matrix, extract_target


class _FakeGSM:
    def __init__(self, table, metadata):
        self.table = table
        self.metadata = metadata


def test_build_expression_matrix_aligns_probes():
    g1 = _FakeGSM(pd.DataFrame({"ID_REF": ["p1", "p2"], "VALUE": [1.0, 2.0]}), {})
    g2 = _FakeGSM(pd.DataFrame({"ID_REF": ["p1", "p2"], "VALUE": ["3.0", "4.0"]}), {})
    X = build_expression_matrix({"S1": g1, "S2": g2})
    assert list(X.index) == ["S1", "S2"]
    assert list(X.columns) == ["p1", "p2"]
    assert X.loc["S2", "p2"] == 4.0  # coerced from str


def test_extract_target_filters_unlabeled_samples():
    g1 = _FakeGSM(None, {"characteristics_ch1": ["subtype: LumB", "age: 50"]})
    g2 = _FakeGSM(None, {"characteristics_ch1": ["age: 60"]})  # no subtype
    y = extract_target({"S1": g1, "S2": g2}, characteristic_key="subtype")
    assert y.to_dict() == {"S1": "LumB"}
