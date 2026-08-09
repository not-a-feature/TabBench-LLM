"""Tests for the synthetic-headline grid payload."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabbench_llm.site import (
    _CLF_COLS,
    _OVERALL_COLS,
    _REG_COLS,
    CATEGORY_COLOR,
    _build_grid,
    _category_of,
    _display_of,
    _model_size_accuracy_figdata,
)


def test_repository_ships_data_not_a_page():
    """The page lives in the juleskreuer.eu repository; this one owns the numbers."""
    root = Path(__file__).parents[1]

    assert not (root / "docs" / "index.html").exists()
    assert not (root / "docs" / "js").exists()


def test_llm_categories_and_colours_follow_model_family():
    expected = {
        "GEMMA": ("Gemma", "#4285f4"),
        "OLLAMA-GEMMA3-4B": ("Gemma", "#4285f4"),
        "NEMOTRON-NANO-30B": ("Nemotron", "#16a34a"),
        "MISTRAL-MEDIUM": ("Mistral", "#c2410c"),
        "OLLAMA-MISTRAL-7B": ("Mistral", "#c2410c"),
        "LOCAL-QWEN3-32B-FP8": ("Qwen", "#7c3aed"),
        "LOCAL-LLAMA-3.1-8B": ("Llama", "#e11d48"),
        "LOCAL-MISTRAL-7B-FP16": ("Mistral", "#c2410c"),
        "OLLAMA-PHI4-MINI": ("Phi", "#0891b2"),
        "GEMINI-3.5-FLASH": ("Gemini", "#0f766e"),
    }

    for model_id, (family, colour) in expected.items():
        assert _category_of(model_id) == family
        assert CATEGORY_COLOR[family] == colour

    assert _category_of("RF") == "Tree-based"
    assert _category_of("NOT-A-MODEL") == "Other"


def test_mlcloud_models_have_descriptive_display_names():
    assert _display_of("GEMMA") == "Gemma 4 31B (MLCloud)"
    assert _display_of("QWEN") == "Qwen 3.6 35B-A3B (MLCloud)"


def test_leaderboard_hides_operational_columns():
    hidden = {"# Failed", "# Skipped", "# Retry", "Train Time s", "Infer. s/1K", "Peak Mem MB"}
    for columns in (_OVERALL_COLS, _CLF_COLS, _REG_COLS):
        assert hidden.isdisjoint(column["key"] for column in columns)


def test_grid_defaults_to_synthetic_opaque_headline(tmp_path):
    rows = []
    datasets = {
        "synthetic_linear": ["hidden"],
        "synthetic_xor": ["hidden"],
        "credit-g": ["visible", "hidden"],
        "bank-marketing": ["visible", "hidden"],
    }
    for dataset, modes in datasets.items():
        for labels in modes:
            for model, score in (("RF", 0.65), ("OLLAMA-QWEN3-8B", 0.70)):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "reasoning": "off",
                        "labels": labels,
                        "feature_cap": "full",
                        "n_train": 100,
                        "f1_macro__mean": score,
                    }
                )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(rows).to_csv(data_dir / "feature_grid_summary.csv", index=False)

    payload = _build_grid(str(tmp_path))

    assert payload is not None
    assert payload["default_domain"] == "synthetic"
    assert payload["default_reason"] == "off"
    assert payload["default_label"] == "hidden"
    assert payload["default_samples"] == "100"
    assert payload["default_samples_by_domain"]["real"] == "100"
    assert payload["label_modes_by_domain"]["synthetic"] == ["hidden"]
    assert payload["label_modes_by_domain"]["real"] == ["visible", "hidden"]
    assert payload["surface_label_modes"]["synthetic_linear"] == ["hidden"]
    assert any(key.endswith("|synthetic") for key in payload["elo"])
    assert any(key.endswith("|real") for key in payload["elo"])


def test_real_grid_gets_an_independent_available_sample_default(tmp_path):
    rows = []
    for dataset, labels, n_train in (
        ("synthetic_linear", "hidden", 100),
        ("credit-g", "visible", 50),
        ("credit-g", "hidden", 50),
    ):
        for model, score in (("RF", 0.65), ("OLLAMA-QWEN3-8B", 0.70)):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "reasoning": "off",
                    "labels": labels,
                    "feature_cap": "full",
                    "n_train": n_train,
                    "f1_macro__mean": score,
                }
            )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(rows).to_csv(data_dir / "feature_grid_summary.csv", index=False)

    payload = _build_grid(str(tmp_path))

    # The synthetic headline has no n=50 cell, so the default falls back to one that exists.
    assert payload["default_samples"] == "100"
    assert payload["default_samples_by_domain"] == {"synthetic": "100", "real": "50"}


def test_grid_opens_on_the_ten_shot_cell_when_available(tmp_path):
    rows = [
        {
            "dataset": dataset,
            "model": model,
            "reasoning": "off",
            "labels": "hidden",
            "feature_cap": "full",
            "n_train": n_train,
            "f1_macro__mean": score,
        }
        for dataset in ("synthetic_linear", "synthetic_xor")
        for n_train in (10, 50, 100)
        for model, score in (("RF", 0.65), ("OLLAMA-QWEN3-8B", 0.70))
    ]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(rows).to_csv(data_dir / "feature_grid_summary.csv", index=False)

    payload = _build_grid(str(tmp_path))

    assert payload["default_samples"] == "10"
    assert payload["default_samples_by_domain"]["synthetic"] == "10"


def test_model_size_accuracy_marks_only_non_dominated_models():
    classification = pd.DataFrame(
        [
            {
                "model_id": "OLLAMA-PHI4-MINI",
                "Model": "Phi",
                "Category": "Phi",
                "Macro-F1": 0.60,
            },
            {
                "model_id": "OLLAMA-MISTRAL-7B",
                "Model": "Mistral",
                "Category": "Mistral",
                "Macro-F1": 0.58,
            },
            {
                "model_id": "OLLAMA-QWEN3-8B",
                "Model": "Qwen",
                "Category": "Qwen",
                "Macro-F1": 0.72,
            },
            {
                "model_id": "RF",
                "Model": "RF",
                "Category": "Tree-based",
                "Macro-F1": 0.80,
            },
        ]
    )

    figure = _model_size_accuracy_figdata(classification)

    assert figure is not None
    points = {point["model_id"]: point for point in figure["points"]}
    assert "RF" not in points
    assert points["OLLAMA-PHI4-MINI"]["pareto"] is True
    assert points["OLLAMA-MISTRAL-7B"]["pareto"] is False
    assert points["OLLAMA-QWEN3-8B"]["pareto"] is True
