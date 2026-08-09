"""Unit tests for the in-context LLM classifier (network-free, client stubbed)."""

from __future__ import annotations

import pathlib
import re
import time

import httpx
import openai
import pandas as pd
import pytest

import tabbench_llm.llm.model as llm_model
from tabbench_llm.dataset import TaskType
from tabbench_llm.llm import LLM_MODELS, is_llm_model

_MODELS_DIR = pathlib.Path(llm_model.__file__).parent / "data"


class _FakeClient:
    """Chat client that returns a fixed completion string for every request."""

    def __init__(self, reply: str):
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.last_kwargs = kwargs

                class _M:
                    content = reply
                    finish_reason = "stop"

                class _C:
                    message = _M()
                    finish_reason = "stop"

                class _R:
                    choices = [_C()]
                    model = ""  # empty -> identity guard cannot verify, so it allows it

                return _R()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _SlowClient:
    """Chat client that sleeps before every reply, to exercise the per-cell wall-clock cap."""

    def __init__(self, reply: str, delay: float):
        class _Completions:
            def create(self, **kwargs):
                time.sleep(delay)

                class _M:
                    content = reply
                    finish_reason = "stop"

                class _C:
                    message = _M()
                    finish_reason = "stop"

                class _R:
                    choices = [_C()]
                    model = ""

                return _R()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_registry_and_router():
    assert is_llm_model("GEMMA") and is_llm_model("QWEN")
    assert not is_llm_model("RF")
    assert LLM_MODELS["GEMMA"]["api_model"].startswith("google/")
    # Only endpoints that can serve a whole grid ship in the default registry: the ml-cloud
    # proxy and the self-hosted Ollama systems. The rate-limited third-party ones are parked.
    assert {v["provider"] for v in LLM_MODELS.values()} == {"ml-cloud", "ollama"}
    assert LLM_MODELS["OLLAMA-GEMMA3-4B"]["context_window"] == 32768
    assert LLM_MODELS["OLLAMA-QWEN3-8B"]["api_model"] == "tabarena-qwen3-8b"
    assert LLM_MODELS["OLLAMA-LLAMA3.1-8B"]["api_model"] == "tabarena-llama31-8b"
    assert LLM_MODELS["OLLAMA-MISTRAL-7B"]["context_window"] == 32768
    assert LLM_MODELS["OLLAMA-PHI4-MINI"]["reasoning"] is False
    assert LLM_MODELS["LOCAL-QWEN3-32B-FP8"]["ollama_source"] == "qwen3:32b-q8_0"
    assert LLM_MODELS["LOCAL-MISTRAL-7B-FP16"]["ollama_source"] == "mistral:7b-instruct-v0.3-fp16"
    assert LLM_MODELS["LOCAL-MISTRAL-SMALL-3.1-24B-FP16"]["parameters_b"] == 24


def test_parked_external_models_reload_on_demand(monkeypatch):
    # The paid endpoints are parked, not deleted: naming their file in $TABBENCH_LLM_EXTRA_MODELS
    # brings them back with no code change.
    from tabbench_llm.llm import registry

    external = _MODELS_DIR / "llm_models_external.json"
    assert external.is_file()
    assert not any(k in LLM_MODELS for k in ("GEMINI-3.5-FLASH", "MISTRAL-MEDIUM"))

    monkeypatch.setenv("TABBENCH_LLM_EXTRA_MODELS", str(external))
    merged = registry._load_registry()
    assert merged["GEMINI-3.5-FLASH"]["provider"] == "google"
    assert merged["MISTRAL-MEDIUM"]["provider"] == "nvidia-nim"
    assert merged["LLAMA-3.1-8B"]["provider"] == "groq"
    assert merged["NEMOTRON-NANO-30B"]["provider"] == "openrouter"
    assert merged["GEMMA"] == LLM_MODELS["GEMMA"]  # bundled entries survive the merge


def test_missing_extra_models_file_fails_loudly(monkeypatch):
    from tabbench_llm.llm import registry

    monkeypatch.setenv("TABBENCH_LLM_EXTRA_MODELS", "does/not/exist.json")
    with pytest.raises(AssertionError):
        registry._load_registry()


def test_parse_word_boundary():
    m = llm_model.LLMModel.__new__(llm_model.LLMModel)
    m._prompt_classes, m._prompt_majority, m._n_unparsed = ["A", "B"], "A", 0
    assert m._parse("The class is B") == "B"  # 'a' in "class" must not match A
    assert m._parse("answer: A") == "A"
    m._prompt_classes, m._prompt_majority = ["FUNC", "LOF", "INT"], "FUNC"
    assert m._parse("likely LOF here") == "LOF"
    assert m._parse("no idea") == "FUNC"  # falls back to majority
    assert m._n_unparsed == 1


def test_parse_batch_alignment():
    m = llm_model.LLMModel.__new__(llm_model.LLMModel)
    m._prompt_classes = ["No", "Yes"]
    m.elicit_proba = False
    text = "reasoning...\nRow 1: Yes\nRow 2: No\n3) Yes"
    # elicit off -> each entry is (class, None); the probability slot is unused.
    assert m._parse_batch(text, 3) == {1: ("Yes", None), 2: ("No", None), 3: ("Yes", None)}


def test_hide_labels_encode_decode(monkeypatch):
    # Hidden mode: the prompt carries opaque C0/C1 tokens (not the real labels), and the
    # model's token answer is decoded back to the real class for scoring.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: C1\nRow 2: C0"))
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "Yes", "No", "Yes"]})
    test = pd.DataFrame({"f1": [1.5, 3.5], "target": ["No", "No"]}, index=[7, 9])
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, hide_labels=True, test_batch_size=2)
    m.fit(train)
    assert m._enc == {"No": "C0", "Yes": "C1"}
    pred = m.predict(test)
    prompt = m._client.last_kwargs["messages"][1]["content"]
    assert "=> C0" in prompt and "=> Yes" not in prompt  # labels hidden in the prompt
    assert list(pred) == ["Yes", "No"]  # C1/C0 decoded back to real labels
    assert list(m.predict_proba(test).columns) == ["No", "Yes"]


def test_fit_predict_batched_transductive(monkeypatch):
    # test_batch_size=2 is the opt-in transductive path: both test rows go into one request and
    # one Row line comes back per row. (The default is 1, i.e. one request per row - see
    # test_default_is_inductive_one_row_per_request.)
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: Yes\nRow 2: No"))
    train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [0.1, 0.2, 0.3, 0.4],
            "target": ["No", "Yes", "No", "Yes"],
        }
    )
    test = pd.DataFrame(
        {"f1": [1.5, 3.5], "f2": [0.15, 0.35], "target": ["No", "No"]}, index=[7, 9]
    )

    m = llm_model.LLMModel("GEMMA", TaskType.Classification, prompt_index=1, test_batch_size=2)
    m.fit(train)
    pred = m.predict(test)
    proba = m.predict_proba(test)

    assert pred.name == "target"
    assert list(pred.index) == [7, 9]
    assert list(pred) == ["Yes", "No"]  # aligned to Row 1 / Row 2
    assert list(proba.columns) == ["No", "Yes"]
    assert proba.loc[7, "Yes"] == 1.0 and proba.loc[7, "No"] == 0.0
    # One call for the whole 2-row batch, reused by predict + predict_proba.
    assert m._n_api_calls == 1
    # interface parity with AutoGluonModel for predictions._cleanup_model
    assert m.autogluon_path is None and m.predictor is None


def test_sampling_seed_and_provenance_are_recorded(monkeypatch):
    fake = _FakeClient("Row 1: Yes")
    monkeypatch.setattr(llm_model, "build_client", lambda *_: fake)
    train = pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]})
    test = pd.DataFrame({"f1": [3], "target": ["No"]})

    m = llm_model.LLMModel(
        "OLLAMA-QWEN3-8B",
        TaskType.Classification,
        seed=123,
        reasoning_effort="none",
        max_workers=1,
    )
    m.fit(train)
    m.predict(test)

    assert fake.last_kwargs["seed"] == 123
    stats = m.get_fit_stats()
    assert stats["llm_seed"] == 123
    assert stats["llm_provider"] == "ollama"
    assert stats["llm_api_model"] == "tabarena-qwen3-8b"
    assert len(stats["llm_system_prompt_sha256"]) == 64


def test_missing_rows_fall_back_to_majority(monkeypatch):
    # Model returns only Row 1; the omitted row 2 falls back to the majority class.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: Yes"))
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "No", "No", "Yes"]})  # majority No
    test = pd.DataFrame({"f1": [5, 6], "target": ["No", "No"]}, index=[0, 1])
    m = llm_model.LLMModel(
        "QWEN", TaskType.Classification, test_batch_size=2, max_unparsed_frac=1.0
    )
    m.fit(train)
    pred = m.predict(test)
    assert list(pred) == ["Yes", "No"]  # row 2 omitted -> majority "No"
    assert m._n_unparsed == 1


def test_parallel_chunks(monkeypatch):
    # test_batch_size=1 -> one 1-row chunk per test row, run across a thread pool.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: Yes"))
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "No", "No", "Yes"]})
    test = pd.DataFrame({"f1": [5, 6, 7], "target": ["No", "No", "No"]}, index=[10, 20, 30])
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, test_batch_size=1, max_workers=4)
    m.fit(train)
    pred = m.predict(test)
    assert list(pred.index) == [10, 20, 30]  # order preserved across parallel chunks
    assert list(pred) == ["Yes", "Yes", "Yes"]
    assert m._n_api_calls == 3  # one call per 1-row chunk


class _CtxSplitClient:
    """Rejects any request carrying more than one test row with a context-window 400, and
    answers single-row requests — exercising the auto-split path in ``_predict_chunk``."""

    def __init__(self):
        class _Completions:
            def create(self, **kwargs):
                msg = kwargs["messages"][1]["content"]
                n = int(re.search(r"Test table \((\d+) row", msg).group(1))
                if n > 1:
                    raise openai.BadRequestError(
                        "maximum context length is 100 tokens ... input_tokens",
                        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                        body=None,
                    )

                class _M:
                    content = "Row 1: Yes"
                    finish_reason = "stop"

                class _C:
                    message = _M()
                    finish_reason = "stop"

                class _R:
                    choices = [_C()]
                    model = ""

                return _R()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_over_context_chunk_is_split(monkeypatch):
    # A 2-row batch overflows; it is halved into two 1-row calls that succeed.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _CtxSplitClient())
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "No", "No", "Yes"]})
    test = pd.DataFrame({"f1": [5, 6], "target": ["No", "No"]}, index=[3, 8])
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, test_batch_size=2, max_workers=1)
    m.fit(train)
    pred = m.predict(test)
    assert list(pred.index) == [3, 8]
    assert list(pred) == ["Yes", "Yes"]
    assert m._n_api_calls == 2  # two single-row calls after the split


def test_single_row_over_context_propagates(monkeypatch):
    # A lone row that still overflows cannot be split further -> the error propagates.
    def _always_ctx(*_):
        class _Completions:
            def create(self, **kwargs):
                raise openai.BadRequestError(
                    "maximum context length is 100 tokens ... input_tokens",
                    response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                    body=None,
                )

        return type("_Cl", (), {"chat": type("_Ch", (), {"completions": _Completions()})()})()

    monkeypatch.setattr(llm_model, "build_client", _always_ctx)
    train = pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]})
    test = pd.DataFrame({"f1": [5], "target": ["No"]}, index=[0])
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, test_batch_size=1, max_workers=1)
    m.fit(train)
    with pytest.raises(openai.BadRequestError):
        m.predict(test)


def test_reasoning_control_extra_body(monkeypatch):
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: No"))
    train = pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]})
    test = pd.DataFrame({"f1": [3]}, index=[0])
    test["target"] = "No"

    def _extra_body(**kw):
        m = llm_model.LLMModel("QWEN", TaskType.Classification, **kw)
        m.fit(train)
        m.predict(test)
        return m._client.last_kwargs.get("extra_body")

    assert _extra_body(thinking=True) is None  # full reasoning
    assert _extra_body(thinking=False) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert _extra_body(reasoning_effort="low") == {"reasoning_effort": "low"}
    # explicit effort wins over thinking
    assert _extra_body(thinking=True, reasoning_effort="none") == {"reasoning_effort": "none"}


def test_reasoning_effort_snaps_to_supported(monkeypatch):
    # Some endpoints expose only part of the none<low<medium<high scale, so a uniform grid
    # setting must snap to the nearest supported value rather than 400. Mistral (none/high) is
    # parked, so this loads it through the extra-models path.
    from tabbench_llm.llm import registry

    monkeypatch.setenv("TABBENCH_LLM_EXTRA_MODELS", str(_MODELS_DIR / "llm_models_external.json"))
    monkeypatch.setattr(llm_model, "LLM_MODELS", registry._load_registry())
    m = llm_model.LLMModel("MISTRAL-MEDIUM", TaskType.Classification, reasoning_effort="low")
    assert m.reasoning_effort == "none" and m._reasoning_on is False
    # A model declaring no subset keeps the requested value.
    assert (
        llm_model.LLMModel(
            "GEMMA", TaskType.Classification, reasoning_effort="low"
        ).reasoning_effort
        == "low"
    )


def test_non_reasoning_model_ignores_reasoning_axis(monkeypatch):
    # A registry "reasoning": false model rejects the reasoning_effort parameter (and has no
    # enable_thinking chat-template kwarg), so it must never send reasoning controls even when
    # the grid sets the reasoning-effort env.
    monkeypatch.setenv("TABBENCH_LLM_REASONING_EFFORT", "medium")
    fake = _FakeClient("Row 1: Yes")
    monkeypatch.setattr(llm_model, "build_client", lambda *_: fake)
    m = llm_model.LLMModel("LOCAL-GEMMA-3-27B", TaskType.Classification)
    assert m.supports_reasoning is False and m._reasoning_on is False and m.reasoning_effort is None
    m.fit(pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]}))
    m.predict(pd.DataFrame({"f1": [1.5], "target": ["No"]}))
    assert fake.last_kwargs["extra_body"] is None


def test_cell_timeout_raises(monkeypatch):
    # A model too slow to finish a cell within cell_timeout_s aborts with CellTimeout, so the
    # caller records a skip rather than letting one straggler stall the whole grid cell.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _SlowClient("Row 1: Yes", delay=0.3))
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "Yes", "No", "Yes"]})
    test = pd.DataFrame({"f1": [1.5, 2.5, 3.5, 4.5], "target": ["No", "No", "No", "No"]})
    m = llm_model.LLMModel(
        "GEMMA",
        TaskType.Classification,
        test_batch_size=1,
        max_workers=4,
        cell_timeout_s=0.05,
    )
    m.fit(train)
    with pytest.raises(llm_model.CellTimeout):
        m.predict(test)


def test_cell_timeout_disabled_completes(monkeypatch):
    # cell_timeout_s=0 (the default) disables the cap: a slow client still completes normally.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _SlowClient("Row 1: Yes", delay=0.02))
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "Yes", "No", "Yes"]})
    test = pd.DataFrame({"f1": [1.5, 2.5], "target": ["No", "No"]})
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, test_batch_size=1, max_workers=2)
    m.fit(train)
    assert list(m.predict(test)) == ["Yes", "Yes"]


def test_elicit_proba_builds_soft_distribution(monkeypatch):
    # elicit_proba on: the predicted class keeps the reported p=, the rest share the remainder.
    monkeypatch.setattr(
        llm_model, "build_client", lambda *_: _FakeClient("Row 1: Yes p=0.8\nRow 2: No p=0.9")
    )
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "Yes", "No", "Yes"]})
    test = pd.DataFrame({"f1": [1.5, 3.5], "target": ["No", "No"]}, index=[7, 9])
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, elicit_proba=True, test_batch_size=2)
    m.fit(train)
    pred = m.predict(test)
    proba = m.predict_proba(test)
    assert list(pred) == ["Yes", "No"]
    assert proba.loc[7, "Yes"] == pytest.approx(0.8) and proba.loc[7, "No"] == pytest.approx(0.2)
    assert proba.loc[9, "No"] == pytest.approx(0.9) and proba.loc[9, "Yes"] == pytest.approx(0.1)
    prompt = m._client.last_kwargs["messages"][1]["content"]
    assert "p=<probability>" in prompt  # the request actually asked for a probability


def test_elicit_proba_missing_falls_back_to_one_hot(monkeypatch):
    # elicit on but the answer line omits p= -> one-hot for that row (no crash).
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: Yes"))
    train = pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]})
    test = pd.DataFrame({"f1": [3]}, index=[0])
    test["target"] = "No"
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, elicit_proba=True)
    m.fit(train)
    proba = m.predict_proba(test)
    assert proba.loc[0, "Yes"] == 1.0 and proba.loc[0, "No"] == 0.0


def test_default_proba_is_one_hot(monkeypatch):
    # Default (elicit off): p= in the reply is ignored, proba stays one-hot.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: Yes p=0.7"))
    train = pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]})
    test = pd.DataFrame({"f1": [3]}, index=[0])
    test["target"] = "No"
    m = llm_model.LLMModel("GEMMA", TaskType.Classification)
    m.fit(train)
    proba = m.predict_proba(test)
    assert proba.loc[0, "Yes"] == 1.0 and proba.loc[0, "No"] == 0.0


def test_context_skip_reason(monkeypatch):
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "Yes", "No", "Yes"]})
    test = pd.DataFrame({"f1": [5, 6], "target": ["No", "No"]}, index=[0, 1])

    # A tiny window can't hold the prompt -> a skip reason is returned (no request made).
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, context_window=10)
    m.fit(train)
    reason = m.context_skip_reason(test)
    assert reason is not None and "context window" in reason and "GEMMA" in reason

    # A generous window fits -> None (the cell runs).
    m2 = llm_model.LLMModel("GEMMA", TaskType.Classification, context_window=100_000)
    m2.fit(train)
    assert m2.context_skip_reason(test) is None


def test_context_window_from_registry_and_env(monkeypatch):
    # Registry default is picked up; env overrides it.
    assert llm_model.LLMModel("GEMMA", TaskType.Classification).context_window == 262144
    monkeypatch.setenv("TABBENCH_LLM_CONTEXT_WINDOW", "2048")
    assert llm_model.LLMModel("GEMMA", TaskType.Classification).context_window == 2048


def test_no_context_window_disables_skip(monkeypatch):
    # No configured window -> pre-skip disabled (reason is always None).
    monkeypatch.delenv("TABBENCH_LLM_CONTEXT_WINDOW", raising=False)
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, context_window=0)
    m.context_window = None  # simulate a model with no window configured anywhere
    m.fit(pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]}))
    assert m.context_skip_reason(pd.DataFrame({"f1": [3]}, index=[0])) is None


def test_regression_rejected():
    with pytest.raises(AssertionError):
        llm_model.LLMModel("GEMMA", TaskType.Regression)


# ---------------------------------------------------------------------------
# Inductive default + unparsed-response ceiling
# ---------------------------------------------------------------------------


def test_default_is_inductive_one_row_per_request(monkeypatch):
    # The default must be one test row per request. Batching test rows would let the model
    # condition on other unlabeled test rows, which is a transductive advantage the trained
    # baselines do not have, so it cannot be the silent default.
    monkeypatch.delenv("TABBENCH_LLM_BATCH_SIZE", raising=False)
    fake = _FakeClient("Row 1: Yes")
    monkeypatch.setattr(llm_model, "build_client", lambda *_: fake)
    m = llm_model.LLMModel("GEMMA", TaskType.Classification)
    assert m.test_batch_size == 1

    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "Yes", "No", "Yes"]})
    test = pd.DataFrame({"f1": [5, 6, 7], "target": ["No", "No", "No"]})
    m.fit(train)
    m.predict(test)
    assert m._n_api_calls == 3  # one request per test row
    prompt = fake.last_kwargs["messages"][1]["content"]
    assert "Test table (1 row(s) to classify)" in prompt


def test_settings_recorded_in_fit_stats(monkeypatch):
    # The transductive/inductive choice and the proba mode are persisted per cell, so a result
    # never has to be interpreted against an unrecorded environment variable.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("Row 1: Yes"))
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, test_batch_size=7, elicit_proba=True)
    stats = m.get_fit_stats()
    assert stats["llm_test_batch_size"] == 7 and stats["llm_elicit_proba"] is True


def test_too_many_unparsed_rows_refuses_the_cell(monkeypatch):
    # Unparsed rows silently become majority-class predictions, which score like any other
    # prediction. Past the ceiling the vector must be refused instead of reported.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("no answer here"))
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "No", "No", "Yes"]})
    test = pd.DataFrame({"f1": [5, 6, 7, 8], "target": ["No"] * 4})
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, max_workers=1)
    m.fit(train)
    with pytest.raises(llm_model.UnparseableResponses, match="4/4"):
        m.predict(test)


def test_unparsed_below_ceiling_still_predicts_and_counts(monkeypatch):
    # Under the ceiling the cell still reports — the fallback is a documented behaviour, not an
    # error — but the count must survive in the stats so the score carries its caveat.
    monkeypatch.setattr(llm_model, "build_client", lambda *_: _FakeClient("no answer here"))
    train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["No", "No", "No", "Yes"]})
    test = pd.DataFrame({"f1": [5, 6, 7, 8], "target": ["No"] * 4})
    m = llm_model.LLMModel("GEMMA", TaskType.Classification, max_workers=1, max_unparsed_frac=1.0)
    m.fit(train)
    assert list(m.predict(test)) == ["No"] * 4  # all majority-class fallbacks
    assert m.get_fit_stats()["llm_unparsed"] == 4


# ---------------------------------------------------------------------------
# Self-hosted cluster Ollama systems
# ---------------------------------------------------------------------------


def _cluster_ollama_keys():
    return [key for key in LLM_MODELS if key.startswith("LOCAL-")]


def test_registry_comment_keys_are_not_models():
    # Underscore-prefixed keys in llm_models.json are file comments, not runnable models.
    assert not any(k.startswith("_") for k in LLM_MODELS)
    assert not is_llm_model("_local_note")


@pytest.mark.parametrize(
    ("key", "filename", "source", "alias"),
    [
        ("OLLAMA-GEMMA3-4B", "Modelfile.gemma3-4b", "gemma3:4b", "tabarena-gemma3-4b"),
        ("OLLAMA-QWEN3-8B", "Modelfile.qwen3-8b", "qwen3:8b", "tabarena-qwen3-8b"),
        (
            "OLLAMA-LLAMA3.1-8B",
            "Modelfile.llama3.1-8b",
            "llama3.1:8b",
            "tabarena-llama31-8b",
        ),
        (
            "OLLAMA-MISTRAL-7B",
            "Modelfile.mistral-7b",
            "mistral:7b-instruct",
            "tabarena-mistral-7b",
        ),
        (
            "OLLAMA-PHI4-MINI",
            "Modelfile.phi4-mini",
            "phi4-mini:3.8b",
            "tabarena-phi4-mini",
        ),
    ],
)
def test_ollama_modelfile_matches_registry(key, filename, source, alias):
    text = (pathlib.Path(__file__).parents[1] / "configs" / "ollama" / filename).read_text()
    assert f"FROM {source}" in text
    assert f"PARAMETER num_ctx {LLM_MODELS[key]['context_window']}" in text
    assert LLM_MODELS[key]["api_model"] == alias


@pytest.mark.parametrize("key", _cluster_ollama_keys())
def test_cluster_ollama_entry_has_reproducible_alias_and_source(key):
    entry = LLM_MODELS[key]
    assert entry["provider"] == "ollama"
    assert entry["api_model"].startswith("tabarena-h100-")
    assert entry["ollama_source"]
    assert entry["context_window"] == 32768


def test_cluster_ollama_provider_maps_reasoning_axis(monkeypatch):
    # Ollama's OpenAI-compatible endpoint accepts reasoning_effort directly.
    fake = _FakeClient("Row 1: No")
    monkeypatch.setattr(llm_model, "build_client", lambda *_: fake)
    train = pd.DataFrame({"f1": [1, 2], "target": ["No", "Yes"]})
    test = pd.DataFrame({"f1": [3], "target": ["No"]}, index=[0])

    def _extra_body(effort):
        m = llm_model.LLMModel("LOCAL-QWEN3-8B", TaskType.Classification, reasoning_effort=effort)
        m.fit(train)
        m.predict(test)
        return fake.last_kwargs["extra_body"]

    assert _extra_body("none") == {"reasoning_effort": "none"}
    assert _extra_body("medium") == {"reasoning_effort": "medium"}

    # A local non-reasoning instruct model still sends nothing — its chat template has no
    # enable_thinking kwarg and would raise on one.
    m = llm_model.LLMModel("LOCAL-LLAMA-3.1-8B", TaskType.Classification, reasoning_effort="medium")
    m.fit(train)
    m.predict(test)
    assert fake.last_kwargs["extra_body"] is None


def test_env_endpoint_overrides_key_file(monkeypatch, tmp_path):
    # A self-hosted server's host:port is only known once the job has started it, so the
    # endpoint comes from the environment and needs no entry in the credentials file.
    from tabbench_llm.llm import client

    monkeypatch.setenv("TABBENCH_LLM_KEY", str(tmp_path / "does_not_exist.json"))
    monkeypatch.delenv("TABBENCH_LLM_OLLAMA_BASE_URL", raising=False)
    assert client.env_endpoint("ollama") is None

    monkeypatch.setenv("TABBENCH_LLM_OLLAMA_BASE_URL", "http://127.0.0.1:8123/v1")
    assert client.env_endpoint("ollama") == {
        "api_key": "EMPTY",
        "base_url": "http://127.0.0.1:8123/v1",
    }
    # Built without touching the (absent) key file.
    assert str(client.build_client("ollama").base_url).startswith("http://127.0.0.1:8123")
