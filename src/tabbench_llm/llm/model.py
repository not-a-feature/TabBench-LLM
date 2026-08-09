"""In-context LLM classifier with a fit/predict API matching the pipeline.

:class:`LLMModel` is the LLM counterpart to
:class:`tabbench_llm.model.AutoGluonModel`: it exposes ``fit`` / ``predict`` /
``predict_proba`` / ``get_fit_stats`` so the prediction step (Step 1) can route a model
key to it with no other change.

It behaves like a regular tabular model: ``fit`` ingests the whole **training table**
(features + labels) and ``predict`` is handed the whole **test table** and returns one
prediction per row. The test table is split into chunks of ``test_batch_size`` rows; each
chunk is one chat-completion request carrying the whole labeled training table plus that
chunk's numbered test rows, and the model returns one ``Row <n>: <label>`` line per row,
parsed back into an aligned prediction vector. Chunk requests are issued concurrently
(``max_workers`` threads) since the API calls are network-bound. Probabilities are one-hot
over the predicted label by default (the endpoint returns a label, not a distribution),
which keeps the downstream probability metrics well-defined; set ``elicit_proba`` to instead
ask the model for a per-row confidence and build a soft distribution from it.

The registered models are reasoning models (chain-of-thought in ``reasoning_content``,
final answer in ``content``), so the completion budget is generous and grows on
truncation. Classification only; regression is out of scope for the LLM path.
"""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from hashlib import sha256

import numpy as np
import openai
import pandas as pd

from tabbench_llm.dataset import TaskType
from tabbench_llm.llm.client import build_client
from tabbench_llm.llm.prompts import SYSTEM_PROMPTS
from tabbench_llm.llm.registry import LLM_MODELS

logger = logging.getLogger(__name__)


class CellTimeout(Exception):  # noqa: N818 - public exception name retained for compatibility
    """Raised when one LLM cell exceeds its wall-clock budget (``TABBENCH_LLM_CELL_TIMEOUT``).

    A slow model on a large test table can hold up the whole grid cell — the predictions step
    does not return until every submitted cell finishes — so a single straggler stalls the
    pipeline for as long as the endpoint crawls. Bounding a cell's total predict time lets the
    caller record a clean skip and move on instead of waiting hours on one model."""


class UnparseableResponses(Exception):  # noqa: N818 - public name retained for compatibility
    """Raised when too many test rows came back without a parseable class label.

    A row the model does not answer falls back to the majority class, which is a *prediction*
    — it scores, it lands in the leaderboard, and on an imbalanced dataset it even scores
    respectably on accuracy. So a cell where the model largely failed to comply is
    indistinguishable from one where it genuinely predicted the majority class, and the
    failure enters the results as a measurement instead of as an error. Above
    ``max_unparsed_frac`` the prediction vector is refused outright: the caller records the
    cell as failed and writes no predictions."""


#: Parses the model's final answer lines, e.g. "Row 3: Yes" / "3) Yes" / "3. Yes".
_ROW_RE = re.compile(r"(?im)^\s*(?:row|line|#)?\s*(\d+)\s*[:).\-]\s*(.+?)\s*$")

#: Parses the optional ``p=<0..1>`` confidence an answer line carries when ``elicit_proba``
#: is on (e.g. "Row 3: Yes p=0.82").
_PROB_RE = re.compile(r"p\s*=\s*(0*\.\d+|1(?:\.0+)?|0|1)", re.IGNORECASE)

#: With reasoning off, tokens reserved for the answer: a base plus one short
#: ``Row n: label`` line per test row. Keeps the request well inside the context window.
_OUTPUT_BASE = 128
_OUTPUT_PER_ROW = 32

#: Chars-per-token used to *estimate* a prompt's token count without a local tokenizer, for
#: the predictive context-window skip. Kept low (conservative) because numeric tables tokenise
#: denser than prose; lower a model's ``context_window`` for extra headroom if needed.
_CHARS_PER_TOKEN = 3.0

#: Ordered reasoning-effort scale; used to snap a requested effort to the nearest value a
#: model actually supports (see the ``reasoning_efforts`` field in ``llm_models.json``).
_EFFORT_ORDER = ["none", "low", "medium", "high"]


def _fmt_value(v) -> str:
    """Compact, LLM-readable rendering of a feature value (missing -> ``?``)."""
    if isinstance(v, float):
        if np.isnan(v):
            return "?"
        if v.is_integer():
            return str(int(v))
        return f"{v:.4g}"
    return str(v)


class LLMModel:
    """Few-shot tabular classifier over an OpenAI-compatible chat endpoint.

    Parameters
    ----------
    model_name : str
        Registered LLM key (see ``llm/data/llm_models.json``); resolved to the API model id.
    task_type : TaskType
        Must be ``Classification``.
    prompt_index : int
        Split index; selects the system prompt (``SYSTEM_PROMPTS[prompt_index % len]``) so
        wording varies across CV folds/repeats.
    temperature : float
        Sampling temperature (default ``0.0`` for deterministic answers).
    max_tokens : int
        Completion budget when reasoning is on: the model spends most of it "thinking"
        before emitting the ``Row n: label`` lines, so it is large by default and doubled
        (up to ``max_tokens_ceiling``) whenever a response is truncated mid-thought. When
        reasoning is off the model emits only one short line per row, so the budget is sized
        to the chunk (``_OUTPUT_BASE + _OUTPUT_PER_ROW * rows``); requesting the full
        ``max_tokens`` there would push a wide table's request past the context window.
    max_tokens_ceiling : int
        Upper bound the budget grows to on repeated truncation.
    request_timeout : int
        Per-request timeout in seconds.
    thinking : bool | None
        Whether the model reasons (chain-of-thought) before answering. ``True`` (default)
        keeps reasoning on; ``False`` disables it via the vLLM ``enable_thinking`` flag —
        the answer comes back in ~2 tokens instead of thousands, i.e. ~50-100x faster (at
        some accuracy cost). ``None`` reads ``$TABBENCH_LLM_THINKING`` (default on;
        ``0``/``false``/``no``/``off`` turn it off). This is the main latency lever.
    reasoning_effort : str | None
        Graded reasoning budget: ``"none"`` (off), ``"low"``, ``"medium"``, ``"high"``.
        When set it takes precedence over ``thinking`` and is sent as the ``reasoning_effort``
        request field — a way to *limit* thinking rather than only on/off (e.g. ``"low"`` ~=
        half the tokens of ``"high"``). ``None`` (default) reads
        ``$TABBENCH_LLM_REASONING_EFFORT``; unset there means full reasoning.
    test_batch_size : int | None
        Rows per API call, i.e. how many test rows share one prompt. **This is not a
        throughput knob — it changes what is being measured.** At the default ``1`` each test
        row is classified in its own request, conditioned only on the training table, which is
        the same inductive setting the trained baselines run in (TabPFN attends from each test
        point to the training set, never between test points). Any larger value makes the
        evaluation *transductive*: the model sees a batch of unlabeled test rows at once and
        can exploit their joint feature distribution and impose a plausible label marginal on
        them, an advantage no baseline has. Larger batches are therefore an explicit ablation
        (and an API-cost lever), not a default. ``None`` reads
        ``$TABBENCH_LLM_BATCH_SIZE`` (default 1).
    max_workers : int | None
        Number of chunk requests issued concurrently (the API calls are network-bound, so a
        thread pool overlaps them). ``None`` reads ``$TABBENCH_LLM_MAX_WORKERS`` (default 8).
    cell_timeout_s : float | None
        Wall-clock budget for classifying one test table. When ``predict`` exceeds it, no new
        chunk requests are collected and :class:`CellTimeout` is raised so the caller records a
        clean skip — a straggler model then can't stall the whole grid cell. ``None`` reads
        ``$TABBENCH_LLM_CELL_TIMEOUT`` (default ``0`` = no timeout). Set it comfortably above a
        single request's ``request_timeout`` so a normal cell is never cut short.
    hide_labels : bool | None
        Whether to hide the semantics of the class labels. ``False`` (default) shows the real
        label strings, so the model can exploit their meaning (e.g. ``malignant``/``benign``).
        ``True`` replaces every class with an opaque token (``C0``, ``C1``, …) in the training
        table and the answer format, so the model must infer the feature→label mapping purely
        in-context; predictions are decoded back to the real labels for scoring. ``None`` reads
        ``$TABBENCH_LLM_HIDE_LABELS`` (default off; ``1``/``true``/``yes``/``on`` turn it on).
    elicit_proba : bool | None
        Whether to ask the model for a class probability, not just a label. ``False`` (default)
        makes ``predict_proba`` one-hot over the predicted class. ``True`` appends ``p=<0..1>``
        to each answer line and builds the probability from it — the predicted class keeps the
        reported confidence (clamped to ``[1/n_classes, 1]``) and the remaining mass is spread
        uniformly over the other classes — so log-loss / ROC-AUC become meaningful rather than
        degenerate one-hot values. Note a self-reported LLM confidence is not a calibrated
        posterior; treat the resulting probability metrics accordingly. Rows whose line omits a
        parseable ``p=`` fall back to one-hot. ``None`` reads ``$TABBENCH_LLM_ELICIT_PROBA``
        (default off; ``1``/``true``/``yes``/``on`` turn it on).
    max_unparsed_frac : float | None
        Fraction of test rows that may come back without a parseable label before the whole
        prediction vector is refused with :class:`UnparseableResponses` (default ``0.2``).
        Unparsed rows silently become majority-class predictions, so without a ceiling a cell
        the model largely failed to answer still scores. ``None`` reads
        ``$TABBENCH_LLM_MAX_UNPARSED``; set it to ``1.0`` to disable the check.
    context_window : int | None
        The model's context window in tokens. Used by :meth:`context_skip_reason` to skip a
        cell up front when even a single-row request would not fit (the whole training table
        rides in every request, so an over-window training block dooms every request). ``None``
        reads ``$TABBENCH_LLM_CONTEXT_WINDOW`` then the registry ``context_window`` field;
        absent everywhere disables the pre-skip. Token counts are estimated from characters (no
        local tokenizer), so configure the window with a little headroom.
    seed : int
        Sampling seed sent to the OpenAI-compatible endpoint and recorded in the cell
        statistics. Temperature zero is not, by itself, a complete reproducibility record.
    """

    def __init__(
        self,
        model_name: str,
        task_type: TaskType = TaskType.Classification,
        prompt_index: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        max_tokens_ceiling: int = 32768,
        request_timeout: int = 600,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        test_batch_size: int | None = None,
        max_workers: int | None = None,
        cell_timeout_s: float | None = None,
        hide_labels: bool | None = None,
        elicit_proba: bool | None = None,
        max_unparsed_frac: float | None = None,
        context_window: int | None = None,
        seed: int = 0,
    ) -> None:
        assert task_type == TaskType.Classification, "LLMModel supports classification only."
        assert (
            model_name in LLM_MODELS
        ), f"{model_name!r} is not a registered LLM model {sorted(LLM_MODELS)}."
        self.model_name = model_name
        model_info = LLM_MODELS[model_name]
        if isinstance(model_info, dict):
            self.api_model = model_info["api_model"]
            self.provider = model_info.get("provider", "ml-cloud")
        else:
            self.api_model = model_info
            self.provider = "ml-cloud"
        self.task_type = task_type
        self.system_prompt = SYSTEM_PROMPTS[prompt_index % len(SYSTEM_PROMPTS)]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_ceiling = max_tokens_ceiling
        self.request_timeout = (
            request_timeout
            if request_timeout is not None
            else int(os.environ.get("TABBENCH_LLM_REQUEST_TIMEOUT", "600"))
        )
        self.seed = int(seed)
        self.thinking = (
            thinking
            if thinking is not None
            else os.environ.get("TABBENCH_LLM_THINKING", "1").lower()
            not in ("0", "false", "no", "off")
        )
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else os.environ.get("TABBENCH_LLM_REASONING_EFFORT") or None
        )
        # Some endpoints only expose a subset of the effort scale (e.g. Mistral: none/high, no
        # low). When the model declares its supported efforts, snap the requested one to the
        # nearest supported value on the none<low<medium<high scale so a uniform grid setting
        # still runs (rather than 400-ing on an unsupported effort).
        if (
            self.reasoning_effort is not None
            and isinstance(model_info, dict)
            and "reasoning_efforts" in model_info
            and self.reasoning_effort not in model_info["reasoning_efforts"]
        ):
            want = _EFFORT_ORDER.index(self.reasoning_effort)
            self.reasoning_effort = min(
                model_info["reasoning_efforts"], key=lambda e: abs(_EFFORT_ORDER.index(e) - want)
            )
        # Effective reasoning state drives the prompt directive: an explicit effort wins
        # ("none" = off), else the thinking flag.
        self._reasoning_on = (
            self.reasoning_effort != "none" if self.reasoning_effort is not None else self.thinking
        )
        # A plain instruct model (registry "reasoning": false) rejects the reasoning_effort /
        # enable_thinking request fields outright, so the grid's reasoning axis is a no-op for it:
        # force reasoning off and never send a reasoning parameter (see _complete).
        self.supports_reasoning = not (
            isinstance(model_info, dict)
            and "reasoning" in model_info
            and model_info["reasoning"] is False
        )
        if not self.supports_reasoning:
            self.reasoning_effort = None
            self.thinking = False
            self._reasoning_on = False
        self.test_batch_size = (
            test_batch_size
            if test_batch_size is not None
            else int(os.environ.get("TABBENCH_LLM_BATCH_SIZE", "1"))
        )
        self.max_workers = (
            max_workers
            if max_workers is not None
            else int(os.environ.get("TABBENCH_LLM_MAX_WORKERS", "8"))
        )
        self.cell_timeout_s = (
            cell_timeout_s
            if cell_timeout_s is not None
            else float(os.environ.get("TABBENCH_LLM_CELL_TIMEOUT", "0"))
        )
        self.hide_labels = (
            hide_labels
            if hide_labels is not None
            else os.environ.get("TABBENCH_LLM_HIDE_LABELS", "0").lower()
            in ("1", "true", "yes", "on")
        )
        self.elicit_proba = (
            elicit_proba
            if elicit_proba is not None
            else os.environ.get("TABBENCH_LLM_ELICIT_PROBA", "0").lower()
            in ("1", "true", "yes", "on")
        )
        self.max_unparsed_frac = (
            max_unparsed_frac
            if max_unparsed_frac is not None
            else float(os.environ.get("TABBENCH_LLM_MAX_UNPARSED", "0.2"))
        )
        # Context window (for the predictive over-window skip): explicit arg, else env, else
        # the registry field, else unknown (pre-skip disabled).
        if context_window is not None:
            self.context_window = context_window
        elif "TABBENCH_LLM_CONTEXT_WINDOW" in os.environ:
            self.context_window = int(os.environ["TABBENCH_LLM_CONTEXT_WINDOW"])
        elif isinstance(model_info, dict) and "context_window" in model_info:
            self.context_window = model_info["context_window"]
        else:
            self.context_window = None

        # Interface parity with AutoGluonModel for predictions._cleanup_model.
        self.autogluon_path = None
        self.predictor = None

        self._client = None
        self._feature_cols: list[str] | None = None
        self._classes: list | None = None
        self._majority = None
        # Prompt-space labels: opaque tokens when hide_labels, else the real classes. Filled in
        # fit alongside the encode/decode maps between real classes and prompt tokens.
        self._prompt_classes: list | None = None
        self._prompt_majority = None
        self._enc: dict = {}
        self._dec: dict = {}
        self._train_block: str | None = None
        self._n_api_calls = 0
        self._n_unparsed = 0
        self._cache: dict[tuple, tuple[pd.Series, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    def fit(self, data_train: pd.DataFrame) -> LLMModel:
        """Ingest the training table: serialise every labeled row as ``feats => label``."""
        label = data_train.columns[-1]
        self._feature_cols = [c for c in data_train.columns if c != label]
        y = data_train[label]
        self._classes = sorted(y.unique(), key=str)
        self._majority = y.value_counts().idxmax()
        # Encode real classes to prompt tokens: opaque C0/C1/… when labels are hidden, else the
        # class strings themselves. The training block and answer format use the prompt tokens;
        # _dec turns the model's token answers back into the real classes.
        if self.hide_labels:
            self._enc = {c: f"C{i}" for i, c in enumerate(self._classes)}
        else:
            self._enc = {c: str(c) for c in self._classes}
        self._dec = {tok: c for c, tok in self._enc.items()}
        self._prompt_classes = [self._enc[c] for c in self._classes]
        self._prompt_majority = self._enc[self._majority]
        self._train_block = "\n".join(
            f"{i}) {self._fmt_row(row)} => {self._enc[row[label]]}"
            for i, (_, row) in enumerate(data_train.iterrows(), 1)
        )
        return self

    def _fmt_row(self, row) -> str:
        return ", ".join(f"{c}={_fmt_value(row[c])}" for c in self._feature_cols)

    def _class_list_str(self) -> str:
        return ", ".join(str(c) for c in self._prompt_classes)

    def _build_user_message(self, data_test: pd.DataFrame) -> str:
        """One message holding the whole labeled train table and the whole test table."""
        test_block = "\n".join(
            f"{i}) {self._fmt_row(row)}"
            for i, (_, row) in enumerate(data_test[self._feature_cols].iterrows(), 1)
        )
        n = len(data_test)
        directive = (
            "Reason step by step, then" if self._reasoning_on else "Without showing your reasoning,"
        )
        if self.elicit_proba:
            fmt = (
                "Row <n>: <class> p=<probability>\n"
                "where <probability> is your confidence that the row belongs to <class>, a "
                "number between 0 and 1."
            )
        else:
            fmt = "Row <n>: <class>"
        return (
            f"Classification task. Feature columns: {', '.join(self._feature_cols)}.\n"
            f"Possible classes: {self._class_list_str()}.\n\n"
            f"Training table (each row ends with its true class after '=>'):\n"
            f"{self._train_block}\n\n"
            f"Test table ({n} row(s) to classify):\n{test_block}\n\n"
            f"{directive} give the predicted class for every test row. On the final lines "
            f"output exactly one prediction per test row, in order, in the format:\n"
            f"{fmt}\n"
            f"Use only the class labels from: {self._class_list_str()}."
        )

    def _match_class(self, text: str):
        """Return the prompt-space label named in ``text`` (exact, case-insensitive, then a
        whole-word occurrence, longest label first), or ``None`` if none matches."""
        t = text.strip()
        for c in self._prompt_classes:
            if t == str(c):
                return c
        low = t.lower()
        for c in self._prompt_classes:
            if low == str(c).lower():
                return c
        for c in sorted(self._prompt_classes, key=lambda c: -len(str(c))):
            if re.search(rf"\b{re.escape(str(c))}\b", text, flags=re.IGNORECASE):
                return c
        return None

    def _parse(self, text: str):
        """Single-label parse (prompt space) with majority fallback (one-row helpers/tests)."""
        cls = self._match_class(text)
        if cls is None:
            self._n_unparsed += 1
            return self._prompt_majority
        return cls

    def _parse_prob(self, text: str) -> float | None:
        """Extract the ``p=<0..1>`` confidence a row line carries, or ``None`` if absent/invalid."""
        m = _PROB_RE.search(text)
        if m is None:
            return None
        v = float(m.group(1))
        return v if 0.0 <= v <= 1.0 else None

    def _parse_batch(self, text: str, n: int) -> dict[int, tuple[object, float | None]]:
        """Map ``Row k: label [p=..]`` lines to ``{1-based row index -> (class, prob)}`` (first
        wins). ``prob`` is the parsed confidence when ``elicit_proba`` is on and the line carries
        a ``p=`` value, else ``None`` (one-hot at scoring time)."""
        out: dict[int, tuple[object, float | None]] = {}
        for m in _ROW_RE.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= n and idx not in out:
                body = m.group(2)
                cls = self._match_class(body)
                if cls is not None:
                    prob = self._parse_prob(body) if self.elicit_proba else None
                    out[idx] = (cls, prob)
        return out

    def _initial_budget(self, n_rows: int) -> int:
        """Completion budget for one chunk. Reasoning on -> the full ``max_tokens`` (grown on
        truncation); reasoning off -> only enough for one short ``Row n: label`` line per row,
        so a wide table's request stays inside the model's context window."""
        if self._reasoning_on:
            return self.max_tokens
        return min(self.max_tokens, _OUTPUT_BASE + _OUTPUT_PER_ROW * n_rows)

    def _verify_model(self, resp) -> None:
        """Refuse a response produced by a different model than requested. The proxy can carry
        server-side fallbacks; a silent substitution would mix models within one result column
        and invalidate the comparison, so a clear mismatch is a hard error, not a quiet accept."""
        returned = (resp.model or "").lower()
        requested = self.api_model.split("/")[-1].lower()
        assert (not returned) or requested in returned or returned in requested, (
            f"Endpoint answered with model {resp.model!r}, but {self.api_model!r} was requested "
            "(server-side fallback/substitution); refusing to use this response."
        )

    def _complete(self, data_test: pd.DataFrame) -> tuple[str, int]:
        """One chat completion for a chunk of test rows; grow the budget on truncation.

        Returns ``(content, n_calls)`` and mutates no shared state, so it is safe to run
        concurrently across chunks (the caller aggregates the counts)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_message(data_test)},
        ]
        # Control reasoning via request flags (the `/no_think` prompt token is ignored by this
        # proxy; these parameters are not). An explicit reasoning_effort wins (none/low/medium/
        # high); else enable_thinking=False turns it off. Off returns the label in ~2 tokens
        # instead of thousands of reasoning tokens.
        if not self.supports_reasoning:
            # Plain instruct model: it has no reasoning controls and 400s on the parameter, so
            # send none — the reasoning axis simply doesn't apply.
            extra_body = None
        elif self.provider == "local":
            # Self-hosted vLLM: `reasoning_effort` is only honoured for a few checkpoint
            # families, whereas the hybrid-thinking models served here (Qwen3-style) take the
            # chat-template flag. Map the whole reasoning axis onto that one flag so the grid's
            # off/on arms mean the same thing locally as they do on the hosted endpoints.
            extra_body = {"chat_template_kwargs": {"enable_thinking": self._reasoning_on}}
        elif self.reasoning_effort is not None:
            extra_body = {"reasoning_effort": self.reasoning_effort}
        elif not self.thinking:
            # vLLM (the ml-cloud proxy) toggles thinking via chat_template_kwargs; the hosted
            # OpenAI-compatible providers (google, nvidia-nim) use reasoning_effort="none".
            if self.provider == "ml-cloud":
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            else:
                extra_body = {"reasoning_effort": "none"}
        else:
            extra_body = None
        budget = self._initial_budget(len(data_test))
        n_calls = 0
        while True:
            resp = self._client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=budget,
                timeout=self.request_timeout,
                extra_body=extra_body,
            )
            n_calls += 1
            self._verify_model(resp)
            choice = resp.choices[0]
            content = choice.message.content or ""
            # A complete answer, a non-length stop, or (reasoning off) the one-shot budget -> done.
            # Growing the budget only makes sense while the model is thinking; when reasoning is
            # off the chunk-sized budget already fits the answer and must not creep toward the
            # context limit on a wide table.
            if (
                content
                or choice.finish_reason != "length"
                or not self._reasoning_on
                or budget >= self.max_tokens_ceiling
            ):
                return content, n_calls
            budget = min(budget * 2, self.max_tokens_ceiling)

    def _predict_chunk(self, chunk: pd.DataFrame) -> tuple[list, list, int, int]:
        """Classify one chunk of test rows -> ``(labels, probs, n_calls, n_unparsed)``.

        ``labels`` is one class per row (rows the model omits fall back to the majority
        class) and ``probs`` the matching per-row confidence (``None`` = one-hot at scoring
        time). Pure w.r.t. shared state, so chunks run concurrently.

        The whole training table rides in every request, so a wide table can exceed the
        endpoint's context window. That hard server-side limit is not knowable in advance
        (no local tokenizer), so an over-context chunk is split in half and each half is
        classified separately — fewer test rows shrink the request. A single row that still
        overflows (the training block alone is too large) is re-raised, and the caller records
        the cell as a failure rather than substituting a guess."""
        try:
            content, n_calls = self._complete(chunk)
        except openai.BadRequestError as e:
            if "context" not in str(e).lower() or len(chunk) <= 1:
                raise
            # Halving test rows only helps when the *test* portion is the overflow. If even a
            # single-row prompt exceeds the window, the training block itself is too big and
            # splitting is futile — re-raise now so the caller records a skip instead of
            # thrashing through log2(N) doomed requests. (No-op when no context_window is set.)
            if self.context_skip_reason(chunk.iloc[:1]) is not None:
                raise
            mid = len(chunk) // 2
            left = self._predict_chunk(chunk.iloc[:mid])
            right = self._predict_chunk(chunk.iloc[mid:])
            return (left[0] + right[0], left[1] + right[1], left[2] + right[2], left[3] + right[3])
        m = len(chunk)
        parsed = self._parse_batch(content, m)  # {idx: (prompt-space token, prob|None)}
        labels, probs = [], []
        for i in range(m):
            tok, prob = parsed.get(i + 1, (self._prompt_majority, None))
            labels.append(self._dec[tok])
            probs.append(prob)
        return labels, probs, n_calls, m - len(parsed)

    def _run(self, data_test: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """Predict the test table, split into ``test_batch_size`` chunks run concurrently
        (``max_workers`` at a time). Cached so ``predict`` + ``predict_proba`` share it."""
        cache_key = tuple(data_test.index)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._client is None:
            self._client = build_client(
                self.provider
            )  # build once, single-threaded, before the pool

        n = len(data_test)
        bs = max(1, self.test_batch_size)
        chunks = [data_test.iloc[i : i + bs] for i in range(0, n, bs)]
        workers = max(1, min(self.max_workers, len(chunks)))
        # Optional wall-clock cap so one slow model can't stall the whole grid cell. When the
        # budget is blown mid-cell we stop collecting chunks and raise CellTimeout (the caller
        # records a skip); in-flight requests are abandoned, not awaited (each clears within its
        # own request_timeout). ``<= 0`` disables the cap and keeps the original blocking wait.
        deadline = time.monotonic() + self.cell_timeout_s if self.cell_timeout_s > 0 else None
        if workers == 1:
            results = []
            for ch in chunks:
                if deadline is not None and time.monotonic() > deadline:
                    raise CellTimeout(
                        f"{self.model_name}: exceeded cell timeout {self.cell_timeout_s:g}s "
                        f"after {len(results)}/{len(chunks)} chunk(s)."
                    )
                results.append(self._predict_chunk(ch))
        else:
            ex = ThreadPoolExecutor(max_workers=workers)
            futures = [ex.submit(self._predict_chunk, ch) for ch in chunks]
            try:
                if deadline is None:
                    results = [f.result() for f in futures]  # order preserved
                else:
                    done: dict = {}
                    try:
                        for f in as_completed(
                            futures, timeout=max(0.0, deadline - time.monotonic())
                        ):
                            done[f] = f.result()
                    except FuturesTimeout:
                        raise CellTimeout(
                            f"{self.model_name}: exceeded cell timeout {self.cell_timeout_s:g}s "
                            f"with {len(done)}/{len(futures)} chunk(s) done."
                        )
                    results = [done[f] for f in futures]  # restore submission order
            finally:
                ex.shutdown(wait=False, cancel_futures=True)

        preds: list = []
        probs_all: list = []
        cell_unparsed = 0
        for labels, probs, n_calls, n_unparsed in results:
            preds.extend(labels)
            probs_all.extend(probs)
            self._n_api_calls += n_calls
            self._n_unparsed += n_unparsed
            cell_unparsed += n_unparsed

        # Every unparsed row above became a majority-class prediction. That is a plausible
        # prediction, not a visible error, so past a ceiling the whole vector is refused rather
        # than allowed to enter the leaderboard as if it were a measurement.
        if n and cell_unparsed / n > self.max_unparsed_frac:
            raise UnparseableResponses(
                f"{self.model_name}: {cell_unparsed}/{n} test rows ({cell_unparsed / n:.0%}) "
                f"had no parseable class label, above the {self.max_unparsed_frac:.0%} limit; "
                "refusing to report majority-class fallbacks as predictions."
            )
        y_pred = pd.Series(preds, index=data_test.index, name="target")

        classes_str = [str(c) for c in self._classes]
        k = len(classes_str)
        proba = pd.DataFrame(0.0, index=data_test.index, columns=classes_str)
        for (idx, c), p in zip(y_pred.items(), probs_all):
            if p is None or k < 2:
                # No elicited confidence (or a single class): one-hot on the predicted label.
                proba.at[idx, str(c)] = 1.0
            else:
                # Clamp so the predicted class keeps the majority of the mass — a model that
                # reports p < 1/k would otherwise contradict its own top-1 choice — then spread
                # the remainder uniformly over the other classes to form a proper distribution.
                p = min(max(p, 1.0 / k), 1.0)
                proba.loc[idx] = (1.0 - p) / (k - 1)
                proba.at[idx, str(c)] = p

        self._cache[cache_key] = (y_pred, proba)
        return y_pred, proba

    def predict(self, data_test: pd.DataFrame) -> pd.Series:
        return self._run(data_test)[0]

    def predict_proba(self, data_test: pd.DataFrame) -> pd.DataFrame:
        return self._run(data_test)[1]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count from characters (no local tokenizer available)."""
        return int(len(text) / _CHARS_PER_TOKEN) + 1

    def context_skip_reason(self, data_test: pd.DataFrame) -> str | None:
        """Reason to skip this cell if even a single-row request exceeds the context window.

        The whole training table rides in every request, so if the training block plus one
        test row plus the reserved completion budget already exceeds the model's window, no
        request for this cell can succeed — the caller skips it up front instead of firing
        requests that 400. Requires :meth:`fit` to have run. Returns ``None`` when the cell
        fits, or when no ``context_window`` is configured for the model (nothing to check)."""
        if self.context_window is None:
            return None
        assert self._train_block is not None, "context_skip_reason() called before fit()."
        text = self.system_prompt + "\n" + self._build_user_message(data_test.iloc[:1])
        est_input = self._estimate_tokens(text)
        reserve = self._initial_budget(1)
        total = est_input + reserve
        if total <= self.context_window:
            return None
        return (
            f"exceeds {self.model_name} context window: ~{total} tokens "
            f"(input ~{est_input} + reserved output {reserve}) > {self.context_window}"
        )

    def get_fit_stats(self) -> dict:
        return {
            "llm_api_calls": self._n_api_calls,
            "llm_unparsed": self._n_unparsed,
            # Recorded per cell so a run is reconstructable from its outputs alone, and so the
            # transductive/inductive setting is never left implicit (see ``test_batch_size``).
            "llm_test_batch_size": self.test_batch_size,
            "llm_elicit_proba": self.elicit_proba,
            "llm_model_key": self.model_name,
            "llm_api_model": self.api_model,
            "llm_provider": self.provider,
            "llm_context_window": self.context_window,
            "llm_temperature": self.temperature,
            "llm_seed": self.seed,
            "llm_reasoning_effort": self.reasoning_effort,
            "llm_reasoning_enabled": self._reasoning_on,
            "llm_hide_labels": self.hide_labels,
            "llm_max_unparsed_frac": self.max_unparsed_frac,
            "llm_system_prompt_sha256": sha256(self.system_prompt.encode()).hexdigest(),
        }
