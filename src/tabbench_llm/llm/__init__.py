"""LLM classification backend for TabBench-LLM.

Adds an OpenAI-compatible, in-context LLM classifier that plugs into the standard
benchmark pipeline alongside the tabular baselines (RF, TabPFN). Nothing here imports
``openai`` at module-import time — the client is built lazily on the first API call — so
importing this package stays cheap and dependency-light.
"""

from __future__ import annotations

from tabbench_llm.llm.model import CellTimeout, LLMModel, UnparseableResponses
from tabbench_llm.llm.prompts import SYSTEM_PROMPTS
from tabbench_llm.llm.registry import LLM_MODELS, is_llm_model

__all__ = [
    "CellTimeout",
    "LLMModel",
    "LLM_MODELS",
    "SYSTEM_PROMPTS",
    "UnparseableResponses",
    "is_llm_model",
]
