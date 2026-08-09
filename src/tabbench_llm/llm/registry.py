"""Registry of LLM "models" runnable in the benchmark.

A model *key* (used in configs, cache keys, and result filenames — so it must be
filesystem-safe, no slashes) maps to the API model id sent to the OpenAI-compatible
proxy. The map lives in ``data/llm_models.json`` so adding a model needs no code edit.

Each entry may also carry an optional ``context_window`` (the model's max token window):
the prediction step uses it to skip a (model, dataset, cell) up front when even a
single-row prompt would overflow it, rather than firing requests that 400. The bundled
values are the models' advertised windows — adjust them to match your actual endpoint (and
leave a little headroom, since the pre-skip estimates tokens from characters).

The bundled file holds only endpoints that can serve a whole grid: the ml-cloud proxy and the
self-hosted Ollama models (including stable ``LOCAL-*`` compatibility keys). The paid
third-party endpoints live in
``data/llm_models_external.json`` and are merged back in on demand via
``$TABBENCH_LLM_EXTRA_MODELS`` (see :data:`_EXTRA_ENV`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_MODELS_FILE = Path(__file__).parent / "data" / "llm_models.json"

#: Extra registry files to merge on top of the bundled one, ``os.pathsep``-separated. This is
#: how the parked third-party endpoints come back::
#:
#:     TABBENCH_LLM_EXTRA_MODELS=src/tabbench_llm/llm/data/llm_models_external.json
#:
#: A later file wins on a duplicate key, so it can also override a bundled entry (e.g. to point
#: a key at a different checkpoint) without editing the shipped file.
_EXTRA_ENV = "TABBENCH_LLM_EXTRA_MODELS"


def _load(path: Path) -> dict:
    """Read one registry file, dropping ``_``-prefixed comment keys (as the run configs do)."""
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


def _load_registry() -> dict[str, dict[str, str] | str]:
    models = _load(_MODELS_FILE)
    for raw in os.environ.get(_EXTRA_ENV, "").split(os.pathsep):
        if raw.strip():
            path = Path(raw.strip())
            assert path.is_file(), f"${_EXTRA_ENV} names a missing file: {path}"
            models.update(_load(path))
    return models


#: ``key -> api_model_id``. The key is what a run config lists as a model; the value is
#: the ``model=`` string passed to the chat-completions endpoint.
LLM_MODELS: dict[str, dict[str, str] | str] = _load_registry()


def is_llm_model(name: str) -> bool:
    """Whether ``name`` is a registered LLM model key (routed to :class:`LLMModel`)."""
    return name in LLM_MODELS
