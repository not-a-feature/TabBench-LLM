"""OpenAI-compatible client for the LiteLLM proxy.

The API key + endpoint live in a small JSON file (kept out of git) so credentials are
never hard-coded. Point ``$TABBENCH_LLM_KEY`` at the file, or drop it at the default
path ``configs/llm_key.json`` relative to the working directory. Schema::

    {"api_key": "sk-...", "base_url": "https://llm.mlcloud.uni-tuebingen.de"}

A provider's endpoint can also come from the environment instead of the file, via
``TABBENCH_LLM_<PROVIDER>_BASE_URL`` (+ optional ``..._API_KEY``, default ``EMPTY``).
This is what the self-hosted ``ollama`` provider uses: an Ollama server started by the job
listens on a job-specific host:port that is not known when the credentials file is written,
and a self-hosted server needs no credential at all.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_ENV = "TABBENCH_LLM_KEY"
_DEFAULT_KEY_FILE = "configs/llm_key.json"


def _key_path() -> Path:
    override = os.environ.get(_ENV)
    return Path(override) if override else Path(_DEFAULT_KEY_FILE)


def env_endpoint(provider: str) -> dict | None:
    """The ``{api_key, base_url}`` for *provider* taken from the environment, or ``None``.

    Reads ``TABBENCH_LLM_<PROVIDER>_BASE_URL`` (provider upper-cased, non-alphanumerics to
    ``_``); the matching ``..._API_KEY`` is optional and defaults to ``EMPTY`` because a
    self-hosted OpenAI-compatible Ollama server accepts any token. When set it wins
    over the credentials file, so a job can point at a server it just started.
    """
    prefix = "TABBENCH_LLM_" + re.sub(r"[^A-Z0-9]", "_", provider.upper())
    base_url = os.environ.get(prefix + "_BASE_URL")
    if not base_url:
        return None
    return {"api_key": os.environ.get(prefix + "_API_KEY", "EMPTY"), "base_url": base_url}


def load_key() -> dict:
    """Load and validate the credentials JSON (supporting nested providers or flat legacy style)."""
    path = _key_path()
    assert path.is_file(), (
        f"LLM key file not found at {path}. Create it as JSON "
        '{"ml-cloud": {"api_key": "...", "base_url": "https://llm.mlcloud.uni-tuebingen.de"}} '
        f"or point ${_ENV} at one."
    )
    cfg = json.loads(path.read_text())

    # If the file has a flat structure (legacy style), wrap it under 'ml-cloud'
    if "api_key" in cfg and "base_url" in cfg:
        cfg = {"ml-cloud": cfg}

    # Validate that every provider block has 'api_key' and 'base_url'
    for provider, prov_cfg in cfg.items():
        assert (
            isinstance(prov_cfg, dict) and "api_key" in prov_cfg and "base_url" in prov_cfg
        ), f"Provider block for {provider!r} in {path} must contain both 'api_key' and 'base_url'."
    return cfg


def build_client(provider: str = "ml-cloud"):
    """Return an ``openai.OpenAI`` client pointed at the configured provider's endpoint.

    The environment (:func:`env_endpoint`) takes precedence over the credentials file, and
    is the only source consulted for a provider the file does not list — so a self-hosted
    server started by the current job needs no edit to the credentials file.
    """
    import openai

    prov_cfg = env_endpoint(provider)
    if prov_cfg is None:
        cfg = load_key()
        assert provider in cfg, (
            f"Provider {provider!r} not configured in llm_key.json and no "
            f"TABBENCH_LLM_{re.sub(r'[^A-Z0-9]', '_', provider.upper())}_BASE_URL in the "
            f"environment. Available providers: {list(cfg.keys())}"
        )
        prov_cfg = cfg[provider]
    base_url = prov_cfg["base_url"]

    # Translate Google's OpenAI-compatible base URL endpoint if needed
    if provider == "google" and "generativelanguage.googleapis.com" in base_url:
        if base_url.endswith("/models/"):
            base_url = base_url.replace("/models/", "/openai/")
        elif base_url.endswith("/models"):
            base_url = base_url.replace("/models", "/openai")

    return openai.OpenAI(api_key=prov_cfg["api_key"], base_url=base_url)
