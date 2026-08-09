#!/bin/bash
# Environment bootstrap for TabBench-LLM (mirrors the layout of other cluster projects).
# Creates/activates a uv-managed virtualenv and installs the benchmark dependencies with the AutoGluon
# fork that lifts the 500-feature cap on tabular foundation models (required for omics).
INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR_LOCAL=${TABBENCH_LLM_PROJECT_ROOT:-$INIT_DIR}
VENV_DIR=${TABBENCH_LLM_VENV:-$BASE_DIR_LOCAL/.venv}
PYTHON_VERSION=3.11   # AutoGluon 1.5 supports 3.9-3.12; 3.11 is the safe choice

# Keep all model/dataset caches inside the project (avoids filling the home quota).
# XDG_CACHE_HOME redirects the generic ~/.cache (OpenML, uv, ...) into the repo;
# the remaining tools use their own cache env var and are pinned explicitly.
export XDG_CACHE_HOME=$BASE_DIR_LOCAL/.cache
export TABBENCH_LLM_CACHE=$BASE_DIR_LOCAL/.cache/bio
export HF_HOME=$BASE_DIR_LOCAL/.cache/huggingface
export TABPFN_MODEL_CACHE_DIR=$BASE_DIR_LOCAL/.cache/tabpfn
export KAGGLEHUB_CACHE=$BASE_DIR_LOCAL/.cache/kagglehub
export OLLAMA_MODELS=$BASE_DIR_LOCAL/.cache/ollama

# Let the CUDA allocator grow/shrink segments instead of pre-reserving fixed blocks: cuts the
# fragmentation-driven OOM a long-lived GPU worker hits fitting many models back-to-back.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR_LOCAL"

### uv bootstrap
# uv ships in the local miniconda; fall back to the standalone installer if absent.
export PATH="$BASE_DIR_LOCAL/.local/bin:$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create + populate the venv on first run, otherwise just activate it.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating uv venv '$VENV_DIR' (python $PYTHON_VERSION)..."
    uv venv --python $PYTHON_VERSION "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    # 1) AutoGluon fork FIRST (git installs that satisfy the autogluon.* extras below)
    echo "Installing AutoGluon fork (lifts the 500-feature cap)..."
    uv pip install -r requirements-autogluon-fork.txt

else
    echo "Activating existing uv venv '$VENV_DIR'..."
    source "$VENV_DIR/bin/activate"
fi

# Sync all declared dependencies after every pull. uv is incremental,
# so an unchanged environment is a quick no-op; an existing venv picks up newly added packages
# such as the Ollama Python client without requiring manual recreation.
echo "Syncing full benchmark dependencies..."
uv pip install -r requirements-full.txt
export PYTHONPATH="$BASE_DIR_LOCAL/src${PYTHONPATH:+:$PYTHONPATH}"

# The venv is populated once at creation, so source-only deps added later
# (tabfm ships from git, not PyPI) can be missing from an existing venv. Guard-install it when
# absent — a no-op once present. Runs on the compute node where the venv interpreter exists.
python -c "import tabfm" 2>/dev/null || {
    echo "Installing missing tabfm backend (git source)..."
    uv pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm"
}
### End uv
