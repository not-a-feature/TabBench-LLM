#!/usr/bin/env bash
# Self-host the cluster model slice with Ollama on one H100 (physical GPU 0 by default).
# Submit: MODELS="LOCAL-QWEN3-8B LOCAL-QWEN3-32B-FP8" sbatch run_ollama_grid.sbatch

#SBATCH --job-name=tabarena-ollama
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=tabarena_ollama.%j.out
#SBATCH --error=tabarena_ollama.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jules.kreuer@uni-tuebingen.de

set -euo pipefail

PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$PROJECT_ROOT"

# An explicit GPU_DEVICE wins; under Slurm, preserve the assigned CUDA visibility; otherwise
# use physical GPU 0 on the non-Slurm data server. CUDA remaps it to logical cuda:0 for both
# Ollama and the PyTorch baselines. GPU 1 is intentionally left available to unrelated work.
GPU_DEVICE=${GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}
export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
source "$PROJECT_ROOT/init.sh"

GRID_CONFIG=${GRID_CONFIG:-configs/grid_all_systems.json}
OUTPUT=${OUTPUT:-results/grid_all_systems}
MODELS=${MODELS:-"LOCAL-QWEN3-8B LOCAL-QWEN3-32B-FP8 LOCAL-QWEN3-30B-A3B LOCAL-GEMMA-3-27B LOCAL-LLAMA-3.1-8B LOCAL-MISTRAL-7B-FP16 LOCAL-MISTRAL-SMALL-3.1-24B-FP16"}
BASELINES=${BASELINES:-"DUMMY,RF,REALTABPFN-V2"}
PORT=${PORT:-$((11434 + SLURM_JOB_ID % 1000))}
SERVER_BOOT_TIMEOUT_S=${SERVER_BOOT_TIMEOUT_S:-300}

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_MODELS=${OLLAMA_MODELS:-"$PROJECT_ROOT/.cache/ollama"}
export OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-2}
export OLLAMA_MAX_QUEUE=${OLLAMA_MAX_QUEUE:-512}
export TABBENCH_LLM_OLLAMA_BASE_URL="http://${OLLAMA_HOST}/v1"
export TABBENCH_LLM_OLLAMA_API_KEY=ollama
export TABBENCH_LLM_CELL_WORKERS=${TABBENCH_LLM_CELL_WORKERS:-2}
export TABBENCH_LLM_MAX_WORKERS=${TABBENCH_LLM_MAX_WORKERS:-4}
export TABBENCH_LLM_REQUEST_TIMEOUT=${TABBENCH_LLM_REQUEST_TIMEOUT:-900}
export TABBENCH_LLM_CELL_TIMEOUT=${TABBENCH_LLM_CELL_TIMEOUT:-3600}

mkdir -p "$PROJECT_ROOT/logs" "$OLLAMA_MODELS"

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is missing. Run 'bash scripts/setup_linux_ollama.sh' on the server first." >&2
    exit 1
fi

python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        "Expected exactly one usable CUDA device after applying CUDA_VISIBLE_DEVICES="
        f"{__import__('os').environ.get('CUDA_VISIBLE_DEVICES')!r}, but PyTorch reports "
        f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
    )
probe = torch.ones(1, device="cuda:0")
print(f"[ollama-grid] CUDA preflight passed: {torch.cuda.get_device_name(0)} {probe}")
PY

OLLAMA_LOG="$PROJECT_ROOT/logs/ollama.${SLURM_JOB_ID}.log"
# A node with a faulty GPU breaks Ollama's index-based CUDA enumeration: with
# CUDA_VISIBLE_DEVICES set it discovers no GPU at all and silently serves on CPU
# (size_vram=0). Unset, Ollama filters the bad device and picks the healthy one itself.
# The Python side keeps CUDA_VISIBLE_DEVICES so the preflight and GPU baselines stay pinned.
if [ "${OLLAMA_AUTODETECT_GPU:-0}" = "1" ]; then
    env -u CUDA_VISIBLE_DEVICES ollama serve > "$OLLAMA_LOG" 2>&1 &
else
    ollama serve > "$OLLAMA_LOG" 2>&1 &
fi
OLLAMA_PID=$!

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

waited=0
until python -c "import urllib.request; urllib.request.urlopen('http://${OLLAMA_HOST}/api/version', timeout=5)" 2>/dev/null; do
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Ollama exited during startup; see $OLLAMA_LOG" >&2
        tail -n 60 "$OLLAMA_LOG" || true
        exit 1
    fi
    if [ "$waited" -ge "$SERVER_BOOT_TIMEOUT_S" ]; then
        echo "Ollama did not become ready within ${SERVER_BOOT_TIMEOUT_S}s." >&2
        exit 1
    fi
    sleep 5
    waited=$((waited + 5))
done

echo "[ollama-grid] job=$SLURM_JOB_ID host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-all}"
ollama --version
nvidia-smi -L

bash scripts/setup_linux_ollama.sh --skip-install --models "$MODELS"

for key in $MODELS; do
    echo "[ollama-grid] running $key"
    python scripts/capture_run_manifest.py --config "$GRID_CONFIG" --output "$OUTPUT" --models "$key"
    python scripts/grid.py --config "$GRID_CONFIG" --output "$OUTPUT" --models "$key"
    alias=$(python -c "from tabbench_llm.llm.registry import LLM_MODELS; print(LLM_MODELS['$key']['api_model'])")
    ollama stop "$alias" || true
done

# Shut the model server down before GPU baselines so it releases all VRAM.
cleanup
trap - EXIT INT TERM

python scripts/capture_run_manifest.py --config "$GRID_CONFIG" --output "$OUTPUT" --models "$BASELINES"
python scripts/grid.py --config "$GRID_CONFIG" --output "$OUTPUT" --models "$BASELINES"
python scripts/grid.py --config "$GRID_CONFIG" --output "$OUTPUT" --skip-run

echo "[ollama-grid] finished: $OUTPUT"
