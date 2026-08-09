#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODELS="LOCAL-QWEN3-8B LOCAL-QWEN3-32B-FP8 LOCAL-QWEN3-30B-A3B LOCAL-GEMMA-3-27B LOCAL-LLAMA-3.1-8B LOCAL-MISTRAL-7B-FP16 LOCAL-MISTRAL-SMALL-3.1-24B-FP16"
SKIP_INSTALL=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models)
            MODELS=${2:?"--models needs a comma- or space-separated value"}
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if ! command -v ollama >/dev/null 2>&1; then
    if [ "$SKIP_INSTALL" -eq 1 ]; then
        echo "Ollama is missing. Run: bash scripts/setup_linux_ollama.sh" >&2
        exit 1
    fi
    install_root=${OLLAMA_INSTALL_ROOT:-"$PROJECT_ROOT/.local"}
    echo "Installing the official Ollama Linux package under $install_root (no sudo needed)..."
    mkdir -p "$install_root"
    curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst \
        | tar -C "$install_root" -x --zstd
    export PATH="$install_root/bin:$PATH"
fi

if ! command -v python >/dev/null 2>&1; then
    echo "Python is missing from PATH. Source init.sh first." >&2
    exit 1
fi

api_ready() {
    python -c 'import os, urllib.request; host=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"); root=host if "://" in host else "http://" + host; urllib.request.urlopen(root.rstrip("/") + "/api/version", timeout=5)' 2>/dev/null
}

server_pid=""
server_log=""
model_file=""
cleanup() {
    [ -z "$model_file" ] || rm -f -- "$model_file"
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    [ -z "$server_log" ] || rm -f -- "$server_log"
}
trap cleanup EXIT

if ! api_ready; then
    server_log=$(mktemp "${TMPDIR:-/tmp}/tabarena-ollama-setup.XXXXXX.log")
    echo "Starting a temporary Ollama server at ${OLLAMA_HOST:-127.0.0.1:11434}..."
    ollama serve > "$server_log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 24); do
        api_ready && break
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Ollama failed to start:" >&2
            tail -n 60 "$server_log" >&2
            exit 1
        fi
        sleep 5
    done
    if ! api_ready; then
        echo "Ollama did not become ready within 120 seconds; see $server_log" >&2
        exit 1
    fi
fi

rows=$(python - "$MODELS" <<'PY'
import sys
from tabbench_llm.llm.registry import LLM_MODELS

keys = sys.argv[1].replace(",", " ").split()
for key in keys:
    if key not in LLM_MODELS:
        raise SystemExit(f"unknown model key: {key}")
    info = LLM_MODELS[key]
    if info.get("provider") != "ollama" or "ollama_source" not in info:
        raise SystemExit(f"{key} is not a cluster Ollama registry entry")
    print("\t".join((key, info["ollama_source"], info["api_model"], str(info["context_window"]))))
PY
)

model_file=$(mktemp "${TMPDIR:-/tmp}/tabarena-ollama-modelfile.XXXXXX")

while IFS=$'\t' read -r key source alias context_window; do
    [ -n "$key" ] || continue
    echo "[ollama-setup] $key: pulling $source"
    ollama pull "$source"
    printf 'FROM %s\nPARAMETER num_ctx %s\n' "$source" "$context_window" > "$model_file"
    echo "[ollama-setup] creating $alias (num_ctx=$context_window)"
    ollama create "$alias" -f "$model_file"
done <<< "$rows"

echo "[ollama-setup] ready:"
ollama list
