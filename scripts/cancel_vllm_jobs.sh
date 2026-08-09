#!/usr/bin/env bash
set -euo pipefail

if ! command -v squeue >/dev/null 2>&1 || ! command -v scancel >/dev/null 2>&1; then
    echo "Run this script on the Slurm login node (squeue/scancel are not available here)." >&2
    exit 1
fi

echo "Queued/running legacy vLLM jobs named tabarena-local:"
squeue --user "$USER" --name tabarena-local
scancel --user "$USER" --name tabarena-local
echo "Cancellation requested. Remaining matches:"
squeue --user "$USER" --name tabarena-local

