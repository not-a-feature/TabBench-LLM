#!/usr/bin/env bash
# Monitor data GPU 1 and resume the Ollama reasoning arm once it is free.
#
# The monitor polls every ten minutes for at most one week. It exits as soon as
# the benchmark runner has been launched successfully. A GPU is considered free
# only when nvidia-smi reports strictly less than 5000 MiB in use.

set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
GPU_INDEX=${GPU_INDEX:-1}
FREE_THRESHOLD_MIB=${FREE_THRESHOLD_MIB:-5000}
POLL_INTERVAL_S=${POLL_INTERVAL_S:-600}
MAX_RUNTIME_S=${MAX_RUNTIME_S:-604800}
PORT=${PORT:-12439}
SLURM_JOB_ID=${SLURM_JOB_ID:-819027}
MODELS=${MODELS:-"LOCAL-QWEN3-8B LOCAL-QWEN3-32B-FP8 LOCAL-QWEN3-30B-A3B"}
BASELINES=${BASELINES:-"DUMMY,RF,REALTABPFN-V2"}

LOG_DIR="$PROJECT_ROOT/logs"
MONITOR_LOG="$LOG_DIR/reasoning_arm_data_gpu1.monitor.log"
MONITOR_PID_FILE="$LOG_DIR/reasoning_arm_data_gpu1.monitor.pid"
LOCK_FILE="$LOG_DIR/reasoning_arm_data_gpu1.monitor.lock"
RUN_LOG="$LOG_DIR/reasoning_arm_data_gpu1.log"
RUN_PID_FILE="$LOG_DIR/reasoning_arm_data_gpu1.pid"

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MONITOR_LOG"
}

if ! command -v flock >/dev/null 2>&1; then
    log "ERROR: flock is required to prevent duplicate monitors."
    exit 1
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "Another GPU monitor already holds $LOCK_FILE; exiting."
    exit 0
fi

printf '%s\n' "$$" > "$MONITOR_PID_FILE"
cleanup() {
    rm -f "$MONITOR_PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$(hostname -s)" != "data" ]; then
    log "ERROR: this monitor must run on host data, not $(hostname -s)."
    exit 1
fi

gpu_memory_mib() {
    nvidia-smi \
        --id="$GPU_INDEX" \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NR == 1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print int($1)}'
}

runner_is_active() {
    local pid command
    [ -s "$RUN_PID_FILE" ] || return 1
    pid=$(cat "$RUN_PID_FILE" 2>/dev/null) || return 1
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    kill -0 "$pid" 2>/dev/null || return 1
    command=$(ps -o args= -p "$pid" 2>/dev/null || true)
    case "$command" in
        *"bash run_ollama_grid.sbatch"*) return 0 ;;
        *) return 1 ;;
    esac
}

port_is_free() {
    ! ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .
}

deadline_epoch=$(( $(date +%s) + MAX_RUNTIME_S ))
log "Monitor started: data GPU $GPU_INDEX, free below ${FREE_THRESHOLD_MIB} MiB, interval ${POLL_INTERVAL_S}s, maximum ${MAX_RUNTIME_S}s."

while [ "$(date +%s)" -lt "$deadline_epoch" ]; do
    if runner_is_active; then
        log "Benchmark runner $(cat "$RUN_PID_FILE") is already active; monitor exiting."
        exit 0
    fi

    used_mib=$(gpu_memory_mib || true)
    if ! [[ "$used_mib" =~ ^[0-9]+$ ]]; then
        log "Could not read data GPU $GPU_INDEX memory usage; retrying in ${POLL_INTERVAL_S}s."
    elif [ "$used_mib" -ge "$FREE_THRESHOLD_MIB" ]; then
        log "GPU $GPU_INDEX is busy: ${used_mib} MiB used (need < ${FREE_THRESHOLD_MIB} MiB)."
    elif ! port_is_free; then
        log "GPU $GPU_INDEX is free, but port $PORT is in use; retrying."
    else
        # Recheck immediately before launch so a newly started workload wins the race.
        sleep 2
        recheck_mib=$(gpu_memory_mib || true)
        if [[ "$recheck_mib" =~ ^[0-9]+$ ]] \
            && [ "$recheck_mib" -lt "$FREE_THRESHOLD_MIB" ] \
            && port_is_free; then
            log "GPU $GPU_INDEX is free (${recheck_mib} MiB); starting the resumable reasoning benchmark."
            cd "$PROJECT_ROOT"
            nohup env \
                SLURM_JOB_ID="$SLURM_JOB_ID" \
                GPU_DEVICE="$GPU_INDEX" \
                PORT="$PORT" \
                MODELS="$MODELS" \
                BASELINES="$BASELINES" \
                OLLAMA_AUTODETECT_GPU=0 \
                bash run_ollama_grid.sbatch >> "$RUN_LOG" 2>&1 < /dev/null &
            runner_pid=$!
            printf '%s\n' "$runner_pid" > "$RUN_PID_FILE"

            sleep 10
            if runner_is_active; then
                log "Benchmark started successfully as PID $runner_pid; log: $RUN_LOG. Monitor exiting."
                exit 0
            fi
            log "Benchmark PID $runner_pid exited during startup; the monitor will retry."
        else
            log "GPU $GPU_INDEX became busy during the launch recheck; retrying."
        fi
    fi

    now_epoch=$(date +%s)
    remaining_s=$(( deadline_epoch - now_epoch ))
    [ "$remaining_s" -gt 0 ] || break
    sleep_s=$POLL_INTERVAL_S
    if [ "$remaining_s" -lt "$sleep_s" ]; then
        sleep_s=$remaining_s
    fi
    sleep "$sleep_s"
done

log "Monitor expired after ${MAX_RUNTIME_S}s without starting the benchmark."
exit 0
