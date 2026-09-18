#!/usr/bin/env bash
set -euo pipefail

# One BF16 model replica across four GPUs. Requires a Qwen3.8-capable SGLang build.
# Override settings with environment variables; extra CLI arguments pass to SGLang.
model_path=${model_path:-/public/llm_models/Qwen/Qwen3.8-Flash-Next}
served_model_name=${served_model_name:-Qwen/Qwen3.8-Flash-Next}
gpu_ids=${gpu_ids:-4,5,6,7}
port=${port:-30000}
host=${host:-0.0.0.0}
mem_frac=${mem_frac:-0.9}
log_dir=${log_dir:-logs/sglang_qwen38_flash_next}
startup_timeout=${startup_timeout:-3600}

IFS=',' read -r -a devices <<< "$gpu_ids"
if [[ ${#devices[@]} -ne 4 ]]; then
    echo 'ERROR: this TP=4 launcher requires exactly four GPU IDs.' >&2
    exit 1
fi
for tool in sglang curl; do
    command -v "$tool" >/dev/null || { echo "ERROR: $tool is not installed." >&2; exit 1; }
done
# These parameters must stay consistent with the readiness probe and GPU allocation.
for arg in "$@"; do
    case "$arg" in
        --port|--port=*|--host|--host=*|--tp|--tp=*|--tp-size|--tp-size=*|--tensor-parallel-size|--tensor-parallel-size=*)
            echo "ERROR: configure host/port using environment variables; TP is fixed at 4." >&2
            exit 1 ;;
    esac
done
probe_host=$host
[[ "$host" != 0.0.0.0 ]] || probe_host=127.0.0.1
endpoint="http://${probe_host}:${port}"
if curl --noproxy '*' -fsS --max-time 2 "$endpoint/health" >/dev/null 2>&1; then
    echo "ERROR: a service is already healthy at $endpoint; choose another port." >&2
    exit 1
fi
mkdir -p "$log_dir"
log_file="$log_dir/server_tp4_port_${port}_$(date '+%Y%m%d_%H%M%S')_$$.log"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Model: $model_path"
echo "GPUs: $gpu_ids (one replica, TP=4, BF16)"
echo "Log: $log_file"
CUDA_VISIBLE_DEVICES="$gpu_ids" sglang serve \
    --model-path "$model_path" \
    --served-model-name "$served_model_name" \
    --tp 4 \
    --host "$host" \
    --port "$port" \
    --dtype bfloat16 \
    --mem-fraction-static "$mem_frac" \
    "$@" >"$log_file" 2>&1 &
server_pid=$!

started=$SECONDS
while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: SGLang exited during startup. Log: $log_file" >&2
        tail -n 100 "$log_file" >&2
        exit 1
    fi
    if curl --noproxy '*' -fsS --max-time 5 "$endpoint/health" >/dev/null 2>&1; then
        break
    fi
    if (( SECONDS - started >= startup_timeout )); then
        echo "ERROR: readiness timed out after ${startup_timeout}s. Log: $log_file" >&2
        tail -n 100 "$log_file" >&2
        exit 1
    fi
    echo "Waiting for model readiness: $((SECONDS - started))s; log: $log_file"
    sleep 10
done

echo "SGLang is ready: $endpoint"
echo "Regen arguments: --model $served_model_name --server-address ${probe_host}:${port} --disable-thinking"
# Keep the launcher in the foreground and propagate the server's exit status.
wait "$server_pid"
