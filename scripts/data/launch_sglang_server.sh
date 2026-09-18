#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SGLang multi-replica launcher
#
# Example:
#   start_gpu=7
#   end_gpu=8
# launches GPU 7 only.
#
# GPU range follows [start_gpu, end_gpu).
# ============================================================

model_path=/public/llm_models/Qwen/Qwen3-4B

start_gpu=4
end_gpu=8

start_port=30000
host=0.0.0.0

dtype=bfloat16
mem_frac=0.9

log_dir=logs/sglang_qwen3_4b
heartbeat_interval=300


# ============================================================
# Get host IP
# ============================================================

get_host_ip() {
    local host_ip=""

    if command -v hostname >/dev/null 2>&1; then
        host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi

    if [[ -z "${host_ip}" ]] && command -v ip >/dev/null 2>&1; then
        host_ip=$(
            ip -4 route get 1.1.1.1 2>/dev/null | awk '
                /src/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "src") {
                            print $(i + 1)
                            exit
                        }
                    }
                }
            '
        )
    fi

    if [[ -z "${host_ip}" ]]; then
        host_ip=127.0.0.1
    fi

    printf '%s\n' "${host_ip}"
}


# ============================================================
# Initialization
# ============================================================

mkdir -p "${log_dir}"

host_ip=$(get_host_ip)

pids=()
gpu_ids=()
ports=()
log_files=()

heartbeat_pid=""


# ============================================================
# Heartbeat
# ============================================================

print_heartbeat() {
    local timestamp
    local alive_count=0

    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[${timestamp}] heartbeat"

    for ((idx = 0; idx < ${#pids[@]}; idx++)); do
        local pid=${pids[$idx]}
        local gpu_id=${gpu_ids[$idx]}
        local port=${ports[$idx]}
        local status=dead

        if kill -0 "${pid}" >/dev/null 2>&1; then
            status=alive
            alive_count=$((alive_count + 1))
        fi

        echo "  gpu=${gpu_id} pid=${pid} port=${port} status=${status}"
    done

    echo "  alive_workers=${alive_count}/${#pids[@]}"
}


heartbeat_loop() {
    while true; do
        sleep "${heartbeat_interval}"

        print_heartbeat
    done
}


# ============================================================
# Cleanup
# ============================================================

cleanup() {
    echo
    echo "Stopping SGLang workers..."

    if [[ -n "${heartbeat_pid}" ]] &&
       kill -0 "${heartbeat_pid}" >/dev/null 2>&1; then
        kill "${heartbeat_pid}" >/dev/null 2>&1 || true
    fi

    for pid in "${pids[@]:-}"; do
        if kill -0 "${pid}" >/dev/null 2>&1; then
            kill "${pid}" >/dev/null 2>&1 || true
        fi
    done

    wait || true
}

trap cleanup INT TERM EXIT


# ============================================================
# Launch workers
# ============================================================

echo "============================================================"
echo "Launching SGLang"
echo "============================================================"
echo "Model:          ${model_path}"
echo "GPU range:      [${start_gpu}, ${end_gpu})"
echo "Host IP:        ${host_ip}"
echo "Base port:      ${start_port}"
echo "dtype:          ${dtype}"
echo "Mem fraction:   ${mem_frac}"
echo "Log directory:  ${log_dir}"
echo "============================================================"


for ((gpu_id = start_gpu; gpu_id < end_gpu; gpu_id++)); do
    port=$((start_port + gpu_id))

    log_file="${log_dir}/worker_${host_ip}_gpu_${gpu_id}_port_${port}.log"

    echo
    echo "Starting worker:"
    echo "  GPU:      ${gpu_id}"
    echo "  Port:     ${port}"
    echo "  Endpoint: http://${host_ip}:${port}"
    echo "  Log:      ${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    sglang serve \
        --model-path "${model_path}" \
        --host "${host}" \
        --port "${port}" \
        --dtype "${dtype}" \
        --mem-fraction-static "${mem_frac}" \
        "$@" \
        >"${log_file}" 2>&1 &

    pid=$!

    pids+=("${pid}")
    gpu_ids+=("${gpu_id}")
    ports+=("${port}")
    log_files+=("${log_file}")
done


# ============================================================
# Initial startup check
# ============================================================

sleep 3

startup_failed=0

for ((idx = 0; idx < ${#pids[@]}; idx++)); do
    pid=${pids[$idx]}
    gpu_id=${gpu_ids[$idx]}
    log_file=${log_files[$idx]}

    if ! kill -0 "${pid}" >/dev/null 2>&1; then
        echo
        echo "ERROR: Worker on GPU ${gpu_id} exited during startup."
        echo "Log: ${log_file}"
        echo "------------------------------------------------------------"
        tail -n 100 "${log_file}" || true
        echo "------------------------------------------------------------"

        startup_failed=1
    fi
done

if [[ "${startup_failed}" -ne 0 ]]; then
    exit 1
fi


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "Workers launched successfully"
echo "============================================================"

for ((idx = 0; idx < ${#pids[@]}; idx++)); do
    echo "  GPU ${gpu_ids[$idx]}:"
    echo "    PID:      ${pids[$idx]}"
    echo "    Endpoint: http://${host_ip}:${ports[$idx]}"
    echo "    Log:      ${log_files[$idx]}"
done

echo
echo "Heartbeat interval: ${heartbeat_interval}s"
echo "============================================================"

print_heartbeat


# ============================================================
# Start heartbeat
# ============================================================

heartbeat_loop &
heartbeat_pid=$!


# ============================================================
# Wait for workers
# ============================================================

wait "${pids[@]}"