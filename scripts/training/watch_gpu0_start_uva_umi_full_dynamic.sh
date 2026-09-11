#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_DIR}/scripts/training/train_uva_umi_dinov2_student_vae_latent_full_dynamic_8gpu.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MEMORY_LIMIT_MIB="${MEMORY_LIMIT_MIB:-1024}"
TRAIN_SESSION="${TRAIN_SESSION:-uva_umi_full_dynamic_01}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/checkpoints/uva_umi_full_dynamic_gpu_watch.log}"

mkdir -p "$(dirname -- "${LOG_FILE}")"
exec >>"${LOG_FILE}" 2>&1

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"

echo "[$(date '+%F %T')] watcher started; GPUs=${GPU_IDS}, limit=${MEMORY_LIMIT_MIB} MiB"

while true; do
    if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
        echo "[$(date '+%F %T')] ${TRAIN_SESSION} already exists; watcher exiting"
        exit 0
    fi

    all_free=1
    status_line=""
    for gpu_id in "${GPU_ID_ARRAY[@]}"; do
        used_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
        if [[ ! "${used_mib}" =~ ^[0-9]+$ ]]; then
            echo "[$(date '+%F %T')] unable to read GPU ${gpu_id} memory: ${used_mib}"
            all_free=0
            break
        fi
        status_line+="gpu${gpu_id}=${used_mib}MiB "
        if (( used_mib > MEMORY_LIMIT_MIB )); then
            all_free=0
        fi
    done

    echo "[$(date '+%F %T')] ${status_line}"

    if (( all_free == 1 )); then
        echo "[$(date '+%F %T')] all training GPUs are free; rechecking after 10 seconds"
        sleep 10
        still_free=1
        for gpu_id in "${GPU_ID_ARRAY[@]}"; do
            used_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
            if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > MEMORY_LIMIT_MIB )); then
                still_free=0
                break
            fi
        done

        if (( still_free == 1 )); then
            echo "[$(date '+%F %T')] starting ${TRAIN_SESSION}"
            tmux new-session -d -s "${TRAIN_SESSION}" -c "${PROJECT_DIR}" \
                "exec bash '${TRAIN_SCRIPT}'"
            echo "[$(date '+%F %T')] training session created; watcher exiting"
            exit 0
        fi
        echo "[$(date '+%F %T')] a GPU became busy during the recheck"
    fi

    sleep "${POLL_SECONDS}"
done
