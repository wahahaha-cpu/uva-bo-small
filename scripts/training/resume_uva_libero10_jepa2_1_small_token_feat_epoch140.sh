#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Preserve the original run's optimization setup exactly.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"

RUN_DIR="${RUN_DIR:-checkpoints/uva_libero10_jepa2_1_small_token_feat_20260731_153521}"
RESUME_CHECKPOINT="${RUN_DIR}/checkpoints/latest.ckpt"

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ID_ARRAY[@]} != NUM_PROCESSES )); then
    echo "GPU_IDS count (${#GPU_ID_ARRAY[@]}) does not match NUM_PROCESSES (${NUM_PROCESSES})." >&2
    exit 2
fi

GLOBAL_BATCH=$((NUM_PROCESSES * PER_DEVICE_BATCH * GRAD_ACCUM_STEPS))
if (( GLOBAL_BATCH != EXPECTED_GLOBAL_BATCH )); then
    echo "Global batch mismatch: ${GLOBAL_BATCH}, expected ${EXPECTED_GLOBAL_BATCH}." >&2
    exit 2
fi

required_paths=(
    pretrained_models/vae/kl16.ckpt
    checkpoints/libero10_video.ckpt
    pretrained_models/jepa/vjepa2_1_vitb_dist_vitG_384.pt
    data/libero_10
    "${RESUME_CHECKPOINT}"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
    if command -v accelerate >/dev/null 2>&1; then
        ACCELERATE_BIN="accelerate"
    else
        ACCELERATE_BIN="/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate"
    fi
fi
if [[ "${ACCELERATE_BIN}" != "accelerate" && ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
    exit 1
fi

echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
echo "Expected saved state: epoch=140, global_step=547220"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    train.py
    --config-dir=unified_video_action/config
    --config-name=uva_libero10_jepa2_1_small_token_feat.yaml
    dataloader.batch_size="${PER_DEVICE_BATCH}"
    val_dataloader.batch_size="${PER_DEVICE_BATCH}"
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
    training.lr_warmup_steps="${LR_WARMUP_STEPS}"
    training.resume=True
    hydra.run.dir="${RUN_DIR}"
)
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
