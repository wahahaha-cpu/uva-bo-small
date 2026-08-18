#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ABLATION="${1:-dino_jepa}"
case "${ABLATION}" in
    baseline)
        CONFIG_NAME="uva_libero10_dino_jepa_minimal_baseline"
        ;;
    dino|dino_only)
        CONFIG_NAME="uva_libero10_dino_jepa_minimal_dino_only"
        ;;
    jepa|jepa_only)
        CONFIG_NAME="uva_libero10_dino_jepa_minimal_jepa_only"
        ;;
    dino_jepa|both)
        CONFIG_NAME="uva_libero10_dino_jepa_minimal"
        ;;
    *)
        echo "Unknown ablation '${ABLATION}'. Use baseline, dino, jepa, or dino_jepa." >&2
        exit 2
        ;;
esac

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
RUN_SANITY_CHECK="${RUN_SANITY_CHECK:-1}"
SANITY_GPU="${SANITY_GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

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
    data/libero_10
)
if [[ "${ABLATION}" == "dino" || "${ABLATION}" == "dino_only" || "${ABLATION}" == "dino_jepa" || "${ABLATION}" == "both" ]]; then
    required_paths+=(pretrained_models/dinov2/dinov2_vits14_pretrain.pth)
fi
if [[ "${ABLATION}" == "jepa" || "${ABLATION}" == "jepa_only" || "${ABLATION}" == "dino_jepa" || "${ABLATION}" == "both" ]]; then
    required_paths+=(pretrained_models/jepa/vjepa2_1_vitb_dist_vitG_384.pt)
fi
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9}"
if [[ "${PYTHON_BIN}" != */* ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        exit 1
    }
elif [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
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

if [[ "${RUN_SANITY_CHECK}" == "1" && "${ABLATION}" != "baseline" ]]; then
    CUDA_VISIBLE_DEVICES="${SANITY_GPU}" "${PYTHON_BIN}" \
        scripts/verify_dino_jepa_minimal.py \
        --config-name "${CONFIG_NAME}" \
        --device cuda
fi

RUN_NAME="${RUN_NAME:-${CONFIG_NAME}_8gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

echo "Ablation: ${ABLATION}"
echo "Config: ${CONFIG_NAME}.yaml"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    train.py
    --config-dir=unified_video_action/config
    --config-name="${CONFIG_NAME}.yaml"
    dataloader.batch_size="${PER_DEVICE_BATCH}"
    val_dataloader.batch_size="${PER_DEVICE_BATCH}"
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
    training.lr_warmup_steps="${LR_WARMUP_STEPS}"
    training.resume=False
    logging.name="${RUN_NAME}"
    hydra.run.dir="${RUN_DIR}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q launch' "${GPU_IDS}" "${ACCELERATE_BIN}"
    printf ' %q' "${launch_args[@]}"
    printf '\n'
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
