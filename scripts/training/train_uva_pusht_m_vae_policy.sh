#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Keep CUDA, MuJoCo, W&B, checkpoints, and the dataset on the configured data1-backed paths.
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"

PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate}"
CONFIG_NAME="${CONFIG_NAME:-uva_pusht_m_vae_policy.yaml}"
DATASET_PATH="${DATASET_PATH:-data/pusht_multitask}"
MAR_CHECKPOINT="${MAR_CHECKPOINT:-pretrained_models/mar/mar_base/checkpoint-last.pth}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-pretrained_models/vae/kl16.ckpt}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29517}"
RUN_NAME="${RUN_NAME:-uva_pusht_m_vae_policy_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/checkpoints/${RUN_NAME}}"

ensure_data1_parent() {
    local variable_name="$1"
    local path_value="${!variable_name}"
    if [[ "${path_value}" != /* ]]; then
        path_value="${REPO_ROOT}/${path_value}"
        printf -v "${variable_name}" '%s' "${path_value}"
    fi
    local resolved_path
    if [[ -e "${path_value}" || -L "${path_value}" ]]; then
        resolved_path="$(readlink -f "${path_value}")"
    else
        resolved_path="$(readlink -f "$(dirname "${path_value}")")"
    fi
    case "${resolved_path}" in
        /data1/*)
            ;;
        *)
            echo "Large-data path must resolve under /data1: ${path_value}" >&2
            exit 2
            ;;
    esac
}

for variable_name in DATASET_PATH MAR_CHECKPOINT VAE_CHECKPOINT RUN_DIR WANDB_DIR; do
    ensure_data1_parent "${variable_name}"
done

if [[ "${PER_DEVICE_BATCH}" != "8" ]]; then
    echo "PER_DEVICE_BATCH must be exactly 8 for this experiment (got ${PER_DEVICE_BATCH})." >&2
    exit 2
fi
if [[ ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer (got ${NUM_PROCESSES})." >&2
    exit 2
fi
if [[ ! "${GRAD_ACCUM_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRAD_ACCUM_STEPS must be a positive integer (got ${GRAD_ACCUM_STEPS})." >&2
    exit 2
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ID_ARRAY[@]} != NUM_PROCESSES )); then
    echo "GPU_IDS count (${#GPU_ID_ARRAY[@]}) does not match NUM_PROCESSES (${NUM_PROCESSES})." >&2
    exit 2
fi

GLOBAL_BATCH=$((NUM_PROCESSES * PER_DEVICE_BATCH * GRAD_ACCUM_STEPS))
if (( GLOBAL_BATCH != 256 )); then
    echo "Global batch mismatch: ${GLOBAL_BATCH}; expected 256 (official PushT training layout)." >&2
    echo "Use NUM_PROCESSES * 8 * GRAD_ACCUM_STEPS = 256." >&2
    exit 2
fi

required_paths=(
    "${DATASET_PATH}/data/img"
    "${DATASET_PATH}/data/state"
    "${DATASET_PATH}/data/action"
    "${DATASET_PATH}/meta/episode_ends"
    "${MAR_CHECKPOINT}"
    "${VAE_CHECKPOINT}"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${DATASET_PATH}" <<'PY'
import sys
import zarr

group = zarr.open(sys.argv[1], mode="r")
expected = {
    "data/img": (35927, 96, 96, 3),
    "data/state": (35927, 5),
    "data/action": (35927, 2),
    "meta/episode_ends": (247,),
}
for key, shape in expected.items():
    actual = tuple(group[key].shape)
    if actual != shape:
        raise SystemExit(f"Unexpected PushT-M shape for {key}: {actual}; expected {shape}")
print("PushT-M dataset shape check: PASS")
PY

echo "Experiment: PushT-M, VAE encoder trainable, action-only policy mode"
echo "Dataset: ${DATASET_PATH}"
echo "MAR checkpoint: ${MAR_CHECKPOINT}"
echo "VAE initialization: ${VAE_CHECKPOINT}"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    --main_process_port="${MAIN_PROCESS_PORT}"
    --mixed_precision="${MIXED_PRECISION}"
    train.py
    --config-dir=unified_video_action/config
    --config-name="${CONFIG_NAME}"
    task.dataset.dataset_path="${DATASET_PATH}"
    task.dataset.dataset_type=multitask
    task.env_runner.fix_goal=False
    task.env_runner.n_train=1
    task.env_runner.n_test=50
    model.policy.selected_training_mode=policy_model
    model.policy.use_student_tokenizer=True
    model.policy.student_tokenizer_backend=vae
    model.policy.vae_student_mode=sample
    model.policy.vae_student_feature=latent
    model.policy.freeze_mar=False
    model.policy.autoregressive_model_params.pretrained_model_path="${MAR_CHECKPOINT}"
    model.policy.vae_model_params.autoencoder_path="${VAE_CHECKPOINT}"
    model.policy.autoregressive_model_params.predict_video=True
    model.policy.action_model_params.predict_action=True
    model.policy.action_model_params.act_model_type=conv_fc
    model.policy.optimizer.learning_rate=1e-4
    dataloader.batch_size=8
    val_dataloader.batch_size=8
    training.lr_warmup_steps=1000
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
    training.resume=False
    logging.name="${RUN_NAME}"
    hydra.run.dir="${RUN_DIR}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
    printf '\n'
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
