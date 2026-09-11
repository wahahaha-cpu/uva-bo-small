#!/usr/bin/env bash
set -euo pipefail

# Run the original UMI path with the frozen KL-VAE:
#   - the VAE follows the original frozen kl16.ckpt path;
#   - the official UMI checkpoint is passed through the existing
#     autoregressive_model_params.pretrained_model_path mechanism;
#   - the selected task mode can train policy-only or full dynamics.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29517}"
TASK_MODE="${TASK_MODE:-full_dynamic_model}"
ENABLE_GRIPPER_LOSS_WEIGHT="${ENABLE_GRIPPER_LOSS_WEIGHT:-false}"
GRIPPER_LOSS_WEIGHT="${GRIPPER_LOSS_WEIGHT:-5.0}"
RESUME_TRAINING="${RESUME_TRAINING:-false}"

# The supplied archive uses the spelling cup_arrangment_0. Keep the canonical
# dataset key expected by the original UMI config, but stage the new zarr under
# /dev/shm so training reads from tmpfs without changing the model path.
DATASET_PATH="${DATASET_PATH:-data/umi_data}"
DATASET_NAME="${DATASET_NAME:-cup_arrangement_2}"
CANONICAL_DATASET_NAME="${CANONICAL_DATASET_NAME:-cup_arrangement_0}"
SHM_DATASET_PATH="${SHM_DATASET_PATH:-/dev/shm/uva_umi_dataset_cup_arrangement_2}"
STAGE_TO_SHM="${STAGE_TO_SHM:-1}"
UMI_CHECKPOINT="${UMI_CHECKPOINT:-checkpoints/umi_multitask.ckpt}"
INDICES_FILE="${INDICES_FILE:-prepared_data/sampled_500_index_3_datasets.json}"
RUN_NAME="${RUN_NAME:-uva_umi_policy_ar02$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
    exit 1
fi

required_paths=(
    pretrained_models/vae/kl16.ckpt
    "${UMI_CHECKPOINT}"
)
if [[ -n "${INDICES_FILE}" ]]; then
    required_paths+=("${INDICES_FILE}")
fi
if [[ "${RESUME_TRAINING}" == "true" ]]; then
    required_paths+=("${RUN_DIR}/checkpoints/latest.ckpt")
fi
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

source_zip="${DATASET_PATH}/${DATASET_NAME}.zarr.zip"
source_zarr="${DATASET_PATH}/${DATASET_NAME}.zarr"
if [[ ! -f "${source_zip}" || ! -d "${source_zarr}" ]]; then
    echo "Missing extracted UMI dataset: ${source_zarr}" >&2
    echo "Expected both ${source_zip} and ${source_zarr}." >&2
    exit 1
fi


# Verify the standard workspace checkpoint format before launching multiple
# workers. Actual parameter loading remains in the original policy code path.


echo "Source dataset: ${source_zarr}"
echo "UMI pretrained checkpoint: ${UMI_CHECKPOINT}"
echo "Task mode: ${TASK_MODE}"
echo "Gripper loss weighting: ${ENABLE_GRIPPER_LOSS_WEIGHT} (weight=${GRIPPER_LOSS_WEIGHT})"
echo "Resume training: ${RESUME_TRAINING}"
echo "VAE: original frozen kl16.ckpt path"
echo "GPUs: ${NUM_PROCESSES}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    --main_process_port="${MAIN_PROCESS_PORT}"
    train.py
    --config-dir=.
    --config-name=uva_umi_multi.yaml
    model.policy.selected_training_mode="${TASK_MODE}"
    model.policy.autoregressive_model_params.predict_video=True
    model.policy.action_model_params.predict_action=True
    model.policy.action_model_params.enable_gripper_loss_weight="${ENABLE_GRIPPER_LOSS_WEIGHT}"
    model.policy.action_model_params.gripper_loss_weight="${GRIPPER_LOSS_WEIGHT}"
    model.policy.use_proprioception=True
    model.policy.predict_proprioception=True
    model.policy.shift_action=False
    model.policy.different_history_freq=True
    model.policy.autoregressive_model_params.pretrained_model_path="${UMI_CHECKPOINT}"
    model.policy.optimizer.learning_rate=1e-4
    task.dataset.dataset_root_dir="/dev/shm/uva_umi_cup_arrangement_2_20260826_004423"
    task.dataset.used_episode_indices_file=""
    training.resume="${RESUME_TRAINING}"
    logging.project=uva
    hydra.run.dir="${RUN_DIR}"
)

exec "${ACCELERATE_BIN}" launch "${launch_args[@]}"
