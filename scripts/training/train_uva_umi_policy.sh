#!/usr/bin/env bash
set -euo pipefail

# Run the original UMI policy-only path:
#   - the VAE follows the original frozen kl16.ckpt path;
#   - the official UMI checkpoint is passed through the existing
#     autoregressive_model_params.pretrained_model_path mechanism;
#   - the only selected task mode is policy_model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29517}"

# DATASET_PATH must contain the extracted directories named below. The UMI
# dataloader opens <dataset>.zarr directly; a .zarr.zip file is not sufficient.
DATASET_PATH="${DATASET_PATH:-data/umi_data}"
UMI_CHECKPOINT="${UMI_CHECKPOINT:-checkpoints/umi_multitask.ckpt}"
INDICES_FILE="${INDICES_FILE:-prepared_data/sampled_500_index_3_datasets.json}"
RUN_NAME="${RUN_NAME:-uva_umi_policy_$(date +%Y%m%d_%H%M%S)}"
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
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

dataset_names=(cup_arrangement_0)
missing_dataset=0
for dataset_name in "${dataset_names[@]}"; do
    dataset_path="${DATASET_PATH}/${dataset_name}.zarr.zip"
    if [[ ! -f "${dataset_path}" ]]; then
        echo "Missing UMI dataset: ${dataset_path}" >&2
        missing_dataset=1
    fi
done
if (( missing_dataset != 0 )); then
    echo "Expected .zarr.zip files under ${DATASET_PATH}." >&2
    echo "A .zarr.zip file must be processed before training; see process_dataset/download_dataset.py and extract_umi_data.py." >&2
    exit 1
fi

# Verify the standard workspace checkpoint format before launching multiple
# workers. Actual parameter loading remains in the original policy code path.
"${PYTHON_BIN}" - "${UMI_CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
payload = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
    mmap=True,
)
state_dicts = payload.get("state_dicts", {})
ema_model = state_dicts.get("ema_model")
if not isinstance(ema_model, dict) or not ema_model:
    raise SystemExit("Checkpoint does not contain state_dicts.ema_model")
print(
    "UMI checkpoint preflight: PASS "
    f"(ema keys={len(ema_model)})"
)
PY

echo "Dataset root: ${DATASET_PATH}"
echo "UMI pretrained checkpoint: ${UMI_CHECKPOINT}"
echo "Task mode: policy_model"
echo "VAE: original frozen kl16.ckpt path"
echo "GPUs: ${NUM_PROCESSES}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    --main_process_port="${MAIN_PROCESS_PORT}"
    train.py
    --config-dir=.
    --config-name=uva_umi_multi.yaml
    model.policy.selected_training_mode=policy_model
    model.policy.action_model_params.predict_action=True
    model.policy.use_proprioception=True
    model.policy.predict_proprioception=True
    model.policy.shift_action=False
    model.policy.different_history_freq=True
    model.policy.autoregressive_model_params.pretrained_model_path="${UMI_CHECKPOINT}"
    model.policy.optimizer.learning_rate=1e-4
    task.dataset.dataset_root_dir="${DATASET_PATH}"
    task.dataset.used_episode_indices_file=""
    training.resume=False
    logging.project=uva
    hydra.run.dir="${RUN_DIR}"
)

exec "${ACCELERATE_BIN}" launch "${launch_args[@]}"
