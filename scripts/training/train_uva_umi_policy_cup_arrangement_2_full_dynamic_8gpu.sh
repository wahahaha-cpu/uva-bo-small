#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the previous frozen-VAE fully-dynamic experiment unchanged and replace
# only the source zarr. UmiMultiDataset still expects the canonical dataset key.
export DATASET_NAME="cup_arrangement_1"
export CANONICAL_DATASET_NAME="cup_arrangement_0"
export SHM_DATASET_PATH="/dev/shm/uva_umi_dataset_cup_arrangement_1"
export TASK_MODE="full_dynamic_model"
export ENABLE_GRIPPER_LOSS_WEIGHT="${ENABLE_GRIPPER_LOSS_WEIGHT:-true}"
export GRIPPER_LOSS_WEIGHT="${GRIPPER_LOSS_WEIGHT:-5.0}"
export RUN_NAME="${RUN_NAME:-uva_umi_full_dynamic_cup_arrangement_1_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/train_uva_umi_policy.sh"
