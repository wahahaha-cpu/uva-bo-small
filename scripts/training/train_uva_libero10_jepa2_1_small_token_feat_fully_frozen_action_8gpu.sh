#!/usr/bin/env bash
set -euo pipefail

# Libero rollout needs MuJoCo/NVIDIA runtime libraries. Accelerate imports
# DeepSpeed while unwrapping the EMA model, which requires a CUDA toolkit root.
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Default experiment: 8 GPUs x batch 16 x accumulation 1 = global batch 128.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
ACTION_CHECKPOINT="${ACTION_CHECKPOINT:-checkpoints/libero10.ckpt}"

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
    "${ACTION_CHECKPOINT}"
    pretrained_models/jepa/vjepa2_1_vitb_dist_vitG_384.pt
    data/libero_10
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

# Strictly frozen MAR must never retain a randomly initialized action head.
"${PYTHON_BIN}" -c '
import sys

import torch
from omegaconf import OmegaConf

path = sys.argv[1]
payload = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
    mmap=True,
)
cfg = payload["cfg"]
state = payload["state_dicts"]["ema_model"]
predict_action = OmegaConf.select(
    cfg, "model.policy.action_model_params.predict_action"
)
act_model_type = OmegaConf.select(
    cfg, "model.policy.action_model_params.act_model_type"
)
action_shape = OmegaConf.select(cfg, "task.shape_meta.action.shape")
action_keys = [key for key in state if key.startswith("model.diffactloss.")]

assert predict_action is True, "Checkpoint predict_action is not true."
assert act_model_type == "conv_ori", (
    f"Checkpoint action head is {act_model_type!r}, expected conv_ori."
)
assert list(action_shape) == [10], (
    f"Checkpoint action shape is {action_shape!r}, expected [10]."
)
assert len(action_keys) == 62, (
    f"Checkpoint contains {len(action_keys)} diffactloss keys, expected 62."
)
print(
    "Action checkpoint preflight: PASS "
    f"(type={act_model_type}, action_shape={list(action_shape)}, "
    f"diffactloss_keys={len(action_keys)})"
)
' "${ACTION_CHECKPOINT}"

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

RUN_NAME="${RUN_NAME:-uva_libero10_jepa2_1_small_token_feat_fully_frozen_action_8gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

echo "JEPA config: uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.yaml"
echo "Action checkpoint: ${ACTION_CHECKPOINT}"
echo "Action head: conv_ori (fully frozen after exact checkpoint load)"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    train.py
    --config-dir=unified_video_action/config
    --config-name=uva_libero10_jepa2_1_small_token_feat_fully_frozen_mar.yaml
    model.policy.autoregressive_model_params.pretrained_model_path="${ACTION_CHECKPOINT}"
    model.policy.action_model_params.predict_action=True
    model.policy.action_model_params.act_model_type=conv_ori
    model.policy.selected_training_mode=policy_model
    model.policy.freeze_mar=True
    model.policy.keep_mar_pos_and_fake_trainable=False
    dataloader.batch_size="${PER_DEVICE_BATCH}"
    val_dataloader.batch_size="${PER_DEVICE_BATCH}"
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
    training.lr_warmup_steps="${LR_WARMUP_STEPS}"
    training.resume=False
    logging.tags="[libero10,jepa2_1_teacher,token_feat_alignment,fully_frozen_mar,pretrained_action_head,conv_ori,8gpu]"
    hydra.run.dir="${RUN_DIR}"
)
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
