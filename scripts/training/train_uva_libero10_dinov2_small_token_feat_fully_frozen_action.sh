#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
ACTION_CHECKPOINT="${ACTION_CHECKPOINT:-checkpoints/libero10.ckpt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-pretrained_models/dinov2/dinov2_vits14_pretrain.pth}"

detect_idle_gpus() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 <= 1000 && $3 <= 10) print $1}' \
        | paste -sd, -
}

if [[ -z "${GPU_IDS:-}" ]]; then
    GPU_IDS="$(detect_idle_gpus || true)"
fi
if [[ -z "${GPU_IDS}" ]]; then
    echo "No idle GPUs detected. Set GPU_IDS explicitly after checking nvidia-smi." >&2
    exit 3
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
if (( ${#GPU_ID_ARRAY[@]} != NUM_PROCESSES )); then
    echo "GPU_IDS count (${#GPU_ID_ARRAY[@]}) does not match NUM_PROCESSES (${NUM_PROCESSES})." >&2
    exit 2
fi

# Keep the effective batch and optimizer/LR step count invariant across the
# supported 1/2/4/8-GPU layouts.  Explicit overrides are accepted for tests.
if [[ -z "${PER_DEVICE_BATCH:-}" && -z "${GRAD_ACCUM_STEPS:-}" ]]; then
    case "${NUM_PROCESSES}" in
        8) PER_DEVICE_BATCH=16; GRAD_ACCUM_STEPS=1 ;;
        4) PER_DEVICE_BATCH=8;  GRAD_ACCUM_STEPS=4 ;;
        2) PER_DEVICE_BATCH=16; GRAD_ACCUM_STEPS=4 ;;
        1) PER_DEVICE_BATCH=16; GRAD_ACCUM_STEPS=8 ;;
        *) echo "Unsupported GPU count ${NUM_PROCESSES}; set PER_DEVICE_BATCH and GRAD_ACCUM_STEPS explicitly." >&2; exit 2 ;;
    esac
elif [[ -z "${PER_DEVICE_BATCH:-}" ]]; then
    if (( NUM_PROCESSES * GRAD_ACCUM_STEPS == 0 || EXPECTED_GLOBAL_BATCH % (NUM_PROCESSES * GRAD_ACCUM_STEPS) != 0 )); then
        echo "Cannot infer PER_DEVICE_BATCH for requested global batch." >&2
        exit 2
    fi
    PER_DEVICE_BATCH=$((EXPECTED_GLOBAL_BATCH / NUM_PROCESSES / GRAD_ACCUM_STEPS))
elif [[ -z "${GRAD_ACCUM_STEPS:-}" ]]; then
    if (( EXPECTED_GLOBAL_BATCH % (NUM_PROCESSES * PER_DEVICE_BATCH) != 0 )); then
        echo "Cannot infer GRAD_ACCUM_STEPS for requested global batch." >&2
        exit 2
    fi
    GRAD_ACCUM_STEPS=$((EXPECTED_GLOBAL_BATCH / NUM_PROCESSES / PER_DEVICE_BATCH))
fi

GLOBAL_BATCH=$((NUM_PROCESSES * PER_DEVICE_BATCH * GRAD_ACCUM_STEPS))
if (( GLOBAL_BATCH != EXPECTED_GLOBAL_BATCH )); then
    echo "Global batch mismatch: ${GLOBAL_BATCH}, expected ${EXPECTED_GLOBAL_BATCH}." >&2
    exit 2
fi

required_paths=(
    pretrained_models/vae/kl16.ckpt
    "${ACTION_CHECKPOINT}"
    "${DINO_CHECKPOINT}"
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

# Verify that the frozen action head is fully initialized from the intended
# checkpoint; otherwise a strict MAR freeze would lock random action weights.
"${PYTHON_BIN}" - "${ACTION_CHECKPOINT}" <<'PY'
import sys
import torch
from omegaconf import OmegaConf

path = sys.argv[1]
payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
cfg = payload["cfg"]
state = payload["state_dicts"]["ema_model"]
assert OmegaConf.select(cfg, "model.policy.action_model_params.predict_action") is True
assert OmegaConf.select(cfg, "model.policy.action_model_params.act_model_type") == "conv_ori"
assert list(OmegaConf.select(cfg, "task.shape_meta.action.shape")) == [10]
keys = [k for k in state if k.startswith("model.diffactloss.")]
assert len(keys) == 62, f"unexpected action-head key count: {len(keys)}"
print("Action checkpoint preflight: PASS")
PY

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

RUN_NAME="${RUN_NAME:-uva_libero10_dinov2_small_token_feat_fully_frozen_action_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

echo "DINOv2 config: uva_libero10_dinov2_small_token_feat_fully_frozen_action.yaml"
echo "DINO checkpoint: ${DINO_CHECKPOINT}"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "LR warmup optimizer steps: ${LR_WARMUP_STEPS}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    train.py
    --config-dir=unified_video_action/config
    --config-name=uva_libero10_dinov2_small_token_feat_fully_frozen_action.yaml
    model.policy.autoregressive_model_params.pretrained_model_path="${ACTION_CHECKPOINT}"
    model.policy.dinov2_teacher_params.checkpoint_path="${DINO_CHECKPOINT}"
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
    logging.tags="[libero10,dinov2_vits14_teacher,token_feat_alignment,fully_frozen_mar,pretrained_action_head,conv_ori,${NUM_PROCESSES}gpu]"
    hydra.run.dir="${RUN_DIR}"
)

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
