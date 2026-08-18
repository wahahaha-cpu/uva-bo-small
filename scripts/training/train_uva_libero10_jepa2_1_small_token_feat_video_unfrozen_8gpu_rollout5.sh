#!/usr/bin/env bash
set -euo pipefail

# Unfrozen JEPA token-feature baseline, initialized from the video-only model.
# Rollout and latest.ckpt are both refreshed every five epochs.
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
VIDEO_CHECKPOINT="${VIDEO_CHECKPOINT:-checkpoints/libero10_video.ckpt}"
WANDB_PROJECT="${WANDB_PROJECT:-uva-repa-jepa}"

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
    "${VIDEO_CHECKPOINT}"
    pretrained_models/jepa/vjepa2_1_vitb_dist_vitG_384.pt
    data/libero_10
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

RUN_NAME="${RUN_NAME:-uva_libero10_jepa2_1_small_token_feat_video_unfrozen_8gpu_rollout5_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

echo "JEPA config: uva_libero10_jepa2_1_small_token_feat.yaml"
echo "Video checkpoint: ${VIDEO_CHECKPOINT}"
echo "MAR: unfrozen"
echo "Action head: conv_fc"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "LR warmup optimizer steps: ${LR_WARMUP_STEPS}"
echo "Rollout/checkpoint interval: ${ROLLOUT_EVERY}/${CHECKPOINT_EVERY} epochs"
echo "W&B project: ${WANDB_PROJECT}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    train.py
    --config-dir=unified_video_action/config
    --config-name=uva_libero10_jepa2_1_small_token_feat.yaml
    model.policy.autoregressive_model_params.pretrained_model_path="${VIDEO_CHECKPOINT}"
    model.policy.action_model_params.predict_action=True
    model.policy.action_model_params.act_model_type=conv_fc
    model.policy.selected_training_mode=policy_model
    +model.policy.freeze_mar=False
    +model.policy.keep_mar_pos_and_fake_trainable=False
    +model.policy.keep_mar_action_head_trainable=False
    dataloader.batch_size="${PER_DEVICE_BATCH}"
    val_dataloader.batch_size="${PER_DEVICE_BATCH}"
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
    training.lr_warmup_steps="${LR_WARMUP_STEPS}"
    training.rollout_every="${ROLLOUT_EVERY}"
    training.checkpoint_every="${CHECKPOINT_EVERY}"
    training.resume=False
    checkpoint.save_last_ckpt=True
    logging.project="${WANDB_PROJECT}"
    logging.name="${RUN_NAME}"
    logging.tags="[libero10,jepa2_1_teacher,token_feat_alignment,video_pretrained_mar,unfrozen_mar,conv_fc,8gpu,rollout5]"
    hydra.run.dir="${RUN_DIR}"
)

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
