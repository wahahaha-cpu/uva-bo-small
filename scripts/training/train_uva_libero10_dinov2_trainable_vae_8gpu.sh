#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the MAR KL-VAE encoder in place of the lightweight Transformer while
# retaining the frozen DINOv2 teacher and the video-pretrained MAR/action setup.
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
VAE_ENCODER_LR_SCALE="${VAE_ENCODER_LR_SCALE:-1.0}"
VAE_STUDENT_FEATURE="${VAE_STUDENT_FEATURE:-encoder}"
DINO_FEATURE_LAYER="${DINO_FEATURE_LAYER:-final}"
CONFIG_NAME="${CONFIG_NAME:-uva_libero10_dinov2_trainable_vae_video_pretrained_conv_fc_action.yaml}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
VIDEO_CHECKPOINT="${VIDEO_CHECKPOINT:-checkpoints/libero10_video.ckpt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-pretrained_models/dinov2/dinov2_vits14_pretrain.pth}"
WANDB_PROJECT="${WANDB_PROJECT:-uva-repa-dinov2}"
LOG_TAGS="${LOG_TAGS:-[libero10,dinov2_vits14_teacher,${VAE_STUDENT_FEATURE}_vae_feature,dinov2_layer_${DINO_FEATURE_LAYER},trainable_vae_encoder,frozen_vae_decoder,video_pretrained_mar,unfrozen_mar,conv_fc,8gpu]}"
WAIT_FOR_GPU_FREE_MB="${WAIT_FOR_GPU_FREE_MB:-0}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

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
    "${DINO_CHECKPOINT}"
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

if (( WAIT_FOR_GPU_FREE_MB > 0 )); then
    echo "Waiting until all ${NUM_PROCESSES} selected GPUs have at least ${WAIT_FOR_GPU_FREE_MB} MiB free."
    while true; do
        READY_GPU_COUNT="$({
            nvidia-smi --id="${GPU_IDS}" \
                --query-gpu=memory.free \
                --format=csv,noheader,nounits
        } | awk -v threshold="${WAIT_FOR_GPU_FREE_MB}" '$1 >= threshold {count++} END {print count + 0}')"
        if (( READY_GPU_COUNT == NUM_PROCESSES )); then
            echo "GPU memory threshold reached; starting training."
            break
        fi
        printf '%s: %s/%s GPUs ready; checking again in %ss.\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" \
            "${READY_GPU_COUNT}" \
            "${NUM_PROCESSES}" \
            "${WAIT_POLL_SECONDS}"
        sleep "${WAIT_POLL_SECONDS}"
    done
fi

RUN_NAME="${RUN_NAME:-uva_libero10_dinov2_trainable_vae_8gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

echo "DINOv2 config: ${CONFIG_NAME}"
echo "Tokenizer: trainable MAR KL-VAE encoder features (decoder frozen)"
echo "VAE student feature: ${VAE_STUDENT_FEATURE}"
echo "DINO teacher feature layer: ${DINO_FEATURE_LAYER}"
echo "VAE encoder LR scale: ${VAE_ENCODER_LR_SCALE}"
echo "Video checkpoint: ${VIDEO_CHECKPOINT}"
echo "DINO checkpoint: ${DINO_CHECKPOINT}"
echo "MAR: unfrozen"
echo "Action head: conv_fc"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "Rollout/checkpoint interval: ${ROLLOUT_EVERY}/${CHECKPOINT_EVERY} epochs"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B tags: ${LOG_TAGS}"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    train.py
    --config-dir=unified_video_action/config
    --config-name="${CONFIG_NAME}"
    model.policy.autoregressive_model_params.pretrained_model_path="${VIDEO_CHECKPOINT}"
    model.policy.dinov2_teacher_params.checkpoint_path="${DINO_CHECKPOINT}"
    model.policy.vae_encoder_lr_scale="${VAE_ENCODER_LR_SCALE}"
    model.policy.vae_student_feature="${VAE_STUDENT_FEATURE}"
    model.policy.dinov2_teacher_params.feature_layer="${DINO_FEATURE_LAYER}"
    model.policy.action_model_params.predict_action=True
    model.policy.action_model_params.act_model_type=conv_fc
    model.policy.selected_training_mode=policy_model
    model.policy.freeze_mar=False
    model.policy.keep_mar_pos_and_fake_trainable=False
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
    logging.tags="${LOG_TAGS}"
    hydra.run.dir="${RUN_DIR}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
    printf '\n'
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "${launch_args[@]}"
