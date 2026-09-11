#!/usr/bin/env bash
set -euo pipefail

# Two-stage Libero10 run from the official MAR-default config, with a trainable
# student tokenizer, a frozen DINOv2 token-feature teacher, and a frozen VAE
# video target. Stage 2 continues from the stage 1 checkpoint.
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-128}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2000}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_CONSECUTIVE_NONFINITE_UPDATES="${MAX_CONSECUTIVE_NONFINITE_UPDATES:-8}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-pretrained_models/dinov2/dinov2_vits14_pretrain.pth}"
DATASET_PATH="${DATASET_PATH:-/data1/local_userdata/jinboning/uva-bo/data/libero_10}"
WANDB_PROJECT="${WANDB_PROJECT:-uva-repa-dinov2}"
STAGE="${STAGE:-all}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3050}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3050}"
STAGE1_RESUME="${STAGE1_RESUME:-false}"
STAGE1_MAX_TRAIN_STEPS="${STAGE1_MAX_TRAIN_STEPS:-null}"
STAGE1_WANDB_ID="${STAGE1_WANDB_ID:-null}"
DRY_RUN="${DRY_RUN:-0}"

case "${STAGE}" in
    all|1|stage1|video|2|stage2|mixed) ;;
    *)
        echo "STAGE must be one of: all, stage1, stage2." >&2
        exit 2
        ;;
esac

case "${STAGE1_RESUME}" in
    true|True|false|False) ;;
    *)
        echo "STAGE1_RESUME must be true or false." >&2
        exit 2
        ;;
esac

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
    "${DINO_CHECKPOINT}"
    "${DATASET_PATH}"
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

RUN_NAME_BASE="${RUN_NAME:-uva_libero10_dinov2_student_mar_only_two_stage_8gpu_$(date +%Y%m%d_%H%M%S)}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${RUN_NAME_BASE}_stage1_video}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${RUN_NAME_BASE}_stage2_mixed}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-${RUN_DIR:-checkpoints/${STAGE1_RUN_NAME}}}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-checkpoints/${STAGE2_RUN_NAME}}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${STAGE1_RUN_DIR}/checkpoints/latest.ckpt}"
STAGE1_RESUME_CHECKPOINT="${STAGE1_RESUME_CHECKPOINT:-${STAGE1_CHECKPOINT}}"

echo "Base config: uva_libero10.yaml + model/student_tokenizer=small"
echo "MAR checkpoint: model/uva.yaml default"
echo "DINO checkpoint: ${DINO_CHECKPOINT}"
echo "Dataset: ${DATASET_PATH}"
echo "MAR: unfrozen"
echo "Student tokenizer: transformer (trainable); DINOv2/VAE: frozen"
echo "Video diffusion target: frozen VAE future latent"
echo "Action head: conv_fc"
echo "Stages: ${STAGE} (video-only ${STAGE1_EPOCHS} epochs, mixed ${STAGE2_EPOCHS} epochs)"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "LR warmup optimizer steps: ${LR_WARMUP_STEPS}"
echo "Mixed precision / max grad norm: ${MIXED_PRECISION} / ${MAX_GRAD_NORM}"
echo "Rollout/checkpoint interval: ${ROLLOUT_EVERY}/${CHECKPOINT_EVERY} epochs"
echo "W&B project: ${WANDB_PROJECT}"
echo "Stage 1 directory: ${STAGE1_RUN_DIR}"
echo "Stage 2 directory: ${STAGE2_RUN_DIR}"
echo "Stage 2 MAR checkpoint: ${STAGE1_CHECKPOINT}"
echo "Stage 1 resume: ${STAGE1_RESUME}"
if [[ "${STAGE1_RESUME,,}" == "true" ]]; then
    echo "Stage 1 resume checkpoint: ${STAGE1_RESUME_CHECKPOINT}"
fi

launch_stage() {
    local stage_name="$1"
    shift
    echo "Launching ${stage_name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q %q launch' "${GPU_IDS}" "${ACCELERATE_BIN}"
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch "$@"
}

run_stage1() {
    if [[ "${STAGE1_RESUME,,}" == "true" && ! -f "${STAGE1_RESUME_CHECKPOINT}" ]]; then
        echo "Stage 1 resume checkpoint not found: ${STAGE1_RESUME_CHECKPOINT}" >&2
        exit 1
    fi

    local -a launch_args=(
        --num_processes="${NUM_PROCESSES}"
        --mixed_precision="${MIXED_PRECISION}"
        train.py
        --config-dir=unified_video_action/config
        --config-name=uva_libero10.yaml
        model/student_tokenizer=small
        model.policy.autoregressive_model_params.predict_video=True
        +model.policy.teacher_type=dinov2
        +model.policy.dinov2_teacher_params.model_name=dinov2_vits14
        +model.policy.dinov2_teacher_params.img_size=224
        +model.policy.dinov2_teacher_params.model_img_size=518
        +model.policy.dinov2_teacher_params.checkpoint_path="${DINO_CHECKPOINT}"
        +model.policy.dinov2_teacher_params.loader=timm
        +model.policy.dinov2_teacher_params.feature_layer=final
        +model.policy.video_target=vae
        model.policy.action_model_params.predict_action=False
        model.policy.action_model_params.act_model_type=conv_fc
        model.policy.selected_training_mode=video_model
        model.policy.use_history_action=False
        task.dataset.dataset_path="${DATASET_PATH}"
        task.env_runner.dataset_path="${DATASET_PATH}"
        task.env_runner.past_action=False
        task.env_runner.n_envs=1
        dataloader.batch_size="${PER_DEVICE_BATCH}"
        val_dataloader.batch_size="${PER_DEVICE_BATCH}"
        training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
        training.mixed_precision="${MIXED_PRECISION}"
        training.max_grad_norm="${MAX_GRAD_NORM}"
        training.max_consecutive_nonfinite_updates="${MAX_CONSECUTIVE_NONFINITE_UPDATES}"
        training.lr_warmup_steps="${LR_WARMUP_STEPS}"
        training.num_epochs="${STAGE1_EPOCHS}"
        training.rollout_every="${ROLLOUT_EVERY}"
        training.checkpoint_every="${CHECKPOINT_EVERY}"
        training.resume="${STAGE1_RESUME}"
        checkpoint.save_last_ckpt=True
        logging.project="${WANDB_PROJECT}"
        logging.name="${STAGE1_RUN_NAME}"
        logging.tags="[libero10,stage1_video_only,dinov2_vits14_final,token_feat_alignment,no_history_action,video_prediction,student_tokenizer,frozen_vae,frozen_dino,mar_base_initialization,unfrozen_mar,conv_fc,8gpu,batch16,rollout5]"
        hydra.run.dir="${STAGE1_RUN_DIR}"
    )

    if [[ "${STAGE1_MAX_TRAIN_STEPS}" != "null" ]]; then
        launch_args+=(training.max_train_steps="${STAGE1_MAX_TRAIN_STEPS}")
    fi
    if [[ "${STAGE1_WANDB_ID}" != "null" ]]; then
        launch_args+=(logging.id="${STAGE1_WANDB_ID}")
    fi
    if [[ "${STAGE1_RESUME,,}" == "true" ]]; then
        # The full workspace checkpoint restores model, EMA, optimizer, and
        # scheduler; avoid an unnecessary MAR warm-start load before restore.
        launch_args+=(model.policy.autoregressive_model_params.pretrained_model_path=null)
        launch_args+=(training.resume_checkpoint_path="${STAGE1_RESUME_CHECKPOINT}")
    fi
    launch_stage "stage1-video-only" "${launch_args[@]}"
}

run_stage2() {
    if [[ "${DRY_RUN}" != "1" && ! -f "${STAGE1_CHECKPOINT}" ]]; then
        echo "Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
        echo "Run STAGE=stage1 first or set STAGE1_CHECKPOINT explicitly." >&2
        exit 1
    fi

    local -a launch_args=(
        --num_processes="${NUM_PROCESSES}"
        --mixed_precision="${MIXED_PRECISION}"
        train.py
        --config-dir=unified_video_action/config
        --config-name=uva_libero10.yaml
        model/student_tokenizer=small
        model.policy.autoregressive_model_params.pretrained_model_path="${STAGE1_CHECKPOINT}"
        model.policy.autoregressive_model_params.predict_video=True
        +model.policy.teacher_type=dinov2
        +model.policy.dinov2_teacher_params.model_name=dinov2_vits14
        +model.policy.dinov2_teacher_params.img_size=224
        +model.policy.dinov2_teacher_params.model_img_size=518
        +model.policy.dinov2_teacher_params.checkpoint_path="${DINO_CHECKPOINT}"
        +model.policy.dinov2_teacher_params.loader=timm
        +model.policy.dinov2_teacher_params.feature_layer=final
        +model.policy.video_target=vae
        model.policy.action_model_params.predict_action=True
        model.policy.action_model_params.act_model_type=conv_fc
        model.policy.selected_training_mode=null
        task.task_modes=[]
        model.policy.use_history_action=True
        # Real history belongs to causal policy modes; the other modes keep the
        # architecture-compatible fake-history token.
        +model.policy.history_action_modes=[policy_model,full_dynamic_model]
        task.dataset.dataset_path="${DATASET_PATH}"
        # get_trajectory keeps a0..15 / a16..31; dropping obs0 leaves the 32-frame
        # visual window obs1..32, whose condition endpoint is obs16.
        task.dataset.horizon=33
        task.env_runner.dataset_path="${DATASET_PATH}"
        task.env_runner.past_action=True
        task.env_runner.n_envs=1
        dataloader.batch_size="${PER_DEVICE_BATCH}"
        val_dataloader.batch_size="${PER_DEVICE_BATCH}"
        training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
        training.mixed_precision="${MIXED_PRECISION}"
        training.max_grad_norm="${MAX_GRAD_NORM}"
        training.max_consecutive_nonfinite_updates="${MAX_CONSECUTIVE_NONFINITE_UPDATES}"
        training.lr_warmup_steps="${LR_WARMUP_STEPS}"
        training.num_epochs="${STAGE2_EPOCHS}"
        training.rollout_every="${ROLLOUT_EVERY}"
        training.checkpoint_every="${CHECKPOINT_EVERY}"
        training.resume=False
        checkpoint.save_last_ckpt=True
        logging.project="${WANDB_PROJECT}"
        logging.name="${STAGE2_RUN_NAME}"
        logging.tags="[libero10,stage2_five_mode_mix,dinov2_vits14_final,token_feat_alignment,history_action_policy_modes,video_prediction,action_prediction,student_tokenizer,frozen_vae,frozen_dino,mar_stage1_continuation,unfrozen_mar,conv_fc,8gpu,batch16,rollout5]"
        hydra.run.dir="${STAGE2_RUN_DIR}"
    )
    launch_stage "stage2-five-mode-mix" "${launch_args[@]}"
}

case "${STAGE}" in
    all)
        run_stage1
        run_stage2
        ;;
    1|stage1|video)
        run_stage1
        ;;
    2|stage2|mixed)
        run_stage2
        ;;
esac
