#!/usr/bin/env bash
set -euo pipefail

# Train the UMI policy with a student tokenizer supervised jointly by shallow
# DINOv2 features and frozen KL-VAE final latents.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/python3.9}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29518}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-56}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-448}"
NUM_WORKERS="${NUM_WORKERS:-14}"

CONFIG_NAME="${CONFIG_NAME:-uva_umi_dinov2_student_vae_latent_policy.yaml}"
SOURCE_DATASET_PATH="${DATASET_PATH:-data/umi_data}"
UMI_CHECKPOINT="${UMI_CHECKPOINT:-checkpoints/umi_multitask.ckpt}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-pretrained_models/dinov2/dinov2_vits14_pretrain.pth}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-pretrained_models/vae/kl16.ckpt}"
DINO_ALIGN_COEFF="${DINO_ALIGN_COEFF:-0.02}"
VAE_LATENT_COEFF="${VAE_LATENT_COEFF:-0.05}"
USE_PROPRIOCEPTION="${USE_PROPRIOCEPTION:-True}"
PREDICT_PROPRIOCEPTION="${PREDICT_PROPRIOCEPTION:-False}"
TASK_MODE_LABEL="${TASK_MODE_LABEL:-policy_model}"
DINO_FEATURE_LABEL="${DINO_FEATURE_LABEL:-block 4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"

# DirectoryStore on tmpfs keeps UMI's random chunk reads off the data1 disk.
# Set STAGE_TO_SHM=0 to read the extracted zarr from DATASET_PATH directly.
STAGE_TO_SHM="${STAGE_TO_SHM:-1}"
SHM_DATASET_PATH="${SHM_DATASET_PATH:-/dev/shm/uva_umi_dataset}"

RUN_NAME="${RUN_NAME:-uva_umi_student_vae_dino_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-uva}"
LOG_TAGS="${LOG_TAGS:-[umi,cup_arrangement_0,policy_model,no_proprioception_prediction,student_tokenizer_small,early_patchify,dinov2_vits14_block4,vae_latent_distillation,frozen_teachers,official_umi_pretrained,full_mar_finetune,8gpu]}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
    exit 1
fi

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
    "${SOURCE_DATASET_PATH}/cup_arrangement_0.zarr"
    "${UMI_CHECKPOINT}"
    "${DINO_CHECKPOINT}"
    "${VAE_CHECKPOINT}"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

TRAIN_DATASET_PATH="${SOURCE_DATASET_PATH}"
if [[ "${STAGE_TO_SHM}" == "1" ]]; then
    TRAIN_DATASET_PATH="${SHM_DATASET_PATH}"
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        mkdir -p "${SHM_DATASET_PATH}"
        source_zarr="${SOURCE_DATASET_PATH}/cup_arrangement_0.zarr"
        staged_zarr="${SHM_DATASET_PATH}/cup_arrangement_0.zarr"
        stage_marker="${SHM_DATASET_PATH}/.cup_arrangement_0.stage_complete"
        (
            flock 9
            if [[ ! -f "${stage_marker}" ]]; then
                echo "Staging ${source_zarr} to ${staged_zarr} ..."
                mkdir -p "${staged_zarr}"
                cp -a "${source_zarr}/." "${staged_zarr}/"
                touch "${stage_marker}"
                echo "UMI zarr staging complete."
            else
                echo "Reusing staged UMI zarr: ${staged_zarr}"
            fi
        ) 9>"${SHM_DATASET_PATH}/.stage.lock"
    fi
fi

echo "Config: ${CONFIG_NAME}"
echo "Dataset root: ${TRAIN_DATASET_PATH}"
echo "UMI checkpoint: ${UMI_CHECKPOINT}"
echo "Student tokenizer: small, early patchify, trainable"
echo "DINOv2: frozen ViT-S/14 ${DINO_FEATURE_LABEL} teacher (coeff ${DINO_ALIGN_COEFF})"
echo "VAE: frozen mode-latent teacher (coeff ${VAE_LATENT_COEFF})"
echo "MAR: full fine-tuning"
echo "Task mode: ${TASK_MODE_LABEL}"
echo "Proprioception input/prediction: ${USE_PROPRIOCEPTION}/${PREDICT_PROPRIOCEPTION}"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
echo "DataLoader: batch ${PER_DEVICE_BATCH}, source UMI worker/persistence settings"
echo "Run directory: ${RUN_DIR}"

launch_args=(
    --num_processes="${NUM_PROCESSES}"
    --main_process_port="${MAIN_PROCESS_PORT}"
    train.py
    --config-dir=unified_video_action/config
    --config-name="${CONFIG_NAME}"
    model.policy.autoregressive_model_params.pretrained_model_path="${UMI_CHECKPOINT}"
    model.policy.dinov2_teacher_params.checkpoint_path="${DINO_CHECKPOINT}"
    model.policy.vae_model_params.autoencoder_path="${VAE_CHECKPOINT}"
    model.policy.use_proprioception="${USE_PROPRIOCEPTION}"
    model.policy.predict_proprioception="${PREDICT_PROPRIOCEPTION}"
    model.policy.align_params.coeff="${DINO_ALIGN_COEFF}"
    model.policy.vae_latent_distill_params.coeff="${VAE_LATENT_COEFF}"
    model.policy.optimizer.learning_rate="${LEARNING_RATE}"
    task.dataset.dataset_root_dir="${TRAIN_DATASET_PATH}"
    task.dataset.used_episode_indices_file=""
    task.dataset.dataloader_cfg.batch_size="${PER_DEVICE_BATCH}"
    task.dataset.dataloader_cfg.num_workers="${NUM_WORKERS}"
    task.dataset.dataloader_cfg.persistent_workers=True
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}"
    training.lr_warmup_steps="${LR_WARMUP_STEPS}"
    training.resume=False
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

CUDA_VISIBLE_DEVICES="${GPU_IDS}" exec "${ACCELERATE_BIN}" launch "${launch_args[@]}"
