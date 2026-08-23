#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The validated UMI batch for this run is 56 per device (8 GPUs -> global 448).
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-56}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-448}"
export CONFIG_NAME="${CONFIG_NAME:-uva_umi_dinov2_student_vae_latent_full_dynamic.yaml}"
export USE_PROPRIOCEPTION=True
export PREDICT_PROPRIOCEPTION=True
export TASK_MODE_LABEL=full_dynamic_model
export DINO_FEATURE_LABEL="final output"
export RUN_NAME="${RUN_NAME:-uva_umi_student_vae_dino_full_dynamic_$(date +%Y%m%d_%H%M%S)}"
export LOG_TAGS="${LOG_TAGS:-[umi,cup_arrangement_0,full_dynamic_model,video_prediction,action_prediction,proprioception_input,proprioception_prediction,student_tokenizer_small,early_patchify,dinov2_vits14_final,vae_latent_distillation,frozen_teachers,official_umi_pretrained,full_mar_finetune,8gpu]}"

exec "${SCRIPT_DIR}/train_uva_umi_dinov2_student_vae_latent_8gpu.sh"
