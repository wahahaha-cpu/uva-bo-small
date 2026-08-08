#!/usr/bin/env bash
set -euo pipefail

# Libero rollout needs MuJoCo/NVIDIA runtime libraries, and Accelerate imports
# DeepSpeed while unwrapping the EMA model, which requires a CUDA toolkit root.
export CUDA_HOME="${CUDA_HOME:-/data1/local_userdata/jinboning/conda/envs/repa}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/lib/nvidia:/home/jinboning/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

# Reference: 2 GPUs x batch 16 x accumulation 4 = global batch 128.
# This launch: 4 GPUs x batch 8 x accumulation 4 = global batch 128.
# Accelerate advances the scheduler once per process, so warmup is scaled from
# 1000 on 2 GPUs to 2000 on 4 GPUs to preserve the reference optimizer-step
# LR trajectory.
# A unique output directory prevents reuse of old checkpoints or normalizers.
 export MUJOCO_EGL_DEVICE_ID=0
NUM_PROCESSES=4
GPU_IDS="0,1,2,3"
PER_DEVICE_BATCH=8
GRAD_ACCUM_STEPS=4
EXPECTED_GLOBAL_BATCH=128
GLOBAL_BATCH=$((NUM_PROCESSES * PER_DEVICE_BATCH * GRAD_ACCUM_STEPS))
if (( GLOBAL_BATCH != EXPECTED_GLOBAL_BATCH )); then
    echo "Global batch mismatch: ${GLOBAL_BATCH}" >&2
    exit 2
fi
echo "Global batch: ${NUM_PROCESSES} x ${PER_DEVICE_BATCH} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH}"
RUN_NAME="${RUN_NAME:-uva_libero10_small_aligned_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/${RUN_NAME}}"

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
    if command -v accelerate >/dev/null 2>&1; then
        ACCELERATE_BIN="accelerate"
    else
        ACCELERATE_BIN="/data1/local_userdata/jinboning/conda/envs/repa/bin/accelerate"
    fi
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${ACCELERATE_BIN}" launch \
    --num_processes="${NUM_PROCESSES}" \
    train.py \
    --config-dir=unified_video_action/config \
    --config-name=uva_libero10.yaml \
    model.policy.autoregressive_model_params.pretrained_model_path=checkpoints/libero10_video.ckpt \
    model.policy.action_model_params.predict_action=True \
    model/student_tokenizer=small \
    model.policy.optimizer.learning_rate=1e-4 \
    model.policy.selected_training_mode=policy_model \
    dataloader.batch_size="${PER_DEVICE_BATCH}" \
    val_dataloader.batch_size="${PER_DEVICE_BATCH}" \
    training.gradient_accumulate_every="${GRAD_ACCUM_STEPS}" \
    training.lr_warmup_steps=2000 \
    training.resume=False \
    training.rollout_every=20 \
    training.checkpoint_every=20 \
    task.env_runner.n_envs=1 \
    logging.project=uva-repa-new \
    hydra.run.dir="${RUN_DIR}"
