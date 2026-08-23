#!/usr/bin/env bash
set -euo pipefail

# Evaluate the official VLA-Adapter Pro checkpoint against LIBERO-Plus.
# LIBERO-Plus stays on a separate PYTHONPATH from the original LIBERO package.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLA_ADAPTER_ROOT="${VLA_ADAPTER_ROOT:-/home/jinboning/project/VLA-Adapter}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${PLUS_ROOT:-/data1/local_userdata/jinboning/LIBERO-plus}}"
LIBERO_PLUS_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-${PLUS_CONFIG_PATH:-/data1/local_userdata/jinboning/LIBERO-plus-runtime}}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-/data1/local_userdata/jinboning/vla-adapter-assets/envs/vla-adapter-jepa-gpu/bin/python}"
IMAGEMAGICK_ROOT="${IMAGEMAGICK_ROOT:-/data1/local_userdata/jinboning/vla-adapter-assets/system-libs/imagemagick6}"
SUITE="${SUITE:-libero_10}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
SEED="${SEED:-7}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$VLA_ADAPTER_ROOT/experiments/logs/libero_plus_${SUITE}}"
SAVE_VERSION="${SAVE_VERSION:-libero-plus}"
RUN_ID_NOTE="${RUN_ID_NOTE:-libero-plus-${SUITE}}"

case "$SUITE" in
  libero_spatial) default_checkpoint_name="LIBERO-Spatial-Pro" ;;
  libero_object) default_checkpoint_name="LIBERO-Object-Pro" ;;
  libero_goal) default_checkpoint_name="LIBERO-Goal-Pro" ;;
  libero_10) default_checkpoint_name="LIBERO-Long-Pro" ;;
  *) echo "Unsupported SUITE: $SUITE" >&2; exit 2 ;;
esac
if [[ -z "$BASE_CHECKPOINT" ]]; then
  BASE_CHECKPOINT="$VLA_ADAPTER_ROOT/pretrained_models/VLA-Adapter/$default_checkpoint_name"
fi
if [[ "$BASE_CHECKPOINT" != /* ]]; then
  BASE_CHECKPOINT="$VLA_ADAPTER_ROOT/$BASE_CHECKPOINT"
fi
if [[ "$LOCAL_LOG_DIR" != /* ]]; then
  LOCAL_LOG_DIR="$VLA_ADAPTER_ROOT/$LOCAL_LOG_DIR"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$IMAGEMAGICK_ROOT/usr/lib/x86_64-linux-gnu/libMagickWand-6.Q16.so.7" ]]; then
  echo "Local ImageMagick runtime is missing: $IMAGEMAGICK_ROOT" >&2
  exit 2
fi
if [[ ! -d "$LIBERO_PLUS_ROOT/libero" ]]; then
  echo "LIBERO-Plus source tree is missing: $LIBERO_PLUS_ROOT" >&2
  exit 2
fi
if [[ ! -f "$LIBERO_PLUS_CONFIG_PATH/config.yaml" ]]; then
  echo "LIBERO-Plus config is missing: $LIBERO_PLUS_CONFIG_PATH/config.yaml" >&2
  echo "Run scripts/eval/configure_libero_plus.sh first." >&2
  exit 2
fi
if [[ "$CUDA_DEVICE" == *,* || -z "$CUDA_DEVICE" ]]; then
  echo "CUDA_DEVICE must select exactly one physical GPU; got: $CUDA_DEVICE" >&2
  exit 2
fi
if ! [[ "$NUM_TRIALS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_TRIALS_PER_TASK must be a positive integer; got: $NUM_TRIALS_PER_TASK" >&2
  exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer; got: $SEED" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1; got: $DRY_RUN" >&2
  exit 2
fi

required_checkpoint_files=(
  config.json
  model.safetensors
  processor_config.json
  preprocessor_config.json
  tokenizer.json
  dataset_statistics.json
  action_head--checkpoint.pt
  proprio_projector--checkpoint.pt
)
for filename in "${required_checkpoint_files[@]}"; do
  if [[ ! -f "$BASE_CHECKPOINT/$filename" ]]; then
    echo "Official checkpoint file is missing: $BASE_CHECKPOINT/$filename" >&2
    echo "Use BASE_CHECKPOINT=/path/to/$default_checkpoint_name for suite $SUITE." >&2
    exit 2
  fi
done

PYTHONPATH_VALUE="$LIBERO_PLUS_ROOT:$VLA_ADAPTER_ROOT"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$PYTHONPATH"
fi
MAGICK_LIBRARY_PATH="$IMAGEMAGICK_ROOT/usr/lib/x86_64-linux-gnu:$IMAGEMAGICK_ROOT/usr/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  MAGICK_LIBRARY_PATH="$MAGICK_LIBRARY_PATH:$LD_LIBRARY_PATH"
fi

preflight_status=0
preflight_args=(--suite "$SUITE")
if [[ "$DRY_RUN" == "1" ]]; then
  preflight_args+=(--allow-missing-data)
fi
env \
  PYTHONPATH="$PYTHONPATH_VALUE" \
  LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_PATH" \
  MAGICK_HOME="$IMAGEMAGICK_ROOT/usr" \
  LD_LIBRARY_PATH="$MAGICK_LIBRARY_PATH" \
  PYTHONNOUSERSITE=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/eval/libero_plus_preflight.py" \
  "${preflight_args[@]}" || preflight_status=$?
if [[ "$preflight_status" -ne 0 && "$DRY_RUN" != "1" ]]; then
  echo "LIBERO-Plus preflight failed; refusing to start evaluation." >&2
  exit "$preflight_status"
fi

# The upstream loader rewrites a few checkpoint files. Use a temporary view and
# symlink the large immutable weights so the downloaded checkpoint stays clean.
EVAL_CHECKPOINT_VIEW="$(mktemp -d "${TMPDIR:-/tmp}/vla-adapter-libero-plus.XXXXXX")"
cleanup() {
  if [[ -n "${EVAL_CHECKPOINT_VIEW:-}" && -d "$EVAL_CHECKPOINT_VIEW" ]]; then
    rm -rf -- "$EVAL_CHECKPOINT_VIEW"
  fi
}
trap cleanup EXIT
for source_path in "$BASE_CHECKPOINT"/*; do
  filename="${source_path##*/}"
  case "$filename" in
    config.json|configuration_prismatic.py|modeling_prismatic.py)
      cp -p -- "$source_path" "$EVAL_CHECKPOINT_VIEW/$filename"
      ;;
    *)
      ln -s -- "$source_path" "$EVAL_CHECKPOINT_VIEW/$filename"
      ;;
  esac
done

mkdir -p "$LOCAL_LOG_DIR"
eval_command=(
  "$PYTHON_BIN" -u experiments/robot/libero/run_libero_eval.py
  --model_family openvla
  --pretrained_checkpoint "$EVAL_CHECKPOINT_VIEW"
  --use_jepa_student_dino False
  --use_l1_regression True
  --use_minivlm True
  --use_proprio True
  --num_images_in_input 2
  --use_film False
  --use_pro_version True
  --load_in_8bit False
  --load_in_4bit False
  --center_crop True
  --phase Inference
  --task_suite_name "$SUITE"
  --initial_states_path DEFAULT
  --num_trials_per_task "$NUM_TRIALS_PER_TASK"
  --num_open_loop_steps 8
  --num_steps_wait 10
  --env_img_res 256
  --seed "$SEED"
  --use_wandb False
  --run_id_note "$RUN_ID_NOTE"
  --local_log_dir "$LOCAL_LOG_DIR"
  --save_version "$SAVE_VERSION"
)

echo "LIBERO-Plus source: $LIBERO_PLUS_ROOT"
echo "LIBERO-Plus config: $LIBERO_PLUS_CONFIG_PATH/config.yaml"
echo "Checkpoint: $BASE_CHECKPOINT"
echo "Evaluation: $SUITE, ${NUM_TRIALS_PER_TASK} trial(s)/task, seed ${SEED}"
printf 'Command:'
printf ' %q' "${eval_command[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: command was not executed."
  exit 0
fi

cd "$VLA_ADAPTER_ROOT"
env \
  PYTHONPATH="$PYTHONPATH_VALUE" \
  PYTHONNOUSERSITE=1 \
  LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_PATH" \
  MAGICK_HOME="$IMAGEMAGICK_ROOT/usr" \
  LD_LIBRARY_PATH="$MAGICK_LIBRARY_PATH" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  "${eval_command[@]}"
