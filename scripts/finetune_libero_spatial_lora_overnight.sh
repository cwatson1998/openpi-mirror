#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/libero_spatial_lora_overnight_${RUN_ID}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NORM_NUM_WORKERS="${NORM_NUM_WORKERS:-8}"
MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_DIR}"

WANDB_FLAG=(--no-wandb-enabled)
if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  WANDB_FLAG=(--wandb-enabled)
fi

COMMON_TRAIN_ARGS=(
  --batch-size "${BATCH_SIZE}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --save-interval "${SAVE_INTERVAL}"
  --max-checkpoints-to-keep "${MAX_CHECKPOINTS_TO_KEEP}"
  --keep-period None
  --num-workers "${NUM_WORKERS}"
  "${WANDB_FLAG[@]}"
)

ensure_norm_stats() {
  local config_name="$1"
  local stats_path="$2"

  if [[ -f "${stats_path}/norm_stats.json" ]]; then
    echo "[$(date --iso-8601=seconds)] Norm stats already exist: ${stats_path}"
    return
  fi

  echo "[$(date --iso-8601=seconds)] Computing norm stats for ${config_name}"
  "${PYTHON_BIN}" scripts/compute_norm_stats.py \
    --config-name "${config_name}" \
    --num-workers "${NORM_NUM_WORKERS}" \
    2>&1 | tee "${LOG_DIR}/compute_norm_stats_${config_name}.log"
}

run_train() {
  local label="$1"
  local config_name="$2"
  local exp_name="$3"

  echo "[$(date --iso-8601=seconds)] Starting ${label}"
  echo "  config: ${config_name}"
  echo "  exp:    ${exp_name}"

  "${PYTHON_BIN}" scripts/train.py "${config_name}" \
    --exp-name "${exp_name}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${exp_name}.log"

  echo "[$(date --iso-8601=seconds)] Finished ${label}"
}

cat > "${LOG_DIR}/manifest.txt" <<EOF
RUN_ID=${RUN_ID}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
BATCH_SIZE=${BATCH_SIZE}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}
SAVE_INTERVAL=${SAVE_INTERVAL}
MAX_CHECKPOINTS_TO_KEEP=${MAX_CHECKPOINTS_TO_KEEP}
KEEP_PERIOD=None
WANDB_ENABLED=${WANDB_ENABLED:-0}
EOF

ensure_norm_stats \
  "pi0_fast_libero_spatial_target_dot_low_mem_finetune" \
  "assets/pi0_fast_libero_spatial_target_dot_low_mem_finetune/libero_spatial_next_object_target_dot_alpha04"

ensure_norm_stats \
  "pi0_fast_libero_spatial_low_mem_finetune" \
  "assets/pi0_fast_libero_spatial_low_mem_finetune/libero_spatial"

run_train \
  "official LIBERO checkpoint -> annotated libero_spatial" \
  "pi0_fast_libero_spatial_target_dot_from_libero_low_mem_finetune" \
  "libero_spatial_target_dot_lora_from_libero_${RUN_ID}"

run_train \
  "official LIBERO checkpoint -> vanilla libero_spatial" \
  "pi0_fast_libero_spatial_from_libero_low_mem_finetune" \
  "libero_spatial_vanilla_lora_from_libero_${RUN_ID}"

run_train \
  "base pi0-FAST checkpoint -> annotated libero_spatial" \
  "pi0_fast_libero_spatial_target_dot_low_mem_finetune" \
  "libero_spatial_target_dot_lora_from_base_${RUN_ID}"

run_train \
  "base pi0-FAST checkpoint -> vanilla libero_spatial" \
  "pi0_fast_libero_spatial_low_mem_finetune" \
  "libero_spatial_vanilla_lora_from_base_${RUN_ID}"

echo "[$(date --iso-8601=seconds)] All overnight LoRA finetunes finished."
