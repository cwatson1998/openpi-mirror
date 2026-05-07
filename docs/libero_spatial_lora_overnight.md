# LIBERO Spatial LoRA Overnight Runs

On May 7, 2026, the two runs initialized from the official pi0-FAST LIBERO checkpoint completed successfully:

- `pi0_fast_libero_spatial_target_dot_from_libero_low_mem_finetune`
  - exp: `libero_spatial_target_dot_lora_from_libero_overnight_20260506_221009`
  - final checkpoint: `checkpoints/pi0_fast_libero_spatial_target_dot_from_libero_low_mem_finetune/libero_spatial_target_dot_lora_from_libero_overnight_20260506_221009/29999`
- `pi0_fast_libero_spatial_from_libero_low_mem_finetune`
  - exp: `libero_spatial_vanilla_lora_from_libero_overnight_20260506_221009`
  - final checkpoint: `checkpoints/pi0_fast_libero_spatial_from_libero_low_mem_finetune/libero_spatial_vanilla_lora_from_libero_overnight_20260506_221009/29999`

The base pi0-FAST initialized runs were intentionally cancelled. The original run failed on the first base-initialized job with a CUDA graph OOM:

```text
RESOURCE_EXHAUSTED: Underlying backend ran out of memory trying to instantiate graph
```

A one-step smoke test confirmed that disabling XLA GPU command buffers avoids this OOM:

```bash
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
```

To restart only the two base-initialized 30k-step runs later, use:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
RUN_ID="base_remainder_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/libero_spatial_lora_${RUN_ID}"
SESSION="libero_spatial_lora_${RUN_ID}"
mkdir -p "${LOG_DIR}"

tmux new-session -d -s "${SESSION}" "
  cd '${REPO_ROOT}' &&
  export PYTHONPATH=src &&
  export CUDA_VISIBLE_DEVICES=0 &&
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 &&
  export XLA_FLAGS=--xla_gpu_enable_command_buffer= &&
  {
    .venv/bin/python scripts/train.py pi0_fast_libero_spatial_target_dot_low_mem_finetune \
      --exp-name libero_spatial_target_dot_lora_from_base_${RUN_ID} \
      --batch-size 1 \
      --num-train-steps 30000 \
      --save-interval 1000 \
      --max-checkpoints-to-keep 2 \
      --keep-period None \
      --num-workers 2 \
      --no-wandb-enabled \
      --overwrite 2>&1 | tee '${LOG_DIR}/base_annotated.log'

    .venv/bin/python scripts/train.py pi0_fast_libero_spatial_low_mem_finetune \
      --exp-name libero_spatial_vanilla_lora_from_base_${RUN_ID} \
      --batch-size 1 \
      --num-train-steps 30000 \
      --save-interval 1000 \
      --max-checkpoints-to-keep 2 \
      --keep-period None \
      --num-workers 2 \
      --no-wandb-enabled \
      --overwrite 2>&1 | tee '${LOG_DIR}/base_vanilla.log'
  } > '${LOG_DIR}/base_remainder_launcher.log' 2>&1
"
```

Monitor with:

```bash
tmux attach -t "${SESSION}"
tail -f "${LOG_DIR}/base_remainder_launcher.log"
```
