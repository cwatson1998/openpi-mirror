#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_libero_spatial_four_bowls_lora_pair.sh [options]

Evaluates the target-dot and vanilla LIBERO Spatial LoRA checkpoints on
libero_spatial_four_bowls.

The target-dot checkpoint uses next-object highlighting plus the placement dot:
  --args.next-object-highlighting --args.next-object-highlight-alpha 0.4 --args.next-object-placement-dot

The vanilla checkpoint is evaluated with unmodified RGB observations.

Options are forwarded to scripts/eval_libero_spatial_lora_pair.sh.
Common options:
  --trials N               rollouts per task (default: 5)
  --port PORT              websocket port for the local policy server (default: 8032)
  --only NAME              target-dot-from-libero or vanilla-from-libero
  --wandb-project NAME     W&B project name (default: libero)
  --run-id NAME            output/log namespace for this pair run
  --help                   show this message
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

export TASK_SUITE="${TASK_SUITE:-libero_spatial_four_bowls}"
export NUM_TRIALS="${NUM_TRIALS:-5}"
export PORT="${PORT:-8032}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_libero_spatial_four_bowls_lora_pair}"

exec "${SCRIPT_DIR}/eval_libero_spatial_lora_pair.sh" "$@"
