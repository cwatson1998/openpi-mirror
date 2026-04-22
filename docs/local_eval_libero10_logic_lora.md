# Local Eval: LIBERO-10 Logic LoRA Checkpoints

This is the local, two-terminal version of the LIBERO eval flow used by:

- [examples/libero/main.py](../examples/libero/main.py)
- [examples/libero/eval_checkpoint.slurm](../examples/libero/eval_checkpoint.slurm)

The important split is intentional:

- run the policy server from the main OpenPI environment with Python 3.11+
- run the LIBERO evaluator from a separate Python 3.10 environment

This evaluator has native W&B support. If you pass the W&B flags below, it will
stream running success metrics during the eval, not just write a final summary.

## Checkpoints

Both checkpoints were trained with config
`pi0_fast_libero_10_logic_low_mem_finetune` and require logic prompt overrides
from `data/libero/libero_10_logic_descriptions.json` during eval.

| Short name | Local checkpoint path |
|---|---|
| **lora** | `checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_20260416_013843/29999` |
| **lora-from-libero-ckpt** | `checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_from_libero_ckpt_20260416_013843/29999` |

## Prerequisites

- main OpenPI env available at `.venv` or usable via `uv run`
- LIBERO eval env available at `examples/libero/.venv`
- `third_party/libero` submodule checked out
- GPU with enough VRAM to serve the checkpoint locally

## One-Time Setup

If the LIBERO eval environment is not already ready, create it from the
maintained requirements file instead of the older hand-written install list:

```bash
uv venv --python 3.10 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip install -r examples/libero/requirements.in \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
python -m ensurepip --upgrade
python -m pip install --upgrade 'setuptools<81'
```

That final `setuptools<81` repair is currently required because W&B still
imports `pkg_resources`, which newer setuptools releases no longer provide.
If you want to sanity-check the env before a long run:

```bash
PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/libero" \
  examples/libero/.venv/bin/python - <<'PY'
import importlib

for module in ["libero.libero", "libero.libero.envs", "openpi_client", "wandb", "pkg_resources"]:
    importlib.import_module(module)
print("LIBERO eval env looks healthy")
PY
```

Write a repo-local LIBERO config so the evaluator does not accidentally use
`~/.libero/config.yaml` from a different checkout:

```bash
REPO_ROOT="$PWD"
mkdir -p .cache/libero-openpi

cat > .cache/libero-openpi/config.yaml <<EOF
benchmark_root: ${REPO_ROOT}/third_party/libero/libero/libero
bddl_files: ${REPO_ROOT}/third_party/libero/libero/libero/bddl_files
init_states: ${REPO_ROOT}/third_party/libero/libero/libero/init_files
datasets: ${REPO_ROOT}/third_party/libero/libero/datasets
assets: ${REPO_ROOT}/third_party/libero/libero/libero/assets
EOF
```

## 1. Verify Checkpoints

```bash
ls checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_20260416_013843/29999/
ls checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_from_libero_ckpt_20260416_013843/29999/
```

## 2. Start The Policy Server

Run this in Terminal 1 from the repo root. Keep this terminal open.

For the `lora` checkpoint:

```bash
export USE_TF=0

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_fast_libero_10_logic_low_mem_finetune \
  --policy.dir checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_20260416_013843/29999
```

For `lora-from-libero-ckpt`, change only `--policy.dir`:

```bash
export USE_TF=0

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_fast_libero_10_logic_low_mem_finetune \
  --policy.dir checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_from_libero_ckpt_20260416_013843/29999
```

Wait until the server is accepting connections before starting the evaluator.

## 3. Run The LIBERO Eval

Run this in Terminal 2 from the repo root. This must use the LIBERO Python 3.10
environment, not the root `.venv`.

Example for the `lora` checkpoint:

```bash
source examples/libero/.venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/libero"
export LIBERO_CONFIG_PATH="$PWD/.cache/libero-openpi"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false

CKPT="checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_20260416_013843/29999"
POLICY_CONFIG="pi0_fast_libero_10_logic_low_mem_finetune"
TRAIN_RUN_NAME="$(basename "$(dirname "$CKPT")")"
CKPT_STEP="$(basename "$CKPT")"
EVAL_NAME="${TRAIN_RUN_NAME}_${CKPT_STEP}_libero_10_local"
VIDEO_OUT="data/libero/evals/${EVAL_NAME}"
WANDB_TAGS="eval,libero,libero_10,logic_prompts,policy:${POLICY_CONFIG},train_run:${TRAIN_RUN_NAME},checkpoint:${CKPT_STEP}"

python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 10 \
  --args.video-out-path "$VIDEO_OUT" \
  --args.results-out-path "$VIDEO_OUT/results.json" \
  --args.progress-out-path "$VIDEO_OUT/eval_progress.json" \
  --args.prompt-override-file data/libero/libero_10_logic_descriptions.json \
  --args.wandb-enabled \
  --args.wandb-project libero \
  --args.wandb-name "$EVAL_NAME" \
  --args.wandb-group "$TRAIN_RUN_NAME" \
  --args.wandb-tags-csv "$WANDB_TAGS" \
  --args.policy-config "$POLICY_CONFIG" \
  --args.checkpoint-dir "$CKPT" \
  --args.train-run-name "$TRAIN_RUN_NAME"
```

For `lora-from-libero-ckpt`, change only `CKPT`:

```bash
CKPT="checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_from_libero_ckpt_20260416_013843/29999"
```

## Why These Flags Matter

- `--args.host 127.0.0.1` is the correct local websocket client target
- `--args.prompt-override-file ...libero_10_logic_descriptions.json` is required because these checkpoints were trained with logic prompts
- the W&B flags enable streaming eval metrics from the evaluator itself
- `--args.policy-config`, `--args.checkpoint-dir`, and `--args.train-run-name` ensure W&B metadata matches the source checkpoint
- `--args.results-out-path` and `--args.progress-out-path` make the output paths explicit and mirror the Slurm wrapper behavior

## Smoke Test

Before a full run, do a short sanity check:

```bash
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_10 \
  --args.task-indices 0 \
  --args.num-trials-per-task 2 \
  --args.prompt-override-file data/libero/libero_10_logic_descriptions.json
```

## Outputs

With the command above, eval outputs land under:

```text
data/libero/evals/<eval-name>/
```

Important files:

- `results.json`: final aggregated results
- `eval_progress.json`: incremental progress during the run
- rollout videos: one video per episode

When W&B is enabled, the evaluator also logs:

- running total success rate as `eval/total_success_rate_running`
- running per-task success rate as `eval/current_task_success_rate_running`
- final summaries and result artifacts at the end

## Evaluate The Second Checkpoint

Stop the policy server in Terminal 1, restart it with the other checkpoint path,
then rerun the Terminal 2 command with the matching `CKPT` value. Keep
`POLICY_CONFIG` the same and use a distinct `EVAL_NAME` so the outputs and W&B
runs stay separate.

## One-Command Persistent Run

For these two specific checkpoints, the repo now includes a sequential runner
that handles the full lifecycle locally:

- starts the first policy server
- waits for the websocket port
- runs `libero_10` with `10` rollouts per task
- tears the server down
- repeats for the second checkpoint

The runner now uses a hardened server cleanup path. If the policy server does
not exit promptly after `SIGINT`, it escalates to `SIGTERM` and then `SIGKILL`
instead of hanging forever between checkpoints.

Launch it in a detached tmux session:

```bash
tmux new-session -d -s libero10-logic-lora-pair \
  "cd '$PWD' && ./scripts/eval_libero10_logic_lora_pair.sh"
tmux set-option -t libero10-logic-lora-pair remain-on-exit on
```

Useful follow-up commands:

```bash
tmux attach -t libero10-logic-lora-pair
tail -f logs/libero_eval/<run-id>/runner.log
```

The sequential runner writes:

- top-level runner log: `logs/libero_eval/<run-id>/runner.log`
- per-checkpoint server logs: `logs/libero_eval/<run-id>/*_server.log`
- per-checkpoint eval logs: `logs/libero_eval/<run-id>/*_eval.log`
- eval outputs: `data/libero/evals/<run-id>/<checkpoint-short-name>/`

## Restart Only One Checkpoint

If the pair run is interrupted after one checkpoint completes, you can rerun
just the remaining checkpoint with `--only`.

Rerun only the second checkpoint:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_libero10_logic_lora_secondonly"

tmux new-session -d -s libero10-logic-lora-second \
  "cd '$PWD' && ./scripts/eval_libero10_logic_lora_pair.sh \
    --run-id '${RUN_ID}' \
    --only lora-from-libero-ckpt"
tmux set-option -t libero10-logic-lora-second remain-on-exit on
```

You can also rerun only the first checkpoint:

```bash
./scripts/eval_libero10_logic_lora_pair.sh --only lora --run-id <run-id>
```

Valid values are:

- `lora`
- `lora-from-libero-ckpt`
