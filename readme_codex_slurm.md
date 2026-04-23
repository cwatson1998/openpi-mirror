# Codex and OpenPI on This Slurm Cluster

This note is a repo-local reference for future Codex sessions on the GRASP Slurm cluster.

It is written for the current setup:

- login node observed: `grasp-login1`
- home directory: `/home/chriswatson`
- user: `ccwatson`
- partitions you said you use: `batch` and `dineshj-compute`
- preferred QoS on `dineshj-compute`: `dj-med`

The main goals are:

- keep Codex responsive on a shared cluster
- avoid stressing the login node
- keep temp files, caches, and experiment outputs off the slowest storage when possible
- make interactive and batch experiment runs predictable

## What We Observed

From the cluster itself:

- `/home` is a shared filesystem and was at `98%` utilization when this note was written
- `/tmp` on the login node is local storage with about `48G` total and was mostly free
- `/scratch` exists, but `/scratch/ccwatson` did not exist yet by default
- `batch` is the default partition and allows QoS `normal`
- `dineshj-compute` allows QoS `dj-med` and `dj-high`
- `dineshj-compute` has `MaxTime=14-00:00:00`
- `dj-med` currently shows `MaxWall=12:00:00`

Representative GPU resources visible at the time of writing:

- `batch`: mixed nodes including `2080 Ti`, `3090`, `A10`, `A40`, `L40`, `L40S`, `A6000`
- `dineshj-compute`: `2080 Ti`, `A10`, `A40`, and `L40`

This means:

- use the login node for light work only
- prefer compute nodes for long installs, tests, training, eval, and anything GPU-related
- treat `/home` as usable but not fast, and do not rely on it for metadata-heavy temp/cache traffic

## Recommended Working Model

Use this split:

- login node: editing, `git`, small searches, short lint/test commands, Codex planning
- interactive Slurm job: heavier Codex sessions, installs, test suites, local servers, evals
- batch jobs: training, long evals, reproducible experiment runs

If Codex is only helping with code reading and small edits, the login node is fine. If Codex is going to run many commands, imports, tests, or long scripts, start it inside an interactive Slurm allocation instead.

## Storage Strategy

### 1. Keep temp files and caches off `/home`

The warning you saw at startup:

`WARNING: failed to clean up stale arg0 temp dirs: Directory not empty (os error 39)`

is usually harmless, but on shared filesystems it often points to temp-dir cleanup races or stale metadata. The safest fix is to give tools a better temp/cache location.

Recommended shell setup:

```bash
mkdir -p /scratch/$USER/tmp
mkdir -p /scratch/$USER/.cache/uv
mkdir -p /scratch/$USER/.cache/pip
mkdir -p /scratch/$USER/.cache/huggingface
mkdir -p /scratch/$USER/.cache/codex

export TMPDIR=/scratch/$USER/tmp
export XDG_CACHE_HOME=/scratch/$USER/.cache
export UV_CACHE_DIR=/scratch/$USER/.cache/uv
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export HF_HOME=/scratch/$USER/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/.cache/huggingface/hub
```

If `/scratch/$USER` does not exist yet, create it first:

```bash
mkdir -p /scratch/$USER
chmod 700 /scratch/$USER
```

Notes:

- use `/scratch/$USER` for temp files and caches
- for large Hugging Face model or dataset downloads, prefer `/mnt/kostas-graid/datasets/ccwatson/huggingface` instead of `/home`; set `HF_HOME` and `HUGGINGFACE_HUB_CACHE` there if you want those downloads to persist in the faster shared dataset area
- do not assume scratch is backed up
- do not keep irreplaceable results only in scratch
- for short-lived local temp on an interactive job, node-local `$TMPDIR` may be even better if Slurm provides it
- for training jobs, node-local `/tmp` can be much faster than shared storage for frequently written files such as checkpoints-in-progress, sharded intermediates, or rollout buffers
- anything written only to node-local `/tmp` will be lost when the job ends or is preempted, so copy or sync important outputs back to `/scratch/$USER` or `/home/chriswatson` during the run, not just at the end
- node-local `/tmp` capacity varies by node and can be small, so check free space before relying on it for large checkpoints or video-heavy outputs

### 2. Be careful with `/tmp`

On this cluster, `/tmp` on the login node appears fast and mostly free. That makes it useful for small temporary files.

But:

- `/tmp` is ephemeral
- it is shared with other users on that node
- it is not a good place for long experiments or large checkpoints

Use `/tmp` for short-lived local temp. Use `/scratch/$USER` for caches and medium-lived working data. Use `/home` for source code and important tracked files unless you choose to keep a scratch worktree.

### 3. Keep large experiment artifacts out of the repo

For this repo specifically, avoid letting these directories grow uncontrolled inside the working tree:

- `data/`
- `logs/`
- `checkpoints/`
- `.venv/`
- large rollout videos

Good pattern:

- repo and git history in `/home/chriswatson/openpi`
- active caches in `/scratch/$USER/.cache`
- heavy outputs in `/scratch/$USER/openpi-runs/...`
- copy or sync back only the results you want to keep

If `/home` becomes too painful, keep a second clone or `git worktree` on `/scratch/$USER/openpi` for active experiment sessions.

## Shell Setup for Future Codex Sessions

Add something like this to `~/.bashrc` or to a dedicated cluster wrapper script:

```bash
mkdir -p /scratch/$USER/tmp
mkdir -p /scratch/$USER/.cache/uv
mkdir -p /scratch/$USER/.cache/pip
mkdir -p /scratch/$USER/.cache/huggingface
mkdir -p /scratch/$USER/.cache/codex

export TMPDIR=/scratch/$USER/tmp
export XDG_CACHE_HOME=/scratch/$USER/.cache
export UV_CACHE_DIR=/scratch/$USER/.cache/uv
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export HF_HOME=/scratch/$USER/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/.cache/huggingface/hub
```

Optional OpenPI-specific extras:

```bash
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
```

`PYTHONUNBUFFERED=1` makes logs easier to watch in Slurm output.

## How to Run Codex Well Here

### Good uses of Codex on the login node

- inspecting code
- editing files
- small `rg` searches
- reading docs
- short `uv run pytest path/to/test.py`
- short `ruff` runs

### Avoid on the login node

- full test suites
- large dependency installs
- anything GPU-backed
- long-running Python jobs
- training
- long LIBERO evals
- lots of parallel subprocesses

### Best practice for heavier Codex sessions

Start an interactive job and run Codex from there.

For your preferred partition:

```bash
srun --pty \
  --partition=dineshj-compute \
  --qos=dj-med \
  --time=04:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --gres=gpu:1 \
  bash
```

Then inside that shell:

```bash
cd /home/chriswatson/openpi
export TMPDIR=${TMPDIR:-/scratch/$USER/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/$USER/.cache}
codex
```

Why this works better:

- Codex can run heavier commands without abusing the login node
- package caches and temp files can stay on scratch
- GPU and larger memory requests are explicit
- experiments and debugging happen in the environment where they will actually run

If you only need CPU and a shell for heavier repo work:

```bash
srun --pty \
  --partition=dineshj-compute \
  --qos=dj-med \
  --time=02:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  bash
```

If `dineshj-compute` is busy and the work is generic, you can fall back to `batch`.

## Slurm Patterns We Should Reuse

### Interactive debug job

```bash
srun --pty \
  --partition=dineshj-compute \
  --qos=dj-med \
  --time=02:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --gres=gpu:1 \
  bash
```

### Interactive CPU-only job

```bash
srun --pty \
  --partition=batch \
  --time=01:00:00 \
  --cpus-per-task=8 \
  --mem=16G \
  bash
```

### Batch training skeleton

```bash
sbatch \
  --partition=dineshj-compute \
  --qos=dj-med \
  --time=12:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  --output=logs/slurm/%x-%j.out \
  --wrap='cd /home/chriswatson/openpi && export TMPDIR=/scratch/$USER/tmp && export XDG_CACHE_HOME=/scratch/$USER/.cache && uv run scripts/train.py ...'
```

`dj-med` currently reports a 12-hour walltime cap, so treat `12:00:00` as the safe default for that QoS unless the cluster policy changes.

Note that you really need to add requeue functionality!! Most jobs take longer than this.

### Queue inspection

```bash
squeue -u "$USER" -o "%8i %9P %20j %8u %2t %10M %6D %20R"
```

Useful state reminders:

- `PD`: pending
- `R`: running
- `CG`: completing

### Partition inspection

```bash
sinfo -o "%P %a %l %D %c %m %G %f"
```

This is useful before requesting a specific GPU type.

## OpenPI-Specific Advice

### Use the root `uv` environment for normal OpenPI work

For the main repo:

```bash
uv sync --dev
```

Then run:

```bash
uv run pytest
uv run ruff check .
uv run scripts/train.py ...
```

With the cache variables above, these commands should avoid pushing as much transient traffic into `/home`.

### For LIBERO, keep the split-environment model

This repo already has LIBERO-specific documentation and a separate environment pattern. Do not collapse everything into one Python env just because you are on Slurm.

Keep in mind:

- root repo env for OpenPI server-side commands
- `examples/libero/.venv` for the LIBERO evaluator workflow when required
- repo-local `LIBERO_CONFIG_PATH` instead of relying on `~/.libero`

Related docs in this repo:

- `docs/libero_eval_tmux.md`
- `readme_lora_eval.md`

### Keep checkpoints and videos off slow shared metadata paths when possible

For long runs:

- write logs to `logs/slurm/`
- consider writing checkpoints or video-heavy outputs to `/scratch/$USER/openpi-runs/...`
- sync back final artifacts you care about

This is especially relevant for:

- rollout videos under `data/libero/videos`
- frequent checkpoint writes
- experiments that create many small files

## Practical Conventions for Future Sessions

When starting a serious coding or experiment session on this cluster:

1. decide whether the work belongs on the login node or a compute node
2. ensure `/scratch/$USER` exists
3. export temp/cache env vars before running heavy tools
4. run Codex inside `srun --pty` if you expect many commands or tests
5. keep large outputs off `/home` when possible

Good default assumptions for this cluster:

- `dineshj-compute` + `--qos=dj-med` for your interactive GPU work
- `batch` for generic short jobs
- `/scratch/$USER` for caches and temp
- `/home/chriswatson/openpi` for the canonical repo

## Suggested Small Improvements

These would improve future sessions further:

- create `/scratch/$USER` once and standardize on it
- add a small shell wrapper such as `~/bin/codex-slurm`
- add `logs/slurm/` to keep Slurm outputs organized
- optionally keep a scratch clone or worktree for heavy experiment days

Example wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /scratch/$USER/tmp
mkdir -p /scratch/$USER/.cache/uv
mkdir -p /scratch/$USER/.cache/pip
mkdir -p /scratch/$USER/.cache/huggingface
mkdir -p /scratch/$USER/.cache/codex

export TMPDIR=/scratch/$USER/tmp
export XDG_CACHE_HOME=/scratch/$USER/.cache
export UV_CACHE_DIR=/scratch/$USER/.cache/uv
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export HF_HOME=/scratch/$USER/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/.cache/huggingface/hub

cd /home/chriswatson/openpi
exec codex "$@"
```

For heavier sessions:

If you expect large Hugging Face downloads, move those off `/home` and point them at `/mnt/kostas-graid/datasets/ccwatson/huggingface` instead of the scratch example above:

```bash
mkdir -p /mnt/kostas-graid/datasets/ccwatson/huggingface
export HF_HOME=/mnt/kostas-graid/datasets/ccwatson/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/kostas-graid/datasets/ccwatson/huggingface/hub
```

```bash
srun --pty \
  --partition=dineshj-compute \
  --qos=dj-med \
  --time=04:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --gres=gpu:1 \
  ~/bin/codex-slurm
```

## Bottom Line

The main cluster risks for Codex here are not Slurm itself. They are:

- doing too much work on the login node
- pushing temp/cache traffic into a very full shared `/home`
- writing lots of small files to slow shared storage

The stable recipe is:

- keep source in the repo
- keep caches and temp in `/scratch/$USER`
- move heavy Codex sessions into `srun --pty`
- run training and long evals through Slurm, not on the login node
