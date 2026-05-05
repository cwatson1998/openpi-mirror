# Repository Guidelines

## What This Repo Is
`openpi` is a Python training and serving repo for OpenPI policies. Most active work happens in:

- `src/openpi/`: training, policies, models, transforms, serving
- `src/annotation/`: LIBERO dataset visualization, replay, and annotation utilities
- `scripts/`: entrypoints like training and policy serving
- `examples/libero/`: LIBERO-specific training and eval tooling
- `packages/openpi-client/`: local client package used by eval and serving flows

Treat `third_party/` as vendored code. Avoid editing it unless the task explicitly requires it.

`third_party/libero` is a git submodule, not just a plain folder. If you need to modify LIBERO itself:

- create or switch to a real branch inside `third_party/libero` before making changes; do not develop on a detached HEAD
- commit the LIBERO changes inside the submodule first
- then return to the main repo and commit the updated `third_party/libero` submodule pointer separately
- if you also change `src/annotation/` or other main-repo code, prefer separate commits for the submodule bump and the main-repo integration

## Important Paths
- `src/openpi/training/config.py`: train config definitions and named configs
- `src/openpi/training/data_loader.py`: dataset wiring, including LIBERO prompt selection
- `src/openpi/serving/websocket_policy_server.py`: websocket serving path
- `src/annotation/libero_coordinate_frame_visualization.py`: simulator-native MuJoCo marker visualization of world and robot-base frames
- `src/annotation/testing/libero_predicates_test.py`: lightweight regression checks for LIBERO predicate helpers added in the submodule
- `src/annotation/libero_demo_replay.py`: simulator-backed LIBERO replay and path-resolution utilities
- `examples/libero/convert_libero_data_to_lerobot.py`: RLDS -> LeRobot conversion, including optional next-object-highlighting rerenders
- `src/annotation/libero_episode_sanity_check.py`: RLDS-vs-simulator-vs-masked-simulator comparison export
- `src/annotation/libero_bddl_object_stack_inspector.py`: BDDL object-category -> implementation-class inspector
- `src/annotation/README.md`: overview of the annotation and replay toolchain
- `src/annotation/libero_masked_replay_visualization.py`: side-by-side original vs masked simulator replay visualization
- `src/annotation/libero_resolution_inspector.py`: RLDS episode -> source HDF5 / BDDL / model XML inspection helper
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`: websocket client used by eval
- `scripts/train.py`: main training entrypoint
- `scripts/serve_policy.py`: local/server policy launcher
- `examples/libero/main.py`: LIBERO eval driver
- `scripts/eval_libero10_logic_lora_pair.sh`: sequential local runner for the two LIBERO-10 logic LoRA checkpoints
- `docs/local_eval_libero10_logic_lora.md`: local two-checkpoint LIBERO-10 logic LoRA workflow
- `examples/libero/train_libero_10_logic_full.slurm`: current Slurm training launcher for logic-prompt LIBERO runs
- `examples/libero/eval_checkpoint.slurm`: current Slurm eval job
- `scripts/submit_libero_eval_slurm.sh`: helper for submitting eval jobs
- `docs/libero_eval_slurm.md`: human-facing instructions for the eval workflow
- `MASKING_README.md`: end-to-end notes for LIBERO RGB instance masking in eval and replay
- `EXPERIMENTS_APRIL_14.md`: concrete notes from the April 14 LIBERO training and eval session

## Environment And Tooling
Use Python 3.11+ for the main repo and prefer `uv`.

Common commands:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format .
uv run scripts/train.py ...
uv run scripts/serve_policy.py --port=8000 policy:checkpoint ...
```

The root project is the primary dev environment. LIBERO eval is different:

- the main repo uses Python 3.11
- LIBERO eval currently uses a separate Python 3.10 environment
- the Slurm eval flow can build a job-local env under `/tmp/...`
- do not assume the root `.venv` is enough for LIBERO simulator work

Use these Python environments for these areas:

- `src/openpi/`, `scripts/train.py`, `scripts/serve_policy.py`, and most repo development:
  use the root repo env, normally `.venv` via `uv run ...`
- `examples/libero/main.py` and real LIBERO eval / simulator execution:
  use `examples/libero/.venv/bin/python` when that repo-local env exists
- `third_party/libero` when you want imports, breakpoints, or runtime behavior to match real LIBERO execution in this repo:
  use `examples/libero/.venv/bin/python`
- `src/annotation/` lightweight checks that do not need LIBERO / MuJoCo:
  use the root repo env, e.g. `uv run` or `.venv/bin/python`
- `src/annotation/` simulator-backed tools that import `libero`, `robosuite`, MuJoCo, or TFDS:
  use the Python 3.10 LIBERO env, preferably `examples/libero/.venv/bin/python`
- ad hoc fallback for simulator-backed annotation work:
  `$HOME/miniconda3/envs/instructvla_libero/bin/python` is acceptable if the repo-local `examples/libero/.venv` is missing or broken, but prefer the repo-local env for reproducibility

If you are choosing a VS Code interpreter while editing `third_party/libero` or debugging LIBERO simulator behavior, pick:

```bash
examples/libero/.venv/bin/python
```

The `src/annotation/` tools follow the same split:

- use `uv` and the root env for lightweight pure-Python checks such as `ruff` and tests that do not need LIBERO / MuJoCo
- use the working Python 3.10 LIBERO simulator environment for commands that import `libero`, `robosuite`, MuJoCo, or TFDS
- for those simulator commands, `PYTHONPATH=src` or `PYTHONPATH=src:third_party/libero` is expected; this makes the local `src/` code and vendored `third_party/libero` package importable without installing them into that env
- do not assume `uv run ...` is the correct launcher for simulator-backed annotation tools unless that Python 3.10 stack has been explicitly mirrored into `uv`

Typical pattern for these annotation utilities:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_resolution_inspector --episode-index 0
```

For masking / segmentation work:

- `third_party/libero/libero/libero/envs/env_wrapper.py` now contains the masking-capable wrappers:
  `MaskedSegmentationRenderEnv` and `DemoMaskedSegmentationRenderEnv`
- `examples/libero/main.py` can use masked observations for policy eval when `--mask-instances-csv` is provided
- `src/annotation/libero_demo_replay.py` can render masked simulator replays with `--masked-instance`
- `src/annotation/libero_masked_replay_visualization.py` is the quickest way to inspect original vs masked replay side by side

For simulator-backed dataset generation:

- `examples/libero/convert_libero_data_to_lerobot.py --next-object-highlighting` re-renders LIBERO RGB observations with the nearest future grasp target highlighted
- that path depends on `src/annotation/libero_demo_replay.py` to resolve RLDS episodes back to source HDF5 demos and restore saved MuJoCo `states`
- it requires the original LIBERO source HDF5 demos under `third_party/libero/libero/datasets` or another explicit `--demo-search-roots` path; raw RLDS shards alone are not enough
- current local suites with both layers available include:
  - `libero_spatial`: RLDS under `data/libero/raw/libero_spatial_no_noops/1.0.0`, demos under `third_party/libero/libero/datasets/libero_spatial`
  - `libero_10`: RLDS under `data/libero/raw/libero_10_no_noops/1.0.0`, demos under `third_party/libero/libero/datasets/libero_10`
- the RLDS-to-HDF5 matcher in `src/annotation/libero_demo_replay.py` is intentionally tolerant to:
  - small RLDS-vs-HDF5 length drift from no-op filtering
  - gripper-action sign mismatches
  - moderate proprio / state drift
- tolerant matching helps identify the right source demo, but highlighted LeRobot conversion still requires the matched source demo to have the same saved-state length as the RLDS episode before it will rerender frames
- if you tighten that matching logic, rerun `PYTHONPATH=src .venv/bin/python -m pytest src/annotation/testing/libero_demo_replay_test.py`

For online next-object highlighting during LIBERO eval:

- `examples/libero/main.py` now supports `--next-object-highlighting`
- the live grasp-order discovery comes from `annotation.build_grasp_order_from_source_demo(...)`, using one source HDF5 demo per benchmark task
- the online rollout state machine is `annotation.OnlineNextObjectHighlightTracker`
- this path requires `MaskedSegmentationRenderEnv`, because the live policy observations themselves must be re-rendered with the active highlight

For BDDL predicate / spatial-relation work:

- runtime predicate implementations live in `third_party/libero/libero/libero/envs/predicates/base_predicates.py`
- BDDL-facing predicate names are registered in `third_party/libero/libero/libero/envs/predicates/__init__.py`
- object and site semantics used by those predicates live in `third_party/libero/libero/libero/envs/object_states/base_object_states.py`
- the current spatial predicate names exposed to BDDL are hyphenated planning-style names such as `left-of`, `right-of`, and `in-front-of`, plus `near`
- `src/annotation/libero_coordinate_frame_visualization.py` is the quickest way to inspect world-vs-robot frame orientation before adding more robot-relative predicates

Two local-eval-specific gotchas matter in this repo right now:

- `examples/libero/main.py` must remain compatible with the Python 3.10 LIBERO eval env; do not use Python-3.11-only stdlib features there
- W&B still imports `pkg_resources`, so the LIBERO eval env currently needs `setuptools<81`; if you repair or recreate `examples/libero/.venv`, verify that `python -c "import pkg_resources"` works inside it

Follow `docs/libero_eval_slurm.md` and the scripts in `examples/libero/` for LIBERO eval instead of inventing a new bootstrap path.

## Repo-Specific Conventions
Use 4-space indentation, type hints, and Python 3.11 syntax. Ruff enforces formatting and import sorting. Keep new code in `src/openpi/...` unless it is clearly a script, example, or package-level change.

Naming:

- functions, modules, variables: `snake_case`
- classes: `PascalCase`
- configs: descriptive names like `pi0_fast_libero_10_logic`

When editing markdown or docs, keep links repo-portable. Do not commit local absolute `/home/...` paths.

## Training And Eval Notes
Current LIBERO work in this repo uses two prompt styles:

- default English dataset task descriptions
- logic prompt overrides from `data/libero/libero_10_logic_descriptions.json`

Be precise about where prompt behavior comes from:

- LeRobot datasets may now carry alternate prompt fields directly in `meta/tasks.jsonl`, for example `task_description` or `task_description_1`, alongside the canonical `task` field
- training can read those dataset-native prompt variants via `data.task_description_field`; when unset, training continues to use the canonical LeRobot `task` field
- training-time LIBERO prompt semantics come from the selected config, especially `task_description_path`, not from the experiment name or checkpoint folder name
- if both `data.task_description_field` and `task_description_path` are set, `task_description_path` wins; this preserves the existing logic-prompt override behavior
- eval-time prompt semantics currently include one naming heuristic: `scripts/submit_libero_eval_slurm.sh` and `examples/libero/eval_checkpoint.slurm` auto-set `PROMPT_OVERRIDE_FILE=data/libero/libero_10_logic_descriptions.json` when `POLICY_CONFIG` contains `_logic`
- do not extend that heuristic to `EXP_NAME`, checkpoint leaf names, or other filenames; if behavior matters, pass `--prompt-override-file` explicitly or wire it through config/CLI fields with explicit semantics

If you touch LIBERO prompt plumbing, make sure task filtering still works correctly with the logic description file. The relevant regression coverage lives in `src/openpi/training/data_loader_test.py`.

When creating new LeRobot datasets for OpenPI training:

- keep the canonical `task` field in `meta/tasks.jsonl` for backwards compatibility, filtering, and external tooling
- store any alternate prompt variants as additional string fields on the same task records, not by replacing `task`
- prefer dataset-native prompt fields for new datasets when you want the prompt variant to travel with the dataset itself
- prefer `task_description_path` only for external override layers such as the current LIBERO logic-prompt workflow

W&B support is wired through both training and eval:

- training supports config-level `wandb_tags` and `wandb_group`
- training Slurm wrappers may pass `OPENPI_WANDB_TAGS` and `OPENPI_WANDB_GROUP`
- eval uses explicit CLI flags in `examples/libero/main.py` and the Slurm wrappers

Do not silently remove or bypass this metadata plumbing when changing launch scripts.

The local two-checkpoint LIBERO-10 logic LoRA workflow now has a dedicated sequential runner:

- `scripts/eval_libero10_logic_lora_pair.sh`

It starts checkpoint A, waits for the websocket server, runs the eval, tears the server down, then repeats for checkpoint B in the same persistent tmux-friendly workflow. Keep that script and `docs/local_eval_libero10_logic_lora.md` in sync if you change this path.

The websocket serving path has already needed one important fix: policy inference must not block keepalive handling for long JAX/XLA steps. Be careful when changing either side of the websocket boundary:

- `src/openpi/serving/websocket_policy_server.py`
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- `examples/libero/main.py`

## Testing Guidance
Run targeted tests first, then broader checks if practical.

Examples:

```bash
uv run pytest src/openpi/training/data_loader_test.py
uv run ruff check src/openpi scripts packages
bash -n examples/libero/eval_checkpoint.slurm scripts/submit_libero_eval_slurm.sh
```

Some tests depend on local datasets, GPU libraries, simulators, or external env setup. If you cannot run something, say exactly what blocked verification.

## Data And Artifact Hygiene
Do not commit large datasets, checkpoints, generated eval outputs, or secrets.

Usually untracked/generated:

- `checkpoints/`
- `logs/`
- `data/libero/evals/`
- ad hoc local envs or temp files

Tracked exceptions are allowed when they are deliberate repo inputs, such as `data/libero/libero_10_logic_descriptions.json`.

Keep experiment notes in tracked markdown like `EXPERIMENTS_APRIL_14.md` or under `docs/`, not in shell history or disposable scratch files.

## Commits And PRs
Keep commits focused and use short imperative subjects. Small stacked commits are preferred over one mixed commit when the changes split cleanly by concern.

If behavior changes, include exact verification steps in the PR or handoff note. If you update scripts or docs that define a workflow, keep them in sync in the same change.
