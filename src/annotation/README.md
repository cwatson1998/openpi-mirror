# Annotation Tools

This directory contains the repo's LIBERO-specific visualization, replay, and
inspection utilities.

The tools here are for three closely related jobs:

- replay a stored LIBERO demo by restoring MuJoCo simulator state
- compare stored dataset images against fresh simulator renders
- inspect how LIBERO resolves BDDL tasks into concrete Python object classes
- inspect world-vs-robot coordinate frames before adding new spatial predicates
- evaluate predicates and derive goal-centered metadata from saved demo timesteps

## Environment

Use the Python 3.10 LIBERO simulator environment for anything here that imports
`libero`, `robosuite`, MuJoCo, or TFDS.

Typical launch pattern:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python -m annotation.<tool> ...
```

Use the root repo env for lightweight checks such as:

```bash
uv run ruff check src/annotation
PYTHONPATH=src uv run pytest src/annotation/libero_demo_replay_test.py
```

## Main Tools

### `annotation.libero_demo_replay`

Core replay library and CLI. It:

- loads an RLDS episode from `data/libero/raw`
- resolves the matching official LIBERO source HDF5 demo
- loads the stored `model_file` XML into the simulator
- restores the saved MuJoCo `states` frame by frame
- renders fresh simulator images

Example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_demo_replay \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --demo-search-root third_party/libero/libero/datasets \
  --camera-width 256 \
  --camera-height 256 \
  --output-dir outputs/libero_demo_replay/ep0_256
```

For masked replay, add:

```bash
  --masked-instance akita_black_bowl_1 \
  --mask-rgb 0,0,0
```

That uses the segmentation-backed masking wrappers in
`third_party/libero/libero/libero/envs/env_wrapper.py`, so the mask affects the
same RGB observations a policy would receive from the simulator.

### `annotation.libero_episode_sanity_check`

Fast visual comparison tool for one RLDS episode.

It exports a labeled video with:

- left: stored RLDS JPEG frames
- middle: fresh simulator replay from saved MuJoCo state
- optional right: masked simulator replay from the same saved state

Example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_episode_sanity_check \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --demo-search-root third_party/libero/libero/datasets \
  --masked-instance akita_black_bowl_1 \
  --mask-rgb 0,0,0 \
  --output-dir outputs/libero_episode_sanity_check/ep0_masked
```

This is the quickest way to sanity-check:

- replay alignment
- camera resolution
- masking behavior

### `annotation.libero_masked_replay_visualization`

Purpose-built side-by-side comparison of unmasked and masked simulator replay
frames.

Use this when you want to inspect only simulator output, not RLDS stored JPEGs.

### `annotation.libero_rlds_visualization`

Exports the saved RLDS image stream directly, without touching the simulator.

Useful when you want to compare:

- what the dataset stores
- what the simulator renders now

### `annotation.libero_resolution_inspector`

Shows how an RLDS episode resolves back to its original LIBERO assets.

For one episode it prints / exports:

- the source demo HDF5 path
- the matched `demo_*` group
- the `model_file` location
- the raw and resolved BDDL path

Example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_resolution_inspector \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --demo-search-root third_party/libero/libero/datasets
```

### `annotation.libero_bddl_object_stack_inspector`

Inspects how one BDDL file, or a whole directory of BDDL files, resolves object
category names like `akita_black_bowl` to concrete implementation classes.

It reports:

- object / fixture categories
- concrete Python class names
- class stacks / MRO
- runtime predicate wrapper types such as `ObjectState` and `SiteObjectState`
- geom / body counts when the object class can be instantiated directly

Example over all `libero_spatial` tasks:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_bddl_object_stack_inspector \
  --bddl-path third_party/libero/libero/libero/bddl_files/libero_spatial
```

This is the best starting point if you are trying to answer questions like:

- "what class implements `plate`?"
- "what does a predicate actually receive at runtime?"
- "is this resolution global, or problem-dependent?"

### `annotation.libero_coordinate_frame_visualization`

Injects small MuJoCo marker dots directly into the scene XML before rendering,
instead of drawing 2D overlays afterward.

It renders:

- a world-frame marker cluster near the robot base
- a robot-base-frame marker cluster attached to the robot root body

This is useful when you are deciding how to define spatial predicates such as
`left-of`, `right-of`, or future robot-relative relations.

Example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_coordinate_frame_visualization \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --demo-search-root third_party/libero/libero/datasets \
  --camera-name agentview \
  --frame-index 0 \
  --output-dir outputs/libero_coordinate_frames/ep0_marker_dots
```

### `annotation.libero_predicate_annotation`

Simulator-backed predicate evaluation and annotation pipeline for saved demos.

It supports two workflows:

- evaluate one predicate with concrete scene-instance arguments at any saved timestep
- annotate the final demo state by extracting all arguments that appear in the
  BDDL goal expression and enumerating which unary / binary predicates hold for
  each of those focus objects
- compare the first and last saved demo states and highlight which predicate
  truth values changed for goal-relevant objects

The automatic annotation pass uses the same LIBERO runtime object wrappers as
normal goal checking, so relations are evaluated against the restored simulator
state rather than inferred from dataset metadata alone.

Single predicate example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_predicate_annotation evaluate \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --predicate on \
  --predicate-args porcelain_mug_1 plate_1
```

Goal-centered final-state annotation example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_predicate_annotation annotate-goal-final-state \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --output-json outputs/libero_predicate_annotations/ep0.json
```

First-vs-last truth-value diff example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_predicate_annotation compare-goal-first-last-state \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --output-json outputs/libero_predicate_annotations/ep0_first_last_diff.json
```

### `annotation.libero_predicate_change_visualization`

Static visualization for the first-vs-last predicate diff workflow.

It renders:

- frame `0` on the left
- frame `-1` on the right
- a table underneath with one row per changed predicate and the truth values at
  the first and last saved states

If the changed-predicate table is long, the tool writes multiple PNG pages while
repeating the same frame comparison header.

Example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  -m annotation.libero_predicate_change_visualization \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --output-dir outputs/libero_predicate_change_visualization/ep0
```

### `annotation.testing.libero_predicates_test`

Lightweight regression coverage for custom LIBERO predicate helpers added in the
submodule.

These tests are intentionally pure Python. They do not build a full simulator;
they just verify predicate semantics and registry-facing helper behavior.

This is the fastest check to run after changing:

- `third_party/libero/libero/libero/envs/predicates/base_predicates.py`
- `third_party/libero/libero/libero/envs/predicates/__init__.py`
- `third_party/libero/libero/libero/envs/object_states/base_object_states.py`

## Related Docs

- [../../MASKING_README.md](../../MASKING_README.md): masking-specific workflow
- [../../examples/libero/README.md](../../examples/libero/README.md): LIBERO eval
  environment and observation contract
- [../../AGENTS.md](../../AGENTS.md): repo workflow, env selection, and mirror
  setup
