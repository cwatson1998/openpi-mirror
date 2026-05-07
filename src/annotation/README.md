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

The same replay layer now also powers the "next object highlighting" dataset
transform used by
`examples/libero/convert_libero_data_to_lerobot.py --next-object-highlighting`.
That transform restores saved MuJoCo state, inspects the BDDL
`obj_of_interest`, and re-renders the RGB stream with the nearest future
grasp target highlighted.

One important implementation detail is that RLDS episodes are not always an
exact byte-for-byte mirror of the source HDF5 demo:

- no-op filtering can change episode length by a small number of timesteps
- the exported gripper action convention can differ in sign from the source
  HDF5 actions
- low-dimensional state can drift slightly even when the episode still matches
  the correct source demo

So `annotation.libero_demo_replay.match_demo_key(...)` is intentionally more
tolerant than a strict equality check. If you change that matching logic, make
sure the regression tests in `src/annotation/testing/libero_demo_replay_test.py`
still cover:

- exact same-length matches
- sign-flipped gripper-action matches with matching proprio
- small length mismatches where the overlapping prefix still identifies the
  right source demo

For highlighted LeRobot conversion, tolerant matching is only the source-demo
resolution step. The converter still requires the resolved HDF5 demo to have
the same number of saved states as the RLDS episode before it will rerender and
write highlighted frames.

### Next Object Highlighting

This repo now supports a simulator-backed "next object highlighting" transform
for building new LIBERO datasets from old ones.

The intended use case is:

- start from an existing RLDS LIBERO dataset
- resolve each episode back to its original source HDF5 demo
- restore the saved MuJoCo state at every timestep
- re-render the RGB observations with an object highlight baked into the images
- write the result back out as a normal LeRobot dataset for OpenPI training

The core rule is:

- look only at the BDDL file's `obj_of_interest`
- at each timestep, find the nearest future timestep, counting the current one,
  whose state has one of those objects grasped
- highlight that object in the current timestep's RGB observations

Because the nearest future grasp is reused until a new grasp event appears, this
can be implemented efficiently with one backward pass over the grasp labels.

#### Where The Logic Lives

The main code is in `annotation.libero_demo_replay`:

- `resolve_demo_replay_spec_from_episode(...)`: reuse an already loaded RLDS
  episode when you want to resolve it back to the source HDF5 demo
- `build_next_object_highlight_plan(...)`: compute which object should be
  highlighted at each saved timestep
- `render_next_object_highlighted_demo(...)`: re-render the RGB cameras with the
  per-timestep highlight applied, optionally with a blue placement-target dot

The current dataset-creation entrypoint is:

- `examples/libero/convert_libero_data_to_lerobot.py`

#### How To Create A New Highlighted Dataset

Use the LIBERO simulator environment, not the root repo env, because this path
needs MuJoCo, robosuite, LIBERO, and TFDS together:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir data/libero/raw \
  --repo_name local/libero_spatial_next_object \
  --suite_names libero_spatial \
  --next-object-highlighting \
  --next-object-placement-dot \
  --highlight-rgb 255,105,180 \
  --demo-search-roots third_party/libero/libero/datasets
```

Important details:

- the default highlight color is pink: `255,105,180`
- `--highlight-alpha` controls blending strength
- `--next-object-placement-dot` adds a blue projected dot at the active object's
  placement target; the first version uses the object's body position after
  release, or its final body position if it is never released
- the converter keeps the standard LeRobot keys:
  `image`, `wrist_image`, `state`, `actions`
- this means the existing OpenPI training pipeline can train on the new dataset
  without special data-loader changes
- the converter records the transform settings in `meta/libero_subset.json`

#### Supported Suites

This path is not specific to `libero_spatial`. It works for multiple LIBERO
suites as long as both inputs are present locally:

- raw RLDS shards under `data/libero/raw/<suite>_no_noops/...`
- source demos under `third_party/libero/libero/datasets/<suite>/...`

Examples:

- `libero_spatial`:
  - RLDS: `data/libero/raw/libero_spatial_no_noops/1.0.0`
  - HDF5: `third_party/libero/libero/datasets/libero_spatial`
- `libero_10`:
  - RLDS: `data/libero/raw/libero_10_no_noops/1.0.0`
  - HDF5: `third_party/libero/libero/datasets/libero_10`

#### Rendering A Highlighted MP4 For One Episode

When you want to spot-check the transform before building a whole dataset, it
is often faster to re-render one RLDS episode and save an MP4.

Example: render `libero_10` episode `10` for the task
`put both moka pots on the stove` with a pink tint at `alpha=0.4`:

```bash
mkdir -p outputs/libero_10_moka_pots_highlight

PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python - <<'PY'
from pathlib import Path
import imageio.v2 as imageio

from annotation.libero_demo_replay import load_rlds_episode
from annotation.libero_demo_replay import render_next_object_highlighted_demo
from annotation.libero_demo_replay import resolve_demo_replay_spec_from_episode

output_dir = Path("outputs/libero_10_moka_pots_highlight")
output_dir.mkdir(parents=True, exist_ok=True)

episode = load_rlds_episode(
    dataset_name="libero_10_no_noops",
    data_dir="data/libero/raw",
    episode_index=10,
)
spec = resolve_demo_replay_spec_from_episode(
    episode,
    demo_search_roots=["third_party/libero/libero/datasets"],
    joint_tolerance=2.0,
    state_tolerance=1.2,
)
rendered = render_next_object_highlighted_demo(
    spec,
    camera_names=["agentview"],
    camera_height=256,
    camera_width=256,
    highlight_rgb="255,105,180",
    highlight_alpha=0.4,
)

imageio.mimsave(
    output_dir / "episode_010_agentview_tinted.mp4",
    rendered.frames_by_camera["agentview"],
    fps=10,
)
PY
```

That gives you a simulator-backed highlighted video for one real RLDS episode
without committing to a full RLDS-to-LeRobot conversion first.

#### What Inputs This Requires

This transform needs both dataset layers locally:

- the RLDS data under `data/libero/raw/...`
- the original LIBERO source HDF5 demos, usually under
  `third_party/libero/libero/datasets/...`

RLDS alone is not enough, because the highlighting is generated by restoring the
saved MuJoCo `states` from the source demo and taking fresh simulator renders.

#### Why This Is Different From Simple Image Postprocessing

This is not a static overlay added to already saved dataset JPEGs.

Instead, it:

- restores the exact saved simulator state
- uses the LIBERO instance-segmentation-backed masking wrappers
- applies the highlight to the same simulator RGB observations a policy would
  receive

That is why this feature belongs in the replay / simulator tooling, not in a
pure image-processing script.

#### Training Compatibility

The output of `--next-object-highlighting` is still a standard LeRobot dataset.
From the training code's perspective, it is just another dataset whose `image`
and `wrist_image` streams happen to contain highlighted objects.

So the usual training path still applies:

- convert RLDS to LeRobot
- point a training config at that LeRobot repo
- train with the existing OpenPI data loader

No special model-side feature flag is required just to read the highlighted
images.

#### Test-Time Requirement

If you train a model on highlighted observations, you should expect a train /
test mismatch unless a similar highlighting signal is also present at test time.

This matters a lot:

- the model may learn to depend on the highlight color as a task-relevant cue
- if eval or deployment images do not contain comparable highlighting, the model
  will see a different observation distribution
- in that case, the trained policy may not be very useful even if training loss
  looks good

So for a highlighted-training dataset to be practically useful, you generally
also need an analogous online test-time highlighting mechanism.

One subtle but important caveat:

- the dataset-generation transform uses the nearest future grasp event from the
  recorded demonstration, which is an offline oracle
- a live policy rollout does not have access to the future trajectory

So exact train-time highlighting is not directly available online unless you add
some separate mechanism that predicts or chooses the next object to highlight at
test time.

That means "next object highlighting" is best understood as a dataset-generation
tool plus a representation experiment, not a drop-in standalone deployment
feature.

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

For the sweep-based workflows, add `--only-consider-obj-of-interest` to
restrict the candidate argument pool to the BDDL file's `obj_of_interest` set
instead of all scene instances.

When a sweep considers an object / fixture, it also considers that entity's
regions. When it considers one of those regions, it also includes the parent
entity and the rest of that parent's regions.

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
  --only-consider-obj-of-interest \
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
  --only-consider-obj-of-interest \
  --rotate-images-180 \
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
