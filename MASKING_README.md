# LIBERO RGB Masking

This repo now supports instance-level RGB masking for LIBERO environments.

The feature is meant for two cases:

- policy inference, where the policy should receive RGB observations with one or more object instances masked out
- visualization, where simulator renders should show the same masking behavior for inspection and debugging

The masking is segmentation-backed. It does not edit physics, object state, or MuJoCo assets. It only post-processes returned RGB observations using the simulator's instance segmentation output.

## Where It Lives

The core implementation is in vendored LIBERO:

- `third_party/libero/libero/libero/envs/env_wrapper.py`

Relevant wrappers:

- `OffScreenRenderEnv`: normal RGB observation env
- `SegmentationRenderEnv`: RGB + instance segmentation env
- `MaskedSegmentationRenderEnv`: RGB + instance segmentation env that can overwrite selected instance pixels in returned RGB observations
- `DemoMaskedSegmentationRenderEnv`: same idea, but for demo-style frontview rendering

Exports:

- `third_party/libero/libero/libero/envs/__init__.py`

The policy-eval entrypoint uses this in:

- `examples/libero/main.py`

The annotation / replay tooling uses this in:

- `src/annotation/libero_demo_replay.py`
- `src/annotation/libero_episode_sanity_check.py`
- `src/annotation/libero_masked_replay_visualization.py`
- `src/annotation/README.md`

## How It Works

`MaskedSegmentationRenderEnv` requests `camera_segmentations="instance"` from robosuite. For each camera image observation:

1. read `<camera>_segmentation_instance`
2. map requested instance names to segmentation ids
3. build a boolean mask for those ids
4. replace or alpha-blend the corresponding RGB pixels

This happens in the wrapper layer, so it affects:

- `reset()`
- `step()`
- `set_init_state()` / simulator-state replay

That means the same masking logic applies to:

- policy-facing observations
- replay videos written from simulator observations

It does not retroactively modify pre-saved RLDS JPEGs or the recorded HDF5 demo RGB frames.

## Policy Eval Usage

`examples/libero/main.py` now accepts:

- `--mask-instances-csv`: comma-separated LIBERO instance names to mask
- `--mask-rgb-csv`: comma-separated `R,G,B` values
- `--mask-alpha`: alpha blend in `[0, 1]`
- `--mask-cameras-csv`: optional comma-separated camera names

Example:

```bash
PYTHONPATH=src:packages/openpi-client/src:third_party/libero \
  examples/libero/.venv/bin/python examples/libero/main.py \
  --task-suite-name libero_spatial \
  --mask-instances-csv akita_black_bowl_1 \
  --mask-rgb-csv 0,0,0 \
  --mask-alpha 1.0
```

If `--mask-instances-csv` is empty, eval keeps using `OffScreenRenderEnv` and behavior is unchanged.

## Replay Usage

You can render a masked simulator replay directly from the annotation tooling:

```bash
PYTHONPATH=src:third_party/libero \
  $HOME/miniconda3/envs/instructvla_libero/bin/python \
  -m annotation.libero_demo_replay \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --demo-search-root third_party/libero/libero/datasets \
  --camera-width 256 \
  --camera-height 256 \
  --masked-instance akita_black_bowl_1 \
  --mask-rgb 0,0,0 \
  --output-dir outputs/libero_demo_replay/ep0_masked \
  --video-path outputs/libero_demo_replay/ep0_masked.mp4
```

Important:

- `annotation.libero_demo_replay` loads RLDS metadata, so it needs an environment with TensorFlow and TFDS available
- on this machine, the fallback env `$HOME/miniconda3/envs/instructvla_libero/bin/python` worked for that

## Side-By-Side Visualization

To compare original simulator replay and masked simulator replay:

```bash
PYTHONPATH=src:third_party/libero \
  $HOME/miniconda3/envs/instructvla_libero/bin/python \
  -m annotation.libero_masked_replay_visualization \
  --dataset-name libero_spatial_no_noops \
  --data-dir data/libero/raw \
  --episode-index 0 \
  --demo-search-root third_party/libero/libero/datasets \
  --masked-instance akita_black_bowl_1 \
  --mask-rgb 0,0,0 \
  --camera-width 256 \
  --camera-height 256 \
  --output-dir outputs/libero_masked_replay/ep0_bowl \
  --video-path outputs/libero_masked_replay/ep0_bowl.mp4
```

That command writes:

- side-by-side PNG frames
- a side-by-side MP4
- `manifest.json`

The current real example artifact is:

- `outputs/libero_masked_replay/ep0_bowl.mp4`

## RLDS vs Simulator vs Masked Simulator

If you want the most direct sanity check for one RLDS episode, use:

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

That export shows:

- left: saved RLDS JPEG frames
- middle: fresh simulator replay from saved MuJoCo state
- right: masked simulator replay from the same saved state

This is the quickest display test for confirming that:

- replay alignment is still good
- the mask is being applied by simulator-backed rendering rather than by copying
  stored dataset images
- the masked RGB is what a policy would see from the wrapped env

## Choosing Instance Names

Masking uses LIBERO instance names, not category names.

Example:

- `akita_black_bowl_1`

These names come from the environment model's `instances_to_ids` mapping. The wrapper resolves them to segmentation ids internally.

If you pass an instance name that is not present in the loaded env, the masking wrapper raises a `KeyError`.

## Notes And Limitations

- The mask is per-instance and per-camera.
- The default behavior is a solid-color overwrite.
- Alpha blending is supported through `mask_alpha`.
- Physics, contact geometry, and object states are unchanged.
- This depends on robosuite instance segmentation observables being available for the selected cameras.
- If you use a Python 3.10 LIBERO env that does not also have TFDS, policy eval masking still works, but RLDS-backed replay helpers may not.

## Verification That Was Run

Lightweight checks:

- `PYTHONPATH=src .venv/bin/python -m pytest src/annotation/libero_demo_replay_test.py`
- `.venv/bin/ruff check src/annotation examples/libero/main.py`

Simulator-backed checks:

- smoke test through `examples/libero/main.py` confirmed that masked eval instantiates `MaskedSegmentationRenderEnv`
- side-by-side masked replay visualization for `libero_spatial_no_noops` episode `0`
- numeric diff check on the first 5 masked replay frames showed changed pixels and nonzero image deltas

## Git Layout

This feature spans two repos:

- main repo branch: `chr/libero-annotation-tools`
- LIBERO submodule branch: `chr/libero-appearance-overrides`

The clean workflow is:

1. commit LIBERO wrapper changes inside `third_party/libero`
2. push that LIBERO branch somewhere writable
3. commit the updated submodule pointer in the main repo
4. push the main repo branch

If `third_party/libero` still points at upstream `https://github.com/Lifelong-Robot-Learning/LIBERO.git`, pushing the submodule branch may fail unless you have permission there or add your own fork as another remote.
