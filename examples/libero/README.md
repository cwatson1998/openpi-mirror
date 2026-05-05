# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

`third_party/libero` is not a plain copied directory. It is the LIBERO upstream
repo vendored into OpenPI as a Git submodule, so the main repo records only the
submodule URL and pinned commit. Treat local LIBERO edits as submodule work:
create or switch to a branch inside `third_party/libero`, commit those changes
there first, and then commit the updated `third_party/libero` pointer from the
OpenPI repo.

## With Docker

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

export SERVER_ARGS="--env LIBERO"
docker compose -f examples/libero/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip install imageio tqdm tyro mujoco==3.2.3 robosuite==1.4.1 opencv-python bddl==1.0.1 \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install "numpy<2"
uv pip install future easydict cloudpickle gym==0.25.2 hydra-core==1.2.0
uv pip install matplotlib==3.5.3 wandb==0.13.1 transformers==4.21.1 robomimic==0.2.0 einops==0.4.1 thop==0.1.1-2209072238
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=$PWD/.cache/libero-openpi

# Run the simulation
python examples/libero/main.py --args.host 127.0.0.1 --args.port 8000
```

Terminal window 2:

```bash
# Run the server
USE_TF=0 uv run scripts/serve_policy.py --port=8000 --env LIBERO
```

Use Python 3.10 for `examples/libero/.venv`. Python 3.8 is too old for the current `examples/libero/main.py`, and Python 3.11 does not have a matching `torch==1.11.0+cu113` wheel.

## Observation Space

The `pi0_libero` and `pi0_fast_libero` setups in this repo use the same LIBERO observation contract:

- third-person RGB camera: `agentview`
- wrist RGB camera: `robot0_eye_in_hand`
- low-dimensional state: 8D proprio input built from end-effector position, end-effector orientation, and gripper state
- language prompt: task description

At eval time, `examples/libero/main.py` reads the simulator observations:

- `obs["agentview_image"]`
- `obs["robot0_eye_in_hand_image"]`

and sends them to the policy server as:

- `observation/image`
- `observation/wrist_image`
- `observation/state`
- `prompt`

The relevant code is in:

- [examples/libero/main.py](main.py): constructs the LIBERO policy request payload
- [src/openpi/policies/libero_policy.py](../../src/openpi/policies/libero_policy.py): maps LIBERO inputs to the model input format
- [src/openpi/training/config.py](../../src/openpi/training/config.py): dataset repacking for LIBERO training
- [src/openpi/models/model.py](../../src/openpi/models/model.py): canonical model observation keys

Model-side image inputs are always the three canonical keys:

- `base_0_rgb`
- `left_wrist_0_rgb`
- `right_wrist_0_rgb`

For LIBERO:

- `base_0_rgb` = `agentview`
- `left_wrist_0_rgb` = `robot0_eye_in_hand`
- `right_wrist_0_rgb` = zero-padded dummy image

So yes, the policy does use a wrist camera on LIBERO in this repo, but only one real wrist view. The second wrist slot is padding.

## Masking And Replay Debugging

This repo also has a simulator-backed annotation / debugging toolkit under
[`src/annotation/`](../../src/annotation/README.md).

The most useful entrypoints are:

- `annotation.libero_demo_replay`: replay one stored demo by restoring MuJoCo
  simulator state and rendering fresh frames
- `annotation.libero_episode_sanity_check`: compare stored RLDS JPEGs against
  fresh simulator replay, with an optional third masked-simulator panel
- `annotation.libero_masked_replay_visualization`: compare original simulator
  replay against masked simulator replay only
- `annotation.libero_bddl_object_stack_inspector`: inspect how BDDL object
  category names resolve to concrete LIBERO / robosuite classes
- `annotation.libero_coordinate_frame_visualization`: render MuJoCo marker dots
  showing world and robot-base coordinate frames

The masking path is segmentation-backed and can affect policy-facing RGB
observations as well as human-visible replay videos. The eval entrypoint
`examples/libero/main.py` exposes this via:

- `--mask-instances-csv`
- `--mask-rgb-csv`
- `--mask-alpha`
- `--mask-cameras-csv`

For the full masking workflow, including replay examples, see
[`MASKING_README.md`](../../MASKING_README.md).

This eval entrypoint now also supports online next-object highlighting:

- `--next-object-highlighting`
- `--next-object-highlight-rgb-csv`
- `--next-object-highlight-alpha`
- `--next-object-highlight-release-steps`

That mode:

- derives a single grasp order from one source demo for each benchmark task
- starts by highlighting the first grasp target online during rollout
- advances to the next target only after the current highlighted object has been
  grasped and then released for the configured number of consecutive timesteps
- keeps the last grasp target highlighted until the episode ends

This is meant to match the highlighted-dataset training setup as closely as
possible, but note the caveat from the annotation docs: the training dataset
uses an offline future-grasp oracle, while live eval must rely on this simpler
state-machine approximation.

## Dataset Conversion

The training pipeline consumes LeRobot datasets, not raw RLDS directly. The
main conversion entrypoint is:

- [`examples/libero/convert_libero_data_to_lerobot.py`](convert_libero_data_to_lerobot.py)

The standard path is still:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir data/libero/raw \
  --repo_name local/libero_spatial
```

That writes the usual LeRobot image keys:

- `image`
- `wrist_image`
- `state`
- `actions`

so the existing training configs and data loader keep working unchanged.

### Next Object Highlighting

`convert_libero_data_to_lerobot.py` also supports a simulator-backed
`--next-object-highlighting` transform for creating new LIBERO datasets from old
ones.

When enabled, the converter:

1. resolves each RLDS episode back to its source LIBERO HDF5 demo
2. reads the BDDL `obj_of_interest`
3. restores the saved MuJoCo state at every timestep
4. finds the nearest future timestep whose state has an object of interest
   grasped, counting the current timestep
5. re-renders the RGB observations with that object highlighted

The output dataset is still a normal LeRobot dataset, so the existing training
pipeline reads it the same way as any other LIBERO dataset.

Example:

```bash
PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
  examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir data/libero/raw \
  --repo_name local/libero_spatial_next_object \
  --suite_names libero_spatial \
  --next-object-highlighting \
  --highlight-rgb 255,105,180 \
  --demo-search-roots third_party/libero/libero/datasets
```

Notes:

- the default highlight color is pink: `255,105,180`
- `--highlight-alpha` controls blending strength
- this transform requires the original LIBERO source HDF5 demos, not just the
  RLDS shards, because the replay uses saved MuJoCo `states`
- in practice, full-suite conversion also relies on tolerant RLDS-to-HDF5
  matching in `annotation.libero_demo_replay`, because no-noops exports can
  differ slightly in length, gripper-action sign, or low-dimensional state
- tolerant matching is only used to identify the source demo; highlighted
  conversion still requires the matched demo's saved-state length to equal the
  RLDS episode length before it will rerender frames
- the converter records the transform settings in `meta/libero_subset.json`

If you only want to spot-check a single episode before building a full
highlighted dataset, the annotation tooling can also re-render one RLDS episode
to MP4 directly. See the "Rendering A Highlighted MP4 For One Episode" example
in [`src/annotation/README.md`](../../src/annotation/README.md).

## Predicate Work

If you are experimenting with new LIBERO BDDL predicates, the main code path is:

- `third_party/libero/libero/libero/envs/predicates/base_predicates.py`
- `third_party/libero/libero/libero/envs/predicates/__init__.py`
- `third_party/libero/libero/libero/envs/object_states/base_object_states.py`

The current spatial predicates exposed to BDDL use hyphenated planning-style
names:

- `left-of`
- `right-of`
- `in-front-of`
- `behind`
- `above`
- `below`
- `near`

The annotation toolkit includes:

- `annotation.libero_coordinate_frame_visualization` for visualizing world and
  robot-base frames in a real scene
- `src/annotation/testing/libero_predicates_test.py` for lightweight predicate
  regression checks outside the full simulator stack

## Results

If you follow the training instructions and hyperparameters in the `pi0_libero` and `pi0_fast_libero` configs, you should get results similar to the following:

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0-FAST @ 30k (finetuned) | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π0 @ 30k (finetuned) | 96.8 | 98.8 | 95.8 | 85.2 | 94.15 |

Note that the hyperparameters for these runs are not tuned and $\pi_0$-FAST does not use a FAST tokenizer optimized for Libero. Likely, the results could be improved with more tuning, we mainly use these results as an example of how to use openpi to fine-tune $\pi_0$ models on a new dataset.
