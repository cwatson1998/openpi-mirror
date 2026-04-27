"""
Convert LIBERO RLDS data to LeRobot format.

Examples:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir data/libero/raw --download

uv run examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir data/libero/raw \
    --repo_name local/libero_object \
    --suite_names libero_object \
    --task_suite_name libero_object \
    --task_indices 0 1 2 3 4 5 6

PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
    examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir data/libero/raw \
    --repo_name local/libero_spatial_next_object \
    --suite_names libero_spatial \
    --next-object-highlighting \
    --demo-search-roots third_party/libero/libero/datasets
"""

from collections import Counter
from collections.abc import Sequence
import json
import pathlib
import shutil

from huggingface_hub import snapshot_download
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tensorflow_datasets as tfds
import tyro

from annotation import RldsEpisode
from annotation import render_next_object_highlighted_demo
from annotation import resolve_demo_replay_spec_from_episode
from openpi.training import libero as libero_utils

_RLDS_TO_SIM_CAMERA_NAMES = {
    "image": "agentview",
    "wrist_image": "robot0_eye_in_hand",
}


def _download_raw_datasets(data_dir: pathlib.Path, raw_dataset_names: Sequence[str]) -> None:
    allow_patterns = [f"{raw_dataset_name}/*" for raw_dataset_name in raw_dataset_names]
    snapshot_download(
        repo_id="openvla/modified_libero_rlds",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=allow_patterns,
    )


def _write_subset_metadata(
    output_path: pathlib.Path,
    *,
    selected_tasks: Sequence[str],
    suite_names: Sequence[str],
    task_counts: Counter[str],
    skipped_task_counts: Counter[str],
    next_object_highlighting: dict[str, object] | None = None,
) -> None:
    metadata = {
        "suite_names": list(suite_names),
        "selected_task_instructions": list(selected_tasks),
        "selected_episode_counts": dict(sorted(task_counts.items())),
        "skipped_episode_counts": dict(sorted(skipped_task_counts.items())),
    }
    if next_object_highlighting is not None:
        metadata["next_object_highlighting"] = next_object_highlighting
    metadata_path = output_path / "meta" / "libero_subset.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def _build_rlds_episode(
    *,
    dataset_name: str,
    episode_index: int,
    source_demo_path_hint: str,
    task_instruction: str,
    steps: Sequence[dict],
) -> RldsEpisode:
    return RldsEpisode(
        dataset_name=dataset_name,
        episode_index=episode_index,
        source_demo_path_hint=source_demo_path_hint,
        task_instruction=task_instruction,
        actions=np.stack([step["action"] for step in steps], axis=0),
        joint_states=np.stack([step["observation"]["joint_state"] for step in steps], axis=0),
        state=np.stack([step["observation"]["state"] for step in steps], axis=0),
    )


def _render_next_object_highlighted_frames(
    *,
    dataset_name: str,
    episode_index: int,
    source_demo_path_hint: str,
    task_instruction: str,
    steps: Sequence[dict],
    demo_search_roots: Sequence[str],
    highlight_rgb: str,
    highlight_alpha: float,
) -> dict[str, np.ndarray]:
    episode = _build_rlds_episode(
        dataset_name=dataset_name,
        episode_index=episode_index,
        source_demo_path_hint=source_demo_path_hint,
        task_instruction=task_instruction,
        steps=steps,
    )
    spec = resolve_demo_replay_spec_from_episode(
        episode,
        demo_search_roots=list(demo_search_roots) or None,
        # Full-suite RLDS exports can drift slightly from source HDF5 lengths and
        # proprio traces after no-op filtering, so the rerender path needs more
        # tolerance than the lightweight replay defaults.
        joint_tolerance=2.0,
        state_tolerance=1.2,
    )

    if len(steps) != int(spec.states.shape[0]):
        raise ValueError(
            "The resolved source demo length does not match the RLDS episode length. "
            f"Episode {episode_index} in {dataset_name} has {len(steps)} steps, but the matched demo "
            f"{spec.demo_hdf5_path}:{spec.demo_key} has {int(spec.states.shape[0])} saved states."
        )

    camera_height = int(steps[0]["observation"]["image"].shape[0])
    camera_width = int(steps[0]["observation"]["image"].shape[1])
    render_result = render_next_object_highlighted_demo(
        spec,
        camera_names=[_RLDS_TO_SIM_CAMERA_NAMES["image"], _RLDS_TO_SIM_CAMERA_NAMES["wrist_image"]],
        camera_height=camera_height,
        camera_width=camera_width,
        highlight_rgb=highlight_rgb,
        highlight_alpha=highlight_alpha,
    )
    return {
        "image": render_result.frames_by_camera[_RLDS_TO_SIM_CAMERA_NAMES["image"]],
        "wrist_image": render_result.frames_by_camera[_RLDS_TO_SIM_CAMERA_NAMES["wrist_image"]],
    }


def main(
    data_dir: str,
    *,
    repo_name: str = "your_hf_username/libero",
    suite_names: Sequence[str] = tuple(libero_utils.LIBERO_RAW_DATASETS),
    download: bool = False,
    task_suite_name: str | None = None,
    task_indices: Sequence[int] = (),
    task_names: Sequence[str] = (),
    task_split_file: str | None = None,
    task_split: str = "train",
    next_object_highlighting: bool = False,
    highlight_rgb: str = "255,105,180",
    highlight_alpha: float = 1.0,
    demo_search_roots: Sequence[str] = (),
    push_to_hub: bool = False,
):
    data_dir_path = pathlib.Path(data_dir).expanduser().resolve()
    raw_dataset_names = [libero_utils.LIBERO_RAW_DATASETS[suite_name] for suite_name in suite_names]
    selected_tasks = libero_utils.resolve_task_filters(
        task_suite_name=task_suite_name,
        task_indices=task_indices,
        task_names=task_names,
        task_split_file=task_split_file,
        task_split=task_split,
    )
    selected_task_set = {task.casefold() for task in selected_tasks}

    if download:
        _download_raw_datasets(data_dir_path, raw_dataset_names)

    output_path = LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    written_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    for raw_dataset_name in raw_dataset_names:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=str(data_dir_path), split="train")
        for episode_index, episode in enumerate(raw_dataset):
            steps = list(episode["steps"].as_numpy_iterator())
            task_instruction = steps[-1]["language_instruction"].decode()

            if selected_task_set and task_instruction.casefold() not in selected_task_set:
                skipped_counts[task_instruction] += 1
                continue

            highlighted_frames: dict[str, np.ndarray] | None = None
            if next_object_highlighting:
                # The train loader still expects the usual `image` / `wrist_image`
                # keys, so we re-render only the RGB streams and keep the rest of
                # the LeRobot episode schema unchanged.
                highlighted_frames = _render_next_object_highlighted_frames(
                    dataset_name=raw_dataset_name,
                    episode_index=episode_index,
                    source_demo_path_hint=episode["episode_metadata"]["file_path"].numpy().decode(),
                    task_instruction=task_instruction,
                    steps=steps,
                    demo_search_roots=demo_search_roots,
                    highlight_rgb=highlight_rgb,
                    highlight_alpha=highlight_alpha,
                )

            for step_index, step in enumerate(steps):
                dataset.add_frame(
                    {
                        "image": (
                            step["observation"]["image"]
                            if highlighted_frames is None
                            else highlighted_frames["image"][step_index]
                        ),
                        "wrist_image": (
                            step["observation"]["wrist_image"]
                            if highlighted_frames is None
                            else highlighted_frames["wrist_image"][step_index]
                        ),
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                    }
                )
            dataset.save_episode(task=task_instruction)
            written_counts[task_instruction] += 1

    if not written_counts:
        raise ValueError("No LIBERO episodes matched the selected task filter.")

    dataset.consolidate(run_compute_stats=False)
    _write_subset_metadata(
        output_path,
        selected_tasks=selected_tasks,
        suite_names=suite_names,
        task_counts=written_counts,
        skipped_task_counts=skipped_counts,
        next_object_highlighting=(
            None
            if not next_object_highlighting
            else {
                "enabled": True,
                "highlight_rgb": [int(channel) for channel in highlight_rgb.split(",")],
                "highlight_alpha": float(highlight_alpha),
                "demo_search_roots": list(demo_search_roots),
                "requires_source_hdf5": True,
                "selection_rule": (
                    "At each timestep, highlight the BDDL obj_of_interest instance "
                    "from the nearest future timestep whose state is grasped. "
                    "The current timestep counts."
                ),
            }
        ),
    )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Wrote {sum(written_counts.values())} episodes to {output_path}")
    for task_name, count in written_counts.most_common():
        print(f"  {count:3d}  {task_name}")


if __name__ == "__main__":
    tyro.cli(main)
