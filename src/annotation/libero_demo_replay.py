"""Replay LIBERO demos by restoring saved MuJoCo simulator state.

This is the main shared library for `src/annotation/`.

What this file does:
- load an RLDS episode from `data/libero/raw`
- resolve the corresponding official LIBERO source HDF5 demo
- match the RLDS episode to the correct `/data/demo_x` group
- resolve the BDDL path and demo `model_file` XML
- restore saved MuJoCo state at each timestep and render fresh frames

This is the file to read if you want to understand:
- how RLDS episode metadata is mapped back to a source demo HDF5
- how the source HDF5 `bddl_file_name` is mapped to a local BDDL file
- how the source HDF5 `model_file` XML is fed back into LIBERO for replay

Example usage:
    PYTHONPATH=src:third_party/libero \\
      /home/christopher/miniconda3/envs/instructvla_libero/bin/python \\
      -m annotation.libero_demo_replay \\
      --dataset-name libero_spatial_no_noops \\
      --data-dir data/libero/raw \\
      --episode-index 0 \\
      --demo-search-root third_party/libero/libero/datasets \\
      --camera-width 256 \\
      --camera-height 256 \\
      --masked-instance akita_black_bowl_1 \\
      --mask-rgb 0,0,0 \\
      --output-dir outputs/libero_demo_replay/ep0_256 \\
      --video-path outputs/libero_demo_replay/ep0_256.mp4

Direct-HDF5 usage:
    PYTHONPATH=src:third_party/libero \\
      /home/christopher/miniconda3/envs/instructvla_libero/bin/python \\
      -m annotation.libero_demo_replay \\
      --source-demo-file third_party/libero/libero/datasets/libero_spatial/<task>_demo.hdf5 \\
      --demo-key demo_40 \\
      --output-dir outputs/libero_demo_replay/manual_demo
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import h5py
import imageio.v2 as imageio
import numpy as np
import yaml


@dataclasses.dataclass(frozen=True)
class RldsEpisode:
    dataset_name: str
    episode_index: int
    source_demo_path_hint: str
    task_instruction: str
    actions: np.ndarray
    joint_states: np.ndarray | None = None
    state: np.ndarray | None = None
    image_frames: np.ndarray | None = None
    wrist_image_frames: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class DemoReplaySpec:
    demo_hdf5_path: Path
    demo_key: str
    bddl_file_name: str
    model_xml: str
    states: np.ndarray
    actions: np.ndarray
    task_instruction: str | None = None
    recorded_agentview_frames: np.ndarray | None = None
    recorded_eye_in_hand_frames: np.ndarray | None = None
    source_demo_path_hint: str | None = None
    matching_summary: dict[str, float | int | str] | None = None


@dataclasses.dataclass(frozen=True)
class ResolutionTrace:
    input_path: str
    checked_paths: tuple[Path, ...]
    resolved_path: Path


@dataclasses.dataclass(frozen=True)
class RenderResult:
    frames: np.ndarray
    camera_name: str
    mean_abs_error: float | None = None
    max_abs_error: float | None = None


def _numeric_demo_sort_key(name: str) -> tuple[str, int]:
    prefix, _, suffix = name.partition("_")
    if prefix == "demo" and suffix.isdigit():
        return prefix, int(suffix)
    return name, -1


def _max_abs_diff(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return float("inf")
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _read_nested_dataset(group: h5py.Group, dataset_path: str) -> np.ndarray | None:
    current: h5py.Group | h5py.Dataset = group
    for part in dataset_path.split("/"):
        if not isinstance(current, h5py.Group) or part not in current:
            return None
        current = current[part]
    if not isinstance(current, h5py.Dataset):
        return None
    return current[()]


def _extract_hdf5_state_vector(group: h5py.Group) -> np.ndarray | None:
    ee_states = _read_nested_dataset(group, "obs/ee_states")
    gripper_states = _read_nested_dataset(group, "obs/gripper_states")
    if ee_states is not None and gripper_states is not None:
        return np.concatenate([ee_states, gripper_states], axis=-1)

    robot_states = _read_nested_dataset(group, "robot_states")
    if robot_states is not None and robot_states.ndim == 2 and robot_states.shape[-1] == 8:
        return robot_states

    return None


def _demo_match_metrics(group: h5py.Group, episode: RldsEpisode) -> dict[str, float | int]:
    actions = group["actions"][()]
    joint_states = _read_nested_dataset(group, "obs/joint_states")
    state = _extract_hdf5_state_vector(group)
    return {
        "length_delta": abs(int(actions.shape[0]) - int(episode.actions.shape[0])),
        "action_max_abs_err": _max_abs_diff(actions, episode.actions),
        "joint_max_abs_err": _max_abs_diff(joint_states, episode.joint_states),
        "state_max_abs_err": _max_abs_diff(state, episode.state),
    }


def match_demo_key(
    hdf5_path: str | Path,
    episode: RldsEpisode,
    *,
    action_tolerance: float = 1e-6,
    joint_tolerance: float = 0.1,
    state_tolerance: float = 0.1,
) -> tuple[str, dict[str, float | int]]:
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5_file:
        data_group = h5_file["data"]
        candidates = []
        for demo_key in sorted(data_group.keys(), key=_numeric_demo_sort_key):
            metrics = _demo_match_metrics(data_group[demo_key], episode)
            candidates.append((demo_key, metrics))

    if not candidates:
        raise ValueError(f"No demos found in {hdf5_path}")

    def sort_key(candidate: tuple[str, dict[str, float | int]]) -> tuple[float, float, float, float]:
        _, metrics = candidate
        return (
            float(metrics["length_delta"]),
            float(metrics["action_max_abs_err"]),
            float(metrics["joint_max_abs_err"]),
            float(metrics["state_max_abs_err"]),
        )

    candidates.sort(key=sort_key)
    best_demo_key, best_metrics = candidates[0]

    joint_ok = np.isinf(best_metrics["joint_max_abs_err"]) or best_metrics["joint_max_abs_err"] <= joint_tolerance
    state_ok = np.isinf(best_metrics["state_max_abs_err"]) or best_metrics["state_max_abs_err"] <= state_tolerance
    if (
        best_metrics["length_delta"] == 0
        and best_metrics["action_max_abs_err"] <= action_tolerance
        and joint_ok
        and state_ok
    ):
        return best_demo_key, best_metrics

    preview = [
        {
            "demo_key": demo_key,
            **metrics,
        }
        for demo_key, metrics in candidates[:5]
    ]
    raise ValueError(
        "Could not deterministically match the RLDS episode to a demo group in "
        f"{hdf5_path}. Best candidates:\n{json.dumps(preview, indent=2)}"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_demo_search_roots() -> list[Path]:
    repo_root = _repo_root()
    return [
        repo_root / "third_party/libero/libero/datasets",
        repo_root / "data/libero",
        repo_root / "data",
        repo_root,
    ]


def _dataset_suite_name(dataset_name: str) -> str:
    if dataset_name.endswith("_no_noops"):
        return dataset_name[: -len("_no_noops")]
    return dataset_name


def trace_bddl_file_resolution(bddl_file_name: str | Path) -> ResolutionTrace:
    # BDDL resolution for replay is centralized in this file. The source demo HDF5
    # stores a `data.attrs["bddl_file_name"]` string, and we map that string back
    # onto this checkout's LIBERO tree, usually under
    # `third_party/libero/libero/libero/bddl_files/...`.
    bddl_path = Path(bddl_file_name).expanduser()
    if bddl_path.exists():
        resolved = bddl_path.resolve()
        return ResolutionTrace(
            input_path=str(bddl_file_name),
            checked_paths=(resolved,),
            resolved_path=resolved,
        )

    repo_root = _repo_root()
    direct_candidates = [
        repo_root / bddl_path,
        repo_root / "third_party" / bddl_path,
        repo_root / "third_party/libero" / bddl_path,
        repo_root / "third_party/libero/libero/libero/bddl_files" / bddl_path.name,
    ]
    checked_paths: list[Path] = list(direct_candidates)
    for candidate in direct_candidates:
        if candidate.exists():
            return ResolutionTrace(
                input_path=str(bddl_file_name),
                checked_paths=tuple(checked_paths),
                resolved_path=candidate.resolve(),
            )

    bddl_files_root = repo_root / "third_party/libero/libero/libero/bddl_files"
    for candidate in bddl_files_root.rglob(bddl_path.name):
        if candidate.is_file():
            checked_paths.append(candidate)
            return ResolutionTrace(
                input_path=str(bddl_file_name),
                checked_paths=tuple(checked_paths),
                resolved_path=candidate.resolve(),
            )

    checked = "\n".join(f"  - {candidate}" for candidate in checked_paths)
    raise FileNotFoundError(f"Could not resolve BDDL file `{bddl_file_name}`. Checked:\n{checked}")


def resolve_bddl_file_path(bddl_file_name: str | Path) -> Path:
    return trace_bddl_file_resolution(bddl_file_name).resolved_path


def trace_demo_hdf5_resolution(
    source_demo_path_hint: str,
    *,
    dataset_name: str | None = None,
    demo_search_roots: list[str | Path] | None = None,
) -> ResolutionTrace:
    # RLDS-to-source-demo resolution also lives in this file. The RLDS export
    # stores an `episode_metadata.file_path` hint, often pointing at a remote
    # author-machine location, and we resolve it locally by basename / suite.
    hinted_path = Path(source_demo_path_hint).expanduser()
    if hinted_path.exists():
        resolved = hinted_path.resolve()
        return ResolutionTrace(
            input_path=source_demo_path_hint,
            checked_paths=(resolved,),
            resolved_path=resolved,
        )

    basename = hinted_path.name
    roots = [Path(root).expanduser() for root in (demo_search_roots or _default_demo_search_roots())]
    suite_name = _dataset_suite_name(dataset_name) if dataset_name is not None else None
    checked_locations: list[Path] = []

    for root in roots:
        candidates = [root / basename]
        if suite_name is not None:
            candidates.append(root / suite_name / basename)

        for candidate in candidates:
            checked_locations.append(candidate)
            if candidate.exists():
                return ResolutionTrace(
                    input_path=source_demo_path_hint,
                    checked_paths=tuple(checked_locations),
                    resolved_path=candidate.resolve(),
                )

        if root.exists():
            for candidate in root.rglob(basename):
                checked_locations.append(candidate)
                if candidate.is_file():
                    return ResolutionTrace(
                        input_path=source_demo_path_hint,
                        checked_paths=tuple(checked_locations),
                        resolved_path=candidate.resolve(),
                    )

    checked_preview = "\n".join(f"  - {location}" for location in checked_locations[:20])
    raise FileNotFoundError(
        "Could not resolve the source demo HDF5 from the RLDS metadata hint.\n"
        f"Hint: {source_demo_path_hint}\n"
        f"Searched basename `{basename}` under:\n{checked_preview}"
    )


def resolve_demo_hdf5_path(
    source_demo_path_hint: str,
    *,
    dataset_name: str | None = None,
    demo_search_roots: list[str | Path] | None = None,
) -> Path:
    return trace_demo_hdf5_resolution(
        source_demo_path_hint,
        dataset_name=dataset_name,
        demo_search_roots=demo_search_roots,
    ).resolved_path


def load_rlds_episode(
    *,
    dataset_name: str,
    data_dir: str | Path,
    episode_index: int,
    include_images: bool = False,
) -> RldsEpisode:
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading RLDS episodes requires `tensorflow` and `tensorflow_datasets`. "
            "Use a LIBERO / TFDS-enabled environment or pass `--source-demo-file` directly."
        ) from exc

    with contextlib.suppress(RuntimeError, ValueError):
        tf.config.set_visible_devices([], "GPU")

    dataset = tfds.load(dataset_name, data_dir=str(Path(data_dir).expanduser()), split="train")
    for current_index, episode in enumerate(dataset):
        if current_index != episode_index:
            continue

        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            raise ValueError(f"Episode {episode_index} in {dataset_name} is empty.")

        source_hint = episode["episode_metadata"]["file_path"].numpy().decode()
        return RldsEpisode(
            dataset_name=dataset_name,
            episode_index=episode_index,
            source_demo_path_hint=source_hint,
            task_instruction=steps[-1]["language_instruction"].decode(),
            actions=np.stack([step["action"] for step in steps], axis=0),
            joint_states=np.stack([step["observation"]["joint_state"] for step in steps], axis=0),
            state=np.stack([step["observation"]["state"] for step in steps], axis=0),
            image_frames=(
                np.stack([step["observation"]["image"] for step in steps], axis=0) if include_images else None
            ),
            wrist_image_frames=(
                np.stack([step["observation"]["wrist_image"] for step in steps], axis=0) if include_images else None
            ),
        )

    raise IndexError(f"Episode index {episode_index} is out of range for dataset {dataset_name}.")


def resolve_demo_replay_spec(
    *,
    dataset_name: str | None = None,
    data_dir: str | Path | None = None,
    episode_index: int | None = None,
    source_demo_file: str | Path | None = None,
    demo_key: str | None = None,
    demo_search_roots: list[str | Path] | None = None,
    action_tolerance: float = 1e-6,
    joint_tolerance: float = 0.1,
    state_tolerance: float = 0.1,
) -> DemoReplaySpec:
    if source_demo_file is None and (dataset_name is None or data_dir is None or episode_index is None):
        raise ValueError(
            "Provide either `source_demo_file` directly or the RLDS selectors "
            "`dataset_name`, `data_dir`, and `episode_index`."
        )

    episode: RldsEpisode | None = None
    if source_demo_file is None:
        episode = load_rlds_episode(
            dataset_name=dataset_name,
            data_dir=data_dir,
            episode_index=episode_index,
        )
        source_demo_file = resolve_demo_hdf5_path(
            episode.source_demo_path_hint,
            dataset_name=episode.dataset_name,
            demo_search_roots=demo_search_roots,
        )

    source_demo_file = Path(source_demo_file).expanduser().resolve()
    matching_summary: dict[str, float | int | str] | None = None
    with h5py.File(source_demo_file, "r") as h5_file:
        data_group = h5_file["data"]
        resolved_demo_key = demo_key
        if resolved_demo_key is None:
            if episode is None:
                raise ValueError("`demo_key` is required when loading directly from a source HDF5.")
            resolved_demo_key, matching_metrics = match_demo_key(
                source_demo_file,
                episode,
                action_tolerance=action_tolerance,
                joint_tolerance=joint_tolerance,
                state_tolerance=state_tolerance,
            )
            matching_summary = {"demo_key": resolved_demo_key, **matching_metrics}

        demo_group = data_group[resolved_demo_key]
        bddl_file_name = str(resolve_bddl_file_path(str(data_group.attrs["bddl_file_name"])))
        model_xml = str(demo_group.attrs["model_file"])
        recorded_agentview_frames = _read_nested_dataset(demo_group, "obs/agentview_rgb")
        recorded_eye_in_hand_frames = _read_nested_dataset(demo_group, "obs/eye_in_hand_rgb")

        return DemoReplaySpec(
            demo_hdf5_path=source_demo_file,
            demo_key=resolved_demo_key,
            bddl_file_name=bddl_file_name,
            model_xml=model_xml,
            states=demo_group["states"][()],
            actions=demo_group["actions"][()],
            task_instruction=episode.task_instruction if episode is not None else None,
            recorded_agentview_frames=recorded_agentview_frames,
            recorded_eye_in_hand_frames=recorded_eye_in_hand_frames,
            source_demo_path_hint=episode.source_demo_path_hint if episode is not None else None,
            matching_summary=matching_summary,
        )


def inspect_demo_replay_resolution(
    *,
    dataset_name: str,
    data_dir: str | Path,
    episode_index: int,
    demo_search_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    # This helper is the user-facing explanation layer for resolution:
    # 1. RLDS episode metadata -> source demo HDF5
    # 2. source demo HDF5 `data.attrs["bddl_file_name"]` -> local BDDL file
    # 3. source demo HDF5 `/data/demo_x.attrs["model_file"]` -> XML string used at reset time
    episode = load_rlds_episode(
        dataset_name=dataset_name,
        data_dir=data_dir,
        episode_index=episode_index,
    )
    demo_trace = trace_demo_hdf5_resolution(
        episode.source_demo_path_hint,
        dataset_name=episode.dataset_name,
        demo_search_roots=demo_search_roots,
    )

    with h5py.File(demo_trace.resolved_path, "r") as h5_file:
        data_group = h5_file["data"]
        demo_key, matching_metrics = match_demo_key(demo_trace.resolved_path, episode)
        demo_group = data_group[demo_key]
        raw_bddl_file_name = str(data_group.attrs["bddl_file_name"])
        bddl_trace = trace_bddl_file_resolution(raw_bddl_file_name)
        model_xml = str(demo_group.attrs["model_file"])

    return {
        "dataset_name": dataset_name,
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "episode_index": episode_index,
        "task_instruction": episode.task_instruction,
        "source_demo_path_hint": episode.source_demo_path_hint,
        "source_demo_resolution": {
            "input_path": demo_trace.input_path,
            "checked_paths": [str(path) for path in demo_trace.checked_paths],
            "resolved_path": str(demo_trace.resolved_path),
        },
        "demo_group": {
            "demo_key": demo_key,
            "hdf5_group_path": f"/data/{demo_key}",
            "model_file_attr_path": f"/data/{demo_key} attrs['model_file']",
            "model_file_xml_prefix": model_xml[:200],
        },
        "bddl_resolution": {
            "raw_bddl_file_name": raw_bddl_file_name,
            "checked_paths": [str(path) for path in bddl_trace.checked_paths],
            "resolved_path": str(bddl_trace.resolved_path),
        },
        "matching_summary": {"demo_key": demo_key, **matching_metrics},
        "resolution_notes": {
            "rlds_to_hdf5": "Resolved in src/annotation/libero_demo_replay.py via trace_demo_hdf5_resolution().",
            "hdf5_to_bddl": "Resolved in src/annotation/libero_demo_replay.py via trace_bddl_file_resolution().",
            "model_xml_usage": (
                "The XML string lives in the source HDF5 at `/data/<demo_key>` "
                "attrs['model_file'] and is later passed to reset_from_xml_string()."
            ),
        },
    }


def _libero_config_root(repo_root: Path) -> Path:
    return repo_root / ".cache" / "libero-openpi"


def _ensure_local_libero_config(repo_root: Path) -> None:
    config_root = _libero_config_root(repo_root)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_root))
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    if config_path.exists():
        return

    benchmark_root = repo_root / "third_party/libero/libero/libero"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(repo_root / "third_party/libero/libero/datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    config_path.write_text(yaml.safe_dump(config))


def _load_libero_modules() -> tuple[Any, Any]:
    repo_root = _repo_root()
    libero_root = repo_root / "third_party/libero"
    _ensure_local_libero_config(repo_root)
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))

    from libero.libero.envs import MaskedSegmentationRenderEnv
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml

    return OffScreenRenderEnv, MaskedSegmentationRenderEnv, postprocess_model_xml


def _parse_rgb_triplet(rgb_value: str | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(rgb_value, str):
        parts = [part.strip() for part in rgb_value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Expected mask RGB in `R,G,B` format, got `{rgb_value}`.")
        return tuple(int(part) for part in parts)

    rgb_array = np.asarray(rgb_value, dtype=np.int32).reshape(-1)
    if rgb_array.shape != (3,):
        raise ValueError("Mask RGB must contain exactly three channels.")
    return tuple(int(channel) for channel in rgb_array)


def _postprocess_demo_model_xml(raw_model_xml: str, libero_postprocess_model_xml: Any) -> str:
    xml_with_robosuite_paths = libero_postprocess_model_xml(raw_model_xml, {})
    repo_root = _repo_root()
    libero_assets_root = (repo_root / "third_party/libero/libero/libero/assets").resolve()

    tree = ET.fromstring(xml_with_robosuite_paths)
    for element in tree.findall(".//*[@file]"):
        current_file = element.get("file")
        if current_file is None:
            continue

        current_path = Path(current_file)
        if current_path.exists():
            continue

        candidate: Path | None = None
        normalized = current_file.replace("\\", "/")
        asset_marker = "assets/"
        if asset_marker in normalized:
            asset_suffix = normalized.split(asset_marker, 1)[1]
            candidate = (libero_assets_root / asset_suffix).resolve()
        elif "libero/libero/assets/" in normalized:
            asset_suffix = normalized.split("libero/libero/assets/", 1)[1]
            candidate = (libero_assets_root / asset_suffix).resolve()

        if candidate is not None and candidate.exists():
            element.set("file", str(candidate))

    return ET.tostring(tree, encoding="utf8").decode("utf8")


def _recorded_frames_for_camera(spec: DemoReplaySpec, camera_name: str) -> np.ndarray | None:
    if camera_name == "agentview":
        return spec.recorded_agentview_frames
    if camera_name in {"robot0_eye_in_hand", "eye_in_hand"}:
        return spec.recorded_eye_in_hand_frames
    return None


def _normalize_state_indices(
    timestep_indices: Sequence[int],
    *,
    num_states: int,
) -> list[int]:
    normalized_indices: list[int] = []
    for timestep_index in timestep_indices:
        normalized_index = int(timestep_index)
        if normalized_index < 0:
            normalized_index += num_states
        if normalized_index < 0 or normalized_index >= num_states:
            raise IndexError(
                f"Timestep index {timestep_index} is out of range for {num_states} saved states."
            )
        normalized_indices.append(normalized_index)
    return normalized_indices


def render_demo_timestep_indices(
    spec: DemoReplaySpec,
    *,
    timestep_indices: Sequence[int],
    camera_name: str = "agentview",
    camera_height: int | None = None,
    camera_width: int | None = None,
    masked_instance_names: list[str] | tuple[str, ...] | None = None,
    mask_rgb: tuple[int, int, int] | list[int] | str = (0, 0, 0),
    mask_alpha: float = 1.0,
    mask_camera_names: list[str] | tuple[str, ...] | None = None,
) -> RenderResult:
    if not timestep_indices:
        raise ValueError("Provide at least one timestep index to render.")

    normalized_indices = _normalize_state_indices(timestep_indices, num_states=int(spec.states.shape[0]))
    recorded_frames = _recorded_frames_for_camera(spec, camera_name)
    if recorded_frames is not None:
        camera_height = camera_height or int(recorded_frames.shape[1])
        camera_width = camera_width or int(recorded_frames.shape[2])
    else:
        camera_height = camera_height or 256
        camera_width = camera_width or 256

    offscreen_render_env_cls, masked_segmentation_env_cls, libero_postprocess_model_xml = _load_libero_modules()
    env_kwargs = {
        "bddl_file_name": spec.bddl_file_name,
        "camera_names": [camera_name],
        "camera_heights": camera_height,
        "camera_widths": camera_width,
    }
    masked_instance_names = list(masked_instance_names or [])
    if masked_instance_names:
        env_kwargs["masked_instance_names"] = masked_instance_names
        env_kwargs["mask_rgb"] = _parse_rgb_triplet(mask_rgb)
        env_kwargs["mask_alpha"] = float(mask_alpha)
        env_kwargs["mask_camera_names"] = list(mask_camera_names or [camera_name])
        env = masked_segmentation_env_cls(**env_kwargs)
    else:
        env = offscreen_render_env_cls(**env_kwargs)

    frame_errors: list[float] = []
    try:
        env.reset()
        env.reset_from_xml_string(_postprocess_demo_model_xml(spec.model_xml, libero_postprocess_model_xml))
        env.sim.reset()

        frames = []
        for timestep_index in normalized_indices:
            observation = env.set_init_state(spec.states[timestep_index])
            observation_key = f"{camera_name}_image"
            if observation_key not in observation:
                available = sorted(key for key in observation if key.endswith("_image"))
                raise KeyError(
                    f"Camera `{camera_name}` not found in observations. Available image keys: {available}"
                )

            frame = np.asarray(observation[observation_key], dtype=np.uint8)
            frames.append(frame)
            if recorded_frames is not None and timestep_index < len(recorded_frames):
                recorded_frame = np.asarray(recorded_frames[timestep_index], dtype=np.uint8)
                if recorded_frame.shape == frame.shape:
                    error = np.abs(recorded_frame.astype(np.int16) - frame.astype(np.int16)).mean()
                    frame_errors.append(float(error))
    finally:
        env.close()

    stacked_frames = np.stack(frames, axis=0)
    mean_abs_error = float(np.mean(frame_errors)) if frame_errors else None
    max_abs_error = float(np.max(frame_errors)) if frame_errors else None
    return RenderResult(
        frames=stacked_frames,
        camera_name=camera_name,
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
    )


def render_demo(
    spec: DemoReplaySpec,
    *,
    camera_name: str = "agentview",
    camera_height: int | None = None,
    camera_width: int | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    masked_instance_names: list[str] | tuple[str, ...] | None = None,
    mask_rgb: tuple[int, int, int] | list[int] | str = (0, 0, 0),
    mask_alpha: float = 1.0,
    mask_camera_names: list[str] | tuple[str, ...] | None = None,
) -> RenderResult:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    timestep_indices = list(range(0, int(spec.states.shape[0]), frame_stride))
    if max_frames is not None:
        timestep_indices = timestep_indices[:max_frames]
    return render_demo_timestep_indices(
        spec,
        timestep_indices=timestep_indices,
        camera_name=camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
        masked_instance_names=masked_instance_names,
        mask_rgb=mask_rgb,
        mask_alpha=mask_alpha,
        mask_camera_names=mask_camera_names,
    )


def write_render_outputs(
    render_result: RenderResult,
    *,
    output_dir: str | Path,
    video_path: str | Path | None = None,
    fps: int = 10,
    manifest: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in enumerate(render_result.frames):
        imageio.imwrite(output_dir / f"{frame_index:05d}.png", frame)

    if video_path is not None:
        video_path = Path(video_path).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, render_result.frames, fps=fps)

    if manifest is not None:
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _rlds_frames_for_camera(episode: RldsEpisode, camera_name: str) -> np.ndarray:
    if camera_name == "image":
        if episode.image_frames is None:
            raise ValueError("RLDS episode does not include `image` frames. Reload with include_images=True.")
        return episode.image_frames
    if camera_name == "wrist_image":
        if episode.wrist_image_frames is None:
            raise ValueError("RLDS episode does not include `wrist_image` frames. Reload with include_images=True.")
        return episode.wrist_image_frames
    raise ValueError(f"Unsupported RLDS camera name `{camera_name}`. Use `image` or `wrist_image`.")


def write_rlds_episode_visualization(
    episode: RldsEpisode,
    *,
    output_dir: str | Path,
    camera_name: str = "image",
    video_path: str | Path | None = None,
    fps: int = 10,
    max_frames: int | None = None,
) -> Path:
    frames = _rlds_frames_for_camera(episode, camera_name)
    if max_frames is not None:
        frames = frames[:max_frames]

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in enumerate(frames):
        imageio.imwrite(output_dir / f"{frame_index:05d}.png", frame)

    if video_path is not None:
        video_path = Path(video_path).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=fps)

    manifest = {
        "dataset_name": episode.dataset_name,
        "episode_index": episode.episode_index,
        "task_instruction": episode.task_instruction,
        "source_demo_path_hint": episode.source_demo_path_hint,
        "camera_name": camera_name,
        "num_frames": int(frames.shape[0]),
        "frame_shape": list(frames.shape[1:]),
        "fps": fps,
        "video_path": str(video_path) if video_path is not None else None,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _build_manifest(
    spec: DemoReplaySpec,
    render_result: RenderResult,
    *,
    dataset_name: str | None,
    data_dir: str | None,
    episode_index: int | None,
    frame_stride: int,
    max_frames: int | None,
    masked_instance_names: list[str] | tuple[str, ...] | None = None,
    mask_rgb: tuple[int, int, int] | list[int] | str = (0, 0, 0),
    mask_alpha: float = 1.0,
    mask_camera_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    manifest = {
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "episode_index": episode_index,
        "task_instruction": spec.task_instruction,
        "source_demo_hdf5": str(spec.demo_hdf5_path),
        "source_demo_path_hint": spec.source_demo_path_hint,
        "demo_key": spec.demo_key,
        "bddl_file_name": spec.bddl_file_name,
        "num_states": int(spec.states.shape[0]),
        "num_rendered_frames": int(render_result.frames.shape[0]),
        "render_source": "simulator_state_replay",
        "render_frame_shape": list(render_result.frames.shape[1:]),
        "camera_name": render_result.camera_name,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "mean_abs_error": render_result.mean_abs_error,
        "max_abs_error": render_result.max_abs_error,
    }
    if masked_instance_names:
        manifest["rgb_mask"] = {
            "instance_names": list(masked_instance_names),
            "mask_rgb": list(_parse_rgb_triplet(mask_rgb)),
            "mask_alpha": float(mask_alpha),
            "camera_names": list(mask_camera_names or [render_result.camera_name]),
        }
    if spec.matching_summary is not None:
        manifest["matching_summary"] = spec.matching_summary
    return manifest


def write_frame_sequence(
    frames: np.ndarray,
    *,
    output_dir: str | Path,
    video_path: str | Path | None = None,
    fps: int = 10,
    manifest: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in enumerate(frames):
        imageio.imwrite(output_dir / f"{frame_index:05d}.png", frame)

    if video_path is not None:
        video_path = Path(video_path).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=fps)

    manifest_path = output_dir / "manifest.json"
    if manifest is not None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a stored LIBERO demonstration by replaying saved MuJoCo states.")
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--data-dir", default="data/libero/raw")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--source-demo-file")
    parser.add_argument("--demo-key")
    parser.add_argument("--demo-search-root", action="append", default=[])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--masked-instance", action="append", default=[])
    parser.add_argument("--mask-rgb", default="0,0,0")
    parser.add_argument("--mask-alpha", type=float, default=1.0)
    parser.add_argument("--mask-camera-name", action="append", default=[])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/libero_demo_replay")
    parser.add_argument("--video-path")
    args = parser.parse_args()

    if args.source_demo_file is None and args.episode_index is None:
        parser.error("Provide either `--source-demo-file` or `--episode-index`.")

    spec = resolve_demo_replay_spec(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        source_demo_file=args.source_demo_file,
        demo_key=args.demo_key,
        demo_search_roots=args.demo_search_root,
    )
    render_result = render_demo(
        spec,
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        masked_instance_names=args.masked_instance,
        mask_rgb=args.mask_rgb,
        mask_alpha=args.mask_alpha,
        mask_camera_names=args.mask_camera_name,
    )
    manifest = _build_manifest(
        spec,
        render_result,
        dataset_name=args.dataset_name if args.source_demo_file is None else None,
        data_dir=args.data_dir if args.source_demo_file is None else None,
        episode_index=args.episode_index,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        masked_instance_names=args.masked_instance,
        mask_rgb=args.mask_rgb,
        mask_alpha=args.mask_alpha,
        mask_camera_names=args.mask_camera_name,
    )
    write_render_outputs(
        render_result,
        output_dir=args.output_dir,
        video_path=args.video_path,
        fps=args.fps,
        manifest=manifest,
    )

    print(f"Rendered {render_result.frames.shape[0]} frames to {Path(args.output_dir).resolve()}")
    if args.video_path is not None:
        print(f"Wrote video to {Path(args.video_path).resolve()}")
    if spec.matching_summary is not None:
        print(f"Matched RLDS episode to `{spec.demo_key}` in {spec.demo_hdf5_path}")
    if render_result.mean_abs_error is not None:
        print(
            "Recorded-frame comparison "
            f"(mean abs error={render_result.mean_abs_error:.3f}, max abs error={render_result.max_abs_error:.3f})"
        )


if __name__ == "__main__":
    main()
