"""Visualize LIBERO coordinate frames with simulator-native MuJoCo markers.

This tool injects small colored marker geoms into the demo XML before render,
instead of drawing 2D arrows after the fact.

The rendered marker clusters are:

- world-frame cluster near the robot base
- robot-base-frame cluster attached to the robot root body

Each cluster contains:

- origin dot
- +X dot
- +Y dot
- +Z dot

This is intended to make frame relationships easier to inspect before adding
robot-relative predicates such as "left of".

RLDS-backed example:
    PYTHONPATH=src:third_party/libero \\
      /home/christopher/miniconda3/envs/instructvla_libero/bin/python \\
      -m annotation.libero_coordinate_frame_visualization \\
      --dataset-name libero_spatial_no_noops \\
      --data-dir data/libero/raw \\
      --episode-index 0 \\
      --demo-search-root third_party/libero/libero/datasets \\
      --camera-name agentview \\
      --frame-index 0 \\
      --output-dir outputs/libero_coordinate_frames/ep0_marker_dots

Direct source-demo example:
    PYTHONPATH=src:third_party/libero \\
      /home/christopher/miniconda3/envs/instructvla_libero/bin/python \\
      -m annotation.libero_coordinate_frame_visualization \\
      --source-demo-file third_party/libero/libero/datasets/libero_spatial/<task>_demo.hdf5 \\
      --demo-key demo_40 \\
      --camera-name agentview \\
      --output-dir outputs/libero_coordinate_frames/manual_demo_marker_dots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from annotation.libero_demo_replay import RenderResult
from annotation.libero_demo_replay import _load_libero_modules
from annotation.libero_demo_replay import _postprocess_demo_model_xml
from annotation.libero_demo_replay import resolve_demo_replay_spec
from annotation.libero_demo_replay import write_render_outputs

WORLD_MARKER_COLORS = {
    "origin": (255, 255, 255),
    "x": (255, 64, 64),
    "y": (64, 220, 64),
    "z": (64, 128, 255),
}

ROBOT_MARKER_COLORS = {
    "origin": (255, 220, 64),
    "x": (255, 64, 64),
    "y": (64, 220, 64),
    "z": (64, 128, 255),
}


def _rgb255_to_rgba01(color: tuple[int, int, int], alpha: float = 1.0) -> str:
    return " ".join(f"{channel / 255.0:.6f}" for channel in color) + f" {alpha:.6f}"


def _vec_to_str(vec: np.ndarray | tuple[float, float, float] | list[float]) -> str:
    flat = np.asarray(vec, dtype=np.float64).reshape(-1)
    if flat.shape != (3,):
        raise ValueError(f"Expected 3-vector, got shape {flat.shape}.")
    return " ".join(f"{value:.8f}" for value in flat)


def _add_marker_body(
    parent: ET.Element,
    *,
    name: str,
    pos: np.ndarray | tuple[float, float, float] | list[float],
    rgba: tuple[int, int, int],
    radius_m: float,
) -> None:
    body = ET.SubElement(parent, "body", attrib={"name": name, "pos": _vec_to_str(pos)})
    ET.SubElement(
        body,
        "geom",
        attrib={
            "name": f"{name}_geom",
            "type": "sphere",
            "size": f"{radius_m:.8f}",
            "rgba": _rgb255_to_rgba01(rgba),
            "group": "1",
            "contype": "0",
            "conaffinity": "0",
            "mass": "1e-8",
        },
    )


def _inject_coordinate_marker_geoms(
    model_xml: str,
    *,
    robot_root_body_name: str,
    world_anchor: np.ndarray,
    robot_local_anchor: np.ndarray,
    axis_scale_m: float,
    marker_radius_m: float,
) -> str:
    tree = ET.fromstring(model_xml)
    worldbody = tree.find("worldbody")
    if worldbody is None:
        raise ValueError("Model XML does not contain a <worldbody> element.")

    robot_root_body = worldbody.find(f".//body[@name='{robot_root_body_name}']")
    if robot_root_body is None:
        raise ValueError(f"Could not find robot root body `{robot_root_body_name}` in the demo XML.")

    world_cluster = ET.SubElement(worldbody, "body", attrib={"name": "coord_markers_world_cluster"})
    _add_marker_body(
        world_cluster,
        name="coord_marker_world_origin",
        pos=world_anchor,
        rgba=WORLD_MARKER_COLORS["origin"],
        radius_m=marker_radius_m,
    )
    _add_marker_body(
        world_cluster,
        name="coord_marker_world_x",
        pos=world_anchor + np.array([axis_scale_m, 0.0, 0.0], dtype=np.float64),
        rgba=WORLD_MARKER_COLORS["x"],
        radius_m=marker_radius_m,
    )
    _add_marker_body(
        world_cluster,
        name="coord_marker_world_y",
        pos=world_anchor + np.array([0.0, axis_scale_m, 0.0], dtype=np.float64),
        rgba=WORLD_MARKER_COLORS["y"],
        radius_m=marker_radius_m,
    )
    _add_marker_body(
        world_cluster,
        name="coord_marker_world_z",
        pos=world_anchor + np.array([0.0, 0.0, axis_scale_m], dtype=np.float64),
        rgba=WORLD_MARKER_COLORS["z"],
        radius_m=marker_radius_m,
    )

    robot_cluster = ET.SubElement(
        robot_root_body,
        "body",
        attrib={"name": "coord_markers_robot_cluster", "pos": _vec_to_str(robot_local_anchor)},
    )
    _add_marker_body(
        robot_cluster,
        name="coord_marker_robot_origin",
        pos=(0.0, 0.0, 0.0),
        rgba=ROBOT_MARKER_COLORS["origin"],
        radius_m=marker_radius_m,
    )
    _add_marker_body(
        robot_cluster,
        name="coord_marker_robot_x",
        pos=(axis_scale_m, 0.0, 0.0),
        rgba=ROBOT_MARKER_COLORS["x"],
        radius_m=marker_radius_m,
    )
    _add_marker_body(
        robot_cluster,
        name="coord_marker_robot_y",
        pos=(0.0, axis_scale_m, 0.0),
        rgba=ROBOT_MARKER_COLORS["y"],
        radius_m=marker_radius_m,
    )
    _add_marker_body(
        robot_cluster,
        name="coord_marker_robot_z",
        pos=(0.0, 0.0, axis_scale_m),
        rgba=ROBOT_MARKER_COLORS["z"],
        radius_m=marker_radius_m,
    )

    return ET.tostring(tree, encoding="unicode")


def _render_frames_for_states(
    env: Any,
    *,
    states: np.ndarray,
    camera_name: str,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for mujoco_state in states:
        observation = env.set_init_state(mujoco_state)
        frames.append(np.asarray(observation[f"{camera_name}_image"], dtype=np.uint8))
    return np.stack(frames, axis=0)


def _make_side_by_side_frame(raw_frame: np.ndarray, marked_frame: np.ndarray, *, header_lines: list[str]) -> np.ndarray:
    if raw_frame.shape != marked_frame.shape:
        raise ValueError(f"Raw and marked frames must match. Got {raw_frame.shape} vs {marked_frame.shape}.")

    frame_height, frame_width = raw_frame.shape[:2]
    header_height = 90
    canvas = np.full((frame_height + header_height, frame_width * 2, 3), 245, dtype=np.uint8)
    canvas[:header_height, :] = 18
    canvas[header_height:, :frame_width] = raw_frame
    canvas[header_height:, frame_width:] = marked_frame
    cv2.line(canvas, (frame_width, 0), (frame_width, frame_height + header_height), (255, 255, 255), 2)

    cv2.putText(canvas, "Raw simulator render", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "MuJoCo marker dots",
        (frame_width + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    y = 48
    for line in header_lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 255, 170), 1, cv2.LINE_AA)
        y += 18
    return canvas


def render_coordinate_frame_visualization(
    spec: Any,
    *,
    camera_name: str,
    camera_height: int,
    camera_width: int,
    frame_index: int = 0,
    max_frames: int = 1,
    axis_scale_m: float = 0.28,
    marker_radius_m: float = 0.018,
    world_anchor_offset: tuple[float, float, float] = (0.0, -0.10, 0.12),
    robot_local_anchor: tuple[float, float, float] = (0.0, 0.12, 0.12),
) -> tuple[np.ndarray, dict[str, Any]]:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")

    offscreen_render_env_cls, _, libero_postprocess_model_xml = _load_libero_modules()
    base_xml = _postprocess_demo_model_xml(spec.model_xml, libero_postprocess_model_xml)

    probe_env = offscreen_render_env_cls(
        bddl_file_name=spec.bddl_file_name,
        camera_names=[camera_name],
        camera_heights=camera_height,
        camera_widths=camera_width,
    )

    try:
        probe_env.reset()
        probe_env.reset_from_xml_string(base_xml)
        probe_env.sim.reset()

        if frame_index >= len(spec.states):
            raise IndexError(f"frame_index {frame_index} is out of range for {len(spec.states)} stored simulator states.")

        selected_states = spec.states[frame_index : frame_index + max_frames]
        if len(selected_states) == 0:
            raise ValueError("No simulator states selected for visualization.")

        first_observation = probe_env.set_init_state(selected_states[0])
        robot = probe_env.env.robots[0]
        robot_root_body_name = robot.robot_model.root_body
        robot_root_body_id = probe_env.sim.model.body_name2id(robot_root_body_name)
        robot_origin = np.asarray(probe_env.sim.data.body_xpos[robot_root_body_id], dtype=np.float64)
        raw_frames = [np.asarray(first_observation[f"{camera_name}_image"], dtype=np.uint8)]
        if len(selected_states) > 1:
            raw_frames.extend(_render_frames_for_states(probe_env, states=selected_states[1:], camera_name=camera_name))
        raw_frames_array = np.stack(raw_frames, axis=0)
    finally:
        probe_env.close()

    world_anchor = robot_origin + np.asarray(world_anchor_offset, dtype=np.float64)
    marked_xml = _inject_coordinate_marker_geoms(
        base_xml,
        robot_root_body_name=robot_root_body_name,
        world_anchor=world_anchor,
        robot_local_anchor=np.asarray(robot_local_anchor, dtype=np.float64),
        axis_scale_m=axis_scale_m,
        marker_radius_m=marker_radius_m,
    )

    marked_env = offscreen_render_env_cls(
        bddl_file_name=spec.bddl_file_name,
        camera_names=[camera_name],
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    try:
        marked_env.reset()
        marked_env.reset_from_xml_string(marked_xml)
        marked_env.sim.reset()
        marked_frames_array = _render_frames_for_states(marked_env, states=selected_states, camera_name=camera_name)
    finally:
        marked_env.close()

    combined_frames = np.stack(
        [
            _make_side_by_side_frame(
                raw_frame,
                marked_frame,
                header_lines=[
                    f"camera={camera_name} frame={frame_index + local_index}",
                    "World cluster: origin=white, +X=red, +Y=green, +Z=blue",
                    "Robot cluster: origin=yellow, axes use the same RGB mapping",
                ],
            )
            for local_index, (raw_frame, marked_frame) in enumerate(zip(raw_frames_array, marked_frames_array, strict=True))
        ],
        axis=0,
    )

    metadata = {
        "camera_name": camera_name,
        "robot_root_body_name": robot_root_body_name,
        "robot_origin_world": robot_origin.tolist(),
        "world_anchor_world": world_anchor.tolist(),
        "robot_local_anchor": [float(value) for value in robot_local_anchor],
        "axis_scale_m": float(axis_scale_m),
        "marker_radius_m": float(marker_radius_m),
        "world_marker_colors_rgb255": {key: list(value) for key, value in WORLD_MARKER_COLORS.items()},
        "robot_marker_colors_rgb255": {key: list(value) for key, value in ROBOT_MARKER_COLORS.items()},
    }
    return combined_frames, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LIBERO coordinate markers as MuJoCo dots in the scene.")
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--data-dir", default="data/libero/raw")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--source-demo-file")
    parser.add_argument("--demo-key")
    parser.add_argument("--demo-search-root", action="append", default=[])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--axis-scale-m", type=float, default=0.28)
    parser.add_argument("--marker-radius-m", type=float, default=0.018)
    parser.add_argument("--output-dir", default="outputs/libero_coordinate_frames")
    parser.add_argument("--video-path")
    args = parser.parse_args()

    if args.source_demo_file is None and args.episode_index is None:
        parser.error("Provide either `--source-demo-file` or `--episode-index`.")

    spec = resolve_demo_replay_spec(
        dataset_name=args.dataset_name if args.source_demo_file is None else None,
        data_dir=args.data_dir if args.source_demo_file is None else None,
        episode_index=args.episode_index,
        source_demo_file=args.source_demo_file,
        demo_key=args.demo_key,
        demo_search_roots=args.demo_search_root,
    )

    frames, metadata = render_coordinate_frame_visualization(
        spec,
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        frame_index=args.frame_index,
        max_frames=args.max_frames,
        axis_scale_m=args.axis_scale_m,
        marker_radius_m=args.marker_radius_m,
    )

    manifest = {
        "dataset_name": args.dataset_name if args.source_demo_file is None else None,
        "episode_index": args.episode_index,
        "source_demo_file": str(spec.demo_hdf5_path),
        "demo_key": spec.demo_key,
        "bddl_file_name": spec.bddl_file_name,
        "camera_name": args.camera_name,
        "frame_index": int(args.frame_index),
        "max_frames": int(args.max_frames),
        "output_frame_shape": [int(dim) for dim in frames.shape[1:]],
        "marker_metadata": metadata,
    }

    render_result = RenderResult(frames=frames, camera_name=args.camera_name)
    write_render_outputs(
        render_result,
        output_dir=args.output_dir,
        video_path=args.video_path,
        fps=10,
        manifest=manifest,
    )

    marker_json = Path(args.output_dir).expanduser().resolve() / "coordinate_marker_metadata.json"
    marker_json.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote coordinate-marker visualization to {Path(args.output_dir).resolve()}")
    if args.video_path is not None:
        print(f"Wrote video to {Path(args.video_path).resolve()}")
    print(f"Wrote marker metadata to {marker_json}")


if __name__ == "__main__":
    main()
