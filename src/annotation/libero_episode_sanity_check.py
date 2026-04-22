"""Create a side-by-side sanity-check video for one LIBERO episode.

Panels:
- saved RLDS image stream
- fresh simulator render produced by restoring saved MuJoCo state
- optional masked simulator render produced by restoring saved MuJoCo state

This is the quickest way to check whether simulator-state replay is aligned with
the dataset images for a specific episode. The output adds labels and frame
numbers directly onto the composed frames.

Example usage:
    PYTHONPATH=src:third_party/libero /home/christopher/miniconda3/envs/instructvla_libero/bin/python -m annotation.libero_episode_sanity_check --dataset-name libero_spatial_no_noops --data-dir data/libero/raw --episode-index 0 --demo-search-root third_party/libero/libero/datasets --output-dir outputs/libero_episode_sanity_check/ep0 --video-path outputs/libero_episode_sanity_check/ep0.mp4

Masked comparison usage:
    PYTHONPATH=src:third_party/libero /home/christopher/miniconda3/envs/instructvla_libero/bin/python -m annotation.libero_episode_sanity_check --dataset-name libero_spatial_no_noops --data-dir data/libero/raw --episode-index 0 --demo-search-root third_party/libero/libero/datasets --masked-instance akita_black_bowl_1 --mask-rgb 0,0,0 --output-dir outputs/libero_episode_sanity_check/ep0_masked --video-path outputs/libero_episode_sanity_check/ep0_masked.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import cv2
import numpy as np

from annotation.libero_demo_replay import load_rlds_episode
from annotation.libero_demo_replay import render_demo
from annotation.libero_demo_replay import resolve_demo_replay_spec
from annotation.libero_demo_replay import write_frame_sequence

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _camera_mapping(rlds_camera_name: str) -> str:
    if rlds_camera_name == "image":
        return "agentview"
    if rlds_camera_name == "wrist_image":
        return "robot0_eye_in_hand"
    raise ValueError(f"Unsupported RLDS camera name `{rlds_camera_name}`.")


def _draw_text_block(
    image: np.ndarray,
    lines: list[str],
    *,
    origin_x: int,
    origin_y: int,
    font_scale: float,
    color: tuple[int, int, int],
    line_height: int,
    thickness: int = 1,
) -> None:
    for line_index, line in enumerate(lines):
        y = origin_y + line_index * line_height
        cv2.putText(image, line, (origin_x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)


def make_sanity_check_frames(
    *,
    rlds_frames: np.ndarray,
    simulator_frames: np.ndarray,
    masked_simulator_frames: np.ndarray | None = None,
    dataset_name: str,
    episode_index: int,
    task_instruction: str,
    left_label: str = "RLDS stored JPEG",
    middle_label: str = "Simulator state replay",
    right_label: str = "Masked simulator replay",
) -> np.ndarray:
    if rlds_frames.ndim != 4 or simulator_frames.ndim != 4:
        raise ValueError("Expected frame arrays with shape [num_frames, height, width, channels].")
    if masked_simulator_frames is not None and masked_simulator_frames.ndim != 4:
        raise ValueError("Expected masked simulator frames with shape [num_frames, height, width, channels].")

    frame_sources = [rlds_frames, simulator_frames]
    panel_labels = [left_label, middle_label]
    if masked_simulator_frames is not None:
        frame_sources.append(masked_simulator_frames)
        panel_labels.append(right_label)

    num_frames = min(int(frames.shape[0]) for frames in frame_sources)
    if num_frames == 0:
        raise ValueError("No frames available for sanity-check visualization.")

    expected_shape = rlds_frames.shape[1:]
    for frames in frame_sources[1:]:
        if frames.shape[1:] != expected_shape:
            raise ValueError(
                "All frame sources must have the same spatial shape for comparison export. "
                f"Got {expected_shape} vs {frames.shape[1:]}"
            )

    frame_height = int(rlds_frames.shape[1])
    frame_width = int(rlds_frames.shape[2])
    canvas_height = frame_height + 66
    panel_count = len(frame_sources)
    canvas_width = frame_width * panel_count
    output_frames = []
    task_lines = textwrap.wrap(task_instruction, width=58)[:2]

    for frame_index in range(num_frames):
        canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
        canvas[:66, :] = 18
        for panel_index, frames in enumerate(frame_sources):
            start_x = panel_index * frame_width
            end_x = start_x + frame_width
            canvas[66:, start_x:end_x] = frames[frame_index]
            if panel_index > 0:
                cv2.line(canvas, (start_x, 0), (start_x, canvas_height), (255, 255, 255), 2)

        header_lines = [f"{dataset_name} episode {episode_index}", *task_lines]

        for panel_index, label in enumerate(panel_labels):
            _draw_text_block(
                canvas,
                [label, f"frame {frame_index:04d}"],
                origin_x=panel_index * frame_width + 12,
                origin_y=23,
                font_scale=0.55,
                color=(255, 255, 255),
                line_height=20,
                thickness=1,
            )
        _draw_text_block(
            canvas,
            header_lines,
            origin_x=max(12, canvas_width // 2 - 180),
            origin_y=23,
            font_scale=0.5,
            color=(170, 255, 170),
            line_height=18,
            thickness=1,
        )
        output_frames.append(canvas)

    return np.stack(output_frames, axis=0)


def _build_manifest(
    *,
    dataset_name: str,
    episode_index: int,
    task_instruction: str,
    rlds_camera_name: str,
    sim_camera_name: str,
    num_rlds_frames: int,
    num_simulator_frames: int,
    num_masked_simulator_frames: int | None,
    num_output_frames: int,
    output_shape: tuple[int, int, int],
    source_demo_hdf5: str,
    demo_key: str,
    masked_instances: list[str],
    mask_rgb: str,
    mask_alpha: float,
) -> dict[str, object]:
    manifest = {
        "dataset_name": dataset_name,
        "episode_index": episode_index,
        "task_instruction": task_instruction,
        "rlds_camera_name": rlds_camera_name,
        "sim_camera_name": sim_camera_name,
        "left_source": "rlds_saved_jpeg",
        "middle_source": "simulator_state_replay",
        "num_rlds_frames": num_rlds_frames,
        "num_simulator_frames": num_simulator_frames,
        "num_output_frames": num_output_frames,
        "output_frame_shape": list(output_shape),
        "source_demo_hdf5": source_demo_hdf5,
        "demo_key": demo_key,
    }
    if masked_instances:
        manifest["right_source"] = "masked_simulator_state_replay"
        manifest["num_masked_simulator_frames"] = num_masked_simulator_frames
        manifest["rgb_mask"] = {
            "instance_names": list(masked_instances),
            "mask_rgb": [int(channel) for channel in mask_rgb.split(",")],
            "mask_alpha": float(mask_alpha),
            "camera_names": [sim_camera_name],
        }
    else:
        manifest["right_source"] = "simulator_state_replay"
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side sanity check video comparing RLDS saved frames and simulator replay renders."
    )
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--data-dir", default="data/libero/raw")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--rlds-camera-name", choices=["image", "wrist_image"], default="image")
    parser.add_argument("--sim-camera-name", choices=["agentview", "robot0_eye_in_hand"], default=None)
    parser.add_argument("--demo-search-root", action="append", default=[])
    parser.add_argument("--masked-instance", action="append", default=[])
    parser.add_argument("--mask-rgb", default="0,0,0")
    parser.add_argument("--mask-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/libero_episode_sanity_check")
    parser.add_argument("--video-path")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    sim_camera_name = args.sim_camera_name or _camera_mapping(args.rlds_camera_name)
    episode = load_rlds_episode(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        include_images=True,
    )
    spec = resolve_demo_replay_spec(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        demo_search_roots=args.demo_search_root,
    )

    rlds_frames = episode.image_frames if args.rlds_camera_name == "image" else episode.wrist_image_frames
    if rlds_frames is None:
        raise ValueError(f"Episode does not include `{args.rlds_camera_name}` frames.")

    camera_height = int(rlds_frames.shape[1])
    camera_width = int(rlds_frames.shape[2])
    render_result = render_demo(
        spec,
        camera_name=sim_camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
        max_frames=args.max_frames,
    )
    masked_render_result = None
    if args.masked_instance:
        masked_render_result = render_demo(
            spec,
            camera_name=sim_camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            max_frames=args.max_frames,
            masked_instance_names=args.masked_instance,
            mask_rgb=args.mask_rgb,
            mask_alpha=args.mask_alpha,
            mask_camera_names=[sim_camera_name],
        )

    if args.max_frames is not None:
        rlds_frames = rlds_frames[: args.max_frames]

    sanity_frames = make_sanity_check_frames(
        rlds_frames=rlds_frames,
        simulator_frames=render_result.frames,
        masked_simulator_frames=None if masked_render_result is None else masked_render_result.frames,
        dataset_name=args.dataset_name,
        episode_index=args.episode_index,
        task_instruction=episode.task_instruction,
        left_label=f"RLDS {args.rlds_camera_name}",
        middle_label=f"SIM {sim_camera_name}",
        right_label=f"MASKED SIM {sim_camera_name}",
    )

    manifest = _build_manifest(
        dataset_name=args.dataset_name,
        episode_index=args.episode_index,
        task_instruction=episode.task_instruction,
        rlds_camera_name=args.rlds_camera_name,
        sim_camera_name=sim_camera_name,
        num_rlds_frames=int(rlds_frames.shape[0]),
        num_simulator_frames=int(render_result.frames.shape[0]),
        num_masked_simulator_frames=(
            None if masked_render_result is None else int(masked_render_result.frames.shape[0])
        ),
        num_output_frames=int(sanity_frames.shape[0]),
        output_shape=tuple(int(dim) for dim in sanity_frames.shape[1:]),
        source_demo_hdf5=str(spec.demo_hdf5_path),
        demo_key=spec.demo_key,
        masked_instances=list(args.masked_instance),
        mask_rgb=args.mask_rgb,
        mask_alpha=args.mask_alpha,
    )
    manifest_path = write_frame_sequence(
        sanity_frames,
        output_dir=args.output_dir,
        video_path=args.video_path,
        fps=args.fps,
        manifest=manifest,
    )

    print(f"Wrote sanity-check frames to {Path(args.output_dir).resolve()}")
    if args.video_path is not None:
        print(f"Wrote video to {Path(args.video_path).resolve()}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
