"""Compare original and masked simulator-state replay renders side by side.

This tool uses the masking-capable LIBERO wrapper to render the same stored
demo twice:
- left: normal simulator replay
- right: simulator replay with selected instance masks applied to RGB

Example usage:
    PYTHONPATH=src:third_party/libero /home/christopher/Documents/openpi-finetune/openpi/examples/libero/.venv/bin/python -m annotation.libero_masked_replay_visualization --dataset-name libero_spatial_no_noops --data-dir data/libero/raw --episode-index 0 --demo-search-root third_party/libero/libero/datasets --masked-instance akita_black_bowl_1 --mask-rgb 0,0,0 --output-dir outputs/libero_masked_replay/ep0 --video-path outputs/libero_masked_replay/ep0.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import cv2
import numpy as np

from annotation.libero_demo_replay import render_demo
from annotation.libero_demo_replay import resolve_demo_replay_spec
from annotation.libero_demo_replay import write_frame_sequence

_FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def make_mask_comparison_frames(
    *,
    original_frames: np.ndarray,
    masked_frames: np.ndarray,
    dataset_name: str,
    episode_index: int,
    task_instruction: str,
    masked_instances: list[str],
    left_label: str = "Simulator replay",
    right_label: str = "Masked simulator replay",
) -> np.ndarray:
    if original_frames.ndim != 4 or masked_frames.ndim != 4:
        raise ValueError("Expected frame arrays with shape [num_frames, height, width, channels].")
    if original_frames.shape[1:] != masked_frames.shape[1:]:
        raise ValueError(
            "Original and masked frames must have the same spatial shape. "
            f"Got {original_frames.shape[1:]} vs {masked_frames.shape[1:]}"
        )

    num_frames = min(int(original_frames.shape[0]), int(masked_frames.shape[0]))
    frame_height = int(original_frames.shape[1])
    frame_width = int(original_frames.shape[2])
    header_height = 84
    output_frames = []
    task_lines = textwrap.wrap(task_instruction, width=56)[:2]
    mask_line = "masked: " + ", ".join(masked_instances)

    for frame_index in range(num_frames):
        canvas = np.full((frame_height + header_height, frame_width * 2, 3), 244, dtype=np.uint8)
        canvas[:header_height, :] = 22
        canvas[header_height:, :frame_width] = original_frames[frame_index]
        canvas[header_height:, frame_width:] = masked_frames[frame_index]
        cv2.line(canvas, (frame_width, 0), (frame_width, canvas.shape[0]), (255, 255, 255), 2)

        _draw_text_block(
            canvas,
            [left_label, f"frame {frame_index:04d}"],
            origin_x=12,
            origin_y=24,
            font_scale=0.55,
            color=(255, 255, 255),
            line_height=20,
        )
        _draw_text_block(
            canvas,
            [right_label, f"frame {frame_index:04d}"],
            origin_x=frame_width + 12,
            origin_y=24,
            font_scale=0.55,
            color=(255, 255, 255),
            line_height=20,
        )
        _draw_text_block(
            canvas,
            [f"{dataset_name} episode {episode_index}", *task_lines, mask_line],
            origin_x=max(12, frame_width - 210),
            origin_y=24,
            font_scale=0.47,
            color=(170, 255, 170),
            line_height=17,
        )
        output_frames.append(canvas)

    return np.stack(output_frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side video comparing original and masked simulator replay renders."
    )
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--data-dir", default="data/libero/raw")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--demo-search-root", action="append", default=[])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--masked-instance", action="append", default=[])
    parser.add_argument("--mask-rgb", default="0,0,0")
    parser.add_argument("--mask-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/libero_masked_replay")
    parser.add_argument("--video-path")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if not args.masked_instance:
        parser.error("Provide at least one `--masked-instance`.")

    spec = resolve_demo_replay_spec(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        demo_search_roots=args.demo_search_root,
    )
    original = render_demo(
        spec,
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    masked = render_demo(
        spec,
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        masked_instance_names=args.masked_instance,
        mask_rgb=args.mask_rgb,
        mask_alpha=args.mask_alpha,
        mask_camera_names=[args.camera_name],
    )

    comparison_frames = make_mask_comparison_frames(
        original_frames=original.frames,
        masked_frames=masked.frames,
        dataset_name=args.dataset_name,
        episode_index=args.episode_index,
        task_instruction=spec.task_instruction or "",
        masked_instances=list(args.masked_instance),
    )
    manifest = {
        "dataset_name": args.dataset_name,
        "episode_index": args.episode_index,
        "camera_name": args.camera_name,
        "num_frames": int(comparison_frames.shape[0]),
        "frame_shape": list(comparison_frames.shape[1:]),
        "masked_instances": list(args.masked_instance),
        "mask_rgb": [int(channel) for channel in args.mask_rgb.split(",")],
        "mask_alpha": float(args.mask_alpha),
        "source_demo_hdf5": str(spec.demo_hdf5_path),
        "demo_key": spec.demo_key,
    }
    manifest_path = write_frame_sequence(
        comparison_frames,
        output_dir=args.output_dir,
        video_path=args.video_path,
        fps=args.fps,
        manifest=manifest,
    )
    print(f"Wrote comparison frames to {Path(args.output_dir).resolve()}")
    if args.video_path is not None:
        print(f"Wrote comparison video to {Path(args.video_path).resolve()}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
