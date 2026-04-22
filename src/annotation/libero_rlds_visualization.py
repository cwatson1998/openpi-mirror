"""Export the saved RLDS image stream for a LIBERO episode.

This tool does not run the simulator. It simply reads the stored JPEG-backed
RLDS observations and writes them out as PNG frames plus an optional video.

Use this when you want to inspect exactly what the RLDS dataset stores.

Example usage:
    PYTHONPATH=src /home/christopher/miniconda3/envs/instructvla_libero/bin/python -m annotation.libero_rlds_visualization --dataset-name libero_spatial_no_noops --data-dir data/libero/raw --episode-index 0 --camera-name image --output-dir outputs/libero_rlds_ep0_image --video-path outputs/libero_rlds_ep0_image.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from annotation.libero_demo_replay import load_rlds_episode
from annotation.libero_demo_replay import write_rlds_episode_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the stored RLDS JPEG frames for a LIBERO episode.")
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--data-dir", default="data/libero/raw")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--camera-name", choices=["image", "wrist_image"], default="image")
    parser.add_argument("--output-dir", default="outputs/libero_rlds_episode")
    parser.add_argument("--video-path")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    episode = load_rlds_episode(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        include_images=True,
    )
    manifest_path = write_rlds_episode_visualization(
        episode,
        output_dir=args.output_dir,
        camera_name=args.camera_name,
        video_path=args.video_path,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    print(f"Wrote RLDS {args.camera_name} frames to {Path(args.output_dir).resolve()}")
    if args.video_path is not None:
        print(f"Wrote video to {Path(args.video_path).resolve()}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
