"""Create PDF summaries of LIBERO benchmark tasks.

Each page shows the BDDL ``:language`` prompt plus the first and last RGB
frames from a successful source HDF5 demonstration.

Example:
    PYTHONPATH=src .venv/bin/python -m annotation.libero_benchmark_task_pdf \
        --benchmark libero_spatial \
        --output-dir outputs/libero_benchmark_task_pdfs
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
from pathlib import Path
import re
import sys
import textwrap
from typing import Any

import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BDDL_ROOT = Path("third_party/libero/libero/libero/bddl_files")
DEFAULT_DEMO_ROOT = Path("third_party/libero/libero/datasets")
DEFAULT_OUTPUT_DIR = Path("outputs/libero_benchmark_task_pdfs")
DEFAULT_BENCHMARKS = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)

CAMERA_DATASETS = {
    "agentview": "obs/agentview_rgb",
    "agentview_rgb": "obs/agentview_rgb",
    "eye_in_hand": "obs/eye_in_hand_rgb",
    "eye_in_hand_rgb": "obs/eye_in_hand_rgb",
}


@dataclasses.dataclass(frozen=True)
class TaskSnapshot:
    index: int
    benchmark: str
    task_name: str
    language: str
    bddl_path: Path
    demo_path: Path
    demo_key: str
    frame_count: int
    camera_dataset: str
    source_kind: str
    initial_frame: np.ndarray
    final_frame: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (_repo_root() / candidate).resolve()


def _numeric_demo_sort_key(name: str) -> tuple[str, int]:
    prefix, _, suffix = name.partition("_")
    if prefix == "demo" and suffix.isdigit():
        return prefix, int(suffix)
    return name, -1


def _extract_language(bddl_path: Path) -> str:
    text = bddl_path.read_text()
    match = re.search(r"\(:language\s+([^)]*?)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find a :language clause in {bddl_path}")
    return " ".join(match.group(1).split())


def _task_bddl_paths(benchmark: str, bddl_root: Path) -> list[Path]:
    benchmark_dir = bddl_root / benchmark
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"BDDL benchmark directory not found: {benchmark_dir}")

    tasks_info = benchmark_dir / "tasks_info.txt"
    if tasks_info.exists():
        paths = []
        for raw_line in tasks_info.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = benchmark_dir / path.name if benchmark_dir.name in path.parts else benchmark_dir / path
            paths.append(path)
        return paths

    return sorted(benchmark_dir.glob("*.bddl"))


def _camera_dataset_name(camera: str) -> str:
    if "/" in camera:
        return camera
    return CAMERA_DATASETS.get(camera, camera)


def _is_successful_demo(group: h5py.Group) -> bool:
    if "rewards" in group:
        rewards = group["rewards"]
        return bool(rewards.shape[0] and np.max(rewards[()]) > 0)
    if "dones" in group:
        dones = group["dones"]
        return bool(dones.shape[0] and dones[-1])
    return True


def _select_demo_group(data_group: h5py.Group, requested_demo_key: str | None) -> tuple[str, h5py.Group]:
    if requested_demo_key is not None:
        if requested_demo_key not in data_group:
            raise KeyError(f"Demo key {requested_demo_key!r} not found.")
        return requested_demo_key, data_group[requested_demo_key]

    for demo_key in sorted(data_group.keys(), key=_numeric_demo_sort_key):
        group = data_group[demo_key]
        if _is_successful_demo(group):
            return demo_key, group

    raise ValueError("No successful demo group found.")


def _read_rgb_endpoints(group: h5py.Group, camera_dataset: str) -> tuple[np.ndarray, np.ndarray, int]:
    current: h5py.Group | h5py.Dataset = group
    for part in camera_dataset.split("/"):
        if not isinstance(current, h5py.Group) or part not in current:
            raise KeyError(f"Camera dataset {camera_dataset!r} not found in demo group.")
        current = current[part]

    if not isinstance(current, h5py.Dataset):
        raise TypeError(f"Camera path {camera_dataset!r} did not resolve to an HDF5 dataset.")
    if current.ndim != 4 or current.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames shaped (T, H, W, 3), got {current.shape}.")
    if current.shape[0] == 0:
        raise ValueError("Camera dataset has no frames.")

    # Source LIBERO HDF5 demos in this repo use the OpenGL image convention, so
    # the raw recorded RGB arrays need a vertical flip for normal display.
    return np.flipud(current[0][()]), np.flipud(current[-1][()]), int(current.shape[0])


def _load_task_snapshot(
    *,
    benchmark: str,
    task_index: int,
    bddl_path: Path,
    demo_root: Path,
    camera_dataset: str,
    requested_demo_key: str | None,
) -> TaskSnapshot:
    task_name = bddl_path.stem
    demo_path = demo_root / benchmark / f"{task_name}_demo.hdf5"
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo HDF5 not found: {demo_path}")

    with h5py.File(demo_path, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(f"{demo_path} does not contain a /data group.")
        demo_key, demo_group = _select_demo_group(h5_file["data"], requested_demo_key)
        initial_frame, final_frame, frame_count = _read_rgb_endpoints(demo_group, camera_dataset)

    return TaskSnapshot(
        index=task_index,
        benchmark=benchmark,
        task_name=task_name,
        language=_extract_language(bddl_path),
        bddl_path=bddl_path,
        demo_path=demo_path,
        demo_key=demo_key,
        frame_count=frame_count,
        camera_dataset=camera_dataset,
        source_kind="hdf5",
        initial_frame=initial_frame,
        final_frame=final_frame,
    )


def _task_name_from_source_hint(source_hint: str) -> str:
    basename = Path(source_hint).name
    if basename.endswith("_demo.hdf5"):
        return basename[: -len("_demo.hdf5")]
    return Path(basename).stem


def _load_rlds_task_snapshots(
    *,
    benchmark: str,
    bddl_paths: list[Path],
    rlds_data_dir: Path,
    camera_name: str,
) -> dict[str, TaskSnapshot]:
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RLDS fallback requires tensorflow and tensorflow_datasets. "
            "Use the LIBERO/TFDS environment described in AGENTS.md."
        ) from exc

    with contextlib.suppress(RuntimeError, ValueError):
        tf.config.set_visible_devices([], "GPU")

    dataset_name = f"{benchmark}_no_noops"
    dataset_path = rlds_data_dir / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"RLDS dataset not found: {dataset_path}")

    bddl_by_task_name = {path.stem: (index, path) for index, path in enumerate(bddl_paths, start=1)}
    remaining_task_names = set(bddl_by_task_name)
    snapshots: dict[str, TaskSnapshot] = {}
    dataset = tfds.load(dataset_name, data_dir=str(rlds_data_dir), split="train")
    for episode_index, episode in enumerate(dataset):
        source_hint = episode["episode_metadata"]["file_path"].numpy().decode()
        task_name = _task_name_from_source_hint(source_hint)
        if task_name not in remaining_task_names:
            continue

        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            continue
        if camera_name not in steps[0]["observation"]:
            raise KeyError(f"RLDS camera `{camera_name}` not found in {dataset_name}.")

        task_index, bddl_path = bddl_by_task_name[task_name]
        snapshots[task_name] = TaskSnapshot(
            index=task_index,
            benchmark=benchmark,
            task_name=task_name,
            language=_extract_language(bddl_path),
            bddl_path=bddl_path,
            demo_path=Path(source_hint),
            demo_key=f"episode_{episode_index}",
            frame_count=len(steps),
            camera_dataset=f"observation/{camera_name}",
            source_kind="rlds",
            initial_frame=np.asarray(steps[0]["observation"][camera_name], dtype=np.uint8),
            final_frame=np.asarray(steps[-1]["observation"][camera_name], dtype=np.uint8),
        )
        remaining_task_names.remove(task_name)
        if not remaining_task_names:
            break

    return snapshots


def _draw_snapshot_page(pdf: PdfPages, snapshot: TaskSnapshot, *, page_size: tuple[float, float]) -> None:
    figure = plt.figure(figsize=page_size)
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[0.55, 0.12, 1.0],
        hspace=0.12,
        wspace=0.08,
        left=0.055,
        right=0.945,
        top=0.935,
        bottom=0.065,
    )

    text_axis = figure.add_subplot(grid[0, :])
    text_axis.axis("off")
    title = f"{snapshot.benchmark} task {snapshot.index:03d}: {snapshot.task_name}"
    wrapped_prompt = textwrap.fill(snapshot.language, width=105)
    text_axis.text(0.0, 0.96, title, va="top", ha="left", fontsize=11, weight="bold")
    text_axis.text(0.0, 0.62, wrapped_prompt, va="top", ha="left", fontsize=18, wrap=True)
    source = f"{snapshot.source_kind}: {snapshot.demo_path.name} / {snapshot.demo_key} / {snapshot.camera_dataset}"
    text_axis.text(0.0, 0.06, source, va="bottom", ha="left", fontsize=8, color="#555555")

    initial_label_axis = figure.add_subplot(grid[1, 0])
    final_label_axis = figure.add_subplot(grid[1, 1])
    for axis, label in ((initial_label_axis, "Initial observation"), (final_label_axis, "Final observation")):
        axis.axis("off")
        axis.text(0.5, 0.5, label, va="center", ha="center", fontsize=12, weight="bold")

    initial_axis = figure.add_subplot(grid[2, 0])
    final_axis = figure.add_subplot(grid[2, 1])
    for axis, frame in ((initial_axis, snapshot.initial_frame), (final_axis, snapshot.final_frame)):
        axis.imshow(frame)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    pdf.savefig(figure)
    plt.close(figure)


def _write_manifest(output_path: Path, snapshots: list[TaskSnapshot], skipped: list[dict[str, str]]) -> None:
    manifest_path = output_path.with_suffix(".json")
    manifest: dict[str, Any] = {
        "pdf_path": str(output_path),
        "num_pages": len(snapshots),
        "tasks": [
            {
                "index": snapshot.index,
                "benchmark": snapshot.benchmark,
                "task_name": snapshot.task_name,
                "language": snapshot.language,
                "bddl_path": str(snapshot.bddl_path),
                "demo_path": str(snapshot.demo_path),
                "demo_key": snapshot.demo_key,
                "frame_count": snapshot.frame_count,
                "camera_dataset": snapshot.camera_dataset,
                "source_kind": snapshot.source_kind,
            }
            for snapshot in snapshots
        ],
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _download_missing_benchmark(benchmark: str, demo_root: Path) -> None:
    sys.path.insert(0, str(_repo_root() / "third_party/libero"))
    from libero.libero.utils.download_utils import libero_dataset_download

    dataset_name = "libero_100" if benchmark in {"libero_10", "libero_90"} else benchmark
    libero_dataset_download(
        datasets=dataset_name,
        download_dir=str(demo_root),
        check_overwrite=False,
    )


def create_benchmark_pdf(
    *,
    benchmark: str,
    bddl_root: Path,
    demo_root: Path,
    output_dir: Path,
    camera: str,
    demo_key: str | None,
    rlds_data_dir: Path,
    rlds_camera: str,
    rlds_fallback: bool,
    download_missing: bool,
    skip_missing: bool,
    page_size: tuple[float, float],
) -> Path:
    camera_dataset = _camera_dataset_name(camera)
    bddl_paths = _task_bddl_paths(benchmark, bddl_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{benchmark}_tasks.pdf"

    if download_missing and not (demo_root / benchmark).exists():
        _download_missing_benchmark(benchmark, demo_root)

    snapshots: list[TaskSnapshot] = []
    skipped: list[dict[str, str]] = []
    rlds_snapshots: dict[str, TaskSnapshot] | None = None
    for index, bddl_path in enumerate(bddl_paths, start=1):
        try:
            snapshots.append(
                _load_task_snapshot(
                    benchmark=benchmark,
                    task_index=index,
                    bddl_path=bddl_path,
                    demo_root=demo_root,
                    camera_dataset=camera_dataset,
                    requested_demo_key=demo_key,
                )
            )
        except Exception as exc:
            if rlds_fallback:
                if rlds_snapshots is None:
                    rlds_snapshots = _load_rlds_task_snapshots(
                        benchmark=benchmark,
                        bddl_paths=bddl_paths,
                        rlds_data_dir=rlds_data_dir,
                        camera_name=rlds_camera,
                    )
                if bddl_path.stem in rlds_snapshots:
                    snapshots.append(rlds_snapshots[bddl_path.stem])
                    continue
            if not skip_missing:
                raise
            skipped.append(
                {
                    "benchmark": benchmark,
                    "task": bddl_path.stem,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not snapshots:
        raise RuntimeError(f"No task snapshots were loaded for {benchmark}.")

    with PdfPages(output_path) as pdf:
        for snapshot in snapshots:
            _draw_snapshot_page(pdf, snapshot, page_size=page_size)

    _write_manifest(output_path, snapshots, skipped)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark to export. Repeat for multiple suites, or use 'all'.",
    )
    parser.add_argument("--bddl-root", default=str(DEFAULT_BDDL_ROOT))
    parser.add_argument("--demo-root", default=str(DEFAULT_DEMO_ROOT))
    parser.add_argument("--rlds-data-dir", default="data/libero/raw")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--camera", default="agentview", help="Camera alias or HDF5 path, e.g. agentview.")
    parser.add_argument("--rlds-camera", default="image", choices=("image", "wrist_image"))
    parser.add_argument("--demo-key", help="Specific demo key to use, e.g. demo_0. Defaults to first successful demo.")
    parser.add_argument(
        "--no-rlds-fallback",
        action="store_true",
        help="Do not use local RLDS RGB observations when official HDF5 demos are missing.",
    )
    parser.add_argument("--download-missing", action="store_true", help="Download missing official LIBERO HDF5 demos.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip tasks with missing demos/frames.")
    parser.add_argument(
        "--page-size",
        choices=("letter-landscape", "letter-portrait"),
        default="letter-landscape",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    requested = args.benchmark or ["all"]
    benchmarks = list(DEFAULT_BENCHMARKS) if "all" in requested else requested
    page_size = (11.0, 8.5) if args.page_size == "letter-landscape" else (8.5, 11.0)

    bddl_root = _resolve_repo_path(args.bddl_root)
    demo_root = _resolve_repo_path(args.demo_root)
    rlds_data_dir = _resolve_repo_path(args.rlds_data_dir)
    output_dir = _resolve_repo_path(args.output_dir)

    for benchmark in benchmarks:
        output_path = create_benchmark_pdf(
            benchmark=benchmark,
            bddl_root=bddl_root,
            demo_root=demo_root,
            output_dir=output_dir,
            camera=args.camera,
            demo_key=args.demo_key,
            rlds_data_dir=rlds_data_dir,
            rlds_camera=args.rlds_camera,
            rlds_fallback=not args.no_rlds_fallback,
            download_missing=args.download_missing,
            skip_missing=args.skip_missing,
            page_size=page_size,
        )
        print(output_path)


if __name__ == "__main__":
    main()
