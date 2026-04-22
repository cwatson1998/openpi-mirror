"""Visualize first-vs-last predicate truth-value changes for one LIBERO demo.

This tool combines:

- simulator renders for the first and last saved demo states
- the goal-focused predicate truth-value diff report from
  `annotation.libero_predicate_annotation`
- a readable table showing which changed predicates flipped between the two
  states

If the table is too long for one page, the tool emits multiple PNG pages while
repeating the same frame comparison header on each page.

Example usage:
    PYTHONPATH=src:third_party/libero $HOME/miniconda3/envs/instructvla_libero/bin/python -m annotation.libero_predicate_change_visualization --dataset-name libero_spatial_no_noops --data-dir data/libero/raw --episode-index 0 --output-dir outputs/libero_predicate_change_visualization/ep0

Note: use the conda env ($HOME/miniconda3/envs/instructvla_libero/bin/python) rather than
examples/libero/.venv/bin/python when loading RLDS/TFDS datasets, because the latter does
not have tensorflow installed. Alternatively, bypass RLDS entirely by passing
--source-demo-file directly with the path to the HDF5 demo file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import cv2
import numpy as np

from annotation.libero_demo_replay import _build_manifest as _build_render_manifest
from annotation.libero_demo_replay import render_demo
from annotation.libero_demo_replay import render_demo_timestep_indices
from annotation.libero_demo_replay import resolve_demo_replay_spec
from annotation.libero_demo_replay import write_frame_sequence
from annotation.libero_predicate_annotation import LiberoPredicateEvaluator

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


def _ellipsize(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _format_predicate_label(change: dict[str, object]) -> str:
    predicate_name = str(change["predicate_name"])
    arguments = ", ".join(str(argument) for argument in change["arguments"])
    focus_object_name = str(change["focus_object_name"])
    return f"{focus_object_name}: {predicate_name}({arguments})"


def _transition_color(transition: str) -> tuple[int, int, int]:
    if transition == "false_to_true":
        return (212, 245, 212)
    if transition == "true_to_false":
        return (226, 226, 255)
    return (240, 240, 240)


def _truth_cell_color(*, is_true: bool) -> tuple[int, int, int]:
    return (214, 246, 214) if is_true else (236, 236, 236)


def make_predicate_change_visualization_pages(
    *,
    first_frame: np.ndarray,
    last_frame: np.ndarray,
    comparison_report: dict[str, object],
    dataset_name: str | None,
    episode_index: int | None,
    camera_name: str,
    rows_per_page: int = 16,
) -> np.ndarray:
    if first_frame.ndim != 3 or last_frame.ndim != 3:
        raise ValueError("Expected frames with shape [height, width, channels].")
    if first_frame.shape != last_frame.shape:
        raise ValueError(
            "First and last frames must have the same shape. "
            f"Got {first_frame.shape} vs {last_frame.shape}."
        )

    changed_predicates = list(comparison_report["truth_value_changes"]["changed_predicates"])
    task_instruction = str(comparison_report["demo"].get("task_instruction") or "")
    frame_height = int(first_frame.shape[0])
    frame_width = int(first_frame.shape[1])
    canvas_width = max(frame_width * 2 + 96, 1400)
    outer_margin = 28
    header_height = 92
    frame_title_height = 32
    table_header_height = 34
    row_height = 30
    footer_height = 22
    frame_block_top = header_height
    frame_top = frame_block_top + frame_title_height
    frame_x_gap_total = canvas_width - frame_width * 2
    frame_x_left = max(outer_margin, frame_x_gap_total // 3)
    frame_x_right = canvas_width - frame_width - frame_x_left
    table_top = frame_top + frame_height + 28
    table_width = canvas_width - outer_margin * 2
    table_x = outer_margin
    table_columns = {
        "predicate": int(table_width * 0.62),
        "first": int(table_width * 0.12),
        "last": int(table_width * 0.12),
        "change": table_width - int(table_width * 0.62) - int(table_width * 0.12) - int(table_width * 0.12),
    }

    change_summary = comparison_report["truth_value_changes"]["summary"]
    summary_line = (
        f"changed={change_summary['num_changed_predicates']}  "
        f"false->true={change_summary['num_false_to_true']}  "
        f"true->false={change_summary['num_true_to_false']}"
    )
    subtitle_parts = []
    if dataset_name is not None:
        subtitle_parts.append(dataset_name)
    if episode_index is not None:
        subtitle_parts.append(f"episode {episode_index}")
    subtitle_parts.append(f"camera={camera_name}")
    subtitle_line = " | ".join(subtitle_parts)
    task_lines = textwrap.wrap(task_instruction, width=100)[:2] if task_instruction else []

    row_chunks: list[list[dict[str, object]]] = [
        changed_predicates[index : index + rows_per_page]
        for index in range(0, max(len(changed_predicates), 1), rows_per_page)
    ]
    if not row_chunks:
        row_chunks = [[]]

    pages: list[np.ndarray] = []
    max_canvas_height = 0
    for page_index, row_chunk in enumerate(row_chunks):
        visible_rows = max(len(row_chunk), 1)
        canvas_height = table_top + table_header_height + visible_rows * row_height + footer_height
        max_canvas_height = max(max_canvas_height, canvas_height)
        canvas = np.full((canvas_height, canvas_width, 3), 248, dtype=np.uint8)
        canvas[:header_height, :] = 22

        _draw_text_block(
            canvas,
            ["Predicate Truth-Value Changes", subtitle_line, summary_line, *task_lines],
            origin_x=outer_margin,
            origin_y=26,
            font_scale=0.62,
            color=(245, 245, 245),
            line_height=18,
            thickness=1,
        )
        _draw_text_block(
            canvas,
            [f"page {page_index + 1}/{len(row_chunks)}"],
            origin_x=canvas_width - 140,
            origin_y=28,
            font_scale=0.58,
            color=(170, 255, 170),
            line_height=18,
            thickness=1,
        )

        _draw_text_block(
            canvas,
            ["frame 0"],
            origin_x=frame_x_left,
            origin_y=frame_block_top + 24,
            font_scale=0.68,
            color=(40, 40, 40),
            line_height=20,
            thickness=2,
        )
        _draw_text_block(
            canvas,
            [f"frame {comparison_report['last_timestep_annotations']['annotation_timestep_index']}"],
            origin_x=frame_x_right,
            origin_y=frame_block_top + 24,
            font_scale=0.68,
            color=(40, 40, 40),
            line_height=20,
            thickness=2,
        )
        canvas[frame_top : frame_top + frame_height, frame_x_left : frame_x_left + frame_width] = first_frame
        canvas[frame_top : frame_top + frame_height, frame_x_right : frame_x_right + frame_width] = last_frame
        cv2.rectangle(
            canvas,
            (frame_x_left - 2, frame_top - 2),
            (frame_x_left + frame_width + 1, frame_top + frame_height + 1),
            (80, 80, 80),
            2,
        )
        cv2.rectangle(
            canvas,
            (frame_x_right - 2, frame_top - 2),
            (frame_x_right + frame_width + 1, frame_top + frame_height + 1),
            (80, 80, 80),
            2,
        )
        cv2.line(
            canvas,
            (canvas_width // 2, frame_block_top),
            (canvas_width // 2, frame_top + frame_height),
            (220, 220, 220),
            2,
        )

        cv2.rectangle(
            canvas,
            (table_x, table_top),
            (table_x + table_width, table_top + table_header_height),
            (52, 52, 52),
            thickness=-1,
        )
        header_labels = [
            ("Predicate", table_columns["predicate"]),
            ("frame 0", table_columns["first"]),
            ("frame -1", table_columns["last"]),
            ("Change", table_columns["change"]),
        ]
        current_x = table_x
        for label, column_width in header_labels:
            _draw_text_block(
                canvas,
                [label],
                origin_x=current_x + 10,
                origin_y=table_top + 23,
                font_scale=0.54,
                color=(255, 255, 255),
                line_height=18,
                thickness=1,
            )
            current_x += column_width
            cv2.line(canvas, (current_x, table_top), (current_x, canvas_height - footer_height), (210, 210, 210), 1)

        if row_chunk:
            for row_index, change in enumerate(row_chunk):
                row_y = table_top + table_header_height + row_index * row_height
                row_color = _transition_color(str(change["transition"]))
                cv2.rectangle(
                    canvas,
                    (table_x, row_y),
                    (table_x + table_width, row_y + row_height),
                    row_color,
                    thickness=-1,
                )
                cv2.line(canvas, (table_x, row_y), (table_x + table_width, row_y), (215, 215, 215), 1)

                predicate_text = _ellipsize(_format_predicate_label(change), 78)
                current_x = table_x
                _draw_text_block(
                    canvas,
                    [predicate_text],
                    origin_x=current_x + 10,
                    origin_y=row_y + 20,
                    font_scale=0.48,
                    color=(30, 30, 30),
                    line_height=18,
                )
                current_x += table_columns["predicate"]

                first_value = bool(change["first_value"])
                cv2.rectangle(
                    canvas,
                    (current_x, row_y),
                    (current_x + table_columns["first"], row_y + row_height),
                    _truth_cell_color(is_true=first_value),
                    thickness=-1,
                )
                _draw_text_block(
                    canvas,
                    ["true" if first_value else "false"],
                    origin_x=current_x + 18,
                    origin_y=row_y + 20,
                    font_scale=0.5,
                    color=(30, 30, 30),
                    line_height=18,
                    thickness=1,
                )
                current_x += table_columns["first"]

                last_value = bool(change["last_value"])
                cv2.rectangle(
                    canvas,
                    (current_x, row_y),
                    (current_x + table_columns["last"], row_y + row_height),
                    _truth_cell_color(is_true=last_value),
                    thickness=-1,
                )
                _draw_text_block(
                    canvas,
                    ["true" if last_value else "false"],
                    origin_x=current_x + 18,
                    origin_y=row_y + 20,
                    font_scale=0.5,
                    color=(30, 30, 30),
                    line_height=18,
                    thickness=1,
                )
                current_x += table_columns["last"]

                _draw_text_block(
                    canvas,
                    [str(change["transition"]).replace("_", " -> ")],
                    origin_x=current_x + 10,
                    origin_y=row_y + 20,
                    font_scale=0.47,
                    color=(30, 30, 30),
                    line_height=18,
                )
        else:
            empty_row_y = table_top + table_header_height
            cv2.rectangle(
                canvas,
                (table_x, empty_row_y),
                (table_x + table_width, empty_row_y + row_height),
                (240, 248, 240),
                thickness=-1,
            )
            _draw_text_block(
                canvas,
                ["No predicate truth-value changes found for the selected goal-relevant predicate sweep."],
                origin_x=table_x + 10,
                origin_y=empty_row_y + 20,
                font_scale=0.48,
                color=(30, 30, 30),
                line_height=18,
            )

        cv2.rectangle(
            canvas,
            (table_x, table_top),
            (table_x + table_width, table_top + table_header_height + visible_rows * row_height),
            (170, 170, 170),
            1,
        )
        pages.append(canvas)

    padded_pages = []
    for page in pages:
        if page.shape[0] == max_canvas_height:
            padded_pages.append(page)
            continue
        padded_page = np.full((max_canvas_height, page.shape[1], page.shape[2]), 248, dtype=np.uint8)
        padded_page[: page.shape[0], :, :] = page
        padded_pages.append(padded_page)

    return np.stack(padded_pages, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render frame 0 and frame -1 side by side and visualize changed predicate truth values "
            "for goal-relevant objects underneath as a table."
        )
    )
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--data-dir", default="data/libero/raw")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--source-demo-file")
    parser.add_argument("--demo-key")
    parser.add_argument("--demo-search-root", action="append", default=[])
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--predicate", action="append", default=[])
    parser.add_argument("--include-self-relations", action="store_true")
    parser.add_argument("--rows-per-page", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/libero_predicate_change_visualization")
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

    with LiberoPredicateEvaluator.from_spec(spec) as evaluator:
        comparison_report = evaluator.compare_goal_object_predicates_between_timesteps(
            first_timestep_index=0,
            last_timestep_index=-1,
            include_self_relations=args.include_self_relations,
            predicate_names=args.predicate or None,
        )

    first_render = render_demo(
        spec,
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        frame_stride=1,
        max_frames=1,
    )
    last_render = render_demo_timestep_indices(
        spec,
        timestep_indices=[-1],
        camera_name=args.camera_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    pages = make_predicate_change_visualization_pages(
        first_frame=first_render.frames[0],
        last_frame=last_render.frames[0],
        comparison_report=comparison_report,
        dataset_name=args.dataset_name if args.source_demo_file is None else None,
        episode_index=args.episode_index,
        camera_name=args.camera_name,
        rows_per_page=args.rows_per_page,
    )

    render_manifest = _build_render_manifest(
        spec,
        first_render,
        dataset_name=args.dataset_name if args.source_demo_file is None else None,
        data_dir=args.data_dir if args.source_demo_file is None else None,
        episode_index=args.episode_index,
        frame_stride=1,
        max_frames=1,
    )
    manifest = {
        "visualization_type": "predicate_change_first_last",
        "camera_name": args.camera_name,
        "num_pages": int(pages.shape[0]),
        "page_shape": list(pages.shape[1:]),
        "rows_per_page": int(args.rows_per_page),
        "truth_value_change_summary": comparison_report["truth_value_changes"]["summary"],
        "comparison_report": comparison_report,
        "first_frame_render": render_manifest,
        "last_frame_render": {
            **render_manifest,
            "frame_stride": int(spec.states.shape[0]),
            "annotation_timestep_index": comparison_report["last_timestep_annotations"]["annotation_timestep_index"],
        },
    }
    manifest_path = write_frame_sequence(
        pages,
        output_dir=args.output_dir,
        manifest=manifest,
    )

    print(f"Wrote predicate-change visualization pages to {Path(args.output_dir).resolve()}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
