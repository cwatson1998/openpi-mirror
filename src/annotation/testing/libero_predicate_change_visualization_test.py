from __future__ import annotations

import numpy as np

from annotation.libero_predicate_change_visualization import _maybe_rotate_image_180
from annotation.libero_predicate_change_visualization import make_predicate_change_visualization_pages


def _comparison_report(*, num_changes: int) -> dict[str, object]:
    changed_predicates = [
        {
            "focus_object_name": f"obj_{index}",
            "focus_object_kind": "object",
            "predicate_name": "on",
            "arguments": [f"obj_{index}", "plate_1"],
            "argument_kinds": ["object", "object"],
            "first_timestep_index": 0,
            "last_timestep_index": 12,
            "first_value": bool(index % 2),
            "last_value": not bool(index % 2),
            "transition": "false_to_true" if index % 2 == 0 else "true_to_false",
        }
        for index in range(num_changes)
    ]
    return {
        "demo": {
            "task_instruction": "put the mug on the plate",
        },
        "last_timestep_annotations": {
            "annotation_timestep_index": 12,
        },
        "truth_value_changes": {
            "changed_predicates": changed_predicates,
            "summary": {
                "num_changed_predicates": num_changes,
                "num_false_to_true": sum(change["transition"] == "false_to_true" for change in changed_predicates),
                "num_true_to_false": sum(change["transition"] == "true_to_false" for change in changed_predicates),
            },
        },
    }


def test_make_predicate_change_visualization_pages_paginates_rows() -> None:
    frame = np.full((20, 30, 3), 127, dtype=np.uint8)
    pages = make_predicate_change_visualization_pages(
        first_frame=frame,
        last_frame=frame,
        comparison_report=_comparison_report(num_changes=5),
        dataset_name="libero_spatial_no_noops",
        episode_index=0,
        camera_name="agentview",
        rows_per_page=2,
    )

    assert pages.shape[0] == 3
    assert pages.shape[-1] == 3


def test_make_predicate_change_visualization_pages_supports_empty_change_list() -> None:
    first_frame = np.zeros((18, 24, 3), dtype=np.uint8)
    last_frame = np.full((18, 24, 3), 255, dtype=np.uint8)
    pages = make_predicate_change_visualization_pages(
        first_frame=first_frame,
        last_frame=last_frame,
        comparison_report=_comparison_report(num_changes=0),
        dataset_name=None,
        episode_index=None,
        camera_name="agentview",
        rows_per_page=4,
    )

    assert pages.shape[0] == 1
    assert pages.shape[-1] == 3
    assert np.any(pages[0] != 248)


def test_maybe_rotate_image_180_rotates_only_when_requested() -> None:
    image = np.array(
        [
            [[1, 0, 0], [2, 0, 0]],
            [[3, 0, 0], [4, 0, 0]],
        ],
        dtype=np.uint8,
    )

    unrotated = _maybe_rotate_image_180(image, rotate_images_180=False)
    rotated = _maybe_rotate_image_180(image, rotate_images_180=True)

    np.testing.assert_array_equal(unrotated, image)
    np.testing.assert_array_equal(
        rotated,
        np.array(
            [
                [[4, 0, 0], [3, 0, 0]],
                [[2, 0, 0], [1, 0, 0]],
            ],
            dtype=np.uint8,
        ),
    )
