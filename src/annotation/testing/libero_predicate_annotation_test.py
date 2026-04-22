from __future__ import annotations

from annotation.libero_predicate_annotation import build_truth_value_change_report
from annotation.libero_predicate_annotation import extract_goal_argument_names
from annotation.libero_predicate_annotation import group_predicates_by_arity
from annotation.libero_predicate_annotation import iter_focus_object_predicate_calls
from annotation.libero_predicate_annotation import normalize_timestep_index
from annotation.libero_predicate_annotation import predicate_callable_arity
from annotation.libero_predicate_annotation import predicate_evaluation_key


class _UnaryPredicate:
    def __call__(self, arg1):
        return bool(arg1)


class _BinaryPredicate:
    def __call__(self, arg1, arg2):
        return bool(arg1) and bool(arg2)


class _VariadicPredicate:
    def __call__(self, *args):
        return bool(args)


def test_extract_goal_argument_names_preserves_first_appearance_order() -> None:
    goal_state = [
        ["on", "porcelain_mug_1", "plate_1"],
        ["right-of", "chocolate_pudding_1", "plate_1"],
        ["close", "microwave_1"],
    ]

    assert extract_goal_argument_names(goal_state) == (
        "porcelain_mug_1",
        "plate_1",
        "chocolate_pudding_1",
        "microwave_1",
    )


def test_predicate_callable_arity_detects_fixed_and_variadic_callables() -> None:
    assert predicate_callable_arity(_UnaryPredicate()) == 1
    assert predicate_callable_arity(_BinaryPredicate()) == 2
    assert predicate_callable_arity(_VariadicPredicate()) is None


def test_group_predicates_by_arity_excludes_variadic_and_debug_predicates() -> None:
    grouped = group_predicates_by_arity(
        {
            "near": _BinaryPredicate(),
            "grasped": _UnaryPredicate(),
            "printjointstate": _UnaryPredicate(),
            "true": _VariadicPredicate(),
        }
    )

    assert grouped == {
        1: ("grasped",),
        2: ("near",),
    }


def test_iter_focus_object_predicate_calls_enumerates_both_binary_argument_orders() -> None:
    calls = list(
        iter_focus_object_predicate_calls(
            ["plate_1"],
            ["plate_1", "mug_1", "table_region"],
            unary_predicates=["close"],
            binary_predicates=["on", "right-of"],
            include_self_relations=False,
        )
    )

    assert calls == [
        ("plate_1", "close", ("plate_1",)),
        ("plate_1", "on", ("plate_1", "mug_1")),
        ("plate_1", "on", ("mug_1", "plate_1")),
        ("plate_1", "on", ("plate_1", "table_region")),
        ("plate_1", "on", ("table_region", "plate_1")),
        ("plate_1", "right-of", ("plate_1", "mug_1")),
        ("plate_1", "right-of", ("mug_1", "plate_1")),
        ("plate_1", "right-of", ("plate_1", "table_region")),
        ("plate_1", "right-of", ("table_region", "plate_1")),
    ]


def test_normalize_timestep_index_supports_negative_indices() -> None:
    assert normalize_timestep_index(0, 5) == 0
    assert normalize_timestep_index(-1, 5) == 4


def test_predicate_evaluation_key_normalizes_predicate_name() -> None:
    assert predicate_evaluation_key("Right-Of", ["mug_1", "plate_1"]) == (
        "right-of",
        ("mug_1", "plate_1"),
    )


def test_build_truth_value_change_report_detects_transitions() -> None:
    first_snapshot = {
        "annotations": [
            {
                "focus_object_name": "plate_1",
                "focus_object_kind": "object",
                "predicate_evaluations": [
                    {
                        "predicate_name": "on",
                        "arguments": ["mug_1", "plate_1"],
                        "argument_kinds": ["object", "object"],
                        "timestep_index": 0,
                        "value": False,
                    },
                    {
                        "predicate_name": "near",
                        "arguments": ["plate_1", "table_region"],
                        "argument_kinds": ["object", "site"],
                        "timestep_index": 0,
                        "value": True,
                    },
                ],
            }
        ]
    }
    last_snapshot = {
        "annotations": [
            {
                "focus_object_name": "plate_1",
                "focus_object_kind": "object",
                "predicate_evaluations": [
                    {
                        "predicate_name": "on",
                        "arguments": ["mug_1", "plate_1"],
                        "argument_kinds": ["object", "object"],
                        "timestep_index": 9,
                        "value": True,
                    },
                    {
                        "predicate_name": "near",
                        "arguments": ["plate_1", "table_region"],
                        "argument_kinds": ["object", "site"],
                        "timestep_index": 9,
                        "value": False,
                    },
                ],
            }
        ]
    }

    report = build_truth_value_change_report(first_snapshot, last_snapshot)

    assert report["summary"] == {
        "num_changed_predicates": 2,
        "num_false_to_true": 1,
        "num_true_to_false": 1,
        "num_comparable_checks": 2,
        "num_incomparable_checks": 0,
    }
    assert report["false_to_true"] == [
        {
            "focus_object_name": "plate_1",
            "focus_object_kind": "object",
            "predicate_name": "on",
            "arguments": ["mug_1", "plate_1"],
            "argument_kinds": ["object", "object"],
            "first_timestep_index": 0,
            "last_timestep_index": 9,
            "first_value": False,
            "last_value": True,
            "transition": "false_to_true",
        }
    ]
    assert report["true_to_false"] == [
        {
            "focus_object_name": "plate_1",
            "focus_object_kind": "object",
            "predicate_name": "near",
            "arguments": ["plate_1", "table_region"],
            "argument_kinds": ["object", "site"],
            "first_timestep_index": 0,
            "last_timestep_index": 9,
            "first_value": True,
            "last_value": False,
            "transition": "true_to_false",
        }
    ]


def test_build_truth_value_change_report_tracks_incomparable_checks() -> None:
    first_snapshot = {
        "annotations": [
            {
                "focus_object_name": "plate_1",
                "focus_object_kind": "object",
                "predicate_evaluations": [
                    {
                        "predicate_name": "on",
                        "arguments": ["mug_1", "plate_1"],
                        "argument_kinds": ["object", "object"],
                        "timestep_index": 0,
                        "value": False,
                    }
                ],
            }
        ]
    }
    last_snapshot = {
        "annotations": [
            {
                "focus_object_name": "plate_1",
                "focus_object_kind": "object",
                "predicate_evaluations": [],
            }
        ]
    }

    report = build_truth_value_change_report(first_snapshot, last_snapshot)

    assert report["summary"]["num_comparable_checks"] == 0
    assert report["summary"]["num_incomparable_checks"] == 1
    assert report["changed_predicates"] == []
    assert report["incomparable_examples"] == [
        {
            "focus_object_name": "plate_1",
            "predicate_name": "on",
            "arguments": ["mug_1", "plate_1"],
            "reason": "evaluation_missing_from_one_snapshot",
        }
    ]
