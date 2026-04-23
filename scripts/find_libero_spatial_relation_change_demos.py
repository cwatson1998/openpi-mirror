#!/usr/bin/env python3
"""Find LIBERO demos whose first-vs-last predicate diff changes key spatial relations.

This script scans every `demo_*` entry in every local source HDF5 under a task
suite directory such as `third_party/libero/libero/datasets/libero_spatial`.
It records only the episodes whose changed predicates include at least one of:

- `left-of`
- `right-of`
- `behind`
- `in-front-of`

Each match record includes the task suite, task name, and episode index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py

from annotation.libero_demo_replay import resolve_demo_replay_spec
from annotation.libero_predicate_annotation import LiberoPredicateEvaluator

TARGET_PREDICATES = ("left-of", "right-of", "behind", "in-front-of")
TRANSITION_FILTER_VALUES = ("any", "false_to_true", "true_to_false")


def _numeric_demo_sort_key(name: str) -> tuple[str, int]:
    prefix, _, suffix = name.partition("_")
    if prefix == "demo" and suffix.isdigit():
        return prefix, int(suffix)
    return name, -1


def _episode_index_from_demo_key(demo_key: str) -> int | None:
    prefix, _, suffix = demo_key.partition("_")
    if prefix == "demo" and suffix.isdigit():
        return int(suffix)
    return None


def _task_name_from_demo_path(demo_path: Path) -> str:
    suffix = "_demo"
    if demo_path.stem.endswith(suffix):
        return demo_path.stem[: -len(suffix)]
    return demo_path.stem


def _iter_demo_keys(demo_path: Path) -> list[str]:
    with h5py.File(demo_path, "r") as h5_file:
        return sorted(h5_file["data"].keys(), key=_numeric_demo_sort_key)


def _serialize_changed_predicates(changed_predicates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "focus_object_name": change["focus_object_name"],
            "focus_object_kind": change["focus_object_kind"],
            "predicate_name": change["predicate_name"],
            "arguments": list(change["arguments"]),
            "argument_kinds": list(change["argument_kinds"]),
            "first_value": bool(change["first_value"]),
            "last_value": bool(change["last_value"]),
            "transition": change["transition"],
        }
        for change in changed_predicates
    ]


def _filter_changed_predicates(
    changed_predicates: list[dict[str, Any]],
    *,
    transition_filter: str,
) -> list[dict[str, Any]]:
    if transition_filter == "any":
        return changed_predicates
    return [
        change
        for change in changed_predicates
        if str(change["transition"]) == transition_filter
    ]


def _scan_demo(
    *,
    demo_path: Path,
    demo_key: str,
    task_suite_name: str,
    only_consider_obj_of_interest: bool,
    include_self_relations: bool,
    transition_filter: str,
) -> dict[str, Any] | None:
    spec = resolve_demo_replay_spec(
        source_demo_file=demo_path,
        demo_key=demo_key,
    )
    with LiberoPredicateEvaluator.from_spec(spec) as evaluator:
        report = evaluator.compare_goal_object_predicates_between_timesteps(
            first_timestep_index=0,
            last_timestep_index=-1,
            include_self_relations=include_self_relations,
            predicate_names=TARGET_PREDICATES,
            only_consider_obj_of_interest=only_consider_obj_of_interest,
            max_skip_examples=0,
        )

    changed_predicates = _filter_changed_predicates(
        list(report["truth_value_changes"]["changed_predicates"]),
        transition_filter=transition_filter,
    )
    if not changed_predicates:
        return None

    num_false_to_true = sum(change["transition"] == "false_to_true" for change in changed_predicates)
    num_true_to_false = sum(change["transition"] == "true_to_false" for change in changed_predicates)
    task_name = _task_name_from_demo_path(demo_path)
    return {
        "task_suite_name": task_suite_name,
        "task_name": task_name,
        "episode_index": _episode_index_from_demo_key(demo_key),
        "demo_key": demo_key,
        "source_demo_file": str(demo_path),
        "transition_filter": transition_filter,
        "num_changed_predicates": len(changed_predicates),
        "num_false_to_true": num_false_to_true,
        "num_true_to_false": num_true_to_false,
        "changed_predicates": _serialize_changed_predicates(changed_predicates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan all local demos in a LIBERO task-suite directory and report the episodes whose "
            "first-vs-last predicate diff changes left-of / right-of / behind / in-front-of."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="third_party/libero/libero/datasets/libero_spatial",
        help="Directory containing task-level `*_demo.hdf5` files.",
    )
    parser.add_argument(
        "--task-suite-name",
        default=None,
        help="Override the task-suite name recorded in the output. Defaults to the dataset directory name.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/libero_spatial_relation_change_demos.json",
        help="Where to write the scan results JSON.",
    )
    parser.add_argument(
        "--only-consider-obj-of-interest",
        action="store_true",
        help="Restrict candidate argument sweeps to the BDDL obj_of_interest set.",
    )
    parser.add_argument(
        "--include-self-relations",
        action="store_true",
        help="Include binary predicate checks where both arguments are the same instance.",
    )
    parser.add_argument(
        "--transition",
        choices=TRANSITION_FILTER_VALUES,
        default="any",
        help="Which truth-value transition to count as a hit.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first demo that fails to scan instead of recording the error and continuing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    task_suite_name = args.task_suite_name or dataset_dir.name
    demo_paths = sorted(path for path in dataset_dir.glob("*_demo.hdf5") if path.is_file())
    if not demo_paths:
        raise FileNotFoundError(f"No `*_demo.hdf5` files found under {dataset_dir}")

    matches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    num_episodes_scanned = 0

    for demo_path in demo_paths:
        demo_keys = _iter_demo_keys(demo_path)
        for demo_key in demo_keys:
            num_episodes_scanned += 1
            print(f"Scanning {demo_path.name} {demo_key}", flush=True)
            try:
                match = _scan_demo(
                    demo_path=demo_path.resolve(),
                    demo_key=demo_key,
                    task_suite_name=task_suite_name,
                    only_consider_obj_of_interest=args.only_consider_obj_of_interest,
                    include_self_relations=args.include_self_relations,
                    transition_filter=args.transition,
                )
            except Exception as exc:
                error_record = {
                    "task_suite_name": task_suite_name,
                    "task_name": _task_name_from_demo_path(demo_path),
                    "episode_index": _episode_index_from_demo_key(demo_key),
                    "demo_key": demo_key,
                    "source_demo_file": str(demo_path.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if args.fail_fast:
                    raise RuntimeError(json.dumps(error_record, indent=2)) from exc
                errors.append(error_record)
                print(
                    "  error: "
                    f"{error_record['error_type']}: {error_record['error']}",
                    flush=True,
                )
                continue

            if match is None:
                continue
            matches.append(match)
            changed_predicate_names = sorted({change["predicate_name"] for change in match["changed_predicates"]})
            print(
                "  match: "
                f"episode_index={match['episode_index']} "
                f"predicates={','.join(changed_predicate_names)} "
                f"count={match['num_changed_predicates']}",
                flush=True,
            )

    results = {
        "task_suite_name": task_suite_name,
        "dataset_dir": str(dataset_dir),
        "predicates": list(TARGET_PREDICATES),
        "transition_filter": args.transition,
        "only_consider_obj_of_interest": bool(args.only_consider_obj_of_interest),
        "include_self_relations": bool(args.include_self_relations),
        "summary": {
            "num_tasks_scanned": len(demo_paths),
            "num_episodes_scanned": num_episodes_scanned,
            "num_matching_episodes": len(matches),
            "num_errors": len(errors),
        },
        "matches": matches,
        "errors": errors,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print()
    print(f"Wrote results to {output_path}")
    print(
        "Summary: "
        f"tasks={results['summary']['num_tasks_scanned']} "
        f"episodes={results['summary']['num_episodes_scanned']} "
        f"matches={results['summary']['num_matching_episodes']} "
        f"errors={results['summary']['num_errors']}"
    )


if __name__ == "__main__":
    main()
