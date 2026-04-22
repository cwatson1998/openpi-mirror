"""Evaluate LIBERO predicates against saved demo timesteps and derive annotations.

This module builds on `annotation.libero_demo_replay`:

- resolve an RLDS episode or source HDF5 demo back to a replay spec
- restore an arbitrary saved MuJoCo state into a LIBERO simulator
- evaluate a predicate with concrete scene-instance arguments at that timestep
- enumerate goal-centered annotations by checking which predicates hold

Typical usage:

    PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
      -m annotation.libero_predicate_annotation evaluate \
      --dataset-name libero_spatial_no_noops \
      --data-dir data/libero/raw \
      --episode-index 0 \
      --predicate on \
      --predicate-args porcelain_mug_1 plate_1

    PYTHONPATH=src:third_party/libero examples/libero/.venv/bin/python \
      -m annotation.libero_predicate_annotation annotate-goal-final-state \
      --dataset-name libero_spatial_no_noops \
      --data-dir data/libero/raw \
      --episode-index 0 \
      --output-json outputs/libero_predicate_annotations/ep0.json
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
import dataclasses
import inspect
import json
from pathlib import Path
from typing import Any

from annotation.libero_demo_replay import DemoReplaySpec
from annotation.libero_demo_replay import _load_libero_modules
from annotation.libero_demo_replay import _postprocess_demo_model_xml
from annotation.libero_demo_replay import resolve_demo_replay_spec

DEFAULT_ANNOTATION_PREDICATE_EXCLUSIONS = frozenset(
    {
        "false",
        "printjointstate",
        "true",
    }
)


@dataclasses.dataclass(frozen=True)
class PredicateEvaluation:
    predicate_name: str
    arguments: tuple[str, ...]
    argument_kinds: tuple[str, ...]
    timestep_index: int
    value: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_name": self.predicate_name,
            "arguments": list(self.arguments),
            "argument_kinds": list(self.argument_kinds),
            "timestep_index": self.timestep_index,
            "value": self.value,
        }


@dataclasses.dataclass(frozen=True)
class PredicateTruthValueChange:
    focus_object_name: str
    focus_object_kind: str
    predicate_name: str
    arguments: tuple[str, ...]
    argument_kinds: tuple[str, ...]
    first_timestep_index: int
    last_timestep_index: int
    first_value: bool
    last_value: bool

    def transition_name(self) -> str:
        if not self.first_value and self.last_value:
            return "false_to_true"
        if self.first_value and not self.last_value:
            return "true_to_false"
        return "unchanged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_object_name": self.focus_object_name,
            "focus_object_kind": self.focus_object_kind,
            "predicate_name": self.predicate_name,
            "arguments": list(self.arguments),
            "argument_kinds": list(self.argument_kinds),
            "first_timestep_index": self.first_timestep_index,
            "last_timestep_index": self.last_timestep_index,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "transition": self.transition_name(),
        }


def extract_goal_argument_names(goal_state: Sequence[Sequence[str]]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered_names: list[str] = []
    for state in goal_state:
        for argument_name in state[1:]:
            if argument_name in seen:
                continue
            seen.add(argument_name)
            ordered_names.append(argument_name)
    return tuple(ordered_names)


def predicate_callable_arity(predicate_fn: Any) -> int | None:
    signature = inspect.signature(predicate_fn)
    arity = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return None
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            arity += 1
    return arity


def group_predicates_by_arity(
    predicate_fns: dict[str, Any],
    *,
    exclude_names: Iterable[str] = DEFAULT_ANNOTATION_PREDICATE_EXCLUSIONS,
) -> dict[int, tuple[str, ...]]:
    excluded = {name.lower() for name in exclude_names}
    grouped: dict[int, list[str]] = {}
    for predicate_name, predicate_fn in sorted(predicate_fns.items()):
        canonical_name = predicate_name.lower()
        if canonical_name in excluded:
            continue
        arity = predicate_callable_arity(predicate_fn)
        if arity is None:
            continue
        grouped.setdefault(arity, []).append(canonical_name)
    return {arity: tuple(names) for arity, names in grouped.items()}


def iter_focus_object_predicate_calls(
    focus_object_names: Sequence[str],
    scene_instance_names: Sequence[str],
    *,
    unary_predicates: Sequence[str],
    binary_predicates: Sequence[str],
    include_self_relations: bool = False,
    include_reverse_binary: bool = True,
) -> Iterator[tuple[str, str, tuple[str, ...]]]:
    for focus_object_name in focus_object_names:
        for predicate_name in unary_predicates:
            yield focus_object_name, predicate_name, (focus_object_name,)

        for predicate_name in binary_predicates:
            for other_object_name in scene_instance_names:
                if not include_self_relations and other_object_name == focus_object_name:
                    continue
                yield focus_object_name, predicate_name, (focus_object_name, other_object_name)
                if include_reverse_binary:
                    yield focus_object_name, predicate_name, (other_object_name, focus_object_name)


def normalize_timestep_index(timestep_index: int, num_states: int) -> int:
    if num_states <= 0:
        raise ValueError("Cannot evaluate predicates because the demo contains no saved states.")
    normalized_index = timestep_index
    if normalized_index < 0:
        normalized_index += num_states
    if normalized_index < 0 or normalized_index >= num_states:
        raise IndexError(
            f"Timestep index {timestep_index} is out of range for {num_states} saved states."
        )
    return normalized_index


def predicate_evaluation_key(
    predicate_name: str,
    arguments: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    return predicate_name.lower(), tuple(arguments)


def build_truth_value_change_report(
    first_snapshot: dict[str, Any],
    last_snapshot: dict[str, Any],
) -> dict[str, Any]:
    first_annotations = {
        annotation["focus_object_name"]: annotation
        for annotation in first_snapshot["annotations"]
    }
    last_annotations = {
        annotation["focus_object_name"]: annotation
        for annotation in last_snapshot["annotations"]
    }

    changed_predicates: list[dict[str, Any]] = []
    incomparable_examples: list[dict[str, Any]] = []
    comparable_checks = 0
    incomparable_checks = 0

    for focus_object_name in sorted(first_annotations.keys() | last_annotations.keys()):
        first_annotation = first_annotations.get(focus_object_name)
        last_annotation = last_annotations.get(focus_object_name)
        if first_annotation is None or last_annotation is None:
            incomparable_checks += 1
            if len(incomparable_examples) < 20:
                incomparable_examples.append(
                    {
                        "focus_object_name": focus_object_name,
                        "reason": "focus_object_missing_from_one_snapshot",
                    }
                )
            continue

        first_evaluations = {
            predicate_evaluation_key(
                evaluation["predicate_name"],
                evaluation["arguments"],
            ): evaluation
            for evaluation in first_annotation["predicate_evaluations"]
        }
        last_evaluations = {
            predicate_evaluation_key(
                evaluation["predicate_name"],
                evaluation["arguments"],
            ): evaluation
            for evaluation in last_annotation["predicate_evaluations"]
        }

        comparable_keys = first_evaluations.keys() & last_evaluations.keys()
        comparable_checks += len(comparable_keys)
        for evaluation_key in sorted(comparable_keys):
            first_evaluation = first_evaluations[evaluation_key]
            last_evaluation = last_evaluations[evaluation_key]
            if first_evaluation["value"] == last_evaluation["value"]:
                continue

            changed_predicates.append(
                PredicateTruthValueChange(
                    focus_object_name=focus_object_name,
                    focus_object_kind=first_annotation["focus_object_kind"],
                    predicate_name=first_evaluation["predicate_name"],
                    arguments=tuple(first_evaluation["arguments"]),
                    argument_kinds=tuple(first_evaluation["argument_kinds"]),
                    first_timestep_index=first_evaluation["timestep_index"],
                    last_timestep_index=last_evaluation["timestep_index"],
                    first_value=bool(first_evaluation["value"]),
                    last_value=bool(last_evaluation["value"]),
                ).to_dict()
            )

        incomparable_keys = first_evaluations.keys() ^ last_evaluations.keys()
        incomparable_checks += len(incomparable_keys)
        for evaluation_key in sorted(incomparable_keys):
            if len(incomparable_examples) >= 20:
                break
            predicate_name, arguments = evaluation_key
            incomparable_examples.append(
                {
                    "focus_object_name": focus_object_name,
                    "predicate_name": predicate_name,
                    "arguments": list(arguments),
                    "reason": "evaluation_missing_from_one_snapshot",
                }
            )

    false_to_true = [change for change in changed_predicates if change["transition"] == "false_to_true"]
    true_to_false = [change for change in changed_predicates if change["transition"] == "true_to_false"]

    return {
        "changed_predicates": changed_predicates,
        "false_to_true": false_to_true,
        "true_to_false": true_to_false,
        "incomparable_examples": incomparable_examples,
        "summary": {
            "num_changed_predicates": len(changed_predicates),
            "num_false_to_true": len(false_to_true),
            "num_true_to_false": len(true_to_false),
            "num_comparable_checks": comparable_checks,
            "num_incomparable_checks": incomparable_checks,
        },
    }


def _load_predicate_registry() -> tuple[Any, dict[str, Any]]:
    _load_libero_modules()
    from libero.libero.envs.predicates import get_predicate_fn
    from libero.libero.envs.predicates import get_predicate_fn_dict

    predicate_fns = {
        name.lower(): predicate_fn
        for name, predicate_fn in get_predicate_fn_dict().items()
    }
    return get_predicate_fn, predicate_fns


class LiberoPredicateEvaluator:
    def __init__(
        self,
        *,
        spec: DemoReplaySpec,
        env: Any,
        predicate_lookup: Any,
        predicate_fns: dict[str, Any],
    ) -> None:
        self.spec = spec
        self.env = env
        self._predicate_lookup = predicate_lookup
        self.predicate_fns = predicate_fns
        self.predicate_names_by_arity = group_predicates_by_arity(predicate_fns)
        self._restored_timestep_index: int | None = None

    @classmethod
    def from_spec(
        cls,
        spec: DemoReplaySpec,
        *,
        camera_name: str = "agentview",
        camera_height: int = 4,
        camera_width: int = 4,
    ) -> LiberoPredicateEvaluator:
        offscreen_render_env_cls, _, libero_postprocess_model_xml = _load_libero_modules()
        predicate_lookup, predicate_fns = _load_predicate_registry()
        env = offscreen_render_env_cls(
            bddl_file_name=spec.bddl_file_name,
            use_camera_obs=False,
            camera_names=[camera_name],
            camera_heights=camera_height,
            camera_widths=camera_width,
        )
        env.reset()
        env.reset_from_xml_string(_postprocess_demo_model_xml(spec.model_xml, libero_postprocess_model_xml))
        env.sim.reset()
        return cls(
            spec=spec,
            env=env,
            predicate_lookup=predicate_lookup,
            predicate_fns=predicate_fns,
        )

    def __enter__(self) -> LiberoPredicateEvaluator:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def parsed_problem(self) -> dict[str, Any]:
        return self.env.env.parsed_problem

    @property
    def scene_instance_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.env.env.object_states_dict))

    @property
    def goal_argument_names(self) -> tuple[str, ...]:
        return extract_goal_argument_names(self.parsed_problem["goal_state"])

    def close(self) -> None:
        self.env.close()

    def describe_instance_kind(self, instance_name: str) -> str:
        if instance_name in self.env.env.objects_dict:
            return "object"
        if instance_name in self.env.env.fixtures_dict:
            return "fixture"
        if instance_name in self.env.env.object_sites_dict:
            return "site"
        return "unknown"

    def restore_timestep(self, timestep_index: int) -> int:
        normalized_index = normalize_timestep_index(timestep_index, len(self.spec.states))
        if normalized_index == self._restored_timestep_index:
            return normalized_index

        self.env.regenerate_obs_from_state(self.spec.states[normalized_index])
        self._restored_timestep_index = normalized_index
        return normalized_index

    def _lookup_object_states(self, argument_names: Sequence[str]) -> tuple[Any, ...]:
        object_states_dict = self.env.env.object_states_dict
        missing = [name for name in argument_names if name not in object_states_dict]
        if missing:
            available_preview = ", ".join(sorted(object_states_dict)[:20])
            raise KeyError(
                "Unknown scene instance(s): "
                f"{missing}. Available instances include: {available_preview}"
            )
        return tuple(object_states_dict[name] for name in argument_names)

    def evaluate(
        self,
        predicate_name: str,
        argument_names: Sequence[str],
        *,
        timestep_index: int = -1,
    ) -> PredicateEvaluation:
        canonical_predicate_name = predicate_name.lower()
        predicate_fn = self._predicate_lookup(canonical_predicate_name)
        arity = predicate_callable_arity(predicate_fn)
        if arity is None:
            raise ValueError(
                f"Predicate `{predicate_name}` does not have a fixed arity and cannot be evaluated via this tool."
            )
        if len(argument_names) != arity:
            raise ValueError(
                f"Predicate `{predicate_name}` expects {arity} argument(s), got {len(argument_names)}."
            )

        normalized_index = self.restore_timestep(timestep_index)
        try:
            value = self._evaluate_restored_predicate(
                predicate_fn,
                argument_names,
            )
        except Exception as exc:
            joined_args = ", ".join(argument_names)
            raise RuntimeError(
                f"Failed to evaluate `{predicate_name}({joined_args})` at timestep {normalized_index}."
            ) from exc

        return PredicateEvaluation(
            predicate_name=canonical_predicate_name,
            arguments=tuple(argument_names),
            argument_kinds=tuple(self.describe_instance_kind(name) for name in argument_names),
            timestep_index=normalized_index,
            value=value,
        )

    def _evaluate_restored_predicate(
        self,
        predicate_fn: Any,
        argument_names: Sequence[str],
    ) -> bool:
        object_states = self._lookup_object_states(argument_names)
        return bool(predicate_fn(*object_states))

    def _resolve_selected_predicates(
        self,
        predicate_names: Sequence[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if predicate_names is None:
            predicate_names_by_arity = self.predicate_names_by_arity
        else:
            selected_names = {name.lower() for name in predicate_names}
            predicate_names_by_arity = {
                arity: tuple(name for name in names if name in selected_names)
                for arity, names in self.predicate_names_by_arity.items()
            }
        return (
            predicate_names_by_arity.get(1, ()),
            predicate_names_by_arity.get(2, ()),
        )

    def _annotation_context(
        self,
        *,
        unary_predicates: Sequence[str],
        binary_predicates: Sequence[str],
    ) -> dict[str, Any]:
        scene_instance_names = self.scene_instance_names
        goal_argument_names = self.goal_argument_names
        return {
            "demo": {
                "demo_hdf5_path": str(self.spec.demo_hdf5_path),
                "demo_key": self.spec.demo_key,
                "bddl_file_name": self.spec.bddl_file_name,
                "task_instruction": self.spec.task_instruction,
                "source_demo_path_hint": self.spec.source_demo_path_hint,
                "matching_summary": self.spec.matching_summary,
                "num_saved_states": int(self.spec.states.shape[0]),
            },
            "goal_state": [list(state) for state in self.parsed_problem["goal_state"]],
            "goal_argument_names": list(goal_argument_names),
            "missing_goal_arguments": [
                name for name in goal_argument_names if name not in self.env.env.object_states_dict
            ],
            "scene_instances": [
                {
                    "name": name,
                    "kind": self.describe_instance_kind(name),
                }
                for name in scene_instance_names
            ],
            "predicate_inventory": {
                "unary": list(unary_predicates),
                "binary": list(binary_predicates),
            },
        }

    def _build_goal_annotation_snapshot(
        self,
        *,
        timestep_index: int = -1,
        unary_predicates: Sequence[str],
        binary_predicates: Sequence[str],
        include_self_relations: bool = False,
        max_skip_examples: int = 20,
    ) -> dict[str, Any]:
        normalized_index = self.restore_timestep(timestep_index)

        scene_instance_names = self.scene_instance_names
        goal_argument_names = self.goal_argument_names

        goal_state_evaluations = [
            self.evaluate(state[0], state[1:], timestep_index=normalized_index).to_dict()
            for state in self.parsed_problem["goal_state"]
        ]

        annotations = []
        total_checks = 0
        total_true = 0
        total_skipped = 0
        skipped_examples: list[dict[str, Any]] = []

        for focus_object_name in goal_argument_names:
            if focus_object_name not in self.env.env.object_states_dict:
                continue

            predicate_evaluations: list[dict[str, Any]] = []
            true_ground_predicates: list[dict[str, Any]] = []
            object_checks = 0
            object_skips = 0

            for _, predicate_name, arguments in iter_focus_object_predicate_calls(
                [focus_object_name],
                scene_instance_names,
                unary_predicates=unary_predicates,
                binary_predicates=binary_predicates,
                include_self_relations=include_self_relations,
            ):
                object_checks += 1
                total_checks += 1
                try:
                    predicate_fn = self._predicate_lookup(predicate_name)
                    value = self._evaluate_restored_predicate(
                        predicate_fn,
                        arguments,
                    )
                except Exception as exc:
                    object_skips += 1
                    total_skipped += 1
                    if len(skipped_examples) < max_skip_examples:
                        skipped_examples.append(
                            {
                                "focus_object_name": focus_object_name,
                                "predicate_name": predicate_name,
                                "arguments": list(arguments),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    continue

                evaluation = PredicateEvaluation(
                    predicate_name=predicate_name,
                    arguments=tuple(arguments),
                    argument_kinds=tuple(self.describe_instance_kind(name) for name in arguments),
                    timestep_index=normalized_index,
                    value=value,
                ).to_dict()
                predicate_evaluations.append(evaluation)
                if value:
                    true_ground_predicates.append(evaluation)
                    total_true += 1

            annotations.append(
                {
                    "focus_object_name": focus_object_name,
                    "focus_object_kind": self.describe_instance_kind(focus_object_name),
                    "num_candidate_checks": object_checks,
                    "num_successful_evaluations": len(predicate_evaluations),
                    "num_true_ground_predicates": len(true_ground_predicates),
                    "num_skipped_checks": object_skips,
                    "predicate_evaluations": predicate_evaluations,
                    "true_ground_predicates": true_ground_predicates,
                }
            )

        return {
            "annotation_timestep_index": normalized_index,
            "goal_state_evaluations": goal_state_evaluations,
            "all_goal_predicates_true": all(result["value"] for result in goal_state_evaluations),
            "annotations": annotations,
            "summary": {
                "num_focus_objects": len(annotations),
                "num_scene_instances": len(scene_instance_names),
                "num_candidate_checks": total_checks,
                "num_true_ground_predicates": total_true,
                "num_skipped_checks": total_skipped,
            },
            "skipped_examples": skipped_examples,
        }

    def annotate_goal_objects(
        self,
        *,
        timestep_index: int = -1,
        include_self_relations: bool = False,
        predicate_names: Sequence[str] | None = None,
        max_skip_examples: int = 20,
    ) -> dict[str, Any]:
        unary_predicates, binary_predicates = self._resolve_selected_predicates(predicate_names)
        report = self._annotation_context(
            unary_predicates=unary_predicates,
            binary_predicates=binary_predicates,
        )
        report.update(
            self._build_goal_annotation_snapshot(
                timestep_index=timestep_index,
                unary_predicates=unary_predicates,
                binary_predicates=binary_predicates,
                include_self_relations=include_self_relations,
                max_skip_examples=max_skip_examples,
            )
        )
        return report

    def compare_goal_object_predicates_between_timesteps(
        self,
        *,
        first_timestep_index: int = 0,
        last_timestep_index: int = -1,
        include_self_relations: bool = False,
        predicate_names: Sequence[str] | None = None,
        max_skip_examples: int = 20,
    ) -> dict[str, Any]:
        unary_predicates, binary_predicates = self._resolve_selected_predicates(predicate_names)
        report = self._annotation_context(
            unary_predicates=unary_predicates,
            binary_predicates=binary_predicates,
        )
        first_snapshot = self._build_goal_annotation_snapshot(
            timestep_index=first_timestep_index,
            unary_predicates=unary_predicates,
            binary_predicates=binary_predicates,
            include_self_relations=include_self_relations,
            max_skip_examples=max_skip_examples,
        )
        last_snapshot = self._build_goal_annotation_snapshot(
            timestep_index=last_timestep_index,
            unary_predicates=unary_predicates,
            binary_predicates=binary_predicates,
            include_self_relations=include_self_relations,
            max_skip_examples=max_skip_examples,
        )
        report["first_timestep_annotations"] = first_snapshot
        report["last_timestep_annotations"] = last_snapshot
        report["truth_value_changes"] = build_truth_value_change_report(
            first_snapshot,
            last_snapshot,
        )
        return report


def _resolve_spec_from_args(args: argparse.Namespace) -> DemoReplaySpec:
    return resolve_demo_replay_spec(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        source_demo_file=args.source_demo_file,
        demo_key=args.demo_key,
        demo_search_roots=args.demo_search_root,
    )


def _write_json_if_requested(report: dict[str, Any], output_json: str | None) -> None:
    if output_json is None:
        return
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LIBERO predicates against saved demo timesteps and derive annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_demo_selector_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
        command_parser.add_argument("--data-dir", default="data/libero/raw")
        command_parser.add_argument("--episode-index", type=int)
        command_parser.add_argument("--source-demo-file")
        command_parser.add_argument("--demo-key")
        command_parser.add_argument("--demo-search-root", action="append", default=[])
        command_parser.add_argument("--output-json")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one predicate with concrete scene-instance arguments at a chosen timestep.",
    )
    add_demo_selector_arguments(evaluate_parser)
    evaluate_parser.add_argument("--timestep-index", type=int, default=-1)
    evaluate_parser.add_argument("--predicate", required=True)
    evaluate_parser.add_argument("--predicate-args", nargs="+", required=True)

    annotate_parser = subparsers.add_parser(
        "annotate-goal-final-state",
        help="Enumerate goal-centered predicate annotations at the chosen saved timestep.",
    )
    add_demo_selector_arguments(annotate_parser)
    annotate_parser.add_argument("--timestep-index", type=int, default=-1)
    annotate_parser.add_argument("--predicate", action="append", default=[])
    annotate_parser.add_argument("--include-self-relations", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare-goal-first-last-state",
        help="Compare goal-centered predicate truth values between the first and last saved demo states.",
    )
    add_demo_selector_arguments(compare_parser)
    compare_parser.add_argument("--first-timestep-index", type=int, default=0)
    compare_parser.add_argument("--last-timestep-index", type=int, default=-1)
    compare_parser.add_argument("--predicate", action="append", default=[])
    compare_parser.add_argument("--include-self-relations", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    spec = _resolve_spec_from_args(args)

    with LiberoPredicateEvaluator.from_spec(spec) as evaluator:
        if args.command == "evaluate":
            evaluation = evaluator.evaluate(
                args.predicate,
                args.predicate_args,
                timestep_index=args.timestep_index,
            )
            report = {
                "demo": {
                    "demo_hdf5_path": str(spec.demo_hdf5_path),
                    "demo_key": spec.demo_key,
                    "bddl_file_name": spec.bddl_file_name,
                    "num_saved_states": int(spec.states.shape[0]),
                },
                "evaluation": evaluation.to_dict(),
            }
        elif args.command == "annotate-goal-final-state":
            report = evaluator.annotate_goal_objects(
                timestep_index=args.timestep_index,
                include_self_relations=args.include_self_relations,
                predicate_names=args.predicate or None,
            )
        elif args.command == "compare-goal-first-last-state":
            report = evaluator.compare_goal_object_predicates_between_timesteps(
                first_timestep_index=args.first_timestep_index,
                last_timestep_index=args.last_timestep_index,
                include_self_relations=args.include_self_relations,
                predicate_names=args.predicate or None,
            )
        else:
            raise ValueError(f"Unsupported command `{args.command}`.")

    print(json.dumps(report, indent=2))
    _write_json_if_requested(report, args.output_json)


if __name__ == "__main__":
    main()
