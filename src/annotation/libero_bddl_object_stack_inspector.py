"""Inspect which Python classes implement objects referenced by one or more LIBERO BDDLs.

This tool answers questions like:
- what Python class implements `akita_black_bowl`?
- what wrapper type do predicates actually receive at runtime?
- which region names become `TargetZone` vs `SiteObject` and then `SiteObjectState`?

It does not need to run a rollout, but it does import LIBERO / robosuite class
definitions, so use the LIBERO Python 3.10 environment.

Single-file usage:
    PYTHONPATH=src:third_party/libero \\
      /home/christopher/Documents/openpi-finetune/openpi/examples/libero/.venv/bin/python \\
      -m annotation.libero_bddl_object_stack_inspector \\
      --bddl-path third_party/libero/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl

Directory usage:
    PYTHONPATH=src:third_party/libero \\
      /home/christopher/Documents/openpi-finetune/openpi/examples/libero/.venv/bin/python \\
      -m annotation.libero_bddl_object_stack_inspector \\
      --bddl-path third_party/libero/libero/libero/bddl_files/libero_spatial

JSON output:
    PYTHONPATH=src:third_party/libero \\
      /home/christopher/Documents/openpi-finetune/openpi/examples/libero/.venv/bin/python \\
      -m annotation.libero_bddl_object_stack_inspector \\
      --bddl-path third_party/libero/libero/libero/bddl_files/libero_spatial \\
      --output-json outputs/libero_bddl_object_stack.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_libero_inspector_modules() -> tuple[Any, Any, Any, Any]:
    repo_root = _repo_root()
    libero_root = repo_root / "third_party/libero"
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))

    from annotation.libero_demo_replay import _ensure_local_libero_config

    _ensure_local_libero_config(repo_root)

    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from libero.libero.envs.object_states.base_object_states import ObjectState
    from libero.libero.envs.object_states.base_object_states import SiteObjectState
    from libero.libero.envs.objects import get_object_fn

    return robosuite_parse_problem, get_object_fn, ObjectState, SiteObjectState


def _qualname(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _class_stack(cls: type[Any]) -> list[str]:
    stack = []
    for base in cls.__mro__:
        if base is object:
            break
        stack.append(_qualname(base))
    return stack


def _class_geom_metadata(implementation_class: type[Any]) -> dict[str, int] | None:
    try:
        obj = implementation_class(name="inspector_tmp")
    except Exception:
        return None

    if not hasattr(obj, "worldbody") or obj.worldbody is None:
        return None

    return {
        "geom_count": len(obj.worldbody.findall(".//geom")),
        "body_count": len(obj.worldbody.findall(".//body")),
    }


def _instance_names(problem: dict[str, Any], category_name: str, role: str) -> list[str]:
    source = problem["objects"] if role == "object" else problem["fixtures"]
    return list(source.get(category_name, []))


def _built_in_fixture_note(category_name: str) -> str | None:
    if category_name == "table":
        return "Built-in arena fixture handled by the problem class, not OBJECTS_DICT."
    return None


def _object_category_entries(problem: dict[str, Any], get_object_fn: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for role in ("object", "fixture"):
        category_map = problem["objects"] if role == "object" else problem["fixtures"]
        for category_name in sorted(category_map):
            instances = _instance_names(problem, category_name, role)
            built_in_note = _built_in_fixture_note(category_name)
            if built_in_note is not None:
                entries.append(
                    {
                        "category_name": category_name,
                        "role": role,
                        "instances": instances,
                        "implementation_class": None,
                        "class_stack": [],
                        "runtime_wrapper_stack": [
                            f"BDDL instance name(s): {', '.join(instances)}",
                            built_in_note,
                            "Not wrapped as an ObjectState entry in the same way as regular movable objects.",
                        ],
                    }
                )
                continue

            implementation_class = get_object_fn(category_name)
            geom_metadata = _class_geom_metadata(implementation_class)
            entries.append(
                {
                    "category_name": category_name,
                    "role": role,
                    "instances": instances,
                    "implementation_class": _qualname(implementation_class),
                    "class_stack": _class_stack(implementation_class),
                    "geom_count": None if geom_metadata is None else geom_metadata["geom_count"],
                    "body_count": None if geom_metadata is None else geom_metadata["body_count"],
                    "runtime_wrapper_stack": [
                        f"BDDL instance name(s): {', '.join(instances)}",
                        (
                            "Instantiated by the problem class via "
                            f"get_object_fn('{category_name}')(name='<instance>')"
                        ),
                        (
                            "Stored in "
                            f"env.{'objects_dict' if role == 'object' else 'fixtures_dict'}['<instance>']"
                        ),
                        (
                            "Wrapped as "
                            f"ObjectState(env, '<instance>'{', is_fixture=True' if role == 'fixture' else ''})"
                        ),
                        "Predicate functions receive that ObjectState wrapper",
                    ],
                }
            )
    return entries


def _region_entries(problem: dict[str, Any], site_object_state_cls: type[Any]) -> list[dict[str, Any]]:
    object_instance_names = {
        instance_name
        for category_instances in problem["objects"].values()
        for instance_name in category_instances
    }
    fixture_instance_names = {
        instance_name
        for category_instances in problem["fixtures"].values()
        for instance_name in category_instances
    }
    built_in_fixture_instances = {
        instance_name
        for category_name, category_instances in problem["fixtures"].items()
        if _built_in_fixture_note(category_name) is not None
        for instance_name in category_instances
    }

    entries: list[dict[str, Any]] = []
    for region_name in sorted(problem["regions"]):
        region_spec = problem["regions"][region_name]
        target_name = region_spec["target"]
        if target_name in built_in_fixture_instances:
            implementation_kind = "libero.libero.envs.objects.target_zones.TargetZone"
            implementation_note = "Region is materialized as a workspace target zone."
        elif target_name in object_instance_names or target_name in fixture_instance_names:
            implementation_kind = "libero.libero.envs.objects.site_object.SiteObject"
            implementation_note = "Region is attached to an object / fixture site."
        else:
            implementation_kind = "libero.libero.envs.objects.target_zones.TargetZone"
            implementation_note = "Region is materialized as a workspace target zone."

        entries.append(
            {
                "region_name": region_name,
                "target_name": target_name,
                "implementation_class": implementation_kind,
                "wrapper_class": _qualname(site_object_state_cls),
                "runtime_wrapper_stack": [
                    f"BDDL region name: {region_name}",
                    implementation_note,
                    f"Stored in env.object_sites_dict['{region_name}']",
                    f"Wrapped as SiteObjectState(env, '{region_name}', parent_name='<target>')",
                    "Predicates whose second argument is a region receive that SiteObjectState wrapper",
                ],
            }
        )
    return entries


def build_bddl_object_implementation_report(bddl_file: str | Path) -> dict[str, Any]:
    robosuite_parse_problem, get_object_fn, object_state_cls, site_object_state_cls = _load_libero_inspector_modules()

    bddl_path = Path(bddl_file).expanduser().resolve()
    problem = robosuite_parse_problem(str(bddl_path))
    object_entries = _object_category_entries(problem, get_object_fn)
    region_entries = _region_entries(problem, site_object_state_cls)
    language_instruction = problem["language_instruction"]
    if isinstance(language_instruction, list):
        language_instruction = " ".join(language_instruction)

    return {
        "bddl_file": str(bddl_path),
        "problem_name": problem["problem_name"],
        "language_instruction": language_instruction,
        "object_state_wrapper_class": _qualname(object_state_cls),
        "site_object_state_wrapper_class": _qualname(site_object_state_cls),
        "object_categories": object_entries,
        "regions": region_entries,
        "notes": {
            "predicate_dispatch": (
                "Goal predicates are evaluated by problem classes like "
                "libero_tabletop_manipulation.py, which call eval_predicate_fn(...) "
                "on ObjectState / SiteObjectState wrappers."
            ),
            "object_construction": (
                "Concrete object implementations come from OBJECTS_DICT via "
                "libero.libero.envs.objects.get_object_fn(category_name)."
            ),
        },
    }


def _resolve_bddl_paths(bddl_path: str | Path) -> list[Path]:
    path = Path(bddl_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".bddl":
            raise ValueError(f"Expected a `.bddl` file, got `{path}`.")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"BDDL path does not exist: {path}")

    paths = sorted(candidate.resolve() for candidate in path.rglob("*.bddl") if candidate.is_file())
    if not paths:
        raise FileNotFoundError(f"No `.bddl` files found under {path}")
    return paths


def build_multi_bddl_object_implementation_report(bddl_path: str | Path) -> dict[str, Any]:
    bddl_paths = _resolve_bddl_paths(bddl_path)
    single_reports = [build_bddl_object_implementation_report(path) for path in bddl_paths]

    object_entries_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    region_entries_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    problems_to_files: dict[str, list[str]] = collections.defaultdict(list)

    for report in single_reports:
        problems_to_files[report["problem_name"]].append(report["bddl_file"])

        for entry in report["object_categories"]:
            key = (entry["role"], entry["category_name"])
            existing = object_entries_by_key.get(key)
            if existing is None:
                object_entries_by_key[key] = {
                    **entry,
                    "instances": sorted(set(entry["instances"])),
                    "source_bddl_files": [report["bddl_file"]],
                    "problem_names": [report["problem_name"]],
                }
                continue

            if existing["implementation_class"] != entry["implementation_class"]:
                raise ValueError(
                    "Object category resolved inconsistently across BDDL files: "
                    f"{entry['category_name']} ({entry['role']}) -> "
                    f"{existing['implementation_class']} vs {entry['implementation_class']}"
                )

            existing["instances"] = sorted(set(existing["instances"]) | set(entry["instances"]))
            existing["source_bddl_files"] = sorted(set(existing["source_bddl_files"]) | {report["bddl_file"]})
            existing["problem_names"] = sorted(set(existing["problem_names"]) | {report["problem_name"]})

        for entry in report["regions"]:
            key = (entry["target_name"], entry["implementation_class"])
            existing = region_entries_by_key.get(key)
            if existing is None:
                region_entries_by_key[key] = {
                    **entry,
                    "region_names": [entry["region_name"]],
                    "source_bddl_files": [report["bddl_file"]],
                    "problem_names": [report["problem_name"]],
                }
                continue

            existing["region_names"] = sorted(set(existing["region_names"]) | {entry["region_name"]})
            existing["source_bddl_files"] = sorted(set(existing["source_bddl_files"]) | {report["bddl_file"]})
            existing["problem_names"] = sorted(set(existing["problem_names"]) | {report["problem_name"]})

    return {
        "bddl_input_path": str(Path(bddl_path).expanduser().resolve()),
        "num_bddl_files": len(bddl_paths),
        "bddl_files": [str(path) for path in bddl_paths],
        "problem_names": sorted(problems_to_files),
        "problem_to_bddl_files": {problem: sorted(paths) for problem, paths in sorted(problems_to_files.items())},
        "object_state_wrapper_class": single_reports[0]["object_state_wrapper_class"],
        "site_object_state_wrapper_class": single_reports[0]["site_object_state_wrapper_class"],
        "object_categories": sorted(
            object_entries_by_key.values(),
            key=lambda entry: (entry["role"], entry["category_name"]),
        ),
        "regions": sorted(
            region_entries_by_key.values(),
            key=lambda entry: (entry["target_name"], entry["implementation_class"]),
        ),
        "notes": {
            "problem_dependence": (
                "Object category -> implementation class resolution is global via OBJECTS_DICT and usually "
                "does not depend on the problem class. Problem classes matter more for built-in fixtures, "
                "workspace regions, and runtime instantiation context."
            ),
            "source_reports": "This aggregate report is built by parsing each BDDL separately and merging categories.",
        },
    }


def _print_object_entry(entry: dict[str, Any]) -> None:
    print(f"- {entry['category_name']} ({entry['role']})")
    print(f"  instances: {', '.join(entry['instances'])}")
    print(f"  implementation class: {entry['implementation_class']}")
    if entry.get("geom_count") is not None:
        print(f"  geom count: {entry['geom_count']}")
    if entry.get("body_count") is not None:
        print(f"  body count: {entry['body_count']}")
    if "problem_names" in entry:
        print(f"  problems: {', '.join(entry['problem_names'])}")
    if entry["class_stack"]:
        print("  class stack:")
        for class_name in entry["class_stack"]:
            print(f"    - {class_name}")
    print("  runtime wrapper stack:")
    for line in entry["runtime_wrapper_stack"]:
        print(f"    - {line}")
    if "source_bddl_files" in entry:
        print(f"  source BDDL files: {len(entry['source_bddl_files'])}")


def _print_region_entry(entry: dict[str, Any]) -> None:
    region_label = entry.get("region_name", ", ".join(entry.get("region_names", [])))
    print(f"- {region_label}")
    print(f"  target: {entry['target_name']}")
    print(f"  implementation class: {entry['implementation_class']}")
    print(f"  wrapper class: {entry['wrapper_class']}")
    if "problem_names" in entry:
        print(f"  problems: {', '.join(entry['problem_names'])}")
    print("  runtime wrapper stack:")
    for line in entry["runtime_wrapper_stack"]:
        print(f"    - {line}")
    if "source_bddl_files" in entry:
        print(f"  source BDDL files: {len(entry['source_bddl_files'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect how one LIBERO BDDL or a directory of BDDLs maps category "
            "names like `akita_black_bowl` to concrete Python classes and runtime "
            "predicate wrappers."
        )
    )
    parser.add_argument("--bddl-path", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    input_path = Path(args.bddl_path).expanduser().resolve()
    if input_path.is_dir():
        report = build_multi_bddl_object_implementation_report(input_path)
        print(f"BDDL root: {report['bddl_input_path']}")
        print(f"Files scanned: {report['num_bddl_files']}")
        print(f"Problems: {', '.join(report['problem_names'])}")
    else:
        report = build_bddl_object_implementation_report(input_path)
        print(f"BDDL: {report['bddl_file']}")
        print(f"Problem: {report['problem_name']}")
        print(f"Language: {report['language_instruction']}")

    print()
    print("Predicate wrapper classes")
    print(f"  Object / fixture wrapper: {report['object_state_wrapper_class']}")
    print(f"  Region wrapper: {report['site_object_state_wrapper_class']}")
    print()
    print("Object categories")
    for entry in report["object_categories"]:
        _print_object_entry(entry)
    print()
    print("Region-backed predicate targets")
    for entry in report["regions"]:
        _print_region_entry(entry)

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print()
        print(f"Wrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
