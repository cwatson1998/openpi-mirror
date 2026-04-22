"""Inspect how an RLDS episode resolves back to LIBERO source assets.

This tool prints and optionally saves the resolution chain for one episode:
- RLDS `episode_metadata.file_path` hint -> local source demo HDF5
- source demo HDF5 `/data/demo_x` match -> `model_file` XML attribute
- source demo HDF5 `data.attrs["bddl_file_name"]` -> local BDDL path

Use this when you forget where `model_file` or the BDDL came from, or when
debugging why a replay command picked a particular demo file.

Example usage:
    PYTHONPATH=src:third_party/libero /home/christopher/miniconda3/envs/instructvla_libero/bin/python -m annotation.libero_resolution_inspector --dataset-name libero_spatial_no_noops --data-dir data/libero/raw --episode-index 0 --demo-search-root third_party/libero/libero/datasets --output-json outputs/libero_resolution_ep0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation.libero_demo_replay import inspect_demo_replay_resolution


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect how an RLDS LIBERO episode resolves to the source demo HDF5, "
            "the per-demo model_file XML, and the local BDDL file."
        )
    )
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--data-dir", default="data/libero/raw")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--demo-search-root", action="append", default=[])
    parser.add_argument("--output-json")
    args = parser.parse_args()

    report = inspect_demo_replay_resolution(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        demo_search_roots=args.demo_search_root,
    )

    print(f"Episode: {report['dataset_name']}[{report['episode_index']}]")
    print(f"Task: {report['task_instruction']}")
    print()
    print("RLDS -> source demo HDF5")
    print(f"  hint: {report['source_demo_path_hint']}")
    print(f"  resolved: {report['source_demo_resolution']['resolved_path']}")
    print("  resolution code: src/annotation/libero_demo_replay.py::trace_demo_hdf5_resolution")
    print()
    print("Source demo group")
    print(f"  demo_key: {report['demo_group']['demo_key']}")
    print(f"  group: {report['demo_group']['hdf5_group_path']}")
    print(f"  model_file attr: {report['demo_group']['model_file_attr_path']}")
    print(f"  model_file prefix: {report['demo_group']['model_file_xml_prefix']}")
    print()
    print("HDF5 -> BDDL")
    print(f"  raw bddl_file_name: {report['bddl_resolution']['raw_bddl_file_name']}")
    print(f"  resolved: {report['bddl_resolution']['resolved_path']}")
    print("  resolution code: src/annotation/libero_demo_replay.py::trace_bddl_file_resolution")
    print()
    print("Checked BDDL candidate paths:")
    for path in report["bddl_resolution"]["checked_paths"]:
        print(f"  - {path}")
    print()
    print("Checked HDF5 candidate paths:")
    for path in report["source_demo_resolution"]["checked_paths"]:
        print(f"  - {path}")

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print()
        print(f"Wrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
