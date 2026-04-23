#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="examples/libero/.venv/bin/python"
DATASET_DIR="third_party/libero/libero/datasets/libero_spatial"
OUTPUT_ROOT="outputs/libero_spatial_representatives"
DEMO_KEY="demo_0"
ROTATE_IMAGES_180="false"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_libero_spatial_predicate_change_visualizations.sh [options]

Runs `annotation.libero_predicate_change_visualization` on one representative
demo (`demo_0`) from each local `libero_spatial` task HDF5.

Options:
  --python PATH            Python executable to use
                           (default: examples/libero/.venv/bin/python)
  --dataset-dir PATH       Source HDF5 directory
                           (default: third_party/libero/libero/datasets/libero_spatial)
  --output-root PATH       Output root for per-task visualization folders
                           (default: outputs/libero_spatial_representatives)
  --demo-key KEY           Demo key to use from each HDF5 (default: demo_0)
  --rotate-images-180      Rotate rendered images in output PNGs
  --help                   Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --demo-key)
      DEMO_KEY="$2"
      shift 2
      ;;
    --rotate-images-180)
      ROTATE_IMAGES_180="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

mapfile -t DEMO_FILES < <(find "$DATASET_DIR" -maxdepth 1 -type f -name '*_demo.hdf5' | sort)

if [[ "${#DEMO_FILES[@]}" -eq 0 ]]; then
  echo "No libero_spatial demo files found under: $DATASET_DIR" >&2
  exit 1
fi

for demo_file in "${DEMO_FILES[@]}"; do
  task_name="$(basename "$demo_file" _demo.hdf5)"
  output_dir="${OUTPUT_ROOT}/${task_name}"

  cmd=(
    "$PYTHON_BIN"
    -m annotation.libero_predicate_change_visualization
    --source-demo-file "$demo_file"
    --demo-key "$DEMO_KEY"
    --only-consider-obj-of-interest
    --output-dir "$output_dir"
  )

  if [[ "$ROTATE_IMAGES_180" == "true" ]]; then
    cmd+=(--rotate-images-180)
  fi

  printf 'Running %s\n' "$task_name"
  PYTHONPATH="src:third_party/libero" "${cmd[@]}"
done

printf 'Wrote outputs under %s\n' "$(realpath "$OUTPUT_ROOT")"
