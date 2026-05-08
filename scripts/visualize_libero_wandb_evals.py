"""Fetch LIBERO eval runs from W&B and visualize their success rates.

Example:
    .venv/bin/python scripts/visualize_libero_wandb_evals.py \
      --run ald7rqk2=vanilla-from-libero \
      --run qfr0siac=target-dot-from-libero \
      --output-dir outputs/libero_spatial_lora_eval5_wandb_viz

The script prefers W&B API data. If the API is unavailable, pass
``--allow-local-fallback`` to read local ``wandb/run-*`` summary files or local
``results.json`` paths from the run summary.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import re
import textwrap
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_ENTITY_PROJECT = "penn-pal/libero"
DEFAULT_RUNS = (
    "ald7rqk2=vanilla-from-libero",
    "qfr0siac=target-dot-from-libero",
)


@dataclasses.dataclass(frozen=True)
class TaskMetric:
    task_id: int
    task_description: str
    successes: int
    episodes: int
    success_rate: float


@dataclasses.dataclass(frozen=True)
class EvalRun:
    label: str
    run_ref: str
    wandb_path: str | None
    status: str | None
    task_suite_name: str | None
    total_successes: int
    total_episodes: int
    total_success_rate: float
    task_metrics: tuple[TaskMetric, ...]
    source: str


def _parse_run_spec(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        return raw, raw
    run_ref, label = raw.split("=", 1)
    return run_ref.strip(), label.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _task_label(task: TaskMetric) -> str:
    text = task.task_description
    prefixes = (
        "pick up the black bowl ",
        "pick_up_the_black_bowl_",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("_and_place_it_on_the_plate", "")
    text = text.replace(" and place it on the plate", "")
    text = text.replace("_", " ")
    return f"{task.task_id:02d}: {text}"


def _task_metrics_from_results(results: dict[str, Any]) -> tuple[TaskMetric, ...]:
    metrics = [
        TaskMetric(
            task_id=int(task["task_id"]),
            task_description=str(task["task_description"]),
            successes=int(task["successes"]),
            episodes=int(task["episodes"]),
            success_rate=float(task["success_rate"]),
        )
        for task in results.get("task_results", [])
    ]
    return tuple(sorted(metrics, key=lambda task: task.task_id))


def _task_metrics_from_summary(summary: dict[str, Any]) -> tuple[TaskMetric, ...]:
    by_id: dict[int, dict[str, Any]] = {}
    pattern = re.compile(r"^eval/per_task/(\d+)_(.+)/(success_rate_final|successes|episodes_completed)$")
    for key, value in summary.items():
        match = pattern.match(key)
        if match is None:
            continue
        task_id = int(match.group(1))
        task_description = match.group(2).replace("_", " ")
        field = match.group(3)
        entry = by_id.setdefault(task_id, {"task_description": task_description})
        entry[field] = value

    metrics = []
    for task_id, values in sorted(by_id.items()):
        episodes = int(values.get("episodes_completed", 0))
        success_rate = float(values.get("success_rate_final", 0.0))
        successes = int(values.get("successes", round(success_rate * episodes)))
        metrics.append(
            TaskMetric(
                task_id=task_id,
                task_description=str(values["task_description"]),
                successes=successes,
                episodes=episodes,
                success_rate=success_rate,
            )
        )
    return tuple(metrics)


def _eval_run_from_results(
    *,
    label: str,
    run_ref: str,
    results: dict[str, Any],
    source: str,
    wandb_path: str | None = None,
) -> EvalRun:
    return EvalRun(
        label=label,
        run_ref=run_ref,
        wandb_path=wandb_path,
        status=results.get("status"),
        task_suite_name=results.get("task_suite_name"),
        total_successes=int(results.get("total_successes", 0)),
        total_episodes=int(results.get("total_episodes", 0)),
        total_success_rate=float(results.get("total_success_rate", 0.0)),
        task_metrics=_task_metrics_from_results(results),
        source=source,
    )


def _eval_run_from_summary(
    *,
    label: str,
    run_ref: str,
    summary: dict[str, Any],
    source: str,
    wandb_path: str | None = None,
) -> EvalRun:
    total_episodes = int(summary.get("eval/total_episodes", summary.get("eval/episodes_completed", 0)))
    total_success_rate = float(summary.get("eval/total_success_rate", summary.get("eval/total_success_rate_running", 0.0)))
    total_successes = int(summary.get("eval/total_successes", round(total_success_rate * total_episodes)))
    return EvalRun(
        label=label,
        run_ref=run_ref,
        wandb_path=wandb_path,
        status=None,
        task_suite_name=None,
        total_successes=total_successes,
        total_episodes=total_episodes,
        total_success_rate=total_success_rate,
        task_metrics=_task_metrics_from_summary(summary),
        source=source,
    )


def _fetch_from_wandb_api(entity_project: str, run_ref: str, label: str) -> EvalRun:
    import wandb

    run_path = run_ref if "/" in run_ref else f"{entity_project}/{run_ref}"
    api_run = wandb.Api().run(run_path)
    summary = dict(api_run.summary)

    results_out_path = summary.get("eval/results_out_path")
    if isinstance(results_out_path, str):
        local_results = Path(results_out_path)
        if local_results.exists():
            return _eval_run_from_results(
                label=label,
                run_ref=run_ref,
                results=_read_json(local_results),
                source=f"wandb-summary-local-results:{local_results}",
                wandb_path=api_run.path,
            )

    return _eval_run_from_summary(
        label=label,
        run_ref=run_ref,
        summary=summary,
        source="wandb-api-summary",
        wandb_path=api_run.path,
    )


def _find_local_wandb_summary(run_ref: str, wandb_dir: Path) -> Path | None:
    candidates = sorted(wandb_dir.glob(f"run-*-{run_ref}/files/wandb-summary.json"))
    return candidates[-1] if candidates else None


def _fetch_from_local_fallback(run_ref: str, label: str, wandb_dir: Path) -> EvalRun:
    summary_path = _find_local_wandb_summary(run_ref, wandb_dir)
    if summary_path is None:
        raise FileNotFoundError(f"No local W&B summary found for run `{run_ref}` under {wandb_dir}.")

    summary = _read_json(summary_path)
    results_out_path = summary.get("eval/results_out_path")
    if isinstance(results_out_path, str) and Path(results_out_path).exists():
        return _eval_run_from_results(
            label=label,
            run_ref=run_ref,
            results=_read_json(Path(results_out_path)),
            source=f"local-wandb-summary-local-results:{results_out_path}",
        )
    return _eval_run_from_summary(
        label=label,
        run_ref=run_ref,
        summary=summary,
        source=f"local-wandb-summary:{summary_path}",
    )


def _fetch_eval_run(
    *,
    entity_project: str,
    run_ref: str,
    label: str,
    wandb_dir: Path,
    allow_local_fallback: bool,
) -> EvalRun:
    try:
        return _fetch_from_wandb_api(entity_project, run_ref, label)
    except Exception as exc:
        if not allow_local_fallback:
            raise RuntimeError(
                f"Failed to fetch `{run_ref}` from W&B. Re-run with --allow-local-fallback "
                "to use local wandb summaries when available."
            ) from exc
        print(f"[warn] W&B API fetch failed for {run_ref}: {exc}")
        return _fetch_from_local_fallback(run_ref, label, wandb_dir)


def _write_summary_json(runs: list[EvalRun], output_path: Path) -> None:
    payload = {
        "runs": [
            {
                "label": run.label,
                "run_ref": run.run_ref,
                "wandb_path": run.wandb_path,
                "source": run.source,
                "status": run.status,
                "task_suite_name": run.task_suite_name,
                "total_successes": run.total_successes,
                "total_episodes": run.total_episodes,
                "total_success_rate": run.total_success_rate,
                "task_metrics": [dataclasses.asdict(task) for task in run.task_metrics],
            }
            for run in runs
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_summary_csv(runs: list[EvalRun], output_path: Path) -> None:
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run_label",
                "run_ref",
                "task_id",
                "task_description",
                "successes",
                "episodes",
                "success_rate",
            ],
        )
        writer.writeheader()
        for run in runs:
            for task in run.task_metrics:
                writer.writerow(
                    {
                        "run_label": run.label,
                        "run_ref": run.run_ref,
                        "task_id": task.task_id,
                        "task_description": task.task_description,
                        "successes": task.successes,
                        "episodes": task.episodes,
                        "success_rate": task.success_rate,
                    }
                )


def _plot_overall_success(runs: list[EvalRun], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [run.label for run in runs]
    values = [run.total_success_rate for run in runs]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(runs)]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("LIBERO Spatial Eval Success Rate")
    ax.grid(axis="y", alpha=0.25)
    for bar, run in zip(bars, runs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{run.total_success_rate:.0%}\n{run.total_successes}/{run.total_episodes}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_per_task_success(runs: list[EvalRun], output_path: Path) -> None:
    if not runs:
        return

    task_ids = sorted({task.task_id for run in runs for task in run.task_metrics})
    descriptions = {
        task.task_id: task.task_description
        for run in runs
        for task in run.task_metrics
    }
    labels = [
        "\n".join(textwrap.wrap(_task_label(TaskMetric(task_id, descriptions[task_id], 0, 0, 0.0)), width=24))
        for task_id in task_ids
    ]
    x_positions = list(range(len(task_ids)))
    width = min(0.8 / max(len(runs), 1), 0.35)

    fig, ax = plt.subplots(figsize=(max(12, len(task_ids) * 1.1), 6.5))
    for run_index, run in enumerate(runs):
        rates_by_id = {task.task_id: task.success_rate for task in run.task_metrics}
        successes_by_id = {task.task_id: task.successes for task in run.task_metrics}
        episodes_by_id = {task.task_id: task.episodes for task in run.task_metrics}
        offset = (run_index - (len(runs) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            [rates_by_id.get(task_id, 0.0) for task_id in task_ids],
            width=width,
            label=run.label,
        )
        for bar, task_id in zip(bars, task_ids, strict=True):
            episodes = episodes_by_id.get(task_id, 0)
            successes = successes_by_id.get(task_id, 0)
            if episodes:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{successes}/{episodes}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Success rate")
    ax.set_title("Per-task LIBERO Spatial Success Rates")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_delta(runs: list[EvalRun], output_path: Path) -> None:
    if len(runs) != 2:
        return

    baseline, comparison = runs
    baseline_rates = {task.task_id: task.success_rate for task in baseline.task_metrics}
    comparison_rates = {task.task_id: task.success_rate for task in comparison.task_metrics}
    task_ids = sorted(set(baseline_rates) | set(comparison_rates))
    deltas = [comparison_rates.get(task_id, 0.0) - baseline_rates.get(task_id, 0.0) for task_id in task_ids]
    descriptions = {
        task.task_id: task.task_description
        for run in runs
        for task in run.task_metrics
    }
    labels = [
        "\n".join(textwrap.wrap(_task_label(TaskMetric(task_id, descriptions[task_id], 0, 0, 0.0)), width=24))
        for task_id in task_ids
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(task_ids) * 1.1), 5.5))
    colors = ["#D62728" if delta < 0 else "#2CA02C" for delta in deltas]
    ax.bar(range(len(task_ids)), deltas, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_ylabel(f"Success-rate delta\n{comparison.label} minus {baseline.label}")
    ax.set_title("Per-task Change Between Eval Runs")
    ax.set_xticks(range(len(task_ids)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_markdown_report(runs: list[EvalRun], output_path: Path, plot_paths: list[Path]) -> None:
    lines = ["# LIBERO W&B Eval Visualization", ""]
    for run in runs:
        lines += [
            f"## {run.label}",
            "",
            f"- run ref: `{run.run_ref}`",
            f"- source: `{run.source}`",
            f"- overall: **{run.total_success_rate:.1%}** ({run.total_successes}/{run.total_episodes})",
            "",
            "| Task | Success | Rate |",
            "|---|---:|---:|",
        ]
        for task in run.task_metrics:
            lines.append(f"| {_task_label(task)} | {task.successes}/{task.episodes} | {task.success_rate:.0%} |")
        lines.append("")

    lines += ["## Plots", ""]
    for path in plot_paths:
        lines += [f"![{path.stem}]({path.name})", ""]
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-project", default=DEFAULT_ENTITY_PROJECT)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run ID/path, optionally with label as RUN=LABEL. Defaults to the two May 7 LIBERO spatial eval runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/libero_wandb_eval_viz"))
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    parser.add_argument("--allow-local-fallback", action="store_true")
    args = parser.parse_args()

    run_specs = args.run or list(DEFAULT_RUNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        _fetch_eval_run(
            entity_project=args.entity_project,
            run_ref=run_ref,
            label=label,
            wandb_dir=args.wandb_dir,
            allow_local_fallback=args.allow_local_fallback,
        )
        for run_ref, label in (_parse_run_spec(raw) for raw in run_specs)
    ]

    summary_json = args.output_dir / "summary.json"
    summary_csv = args.output_dir / "per_task_success.csv"
    overall_plot = args.output_dir / "overall_success.png"
    per_task_plot = args.output_dir / "per_task_success.png"
    delta_plot = args.output_dir / "per_task_delta.png"
    report = args.output_dir / "report.md"

    _write_summary_json(runs, summary_json)
    _write_summary_csv(runs, summary_csv)
    _plot_overall_success(runs, overall_plot)
    _plot_per_task_success(runs, per_task_plot)
    plot_paths = [overall_plot, per_task_plot]
    if len(runs) == 2:
        _plot_delta(runs, delta_plot)
        plot_paths.append(delta_plot)
    _write_markdown_report(runs, report, plot_paths)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
