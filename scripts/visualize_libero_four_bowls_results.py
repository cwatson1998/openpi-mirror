"""Create slide-ready visualizations for LIBERO Spatial Four Bowls evals."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import re
from typing import Any

from matplotlib import colors as mcolors
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_ROOT = Path("data/libero/evals/libero_spatial_four_bowls_eval5_h220_20260508_002332")
DEFAULT_OUTPUT_DIR = Path("outputs/libero_spatial_four_bowls_eval5_h220_slides")

BACKGROUND = "#F8FAFC"
INK = "#0F172A"
MUTED = "#64748B"
GRID = "#CBD5E1"
VANILLA = "#2563EB"
TARGET_DOT = "#DB2777"
GREEN = "#059669"
RED = "#DC2626"
AMBER = "#D97706"


@dataclasses.dataclass(frozen=True)
class TaskResult:
    task_id: int
    description: str
    successes: int
    episodes: int
    success_rate: float


@dataclasses.dataclass(frozen=True)
class EvalResult:
    key: str
    label: str
    color: str
    results_path: Path
    status: str
    task_suite_name: str
    total_successes: int
    total_episodes: int
    total_success_rate: float
    tasks: tuple[TaskResult, ...]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_eval_result(results_path: Path, *, key: str, label: str, color: str) -> EvalResult:
    data = _read_json(results_path)
    tasks = tuple(
        sorted(
            (
                TaskResult(
                    task_id=int(task["task_id"]),
                    description=str(task["task_description"]),
                    successes=int(task["successes"]),
                    episodes=int(task["episodes"]),
                    success_rate=float(task["success_rate"]),
                )
                for task in data["task_results"]
            ),
            key=lambda task: task.task_id,
        )
    )
    return EvalResult(
        key=key,
        label=label,
        color=color,
        results_path=results_path,
        status=str(data.get("status", "unknown")),
        task_suite_name=str(data.get("task_suite_name", "unknown")),
        total_successes=int(data["total_successes"]),
        total_episodes=int(data["total_episodes"]),
        total_success_rate=float(data["total_success_rate"]),
        tasks=tasks,
    )


def _task_short_label(description: str) -> str:
    patterns = [
        (r"between the plate and the ramekin", "Between plate & ramekin"),
        (r"next to the ramekin", "Next to ramekin"),
        (r"from table center", "Table center"),
        (r"on the cookie box", "On cookie box"),
        (r"top drawer", "Top drawer"),
        (r"on the ramekin", "On ramekin"),
        (r"next to the cookie box", "Next to cookie box"),
        (r"on the stove", "On stove"),
        (r"next to the plate", "Next to plate"),
        (r"wooden cabinet", "On wooden cabinet"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, description):
            return label
    cleaned = description
    cleaned = cleaned.removeprefix("pick up the black bowl ")
    cleaned = cleaned.replace(" and place it on the plate", "")
    return cleaned[:42]


def _task_lookup(run: EvalResult) -> dict[int, TaskResult]:
    return {task.task_id: task for task in run.tasks}


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_paths = []
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        save_kwargs: dict[str, Any] = {
            "facecolor": fig.get_facecolor(),
            "edgecolor": "none",
        }
        if suffix == "png":
            save_kwargs["dpi"] = 240
        fig.savefig(path, **save_kwargs)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(colors=INK, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    title: str,
    rate: float,
    successes: int,
    episodes: int,
    note: str,
) -> None:
    shadow = patches.FancyBboxPatch(
        (x + 0.008, y - 0.012),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0,
        facecolor="#CBD5E1",
        alpha=0.35,
        transform=ax.transAxes,
        zorder=1,
    )
    card = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor="#E2E8F0",
        facecolor="#FFFFFF",
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(shadow)
    ax.add_patch(card)
    ax.add_patch(
        patches.Rectangle(
            (x, y + height - 0.025),
            width,
            0.025,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
    )
    ax.text(x + 0.04, y + height - 0.105, title, transform=ax.transAxes, color=INK, fontsize=18, weight="bold")
    ax.text(
        x + 0.04,
        y + 0.17,
        f"{rate:.0%}",
        transform=ax.transAxes,
        color=color,
        fontsize=58,
        weight="bold",
    )
    ax.text(
        x + 0.04,
        y + 0.10,
        f"{successes}/{episodes} successful rollouts",
        transform=ax.transAxes,
        color=INK,
        fontsize=15,
        weight="bold",
    )
    ax.text(x + 0.04, y + 0.045, note, transform=ax.transAxes, color=MUTED, fontsize=12)


def plot_headline(runs: list[EvalResult], output_dir: Path) -> list[Path]:
    by_key = {run.key: run for run in runs}
    vanilla = by_key["vanilla"]
    target = by_key["target_dot"]
    delta = target.total_success_rate - vanilla.total_success_rate

    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.055,
        0.905,
        "LIBERO Spatial Four Bowls",
        transform=ax.transAxes,
        color=INK,
        fontsize=34,
        weight="bold",
    )
    ax.text(
        0.055,
        0.852,
        "5 trials per task, 10 tasks. Horizon fixed to 220 policy steps + 10 simulator wait steps.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=15,
    )

    _draw_card(
        ax,
        x=0.075,
        y=0.385,
        width=0.38,
        height=0.36,
        color=vanilla.color,
        title=vanilla.label,
        rate=vanilla.total_success_rate,
        successes=vanilla.total_successes,
        episodes=vanilla.total_episodes,
        note="Unmodified RGB observations",
    )
    _draw_card(
        ax,
        x=0.545,
        y=0.385,
        width=0.38,
        height=0.36,
        color=target.color,
        title=target.label,
        rate=target.total_success_rate,
        successes=target.total_successes,
        episodes=target.total_episodes,
        note="Next-object highlight + placement dot",
    )

    delta_color = GREEN if delta >= 0 else RED
    ax.text(0.5, 0.265, "Target-dot vs. vanilla", transform=ax.transAxes, ha="center", color=MUTED, fontsize=14)
    ax.text(
        0.5,
        0.19,
        f"{delta * 100:+.0f} pp",
        transform=ax.transAxes,
        ha="center",
        color=delta_color,
        fontsize=34,
        weight="bold",
    )
    ax.text(
        0.5,
        0.132,
        "On this four-bowls benchmark, the vanilla RGB checkpoint outperformed the target-dot checkpoint.",
        transform=ax.transAxes,
        ha="center",
        color=INK,
        fontsize=14,
    )
    ax.text(
        0.055,
        0.055,
        "Source: local eval results.json files from "
        "libero_spatial_four_bowls_eval5_h220_20260508_002332",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10.5,
    )

    return _save_figure(fig, output_dir, "slide_01_headline")


def plot_per_task_bars(runs: list[EvalResult], output_dir: Path) -> list[Path]:
    by_key = {run.key: run for run in runs}
    vanilla = by_key["vanilla"]
    target = by_key["target_dot"]
    vanilla_tasks = _task_lookup(vanilla)
    target_tasks = _task_lookup(target)
    task_ids = sorted(set(vanilla_tasks) | set(target_tasks))
    labels = [_task_short_label(vanilla_tasks.get(task_id, target_tasks[task_id]).description) for task_id in task_ids]
    y = np.arange(len(task_ids))
    bar_height = 0.34

    fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=BACKGROUND)
    _style_axes(ax)
    ax.set_title(
        "Per-task success rates",
        loc="left",
        pad=18,
        color=INK,
        fontsize=25,
        weight="bold",
    )
    ax.text(
        0.0,
        1.012,
        "Blue = vanilla RGB; pink = target-dot + highlight. Each task has 5 rollouts.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=12,
    )

    vanilla_rates = np.array([vanilla_tasks[task_id].success_rate for task_id in task_ids])
    target_rates = np.array([target_tasks[task_id].success_rate for task_id in task_ids])
    vanilla_bars = ax.barh(y + bar_height / 2, vanilla_rates * 100, height=bar_height, color=vanilla.color, label=vanilla.label)
    target_bars = ax.barh(y - bar_height / 2, target_rates * 100, height=bar_height, color=target.color, label=target.label)

    for bars, task_map in [(vanilla_bars, vanilla_tasks), (target_bars, target_tasks)]:
        for bar, task_id in zip(bars, task_ids, strict=True):
            task = task_map[task_id]
            x = bar.get_width()
            label_x = min(x + 2.0, 96.0) if x > 0 else 2.0
            ha = "left" if x < 92 else "right"
            color = INK if x < 92 else "#FFFFFF"
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{task.successes}/{task.episodes}",
                va="center",
                ha=ha,
                color=color,
                fontsize=10,
                weight="bold",
            )

    ax.axvline(vanilla.total_success_rate * 100, color=vanilla.color, linewidth=2, alpha=0.55)
    ax.axvline(target.total_success_rate * 100, color=target.color, linewidth=2, alpha=0.55)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Success rate (%)", color=INK, fontsize=12, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, alpha=0.75, linewidth=0.8)
    fig.subplots_adjust(left=0.23, right=0.97, top=0.88, bottom=0.105)
    return _save_figure(fig, output_dir, "slide_02_per_task_bars")


def plot_heatmap(runs: list[EvalResult], output_dir: Path) -> list[Path]:
    by_key = {run.key: run for run in runs}
    ordered_runs = [by_key["vanilla"], by_key["target_dot"]]
    task_ids = [task.task_id for task in ordered_runs[0].tasks]
    labels = [_task_short_label(task.description) for task in ordered_runs[0].tasks]
    values = np.array([[task.success_rate for task in run.tasks] for run in ordered_runs]).T
    counts = [[f"{task.successes}/{task.episodes}" for task in run.tasks] for run in ordered_runs]

    cmap = mcolors.LinearSegmentedColormap.from_list("success", ["#FEF2F2", "#FEF3C7", "#DCFCE7", "#059669"])
    fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=BACKGROUND)
    _style_axes(ax)
    im = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_title("Task-level success matrix", loc="left", pad=18, color=INK, fontsize=25, weight="bold")
    ax.text(
        0.0,
        1.012,
        "Color encodes success rate; text shows successful rollouts out of 5.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=12,
    )
    ax.set_xticks(range(len(ordered_runs)))
    ax.set_xticklabels([run.label for run in ordered_runs], color=INK, fontsize=14, weight="bold")
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(labels, color=INK, fontsize=12)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            rate = values[row, col]
            text_color = "#FFFFFF" if rate >= 0.78 else INK
            ax.text(
                col,
                row,
                f"{rate:.0%}\n{counts[col][row]}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                weight="bold",
            )

    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, colors=INK)
    cbar.set_label("Success rate", color=INK, fontsize=11, weight="bold")
    fig.subplots_adjust(left=0.26, right=0.90, top=0.88, bottom=0.10)
    return _save_figure(fig, output_dir, "slide_03_success_matrix")


def plot_delta(runs: list[EvalResult], output_dir: Path) -> list[Path]:
    by_key = {run.key: run for run in runs}
    vanilla = by_key["vanilla"]
    target = by_key["target_dot"]
    vanilla_tasks = _task_lookup(vanilla)
    target_tasks = _task_lookup(target)
    task_ids = sorted(vanilla_tasks)
    labels = [_task_short_label(vanilla_tasks[task_id].description) for task_id in task_ids]
    deltas = np.array([target_tasks[task_id].success_rate - vanilla_tasks[task_id].success_rate for task_id in task_ids])
    y = np.arange(len(task_ids))

    fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor=BACKGROUND)
    _style_axes(ax)
    colors = [GREEN if delta > 0 else RED if delta < 0 else AMBER for delta in deltas]
    bars = ax.barh(y, deltas * 100, color=colors)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_title("Where target-dot helped or hurt", loc="left", pad=18, color=INK, fontsize=25, weight="bold")
    ax.text(
        0.0,
        1.012,
        "Difference is target-dot success rate minus vanilla success rate.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=12,
    )
    for bar, delta in zip(bars, deltas, strict=True):
        x = bar.get_width()
        ax.text(
            x + (2.0 if x >= 0 else -2.0),
            bar.get_y() + bar.get_height() / 2,
            f"{delta * 100:+.0f} pp",
            va="center",
            ha="left" if x >= 0 else "right",
            color=INK,
            fontsize=11,
            weight="bold",
        )

    ax.set_xlim(-105, 105)
    ax.set_xlabel("Success-rate difference (percentage points)", color=INK, fontsize=12, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, alpha=0.75, linewidth=0.8)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.88, bottom=0.105)
    return _save_figure(fig, output_dir, "slide_04_target_dot_delta")


def write_csv(runs: list[EvalResult], output_path: Path) -> None:
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run",
                "task_id",
                "task_label",
                "task_description",
                "successes",
                "episodes",
                "success_rate",
            ],
        )
        writer.writeheader()
        for run in runs:
            for task in run.tasks:
                writer.writerow(
                    {
                        "run": run.label,
                        "task_id": task.task_id,
                        "task_label": _task_short_label(task.description),
                        "task_description": task.description,
                        "successes": task.successes,
                        "episodes": task.episodes,
                        "success_rate": task.success_rate,
                    }
                )


def write_markdown(runs: list[EvalResult], output_path: Path, figure_paths: list[Path]) -> None:
    by_key = {run.key: run for run in runs}
    vanilla = by_key["vanilla"]
    target = by_key["target_dot"]
    lines = [
        "# LIBERO Spatial Four Bowls Eval",
        "",
        f"- Run root: `{DEFAULT_RESULTS_ROOT}`",
        f"- Vanilla RGB: **{vanilla.total_success_rate:.0%}** ({vanilla.total_successes}/{vanilla.total_episodes})",
        f"- Target-dot + highlight: **{target.total_success_rate:.0%}** ({target.total_successes}/{target.total_episodes})",
        f"- Delta, target-dot minus vanilla: **{(target.total_success_rate - vanilla.total_success_rate) * 100:+.0f} pp**",
        "- Horizon: 220 policy steps + 10 wait steps",
        "- W&B: target-dot `18qphdzw`; vanilla `djhgixjp`",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in figure_paths if path.suffix == ".png")
    lines += [
        "",
        "## Per-task Results",
        "",
        "| Task | Vanilla | Target-dot | Delta |",
        "|---|---:|---:|---:|",
    ]
    vanilla_tasks = _task_lookup(vanilla)
    target_tasks = _task_lookup(target)
    for task_id in sorted(vanilla_tasks):
        v = vanilla_tasks[task_id]
        t = target_tasks[task_id]
        lines.append(
            f"| {_task_short_label(v.description)} | {v.successes}/{v.episodes} ({v.success_rate:.0%}) "
            f"| {t.successes}/{t.episodes} ({t.success_rate:.0%}) "
            f"| {(t.success_rate - v.success_rate) * 100:+.0f} pp |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    target_results = args.results_root / "target-dot-from-libero" / "results.json"
    vanilla_results = args.results_root / "vanilla-from-libero" / "results.json"
    if not target_results.exists():
        raise FileNotFoundError(target_results)
    if not vanilla_results.exists():
        raise FileNotFoundError(vanilla_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        _read_eval_result(
            vanilla_results,
            key="vanilla",
            label="Vanilla RGB",
            color=VANILLA,
        ),
        _read_eval_result(
            target_results,
            key="target_dot",
            label="Target-dot + highlight",
            color=TARGET_DOT,
        ),
    ]

    figure_paths: list[Path] = []
    figure_paths.extend(plot_headline(runs, args.output_dir))
    figure_paths.extend(plot_per_task_bars(runs, args.output_dir))
    figure_paths.extend(plot_heatmap(runs, args.output_dir))
    figure_paths.extend(plot_delta(runs, args.output_dir))
    write_csv(runs, args.output_dir / "per_task_success.csv")
    write_markdown(runs, args.output_dir / "README.md", figure_paths)

    print(f"Wrote slide figures to {args.output_dir}")
    for path in figure_paths:
        print(path)
    print(args.output_dir / "per_task_success.csv")
    print(args.output_dir / "README.md")


if __name__ == "__main__":
    main()
