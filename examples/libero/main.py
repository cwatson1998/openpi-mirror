import collections
import dataclasses
from datetime import datetime
from datetime import timezone
import hashlib
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import wandb

from annotation import OnlineNextObjectAnnotationPlan
from annotation import OnlineNextObjectHighlightTracker
from annotation import build_online_next_object_annotation_plan_from_source_demo
from annotation import draw_placement_dot_on_observations
from openpi.training import libero as libero_utils

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _ensure_repo_local_libero_config() -> None:
    if "LIBERO_CONFIG_PATH" in os.environ:
        return

    config_root = _REPO_ROOT / ".cache/libero-openpi"
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    config_root.mkdir(parents=True, exist_ok=True)

    config_path = config_root / "config.yaml"
    if config_path.exists():
        return

    benchmark_root = _REPO_ROOT / "third_party/libero/libero/libero"
    config_path.write_text(
        "\n".join(
            [
                f"assets: {benchmark_root / 'assets'}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"benchmark_root: {benchmark_root}",
                f"datasets: {_REPO_ROOT / 'third_party/libero/libero/datasets'}",
                f"init_states: {benchmark_root / 'init_files'}",
            ]
        )
        + "\n"
    )


_ensure_repo_local_libero_config()

from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import MaskedSegmentationRenderEnv  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
REPLAY_VIDEO_FPS = 10
_WANDB_MAX_TAG_LENGTH = 64
_ONLINE_NEXT_OBJECT_SOURCE_SUITE_BY_EVAL_SUITE = {
    "libero_spatial": "libero_spatial",
    "libero_spatial_four_bowls": "libero_spatial",
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite. Options include libero_spatial, libero_spatial_four_bowls, libero_object, libero_goal, libero_10, libero_90
    task_indices: tuple[int, ...] = ()
    task_names: tuple[str, ...] = ()
    task_split_file: str | None = None
    task_split: str = "eval"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    mask_instances_csv: str = ""  # Optional comma-separated instance names to mask in returned RGB observations.
    mask_rgb_csv: str = "0,0,0"  # RGB color used for masked pixels.
    mask_alpha: float = 1.0  # Alpha used to blend masked pixels with mask_rgb_csv.
    mask_cameras_csv: str = ""  # Optional comma-separated cameras to mask, e.g. agentview,robot0_eye_in_hand.
    next_object_highlighting: bool = False  # Enable demo-derived online highlighting during eval.
    next_object_highlight_rgb_csv: str = "255,105,180"  # RGB color used for next-object highlighting.
    next_object_highlight_alpha: float = 1.0  # Alpha used to blend highlighted pixels.
    next_object_highlight_release_steps: int = 4  # Consecutive non-grasped steps required before advancing.
    next_object_placement_dot: bool = False  # Draw a dot at the demo-derived placement target for the active object.
    placement_dot_rgb_csv: str = "0,96,255"  # RGB color used for the placement target dot.
    placement_dot_alpha: float = 1.0  # Alpha used to blend the placement target dot.
    placement_dot_radius_px: int = 5  # Radius of the placement target dot in raw simulator pixels.

    #################################################################################################################
    # Utils
    #################################################################################################################
    # Output directory for rollout videos and results.json.
    # Defaults to data/libero/runs/<YYYYMMDD_HHMMSS>_<task_suite_name> so each run
    # gets its own directory and nothing is ever clobbered.
    # Pass an explicit path to override (e.g. --args.video-out-path data/libero/my_run).
    video_out_path: str | None = None
    results_out_path: str | None = None  # Optional path to save aggregated results JSON
    progress_out_path: str | None = None  # Optional path to save incremental eval progress JSON

    # Optional JSON file that maps task_instruction -> logic_task_description.
    # Expected format matches libero_object_train7_logic_descriptions.json:
    #   {"tasks": [{"task_instruction": "...", "logic_task_description": "..."}, ...]}
    # When provided, any task whose natural-language instruction matches an entry will
    # have its prompt replaced with the corresponding logic_task_description.
    # Tasks with no entry in the file keep their default natural-language prompt.
    prompt_override_file: str | None = None

    #################################################################################################################
    # W&B
    #################################################################################################################
    wandb_enabled: bool = False
    wandb_project: str = "libero"
    wandb_name: str | None = None
    wandb_group: str | None = None
    wandb_tags_csv: str = ""
    policy_config: str | None = None
    checkpoint_dir: str | None = None
    train_run_name: str | None = None
    wandb_upload_episode_videos: bool = True

    seed: int = 7  # Random Seed (for reproducibility)


def _parse_tags(csv_value: str) -> list[str]:
    tags = []
    for tag in (tag.strip() for tag in csv_value.split(",") if tag.strip()):
        parsed_tag = tag
        if len(tag) > _WANDB_MAX_TAG_LENGTH:
            tag_hash = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
            parsed_tag = f"{tag[: _WANDB_MAX_TAG_LENGTH - 9]}_{tag_hash}"
        tags.append(parsed_tag)
    return tags


def _parse_csv_list(csv_value: str) -> list[str]:
    return [item.strip() for item in csv_value.split(",") if item.strip()]


def _parse_rgb_csv(csv_value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in csv_value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated RGB values, got `{csv_value}`.")
    return tuple(int(part) for part in parts)


def _validate_alpha(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0.0, 1.0].")


def _validate_online_next_object_args(args: Args) -> None:
    online_annotation_requested = args.next_object_highlighting or args.next_object_placement_dot
    if not online_annotation_requested:
        return
    if args.task_suite_name not in _ONLINE_NEXT_OBJECT_SOURCE_SUITE_BY_EVAL_SUITE:
        supported_suites = ", ".join(sorted(_ONLINE_NEXT_OBJECT_SOURCE_SUITE_BY_EVAL_SUITE))
        raise ValueError(
            "Online next-object highlighting and placement dots are currently supported only for "
            f"{supported_suites}, got `{args.task_suite_name}`."
        )
    if args.next_object_placement_dot and not args.next_object_highlighting:
        raise ValueError("`next_object_placement_dot` requires `next_object_highlighting`.")
    if args.next_object_highlight_release_steps <= 0:
        raise ValueError("next_object_highlight_release_steps must be positive.")
    _validate_alpha("next_object_highlight_alpha", args.next_object_highlight_alpha)
    _parse_rgb_csv(args.next_object_highlight_rgb_csv)
    if args.next_object_placement_dot:
        _validate_alpha("placement_dot_alpha", args.placement_dot_alpha)
        _parse_rgb_csv(args.placement_dot_rgb_csv)
        if args.placement_dot_radius_px <= 0:
            raise ValueError("placement_dot_radius_px must be positive.")


def _default_eval_wandb_name(args: Args) -> str:
    checkpoint_name = pathlib.Path(args.checkpoint_dir).name if args.checkpoint_dir else "unknown_ckpt"
    train_run_name = args.train_run_name or (
        pathlib.Path(args.checkpoint_dir).parent.name if args.checkpoint_dir else "unknown_run"
    )
    return f"eval_{train_run_name}_ckpt_{checkpoint_name}_{args.task_suite_name}"


def _init_wandb(args: Args) -> None:
    if not args.wandb_enabled:
        wandb.init(mode="disabled")
        return

    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name or _default_eval_wandb_name(args),
        "job_type": "eval",
        "config": {
            "task_suite_name": args.task_suite_name,
            "task_indices": list(args.task_indices),
            "task_names": list(args.task_names),
            "task_split_file": args.task_split_file,
            "task_split": args.task_split,
            "num_trials_per_task": args.num_trials_per_task,
            "mask_instances_csv": args.mask_instances_csv,
            "mask_rgb_csv": args.mask_rgb_csv,
            "mask_alpha": args.mask_alpha,
            "mask_cameras_csv": args.mask_cameras_csv,
            "next_object_highlighting": args.next_object_highlighting,
            "next_object_highlight_rgb_csv": args.next_object_highlight_rgb_csv,
            "next_object_highlight_alpha": args.next_object_highlight_alpha,
            "next_object_highlight_release_steps": args.next_object_highlight_release_steps,
            "next_object_placement_dot": args.next_object_placement_dot,
            "placement_dot_rgb_csv": args.placement_dot_rgb_csv,
            "placement_dot_alpha": args.placement_dot_alpha,
            "placement_dot_radius_px": args.placement_dot_radius_px,
            "replan_steps": args.replan_steps,
            "resize_size": args.resize_size,
            "seed": args.seed,
            "prompt_override_file": args.prompt_override_file,
            "policy_config": args.policy_config,
            "checkpoint_dir": args.checkpoint_dir,
            "train_run_name": args.train_run_name,
        },
    }
    tags = _parse_tags(args.wandb_tags_csv)
    if tags:
        init_kwargs["tags"] = tags
    if args.wandb_group is not None:
        init_kwargs["group"] = args.wandb_group
    wandb.init(**init_kwargs)


def _task_metric_prefix(task_id: int) -> str:
    return f"streaming/task_{task_id}"


def _slugify_metric_component(value: str) -> str:
    slug_chars = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    return "_".join("".join(slug_chars).split("_"))


def _task_named_metric_prefix(task_id: int, task_description: str) -> str:
    return f"eval/per_task/{task_id:02d}_{_slugify_metric_component(task_description)}"


def _define_wandb_metrics(selected_task_ids: list[int], task_descriptions_by_id: dict[int, str]) -> None:
    if wandb.run is None:
        return

    wandb.define_metric("eval/episode", hidden=True)
    for metric_name in (
        "eval/episode_success",
        "eval/episode_steps",
        "eval/episode_wait_steps",
        "eval/episode_policy_steps",
        "eval/episode_policy_inference_calls",
        "eval/episode_reached_policy_step",
        "eval/episode_runtime_s",
        "eval/episode_had_error",
        "eval/episodes_completed",
        "eval/tasks_completed",
        "eval/task_id",
        "eval/task_success_rate",
        "eval/total_successes",
        "eval/total_success_rate_running",
        "eval/current_task_id",
        "eval/current_task_episode",
        "eval/current_task_successes",
        "eval/current_task_success_rate_running",
    ):
        wandb.define_metric(metric_name, step_metric="eval/episode")

    for task_id in selected_task_ids:
        prefix = _task_metric_prefix(task_id)
        wandb.define_metric(f"{prefix}/episode", hidden=True)
        for metric_name in (
            "episode_success",
            "episode_steps",
            "episode_wait_steps",
            "episode_policy_steps",
            "episode_policy_inference_calls",
            "episode_reached_policy_step",
            "episode_runtime_s",
            "episode_had_error",
            "episodes_completed",
            "successes",
            "success_rate_running",
        ):
            wandb.define_metric(f"{prefix}/{metric_name}", step_metric=f"{prefix}/episode")

        named_prefix = _task_named_metric_prefix(task_id, task_descriptions_by_id[task_id])
        for metric_name in (
            "episode_success",
            "episodes_completed",
            "successes",
            "success_rate_running",
        ):
            wandb.define_metric(f"{named_prefix}/{metric_name}", step_metric="eval/episode")


def _log_wandb_episode_metrics(
    *,
    task_id: int,
    task_description: str,
    episode_index: int,
    success: bool,
    steps_taken: int,
    wait_steps_taken: int,
    policy_steps_taken: int,
    policy_inference_calls: int,
    episode_runtime_s: float,
    had_error: bool,
    task_episodes: int,
    task_successes: int,
    total_episodes: int,
    total_successes: int,
    tasks_completed: int,
) -> None:
    if wandb.run is None:
        return

    task_prefix = _task_metric_prefix(task_id)
    named_task_prefix = _task_named_metric_prefix(task_id, task_description)
    task_success_rate = float(task_successes) / float(task_episodes)
    total_success_rate = float(total_successes) / float(total_episodes)
    wandb.log(
        {
            "eval/episode": total_episodes,
            "eval/episode_success": float(success),
            "eval/episode_steps": steps_taken,
            "eval/episode_wait_steps": wait_steps_taken,
            "eval/episode_policy_steps": policy_steps_taken,
            "eval/episode_policy_inference_calls": policy_inference_calls,
            "eval/episode_reached_policy_step": float(policy_steps_taken > 0),
            "eval/episode_runtime_s": episode_runtime_s,
            "eval/episode_had_error": float(had_error),
            "eval/episodes_completed": total_episodes,
            "eval/tasks_completed": tasks_completed,
            "eval/total_successes": total_successes,
            "eval/total_success_rate_running": total_success_rate,
            "eval/current_task_id": task_id,
            "eval/current_task_episode": task_episodes,
            "eval/current_task_successes": task_successes,
            "eval/current_task_success_rate_running": task_success_rate,
            f"{task_prefix}/episode": episode_index + 1,
            f"{task_prefix}/episode_success": float(success),
            f"{task_prefix}/episode_steps": steps_taken,
            f"{task_prefix}/episode_wait_steps": wait_steps_taken,
            f"{task_prefix}/episode_policy_steps": policy_steps_taken,
            f"{task_prefix}/episode_policy_inference_calls": policy_inference_calls,
            f"{task_prefix}/episode_reached_policy_step": float(policy_steps_taken > 0),
            f"{task_prefix}/episode_runtime_s": episode_runtime_s,
            f"{task_prefix}/episode_had_error": float(had_error),
            f"{task_prefix}/episodes_completed": task_episodes,
            f"{task_prefix}/successes": task_successes,
            f"{task_prefix}/success_rate_running": task_success_rate,
            f"{named_task_prefix}/episode_success": float(success),
            f"{named_task_prefix}/episodes_completed": task_episodes,
            f"{named_task_prefix}/successes": task_successes,
            f"{named_task_prefix}/success_rate_running": task_success_rate,
        }
    )


def _log_wandb_task_metrics(
    *,
    task_id: int,
    task_description: str,
    task_episodes: int,
    task_successes: int,
    total_episodes: int,
    total_successes: int,
    tasks_completed: int,
) -> None:
    if wandb.run is None:
        return

    named_task_prefix = _task_named_metric_prefix(task_id, task_description)
    task_success_rate = float(task_successes) / float(task_episodes)
    wandb.log(
        {
            "eval/episode": total_episodes,
            "eval/task_id": task_id,
            "eval/task_success_rate": task_success_rate,
            "eval/total_success_rate_running": float(total_successes) / float(total_episodes),
            "eval/tasks_completed": tasks_completed,
            "eval/episodes_completed": total_episodes,
            f"{named_task_prefix}/success_rate_final": task_success_rate,
        }
    )


def _log_wandb_final_task_summary(task_results: list[dict], *, total_episodes: int) -> None:
    if wandb.run is None:
        return

    table = wandb.Table(
        columns=[
            "task_id",
            "task_description",
            "episodes",
            "successes",
            "success_rate",
        ]
    )
    for task_result in task_results:
        table.add_data(
            task_result["task_id"],
            task_result["task_description"],
            task_result["episodes"],
            task_result["successes"],
            task_result["success_rate"],
        )

    wandb.log(
        {
            "eval/episode": total_episodes,
            "eval/per_task_success_rates_table": table,
            "eval/per_task_success_rates_bar": wandb.plot.bar(
                table,
                "task_description",
                "success_rate",
                title="Final per-task success rates",
            ),
        }
    )


def _build_task_result(
    *,
    task_id: int,
    task_description: str,
    task_episodes: int,
    task_successes: int,
    episode_results: list[dict],
    next_object_grasp_order: tuple[str, ...] = (),
    next_object_placement_targets: tuple[tuple[float, float, float] | None, ...] = (),
) -> dict:
    return {
        "task_id": task_id,
        "task_description": task_description,
        "episodes": task_episodes,
        "successes": task_successes,
        "success_rate": float(task_successes) / float(task_episodes) if task_episodes else 0.0,
        "next_object_grasp_order": list(next_object_grasp_order),
        "next_object_placement_targets": [
            None if target is None else [float(value) for value in target] for target in next_object_placement_targets
        ],
        "episode_results": episode_results,
    }


def _build_results_payload(
    *,
    args: Args,
    run_started_at: str,
    selected_task_ids: list[int],
    selected_tasks: tuple[str, ...],
    video_out_path: pathlib.Path,
    task_results: list[dict],
    total_episodes: int,
    total_successes: int,
    status: str,
    active_task: dict | None = None,
) -> dict:
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    return {
        "schema_version": 1,
        "status": status,
        "generated_at": run_started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - keep Python 3.10 compatibility
        "active_task": active_task,
        "task_suite_name": args.task_suite_name,
        "task_split_file": args.task_split_file,
        "task_split": args.task_split,
        "selected_task_ids": selected_task_ids,
        "selected_tasks": selected_tasks,
        "num_trials_per_task": args.num_trials_per_task,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "resize_size": args.resize_size,
        "seed": args.seed,
        "next_object_highlighting": args.next_object_highlighting,
        "next_object_highlight_rgb_csv": args.next_object_highlight_rgb_csv,
        "next_object_highlight_alpha": args.next_object_highlight_alpha,
        "next_object_highlight_release_steps": args.next_object_highlight_release_steps,
        "next_object_placement_dot": args.next_object_placement_dot,
        "placement_dot_rgb_csv": args.placement_dot_rgb_csv,
        "placement_dot_alpha": args.placement_dot_alpha,
        "placement_dot_radius_px": args.placement_dot_radius_px,
        "host": args.host,
        "port": args.port,
        "video_out_path": str(video_out_path),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "task_results": task_results,
    }


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _persist_eval_outputs(
    *,
    args: Args,
    run_started_at: str,
    selected_task_ids: list[int],
    selected_tasks: tuple[str, ...],
    video_out_path: pathlib.Path,
    results_out_path: pathlib.Path,
    progress_out_path: pathlib.Path,
    task_results: list[dict],
    total_episodes: int,
    total_successes: int,
    status: str,
    active_task: dict | None = None,
) -> dict:
    payload = _build_results_payload(
        args=args,
        run_started_at=run_started_at,
        selected_task_ids=selected_task_ids,
        selected_tasks=selected_tasks,
        video_out_path=video_out_path,
        task_results=task_results,
        total_episodes=total_episodes,
        total_successes=total_successes,
        status=status,
        active_task=active_task,
    )
    _write_json(results_out_path, payload)
    if progress_out_path != results_out_path:
        _write_json(progress_out_path, payload)
    return payload


def _maybe_log_wandb_episode_video(
    *,
    task_id: int,
    task_description: str,
    episode_index: int,
    total_episodes: int,
    success: bool,
    had_error: bool,
    video_path: pathlib.Path,
    uploaded_video_categories: dict[int, set[str]],
) -> None:
    if wandb.run is None:
        return

    category = "error" if had_error else "success" if success else "failure"
    if category in uploaded_video_categories[task_id]:
        return

    uploaded_video_categories[task_id].add(category)
    wandb.log(
        {
            "eval/episode": total_episodes,
            f"eval/videos/task_{task_id}/{category}": wandb.Video(str(video_path), fps=REPLAY_VIDEO_FPS, format="mp4"),
            f"eval/videos/task_{task_id}/{category}_episode": episode_index + 1,
            f"eval/videos/task_{task_id}/{category}_task_description": task_description,
        }
    )


def _highlight_camera_names(args: Args) -> list[str]:
    camera_names = _parse_csv_list(args.mask_cameras_csv)
    return camera_names or ["agentview", "robot0_eye_in_hand"]


def _resolve_online_annotation_plan(
    task_suite,
    task_id: int,
    task_suite_name: str,
) -> OnlineNextObjectAnnotationPlan:
    datasets_root = pathlib.Path(get_libero_path("datasets"))
    source_suite_name = _ONLINE_NEXT_OBJECT_SOURCE_SUITE_BY_EVAL_SUITE[task_suite_name]
    source_task_suite = task_suite
    if source_suite_name != task_suite_name:
        source_task_suite = benchmark.get_benchmark_dict()[source_suite_name]()
    source_demo_path = datasets_root / source_task_suite.get_task_demonstration(task_id)
    annotation_plan = build_online_next_object_annotation_plan_from_source_demo(source_demo_path)
    logging.info(
        "Resolved online next-object plan for %s from %s: grasp_order=%s placement_targets=%s",
        task_suite_name,
        source_demo_path,
        list(annotation_plan.grasp_order),
        [
            None if target is None else [float(value) for value in target]
            for target in annotation_plan.placement_targets
        ],
    )
    return annotation_plan


def _apply_online_highlight_mask(
    env: MaskedSegmentationRenderEnv,
    tracker: OnlineNextObjectHighlightTracker | None,
    args: Args,
) -> None:
    if tracker is None or tracker.current_object is None:
        env.clear_instance_mask()
        return

    # Keep the eval-time observation transform aligned with the highlighted
    # training datasets: we overwrite the RGB observations returned by the env
    # wrapper rather than doing any external image postprocessing.
    env.set_instance_mask(
        tracker.current_object,
        mask_rgb=_parse_rgb_csv(args.next_object_highlight_rgb_csv),
        mask_alpha=float(args.next_object_highlight_alpha),
        camera_names=_highlight_camera_names(args),
    )


def _update_online_highlight_tracker(
    env: MaskedSegmentationRenderEnv,
    obs: dict,
    tracker: OnlineNextObjectHighlightTracker | None,
    args: Args,
) -> dict:
    if tracker is None or tracker.current_object is None:
        return obs

    current_object = tracker.current_object
    if current_object not in env.env.object_states_dict:
        raise KeyError(f"Current highlighted object `{current_object}` is missing from the live environment.")

    advanced = tracker.observe(is_current_object_grasped=bool(env.env.object_states_dict[current_object].is_grasped()))
    if not advanced:
        return obs

    _apply_online_highlight_mask(env, tracker, args)
    # The mask target changed for the current simulator state, so re-render the
    # current observations before they are passed to the policy on the next loop.
    return env.regenerate_obs_from_state(env.get_sim_state())


def _current_online_placement_target(
    tracker: OnlineNextObjectHighlightTracker | None,
    annotation_plan: OnlineNextObjectAnnotationPlan | None,
) -> tuple[float, float, float] | None:
    if tracker is None or annotation_plan is None or tracker.current_object is None:
        return None
    if tracker.current_index >= len(annotation_plan.placement_targets):
        raise IndexError(
            "The online next-object tracker advanced beyond the demo-derived placement target sequence: "
            f"index={tracker.current_index}, targets={len(annotation_plan.placement_targets)}."
        )
    return annotation_plan.placement_targets[tracker.current_index]


def _apply_online_placement_dot(
    env: MaskedSegmentationRenderEnv,
    obs: dict,
    tracker: OnlineNextObjectHighlightTracker | None,
    annotation_plan: OnlineNextObjectAnnotationPlan | None,
    args: Args,
) -> dict:
    if not args.next_object_placement_dot:
        return obs

    return draw_placement_dot_on_observations(
        env,
        obs,
        placement_target=_current_online_placement_target(tracker, annotation_plan),
        camera_names=_highlight_camera_names(args),
        dot_rgb=_parse_rgb_csv(args.placement_dot_rgb_csv),
        dot_alpha=float(args.placement_dot_alpha),
        dot_radius_px=int(args.placement_dot_radius_px),
    )


def eval_libero(args: Args) -> None:
    _validate_online_next_object_args(args)
    # Set random seed
    np.random.seed(args.seed)
    _init_wandb(args)
    run_started_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - keep Python 3.10 compatibility

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    selected_tasks = libero_utils.resolve_task_filters(
        task_suite_name=args.task_suite_name,
        task_indices=args.task_indices,
        task_names=args.task_names,
        task_split_file=args.task_split_file,
        task_split=args.task_split,
    )
    selected_task_set = {task.casefold() for task in selected_tasks}
    selected_task_ids = [
        task_id
        for task_id in range(num_tasks_in_suite)
        if not selected_task_set or task_suite.get_task(task_id).language.casefold() in selected_task_set
    ]
    if not selected_task_ids:
        raise ValueError("No LIBERO evaluation tasks matched the requested task filter.")
    task_descriptions_by_id = {task_id: task_suite.get_task(task_id).language for task_id in selected_task_ids}
    _define_wandb_metrics(selected_task_ids, task_descriptions_by_id)

    prompt_overrides: dict[str, str] = {}
    if args.prompt_override_file:
        with pathlib.Path(args.prompt_override_file).open() as f:
            override_data = json.load(f)
        prompt_overrides = {
            entry["task_instruction"]: entry["logic_task_description"] for entry in override_data["tasks"]
        }
        logging.info(f"Loaded {len(prompt_overrides)} prompt overrides from {args.prompt_override_file}")

    if args.video_out_path:
        video_out_path = pathlib.Path(args.video_out_path)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017 - keep Python 3.10 compatibility
        run_tag = args.task_suite_name
        if args.prompt_override_file:
            run_tag += "_logic"
        if args.next_object_highlighting:
            run_tag += "_next_object"
        if args.next_object_placement_dot:
            run_tag += "_placement_dot"
        video_out_path = pathlib.Path("data/libero/runs") / f"{timestamp}_{run_tag}"
    video_out_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Run output directory: {video_out_path}")

    results_out_path = pathlib.Path(args.results_out_path) if args.results_out_path else video_out_path / "results.json"
    results_out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_out_path = (
        pathlib.Path(args.progress_out_path) if args.progress_out_path else video_out_path / "eval_progress.json"
    )
    progress_out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_spatial_four_bowls":
        max_steps = 700
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []
    uploaded_video_categories: dict[int, set[str]] = collections.defaultdict(set)
    _persist_eval_outputs(
        args=args,
        run_started_at=run_started_at,
        selected_task_ids=selected_task_ids,
        selected_tasks=selected_tasks,
        video_out_path=video_out_path,
        results_out_path=results_out_path,
        progress_out_path=progress_out_path,
        task_results=task_results,
        total_episodes=total_episodes,
        total_successes=total_successes,
        status="running",
    )
    for task_id in tqdm.tqdm(selected_task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args)
        task_description = prompt_overrides.get(task_description, task_description)
        online_annotation_plan = None
        if args.next_object_highlighting:
            online_annotation_plan = _resolve_online_annotation_plan(task_suite, task_id, args.task_suite_name)

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_results = []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            episode_start_time = time.perf_counter()

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            highlight_tracker = None
            if args.next_object_highlighting:
                highlight_tracker = OnlineNextObjectHighlightTracker(
                    grasp_order=tuple(online_annotation_plan.grasp_order if online_annotation_plan is not None else ()),
                    min_grasp_steps=1,
                    release_steps=args.next_object_highlight_release_steps,
                )
                _apply_online_highlight_mask(env, highlight_tracker, args)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            if args.next_object_highlighting:
                obs = _apply_online_placement_dot(env, obs, highlight_tracker, online_annotation_plan, args)

            # Setup
            t = 0
            wait_steps_taken = 0
            policy_steps_taken = 0
            policy_inference_calls = 0
            replay_images = []
            done = False
            error = None
            error_stage = None
            termination_reason = "max_steps"
            episode_phase = "wait"

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        episode_phase = "wait"
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        if args.next_object_highlighting:
                            obs = _update_online_highlight_tracker(env, obs, highlight_tracker, args)
                            obs = _apply_online_placement_dot(env, obs, highlight_tracker, online_annotation_plan, args)
                        t += 1
                        wait_steps_taken += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        episode_phase = "policy_inference"
                        policy_inference_calls += 1
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    episode_phase = "policy_action"
                    obs, reward, done, info = env.step(action.tolist())
                    if args.next_object_highlighting:
                        obs = _update_online_highlight_tracker(env, obs, highlight_tracker, args)
                        obs = _apply_online_placement_dot(env, obs, highlight_tracker, online_annotation_plan, args)
                    t += 1
                    policy_steps_taken += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        termination_reason = "success"
                        break

                except Exception as e:
                    logging.error(f"Caught exception during {episode_phase}: {e}")
                    error = str(e)
                    error_stage = episode_phase
                    termination_reason = "error"
                    break
            else:
                termination_reason = "max_steps"

            task_episodes += 1
            total_episodes += 1
            episode_runtime_s = time.perf_counter() - episode_start_time
            _log_wandb_episode_metrics(
                task_id=task_id,
                task_description=task_description,
                episode_index=episode_idx,
                success=bool(done),
                steps_taken=t,
                wait_steps_taken=wait_steps_taken,
                policy_steps_taken=policy_steps_taken,
                policy_inference_calls=policy_inference_calls,
                episode_runtime_s=episode_runtime_s,
                had_error=error is not None,
                task_episodes=task_episodes,
                task_successes=task_successes,
                total_episodes=total_episodes,
                total_successes=total_successes,
                tasks_completed=len(task_results),
            )

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = video_out_path / f"rollout_{task_segment}_ep_{episode_idx:03d}_{suffix}.mp4"
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=REPLAY_VIDEO_FPS)
            episode_results.append(
                {
                    "episode_index": episode_idx,
                    "success": bool(done),
                    "steps_taken": t,
                    "wait_steps_taken": wait_steps_taken,
                    "policy_steps_taken": policy_steps_taken,
                    "policy_inference_calls": policy_inference_calls,
                    "reached_policy_step": policy_steps_taken > 0,
                    "termination_reason": termination_reason,
                    "error_stage": error_stage,
                    "video_path": str(video_path),
                    "error": error,
                }
            )
            if args.wandb_upload_episode_videos:
                _maybe_log_wandb_episode_video(
                    task_id=task_id,
                    task_description=task_description,
                    episode_index=episode_idx,
                    total_episodes=total_episodes,
                    success=bool(done),
                    had_error=error is not None,
                    video_path=video_path,
                    uploaded_video_categories=uploaded_video_categories,
                )

            _persist_eval_outputs(
                args=args,
                run_started_at=run_started_at,
                selected_task_ids=selected_task_ids,
                selected_tasks=selected_tasks,
                video_out_path=video_out_path,
                results_out_path=results_out_path,
                progress_out_path=progress_out_path,
                task_results=[
                    *task_results,
                    _build_task_result(
                        task_id=task_id,
                        task_description=task_description,
                        task_episodes=task_episodes,
                        task_successes=task_successes,
                        next_object_grasp_order=(
                            tuple(online_annotation_plan.grasp_order) if online_annotation_plan is not None else ()
                        ),
                        next_object_placement_targets=(
                            tuple(online_annotation_plan.placement_targets)
                            if online_annotation_plan is not None
                            else ()
                        ),
                        episode_results=episode_results,
                    ),
                ],
                total_episodes=total_episodes,
                total_successes=total_successes,
                status="running",
                active_task={
                    "task_id": task_id,
                    "task_description": task_description,
                    "task_episodes_completed": task_episodes,
                    "task_successes": task_successes,
                    "episode_index": episode_idx,
                },
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(
                "Episode summary: termination=%s total_steps=%d wait_steps=%d policy_steps=%d "
                "policy_inference_calls=%d error_stage=%s",
                termination_reason,
                t,
                wait_steps_taken,
                policy_steps_taken,
                policy_inference_calls,
                error_stage,
            )
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        _log_wandb_task_metrics(
            task_id=task_id,
            task_description=task_description,
            task_episodes=task_episodes,
            task_successes=task_successes,
            total_episodes=total_episodes,
            total_successes=total_successes,
            tasks_completed=len(task_results) + 1,
        )
        task_results.append(
            _build_task_result(
                task_id=task_id,
                task_description=task_description,
                task_episodes=task_episodes,
                task_successes=task_successes,
                next_object_grasp_order=(
                    tuple(online_annotation_plan.grasp_order) if online_annotation_plan is not None else ()
                ),
                next_object_placement_targets=(
                    tuple(online_annotation_plan.placement_targets) if online_annotation_plan is not None else ()
                ),
                episode_results=episode_results,
            )
        )
        _persist_eval_outputs(
            args=args,
            run_started_at=run_started_at,
            selected_task_ids=selected_task_ids,
            selected_tasks=selected_tasks,
            video_out_path=video_out_path,
            results_out_path=results_out_path,
            progress_out_path=progress_out_path,
            task_results=task_results,
            total_episodes=total_episodes,
            total_successes=total_successes,
            status="running",
        )

    final_payload = _persist_eval_outputs(
        args=args,
        run_started_at=run_started_at,
        selected_task_ids=selected_task_ids,
        selected_tasks=selected_tasks,
        video_out_path=video_out_path,
        results_out_path=results_out_path,
        progress_out_path=progress_out_path,
        task_results=task_results,
        total_episodes=total_episodes,
        total_successes=total_successes,
        status="completed",
    )
    total_success_rate = final_payload["total_success_rate"]
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Saved results JSON to {results_out_path}")
    if wandb.run is not None:
        wandb.summary["eval/total_success_rate"] = total_success_rate
        wandb.summary["eval/total_successes"] = total_successes
        wandb.summary["eval/total_episodes"] = total_episodes
        wandb.summary["eval/results_out_path"] = str(results_out_path)
        wandb.summary["eval/progress_out_path"] = str(progress_out_path)
        wandb.summary["eval/video_out_path"] = str(video_out_path)
        for task_result in task_results:
            task_slug = _slugify_metric_component(task_result["task_description"])
            wandb.summary[f"eval/task_success_rate/{task_slug}"] = task_result["success_rate"]

        _log_wandb_final_task_summary(task_results, total_episodes=total_episodes)

        artifact = wandb.Artifact(f"{wandb.run.id}-libero-eval-results", type="libero-eval-results")
        artifact.add_file(str(results_out_path), name="results.json")
        if progress_out_path.exists():
            artifact.add_file(str(progress_out_path), name="eval_progress.json")
        wandb.log_artifact(artifact)
        wandb.finish()


def _get_libero_env(task, resolution, seed, args: Args):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    masked_instances = _parse_csv_list(args.mask_instances_csv)
    if masked_instances and args.next_object_highlighting:
        raise ValueError("Use either static RGB masking or next-object highlighting, not both at once.")
    if masked_instances:
        env_args["masked_instance_names"] = masked_instances
        env_args["mask_rgb"] = _parse_rgb_csv(args.mask_rgb_csv)
        env_args["mask_alpha"] = float(args.mask_alpha)
        mask_cameras = _parse_csv_list(args.mask_cameras_csv)
        if mask_cameras:
            env_args["mask_camera_names"] = mask_cameras
        env = MaskedSegmentationRenderEnv(**env_args)
    elif args.next_object_highlighting:
        env = MaskedSegmentationRenderEnv(**env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
