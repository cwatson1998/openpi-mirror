"""Evaluate scripted policies on the LIBERO spatial benchmark.

Example:
PYTHONPATH=third_party/libero MUJOCO_GL=osmesa examples/libero/.venv/bin/python \
  examples/libero/eval_scripted_spatial.py --args.num-trials-per-task 1

PYTHONPATH=third_party/libero MUJOCO_GL=osmesa examples/libero/.venv/bin/python \
  examples/libero/eval_scripted_spatial.py --args.task-suite-name libero_spatial_four_bowls \
  --args.policy-version v2 --args.num-trials-per-task 1
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from datetime import timezone
import json
import os
import pathlib
import time

import imageio
import numpy as np
from scripted_spatial_four_bowls_policy import FourBowlSpatialPolicy
from scripted_spatial_policy import DynamicSpatialPolicy
from scripted_spatial_policy_v1 import SnapshotWaypointSpatialPolicy
from scripted_spatial_policy_v3 import NoisyDynamicSpatialPolicy
import tyro

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

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"
    policy_version: str = "v2"
    task_indices: tuple[int, ...] = ()
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    max_steps: int = 220
    seed: int = 7
    camera_resolution: int = 128
    save_videos: bool = False
    output_dir: str | None = None
    next_object_highlighting: bool = False
    next_object_highlight_rgb_csv: str = "255,105,180"
    next_object_highlight_alpha: float = 1.0
    next_object_highlight_release_steps: int = 4
    next_object_placement_dot: bool = False
    placement_dot_rgb_csv: str = "0,96,255"
    placement_dot_alpha: float = 1.0
    placement_dot_radius_px: int = 5
    annotation_cameras_csv: str = ""


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
    if args.task_suite_name != "libero_spatial":
        raise ValueError(
            "Online next-object highlighting and placement dots are currently supported only for "
            f"`libero_spatial`, got `{args.task_suite_name}`."
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


def _annotation_camera_names(args: Args) -> list[str]:
    return _parse_csv_list(args.annotation_cameras_csv) or ["agentview", "robot0_eye_in_hand"]


def _make_policy(
    version: str,
    *,
    task_suite_name: str,
    task_language: str,
    rng: np.random.Generator,
):
    if version == "v1":
        return SnapshotWaypointSpatialPolicy()
    if version == "v2":
        if task_suite_name == "libero_spatial_four_bowls":
            return FourBowlSpatialPolicy(task_language=task_language)
        return DynamicSpatialPolicy()
    if version == "v3":
        return NoisyDynamicSpatialPolicy(rng=rng)
    raise ValueError(f"Unknown scripted policy version: {version}")


def _task_bddl_path(task) -> pathlib.Path:
    return pathlib.Path("third_party/libero/libero/libero/bddl_files") / task.problem_folder / task.bddl_file


def _make_env(task, args: Args):
    if args.next_object_highlighting:
        from libero.libero.envs import MaskedSegmentationRenderEnv

        return MaskedSegmentationRenderEnv(
            bddl_file_name=str(_task_bddl_path(task)),
            camera_names=_annotation_camera_names(args),
            camera_heights=args.camera_resolution,
            camera_widths=args.camera_resolution,
            use_camera_obs=True,
        )

    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=str(_task_bddl_path(task)),
        camera_heights=args.camera_resolution,
        camera_widths=args.camera_resolution,
        use_camera_obs=args.save_videos,
    )


def _resolve_online_annotation_plan(task_suite, task_id: int):
    from libero.libero import get_libero_path

    from annotation import build_online_next_object_annotation_plan_from_source_demo

    datasets_root = pathlib.Path(get_libero_path("datasets"))
    source_demo_path = datasets_root / task_suite.get_task_demonstration(task_id)
    return build_online_next_object_annotation_plan_from_source_demo(source_demo_path)


def _apply_online_highlight_mask(env, tracker, args: Args) -> None:
    if tracker is None or tracker.current_object is None:
        env.clear_instance_mask()
        return

    env.set_instance_mask(
        tracker.current_object,
        mask_rgb=_parse_rgb_csv(args.next_object_highlight_rgb_csv),
        mask_alpha=float(args.next_object_highlight_alpha),
        camera_names=_annotation_camera_names(args),
    )


def _update_online_highlight_tracker(env, obs: dict, tracker, args: Args) -> dict:
    if tracker is None or tracker.current_object is None:
        return obs

    current_object = tracker.current_object
    if current_object not in env.env.object_states_dict:
        raise KeyError(f"Current highlighted object `{current_object}` is missing from the live environment.")

    advanced = tracker.observe(is_current_object_grasped=bool(env.env.object_states_dict[current_object].is_grasped()))
    if not advanced:
        return obs

    _apply_online_highlight_mask(env, tracker, args)
    return env.regenerate_obs_from_state(env.get_sim_state())


def _current_online_placement_target(tracker, annotation_plan) -> tuple[float, float, float] | None:
    if tracker is None or annotation_plan is None or tracker.current_object is None:
        return None
    if tracker.current_index >= len(annotation_plan.placement_targets):
        raise IndexError(
            "The online next-object tracker advanced beyond the demo-derived placement target sequence: "
            f"index={tracker.current_index}, targets={len(annotation_plan.placement_targets)}."
        )
    return annotation_plan.placement_targets[tracker.current_index]


def _apply_online_placement_dot(env, obs: dict, tracker, annotation_plan, args: Args) -> dict:
    if not args.next_object_placement_dot:
        return obs

    from annotation import draw_placement_dot_on_observations

    return draw_placement_dot_on_observations(
        env,
        obs,
        placement_target=_current_online_placement_target(tracker, annotation_plan),
        camera_names=_annotation_camera_names(args),
        dot_rgb=_parse_rgb_csv(args.placement_dot_rgb_csv),
        dot_alpha=float(args.placement_dot_alpha),
        dot_radius_px=int(args.placement_dot_radius_px),
    )


def eval_scripted_spatial(args: Args) -> dict:
    from libero.libero import benchmark

    _validate_online_next_object_args(args)
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = list(args.task_indices) if args.task_indices else list(range(task_suite.n_tasks))
    run_started_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - keep Python 3.10 compatibility
    output_dir = pathlib.Path(
        args.output_dir or f"data/libero/runs/{run_started_at}_{args.task_suite_name}_{args.policy_version}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = 700 if args.task_suite_name == "libero_spatial_four_bowls" and args.max_steps == 220 else args.max_steps

    task_results = []
    total_episodes = 0
    total_successes = 0
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_successes = 0
        episode_results = []
        online_annotation_plan = None
        if args.next_object_highlighting:
            from annotation import OnlineNextObjectHighlightTracker

            online_annotation_plan = _resolve_online_annotation_plan(task_suite, task_id)
        env = _make_env(task, args)
        env.seed(args.seed)
        try:
            for episode_index in range(args.num_trials_per_task):
                start_time = time.perf_counter()
                highlight_tracker = None
                obs = env.reset()
                if args.next_object_highlighting:
                    highlight_tracker = OnlineNextObjectHighlightTracker(
                        grasp_order=tuple(
                            online_annotation_plan.grasp_order if online_annotation_plan is not None else ()
                        ),
                        min_grasp_steps=1,
                        release_steps=args.next_object_highlight_release_steps,
                    )
                    _apply_online_highlight_mask(env, highlight_tracker, args)
                obs = env.set_init_state(initial_states[episode_index])
                if args.next_object_highlighting:
                    obs = _apply_online_placement_dot(env, obs, highlight_tracker, online_annotation_plan, args)
                policy_rng = np.random.default_rng(args.seed + task_id * 10_000 + episode_index)
                policy = _make_policy(
                    args.policy_version,
                    task_suite_name=args.task_suite_name,
                    task_language=task.language,
                    rng=policy_rng,
                )
                replay_images = []
                done = False

                for _ in range(args.num_steps_wait):
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    if args.next_object_highlighting:
                        obs = _update_online_highlight_tracker(env, obs, highlight_tracker, args)
                        obs = _apply_online_placement_dot(env, obs, highlight_tracker, online_annotation_plan, args)
                    if done:
                        break

                policy.reset(env, obs)
                policy_steps = 0
                while not done and policy_steps < max_steps:
                    if args.save_videos and "agentview_image" in obs:
                        replay_images.append(np.asarray(obs["agentview_image"][::-1, ::-1]))
                    action = policy.action(env, obs)
                    obs, _, done, _ = env.step(action.tolist())
                    if args.next_object_highlighting:
                        obs = _update_online_highlight_tracker(env, obs, highlight_tracker, args)
                        obs = _apply_online_placement_dot(env, obs, highlight_tracker, online_annotation_plan, args)
                    policy_steps += 1

                success = bool(done or env.check_success())
                task_successes += int(success)
                total_successes += int(success)
                total_episodes += 1
                video_path = None
                if args.save_videos and replay_images:
                    suffix = "success" if success else "failure"
                    task_slug = task.language.replace(" ", "_")
                    video_path = output_dir / f"rollout_{task_id:02d}_{task_slug}_ep_{episode_index:03d}_{suffix}.mp4"
                    imageio.mimwrite(video_path, replay_images, fps=10)

                result = {
                    "episode_index": episode_index,
                    "success": success,
                    "policy_steps": policy_steps,
                    "runtime_s": time.perf_counter() - start_time,
                    "video_path": str(video_path) if video_path is not None else None,
                }
                episode_results.append(result)
                print(
                    f"task={task_id} episode={episode_index} success={success} "
                    f"policy_steps={policy_steps} version={args.policy_version}",
                    flush=True,
                )
        finally:
            env.close()

        task_results.append(
            {
                "task_id": task_id,
                "task_description": task.language,
                "episodes": args.num_trials_per_task,
                "successes": task_successes,
                "success_rate": task_successes / args.num_trials_per_task,
                "next_object_grasp_order": (
                    list(online_annotation_plan.grasp_order) if online_annotation_plan is not None else []
                ),
                "next_object_placement_targets": (
                    [
                        None if target is None else [float(value) for value in target]
                        for target in online_annotation_plan.placement_targets
                    ]
                    if online_annotation_plan is not None
                    else []
                ),
                "episode_results": episode_results,
            }
        )

    payload = {
        "schema_version": 1,
        "status": "completed",
        "generated_at": run_started_at,
        "task_suite_name": args.task_suite_name,
        "policy_version": args.policy_version,
        "next_object_highlighting": args.next_object_highlighting,
        "next_object_highlight_rgb_csv": args.next_object_highlight_rgb_csv,
        "next_object_highlight_alpha": args.next_object_highlight_alpha,
        "next_object_highlight_release_steps": args.next_object_highlight_release_steps,
        "next_object_placement_dot": args.next_object_placement_dot,
        "placement_dot_rgb_csv": args.placement_dot_rgb_csv,
        "placement_dot_alpha": args.placement_dot_alpha,
        "placement_dot_radius_px": args.placement_dot_radius_px,
        "task_indices": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "task_results": task_results,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"total_success_rate={payload['total_success_rate']:.3f} results={results_path}", flush=True)
    return payload


if __name__ == "__main__":
    tyro.cli(eval_scripted_spatial)
