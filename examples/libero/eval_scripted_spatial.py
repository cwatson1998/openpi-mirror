"""Evaluate scripted policies on the LIBERO spatial benchmark.

Example:
PYTHONPATH=third_party/libero MUJOCO_GL=osmesa examples/libero/.venv/bin/python \
  examples/libero/eval_scripted_spatial.py --args.num-trials-per-task 1
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from datetime import timezone
import json
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from scripted_spatial_policy import DynamicSpatialPolicy
from scripted_spatial_policy_v1 import SnapshotWaypointSpatialPolicy
from scripted_spatial_policy_v3 import NoisyDynamicSpatialPolicy
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    policy_version: str = "v2"
    task_indices: tuple[int, ...] = ()
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
    max_steps: int = 220
    seed: int = 7
    camera_resolution: int = 128
    save_videos: bool = False
    output_dir: str | None = None


def _make_policy(version: str, *, rng: np.random.Generator):
    if version == "v1":
        return SnapshotWaypointSpatialPolicy()
    if version == "v2":
        return DynamicSpatialPolicy()
    if version == "v3":
        return NoisyDynamicSpatialPolicy(rng=rng)
    raise ValueError(f"Unknown scripted policy version: {version}")


def _task_bddl_path(task) -> pathlib.Path:
    return pathlib.Path("third_party/libero/libero/libero/bddl_files") / task.problem_folder / task.bddl_file


def _make_env(task, args: Args) -> OffScreenRenderEnv:
    return OffScreenRenderEnv(
        bddl_file_name=str(_task_bddl_path(task)),
        camera_heights=args.camera_resolution,
        camera_widths=args.camera_resolution,
        use_camera_obs=args.save_videos,
    )


def eval_scripted_spatial(args: Args) -> dict:
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_ids = list(args.task_indices) if args.task_indices else list(range(task_suite.n_tasks))
    run_started_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - keep Python 3.10 compatibility
    output_dir = pathlib.Path(
        args.output_dir or f"data/libero/runs/{run_started_at}_scripted_spatial_{args.policy_version}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task_results = []
    total_episodes = 0
    total_successes = 0
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_successes = 0
        episode_results = []
        env = _make_env(task, args)
        env.seed(args.seed)
        try:
            for episode_index in range(args.num_trials_per_task):
                start_time = time.perf_counter()
                obs = env.reset()
                obs = env.set_init_state(initial_states[episode_index])
                policy_rng = np.random.default_rng(args.seed + task_id * 10_000 + episode_index)
                policy = _make_policy(args.policy_version, rng=policy_rng)
                replay_images = []
                done = False

                for _ in range(args.num_steps_wait):
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    if done:
                        break

                policy.reset(env, obs)
                policy_steps = 0
                while not done and policy_steps < args.max_steps:
                    if args.save_videos and "agentview_image" in obs:
                        replay_images.append(np.asarray(obs["agentview_image"][::-1, ::-1]))
                    action = policy.action(env, obs)
                    obs, _, done, _ = env.step(action.tolist())
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
                "episode_results": episode_results,
            }
        )

    payload = {
        "schema_version": 1,
        "status": "completed",
        "generated_at": run_started_at,
        "policy_version": args.policy_version,
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
