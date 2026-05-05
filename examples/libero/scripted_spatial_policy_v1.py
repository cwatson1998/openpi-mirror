"""First-pass scripted policy for LIBERO spatial tasks.

This version snapshots the bowl and plate poses once at reset and then follows
fixed waypoints. It is intentionally kept as a baseline for comparing later
scripted policy iterations.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SnapshotWaypointConfig:
    grasp_offset: tuple[float, float, float] = (-0.012, 0.042, 0.025)
    position_gain: float = 18.0
    waypoint_tolerance: float = 0.012
    max_lift_z: float = 1.30


class SnapshotWaypointSpatialPolicy:
    """Simple waypoint policy that plans once from initial low-dimensional state."""

    def __init__(self, config: SnapshotWaypointConfig | None = None):
        self._config = config or SnapshotWaypointConfig()
        self._waypoints: list[tuple[np.ndarray, float, int]] = []
        self._waypoint_index = 0
        self._steps_on_waypoint = 0

    def reset(self, env, obs: dict) -> None:
        del obs
        bowl_pos = self._object_pos(env, "akita_black_bowl_1")
        plate_pos = self._object_pos(env, "plate_1")
        offset = np.asarray(self._config.grasp_offset, dtype=np.float64)
        lift_z = min(max(bowl_pos[2] + 0.25, plate_pos[2] + 0.25, 1.15), self._config.max_lift_z)

        self._waypoints = [
            (np.r_[bowl_pos[:2] + offset[:2], max(bowl_pos[2] + 0.22, 1.12)], -1.0, 70),
            (bowl_pos + offset, -1.0, 90),
            (bowl_pos + offset, 1.0, 20),
            (np.r_[bowl_pos[:2] + offset[:2], lift_z], 1.0, 70),
            (np.r_[plate_pos[:2] + offset[:2], lift_z], 1.0, 90),
            (plate_pos + offset, 1.0, 90),
            (plate_pos + offset, -1.0, 20),
            (np.r_[plate_pos[:2] + offset[:2], lift_z], -1.0, 40),
        ]
        self._waypoint_index = 0
        self._steps_on_waypoint = 0

    def action(self, env, obs: dict) -> np.ndarray:
        del env
        while self._waypoint_index < len(self._waypoints):
            target, gripper, max_steps = self._waypoints[self._waypoint_index]
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            delta = target - eef_pos
            if np.linalg.norm(delta) > self._config.waypoint_tolerance and self._steps_on_waypoint < max_steps:
                self._steps_on_waypoint += 1
                action = np.zeros(7, dtype=np.float64)
                action[:3] = np.clip(delta * self._config.position_gain, -1.0, 1.0)
                action[6] = gripper
                return action
            self._waypoint_index += 1
            self._steps_on_waypoint = 0

        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)

    @staticmethod
    def _object_pos(env, object_name: str) -> np.ndarray:
        return np.asarray(env.env.object_states_dict[object_name].get_geom_state()["pos"], dtype=np.float64)
