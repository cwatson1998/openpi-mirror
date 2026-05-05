"""Noisy scripted policy for generating varied LIBERO spatial demos.

This version intentionally leaves v2 untouched and wraps the same state-machine
structure with episode-local randomization. The noise is kept small enough to
preserve task success while producing less identical trajectories for imitation
learning data.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
from scripted_spatial_policy import DynamicSpatialPolicy
from scripted_spatial_policy import DynamicSpatialPolicyConfig
from scripted_spatial_policy import Phase


@dataclasses.dataclass(frozen=True)
class NoisySpatialPolicyConfig:
    base_policy: DynamicSpatialPolicyConfig = dataclasses.field(default_factory=DynamicSpatialPolicyConfig)
    table_grasp_xy_std: float = 0.006
    table_grasp_z_std: float = 0.004
    cabinet_grasp_xy_std: float = 0.0
    cabinet_grasp_z_std: float = 0.0
    place_xy_std: float = 0.008
    place_z_std: float = 0.003
    cabinet_place_xy_std: float = 0.003
    cabinet_place_z_std: float = 0.001
    lift_z_range: tuple[float, float] = (-0.01, 0.04)
    cabinet_lift_z_range: tuple[float, float] = (0.00, 0.025)
    gain_range: tuple[float, float] = (14.0, 20.0)
    cabinet_gain_range: tuple[float, float] = (16.0, 20.0)
    waypoint_tolerance_range: tuple[float, float] = (0.010, 0.018)
    cabinet_waypoint_tolerance_range: tuple[float, float] = (0.010, 0.014)
    close_step_jitter: int = 4
    open_step_jitter: int = 4
    action_noise_std: float = 0.018
    rotation_noise_std: float = 0.006
    cabinet_action_noise_std: float = 0.004
    cabinet_rotation_noise_std: float = 0.002
    action_noise_decay: float = 0.65


class NoisyDynamicSpatialPolicy(DynamicSpatialPolicy):
    """V3 policy: v2 pick-place behavior plus controlled random variation."""

    _MOVING_PHASES: ClassVar[set[Phase]] = {
        Phase.APPROACH_BOWL,
        Phase.DESCEND_TO_GRASP,
        Phase.LIFT_BOWL,
        Phase.MOVE_TO_PLATE,
        Phase.LOWER_TO_PLATE,
        Phase.RETREAT,
    }

    def __init__(
        self,
        *,
        rng: np.random.Generator | None = None,
        config: NoisySpatialPolicyConfig | None = None,
    ):
        self._noisy_config = config or NoisySpatialPolicyConfig()
        self._rng = rng or np.random.default_rng()
        self._waypoint_tolerance = self._noisy_config.base_policy.waypoint_tolerance
        self._close_steps = self._noisy_config.base_policy.close_steps
        self._open_steps = self._noisy_config.base_policy.open_steps
        self._action_noise_std = self._noisy_config.action_noise_std
        self._rotation_noise_std = self._noisy_config.rotation_noise_std
        self._smooth_action_noise = np.zeros(6, dtype=np.float64)
        super().__init__(self._noisy_config.base_policy)

    def reset(self, env, obs: dict) -> None:
        super().reset(env, obs)
        scene = self._read_scene(env, obs)
        is_cabinet_start = bool(scene.bowl_pos[2] > 1.02 and scene.bowl_pos[1] < -0.10)
        gain_range = self._noisy_config.cabinet_gain_range if is_cabinet_start else self._noisy_config.gain_range
        tolerance_range = (
            self._noisy_config.cabinet_waypoint_tolerance_range
            if is_cabinet_start
            else self._noisy_config.waypoint_tolerance_range
        )

        self._config = dataclasses.replace(
            self._config,
            position_gain=float(self._rng.uniform(*gain_range)),
        )
        self._waypoint_tolerance = float(self._rng.uniform(*tolerance_range))
        self._close_steps = self._jitter_steps(self._config.close_steps, self._noisy_config.close_step_jitter)
        self._open_steps = self._jitter_steps(self._config.open_steps, self._noisy_config.open_step_jitter)
        self._action_noise_std = (
            self._noisy_config.cabinet_action_noise_std if is_cabinet_start else self._noisy_config.action_noise_std
        )
        self._rotation_noise_std = (
            self._noisy_config.cabinet_rotation_noise_std if is_cabinet_start else self._noisy_config.rotation_noise_std
        )
        self._smooth_action_noise = np.zeros(6, dtype=np.float64)

        grasp_xy_std = (
            self._noisy_config.cabinet_grasp_xy_std if is_cabinet_start else self._noisy_config.table_grasp_xy_std
        )
        grasp_z_std = (
            self._noisy_config.cabinet_grasp_z_std if is_cabinet_start else self._noisy_config.table_grasp_z_std
        )
        place_xy_std = self._noisy_config.cabinet_place_xy_std if is_cabinet_start else self._noisy_config.place_xy_std
        place_z_std = self._noisy_config.cabinet_place_z_std if is_cabinet_start else self._noisy_config.place_z_std
        lift_z_range = self._noisy_config.cabinet_lift_z_range if is_cabinet_start else self._noisy_config.lift_z_range
        self._grasp_offset = self._grasp_offset + np.array(
            [
                self._rng.normal(0.0, grasp_xy_std),
                self._rng.normal(0.0, grasp_xy_std),
                self._rng.normal(0.0, grasp_z_std),
            ],
            dtype=np.float64,
        )
        self._place_offset = self._place_offset + np.array(
            [
                self._rng.normal(0.0, place_xy_std),
                self._rng.normal(0.0, place_xy_std),
                self._rng.normal(0.0, place_z_std),
            ],
            dtype=np.float64,
        )
        self._lift_z = min(
            self._config.max_lift_z,
            max(1.13, self._lift_z + float(self._rng.uniform(*lift_z_range))),
        )

    def action(self, env, obs: dict) -> np.ndarray:
        phase_before = self.phase
        action = super().action(env, obs)
        if phase_before not in self._MOVING_PHASES or self.phase is Phase.DONE:
            self._smooth_action_noise[:] = 0.0
            return action

        raw_noise = np.array(
            [
                *self._rng.normal(0.0, self._action_noise_std, size=3),
                *self._rng.normal(0.0, self._rotation_noise_std, size=3),
            ],
            dtype=np.float64,
        )
        self._smooth_action_noise = (
            self._noisy_config.action_noise_decay * self._smooth_action_noise
            + (1.0 - self._noisy_config.action_noise_decay) * raw_noise
        )
        action[:6] = np.clip(action[:6] + self._smooth_action_noise, -1.0, 1.0)
        return action

    def _hold_complete(self) -> bool:
        if self.phase is Phase.CLOSE_GRIPPER:
            return self._phase_step >= self._close_steps
        if self.phase is Phase.OPEN_GRIPPER:
            return self._phase_step >= self._open_steps
        return True

    def _target_reached(self, delta: np.ndarray) -> bool:
        return bool(np.linalg.norm(delta) <= self._waypoint_tolerance)

    def _jitter_steps(self, base_steps: int, jitter: int) -> int:
        if jitter <= 0:
            return base_steps
        return max(1, base_steps + int(self._rng.integers(-jitter, jitter + 1)))
