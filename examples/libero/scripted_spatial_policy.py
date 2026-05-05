"""Dynamic scripted policy for the LIBERO spatial benchmark.

The policy uses only simulator state: end-effector proprioception from the
observation dict plus live object states from LIBERO's object_states_dict.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar

import numpy as np


class Phase(enum.Enum):
    APPROACH_BOWL = enum.auto()
    DESCEND_TO_GRASP = enum.auto()
    CLOSE_GRIPPER = enum.auto()
    LIFT_BOWL = enum.auto()
    MOVE_TO_PLATE = enum.auto()
    LOWER_TO_PLATE = enum.auto()
    OPEN_GRIPPER = enum.auto()
    RETREAT = enum.auto()
    DONE = enum.auto()


@dataclasses.dataclass(frozen=True)
class DynamicSpatialPolicyConfig:
    grasp_offset: tuple[float, float, float] = (-0.012, 0.042, 0.025)
    cabinet_grasp_offset: tuple[float, float, float] = (-0.035, 0.030, 0.032)
    cabinet_top_grasp_offset: tuple[float, float, float] = (-0.045, 0.035, 0.025)
    cabinet_top_right_grasp_offset: tuple[float, float, float] = (-0.040, 0.035, 0.025)
    position_gain: float = 18.0
    waypoint_tolerance: float = 0.012
    close_steps: int = 20
    open_steps: int = 20
    max_lift_z: float = 1.30


@dataclasses.dataclass(frozen=True)
class SpatialSceneState:
    eef_pos: np.ndarray
    bowl_pos: np.ndarray
    plate_pos: np.ndarray
    success: bool


class DynamicSpatialPolicy:
    """State-machine pick-and-place policy for all LIBERO spatial tasks."""

    _PHASE_LIMITS: ClassVar[dict[Phase, int]] = {
        Phase.APPROACH_BOWL: 70,
        Phase.DESCEND_TO_GRASP: 90,
        Phase.CLOSE_GRIPPER: 20,
        Phase.LIFT_BOWL: 70,
        Phase.MOVE_TO_PLATE: 90,
        Phase.LOWER_TO_PLATE: 90,
        Phase.OPEN_GRIPPER: 20,
        Phase.RETREAT: 40,
    }

    def __init__(self, config: DynamicSpatialPolicyConfig | None = None):
        self._config = config or DynamicSpatialPolicyConfig()
        self._phase = Phase.APPROACH_BOWL
        self._phase_step = 0
        self._lift_z = 1.15
        self._grasp_offset = np.asarray(self._config.grasp_offset, dtype=np.float64)
        self._place_offset = np.asarray(self._config.grasp_offset, dtype=np.float64)

    @property
    def phase(self) -> Phase:
        return self._phase

    def reset(self, env, obs: dict) -> None:
        scene = self._read_scene(env, obs)
        self._phase = Phase.APPROACH_BOWL
        self._phase_step = 0
        self._lift_z = self._compute_lift_z(scene)
        self._grasp_offset = self._select_grasp_offset(scene)
        self._place_offset = np.asarray(self._config.grasp_offset, dtype=np.float64)

    def action(self, env, obs: dict) -> np.ndarray:
        scene = self._read_scene(env, obs)
        if scene.success:
            self._phase = Phase.DONE

        for _ in range(len(Phase)):
            if self._phase is Phase.DONE:
                return self._zero_action(gripper=-1.0)

            target = self._target_for_phase(scene)
            gripper = self._gripper_for_phase()
            if target is None:
                self._phase_step += 1
                if self._hold_complete():
                    self._advance_phase()
                return self._zero_action(gripper=gripper)

            delta = target - scene.eef_pos
            if self._target_reached(delta) or self._phase_timed_out():
                self._advance_phase()
                continue

            self._phase_step += 1
            action = np.zeros(7, dtype=np.float64)
            action[:3] = np.clip(delta * self._config.position_gain, -1.0, 1.0)
            action[6] = gripper
            return action

        return self._zero_action(gripper=-1.0)

    def _target_for_phase(self, scene: SpatialSceneState) -> np.ndarray | None:
        offset = self._grasp_offset
        bowl_grasp = scene.bowl_pos + offset
        plate_place = scene.plate_pos + self._place_offset

        if self._phase is Phase.APPROACH_BOWL:
            return np.r_[bowl_grasp[:2], max(scene.bowl_pos[2] + 0.22, 1.12)]
        if self._phase is Phase.DESCEND_TO_GRASP:
            return bowl_grasp
        if self._phase is Phase.CLOSE_GRIPPER:
            return None
        if self._phase is Phase.LIFT_BOWL:
            return np.r_[scene.eef_pos[:2], self._lift_z]
        if self._phase is Phase.MOVE_TO_PLATE:
            return np.r_[scene.plate_pos[:2] + self._place_offset[:2], self._lift_z]
        if self._phase is Phase.LOWER_TO_PLATE:
            return plate_place
        if self._phase is Phase.OPEN_GRIPPER:
            return None
        if self._phase is Phase.RETREAT:
            return np.r_[scene.plate_pos[:2] + self._place_offset[:2], self._lift_z]
        return None

    def _gripper_for_phase(self) -> float:
        if self._phase in {
            Phase.CLOSE_GRIPPER,
            Phase.LIFT_BOWL,
            Phase.MOVE_TO_PLATE,
            Phase.LOWER_TO_PLATE,
        }:
            return 1.0
        return -1.0

    def _hold_complete(self) -> bool:
        if self._phase is Phase.CLOSE_GRIPPER:
            return self._phase_step >= self._config.close_steps
        if self._phase is Phase.OPEN_GRIPPER:
            return self._phase_step >= self._config.open_steps
        return True

    def _target_reached(self, delta: np.ndarray) -> bool:
        return bool(np.linalg.norm(delta) <= self._config.waypoint_tolerance)

    def _phase_timed_out(self) -> bool:
        return self._phase_step >= self._PHASE_LIMITS[self._phase]

    def _advance_phase(self) -> None:
        phase_order = list(Phase)
        self._phase = phase_order[min(phase_order.index(self._phase) + 1, len(phase_order) - 1)]
        self._phase_step = 0

    def _compute_lift_z(self, scene: SpatialSceneState) -> float:
        return min(
            max(scene.bowl_pos[2] + 0.25, scene.plate_pos[2] + 0.25, 1.15),
            self._config.max_lift_z,
        )

    def _select_grasp_offset(self, scene: SpatialSceneState) -> np.ndarray:
        # The two wooden-cabinet starts require the side-entry grasp used by
        # the demos; the default table grasp bumps into the cabinet face.
        if scene.bowl_pos[2] > 1.02 and scene.bowl_pos[1] < -0.20:
            if scene.bowl_pos[0] > 0.05:
                return np.asarray(self._config.cabinet_top_right_grasp_offset, dtype=np.float64)
            return np.asarray(self._config.cabinet_top_grasp_offset, dtype=np.float64)
        if scene.bowl_pos[2] > 1.02 and scene.bowl_pos[1] < -0.10:
            return np.asarray(self._config.cabinet_grasp_offset, dtype=np.float64)
        return np.asarray(self._config.grasp_offset, dtype=np.float64)

    @staticmethod
    def _read_scene(env, obs: dict) -> SpatialSceneState:
        object_states = env.env.object_states_dict
        return SpatialSceneState(
            eef_pos=np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            bowl_pos=np.asarray(object_states["akita_black_bowl_1"].get_geom_state()["pos"], dtype=np.float64),
            plate_pos=np.asarray(object_states["plate_1"].get_geom_state()["pos"], dtype=np.float64),
            success=bool(env.check_success()),
        )

    @staticmethod
    def _zero_action(*, gripper: float) -> np.ndarray:
        action = np.zeros(7, dtype=np.float64)
        action[6] = gripper
        return action


def create_policy(version: str) -> DynamicSpatialPolicy:
    if version != "v2":
        raise ValueError(f"scripted_spatial_policy.py only provides v2, got {version!r}.")
    return DynamicSpatialPolicy()
