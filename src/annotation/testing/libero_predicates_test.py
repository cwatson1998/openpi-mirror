from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base_predicates_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "third_party/libero/libero/libero/envs/predicates/base_predicates.py"
    spec = importlib.util.spec_from_file_location("libero_base_predicates_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyObjectState:
    def __init__(
        self,
        *,
        grasped: bool,
        near_gripper: bool,
        centroid_y: float = 0.0,
    ) -> None:
        self._grasped = grasped
        self._near_gripper = near_gripper
        self._centroid_y = centroid_y

    def is_grasped(self) -> bool:
        return self._grasped

    def near_gripper(self, max_distance: float) -> bool:
        assert max_distance == 0.10
        return self._near_gripper

    def get_centroid_state(self) -> dict[str, tuple[float, float, float]]:
        return {"pos": (0.0, self._centroid_y, 0.0)}

    def get_axis_aligned_bounds(self) -> dict[str, tuple[float, float, float]]:
        return {
            "min": (0.0, self._centroid_y - 0.05, 0.0),
            "max": (0.0, self._centroid_y + 0.05, 0.0),
        }


def test_grasped_predicate_calls_object_state_helper() -> None:
    module = _load_base_predicates_module()
    assert module.Grasped()(_DummyObjectState(grasped=True, near_gripper=False)) is True
    assert module.Grasped()(_DummyObjectState(grasped=False, near_gripper=True)) is False


def test_near_gripper_10cm_predicate_uses_10cm_threshold() -> None:
    module = _load_base_predicates_module()
    assert module.NearGripper10cm()(_DummyObjectState(grasped=False, near_gripper=True)) is True
    assert module.NearGripper10cm()(_DummyObjectState(grasped=True, near_gripper=False)) is False


def test_near_uses_centroid_distance_threshold() -> None:
    module = _load_base_predicates_module()

    class _AxisState(_DummyObjectState):
        def __init__(self, *, x: float, y: float, z: float) -> None:
            super().__init__(grasped=False, near_gripper=False, centroid_y=y)
            self._pos = (x, y, z)

        def get_centroid_state(self) -> dict[str, tuple[float, float, float]]:
            return {"pos": self._pos}

    origin = _AxisState(x=0.0, y=0.0, z=0.0)
    close = _AxisState(x=0.03, y=0.04, z=0.0)
    far = _AxisState(x=0.2, y=0.0, z=0.0)

    assert module.Near(0.05)(origin, close) is True
    assert module.Near(0.049)(origin, close) is False
    assert module.Near(0.1)(origin, far) is False


def test_left_of_uses_strict_centroid_y_comparison() -> None:
    module = _load_base_predicates_module()
    left = _DummyObjectState(grasped=False, near_gripper=False, centroid_y=-0.2)
    right = _DummyObjectState(grasped=False, near_gripper=False, centroid_y=0.3)
    tied = _DummyObjectState(grasped=False, near_gripper=False, centroid_y=-0.2)

    assert module.LeftOf()(left, right) is True
    assert module.LeftOf()(right, left) is False
    assert module.LeftOf()(left, tied) is False


def test_fully_left_of_uses_axis_aligned_bounds_and_threshold() -> None:
    module = _load_base_predicates_module()

    class _BoundsState(_DummyObjectState):
        def __init__(self, *, min_y: float, max_y: float) -> None:
            super().__init__(grasped=False, near_gripper=False, centroid_y=(min_y + max_y) / 2.0)
            self._bounds = {
                "min": (0.0, min_y, 0.0),
                "max": (0.0, max_y, 0.0),
            }

        def get_axis_aligned_bounds(self) -> dict[str, tuple[float, float, float]]:
            return self._bounds

    clearly_left = _BoundsState(min_y=-0.50, max_y=-0.20)
    clearly_right = _BoundsState(min_y=0.10, max_y=0.40)
    touching_gap = _BoundsState(min_y=-0.10, max_y=0.00)

    assert module.FullyLeftOf()(clearly_left, clearly_right) is True
    assert module.FullyLeftOf()(touching_gap, clearly_right) is True
    assert module.FullyLeftOf()(clearly_right, clearly_left) is False
    assert module.FullyLeftOf(0.15)(touching_gap, clearly_right) is False


def test_directional_predicates_use_expected_world_axes() -> None:
    module = _load_base_predicates_module()

    class _AxisState(_DummyObjectState):
        def __init__(self, *, x: float, y: float, z: float) -> None:
            super().__init__(grasped=False, near_gripper=False, centroid_y=y)
            self._pos = (x, y, z)

        def get_centroid_state(self) -> dict[str, tuple[float, float, float]]:
            return {"pos": self._pos}

    origin = _AxisState(x=0.0, y=0.0, z=0.0)
    right = _AxisState(x=0.0, y=0.3, z=0.0)
    front = _AxisState(x=0.4, y=0.0, z=0.0)
    above = _AxisState(x=0.0, y=0.0, z=0.5)

    assert module.RightOf()(right, origin) is True
    assert module.FrontOf()(front, origin) is True
    assert module.Behind()(origin, front) is True
    assert module.Above()(above, origin) is True
    assert module.Below()(origin, above) is True
    assert module.RightOf(0.2)(right, origin) is True
    assert module.FrontOf(0.2)(front, origin) is True
    assert module.Above(0.2)(above, origin) is True


def test_fully_directional_predicates_use_axis_aligned_bounds() -> None:
    module = _load_base_predicates_module()

    class _BoundsState(_DummyObjectState):
        def __init__(
            self,
            *,
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            min_z: float,
            max_z: float,
        ) -> None:
            super().__init__(
                grasped=False,
                near_gripper=False,
                centroid_y=(min_y + max_y) / 2.0,
            )
            self._bounds = {
                "min": (min_x, min_y, min_z),
                "max": (max_x, max_y, max_z),
            }

        def get_axis_aligned_bounds(self) -> dict[str, tuple[float, float, float]]:
            return self._bounds

    left = _BoundsState(min_x=0.0, max_x=0.1, min_y=-0.5, max_y=-0.2, min_z=0.0, max_z=0.1)
    right = _BoundsState(min_x=0.0, max_x=0.1, min_y=0.2, max_y=0.5, min_z=0.0, max_z=0.1)
    behind = _BoundsState(min_x=-0.6, max_x=-0.3, min_y=0.0, max_y=0.1, min_z=0.0, max_z=0.1)
    front = _BoundsState(min_x=0.2, max_x=0.5, min_y=0.0, max_y=0.1, min_z=0.0, max_z=0.1)
    below = _BoundsState(min_x=0.0, max_x=0.1, min_y=0.0, max_y=0.1, min_z=-0.6, max_z=-0.2)
    above = _BoundsState(min_x=0.0, max_x=0.1, min_y=0.0, max_y=0.1, min_z=0.2, max_z=0.5)

    assert module.FullyRightOf()(right, left) is True
    assert module.FullyFrontOf()(front, behind) is True
    assert module.FullyBehind()(behind, front) is True
    assert module.FullyAbove()(above, below) is True
    assert module.FullyBelow()(below, above) is True
    assert module.FullyRightOf(0.1)(right, left) is True
    assert module.FullyFrontOf(0.1)(front, behind) is True
    assert module.FullyAbove(0.1)(above, below) is True
