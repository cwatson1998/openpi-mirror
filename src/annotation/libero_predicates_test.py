from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base_predicates_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "third_party/libero/libero/libero/envs/predicates/base_predicates.py"
    spec = importlib.util.spec_from_file_location("libero_base_predicates_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyObjectState:
    def __init__(self, *, grasped: bool, near_gripper: bool) -> None:
        self._grasped = grasped
        self._near_gripper = near_gripper

    def is_grasped(self) -> bool:
        return self._grasped

    def near_gripper(self, max_distance: float) -> bool:
        assert max_distance == 0.10
        return self._near_gripper


def test_grasped_predicate_calls_object_state_helper() -> None:
    module = _load_base_predicates_module()
    assert module.Grasped()(_DummyObjectState(grasped=True, near_gripper=False)) is True
    assert module.Grasped()(_DummyObjectState(grasped=False, near_gripper=True)) is False


def test_near_gripper_10cm_predicate_uses_10cm_threshold() -> None:
    module = _load_base_predicates_module()
    assert module.NearGripper10cm()(_DummyObjectState(grasped=False, near_gripper=True)) is True
    assert module.NearGripper10cm()(_DummyObjectState(grasped=True, near_gripper=False)) is False
