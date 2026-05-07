from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from annotation.libero_demo_replay import NextObjectHighlightPlan
from annotation.libero_demo_replay import OnlineNextObjectHighlightTracker
from annotation.libero_demo_replay import RldsEpisode
from annotation.libero_demo_replay import _backfill_next_highlight_targets
from annotation.libero_demo_replay import _compute_placement_targets_by_timestep
from annotation.libero_demo_replay import _parse_rgb_triplet
from annotation.libero_demo_replay import _select_grasped_object
from annotation.libero_demo_replay import build_grasp_sequence
from annotation.libero_demo_replay import build_online_next_object_annotation_plan
from annotation.libero_demo_replay import draw_placement_dot_on_observations
from annotation.libero_demo_replay import match_demo_key
from annotation.libero_demo_replay import resolve_demo_hdf5_path
from annotation.libero_demo_replay import trace_bddl_file_resolution
from annotation.libero_demo_replay import trace_demo_hdf5_resolution
from annotation.libero_demo_replay import write_rlds_episode_visualization
from annotation.libero_episode_sanity_check import make_sanity_check_frames
from annotation.libero_masked_replay_visualization import make_mask_comparison_frames


def _write_demo(
    data_group: h5py.Group,
    demo_key: str,
    *,
    actions: np.ndarray,
    joint_states: np.ndarray,
    state: np.ndarray,
) -> None:
    demo_group = data_group.create_group(demo_key)
    demo_group.attrs["model_file"] = "<mujoco/>"
    demo_group.create_dataset("actions", data=actions)
    demo_group.create_dataset(
        "states", data=np.arange(actions.shape[0] * 4, dtype=np.float32).reshape(actions.shape[0], 4)
    )

    obs_group = demo_group.create_group("obs")
    obs_group.create_dataset("joint_states", data=joint_states)
    obs_group.create_dataset("ee_states", data=state[:, :6])
    obs_group.create_dataset("gripper_states", data=state[:, 6:])


def test_match_demo_key_uses_actions_and_proprio(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5"
    with h5py.File(hdf5_path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        data_group.attrs["bddl_file_name"] = "/tmp/task.bddl"

        _write_demo(
            data_group,
            "demo_0",
            actions=np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
            joint_states=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            state=np.array([[2.0] * 8, [2.1] * 8], dtype=np.float32),
        )
        _write_demo(
            data_group,
            "demo_1",
            actions=np.array([[3.0, 3.1], [3.2, 3.3]], dtype=np.float32),
            joint_states=np.array([[4.0, 4.1], [4.2, 4.3]], dtype=np.float32),
            state=np.array([[5.0] * 8, [5.1] * 8], dtype=np.float32),
        )

    episode = RldsEpisode(
        dataset_name="libero_spatial_no_noops",
        episode_index=7,
        source_demo_path_hint="/remote/path/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
        task_instruction="pick up the black bowl next to the cookie box and place it on the plate",
        actions=np.array([[3.0, 3.1], [3.2, 3.3]], dtype=np.float32),
        joint_states=np.array([[4.0, 4.1], [4.2, 4.3]], dtype=np.float32),
        state=np.array([[5.0] * 8, [5.1] * 8], dtype=np.float32),
    )

    demo_key, metrics = match_demo_key(hdf5_path, episode)

    assert demo_key == "demo_1"
    assert metrics["length_delta"] == 0
    assert metrics["action_max_abs_err"] == 0.0
    assert metrics["joint_max_abs_err"] == 0.0
    assert metrics["state_max_abs_err"] == 0.0


def test_match_demo_key_accepts_unique_proprio_match_when_actions_differ(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
    with h5py.File(hdf5_path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        data_group.attrs["bddl_file_name"] = "/tmp/task.bddl"

        _write_demo(
            data_group,
            "demo_0",
            actions=np.array([[0.0, -1.0], [0.1, -1.0]], dtype=np.float32),
            joint_states=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            state=np.array([[2.0] * 8, [2.1] * 8], dtype=np.float32),
        )
        _write_demo(
            data_group,
            "demo_1",
            actions=np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
            joint_states=np.array([[8.0, 8.1], [8.2, 8.3]], dtype=np.float32),
            state=np.array([[9.0] * 8, [9.1] * 8], dtype=np.float32),
        )

    episode = RldsEpisode(
        dataset_name="libero_spatial_no_noops",
        episode_index=22,
        source_demo_path_hint="/remote/path/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
        task_instruction="pick up the black bowl from table center and place it on the plate",
        actions=np.array([[0.0, 1.0], [0.1, 1.0]], dtype=np.float32),
        joint_states=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
        state=np.array([[2.0] * 8, [2.1] * 8], dtype=np.float32),
    )

    demo_key, metrics = match_demo_key(hdf5_path, episode)

    assert demo_key == "demo_0"
    assert metrics["length_delta"] == 0
    assert metrics["action_max_abs_err"] == 2.0
    assert metrics["joint_max_abs_err"] == 0.0
    assert metrics["state_max_abs_err"] == 0.0


def test_match_demo_key_accepts_small_length_mismatch_using_prefix_metrics(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
    with h5py.File(hdf5_path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        data_group.attrs["bddl_file_name"] = "/tmp/task.bddl"

        _write_demo(
            data_group,
            "demo_0",
            actions=np.array([[0.0, -1.0], [0.1, -1.0], [0.2, -1.0]], dtype=np.float32),
            joint_states=np.array([[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]], dtype=np.float32),
            state=np.array([[2.0] * 8, [2.1] * 8, [2.2] * 8], dtype=np.float32),
        )
        _write_demo(
            data_group,
            "demo_1",
            actions=np.array([[0.0, -1.0], [0.1, -1.0], [0.2, -1.0]], dtype=np.float32),
            joint_states=np.array([[4.0, 4.1], [4.2, 4.3], [4.4, 4.5]], dtype=np.float32),
            state=np.array([[5.0] * 8, [5.1] * 8, [5.2] * 8], dtype=np.float32),
        )

    episode = RldsEpisode(
        dataset_name="libero_spatial_no_noops",
        episode_index=28,
        source_demo_path_hint="/remote/path/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
        task_instruction="pick up the black bowl from table center and place it on the plate",
        actions=np.array([[0.0, 1.0], [0.1, 1.0]], dtype=np.float32),
        joint_states=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
        state=np.array([[2.0] * 8, [2.1] * 8], dtype=np.float32),
    )

    demo_key, metrics = match_demo_key(hdf5_path, episode, joint_tolerance=0.5, state_tolerance=0.5)

    assert demo_key == "demo_0"
    assert metrics["length_delta"] == 1
    assert metrics["action_max_abs_err"] == 2.0
    assert metrics["joint_max_abs_err"] == 0.0
    assert metrics["state_max_abs_err"] == 0.0


def test_resolve_demo_hdf5_path_finds_basename_under_search_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "libero_spatial"
    dataset_root.mkdir()
    demo_path = dataset_root / "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    demo_path.write_bytes(b"")

    resolved = resolve_demo_hdf5_path(
        "/remote/cache/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
        dataset_name="libero_spatial_no_noops",
        demo_search_roots=[tmp_path],
    )

    assert resolved == demo_path.resolve()


def test_trace_demo_hdf5_resolution_records_checked_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "libero_spatial"
    dataset_root.mkdir()
    demo_path = dataset_root / "task_demo.hdf5"
    demo_path.write_bytes(b"")

    trace = trace_demo_hdf5_resolution(
        "/remote/cache/libero_spatial/task_demo.hdf5",
        dataset_name="libero_spatial_no_noops",
        demo_search_roots=[tmp_path],
    )

    assert trace.resolved_path == demo_path.resolve()
    assert trace.checked_paths[0] == tmp_path / "task_demo.hdf5"
    assert demo_path in trace.checked_paths


def test_trace_bddl_file_resolution_records_checked_paths(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    bddl_path = repo_root / "third_party/libero/libero/libero/bddl_files/libero_spatial/task.bddl"
    bddl_path.parent.mkdir(parents=True)
    bddl_path.write_text("(define (problem task))\n")
    monkeypatch.setattr("annotation.libero_demo_replay._repo_root", lambda: repo_root)

    trace = trace_bddl_file_resolution("task.bddl")

    assert trace.resolved_path == bddl_path.resolve()
    assert trace.checked_paths[0] == repo_root / "task.bddl"
    assert bddl_path in trace.checked_paths


def test_write_rlds_episode_visualization_writes_pngs_and_manifest(tmp_path: Path) -> None:
    episode = RldsEpisode(
        dataset_name="libero_spatial_no_noops",
        episode_index=0,
        source_demo_path_hint="/remote/demo.hdf5",
        task_instruction="pick up the black bowl next to the cookie box and place it on the plate",
        actions=np.zeros((2, 7), dtype=np.float32),
        image_frames=np.stack(
            [
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.full((4, 4, 3), 255, dtype=np.uint8),
            ],
            axis=0,
        ),
        wrist_image_frames=np.stack(
            [
                np.full((4, 4, 3), 7, dtype=np.uint8),
                np.full((4, 4, 3), 9, dtype=np.uint8),
            ],
            axis=0,
        ),
    )

    manifest_path = write_rlds_episode_visualization(
        episode,
        output_dir=tmp_path / "rlds_ep",
        camera_name="image",
    )

    assert manifest_path.exists()
    assert (tmp_path / "rlds_ep" / "00000.png").exists()
    assert (tmp_path / "rlds_ep" / "00001.png").exists()


def test_make_sanity_check_frames_stacks_and_labels_frames() -> None:
    left = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    right = np.full((2, 8, 10, 3), 255, dtype=np.uint8)

    combined = make_sanity_check_frames(
        rlds_frames=left,
        simulator_frames=right,
        dataset_name="libero_spatial_no_noops",
        episode_index=0,
        task_instruction="pick up the black bowl next to the cookie box and place it on the plate",
    )

    assert combined.shape == (2, 8 + 66, 20, 3)
    assert np.all(combined[:, 66:, :9] == 0)
    assert np.all(combined[:, 66:, 11:] == 255)
    assert np.any(combined[:, :66] != 18)


def test_make_sanity_check_frames_supports_optional_masked_panel() -> None:
    left = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    middle = np.full((2, 8, 10, 3), 127, dtype=np.uint8)
    right = np.full((2, 8, 10, 3), 255, dtype=np.uint8)

    combined = make_sanity_check_frames(
        rlds_frames=left,
        simulator_frames=middle,
        masked_simulator_frames=right,
        dataset_name="libero_spatial_no_noops",
        episode_index=0,
        task_instruction="pick up the black bowl next to the cookie box and place it on the plate",
    )

    assert combined.shape == (2, 8 + 66, 30, 3)
    assert np.all(combined[:, 66:, 1:9] == 0)
    assert np.all(combined[:, 66:, 12:18] == 127)
    assert np.all(combined[:, 66:, 22:28] == 255)
    assert np.any(combined[:, :66] != 18)


def test_parse_rgb_triplet_accepts_string_and_sequence() -> None:
    assert _parse_rgb_triplet("1,2,3") == (1, 2, 3)
    assert _parse_rgb_triplet([4, 5, 6]) == (4, 5, 6)


def test_select_grasped_object_uses_candidate_priority_order() -> None:
    selected = _select_grasped_object(
        ["akita_black_bowl_1", "plate_1", "cookie_box_1"],
        {
            "plate_1": True,
            "akita_black_bowl_1": True,
        },
    )

    assert selected == "akita_black_bowl_1"


def test_backfill_next_highlight_targets_uses_current_or_next_grasp() -> None:
    highlighted = _backfill_next_highlight_targets([None, None, "akita_black_bowl_1", None, "plate_1", None])

    assert highlighted == (
        "akita_black_bowl_1",
        "akita_black_bowl_1",
        "akita_black_bowl_1",
        "plate_1",
        "plate_1",
        None,
    )


def test_compute_placement_targets_uses_release_or_final_position() -> None:
    grasped = [None, "akita_black_bowl_1", "akita_black_bowl_1", None, "plate_1", "plate_1"]
    targets = _compute_placement_targets_by_timestep(
        grasped_object_by_timestep=grasped,
        highlighted_object_by_timestep=_backfill_next_highlight_targets(grasped),
        object_positions_by_timestep={
            "akita_black_bowl_1": [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (5.0, 0.0, 0.0),
            ],
            "plate_1": [
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 2.0),
                (0.0, 0.0, 3.0),
                (0.0, 0.0, 4.0),
                (0.0, 0.0, 5.0),
                (0.0, 0.0, 6.0),
            ],
        },
    )

    assert targets == (
        (3.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 0.0, 6.0),
        (0.0, 0.0, 6.0),
        (0.0, 0.0, 6.0),
    )


def test_compute_placement_targets_uses_last_grasp_in_active_run() -> None:
    grasped = [None, "akita_black_bowl_1", None, "akita_black_bowl_1", None]
    targets = _compute_placement_targets_by_timestep(
        grasped_object_by_timestep=grasped,
        highlighted_object_by_timestep=_backfill_next_highlight_targets(grasped),
        object_positions_by_timestep={
            "akita_black_bowl_1": [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ],
        },
    )

    assert targets == (
        (4.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        None,
    )


def test_build_grasp_sequence_deduplicates_consecutive_grasps() -> None:
    grasp_sequence = build_grasp_sequence(
        [None, "akita_black_bowl_1", "akita_black_bowl_1", None, "plate_1", "plate_1", "akita_black_bowl_1"]
    )

    assert grasp_sequence == ("akita_black_bowl_1", "plate_1", "akita_black_bowl_1")


def test_build_online_next_object_annotation_plan_keeps_targets_aligned_with_grasps() -> None:
    plan = build_online_next_object_annotation_plan(
        NextObjectHighlightPlan(
            obj_of_interest=("akita_black_bowl_1", "plate_1"),
            graspable_obj_of_interest=("akita_black_bowl_1", "plate_1"),
            grasped_object_by_timestep=(
                None,
                "akita_black_bowl_1",
                "akita_black_bowl_1",
                None,
                "plate_1",
                "akita_black_bowl_1",
            ),
            highlighted_object_by_timestep=(
                "akita_black_bowl_1",
                "akita_black_bowl_1",
                "akita_black_bowl_1",
                "plate_1",
                "plate_1",
                "akita_black_bowl_1",
            ),
            placement_target_by_timestep=(
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
            ),
        )
    )

    assert plan.grasp_order == ("akita_black_bowl_1", "plate_1", "akita_black_bowl_1")
    assert plan.placement_targets == ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))


class _FakeCameraModel:
    cam_fovy = np.asarray([90.0], dtype=np.float64)

    @staticmethod
    def camera_name2id(camera_name: str) -> int:
        if camera_name != "agentview":
            raise KeyError(camera_name)
        return 0


class _FakeCameraData:
    cam_xpos = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    cam_xmat = np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


class _FakeCameraSim:
    model = _FakeCameraModel()
    data = _FakeCameraData()


class _FakeCameraEnv:
    sim = _FakeCameraSim()


def test_draw_placement_dot_on_observations_projects_target_into_camera() -> None:
    observations = {"agentview_image": np.zeros((9, 9, 3), dtype=np.uint8)}

    drawn = draw_placement_dot_on_observations(
        _FakeCameraEnv(),
        observations,
        placement_target=(0.0, 0.0, -1.0),
        camera_names=["agentview"],
        dot_rgb=(0, 96, 255),
        dot_alpha=1.0,
        dot_radius_px=1,
    )

    assert drawn is not observations
    assert np.array_equal(observations["agentview_image"], np.zeros((9, 9, 3), dtype=np.uint8))
    assert np.array_equal(drawn["agentview_image"][4, 4], np.asarray([0, 96, 255], dtype=np.uint8))
    assert np.count_nonzero(drawn["agentview_image"]) > 0


def test_online_next_object_highlight_tracker_advances_after_grasp_then_release_window() -> None:
    tracker = OnlineNextObjectHighlightTracker(
        grasp_order=("akita_black_bowl_1", "plate_1"),
        min_grasp_steps=1,
        release_steps=4,
    )

    assert tracker.current_object == "akita_black_bowl_1"
    assert tracker.observe(is_current_object_grasped=False) is False
    assert tracker.current_object == "akita_black_bowl_1"

    assert tracker.observe(is_current_object_grasped=True) is False
    assert tracker.current_object == "akita_black_bowl_1"

    assert tracker.observe(is_current_object_grasped=False) is False
    assert tracker.observe(is_current_object_grasped=False) is False
    assert tracker.observe(is_current_object_grasped=False) is False
    assert tracker.current_object == "akita_black_bowl_1"

    assert tracker.observe(is_current_object_grasped=False) is True
    assert tracker.current_object == "plate_1"


def test_online_next_object_highlight_tracker_keeps_last_object_highlighted() -> None:
    tracker = OnlineNextObjectHighlightTracker(
        grasp_order=("akita_black_bowl_1",),
        min_grasp_steps=1,
        release_steps=4,
    )

    assert tracker.current_object == "akita_black_bowl_1"
    assert tracker.observe(is_current_object_grasped=True) is False
    assert tracker.observe(is_current_object_grasped=False) is False
    assert tracker.current_object == "akita_black_bowl_1"


def test_make_mask_comparison_frames_stacks_and_labels_frames() -> None:
    left = np.zeros((2, 6, 8, 3), dtype=np.uint8)
    right = np.full((2, 6, 8, 3), 255, dtype=np.uint8)

    combined = make_mask_comparison_frames(
        original_frames=left,
        masked_frames=right,
        dataset_name="libero_spatial_no_noops",
        episode_index=0,
        task_instruction="pick up the black bowl next to the cookie box and place it on the plate",
        masked_instances=["akita_black_bowl_1"],
    )

    assert combined.shape == (2, 6 + 84, 16, 3)
    assert np.all(combined[:, 84:, :7] == 0)
    assert np.all(combined[:, 84:, 9:] == 255)
    assert np.any(combined[:, :84] != 22)
