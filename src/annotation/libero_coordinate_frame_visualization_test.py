from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from annotation.libero_coordinate_frame_visualization import _inject_coordinate_marker_geoms
from annotation.libero_coordinate_frame_visualization import _make_side_by_side_frame


def test_inject_coordinate_marker_geoms_adds_world_and_robot_marker_clusters() -> None:
    xml = """
    <mujoco model="base">
      <worldbody>
        <body name="robot_root_body">
          <geom name="robot_geom" type="box" size="0.01 0.01 0.01"/>
        </body>
      </worldbody>
    </mujoco>
    """

    injected = _inject_coordinate_marker_geoms(
        xml,
        robot_root_body_name="robot_root_body",
        world_anchor=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        robot_local_anchor=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        axis_scale_m=0.5,
        marker_radius_m=0.05,
    )

    tree = ET.fromstring(injected)
    assert tree.find(".//body[@name='coord_markers_world_cluster']") is not None
    assert tree.find(".//body[@name='coord_markers_robot_cluster']") is not None
    assert tree.find(".//body[@name='coord_marker_world_origin']") is not None
    assert tree.find(".//body[@name='coord_marker_world_x']") is not None
    assert tree.find(".//body[@name='coord_marker_robot_origin']") is not None
    assert tree.find(".//body[@name='coord_marker_robot_z']") is not None


def test_make_side_by_side_frame_places_raw_and_marked_panels() -> None:
    raw = np.zeros((8, 10, 3), dtype=np.uint8)
    marked = np.full((8, 10, 3), 255, dtype=np.uint8)

    combined = _make_side_by_side_frame(raw, marked, header_lines=["test"])

    assert combined.shape == (8 + 90, 20, 3)
    np.testing.assert_array_equal(combined[90:, :8], raw[:, :8])
    np.testing.assert_array_equal(combined[90:, 12:], marked[:, 2:])
