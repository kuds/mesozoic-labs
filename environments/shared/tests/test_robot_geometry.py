"""Geometry measurements use compiled transforms and exclude nonvisual groups."""

import mujoco
import numpy as np
import pytest

from environments.shared.robot_geometry import internal_contacts, visual_bounds


@pytest.mark.parametrize(
    "kind,size,extent",
    [
        ("box", "0.1 0.2 0.3", [0.3, 0.2, 0.1]),
        ("ellipsoid", "0.1 0.2 0.3", [0.3, 0.2, 0.1]),
        ("cylinder", "0.1 0.3", [0.3, 0.1, 0.1]),
        ("capsule", "0.1 0.3", [0.4, 0.1, 0.1]),
        ("sphere", "0.1", [0.1, 0.1, 0.1]),
    ],
)
def test_visual_bounds_apply_world_rotation_and_ignore_hidden_geometry(kind, size, extent):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .1" group="0"/>
        <body pos="1 2 3" quat="1 0 1 0">
          <geom type="{kind}" size="{size}" group="1"/>
          <geom type="box" size="10 10 10" group="3"/>
          <geom type="box" size="20 20 20" group="4"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = visual_bounds(model, data)
    assert low == pytest.approx(np.array([1, 2, 3]) - extent)
    assert high == pytest.approx(np.array([1, 2, 3]) + extent)


def test_visual_bounds_use_compiled_mesh_vertices():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <asset><mesh name="tetra" vertex="0 0 0  .2 0 0  0 .3 0  0 0 .4"/></asset>
      <worldbody>
        <geom type="mesh" mesh="tetra" pos="1 2 3" quat="1 0 0 1" group="2"/>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = visual_bounds(model, data)
    assert low == pytest.approx([0.7, 2, 3], abs=1e-7)
    assert high == pytest.approx([1, 2.2, 3.4], abs=1e-7)


def test_internal_contacts_exclude_only_the_declared_floor():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="10 10 .1"/>
        <body pos="0 0 .1"><freejoint/><geom name="first" type="box" size=".1 .1 .1"/></body>
        <body pos=".15 0 .1"><freejoint/><geom name="second" type="box" size=".1 .1 .1"/></body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    contacts = internal_contacts(model, data, floor_geom="ground")
    assert contacts
    assert len(contacts) < data.ncon
    assert all(set(row["pair"]) == {"first", "second"} for row in contacts)
    assert max(row["penetration_m"] for row in contacts) == pytest.approx(0.05)
