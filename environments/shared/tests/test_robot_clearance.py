"""Independent separation certificates and controls for native false zeros."""

import math

import mujoco
import numpy as np
import pytest

from environments.shared.robot_clearance import clearance_distance, separation_lower_bound


def test_rotated_boxes_have_certified_clearance_despite_native_false_zero():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body><freejoint/><geom type="box" size=".13 .08 .024"/></body>
      <body><freejoint/><geom type="box" size=".1125 .072 .01"/></body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    # Independent rigid poses from a seeded box-separation probe. A plane
    # normal to the first box's local Z axis separates their surfaces by25mm.
    data.qpos[:] = [
        -0.3682672892714538,
        -0.010195052175700492,
        -0.1135469435398726,
        -0.5695977917720128,
        -0.7728216614531113,
        -0.246547559509853,
        -0.13236062895376025,
        -0.2959835393746608,
        -0.13156856512415022,
        -0.16061129622535747,
        -0.02004419955684049,
        -0.9934225943119477,
        -0.1124796781520457,
        -0.00762241283697646,
    ]
    mujoco.mj_forward(model, data)
    assert mujoco.mj_geomDistance(model, data, 0, 1, 1, None) == 0
    positions, orientations = data.geom_xpos.copy(), data.geom_xmat.copy()
    result = clearance_distance(model, data, 0, 1, required_clearance_m=0.002, search_distance_m=1)
    assert result["distance_method"] == "support_plane_lower_bound"
    assert result["distance_is_lower_bound"]
    assert result["native_distance_m"] == 0
    assert result["min_distance_m"] >= 0.025 - 1e-12
    proof = result["separation_plane"]
    axis = np.array(proof["axis_world"])
    # Enumerate all8corners, independently of the support-radius implementation.
    signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    projections = [
        ((signs * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]) @ axis for g in (0, 1)
    ]
    corner_gap = max(projections[1].min() - projections[0].max(), projections[0].min() - projections[1].max())
    assert 0 < result["min_distance_m"] <= corner_gap
    np.testing.assert_array_equal(data.geom_xpos, positions)
    np.testing.assert_array_equal(data.geom_xmat, orientations)


@pytest.mark.parametrize(
    "kind,size,half_height",
    [
        ("sphere", ".1", 0.1),
        ("capsule", ".05 .12", 0.17),
        ("ellipsoid", ".1 .08 .06", 0.06),
        ("cylinder", ".05 .12", 0.12),
        ("box", ".1 .08 .06", 0.06),
    ],
)
def test_support_bounds_agree_with_known_axial_separation(kind, size, half_height):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody>
      <geom type="{kind}" size="{size}"/>
      <geom type="sphere" size=".02" pos="0 0 .5"/>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    proof = separation_lower_bound(model, data, 0, 1)
    exact_distance = 0.5 - half_height - 0.02
    assert proof is not None
    assert proof["distance_lower_bound_m"] == pytest.approx(exact_distance)
    assert proof["distance_lower_bound_m"] <= exact_distance


def test_mesh_support_uses_compiled_scaled_rotated_vertices():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <asset><mesh name="tetrahedron" scale=".1 .2 .3" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/></asset>
      <worldbody>
        <geom type="mesh" mesh="tetrahedron" euler="0 0 90"/>
        <geom type="sphere" size=".02" pos="0 0 .5"/>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    proof = separation_lower_bound(model, data, 0, 1)
    assert proof is not None
    assert proof["distance_lower_bound_m"] == pytest.approx(0.18, abs=1e-7)
    # MuJoCo compiles mesh coordinates as float32; compare to those vertices.
    verts = model.mesh_vert @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]
    axis = np.array(proof["axis_world"])
    sphere_center = data.geom_xpos[1] @ axis
    exact_projection_gap = max(sphere_center - 0.02 - (verts @ axis).max(), (verts @ axis).min() - sphere_center - 0.02)
    assert proof["distance_lower_bound_m"] <= exact_projection_gap


@pytest.mark.parametrize("offset", [0.0, 0.10, 0.199, 0.20])
def test_touching_or_overlapping_boxes_do_not_receive_a_separation_certificate(offset):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody>
      <geom type="box" size=".1 .1 .1"/>
      <geom type="box" size=".1 .1 .1" pos="{offset} 0 0"/>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert separation_lower_bound(model, data, 0, 1) is None
    result = clearance_distance(model, data, 0, 1, required_clearance_m=0.002, search_distance_m=1)
    assert result["min_distance_m"] < 0.002
    assert result["distance_method"] == "mujoco"


def test_capped_native_distance_remains_an_explicit_lower_bound():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom type="sphere" size=".1"/>
      <geom type="sphere" size=".1" pos="2 0 0"/>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = clearance_distance(model, data, 0, 1, required_clearance_m=0.002, search_distance_m=1)
    assert result["min_distance_m"] == 1
    assert result["distance_is_lower_bound"]
    assert result["distance_method"] == "mujoco"


def test_small_support_bound_does_not_establish_a_clearance_failure(monkeypatch):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom type="box" size=".1 .1 .1"/>
      <geom type="sphere" size=".001" pos=".101 .1029 0"/>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # The closest point is a box corner. Our finite direction set certifies
    # only1.9mm even though the analytical Euclidean gap exceeds2mm.
    exact_distance = math.hypot(0.001, 0.0029) - 0.001
    assert exact_distance > 0.002
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: 0.0)
    evidence = clearance_distance(model, data, 0, 1, required_clearance_m=0.002, search_distance_m=1)
    assert evidence["distance_is_lower_bound"]
    assert evidence["min_distance_m"] == pytest.approx(0.0019)
    assert evidence["min_distance_m"] < 0.002 < exact_distance


def test_oblique_sphere_gap_never_exceeds_analytical_distance():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom type="sphere" size=".1"/>
      <geom type="sphere" size=".2" pos=".4 .5 .6"/>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    proof = separation_lower_bound(model, data, 0, 1)
    exact_distance = math.sqrt(0.4**2 + 0.5**2 + 0.6**2) - 0.3
    assert proof is not None
    assert proof["distance_lower_bound_m"] <= exact_distance
    assert proof["distance_lower_bound_m"] == pytest.approx(exact_distance)
