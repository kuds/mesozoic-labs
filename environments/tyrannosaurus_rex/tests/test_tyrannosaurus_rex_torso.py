"""Keep the longer torso continuous, bounded and clear of its hardware bays."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from environments.shared.robot_clearance import clearance_distance
from examples.tyrannosaurus_rex_50cm import demo, torso_checks, torso_structure
from examples.tyrannosaurus_rex_50cm.hardware_layout import load_manifest
from examples.tyrannosaurus_rex_50cm.torso_shell import COLLISIONS

HERE = Path(demo.__file__).parent


@pytest.fixture
def candidate():
    model, data = demo.load_model(path=HERE / "long_body/model.xml")
    return model, data, load_manifest()


def _row(report, name):
    return next(row for row in report["checks"] if row["name"] == name)


def test_compiled_torso_is_closed_covered_and_contains_whole_hardware_reserves(candidate):
    model, _, manifest = candidate
    before = {name: getattr(model, name).copy() for name in ("geom_pos", "geom_size", "mesh_vert", "mesh_face")}
    report = torso_checks.check_torso(model, manifest)
    assert report["known_limits_passed"], report["checks"]
    assert report["status"] == "incomplete"
    assert report["coverage"]["convex_sections"] == len(COLLISIONS)
    assert len(COLLISIONS) >= 3
    assert report["coverage"]["hardware_reserves"] == 4
    assert _row(report, "continuous_closed_surface")["connected_components"] == 1
    assert _row(report, "visible_surface_collision_coverage")["uncovered_triangles"] == 0
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(model, name), values)


def test_missing_triangle_is_an_open_seam(candidate):
    model, _, manifest = candidate
    mesh = model.geom("torso_shell").dataid[0]
    first = model.mesh_faceadr[mesh]
    model.mesh_face[first] = model.mesh_face[first + 1]
    report = torso_checks.check_torso(model, manifest)
    assert report["status"] == "failed"
    assert _row(report, "continuous_closed_surface")["status"] == "failed"


def test_two_separate_closed_surfaces_are_not_one_continuous_shell():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert torso_checks._topology(vertices, faces)["status"] == "passed"
    report = torso_checks._topology(np.concatenate((vertices, vertices + 3)), np.concatenate((faces, faces + 4)))
    assert report["nonmanifold_edges"] == 0
    assert report["connected_components"] == 2
    assert report["status"] == "failed"
    # Touching at one vertex does not make two closed pods one shell.
    second = np.array([0, 4, 5, 6])[faces]
    pinched = torso_checks._topology(np.concatenate((vertices, -vertices[1:])), np.concatenate((faces, second)))
    assert pinched["connected_vertex_components"] == 1
    assert pinched["connected_components"] == 2
    assert pinched["status"] == "failed"


def test_visible_shell_cannot_extend_beyond_collision_sections(candidate):
    model, _, manifest = candidate
    model.geom(COLLISIONS[0]).pos[0] -= 0.020
    report = torso_checks.check_torso(model, manifest)
    assert _row(report, "visible_surface_collision_coverage")["uncovered_triangles"] > 0
    assert report["status"] == "failed"


def test_box_intersection_uses_edge_cross_axes_not_only_triangle_bounds():
    # Triangle AABB overlaps the box, but the diagonal x+y>2.1 separates them.
    outside = np.array([[[2.1, 0, 0], [0, 2.1, 0], [2.1, 2.1, 0]]])
    halfsize = np.ones(3)
    assert np.all(outside.min((0, 1)) <= halfsize)
    assert np.all(outside.max((0, 1)) >= -halfsize)
    assert not torso_checks._surface_intersects_box(outside, halfsize)
    touching = np.array([[[2, 0, 0], [0, 2, 0], [2, 2, 0]]])
    assert torso_checks._surface_intersects_box(touching, halfsize)
    assert torso_checks._surface_intersects_box(outside * 0.4, halfsize)


def test_hardware_in_hip_opening_is_outside_actual_shell_despite_full_convex_hull(candidate):
    model, data, manifest = candidate
    geom = model.geom("compute_tray_envelope")
    geom.pos[:] = [0, 0, 0.075]
    mujoco.mj_forward(model, data)
    vertices, _ = torso_checks._mesh(model, data, "torso_shell")
    equations = ConvexHull(vertices).equations
    center = data.geom_xpos[geom.id]
    assert np.all(equations[:, :3] @ center + equations[:, 3] < 0)
    report = torso_checks.check_torso(model, manifest)
    assert not _row(report, "compute:outer_shell_containment")["center_inside"]
    assert report["status"] == "failed"


def test_enlarged_hardware_reserve_detects_surface_protrusion(candidate):
    model, _, manifest = candidate
    model.geom("power_board_envelope").size[1] = 0.2
    report = torso_checks.check_torso(model, manifest)
    row = _row(report, "power:outer_shell_containment")
    assert row["center_inside"]
    assert row["inflated_box_intersects_surface"]
    assert row["status"] == "failed"


def test_containment_requires_hardware_and_shell_to_share_a_rigid_frame(candidate):
    model, _, manifest = candidate
    geom = model.geom("compute_tray_envelope").id
    model.geom_bodyid[geom] = model.body("left_hip_yaw_body").id
    row = _row(torso_checks.check_torso(model, manifest), "compute:outer_shell_containment")
    assert not row["same_rigid_body"]
    assert row["status"] == "failed"


def test_side_silhouette_is_continuous_above_the_open_hip_bay(candidate):
    model, data, _ = candidate
    shell = model.geom("torso_shell").id
    model.geom_group[:] = 5
    model.geom_group[shell] = 1
    groups = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    hit = np.array([-1], dtype=np.int32)
    body = model.body("torso").id
    rotation = data.xmat[body].reshape(3, 3)
    for x in np.linspace(-0.185, 0.140, 66):
        origin = data.xpos[body] + rotation @ np.array([x, -1, 0.160])
        distance = mujoco.mj_ray(model, data, origin, rotation @ np.array([0.0, 1.0, 0.0]), groups, 1, -1, hit)
        assert distance > 0 and hit[0] == shell, x
    # A full convex torso hull would obscure this necessary motor opening.
    origin = data.xpos[body] + rotation @ np.array([0.0, -1, 0.075])
    assert mujoco.mj_ray(model, data, origin, rotation @ np.array([0.0, 1.0, 0.0]), groups, 1, -1, hit) < 0


def test_visible_supports_match_their_inspection_geometry_and_touch_mounts(candidate):
    model, data, _ = candidate
    for proxy_name in torso_structure.SUPPORT_PROXIES:
        proxy = model.geom(proxy_name).id
        visible = model.geom(proxy_name.removesuffix("_collision")).id
        for field in ("geom_bodyid", "geom_type", "geom_pos", "geom_size", "geom_quat"):
            np.testing.assert_array_equal(getattr(model, field)[proxy], getattr(model, field)[visible])
        assert model.geom_contype[proxy] == model.geom_conaffinity[proxy] == 0
    for first, second in torso_structure.SUPPORT_ATTACHMENTS:
        distance = clearance_distance(
            model,
            data,
            model.geom(first).id,
            model.geom(second).id,
            required_clearance_m=0.001,
            search_distance_m=1,
        )
        assert distance["min_distance_m"] < 0.0001, (first, second, distance)
    assert model.body_mass.sum() == pytest.approx(8.207)
    report = torso_structure.check_supports(model)
    assert report["known_limits_passed"]
    assert report["status"] == "incomplete"
    assert report["coverage"]["sampled_poses"] == 790


def test_inspection_only_supports_still_detect_a_yoke_crossing_the_head(candidate):
    model, _, _ = candidate
    model.geom("neck_yoke_rail_collision").pos[1] = 0
    report = torso_structure.check_supports(model)
    assert report["status"] == "failed"
    assert any(
        row["second"] in ("head_shell", "jaw_shell") and row["status"] == "failed" for row in report["clearance_checks"]
    )
