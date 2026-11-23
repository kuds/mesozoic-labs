"""Exercise the shared walking measurements independently of the Triceratops controller."""

import mujoco
import numpy as np
import pytest

from examples.shared.footfalls import swing_events
from examples.shared.ground_contacts import GroundContacts
from examples.shared.locomotion_courses import Course, Mound
from examples.shared.turn_metrics import TurnCourse, circular_path_metrics
from examples.stegosaurus_stenops.build import MODEL_PATH as STEGOSAURUS_MODEL
from examples.stegosaurus_stenops.walk import WalkController as StegosaurusWalk


def test_swing_events_require_both_a_complete_unloading_and_local_clearance():
    loaded = [False, True, False, True, True, False, False, True, False]
    gap = [0.06, 0, 0.005, 0, 0, 0.023, 0.038, 0, 0.04]
    landings, peaks, reloads = swing_events(loaded, gap)
    np.testing.assert_array_equal(landings, [7])
    np.testing.assert_allclose(peaks, [0.038])
    assert reloads == 1


@pytest.mark.parametrize("radius", [-3.0, 3.0])
def test_arc_measurements_are_signed_and_independent_of_initial_pose(radius):
    course = TurnCourse(radius_m=radius, entry_s=2.0, turn_s=8.0, exit_s=2.0)
    assert [course.curvature_at(t) for t in (0.0, 2.0, 9.99, 10.0)] == [0, 1 / radius, 1 / radius, 0]
    heading = np.radians(168) + np.sign(radius) * np.linspace(0, np.radians(224), 601)
    # The measured arc starts far from the origin and the Euler heading wraps at ±π.
    xy = [17, -23] + radius * np.column_stack((np.sin(heading), -np.cos(heading)))
    wrapped = np.arctan2(np.sin(heading), np.cos(heading))
    results = circular_path_metrics(xy, wrapped, radius)
    assert results["heading_change_deg"] == pytest.approx(np.sign(radius) * 224)
    assert results["swept_angle_deg"] == pytest.approx(np.sign(radius) * 224)
    assert results["projected_arc_distance_m"] == pytest.approx(abs(radius) * np.radians(224))
    assert results["measured_curvature_per_m"] == pytest.approx(1 / radius, rel=1e-5)
    assert results["max_radial_error_m"] < 1e-10


def test_stegosaurus_walk_uses_the_same_mound_and_contact_probe():
    course = Course(mounds=(Mound(x=1.42, y=0.25, x_radius=0.35, y_radius=0.25, height=0.035),))
    model = course.model(STEGOSAURUS_MODEL)
    data = mujoco.MjData(model)
    course.reset(model, data)
    pads = {}
    for leg in ("lf", "rf", "lh", "rh"):
        body = model.body(f"{leg}_foot").id
        pads[leg] = [
            model.geom(i).name for i in np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0))
        ]
    probe = GroundContacts(model, pads, course.surface_names())
    controller = StegosaurusWalk(model)
    peak_mound_load = 0.0
    landed = []
    lifted = []
    for step in range(round(7.0 / model.opt.timestep)):
        controller.step(data)
        mujoco.mj_step(model, data)
        if step % 10 == 0:
            mujoco.mj_forward(model, data)
            sample = probe.measure(model, data)
            np.testing.assert_allclose(sample.loads.sum(), sample.surface_loads.sum())
            peak_mound_load = max(peak_mound_load, sample.surface_loads[1])
            landed.append(sample.loads > 0.01 * mujoco.mj_getTotalmass(model) * abs(model.opt.gravity[2]))
            lifted.append(sample.clearance)
    assert peak_mound_load > 100.0  # the second species actually stepped on the raised surface
    for load, gap in zip(np.asarray(landed).T, np.asarray(lifted).T, strict=True):
        touchdowns, _, _ = swing_events(load, gap)
        assert len(touchdowns) >= 1


def test_sloping_contact_uses_local_normal_and_tangent(tmp_path):
    scene = tmp_path / "contact.xml"
    scene.write_text("""<mujoco><worldbody>
      <frame euler="10 20 -4"><geom name="floor" type="plane" size="0 0 0.1" axisangle="0 1 0 25"/></frame>
      <body name="foot" pos="0 0 0.097"><freejoint name="base"/>
      <geom name="pad" type="sphere" size="0.1" mass="1"/></body>
    </worldbody></mujoco>""")
    course = Course(pitch_deg=12, bank_deg=-6)
    model = course.model(scene, floor_name="floor")
    data = mujoco.MjData(model)
    course.reset(model, data, root_joint="base")
    normal = data.geom("floor").xmat.reshape(3, 3)[:, 2]
    tangent = data.geom("floor").xmat.reshape(3, 3)[:, 0]
    data.qvel[:3] = -0.04 * normal + 0.13 * tangent
    mujoco.mj_forward(model, data)
    sample = GroundContacts(model, {"foot": ["pad"]}, ["floor"]).measure(model, data)
    assert sample.loads[0] > 1.0
    assert sample.clearance[0] == pytest.approx(-0.003)
    assert sample.slip == pytest.approx(0.13)
    assert sample.other_load == 0.0
    assert normal @ [1.2, -0.6, course.height(1.2, -0.6)] == pytest.approx(0.0, abs=1e-10)
