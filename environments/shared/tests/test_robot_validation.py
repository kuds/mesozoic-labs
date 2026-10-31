"""Negative controls for honest unsupported-robot engineering measurements."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from environments.shared.robot_validation import GroundSupportProbe, require_unsupported


def _scene(*, fixture: str = "", equality: str = "", torso_z: float = 0.0) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        {fixture}
        <body name="torso" pos="0 0 0.1">
          <freejoint name="root"/>
          <geom name="torso_box" type="box" pos="0 0 {torso_z}" size="0.04 0.025 0.02" mass="1"/>
          <site name="robot_site"/>
          <geom name="left_pad" type="box" pos="0 0.05 -0.09" size="0.06 0.025 0.01" mass="0.05"/>
          <geom name="right_pad" type="box" pos="0 -0.05 -0.09" size="0.06 0.025 0.01" mass="0.05"/>
        </body>
      </worldbody>
      <equality>{equality}</equality>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _probe(model: mujoco.MjModel) -> GroundSupportProbe:
    return GroundSupportProbe(model, feet={"left": ("left_pad",), "right": ("right_pad",)})


@pytest.mark.parametrize(
    "constraint",
    [
        '<weld body1="torso" body2="fixture"/>',
        '<weld site1="robot_site" site2="fixture_site"/>',
        '<connect body1="torso" body2="fixture" anchor="0 0 0.1"/>',
    ],
)
@pytest.mark.parametrize("mocap", [False, True])
def test_rejects_fixed_and_mocap_fixture_constraints(constraint, mocap):
    model, data = _scene(
        fixture=f'<body name="fixture" mocap="{str(mocap).lower()}"><site name="fixture_site"/></body>',
        equality=constraint,
    )
    with pytest.raises(ValueError, match="fixture constraint"):
        require_unsupported(model, data)


@pytest.mark.parametrize("disabled", [mujoco.mjtDisableBit.mjDSBL_GRAVITY, mujoco.mjtDisableBit.mjDSBL_CONTACT])
def test_rejects_disabled_physics_even_when_gravity_vector_looks_normal(disabled):
    model, data = _scene()
    model.opt.disableflags |= disabled
    with pytest.raises(ValueError, match="physics must be enabled"):
        require_unsupported(model, data)


def test_rejects_gravity_compensation_and_external_forces():
    model, data = _scene()
    model.body_gravcomp[1] = 0.5
    with pytest.raises(ValueError, match="gravity compensation"):
        require_unsupported(model, data)
    model.body_gravcomp[:] = 0
    for forces in (data.xfrc_applied, data.qfrc_applied):
        forces.flat[0] = 1
        with pytest.raises(ValueError, match="Applied external forces"):
            require_unsupported(model, data)
        forces[:] = 0


def test_floor_loads_reconcile_with_body_weight_without_touch_sensors():
    model, data = _scene()
    require_unsupported(model, data)
    probe = _probe(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    loads, unexpected = probe.sample(data)
    assert loads == pytest.approx(np.repeat(model.body_mass.sum() * 9.81 / 2, 2), rel=0.01)
    assert unexpected == 0


def test_torso_on_unlisted_fixture_is_not_foot_support():
    model, data = _scene(fixture='<body name="fixture"><geom type="box" size="0.03 0.02 0.04" pos="0 0 0.05"/></body>')
    probe = _probe(model)
    loads, unexpected = probe.sample(data)
    assert unexpected > 0.05
    assert sum(loads) < model.body_mass.sum() * 9.81


def test_nonfoot_ground_contact_is_rejected_even_without_a_fixture():
    model, data = _scene(torso_z=-0.085)
    assert _probe(model).sample(data)[1] > 0.05


def test_stacked_feet_do_not_count_as_two_grounded_feet():
    model, data = _scene()
    model.geom_pos[model.geom("right_pad").id] = [0, 0.05, -0.075]
    mujoco.mj_forward(model, data)
    loads, _ = _probe(model).sample(data)
    assert loads[1] == 0


def test_support_metadata_cannot_alias_feet_or_use_a_moving_floor():
    model, _ = _scene()
    with pytest.raises(ValueError, match="exactly one robot foot"):
        GroundSupportProbe(model, feet={"left": ("left_pad",), "right": ("left_pad",)})
    with pytest.raises(ValueError, match="fixed external floor"):
        GroundSupportProbe(model, feet={"left": ("left_pad",)}, floor_geoms=("right_pad",))
