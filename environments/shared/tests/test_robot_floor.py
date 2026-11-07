"""Exercise effective contact parameters when switching surfaces between episodes."""

import mujoco
import numpy as np
import pytest

from environments.shared.robot_floor import apply_surface


def test_carpet_changes_loaded_indentation_and_foot_material_can_be_reset():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .01"/>
      <body pos="0 0 .01"><freejoint/><geom type="box" size=".05 .04 .01"
          mass="5" priority="2" friction="1.4 .01 .001"/></body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    depth = []
    for surface, friction in (("hard", 0.8), ("carpet", 0.95), ("hard", 0.8)):
        apply_surface(model, surface)
        mujoco.mj_resetData(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)
        assert data.ncon > 0
        np.testing.assert_allclose(data.contact.friction[:, 0], friction)
        assert all(c.dim == 6 for c in data.contact)
        depth.append(float(-data.contact.dist.min()))
    assert depth[1] > depth[0] + 0.0003
    assert depth[0] == pytest.approx(depth[2], abs=1e-9)
