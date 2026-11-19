"""Tests of the robustness sweep's mechanics (no checkpoint needed): the push, the factor application and the terrain.

  python -m pytest hardware/triceratops_build/rl/robustness/tests -q
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")  # before MuJoCo loads (see rl/tests/conftest.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sweep  # noqa: E402

FREE_BODY = """
<mujoco>
  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="box" pos="0 0 1" euler="0 0 30">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.05" mass="2.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def nominal_env(**changes):
    cfg = sweep.TrikeEnvConfig(domain_randomization=False, phase_clock_hz=1.5).replace(**changes)
    return sweep.PointEnv(cfg)


MODEL_ARRAYS = ("geom_friction", "body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm",
                "actuator_forcerange", "body_subtreemass", "body_invweight0", "dof_invweight0", "geom_size")


def snapshot(m):
    return {k: getattr(m, k).copy() for k in MODEL_ARRAYS}


# ---------------------------------------------------------------- the push
@pytest.mark.parametrize("dv", [0.3, -0.5, 1.0])
def test_push_gives_the_intended_velocity_change_to_a_free_body(dv):
    m = mujoco.MjModel.from_xml_string(FREE_BODY)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = m.body("box").id
    pusher = sweep.Pusher(m, d, body, dv, t0=0.05, duration=0.1)
    assert pusher.mass == pytest.approx(2.5)
    v_before = None
    for _ in range(300):
        mujoco.mj_step(m, d)
        pusher.after_step()
        if abs(d.time - 0.05) < 1e-9:
            v_before = d.qvel[:3].copy()
    assert pusher.steps == 100  # exactly 0.1 s of 1 ms steps
    assert np.allclose(v_before, 0.0)
    left = np.array([-math.sin(math.radians(30)), math.cos(math.radians(30)), 0.0])  # the body's +y, yawed 30 deg
    assert np.allclose(pusher.direction, left)
    assert np.allclose(d.qvel[:3], dv * left, atol=1e-9)
    assert np.allclose(d.xfrc_applied, 0.0)  # switched off after the push


def test_push_force_is_mass_times_dv_over_duration():
    f = sweep.push_force(12.7, 0.4, 0.1, [0.0, 2.0, 0.0])
    assert np.allclose(f, [0.0, 12.7 * 0.4 / 0.1, 0.0])


def test_push_in_an_episode_starts_at_8_s_and_lasts_0_1_s():
    torch = pytest.importorskip("torch")
    env = nominal_env(episode_seconds=8.5)
    applied = []

    class Zero:  # a stand-in policy: all-zero actions (the robot stands)
        def __call__(self, obs, deterministic=True):
            return torch.zeros((1, 12)), None, None

    n = env.observation_space.shape[0]
    stats = (np.zeros(n), np.ones(n), 10.0, 1e-8)
    orig = sweep.Pusher.after_step

    def spy(self):
        orig(self)
        if np.any(self.d.xfrc_applied[self.body] != 0):
            applied.append(round(self.d.time, 6))

    sweep.Pusher.after_step = spy
    try:
        rec = sweep.run_episode(env, Zero(), stats, 200000, sweep.make_point(push_dv=0.3))
    finally:
        sweep.Pusher.after_step = orig
    assert rec["push"]["applied_steps"] == 100
    assert applied[0] == pytest.approx(8.0) and applied[-1] == pytest.approx(8.099)  # set for steps 8.000-8.099
    assert rec["push"]["force_n"] == pytest.approx(env.nominal_mass * 0.3 / 0.1, rel=1e-4)
    assert not rec["fell"]
    assert np.allclose(env.data.xfrc_applied, 0.0)


# ---------------------------------------------------------------- factor application
def test_mass_scaling_matches_the_env_randomisation():
    env = nominal_env()
    f = env._draw_factors(np.random.default_rng(0))
    f["mass_scale"][:] = 1.15
    env._apply_factors(f)  # the env's own randomisation, every body at x1.15
    ref = snapshot(env.model)
    env._apply_factors({})
    env.point = sweep.make_point(mass_scale=1.15)
    env._apply_factors({})  # the sweep's
    got = snapshot(env.model)
    for k in ("body_mass", "body_inertia", "body_subtreemass", "body_invweight0", "dof_invweight0"):
        assert np.allclose(got[k], ref[k], rtol=1e-12, atol=0), k
    assert env.model.body_mass.sum() == pytest.approx(1.15 * env.nominal_mass, rel=1e-12)


def test_friction_and_kp_match_the_env_on_the_floor_soles_and_leg_drives():
    env = nominal_env()
    f = env._draw_factors(np.random.default_rng(1))
    f["friction"], f["kp_scale"][:], f["force_limit_scale"][:], f["mass_scale"][:] = 0.4, 1.2, 0.7, 1.0
    env._apply_factors(f)
    ref = snapshot(env.model)
    env.point = sweep.make_point(friction=0.4, kp_scale=1.2, force_scale=0.7)
    env._apply_factors({})
    got = snapshot(env.model)
    leg = env._leg_act
    assert np.array_equal(got["geom_friction"], ref["geom_friction"])
    for k in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
        assert np.allclose(got[k][leg], ref[k][leg], rtol=1e-12), k
    # the head and tail servos stay nominal in the sweep (the env's randomisation scales them too)
    nominal = env._nominal
    other = env._other_act
    assert np.array_equal(got["actuator_forcerange"][other], nominal["actuator_forcerange"][other])
    assert np.array_equal(got["actuator_gainprm"][other], nominal["actuator_gainprm"][other])


def test_payload_adds_mass_at_the_chassis_only():
    env = nominal_env()
    env.point = sweep.make_point(payload_kg=0.5)
    env._apply_factors({})
    c = env._chassis
    m0, i0 = env._nominal["body_mass"], env._nominal["body_inertia"]
    assert env.model.body_mass.sum() == pytest.approx(env.nominal_mass + 0.5, rel=1e-12)
    assert env.model.body_mass[c] == pytest.approx(m0[c] + 0.5)
    assert np.allclose(env.model.body_inertia[c], i0[c] * (m0[c] + 0.5) / m0[c])
    others = np.arange(env.model.nbody) != c
    assert np.array_equal(env.model.body_mass[others], m0[others])


def test_the_sweep_restores_the_nominal_model_between_points():
    env = nominal_env()
    env.point = sweep.make_point()
    env.reset(seed=0)
    nominal = snapshot(env.model)
    fresh = nominal_env()
    fresh.reset(seed=0)
    for k, v in snapshot(fresh.model).items():
        assert np.array_equal(nominal[k], v), k  # the nominal point is exactly the env's nominal model
    points = [sweep.make_point(friction=0.3), sweep.make_point(mass_scale=1.2), sweep.make_point(payload_kg=1.0),
              sweep.make_point(force_scale=0.7), sweep.make_point(kp_scale=0.8),
              sweep.make_point(mass_scale=0.8, friction=1.2, kp_scale=1.2, force_scale=0.85, payload_kg=0.5)]
    for p in points:
        env.point = p
        env.reset(seed=1)
        changed = snapshot(env.model)
        assert any(not np.array_equal(changed[k], nominal[k]) for k in nominal), p
        env.data.xfrc_applied[env._chassis, :3] = [5.0, 5.0, 0.0]  # as if a push were under way
        env.point = sweep.make_point()
        env.reset(seed=2)
        for k, v in snapshot(env.model).items():
            assert np.array_equal(v, nominal[k]), (p, k)
        assert not env.data.xfrc_applied.any()


def test_delay_and_command_come_from_the_point():
    env = nominal_env()
    env.point = sweep.make_point(delay_ms=20, command=0.15)
    p = env.point
    env.reset(seed=0, options={"command_speed": p["command"], "delay_steps": p["delay_ms"]})
    assert env.delay_steps == 20 and env.command_speed == pytest.approx(0.15)
    assert not env.factors  # randomisation off: nothing drawn is applied


# ---------------------------------------------------------------- terrain
def _height_at(m, d, x, y=0.0):
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, 1.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
    return 1.0 - dist, int(geomid[0])


@pytest.mark.parametrize("terrain", [("step", 0.02), ("ramp", 5.0)])
def test_terrain_box_geometry_and_contact_settings(terrain):
    env = nominal_env()
    m = mujoco.MjModel.from_xml_string(sweep.terrain_xml(env.cfg.model_path, terrain))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tg, fl = m.geom("terrain").id, m.geom("floor").id
    for attr in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_contype", "geom_conaffinity",
                 "geom_priority", "geom_margin", "geom_gap", "geom_solmix"):
        assert np.array_equal(getattr(m, attr)[tg], getattr(m, attr)[fl]), attr
    x0 = sweep.TERRAIN_EDGE_M
    h, g = _height_at(m, d, x0 - 0.05, y=1.0)  # clear of the robot
    assert g == fl and h == pytest.approx(0.0, abs=1e-9)
    for dx in (0.05, 0.5, 2.0):
        h, g = _height_at(m, d, x0 + dx, y=0.5)
        assert g == tg
        want = terrain[1] if terrain[0] == "step" else dx * math.tan(math.radians(terrain[1]))
        assert h == pytest.approx(want, abs=1e-5)  # MJCF text rounds the box's pose to about 1e-7 m
    assert m.nbody == env.model.nbody and m.nu == env.model.nu  # the robot itself is unchanged


def test_terrain_env_starts_on_the_flat_floor():
    env = sweep.build_env(nominal_env().cfg.to_dict(), terrain=("step", 0.02))
    env.point = sweep.make_point(terrain=("step", 0.02))
    env.reset(seed=0)
    front_edge = max(env.data.geom_xpos[g][0] + env.model.geom_size[g][0] for g in env._soles)
    assert front_edge < sweep.TERRAIN_EDGE_M - 0.2
    assert env._terrain_geom == env.model.geom("terrain").id


# ---------------------------------------------------------------- reading the results
def test_find_breaks_walks_outward_from_nominal():
    def s(falls=0, v=0.24):
        return {"falls": falls, "episodes": 8, "speed_mps": v}

    rows = [(0.3, s(falls=2)), (0.4, s(v=0.18)), (0.5, s()), (0.85, s()), (1.0, s()), (1.2, s(v=0.15))]
    b = sweep.find_breaks(rows, 0.85, 0.24)
    assert b["low"]["x"] == 0.4 and "below nominal" in b["low"]["reasons"][0]
    assert b["high"]["x"] == 1.2
    b = sweep.find_breaks([(0.0, s()), (0.5, s()), (1.0, s())], 0.0, 0.24)
    assert "low" not in b and b["high"] is None


def test_plan_shares_the_nominal_point_across_sweeps():
    fams, points = sweep.plan()
    nominal = sweep.point_key(sweep.make_point())
    uses = sum(r["key"] == nominal for f in fams for r in f["rows"])
    assert uses >= 6  # delay 0.25, friction, mass, payload, force, kp, push, step, ramp all share it
    assert all(p["randomise"] is False for p in points.values())
