"""The servo model of gate 1, package S1 (``designlab/stage2/servo.py``; ``stage2/design/servo_model/
SERVO_MODEL_DESIGN.md`` section 5.5, tests T4 to T12 and T14 on the plant).

Simulation only: every check here is of the model's arithmetic and its hooks on a Stage 1 plant in memory; none is a
result, no episode is a record, and no file is written.  T1 to T3 and T13 (the record hashes, the env option, the
served rows and ``g8s``) and T14's case of a delay queue at a mid-task start are package S2's; a plant-level form of
T3 (``today`` is the plant bit for bit) is here as well.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import candidate_id, case_dir, drive, env, plant, speedfit  # noqa: E402
from designlab.stage2 import servo  # noqa: E402

BODY = servo.BODY


def _have() -> bool:
    try:  # (the round-two model files, read where they are: plant.model_files would rebuild missing ones)
        base = case_dir(BODY) / "models" / candidate_id(BODY)
        return base.with_suffix(".xml").exists() and base.with_suffix(".json").exists()
    except Exception:  # noqa: BLE001  (no model record: the same skip)
        return False


needs_plant = pytest.mark.skipif(not _have(), reason="needs the round-two model files (tools/restore_inputs.py)")


@pytest.fixture(scope="module")
def p():
    return plant.build(BODY, "hard")


def _arrays(m) -> dict:
    return {k: np.array(v, copy=True) for k, v in env._model_arrays(m).items()}


def _model_diff(m, saved) -> list:
    now = env._model_arrays(m)
    return sorted(k for k in saved if not np.array_equal(now[k], saved[k], equal_nan=True))


def _digest(p) -> str:
    d = p.d
    h = hashlib.sha256()
    for a in (d.qpos, d.qvel, d.ctrl, d.sensordata, d.actuator_force, d.qacc_warmstart, p.m.actuator_forcerange):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _neutral(**kw) -> servo.ServoParams:
    """``today`` with the named fields changed: one part of the layer at a time."""
    return replace(servo.setting("today"), name="test", **kw)


def _random_state(p, rng, speed=3.0):
    q = p.q_settled.copy()
    q[p.qadr] += rng.uniform(-0.05, 0.05, len(p.qadr))
    qd = np.zeros(p.m.nv)
    qd[p.dofs] = rng.uniform(-speed, speed, len(p.dofs))
    return q, qd


def _expected_force(p, lay, dead_zone=True):
    """The servo law with the run's model gains and the layer's command, clipped to the force range."""
    m, d = p.m, p.d
    L = p.G @ d.qpos[p.qadr]
    w = drive.motor_matrix(p.joints) @ d.qvel[p.dofs]
    kp, kv = m.actuator_gainprm[:, 0], -m.actuator_biasprm[:, 2]
    u = kp * (lay._c - lay.offset - L) - kv * w
    kb = kp * lay.b
    tau = np.where(u > kb, u - kb, np.where(u < -kb, u + kb, 0.0)) if dead_zone else u
    return np.clip(tau, m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1])


# ------------------------------------------------------------------------------------------------
# The table, the description, the named settings
# ------------------------------------------------------------------------------------------------
def test_the_table_has_the_designs_39_rows_with_a_source_and_a_kind_each():
    assert len(servo.TABLE) == 39
    for r in servo.TABLE:
        assert r["kind"] in servo.KINDS and r["source"] and r["replaced_by"], r["param"]
    # design section 3: 1 maker's own figure, 4 a maker's figure for a related quantity taken as the value, 6
    # third-party, 28 the study's (4 of them estimates on maker figures)
    k = servo.kind_counts()
    assert k[servo.MAKER] == 1 and k[servo.THIRD] == 6 and k[servo.ESTIMATE] == 4
    assert k["of which a maker's figure for a related quantity, taken as the value"] == 4
    assert k[servo.ASSUMED] - 4 + k[servo.ESTIMATE] + k[servo.UNSOURCED] == 28
    assert servo.TABLE_SHA256 == servo.table_sha256()


def test_the_layer_prints_every_parameter_with_its_source_and_kind():
    for name in ("N", "today", "X6L"):
        lines = servo.describe_lines(servo.setting(name))
        body = lines[2:] if name != "M+" else lines[3:]
        assert len(body) == sum(len(r["classes"]) for r in servo.TABLE)
        for line, r in zip(body, servo.describe_rows(servo.setting(name)), strict=True):
            assert r["kind"] in line and r["source"] in line
            assert ("UNSOURCED" in line) == (r["kind"] == servo.UNSOURCED)
    assert "caller" in " ".join(servo.describe_lines(servo.setting("M+"))[:3])


def test_named_settings_move_one_factor_from_N():
    n = servo.setting("N")
    moved = {"F-": {"fric_c", "fric_v", "fric_load"}, "F+": {"fric_c", "fric_v", "fric_load"},
             "P0": {"play_deg"}, "P+": {"play_deg"}, "M-": {"mass"}, "M+": {"mass"}, "V-": {"bus_v0", "bus_r"},
             "J+": {"armature"}, "G5": {"gain_cap"}, "T+": {"ts_corner"}, "X6L": {"fric_c", "peak"},
             "W1": {"fric_c", "fric_v", "fric_load", "bus_v0", "bus_r", "armature", "mass"},
             "W2": {"fric_c", "fric_v", "fric_load", "play_deg", "mass"}}
    for name, fields_ in moved.items():
        s = servo.setting(name)
        diff = {f for f, v in s.as_dict().items() if f != "name" and v != n.as_dict()[f]}
        assert diff == fields_, name
    legs = [c in servo.LEG_CLASSES for _, c in servo.DRIVES]
    assert servo.setting("P+").play_deg == tuple(1.0 if g else 0.0 for g in legs)
    assert n.play_deg == tuple(0.25 if g else 0.0 for g in legs)
    assert (servo.setting("V-").bus_v0, servo.setting("V-").bus_r) == (21.0, 0.15)
    assert (n.bus_v0, n.bus_r, n.ts_corner[:2], n.cmd_lag_ms, n.armature) == (
        22.2, 0.05, (5.76, 4.16), (0.5, 1.2, 0.0, 0.0), (None, None, None, 0.028))
    assert servo.setting("J+").armature == (0.106, 0.0254, 0.026, 0.028)
    assert servo.setting("X6L").peak[1] == pytest.approx(31.5) and servo.setting("X6L").fric_c[1] == 0.2
    assert set(servo.JUDGED) == {"F-", "F+", "P0", "P+", "M-", "M+", "D"}  # P41's six extremes and the panel
    assert set(servo.JUDGED) | set(servo.REPORTED) | {"N", "today"} == set(servo.NAMED)


# ------------------------------------------------------------------------------------------------
# T12: the draw
# ------------------------------------------------------------------------------------------------
def test_t12_same_seed_same_parameters_and_the_panels_own_stream():
    for s in (3042, 3061, 5042):
        assert servo.panel_draw(s) == servo.panel_draw(s) == servo.setting("D", s)
        assert servo.training_draw(s) == servo.draw(s, 2, f"train/{s}")
        d, t = servo.panel_draw(s), servo.training_draw(s)
        assert d.fric_c != t.fric_c and d.enc_offset != t.enc_offset  # tag 3 is the panel's own
        raw = servo.draw(s, servo.PANEL_TAG)
        n = servo.nominal()
        assert (d.armature, d.bus_r) == (n.armature, n.bus_r)  # held at nominal in D
        assert replace(raw, armature=n.armature, bus_r=n.bus_r, name=d.name) == d  # nothing else differs
        assert (d.mass, d.gain_cap, d.cmd_hold) == (1.0, False, True)  # no trunk mass, no cap; latency not drawn
        assert not hasattr(d, "latency_steps")
    head = [servo.panel_draw(s) for s in range(3042, 3062)]
    second = [servo.panel_draw(s) for s in range(5042, 5062)]
    keys = {(x.fric_c, x.enc_offset, x.bus_v0) for x in head + second}
    assert len(keys) == 40  # the second block draws 20 other sets


def test_t12_no_episode_seed_no_draw():
    with pytest.raises(ValueError, match="named setting"):
        servo.training_draw(None)
    with pytest.raises(ValueError, match="bar seed"):
        servo.setting("D")
    with pytest.raises(ValueError, match="draws nothing"):
        servo.setting("N", 3042)


def test_t12_draws_lie_in_the_tables_ranges():
    for s in range(200):
        x = servo.draw(s, 2)
        for i, c in enumerate(servo.CLASSES):
            for f in ("fric_c", "fric_v", "fric_load", "cmd_lag_ms"):
                assert servo._number(f, c, "low") <= getattr(x, f)[i] <= servo._number(f, c, "high")
            assert servo._number("armature", c, "low") * (1 - 1e-12) <= x.armature[i] <= servo._number(
                "armature", c, "high") * (1 + 1e-12)
        for (a, c), v, o, mj in zip(servo.DRIVES, x.play_deg, x.enc_offset, x.fric_joint, strict=True):
            assert (0.0 <= v <= 1.0) if c in servo.LEG_CLASSES else v == 0.0
            assert -0.015 <= o <= 0.015 and 0.8 <= mj <= 1.25
        c6, z6 = x.ts_corner[1], x.ts_zero[1]
        assert 2.15 <= c6 <= servo.TODAY_CORNER["X6"] and 5.76 <= x.ts_corner[0] <= servo.TODAY_CORNER["X4"]
        assert max(8.01, c6 + 1.20) <= z6 <= servo.TODAY_ZERO["X6"]  # the lead's rule for the X6's zero
        assert 0 <= x.hold_phase <= 4 and 0 <= x.sense_steps <= 2 and 21.0 <= x.bus_v0 <= 25.2
        assert x.kp_scale[:2] == (1.0, 1.0) and 0.3 <= min(x.kp_scale[2:]) and max(x.kp_scale[2:]) <= 1.2
        assert 0.5 <= x.kv_scale[2] <= 1.2 and 0.15 <= x.kv_scale[3] <= 1.2


# ------------------------------------------------------------------------------------------------
# T4: exact undo; the hooks
# ------------------------------------------------------------------------------------------------
@needs_plant
@pytest.mark.parametrize("name", list(servo.NAMED[:-1]) + ["D/3042", "train/7"])
def test_t4_exact_undo(p, name):
    params = (servo.panel_draw(3042) if name == "D/3042" else servo.training_draw(7) if name == "train/7"
              else servo.setting(name))
    params = replace(params, mass=1.0)  # (the trunk mass is the caller's: a test of its own below)
    saved, kp0, kv0 = _arrays(p.m), p.kp.copy(), p.kv.copy()
    env0, keys0 = p.envelope, set(p.__dict__)
    p.reset(seed=3, joint_noise=0.01, root_velocity_noise=0.02)
    with servo.served(p, params) as lay:
        assert isinstance(p.envelope, servo.ServoEnvelope) and "step" in p.__dict__
        for k in range(120):
            p.command(p.q_settled[p.qadr] + 0.02 * math.sin(0.05 * k), np.zeros(14), np.zeros(14))
            p.step()
        p.reset(seed=4, joint_noise=0.01)  # a reset inside: the model's arrays back, the layer armed again
        assert _model_diff(p.m, saved) == ([] if params.peak == (None,) * 4 else ["actuator_forcerange"])
        p.step(30)
        assert lay.readings()["physics_steps"] == 30
    assert _model_diff(p.m, saved) == []
    assert np.array_equal(p.kp, kp0) and np.array_equal(p.kv, kv0)
    assert p.envelope is env0 and env0.bound_to(p.m, p.d)
    assert "step" not in p.__dict__ and set(p.__dict__) == keys0


@needs_plant
def test_t4_undo_after_an_error_and_the_refusals(p):
    saved = _arrays(p.m)
    p.reset(seed=1)
    with pytest.raises(KeyError):
        with servo.served(p, servo.setting("J+")):
            p.step(5)
            raise KeyError("inside")
    assert _model_diff(p.m, saved) == [] and "step" not in p.__dict__
    with servo.served(p, servo.setting("N")):
        with pytest.raises(servo.ServoError, match="envelope"):  # no second layer
            servo.Layer(p, servo.setting("N"))
    p.step = lambda n=1: None
    try:
        with pytest.raises(servo.ServoError, match="step"):  # a digest's or another hook in place
            servo.served(p, servo.setting("N")).__enter__()
    finally:
        del p.step
    old = p.envelope
    p.envelope = drive.Envelope(old.drives, old.moment, torque_scale=0.7)
    try:
        with pytest.raises(servo.ServoError, match="own"):  # the ledger's scaled envelope: the layer owns the line
            servo.Layer(p, servo.setting("N"))
    finally:
        p.envelope = old
        old.bind(p.m, p.d)
    assert _model_diff(p.m, saved) == []


@needs_plant
def test_a_mass_setting_needs_the_callers_trunk_mass(p):
    """M-, M+, W1 and W2 carry a trunk mass factor the layer never applies: refused while the trunk is as built,
    accepted once the caller has applied it (design 5.1)."""
    saved = _arrays(p.m)
    with pytest.raises(servo.ServoError, match="caller"):
        servo.Layer(p, servo.setting("M+"))
    nominal, scratch = copy.deepcopy(p.m), mujoco.MjData(p.m)
    env.set_trunk_mass(p.m, p.trunk, 1.1, nominal, scratch)
    try:
        p.reset(seed=1)
        with servo.served(p, servo.setting("M+")):
            p.step(10)
    finally:
        env.set_trunk_mass(p.m, p.trunk, 1.0, nominal, scratch)
    assert _model_diff(p.m, saved) == []


@needs_plant
def test_t4_a_step_hook_removed_inside_is_refused_and_restored(p):
    saved = _arrays(p.m)
    p.reset(seed=1)
    with pytest.raises(servo.ServoError, match="replaced inside"):
        with servo.served(p, servo.setting("N")):
            p.step(3)
            del p.step  # (policy_rows.digesting pops p.step on exit: trap 8)
            with pytest.raises(servo.ServoError, match="outside the layer"):
                p.step()  # the class's step would skip the layer's after-step: refused
    assert _model_diff(p.m, saved) == [] and "step" not in p.__dict__


@needs_plant
def test_t4_with_the_delay_in_either_order(p):
    """``servo_delay`` replaces command and reset, the layer neither: either order runs the same physics."""
    saved = _arrays(p.m)

    def run(order):
        out = []
        p.reset(seed=5, joint_noise=0.01)
        outer = speedfit.servo_delay(p, 3) if order == "delay first" else servo.served(p, servo.setting("P+"))
        with outer:
            inner = servo.served(p, servo.setting("P+")) if order == "delay first" else speedfit.servo_delay(p, 3)
            with inner:
                for k in range(60):
                    p.command(p.q_settled[p.qadr] + 0.03 * math.sin(0.1 * k), np.zeros(14), np.zeros(14))
                    p.step()
                    out.append(_digest(p))
        return out

    assert run("delay first") == run("layer first")
    assert _model_diff(p.m, saved) == [] and "step" not in p.__dict__ and "command" not in p.__dict__


@needs_plant
def test_hooks_arm_at_entry_and_act_at_the_first_physics_step(p):
    """Before the first physics step only the plant's gain copies are the run's: the reset's forward pass, its first
    reading and any tracker's copy of the friction are today's (traps 1 and 9; design 2d's gain copies)."""
    saved = _arrays(p.m)
    p.reset(seed=2)
    with servo.served(p, servo.setting("J+")) as lay:
        p.reset(seed=2)
        assert _model_diff(p.m, saved) == []
        assert np.array_equal(p.kv, lay.kv_copy) and not np.array_equal(p.kv, lay.kv_copy0)
        assert np.array_equal(p.m.dof_frictionloss, saved["dof_frictionloss"])
        p.step()
        assert np.array_equal(-p.m.actuator_biasprm[:, 2], lay.kv_model)
        assert np.array_equal(p.m.tendon_armature, lay.j_run[lay.tendon_idx])
    assert _model_diff(p.m, saved) == []


@needs_plant
def test_today_is_the_plant_bit_for_bit(p):
    """The setting ``today`` adds nothing of its own: every physics step's state, command, readings, forces and force
    ranges as without the layer, also past the envelope's corner (a plant-level form of T3)."""
    rng = np.random.default_rng(11)
    q0, qd0 = _random_state(p, rng, speed=1.0)
    qd0[p.dofs[5]] = 10.0  # the knee past its corner (8.69 rad/s): the envelope narrows its force range
    cmds = [p.q_settled[p.qadr] + rng.uniform(-0.03, 0.03, 14) for _ in range(150)]

    def run(layer):
        out = []
        p.set_state(q0, qd0)
        with (servo.served(p, servo.setting("today")) if layer else _null()):
            for c in cmds:
                p.command(c, np.zeros(14), np.zeros(14))
                p.step()
                out.append((_digest(p), p.envelope.active))
        return out

    assert run(True) == run(False)
    assert any(a for _, a in run(False))


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------------------------------------------
# T5: friction
# ------------------------------------------------------------------------------------------------
@needs_plant
@pytest.mark.parametrize("name", ["N", "F+", "D"])
def test_t5_friction_arrays_are_the_formula(p, name):
    params = servo.panel_draw(3050) if name == "D" else servo.setting(name)
    p.reset(seed=6, joint_noise=0.01)
    with servo.served(p, params) as lay:
        p.step(20)
        m = p.m
        cls = [servo.CLASSES.index(c) for _, c in servo.DRIVES]
        mj = np.array(params.fric_joint)
        Tc, bv, mu = (np.array([getattr(params, f)[i] for i in cls]) for f in ("fric_c", "fric_v", "fric_load"))
        f_before, tbar_before = p.d.actuator_force.copy(), lay._tbar.copy()
        p.step()  # tau_bar: the motor torque's magnitude of the last step through the 10 ms low-pass
        alpha = 1.0 - math.exp(-0.002 / 0.010)
        assert np.allclose(lay._tbar, tbar_before + alpha * (np.abs(f_before) - tbar_before), rtol=1e-12, atol=0)
        F = servo.friction_loss(mj, Tc, mu, lay._tbar)
        fj = m.dof_frictionloss[lay.joint_dofs]
        ft = m.tendon_frictionloss[lay.tendons]
        assert np.allclose(fj, F[lay.joint_idx], rtol=1e-13, atol=0) and np.allclose(ft, F[lay.tendon_idx], rtol=1e-13)
        assert np.array_equal(m.dof_damping[lay.joint_dofs], (mj * bv)[lay.joint_idx])
        assert np.array_equal(m.tendon_damping[lay.tendons], (mj * bv)[lay.tendon_idx])
        assert not m.dof_frictionloss[lay.ankle_joint_dofs].any() and not m.dof_damping[lay.ankle_joint_dofs].any()
        tbar = np.linspace(0.0, 20.0, 14)  # a set tau_bar
        lay._tbar[:] = tbar
        lay._write_friction()
        F = servo.friction_loss(mj, Tc, mu, tbar)
        assert np.allclose(m.dof_frictionloss[lay.joint_dofs], F[lay.joint_idx], rtol=1e-13, atol=0)
        assert np.allclose(m.tendon_frictionloss[lay.tendons], F[lay.tendon_idx], rtol=1e-13, atol=0)


ONE_JOINT = """<mujoco><option timestep="0.002" integrator="implicitfast" gravity="0 0 0"/>
<worldbody><body><joint name="j" type="hinge" axis="0 0 1" armature="0.01" frictionloss="{F}"/>
<geom type="box" size="0.1 0.02 0.02" mass="1"/></body></worldbody>{act}</mujoco>"""


def test_t5_break_away_on_one_joint_is_the_dry_friction():
    """MuJoCo's friction loss of the value the layer writes holds a joint below it and lets go above it: with tau_bar
    0 and m_j 1 the break-away torque is T_c; under a load it is T_c + mu_L tau_bar.  Below it the joint does not
    speed up (MuJoCo's friction constraint is soft: it creeps at a small constant speed); above it the joint speeds
    up as the applied torque less F over its inertia."""
    for tc, mu, tbar in ((0.2, 0.0, 0.0), (1.6, 0.05, 20.0)):
        F = float(servo.friction_loss(1.0, tc, mu, tbar))
        assert F == pytest.approx(tc + mu * tbar)
        m = mujoco.MjModel.from_xml_string(ONE_JOINT.format(F=F, act=""))
        inertia = float(m.dof_armature[0] + m.body_inertia[1, 2])
        gain = {}
        for share in (0.8, 1.2):
            d = mujoco.MjData(m)
            v = []
            for k in range(500):  # 1 s
                d.qfrc_applied[0] = share * F
                mujoco.mj_step(m, d)
                if k in (249, 499):
                    v.append(float(d.qvel[0]))
            gain[share] = v[1] - v[0]  # the speed gained in the last 0.5 s
        assert gain[1.2] == pytest.approx(0.2 * F * 0.5 / inertia, rel=0.01)
        assert abs(gain[0.8]) < 1e-4 * gain[1.2], gain


# ------------------------------------------------------------------------------------------------
# T6: gear play
# ------------------------------------------------------------------------------------------------
@needs_plant
def test_t6_the_force_is_the_dead_zone_at_random_states(p):
    rng = np.random.default_rng(12)
    params = replace(servo.setting("P+"), enc_offset=tuple(rng.uniform(-0.015, 0.015, 14)))
    with servo.served(p, params) as lay:
        for _ in range(40):
            p.set_state(*_random_state(p, rng))
            p.command(p.q_settled[p.qadr] + rng.uniform(-0.05, 0.05, 14), rng.uniform(-1, 1, 14),
                      rng.uniform(-5, 5, 14))
            lay._before()  # (the first step after the reset: no lag yet, y = c)
            mujoco.mj_forward(p.m, p.d)
            assert np.allclose(p.d.actuator_force, _expected_force(p, lay), rtol=0, atol=1e-9)
            box = lay.box_idx
            assert np.array_equal(lay._sent[box], lay._c[box] - lay.offset[box])  # no dead zone on neck and tail
            p.d.ctrl[:] = lay._c


def test_t6_static_deflection_on_one_joint_is_load_over_kp_plus_half_the_play():
    kp, kv, b, load = 100.0, 2.0, math.radians(1.0) / 2, 2.0
    m = mujoco.MjModel.from_xml_string(ONE_JOINT.format(
        F=0.0, act=f'<actuator><position joint="j" kp="{kp}" kv="{kv}"/></actuator>'))
    d = mujoco.MjData(m)
    c = 0.1
    for _ in range(3000):  # 6 s
        sent, _ = servo.dead_zone(np.array([c]), d.qpos[:1], d.qvel[:1], kp, kv, b)
        d.ctrl[0] = sent[0]
        d.qfrc_applied[0] = load
        mujoco.mj_step(m, d)
    assert float(d.qpos[0]) - c == pytest.approx(load / kp + b, rel=1e-6)


@needs_plant
def test_t6_the_ankle_reads_its_two_motors_errors(p):
    rng = np.random.default_rng(13)
    params = replace(servo.setting("P+"), enc_offset=tuple(rng.uniform(-0.015, 0.015, 14)), cmd_hold=False)
    p.reset(seed=8, joint_noise=0.02)
    with servo.served(p, params) as lay:
        for _ in range(10):
            p.command(p.q_settled[p.qadr] + rng.uniform(-0.05, 0.05, 14), np.zeros(14), np.zeros(14))
            q = p.d.qpos[p.qadr].copy()
            p.step()
            read = p.d.sensordata[lay._q_adr] - q  # the sensors describe the step's start
            e = lay._e
            for side in ("left", "right"):
                ia, ib = (servo.ACTUATORS.index(f"{side}_ankle_{x}_servo") for x in "ab")
                jp, jr = p.joints.index(f"{side}_ankle_pitch"), p.joints.index(f"{side}_ankle_roll")
                assert read[jp] == pytest.approx((e[ia] + e[ib]) / 2, abs=1e-12)
                assert read[jr] == pytest.approx((e[ia] - e[ib]) / 2, abs=1e-12)
            for i in lay.joint_idx:  # a joint drive reads L + o + clip, the clip within half the play
                assert read[i] == pytest.approx(e[i], abs=1e-12) and abs(e[i] - lay.offset[i]) <= lay.b[i] + 1e-15
            assert np.array_equal(e[lay.box_idx], lay.offset[lay.box_idx])  # the neck and tail: the offset only
            assert any(abs(e[i] - lay.offset[i]) > 0 for i in lay.joint_idx if lay.b[i])  # some play shift


# ------------------------------------------------------------------------------------------------
# T7: the torque-speed line and the bus
# ------------------------------------------------------------------------------------------------
def test_t7_the_designs_examples():
    x4z, x6z = servo.TODAY_ZERO["X4"], servo.TODAY_ZERO["X6"]
    lo, hi = servo.line_limits([8.69], [21.0], [5.76], [x4z], servo.zero_speed([x4z], 24.0, 22.2))
    assert hi[0] == pytest.approx(21.0 / (x4z - 5.76) * (x4z * 22.2 / 24 - 8.69)) and hi[0] == pytest.approx(7.4,
                                                                                                         abs=0.05)
    assert lo[0] == -21.0  # braking at the peak
    _, hi = servo.line_limits([8.01], [42.0], [4.16], [x6z], servo.zero_speed([x6z], 48.0, 21.0))
    assert hi[0] == pytest.approx(0.4, abs=0.05)
    assert servo.zero_speed([x4z], 24.0, 25.2)[0] == x4z  # no credit above the nominal voltage


def test_t7_the_sag_is_the_higher_root_and_a_collapse_is_flagged():
    for v0, r, pw in ((22.2, 0.05, 214.0), (21.0, 0.15, 214.0), (25.2, 0.02, 900.0), (21.0, 0.15, 734.9)):
        v, collapsed = servo.bus_voltage(v0, r, pw)
        assert not collapsed and v <= v0 and abs(v - (v0 - r * pw / v)) < 1e-9 and v > v0 / 2
    assert servo.bus_voltage(21.0, 0.15, 214.0)[0] == pytest.approx(19.3, abs=0.05)  # design 2c: below 20 V
    assert servo.bus_voltage(21.0, 0.15, 736.0) == (10.5, True)  # 735 W collapses V- (design 2c)
    assert servo.bus_voltage(22.2, 0.05, 2465.0) == (11.1, True)  # 2,464 W collapses N
    assert servo.bus_voltage(24.0, 0.0, 1e6) == (24.0, False)


@needs_plant
def test_t7_the_force_range_on_the_plant(p):
    knee, hp = servo.ACTUATORS.index("left_knee_servo"), servo.ACTUATORS.index("left_hip_pitch_servo")
    for v0, i, w, want in ((22.2, knee, 8.69, 7.4), (21.0, hp, 8.01, 0.4)):
        with servo.served(p, replace(servo.setting("N"), bus_v0=v0, bus_r=0.0)) as lay:
            q, qd = p.q_settled.copy(), np.zeros(p.m.nv)
            qd[p.dofs[p.joints.index(p.drives[i].coordinate)]] = w
            p.set_state(q, qd)
            lay._before()
            fr = p.m.actuator_forcerange
            assert lay.V == v0 and fr[i, 1] == pytest.approx(want, abs=0.05) and fr[i, 0] == -p.drives[i].peak
            assert fr[i, 1] == pytest.approx(servo.line_limits(
                [w], [lay.peak_run[i]], [lay.corner24[i]], [lay.zero24[i]], lay.zero_at(v0)[i:i + 1])[1][0])
            p.d.ctrl[:] = lay._c


@needs_plant
def test_t7_the_bus_on_the_plant_and_the_supply_window(p):
    p.reset(seed=9, joint_noise=0.01)
    with servo.served(p, servo.setting("V-")) as lay:
        p.step(5)
        f, w_prev = p.d.actuator_force.copy(), lay._w_prev.copy()
        p.step()
        leg = lay.is_leg
        P = (np.maximum(f * w_prev, 0.0)[leg].sum() + (lay.c_cu * f * f).sum()) / 0.92 + lay.aux_w
        assert lay.aux_w == pytest.approx(20.34, abs=0.01)  # the record's auxiliary input
        assert lay.V == pytest.approx(servo.bus_voltage(21.0, 0.15, P)[0], abs=1e-12)
    for r, collapsed in ((1.0, False), (100.0, True)):
        p.reset(seed=9)
        with servo.served(p, replace(servo.setting("N"), bus_v0=21.0, bus_r=r)) as lay:
            p.step(20)
            got = lay.readings()
        assert got["left_supply_window"] and got["steps_below_supply_floor"] > 0
        assert (got["collapsed_steps"] > 0) is collapsed and got["supply_floor_v"] == 20.0
    p.reset(seed=9)
    with servo.served(p, servo.setting("N")) as lay:
        p.step(20)
        assert not lay.readings()["left_supply_window"]


# ------------------------------------------------------------------------------------------------
# T8: the command's lag and the box servos' update
# ------------------------------------------------------------------------------------------------
@needs_plant
def test_t8_a_step_in_the_command_rises_by_the_lag(p):
    lag = (2.0, 3.0, 10.0, 10.0)
    p.reset(seed=1)
    with servo.served(p, _neutral(cmd_lag_ms=lag)) as lay:
        c0 = p.d.ctrl.copy()
        p.step(3)
        assert np.array_equal(lay._y, c0)
        c1 = c0 + 0.01
        p.d.ctrl[:] = c1
        T = np.array([lag[servo.CLASSES.index(c)] for _, c in servo.DRIVES]) * 1e-3
        for n in range(1, 11):
            p.step()
            assert np.allclose((lay._y - c0) / 0.01, 1.0 - np.exp(-n * 0.002 / T), rtol=0, atol=1e-10)
            assert np.array_equal(p.d.ctrl, c1)  # the controller's own command back after the step


@needs_plant
def test_t8_the_box_servos_take_a_new_target_every_fifth_step(p):
    p.reset(seed=1)
    with servo.served(p, replace(servo.setting("N"), hold_phase=2)) as lay:
        box, leg = lay.box_idx, np.flatnonzero(lay.is_leg)
        c0, held, prev_leg = p.d.ctrl.copy(), None, None
        for k in range(23):
            c = c0 + 1e-3 * k
            p.d.ctrl[:] = c
            p.step()
            if k == 0 or (k - 2) % 5 == 0:
                held = c[box].copy()
            assert np.array_equal(lay._sent[box], held), k
            if prev_leg is not None:
                assert not np.array_equal(lay._y[leg], prev_leg)  # the leg drives follow every step
            prev_leg = lay._y[leg].copy()


# ------------------------------------------------------------------------------------------------
# T9: the encoder's zero error and the feedback age
# ------------------------------------------------------------------------------------------------
def test_t9_a_servo_rests_at_target_minus_offset_and_reads_the_target():
    kp, kv, o, c = 50.0, 1.0, 0.012, 0.3
    m = mujoco.MjModel.from_xml_string(ONE_JOINT.format(
        F=0.0, act=f'<actuator><position joint="j" kp="{kp}" kv="{kv}"/></actuator>'))
    d = mujoco.MjData(m)
    for _ in range(2000):
        sent, clip = servo.dead_zone(np.array([c - o]), d.qpos[:1], d.qvel[:1], kp, kv, 0.0)
        d.ctrl[0] = sent[0]
        mujoco.mj_step(m, d)
    assert float(d.qpos[0]) == pytest.approx(c - o, abs=1e-9)
    assert float(d.qpos[0]) + o + float(clip[0]) == pytest.approx(c, abs=1e-9)  # the reported angle: L + o


@needs_plant
def test_t9_the_reading_carries_the_offset_and_the_buffer_its_age(p):
    rng = np.random.default_rng(14)
    off = tuple(rng.uniform(-0.015, 0.015, 14))
    p.reset(seed=10, joint_noise=0.01)
    with servo.served(p, _neutral(enc_offset=off)) as lay:
        for _ in range(5):
            c = p.d.ctrl.copy()
            q = p.d.qpos[p.qadr].copy()
            p.step()
            assert np.array_equal(lay._sent, c - np.array(off))
            assert np.allclose(p.d.sensordata[lay._q_adr] - q, np.linalg.inv(p.G) @ np.array(off), rtol=0, atol=1e-14)

    q_adr = [p._sens[f"enc_q_{jn}"][0] for jn in p.joints]
    qd_adr = [p._sens[f"enc_qd_{jn}"][0] for jn in p.joints]

    def readings(n):
        p.reset(seed=10, joint_noise=0.01)
        first = np.r_[p.d.sensordata[q_adr], p.d.sensordata[qd_adr]]
        out = []
        with servo.served(p, _neutral(sense_steps=n)):
            for _ in range(12):
                p.step()
                out.append(np.r_[p.d.sensordata[q_adr], p.d.sensordata[qd_adr]].copy())
        return first, out

    first, now = readings(0)
    _, aged = readings(2)
    assert np.array_equal(aged[0], first) and np.array_equal(aged[1], first)  # the reset's reading, exact
    for k in range(2, 12):
        assert np.array_equal(aged[k], now[k - 2])


# ------------------------------------------------------------------------------------------------
# T10: the reflected inertia
# ------------------------------------------------------------------------------------------------
@needs_plant
def test_t10_the_mass_matrix_rises_by_the_drawn_difference(p):
    def motor_diag(m, q):
        d = mujoco.MjData(m)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        Gi = np.linalg.inv(p.G)
        return np.diag(Gi.T @ M[np.ix_(p.dofs, p.dofs)] @ Gi), M

    p.reset(seed=1)
    with servo.served(p, servo.setting("J+")) as lay:
        p.step()
        q = p.d.qpos.copy()
        inside, M1 = motor_diag(p.m, q)
        j_run, j0 = lay.j_run.copy(), lay.j_found.copy()
        inertia = np.array([p.meta["gains"]["drives"][a]["inertia_kg_m2"] for a in servo.ACTUATORS])
        kv = servo.kv_rule(lay.kp_copy0, inertia, j_run, j0)
        assert np.allclose(-p.m.actuator_biasprm[:, 2], kv, rtol=1e-14) and np.allclose(p.kv, kv, rtol=1e-14)
    outside, M0 = motor_diag(p.m, q)
    assert np.allclose(inside - outside, j_run - j0, rtol=1e-9, atol=1e-14)
    off = ~np.eye(p.m.nv, dtype=bool)
    off[np.ix_(p.dofs, p.dofs)] = False  # (the ankle's pitch-roll pair couples through its motors' tendons)
    assert np.allclose(M1[off], M0[off], rtol=0, atol=1e-12)
    with servo.served(p, servo.setting("N")) as lay:
        t = servo.ACTUATORS.index("tail_yaw_servo")
        assert round(float(lay.kv_copy[t]), 2) == 3.59 and round(float(lay.kv_copy0[t]), 2) == 2.72


@needs_plant
def test_t10_set_const_after_the_undo_leaves_the_model_bit_for_bit(p):
    saved = _arrays(p.m)
    p.reset(seed=1)
    with servo.served(p, replace(servo.setting("W1"), mass=1.0)):
        p.step(10)
    assert _model_diff(p.m, saved) == []
    assert env.setconst_changes(p.m) == []


@needs_plant
def test_t10_the_layer_restores_every_array_set_const_writes(p):
    """Each float array of a copy perturbed in turn: those mj_setConst writes back are all in SETCONST_FIELDS."""
    m = p.m
    written = []
    for k in dir(m):
        a = getattr(m, k)
        if k.startswith("_") or not isinstance(a, np.ndarray) or a.dtype.kind != "f" or not a.size:
            continue
        m2 = copy.deepcopy(m)
        getattr(m2, k)[...] += 1.0
        mujoco.mj_setConst(m2, mujoco.MjData(m2))
        if not np.array_equal(getattr(m2, k), a + 1.0):
            written.append(k)
    assert set(written) <= set(servo.SETCONST_FIELDS)


# ------------------------------------------------------------------------------------------------
# T11: the judge's registered limits
# ------------------------------------------------------------------------------------------------
@needs_plant
def test_t11_the_judge_reads_the_nominal_limits_under_every_setting(p):
    env0, drives0 = p.envelope, list(p.drives)
    for params in [replace(servo.setting(n), mass=1.0) for n in servo.NAMED[:-1]] + [servo.panel_draw(3042)]:
        p.reset(seed=1)
        with servo.served(p, params):
            p.step(3)
            e = p.envelope
            for f in ("peak", "rated", "no_load", "moment"):
                assert np.array_equal(getattr(e, f), getattr(env0, f)), (params.name, f)
            assert p.drives == drives0 and all(a is b for a, b in zip(p.drives, drives0, strict=True))
    with servo.served(p, servo.setting("X6L")) as lay:
        i = servo.ACTUATORS.index("left_hip_pitch_servo")
        assert lay.peak_run[i] == pytest.approx(31.5) and p.envelope.peak[i] == 42.0


# ------------------------------------------------------------------------------------------------
# T14: the gain copies
# ------------------------------------------------------------------------------------------------
@needs_plant
@pytest.mark.parametrize("case", ["J+", "G5", "tail N"])
def test_t14_the_force_uses_the_runs_gains(p, case):
    params = {"J+": _neutral(armature=servo.setting("J+").armature), "G5": _neutral(gain_cap=True),
              "tail N": _neutral(armature=servo.setting("N").armature)}[case]
    rng = np.random.default_rng(15)
    G, GinvT = p.G, np.linalg.inv(p.G.T)
    with servo.served(p, params) as lay:
        if case == "G5":
            assert lay.kp_copy[lay.is_leg].max() <= 500.0 and lay.kv_copy[lay.is_leg].max() <= 5.0
        if case == "tail N":
            assert round(float(lay.kv_model[servo.ACTUATORS.index("tail_yaw_servo")]), 2) == 3.59
        for _ in range(30):
            p.set_state(*_random_state(p, rng, speed=0.5))
            qs, qds, tau = (p.q_settled[p.qadr] + rng.uniform(-0.02, 0.02, 14), rng.uniform(-0.5, 0.5, 14),
                            rng.uniform(-1, 1, 14))
            p.command(qs, qds, tau)
            lay._before()
            mujoco.mj_forward(p.m, p.d)
            kp, kv = p.m.actuator_gainprm[:, 0], -p.m.actuator_biasprm[:, 2]
            assert np.array_equal(kp, lay.kp_model) and np.array_equal(kv, lay.kv_model)
            q, qd = p.d.qpos[p.qadr], p.d.qvel[p.dofs]
            want = kp * (G @ qs - G @ q) + kv * (G @ qds - G @ qd) + GinvT @ tau
            want = np.clip(want, p.m.actuator_forcerange[:, 0], p.m.actuator_forcerange[:, 1])
            assert np.allclose(p.d.actuator_force, want, rtol=0, atol=1e-9)
            p.d.ctrl[:] = lay._c
    assert np.array_equal(p.kp, lay.kp_copy0) and np.array_equal(p.kv, lay.kv_copy0)


@needs_plant
def test_t14_the_box_servos_scales_leave_the_copies_nominal(p):
    s, v = (1.0, 1.0, 0.4, 0.7), (1.0, 1.0, 0.6, 0.2)
    rng = np.random.default_rng(16)
    with servo.served(p, _neutral(kp_scale=s, kv_scale=v)) as lay:
        assert np.array_equal(p.kp, lay.kp_copy0) and np.array_equal(p.kv, lay.kv_copy0)
        for _ in range(20):
            p.set_state(*_random_state(p, rng, speed=0.5))
            qs, qds, tau = (p.q_settled[p.qadr] + rng.uniform(-0.02, 0.02, 14), rng.uniform(-0.5, 0.5, 14),
                            rng.uniform(-0.3, 0.3, 14))
            p.command(qs, qds, tau)
            lay._before()
            mujoco.mj_forward(p.m, p.d)
            q, qd = p.d.qpos[p.qadr], p.d.qvel[p.dofs]
            for i, c in ((0, "neck"), (1, "tail")):
                k = servo.CLASSES.index(c)
                kp, kv = lay.kp_copy0[i], lay.kv_copy0[i]
                want = s[k] * (kp * (qs[i] - q[i]) + kv * qds[i] + tau[i]) - v[k] * kv * qd[i]
                want = min(max(want, p.m.actuator_forcerange[i, 0]), p.m.actuator_forcerange[i, 1])
                assert p.d.actuator_force[i] == pytest.approx(want, abs=1e-9)
            p.d.ctrl[:] = lay._c
