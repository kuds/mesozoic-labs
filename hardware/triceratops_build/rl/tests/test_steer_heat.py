"""The steering run's training additions (D26, D29): the hottest drive's heat cost (r_heat_peak) and the whole-robot
mass and payload factors. Each is off by default; the regression digests in test_trike_env.py cover that."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from conftest import RL_DIR  # noqa: F401

from trike_env import TrikeEnv, TrikeEnvConfig, reward_terms

FORE128 = str(RL_DIR.parent / "sim" / "generated" / "trike60_C_closed_capsule_fore128.xml")


def total_mass(env):
    return float(env.model.body_mass.sum())


def test_heat_peak_term_is_exactly_zero_when_off():
    e = TrikeEnv(TrikeEnvConfig(model_path=FORE128))
    e.reset(seed=3)
    for _ in range(20):
        _, _, _, _, info = e.step(np.zeros(12))
    assert info["r_heat_peak"] == 0.0 and e.heat_peak_rms == 0.0


def test_heat_peak_reward_term_arithmetic():
    cfg = TrikeEnvConfig(w_heat_peak=2.0)
    base = dict(v_fwd=0.25, v_cmd=0.25, v_lat=0.0, yaw_rate=0.0, yaw_err=0.0, grav_b=np.array([0.0, 0.0, -1.0]),
                z=0.3, z_nominal=0.3, mean_power=0.0, action=np.zeros(12), prev_action=np.zeros(12), slip=0.0,
                contact=np.ones(4, dtype=bool))
    t = reward_terms(cfg, late_scale=0.5, heat_peak_excess=0.2, **base)
    assert t["r_heat_peak"] == pytest.approx(-2.0 * 0.5 * 0.2)
    assert reward_terms(cfg, late_scale=0.0, heat_peak_excess=0.2, **base)["r_heat_peak"] == 0.0


def test_heat_peak_follows_an_independent_recomputation():
    """With the cost on, the env's hottest-drive RMS equals a recomputation from the per-physics-step torques (read by
    a substep hook): each drive's step mean of (torque / rating)^2, low-passed with heat_peak_tau_s."""
    cfg = TrikeEnvConfig(model_path=FORE128, w_heat_peak=1.0, heat_peak_margin=0.05, heat_peak_tau_s=0.5,
                         domain_randomization=False, observation_noise=False)
    e = TrikeEnv(cfg)
    e.reset(seed=5)
    e.late_scale = 1.0
    inv_c2 = e._inv_cont2.copy()
    acc = []
    e.substep_hook = lambda env: acc.append(env.data.actuator_force[env._leg_ctrl_index] ** 2 * inv_c2)
    ema = np.zeros(12)
    alpha = 1.0 - math.exp(-e.dt / cfg.heat_peak_tau_s)
    rng = np.random.default_rng(0)
    for _ in range(40):
        acc.clear()
        _, _, term, trunc, info = e.step(rng.uniform(-0.5, 0.5, 12))
        ema += alpha * (np.mean(acc, axis=0) - ema)
        peak = math.sqrt(ema.max())
        assert e.heat_peak_rms == pytest.approx(peak, rel=1e-12)
        assert info["heat_peak_rms"] == pytest.approx(peak, rel=1e-12)
        assert info["r_heat_peak"] == pytest.approx(-1.0 * max(0.0, peak - 0.05), rel=1e-12)
        if term or trunc:
            break
    assert e.heat_peak_rms > 0.05  # the term was exercised


def test_whole_robot_mass_and_payload_apply_without_accumulating():
    nominal = TrikeEnv(TrikeEnvConfig(model_path=FORE128, domain_randomization=False))
    nominal.reset(seed=1)
    m0 = total_mass(nominal)
    cfg = TrikeEnvConfig(model_path=FORE128, mass_scale_range=(1.0, 1.0), mass_scale_whole_range=(1.2, 1.2),
                         payload_kg_range=(0.5, 0.5))
    e = TrikeEnv(cfg)
    c = e._chassis
    for seed in (1, 2, 3):  # each reset starts from the nominal model again
        _, info = e.reset(seed=seed)
        assert total_mass(e) == pytest.approx(1.2 * m0 + 0.5, rel=1e-12)
        assert e.factors["mass_scale_whole"] == pytest.approx(1.2) and e.factors["payload_kg"] == pytest.approx(0.5)
        # the chassis inertia scaled by the chassis's own mass ratio (as the sweep's payload)
        ratio = (1.2 * e._nominal["body_mass"][c] + 0.5) / (1.2 * e._nominal["body_mass"][c])
        assert np.allclose(e.model.body_inertia[c], 1.2 * e._nominal["body_inertia"][c] * ratio, rtol=1e-12)
        # the derived constants follow (mj_setConst): the root's subtree mass is the whole robot
        assert e.model.body_subtreemass[1] == pytest.approx(total_mass(e), rel=1e-9)


def test_whole_robot_factors_draw_after_every_other_draw_and_only_apply_with_randomisation():
    """The two new uniforms come after every older draw, so a seed's delay, command, friction and per-body factors are
    those of the config without them; with domain randomisation off they are drawn but not applied."""
    base = TrikeEnvConfig(model_path=FORE128, delay_steps_range=(0, 20))
    with_new = base.replace(mass_scale_whole_range=(0.9, 1.1), payload_kg_range=(0.0, 1.0))
    a, b = TrikeEnv(base), TrikeEnv(with_new)
    for seed in (7, 8):
        _, ia = a.reset(seed=seed)
        _, ib = b.reset(seed=seed)
        assert ia["delay_steps"] == ib["delay_steps"] and ia["command_speed"] == ib["command_speed"]
        assert ia["friction"] == ib["friction"]
        assert np.array_equal(a.factors["mass_scale"], b.factors["mass_scale"])
        assert 0.9 <= b.factors["mass_scale_whole"] <= 1.1 and 0.0 <= b.factors["payload_kg"] <= 1.0
    off = TrikeEnv(with_new.replace(domain_randomization=False))
    nominal = TrikeEnv(base.replace(domain_randomization=False))
    off.reset(seed=7)
    nominal.reset(seed=7)
    assert total_mass(off) == total_mass(nominal) and "mass_scale_whole" not in off.factors


def test_steering_draw_unchanged_by_the_new_factors():
    base = TrikeEnvConfig(model_path=FORE128, command_mode="heading_and_speed")
    a, b = TrikeEnv(base), TrikeEnv(base.replace(mass_scale_whole_range=(0.9, 1.1), payload_kg_range=(0.0, 1.0)))
    a.reset(seed=4)
    b.reset(seed=4)
    for _ in range(300):  # the request schedule (heading and speed) is the same for 6 s
        assert a._cmd_exec["desired_heading"] == pytest.approx(b._cmd_exec["desired_heading"])
        assert a._cmd_exec["desired_speed"] == pytest.approx(b._cmd_exec["desired_speed"])
        a.step(np.zeros(12))
        b.step(np.zeros(12))


@pytest.mark.parametrize("bad", [dict(w_heat_peak=-1.0), dict(heat_peak_margin=0.0), dict(heat_peak_tau_s=0.0),
                                 dict(mass_scale_whole_range=(1.1, 0.9)), dict(mass_scale_whole_range=(0.0, 1.0)),
                                 dict(payload_kg_range=(-0.5, 1.0))])
def test_bad_settings_are_refused(bad):
    with pytest.raises(ValueError):
        TrikeEnv(TrikeEnvConfig(model_path=FORE128, **bad))


def test_delay_range_to_20_ms_reaches_its_ends():
    e = TrikeEnv(TrikeEnvConfig(model_path=FORE128, delay_steps_range=(0, 20)))
    seen = {e.reset(seed=s)[1]["delay_steps"] for s in range(200)}
    assert min(seen) == 0 and max(seen) == 20
    e.reset(seed=0, options={"delay_steps": 20})
    for _ in range(5):
        e.step(np.zeros(12))
    assert len(e._fifo) == 21 and np.isfinite(e.data.qpos).all() and mujoco.__version__
