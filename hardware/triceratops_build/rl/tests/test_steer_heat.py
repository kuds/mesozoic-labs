"""The steering run's training additions (D26, D29): the hottest drive's heat cost (r_heat_peak) and the whole-robot
mass and payload factors. Each is off by default; the regression digests in test_trike_env.py cover that."""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest
from conftest import RL_DIR  # noqa: F401

from trike_env import FLAT_FLOOR, TrikeEnv, TrikeEnvConfig, flat_floor, heat_peak_update, reward_terms

FORE128 = str(RL_DIR.parent / "sim" / "generated" / "trike60_C_closed_capsule_fore128.xml")
CONFIGS = RL_DIR / "configs"


def total_mass(env):
    return float(env.model.body_mass.sum())


def test_heat_peak_term_is_exactly_zero_when_off():
    e = TrikeEnv(TrikeEnvConfig(model_path=FORE128))
    e.reset(seed=3)
    for _ in range(20):
        _, _, _, _, info = e.step(np.zeros(12))
    assert info["r_heat_peak"] == 0.0 and "heat_peak_rms" not in info  # no figure that looks like a measurement


def test_heat_peak_reward_term_arithmetic():
    cfg = TrikeEnvConfig(w_heat_peak=2.0)
    base = dict(v_fwd=0.25, v_cmd=0.25, v_lat=0.0, yaw_rate=0.0, yaw_err=0.0, grav_b=np.array([0.0, 0.0, -1.0]),
                z=0.3, z_nominal=0.3, mean_power=0.0, action=np.zeros(12), prev_action=np.zeros(12), slip=0.0,
                contact=np.ones(4, dtype=bool))
    t = reward_terms(cfg, late_scale=0.5, heat_peak_excess=0.2, **base)
    assert t["r_heat_peak"] == pytest.approx(-2.0 * 0.5 * 0.2)
    assert reward_terms(cfg, late_scale=0.0, heat_peak_excess=0.2, **base)["r_heat_peak"] == 0.0


def test_heat_peak_update_starts_at_the_margin_and_weights_every_step_alike():
    """From margin^2, a constant input x gives x + (margin^2 - x)(1 - alpha)^n; the average is linear in its inputs,
    with the same weight for a step wherever it falls."""
    alpha = 1.0 - math.exp(-0.02 / 4.0)
    m2 = 0.8 ** 2
    x = np.array([0.25, 0.81, 1.44])
    ema = np.full(3, m2)
    for n in range(1, 400):
        got = heat_peak_update(ema, x, alpha)
        assert np.allclose(got ** 2, x + (m2 - x) * (1.0 - alpha) ** n, rtol=1e-12)
    # one extra unit of heat at step k moves the average at step N by alpha (1 - alpha)^(N - k), for any k
    base, bump = np.full(1, m2), np.full(1, m2)
    for n in range(300):
        heat_peak_update(base, np.array([0.5]), alpha)
        heat_peak_update(bump, np.array([0.5 + (1.0 if n == 10 else 0.0)]), alpha)
    assert bump[0] - base[0] == pytest.approx(alpha * (1.0 - alpha) ** 289, rel=1e-9)


def _square_wave_cost(rms, stance_first=True, steps=900, period=74, margin=0.8):
    """A stride-like input (about 0.68 Hz at 20 ms steps): (torque / rating)^2 at 2 rms^2 for half a stride, 0 for
    the other half. Returns the per-step cost at weight 1 from the average."""
    alpha = 1.0 - math.exp(-0.02 / 4.0)
    ema = np.full(1, margin ** 2)
    out = []
    for n in range(steps):
        on = ((n if stance_first else n + period // 2) % period) < period // 2
        rms_now = heat_peak_update(ema, np.array([2 * rms * rms if on else 0.0]), alpha)[0]
        out.append(max(0.0, rms_now - margin))
    return np.array(out)


@pytest.mark.parametrize("period", [50, 60, 74, 90])  # strides of 1.0-1.8 s at 20 ms steps (the trot: about 74)
def test_heat_peak_prices_a_stride_like_gait_without_an_early_spike(period):
    """A gait under the margin (RMS 0.75) costs at most 0.06 per step (in its first stride, when it starts in stance)
    and under 0.005 after 8 s; over the margin (RMS 0.95) the 18 s cost is 0.7-0.9 of F2's own price, 0.15 per step,
    with under 8 % of it in the first 100 steps (11 % of the time). The bias-corrected average, the second version,
    gave 0.26 on the cool gait's first step, 1.16-1.25 of the price and 20-23 % in the first 100 steps when started in
    stance (started in swing it would pass, so both phases are run)."""
    for first in (True, False):
        cool = _square_wave_cost(0.75, first, period=period)
        assert cool.max() < 0.06 and cool[400:].max() < 0.005 and cool.mean() < 0.005
        hot = _square_wave_cost(0.95, first, period=period)
        assert 0.7 * 0.15 < hot.mean() < 0.9 * 0.15
        assert hot[:100].sum() < 0.08 * hot.sum()


def test_heat_peak_follows_an_independent_recomputation():
    """With the cost on, the env's hottest-drive RMS and cost equal a recomputation from the per-physics-step torques
    (read by a substep hook): from heat_peak_start_s on, each drive's step mean of (torque / rating)^2, low-passed with
    heat_peak_tau_s from heat_peak_margin^2; 0 before. A second episode starts from the margin again."""
    cfg = TrikeEnvConfig(model_path=FORE128, w_heat_peak=1.5, heat_peak_margin=0.3, heat_peak_tau_s=0.5,
                         heat_peak_start_s=0.2, domain_randomization=False,
                         observation_noise=False)
    e = TrikeEnv(cfg)
    inv_c2 = e._inv_cont2.copy()
    acc = []
    e.substep_hook = lambda env: acc.append(env.data.actuator_force[env._leg_ctrl_index] ** 2 * inv_c2)
    alpha = 1.0 - math.exp(-e.dt / cfg.heat_peak_tau_s)
    for episode in range(2):
        e.reset(seed=5)
        e.late_scale = 1.0
        ema, n = np.full(12, 0.3 ** 2), 0
        rng = np.random.default_rng(0)
        trace = []
        for k in range(1, 41):
            acc.clear()
            _, _, term, trunc, info = e.step(rng.uniform(-0.5, 0.5, 12))
            if (k - 1) * e.dt < cfg.heat_peak_start_s - 1e-9:  # this step began before the start
                assert info["r_heat_peak"] == 0.0 and info["heat_peak_rms"] == 0.0
            else:
                n += 1
                ema += alpha * (np.mean(acc, axis=0) - ema)
                rms = np.sqrt(ema)
                assert e.heat_peak_rms == pytest.approx(rms.max(), rel=1e-12)
                assert info["heat_peak_rms"] == pytest.approx(rms.max(), rel=1e-12)
                assert info["r_heat_peak"] == pytest.approx(-1.5 * max(0.0, rms.max() - 0.3), rel=1e-12)
            trace.append(info["r_heat_peak"])
            if term or trunc:
                break
        assert n == 30 and e.heat_peak_rms > 0.3  # the term was exercised from step 11 (0.2 s) on
        if episode == 0:
            first = trace
        else:
            assert trace == first  # the average restarts from the margin at reset


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


def _steps(env, n=200):
    out = []
    for _ in range(n):
        env.step(np.zeros(12))
        out.append(env.data.qpos.copy())
    return np.array(out)


def test_whole_robot_factors_keep_the_reset_state():
    """The factors are applied after the reset state is placed (home keyframe, leg joint noise, root at the sole
    clearance); recomputing the constants must not move it. With factors that change no mass, the reset state and a
    200-step trajectory equal those of the config without them, bit for bit; the neck stays inside its range."""
    base = TrikeEnvConfig(model_path=FORE128, observation_noise=False)
    unit = base.replace(mass_scale_whole_range=(1.0, 1.0), payload_kg_range=(0.0, 0.0))
    heavy = base.replace(mass_scale_whole_range=(1.2, 1.2), payload_kg_range=(1.0, 1.0))
    a, b, c = TrikeEnv(base), TrikeEnv(unit), TrikeEnv(heavy)
    neck = c.model.joint("neck_pitch")
    for seed in (0, 1, 2):
        a.reset(seed=seed)
        b.reset(seed=seed)
        c.reset(seed=seed)
        assert np.array_equal(a.data.qpos, b.data.qpos) and np.array_equal(a.data.qpos, c.data.qpos)
        legs = a.data.qpos[a._leg_qadr] - a.model.key_qpos[a._key][a._leg_qadr]
        assert np.abs(legs).max() > 1e-3  # the reset noise is there
        assert c._lowest_sole_z() == pytest.approx(c.cfg.reset_sole_clearance, abs=1e-9)
        lo, hi = neck.range
        assert lo <= c.data.qpos[neck.qposadr[0]] <= hi
        assert np.array_equal(_steps(a), _steps(b))


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
                                 dict(payload_kg_range=(-0.5, 1.0)), dict(heat_peak_start_s=-1.0)])
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
    assert len(e._fifo) == 21 and np.isfinite(e.data.qpos).all()


def test_measuring_configs_drop_the_whole_robot_draws():
    """Evaluations and the analysis tools measure with the per-body draws only, as every run before the steering run:
    the three flat-floor reductions turn the whole-robot mass and payload draws off, and the measured robot's mass
    is that of the same config without them. The sweep's mass and payload points carry the heavy cases."""
    sys.path.insert(0, str(RL_DIR / "analysis"))
    sys.path.insert(0, str(RL_DIR / "robustness"))
    import gait_pipe1
    import sweep
    from train import flat_eval_config

    cfg = TrikeEnvConfig.from_dict({**TrikeEnvConfig(model_path=FORE128).to_dict(),
                                    **json.loads((CONFIGS / "steer1_dev_heat1.json").read_text())})
    assert cfg.mass_scale_whole_range is not None and cfg.payload_kg_range is not None
    plain = cfg.replace(mass_scale_whole_range=None, payload_kg_range=None)
    for f in (flat_floor, flat_eval_config, gait_pipe1.flat_floor):
        c = f(cfg)
        assert c.mass_scale_whole_range is None and c.payload_kg_range is None and c.domain_randomization
        e, p = TrikeEnv(c), TrikeEnv(flat_floor(plain))
        for seed in (100000, 100001):
            e.reset(seed=seed)
            p.reset(seed=seed)
            assert np.array_equal(e.model.body_mass, p.model.body_mass)
    assert sweep.FLAT_FLOOR == FLAT_FLOOR
    ec = json.loads(json.dumps(cfg.to_dict()))
    assert sweep.config_diffs(ec, flat_floor(cfg)) == {}


def test_sweep_reads_the_training_ranges_from_the_record():
    sys.path.insert(0, str(RL_DIR / "robustness"))
    import sweep

    cfg = TrikeEnvConfig.from_dict({**TrikeEnvConfig(model_path=FORE128).to_dict(),
                                    **json.loads((CONFIGS / "steer1_dev_heat1.json").read_text())})
    tr = sweep.training_ranges(json.loads(json.dumps(cfg.to_dict())))
    assert tr["delay_ms"] == (0, 20) and tr["friction"] == (0.3, 1.0) and tr["mass_scale"] == (0.9, 1.1)
    assert tr["mass_whole"] == (0.9, 1.1) and tr["payload_kg"] == (0.0, 1.0)
    assert tr["terrain_boxes"] == 0 and tr["pushes"] == (0.5, 1.0)
    line = sweep.training_line(tr)
    assert "delay 0-20 ms" in line and "x0.9-1.1 for the whole robot" in line and "0-1 kg payload" in line
    assert line.endswith("flat floor, sideways pushes in 50 % of episodes, up to 1 m/s.")
    old = sweep.training_ranges(TrikeEnvConfig().to_dict())
    assert {k: old[k] for k in sweep.TRAINING} == sweep.TRAINING and old["mass_whole"] is None
    assert sweep.training_line(old).endswith("flat floor, no pushes.")
    assert sweep.training_line(dict(sweep.TRAINING)) == sweep.OLD_TRAINING_LINE  # a result written before


def test_steering_presets_match_the_plan_and_the_notebook_menu():
    """Each steer1 preset, as train.py merges it over the defaults: the plan's settings (PLAN_STEERING_RUN.md; D25,
    D26, D28-D30). The notebook's ENV_PRESET menu and its check list exactly the presets in configs/."""
    presets = sorted(p.stem for p in CONFIGS.glob("*.json"))
    steer = [p for p in presets if p.startswith("steer1_")]
    assert steer == ["steer1_dev_heat1", "steer1_dev_heat2"]
    weights = []
    for name in steer:
        d = json.loads((CONFIGS / f"{name}.json").read_text())
        cfg = TrikeEnvConfig.from_dict({**TrikeEnvConfig(model_path=FORE128).to_dict(), **d})
        assert cfg.command_mode == "heading_and_speed" and cfg.terrain_boxes == 0  # D20; D28: flat
        assert (cfg.push_prob, cfg.push_dv_max_mps, cfg.friction_low_hard) == (0.5, 1.0, 0.3)
        assert cfg.delay_steps_range == (0, 20) and cfg.mass_scale_whole_range == (0.9, 1.1)  # D26
        assert cfg.payload_kg_range == (0.0, 1.0) and cfg.w_heat == 0.3  # D29: today's average kept
        assert (cfg.heat_peak_margin, cfg.heat_peak_tau_s, cfg.heat_peak_start_s) == (0.8, 4.0, 2.0)
        weights.append(cfg.w_heat_peak)
        assert (cfg.w_lateral_drift, cfg.slip_at_contact) == (100.0, True)  # as run 3
        assert cfg.domain_randomization and cfg.observation_noise
    assert weights == [1.0, 2.0]  # the dev runs' two weights (README: the reward margin over standing)
    nb = json.loads((RL_DIR / "trike60_training.ipynb").read_text())
    src = "".join(nb["cells"][1]["source"])
    menu = src[src.index("ENV_PRESET = "):].splitlines()[0]
    assert sorted(x for x in json.loads(menu[menu.index("["):menu.index("]") + 1]) if x) == presets
    check = src[src.index("if ENV_PRESET not in ("):].splitlines()[0]
    assert sorted(x.strip(' "') for x in check[check.index("(") + 1:check.index(")")].split(",") if x.strip(' "')) \
        == presets
    assert 'ENV_PRESET.startswith("steer1_") and MODEL != "trike60_C_closed_capsule_fore128"' in src


def test_weight_script_prices_both_costs_at_the_late_scale():
    """steer_heat_weight.episode: at late scale 0.5 the hottest-drive cost and the every-drive reference are both half
    of what they are at full scale (same episode: the late scale changes no action), and their ratio is unchanged."""
    sys.path.insert(0, str(RL_DIR / "analysis"))
    import steer_heat_weight as shw

    cfg = TrikeEnvConfig(model_path=FORE128).replace(**shw.HEAT)
    out = {}
    for late in (1.0, 0.5):
        env = TrikeEnv(flat_floor(cfg))
        env.late_scale, env.penalty_scale = late, 1.0
        out[late] = shw.episode(env, None, (None, None, None, None), 100000, 0.25, 0.2)
    assert out[1.0]["cost_w1"] > 0.05 and out[1.0]["cost_sum_w1"] > out[1.0]["cost_w1"]
    for k in ("cost_w1", "cost_sum_w1"):
        assert out[0.5][k] == pytest.approx(0.5 * out[1.0][k], rel=1e-9)


@pytest.mark.parametrize("args", [["--late", "0.3"], ["--late", "1.5"], ["--stds", "0.1,0.1"],
                                  ["--stds", "0.125,0.1250001"], ["--stds", "nan"], ["--stds", "-0.1"]])
def test_weight_script_refuses_bad_options_before_running(args, tmp_path):
    import subprocess

    p = subprocess.run([sys.executable, str(RL_DIR / "analysis" / "steer_heat_weight.py"), "--runs-dir",
                        str(tmp_path), "--out", str(tmp_path / "out"), *args], capture_output=True, text=True,
                       timeout=120)
    assert p.returncode != 0 and ("--late" in p.stderr or "--stds" in p.stderr) and not (tmp_path / "out").exists()


def test_noise_reader_keeps_the_latest_branch_and_skips_bad_rows(tmp_path):
    """analysis/d22_action_std.py, which the judged-weight rule reads a dev run's 3 M noise with: a resumed run's
    discarded branch is dropped, a torn line is skipped, a null action_std counts as missing, and a point with no
    rollout within 8192 steps is null."""
    sys.path.insert(0, str(RL_DIR / "analysis"))
    import d22_action_std as das

    rows = []
    for n in range(8192, 3_280_000, 8192):  # session A, killed at 3.28 M
        rows.append({"session": 0, "session_start": 0, "timesteps": n, "action_std": 0.15})
    for n in range(2_750_000 + 8192, 4_000_000, 8192):  # session B resumes from 2.75 M
        rows.append({"session": 1, "session_start": 2_750_000, "timesteps": n, "action_std": 0.12})
    rows[-1]["action_std"] = None  # a non-finite std, written as null
    lines = [json.dumps(r) for r in rows]
    lines.insert(len(lines) // 2, '{"session": 1, "timest')  # a torn line
    p = tmp_path / "run.jsonl"
    p.write_text("\n".join(lines) + "\n")
    t = das.trajectory(p, points=(1_000_000, 3_000_000, 5_000_000))
    assert t["action_std"]["1000000"]["action_std"] == 0.15  # before the resume point: session A's row stands
    assert t["action_std"]["3000000"]["action_std"] == 0.12  # A's discarded rows past 2.75 M are not used
    assert t["action_std"]["5000000"] is None  # not reached
    assert t["last_timesteps"] < 4_000_000 and all(isinstance(v, dict) or v is None for v in t["action_std"].values())
