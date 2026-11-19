"""Tests for the 60 cm Triceratops RL setup: the environment, train.py, evaluate.py and the notebook.

Simulation only. Run by hand (they are not in the repository's pytest testpaths or CI):
    python -m pytest hardware/triceratops_build/rl/tests
"""

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
from trike_env import (
    CORNERS,
    LEG_ACTUATORS,
    REWARD_BONUSES,
    REWARD_TERMS,
    TrikeEnv,
    TrikeEnvConfig,
    diagonal_score,
    reward_terms,
)

RL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_DIR.parents[2]
NOTEBOOK = RL_DIR / "trike60_training.ipynb"
QUIET = dict(domain_randomization=False, observation_noise=False, delay_steps_fixed=0)


@pytest.fixture(scope="module")
def env():
    e = TrikeEnv()
    yield e
    e.close()


def _run(*args, timeout=600):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, timeout=timeout,
                          cwd=str(RL_DIR))


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_record.json").read_text())


def _pair_ok(ckpt: Path, steps: int) -> bool:
    from environments.shared.curriculum.checkpoints import checkpoint_pair_problem

    return checkpoint_pair_problem(ckpt / f"trike60_{steps}_steps.zip",
                                   ckpt / f"trike60_vecnormalize_{steps}_steps.pkl") is None


def _start_and_wait(cmd, pattern="steps ", new_session=True):
    """Start cmd, read its output until a line starts with ``pattern``; returns (proc, lines so far)."""
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(RL_DIR), start_new_session=new_session)
    lines, deadline = [], time.time() + 300
    for line in proc.stdout:
        lines.append(line)
        if line.startswith(pattern) or time.time() > deadline:
            break
    return proc, lines


# ======================================================================================== environment
def test_reset_step_shapes_and_finiteness(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (sum(w for _, w in env.obs_layout),)
    assert obs.dtype == np.float32 and np.isfinite(obs).all()
    assert env.action_space.shape == (12,)
    assert 0 <= info["delay_steps"] <= 10 and 0.0 <= info["command_speed"] <= 0.27
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, r, term, trunc, inf = env.step(rng.uniform(-1, 1, 12).astype(np.float32))
        assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
        assert math.isfinite(r)
        assert set(REWARD_TERMS) <= set(inf)
        terms = sum(inf[k] for k in REWARD_TERMS)
        assert r == pytest.approx(terms - (env.cfg.termination_penalty if term else 0.0))
        if term or trunc:
            env.reset()


@pytest.mark.filterwarnings("ignore:.*Box observation space m.* value is.*:UserWarning")
def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(TrikeEnv(), skip_render_check=True)


def test_zero_action_stands_for_two_seconds():
    env = TrikeEnv(TrikeEnvConfig(**QUIET))
    env.reset(seed=1)
    for _ in range(int(round(2.0 / env.dt))):
        obs, r, term, trunc, info = env.step(np.zeros(12))
        assert not term, info["termination"]
    assert info["tilt_deg"] < 2.0
    assert abs(info["height"] - env.nominal_height) < 0.02 * env.nominal_height
    assert all(info["foot_contacts"])  # four feet down


@pytest.mark.parametrize("k", [0, 3, 10])
def test_delay_fifo_is_exact(k):
    env = TrikeEnv(TrikeEnvConfig(domain_randomization=False, observation_noise=False, delay_steps_fixed=k,
                                  action_filter_hz=0.0))
    _, info = env.reset(seed=2)
    assert info["delay_steps"] == k == env.delay_steps
    log = []
    env.substep_hook = lambda e: log.append(e.leg_ctrl())
    neutral = env.leg_targets_for(np.zeros(12))
    a = np.linspace(-0.6, 0.6, 12)
    target = env.leg_targets_for(a)
    assert not np.allclose(target, neutral)
    env.step(np.zeros(12))  # physics steps 0 .. 19 at neutral
    env.step(a)  # the new target is set at the start of physics step 20
    env.step(a)
    log = np.array(log)
    fs = env.cfg.frame_skip
    assert len(log) == 3 * fs
    changed = [i for i in range(len(log)) if np.array_equal(log[i], target)]
    assert changed[0] == fs + k  # reaches data.ctrl exactly k physics steps after it was set
    assert all(np.array_equal(log[i], neutral) for i in range(fs + k))
    assert all(np.array_equal(log[i], target) for i in range(fs + k, len(log)))
    other = np.setdiff1d(np.arange(env.model.nu), env._leg_act)  # head and tail held at their neutral target
    assert np.array_equal(env.data.ctrl[other], np.zeros(len(other)))


def test_action_filter_is_first_order():
    env = TrikeEnv(TrikeEnvConfig(**QUIET))  # 4 Hz default
    alpha = 1.0 - math.exp(-2.0 * math.pi * env.cfg.action_filter_hz * env.dt)
    assert env._filter_alpha == pytest.approx(alpha)
    env.reset(seed=0)
    a = np.full(12, 0.5)
    expected = np.zeros(12)
    for _ in range(5):
        obs, *_ = env.step(a)
        expected = expected + alpha * (a - expected)
        assert np.allclose(env._action_filtered, expected)
        assert np.allclose(env.obs_slots(obs)["previous_action_filtered"], expected, atol=1e-6)
    assert np.allclose(env._target, env.leg_targets_for(expected))
    # jitter is cut: the filtered response to +-1 alternating actions stays well inside +-alpha
    env.reset(seed=0)
    for i in range(50):
        env.step(np.full(12, 1.0 if i % 2 else -1.0))
    assert np.abs(env._action_filtered).max() < alpha


def test_delay_is_drawn_per_episode_unless_fixed():
    env = TrikeEnv()
    delays = {env.reset(seed=s)[1]["delay_steps"] for s in range(60)}
    assert delays <= set(range(11)) and len(delays) >= 8
    env10 = TrikeEnv(TrikeEnvConfig(delay_steps_fixed=10))
    assert {env10.reset(seed=s)[1]["delay_steps"] for s in range(10)} == {10}


def test_targets_are_clipped_to_ctrlrange(env):
    lo, hi = env.model.actuator_ctrlrange[env._leg_act].T
    for a in (np.ones(12), -np.ones(12), np.full(12, 5.0)):
        t = env.leg_targets_for(a)
        assert (t >= lo - 1e-12).all() and (t <= hi + 1e-12).all()


def test_randomization_within_ranges_and_restored_when_off():
    cfg = TrikeEnvConfig()
    env = TrikeEnv(cfg)
    m, nom = env.model, mujoco.MjModel.from_xml_path(cfg.model_path)
    floor, soles = env._floor, env._soles
    for seed in range(12):
        env.reset(seed=seed)
        mu = m.geom_friction[floor, 0]
        assert cfg.friction_range[0] <= mu <= cfg.friction_range[1]
        assert np.all(m.geom_friction[soles, 0] == mu)
        others = np.setdiff1d(np.arange(m.ngeom), np.r_[floor, soles])
        assert np.array_equal(m.geom_friction[others], nom.geom_friction[others])
        assert np.array_equal(m.geom_friction[:, 1:], nom.geom_friction[:, 1:])
        s = np.asarray(env.factors["mass_scale"])[1:]  # per body; the world body (0) is never scaled
        assert np.all((s >= 0.9) & (s <= 1.1))
        assert np.array_equal(m.body_mass[1:], nom.body_mass[1:] * s) and m.body_mass[0] == nom.body_mass[0]
        assert np.array_equal(m.body_inertia[1:], nom.body_inertia[1:] * s[:, None])
        kp = m.actuator_gainprm[:, 0] / nom.actuator_gainprm[:, 0]
        assert np.all((kp >= 0.9 - 1e-12) & (kp <= 1.1 + 1e-12))
        assert np.array_equal(m.actuator_biasprm[:, 1], -m.actuator_gainprm[:, 0])  # position servo kept consistent
        assert np.array_equal(m.actuator_biasprm[:, 2], nom.actuator_biasprm[:, 2])  # kv not randomised
        fr = m.actuator_forcerange / nom.actuator_forcerange
        assert np.all((fr >= 0.85 - 1e-12) & (fr <= 1.0 + 1e-12)) and np.allclose(fr[:, 0], fr[:, 1])
        assert np.isclose(m.body_subtreemass[0], m.body_mass.sum())  # mj_setConst ran
    env.cfg.domain_randomization = False
    env.reset(seed=99)
    for name in ("geom_friction", "body_mass", "body_inertia", "actuator_gainprm", "actuator_biasprm",
                 "actuator_forcerange", "body_subtreemass", "dof_invweight0", "body_invweight0", "actuator_acc0"):
        assert np.array_equal(getattr(m, name), getattr(nom, name)), name


def test_same_seed_same_first_50_observations():
    actions = np.random.default_rng(7).uniform(-1, 1, (49, 12))

    def rollout(seed):
        e = TrikeEnv()
        obs, _ = e.reset(seed=seed)
        out = [obs]
        for a in actions:
            obs, *_ = e.step(a)
            out.append(obs)
        return np.array(out)

    a, b, c = rollout(123), rollout(123), rollout(124)
    assert a.shape == (50, TrikeEnv().observation_space.shape[0])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_observation_slots_hold_the_named_quantities():
    """Every slot against the quantity it names (noise off), not only the names."""
    env = TrikeEnv(TrikeEnvConfig(**QUIET))
    env.reset(seed=5, options={"command_speed": 0.2})
    a = np.linspace(-0.4, 0.4, 12)
    for _ in range(2):
        obs, *_ = env.step(a)
    b = -a
    seen_raw = env.obs_slots(obs)["previous_action_raw"].astype(np.float64)
    obs, _, _, _, info = env.step(b)
    # the action-rate penalty depends only on what the policy saw (its previous raw action) and its new action
    assert info["r_action_rate"] == pytest.approx(-env.cfg.w_action_rate * np.sum((b - seen_raw) ** 2), rel=1e-6)
    o, d, m = env.obs_slots(obs), env.data, env.model
    assert sum(w for _, w in env.obs_layout) == obs.size

    def close(x, y, tol=1e-5):
        return np.allclose(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), atol=tol)

    assert close(o["leg_joint_pos"], d.qpos[env._leg_qadr] - env._q_neutral)
    assert close(o["leg_joint_vel"], d.qvel[env._leg_dadr], 1e-4)
    assert close(o["ankle_pitch_pos"], [d.qpos[m.jnt_qposadr[m.joint(f"{c}_ankle").id]] for c in CORNERS])
    quat = d.sensordata[env._quat_adr:env._quat_adr + 4]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat / np.linalg.norm(quat))
    R = R.reshape(3, 3)
    assert close(o["gravity_body"], -R[2, :])
    yaw_err = math.atan2(R[1, 0], R[0, 0]) - env._heading
    assert close(o["heading_error_sin_cos_ESTIMATE"], [math.sin(yaw_err), math.cos(yaw_err)])
    assert close(o["gyro"], d.sensordata[env._gyro_adr:env._gyro_adr + 3], 1e-4)
    Rc = d.xmat[env._chassis].reshape(3, 3)
    assert close(o["base_lin_vel_body_PRIVILEGED"], Rc.T @ d.qvel[0:3], 1e-5)
    loads = [d.sensordata[m.sensor_adr[m.sensor(f"{c}_normal_load").id]] for c in CORNERS]
    assert close(o["foot_load_over_weight"], np.array(loads) / env.nominal_weight)
    assert close(o["previous_action_filtered"], env._action_filtered)
    assert close(o["previous_action_raw"], b)
    assert close(o["command_speed_mps"], [0.2])
    ph = 2 * math.pi * env.cfg.phase_clock_hz * 3 * env.dt
    assert close(o["phase_sin_cos"], [math.sin(ph), math.cos(ph)])


def test_observation_signs_follow_the_body():
    """Forward base speed reads +x; a roll to the left tilts gravity toward body -y; a yaw reads in the heading."""
    env = TrikeEnv(TrikeEnvConfig(**QUIET))
    env.reset(seed=0)
    d = env.data
    d.qvel[0] = 0.2  # world +x = forward at yaw 0
    phi, psi = math.radians(10), math.radians(20)
    q_roll = np.array([math.cos(phi / 2), math.sin(phi / 2), 0, 0])
    q_yaw = np.array([math.cos(psi / 2), 0, 0, math.sin(psi / 2)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_roll)
    d.qpos[3:7] = q
    mujoco.mj_forward(env.model, d)
    o = env.obs_slots(env._get_obs(noise=False))
    assert o["gravity_body"][1] == pytest.approx(-math.sin(phi), abs=1e-5)
    assert o["heading_error_sin_cos_ESTIMATE"][0] == pytest.approx(math.sin(psi), abs=1e-5)
    assert o["base_lin_vel_body_PRIVILEGED"][0] > 0.15


def test_foot_load_noise_and_heading_drift():
    cfg = TrikeEnvConfig(domain_randomization=False, delay_steps_fixed=0)
    env = TrikeEnv(cfg)
    env.reset(seed=3)
    diffs, drifts = [], []
    for _ in range(300):
        obs, *_, info = env.step(np.zeros(12))
        true = env.data.sensordata[env._touch_adr] / env.nominal_weight
        diffs.append(env.obs_slots(obs)["foot_load_over_weight"] - true)
        drifts.append(info["heading_drift"])
    assert np.std(diffs) == pytest.approx(cfg.noise_foot_load, rel=0.15)
    assert drifts[-1] != 0.0 and len(set(drifts)) > 250  # a random walk, one increment per control step
    for off in (cfg.replace(heading_drift_rw=0.0), cfg.replace(observation_noise=False)):
        e = TrikeEnv(off)
        e.reset(seed=3)
        for _ in range(20):
            *_, info = e.step(np.zeros(12))
        assert info["heading_drift"] == 0.0


def _terms(cfg, **kw):
    base = dict(v_fwd=0.0, v_cmd=0.25, v_lat=0.0, yaw_rate=0.0, yaw_err=0.0, grav_b=(0.0, 0.0, -1.0), z=0.3,
                z_nominal=0.3, mean_power=0.0, action=np.zeros(12), prev_action=np.zeros(12), slip=0.0,
                contact=(1, 1, 1, 1), penalty_scale=1.0)
    base.update(kw)
    return reward_terms(cfg, **base)


def test_tracking_terms_shape():
    cfg = TrikeEnvConfig()
    for v_cmd in (0.0, 0.1, 0.25, 0.27):
        peak = _terms(cfg, v_fwd=v_cmd, v_cmd=v_cmd)
        assert peak["r_track"] == pytest.approx(cfg.w_track) and peak["r_track_linear"] == 0.0
        for err in (cfg.track_sigma, -cfg.track_sigma):
            t = _terms(cfg, v_fwd=v_cmd + err, v_cmd=v_cmd)
            assert t["r_track"] == pytest.approx(cfg.w_track / math.e)
            assert t["r_track_linear"] == pytest.approx(-cfg.w_track_linear * cfg.track_sigma)
        errs = np.linspace(0, 0.3, 31)
        track = [(_terms(cfg, v_fwd=v_cmd + e, v_cmd=v_cmd)["r_track"]
                  + _terms(cfg, v_fwd=v_cmd + e, v_cmd=v_cmd)["r_track_linear"]) for e in errs]
        assert all(x > y for x, y in zip(track, track[1:]))  # strictly worse the further from the command
    assert cfg.track_sigma <= 0.07  # tight: about a quarter of the 0-0.27 m/s range at most
    # standing still earns less than moving at the command, by a margin, for every command from 0.05 m/s
    for v_cmd in (0.05, 0.1, 0.2, 0.25, 0.27):
        stand = sum(_terms(cfg, v_fwd=0.0, v_cmd=v_cmd).values())
        move = sum(_terms(cfg, v_fwd=v_cmd, v_cmd=v_cmd).values())
        half = sum(_terms(cfg, v_fwd=v_cmd / 2, v_cmd=v_cmd).values())
        assert move > half > stand and move - stand > 0.5 * cfg.w_track
        assert stand > 0  # but standing still never pays less than falling over at once


def test_reward_term_signs():
    cfg = TrikeEnvConfig(w_diagonal=0.3)
    rng = np.random.default_rng(0)
    for _ in range(500):
        t = _terms(cfg, v_fwd=rng.normal(0, 0.3), v_cmd=rng.uniform(0, 0.27), v_lat=rng.normal(0, 0.2),
                   yaw_rate=rng.normal(0, 1), yaw_err=rng.normal(0, 1), grav_b=rng.normal(0, 1, 3),
                   z=rng.uniform(0.1, 0.4), z_nominal=0.32, mean_power=rng.uniform(0, 500),
                   action=rng.uniform(-1, 1, 12), prev_action=rng.uniform(-1, 1, 12), slip=rng.uniform(0, 1),
                   contact=rng.integers(0, 2, 4), penalty_scale=rng.uniform(0, 1))
        assert set(t) == set(REWARD_TERMS)
        for k, v in t.items():
            if k in REWARD_BONUSES:
                assert v >= 0, k
            elif k == "r_diagonal":
                assert -cfg.w_diagonal <= v <= cfg.w_diagonal
            else:
                assert v <= 0, k
    # the ramp scales the energy and slip penalties only
    full = _terms(cfg, mean_power=100.0, slip=0.2, action=np.ones(12), v_lat=0.1)
    tenth = _terms(cfg, mean_power=100.0, slip=0.2, action=np.ones(12), v_lat=0.1, penalty_scale=0.1)
    for k in REWARD_TERMS:
        expected = 0.1 * full[k] if k in ("r_energy", "r_slip") else full[k]
        assert tenth[k] == pytest.approx(expected), k


def test_diagonal_score():
    # order: left_fore, right_fore, left_hind, right_hind
    assert diagonal_score((1, 0, 0, 1)) == 1.0  # trot
    assert diagonal_score((1, 0, 1, 0)) == 0.0  # pace (left pair together)
    assert diagonal_score((1, 1, 0, 0)) == -1.0  # bound (front pair together)
    assert diagonal_score((1, 1, 1, 1)) == 0.0 and diagonal_score((0, 0, 0, 0)) == 0.0


def test_penalty_scale_reaches_the_reward():
    env = TrikeEnv(TrikeEnvConfig(**QUIET))
    env.reset(seed=0)
    env.set_penalty_scale(0.25)
    *_, info = env.step(np.full(12, 0.5))
    assert info["penalty_scale"] == 0.25
    env.set_penalty_scale(7.0)
    assert env.penalty_scale == 1.0


def _tipped(env, quat, z):
    env.reset(seed=0)
    env.data.qpos[2] = z
    env.data.qpos[3:7] = quat
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    return env.step(np.zeros(12))


def test_termination_on_tilt():
    env = TrikeEnv(TrikeEnvConfig(**QUIET))
    h = math.radians(40.0) / 2
    _, r, term, trunc, info = _tipped(env, [math.cos(h), math.sin(h), 0, 0], 0.45)
    assert term and info["termination"] == "tilt"
    assert r == pytest.approx(sum(info[k] for k in REWARD_TERMS) - env.cfg.termination_penalty)


def test_termination_on_body_contact_and_height():
    upside_down = [0.0, 1.0, 0.0, 0.0]
    env = TrikeEnv(TrikeEnvConfig(**QUIET, max_tilt_deg=181.0, min_height_frac=0.0))
    _, _, term, _, info = _tipped(env, upside_down, 0.08)
    assert term and info["termination"].startswith("floor_contact:")
    assert not info["termination"].endswith("_sole")
    env = TrikeEnv(TrikeEnvConfig(**QUIET, max_tilt_deg=181.0, terminate_on_body_contact=False))
    _, _, term, _, info = _tipped(env, upside_down, 0.08)
    assert term and info["termination"] == "height"


def test_truncation_at_time_limit():
    env = TrikeEnv(TrikeEnvConfig(**QUIET, episode_seconds=0.2))
    env.reset(seed=0)
    for i in range(10):
        _, _, term, trunc, _ = env.step(np.zeros(12))
        assert not term and trunc == (i == 9)


def test_ankle_roll_model_loads():
    from trike_env import ANKLE_ROLL_MODEL

    env = TrikeEnv(TrikeEnvConfig(model_path=str(ANKLE_ROLL_MODEL), **QUIET))
    obs, _ = env.reset(seed=0)
    assert obs.shape == TrikeEnv().observation_space.shape  # the same policy interface on both models
    for _ in range(25):
        _, _, term, _, info = env.step(np.zeros(12))
        assert not term, info["termination"]
    assert len(LEG_ACTUATORS) == 12


# ======================================================================================== train.py
BASE_ARGS = ["--n-envs", 2, "--n-steps", 256, "--batch-size", 512, "--checkpoint-every", 1024, "--log-every", 512,
             "--seed", 3, "--eval-every", 1024, "--eval-seconds", 0.5]


def test_train_stop_resume_settings_and_broken_newest(tmp_path):
    out = tmp_path / "run"
    ckpt = out / "checkpoints"
    base = [RL_DIR / "train.py", "--out", out, *BASE_ARGS]

    # session 1: stopped with SIGINT after the first progress line (512 steps)
    proc, lines = _start_and_wait([sys.executable, "-u", *base, "--total-steps", 3072])
    proc.send_signal(signal.SIGINT)
    rest, _ = proc.communicate(timeout=300)
    text = "".join(lines) + rest
    assert proc.returncode == 130, text
    s0 = _record(out)["sessions"][0]
    assert s0["status"] == "interrupted" and _pair_ok(ckpt, s0["steps_end"])
    assert "training rollouts (stochastic actions)" in text and "Machine: " in text

    # without --resume an existing run is refused and left alone
    p = _run(*base, "--total-steps", 3072)
    assert p.returncode == 2 and "REFUSED" in p.stdout
    # a changed setting is refused; the policy's shape even with --allow-changes
    p = _run(*base, "--total-steps", 4096, "--resume")
    assert p.returncode == 2 and "total_steps: run 3072, now 4096" in p.stdout
    p = _run(*base, "--total-steps", 3072, "--resume", "--net", "64,64", "--allow-changes")
    assert p.returncode == 2 and "policy's shape" in p.stdout
    assert len(_record(out)["sessions"]) == 1

    # --resume continues from the newest pair with the step count carried over, to the same total
    p = _run(*base, "--total-steps", 3072, "--resume")
    assert p.returncode == 0, p.stdout + p.stderr
    n1 = s0["steps_end"]
    assert f"from trike60_{n1}_steps.zip ({n1} of 3072 steps)" in p.stdout
    for n in (1024, 2048, 3072):
        assert f"Saved checkpoint pair at {n} steps: trike60_{n}_steps.zip and trike60_vecnormalize_{n}_steps.pkl" \
            in p.stdout
        assert _pair_ok(ckpt, n)
    end = n1 + math.ceil((3072 - n1) / 512) * 512  # rollouts of 512 steps from wherever session 1 stopped
    assert f"Session completed: {end} steps (target 3072)" in p.stdout and _pair_ok(ckpt, end)
    assert "eval at 1024 steps (deterministic, delay 10 ms)" in p.stdout
    rec = _record(out)
    s1 = rec["sessions"][1]
    assert s1["started_from"] == f"trike60_{n1}_steps.zip" and s1["steps_start"] == n1 and s1["steps_end"] == end
    for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef"):
        assert s1["ppo_in_use"][k] == pytest.approx(rec["settings"][k]), k  # recorded = really used
    assert s1["ppo_in_use"]["learning_rate"] == "LinearSchedule(0.0003, 3e-05)"
    # every periodic save is in the record as it happens, and the end
    assert [x for x in s1["saves"] if x.startswith("trike60_")][:3] == [
        "trike60_1024_steps.zip", "trike60_2048_steps.zip", "trike60_3072_steps.zip"]
    assert s1["steps_reached"] == s1["steps_end"] == end and s1["saves"][-1] == f"trike60_{end}_steps.zip"
    evals = [json.loads(x) for x in (out / "eval_during_training.jsonl").read_text().splitlines()]
    assert [e["steps"] for e in evals] == [1024, 2048, 3072]
    assert all(e["deterministic"] and e["delay_steps"] == 10 and len(e["episodes"]) == 3 for e in evals)
    assert not list(out.rglob("*.tmp"))

    # a finished run resumes to nothing
    p = _run(*base, "--total-steps", 3072, "--resume")
    assert p.returncode == 0 and "Run already complete" in p.stdout

    # --allow-changes runs with the changed total and logs it
    newest = ckpt / f"trike60_{end}_steps.zip"
    newest.write_bytes(newest.read_bytes()[:2000])  # and a truncated newest pair is skipped for the one before
    fallback = 3072 if end != 3072 else 2048
    p = _run(*base, "--total-steps", 4096, "--resume", "--allow-changes")
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"skipping checkpoint trike60_{end}_steps.zip" in p.stdout
    assert f"from trike60_{fallback}_steps.zip" in p.stdout
    assert "learning rate follows steps / total_steps" in p.stdout
    s2 = _record(out)["sessions"][2]
    assert s2["changed_settings"]["total_steps"] == {"run": 3072, "now": 4096} and _pair_ok(ckpt, 4096)


@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM"])
def test_train_saves_a_pair_when_stopped(tmp_path, sig):
    out = tmp_path / "run"
    proc, _ = _start_and_wait([sys.executable, "-u", RL_DIR / "train.py", "--out", out, "--n-envs", 2,
                               "--n-steps", 256, "--total-steps", 10_000_000, "--log-every", 512])
    proc.send_signal(getattr(signal, sig))
    proc.send_signal(getattr(signal, sig))  # a repeated signal must not cut the save short
    text, _ = proc.communicate(timeout=300)
    assert proc.returncode == 130, text
    s = _record(out)["sessions"][0]
    assert s["status"] == "interrupted" and s["steps_end"] > 0
    assert _pair_ok(out / "checkpoints", s["steps_end"])
    assert not list(out.rglob("*.tmp"))


def test_notebook_stop_button_reaches_train_once(tmp_path):
    """A Jupyter interrupt signals the kernel's whole process group (os.killpg). The notebook's run_step starts
    train.py in its own session and forwards one SIGINT: the pair is saved intact and the session is recorded."""
    out = tmp_path / "run"
    kernel = tmp_path / "kernel_sim.py"
    kernel.write_text(f"""
import json, signal, subprocess, sys
cell2 = "".join(json.load(open({str(NOTEBOOK)!r}))["cells"][2]["source"])
ns = {{"subprocess": subprocess, "signal": signal}}
exec(cell2[cell2.index("def run_step"):cell2.index("def git(")], ns)
cmd = [sys.executable, "-u", {str(RL_DIR / 'train.py')!r}, "--out", {str(out)!r}, "--n-envs", "2", "--n-steps", "256",
       "--total-steps", "10000000", "--log-every", "512", "--exit-with-parent"]
try:
    print("KERNEL: run_step returned", ns["run_step"](cmd), flush=True)
except KeyboardInterrupt:
    print("KERNEL: KeyboardInterrupt re-raised as in the notebook", flush=True)
""")
    for trial in range(2):
        proc, _ = _start_and_wait([sys.executable, "-u", kernel])
        os.killpg(proc.pid, signal.SIGINT)  # what a Jupyter interrupt does to the kernel's process group
        text, _ = proc.communicate(timeout=300)
        assert "KERNEL: KeyboardInterrupt re-raised" in text and "Session interrupted" in text, text
        s = _record(out)["sessions"][-1]
        assert s["status"] == "interrupted" and _pair_ok(out / "checkpoints", s["steps_end"])
        assert not list(out.rglob("*.tmp"))
        shutil.rmtree(out)


def test_train_exits_and_saves_when_its_parent_dies(tmp_path):
    """--exit-with-parent: a killed notebook kernel must not leave train.py running as an orphan."""
    out = tmp_path / "run"
    parent = tmp_path / "parent.py"
    parent.write_text(f"""
import subprocess, sys
p = subprocess.Popen([sys.executable, "-u", {str(RL_DIR / 'train.py')!r}, "--out", {str(out)!r}, "--n-envs", "2",
    "--n-steps", "256", "--total-steps", "10000000", "--log-every", "512", "--exit-with-parent"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
print("CHILD", p.pid, flush=True)
for line in p.stdout:
    print(line, end="", flush=True)
""")
    proc, lines = _start_and_wait([sys.executable, "-u", parent])
    child = int(next(x for x in lines if x.startswith("CHILD")).split()[1])
    proc.kill()
    proc.wait()
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
    s = _record(out)["sessions"][0]
    assert s["status"] == "interrupted" and _pair_ok(out / "checkpoints", s["steps_end"])


def test_broken_only_checkpoints_are_moved_aside_and_the_run_restarts(tmp_path):
    out = tmp_path / "run"
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "trike60_1024_steps.zip").write_bytes(b"")  # what a killed save used to leave
    (ckpt / "trike60_2048_steps.zip").write_bytes(b"PK\x03\x04 truncated")
    (ckpt / "trike60_vecnormalize_2048_steps.pkl").write_bytes(b"\x80")
    (out / "run_record.json").write_text('{"sessions": [{"status": ')  # truncated record
    base = [RL_DIR / "train.py", "--out", out, *BASE_ARGS, "--eval-every", 0, "--total-steps", 1024]
    p = _run(*base)
    assert p.returncode == 2 and "REFUSED" in p.stdout  # without --resume nothing is touched
    assert (ckpt / "trike60_2048_steps.zip").exists()
    p = _run(*base, "--resume")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "no intact checkpoint pair; moved 3 broken file(s)" in p.stdout and "(0 of 1024 steps)" in p.stdout
    broken = list(ckpt.glob("broken_*"))
    assert len(broken) == 1 and sorted(x.name for x in broken[0].iterdir()) == [
        "trike60_1024_steps.zip", "trike60_2048_steps.zip", "trike60_vecnormalize_2048_steps.pkl"]
    rec = _record(out)
    assert rec["quarantined"][0]["files"] and rec["sessions"][-1]["quarantined"]
    assert list(out.glob("run_record.broken_*.json"))  # the unreadable record was moved aside, not a dead end
    assert _pair_ok(ckpt, 1024)


# ======================================================================================== evaluate.py
@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("trained") / "run"
    p = _run(RL_DIR / "train.py", "--out", out, *BASE_ARGS, "--eval-every", 0, "--total-steps", 1024,
             "--model", "trike60_C_closed_ankleroll")
    assert p.returncode == 0, p.stdout + p.stderr
    return out


def test_evaluate_sweep_and_the_runs_own_model(trained_run):
    p = _run(RL_DIR / "evaluate.py", "--run-dir", trained_run, "--episodes", 1, "--delays", 10, 0, "--sweep",
             "--sweep-episodes", 1, "--seconds", 0.5)
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads((trained_run / "eval" / "eval_1024.json").read_text())
    assert rep["model"] == "trike60_C_closed_ankleroll.xml"  # from run_record.json, not the default
    assert list(rep["results"]) == ["delay_10ms", "delay_0ms", "sweep_delay_10ms"]
    assert list(rep["results"]["sweep_delay_10ms"]) == ["0.00", "0.10", "0.20", "0.25", "0.27"]
    for c, s in rep["results"]["sweep_delay_10ms"].items():
        assert all(e["delay_steps"] == 10 and e["command_mps"] == float(c) for e in s["per_episode"])
    assert rep["target"]["speed_mps"] == 0.25 and rep["target"]["summary"]["mean_command_mps"] == 0.25
    assert "TARGET 0.25 m/s at 10 ms" in p.stdout
    assert rep["env_config"]["domain_randomization"] is True and rep["deterministic"] is True


def test_evaluate_uses_vecnormalize_in_eval_mode(trained_run):
    import evaluate

    from environments.shared.curriculum.checkpoints import newest_intact_periodic_pair

    (_, zip_path, vn_path), _ = newest_intact_periodic_pair(trained_run / "checkpoints", "trike60")
    env = TrikeEnv(TrikeEnvConfig(model_path=str(RL_DIR.parent / "sim" / "generated" /
                                                 "trike60_C_closed_ankleroll.xml")))
    policy = evaluate.make_policy(zip_path, vn_path, env, deterministic=True)
    vn = policy.vecnormalize
    assert vn.training is False and vn.norm_reward is False
    mean0, var0, count0 = vn.obs_rms.mean.copy(), vn.obs_rms.var.copy(), vn.obs_rms.count
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        a = policy(obs)
        assert a.shape == (12,) and np.all(np.abs(a) <= 1.0)
        obs, *_ = env.step(a)
    assert np.array_equal(vn.obs_rms.mean, mean0) and np.array_equal(vn.obs_rms.var, var0)
    assert vn.obs_rms.count == count0  # statistics read, never updated
    expected = np.clip((obs - mean0) / np.sqrt(var0 + vn.epsilon), -vn.clip_obs, vn.clip_obs)
    assert np.allclose(vn.normalize_obs(obs[None, :])[0], expected, atol=1e-5)


def test_evaluate_refuses_without_a_model(trained_run, tmp_path):
    lone = tmp_path / "lone"
    lone.mkdir()
    for f in ("trike60_1024_steps.zip", "trike60_vecnormalize_1024_steps.pkl"):
        shutil.copy(trained_run / "checkpoints" / f, lone / f)
    p = _run(RL_DIR / "evaluate.py", "--checkpoint", lone / "trike60_1024_steps.zip", "--episodes", 1,
             "--seconds", 0.2)
    assert p.returncode != 0 and "REFUSED" in (p.stdout + p.stderr)
    p = _run(RL_DIR / "evaluate.py", "--checkpoint", lone / "trike60_1024_steps.zip", "--episodes", 1,
             "--seconds", 0.2, "--model", "trike60_C_closed_ankleroll")
    assert p.returncode == 0, p.stdout + p.stderr


# ======================================================================================== notebook
def _cells():
    nb = json.loads(NOTEBOOK.read_text())
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_cells_run_locally(tmp_path, monkeypatch, capsys):
    """Cells 1-7 outside Colab (fetch, pip, Drive and disconnect are skipped): a tiny run trains, re-running the
    training cell finds it complete, and the evaluation (with its sweep) writes its JSON."""
    nb, code = _cells()
    assert len(code) == 7 and len(nb["cells"]) <= 8
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RL_DIR, text=True).strip()
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"REPO_REF": head, "RUN_ID": "nb-test", "TOTAL_STEPS": 2048, "N_ENVS": 2,
                                  "EVAL_EPISODES": 1, "MODEL": "trike60_C_closed",
                                  "LOCAL_RUNS_DIR": str(tmp_path / "runs")}))
    monkeypatch.setenv("TRIKE60_PARAMS", str(params))
    monkeypatch.chdir(RL_DIR)
    ns: dict = {}
    for i, src in enumerate(code, start=1):
        exec(compile(src, f"cell{i}", "exec"), ns)
    run_dir = tmp_path / "runs" / "nb-test"
    assert (run_dir / "checkpoints" / "trike60_2048_steps.zip").exists()
    rep = json.loads((run_dir / "eval" / "eval_2048.json").read_text())
    assert "sweep_delay_10ms" in rep["results"] and rep["model"] == "trike60_C_closed.xml"
    assert _record(run_dir)["settings"]["model"] == "trike60_C_closed.xml"
    out = capsys.readouterr().out
    assert "(new run)" in out and "Auto-disconnect skipped" in out and "--resume --exit-with-parent" in out
    assert "CPython 3.12-3.14 needed): yes" in out
    exec(compile(code[3], "cell4", "exec"), ns)  # the run folder is found again ...
    exec(compile(code[4], "cell5", "exec"), ns)  # ... and the training cell finds the run complete
    out = capsys.readouterr().out
    assert "resuming: " in out and "Run already complete" in out


def test_parameters_cell_refuses_bad_values(tmp_path, monkeypatch):
    _, code = _cells()
    for bad, message in (({"REPO_REF": "main", "RUN_ID": "x"}, "40 hex"),
                         ({"REPO_REF": "a" * 40, "RUN_ID": "x", "MODEL": "trike75"}, "MODEL")):
        params = tmp_path / "params.json"
        params.write_text(json.dumps(bad))
        monkeypatch.setenv("TRIKE60_PARAMS", str(params))
        with pytest.raises(ValueError, match=message):
            exec(compile(code[0], "cell1", "exec"), {})


def test_python_version_check_in_cell3():
    _, code = _cells()
    check = code[2][:code[2].index('print(f"Python')]

    class FakeSys:
        version_info = (3, 11, 9)

    with pytest.raises(RuntimeError, match="CPython 3.12, 3.13 or 3.14"):
        exec(compile(check, "cell3", "exec"), {"sys": FakeSys})
    FakeSys.version_info = (3, 13, 1)
    exec(compile(check, "cell3", "exec"), {"sys": FakeSys})


def test_cell2_fetches_exactly_one_commit_shallowly(tmp_path):
    """The Colab checkout function against a local bare copy of this repository (a stand-in for GitHub, which
    serves a commit by its hash): the result is that commit, clean, with no history."""
    _, code = _cells()
    src = code[1]
    ns = {"subprocess": subprocess, "signal": signal, "pathlib": __import__("pathlib"), "REPO_DIR": None}
    exec(src[src.index("def run_step"):src.index("if IN_COLAB:\n    REPO_DIR")], ns)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "--bare", "--quiet", str(REPO_ROOT), str(bare)], check=True)
    subprocess.run(["git", "-C", str(bare), "config", "uploadpack.allowAnySHA1InWant", "true"], check=True)
    target = tmp_path / "checkout"
    got, clean = ns["checkout_commit"](target, f"file://{bare}", head)
    assert got == head and clean
    assert (target / ".git" / "shallow").read_text().split() == [head]  # one commit, no history
    assert (target / "hardware" / "triceratops_build" / "rl" / "train.py").exists()
    got, clean = ns["checkout_commit"](target, f"file://{bare}", head)  # running the cell again is harmless
    assert got == head and clean


# ======================================================================================== recheck items (N1-N8)
def _copy_run(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _eio_zipfile(monkeypatch, fail_times: int | None):
    """zipfile.ZipFile raising EIO (a Drive read error) ``fail_times`` times, or always when None."""
    import errno
    import zipfile

    real = zipfile.ZipFile
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if fail_times is None or calls["n"] <= fail_times:
            raise OSError(errno.EIO, "Input/output error (simulated)")
        return real(*a, **kw)

    monkeypatch.setattr(zipfile, "ZipFile", flaky)
    return calls


def test_io_error_is_not_a_broken_checkpoint(trained_run, tmp_path, monkeypatch):
    import trike_checkpoints as tc

    ckpt = trained_run / "checkpoints"
    with monkeypatch.context() as mp:
        _eio_zipfile(mp, None)
        with pytest.raises(OSError):
            tc.pair_problem(ckpt / "trike60_1024_steps.zip", ckpt / "trike60_vecnormalize_1024_steps.pkl")
        scan = tc.scan_pairs(ckpt, retries=2, wait_s=0, log=lambda s: None)
        assert scan.io_error and scan.pair is None and not scan.skipped
    with monkeypatch.context() as mp:  # a transient error: the re-read finds the pair
        calls = _eio_zipfile(mp, 1)
        scan = tc.scan_pairs(ckpt, retries=2, wait_s=0, log=lambda s: None)
        assert scan.pair is not None and scan.pair[0] == 1024 and calls["n"] == 2

    # train.py on an intact run whose files cannot be read: refuses, moves and writes nothing
    import train

    run = _copy_run(trained_run, tmp_path / "run")
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with monkeypatch.context() as mp:
        _eio_zipfile(mp, None)
        code = train.main(["--out", str(run), *map(str, BASE_ARGS), "--eval-every", "0", "--total-steps", "2048",
                           "--model", "trike60_C_closed_ankleroll", "--resume", "--io-retry-wait", "0"])
    assert code == 2
    after = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert after == before and not list((run / "checkpoints").glob("broken_*"))


def test_saves_recorded_but_not_listed_are_refused(tmp_path):
    import train

    out = tmp_path / "run"
    out.mkdir()
    (out / "run_record.json").write_text(json.dumps({"sessions": [{"last_checkpoint": "trike60_1024_steps.zip"}]}))
    code = train.main(["--out", str(out), "--total-steps", "2048", "--resume"])
    assert code == 2 and not (out / "checkpoints").exists()


def test_refusals_come_before_any_change(trained_run, tmp_path):
    run = _copy_run(trained_run, tmp_path / "run")
    ckpt = run / "checkpoints"
    (ckpt / "trike60_2048_steps.zip.tmp").write_bytes(b"half a save")
    p = _run(RL_DIR / "train.py", "--out", run, *BASE_ARGS, "--eval-every", 0, "--total-steps", 4096,
             "--model", "trike60_C_closed_ankleroll", "--resume")  # total_steps differs: refused
    assert p.returncode == 2 and "Nothing was changed" in p.stdout
    assert (ckpt / "trike60_2048_steps.zip.tmp").exists()
    # only broken saves, and a refused resume: they stay where they are
    for f in ckpt.glob("trike60_*"):
        f.write_bytes(b"")
    p = _run(RL_DIR / "train.py", "--out", run, *BASE_ARGS, "--eval-every", 0, "--total-steps", 4096,
             "--model", "trike60_C_closed_ankleroll", "--resume")
    assert p.returncode == 2 and not list(ckpt.glob("broken_*"))
    assert (ckpt / "trike60_1024_steps.zip").exists()


def test_nonfinite_parameters_are_never_a_resume_candidate(trained_run, tmp_path):
    import torch
    import trike_checkpoints as tc
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.policy_loading import load_sb3_model

    src = trained_run / "checkpoints"
    model = load_sb3_model(src / "trike60_1024_steps.zip", algorithm="ppo", device="cpu")
    vn = VecNormalize.load(str(src / "trike60_vecnormalize_1024_steps.pkl"),
                           DummyVecEnv([lambda: TrikeEnv(TrikeEnvConfig(model_path=str(
                               RL_DIR.parent / "sim" / "generated" / "trike60_C_closed_ankleroll.xml")))]))
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    name, finite = tc.save_pair(model, vn, ckpt)
    assert finite and name == "trike60_1024_steps.zip"
    with torch.no_grad():
        next(model.policy.parameters()).fill_(float("nan"))
    model.num_timesteps = 2048
    name, finite = tc.save_pair(model, vn, ckpt)
    assert not finite and name == "nonfinite/trike60_2048_steps.zip" and (ckpt / name).exists()
    scan = tc.scan_pairs(ckpt, retries=0, wait_s=0)
    assert scan.pair[0] == 1024 and scan.candidates == ["trike60_1024_steps.zip"]


def _kernel_running_cell5(tmp_path: Path, fake_train: str) -> Path:
    """A stand-in kernel that runs the notebook's own cell 5 (with cell 2's run_step) on a fake train.py; its
    flush_drive() reports whether the fake had finished saving when the flush began."""
    fake_dir = tmp_path / "fake_rl"
    fake_dir.mkdir()
    (fake_dir / "train.py").write_text(fake_train)
    kernel = tmp_path / "kernel.py"
    kernel.write_text(f"""
import json, os, pathlib, signal, subprocess, sys
code = ["".join(c["source"]) for c in json.load(open({str(NOTEBOOK)!r}))["cells"] if c["cell_type"] == "code"]
ns = {{"subprocess": subprocess, "signal": signal, "os": os, "sys": sys, "pathlib": pathlib}}
exec(code[1][code[1].index("def run_step"):code[1].index("def git(")], ns)
MARK = pathlib.Path({str(tmp_path / 'saved')!r})
def flush_drive():
    print(f"FLUSH: the step had finished saving: {{MARK.exists()}}", flush=True)
ns.update(ensure_drive=lambda: None, flush_drive=flush_drive, RL_DIR=pathlib.Path({str(fake_dir)!r}), RUN_DIR="run",
          TOTAL_STEPS=1, N_ENVS=1, SEED=0, MODEL="m", REPO_DIR=".", STEP_ENV=dict(os.environ))
try:
    exec(code[4], ns)
    print("KERNEL: cell 5 returned", flush=True)
except KeyboardInterrupt:
    print("KERNEL: cell 5 raised KeyboardInterrupt", flush=True)
except RuntimeError as exc:
    print(f"KERNEL: cell 5 raised RuntimeError: {{exc}}", flush=True)
""")
    return kernel


def test_cell5_waits_for_the_save_through_a_second_stop_click(tmp_path):
    mark = tmp_path / "saved"
    kernel = _kernel_running_cell5(tmp_path, f"""
import pathlib, signal, sys, time
def stop(signum, frame):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("Interrupted: saving (slowly)", flush=True)
    time.sleep(2.0)
    pathlib.Path({str(mark)!r}).write_text("saved")
    print("Session interrupted", flush=True)
    sys.exit(130)
signal.signal(signal.SIGINT, stop)
print("steps 1", flush=True)
while True:
    time.sleep(0.05)
""")
    proc, lines = _start_and_wait([sys.executable, "-u", kernel])
    os.killpg(proc.pid, signal.SIGINT)  # the stop button
    time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGINT)  # clicked again while the save runs
    text = "".join(lines) + proc.communicate(timeout=60)[0]
    assert "this click is ignored" in text
    assert text.index("Session interrupted") < text.index("FLUSH: the step had finished saving: True") < text.index(
        "KERNEL: cell 5 raised KeyboardInterrupt"), text


def test_cell5_flushes_drive_after_a_failed_step(tmp_path):
    mark = tmp_path / "saved"
    kernel = _kernel_running_cell5(tmp_path, f"""
import pathlib, sys
print("steps 1", flush=True)
pathlib.Path({str(mark)!r}).write_text("saved")
sys.exit(3)
""")
    p = subprocess.run([sys.executable, "-u", str(kernel)], capture_output=True, text=True, timeout=60)
    assert p.stdout.index("FLUSH: the step had finished saving: True") < p.stdout.index(
        "KERNEL: cell 5 raised RuntimeError: train.py stopped with exit code 3"), p.stdout


def test_exit_with_parent_refuses_when_the_parent_is_already_gone(tmp_path):
    p = _run(RL_DIR / "train.py", "--out", tmp_path / "run", "--total-steps", 1024, "--exit-with-parent", 999_999)
    assert p.returncode == 3 and "REFUSED: the parent process 999999 is gone" in p.stdout
    assert not (tmp_path / "run").exists()


def test_evaluate_with_an_unreadable_run_record(trained_run, tmp_path):
    run = _copy_run(trained_run, tmp_path / "run")
    (run / "run_record.json").write_text('{"settings": ')
    p = _run(RL_DIR / "evaluate.py", "--run-dir", run, "--episodes", 1, "--seconds", 0.2)
    assert p.returncode != 0 and "REFUSED" in p.stdout + p.stderr and "could not be read" in p.stdout
    p = _run(RL_DIR / "evaluate.py", "--run-dir", run, "--episodes", 1, "--seconds", 0.2,
             "--model", "trike60_C_closed_ankleroll")
    assert p.returncode == 0, p.stdout + p.stderr


def test_allowed_gamma_change_reaches_vecnormalize(trained_run, tmp_path):
    import pickle

    run = _copy_run(trained_run, tmp_path / "run")
    p = _run(RL_DIR / "train.py", "--out", run, *BASE_ARGS, "--eval-every", 0, "--total-steps", 1024,
             "--model", "trike60_C_closed_ankleroll", "--resume", "--allow-changes", "--gamma", 0.98)
    assert p.returncode == 0 and "Run already complete" in p.stdout  # nothing to train, nothing changed
    p = _run(RL_DIR / "train.py", "--out", run, *BASE_ARGS, "--eval-every", 0, "--total-steps", 1536,
             "--model", "trike60_C_closed_ankleroll", "--resume", "--allow-changes", "--gamma", 0.98)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "VecNormalize's gamma set to 0.98" in p.stdout
    s = _record(run)["sessions"][-1]
    assert s["vecnormalize_gamma"] == 0.98 and s["ppo_in_use"]["gamma"] == 0.98
    with open(run / "checkpoints" / f"trike60_vecnormalize_{s['steps_end']}_steps.pkl", "rb") as f:
        assert pickle.load(f).gamma == 0.98


# ======================================================================================== recheck 2 (R1)
def _killed_run(tmp_path: Path, until: str) -> Path:
    """A run whose only session is SIGKILLed (no stop, as a lost runtime) right after a line starting with ``until``."""
    out = tmp_path / "run"
    proc, _ = _start_and_wait([sys.executable, "-u", RL_DIR / "train.py", "--out", out, "--n-envs", 2,
                               "--n-steps", 256, "--total-steps", 10_000_000, "--checkpoint-every", 1024,
                               "--log-every", 512, "--eval-every", 0, "--resume"], pattern=until)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    return out


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_killed_sessions_saves_are_never_overwritten(tmp_path, monkeypatch):
    import errno

    import train
    import trike_checkpoints as tc

    out = _killed_run(tmp_path, "Saved checkpoint pair at 1024")
    ckpt = out / "checkpoints"
    s = _record(out)["sessions"][0]
    assert s["status"] == "running" and s["saves"] == ["trike60_1024_steps.zip"] and s["steps_reached"] == 1024
    sha = _sha(ckpt / "trike60_1024_steps.zip")
    args = ["--out", str(out), "--n-envs", "2", "--n-steps", "256", "--total-steps", "10000000",
            "--checkpoint-every", "1024", "--log-every", "512", "--eval-every", "0", "--resume", "--io-retry-wait", "0"]

    # the reviewer's case: isdir() answers False for the folder (a stat error on a mount). The scan does not ask
    # isdir any more, finds the pair and resumes from it.
    wrapper = tmp_path / "isdir_false.py"
    wrapper.write_text(f"""
import os, sys
sys.path[:0] = [{str(RL_DIR)!r}, {str(REPO_ROOT)!r}]
import train
if __name__ == "__main__":  # SubprocVecEnv workers import this file again: they must not run it
    ck = {str(ckpt)!r}
    real = os.path.isdir
    os.path.isdir = lambda p: False if os.path.abspath(str(p)) == ck else real(p)
    sys.exit(train.main(sys.argv[1:]))
""")
    proc, lines = _start_and_wait([sys.executable, "-u", wrapper, *args], pattern="Resuming")
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    assert lines[-1].startswith(f"Resuming {out} from trike60_1024_steps.zip (1024 of 10000000 steps)"), lines
    assert _sha(ckpt / "trike60_1024_steps.zip") == sha

    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    # stat() of the folder fails with EIO: re-read, then refused, nothing changed
    real_stat = tc.os.stat

    def stat_eio(path, *a, **kw):
        if os.fspath(path) == str(ckpt):
            raise OSError(errno.EIO, "Input/output error (simulated)", os.fspath(path))
        return real_stat(path, *a, **kw)

    with monkeypatch.context() as mp:
        mp.setattr(tc.os, "stat", stat_eio)
        assert train.main(args) == 2

    # the folder is not there at all (a storage that answers "no folder"): the record's saves forbid step 0
    hidden = out / "checkpoints_hidden"
    ckpt.rename(hidden)
    assert train.main(args) == 2 and not ckpt.exists()
    hidden.rename(ckpt)
    after = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert after == before


def test_a_session_killed_before_its_first_save_may_start_again(tmp_path):
    out = _killed_run(tmp_path, "steps 512")
    s = _record(out)["sessions"][0]
    assert s["saves"] == [] and s["steps_reached"] == 512
    p, lines = _start_and_wait([sys.executable, "-u", RL_DIR / "train.py", "--out", out, "--n-envs", 2,
                                "--n-steps", 256, "--total-steps", 10_000_000, "--checkpoint-every", 1024,
                                "--log-every", 512, "--eval-every", 0, "--resume"],
                               pattern=("Starting", "REFUSED", "Resuming"))
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    assert lines[-1].startswith("Starting a new run"), lines


def test_unaccounted_saves_rules(tmp_path):
    from train import unaccounted_saves

    ckpt = tmp_path / "checkpoints"
    ended = {"saves": ["trike60_1024_steps.zip"], "last_checkpoint": "trike60_1024_steps.zip",
             "ended_utc": "x", "steps_end": 1024}
    assert "not there: trike60_1024_steps.zip" in unaccounted_saves({"sessions": [ended]}, ckpt, 250_000)
    early = {"saves": [], "steps_reached": 800}
    assert unaccounted_saves({"sessions": [early]}, ckpt, 1024) is None  # killed before its first save
    far = {"saves": [], "steps_reached": 300_000}  # its save may not have been recorded: still refused
    assert "reached 300000 steps" in unaccounted_saves({"sessions": [far]}, ckpt, 250_000)
    legacy = {"status": "running", "last_checkpoint": None}  # older format, killed: unknown
    assert "older record format" in unaccounted_saves({"sessions": [legacy]}, ckpt, 250_000)
    # after a quarantine only the sessions from the quarantining one on count; their files went to broken_*
    q = {"quarantined": {"files": ["trike60_1024_steps.zip"]}, "saves": [], "steps_reached": 300}
    assert unaccounted_saves({"sessions": [ended, q]}, ckpt, 1024) is None
    (ckpt / "broken_x").mkdir(parents=True)
    (ckpt / "broken_x" / "trike60_1024_steps.zip").write_bytes(b"")
    assert unaccounted_saves({"sessions": [ended]}, ckpt, 250_000) is None  # the file is kept, set aside
