"""Tests for the 60 cm Triceratops RL setup: the environment, train.py and evaluate.py. Simulation only."""

import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from trike_env import CORNERS, LEG_ACTUATORS, TrikeEnv, TrikeEnvConfig

RL_DIR = Path(__file__).resolve().parents[1]
QUIET = dict(domain_randomization=False, observation_noise=False, delay_steps_fixed=0)
REWARD_TERMS = ("r_track", "r_alive", "r_upright", "r_height", "r_lateral", "r_yaw_rate", "r_heading", "r_energy",
                "r_action_rate", "r_slip", "r_diagonal")


@pytest.fixture(scope="module")
def env():
    e = TrikeEnv()
    yield e
    e.close()


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
    env = TrikeEnv(TrikeEnvConfig(domain_randomization=False, observation_noise=False, delay_steps_fixed=k))
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
    # head and tail servos are held at their neutral target
    other = np.setdiff1d(np.arange(env.model.nu), env._leg_act)
    assert np.array_equal(env.data.ctrl[other], np.zeros(len(other)))


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


def _fresh_model(cfg):
    return mujoco.MjModel.from_xml_path(cfg.model_path)


def test_randomization_within_ranges_and_restored_when_off():
    cfg = TrikeEnvConfig()
    env = TrikeEnv(cfg)
    m, nom = env.model, _fresh_model(cfg)
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
        total = m.body_mass.sum() / nom.body_mass.sum()
        assert 0.9 <= total <= 1.1
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
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1, 1, (49, 12))

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


def test_layout_names_cover_the_task():
    names = [n for n, _ in TrikeEnv().obs_layout]
    for n in ("leg_joint_pos", "leg_joint_vel", "gravity_body", "gyro", "base_lin_vel_body_PRIVILEGED",
              "foot_load_over_weight", "previous_action", "command_speed_mps", "phase_sin_cos"):
        assert n in names
    assert len(LEG_ACTUATORS) == 12 and len(CORNERS) == 4


def _run(*args, timeout=600):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, timeout=timeout,
                          cwd=str(RL_DIR))


def test_train_smoke_resume_and_evaluate(tmp_path):
    out = tmp_path / "run"
    ckpt = out / "checkpoints"
    base = [RL_DIR / "train.py", "--out", out, "--n-envs", 2, "--n-steps", 512, "--batch-size", 512,
            "--checkpoint-every", 1024, "--seed", 3, "--verbose", 0]
    p = _run(*base, "--total-steps", 2048)
    assert p.returncode == 0, p.stdout + p.stderr
    for s in (1024, 2048):
        assert (ckpt / f"trike60_{s}_steps.zip").exists()
        assert (ckpt / f"trike60_vecnormalize_{s}_steps.pkl").exists()
    rec = json.loads((out / "run_record.json").read_text())
    assert rec["sessions"][0]["steps_end"] == 2048 and rec["sessions"][0]["status"] == "completed"
    assert rec["sessions"][0]["versions"]["mujoco"] == "3.10.0"

    # without --resume an existing run is refused and left alone
    p = _run(*base, "--total-steps", 4096)
    assert p.returncode == 2 and "REFUSED" in p.stdout
    assert not (ckpt / "trike60_3072_steps.zip").exists()

    # --resume continues from the newest pair with the step count carried over
    p = _run(*base, "--total-steps", 3072, "--resume")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "from trike60_2048_steps.zip" in p.stdout
    rec = json.loads((out / "run_record.json").read_text())
    s1 = rec["sessions"][1]
    assert s1["resumed_from"] == "trike60_2048_steps.zip" and s1["steps_start"] == 2048 and s1["steps_end"] == 3072
    assert (ckpt / "trike60_3072_steps.zip").exists()

    # a truncated newest checkpoint (a runtime killed mid-write) is skipped for the one before it
    newest = ckpt / "trike60_3072_steps.zip"
    newest.write_bytes(newest.read_bytes()[:2000])
    p = _run(*base, "--total-steps", 4096, "--resume")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "skipping checkpoint trike60_3072_steps.zip" in p.stdout and "from trike60_2048_steps.zip" in p.stdout
    assert (ckpt / "trike60_4096_steps.zip").exists()

    # a finished run resumes to nothing
    p = _run(*base, "--total-steps", 4096, "--resume")
    assert p.returncode == 0 and "already complete" in p.stdout

    # evaluation at the fixed 10 ms delay (and 0 ms) writes its JSON
    p = _run(RL_DIR / "evaluate.py", "--run-dir", out, "--episodes", 1, "--delays", 10, 0, "--command", 0.1)
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads((out / "eval" / "eval_4096.json").read_text())
    assert list(rep["results"]) == ["delay_10ms", "delay_0ms"]
    r10 = rep["results"]["delay_10ms"]
    assert r10["per_episode"][0]["delay_steps"] == 10 and rep["env_config"]["domain_randomization"] is True
    for key in ("mean_forward_speed_mps", "mean_abs_speed_error_mps", "falls", "mean_episode_length_s"):
        assert key in r10


def test_notebook_cells_run_locally(tmp_path, monkeypatch, capsys):
    """Cells 1-7 of trike60_training.ipynb, run here outside Colab (clone, pip, Drive and disconnect are skipped):
    a tiny run trains, re-running the training cell resumes it (and finds it complete), evaluation writes its JSON."""
    nb = json.loads((RL_DIR / "trike60_training.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 7 and len(nb["cells"]) <= 8
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RL_DIR, text=True).strip()
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"REPO_REF": head, "RUN_ID": "nb-test", "TOTAL_STEPS": 2048, "N_ENVS": 2,
                                  "EVAL_EPISODES": 1, "LOCAL_RUNS_DIR": str(tmp_path / "runs")}))
    monkeypatch.setenv("TRIKE60_PARAMS", str(params))
    monkeypatch.chdir(RL_DIR)
    ns: dict = {}
    for i, src in enumerate(code, start=1):
        exec(compile(src, f"cell{i}", "exec"), ns)
    run_dir = tmp_path / "runs" / "nb-test"
    assert (run_dir / "checkpoints" / "trike60_2048_steps.zip").exists()
    assert (run_dir / "eval" / "eval_2048.json").exists()
    out = capsys.readouterr().out
    assert "(new run)" in out and "Auto-disconnect skipped" in out
    exec(compile(code[3], "cell4", "exec"), ns)  # the run folder is found again ...
    exec(compile(code[4], "cell5", "exec"), ns)  # ... and the training cell resumes it with --resume
    out = capsys.readouterr().out
    assert "resuming: " in out and "--resume" in out and "already complete" in out


def test_parameters_cell_refuses_a_branch_name(tmp_path, monkeypatch):
    nb = json.loads((RL_DIR / "trike60_training.ipynb").read_text())
    cell1 = "".join(nb["cells"][1]["source"])
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"REPO_REF": "main", "RUN_ID": "x"}))
    monkeypatch.setenv("TRIKE60_PARAMS", str(params))
    with pytest.raises(ValueError, match="40 hex"):
        exec(compile(cell1, "cell1", "exec"), {})


@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM"])
def test_train_saves_a_pair_when_stopped(tmp_path, sig):
    """The notebook's stop button sends SIGINT to train.py (a killed runtime may send SIGTERM): it saves and exits."""
    import signal
    import time

    out = tmp_path / "run"
    proc = subprocess.Popen([sys.executable, "-u", str(RL_DIR / "train.py"), "--out", str(out), "--n-envs", "2",
                             "--n-steps", "256", "--total-steps", "10000000", "--log-every", "512"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(RL_DIR))
    lines = []
    deadline = time.time() + 300
    for line in proc.stdout:  # wait until learning is under way
        lines.append(line)
        if line.startswith("steps ") or time.time() > deadline:
            break
    proc.send_signal(getattr(signal, sig))
    rest, _ = proc.communicate(timeout=300)
    text = "".join(lines) + rest
    assert proc.returncode == 130, text
    assert "Session interrupted" in text
    rec = json.loads((out / "run_record.json").read_text())
    s = rec["sessions"][0]
    assert s["status"] == "interrupted" and s["steps_end"] > 0
    assert (out / "checkpoints" / f"trike60_{s['steps_end']}_steps.zip").exists()
    assert (out / "checkpoints" / f"trike60_vecnormalize_{s['steps_end']}_steps.pkl").exists()
