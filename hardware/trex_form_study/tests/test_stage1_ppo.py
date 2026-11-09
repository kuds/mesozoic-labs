"""Tests for Stage 1's PPO pieces (designlab/stage1 env.py, policy.py, train.py; STAGE1_PLAN.md section 2 and section 3
step 7, gate G6a).

Run from hardware/trex_form_study, in an environment with Stable-Baselines3, Gymnasium and torch (the PPO venv):
    python -m pytest tests/test_stage1_ppo.py -q
The env tests need the run directory's generated references and transitions (``python -m designlab.stage1 prepare``;
git-ignored) and skip without them; the policy tests need only torch and SB3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("gymnasium")
th = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab.stage1 import control, policy, train, tune  # noqa: E402
from designlab.stage1 import env as env_mod  # noqa: E402

BODY = "60-N-T-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"


def _need_run():
    for f in ("refs/nominal.npz", "transitions/start.npz", "transitions/stop.npz"):
        if not (RUN / BODY / f).exists():
            pytest.skip("the run directory's references and transitions are not generated "
                        "(python -m designlab.stage1 prepare ...)")


@pytest.fixture(scope="module")
def r3():
    _need_run()
    cfg = env_mod.EnvConfig(body=BODY, tasks=("walk", "stand"), randomize=False, push_prob=0.0, max_seconds=2.0)
    return env_mod.ResidualEnv(cfg)


# ------------------------------------------------------------------------------------------------
# The env
# ------------------------------------------------------------------------------------------------
def test_observation_layout_puts_the_actor_first():
    lay = env_mod.obs_layout()
    assert lay["actor_dim"] == env_mod.ACTOR_DIM == sum(w for _, w in env_mod.ACTOR_OBS)
    assert lay["critic_dim"] == env_mod.OBS_DIM
    starts = [lay["blocks"][n][0] for n, _ in env_mod.ACTOR_OBS + env_mod.CRITIC_OBS]
    assert starts == sorted(starts) and starts[0] == 0
    assert max(lay["blocks"][n][1] for n, _ in env_mod.ACTOR_OBS) == env_mod.ACTOR_DIM
    names = {n for n, _ in env_mod.CRITIC_OBS}
    assert {"root_vel", "height", "friction", "floor", "push", "joint_error", "dcm_error"} <= names


def test_r3_steps_at_50_hz_and_returns_finite_values(r3):
    obs, info = r3.reset(seed=3, options=dict(task="walk"))
    assert obs.shape == (env_mod.OBS_DIM,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))
    t0 = float(r3.p.d.time)
    obs, r, term, trunc, info = r3.step(np.zeros(env_mod.ACTION_DIM, np.float32))
    assert r3.p.d.time == pytest.approx(t0 + env_mod.ENV_SUBSTEPS * r3.dt)
    assert np.isfinite(r) and np.all(np.isfinite(obs)) and not term
    assert set(env_mod.PENALTIES) <= set(info["reward_terms"])
    assert 0.0 <= info["reward_terms"]["imitation"] <= 1.0 and 0.0 <= info["reward_terms"]["tracking"] <= 1.0


def test_zero_residual_reproduces_r2_exactly():
    """G6a: R3 with a zero action is R2, every physics step's state bit for bit."""
    _need_run()
    out = env_mod.zero_residual_check(BODY, task="walk", seconds=1.0, episode_seed=11)
    assert out["identical"], out
    assert out["physics_steps"] == out["reference_steps"] == 500


def test_a_residual_moves_the_servo_targets(r3):
    r3.reset(seed=5, options=dict(task="walk", episode_seed=5))
    snap_q = r3.p.d.qpos.copy()
    r3.step(np.zeros(env_mod.ACTION_DIM, np.float32))
    ctrl_zero = r3.p.d.ctrl.copy()
    r3.reset(seed=5, options=dict(task="walk", episode_seed=5))
    assert np.array_equal(r3.p.d.qpos, snap_q)
    a = np.zeros(env_mod.ACTION_DIM, np.float32)
    a[r3.p.joints.index("left_knee")] = 1.0
    r3.step(a)
    i = r3.p.joints.index("left_knee")
    assert r3.tk.residual is not None and r3.tk.residual[i] == pytest.approx(r3.alpha * env_mod.RESIDUAL_LEG_RAD)
    assert not np.array_equal(r3.p.d.ctrl, ctrl_zero)


def test_the_rate_residual_scales_the_reference_clock(r3):
    r3.reset(seed=6, options=dict(task="walk", episode_seed=6))
    a = np.zeros(env_mod.ACTION_DIM, np.float32)
    a[-1] = 1.0
    tau0 = r3.tk.tau_ref
    r3.step(a)
    assert r3.tk.rate_factor == pytest.approx(1.0 + env_mod.RESIDUAL_RATE * r3.alpha)
    assert r3.tk.tau_ref > tau0


def test_randomisation_is_drawn_per_episode_and_restored(r3):
    m, p = r3.p.m, r3.p
    nominal = (m.geom_friction.copy(), float(m.body_mass[p.trunk]), m.actuator_gainprm[:, 0].copy())
    r3.reset(seed=1, options=dict(task="stand", factors=dict(friction=0.6, mass=1.1, kp=0.8, noise_scale=1.3)))
    assert float(m.body_mass[p.trunk]) == pytest.approx(1.1 * nominal[1])
    assert np.allclose(m.actuator_gainprm[:, 0], 0.8 * nominal[2])
    assert r3.tk.noise.scale == pytest.approx(1.3)
    r3.reset(seed=1, options=dict(task="stand"))  # the config draws nothing (randomize False): the nominal plant
    assert np.array_equal(m.geom_friction, nominal[0]) and float(m.body_mass[p.trunk]) == nominal[1]
    assert np.array_equal(m.actuator_gainprm[:, 0], nominal[2])


def test_latency_applies_the_command_of_steps_ago(r3):
    r3.reset(seed=2, options=dict(task="walk", factors=dict(latency_steps=3)))
    p = r3.p
    held = p.d.ctrl.copy()
    z = np.zeros(r3.nj)
    out = []
    try:
        for k in range(6):
            p.command(np.full(r3.nj, 0.01 * (k + 1)), z, z)
            out.append(p.d.ctrl.copy())
    finally:
        r3._latency_off()
    assert all(np.array_equal(o, held) for o in out[:3])  # the first three steps hold what was there
    for k in range(3, 6):
        assert np.allclose(out[k], p.G @ np.full(r3.nj, 0.01 * (k - 2)))  # then the command of three steps before
    assert "command" not in p.__dict__  # the plant's own command again


def test_policy_only_env_sets_targets_within_the_ranges():
    _need_run()
    cfg = env_mod.EnvConfig(body=BODY, tasks=("walk",), randomize=False, push_prob=0.0, max_seconds=1.0)
    e = env_mod.PolicyOnlyEnv(cfg)
    obs, _ = e.reset(seed=4)
    assert obs.shape == (env_mod.OBS_DIM,)
    e.step(np.ones(env_mod.ACTION_DIM, np.float32))
    assert np.all(e.q_target <= e.hi + 1e-12) and np.all(e.q_target >= e.lo - 1e-12)
    assert np.all(e.span >= env_mod.POLICY_ONLY_MIN_SPAN_RAD)


def test_tuned_gains_come_from_tune_json_else_the_start_flagged(tmp_path):
    g, info = env_mod.tuned_gains(tmp_path, 42)
    assert np.array_equal(g, control.START_GAINS) and info["flag"] == "untuned"
    sel = {n: float(v) + 0.25 for n, v in control.gains_dict().items()}
    (tmp_path / "tune.json").write_text(json.dumps(dict(seeds={"43": dict(selected="seed43_gen20",
                                                                        selected_gains=sel, pass_=True)})))
    g, info = env_mod.tuned_gains(tmp_path, 43)
    assert np.allclose(g, control.START_GAINS + 0.25) and info["flag"] is None
    g, info = env_mod.tuned_gains(tmp_path, 42)  # another seed's selection, flagged
    assert np.allclose(g, control.START_GAINS + 0.25) and "43" in info["flag"]


# ------------------------------------------------------------------------------------------------
# The privileged-critic policy
# ------------------------------------------------------------------------------------------------
def _policy(actor_dim=5, full=9):
    from gymnasium import spaces

    obs = spaces.Box(-np.inf, np.inf, (full,), np.float32)
    act = spaces.Box(-1.0, 1.0, (3,), np.float32)
    th.manual_seed(0)
    return policy.PrivilegedCriticPolicy(obs, act, lambda _: 1e-4, actor_dim=actor_dim,
                                         net_arch=dict(pi=[16, 16], vf=[16, 16]))


def test_the_actor_reads_its_columns_only_and_the_critic_all():
    pol = _policy()
    x = th.randn(7, 9)
    y = x.clone()
    y[:, 5:] = th.randn(7, 4) * 100.0
    with th.no_grad():
        assert th.equal(pol.get_distribution(x).distribution.mean, pol.get_distribution(y).distribution.mean)
        assert not th.equal(pol.predict_values(x), pol.predict_values(y))
    assert pol.mlp_extractor.policy_net[0].in_features == 5 and pol.mlp_extractor.value_net[0].in_features == 9


def test_predict_takes_the_actors_columns_alone():
    pol = _policy()
    x = np.random.default_rng(0).standard_normal((4, 9)).astype(np.float32)
    a_full, _ = pol.predict(x, deterministic=True)
    a_actor, _ = pol.predict(x[:, :5], deterministic=True)
    assert np.array_equal(a_full, a_actor)
    a1, _ = pol.predict(x[0, :5], deterministic=True)  # one observation (torch's batch-of-one kernels on both sides)
    assert np.array_equal(a1, pol.predict(x[0], deterministic=True)[0])
    assert a1.shape == (3,) and np.allclose(a1, a_full[0], atol=1e-6)


def test_the_actor_gets_no_gradient_from_the_privileged_inputs():
    pol = _policy()
    x = th.randn(3, 9, requires_grad=True)
    pol.get_distribution(x).distribution.mean.sum().backward()
    assert th.all(x.grad[:, 5:] == 0) and th.any(x.grad[:, :5] != 0)


def test_a_saved_model_keeps_its_actor_width(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    venv = make_vec_env("Pendulum-v1", n_envs=1, seed=0)  # a 3-value observation: the actor reads the first 2
    kw = dict(policy.model_kwargs(2, net_arch=(8, 8)), n_steps=16, batch_size=16)
    model = PPO(env=venv, device="cpu", **kw)
    model.save(tmp_path / "m.zip")
    loaded = PPO.load(tmp_path / "m.zip", device="cpu")
    assert isinstance(loaded.policy, policy.PrivilegedCriticPolicy) and loaded.policy.actor_dim == 2
    x = np.random.default_rng(1).standard_normal((5, 3)).astype(np.float32)
    assert np.array_equal(model.predict(x, deterministic=True)[0], loaded.predict(x, deterministic=True)[0])


# ------------------------------------------------------------------------------------------------
# Training: checkpoints, resume, the final-checkpoint rule
# ------------------------------------------------------------------------------------------------
def test_final_checkpoint_is_the_budget_else_the_last_common(tmp_path):
    for name, steps in (("a", (100, 200, 300)), ("b", (100, 200))):
        ck = tmp_path / name / "checkpoints"
        ck.mkdir(parents=True)
        for s in steps:
            (ck / f"model_{s}.zip").write_text("x")
            (ck / f"vecnormalize_{s}.pkl").write_text("x")
    (tmp_path / "b" / "checkpoints" / "model_300.zip").write_text("x")  # no normalisation saved: incomplete
    assert train.checkpoints(tmp_path / "b") == [100, 200]
    assert train.final_checkpoint([tmp_path / "a", tmp_path / "b"], budget=300) == 200
    assert train.final_checkpoint([tmp_path / "a"], budget=300) == 300


def test_training_checkpoints_and_resumes(tmp_path):
    _need_run()
    out = tmp_path / "runs"
    (out / BODY).mkdir(parents=True)
    for sub in ("refs", "transitions"):
        (out / BODY / sub).symlink_to(RUN / BODY / sub)
    ppo = dict(n_steps=16, batch_size=16, n_epochs=1)
    cfg = dict(tasks=("walk",), max_seconds=1.0)
    doc = train.train(BODY, 42, steps=32, out=out, n_envs=1, checkpoint_every=16, g6=False, config=cfg, ppo=ppo,
                      log=lambda *_: None)
    rd = train.run_dir_of(out, BODY, "R3", 42)
    assert doc["status"] == "done" and doc["num_timesteps"] == 32 and train.checkpoints(rd) == [16, 32]
    assert doc["env"]["gains_info"]["flag"] == "untuned" and doc["settings"]["design2"]["n_steps"] == 2048
    assert doc["settings"]["ppo_overrides"] == ppo
    with pytest.raises(SystemExit):
        train.train(BODY, 42, steps=32, out=out, n_envs=1, checkpoint_every=16, g6=False, config=cfg, ppo=ppo)
    doc = train.train(BODY, 42, steps=64, out=out, resume=True, n_envs=1, checkpoint_every=16, g6=False, config=cfg,
                      ppo=ppo, log=lambda *_: None)
    assert doc["num_timesteps"] == 64 and doc["sessions"][-1]["resumed_from"] == 32 and len(doc["sessions"]) == 2
    assert train.checkpoints(rd)[-1] == 64 and json.loads((rd / "run.json").read_text())["status"] == "done"
    with pytest.raises(SystemExit):  # other settings: refused
        train.train(BODY, 42, steps=96, out=out, resume=True, n_envs=1, checkpoint_every=32, g6=False, config=cfg,
                    ppo=ppo)


def test_cli_trains_and_records(tmp_path, monkeypatch):
    calls = {}

    def fake_train(body, seed, **kw):
        calls.update(kw, body=body, seed=seed)
        return dict(num_timesteps=10, status="done", g6={"G6a_zero_residual": dict(identical=True)})

    monkeypatch.setattr(train, "train", fake_train)
    from designlab.stage1.__main__ import main

    assert main(["train", "--body", BODY, "--seed", "43", "--mode", "R4", "--steps", "10", "--out", str(tmp_path)]) == 0
    assert calls["body"] == BODY and calls["seed"] == 43 and calls["mode"] == "R4" and calls["steps"] == 10
    assert tune.RUN.name == "2026-11-08"
