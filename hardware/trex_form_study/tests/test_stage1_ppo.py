"""Tests for Stage 1's PPO pieces (designlab/stage1 env.py, policy.py, train.py; STAGE1_PLAN.md section 2 and section 3
step 7, gate G6a).

Run from hardware/trex_form_study, in an environment with Stable-Baselines3, Gymnasium and torch (the PPO venv):
    python -m pytest tests/test_stage1_ppo.py -q
The env tests need the run directory's generated references and transitions (``python -m designlab.stage1 prepare``;
git-ignored) and skip without them; the policy tests need only torch and SB3.
"""

from __future__ import annotations

import copy
import json
import math
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
import mujoco  # noqa: E402
from designlab.stage1 import control, metrics, plant, policy, train, tune  # noqa: E402
from designlab.stage1 import env as env_mod  # noqa: E402

BODY = "60-N-T-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
#: The fields mj_setConst derives from the masses (MuJoCo 3.10), with the masses themselves.
MASS_FIELDS = {"actuator_acc0", "body_inertia", "body_invweight0", "body_mass", "body_subtreemass", "dof_M0",
               "dof_invweight0", "light_poscom0", "tendon_invweight0", "stat.meaninertia", "stat.meanmass"}


def _need_run():
    for f in ("refs/nominal.npz", "refs/carpet.npz", "transitions/start.npz", "transitions/stop.npz"):
        if not (RUN / BODY / f).exists():
            pytest.skip("the run directory's references and transitions are not generated "
                        "(python -m designlab.stage1 prepare ...)")


@pytest.fixture(scope="module")
def r3():
    _need_run()
    cfg = env_mod.EnvConfig(body=BODY, tasks=("walk", "stand", "carpet_walk"), randomize=False, push_prob=0.0,
                            max_seconds=2.0)
    return env_mod.ResidualEnv(cfg)


def _out(tmp_path) -> Path:
    """A run directory holding the body's references and transitions (no tune.json)."""
    out = tmp_path / "runs"
    (out / BODY).mkdir(parents=True)
    for sub in ("refs", "transitions"):
        (out / BODY / sub).symlink_to(RUN / BODY / sub)
    return out


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
    assert set(env_mod.PENALTIES) <= set(info["reward_terms"]) and info["floor"] == "hard"
    assert 0.0 <= info["reward_terms"]["imitation"] <= 1.0 and 0.0 <= info["reward_terms"]["tracking"] <= 1.0


def test_zero_residual_reproduces_r2_exactly():
    """G6a: R3 with a zero action is R2, every physics step's state bit for bit, and ends where R2 does."""
    _need_run()
    out = env_mod.zero_residual_check(BODY, task="walk", seconds=1.0, episode_seed=11)
    assert out["identical"] and out["ends_alike"] and not out["reference_fell"], out
    assert out["physics_steps"] == out["reference_steps"] == out["compared_steps"] == 500


def test_zero_residual_check_compares_to_r2s_fall_and_its_env_step():
    """A falling R2 (60-N-T-10 at the tuning start falls on robot_floor_carpet in every start): R3 matches it to the
    physics step it fell in and terminates in the env step that holds that step, not later."""
    _need_run()
    out = env_mod.zero_residual_check(BODY, task="walk", seconds=8.0, episode_seed=11, floor="robot_floor_carpet")
    assert out["reference_fell"] and out["identical"], out
    assert out["compared_steps"] == out["reference_steps"] and out["r3_terminated"]
    assert out["env_steps"] == out["fall_env_step"] == math.ceil(out["reference_steps"] / env_mod.ENV_SUBSTEPS)
    assert out["physics_steps"] == env_mod.ENV_SUBSTEPS * out["env_steps"] >= out["reference_steps"]


def test_a_heavier_or_lighter_trunk_keeps_the_plant_consistent_and_r3_is_r2_on_it():
    """Mass randomisation (MASS_RULE): the plant is the model MuJoCo derives for that trunk (its CoM the mass-weighted
    mean), the controller keeps the nominal model, R3 with a zero residual is R2 on that same plant, and neither falls
    where nominal R2 does not (with the subtree masses left stale the CoM was 41-47 mm off and R2 fell at 4-6 s)."""
    _need_run()
    e = env_mod.ResidualEnv(env_mod.EnvConfig(body=BODY, tasks=("walk",), randomize=False, push_prob=0.0))
    m, nom, trunk = e.p.m, e.nominal_model, e.p.trunk
    e.reset(seed=0, options=dict(task="walk", episode_seed=11, push=None, factors=dict(mass=1.1)))
    want = copy.deepcopy(nom)  # the same trunk compiled consistently, independently of the env
    want.body_mass[trunk] *= 1.1
    want.body_inertia[trunk] *= 1.1
    mujoco.mj_setConst(want, mujoco.MjData(want))
    assert env_mod.model_differences(m, want) == []
    assert set(env_mod.model_differences(m, nom)) == MASS_FIELDS
    d = e.p.d
    d.qpos[0] += 1.0  # away from the origin, where a stale subtree mass shows
    mujoco.mj_forward(m, d)
    exact = (m.body_mass[1:, None] * d.xipos[1:]).sum(0) / m.body_mass[1:].sum()
    assert np.allclose(d.subtree_com[trunk], exact, rtol=0, atol=1e-12)
    tk = e.tk
    assert tk.model is nom and tk.est.m is nom and tk.mass == float(nom.body_subtreemass[trunk])
    assert tk.sensors.weight == pytest.approx(e.weight) and e.weight_true == pytest.approx(
        float(m.body_subtreemass[trunk]) * env_mod.GRAVITY) and e.weight_true > e.weight
    nominal = env_mod.zero_residual_check(BODY, task="walk", seconds=8.0, episode_seed=11)
    assert nominal["identical"] and not nominal["reference_fell"], nominal
    for f in (1.1, 0.9):
        out = env_mod.zero_residual_check(BODY, task="walk", seconds=8.0, episode_seed=11, factors=dict(mass=f))
        assert out["identical"] and out["ends_alike"], out
        assert not out["reference_fell"] and not out["r3_terminated"], (f, out)


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
    """Every factor lands in the plant (the controller's model stays nominal) and the next nominal episode's model is
    the nominal model bit for bit, every array mj_setConst derives included."""
    m, p = r3.p.m, r3.p
    nominal = (m.geom_friction.copy(), float(m.body_mass[p.trunk]), m.actuator_gainprm[:, 0].copy())
    r3.reset(seed=1, options=dict(task="stand", factors=dict(friction=0.6, mass=1.1, kp=0.8, noise_scale=1.3)))
    assert float(m.body_mass[p.trunk]) == pytest.approx(1.1 * nominal[1])
    assert np.allclose(m.actuator_gainprm[:, 0], 0.8 * nominal[2])
    assert r3.tk.noise.scale == pytest.approx(1.3)
    assert set(env_mod.model_differences(m, r3.nominal_model)) == MASS_FIELDS | {
        "geom_friction", "actuator_gainprm", "actuator_biasprm"}
    assert r3.tk.est.m is r3.nominal_model and r3.nominal_model.body_mass[p.trunk] == nominal[1]
    r3.reset(seed=1, options=dict(task="stand"))  # the config draws nothing (randomize False): the nominal plant
    assert np.array_equal(m.geom_friction, nominal[0]) and float(m.body_mass[p.trunk]) == nominal[1]
    assert np.array_equal(m.actuator_gainprm[:, 0], nominal[2])
    assert env_mod.model_differences(m, r3.nominal_model) == []
    assert env_mod.setconst_changes(r3.nominal_model) == []


def test_the_carpet_walk_runs_the_carpet_swing_at_mu_054():
    _need_run()
    e = env_mod.ResidualEnv(env_mod.EnvConfig(body=BODY, tasks=("carpet_walk",), push_prob=0.0, max_seconds=1.0))
    _, info = e.reset(seed=8)
    assert e.task == "carpet_walk" and info["factors"]["friction"] == env_mod.TASK_FRICTION["carpet_walk"] == 0.54
    cg = (e.p.m.geom_contype | e.p.m.geom_conaffinity) > 0
    assert np.all(e.p.m.geom_friction[cg, 0] == 0.54)
    assert [s.name for s in e.track.segments] == ["start", "carpet", "stop"]
    from designlab.stage1.refs import Reference

    assert e.period["carpet_walk"] == Reference.load(RUN / BODY / "refs" / "nominal.npz").period  # the nominal stride


def test_a_pile_env_steps_and_its_feet_skip_the_probes():
    """The training pile (plan section 2's 8 mm grid): the swing foot's lowest point leaves out the pile's probes."""
    _need_run()
    e = env_mod.ResidualEnv(env_mod.EnvConfig(body=BODY, floor="low", tasks=("walk",), randomize=False, push_prob=0.0,
                                              max_seconds=0.2))
    hard = plant.build(BODY, "hard", settle=False)
    want = metrics._lowest_point_tables(hard.m, hard.feet)
    got = env_mod.foot_tables(hard.m, hard.feet)
    assert all(np.array_equal(a, b) for a, b in zip(want[0] + want[1], got[0] + got[1], strict=True))
    assert e.p.meta["floor_info"]["probes_per_foot"]["left"] > 0
    e.reset(seed=2, options=dict(task="walk"))
    obs, r, term, trunc, info = e.step(np.zeros(env_mod.ACTION_DIM, np.float32))
    assert np.all(np.isfinite(obs)) and info["floor"] == "low"
    assert obs[env_mod.obs_layout()["blocks"]["floor"][0] + 2] == 1.0  # the pile's floor code


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


def test_episode_seeds_follow_the_reset_seed(r3):
    """EPISODE_SEED_RULE: episode k after a reset seeded s is 1,000,000 s + k; a given seed is kept."""
    r3.reset(seed=420001, options=dict(task="stand"))
    seen = [r3.episode_seed]
    for _ in range(2):
        r3.reset(options=dict(task="stand"))
        seen.append(r3.episode_seed)
    assert seen == [env_mod.EPISODE_STRIDE * 420001 + k for k in range(3)]
    assert r3.episode_stream() == dict(reset_seed=420001, episodes=3, first_episode_seed=420001 * 10**6)
    r3.reset(options=dict(task="stand", episode_seed=5))
    assert r3.episode_seed == 5 and r3.episode_stream()["episodes"] == 3


def test_policy_only_env_sets_targets_within_the_ranges():
    _need_run()
    cfg = env_mod.EnvConfig(body=BODY, tasks=("walk",), randomize=False, push_prob=0.0, max_seconds=1.0)
    e = env_mod.PolicyOnlyEnv(cfg)
    obs, _ = e.reset(seed=4)
    assert obs.shape == (env_mod.OBS_DIM,)
    e.step(np.ones(env_mod.ACTION_DIM, np.float32))
    assert np.all(e.q_target <= e.hi + 1e-12) and np.all(e.q_target >= e.lo - 1e-12)
    assert np.all(e.span >= env_mod.POLICY_ONLY_MIN_SPAN_RAD)
    assert e.est.m is e.nominal_model and e.gains is None and e.gains_info["status"] == "not used"


def _tune_doc(seeds: dict, body: str = BODY, pass_=None) -> dict:
    """A tune.json with ``seeds``: seed -> (pass_, falls, S, P, J, gain offset) of its selection."""
    out = {}
    for s, (ok, falls, S, P, J, off) in seeds.items():
        sel = f"seed{s}_gen60"
        out[str(s)] = dict(selected=sel, pass_=ok, refused=False,
                           selected_gains={n: float(v) + off for n, v in control.gains_dict().items()},
                           selection=[dict(candidate="start", rank=2, falls=9, S_mean=1.0, P_mean=1.0, J_mean=2.0),
                                      dict(candidate=sel, rank=1, falls=falls, S_mean=S, P_mean=P, J_mean=J)])
    judged = [v["pass_"] for v in out.values() if v["pass_"] is not None]
    return dict(body=body, seeds=out, pass_=pass_ if pass_ is not None else (any(judged) if judged else None),
                seeds_passing=[int(s) for s, v in out.items() if v["pass_"]])


def test_tuned_gains_are_the_body_level_g5_selection(tmp_path):
    g, info = env_mod.tuned_gains(tmp_path)
    assert np.array_equal(g, control.START_GAINS) and info["flag"] == "untuned" and info["status"] == "missing"
    path = tmp_path / "tune.json"
    path.write_text(json.dumps(_tune_doc({42: (True, 0, 0.0, 0.05, 0.05, 0.1), 43: (True, 0, 0.0, 0.04, 0.04, 0.2)})))
    g, info = env_mod.tuned_gains(tmp_path, BODY)  # both pass: the better selection, for both PPO seeds
    assert np.allclose(g, control.START_GAINS + 0.2) and info["status"] == "passed" and info["flag"] is None
    assert info["cma_seed"] == 43 and info["candidate"] == "seed43_gen60" and info["sha256"] == plant.sha256(path)
    path.write_text(json.dumps(_tune_doc({42: (True, 1, 0.1, 0.05, 0.15, 0.1), 43: (False, 0, 0.0, 0.01, 0.01, 0.2)})))
    g, info = env_mod.tuned_gains(tmp_path, BODY)  # only a selection that passes G5 counts
    assert np.allclose(g, control.START_GAINS + 0.1) and info["cma_seed"] == 42 and info["status"] == "passed"
    path.write_text(json.dumps(_tune_doc({42: (False, 3, 0.5, 0.05, 0.55, 0.1), 43: (False, 2, 0.6, 0.04, 0.64, 0.2)})))
    g, info = env_mod.tuned_gains(tmp_path, BODY)  # G5 failed: the best judged selection, flagged
    assert info["status"] == "failed" and info["flag"] == "G5 failed" and info["cma_seed"] == 43
    path.write_text(json.dumps(_tune_doc({42: (None, 0, 0.0, 0.05, 0.05, 0.1)})))
    assert env_mod.tuned_gains(tmp_path, BODY)[1]["status"] == "not judged"
    with pytest.raises(ValueError, match="75-M-H-10"):
        env_mod.tuned_gains(tmp_path, "75-M-H-10")


def test_the_g5_record_of_75_m_h_10_selects_one_candidate_for_both_seeds():
    """On a real G5 record where the run directory holds one (75-M-H-10's, the first fit to finish)."""
    path = RUN / "75-M-H-10" / "tune.json"
    if not path.exists():
        pytest.skip("no G5 record for 75-M-H-10 in the run directory")
    doc = json.loads(path.read_text())
    g, info = env_mod.tuned_gains(path.parent, "75-M-H-10")
    assert info["status"] == ("passed" if doc["pass_"] else "failed" if doc["pass_"] is False else "not judged")
    sel = doc["seeds"][str(info["cma_seed"])]
    assert np.array_equal(g, [sel["selected_gains"][n] for n in control.GAIN_NAMES])
    if doc["pass_"]:
        assert info["cma_seed"] in doc["seeds_passing"]


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
    assert th.allclose(loaded.policy.log_std, th.full((1,), policy.LOG_STD_INIT))


def test_log_std_init_is_the_measured_noise():
    """LOG_STD_RULE: the largest measured sigma at and below which R3 keeps 18/20 of its nominal walks."""
    path = ROOT / policy.ACTION_NOISE_RECORD
    if not path.exists():
        pytest.skip("no action-noise record")
    doc = json.loads(path.read_text())
    assert policy.ACTION_NOISE_SIGMA == doc["sigma"] and policy.LOG_STD_INIT == pytest.approx(math.log(doc["sigma"]))
    assert doc["of"] == 20 and doc["body"] == "75-M-H-10" and doc["gains"]["tuned"]["info"]["status"] == "passed"
    sig = sorted(doc["sigmas"])
    k = sig.index(doc["sigma"])
    assert all(doc["survived"][repr(s)] >= 18 for s in sig[: k + 1])
    assert k + 1 == len(sig) or doc["survived"][repr(sig[k + 1])] < 18
    assert policy.model_kwargs(4)["policy_kwargs"]["log_std_init"] == policy.LOG_STD_INIT


# ------------------------------------------------------------------------------------------------
# The tracker's solve
# ------------------------------------------------------------------------------------------------
def test_solve2_is_np_linalg_solve_and_falls_back_to_it(monkeypatch):
    rng = np.random.default_rng(3)
    for n in (2, 6):
        for _ in range(20):
            A, b = rng.standard_normal((n, n)) + n * np.eye(n), rng.standard_normal(n)
            assert np.array_equal(control._solve2(A, b), np.linalg.solve(A, b))
    monkeypatch.setattr(control, "_SOLVE1", None)  # a numpy without the private gufunc
    A, b = rng.standard_normal((2, 2)) + 2 * np.eye(2), rng.standard_normal(2)
    assert np.array_equal(control._solve2(A, b), np.linalg.solve(A, b))


# ------------------------------------------------------------------------------------------------
# Training: floors, gains, seeds, checkpoints, resume, the judged checkpoint
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


def test_the_judged_checkpoint_is_over_every_run_of_a_mode(tmp_path):
    runs = train.expected_runs(tmp_path, "R3")
    assert len(runs) == 6 and len(train.expected_runs(tmp_path, "R4")) == 3
    for i, d in enumerate(runs.values()):
        (d / "checkpoints").mkdir(parents=True)
        for s in (100, 200) if i else (100, 200, 300):
            (d / "checkpoints" / f"model_{s}.zip").write_text("x")
            (d / "checkpoints" / f"vecnormalize_{s}.pkl").write_text("x")
        (d / "run.json").write_text(json.dumps(dict(status="partial")))
    rec = train.judged_checkpoint(tmp_path, "R3", budget=300)
    assert rec["steps"] == 200 and not rec["final"]
    rec = train.judge(tmp_path, "R3", budget=300, final=True)  # the cap stopped training
    assert rec["final"] and all(json.loads((d / "run.json").read_text())["judged_checkpoint"] == rec
                                for d in runs.values())


def test_the_budget_is_4m_steps_in_whole_rollouts():
    assert train.whole_rollouts(4_000_000) == 4_005_888 == train.BUDGET_STEPS and train.BUDGET_ASKED == 4_000_000
    assert train.whole_rollouts(32, 16) == 32 and train.whole_rollouts(33, 16) == 48 and train.whole_rollouts(1, 16) == 16
    assert train.BUDGET_STEPS % (policy.DESIGN2["n_envs"] * policy.DESIGN2["n_steps"]) == 0


def test_the_learning_rate_is_design_2s_x_01():
    kw = policy.model_kwargs(env_mod.ACTOR_DIM)
    assert (policy.LR_START, policy.LR_END) == (1e-5, 3e-6) and kw["target_kl"] == policy.DESIGN2["target_kl"] == 0.03
    assert kw["learning_rate"](1.0) == pytest.approx(1e-5) and kw["learning_rate"](0.0) == pytest.approx(3e-6)
    assert policy.LR_START == pytest.approx(0.1 * policy.DESIGN2["lr_start"])
    assert policy.LR_END == pytest.approx(0.1 * policy.DESIGN2["lr_end"])


def _mon_rows(rewards, falls):
    return [dict(episode_seed=5001 + i, mean_reward=r, fell=f) for i, (r, f) in enumerate(zip(rewards, falls))]


def test_g6c_needs_fewer_falls_or_a_paired_gain_beyond_the_margin():
    z = _mon_rows([0.700, 0.702, 0.701], [False] * 3)
    # the verifier's case: a 0.0039 gain after one Adam step is not a win
    assert not train.beats_r2(_mon_rows([0.7039, 0.7059, 0.7049], [False] * 3), z)["beats"]
    assert train.beats_r2(_mon_rows([0.712, 0.714, 0.713], [False] * 3), z)["beats"]  # +0.012 on every seed
    assert not train.beats_r2(_mon_rows([0.730, 0.699, 0.730], [False] * 3), z)["beats"]  # one seed worse
    wide = _mon_rows([0.60, 0.70, 0.65], [False] * 3)  # R2's spread 0.10: the gain must reach it
    assert not train.beats_r2(_mon_rows([0.62, 0.72, 0.67], [False] * 3), wide)["beats"]
    r = train.beats_r2(_mon_rows([0.71, 0.81, 0.76], [False] * 3), wide)
    assert r["beats"] and r["gain_needed"] == pytest.approx(0.10) and r["mean_gain"] == pytest.approx(0.11)
    fall = _mon_rows([0.5, 0.5, 0.5], [True] * 3)
    assert train.beats_r2(_mon_rows([0.2, 0.5, 0.5], [False, True, True]), fall)["fewer_falls"]  # fewer falls wins
    assert not train.beats_r2(_mon_rows([0.6, 0.6, 0.6], [True, True, True]), _mon_rows([0.5] * 3, [True, True, False]))[
        "beats"]  # more falls never wins


def test_the_monitor_runs_each_condition_on_its_floor(monkeypatch):
    _need_run()
    monkeypatch.setattr(train, "MONITOR", dict(conditions=(("walk", "hard"), ("carpet_walk", "robot_floor_carpet")),
                                               episode_seeds=(5001,)))
    floors = []
    real = env_mod.make

    def make(mode, cfg):
        floors.append((cfg.tasks, cfg.floor, cfg.randomize, cfg.push_prob))
        return real(mode, cfg)

    monkeypatch.setattr(env_mod, "make", make)

    class Zero:
        def predict(self, obs, deterministic=True):
            return np.zeros((1, env_mod.ACTION_DIM), np.float32), None

    class Same:
        def normalize_obs(self, obs):
            return obs

    cfg = train.env_configs(BODY, 1, dict(max_seconds=0.4), RUN)[0]
    mon = train.monitor(Zero(), Same(), BODY, "R3", cfg)
    assert floors == [(("walk",), "hard", False, 0.0), (("carpet_walk",), "robot_floor_carpet", False, 0.0)]
    assert set(mon["summary"]["policy"]) == {"walk", "carpet_walk on robot_floor_carpet"}
    assert mon["rows"]["policy"] == mon["rows"]["zero"]  # a zero policy is R2
    assert mon["policy_beats_zero_on"] == [] and not any(v["beats"] for v in mon["g6c_by_condition"].values())


def test_env_configs_spread_the_floors():
    cfgs = train.env_configs(BODY, 6, dict(max_seconds=1.0), Path("/x"))
    assert [c.floor for c in cfgs] == ["hard", "hard", "robot_floor_carpet", "low", "hard", "hard"]
    assert all(c.run_dir == f"/x/{BODY}" and c.max_seconds == 1.0 and c.pile_grid == 0.008 for c in cfgs)
    assert {c.floor for c in train.env_configs(BODY, 4, dict(floor="hard"), Path("/x"))} == {"hard"}
    assert "carpet_walk" in cfgs[0].tasks and cfgs[0].kp == (0.8, 1.2)  # KP_RULE: training's range


def test_env_seeds_are_disjoint_across_seeds_and_sessions():
    seen = set()
    for seed in train.SEEDS:
        for k in range(3):
            base, s = train.env_seeds(seed, k, 4)
            assert base == 10_000 * seed + 100 * k and s == [base + i for i in range(4)]
            assert not seen & set(s)
            seen |= set(s)
    with pytest.raises(SystemExit):
        train.env_seeds(42, 100, 4)


def test_r3_is_refused_without_a_g5_pass(tmp_path):
    _need_run()
    out = _out(tmp_path)
    with pytest.raises(SystemExit, match="allow-untuned"):
        train.train(BODY, 42, steps=16, out=out, n_envs=1, g6=False, config=dict(tasks=("walk",)))
    (out / BODY / "tune.json").write_text(json.dumps(_tune_doc({42: (False, 4, 0.5, 0.1, 0.6, 0.0)})))
    with pytest.raises(SystemExit, match="G5 failed|failed"):
        train.train(BODY, 42, steps=16, out=out, n_envs=1, g6=False, config=dict(tasks=("walk",)))
    assert not (out / BODY / "ppo" / "42" / "run.json").exists()


def test_training_checkpoints_and_resumes(tmp_path, monkeypatch):
    _need_run()
    out = _out(tmp_path)
    monkeypatch.setattr(train, "G6A_MEASURE_STEPS", 16)
    monkeypatch.setattr(train, "G6A_CHECK", dict(task="walk", seconds=0.2, episode_seed=7))
    ppo = dict(n_steps=16, batch_size=16, n_epochs=1)
    cfg = dict(tasks=("walk",), max_seconds=1.0)
    written = []
    real_write = train._write_json
    monkeypatch.setattr(train, "_write_json", lambda path, d: (written.append(copy.deepcopy(d)), real_write(path, d)))
    doc = train.train(BODY, 42, steps=30, out=out, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo,
                      allow_untuned=True, log=lambda *_: None)
    rd = train.run_dir_of(out, BODY, "R3", 42)
    assert doc["status"] == "done" and doc["num_timesteps"] == 32 and train.checkpoints(rd) == [16, 32]
    assert doc["budget"] == 32 and doc["budget_asked"] == 30  # BUDGET_RULE: whole rollouts
    # CHECKPOINT_RULE: saved at the rollout start after the first update, and after the last update; the streams with
    # every checkpoint (a killed session keeps them)
    at16 = next(d for d in written if d.get("checkpoints") == [16])
    assert at16["sessions"][0]["episode_streams"][0]["reset_seed"] == 420000
    from stable_baselines3 import PPO
    w16, w32 = (PPO.load(str(rd / "checkpoints" / f"model_{n}.zip"), device="cpu").policy.state_dict() for n in (16, 32))
    assert any(not th.equal(w16[k], w32[k]) for k in w16)  # the last update is saved
    assert doc["settings"]["learning_rate"]["start"] == 1e-5 and "rollout start" in doc["settings"]["checkpoint_rule"]
    assert doc["gains"]["flag"] == "untuned" and doc["gains"]["allow_untuned"]
    assert doc["settings"]["design2"]["n_steps"] == 2048 and doc["settings"]["ppo_overrides"] == ppo
    assert doc["settings"]["log_std_init"] == policy.LOG_STD_INIT
    s0 = doc["sessions"][0]
    assert s0["env_seeds"] == [420000] and s0["seed_base"] == 420000 and s0["floors"] == ["hard"]
    assert s0["env"]["config"]["run_dir"] == str(out / BODY) and s0["env"]["gains"] == doc["gains"]["values"]
    assert doc["env_configs"][0]["run_dir"] == str(out / BODY) and doc["env_configs"][0]["floor"] == "hard"
    assert s0["versions"]["torch"].startswith("2.") and s0["versions"]["stable_baselines3"]
    assert s0["ppo_venv"] == train.PPO_VENV and s0["solve2"] == control.SOLVE2_PATH
    assert s0["episode_streams"][0]["reset_seed"] == 420000 and s0["episode_streams"][0]["episodes"] >= 1
    g6a = doc["g6"]["G6a_rate"]  # judged per wall second, with the process count
    assert g6a["judged_on"].startswith("env steps per wall second") and g6a["env_processes"] == 1
    assert g6a["pass_"] == (g6a["env_steps_per_wall_s"] >= train.G6A_ENV_STEPS_PER_S) and g6a["cpu_s_env_workers"] == 0
    assert doc["g6"]["G6a_zero_residual"]["identical"]
    assert doc["judged_checkpoint"]["steps"] is None and not doc["judged_checkpoint"]["final"]  # the other runs: none
    with pytest.raises(SystemExit):
        train.train(BODY, 42, steps=32, out=out, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo,
                    allow_untuned=True)
    doc = train.train(BODY, 42, steps=64, out=out, resume=True, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo,
                      allow_untuned=True, log=lambda *_: None)
    assert doc["num_timesteps"] == 64 and doc["sessions"][-1]["resumed_from"] == 32 and len(doc["sessions"]) == 2
    assert doc["sessions"][1]["env_seeds"] == [420100] and doc["sessions"][1]["env"] == s0["env"]
    assert train.checkpoints(rd)[-1] == 64 and json.loads((rd / "run.json").read_text())["status"] == "done"
    with pytest.raises(SystemExit, match="settings"):  # other settings: refused
        train.train(BODY, 42, steps=96, out=out, resume=True, n_envs=1, checkpoint_every=32, config=cfg, ppo=ppo,
                    allow_untuned=True)
    prov = plant.provenance()
    other = copy.deepcopy(prov)
    other["stage1_sources"]["sha256"] = "0" * 64
    monkeypatch.setattr(plant, "provenance", lambda: copy.deepcopy(other))
    with pytest.raises(SystemExit, match="Stage 1 sources"):  # other sources: refused unless allowed (flagged)
        train.train(BODY, 42, steps=96, out=out, resume=True, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo,
                    allow_untuned=True)
    doc = train.train(BODY, 42, steps=80, out=out, resume=True, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo,
                      allow_untuned=True, allow_source_change=True, log=lambda *_: None)
    assert doc["num_timesteps"] == 80 and doc["sessions"][-1]["source_change"] == dict(
        allowed=True, first_session=prov["stage1_sources"]["sha256"], this_session="0" * 64)
    monkeypatch.setattr(plant, "provenance", lambda: copy.deepcopy(prov))
    (out / BODY / "tune.json").write_text(json.dumps(_tune_doc({42: (True, 0, 0.0, 0.05, 0.05, 0.3)})))
    with pytest.raises(SystemExit, match="gains changed"):  # tune.json now selects other gains: refused
        train.train(BODY, 42, steps=96, out=out, resume=True, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo,
                    allow_untuned=True)


def test_g6bs_one_fix_changes_the_task_weights_and_stops_on_a_second_failure(tmp_path, monkeypatch):
    """FIX_RULE: one fix on resume (task weights only), recorded with the configs before it; G6b again at the fix's
    checkpoint + G6B_STEPS; a second failure stops the run, which then refuses to resume."""
    _need_run()
    out = _out(tmp_path)
    monkeypatch.setattr(train, "G6A_MEASURE_STEPS", 16)
    monkeypatch.setattr(train, "G6A_CHECK", dict(task="walk", seconds=0.2, episode_seed=7))
    ppo = dict(n_steps=16, batch_size=16, n_epochs=1)
    cfg = dict(tasks=("walk", "stand"), max_seconds=1.0)
    kw = dict(out=out, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo, allow_untuned=True, log=lambda *_: None)
    train.train(BODY, 42, steps=32, **kw)
    rd = train.run_dir_of(out, BODY, "R3", 42)
    with pytest.raises(SystemExit, match="only on resume"):
        train.train(BODY, 43, steps=32, fix=dict(task_weights=(1.0, 2.0)), **kw)
    with pytest.raises(SystemExit, match="task_weights"):
        train.train(BODY, 42, steps=64, resume=True, fix=dict(push_prob=0.0), **kw)
    seen = []

    def failing_monitor(model, vecnorm, body, mode, config):
        seen.append(config.task_weights)
        return dict(mean_survival_s=dict(policy=1.0, zero=2.0), policy_beats_zero_on=[], rows={}, summary={},
                    g6c_by_condition={})

    monkeypatch.setattr(train, "monitor", failing_monitor)
    monkeypatch.setattr(train, "G6B_STEPS", 16)
    doc = train.train(BODY, 42, steps=96, resume=True, fix=dict(task_weights=(1.0, 2.0)), fix_reason="test", **kw)
    assert doc["fix"]["from_checkpoint"] == 32 and doc["fix"]["overrides"] == {"task_weights": [1.0, 2.0]}
    assert doc["fix"]["reason"] == "test" and doc["fix"]["env_configs_before"][0]["task_weights"] is None
    assert doc["env_configs"][0]["task_weights"] == [1.0, 2.0] and doc["sessions"][-1]["fix"]["from_checkpoint"] == 32
    assert doc["g6"]["G6b"]["at_steps"] == 32  # the original G6b (G6B_STEPS patched to 16 for this session)
    after = doc["g6"]["G6b_after_fix"]
    assert after["at_steps"] == 48 and after["fix_from"] == 32 and not after["pass_"]
    assert seen and all(w == (1.0, 2.0) for w in seen)  # the monitor ran on the fixed configs
    assert doc["status"] == "stopped_g6b" and doc["stopped"]["at_steps"] == 48 and train.checkpoints(rd)[-1] == 48
    with pytest.raises(SystemExit, match="stopped"):
        train.train(BODY, 42, steps=96, resume=True, **kw)


def test_a_fixed_run_keeps_its_fix_on_later_resumes(tmp_path, monkeypatch):
    _need_run()
    out = _out(tmp_path)
    monkeypatch.setattr(train, "G6A_MEASURE_STEPS", 16)
    monkeypatch.setattr(train, "G6A_CHECK", dict(task="walk", seconds=0.2, episode_seed=7))
    ppo = dict(n_steps=16, batch_size=16, n_epochs=1)
    cfg = dict(tasks=("walk", "stand"), max_seconds=1.0)
    kw = dict(out=out, n_envs=1, checkpoint_every=16, config=cfg, ppo=ppo, allow_untuned=True, log=lambda *_: None)
    train.train(BODY, 42, steps=32, **kw)
    train.train(BODY, 42, steps=48, resume=True, fix=dict(task_weights=(1.0, 3.0)), **kw)
    with pytest.raises(SystemExit, match="one fix"):
        train.train(BODY, 42, steps=64, resume=True, fix=dict(task_weights=(1.0, 1.0)), **kw)
    doc = train.train(BODY, 42, steps=64, resume=True, **kw)
    assert doc["num_timesteps"] == 64 and doc["env_configs"][0]["task_weights"] == [1.0, 3.0]
    assert doc["sessions"][-1]["env"]["config"]["task_weights"] == [1.0, 3.0]


def test_two_ppo_seeds_and_a_resume_draw_disjoint_episode_seeds(tmp_path, monkeypatch):
    """SEED_RULE through SB3 itself (its constructor and PPO.load reseed the vec env with seed + i): every episode the
    envs draw in training, per run and session, from its own stream."""
    _need_run()
    out = _out(tmp_path)
    drawn = []
    orig = env_mod.Stage1Env.reset

    def reset(self, *, seed=None, options=None):
        r = orig(self, seed=seed, options=options)
        drawn.append((tag[0], self.episode_seed))
        return r

    monkeypatch.setattr(env_mod.Stage1Env, "reset", reset)
    tag = [None]
    ppo = dict(n_steps=16, batch_size=16, n_epochs=1)
    cfg = dict(tasks=("walk", "stand"), max_seconds=0.1)
    docs = {}
    for seed, steps, resume in ((42, 32, False), (43, 32, False), (42, 64, True)):
        tag[0] = (seed, resume)
        docs[tag[0]] = train.train(BODY, seed, steps=steps, out=out, resume=resume, n_envs=2, g6=False, config=cfg,
                                   ppo=ppo, allow_untuned=True, log=lambda *_: None)
    by = {}
    for t, ep in drawn:
        by.setdefault(t, set()).add(ep)
    assert set(by) == {(42, False), (43, False), (42, True)} and all(len(v) >= 4 for v in by.values())
    keys = list(by)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not by[a] & by[b], (a, b)
    for t, doc in docs.items():  # the record names every episode seed drawn
        s = doc["sessions"][-1]
        rec = {env_mod.EPISODE_STRIDE * st["reset_seed"] + k for st in s["episode_streams"] for k in range(st["episodes"])}
        assert rec == by[t], t
        assert s["env_seeds"] == [10_000 * t[0] + 100 * (1 if t[1] else 0) + i for i in range(2)]


def test_subproc_envs_report_their_processes_and_cpu(tmp_path, monkeypatch):
    _need_run()
    out = _out(tmp_path)
    monkeypatch.setattr(train, "G6A_MEASURE_STEPS", 16)
    doc = train.train(BODY, 42, steps=32, out=out, n_envs=2, vec="subproc", g6=False,
                      config=dict(tasks=("stand",), max_seconds=0.2), ppo=dict(n_steps=16, batch_size=16, n_epochs=1),
                      allow_untuned=True, log=lambda *_: None)
    rate = doc["throughput"][0]
    assert rate["first_rollouts"] and rate["env_processes"] == 2 and rate["cpu_s_env_workers"] > 0
    assert rate["cpu_s"] == pytest.approx(rate["cpu_s_main"] + rate["cpu_s_env_workers"])
    assert doc["sessions"][0]["floors"] == ["hard", "hard"]


def test_cli_trains_and_records(tmp_path, monkeypatch):
    calls = {}

    def fake_train(body, seed, **kw):
        calls.update(kw, body=body, seed=seed)
        return dict(num_timesteps=10, status="done", g6={"G6a_zero_residual": dict(identical=True)})

    monkeypatch.setattr(train, "train", fake_train)
    from designlab.stage1.__main__ import main

    assert main(["train", "--body", BODY, "--seed", "43", "--mode", "R4", "--steps", "10", "--out", str(tmp_path)]) == 0
    assert calls["body"] == BODY and calls["seed"] == 43 and calls["mode"] == "R4" and calls["steps"] == 10
    assert calls["allow_untuned"] is False and calls["log_std_init"] is None and calls["vec"] == "dummy"
    assert main(["train", "--body", BODY, "--seed", "42", "--allow-untuned", "--tune-json", "/t.json", "--out",
                 str(tmp_path)]) == 0
    assert calls["allow_untuned"] is True and calls["config"] == dict(tune_json="/t.json")
    assert main(["judge", "--mode", "R3", "--out", str(tmp_path)]) == 0
    assert tune.RUN.name == "2026-11-08"
