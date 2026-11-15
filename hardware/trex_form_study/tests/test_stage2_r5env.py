"""R5's and R6's env (``designlab/stage2/r5env.py``; round C design D1, D6, D8, D10, D16, section 3.2; rules E1-E5,
T3, W4, J1-J5).

Every episode here runs on the S1 view (Stage 1's own run directory; ``stage2/x`` for the 16-joint body): no
Stage 2 reference set is touched (rule X1).  Digests are ``env._state_digest`` after every physics step.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import env, plant, tune  # noqa: E402
from designlab.stage2 import r5core, r5env, r5lib, s2io  # noqa: E402

BODY, BODY_X = "75-M-H-10", "75-M-H-10-X"
RATE = 1.05
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)
SLOW = os.environ.get("STAGE2_SLOW") == "1"


def _have(body: str) -> bool:
    try:
        return ((r5lib.s1_root(body) / body / "transitions" / "stop.npz").exists()
                and all(Path(f).exists() for f in plant.model_files(body)))
    except Exception:  # noqa: BLE001  (no model record: the same skip)
        return False


needs_run = pytest.mark.skipif(not (_have(BODY) and ONE_THREAD), reason=(
    "needs the Stage 1 arrays and the round-two models (tools/restore_inputs.py) and one BLAS thread"))
needs_x = pytest.mark.skipif(not (_have(BODY_X) and ONE_THREAD), reason=(
    "needs the expressive-head body's models and arrays (tools/restore_inputs.py --group x) and one BLAS thread"))
BODIES = [BODY, pytest.param(BODY_X, marks=needs_x)]


def cfg_of(body: str = BODY, **over) -> r5env.R5Config:
    return r5env.R5Config(body=body, set="S1", **over)


def nominal(body: str = BODY, **over) -> r5env.R5Config:
    """No randomisation, no push, no head event, every start as the test gives it."""
    return cfg_of(body, **{**dict(randomize=False, push_prob=0.0, head_events=False, mid_start=0.0), **over})


@pytest.fixture(scope="module")
def e0():
    """A nominal 14-joint env on hard floor, shared (every test resets it: reset is history-free)."""
    e = r5env.R5Env(nominal())
    yield e
    e.close()


def play(e, actions, **reset) -> dict:
    """One episode: reset, then ``actions`` until they run out or it ends; every physics step's digest, every
    observation, reward and what the reset drew."""
    rows: list = []
    e._step_hook = lambda p: rows.append(env._state_digest(p.d))
    try:
        obs, info = e.reset(**reset)
        out = dict(digests=rows, obs=[obs], rewards=[], ended=None, task=info["task"], start=info["start"],
                   t_start=info["t_start"], factors=dict(info["factors"]), push=info["push"], head=info["head"],
                   seeds=info["seeds"], n_steps=info["n_steps"])
        for k, a in enumerate(actions):
            obs, r, term, trunc, info = e.step(a)
            out["obs"].append(obs)
            out["rewards"].append(r)
            if term or trunc:
                out["ended"] = (k, term, trunc, info["fall_cause"])
                break
    finally:
        e._step_hook = None
    return out


def same(a: dict, b: dict) -> None:
    assert len(a["digests"]) == len(b["digests"]) > 0
    first = next((i for i, (x, y) in enumerate(zip(a["digests"], b["digests"], strict=True)) if x != y), None)
    assert first is None, f"physics step {first} differs"
    assert all(np.array_equal(x, y) for x, y in zip(a["obs"], b["obs"], strict=True))
    assert a["rewards"] == b["rewards"]
    for k in ("ended", "task", "start", "t_start", "factors", "push", "head", "seeds", "n_steps"):
        assert a[k] == b[k], k


def noise(n: int, steps: int, sigma: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [(sigma * rng.standard_normal(n)).astype(np.float32) for _ in range(steps)]


# ------------------------------------------------------------------------------------------------
# Zero is zero
# ------------------------------------------------------------------------------------------------
@needs_run
def test_zero_action_leaves_residual_none_and_rate_one(e0):
    e0.reset(seed=3, options=dict(task="walk", start="stance"))
    zero = np.zeros(15, np.float32)
    for _ in range(12):
        _, _, term, _, info = e0.step(zero)
        assert e0.tk.residual is None and e0.tk.rate_factor == 1.0 and type(e0.tk.rate_factor) is float
        assert info["clock_factor"] == 1.0 and info["reward_terms"]["action_rate"] == 0.0 and not term
    assert e0.core.clock_stats() == dict(steps=12, mean=1.0, min=1.0, reference_s=pytest.approx(12 * 0.02 * RATE))
    assert isinstance(e0.tk, r5env.R5Tracker) and (e0.tk.mode, e0.tk.terms, e0.tk.ankle_cap) == ("R2", frozenset(), ())
    assert e0.tk.rate == RATE and e0.tk.model is e0.nominal_model


@needs_run
def test_any_offset_breaks_the_digest(e0):
    """A zero action is not "almost" zero: one action entry at 1e-6 moves the first env step's physics."""
    opts = dict(task="walk", start="stance", episode_seed=11)
    zero = play(e0, [np.zeros(15, np.float32)], seed=0, options=opts)
    again = play(e0, [np.zeros(15, np.float32)], seed=0, options=opts)
    same(zero, again)
    for entry in (4, 14):  # a leg joint's offset; the clock
        a = np.zeros(15, np.float32)
        a[entry] = 1e-6
        moved = play(e0, [a], seed=0, options=opts)
        assert len(moved["digests"]) == 10 and moved["digests"] != zero["digests"]
        assert moved["digests"][-1] != zero["digests"][-1]
        assert (e0.tk.residual is None) == (entry == 14) and (e0.tk.rate_factor == 1.0) == (entry != 14)


# ------------------------------------------------------------------------------------------------
# Reset is history-free (design D16)
# ------------------------------------------------------------------------------------------------
@needs_run
@pytest.mark.parametrize("mode,body,floor", [("R5", BODY, "hard"), ("R5", BODY, "robot_floor_carpet"),
                                             pytest.param("R5", BODY_X, "hard", marks=needs_x), ("R6", BODY, "hard")])
def test_reset_is_history_free(mode, body, floor):
    """An env after six randomised episodes (noise actions, drawn tasks, starts, factors, pushes and head events)
    against a fresh env: after ``reset(seed)`` and the same actions, every physics step's digest, every
    observation and reward and everything the reset drew are equal: from the stance, from mid-task and with the
    start drawn."""
    cfg = cfg_of(body, floor=floor, mode=mode)
    used, fresh = r5env.make(mode, cfg), r5env.make(mode, cfg)
    n = used.layout["action_dim"]
    assert n == (17 if body == BODY_X else 15)
    seen = []
    used.reset(seed=991)
    for ep in range(6):
        for a in noise(n, 40, 0.3, ep):
            _, _, term, trunc, info = used.step(a)
            if term or trunc:
                break
        seen.append((info["task"], info["start"], info["factors"]["latency_steps"], info["push"] is not None))
        used.reset()
    assert len({s[2] for s in seen}) > 2 and any(f["joint_damping"] != 1.0 for f in [used.factors])
    acts = noise(n, 60, 0.2, 77)
    for seed, opts in ((4242, dict(start="stance")), (4243, dict(task="brisk", start="mid")), (4244, {}), (4245, {})):
        a, b = play(used, acts, seed=seed, options=opts), play(fresh, acts, seed=seed, options=opts)
        same(a, b)
        assert len(a["digests"]) >= 10
        if "start" in opts:
            assert a["start"] == opts["start"]
    # the stream goes on alike without a seed (env.EPISODE_SEED_RULE: the next episode of the same reset)
    same(play(used, acts[:20]), play(fresh, acts[:20]))
    assert used.episode_stream() == fresh.episode_stream() == dict(reset_seed=4245, episodes=2,
                                                                  first_episode_seed=4245 * 1_000_000)
    used.close()
    fresh.close()


@needs_run
def test_trackers_are_built_nominal():
    e = r5env.R5Env(cfg_of())
    p, m = e.p, e.p.m
    damp0, fric0 = e.nominal_model.dof_damping[p.dofs].copy(), e.nominal_model.dof_frictionloss[p.dofs].copy()
    assert sorted(e.trackers) == ["brisk", "stand", "stop", "walk"] and fric0.any() and damp0.any()
    e.reset(seed=5)
    for ep in range(4):
        for a in noise(15, 15, 0.3, ep):
            if any(e.step(a)[2:4]):
                break
        e.reset()
    for tk in e.trackers.values():
        assert np.array_equal(tk.damp, damp0) and np.array_equal(tk.fric, fric0) and tk.model is e.nominal_model
        assert tk.mass == float(e.nominal_model.body_subtreemass[p.trunk])
    # a tracker for a plan given by options, first used right after a randomised episode
    e.reset(options=dict(factors=dict(joint_damping=1.5, joint_friction=0.5, mass=1.1, kp=0.7, friction=0.5)))
    assert np.array_equal(m.dof_damping[p.dofs], 1.5 * damp0)
    assert m.body_mass[p.trunk] > e.nominal_model.body_mass[p.trunk]
    plan = [("transition", "start"), ("ref", "nominal", 3)]
    e.reset(options=dict(plan=plan, seconds=1.0, factors=dict(joint_damping=0.6)))
    script, tk = e._plans[tuple(plan)]
    assert e.tk is tk and e.script is script and e.task == "plan"
    assert np.array_equal(tk.damp, damp0) and np.array_equal(tk.fric, fric0)
    assert np.array_equal(m.dof_damping[p.dofs], 0.6 * damp0)  # the episode's own factor, applied after the build
    fresh = plant.build(BODY, "hard")
    ref = r5lib.Library(fresh, BODY, "S1", RATE).from_plan(plan)
    assert e.lib.omega == ref.track.omega == script.track.omega
    for a, b in zip(script.track.segments, ref.track.segments, strict=True):  # the track read the nominal masses
        for key in ("dcm_rel", "comv_rel", "zmp_rel", "yaw"):
            assert np.array_equal(a.info[key], b.info[key]), key
    e.reset(options=dict(plan=[list(x) for x in plan], seconds=1.0))
    assert e.tk is tk and len(e._plans) == 1  # kept by the plan
    # a tracker built on a randomised model is refused at the reset that would use it
    e.reset(options=dict(task="walk", factors=dict(joint_damping=1.4)))
    e.trackers["walk"] = e._tracker(e.scripts["walk"])
    with pytest.raises(RuntimeError, match="not the nominal"):
        e.reset(options=dict(task="walk"))
    e.close()


@needs_run
def test_factors_are_applied_and_restored(e0):
    p, m, nom = e0.p, e0.p.m, e0.nominal_model
    assert env.model_differences(m, nom) == []
    f = dict(friction=0.5, mass=1.1, kp=0.8, latency_steps=4, noise_scale=1.3, joint_damping=1.5, joint_friction=0.5)
    _, info = e0.reset(seed=1, options=dict(task="walk", factors=f))
    assert info["factors"] == f and e0.tk.noise.scale == 1.3
    assert np.array_equal(m.dof_damping[p.dofs], 1.5 * nom.dof_damping[p.dofs])
    assert np.array_equal(m.dof_frictionloss[p.dofs], 0.5 * nom.dof_frictionloss[p.dofs])
    assert np.array_equal(m.dof_damping[:6], nom.dof_damping[:6])  # the hinges only
    assert np.array_equal(m.actuator_gainprm[:, 0], 0.8 * nom.actuator_gainprm[:, 0])
    assert np.array_equal(m.actuator_biasprm[:, 1], 0.8 * nom.actuator_biasprm[:, 1])
    assert m.body_mass[p.trunk] == 1.1 * nom.body_mass[p.trunk]
    np.testing.assert_array_equal(m.body_inertia[p.trunk], 1.1 * nom.body_inertia[p.trunk])
    colliding = (m.geom_contype | m.geom_conaffinity) > 0
    assert np.all(m.geom_friction[colliding, 0] == 0.5)
    assert e0.weight_true > e0.weight and e0.core.weight_true == e0.weight_true and e0.core.weight == e0.weight
    changed = env.model_differences(m, nom)
    assert {"dof_damping", "dof_frictionloss", "body_mass", "geom_friction", "actuator_gainprm"} <= set(changed)
    assert "command" in p.__dict__  # the delay is the bar's wrapper on the plant
    for _ in range(3):
        e0.step(np.zeros(15, np.float32))
    # the next nominal episode's model is the nominal one bit for bit, damping and friction-loss included
    e0.reset(options=dict(task="walk"))
    assert env.model_differences(m, nom) == [] and e0.weight_true == e0.weight
    assert "command" not in p.__dict__ and "reset" not in p.__dict__
    for tk in e0.trackers.values():  # the controller's copies never moved
        assert np.array_equal(tk.damp, nom.dof_damping[p.dofs])
        assert np.array_equal(tk.fric, nom.dof_frictionloss[p.dofs])
    # training's ranges (rules J1, J2): a drawn episode's factors lie in them, in the env's own order of draws
    e = e0.cfg
    assert (e.latency_steps, e.kp, e.friction, e.mass) == ((0, 12), (0.6, 1.2), (0.4, 1.2), (0.9, 1.1))
    assert (e.joint_damping, e.joint_friction, e.noise_scale) == ((0.5, 1.5), (0.5, 1.5), (0.5, 1.5))
    rng = np.random.default_rng(8)
    drawn = dataclasses.replace(e, randomize=True)
    probe = r5env.R5Env.__new__(r5env.R5Env)
    probe.cfg = drawn
    got = probe._draw_factors(rng)
    rng = np.random.default_rng(8)
    want = dict(friction=float(rng.uniform(0.4, 1.2)), mass=float(rng.uniform(0.9, 1.1)),
                kp=float(rng.uniform(0.6, 1.2)), latency_steps=int(rng.integers(0, 13)),
                noise_scale=float(rng.uniform(0.5, 1.5)),
                joint_damping=float(rng.uniform(0.5, 1.5)), joint_friction=float(rng.uniform(0.5, 1.5)))
    assert got == want and list(got) == list(r5env.FACTORS)


# ------------------------------------------------------------------------------------------------
# Mid-task starts (design D6, rule T3)
# ------------------------------------------------------------------------------------------------
@needs_run
def test_mid_start_state_is_the_files_frame(e0):
    s = e0.scripts["brisk"]
    seeds = dict(start=123, noise=456)
    for t in (8.31, 14.02, 20.9):
        t_frame, q, qd = s.start_state(t)
        _, info = e0.reset(options=dict(task="brisk", start=t, seeds=seeds))
        p = e0.p
        r = np.random.default_rng(123)
        q[p.qadr] += r.uniform(-0.01, 0.01, 14)
        qd = qd * RATE
        qd[:3] += r.uniform(-0.02, 0.02, 3)
        assert np.array_equal(p.d.qpos, q) and np.array_equal(p.d.qvel, qd) and p.d.time == 0.0
        assert info["start"] == "mid" and info["t_start"] == t_frame == e0.tk.tau_ref and t - 0.008 < t_frame <= t
        assert info["seconds"] == pytest.approx(min((s.t_end - t_frame) / RATE, 30.0))
        est = e0.tk.est  # reset from the known state: Stage 1's Estimator.reset
        np.testing.assert_array_equal(est.q, q[p.qadr])
        np.testing.assert_array_equal(est.base, q[:3])
        # the track got its own first frame back
        assert np.array_equal(s.track.q0, s.start_state(0.0)[1]) and e0.tk.start == "frame"
        obs = e0._observation()
        lay = e0.layout["blocks"]
        assert not obs[lay["prev_actions"][0] : lay["prev_actions"][1]].any()
        assert abs(obs[lay["heading"][0]]) < 1e-3 and abs(obs[lay["heading_error"][0]]) < 1e-3
        np.testing.assert_allclose(obs[lay["joint_error"][0] : lay["joint_error"][1]], 0.0, atol=0.0101)
    with pytest.raises(ValueError, match="no mid-task start"):
        e0.reset(options=dict(task="stand", start="mid"))
    with pytest.raises(ValueError, match="hold"):
        e0.reset(options=dict(task="stop", start=e0.scripts["stop"].t_end - 1.0))
    with pytest.raises(ValueError, match="start"):
        e0.reset(options=dict(task="walk", start="frame"))


@needs_run
def test_mid_start_queue_holds_the_references_commands(e0):
    """k = 3: the servos get the commands of an undelayed replay, three physics steps late, from the first step on:
    the queue was refilled with what the zero action would have given before the start."""
    t, k = 9.6, 3
    seeds = dict(start=5, noise=6)
    ctrl: list = []

    def run(latency: int) -> list:
        ctrl.clear()
        e0.reset(options=dict(task="brisk", start=t, seeds=seeds, factors=dict(latency_steps=latency)))
        at_reset = e0.p.d.ctrl.copy()
        e0._step_hook = lambda p: ctrl.append(p.d.ctrl.copy())
        try:
            e0.step(np.zeros(15, np.float32))
        finally:
            e0._step_hook = None
        return [at_reset] + list(ctrl)

    late, now = run(k), run(0)
    assert len(late) == len(now) == 11
    for j in range(k, 10):  # c(j - 3) reaches the servos at step j: bit for bit, the same code computed it
        assert np.array_equal(late[1 + j], now[1 + j - k])
    assert np.array_equal(late[0], late[1])  # the oldest of the queue, set at the reset
    assert not np.array_equal(late[0], e0.p.ctrl_home)
    assert np.array_equal(now[0], e0.p.ctrl_home)  # (no delay: Plant.set_state's targets until the first command)
    # the three before the start, against an undelayed tracker played from three steps earlier
    e0.close()
    t_frame, q, qd = e0.scripts["brisk"].start_state(t)
    tk = e0._tracker(e0.scripts["brisk"])
    tk.reset_at(t_frame - k * 0.002 * RATE, q, qd * RATE, 6)
    for j in range(k):
        tk.step()
        np.testing.assert_allclose(e0.p.d.ctrl, late[1 + j], rtol=0, atol=1e-12)
    tk.step()
    np.testing.assert_allclose(e0.p.d.ctrl, now[1], rtol=0, atol=1e-12)
    # and R5Tracker.zero_command is what the tracker's own step commands
    type(e0.p).command(e0.p, *tk.zero_command(tk.tau_ref))
    want = e0.p.d.ctrl.copy()
    tk.step()
    assert np.array_equal(e0.p.d.ctrl, want)


@needs_run
def test_mid_start_never_lands_in_a_hold():
    e = r5env.R5Env(nominal(tasks=("stop", "stand", "walk"), mid_start=1.0, interference="off"))
    count = {"stop": 0, "stand": 0, "walk": 0}
    e.reset(seed=17)
    for _ in range(150):
        _, info = e.reset()
        task, s = info["task"], e.script
        count[task] += 1
        if task == "stand":  # rule T3: every task but the stand
            assert info["start"] == "stance" and info["t_start"] == 0.0 and info["seconds"] == 10.0
            continue
        assert info["start"] == "mid" and s.kind(info["t_start"]) != "hold"
        spans = s.mid_spans()
        assert any(a - 0.008 <= info["t_start"] <= b for a, b in spans)
        assert spans[0][0] - 0.008 <= info["t_start"] <= s.t_end - 2.0 * RATE
        assert 2.0 - 1e-9 <= info["seconds"] <= 30.0 and info["n_steps"] == round(info["seconds"] / 0.02)
    assert min(count.values()) > 20
    # with the rule's one half, about half of the walk's episodes start mid-task; none with zero
    e2 = r5env.R5Env(nominal(tasks=("walk",), mid_start=0.5, interference="off"))
    e2.reset(seed=3)
    mids = sum(1 for _ in range(200) if e2.reset()[1]["start"] == "mid")
    assert 75 <= mids <= 125
    e2.close()
    e.close()


@needs_run
@pytest.mark.skipif(not SLOW, reason="slow: set STAGE2_SLOW=1")
def test_mid_start_walks_on(e0):
    """Scratch C3 through the real env: the zero action from three frames of the walk, at 0 and at 10 ms, stays up
    for 4 s on hard floor."""
    for t in (6.2, 8.9, 11.4):
        for k in (0, 5):
            _, info = e0.reset(options=dict(task="walk", start=t, seeds=dict(start=1, noise=2),
                                            factors=dict(latency_steps=k)))
            assert info["start"] == "mid" and info["seconds"] > 4.0
            for step in range(200):
                _, _, term, trunc, info = e0.step(np.zeros(15, np.float32))
                assert not term, (t, k, step, info["fall_cause"])
            assert e0.p.tilt_from_home() < math.radians(3.0)


# ------------------------------------------------------------------------------------------------
# Episodes: lengths, seeds, pushes, the clock past the plan's end, interference
# ------------------------------------------------------------------------------------------------
@needs_run
def test_lengths_seeds_and_the_plan_by_options(e0):
    _, info = e0.reset(seed=7, options=dict(task="stand"))
    assert (info["seconds"], info["n_steps"], info["start"]) == (10.0, 500, "stance")
    assert info["episode_seed"] == 7_000_000 and e0.reset()[1]["episode_seed"] == 7_000_001
    assert info["seeds"] == dict(start=int(np.random.SeedSequence([7_000_000, 0]).generate_state(1)[0]),
                                 noise=int(np.random.SeedSequence([7_000_000, 1]).generate_state(1)[0]))
    _, info = e0.reset(options=dict(task="walk"))
    assert info["seconds"] == pytest.approx(e0.scripts["walk"].t_end / RATE)
    assert info["n_steps"] == round(info["seconds"] / 0.02)
    assert e0.reset(options=dict(task="walk", episode_seed=99))[1]["episode_seed"] == 99
    # a plan given by options with its seconds: ceil(round(seconds / dt) / 10) env steps (the bar's lengths)
    plan = [("transition", "start"), ("ref", "nominal", 2)]
    for seconds, n in ((1.0, 50), (1.013, 51), (0.999, 50), (1.0021, 51)):
        _, info = e0.reset(options=dict(plan=plan, seconds=seconds, seeds=dict(start=1, noise=2)))
        assert info["n_steps"] == n == math.ceil(round(seconds / 0.002) / 10) and info["task"] == "plan"
        assert info["seeds"] == dict(start=1, noise=2) and info["episode_seed"] is None
    steps = 0
    while True:
        _, _, term, trunc, _ = e0.step(np.zeros(15, np.float32))
        steps += 1
        if term or trunc:
            break
    assert (steps, term, trunc) == (51, False, True)
    short = dataclasses.replace(e0.cfg, max_seconds=3.0)
    assert short.max_seconds == 3.0 and e0.cfg.max_seconds == 30.0


@needs_run
def test_pushes_are_drawn_by_rule_j3():
    e = r5env.R5Env(nominal(tasks=("walk", "stand"), push_prob=1.0, interference="off"))
    bound, mass = env.PUSH_BOUND_M_S[BODY], float(e.p.m.body_subtreemass[e.p.trunk])
    assert e.push_bound == bound == 0.589 and e.cfg.push_walking == (0.0, 0.5) and e.cfg.push_standing == (0.0, 1.0)
    e.reset(seed=1)
    us = {"walk": [], "stand": [], "start": []}
    for _ in range(120):
        _, info = e.reset()
        t0, dur, fx, fy = info["push"]
        assert dur == 0.1 and 2.0 <= t0 <= info["seconds"] - 2.0
        u = math.hypot(fx, fy) * dur / mass / bound
        walking = e.script.walking(info["t_start"] + t0 * RATE)
        us["walk" if walking else ("stand" if info["task"] == "stand" else "start")].append(u)
        assert u <= (0.5 if walking else 1.0) + 1e-9
    assert len(us["walk"]) > 20 and len(us["stand"]) > 20 and max(us["stand"]) > 0.6 and max(us["walk"]) > 0.3
    # an episode of 4 s or less gets none; neither does push_prob 0
    assert e.reset(options=dict(task="walk", start=e.scripts["walk"].t_end - 3.0 * RATE))[1]["push"] is None
    e.close()
    with pytest.raises(KeyError):
        env.PUSH_BOUND_M_S["no-such-body"]  # no default bound (the env's build reads it by key)


@needs_run
def test_env_steps_past_the_plans_end_at_the_fastest_clock(e0):
    """A clock 25 % fast passes the end of an episode sized for the commanded clock: the tables continue."""
    fast = np.zeros(15, np.float32)
    fast[-1] = 1.0
    for task in ("walk", "stop"):
        s = e0.scripts[task]
        tail = 2.4 * RATE if task == "walk" else None
        if tail is None:  # the stop task ends in a hold: start in its last move, as late as rule T3 allows
            t = s.mid_spans()[-1][1] - 0.01
        else:
            t = s.t_end - tail
        _, info = e0.reset(options=dict(task=task, start=t, seeds=dict(start=3, noise=4)))
        ended = None
        for k in range(info["n_steps"]):
            obs, r, term, trunc, info = e0.step(fast)
            assert np.all(np.isfinite(obs)) and all(0.0 <= v <= 1.0 for v in info["reward_terms"].values())
            if term or trunc:
                ended = (k + 1, term, trunc)
                break
        assert ended is not None and ended[2] and not ended[1], (task, ended, info["fall_cause"])
        assert info["t_ref"] > s.t_end + 0.2 and info["clock_factor"] > 1.24
        stats = e0.core.clock_stats()
        assert 1.15 < stats["mean"] <= 1.25 and stats["reference_s"] == pytest.approx(info["t_ref"] - info["t_start"])


class _Gaps:
    cap = 0.025
    moving = np.array([False, True])

    def __init__(self):
        self.d = np.array([0.004, 0.020])

    def at(self, qpos):
        return self.d.copy()


@needs_run
def test_interference_terminates_at_two_step_ends_in_a_row():
    e = r5env.R5Env(nominal(tasks=("stand",)))
    assert e.lg is not None and e.cfg.interference == "terminate" and e.describe()["fixed_pairs"] == e.fixed_pairs
    gaps = e.core.lg = _Gaps()
    zero = np.zeros(15, np.float32)
    e.reset(options=dict(task="stand"))
    gaps.d[:] = (0.004, 0.001)  # tight: penalised, never terminated
    _, _, term, _, info = e.step(zero)
    assert not term and info["interference"]["penalty"] == pytest.approx(2.0 / 3.0)
    assert info["reward_terms"]["interference"] == pytest.approx(2.0 / 3.0) and not info["interference"]["below_zero"]
    gaps.d[:] = (0.004, -0.001)  # one pose in contact: penalised, not terminated (rule W4)
    _, _, term, _, info = e.step(zero)
    assert not term and info["interference"] == dict(penalty=1.0, below_zero=True, consecutive=1, min_gap_m=-0.001,
                                                     first_env_step=None)
    gaps.d[:] = (0.004, 0.02)
    assert e.step(zero)[4]["interference"]["consecutive"] == 0
    gaps.d[:] = (-0.0005, 0.02)  # a fixed pair below zero: no penalty, but it counts for the termination
    assert e.step(zero)[2] is False
    _, _, term, trunc, info = e.step(zero)
    assert term and not trunc and info["fall_cause"] == "leg interference"
    assert info["interference"]["first_env_step"] == 5
    assert info["reward_terms"]["interference"] == 0.0
    # the gates switch the termination off for an episode and read when it would have fired
    e.reset(options=dict(task="stand", terminate_on_interference=False))
    for k in range(4):
        _, _, term, _, info = e.step(zero)
        assert not term and info["fall_cause"] is None
    assert e.interference_step == 2 and info["interference"]["consecutive"] == 4
    e.close()
    # "reward": the penalty without the termination; "off": no reading at all
    e = r5env.R5Env(nominal(tasks=("stand",), interference="reward"))
    e.core.lg = gaps
    e.reset()
    assert [e.step(zero)[2] for _ in range(3)] == [False] * 3 and e.interference_step == 2
    e.close()
    e = r5env.R5Env(nominal(tasks=("stand",), interference="off"))
    assert e.lg is None and e.fixed_pairs == []
    e.reset()
    assert e.step(zero)[4]["interference"] == dict(penalty=0.0, below_zero=False, consecutive=0, min_gap_m=None,
                                                   first_env_step=None)
    e.close()


def test_importing_the_env_loads_neither_torch_nor_sb3():
    """A worker process hosts envs without the trainer's libraries (design D13): about 245 MB less each."""
    code = ("import sys; from designlab.stage2 import r5env, r5core, r5lib; "
            "print(*(int(m in sys.modules) for m in ('torch', 'stable_baselines3', 'designlab.stage2.bar')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": ".", "MUJOCO_GL": "disable"})
    assert out.stdout.split() == ["0", "0", "0"], out.stdout + out.stderr


# ------------------------------------------------------------------------------------------------
# What the env says of itself
# ------------------------------------------------------------------------------------------------
@needs_run
def test_describe_holds_the_rules_and_no_path(e0):
    doc = e0.describe()
    text = json.dumps(s2io.plain(doc))
    assert str(ROOT) not in text and "/home" not in text and "/tmp" not in text and "stage1/runs" not in text
    assert doc["mode"] == "R5" and doc["config"] == e0.cfg.describe() and doc["config"]["set"] == "S1"
    assert doc["reward"] == r5core.REWARD_WEIGHTS
    assert doc["config"]["reward"] == [list(x) for x in r5core.REWARD_WEIGHTS_ITEMS]
    scales = {k: (list(v) if isinstance(v, tuple) else v) for k, v in r5core.REWARD_SCALES.items()}
    assert doc["reward_scales"] == scales
    assert doc["reward_rule"] == r5core.REWARD_RULE and doc["command_rule"] == r5lib.COMMAND_RULE
    assert doc["layout"] == r5core.layout(e0.p.joints) and doc["head_rule"] is None and doc["set_freeze"] is None
    assert list(doc["tasks"]) == ["stand", "walk", "brisk", "stop"] and doc["absent"] == {}
    assert [doc["tasks"][t]["weight"] for t in doc["tasks"]] == [1.0, 2.0, 2.0, 2.0]
    assert doc["inputs"] == e0.lib.inputs and doc["push_bound_m_s"] == 0.589 and doc["interference"] == "terminate"
    assert doc["core"]["action"] == dict(leg_rad=0.35, other_rad=0.15, clock=0.25, scale=[0.15] * 2 + [0.35] * 12)
    assert doc["core"]["filter_hz"] == 10.0 and doc["core"]["span"] is None
    for key in ("reset_order", "step_order", "draw_order", "mid_start_rule", "push_rule", "factor_rule"):
        assert isinstance(doc[key], str) and len(doc[key]) > 100
    # a config crosses a process boundary as its description
    back = r5env.config_from(json.loads(json.dumps(e0.cfg.describe())))
    assert back == e0.cfg and isinstance(back.reward, tuple) and isinstance(back.reward[0], tuple)
    assert e0.process_cpu()[0] == os.getpid()
    with pytest.raises(ValueError, match="mode"):
        r5env.R6Env(e0.cfg)
    with pytest.raises(ValueError, match="pile"):
        r5env.R5Env(nominal(floor="low"))
    with pytest.raises(ValueError, match="start noise"):
        r5env.R5Env(nominal(start_joint_noise=0.0))
    with pytest.raises(ValueError, match="not available"):
        r5env.R5Env(nominal(tasks=("walk_v0.4",)))
    assert (e0.cfg.start_joint_noise, e0.cfg.start_root_velocity_noise) == (tune.START_JOINT_NOISE_RAD,
                                                                          tune.START_ROOT_VELOCITY_NOISE_M_S)


@needs_run
def test_task_weights_and_env_configs(e0):
    np.testing.assert_allclose(e0.task_p, np.array([1, 2, 2, 2]) / 7.0)
    try:
        e0.set_task_weights(dict(walk=4.0))  # rule M4's fix doubles the named tasks' weights
        np.testing.assert_allclose(e0.task_p, np.array([1, 4, 2, 2]) / 9.0)
        assert e0.describe()["tasks"]["walk"]["weight"] == 4.0
        e0.set_task_weights([0.0, 0.0, 1.0, 0.0])
        e0.reset(seed=2)
        assert {e0.reset()[1]["task"] for _ in range(8)} == {"brisk"}
        for bad in (dict(run=1.0), [1.0, 2.0], [0.0] * 4, [1.0, -1.0, 1.0, 1.0]):
            with pytest.raises(ValueError):
                e0.set_task_weights(bad)
    finally:
        e0.set_task_weights([1.0, 2.0, 2.0, 2.0])
    cfgs = r5env.env_configs("R5", BODY, "S1")
    assert [c.floor for c in cfgs] == ["hard", "hard", "hard", "robot_floor_carpet"] * 2  # design D13, rule J5
    c = cfgs[0]
    assert (c.mode, c.set, c.set_freeze, c.randomize, c.mid_start, c.push_prob) == ("R5", "S1", None, True, 0.5, 0.5)
    assert (c.latency_steps, c.interference, c.heading_input) == ((0, 12), "terminate", True)
    assert (c.filter_hz, c.rate, c.gaze_prob, c.jaw_prob) == (10.0, 1.05, 0.5, 0.25)
    assert (c.action_leg_rad, c.action_other_rad, c.action_rate) == (0.35, 0.15, 0.25)
    assert dict(c.reward) == r5core.REWARD_WEIGHTS and c.max_seconds == 30.0 and c.tasks == ()
    r6 = r5env.env_configs("R6", BODY, "S1", n_envs=3, overrides=dict(floor="hard", mid_start=0.0))
    assert [(x.mode, x.floor, x.mid_start) for x in r6] == [("R6", "hard", 0.0)] * 3
    with pytest.raises(ValueError, match="mode"):
        r5env.env_configs("R3", BODY, "S1")
    assert r5env.FLOOR_SPREAD == ("hard", "hard", "hard", "robot_floor_carpet")


# ------------------------------------------------------------------------------------------------
# R6
# ------------------------------------------------------------------------------------------------
@needs_run
@pytest.mark.parametrize("body", BODIES)
def test_r6_steps_on_14_and_16_joints(body):
    e = r5env.make("R6", cfg_of(body, mode="R6", interference="off"), seed=4)
    nj = e.nj
    assert isinstance(e, r5env.R6Env) and all(isinstance(tk, r5core.R6Tracker) for tk in e.trackers.values())
    assert e.observation_space.shape == ((130,) if nj == 14 else (146,)) and e.action_space.shape == (nj + 1,)
    np.testing.assert_array_equal(e.span, r5core.policy_span(e.p, [e.scripts[t].track for t in e.tasks]))
    doc = e.describe()
    assert doc["mode"] == "R6" and doc["core"]["span"] == [float(x) for x in e.span]
    assert doc["core"]["span_rule"] == env.POLICY_ONLY_SPAN_RULE and np.all(e.span >= 0.10)
    lo, hi = e.core.lo, e.core.hi
    rng = np.random.default_rng(1)
    steps = 0
    for _ in range(3):
        obs, info = e.reset()
        for _ in range(15):
            a = rng.uniform(-1.5, 1.5, nj + 1).astype(np.float32)
            obs, r, term, trunc, info = e.step(a)
            steps += 1
            assert np.all(e.tk.q_target >= lo) and np.all(e.tk.q_target <= hi)  # targets inside the ranges
            assert 0.75 <= e.tk.rate_factor <= 1.25 and np.all(np.isfinite(obs)) and math.isfinite(r)
            if term or trunc:
                break
    assert steps > 20 and e.tk.residual is None
    # a zero action holds the settled home pose (R6 has no reference controller)
    e.reset(options=dict(task="walk", start="stance", factors=dict(r5env.NOMINAL_FACTORS)))
    e.step(np.zeros(nj + 1, np.float32))
    np.testing.assert_array_equal(e.tk.q_target, e.p.q_settled[e.p.qadr])
    np.testing.assert_allclose(e.p.d.ctrl, e.p.G @ e.p.q_settled[e.p.qadr], atol=1e-12)
    e.close()


@needs_run
def test_r6_mid_start_history():
    """Design D8: after a mid-task start the three-action history and the filter hold the actions that give the
    reference's joint targets, and the delay queue the reference's targets, so that after k steps the servos'
    target is the reference's, not home."""
    e = r5env.R6Env(nominal(mode="R6", interference="off"))
    p, k, t = e.p, 3, 9.6
    s = e.scripts["brisk"]
    ctrl: list = []
    _, info = e.reset(options=dict(task="brisk", start=t, seeds=dict(start=5, noise=6),
                                   factors=dict(latency_steps=k)))
    t0 = info["t_start"]
    step = 0.002 * RATE
    q_at = lambda x: s.track.joint(x)[:14]  # noqa: E731
    acts = e.core.reference_actions(t0)
    np.testing.assert_array_equal(e.core.hist, acts)
    np.testing.assert_array_equal(e.core.a_f, acts[0])
    obs = e._observation()
    a, b = e.layout["blocks"]["prev_actions"]
    np.testing.assert_allclose(obs[a:b], acts.ravel(), atol=1e-7)
    np.testing.assert_allclose(e.p.q_settled[p.qadr] + acts[0, :14] * e.span, q_at(t0 - 0.02 * RATE), atol=1e-12)
    np.testing.assert_allclose(p.d.ctrl, p.G @ q_at(t0 - k * step), atol=1e-12)  # the oldest of the queue
    np.testing.assert_array_equal(e.tk.q_target, np.clip(q_at(t0), e.core.lo, e.core.hi))
    e._step_hook = lambda pp: ctrl.append(pp.d.ctrl.copy())
    try:
        e.step(acts[0].astype(np.float64))  # the policy repeats its last action
    finally:
        e._step_hook = None
    for j in range(k):  # the queue: the reference's joint targets of the k steps before the start
        np.testing.assert_allclose(ctrl[j], p.G @ q_at(t0 - (k - j) * step), atol=1e-12)
    # when the queue has run out the servos' target is the reference's of one env step before the start, not home
    target = p.G @ q_at(t0 - 0.02 * RATE)
    np.testing.assert_allclose(ctrl[k], target, atol=1e-9)
    home = p.G @ e.p.q_settled[p.qadr]
    assert np.max(np.abs(ctrl[k] - home)) > 0.1 and np.max(np.abs(ctrl[k] - target)) < 1e-9
    # a stance start begins with an empty history
    e.reset(options=dict(task="brisk", start="stance"))
    assert not e.core.hist.any() and not e.core.a_f.any()
    e.close()
