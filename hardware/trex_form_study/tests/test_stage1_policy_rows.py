"""Tests for the Stage 1 policy rows R3 and R4 (designlab/stage1/policy_rows.py; STAGE1_PLAN.md section 2's R3 and R4,
section 3 step 8; the panels' Row interface).

Run from hardware/trex_form_study in the PPO venv (torch, Stable-Baselines3, gymnasium):
    STAGE1_POLICY_RUNS=<training run directory> python -m pytest tests/test_stage1_policy_rows.py -q
The PPO checkpoints are git-ignored: ``STAGE1_POLICY_RUNS`` names the directory training wrote them to (default: this
checkout's stage1/runs/2026-11-08), and the tests that need a checkpoint skip cleanly without one; the tests that run
episodes also need the run directory's references and transitions.  Every test is short: the equivalence with
training's env (policy_rows.EQUIVALENCE_RULE) runs a few simulated seconds of each episode, and the whole episodes
with STAGE1_SLOW=1.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("gymnasium")

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
from designlab.stage1 import control, env, panels, plant, policy_rows, train, tune  # noqa: E402

RUN = ROOT / "stage1" / "runs" / "2026-11-08"
RUNS = Path(os.environ.get(policy_rows.RUNS_ENV) or RUN)
SLOW = bool(os.environ.get("STAGE1_SLOW"))
R3_BODY = "75-M-H-10"
R4_BODY = "60-N-T-10"
COND = {c.key: c for c in panels.conditions()}
#: A short cut of each episode the equivalence tests compare (whole episodes with STAGE1_SLOW=1).
CUT_S = None if SLOW else 3.0
EPISODE_KEYS = {"seed", "fell", "fall_time_s", "fall_cause", "fall_onset_s", "fall_candidates", "limits_before_onset",
                "survival_s", "planned_s", "strides", "worst_sustained_ratio", "worst_speed_ratio",
                "tilt_from_home_max_deg", "joint_margin_min_rad", "joint_shortfall_rad", "joint_worst", "estimator",
                "ankle_cap_residual", "captures", "phase_rate_range", "success", "succeeded", "failed_terms",
                "measures", "relaxed", "wall_s"}


def _needs_ppo():
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")


def _newest(body: str, mode: str, seed: int = 42) -> int:
    """The run's newest complete checkpoint, or a skip (they are git-ignored)."""
    _needs_ppo()
    if not (RUN / body / "refs" / "nominal.npz").exists():
        pytest.skip("the run directory's references are not generated (designlab.stage1 prepare)")
    rd = train.run_dir_of(RUNS, body, mode, seed)
    have = train.checkpoints(rd)
    if not have:
        pytest.skip(f"no {mode} checkpoint of {body} seed {seed} under {RUNS} (set {policy_rows.RUNS_ENV})")
    doc = json.loads((rd / "run.json").read_text())
    try:
        policy_rows.check_versions(dict(versions=doc["sessions"][0]["versions"]), str(rd))
    except policy_rows.PolicyRowError as exc:
        pytest.skip(f"{exc}")
    return have[-1]


@pytest.fixture()
def runs(monkeypatch):
    monkeypatch.setenv(policy_rows.RUNS_ENV, str(RUNS))
    monkeypatch.delenv(policy_rows.CHECKPOINT_ENV, raising=False)
    return RUNS


def _ctx(body):
    _, docs = tune.read_inputs(RUN / body, panels.panel_files())
    return panels.Context(body, RUN / body, docs)


# ------------------------------------------------------------------------------------------------
# Registration, keys, checkpoints
# ------------------------------------------------------------------------------------------------
def test_importing_registers_r3_and_r4_and_leaves_the_panels_keys():
    """The module registers R3 and R4 with the panels (eval --row-module) and is not among the panels' or the tuner's
    sources, so the R0-R2 records' keys and the fit's sources do not change (panels.PANEL_SOURCES_RULE)."""
    assert panels.ROW_FACTORIES["R3"] is policy_rows.r3_row and panels.ROW_FACTORIES["R4"] is policy_rows.r4_row
    files = tune.package_files()
    assert "policy_rows.py" in files and "policy_rows.py" not in panels.key_modules(files)
    without = {k: v for k, v in files.items() if k != "policy_rows.py"}
    assert panels.panel_sources(without)["sha256"] == panels.panel_sources(files)["sha256"]
    assert "policy_rows.py" not in tune.tuner_modules(files)
    assert "R3" in panels.ATTRIBUTED and "R4" not in panels.ATTRIBUTED  # decision 20 on R3's falls only
    assert panels.VERDICT_ROWS["R3"] == ("R3/42", "R3/43") and policy_rows.R4_ROWS == ("R4/42",)


def test_the_checkpoint_choice():
    """STAGE1_POLICY_CHECKPOINT: judged by default; a step count, 'newest', or per mode (CHECKPOINT_SELECTION_RULE)."""
    c = policy_rows.checkpoint_choice
    assert c("R3", "") == c("R4", " ") == "judged"
    assert c("R3", "newest") == "newest" and c("R4", "753664") == 753664
    assert c("R3", "R3=newest,R4=1007616") == "newest" and c("R4", "R3=newest,R4=1007616") == 1007616
    assert c("R3", "500,R4=newest") == 500 and c("R4", "500,R4=newest") == "newest"
    assert c("R4", "R3=500") == "judged"
    for bad in ("latest", "R5=1", "R3=-1"):
        with pytest.raises(policy_rows.PolicyRowError):
            c("R3", bad)


def _fake_runs(tmp_path, steps_by_run, final_record=None):
    """A training run directory of R3 runs with run.json and (empty) checkpoint files."""
    root = tmp_path / "runs"
    cfg = dict(rate=1.05, tasks=list(env.TASKS), task_weights=None, max_seconds=30.0, randomize=True,
               friction=[0.54, 1.25], mass=[0.9, 1.1], kp=[0.8, 1.2], latency_steps=[0, 10], noise_scale=[0.5, 1.5],
               push_prob=0.5, push_frac=[0.0, 1.0], push_s=0.1, start_joint_noise=0.01, start_root_velocity_noise=0.02,
               gains="tuned", pile_grid=0.008)
    for body in panels.BODIES:
        for seed in train.SEEDS:
            rd = train.run_dir_of(root, body, "R3", seed)
            (rd / "checkpoints").mkdir(parents=True)
            for n in steps_by_run.get((body, seed), (253952, 507904)):
                (rd / "checkpoints" / f"model_{n}.zip").write_bytes(f"model {body} {seed} {n}".encode())
                (rd / "checkpoints" / f"vecnormalize_{n}.pkl").write_bytes(f"vn {body} {seed} {n}".encode())
            doc = dict(mode="R3", body=body, seed=seed, budget=train.BUDGET_STEPS,
                       sessions=[dict(versions=policy_rows.library_versions())],
                       settings=dict(log_std_init=-3.9, learning_rate=dict(start=1e-5, end=3e-6, rule="x"),
                                     design2={}, n_envs=4, checkpoint_every=250000, floor_rule="f"),
                       env_configs=[dict(cfg, floor=f) for f in env.FLOOR_SPREAD],
                       gains=dict(values={}, candidate="c", cma_seed=43, status="passed"))
            if final_record is not None:
                doc["judged_checkpoint"] = final_record
            (rd / "run.json").write_text(json.dumps(doc))
    return root


def test_the_judged_checkpoint_and_an_interim_one(tmp_path, monkeypatch):
    """By default the mode's judged checkpoint (the last every run reached), provisional until every run holds the
    budget's or a 'judge --final' record names it; an explicit or newest checkpoint is never final; a run trained at
    another commanded rate, or with other library versions than this process's, is refused."""
    monkeypatch.delenv(policy_rows.RUNS_ENV, raising=False)  # the runs: the panels' run directory's parent
    root = _fake_runs(tmp_path, {("60-N-T-10", 42): (253952, 507904, 753664)})
    rd = root / "60-N-T-10"
    c = policy_rows.resolve("60-N-T-10", "R3", 42, rd, "judged")
    assert c["steps"] == 507904 and c["final"] is False and c["judged"]["every_run_at_budget"] is False
    assert c["files"]["model"]["file"] == "60-N-T-10/ppo/42/checkpoints/model_507904.zip"
    assert c["files"]["model"]["sha256"] == plant.sha256(root / c["files"]["model"]["file"])
    assert c["run"]["log_std_init"] == -3.9 and c["run"]["env"]["floors"] == list(env.FLOOR_SPREAD)
    n = policy_rows.resolve("60-N-T-10", "R3", 42, rd, "newest")
    assert n["steps"] == 753664 and n["final"] is False and n["judged"] is None and "interim" in n["selection"]
    assert policy_rows.resolve("60-N-T-10", "R3", 42, rd, 253952)["steps"] == 253952
    with pytest.raises(policy_rows.PolicyRowError, match="no complete checkpoint at 1000"):
        policy_rows.resolve("60-N-T-10", "R3", 42, rd, 1000)
    final = dict(steps=507904, final=True)
    root2 = _fake_runs(tmp_path / "b", {}, final_record=final)
    f = policy_rows.resolve("60-N-T-10", "R3", 43, root2 / "60-N-T-10", "judged")
    assert f["final"] is True and f["judged"]["judge_final_recorded"] is True
    stale = _fake_runs(tmp_path / "c", {}, final_record=dict(steps=253952, final=True))
    assert policy_rows.resolve("60-N-T-10", "R3", 43, stale / "60-N-T-10", "judged")["final"] is False
    path = train.run_dir_of(root, "60-N-T-10", "R3", 43) / "run.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps(dict(doc, env_configs=[dict(x, rate=1.0) for x in doc["env_configs"]])))
    with pytest.raises(policy_rows.PolicyRowError, match="commanded rate"):
        policy_rows.resolve("60-N-T-10", "R3", 43, rd, "judged")
    other = dict(policy_rows.library_versions(), torch="0.0.0")
    path.write_text(json.dumps(dict(doc, sessions=[dict(versions=other)])))
    with pytest.raises(policy_rows.PolicyRowError, match="libraries differ"):
        policy_rows.resolve("60-N-T-10", "R3", 43, rd, "judged")


def test_rows_name_their_checkpoint_run_and_sources(runs, monkeypatch):
    """R3/42 and R4/42 through panels.tracker_rows: each names its model and VecNormalize with their sha256, the step
    count, the run's settings and, for R3, the gains the run trained on; its key names env.py, policy_rows.py and
    policy.py; its spec rebuilds the same row in a worker."""
    n3, n4 = _newest(R3_BODY, "R3"), _newest(R3_BODY, "R4")
    monkeypatch.setenv(policy_rows.CHECKPOINT_ENV, f"R3={n3},R4={n4}")
    rows = panels.tracker_rows(R3_BODY, RUN / R3_BODY, ("R3/42", "R4/42"))
    for name, mode, n in (("R3/42", "R3", n3), ("R4/42", "R4", n4)):
        row = rows[name]
        d = row.describe()
        ck = d["policy"]["checkpoint"]
        rd = train.run_dir_of(RUNS, R3_BODY, mode, 42)
        doc = json.loads((rd / "run.json").read_text())
        assert ck["steps"] == n and ck["final"] is False and "interim" in ck["selection"]
        assert ck["model"]["sha256"] == plant.sha256(rd / "checkpoints" / f"model_{n}.zip")
        assert ck["vecnormalize"]["sha256"] == plant.sha256(rd / "checkpoints" / f"vecnormalize_{n}.pkl")
        assert d["policy"]["run"]["log_std_init"] == doc["settings"]["log_std_init"]
        assert d["policy"]["run"]["learning_rate"] == doc["settings"]["learning_rate"]
        assert set(d["policy"]["libraries"]) == {"torch", "stable_baselines3", "gymnasium"}
        assert set(row.sources()) == {"designlab/stage1/env.py", "designlab/stage1/policy_rows.py",
                                      "designlab/stage1/policy.py"}
        again = panels.row_from_spec(pickle.loads(pickle.dumps(row.spec())))
        assert type(again) is type(row) and again.describe() == d and again.sources() == row.sources()
        assert row.gated(COND["carpet/low"]) and not row.hard_only  # decision 23: carpet gated for R3 and R4
    r3 = rows["R3/42"]
    gains = json.loads((train.run_dir_of(RUNS, R3_BODY, "R3", 42) / "run.json").read_text())["gains"]
    assert control.gains_dict(r3.gains) == gains["values"] and r3.describe()["tracker_mode"] == "R2"
    assert r3.describe()["gains_info"]["candidate"] == gains["candidate"]
    assert rows["R4/42"].describe()["gains"] is None and rows["R4/42"].describe()["mapping"]["inputs"]


def test_r3_refuses_gains_other_than_the_runs(runs, monkeypatch):
    n = _newest(R3_BODY, "R3")
    real = env.tuned_gains

    def nudged(run_dir, body=None, path=None):
        g, info = real(run_dir, body, path)
        return g + np.r_[1e-12, np.zeros(len(g) - 1)], info

    monkeypatch.setattr(env, "tuned_gains", nudged)
    with pytest.raises(policy_rows.PolicyRowError, match="refusing"):
        policy_rows.r3_row(R3_BODY, RUN / R3_BODY, "R3/42", n)


def test_a_changed_checkpoint_file_is_refused(runs):
    n = _newest(R3_BODY, "R3")
    row = policy_rows.r3_row(R3_BODY, RUN / R3_BODY, "R3/42", n)
    ck = json.loads(json.dumps(row.policy.ckpt))
    ck["files"]["vecnormalize"]["sha256"] = "0" * 64
    with pytest.raises(policy_rows.PolicyRowError, match="checkpoint changed"):
        policy_rows.TrainedPolicy(ck).load()
    model, vn = row.policy.load()
    assert model.policy.actor_dim == env.ACTOR_DIM and vn.training is False


def test_the_phase_period_follows_the_env_tasks():
    """PERIOD_RULE: a condition whose plan is an env task's takes that task's stride reference; every other condition
    the nominal walk's, as env.TASKS names it for every task but the carpet walk."""
    assert {r for t, (_, r) in env.TASKS.items() if t != "carpet_walk"} == {"nominal"}
    assert env.TASKS["carpet_walk"][1] == "carpet"
    want = {"walk/nominal": "walk", "walk/nominal_mu0.54": "walk", "turn/walking_0.35_left": "walking_turn_left",
            "turn/walking_0.35_right": "walking_turn_right", "turn/in_place_0.50_left": "in_place_turn_left",
            "turn/in_place_0.50_right": "in_place_turn_right", **{f"carpet/{f}": "carpet_walk" for f in
                                                                   ("robot_floor_carpet", "low", "medium", "high")}}
    for key, cond in COND.items():
        task, ref = policy_rows.env_task(panels.template(cond))
        assert task == want.get(key) and ref == ("carpet" if cond.behavior == "carpet" else "nominal"), key


# ------------------------------------------------------------------------------------------------
# Equivalence with training's env (EQUIVALENCE_RULE), per physics step
# ------------------------------------------------------------------------------------------------
def _assert_identical(res):
    assert res["identical"], {k: v for k, v in res.items() if k != "rule"}
    assert res["first_difference"] is None and res["ends_alike"] and res["compared_steps"] > 0


def test_r3_is_training_env_on_the_walk(runs):
    """R3/42 on the panels' nominal walk (hard floor) is env.ResidualEnv on 'walk' (randomize False, no push,
    latency 0, the panel's start and sensor-noise seeds): every physics step's qpos, qvel, ctrl and sensordata."""
    n = _newest(R3_BODY, "R3")
    res = policy_rows.equivalence_check(R3_BODY, "R3/42", "walk/nominal", run_dir=RUN / R3_BODY, seconds=CUT_S,
                                        choice=n)
    _assert_identical(res)
    assert res["task"] == "walk" and res["checkpoint"]["steps"] == n and res["seeds"]["slot"] == 0


def test_r3_is_training_env_on_the_carpet_walk_on_low_pile(runs):
    """The same on the carpet walk on the panels' 4 mm low pile at mu 0.54 (the env on the same pile: EnvConfig
    pile_grid None, as the panels build it; training used 8 mm)."""
    n = _newest(R3_BODY, "R3")
    res = policy_rows.equivalence_check(R3_BODY, "R3/42", "carpet/low", run_dir=RUN / R3_BODY, seconds=CUT_S,
                                        choice=n)
    _assert_identical(res)
    assert res["task"] == "carpet_walk" and res["floor"] == "low" and res["friction"] == 0.54


def test_r4_is_training_env_on_the_walk(runs):
    """R4/42 on the panels' nominal walk is env.PolicyOnlyEnv on 'walk' (its span over the run's env tasks)."""
    n = _newest(R4_BODY, "R4")
    res = policy_rows.equivalence_check(R4_BODY, "R4/42", "walk/nominal", run_dir=RUN / R4_BODY, seconds=CUT_S,
                                        choice=n)
    _assert_identical(res)


def test_the_check_sees_another_filter(runs, monkeypatch):
    """The comparison is sensitive: without the 10 Hz filter (alpha 1) the row departs from training's env at once."""
    n = _newest(R3_BODY, "R3")
    monkeypatch.setattr(policy_rows, "filter_alpha", lambda dt: 1.0)
    res = policy_rows.equivalence_check(R3_BODY, "R3/42", "walk/nominal", run_dir=RUN / R3_BODY, seconds=0.2, choice=n)
    assert not res["identical"] and res["first_difference"] is not None and res["first_difference"] < 20


def test_the_policy_acts(runs):
    """R3 is not R2: the trained residual moves the servo targets from the first physics step (the action before it,
    POLICY_RULE), and R4's targets are the policy's."""
    n = _newest(R3_BODY, "R3")
    ctx = _ctx(R3_BODY)
    r3 = policy_rows.r3_row(R3_BODY, RUN / R3_BODY, "R3/42", n)
    r2 = panels.TrackerRow("R2/zero", "R2", r3.gains, r3.gains_info, policy=panels.ZeroResidual())
    cond = dataclasses.replace(COND["walk/nominal"], seconds=0.1)
    digests = {}
    for name, row in (("R3", r3), ("R2", r2)):
        b = ctx.built(cond)
        rows = []
        with policy_rows.digesting(b.p, rows):
            row.episode(ctx, cond, 3042)
        digests[name] = rows
    assert len(digests["R3"]) == len(digests["R2"]) == 50 and digests["R3"][0] != digests["R2"][0]
    assert r3.policy.view.tk.residual is not None and r3.policy.a_filt.any()


# ------------------------------------------------------------------------------------------------
# Every condition, relaxations, the panels' runner
# ------------------------------------------------------------------------------------------------
def _short(cond: panels.Condition) -> panels.Condition:
    """A condition cut to a fraction of a second (a push's pulse inside it)."""
    if cond.judge in ("push_stand", "push_walk"):
        return dataclasses.replace(cond, push_at=0.05, push_spread=0.0, recovery_s=0.05, hold_s=0.05)
    return dataclasses.replace(cond, seconds=0.2)


def test_every_condition_runs_on_both_rows(runs):
    """R3/42 and R4/42 run every condition of a body whose floors settle (cut short), each judged into the panels'
    episode record, and take every relaxation decision 20's attribution lifts."""
    n3, n4 = _newest(R4_BODY, "R3"), _newest(R4_BODY, "R4")
    ctx = _ctx(R4_BODY)
    rows = [policy_rows.r3_row(R4_BODY, RUN / R4_BODY, "R3/42", n3),
            policy_rows.r4_row(R4_BODY, RUN / R4_BODY, "R4/42", n4)]
    for cond in panels.conditions():
        if not panels.stands(RUN / R4_BODY, cond.floor)[0]:
            continue  # panels.NOT_RUN_RULE: the 60 cm bodies on medium and high pile
        for row in rows:
            e = row.run(ctx, _short(cond), 3042)
            assert set(e) == EPISODE_KEYS and e["planned_s"] > 0 and e["relaxed"] == [], (row.name, cond.key)
            assert not e.get("error"), (row.name, cond.key)
    stand = _short(COND["stand/hard"])
    for row in rows:
        for r in control.RELAXATIONS:
            e = row.run(ctx, stand, 3042, relax=(r,))
            assert e["relaxed"] == [r] and not e["fell"], (row.name, r)
    p = ctx.plant("hard", None)
    assert p.envelope.enabled and not np.any(p.m.geom_contype == 2)  # every relaxation restored


def test_r4_reports_as_r4_with_no_tracker_terms(runs):
    n = _newest(R4_BODY, "R4")
    ctx = _ctx(R4_BODY)
    row = policy_rows.r4_row(R4_BODY, RUN / R4_BODY, "R4/42", n)
    b, out, rec = row.episode(ctx, dataclasses.replace(COND["walk/nominal"], seconds=0.3), 3042)
    assert out["mode"] == "R4" and out["terms"] == [] and out["ankle_cap"] == [] and out["ankle_cap_residual"] is None
    assert out["phase_rate_range"] == [0.0, 0.0] and out["estimator"]["updates"] == rec.rows
    q_home, span = row.mapping(ctx, b)
    assert np.array_equal(q_home, b.p.q_settled[b.p.qadr]) and np.all(span >= env.POLICY_ONLY_MIN_SPAN_RAD)


def test_eval_runs_and_resumes_the_policy_rows(tmp_path, runs, monkeypatch):
    """eval --row-module with the policy rows: one record per row, its key naming the policy's checkpoint and sources;
    a second session reuses them (panels.RESUME_RULE); a new checkpoint selection re-runs them."""
    n3, n4 = _newest(R4_BODY, "R3"), _newest(R4_BODY, "R4")
    cut = [dataclasses.replace(COND["stand/hard"], seconds=0.3)]
    monkeypatch.setattr(panels, "conditions", lambda: cut)
    monkeypatch.setenv(policy_rows.CHECKPOINT_ENV, f"R3={n3},R4={n4}")
    from designlab.stage1.__main__ import main as cli

    args = ["eval", "--body", R4_BODY, "--rows", "R3/42,R4/42", "--episodes", "1", "--no-traces", "--row-module",
            "designlab.stage1.policy_rows", "--inputs", str(RUN), "--out", str(tmp_path)]
    assert cli(args) == 0
    for name, n in (("R3/42", n3), ("R4/42", n4)):
        rec = json.loads(panels.record_path(tmp_path, R4_BODY, name, cut[0]).read_text())
        assert rec["controller"]["policy"]["checkpoint"]["steps"] == n and rec["gated"] is True
        assert "designlab/stage1/policy_rows.py" in rec["key"]["row_sources"] and len(rec["episodes"]) == 1
    doc = panels.evaluate(R4_BODY, tmp_path, rows=("R3/42", "R4/42"), episodes=1, inputs=RUN, traces=False,
                          log=lambda m: None)
    assert doc["sessions"][-1]["episodes_run"] == 0 and len(doc["sessions"][-1]["skipped"]) == 2
    older = train.checkpoints(train.run_dir_of(RUNS, R4_BODY, "R3", 42))[0]
    monkeypatch.setenv(policy_rows.CHECKPOINT_ENV, f"R3={older},R4={n4}")
    doc = panels.evaluate(R4_BODY, tmp_path, rows=("R3/42", "R4/42"), episodes=1, inputs=RUN, traces=False,
                          log=lambda m: None)
    assert doc["sessions"][-1]["episodes_run"] == (1 if older != n3 else 0)


# ------------------------------------------------------------------------------------------------
# Verdicts
# ------------------------------------------------------------------------------------------------
def _record(row, pass_, label=None, falls=0):
    eps = [dict(seed=s, fell=i < falls, succeeded=i >= falls, failed_terms=[] if i >= falls else ["survived"],
                fall_candidates=[], fall_cause="tilt" if i < falls else None) for i, s in enumerate(panels.SEEDS)]
    c = panels.counts(eps)
    c["pass_"] = pass_
    fail = None if pass_ else panels.failure(row, eps, None, pass_) if label is None else dict(label=label)
    return dict(status="done", row=row, counts=c, episodes=eps, failure=fail)


def test_r4_is_at_most_provisional(tmp_path):
    """R4_VERDICT_RULE: one seed's pass is provisional; failures read as the panels read them (R4's falls are not
    attributed); policy_verdicts writes R3's (the panels' VERDICT_RULE) and R4's per condition."""
    assert policy_rows.r4_verdict([_record("R4/42", True)])["verdict"] == "provisional"
    v = policy_rows.r4_verdict([_record("R4/42", False, falls=3)])
    assert v["verdict"] == "not attributed" and v["rule"] == policy_rows.R4_VERDICT_RULE
    assert policy_rows.r4_verdict([_record("R4/42", False, label="body-limited: slip")])["verdict"] == "body-limited"
    assert policy_rows.r4_verdict([None])["verdict"] == "incomplete"
    cond = COND["carpet/low"]
    for row, ok in (("R3/42", True), ("R3/43", False), ("R4/42", True)):
        rec = _record(row, ok, label=None if ok else "controller-limited")
        rec["controller"] = dict(policy=dict(checkpoint=dict(steps=753664, selection="judged", final=True)))
        panels._write_json(panels.record_path(tmp_path, R4_BODY, row, cond), rec)
    doc = policy_rows.policy_verdicts(R4_BODY, tmp_path)
    cell = doc["grid"]["carpet"]["low"]
    assert cell["R3"]["verdict"] == "provisional" and cell["R4"]["verdict"] == "provisional"
    assert cell["R3"]["gated"] and cell["R4"]["gated"]  # decision 23: the carpet walk is gated for R3 and R4
    assert doc["checkpoints"]["R4/42"] == [dict(steps=753664, selection="judged", final=True)]
    assert (tmp_path / R4_BODY / "eval" / "policy_rows.json").exists()
    assert doc["grid"]["walk"]["nominal"]["R4"]["verdict"] == "incomplete"
