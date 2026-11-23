"""judge_d22.py's choice (D23, D24) on constructed measures, its entry point on real tool output (no crash on the
cases the pre-registration names), and the 10 ms foot-load window on a known series."""

from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pytest
from conftest import RL_DIR  # noqa: F401

sys.path.insert(0, str(RL_DIR / "analysis"))
import judge_d22 as jd  # noqa: E402


def run(L, N=20, h_all=0.80, h_front=0.75, f1=True, f2=True):
    rules = {k: True for k in ("F1", "F2", "T1", "T2", "T3", "T4", "T5")}
    rules.update(F1=f1, F2=f2)
    return {"rules": rules, "L": L, "N": N, "H_all": h_all, "H_front": h_front, "H_elbow": 0.6}


def runs(l128=(140, 120, 130, 110), l160=(100, 105, 110, 100), **kw160):
    r = {(128, s): run(l128[s]) for s in jd.SEEDS}
    r.update({(160, s): run(l160[s], **kw160) for s in jd.SEEDS})
    return r


def test_loads_compare_four_seed_means():
    # 128 mean 125 -> threshold 106.25; 160 mean 103.75 passes, 108 does not
    assert jd.choose(runs())["choice"] == 160
    assert jd.choose(runs(l160=(110, 105, 110, 107)))["choice"] == 128
    # one high 160 seed does not decide it when the mean is clearly lower (D24: averages)
    assert jd.choose(runs(l160=(130, 95, 95, 95)))["choice"] == 160


def test_falls_and_heat_are_not_clearly_worse_within_run3s_scatter():
    assert jd.choose(runs(N=27))["choice"] == 160  # mean 27 <= 20 + 7
    assert jd.choose(runs(N=28))["choice"] == 128
    assert jd.choose(runs(h_all=0.835))["choice"] == 160  # <= 0.80 + 0.035
    assert jd.choose(runs(h_all=0.84))["choice"] == 128
    assert jd.choose(runs(h_front=0.85))["choice"] == 160  # <= 0.75 + 0.10
    assert jd.choose(runs(h_front=0.86))["choice"] == 128  # a hotter front drive keeps 128 mm


def test_the_walk_gate_and_undecided_cases():
    r = runs()
    r[(160, 3)] = run(100, f1=False)
    assert jd.choose(r)["choice"] == 128  # 160 fails flat walking on one seed; 128 walks on all
    r = runs(l160=(200, 200, 200, 200))
    r[(128, 0)] = run(140, f2=False)
    assert jd.choose(r)["choice"] == 160  # only 160 walks on all four, whatever its loads
    r = runs()
    r[(128, 0)] = None
    assert jd.choose(r)["choice"] is None
    r = runs()
    r[(160, 2)]["L"] = None
    assert jd.choose(r)["choice"] is None  # a foot-load episode fell
    r = runs()
    for k in r:
        r[k]["rules"]["F1"] = False
    assert jd.choose(r)["choice"] is None


@pytest.fixture()
def real_inputs(tmp_path):
    """Run 3 seed 0's stored gait report and sweep, a foot-load file and a run record shaped as the tools write them."""
    g = RL_DIR / "analytics" / "results" / "run3-s0" / "gait_report.json"
    w = RL_DIR / "robustness" / "results" / "run3-s0" / "sweep_run3-s0.json.gz"
    foot = {"settings": jd.FOOT_SETTINGS, "falls": 0, "front_p99_window_n": None, "hind_p99_window_n": 120.0,
            "provenance": {"model": "x", "model_sha256": "x", "trike_env_sha256": "x", "run_commit": "x",
                           "zip_sha256": "x", "vecnormalize_sha256": "x"}}
    (tmp_path / "foot.json").write_text(json.dumps(foot))
    (tmp_path / "record.json").write_text(json.dumps({"sessions": [], "settings": {}}))
    shutil.copy(g, tmp_path / "gait.json")
    shutil.copy(w, tmp_path / "sweep.json.gz")
    return tmp_path


def test_main_reports_not_judged_without_crashing(real_inputs, capsys):
    p = real_inputs
    args = ["--run-commit", "HEAD", "--out", str(p / "out.json")]
    for n in jd.LENGTHS:
        for s in jd.SEEDS:
            args += ["--run", str(n), str(s), str(p / "record.json"), str(p / "gait.json"), str(p / "sweep.json.gz"),
                     str(p / "foot.json")]
    args += ["--run", "128", "0", str(p / "missing.json"), str(p / "gait.json"), str(p / "sweep.json.gz"),
             str(p / "foot.json")]
    assert jd.main(args) == 0
    out = capsys.readouterr().out
    assert "NOT JUDGED" in out and "CHOICE: None" in out
    res = json.loads((p / "out.json").read_text())
    assert res["choice"]["choice"] is None and res["problems"]["128_s0"] == ["given twice"]


def test_main_prints_undefined_figures(real_inputs, capsys, monkeypatch):
    """With every check passed (patched), a foot-load file whose L is undefined prints 'undefined', no crash."""
    p = real_inputs
    monkeypatch.setattr(jd, "checks", lambda *a, **k: [])
    shas = iter(range(100))
    real_measure = jd.measure
    monkeypatch.setattr(jd, "measure", lambda g, w, f: {**real_measure(g, w, f), "checkpoint_sha256": next(shas)})
    args = ["--run-commit", "HEAD"]
    for n in jd.LENGTHS:
        for s in jd.SEEDS:
            args += ["--run", str(n), str(s), str(p / "record.json"), str(p / "gait.json"), str(p / "sweep.json.gz"),
                     str(p / "foot.json")]
    assert jd.main(args) == 0
    out = capsys.readouterr().out
    assert "L undefined N" in out and "CHOICE: None - a foot load or front-drive figure is undefined" in out


def test_the_ten_millisecond_window_of_the_load_recorder():
    from telemetry.loads import LoadRecorder

    rec = LoadRecorder.__new__(LoadRecorder)  # only the summary's arithmetic is under test
    rec.names, rec.foot_names, rec.every, rec.ball, rec.shared = [], ["f"], 1, [], {}
    rec.force, rec.moment = [], []
    series = np.zeros(1000)
    series[500:505] = 100.0  # a 5 ms spike of 100 N
    rec.foot = [[v] for v in series]
    s = rec.summary(100.0, physics_dt=0.001)
    assert s["window_s"] == pytest.approx(0.010) and s["foot_samples"] == 1000
    assert s["feet"]["f"]["normal_n"]["max"] == 100.0
    assert s["feet"]["f"]["normal_window_n"]["max"] == pytest.approx(50.0)  # 5 of 10 samples


RUN3_COMMIT = "efd1e1c7d391062c96880b539f76bbc99d30b48c"


@pytest.fixture()
def run3_as_a_d22_run(monkeypatch):
    """Run 3 seed 0's real record settings and tool output, shaped as a D22 run on run 3's own model and commit:
    the checks must pass it, and catch each slip."""
    import copy

    have = __import__("subprocess").run(["git", "-C", str(jd.j3.REPO), "cat-file", "-e", RUN3_COMMIT],
                                        capture_output=True).returncode == 0
    if not have:
        pytest.skip("run 3's commit is not in this clone")
    monkeypatch.setattr(jd, "model_of", lambda n: "trike60_C_closed_capsule.xml")
    ref = json.loads(jd.REFERENCE.read_text())["settings"]
    args = {k: ref[k] for k in jd.PPO_KEYS} | {"seed": 0, "model": "trike60_C_closed_capsule", "allow_changes": False}
    record = {"settings": copy.deepcopy(ref), "sessions": [{"git": {"commit": RUN3_COMMIT, "dirty": False},
                                                            "status": "completed", "args": args,
                                                            "changed_settings": None,
                                                            "saves": ["trike60_10002432_steps.zip"]}]}
    gait = jd.j3._load(str(RL_DIR / "analytics" / "results" / "run3-s0" / "gait_report.json"))
    sweep = jd.j3._load(str(RL_DIR / "robustness" / "results" / "run3-s0" / "sweep_run3-s0.json.gz"))
    pol = gait["policy"]
    foot = {"settings": dict(jd.FOOT_SETTINGS), "provenance": {
        "model": "trike60_C_closed_capsule.xml", "model_sha256": gait["environment"]["model_sha256"][
            "trike60_C_closed_capsule.xml"], "trike_env_sha256": gait["environment"]["trike_env_sha256"],
        "run_commit": RUN3_COMMIT, "zip_sha256": pol["zip_sha256"], "vecnormalize_sha256": pol["vecnormalize_sha256"]}}
    return record, gait, sweep, foot


def test_a_correct_run_passes_every_check(run3_as_a_d22_run):
    assert jd.checks(128, 0, RUN3_COMMIT, *run3_as_a_d22_run) == []


@pytest.mark.parametrize("slip,expect", [
    (lambda r, g, w, f: g["settings"].update(episode_s=10.0), "gait report"),  # --quick
    (lambda r, g, w, f: g["cases"]["cmd_0.15"]["summary"].update(episodes=2), "gait report"),
    (lambda r, g, w, f: r["settings"].update(n_steps=512), "training settings"),
    (lambda r, g, w, f: r["settings"]["env_config"].update(w_heat=0.0), "environment differs"),
    (lambda r, g, w, f: r["settings"]["env_config"].update(command_mode="heading"), "steering is on"),
    (lambda r, g, w, f: r["sessions"].insert(0, {**r["sessions"][0], "args": {**r["sessions"][0]["args"],
                                                                              "n_envs": 4}}), "arguments differ"),
    (lambda r, g, w, f: r["settings"].update(seed=1), "another seed"),
    (lambda r, g, w, f: f["settings"].update(seconds=10.0), "foot loads: settings"),
    (lambda r, g, w, f: w.update(seeds_per_point=1), "sweep: not 8"),
])
def test_each_slip_makes_the_run_not_judged(run3_as_a_d22_run, slip, expect):
    record, gait, sweep, foot = run3_as_a_d22_run
    slip(record, gait, sweep, foot)
    problems = jd.checks(128, 0, RUN3_COMMIT, record, gait, sweep, foot)
    assert any(expect in p for p in problems), problems


def test_a_not_finite_load_decides_nothing():
    r = runs()
    r[(128, 1)]["L"] = float("nan")
    assert jd.choose(r)["choice"] is None
