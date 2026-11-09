"""Tests for the Stage 1 report tables (designlab/stage1/report.py; STAGE1_PLAN.md section 8).

Run from hardware/trex_form_study (the fit's venv):  python -m pytest tests/test_stage1_report.py -q
Fast: the cells, flags, verdicts, ledger, PPO, fit and cost tables on a synthetic run directory, and the report on
the real run directory (read only: every file it writes goes to a temporary directory).  The real directory is
``$STAGE1_REPORT_RUN`` if set (e.g. the main checkout's live run, which the panels and the training are writing),
else this checkout's stage1/runs/2026-11-08.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401
from designlab.stage1 import control, ledger, panels, report  # noqa: E402
from designlab.stage1.__main__ import main as cli  # noqa: E402

BODY = "60-N-T-10"
REAL = Path(os.environ.get("STAGE1_REPORT_RUN") or ROOT / "stage1" / "runs" / "2026-11-08")
COND = {c.key: c for c in panels.conditions()}
SRC = panels.panel_sources()["sha256"]  # the records' panel sources: the ones read now (FLAGS_RULE)


# ------------------------------------------------------------------------------------------------
# A synthetic run directory
# ------------------------------------------------------------------------------------------------
def write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)


def panel_record(row, key, survived, succeeded, n=20, label=None, kind=None, falls_label=None, sources=SRC,
                 controller=None, status="done", attribution=None):
    cond = COND[key]
    judged = n == 20
    pass_ = (survived == 20 and succeeded >= 17) if judged else None
    failure = None if pass_ is not False else dict(label=label, kind=kind, falls_label=falls_label,
                                                   success_label=label if kind == "success" else None, untestable=[])
    return dict(schema=2, body=BODY, row=row, behavior=cond.behavior, condition=cond.name, key_name=key,
                gated=panels.gated_for(row, cond), plan_gated=cond.gated, status=status,
                controller=controller or dict(row=row), counts=dict(
                    episodes=n, survived=survived, succeeded=succeeded, falls=n - survived, errors=0,
                    survival_lcb=panels.lcb(survived, n), success_lcb=panels.lcb(succeeded, n), judged=judged,
                    pass_=pass_), failure=failure, attribution=attribution, key=dict(sources=sources), wall_s=100.0,
                episode_wall_s=90.0, episodes=[])


def bisected(search, threshold):
    """A search's two results on a monotone synthetic judge (survival 0.1 further toward the far end)."""
    out = dict(search.describe())
    sign = 1 if search.far > search.nominal else -1
    for crit, t in (("success", threshold), ("survival", threshold + sign * 0.1)):
        b = ledger.Bisect(search, crit)
        while not b.done:
            v = b.need()
            b.feed(v, (v <= t) if sign > 0 else (v >= t))
        r = b.result()
        r["binding"] = dict(label="body-limited: slip_worst" if crit == "success" else "falls: candidates slip x3 "
                                                                                       "(not attributed: 3 seeds)")
        out[crit] = r
    return out


def make_run(root: Path) -> Path:
    b = root / BODY
    gains = {n: float(v) for n, v in zip(control.GAIN_NAMES, control.START_GAINS, strict=True)}
    conds = {k: dict(successes=s, survived=20, runs=20, pass_=True) for k, s in (("walk", 18), ("walking_turn", 17),
                                                                                  ("in_place_turn", 20))}
    seeds = {str(s): dict(selected=f"seed{s}_gen20", selected_gains=gains, pass_=True, label="passes G5",
                          selection=[dict(candidate=f"seed{s}_gen20", falls=0, S_mean=0.0, P_mean=0.05 + s * 1e-3,
                                          J_mean=0.05 + s * 1e-3)], conditions=conds) for s in (42, 43)}
    write(b / "tune.json", dict(body=BODY, pass_=True, seeds_passing=[42, 43], seeds=seeds, seconds=1800.0,
                                runs={s: dict(status="done", evals=720, generations=60, sigma=0.06) for s in seeds},
                                objective=dict(name="stage1-tune-v2")))
    for s in (42, 43):
        write(b / "tune" / str(s) / "cma.json", dict(sessions=[dict(started_at="2026-11-09T08:00:00+00:00",
                                                                     ended_at="2026-11-09T10:00:00+00:00",
                                                                     workers=1)]))
        write(b / "tune" / str(s) / "evals.jsonl", "\n".join(json.dumps(dict(wall_s=10.0)) for _ in range(3)))
    write(b / "refs.json", dict(seconds=1800.0))
    write(b / "transitions.json", dict(seconds=1800.0))
    write(b / "gates.json", dict(G1=dict(seconds=360.0), G2=dict(seconds=360.0), G4=dict(seconds=0.0)))
    recs = [panel_record("R2/42", "walk/nominal", 20, 18), panel_record("R2/43", "walk/nominal", 20, 15,
                                                                         label="body-limited: slip_worst",
                                                                         kind="success"),
            panel_record("R2/42", "stand/hard", 20, 20), panel_record("R2/43", "stand/hard", 20, 20),
            panel_record("R2/42", "walk/brisk", 17, 17, label="not attributed (falls; run eval --attribute)",
                         kind="falls", falls_label="not attributed (falls; run eval --attribute)"),
            panel_record("R0", "walk/nominal", 2, 2, n=2), panel_record("R1", "walk/nominal", 20, 20, sources="old"),
            panel_record("R3/42", "walk/nominal", 20, 19, controller=dict(row="R3/42", checkpoint_steps=507904)),
            dict(panel_record("R2/42", "stand/low", 0, 0), status="not run", counts=dict(episodes=0, judged=False,
                                                                                         pass_=None))]
    for r in recs:
        write(panels.record_path(root, BODY, r["row"], COND[r["key_name"]]), r)
    write(b / "eval" / "g7.json", dict(pass_=True, measured=dict(sway_H=0.1681, cadence_steps_s=1.9612),
                                       stage0=dict(sway_H=0.1680, cadence_steps_s=1.9610),
                                       plan=dict(sway_H=0.168, cadence_steps_s=1.96)))
    m = {x.name: x for x in ledger.margins(BODY)}
    grip = m["grip"]
    searches = {s.name: bisected(s, t) for s, t in zip(grip.searches, (0.53, 0.47, 0.61, 0.40), strict=True)}
    write(ledger.record_path(root, BODY, "R2/42", "grip"), dict(
        row="R2/42", margin="grip", status="done", summary=ledger.margin_summary(grip, searches, {}), cut_s=None,
        halvings=7, cost=dict(cpu_s=3600.0, wall_s=4000.0), definition=grip.describe()))
    write(ledger.record_path(root, BODY, "R2/42", "stop"), dict(
        row="R2/42", margin="stop", status="not run", not_run=m["stop"].not_run, summary=ledger.margin_summary(
            m["stop"], {}, {}), reference_stop=dict(peak_deceleration_m_s2=0.43), panel_reading=None,
        definition=m["stop"].describe(), cost=dict(cpu_s=0.0, wall_s=0.0)))
    mass = m["mass"]
    write(ledger.record_path(root, BODY, "R2/42", "mass"), dict(
        row="R2/42", margin="mass", status="partial", summary=ledger.margin_summary(
            mass, {"walk_nominal": bisected(mass.searches[0], 1.2)},
            {"walk_nominal_trunk_x1.1": dict(status="done", value=1.1, condition="walk/nominal", works=True,
                                             works_outright=False, survives=True, succeeded=2, survived=3,
                                             same_as_nominal=False)}),
        definition=mass.describe(), cost=dict(cpu_s=360.0, wall_s=400.0),
        masses=dict(trunk_body_kg=4.6998, robot_kg=10.8117, trunk_share=4.6998 / 10.8117)))
    write(b / "sweeps" / "summary.json", dict(sessions=[dict(wall_s=7200.0)]))
    first = dict(first_rollouts=True, session=0, wall_s=100.0, cpu_s=50.0, env_processes=1)
    write(b / "ppo" / "42" / "run.json", dict(
        mode="R3", seed=42, status="running", num_timesteps=507904, budget=4005888, checkpoints=[253952, 507904],
        judged_checkpoint=dict(steps=507904, final=False, budget=4005888),
        g6=dict(G6a_zero_residual=dict(identical=True), G6a_rate=dict(env_steps_per_wall_s=83.3, pass_=False,
                                                                      threshold=600),
                G6b=dict(at_steps=1007616, pass_=True, monitor=dict(mean_survival_s=dict(policy=12.0, zero=11.5)))),
        throughput=[first, dict(session=0, wall_s=3600.0, cpu_s=1800.0)],
        sessions=[dict(index=0, started_at="2026-11-09T13:00:00+00:00", ended_at="2026-11-09T14:00:00+00:00")]))
    write(b / "policy_only" / "42" / "run.json", dict(
        mode="R4", seed=42, status="done", num_timesteps=4005888, budget=4005888, checkpoints=[4005888],
        judged_checkpoint=dict(steps=4005888, final=True, budget=4005888), g6=dict(),
        throughput=[dict(session=0, wall_s=7200.0, cpu_s=7000.0)], sessions=[dict(index=0, ended_at="x")]))
    return root


# ------------------------------------------------------------------------------------------------
# Cells, flags, verdicts
# ------------------------------------------------------------------------------------------------
def test_cells_and_their_flags():
    """GRID_RULE and FLAGS_RULE on single records: n/a off the hard floor for R0 and R1, not run, a floor with no
    settled stance, pass and FAIL with their counts and bounds, the interim flags."""
    walk, low = COND["walk/nominal"], COND["carpet/low"]
    runs = {"R3/42": dict(judged_checkpoint=dict(steps=7, final=False)),
            "R4/42": dict(judged_checkpoint=dict(steps=7, final=True))}
    assert report.cell(None, "R0", low, SRC, runs)["text"] == "n/a"
    assert report.cell(None, "R2/42", low, SRC, runs)["text"] == "not run"
    c = report.cell(panel_record("R2/42", "walk/nominal", 20, 18), "R2/42", walk, SRC, runs)
    assert c["text"] == "pass 20,18 (0.86,0.72)" and c["pass_"] and not c["flags"] and c["gated"]
    c = report.cell(panel_record("R2/43", "walk/nominal", 20, 15, label="body-limited: slip_worst", kind="success"),
                    "R2/43", walk, SRC, runs)
    assert c["text"].startswith("FAIL 20,15") and c["label"] == "body-limited: slip_worst"
    c = report.cell(panel_record("R0", "walk/nominal", 2, 2, n=2), "R0", walk, SRC, runs)
    assert c["text"].startswith("unjudged 2,2 of 2") and c["flags"] == ["not judged"] and c["text"].endswith("*")
    assert report.cell(panel_record("R1", "walk/nominal", 20, 20, sources="old"), "R1", walk, SRC,
                       runs)["flags"] == ["other panel sources"]
    c = report.cell(panel_record("R3/42", "carpet/low", 20, 20, controller=dict(checkpoint_steps=7)), "R3/42", low,
                    SRC, runs)
    assert c["flags"] == ["checkpoint provisional"] and c["checkpoint"] == 7 and c["gated"]  # decision 23: gated
    assert report.cell(panel_record("R2/42", "carpet/low", 20, 20), "R2/42", low, SRC, runs)["gated"] is False
    nr = dict(panel_record("R2/42", "stand/low", 0, 0), status="not run")
    assert report.cell(nr, "R2/42", COND["stand/low"], SRC, runs)["text"] == "no stance"  # a result, not a gap
    c = report.cell(panel_record("R4/42", "carpet/low", 20, 20, controller=dict(checkpoint_steps=7)), "R4/42", low,
                    SRC, runs)
    assert c["flags"] == [] and c["checkpoint_final"] is True  # run.json: final at the record's steps
    falls = "not attributed (falls; attribution is applied to R2, R3 rows)"
    c = report.cell(panel_record("R0", "walk/nominal", 15, 15, label=falls, kind="falls", falls_label=falls), "R0",
                    walk, SRC, runs)
    assert c["flags"] == [] and c["attribution_by_design"] and "(by design)" in report._cell_label(c)
    c = report.cell(panel_record("R2/42", "walk/nominal", 15, 15, label="not attributed (falls; run eval --attribute)",
                                 kind="falls", falls_label="not attributed (falls; run eval --attribute)"), "R2/42",
                    walk, SRC, runs)
    assert c["flags"] == ["not attributed"] and not c["attribution_by_design"]  # pending: R2 is attributed


def test_report_on_a_synthetic_run(tmp_path):
    """Every table from a synthetic run: the grid with its verdicts (R3 shown once its records exist), G7, the ledger
    with Stage 0 beside, the PPO runs, the fit and the core-hours; missing records 'not run'."""
    run = make_run(tmp_path / "run")
    doc = report.write(run, tmp_path / "out", (BODY,))
    md = (tmp_path / "out" / report.MARKDOWN).read_text()
    js = json.loads((tmp_path / "out" / report.JSON_NAME).read_text())
    assert doc["bodies"] == [BODY] and set(js["rules"]) == set(report.RULES)
    g = js["bodies"][BODY]["grid"]
    assert g["verdict_controller"] == "R3" and "R3" in g["verdict_controller_note"]
    walk = g["behaviors"]["walk"]["nominal"]
    assert walk["verdicts"]["R2"]["verdict"] == "provisional" and walk["verdicts"]["R3"]["verdict"] == "incomplete"
    assert g["behaviors"]["stand"]["hard"]["verdicts"]["R2"]["verdict"] == "shown"
    assert g["behaviors"]["walk"]["brisk"]["verdicts"]["R2"]["verdict"] == "incomplete"  # R2/43 missing
    assert g["behaviors"]["stand"]["low"]["verdicts"]["R2"]["verdict"] == "not run"  # no settled stance
    assert g["behaviors"]["carpet"]["low"]["verdicts"]["R2"]["gated"] is False  # decision 23
    assert g["gated_verdicts"]["R2"]["shown"] == 1 and g["gated_verdicts"]["R2"]["provisional"] == 1
    assert walk["cells"]["R2/42"]["text"] == "pass 20,18 (0.86,0.72)"
    assert walk["cells"]["R3/42"]["flags"] == ["checkpoint provisional"]
    assert ("| nominal | all | unjudged 2,2 of 2 (0.22,0.22) * | pass 20,20 (0.86,0.86) * | pass 20,18 (0.86,0.72) |"
            in md)
    assert "- R2/43 walk/nominal: body-limited: slip_worst" in md
    assert "- R2/42 walk/brisk: not attributed (falls; run eval --attribute); flags: not attributed" in md
    assert "| robot_floor_carpet | R3/42, R3/43, R4/42 | n/a | n/a |" in md
    assert "G7 pass: sway 0.1681 H (Stage 0 0.1680, plan 0.168)" in md
    lt = js["bodies"][BODY]["ledger"]
    assert lt["rows"][:2] == ["R2/42", "R3/42"]
    s = lt["margins"]["grip"]["rows"]["R2/42"]["summary"]["searches"]
    assert s["walk_nominal"]["success"]["threshold"] >= 0.53 > s["walk_nominal"]["success"]["fails"]
    assert s["turn"]["success"]["search"] == "turn_right"  # the worse direction: the higher mu
    assert lt["margins"]["kp"]["rows"]["R2/42"]["status"] == "not run"
    line = next(x for x in md.splitlines() if x.startswith("| Grip | walk_nominal |"))
    assert "body-limited: slip_worst" in line and line.endswith("| 0.48 |") and "| not run |" in line
    assert "| Grip | turn (worse of turn_left, turn_right) |" in md
    assert "| Push | standing_left | not run | not run | not run | not run | (forward 0.487) |" in md
    assert "the existing stop reference decelerates at 0.430 m/s^2" in md
    assert "| Stop | - | not run (plan deviation, decision 33) |" in md
    mass = next(x for x in md.splitlines() if x.startswith("| Model doubt: trunk mass | walk_nominal_trunk_x1.1 "))
    assert "(trunk x1.1: robot +4.3 %)" in mass and "works 2/3; outright fails" in mass  # WORKS_RULE beside outright
    line = next(x for x in md.splitlines() if x.startswith("| Model doubt: trunk mass | walk_nominal |"))
    assert line.count("*") == 2 and "(robot +" in line  # the search is not finished (a partial record)
    assert "| Drive speed | walk_fr05 |" in md and "0.93 (0.81 without the 15 % margin) |" in md
    assert "0.76 (walk RMS 0.76) |" in md
    assert report.CARPET_NOTE in md and "Provenance of each row" in md
    assert "- R2/42 = R2/42" in md or "- R2/42 = " in md
    p = js["bodies"][BODY]["ppo"]
    assert p["R3/42"]["judged_checkpoint"] == dict(steps=507904, final=False, budget=4005888, recorded=True)
    assert p["R3/42"]["g6b"]["pass_"] and p["R3/43"]["status"] == "not run" and p["R4/42"]["status"] == "done"
    assert ("| R3/42 | running | 507904 / 4005888 | 507904 | 507904 (provisional) | yes | 83 (FAIL) | pass at 1007616"
            in md)
    assert "Elapsed PPO wall clock" in md and "job-hours" in md
    f = js["bodies"][BODY]["fit"]
    assert f["body_level"]["cma_seed"] == 42 and f["seeds"]["43"]["conditions"]["walk"]["successes"] == 18
    assert "| 42 | done | 720 | 0.0600 | seed42_gen20 | 18/20 (surv 20) | 17/20 (surv 20) | 20/20 (surv 20) |" in md
    c = js["bodies"][BODY]["costs"]["steps"]
    assert c["prepare"]["core_h"] == pytest.approx(1.0) and c["gates"]["core_h"] == pytest.approx(0.2)
    assert c["fit"]["core_h"] == pytest.approx(4.0) and c["fit"]["evaluation_wall_h"] == pytest.approx(60 / 3600)
    assert c["g5"]["core_h"] == pytest.approx(0.5)
    assert c["ppo"]["finished_cpu_h"] == pytest.approx((1800 + 7000) / 3600) and c["ppo"]["running"] == []
    assert c["panels"]["records"] == 9 and c["panels"]["core_h"] == pytest.approx(9 * 90 / 3600)
    assert c["ledger"]["core_h"] == pytest.approx(3960 / 3600) and c["ledger"]["wall_h"] == pytest.approx(2.0)
    for b in report.BEHAVIOR_TITLES.values():
        assert f"#### {b}" in md
    assert not (run / report.MARKDOWN).exists()


def policy_controller(row, steps, final, selection="the judged checkpoint (train.judged_checkpoint)"):
    """A policy row's controller description as policy_rows writes it (the policy's checkpoint, nested)."""
    return dict(row=row, controller=f"{row}: a policy row", tracker_mode="R2" if row.startswith("R3") else None,
                policy=dict(policy="TrainedPolicy", mode=row.split("/")[0], ppo_seed=int(row.split("/")[1]),
                            checkpoint=dict(steps=steps, selection=selection, final=final, judged=None,
                                            model=dict(file=f"x/model_{steps}.zip", sha256="0" * 64),
                                            vecnormalize=dict(file=f"x/vecnormalize_{steps}.pkl", sha256="0" * 64))))


def test_the_policy_rows_cells_and_verdicts(tmp_path):
    """The R3 and R4 rows' records as policy_rows writes them: a cell names the checkpoint its record ran and is
    flagged by that checkpoint (final or not, whatever run.json's judged checkpoint says); R3's verdicts are the
    panels' on R3/42 and R3/43 (shown once both pass), R4's policy_rows.R4_VERDICT_RULE's on R4/42 (at most
    provisional), each counted over the conditions gated for it (the carpet walk too, decision 23)."""
    from designlab.stage1 import policy_rows

    assert report.SINGLE_SEED["R4"] == policy_rows.R4_ROWS and report.CONTROLLERS == ("R2", "R3", "R4")
    run = make_run(tmp_path / "run")  # run.json: R3's judged checkpoint final at the budget, R4/42's too
    for seed in (42, 43):
        path = run / BODY / "ppo" / str(seed) / "run.json"
        doc = json.loads(path.read_text()) if path.exists() else dict(mode="R3", seed=seed, status="done")
        write(path, dict(doc, judged_checkpoint=dict(steps=4005888, final=True, budget=4005888)))
    rows = {"R3/42": (4005888, True), "R3/43": (4005888, True), "R4/42": (1007616, False)}
    for key in ("walk/nominal", "carpet/low"):
        for row, (steps, final) in rows.items():
            sel = "the judged checkpoint (train.judged_checkpoint)" if final else "the run's newest complete checkpoint"
            write(panels.record_path(run, BODY, row, COND[key]),
                  panel_record(row, key, 20, 18, controller=policy_controller(row, steps, final, sel)))
    kp = {m.name: m for m in ledger.margins(BODY)}["kp"]
    write(ledger.record_path(run, BODY, "R3/42", "kp"), dict(
        row="R3/42", margin="kp", status="done", cut_s=None, halvings=7, definition=kp.describe(),
        summary=ledger.margin_summary(kp, {"walk_nominal": bisected(kp.searches[0], 0.6)}, {}),
        cost=dict(cpu_s=60.0, wall_s=90.0),
        controller=policy_controller("R3/42", 1007616, False, "the run's newest complete checkpoint")))
    doc = report.collect(run, (BODY,))
    g = doc["bodies"][BODY]["grid"]
    walk, carpet = g["behaviors"]["walk"]["nominal"], g["behaviors"]["carpet"]["low"]
    c3, c4 = walk["cells"]["R3/42"], walk["cells"]["R4/42"]
    assert c3["checkpoint"] == 4005888 and c3["checkpoint_final"] is True and c3["flags"] == []
    assert c4["checkpoint"] == 1007616 and c4["checkpoint_final"] is False and c4["flags"] == ["checkpoint provisional"]
    assert walk["verdicts"]["R3"]["verdict"] == "shown" and walk["verdicts"]["R4"]["verdict"] == "provisional"
    assert walk["verdicts"]["R4"]["read_from"] == "shown" and walk["shown"] == "shown"  # R3's verdicts are shown
    assert carpet["verdicts"]["R3"]["gated"] and carpet["verdicts"]["R4"]["gated"]
    assert not carpet["verdicts"]["R2"]["gated"] and carpet["cells"]["R4/42"]["gated"]
    assert g["verdict_controller"] == "R3" and g["gated_verdicts"]["R3"]["shown"] == 2
    assert g["gated_verdicts"]["R4"]["provisional"] == 2 and "shown" not in g["gated_verdicts"]["R4"]
    assert g["single_seed"]["R4"]["pass"] == 2
    assert walk["verdicts"]["R4"]["verdict"] == policy_rows.r4_verdict(
        [json.loads(panels.record_path(run, BODY, "R4/42", COND["walk/nominal"]).read_text())])["verdict"]
    assert g["behaviors"]["stand"]["hard"]["verdicts"]["R4"]["verdict"] == "not run"
    md = report.markdown(doc)
    assert "| R2 verdict | R3 verdict (shown) | R4 verdict |" in md
    line = next(x for x in md.splitlines() if x.startswith("| nominal | all |"))
    assert line.endswith("| provisional | shown | provisional |") and "pass 20,18 (0.86,0.72) * |" in line
    assert "- R4/42 walk/nominal: flags: checkpoint provisional; checkpoint 1007616" in md
    assert next(x for x in md.splitlines() if x.startswith("| R4 |")).startswith("| R4 | 0 | 2 |")
    assert "R4 has one seed (decisions 3 and 28): its verdicts are at most provisional" in md
    lt = doc["bodies"][BODY]["ledger"]  # the ledger's R3 row names its checkpoint too
    assert lt["rows"][:2] == ["R2/42", "R3/42"] and lt["checkpoints"] == {"R3/42": [dict(steps=1007616, final=False)]}
    assert lt["margins"]["kp"]["rows"]["R3/42"]["checkpoint"] == dict(steps=1007616, final=False)
    assert "R3/42 = R3/42: a policy row; checkpoint 4005888 steps (final)" in md  # PROVENANCE_RULE
    assert "- R3/42 ran checkpoint 1007616 (provisional) *: not the final judged checkpoint (FLAGS_RULE)" in md


def test_a_running_ppo_session_is_estimated(tmp_path):
    """A session still running has no CPU total: its wall to the record's last write x its first rollouts' CPU per
    wall second."""
    run = make_run(tmp_path / "run")
    path = run / BODY / "ppo" / "42" / "run.json"
    doc = json.loads(path.read_text())
    doc["sessions"].append(dict(index=1, started_at="2026-11-09T15:00:00+00:00"))
    path.write_text(json.dumps(doc))
    t1 = datetime.datetime(2026, 11, 9, 17, tzinfo=datetime.timezone.utc).timestamp()  # two hours after its start
    os.utime(path, (t1, t1))
    t = report.ppo(run, BODY)["R3/42"]["times"]
    assert t["running_wall_s"] == pytest.approx(7200.0) and t["cpu_per_wall_first_rollouts"] == pytest.approx(0.5)
    c = report.costs(run, BODY)["steps"]["ppo"]
    assert c["running"] == ["R3/42"] and c["running_estimate_h"] == pytest.approx(1.0)
    assert c["core_h"] == pytest.approx((1800 + 7000) / 3600 + 1.0)
    # PPO_RULE: the elapsed wall clock is the union of the sessions, not the sum of the concurrent jobs' hours
    write(run / BODY / "ppo" / "43" / "run.json", dict(mode="R3", seed=43, status="partial", sessions=[dict(
        index=0, started_at="2026-11-09T13:30:00+00:00", ended_at="2026-11-09T14:30:00+00:00")]))
    w = report.ppo_wall(run, BODY)
    assert w["elapsed_s"] == pytest.approx(3.5 * 3600) and w["job_s"] == pytest.approx(4 * 3600)
    assert report.costs(run, BODY)["steps"]["ppo"]["wall_h"] == pytest.approx(3.5)


def test_a_stale_run_and_an_unrecorded_judged_checkpoint(tmp_path):
    """PPO_RULE: a 'running' run whose run.json has not been written for two checkpoint intervals at its measured
    rate is marked 'stopped?'; with no judged checkpoint in run.json the table shows train.judged_checkpoint's now,
    'not yet recorded'; each G6 failure shows the plan's consequence."""
    run = make_run(tmp_path / "run")
    path = run / BODY / "ppo" / "42" / "run.json"
    doc = json.loads(path.read_text())
    doc.pop("judged_checkpoint")
    doc["g6"]["G6b"] = dict(at_steps=1007616, pass_=False, consequence_if_failed="make one fix, then stop")
    doc["fix"] = dict(from_checkpoint=1007616, overrides=dict(task_weights=[1, 1, 1, 1, 1, 2, 2, 1, 1]))
    now = datetime.datetime.now(datetime.timezone.utc)
    t = now - datetime.timedelta(seconds=4000)  # the last write; the session trained 507904 steps in 5079.04 s
    doc["sessions"] = [dict(index=0, started_at=(t - datetime.timedelta(seconds=5079.04)).isoformat())]
    path.write_text(json.dumps(doc))
    os.utime(path, (t.timestamp(), t.timestamp()))
    r = report.ppo(run, BODY, now=now)["R3/42"]  # 100 steps/s: a checkpoint every 2500 s, 'stopped?' after 7500 s
    assert r["write"]["stopped"] is False and r["write"]["limit_s"] == pytest.approx(7500.0)
    t = now - datetime.timedelta(seconds=8000)
    doc["sessions"] = [dict(index=0, started_at=(t - datetime.timedelta(seconds=5079.04)).isoformat())]
    path.write_text(json.dumps(doc))
    os.utime(path, (t.timestamp(), t.timestamp()))
    p = report.ppo(run, BODY, now=now)
    assert p["R3/42"]["write"]["stopped"] is True and p["R3/42"]["judged_checkpoint"]["recorded"] is False
    md = "\n".join(report._ppo_md(p))
    assert "| R3/42 | running (stopped?) |" in md and "(provisional, not yet recorded)" in md
    assert "FAIL at 1007616: make one fix, then stop" in md and "from 1007616 (task_weights 1,1,1,1,1,2,2,1,1)" in md


def test_a_body_judged_with_r2(tmp_path):
    """VERDICTS_RULE: when seed 42 failed G6c the grid shows R2's verdicts, flagged, and R3's column reads 'not
    judged'; before G6c the note says 'G6c not reached'."""
    run = make_run(tmp_path / "run")
    g = report.grid(run, BODY, SRC)
    assert g["verdict_controller"] == "R3" and "G6c not reached" in g["verdict_controller_note"]
    path = run / BODY / "ppo" / "42" / "run.json"
    doc = json.loads(path.read_text())
    doc["g6"]["G6c"] = dict(at_steps=2506752, pass_=False, consequence_if_failed="skip seed 43 and judge with R2")
    path.write_text(json.dumps(doc))
    g = report.grid(run, BODY, SRC)
    assert g["verdict_controller"] == "R2" and "G6c at 2506752" in g["verdict_controller_note"]
    walk = g["behaviors"]["walk"]["nominal"]
    assert walk["shown"] == walk["verdicts"]["R2"]["verdict"] and walk["verdicts"]["R3"]["judged"] is False
    assert "R3" not in [k for k, v in g["gated_verdicts"].items() if v]
    md = "\n".join(report._grid_md(g))
    assert "R3 verdict (not judged: G6c)" in md and "R2 verdict (shown)" in md
    lt = report.ledger_table(run, BODY)
    assert "G6c" in lt["r3_judged_with_r2"] and "its ledger row is not run" in "\n".join(report._ledger_md(lt))


def test_looks_and_ledger_flags(tmp_path, monkeypatch):
    """LOOKS_RULE: sway and head and tail amplitude per row on the walks and carpet walks; LEDGER_TABLE_RULE: a record
    made by other ledger sources is flagged, and so is a search that fails at nominal where the panels pass."""
    run = make_run(tmp_path / "run")
    rec = json.loads(panels.record_path(run, BODY, "R2/42", COND["walk/nominal"]).read_text())
    look = dict(sway=dict(mean_H=0.17, max_H=0.18), head=dict(amplitude_rad=0.05), tail=dict(amplitude_rad=0.12))
    rec["episodes"] = [dict(seed=s, measures=dict(look=look)) for s in panels.SEEDS[:2]]
    rec["episodes"].append(dict(seed=3044, measures={}))
    write(panels.record_path(run, BODY, "R2/42", COND["walk/nominal"]), rec)
    lk = report.looks(run, BODY)["conditions"]["walk/nominal"]["R2/42"]
    assert lk["episodes"] == 3 and lk["sway"]["read"] == 2 and lk["sway"]["mean_H"] == pytest.approx(0.17)
    assert lk["head"]["in_band"] == 2 and lk["tail"]["mean_rad"] == pytest.approx(0.12)
    assert "| walk/nominal | R2/42 | 0.170 / 0.180 |" in "\n".join(report._looks_md(report.looks(run, BODY)))
    path = ledger.record_path(run, BODY, "R2/42", "grip")
    doc = json.loads(path.read_text())
    doc["key"] = dict(ledger="an older ledger.py")
    s = doc["summary"]["searches"]["walk_brisk"]
    s["success"] = dict(s["success"], status="fails at nominal", flag="the panels pass walk/brisk (20/20 survive, 17 "
                                                                     "succeed), the ledger's 3 seeds fail at nominal")
    path.write_text(json.dumps(doc))
    lt = report.ledger_table(run, BODY)
    assert lt["margins"]["grip"]["rows"]["R2/42"]["flags"] == ["other ledger sources"]
    assert any("other ledger sources" in n for n in lt["notes"]) and any("panels pass" in n for n in lt["notes"])
    md = "\n".join(report._ledger_md(lt))
    assert "- * R2/42 grip: other ledger sources" in md


def test_report_command_line(tmp_path, capsys):
    run = make_run(tmp_path / "run")
    assert cli(["report", "--out", str(run), "--dest", str(tmp_path / "d"), "--body", BODY]) == 0
    assert "STAGE1_TABLES.md" in capsys.readouterr().out and (tmp_path / "d" / report.JSON_NAME).exists()


@pytest.mark.skipif(not (REAL / BODY / "tune.json").exists(), reason="no real run directory")
def test_report_on_the_real_run_reads_only(tmp_path, monkeypatch):
    """On the real (partial) run directory: every body's tables, missing records 'not run', the fit's G5 as recorded;
    every file the report writes is under the destination."""
    written = []
    real_write, real_replace = Path.write_text, os.replace

    def write_text(self, *a, **k):
        written.append(Path(self).resolve())
        return real_write(self, *a, **k)

    def replace(src, dst, *a, **k):
        written.append(Path(dst).resolve())
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(panels, "_write_json", lambda *a, **k: pytest.fail("the report wrote a record"))
    doc = report.write(REAL, tmp_path)
    dest = tmp_path.resolve()
    assert written and all(dest in w.parents for w in written), written
    md = (tmp_path / report.MARKDOWN).read_text()
    js = json.loads((tmp_path / report.JSON_NAME).read_text())
    assert set(doc["bodies"]) <= set(ledger.BODIES) and BODY in doc["bodies"]
    for body in doc["bodies"]:
        assert f"## {body}" in md
        b = js["bodies"][body]
        assert b["fit"]["status"] == "done" and b["fit"]["pass_"] is True
        assert set(b["ppo"]) == {"R3/42", "R3/43", "R4/42"} and set(b["ledger"]["margins"]) == set(ledger.MARGIN_NAMES)
        assert len([c for beh in b["grid"]["behaviors"].values() for c in beh.values()]) == len(panels.conditions())
    assert "not run" in md and "passes G5" in md and "### Core-hours per step" in md
