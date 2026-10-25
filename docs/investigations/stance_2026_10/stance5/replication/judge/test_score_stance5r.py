"""score_stance5r.py on hand-made judge files of the two new runs, beside the real stance5 runs and the real control.

    cd R/judge && python3 -I -B -m pytest -p no:cacheprovider -q test_score_stance5r.py --basetemp=<work>/tmp

(-B: no __pycache__ is written next to the scorer or this file.)

Covers: every decision-table case (both new runs hold: line 1, ADOPT, H = 3; exactly one holds, either way round:
line 2; neither: line 3) with each K test failing in turn; a missing new J file before the runs end (incomplete, the
parent block inputs only) and after (counted as a fail: line 2 or 3, listed), both runs missing entirely; a missing
report-only file (no effect); invalid new J files (a stance5 judge file, a sigma other than the sidecar's) before the
runs end (provisional) and after (a fail); a placed but unjudged J checkpoint after the runs end (provisional); the
criteria, judge, task and HEAD pins and placeholders; --validation; the refusal when the stance5 runs' recorded
verdicts do not reproduce (score_final.json's pin, a tampered record, a changed stance5 judge file, a record that does
not read line 3); --validate on the real stance5 files; the sidecar (the recomputed mean, a tampered copy refused, a
wrong sha256 refused, frozen_params' own entries untouched); the report-only K3 fraction and per-test fractions; the
parent comparison over five finals and a tie with s52; control-check problems as notes, not provisional; each decision
line verbatim in criteria.txt; the pins equal the files as they stand; judge_diff.txt is the diff of the two judges;
the validation judge files (when present) bind to the sidecar and reproduce judge_stance5.py's counts; the command line.
Run handling (criteria RUN HANDLING): a judge refusal record counts as an invalid J file (provisional before the runs
end, a fail after; never "missing"; a placed checkpoint with a record is not "placed but unjudged"); a record of another
slot, of another judge, or beside a judge file is provisional; an early stop (gate_verdict.json below 12,001,280) makes
the final J file invalid, a judge file there included, and its placed stage1_final is not waiting for a judge; a
gate_verdict.json at 12,001,280 changes nothing; the early stop is read from the placed gate_verdict.json under --ledgers
too; a source_folder.json is a note; records placed with place5r.py --run-records-fail and a refusal held for review are
notes, and a held review counts for nothing (its placed checkpoint is placed but unjudged); the refusal message names
what follows a refusal; judge_one5r.sh writes the record schema the scorer reads.  The tools' side of the run-handling
rules is tested in test_tools_run_handling.py.

The hand-made files carry the real HEAD (cf80ecb1) and the band-on task, and a test judge sha256, so an autouse fixture
sets JUDGE_SHA256_S5R to it; the decision tests score against a synthetic criteria file with a matching pin.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_stance5r as SR  # noqa: E402

FILE_PINS = {k: getattr(SR, k) for k in ("CRITERIA_SHA256_R", "JUDGE_SHA256_S5R", "SIDECAR_SHA256", "HEAD_LTSB")}
S5, SC3 = SR.S5, SR.SC3
FROZEN = json.load(open(SC3.FROZEN_PARAMS))
SIDECAR = json.load(open(SR.SIDECAR))
TEST_JUDGE_SHA = "sha256:" + "b" * 64
HEAD, TASK = SR.HEAD_LTSB, SR.TASK_SHA256_LTSB
SLOTS = {**dict(SC3.J), **dict(S5.REPORT_1M)}
R4, R5 = (r[0] for r in SR.NEW_RUNS)
SEED = {run: seed for run, _arm, seed, _name in SR.NEW_RUNS}
S52, S60, S42 = (r[0] for r in SR.STANCE5_RUNS)
JUDGE5 = S5.S5DIR + "/judge/judge_stance5.py"
JUDGE5R = os.path.join(HERE, "judge_stance5r.py")
VALID = os.path.join(HERE, "validation")


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    monkeypatch.setattr(SR, "JUDGE_SHA256_S5R", TEST_JUDGE_SHA)


# ------------------------------------------------------------------------------------------------ synthetic files
def episode(seed, *, full=True, clean=True, pad=0.4, w=(0.1, 0.1), xc=0.066, tilt=0.5, lift=0.001):
    reasons = []
    if not full:
        reasons.append("horizon: 412 < 1000")
    if pad > 0.8:
        reasons.append(f"max_sole_cop_fore_aft: {pad} > 0.8")
    if tilt > 2.0:
        reasons.append(f"max_sole_tilt_deg: {tilt} > 2.0")
    if not clean and not reasons:
        reasons.append("min_foot_load_share: 0.3 < 0.4")
    return {"seed": seed, "length": 1000 if full else 412, "reached_horizon": full, "clean": not reasons,
            "reasons": reasons, "reward": 3900.0, "termination_reason": "truncated" if full else "nosedive",
            "max_sole_cop_fore_aft": pad, "max_foot_cop_fore_aft": 0.3, "max_sole_tilt_deg": tilt,
            "max_sole_corner_lift_m": lift, "foot_cop_fore_aft": {"r": w[0], "l": w[1]},
            "xc_both_feet_m": xc if full else None, "com_mtp3_head_m": xc if full else None,
            "com_fore_aft_footsite_m": -0.016 if full else None}


def eps_with(n_bad, **alter):
    return [episode(3042 + i, **(alter if i < n_bad else {})) for i in range(40)]


def noise(seed, space, falls, sigma_seed=None):
    rows = [{"seed": 3042 + i, "length": 1000 if i >= falls else 300, "reached_horizon": i >= falls, "reward": 3800.0}
            for i in range(20)]
    d = {"n": 20, "falls": falls, "seeds": [3042, 3061], "space": space, "episodes_detail": rows}
    if space == "joint":
        s = str(sigma_seed or seed)
        ref = SIDECAR["control_sigma"][s] if s in SR.SIDECAR_SEEDS else FROZEN["criteria"]["H4"]["control_sigma"][s]
        d.update({"sigma": "control", "sigma_source": ref["source"], "sigma_vector": list(ref["vector"])})
    else:
        d.update({"sigma": "own", "sigma_source": "own", "sigma_vector": [0.1] * 15})
    return d


def judge(seed, label, *, eps=None, joint_falls=0, own_falls=0, schema=SR.JUDGE_SCHEMA_S5R, jsha=TEST_JUDGE_SHA,
          task=TASK, sigma_seed=None):
    """A valid stance5r judge file of a new run (*seed*) at slot *label*: 40 clean, centred episodes unless *eps*."""
    eps = eps if eps is not None else [episode(3042 + i) for i in range(40)]
    k = sum(e["clean"] for e in eps)
    full = sum(e["reached_horizon"] for e in eps)
    hop = sum(not e["reached_horizon"] for e in eps)
    lcb = SC3.binomial_lcb(k, 40)
    verdict = "PASS" if (k >= 37 and lcb >= 0.8 and full >= 38 and hop <= 1) else "FAIL"
    steps = SLOTS.get(label, label)
    stem = "stage1_final" if steps == "final" else f"{steps}_model" if steps in ("best", "robust_best") else f"stage1_{steps}_steps"
    d = {"schema": schema, "mode": "full", "smoke": False, "arm": "LTSB", "seed": seed, "steps": steps, "checkpoint": stem,
         "judge_sha256": jsha, "git": {"head": HEAD, "status": "", "branch": "x"}, "model_sha256": f"sha256:{seed}{steps}",
         "plant_identity": {"physics_sha256": SC3.R9_PHYSICS_SHA256, "physics_revision": 9},
         "arm_env_kwargs": {**SC3.ARM_ENV_COMMON, **S5.ARM_ENV_LTSB},
         "checks": {"relaxed": {"no_symmetry_check": False, "task_binding": "require", "allow_missing_records": False,
                                "allow_seed_mismatch": False, "allow_commit_mismatch": False},
                    "hard_failures": [], "plant_physics_sha256": {"match": True}, "symmetry": {"mirror_map_sha256_match": True},
                    "seed": {"label": seed, "checkpoint": seed, "match": True},
                    "checkpoint_num_timesteps": SC3.FINAL_TIMESTEPS if steps == "final" else steps,
                    "stage_config": {"git_commit_match": True, "git_commit": HEAD, "clone_head": HEAD, "run_seed": seed,
                                     "run_n_envs": SC3.N_ENVS, "run_hyperparameters_sha256": SC3.HP_SHA256}},
         "panel": {"verdict": verdict, "passed": verdict == "PASS", "n_clean": k, "clean_lcb": lcb, "n_full_horizon": full,
                   "hop_or_fall_episodes": hop, "mean_reward": 3900.0, "statue_mean_reward": 3980.0,
                   "terminations": {"truncated": full}, "scored_gate_kind": "stance_quality/v2", "task_sha256": task,
                   "thresholds": dict(SC3.R9_BARS), "statue_on_same_seeds": {"n": 40},
                   "statue_episodes_detail": [episode(3042 + i, full=(i != 39), xc=0.065) for i in range(40)],
                   "summary": {}, "episodes_detail": eps}}
    if label in ("6.3M", "final"):
        d["noise_matched_control_joint"] = noise(seed, "joint", joint_falls, sigma_seed)
        d["noise_matched_own"] = noise(seed, "policy", own_falls)
    return d


#: How each named failure is made: K1 4 unclean at 6.3M; K2 4 split episodes at 8.4M (K2b); K3 3/20 joint falls at
#: final; K3own 2/20 own falls at final; K4 4 tilted episodes at final (K1 fails with it).
FAILS = {
    "K1": ("6.3M", lambda: {"eps": eps_with(4, clean=False)}),
    "K2": ("8.4M", lambda: {"eps": eps_with(4, w=(0.7, -0.3))}),
    "K3": ("final", lambda: {"joint_falls": 3}),
    "K3own": ("final", lambda: {"own_falls": 2}),
    "K4": ("final", lambda: {"eps": eps_with(4, tilt=2.01)}),
}


def cells(fail: dict | None = None, at: dict | None = None, drop: dict | None = None, runs=(R4, R5)) -> dict:
    """{run: {label: judge file or None}}: fail = {run: [FAILS names]}, at = {run: {label: judge kwargs}},
    drop = {run: [labels left out]}."""
    out = {}
    for run in runs:
        kw = {lab: {} for lab, _ in SC3.J}
        for name in (fail or {}).get(run, []):
            lab, make = FAILS[name]
            kw[lab].update(make())
        for lab, extra in ((at or {}).get(run) or {}).items():
            kw.setdefault(lab, {}).update(extra)
        out[run] = {lab: (None if lab in (drop or {}).get(run, []) else judge(SEED[run], lab, **kw[lab])) for lab in kw}
    return out


def write_tree(root, tree: dict) -> str:
    for run, slots in tree.items():
        os.makedirs(os.path.join(root, run), exist_ok=True)
        for lab, d in slots.items():
            if d is not None:
                with open(os.path.join(root, run, f"judge_{SLOTS.get(lab, lab)}.json"), "w") as fh:
                    json.dump(d, fh)
    return str(root)


def refusal(run, steps, *, jsha=TEST_JUDGE_SHA, **over):
    """A judge refusal record as judge_one5r.sh writes it (the run's own records: a seed mismatch)."""
    text = f"{run}/01_stance/stage_config.json: run seed 52 != --seed {SEED[run]}"
    d = {"schema": SR.REFUSAL_SCHEMA, "run": run, "seed": SEED[run], "steps": steps, "mode": "full", "refusal": text,
         "items": [text], "judge_sha256": jsha, "judge_rc": 1}
    d.update(over)
    return d


def score(tmp, tree, *, runs_ended=False, pin="match", validation=False, head=None, place=None, refusals=None,
          gate=None, source=None, **kw):
    """Write *tree* under *tmp* and score it beside the real stance5 runs and control; a synthetic criteria file with a
    matching pin unless *pin* says otherwise.  place = {run: [steps]} puts a checkpoint pair under the runs root;
    refusals = {run: {steps: record}} writes judged/<run>/refused_<steps>.json; gate = {run: timesteps} writes the run's
    01_stance/gate_verdict.json (stage_result.timesteps); source = {run: folder} writes runs/<run>/source_folder.json."""
    os.makedirs(tmp / "judged", exist_ok=True)
    judged = write_tree(tmp / "judged", tree)
    for run, recs in (refusals or {}).items():
        os.makedirs(tmp / "judged" / run, exist_ok=True)
        for steps, d in recs.items():
            (tmp / "judged" / run / f"refused_{steps}.json").write_text(json.dumps(d))
    for run, ts in (gate or {}).items():
        os.makedirs(tmp / "runs" / run / "01_stance", exist_ok=True)
        (tmp / "runs" / run / "01_stance" / "gate_verdict.json").write_text(json.dumps({"stage_result": {"timesteps": ts}}))
    for run, folder in (source or {}).items():
        os.makedirs(tmp / "runs" / run, exist_ok=True)
        (tmp / "runs" / run / "source_folder.json").write_text(json.dumps({"run_id": run, "source_folder": folder}))
    crit = tmp / "criteria.txt"
    crit.write_text("synthetic replication criteria\n")
    sha = {"match": hashlib.sha256(crit.read_bytes()).hexdigest(), "placeholder": None, "mismatch": "0" * 64}[pin]
    runs_root = tmp / "runs"
    for run, steps_list in (place or {}).items():
        m = runs_root / run / "01_stance" / "models"
        os.makedirs(m, exist_ok=True)
        for s in steps_list:
            names = (("stage1_final.zip", "stage1_final_vecnorm.pkl") if s == "final" else
                     (f"stage1_{s}_steps.zip", f"stage1_vecnormalize_{s}_steps.pkl"))
            for n in names:
                (m / n).write_bytes(b"x")
    os.makedirs(runs_root, exist_ok=True)
    res = SR.score(judged, runs_root=str(runs_root), head_ltsb=HEAD if head is None else head, criteria_file=str(crit),
                   criteria_sha256=sha, runs_ended=runs_ended, validation=validation, **kw)
    SR.render(res)  # renders without error
    return res


def line(res):
    return res["decision"]["line"]


def status(res):
    return res["decision"]["status"]


# ------------------------------------------------------------------------------------------------ the decision table
def test_both_hold_adopt(tmp_path):
    res = score(tmp_path, cells())
    d = res["decision"]
    assert (line(res), status(res), d["adopt"], d["H"], d["of"]) == (1, "decided", True, 3, 5)
    assert d["new_holds"] == {R4: True, R5: True}
    assert d["stance5_holds_recorded"] == {S52: True, S60: False, S42: False}
    assert d["text"] == SR.DECISION_LINES_R[1]
    assert "ADOPT" in SR.render(res)
    res2 = score(tmp_path / "ended", cells(), runs_ended=True)
    assert (line(res2), status(res2)) == (1, "decided")


@pytest.mark.parametrize("which", ["K1", "K2", "K3", "K3own", "K4"])
def test_one_holds_line_2(tmp_path, which):
    res = score(tmp_path, cells(fail={R5: [which]}))
    assert (line(res), status(res), res["decision"]["H"], res["decision"]["adopt"]) == (2, "decided", 2, False)
    want = {"K3own": ["K3"], "K4": ["K1", "K4"]}.get(which, [which])
    assert res["targets"][R5]["failed"] == want
    text = SR.render(res)
    assert f"next-round inputs, {R5}" in text and f"next-round inputs, {R4}" not in text


def test_one_holds_line_2_other_way_round(tmp_path):
    res = score(tmp_path, cells(fail={R4: ["K1"]}))
    assert (line(res), res["decision"]["new_holds"]) == (2, {R4: False, R5: True})
    assert res["targets"][R4]["failed"] == ["K1"] and res["targets"][R4]["k1_failure_modes"]


def test_neither_holds_line_3(tmp_path):
    res = score(tmp_path, cells(fail={R4: ["K2"], R5: ["K3"]}))
    assert (line(res), status(res), res["decision"]["H"]) == (3, "decided", 1)
    assert res["decision"]["text"] == SR.DECISION_LINES_R[3]
    text = SR.render(res)
    assert f"next-round inputs, {R4}" in text and f"next-round inputs, {R5}" in text


def test_decision_line_function_is_exhaustive():
    assert [SR.decision_line_r({R4: a, R5: b}) for a in (True, False) for b in (True, False)] == [1, 2, 2, 3]


# ------------------------------------------------------------------------------------------------ missing and invalid
def test_missing_before_and_after_the_runs_end(tmp_path):
    tree = cells(drop={R5: ["final"]})
    res = score(tmp_path / "a", tree)
    assert (status(res), line(res)) == ("incomplete", None)
    assert res["parents"]["inputs_only"]
    res = score(tmp_path / "b", tree, runs_ended=True)
    assert (line(res), status(res)) == (2, "decided")
    assert any(p.endswith(f"{R5}/judge_final.json") for p in res["decision"]["counted_as_fails"])
    assert res["new"][R5]["criteria"]["failed"] == ["K1", "K2", "K3", "K4"]   # K2 reads every J checkpoint
    assert not res["parents"]["inputs_only"]


def test_next_round_line_names_missing_files_and_an_uncomputed_alarm(tmp_path):
    res = score(tmp_path / "a", cells(drop={R5: ["final"]}), runs_ended=True)
    t = res["targets"][R5]
    assert t["j_files_missing_or_invalid"] == ["judge_final.json"] and t["saturation_alarm_computed"] is False
    row = next(x for x in SR.render(res).splitlines() if x.startswith(f"   next-round inputs, {R5}:"))
    assert "J files missing or invalid: judge_final.json" in row and "saturation alarm not computed (" in row
    res = score(tmp_path / "b", cells(at={R4: {"6.3M": {"schema": "stance5-judge/v1"}}}), runs_ended=True)
    assert res["targets"][R4]["j_files_missing_or_invalid"] == ["judge_6291456.json"]
    res = score(tmp_path / "c", cells(fail={R5: ["K3"]}), runs_ended=True)
    row = next(x for x in SR.render(res).splitlines() if x.startswith(f"   next-round inputs, {R5}:"))
    assert res["targets"][R5]["j_files_missing_or_invalid"] == [] and "J files missing or invalid" not in row


def test_both_runs_missing_entirely(tmp_path):
    res = score(tmp_path / "a", cells(runs=()))
    assert (status(res), line(res)) == ("incomplete", None)
    res = score(tmp_path / "b", cells(runs=()), runs_ended=True)
    assert (line(res), status(res), res["decision"]["H"]) == (3, "decided", 1)
    assert len(res["decision"]["counted_as_fails"]) == 12


def test_missing_report_only_file_changes_nothing(tmp_path):
    tree = cells()
    for run in (R4, R5):
        tree[run]["7.3M"] = None
    assert line(score(tmp_path, tree)) == 1


def test_invalid_file_provisional_then_fail(tmp_path):
    tree = cells(at={R4: {"6.3M": {"schema": "stance5-judge/v1"}}})   # a stance5 judge's file in a replication slot
    res = score(tmp_path / "a", tree)
    assert status(res) == "provisional" and line(res) == 2
    assert any("invalid new-run J judge files" in w for w in res["decision"]["provisional_why"])
    assert any("schema" in i for x in res["new"][R4]["invalid"] for i in x["issues"])
    res = score(tmp_path / "b", tree, runs_ended=True)
    assert (status(res), line(res)) == ("decided", 2)
    assert res["decision"]["counted_as_fails"] and not res["new"][R4]["criteria"]["HOLDS"]


@pytest.mark.parametrize("label", ["6.3M", "final"])
def test_noise_sigma_must_be_the_sidecar(tmp_path, label):
    tree = cells(at={R5: {label: {"sigma_seed": 52}}})   # s52's frozen sigma instead of the sidecar's mean
    res = score(tmp_path, tree, runs_ended=True)
    bad = [i for x in res["new"][R5]["invalid"] for i in x["issues"]]
    assert "H4 noise configuration: control joint sigma (frozen, this seed)" in bad
    assert line(res) == 2 and not res["decision"]["new_holds"][R5]


def test_placed_but_unjudged(tmp_path):
    tree = cells(drop={R4: ["8.4M"]})
    res = score(tmp_path, tree, runs_ended=True, place={R4: [8388608]})
    assert status(res) == "provisional" and line(res) == 2
    assert any("placed under the runs root but not judged" in w for w in res["decision"]["provisional_why"])
    assert res["decision"]["placed_unjudged"] == [f"{R4}/stage1_8388608_steps.zip"]


# ------------------------------------------------------------------------------------------------ run handling
def test_refusal_record_counts_as_an_invalid_j_file(tmp_path):
    """criteria RUN HANDLING (rule 2): a judge refused on the run's own records is an INVALID J file, not a missing one."""
    tree = cells(drop={R4: ["6.3M"]})
    rec = {R4: {6291456: refusal(R4, 6291456)}}
    # without the record the slot is missing: incomplete before the runs end
    assert status(score(tmp_path / "none", tree)) == "incomplete"
    res = score(tmp_path / "a", tree, refusals=rec, place={R4: [6291456]})
    r4 = res["new"][R4]
    assert (status(res), line(res)) == ("provisional", 2)   # invalid before the runs end: provisional, not incomplete
    assert not res["decision"]["missing"] and res["decision"]["new_holds"] == {R4: False, R5: True}
    assert [os.path.basename(x["path"]) for x in r4["invalid"]] == ["refused_6291456.json"]
    assert any("refused on the run's own records" in i for x in r4["invalid"] for i in x["issues"])
    assert any("invalid new-run J judge files" in w for w in res["decision"]["provisional_why"])
    assert res["decision"]["placed_unjudged"] == []   # the placed 6.3M pair is judged: its judge refused
    assert r4["criteria"]["failed"] == ["K1", "K2", "K3", "K4"]   # each needs 6.3M (K2 reads every J checkpoint)
    assert res["targets"][R4]["j_files_missing_or_invalid"] == ["refused_6291456.json"]
    assert "judge refusal recorded, an invalid J file (refused_6291456.json)" in SR.render(res)
    res = score(tmp_path / "b", tree, refusals=rec, place={R4: [6291456]}, runs_ended=True)
    assert (status(res), line(res), res["decision"]["H"]) == ("decided", 2, 2)
    assert [os.path.basename(p) for p in res["decision"]["counted_as_fails"]] == ["refused_6291456.json"]
    # both new runs refused at a J slot: line 3
    res = score(tmp_path / "c", cells(drop={R4: ["4.2M"], R5: ["final"]}), runs_ended=True,
                refusals={R4: {4194304: refusal(R4, 4194304)}, R5: {"final": refusal(R5, "final")}})
    assert (status(res), line(res)) == ("decided", 3)


def test_refusal_record_problems_are_provisional(tmp_path):
    tree = cells(drop={R5: ["8.4M"]})
    cases = {"other_slot": {R5: {8388608: refusal(R5, 8388608, seed=70)}},     # R5 is seed 80
             "other_judge": {R5: {8388608: refusal(R5, 8388608, jsha="sha256:" + "c" * 64)}},
             "unreadable": None}
    for name, recs in cases.items():
        if recs is None:
            d = tmp_path / name / "judged" / R5
            os.makedirs(d)
            (d / "refused_8388608.json").write_text("{not json")
        res = score(tmp_path / name, tree, refusals=recs, runs_ended=True)
        why = res["decision"]["provisional_why"]
        assert status(res) == "provisional" and line(res) == 2, name
        assert any("refused_8388608.json" in w for w in why), (name, why)
        assert not res["decision"]["new_holds"][R5]
    # a record beside a valid judge file of the same slot: provisional (one of them must be moved aside)
    res = score(tmp_path / "both", cells(), refusals={R5: {8388608: refusal(R5, 8388608)}}, runs_ended=True)
    assert status(res) == "provisional" and any("both judge_8388608.json and refused_8388608.json" in w
                                                for w in res["decision"]["provisional_why"])


def test_early_stop_makes_the_final_invalid(tmp_path):
    """criteria RUN HANDLING (rule 5): gate_verdict.json written before 12,001,280 is an early stop; its stage1_final is
    not the final slot, so the final J file is invalid (never missing), and its placed pair waits for no judge."""
    early = {R5: 9437184}
    tree = cells(drop={R5: ["final"]})
    res = score(tmp_path / "a", tree, gate=early, place={R5: ["final"]})
    r5 = res["new"][R5]
    assert (status(res), line(res)) == ("provisional", 2)   # not "incomplete": the final will never come
    assert r5["early_stop"]["early"] and r5["early_stop"]["timesteps"] == 9437184
    assert [os.path.basename(x["path"]) for x in r5["invalid"]] == ["judge_final.json"]
    assert any("stopped early at 9,437,184" in i for x in r5["invalid"] for i in x["issues"])
    assert res["decision"]["placed_unjudged"] == []
    assert "early stop at 9,437,184 (gate_verdict.json): the final J file is invalid" in SR.render(res)
    res = score(tmp_path / "b", tree, gate=early, place={R5: ["final"]}, runs_ended=True)
    assert (status(res), line(res)) == ("decided", 2)
    assert [os.path.basename(p) for p in res["decision"]["counted_as_fails"]] == ["judge_final.json"]
    assert res["new"][R5]["criteria"]["failed"] == ["K1", "K2", "K3", "K4"]
    # a judge file in the final slot does not save an early-stopped run (its stage1_final is not the final slot)
    res = score(tmp_path / "c", cells(), gate=early, runs_ended=True)
    assert (status(res), line(res)) == ("decided", 2) and not res["decision"]["new_holds"][R5]
    assert res["new"][R5]["cells"]["final"] is None and "final" not in res["new"][R5]["readouts"]
    assert res["new"][R5]["criteria"]["K1"]["pass_at"]["final"] is None
    # a refused final of an early-stopped run is one invalid file, not two
    res = score(tmp_path / "d", tree, gate=early, refusals={R5: {"final": refusal(R5, "final")}}, runs_ended=True)
    inv = res["new"][R5]["invalid"]
    assert [os.path.basename(x["path"]) for x in inv] == ["refused_final.json"] and len(inv[0]["issues"]) == 2


def test_a_finished_run_is_not_an_early_stop(tmp_path):
    res = score(tmp_path / "a", cells(), gate={R4: SC3.FINAL_TIMESTEPS, R5: SC3.FINAL_TIMESTEPS}, runs_ended=True)
    assert (status(res), line(res)) == ("decided", 1)
    assert res["new"][R4]["early_stop"] == {"gate_verdict": str(tmp_path / "a" / "runs" / R4 / "01_stance" / "gate_verdict.json"),
                                            "timesteps": SC3.FINAL_TIMESTEPS, "early": False}
    # without an early stop, a placed final pair with no judge file still waits for its judge
    res = score(tmp_path / "b", cells(drop={R5: ["final"]}), gate={R5: SC3.FINAL_TIMESTEPS}, place={R5: ["final"]},
                runs_ended=True)
    assert status(res) == "provisional" and res["decision"]["placed_unjudged"] == [f"{R5}/stage1_final.zip"]
    # a gate_verdict.json that records no timesteps is a note, not an early stop
    d = tmp_path / "c" / "runs" / R4 / "01_stance"
    os.makedirs(d)
    (d / "gate_verdict.json").write_text(json.dumps({"stage_result": {}}))
    res = score(tmp_path / "c", cells())
    assert line(res) == 1 and any("records no stage_result.timesteps" in n for n in res["notes"])


def test_early_stop_is_read_from_the_placed_copy_under_ledgers(tmp_path):
    """--ledgers RUN_ID=DIR (another ledger directory) does not hide the placed gate_verdict.json (criteria RUN
    HANDLING, Ended): score_stance3.py's _find with a ledger directory looks only there."""
    led = tmp_path / "ledgers"
    os.makedirs(led)
    res = score(tmp_path / "a", cells(drop={R5: ["final"]}), gate={R5: 9437184}, place={R5: ["final"]},
                ledger_dirs={R5: str(led)})
    es = res["new"][R5]["early_stop"]
    assert es["early"] and es["timesteps"] == 9437184
    assert es["gate_verdict"] == str(tmp_path / "a" / "runs" / R5 / "01_stance" / "gate_verdict.json")
    assert res["decision"]["placed_unjudged"] == [] and status(res) == "provisional"
    assert "early stop at 9,437,184 (gate_verdict.json): the final J file is invalid" in SR.render(res)


def test_records_check_failed_and_a_review_hold_are_notes(tmp_path):
    """place5r.py --run-records-fail's marker and judge_one5r.sh's review hold are report only: a review counts for
    nothing (the slot stays missing, and with --runs-ended its placed pair is placed but unjudged)."""
    tree = cells(drop={R4: ["6.3M"]})
    st = tmp_path / "runs" / R4 / "01_stance"
    os.makedirs(st)
    (st / SR.RECORDS_FAILED).write_text(json.dumps({"problems": ["stage_config.json: run.seed 52 != 70"]}))
    os.makedirs(tmp_path / "judged" / R4)
    (tmp_path / "judged" / R4 / "refused_6291456.review.json").write_text(json.dumps(refusal(R4, 6291456)))
    res = score(tmp_path, tree, place={R4: [6291456]})
    assert status(res) == "incomplete" and [os.path.basename(p) for p in res["decision"]["missing"]] == ["judge_6291456.json"]
    assert any(n.startswith(f"{R4}: its run records were placed with place5r.py --run-records-fail "
                            "(stage_config.json: run.seed 52 != 70)") for n in res["notes"])
    assert any("refused_6291456.review.json is a judge refusal on the checkpoint's own attributes only" in n
               for n in res["notes"])
    res = score(tmp_path, tree, place={R4: [6291456]}, runs_ended=True)
    assert status(res) == "provisional" and res["decision"]["placed_unjudged"] == [f"{R4}/stage1_6291456_steps.zip"]


def test_a_restart_folder_is_the_run(tmp_path):
    res = score(tmp_path, cells(), source={R4: f"{R4}_b"}, runs_ended=True)
    assert (status(res), line(res)) == ("decided", 1) and res["new"][R4]["source_folder"] == f"{R4}_b"
    assert any(n.startswith(f"{R4}: placed from the Drive folder {R4}_b") for n in res["notes"])


def test_refusal_schema_matches_the_tool():
    tool = open(os.path.join(SR.RDIR, "tools", "judge_one5r.sh")).read()
    assert f'"schema": "{SR.REFUSAL_SCHEMA}"' in tool and "refused_$STEPS.json" in tool


# ------------------------------------------------------------------------------------------------ pins
def test_criteria_pin(tmp_path):
    for pin, why in (("placeholder", "PLACEHOLDER"), ("mismatch", "!= the pinned")):
        res = score(tmp_path / pin, cells(), pin=pin)
        assert status(res) == "provisional" and line(res) == 1
        assert any(why in w for w in res["decision"]["provisional_why"])


def test_judge_pin(tmp_path, monkeypatch):
    res = score(tmp_path / "a", cells(at={R4: {"4.2M": {"jsha": "sha256:" + "c" * 64}}}))
    assert status(res) == "provisional" and any("not judged by the pinned judge" in w for w in res["decision"]["provisional_why"])
    monkeypatch.setattr(SR, "JUDGE_SHA256_S5R", None)
    res = score(tmp_path / "b", cells())
    assert status(res) == "provisional" and any("JUDGE_SHA256_S5R is a PLACEHOLDER" in w for w in res["decision"]["provisional_why"])


def test_task_pin(tmp_path):
    res = score(tmp_path, cells(at={R5: {"final": {"task": "sha256:" + "d" * 64}}}))
    assert status(res) == "provisional"
    assert any("task_sha256 other than the band-on" in w for w in res["decision"]["provisional_why"])


def test_head_and_validation(tmp_path):
    res = score(tmp_path / "a", cells(), head="e" * 40)
    assert status(res) == "provisional" and any("is not the pinned HEAD_LTSB" in w for w in res["decision"]["provisional_why"])
    res = score(tmp_path / "b", cells(), validation=True)
    assert status(res) == "provisional" and any("--validation" in w for w in res["decision"]["provisional_why"])


# ------------------------------------------------------------------------------------------------ the stance5 runs
def test_validate_reproduces_score_final():
    res = SR.score(validate=True)
    rep = res["stance5_reproduction"]
    assert rep["reproduced"] and not rep["problems"]
    assert {run: rep["recorded"][run]["HOLDS"] for run in (S52, S60, S42)} == {S52: True, S60: False, S42: False}
    assert rep["recorded"][S60]["failed"] == ["K3"] and rep["recorded"][S42]["failed"] == ["K1", "K3"]
    assert "reproduced: yes" in SR.render(res)


def test_refuses_on_the_score_final_pin(tmp_path):
    with pytest.raises(SystemExit, match="do not reproduce") as exc:
        score(tmp_path, cells(), stance5_score_final_sha256="0" * 64)
    assert str(exc.value).endswith(SR.REFUSAL_SEQUEL) and "Tell the maintainer at once" in SR.REFUSAL_SEQUEL


def _record(tmp_path, change, name) -> tuple[str, str]:
    with open(SR.STANCE5_SCORE_FINAL) as fh:
        rec = json.load(fh)
    change(rec)
    p = tmp_path / name
    p.write_text(json.dumps(rec))
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def test_refuses_on_a_tampered_record(tmp_path):
    def flip(rec):
        rec["band"][S60]["criteria"]["K3"]["pass"] = True
    path, sha = _record(tmp_path, flip, "flipped.json")
    with pytest.raises(SystemExit, match=f"{S60}: criteria"):
        score(tmp_path, cells(), stance5_score_final=path, stance5_score_final_sha256=sha)

    def other_line(rec):
        rec["decision"]["line"] = 1
    path, sha = _record(tmp_path, other_line, "other_line.json")
    with pytest.raises(SystemExit, match="does not record decision line 3"):
        score(tmp_path / "l", cells(), stance5_score_final=path, stance5_score_final_sha256=sha)


def test_refuses_on_a_changed_stance5_judge_file(tmp_path):
    """A copy of S5/judged (symlinks) in which s60's final judge file reads 0 joint-noise falls: K3 would pass."""
    root = tmp_path / "s5judged"
    for run in (S52, S60, S42):
        os.makedirs(root / run)
        src = os.path.join(S5.S5_JUDGED, run)
        for name in os.listdir(src):
            if name.startswith("judge_") and name.endswith(".json"):
                os.symlink(os.path.join(src, name), root / run / name)
    os.remove(root / S60 / "judge_final.json")
    with open(os.path.join(S5.S5_JUDGED, S60, "judge_final.json")) as fh:
        d = json.load(fh)
    for row in d["noise_matched_control_joint"]["episodes_detail"]:
        row["reached_horizon"], row["length"] = True, 1000
    d["noise_matched_control_joint"]["falls"] = 0
    (root / S60 / "judge_final.json").write_text(json.dumps(d))
    res = SR.score(validate=True, stance5_judged=str(root))
    assert not res["stance5_reproduction"]["reproduced"]
    assert res["stance5_reproduction"]["runs"][S60]["criteria_identical"] is False
    with pytest.raises(SystemExit, match="refusing to score"):
        score(tmp_path / "s", cells(), stance5_judged=str(root))


# ------------------------------------------------------------------------------------------------ the sidecar
def test_sidecar_is_the_recomputed_mean():
    want = SR.sidecar_mean(FROZEN)
    assert len(want) == 15
    for s in SR.SIDECAR_SEEDS:
        assert SIDECAR["control_sigma"][s]["vector"] == want
    src = [FROZEN["criteria"]["H4"]["control_sigma"][s]["vector"] for s in SR.SIDECAR_SOURCE_SEEDS]
    assert all(min(v[i] for v in src) <= want[i] <= max(v[i] for v in src) for i in range(15))
    assert SIDECAR["source_sha256"] == SC3.FROZEN_PARAMS_SHA256
    assert SIDECAR["rule"] == "element-wise mean of frozen_params criteria.H4.control_sigma 42, 52, 60"
    r = subprocess.run([sys.executable, "-I", os.path.join(SR.RDIR, "build", "make_sigma.py"), "--check", SR.SIDECAR],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0 and '"ok": true' in r.stdout, r.stdout + r.stderr


def test_sidecar_refusals(tmp_path):
    with pytest.raises(SystemExit, match="!= the pinned"):
        score(tmp_path / "a", cells(), sidecar_sha256="0" * 64)
    bad = copy.deepcopy(SIDECAR)
    bad["control_sigma"]["80"]["vector"][3] *= 1.01
    p = tmp_path / "sidecar_bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="not the element-wise mean"):
        score(tmp_path / "b", cells(), sidecar_path=str(p), sidecar_sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def test_frozen_with_sidecar_leaves_the_frozen_entries():
    before = copy.deepcopy(FROZEN)
    got = SR.frozen_with_sidecar(FROZEN, SIDECAR)
    assert FROZEN == before   # not mutated
    cs = got["criteria"]["H4"]["control_sigma"]
    for s in ("42", "52", "60"):
        assert cs[s] == FROZEN["criteria"]["H4"]["control_sigma"][s]
    for s in SR.SIDECAR_SEEDS:
        assert cs[s] == {"source": SIDECAR["control_sigma"][s]["source"], "vector": SIDECAR["control_sigma"][s]["vector"]}


# ------------------------------------------------------------------------------------------------ report only
def test_k3_fraction_and_per_test_fractions(tmp_path):
    res = score(tmp_path / "a", cells())
    fr = res["report"]["fractions"]
    assert (fr["K3"]["band"], fr["K3"]["control"], fr["K3"]["band_at_least_control"]) == ("3/5", "2/3", False)
    assert {k: fr[k]["band"] for k in SR.KS} == {"K1": "4/5", "K2": "5/5", "K3": "3/5", "K4": "5/5"}
    assert res["report"]["rule_b_applied"] is False and line(res) == 1   # not a veto
    assert res["report"]["fore_aft"] == {S52: True, S60: True, S42: False, R4: True, R5: True}
    res = score(tmp_path / "b", cells(fail={R4: ["K3"], R5: ["K3"]}))
    assert res["report"]["fractions"]["K3"]["band"] == "1/5" and line(res) == 3
    assert res["report"]["fore_aft"][R4] is True


def test_parent_comparison_five_finals(tmp_path):
    res = score(tmp_path / "a", cells())   # s70 and s80 finals 40/40 with 0 noise falls: tied with s52
    p = res["parents"]
    assert p["best_holding_band_final"]["best"] is None
    assert set(p["best_holding_band_final"]["tied"]) == {f"Run 1 {S52}", f"Run 4 {R4}", f"Run 5 {R5}"}
    assert f"stance3 Run 2 stance3_armLT_s60" in p["comparison"]["top_tied"]
    res = score(tmp_path / "b", cells(at={R4: {"final": {"eps": eps_with(1, clean=False)}},
                                          R5: {"final": {"joint_falls": 1}}}))
    p = res["parents"]
    assert line(res) == 1 and p["best_holding_band_final"]["best"] == f"Run 1 {S52}"


def test_control_problems_are_notes(tmp_path):
    p = tmp_path / "stance4_score_final_copy.json"
    p.write_bytes(open(S5.STANCE4_SCORE_FINAL, "rb").read() + b"\n")   # same content, another sha256
    res = score(tmp_path, cells(), control_score_final=str(p))
    assert status(res) == "decided" and line(res) == 1
    assert any(n.startswith("control check (report only") for n in res["notes"])


# ------------------------------------------------------------------------------------------------ texts and pins
def test_decision_lines_are_verbatim():
    with open(SR.CRITERIA_FILE_R) as fh:
        text = fh.read()
    for n, s in SR.DECISION_LINES_R.items():
        assert s in text.splitlines(), n


def test_pins_are_the_files_as_they_stand():
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    assert FILE_PINS["CRITERIA_SHA256_R"] == sha(SR.CRITERIA_FILE_R)
    assert FILE_PINS["JUDGE_SHA256_S5R"] == "sha256:" + sha(JUDGE5R)
    assert FILE_PINS["SIDECAR_SHA256"] == sha(SR.SIDECAR)
    assert FILE_PINS["HEAD_LTSB"] == "cf80ecb14d1630ce3b4dcdea28a05835d143470b"
    jt = open(JUDGE5R).read()
    assert re.search(r'^CONTROL_SIGMA_SIDECAR_SHA256 = "([0-9a-f]{64})"$', jt, re.M).group(1) == FILE_PINS["SIDECAR_SHA256"]
    assert re.search(r'^SCHEMA = "([^"]+)"$', jt, re.M).group(1) == SR.JUDGE_SCHEMA_S5R
    assert sha(JUDGE5) == SR.JUDGE_SHA256_S5.split(":")[1]   # the stance5 judge is untouched


def test_judge_diff_is_the_recorded_one():
    a, b = open(JUDGE5).read().splitlines(), open(JUDGE5R).read().splitlines()
    got = [x for x in difflib.unified_diff(a, b, lineterm="", n=3)][2:]
    rec = [x for x in open(os.path.join(HERE, "judge_diff.txt")).read().splitlines()][2:]
    assert got == rec
    removed = [x[1:] for x in got if x.startswith("-")]
    assert len(removed) == 9   # the usage lines (3), SCHEMA, SEEDS, the --seed help, the ctrl line, the two log prefixes
    added = [x[1:] for x in got if x.startswith("+")]
    assert any(x.startswith("SCHEMA = \"stance5r-judge/v1\"") for x in added)


# ------------------------------------------------------------------------------------------------ the validation judges
def _validation(name):
    p = os.path.join(VALID, name)
    if not os.path.exists(p):
        pytest.skip(f"{p} not written (the validation judge has not run)")
    with open(p) as fh:
        return p, json.load(fh)


def test_validation_seed52_reproduces_judge_stance5(tmp_path):
    p, d = _validation("judge5r_s52_4194304.json")
    assert d["schema"] == SR.JUDGE_SCHEMA_S5R and d["judge_sha256"] == FILE_PINS["JUDGE_SHA256_S5R"]
    assert d["control_sigma_file"] == SC3.FROZEN_PARAMS and d["control_sigma_file_sha256"] == SC3.FROZEN_PARAMS_SHA256
    root = tmp_path / "j"
    os.makedirs(root / S52)
    os.symlink(p, root / S52 / "judge_4194304.json")
    notes = []
    c5r, why, entry = SR.SC5R.load_cell(str(root), S52, "LTSB", 52, 4194304, "4.2M", expect_head=HEAD, frozen=FROZEN,
                                        validation=False, needs_noise=True, notes=notes)
    assert why is None, entry
    c5, why, _ = SR.SC5.load_cell(S5.S5_JUDGED, S52, "LTSB", 52, 4194304, "4.2M", expect_head=HEAD, frozen=FROZEN,
                                  validation=False, needs_noise=True, notes=[])
    drop = ("path",)
    assert {k: v for k, v in c5r.items() if k not in drop} == {k: v for k, v in c5.items() if k not in drop}


def test_validation_seed70_reads_the_sidecar():
    p, d = _validation("judge5r_s52ckpt_as_s70_4194304.json")
    cj = d["noise_matched_control_joint"]
    assert cj["sigma_vector"] == SIDECAR["control_sigma"]["70"]["vector"]
    assert cj["sigma_source"] == SIDECAR["control_sigma"]["70"]["source"]
    assert d["control_sigma_file"] == SR.SIDECAR and d["control_sigma_file_sha256"] == FILE_PINS["SIDECAR_SHA256"]
    frozen_r = SR.frozen_with_sidecar(FROZEN, SIDECAR)
    bad, soft = SR.SC5R.validity_issues(d, arm="LTSB", seed=70, steps=4194304, label="4.2M", expect_head=HEAD,
                                        frozen=frozen_r, validation=True, needs_noise=True)
    # s52's checkpoint judged as seed 70 (validation only): refused for its seed, never for the noise configuration
    assert bad == ["checkpoint seed 52 != slot seed 70"] and any("allow_seed_mismatch" in s for s in soft)
    bad, _ = SR.SC5R.validity_issues(d, arm="LTSB", seed=70, steps=4194304, label="4.2M", expect_head=HEAD,
                                     frozen=frozen_r, validation=False, needs_noise=True)
    assert bad and all("noise" not in b for b in bad)
    bad, _ = SR.SC5R.validity_issues(d, arm="LTSB", seed=70, steps=4194304, label="4.2M", expect_head=HEAD,
                                     frozen=FROZEN | {"criteria": {"H4": {**FROZEN["criteria"]["H4"], "control_sigma": {
                                         **FROZEN["criteria"]["H4"]["control_sigma"], "70": FROZEN["criteria"]["H4"][
                                             "control_sigma"]["52"]}}}}, validation=True, needs_noise=True)
    assert "H4 noise configuration: control joint sigma (frozen, this seed)" in bad   # another sigma is refused


# ------------------------------------------------------------------------------------------------ the command line
def test_command_line(tmp_path):
    script = os.path.join(HERE, "score_stance5r.py")
    r = subprocess.run([sys.executable, "-I", script, "--validate"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "reproduced: yes" in r.stdout, r.stderr[-2000:]
    out = tmp_path / "s.json"
    r = subprocess.run([sys.executable, "-I", script, "--judged", str(tmp_path / "none"), "--runs-root",
                        str(tmp_path / "runs"), "--json", str(out)], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "DECISION: incomplete" in r.stdout, r.stderr[-2000:]
    assert json.load(open(out))["decision"]["status"] == "incomplete"
    r = subprocess.run([sys.executable, "-I", script, "--json", str(out)], capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr
