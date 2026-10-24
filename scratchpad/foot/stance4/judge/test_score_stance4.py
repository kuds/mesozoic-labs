"""score_stance4.py on hand-made stagger judge files, beside the real stance3 control (and a synthetic one).

    python3 -m pytest -p no:cacheprovider test_score_stance4.py --basetemp=<work>/tmp

Covers: the control rows reproduce stance3's score_final.json exactly; K1-K4 are score_stance3.py's code (both module
instances against a fresh import); every decision-table line and its boundaries (H = 2 with one K3 or K4 fail, H = 2
with one K1 or K2 fail, H = 1, H = 0 with K2p = 3, 2, 1, 0); (b) in its general form against synthetic control records;
the control's passes come from the record, not from its recomputed files; a missing J file (incomplete) against a
missing report-only one, and both after the runs have ended (--runs-ended); invalid files (provisional, then fails once
the runs have ended); the HEAD, criteria, task and judge pins and placeholders; a null or missing task_sha256 on a
stagger or control J file; --validation; the parent comparison, its ties and its inputs-only form while the decision is
incomplete; the 1,048,576-step report-only rows; the saturation alarm; the command line.

The hand-made stagger files carry a test HEAD and a test judge sha256, so an autouse fixture sets HEAD_LTS and
JUDGE_SHA256_S4 to those values; the tests that check the placeholders unset them.  The command-line test runs the file
as it stands (its own pins, FILE_PINS).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_stance4 as SS  # noqa: E402

#: the pins as they stand in score_stance4.py (read before the autouse fixture replaces any)
FILE_PINS = {k: getattr(SS, k) for k in ("HEAD_LTS", "CRITERIA_SHA256", "TASK_SHA256_LTS", "JUDGE_SHA256_S4")}
SC3, SC4 = SS.SC3, SS.SC4
FROZEN = json.load(open(SC3.FROZEN_PARAMS))
TEST_HEAD = "a" * 40
TEST_JUDGE_SHA = "sha256:" + "b" * 64
STANCE3_JUDGE_SHA = "sha256:5dd3cf4d67ac696ddf6c9f6b986aefea4801aee46f25e72a67774d95c67526c3"
LT_HEAD = SC3.HEADS["LT"]
SLOTS = {**dict(SC3.J), **dict(SS.REPORT_1M)}
STAGGER = {run: seed for run, _arm, seed, _name in SS.STAGGER_RUNS}
CONTROL = {run: seed for run, _arm, seed, _name in SS.CONTROL_RUNS}
R1, R2, R3 = (r[0] for r in SS.STAGGER_RUNS)
C2, C3 = (r[0] for r in SS.CONTROL_RUNS)
KINDS = {  # schema, arm, head, [env]
    "LTS": (SS.JUDGE_SCHEMA_S4, "LTS", TEST_HEAD, {**SC3.ARM_ENV_COMMON, **SS.ARM_ENV_LTS}),
    "LT3": (SC3.JUDGE_SCHEMA, "LT", LT_HEAD, {**SC3.ARM_ENV_COMMON, **SC3.ARM_ENV["LT"]}),
    "LT4": (SS.JUDGE_SCHEMA_S4, "LT", LT_HEAD, {**SC3.ARM_ENV_COMMON, **SC3.ARM_ENV["LT"]}),
}
STATUE_XC = 0.065
JUDGE_SHAS = {"LTS": TEST_JUDGE_SHA, "LT3": STANCE3_JUDGE_SHA, "LT4": TEST_JUDGE_SHA}


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    """The hand-made stagger files' HEAD and judge sha256 as the scorer's pins (the file's own pins: FILE_PINS)."""
    monkeypatch.setattr(SS, "HEAD_LTS", TEST_HEAD)
    monkeypatch.setattr(SS, "JUDGE_SHA256_S4", TEST_JUDGE_SHA)


# ------------------------------------------------------------------------------------------------ synthetic files
def episode(seed, *, full=True, clean=True, pad=0.4, w=(0.1, 0.1), xc=0.066, tilt=0.5, lift=0.001, hop=False):
    reasons = []
    if not full:
        reasons.append("horizon: 412 < 1000")
    if hop and full:
        reasons.append("min_all_feet_support: 0.9 < 0.98")
    if pad is None or pad > 0.8:
        reasons.append(f"max_sole_cop_fore_aft: {pad} > 0.8")
    if tilt > 2.0:
        reasons.append(f"max_sole_tilt_deg: {tilt} > 2.0")
    if lift > 0.004:
        reasons.append(f"max_sole_corner_lift_m: {lift} > 0.004")
    if not clean and not reasons:
        reasons.append("min_foot_load_share: 0.3 < 0.4")
    return {"seed": seed, "length": 1000 if full else 412, "reached_horizon": full, "clean": not reasons,
            "reasons": reasons, "reward": 3900.0, "termination_reason": "truncated" if full else "nosedive",
            "max_sole_cop_fore_aft": pad, "max_foot_cop_fore_aft": 0.3, "max_sole_tilt_deg": tilt,
            "max_sole_corner_lift_m": lift, "foot_cop_fore_aft": {"r": w[0], "l": w[1]},
            "xc_both_feet_m": xc if full else None, "com_mtp3_head_m": xc if full else None,
            "com_fore_aft_footsite_m": -0.016 if full else None}


def eps_with(n_bad=0, kind="unclean", **kw):
    """40 episodes, the first *n_bad* altered by *kind*."""
    alter = {"unclean": {"clean": False}, "w": {"w": kw.get("w")}, "tilt": {"tilt": 2.01}, "pad": {"pad": 0.81},
             "fall": {"full": False}}[kind]
    return [episode(3042 + i, **(alter if i < n_bad else {})) for i in range(40)]


def noise(seed, space, falls):
    rows = [{"seed": 3042 + i, "length": 1000 if i >= falls else 300, "reached_horizon": i >= falls, "reward": 3800.0}
            for i in range(20)]
    d = {"n": 20, "falls": falls, "seeds": [3042, 3061], "space": space, "episodes_detail": rows}
    if space == "joint":
        ref = FROZEN["criteria"]["H4"]["control_sigma"][str(seed)]
        d.update({"sigma": "control", "sigma_source": ref["source"], "sigma_vector": list(ref["vector"])})
    else:
        d.update({"sigma": "own", "sigma_source": "own", "sigma_vector": [0.1] * 15})
    return d


def judge(kind, seed, label, *, eps=None, joint_falls=0, own_falls=0):
    """A valid judge file of *kind* (LTS: a stagger run; LT3: a stance3 control file; LT4: a control file of the stance4
    judge) for *seed* at slot *label*; eps defaults to 40 clean, full-horizon, centred episodes."""
    schema, arm, head, env = KINDS[kind]
    eps = eps if eps is not None else [episode(3042 + i) for i in range(40)]
    k = sum(e["clean"] for e in eps)
    full = sum(e["reached_horizon"] for e in eps)
    hop = sum((not e["reached_horizon"]) or any(r.startswith("min_all_feet_support") for r in e["reasons"]) for e in eps)
    lcb = SC3.binomial_lcb(k, 40)
    verdict = "PASS" if (k >= 37 and lcb >= 0.8 and full >= 38 and hop <= 1) else "FAIL"
    steps = SLOTS.get(label, label)
    stem = "stage1_final" if steps == "final" else f"{steps}_model" if steps in ("best", "robust_best") else f"stage1_{steps}_steps"
    d = {"schema": schema, "mode": "full", "smoke": False, "arm": arm, "seed": seed, "steps": steps, "checkpoint": stem,
         "judge_sha256": JUDGE_SHAS[kind],
         "git": {"head": head, "status": "", "branch": "x"}, "model_sha256": f"sha256:{kind}{seed}{steps}",
         "plant_identity": {"physics_sha256": SC3.R9_PHYSICS_SHA256, "physics_revision": 9}, "arm_env_kwargs": dict(env),
         "checks": {"relaxed": {"no_symmetry_check": False, "task_binding": "require", "allow_missing_records": False,
                                "allow_seed_mismatch": False, "allow_commit_mismatch": False},
                    "hard_failures": [], "plant_physics_sha256": {"match": True}, "symmetry": {"mirror_map_sha256_match": True},
                    "seed": {"label": seed, "checkpoint": seed, "match": True},
                    "checkpoint_num_timesteps": SC3.FINAL_TIMESTEPS if steps == "final" else steps,
                    "stage_config": {"git_commit_match": True, "git_commit": head, "clone_head": head, "run_seed": seed,
                                     "run_n_envs": SC3.N_ENVS, "run_hyperparameters_sha256": SC3.HP_SHA256}},
         "panel": {"verdict": verdict, "passed": verdict == "PASS", "n_clean": k, "clean_lcb": lcb, "n_full_horizon": full,
                   "hop_or_fall_episodes": hop, "mean_reward": 3900.0, "statue_mean_reward": 3980.0,
                   "terminations": {"truncated": full}, "scored_gate_kind": "stance_quality/v2",
                   "task_sha256": SS.TASK_SHA256_LTS if arm == "LTS" else SS.TASK_SHA256_LT,
                   "thresholds": dict(SC3.R9_BARS), "statue_on_same_seeds": {"n": 40},
                   "statue_episodes_detail": [episode(3042 + i, full=(i != 39), xc=STATUE_XC) for i in range(40)],
                   "summary": {}, "episodes_detail": eps}}
    if label in ("6.3M", "final"):
        d["noise_matched_control_joint"] = noise(seed, "joint", joint_falls)
        d["noise_matched_own"] = noise(seed, "policy", own_falls)
    return d


def arm_cells(kind, runs: dict, fail: dict | None = None, final: dict | None = None) -> dict:
    """{run: {label: file}} for *runs* ({run: seed}); fail = {run: [K...]} makes each named test fail:
    K1 36/40 clean at 6.3M; K2 four episodes outside the K2b band at 8.4M (clean); K3 3/20 joint falls at final; K4 four
    tilted episodes at final (so K1 fails too: the K4 rails are v2 bars).  final = {run: judge kwargs} overrides final."""
    out = {}
    for run, seed in runs.items():
        kw = {lab: {} for lab, _ in SC3.J}
        for k in (fail or {}).get(run, []):
            if k == "K1":
                kw["6.3M"]["eps"] = eps_with(4, "unclean")
            elif k == "K2":
                kw["8.4M"]["eps"] = eps_with(4, "w", w=(0.7, 0.0))
            elif k == "K3":
                kw["final"]["joint_falls"] = 3
            elif k == "K4":
                kw["final"]["eps"] = eps_with(4, "tilt")
        kw["final"].update((final or {}).get(run, {}))
        out[run] = {lab: judge(kind, seed, lab, **kw[lab]) for lab, _ in SC3.J}
    return out


def write_tree(root, cells: dict) -> str:
    for run, slots in cells.items():
        os.makedirs(os.path.join(root, run), exist_ok=True)
        for lab, d in slots.items():
            if d is not None:
                with open(os.path.join(root, run, f"judge_{SLOTS.get(lab, lab)}.json"), "w") as fh:
                    json.dump(d, fh)
    return str(root)


def score(tmp, stagger_cells, *, control_cells=None, head=TEST_HEAD, pin="match", ledgers=None, runs_ended=False,
          score_final=SS.STANCE3_SCORE_FINAL, validation=False):
    """Write the trees under *tmp* and score them; the control is the real stance3 one unless *control_cells* is given."""
    judged = write_tree(tmp / "judged", stagger_cells)
    control_root = SS.S3_JUDGED if control_cells is None else write_tree(tmp / "control", control_cells)
    crit = tmp / "criteria.txt"
    crit.write_text("synthetic criteria\n")
    sha = {"match": hashlib.sha256(crit.read_bytes()).hexdigest(), "placeholder": None, "mismatch": "0" * 64}[pin]
    runs_root = tmp / "runs"
    for run, rows in (ledgers or {}).items():
        os.makedirs(runs_root / run / "01_stance", exist_ok=True)
        with open(runs_root / run / "01_stance" / "eval_episodes.csv", "w") as fh:
            fh.write("evaluation,timesteps,episode,full_horizon,screen_v2_reasons,max_sole_cop_fore_aft,"
                     "max_foot_cop_fore_aft,com_fore_aft_mean_m\n")
            for i, r in enumerate(rows):
                fh.write(f"0,{r['timesteps']},{i},1,{r['screen_v2_reasons']},0.3,0.3,-0.016\n")
    res = SS.score(judged, control_root, runs_root=str(runs_root), control_runs_root=str(tmp / "control_runs"),
                   head_lts=head, criteria_file=str(crit), criteria_sha256=sha, runs_ended=runs_ended,
                   score_final=str(score_final), validation=validation)
    cr = res["credit_rule"]
    # the two forms of (b) agree under (a) whenever the control stands as the criteria state it
    if cr["a"] and cr["control_as_stated"]:
        assert cr["b_specific"] == cr["b"]
    SS.render(res)  # renders without error
    return res


def holds(res):
    return {run: res["stagger"][run]["criteria"]["HOLDS"] for run in STAGGER}


def synthetic_record(tmp, ctrl_cells, monkeypatch) -> str:
    """A score_final.json-like record of a synthetic control: its recomputed criteria, as stance3's scorer stores them;
    its sha256 is pinned in place of the real file's (so only the stated-results check differs)."""
    first = score(tmp / "first", arm_cells("LTS", STAGGER), control_cells=ctrl_cells)
    rec = {"runs": {run: {"criteria": json.loads(json.dumps(first["control"][run]["criteria"]))} for run in CONTROL}}
    path = tmp / "score_final_synthetic.json"
    path.write_text(json.dumps(rec))
    monkeypatch.setattr(SS, "STANCE3_SCORE_FINAL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return str(path)


# ------------------------------------------------------------------------------------------------ the control
def test_control_reproduces_stance3_score_final(tmp_path):
    res = SS.score(str(tmp_path / "none"), SS.S3_JUDGED, runs_root=str(tmp_path / "r"), control_runs_root=SS.S3_RUNS,
                   head_lts=TEST_HEAD)
    with open(SS.STANCE3_SCORE_FINAL) as fh:
        stored = json.load(fh)
    assert SS.sha256_file(SS.STANCE3_SCORE_FINAL) == SS.STANCE3_SCORE_FINAL_SHA256
    for run in CONTROL:
        got, want = res["control"][run], stored["runs"][run]
        assert json.loads(json.dumps(got["criteria"])) == want["criteria"]
        for lab, _ in SC3.J:
            assert json.loads(json.dumps(got["cells"][lab])) == want["cells"][lab]
        assert json.loads(json.dumps(got["k1_failure_modes"])) == want["k1_failure_modes"]
        assert got["invalid"] == [] and got["missing"] == []
    assert res["control"][C2]["criteria"]["failed"] == ["K1", "K2"]
    assert res["control"][C3]["criteria"]["failed"] == ["K2"]
    x = res["control_crosscheck"]
    assert x["match"] and x["stated_match"] and all(r["criteria_identical"] for r in x["runs"].values())
    assert all(r["task_sha256_seen"] == [SS.TASK_SHA256_LT] for r in x["runs"].values()) and not x["problems"]
    assert res["credit_rule"]["control_passes"] == {"K1": 1, "K2": 0, "K3": 2, "K4": 2}
    assert res["credit_rule"]["control_as_stated"]
    assert res["credit_rule"]["control_source"] == "recorded in stance3's score_final.json"


# ------------------------------------------------------------------------------------------------ same code
def _fresh_stance3():
    spec = importlib.util.spec_from_file_location("fresh_score_stance3", SS.SC3_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_k_code_is_score_stance3(tmp_path):
    fresh = _fresh_stance3()
    assert SS.sha256_file(SS.SC3_PATH) == SS.SC3_SHA256
    score(tmp_path, arm_cells("LTS", STAGGER))  # a full scoring pass first: nothing it does may leave a change behind
    funcs = [n for n, v in vars(fresh).items() if isinstance(v, types.FunctionType)]
    assert {"run_criteria", "panel_counts", "validity_issues", "load_cell", "k1_failure_modes", "best_final"} <= set(funcs)
    for n in funcs:
        assert getattr(SC3, n).__code__ == getattr(fresh, n).__code__ == getattr(SC4, n).__code__, n
    for n, v in vars(fresh).items():
        if n.startswith("__") or isinstance(v, (types.FunctionType, types.ModuleType, type)):
            continue
        assert getattr(SC3, n) == v, n
        if n not in SS.PATCHED:
            assert getattr(SC4, n) == v, n
    assert SC4.JUDGE_SCHEMA == "stance4-judge/v1" and SC3.JUDGE_SCHEMA == "stance3-judge/v1"
    assert SC4.ARM_ENV["L"] == fresh.ARM_ENV["L"] and SC4.ARM_ENV["LT"] == fresh.ARM_ENV["LT"]
    assert SC4.ARM_ENV["LTS"] == {**fresh.ARM_ENV["LT"], "stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045}
    assert {k: SC4.HEADS[k] for k in ("L", "LT")} == fresh.HEADS
    src = open(os.path.join(HERE, "score_stance4.py")).read()
    for name in ("run_criteria", "panel_counts", "validity_issues", "load_cell", "k2a_over", "k2b_outside",
                 "binomial_lcb", "best_final", "k1_failure_modes"):
        assert f"def {name}(" not in src, name


def test_same_files_same_criteria_in_both_instances(tmp_path):
    """The same episodes scored as a control file (SC3) and as a stagger file (SC4) give identical K1-K4."""
    fail = {R1: ["K2", "K3"]}
    lts = arm_cells("LTS", {R1: 52}, fail)
    lt = arm_cells("LT3", {C3: 52}, {C3: ["K2", "K3"]})
    for kind, cells, mod, head in (("LTS", lts, SC4, TEST_HEAD), ("LT3", lt, SC3, LT_HEAD)):
        write_tree(tmp_path / kind, cells)
    notes = []
    a = SS.load_run(SC4, str(tmp_path / "LTS"), R1, "LTS", 52, "x", head=TEST_HEAD, frozen=FROZEN, validation=False,
                    notes=notes, report_sources=[], runs_root=str(tmp_path), ledger_dir=None)
    b = SS.load_run(SC3, str(tmp_path / "LT3"), C3, "LT", 52, "x", head=LT_HEAD, frozen=FROZEN, validation=False,
                    notes=notes, report_sources=[], runs_root=str(tmp_path), ledger_dir=None)
    assert a["criteria"] == b["criteria"] and a["criteria"]["failed"] == ["K2", "K3"]
    assert not notes


# ------------------------------------------------------------------------------------------------ the decision table
@pytest.mark.parametrize("fail, H, K2p, line, b", [
    ({}, 3, 3, 1, True),
    ({R3: ["K1"]}, 2, 3, 1, True),            # K1 2/3 >= control 1/2
    ({R3: ["K2"]}, 2, 2, 1, True),            # K2 2/3 >= control 0/2
    ({R3: ["K3"]}, 2, 3, 2, False),           # boundary: H = 2 with one K3 fail
    ({R1: ["K4"]}, 2, 3, 2, False),           # one K4 fail (and K1 with it)
    ({R1: ["K2"], R2: ["K2"]}, 1, 1, 3, True),    # (b) holds (K2 is 0/2 for the control) but H = 1
    ({R1: ["K3"], R2: ["K2"]}, 1, 2, 3, False),
    ({R1: ["K3"], R2: ["K3"], R3: ["K3"]}, 0, 3, 4, False),
    ({R1: ["K3"], R2: ["K1"], R3: ["K2"]}, 0, 2, 4, False),   # boundary: K2p = 2
    ({R1: ["K2"], R2: ["K2"], R3: ["K3"]}, 0, 1, 5, False),   # boundary: K2p = 1
    ({R1: ["K2"], R2: ["K2"], R3: ["K2"]}, 0, 0, 5, True),
])
def test_decision_lines(tmp_path, fail, H, K2p, line, b):
    res = score(tmp_path, arm_cells("LTS", STAGGER, fail))
    cr, d = res["credit_rule"], res["decision"]
    assert (cr["H"], cr["K2p"], cr["b"], cr["a"]) == (H, K2p, b, H >= 2)
    assert d["status"] == "decided" and d["line"] == line and d["text"] == SS.DECISION_LINES[line]
    assert res["control_crosscheck"]["match"]
    for run in STAGGER:
        assert res["stagger"][run]["criteria"]["failed"] == sorted(
            set(fail.get(run, [])) | ({"K1"} if "K4" in fail.get(run, []) else set()))


def test_h2_k3_fail_both_forms(tmp_path):
    res = score(tmp_path, arm_cells("LTS", STAGGER, {R2: ["K3"]}))
    cr = res["credit_rule"]
    assert cr["a"] and not cr["b"] and not cr["b_specific"] and cr["forms_agree"]
    assert cr["b_by_test"]["K3"] == {"stagger": "2/3", "control": "2/2", "no_worse": False}
    assert all(cr["b_by_test"][k]["no_worse"] for k in ("K1", "K2", "K4"))
    assert res["decision"]["line"] == 2


# ------------------------------------------------------------------------------------------------ (b), general form
def test_general_form_uses_the_controls_actual_scores(tmp_path, monkeypatch):
    """A control record with one K3 fail: a stagger arm at K3 2/3 is no worse (general form holds) although a
    stagger run fails K3 (the specific form, valid only for the real control, would say no)."""
    ctrl = arm_cells("LT3", CONTROL, {C2: ["K3"], C3: ["K2"]})
    rec = synthetic_record(tmp_path, ctrl, monkeypatch)
    res = score(tmp_path, arm_cells("LTS", STAGGER, {R3: ["K3"]}), control_cells=ctrl, score_final=rec)
    cr = res["credit_rule"]
    assert cr["control_passes"] == {"K1": 2, "K2": 1, "K3": 1, "K4": 2} and not cr["control_as_stated"]
    assert cr["control_source"] == "recorded in stance3's score_final.json"
    assert cr["H"] == 2 and cr["b"] and not cr["b_specific"] and cr["forms_agree"]
    assert res["decision"]["line"] == 1
    x = res["control_crosscheck"]
    assert x["reproduced"] and x["sha256_matches"] and not x["stated_match"]
    assert res["decision"]["status"] == "provisional"  # the record is not the control the criteria state
    assert any("not the ones the criteria state" in w for w in res["decision"]["provisional_why"])


def test_general_form_catches_k1(tmp_path, monkeypatch):
    """A control record passing K1 on both seeds: a stagger arm at K1 2/3 is worse, so (b) fails although no run fails
    K3 or K4."""
    ctrl = arm_cells("LT3", CONTROL, {C2: ["K2"], C3: ["K2"]})
    rec = synthetic_record(tmp_path, ctrl, monkeypatch)
    res = score(tmp_path, arm_cells("LTS", STAGGER, {R1: ["K1"]}), control_cells=ctrl, score_final=rec)
    cr = res["credit_rule"]
    assert cr["H"] == 2 and not cr["b"] and cr["b_specific"] and cr["b_by_test"]["K1"]["no_worse"] is False
    assert res["decision"]["line"] == 2


def test_control_passes_come_from_the_record_not_the_files(tmp_path):
    """The critic's case: without stance3_armLT_s52's final judge file the recomputed control would stand at K1 0/2,
    K3 1/2, K4 1/2 and let a K3 fail through (line 1).  The recorded passes (K3 2/2) give line 2, provisional."""
    ctrl = arm_cells("LT3", CONTROL, {C2: ["K1", "K2"], C3: ["K2"]})
    ctrl[C3]["final"] = None
    res = score(tmp_path, arm_cells("LTS", STAGGER, {R3: ["K3"]}), control_cells=ctrl)
    rec = res["control"][C3]["criteria"]
    assert not rec["K3"]["pass"] and not rec["K1"]["pass"]           # the recomputation alone would say K3 1/2
    cr = res["credit_rule"]
    assert cr["control_passes"] == {"K1": 1, "K2": 0, "K3": 2, "K4": 2} and cr["control_as_stated"]
    assert cr["H"] == 2 and not cr["b"] and res["decision"]["line"] == 2
    assert res["decision"]["status"] == "provisional"
    assert any("does not reproduce" in w for w in res["decision"]["provisional_why"])


def test_unreadable_control_record_falls_back_to_the_stated_results(tmp_path):
    res = score(tmp_path, arm_cells("LTS", STAGGER, {R3: ["K3"]}), score_final=tmp_path / "no_such_score_final.json")
    cr = res["credit_rule"]
    assert cr["control_passes"] == {"K1": 1, "K2": 0, "K3": 2, "K4": 2}
    assert cr["control_source"].startswith("the criteria's stated results")
    assert res["decision"]["line"] == 2 and res["decision"]["status"] == "provisional"
    assert any("score_final.json is missing" in w for w in res["decision"]["provisional_why"])


# ------------------------------------------------------------------------------------------------ missing / invalid
def test_missing_stagger_j_file_is_incomplete(tmp_path):
    cells = arm_cells("LTS", STAGGER)
    cells[R2]["final"] = None
    res = score(tmp_path, cells)
    d = res["decision"]
    assert d["status"] == "incomplete" and d["line"] is None and d["text"] is None and len(d["missing"]) == 1
    assert res["stagger"][R2]["criteria"]["failed"] == ["K1", "K2", "K3", "K4"]
    assert holds(res) == {R1: True, R2: False, R3: True}
    assert "DECISION: incomplete" in SS.render(res)


def test_missing_stagger_j_file_after_the_runs_ended_is_a_fail(tmp_path):
    """--runs-ended: the missing file fails K1-K4 for its run and the table is read (H = 2, K3 2/3 < 2/2: line 2)."""
    cells = arm_cells("LTS", STAGGER)
    cells[R2]["final"] = None
    res = score(tmp_path, cells, runs_ended=True)
    d = res["decision"]
    assert d["status"] == "decided" and d["line"] == 2 and len(d["counted_as_fails"]) == 1
    assert d["counted_as_fails"][0].endswith(f"{R2}/judge_final.json")
    assert "counted as fails" in SS.render(res)

def test_missing_report_only_file_is_not_incomplete(tmp_path):
    cells = arm_cells("LTS", STAGGER)
    cells[R1]["1.0M"] = judge("LTS", 52, "1.0M", eps=eps_with(40, "unclean"))  # 0/40 clean, report only
    res = score(tmp_path, cells)
    assert res["stagger"][R1]["cells"]["1.0M"]["n_clean"] == 0 and res["stagger"][R2]["cells"]["1.0M"] is None
    assert holds(res) == {R1: True, R2: True, R3: True}
    assert res["decision"]["status"] == "decided" and res["decision"]["line"] == 1


def test_missing_control_j_file_is_provisional_not_incomplete(tmp_path):
    ctrl = arm_cells("LT3", CONTROL, {C2: ["K1", "K2"], C3: ["K2"]})
    ctrl[C3]["4.2M"] = None
    res = score(tmp_path, arm_cells("LTS", STAGGER), control_cells=ctrl)
    assert res["decision"]["status"] == "provisional" and res["decision"]["line"] == 1
    assert any(w.startswith("control check: control J judge files missing or invalid") and f"{C3}/judge_4194304.json" in w
               for w in res["decision"]["provisional_why"])


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="stance3-judge/v1"),
    lambda d: d.update(arm="LT"),
    lambda d: d["git"].update(head=LT_HEAD),
    lambda d: d["arm_env_kwargs"].update(stance_stagger_weight=0.0),
    lambda d: d["noise_matched_control_joint"].update(sigma_vector=FROZEN["criteria"]["H4"]["control_sigma"]["52"]["vector"]),
    lambda d: d["checks"]["relaxed"].update(allow_commit_mismatch=True),
])
def test_invalid_stagger_file_fails_and_is_provisional(tmp_path, mutate):
    cells = arm_cells("LTS", STAGGER)
    d = copy.deepcopy(cells[R3]["6.3M"])  # Run 3 is seed 42: its frozen control sigma is seed 42's
    mutate(d)
    cells[R3]["6.3M"] = d
    res = score(tmp_path, cells)
    assert res["stagger"][R3]["criteria"]["failed"] == ["K1", "K2", "K3", "K4"] and len(res["stagger"][R3]["invalid"]) == 1
    assert res["decision"]["status"] == "provisional" and res["decision"]["line"] == 2  # H = 2, K3 2/3
    ended = score(tmp_path / "ended", cells, runs_ended=True)  # the runs have ended: a fail, no longer provisional
    assert ended["decision"]["status"] == "decided" and ended["decision"]["line"] == 2
    assert len(ended["decision"]["counted_as_fails"]) == 1


def test_head_placeholder_invalidates_every_stagger_file(tmp_path, monkeypatch):
    monkeypatch.setattr(SS, "HEAD_LTS", None)
    res = score(tmp_path, arm_cells("LTS", STAGGER), head=None)
    assert all(len(res["stagger"][r]["invalid"]) == 6 for r in STAGGER)
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 5
    assert any("HEAD_LTS is a PLACEHOLDER" in w for w in d["provisional_why"])


@pytest.mark.parametrize("pin, status", [("match", "decided"), ("placeholder", "provisional"), ("mismatch", "provisional")])
def test_criteria_pin(tmp_path, pin, status):
    res = score(tmp_path, arm_cells("LTS", STAGGER), pin=pin)
    assert res["decision"]["status"] == status and res["decision"]["line"] == 1


def test_head_override_is_provisional(tmp_path):
    other = "c" * 40
    cells = arm_cells("LTS", STAGGER)
    for slots in cells.values():
        for d in slots.values():
            d["git"]["head"] = d["checks"]["stage_config"]["git_commit"] = d["checks"]["stage_config"]["clone_head"] = other
    res = score(tmp_path, cells, head=other)
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 1
    assert any("is not the pinned HEAD_LTS" in w for w in d["provisional_why"])


def test_validation_is_provisional(tmp_path):
    res = score(tmp_path, arm_cells("LTS", STAGGER), validation=True)
    assert res["decision"]["status"] == "provisional"
    assert any("--validation" in w for w in res["decision"]["provisional_why"])


@pytest.mark.parametrize("const", ["TASK_SHA256_LTS", "JUDGE_SHA256_S4"])
def test_task_and_judge_placeholders_are_provisional(tmp_path, monkeypatch, const):
    monkeypatch.setattr(SS, const, None)
    res = score(tmp_path, arm_cells("LTS", STAGGER))
    assert res["decision"]["status"] == "provisional" and res["decision"]["line"] == 1
    assert any(f"{const} is a PLACEHOLDER" in w for w in res["decision"]["provisional_why"])


@pytest.mark.parametrize("key, value, text", [
    ("judge_sha256", "sha256:" + "d" * 64, "not judged by the recorded judge"),
    ("task_sha256", "sha256:" + "e" * 64, "is not the criteria's term-on"),
])
def test_wrong_judge_or_task_on_a_j_file_is_provisional(tmp_path, key, value, text):
    cells = arm_cells("LTS", STAGGER)
    (cells[R1]["8.4M"] if key == "judge_sha256" else cells[R1]["8.4M"]["panel"])[key] = value
    res = score(tmp_path, cells)
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 1 and holds(res)[R1]
    assert any(text in w for w in d["provisional_why"])


@pytest.mark.parametrize("how", ["null", "missing"])
def test_null_or_missing_task_on_a_stagger_j_file_is_provisional(tmp_path, how):
    """A valid stagger J file whose panel task_sha256 is null or missing is not the term-on task: provisional."""
    cells = arm_cells("LTS", STAGGER)
    panel = cells[R1]["8.4M"]["panel"]
    if how == "null":
        panel["task_sha256"] = None
    else:
        del panel["task_sha256"]
    res = score(tmp_path, cells)
    d = res["decision"]
    assert res["stagger"][R1]["invalid"] == [] and holds(res)[R1]  # the file stays valid; only the task check catches it
    assert d["status"] == "provisional" and d["line"] == 1
    assert res["stagger"][R1]["task_sha256_seen"] == [SS.TASK_SHA256_LTS, None]
    want = (f"{R1}: J task_sha256 [{SS.TASK_SHA256_LTS!r}, None] is not the criteria's term-on {SS.TASK_SHA256_LTS} "
            "(J files ['8.4M']; None is a null or missing one)")
    assert [w for w in d["provisional_why"] if "task_sha256" in w] == [want]


@pytest.mark.parametrize("how", ["null", "missing"])
def test_null_or_missing_task_on_a_report_only_stagger_file_is_a_note(tmp_path, how):
    cells = arm_cells("LTS", STAGGER)
    cells[R2]["2.1M"] = judge("LTS", 60, "2.1M")
    if how == "null":
        cells[R2]["2.1M"]["panel"]["task_sha256"] = None
    else:
        del cells[R2]["2.1M"]["panel"]["task_sha256"]
    res = score(tmp_path, cells)
    assert res["decision"]["status"] == "decided" and res["decision"]["line"] == 1
    assert any(n.startswith(f"{R2}: a report-only file's task_sha256") for n in res["notes"])


@pytest.mark.parametrize("how", ["null", "missing", "other"])
def test_control_j_file_must_carry_the_lt_task(tmp_path, monkeypatch, how):
    """Every valid control J file must carry L+T's task_sha256; a null, missing or other one fails the control check."""
    ctrl = arm_cells("LT3", CONTROL, {C2: ["K1", "K2"], C3: ["K2"]})
    record = synthetic_record(tmp_path, ctrl, monkeypatch)
    ok = score(tmp_path / "ok", arm_cells("LTS", STAGGER), control_cells=ctrl, score_final=record)
    assert ok["decision"]["status"] == "decided" and ok["control_crosscheck"]["problems"] == []
    panel = ctrl[C3]["5.2M"]["panel"]
    if how == "null":
        panel["task_sha256"] = None
    elif how == "missing":
        del panel["task_sha256"]
    else:
        panel["task_sha256"] = SS.TASK_SHA256_LTS
    res = score(tmp_path / "bad", arm_cells("LTS", STAGGER), control_cells=ctrl, score_final=record)
    x = res["control_crosscheck"]
    assert x["runs"][C2]["task_sha256_match"] and not x["runs"][C3]["task_sha256_match"]
    assert x["runs"][C3]["verdicts_match"] and x["runs"][C3]["criteria_identical"]  # only the task differs
    assert not x["reproduced"] and not x["match"]
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 1
    want = ("control check: the control does not reproduce from its J judge files (verdicts, criteria or task_sha256): "
            f"['{C3}']")
    assert d["provisional_why"] == [want]


# ------------------------------------------------------------------------------------------------ parents
def test_parent_block_is_inputs_only_while_incomplete(tmp_path):
    """Incomplete (a stagger J file missing, the runs not ended): the parent block is labelled inputs only and offers no
    parent; once the runs have ended, the same files give an offer."""
    cells = arm_cells("LTS", STAGGER, final={R1: {"joint_falls": 1}, R2: {"own_falls": 1}})
    cells[R3]["final"] = None
    res = score(tmp_path, cells)
    assert res["decision"]["status"] == "incomplete"
    p = res["parents"]
    assert p["inputs_only"] is True
    w = p["walking_pilot_2"]
    assert w["offered"] is None and w["stays_incumbent"] is None and w["tie_with_incumbent"] is None
    assert [c["run"] for c in w["candidates"]]  # the inputs are still listed
    out = SS.render(res)
    assert "INPUTS ONLY: the decision is incomplete" in out
    assert "walking pilot 2 parent: none offered (inputs only)" in out
    assert "parent offered:" not in out and "the incumbent stays" not in out
    ended = score(tmp_path / "ended", cells, runs_ended=True)
    assert ended["decision"]["status"] == "decided" and ended["parents"]["inputs_only"] is False
    assert ended["parents"]["walking_pilot_2"]["offered"] == "stance3 Run 2 stance3_armLT_s60"
    out = SS.render(ended)
    assert "INPUTS ONLY" not in out
    assert "walking pilot 2 parent offered: stance3 Run 2 stance3_armLT_s60  (the incumbent stays)" in out


def test_parent_ties_with_the_incumbent(tmp_path):
    """Every stagger final at 40/40 with 0 noise falls ties stance3 Run 2's final (40/40, 0/0): no strict winner."""
    res = score(tmp_path, arm_cells("LTS", STAGGER))
    p = res["parents"]
    assert p["incumbent"] == {"run": "stance3 Run 2 stance3_armLT_s60", "clean": 40, "joint_noise_falls": 0, "own_noise_falls": 0}
    w = p["walking_pilot_2"]
    assert w["offered"] is None and w["tie_with_incumbent"] and len(w["tied"]) == 4
    assert p["best_stagger_final"]["best"] is None and len(p["best_stagger_final"]["tied"]) == 3
    out = SS.render(res)
    assert "walking pilot 2 parent offered: TIE: " in out and "(the maintainer chooses)" in out
    assert "both are named; the maintainer chooses" in out  # line 1, and the holding finals tie too


def test_parent_incumbent_stays(tmp_path):
    final = {R1: {"joint_falls": 1}, R2: {"own_falls": 1}, R3: {"eps": eps_with(1, "unclean")}}
    res = score(tmp_path, arm_cells("LTS", STAGGER, final=final))
    p = res["parents"]
    assert p["walking_pilot_2"]["offered"] == "stance3 Run 2 stance3_armLT_s60" and p["walking_pilot_2"]["stays_incumbent"]
    assert p["best_stagger_final"]["tied"] == ["Run 1 stance4_LTS_s52", "Run 2 stance4_LTS_s60"]


def test_parent_stagger_beats_a_weaker_incumbent_and_holding_rule(tmp_path):
    ctrl = arm_cells("LT3", CONTROL, {C2: ["K2"], C3: ["K2"]}, final={C2: {"eps": eps_with(1, "unclean")}})
    final = {R1: {}, R2: {"eps": eps_with(1, "unclean")}, R3: {"eps": eps_with(2, "unclean")}}
    res = score(tmp_path, arm_cells("LTS", STAGGER, {R1: ["K2"]}, final=final), control_cells=ctrl)
    p = res["parents"]
    assert p["best_stagger_final"]["best"] == "Run 1 stance4_LTS_s52"           # 40/40, but it does not HOLD
    assert p["best_holding_stagger_final"]["best"] == "Run 2 stance4_LTS_s60"   # 39/40 beats 38/40
    assert p["walking_pilot_2"]["offered"] == "Run 1 stance4_LTS_s52" and not p["walking_pilot_2"]["stays_incumbent"]


# ------------------------------------------------------------------------------------------------ report only
def test_control_report_only_from_the_stance4_judge(tmp_path):
    cells = arm_cells("LTS", STAGGER)
    cells[C3] = {"11.5M": judge("LT4", 52, "11.5M", eps=eps_with(2, "unclean")), "7.3M": judge("LT3", 52, "7.3M")}
    res = score(tmp_path, cells)
    c = res["control"][C3]
    assert c["cells"]["11.5M"]["n_clean"] == 38      # read from the stance4 tree (stance4 schema, arm LT)
    assert c["cells"]["3.1M"]["n_clean"] == 40       # not there: read from stance3's judged tree
    assert c["cells"]["7.3M"] is None and len(c["report_invalid"]) == 1  # a stance3-schema file in the stance4 tree
    assert c["criteria"]["failed"] == ["K2"] and res["decision"]["status"] == "decided"


def _sat_rows(counts, every=131072):
    return [{"timesteps": str((i + 1) * every),
             "screen_v2_reasons": "max_actuator_saturation_fraction;max_sole_cop_fore_aft" if j < k else ""}
            for i, k in enumerate(counts) for j in range(10)]


@pytest.mark.parametrize("counts, alarm, first, flagged", [
    ([5, 5], True, 2 * 131072, [131072, 262144]),
    ([5, 4, 5], False, None, [131072, 393216]),
    ([4, 5, 5, 0, 10, 10, 10], True, 3 * 131072, [262144, 393216, 655360, 786432, 917504]),
    ([4, 4, 0], False, None, []),
])
def test_saturation_alarm(counts, alarm, first, flagged):
    a = SS.saturation_alarm(_sat_rows(counts))
    assert a["computed"] and a["alarm"] is alarm and a["first_alarm_at"] == first and a["flagged"] == flagged


def test_saturation_alarm_needs_the_column():
    assert SS.saturation_alarm([{"timesteps": "131072"}])["computed"] is False
    assert SS.saturation_alarm([])["computed"] is False


def test_saturation_alarm_from_a_run_ledger(tmp_path):
    res = score(tmp_path, arm_cells("LTS", STAGGER), ledgers={R2: _sat_rows([0, 6, 7, 0])})
    a = res["stagger"][R2]["saturation_alarm"]
    assert a["alarm"] and a["first_alarm_at"] == 3 * 131072 and a["runs"][0]["evaluations"] == 2
    assert res["stagger"][R1]["saturation_alarm"]["computed"] is False
    assert holds(res) == {R1: True, R2: True, R3: True}   # report only


# ------------------------------------------------------------------------------------------------ command line
def test_command_line(tmp_path):
    out = tmp_path / "score.json"
    cmd = [sys.executable, "-I", "-B", os.path.join(HERE, "score_stance4.py"), "--judged", str(tmp_path / "empty"),
           "--runs-root", str(tmp_path / "runs"), "--json", str(out)]
    # the file's own pins: the decision is provisional only while a pin is empty or the criteria file is not its pin
    empty = sorted(k for k, v in FILE_PINS.items() if v is None)
    pinned = not empty and SS.sha256_file(SS.CRITERIA_FILE) == FILE_PINS["CRITERIA_SHA256"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "DECISION: incomplete (18 stagger J judge files missing" in p.stdout
    assert "control verdicts reproduce stance3 score_final.json: yes" in p.stdout
    assert "INPUTS ONLY" in p.stdout and "parent offered:" not in p.stdout
    assert ("   provisional: " in p.stdout) is not pinned
    for k in empty:
        assert f"{k} is a PLACEHOLDER" in p.stdout
    d = json.loads(out.read_text())
    assert d["schema"] == "stance4-score/v1" and d["decision"]["status"] == "incomplete"
    assert (d["decision"]["provisional_why"] == []) is pinned
    p2 = subprocess.run(cmd, capture_output=True, text=True)
    assert p2.returncode != 0 and "refusing to overwrite" in p2.stderr
    p3 = subprocess.run(cmd[:-2] + ["--runs-ended"], capture_output=True, text=True)  # every J file counted as a fail
    assert p3.returncode == 0, p3.stderr
    assert "18 stagger J judge files missing or invalid" in p3.stdout
    if pinned:
        assert "DECISION: 5. H == 0" in p3.stdout and "(PROVISIONAL)" not in p3.stdout
        assert "walking pilot 2 parent offered: stance3 Run 2 stance3_armLT_s60  (the incumbent stays)" in p3.stdout
    else:
        assert "DECISION (PROVISIONAL): 5. H == 0" in p3.stdout
