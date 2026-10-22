"""score_stance5.py on hand-made band judge files, beside the real stance4 control (and synthetic ones).

    python3 -m pytest -p no:cacheprovider test_score_stance5.py --basetemp=<work>/tmp

Covers: the control rows reproduce stance4's score_final.json exactly; K1-K4 are score_stance3.py's code (all three
module instances against a fresh import, and the control instance against score_stance4.py's own); every decision-table
line and its boundaries (H = 3, 2, 1, 0; K2p = 2 against 1; D = 2 against 1; S = 0 against 1; line 2 only through a
control that is not the stated one, and then provisional), each line's text verbatim from criteria.txt; the DRIFT OR
SPLIT classification (COMMON and SPLIT episodes, K2a-only episodes, near-reference episodes and the 15 mm margin, counts
only at the J checkpoints where K2a or K2b fails, n_c == n_s with K2c failing and passing, a K2c-only failure with and
without episodes under the bar, falls only, a K2-passing run that is not classed, a missing file before and after the
runs end, an invalid file, and the real stance3 runs, arm L s60 among them); (b) in its general form against
synthetic control records; the control's passes come from the record, not from its recomputed files; a missing J file
(incomplete) against a missing report-only one, and both after the runs have ended (--runs-ended); invalid files
(provisional, then fails once the runs have ended); a placed J checkpoint without its judge file after the runs have
ended (provisional); the HEAD, criteria, task and judge pins and placeholders; a null or
missing task_sha256 on a band or control J file; --validation; the parent comparison, its ties and its inputs-only form;
the report-only band readouts and saturation alarm; --validate-on-stance4 (stance4's K results reproduced); the command
line.

The hand-made band files carry a test HEAD and a test judge sha256, so an autouse fixture sets HEAD_LTSB,
JUDGE_SHA256_S5 and TASK_SHA256_LTSB to those values; the tests that check the placeholders unset them.  The
command-line test runs the file as it stands (its own pins, FILE_PINS).
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
import score_stance5 as SS  # noqa: E402

#: the pins as they stand in score_stance5.py (read before the autouse fixture replaces any)
FILE_PINS = {k: getattr(SS, k) for k in ("HEAD_LTSB", "CRITERIA_SHA256", "TASK_SHA256_LTSB", "JUDGE_SHA256_S5")}
SC3, SC4, SC5 = SS.SC3, SS.SC4, SS.SC5
FROZEN = json.load(open(SC3.FROZEN_PARAMS))
TEST_HEAD = "a" * 40
TEST_JUDGE_SHA = "sha256:" + "b" * 64
TEST_TASK = "sha256:8c6998d260b850ae054c9c2d49a8e5bef2bf59d4770ad77d3674b911721918cf"
SLOTS = {**dict(SC3.J), **dict(SS.REPORT_1M)}
BAND = {run: seed for run, _arm, seed, _name in SS.BAND_RUNS}
CONTROL = {run: seed for run, _arm, seed, _name in SS.CONTROL_RUNS}
R1, R2, R3 = (r[0] for r in SS.BAND_RUNS)
C1, C2, C3 = (r[0] for r in SS.CONTROL_RUNS)
KINDS = {  # schema, arm, head, [env], task, judge sha256
    "LTSB": (SS.JUDGE_SCHEMA_S5, "LTSB", TEST_HEAD, {**SC3.ARM_ENV_COMMON, **SS.ARM_ENV_LTSB}, TEST_TASK, TEST_JUDGE_SHA),
    "LTS4": (SS.JUDGE_SCHEMA_S4, "LTS", SS.HEAD_LTS, {**SC3.ARM_ENV_COMMON, **SS.ARM_ENV_LTS}, SS.TASK_SHA256_LTS,
             SS.JUDGE_SHA256_S4),
}
STATUE_W = 0.1   # every statue foot reads +0.1, so m0 = 0.1 exactly; a foot meant to vote lies >= 15/70 from it
STATUE_XC = 0.065


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    """The hand-made band files' HEAD, judge sha256 and task as the scorer's pins (the file's own pins: FILE_PINS)."""
    monkeypatch.setattr(SS, "HEAD_LTSB", TEST_HEAD)
    monkeypatch.setattr(SS, "JUDGE_SHA256_S5", TEST_JUDGE_SHA)
    monkeypatch.setattr(SS, "TASK_SHA256_LTSB", TEST_TASK)


# ------------------------------------------------------------------------------------------------ synthetic files
def episode(seed, *, full=True, clean=True, pad=0.4, w=(0.1, 0.1), xc=0.066, tilt=0.5, lift=0.001, hop=False, share=False):
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
    if share:
        reasons.append("min_foot_load_share_windowed: 0.3 < 0.35")
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
    alter = {"unclean": {"clean": False}, "w": {"w": kw.get("w")}, "tilt": {"tilt": 2.01}, "pad": {"pad": 0.81, "w": kw.get("w", (0.1, 0.1))},
             "fall": {"full": False}, "xc": {"xc": kw.get("xc")}}[kind]
    eps = [episode(3042 + i, **(alter if i < n_bad else {})) for i in range(40)]
    if kind == "xc":  # the whole panel shifted (K2c reads the panel median)
        eps = [episode(3042 + i, xc=kw.get("xc")) for i in range(40)]
    return eps


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


def judge(kind, seed, label, *, eps=None, joint_falls=0, own_falls=0, band_summary=None):
    """A valid judge file of *kind* (LTSB: a band run; LTS4: a stance4 control file) for *seed* at slot *label*; eps
    defaults to 40 clean, full-horizon, centred episodes."""
    schema, arm, head, env, task, jsha = KINDS[kind]
    eps = eps if eps is not None else [episode(3042 + i) for i in range(40)]
    k = sum(e["clean"] for e in eps)
    full = sum(e["reached_horizon"] for e in eps)
    hop = sum((not e["reached_horizon"]) or any(r.startswith("min_all_feet_support") for r in e["reasons"]) for e in eps)
    lcb = SC3.binomial_lcb(k, 40)
    verdict = "PASS" if (k >= 37 and lcb >= 0.8 and full >= 38 and hop <= 1) else "FAIL"
    steps = SLOTS.get(label, label)
    stem = "stage1_final" if steps == "final" else f"{steps}_model" if steps in ("best", "robust_best") else f"stage1_{steps}_steps"
    d = {"schema": schema, "mode": "full", "smoke": False, "arm": arm, "seed": seed, "steps": steps, "checkpoint": stem,
         "judge_sha256": jsha,
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
                   "terminations": {"truncated": full}, "scored_gate_kind": "stance_quality/v2", "task_sha256": task,
                   "thresholds": dict(SC3.R9_BARS), "statue_on_same_seeds": {"n": 40},
                   "statue_episodes_detail": [episode(3042 + i, full=(i != 39), xc=STATUE_XC, w=(STATUE_W, STATUE_W))
                                              for i in range(40)],
                   "summary": {}, "episodes_detail": eps}}
    if band_summary is not None:
        d["panel"]["band_summary"] = band_summary
    if label in ("6.3M", "final"):
        d["noise_matched_control_joint"] = noise(seed, "joint", joint_falls)
        d["noise_matched_own"] = noise(seed, "policy", own_falls)
    return d


#: How each named failure is made (all at 8.4M unless said; m0 = 0.1, and every voting foot lies >= 15/70 from it): K1
#: 4 unclean at 6.3M; K2 (a split) 4 episodes r +0.7, l -0.3; K2common 4 episodes r +0.7, l +0.65; K2pad (K2a only,
#: common) 4 pad-over episodes r +0.4, l +0.35; K2padsplit (K2a only, split) r +0.4, l -0.2; K2near (one-foot
#: excursions) 4 episodes r +0.7, l +0.104; K2c the whole panel's x_c +45 mm off the statue; K2fall 4 falls (K1 fails
#: too); K3 3/20 joint falls at final; K4 four tilted episodes at final (K1 fails too).
FAILS = {
    "K1": ("6.3M", lambda: {"eps": eps_with(4, "unclean")}),
    "K2": ("8.4M", lambda: {"eps": eps_with(4, "w", w=(0.7, -0.3))}),
    "K2common": ("8.4M", lambda: {"eps": eps_with(4, "w", w=(0.7, 0.65))}),
    "K2pad": ("8.4M", lambda: {"eps": eps_with(4, "pad", w=(0.4, 0.35))}),
    "K2padsplit": ("8.4M", lambda: {"eps": eps_with(4, "pad", w=(0.4, -0.2))}),
    "K2near": ("8.4M", lambda: {"eps": eps_with(4, "w", w=(0.7, 0.104))}),
    "K2c": ("8.4M", lambda: {"eps": eps_with(0, "xc", xc=0.110)}),
    "K2fall": ("8.4M", lambda: {"eps": eps_with(4, "fall")}),
    "K3": ("final", lambda: {"joint_falls": 3}),
    "K4": ("final", lambda: {"eps": eps_with(4, "tilt")}),
}


def arm_cells(kind, runs: dict, fail: dict | None = None, final: dict | None = None, at: dict | None = None) -> dict:
    """{run: {label: file}} for *runs* ({run: seed}); fail = {run: [names of FAILS]}; final = {run: judge kwargs}
    overrides final; at = {run: {label: judge kwargs}} overrides any slot (after fail)."""
    out = {}
    for run, seed in runs.items():
        kw = {lab: {} for lab, _ in SC3.J}
        for name in (fail or {}).get(run, []):
            lab, make = FAILS[name]
            kw[lab].update(make())
        kw["final"].update((final or {}).get(run, {}))
        for lab, extra in ((at or {}).get(run) or {}).items():
            kw.setdefault(lab, {}).update(extra)
        out[run] = {lab: judge(kind, seed, lab, **kw[lab]) for lab in kw}
    return out


def write_tree(root, cells: dict) -> str:
    for run, slots in cells.items():
        os.makedirs(os.path.join(root, run), exist_ok=True)
        for lab, d in slots.items():
            if d is not None:
                with open(os.path.join(root, run, f"judge_{SLOTS.get(lab, lab)}.json"), "w") as fh:
                    json.dump(d, fh)
    return str(root)


def score(tmp, band_cells, *, control_cells=None, head=TEST_HEAD, pin="match", ledgers=None, runs_ended=False,
          score_final=SS.STANCE4_SCORE_FINAL, validation=False):
    """Write the trees under *tmp* and score them; the control is the real stance4 one unless *control_cells* is given."""
    judged = write_tree(tmp / "judged", band_cells)
    control_root = SS.S4_JUDGED if control_cells is None else write_tree(tmp / "control", control_cells)
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
    res = SS.score(judged, control_root, runs_root=str(runs_root),
                   control_runs_root=SS.S4_RUNS if control_cells is None else str(tmp / "control_runs"),
                   head_ltsb=head, criteria_file=str(crit), criteria_sha256=sha, runs_ended=runs_ended,
                   score_final=str(score_final), validation=validation)
    cr = res["credit_rule"]
    assert cr["forms_agree"] and cr["a_implies_b"]   # the two forms of (b) agree while the control stands as stated
    SS.render(res)  # renders without error
    return res


def holds(res):
    return {run: res["band"][run]["criteria"]["HOLDS"] for run in BAND}


def classes(res):
    return {run: res["band"][run]["drift_split"]["class"] for run in BAND}


def synthetic_record(tmp, ctrl_cells, monkeypatch) -> str:
    """A score_final.json-like record of a synthetic control (its runs under "stagger", as stance4's scorer stores
    them): its recomputed criteria; its sha256 is pinned in place of the real file's."""
    first = score(tmp / "first", arm_cells("LTSB", BAND), control_cells=ctrl_cells)
    rec = {"stagger": {run: {"criteria": json.loads(json.dumps(first["control"][run]["criteria"]))} for run in CONTROL}}
    path = tmp / "score_final_synthetic.json"
    path.write_text(json.dumps(rec))
    monkeypatch.setattr(SS, "STANCE4_SCORE_FINAL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return str(path)


# ------------------------------------------------------------------------------------------------ the control
def test_control_reproduces_stance4_score_final(tmp_path):
    res = SS.score(str(tmp_path / "none"), SS.S4_JUDGED, runs_root=str(tmp_path / "r"), control_runs_root=SS.S4_RUNS,
                   head_ltsb=TEST_HEAD)
    with open(SS.STANCE4_SCORE_FINAL) as fh:
        stored = json.load(fh)
    assert SS.sha256_file(SS.STANCE4_SCORE_FINAL) == SS.STANCE4_SCORE_FINAL_SHA256
    for run in CONTROL:
        got, want = res["control"][run], stored["stagger"][run]
        assert json.loads(json.dumps(got["criteria"])) == want["criteria"]
        for lab in list(SS.J_LABELS) + [lab for lab, _ in SS.REPORT_1M] + ["robust_best", "best"]:
            if lab in SS.J_LABELS or want["cells"].get(lab) is not None:  # best / robust_best of s52 were judged after
                assert json.loads(json.dumps(got["cells"][lab])) == want["cells"][lab], (run, lab)  # the record
        assert json.loads(json.dumps(got["k1_failure_modes"])) == want["k1_failure_modes"]
        assert json.loads(json.dumps(got["saturation_alarm"])) == want["saturation_alarm"]
        assert got["invalid"] == [] and got["missing"] == []
    assert res["control"][C1]["criteria"]["failed"] == ["K1", "K2"]
    assert res["control"][C2]["criteria"]["failed"] == ["K1", "K2"]
    assert res["control"][C3]["criteria"]["failed"] == ["K1", "K2", "K3", "K4"]
    x = res["control_crosscheck"]
    assert x["match"] and x["stated_match"] and all(r["criteria_identical"] for r in x["runs"].values())
    assert all(r["task_sha256_seen"] == [SS.TASK_SHA256_LTS] for r in x["runs"].values()) and not x["problems"]
    assert res["credit_rule"]["control_passes"] == {"K1": 0, "K2": 0, "K3": 2, "K4": 2}
    assert res["credit_rule"]["control_as_stated"]
    assert res["credit_rule"]["control_source"] == "recorded in stance4's score_final.json"
    # the control's own drift or split classes, report only (criteria READOUTS; criteria_calib/drift_split_calib.txt)
    got = {run: (r["drift_split"]["class"], r["drift_split"]["n_common"], r["drift_split"]["n_split"],
                 r["drift_split"]["n_near_reference"], r["drift_split"]["not_classed"]) for run, r in res["control"].items()}
    assert got == {C1: ("SPLIT", 0, 101, 0, 0), C2: ("SPLIT", 0, 67, 2, 0), C3: ("SPLIT", 0, 27, 1, 3)}
    assert res["control"][C3]["drift_split"]["counted"] == ["5.2M", "6.3M", "10.5M", "final"]  # 4.2M: K2a 0, K2b 2
    near = {(run, lab, c["seed"]) for run, r in res["control"].items() for lab in r["drift_split"]["counted"]
            for c in r["readouts"][lab]["classed"] if c["class"] == "NEAR-REFERENCE"}
    assert near == {(C2, "8.4M", 3052), (C2, "8.4M", 3081), (C3, "final", 3080)}   # criteria READOUTS
    for run in CONTROL:  # the statue's m0 is +0.0577 in every control J file (criteria)
        assert {round(v["m0"], 4) for v in res["control"][run]["drift_split"]["per_checkpoint"].values()} == {0.0577}
    # the band readouts the criteria quote from the control's J files: 20, 22 and 11 of 40 outside [-0.6, +0.4] at 4.2M
    assert [res["control"][r]["readouts"]["4.2M"]["band_outside_fh"] for r in (C1, C2, C3)] == [20, 22, 11]
    ref = res["parents"]["reference"]
    assert (ref["clean"], ref["joint_noise_falls"], ref["own_noise_falls"], ref["matches_stance4_record"]) == (40, 0, 0, True)


# ------------------------------------------------------------------------------------------------ same code
def _fresh(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_k_code_is_score_stance3(tmp_path):
    fresh = _fresh(SS.SC3_PATH, "fresh_score_stance3")
    assert SS.sha256_file(SS.SC3_PATH) == SS.SC3_SHA256
    score(tmp_path, arm_cells("LTSB", BAND))  # a full scoring pass first: nothing it does may leave a change behind
    funcs = [n for n, v in vars(fresh).items() if isinstance(v, types.FunctionType)]
    assert {"run_criteria", "panel_counts", "validity_issues", "load_cell", "k1_failure_modes", "best_final",
            "k2a_over", "k2b_outside"} <= set(funcs)
    for n in funcs:
        assert getattr(SC3, n).__code__ == getattr(fresh, n).__code__ == getattr(SC4, n).__code__ \
            == getattr(SC5, n).__code__, n
    for n, v in vars(fresh).items():
        if n.startswith("__") or isinstance(v, (types.FunctionType, types.ModuleType, type)):
            continue
        assert getattr(SC3, n) == v, n
        if n not in SS.PATCHED:
            assert getattr(SC4, n) == v and getattr(SC5, n) == v, n
    assert SC5.JUDGE_SCHEMA == "stance5-judge/v1" and SC4.JUDGE_SCHEMA == "stance4-judge/v1"
    assert SC3.JUDGE_SCHEMA == "stance3-judge/v1"
    for k in ("L", "LT"):
        assert SC5.ARM_ENV[k] == fresh.ARM_ENV[k] == SC4.ARM_ENV[k]
    assert SC5.ARM_ENV["LTSB"] == {**fresh.ARM_ENV["LT"], "stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045,
                                   "stance_cop_band_weight": 0.30}
    assert {k: SC5.HEADS[k] for k in ("L", "LT")} == fresh.HEADS
    src = open(os.path.join(HERE, "score_stance5.py")).read()
    for name in ("run_criteria", "panel_counts", "validity_issues", "load_cell", "k2a_over", "k2b_outside",
                 "binomial_lcb", "best_final", "k1_failure_modes"):
        assert f"def {name}(" not in src, name


def test_control_instance_is_score_stance4s():
    """SC4 here is score_stance4.py's stagger instance: the same identity constants, the same code."""
    assert SS.sha256_file(SS.SC4_SCORER_PATH) == SS.SC4_SCORER_SHA256
    s4 = _fresh(SS.SC4_SCORER_PATH, "fresh_score_stance4")
    for n in SS.PATCHED:
        assert getattr(SC4, n) == getattr(s4.SC4, n), n
    assert SS.HEAD_LTS == s4.HEAD_LTS and SS.TASK_SHA256_LTS == s4.TASK_SHA256_LTS
    assert SS.JUDGE_SHA256_S4 == s4.JUDGE_SHA256_S4 and SS.ARM_ENV_LTS == s4.ARM_ENV_LTS
    assert SS.saturation_alarm.__code__.co_code == s4.saturation_alarm.__code__.co_code


def test_same_files_same_criteria_in_both_instances(tmp_path):
    """The same episodes scored as a stance4 control file (SC4) and as a band file (SC5) give identical K1-K4."""
    band = arm_cells("LTSB", {R1: 52}, {R1: ["K2", "K3"]})
    ctl = arm_cells("LTS4", {C1: 52}, {C1: ["K2", "K3"]})
    write_tree(tmp_path / "B", band)
    write_tree(tmp_path / "C", ctl)
    notes = []
    a = SS.load_run(SC5, str(tmp_path / "B"), R1, "LTSB", 52, "x", head=TEST_HEAD, frozen=FROZEN, validation=False,
                    notes=notes, report_sources=[], runs_root=str(tmp_path), ledger_dir=None)
    b = SS.load_run(SC4, str(tmp_path / "C"), C1, "LTS", 52, "x", head=SS.HEAD_LTS, frozen=FROZEN, validation=False,
                    notes=notes, report_sources=[], runs_root=str(tmp_path), ledger_dir=None)
    assert a["criteria"] == b["criteria"] and a["criteria"]["failed"] == ["K2", "K3"]
    assert not notes


# ------------------------------------------------------------------------------------------------ the decision table
@pytest.mark.parametrize("fail, H, K2p, D, S, line", [
    ({}, 3, 3, 0, 0, 1),
    ({R3: ["K1"]}, 2, 3, 0, 0, 1),                               # K1 2/3 >= control 0/3
    ({R3: ["K2"]}, 2, 2, 0, 1, 1),                               # K2 2/3 >= control 0/3
    ({R3: ["K3"]}, 2, 3, 0, 0, 1),                               # K3 2/3 >= control 2/3: (a) implies (b)
    ({R1: ["K4"]}, 2, 3, 0, 0, 1),                               # K4 (and K1) 2/3
    ({R1: ["K2"], R2: ["K3"]}, 1, 2, 0, 1, 3),
    ({R1: ["K2common"], R2: ["K2common"]}, 1, 1, 2, 0, 3),      # H = 1 decides before D (D and S printed only)
    ({R1: ["K3"], R2: ["K3"], R3: ["K3"]}, 0, 3, 0, 0, 4),
    ({R1: ["K3"], R2: ["K1"], R3: ["K2common"]}, 0, 2, 1, 0, 4),  # boundary: K2p = 2 (line 4 whatever D)
    ({R1: ["K2common"], R2: ["K2common"], R3: ["K3"]}, 0, 1, 2, 0, 5),   # boundary: K2p = 1, D = 2, S = 0 (the trigger)
    ({R1: ["K2common"], R2: ["K2common"], R3: ["K2common"]}, 0, 0, 3, 0, 5),
    ({R1: ["K2common"], R2: ["K2common"], R3: ["K2"]}, 0, 0, 2, 1, 6),   # boundary: D = 2 but S = 1 (split not fixed)
    ({R1: ["K2common"], R2: ["K2common"], R3: ["K2fall"]}, 0, 0, 2, 0, 5),  # the third run OTHER (falls only): S = 0
    ({R1: ["K2common"], R2: ["K2"], R3: ["K3"]}, 0, 1, 1, 1, 6),         # boundary: D = 1
    ({R1: ["K2"], R2: ["K2"], R3: ["K2"]}, 0, 0, 0, 3, 6),               # all SPLIT: stance4's own pattern
    ({R1: ["K2c"], R2: ["K2c"], R3: ["K4"]}, 0, 1, 2, 0, 5),             # K2c-only failures are DRIFT
    ({R1: ["K2fall"], R2: ["K2fall"], R3: ["K2fall"]}, 0, 0, 0, 0, 6),   # falls only: OTHER, not DRIFT
    ({R1: ["K2near"], R2: ["K2near"], R3: ["K3"]}, 0, 1, 0, 0, 6),       # one-foot excursions only: OTHER
])
def test_decision_lines(tmp_path, fail, H, K2p, D, S, line):
    res = score(tmp_path, arm_cells("LTSB", BAND, fail))
    cr, d = res["credit_rule"], res["decision"]
    assert (cr["H"], cr["K2p"], cr["D"], cr["S"], cr["a"]) == (H, K2p, D, S, H >= 2)
    assert cr["b"] == (cr["band_passes"]["K3"] >= 2 and cr["band_passes"]["K4"] >= 2) == cr["b_specific"]
    assert d["status"] == "decided" and d["line"] == line and d["text"] == SS.DECISION_LINES[line]
    assert res["control_crosscheck"]["match"]


def _criteria_table_items() -> dict:
    """criteria.txt's DECISION TABLE items, each with its continuation lines (newlines kept)."""
    import re
    lines = open(SS.CRITERIA_FILE).read().splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith("DECISION TABLE"))
    items, cur = {}, None
    for s in lines[i + 1:]:
        if s.startswith("Note:"):
            break
        m = re.match(r"^(\d)\. ", s)
        if m:
            cur = int(m.group(1))
            items[cur] = [s]
        else:
            items[cur].append(s)
    return {n: "\n".join(v) for n, v in items.items()}


def test_decision_lines_are_verbatim():
    """Every printed decision line is criteria.txt's own text, word for word (stance4's scorer printed every line verbatim)."""
    assert SS.sha256_file(SS.CRITERIA_FILE) == FILE_PINS["CRITERIA_SHA256"]
    assert _criteria_table_items() == SS.DECISION_LINES


def test_d_and_s_are_printed_at_h_1(tmp_path):
    """At H == 1 line 3 decides; D, S and each run's class are still printed (report only)."""
    res = score(tmp_path, arm_cells("LTSB", BAND, {R1: ["K2common"], R2: ["K2common"]}))
    out = SS.render(res)
    assert res["decision"]["line"] == 3 and "D = 2 (K2-failing band runs classed DRIFT); S = 0" in out
    assert f"{R1} DRIFT (4, 0, 0)" in out and f"{R3} not classed (K2 passes)" in out


def test_line_2_needs_a_control_other_than_the_stated_one(tmp_path, monkeypatch):
    """A control record passing K3 on all three runs: a band arm at K3 2/3 is worse, so (b) fails at H = 2: line 2,
    and the decision is PROVISIONAL (the record is not the stated control, and line 2 was reached)."""
    ctrl = arm_cells("LTS4", CONTROL, {C1: ["K1", "K2"], C2: ["K1", "K2"], C3: ["K1", "K2", "K4"]})
    rec = synthetic_record(tmp_path, ctrl, monkeypatch)
    res = score(tmp_path, arm_cells("LTSB", BAND, {R3: ["K3"]}), control_cells=ctrl, score_final=rec)
    cr, d = res["credit_rule"], res["decision"]
    assert cr["control_passes"] == {"K1": 0, "K2": 0, "K3": 3, "K4": 2} and not cr["control_as_stated"]
    assert cr["H"] == 2 and not cr["b"] and cr["b_by_test"]["K3"] == {"band": "2/3", "control": "3/3", "no_worse": False}
    assert d["line"] == 2 and d["status"] == "provisional"
    assert any("not the ones the criteria state" in w for w in d["provisional_why"])
    assert any(w.startswith("decision line 2 reached") for w in d["provisional_why"])


def test_general_form_uses_the_controls_actual_scores(tmp_path, monkeypatch):
    """A control record passing K1 on one run: a band arm at K1 0/3 is worse, so (b) fails in general form (provisional:
    not the stated control); the specific form, valid only for the stated control, would say it holds."""
    ctrl = arm_cells("LTS4", CONTROL, {C1: ["K2"], C2: ["K1", "K2"], C3: ["K1", "K2", "K3", "K4"]})
    rec = synthetic_record(tmp_path, ctrl, monkeypatch)
    res = score(tmp_path, arm_cells("LTSB", BAND, {R1: ["K1"], R2: ["K1"], R3: ["K1"]}), control_cells=ctrl,
                score_final=rec)
    cr = res["credit_rule"]
    assert cr["control_passes"] == {"K1": 1, "K2": 0, "K3": 2, "K4": 2} and not cr["control_as_stated"]
    assert not cr["b"] and cr["b_specific"] and cr["b_by_test"]["K1"]["no_worse"] is False
    assert res["decision"]["line"] == 4 and res["decision"]["status"] == "provisional"


def test_control_passes_come_from_the_record_not_the_files(tmp_path):
    """Without stance4_LTS_s52's final judge file the recomputed control would stand at K3 1/3, K4 1/3; the recorded
    passes (K3 2/3, K4 2/3) stay the credit rule's input, and the decision is provisional."""
    ctrl = arm_cells("LTS4", CONTROL, {C1: ["K1", "K2"], C2: ["K1", "K2"], C3: ["K1", "K2", "K3", "K4"]})
    ctrl[C1]["final"] = None
    res = score(tmp_path, arm_cells("LTSB", BAND, {R3: ["K3"]}), control_cells=ctrl)
    rec = res["control"][C1]["criteria"]
    assert not rec["K3"]["pass"] and not rec["K4"]["pass"]           # the recomputation alone would say K3 1/3
    cr = res["credit_rule"]
    assert cr["control_passes"] == {"K1": 0, "K2": 0, "K3": 2, "K4": 2} and cr["control_as_stated"]
    assert cr["H"] == 2 and cr["b"] and res["decision"]["line"] == 1
    assert res["decision"]["status"] == "provisional"
    assert any("does not reproduce" in w for w in res["decision"]["provisional_why"])


def test_unreadable_control_record_falls_back_to_the_stated_results(tmp_path):
    res = score(tmp_path, arm_cells("LTSB", BAND, {R3: ["K3"]}), score_final=tmp_path / "no_such_score_final.json")
    cr = res["credit_rule"]
    assert cr["control_passes"] == {"K1": 0, "K2": 0, "K3": 2, "K4": 2}
    assert cr["control_source"].startswith("the criteria's stated results")
    assert res["decision"]["line"] == 1 and res["decision"]["status"] == "provisional"
    assert any("score_final.json is missing" in w for w in res["decision"]["provisional_why"])


def test_control_record_with_another_sha256_is_provisional(tmp_path, monkeypatch):
    monkeypatch.setattr(SS, "STANCE4_SCORE_FINAL_SHA256", "0" * 64)
    res = score(tmp_path, arm_cells("LTSB", BAND))
    assert res["decision"]["status"] == "provisional" and res["decision"]["line"] == 1
    assert any("is not the pinned" in w for w in res["decision"]["provisional_why"])


# ------------------------------------------------------------------------------------------------ drift or split
@pytest.mark.parametrize("names, cls, nc, ns, nu, counted", [
    (["K2common"], "DRIFT", 4, 0, 0, ["8.4M"]),
    (["K2"], "SPLIT", 0, 4, 0, ["8.4M"]),
    (["K2pad"], "DRIFT", 4, 0, 0, ["8.4M"]),          # K2a-only episodes are classed too (both feet above m0)
    (["K2padsplit"], "SPLIT", 0, 4, 0, ["8.4M"]),
    (["K2near"], "OTHER", 0, 0, 4, ["8.4M"]),         # one-foot excursions (l 0.3 mm from m0): not classed
    (["K2c"], "DRIFT", 0, 0, 0, []),                  # K2 failed through K2c alone: DRIFT outright
    (["K2fall"], "OTHER", 0, 0, 0, ["8.4M"]),         # falls are not classed: K2 failed through falls only
])
def test_drift_split_classes(tmp_path, names, cls, nc, ns, nu, counted):
    res = score(tmp_path, arm_cells("LTSB", BAND, {R1: names, R2: ["K3"], R3: ["K3"]}))
    ds = res["band"][R1]["drift_split"]
    assert (ds["class"], ds["n_common"], ds["n_split"], ds["n_near_reference"], ds["counted"]) == (cls, nc, ns, nu, counted)
    assert not ds["K2_pass"]
    assert ds["per_checkpoint"]["8.4M"]["m0"] == STATUE_W
    assert {lab: x["counted"] for lab, x in ds["per_checkpoint"].items()} == {lab: lab in counted for lab in SS.J_LABELS}
    if names == ["K2c"]:
        assert ds["why"] == "K2a and K2b pass at every J checkpoint: K2 failed through K2c alone"


@pytest.mark.parametrize("near, cls", [
    (0.004, "NEAR-REFERENCE"),     # the quiet foot 0.3 mm ahead of m0 (stance4 s60 8.4M seed 3081's shape)
    (-0.071, "NEAR-REFERENCE"),    # 5.0 mm behind (s60 8.4M seed 3052's shape)
    (0.0, "NEAR-REFERENCE"),       # exactly at m0
    (0.21, "NEAR-REFERENCE"),      # 14.7 mm: inside the 15 mm margin
    (-0.21, "NEAR-REFERENCE"),
    (0.22, "COMMON"),              # 15.4 mm: classed, the same side as the far foot
    (-0.22, "SPLIT"),              # 15.4 mm on the other side
])
def test_near_reference_margin(tmp_path, near, cls):
    """criteria: an episode is classed only when both feet lie >= 15/70 (15 mm) from m0; a one-foot excursion is not."""
    at = {R1: {"8.4M": {"eps": eps_with(4, "w", w=(0.7, STATUE_W + near))}}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R2: ["K3"], R3: ["K3"]}, at=at))
    ds = res["band"][R1]["drift_split"]
    got = {c["class"] for c in res["band"][R1]["readouts"]["8.4M"]["classed"]}
    assert got == {cls} and ds["counted"] == ["8.4M"]
    want = {"NEAR-REFERENCE": ("OTHER", 0, 0, 4), "COMMON": ("DRIFT", 4, 0, 0), "SPLIT": ("SPLIT", 0, 4, 0)}[cls]
    assert (ds["class"], ds["n_common"], ds["n_split"], ds["n_near_reference"]) == want
    assert all(abs(c["nearer_foot_mm"] - 70 * abs(near)) < 1e-9 for c in res["band"][R1]["readouts"]["8.4M"]["classed"])
    assert SS.NEAR_REF == 15.0 / 70.0


def test_near_reference_episodes_with_k2c_failing_are_drift(tmp_path):
    """Four one-foot excursions at a failing checkpoint: n_c == n_s == 0, so K2c decides (OTHER above; DRIFT here)."""
    at = {R1: {"8.4M": {"eps": eps_with(4, "w", w=(0.7, 0.104))}, "10.5M": {"eps": eps_with(0, "xc", xc=0.110)}}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R2: ["K3"], R3: ["K3"]}, at=at))
    ds = res["band"][R1]["drift_split"]
    assert (ds["class"], ds["n_common"], ds["n_split"], ds["n_near_reference"], ds["K2c_pass"]) == ("DRIFT", 0, 0, 4, False)
    assert ds["why"] == "n_c == n_s and K2c fails"


@pytest.mark.parametrize("extra", [
    {"8.4M": {"eps": eps_with(1, "w", w=(0.7, -0.3))}},                                     # 0 COMMON / 1 SPLIT under the bar
    {"8.4M": {"eps": eps_with(3, "w", w=(0.7, -0.3))}, "10.5M": {"eps": eps_with(3, "w", w=(0.7, -0.3))}},  # 3 + 3 SPLIT
    {"4.2M": {"eps": eps_with(2, "w", w=(0.7, 0.65))}, "8.4M": {"eps": eps_with(3, "w", w=(0.7, -0.3))}},   # 2 C, 3 S
    {"8.4M": {"eps": eps_with(3, "w", w=(0.7, 0.65))}},                                     # 3 COMMON under the bar
])
def test_k2c_only_failure_with_episodes_under_the_bar_is_drift(tmp_path, extra):
    """K2 fails through K2c alone (x_c +45 mm at 6.3M) while K2a and K2b pass at all six J: DRIFT outright, whatever
    the episodes under the 3-of-40 bar (they caused no K2 failure and are not counted)."""
    at = {R1: {**extra, "6.3M": {"eps": eps_with(0, "xc", xc=0.110)}}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R2: ["K3"], R3: ["K3"]}, at=at))
    k2 = res["band"][R1]["criteria"]["K2"]
    assert k2["K2a"]["pass"] and k2["K2b"]["pass"] and not k2["K2c"]["pass"]
    ds = res["band"][R1]["drift_split"]
    assert (ds["class"], ds["n_common"], ds["n_split"], ds["counted"]) == ("DRIFT", 0, 0, [])
    assert sum(x["n_split"] + x["n_common"] for x in ds["per_checkpoint"].values()) > 0   # present, not counted


def test_two_k2c_only_runs_with_stray_splits_read_the_trigger_line(tmp_path):
    """Two runs failing K2 through K2c alone, each with 3 SPLIT episodes under the bar, and a K4 run: D 2, S 0, line 5
    (the rule as first written read D 0 and line 6 here)."""
    stray = {"8.4M": {"eps": eps_with(3, "w", w=(0.7, -0.3))}, "6.3M": {"eps": eps_with(0, "xc", xc=0.110)}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R3: ["K4"]}, at={R1: stray, R2: stray}))
    cr = res["credit_rule"]
    assert (cr["H"], cr["K2p"], cr["D"], cr["S"], res["decision"]["line"]) == (0, 1, 2, 0, 5)


def test_split_at_the_failing_checkpoint_outweighs_common_episodes_under_the_bar(tmp_path):
    """K2a fails only at 8.4M on 4 SPLIT pad-over episodes, K2c passes, and 3 COMMON episodes under the bar sit at each
    other J checkpoint: SPLIT 0/4 (the rule as first written read DRIFT 15/4)."""
    at = {R1: {lab: {"eps": eps_with(3, "w", w=(0.7, 0.65))} for lab in SS.J_LABELS if lab != "8.4M"}}
    at[R1]["8.4M"] = {"eps": eps_with(4, "pad", w=(0.4, -0.2))}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R2: ["K3"], R3: ["K3"]}, at=at))
    ds = res["band"][R1]["drift_split"]
    assert res["band"][R1]["criteria"]["K2"]["K2c"]["pass"]
    assert (ds["class"], ds["n_common"], ds["n_split"], ds["counted"]) == ("SPLIT", 0, 4, ["8.4M"])


@pytest.mark.parametrize("run, arm, seed, want", [
    ("stance3_armL_s60", "L", 60, ("DRIFT", 0, 0, 0, [])),                     # K2c's range alone (33.9 mm)
    ("stance3_armLT_s60", "LT", 60, ("SPLIT", 0, 30, 0, ["4.2M", "5.2M", "6.3M"])),
    ("stance3_armLT_s52", "LT", 52, ("SPLIT", 0, 6, 4, ["8.4M", "final"])),     # K2c fails on its range too
])
def test_the_stance3_runs_class_as_calibrated(run, arm, seed, want):
    """criteria READOUTS (criteria_calib/drift_split_calib.txt): the real stance3 J files through score_stance3.py itself."""
    notes = []
    r = SS.load_run(SC3, SS.S3_JUDGED, run, arm, seed, run, head=SC3.HEADS[arm], frozen=FROZEN, validation=False,
                    notes=notes, report_sources=[(SC3, SS.S3_JUDGED)], runs_root="/nonexistent", ledger_dir=None)
    assert r["missing"] == [] and r["invalid"] == [] and not r["criteria"]["K2"]["pass"]
    ds = SS.drift_split(r, runs_ended=True)
    assert (ds["class"], ds["n_common"], ds["n_split"], ds["n_near_reference"], ds["counted"]) == want
    if run == "stance3_armL_s60":
        assert ds["why"].endswith("K2c alone") and r["criteria"]["K2"]["K2c"]["range_mm"] == 33.918517


@pytest.mark.parametrize("k2c_fails, cls", [(True, "DRIFT"), (False, "SPLIT")])
def test_equal_counts_go_by_k2c(tmp_path, k2c_fails, cls):
    """n_c == n_s > 0: DRIFT when K2c fails, SPLIT when it passes."""
    eps = [episode(3042 + i, w=(0.7, 0.65) if i < 2 else (0.7, -0.3) if i < 4 else (0.1, 0.1)) for i in range(40)]
    at = {R1: {"8.4M": {"eps": eps}}}
    if k2c_fails:
        at[R1]["10.5M"] = {"eps": eps_with(0, "xc", xc=0.110)}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R2: ["K3"], R3: ["K3"]}, at=at))
    ds = res["band"][R1]["drift_split"]
    assert (ds["n_common"], ds["n_split"], ds["K2c_pass"], ds["class"]) == (2, 2, not k2c_fails, cls)


def test_a_k2_passing_run_is_not_classed(tmp_path):
    """A run passing K2 with 3 COMMON episodes at one checkpoint (under the bar) is not classed, and D counts only the
    K2-failing runs: D = 1, S = 1, line 6, not the trigger line."""
    at = {R3: {"8.4M": {"eps": eps_with(3, "w", w=(0.7, 0.65))}}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R1: ["K2common"], R2: ["K2"], R3: ["K3"]}, at=at))
    assert classes(res) == {R1: "DRIFT", R2: "SPLIT", R3: "not classed (K2 passes)"}
    assert res["band"][R3]["criteria"]["K2"]["pass"]
    cr = res["credit_rule"]
    assert (cr["H"], cr["K2p"], cr["D"], cr["S"], res["decision"]["line"]) == (0, 1, 1, 1, 6)
    assert "not classed (K2 passes)" in SS.render(res)


def test_counts_only_at_the_failing_checkpoints(tmp_path):
    """3 COMMON at 4.2M and 3 at 5.2M (each within K2's 3-episode allowance, so not counted) against 4 SPLIT at 8.4M,
    where K2b fails: SPLIT 0/4."""
    at = {R1: {"4.2M": {"eps": eps_with(3, "w", w=(0.7, 0.65))}, "5.2M": {"eps": eps_with(3, "w", w=(0.7, 0.65))},
               "8.4M": {"eps": eps_with(4, "w", w=(0.7, -0.3))}}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R2: ["K3"], R3: ["K3"]}, at=at))
    ds = res["band"][R1]["drift_split"]
    assert (ds["n_common"], ds["n_split"], ds["class"], ds["counted"]) == (0, 4, "SPLIT", ["8.4M"])
    assert (ds["per_checkpoint"]["4.2M"]["n_common"], ds["per_checkpoint"]["4.2M"]["counted"]) == (3, False)


def test_counts_sum_over_the_failing_checkpoints(tmp_path):
    """5 COMMON at 4.2M and 4 SPLIT at 8.4M, both failing K2b: n_c 5 > n_s 4, DRIFT."""
    at = {R1: {"4.2M": {"eps": eps_with(5, "w", w=(0.7, 0.65))}, "8.4M": {"eps": eps_with(4, "w", w=(0.7, -0.3))}}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R2: ["K3"], R3: ["K3"]}, at=at))
    ds = res["band"][R1]["drift_split"]
    assert (ds["n_common"], ds["n_split"], ds["class"], ds["counted"]) == (5, 4, "DRIFT", ["4.2M", "8.4M"])


def test_missing_j_file_class(tmp_path):
    cells = arm_cells("LTSB", BAND, {R1: ["K2common"], R2: ["K2common"], R3: ["K3"]})
    cells[R2]["final"] = None
    res = score(tmp_path, cells)
    assert res["decision"]["status"] == "incomplete" and classes(res)[R2] == "incomplete"
    ended = score(tmp_path / "ended", cells, runs_ended=True)
    assert classes(ended)[R2] == "OTHER"          # missing after the runs ended: OTHER, so D = 1, line 6
    assert (ended["credit_rule"]["D"], ended["decision"]["line"], ended["decision"]["status"]) == (1, 6, "decided")


def test_invalid_j_file_class(tmp_path):
    cells = arm_cells("LTSB", BAND, {R1: ["K2common"], R2: ["K2common"], R3: ["K3"]})
    cells[R2]["5.2M"]["schema"] = "stance4-judge/v1"
    res = score(tmp_path, cells)
    assert classes(res)[R2] == "OTHER" and res["decision"]["status"] == "provisional" and res["decision"]["line"] == 6
    ended = score(tmp_path / "ended", cells, runs_ended=True)
    assert ended["decision"]["status"] == "decided" and ended["decision"]["line"] == 6


# ------------------------------------------------------------------------------------------------ missing / invalid
def test_missing_band_j_file_is_incomplete(tmp_path):
    cells = arm_cells("LTSB", BAND)
    cells[R2]["final"] = None
    res = score(tmp_path, cells)
    d = res["decision"]
    assert d["status"] == "incomplete" and d["line"] is None and d["text"] is None and len(d["missing"]) == 1
    assert res["band"][R2]["criteria"]["failed"] == ["K1", "K2", "K3", "K4"]
    assert holds(res) == {R1: True, R2: False, R3: True}
    assert "DECISION: incomplete" in SS.render(res)


def test_missing_band_j_file_after_the_runs_ended_is_a_fail(tmp_path):
    """--runs-ended: the missing file fails K1-K4 for its run and the table is read (H = 2, K3 2/3, K4 2/3: line 1)."""
    cells = arm_cells("LTSB", BAND)
    cells[R2]["final"] = None
    res = score(tmp_path, cells, runs_ended=True)
    d = res["decision"]
    assert d["status"] == "decided" and d["line"] == 1 and len(d["counted_as_fails"]) == 1
    assert d["counted_as_fails"][0].endswith(f"{R2}/judge_final.json")
    assert "counted as fails" in SS.render(res)


def test_placed_j_checkpoint_without_its_judge_file_is_provisional(tmp_path):
    """--runs-ended: a band J checkpoint placed under the runs root whose judge file is missing makes the decision
    provisional (criteria: the table is read once every J checkpoint that exists has been judged)."""
    cells = arm_cells("LTSB", BAND)
    cells[R2]["final"] = None
    models = tmp_path / "runs" / R2 / "01_stance" / "models"
    os.makedirs(models)
    (models / "stage1_final.zip").write_bytes(b"z")
    (models / "stage1_final_vecnorm.pkl").write_bytes(b"p")
    res = score(tmp_path, cells, runs_ended=True)
    d = res["decision"]
    assert d["placed_unjudged"] == [f"{R2}/stage1_final.zip"] and d["status"] == "provisional"
    assert any("placed under the runs root but not judged" in w for w in d["provisional_why"])
    assert "placed but not judged" in SS.render(res)
    (models / "stage1_final_vecnorm.pkl").unlink()        # a half-placed pair is not a placed checkpoint
    again = score(tmp_path / "again", cells, runs_ended=True)
    assert again["decision"]["placed_unjudged"] == [] and again["decision"]["status"] == "decided"


def test_missing_report_only_file_is_not_incomplete(tmp_path):
    cells = arm_cells("LTSB", BAND)
    cells[R1]["1.0M"] = judge("LTSB", 52, "1.0M", eps=eps_with(40, "unclean"))  # 0/40 clean, report only
    res = score(tmp_path, cells)
    assert res["band"][R1]["cells"]["1.0M"]["n_clean"] == 0 and res["band"][R2]["cells"]["1.0M"] is None
    assert holds(res) == {R1: True, R2: True, R3: True}
    assert res["decision"]["status"] == "decided" and res["decision"]["line"] == 1


def test_missing_control_j_file_is_provisional_not_incomplete(tmp_path):
    ctrl = arm_cells("LTS4", CONTROL, {C1: ["K1", "K2"], C2: ["K1", "K2"], C3: ["K1", "K2", "K3", "K4"]})
    ctrl[C2]["4.2M"] = None
    res = score(tmp_path, arm_cells("LTSB", BAND), control_cells=ctrl)
    assert res["decision"]["status"] == "provisional" and res["decision"]["line"] == 1
    assert any(w.startswith("control check: control J judge files missing or invalid") and f"{C2}/judge_4194304.json" in w
               for w in res["decision"]["provisional_why"])


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="stance4-judge/v1"),
    lambda d: d.update(arm="LTS"),
    lambda d: d["git"].update(head=SS.HEAD_LTS),
    lambda d: d["arm_env_kwargs"].update(stance_cop_band_weight=0.0),
    lambda d: d["arm_env_kwargs"].pop("stance_cop_band_weight"),
    lambda d: d["noise_matched_control_joint"].update(sigma_vector=FROZEN["criteria"]["H4"]["control_sigma"]["52"]["vector"]),
    lambda d: d["checks"]["relaxed"].update(allow_commit_mismatch=True),
])
def test_invalid_band_file_fails_and_is_provisional(tmp_path, mutate):
    cells = arm_cells("LTSB", BAND)
    d = copy.deepcopy(cells[R3]["6.3M"])  # Run 3 is seed 42: its frozen control sigma is seed 42's
    mutate(d)
    cells[R3]["6.3M"] = d
    res = score(tmp_path, cells)
    assert res["band"][R3]["criteria"]["failed"] == ["K1", "K2", "K3", "K4"] and len(res["band"][R3]["invalid"]) == 1
    assert res["decision"]["status"] == "provisional" and res["decision"]["line"] == 1  # H = 2, K3 2/3 = control
    ended = score(tmp_path / "ended", cells, runs_ended=True)  # the runs have ended: a fail, no longer provisional
    assert ended["decision"]["status"] == "decided" and ended["decision"]["line"] == 1
    assert len(ended["decision"]["counted_as_fails"]) == 1


def test_head_placeholder_invalidates_every_band_file(tmp_path, monkeypatch):
    monkeypatch.setattr(SS, "HEAD_LTSB", None)
    res = score(tmp_path, arm_cells("LTSB", BAND), head=None)
    assert all(len(res["band"][r]["invalid"]) == 6 for r in BAND)
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 6
    assert any("HEAD_LTSB is a PLACEHOLDER" in w for w in d["provisional_why"])
    assert set(classes(res).values()) == {"OTHER"}


@pytest.mark.parametrize("pin, status", [("match", "decided"), ("placeholder", "provisional"), ("mismatch", "provisional")])
def test_criteria_pin(tmp_path, pin, status):
    res = score(tmp_path, arm_cells("LTSB", BAND), pin=pin)
    assert res["decision"]["status"] == status and res["decision"]["line"] == 1


def test_head_override_is_provisional(tmp_path):
    other = "c" * 40
    cells = arm_cells("LTSB", BAND)
    for slots in cells.values():
        for d in slots.values():
            d["git"]["head"] = d["checks"]["stage_config"]["git_commit"] = d["checks"]["stage_config"]["clone_head"] = other
    res = score(tmp_path, cells, head=other)
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 1
    assert any("is not the pinned HEAD_LTSB" in w for w in d["provisional_why"])


def test_validation_is_provisional(tmp_path):
    res = score(tmp_path, arm_cells("LTSB", BAND), validation=True)
    assert res["decision"]["status"] == "provisional"
    assert any("--validation" in w for w in res["decision"]["provisional_why"])


@pytest.mark.parametrize("const", ["TASK_SHA256_LTSB", "JUDGE_SHA256_S5"])
def test_task_and_judge_placeholders_are_provisional(tmp_path, monkeypatch, const):
    monkeypatch.setattr(SS, const, None)
    res = score(tmp_path, arm_cells("LTSB", BAND))
    assert res["decision"]["status"] == "provisional" and res["decision"]["line"] == 1
    assert any(f"{const} is a PLACEHOLDER" in w for w in res["decision"]["provisional_why"])


@pytest.mark.parametrize("key, value, text", [
    ("judge_sha256", "sha256:" + "d" * 64, "not judged by the recorded judge"),
    ("task_sha256", SS.TASK_SHA256_LTS, "is not the criteria's band-on"),     # the band-off (stance4) task
])
def test_wrong_judge_or_task_on_a_j_file_is_provisional(tmp_path, key, value, text):
    cells = arm_cells("LTSB", BAND)
    (cells[R1]["8.4M"] if key == "judge_sha256" else cells[R1]["8.4M"]["panel"])[key] = value
    res = score(tmp_path, cells)
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 1 and holds(res)[R1]
    assert any(text in w for w in d["provisional_why"])


@pytest.mark.parametrize("how", ["null", "missing"])
def test_null_or_missing_task_on_a_band_j_file_is_provisional(tmp_path, how):
    cells = arm_cells("LTSB", BAND)
    panel = cells[R1]["8.4M"]["panel"]
    if how == "null":
        panel["task_sha256"] = None
    else:
        del panel["task_sha256"]
    res = score(tmp_path, cells)
    d = res["decision"]
    assert res["band"][R1]["invalid"] == [] and holds(res)[R1]  # the file stays valid; only the task check catches it
    assert d["status"] == "provisional" and d["line"] == 1
    assert res["band"][R1]["task_sha256_seen"] == [TEST_TASK, None]
    want = (f"{R1}: J task_sha256 [{TEST_TASK!r}, None] is not the criteria's band-on {TEST_TASK} "
            "(J files ['8.4M']; None is a null or missing one)")
    assert [w for w in d["provisional_why"] if "task_sha256" in w] == [want]


@pytest.mark.parametrize("how", ["null", "missing"])
def test_null_or_missing_task_on_a_report_only_band_file_is_a_note(tmp_path, how):
    cells = arm_cells("LTSB", BAND)
    cells[R2]["2.1M"] = judge("LTSB", 60, "2.1M")
    if how == "null":
        cells[R2]["2.1M"]["panel"]["task_sha256"] = None
    else:
        del cells[R2]["2.1M"]["panel"]["task_sha256"]
    res = score(tmp_path, cells)
    assert res["decision"]["status"] == "decided" and res["decision"]["line"] == 1
    assert any(n.startswith(f"{R2}: a report-only file's task_sha256") for n in res["notes"])


@pytest.mark.parametrize("how", ["null", "missing", "other"])
def test_control_j_file_must_carry_the_lts_task(tmp_path, monkeypatch, how):
    """Every valid control J file must carry LTS's task_sha256 (3fa46708); a null, missing or other one fails the
    control check."""
    ctrl = arm_cells("LTS4", CONTROL, {C1: ["K1", "K2"], C2: ["K1", "K2"], C3: ["K1", "K2", "K3", "K4"]})
    record = synthetic_record(tmp_path, ctrl, monkeypatch)
    ok = score(tmp_path / "ok", arm_cells("LTSB", BAND), control_cells=ctrl, score_final=record)
    assert ok["decision"]["status"] == "decided" and ok["control_crosscheck"]["problems"] == []
    panel = ctrl[C3]["5.2M"]["panel"]
    if how == "null":
        panel["task_sha256"] = None
    elif how == "missing":
        del panel["task_sha256"]
    else:
        panel["task_sha256"] = TEST_TASK
    res = score(tmp_path / "bad", arm_cells("LTSB", BAND), control_cells=ctrl, score_final=record)
    x = res["control_crosscheck"]
    assert x["runs"][C1]["task_sha256_match"] and not x["runs"][C3]["task_sha256_match"]
    assert x["runs"][C3]["verdicts_match"] and x["runs"][C3]["criteria_identical"]  # only the task differs
    assert not x["reproduced"] and not x["match"]
    d = res["decision"]
    assert d["status"] == "provisional" and d["line"] == 1
    want = ("control check: the control does not reproduce from its J judge files (verdicts, criteria or task_sha256): "
            f"['{C3}']")
    assert d["provisional_why"] == [want]


# ------------------------------------------------------------------------------------------------ parents
def test_parent_block_is_inputs_only_while_incomplete(tmp_path):
    cells = arm_cells("LTSB", BAND, final={R1: {"joint_falls": 1}, R2: {"own_falls": 1}})
    cells[R3]["final"] = None
    res = score(tmp_path, cells)
    p = res["parents"]
    assert res["decision"]["status"] == "incomplete" and p["inputs_only"] is True
    assert p["comparison"]["top"] is None and p["comparison"]["top_tied"] == []
    out = SS.render(res)
    assert "INPUTS ONLY: the decision is incomplete" in out and "top under the rule" not in out
    ended = score(tmp_path / "ended", cells, runs_ended=True)
    assert ended["parents"]["inputs_only"] is False
    assert ended["parents"]["comparison"]["top"] == "stance3 Run 2 stance3_armLT_s60"
    assert "not chosen here" in SS.render(ended)


def test_parent_ties_are_named(tmp_path):
    """Every band final at 40/40 with 0 noise falls ties stance3 Run 2's final (40/40, 0/0): no strict winner."""
    res = score(tmp_path, arm_cells("LTSB", BAND))
    p = res["parents"]
    assert p["best_band_final"]["best"] is None and len(p["best_band_final"]["tied"]) == 3
    assert p["best_holding_band_final"]["best"] is None and len(p["best_holding_band_final"]["tied"]) == 3
    assert p["best_stance4_final"]["best"] == "stance4 Run 1 stance4_LTS_s52"
    assert p["comparison"]["top"] is None and len(p["comparison"]["top_tied"]) == 4
    assert "stance3 Run 2 stance3_armLT_s60" in p["comparison"]["top_tied"]
    out = SS.render(res)
    assert "TIE: " in out and "(the maintainer chooses)" in out


def test_parent_holding_rule(tmp_path):
    final = {R1: {}, R2: {"eps": eps_with(1, "unclean")}, R3: {"eps": eps_with(2, "unclean")}}
    res = score(tmp_path, arm_cells("LTSB", BAND, {R1: ["K2"]}, final=final))
    p = res["parents"]
    assert p["best_band_final"]["best"] == "Run 1 stance5_LTSB_s52"            # 40/40, but it does not HOLD
    assert p["best_holding_band_final"]["best"] == "Run 2 stance5_LTSB_s60"    # 39/40 beats 38/40
    assert res["decision"]["line"] == 1


# ------------------------------------------------------------------------------------------------ report only
def test_band_readouts_from_the_judge_files(tmp_path):
    bs = {"episodes": 40, "full_horizon_episodes": 40,
          "full_horizon": {"window_foot_steps": 64000, "guard_0": 2, "guard_ramp": 5, "guard_1": 63993, "counted": 63993,
                           "whole_out": 100, "pad_out": 40, "pad_binds": 3, "share_ramp_025_035": 5, "share_below_025": 2,
                           "window_steps_paying": 900, "guard_below_1_before_window": 7, "counted_equals_guard_ge_1": True,
                           "share_min": 0.2},
          "paid_per_full_horizon_episode": {"mean": -3.5, "median": -1.0, "worst": -20.0, "least": 0.0,
                                            "episodes_paying_over_5": 4}}
    eps = [episode(3042 + i, w=(0.5, 0.1) if i < 3 else (0.1, 0.1), share=i == 5) for i in range(40)]
    eps[7]["reasons"] = eps[7]["reasons"] + ["min_foot_load_share: 0.38 < 0.4"]   # the window rail (criteria: both rails)
    eps[7]["clean"] = False
    eps[5]["reasons"] = eps[5]["reasons"] + ["min_foot_load_share: 0.39 < 0.4"]
    at = {R1: {"4.2M": {"eps": eps, "band_summary": bs}}}
    res = score(tmp_path, arm_cells("LTSB", BAND, at=at))
    ro = res["band"][R1]["readouts"]["4.2M"]
    assert (ro["band_outside_fh"], ro["load_share_windowed_fail_fh"]) == (3, 1)
    assert ro["band_outside_fh_seeds"] == [3042, 3043, 3044] and ro["load_share_windowed_fail_fh_seeds"] == [3047]
    assert ro["load_share_window_fail_fh_seeds"] == [3047, 3049] and ro["load_share_either_fail_fh_seeds"] == [3047, 3049]
    assert ro["band_summary"] == bs
    assert holds(res)[R1] and res["decision"]["line"] == 1      # report only
    assert "guard 0/ramp 2/5 of 64000" in SS.render(res)


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


def test_saturation_alarm_from_a_run_ledger(tmp_path):
    res = score(tmp_path, arm_cells("LTSB", BAND), ledgers={R2: _sat_rows([0, 6, 7, 0])})
    a = res["band"][R2]["saturation_alarm"]
    assert a["alarm"] and a["first_alarm_at"] == 3 * 131072 and a["runs"][0]["evaluations"] == 2
    assert res["band"][R1]["saturation_alarm"]["computed"] is False
    assert holds(res) == {R1: True, R2: True, R3: True}   # report only
    assert "name the actuator with tools/satroll5.py" in SS.render(res)


def test_the_control_saturation_alarms_are_stance4s(tmp_path):
    """stance4 s42's alarm first fired at 2,228,224 (criteria); s52 and s60 had none."""
    res = score(tmp_path, arm_cells("LTSB", BAND))
    sa = {run: res["control"][run]["saturation_alarm"] for run in CONTROL}
    assert sa[C3]["alarm"] and sa[C3]["first_alarm_at"] == 2228224 and sa[C3]["flagged"][0] == 2097152
    assert not sa[C1]["alarm"] and not sa[C2]["alarm"]


# ------------------------------------------------------------------------------------------------ validation on stance4
def test_validate_on_stance4_reproduces_stance4():
    res = SS.score(validate_on_stance4=True, runs_ended=True)
    v = res["validation_on_stance4"]
    assert v["all_match"]
    assert {run: x["failed"] for run, x in v["runs"].items()} == {C1: ["K1", "K2"], C2: ["K1", "K2"],
                                                                  C3: ["K1", "K2", "K3", "K4"]}
    cr, d = res["credit_rule"], res["decision"]
    assert (cr["H"], cr["K2p"], cr["D"], cr["S"]) == (0, 0, 0, 3) and d["line"] == 6 and d["status"] == "provisional"
    assert {run: x["drift_split"] for run, x in v["runs"].items()} == {
        C1: ["SPLIT", 0, 101, 0], C2: ["SPLIT", 0, 67, 2], C3: ["SPLIT", 0, 27, 1]}
    assert any("--validate-on-stance4" in w for w in d["provisional_why"])


# ------------------------------------------------------------------------------------------------ command line
def test_command_line(tmp_path):
    out = tmp_path / "score.json"
    cmd = [sys.executable, "-I", "-B", os.path.join(HERE, "score_stance5.py"), "--judged", str(tmp_path / "empty"),
           "--runs-root", str(tmp_path / "runs"), "--json", str(out)]
    # the file's own pins: the decision is provisional only while a pin is empty or the criteria file is not its pin
    empty = sorted(k for k, v in FILE_PINS.items() if v is None)
    pinned = not empty and SS.sha256_file(SS.CRITERIA_FILE) == FILE_PINS["CRITERIA_SHA256"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "DECISION: incomplete (18 band J judge files missing" in p.stdout
    assert "control verdicts reproduce stance4 score_final.json: yes" in p.stdout
    assert "INPUTS ONLY" in p.stdout and "top under the rule" not in p.stdout
    assert ("   provisional: " in p.stdout) is not pinned
    for k in empty:
        assert f"{k} is a PLACEHOLDER" in p.stdout
    d = json.loads(out.read_text())
    assert d["schema"] == "stance5-score/v1" and d["decision"]["status"] == "incomplete"
    assert (d["decision"]["provisional_why"] == []) is pinned
    p2 = subprocess.run(cmd, capture_output=True, text=True)
    assert p2.returncode != 0 and "refusing to overwrite" in p2.stderr
    p3 = subprocess.run(cmd[:-2] + ["--runs-ended"], capture_output=True, text=True)  # every J file counted as a fail
    assert p3.returncode == 0, p3.stderr
    assert "18 band J judge files missing or invalid" in p3.stdout
    if pinned:
        assert "DECISION: 6. H == 0" in p3.stdout and "(PROVISIONAL)" not in p3.stdout
    else:
        assert "DECISION (PROVISIONAL): 6. H == 0" in p3.stdout
    p4 = subprocess.run(cmd[:4] + ["--validate-on-stance4"], capture_output=True, text=True)
    assert p4.returncode == 0, p4.stderr
    assert "VALIDATION ON STANCE4: stance4's K results and the calibrated drift-or-split classes reproduced: yes" in p4.stdout
    assert "DECISION (PROVISIONAL): 6. H == 0" in p4.stdout
