"""Score the T. rex stance4 screen (three L+T + stagger runs) against its fixed criteria, beside the stance3 L+T control.

    python3 score_stance4.py [--judged S4/judged] [--control-judged S3/judged] [--runs-root S4/runs]
                             [--control-runs-root S3/runs] [--head-LTS SHA] [--runs-ended] [--ledgers RUN_ID=DIR ...]
                             [--json OUT.json]

The criteria: foot/stance4/criteria.txt (CRITERIA_SHA256 below, pinned).

K1-K4 per run are score_stance3.py's own code, not a reimplementation: the file (sha256 pinned, SC3_SHA256) is loaded
unchanged as two module instances.
  * SC3, as it is: the control arm's J files, judge_stance3.py's (schema stance3-judge/v1, arm LT, HEAD 1af2d4d7).
  * SC4, the same code with only the judge-identity constants extended (PATCHED): JUDGE_SCHEMA stance4-judge/v1,
    ARM_ENV and HEADS with arm LTS added (L+T's [env] plus stance_stagger_weight 0.30 and stance_stagger_tolerance_m
    0.045; HEAD the config commit, HEAD_LTS).  Every bar, slot and rule is SC3's.
Each run goes through load_cell / validity_issues / panel_counts / run_criteria / k1_failure_modes of its instance.

  stagger arm  Run 1 stance4_LTS_s52, Run 2 stance4_LTS_s60, Run 3 stance4_LTS_s42: JUDGED/<RUN_ID>/judge_<slot>.json
  control arm  stance3 Run 2 stance3_armLT_s60, stance3 Run 3 stance3_armLT_s52: the K1-K4 passes RECORDED in stance3's
               score_final.json (sha256 pinned) are the control's passes in the credit rule (criteria: "the control's
               passes are those recorded in score_final.json").  Their J files, CONTROL_JUDGED/<RUN_ID>/judge_<slot>.json
               as judged for stance3, are re-scored only as a cross-check.

A missing or invalid judge file fails every criterion that needs it (stance3's rule).  Until the runs have ended
(--runs-ended), a missing stagger J file makes the decision "incomplete" (no line).  With --runs-ended, missing and
invalid stagger J files are counted as fails and listed, and the line is given.  The decision is PROVISIONAL (each
reason printed) while: a placeholder is not filled (HEAD_LTS, CRITERIA_SHA256, TASK_SHA256_LTS, JUDGE_SHA256_S4) or the
criteria file does not match its pin; --validation or a HEAD other than HEAD_LTS is used; a valid stagger J file's
task_sha256 is not TASK_SHA256_LTS (a null or missing one included) or its judge_sha256 is not JUDGE_SHA256_S4; before
--runs-ended, a stagger J file is invalid; or the control check fails (score_final.json missing or not its pinned
sha256, its passes not the stated ones, or the control's verdicts not reproduced from its J files, or one of its valid J
files not carrying L+T's task_sha256, a null or missing one included).

Then the credit rule, (a) H >= 2 and (b) in its general form (for each of K1-K4, the fraction of stagger runs passing
>= the fraction of control runs passing, from the control's recorded passes), beside its specific form (no stagger run
fails K3 or K4, equivalent under (a) to the general form only while the control stands at K1 1/2, K2 0/2, K3 2/2,
K4 2/2), H, K2p, the decision-table line and the parent comparison inputs (stance3's best-final rule, best_final, on
the stagger finals, on the holding ones, and on the stagger finals plus stance3 Run 2's final).  A tie under that rule
is printed as a TIE with every tied final named: the maintainer chooses (criteria, "Ties").  While the decision is
incomplete, the parent block is printed as inputs only and offers no parent (the criteria read it once all three runs
have ended).

Report only (never in a verdict): the 1,048,576-step checkpoints not in J (JUDGED first, then for the control
CONTROL_JUDGED), robust_best / best / the late-stable S3 checkpoint, K5, late falls and the saturation alarm (an
evaluation of eval_episodes.csv with >= 5 episodes listing max_actuator_saturation_fraction in screen_v2_reasons; two
such evaluations in a row is an alarm).  Plain python (no repository import).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S3DIR = SCRATCH + "/foot/stance3"
S4DIR = SCRATCH + "/foot/stance4"
SC3_PATH = S3DIR + "/judge/score_stance3.py"
SC3_SHA256 = "2c3879d3a83d453159773d6fed7cacdfbe88e5337c83605834b14377588cc66a"
CRITERIA_FILE = S4DIR + "/criteria.txt"
CRITERIA_SHA256 = "978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38"  # foot/stance4/criteria.txt, fixed
STANCE3_SCORE_FINAL = S3DIR + "/scores/score_final.json"
STANCE3_SCORE_FINAL_SHA256 = "559b7ef12cb924267e0fc4f293136f69c72ab4006f6d7ff22e2a821c297df2cc"
S4_JUDGED, S3_JUDGED = S4DIR + "/judged", S3DIR + "/judged"
S4_RUNS, S3_RUNS = S4DIR + "/runs", S3DIR + "/runs"
JUDGE_SCHEMA_S4 = "stance4-judge/v1"
#: claude/screen-trex-r9-armlt-stagger's config commit (judge_stance4.py ARMS["LTS"]["head"]), filled before training.
HEAD_LTS = "c1e146a82658112a9a1c0838250020b214ef303a"
#: criteria.txt: the term-on stance task_sha256 (measured on the config tree, S4/build/A1.json).
TASK_SHA256_LTS = "sha256:3fa467083e34fa280ccb6d1bd04eebccdd3aac2b820fd89901b8d418ef32f2db"
TASK_SHA256_LT = "sha256:d4a3b8f83586b38852f80295387b0cc19f97b17188b3e4991da9e3e81e42faef"
#: judge_stance4.py's "judge_sha256" ("sha256:<hex>", as the judge records it), with its HEAD filled; the same sha256 is
#: recorded in colab_steps.txt before the first training step.
JUDGE_SHA256_S4 = "sha256:3c8c07592cd76db84531f187a6d263e33765eea959d4dc8adb3607ffdb8f427d"
_PINNED = object()  # score(head_lts=...) default: HEAD_LTS as it stands when score() runs
ARM_ENV_LTS = {"leg_home_pose_weight": 1.0, "leg_home_pose_tolerance": 0.05,
               "stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045}
STAGGER_RUNS = (("stance4_LTS_s52", "LTS", 52, "Run 1"), ("stance4_LTS_s60", "LTS", 60, "Run 2"),
                ("stance4_LTS_s42", "LTS", 42, "Run 3"))
CONTROL_RUNS = (("stance3_armLT_s60", "LT", 60, "stance3 Run 2"), ("stance3_armLT_s52", "LT", 52, "stance3 Run 3"))
INCUMBENT = CONTROL_RUNS[0]  # stance3 Run 2's final: the current walking pilot 2 parent
KS = ("K1", "K2", "K3", "K4")
#: The control's results as the criteria state them (stance3 score_final.txt); the specific form of (b) rests on them.
CONTROL_STATED = {"stance3_armLT_s60": {"K1": False, "K2": False, "K3": True, "K4": True},
                  "stance3_armLT_s52": {"K1": True, "K2": False, "K3": True, "K4": True}}
SAT_KEY, SAT_MIN, SAT_IN_A_ROW = "max_actuator_saturation_fraction", 5, 2
PATCHED = ("JUDGE_SCHEMA", "ARM_ENV", "HEADS")
DECISION_LINES = {
    1: "1. H >= 2 and (b) holds: ADOPT. Propose committing the stagger option (off by default) plus the two config keys "
       "to the maintainer as the T. rex stance recipe; the best holding stagger final is offered as the stance parent.",
    2: "2. H >= 2 but (b) fails (a stagger run fails K3 or K4): not adopted. The term holds fore-aft but may cost "
       "robustness; replicate on 2 more seeds before any commit.",
    3: "3. H == 1: not adopted; not separated from seed luck; replicate on 2 more seeds before any commit.",
    4: "4. H == 0 and K2p >= 2: not adopted; the term holds fore-aft from scratch but the runs fail elsewhere (K1, K3 or "
       "K4); the next round targets those failures (for K3, the leg exploration-noise floor of stance3 line 2).",
    5: "5. H == 0 and K2p <= 1: not adopted; the stagger penalty does not hold fore-aft from scratch; the next round tests "
       "the per-foot CoP band term (new contact-reward code) or the v6b ankle spring (stance3 line 3b).",
}


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_stance3(name: str):
    """A fresh module instance of the pinned score_stance3.py (refuses any other file content)."""
    got = sha256_file(SC3_PATH)
    if got != SC3_SHA256:
        raise SystemExit(f"{SC3_PATH} sha256 {got} != the pinned {SC3_SHA256}")
    spec = importlib.util.spec_from_file_location(name, SC3_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC3 = load_stance3("score_stance3")              # the control arm, unchanged
SC4 = load_stance3("score_stance3_for_stance4")  # the stagger arm: the same code, the judge identity extended
SC4.JUDGE_SCHEMA = JUDGE_SCHEMA_S4
SC4.ARM_ENV = {**SC3.ARM_ENV, "LTS": dict(ARM_ENV_LTS)}
SC4.HEADS = {**SC3.HEADS, "LTS": HEAD_LTS}
J = SC3.J
J_STEPS = {s for _, s in J}
REPORT_1M = tuple((f"{k * 1048576 / 1e6:.1f}M", k * 1048576) for k in range(1, 12) if k * 1048576 not in J_STEPS)
SELECTION = (("robust_best", "robust_best"), ("best", "best"))


# ------------------------------------------------------------------------------------------------ report only
def saturation_alarm(rows: list[dict]) -> dict:
    """Per evaluation of eval_episodes.csv: episodes listing max_actuator_saturation_fraction in screen_v2_reasons;
    an evaluation with >= 5 is flagged, and two flagged evaluations in a row are an alarm (report only)."""
    if not rows:
        return {"computed": False, "why": "no rows"}
    if "screen_v2_reasons" not in rows[0]:
        return {"computed": False, "why": "no screen_v2_reasons column"}
    by: dict = {}
    for r in rows:
        ts = int(float(r["timesteps"]))
        keys = {SC3.reason_key(x) for x in (r.get("screen_v2_reasons") or "").split(";") if x.strip()}
        e = by.setdefault(ts, [0, 0])
        e[0] += int(SAT_KEY in keys)
        e[1] += 1
    evals = [{"timesteps": ts, "n": n, "saturated": k, "flag": k >= SAT_MIN} for ts, (k, n) in sorted(by.items())]
    runs, cur = [], []
    for e in evals + [None]:
        if e is not None and e["flag"]:
            cur.append(e)
            continue
        if len(cur) >= SAT_IN_A_ROW:
            runs.append({"from": cur[0]["timesteps"], "to": cur[-1]["timesteps"], "evaluations": len(cur),
                         "alarm_at": cur[SAT_IN_A_ROW - 1]["timesteps"]})
        cur = []
    return {"computed": True, "evaluations": len(evals), "flagged": [e["timesteps"] for e in evals if e["flag"]],
            "alarm": bool(runs), "runs": runs, "first_alarm_at": runs[0]["alarm_at"] if runs else None}


# ------------------------------------------------------------------------------------------------ one run
def load_run(mod, root: str, run: str, arm: str, seed: int, name: str, *, head, frozen: dict, validation: bool,
             notes: list, report_sources: list, runs_root: str, ledger_dir: str | None) -> dict:
    """One run's cells (J through *mod*; report-only slots from the first of *report_sources* holding the file), its
    K1-K4 (mod.run_criteria) and its report-only readouts."""
    cells, missing, invalid, report_invalid = {}, [], [], []
    for lab, steps in J:
        c, why, entry = mod.load_cell(root, run, arm, seed, steps, lab, expect_head=head, frozen=frozen,
                                      validation=validation, needs_noise=lab in mod.NOISE_SLOTS, notes=notes)
        cells[lab] = c
        if why == "missing":
            missing.append(entry["path"])
        elif why == "invalid":
            invalid.append(entry)

    def report_cell(lab, steps):
        for m, rt in report_sources:
            if not os.path.exists(os.path.join(rt, run, f"judge_{steps}.json")):
                continue
            c, why, entry = m.load_cell(rt, run, arm, seed, steps, lab, expect_head=head, frozen=frozen,
                                        validation=validation, needs_noise=False, notes=notes)
            if why == "invalid":
                report_invalid.append(entry)
            return c
        return None

    for lab, steps in REPORT_1M + SELECTION:
        cells[lab] = report_cell(lab, steps)
    ev_path = SC3._find(runs_root, run, "eval_episodes.csv", ledger_dir)
    tr_path = SC3._find(runs_root, run, "train_episodes.csv", ledger_dir)
    ev_rows = SC3.read_csv(ev_path) if ev_path else []
    evals = SC3.evaluations(ev_rows) if ev_path else []
    statue_site = next((cells[lab]["statue_com_footsite_mean_m"] for lab, _ in J + REPORT_1M
                        if cells.get(lab) is not None and cells[lab]["statue_com_footsite_mean_m"] is not None), None)
    s3 = SC3.s3_step(evals) if ev_path else {"step": None, "why": "no eval_episodes.csv"}
    if s3.get("step") is not None:
        cells["S3"] = report_cell("S3", s3["step"])
    if ledger_dir:
        notes.append(f"{run}: ledgers read from {ledger_dir} (--ledgers)")
    crits = mod.run_criteria(cells)
    return {"arm": arm, "seed": seed, "name": name, "head": head, "cells": cells, "missing": missing, "invalid": invalid,
            "report_invalid": report_invalid, "criteria": crits, "k1_failure_modes": mod.k1_failure_modes(cells),
            "ledgers": {"eval_episodes": ev_path, "train_episodes": tr_path},
            "K5": SC3.k5_readout(evals, statue_site) if ev_path else {"computed": False, "why": "no eval_episodes.csv"},
            "late_falls": SC3.late_falls(SC3.read_csv(tr_path)) if tr_path else None,
            "saturation_alarm": saturation_alarm(ev_rows) if ev_path else {"computed": False, "why": "no eval_episodes.csv"},
            "S3": s3}


# ------------------------------------------------------------------------------------------------ arm level
def verdicts(crits: dict) -> dict:
    return {**{k: bool(crits[k]["pass"]) for k in KS}, "HOLDS": bool(crits["HOLDS"]),
            "FORE_AFT_HOLDS": bool(crits["FORE_AFT_HOLDS"]), "failed": list(crits["failed"])}


def control_crosscheck(control: dict, score_final: str = STANCE3_SCORE_FINAL) -> dict:
    """The control's passes as RECORDED in stance3's score_final.json (the credit rule's control input), checked against
    its pinned sha256, against the results the criteria state, and against the control's recomputed criteria (verdicts,
    the whole criteria block and each J cell's task_sha256).  *problems* lists every check that fails; each one makes the
    decision provisional.  When the file is missing or unreadable, the criteria's stated results stand in."""
    out = {"score_final": score_final, "sha256": None, "sha256_matches": None, "runs": {}, "match": False,
           "stated_match": False, "readable": False, "recorded": None, "reproduced": False, "problems": []}
    stored = None
    if not os.path.exists(score_final):
        out["why"] = "missing"
    else:
        out["sha256"] = sha256_file(score_final)
        out["sha256_matches"] = out["sha256"] == STANCE3_SCORE_FINAL_SHA256
        try:
            with open(score_final) as fh:
                stored = json.load(fh)
        except (OSError, ValueError) as exc:
            out["why"] = f"unreadable: {exc}"
    recorded, reproduced = {}, True
    for run, *_ in CONTROL_RUNS:
        c = control[run]["criteria"]
        s = ((((stored or {}).get("runs") or {}).get(run) or {}).get("criteria")) if isinstance(stored, dict) else None
        v = verdicts(c)
        try:
            sv = verdicts(s) if s else None
        except (KeyError, TypeError):
            sv = None
        if sv is not None:
            recorded[run] = {k: sv[k] for k in KS}
        same_v = sv == v
        same_all = s is not None and json.loads(json.dumps(c)) == s
        # every valid control J cell must carry L+T's task_sha256; a null or missing one is a mismatch too
        j_cells = [x for lab, _ in J for x in [control[run]["cells"].get(lab)] if x is not None]
        tasks = sorted({x.get("task_sha256") for x in j_cells}, key=lambda t: (t is None, t or ""))
        task_ok = bool(j_cells) and all(x.get("task_sha256") == TASK_SHA256_LT for x in j_cells)
        out["runs"][run] = {"verdicts": v, "stance3_verdicts": sv, "verdicts_match": same_v, "criteria_identical": same_all,
                            "stated_match": sv is not None and recorded[run] == CONTROL_STATED[run],
                            "task_sha256_seen": tasks, "task_sha256_match": task_ok}
        reproduced = reproduced and same_v and same_all and task_ok
    out["readable"] = len(recorded) == len(CONTROL_RUNS)
    out["recorded"] = recorded if out["readable"] else None
    out["stated_match"] = out["readable"] and recorded == CONTROL_STATED
    out["reproduced"] = reproduced
    out["match"] = out["readable"] and reproduced and bool(out["sha256_matches"])
    p = out["problems"]
    if not out["readable"]:
        p.append(f"stance3's score_final.json is {out.get('why') or 'without both control runs'}: the control passes are "
                 "the criteria's stated ones")
    if out["sha256"] is not None and not out["sha256_matches"]:
        p.append(f"stance3's score_final.json sha256 {out['sha256']} is not the pinned {STANCE3_SCORE_FINAL_SHA256}")
    if out["readable"] and not out["stated_match"]:
        p.append("the control passes recorded in score_final.json are not the ones the criteria state "
                 "(K1 1/2, K2 0/2, K3 2/2, K4 2/2)")
    gone = [x for run, *_ in CONTROL_RUNS for x in control[run].get("missing", [])
            + [e["path"] for e in control[run].get("invalid", [])]]
    if gone:
        p.append(f"control J judge files missing or invalid: {gone}")
    if not reproduced:
        bad = [run for run, r in out["runs"].items()
               if not (r["verdicts_match"] and r["criteria_identical"] and r["task_sha256_match"])]
        p.append(f"the control does not reproduce from its J judge files (verdicts, criteria or task_sha256): {bad}")
    return out


def control_passes(xcheck: dict) -> tuple[list[dict], str]:
    """The control's K1-K4 passes for the credit rule: as recorded in score_final.json, else the criteria's stated ones."""
    if xcheck["readable"]:
        return [xcheck["recorded"][r[0]] for r in CONTROL_RUNS], "recorded in stance3's score_final.json"
    return [dict(CONTROL_STATED[r[0]]) for r in CONTROL_RUNS], "the criteria's stated results (score_final.json unreadable)"


def credit_rule(stagger: list[dict], control: list[dict]) -> dict:
    """(a), (b) in its general form (exact fractions, from the control's recorded passes) and in its specific form, H
    and K2p.  *stagger* are run_criteria outputs; *control* are {K1..K4: bool} (control_passes)."""
    ns, nc = len(stagger), len(control)
    ps = {k: sum(bool(c[k]["pass"]) for c in stagger) for k in KS}
    pc = {k: sum(bool(c[k]) for c in control) for k in KS}
    by_test = {k: {"stagger": f"{ps[k]}/{ns}", "control": f"{pc[k]}/{nc}", "no_worse": ps[k] * nc >= pc[k] * ns}
               for k in KS}
    H = sum(bool(c["HOLDS"]) for c in stagger)
    a = H >= 2
    b = all(t["no_worse"] for t in by_test.values())
    specific = not any(not c["K3"]["pass"] or not c["K4"]["pass"] for c in stagger)
    control_as_stated = nc == 2 and pc == {"K1": 1, "K2": 0, "K3": 2, "K4": 2}
    return {"H": H, "K2p": ps["K2"], "a": a, "b": b, "b_by_test": by_test, "b_specific": specific,
            "control_as_stated": control_as_stated,
            "forms_agree": (not a) or (not control_as_stated) or (specific == b),
            "stagger_passes": ps, "control_passes": pc, "n_stagger": ns, "n_control": nc}


def decision_line(H: int, K2p: int, b: bool) -> int:
    """The decision table (exclusive and exhaustive)."""
    if H >= 2:
        return 1 if b else 2
    if H == 1:
        return 3
    return 4 if K2p >= 2 else 5


def best_final_among(mod, runs, per_run: dict) -> dict:
    """stance3's best_final (most clean at final; tie -> fewer joint + own noise falls) over *runs*."""
    saved = mod.RUNS
    mod.RUNS = tuple(runs)
    try:
        return mod.best_final(per_run)
    finally:
        mod.RUNS = saved


def parents(stagger: dict, control: dict, *, inputs_only: bool = False) -> dict:
    """The parent comparison inputs.  *inputs_only* (the decision is incomplete): the criteria read the comparison once
    all three runs have ended, so no walking pilot 2 parent is offered (offered, stays_incumbent and tie_with_incumbent
    are None); the candidates are still listed."""
    holding = [r for r in STAGGER_RUNS if stagger[r[0]]["criteria"]["HOLDS"]]
    both = {**stagger, INCUMBENT[0]: control[INCUMBENT[0]]}
    inc = control[INCUMBENT[0]]["cells"].get("final")
    wp2 = best_final_among(SC4, STAGGER_RUNS + (INCUMBENT,), both)
    inc_label = f"{INCUMBENT[3]} {INCUMBENT[0]}"
    offered = None if inputs_only else wp2["best"]
    return {"inputs_only": inputs_only,
            "best_stagger_final": best_final_among(SC4, STAGGER_RUNS, stagger),
            "best_holding_stagger_final": best_final_among(SC4, holding, stagger) if holding else
            {"best": None, "tied": [], "candidates": []},
            "incumbent": {"run": inc_label, "clean": None if inc is None else inc["n_clean"],
                          "joint_noise_falls": None if inc is None else inc["noise_joint_falls"],
                          "own_noise_falls": None if inc is None else inc["noise_own_falls"]},
            "walking_pilot_2": {**wp2, "offered": offered,
                                "stays_incumbent": None if inputs_only else offered == inc_label,
                                "tie_with_incumbent": None if inputs_only else
                                offered is None and inc_label in wp2.get("tied", [])}}


# ------------------------------------------------------------------------------------------------ scoring
def score(judged: str = S4_JUDGED, control_judged: str = S3_JUDGED, *, runs_root: str = S4_RUNS,
          control_runs_root: str = S3_RUNS, head_lts=_PINNED, frozen_path: str = SC3.FROZEN_PARAMS,
          validation: bool = False, ledger_dirs: dict | None = None, criteria_file: str = CRITERIA_FILE,
          criteria_sha256=_PINNED, score_final: str = STANCE3_SCORE_FINAL, runs_ended: bool = False) -> dict:
    if head_lts is _PINNED:
        head_lts = HEAD_LTS
    if criteria_sha256 is _PINNED:
        criteria_sha256 = CRITERIA_SHA256
    fsha = sha256_file(frozen_path)
    if fsha != SC3.FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{frozen_path} sha256 {fsha} != {SC3.FROZEN_PARAMS_SHA256}")
    with open(frozen_path) as fh:
        frozen = json.load(fh)
    ledger_dirs = {k: os.path.abspath(v) for k, v in (ledger_dirs or {}).items()}
    known = {r[0] for r in STAGGER_RUNS + CONTROL_RUNS}
    if set(ledger_dirs) - known:
        raise SystemExit(f"--ledgers names no screen run: {sorted(set(ledger_dirs) - known)}")
    notes: list[str] = []
    crit = {"path": criteria_file, "sha256": sha256_file(criteria_file) if os.path.exists(criteria_file) else None,
            "pinned": criteria_sha256}
    crit["matches"] = criteria_sha256 is not None and crit["sha256"] == criteria_sha256
    provisional_why = []
    if criteria_sha256 is None:
        provisional_why.append("criteria sha256 is a PLACEHOLDER (not pinned)")
    elif not crit["matches"]:
        provisional_why.append(f"criteria file {criteria_file} sha256 {crit['sha256']} != the pinned {criteria_sha256}")
    if HEAD_LTS is None:
        provisional_why.append("HEAD_LTS is a PLACEHOLDER (not pinned)")
    if head_lts != HEAD_LTS:
        provisional_why.append(f"HEAD {head_lts} is not the pinned HEAD_LTS {HEAD_LTS}")
    if head_lts is None:
        provisional_why.append("no stagger HEAD: every stagger file is invalid until it is filled")
    if TASK_SHA256_LTS is None:
        provisional_why.append("TASK_SHA256_LTS is a PLACEHOLDER (not pinned)")
    if JUDGE_SHA256_S4 is None:
        provisional_why.append("JUDGE_SHA256_S4 is a PLACEHOLDER (not pinned)")
    if validation:
        provisional_why.append("--validation: relaxed-binding judge files accepted; not a screen verdict")
    SC4.HEADS = {**SC3.HEADS, "LTS": head_lts}
    stagger = {run: load_run(SC4, judged, run, arm, seed, name, head=head_lts, frozen=frozen, validation=validation,
                             notes=notes, report_sources=[(SC4, judged)], runs_root=runs_root,
                             ledger_dir=ledger_dirs.get(run))
               for run, arm, seed, name in STAGGER_RUNS}
    control = {run: load_run(SC3, control_judged, run, arm, seed, name, head=SC3.HEADS[arm], frozen=frozen,
                             validation=validation, notes=notes, report_sources=[(SC4, judged), (SC3, control_judged)],
                             runs_root=control_runs_root, ledger_dir=ledger_dirs.get(run))
               for run, arm, seed, name in CONTROL_RUNS}
    j_labels = {lab for lab, _ in J}

    def _tasks(cells):  # every task_sha256 the cells carry, a null or missing one as None (sorted last)
        return sorted({c.get("task_sha256") for c in cells}, key=lambda t: (t is None, t or ""))

    for run, r in stagger.items():
        valid = {lab: c for lab, c in r["cells"].items() if c is not None}
        tasks = _tasks(valid.values())
        j_tasks = _tasks(c for lab, c in valid.items() if lab in j_labels)
        r["task_sha256_seen"] = tasks
        # criteria: a stagger J file's task_sha256 other than the term-on one, or a judge_sha256 other than the
        # recorded judge's, makes the decision provisional (report-only files: a note).
        judges = {}
        for lab, c in r["cells"].items():
            if c is None:
                continue
            try:
                with open(c["path"]) as fh:
                    judges[lab] = json.load(fh).get("judge_sha256")
            except (OSError, ValueError):
                judges[lab] = None
        r["judge_sha256_seen"] = judges
        if TASK_SHA256_LTS is not None:
            off_t = sorted(lab for lab, c in valid.items() if lab in j_labels and c.get("task_sha256") != TASK_SHA256_LTS)
            if off_t:
                provisional_why.append(f"{run}: J task_sha256 {j_tasks} is not the criteria's term-on {TASK_SHA256_LTS} "
                                       f"(J files {off_t}; None is a null or missing one)")
            elif any(t != TASK_SHA256_LTS for t in tasks):
                notes.append(f"{run}: a report-only file's task_sha256 {tasks} is not the term-on {TASK_SHA256_LTS}")
        if JUDGE_SHA256_S4 is not None:
            off = sorted(lab for lab, s in judges.items() if lab in j_labels and s != JUDGE_SHA256_S4)
            if off:
                provisional_why.append(f"{run}: J files {off} were not judged by the recorded judge {JUDGE_SHA256_S4}")
            off_r = sorted(lab for lab, s in judges.items() if lab not in j_labels and s != JUDGE_SHA256_S4)
            if off_r:
                notes.append(f"{run}: report-only files {off_r} were not judged by the recorded judge")
    xcheck = control_crosscheck(control, score_final)
    provisional_why += [f"control check: {p}" for p in xcheck["problems"]]
    cpasses, csource = control_passes(xcheck)
    cr = credit_rule([stagger[r[0]]["criteria"] for r in STAGGER_RUNS], cpasses)
    cr["control_source"] = csource
    if not cr["control_as_stated"]:
        notes.append("the control's passes are not K1 1/2, K2 0/2, K3 2/2, K4 2/2: the specific form of (b) "
                     "does not apply; the general form decides")
    missing = [p for run, *_ in STAGGER_RUNS for p in stagger[run]["missing"]]
    invalid = [x["path"] for r in stagger.values() for x in r["invalid"]]
    if invalid and not runs_ended:
        provisional_why.append(f"invalid stagger J judge files (the runs have not ended): {invalid}")
    if missing and not runs_ended:
        dec = {"status": "incomplete", "line": None, "text": None, "missing": missing, "provisional_why": provisional_why,
               "runs_ended": False, "counted_as_fails": []}
    else:
        n = decision_line(cr["H"], cr["K2p"], cr["b"])
        dec = {"status": "provisional" if provisional_why else "decided", "line": n, "text": DECISION_LINES[n],
               "missing": missing, "provisional_why": provisional_why, "runs_ended": runs_ended,
               "counted_as_fails": (missing + invalid) if runs_ended else []}
    return {"schema": "stance4-score/v1", "criteria": crit,
            "stance3_scorer": {"path": SC3_PATH, "sha256": SC3_SHA256, "patched_in_stagger_instance": list(PATCHED),
                               "stagger_judge_schema": JUDGE_SCHEMA_S4, "arm_env_lts": ARM_ENV_LTS},
            "judged_root": os.path.abspath(judged), "control_judged_root": os.path.abspath(control_judged),
            "runs_root": os.path.abspath(runs_root), "control_runs_root": os.path.abspath(control_runs_root),
            "ledger_dirs": ledger_dirs, "validation": validation, "runs_ended": runs_ended, "head_lts": head_lts,
            "head_lts_pinned": HEAD_LTS, "frozen_params_sha256": fsha,
            "task_sha256": {"LTS_pinned": TASK_SHA256_LTS, "LT": TASK_SHA256_LT}, "judge_sha256_pinned": JUDGE_SHA256_S4,
            "stagger": stagger, "control": control, "control_crosscheck": xcheck, "credit_rule": cr, "decision": dec,
            "parents": parents(stagger, control, inputs_only=dec["status"] == "incomplete"), "notes": notes}


# ------------------------------------------------------------------------------------------------ printing
fmt = SC3.fmt


def run_block(run: str, r: dict) -> list[str]:
    c, cells = r["criteria"], r["cells"]
    L = [f"== {r['name']} {run} (arm {r['arm']}, seed {r['seed']}, clone HEAD {r['head'][:8] if r['head'] else 'PLACEHOLDER'})"
         + ("" if c["complete"] else "   INCOMPLETE (a J judge file is missing or invalid)"),
         "   ckpt    verdict clean  LCB   full hop | K2a K2b  x_c med  statue  offset mm | roll | noise joint/own"]
    not_judged = []
    for lab, steps in list(J) + list(REPORT_1M):
        x = cells.get(lab)
        tag = "" if (lab, steps) in J else "  (report only)"
        if x is None:
            if tag:
                not_judged.append(lab)
            else:
                L.append(f"   {lab:<7} missing/invalid")
            continue
        L.append(f"   {lab:<7} {x['verdict']:>7} {x['n_clean']:>2}/40 {x['lcb']:.3f} {x['full']:>4} {x['hop_or_fall']:>3} |"
                 f" {x['k2a']:>3} {x['k2b']:>3} {fmt(x['xc_median_mm']):>8} {fmt(x['statue_xc_median_mm']):>7} "
                 f"{fmt(x['xc_offset_mm']):>8}  | {x['roll_rail']:>4} | {fmt(x['noise_joint_falls'])}/{fmt(x['noise_own_falls'])}{tag}")
    if not_judged:
        L.append(f"   report-only 1,048,576-step checkpoints not judged (or invalid): {', '.join(not_judged)}")
    k1, k2 = c["K1"], c["K2"]
    L.append(f"   K1 {'PASS' if k1['pass'] else 'FAIL'}: v2 PASS at 6.3M {fmt(k1['pass_at']['6.3M'])}, at final "
             f"{fmt(k1['pass_at']['final'])}; below 37/40 {k1['below_37'] or 'none'} (<= 1 allowed), below 33/40 "
             f"{k1['below_33'] or 'none'}" + (f"; missing {k1['missing']}" if k1["missing"] else ""))
    L.append(f"   K2 {'PASS' if k2['pass'] else 'FAIL'}: K2a {'pass' if k2['K2a']['pass'] else 'FAIL ' + str(k2['K2a']['over'])}; "
             f"K2b {'pass' if k2['K2b']['pass'] else 'FAIL ' + str(k2['K2b']['over'])}; "
             f"K2c window {'pass' if k2['K2c']['window_pass'] else 'FAIL'} [-60, +30] mm, range "
             f"{fmt(k2['K2c']['range_mm'])} mm {'pass' if k2['K2c']['range_pass'] else 'FAIL'} (<= 30)")
    L.append(f"   K3 {'PASS' if c['K3']['pass'] else 'FAIL'}: joint {c['K3']['joint_falls']} (<= 2/20), own "
             f"{c['K3']['own_falls']} (<= 1/20)")
    L.append(f"   K4 {'PASS' if c['K4']['pass'] else 'FAIL'}: roll rail {c['K4']['roll_rail']} (<= 3/40)")
    L.append(f"   -> {'HOLDS' if c['HOLDS'] else 'does not hold (failed ' + ', '.join(c['failed']) + ')'}; "
             f"FORE-AFT {'HOLDS' if c['FORE_AFT_HOLDS'] else 'does not hold'}")
    if r["missing"]:
        L.append("   missing J judge files (fail every criterion that needs them): " + "; ".join(r["missing"]))
    for x in r["invalid"] + r["report_invalid"]:
        L.append(f"   invalid: {x['path']}: {'; '.join(x['issues'])}")
    L.append("   report only:")
    k5 = r["K5"]
    if k5.get("computed"):
        L.append(f"     K5 trajectory: statue {k5['statue_m']:+.4f} m, window [{k5['window_m'][0]:+.4f}, {k5['window_m'][1]:+.4f}] m, "
                 f"{k5['evaluations_considered']} evaluations from 3,145,728: "
                 + ("FLAG " + "; ".join(f"{x['from']:,}-{x['to']:,} ({x['evaluations']})" for x in k5["runs"]) if k5["flag"] else "no flag"))
    else:
        L.append(f"     K5 trajectory: not computed ({k5.get('why')})")
    lf = r["late_falls"]
    if lf and lf["peak"]:
        pk = lf["peak"]
        L.append(f"     late falls (200 < length < 1000) per 262,144 steps: peak "
                 f"{fmt(None if pk['fraction'] is None else 100.0 * pk['fraction'])}% at {pk['bin_end']:,}"
                 + (("; above 5%: " + ", ".join(f"{t['bin_end']:,} {100 * t['fraction']:.1f}%" for t in lf["flagged"]))
                    if lf["flagged"] else ""))
    else:
        L.append("     late falls: no train_episodes.csv rows")
    sa = r["saturation_alarm"]
    if sa.get("computed"):
        L.append(f"     saturation alarm: {'ALARM' if sa['alarm'] else 'none'} ({len(sa['flagged'])} of {sa['evaluations']} "
                 "evaluations with >= 5 saturation episodes"
                 + (f"; first alarm at {sa['first_alarm_at']:,}; runs "
                    + ", ".join(f"{x['from']:,}-{x['to']:,} ({x['evaluations']})" for x in sa["runs"]) if sa["alarm"] else "")
                 + ")")
    else:
        L.append(f"     saturation alarm: not computed ({sa.get('why')})")
    s3 = r["S3"]
    sel = []
    for tag, lab in (("S0 robust_best", "robust_best"), ("S1 best", "best"), ("S2 final", "final"),
                     (f"S3 {s3['step']:,}" if s3.get("step") else "S3 -", "S3")):
        x = cells.get(lab)
        sel.append(f"{tag}: " + ("not judged/invalid" if x is None else
                                 f"clean {x['n_clean']}/40, own-noise falls {fmt(x['noise_own_falls'])}, K2a {x['k2a']}, K2b {x['k2b']}"))
    L.append("     selection: " + " | ".join(sel) + ("" if s3.get("step") else f"  (S3: {s3.get('why')})"))
    return L


def render(res: dict) -> str:
    crit = res["criteria"]
    L = ["T. rex stance4 screen: score against foot/stance4/criteria.txt (sha256 "
         f"{crit['sha256']}; " + ("PLACEHOLDER, not pinned" if crit["pinned"] is None else
                                   "matches the fixed text" if crit["matches"] else "DOES NOT MATCH the pinned " + crit["pinned"]) + ")",
         f"K1-K4 per run: score_stance3.py (sha256 {SC3_SHA256[:16]}...) loaded unchanged; the stagger instance differs only "
         f"in {', '.join(PATCHED)} (schema {JUDGE_SCHEMA_S4}, arm LTS, HEAD {res['head_lts'] or 'PLACEHOLDER'})."]
    if res["validation"]:
        L.append("*** VALIDATION ONLY: relaxed-binding judge files accepted (--validation); not a screen verdict ***")
    L.append(f"stagger judge files: {res['judged_root']}   control judge files: {res['control_judged_root']}")
    L.append("")
    L.append("STAGGER ARM (LTS: L+T + stagger, w 0.30, tolerance 0.045 m)")
    for run, *_ in STAGGER_RUNS:
        L += run_block(run, res["stagger"][run])
    L.append("")
    L.append("CONTROL ARM (stance3 L+T, not retrained; judged and scored under the stance3 criteria)")
    for run, *_ in CONTROL_RUNS:
        L += run_block(run, res["control"][run])
    x = res["control_crosscheck"]
    L.append(f"   control verdicts reproduce stance3 score_final.json: {'yes' if x['match'] else 'NO'} "
             f"(file sha256 {'pinned' if x['sha256_matches'] else 'NOT the pinned one'}); as the criteria state them "
             f"(Run 2 fails K1 K2, Run 3 fails K2): {'yes' if x['stated_match'] else 'NO'}")
    for p in x["problems"]:
        L.append(f"   control check: {p}")
    L.append("")
    cr = res["credit_rule"]
    L.append(f"CREDIT RULE (control passes: {cr['control_source']})")
    L.append("   per test, runs passing (stagger vs control): " + "; ".join(
        f"{k} {t['stagger']} vs {t['control']} {'ok' if t['no_worse'] else 'WORSE'}" for k, t in cr["b_by_test"].items()))
    L.append(f"   H = {cr['H']} (stagger runs that HOLD); K2p = {cr['K2p']} (stagger runs passing K2)")
    L.append(f"   (a) at least 2 of 3 stagger runs HOLD: {'yes' if cr['a'] else 'no'}")
    L.append(f"   (b) general form, from the control's recorded passes: {'holds' if cr['b'] else 'fails'}")
    L.append(f"   (b) specific form, no stagger run fails K3 or K4: {'holds' if cr['b_specific'] else 'fails'} "
             f"(equivalent under (a) while the control stands at K1 1/2, K2 0/2, K3 2/2, K4 2/2: "
             f"{'it does' if cr['control_as_stated'] else 'it does NOT; the general form decides'}"
             + ("" if cr["forms_agree"] else "; THE FORMS DISAGREE") + ")")
    d = res["decision"]
    if d["status"] == "incomplete":
        L.append(f"DECISION: incomplete ({len(d['missing'])} stagger J judge files missing; the counts above treat them as fails). "
                 "The decision is read once all three runs have ended (--runs-ended).")
    else:
        L.append("DECISION" + (" (PROVISIONAL)" if d["status"] == "provisional" else "") + f": {d['text']}")
    if d.get("counted_as_fails"):
        L.append(f"   runs ended (--runs-ended): {len(d['counted_as_fails'])} stagger J judge files missing or invalid, "
                 "counted as fails: " + "; ".join(d["counted_as_fails"]))
    for why in d["provisional_why"]:
        L.append(f"   provisional: {why}")
    p = res["parents"]
    tie = " (the maintainer chooses)"
    L.append("")
    L.append("PARENT COMPARISON INPUTS (stance3's rule: most clean panel episodes at final; ties to fewer joint + own falls;"
             " a tie under the rule names every tied final and the maintainer chooses)")
    if p["inputs_only"]:
        L.append("   INPUTS ONLY: the decision is incomplete; the comparison is read once all three runs have ended, and no "
                 "parent is offered before then.")
    for key, title in (("best_stagger_final", "best stagger final"),
                       ("best_holding_stagger_final", "best holding stagger final (the stance parent of line 1)")):
        b = p[key]
        L.append(f"   {title}: " + (b["best"] or ("TIE: " + ", ".join(b["tied"]) + tie if b.get("tied") else "none")))
    w, inc = p["walking_pilot_2"], p["incumbent"]
    L.append(f"   incumbent {inc['run']} final: clean {fmt(inc['clean'])}/40, noise falls joint {fmt(inc['joint_noise_falls'])}"
             f" / own {fmt(inc['own_noise_falls'])}")
    for c in w["candidates"]:
        L.append(f"     {c['run']:<32} clean {c['clean']:>2}/40  joint + own noise falls {fmt(c['noise_falls'])}")
    if p["inputs_only"]:
        L.append("   walking pilot 2 parent: none offered (inputs only)")
    else:
        L.append("   walking pilot 2 parent offered: " + (w["offered"] if w["offered"] else
                                                           ("TIE: " + ", ".join(w["tied"]) + tie) if w["tied"] else "none")
                 + ("  (the incumbent stays)" if w["stays_incumbent"] else ""))
    hold = p["best_holding_stagger_final"]
    if d.get("line") == 1 and (hold["best"] != w["offered"] or hold.get("tied") or w.get("tied")):
        L.append("   line 1's stance parent and the walking pilot 2 parent do not name one checkpoint: both are named; "
                 "the maintainer chooses")
    L.append("   walking pilot 2 stays parked until the maintainer says otherwise.")
    if res["notes"]:
        L.append("")
        L.append("Notes:")
        for n in res["notes"]:
            L.append(f"   {n}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judged", default=S4_JUDGED, help="directory holding the stagger runs' <RUN_ID>/judge_<slot>.json")
    ap.add_argument("--control-judged", default=S3_JUDGED, help="directory holding the control runs' judge files (stance3)")
    ap.add_argument("--runs-root", default=S4_RUNS, help="the stagger runs' <RUN_ID>/[01_stance/]{eval,train}_episodes.csv")
    ap.add_argument("--control-runs-root", default=S3_RUNS, help="the control runs' ledgers")
    ap.add_argument("--head-LTS", default=HEAD_LTS,
                    help="the stagger branch's config commit (default: HEAD_LTS; any other makes the decision provisional)")
    ap.add_argument("--runs-ended", action="store_true",
                    help="all three stagger runs have ended (criteria): missing or invalid J files count as fails")
    ap.add_argument("--frozen-params", default=SC3.FROZEN_PARAMS)
    ap.add_argument("--validation", action="store_true", help="validation of this script only: relaxed-binding files accepted")
    ap.add_argument("--ledgers", action="append", default=[], metavar="RUN_ID=DIR",
                    help="read RUN_ID's eval/train_episodes.csv from DIR[/01_stance] (a run restarted as <RUN_ID>_b)")
    ap.add_argument("--json", default=None, help="also write the full result here (never overwritten)")
    a = ap.parse_args()
    if a.json and os.path.exists(a.json):
        raise SystemExit(f"refusing to overwrite {a.json}")
    ledger_dirs = {}
    for item in a.ledgers:
        run_id, sep, path = item.partition("=")
        if not sep or not run_id or not path:
            ap.error(f"--ledgers {item!r} is not RUN_ID=DIR")
        ledger_dirs[run_id] = path
    res = score(a.judged, a.control_judged, runs_root=a.runs_root, control_runs_root=a.control_runs_root,
                head_lts=a.head_LTS, frozen_path=a.frozen_params, validation=a.validation, ledger_dirs=ledger_dirs,
                runs_ended=a.runs_ended)
    print(render(res))
    if a.json:
        with open(a.json, "x") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
