"""Score the T. rex stance5 screen (three LTSB runs: L+T + stagger + the per-foot CoP band) against its fixed criteria,
beside the stance4 LTS control.

    python3 score_stance5.py [--judged S5/judged] [--control-judged S4/judged] [--runs-root S5/runs]
                             [--control-runs-root S4/runs] [--head-LTSB SHA] [--runs-ended] [--ledgers RUN_ID=DIR ...]
                             [--json OUT.json]
    python3 score_stance5.py --validate-on-stance4 [--json OUT.json]

The criteria: foot/stance5/criteria.txt (CRITERIA_SHA256 below, pinned).

K1-K4 per run are score_stance3.py's own code, not a reimplementation: the file (sha256 pinned, SC3_SHA256) is loaded
unchanged as three module instances.
  * SC3, as it is: stance3 Run 2's final (stance3_armLT_s60, judge_stance3.py's schema stance3-judge/v1, arm LT, HEAD
    1af2d4d7), read only for the report-only parent comparison.
  * SC4, the same code with only the judge-identity constants extended (PATCHED) exactly as score_stance4.py extends
    them: JUDGE_SCHEMA stance4-judge/v1, ARM_ENV and HEADS with arm LTS (L+T's [env] plus stance_stagger_weight 0.30 and
    stance_stagger_tolerance_m 0.045; HEAD c1e146a8, HEAD_LTS).  It reads the control's J files (score_stance4.py's
    stagger instance).
  * SC5, the same code with the stance5 judge identity (PATCHED): JUDGE_SCHEMA stance5-judge/v1, ARM_ENV and HEADS with
    arm LTSB (arm LTS's [env] plus stance_cop_band_weight 0.30; HEAD the config commit, HEAD_LTSB).
Every bar, slot and rule is SC3's.  Each run goes through load_cell / validity_issues / panel_counts / run_criteria /
k1_failure_modes of its instance.

  band arm     Run 1 stance5_LTSB_s52, Run 2 stance5_LTSB_s60, Run 3 stance5_LTSB_s42: JUDGED/<RUN_ID>/judge_<slot>.json
  control arm  stance4 Run 1 stance4_LTS_s52, Run 2 stance4_LTS_s60, Run 3 stance4_LTS_s42: the K1-K4 passes RECORDED in
               stance4's score_final.json (sha256 pinned; the verdict of record) are the control's passes in the credit
               rule (criteria: "the control's passes are those recorded in score_final.json").  Their J files,
               CONTROL_JUDGED/<RUN_ID>/judge_<slot>.json as judged for stance4, are re-scored only as a cross-check.

A missing or invalid judge file fails every criterion that needs it (stance3's rule).  Until the runs have ended
(--runs-ended), a missing band J file makes the decision "incomplete" (no line).  With --runs-ended, missing and invalid
band J files are counted as fails and listed, and the line is given.  The decision is PROVISIONAL (each reason printed)
while: a placeholder is not filled (HEAD_LTSB, CRITERIA_SHA256, TASK_SHA256_LTSB, JUDGE_SHA256_S5) or the criteria file
does not match its pin; --validation (or --validate-on-stance4) or a HEAD other than HEAD_LTSB is used; a valid band J
file's task_sha256 is not TASK_SHA256_LTSB (a null or missing one included) or its judge_sha256 is not JUDGE_SHA256_S5;
before --runs-ended, a band J file is invalid; the control check fails (score_final.json missing or not its pinned
sha256, its passes not the stated ones, or the control's verdicts not reproduced from its J files, or one of its valid J
files not carrying LTS's task_sha256 3fa46708, a null or missing one included); or decision line 2 is reached (which the
control's stated passes rule out, criteria CREDIT RULE).

Then the credit rule, (a) H >= 2 and (b) in its general form (for each of K1-K4, the fraction of band runs passing >=
the fraction of control runs passing, from the control's recorded passes), beside its specific form (at least 2 of the 3
band runs pass K3 and at least 2 pass K4; equal to the general form while the control stands at K1 0/3, K2 0/3, K3 2/3,
K4 2/3), H, K2p, the DRIFT OR SPLIT class of every K2-failing band run (criteria section DRIFT OR SPLIT: per J file
the statue's median common CoP m0; the checkpoints counted are the J checkpoints where K2a or K2b fails (run_criteria's
own "over" lists); at those, each full-horizon K2a- or K2b-failing episode with finite per-foot readings is
NEAR-REFERENCE when a foot lies within 15 mm (15/70 span half-lengths) of m0, else COMMON when (r - m0)(l - m0) > 0 and
SPLIT otherwise; per run DRIFT when K2a and K2b pass at every J checkpoint (K2c alone), else DRIFT / SPLIT / OTHER from
n_c, n_s and K2c; a K2-passing run is not classed), D and S (the band runs classed DRIFT and SPLIT), the decision-table
line, and the parent comparison (report only: stance3's best-final rule, best_final, on the band
finals, on the holding ones (line 1's stance parent), beside stance3 Run 2's final and the best stance4 final; ties are
named, never settled; it does not choose walking pilot 2's parent).  While the decision is incomplete, the parent block
is printed as inputs only.

Report only (never in a verdict): the 1,048,576-step checkpoints not in J, robust_best / best / the late-stable S3
checkpoint, K5, late falls, the saturation alarm (an evaluation of eval_episodes.csv with >= 5 episodes listing
max_actuator_saturation_fraction in screen_v2_reasons; two such evaluations in a row is an alarm; the actuator is named by
tools/satroll5.py), the band readouts the judge files carry (episodes with either foot's window mean outside [-0.6, +0.4],
full-horizon episodes failing min_foot_load_share_windowed, and for the band runs the judge's band_summary), the drift or
split class of the control runs, and each run's throughput from progress.json when placed.  With --runs-ended, a band J
checkpoint placed under --runs-root (01_stance/models) without its judge file makes the decision PROVISIONAL (criteria:
the table is read once every J checkpoint that exists has been judged).

--validate-on-stance4 (validation of this script only): the stance4 runs' own J files stand in for the band arm (read
through SC4, the stance4 judge identity, HEAD c1e146a8, task 3fa46708, judge 3c8c0759), so the scorer must reproduce
stance4's K results exactly (each run's criteria equal to the ones recorded in stance4's score_final.json) and the
drift-or-split classes of SP/foot/stance5/criteria_calib/drift_split_calib.txt (SPLIT 0/101/0, 0/67/2, 0/27/1 as
n_c/n_s/n_u).  Its decision is PROVISIONAL by
construction.  Plain python (no repository import).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys

sys.dont_write_bytecode = True

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S3DIR = SCRATCH + "/foot/stance3"
S4DIR = SCRATCH + "/foot/stance4"
S5DIR = SCRATCH + "/foot/stance5"
SC3_PATH = S3DIR + "/judge/score_stance3.py"
SC3_SHA256 = "2c3879d3a83d453159773d6fed7cacdfbe88e5337c83605834b14377588cc66a"
#: score_stance4.py, the control's scorer (its SC4 instance is rebuilt here; the tests compare the two).
SC4_SCORER_PATH = S4DIR + "/judge/score_stance4.py"
SC4_SCORER_SHA256 = "0d35e5405ad007829b2c567d781df859e62e724e0bcc7c8f6a538580c488593b"
CRITERIA_FILE = S5DIR + "/criteria.txt"
CRITERIA_SHA256 = "3b28963f644ffb765e666a8cf6ada5d31ab8a2769dd1a77ff53cbc1e0cf3cf16"  # foot/stance5/criteria.txt, fixed
STANCE4_SCORE_FINAL = S4DIR + "/scores/score_final.json"
STANCE4_SCORE_FINAL_SHA256 = "7c98d8ae592357150c4d379745eab7fafa6ad2ac83e327b4291d2c23974df532"
S5_JUDGED, S4_JUDGED, S3_JUDGED = S5DIR + "/judged", S4DIR + "/judged", S3DIR + "/judged"
S5_RUNS, S4_RUNS = S5DIR + "/runs", S4DIR + "/runs"
JUDGE_SCHEMA_S5 = "stance5-judge/v1"
JUDGE_SCHEMA_S4 = "stance4-judge/v1"
#: claude/screen-trex-r9-armlts-copband's config commit (judge_stance5.py ARMS["LTSB"]["head"]), filled before training.
HEAD_LTSB = "cf80ecb14d1630ce3b4dcdea28a05835d143470b"
#: claude/screen-trex-r9-armlt-stagger's config commit: the commit all three control runs trained.
HEAD_LTS = "c1e146a82658112a9a1c0838250020b214ef303a"
#: criteria.txt: the band-on stance task_sha256 (measured on the config commit's tree).
TASK_SHA256_LTSB = "sha256:8c6998d260b850ae054c9c2d49a8e5bef2bf59d4770ad77d3674b911721918cf"
TASK_SHA256_LTS = "sha256:3fa467083e34fa280ccb6d1bd04eebccdd3aac2b820fd89901b8d418ef32f2db"
#: judge_stance5.py's "judge_sha256" ("sha256:<hex>", as the judge records it), with its HEAD filled; the same sha256 is
#: recorded in stance5_colab_steps.txt before the first training step.
JUDGE_SHA256_S5 = "sha256:d9286724550b83eccfd11e2ceba634cfca847f1e52e03988f9c486c6db6bda13"
#: judge_stance4.py's, as the control's J files record it (report only here: a note if one differs).
JUDGE_SHA256_S4 = "sha256:3c8c07592cd76db84531f187a6d263e33765eea959d4dc8adb3607ffdb8f427d"
_PINNED = object()  # score(head_ltsb=...) default: HEAD_LTSB as it stands when score() runs
ARM_ENV_LTS = {"leg_home_pose_weight": 1.0, "leg_home_pose_tolerance": 0.05,
               "stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045}
ARM_ENV_LTSB = {**ARM_ENV_LTS, "stance_cop_band_weight": 0.30}
BAND_RUNS = (("stance5_LTSB_s52", "LTSB", 52, "Run 1"), ("stance5_LTSB_s60", "LTSB", 60, "Run 2"),
             ("stance5_LTSB_s42", "LTSB", 42, "Run 3"))
CONTROL_RUNS = (("stance4_LTS_s52", "LTS", 52, "stance4 Run 1"), ("stance4_LTS_s60", "LTS", 60, "stance4 Run 2"),
                ("stance4_LTS_s42", "LTS", 42, "stance4 Run 3"))
#: stance3 Run 2's final: walking pilot 2's parent (D-D36 as amended on 2026-10-20: 7a, option i); a reference, report only.
REFERENCE = ("stance3_armLT_s60", "LT", 60, "stance3 Run 2")
KS = ("K1", "K2", "K3", "K4")
#: The control's results as the criteria state them (stance4 score_final.txt); the specific form of (b) rests on them.
CONTROL_STATED = {"stance4_LTS_s52": {"K1": False, "K2": False, "K3": True, "K4": True},
                  "stance4_LTS_s60": {"K1": False, "K2": False, "K3": True, "K4": True},
                  "stance4_LTS_s42": {"K1": False, "K2": False, "K3": False, "K4": False}}
CONTROL_STATED_PASSES = {"K1": 0, "K2": 0, "K3": 2, "K4": 2}
SAT_KEY, SAT_MIN, SAT_IN_A_ROW = "max_actuator_saturation_fraction", 5, 2
BAND_LO, BAND_HI = -0.6, 0.4   # the band's whole-foot edges (report only: episodes with a foot's window mean outside)
LOAD_SHARE_RAIL = "min_foot_load_share_windowed"
LOAD_SHARE_RAIL_WINDOW = "min_foot_load_share"   # the lighter foot's share of the whole window (criteria: the door's rails)
#: criteria DRIFT OR SPLIT: an episode with a foot within 15 mm of m0 is NEAR-REFERENCE (not classed); r9: x = 60 + 70 c mm.
NEAR_REF_MM = 15.0
NEAR_REF = NEAR_REF_MM / 70.0
PATCHED = ("JUDGE_SCHEMA", "ARM_ENV", "HEADS")
DECISION_LINES = {  # verbatim from criteria.txt's DECISION TABLE (test_decision_lines_are_verbatim)
    1: '1. H >= 2 and (b) holds: ADOPT. Propose committing the band option (off by default) plus its config '
       'keys, with the stance stagger term, to the maintainer as the T. rex stance recipe; the best holding '
       'band final is offered as the stance parent.',
    2: '2. H >= 2 but (b) fails: not adopted. The band holds fore-aft but may cost robustness; replicate on '
       "2 more seeds before any commit. With the control's stated passes this line cannot be reached (CREDIT"
       ' RULE). If it is reached, the control check has failed and the decision is PROVISIONAL.',
    3: '3. H == 1: not adopted; not separated from seed luck; replicate on 2 more seeds before any commit.',
    4: '4. H == 0 and K2p >= 2: not adopted; the band holds fore-aft from scratch but the runs fail '
       "elsewhere (K1, K3 or K4). The next round targets those failures. The scorer names each failing run's"
       ' tests and K1 failure modes, and the maintainer chooses among:\n   - for K1 on the '
       'actuator-saturation rail, the tail saturation habit. Evidence (measured, report only; '
       'SP/foot/stance4/satdiag, 20-episode rollouts): stance4 s42 held tail_3_pitch_act at +1 (tail '
       'raised). Its mean command was +0.89 to +1.00 at every rolled checkpoint from 2,097,152 to final, '
       'with |command| >= 0.9 on 74% to 100% of window steps, against +0.07 and 0% at 1,048,576. No other '
       'actuator reached 0.9 on any window step. stance3 Run 2 held the same actuator at the same sign at '
       '3,145,728 to 5,242,880 (90% to 100% of window steps);\n   - for K1 through other new failure modes '
       "(hop, nosedive, head contact), stance3 line 4's reading: the dose may be too strong;\n   - for K3 or "
       'K4, the leg exploration-noise floor of stance3 line 2 (log_std >= ln 0.08 on the hip-pitch, knee and'
       ' ankle actuators).',
    5: '5. H == 0, K2p <= 1, D >= 2 and S == 0 (the TRIGGER LINE, D-D40 4a): not adopted. At least two band '
       'runs fail K2 on drift, and none fails it mainly on split: each K2-failing band run is DRIFT (it '
       'failed K2 through K2c alone, or its COMMON episodes outnumber its SPLIT ones at the checkpoints '
       'counted, or tie with K2c failing) or OTHER. The criteria read this as D-D40 4a\'s "the band fixes the'
       ' split but runs fail K2 on drift (a common fore-aft shift)". How far the split is fixed is not '
       "tested beyond this; the scorer prints each run's n_c and n_s beside the control's. The next round "
       'tests the v6b ankle spring held in reserve: ankle stiffness 40 -> 250 N.m/rad, damping 45 -> 90, the'
       " spring reference at the statue's settled 78.6 deg (a plant revision).",
    6: '6. H == 0, K2p <= 1, and D <= 1 or S >= 1: not adopted; the band does not hold fore-aft from '
       'scratch, and fewer than two band runs fail K2 on drift or at least one fails it mainly on split. The'
       " next round chooses between a stronger band (w 0.50, design.md section 3.4's named alternative) and "
       'another candidate the maintainer names (with D >= 2, the v6b spring of line 5 among them); the '
       'maintainer decides.',
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


SC3 = load_stance3("score_stance3")              # stance3 Run 2's final (report only), unchanged
SC4 = load_stance3("score_stance3_for_stance4")  # the control: score_stance4.py's stagger instance
SC4.JUDGE_SCHEMA = JUDGE_SCHEMA_S4
SC4.ARM_ENV = {**SC3.ARM_ENV, "LTS": dict(ARM_ENV_LTS)}
SC4.HEADS = {**SC3.HEADS, "LTS": HEAD_LTS}
SC5 = load_stance3("score_stance3_for_stance5")  # the band arm: the same code, the stance5 judge identity
SC5.JUDGE_SCHEMA = JUDGE_SCHEMA_S5
SC5.ARM_ENV = {**SC3.ARM_ENV, "LTSB": dict(ARM_ENV_LTSB)}
SC5.HEADS = {**SC3.HEADS, "LTSB": HEAD_LTSB}
J = SC3.J
J_LABELS = tuple(lab for lab, _ in J)
J_STEPS = {s for _, s in J}
REPORT_1M = tuple((f"{k * 1048576 / 1e6:.1f}M", k * 1048576) for k in range(1, 12) if k * 1048576 not in J_STEPS)
SELECTION = (("robust_best", "robust_best"), ("best", "best"))


def num(v):
    """A finite float or None (as score_stance3.num)."""
    return SC3.num(v)


# ------------------------------------------------------------------------------------------------ report only
def saturation_alarm(rows: list[dict]) -> dict:
    """Per evaluation of eval_episodes.csv: episodes listing max_actuator_saturation_fraction in screen_v2_reasons;
    an evaluation with >= 5 is flagged, and two flagged evaluations in a row are an alarm (report only).  Identical to
    score_stance4.py's saturation_alarm (the criteria's reference for the alarm step)."""
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


def file_readouts(mod, path: str) -> dict:
    """From one valid judge file: the DRIFT OR SPLIT counts of its K2a- or K2b-failing episodes (criteria; drift_split
    sums them only over the J checkpoints where K2a or K2b fails) and the report-only band readouts."""
    with open(path) as fh:
        d = json.load(fh)
    p = d["panel"]
    eps, st = p["episodes_detail"], p.get("statue_episodes_detail") or []

    def feet(e):
        w = e.get("foot_cop_fore_aft") or {}
        return num(w.get("r")), num(w.get("l"))

    commons = [(r + l) / 2.0 for e in st if e.get("reached_horizon") for r, l in [feet(e)] if r is not None and l is not None]
    m0 = statistics.median(commons) if commons else None
    n_c = n_s = n_u = n_unclassed = 0
    classed = []
    for e in eps:
        if not (mod.k2a_over(e) or mod.k2b_outside(e)):
            continue
        r, l = feet(e)
        if not e.get("reached_horizon") or r is None or l is None or m0 is None:
            n_unclassed += 1
            continue
        a, b = r - m0, l - m0
        near = min(abs(a), abs(b))
        cls = "NEAR-REFERENCE" if near < NEAR_REF else "COMMON" if a * b > 0 else "SPLIT"
        n_c += int(cls == "COMMON")
        n_s += int(cls == "SPLIT")
        n_u += int(cls == "NEAR-REFERENCE")
        classed.append({"seed": e["seed"], "r": r, "l": l, "class": cls, "nearer_foot_mm": 70.0 * near})
    full = [e for e in eps if e.get("reached_horizon")]
    out_band = [e["seed"] for e in full for r, l in [feet(e)] if r is not None and l is not None
                and (r < BAND_LO or r > BAND_HI or l < BAND_LO or l > BAND_HI)]
    keys = {e["seed"]: {SC3.reason_key(x) for x in (e.get("reasons") or [])} for e in full}
    share_fail = [s for s, k in keys.items() if LOAD_SHARE_RAIL in k]
    share_fail_w = [s for s, k in keys.items() if LOAD_SHARE_RAIL_WINDOW in k]
    share_fail_any = [s for s, k in keys.items() if LOAD_SHARE_RAIL in k or LOAD_SHARE_RAIL_WINDOW in k]
    return {"m0": m0, "statue_episodes_for_m0": len(commons), "n_common": n_c, "n_split": n_s, "n_near_reference": n_u,
            "k2_failing_not_classed": n_unclassed, "classed": classed,
            "band_outside_fh": len(out_band), "band_outside_fh_seeds": out_band, "full_horizon": len(full),
            "load_share_windowed_fail_fh": len(share_fail), "load_share_windowed_fail_fh_seeds": share_fail,
            "load_share_window_fail_fh": len(share_fail_w), "load_share_window_fail_fh_seeds": share_fail_w,
            "load_share_either_fail_fh": len(share_fail_any), "load_share_either_fail_fh_seeds": share_fail_any,
            "band_summary": p.get("band_summary"),
            "statue_band_summary": (p.get("statue_on_same_seeds") or {}).get("band_summary"),
            "judge_sha256": d.get("judge_sha256")}


def progress_readout(runs_root: str, run: str, ledger_dir: str | None) -> dict | None:
    path = SC3._find(runs_root, run, "progress.json", ledger_dir)
    if not path:
        return None
    try:
        with open(path) as fh:
            prog = json.load(fh)
    except (OSError, ValueError):
        return None
    return {"path": path, "status": prog.get("status"), "timesteps": prog.get("timesteps"),
            "steps_per_s": prog.get("steps_per_s"), "written_at": prog.get("written_at")}


# ------------------------------------------------------------------------------------------------ one run
def load_run(mod, root: str, run: str, arm: str, seed: int, name: str, *, head, frozen: dict, validation: bool,
             notes: list, report_sources: list, runs_root: str, ledger_dir: str | None) -> dict:
    """One run's cells (J through *mod*; report-only slots from the first of *report_sources* holding the file), its
    K1-K4 (mod.run_criteria), its per-file DRIFT OR SPLIT counts and its report-only readouts."""
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
    readouts = {lab: file_readouts(mod, c["path"]) for lab, c in cells.items()
                if c is not None and (lab in J_LABELS or lab in dict(REPORT_1M))}
    return {"arm": arm, "seed": seed, "name": name, "head": head, "cells": cells, "missing": missing, "invalid": invalid,
            "report_invalid": report_invalid, "criteria": crits, "k1_failure_modes": mod.k1_failure_modes(cells),
            "ledgers": {"eval_episodes": ev_path, "train_episodes": tr_path},
            "K5": SC3.k5_readout(evals, statue_site) if ev_path else {"computed": False, "why": "no eval_episodes.csv"},
            "late_falls": SC3.late_falls(SC3.read_csv(tr_path)) if tr_path else None,
            "saturation_alarm": saturation_alarm(ev_rows) if ev_path else {"computed": False, "why": "no eval_episodes.csv"},
            "S3": s3, "readouts": readouts, "progress": progress_readout(runs_root, run, ledger_dir)}


def drift_split(r: dict, *, runs_ended: bool) -> dict:
    """criteria DRIFT OR SPLIT: the run's class from its six J files and its K2c.  The checkpoints counted are the J
    checkpoints where K2a or K2b fails (run_criteria's own "over" lists, score_stance3.py's code); n_c, n_s, n_u and the
    falls or unread episodes are summed over those only.  A K2-passing run is not classed."""
    gone = [lab for lab in J_LABELS if r["cells"].get(lab) is None]
    k2 = r["criteria"]["K2"]
    failing = set(k2["K2a"]["over"]) | set(k2["K2b"]["over"])
    counted = [lab for lab in J_LABELS if lab in failing]
    out = {"K2_pass": bool(k2["pass"]), "K2c_pass": bool(k2["K2c"]["pass"]), "K2c_range_mm": k2["K2c"]["range_mm"],
           "counted": counted, "n_common": None, "n_split": None, "n_near_reference": None, "not_classed": None,
           "per_checkpoint": {}, "class": None, "why": None}
    present = {lab: r["readouts"][lab] for lab in J_LABELS if lab in r["readouts"]}
    out["per_checkpoint"] = {lab: {"m0": x["m0"], "counted": lab in failing, "n_common": x["n_common"],
                                   "n_split": x["n_split"], "n_near_reference": x["n_near_reference"],
                                   "not_classed": x["k2_failing_not_classed"]} for lab, x in present.items()}
    cnt = [x for lab, x in present.items() if lab in failing]
    out["n_common"] = sum(x["n_common"] for x in cnt)
    out["n_split"] = sum(x["n_split"] for x in cnt)
    out["n_near_reference"] = sum(x["n_near_reference"] for x in cnt)
    out["not_classed"] = sum(x["k2_failing_not_classed"] for x in cnt)
    if gone:
        only_missing = all(r["cells"].get(lab) is None and any(p.endswith(f"/judge_{s}.json") for p in r["missing"])
                           for lab, s in J if lab in gone)
        if only_missing and not runs_ended:
            out.update({"class": "incomplete", "why": f"J judge files missing ({gone}); the runs have not ended"})
        else:
            out.update({"class": "OTHER", "why": f"J judge files missing or invalid: {gone}"})
        return out
    if out["K2_pass"]:
        out.update({"class": "not classed (K2 passes)", "why": "K2 passes"})
        return out
    if not counted:
        out.update({"class": "DRIFT", "why": "K2a and K2b pass at every J checkpoint: K2 failed through K2c alone"})
        return out
    nc, ns, k2c = out["n_common"], out["n_split"], out["K2c_pass"]
    if nc > ns or (nc == ns and not k2c):
        out["class"], out["why"] = "DRIFT", ("n_c > n_s" if nc > ns else "n_c == n_s and K2c fails")
    elif ns > nc or (ns == nc and ns > 0 and k2c):
        out["class"], out["why"] = "SPLIT", ("n_s > n_c" if ns > nc else "n_s == n_c > 0 and K2c passes")
    else:
        out["class"], out["why"] = "OTHER", ("n_c == n_s == 0 and K2c passes (K2a or K2b failed only through falls, missing "
                                             "readings or near-reference episodes)")
    return out


# ------------------------------------------------------------------------------------------------ arm level
def verdicts(crits: dict) -> dict:
    return {**{k: bool(crits[k]["pass"]) for k in KS}, "HOLDS": bool(crits["HOLDS"]),
            "FORE_AFT_HOLDS": bool(crits["FORE_AFT_HOLDS"]), "failed": list(crits["failed"])}


def control_crosscheck(control: dict, score_final: str = STANCE4_SCORE_FINAL) -> dict:
    """The control's passes as RECORDED in stance4's score_final.json (the credit rule's control input; its runs are
    under "stagger" there), checked against its pinned sha256, against the results the criteria state, and against the
    control's recomputed criteria (verdicts, the whole criteria block and each J cell's task_sha256).  *problems* lists
    every check that fails; each one makes the decision provisional.  When the file is missing or unreadable, the
    criteria's stated results stand in."""
    out = {"score_final": score_final, "sha256": None, "sha256_matches": None, "runs": {}, "match": False,
           "stated_match": False, "readable": False, "recorded": None, "reproduced": False, "problems": []}
    stored = None
    if not os.path.exists(score_final):
        out["why"] = "missing"
    else:
        out["sha256"] = sha256_file(score_final)
        out["sha256_matches"] = out["sha256"] == STANCE4_SCORE_FINAL_SHA256
        try:
            with open(score_final) as fh:
                stored = json.load(fh)
        except (OSError, ValueError) as exc:
            out["why"] = f"unreadable: {exc}"
    recorded, reproduced = {}, True
    for run, *_ in CONTROL_RUNS:
        c = control[run]["criteria"]
        s = ((((stored or {}).get("stagger") or {}).get(run) or {}).get("criteria")) if isinstance(stored, dict) else None
        v = verdicts(c)
        try:
            sv = verdicts(s) if s else None
        except (KeyError, TypeError):
            sv = None
        if sv is not None:
            recorded[run] = {k: sv[k] for k in KS}
        same_v = sv == v
        same_all = s is not None and json.loads(json.dumps(c)) == s
        j_cells = [x for lab, _ in J for x in [control[run]["cells"].get(lab)] if x is not None]
        tasks = sorted({x.get("task_sha256") for x in j_cells}, key=lambda t: (t is None, t or ""))
        task_ok = bool(j_cells) and all(x.get("task_sha256") == TASK_SHA256_LTS for x in j_cells)
        out["runs"][run] = {"verdicts": v, "stance4_verdicts": sv, "verdicts_match": same_v, "criteria_identical": same_all,
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
        p.append(f"stance4's score_final.json is {out.get('why') or 'without the three control runs'}: the control passes "
                 "are the criteria's stated ones")
    if out["sha256"] is not None and not out["sha256_matches"]:
        p.append(f"stance4's score_final.json sha256 {out['sha256']} is not the pinned {STANCE4_SCORE_FINAL_SHA256}")
    if out["readable"] and not out["stated_match"]:
        p.append("the control passes recorded in score_final.json are not the ones the criteria state "
                 "(K1 0/3, K2 0/3, K3 2/3, K4 2/3)")
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
        return [xcheck["recorded"][r[0]] for r in CONTROL_RUNS], "recorded in stance4's score_final.json"
    return [dict(CONTROL_STATED[r[0]]) for r in CONTROL_RUNS], "the criteria's stated results (score_final.json unreadable)"


def credit_rule(band: list[dict], control: list[dict]) -> dict:
    """(a), (b) in its general form (exact fractions, from the control's recorded passes) and in its specific form, H
    and K2p.  *band* are run_criteria outputs; *control* are {K1..K4: bool} (control_passes)."""
    nb, nc = len(band), len(control)
    pb = {k: sum(bool(c[k]["pass"]) for c in band) for k in KS}
    pc = {k: sum(bool(c[k]) for c in control) for k in KS}
    by_test = {k: {"band": f"{pb[k]}/{nb}", "control": f"{pc[k]}/{nc}", "no_worse": pb[k] * nc >= pc[k] * nb}
               for k in KS}
    H = sum(bool(c["HOLDS"]) for c in band)
    a = H >= 2
    b = all(t["no_worse"] for t in by_test.values())
    specific = pb["K3"] >= 2 and pb["K4"] >= 2
    control_as_stated = nc == 3 and pc == CONTROL_STATED_PASSES
    return {"H": H, "K2p": pb["K2"], "a": a, "b": b, "b_by_test": by_test, "b_specific": specific,
            "control_as_stated": control_as_stated,
            "forms_agree": (not control_as_stated) or (specific == b),
            "a_implies_b": (not control_as_stated) or (not a) or b,
            "band_passes": pb, "control_passes": pc, "n_band": nb, "n_control": nc}


def decision_line(H: int, K2p: int, b: bool, D: int, S: int) -> int:
    """The decision table (exclusive and exhaustive)."""
    if H >= 2:
        return 1 if b else 2
    if H == 1:
        return 3
    if K2p >= 2:
        return 4
    return 5 if (D >= 2 and S == 0) else 6


def best_final_among(mod, runs, per_run: dict) -> dict:
    """stance3's best_final (most clean at final; tie -> fewer joint + own noise falls) over *runs*."""
    saved = mod.RUNS
    mod.RUNS = tuple(runs)
    try:
        return mod.best_final(per_run)
    finally:
        mod.RUNS = saved


def parents(band: dict, control: dict, reference: dict, runs: tuple, ctl_runs: tuple, *, inputs_only: bool) -> dict:
    """Report only.  Line 1's stance parent (best holding band final), the best band final, and the comparison with
    stance3 Run 2's final and the best stance4 final (stance3's rule; a tie names every tied final).  It does not choose
    walking pilot 2's parent (D-D36 as amended on 2026-10-20, 7a, option i: stance3 Run 2's final)."""
    holding = [r for r in runs if band[r[0]]["criteria"]["HOLDS"]]
    best_band = best_final_among(SC5, runs, band)
    best_ctl = best_final_among(SC4, ctl_runs, control)
    pool_runs, pool = list(ctl_runs), {**control}
    if reference.get("cell") is not None:
        pool_runs.append(REFERENCE)
        pool[REFERENCE[0]] = {"cells": {"final": reference["cell"]}}
    best_ctl_name = best_ctl.get("best")
    comparison_runs = list(runs) + [r for r in pool_runs if r[0] not in {x[0] for x in runs} and (
        r[0] == REFERENCE[0] or f"{r[3]} {r[0]}" == best_ctl_name or f"{r[3]} {r[0]}" in best_ctl.get("tied", []))]
    comparison = best_final_among(SC5, comparison_runs, {**band, **pool})
    return {"inputs_only": inputs_only, "rule": "most clean panel episodes at final; tie -> fewer noise falls (joint-space "
            "+ own-sigma)", "best_band_final": best_band,
            "best_holding_band_final": best_final_among(SC5, holding, band) if holding else
            {"best": None, "tied": [], "candidates": []},
            "best_stance4_final": best_ctl, "reference": reference,
            "comparison": {**comparison, "top": None if inputs_only else comparison.get("best"),
                           "top_tied": [] if inputs_only else comparison.get("tied", [])},
            "walking_pilot_2_parent": "stance3 Run 2's final (D-D36 as amended on 2026-10-20: 7a, option i); not chosen here"}


def reference_final(frozen: dict, notes: list, score_final: str) -> dict:
    """stance3 Run 2's final cell (SC3, stance3's own judge file), cross-checked against the copy stance4's
    score_final.json recorded (report only)."""
    run, arm, seed, name = REFERENCE
    c, why, entry = SC3.load_cell(S3_JUDGED, run, arm, seed, "final", "final", expect_head=SC3.HEADS[arm], frozen=frozen,
                                  validation=False, needs_noise=True, notes=notes)
    out = {"run": f"{name} {run}", "cell": c, "why": why, "clean": None if c is None else c["n_clean"],
           "joint_noise_falls": None if c is None else c["noise_joint_falls"],
           "own_noise_falls": None if c is None else c["noise_own_falls"], "matches_stance4_record": None}
    try:
        with open(score_final) as fh:
            stored = (((json.load(fh).get("control") or {}).get(run) or {}).get("cells") or {}).get("final")
        out["matches_stance4_record"] = c is not None and stored is not None and json.loads(json.dumps(c)) == stored
    except (OSError, ValueError):
        pass
    if out["matches_stance4_record"] is False:
        notes.append(f"{run} final: the cell recomputed from stance3's judge file differs from stance4's record")
    return out


# ------------------------------------------------------------------------------------------------ scoring
def score(judged: str = S5_JUDGED, control_judged: str = S4_JUDGED, *, runs_root: str = S5_RUNS,
          control_runs_root: str = S4_RUNS, head_ltsb=_PINNED, frozen_path: str = SC3.FROZEN_PARAMS,
          validation: bool = False, ledger_dirs: dict | None = None, criteria_file: str = CRITERIA_FILE,
          criteria_sha256=_PINNED, score_final: str = STANCE4_SCORE_FINAL, runs_ended: bool = False,
          validate_on_stance4: bool = False) -> dict:
    if head_ltsb is _PINNED:
        head_ltsb = HEAD_LTSB
    if criteria_sha256 is _PINNED:
        criteria_sha256 = CRITERIA_SHA256
    fsha = sha256_file(frozen_path)
    if fsha != SC3.FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{frozen_path} sha256 {fsha} != {SC3.FROZEN_PARAMS_SHA256}")
    with open(frozen_path) as fh:
        frozen = json.load(fh)
    # The band arm: the stance5 runs through SC5, or (--validate-on-stance4) the stance4 runs' own files through SC4.
    if validate_on_stance4:
        bmod, runs, bhead = SC4, CONTROL_RUNS, HEAD_LTS
        task_pin, judge_pin, judged, runs_root = TASK_SHA256_LTS, JUDGE_SHA256_S4, S4_JUDGED, S4_RUNS
    else:
        bmod, runs, bhead = SC5, BAND_RUNS, head_ltsb
        task_pin, judge_pin = TASK_SHA256_LTSB, JUDGE_SHA256_S5
    ledger_dirs = {k: os.path.abspath(v) for k, v in (ledger_dirs or {}).items()}
    known = {r[0] for r in runs + CONTROL_RUNS}
    if set(ledger_dirs) - known:
        raise SystemExit(f"--ledgers names no screen run: {sorted(set(ledger_dirs) - known)}")
    notes: list[str] = []
    crit = {"path": criteria_file, "sha256": sha256_file(criteria_file) if os.path.exists(criteria_file) else None,
            "pinned": criteria_sha256}
    crit["matches"] = criteria_sha256 is not None and crit["sha256"] == criteria_sha256
    provisional_why = []
    if criteria_sha256 is None:
        provisional_why.append("CRITERIA_SHA256 is a PLACEHOLDER (the criteria sha256 is not pinned)")
    elif not crit["matches"]:
        provisional_why.append(f"criteria file {criteria_file} sha256 {crit['sha256']} != the pinned {criteria_sha256}")
    if HEAD_LTSB is None:
        provisional_why.append("HEAD_LTSB is a PLACEHOLDER (not pinned)")
    if validate_on_stance4:
        provisional_why.append("--validate-on-stance4: the stance4 runs' files stand in for the band arm; not a screen verdict")
    else:
        if head_ltsb != HEAD_LTSB:
            provisional_why.append(f"HEAD {head_ltsb} is not the pinned HEAD_LTSB {HEAD_LTSB}")
        if head_ltsb is None:
            provisional_why.append("no band HEAD: every band file is invalid until it is filled")
    if TASK_SHA256_LTSB is None:
        provisional_why.append("TASK_SHA256_LTSB is a PLACEHOLDER (not pinned)")
    if JUDGE_SHA256_S5 is None:
        provisional_why.append("JUDGE_SHA256_S5 is a PLACEHOLDER (not pinned)")
    if validation:
        provisional_why.append("--validation: relaxed-binding judge files accepted; not a screen verdict")
    SC5.HEADS = {**SC3.HEADS, "LTSB": head_ltsb}
    band = {run: load_run(bmod, judged, run, arm, seed, name, head=bhead, frozen=frozen, validation=validation,
                          notes=notes, report_sources=[(bmod, judged)], runs_root=runs_root,
                          ledger_dir=ledger_dirs.get(run))
            for run, arm, seed, name in runs}
    control = {run: load_run(SC4, control_judged, run, arm, seed, name, head=HEAD_LTS, frozen=frozen,
                             validation=validation, notes=notes, report_sources=[(SC4, control_judged)],
                             runs_root=control_runs_root, ledger_dir=ledger_dirs.get(run))
               for run, arm, seed, name in CONTROL_RUNS}

    def _tasks(cells):  # every task_sha256 the cells carry, a null or missing one as None (sorted last)
        return sorted({c.get("task_sha256") for c in cells}, key=lambda t: (t is None, t or ""))

    for run, r in band.items():
        valid = {lab: c for lab, c in r["cells"].items() if c is not None}
        tasks = _tasks(valid.values())
        j_tasks = _tasks(c for lab, c in valid.items() if lab in J_LABELS)
        r["task_sha256_seen"] = tasks
        # criteria: a band J file's task_sha256 other than the band-on one, or a judge_sha256 other than the recorded
        # judge's, makes the decision provisional (report-only files: a note).
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
        if task_pin is not None:
            off_t = sorted(lab for lab, c in valid.items() if lab in J_LABELS and c.get("task_sha256") != task_pin)
            if off_t:
                provisional_why.append(f"{run}: J task_sha256 {j_tasks} is not the criteria's band-on {task_pin} "
                                       f"(J files {off_t}; None is a null or missing one)")
            elif any(t != task_pin for t in tasks):
                notes.append(f"{run}: a report-only file's task_sha256 {tasks} is not the band-on {task_pin}")
        if judge_pin is not None:
            off = sorted(lab for lab, s in judges.items() if lab in J_LABELS and s != judge_pin)
            if off:
                provisional_why.append(f"{run}: J files {off} were not judged by the recorded judge {judge_pin}")
            off_r = sorted(lab for lab, s in judges.items() if lab not in J_LABELS and s != judge_pin)
            if off_r:
                notes.append(f"{run}: report-only files {off_r} were not judged by the recorded judge")
    for run, r in control.items():
        off = sorted(lab for lab in J_LABELS if lab in r["readouts"] and r["readouts"][lab]["judge_sha256"] != JUDGE_SHA256_S4)
        if off:
            notes.append(f"control {run}: J files {off} carry a judge_sha256 other than stance4's recorded judge")
    xcheck = control_crosscheck(control, score_final)
    provisional_why += [f"control check: {p}" for p in xcheck["problems"]]
    cpasses, csource = control_passes(xcheck)
    cr = credit_rule([band[r[0]]["criteria"] for r in runs], cpasses)
    cr["control_source"] = csource
    if not cr["control_as_stated"]:
        notes.append("the control's passes are not K1 0/3, K2 0/3, K3 2/3, K4 2/3: the specific form of (b) "
                     "does not apply; the general form decides")
    # DRIFT OR SPLIT: every band run is classed (report only); D counts the K2-failing ones classed DRIFT.
    for run, r in band.items():
        r["drift_split"] = drift_split(r, runs_ended=runs_ended)
    for run, r in control.items():
        r["drift_split"] = drift_split(r, runs_ended=True)
    D = sum(1 for r in band.values() if not r["criteria"]["K2"]["pass"] and r["drift_split"]["class"] == "DRIFT")
    S = sum(1 for r in band.values() if not r["criteria"]["K2"]["pass"] and r["drift_split"]["class"] == "SPLIT")
    cr["D"], cr["S"] = D, S
    cr["drift_split_classes"] = {run: r["drift_split"]["class"] for run, r in band.items()}
    missing = [p for run, *_ in runs for p in band[run]["missing"]]
    invalid = [x["path"] for r in band.values() for x in r["invalid"]]
    if invalid and not runs_ended:
        provisional_why.append(f"invalid band J judge files (the runs have not ended): {invalid}")
    unjudged = placed_unjudged(runs, judged, runs_root)
    if unjudged and runs_ended:
        provisional_why.append("band J checkpoints placed under the runs root but not judged (the table is read once every "
                               f"J checkpoint that exists has been judged): {unjudged}")
    if missing and not runs_ended:
        dec = {"status": "incomplete", "line": None, "text": None, "missing": missing, "provisional_why": provisional_why,
               "runs_ended": False, "counted_as_fails": []}
    else:
        n = decision_line(cr["H"], cr["K2p"], cr["b"], D, S)
        if n == 2:
            provisional_why.append("decision line 2 reached: (a) holds but (b) fails, which the control's stated passes "
                                   "rule out (criteria CREDIT RULE); the control check has failed")
        dec = {"status": "provisional" if provisional_why else "decided", "line": n, "text": DECISION_LINES[n],
               "missing": missing, "provisional_why": provisional_why, "runs_ended": runs_ended,
               "counted_as_fails": (missing + invalid) if runs_ended else [], "placed_unjudged": unjudged}
    reference = reference_final(frozen, notes, score_final)
    res = {"schema": "stance5-score/v1", "criteria": crit, "validate_on_stance4": validate_on_stance4,
           "stance3_scorer": {"path": SC3_PATH, "sha256": SC3_SHA256, "patched": list(PATCHED),
                              "band_judge_schema": JUDGE_SCHEMA_S5, "arm_env_ltsb": ARM_ENV_LTSB,
                              "control_judge_schema": JUDGE_SCHEMA_S4, "arm_env_lts": ARM_ENV_LTS},
           "judged_root": os.path.abspath(judged), "control_judged_root": os.path.abspath(control_judged),
           "runs_root": os.path.abspath(runs_root), "control_runs_root": os.path.abspath(control_runs_root),
           "ledger_dirs": ledger_dirs, "validation": validation, "runs_ended": runs_ended, "head_ltsb": head_ltsb,
           "head_ltsb_pinned": HEAD_LTSB, "head_band_used": bhead, "frozen_params_sha256": fsha,
           "task_sha256": {"LTSB_pinned": TASK_SHA256_LTSB, "LTS": TASK_SHA256_LTS, "band_pin_used": task_pin},
           "judge_sha256_pinned": JUDGE_SHA256_S5, "judge_sha256_band_pin_used": judge_pin,
           "band_runs": [list(r) for r in runs], "band": band, "control": control, "control_crosscheck": xcheck,
           "credit_rule": cr, "decision": dec, "notes": notes}
    res["parents"] = parents(band, control, {k: v for k, v in reference.items()}, runs, CONTROL_RUNS,
                             inputs_only=dec["status"] == "incomplete")
    if validate_on_stance4:
        res["validation_on_stance4"] = stance4_reproduction(band, score_final)
    return res


def placed_unjudged(runs, judged: str, runs_root: str) -> list[str]:
    """The band J checkpoints placed under *runs_root* (<RUN>/01_stance/models, the zip and its VecNormalize pkl) whose
    judge file (*judged*/<RUN>/judge_<steps>.json) does not exist."""
    out = []
    for run, *_ in runs:
        models = os.path.join(runs_root, run, "01_stance", "models")
        for _lab, steps in J:
            pair = (("stage1_final.zip", "stage1_final_vecnorm.pkl") if steps == "final" else
                    (f"stage1_{steps}_steps.zip", f"stage1_vecnormalize_{steps}_steps.pkl"))
            if all(os.path.exists(os.path.join(models, x)) for x in pair) and not os.path.exists(
                    os.path.join(judged, run, f"judge_{steps}.json")):
                out.append(f"{run}/{pair[0]}")
    return out


def stance4_reproduction(band: dict, score_final: str) -> dict:
    """--validate-on-stance4: each stance4 run's criteria, recomputed here as the band arm, against stance4's record, and
    the drift-or-split classes against the calibration's (SPLIT, SPLIT, SPLIT; n_c/n_s/n_u 0/101/0, 0/67/2, 0/27/1)."""
    want_cls = {"stance4_LTS_s52": ("SPLIT", 0, 101, 0), "stance4_LTS_s60": ("SPLIT", 0, 67, 2),
                "stance4_LTS_s42": ("SPLIT", 0, 27, 1)}
    with open(score_final) as fh:
        stored = json.load(fh)
    out = {"runs": {}, "all_match": True}
    for run, r in band.items():
        s = ((stored.get("stagger") or {}).get(run) or {})
        crit_same = json.loads(json.dumps(r["criteria"])) == s.get("criteria")
        cells_same = all(json.loads(json.dumps(r["cells"][lab])) == (s.get("cells") or {}).get(lab) for lab in J_LABELS)
        ds = r["drift_split"]
        got_cls = (ds["class"], ds["n_common"], ds["n_split"], ds["n_near_reference"])
        cls_same = got_cls == want_cls[run]
        out["runs"][run] = {"criteria_identical": crit_same, "J_cells_identical": cells_same, "failed": r["criteria"]["failed"],
                            "drift_split": list(got_cls), "drift_split_as_calibrated": cls_same}
        out["all_match"] = out["all_match"] and crit_same and cells_same and cls_same
    return out


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
    ds = r.get("drift_split") or {}
    L.append(f"   drift or split: {ds.get('class')} (checkpoints counted {ds.get('counted') or 'none'}; n_c "
             f"{fmt(ds.get('n_common'))}, n_s {fmt(ds.get('n_split'))}, n_u {fmt(ds.get('n_near_reference'))} near-reference, "
             f"falls or unread {fmt(ds.get('not_classed'))}; K2c {'pass' if ds.get('K2c_pass') else 'FAIL'}; {ds.get('why')})")
    near = [f"{lab} {c['seed']} ({c['class'][0]}, nearer foot {c['nearer_foot_mm']:.1f} mm)"
            for lab in (ds.get("counted") or []) for c in (r.get("readouts") or {}).get(lab, {}).get("classed", [])
            if c["class"] != "SPLIT" or c["nearer_foot_mm"] < NEAR_REF_MM + 5.0]
    if near:
        L.append("     COMMON, near-reference (N) and near-the-margin (< 20 mm) episodes counted (report only): "
                 + "; ".join(near))
    fm = r["k1_failure_modes"]
    if not c["K1"]["pass"]:
        L.append(f"   K1 failure modes: {json.dumps(fm['by_checkpoint'], sort_keys=True)} (new modes: {fmt(fm['new_modes'])})")
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
                    + ", ".join(f"{x['from']:,}-{x['to']:,} ({x['evaluations']})" for x in sa["runs"])
                    + "; name the actuator with tools/satroll5.py" if sa["alarm"] else "")
                 + ")")
    else:
        L.append(f"     saturation alarm: not computed ({sa.get('why')})")
    ro = r.get("readouts") or {}
    band_rows = []
    for lab, _steps in list(J) + list(REPORT_1M):
        x = ro.get(lab)
        if x is None:
            continue
        s = (f"{lab} out-of-band {x['band_outside_fh']}/{x['full_horizon']}, load-share rails windowed / window / either "
             f"{x['load_share_windowed_fail_fh']}/{x['load_share_window_fail_fh']}/{x['load_share_either_fail_fh']}")
        bs = x.get("band_summary")
        if bs:
            fh = bs["full_horizon"]
            pp = bs.get("paid_per_full_horizon_episode") or {}
            s += (f", guard 0/ramp {fh['guard_0']}/{fh['guard_ramp']} of {fh['window_foot_steps']}, whole/pad out "
                  f"{fh['whole_out']}/{fh['pad_out']}, pad binds {fh['pad_binds']}, share [0.25,0.35)/<0.25 "
                  f"{fh['share_ramp_025_035']}/{fh['share_below_025']}, paid mean {fmt(pp.get('mean'), 2)}")
        band_rows.append(s)
    L.append("     band readouts (full-horizon): " + ("; ".join(band_rows) if band_rows else "none"))
    pr = r.get("progress")
    if pr:
        L.append(f"     throughput: {fmt(pr.get('steps_per_s'))} steps/s at {fmt(pr.get('timesteps'))} ({pr.get('status')}, "
                 f"{pr.get('written_at')})")
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
    v4 = res["validate_on_stance4"]
    L = ["T. rex stance5 screen: score against foot/stance5/criteria.txt (sha256 "
         f"{crit['sha256']}; " + ("PLACEHOLDER, not pinned" if crit["pinned"] is None else
                                   "matches the fixed text" if crit["matches"] else "DOES NOT MATCH the pinned " + crit["pinned"]) + ")",
         f"K1-K4 per run: score_stance3.py (sha256 {SC3_SHA256[:16]}...) loaded unchanged; the band instance differs only "
         f"in {', '.join(PATCHED)} (schema {JUDGE_SCHEMA_S5}, arm LTSB, HEAD {res['head_ltsb'] or 'PLACEHOLDER'}); the "
         f"control instance is score_stance4.py's (schema {JUDGE_SCHEMA_S4}, arm LTS, HEAD {HEAD_LTS[:8]})."]
    if res["validation"]:
        L.append("*** VALIDATION ONLY: relaxed-binding judge files accepted (--validation); not a screen verdict ***")
    if v4:
        L.append("*** VALIDATION ONLY (--validate-on-stance4): the stance4 runs' J files stand in for the band arm ***")
    L.append(f"band judge files: {res['judged_root']}   control judge files: {res['control_judged_root']}")
    L.append("")
    L.append("BAND ARM (LTSB: L+T + stagger 0.30 / 0.045 + CoP band w 0.30)" + ("  [stand-ins: stance4 runs]" if v4 else ""))
    for run in res["band"]:
        L += run_block(run, res["band"][run])
    L.append("")
    L.append("CONTROL ARM (stance4 LTS, not retrained; judged and scored under the stance4 criteria)")
    for run, *_ in CONTROL_RUNS:
        L += run_block(run, res["control"][run])
    x = res["control_crosscheck"]
    L.append(f"   control verdicts reproduce stance4 score_final.json: {'yes' if x['match'] else 'NO'} "
             f"(file sha256 {'pinned' if x['sha256_matches'] else 'NOT the pinned one'}); as the criteria state them "
             f"(s52 and s60 fail K1 K2, s42 fails K1-K4): {'yes' if x['stated_match'] else 'NO'}")
    for p in x["problems"]:
        L.append(f"   control check: {p}")
    L.append("")
    cr = res["credit_rule"]
    L.append(f"CREDIT RULE (control passes: {cr['control_source']})")
    L.append("   per test, runs passing (band vs control): " + "; ".join(
        f"{k} {t['band']} vs {t['control']} {'ok' if t['no_worse'] else 'WORSE'}" for k, t in cr["b_by_test"].items()))
    L.append(f"   H = {cr['H']} (band runs that HOLD); K2p = {cr['K2p']} (band runs passing K2); D = {cr['D']} "
             f"(K2-failing band runs classed DRIFT); S = {cr['S']} (classed SPLIT); D and S are read only at H == 0 and "
             "K2p <= 1")
    L.append(f"   (a) at least 2 of 3 band runs HOLD: {'yes' if cr['a'] else 'no'}")
    L.append(f"   (b) general form, from the control's recorded passes: {'holds' if cr['b'] else 'fails'}")
    L.append(f"   (b) specific form, at least 2 of 3 band runs pass K3 and at least 2 pass K4: "
             f"{'holds' if cr['b_specific'] else 'fails'} (equal to the general form while the control stands at K1 0/3, "
             f"K2 0/3, K3 2/3, K4 2/3: {'it does' if cr['control_as_stated'] else 'it does NOT; the general form decides'}"
             + ("" if cr["forms_agree"] else "; THE FORMS DISAGREE") + ")")
    L.append("   drift or split, band runs (n_c, n_s, n_u): " + "; ".join(
        f"{run} {r['drift_split']['class']} ({fmt(r['drift_split']['n_common'])}, {fmt(r['drift_split']['n_split'])}, "
        f"{fmt(r['drift_split']['n_near_reference'])})" for run, r in res["band"].items())
             + "   (control, report only: " + "; ".join(f"{run} {r['drift_split']['class']} ({r['drift_split']['n_common']}, "
                                                        f"{r['drift_split']['n_split']}, {r['drift_split']['n_near_reference']})"
                                                        for run, r in res["control"].items()) + ")")
    d = res["decision"]
    if d["status"] == "incomplete":
        L.append(f"DECISION: incomplete ({len(d['missing'])} band J judge files missing; the counts above treat them as fails). "
                 "The decision is read once all three runs have ended (--runs-ended).")
    else:
        L.append("DECISION" + (" (PROVISIONAL)" if d["status"] == "provisional" else "") + f": {d['text']}")
        if d["line"] == 4:
            for run, r in res["band"].items():
                L.append(f"   line 4 targets, {run}: failed {r['criteria']['failed']}; K1 failure modes "
                         f"{json.dumps(r['k1_failure_modes']['by_checkpoint'], sort_keys=True)}; saturation alarm "
                         f"{r['saturation_alarm'].get('first_alarm_at') if r['saturation_alarm'].get('alarm') else 'none'}")
    if d.get("placed_unjudged"):
        L.append("   placed but not judged (band J checkpoints): " + "; ".join(d["placed_unjudged"]))
    if d.get("counted_as_fails"):
        L.append(f"   runs ended (--runs-ended): {len(d['counted_as_fails'])} band J judge files missing or invalid, "
                 "counted as fails: " + "; ".join(d["counted_as_fails"]))
    for why in d["provisional_why"]:
        L.append(f"   provisional: {why}")
    p = res["parents"]
    tie = " (the maintainer chooses)"
    L.append("")
    L.append("PARENT COMPARISON (report only; stance3's rule: most clean panel episodes at final; ties to fewer joint + own "
             "falls; a tie names every tied final)")
    if p["inputs_only"]:
        L.append("   INPUTS ONLY: the decision is incomplete; the comparison is read once all three runs have ended.")
    for key, title in (("best_band_final", "best band final"),
                       ("best_holding_band_final", "best holding band final (line 1's stance parent)"),
                       ("best_stance4_final", "best stance4 final")):
        b = p[key]
        L.append(f"   {title}: " + (b["best"] or ("TIE: " + ", ".join(b["tied"]) + tie if b.get("tied") else "none")))
    ref = p["reference"]
    L.append(f"   reference {ref['run']} final: clean {fmt(ref['clean'])}/40, noise falls joint {fmt(ref['joint_noise_falls'])}"
             f" / own {fmt(ref['own_noise_falls'])} (as stance4 recorded it: {fmt(ref['matches_stance4_record'])})")
    for c in p["comparison"]["candidates"]:
        L.append(f"     {c['run']:<34} clean {c['clean']:>2}/40  joint + own noise falls {fmt(c['noise_falls'])}")
    if not p["inputs_only"]:
        cm = p["comparison"]
        L.append("   top under the rule (band finals, stance3 Run 2's final, the best stance4 final): "
                 + (cm["top"] if cm["top"] else ("TIE: " + ", ".join(cm["top_tied"]) + tie if cm["top_tied"] else "none")))
    L.append(f"   walking pilot 2's parent: {p['walking_pilot_2_parent']}")
    if v4:
        vv = res["validation_on_stance4"]
        L.append("")
        L.append(f"VALIDATION ON STANCE4: stance4's K results and the calibrated drift-or-split classes reproduced: "
                 f"{'yes' if vv['all_match'] else 'NO'}")
        for run, x in vv["runs"].items():
            L.append(f"   {run}: criteria identical {fmt(x['criteria_identical'])}, J cells identical "
                     f"{fmt(x['J_cells_identical'])}, failed {x['failed']}, drift or split {x['drift_split']} "
                     f"(as calibrated: {fmt(x['drift_split_as_calibrated'])})")
    if res["notes"]:
        L.append("")
        L.append("Notes:")
        for n in res["notes"]:
            L.append(f"   {n}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judged", default=S5_JUDGED, help="directory holding the band runs' <RUN_ID>/judge_<slot>.json")
    ap.add_argument("--control-judged", default=S4_JUDGED, help="directory holding the control runs' judge files (stance4)")
    ap.add_argument("--runs-root", default=S5_RUNS, help="the band runs' <RUN_ID>/[01_stance/]{eval,train}_episodes.csv")
    ap.add_argument("--control-runs-root", default=S4_RUNS, help="the control runs' ledgers")
    ap.add_argument("--head-LTSB", default=HEAD_LTSB,
                    help="the band branch's config commit (default: HEAD_LTSB; any other makes the decision provisional)")
    ap.add_argument("--runs-ended", action="store_true",
                    help="all three band runs have ended (criteria): missing or invalid J files count as fails")
    ap.add_argument("--frozen-params", default=SC3.FROZEN_PARAMS)
    ap.add_argument("--validation", action="store_true", help="validation of this script only: relaxed-binding files accepted")
    ap.add_argument("--validate-on-stance4", action="store_true",
                    help="validation of this script only: score the stance4 runs' own J files as the band arm")
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
                head_ltsb=a.head_LTSB, frozen_path=a.frozen_params, validation=a.validation, ledger_dirs=ledger_dirs,
                runs_ended=a.runs_ended or a.validate_on_stance4, validate_on_stance4=a.validate_on_stance4)
    print(render(res))
    if a.json:
        with open(a.json, "x") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
