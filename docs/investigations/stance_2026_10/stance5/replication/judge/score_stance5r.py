"""Score the stance5 replication (two more LTSB band runs, seeds 70 and 80) with the three stance5 band runs, against
the replication's fixed criteria.

    python3 -I score_stance5r.py [--judged R/judged] [--runs-root R/runs] [--head-LTSB SHA] [--runs-ended]
                                 [--ledgers RUN_ID=DIR ...] [--validation] [--json OUT.json]
    python3 -I score_stance5r.py --validate [--json OUT.json]

R = SP/foot/stance5/replication.  The criteria: R/criteria.txt (CRITERIA_SHA256_R below, pinned).

What is reused, unchanged (never edited):
  * score_stance5.py (sha256 pinned, S5_SCORER_SHA256), loaded by path as a module; through it score_stance3.py (its
    own pin) as score_stance5.py loads it: SC3 (stance3 Run 2's final, report only), SC4 (the stance4 control, schema
    stance4-judge/v1) and SC5 (the stance5 band runs, schema stance5-judge/v1, arm LTSB, HEAD cf80ecb1).
  * One more instance of score_stance3.py's code, SC5R, made the way score_stance5.py makes SC5 (load_stance3, then
    only the judge-identity constants JUDGE_SCHEMA, ARM_ENV, HEADS): schema stance5r-judge/v1, arm LTSB, HEAD cf80ecb1.
    It reads the two new runs.  Every bar, slot and rule is score_stance3.py's.
  * score_stance5.py's load_run, drift_split, control_crosscheck, control_passes, parents, reference_final,
    placed_unjudged, run_block and saturation_alarm, called as they are.

The runs:
  stance5 (recorded)  Run 1 stance5_LTSB_s52, Run 2 stance5_LTSB_s60, Run 3 stance5_LTSB_s42: S5/judged through SC5 with
                      frozen_params.json as it is.  Their verdicts are the ones S5/scores/score_final.json records (sha256
                      pinned).  They are recomputed only as a check: unless every criteria block, J cell and K1
                      failure-mode block is identical to the record, every J file carries the band-on task and
                      judge_stance5.py's sha256, and the record reads decision line 3 decided with the runs ended, the
                      scorer REFUSES (SystemExit, no decision).
  new                 Run 4 stance5_LTSB_s70, Run 5 stance5_LTSB_s80: R/judged/<RUN_ID>/judge_<slot>.json through SC5R.
                      The joint-noise check (score_stance3.validity_issues) reads frozen_params.json's
                      criteria.H4.control_sigma, so the new runs are loaded with a copy of it to which the sidecar's
                      seeds 70 and 80 are added (vector and source, as judge_stance5r.py records them).  The sidecar is
                      pinned by its sha256 and its vector is recomputed here from frozen_params.json (refused if not
                      equal).
  control             the three stance4 runs through SC4 and score_stance5.py's control cross-check, read only for the
                      report-only K3 fraction and the parent comparison: its problems are notes, never provisional.

The decision (criteria DECISION TABLE): line 1 ADOPT when s70 and s80 both HOLD (H = 3 of 5 with s52), line 2 when
exactly one does, line 3 when neither does.  Until --runs-ended, a missing new J file makes the decision "incomplete".
With --runs-ended a missing or invalid new J file fails every criterion that needs it (the run does not hold) and is
listed.  PROVISIONAL (each reason printed) while a pin is a placeholder (CRITERIA_SHA256_R, JUDGE_SHA256_S5R, HEAD_LTSB)
or the criteria file does not match its pin; --validation or a HEAD other than HEAD_LTSB; a new J file's task_sha256 is
not the band-on one or its judge_sha256 not judge_stance5r.py's; before --runs-ended a new J file is invalid; with
--runs-ended a new J checkpoint placed under the runs root has no judge file and no refusal record.

Run handling (criteria RUN HANDLING, the maintainer's rules of 2026-10-22 16:32 UTC):
  * a judge refused on the run's own records leaves R/judged/<RUN_ID>/refused_<steps>.json (judge_one5r.sh, schema
    stance5r-refusal/v1): that J slot counts as INVALID (never "missing").  A record that is not its slot's, or not
    written for the pinned judge, or that stands beside a judge file of the same slot, makes the decision provisional.
  * an early stop (the placed 01_stance/gate_verdict.json's stage_result.timesteps below 12,001,280, read under
    --runs-root whatever --ledgers names): its stage1_final is not the final slot, so the final J file is INVALID (a
    judge file there included), and a placed stage1_final pair is not a J checkpoint waiting for a judge.
  * report only (notes): run records placed with place5r.py config --run-records-fail (01_stance/
    records_check_failed.json), and a judge refusal on the checkpoint's own attributes held for review
    (refused_<steps>.review.json), which counts for nothing: with --runs-ended its placed checkpoint is placed but
    unjudged.
  * a run restarted under <RUN_ID>_b (or _c) before its first checkpoint is that run: place5r.py --source-folder places
    it under the RUN_ID (R/runs/<RUN_ID>/source_folder.json, printed as a note); it is scored like any other.
  * on the refusal above (the stance5 runs do not reproduce), the message says what follows: the maintainer is told at
    once, no line is read or acted on, and the scorer is run again with every pinned file unchanged once the cause is
    found.

Report only: the K3 pass fraction of the five band runs against the control's recorded passes (with K1, K2 and K4
beside it; not a veto, stance5's rule (b) is not applied), FORE-AFT HOLDS per run, each new run's band readouts,
saturation alarm, K5, late falls, selection and drift-or-split class (score_stance5.py's run block), and the parent
comparison over the five band finals (stance3's rule) beside stance3 Run 2's final and the best stance4 final.

--validate (validation of this script): the three stance5 runs only; prints their per-run tables and whether each
run's criteria, J cells and K1 failure modes reproduce score_final.json's (exit 0 when all do, 1 otherwise).  It reads
no new run and gives no decision.  Plain python (no repository import); nothing is unpickled.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True  # nothing is written next to score_stance5.py or score_stance3.py when they are loaded

SCRATCH = "scratchpad"
S5DIR = SCRATCH + "/foot/stance5"
RDIR = S5DIR + "/replication"
S5_SCORER_PATH = S5DIR + "/judge/score_stance5.py"
S5_SCORER_SHA256 = "d5515041f73a299a96963f953bac7117e3ffb7c73570572ad9fc6bc189888a77"
CRITERIA_FILE_R = RDIR + "/criteria.txt"
#: criteria.txt's sha256 (None = a PLACEHOLDER: the decision is provisional); re-pinned if the text changes before
#: the first training step (test_pins_are_the_files_as_they_stand fails until then).
CRITERIA_SHA256_R = "a38536de152e735947fd5322960f9e75b6520b59a5526a414afc0d0c4a0f3ef2"
STANCE5_SCORE_FINAL = S5DIR + "/scores/score_final.json"
STANCE5_SCORE_FINAL_SHA256 = "c690165043dbc32fc8bf5bcce7a57c8bc38692c67e21464e706017d0dd45e307"
SIDECAR = RDIR + "/sigma_s70_s80.json"
SIDECAR_SHA256 = "89778b3f81f83f05d4c15423b3658593dfaa767388213044c50a72e2d96b8522"
SIDECAR_SEEDS = ("70", "80")
SIDECAR_SOURCE_SEEDS = ("42", "52", "60")
JUDGE_SCHEMA_S5R = "stance5r-judge/v1"
#: judge_stance5r.py's "judge_sha256" as the judge records it ("sha256:<hex>"); recorded in colab_steps.txt too.
JUDGE_SHA256_S5R = "sha256:16d39f7dcb14a8b6f94061331933df8f1dcee76aa988479e293b132bf8ac05ae"
R_JUDGED, R_RUNS = RDIR + "/judged", RDIR + "/runs"
NEW_RUNS = (("stance5_LTSB_s70", "LTSB", 70, "Run 4"), ("stance5_LTSB_s80", "LTSB", 80, "Run 5"))
KS = ("K1", "K2", "K3", "K4")
DECISION_LINES_R = {  # verbatim from criteria.txt's DECISION TABLE (test_decision_lines_are_verbatim)
    1: "1. s70 and s80 both HOLD (with s52, H = 3 of 5): ADOPT. Propose committing the band option (off by default) plus "
       "its config keys, with the stance stagger term, to the maintainer as the T. rex stance recipe; the best holding "
       "band final among the five, under stance3's parent rule, is offered as the stance parent (tied finals are named, "
       "and the maintainer chooses).",
    2: "2. Exactly one of s70 and s80 HOLDS (H = 2 of 5): not adopted. The scorer names the other new run's failing tests "
       "and K1 failure modes. The maintainer chooses the next round; stance5 decision line 4's options are the menu (the "
       "tail saturation habit, the dose, the leg exploration-noise floor).",
    3: "3. Neither s70 nor s80 HOLDS (H = 1 of 5): not adopted. The scorer names each new run's failing tests and K1 "
       "failure modes. The maintainer chooses the next round from the same menu.",
}
_PINNED = object()


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_stance5():
    """score_stance5.py as a module (refuses any other file content); it loads score_stance3.py with its own pin."""
    got = sha256_file(S5_SCORER_PATH)
    if got != S5_SCORER_SHA256:
        raise SystemExit(f"{S5_SCORER_PATH} sha256 {got} != the pinned {S5_SCORER_SHA256}")
    spec = importlib.util.spec_from_file_location("score_stance5_for_stance5r", S5_SCORER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S5 = load_stance5()
SC3, SC4, SC5 = S5.SC3, S5.SC4, S5.SC5
HEAD_LTSB = S5.HEAD_LTSB                       # cf80ecb1, the config commit the stance5 runs trained (unchanged)
TASK_SHA256_LTSB = S5.TASK_SHA256_LTSB         # the band-on task 8c6998d2 (unchanged)
JUDGE_SHA256_S5 = S5.JUDGE_SHA256_S5           # judge_stance5.py's, on the stance5 J files
STANCE5_RUNS = S5.BAND_RUNS
FIVE = STANCE5_RUNS + NEW_RUNS
SC5R = S5.load_stance3("score_stance3_for_stance5r")  # the new runs: the same code, the stance5r judge identity
SC5R.JUDGE_SCHEMA = JUDGE_SCHEMA_S5R
SC5R.ARM_ENV = {**SC3.ARM_ENV, "LTSB": dict(S5.ARM_ENV_LTSB)}
SC5R.HEADS = {**SC3.HEADS, "LTSB": HEAD_LTSB}
PATCHED = S5.PATCHED
J, J_LABELS = S5.J, S5.J_LABELS
fmt = S5.fmt
FINAL_TIMESTEPS = SC3.FINAL_TIMESTEPS          # 12,001,280: the final slot (criteria; score_stance3.py's)
REFUSAL_SCHEMA = "stance5r-refusal/v1"         # judge_one5r.sh's record of a judge refused on the run's own records
RECORDS_FAILED = "records_check_failed.json"    # place5r.py config --run-records-fail (report only here)
REFUSAL_SEQUEL = ("Tell the maintainer at once; no line is read or acted on.  Once the cause is found, run the scorer "
                  "again with every pinned file unchanged (criteria RUN HANDLING).")


# ------------------------------------------------------------------------------------------------ the sigma sidecar
def sidecar_mean(frozen: dict) -> list[float]:
    """The criteria's rule, recomputed: mean[i] = math.fsum((v42[i], v52[i], v60[i])) / 3.0."""
    cs = frozen["criteria"]["H4"]["control_sigma"]
    vecs = [cs[s]["vector"] for s in SIDECAR_SOURCE_SEEDS]
    return [math.fsum(v[i] for v in vecs) / 3.0 for i in range(len(vecs[0]))]


def load_sidecar(frozen: dict, frozen_sha: str, path: str = SIDECAR, pin: str = _PINNED) -> dict:
    """The pinned sidecar, its source sha256 and its vectors checked against the recomputed mean (refused otherwise)."""
    pin = SIDECAR_SHA256 if pin is _PINNED else pin
    got = sha256_file(path)
    if got != pin:
        raise SystemExit(f"{path} sha256 {got} != the pinned {pin}")
    with open(path) as fh:
        sc = json.load(fh)
    if sc.get("source_sha256") != frozen_sha:
        raise SystemExit(f"{path} source_sha256 {sc.get('source_sha256')} is not frozen_params.json's {frozen_sha}")
    want = sidecar_mean(frozen)
    for s in SIDECAR_SEEDS:
        e = (sc.get("control_sigma") or {}).get(s) or {}
        if e.get("vector") != want or not isinstance(e.get("source"), str):
            raise SystemExit(f"{path}: control_sigma[{s}] is not the element-wise mean of control_sigma 42, 52, 60 "
                             "(or has no source string)")
    return sc


def frozen_with_sidecar(frozen: dict, sidecar: dict) -> dict:
    """A copy of frozen_params.json with the sidecar's seeds 70 and 80 added to criteria.H4.control_sigma (42, 52, 60
    untouched): what score_stance3.validity_issues reads for the new runs' joint-noise check."""
    out = copy.deepcopy(frozen)
    cs = out["criteria"]["H4"]["control_sigma"]
    for s in SIDECAR_SEEDS:
        if s in cs:
            raise SystemExit(f"frozen_params.json already holds control_sigma[{s}]")
        cs[s] = {"source": sidecar["control_sigma"][s]["source"], "vector": list(sidecar["control_sigma"][s]["vector"])}
    return out


# ------------------------------------------------------------------------------------------------ the stance5 runs
def stance5_reproduction(band5: dict, score_final: str, pin: str) -> dict:
    """Each stance5 run's recomputed criteria, J cells and K1 failure modes against score_final.json's record, its J
    files' task and judge sha256, and the record's decision (line 3, decided, runs ended).  *problems* empty = it
    reproduces."""
    out = {"score_final": score_final, "sha256": None, "sha256_pinned": pin, "runs": {}, "problems": [],
           "recorded": {}, "decision_recorded": None}
    p = out["problems"]
    if not os.path.exists(score_final):
        p.append(f"{score_final} is missing")
        return out
    out["sha256"] = sha256_file(score_final)
    if out["sha256"] != pin:
        p.append(f"{score_final} sha256 {out['sha256']} is not the pinned {pin}")
    with open(score_final) as fh:
        stored = json.load(fh)
    dec = stored.get("decision") or {}
    out["decision_recorded"] = {k: dec.get(k) for k in ("status", "line", "runs_ended")}
    if not (dec.get("status") == "decided" and dec.get("line") == 3 and dec.get("runs_ended") is True):
        p.append(f"score_final.json does not record decision line 3, decided, with the runs ended: {out['decision_recorded']}")
    for run, *_ in STANCE5_RUNS:
        r, s = band5[run], (stored.get("band") or {}).get(run) or {}
        crit_same = json.loads(json.dumps(r["criteria"])) == s.get("criteria")
        cells_same = all(json.loads(json.dumps(r["cells"].get(lab))) == (s.get("cells") or {}).get(lab) for lab in J_LABELS)
        k1m_same = json.loads(json.dumps(r["k1_failure_modes"])) == s.get("k1_failure_modes")
        tasks = {lab: (r["cells"][lab] or {}).get("task_sha256") for lab in J_LABELS if r["cells"].get(lab) is not None}
        judges = {lab: r["readouts"][lab]["judge_sha256"] for lab in J_LABELS if lab in r["readouts"]}
        task_ok = len(tasks) == len(J_LABELS) and all(t == TASK_SHA256_LTSB for t in tasks.values())
        judge_ok = len(judges) == len(J_LABELS) and all(j == JUDGE_SHA256_S5 for j in judges.values())
        try:
            rec = S5.verdicts(s["criteria"])
        except (KeyError, TypeError):
            rec = None
        out["recorded"][run] = rec
        out["runs"][run] = {"criteria_identical": crit_same, "J_cells_identical": cells_same,
                            "k1_failure_modes_identical": k1m_same, "task_sha256_ok": task_ok, "judge_sha256_ok": judge_ok,
                            "recomputed": S5.verdicts(r["criteria"]), "recorded": rec}
        bad = [k for k, ok in (("criteria", crit_same), ("J cells", cells_same), ("K1 failure modes", k1m_same),
                               ("J task_sha256", task_ok), ("J judge_sha256", judge_ok)) if not ok]
        if bad:
            p.append(f"{run}: {', '.join(bad)} do not reproduce score_final.json's record")
    out["reproduced"] = not p
    return out


# ------------------------------------------------------------------------------------------------ run handling
def refusal_entry(path: str, run: str, seed: int, steps) -> dict:
    """A judge refusal record (criteria RUN HANDLING): an invalid J file.  slot_ok: it is this slot's record."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        return {"path": path, "issues": [f"a judge refusal record that cannot be read ({exc})"], "refusal": None,
                "slot_ok": False, "judge_sha256": None}
    if not isinstance(d, dict):
        d = {}
    ok = (d.get("schema") == REFUSAL_SCHEMA and d.get("run") == run and d.get("seed") == seed
          and str(d.get("steps")) == str(steps) and d.get("mode") == "full")
    return {"path": path, "issues": [f"the judge refused on the run's own records (criteria RUN HANDLING): "
                                     f"{d.get('refusal')}"],
            "refusal": d.get("refusal"), "slot_ok": ok, "judge_sha256": d.get("judge_sha256")}


def early_stop(runs_root: str, run: str) -> dict | None:
    """criteria RUN HANDLING, Ended: a gate_verdict.json written before 12,001,280 is an early stop.  None when no
    gate_verdict.json is placed; timesteps None when it records no stage_result.timesteps.  It reads the placed copy
    (runs_root/<RUN_ID>/01_stance/gate_verdict.json) whatever --ledgers names: score_stance3.py's _find with a ledger
    directory looks only there, and would miss it."""
    path = SC3._find(runs_root, run, "gate_verdict.json")
    if not path:
        return None
    try:
        with open(path) as fh:
            ts = ((json.load(fh) or {}).get("stage_result") or {}).get("timesteps")
    except (OSError, ValueError, AttributeError):
        ts = None
    ts = ts if isinstance(ts, int) and not isinstance(ts, bool) else None
    return {"gate_verdict": path, "timesteps": ts, "early": ts is not None and ts < FINAL_TIMESTEPS}


def apply_run_handling(mod, r: dict, run: str, seed: int, judged: str, runs_root: str, notes: list,
                       provisional_why: list) -> None:
    """A new run's refusal records and early stop (criteria RUN HANDLING) applied to its J slots; K1-K4 and the K1
    failure modes are recomputed with score_stance3.py's code when a cell changes."""
    r["refusals"], r["early_stop"], r["source_folder"] = [], early_stop(runs_root, run), None
    mk = os.path.join(runs_root, run, "01_stance", RECORDS_FAILED)
    if os.path.exists(mk):  # report only: place5r.py config --run-records-fail
        try:
            with open(mk) as fh:
                probs = "; ".join(json.load(fh).get("problems") or [])
        except (OSError, ValueError, AttributeError, TypeError):
            probs = "unreadable"
        notes.append(f"{run}: its run records were placed with place5r.py --run-records-fail ({probs}); the judge "
                     "refuses on them, and each slot it refused holds refused_<steps>.json, an invalid J file (criteria "
                     "RUN HANDLING)")
    sf = os.path.join(runs_root, run, "source_folder.json")
    if os.path.exists(sf):
        try:
            with open(sf) as fh:
                r["source_folder"] = json.load(fh).get("source_folder")
        except (OSError, ValueError, AttributeError):
            r["source_folder"] = "unreadable"
        notes.append(f"{run}: placed from the Drive folder {r['source_folder']} (a restart before the first periodic "
                     "checkpoint is that run; criteria RUN HANDLING)")
    for _lab, steps in J:
        rp = os.path.join(judged, run, f"refused_{steps}.json")
        rv = os.path.join(judged, run, f"refused_{steps}.review.json")
        if os.path.exists(rv):  # report only: judge_one5r.sh holds it until it is confirmed (renamed) or moved aside
            notes.append(f"{run}: {rv} is a judge refusal on the checkpoint's own attributes only, held for review: it "
                         "counts for nothing until it is renamed to refused_<steps>.json or moved aside")
        if not os.path.exists(rp):
            continue
        e = refusal_entry(rp, run, seed, steps)
        r["refusals"].append(e)
        if not e["slot_ok"]:
            provisional_why.append(f"{run}: {rp} is not this slot's judge refusal record ({REFUSAL_SCHEMA}: run, seed, "
                                   "steps, mode full)")
        if JUDGE_SHA256_S5R is not None and e["judge_sha256"] != JUDGE_SHA256_S5R:
            provisional_why.append(f"{run}: {rp} was not written for the pinned judge {JUDGE_SHA256_S5R}")
        jp = os.path.join(judged, run, f"judge_{steps}.json")
        if os.path.exists(jp):
            provisional_why.append(f"{run}: both {os.path.basename(jp)} and {os.path.basename(rp)} exist (move one aside)")
            continue
        r["missing"] = [p for p in r["missing"] if p != jp]
        r["invalid"].append({"path": rp, "issues": list(e["issues"])})
    es, changed = r["early_stop"], False
    if es is not None and es["timesteps"] is None:
        notes.append(f"{run}: {es['gate_verdict']} records no stage_result.timesteps (an early stop is not read from it)")
    if es is not None and es["early"]:
        why = (f"the run stopped early at {es['timesteps']:,} (gate_verdict.json written before {FINAL_TIMESTEPS:,}): its "
               "stage1_final is not the final slot (criteria RUN HANDLING)")
        jp = os.path.join(judged, run, "judge_final.json")
        r["missing"] = [p for p in r["missing"] if p != jp]
        for x in r["invalid"]:
            if x["path"] in (jp, os.path.join(judged, run, "refused_final.json")):
                x["issues"].append(why)
                break
        else:
            r["invalid"].append({"path": jp, "issues": [why]})
        if r["cells"].get("final") is not None:
            r["cells"]["final"] = None
            r["readouts"].pop("final", None)
            changed = True
    if changed:
        r["criteria"] = mod.run_criteria(r["cells"])
        r["k1_failure_modes"] = mod.k1_failure_modes(r["cells"])


def placed_unjudged_r(judged: str, runs_root: str, new: dict) -> list[str]:
    """score_stance5.py's placed_unjudged for the new runs, less the J checkpoints that are settled without a judge
    file: a recorded judge refusal, and an early stop's stage1_final (not a J checkpoint; criteria RUN HANDLING)."""
    out = []
    for u in S5.placed_unjudged(NEW_RUNS, judged, runs_root):
        run, name = u.split("/", 1)
        m = re.fullmatch(r"stage1_(\d+)_steps\.zip", name)
        steps = m.group(1) if m else "final"
        if os.path.exists(os.path.join(judged, run, f"refused_{steps}.json")):
            continue
        if steps == "final" and (new[run].get("early_stop") or {}).get("early"):
            continue
        out.append(u)
    return out


# ------------------------------------------------------------------------------------------------ decision
def decision_line_r(new_holds: dict) -> int:
    """The replication's table: 1 when both new runs HOLD, 2 when exactly one does, 3 when neither does."""
    n = sum(bool(v) for v in new_holds.values())
    return {2: 1, 1: 2, 0: 3}[n]


def fractions(band: list[dict], control: list[dict]) -> dict:
    """Report only: per K test, band runs passing (of the five) beside the control's recorded passes (of three)."""
    nb, nc = len(band), len(control)
    pb = {k: sum(bool(c[k]["pass"]) for c in band) for k in KS}
    pc = {k: sum(bool(c[k]) for c in control) for k in KS}
    return {k: {"band": f"{pb[k]}/{nb}", "control": f"{pc[k]}/{nc}", "band_passes": pb[k], "control_passes": pc[k],
                "band_at_least_control": pb[k] * nc >= pc[k] * nb} for k in KS}


def score(judged: str = R_JUDGED, *, runs_root: str = R_RUNS, head_ltsb=_PINNED, frozen_path: str = SC3.FROZEN_PARAMS,
          validation: bool = False, ledger_dirs: dict | None = None, criteria_file: str = CRITERIA_FILE_R,
          criteria_sha256=_PINNED, runs_ended: bool = False, validate: bool = False,
          stance5_judged: str = S5.S5_JUDGED, stance5_runs_root: str = S5.S5_RUNS,
          stance5_score_final: str = STANCE5_SCORE_FINAL, stance5_score_final_sha256=_PINNED,
          control_judged: str = S5.S4_JUDGED, control_runs_root: str = S5.S4_RUNS,
          control_score_final: str = S5.STANCE4_SCORE_FINAL, sidecar_path: str = SIDECAR, sidecar_sha256=_PINNED) -> dict:
    head_ltsb = HEAD_LTSB if head_ltsb is _PINNED else head_ltsb
    criteria_sha256 = CRITERIA_SHA256_R if criteria_sha256 is _PINNED else criteria_sha256
    s5_pin = STANCE5_SCORE_FINAL_SHA256 if stance5_score_final_sha256 is _PINNED else stance5_score_final_sha256
    fsha = sha256_file(frozen_path)
    if fsha != SC3.FROZEN_PARAMS_SHA256:
        raise SystemExit(f"{frozen_path} sha256 {fsha} != {SC3.FROZEN_PARAMS_SHA256}")
    with open(frozen_path) as fh:
        frozen = json.load(fh)
    sidecar = load_sidecar(frozen, fsha, sidecar_path, SIDECAR_SHA256 if sidecar_sha256 is _PINNED else sidecar_sha256)
    frozen_r = frozen_with_sidecar(frozen, sidecar)
    ledger_dirs = {k: os.path.abspath(v) for k, v in (ledger_dirs or {}).items()}
    known = {r[0] for r in NEW_RUNS}
    if set(ledger_dirs) - known:
        raise SystemExit(f"--ledgers names no replication run: {sorted(set(ledger_dirs) - known)}")
    notes: list[str] = []

    # The stance5 runs, as score_stance5.py read them (SC5, frozen_params as it is, HEAD cf80ecb1).
    band5 = {run: S5.load_run(SC5, stance5_judged, run, arm, seed, name, head=HEAD_LTSB, frozen=frozen, validation=False,
                              notes=notes, report_sources=[(SC5, stance5_judged)], runs_root=stance5_runs_root,
                              ledger_dir=None)
             for run, arm, seed, name in STANCE5_RUNS}
    for r in band5.values():
        r["drift_split"] = S5.drift_split(r, runs_ended=True)
    repro = stance5_reproduction(band5, stance5_score_final, s5_pin)
    if validate:
        return {"schema": "stance5r-score/v1", "validate": True, "stance5": band5, "stance5_reproduction": repro,
                "notes": notes}
    if not repro["reproduced"]:
        raise SystemExit("refusing to score: the stance5 runs' recorded verdicts do not reproduce (criteria THE STANCE5 "
                         "RUNS): " + " | ".join(repro["problems"]) + ".  " + REFUSAL_SEQUEL)

    crit = {"path": criteria_file, "sha256": sha256_file(criteria_file) if os.path.exists(criteria_file) else None,
            "pinned": criteria_sha256}
    crit["matches"] = criteria_sha256 is not None and crit["sha256"] == criteria_sha256
    provisional_why = []
    if criteria_sha256 is None:
        provisional_why.append("CRITERIA_SHA256_R is a PLACEHOLDER (the criteria sha256 is not pinned)")
    elif not crit["matches"]:
        provisional_why.append(f"criteria file {criteria_file} sha256 {crit['sha256']} != the pinned {criteria_sha256}")
    if JUDGE_SHA256_S5R is None:
        provisional_why.append("JUDGE_SHA256_S5R is a PLACEHOLDER (judge_stance5r.py's sha256 is not pinned)")
    if HEAD_LTSB is None or head_ltsb is None:
        provisional_why.append("HEAD_LTSB is a PLACEHOLDER (not pinned)")
    elif head_ltsb != HEAD_LTSB:
        provisional_why.append(f"HEAD {head_ltsb} is not the pinned HEAD_LTSB {HEAD_LTSB}")
    if validation:
        provisional_why.append("--validation: relaxed-binding judge files accepted; not a replication verdict")
    SC5R.HEADS = {**SC3.HEADS, "LTSB": head_ltsb}

    new = {run: S5.load_run(SC5R, judged, run, arm, seed, name, head=head_ltsb, frozen=frozen_r, validation=validation,
                            notes=notes, report_sources=[(SC5R, judged)], runs_root=runs_root,
                            ledger_dir=ledger_dirs.get(run))
           for run, arm, seed, name in NEW_RUNS}
    for run, r in new.items():
        apply_run_handling(SC5R, r, run, r["seed"], judged, runs_root, notes, provisional_why)
        r["drift_split"] = S5.drift_split(r, runs_ended=runs_ended)
        valid = {lab: c for lab, c in r["cells"].items() if c is not None}
        r["task_sha256_seen"] = sorted({c.get("task_sha256") for c in valid.values()}, key=lambda t: (t is None, t or ""))
        off_t = sorted(lab for lab, c in valid.items() if lab in J_LABELS and c.get("task_sha256") != TASK_SHA256_LTSB)
        if off_t:
            provisional_why.append(f"{run}: J files {off_t} carry a task_sha256 other than the band-on {TASK_SHA256_LTSB}")
        judges = {lab: x["judge_sha256"] for lab, x in r["readouts"].items()}
        r["judge_sha256_seen"] = judges
        if JUDGE_SHA256_S5R is not None:
            off = sorted(lab for lab, s in judges.items() if lab in J_LABELS and s != JUDGE_SHA256_S5R)
            if off:
                provisional_why.append(f"{run}: J files {off} were not judged by the pinned judge {JUDGE_SHA256_S5R}")
            off_r = sorted(lab for lab, s in judges.items() if lab not in J_LABELS and s != JUDGE_SHA256_S5R)
            if off_r:
                notes.append(f"{run}: report-only files {off_r} were not judged by the pinned judge")

    # The control: report only (K3 fraction, parent comparison).  Its cross-check problems are notes.
    control = {run: S5.load_run(SC4, control_judged, run, arm, seed, name, head=S5.HEAD_LTS, frozen=frozen,
                                validation=False, notes=notes, report_sources=[(SC4, control_judged)],
                                runs_root=control_runs_root, ledger_dir=None)
               for run, arm, seed, name in S5.CONTROL_RUNS}
    xcheck = S5.control_crosscheck(control, control_score_final)
    notes += [f"control check (report only; the control enters no decision): {p}" for p in xcheck["problems"]]
    cpasses, csource = S5.control_passes(xcheck)

    five = {**band5, **new}
    missing = [p for run, *_ in NEW_RUNS for p in new[run]["missing"]]
    invalid = [x["path"] for r in new.values() for x in r["invalid"]]
    if invalid and not runs_ended:
        provisional_why.append(f"invalid new-run J judge files (the runs have not ended): {invalid}")
    unjudged = placed_unjudged_r(judged, runs_root, new)
    if unjudged and runs_ended:
        provisional_why.append("new-run J checkpoints placed under the runs root but not judged (the table is read once "
                               f"every J checkpoint that exists has been judged): {unjudged}")
    recorded = {run: repro["recorded"][run]["HOLDS"] for run, *_ in STANCE5_RUNS}
    new_holds = {run: bool(new[run]["criteria"]["HOLDS"]) for run, *_ in NEW_RUNS}
    H = sum(recorded.values()) + sum(new_holds.values())
    if missing and not runs_ended:
        dec = {"status": "incomplete", "line": None, "text": None, "missing": missing, "provisional_why": provisional_why,
               "runs_ended": False, "counted_as_fails": [], "placed_unjudged": unjudged}
    else:
        n = decision_line_r(new_holds)
        dec = {"status": "provisional" if provisional_why else "decided", "line": n, "text": DECISION_LINES_R[n],
               "missing": missing, "provisional_why": provisional_why, "runs_ended": runs_ended,
               "counted_as_fails": (missing + invalid) if runs_ended else [], "placed_unjudged": unjudged}
    dec.update({"H": H, "of": len(FIVE), "new_holds": new_holds, "stance5_holds_recorded": recorded,
                "adopt": dec["line"] == 1})
    targets = {run: {"failed": list(new[run]["criteria"]["failed"]),
                     "k1_failure_modes": new[run]["k1_failure_modes"]["by_checkpoint"],
                     "new_k1_modes": new[run]["k1_failure_modes"]["new_modes"],
                     "saturation_alarm_at": new[run]["saturation_alarm"].get("first_alarm_at")
                     if new[run]["saturation_alarm"].get("alarm") else None,
                     # report only: which J files are missing or invalid (a test that needs one fails on the file, not
                     # on a measurement), and whether the saturation alarm was computed at all
                     "j_files_missing_or_invalid": sorted(os.path.basename(p) for p in new[run]["missing"])
                     + sorted(os.path.basename(x["path"]) for x in new[run]["invalid"]),
                     "saturation_alarm_computed": bool(new[run]["saturation_alarm"].get("computed")),
                     "saturation_alarm_why": new[run]["saturation_alarm"].get("why")}
               for run, *_ in NEW_RUNS}
    report = {"fractions": fractions([five[r[0]]["criteria"] for r in FIVE], cpasses), "control_source": csource,
              "fore_aft": {run: bool(five[run]["criteria"]["FORE_AFT_HOLDS"]) for run, *_ in FIVE},
              "drift_split_classes": {run: five[run]["drift_split"]["class"] for run, *_ in FIVE},
              "rule_b_applied": False}
    reference = S5.reference_final(frozen, notes, control_score_final)
    res = {"schema": "stance5r-score/v1", "criteria": crit, "validate": False,
           "stance5_scorer": {"path": S5_SCORER_PATH, "sha256": S5_SCORER_SHA256},
           "stance3_scorer": {"path": S5.SC3_PATH, "sha256": S5.SC3_SHA256, "patched": list(PATCHED),
                              "new_runs_judge_schema": JUDGE_SCHEMA_S5R, "stance5_judge_schema": S5.JUDGE_SCHEMA_S5},
           "judged_root": os.path.abspath(judged), "runs_root": os.path.abspath(runs_root),
           "stance5_judged_root": os.path.abspath(stance5_judged), "control_judged_root": os.path.abspath(control_judged),
           "ledger_dirs": ledger_dirs, "validation": validation, "runs_ended": runs_ended, "head_ltsb": head_ltsb,
           "head_ltsb_pinned": HEAD_LTSB, "frozen_params_sha256": fsha,
           "sidecar": {"path": sidecar_path, "sha256": sha256_file(sidecar_path), "rule": sidecar.get("rule"),
                       "vector": sidecar["control_sigma"]["70"]["vector"]},
           "task_sha256_pinned": TASK_SHA256_LTSB, "judge_sha256_pinned": JUDGE_SHA256_S5R,
           "stance5_judge_sha256": JUDGE_SHA256_S5, "runs": [list(r) for r in FIVE], "stance5": band5, "new": new,
           "control": control, "control_crosscheck": xcheck, "stance5_reproduction": repro, "decision": dec,
           "targets": targets, "report": report, "notes": notes}
    res["parents"] = S5.parents(five, control, dict(reference), FIVE, S5.CONTROL_RUNS,
                                inputs_only=dec["status"] == "incomplete")
    return res


# ------------------------------------------------------------------------------------------------ printing
def repro_lines(repro: dict) -> list[str]:
    L = [f"   stance5 verdicts as score_final.json records them (sha256 "
         f"{'pinned' if repro['sha256'] == repro['sha256_pinned'] else 'NOT the pinned one'}; decision recorded "
         f"{repro['decision_recorded']}): reproduced {'yes' if repro['reproduced'] else 'NO'}"]
    for run, x in repro["runs"].items():
        rec = x["recorded"] or {}
        L.append(f"     {run}: recorded {'HOLDS' if rec.get('HOLDS') else 'does not hold (failed ' + ', '.join(rec.get('failed') or []) + ')'}; "
                 f"criteria {fmt(x['criteria_identical'])}, J cells {fmt(x['J_cells_identical'])}, K1 modes "
                 f"{fmt(x['k1_failure_modes_identical'])}, task {fmt(x['task_sha256_ok'])}, judge {fmt(x['judge_sha256_ok'])}")
    for p in repro["problems"]:
        L.append(f"     problem: {p}")
    return L


def render(res: dict) -> str:
    if res.get("validate"):
        L = ["T. rex stance5 replication scorer, --validate: the three stance5 runs re-scored through score_stance5.py's "
             "band instance (stance5-judge/v1), against score_final.json", ""]
        for run, *_ in STANCE5_RUNS:
            L += S5.run_block(run, res["stance5"][run])
        L.append("")
        L += repro_lines(res["stance5_reproduction"])
        L.append(f"VALIDATION: score_final.json's per-run verdicts reproduced: "
                 f"{'yes' if res['stance5_reproduction']['reproduced'] else 'NO'}")
        return "\n".join(L)
    crit = res["criteria"]
    L = ["T. rex stance5 replication (s70, s80): score against foot/stance5/replication/criteria.txt (sha256 "
         f"{crit['sha256']}; " + ("PLACEHOLDER, not pinned" if crit["pinned"] is None else
                                   "matches the fixed text" if crit["matches"] else "DOES NOT MATCH the pinned " + str(crit["pinned"])) + ")",
         f"K1-K4 per run: score_stance3.py (sha256 {S5.SC3_SHA256[:16]}...) loaded unchanged through score_stance5.py (sha256 "
         f"{S5_SCORER_SHA256[:16]}...); the new runs' instance differs only in {', '.join(PATCHED)} (schema "
         f"{JUDGE_SCHEMA_S5R}, arm LTSB, HEAD {res['head_ltsb'] or 'PLACEHOLDER'}); the stance5 runs read through "
         f"score_stance5.py's band instance ({S5.JUDGE_SCHEMA_S5}).",
         f"new runs' joint-noise sigma: {res['sidecar']['path']} (sha256 {res['sidecar']['sha256'][:16]}...; "
         f"{res['sidecar']['rule']}; mean {sum(res['sidecar']['vector']) / len(res['sidecar']['vector']):.6f})"]
    if res["validation"]:
        L.append("*** VALIDATION ONLY: relaxed-binding judge files accepted (--validation); not a replication verdict ***")
    L.append(f"new-run judge files: {res['judged_root']}   stance5 judge files: {res['stance5_judged_root']}")
    L.append("")
    L.append("STANCE5 BAND RUNS (recorded verdicts; not re-judged, not re-scored)")
    for run, *_ in STANCE5_RUNS:
        L += S5.run_block(run, res["stance5"][run])
    L += repro_lines(res["stance5_reproduction"])
    L.append("")
    L.append("NEW BAND RUNS (LTSB, branch claude/screen-trex-r9-armlts-copband at cf80ecb1; seeds 70 and 80)")
    for run, *_ in NEW_RUNS:
        L += S5.run_block(run, res["new"][run])
    L.append("")
    d = res["decision"]
    L.append("REPLICATION RULE (criteria DECISION TABLE: ADOPT only if s70 and s80 both HOLD)")
    for run, *_ in NEW_RUNS:
        r = res["new"][run]
        L.append(f"   {run}: {'HOLDS' if d['new_holds'][run] else 'does not hold (failed ' + ', '.join(r['criteria']['failed']) + ')'}"
                 + ("" if r["criteria"]["complete"] else "  [a J judge file is missing or invalid]"))
        es = r.get("early_stop") or {}
        if es.get("early"):
            L.append(f"      early stop at {es['timesteps']:,} (gate_verdict.json): the final J file is invalid (criteria "
                     "RUN HANDLING)")
        for e in r.get("refusals") or []:
            L.append(f"      judge refusal recorded, an invalid J file ({os.path.basename(e['path'])}): {e['refusal']}")
        if r.get("source_folder"):
            L.append(f"      placed from the Drive folder {r['source_folder']} (a restart before the first checkpoint)")
    L.append(f"   H = {d['H']} of {d['of']} band runs HOLD (stance5 as recorded: "
             + ", ".join(f"{run} {'holds' if h else 'does not'}" for run, h in d["stance5_holds_recorded"].items()) + ")")
    if d["status"] == "incomplete":
        L.append(f"DECISION: incomplete ({len(d['missing'])} new-run J judge files missing; the counts above treat them as "
                 "fails). The decision is read once both new runs have ended (--runs-ended).")
    else:
        L.append("DECISION" + (" (PROVISIONAL)" if d["status"] == "provisional" else "") + f": {d['text']}")
        if d["line"] in (2, 3):
            for run, t in res["targets"].items():
                if d["new_holds"][run]:
                    continue
                sat = (f"at {t['saturation_alarm_at']:,}" if t["saturation_alarm_at"] is not None else "none"
                       ) if t["saturation_alarm_computed"] else f"not computed ({t['saturation_alarm_why']})"
                gone = t["j_files_missing_or_invalid"]
                L.append(f"   next-round inputs, {run}: failed {t['failed']}; K1 failure modes "
                         f"{json.dumps(t['k1_failure_modes'], sort_keys=True)} (new modes: {fmt(t['new_k1_modes'])}); "
                         f"saturation alarm {sat}"
                         + (f"; J files missing or invalid: {', '.join(gone)} (a test that needs one of them fails on "
                            "the missing or invalid file, not on a measurement)" if gone else ""))
            L.append("   the maintainer chooses the next round (menu: stance5 criteria decision line 4's sub-items)")
    if d.get("placed_unjudged"):
        L.append("   placed but not judged (new-run J checkpoints): " + "; ".join(d["placed_unjudged"]))
    if d.get("counted_as_fails"):
        L.append(f"   runs ended (--runs-ended): {len(d['counted_as_fails'])} new-run J judge files missing or invalid, "
                 "counted as fails: " + "; ".join(d["counted_as_fails"]))
    for why in d["provisional_why"]:
        L.append(f"   provisional: {why}")
    L.append("")
    rep = res["report"]
    L.append(f"REPORT ONLY (never in the decision; control passes: {rep['control_source']})")
    fr = rep["fractions"]
    L.append(f"   K3 pass fraction, five band runs vs the control: {fr['K3']['band']} vs {fr['K3']['control']} "
             f"({'not below' if fr['K3']['band_at_least_control'] else 'below'} the control's; reported, not a veto; "
             "stance5's rule (b) is not applied)")
    L.append("   per test, band (of 5) vs control (of 3): " + "; ".join(f"{k} {fr[k]['band']} vs {fr[k]['control']}" for k in KS))
    L.append("   FORE-AFT HOLDS: " + "; ".join(f"{run} {'yes' if v else 'no'}" for run, v in rep["fore_aft"].items()))
    L.append("   drift or split (printed, not read): " + "; ".join(f"{run} {c}" for run, c in rep["drift_split_classes"].items()))
    x = res["control_crosscheck"]
    L.append(f"   control verdicts reproduce stance4 score_final.json: {'yes' if x['match'] else 'NO'}")
    p = res["parents"]
    tie = " (the maintainer chooses)"
    L.append("")
    L.append("PARENT COMPARISON (report only; stance3's rule: most clean panel episodes at final; ties to fewer joint + own "
             "falls; a tie names every tied final; the new runs' joint-noise falls use the sidecar's mean sigma)")
    if p["inputs_only"]:
        L.append("   INPUTS ONLY: the decision is incomplete; the comparison is read once both new runs have ended.")
    for key, title in (("best_band_final", "best band final of the five"),
                       ("best_holding_band_final", "best holding band final of the five (line 1's stance parent)"),
                       ("best_stance4_final", "best stance4 final")):
        b = p[key]
        L.append(f"   {title}: " + (b["best"] or ("TIE: " + ", ".join(b["tied"]) + tie if b.get("tied") else "none")))
    ref = p["reference"]
    L.append(f"   reference {ref['run']} final: clean {fmt(ref['clean'])}/40, noise falls joint {fmt(ref['joint_noise_falls'])}"
             f" / own {fmt(ref['own_noise_falls'])}")
    for c in p["comparison"]["candidates"]:
        L.append(f"     {c['run']:<34} clean {c['clean']:>2}/40  joint + own noise falls {fmt(c['noise_falls'])}")
    if not p["inputs_only"]:
        cm = p["comparison"]
        L.append("   top under the rule (band finals, stance3 Run 2's final, the best stance4 final): "
                 + (cm["top"] if cm["top"] else ("TIE: " + ", ".join(cm["top_tied"]) + tie if cm["top_tied"] else "none")))
    L.append("   walking pilot 2's parent: stance3 Run 2's final (maintainer decision (3) of 2026-10-22 15:08 UTC, run_log); "
             "not chosen here")
    if res["notes"]:
        L.append("")
        L.append("Notes:")
        for n in res["notes"]:
            L.append(f"   {n}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judged", default=R_JUDGED, help="the new runs' <RUN_ID>/judge_<slot>.json (judge_stance5r.py)")
    ap.add_argument("--runs-root", default=R_RUNS, help="the new runs' <RUN_ID>/01_stance (ledgers, placed checkpoints)")
    ap.add_argument("--head-LTSB", default=HEAD_LTSB, help="default: the pinned HEAD (any other makes the decision provisional)")
    ap.add_argument("--runs-ended", action="store_true",
                    help="both new runs have ended (criteria): missing or invalid J files count as fails")
    ap.add_argument("--validation", action="store_true", help="validation of this script only: relaxed-binding files accepted")
    ap.add_argument("--validate", action="store_true",
                    help="the three stance5 runs only: their per-run verdicts must reproduce score_final.json's")
    ap.add_argument("--ledgers", action="append", default=[], metavar="RUN_ID=DIR",
                    help="read RUN_ID's eval/train_episodes.csv from DIR[/01_stance] (report only; a restart under "
                         "<RUN_ID>_b placed by place5r.py --source-folder needs none)")
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
    res = score(a.judged, runs_root=a.runs_root, head_ltsb=a.head_LTSB, validation=a.validation,
                ledger_dirs=ledger_dirs, runs_ended=a.runs_ended, validate=a.validate)
    print(render(res))
    if a.json:
        with open(a.json, "x") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(f"\nwrote {a.json}")
    if a.validate:
        return 0 if res["stance5_reproduction"]["reproduced"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
