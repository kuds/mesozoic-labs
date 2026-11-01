"""Second re-review: synthetic roots for the maintainer's "Phase first" waves (RUNLOG line 876) through the PC scorer
(ST/tools/pc/score_pc.py via ST/tools/tests/synth_pc.py).  Wave 1: the three PHASE runs at T0, T0+2, T0+4 min, ended
15 h later with TRIGGER-a / FOUND-GONE notes "established from metadata only (outcome blindness)" (the fixed text's
form).  Wave 2: the three OFF runs as wave 1's runtimes free (s60 +15:05, s42 +15:10, s80 +15:15).  F2 for every
launched RUN_ID at one time, 7 h after the last end.  PHASE TDs 0.05/0.06/0.07, OFF 0.5/0.4/0.3.
  A  nothing opened in the blind period                                     -> every seed blind, k = 1 (LOWER)
  B  a PHASE SNAPSHOT (an opening in the blind period) between OFF s42 and s80 -> OFF s80 NOT BLIND, k = 11
  C  a maintainer's report-only SNAPSHOT 20 min after the last OFF LAUNCH (the text allows it then), and OFF s80's
     first run faulted (8) and relaunched as _b 35 min after its LAUNCH (REPLACEMENT (ii))  -> the _b NOT BLIND, k = 11
  D  a SNAPSHOT line naming no RUN_ID between OFF s42 and s80               -> no opening, every seed blind, k = 1
  E  as A, but OFF s42 has no F2 store (an open item of the OFF arm, I4)    -> INCOMPLETE (exit 10), no reading
Report only; roots under --work (deleted after).      PY -I -B root_waves.py --work DIR --out FILE"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, "/home/user/work/pc_study/tools/tests")
import synth_pc as S  # noqa: E402

C, SP = S.C, S.SP
META = dict(note="established from metadata only (outcome blindness)")
PC_TD = {60: 0.05, 42: 0.06, 80: 0.07}
OFF_TD = {60: 0.5, 42: 0.4, 80: 0.3}
T0 = S.T0
PH_LAUNCH = {60: T0, 42: T0 + timedelta(minutes=2), 80: T0 + timedelta(minutes=4)}
OFF_LAUNCH = {60: T0 + timedelta(hours=15, minutes=5), 42: T0 + timedelta(hours=15, minutes=10),
              80: T0 + timedelta(hours=15, minutes=15)}


def build(root: Path, variant: str):
    rnd = S.Round(root)
    runs = {}
    ends = []
    for s in (60, 42, 80):
        run, _d = S.standard_run(rnd, s, launch=PH_LAUNCH[s], ledger=S.ledger_with_cmd(PC_TD[s]), ended_note=False,
                                 f2=False)
        end = PH_LAUNCH[s] + timedelta(hours=15)
        rnd.add_note(end, "TRIGGER-a", run.run_id, **META)
        rnd.add_note(end, "FOUND-GONE", run.run_id, **META)
        runs[("pc", s)] = run
        ends.append(end)
    for s in (60, 42, 80):
        if variant == "C" and s == 80:
            continue
        run, _d = S.standard_run(rnd, s, arm="off", launch=OFF_LAUNCH[s], ledger=S.ledger_with_cmd(OFF_TD[s]),
                                 f2=False)
        runs[("off", s)] = run
        ends.append(OFF_LAUNCH[s] + timedelta(hours=15))
    extra = []
    if variant == "C":
        first = f"{C.ARM_PREFIX['off']}80"
        t = OFF_LAUNCH[80]
        cut = C.GRID_STEPS[3]
        run0 = S.Run(first, t, rnd.pins, sc_over={"hyperparameters.critic_phase": True},
                     ledger=S.ledger_with_cmd(OFF_TD[80], E=cut), cut_at=cut)
        rnd.add_note(t, "LAUNCH", first, **S.launch_fields(80, rnd.pins["branch_head"], "off"))
        S.write_store(rnd.root, "F0", run0, run0.files(), t + timedelta(minutes=25))
        rnd.add_note(t + timedelta(minutes=20), "SNAPSHOT", runs[("pc", 60)].run_id,
                     note="report-only look at the maintainer's request after every OFF LAUNCH line")
        rnd.add_note(t + timedelta(minutes=28), "STOP", first, fault=8)
        rnd.add_note(t + timedelta(minutes=28), "TRIGGER-d", first)
        rnd.add_note(t + timedelta(minutes=32), "DELETED", first)
        b = first + "_b"
        tb = t + timedelta(minutes=35)
        run_b = S.Run(b, tb, rnd.pins, ledger=S.ledger_with_cmd(OFF_TD[80]))
        rnd.add_note(tb, "RELAUNCH", b, replaces=first, **S.launch_fields(80, rnd.pins["branch_head"], "off",
                                                                          RUN_ID=b))
        rnd.add_note(tb + timedelta(hours=15), "TRIGGER-a", b)
        rnd.add_note(tb + timedelta(hours=15), "FOUND-GONE", b)
        S.write_panels(rnd.root, run_b)
        runs[("off", 80)] = run_b
        extra.append(run0)
        ends.append(tb + timedelta(hours=15))
    if variant == "B":
        rnd.add_note(T0 + timedelta(hours=15, minutes=12), "SNAPSHOT", runs[("pc", 42)].run_id,
                     note="downloaded notebook read in the blind period")
    if variant == "D":
        rnd.add_note(T0 + timedelta(hours=15, minutes=12), "SNAPSHOT", "phase-runs", note="a look at all three")
    f2t = max(ends) + timedelta(hours=7)
    for r in [*runs.values(), *extra]:
        if variant == "E" and r is runs[("off", 42)]:
            continue
        S.write_store(rnd.root, "F2", r, r.files(), f2t)
        rnd.add_note(f2t, "F2-START", r.run_id)
    out, status = rnd.score()
    return rnd, out, status


def summary(variant, rnd, out, status):
    bl = out["blindness"]
    ot = (out["reading"] or {}).get("one_test") or {k: None for k in ("k", "text", "off_missing",
                                                                     "not_blind_entered_as_missing")}
    lines = [ln for ln in rnd.runlog.read_text().splitlines() if "SNAPSHOT" in ln or "RELAUNCH" in ln]
    return {
        "variant": variant, "exit_status": status, "incomplete": out["incomplete"],
        "first_opening": bl.get("first_opening"), "not_blind": bl["not_blind"],
        "blind_by_seed": {s: bl["seeds"][s]["blind"] for s in bl["seeds"]},
        "off_run_by_seed": {s: bl["seeds"][s].get("off_run") for s in bl["seeds"]},
        "one_test_k": ot["k"], "one_test_text": ot["text"], "off_missing": ot["off_missing"],
        "not_blind_entered_as_missing": ot["not_blind_entered_as_missing"],
        "phase_td": ot.get("phase"), "off_td": ot.get("off"),
        "reading_line": (out["reading"] or {}).get("line"),
        "snapshot_and_relaunch_lines": lines,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    if work.exists():
        print(f"REFUSED: {work} exists")
        return 2
    expect = {"A": ([], 1), "B": ([f"{C.ARM_PREFIX['off']}80"], 11), "C": ([f"{C.ARM_PREFIX['off']}80_b"], 11),
              "D": ([], 1), "E": None}
    res, ok_all = [], True
    for v in "ABCDE":
        rnd, out, status = build(work / v, v)
        sm = summary(v, rnd, out, status)
        if v == "D":
            sm["text_has_other_line"] = "SNAPSHOT phase-runs" in SP.render(out)
        if expect[v] is None:
            ok = status == SP.EXIT_INCOMPLETE and any("LTSBoff_s42" in str(x) for x in sm["incomplete"])
        else:
            ok = (sm["not_blind"] == expect[v][0] and sm["one_test_k"] == expect[v][1]
                  and sm["incomplete"] == [])
        sm["AS_EXPECTED"] = ok
        ok_all &= ok
        res.append(sm)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    for sm in res:
        print(sm["variant"], "AS EXPECTED" if sm["AS_EXPECTED"] else "NOT AS EXPECTED", "k", sm["one_test_k"],
              "not_blind", sm["not_blind"], "incomplete", sm["incomplete"], "status", sm["exit_status"])
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
