"""Validate pilot3_habit.py's ledger reading on the PC round's six stance ledgers (criteria_pilot3.txt, pre-launch
checks): its alarms must equal the PC round's recorded readings. Exit 1 on any difference (the freeze then stops).

    python3 -I -B tests/validate_pc_ledgers.py FETCH_DIR

FETCH_DIR holds stance7_r10LTSB{off,pc}_s{42,60,80}/01_stance/eval_episodes.csv. Read with stance's grid (every
evaluation of the 12M run, 131,072 apart; no window end).
Recorded readings [measured as recorded: handoff PC_REPORT.md per-run table "Alarms: ANY / TAIL (tail-flagged evals)",
quoting score_pc.txt lines 321-346; NEXT_STEPS.md lines 102 and 146]:
  PHASE s60 0/0 (0); OFF s60 0/0 (0); PHASE s42 0/0 (0); OFF s42 22/22 (26), first 3,932,160; PHASE s80 0/0 (0);
  OFF s80 6 (neck_pitch, 5,505,024-7,077,888) / 0 (0). Statistic A (runs with an ANY ALARM): PHASE 0 of 3, OFF 2 of 3.
  "Habit elsewhere" FREE in the 3 PHASE runs (no NECK or LEGS alarm). TD (91 evaluations): PHASE s60 0.0079, OFF s60
  0.3882, PHASE s42 0.0173, OFF s42 0.6353, PHASE s80 0.0122, OFF s80 0.1543 (report only here; checked too).
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot3_habit as ph  # noqa: E402

EXPECTED = {  # run: (ANY alarms, TAIL alarms, tail-flagged evaluations, NECK alarms, first/last alarm, TD)
    "stance7_r10LTSBpc_s60": dict(any=0, tail=0, tail_flagged=0, neck=0, legs=0, span=None, td=0.0079),
    "stance7_r10LTSBoff_s60": dict(any=0, tail=0, tail_flagged=0, neck=0, legs=0, span=None, td=0.3882),
    "stance7_r10LTSBpc_s42": dict(any=0, tail=0, tail_flagged=0, neck=0, legs=0, span=None, td=0.0173),
    "stance7_r10LTSBoff_s42": dict(any=22, tail=22, tail_flagged=26, neck=0, legs=0, span=("TAIL", 3_932_160, None),
                                   td=0.6353),
    "stance7_r10LTSBpc_s80": dict(any=0, tail=0, tail_flagged=0, neck=0, legs=0, span=None, td=0.0122),
    "stance7_r10LTSBoff_s80": dict(any=6, tail=0, tail_flagged=0, neck=6, legs=0, span=("NECK", 5_505_024, 7_077_888),
                                   td=0.1543),
}


def main(argv: list[str]) -> int:
    fetch = Path(argv[1]) if len(argv) > 1 else None
    if fetch is None or not fetch.is_dir():
        print("usage: validate_pc_ledgers.py FETCH_DIR")
        return 2
    mismatches, with_any = [], {"pc": 0, "off": 0}
    for run, want in EXPECTED.items():
        path = fetch / run / "01_stance" / "eval_episodes.csv"
        header, rows = ph.read_ledger(path)
        lr = ph.ledger_reading(header, rows, None)
        g = lr["groups"]
        got = dict(any=len(g["ANY"]["alarms"]), tail=len(g["TAIL"]["alarms"]), tail_flagged=len(g["TAIL"]["flagged"]),
                   neck=len(g["NECK"]["alarms"]), legs=len(g["LEGS"]["alarms"]), td=round(lr["td"]["overall"], 4))
        checks = [(k, got[k], want[k]) for k in ("any", "tail", "tail_flagged", "neck", "legs", "td")]
        checks.append(("evaluations", lr["window"]["evaluations"], 91))
        checks.append(("possible alarms TAIL/NECK", (g["TAIL"]["possible_alarms"], g["NECK"]["possible_alarms"]), ([], [])))
        checks.append(("superseded / incomplete", (lr["superseded_row_sets"], lr["incomplete_evaluations"]), ({}, [])))
        if want["span"]:
            grp, first, last = want["span"]
            al = g[grp]["alarms"]
            checks.append((f"{grp} first alarm", al[0] if al else None, first))
            if last is not None:
                checks.append((f"{grp} last alarm", al[-1] if al else None, last))
        if run.endswith("s80") and "off" in run:
            named = sorted({n for e in lr["evaluations"] if e["step"] in g["NECK"]["alarms"] for n in e["named_over"]})
            checks.append(("NECK alarm actuator", named, ["neck_pitch_act"]))
        bad = [(k, a, b) for k, a, b in checks if a != b]
        mismatches += [(run, *x) for x in bad]
        arm = "pc" if "LTSBpc" in run else "off"
        with_any[arm] += int(got["any"] > 0)
        print(f"{'ok  ' if not bad else 'FAIL'} {run}: ANY {got['any']} / TAIL {got['tail']} ({got['tail_flagged']} "
              f"tail-flagged) / NECK {got['neck']} / LEGS {got['legs']}; TD {got['td']:.4f}"
              + (f"; TAIL alarms {g['TAIL']['alarms'][0]:,}-{g['TAIL']['alarms'][-1]:,}" if g["TAIL"]["alarms"] else "")
              + (f"; NECK alarms {g['NECK']['alarms'][0]:,}-{g['NECK']['alarms'][-1]:,}" if g["NECK"]["alarms"] else "")
              + (f"  MISMATCH {bad}" if bad else ""))
    stat_a = (with_any["pc"], with_any["off"])
    print(f"{'ok  ' if stat_a == (0, 2) else 'FAIL'} Statistic A, runs with an ANY ALARM: PHASE {stat_a[0]} of 3, "
          f"OFF {stat_a[1]} of 3 (recorded: PHASE 0 of 3, OFF 2 of 3)")
    if stat_a != (0, 2):
        mismatches.append(("Statistic A", stat_a, (0, 2)))
    print(f"{len(mismatches)} mismatch(es)" + (f": {mismatches}" if mismatches else ""))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
