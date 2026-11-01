"""PC dry run (report only): the PC scorer (pc/score_pc.py) and the adapted completeness tool on the stance6 round's own
F2 data as a STAND-IN, read only where it lies (S1/foot/stance6: fetch/F2, fetch/manifest.csv, panels/, APPROVED,
criteria.txt; RUNLOG S1/foot/stance4/live/run_log.txt).  Stand-in substitutions (printed; nothing else changed):
  - the RUN_IDs, the RUNLOG kind and the token pattern are STANCE6's (stance6_r10LTSB_s60/42/80, "STANCE6");
  - the pins are the round's (pins_r10round.json) with critic_phase_switch {"path": "hyperparameters.critic_phase",
    "value": null} (a switch-OFF run records no key: C4 (p) and fact (8) then read the absence as the pin) and the
    comparator pins of pins_pc.json;
  - C8: a stock-critic run has no manipulation check (it would read NOT IN FORCE); where C1-C7 all pass and C8 is
    absent, the stand-in substitutes PASS and says so (C8's own code path is tested on synthetic roots and on the
    smoke checkpoint, not here).
Expected [measured, S2/r10round/f2/score.json]: every class key equal to the round's scorer's; THE READING line 1
(the round's s42 ledger rail).  Writes only --out (the report and a JSON).

    PY -I -B standin_s6.py --out DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

T = Path(__file__).resolve().parent
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
S2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad")


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, T / "pc" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CLASS_KEYS = ("cls", "kind", "tail_alarms", "resolved_by_trace", "tail_alarm_detail", "flagged_evaluations",
              "trace_rail", "single_tail_flags", "parked", "driven", "mean", "mean_basis", "cmd", "cmd_episodes",
              "cmd_others_ended_by_200", "gate", "gate_why", "outcome", "reasons", "any_alarm", "any_alarms", "stopped",
              "E", "fall_flags", "open")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    C = load("r10round_common")
    # ---- stand-in constants (STANCE6's names)
    C.RUN_ID_PREFIX = "stance6_r10LTSB_s"
    C.RUN_IDS.clear()
    C.RUN_IDS.update({f"stance6_r10LTSB_s{s}": s for s in C.SEEDS})
    C.NOTE_KIND = "STANCE6"
    C.NOTE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?) UTC\s+STANCE6\s+(\S+)\s+(\S+)(?:\s+(.*))?$")
    C.RUN_TOKEN_RE = re.compile(r"stance6_r10LTSB_s\d+\w*")
    CP = load("complete_r10round")
    S = load("score_pc")
    orig_completeness = CP.completeness

    def standin_completeness(run_id, store, notes, pins, panels):
        comp = orig_completeness(run_id, store, notes, pins, panels)
        pre = [comp[k]["ok"] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")]
        if all(x is True for x in pre) and comp["C8"]["ok"] is None:
            comp["C8"] = CP.fact("C8 (STAND-IN: a stock-critic STANCE6 run has no manipulation check; substituted PASS "
                                 "for the stand-in read only)", True, comp["C8"]["value"], "stand-in substitution")
            comp["verdict"] = "COMPLETE"
        return comp

    CP.completeness = standin_completeness
    pins = json.loads((T / "orig/laneB/pins_r10round.json").read_text())
    pc_pins = json.loads((T / "pc/pins_pc.json").read_text())
    pins["critic_phase_switch"] = {"path": "hyperparameters.critic_phase", "value": None}
    pins["critic_phase_spec"] = "stand-in: none"
    pins["comparators"] = pc_pins["comparators"]
    pins_path = out_dir / "standin_pins.json"
    pins_path.write_text(json.dumps(pins, indent=1) + "\n")
    out, status = S.score(round_dir=S1 / "foot/stance6", runlog=S1 / "foot/stance4/live/run_log.txt",
                          pins_path=pins_path, approved_path=S1 / "foot/stance6/APPROVED")
    text = S.render(out)
    # the round's own classes, key by key
    ref = json.loads((S2 / "r10round/f2/score.json").read_text())
    ref_seed = {d["seed"]: d for d in ref["per_seed"]}
    lines = ["STAND-IN CLASS COMPARISON (PC scorer vs the round's score.json, cb380a7e...):"]
    all_equal = True
    for x in out["per_seed"]:
        mine = json.loads(json.dumps(S.SR._jsonable(x.get("classes") or {}), default=str))
        theirs = (ref_seed.get(x["seed"]) or {}).get("classes") or {}
        diff = [k for k in CLASS_KEYS if mine.get(k) != theirs.get(k)]
        all_equal &= not diff and bool(mine)
        lines.append(f"  s{x['seed']} {x['run']}: {len(CLASS_KEYS) - len(diff)} of {len(CLASS_KEYS)} class keys equal"
                     + (f"; differ: {diff}" if diff else ""))
    lines.append(f"  ALL EQUAL: {all_equal}")
    td = [(x["seed"], (x.get("td") or {}).get("td")) for x in out["per_seed"]]
    comp_td = [(s, out["comparators"]["runs"][s]["td"]["td"]) for s in (60, 42, 80)]
    lines.append(f"  TD from the scorer's C3-verified ledgers {td}; from the comparators' FMAN-checked copies {comp_td}; "
                 f"equal {td == comp_td}")
    report = text + "\n" + "\n".join(lines) + f"\nEXIT STATUS {status}\n"
    (out_dir / "standin_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "standin_score.json").write_text(json.dumps(S.SR._jsonable(out), indent=1, default=str) + "\n")
    print(report)
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
