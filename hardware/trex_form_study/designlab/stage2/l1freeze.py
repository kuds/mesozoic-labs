"""Gate L1's outcome, frozen by hash (``STAGE2_PLAN.md`` section 4 A.5; amendment S2-15; ``DECISION_LOG.md`` P30).

On 14 Nov 2026 the owner picked the look ladder's V6 (every look feature) as the reference Stage 2 freezes, allowed a
look feature to cost a speed rung and widened the duty band to 0.55-0.72 below 0.30 m/s.  This module writes that
outcome as a record, ``stage2/look/L1_FREEZE.json``: the sha256 of every V6 rung of the committed ladder
(``stage2/look/look_refs.json``) with what the committed Stage 0 screen (``stage2/look/screen.json``) says of it, and
which rungs stay in the ladder under the plan's rule (4 A.3: over a drive limit a variant keeps its features and loses
that speed rung).  It computes nothing new: every number is read from the two committed records, and the record
regenerates from them byte for byte (``tests/test_stage2_l1freeze.py``).

The rungs are the look ladder's periodic walks in the look format (no feedforward torque).  The Stage 1-format
reference set built from the same specs (walks at more speeds, starts, changes, stops, turns) is a set of
``designlab.stage2.sets`` and is frozen in its own ``set.json``.

    python -m designlab.stage2.l1freeze write      # rewrite stage2/look/L1_FREEZE.json from the committed records
    python -m designlab.stage2.l1freeze check      # exit 1 if the record differs, or an array on disk has another sha256
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
LOOK = STUDY / "stage2" / "look"
RECORD = LOOK / "L1_FREEZE.json"
VARIANT = "V6"
#: The owner's decisions of 14 Nov 2026 that this record carries (DECISION_LOG.md P30; STAGE2_PLAN.md S2-15).
DECIDED = {
    "by": "owner",
    "date": "2026-11-14",
    "decision_log": "P30",
    "pick": VARIANT,
    "pick_text": "V6, every look feature (the look review page's proposal)",
    "feature_may_cost_a_speed_rung": True,
    "duty_band": {"below_0.30_m_s": [0.55, 0.72], "from_0.30_m_s": [0.55, 0.65]},
    "viewers": ("the owner's pick decided L1; the scores of 3-5 viewers new to the project are added later as a "
                "labelled late reading (a change to plan A.5's order, taken by the owner before any Stage 2 run)"),
    "candidate": ("the tuned walk with the counter-phased tail and the duty and lift margin (set F25F9cl of "
                  "STAGE2_75_PLAN.md) is a labelled candidate for the 75 cm body; it is not frozen here"),
}
RULE = ("a rung is in the ladder if the committed Stage 0 screen has it within every limit (screen.json rungs[body]"
        "[speed][V6].within_limits); a rung over a drive limit keeps its features and loses that speed rung (plan 4 "
        "A.3; the owner's decision of 14 Nov that a look feature may cost a speed rung)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    refs = json.loads((LOOK / "look_refs.json").read_text())
    screen = json.loads((LOOK / "screen.json").read_text())
    rungs = {}
    for body in sorted(refs["bodies"]):
        rungs[body] = {}
        for speed in refs["speeds"]:
            e = refs["bodies"][body][speed][VARIANT]
            s = screen["rungs"][body][speed][VARIANT]
            rungs[body][speed] = {
                "variant": VARIANT,
                "features": list(e["features"]),
                "file": f"stage2/look/{e['file']}",
                "sha256": e["sha256"],
                "speed_m_s": float(e["info"]["speed"]),
                "duty": float(e["info"]["duty"]),
                "strides_in_window": int(e["info"].get("strides", 1)),
                "governing": list(s["governing"]),
                "over": list(s["over"]),
                "within_limits": bool(s["within_limits"]),
                "in_ladder": bool(s["within_limits"]),
                "screen_rung": s["rung"],
                "flags": list(e["flags"]),
            }
    return {
        "what": "gate L1's outcome, frozen by hash: the look ladder's V6 rungs as committed (designlab.stage2.l1freeze)",
        "simulation_only": True,
        "decided": DECIDED,
        "rule": RULE,
        "sources": {
            "look_refs": {"file": "stage2/look/look_refs.json", "sha256": _sha256(LOOK / "look_refs.json")},
            "screen": {"file": "stage2/look/screen.json", "sha256": _sha256(LOOK / "screen.json")},
        },
        "rungs": rungs,
        "not_in_ladder": [f"{b} {sp}" for b in rungs for sp in rungs[b] if not rungs[b][sp]["in_ladder"]],
    }


def text(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def check(arrays: bool = True) -> list[str]:
    """Problems found: the record against the committed sources, and each array on disk against its sha256 (arrays
    are git-ignored; a missing one is reported as missing, not as wrong)."""
    out = []
    doc = build()
    if not RECORD.exists():
        return [f"{RECORD.relative_to(STUDY)} is missing"]
    if RECORD.read_text() != text(doc):
        out.append(f"{RECORD.relative_to(STUDY)} differs from what the committed look records give")
    if arrays:
        for body, speeds in doc["rungs"].items():
            for speed, r in speeds.items():
                p = STUDY / r["file"]
                if not p.exists():
                    out.append(f"missing array {r['file']} (restore it: tools/restore_inputs.py)")
                elif _sha256(p) != r["sha256"]:
                    out.append(f"{r['file']} has another sha256 than the frozen one")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["write"]:
        RECORD.write_text(text(build()))
        print(f"wrote {RECORD.relative_to(STUDY)}")
        return 0
    if argv == ["check"]:
        problems = check()
        for p in problems:
            print(p)
        print("L1 freeze: " + ("ok" if not problems else f"{len(problems)} problem(s)"))
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
