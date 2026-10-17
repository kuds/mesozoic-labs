"""Pick the walking pilot's parent from the r10 screen's score, by the screen's fixed rule (foot/r10screen/criteria.txt).

Plain Python, no repository import.

    python3 pilot_parent.py SCORE_JSON [--json OUT.json] [--validation]

SCORE_JSON is what foot/r10screen/judge/score_r10.py --json wrote. Its "best_stance_policy" ranks all six 6M
checkpoints whatever the verdict; that ranking is the rule only when the screen FAILS. The rule, from the screen's
criteria ("What follows"):
  K1-K4 all hold (overall PASS)  -> the pilot starts from the best r10 arm L 6M checkpoint: among the three r10
                                    checkpoints only, the most clean panel episodes; tie -> fewer noise falls
                                    (joint-space + own-sigma, summed). PATH B (claude/pilot-walk-r10).
  any of K1-K4 fails             -> the best stance policy among the six 6M checkpoints (three r10, three r9), same
                                    ranking. PATH A (claude/pilot-walk-r9, the committed r9 s60 pair) when that is r9
                                    arm L s60; PATH B when it is an r10 checkpoint.
A tie the ranking leaves (equal clean and equal noise falls) is not resolved by the fixed rule: the output names the
tied checkpoints and no parent (a maintainer decision). Nor is a tie on clean that needs the noise-fall tie-break when
any checkpoint in it has unmeasured noise falls (score_r10.py writes noise_falls None): an unmeasured count cannot win
or lose a tie-break, so the output names the tied checkpoints, says which are unmeasured, and gives no parent.
(score_r10.py's own printed ranking puts an unmeasured count last, so in that case its "best" differs from this tool.)
A checkpoint with the most clean episodes alone wins on clean; its noise falls are not read.
It prints the path, the branch, the MANUAL_LOAD_PATH line, and the parent sha256 pilot_judge.py --parent-sha256 needs
(path A: the committed file's; path B: the model_sha256 the r10 judge bound, read from that judge file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

R9_S60 = "r9 arm L s60 6M"
R9_S60_SHA256 = "sha256:70f3a4447794a01036c27bca5e0377a17db96d0e3d7bfb2d179e9ce94e6f353e"  # SOURCE.txt, r9 judge
R9_LOAD = "/content/mesozoic-labs/pilot_checkpoints/r9_arm_l_s60_6291456/stage1_6291456_steps"
R10_LOAD = ("/content/drive/MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/r10_armL_s{seed}/01_stance/"
            "models/stage1_6291456_steps")


def rank(cands: list[dict]) -> dict:
    """Most clean; tie on clean -> fewer noise falls. Returns the best, or no best with the tie and why.

    A candidate whose noise falls are unmeasured (None) cannot win a tie-break: if the checkpoints tied on clean include
    one, there is no best ("equal clean and a noise-fall count unmeasured").
    """
    if not cands:
        return {"best": None, "tied": [], "why": "no candidate", "unmeasured": []}
    unmeasured = [c["run"] for c in cands if c.get("noise_falls") is None]
    most = max(int(c["clean"]) for c in cands)
    top_clean = [c for c in cands if int(c["clean"]) == most]
    if len(top_clean) == 1:
        return {"best": top_clean[0], "tied": [], "why": f"most clean ({most}/40)", "unmeasured": unmeasured}
    tied_unmeasured = [c["run"] for c in top_clean if c.get("noise_falls") is None]
    if tied_unmeasured:
        return {"best": None, "tied": [c["run"] for c in top_clean],
                "why": (f"equal clean ({most}/40) and a noise-fall count unmeasured for "
                        f"{', '.join(tied_unmeasured)}: the tie-break cannot be applied"),
                "unmeasured": unmeasured}
    fewest = min(int(c["noise_falls"]) for c in top_clean)
    top = [c for c in top_clean if int(c["noise_falls"]) == fewest]
    if len(top) > 1:
        return {"best": None, "tied": [c["run"] for c in top],
                "why": f"equal clean ({most}/40) and equal noise falls ({fewest})", "unmeasured": unmeasured}
    return {"best": top[0], "tied": [], "why": f"most clean ({most}/40); tie on clean broken by fewer noise falls "
                                               f"({fewest})", "unmeasured": unmeasured}


def choose(score: dict) -> dict:
    cands = score.get("best_stance_policy", {}).get("candidates", [])
    passed = score.get("overall") == "PASS" and not score.get("failed") and not score.get("missing") \
        and not score.get("invalid")
    pool = [c for c in cands if c["run"].startswith("r10 ")] if passed else list(cands)
    r = rank(pool)
    out = {"screen_overall": score.get("overall"), "screen_failed": score.get("failed"),
           "rule": ("K1-K4 all hold: best r10 arm L 6M checkpoint (r10 only; most clean, tie -> fewer noise falls)"
                    if passed else "a criterion fails: best of the six 6M checkpoints (most clean, tie -> fewer noise falls)"),
           "pool": [c["run"] for c in pool], "ranking_why": r["why"], "tied": r["tied"],
           "unmeasured_noise_falls": r.get("unmeasured", []), "parent": None}
    best = r["best"]
    if best is None:
        out["decision_needed"] = ("the fixed rule leaves a tie between " + ", ".join(r["tied"]) + " (" + r["why"] +
                                  "): the maintainer decides (design.txt, decisions before launch)") if r["tied"] else \
            "no candidate checkpoint in the score"
        return out
    run = best["run"]
    if run.startswith("r9 "):
        if run != R9_S60:
            out["decision_needed"] = f"{run} would be the parent, but only r9 s60 ships on claude/pilot-walk-r9"
            return out
        out.update(parent=run, path="A", branch="claude/pilot-walk-r9", manual_load_path=R9_LOAD,
                   parent_sha256=R9_S60_SHA256, parent_sha256_source="pilot_checkpoints/.../SOURCE.txt (the r9 judge's)")
    else:
        seed = int(run.split(" s")[1].split()[0])
        sha, src = None, None
        jp = best.get("path")
        if jp and Path(jp).is_file():
            j = json.loads(Path(jp).read_text())
            sha, src = j.get("model_sha256"), f"{jp} model_sha256"
        out.update(parent=run, path="B", branch="claude/pilot-walk-r10", manual_load_path=R10_LOAD.format(seed=seed),
                   parent_sha256=sha, parent_sha256_source=src or "the r10 judge file is not readable here: read its "
                                                                  "model_sha256 before judging")
    out["candidate"] = best
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("score_json")
    ap.add_argument("--json", default=None)
    ap.add_argument("--validation", action="store_true", help="accept a validation-synthetic score (never for launch)")
    a = ap.parse_args(argv)
    score = json.loads(Path(a.score_json).read_text())
    if score.get("validation_synthetic") and not a.validation:
        print("refusing: the score is validation-synthetic (score_r10.py --validation-synthetic)")
        return 2
    res = choose(score)
    print(f"Screen: {res['screen_overall']} (failed: {res['screen_failed']}); rule: {res['rule']}")
    print(f"Pool: {res['pool']}")
    if res["parent"]:
        print(f"PARENT: {res['parent']} -> PATH {res['path']}, branch {res['branch']} ({res['ranking_why']})")
        print(f'  MANUAL_LOAD_PATH = "{res["manual_load_path"]}"')
        print(f"  pilot_judge.py --parent-sha256 {res['parent_sha256']}   [{res['parent_sha256_source']}]")
    else:
        print(f"NO PARENT: {res.get('decision_needed')}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    return 0 if res["parent"] else 1


if __name__ == "__main__":
    sys.exit(main())
