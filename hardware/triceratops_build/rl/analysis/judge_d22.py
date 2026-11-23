"""The foreleg comparison's rules (PREREG_FORELEG_D22.md, decisions D22 and D23), read from the tools' JSON for the
two lengths, two seeds each. Simulation only.

Per run (length x seed): provenance first (the model and trike_env.py sha256 as each tool recorded them against the
run commit's files; the run commit; the run's final pair; the same pair in all three tools), then run 3's rules
F1, F2 and T1-T5 (`judge_run3.judge_seed`, unchanged), the foot load L (`foot_loads.py`: the mean over 5 episodes
of the worst foot's 99th percentile of the 10 ms mean normal force), the heat H (F2's worst joint RMS torque over
its continuous rating) and the falls N (at run 3's judged sweep points and in the gait report's four cases).

The choice (D23), in this order:
1. A length "walks" when F1 and F2 pass on both its seeds. If exactly one length walks, it is chosen. If neither,
   no length is chosen.
2. Otherwise 128 mm, unless 160 mm beats it on all three:
   - loads: the larger of 160 mm's two L is at most 0.85 x the smaller of 128 mm's two L;
   - falls: 160 mm's N summed over its seeds is at most 128 mm's;
   - heat: 160 mm's H averaged over its seeds is at most 128 mm's average + 0.05.
A run without its tools' output, or failing provenance, is NOT JUDGED, and then no choice is made.

Usage:
  python hardware/triceratops_build/rl/analysis/judge_d22.py --run-commit SHA \\
      --run 128 0 GAIT SWEEP FOOT --run 128 1 ... --run 160 0 ... --run 160 1 ... [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_run3 as j3  # noqa: E402

LENGTHS = (128, 160)
SEEDS = (0, 1)
LOAD_MARGIN, HEAT_TOLERANCE = 0.85, 0.05
JUDGED_POINTS = (j3.STEP[2], j3.STEP[3], j3.RAMP[5], j3.RAMP[8], j3.FRICTION_03, *j3.PUSH.values())


def model_of(length: int) -> str:
    return f"trike60_C_closed_capsule_fore{length}.xml"


def provenance(length: int, run_commit: str, gait: dict, sweep: dict, foot: dict) -> list[str]:
    bad = []
    model = model_of(length)
    j3.RUN_COMMIT = run_commit  # judge_run3's sha helpers read the run commit from here
    model_sha = j3._blob_sha256(f"hardware/triceratops_build/sim/generated/{model}")
    env_sha = j3._blob_sha256("hardware/triceratops_build/rl/trike_env.py")
    if gait["environment"]["model_sha256"].get(model) != model_sha:
        bad.append("gait report: model sha256 differs from the run commit's")
    if sweep["env"]["model"] != model or sweep["env"]["model_sha256"] != model_sha:
        bad.append("sweep: model differs from the run commit's")
    fp = foot["provenance"]
    if fp["model"] != model or fp["model_sha256"] != model_sha:
        bad.append("foot loads: model differs from the run commit's")
    if {gait["environment"]["trike_env_sha256"], sweep["env"]["trike_env_sha256"], fp["trike_env_sha256"]} != {env_sha}:
        bad.append("trike_env.py differs from the run commit's")
    if {gait["policy"]["run_commit"], sweep["run_commit"], fp["run_commit"]} != {run_commit}:
        bad.append("run commit differs")
    if not j3.TOTAL_STEPS <= gait["policy"]["steps"] < j3.TOTAL_STEPS + j3.BLOCK:
        bad.append(f"checkpoint at {gait['policy']['steps']} steps is not the run's final pair")
    pair = (gait["policy"]["zip_sha256"], gait["policy"]["vecnormalize_sha256"])
    if (sweep["checkpoint"]["zip_sha256"], sweep["checkpoint"]["vecnormalize_sha256"]) != pair or \
            (fp["zip_sha256"], fp["vecnormalize_sha256"]) != pair:
        bad.append("the tools measured different checkpoints")
    return bad


def measure(gait: dict, sweep: dict, foot: dict) -> dict:
    rules = j3.judge_seed(gait, sweep)
    falls = sum(sweep["points"][k]["summary"]["falls"] for k in JUDGED_POINTS) + sum(
        c["summary"].get("falls", 0) for c in gait["cases"].values())
    return {"rules": {k: rules[k]["pass"] for k in j3.RULES}, "rule_values": rules,
            "load_n": foot.get("worst_foot_p99_window_n"), "heat": rules["F2"]["worst"], "falls": falls,
            "foot_falls": foot.get("falls")}


def choose(runs: dict) -> dict:
    """``runs[(length, seed)]``: a measure() dict, or None when not judged."""
    if any(runs.get((n, s)) is None for n in LENGTHS for s in SEEDS):
        return {"choice": None, "why": "not judged: a run lacks its tools' output or fails provenance"}
    walks = {n: all(runs[(n, s)]["rules"]["F1"] and runs[(n, s)]["rules"]["F2"] for s in SEEDS) for n in LENGTHS}
    if walks[128] != walks[160]:
        n = 128 if walks[128] else 160
        return {"choice": n, "why": f"only {n} mm passes F1 and F2 on both seeds", "walks": walks}
    if not any(walks.values()):
        return {"choice": None, "why": "neither length passes F1 and F2 on both seeds", "walks": walks}
    L = {n: [runs[(n, s)]["load_n"] for s in SEEDS] for n in LENGTHS}
    if any(v is None for n in LENGTHS for v in L[n]):
        return {"choice": None, "why": "a foot-load figure is missing (a fall in the foot-load episodes)", "walks": walks}
    loads = max(L[160]) <= LOAD_MARGIN * min(L[128])
    falls = sum(runs[(160, s)]["falls"] for s in SEEDS) <= sum(runs[(128, s)]["falls"] for s in SEEDS)
    H = {n: sum(runs[(n, s)]["heat"] for s in SEEDS) / len(SEEDS) for n in LENGTHS}
    heat = H[160] <= H[128] + HEAT_TOLERANCE
    tests = {"loads": loads, "falls": falls, "heat": heat, "L": L, "H_mean": H,
             "N": {n: [runs[(n, s)]["falls"] for s in SEEDS] for n in LENGTHS}}
    if loads and falls and heat:
        return {"choice": 160, "why": "160 mm beats the 128 mm default on loads, falls and heat", "walks": walks,
                "tests": tests}
    return {"choice": 128, "why": "the default: 160 mm does not beat it on " +
            ", ".join(k for k in ("loads", "falls", "heat") if not tests[k]), "walks": walks, "tests": tests}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-commit", required=True)
    ap.add_argument("--run", nargs=5, action="append", required=True,
                    metavar=("LENGTH", "SEED", "GAIT_JSON", "SWEEP_JSON", "FOOT_JSON"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs, problems = {}, {}
    for length, seed, g, w, f in a.run:
        key = (int(length), int(seed))
        gait, sweep, foot = j3._load(g), j3._load(w), json.loads(Path(f).read_text())
        problems[key] = provenance(key[0], a.run_commit, gait, sweep, foot)
        runs[key] = None if problems[key] else measure(gait, sweep, foot)
    res = choose(runs)
    for key, p in sorted(problems.items()):
        r = runs[key]
        print(f"{key[0]} mm seed {key[1]}: " + ("PROVENANCE PROBLEM: " + "; ".join(p) if p else
              f"rules {''.join('P' if v else 'F' for v in r['rules'].values())} (F1 F2 T1-T5), L {r['load_n']:.1f} N, "
              f"H {r['heat']:.3f}, N {r['falls']}"))
    print("CHOICE:", res["choice"], "-", res["why"])
    out = {"simulation_only": True, "prereg": "hardware/triceratops_build/rl/analysis/PREREG_FORELEG_D22.md",
           "run_commit": a.run_commit, "provenance_problems": {f"{k[0]}_s{k[1]}": v for k, v in problems.items()},
           "runs": {f"{k[0]}_s{k[1]}": v for k, v in runs.items()}, "choice": res}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
