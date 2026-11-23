"""Run 3's rules (``PREREG_RUN3.md``), read from the two tools' JSON for each seed. Simulation only.

Each rule is judged per seed on the values as stored (no rounding), then across seeds: PASS when both seeds pass,
FAIL when neither does, MIXED when one does, NOT JUDGED for a seed without its tools' output (and then no rule can
pass). Before judging, the checks the pre-registration asks for, as far as the tools' JSON allows: the model's and trike_env.py's sha256 as
the tools recorded them against the run commit's files, the gait report's checkpoint in the run's final block of
steps, the same checkpoint pair in both tools, and the run commit. It does not read the run records (sessions,
clean tree, seeds, the last checkpoint): those are checked by hand and reported.

Usage:
  python hardware/triceratops_build/rl/analysis/judge_run3.py --seed s0 GAIT_JSON SWEEP_JSON[.gz] \\
      [--seed s1 GAIT_JSON SWEEP_JSON[.gz]] [--out results.json]
Writes nothing unless --out; prints a table.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RUN_COMMIT = "efd1e1c7d391062c96880b539f76bbc99d30b48c"
MODEL = "trike60_C_closed_capsule.xml"
REPO = Path(__file__).resolve().parents[4]
TOTAL_STEPS = 10_000_000
BLOCK = 8 * 1024  # steps per PPO block: the final pair lands in [TOTAL_STEPS, TOTAL_STEPS + BLOCK)

# sweep point keys (the sweep's own key strings: command, delay, friction, mass, payload, force, kp, push, terrain,
# randomise)
NOMINAL = "[0.25, 10, null, 1.0, 0.0, 1.0, 1.0, 0.0, null, false]"


def key(friction="null", push=0.0, terrain="null"):
    return f"[0.25, 10, {friction}, 1.0, 0.0, 1.0, 1.0, {push}, {terrain}, false]"


STEP = {2: key(terrain='["step", 0.02]'), 3: key(terrain='["step", 0.03]')}
RAMP = {5: key(terrain='["ramp", 5.0]'), 8: key(terrain='["ramp", 8.0]')}
FRICTION_03 = key(friction="0.3")
PUSH = {dv: key(push=dv) for dv in (-1.25, -1.0, 1.0, 1.25)}


def _blob_sha256(path_in_repo: str) -> str:
    data = subprocess.run(["git", "-C", str(REPO), "show", f"{RUN_COMMIT}:{path_in_repo}"], capture_output=True,
                          check=True).stdout
    return hashlib.sha256(data).hexdigest()


def _load(path: str) -> dict:
    p = Path(path)
    return json.load(gzip.open(p, "rt") if p.suffix == ".gz" else p.open())


def provenance(gait: dict, sweep: dict) -> list[str]:
    """Problems that stop a seed being judged (empty: fine)."""
    bad = []
    model_sha = _blob_sha256(f"hardware/triceratops_build/sim/generated/{MODEL}")
    env_sha = _blob_sha256("hardware/triceratops_build/rl/trike_env.py")
    if gait["environment"]["model_sha256"].get(MODEL) != model_sha:
        bad.append("gait report: model sha256 differs from the run commit's")
    if sweep["env"]["model_sha256"] != model_sha or sweep["env"]["model"] != MODEL:
        bad.append("sweep: model differs from the run commit's")
    if gait["environment"]["trike_env_sha256"] != env_sha or sweep["env"]["trike_env_sha256"] != env_sha:
        bad.append("trike_env.py differs from the run commit's")
    if gait["policy"]["run_commit"] != RUN_COMMIT or sweep["run_commit"] != RUN_COMMIT:
        bad.append("run commit is not run 3's")
    steps = gait["policy"]["steps"]
    if not TOTAL_STEPS <= steps < TOTAL_STEPS + BLOCK:
        bad.append(f"checkpoint at {steps} steps is not the run's final pair")
    ck = sweep["checkpoint"]  # the sweep records names and sha256 of the pair; the gait report under "policy"
    if (ck["zip_sha256"], ck["vecnormalize_sha256"]) != (gait["policy"]["zip_sha256"],
                                                         gait["policy"]["vecnormalize_sha256"]):
        bad.append("the two tools measured different checkpoints")
    return bad


def judge_seed(gait: dict, sweep: dict) -> dict:
    """The rules on one seed. The tools leave out a point's speed and drift when every episode fell, and the gait
    report its speed and drive figures when no target episode could be analysed (all shorter than 3.5 s): each such
    rule fails (PREREG_RUN3.md, "a point where every episode fell has neither and fails"; F2's added rule)."""
    t = gait["cases"]["target_0.25"]["summary"]
    pts = sweep["points"]

    def pt(k):
        return pts[k]["summary"]

    nominal = pt(NOMINAL).get("speed_mps")
    if nominal is not None and nominal <= 0.0:
        nominal = None  # a policy that does not walk at the nominal point: the share rules fail, never divide by 0
    share = lambda sp: None if sp is None or nominal is None else sp / nominal  # noqa: E731
    out = {}
    # F1: mean speed (analysed episodes) >= 0.225 and no falls in 5 (short episodes count as falls)
    v = (t.get("speed") or {}).get("forward_speed_mps")
    falls = t.get("falls")
    out["F1"] = {"pass": v is not None and v >= 0.225 and falls == 0 and t.get("episodes") == 5,
                 "speed_mps": v, "falls": falls, "episodes": t.get("episodes"), "short": t.get("short_episodes")}
    # F2: every joint's mean RMS torque / continuous rating < 1.0; no value (no analysed episode) fails
    per = (t.get("drives") or {}).get("per_joint") or {}
    worst = max(((j, d["rms_over_continuous"]["mean"]) for j, d in per.items()), key=lambda x: x[1], default=None)
    out["F2"] = {"pass": worst is not None and worst[1] < 1.0, "worst_joint": worst[0] if worst else None,
                 "worst": worst[1] if worst else None}
    for rule, k in (("T1", STEP[3]), ("T2", RAMP[8])):
        s = pt(k)
        sp = s.get("speed_mps")
        out[rule] = {"pass": s["falls"] == 0 and sp is not None and share(sp) is not None and sp >= 0.8 * nominal,
                     "falls": s["falls"], "speed_mps": sp, "share_of_nominal": share(sp)}
    rows = {}
    for name, k in (("step_2cm", STEP[2]), ("step_3cm", STEP[3]), ("ramp_5deg", RAMP[5]), ("ramp_8deg", RAMP[8])):
        s = pt(k)
        dr = s.get("drift_abs_mean_m")
        rows[name] = {"falls": s["falls"], "drift_m": dr, "pass": s["falls"] == 0 and dr is not None and dr <= 0.30}
    out["T3"] = {"pass": all(r["pass"] for r in rows.values()), "points": rows}
    s = pt(FRICTION_03)
    sp, dr = s.get("speed_mps"), s.get("drift_abs_mean_m")
    out["T4"] = {"pass": s["falls"] == 0 and dr is not None and dr <= 0.20 and share(sp) is not None
                 and sp >= 0.8 * nominal, "falls": s["falls"], "drift_m": dr, "speed_mps": sp,
                 "share_of_nominal": share(sp)}
    f = {dv: pt(k)["falls"] for dv, k in PUSH.items()}
    out["T5"] = {"pass": f[-1.0] == 0 and f[1.0] == 0 and f[-1.25] <= 2 and f[1.25] <= 2,
                 "falls": {f"{dv:+g}": n for dv, n in f.items()}}
    out["nominal_speed_mps"] = nominal
    return out


RULES = ("F1", "F2", "T1", "T2", "T3", "T4", "T5")


def verdict(per_seed: dict) -> dict:
    out = {}
    for r in RULES:
        vals = [None if s is None else s[r]["pass"] for s in per_seed.values()]
        if len(vals) < 2 or any(v is None for v in vals):
            out[r] = "NOT JUDGED (needs both seeds)" if len(vals) < 2 else "NOT JUDGED"
        elif all(vals):
            out[r] = "PASS"
        elif not any(vals):
            out[r] = "FAIL"
        else:
            out[r] = "MIXED"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", nargs=3, action="append", metavar=("NAME", "GAIT_JSON", "SWEEP_JSON"), required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    per_seed, problems = {}, {}
    for name, gpath, spath in a.seed:
        gait, sweep = _load(gpath), _load(spath)
        problems[name] = provenance(gait, sweep)
        per_seed[name] = None if problems[name] else judge_seed(gait, sweep)
    v = verdict(per_seed)
    for name, p in problems.items():
        print(f"{name}: provenance " + ("OK (model, trike_env.py, run commit, final checkpoint)" if not p
                                       else "PROBLEM: " + "; ".join(p)))
    for r in RULES:
        cells = []
        for name, s in per_seed.items():
            cells.append(f"{name} {'-' if s is None else ('pass' if s[r]['pass'] else 'FAIL')}")
        print(f"{r}: {v[r]:<30} " + ", ".join(cells))
    res = {"simulation_only": True, "prereg": "hardware/triceratops_build/rl/analysis/PREREG_RUN3.md",
           "run_commit": RUN_COMMIT, "provenance_problems": problems, "per_seed": per_seed, "verdict": v}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    else:
        print(json.dumps(per_seed, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
