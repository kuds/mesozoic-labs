"""Judged runs of the model-based trot, per PREREG_WALK60.md. Simulation only.

Usage (from the repository root):
  python hardware/triceratops_build/walk/judge_walk60.py --out hardware/triceratops_build/walk/results/walk60_<date>
  [--workers 3] [--seeds 101-110] [--report-seeds 101-105]

Writes results.json (every episode's facts and measures) and RESULTS.md (the rules, judged). The rules here are the
pre-registration's; changing them here changes nothing about what was registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import walk60  # noqa: E402

COMMAND = 0.25
JUDGED = {  # name: (label, plant overrides, impratio)
    "J1": ("grippy (friction 1.0), the model's contact", {"friction_range": (1.0, 1.0)}, None),
    "J2": ("slippery (friction 0.4), the model's contact", {"friction_range": (0.4, 0.4)}, None),
    "J3": ("grippy (friction 1.0), stiffer contact (impratio 10)", {"friction_range": (1.0, 1.0)}, 10.0),
    "J4": ("slippery (friction 0.4), stiffer contact (impratio 10)", {"friction_range": (0.4, 0.4)}, 10.0),
}
REPORTED = {  # name: (label, plant overrides, impratio, command, delay)
    "R1": ("J1 at 0 ms delay", {"friction_range": (1.0, 1.0)}, None, COMMAND, 0),
    "R2": ("J2 at 0 ms delay", {"friction_range": (0.4, 0.4)}, None, COMMAND, 0),
    "R3": ("J1 at the 0.15 m/s command", {"friction_range": (1.0, 1.0)}, None, 0.15, 10),
    "R4": ("J1 standing (0 m/s)", {"friction_range": (1.0, 1.0)}, None, 0.0, 10),
    "R5": ("nominal model, no randomisation or noise", {"domain_randomization": False, "observation_noise": False},
           None, COMMAND, 10),
}
W2_MEAN, W2_MIN, W3_HEAT, W4_SLIP, FLAG_LIMIT = 0.225, 0.20, 1.0, 0.25, 0.01


def seeds(text: str) -> list[int]:
    a, b = text.split("-") if "-" in text else (text, text)
    return list(range(int(a), int(b) + 1))


def _episode(job):
    name, seed, plant, impratio, command, delay = job
    r = walk60.run_episode(walk60.WalkParams(), seed=seed, command=command, delay_steps=delay, impratio=impratio,
                           **plant)
    return name, r


def judge(eps: list[dict]) -> dict:
    ms = [e["measures"] for e in eps]
    speeds = [m["forward_speed_mps"] for m in ms]
    heat = max(d["rms_over_continuous"] for m in ms for d in m["drives"].values())
    feet = ms[0]["slip_ratio"].keys()
    slip = {c: (sum(m["slip_ratio"][c] for m in ms if m["slip_ratio"][c] is not None)
                / max(1, sum(m["slip_ratio"][c] is not None for m in ms))) for c in feet}
    falls = sum(e["fell"] for e in eps)
    out = {
        "W1": {"falls": falls, "pass": falls == 0},
        "W2": {"mean_mps": sum(speeds) / len(speeds), "min_mps": min(speeds),
               "pass": sum(speeds) / len(speeds) >= W2_MEAN and min(speeds) >= W2_MIN},
        "W3": {"worst_rms_over_continuous": heat, "pass": heat < W3_HEAT},
        "W4": {"slip_ratio_mean_by_foot": slip, "pass": max(slip.values()) <= W4_SLIP},
    }
    out["all_pass"] = all(out[k]["pass"] for k in ("W1", "W2", "W3", "W4"))
    return out


def reported(eps: list[dict]) -> dict:
    ms = [e["measures"] for e in eps]
    worst = lambda key: max((d[key], j) for m in ms for j, d in m["drives"].items())  # noqa: E731
    return {"speed_mps": [m["forward_speed_mps"] for m in ms], "falls": sum(e["fell"] for e in eps),
            "heat_worst": worst("rms_over_continuous"), "at_limit_episode_worst": worst("at_limit_episode"),
            "at_limit_nominal_worst": worst("at_limit_nominal"),
            "heading_change_abs_worst_deg": max(abs(m["heading_change_deg"]) for m in ms),
            "sideways_drift_abs_worst_m": max(abs(m["sideways_drift_m"]) for m in ms),
            "power_w_mean": sum(m["power_w"] for m in ms) / len(ms),
            "touchdowns_per_s_mean": {c: sum(m["touchdowns_per_s"][c] for m in ms) / len(ms)
                                      for c in ms[0]["touchdowns_per_s"]},
            "ik_err_max_m": max(e["ik_err_max_m"] for e in eps)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", default="101-110")
    ap.add_argument("--report-seeds", default="101-105")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s, plant, imp, COMMAND, 10) for n, (_, plant, imp) in JUDGED.items() for s in seeds(args.seeds)]
    jobs += [(n, s, plant, imp, cmd, delay) for n, (_, plant, imp, cmd, delay) in REPORTED.items()
             for s in seeds(args.report_seeds)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        done = pool.map(_episode, jobs, chunksize=1)
    by = {}
    for name, r in done:
        by.setdefault(name, []).append(r)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(HERE), str(walk60.RL_DIR)],
                                capture_output=True, text=True, cwd=HERE).stdout.strip())
    res = {"simulation_only": True, "tool": "walk/judge_walk60.py", "commit": head, "dirty": dirty,
           "walk60_sha256": hashlib.sha256((HERE / "walk60.py").read_bytes()).hexdigest(),
           "params": walk60.dataclasses.asdict(walk60.WalkParams()), "command_mps": COMMAND,
           "seeds": seeds(args.seeds), "report_seeds": seeds(args.report_seeds),
           "wall_s": round(time.time() - t0, 1), "judged": {}, "reported": {}}
    for n, (label, plant, imp) in JUDGED.items():
        res["judged"][n] = {"label": label, "plant": plant, "impratio": imp, "rules": judge(by[n]),
                            "summary": reported(by[n]), "episodes": by[n]}
    for n, (label, plant, imp, cmd, delay) in REPORTED.items():
        res["reported"][n] = {"label": label, "plant": plant, "impratio": imp, "command_mps": cmd,
                              "delay_steps": delay, "summary": reported(by[n]), "episodes": by[n]}
    res["verdict_pass"] = all(v["rules"]["all_pass"] for v in res["judged"].values())
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "RESULTS.md").write_text(markdown(res))
    print(markdown(res))
    return 0


def markdown(res: dict) -> str:
    f = lambda x, n=3: f"{x:.{n}f}"  # noqa: E731
    L = ["# Model-based trot, judged runs (SIMULATION)", "",
         f"Written by `walk/judge_walk60.py` at commit {res['commit'][:8]}{' (dirty)' if res['dirty'] else ''}; "
         f"rules from `walk/PREREG_WALK60.md`. Command {res['command_mps']} m/s, delay 10 ms unless stated, "
         f"randomisation and noise on unless stated. Judged seeds {res['seeds'][0]}-{res['seeds'][-1]}.", "",
         f"**Verdict: {'PASS' if res['verdict_pass'] else 'NOT PASSED'}** (W1-W4 in all four judged conditions).", "",
         "| Condition | W1 falls | W2 speed, mean (min), m/s | W3 worst RMS / rating | W4 worst foot slip ratio | All |",
         "|---|---|---|---|---|---|"]
    for n, v in res["judged"].items():
        r = v["rules"]
        L.append(f"| {n} {v['label']} | {r['W1']['falls']} {'pass' if r['W1']['pass'] else 'FAIL'} | "
                 f"{f(r['W2']['mean_mps'])} ({f(r['W2']['min_mps'])}) {'pass' if r['W2']['pass'] else 'FAIL'} | "
                 f"{f(r['W3']['worst_rms_over_continuous'], 2)} {'pass' if r['W3']['pass'] else 'FAIL'} | "
                 f"{f(max(r['W4']['slip_ratio_mean_by_foot'].values()), 2)} {'pass' if r['W4']['pass'] else 'FAIL'} | "
                 f"{'PASS' if r['all_pass'] else 'FAIL'} |")
    L += ["", "## Reported (no rule)", "",
          "| Case | Speed, m/s (per episode) | Falls | Worst RMS / rating (joint) | At episode limit, worst joint | "
          "At nominal limit, worst joint | Heading change, worst | Drift, worst | Power | IK error |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(n, v["label"], v["summary"]) for n, v in res["judged"].items()]
    rows += [(n, v["label"], v["summary"]) for n, v in res["reported"].items()]
    for n, label, s in rows:
        L.append(f"| {n} {label} | {', '.join(f(x) for x in s['speed_mps'])} | {s['falls']} | "
                 f"{f(s['heat_worst'][0], 2)} ({s['heat_worst'][1]}) | {100 * s['at_limit_episode_worst'][0]:.1f} % "
                 f"({s['at_limit_episode_worst'][1]}) | {100 * s['at_limit_nominal_worst'][0]:.1f} % "
                 f"({s['at_limit_nominal_worst'][1]}) | {s['heading_change_abs_worst_deg']:.1f}° | "
                 f"{f(s['sideways_drift_abs_worst_m'], 2)} m | {s['power_w_mean']:.0f} W | "
                 f"{1000 * s['ik_err_max_m']:.2f} mm |")
    L += ["", "Time at the torque limit is reported against the 1 % flag used for the learned policies; it is not "
          "judged (PREREG_WALK60.md). Heading and drift follow the simulated gyro drift. Simulation only."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
