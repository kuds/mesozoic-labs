"""V4 (THIS12 lines 856-857, the criteria's V4): M1 on the zero-action statue under the branch's tree: hip roll joint
positions present for actuators 4 and 8; no flags; CMD 0 on every actuator.

    PY -I -B v4_check.py M1_DIR OUT.json

Reads the runner's m1_summary.json and, independently, the gzipped per-step CSVs (sha256 checked against
csv_sha256.txt): every command of every actuator at every step is exactly 0; the joint positions of r_hip_roll (4)
and l_hip_roll (8) are finite at every step; no actuator is over the BAR in any episode (no TAIL FLAG, no FALL FLAG,
no other actuator); CMD_a = 0 for every actuator; 20 episodes, seeds 3042-3061, physics_revision 10.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import sys

sys.dont_write_bytecode = True
VERSION = "v4_check 1.0 (2026-10-26)"
HIP = {4: "r_hip_roll_act", 8: "l_hip_roll_act"}


def main() -> int:
    print(VERSION, flush=True)
    d, out = sys.argv[1], sys.argv[2]
    s = json.load(open(f"{d}/m1_summary.json"))
    sums = dict(reversed(line.split("  ", 1)) for line in open(f"{d}/csv_sha256.txt").read().splitlines())
    names = None
    max_abs = 0.0
    hip_finite = {4: True, 8: True}
    steps = 0
    problems = []
    for name, want in sorted(sums.items()):
        raw = gzip.open(f"{d}/{name}.gz", "rb").read()
        if hashlib.sha256(raw).hexdigest() != want:
            problems.append(f"{name}.gz does not decompress to its recorded sha256")
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        if names is None:
            names = [c[: -len("_command")] for c in rows[0] if c.endswith("_command")]
        for r in rows:
            steps += 1
            for n in names:
                v = float(r[f"{n}_command"])
                max_abs = max(max_abs, abs(v))
            for i, n in HIP.items():
                cell = r[f"{n}_joint_pos"]
                if cell == "" or not math.isfinite(float(cell)):
                    hip_finite[i] = False
    res = {
        "tool": VERSION, "m1_dir": d, "physics_revision": s["physics_revision"], "n_episodes": s["n_episodes"],
        "seeds": s["seeds"], "csv_files": len(sums), "steps_read": steps,
        "max_abs_command_from_csvs": max_abs, "max_abs_command_runner": s["max_abs_command_any_step"],
        "hip_roll_joint_pos_finite_every_step": {str(k): v for k, v in hip_finite.items()},
        "hip_roll_runner": {k: {"present_every_step": v["present_every_step"], "mean_over_episodes": v["mean_over_episodes"]}
                            for k, v in s["hip_roll"].items()},
        "tail_flags": len(s["tail_flags"]), "fall_flags": len(s["fall_flags"]),
        "any_actuator_over_bar_episodes": len(s["any_actuator_over_bar_episodes"]),
        "cmd_all_actuators": s["cmd_all_actuators"], "problems": problems,
        "full_horizon_episodes": s["full_horizon_episodes"],
    }
    res["PASS"] = bool(
        s["physics_revision"] == 10 and s["n_episodes"] == 20 and s["seeds"] == list(range(3042, 3062))
        and len(sums) == 20 and max_abs == 0.0 and s["max_abs_command_any_step"] == 0.0
        and all(hip_finite.values()) and all(v["present_every_step"] for v in s["hip_roll"].values())
        and res["tail_flags"] == 0 and res["fall_flags"] == 0 and res["any_actuator_over_bar_episodes"] == 0
        and all(v == 0.0 for v in s["cmd_all_actuators"].values()) and not problems)
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
