"""Report-only: per-actuator command and force use at the final checkpoint, from existing M1 traces.

Reads ST/round/panels/<run>/final/m1/episode_trace_seed*.csv.gz (deterministic replays, 20 seeds,
already produced by F2's M1). Runs no simulation. Window = rows with in_window true (steps 200-999).
Per actuator: mean command, max |command|, share of window steps with |command| >= 0.9,
mean and max |force_frac|, and the share of steps with |force_frac| >= 0.9.
"""
import csv
import glob
import gzip
import sys

ROOT = "/home/user/work/pc_study/round/panels"
RUNS = ["stance7_r10LTSBpc_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBpc_s80",
        "stance7_r10LTSBoff_s60", "stance7_r10LTSBoff_s42", "stance7_r10LTSBoff_s80"]
ACTS = ["neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
        "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
        "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]


def truthy(x):
    return str(x).strip().lower() in ("1", "true", "1.0")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "final"
    for run in RUNS:
        files = sorted(glob.glob(f"{ROOT}/{run}/{label}/m1/episode_trace_seed*.csv.gz"))
        if not files:
            print(run, label, "not found")
            continue
        acc = {a: {"n": 0, "cmd_sum": 0.0, "cmd_absmax": 0.0, "cmd_ge09": 0, "ff_sum": 0.0, "ff_max": 0.0,
                   "ff_ge09": 0} for a in ACTS}
        rewards = []
        for fpath in files:
            with gzip.open(fpath, "rt", newline="") as fh:
                ep_r = 0.0
                for row in csv.DictReader(fh):
                    try:
                        ep_r += float(row["reward"])
                    except (TypeError, ValueError):
                        pass
                    if not truthy(row["in_window"]):
                        continue
                    for a in ACTS:
                        c = float(row[f"{a}_command"])
                        ff = abs(float(row[f"{a}_force_frac"]))
                        s = acc[a]
                        s["n"] += 1
                        s["cmd_sum"] += c
                        s["cmd_absmax"] = max(s["cmd_absmax"], abs(c))
                        s["cmd_ge09"] += abs(c) >= 0.9
                        s["ff_sum"] += ff
                        s["ff_max"] = max(s["ff_max"], ff)
                        s["ff_ge09"] += ff >= 0.9
                rewards.append(ep_r)
        print(f"=== {run} {label}: {len(files)} deterministic episodes; mean episode reward "
              f"{sum(rewards)/len(rewards):.1f}")
        for a in ACTS:
            s = acc[a]
            n = s["n"]
            print(f"  {a:18s} cmd_mean {s['cmd_sum']/n:+.4f} |cmd|max {s['cmd_absmax']:.3f} "
                  f"|cmd|>=0.9 {s['cmd_ge09']/n:.4f} | |force_frac| mean {s['ff_sum']/n:.3f} max {s['ff_max']:.3f} "
                  f">=0.9 {s['ff_ge09']/n:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
