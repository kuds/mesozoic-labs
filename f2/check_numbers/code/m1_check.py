"""Independent reduction of the M1 per-step traces (ROUND l.119-121, 163-191, 271-279; criteria.txt l.202-210).

Per run, per traced label (J = 4194304 5242880 6291456 8388608 10485760 final; R1 = 10485760, R2 = final):
  per episode and actuator: window (in_window == 1, steps 200..L-1) mean of <a>_command; share of window steps
  with |command| >= 0.99; sign counts (+ when command >= 0.99, - when <= -0.99).
  CMD_a(X) = mean over episodes with a window of their window-mean command of a.
  TAIL FLAG: full-horizon (1000-step) episode, tail actuator 11-14, share > 0.10, with sign (tie: both signs).
  MEAN_a = (CMD_a(R1) + CMD_a(R2)) / 2.
  PARKED: tail pitch actuator 11, 13, 14: MEAN_a >= +0.60, or CMD_a(J) >= +0.60 at >= 2 J labels.
  TRACE RAIL: among R1 and R2 traced episodes, >= 2 full-horizon tail flags on one actuator and sign, or 1 on an
  actuator whose MEAN_a in that sign >= 0.60.
  Habit sites from M1 (criteria l.202-210), for every actuator a other than tail_3_pitch_act:
   T: at a traced label, >= 2 full-horizon episodes over the BAR on a in one sign;
   D: |CMD_a| >= 0.60 in one sign at >= 2 J labels, or |mean(CMD_a(R1), CMD_a(R2))| >= 0.60.
Also checks each decompressed trace's sha256 against m1_summary.json csv_sha256 and compares CMD_a with
m1_summary.json cmd_all_actuators.
"""
import csv
import gzip
import hashlib
import io
import json
import math
import sys

PAN = "/home/user/work/pc_study/round/panels"
LABELS = ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]
ACTS = ["neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
        "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
        "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]
TAIL = {11, 12, 13, 14}
TAIL_PITCH = {11, 13, 14}


def episode(path):
    raw = gzip.open(path, "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    rd = csv.DictReader(io.StringIO(raw.decode()))
    n = 0
    sums = [0.0] * 15
    cnt = 0
    over = [0] * 15
    pos = [0] * 15
    neg = [0] * 15
    for r in rd:
        n += 1
        if r["in_window"] not in ("1", "True", "true", "1.0"):
            continue
        cnt += 1
        for i, a in enumerate(ACTS):
            c = float(r[a + "_command"])
            sums[i] += c
            if abs(c) >= 0.99:
                over[i] += 1
                if c >= 0.99:
                    pos[i] += 1
                else:
                    neg[i] += 1
    out = {"length": n, "window": cnt, "sha256": h}
    if cnt:
        out["mean"] = [s / cnt for s in sums]
        out["share"] = [o / cnt for o in over]
        out["pos"] = pos
        out["neg"] = neg
    return out


def signs(ep, i):
    if ep["pos"][i] > ep["neg"][i]:
        return ["+"]
    if ep["neg"][i] > ep["pos"][i]:
        return ["-"]
    return ["+", "-"]


def run(run_id):
    res = {"labels": {}}
    for lab in LABELS:
        d = f"{PAN}/{run_id}/{lab}/m1"
        summ = json.load(open(d + "/m1_summary.json"))
        eps = []
        sha_ok = True
        for s in summ["seeds"]:
            e = episode(f"{d}/episode_trace_seed{s}.csv.gz")
            if summ["csv_sha256"].get(f"episode_trace_seed{s}.csv") != e["sha256"]:
                sha_ok = False
            eps.append(e)
        win = [e for e in eps if e["window"] > 0]
        cmd = [sum(e["mean"][i] for e in win) / len(win) if win else None for i in range(15)]
        full = [e for e in eps if e["length"] == 1000]
        flags = []  # (actuator index, sign) for full-horizon over-bar episodes, every actuator
        for e in full:
            for i in range(15):
                if e["share"][i] > 0.10:
                    for sg in signs(e, i):
                        flags.append((i, sg))
        tool_cmd = summ.get("cmd_all_actuators", {})
        maxdiff = max(abs(cmd[i] - tool_cmd[str(i)]) for i in range(15) if cmd[i] is not None)
        res["labels"][lab] = {"n_eps": len(eps), "full": len(full), "with_window": len(win), "sha_ok": sha_ok,
                              "cmd": cmd, "flags": flags, "max_abs_diff_vs_tool_cmd": maxdiff,
                              "tool_tail_flags": summ.get("tail_flags"),
                              "tool_any_over_bar": summ.get("any_actuator_over_bar_episodes")}
    L = res["labels"]
    mean_a = {}
    for i in range(15):
        c1, c2 = L["10485760"]["cmd"][i], L["final"]["cmd"][i]
        vals = [c for c in (c1, c2) if c is not None]
        mean_a[i] = sum(vals) / len(vals) if vals else None
    res["MEAN_a"] = {ACTS[i]: mean_a[i] for i in range(15)}
    parked = []
    for i in TAIL_PITCH:
        nj = sum(1 for lab in LABELS if (L[lab]["cmd"][i] or -9) >= 0.60)
        if (mean_a[i] or -9) >= 0.60 or nj >= 2:
            parked.append((ACTS[i], mean_a[i], nj))
    res["PARKED"] = parked
    driven = [(ACTS[i], mean_a[i]) for i in TAIL if mean_a[i] is not None and abs(mean_a[i]) >= 0.60
              and not any(p[0] == ACTS[i] for p in parked)]
    res["DRIVEN"] = driven
    # trace rail: R1 and R2 tail flags
    tf = {}
    for lab in ("10485760", "final"):
        for (i, sg) in L[lab]["flags"]:
            if i in TAIL:
                tf[(i, sg)] = tf.get((i, sg), 0) + 1
    rail = []
    for (i, sg), n in tf.items():
        m = mean_a[i] if sg == "+" else (-mean_a[i] if mean_a[i] is not None else None)
        if n >= 2 or (n == 1 and m is not None and m >= 0.60):
            rail.append((ACTS[i], sg, n))
    res["TRACE_RAIL"] = rail
    res["tail_flags_R1_R2"] = {f"{ACTS[i]}{sg}": n for (i, sg), n in tf.items()}
    # habit sites T and D for a != tail_3 (index 14)
    T, D = {}, {}
    for i in range(14):
        for lab in LABELS:
            for sg in ("+", "-"):
                n = sum(1 for f in L[lab]["flags"] if f == (i, sg))
                if n >= 2:
                    T.setdefault(ACTS[i], []).append((lab, sg, n))
        for sg, mul in (("+", 1), ("-", -1)):
            nj = sum(1 for lab in LABELS if L[lab]["cmd"][i] is not None and mul * L[lab]["cmd"][i] >= 0.60)
            if nj >= 2:
                D.setdefault(ACTS[i], []).append(("J", sg, nj))
        if mean_a[i] is not None and abs(mean_a[i]) >= 0.60:
            D.setdefault(ACTS[i], []).append(("R1R2", mean_a[i]))
    res["habit_T"] = T
    res["habit_D"] = D
    res["cmd_tail3"] = {lab: L[lab]["cmd"][14] for lab in LABELS}
    return res


def main():
    out = {}
    for arm in ("pc", "off"):
        for s in (60, 42, 80):
            rid = f"stance7_r10LTSB{arm}_s{s}"
            out[rid] = run(rid)
    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()
