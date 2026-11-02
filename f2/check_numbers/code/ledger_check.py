"""Independent ledger (M5) reading of each run's eval_episodes.csv (ROUND l.133-141, 165-172; criteria.txt l.202-214).

Per row: TAIL over-bar = saturated_actuator_fraction > 0.10 and saturated_actuator a tail actuator (11-14);
ANY over-bar = (screen measured on the row: 'max_actuator_saturation_fraction' in screen_v2_reasons;
screen unmeasured (screen_v2_clean blank/NaN): saturated_actuator_fraction > 0.10) or TAIL over-bar.
Per evaluation: ANY-flagged >= 5 rows; TAIL-flagged >= 5 tail rows. ALARM = second of two consecutive flagged
evaluations. Per actuator a: rows over the bar naming a (fraction > 0.10 and saturated_actuator == a);
L = two consecutive evaluations each with >= 5 such rows; A = an ANY ALARM one of whose two evaluations has >= 5.
Early signs: first tail_3 over-bar evaluation, first tail over-bar, first any over-bar; CMD_L >= 0.30/0.60/0.90 first.
"""
import csv
import json
import math
import sys

F2 = "/home/user/work/pc_study/round/fetch/F2"
TAILS = {"tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"}


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read(path):
    evs = []
    idx = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            e = int(r["evaluation"])
            if e not in idx:
                idx[e] = len(evs)
                evs.append({"e": e, "t": int(r["timesteps"]), "rows": []})
            evs[idx[e]]["rows"].append(r)
    return evs


def analyse(evs):
    out = {"evals": len(evs)}
    per = []
    for ev in evs:
        any_n = tail_n = 0
        by_act = {}
        tail_signs = {}
        cmd = [v for v in (fnum(r["tail_3_pitch_act_command_mean"]) for r in ev["rows"]) if v is not None]
        for r in ev["rows"]:
            frac = fnum(r["saturated_actuator_fraction"])
            act = r["saturated_actuator"]
            over = frac is not None and frac > 0.10 and act != ""
            tail_over = over and act in TAILS
            clean = fnum(r["screen_v2_clean"])
            if clean is not None:
                any_over = "max_actuator_saturation_fraction" in (r["screen_v2_reasons"] or "").split(";")
            else:
                any_over = frac is not None and frac > 0.10
            any_over = any_over or tail_over
            any_n += any_over
            tail_n += tail_over
            if over:
                by_act[act] = by_act.get(act, 0) + 1
            if tail_over:
                ps = fnum(r["saturated_actuator_positive_share"])
                sg = "+" if ps is not None and ps > 0.5 else ("-" if ps is not None and ps < 0.5 else "tie")
                tail_signs[act + sg] = tail_signs.get(act + sg, 0) + 1
        per.append({"t": ev["t"], "any": any_n, "tail": tail_n, "by_act": by_act, "tail_signs": tail_signs,
                    "cmd_l": (sum(cmd) / len(cmd)) if cmd else None})
    any_alarms = [per[i]["t"] for i in range(1, len(per)) if per[i]["any"] >= 5 and per[i - 1]["any"] >= 5]
    tail_alarms = [per[i]["t"] for i in range(1, len(per)) if per[i]["tail"] >= 5 and per[i - 1]["tail"] >= 5]
    out["any_flagged_evals"] = sum(1 for p in per if p["any"] >= 5)
    out["tail_flagged_evals"] = sum(1 for p in per if p["tail"] >= 5)
    out["ANY_ALARMS"] = len(any_alarms)
    out["ANY_ALARM_steps"] = any_alarms
    out["TAIL_ALARMS"] = len(tail_alarms)
    out["first_TAIL_ALARM"] = tail_alarms[0] if tail_alarms else None
    acts = sorted({a for p in per for a in p["by_act"]})
    L, A = {}, {}
    for a in acts:
        l_steps = [per[i]["t"] for i in range(1, len(per))
                   if per[i]["by_act"].get(a, 0) >= 5 and per[i - 1]["by_act"].get(a, 0) >= 5]
        if l_steps:
            L[a] = l_steps
        a_steps = [per[i]["t"] for i in range(1, len(per)) if per[i]["any"] >= 5 and per[i - 1]["any"] >= 5
                   and (per[i]["by_act"].get(a, 0) >= 5 or per[i - 1]["by_act"].get(a, 0) >= 5)]
        if a_steps:
            A[a] = a_steps
    out["habit_L"] = {a: len(v) for a, v in L.items()}
    out["habit_A"] = {a: len(v) for a, v in A.items()}
    out["over_bar_rows_by_actuator"] = {a: sum(p["by_act"].get(a, 0) for p in per) for a in acts}
    first = lambda pred: next((p["t"] for p in per if pred(p)), None)
    out["first_t3_over"] = first(lambda p: p["by_act"].get("tail_3_pitch_act", 0) >= 1)
    out["first_tail_over"] = first(lambda p: any(p["by_act"].get(a, 0) >= 1 for a in TAILS))
    out["first_any_over"] = first(lambda p: p["any"] >= 1)
    for thr in (0.30, 0.60, 0.90):
        out[f"cmd_l_first_ge_{thr:.2f}"] = first(lambda p, thr=thr: p["cmd_l"] is not None and p["cmd_l"] >= thr)
    vals = [p["cmd_l"] for p in per if p["cmd_l"] is not None]
    out["cmd_l_ge_0.60_count"] = sum(1 for v in vals if v >= 0.60)
    out["cmd_l_max"] = max(vals)
    out["cmd_l_last"] = vals[-1]
    out["tail_flagged_signs"] = {k: v for p in per for k, v in p["tail_signs"].items()} if False else \
        _sum_dicts([p["tail_signs"] for p in per if p["tail"] >= 5])
    return out


def _sum_dicts(ds):
    o = {}
    for d in ds:
        for k, v in d.items():
            o[k] = o.get(k, 0) + v
    return o


def main():
    out = {}
    for arm in ("pc", "off"):
        for s in (60, 42, 80):
            rid = f"stance7_r10LTSB{arm}_s{s}"
            out[rid] = analyse(read(f"{F2}/{rid}/copies/01_stance/eval_episodes.csv.copy1"))
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
