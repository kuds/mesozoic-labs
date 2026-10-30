"""warning_r9: apply the pre-registered rules (PREREG.txt, PREREG_STAGE2.txt) to probe + ledger series.
Read only on every input.  No simulation, no network evaluation.

    python -B wr9_apply.py r10 > wr9_r10_repro.txt     (reproduction on the r10 runs, the F2 copies)
    python -B wr9_apply.py r9 --thr9 X > wr9_r9.txt     (the held-out test; X from PREREG_STAGE2.txt)

Definitions copied from early_warning/ew_events.py and ew_analyze.py (see PREREG.txt sections 1, 4, 5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
ST5 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5")
TR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/fetch")
A_T3 = 14
TAIL_ACTS = {"tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"}
EARLY = 524_288


def sources(which):
    if which == "r10":
        return {r: (F2 / f"stance6_r10LTSB_{r}" / "copies" / "01_stance" / "eval_episodes.csv.copy1",
                    F2 / f"stance6_r10LTSB_{r}" / "copies" / "01_stance" / "policy_probe.npz.copy1")
                for r in ("s42", "s60", "s80")}
    out = {}
    for r in ("s42", "s52", "s60", "s70", "s80"):
        sub = "runs" if r in ("s42", "s52", "s60") else "replication/runs"
        out[r] = (ST5 / sub / f"stance5_LTSB_{r}" / "01_stance" / "eval_episodes.csv",
                  TR / f"stance5_LTSB_{r}" / "A" / "policy_probe.npz")
    return out


def trailing(x, k):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        j = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[j]) / (i + 1 - j)
    return out


def per_eval(path, unnamed=False):
    """Per evaluation.  Named mode (r10 schema): over the bar = saturated_actuator_fraction > 0.10, TAIL = a tail
    actuator named, CMD_L from tail_3_pitch_act_command_mean.  Unnamed mode (r9 schema, or forced): over the bar =
    screen_v2_reasons lists max_actuator_saturation_fraction (identical to fraction > 0.10 on all 2,730 r10 episodes,
    wr9_reasoncheck.txt); no actuator name, no CMD_L (PREREG_STAGE2.txt)."""
    d = pd.read_csv(path)
    named = (not unnamed) and {"saturated_actuator_fraction", "saturated_actuator",
                               "tail_3_pitch_act_command_mean"} <= set(d.columns)
    if named:
        d["over"] = d["saturated_actuator_fraction"] > 0.10
        d["tail_over"] = d["over"] & d["saturated_actuator"].isin(TAIL_ACTS)
        d["cmd"] = d["tail_3_pitch_act_command_mean"]
        d["act"] = d["saturated_actuator"]
    else:
        d["over"] = d["screen_v2_reasons"].fillna("").astype(str).str.contains("max_actuator_saturation_fraction")
        d["tail_over"] = False
        d["cmd"] = np.nan
        d["act"] = np.where(d["over"], "unnamed", None)
    g = d.groupby("timesteps")
    e = pd.DataFrame({
        "n": g.size(), "n_over": g["over"].sum(), "n_tail_over": g["tail_over"].sum(),
        "cmd_l": g["cmd"].mean(),
        "over_acts": d[d["over"]].groupby("timesteps")["act"].agg(lambda s: ",".join(sorted(set(map(str, s))))),
    }).reset_index()
    e["over_acts"] = e["over_acts"].fillna("")
    need = np.where(e["n"] == 10, 5, np.ceil(e["n"] / 2.0)).astype(int)
    e["any_flag"] = e["n_over"] >= need
    e["tail_flag"] = e["n_tail_over"] >= need
    e["any_alarm"] = e["any_flag"] & e["any_flag"].shift(1, fill_value=False)
    e["tail_alarm"] = e["tail_flag"] & e["tail_flag"].shift(1, fill_value=False)
    e["fob"] = e["n_tail_over"] > 0
    e["over_any_ep"] = e["n_over"] > 0
    e["park"] = e["cmd_l"] >= 0.60
    e["rail"] = e["cmd_l"] >= 0.90
    e.attrs["named"] = named
    return e


def first_eval(e, col):
    s = e[e[col]]
    return int(s["timesteps"].iloc[0]) if len(s) else None


def probe(path):
    with np.load(path, allow_pickle=False) as d:
        t = np.asarray(d["update_timesteps"]).astype(np.int64)
        a = np.asarray(d["bank_action_now"])[:, :, A_T3].astype(float)
        lab = np.asarray(d["bank_labels"]).astype(str)
        r = np.asarray(d["rollout_mu_mean"])[:, A_T3].astype(float) if "rollout_mu_mean" in d.files else None
        bsha = hashlib.sha1(np.asarray(d["bank_raw_obs"]).tobytes()).hexdigest()[:10]
    ss = lab == "statue_settled"
    assert np.all(np.diff(t) > 0), "update_timesteps not strictly increasing"
    return {"t": t, "bank_ss": a[:, ss].mean(axis=1) if ss.any() else None, "rollout_mu": r, "n_ss": int(ss.sum()),
            "labels": lab.tolist(), "bank_sha1": bsha}


def rules(pr, e, thr9):
    t = pr["t"]
    out = {}
    if pr["bank_ss"] is not None:
        tr = trailing(pr["bank_ss"], 8)
        B = float(np.mean(pr["bank_ss"][t <= 1_048_576]))
        out["P1 trail8(bank_ss)>0.150"] = ("probe", t, tr, 0.150, ">")
        out["S1 trail8(bank_ss)>0.186"] = ("probe", t, tr, 0.186, ">")
        out[f"S2 trail8(bank_ss)>thr9={thr9:.4f}"] = ("probe", t, tr, thr9, ">")
        out[f"S3 trail8(bank_ss)-B>0.13 (B={B:+.4f})"] = ("probe", t, tr - B, 0.13, ">")
    if e.attrs.get("named"):
        out["P2 CMD_L>=0.30"] = ("ledger", e["timesteps"].to_numpy(), e["cmd_l"].to_numpy(), 0.30, ">=")
        out["S4 CMD_L>0.723"] = ("ledger", e["timesteps"].to_numpy(), e["cmd_l"].to_numpy(), 0.723, ">")
    if pr["rollout_mu"] is not None:
        out["S5 trail8(rollout_mu)>0.638"] = ("probe", t, trailing(pr["rollout_mu"], 8), 0.638, ">")
    return out


def fire_info(tt, v, thr, op, last_eval):
    keep = tt <= last_eval
    tt, v = tt[keep], v[keep]
    m = (v >= thr) if op == ">=" else (v > thr)
    idx = np.flatnonzero(m)
    if not len(idx):
        return None, [], tt, v, m
    # up-crossings (episodes above threshold)
    ups = [int(tt[i]) for i in idx if i == 0 or not m[i - 1]]
    return int(tt[idx[0]]), ups, tt, v, m


def classify(f, o):
    if o is None:
        return ("FALSE ALARM" if f is not None else "true negative"), None
    if f is None:
        return "MISS", None
    if f < o:
        return ("TRUE WARNING (early)" if o - f >= EARLY else "TRUE WARNING (short)"), o - f
    return "LATE (miss)", o - f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["r10", "r9"])
    ap.add_argument("--thr9", type=float, default=0.150)
    ap.add_argument("--unnamed", action="store_true", help="force the r9-schema outcome definitions")
    a = ap.parse_args()
    srcs = sources(a.which)
    res = {}
    print(f"warning_r9 apply: {a.which}; thr9 = {a.thr9:.4f}; EARLY = lead >= {EARLY}")
    print("=" * 118)
    print("A. METADATA AND OUTCOMES (ledger)")
    print("=" * 118)
    tallies = {}
    for run, (lp, pp) in srcs.items():
        e = per_eval(lp, unnamed=a.unnamed)
        pr = probe(pp)
        last = int(e["timesteps"].max())
        dt = np.diff(e["timesteps"].to_numpy())
        if e.attrs["named"]:
            oc = {"O1 FOB": first_eval(e, "fob"), "O2 ANY alarm": first_eval(e, "any_alarm"),
                  "O3 TAIL alarm": first_eval(e, "tail_alarm"), "O4 PARK": first_eval(e, "park")}
        else:
            oc = {"O1u over-bar": first_eval(e, "over_any_ep"), "O2u ANY flag": first_eval(e, "any_flag"),
                  "O2u ANY alarm": first_eval(e, "any_alarm")}
        extra = {"first over-bar (any)": first_eval(e, "over_any_ep"), "ANY flag": first_eval(e, "any_flag"),
                 "TAIL flag": first_eval(e, "tail_flag"), "RAIL": first_eval(e, "rail")}
        print(f"--- {run}: ledger {lp} (schema {'named' if e.attrs['named'] else 'UNNAMED (screen_v2_reasons)'})")
        print(f"    evals {len(e)}, cadence median {int(np.median(dt))} (min {int(dt.min())}, max {int(dt.max())}), "
              f"episodes/eval {sorted(set(e['n'].tolist()))}, first {int(e['timesteps'].min())}, last {last}")
        print(f"    probe {pp}")
        print(f"    probe rows {len(pr['t'])}, row spacing median {int(np.median(np.diff(pr['t'])))}, "
              f"t {int(pr['t'][0])}..{int(pr['t'][-1])}; bank sha1 {pr['bank_sha1']}; statue_settled rows {pr['n_ss']}; "
              f"labels {pr['labels']}")
        print("    outcomes: " + "; ".join(f"{k} {v}" for k, v in oc.items()))
        print("    also:     " + "; ".join(f"{k} {v}" for k, v in extra.items()))
        print(f"    n over-bar episodes {int(e['n_over'].sum())} (tail {int(e['n_tail_over'].sum())}); "
              f"ANY-flagged evals {int(e['any_flag'].sum())}, TAIL-flagged {int(e['tail_flag'].sum())}; "
              f"over-bar actuators {sorted({x for s in e['over_acts'] for x in s.split(',') if x})}; "
              f"max CMD_L {(f"{e['cmd_l'].max():+.3f} at {int(e.loc[e['cmd_l'].idxmax(), 'timesteps'])}" if e.attrs['named'] else 'n/a (no column)')}")
        res[run] = {"e": e, "pr": pr, "oc": oc, "extra": extra, "last": last}

    print("")
    print("=" * 118)
    print("B. RULES PER RUN: first firing, lead against each outcome (o - f, steps), classification")
    print("=" * 118)
    rows_out = []
    for run, R in res.items():
        print(f"--- {run} (ledger last {R['last']})")
        for name, (kind, tt, v, thr, op) in rules(R["pr"], R["e"], a.thr9).items():
            f, ups, tk, vk, m = fire_info(tt, v, thr, op, R["last"])
            o1 = list(R["oc"].values())[0]
            pre = vk[tk < o1] if o1 is not None else vk
            mx_pre = float(np.max(pre)) if len(pre) else float("nan")
            mx_all = float(np.max(vk))
            t_mx = int(tk[int(np.argmax(vk))])
            fell = None
            if f is not None and o1 is not None and f < o1:
                after = (tk > f) & (tk < o1)
                fell = bool(np.any(~m[after])) if after.any() else False
            cells = []
            for oname, o in R["oc"].items():
                cls, lead = classify(f, o)
                cells.append(f"{oname}: {cls}" + (f" lead {lead}" if lead is not None else ""))
                tallies.setdefault((name.split(" ")[0], oname), []).append((run, cls, lead))
            cen = f"; steps from firing to ledger end {R['last'] - f}" if f is not None else ""
            print(f"  {name}")
            print(f"      fires {f}; up-crossings {ups[:6]}{' ...' if len(ups) > 6 else ''} (n {len(ups)}){cen}")
            print(f"      max before first outcome {mx_pre:+.4f}; max over run {mx_all:+.4f} at {t_mx}; fell back below before it: {fell}")
            print("      " + " | ".join(cells))
            rows_out.append({"run": run, "rule": name, "fires": f, "n_up": len(ups), "max_pre_O1": mx_pre,
                             "max_run": mx_all, "t_max": t_mx, "fell_back_before_O1": fell,
                             **{k: v for k, v in R["oc"].items()}})
    print("")
    print("=" * 118)
    print("C. TALLY PER RULE AND OUTCOME (runs: true warnings / positives; false alarms / negatives; leads)")
    print("=" * 118)
    for (rule, oname), lst in tallies.items():
        pos = [x for x in lst if x[1] in ("MISS", "LATE (miss)") or x[1].startswith("TRUE WARNING")]
        neg = [x for x in lst if x[1] in ("FALSE ALARM", "true negative")]
        tw = [x for x in pos if x[1].startswith("TRUE WARNING")]
        fa = [x for x in neg if x[1] == "FALSE ALARM"]
        if oname in ("O1 FOB", "O2u ANY alarm"):
            if pos and len(tw) == len(pos) and not fa:
                verdict = "HOLDS"
            elif tw and len(fa) <= 1:
                verdict = "PARTIAL"
            else:
                verdict = "FAILS"
        else:
            verdict = ""
        print(f"  {rule:3s} {oname:14s}: warned {len(tw)}/{len(pos)} "
              f"[{', '.join(f'{r} {l}' for r, c, l in tw)}]; missed/late "
              f"[{', '.join(f'{r} {c}' for r, c, l in pos if not c.startswith('TRUE'))}]; "
              f"false alarms {len(fa)}/{len(neg)} [{', '.join(r for r, c, l in fa)}]  {verdict}")
    with open(HERE / f"wr9_{a.which}{'_unnamed' if a.unnamed else ''}_rows.json", "w") as fh:
        json.dump(rows_out, fh, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
