"""Verifier's own re-derivation of the warning_r9 verdicts (written independently of warning_r9/wr9_apply.py).
Read only on every input; no simulation; no network evaluation.

Rules (warning_r9 PREREG.txt + PREREG_STAGE2.txt section 4): bank_ss = mean of bank_action_now[:, statue_settled, 14];
trail8 = causal 8-row trailing mean (pandas rolling, min_periods=1); P1 > 0.150, S1 > 0.186, S2 > 0.3376,
S3 trail8 - B > 0.13 (B = mean raw bank_ss at update_timesteps <= 1,048,576).  Probe rows after the last evaluation
are ignored.  Outcomes (r9 schema): over-bar episode = screen_v2_reasons mentions max_actuator_saturation_fraction;
flag = >= 5 of 10 over the bar; alarm = second of two consecutive flags.  Verdict on the ANY alarm:
HOLDS (every positive warned with f < o, no false alarm), PARTIAL (>= 1 warned, <= 1 false alarm), FAILS otherwise.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ST5 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5")
TR = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/tailround/fetch")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
EW = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_warning")
RULES = {"P1": 0.150, "S1": 0.186, "S2": 0.3376, "S3": 0.13}


def r9_paths(run):
    sub = "runs" if run in ("s42", "s52", "s60") else "replication/runs"
    return ST5 / sub / f"stance5_LTSB_{run}" / "01_stance" / "eval_episodes.csv", TR / f"stance5_LTSB_{run}" / "A" / "policy_probe.npz"


def r10_paths(run):
    d = F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance"
    return d / "eval_episodes.csv.copy1", d / "policy_probe.npz.copy1"


def outcomes(csv, named=False):
    d = pd.read_csv(csv)
    if named:
        over = d["saturated_actuator_fraction"].to_numpy() > 0.10
    else:
        over = d["screen_v2_reasons"].astype(str).str.contains("max_actuator_saturation_fraction", regex=False).to_numpy()
    d = d.assign(over=over)
    ev = d.groupby("timesteps").agg(n=("over", "size"), k=("over", "sum")).sort_index()
    ts = ev.index.to_numpy()
    flag = (ev["k"] >= np.ceil(ev["n"] / 2)).to_numpy()
    alarm = flag & np.concatenate([[False], flag[:-1]])
    first = lambda m: int(ts[np.argmax(m)]) if m.any() else None  # noqa: E731
    reasons = Counter()
    if "screen_v2_reasons" in d.columns:
        for s in d["screen_v2_reasons"].dropna().astype(str):
            for tok in s.replace(";", ",").replace("|", ",").split(","):
                tok = tok.strip()
                if tok:
                    reasons[tok.split("=")[0].split(":")[0].split(" ")[0]] += 1
    return {"first_over": first(ev["k"].to_numpy() > 0), "first_flag": first(flag), "first_alarm": first(alarm),
            "n_over": int(ev["k"].sum()), "n_flag": int(flag.sum()), "n_eval": len(ev), "eps": sorted(set(ev["n"])),
            "last": int(ts[-1]), "reasons": reasons}


def signals(npz):
    z = np.load(npz, allow_pickle=False)
    t = z["update_timesteps"].astype(np.int64)
    lab = z["bank_labels"].astype(str)
    cols = np.flatnonzero(lab == "statue_settled")
    ss = z["bank_action_now"][:, cols, 14].astype(np.float64).mean(axis=1)
    tr = pd.Series(ss).rolling(8, min_periods=1).mean().to_numpy()
    B = float(ss[t <= 1_048_576].mean())
    return t, ss, tr, B, cols.tolist()


def first_cross(t, x, thr, last):
    m = (x > thr) & (t <= last)
    return int(t[np.argmax(m)]) if m.any() else None


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "r9"
    runs = ("s42", "s52", "s60", "s70", "s80") if which == "r9" else ("s42", "s60", "s80")
    rows = {}
    for run in runs:
        csv, npz = r9_paths(run) if which == "r9" else r10_paths(run)
        o = outcomes(csv, named=(which == "r10"))
        t, ss, tr, B, cols = signals(npz)
        fires = {k: first_cross(t, (tr - B) if k == "S3" else tr, v, o["last"]) for k, v in RULES.items()}
        rows[run] = (o, fires, t, tr, B)
        print(f"{run}: evals {o['n_eval']} eps/eval {o['eps']} last {o['last']}; over-bar eps {o['n_over']}; flagged evals "
              f"{o['n_flag']}; first over-bar {o['first_over']}; first flag {o['first_flag']}; first alarm {o['first_alarm']}")
        print(f"     statue_settled cols {cols}; B {B:+.4f}; fires " + ", ".join(f"{k} {v}" for k, v in fires.items()))
        if which == "r9":
            print(f"     reason tokens: {dict(o['reasons'].most_common(8))}")
    print()
    print("Verdicts on the ANY alarm (positives = runs with an alarm):")
    for k in RULES:
        tw, late, fa, tn = [], [], [], []
        for run, (o, fires, *_r) in rows.items():
            f, a = fires[k], o["first_alarm"]
            if a is None:
                (fa if f is not None else tn).append(run)
            elif f is not None and f < a:
                tw.append(f"{run} lead {a - f}")
            else:
                late.append(f"{run} f={f}")
        npos = sum(1 for o, *_r in rows.values() if o["first_alarm"] is not None)
        ok = len(tw) == npos and not fa
        verdict = "HOLDS" if ok else ("PARTIAL" if tw and len(fa) <= 1 else "FAILS")
        print(f"  {k}: warned {len(tw)}/{npos} {tw}; late/miss {late}; false alarms {len(fa)}/{len(rows) - npos} {fa}; "
              f"true neg {tn} -> {verdict}")
    print()
    print("Against the first over-bar episode (O1u): lead = first over-bar - firing")
    for k in RULES:
        print(f"  {k}: " + ", ".join(f"{run} {(o['first_over'] - f) if (f is not None and o['first_over'] is not None) else None}"
                                     for run, (o, fires, *_r) in rows.items() for f in [fires[k]]))
    print()
    print("Exploratory: trail8 maxima (pre-alarm for positives; whole run to the last eval for negatives; matched windows)")
    for run, (o, fires, t, tr, B) in rows.items():
        last = o["last"]
        a = o["first_alarm"]
        pre = tr[(t < a) & (t <= last)].max() if a is not None else None
        allm = tr[t <= last].max()
        win = {H: float(tr[t <= min(H, (a - 1) if a else H)].max()) for H in (2_097_152, 3_145_728, 4_194_304)}
        print(f"  {run}: pre-alarm max {pre if pre is None else round(float(pre), 4)}; run max {allm:.4f}; "
              f"max to 2.10/3.15/4.19M (positives cut before alarm) " + " / ".join(f"{v:.3f}" for v in win.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
