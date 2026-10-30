"""VERIFY TEST 1 (posture before onset): my own re-derivation from the raw records.  Read only; no simulation.

Written by the adversarial verifier (verify_posture); no code from posture_timing is imported or copied.
Sources (read only):
  F2 copies of r10 stance6 s42 / s60 / s80: eval_episodes.csv.copy1, train_episodes.csv.copy1, policy_probe.npz.copy1
  ECS runs (24 continuations): eval_episodes.csv, policy_probe.npz
Definitions (stated before reading any time course; taken from ledger_r10round.py M5 for the onsets):
  over-bar episode: saturated_actuator is the named actuator and saturated_actuator_fraction > 0.10
  flagged evaluation: >= 5 over-bar episodes of 10; alarm: the second of two consecutive flagged evaluations
  per-rollout training observation mean: SymmetricRunningMeanStd adds [x; M x] per update, so
      union batch mean = (m_n c_n - m_{n-1} c_{n-1}) / (c_n - c_{n-1}); sagittal channels keep the batch mean.
  pelvis pitch (deg, + nose up) = asin(-g_x) with g = R^T (0,0,-1) (obs 42) -- I re-derive the sign below from
      the ledger, not assume it.
  Change-point ("start of change"): least-squares hinge y = a + b * max(0, t - t0) on [0.524M, END], t0 profiled
      over the rollout grid; approx. interval = t0 whose SSE <= SSE_min * (1 + 4/(n-3)) (an F(1,n-3)~4 rule; it ignores
      autocorrelation so it is too narrow -- reported as descriptive).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np

S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ECS = f"{S2}/r10round/ecs/runs"
OUT = f"{S2}/com_revisit/notes/verify_posture"
R2D = 180.0 / math.pi


def f2(run, name):
    return f"{S1}/foot/stance6/fetch/F2/stance6_r10LTSB_{run}/copies/01_stance/{name}.copy1"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def ledger(path):
    by = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by[int(r["timesteps"])].append(r)
    ts = sorted(by)
    return ts, by


def lmean(by, t, col, full_only=True):
    xs = []
    for r in by[t]:
        if full_only and num(r.get("length")) < 1000:
            continue
        v = num(r.get(col))
        if not math.isnan(v):
            xs.append(v)
    return float(np.mean(xs)) if xs else float("nan")


def overbar(by, t, act):
    return sum(1 for r in by[t] if r.get("saturated_actuator") == act and num(r.get("saturated_actuator_fraction")) > 0.10)


def onsets(ts, by, act):
    cnt = [overbar(by, t, act) for t in ts]
    first = next((t for t, c in zip(ts, cnt) if c > 0), None)
    flagged = [t for t, c in zip(ts, cnt) if c >= 5]
    alarm = None
    for i in range(1, len(ts)):
        if cnt[i] >= 5 and cnt[i - 1] >= 5:
            alarm = ts[i]
            break
    return dict(first_overbar=first, first_flag=flagged[0] if flagged else None, first_alarm=alarm,
                n_overbar=int(sum(cnt)))


def probe(path):
    d = np.load(path, allow_pickle=False)
    m = d["obs_rms_mean"].astype(np.float64)
    c = d["obs_rms_count"].astype(np.float64)
    t = d["update_timesteps"].astype(np.int64)
    # my own recovery: explicit loop, initial running state (mean 0, count 1e-4 = SB3 RunningMeanStd epsilon)
    bm = np.empty_like(m)
    pm, pc = np.zeros(m.shape[1]), 1e-4
    for i in range(len(t)):
        bm[i] = (m[i] * c[i] - pm * pc) / (c[i] - pc)
        pm, pc = m[i], c[i]
    return dict(t=t, bm=bm, mu=d["rollout_mu_mean"].astype(np.float64), dc=np.diff(np.concatenate([[1e-4], c])))


def train_eps(path):
    t, com, pit = [], [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if num(r["length"]) < 1000:
                continue
            t.append(int(r["timesteps"]))
            com.append(num(r["com_fore_aft_mean_m"]) * 1000)
            pit.append(num(r["root_pitch_mean_deg"]))
    return np.array(t), np.array(com), np.array(pit)


def binmed(t, y, lo, hi):
    sel = (t > lo) & (t <= hi) & ~np.isnan(y)
    return float(np.median(y[sel])) if sel.any() else float("nan"), int(sel.sum())


def binmean(t, y, lo, hi):
    sel = (t > lo) & (t <= hi) & ~np.isnan(y)
    return float(np.mean(y[sel])) if sel.any() else float("nan"), int(sel.sum())


def hinge(t, y, lo, hi, step=8192):
    sel = (t >= lo) & (t <= hi) & ~np.isnan(y)
    tt, yy = t[sel].astype(float), y[sel]
    n = len(tt)
    best = None
    sse = {}
    for t0 in np.arange(lo, hi - 4 * step, step):
        X = np.column_stack([np.ones(n), np.maximum(0, tt - t0)])
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ coef
        s = float(r @ r)
        sse[int(t0)] = (s, float(coef[1]))
        if best is None or s < best[1]:
            best = (int(t0), s, float(coef[1]))
    thr = best[1] * (1 + 4.0 / max(n - 3, 1))
    ok = [k for k, (s, _b) in sse.items() if s <= thr]
    return dict(t0=best[0], slope_per_M=best[2] * 1e6, lo=min(ok), hi=max(ok), n=n)


def prop_time(t, y, lo, hi, fracs=(0.1, 0.25, 0.5), ref_lo=524288, ref_hi=1048576, end_val=None, smooth=8):
    """Step at which the smoothed y first completes each fraction of its change from the reference median
    (ref window) to end_val (default: the smoothed value at hi), searching (ref_hi, hi]."""
    ys = np.convolve(np.nan_to_num(y), np.ones(smooth) / smooth, mode="full")[: len(y)]
    ref = float(np.median(y[(t >= ref_lo) & (t < ref_hi)]))
    if end_val is None:
        end_val = float(ys[np.searchsorted(t, hi, side="right") - 1])
    tot = end_val - ref
    out = []
    for f in fracs:
        target = ref + f * tot
        sel = np.where((t > ref_hi) & (t <= hi) & ((ys - target) * np.sign(tot) >= 0))[0]
        out.append(int(t[sel[0]]) if len(sel) else None)
    return ref, end_val, out


def fm(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{x/1e6:.3f}M"
    return f"{x:+.{nd}f}"


L = []
P = L.append
P("VERIFY TEST 1: own re-derivation (vt1_timing.py).  Read only.  pitch + = nose up; CoM + = forward (mm).")
P("")
res = {}
data = {}
for run in ("s42", "s60", "s80"):
    ts, by = ledger(f2(run, "eval_episodes.csv"))
    pr = probe(f2(run, "policy_probe.npz"))
    tt, tcom, tpit = train_eps(f2(run, "train_episodes.csv"))
    data[run] = (ts, by, pr, tt, tcom, tpit)

# ---------------------------------------------------------------- 0. sign of pitch from obs, and recovery checks
P("0. RECOVERY AND SIGN CHECKS")
for run in ("s42", "s60", "s80"):
    ts, by, pr, tt, tcom, tpit = data[run]
    gx = pr["bm"][:, 42]
    pitch_a = np.degrees(np.arcsin(np.clip(-gx, -1, 1)))
    # bins of 262,144 from 524,288 to end: probe vs training episodes, both signs
    edges = np.arange(524288, pr["t"][-1] + 1, 262144)
    a, b, q, lq = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a.append(binmean(pr["t"], pitch_a, lo, hi)[0])
        b.append(binmean(tt, tpit, lo, hi)[0])
        q.append(binmean(pr["t"], pr["bm"][:, 6] * R2D, lo, hi)[0])
        lt = [t for t in ts if lo < t <= hi]
        lq.append(np.nanmean([lmean(by, t, "tail_3_pitch_mean_deg") for t in lt]) if lt else np.nan)
    a, b, q, lq = map(np.array, (a, b, q, lq))
    ok = ~np.isnan(a) & ~np.isnan(b)
    r_pos = np.corrcoef(a[ok], b[ok])[0, 1]
    ok2 = ~np.isnan(q) & ~np.isnan(lq)
    r_q = np.corrcoef(q[ok2], lq[ok2])[0, 1]
    gy = np.abs(pr["bm"][:, 43]).max()
    nonsym = np.abs(pr["bm"][:, 7] - pr["bm"][:, 14]).max()
    P(f"  {run}: probe pitch asin(-g_x) vs train-episode pitch, 262,144 bins n={ok.sum()}: r {r_pos:+.3f}, "
      f"median diff {np.median(a[ok]-b[ok]):+.3f} deg (sign asin(+g_x) would give r {-r_pos:+.3f}); "
      f"probe tail_3 angle vs ledger tail_3 angle r {r_q:+.3f} (n={ok2.sum()}); max|batch g_y| {gy:.1e}; "
      f"max|r_hip - l_hip batch| {nonsym:.1e}")
    res.setdefault(run, {})["recovery"] = dict(r_pitch=r_pos, r_tail=r_q)
P("")

# ---------------------------------------------------------------- 1. onsets
P("1. ONSETS FROM THE LEDGER (M5 bar: fraction > 0.10; flag >= 5 of 10; alarm = 2nd consecutive flag)")
for run in ("s42", "s60", "s80"):
    ts, by, pr, *_ = data[run]
    o = onsets(ts, by, "tail_3_pitch_act")
    on = onsets(ts, by, "neck_pitch_act")
    mu14 = pr["mu"][:, 14]
    sm = np.convolve(mu14, np.ones(8) / 8, mode="full")[: len(mu14)]
    P(f"  {run}: tail_3 first over-bar {fm(o['first_overbar'])}, first flag {fm(o['first_flag'])}, first alarm "
      f"{fm(o['first_alarm'])}, over-bar rows {o['n_overbar']};  neck_pitch first over-bar {fm(on['first_overbar'])}, "
      f"flag {fm(on['first_flag'])}, alarm {fm(on['first_alarm'])}, rows {on['n_overbar']}")
    res[run]["onsets"] = o
P("")

# ---------------------------------------------------------------- 2. s42 fine time course 0.9-2.4M
P("2. r10 s42 time course around the rise (131,072-step bins ending at the step shown; probe = mean of 16 rollouts;")
P("   train = median of full-horizon training episodes; ledger = mean of 10 deterministic episodes at that step)")
ts, by, pr, tt, tcom, tpit = data["s42"]
pitch_p = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
P("   step   mu14_probe  pitch_probe  q3_probe  pitch_train  com_train(n)   L_tcmd  L_pitch   L_com   L_head  L_neck")
for end in range(917504, 2490368 + 1, 131072):
    lo = end - 131072
    m14, _ = binmean(pr["t"], pr["mu"][:, 14], lo, end)
    pp, _ = binmean(pr["t"], pitch_p, lo, end)
    q3, _ = binmean(pr["t"], pr["bm"][:, 6] * R2D, lo, end)
    tp, _ = binmed(tt, tpit, lo, end)
    tc, nc = binmed(tt, tcom, lo, end)
    P(f"  {end/1e6:5.3f}  {m14:+9.3f}  {pp:+10.3f}  {q3:+8.3f}  {tp:+10.3f}  {tc:+8.2f}({nc:3d})  "
      f"{lmean(by, end, 'tail_3_pitch_act_command_mean'):+7.3f}  {lmean(by, end, 'root_pitch_mean_deg'):+7.3f}  "
      f"{lmean(by, end, 'com_fore_aft_mean_m')*1000:+7.2f}  {lmean(by, end, 'head_tip_height_mean_m'):6.3f}  "
      f"{lmean(by, end, 'neck_pitch_mean_deg'):+6.2f}")
P("")

# ---------------------------------------------------------------- 3. change points and proportional timing
P("3. CHANGE-POINT (hinge) FITS, window [0.524M, END]; t0 = start of the linear change; [approx. range]")
cp = {}
for run, END in (("s42", 3670016), ("s42_reformed", None), ("s60", 8126464), ("s80", None)):
    if run == "s80":
        continue
    rr = run[:3]
    ts, by, pr, tt, tcom, tpit = data[rr]
    pitch_p = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
    if run == "s42_reformed":
        lo, END = 6815744, 7602176
    elif run == "s60":
        lo = 5505024
    else:
        lo = 524288
    # training-episode series at 8,192 resolution: per-rollout median of episodes ending in that rollout
    sig = {"tail mu (probe)": (pr["t"], pr["mu"][:, 14]), "pitch (probe)": (pr["t"], pitch_p),
           "q3 angle (probe)": (pr["t"], pr["bm"][:, 6] * R2D),
           "neck joint (probe)": (pr["t"], pr["bm"][:, 0] * R2D), "head joint (probe)": (pr["t"], pr["bm"][:, 2] * R2D)}
    g = defaultdict(list)
    for a_, c_, p_ in zip(tt, tcom, tpit):
        g[int(math.ceil(a_ / 65536.0) * 65536)].append((c_, p_))
    gt = np.array(sorted(g))
    sig["CoM (train, 65,536 bins)"] = (gt, np.array([np.median([x[0] for x in g[k]]) for k in gt]))
    sig["pitch (train, 65,536 bins)"] = (gt, np.array([np.median([x[1] for x in g[k]]) for k in gt]))
    lt = np.array(ts)
    sig["tail cmd (ledger)"] = (lt, np.array([lmean(by, t, "tail_3_pitch_act_command_mean") for t in ts]))
    sig["pitch (ledger)"] = (lt, np.array([lmean(by, t, "root_pitch_mean_deg") for t in ts]))
    sig["CoM (ledger)"] = (lt, np.array([lmean(by, t, "com_fore_aft_mean_m") * 1000 for t in ts]))
    P(f"  {run}: window [{lo/1e6:.3f}M, {END/1e6:.3f}M]")
    cp[run] = {}
    for name, (t_, y_) in sig.items():
        stp = 131072 if "ledger" in name else (65536 if "train" in name else 8192)
        h = hinge(t_, y_, lo, END, step=stp)
        cp[run][name] = h
        P(f"    {name:28s} t0 {h['t0']/1e6:6.3f}M [{h['lo']/1e6:.3f}, {h['hi']/1e6:.3f}]  slope {h['slope_per_M']:+.3f}/M  n {h['n']}")
P("")
P("3b. PROPORTIONAL TIMING (probe, 8-rollout trailing mean), fractions 10/25/50% of the change from the reference")
P("    median [0.524M,1.049M) to the value at END; END varied to show sensitivity")
ts, by, pr, tt, tcom, tpit = data["s42"]
pitch_p = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
for END in (3145728, 3407872, 3670016, 3932160):
    r1 = prop_time(pr["t"], pr["mu"][:, 14], 0, END)
    r2 = prop_time(pr["t"], pitch_p, 0, END)
    P(f"  s42 END {END/1e6:.3f}M: tail mu ref {r1[0]:+.3f} end {r1[1]:+.3f} -> " + " / ".join(fm(x) for x in r1[2])
      + f";  pitch ref {r2[0]:+.3f} end {r2[1]:+.3f} -> " + " / ".join(fm(x) for x in r2[2]))
P("")

# ---------------------------------------------------------------- 4. lagged cross-correlation of first differences
P("4. FIRST-DIFFERENCE CROSS-CORRELATION, probe tail mu vs probe pitch, 65,536-step bins (8 rollouts), window")
P("   [0.524M, 3.67M] for s42; r(lag k) = corr(d tail[t], d pitch[t+k]) (k > 0: pitch moves after tail)")
for run, lo, hi in (("s42", 524288, 3670016), ("s60", 3145728, 8126464), ("s80", 524288, 8126464)):
    ts, by, pr, *_ = data[run]
    pitch_p = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
    edges = np.arange(lo, hi + 1, 65536)
    a = np.array([binmean(pr["t"], pr["mu"][:, 14], x, y)[0] for x, y in zip(edges[:-1], edges[1:])])
    b = np.array([binmean(pr["t"], pitch_p, x, y)[0] for x, y in zip(edges[:-1], edges[1:])])
    da, db = np.diff(a), np.diff(b)
    row = []
    for k in range(-4, 5):
        if k >= 0:
            x, y = da[: len(da) - k], db[k:]
        else:
            x, y = da[-k:], db[: len(db) + k]
        row.append(f"{k:+d}:{np.corrcoef(x, y)[0,1]:+.2f}")
    P(f"  {run} [{lo/1e6:.2f},{hi/1e6:.2f}] n={len(da)}: " + "  ".join(row))
P("")

# ---------------------------------------------------------------- 5. pre-rise lineage gap
P("5. PRE-RISE GAP: s42 minus s80 (and s42 minus s60), windows (lo, hi]")
for lo, hi in ((524288, 786432), (786432, 1048576), (524288, 1048576), (1048576, 1572864), (2097152, 3145728),
               (3145728, 3670016)):
    vals = {}
    for run in ("s42", "s60", "s80"):
        ts, by, pr, tt, tcom, tpit = data[run]
        pitch_p = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
        lts = [t for t in ts if lo < t <= hi]
        vals[run] = dict(
            Lp=np.mean([lmean(by, t, "root_pitch_mean_deg") for t in lts]),
            Lc=np.mean([lmean(by, t, "com_fore_aft_mean_m") for t in lts]) * 1000,
            Tp=binmed(tt, tpit, lo, hi)[0], Tc=binmed(tt, tcom, lo, hi)[0], Pp=binmean(pr["t"], pitch_p, lo, hi)[0],
            mu=binmean(pr["t"], pr["mu"][:, 14], lo, hi)[0])
    d8 = {k: vals["s42"][k] - vals["s80"][k] for k in vals["s42"]}
    d6 = {k: vals["s42"][k] - vals["s60"][k] for k in vals["s42"]}
    P(f"  ({lo/1e6:.3f},{hi/1e6:.3f}]  s42-s80: pitch ledger {d8['Lp']:+.2f} train {d8['Tp']:+.2f} probe {d8['Pp']:+.2f} deg;"
      f" CoM ledger {d8['Lc']:+.1f} train {d8['Tc']:+.1f} mm; mu14 {d8['mu']:+.3f}"
      f"  | s42-s60: pitch L {d6['Lp']:+.2f} T {d6['Tp']:+.2f} P {d6['Pp']:+.2f}; CoM L {d6['Lc']:+.1f} T {d6['Tc']:+.1f}")
P("")

# ---------------------------------------------------------------- 6. s60 and s80 episodes
P("6. s60 / s80 EPISODES: start -> end values over the window (probe means over 16 rollouts at each end; train")
P("   medians over 131,072 at each end; ledger mean of the 2 evaluations at each end)")


def window_change(run, a, b):
    ts, by, pr, tt, tcom, tpit = data[run]
    pitch_p = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
    w = 131072
    out = {}
    out["mu14"] = (binmean(pr["t"], pr["mu"][:, 14], a - w, a)[0], binmean(pr["t"], pr["mu"][:, 14], b - w, b)[0])
    out["pitch_probe"] = (binmean(pr["t"], pitch_p, a - w, a)[0], binmean(pr["t"], pitch_p, b - w, b)[0])
    out["pitch_train"] = (binmed(tt, tpit, a - w, a)[0], binmed(tt, tpit, b - w, b)[0])
    out["com_train"] = (binmed(tt, tcom, a - w, a)[0], binmed(tt, tcom, b - w, b)[0])
    out["neck_probe"] = (binmean(pr["t"], pr["bm"][:, 0] * R2D, a - w, a)[0], binmean(pr["t"], pr["bm"][:, 0] * R2D, b - w, b)[0])

    def lw(col, x, scale=1.0):
        lts = [t for t in ts if x - 2 * 131072 < t <= x]
        return np.mean([lmean(by, t, col) for t in lts]) * scale

    out["L_tcmd"] = (lw("tail_3_pitch_act_command_mean", a), lw("tail_3_pitch_act_command_mean", b))
    out["L_pitch"] = (lw("root_pitch_mean_deg", a), lw("root_pitch_mean_deg", b))
    out["L_com"] = (lw("com_fore_aft_mean_m", a, 1000), lw("com_fore_aft_mean_m", b, 1000))
    out["L_head"] = (lw("head_tip_height_mean_m", a), lw("head_tip_height_mean_m", b))
    out["L_neck"] = (lw("neck_pitch_mean_deg", a), lw("neck_pitch_mean_deg", b))
    return out


for run, a, b, label in (("s60", 6029312, 7340032, "s60 main rise (6.03 -> 7.34M)"),
                         ("s60", 7471104, 8126464, "s60 final approach (7.47 -> 8.13M)"),
                         ("s60", 4980736, 5505024, "s60 5.0-5.5M (claimed nose-down / CoM fwd, no tail rise)"),
                         ("s60", 1048576, 4456448, "s60 1.05 -> 4.46M (claimed head raised before tail)"),
                         ("s80", 2359296, 3604480, "s80 bump 1 (2.36 -> 3.60M)"),
                         ("s80", 5636096, 6553600, "s80 bump 2 (5.64 -> 6.55M)"),
                         ("s80", 6553600, 7077888, "s80 after bump 2 (6.55 -> 7.08M)"),
                         ("s42", 4718592, 6291456, "s42 decoupling (4.72 -> 6.29M)"),
                         ("s42", 6815744, 7340032, "s42 re-formed rail (6.82 -> 7.34M)")):
    ch = window_change(run, a, b)
    P(f"  {label}:")
    P("    " + "; ".join(f"{k} {v[0]:+.3f}->{v[1]:+.3f} ({v[1]-v[0]:+.3f})" for k, v in ch.items()))
P("")

# ---------------------------------------------------------------- 7. ECS continuations
P("7. ECS CONTINUATIONS (ledger): from S (the parent's evaluation at S) to the last evaluation before the first")
P("   tail over-bar episode (PRE), and S to the end (END).  Means over the 10 deterministic episodes per evaluation.")
cols = [("tail_3_pitch_act_command_mean", 1, "tcmd"), ("root_pitch_mean_deg", 1, "pitch"),
        ("com_fore_aft_mean_m", 1000, "CoM"), ("head_tip_height_mean_m", 1, "head"), ("neck_pitch_mean_deg", 1, "neck"),
        ("head_pitch_mean_deg", 1, "headj")]
ecs = {}
for lin, S in (("s42", 3145728), ("s80", 4194304)):
    names = sorted(n for n in __import__("os").listdir(ECS) if n.startswith(lin))
    pre_all, end_all = defaultdict(list), defaultdict(list)
    P(f"  lineage {lin}, S = {S:,}")
    for nm in names:
        ts, by = ledger(f"{ECS}/{nm}/01_stance/eval_episodes.csv")
        cont = [t for t in ts if t > S]
        o = onsets([S] + cont, by, "tail_3_pitch_act")
        fo = o["first_overbar"]
        pre_t = None
        if fo is not None:
            prev = [t for t in [S] + cont if t < fo]
            pre_t = prev[-1]
        base = {k: lmean(by, S, c) * sc for c, sc, k in cols}
        end = {k: lmean(by, cont[-1], c) * sc for c, sc, k in cols}
        pre = {k: lmean(by, pre_t, c) * sc for c, sc, k in cols} if pre_t else None
        for k in base:
            end_all[k].append(end[k] - base[k])
            if pre and pre_t != S:
                pre_all[k].append(pre[k] - base[k])
        ecs[nm] = dict(first_overbar=fo, pre_t=pre_t, n_cont=len(cont))
        P(f"    {nm}: first over-bar {fm(fo)}; PRE at {fm(pre_t)}: " +
          (", ".join(f"{k} {pre[k]-base[k]:+.3f}" for k in base) if pre else "-") +
          f" | END {fm(cont[-1])}: " + ", ".join(f"{k} {end[k]-base[k]:+.3f}" for k in base))
    P(f"    S values: " + ", ".join(f"{k} {v:+.3f}" for k, v in base.items()))
    for lab, D in (("PRE", pre_all), ("END", end_all)):
        if D:
            P(f"    {lab} medians (n {len(D['tcmd'])}): " + ", ".join(
                f"{k} {np.median(v):+.3f} (+{sum(1 for x in v if x > 0)}/-{sum(1 for x in v if x < 0)})" for k, v in D.items()))
P("")

with open(f"{OUT}/vt1_timing.txt", "w") as fh:
    fh.write("\n".join(L) + "\n")
with open(f"{OUT}/vt1_timing.json", "w") as fh:
    json.dump(dict(cp=cp, ecs=ecs, res={k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}), fh, default=str,
              indent=1)
print("\n".join(L))
