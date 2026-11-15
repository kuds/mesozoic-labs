"""Lead's own implementation of C3 as amended on 15 Nov (STAGE2_75_PLAN.md), written from the rule text, compared with
designlab.stage2.readings.toe_clearance_own_swing on random synthetic traces.  Scratch check; no record."""
import math, sys
import numpy as np
from designlab.stage2 import readings as R

DT, EVERY = 0.002, 5
EPS = 1e-9


def mine(t, low, contact, ref_swing, ta, tb, gap_max, planned):
    shortest = gap_max / 0.5
    t_last = t[-1]
    out = dict(judged=[], first=[], own=[0, 0], refs=[0, 0], interrupted=0, gaps=0)
    for foot in (0, 1):
        # reference swings of this foot: maximal row runs
        rr, k = [], 0
        while k < len(t):
            if ref_swing[k] == foot:
                j = k
                while j < len(t) and ref_swing[j] == foot:
                    j += 1
                vals = [p for p in planned[k:j] if math.isfinite(p) and p > 0]
                rr.append(dict(a=k, b=j, T=vals[0] if vals else shortest, n=0, cut=False))
                k = j
            else:
                k += 1
        # own air runs: complete only
        c = contact[:, foot]
        s = 1
        while s < len(c):
            if (not c[s]) and c[s - 1]:
                e = s
                while e < len(c) and not c[e]:
                    e += 1
                if e < len(c):
                    t_lo, t_td = s * DT, e * DT
                    shared = []
                    for r in rr:
                        n = sum(1 for q in range(r["a"], r["b"]) if t_lo - EPS <= t[q] < t_td - EPS)
                        shared.append(n)
                    is_swing = any(shared) or (t_td - t_lo) >= gap_max - EPS
                    T_p = rr[int(np.argmax(shared))]["T"] if any(shared) else shortest
                    t1, t2 = t_lo + 0.25 * T_p, t_lo + 0.75 * T_p
                    inside = t_lo >= ta - EPS and t_td <= tb + EPS
                    read = is_swing and inside and t2 <= tb + EPS and t2 <= t_last + EPS
                    if is_swing:
                        for r, n in zip(rr, shared):
                            if n:
                                r["n"] += 1
                                r["cut"] = r["cut"] or not read
                    if read:
                        vals = [np.interp(t1, t, low[:, foot]), np.interp(t2, t, low[:, foot])]
                        vals += [low[q, foot] for q in range(len(t)) if t1 < t[q] < t2]
                        out["judged"].append(min(vals)); out["own"][foot] += 1
                    if is_swing and inside:
                        D = t_td - t_lo
                        u1, u2 = t_lo + 0.25 * D, t_lo + 0.75 * D
                        vals = [np.interp(u1, t, low[:, foot]), np.interp(u2, t, low[:, foot])]
                        vals += [low[q, foot] for q in range(len(t)) if u1 < t[q] < u2]
                        out["first"].append(min(vals))
                    if inside and not is_swing:
                        out["gaps"] += 1
                s = e
            else:
                s += 1
        for r in rr:
            whole = r["a"] > 0 and r["b"] < len(t) and t[r["a"]] >= ta - EPS and t[r["b"] - 1] <= tb + EPS
            if whole:
                out["interrupted"] += r["n"] >= 2
                if not r["cut"]:
                    out["refs"][foot] += 1
    out["count_ok"] = all(out["own"][i] >= out["refs"][i] - 1 for i in (0, 1))
    out["judged_min"] = min(out["judged"]) if out["judged"] else None
    out["first_min"] = min(out["first"]) if out["first"] else None
    return out


def trace(rng):
    n_rows = int(rng.integers(700, 1100))
    t = np.arange(n_rows) * EVERY * DT
    steps = n_rows * EVERY
    T = rng.uniform(0.6, 1.0); duty = rng.uniform(0.55, 0.70); Ts = (1 - duty) * T
    ref = np.full(n_rows, -1); planned = np.full(n_rows, math.nan)
    contact = np.ones((steps, 2), bool); low = np.zeros((n_rows, 2))
    give_planned = rng.random() < 0.8
    Ts_min = Ts
    for foot in (0, 1):
        k = 0
        while True:
            Tsk = Ts * (rng.uniform(0.9, 1.1) if rng.random() < 0.3 else 1.0)
            Ts_min = min(Ts_min, Tsk)
            r0 = 0.3 + foot * T / 2 + k * T; r1 = r0 + Tsk
            if r1 > t[-1] - 0.2:
                break
            rows = np.flatnonzero((t >= r0) & (t < r1))
            ref[rows] = foot
            if give_planned:
                planned[rows] = Tsk
            lo = r0 + rng.uniform(-0.03, 0.05); td = r1 + (rng.uniform(-0.15, 0.06) if rng.random() < 0.5 else rng.uniform(0.0, 0.03))
            a, b = int(round(lo / DT)), int(round(td / DT))
            if b - a > 12:
                contact[a:b, foot] = False
                if rng.random() < 0.12:   # a mid-swing stub
                    m = (a + b) // 2
                    contact[m:m + int(rng.integers(11, 25)), foot] = True
                lift = rng.uniform(0.025, 0.06)
                sel = np.flatnonzero((t >= a * DT) & (t <= b * DT))
                low[sel, foot] = lift * np.sin(np.pi * (t[sel] - a * DT) / ((b - a) * DT)) ** 2
            if rng.random() < 0.12:       # stance chatter: a short air run in stance
                g = int(round((r1 + rng.uniform(0.08, 0.2)) / DT)); L = int(rng.integers(11, 60))
                if g + L < steps - 5 and contact[g - 1:g + L + 1, foot].all():
                    contact[g:g + L, foot] = False
            k += 1
    low += rng.normal(0, 2e-4, low.shape)
    ta = rng.uniform(0.5, 2.5); tb = ta + rng.uniform(2.5, 4.5)
    tb = min(tb, t[-1] + (0.5 if rng.random() < 0.2 else -0.05))
    return t, low, contact, ref, ta, tb, 0.5 * Ts_min, planned


rng = np.random.default_rng(15112026)
N, bad, read_some, cuts, interr, gaps, countfail = int(sys.argv[1]), 0, 0, 0, 0, 0, 0
for it in range(N):
    t, low, contact, ref, ta, tb, gmax, planned = trace(rng)
    a = R.toe_clearance_own_swing(t, low, contact, DT, ref, ta, tb, gmax, planned_s=planned)
    b = mine(t, low, contact, ref, ta, tb, gmax, planned)
    same = (a["own_swings"] == b["own"] and a["ref_swings"] == b["refs"] and a["interrupted_swings"] == b["interrupted"]
            and a["air_gaps"] == b["gaps"] and a["count_ok"] == b["count_ok"]
            and ((a["toe_clearance_own_swing_m"] is None) == (b["judged_min"] is None))
            and (a["toe_clearance_own_swing_m"] is None or abs(a["toe_clearance_own_swing_m"] - b["judged_min"]) < 1e-12)
            and ((a["toe_clearance_own_duration_m"] is None) == (b["first_min"] is None))
            and (a["toe_clearance_own_duration_m"] is None or abs(a["toe_clearance_own_duration_m"] - b["first_min"]) < 1e-12))
    if not same:
        bad += 1
        if bad <= 3:
            print("DIFF", it, {k: a[k] for k in ("toe_clearance_own_swing_m", "own_swings", "ref_swings", "interrupted_swings", "air_gaps", "count_ok", "toe_clearance_own_duration_m")}, {k: b[k] for k in ("judged_min", "own", "refs", "interrupted", "gaps", "count_ok", "first_min")})
    read_some += b["judged_min"] is not None; cuts += sum(a["ref_swings_cut"]) > 0; interr += b["interrupted"] > 0
    gaps += b["gaps"] > 0; countfail += not b["count_ok"]
print(f"{N} traces: {bad} differ; with a reading {read_some}; with a reference swing left out for a cut {cuts}; "
      f"with an interrupted swing {interr}; with an air gap {gaps}; count rule failing {countfail}")
