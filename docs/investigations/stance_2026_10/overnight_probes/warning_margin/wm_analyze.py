"""MARGIN-TO-RAIL TEST: analysis of episodes.jsonl against PREREG.txt (criteria C1, C2, Q1, Q3; descriptive D1-D5).
No simulation.  Writes wm_analyze.txt and wm_summary.json."""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

import wm_common as C

ORDER = [("s80", 10_485_760), ("s80", 11_534_336), ("s80", 12_001_280), ("s60", 6_291_456), ("s60", 7_340_032),
         ("s42", 2_097_152), ("s42", 3_145_728)]
LABEL = {("s80", 10_485_760): "s80 10.49M", ("s80", 11_534_336): "s80 11.53M", ("s80", 12_001_280): "s80 12.00M",
         ("s60", 6_291_456): "s60 6.29M*", ("s60", 7_340_032): "s60 7.34M+", ("s42", 2_097_152): "s42 2.10M*",
         ("s42", 3_145_728): "s42 3.15M+"}
SEEDS = list(range(3042, 3062))
K_ORDER = 11  # the 11th smallest of 20
BIAS_HI, OBS_HI = 2.0, 12.0
T_RAIL = 5.19
PLANT = 5.16
rng = np.random.default_rng(20261030)

pred = json.loads((C.HERE / "wm_predict.json").read_text())
stat = json.loads((C.HERE / "wm_static.json").read_text())
eps = [json.loads(x) for x in (C.HERE / "episodes.jsonl").read_text().splitlines() if x.strip()]
by = defaultdict(list)
for e in eps:
    by[(e["run"], e["ckpt_steps"], e["seed"])].append(e)

L: list[str] = []


def P(s=""):
    L.append(s)


def rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rank(a[ok]), rank(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def kth(vals, k=K_ORDER):
    v = sorted(vals)
    return v[k - 1] if len(v) >= k else float("nan")


def boot_kth(vals, n=2000):
    v = np.array(vals, dtype=float)
    out = []
    for _ in range(n):
        s = np.sort(rng.choice(v, size=len(v), replace=True))
        out.append(s[K_ORDER - 1])
    out = np.array(out)
    fin = out[np.isfinite(out)]
    share_inf = 1.0 - len(fin) / len(out)
    if share_inf > 0.025:
        hi = float("inf")
    else:
        hi = float(np.quantile(out, 0.975))
    lo = float(np.quantile(out, 0.025)) if share_inf < 0.975 else float("inf")
    return lo, hi


def seed_threshold(rows, kind, hi_default):
    """T_s, bracket lower end, monotonicity violation flag, visited (value, parked) pairs."""
    base = [r for r in rows if r["kind"] == "none"]
    pts = [(0.0, base[0]["parked"])] if base else []
    pts += [(r["value"], r["parked"]) for r in rows if r["kind"] == kind]
    parked = [v for v, p in pts if p]
    unpark = [v for v, p in pts if not p]
    if parked:
        T = min(parked)
    else:
        T = hi_default  # bias: 2.0 by construction; obs: inf (12 deg not parked)
    lo = max([v for v in unpark if v < T], default=0.0)
    viol = any(pv < uv for pv in parked for uv in unpark)
    return T, lo, viol, sorted(pts)


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "unreach"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


summary = {"n_episodes": len(eps), "ckpts": {}}
P("=" * 118)
P("MARGIN-TO-RAIL TEST: analysis (design and criteria: PREREG.txt).  * = two checkpoints before FOB, + = one before.")
P("PARKED = > 250 of the 500 window steps (501-1000) with command >= 0.99.  T_s = smallest visited value that parks.")
P(f"M = the {K_ORDER}th smallest T_s of the 20 seeds (3042-3061).  [lo, hi] = bootstrap 95% over seeds of M.")
P("=" * 118)
P(f"episodes in file: {len(eps)}; terminated before 1,000 steps: {sum(e['terminated'] for e in eps)}; "
  f"CPU s per episode median {np.median([e['cpu_s'] for e in eps]):.2f}")
P()
P("1. BASELINE (no nudge), seeds 3042-3061, against the early-warning readouts (seeds 5001-5010)")
P(f"  {'ckpt':12s} | base cmd med [min,max]       | own14(ew) FP(ew) | base tail3 deg med | parked at 0 | frac99>0.10")
for ck in ORDER:
    rows = [by[(ck[0], ck[1], s)] for s in SEEDS if by[(ck[0], ck[1], s)]]
    base = [[r for r in rr if r["kind"] == "none"][0] for rr in rows]
    if not base:
        continue
    cm = [b["cmd_win"] for b in base]
    p = pred[f"{ck[0]}_{ck[1]}"]
    P(f"  {LABEL[ck]:12s} | {np.median(cm):+.3f} [{min(cm):+.3f},{max(cm):+.3f}] (n {len(cm)}) | {p['own14']:+.3f}  {p['FP']:+.3f} | "
      f"{np.median([b['tail3_deg_win'] for b in base]):+.2f}              | {sum(b['parked'] for b in base)}/{len(base)}        | "
      f"{sum(b['frac99_win'] > 0.10 for b in base)}/{len(base)}")
    summary["ckpts"].setdefault(LABEL[ck], {})["base_cmd_median"] = float(np.median(cm))
P()

res = {}
for kind, hi_def, unit in (("bias", BIAS_HI, ""), ("obs", float("inf"), " deg")):
    P(f"2{'a' if kind == 'bias' else 'b'}. {kind.upper()} NUDGE: per-seed thresholds and the checkpoint margin M")
    P(f"  {'ckpt':12s} | n  | M{unit:4s}  [boot 95%]          | median mid | min    max     | parked at top | monot. viol | per-seed T_s (sorted)")
    for ck in ORDER:
        Ts, mids, viols, ntop = [], [], 0, 0
        for s in SEEDS:
            rr = by[(ck[0], ck[1], s)]
            if not rr:
                continue
            T, lo, viol, pts = seed_threshold(rr, kind, hi_def)
            Ts.append(T)
            mids.append((T + lo) / 2 if math.isfinite(T) else float("inf"))
            viols += viol
            if kind == "obs":
                top = [r for r in rr if r["kind"] == "obs" and r["value"] == OBS_HI]
                ntop += bool(top and top[0]["parked"])
        if not Ts:
            continue
        M = kth(Ts) if len(Ts) == 20 else float("nan")
        b_lo, b_hi = boot_kth(Ts) if len(Ts) == 20 else (float("nan"), float("nan"))
        res[(kind, ck)] = {"M": M, "boot": (b_lo, b_hi), "Ts": Ts, "median_mid": float(np.median(mids)), "n": len(Ts),
                           "viol": viols}
        top_s = f"{ntop}/{len(Ts)}" if kind == "obs" else "2.0 by constr."
        P(f"  {LABEL[ck]:12s} | {len(Ts):2d} | {f(M)}  [{f(b_lo)}, {f(b_hi)}] | {f(float(np.median(mids)))}      | "
          f"{f(min(Ts))} {f(max(Ts))} | {top_s:13s} | {viols:2d}          | {' '.join(f(t, 3) for t in sorted(Ts))}")
        summary["ckpts"].setdefault(LABEL[ck], {})[f"M_{kind}"] = M
        summary["ckpts"][LABEL[ck]][f"M_{kind}_boot95"] = [b_lo, b_hi]
        summary["ckpts"][LABEL[ck]][f"T_{kind}_sorted"] = sorted(Ts)
    P()

P("3. C1: BIAS margin against the predictors (PREREG section 3).  match = |M - pred| <= max(0.05, 0.20 pred)")
names = [("bstar_P", "b*_P (verifier)"), ("bstar_L", "b*_L"), ("bstar_S", "b*_S (static)"), ("gap_open", "1 - own14")]
P(f"  {'ckpt':12s} | M_bias | " + " | ".join(f"{n:>16s} err  ok" for _, n in names))
cks = [ck for ck in ORDER if ("bias", ck) in res and math.isfinite(res[("bias", ck)]["M"])]
Mv = np.array([res[("bias", ck)]["M"] for ck in cks])
for ck in cks:
    p = pred[f"{ck[0]}_{ck[1]}"]
    M = res[("bias", ck)]["M"]
    cells = []
    for key, _ in names:
        v = p[key]
        ok = abs(M - v) <= max(0.05, 0.20 * v)
        cells.append(f"{v:16.3f} {M - v:+.3f} {'Y' if ok else 'n'}")
    P(f"  {LABEL[ck]:12s} | {M:.3f}  | " + " | ".join(cells))
P()
c1 = {}
for key, nm in names:
    pv = np.array([pred[f"{ck[0]}_{ck[1]}"][key] for ck in cks])
    n_ok = int(sum(abs(m - v) <= max(0.05, 0.20 * v) for m, v in zip(Mv, pv)))
    rho = spearman(Mv, pv)
    mae = float(np.mean(np.abs(Mv - pv)))
    mse = float(np.mean(Mv - pv))
    if len(cks) == 7:
        verdict = ("MATCHES" if (n_ok >= 6 and rho >= 0.9) else
                   ("ranks right, scale wrong" if rho >= 0.9 else "does not match"))
    else:
        verdict = f"incomplete ({len(cks)}/7 checkpoints)"
    c1[key] = {"n_match": n_ok, "rho": rho, "mae": mae, "mean_signed_err": mse, "verdict": verdict}
    P(f"  {nm:16s}: {n_ok}/{len(cks)} match | Spearman rho {rho:+.3f} | MAE {mae:.3f} | mean signed error (M - pred) "
      f"{mse:+.3f} | {verdict}")
for key, nm in (("G_P", "G_P (gain)"), ("FP", "FP"), ("c0", "c0"), ("own14", "own14")):
    pv = np.array([pred[f"{ck[0]}_{ck[1]}"][key] for ck in cks])
    P(f"  {nm:16s}: Spearman rho with M_bias {spearman(Mv, pv):+.3f} (ranking only; a perfect inverse ranking is -1)")
summary["C1"] = c1
P()

P("4. Q1: late s80 against the pre-FOB pair (BIAS).  AS CLOSE iff M(s80) <= max(M(s42 3.15M), M(s60 7.34M)) + 0.05")
pre = [("s42", 3_145_728), ("s60", 7_340_032)]
if all(("bias", c) in res for c in pre):
    ref = max(res[("bias", c)]["M"] for c in pre)
    for c in pre:
        r = res[("bias", c)]
        P(f"  pre-FOB {LABEL[c]:12s}: M {f(r['M'])} [{f(r['boot'][0])}, {f(r['boot'][1])}]")
    q1 = {}
    for c in [("s80", 10_485_760), ("s80", 11_534_336), ("s80", 12_001_280)]:
        if ("bias", c) not in res:
            continue
        r = res[("bias", c)]
        v = "AS CLOSE" if r["M"] <= ref + 0.05 else "FARTHER"
        q1[LABEL[c]] = v
        P(f"  {LABEL[c]:12s}: M {f(r['M'])} [{f(r['boot'][0])}, {f(r['boot'][1])}]  vs max pre-FOB {ref:.3f} + 0.05 -> {v}")
    summary["Q1"] = q1
P()

P("5. C2: OBS reachability and margin.  Linear P predicts 3.26 deg at s60 7.34M and unreachable elsewhere; static S")
P("   predicts unreachable everywhere.  match(unreachable) = fewer than 11/20 park at 12 deg.")
c2 = {}
for ck in ORDER:
    if ("obs", ck) not in res:
        continue
    r = res[("obs", ck)]
    p = pred[f"{ck[0]}_{ck[1]}"]
    dP = p["delta_star_P"]
    M = r["M"]
    if dP is None:
        okP = not math.isfinite(M)
    else:
        okP = math.isfinite(M) and abs(M - dP) <= max(0.5, 0.2 * dP)
    okS = not math.isfinite(M)
    c2[LABEL[ck]] = {"M_obs": M, "pred_P": dP, "P_ok": okP, "S_ok": okS}
    P(f"  {LABEL[ck]:12s}: M_obs {f(M)} deg | pred P {f(dP) if dP is not None else 'unreach'} -> {'Y' if okP else 'n'} | "
      f"pred S unreach -> {'Y' if okS else 'n'}")
summary["C2"] = c2
P(f"  linear P: {sum(v['P_ok'] for v in c2.values())}/{len(c2)} rows match; static S: {sum(v['S_ok'] for v in c2.values())}/{len(c2)}")
P()

P("6. Q3: which single number best says how close a policy is to the rail (smallest MAE to M_bias, PREREG Q3)")
best = sorted(((c1[k]["mae"], -c1[k]["rho"], k) for k, _ in names))
for mae, nrho, k in best:
    P(f"  {k:9s}: MAE {mae:.3f}, rho {-nrho:+.3f}")
summary["Q3_best"] = best[0][2] if best else None
P()

P("7. D1: closed-loop amplification below the rail (BIAS): slope of window command vs b over visited values with 0")
P("   railed window steps (b = 0 included); G_cl = 1 - 1/slope; median over seeds with >= 2 such values")
d1 = {}
for ck in ORDER:
    gs, ns = [], []
    for s in SEEDS:
        rr = by[(ck[0], ck[1], s)]
        pts = [(0.0, r["cmd_win"]) for r in rr if r["kind"] == "none" and r["n_rail_win"] == 0]
        pts += [(r["value"], r["cmd_win"]) for r in rr if r["kind"] == "bias" and r["n_rail_win"] == 0]
        if len(pts) >= 2:
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            sl = float(np.polyfit(x, y, 1)[0])
            gs.append(1.0 - 1.0 / sl)
            ns.append(len(pts))
    if gs:
        p = pred[f"{ck[0]}_{ck[1]}"]
        d1[LABEL[ck]] = float(np.median(gs))
        P(f"  {LABEL[ck]:12s}: G_cl median {np.median(gs):+.3f} [q25 {np.quantile(gs, .25):+.3f}, q75 {np.quantile(gs, .75):+.3f}] "
          f"(seeds {len(gs)}, values per seed median {int(np.median(ns))}) | G_P {p['G_P']:+.3f} G_L {p['G_L']:+.3f}")
summary["D1_G_cl"] = d1
P()

P("8. D2: over-bar analogue (frac99 > 0.10 in the window), BIAS: 11th smallest per-seed smallest visited value")
for ck in ORDER:
    Ts = []
    for s in SEEDS:
        rr = by[(ck[0], ck[1], s)]
        if not rr:
            continue
        pts = [(0.0, r["frac99_win"]) for r in rr if r["kind"] == "none"]
        pts += [(r["value"], r["frac99_win"]) for r in rr if r["kind"] == "bias"]
        hit = [v for v, fr in pts if fr > 0.10]
        Ts.append(min(hit) if hit else BIAS_HI)
    if len(Ts) == 20 and ("bias", ck) in res:
        P(f"  {LABEL[ck]:12s}: M_overbar <= {kth(Ts):.3f} (upper bound: visited values only) vs M_park {res[('bias', ck)]['M']:.3f}")
P()

P("9. D3: OBS response, seeds 3042-3046 (n 5): window command at d = 0, 1, 2, 4, 12 deg; linear P fixed point (cap applied);")
P("   static S fixed point a = c_S(5.19 a + d) (observed angle capped)")
for ck in ORDER:
    p = pred[f"{ck[0]}_{ck[1]}"]
    s_ = stat[f"{ck[0]}_{ck[1]}"]
    th, med = np.array(s_["thetas"]), np.array(s_["median_cmd"])
    cells = []
    for d in (0.0, 1.0, 2.0, 4.0, 12.0):
        vals = []
        for s in range(3042, 3047):
            rr = by[(ck[0], ck[1], s)]
            if d == 0.0:
                vals += [r["cmd_win"] for r in rr if r["kind"] == "none"]
            else:
                vals += [r["cmd_win"] for r in rr if r["kind"] == "obs" and r["value"] == d]
        lin = 0.0
        for _ in range(500):  # linear P with the observed angle capped (PREREG: th_need <= cap)
            lin = float(np.clip(p["c0"] + p["G_P"] / PLANT * min(PLANT * lin + d, p["cap_deg"]), -1, 1))
        a = 0.0
        for _ in range(200):
            a = float(np.clip(np.interp(min(T_RAIL * a + d, th[-1]), th, med), -1, 1))
        cells.append(f"d{d:>4.1f}: {np.median(vals) if vals else float('nan'):+.3f} (P {lin:+.3f} S {a:+.3f})")
    P(f"  {LABEL[ck]:12s}: " + " | ".join(cells))
P()

P("10. D5: the policy's own (pre-bias) window command in the parked episode at each seed's T_s (BIAS), against the")
P("    linear c0 + G_P and the static c_S(5.19 deg); tail_3 angle in that episode")
for ck in ORDER:
    owns, t3s = [], []
    for s in SEEDS:
        rr = by[(ck[0], ck[1], s)]
        T, lo, viol, pts = seed_threshold(rr, "bias", BIAS_HI)
        hit = [r for r in rr if r["kind"] == "bias" and r["value"] == T]
        if hit:
            owns.append(hit[0]["own_win"])
            t3s.append(hit[0]["tail3_deg_win"])
    if owns:
        p = pred[f"{ck[0]}_{ck[1]}"]
        P(f"  {LABEL[ck]:12s}: own at rail median {np.median(owns):+.3f} [{min(owns):+.3f}, {max(owns):+.3f}] (n {len(owns)}) | "
          f"c0 + G_P {p['c0'] + p['G_P']:+.3f} | c_S(5.19) {1 - p['bstar_S']:+.3f} | tail3 {np.median(t3s):+.2f} deg")
P()
(C.HERE / "wm_analyze.txt").write_text("\n".join(L) + "\n")
(C.HERE / "wm_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: None))
print("\n".join(L))
