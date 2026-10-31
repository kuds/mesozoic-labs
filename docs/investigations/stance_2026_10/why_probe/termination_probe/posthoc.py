"""POST HOC (not pre-registered; written after PREREG.txt and after the records were read, while the rollouts ran).
Descriptive only; nothing here changes a pre-registered verdict.

    PY -I -B posthoc.py > posthoc.txt

PH1  The termination-risk BUDGET.  However the risk depends on the tail, the value the critic can owe to termination
     risk is at most C_c x E[D_any] per state (the expected discounted exposure to ANY termination times the value a
     termination costs).  Compare it with the value spread the critic assigns to the tail: s_crit x (the on-policy
     q3 5-95% range).  If the tail's value spread exceeds the whole risk budget, no dependence of risk on the tail can
     explain the slope.  Upper version with the Clopper-Pearson 95% upper count of terminated episodes.
PH2  The critic's mean slope and the realised termination rates across checkpoints (the slope's trend against the
     risk's trend; the parent's records say terminations fell from 36% of episodes before 1.05M to 0.2% after).
PH3  The parent's and BS2's training-record terminations: when (timesteps) and how long the episodes were.
PH4  The terminated rollout episodes: the state of the tail and body in their last 50 steps (q3, tip, low, pelvis,
     forward_z) and the critic's slope there.
PH5  M4.2 recomputed without terminated episodes, with in-sample R2 beside the out-of-sample R2 (added after the
     17:11 debug pass showed one fall episode at p1048576 driving the two-fold R2 to -21).
PH6  The critic's tail-joint gradients state by state: sign agreement and correlation of dV/dq1, dV/dq2 with dV/dq3
     (added after the debug pass showed dV/dq1 < 0 < dV/dq3 at p2097152).
PH7  Cross-lens (lens A's ground, no verdict): does the episode phase absorb the tail_3 coefficient of V and of the
     realised G (added after the six-condition debug pass showed M4.4 met with no terminations)?
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_rollouts as A  # noqa: E402
import tp_common as T  # noqa: E402

C = A.load_all()
G = T.GAMMA
out = {}
print("POST HOC (not pre-registered).  Descriptive.\n")
print("PH1  the termination-risk budget against the tail's value spread (normalised V units)")
for lab, c in C.items():
    st = c.t <= 800
    Dany = np.where(c.terms, G ** (c.Ts - 1 - c.t).astype(float), 0.0)[st]
    trunc_states = (~c.terms) & (c.t >= 200) & (c.t <= 800)
    Cc = 100.0 / c.sigR + float(np.median(c.V[trunc_states]))
    budget = Cc * Dany.mean()
    n_term = int(c.term.sum())
    k_u = c.n_ep * A.cp(n_term, c.n_ep)[1]
    geo = (1 - G ** 800) / (1 - G)
    budget_u = Cc * k_u * geo / st.sum()
    s = c.s if lab != "b2511_E" else c.a["all_dV3raw"].astype(np.float64)
    q5, q95 = np.percentile(c.q3, [5, 95])
    spread = float(s.mean() * (q95 - q5))
    out[f"PH1_{lab}"] = {"C_c": Cc, "budget": budget, "budget_upper": budget_u, "terminated": n_term,
                         "q3_5_95": [float(q5), float(q95)], "tail_value_spread": spread,
                         "spread_over_budget_upper": spread / budget_u if budget_u > 0 else None}
    print(f"  {lab:15s} C_c {Cc:.2f}; terminated {n_term}/{c.n_ep}; risk budget mean C*D_any {budget:.2e} (upper "
          f"{budget_u:.2e}); q3 5-95% {q5:+.2f}..{q95:+.2f} deg; slope {s.mean():+.5f}/deg -> tail value spread "
          f"{spread:+.4f}; spread / upper budget {spread / budget_u if budget_u > 0 else float('nan'):.1f}")

# The same budget from the TRAINING RECORDS' termination rate in the era (analysis_records.json: the parent's
# (1,048,576, 3,538,944] training episodes, and BS2's eight runs' (2,097,152, 3,538,944] training episodes; any early end).
recs = json.loads((T.ME / "analysis_records.json").read_text())
n_par = recs["M1.2"]["n"]
k_par = n_par - recs["M1.2"]["causes"].get("truncated", 0)
n_bs = sum(v["n"] for v in recs["M1.3"]["per_run"].values())
k_bs = n_bs - sum(v["causes"].get("truncated", 0) for v in recs["M1.3"]["per_run"].values())
geo = (1 - G ** 800) / (1 - G)
print(f"  records: parent {k_par}/{n_par} training episodes ended early; BS2 {k_bs}/{n_bs}")
for name, k, n in (("parent records", k_par, n_par), ("BS2 records", k_bs, n_bs)):
    pu = A.cp(k, n)[1]
    # a terminating episode exposes its states to at most sum_j 0.98^j ~ 50 in total; ~1,000 states per episode
    bud = 7.2 * (k / n) * geo / 1000.0
    bud_u = 7.2 * pu * geo / 1000.0
    out[f"PH1_records_{name}"] = {"k": k, "n": n, "rate": k / n, "rate_upper": pu, "budget": bud, "budget_upper": bud_u}
    print(f"  {name}: early-end rate {k / n:.4%} (CP upper {pu:.4%}); risk budget with C = 7.2: {bud:.2e} "
          f"(upper {bud_u:.2e}) normalised V per state")
    for lab, c in C.items():
        s = c.s if lab != "b2511_E" else c.a["all_dV3raw"].astype(np.float64)
        q5, q95 = np.percentile(c.q3, [5, 95])
        spread = float(s.mean() * (q95 - q5))
        print(f"     {lab:15s} tail value spread {spread:+.4f} = {spread / bud_u:+.0f} x the upper budget")

print("\nPH2  critic's mean slope (FD, normalised per degree; raw MLP for b2511_E) and realised terminations")
for lab, c in C.items():
    s = c.s if lab != "b2511_E" else c.a["all_dV3raw"].astype(np.float64)
    print(f"  {lab:15s} slope {s.mean():+.5f} (x sigma_R = {s.mean() * c.sigR:+.3f} raw/deg)  q3 {c.q3.mean():+.3f} "
          f"deg  terminated {int(c.term.sum())}/{c.n_ep}  tail_contact {int((c.reason == 'tail_contact').sum())}")

print("\nPH3  terminations in the training records (timesteps, env, length, cause)")
rec = []
for path, name, lo, hi in ((f"{T.F2}/train_episodes.csv.copy1", "parent s42", 1_048_576, 3_538_944),):
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            ts = int(float(r["timesteps"]))
            if lo < ts <= hi and r["cause"] != "truncated":
                rec.append((name, ts, r["env"], r["length"], r["cause"]))
for s_ in (2501, 2511, 2521, 2531):
    for arm in ("control", "critic_blind_tail3"):
        with open(f"{T.BS2}/runs/{arm}_{s_}/01_stance/train_episodes.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                ts = int(float(r["timesteps"]))
                if 2_097_152 < ts <= 3_538_944 and r["cause"] != "truncated":
                    rec.append((f"{arm}_{s_}", ts, r["env"], r["length"], r["cause"]))
for row in rec:
    print("  ", row)
# the early era by 262,144 bins: terminations by cause
print("  parent s42, (0, 1,048,576] by 131,072-step bins: episodes, tail_contact, fallen, nosedive")
bins = {}
with open(f"{T.F2}/train_episodes.csv.copy1", newline="") as fh:
    for r in csv.DictReader(fh):
        ts = int(float(r["timesteps"]))
        if ts <= 1_048_576:
            b = (ts - 1) // 131_072
            d = bins.setdefault(b, {"n": 0})
            d["n"] += 1
            d[r["cause"]] = d.get(r["cause"], 0) + 1
for b in sorted(bins):
    d = bins[b]
    print(f"    ({b * 131072:>9,d}, {(b + 1) * 131072:>9,d}]: n {d['n']:4d}  tail_contact {d.get('tail_contact', 0):3d}"
          f"  fallen {d.get('fallen', 0):3d}  nosedive {d.get('nosedive', 0):3d}  truncated {d.get('truncated', 0):4d}")
out["PH3"] = {"window_terminations": rec, "early_bins": bins}

print("\nPH4  terminated rollout episodes: last-50-step means (and the final state)")
for lab, c in C.items():
    for e in np.flatnonzero(c.term):
        o, n = c.offset[e], c.Tlen[e]
        sl = slice(o + max(n - 50, 0), o + n)
        fz = c.a["all_fwdz"][sl].astype(float)
        s = c.s if lab != "b2511_E" else c.a["all_dV3raw"].astype(np.float64)
        print(f"  {lab:15s} ep {e:3d} seed {c.eps[e]['seed']} len {n:4d} {c.reason[e]:12s}  last50: q3 "
              f"{c.q3[sl].mean():+.2f} deg, tip {c.tip[sl].mean():.3f}, low {c.low[sl].mean():.3f}, pelvis "
              f"{c.pelvis[sl].mean():.3f}, fwd_z {fz.mean():+.3f}, slope {s[sl].mean():+.4f};  final: tip "
              f"{c.tip[o + n - 1]:.3f} low {c.low[o + n - 1]:.3f} pelvis {c.pelvis[o + n - 1]:.3f} fwd_z {fz[-1]:+.3f}"
              f"  episode-mean q3 {c.q3[o:o + n].mean():+.2f} (condition mean {c.q3.mean():+.2f})")
print("\nPH5  M4.2 again (the PREREG's form had no rule for outlier episodes): terminated episodes excluded, and the")
print("     in-sample R2 beside the two-fold out-of-sample R2 (V, NON-BLIND conditions and b2511_E)")
for lab, c in C.items():
    a = c.a
    gi = c.sub_index()
    z = a["sub_z"].astype(np.float64)
    ep = a["sub_ep"].astype(np.int64)
    keep = (np.abs(z[:, T.OBS_TAIL3]) < 10) & (~c.term[ep])
    gi, z, ep = gi[keep], z[keep], ep[keep]
    V = c.V[gi]
    RF = np.column_stack([c.tip[gi], c.low[gi]])
    RF = (RF - RF.mean(0)) / RF.std(0)
    nonconst = [i for i in np.flatnonzero(z.std(0) > 1e-6)]
    base = [i for i in nonconst if i not in (3, 4, 5, 6)]
    one = np.ones((len(z), 1))
    B0 = np.hstack([one, z[:, base]])
    J = np.hstack([B0, z[:, [3, 4, 5, 6]]])
    Rm = np.hstack([B0, RF])
    fold = ep % 2 == 0
    res = {}
    for name, X in (("B0", B0), ("J", J), ("R", Rm)):
        oos = 0.5 * (A.r2_oos(X[fold], V[fold], X[~fold], V[~fold]) + A.r2_oos(X[~fold], V[~fold], X[fold], V[fold]))
        b = A.ols(X, V)
        e = V - X @ b
        ins = 1 - float(e @ e) / float(((V - V.mean()) ** 2).sum())
        res[name] = (oos, ins)
    dJo, dJi = res["J"][0] - res["B0"][0], res["J"][1] - res["B0"][1]
    Qo = (res["R"][0] - res["B0"][0]) / dJo if dJo else float("nan")
    Qi = (res["R"][1] - res["B0"][1]) / dJi if dJi else float("nan")
    bR = A.ols(Rm, V)
    bJ = A.ols(J, V)
    sd_ang = np.degrees(np.asarray(c.meta["obs_rms_std"]))[[3, 4, 5, 6]]
    rf_sd = np.array([c.tip[gi].std(), c.low[gi].std()])
    out[f"PH5_{lab}"] = {"R2_oos_ins": res, "Q_oos": Qo, "Q_ins": Qi, "dJ_oos": dJo, "excluded_terminated": int(c.term.sum()),
                         "R_coef_per_m_tip_low": (bR[-2:] / rf_sd).tolist(),
                         "J_coef_per_deg_q1_q1yaw_q2_q3": (bJ[-4:] / sd_ang).tolist()}
    print(f"  {lab:15s} (terminated episodes excluded: {int(c.term.sum())})  out-of-sample R2 B0 {res['B0'][0]:.4f} J "
          f"{res['J'][0]:.4f} R {res['R'][0]:.4f} -> Q {Qo:+.3f} (dJ {dJo:+.4f});  in-sample B0 {res['B0'][1]:.4f} J "
          f"{res['J'][1]:.4f} R {res['R'][1]:.4f} -> Q {Qi:+.3f}")
    print(f"       model R coefficients, V per metre: tip {bR[-2] / rf_sd[0]:+.3f}, lowest point {bR[-1] / rf_sd[1]:+.3f};"
          f"  model J coefficients, V per degree: q1 {bJ[-4] / sd_ang[0]:+.5f} q1yaw {bJ[-3] / sd_ang[1]:+.5f} q2 "
          f"{bJ[-2] / sd_ang[2]:+.5f} q3 {bJ[-1] / sd_ang[3]:+.5f}")

print("\nPH6  the critic's own tail-joint gradients state by state (effective critic; raw MLP for b2511_E):")
print("     share of states where raising tail_1 (or tail_2) and raising tail_3 move V the same way, and the")
print("     correlation of the per-state slopes; a height (risk) valuation makes them agree in sign everywhere")
for lab, c in C.items():
    a = c.a
    z = a["sub_z"].astype(np.float64)
    keep = np.abs(z[:, T.OBS_TAIL3]) < 10
    gname = "sub_graw" if lab == "b2511_E" else "sub_g"
    g = a[gname][keep].astype(np.float64)
    gidx = c.meta["g_idx"]
    sd = np.degrees(np.asarray(c.meta["obs_rms_std"]))
    d1 = g[:, gidx.index(3)] / sd[3]
    d2 = g[:, gidx.index(5)] / sd[5]
    d3 = g[:, gidx.index(6)] / sd[6]
    agree1 = float(np.mean(np.sign(d1) == np.sign(d3)))
    agree2 = float(np.mean(np.sign(d2) == np.sign(d3)))
    r13 = float(np.corrcoef(d1, d3)[0, 1])
    r23 = float(np.corrcoef(d2, d3)[0, 1])
    pos1, pos3 = float(np.mean(d1 > 0)), float(np.mean(d3 > 0))
    out[f"PH6_{lab}"] = {"agree_q1_q3": agree1, "agree_q2_q3": agree2, "corr_q1_q3": r13, "corr_q2_q3": r23,
                         "share_pos_q1": pos1, "share_pos_q3": pos3,
                         "mean_per_deg": [float(d1.mean()), float(d2.mean()), float(d3.mean())]}
    print(f"  {lab:15s} mean dV/dq per deg q1 {d1.mean():+.5f} q2 {d2.mean():+.5f} q3 {d3.mean():+.5f};  share > 0: "
          f"q1 {pos1:.3f} q3 {pos3:.3f};  sign agreement q1~q3 {agree1:.3f}, q2~q3 {agree2:.3f};  corr q1~q3 "
          f"{r13:+.3f}, q2~q3 {r23:+.3f}")
print("\nPH7  (cross-lens, lens A's ground; no verdict) M4.4 found the realised return-to-go G rising with q3 about as much")
print("     as V does, with no terminations to explain it.  Does the episode PHASE absorb the tail_3 coefficient?  Model 0")
print("     (V or G ~ 1 + z) against Model 0 + phase (indicators of t in 25-step bins up to 300, then 100-step bins).")
print("     Also mean q3, V and G by phase.  States t % 4 == 0, |z6| < 10, terminated episodes excluded.")
edges = list(range(0, 300, 25)) + list(range(300, 1001, 100))
for lab, c in C.items():
    a = c.a
    gi = c.sub_index()
    z = a["sub_z"].astype(np.float64)
    ep = a["sub_ep"].astype(np.int64)
    keep = (np.abs(z[:, T.OBS_TAIL3]) < 10) & (~c.term[ep])
    gi, z = gi[keep], z[keep]
    tt = c.t[gi]
    nonconst = [i for i in np.flatnonzero(z.std(0) > 1e-6)]
    j6 = nonconst.index(T.OBS_TAIL3) + 1
    X0 = np.hstack([np.ones((len(z), 1)), z[:, nonconst]])
    ph = np.digitize(tt, edges[1:-1])
    Ph = np.zeros((len(z), len(edges) - 2))
    for k in range(1, len(edges) - 1):
        Ph[:, k - 1] = ph == k
    X1 = np.hstack([X0, Ph])
    res = {}
    for yname, y in (("V", c.V[gi]), ("G", c.G[gi])):
        b0 = A.ols(X0, y)[j6] / c.sd6_deg
        b1 = A.ols(X1, y)[j6] / c.sd6_deg
        res[yname] = (float(b0), float(b1))
    q3b = [float(c.q3[gi][(tt >= lo) & (tt < hi)].mean()) for lo, hi in ((0, 50), (50, 200), (200, 400), (400, 1000))]
    Vb = [float(c.V[gi][(tt >= lo) & (tt < hi)].mean()) for lo, hi in ((0, 50), (50, 200), (200, 400), (400, 1000))]
    Gb = [float(c.G[gi][(tt >= lo) & (tt < hi)].mean()) for lo, hi in ((0, 50), (50, 200), (200, 400), (400, 1000))]
    out[f"PH7_{lab}"] = {"b6_V_noPhase_phase": res["V"], "b6_G_noPhase_phase": res["G"], "q3_by_phase": q3b,
                         "V_by_phase": Vb, "G_by_phase": Gb}
    print(f"  {lab:15s} b6 per deg  V: {res['V'][0]:+.5f} -> {res['V'][1]:+.5f} with phase;  G: {res['G'][0]:+.5f} -> "
          f"{res['G'][1]:+.5f} with phase;  by t [0,50) [50,200) [200,400) [400,1000): q3 "
          f"{' '.join(f'{x:+.2f}' for x in q3b)} deg; V {' '.join(f'{x:.3f}' for x in Vb)}; G {' '.join(f'{x:.3f}' for x in Gb)}")
(T.ME / "posthoc.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print("\nwritten posthoc.json")
