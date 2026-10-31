"""The pre-registered analysis of the training-like rollouts (PREREG.txt, sha256 7c69819d...): items 1-5.

    PY -I -B analyze_rollouts.py > analysis_rollouts.txt      (also writes analysis_rollouts.json)

Reads ME/runs/<label>_free.{npz,meta.json,episodes.jsonl} and the clamp arms.  Offline only: no simulation, no
training.  Every rule below is the PREREG's; nothing here may change a threshold.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # one core: the machine is shared
    os.environ[_k] = "1"

import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_common as T  # noqa: E402

RUNS = T.ME / "runs"
NONBLIND = ("p1048576", "p2097152", "c2511_3145728", "c2531_3145728", "c2511_E", "c2531_E")
MAIN = NONBLIND + ("b2511_E",)
CLAMP = (("free", "p2097152_free"), ("hold 0", "p2097152_hold0.0"), ("hold -1", "p2097152_hold-1.0"),
         ("hold +1", "p2097152_hold1.0"))
G = T.GAMMA
SEED = 20261030
NB_12 = 1000
NB_4 = 200
TC = "tail_contact"
OUT: dict = {}


def P(*a):
    print(*a, flush=True)


def cp(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return (float(ci.low), float(ci.high))


class Cond:
    def __init__(self, tag):
        z = np.load(RUNS / f"{tag}.npz")
        self.a = {k: z[k] for k in z.files}
        self.meta = json.loads((RUNS / f"{tag}.meta.json").read_text())
        self.eps = [json.loads(line) for line in open(RUNS / f"{tag}.episodes.jsonl")]
        self.tag = tag
        a = self.a
        self.ep = a["all_ep"].astype(np.int64)
        self.t = a["all_t"].astype(np.int64)
        self.n_ep = len(self.eps)
        self.Tlen = np.array([e["steps"] for e in self.eps])
        self.reason = np.array([e["reason"] for e in self.eps])
        self.term = np.array([e["terminated"] for e in self.eps])
        counts = np.bincount(self.ep, minlength=self.n_ep)
        assert np.array_equal(counts, self.Tlen), "states per episode != steps"
        self.offset = np.concatenate([[0], np.cumsum(self.Tlen)[:-1]])
        self.Ts = self.Tlen[self.ep]                # episode length per state
        self.rs = self.reason[self.ep]              # end reason per state
        self.terms = self.term[self.ep]
        self.sigR = math.sqrt(self.meta["ret_var"] + self.meta["epsilon"])
        self.sd6_deg = math.degrees(self.meta["obs_rms_std"][T.OBS_TAIL3])
        self.q3 = a["all_q3"].astype(np.float64)
        self.tip = a["all_tip"].astype(np.float64)
        self.low = a["all_low"].astype(np.float64)
        self.pelvis = a["all_pelvis"].astype(np.float64)
        self.V = a["all_V"].astype(np.float64)
        self.s = a["all_dV3"].astype(np.float64)
        self.G = self._returns()

    def _returns(self):
        r = np.clip(self.a["all_r"].astype(np.float64) / self.sigR, -50.0, 50.0)
        Gt = np.empty_like(r)
        for e, ep in enumerate(self.eps):
            o, n = self.offset[e], self.Tlen[e]
            acc = (G * ep["V_final"]) if (ep["truncated"] and not ep["terminated"]) else 0.0
            for i in range(n - 1, -1, -1):
                acc = r[o + i] + G * acc
                Gt[o + i] = acc
        return Gt

    def sub_index(self):
        return self.offset[self.a["sub_ep"].astype(np.int64)] + self.a["sub_t"].astype(np.int64)


def load_all():
    C = {}
    for lab in MAIN:
        p = RUNS / f"{lab}_free.npz"
        if p.is_file():
            C[lab] = Cond(f"{lab}_free")
    return C


# ================================================================================================ ITEM 1
def item1(C):
    P("=" * 110)
    P("ITEM 1.  END REASONS OF TRAINING-LIKE EPISODES (128 per condition, seeds 7001-7128)")
    P("=" * 110)
    res = {}
    for lab, c in C.items():
        u, k = np.unique(c.reason, return_counts=True)
        d = dict(zip(u.tolist(), k.tolist()))
        n = c.n_ep
        ntc = d.get(TC, 0)
        nany = int(c.term.sum())
        res[lab] = {"n": n, "reasons": d, "tail_contact": ntc, "tail_contact_ci": cp(ntc, n), "any_term": nany,
                    "any_term_ci": cp(nany, n)}
        P(f"  {lab:15s} n {n:3d}  {d}   tail_contact {ntc}/{n} CI {cp(ntc, n)[0]:.4f}-{cp(ntc, n)[1]:.4f}"
          f"   any termination {nany}/{n} CI {cp(nany, n)[0]:.4f}-{cp(nany, n)[1]:.4f}")
    nb = [lab for lab in NONBLIND if lab in C]
    ntc = sum(res[x]["tail_contact"] for x in nb)
    nany = sum(res[x]["any_term"] for x in nb)
    n = sum(res[x]["n"] for x in nb)
    rate = ntc / n
    v11 = "SUPPORTED" if rate <= 0.02 else "NOT SUPPORTED"
    P(f"  POOLED NON-BLIND ({len(nb)} conditions): tail_contact {ntc}/{n} = {rate:.4%} CI {cp(ntc, n)[0]:.4%}-"
      f"{cp(ntc, n)[1]:.4%};  any termination {nany}/{n} = {nany / n:.4%} CI {cp(nany, n)[0]:.4%}-{cp(nany, n)[1]:.4%}")
    P(f"  P1.1 (prediction: pooled tail_contact <= 2%): {v11}")
    # P1.4
    if "b2511_E" in C and "c2511_E" in C:
        a_, b_ = res["b2511_E"]["tail_contact"], res["c2511_E"]["tail_contact"]
        n1, n2 = res["b2511_E"]["n"], res["c2511_E"]["n"]
        p14 = stats.fisher_exact([[a_, n1 - a_], [b_, n2 - b_]], alternative="greater")[1]
        v14 = "UNTESTABLE" if a_ + b_ < 3 else ("SUPPORTED" if p14 <= 0.05 else "NOT SUPPORTED")
        P(f"  P1.4 (B-genuine: b2511_E more tail_contact than c2511_E): {a_}/{n1} vs {b_}/{n2}; one-sided Fisher p "
          f"{p14:.4f} -> {v14}")
    else:
        p14, v14 = None, "MISSING"
    OUT["item1"] = {"per_condition": res, "pooled_nonblind": {"tail_contact": ntc, "any_term": nany, "n": n,
                    "rate": rate, "ci": cp(ntc, n), "any_ci": cp(nany, n)}, "P1.1": v11, "P1.4": v14, "p14": p14}


# ================================================================================================ ITEM 2
def eligible(c, H):
    return c.terms | (c.t <= 1000 - H)


def event(c, H):
    return (c.rs == TC) & ((c.Ts - 1 - c.t) < H)


def logistic(X, y, iters=60):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-12
        H_ = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H_, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def item2(C):
    P("\n" + "=" * 110)
    P("ITEM 2.  IS A LOWER TAIL ACTUALLY RISKIER?")
    P("=" * 110)
    nb = [lab for lab in NONBLIND if lab in C]
    rng = np.random.default_rng(SEED)
    res = {}
    n_tc = sum(int((C[x].reason == TC).sum()) for x in nb)
    P(f"  tail_contact episodes in the NON-BLIND SET: {n_tc}")
    # ---- M2.1
    for H in (200, 50):
        for feat in ("q3", "tip", "low"):
            ranks = []
            for lab in nb:
                c = C[lab]
                el = eligible(c, H)
                f = getattr(c, feat)
                pool = f[el]
                for e in np.flatnonzero(c.reason == TC):
                    ts = max(int(c.Tlen[e]) - H, 0)
                    v = f[c.offset[e] + ts]
                    ranks.append(float((np.sum(pool < v) + 0.5 * np.sum(pool == v)) / len(pool)))
            k = sum(r < 0.5 for r in ranks)
            n = len(ranks)
            p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
            if feat == "q3" and H == 200:
                verdict = "UNTESTABLE" if n < 5 else ("SUPPORTED" if p <= 0.05 else "NOT SUPPORTED")
            else:
                verdict = "secondary"
            res[f"M2.1_{feat}_H{H}"] = {"ranks": ranks, "k_below": k, "n": n, "p": p, "verdict": verdict}
            P(f"  M2.1 {feat:4s} H={H:3d}: events {n}; ranks {np.round(ranks, 3).tolist()}; below 0.5: {k}/{n}; "
              f"sign p {p:.4f} -> {verdict}")
    # ---- M2.2
    for H in (200, 50):
        for feat in ("q3", "tip", "low"):
            labs = [lab for lab in nb if (event(C[lab], H) & eligible(C[lab], H)).any()]
            if not labs:
                res[f"M2.2_{feat}_H{H}"] = {"beta": None, "note": "no events"}
                P(f"  M2.2 {feat} H={H}: no events")
                continue
            Xs, ys, cl, cond = [], [], [], []
            for j, lab in enumerate(labs):
                c = C[lab]
                el = eligible(c, H)
                f = getattr(c, feat)[el]
                Xs.append(f)
                ys.append(event(c, H)[el].astype(float))
                cl.append(c.ep[el])
                cond.append(np.full(el.sum(), j))
            f = np.concatenate(Xs)
            mu, sd = f.mean(), f.std()
            y = np.concatenate(ys)
            cond = np.concatenate(cond)
            cl = np.concatenate(cl)
            D = np.zeros((len(f), len(labs) + 1))
            D[np.arange(len(f)), cond] = 1
            D[:, -1] = (f - mu) / sd
            beta = logistic(D, y)[-1]
            bs = []
            by_cond = []
            for j, lab in enumerate(labs):
                rows = np.flatnonzero(cond == j)
                order = np.argsort(cl[rows], kind="stable")
                rows = rows[order]
                eps_j, starts = np.unique(cl[rows], return_index=True)
                bounds = np.append(starts, len(rows))
                by_cond.append([rows[bounds[i]:bounds[i + 1]] for i in range(len(eps_j))])
            for _ in range(NB_12):
                idx = []
                for blocks in by_cond:
                    pick = rng.integers(0, len(blocks), len(blocks))
                    idx.append(np.concatenate([blocks[i] for i in pick]))
                idx = np.concatenate(idx)
                if y[idx].sum() == 0:
                    bs.append(np.nan)
                    continue
                bs.append(logistic(D[idx], y[idx])[-1])
            bs = np.asarray(bs)
            lo_, hi_ = np.nanpercentile(bs, [2.5, 97.5])
            res[f"M2.2_{feat}_H{H}"] = {"beta_per_sd": float(beta), "ci": [float(lo_), float(hi_)],
                                        "conditions": labs, "events_states": int(y.sum()), "sd": float(sd)}
            P(f"  M2.2 {feat:4s} H={H:3d}: logistic slope per SD {beta:+.3f} (95% {lo_:+.3f} .. {hi_:+.3f}); "
              f"conditions {labs}; event states {int(y.sum())}; SD of feature {sd:.4f}")
    # ---- M2.3 the funding test
    for kind in ("tail_contact", "any"):
        per = []
        for lab in nb:
            c = C[lab]
            st = c.t <= 800
            ends = (c.rs == TC) if kind == "tail_contact" else c.terms
            Dt = np.where(ends, G ** (c.Ts - 1 - c.t).astype(float), 0.0)
            trunc_states = (~c.terms) & (c.t >= 200) & (c.t <= 800)
            Cc = 100.0 / c.sigR + float(np.median(c.V[trunc_states]))
            per.append({"lab": lab, "ep": c.ep[st], "q": c.q3[st], "Y": Cc * Dt[st], "s": c.s[st], "Cc": Cc,
                        "nstates": int(st.sum()), "n_ep": c.n_ep})

        def suff(blocks):
            # per condition: per-episode sums -> condition sums with multiplicities
            out = []
            for b in blocks:
                ne = b["n_ep"]
                n = np.bincount(b["ep"], minlength=ne).astype(float)
                sq = np.bincount(b["ep"], weights=b["q"], minlength=ne)
                sqq = np.bincount(b["ep"], weights=b["q"] ** 2, minlength=ne)
                sy = np.bincount(b["ep"], weights=b["Y"], minlength=ne)
                sqy = np.bincount(b["ep"], weights=b["q"] * b["Y"], minlength=ne)
                ss = np.bincount(b["ep"], weights=b["s"], minlength=ne)
                out.append(np.stack([n, sq, sqq, sy, sqy, ss]))
            return out

        S_ = suff(per)

        def estimate(mults):
            num = den = 0.0
            stot = ntot = 0.0
            for Sm, m in zip(S_, mults):
                n, sq, sqq, sy, sqy, ss = (Sm * m).sum(axis=1)
                num += sqy - sq * sy / n
                den += sqq - sq * sq / n
                stot += ss
                ntot += n
            b = num / den
            s_crit = stot / ntot
            return -b, s_crit
        ones = [np.ones(b["n_ep"]) for b in per]
        s_risk, s_crit = estimate(ones)
        F = s_risk / s_crit
        Fb = []
        for _ in range(NB_12):
            mults = [np.bincount(rng.integers(0, b["n_ep"], b["n_ep"]), minlength=b["n_ep"]).astype(float)
                     for b in per]
            sr, sc = estimate(mults)
            Fb.append(sr / sc)
        lo_, hi_ = np.percentile(Fb, [2.5, 97.5])
        n_ev = sum(int(((C[x].reason == TC) if kind == "tail_contact" else C[x].term).sum()) for x in nb)
        n_states = sum(b["nstates"] for b in per)
        n_eps = sum(b["n_ep"] for b in per)
        Cbar = float(np.mean([b["Cc"] for b in per]))
        geo = (1 - G ** 800) / (1 - G)
        k_u = n_eps * cp(n_ev, n_eps)[1]
        Dbar_u = k_u * geo / n_states
        F_bound = Cbar * 1.0 * Dbar_u / s_crit
        beta_req = 0.5 * s_crit / (Cbar * Dbar_u)
        if kind == "tail_contact":
            if n_ev < 5:
                verdict = "NOT SUPPORTED" if F_bound < 0.5 else "INCONCLUSIVE"
                basis = "bound (fewer than 5 events)"
            else:
                verdict = ("SUPPORTED" if (F >= 0.5 and lo_ > 0) else ("NOT SUPPORTED" if hi_ < 0.5 else
                                                                         "INCONCLUSIVE"))
                basis = "bootstrap interval"
        else:
            verdict, basis = "report only", "secondary"
        res[f"M2.3_{kind}"] = {"s_risk": s_risk, "s_crit": s_crit, "F": F, "F_ci": [float(lo_), float(hi_)],
                               "events": n_ev, "episodes": n_eps, "states": n_states, "C_c": {b["lab"]: b["Cc"]
                                                                                              for b in per},
                               "Cbar": Cbar, "k_u": k_u, "Dbar_u": Dbar_u, "F_bound_beta1": F_bound,
                               "beta_required_for_F_half": beta_req, "verdict": verdict, "basis": basis}
        P(f"  M2.3 [{kind}] events {n_ev} in {n_eps} episodes, {n_states} states t<=800.  C_c "
          f"{ {b['lab']: round(b['Cc'], 3) for b in per} }")
        P(f"       s_crit (mean critic slope) {s_crit:+.5f} per deg (normalised); s_risk {s_risk:+.6f}; F {F:+.5f} "
          f"(95% {lo_:+.5f} .. {hi_:+.5f})")
        P(f"       bound: CP upper count {k_u:.2f}, Dbar_u {Dbar_u:.3e}, F_bound(beta*=1/deg) {F_bound:.4f}, beta needed "
          f"for F=0.5: {beta_req:.1f} per degree -> {verdict} ({basis})")
    # ---- M2.4
    for H in (200, 50):
        for lab in C:
            c = C[lab]
            Ms = np.empty(len(c.t))
            for e in range(c.n_ep):
                o, n = c.offset[e], c.Tlen[e]
                lw = np.concatenate([c.low[o:o + n], np.full(H, np.inf)])
                Ms[o:o + n] = sliding_window_view(lw, H + 1)[:n].min(axis=1)
            el = eligible(c, H)
            pc = np.percentile(Ms[el], [0.1, 1, 5])
            rho = stats.spearmanr(c.q3[el], Ms[el])[0]
            res[f"M2.4_{lab}_H{H}"] = {"pct_0.1_1_5": pc.tolist(), "rho_q3_M": float(rho), "min": float(Ms[el].min())}
            P(f"  M2.4 H={H:3d} {lab:15s} min-ahead lowest tail point: p0.1 {pc[0]:.3f}  p1 {pc[1]:.3f}  p5 {pc[2]:.3f}"
              f"  min {Ms[el].min():.3f} m;  Spearman(q3, M) {rho:+.3f}")
    OUT["item2"] = res


# ================================================================================================ ITEM 3
def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def item3(C):
    P("\n" + "=" * 110)
    P("ITEM 3.  DOES THE CRITIC'S TAIL SLOPE (dV/dq3, normalised V per degree, effective critic) FOLLOW THE RISK?")
    P("=" * 110)
    res = {}
    n_neg = n_negp = n_negpel = 0
    pooled = {"near": [], "far": [], "near_eps": set(), "pel_near": [], "pel_far": []}
    for lab, c in C.items():
        s = c.s
        if lab == "b2511_E":
            s_raw = c.a["all_dV3raw"].astype(np.float64)
            P(f"  {lab}: effective slope is 0 by construction (mean {s.mean():+.2e}); the raw value MLP's slope "
              f"follows")
            s = s_raw
        r = {"mean": float(s.mean()), "median": float(np.median(s)), "sd": float(s.std())}
        P(f"  -- {lab}: s mean {s.mean():+.5f} median {np.median(s):+.5f} sd {s.std():.5f}  (x sigma_R "
          f"{c.sigR:.1f} = {s.mean() * c.sigR:+.3f} raw reward units per degree)")
        for name, f in (("tip", c.tip), ("pelvis", c.pelvis)):
            edges = np.quantile(f, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, 9)
            tab = [(float(edges[i]), float(edges[i + 1]), float(s[idx == i].mean()), int((idx == i).sum()))
                   for i in range(10)]
            r[f"by_{name}_decile"] = tab
            P(f"     by {name} decile: " + "  ".join(f"[{a_:.3f},{b_:.3f}] {m:+.4f}" for a_, b_, m, _n in tab))
        phases = ((0, 50), (50, 200), (200, 400), (400, 1000))
        ph = [(a_, b_, float(s[(c.t >= a_) & (c.t < b_)].mean()), int(((c.t >= a_) & (c.t < b_)).sum()))
              for a_, b_ in phases]
        r["by_phase"] = ph
        last = c.terms & ((c.Ts - c.t) <= 50)
        r["last50_terminated"] = (float(s[last].mean()) if last.any() else None, int(last.sum()))
        P("     by phase: " + "  ".join(f"t[{a_},{b_}) {m:+.4f} (n {n})" for a_, b_, m, n in ph)
          + f";  last 50 states of terminated episodes: {r['last50_terminated']}")
        rho = float(stats.spearmanr(s, c.tip)[0])
        rho_p = partial_spearman(s, c.tip, c.q3)
        rho_pel = float(stats.spearmanr(s, c.pelvis)[0])
        early, late = s[c.t < 200].mean(), s[c.t >= 200].mean()
        r.update(rho_tip=rho, rho_tip_partial_q3=rho_p, rho_pelvis=rho_pel, early_over_late=float(early / late),
                 rho_q3=float(stats.spearmanr(s, c.q3)[0]))
        P(f"     M3.1 Spearman(s, tip) {rho:+.3f};  M3.1b partial (| q3) {rho_p:+.3f};  M3.3 Spearman(s, pelvis) "
          f"{rho_pel:+.3f};  Spearman(s, q3) {r['rho_q3']:+.3f};  M3.4 early/late {early / late:.3f}")
        if lab in NONBLIND:
            n_neg += rho <= -0.10
            n_negp += rho_p <= -0.10
            n_negpel += rho_pel <= -0.10
            pooled["near"].append(s[c.tip < 0.60])
            pooled["far"].append(s[c.tip >= 0.60])
            pooled["near_eps"] |= {(lab, int(e)) for e in np.unique(c.ep[c.tip < 0.60])}
            pooled["pel_near"].append(s[c.pelvis < 0.85])
            pooled["pel_far"].append(s[c.pelvis >= 0.85])
        res[lab] = r
    near, far = np.concatenate(pooled["near"]), np.concatenate(pooled["far"])
    pn, pf = np.concatenate(pooled["pel_near"]), np.concatenate(pooled["pel_far"])
    nn, ne = len(near), len(pooled["near_eps"])
    v31 = "SUPPORTED" if n_neg >= 4 else "NOT SUPPORTED"
    ratio = float(near.mean() / far.mean()) if nn else None
    v32 = "UNTESTABLE" if (nn < 200 or ne < 3) else ("SUPPORTED" if ratio >= 1.5 else "NOT SUPPORTED")
    P(f"  M3.1 conditions with rho(s, tip) <= -0.10: {n_neg} of 6 -> {v31}")
    P(f"  M3.1b (secondary) partial rho <= -0.10: {n_negp} of 6;  M3.3 (secondary) rho(s, pelvis) <= -0.10: "
      f"{n_negpel} of 6")
    P(f"  M3.2 pooled near-floor (tip < 0.60 m): {nn} states from {ne} episodes, mean s "
      f"{(near.mean() if nn else float('nan')):+.5f} vs {far.mean():+.5f} (tip >= 0.60) ratio {ratio} -> {v32}")
    P(f"  pooled pelvis < 0.85 m: {len(pn)} states, mean s {(pn.mean() if len(pn) else float('nan')):+.5f} vs "
      f"{pf.mean():+.5f}")
    OUT["item3"] = {"per_condition": res, "M3.1_count": int(n_neg), "M3.1": v31, "M3.1b_count": int(n_negp),
                    "M3.3_count": int(n_negpel), "M3.2": {"n_states": nn, "n_eps": ne, "ratio": ratio, "verdict": v32},
                    "pelvis_near": {"n": int(len(pn)), "mean": float(pn.mean()) if len(pn) else None,
                                    "far_mean": float(pf.mean())}}


# ================================================================================================ ITEM 4
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def r2_oos(Xa, ya, Xb, yb):
    b = ols(Xa, ya)
    e = yb - Xb @ b
    return 1 - float(e @ e) / float(((yb - yb.mean()) ** 2).sum())


def item4(C):
    P("\n" + "=" * 110)
    P("ITEM 4.  IS THE CRITIC'S TAIL SLOPE EXPLAINED BY THE RISK?  (states t % 4 == 0, |z6| < 10)")
    P("=" * 110)
    rng = np.random.default_rng(SEED + 4)
    res = {}
    cnt = {"M4.1_sup": 0, "M4.1_unt": 0, "M4.2_sup": 0, "M4.2_not": 0, "M4.3_sup": 0, "M4.3_not": 0,
           "M4.3low_sup": 0, "M4.3low_not": 0, "M4.4_met": 0}
    for lab, c in C.items():
        a = c.a
        gi = c.sub_index()
        z = a["sub_z"].astype(np.float64)
        keep = np.abs(z[:, T.OBS_TAIL3]) < 10
        gi, z = gi[keep], z[keep]
        ep = a["sub_ep"].astype(np.int64)[keep]
        V, Gt = c.V[gi], c.G[gi]
        RF = np.column_stack([c.tip[gi], c.low[gi], c.pelvis[gi]])
        RF = (RF - RF.mean(0)) / RF.std(0)
        nonconst = np.flatnonzero(z.std(0) > 1e-6)
        assert T.OBS_TAIL3 in nonconst
        j6 = int(np.flatnonzero(nonconst == T.OBS_TAIL3)[0]) + 1
        one = np.ones((len(z), 1))
        X0 = np.hstack([one, z[:, nonconst]])
        X1 = np.hstack([X0, RF])
        r = {"n_states": int(len(z)), "dropped_const_cols": [int(i) for i in np.setdiff1d(np.arange(63), nonconst)]}
        for yname, y in (("V", V), ("G", Gt)):
            for mname, X in (("M0", X0), ("M1", X1)):
                b = ols(X, y)
                e = y - X @ b
                r[f"b6_{yname}_{mname}"] = float(b[j6] / c.sd6_deg)
                r[f"R2_{yname}_{mname}"] = 1 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
                if mname == "M1":
                    r[f"RF_coef_{yname}"] = [float(x) for x in b[-3:]]
        # VIF of z6 in M1
        others = np.delete(X1, j6, axis=1)
        bz = ols(others, X1[:, j6])
        ez = X1[:, j6] - others @ bz
        r2z = 1 - float(ez @ ez) / float(((X1[:, j6] - X1[:, j6].mean()) ** 2).sum())
        r["VIF_z6_M1"] = 1 / max(1 - r2z, 1e-12)
        r["S_V"] = 1 - r["b6_V_M1"] / r["b6_V_M0"]
        # bootstrap b6 (episodes within condition): OLS from per-episode Gram blocks, sum_e m_e X_e'X_e and
        # sum_e m_e X_e'y_e with multinomial episode counts m (the same estimator as refitting the resampled rows)
        eps_u, ep_ix = np.unique(ep, return_inverse=True)
        ne = len(eps_u)
        rows_of = [np.flatnonzero(ep_ix == i) for i in range(ne)]
        gram = {}
        for mname, X in (("M0", X0), ("M1", X1)):
            A = np.stack([X[rw].T @ X[rw] for rw in rows_of])
            for yname, y in (("V", V), ("G", Gt)):
                bvec = np.stack([X[rw].T @ y[rw] for rw in rows_of])
                gram[(yname, mname)] = (A, bvec)
        for yname in ("V", "G"):
            for mname in ("M0", "M1"):
                A, bvec = gram[(yname, mname)]
                bfull = np.linalg.lstsq(A.sum(0), bvec.sum(0), rcond=None)[0]
                r[f"b6_{yname}_{mname}_gram"] = float(bfull[j6] / c.sd6_deg)
        boot = {k: [] for k in ("V_M0", "V_M1", "G_M0", "G_M1")}
        for _ in range(NB_4):
            m = np.bincount(rng.integers(0, ne, ne), minlength=ne).astype(float)
            for yname in ("V", "G"):
                for mname in ("M0", "M1"):
                    A, bvec = gram[(yname, mname)]
                    bb = np.linalg.lstsq(np.tensordot(m, A, axes=1), m @ bvec, rcond=None)[0]
                    boot[f"{yname}_{mname}"].append(bb[j6] / c.sd6_deg)
        for k, v in boot.items():
            r[f"b6_{k}_ci"] = [float(x) for x in np.percentile(v, [2.5, 97.5])]
        if r["VIF_z6_M1"] > 30:
            v41 = "UNTESTABLE"
            cnt["M4.1_unt"] += lab in NONBLIND
        elif r["S_V"] >= 0.5 and r["RF_coef_V"][0] > 0:
            v41 = "SUPPORTED"
            cnt["M4.1_sup"] += lab in NONBLIND
        else:
            v41 = "NOT SUPPORTED"
        r["M4.1"] = v41
        # ---- M4.2
        tail_cols = [3, 4, 5, 6]
        base_cols = [i for i in nonconst if i not in tail_cols]
        B0 = np.hstack([one, z[:, base_cols]])
        J = np.hstack([B0, z[:, tail_cols]])
        Rm = np.hstack([B0, RF[:, :2]])
        fold = (ep % 2 == 0)
        for yname, y in (("V", V), ("G", Gt)):
            r2 = {}
            for mname, X in (("B0", B0), ("J", J), ("R", Rm)):
                r2[mname] = 0.5 * (r2_oos(X[fold], y[fold], X[~fold], y[~fold]) +
                                   r2_oos(X[~fold], y[~fold], X[fold], y[fold]))
            dJ = r2["J"] - r2["B0"]
            Q = (r2["R"] - r2["B0"]) / dJ if dJ != 0 else float("nan")
            r[f"M4.2_{yname}"] = {"R2": r2, "Q": Q, "dJ": dJ}
        dJ, Q = r["M4.2_V"]["dJ"], r["M4.2_V"]["Q"]
        if dJ < 0.005:
            v42 = "UNTESTABLE"
        elif Q >= 0.75:
            v42 = "SUPPORTED"
        elif Q < 0.5:
            v42 = "NOT SUPPORTED"
        else:
            v42 = "INCONCLUSIVE"
        r["M4.2"] = v42
        if lab in NONBLIND:
            cnt["M4.2_sup"] += v42 == "SUPPORTED"
            cnt["M4.2_not"] += v42 == "NOT SUPPORTED"
        # ---- M4.3
        gidx = c.meta["g_idx"]
        sd = np.degrees(np.asarray(c.meta["obs_rms_std"]))
        forms = [("effective", a["sub_g"][keep].astype(np.float64))]
        if "sub_graw" in a:
            forms.append(("raw", a["sub_graw"][keep].astype(np.float64)))
        zq = {1: 3, 2: 5, 3: 6}
        sdq = {q: float(z[:, zq[q]].std() * sd[zq[q]]) for q in (1, 2, 3)}
        r["onpolicy_sd_deg"] = sdq
        flag = (sdq[1] < 0.25 * sdq[3]) or (sdq[2] < 0.25 * sdq[3])
        r["M4.3_flag_small_q1q2_sd"] = bool(flag)
        clipped_any = (np.abs(z[:, 3]) >= 10) | (np.abs(z[:, 5]) >= 10)
        r["n_states_z3_or_z5_clipped"] = int(clipped_any.sum())
        for fname, gmat in forms:
            dV = {q: gmat[:, gidx.index(zq[q])] / sd[zq[q]] for q in (1, 2, 3)}
            for hname in ("tip", "low"):
                dh = {q: a[f"sub_d{hname}_dq{q}"][keep].astype(np.float64) for q in (1, 2, 3)}
                for sens_name, m in (("all", np.ones(len(z), bool)), ("no_z3z5_clip", ~clipped_any)):
                    v = {q: float(dV[q][m].mean() / dh[q][m].mean()) for q in (1, 2, 3)}
                    R31, R32 = v[3] / v[1], v[3] / v[2]
                    r[f"M4.3_{fname}_{hname}_{sens_name}"] = {
                        "dVdq_mean": {q: float(dV[q][m].mean()) for q in (1, 2, 3)},
                        "dh_dq_mean_m_per_deg": {q: float(dh[q][m].mean()) for q in (1, 2, 3)},
                        "v_per_m": v, "R31": R31, "R32": R32}
            mm = r[f"M4.3_{fname}_tip_all"]
            inb = (0.5 <= mm["R31"] <= 2) and (0.5 <= mm["R32"] <= 2)
            r[f"M4.3_{fname}"] = "SUPPORTED" if inb else "NOT SUPPORTED"
            ml = r[f"M4.3_{fname}_low_all"]
            r[f"M4.3low_{fname}"] = ("SUPPORTED" if ((0.5 <= ml["R31"] <= 2) and (0.5 <= ml["R32"] <= 2))
                                     else "NOT SUPPORTED")
        if lab in NONBLIND:
            cnt["M4.3_sup"] += r["M4.3_effective"] == "SUPPORTED"
            cnt["M4.3_not"] += r["M4.3_effective"] == "NOT SUPPORTED"
            cnt["M4.3low_sup"] += r["M4.3low_effective"] == "SUPPORTED"
            cnt["M4.3low_not"] += r["M4.3low_effective"] == "NOT SUPPORTED"
        # ---- M4.4
        met = (r["b6_G_M0"] >= 0.5 * r["b6_V_M0"]) and (r["b6_G_M0_ci"][0] > 0)
        r["M4.4_met"] = bool(met)
        if lab in NONBLIND:
            cnt["M4.4_met"] += met
        # critic's mean slope on these states, two ways
        r["mean_fd_slope"] = float(c.s[gi].mean())
        r["mean_autograd_slope"] = float((a["sub_g"][keep][:, gidx.index(6)] / sd[6]).mean())
        res[lab] = r
        P(f"  -- {lab}: n {len(z)}; critic mean slope FD {r['mean_fd_slope']:+.5f}, autograd "
          f"{r['mean_autograd_slope']:+.5f} per deg")
        P(f"     M4.1 b6 per deg: V M0 {r['b6_V_M0']:+.5f} [{r['b6_V_M0_ci'][0]:+.5f},{r['b6_V_M0_ci'][1]:+.5f}]  "
          f"V M1 {r['b6_V_M1']:+.5f} [{r['b6_V_M1_ci'][0]:+.5f},{r['b6_V_M1_ci'][1]:+.5f}]  S {r['S_V']:+.3f}  "
          f"RF coef V (tip, low, pelvis per SD) {np.round(r['RF_coef_V'], 5).tolist()}  VIF(z6) {r['VIF_z6_M1']:.1f}"
          f"  R2 V M0 {r['R2_V_M0']:.4f} M1 {r['R2_V_M1']:.4f} -> {v41}")
        P(f"          G M0 {r['b6_G_M0']:+.5f} [{r['b6_G_M0_ci'][0]:+.5f},{r['b6_G_M0_ci'][1]:+.5f}]  G M1 "
          f"{r['b6_G_M1']:+.5f} [{r['b6_G_M1_ci'][0]:+.5f},{r['b6_G_M1_ci'][1]:+.5f}]  RF coef G "
          f"{np.round(r['RF_coef_G'], 5).tolist()}  R2 G M0 {r['R2_G_M0']:.4f} M1 {r['R2_G_M1']:.4f};  M4.4 met "
          f"{met}")
        P(f"     M4.2 V: out-of-sample R2 B0 {r['M4.2_V']['R2']['B0']:.4f} J {r['M4.2_V']['R2']['J']:.4f} R "
          f"{r['M4.2_V']['R2']['R']:.4f}  dJ {dJ:+.4f}  Q {Q:+.3f} -> {v42};  (G: B0 {r['M4.2_G']['R2']['B0']:.4f}"
          f" J {r['M4.2_G']['R2']['J']:.4f} R {r['M4.2_G']['R2']['R']:.4f} Q {r['M4.2_G']['Q']:+.3f})")
        for fname, _g in forms:
            mm = r[f"M4.3_{fname}_tip_all"]
            ml = r[f"M4.3_{fname}_low_all"]
            P(f"     M4.3 [{fname}] dV/dq per deg q1 {mm['dVdq_mean'][1]:+.5f} q2 {mm['dVdq_mean'][2]:+.5f} q3 "
              f"{mm['dVdq_mean'][3]:+.5f}; dtip/dq mm/deg {[round(1000 * mm['dh_dq_mean_m_per_deg'][q], 2) for q in (1, 2, 3)]}"
              f"; V per m of tip q1 {mm['v_per_m'][1]:+.3f} q2 {mm['v_per_m'][2]:+.3f} q3 {mm['v_per_m'][3]:+.3f};"
              f" R31 {mm['R31']:+.3f} R32 {mm['R32']:+.3f} -> {r[f'M4.3_{fname}']};  (lowest point: R31 "
              f"{ml['R31']:+.3f} R32 {ml['R32']:+.3f} -> {r[f'M4.3low_{fname}']})")
        P(f"          on-policy SD (deg) q1 {sdq[1]:.3f} q2 {sdq[2]:.3f} q3 {sdq[3]:.3f}  flag {flag};  states with "
          f"|z3| or |z5| >= 10: {r['n_states_z3_or_z5_clipped']}")
    if cnt["M4.1_unt"] >= 3:
        v41o = "UNTESTABLE"
    else:
        v41o = "SUPPORTED" if cnt["M4.1_sup"] >= 4 else "NOT SUPPORTED"
    v42o = "SUPPORTED" if cnt["M4.2_sup"] >= 4 else ("NOT SUPPORTED" if cnt["M4.2_not"] >= 4 else "INCONCLUSIVE")
    v43o = "SUPPORTED" if cnt["M4.3_sup"] >= 4 else ("NOT SUPPORTED" if cnt["M4.3_not"] >= 4 else "INCONCLUSIVE")
    v43l = ("SUPPORTED" if cnt["M4.3low_sup"] >= 4 else ("NOT SUPPORTED" if cnt["M4.3low_not"] >= 4
                                                          else "INCONCLUSIVE"))
    v44 = "met" if cnt["M4.4_met"] >= 4 else "not met"
    P(f"  OVERALL M4.1 {v41o} (supported {cnt['M4.1_sup']}, untestable {cnt['M4.1_unt']} of 6);  M4.2 {v42o} "
      f"(supported {cnt['M4.2_sup']}, not {cnt['M4.2_not']});  M4.3 {v43o} (supported {cnt['M4.3_sup']}, not "
      f"{cnt['M4.3_not']});  M4.3 lowest-point (secondary) {v43l};  M4.4 {v44} ({cnt['M4.4_met']} of 6)")
    OUT["item4"] = {"per_condition": res, "counts": cnt, "M4.1": v41o, "M4.2": v42o, "M4.3": v43o,
                    "M4.3_low": v43l, "M4.4": v44}


# ================================================================================================ ITEM 5
def item5():
    P("\n" + "=" * 110)
    P("ITEM 5.  REPORT ONLY: p2097152 WITH THE STEPPED tail_3 COMMAND HELD (training-like otherwise)")
    P("=" * 110)
    res = {}
    base = None
    for name, tag in CLAMP:
        if not (RUNS / f"{tag}.npz").is_file():
            P(f"  {name}: missing")
            continue
        c = Cond(tag)
        u, k = np.unique(c.reason, return_counts=True)
        d = dict(zip(u.tolist(), k.tolist()))
        ntc, nany, n = d.get(TC, 0), int(c.term.sum()), c.n_ep
        rets = np.array([e["return"] for e in c.eps])
        r = {"n": n, "reasons": d, "tail_contact": ntc, "tail_contact_ci": cp(ntc, n), "any": nany,
             "any_ci": cp(nany, n), "q3_mean": float(c.q3.mean()), "tip_mean": float(c.tip.mean()),
             "low_mean": float(c.low.mean()), "low_min": float(c.low.min()), "tip_min": float(c.tip.min()),
             "return_mean": float(rets.mean()), "cmd_mean": float(c.a["all_cmd"].astype(float).mean()),
             "own_mean": float(c.a["all_own"].astype(float).mean())}
        if base is None:
            base = r
        else:
            r["fisher_tc_p"] = float(stats.fisher_exact([[ntc, n - ntc], [base["tail_contact"], base["n"] -
                                                                          base["tail_contact"]]])[1])
            r["fisher_any_p"] = float(stats.fisher_exact([[nany, n - nany], [base["any"], base["n"] - base["any"]]])[1])
        res[name] = r
        P(f"  {name:8s} n {n}  {d}  tail_contact {ntc} CI {r['tail_contact_ci'][1]:.4f}(upper)  any {nany}  "
          f"q3 {r['q3_mean']:+.3f} deg  tip {r['tip_mean']:.4f} (min {r['tip_min']:.3f})  lowest {r['low_mean']:.4f} "
          f"(min {r['low_min']:.3f}) m  return {r['return_mean']:.1f}  cmd {r['cmd_mean']:+.3f} own {r['own_mean']:+.3f}"
          + (f"  Fisher p (tc) {r['fisher_tc_p']:.3f} (any) {r['fisher_any_p']:.3f}" if "fisher_tc_p" in r else ""))
    OUT["item5"] = res


def descriptive(C):
    P("\n" + "=" * 110)
    P("DESCRIPTIVE (per condition, all states)")
    P("=" * 110)
    res = {}
    for lab, c in C.items():
        rets = np.array([e["return"] for e in c.eps])
        r = {"q3_mean_deg": float(c.q3.mean()), "q3_sd_deg": float(c.q3.std()),
             "own14_mean": float(c.a["all_own"].astype(float).mean()),
             "mu14_mean": float(c.a["all_mu"].astype(float).mean()), "tip_mean": float(c.tip.mean()),
             "tip_min": float(c.tip.min()), "low_mean": float(c.low.mean()), "low_min": float(c.low.min()),
             "pelvis_mean": float(c.pelvis.mean()), "pelvis_min": float(c.pelvis.min()), "V_mean": float(c.V.mean()),
             "G_mean": float(c.G.mean()), "s_mean": float(c.s.mean()), "return_mean": float(rets.mean()),
             "sigma_R": c.sigR, "sd6_deg": c.sd6_deg}
        res[lab] = r
        P(f"  {lab:15s} q3 {r['q3_mean_deg']:+.3f}+-{r['q3_sd_deg']:.3f} deg  own14 {r['own14_mean']:+.3f}  mu14 "
          f"{r['mu14_mean']:+.3f}  tip {r['tip_mean']:.4f} (min {r['tip_min']:.3f})  lowest {r['low_mean']:.4f} "
          f"(min {r['low_min']:.3f})  pelvis {r['pelvis_mean']:.4f} (min {r['pelvis_min']:.3f})  V {r['V_mean']:.3f}"
          f"  G {r['G_mean']:.3f}  s {r['s_mean']:+.5f}  return {r['return_mean']:.1f}")
    OUT["descriptive"] = res


def main():
    C = load_all()
    P(f"conditions loaded: {list(C)}")
    descriptive(C)
    item1(C)
    item2(C)
    item3(C)
    item4(C)
    item5()
    # overall reading
    i2, i3, i4 = OUT["item2"], OUT["item3"], OUT["item4"]
    m23 = i2["M2.3_tail_contact"]["verdict"]
    m21 = i2["M2.1_q3_H200"]["verdict"]
    if m23 == "SUPPORTED" and m21 == "SUPPORTED":
        bg = "SUPPORTED"
    elif m23 == "NOT SUPPORTED":
        bg = "NOT SUPPORTED"
    else:
        bg = "INCONCLUSIVE"
    m43, m31 = i4["M4.3"], i3["M3.1"]
    if m43 == "SUPPORTED" and m31 == "SUPPORTED":
        ba = "SUPPORTED"
    elif m43 == "NOT SUPPORTED" and m31 == "NOT SUPPORTED":
        ba = "NOT SUPPORTED"
    else:
        ba = "MIXED"
    P("\n" + "=" * 110)
    P(f"OVERALL (PREREG rules): B-GENUINE {bg} (M2.3 {m23}, M2.1 {m21});  B-APPARENT {ba} (M4.3 {m43}, M3.1 {m31})")
    OUT["overall"] = {"B_genuine": bg, "B_apparent": ba}
    (T.ME / "analysis_rollouts.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")
    P("written analysis_rollouts.json")


if __name__ == "__main__":
    main()
