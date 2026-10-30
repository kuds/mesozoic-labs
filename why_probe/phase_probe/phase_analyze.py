"""PHASE PROBE: the pre-registered analysis (PREREG.txt, sha256 639cd879...).  Offline only: reads
rollouts/<label>.npz, fits linear regressions and small MLP regressors on the recorded data (no policy is trained),
writes analysis.json, analysis.txt and verdicts.txt in this folder.

    PY -I -B phase_analyze.py [--labels P1,P2,...] [--no-mlp]
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG_SHA = "639cd879facdd2168cee6aa545c14be29f2c7a9125f84cb228f3ac860ab7ee76"
LABELS = ("P1", "P2", "C11a", "C11E", "C31a", "C31E", "B11E", "N80")
PS = ("P2", "C11a", "C11E", "C31a", "C31E")
NONBLIND = ("P1", "P2", "C11a", "C11E", "C31a", "C31E")
GATED = ("reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band")
NBOOT = 400
BOOT_SEED = 12345
SETTLE = 200
FD_DEG = 0.25


# ------------------------------------------------------------------------------------------------ helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ci(samples):
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if not len(s):
        return [None, None]
    return [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))]


def halfwidth(c):
    return None if c[0] is None else (c[1] - c[0]) / 2.0


def excludes0(c):
    return c[0] is not None and (c[0] > 0 or c[1] < 0)


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), by ranks."""
    from scipy.stats import rankdata

    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


class Data:
    def __init__(self, label):
        z = np.load(HERE / "rollouts" / f"{label}.npz", allow_pickle=False)
        self.label = label
        self.meta = json.loads(str(z["meta_json"]))
        self.ep = z["ep"].astype(np.int64)
        self.k = z["k"].astype(np.int64)
        self.raw = z["raw_obs"].astype(np.float64)
        self.Z = z["norm_obs"].astype(np.float64)
        self.mu = z["mu"].astype(np.float64)
        self.act = z["action"].astype(np.float64)
        self.V = z["V"].astype(np.float64)
        self.rn = z["r_norm"].astype(np.float64)
        self.rr = z["r_raw"].astype(np.float64)
        self.terms = z["terms"].astype(np.float64)
        self.reward_keys = [str(x) for x in z["reward_keys"]]
        self.extra = z["extra"].astype(np.float64)
        self.extra_keys = [str(x) for x in z["extra_keys"]]
        self.phys = z["phys"].astype(np.float64)
        self.phys_keys = [str(x) for x in z["phys_keys"]]
        self.fd = z["fd_slope"].astype(np.float64)
        self.ag = z["ag_slope"].astype(np.float64)
        self.V_recomp = z["V_recomp"].astype(np.float64)
        self.blind = bool(self.meta["blind"])
        if self.blind:
            self.V_raw = z["V_raw"].astype(np.float64)
            self.fd_raw = z["fd_slope_raw"].astype(np.float64)
            self.ag_raw = z["ag_slope_raw"].astype(np.float64)
        self.fd_t3 = z["fd_slope_tail3"].astype(np.float64) if "fd_slope_tail3" in z.files else None
        self.ep_len = z["ep_len"].astype(np.int64)
        self.ep_trunc = z["ep_trunc"].astype(bool)
        self.ep_term = z["ep_term"].astype(bool)
        self.ep_reason = z["ep_reason"].astype(np.int64)
        self.ep_vterm = z["ep_vterm"].astype(np.float64)
        self.n_ep = len(self.ep_len)
        self.gamma = float(self.meta["gamma"])
        self.sigma = float(self.meta["sigma_r"])
        self.qidx = int(self.meta["qidx"])
        self.sd_q = math.sqrt(float(self.meta["obs_var"][self.qidx]) + float(self.meta["epsilon"]))
        self.sd6 = math.sqrt(float(self.meta["obs_var"][6]) + float(self.meta["epsilon"]))
        self.rows = [np.flatnonzero(self.ep == j) for j in range(self.n_ep)]
        for j, r in enumerate(self.rows):
            if not np.array_equal(self.k[r], np.arange(len(r))) or len(r) != self.ep_len[j]:
                raise SystemExit(f"{label}: episode {j} rows are not k = 0..T-1")
        self._returns()

    def _returns(self):
        g = self.gamma
        gi = [self.reward_keys.index(k) for k in GATED]
        self.gate_step = self.terms[:, gi].sum(axis=1) / self.sigma
        self.G = np.zeros(len(self.k))
        self.Gg = np.zeros(len(self.k))
        for j, r in enumerate(self.rows):
            acc = self.ep_vterm[j] if self.ep_trunc[j] else 0.0
            if self.ep_trunc[j] and not np.isfinite(acc):
                raise SystemExit(f"{self.label}: truncated episode {j} without a bootstrap value")
            accg = 0.0
            for t in range(len(r) - 1, -1, -1):
                acc = self.rn[r[t]] + g * acc
                accg = self.gate_step[r[t]] + g * accg
                self.G[r[t]] = acc
                self.Gg[r[t]] = accg
        self.Grest = self.G - self.Gg
        k = self.k
        self.D = np.stack([(k < 50), (k >= 50) & (k < 200), (k >= 200) & (k < 400), (k >= 400)], 1).astype(float)
        self.S = (k >= SETTLE).astype(float)
        self.R = g ** np.maximum(0, SETTLE - k)
        keep = np.std(self.Z, axis=0) > 1e-6
        self.zcols = np.flatnonzero(keep)
        self.zq = int(np.flatnonzero(self.zcols == self.qidx)[0])
        self.q = np.degrees(self.raw[:, self.qidx])


def per_degree(beta_z, sd):
    return beta_z * (math.pi / 180.0) / sd


# ------------------------------------------------------------------------------------------------ OLS with episode bootstrap
class EpOLS:
    """OLS on rows grouped by episode; the episode-clustered bootstrap reweights per-episode X'X and X'y."""

    def __init__(self, X, y, rows):
        self.p = X.shape[1]
        self.XtX = np.stack([X[r].T @ X[r] for r in rows])
        self.Xty = np.stack([X[r].T @ y[r] for r in rows])
        self.n = len(rows)

    def solve(self, w=None):
        w = np.ones(self.n) if w is None else w
        A = np.tensordot(w, self.XtX, axes=1)
        b = np.tensordot(w, self.Xty, axes=1)
        ridge = 1e-10 * np.trace(A) / self.p
        return np.linalg.solve(A + ridge * np.eye(self.p), b)


def design(d, phase, sel=None):
    """M0: Z; M1: Z + D0 D1 D2; M2: M1 + R; M3: Z + 19 bins of 50 + R.  Intercept first."""
    sel = np.arange(len(d.k)) if sel is None else sel
    cols = [np.ones(len(sel)), *d.Z[sel][:, d.zcols].T]
    if phase in ("M1", "M2"):
        cols += [d.D[sel, 0], d.D[sel, 1], d.D[sel, 2]]
    if phase == "M2":
        cols += [d.R[sel]]
    if phase == "M3":
        b50 = np.minimum(d.k[sel] // 50, 19)
        cols += [(b50 == j).astype(float) for j in range(19)]
        cols += [d.R[sel]]
    return np.stack(cols, 1)


def boot_weights(n_ep, nboot=NBOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    W = np.zeros((nboot, n_ep))
    for b in range(nboot):
        idx = rng.integers(0, n_ep, n_ep)
        W[b] = np.bincount(idx, minlength=n_ep)
    return W


# ------------------------------------------------------------------------------------------------ MLP (secondary)
def mlp_slope(d, y, with_phase, seed, sd_q_override=None):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = d.Z[:, d.zcols]
    if with_phase:
        X = np.concatenate([X, d.D, d.S[:, None], d.R[:, None]], 1)
    ys = float(np.std(y)) or 1.0
    ym = float(np.mean(y))
    yt = (y - ym) / ys
    hold = rng.permutation(d.n_ep)[: max(1, d.n_ep // 5)]
    is_hold = np.isin(d.ep, hold)
    Xt = torch.as_tensor(X[~is_hold], dtype=torch.float32)
    Yt = torch.as_tensor(yt[~is_hold], dtype=torch.float32)
    Xh = torch.as_tensor(X[is_hold], dtype=torch.float32)
    Yh = torch.as_tensor(yt[is_hold], dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                              torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, bad = float("inf"), None, 0
    for _epoch in range(40):
        perm = torch.as_tensor(rng.permutation(len(Xt)))
        for s in range(0, len(Xt), 512):
            b = perm[s:s + 512]
            opt.zero_grad()
            loss = ((net(Xt[b])[:, 0] - Yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            hl = float(((net(Xh)[:, 0] - Yh) ** 2).mean())
        if hl < best - 1e-5:
            best, bad = hl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    sd = d.sd_q if sd_q_override is None else sd_q_override
    dz = math.radians(FD_DEG) / sd
    clip = float(d.meta["clip_obs"])
    Xp, Xm = X.copy(), X.copy()
    Xp[:, d.zq] = np.clip(Xp[:, d.zq] + dz, -clip, clip)
    Xm[:, d.zq] = np.clip(Xm[:, d.zq] - dz, -clip, clip)
    with torch.no_grad():
        fp = net(torch.as_tensor(Xp, dtype=torch.float32))[:, 0].numpy().astype(np.float64)
        fm = net(torch.as_tensor(Xm, dtype=torch.float32))[:, 0].numpy().astype(np.float64)
    slope = (fp - fm) / (2 * FD_DEG) * ys
    return {"slope_all": float(slope.mean()), "slope_B3": float(slope[d.D[:, 3] > 0].mean()),
            "holdout_mse_std_units": best}


# ------------------------------------------------------------------------------------------------ one checkpoint
def analyse(d, do_mlp=True):
    out = {"label": d.label, "n_episodes": d.n_ep, "n_states": int(len(d.k)), "blind": d.blind,
           "num_timesteps": d.meta["num_timesteps"], "sigma_r": d.sigma, "gamma": d.gamma, "qidx": d.qidx,
           "joint": d.meta["joint"], "n_truncated": int(d.ep_trunc.sum()), "n_terminated": int(d.ep_term.sum()),
           "termination_reasons": [int(x) for x in d.ep_reason[d.ep_term]],
           "episode_lengths_terminated": [int(x) for x in d.ep_len[d.ep_term]],
           "zip_sha256": d.meta["zip_sha256"], "pkl_sha256": d.meta["pkl_sha256"],
           "max_abs_V_recomp_minus_V": d.meta["max_abs_V_recomp_minus_V"]}
    W = boot_weights(d.n_ep)
    bins = [d.D[:, i] > 0 for i in range(4)]
    names = ["B0", "B1", "B2", "B3"]
    k = d.k
    settled = k >= SETTLE
    B1 = bins[1]
    B23 = bins[2] | bins[3]

    # B_bar
    out["B_bar"] = float(d.gate_step[settled].mean())
    out["gated_terms_settled_mean_raw"] = {key: float(d.terms[settled, d.reward_keys.index(key)].mean())
                                           for key in GATED}
    sgnB = 1.0 if out["B_bar"] > 0 else -1.0

    # ---- 1. q (tail_3 by default) tracks the phase
    q = d.q
    mu14 = d.mu[:, 14]
    cmd = np.clip(d.act[:, 14], -1, 1)
    m1 = {"by_bin": {}}
    for nm, b in zip(names, bins):
        m1["by_bin"][nm] = {"q_mean": float(q[b].mean()), "q_sd": float(q[b].std()),
                            "mu14_mean": float(mu14[b].mean()), "mu14_sd": float(mu14[b].std()),
                            "cmd_mean": float(cmd[b].mean()), "cmd_sd": float(cmd[b].std()), "n": int(b.sum())}
    m1["r_q_k_all"] = float(np.corrcoef(q, k)[0, 1])
    m1["r_q_k_k50"] = float(np.corrcoef(q[k >= 50], k[k >= 50])[0, 1])

    def ep_diff(x, a0, a1, b0, b1):
        """per-episode mean of x over [a0,a1) minus over [b0,b1); returns (per-episode array, mask of valid)."""
        vals = np.full(d.n_ep, np.nan)
        for j, r in enumerate(d.rows):
            kk = d.k[r]
            ma, mb = (kk >= a0) & (kk < a1), (kk >= b0) & (kk < b1)
            if ma.any() and mb.any():
                vals[j] = x[r][ma].mean() - x[r][mb].mean()
        return vals

    def ep_mean_boot(vals):
        ok = np.isfinite(vals)
        est = float(np.nanmean(vals))
        bs = []
        for w in W:
            ww = w * ok
            if ww.sum() > 0:
                bs.append(float(np.nansum(np.where(ok, vals, 0.0) * ww) / ww.sum()))
        return est, ci(bs), int(ok.sum())

    dB = ep_diff(q, 200, 250, 150, 200)
    dS = ep_diff(q, 200, 10 ** 6, 50, 200)
    m1["dB"], m1["dB_ci"], m1["dB_n"] = ep_mean_boot(dB)
    m1["dS"], m1["dS_ci"], m1["dS_n"] = ep_mean_boot(dS)
    m1["AUC_q"] = auc(q[settled], q[B1])
    # mu14 too (descriptive)
    dSmu = ep_diff(mu14, 200, 10 ** 6, 50, 200)
    m1["dS_mu14"], m1["dS_mu14_ci"], _ = ep_mean_boot(dSmu)
    # finer profile (descriptive): q mean in 25-step bins up to 300, then 100-step bins
    edges = list(range(0, 300, 25)) + list(range(300, 1001, 100))
    m1["q_profile"] = [[a, b, float(q[(k >= a) & (k < b)].mean()) if ((k >= a) & (k < b)).any() else None]
                       for a, b in zip(edges[:-1], edges[1:])]
    m1["mu14_profile"] = [[a, b, float(mu14[(k >= a) & (k < b)].mean()) if ((k >= a) & (k < b)).any() else None]
                          for a, b in zip(edges[:-1], edges[1:])]
    a1_ok = excludes0(m1["dS_ci"]) and np.sign(m1["dS"]) == sgnB and m1["AUC_q"] is not None and (
        (m1["AUC_q"] - 0.5) * sgnB >= 0.10)
    m1["A1_holds"] = bool(a1_ok)
    out["m1"] = m1

    # ---- 2. the true value depends on the phase at matched posture
    m2 = {"by_bin": {}}
    for nm, b in zip(names, bins):
        m2["by_bin"][nm] = {"G": float(d.G[b].mean()), "G_gate": float(d.Gg[b].mean()),
                            "G_rest": float(d.Grest[b].mean()), "V": float(d.V[b].mean()),
                            "gate_step": float(d.gate_step[b].mean()), "r_norm": float(d.rn[b].mean())}
    X1 = design(d, "M1")
    p0 = 1 + len(d.zcols)
    for yname, y in (("G", d.G), ("G_rest", d.Grest), ("V", d.V)):
        ols = EpOLS(X1, y, d.rows)
        b = ols.solve()
        bs = np.stack([ols.solve(w) for w in W])
        m2[f"M1_{yname}"] = {nm: {"coef": float(b[p0 + i]), "ci": ci(bs[:, p0 + i])}
                             for i, nm in enumerate(("c_B0", "c_B1", "c_B2"))}
    c = m2["M1_G"]["c_B1"]
    m2["A2_resolvable"] = bool(halfwidth(c["ci"]) is not None and halfwidth(c["ci"]) <= 0.05)
    m2["A2_holds"] = bool(excludes0(c["ci"]) and np.sign(c["coef"]) == -sgnB and abs(c["coef"]) >= 0.05)
    out["m2"] = m2

    # ---- 3. the critic's error depends on the phase
    e = d.V - d.G
    m3 = {"by_bin": {nm: float(e[b].mean()) for nm, b in zip(names, bins)}}
    derr = np.full(d.n_ep, np.nan)
    for j, r in enumerate(d.rows):
        kk = d.k[r]
        a, bb = (kk >= 50) & (kk < 200), kk >= 200
        if a.any() and bb.any():
            derr[j] = e[r][a].mean() - e[r][bb].mean()
    m3["D_err"], m3["D_err_ci"], m3["D_err_n"] = ep_mean_boot(derr)
    m3["A3_resolvable"] = bool(halfwidth(m3["D_err_ci"]) is not None and halfwidth(m3["D_err_ci"]) <= 0.05)
    m3["A3_holds"] = bool(excludes0(m3["D_err_ci"]) and np.sign(m3["D_err"]) == sgnB and abs(m3["D_err"]) >= 0.05)
    if d.blind:
        er = d.V_raw - d.G
        m3["raw_value_mlp_by_bin"] = {nm: float(er[b].mean()) for nm, b in zip(names, bins)}
    out["m3"] = m3

    # ---- 4. is the critic's tail slope explained by the phase
    m4 = {}
    fd = d.fd
    m4["critic_fd"] = {"all": float(fd.mean()), **{nm: float(fd[b].mean()) for nm, b in zip(names, bins)},
                       "pre_settle": float(fd[~settled].mean()), "settled": float(fd[settled].mean())}
    m4["critic_ag"] = {"all": float(d.ag.mean()), **{nm: float(d.ag[b].mean()) for nm, b in zip(names, bins)},
                       "pre_settle": float(d.ag[~settled].mean()), "settled": float(d.ag[settled].mean())}
    if d.blind:
        m4["raw_mlp_fd"] = {"all": float(d.fd_raw.mean()),
                            **{nm: float(d.fd_raw[b].mean()) for nm, b in zip(names, bins)},
                            "pre_settle": float(d.fd_raw[~settled].mean()),
                            "settled": float(d.fd_raw[settled].mean())}
    if d.fd_t3 is not None:
        m4["critic_fd_tail3"] = {"all": float(d.fd_t3.mean()),
                                 **{nm: float(d.fd_t3[b].mean()) for nm, b in zip(names, bins)}}
    sols = {}
    for model in ("M0", "M1", "M2", "M3"):
        X = design(d, model)
        for yname, y in (("G", d.G), ("V", d.V), ("G_rest", d.Grest)):
            ols = EpOLS(X, y, d.rows)
            b = ols.solve()
            bs = np.array([ols.solve(w)[1 + d.zq] for w in W])
            sols[(model, yname)] = (per_degree(b[1 + d.zq], d.sd_q), per_degree(bs, d.sd_q))
            m4[f"beta_{yname}_{model}"] = {"est": sols[(model, yname)][0], "ci": ci(sols[(model, yname)][1])}
    for yname in ("G", "V", "G_rest"):
        diff = sols[("M0", yname)][1] - sols[("M2", yname)][1]
        m4[f"beta_{yname}_M0_minus_M2"] = {"est": sols[("M0", yname)][0] - sols[("M2", yname)][0], "ci": ci(diff)}
    # within B3
    sel3 = np.flatnonzero(bins[3])
    rows3 = [r[d.k[r] >= 400] for r in d.rows]
    X3 = design(d, "M0")
    ols3 = EpOLS(X3, d.G, rows3)
    b3 = ols3.solve()
    bs3 = np.array([per_degree(ols3.solve(w)[1 + d.zq], d.sd_q) for w in W])
    m4["beta_G_W3"] = {"est": per_degree(b3[1 + d.zq], d.sd_q), "ci": ci(bs3)}
    olsV3 = EpOLS(X3, d.V, rows3)
    m4["beta_V_W3"] = {"est": per_degree(olsV3.solve()[1 + d.zq], d.sd_q)}
    # critic FD over B3 per bootstrap resample (episode means weighted)
    fd_sum3 = np.array([fd[r].sum() for r in rows3])
    n3 = np.array([len(r) for r in rows3], dtype=float)
    fd3_boot = np.array([(w * fd_sum3).sum() / max((w * n3).sum(), 1) for w in W])
    m4["critic_fd_B3"] = float(fd[sel3].mean()) if len(sel3) else None
    m4["critic_fd_B3_minus_beta_G_W3"] = {"est": (m4["critic_fd_B3"] or 0.0) - m4["beta_G_W3"]["est"],
                                          "ci": ci(fd3_boot - bs3)}
    bG0 = m4["beta_G_M0"]
    bG2 = m4["beta_G_M2"]
    m4["A4_resolvable"] = bool(halfwidth(bG0["ci"]) is not None
                               and halfwidth(bG0["ci"]) <= max(abs(m4["critic_fd"]["all"]), 0.01))
    m4["A4_holds"] = bool(bG0["est"] > 0 and bG0["ci"][0] is not None and bG0["ci"][0] > 0
                          and bG2["est"] <= 0.5 * bG0["est"]
                          and m4["beta_G_M0_minus_M2"]["ci"][0] is not None and m4["beta_G_M0_minus_M2"]["ci"][0] > 0)
    w3 = m4["beta_G_W3"]
    m4["A4c_resolvable"] = bool(halfwidth(w3["ci"]) is not None
                                and halfwidth(w3["ci"]) <= max(m4["critic_fd_B3"] or 0.0, 0.01))
    dd = m4["critic_fd_B3_minus_beta_G_W3"]
    m4["A4c_holds"] = bool((m4["critic_fd_B3"] or 0.0) > 0 and not (w3["ci"][0] is not None and w3["ci"][0] > 0)
                           and dd["ci"][0] is not None and dd["ci"][0] > 0)
    if do_mlp:
        mlp = {}
        t0 = time.time()
        for yname, y in (("G", d.G), ("V", d.V)):
            for wp in (False, True):
                res = [mlp_slope(d, y, wp, seed) for seed in (1, 2)]
                mlp[f"{yname}_{'phase' if wp else 'nophase'}"] = {
                    "slope_all_mean": float(np.mean([r["slope_all"] for r in res])),
                    "slope_all_seeds": [r["slope_all"] for r in res],
                    "slope_B3_mean": float(np.mean([r["slope_B3"] for r in res])),
                    "holdout_mse": [r["holdout_mse_std_units"] for r in res]}
        mlp["seconds"] = round(time.time() - t0, 1)
        m4["mlp"] = mlp
    out["m4"] = m4

    # ---- secondary: phase decodability from Z (linear probability, k >= 50)
    sel = np.flatnonzero(k >= 50)
    dec = {}
    for nm, cols in (("all_Z", d.zcols), ("Z_without_q", d.zcols[d.zcols != d.qidx])):
        X = np.concatenate([np.ones((len(sel), 1)), d.Z[sel][:, cols]], 1)
        beta = np.linalg.lstsq(X, d.S[sel], rcond=None)[0]
        f = X @ beta
        dec[nm] = auc(f[d.S[sel] > 0], f[d.S[sel] == 0])
    dec["q_alone"] = m1["AUC_q"]
    out["phase_decodability_AUC"] = dec

    # ---- hypothesis B side data (report only)
    pk = d.phys_keys
    tmin = d.phys[:, [pk.index("tail3_floor_dist"), pk.index("tail4_floor_dist"), pk.index("tail5_floor_dist")]].min(1)
    out["B_side"] = {nm: {"tail_tip_z": float(d.phys[b, pk.index("tail_tip_z")].mean()),
                          "tail_min_floor_dist_mean": float(tmin[b].mean()),
                          "tail_min_floor_dist_min": float(tmin[b].min()),
                          "pelvis_z": float(d.phys[b, pk.index("pelvis_z")].mean())} for nm, b in zip(names, bins)}
    out["B_side"]["corr_q_tailmin_dist"] = float(np.corrcoef(q, tmin)[0, 1])
    return out


# ------------------------------------------------------------------------------------------------ verdicts
def verdicts(res):
    lines, v = [], {}

    def tally(key_holds, key_res=None):
        hold = [lab for lab in PS if lab in res and res[lab][key_holds[0]][key_holds[1]]]
        if key_res is None:
            resolvable = [lab for lab in PS if lab in res]
        else:
            resolvable = [lab for lab in PS if lab in res and res[lab][key_res[0]][key_res[1]]]
        if len(resolvable) < 3:
            return "UNTESTABLE", hold, resolvable
        return ("SUPPORTED" if len(hold) >= 4 else "NOT SUPPORTED"), hold, resolvable

    for name, kh, kr in (("A1", ("m1", "A1_holds"), None), ("A2", ("m2", "A2_holds"), ("m2", "A2_resolvable")),
                         ("A3", ("m3", "A3_holds"), ("m3", "A3_resolvable")),
                         ("A4", ("m4", "A4_holds"), ("m4", "A4_resolvable")),
                         ("A4c", ("m4", "A4c_holds"), ("m4", "A4c_resolvable"))):
        verdict, hold, resolvable = tally(kh, kr)
        v[name] = {"verdict": verdict, "holds_at": hold, "resolvable_at": resolvable}
        lines.append(f"{name}: {verdict}  (holds at {len(hold)}/5 PS: {hold}; resolvable {len(resolvable)}/5: "
                     f"{resolvable})")
    match = []
    for lab in NONBLIND:
        if lab not in res:
            continue
        r = res[lab]
        s_fd = np.sign(r["m4"]["critic_fd"]["all"])
        s_pred = np.sign(r["m1"]["dS"]) * np.sign(r["B_bar"])
        match.append((lab, bool(s_fd == s_pred), float(r["m4"]["critic_fd"]["all"]), float(r["m1"]["dS"]),
                      float(r["B_bar"])))
    nm = sum(m[1] for m in match)
    v["A5"] = {"verdict": "SUPPORTED" if nm >= 5 else ("UNTESTABLE" if len(match) < 6 else "NOT SUPPORTED"),
               "matches": match}
    lines.append(f"A5: {v['A5']['verdict']}  ({nm}/{len(match)} match: " + "; ".join(
        f"{m[0]} fd {m[2]:+.4f} dS {m[3]:+.3f} B_bar {m[4]:+.5f} {'match' if m[1] else 'MISMATCH'}" for m in match)
        + ")")
    a = {kk: v[kk]["verdict"] for kk in ("A1", "A2", "A3", "A4")}
    if all(x == "SUPPORTED" for x in a.values()):
        overall = "A SUPPORTED"
    elif a["A1"] == "NOT SUPPORTED" or a["A4"] == "NOT SUPPORTED":
        overall = "A NOT SUPPORTED"
    else:
        overall = "A PARTIAL"
    v["overall"] = overall
    lines.append(f"OVERALL: {overall}  (A1 {a['A1']}, A2 {a['A2']}, A3 {a['A3']}, A4 {a['A4']}; qualifiers A4c "
                 f"{v['A4c']['verdict']}, A5 {v['A5']['verdict']})")
    return v, lines


def fmt_ci(c):
    return "[None]" if c[0] is None else f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def render(res):
    L = []
    for lab, r in res.items():
        m1, m2, m3, m4 = r["m1"], r["m2"], r["m3"], r["m4"]
        L.append(f"=== {lab} ({r['joint']}, obs {r['qidx']}) num_timesteps {r['num_timesteps']}  episodes "
                 f"{r['n_episodes']} (truncated {r['n_truncated']}, terminated {r['n_terminated']} "
                 f"{r['termination_reasons']} at {r['episode_lengths_terminated']})  sigma_r {r['sigma_r']:.3f}  "
                 f"blind {r['blind']}")
        L.append(f"  B_bar (gated terms per settled step, normalised) {r['B_bar']:+.5f}; raw means "
                 + ", ".join(f"{k.replace('reward_', '')} {v:+.4f}" for k, v in r["gated_terms_settled_mean_raw"].items()))
        L.append("  1. angle by bin (deg): " + "; ".join(
            f"{nm} {b['q_mean']:+.3f} (SD {b['q_sd']:.3f}) mu14 {b['mu14_mean']:+.3f} cmd {b['cmd_mean']:+.3f}"
            for nm, b in m1["by_bin"].items()))
        L.append(f"     r(q,k) all {m1['r_q_k_all']:+.3f}, k>=50 {m1['r_q_k_k50']:+.3f}; dB {m1['dB']:+.4f} "
                 f"{fmt_ci(m1['dB_ci'])}; dS {m1['dS']:+.4f} {fmt_ci(m1['dS_ci'])}; AUC_q {m1['AUC_q']:.3f}; "
                 f"dS(mu14) {m1['dS_mu14']:+.4f}; A1 holds {m1['A1_holds']}")
        L.append("     q profile: " + " ".join(f"{a}-{b}:{v:+.2f}" for a, b, v in m1["q_profile"] if v is not None))
        L.append("     mu14 profile: " + " ".join(f"{a}-{b}:{v:+.3f}" for a, b, v in m1["mu14_profile"]
                                                  if v is not None))
        L.append("  2. by bin: " + "; ".join(
            f"{nm} G {b['G']:.3f} Ggate {b['G_gate']:+.3f} Grest {b['G_rest']:.3f} V {b['V']:.3f}"
            for nm, b in m2["by_bin"].items()))
        L.append("     M1 G: " + ", ".join(f"{k} {v['coef']:+.4f} {fmt_ci(v['ci'])}" for k, v in m2["M1_G"].items())
                 + f"; A2 holds {m2['A2_holds']} resolvable {m2['A2_resolvable']}")
        L.append("     M1 G_rest: " + ", ".join(f"{k} {v['coef']:+.4f} {fmt_ci(v['ci'])}"
                                                for k, v in m2["M1_G_rest"].items()))
        L.append("     M1 V: " + ", ".join(f"{k} {v['coef']:+.4f} {fmt_ci(v['ci'])}" for k, v in m2["M1_V"].items()))
        L.append("  3. V-G by bin: " + ", ".join(f"{k} {v:+.4f}" for k, v in m3["by_bin"].items())
                 + f"; D_err {m3['D_err']:+.4f} {fmt_ci(m3['D_err_ci'])}; A3 holds {m3['A3_holds']} resolvable "
                 f"{m3['A3_resolvable']}")
        if "raw_value_mlp_by_bin" in m3:
            L.append("     raw value MLP V-G by bin: " + ", ".join(f"{k} {v:+.4f}"
                                                                  for k, v in m3["raw_value_mlp_by_bin"].items()))
        L.append("  4. critic FD slope /deg: " + ", ".join(f"{k} {v:+.4f}" for k, v in m4["critic_fd"].items())
                 + " | autograd: " + ", ".join(f"{k} {v:+.4f}" for k, v in m4["critic_ag"].items()))
        if "raw_mlp_fd" in m4:
            L.append("     raw value MLP FD: " + ", ".join(f"{k} {v:+.4f}" for k, v in m4["raw_mlp_fd"].items()))
        if "critic_fd_tail3" in m4:
            L.append("     critic FD on tail_3: " + ", ".join(f"{k} {v:+.4f}" for k, v in m4["critic_fd_tail3"].items()))
        for yname in ("G", "V", "G_rest"):
            L.append(f"     beta_{yname} /deg: " + ", ".join(
                f"{m} {m4[f'beta_{yname}_{m}']['est']:+.4f} {fmt_ci(m4[f'beta_{yname}_{m}']['ci'])}"
                for m in ("M0", "M1", "M2", "M3"))
                + f"; M0-M2 {m4[f'beta_{yname}_M0_minus_M2']['est']:+.4f} "
                f"{fmt_ci(m4[f'beta_{yname}_M0_minus_M2']['ci'])}")
        L.append(f"     W3: beta_G {m4['beta_G_W3']['est']:+.4f} {fmt_ci(m4['beta_G_W3']['ci'])}, beta_V "
                 f"{m4['beta_V_W3']['est']:+.4f}; critic FD B3 {m4['critic_fd_B3']:+.4f}; FD_B3 - beta_G_W3 "
                 f"{m4['critic_fd_B3_minus_beta_G_W3']['est']:+.4f} {fmt_ci(m4['critic_fd_B3_minus_beta_G_W3']['ci'])}")
        L.append(f"     A4 holds {m4['A4_holds']} resolvable {m4['A4_resolvable']}; A4c holds {m4['A4c_holds']} "
                 f"resolvable {m4['A4c_resolvable']}")
        if "mlp" in m4:
            L.append("     MLP FD slopes /deg (mean of 2 seeds; all states | B3): " + "; ".join(
                f"{k} {v['slope_all_mean']:+.4f} | {v['slope_B3_mean']:+.4f}" for k, v in m4["mlp"].items()
                if isinstance(v, dict)))
        L.append("  decodability AUC (settled vs not, k>=50): " + ", ".join(
            f"{k} {v:.3f}" for k, v in r["phase_decodability_AUC"].items() if v is not None))
        L.append("  B side: " + "; ".join(
            f"{nm} tip {b['tail_tip_z']:.3f} m, tail min floor dist mean {b['tail_min_floor_dist_mean']:.3f} "
            f"min {b['tail_min_floor_dist_min']:.3f}, pelvis {b['pelvis_z']:.3f}"
            for nm, b in r["B_side"].items() if isinstance(b, dict))
            + f"; corr(q, tail min dist) {r['B_side']['corr_q_tailmin_dist']:+.3f}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(1)
    prereg = sha256_file(HERE / "PREREG.txt")
    if prereg != PREREG_SHA:
        raise SystemExit(f"PREREG.txt sha256 {prereg} != {PREREG_SHA}")
    res = {}
    for lab in args.labels.split(","):
        f = HERE / "rollouts" / f"{lab}.npz"
        if not f.is_file():
            print(f"{lab}: no rollout file, skipped", flush=True)
            continue
        t0 = time.time()
        d = Data(lab)
        res[lab] = analyse(d, do_mlp=not args.no_mlp)
        res[lab]["npz_sha256"] = sha256_file(f)
        print(f"{lab}: analysed in {time.time() - t0:.1f} s", flush=True)
    v, vlines = verdicts(res)
    tag = args.tag
    (HERE / f"analysis{tag}.json").write_text(json.dumps({"prereg_sha256": prereg, "results": res, "verdicts": v},
                                                         indent=1))
    txt = render(res)
    (HERE / f"analysis{tag}.txt").write_text("\n".join(txt) + "\n")
    (HERE / f"verdicts{tag}.txt").write_text("\n".join(vlines) + "\n")
    print("\n".join(txt))
    print("\n".join(vlines))


if __name__ == "__main__":
    main()
