"""verify_why PART R (phase probe re-derivation) + K2, K6, K7 on the phase probe's recorded rollouts.  Offline: reads
the probe's rollouts/*.npz and analysis.json and the checkpoint files (read only); writes only redo_phase.json/.txt here.

    PY -I -B vw_redo.py [--labels P1,P2,C11a,C11E,C31a,C31E,B11E] [--nboot 300]
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vw_common as C  # noqa: E402

PAIRS = {
    "P1": (f"{C.F2}/models/stage1_1048576_steps.zip.copy1", f"{C.F2}/models/stage1_vecnormalize_1048576_steps.pkl.copy1", False),
    "P2": (f"{C.F2}/models/stage1_2097152_steps.zip.copy1", f"{C.F2}/models/stage1_vecnormalize_2097152_steps.pkl.copy1", False),
    "C11a": (f"{C.BS2R}/control_2511/01_stance/models/stage1_3145728_steps.zip",
             f"{C.BS2R}/control_2511/01_stance/models/stage1_vecnormalize_3145728_steps.pkl", False),
    "C11E": (f"{C.BS2R}/control_2511/01_stance/models/stage1_final.zip",
             f"{C.BS2R}/control_2511/01_stance/models/stage1_final_vecnorm.pkl", False),
    "C31a": (f"{C.BS2R}/control_2531/01_stance/models/stage1_3145728_steps.zip",
             f"{C.BS2R}/control_2531/01_stance/models/stage1_vecnormalize_3145728_steps.pkl", False),
    "C31E": (f"{C.BS2R}/control_2531/01_stance/models/stage1_final.zip",
             f"{C.BS2R}/control_2531/01_stance/models/stage1_final_vecnorm.pkl", False),
    "B11E": (f"{C.BS2R}/critic_blind_tail3_2511/01_stance/models/stage1_final.zip",
             f"{C.BS2R}/critic_blind_tail3_2511/01_stance/models/stage1_final_vecnorm.pkl", True),
}
GATED = ("reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band")


def sha256_file(p):
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def analyse(label, nboot, theirs, log):
    zp, pp, blind = PAIRS[label]
    f = np.load(f"{C.PP}/rollouts/{label}.npz", allow_pickle=False)
    meta = json.loads(str(f["meta_json"]))
    out = {"label": label, "blind": blind}
    # ---- provenance
    out["zip_match"] = (meta["zip"] == zp and meta["zip_sha256"] == sha256_file(zp))
    out["pkl_match"] = (meta["pkl"] == pp and meta["pkl_sha256"] == sha256_file(pp))
    st = C.load_pkl_stats(pp)
    out["pkl_classes"] = [st["cls"], st["obs_rms_cls"]]
    raw = f["raw_obs"].astype(np.float64)
    Zrec = f["norm_obs"].astype(np.float64)
    k = f["k"].astype(np.int64)
    ep = f["ep"]
    sl = C.episodes(ep)
    n_ep = len(sl)
    rn, rr = f["r_norm"], f["r_raw"]
    V = f["V"]
    # ---- R1 normalisation
    out["R1_meta_mean_equal"] = bool(np.array_equal(np.asarray(meta["obs_mean"]), st["mean"]))
    out["R1_meta_var_equal"] = bool(np.array_equal(np.asarray(meta["obs_var"]), st["var"]))
    out["R1_sigma_r_mine"] = st["sigma_r"]
    out["R1_sigma_r_meta"] = meta["sigma_r"]
    Zmine = C.normalise(raw, st)
    out["R1_max_abs_norm_diff"] = float(np.max(np.abs(Zmine - Zrec)))
    rn_mine = np.clip(rr / st["sigma_r"], -st["clip_reward"], st["clip_reward"])
    out["R1_max_rel_rnorm_diff"] = float(np.max(np.abs(rn_mine - rn) / np.maximum(np.abs(rn), 1e-12)))
    out["clip_obs"], out["n_z6_clipped"] = st["clip_obs"], int(np.sum(np.abs(Zrec[:, C.OBS_T3]) >= st["clip_obs"]))
    # reward_total vs r_raw (non-terminal steps)
    keys = [str(x) for x in f["reward_keys"]]
    terms = f["terms"].astype(np.float64)
    out["R1_max_abs_rewardtotal_minus_rraw"] = float(np.max(np.abs(terms[:, keys.index("reward_total")] - rr)))
    # ---- R2 network read-back
    vnet = C.ValueNet(zp)
    rng = np.random.default_rng(30101030)
    idx = np.sort(rng.choice(len(k), 4000, replace=False))
    Vm = vnet.value(Zrec[idx], blind=blind)
    out["R2_max_abs_V_diff"] = float(np.max(np.abs(Vm - V[idx])))
    fdm = C.fd_q3(vnet, raw[idx], st, blind=blind)
    out["R2_max_abs_fd_diff"] = float(np.max(np.abs(fdm - f["fd_slope"][idx])))
    if blind:
        fdr = C.fd_q3(vnet, raw[idx], st, blind=False)
        out["R2_max_abs_fd_raw_diff"] = float(np.max(np.abs(fdr - f["fd_slope_raw"][idx])))
    # ---- G
    trunc = f["ep_trunc"]
    vterm = f["ep_vterm"]
    boot = np.where(trunc, vterm, 0.0)
    G = C.returns_to_go(rn, sl, boot)
    gated = sum(terms[:, keys.index(g)] for g in GATED) / st["sigma_r"]
    Ggate = C.returns_to_go(gated, sl, np.zeros(n_ep))
    Grest = G - Ggate
    Gwidth = C.returns_to_go(terms[:, keys.index("reward_stance_width")] / st["sigma_r"], sl, np.zeros(n_ep))
    b = C.bin_of(k)
    out["G_by_bin"] = [float(G[b == i].mean()) for i in range(4)]
    out["Ggate_by_bin"] = [float(Ggate[b == i].mean()) for i in range(4)]
    out["V_by_bin"] = [float(V[b == i].mean()) for i in range(4)]
    out["n_trunc"], out["n_term"] = int(trunc.sum()), int((~trunc).sum())
    out["ep_len_min"] = int(f["ep_len"].min())
    # ---- B_bar
    settled = k >= 200
    out["B_bar"] = float(gated[settled].mean())
    out["gated_raw_settled"] = {g: float(terms[settled, keys.index(g)].mean()) for g in GATED}
    out["gated_raw_before201_maxabs"] = float(np.max(np.abs((gated * st["sigma_r"])[k < 200])))
    out["gated_raw_at200_mean"] = float((gated * st["sigma_r"])[k == 200].mean())
    # ---- q3
    q3 = np.degrees(raw[:, C.OBS_T3])
    m_set = np.array([q3[s][k[s] >= 200].mean() for s in sl])
    m_b1 = np.array([q3[s][(k[s] >= 50) & (k[s] < 200)].mean() for s in sl])
    dS_ep = m_set - m_b1
    out["dS"] = float(dS_ep.mean())
    out["AUC_q3"] = float(C.auc(q3[settled], q3[(k >= 50) & (k < 200)]))
    # ---- D_err
    e = V - G
    derr_ep = np.array([e[s][(k[s] >= 50) & (k[s] < 200)].mean() - e[s][k[s] >= 200].mean() for s in sl])
    out["D_err"] = float(derr_ep.mean())
    # ---- regressions
    X0, keep = C.design(Zrec)
    j6 = int(np.flatnonzero(keep == C.OBS_T3)[0]) + 1          # column of z6 in X0 (after the intercept)
    ph = C.phase_cols(k)
    X1 = np.hstack([X0, ph[:, :3]])
    X2 = np.hstack([X0, ph])
    var6 = float(st["var"][C.OBS_T3])
    pdeg = lambda c: C.per_deg(c, var6, st["eps"])  # noqa: E731
    res = {}
    for yname, y in (("G", G), ("G_rest", Grest), ("V", V), ("G_gate", Ggate)):
        o0 = C.ClusterOLS(X0, y, sl)
        o2 = C.ClusterOLS(X2, y, sl)
        res[yname] = (o0, o2)
        out[f"beta_{yname}_M0"] = float(pdeg(o0.beta[j6]))
        out[f"beta_{yname}_M2"] = float(pdeg(o2.beta[j6]))
    o1 = C.ClusterOLS(X1, G, sl)
    out["c_B1"] = float(o1.beta[X0.shape[1] + 1])
    b3 = k >= 400
    sl3 = C.episodes(ep[b3])
    X0b3, _ = C.design(Zrec[b3], keep=keep)
    ow3 = C.ClusterOLS(X0b3, G[b3], sl3)
    out["beta_G_W3"] = float(pdeg(ow3.beta[j6]))
    oww = C.ClusterOLS(X0b3, Gwidth[b3], sl3)
    out["K2_beta_Gwidth_W3"] = float(pdeg(oww.beta[j6]))
    # W3 sensitivity: k in [400, 800] (truncation bootstrap weight < 0.018)
    b38 = (k >= 400) & (k <= 800)
    sl38 = C.episodes(ep[b38])
    X38, _ = C.design(Zrec[b38], keep=keep)
    out["beta_G_W3_400_800"] = float(pdeg(C.ClusterOLS(X38, G[b38], sl38).beta[j6]))
    fd = f["fd_slope"]
    out["fd_all"] = float(fd.mean())
    out["fd_B3"] = float(fd[b3].mean())
    out["fd_by_bin"] = [float(fd[b == i].mean()) for i in range(4)]
    if blind:
        out["fd_raw_all"] = float(f["fd_slope_raw"].mean())
    # ---- bootstrap (episodes; the same weights for every statistic of a replicate)
    rng = np.random.default_rng(30101030)
    fdb3_ep = np.array([fd[s][k[s] >= 400].sum() for s in sl])
    nb3_ep = np.array([(k[s] >= 400).sum() for s in sl], dtype=float)
    reps = {key: [] for key in ("dS", "D_err", "c_B1", "dM0M2", "W3", "fdB3_minus_W3", "Gwidth_W3", "M0", "M2")}
    o0G, o2G = res["G"]
    for _ in range(nboot):
        m = np.bincount(rng.integers(0, n_ep, n_ep), minlength=n_ep).astype(float)
        reps["dS"].append(float((m * dS_ep).sum() / m.sum()))
        reps["D_err"].append(float((m * derr_ep).sum() / m.sum()))
        bm0 = np.linalg.solve(np.tensordot(m, o0G.XtX, 1), np.tensordot(m, o0G.Xty, 1))[j6]
        bm2 = np.linalg.solve(np.tensordot(m, o2G.XtX, 1), np.tensordot(m, o2G.Xty, 1))[j6]
        reps["M0"].append(float(pdeg(bm0)))
        reps["M2"].append(float(pdeg(bm2)))
        reps["dM0M2"].append(float(pdeg(bm0 - bm2)))
        reps["c_B1"].append(float(np.linalg.solve(np.tensordot(m, o1.XtX, 1),
                                                  np.tensordot(m, o1.Xty, 1))[X0.shape[1] + 1]))
        w3 = float(pdeg(np.linalg.solve(np.tensordot(m, ow3.XtX, 1), np.tensordot(m, ow3.Xty, 1))[j6]))
        reps["W3"].append(w3)
        reps["fdB3_minus_W3"].append(float((m * fdb3_ep).sum() / (m * nb3_ep).sum()) - w3)
        reps["Gwidth_W3"].append(float(pdeg(np.linalg.solve(np.tensordot(m, oww.XtX, 1),
                                                             np.tensordot(m, oww.Xty, 1))[j6])))
    out["ci"] = {key: C.ci(v) for key, v in reps.items()}
    # ---- K2 (settled-width reference)
    extra_keys = [str(x) for x in f["extra_keys"]]
    tgt = f["extra"][:, extra_keys.index("stance_width_target")].astype(np.float64)
    wrew = terms[:, keys.index("reward_stance_width")]
    tgt_ep = np.array([tgt[s][k[s] >= 201].mean() for s in sl])
    wB3_ep = np.array([wrew[s][k[s] >= 400].mean() for s in sl])
    q200_ep = np.array([q3[s][(k[s] >= 190) & (k[s] < 210)].mean() for s in sl])
    out["K2_sd_target_width_m"] = float(tgt_ep.std(ddof=1))
    out["K2_sd_B3_width_reward_raw"] = float(wB3_ep.std(ddof=1))
    out["K2_mean_B3_width_reward_raw"] = float(wB3_ep.mean())
    out["K2_r_q200_vs_B3width"] = float(np.corrcoef(q200_ep, wB3_ep)[0, 1])
    out["K2_r_q200_vs_target"] = float(np.corrcoef(q200_ep, tgt_ep)[0, 1])
    # ---- K6 (cross-joint)
    g = vnet.grad(Zrec, blind=blind)
    c_j = g[:, :21].mean(0)
    bM0 = o0G.beta[1:1 + len(keep)]
    bM2 = o2G.beta[1:1 + len(keep)]
    full_b = np.full(63, np.nan)
    full_b[keep] = bM0 - bM2
    b_j = full_b[:21]
    ok = np.isfinite(b_j)
    out["K6_n_joints"] = int(ok.sum())
    out["K6_r"] = float(np.corrcoef(c_j[ok], b_j[ok])[0, 1])
    ok6 = ok.copy()
    ok6[C.OBS_T3] = False
    out["K6_r_without_obs6"] = float(np.corrcoef(c_j[ok6], b_j[ok6])[0, 1])
    out["K6_c"] = c_j.tolist()
    out["K6_b"] = [None if not np.isfinite(x) else float(x) for x in b_j]
    out["ag_q3_mine_all"] = float(pdeg(g[:, C.OBS_T3].mean()))
    # ---- K7 (LSTD(0.95))
    add = np.where(trunc, C.GAMMA * vterm, 0.0)
    for name, Phi in (("blind", X0), ("phase", X2)):
        A, bb = C.lstd_blocks(Phi, rn, sl, add)
        w = C.lstd_solve(A, bb)
        rr_ = []
        rngl = np.random.default_rng(30101031)
        for _ in range(200):
            m = np.bincount(rngl.integers(0, n_ep, n_ep), minlength=n_ep).astype(float)
            rr_.append(float(pdeg(C.lstd_solve(A, bb, m)[j6])))
        out[f"K7_lstd_{name}"] = float(pdeg(w[j6]))
        out[f"K7_lstd_{name}_ci"] = C.ci(rr_)
    # ---- comparison with the probe's analysis.json
    t = theirs[label]
    cmp = {
        "dS": (out["dS"], t["m1"]["dS"], 0.005), "AUC_q3": (out["AUC_q3"], t["m1"]["AUC_q"], 0.002),
        "B_bar": (out["B_bar"], t["B_bar"], 2e-5), "c_B1": (out["c_B1"], t["m2"]["M1_G"]["c_B1"]["coef"], 0.002),
        "D_err": (out["D_err"], t["m3"]["D_err"], 0.002),
        "beta_G_M0": (out["beta_G_M0"], t["m4"]["beta_G_M0"]["est"], 0.002),
        "beta_G_M2": (out["beta_G_M2"], t["m4"]["beta_G_M2"]["est"], 0.002),
        "beta_G_W3": (out["beta_G_W3"], t["m4"]["beta_G_W3"]["est"], 0.002),
        "beta_G_rest_M0": (out["beta_G_rest_M0"], t["m4"]["beta_G_rest_M0"]["est"], 0.002),
        "fd_all": (out["fd_all"], t["m4"]["critic_fd"]["all"], 0.0005),
        "fd_B3": (out["fd_B3"], t["m4"]["critic_fd"]["B3"], 0.0005),
    }
    out["compare"] = {kk: {"mine": a, "theirs": bth, "diff": a - bth, "tol": tol, "ok": abs(a - bth) <= tol}
                      for kk, (a, bth, tol) in cmp.items()}
    out["their_ci"] = {"dS": t["m1"]["dS_ci"], "c_B1": t["m2"]["M1_G"]["c_B1"]["ci"], "D_err": t["m3"]["D_err_ci"],
                       "dM0M2": t["m4"]["beta_G_M0_minus_M2"]["ci"], "W3": t["m4"]["beta_G_W3"]["ci"],
                       "fdB3_minus_W3": t["m4"]["critic_fd_B3_minus_beta_G_W3"]["ci"]}
    log(f"{label}: done")
    return out


def fmt(o):
    L = [f"=== {o['label']} (blind {o['blind']})  zip/pkl provenance {o['zip_match']}/{o['pkl_match']}  "
         f"pkl classes {o['pkl_classes']}  episodes trunc {o['n_trunc']} term {o['n_term']}"]
    L.append(f"  R1 meta mean/var equal to my pkl read: {o['R1_meta_mean_equal']}/{o['R1_meta_var_equal']}; sigma_r "
             f"mine {o['R1_sigma_r_mine']:.6f} meta {o['R1_sigma_r_meta']:.6f}; max |norm_obs - mine| "
             f"{o['R1_max_abs_norm_diff']:.2e}; max rel |r_norm - mine| {o['R1_max_rel_rnorm_diff']:.2e}; z6 clipped "
             f"{o['n_z6_clipped']}; max |reward_total - r_raw| {o['R1_max_abs_rewardtotal_minus_rraw']:.2e}")
    L.append(f"  R2 max |V mine - V| {o['R2_max_abs_V_diff']:.2e}; max |FD mine - fd_slope| {o['R2_max_abs_fd_diff']:.2e}"
             + (f"; raw MLP FD diff {o['R2_max_abs_fd_raw_diff']:.2e}" if 'R2_max_abs_fd_raw_diff' in o else ""))
    L.append("  G by bin " + " ".join(f"{x:.4f}" for x in o["G_by_bin"]) + "; G_gate " +
             " ".join(f"{x:.4f}" for x in o["Ggate_by_bin"]) + "; V " + " ".join(f"{x:.4f}" for x in o["V_by_bin"]))
    L.append(f"  B_bar {o['B_bar']:+.6f} raw settled {', '.join(f'{kk[7:]} {v:+.4f}' for kk, v in o['gated_raw_settled'].items())}; "
             f"max |gated raw| at k<200 {o["gated_raw_before201_maxabs"]:.2e}, mean at k=200 {o["gated_raw_at200_mean"]:+.3f}")
    ci = o["ci"]
    L.append(f"  dS {o['dS']:+.4f} [{ci['dS'][0]:+.4f}, {ci['dS'][1]:+.4f}]  AUC {o['AUC_q3']:.4f}  c_B1 {o['c_B1']:+.4f} "
             f"[{ci['c_B1'][0]:+.4f}, {ci['c_B1'][1]:+.4f}]  D_err {o['D_err']:+.4f} [{ci['D_err'][0]:+.4f}, {ci['D_err'][1]:+.4f}]")
    L.append(f"  beta/deg G M0 {o['beta_G_M0']:+.4f} [{ci['M0'][0]:+.4f}, {ci['M0'][1]:+.4f}] M2 {o['beta_G_M2']:+.4f} "
             f"[{ci['M2'][0]:+.4f}, {ci['M2'][1]:+.4f}] M0-M2 [{ci['dM0M2'][0]:+.4f}, {ci['dM0M2'][1]:+.4f}];  G_rest M0 "
             f"{o['beta_G_rest_M0']:+.4f} M2 {o['beta_G_rest_M2']:+.4f};  G_gate M0 {o['beta_G_gate_M0']:+.4f} M2 "
             f"{o['beta_G_gate_M2']:+.4f};  V M0 {o['beta_V_M0']:+.4f} M2 {o['beta_V_M2']:+.4f}")
    L.append(f"  W3 beta_G {o['beta_G_W3']:+.4f} [{ci['W3'][0]:+.4f}, {ci['W3'][1]:+.4f}] (k 400-800: "
             f"{o['beta_G_W3_400_800']:+.4f});  critic FD all {o['fd_all']:+.4f} B3 {o['fd_B3']:+.4f}; FD_B3 - W3 "
             f"[{ci['fdB3_minus_W3'][0]:+.4f}, {ci['fdB3_minus_W3'][1]:+.4f}];  FD by bin "
             + " ".join(f"{x:+.4f}" for x in o["fd_by_bin"]) + (f"; raw MLP FD all {o['fd_raw_all']:+.4f}" if o["blind"] else ""))
    L.append(f"  K2 width ref: SD target {o['K2_sd_target_width_m'] * 1000:.2f} mm; B3-mean width reward raw mean "
             f"{o['K2_mean_B3_width_reward_raw']:.4f} SD {o['K2_sd_B3_width_reward_raw']:.4f}; r(q3@200, B3 width rew) "
             f"{o['K2_r_q200_vs_B3width']:+.3f}; r(q3@200, target) {o['K2_r_q200_vs_target']:+.3f}; W3 beta_Gwidth "
             f"{o['K2_beta_Gwidth_W3']:+.5f} [{ci['Gwidth_W3'][0]:+.5f}, {ci['Gwidth_W3'][1]:+.5f}] per deg")
    L.append(f"  K6 cross-joint r(c_j, b_j) over {o['K6_n_joints']} joints {o['K6_r']:+.3f} (without obs 6 "
             f"{o['K6_r_without_obs6']:+.3f}); my autograd q3 slope all {o['ag_q3_mine_all']:+.4f}")
    L.append(f"  K7 LSTD(0.95) tail coef/deg: phase-blind {o['K7_lstd_blind']:+.4f} [{o['K7_lstd_blind_ci'][0]:+.4f}, "
             f"{o['K7_lstd_blind_ci'][1]:+.4f}]  phase-aware {o['K7_lstd_phase']:+.4f} [{o['K7_lstd_phase_ci'][0]:+.4f}, "
             f"{o['K7_lstd_phase_ci'][1]:+.4f}]")
    bad = [f"{kk} mine {v['mine']:+.5f} theirs {v['theirs']:+.5f} (tol {v['tol']})" for kk, v in o["compare"].items()
           if not v["ok"]]
    L.append("  R4 compare: " + ("ALL WITHIN TOLERANCE" if not bad else "DISCREPANT: " + "; ".join(bad)))
    L.append("  their CIs: " + "; ".join(f"{kk} [{v[0]:+.4f}, {v[1]:+.4f}]" for kk, v in o["their_ci"].items()))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="P1,P2,C11a,C11E,C31a,C31E,B11E")
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    theirs = json.loads(Path(f"{C.PP}/analysis.json").read_text())["results"]
    t0 = time.time()

    def log(m):
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} {m} ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)

    res = {}
    for lab in a.labels.split(","):
        res[lab] = analyse(lab, a.nboot, theirs, log)
    txt = "\n".join(fmt(res[lab]) for lab in res)
    Path(C.HERE / f"redo_phase{a.tag}.json").write_text(json.dumps(res, indent=1))
    Path(C.HERE / f"redo_phase{a.tag}.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
