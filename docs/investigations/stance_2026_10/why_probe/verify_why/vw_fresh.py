"""verify_why PART F + K2/K3/K5/K6/K7 on my fresh rollouts (runs/FR*.npz).  Offline; writes fresh.json/.txt here.

    PY -I -B vw_fresh.py
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vw_common as C  # noqa: E402

RUNS = {"FR1": "runs/FR1_C11E_stoch.npz", "FR2": "runs/FR2_C11E_det.npz", "FR3": "runs/FR3_P2_stoch.npz"}
GATED = ("reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band")
NBOOT = 300


def gae_returns(rew, val, starts, last_val, last_done, gamma=C.GAMMA, lam=C.LAM):
    """SB3 RolloutBuffer.compute_returns_and_advantage for one env's rollout."""
    n = len(rew)
    adv = np.zeros(n)
    last = 0.0
    for t in range(n - 1, -1, -1):
        if t == n - 1:
            nnt = 1.0 - float(last_done)
            nv = last_val
        else:
            nnt = 1.0 - starts[t + 1]
            nv = val[t + 1]
        delta = rew[t] + gamma * nv * nnt - val[t]
        last = delta + gamma * lam * nnt * last
        adv[t] = last
    return adv + val


def analyse(name, path):
    f = np.load(C.HERE / path, allow_pickle=False)
    meta = json.loads(str(f["meta_json"]))
    o = {"name": name, "label": meta["label"], "mode": meta["mode"], "seed": meta["seed"]}
    sig = meta["sigma_r"]
    var6, eps = meta["obs_var"][C.OBS_T3], meta["epsilon"]
    pdeg = lambda c: C.per_deg(c, var6, eps)  # noqa: E731
    k = f["k"].astype(np.int64)
    ep = f["ep"]
    sl = C.episodes(ep)
    n_ep = len(sl)
    raw = f["raw"].astype(np.float64)
    Z = f["z"].astype(np.float64)
    V = f["V"]
    rn, rr = f["rn"], f["rr"]
    ik = [str(x) for x in f["info_keys"]]
    info = f["info"]
    trunc = f["ep_trunc"]
    vterm = f["ep_vterm"]
    reasons = [str(x) for x in f["ep_reason"]]
    o["n_ep"], o["n_trunc"] = n_ep, int(trunc.sum())
    o["reasons"] = {r: reasons.count(r) for r in set(reasons)}
    o["n_tail_contact"] = reasons.count("tail_contact")
    o["ep_len_min"] = int(f["ep_len"].min())
    # ---- normalisation identity (R1 on fresh data)
    st = {"mean": np.asarray(meta["obs_mean"]), "var": np.asarray(meta["obs_var"]), "eps": eps,
          "clip_obs": meta["clip_obs"]}
    o["R1_max_abs_norm_diff"] = float(np.max(np.abs(C.normalise(raw, st) - Z)))
    o["R1_max_abs_rn_diff"] = float(np.max(np.abs(np.clip(rr / sig, -50, 50) - rn)))
    # ---- R3: SB3 buffer checks (stochastic runs)
    if "buf_rewards" in f.files:
        br, bv, bret = f["buf_rewards"], f["buf_values"], f["buf_returns"]
        bst = f["buf_episode_starts"]
        ern, eV, etr, evt = f["env_rn"], f["env_V"], f["env_trunc"], f["env_vterm"]
        edone = f["env_done"]
        n = br.shape[1]
        expect = ern[:, :n] + np.where(etr[:, :n], C.GAMMA * np.nan_to_num(evt[:, :n]), 0.0)
        o["R3_max_abs_bufreward_minus_expected"] = float(np.max(np.abs(br - expect)))
        o["R3_max_abs_bufvalue_minus_V"] = float(np.max(np.abs(bv - eV[:, :n])))
        # my G vs the discounted sum of the buffer's rewards within each complete episode
        diffs = []
        for e in range(n_ep):
            env = int(f["ep_env"][e])
            s0 = int(f["ep_start"][e])
            T = int(f["ep_len"][e])
            seg = br[env, s0:s0 + T]
            Gb = np.empty(T)
            acc = 0.0
            for t in range(T - 1, -1, -1):
                acc = seg[t] + C.GAMMA * acc
                Gb[t] = acc
            Gm = C.returns_to_go(rn[sl[e]], [np.arange(T)], [vterm[e] if trunc[e] else 0.0])
            diffs.append(np.max(np.abs(Gb - Gm)))
        o["R3_max_abs_G_minus_buffer_sum"] = float(max(diffs))
        # SB3's GAE returns recomputed by me, rollouts 0..4 (rollout r's last value = V of the first state of r+1)
        nst = 2048
        gd = []
        for env in range(br.shape[0]):
            for r in range(br.shape[1] // nst - 1):
                a, b = r * nst, (r + 1) * nst
                mine = gae_returns(br[env, a:b], bv[env, a:b], bst[env, a:b], eV[env, b], edone[env, b - 1])
                gd.append(np.max(np.abs(mine - bret[env, a:b])))
        o["R3_max_abs_gae_returns_diff"] = float(max(gd))
    # ---- G and components
    boot = np.where(trunc, vterm, 0.0)
    G = C.returns_to_go(rn, sl, boot)
    gated_raw = sum(info[:, ik.index(g)] for g in GATED)
    Ggate = C.returns_to_go(gated_raw / sig, sl, np.zeros(n_ep))
    Grest = G - Ggate
    Gwidth = C.returns_to_go(info[:, ik.index("reward_stance_width")] / sig, sl, np.zeros(n_ep))
    settled = k >= 200
    o["B_bar"] = float((gated_raw / sig)[settled].mean())
    o["gated_raw_k_lt_200_maxabs"] = float(np.max(np.abs(gated_raw[k < 200])))
    b = C.bin_of(k)
    o["G_by_bin"] = [float(G[b == i].mean()) for i in range(4)]
    o["V_by_bin"] = [float(V[b == i].mean()) for i in range(4)]
    # ---- K5 stationary counterfactual
    w = meta["weights"]
    feet = (info[:, ik.index("r_foot_terms_supported")] > 0.5) & (info[:, ik.index("l_foot_terms_supported")] > 0.5)
    lean_u = w["lateral_lean_weight"] * info[:, ik.index("lateral_lean_quality")] * feet
    stag_u = -w["stance_stagger_weight"] * (1.0 - info[:, ik.index("stance_stagger_quality")])
    cop_u = -w["stance_cop_band_weight"] * (1.0 - info[:, ik.index("stance_cop_band_quality")])
    gate_open = k >= 200
    ok_lean = np.abs(lean_u - info[:, ik.index("reward_lateral_lean")])[gate_open] <= 1e-6
    ok_stag = np.abs(stag_u - info[:, ik.index("reward_stance_stagger")])[gate_open] <= 1e-6
    ok_cop = np.abs(cop_u - info[:, ik.index("reward_stance_cop_band")])[gate_open] <= 1e-6
    o["K5_validation_frac"] = [float(ok_lean.mean()), float(ok_stag.mean()), float(ok_cop.mean())]
    k5_valid = min(o["K5_validation_frac"]) >= 0.999
    o["K5_valid"] = bool(k5_valid)
    r_stat = rr - gated_raw + lean_u + stag_u + cop_u
    Gstat = C.returns_to_go(np.clip(r_stat / sig, -50, 50), sl, boot)
    # ---- q3 creep
    q3 = np.degrees(raw[:, C.OBS_T3])
    dS_ep = np.array([q3[s][k[s] >= 200].mean() - q3[s][(k[s] >= 50) & (k[s] < 200)].mean() for s in sl])
    o["dS"] = float(dS_ep.mean())
    o["AUC_q3"] = float(C.auc(q3[settled], q3[(k >= 50) & (k < 200)]))
    o["q3_by_bin"] = [float(q3[b == i].mean()) for i in range(4)]
    mu14 = f["mu"][:, 14].astype(np.float64)
    o["mu14_by_bin"] = [float(mu14[b == i].mean()) for i in range(4)]
    e_ = V - G
    derr_ep = np.array([e_[s][(k[s] >= 50) & (k[s] < 200)].mean() - e_[s][k[s] >= 200].mean() for s in sl])
    o["D_err"] = float(derr_ep.mean())
    # ---- regressions
    X0, keep = C.design(Z)
    j6 = int(np.flatnonzero(keep == C.OBS_T3)[0]) + 1
    ph = C.phase_cols(k)
    X1 = np.hstack([X0, ph[:, :3]])
    X2 = np.hstack([X0, ph])
    regs = {}
    for yname, y in (("G", G), ("G_rest", Grest), ("G_gate", Ggate), ("G_stat", Gstat), ("V", V)):
        regs[yname] = (C.ClusterOLS(X0, y, sl), C.ClusterOLS(X2, y, sl))
        o[f"beta_{yname}_M0"] = float(pdeg(regs[yname][0].beta[j6]))
        o[f"beta_{yname}_M2"] = float(pdeg(regs[yname][1].beta[j6]))
    k8 = k <= 800
    sl8 = C.episodes(ep[k8])
    X08, _ = C.design(Z[k8], keep=keep)
    o["beta_G_stat_M0_k800"] = float(pdeg(C.ClusterOLS(X08, Gstat[k8], sl8).beta[j6]))
    o["beta_G_M0_k800"] = float(pdeg(C.ClusterOLS(X08, G[k8], sl8).beta[j6]))
    o1 = C.ClusterOLS(X1, G, sl)
    o["c_B1"] = float(o1.beta[X0.shape[1] + 1])
    b3 = k >= 400
    sl3 = C.episodes(ep[b3])
    X0b3, _ = C.design(Z[b3], keep=keep)
    ow3 = C.ClusterOLS(X0b3, G[b3], sl3)
    oww = C.ClusterOLS(X0b3, Gwidth[b3], sl3)
    ows = C.ClusterOLS(X0b3, Gstat[b3], sl3)
    o["beta_G_W3"] = float(pdeg(ow3.beta[j6]))
    o["K2_beta_Gwidth_W3"] = float(pdeg(oww.beta[j6]))
    o["beta_G_stat_W3"] = float(pdeg(ows.beta[j6]))
    fd = f["fd6"].astype(np.float64)
    o["fd_all"] = float(fd.mean())
    o["fd_B3"] = float(fd[b3].mean())
    o["fd_by_bin"] = [float(fd[b == i].mean()) for i in range(4)]
    # ---- bootstrap
    rng = np.random.default_rng(30101033)
    fdb3_ep = np.array([fd[s][k[s] >= 400].sum() for s in sl])
    nb3_ep = np.array([(k[s] >= 400).sum() for s in sl], dtype=float)
    reps = {x: [] for x in ("dS", "D_err", "c_B1", "M0", "M2", "dM0M2", "W3", "fdB3_minus_W3", "Gstat_M0", "Ggate_M0")}
    o0, o2 = regs["G"]
    os0 = regs["G_stat"][0]
    og0 = regs["G_gate"][0]

    def sol(obj, m):
        return np.linalg.solve(np.tensordot(m, obj.XtX, 1), np.tensordot(m, obj.Xty, 1))

    for _ in range(NBOOT):
        m = np.bincount(rng.integers(0, n_ep, n_ep), minlength=n_ep).astype(float)
        reps["dS"].append(float((m * dS_ep).sum() / m.sum()))
        reps["D_err"].append(float((m * derr_ep).sum() / m.sum()))
        bm0, bm2 = sol(o0, m)[j6], sol(o2, m)[j6]
        reps["M0"].append(float(pdeg(bm0)))
        reps["M2"].append(float(pdeg(bm2)))
        reps["dM0M2"].append(float(pdeg(bm0 - bm2)))
        reps["c_B1"].append(float(sol(o1, m)[X0.shape[1] + 1]))
        w3 = float(pdeg(sol(ow3, m)[j6]))
        reps["W3"].append(w3)
        reps["fdB3_minus_W3"].append(float((m * fdb3_ep).sum() / (m * nb3_ep).sum()) - w3)
        reps["Gstat_M0"].append(float(pdeg(sol(os0, m)[j6])))
        reps["Ggate_M0"].append(float(pdeg(sol(og0, m)[j6])))
    o["ci"] = {x: C.ci(v) for x, v in reps.items()}
    ci = o["ci"]
    # ---- predictions F1-F6 (FR1, FR3) and K3 (FR2)
    auc_bar = 0.55 if meta["label"] == "P2" else 0.85
    o["F1"] = bool(o["dS"] > 0 and ci["dS"][0] > 0 and o["AUC_q3"] >= auc_bar)
    o["F2"] = bool(o["c_B1"] < 0 and ci["c_B1"][1] < 0 and abs(o["c_B1"]) >= 0.05)
    o["F3"] = bool(o["D_err"] > 0 and ci["D_err"][0] > 0)
    o["F4"] = bool(o["beta_G_M0"] > 0 and ci["M0"][0] > 0 and o["beta_G_M2"] <= 0.5 * o["beta_G_M0"])
    o["F5"] = bool(o["fd_B3"] > 0 and ci["W3"][0] <= 0 and ci["fdB3_minus_W3"][0] > 0)
    o["F6"] = bool(o["n_tail_contact"] == 0 and (o["n_ep"] - o["n_trunc"]) <= 1)
    o["replication_holds"] = bool(o["F1"] and o["F2"] and o["F3"] and o["F4"] and o["F5"])
    o["K3_det_holds"] = bool(o["dS"] > 0 and ci["dS"][0] > 0 and o["AUC_q3"] >= 0.75)
    if k5_valid:
        o["K5_stat_holds"] = bool(o["beta_G_stat_M0"] <= 0.25 * o["beta_G_M0"] or ci["Gstat_M0"][0] <= 0 <= ci["Gstat_M0"][1])
        o["K5_gate_holds"] = bool(o["beta_G_gate_M0"] >= 0.75 * o["beta_G_M0"])
    # ---- K2 width reference
    tgt = info[:, ik.index("stance_width_target")]
    wrew = info[:, ik.index("reward_stance_width")]
    tgt_ep = np.array([tgt[s][k[s] >= 200].mean() for s in sl])
    wB3_ep = np.array([wrew[s][k[s] >= 400].mean() for s in sl])
    q200_ep = np.array([q3[s][(k[s] >= 190) & (k[s] < 210)].mean() for s in sl])
    o["K2_sd_target_mm"] = float(tgt_ep.std(ddof=1) * 1000)
    o["K2_B3_width_reward_mean_sd"] = [float(wB3_ep.mean()), float(wB3_ep.std(ddof=1))]
    o["K2_r_q200_target"] = float(np.corrcoef(q200_ep, tgt_ep)[0, 1])
    o["K2_r_q200_B3width"] = float(np.corrcoef(q200_ep, wB3_ep)[0, 1])
    # ---- K6 cross-joint
    g = f["grad"].astype(np.float64)
    c_j = g[:, :21].mean(0)
    full = np.full(63, np.nan)
    full[keep] = o0.beta[1:1 + len(keep)] - o2.beta[1:1 + len(keep)]
    b_j = full[:21]
    okm = np.isfinite(b_j)
    o["K6_r"] = float(np.corrcoef(c_j[okm], b_j[okm])[0, 1])
    ok6 = okm.copy()
    ok6[C.OBS_T3] = False
    o["K6_r_without_obs6"] = float(np.corrcoef(c_j[ok6], b_j[ok6])[0, 1])
    # ---- K7 LSTD(0.95)
    add = np.where(trunc, C.GAMMA * vterm, 0.0)
    for nm, Phi in (("blind", X0), ("phase", X2)):
        A, bb = C.lstd_blocks(Phi, rn, sl, add)
        wv = C.lstd_solve(A, bb)
        rl = np.random.default_rng(30101034)
        rr_ = []
        for _ in range(200):
            m = np.bincount(rl.integers(0, n_ep, n_ep), minlength=n_ep).astype(float)
            rr_.append(float(pdeg(C.lstd_solve(A, bb, m)[j6])))
        o[f"K7_lstd_{nm}"] = float(pdeg(wv[j6]))
        o[f"K7_lstd_{nm}_ci"] = C.ci(rr_)
    # ---- K1b on fresh data (report) and the tail's distance from the floor
    tip = f["phys"][:, 0].astype(np.float64)
    low = f["phys"][:, 1].astype(np.float64)
    o["K1b_rho_s_tip_B3"] = float(spearmanr(fd[b3], tip[b3])[0])
    o["low_min_m"] = float(low.min())
    o["low_min_after50_m"] = float(low[k >= 50].min())
    o["tip_min_m"] = float(tip.min())
    return o


def fmt(o):
    ci = o["ci"]
    L = [f"=== {o['name']} {o['label']} {o['mode']} seed {o['seed']}: {o['n_ep']} episodes, truncated {o['n_trunc']}, "
         f"reasons {o['reasons']}, min length {o['ep_len_min']}"]
    L.append(f"  R1 fresh: max |z - normalise(raw)| {o['R1_max_abs_norm_diff']:.2e}; max |rn - clip(rr/sigma)| "
             f"{o['R1_max_abs_rn_diff']:.2e}")
    if "R3_max_abs_bufreward_minus_expected" in o:
        L.append(f"  R3 SB3 buffer: max |buf reward - (r~ + gamma V(s_T) at truncation)| "
                 f"{o['R3_max_abs_bufreward_minus_expected']:.2e}; max |buf value - V| {o['R3_max_abs_bufvalue_minus_V']:.2e}; "
                 f"max |my G - buffer discounted sum| {o['R3_max_abs_G_minus_buffer_sum']:.2e}; max |my GAE returns - "
                 f"SB3 returns| {o['R3_max_abs_gae_returns_diff']:.2e}")
    L.append(f"  q3 by bin " + " ".join(f"{x:+.3f}" for x in o["q3_by_bin"]) + "  mu14 by bin " +
             " ".join(f"{x:+.3f}" for x in o["mu14_by_bin"]) + "  G by bin " + " ".join(f"{x:.4f}" for x in o["G_by_bin"])
             + "  V by bin " + " ".join(f"{x:.4f}" for x in o["V_by_bin"]))
    L.append(f"  B_bar {o['B_bar']:+.5f} (max |gated raw| at k<200 {o['gated_raw_k_lt_200_maxabs']:.1e})")
    L.append(f"  F1 dS {o['dS']:+.3f} [{ci['dS'][0]:+.3f}, {ci['dS'][1]:+.3f}] AUC {o['AUC_q3']:.3f} -> {o['F1']}")
    L.append(f"  F2 c_B1 {o['c_B1']:+.4f} [{ci['c_B1'][0]:+.4f}, {ci['c_B1'][1]:+.4f}] -> {o['F2']}")
    L.append(f"  F3 D_err {o['D_err']:+.4f} [{ci['D_err'][0]:+.4f}, {ci['D_err'][1]:+.4f}] -> {o['F3']}")
    L.append(f"  F4 beta_G M0 {o['beta_G_M0']:+.4f} [{ci['M0'][0]:+.4f}, {ci['M0'][1]:+.4f}] M2 {o['beta_G_M2']:+.4f} "
             f"[{ci['M2'][0]:+.4f}, {ci['M2'][1]:+.4f}] M0-M2 [{ci['dM0M2'][0]:+.4f}, {ci['dM0M2'][1]:+.4f}] -> {o['F4']}")
    L.append(f"  F5 critic FD B3 {o['fd_B3']:+.4f}; W3 beta_G {o['beta_G_W3']:+.4f} [{ci['W3'][0]:+.4f}, {ci['W3'][1]:+.4f}]; "
             f"FD_B3 - W3 [{ci['fdB3_minus_W3'][0]:+.4f}, {ci['fdB3_minus_W3'][1]:+.4f}] -> {o['F5']};  FD all "
             f"{o['fd_all']:+.4f}, by bin " + " ".join(f"{x:+.4f}" for x in o["fd_by_bin"]))
    L.append(f"  F6 tail_contact {o['n_tail_contact']}, early ends {o['n_ep'] - o['n_trunc']} -> {o['F6']}; lowest tail "
             f"point min {o['low_min_m']:.3f} m (after step 50: {o['low_min_after50_m']:.3f}); tip min {o['tip_min_m']:.3f}")
    L.append(f"  REPLICATION (F1-F5) {o['replication_holds']};  K3 deterministic-creep rule {o['K3_det_holds']}")
    L.append(f"  K5 validation (lean, stagger, cop reconstructed == paid for k>=200): {o['K5_validation_frac']} valid "
             f"{o['K5_valid']};  beta M0: G {o['beta_G_M0']:+.4f}  G_gate {o['beta_G_gate_M0']:+.4f} "
             f"[{ci['Ggate_M0'][0]:+.4f}, {ci['Ggate_M0'][1]:+.4f}]  G_rest {o['beta_G_rest_M0']:+.4f}  G_stat "
             f"{o['beta_G_stat_M0']:+.4f} [{ci['Gstat_M0'][0]:+.4f}, {ci['Gstat_M0'][1]:+.4f}] (k<=800: G "
             f"{o['beta_G_M0_k800']:+.4f}, G_stat {o['beta_G_stat_M0_k800']:+.4f}); G_stat M2 {o['beta_G_stat_M2']:+.4f}, "
             f"W3 {o['beta_G_stat_W3']:+.4f}" + (f" -> stat {o['K5_stat_holds']}, gate {o['K5_gate_holds']}"
                                                  if o["K5_valid"] else ""))
    L.append(f"  K2 SD target {o['K2_sd_target_mm']:.1f} mm; B3 width reward mean/SD {o['K2_B3_width_reward_mean_sd']}; "
             f"r(q3@200, target) {o['K2_r_q200_target']:+.3f}; r(q3@200, B3 width) {o['K2_r_q200_B3width']:+.3f}; W3 "
             f"beta_Gwidth {o['K2_beta_Gwidth_W3']:+.5f}")
    L.append(f"  K6 r(c_j, b_j) {o['K6_r']:+.3f} (without obs 6 {o['K6_r_without_obs6']:+.3f});  K7 LSTD phase-blind "
             f"{o['K7_lstd_blind']:+.4f} [{o['K7_lstd_blind_ci'][0]:+.4f}, {o['K7_lstd_blind_ci'][1]:+.4f}] phase-aware "
             f"{o['K7_lstd_phase']:+.4f} [{o['K7_lstd_phase_ci'][0]:+.4f}, {o['K7_lstd_phase_ci'][1]:+.4f}];  K1b "
             f"rho(s, tip | t>=400) {o['K1b_rho_s_tip_B3']:+.3f};  V M0 {o['beta_V_M0']:+.4f} M2 {o['beta_V_M2']:+.4f}")
    return "\n".join(L)


def main():
    res = {n: analyse(n, p) for n, p in RUNS.items()}
    txt = "\n".join(fmt(res[n]) for n in res)
    Path(C.HERE / "fresh.json").write_text(json.dumps(res, indent=1))
    Path(C.HERE / "fresh.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
