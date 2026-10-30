"""verify_why PART R6 + K1 + K4: the termination probe's recorded runs and the training records, re-derived with my
own code.  Offline; reads only; writes redo_term.json/.txt here.

    PY -I -B vw_redo_term.py
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vw_common as C  # noqa: E402

NB6 = ("p1048576", "p2097152", "c2511_3145728", "c2531_3145728", "c2511_E", "c2531_E")
ALL7 = NB6 + ("b2511_E",)
HOLDS = ("p2097152_hold0.0", "p2097152_hold-1.0", "p2097152_hold1.0")
ZIPS = {"p1048576": f"{C.F2}/models/stage1_1048576_steps.zip.copy1",
        "p2097152": f"{C.F2}/models/stage1_2097152_steps.zip.copy1",
        "c2511_3145728": f"{C.BS2R}/control_2511/01_stance/models/stage1_3145728_steps.zip",
        "c2531_3145728": f"{C.BS2R}/control_2531/01_stance/models/stage1_3145728_steps.zip",
        "c2511_E": f"{C.BS2R}/control_2511/01_stance/models/stage1_final.zip",
        "c2531_E": f"{C.BS2R}/control_2531/01_stance/models/stage1_final.zip",
        "b2511_E": f"{C.BS2R}/critic_blind_tail3_2511/01_stance/models/stage1_final.zip"}
G_IDX = (3, 4, 5, 6, 24, 25, 26, 27, 42, 43, 44)


def cp_upper(x, n, conf=0.95):
    a = 1 - conf
    return 1.0 if x >= n else float(beta_dist.ppf(1 - a / 2, x + 1, n - x))


def records(path, lo, hi):
    c = Counter()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(float(row["timesteps"]))
            if lo < t <= hi:
                c[row["cause"]] += 1
    return dict(c)


def load(label):
    f = np.load(f"{C.TP}/runs/{label}_free.npz" if "hold" not in label else f"{C.TP}/runs/{label}.npz",
                allow_pickle=False)
    stem = f"{C.TP}/runs/{label}_free" if "hold" not in label else f"{C.TP}/runs/{label}"
    eps = [json.loads(x) for x in Path(f"{stem}.episodes.jsonl").read_text().splitlines() if x.strip()]
    meta = json.loads(Path(f"{stem}.meta.json").read_text())
    return f, eps, meta


def main():
    out = {"conditions": {}}
    L = []
    # ---- R6a end reasons
    for lab in ALL7 + HOLDS:
        f, eps, meta = load(lab)
        c = Counter(e["reason"] for e in eps)
        steps = np.bincount(f["all_ep"].astype(np.int64))
        consistent = all(int(steps[e["ep"]]) == int(e["steps"]) for e in eps)
        out["conditions"][lab] = {"n": len(eps), "reasons": dict(c), "steps_consistent": consistent}
        L.append(f"R6a {lab:20s} n {len(eps)} reasons {dict(c)} npz/jsonl step counts consistent {consistent}")
    # ---- R6b records
    par = f"{C.F2}/train_episodes.csv.copy1"
    out["parent_window"] = records(par, 1_048_576, 3_538_944)
    out["parent_before"] = records(par, -1, 1_048_576)
    L.append(f"R6b parent (1,048,576, 3,538,944]: {out['parent_window']} total {sum(out['parent_window'].values())}")
    L.append(f"    parent <= 1,048,576: {out['parent_before']} total {sum(out['parent_before'].values())}")
    bs = Counter()
    per = {}
    for run in sorted(os.listdir(C.BS2R)):
        p = f"{C.BS2R}/{run}/01_stance/train_episodes.csv"
        if os.path.exists(p):
            r = records(p, 2_097_152, 3_538_944)
            per[run] = r
            bs.update(r)
    out["bs2_window"] = dict(bs)
    out["bs2_per_run_nonTrunc"] = {k: {kk: vv for kk, vv in v.items() if kk != "truncated"} for k, v in per.items()}
    L.append(f"R6b BS2 8 runs (2,097,152, 3,538,944]: {dict(bs)} total {sum(bs.values())}; non-truncated per run "
             f"{out['bs2_per_run_nonTrunc']}")
    # ---- R6c pooled slope, C_c, F_bound; K4
    s_all, n_states, Cc = [], 0, {}
    for lab in NB6:
        f, eps, meta = load(lab)
        t = f["all_t"].astype(np.int64)
        epi = f["all_ep"].astype(np.int64)
        sig = math.sqrt(meta["ret_var"] + meta["epsilon"])
        trunc = np.array([e["truncated"] and not e["terminated"] for e in eps])
        m = t <= 800
        s_all.append(f["all_dV3"][m].astype(np.float64))
        n_states += int(m.sum())
        sel = (t >= 200) & (t <= 800) & trunc[epi]
        Cc[lab] = 100.0 / sig + float(np.median(f["all_V"][sel].astype(np.float64)))
    s_crit = float(np.concatenate(s_all).mean())
    n_eps = 768
    up_count = cp_upper(0, n_eps) * n_eps
    S800 = sum(C.GAMMA ** j for j in range(800))
    Dbar_u = up_count * S800 / n_states
    Cbar = float(np.mean(list(Cc.values())))
    F_bound = Cbar * 1.0 * Dbar_u / s_crit
    beta_needed = 0.5 * s_crit / (Cbar * Dbar_u)
    out["R6c"] = {"s_crit": s_crit, "n_states": n_states, "C_c": Cc, "C_mean": Cbar, "CP_upper_count": up_count,
                  "Dbar_u": Dbar_u, "F_bound": F_bound, "beta_needed_per_deg": beta_needed}
    L.append(f"R6c pooled critic slope (NB6, t<=800) {s_crit:+.5f}/deg over {n_states} states; C_c "
             + ", ".join(f"{k} {v:.3f}" for k, v in Cc.items()) + f"; CP upper count {up_count:.3f}; Dbar_u "
             f"{Dbar_u:.4e}; F_bound {F_bound:.4f}; beta needed {beta_needed:.2f}/deg")
    pw = out["parent_window"]
    n_par = sum(pw.values())
    p_u = cp_upper(pw.get("tail_contact", 0), n_par)
    Dbar_rec = p_u * S800 / (n_states / n_eps)
    out["K4"] = {"records_n": n_par, "records_tail_contact": pw.get("tail_contact", 0), "p_upper": p_u,
                 "Dbar_u_records": Dbar_rec, "F_bound_records": Cbar * Dbar_rec / s_crit}
    L.append(f"K4  bound with the parent's training-record rate (0/{n_par}, CP upper {p_u:.5f} per episode): "
             f"F_bound {out['K4']['F_bound_records']:.4f}")
    # ---- R6d M3.1, K1
    for lab in ALL7:
        f, eps, meta = load(lab)
        t = f["all_t"].astype(np.int64)
        s = f["all_dV3"].astype(np.float64)
        if lab == "b2511_E":
            continue
        tip = f["all_tip"].astype(np.float64)
        rho_all = float(spearmanr(s, tip)[0])
        b3 = t >= 400
        rho_b3 = float(spearmanr(s[b3], tip[b3])[0])
        rho_tip_t = float(spearmanr(tip, t)[0])
        auc_tip = float(C.auc(tip[t >= 200], tip[(t >= 50) & (t < 200)]))
        qs = np.quantile(tip[b3], [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(qs, tip[b3])
        quint = [float(s[b3][qi == i].mean()) for i in range(5)]
        q3 = f["all_q3"].astype(np.float64)
        rho_tip_q3_b3 = float(spearmanr(tip[b3], q3[b3])[0])
        out["conditions"][lab].update(M31_rho=rho_all, K1b_rho_B3=rho_b3, K1a_rho_tip_t=rho_tip_t, K1a_auc_tip=auc_tip,
                                      K1c_quintiles=quint, rho_tip_q3_B3=rho_tip_q3_b3)
        L.append(f"R6d/K1 {lab:14s} M3.1 rho(s,tip) {rho_all:+.3f} | K1b within t>=400 {rho_b3:+.3f} | K1a rho(tip,t) "
                 f"{rho_tip_t:+.3f} AUC(tip settled vs 50-200) {auc_tip:.3f} | rho(tip,q3) in B3 {rho_tip_q3_b3:+.3f} | "
                 f"K1c s by tip quintile in B3 " + " ".join(f"{x:+.4f}" for x in quint))
    nflip = sum(out["conditions"][lab]["K1b_rho_B3"] <= -0.10 for lab in NB6)
    out["K1b_count_le_m010"] = int(nflip)
    L.append(f"K1b: rho <= -0.10 within t>=400 at {nflip} of 6 (rule: >= 4 means M3.1 is phase-confounded)")
    # ---- R6e M4.3 and R2 (network read-back on sub states)
    for lab in ALL7:
        f, eps, meta = load(lab)
        std = np.asarray(meta["obs_rms_std"])
        st_t, st_ep = f["sub_t"].astype(np.int64), f["sub_ep"].astype(np.int64)
        z = f["sub_z"].astype(np.float64)
        g = f["sub_g"].astype(np.float64)
        sel = np.abs(z[:, C.OBS_T3]) < 10
        k = {1: 0, 2: 2, 3: 3}
        obs = {1: 3, 2: 5, 3: 6}
        dv = {j: g[sel, k[j]] * (math.pi / 180) / std[obs[j]] for j in (1, 2, 3)}
        dtip = {j: f[f"sub_dtip_dq{j}"][sel].astype(np.float64) for j in (1, 2, 3)}
        v = {j: dv[j].mean() / dtip[j].mean() for j in (1, 2, 3)}
        R31, R32 = v[3] / v[1] if v[1] else float("nan"), v[3] / v[2] if v[2] else float("nan")
        out["conditions"][lab].update(M43_dVdq=[float(dv[j].mean()) for j in (1, 2, 3)],
                                      M43_v=[float(v[j]) for j in (1, 2, 3)], R31=float(R31), R32=float(R32))
        # R2: V and FD on 2,000 sub states vs the recorded all_V / all_dV3
        vnet = C.ValueNet(ZIPS[lab])
        blind = lab == "b2511_E"
        key_all = f["all_ep"].astype(np.int64) * 2000 + f["all_t"].astype(np.int64)
        rng = np.random.default_rng(30101032)
        pick = np.sort(rng.choice(len(st_t), 2000, replace=False))
        pos = np.searchsorted(key_all, st_ep[pick] * 2000 + st_t[pick])
        Vm = vnet.value(z[pick], blind=blind)
        dz = math.radians(0.25) / std[C.OBS_T3]
        zp, zm = z[pick].copy(), z[pick].copy()
        zp[:, C.OBS_T3] = np.clip(zp[:, C.OBS_T3] + dz, -10, 10)
        zm[:, C.OBS_T3] = np.clip(zm[:, C.OBS_T3] - dz, -10, 10)
        fdm = (vnet.value(zp, blind) - vnet.value(zm, blind)) / 0.5
        out["conditions"][lab].update(R2_maxabs_V=float(np.max(np.abs(Vm - f["all_V"][pos]))),
                                      R2_maxabs_FD=float(np.max(np.abs(fdm - f["all_dV3"][pos]))),
                                      R2_mean_FD_mine=float(fdm.mean()), R2_mean_FD_rec=float(f["all_dV3"][pos].mean()))
        L.append(f"R6e {lab:14s} dV/dq per deg q1 {dv[1].mean():+.5f} q2 {dv[2].mean():+.5f} q3 {dv[3].mean():+.5f}; "
                 f"V per m tip {v[1]:+.3f} {v[2]:+.3f} {v[3]:+.3f}; R31 {R31:+.3f} R32 {R32:+.3f} | R2 (float16 z) max "
                 f"|V mine - V| {out['conditions'][lab]['R2_maxabs_V']:.2e}, max |FD mine - dV3| "
                 f"{out['conditions'][lab]['R2_maxabs_FD']:.2e} (means {fdm.mean():+.5f} vs "
                 f"{f['all_dV3'][pos].mean():+.5f})")
    txt = "\n".join(L)
    Path(C.HERE / "redo_term.json").write_text(json.dumps(out, indent=1))
    Path(C.HERE / "redo_term.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
