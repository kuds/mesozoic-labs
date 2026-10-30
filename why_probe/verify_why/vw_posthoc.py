"""verify_why POST HOC (not pre-registered; descriptive).  Offline; writes posthoc.json/.txt here.

PH1 K1b decomposition within t >= 400: Spearman(s, q3), partial Spearman(s, tip | q3), (s, low | q3), (s, fwdz | q3)
    (termination probe's NB6 runs; my FR1, FR3).  Partial = Pearson of rank residuals on rank(q3).
PH2 the push on the tail_3 command mean with SB3's OWN advantages (the training signal before minibatch
    normalisation): mean A x (a14 - mu14) / sigma14^2 over complete-episode states, vs the same with Monte Carlo
    advantages G - V (unbiased for the realised returns: a state-only baseline does not move the expected push).
PH3 the parent's tail_contact training episodes by 131,072-step bin (records).
PH4 the critic's slope by bin and the B3/B1 ratio on the fresh runs (descriptive).
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
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vw_common as C  # noqa: E402

NB6 = ("p1048576", "p2097152", "c2511_3145728", "c2531_3145728", "c2511_E", "c2531_E")


def partial_rank(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    out, L = {}, []
    # ---- PH1 (termination probe runs)
    for lab in NB6:
        f = np.load(f"{C.TP}/runs/{lab}_free.npz", allow_pickle=False)
        t = f["all_t"].astype(np.int64)
        b3 = t >= 400
        s, q3, tip, low = (f[k][b3].astype(np.float64) for k in ("all_dV3", "all_q3", "all_tip", "all_low"))
        fz = f["all_fwdz"][b3].astype(np.float64)
        r = {"rho_s_q3": float(spearmanr(s, q3)[0]), "prho_s_tip_q3": partial_rank(s, tip, q3),
             "prho_s_low_q3": partial_rank(s, low, q3), "prho_s_fwdz_q3": partial_rank(s, fz, q3),
             "rho_s_tip": float(spearmanr(s, tip)[0])}
        out[f"PH1_{lab}"] = r
        L.append(f"PH1 {lab:14s} within t>=400: rho(s,tip) {r['rho_s_tip']:+.3f}  rho(s,q3) {r['rho_s_q3']:+.3f}  "
                 f"partial rho(s,tip|q3) {r['prho_s_tip_q3']:+.3f}  (s,low|q3) {r['prho_s_low_q3']:+.3f}  "
                 f"(s,fwdz|q3) {r['prho_s_fwdz_q3']:+.3f}")
    for name, path in (("FR1", "runs/FR1_C11E_stoch.npz"), ("FR3", "runs/FR3_P2_stoch.npz")):
        f = np.load(C.HERE / path, allow_pickle=False)
        meta = json.loads(str(f["meta_json"]))
        k = f["k"].astype(np.int64)
        b3 = k >= 400
        s = f["fd6"][b3].astype(np.float64)
        q3 = np.degrees(f["raw"][b3, C.OBS_T3].astype(np.float64))
        tip = f["phys"][b3, 0].astype(np.float64)
        low = f["phys"][b3, 1].astype(np.float64)
        r = {"rho_s_q3": float(spearmanr(s, q3)[0]), "prho_s_tip_q3": partial_rank(s, tip, q3),
             "prho_s_low_q3": partial_rank(s, low, q3), "rho_s_tip": float(spearmanr(s, tip)[0])}
        out[f"PH1_{name}"] = r
        L.append(f"PH1 {name:14s} within k>=400: rho(s,tip) {r['rho_s_tip']:+.3f}  rho(s,q3) {r['rho_s_q3']:+.3f}  "
                 f"partial rho(s,tip|q3) {r['prho_s_tip_q3']:+.3f}  (s,low|q3) {r['prho_s_low_q3']:+.3f}")
        # ---- PH2 push
        sl = C.episodes(f["ep"])
        n_ep = len(sl)
        trunc, vterm = f["ep_trunc"], f["ep_vterm"]
        G = C.returns_to_go(f["rn"], sl, np.where(trunc, vterm, 0.0))
        env, start = f["ep_env"].astype(np.int64), f["ep_start"].astype(np.int64)
        adv = np.concatenate([f["buf_advantages"][env[e], start[e]:start[e] + len(sl[e])] for e in range(n_ep)])
        sig14 = math.exp(meta["log_std"][14])
        score = (f["a"][:, 14].astype(np.float64) - f["mu"][:, 14].astype(np.float64)) / sig14 ** 2
        a_mc = G - f["V"]
        bins = C.bin_of(k)
        res = {}
        rng = np.random.default_rng(30101035)
        for nm, A in (("sb3_gae", adv), ("mc", a_mc)):
            per = np.array([(A[s_] * score[s_]).sum() for s_ in sl])
            n_ = np.array([len(s_) for s_ in sl], dtype=float)
            est = float(per.sum() / n_.sum())
            reps = []
            for _ in range(1000):
                m = np.bincount(rng.integers(0, n_ep, n_ep), minlength=n_ep).astype(float)
                reps.append(float((m * per).sum() / (m * n_).sum()))
            by_bin = [float((A * score)[bins == i].mean()) for i in range(4)]
            res[nm] = {"est": est, "ci": C.ci(reps), "by_bin": by_bin}
        out[f"PH2_{name}"] = res
        L.append(f"PH2 {name} push on mu14 (mean A x (a14-mu14)/sigma^2; sigma14 {sig14:.3f}): SB3 GAE advantages "
                 f"{res['sb3_gae']['est']:+.5f} [{res['sb3_gae']['ci'][0]:+.5f}, {res['sb3_gae']['ci'][1]:+.5f}] by bin "
                 + " ".join(f"{x:+.5f}" for x in res["sb3_gae"]["by_bin"]) + f";  Monte Carlo G - V "
                 f"{res['mc']['est']:+.5f} [{res['mc']['ci'][0]:+.5f}, {res['mc']['ci'][1]:+.5f}] by bin "
                 + " ".join(f"{x:+.5f}" for x in res["mc"]["by_bin"]))
        fd = f["fd6"].astype(np.float64)
        out[f"PH4_{name}"] = [float(fd[bins == i].mean()) for i in range(4)]
    # ---- PH3 records
    c = Counter()
    with open(f"{C.F2}/train_episodes.csv.copy1", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["cause"] == "tail_contact":
                c[int(float(row["timesteps"])) // 131072] += 1
    out["PH3_parent_tail_contact_by_131072_bin"] = {str(k_ * 131072): v for k_, v in sorted(c.items())}
    L.append(f"PH3 parent tail_contact training episodes by 131,072-step bin (bin start: count): "
             f"{out['PH3_parent_tail_contact_by_131072_bin']}")
    txt = "\n".join(L)
    Path(C.HERE / "posthoc.json").write_text(json.dumps(out, indent=1))
    Path(C.HERE / "posthoc.txt").write_text("POST HOC (not pre-registered; descriptive)\n" + txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
