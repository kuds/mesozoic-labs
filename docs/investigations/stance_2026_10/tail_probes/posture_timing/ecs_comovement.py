"""TEST 1, ECS: within the s42 lineage (12 continuations from 3,145,728) and the s80 lineage (12 from 4,194,304), do
posture and the tail_3 command move together step by step?  Read only on r10round/ecs/runs.  No simulation.
Output: ecs_comovement.txt (stdout), ecs_series.json.

Per continuation, only rows after the start S are used (the parent's rows are seeded into each ledger and probe):
  LEDGER per evaluation (12, deterministic, 10 episodes): tail_3 command, tail over-bar episodes, pelvis pitch, CoM
         fore-aft, head tip height, neck / head joint, root height
  PROBE  per rollout (191): tail_3 mu; recovered observation means (pitch from projected gravity, tail_3, neck, head)
  DIAG   diagnostics.npz per rollout (191): tilt_angle, head_tip_z, pelvis_height (the run's own training means)
  TRAIN  train_episodes.csv full-horizon episodes after S: CoM fore-aft, pitch
The parent's value at S is its own ledger row at S (the evaluation the continuation starts from).
"""
import glob
import json
import math
import os

import numpy as np

import pt_common as C

S = {"s42": 3_145_728, "s80": 4_194_304}
LCOLS = ["tail_3_pitch_act_command_mean", "root_pitch_mean_deg", "com_fore_aft_mean_m", "head_tip_height_mean_m",
         "neck_pitch_mean_deg", "head_pitch_mean_deg", "root_height_mean_m", "tail_3_pitch_mean_deg",
         "head_pitch_act_command_mean", "neck_pitch_act_command_mean"]


def sign_test_p(k, n):
    """Two-sided exact sign test."""
    if n == 0:
        return float("nan")
    m = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, m + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(name):
    lin = name[:3]
    s = S[lin]
    p = f"{C.ECS_RUNS}/{name}/01_stance"
    rows = C.read_csv(f"{p}/eval_episodes.csv")
    L = C.ledger_per_eval(rows, LCOLS)
    L["tail_ob"] = C.tail_overbar_counts(L["_by"], L["t"])
    L["any_ob"] = C.any_overbar_counts(L["_by"], L["t"])
    P = C.probe_batch_means(f"{p}/policy_probe.npz")
    bm = P["batch_mean"]
    pr = {"t": P["t"], "tail_mu": P["rollout_mu_mean"][:, C.TAIL3].astype(float),
          "pitch": C.pitch_from_gx(bm[:, C.OBS["g_x"]]), "q3": bm[:, C.OBS["tail_3_pitch"]] * C.R2D,
          "neck": bm[:, C.OBS["neck_pitch"]] * C.R2D, "head": bm[:, C.OBS["head_pitch"]] * C.R2D}
    D = np.load(f"{p}/diagnostics.npz", allow_pickle=False)
    dg = {"t": D["timesteps"], "tilt": D["tilt_angle"], "head_z": D["head_tip_z"], "pel_h": D["pelvis_height"]}
    T = C.train_episode_series(f"{p}/train_episodes.csv")
    return s, L, pr, dg, T


def main():
    names = sorted(os.path.basename(os.path.dirname(x)) for x in glob.glob(f"{C.ECS_RUNS}/*/01_stance"))
    print("TEST 1 / ECS within-lineage co-movement (ecs_comovement.py; read only on r10round/ecs/runs)")
    print("Exploratory, report only, correlational.  n = 12 continuations per lineage (6 per arm); arms pooled "
          "unless stated (the screen found no arm difference in flagging).\n")
    res = {}
    for nm in names:
        s, L, pr, dg, T = load(nm)
        aft = L["t"] > s
        at_s = np.where(L["t"] == s)[0]
        i_s = int(at_s[0])
        ev = {k: L[k][aft] for k in LCOLS}
        ev["t"] = L["t"][aft]
        ev["tail_ob"] = L["tail_ob"][aft]
        ev["any_ob"] = L["any_ob"][aft]
        s_val = {k: float(L[k][i_s]) for k in LCOLS}
        pa = pr["t"] > s
        ta = T["t"] > s
        first_tail_ob = next((int(t) for t, k in zip(ev["t"], ev["tail_ob"]) if k >= 1), None)
        first_tail_flag = next((int(t) for t, k in zip(ev["t"], ev["tail_ob"]) if k >= 5), None)
        first_any_flag = next((int(t) for t, k in zip(ev["t"], ev["any_ob"]) if k >= 5), None)
        res[nm] = {"s": s, "ev": {k: [float(x) for x in v] for k, v in ev.items()}, "s_val": s_val,
                   "first_tail_ob": first_tail_ob, "first_tail_flag": first_tail_flag, "first_any_flag": first_any_flag,
                   "pr": {k: [float(x) for x in np.asarray(v)[pa]] for k, v in pr.items()},
                   "dg": {k: [float(x) for x in np.asarray(v)] for k, v in dg.items()},
                   "tr": {"t": [int(x) for x in T["t"][ta]], "com": [float(x) for x in T["com"][ta]],
                          "pitch": [float(x) for x in T["pitch"][ta]]}}

    # Parent rows at S: check they are the same in every continuation of a lineage
    print("0. PARENT ROW AT S (ledger, the evaluation each continuation starts from) [measured]")
    for lin in ("s42", "s80"):
        vals = {json.dumps({k: round(v, 9) for k, v in res[n]["s_val"].items()}) for n in names if n.startswith(lin)}
        sv = res[[n for n in names if n.startswith(lin)][0]]["s_val"]
        print(f"  {lin} at {S[lin]:,}: identical in all 12: {len(vals) == 1}; tail cmd {sv['tail_3_pitch_act_command_mean']:+.3f}, "
              f"pitch {sv['root_pitch_mean_deg']:+.3f} deg, CoM {sv['com_fore_aft_mean_m']*1000:+.2f} mm, head tip "
              f"{sv['head_tip_height_mean_m']:.3f} m, neck {sv['neck_pitch_mean_deg']:+.2f}, head {sv['head_pitch_mean_deg']:+.2f} deg")
    print()

    # 1. Per-continuation table: start -> each evaluation (ledger), compact
    print("1. PER CONTINUATION, LEDGER (deterministic, 10 episodes per evaluation): first tail over-bar episode, first "
          "tail-flagged evaluation, and change from S to the mean of the last 3 evaluations (end):")
    print("   name         1st tail ob  1st tail flag  1st ANY flag   d tail_cmd  d pitch  d CoM mm  d head m  d neck  d headj  d hgt mm")
    for nm in names:
        r = res[nm]
        ev, sv = r["ev"], r["s_val"]
        def d(k, scale=1.0):
            return (np.mean(ev[k][-3:]) - sv[k]) * scale
        r["delta_end"] = {"tail": d("tail_3_pitch_act_command_mean"), "pitch": d("root_pitch_mean_deg"),
                          "com": d("com_fore_aft_mean_m", 1000), "head": d("head_tip_height_mean_m"),
                          "neck": d("neck_pitch_mean_deg"), "headj": d("head_pitch_mean_deg"),
                          "hgt": d("root_height_mean_m", 1000)}
        de = r["delta_end"]
        print(f"   {nm:12s} {C.fmt(r['first_tail_ob']):>11s}  {C.fmt(r['first_tail_flag']):>13s}  {C.fmt(r['first_any_flag']):>12s}"
              f"   {de['tail']:+9.3f}  {de['pitch']:+7.3f}  {de['com']:+8.2f}  {de['head']:+8.3f}  {de['neck']:+6.2f}  {de['headj']:+7.2f}  {de['hgt']:+7.2f}")
    print()
    print("   sign counts of the start-to-end change, per lineage (n 12; exact two-sided sign test p, description only):")
    for lin in ("s42", "s80"):
        sub = [n for n in names if n.startswith(lin)]
        cells = []
        for k in ("tail", "pitch", "com", "head", "neck", "headj", "hgt"):
            v = np.array([res[n]["delta_end"][k] for n in sub])
            kpos = int((v > 0).sum())
            cells.append(f"{k} median {np.median(v):+.3f} (+{kpos}/-{12-kpos}, p {sign_test_p(kpos, 12):.3f})")
        print(f"   {lin}: " + "; ".join(cells))
    print()

    # 2. Within-continuation co-movement over time
    print("2. WITHIN EACH CONTINUATION, OVER TIME: Spearman rho across its 12 evaluations between the ledger tail_3 command "
          "and each posture signal; and across its 191 rollouts between the probe tail_3 mu and the training-time posture "
          "(probe pitch, neck, head; diagnostics head_tip_z, tilt).  Median rho and sign counts per lineage.")
    pairs_ev = [("pitch", "root_pitch_mean_deg"), ("com", "com_fore_aft_mean_m"), ("head tip", "head_tip_height_mean_m"),
                ("neck", "neck_pitch_mean_deg"), ("headj", "head_pitch_mean_deg"), ("height", "root_height_mean_m")]
    for lin in ("s42", "s80"):
        sub = [n for n in names if n.startswith(lin)]
        cells = []
        for lab, k in pairs_ev:
            rs = np.array([C.spearman(res[n]["ev"]["tail_3_pitch_act_command_mean"], res[n]["ev"][k]) for n in sub])
            cells.append(f"{lab} {np.nanmedian(rs):+.2f} (+{int((rs > 0).sum())}/-{int((rs < 0).sum())})")
        print(f"   {lin} ledger (12 evals):   " + "; ".join(cells))
        cells = []
        for lab, src, k in (("pitch", "pr", "pitch"), ("neck", "pr", "neck"), ("headj", "pr", "head"),
                            ("head_z", "dg", "head_z"), ("tilt", "dg", "tilt"), ("pel_h", "dg", "pel_h")):
            rs = []
            for n in sub:
                tm = np.array(res[n]["pr"]["tail_mu"])
                y = np.array(res[n][src][k])
                m = min(len(tm), len(y))
                rs.append(C.spearman(tm[-m:], y[-m:]))
            rs = np.array(rs)
            cells.append(f"{lab} {np.nanmedian(rs):+.2f} (+{int((rs > 0).sum())}/-{int((rs < 0).sum())})")
        print(f"   {lin} rollouts (191):      " + "; ".join(cells))
    print("   (pitch + nose up; neck/head joint + lowers the head; tilt_angle is unsigned)\n")

    # 3. Lead-lag on 8-rollout bins, pooled within lineage
    print("3. LEAD-LAG, POOLED PER LINEAGE (descriptive): first differences of 8-rollout (65,536-step) bin means; Pearson "
          "r of d posture(t) with d tail_mu(t+lag); lag > 0 = posture change precedes the tail change.  Each "
          "continuation contributes its 23 bins; pooled n given.")
    for lin in ("s42", "s80"):
        sub = [n for n in names if n.startswith(lin)]
        for lab, src, k in (("probe pitch", "pr", "pitch"), ("probe neck", "pr", "neck"), ("probe head", "pr", "head"),
                            ("diag head_z", "dg", "head_z")):
            rs_by_lag = {lag: ([], []) for lag in range(-4, 5)}
            for n in sub:
                tm = np.array(res[n]["pr"]["tail_mu"])
                y = np.array(res[n][src][k])
                m = min(len(tm), len(y))
                tm, y = tm[-m:], y[-m:]
                nb = m // 8
                tb = tm[m - nb * 8:].reshape(nb, 8).mean(1)
                yb = y[m - nb * 8:].reshape(nb, 8).mean(1)
                dtb, dyb = np.diff(tb), np.diff(yb)
                for lag in range(-4, 5):
                    if lag >= 0:
                        a, b = dyb[:len(dyb) - lag], dtb[lag:]
                    else:
                        a, b = dyb[-lag:], dtb[:len(dtb) + lag]
                    rs_by_lag[lag][0].extend(a)
                    rs_by_lag[lag][1].extend(b)
            out = []
            for lag in range(-4, 5):
                a, b = np.array(rs_by_lag[lag][0]), np.array(rs_by_lag[lag][1])
                out.append(np.corrcoef(a, b)[0, 1])
            nn = len(rs_by_lag[0][0])
            print(f"   {lin} {lab:12s} n(lag 0) {nn:3d}  r(lag -4..+4) " + " ".join(f"{x:+.2f}" for x in out))
    print()

    # 4. s42: posture before the first tail over-bar episode / flag, against S
    print("4. s42 CONTINUATIONS: STATE AT THE LAST EVALUATION BEFORE THE FIRST TAIL OVER-BAR EPISODE, minus S (ledger)")
    print("   name          eval before   d tail_cmd  d pitch  d CoM mm  d head m  d neck  d headj")
    pre = []
    for nm in [n for n in names if n.startswith("s42")]:
        r = res[nm]
        ev, sv = r["ev"], r["s_val"]
        fo = r["first_tail_ob"]
        idx = [i for i, t in enumerate(ev["t"]) if fo is None or t < fo]
        if not idx:
            print(f"   {nm:12s} first over-bar at the first evaluation after S; no pre-onset evaluation")
            continue
        i = idx[-1]
        row = (ev["tail_3_pitch_act_command_mean"][i] - sv["tail_3_pitch_act_command_mean"],
               ev["root_pitch_mean_deg"][i] - sv["root_pitch_mean_deg"],
               (ev["com_fore_aft_mean_m"][i] - sv["com_fore_aft_mean_m"]) * 1000,
               ev["head_tip_height_mean_m"][i] - sv["head_tip_height_mean_m"],
               ev["neck_pitch_mean_deg"][i] - sv["neck_pitch_mean_deg"], ev["head_pitch_mean_deg"][i] - sv["head_pitch_mean_deg"])
        pre.append(row)
        print(f"   {nm:12s} {int(ev['t'][i]):>11,d}   " + "  ".join(f"{x:+8.3f}" for x in row))
    if pre:
        a = np.array(pre)
        print("   median  " + " " * 16 + "  ".join(f"{x:+8.3f}" for x in np.median(a, 0)) + f"   (n {len(pre)})")
    print()

    # 5. Between continuations: does an earlier first flag go with a bigger early posture shift?
    print("5. BETWEEN CONTINUATIONS (n 12 per lineage), Spearman rho, descriptive:")
    for lin in ("s42", "s80"):
        sub = [n for n in names if n.startswith(lin)]
        e2 = {}
        for n in sub:
            ev, sv = res[n]["ev"], res[n]["s_val"]
            e2[n] = {"pitch2": np.mean(ev["root_pitch_mean_deg"][:2]) - sv["root_pitch_mean_deg"],
                     "com2": (np.mean(ev["com_fore_aft_mean_m"][:2]) - sv["com_fore_aft_mean_m"]) * 1000,
                     "head2": np.mean(ev["head_tip_height_mean_m"][:2]) - sv["head_tip_height_mean_m"],
                     "tail2": np.mean(ev["tail_3_pitch_act_command_mean"][:2]) - sv["tail_3_pitch_act_command_mean"]}
        if lin == "s42":
            ff = [res[n]["first_tail_ob"] or 5_000_000 for n in sub]
            for k in ("pitch2", "com2", "head2", "tail2"):
                print(f"   s42 first tail over-bar step vs {k} (change over the first 2 evaluations): rho "
                      f"{C.spearman(ff, [e2[n][k] for n in sub]):+.2f}")
        endt = [np.mean(res[n]["ev"]["tail_3_pitch_act_command_mean"][-3:]) for n in sub]
        for k, lab in (("root_pitch_mean_deg", "pitch"), ("com_fore_aft_mean_m", "CoM"), ("head_tip_height_mean_m", "head tip"),
                       ("neck_pitch_mean_deg", "neck")):
            endp = [np.mean(res[n]["ev"][k][-3:]) for n in sub]
            print(f"   {lin} end level (last 3 evals) tail_3 command vs {lab}: rho {C.spearman(endt, endp):+.2f}")
        print(f"   {lin} end tail_3 command range {min(endt):+.3f} .. {max(endt):+.3f}")
    print()

    # 6. Time course per lineage: median over continuations at each evaluation (ledger), relative to S
    print("6. LINEAGE MEDIAN TIME COURSE (ledger, change from S; median over the 12 continuations at each evaluation)")
    for lin in ("s42", "s80"):
        sub = [n for n in names if n.startswith(lin)]
        ts = res[sub[0]]["ev"]["t"]
        print(f"   {lin}: step(M)  tail_cmd  pitch    CoM mm   head m   neck    headj   tail_ob(sum of 12)")
        for i, t in enumerate(ts):
            def med(k, sc=1.0):
                return np.median([(res[n]["ev"][k][i] - res[n]["s_val"][k]) * sc for n in sub])
            ob = int(sum(res[n]["ev"]["tail_ob"][i] for n in sub))
            print(f"        {t/1e6:7.3f}  {med('tail_3_pitch_act_command_mean'):+7.3f}  {med('root_pitch_mean_deg'):+6.3f}  "
                  f"{med('com_fore_aft_mean_m', 1000):+7.2f}  {med('head_tip_height_mean_m'):+7.3f}  "
                  f"{med('neck_pitch_mean_deg'):+6.2f}  {med('head_pitch_mean_deg'):+6.2f}   {ob:3d}")
    with open(f"{C.OUT}/ecs_series.json", "w") as fh:
        json.dump({n: {k: res[n][k] for k in ("s", "s_val", "first_tail_ob", "first_tail_flag", "first_any_flag",
                                              "delta_end")} for n in names}, fh, indent=1)


if __name__ == "__main__":
    main()
