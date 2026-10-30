"""TEST 1, r9 (physics revision 9): stance5 LTSB s42 / s52 / s60 / s70 / s80 and stance4 LTS s42 / s52 / s60.
Tail signals and posture signals over training.  Read only.  No simulation.  Output: r9_timeline.txt (stdout).

What r9 has locally (inventory.txt):
  ledger eval_episodes.csv (63 columns; no actuator columns: an over-bar episode is one listing
    max_actuator_saturation_fraction in screen_v2_reasons, actuator unnamed; com_fore_aft_mean_m, root_pitch_mean_deg)
  stance5 policy_probe.npz (tailround/fetch; older schema: no rollout_mu_*): the recovered per-rollout observation
    means (tail_3 angle, pitch from projected gravity, neck / head joints) and bank_action_now at statue_settled
  train_episodes.csv for s70 and s80 only (stance5 replication)
The r9 tail command is not recorded per rollout.  Its proxy is the training-time tail_3 angle q3, turned into an
implied command with the r9 options study's plant map q3 = -0.0924 + 0.0863 a (rad), measured at the statue posture
(tailround/posture/brief_draft.md l48) [inferred proxy; the map's validity away from the statue posture is not tested].
Onsets as the records give them: rail_check2.json (r10round/draft/verify), the M5 rule on the r9 ledgers (first
over-bar episode, first flagged evaluation with >= 5 of 10, ALARM = two consecutive flagged).
"""
import json

import numpy as np

import pt_common as C

ONS = json.load(open(f"{C.S2}/r10round/draft/verify/rail_check2.json"))
ST5 = ("stance5_LTSB_s42", "stance5_LTSB_s52", "stance5_LTSB_s60", "stance5_LTSB_s70", "stance5_LTSB_s80")
ST4 = ("stance4_LTS_s42", "stance4_LTS_s52", "stance4_LTS_s60")
REF = (524_288, 1_048_576)


def load(run):
    p = C.R9_RUNS[run]
    rows = C.read_csv(f"{p}/eval_episodes.csv")
    L = C.ledger_per_eval(rows, ["root_pitch_mean_deg", "com_fore_aft_mean_m", "leg_q_l_r_mean_hip_pitch_deg"])
    L["sat"] = C.r9_sat_reason_counts(L["_by"], L["t"])
    pr = None
    if run in C.R9_PROBE:
        P = C.probe_batch_means(C.R9_PROBE[run])
        bm = P["batch_mean"]
        q3 = bm[:, C.OBS["tail_3_pitch"]]
        d = np.load(C.R9_PROBE[run], allow_pickle=False)
        i0 = list(d["bank_labels"]).index("statue_settled")
        pr = {"t": P["t"], "q3deg": q3 * C.R2D, "a_imp": (q3 + 0.0924) / 0.0863, "pitch": C.pitch_from_gx(bm[:, C.OBS["g_x"]]),
              "neck": bm[:, C.OBS["neck_pitch"]] * C.R2D, "head": bm[:, C.OBS["head_pitch"]] * C.R2D,
              "bankT": d["bank_action_now"][:, i0, C.TAIL3].astype(float)}
    T = None
    try:
        T = C.train_episode_series(f"{p}/train_episodes.csv")
    except FileNotFoundError:
        pass
    return L, pr, T


def ref_of(t, y):
    t, y = np.asarray(t), np.asarray(y)
    sel = (t >= REF[0]) & (t < REF[1]) & ~np.isnan(y)
    return float(np.median(y[sel]))


def main():
    print("TEST 1 / r9: tail and posture over training (r9_timeline.py; read only)\n")
    data = {r: load(r) for r in ST5 + ST4}
    edges = np.arange(0, 12_058_624 + 1, 262_144)
    mids = edges[1:]
    series = {}
    print("0. CHECK: probe-recovered pitch vs training-episode pitch (s70, s80; 262,144-step bins from 524,288) and vs "
          "ledger pitch (all stance5) [measured]")
    for r in ST5:
        L, pr, T = data[r]
        a = C.bin_series(pr["t"], pr["pitch"], edges)
        b = C.bin_series(L["t"], L["root_pitch_mean_deg"], edges)
        ok = ~np.isnan(a) & ~np.isnan(b)
        msg = f"  {r}: vs ledger r {np.corrcoef(a[ok][2:], b[ok][2:])[0, 1]:+.3f}"
        if T is not None:
            c = C.bin_series(T["t"], T["pitch"], edges)
            ok2 = ~np.isnan(a) & ~np.isnan(c)
            msg += (f"; vs training episodes r {np.corrcoef(a[ok2][2:], c[ok2][2:])[0, 1]:+.3f}, median offset "
                    f"{np.median(a[ok2][2:] - c[ok2][2:]):+.3f} deg")
        print(msg)
    print()
    for r in ST5 + ST4:
        L, pr, T = data[r]
        cols = {"sat_eps": np.array([L["sat"][(L["t"] >= a) & (L["t"] < b)].sum() for a, b in zip(edges[:-1], edges[1:])], float),
                "L_pitch": C.bin_series(L["t"], L["root_pitch_mean_deg"], edges),
                "L_com": C.bin_series(L["t"], L["com_fore_aft_mean_m"] * 1000, edges),
                "L_hip": C.bin_series(L["t"], L["leg_q_l_r_mean_hip_pitch_deg"], edges)}
        if pr is not None:
            cols.update({"a_imp": C.bin_series(pr["t"], pr["a_imp"], edges), "q3deg": C.bin_series(pr["t"], pr["q3deg"], edges),
                         "bankT": C.bin_series(pr["t"], pr["bankT"], edges), "P_pitch": C.bin_series(pr["t"], pr["pitch"], edges),
                         "P_neck": C.bin_series(pr["t"], pr["neck"], edges), "P_headj": C.bin_series(pr["t"], pr["head"], edges)})
        if T is not None:
            cols.update({"T_pitch": C.bin_series(T["t"], T["pitch"], edges), "T_com": C.bin_series(T["t"], T["com"] * 1000, edges)})
        series[r] = cols
        o = ONS[r]
        print(f"1. {r}: records: first over-bar episode {C.fmt(o['first_sat'])}, first flagged evaluation "
              f"{C.fmt(o['first_flag'])}, ALARM {C.fmt(o['alarm'])}, over-bar episodes {o['sat_eps']}")
        print("   step(M)  " + " ".join(f"{k:>8s}" for k in cols))
        for i, m in enumerate(mids):
            if m > 12_000_000 or (m > 6_300_000 and i % 2):
                continue
            print(f"   {m/1e6:7.3f}  " + " ".join(f"{cols[k][i]:8.3f}" for k in cols))
        print()

    print("2. TIMING PER stance5 RUN (tail rise start on the implied command a_imp, 16-rollout mean sustained 16 rollouts; "
          "posture shifts against the run's own [524,288, 1,048,576) reference, pre-stated thresholds pitch 0.3 deg, "
          "CoM 3 mm, neck/head joint 3 deg, searched after 1,048,576)")
    for r in ST5:
        L, pr, T = data[r]
        sm = C.smooth(pr["a_imp"], 16)
        a_ref = ref_of(pr["t"], sm)
        trs = {thr: C.first_sustained(pr["t"], sm, a_ref + thr, True, 16) for thr in (0.1, 0.2, 0.3, 0.5)}
        print(f"  {r}: a_imp reference {a_ref:+.3f}; TRS (rise of a_imp above its reference by) "
              + ", ".join(f"{k}: {C.fmt(v)}" for k, v in trs.items()))
        specs = [("probe pitch (deg)", pr["t"], C.smooth(pr["pitch"], 16), -0.3, 16),
                 ("ledger pitch (deg)", L["t"], L["root_pitch_mean_deg"], -0.3, 2),
                 ("ledger CoM f-a (mm)", L["t"], L["com_fore_aft_mean_m"] * 1000, +3.0, 2),
                 ("probe neck joint (deg)", pr["t"], C.smooth(pr["neck"], 16), -3.0, 16),
                 ("probe head joint (deg)", pr["t"], C.smooth(pr["head"], 16), -3.0, 16)]
        if T is not None:
            specs.insert(1, ("train-ep pitch (deg)", T["t"], C.smooth(T["pitch"], 64), -0.3, 64))
            specs.insert(3, ("train-ep CoM f-a (mm)", T["t"], C.smooth(T["com"] * 1000, 64), +3.0, 64))
        for name, t, y, thr, k in specs:
            t, y = np.asarray(t), np.asarray(y)
            ref = ref_of(t, y)
            aft = t >= REF[1]
            hyp = C.first_sustained(t[aft], y[aft], ref + thr, thr > 0, k)
            opp = C.first_sustained(t[aft], y[aft], ref - thr, thr < 0, k)
            print(f"      {name:24s} ref {ref:+8.3f}  shift {thr:+.2f}: hyp {C.fmt(hyp):>11s}   opposite {C.fmt(opp):>11s}")
    print()

    print("3. CROSS-RUN GAPS AT MATCHED STEPS: railing run minus the median of the non-alarm stance5 runs (s52, s60, s70) "
          "[measured; bin medians]")
    keys = ["L_pitch", "P_pitch", "L_com", "P_neck", "P_headj", "a_imp", "q3deg"]
    for r in ("stance5_LTSB_s42", "stance5_LTSB_s80"):
        print(f"  {r}:")
        for lo, hi in ((524_288, 1_048_576), (1_048_576, 2_097_152), (2_097_152, 3_145_728), (3_145_728, 3_670_016),
                       (3_670_016, 4_456_448), (4_456_448, 6_291_456), (6_291_456, 12_058_624)):
            sel = (mids > lo) & (mids <= hi)
            cells = []
            for k in keys:
                ctrl = np.nanmedian([np.nanmedian(series[c][k][sel]) for c in ("stance5_LTSB_s52", "stance5_LTSB_s60", "stance5_LTSB_s70")])
                cells.append(f"{k} {np.nanmedian(series[r][k][sel]) - ctrl:+.2f}")
            print(f"     [{lo/1e6:.2f}, {hi/1e6:.2f}]M: " + "; ".join(cells))
    print("  stance4_LTS_s42 minus the median of stance4 s52 and s60 (ledger only):")
    for lo, hi in ((524_288, 1_048_576), (1_048_576, 1_835_008), (1_835_008, 2_228_224), (2_228_224, 3_145_728),
                   (3_145_728, 6_291_456), (6_291_456, 12_058_624)):
        sel = (mids > lo) & (mids <= hi)
        cells = []
        for k in ("L_pitch", "L_com", "L_hip"):
            ctrl = np.nanmedian([np.nanmedian(series[c][k][sel]) for c in ("stance4_LTS_s52", "stance4_LTS_s60")])
            cells.append(f"{k} {np.nanmedian(series['stance4_LTS_s42'][k][sel]) - ctrl:+.2f}")
        print(f"     [{lo/1e6:.2f}, {hi/1e6:.2f}]M: " + "; ".join(cells))
    print()

    print("4. LEAD-LAG (descriptive): first differences of 65,536-step bin means, d posture(t) vs d a_imp(t+lag); lag > 0 = "
          "posture leads.  Window [0.5M, first ALARM + 2M] for s42 / s80; [0.5M, 8M] for the others.")
    e = np.arange(0, 12_058_624 + 1, 65_536)
    for r in ST5:
        L, pr, T = data[r]
        al = ONS[r]["alarm"]
        hi_end = (al + 2_000_000) if al else 8_000_000
        w = (e[1:] > 524_288) & (e[1:] <= hi_end)
        tail = C.bin_series(pr["t"], pr["a_imp"], e)
        for nm, y in (("probe pitch", C.bin_series(pr["t"], pr["pitch"], e)), ("probe neck", C.bin_series(pr["t"], pr["neck"], e)),
                      ("probe head", C.bin_series(pr["t"], pr["head"], e))):
            dt_, dy = np.diff(tail[w]), np.diff(y[w])
            rs = []
            for lag in range(-6, 7):
                a, b = (dy[:len(dy) - lag], dt_[lag:]) if lag >= 0 else (dy[-lag:], dt_[:len(dt_) + lag])
                ok = ~np.isnan(a) & ~np.isnan(b)
                rs.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 5 else np.nan)
            print(f"  {r:17s} {nm:12s} n {w.sum()-1:3d}  r(lag -6..+6) " + " ".join(f"{x:+.2f}" for x in rs))


if __name__ == "__main__":
    main()
