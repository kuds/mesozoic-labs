"""TEST 1, r10 stance6 s42 / s60 / s80: tail_3 signals and posture signals lined up over training steps.
Read only (F2 copies).  No simulation.  Output: r10_timeline.txt (stdout), r10_series.npz.

Signal sources (all per run):
  LEDGER  eval_episodes.csv: 91 evaluations x 10 deterministic episodes (window means as the ledger records them)
  PROBE   policy_probe.npz: 1,465 rollouts (8,192 steps, stochastic policy); rollout_mu_mean[:, 14] = the policy's
          mean tail_3 command; per-rollout observation means recovered from obs_rms (pt_common.probe_batch_means):
          tail_3 / neck / head joint angles, pelvis pitch from projected gravity; bank_action_now at the fixed
          'statue_settled' observation (bank row 0) = the policy's own tail_3 command at one fixed state
  TRAIN   train_episodes.csv: per training episode (stochastic), full-horizon only: CoM fore-aft and pelvis pitch

Pre-stated (before any time course was read) thresholds for a "posture shift", from the size of the known lineage
gaps (pitch 1.1 deg, CoM 6.7 mm, head tip 0.35 m): pitch 0.3 deg, CoM fore-aft 3 mm, head tip 0.03 m, neck/head
joint 3 deg; each against the run's own reference = median over [524,288, 1,048,576) steps (standing, tail ~0).
Tail rise start (TRS): first step where the 16-rollout (131,072-step) moving mean of the probe's tail_3 mu reaches
0.1 / 0.2 / 0.3 / 0.5 and stays there for 16 consecutive rollouts.
"""
import json

import numpy as np

import pt_common as C

RUNS = ("s42", "s60", "s80")
LEDGER_COLS = ["tail_3_pitch_act_command_mean", "tail_3_pitch_mean_deg", "root_pitch_mean_deg", "com_fore_aft_mean_m",
               "root_height_mean_m", "head_tip_height_mean_m", "neck_pitch_mean_deg", "head_pitch_mean_deg",
               "neck_pitch_act_command_mean", "head_pitch_act_command_mean", "leg_q_l_r_mean_hip_pitch_deg",
               "leg_q_l_r_mean_knee_deg", "leg_q_l_r_mean_ankle_deg", "root_roll_mean_deg", "tail_tip_chord_mean_deg"]
REF = (524_288, 1_048_576)
# Onsets as the records give them (REVIEW.txt D-1, D-2, D-3; SCORE): first over-bar episode, first tail-flagged
# evaluation, first TAIL ALARM, M1 DRIVE ONSET and TRACE ONSET, first rollout with muHi >= 0.05.
REC_ONSETS = {
    "s42": {"first over-bar ep (tail)": 3_670_016, "first tail-flagged eval": 3_801_088, "first TAIL ALARM": 3_932_160,
            "DRIVE ONSET (M1)": 3_670_016, "TRACE ONSET (M1)": 3_670_016, "first muHi>=0.05 rollout": 3_735_552},
    "s60": {"first over-bar ep (tail)": 8_126_464, "DRIVE ONSET (M1)": 7_340_032, "TRACE ONSET (M1)": 8_388_608,
            "first muHi>=0.05 rollout": 7_962_624},
    "s80": {"first over-bar ep (neck -)": 3_801_088, "first ANY-flagged eval (neck)": 4_849_664,
            "first ANY ALARM (neck)": 5_505_024, "first neck muLo>=0.05 rollout": 4_562_944},
}


def load(run):
    rows = C.read_csv(C.f2_file(run, "eval_episodes.csv"))
    L = C.ledger_per_eval(rows, LEDGER_COLS)
    L["tail_ob"] = C.tail_overbar_counts(L["_by"], L["t"])
    L["neck_ob"] = C.any_overbar_counts(L["_by"], L["t"], "neck_pitch_act")
    P = C.probe_batch_means(C.f2_file(run, "policy_probe.npz"))
    bm = P["batch_mean"]
    pr = {"t": P["t"], "tail_mu": P["rollout_mu_mean"][:, C.TAIL3].astype(float),
          "neck_mu": P["rollout_mu_mean"][:, C.NECK].astype(float), "head_mu": P["rollout_mu_mean"][:, C.HEAD].astype(float),
          "muHi": P["rollout_mu_beyond_high_share"][:, C.TAIL3].astype(float),
          "tail3_deg": bm[:, C.OBS["tail_3_pitch"]] * C.R2D, "tail2_deg": bm[:, C.OBS["tail_2_pitch"]] * C.R2D,
          "neck_deg": bm[:, C.OBS["neck_pitch"]] * C.R2D, "head_deg": bm[:, C.OBS["head_pitch"]] * C.R2D,
          "pitch": C.pitch_from_gx(bm[:, C.OBS["g_x"]]), "hip_deg": bm[:, C.OBS["r_hip_pitch"]] * C.R2D,
          "knee_deg": bm[:, C.OBS["r_knee"]] * C.R2D, "ankle_deg": bm[:, C.OBS["r_ankle"]] * C.R2D,
          "gy_abs_max": float(np.max(np.abs(bm[:, C.OBS["g_y"]])))}
    d = np.load(C.f2_file(run, "policy_probe.npz"), allow_pickle=False)
    lab = list(d["bank_labels"])
    i0 = lab.index("statue_settled")
    pr["bank_tail"] = d["bank_action_now"][:, i0, C.TAIL3].astype(float)
    pr["bank_head"] = d["bank_action_now"][:, i0, C.HEAD].astype(float)
    pr["bank_neck"] = d["bank_action_now"][:, i0, C.NECK].astype(float)
    T = C.train_episode_series(C.f2_file(run, "train_episodes.csv"))
    return L, pr, T


def ref_of(t, y):
    sel = (t >= REF[0]) & (t < REF[1]) & ~np.isnan(y)
    return float(np.median(y[sel]))


def main():
    out = {}
    series = {}
    print("TEST 1 / r10 stance6: tail_3 and posture over training (r10_timeline.py; read only)")
    print("Sources: F2 copies eval_episodes.csv, policy_probe.npz, train_episodes.csv (.copy1) of stance6_r10LTSB_<r>.")
    print("Sign conventions: pitch + nose up (ledger root_pitch_mean_deg); CoM fore-aft + forward (m, shown in mm);")
    print("tail_3 + tip up; neck/head pitch + lowers the head (probe_signs.txt in clamp/).\n")
    data = {r: load(r) for r in RUNS}

    # 0. Validation of the obs_rms recovery against independent records
    print("0. CHECKS OF THE RECOVERED PER-ROLLOUT OBSERVATION MEANS [measured]")
    for r in RUNS:
        L, pr, T = data[r]
        edges = np.arange(0, 12_058_624 + 1, 262_144)
        a = C.bin_series(pr["t"], pr["pitch"], edges)
        b = C.bin_series(T["t"], T["pitch"], edges)
        ok = ~np.isnan(a) & ~np.isnan(b)
        c = np.corrcoef(a[ok][2:], b[ok][2:])[0, 1]
        dlt = np.median(a[ok][2:] - b[ok][2:])
        x = C.bin_series(pr["t"], pr["tail3_deg"], edges)
        y = C.bin_series(L["t"], L["tail_3_pitch_mean_deg"], edges)
        ok2 = ~np.isnan(x) & ~np.isnan(y)
        c2 = np.corrcoef(x[ok2][2:], y[ok2][2:])[0, 1]
        print(f"  {r}: probe-recovered pelvis pitch vs training-episode pitch (262,144-step bins from 524,288, n "
              f"{ok[2:].sum()}): r {c:+.3f}, median offset {dlt:+.3f} deg;")
        print(f"       probe-recovered tail_3 angle vs ledger tail_3_pitch_mean_deg (deterministic): r {c2:+.3f}; "
              f"max |batch mean g_y| {pr['gy_abs_max']:.2e} (mirror symmetry, expected 0)")
    print()

    # 1. Binned table per run
    edges = np.arange(0, 12_058_624 + 1, 262_144)
    mids = edges[1:]
    for r in RUNS:
        L, pr, T = data[r]
        cols = {
            "tail_mu": C.bin_series(pr["t"], pr["tail_mu"], edges),
            "bankT": C.bin_series(pr["t"], pr["bank_tail"], edges),
            "q3deg": C.bin_series(pr["t"], pr["tail3_deg"], edges),
            "L_tcmd": C.bin_series(L["t"], L["tail_3_pitch_act_command_mean"], edges),
            "L_tob": np.array([L["tail_ob"][(L["t"] >= a) & (L["t"] < b)].sum() for a, b in zip(edges[:-1], edges[1:])], dtype=float),
            "P_pitch": C.bin_series(pr["t"], pr["pitch"], edges),
            "T_pitch": C.bin_series(T["t"], T["pitch"], edges),
            "L_pitch": C.bin_series(L["t"], L["root_pitch_mean_deg"], edges),
            "T_com": C.bin_series(T["t"], T["com"] * 1000, edges),
            "L_com": C.bin_series(L["t"], L["com_fore_aft_mean_m"] * 1000, edges),
            "L_head": C.bin_series(L["t"], L["head_tip_height_mean_m"], edges),
            "P_neck": C.bin_series(pr["t"], pr["neck_deg"], edges),
            "P_headj": C.bin_series(pr["t"], pr["head_deg"], edges),
            "L_hgt": C.bin_series(L["t"], L["root_height_mean_m"] * 1000, edges),
        }
        series[r] = cols
        print(f"1. {r}: 262,144-step bins (bin end shown). tail_mu = probe rollout mean tail_3 command (median of 32 "
              f"rollouts); bankT = policy's tail_3 at the fixed statue_settled obs; q3deg = probe-recovered tail_3 angle; "
              f"L_tcmd = ledger tail_3 command; L_tob = ledger tail over-bar episodes in the bin (of 20); P_/T_/L_pitch = "
              f"pelvis pitch from probe / training episodes / ledger (deg); T_/L_com = CoM fore-aft (mm); L_head = ledger "
              f"head tip height (m); P_neck, P_headj = probe neck / head joint (deg); L_hgt = ledger root height (mm)")
        hdr = "   step(M)  " + " ".join(f"{k:>8s}" for k in cols)
        print(hdr)
        for i, m in enumerate(mids):
            if m > 12_000_000:
                break
            print(f"   {m/1e6:7.3f}  " + " ".join(f"{cols[k][i]:8.3f}" for k in cols))
        print()

    # 1b. Finer bins (65,536 steps = 8 rollouts) around each run's rise windows
    print("1b. FINER BINS AROUND THE RISES (65,536-step bins; probe and training-episode medians; ledger values at the "
          "evaluation inside the bin, blank if none)")
    wins = {"s42": [(1_048_576, 4_194_304), (6_029_312, 8_650_752)], "s60": [(3_407_872, 5_505_024), (5_767_168, 8_650_752)],
            "s80": [(2_097_152, 4_456_448), (5_242_880, 7_077_888), (9_175_040, 10_747_904)]}
    e = np.arange(0, 12_058_624 + 1, 65_536)
    for r in RUNS:
        L, pr, T = data[r]
        cols = {"tail_mu": C.bin_series(pr["t"], pr["tail_mu"], e), "q3deg": C.bin_series(pr["t"], pr["tail3_deg"], e),
                "P_pitch": C.bin_series(pr["t"], pr["pitch"], e), "T_pitch": C.bin_series(T["t"], T["pitch"], e),
                "T_com": C.bin_series(T["t"], T["com"] * 1000, e), "P_neck": C.bin_series(pr["t"], pr["neck_deg"], e),
                "P_headj": C.bin_series(pr["t"], pr["head_deg"], e), "bankT": C.bin_series(pr["t"], pr["bank_tail"], e),
                "L_tcmd": C.bin_series(L["t"], L["tail_3_pitch_act_command_mean"], e),
                "L_pitch": C.bin_series(L["t"], L["root_pitch_mean_deg"], e),
                "L_com": C.bin_series(L["t"], L["com_fore_aft_mean_m"] * 1000, e),
                "L_head": C.bin_series(L["t"], L["head_tip_height_mean_m"], e)}
        for lo, hi in wins[r]:
            print(f"  {r} [{lo/1e6:.3f}, {hi/1e6:.3f}]M")
            print("   step(M)  " + " ".join(f"{k:>8s}" for k in cols))
            for i, m in enumerate(e[1:]):
                if lo < m <= hi:
                    print(f"   {m/1e6:7.3f}  " + " ".join(("        " if np.isnan(cols[k][i]) else f"{cols[k][i]:8.3f}") for k in cols))
        print()

    # 2. Onsets and posture-shift timing
    print("2. ONSETS AND POSTURE-SHIFT TIMING PER RUN")
    for r in RUNS:
        L, pr, T = data[r]
        sm_tail = C.smooth(pr["tail_mu"], 16)
        trs = {thr: C.first_sustained(pr["t"], sm_tail, thr, True, 16) for thr in (0.1, 0.2, 0.3, 0.5)}
        mu05 = next((int(t) for t, v in zip(pr["t"], pr["muHi"]) if v >= 0.05), None)
        ob = L["tail_ob"]
        first_ob = next((int(t) for t, k in zip(L["t"], ob) if k >= 1), None)
        first_flag = next((int(t) for t, k in zip(L["t"], ob) if k >= 5), None)
        alarm = next((int(L["t"][i]) for i in range(1, len(ob)) if ob[i] >= 5 and ob[i - 1] >= 5), None)
        print(f"  {r}: record onsets {json.dumps(REC_ONSETS[r])}")
        print(f"      recomputed from the ledger: first tail over-bar episode {C.fmt(first_ob)}, first tail-flagged "
              f"evaluation {C.fmt(first_flag)}, first TAIL ALARM {C.fmt(alarm)}; first probe rollout muHi>=0.05 "
              f"{C.fmt(mu05)}")
        print(f"      tail rise start (TRS, probe tail_mu 16-rollout mean, sustained 16): "
              + ", ".join(f">= {k}: {C.fmt(v)}" for k, v in trs.items()))
        # ledger tail cmd crossing, sustained 2 evaluations
        ltr = {thr: C.first_sustained(L["t"], L["tail_3_pitch_act_command_mean"], thr, True, 2) for thr in (0.1, 0.2, 0.3, 0.5)}
        print(f"      ledger tail_3 command crossing (2 evaluations sustained): "
              + ", ".join(f">= {k}: {C.fmt(v)}" for k, v in ltr.items()))
        # posture references and shifts
        specs = [
            ("probe pitch (deg)", pr["t"], C.smooth(pr["pitch"], 16), -0.3, 16),
            ("train-ep pitch (deg)", T["t"], C.smooth(T["pitch"], 64), -0.3, 64),
            ("ledger pitch (deg)", L["t"], L["root_pitch_mean_deg"], -0.3, 2),
            ("train-ep CoM f-a (mm)", T["t"], C.smooth(T["com"] * 1000, 64), +3.0, 64),
            ("ledger CoM f-a (mm)", L["t"], L["com_fore_aft_mean_m"] * 1000, +3.0, 2),
            ("ledger head tip (m)", L["t"], L["head_tip_height_mean_m"], +0.03, 2),
            ("probe neck joint (deg)", pr["t"], C.smooth(pr["neck_deg"], 16), -3.0, 16),
            ("probe head joint (deg)", pr["t"], C.smooth(pr["head_deg"], 16), -3.0, 16),
        ]
        out[r] = {"trs": trs, "ledger_cross": ltr, "rec": REC_ONSETS[r], "recomputed": {
            "first_ob": first_ob, "first_flag": first_flag, "alarm": alarm, "mu05": mu05}, "shift": {}}
        print("      posture vs own reference [524,288, 1,048,576): ref; first sustained shift in the hypothesised "
              "direction (nose-down / forward / raised / neck-head joint more negative = head up) and the opposite one:")
        for name, t, y, thr, k in specs:
            t, y = np.asarray(t), np.asarray(y)
            ref = ref_of(t, y)
            after = t >= REF[1]  # search only after the reference window
            hyp = C.first_sustained(t[after], y[after], ref + thr, thr > 0, k)
            opp = C.first_sustained(t[after], y[after], ref - thr, thr < 0, k)
            out[r]["shift"][name] = {"ref": ref, "hyp": hyp, "opp": opp}
            print(f"        {name:24s} ref {ref:+8.3f}  shift {thr:+.2f}: hyp {C.fmt(hyp):>11s}   opposite {C.fmt(opp):>11s}")
        print()

    # 3. Cross-run gaps at matched steps, before each run's rise
    print("3. CROSS-RUN GAPS AT MATCHED STEPS (bin medians; s42 - s80 and s60 - s80) [measured]")
    keys = ["T_pitch", "P_pitch", "L_pitch", "T_com", "L_com", "L_head", "P_neck", "P_headj", "tail_mu", "q3deg"]
    for lo, hi in ((524_288, 1_048_576), (1_048_576, 2_097_152), (2_097_152, 3_145_728), (3_145_728, 3_670_016),
                   (3_670_016, 4_194_304), (4_194_304, 6_291_456), (6_291_456, 7_340_032), (7_340_032, 8_388_608),
                   (8_388_608, 12_058_624)):
        sel = (mids > lo) & (mids <= hi)
        cells = []
        for k in keys:
            a = np.nanmedian(series["s42"][k][sel]) - np.nanmedian(series["s80"][k][sel])
            b = np.nanmedian(series["s60"][k][sel]) - np.nanmedian(series["s80"][k][sel])
            cells.append(f"{k} {a:+.2f}/{b:+.2f}")
        print(f"  [{lo/1e6:.2f}, {hi/1e6:.2f}]M: " + "; ".join(cells))
    print("  (each cell: s42-s80 / s60-s80; pitch deg, CoM mm, head m, joints deg, tail_mu command units, q3 deg)\n")

    # 4. Lead-lag on 65,536-step bins (8 rollouts): correlation of first differences, tail_mu vs probe pitch and neck
    print("4. LEAD-LAG, DESCRIPTIVE ONLY: Pearson r of first differences of 65,536-step bin means, posture(t) vs "
          "tail_mu(t+lag); lag > 0 = posture change precedes tail change. Window [0.5M, TRS(0.3)+2M] (s80: [0.5M, 6M]).")
    for r in RUNS:
        L, pr, T = data[r]
        e = np.arange(0, 12_058_624 + 1, 65_536)
        tail = C.bin_series(pr["t"], pr["tail_mu"], e)
        hi_end = (out[r]["trs"][0.3] or 4_000_000) + 2_000_000 if r != "s80" else 6_000_000
        w = (e[1:] > 524_288) & (e[1:] <= hi_end)
        for nm, y in (("probe pitch", C.bin_series(pr["t"], pr["pitch"], e)),
                      ("probe neck", C.bin_series(pr["t"], pr["neck_deg"], e)),
                      ("probe head", C.bin_series(pr["t"], pr["head_deg"], e)),
                      ("train CoM", C.bin_series(T["t"], T["com"] * 1000, e))):
            dt_, dy = np.diff(tail[w]), np.diff(y[w])
            rs = []
            for lag in range(-6, 7):
                if lag >= 0:
                    a, b = dy[:len(dy) - lag], dt_[lag:]
                else:
                    a, b = dy[-lag:], dt_[:len(dt_) + lag]
                ok = ~np.isnan(a) & ~np.isnan(b)
                rs.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 5 else np.nan)
            best = int(np.nanargmax(np.abs(rs))) - 6
            print(f"  {r} {nm:12s} n {w.sum()-1:3d}  r(lag -6..+6) " + " ".join(f"{x:+.2f}" for x in rs)
                  + f"   max |r| at lag {best:+d}")
    print()
    np.savez(f"{C.OUT}/r10_series.npz", **{f"{r}__{k}": v for r in RUNS for k, v in series[r].items()}, mids=mids)
    with open(f"{C.OUT}/r10_timing.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
