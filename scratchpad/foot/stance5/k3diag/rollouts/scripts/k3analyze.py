"""K3 diagnosis: reduce the k3roll.py npz files to episode and condition tables (REPORT ONLY; numpy only).

    python3 k3analyze.py NPZ_DIR OUT_DIR

Definitions (control step dt = 0.01 s; window = steps 200 .. L-1, the judge's):
  reference posture   per checkpoint, from its noise0 (deterministic) condition: median window pelvis pitch, height,
                      COM fore-aft (ahead of the mean MTP III anchor along the pelvis heading).
  nominal step        |roll| <= 3 deg, |pitch - pitch_ref| <= 3 deg, pelvis_z >= z_ref - 0.02 m.
  fall onset          the first step of the final run of non-nominal steps before the end (falls only);
                      time to fall = (L - onset) dt.
  final unload        the first step of the final run with a foot's floor load < 0.02 body weight (falls only).
  direction           COM travel from onset to the last step in the heading frame: angle 0 = forward, +90 = right,
                      -90 = left, 180 = back; classes forward / fwd-right / right / back-right / back / ... (45 deg
                      sectors centred on the axes).
  pre-failure steps   window steps before onset (falls) or the whole window (full-horizon episodes): the regime the
                      CoP / margin / headroom statistics are read on.
  CoP                 whole-foot CoP in span half-lengths (-1 heel line, +1 toe line), loaded steps only; pad CoP
                      in pad half-lengths.  Toe-edge step: CoP >= 0.9; heel-edge step: CoP <= -0.9.
  excursion           a run of non-nominal steps; recovered if it ends before the episode ends.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

DT = 0.01
W0 = 200
TOE, HEEL = 0.9, -0.9


def load(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    cols = meta["core_cols"]
    core = d["core"].astype(np.float64)
    out = {"meta": meta, "length": d["length"], "term": d["term"], "seed": d["seed"], "reward": d["reward"],
           "col": {k: core[:, :, i] for i, k in enumerate(cols)}}
    if "per_act" in d.files:
        pa = d["per_act"].astype(np.float64)
        out["mu"], out["q"], out["ff"] = pa[:, :, 0], pa[:, :, 1], pa[:, :, 2]
    return out


def runs_of(mask):
    """[(start, end_exclusive)] of True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def sector(angle):
    names = ["forward", "fwd-right", "right", "back-right", "back", "back-left", "left", "fwd-left"]
    return names[int(((angle + 22.5) % 360) // 45)]


def episode_rows(D, ref, key):
    c = D["col"]
    rows = []
    for e in range(len(D["length"])):
        L = int(D["length"][e])
        fell = L < 1000
        g = {k: v[e, :L] for k, v in c.items()}
        nominal = (np.abs(g["roll_deg"]) <= 3) & (np.abs(g["pitch_deg"] - ref["pitch"]) <= 3) & (g["pelvis_z"] >= ref["z"] - 0.02)
        r = {**key, "seed": int(D["seed"][e]), "length": L, "fell": int(fell), "term": str(D["term"][e]) or "horizon",
             "reward": round(float(D["reward"][e]), 2)}
        onset = L
        if fell:
            bad = np.flatnonzero(nominal)
            onset = int(bad[-1] + 1) if bad.size else 0
            r["onset"] = onset
            r["time_to_fall_s"] = round((L - onset) * DT, 2)
            unl = (g["r_load"] < 0.02) | (g["l_load"] < 0.02)
            ok = np.flatnonzero(~unl)
            u0 = int(ok[-1] + 1) if ok.size else 0
            r["unload_to_fall_s"] = round((L - u0) * DT, 2) if u0 < L else 0.0
            r["unloaded_foot"] = ("r" if g["r_load"][u0:].mean() < g["l_load"][u0:].mean() else "l") if u0 < L else ""
            a = max(onset - 1, 0)
            dfa = g["com_fa"][L - 1] - g["com_fa"][a]
            dlat = g["com_lat"][L - 1] - g["com_lat"][a]
            ang = float(np.degrees(np.arctan2(-dlat, dfa)))  # + right (lateral axis is + left)
            r.update({"dir_angle": round(ang, 1), "direction": sector(ang), "dcom_fa_m": round(dfa, 3), "dcom_lat_m": round(dlat, 3),
                      "end_roll_deg": round(float(g["roll_deg"][L - 1]), 1),
                      "end_pitch_dev_deg": round(float(g["pitch_deg"][L - 1] - ref["pitch"]), 1),
                      "com_fa_at_onset_m": round(float(g["com_fa"][a]), 3),
                      "com_fa_minus_ref_at_onset_m": round(float(g["com_fa"][a] - ref["com_fa"]), 3)})
            lo = max(L - 100, 0)
            for cat in ("tail", "head", "tibia"):
                k = f"c_{cat}"
                if k in g:
                    hit = np.flatnonzero(g[k][lo:] > 1.0)
                    r[f"contact_{cat}_steps_before_end"] = int(100 - hit[0]) if hit.size else ""
            for cat in ("torso", "arms", "thigh", "metatarsus"):
                k = f"c_{cat}"
                if k in g:
                    hit = np.flatnonzero(g[k][lo:] > 1.0)
                    r[f"contact_{cat}_steps_before_end"] = int(100 - hit[0]) if hit.size else ""
            # slip while loaded, in the 100 steps before onset and between onset and the end
            for tag, sl in (("pre", slice(max(onset - 100, 0), onset)), ("fall", slice(onset, L))):
                sm = []
                for s in ("r", "l"):
                    ld = g[f"{s}_load"][sl] > 0.1
                    sp = g[f"{s}_slip"][sl]
                    sm.append(float(sp[ld].max()) if ld.any() else 0.0)
                r[f"slip_max_loaded_{tag}_mps"] = round(max(sm), 3)
            # CoP at the edges in the 100 steps before onset (loaded foot-steps)
            pre = slice(max(onset - 100, W0 if onset > W0 + 20 else 0), onset)
            for s in ("r", "l"):
                w = g[f"{s}_wcop"][pre]
                w = w[np.isfinite(w)]
                r[f"{s}_wcop_toe_share_pre"] = round(float((w >= TOE).mean()), 3) if w.size else ""
                r[f"{s}_wcop_heel_share_pre"] = round(float((w <= HEEL).mean()), 3) if w.size else ""
                p = g[f"{s}_pcop_fa"][pre]
                p = p[np.isfinite(p)]
                r[f"{s}_pcop_toe_share_pre"] = round(float((p >= TOE).mean()), 3) if p.size else ""
            r["max_sole_tilt_at_onset_deg"] = round(float(np.nanmax([g["r_tilt"][a], g["l_tilt"][a]])), 1) if "r_tilt" in g else ""
            if "r_pcop_lat" in g:
                pl = np.concatenate([g["r_pcop_lat"][pre], g["l_pcop_lat"][pre]])
                pl = pl[np.isfinite(pl)]
                r["pcop_lat_edge_share_pre"] = round(float((np.abs(pl) >= 0.9).mean()), 3) if pl.size else ""
            if "i_foot_stagger_m" in g:
                sg = g["i_foot_stagger_m"]
                r["stagger_at_onset_m"] = round(float(sg[a]), 3)
                r["stagger_absmax_2s_pre_m"] = round(float(np.nanmax(np.abs(sg[max(a - 200, 0):a + 1]))), 3)
            r["ankle_ff_absmax_last100"] = round(float(np.nanmax(np.abs(np.concatenate([g["r_ankle_ff"][lo:], g["l_ankle_ff"][lo:]])))), 3)
        # pre-failure window statistics (per episode)
        hi = onset if fell else L
        sl = slice(W0, max(hi, W0))
        if hi > W0 + 10:
            for s in ("r", "l"):
                w = g[f"{s}_wcop"][sl]
                w = w[np.isfinite(w)]
                r[f"{s}_wcop_med"] = round(float(np.median(w)), 3) if w.size else ""
                r[f"{s}_wcop_toe_share"] = round(float((w >= TOE).mean()), 3) if w.size else ""
                r[f"{s}_wcop_heel_share"] = round(float((w <= HEEL).mean()), 3) if w.size else ""
            r["com_fa_mean_m"] = round(float(np.nanmean(g["com_fa"][sl])), 4)
            r["pitch_mean_deg"] = round(float(np.nanmean(g["pitch_deg"][sl])), 2)
            r["t3_cmd_ge09_share"] = round(float((np.clip(g["t3_mu"][sl], -1, 1) >= 0.9).mean()), 3)
            r["t3_mu_mean"] = round(float(np.nanmean(g["t3_mu"][sl])), 3)
        rows.append(r)
    return rows


def condition_row(D, ref, key, ep_rows):
    c = D["col"]
    E = len(D["length"])
    falls = [r for r in ep_rows if r["fell"]]
    out = {**key, "n": E, "falls": len(falls), "fall_seeds": " ".join(str(r["seed"]) for r in falls)}
    terms = {}
    for r in falls:
        terms[r["term"]] = terms.get(r["term"], 0) + 1
    out["terms"] = " ".join(f"{k}:{v}" for k, v in sorted(terms.items()))
    dirs = {}
    for r in falls:
        dirs[r["direction"]] = dirs.get(r["direction"], 0) + 1
    out["directions"] = " ".join(f"{k}:{v}" for k, v in sorted(dirs.items(), key=lambda kv: -kv[1]))
    if falls:
        out["time_to_fall_med_s"] = float(np.median([r["time_to_fall_s"] for r in falls]))
        out["time_to_fall_min_s"] = float(np.min([r["time_to_fall_s"] for r in falls]))
        out["time_to_fall_max_s"] = float(np.max([r["time_to_fall_s"] for r in falls]))
        out["unload_to_fall_med_s"] = float(np.median([r["unload_to_fall_s"] for r in falls]))
        out["fall_length_med"] = float(np.median([r["length"] for r in falls]))
        out["dcom_fa_med_m"] = float(np.median([r["dcom_fa_m"] for r in falls]))
        out["com_fa_minus_ref_at_onset_med_m"] = float(np.median([r["com_fa_minus_ref_at_onset_m"] for r in falls]))
        out["slip_pre_med_mps"] = float(np.median([r["slip_max_loaded_pre_mps"] for r in falls]))
        out["slip_fall_med_mps"] = float(np.median([r["slip_max_loaded_fall_mps"] for r in falls]))
        for cat in ("tail", "head", "tibia", "torso", "arms", "thigh", "metatarsus"):
            k = f"contact_{cat}_steps_before_end"
            out[f"falls_with_{cat}_contact_last100"] = sum(1 for r in falls if r.get(k, "") != "")
    # pooled pre-failure steps
    pool = {k: [] for k in ("r_wcop", "l_wcop", "r_pcop_fa", "l_pcop_fa", "com_fa", "xcom_fa", "toe_line_fa", "heel_line_fa",
                            "pitch_deg", "roll_deg", "r_ankle_ff", "l_ankle_ff", "t3_mu", "split", "both_toe")}
    exc_n, exc_rec, exc_dur, exc_peak_rec = 0, 0, [], []
    steps_pre = 0
    for e in range(E):
        L = int(D["length"][e])
        er = ep_rows[e]
        hi = er.get("onset", L) if er["fell"] else L
        if hi <= W0:
            continue
        sl = slice(W0, hi)
        steps_pre += hi - W0
        for k in pool:
            if k in ("split", "both_toe"):
                continue
            if k in c:
                pool[k].append(c[k][e, sl])
        rw, lw = c["r_wcop"][e, sl], c["l_wcop"][e, sl]
        both = np.isfinite(rw) & np.isfinite(lw)
        pool["split"].append((((rw >= TOE) & (lw <= HEEL)) | ((lw >= TOE) & (rw <= HEEL)))[both])
        pool["both_toe"].append(((rw >= TOE) & (lw >= TOE))[both])
        # posture excursions over the whole window [200, L)
        g_roll, g_pitch, g_z = c["roll_deg"][e, W0:L], c["pitch_deg"][e, W0:L], c["pelvis_z"][e, W0:L]
        nom = (np.abs(g_roll) <= 3) & (np.abs(g_pitch - ref["pitch"]) <= 3) & (g_z >= ref["z"] - 0.02)
        for a, b in runs_of(~nom):
            exc_n += 1
            if b < L - W0:
                exc_rec += 1
                exc_dur.append((b - a) * DT)
                exc_peak_rec.append(float(np.max(np.abs(g_roll[a:b]))))
    # closed-loop persistence: slope of x[t+k] on x[t] (episode-demeaned, pre-failure steps), k = 10 and 50 steps
    for name in ("roll_deg", "pitch_deg", "com_lat", "com_fa"):
        for lag in (10, 50):
            num = den = 0.0
            for e in range(E):
                L = int(D["length"][e])
                hi = ep_rows[e].get("onset", L) if ep_rows[e]["fell"] else L
                x = c[name][e, W0:hi]
                x = x[np.isfinite(x)]
                if x.size <= lag + 10:
                    continue
                x = x - x.mean()
                num += float(x[:-lag] @ x[lag:])
                den += float(x[:-lag] @ x[:-lag])
            out[f"ar{lag}_{name}"] = num / den if den > 0 else float("nan")
    # the same on full-horizon episodes only (no selection on the falling ones), window 200..999
    for name in ("roll_deg", "com_lat", "pitch_deg", "com_fa"):
        num = den = 0.0
        for e in range(E):
            if ep_rows[e]["fell"]:
                continue
            x = c[name][e, W0:1000]
            x = x[np.isfinite(x)] - np.nanmean(x)
            num += float(x[:-50] @ x[50:])
            den += float(x[:-50] @ x[:-50])
        out[f"ar50fh_{name}"] = num / den if den > 0 else float("nan")
    for name in ("roll_deg", "com_lat"):
        v = np.concatenate([c[name][e, W0:1000] for e in range(E) if not ep_rows[e]["fell"]] or [np.array([np.nan])])
        v = v[np.isfinite(v)]
        out[f"fh_sd_{name}"] = float(v.std()) if v.size else float("nan")
    out["n_full_horizon"] = sum(1 for r in ep_rows if not r["fell"])
    P = {k: (np.concatenate(v) if v else np.array([])) for k, v in pool.items()}
    out["pre_failure_steps"] = steps_pre
    for s in ("r", "l"):
        pass
    w = np.concatenate([P["r_wcop"], P["l_wcop"]])
    w = w[np.isfinite(w)]
    p = np.concatenate([P["r_pcop_fa"], P["l_pcop_fa"]])
    p = p[np.isfinite(p)]
    if w.size:
        out.update({"wcop_med": float(np.median(w)), "wcop_p5": float(np.percentile(w, 5)), "wcop_p95": float(np.percentile(w, 95)),
                    "wcop_toe_share": float((w >= TOE).mean()), "wcop_heel_share": float((w <= HEEL).mean()),
                    "wcop_out_band_share": float(((w > 0.4) | (w < -0.6)).mean()),
                    "toe_margin_p5": float(1 - np.percentile(w, 95)), "heel_margin_p5": float(np.percentile(w, 5) + 1)})
    if p.size:
        out.update({"pcop_med": float(np.median(p)), "pcop_toe_share": float((p >= TOE).mean()), "pcop_heel_share": float((p <= HEEL).mean())})
    if P["split"].size:
        out["split_share"] = float(P["split"].mean())
        out["both_toe_share"] = float(P["both_toe"].mean())
    af = np.abs(np.concatenate([P["r_ankle_ff"], P["l_ankle_ff"]]))
    af = af[np.isfinite(af)]
    if af.size:
        out.update({"ankle_ff_p50": float(np.median(af)), "ankle_ff_p99": float(np.percentile(af, 99)), "ankle_ff_max": float(af.max()),
                    "ankle_headroom_min": float(1 - af.max())})
    for k in ("com_fa", "pitch_deg"):
        v = P[k][np.isfinite(P[k])]
        if v.size:
            out[f"{k}_mean"] = float(v.mean())
            out[f"{k}_sd"] = float(v.std())
    v = P["roll_deg"][np.isfinite(P["roll_deg"])]
    if v.size:
        out["roll_sd"] = float(v.std())
    lat = np.concatenate([c["com_lat"][e, W0:(ep_rows[e].get("onset", int(D["length"][e])) if ep_rows[e]["fell"] else int(D["length"][e]))]
                          for e in range(E)])
    lat = lat[np.isfinite(lat)]
    if lat.size:
        out["com_lat_sd"] = float(lat.std())
    if P["xcom_fa"].size:
        m_toe = P["toe_line_fa"] - P["xcom_fa"]
        m_heel = P["xcom_fa"] - P["heel_line_fa"]
        m_toe, m_heel = m_toe[np.isfinite(m_toe)], m_heel[np.isfinite(m_heel)]
        out["xcom_toe_margin_p1_m"] = float(np.percentile(m_toe, 1))
        out["xcom_toe_margin_med_m"] = float(np.median(m_toe))
        out["xcom_heel_margin_p1_m"] = float(np.percentile(m_heel, 1))
    t3 = np.clip(P["t3_mu"][np.isfinite(P["t3_mu"])], -1, 1)
    if t3.size:
        out["t3_cmd_ge09_share"] = float((t3 >= 0.9).mean())
        out["t3_cmd_mean"] = float(t3.mean())
    # foot-unload events over the whole window [200, L): runs of >= 10 steps with a foot's load < 0.02 BW
    un_n = un_rec = 0
    un_dur_rec, un_steps, all_steps = [], 0, 0
    for e in range(E):
        L = int(D["length"][e])
        for s in ("r", "l"):
            ld = c[f"{s}_load"][e, W0:L]
            unl = ld < 0.02
            un_steps += int(unl.sum())
            all_steps += unl.size
            for a, b in runs_of(unl):
                if b - a < 10:
                    continue
                un_n += 1
                if b < L - W0:
                    un_rec += 1
                    un_dur_rec.append((b - a) * DT)
    out["unload_step_share"] = un_steps / all_steps if all_steps else float("nan")
    out["unload_events_ge10"] = un_n
    out["unload_events_replanted"] = un_rec
    out["unload_replanted_dur_med_s"] = float(np.median(un_dur_rec)) if un_dur_rec else float("nan")
    out["unload_replanted_dur_max_s"] = float(np.max(un_dur_rec)) if un_dur_rec else float("nan")
    # baselines on the pre-failure steps: pad lateral CoP at an edge, sole tilt, loaded-foot slip
    pl, tl, sl_ = [], [], []
    for e in range(E):
        L = int(D["length"][e])
        hi = ep_rows[e].get("onset", L) if ep_rows[e]["fell"] else L
        if hi <= W0:
            continue
        for s in ("r", "l"):
            if f"{s}_pcop_lat" in c:
                pl.append(c[f"{s}_pcop_lat"][e, W0:hi])
                tl.append(c[f"{s}_tilt"][e, W0:hi])
            ld = c[f"{s}_load"][e, W0:hi] > 0.1
            sl_.append(c[f"{s}_slip"][e, W0:hi][ld])
    if pl:
        v = np.concatenate(pl)
        v = v[np.isfinite(v)]
        out["pcop_lat_edge_share"] = float((np.abs(v) >= 0.9).mean()) if v.size else float("nan")
        out["pcop_lat_med"] = float(np.median(v)) if v.size else float("nan")
        v = np.concatenate(tl)
        v = v[np.isfinite(v)]
        out["sole_tilt_p95_deg"] = float(np.percentile(v, 95)) if v.size else float("nan")
        out["sole_tilt_med_deg"] = float(np.median(v)) if v.size else float("nan")
    if sl_:
        v = np.concatenate(sl_)
        v = v[np.isfinite(v)]
        out["slip_loaded_p99_mps"] = float(np.percentile(v, 99)) if v.size else float("nan")
        out["slip_loaded_med_mps"] = float(np.median(v)) if v.size else float("nan")
    out["excursions"] = exc_n
    out["excursions_recovered"] = exc_rec
    out["recovered_share"] = float(exc_rec / exc_n) if exc_n else float("nan")
    out["recovered_dur_med_s"] = float(np.median(exc_dur)) if exc_dur else float("nan")
    out["recovered_peak_roll_p90_deg"] = float(np.percentile(exc_peak_rec, 90)) if exc_peak_rec else float("nan")
    out["recovered_peak_roll_max_deg"] = float(np.max(exc_peak_rec)) if exc_peak_rec else float("nan")
    meta = D["meta"]
    own = np.asarray(meta["own_sigma"])
    out["own_sigma_mean"] = float(own.mean())
    out["own_sigma_legs_mean"] = float(own[3:11].mean())
    out["own_sigma_ankles"] = float(own[[6, 10]].mean())
    out["own_sigma_tail3"] = float(own[14])
    # foot stagger and hip-pitch differential (full-set conditions): spread and 0.5 s persistence, pre-failure steps
    if "i_foot_stagger_m" in c and "q" in D:
        stg, hpd, num, den = [], [], 0.0, 0.0
        for e in range(E):
            L = int(D["length"][e])
            hi = ep_rows[e].get("onset", L) if ep_rows[e]["fell"] else L
            if hi <= W0 + 60:
                continue
            x = c["i_foot_stagger_m"][e, W0:hi]
            stg.append(x)
            hpd.append(D["q"][e, W0:hi, 3] - D["q"][e, W0:hi, 7])
            xx = x - x.mean()
            num += float(xx[:-50] @ xx[50:])
            den += float(xx[:-50] @ xx[:-50])
        if stg:
            v = np.concatenate(stg)
            out["stagger_sd_m"] = float(np.nanstd(v))
            out["stagger_abs_p95_m"] = float(np.nanpercentile(np.abs(v), 95))
            out["ar50_stagger"] = num / den if den > 0 else float("nan")
            out["hip_pitch_diff_sd_rad"] = float(np.nanstd(np.concatenate(hpd)))
    if "q" in D:  # leg posture over the pre-failure window (episode-pooled means)
        qs = []
        for e in range(E):
            L = int(D["length"][e])
            hi = ep_rows[e].get("onset", L) if ep_rows[e]["fell"] else L
            if hi > W0:
                qs.append(D["q"][e, W0:hi])
        if qs:
            q = np.concatenate(qs)
            for name, idx in (("hip_pitch", [3, 7]), ("hip_roll", [4, 8]), ("knee", [5, 9]), ("ankle", [6, 10]), ("tail3", [14])):
                out[f"q_{name}_mean_rad"] = float(np.nanmean(q[:, idx]))
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.npz")))
    groups = {}
    for f in files:
        run, steps, cond = os.path.basename(f)[:-4].split("__")
        groups.setdefault((run, steps), {})[cond] = f
    ep_all, cond_all = [], []
    for (run, steps), conds in sorted(groups.items()):
        if "noise0" not in conds:
            print("no noise0 reference for", run, steps, "- skipped")
            continue
        D0 = load(conds["noise0"])
        c0 = D0["col"]
        w = slice(W0, 1000)
        ref = {"pitch": float(np.nanmedian(c0["pitch_deg"][:, w])), "z": float(np.nanmedian(c0["pelvis_z"][:, w])),
               "com_fa": float(np.nanmedian(c0["com_fa"][:, w]))}
        for cond, f in sorted(conds.items()):
            D = D0 if cond == "noise0" else load(f)
            key = {"run": run, "steps": steps, "cond": cond}
            er = episode_rows(D, ref, key)
            cr = condition_row(D, ref, key, er)
            cr.update({"ref_pitch_deg": ref["pitch"], "ref_z_m": ref["z"], "ref_com_fa_m": ref["com_fa"]})
            v = D["meta"].get("verified")
            if v:
                cr["verified_vs_judge"] = bool(v["lengths_terms_equal"] and v["reward_max_abs_diff"] < 1e-6 and v["max_sole_cop_equal"]
                                               and v["max_sole_tilt_equal"])
                cr["judge_falls"] = v["judge_falls"]
            ep_all += er
            cond_all.append(cr)
    for name, rows in (("episodes.csv", ep_all), ("conditions.csv", cond_all)):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(dst, name), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(ep_all)} episode rows, {len(cond_all)} condition rows to {dst}")


if __name__ == "__main__":
    main()
