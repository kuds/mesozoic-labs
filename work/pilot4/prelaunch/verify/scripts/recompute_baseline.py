"""Independent recompute of the entry baseline from the raw files (numpy + json only; no repository or tool import).

Usage: python -I recompute_baseline.py VAL_FINAL_DIR E2E_OUT_DIR OUT_JSON

Per seed (42, 52, 60):
  labels   pilot 3's episode labels re-derived from the TEXT of P3 L537-545 (written here, not imported) applied to
           the gait metrics of the judge's probe.json on the e2e stand-in 2,000,000 pair (the edge-prepared 8M walker,
           the same weights), then pilot 3's seed outcome (P3 L615-620); compared with baseline_s<seed>.json labels
  (a), (b) from the npz panel traces err_deg = psi - psi_cmd (deg, states 0..T), v3 L346-368
  (c)      from the npz recovery traces, v3 L369-397 (pass at 650 read from the record, since prey x is not traced)
  twin     heading at state 1000 of each nudged (c) episode minus its panel twin's, minus the angle
  v3 rows  the "pilot-4 entry" rows of v3's evidence table (L849-851; resets 3042-3061) recomputed on 20 episodes
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

VAL, E2E, OUT = (Path(a) for a in sys.argv[1:4])
SEEDS = (42, 52, 60)


def pct(vals, p):
    a = np.sort(np.asarray(vals, dtype=float))
    if len(a) == 0:
        return None
    v = float(np.percentile(a, p))
    return math.inf if math.isnan(v) else v


def label_from_text(row):
    def num(k):
        try:
            v = float(row.get(k))
        except (TypeError, ValueError):
            return math.nan
        return v
    # FALL: ends before 1,000 steps
    if not (num("length") >= 1000):
        return "FALL"
    # HOP: >= 0.5 touchdowns/s (all feet) and simultaneous-touchdown fraction >= 0.5
    if num("floor_touchdowns_per_s_all_feet") >= 0.5 and num("floor_simultaneous_fraction") >= 0.5:
        return "HOP"
    # STILL: window displacement speed below 0.10 m/s (undefined fails the test that reads it -> STILL)
    if not (num("speed_displacement_mps") >= 0.10):
        return "STILL"
    lag = num("jointlag_frac_hip_pitch")
    dl, dr = num("floor_duty_L"), num("floor_duty_R")
    walk = (num("floor_alternation_index") >= 0.6 and num("floor_lead_swaps_per_stride") >= 1.0
            and num("floor_flight_fraction") <= 0.2 and 0.35 <= lag <= 0.65
            and num("floor_stride_frequency_hz") <= 5.0 and dl <= 0.80 and dr <= 0.80)
    return "WALK" if walk else "OTHER"


def seed_outcome(c):
    if c.get("WALK", 0) >= 30 and c.get("FALL", 0) <= 4:
        return "WALKS"
    if c.get("HOP", 0) >= 20:
        return "HOPS"
    if c.get("FALL", 0) >= 5:
        return "FALLS"
    if c.get("STILL", 0) >= 20:
        return "STILL"
    return "MIXED"


def straight_metrics(err, T):
    """err: deg, index = state. Returns yaw, D, W0, R, last-mean error on a full episode."""
    e = np.asarray(err[: T + 1], dtype=float)
    yaw = e[T] - e[101]
    ks = range(151, T - 50 + 1)
    m = np.array([e[k - 50: k + 51].mean() for k in ks])
    D = float(np.max(np.abs(m)))
    W0 = float(np.max(np.abs(e[101: T + 1])))
    R = float(m.max() - m.min())
    return float(yaw), D, W0, R, float(m[-1])


res = {}
for s in SEEDS:
    rec = json.load(open(VAL / f"baseline_s{s}.json"))
    z = np.load(VAL / f"baseline_s{s}.npz", allow_pickle=False)
    out = {}
    eps = rec["panel"]["episodes"]
    # ---- labels
    probe = json.load(open(E2E / f"s{s}" / "2000000" / "probe.json"))
    pe = {int(e["reset_seed"]): e for e in probe["episodes"]}
    lab_mine = {}
    for e in eps:
        r = int(e["reset_seed"])
        lab_mine[r] = label_from_text(pe[r])
    counts = {}
    for v in lab_mine.values():
        counts[v] = counts.get(v, 0) + 1
    rec_labels = rec["labels"]["labels"]
    rec_by_reset = {int(e["reset_seed"]): rec_labels[i] for i, e in enumerate(eps)}
    out["labels_mine_counts"] = counts
    out["labels_mine_outcome"] = seed_outcome(counts)
    out["labels_record_counts"] = rec["labels"]["counts"]
    out["labels_record_outcome"] = rec["labels"]["seed_outcome"]
    out["labels_disagree"] = [r for r in lab_mine if lab_mine[r] != rec_by_reset[r]]
    out["panel_episode_label_field_disagree"] = [int(e["reset_seed"]) for e in eps if e.get("label") != rec_by_reset[int(e["reset_seed"])]]
    out["falls"] = {int(e["reset_seed"]): int(e["T"]) for e in eps if int(e["T"]) < 1000}
    yaw_probe_vs_rec = [abs(float(pe[int(e["reset_seed"])]["yaw_change_deg"]) - float(e["yaw_change_deg"]))
                        for e in eps if e["yaw_change_deg"] is not None]
    out["yaw_probe_e2e_vs_record_max_abs_diff"] = max(yaw_probe_vs_rec)
    out["probe_e2e_length_vs_record_T_mismatch"] = [int(e["reset_seed"]) for e in eps
                                                     if int(float(pe[int(e["reset_seed"])]["length"])) != int(e["T"])]
    # ---- (a), (b) on full panel episodes, from traces
    rows = []
    for i, e in enumerate(eps):
        r = int(e["reset_seed"])
        key = f"panel_{i:02d}_reset{r}_err_deg"
        T = int(e["T"])
        if T < 1000:
            continue
        yaw, D, W0, R, last = straight_metrics(z[key], T)
        rows.append({"reset": r, "yaw": yaw, "D": D, "W0": W0, "R": R, "last": last,
                     "rec_yaw": e["yaw_change_deg"], "rec_D": e["D_deg"], "rec_W0": e["W0_deg"], "rec_R": e["R_deg"]})
    out["n_full"] = len(rows)
    out["per_episode_max_abs_diff_vs_record"] = {
        k: max(abs(x[k] - x["rec_" + k]) for x in rows) for k in ("yaw", "D", "W0", "R")}
    yaws = [x["yaw"] for x in rows]
    a_med = float(np.median(yaws))
    a_p95 = pct([abs(y) for y in yaws], 95)
    out["a"] = {"n": len(rows), "median_signed": a_med, "p95_abs": a_p95,
                "verdict": "PASS" if (len(rows) >= 36 and abs(a_med) <= 5.0 and a_p95 <= 15.0) else "FAIL",
                "record": [rec["parts"]["a"]["median_signed_deg"], rec["parts"]["a"]["p95_abs_deg"], rec["parts"]["a"]["verdict"]]}
    Dp = pct([x["D"] for x in rows], 95)
    Wp = pct([x["W0"] for x in rows], 95)
    out["b"] = {"D_p95": Dp, "W0_p95": Wp, "R_median": float(np.median([x["R"] for x in rows])),
                "R_p95": pct([x["R"] for x in rows], 95),
                "verdict": "PASS" if (len(rows) >= 36 and Dp <= 11.5 and Wp <= 18.5) else "FAIL",
                "record": [rec["parts"]["b"]["D_p95_deg"], rec["parts"]["b"]["W0_p95_deg"], rec["parts"]["b"]["verdict"]]}
    # ---- (c) from recovery traces
    reps = rec["recovery"]["episodes"]
    panel_by_reset = {int(e["reset_seed"]): (i, e) for i, e in enumerate(eps)}
    rs, counting, twin_dev, mismatch_r = [], 0, [], []
    left_before, left_notpassed = 0, 0
    for j, ep in enumerate(reps):
        r, ang = int(ep["reset_seed"]), float(ep["angle_deg"])
        key = f"recovery_{j:02d}_reset{r}_{'+' if ang > 0 else '-'}{abs(int(ang))}_err_deg"
        err = np.asarray(z[key], dtype=float)
        xy = np.asarray(z[key.replace("err_deg", "root_xy")], dtype=float)
        T = int(ep["T"])
        if T < 650 or not ep["reached_nudge"]:
            left_before += 1
            continue
        if not (ep["pelvis_minus_prey_x_650_m"] is not None and ep["pelvis_minus_prey_x_650_m"] > 0):
            left_notpassed += 1
            continue
        counting += 1
        if T < 1000:
            rr = math.inf
        else:
            moved = float(np.hypot(*(xy[1000] - xy[650])))
            rr = math.inf if moved < 0.35 else abs(float(err[900:1001].mean()))
        rs.append(rr)
        recr = ep["r_deg"] if ep["r_deg"] is not None else math.inf
        if not (rr == recr or abs(rr - float(recr)) < 1e-9):
            mismatch_r.append([r, ang, rr, recr])
        pi, pe_ = panel_by_reset[r]
        if T == 1000 and int(pe_["T"]) == 1000:
            perr = np.asarray(z[f"panel_{pi:02d}_reset{r}_err_deg"], dtype=float)
            twin_dev.append([r, ang, float(err[1000] - perr[1000] - ang)])
    if counting >= 30:
        med, p95 = pct(rs, 50), pct(rs, 95)
        verdict = "PASS" if (med <= 5.0 and p95 <= 11.5) else "FAIL"
    else:
        med = p95 = None
        verdict = "NOT READ"
    out["c"] = {"n_counting": counting, "left_before": left_before, "left_not_passed": left_notpassed,
                "median_r": med, "p95_r": p95, "within_11_5": sum(1 for x in rs if x <= 11.5),
                "within_5": sum(1 for x in rs if x <= 5.0), "verdict": verdict, "r_mismatch_vs_record": mismatch_r,
                "record": [rec["parts"]["c"]["n_counting"], rec["parts"]["c"]["median_r_deg"], rec["parts"]["c"]["p95_r_deg"],
                           rec["parts"]["c"]["within_11_5"], rec["parts"]["c"]["within_5"], rec["parts"]["c"]["verdict"]]}
    devs = [abs(d[2]) for d in twin_dev]
    out["twin"] = {"episodes": len(twin_dev), "max_abs_dev_deg": max(devs),
                   "over_1e-4": [d for d in twin_dev if abs(d[2]) > 1e-4],
                   "over_1e-5": [d for d in twin_dev if abs(d[2]) > 1e-5],
                   "record": [rec["checks"]["turned_twin"]["episodes"], rec["checks"]["turned_twin"]["max_abs_dev_deg"],
                              len(rec["checks"]["turned_twin"]["bad"])]}
    out["straight_mine"] = "PASS" if all(out[p]["verdict"] == "PASS" for p in ("a", "b", "c")) else "FAIL"
    out["straight_record"] = rec["parts"]["straight"]
    # ---- v3 entry rows on resets 3042-3061 (20 episodes; full-length only for D, R, W0; yaw over every ep > 110 steps)
    sub = [x for x in rows if 3042 <= x["reset"] <= 3061]
    sub_yaw = [float(e["yaw_change_deg"]) for e in eps if 3042 <= int(e["reset_seed"]) <= 3061
               and e["yaw_change_deg"] is not None and int(e["T"]) > 110]
    out["v3_entry_row_20"] = {
        "n_full": len(sub), "yaw_med": float(np.median(sub_yaw)), "yaw_p95_abs": pct([abs(y) for y in sub_yaw], 95),
        "D_p95": pct([x["D"] for x in sub], 95), "D_med": float(np.median([x["D"] for x in sub])),
        "D_max": max(x["D"] for x in sub), "R_med": float(np.median([x["R"] for x in sub])),
        "R_p95": pct([x["R"] for x in sub], 95), "W0_p95": pct([x["W0"] for x in sub], 95)}
    res[str(s)] = out

V3_ROWS = {"42": [-25.3, 31.9, 27.2, 17.0, 35.9, 22.4, 26.6, 31.3],
           "52": [-46.6, 53.9, 44.8, 32.8, 45.6, 39.0, 44.8, 49.4],
           "60": [-29.6, 45.6, 43.1, 31.3, 45.2, 38.3, 45.2, 48.4]}
for s, row in V3_ROWS.items():
    m = res[s]["v3_entry_row_20"]
    mine = [m["yaw_med"], m["yaw_p95_abs"], m["D_p95"], m["D_med"], m["D_max"], m["R_med"], m["R_p95"], m["W0_p95"]]
    res[s]["v3_entry_row_compare"] = {"v3": row, "mine_rounded": [round(x, 1) for x in mine],
                                      "max_abs_diff": max(abs(round(a, 1) - b) for a, b in zip(mine, row))}
json.dump(res, open(OUT, "w"), indent=1, default=str)
for s in res:
    o = res[s]
    print(f"s{s}: labels mine {o['labels_mine_counts']} -> {o['labels_mine_outcome']}; record {o['labels_record_counts']} "
          f"-> {o['labels_record_outcome']}; disagree {o['labels_disagree']}; falls {o['falls']}; "
          f"yaw e2e-probe vs record max diff {o['yaw_probe_e2e_vs_record_max_abs_diff']:.3g}")
    print(f"   per-episode max diff vs record {o['per_episode_max_abs_diff_vs_record']}")
    print(f"   (a) {o['a']}")
    print(f"   (b) {o['b']}")
    print(f"   (c) {o['c']}")
    print(f"   twin {o['twin']['episodes']} max {o['twin']['max_abs_dev_deg']:.3g} over1e-4 {len(o['twin']['over_1e-4'])} "
          f"over1e-5 {o['twin']['over_1e-5']} record {o['twin']['record']}")
    print(f"   STRAIGHT mine {o['straight_mine']} record {o['straight_record']}")
    print(f"   v3 entry row {o['v3_entry_row_compare']}")
