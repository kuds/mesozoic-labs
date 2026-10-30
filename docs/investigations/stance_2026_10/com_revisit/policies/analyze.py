"""Read run_policies.py's output: settle, the four families, lean signs across seeds, creep, groups, checks.

PYTHONPATH=<code_copy> python -B analyze.py RUN_DIR WINDOW_S OUT_PREFIX [--heldout RUN_DIR2]
Pure numpy/scipy on the saved episodes.jsonl (metrics reduced in the run, support keys included) and
tracks/*.npz (re-reduced here for the per-second settle profile and the creep numbers).  No simulation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from environments.shared.tests.stance_track_helpers import (
    FAMILIES,
    StanceTrack,
    lean_sign_summary,
    sign_test_p,
    stance_track_series,
)

GROUPS = {
    "statue": lambda w: w == "statue",
    "starts": lambda w: w.startswith("start_"),
    "s42 ends (rail)": lambda w: w.startswith("s42_"),
    "s80 ends (no rail)": lambda w: w.startswith("s80_"),
}
SIGN_KEYS = (
    "com_lateral_mean_m",
    "root_roll_mean_deg",
    "left_load_excess",
    "com_lateral_drift_m",
    "root_roll_drift_deg",
    "root_turn_deg",
    "root_turn_from_spawn_deg",
    "root_minus_feet_yaw_from_spawn_deg",
)
SHOW = {
    "lean": FAMILIES["lean"] + ("com_margin_lateral_min_m", "xcom_margin_lateral_min_m"),
    "pivot": FAMILIES["pivot"],
    "height": FAMILIES["height"],
    "sway": FAMILIES["sway"],
    "sagittal": ("com_fore_aft_mean_m", "com_fore_aft_drift_m", "root_pitch_mean_deg"),
    "support": ("com_margin_min_m", "xcom_margin_min_m", "support_defined_fraction"),
}
SCALE = {"_m": (1000.0, "mm"), "_mps": (1000.0, "mm/s"), "_deg": (1.0, "deg")}


def unit(key: str) -> tuple[float, str]:
    for suffix, su in SCALE.items():
        if key.endswith(suffix):
            return su
    return 1.0, ""


def load_rows(run: Path, window: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in open(run / "episodes.jsonl"):
        r = json.loads(line)
        m = dict(r["metrics"].get(window, {}))
        if m:
            m["left_load_excess"] = m["left_load_share_mean"] - 0.5
        m["_stood"] = r["stood"]
        m["_cmd_mean"] = r["cmd_mean_from_2s"]
        m["_cmd_sat"] = r["cmd_sat_share_from_2s"]
        out[r["who"]][r["seed"]] = m
    return out


def load_tracks(run: Path, who: str) -> list[tuple[int, StanceTrack, np.ndarray]]:
    z = np.load(run / "tracks" / f"{who}.npz")
    seeds = [int(s) for s in z["seeds"]]
    labels = tuple(str(x) for x in z["labels"])
    out = []
    for i, seed in enumerate(seeds):
        tr = StanceTrack(
            t=z[f"t_{i}"], com=z[f"com_{i}"].astype(np.float64), root_quat=z[f"quat_{i}"].astype(np.float64),
            foot_pos=z[f"feet_{i}"].astype(np.float64), labels=labels, foot_load=z[f"load_{i}"].astype(np.float64),
            foot_heading=z[f"heading_{i}"].astype(np.float64), floor_z=float(z["floor_z"]), gravity=float(z["gravity"]),
        )
        out.append((seed, tr, z[f"action_{i}"]))
    return out


def per_second(series: dict, name: str) -> list[float]:
    t = series["t"]
    x = series[name]
    out = []
    for k in range(int(round(t[-1] - t[0]))):
        m = (t >= t[0] + k - 1e-9) & (t < t[0] + k + 1 - 1e-9)
        out.append(float(np.sqrt(np.mean(x[m] ** 2))))
    return out


def settle_second(rms: list[float], bar: float = 1e-3) -> int:
    """The first whole second from which every second's RMS stays under *bar* to the end (len(rms) if never)."""
    for k in range(len(rms)):
        if all(v < bar for v in rms[k:]):
            return k
    return len(rms)


def creep_numbers(series: dict, window_s: float) -> dict[str, float]:
    t = series["t"]
    w = t >= t[0] + window_s - 1e-9
    feet = series["foot_xy"][w]
    feet_mean = feet.mean(axis=1)
    tw = t[w]
    # least-squares slope of the feet's mean horizontal position: a steady creep, in mm/s
    A = np.vstack([tw - tw.mean(), np.ones_like(tw)]).T
    slope = np.linalg.lstsq(A, feet_mean, rcond=None)[0][0]
    com_xy_dt = np.gradient(series["xcom"][w], tw, axis=0)  # not used for the bars; kept implicit
    del com_xy_dt
    last2 = tw >= tw[-1] - 2.0 - 1e-9
    slope_end = np.linalg.lstsq(np.vstack([tw[last2] - tw[last2].mean(), np.ones(last2.sum())]).T, feet_mean[last2], rcond=None)[0][0]
    return {
        "feet_mean_travel_window_mm": float(np.linalg.norm(feet_mean[-1] - feet_mean[0]) * 1000),
        "feet_creep_speed_mmps": float(np.linalg.norm(slope) * 1000),
        "feet_creep_speed_end_mmps": float(np.linalg.norm(slope_end) * 1000),
    }


def fmt(key: str, v: float) -> str:
    s, _ = unit(key)
    return f"{v * s:+.3f}" if abs(v * s) < 100 else f"{v * s:+.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("window", type=float)
    ap.add_argument("out", type=Path)
    ap.add_argument("--heldout", type=Path)
    args = ap.parse_args()
    wkey = str(float(args.window))
    rows = load_rows(args.run, wkey)
    whos = [w for w in rows if (args.run / "tracks" / f"{w}.npz").exists()]
    L: list[str] = []
    J: dict = {"window_s": args.window, "who": {}}
    L.append(f"window from {args.window} s to the end of 1,000 steps (10 s); run {args.run}")
    L.append(f"policies and controls: {len(whos)}; episodes stood: "
             + ", ".join(f"{w} {sum(1 for m in rows[w].values() if m['_stood'])}/{len(rows[w])}" for w in whos if sum(1 for m in rows[w].values() if m['_stood']) != len(rows[w])) or "all")

    # ── settle and creep from the tracks ─────────────────────────────────────
    settle: dict[str, list[int]] = {}
    settle_lat: dict[str, list[int]] = {}
    speed_profile: dict[str, list[list[float]]] = {}
    creep: dict[str, dict[int, dict]] = defaultdict(dict)
    for who in whos:
        settle[who], settle_lat[who], speed_profile[who] = [], [], []
        for seed, tr, _acts in load_tracks(args.run, who):
            if len(tr.t) < 1001:
                continue
            s = stance_track_series(tr)
            rms = per_second(s, "com_speed")
            dlat = np.gradient(s["com_lateral"], s["t"])
            s["com_lateral_speed"] = dlat
            rms_lat = per_second(s, "com_lateral_speed")
            settle[who].append(settle_second(rms))
            settle_lat[who].append(settle_second(rms_lat))
            speed_profile[who].append(rms)
            creep[who][seed] = creep_numbers(s, args.window)
    L.append("")
    L.append("SETTLE (the PR's rule: the first whole second after which the horizontal CoM speed RMS stays under 1 mm/s"
             " in every second to step 1,000; 10 = not settled within the episode)")
    L.append(f"  {'who':14s} settle s: median [min max] n<=5s  | lateral-only settle median [max] | per-second RMS mm/s, median over seeds (s 0-9)")
    for who in whos:
        st, sl = np.array(settle[who]), np.array(settle_lat[who])
        prof = np.median(np.array(speed_profile[who]), axis=0) * 1000
        L.append(f"  {who:14s} {np.median(st):4.1f} [{st.min():2d} {st.max():2d}] {int((st <= 5).sum()):2d}/{st.size}"
                 f"   | {np.median(sl):4.1f} [{sl.max():2d}] | " + " ".join(f"{x:5.2f}" for x in prof))
        J["who"].setdefault(who, {})["settle"] = st.tolist()
        J["who"][who]["settle_lateral"] = sl.tolist()
        J["who"][who]["speed_profile_median_mmps"] = prof.round(4).tolist()
    for g, pred in GROUPS.items():
        if not any(pred(w) for w in whos):
            continue
        st = np.concatenate([settle[w] for w in whos if pred(w)])
        L.append(f"  group {g:20s}: settle median {np.median(st):.1f} s, 90th pct {np.percentile(st, 90):.1f}, max {st.max()}, "
                 f"share settled by 2/3/4/5 s: " + "/".join(f"{np.mean(st <= k):.2f}" for k in (2, 3, 4, 5)))

    # ── families ─────────────────────────────────────────────────────────────
    L.append("")
    L.append("FAMILIES at the window: median over the stood seeds [min, max]; units mm, mm/s, deg, load share")
    for fam, keys in SHOW.items():
        L.append(f"  -- {fam}")
        for key in keys:
            parts = []
            for g, pred in GROUPS.items():
                vals = np.array([m[key] for w in whos if pred(w) for m in rows[w].values() if key in m])
                if vals.size:
                    parts.append(f"{g.split(' ')[0]} {fmt(key, np.median(vals))} [{fmt(key, vals.min())}, {fmt(key, vals.max())}]")
            L.append(f"    {key:36s} " + " | ".join(parts))
    for who in whos:
        J["who"][who]["family_medians"] = {
            key: float(np.median([m[key] for m in rows[who].values() if key in m]))
            for keys in SHOW.values() for key in keys if any(key in m for m in rows[who].values())
        }

    # ── lean signs across seeds ──────────────────────────────────────────────
    L.append("")
    L.append("LEAN SIGN ACROSS THE SEEDS (lean_sign_summary; signs: com_lateral + = CoM left of the feet's centre; root_roll"
             " + = left side up; left_load_excess = left_load_share_mean - 0.5, + = left foot heavier; turn/yaw + = to the"
             " left (counter-clockwise from above); p = two-sided sign test; m = mean over seeds)")
    head = f"  {'who':14s} " + " ".join(f"{k[:22]:>27s}" for k in SIGN_KEYS[:4])
    L.append(head)
    for who in whos:
        per = [m for m in rows[who].values() if m.get("_stood") and "com_lateral_mean_m" in m]
        summ = lean_sign_summary(per, keys=SIGN_KEYS)
        J["who"][who]["sign"] = summ
        cells = []
        for k in SIGN_KEYS[:4]:
            d = summ[k]
            s, _ = unit(k)
            cells.append(f"{int(d['positive']):2d}+{int(d['negative']):2d}- p{d['sign_test_p']:.3f} m{d['mean'] * s:+.3f}")
        L.append(f"  {who:14s} " + " ".join(f"{c:>27s}" for c in cells))
    L.append(f"  {'who':14s} " + " ".join(f"{k[:22]:>27s}" for k in SIGN_KEYS[4:]))
    for who in whos:
        summ = J["who"][who]["sign"]
        cells = []
        for k in SIGN_KEYS[4:]:
            d = summ[k]
            s, _ = unit(k)
            cells.append(f"{int(d['positive']):2d}+{int(d['negative']):2d}- p{d['sign_test_p']:.3f} m{d['mean'] * s:+.3f}")
        L.append(f"  {who:14s} " + " ".join(f"{c:>27s}" for c in cells))

    # paired against the statue at the same spawns
    L.append("")
    L.append("PAIRED AGAINST THE STATUE AT THE SAME SPAWN (policy minus statue, per seed): sign counts and sign-test p")
    statue = rows["statue"]
    for who in whos:
        if who == "statue":
            continue
        cells = []
        J["who"][who]["paired"] = {}
        for k in ("com_lateral_mean_m", "root_roll_mean_deg", "left_load_excess"):
            d = np.array([rows[who][s][k] - statue[s][k] for s in rows[who] if k in rows[who][s] and k in statue.get(s, {})])
            pos, neg = int((d > 0).sum()), int((d < 0).sum())
            sc, _ = unit(k)
            cells.append(f"{k[:18]} {pos:2d}+{neg:2d}- p{sign_test_p(pos, neg):.3f} med{np.median(d) * sc:+.3f}")
            J["who"][who]["paired"][k] = {"pos": pos, "neg": neg, "p": sign_test_p(pos, neg), "median": float(np.median(d))}
        L.append(f"  {who:14s} " + " | ".join(cells))

    # ── groups ───────────────────────────────────────────────────────────────
    L.append("")
    L.append("GROUPS (one number per policy: its median over the seeds; exact two-sided Mann-Whitney U on those numbers)")
    ends = [w for w in whos if w[:3] in ("s42", "s80")]
    comp_keys = ("com_lateral_mean_m", "root_roll_mean_deg", "left_load_excess", "lightest_foot_share_min",
                 "com_margin_lateral_min_m", "root_turn_from_spawn_deg", "foot_travel_from_spawn_max_m",
                 "foot_travel_max_m", "com_height_mean_m", "com_speed_rms_mps", "com_speed_end_rms_mps",
                 "root_pitch_sd_deg", "com_fore_aft_mean_m", "root_pitch_mean_deg", "com_fore_aft_sd_m", "root_roll_sd_deg",
                 "root_yaw_sd_deg")
    med = {w: {k: float(np.median([m[k] for m in rows[w].values() if k in m])) for k in comp_keys} for w in whos}
    J["groups"] = {}
    for label, a, b in (
        ("s42 vs s80 ends", [w for w in ends if w.startswith("s42")], [w for w in ends if w.startswith("s80")]),
        ("s42 A vs B", [w for w in ends if w.startswith("s42_A")], [w for w in ends if w.startswith("s42_B")]),
        ("s80 A vs B", [w for w in ends if w.startswith("s80_A")], [w for w in ends if w.startswith("s80_B")]),
        ("A vs B (both)", [w for w in ends if w[4] == "A"], [w for w in ends if w[4] == "B"]),
    ):
        if not a or not b:
            continue
        L.append(f"  -- {label} ({len(a)} vs {len(b)})")
        J["groups"][label] = {}
        for k in comp_keys:
            xa, xb = np.array([med[w][k] for w in a]), np.array([med[w][k] for w in b])
            p = float(mannwhitneyu(xa, xb, alternative="two-sided", method="exact").pvalue)
            sc, u = unit(k)
            J["groups"][label][k] = {"a_median": float(np.median(xa)), "b_median": float(np.median(xb)), "p": p}
            L.append(f"    {k:32s} {np.median(xa) * sc:+9.3f} vs {np.median(xb) * sc:+9.3f} {u:5s} p {p:.4f}{'  *' if p < 0.01 else ''}")
    L.append("  start policies' medians beside their 12 ends' range:")
    for st, pre in (("start_s42", "s42"), ("start_s80", "s80")):
        for k in ("com_lateral_mean_m", "root_roll_mean_deg", "left_load_excess", "com_fore_aft_mean_m", "root_pitch_mean_deg", "com_speed_rms_mps"):
            e = np.array([med[w][k] for w in ends if w.startswith(pre)])
            if not e.size or st not in med:
                continue
            sc, u = unit(k)
            L.append(f"    {st} {k:26s} {med[st][k] * sc:+8.3f} {u:5s}; its ends {np.min(e) * sc:+8.3f} .. {np.max(e) * sc:+8.3f}"
                     f" (median {np.median(e) * sc:+8.3f}; ends with the start's sign {int(np.sum(np.sign(e) == np.sign(med[st][k])))}/12)")
    J["policy_medians"] = med

    # ── creep ────────────────────────────────────────────────────────────────
    L.append("")
    L.append("CREEP over the window: the feet's mean horizontal position, its travel and least-squares speed (whole window"
             " and last 2 s), and the reduction's foot_travel_max_m and com_speed_end_rms_mps")
    for g, pred in GROUPS.items():
        ws = [w for w in whos if pred(w)]
        if not ws:
            continue
        c = [creep[w][s] for w in ws for s in creep[w]]
        trav = np.array([x["feet_mean_travel_window_mm"] for x in c])
        sp = np.array([x["feet_creep_speed_mmps"] for x in c])
        spe = np.array([x["feet_creep_speed_end_mmps"] for x in c])
        ft = np.array([rows[w][s]["foot_travel_max_m"] for w in ws for s in rows[w] if "foot_travel_max_m" in rows[w][s]]) * 1000
        ce = np.array([rows[w][s]["com_speed_end_rms_mps"] for w in ws for s in rows[w] if "com_speed_end_rms_mps" in rows[w][s]]) * 1000
        L.append(f"  {g:20s} n {trav.size:3d}: feet-mean travel mm median {np.median(trav):.3f} max {trav.max():.3f}; creep speed mm/s"
                 f" median {np.median(sp):.4f} max {sp.max():.4f} (end {np.median(spe):.4f} / {spe.max():.4f}); foot_travel_max mm median"
                 f" {np.median(ft):.3f} max {ft.max():.3f}; CoM speed end RMS mm/s median {np.median(ce):.3f} max {ce.max():.3f}")
    worst = sorted(((creep[w][s]["feet_creep_speed_mmps"], w, s) for w in whos for s in creep[w]), reverse=True)[:8]
    L.append("  largest feet creep speeds: " + "; ".join(f"{w} {s} {v:.3f} mm/s" for v, w, s in worst))
    J["creep"] = {w: creep[w] for w in whos}

    # ── the commands behind the groups (the rail) ────────────────────────────
    L.append("")
    L.append("COMMANDS from 2 s (mean over seeds of the window mean; tail_3 = actuator 14; hip rolls 4 r, 8 l)")
    for who in whos:
        cm = np.array([m["_cmd_mean"] for m in rows[who].values() if m["_cmd_mean"]])
        L.append(f"  {who:14s} tail_3 {cm[:, 14].mean():+.3f}  r_hip_roll {cm[:, 4].mean():+.4f}  l_hip_roll {cm[:, 8].mean():+.4f}"
                 f"  (r+l {cm[:, 4].mean() + cm[:, 8].mean():+.4f})")

    args.out.with_suffix(".txt").write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
