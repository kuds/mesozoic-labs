"""Tail clamp test: paired analysis (exploratory, report-only, not pre-registered).

PYTHONPATH=<policies/code_copy> python -B analyze_clamp.py > analysis.txt   (also writes analysis.json)

Pairs every forced episode with the unforced episode of the same (policy, seed) and reports forced - unforced:
per policy the median over its seeds (and how many seeds moved each way); across the 12 policies of a condition how
many medians are > 0 and < 0, with a two-sided sign test (report only).  Window: from 5 s (the policies lens's), the
reward over the whole episode, steps 200-end (D-9's window) and steps 500-end.  Rail effect, both groups on one sign
convention: RAIL - NO RAIL, i.e. s42 (unforced - held at 0) and s80 (forced to +1 - unforced).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from environments.shared.tests.stance_track_helpers import StanceTrack, stance_track_series

HERE = Path(__file__).resolve().parent
LENS = HERE.parent / "policies" / "run"
ARMS = {
    "s42": {"unforced": HERE / "run/s42_unforced", "forced": HERE / "run/s42_held0", "what": "tail_3 held at 0",
            "rail_sign": -1.0, "replay": HERE / "replay/s42_held0"},
    "s80": {"unforced": HERE / "run/s80_unforced", "forced": HERE / "run/s80_forced1", "what": "tail_3 forced to +1",
            "rail_sign": +1.0, "replay": HERE / "replay/s80_forced1"},
}
STATUE = {"unforced": HERE / "replay/statue", "forced": HERE / "replay/statue_forced1"}
LONG = {
    "s42": {"unforced": HERE / "long/s42_unforced", "forced": HERE / "long/s42_held0", "rail_sign": -1.0},
    "s80": {"unforced": HERE / "long/s80_unforced", "forced": HERE / "long/s80_forced1", "rail_sign": +1.0},
}
ACT = ["neck_pitch", "neck_yaw", "head_pitch", "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "l_hip_pitch",
       "l_hip_roll", "l_knee", "l_ankle", "tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch"]
HINGE = ["neck_pitch", "neck_yaw", "head_pitch", "tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch",
         "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "r_toe_d2_joint", "r_toe_d3_joint", "r_toe_d4_joint",
         "l_hip_pitch", "l_hip_roll", "l_knee", "l_ankle", "l_toe_d2_joint", "l_toe_d3_joint", "l_toe_d4_joint"]
# (key in metrics["5.0"], scale, unit)
COM = [
    ("com_fore_aft_mean_m", 1e3, "mm"), ("com_fore_aft_drift_m", 1e3, "mm"), ("root_pitch_mean_deg", 1, "deg"),
    ("com_height_mean_m", 1e3, "mm"), ("com_height_drift_m", 1e3, "mm"),
    ("com_lateral_mean_m", 1e3, "mm"), ("abs_com_lateral_mean_m", 1e3, "mm"), ("com_lateral_drift_m", 1e3, "mm"),
    ("root_roll_mean_deg", 1, "deg"), ("abs_root_roll_mean_deg", 1, "deg"), ("root_roll_drift_deg", 1, "deg"),
    ("com_speed_rms_mps", 1e3, "mm/s"), ("com_speed_end_rms_mps", 1e3, "mm/s"), ("com_fore_aft_sd_m", 1e3, "mm"),
    ("com_lateral_sd_m", 1e3, "mm"), ("com_height_sd_m", 1e3, "mm"), ("root_pitch_sd_deg", 1, "deg"),
    ("root_roll_sd_deg", 1, "deg"), ("root_turn_deg", 1, "deg"), ("root_turn_from_spawn_deg", 1, "deg"),
    ("foot_travel_max_m", 1e3, "mm"), ("foot_travel_from_spawn_max_m", 1e3, "mm"),
    ("com_margin_min_m", 1e3, "mm"), ("com_margin_fore_aft_min_m", 1e3, "mm"), ("com_margin_lateral_min_m", 1e3, "mm"),
    ("xcom_margin_min_m", 1e3, "mm"), ("left_load_share_mean", 1, ""), ("lightest_foot_share_min", 1, ""),
    ("creep_mmps", 1, "mm/s"), ("creep_fa_mmps", 1, "mm/s"),
]
TERMS = ["reward_action_saturation", "reward_tail_home_pose", "reward_tail", "reward_posture", "reward_energy",
         "reward_nosedive", "reward_neck_posture", "reward_leg_home_pose", "reward_lateral_lean", "reward_foot_flatness",
         "reward_height", "reward_heading", "reward_stance_width", "reward_bilateral_support",
         "reward_foot_load_balance", "reward_smoothness", "reward_action_jerk", "reward_drift", "reward_alive",
         "reward_stance_stagger", "reward_stance_cop_band", "reward_spin", "reward_speed", "reward_head_clearance"]
DIAG = ["tail_instability", "tail_home_pose_error", "action_saturation", "tilt_angle", "forward_z", "pelvis_height",
        "neck_posture_error", "leg_home_pose_error", "r_foot_contact", "l_foot_contact"]


def sign_p(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)


def load(folder: Path) -> dict:
    rows = {}
    if not (folder / "episodes.jsonl").exists():
        return rows
    for line in open(folder / "episodes.jsonl"):
        r = json.loads(line)
        rows[(r["who"], r["seed"])] = r
    return rows


def creep_from_npz(z, i: int, start: int = 500) -> tuple[float, float]:
    """Feet-centroid least-squares velocity over samples start..end, mm/s: magnitude and fore-aft (heading frame)."""
    t = z[f"t_{i}"].astype(np.float64)
    feet = z[f"feet_{i}"].astype(np.float64)[:, :, :2].mean(axis=1)
    q = z[f"quat_{i}"].astype(np.float64)
    w, x, y, zq = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (x * y + w * zq), 1 - 2 * (y * y + zq * zq))
    s = slice(start, None)
    tw = t[s] - t[s].mean()
    slope = np.linalg.lstsq(np.vstack([tw, np.ones_like(tw)]).T, feet[s], rcond=None)[0][0]
    f = np.array([np.cos(yaw[s]).mean(), np.sin(yaw[s]).mean()])
    f /= np.linalg.norm(f)
    return 1e3 * float(np.linalg.norm(slope)), 1e3 * float(slope @ f)


def value(r: dict, key: str, window: str = "5.0") -> float:
    m = r["metrics"].get(window)
    if not m:
        return float("nan")
    if key.startswith("abs_"):
        return abs(float(m[key[4:]]))
    if key in ("creep_mmps", "creep_fa_mmps"):
        return float(r.get("_creep", {}).get(key, float("nan")))
    return float(m.get(key, float("nan")))


def attach_creep(rows: dict, folder: Path, lens: bool) -> None:
    cache = {}
    for (who, seed), r in rows.items():
        if not r["stood"]:
            continue
        src = (LENS if lens else folder) / "tracks" / f"{who}.npz"
        if src not in cache:
            if not src.exists():
                continue
            cache[src] = np.load(src)
        z = cache[src]
        seeds = [int(s) for s in z["seeds"]]
        i = seeds.index(seed)
        c, cf = creep_from_npz(z, i)
        r["_creep"] = {"creep_mmps": c, "creep_fa_mmps": cf}


def paired(un: dict, fo: dict, fn) -> dict:
    """Per policy: the seeds' paired differences fn(forced) - fn(unforced); summary across policies."""
    per = {}
    for (who, seed), f in fo.items():
        u = un.get((who, seed))
        if u is None:
            continue
        a, b = fn(f), fn(u)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        per.setdefault(who, []).append(a - b)
    pol = {}
    for who, d in sorted(per.items()):
        d = np.array(d)
        pol[who] = {"n": int(d.size), "median": float(np.median(d)), "mean": float(d.mean()),
                    "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}
    meds = np.array([p["median"] for p in pol.values()]) if pol else np.array([])
    allv = np.concatenate([np.array(v) for v in per.values()]) if per else np.array([])
    pos, neg = int((meds > 0).sum()), int((meds < 0).sum())
    return {"policies": pol, "n_policies": len(pol), "median_of_medians": float(np.median(meds)) if meds.size else float("nan"),
            "min_median": float(meds.min()) if meds.size else float("nan"),
            "max_median": float(meds.max()) if meds.size else float("nan"), "pos": pos, "neg": neg,
            "sign_p": sign_p(pos, neg), "episodes": int(allv.size),
            "episodes_pos": int((allv > 0).sum()), "episodes_neg": int((allv < 0).sum()),
            "episode_median": float(np.median(allv)) if allv.size else float("nan")}


def level(rows: dict, fn) -> dict:
    per = {}
    for (who, seed), r in rows.items():
        v = fn(r)
        if np.isfinite(v):
            per.setdefault(who, []).append(v)
    meds = np.array([np.median(v) for v in per.values()])
    return {"median_of_medians": float(np.median(meds)) if meds.size else float("nan"),
            "min": float(meds.min()) if meds.size else float("nan"), "max": float(meds.max()) if meds.size else float("nan")}


def fmt_pair(name: str, s: dict, scale: float, unit: str) -> str:
    return (f"  {name:34s} {s['median_of_medians'] * scale:+10.4f} {unit:5s} [{s['min_median'] * scale:+9.4f}, "
            f"{s['max_median'] * scale:+9.4f}]  policies +{s['pos']}/-{s['neg']} p {s['sign_p']:.4f}   "
            f"episodes +{s['episodes_pos']}/-{s['episodes_neg']}")


def time_course(group: str, arm: dict) -> dict:
    """Mean over the group's episodes of fore-aft CoM (mm), root pitch (deg), tail_3 (deg) per 1 s bin, both arms."""
    out = {}
    for which in ("unforced", "forced"):
        acc = {"fa": [], "pitch": [], "height": [], "tail3": []}
        folder = arm[which]
        for npz in sorted((folder / "tracks").glob("*.npz")):
            z = np.load(npz)
            src = np.load(LENS / "tracks" / npz.name) if which == "unforced" else z
            for i, _ in enumerate(z["seeds"]):
                if f"com_{i}" not in src or len(src[f"t_{i}"]) != 1001:
                    continue
                tr = StanceTrack(t=src[f"t_{i}"].astype(np.float64), com=src[f"com_{i}"].astype(np.float64),
                                 root_quat=src[f"quat_{i}"].astype(np.float64),
                                 foot_pos=src[f"feet_{i}"].astype(np.float64), labels=tuple(str(x) for x in z["labels"]),
                                 floor_z=float(z["floor_z"]), gravity=float(z["gravity"]))
                s = stance_track_series(tr)
                acc["fa"].append(1e3 * s["com_fore_aft"][1:].reshape(10, 100).mean(axis=1))
                acc["pitch"].append(np.degrees(s["root_pitch"][1:]).reshape(10, 100).mean(axis=1))
                acc["height"].append(1e3 * s["com_height"][1:].reshape(10, 100).mean(axis=1))
                acc["tail3"].append(np.degrees(z[f"tailq_{i}"][:, 3].astype(np.float64)).reshape(10, 100).mean(axis=1))
        out[which] = {k: np.mean(v, axis=0).round(4).tolist() if v else None for k, v in acc.items()}
        out[which]["n"] = len(acc["fa"])
    return out


def main() -> None:
    res: dict = {"groups": {}}
    print("TAIL CLAMP TEST: paired forced - unforced, per (policy, seed); window from 5 s unless named.")
    print("Exploratory, report-only, not pre-registered.  'policies +a/-b' counts per-policy medians; p = two-sided sign test.")
    for group, arm in ARMS.items():
        un, fo = load(arm["unforced"]), load(arm["forced"])
        attach_creep(un, arm["unforced"], lens=True)
        attach_creep(fo, arm["forced"], lens=False)
        g: dict = {"what": arm["what"], "n_unforced": len(un), "n_forced": len(fo)}
        print(f"\n==================== {group}: {arm['what']}  (unforced {len(un)} episodes, forced {len(fo)}) ====================")
        # terminations
        for name, rows in (("unforced", un), ("forced", fo)):
            bad = [(w, s, r["steps"], r["reason"], r["end_time_s"]) for (w, s), r in sorted(rows.items()) if not r["stood"]]
            g[f"falls_{name}"] = bad
            print(f"  falls/terminations {name}: {len(bad)} of {len(rows)}" + (f": {bad}" if bad else ""))
        print("\n  LEVELS (median over policies of the per-policy median over seeds): unforced -> forced")
        g["levels"] = {}
        for key, scale, unit in COM[:6] + [COM[8]]:
            lu, lf = level(un, lambda r, k=key: value(r, k)), level(fo, lambda r, k=key: value(r, k))
            g["levels"][key] = {"unforced": lu, "forced": lf}
            print(f"  {key:34s} {lu['median_of_medians'] * scale:+10.3f} -> {lf['median_of_medians'] * scale:+10.3f} {unit}")
        for name, fn, scale, unit in (
            ("return", lambda r: r["return"], 1, ""),
            ("tail_3 joint 5-10 s", lambda r: r["win"]["500_end"]["qpos_mean"][6], 180 / math.pi, "deg"),
            ("tail_3 applied cmd 5-10 s", lambda r: r["win"]["500_end"]["applied_mean"][14], 1, ""),
            ("tail_3 policy's own cmd 5-10 s", lambda r: r["win"]["500_end"]["own_mean"][14], 1, ""),
        ):
            lu, lf = level(un, fn), level(fo, fn)
            g["levels"][name] = {"unforced": lu, "forced": lf}
            print(f"  {name:34s} {lu['median_of_medians'] * scale:+10.3f} -> {lf['median_of_medians'] * scale:+10.3f} {unit}"
                  f"   (forced range {lf['min'] * scale:+.3f}..{lf['max'] * scale:+.3f})")
        print("\n  COM FAMILIES, forced - unforced (median of per-policy medians [min, max])")
        g["com"] = {}
        for key, scale, unit in COM:
            s = paired(un, fo, lambda r, k=key: value(r, k))
            g["com"][key] = s
            print(fmt_pair(key, s, scale, unit))
        print("\n  RETURN AND REWARD TERMS, forced - unforced (per episode sums)")
        g["reward"] = {}
        for name, fn in (("return (0-10 s)", lambda r: r["return"]),
                         ("window return 2-10 s (D-9's)", lambda r: r["win"]["200_end"]["reward_sum"]),
                         ("window return 5-10 s", lambda r: r["win"]["500_end"]["reward_sum"])):
            s = paired(un, fo, fn)
            g["reward"][name] = s
            print(fmt_pair(name, s, 1, ""))
        for term in TERMS:
            s = paired(un, fo, lambda r, t=term: r["win"]["0_end"]["reward_terms_sum"].get(t, float("nan")))
            g["reward"][term] = s
            if max(abs(s["min_median"]), abs(s["max_median"])) > 0.005:
                print(fmt_pair(term + " (0-10 s)", s, 1, ""))
        print("\n  DIAGNOSTICS 5-10 s means, forced - unforced")
        g["diag"] = {}
        for d in DIAG:
            s = paired(un, fo, lambda r, d=d: r["win"]["500_end"]["diag_mean"].get(d, float("nan")))
            g["diag"][d] = s
            print(fmt_pair(d, s, 1, ""))
        print("\n  TAIL (5-10 s means), forced - unforced")
        g["tail"] = {}
        for j, jn in ((3, "tail_1_pitch"), (4, "tail_1_yaw"), (5, "tail_2_pitch"), (6, "tail_3_pitch")):
            s = paired(un, fo, lambda r, j=j: r["win"]["500_end"]["qpos_mean"][j])
            g["tail"][jn + "_deg"] = s
            print(fmt_pair(jn + " joint", s, 180 / math.pi, "deg"))
        for j, nm, sc, u in ((0, "tail tip height", 1e3, "mm"), (1, "tail tip fwd of pelvis", 1e3, "mm")):
            s = paired(un, fo, lambda r, j=j: r["win"]["500_end"]["tip_mean"][j])
            g["tail"][nm] = s
            print(fmt_pair(nm, s, sc, u))
        print("\n  OTHER ACTUATORS: applied command means 5-10 s (forced - unforced); then joint angles (deg)")
        g["commands"] = {}
        for a in range(15):
            s = paired(un, fo, lambda r, a=a: r["win"]["500_end"]["applied_mean"][a])
            g["commands"][ACT[a]] = s
            print(fmt_pair("cmd " + ACT[a], s, 1, ""))
        s = paired(un, fo, lambda r: sum(abs(x) for x in r["win"]["500_end"]["applied_mean"][:14]))
        g["commands"]["sum_abs_other14"] = s
        print(fmt_pair("sum |cmd| of the other 14", s, 1, ""))
        g["joints"] = {}
        for j, jn in enumerate(HINGE):
            s = paired(un, fo, lambda r, j=j: r["win"]["500_end"]["qpos_mean"][j])
            g["joints"][jn] = s
            if "toe" not in jn:
                print(fmt_pair("joint " + jn, s, 180 / math.pi, "deg"))
        print("\n  ACTUATOR FORCE means 5-10 s (N or N m), forced - unforced")
        g["force"] = {}
        for a in range(15):
            s = paired(un, fo, lambda r, a=a: r["win"]["500_end"]["force_mean"][a])
            g["force"][ACT[a]] = s
            print(fmt_pair("force " + ACT[a], s, 1, ""))
        # spawn-dependent lean: per-policy SD over seeds of the lateral CoM offset, forced vs unforced
        sd = {}
        for name, rows in (("unforced", un), ("forced", fo)):
            per = {}
            for (w, sdd), r in rows.items():
                v = value(r, "com_lateral_mean_m")
                if np.isfinite(v):
                    per.setdefault(w, []).append(v)
            sd[name] = {w: float(np.std(v, ddof=1)) for w, v in per.items()}
        diffs = np.array([sd["forced"][w] - sd["unforced"][w] for w in sd["forced"] if w in sd["unforced"]])
        g["lateral_sd_over_seeds"] = {"unforced_median_mm": 1e3 * float(np.median(list(sd["unforced"].values()))),
                                      "forced_median_mm": 1e3 * float(np.median(list(sd["forced"].values()))),
                                      "pos": int((diffs > 0).sum()), "neg": int((diffs < 0).sum()),
                                      "sign_p": sign_p(int((diffs > 0).sum()), int((diffs < 0).sum()))}
        print(f"\n  per-policy SD over seeds of com_lateral: unforced {g['lateral_sd_over_seeds']['unforced_median_mm']:.3f} mm"
              f" -> forced {g['lateral_sd_over_seeds']['forced_median_mm']:.3f} mm (policies +{g['lateral_sd_over_seeds']['pos']}/"
              f"-{g['lateral_sd_over_seeds']['neg']}, p {g['lateral_sd_over_seeds']['sign_p']:.4f})")
        # open-loop decomposition
        rp = load(arm["replay"])
        if rp:
            print(f"\n  OPEN-LOOP REPLAY ({len(rp)} episodes): the unforced episode's own 14 other commands replayed, tail_3 forced")
            g["replay"] = {"n": len(rp), "falls": [(w, s, r["steps"]) for (w, s), r in rp.items() if not r["stood"]]}
            print(f"  falls/terminations replay: {len(g['replay']['falls'])}")
            for key, scale, unit in [c for c in COM if c[0] in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m",
                                                                 "com_lateral_mean_m", "root_roll_mean_deg", "com_speed_rms_mps",
                                                                 "com_fore_aft_drift_m", "com_height_drift_m")]:
                mech = paired(un, rp, lambda r, k=key: value(r, k))
                feed = paired(rp, fo, lambda r, k=key: value(r, k))
                g["replay"][key] = {"replay_minus_unforced": mech, "closed_minus_replay": feed}
                print(fmt_pair(key + " [plant]", mech, scale, unit))
                print(fmt_pair(key + " [policy feedback]", feed, scale, unit))
            for name, fn, sc, u in (("tail_3 joint", lambda r: r["win"]["500_end"]["qpos_mean"][6], 180 / math.pi, "deg"),
                                    ("return (0-10 s)", lambda r: r["return"], 1, "")):
                mech = paired(un, rp, fn)
                g["replay"][name] = {"replay_minus_unforced": mech}
                print(fmt_pair(name + " [plant]", mech, sc, u))
        # per-policy: unforced level against the paired change, and the spread of the levels across policies
        from scipy.stats import spearmanr

        g["per_policy"] = {}
        print("\n  PER POLICY: unforced level vs paired change (Spearman over the 12), and the across-policy SD of the levels")
        for key, scale, unit in [c for c in COM if c[0] in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m")]:
            names = sorted(g["com"][key]["policies"])
            lu = [float(np.median([value(r, key) for (w, sd_), r in un.items() if w == n])) for n in names]
            lf = [float(np.median([value(r, key) for (w, sd_), r in fo.items() if w == n])) for n in names]
            dd = [g["com"][key]["policies"][n]["median"] for n in names]
            rho, p = spearmanr(lu, dd)
            g["per_policy"][key] = {"names": names, "unforced": lu, "forced": lf, "change": dd, "rho": float(rho),
                                    "p": float(p), "sd_unforced": float(np.std(lu, ddof=1)), "sd_forced": float(np.std(lf, ddof=1))}
            print(f"  {key:28s} rho(level, change) {rho:+.2f} (p {p:.3f}); across-policy SD {np.std(lu, ddof=1) * scale:.3f} -> "
                  f"{np.std(lf, ddof=1) * scale:.3f} {unit}")
            print("    " + "  ".join(f"{n[4:]}:{a * scale:+.2f}->{b * scale:+.2f}" for n, a, b in zip(names, lu, lf)))
        g["time_course"] = time_course(group, arm)
        tc = g["time_course"]
        if tc["forced"]["fa"] and tc["unforced"]["fa"]:
            print("\n  TIME COURSE: mean over episodes per 1 s bin (0-1 ... 9-10 s), forced - unforced")
            for k, u in (("fa", "mm"), ("pitch", "deg"), ("height", "mm"), ("tail3", "deg")):
                d = np.array(tc["forced"][k]) - np.array(tc["unforced"][k])
                print(f"  {k:7s} {u:4s} " + " ".join(f"{x:+7.3f}" for x in d))
        res["groups"][group] = g
    # rail effect on one sign convention
    print("\n==================== RAIL - NO RAIL on one convention: s42 (unforced - held 0), s80 (forced +1 - unforced) ====================")
    res["rail"] = {}
    for key, scale, unit in COM[:11] + [("return", 1, "")]:
        line = f"  {key:28s}"
        res["rail"][key] = {}
        for group, arm in ARMS.items():
            if key == "return":
                s = res["groups"][group]["reward"]["return (0-10 s)"]
            else:
                s = res["groups"][group]["com"][key]
            sgn = arm["rail_sign"]
            mm = sgn * s["median_of_medians"] * scale
            pos, neg = (s["pos"], s["neg"]) if sgn > 0 else (s["neg"], s["pos"])
            res["rail"][key][group] = {"median_of_medians": mm, "policies_pos": pos, "policies_neg": neg, "sign_p": s["sign_p"]}
            line += f"   {group} {mm:+9.4f} {unit:4s} (+{pos}/-{neg}, p {s['sign_p']:.4f})"
        print(line)
    # the start checkpoints, clamped as their ends were
    print("\n==================== START CHECKPOINTS clamped the same way: does the s42 lineage's drift survive the clamp? ====================")
    res["start"] = {}
    for group, arm in ARMS.items():
        su_, sf_ = load(HERE / f"start/{group}_unforced"), load(HERE / f"start/{group}_{'held0' if group == 's42' else 'forced1'}")
        if not su_ or not sf_:
            print(f"  {group}: not run")
            continue
        un, fo = load(arm["unforced"]), load(arm["forced"])
        res["start"][group] = {}
        falls = [(w, s_, r["steps"]) for rows in (su_, sf_) for (w, s_), r in rows.items() if not r["stood"]]
        print(f"  {group}: start unforced {len(su_)}, forced {len(sf_)} episodes; falls {falls}")
        for key, scale, unit in [c for c in COM if c[0] in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m",
                                                             "com_lateral_mean_m", "root_roll_mean_deg")]:
            st_u = float(np.median([value(r, key) for r in su_.values()]))
            st_f = float(np.median([value(r, key) for r in sf_.values()]))
            sp = paired(su_, sf_, lambda r, k=key: value(r, k))
            per_u, per_f = {}, {}
            for (w, s_), r in un.items():
                per_u.setdefault(w, []).append(value(r, key))
            for (w, s_), r in fo.items():
                per_f.setdefault(w, []).append(value(r, key))
            eu = np.array([np.median(v) for v in per_u.values()])
            ef = np.array([np.median(v) for v in per_f.values()])
            du, df = eu - st_u, ef - st_f
            res["start"][group][key] = {"start_unforced": st_u, "start_forced": st_f, "start_pair": sp,
                                        "ends_minus_start_unforced_median": float(np.median(du)),
                                        "ends_minus_start_forced_median": float(np.median(df)),
                                        "unforced_above_below": [int((du > 0).sum()), int((du < 0).sum())],
                                        "forced_above_below": [int((df > 0).sum()), int((df < 0).sum())]}
            print(f"  {key:22s} start {st_u * scale:+9.3f} -> clamped {st_f * scale:+9.3f} {unit} (seeds +{sp['episodes_pos']}/-{sp['episodes_neg']});"
                  f" ends - start: unforced {np.median(du) * scale:+7.3f} ({int((du > 0).sum())}/{int((du < 0).sum())}, p {sign_p(int((du > 0).sum()), int((du < 0).sum())):.4f}),"
                  f" both clamped {np.median(df) * scale:+7.3f} ({int((df > 0).sum())}/{int((df < 0).sum())}, p {sign_p(int((df > 0).sum()), int((df < 0).sum())):.4f})")
        for name, fn in (("return", lambda r: r["return"]),
                         ("tail_3 joint deg", lambda r: math.degrees(r["win"]["500_end"]["qpos_mean"][6]))):
            print(f"  {name:22s} start {np.median([fn(r) for r in su_.values()]):+9.3f} -> clamped {np.median([fn(r) for r in sf_.values()]):+9.3f}")
    # statue
    su, sf = load(STATUE["unforced"]), load(STATUE["forced"])
    if su and sf:
        print(f"\n==================== STATUE (neutral action, tail_3 at 0) with tail_3 forced to +1: {len(sf)} paired spawns ====================")
        res["statue"] = {}
        for key, scale, unit in COM[:11]:
            per = [value(sf[k], key) - value(su[k], key) for k in sf if k in su and sf[k]["stood"] and su[k]["stood"]]
            per = np.array(per)
            pos, neg = int((per > 0).sum()), int((per < 0).sum())
            res["statue"][key] = {"median": float(np.median(per)), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg)}
            print(f"  {key:34s} {np.median(per) * scale:+10.4f} {unit:5s} seeds +{pos}/-{neg} p {sign_p(pos, neg):.4f}")
        per = np.array([np.degrees(sf[k]["win"]["500_end"]["qpos_mean"][6] - su[k]["win"]["500_end"]["qpos_mean"][6]) for k in sf if k in su])
        print(f"  tail_3 joint                       {np.median(per):+10.4f} deg")
        per = np.array([sf[k]["return"] - su[k]["return"] for k in sf if k in su])
        print(f"  return                             {np.median(per):+10.4f}")
        falls = [k for k in list(su) + list(sf) if not (su.get(k, {"stood": True})["stood"] and sf.get(k, {"stood": True})["stood"])]
        print(f"  falls: {falls}")
    # long runs
    print("\n==================== LONG RUNS (2,000 steps; 3 policies per group, 20 seeds), forced - unforced ====================")
    res["long"] = {}
    for group, arm in LONG.items():
        un, fo = load(arm["unforced"]), load(arm["forced"])
        if not un or not fo:
            print(f"  {group}: not run")
            continue
        res["long"][group] = {}
        falls = [(w, s, r["steps"], r["reason"]) for rows in (un, fo) for (w, s), r in rows.items() if not r["stood"]]
        print(f"  {group}: unforced {len(un)}, forced {len(fo)} episodes; falls {falls}")
        for key, scale, unit in [c for c in COM if c[0] in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m",
                                                             "com_lateral_mean_m", "abs_com_lateral_mean_m", "root_roll_mean_deg",
                                                             "abs_root_roll_mean_deg", "com_speed_rms_mps", "com_fore_aft_drift_m",
                                                             "com_lateral_drift_m")]:
            for win in ("5.0_to10", "10.0", "15.0"):
                s = paired(un, fo, lambda r, k=key, w=win: value(r, k, w))
                lu = level(un, lambda r, k=key, w=win: value(r, k, w))
                lf = level(fo, lambda r, k=key, w=win: value(r, k, w))
                res["long"][group][f"{key}@{win}"] = {"pair": s, "unforced": lu, "forced": lf}
                print(fmt_pair(f"{key}@{win}", s, scale, unit) + f"   level {lu['median_of_medians'] * scale:+.3f} -> {lf['median_of_medians'] * scale:+.3f}")
        for name, fn in (("return 10-20 s", lambda r: r["win"]["1000_end"]["reward_sum"]),
                         ("return 0-20 s", lambda r: r["return"])):
            s = paired(un, fo, fn)
            res["long"][group][name] = s
            print(fmt_pair(name, s, 1, ""))
    (HERE / "analysis.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
