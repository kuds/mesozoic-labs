"""Verifier's independent recomputation of the clamp runner's paired statistics from its raw episodes.jsonl files.

Read only: reads com_revisit/notes/clamp/{run,replay,start,long}/*/episodes.jsonl and the lens's
com_revisit/notes/policies/run/episodes.jsonl; writes only to stdout.  Written from scratch (does not import or copy
analyze_clamp.py).  Run: python -B recompute.py > recompute.txt

Convention (the runner's): paired forced - unforced per (policy, seed); per-policy median over seeds; across the 12
policies: median of per-policy medians, min, max, count >0 / <0, two-sided exact sign test (zeros dropped).
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

C = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/clamp")
LENS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/policies/run")
ACT = ['neck_pitch', 'neck_yaw', 'head_pitch', 'r_hip_pitch', 'r_hip_roll', 'r_knee', 'r_ankle', 'l_hip_pitch',
       'l_hip_roll', 'l_knee', 'l_ankle', 'tail_1_pitch', 'tail_1_yaw', 'tail_2_pitch', 'tail_3_pitch']
HINGE = ['neck_pitch', 'neck_yaw', 'head_pitch', 'tail_1_pitch', 'tail_1_yaw', 'tail_2_pitch', 'tail_3_pitch',
         'r_hip_pitch', 'r_hip_roll', 'r_knee', 'r_ankle', 'r_toe_d2_joint', 'r_toe_d3_joint', 'r_toe_d4_joint',
         'l_hip_pitch', 'l_hip_roll', 'l_knee', 'l_ankle', 'l_toe_d2_joint', 'l_toe_d3_joint', 'l_toe_d4_joint']


def load(path):
    return {(r["who"], r["seed"]): r for r in map(json.loads, open(path))}


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def getters():
    g = {}
    for k in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m", "com_lateral_mean_m",
              "root_roll_mean_deg", "com_speed_rms_mps", "creep_fa_mmps"):
        g[k] = (lambda r, k=k, w="5.0": r["metrics"][w].get(k, float("nan")))
    g["abs_com_lateral_mean_m"] = lambda r: abs(r["metrics"]["5.0"]["com_lateral_mean_m"])
    g["abs_root_roll_mean_deg"] = lambda r: abs(r["metrics"]["5.0"]["root_roll_mean_deg"])
    g["return"] = lambda r: r["return"]
    g["window_return_200_end"] = lambda r: r["win"]["200_end"]["reward_sum"]
    for t in ("action_saturation", "tail_home_pose", "energy", "leg_home_pose", "nosedive", "head_clearance",
              "neck_posture", "lateral_lean"):
        g["reward_" + t] = (lambda r, t=t: r["win"]["0_end"]["reward_terms_sum"]["reward_" + t])
    g["cmd_head_pitch"] = lambda r: r["win"]["500_end"]["applied_mean"][2]
    g["joint_head_pitch_deg"] = lambda r: math.degrees(r["win"]["500_end"]["qpos_mean"][2])
    g["joint_neck_pitch_deg"] = lambda r: math.degrees(r["win"]["500_end"]["qpos_mean"][0])
    g["joint_tail3_deg"] = lambda r: math.degrees(r["win"]["500_end"]["qpos_mean"][6])
    g["own_tail3"] = lambda r: r["win"]["500_end"]["own_mean"][14]
    g["force_tail3"] = lambda r: r["win"]["500_end"]["force_mean"][14]
    return g


G = getters()
SCALE = {k: 1000.0 for k in ("com_fore_aft_mean_m", "com_height_mean_m", "com_lateral_mean_m", "abs_com_lateral_mean_m",
                             "com_speed_rms_mps")}


def paired(a, b, key, names):
    """b - a per (who, seed); returns per-policy medians dict and episode signs."""
    per = {}
    ep_pos = ep_neg = 0
    for who in names:
        d = []
        for (w, s), rb in b.items():
            if w != who:
                continue
            ra = a[(w, s)]
            x = (G[key](rb) - G[key](ra)) * SCALE.get(key, 1.0)
            d.append(x)
            ep_pos += x > 0
            ep_neg += x < 0
        per[who] = float(np.median(d))
    return per, ep_pos, ep_neg


def summarize(per):
    v = np.array(list(per.values()))
    pos, neg = int((v > 0).sum()), int((v < 0).sum())
    return float(np.median(v)), float(v.min()), float(v.max()), pos, neg, sign_p(pos, neg)


def level(arm, key, names):
    vals = []
    for who in names:
        vals.append(np.median([G[key](r) * SCALE.get(key, 1.0) for (w, s), r in arm.items() if w == who]))
    return float(np.median(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def line(label, per, ep):
    m, lo, hi, pos, neg, p = summarize(per)
    print(f"  {label:28s} {m:+10.4f} [{lo:+9.4f}, {hi:+9.4f}] policies +{pos}/-{neg} p {p:.4f}  episodes +{ep[0]}/-{ep[1]}")


def lateral_sd(arm, names):
    return {w: float(np.std([r["metrics"]["5.0"]["com_lateral_mean_m"] * 1000 for (x, s), r in arm.items() if x == w], ddof=1))
            for w in names}


def main():
    keys = ["com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m", "com_lateral_mean_m",
            "abs_com_lateral_mean_m", "abs_root_roll_mean_deg", "com_speed_rms_mps", "creep_fa_mmps", "return",
            "window_return_200_end", "reward_action_saturation", "reward_tail_home_pose", "reward_energy",
            "reward_leg_home_pose", "reward_nosedive", "reward_head_clearance", "reward_neck_posture",
            "cmd_head_pitch", "joint_head_pitch_deg", "joint_neck_pitch_deg", "joint_tail3_deg", "own_tail3",
            "force_tail3"]
    arms = {}
    for g, forced in (("s42", "held0"), ("s80", "forced1")):
        u = load(C / f"run/{g}_unforced/episodes.jsonl")
        f = load(C / f"run/{g}_{forced}/episodes.jsonl")
        rp = load(C / f"replay/{g}_{forced}/episodes.jsonl")
        arms[g] = (u, f, rp)
        names = sorted({w for w, s in u})
        print(f"===== {g}: {forced} - unforced  (n unforced {len(u)}, forced {len(f)}, replay {len(rp)})")
        falls = sum(not r["stood"] or r["terminated"] for arm in (u, f, rp) for r in arm.values())
        print(f"  falls/terminations/not-stood across unforced+forced+replay: {falls}")
        for k in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m", "joint_tail3_deg", "own_tail3", "return"):
            lu, su = level(u, k, names)
            lf, sf = level(f, k, names)
            print(f"  LEVEL {k:26s} {lu:+10.3f} -> {lf:+10.3f}   across-policy SD {su:.3f} -> {sf:.3f}")
        # tail_3 joint per-policy medians unforced (range)
        tj = [np.median([G['joint_tail3_deg'](r) for (w, s), r in u.items() if w == who]) for who in names]
        print(f"  unforced per-policy median tail_3 joint range {min(tj):+.3f}..{max(tj):+.3f} deg")
        tjf = [np.median([G['joint_tail3_deg'](r) for (w, s), r in f.items() if w == who]) for who in names]
        print(f"  forced   per-policy median tail_3 joint range {min(tjf):+.3f}..{max(tjf):+.3f} deg")
        pers = {}
        for k in keys:
            per, a, b = paired(u, f, k, names)
            pers[k] = per
            line(k, per, (a, b))
        print("  -- plant (replay - unforced) and policy feedback (closed - replay)")
        for k in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m", "com_lateral_mean_m", "return",
                  "com_speed_rms_mps"):
            per, a, b = paired(u, rp, k, names)
            line(k + " [plant]", per, (a, b))
            per, a, b = paired(rp, f, k, names)
            line(k + " [feedback]", per, (a, b))
        su, sf = lateral_sd(u, names), lateral_sd(f, names)
        d = {w: sf[w] - su[w] for w in names}
        m, lo, hi, pos, neg, p = summarize(d)
        print(f"  per-seed lateral SD: median unforced {np.median(list(su.values())):.3f} -> forced {np.median(list(sf.values())):.3f} mm; change +{pos}/-{neg} p {p:.4f}")
        print("  per-policy (fore-aft d, pitch d, height d, neck joint d, head joint d, head_clearance d):")
        for w in names:
            print(f"    {w}: {pers['com_fore_aft_mean_m'][w]:+7.2f} mm {pers['root_pitch_mean_deg'][w]:+6.3f} deg "
                  f"{pers['com_height_mean_m'][w]:+7.2f} mm  neck {pers['joint_neck_pitch_deg'][w]:+7.2f} head "
                  f"{pers['joint_head_pitch_deg'][w]:+6.2f}  hc {pers['reward_head_clearance'][w]:+8.2f}")
        x = lambda k: np.array([pers[k][w] for w in names])  # noqa: E731
        print(f"  Spearman d neck vs d fore-aft {spearman(x('joint_neck_pitch_deg'), x('com_fore_aft_mean_m')):+.2f}; "
              f"d pitch vs d fore-aft {spearman(x('root_pitch_mean_deg'), x('com_fore_aft_mean_m')):+.2f}; "
              f"d neck vs d height {spearman(x('joint_neck_pitch_deg'), x('com_height_mean_m')):+.2f}")

    # statue
    su = load(C / "replay/statue/episodes.jsonl")
    sf = load(C / "replay/statue_forced1/episodes.jsonl")
    print("===== statue forced1 - statue (20 spawns)")
    for k in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m", "com_lateral_mean_m", "return", "joint_tail3_deg"):
        d = [(G[k](sf[key]) - G[k](su[key])) * SCALE.get(k, 1.0) for key in su]
        print(f"  {k:26s} median {np.median(d):+.4f}  +{sum(v > 0 for v in d)}/-{sum(v < 0 for v in d)}")
    print(f"  statue unforced tail_3 joint median {np.median([G['joint_tail3_deg'](r) for r in su.values()]):+.4f} deg; "
          f"forced {np.median([G['joint_tail3_deg'](r) for r in sf.values()]):+.4f} deg")

    # start checkpoints and the lineage drift
    print("===== start checkpoints")
    u42, f42, _ = arms["s42"]
    names42 = sorted({w for w, s in u42})
    s42u = load(C / "start/s42_unforced/episodes.jsonl")
    s42h = load(C / "start/s42_held0/episodes.jsonl")
    for k in ("com_fore_aft_mean_m", "root_pitch_mean_deg", "com_height_mean_m", "joint_tail3_deg", "own_tail3"):
        a = np.median([G[k](r) * SCALE.get(k, 1.0) for r in s42u.values()])
        b = np.median([G[k](r) * SCALE.get(k, 1.0) for r in s42h.values()])
        dd = [(G[k](s42h[key]) - G[k](s42u[key])) * SCALE.get(k, 1.0) for key in s42u]
        ends_u = {w: np.median([G[k](r) * SCALE.get(k, 1.0) for (x, s), r in u42.items() if x == w]) - a for w in names42}
        ends_h = {w: np.median([G[k](r) * SCALE.get(k, 1.0) for (x, s), r in f42.items() if x == w]) - b for w in names42}
        mu = summarize(ends_u)
        mh = summarize(ends_h)
        print(f"  {k:22s} start {a:+9.3f} -> held {b:+9.3f} (seeds +{sum(v > 0 for v in dd)}/-{sum(v < 0 for v in dd)}); "
              f"ends-start unforced {mu[0]:+.3f} ({mu[3]}/{mu[4]} p {mu[5]:.4f}); both held {mh[0]:+.3f} ({mh[3]}/{mh[4]} p {mh[5]:.4f})")
    # long runs
    print("===== long runs (abs lateral and pitch, by window)")
    for g, forced in (("s42", "held0"), ("s80", "forced1")):
        lu = load(C / f"long/{g}_unforced/episodes.jsonl")
        lf = load(C / f"long/{g}_{forced}/episodes.jsonl")
        names = sorted({w for w, s in lu})
        for w in ("5.0_to10", "10.0", "15.0"):
            for k, f_ in (("abs_lat_mm", lambda r, w=w: abs(r["metrics"][w]["com_lateral_mean_m"]) * 1000),
                          ("pitch_deg", lambda r, w=w: r["metrics"][w]["root_pitch_mean_deg"]),
                          ("speed_mmps", lambda r, w=w: r["metrics"][w]["com_speed_rms_mps"] * 1000)):
                per = {n: float(np.median([f_(lf[key]) - f_(lu[key]) for key in lu if key[0] == n])) for n in names}
                lvu = np.median([np.median([f_(lu[key]) for key in lu if key[0] == n]) for n in names])
                lvf = np.median([np.median([f_(lf[key]) for key in lf if key[0] == n]) for n in names])
                m, lo, hi, pos, neg, p = summarize(per)
                print(f"  {g} @{w:9s} {k:10s} d {m:+.3f} [{lo:+.3f},{hi:+.3f}] +{pos}/-{neg}  level {lvu:.3f} -> {lvf:.3f}")
    # lens equality of unforced metrics (independent of compare.jsonl)
    lens = load(LENS / "episodes.jsonl")
    nmatch = ntot = 0
    for g in ("s42", "s80"):
        u = arms[g][0]
        for key, r in u.items():
            ntot += 1
            nmatch += json.dumps(lens[key]["metrics"]["5.0"], sort_keys=True) == json.dumps(r["metrics"]["5.0"], sort_keys=True)
    print(f"===== lens equality: unforced 5 s metrics identical to the lens's episodes.jsonl in {nmatch}/{ntot}")


if __name__ == "__main__":
    main()
