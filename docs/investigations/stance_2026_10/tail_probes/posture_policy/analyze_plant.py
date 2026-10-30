"""Test 3 summary: run_plant.py static and replay outputs.  Read only; plain numpy.

python -B analyze_plant.py > analysis_plant.txt

Groups: statue (20 spawns), start_s42, start_s80 (20 seeds each), s42 ends and s80 ends (12 policies x 20 seeds for
the static part, 12 x 3 seeds for the replays).  Per policy the median over seeds, then the median [range] over
policies; s42 minus statue and s42 minus s80 per policy-median, with counts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def grp(who: str) -> str:
    return who if who in ("statue", "start_s42", "start_s80") else who[:3]


def fmt(vals) -> str:
    v = np.asarray(vals, dtype=float)
    if len(v) == 1:
        return f"{v[0]:+.4f}"
    return f"{np.median(v):+.4f} [{v.min():+.4f},{v.max():+.4f}]"


def static_section() -> None:
    res = json.loads((HERE / "plant_static.json").read_text())
    print("=" * 110)
    print("STATIC TAIL BALANCE IN THE HELD POSTURES (run_plant.py static -> plant_static.json)")
    print("   a grid", res["a_grid"], "; tail_home target (rad)", [round(x, 5) for x in res["tail_home_target_rad"]])
    hk = res["home_keyframe"]
    print(f"   home keyframe (noise-free, home controls): pelvis pitch {hk['pelvis_pitch_deg']:+.3f} deg; q3_eq(a) "
          + " ".join(f"{x:+.3f}" for x in hk["q3_eq_deg_vs_a"])
          + f"; slope {hk['plant_slope_deg_per_unit']:.4f} deg/unit, intercept {hk['plant_intercept_deg']:+.4f} deg;"
          f" a to hold target {hk['a_to_hold_target']:+.4f}; gravity on tail_3 {hk['tail_3_pitch']['gravity_Nm']:+.3f} N m")
    per = defaultdict(lambda: defaultdict(list))
    for who, rows in res["postures"].items():
        for r in rows:
            vals = {
                "pelvis_pitch_deg": r["pelvis_pitch_deg"],
                "tail3_world_pitch_deg": r["tail3_segment_world_pitch_deg"],
                "tail_tip_z_mm": 1000 * r["tail_tip_z_m"],
                "plant_slope_deg_per_unit": r["plant_slope_deg_per_unit"],
                "plant_intercept_deg": r["plant_intercept_deg"],
                "q3_eq_at_a-1": r["q3_eq_deg_vs_a"][0],
                "q3_eq_at_a0": r["q3_eq_deg_vs_a"][4],
                "q3_eq_at_a+1": r["q3_eq_deg_vs_a"][8],
                "a_to_hold_target": r["a_to_hold_target"],
                "grav_tail3_at_target_Nm": r["gravity_Nm_at_target_tail3"],
            }
            for n in ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch"):
                t = r[n]
                vals[f"{n}:q_deg"] = t["q_deg"]
                vals[f"{n}:cmd"] = t["cmd"]
                vals[f"{n}:gravity_Nm"] = t["gravity_Nm"]
                vals[f"{n}:spring_Nm"] = t["spring_Nm"]
                vals[f"{n}:servo_Nm"] = t["servo_Nm"]
                vals[f"{n}:net_static_Nm"] = t["net_static_Nm"]
            for k, v in vals.items():
                per[who][k].append(v)
    keys = list(next(iter(per.values())).keys())
    med = {who: {k: float(np.median(v)) for k, v in d.items()} for who, d in per.items()}
    print("   level per group: per-posture-seed median for single checkpoints; median [range] of per-policy medians")
    for k in keys:
        cells = []
        for g in ("statue", "start_s42", "start_s80", "s42", "s80"):
            whos = [w for w in med if grp(w) == g]
            cells.append(f"{g} {fmt([med[w][k] for w in whos])}")
        print(f"   {k:28s} " + " | ".join(cells))
    print("   differences of per-policy medians (12 s42 ends against the statue's median, and against the s80 ends' median):")
    st = med["statue"]
    s80med = {k: float(np.median([med[w][k] for w in med if grp(w) == "s80"])) for k in keys}
    for k in keys:
        d1 = np.array([med[w][k] - st[k] for w in med if grp(w) == "s42"])
        d2 = np.array([med[w][k] - s80med[k] for w in med if grp(w) == "s42"])
        print(f"   {k:28s} s42-statue {np.median(d1):+.4f} (+{int((d1 > 0).sum())}/-{int((d1 < 0).sum())})"
              f"   s42-s80 {np.median(d2):+.4f} (+{int((d2 > 0).sum())}/-{int((d2 < 0).sum())})")
    # association across all held postures: does a more nose-down pelvis go with a different plant map?
    xs, ys, zs, gs = [], [], [], []
    for who, d in per.items():
        if who == "statue":
            continue
        xs += d["pelvis_pitch_deg"]
        ys += d["q3_eq_at_a0"]
        zs += d["a_to_hold_target"]
        gs += d["tail_3_pitch:gravity_Nm"]
    xs, ys, zs, gs = map(np.asarray, (xs, ys, zs, gs))
    for name, y in (("q3_eq_at_a0 (deg)", ys), ("a_to_hold_target", zs), ("gravity on tail_3 (N m)", gs)):
        b = np.polyfit(xs, y, 1)[0]
        print(f"   across all {len(xs)} policy postures: slope of {name} on pelvis pitch {b:+.5f} per deg"
              f" (pelvis pitch range {xs.min():+.2f}..{xs.max():+.2f} deg) [correlational]")


def replay_section() -> None:
    p = HERE / "plant_replay.jsonl"
    if not p.exists():
        return
    rows = [json.loads(line) for line in open(p)]
    base = {}
    for line in open(HERE / "run/base/episodes.jsonl"):
        r = json.loads(line)
        base[(r["who"], r["seed"])] = r["return"]
    own = [r for r in rows if r["a"] == "own"]
    same = sum(1 for r in own if base.get((r["who"], r["seed"])) == r["return"])
    print("=" * 110)
    print("OPEN-LOOP REPLAYS (run_plant.py replay -> plant_replay.jsonl)")
    print(f"   check: replay without replacement reproduces the baseline return in {same} of {len(own)} episodes;"
          f" stood {sum(r['steps'] == 1000 for r in rows)} of {len(rows)} replays")
    terms = ["reward_tail", "reward_tail_home_pose", "reward_action_saturation", "reward_energy", "reward_smoothness",
             "reward_action_jerk", "reward_nosedive", "reward_posture"]
    A = [-1.0, -0.5, 0.0, 0.5, 1.0]
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["a"] == "own":
            continue
        a = float(r["a"])
        s = r["per_step_500_1000"]
        vals = {"tail3_deg": r["tail3_deg"], "tail_instability": s["tail_instability"],
                "tail_tip_gyro_norm": r["tail_tip_gyro_norm"], "pitch_deg": r["pitch_deg"],
                "reward_per_step": r["reward_per_step_500_1000"]}
        vals.update({k: s.get(k, float("nan")) for k in terms})
        for k, v in vals.items():
            per[r["who"]][a][k].append(v)
    whos = list(per)
    keys = list(per[whos[0]][0.0].keys())
    print("   per-step window means by command a (per policy: mean over its seeds; group: median [range] over policies)")
    for k in keys:
        print(f"   {k}:")
        for g in ("statue", "start_s42", "start_s80", "s42", "s80"):
            ws = [w for w in whos if grp(w) == g]
            cells = []
            for a in A:
                cells.append(fmt([np.mean(per[w][a][k]) for w in ws]))
            print(f"     {g:10s} " + " | ".join(f"a={a:+.1f}: {c}" for a, c in zip(A, cells)))
    print("   tail-up cost per step, a=+1 minus a=0 (and a=+0.5 minus 0), by term; s42 against statue and s80:")
    for k in ["tail3_deg", "pitch_deg", "reward_per_step"] + terms + ["tail_instability"]:
        cells = []
        for g in ("statue", "start_s42", "start_s80", "s42", "s80"):
            ws = [w for w in whos if grp(w) == g]
            d1 = [np.mean(per[w][1.0][k]) - np.mean(per[w][0.0][k]) for w in ws]
            d5 = [np.mean(per[w][0.5][k]) - np.mean(per[w][0.0][k]) for w in ws]
            cells.append(f"{g} {fmt(d1)} / {fmt(d5)}")
        print(f"   {k:26s} " + " | ".join(cells))
    print("   plant map in the moving posture: least-squares slope of tail3_deg on a, and the intercept")
    for g in ("statue", "start_s42", "start_s80", "s42", "s80"):
        ws = [w for w in whos if grp(w) == g]
        sl, ic = [], []
        for w in ws:
            y = [np.mean(per[w][a]["tail3_deg"]) for a in A]
            b, c = np.polyfit(A, y, 1)
            sl.append(b)
            ic.append(c)
        print(f"     {g:10s} slope {fmt(sl)} deg/unit; intercept {fmt(ic)} deg")


if __name__ == "__main__":
    static_section()
    replay_section()
