"""Reduce the four public scripted-walk traces to load, motion, and contact summaries."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BUILD = Path(__file__).resolve().parent.parent


def from_clock(x):
    return np.percentile(x, [1, 50, 99]).round(3).tolist()


def summary(name, trace_dir, gc):
    af = np.load(trace_dir / f"{name}.npz")
    w = np.flatnonzero(af["t"] > 1.99999)

    def get(key):
        return af[key][w]

    t = get("t")
    f = get("F")
    touch = get("touch")
    weight = float(af["weight"])
    x = get("body")
    com = get("com")
    head = get("head")
    planned = get("planned").astype(bool)
    raw = f > 2
    deb = gc.debounce_contact(raw, 10)
    steps = gc.step_contact(raw, 10, 50)
    dtspan = t[-1] - t[0]
    size = af["sole_size"]
    sole = get("sole")
    sm = get("solemat")
    ends = gc.sole_bottom_points(sole, sm, size, "capsule")
    r = get("xmat")
    roll = np.rad2deg(np.arctan2(r[:, 2, 1], r[:, 2, 2]))
    pitch = np.rad2deg(np.arcsin(-np.clip(r[:, 2, 0], -1, 1)))
    yaw = np.unwrap(np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    front = np.sum(f[:, :2], axis=1)
    tot = np.sum(f, axis=1)
    fore = np.divide(front, tot, out=np.ones_like(front) * np.nan, where=tot > 2)
    shares = 100 * f.sum(axis=0) / f.sum()
    skim = np.arange(0, len(t), 5)
    m, _ = gc.margins_over_time(ends[skim, :, :, :2], steps[skim], com[skim, :2], get("cop")[skim])
    down = np.sum(steps, axis=1)
    diagonly = (down == 2) & ((steps[:, 0] & steps[:, 3]) | (steps[:, 1] & steps[:, 2]))
    vxy = x[-1, :2] - x[0, :2]
    v0 = float(af["heading"])
    cs, ss = np.cos(v0), np.sin(v0)
    ft = [gc.touchdowns(steps[:, j]) for j in range(4)]
    yhead = head[:, 1] * cs - head[:, 0] * ss
    dyhead = yhead - gc.moving_average(yhead, 1000)
    dzhead = head[:, 2] - gc.moving_average(head[:, 2], 1000)
    z = np.min(ends[:, :, :, 2], axis=2)
    gapht = []
    stance = []
    swing = []
    air = []
    for j in range(4):
        off = gc.runs(deb[:, j])
        gapht.append(
            round(
                float(np.median([z[a : a + n, j].max() * 1000 for on, a, n in off if not on and 10 <= n < 50])),
                3,
            )
            if any(not on and 10 <= n < 50 for on, a, n in off)
            else None
        )
        strides = gc.runs(steps[:, j])[1:-1]
        stance.append(int(np.median([n for on, a, n in strides if on])))
        swing.append(int(np.median([n for on, a, n in strides if not on])))
        air.append(
            round(
                float(np.median([z[a : a + n, j].max() * 1000 for on, a, n in strides if not on])),
                1,
            )
        )
    headings = {1: "RF+LH", 0: "LF+RH"}
    result = {
        "mean_N": f.mean(axis=0).round(2).tolist(),
        "share_pc": shares.round(2).tolist(),
        "diagonal_LF_RH_pc": round(shares[[0, 3]].sum(), 2),
        "p99_N": np.percentile(f, 99, axis=0).round(1).tolist(),
        "pk_N": f.max(axis=0).round(1).tolist(),
        "p99_BW": (np.percentile(f, 99, axis=0) / weight).round(2).tolist(),
        "pk_BW": (f.max(axis=0) / weight).round(2).tolist(),
        "weight_N": round(weight, 3),
        "touch_truth_absdiff_max_N": round(abs(f - touch).max(), 7),
        "speed": round((vxy @ [cs, ss]) / dtspan, 5),
        "drift_m": round(vxy @ [-ss, cs], 4),
        "heading_change_deg": round(math.degrees(yaw[-1] - yaw[0]), 2),
        "duty_raw_pc": (100 * raw.mean(axis=0)).round(1).tolist(),
        "duty_filtered_pc": (100 * steps.mean(axis=0)).round(1).tolist(),
        "td_env_Hz": (np.array([len(gc.touchdowns(deb[:, j])) for j in range(4)]) / dtspan).round(2).tolist(),
        "td_steps_Hz": (np.array([len(ft[j]) for j in range(4)]) / dtspan).round(2).tolist(),
        "median_stance_ms": stance,
        "median_swing_ms": swing,
        "median_swing_clearance_mm": air,
        "median_microgap_lift_mm": gapht,
        "roll_mean_p1_p99_deg": [
            round(roll.mean(), 3),
            *np.percentile(roll, [1, 99]).round(3).tolist(),
        ],
        "pitch_mean_p1_p99_deg": [
            round(pitch.mean(), 3),
            *np.percentile(pitch, [1, 99]).round(3).tolist(),
        ],
        "head_bob_sway_1_99_mm": [round(np.ptp(np.percentile(v, [1, 99])) * 1000, 1) for v in (dzhead, dyhead)],
        "body_z_p1_p99_mm": np.round(1000 * np.percentile(x[:, 2], [1, 99]), 1).tolist(),
        "COM_outside_finite_support_pc": round(100 * np.mean(m[np.isfinite(m)] < -1e-6), 1),
        "support_undefined_pc": round(100 * np.mean(~np.isfinite(m)), 1),
        "support_margin_mm_p1_p50_p99": from_clock(1000 * m[np.isfinite(m)]),
        "feet_down_0to4_pc": (100 * np.bincount(down, minlength=5) / len(down)).round(2).tolist(),
        "two_diagonal_pc": round(100 * diagonly.mean(), 2),
        "imposed_phase_average_share_pc": {
            headings[p]: (
                100 * f[planned[:, p] & ~planned[:, 1 - p]].sum(axis=0) / f[planned[:, p] & ~planned[:, 1 - p]].sum()
            )
            .round(1)
            .tolist()
            for p in (0, 1)
        },
        "front_share_inst_percentile1_50_99": np.nanpercentile(fore * 100, [1, 50, 99]).round(1).tolist(),
    }
    af.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--traces", type=Path, required=True, help="Directory created by sample.py")
    parser.add_argument("--out", type=Path, required=True, help="Destination for metrics.json")
    args = parser.parse_args()
    sys.path.insert(0, str(args.build.resolve() / "rl/analytics"))
    import gait_core

    data = {name: summary(name, args.traces, gait_core) for name in ("clip", "slow", "J1", "J4")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
