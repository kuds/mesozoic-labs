"""Report-only: per-heading, per-check margins of the M3 gate certificate at the final checkpoint.

Reads the existing M3 outputs (stance_heading_report.json) of the six PC runs; runs no simulation.
For each run and heading: worst value of each gated per-episode check over 40 episodes, the threshold,
and the worst value as a fraction of the threshold's allowance. Also: are the six headings identical?
And the zero-action statue's worst value on the same checks (same seeds), for scale.
"""
import json
import sys

ROOT = "/home/user/work/pc_study/round/panels"
RUNS = ["stance7_r10LTSBpc_s60", "stance7_r10LTSBpc_s42", "stance7_r10LTSBpc_s80",
        "stance7_r10LTSBoff_s60", "stance7_r10LTSBoff_s42", "stance7_r10LTSBoff_s80"]
# check name in thresholds -> (episode field, direction)
CHECKS = {
    "min_all_feet_support": ("all_feet_support", "min"),
    "max_touchdown_rate": ("touchdown_rate", "max"),
    "max_window_displacement_m": ("window_displacement_m", "max"),
    "min_foot_load_share": ("min_foot_load_share", "min"),
    "max_actuator_saturation_fraction": ("max_actuator_saturation_fraction", "max"),
    "max_settle_peak_floor_force_bw": ("settle_peak_floor_force_bw", "max"),
    "min_foot_load_share_windowed": ("min_foot_load_share_windowed", "min"),
    "max_settle_stance_width_change_m": ("settle_stance_width_change_m", "max"),
    "max_sole_tilt_deg": ("max_sole_tilt_deg", "max"),
    "max_sole_corner_lift_m": ("max_sole_corner_lift_m", "max"),
    "min_sole_contacts": ("min_sole_contacts", "min"),
    "max_episode_yaw_change_deg": ("episode_yaw_change_deg", "max"),
    "max_settle_touchdowns": ("settle_touchdowns", "max"),
    "max_sole_cop_fore_aft": ("max_sole_cop_fore_aft", "max"),
}
EXTRA = ["max_foot_slide_m", "max_sole_cop_outer", "window_yaw_change_deg", "max_stance_width_change_m"]


def worst(eps, field, direction):
    vals = [e[field] for e in eps if e.get(field) is not None]
    if field == "episode_yaw_change_deg":
        vals = [abs(v) for v in vals]
    if not vals:
        return None
    return min(vals) if direction == "min" else max(vals)


def used(w, thr, direction):
    """Fraction of the allowance used: 0 = ideal end, 1 = at the threshold."""
    if w is None:
        return None
    if direction == "max":
        return w / thr if thr else None
    return None  # for min checks the ideal end differs per check; report raw headroom instead


def main():
    for run in RUNS:
        path = f"{ROOT}/{run}/final/m3/stance_heading_report.json"
        try:
            d = json.load(open(path))
        except OSError as exc:
            print(run, "not found", exc)
            continue
        thr = d["thresholds"]
        print(f"=== {run} passed={d['passed']}")
        sig = []
        for h in d["headings"]:
            eps = h["episode_evidence"]
            st = h["statue"]["episode_evidence"]
            r = h["result"]
            row = [f"head {h['yaw_deg']:+.0f}", f"clean {r['n_clean']}/{r['n_episodes']}",
                   f"hop_or_fall {r['hop_or_fall_episodes']}", f"reward {r['mean_reward']:.1f}",
                   f"statue {r['statue_mean_reward']:.1f}"]
            print("  " + " | ".join(row))
            s = []
            for chk, (field, direction) in CHECKS.items():
                w = worst(eps, field, direction)
                ws = worst(st, field, direction)
                t = thr[chk]
                fails = r["criterion_failures"].get(chk)
                u = used(w, t, direction)
                head = (t - w) if direction == "max" else (w - t)
                s.append(round(w, 9) if w is not None else None)
                ustr = f"used {u:.2f}" if u is not None else ""
                print(f"    {chk:36s} thr {t:<7g} worst {w:.4g} (headroom {head:.4g}) {ustr:10s} fails {fails}"
                      f" | statue worst {ws:.4g}")
            for f in EXTRA:
                w = worst(eps, f, "max")
                ws = worst(st, f, "max")
                print(f"    [report-only] {f:30s} worst {w:.4g} | statue worst {ws:.4g}")
            sig.append(tuple(s) + (round(r['mean_reward'], 6),))
        print(f"  distinct heading signatures: {len(set(sig))} of {len(sig)}")
        # per-actuator action statistics at heading 0
        h0 = [h for h in d["headings"] if abs(h["yaw_deg"]) < 1e-9][0]
        print("  per-actuator (heading 0): joint ctrl_mean dc dc_deg ac_rms range_deg")
        for a in h0["action"]["per_actuator"]:
            print(f"    {a['joint']:22s} ctrl_mean {a['ctrl_mean']:+.4f} dc {a['dc']:+.4f} dc_deg {a['dc_deg']:+.3f}"
                  f" ac_rms {a['ac_rms']:.4f} range_deg {a['range_deg']:.2f}")
        act = h0["action"]
        print("  action summary (heading 0): " + ", ".join(f"{k} {act[k]:.6g}" for k in act if k != "per_actuator"))
        rc = h0["reward_components"]
        print("  reward components (heading 0, nonzero): " +
              ", ".join(f"{k[7:]} {v:.2f}" for k, v in sorted(rc.items()) if abs(v) > 1e-9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
