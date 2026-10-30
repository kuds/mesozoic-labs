"""Compare fixcheck.jsonl against the base rows in scan.jsonl, same seed (no simulation)."""
import json
import numpy as np
base = {}
for line in open("scan.jsonl"):
    r = json.loads(line)
    if r["model"] == "velo" and r["noise"] is not None:
        base[(r["noise"], r["seed"])] = r
rows = [json.loads(l) for l in open("fixcheck.jsonl")]
STANCE = ("com_fore_aft_mean_m", "com_lateral_mean_m", "com_height_mean_m", "root_pitch_mean_deg", "root_roll_mean_deg",
          "left_load_share_mean", "com_margin_min_m", "com_margin_lateral_min_m", "root_turn_from_spawn_deg",
          "foot_travel_from_spawn_max_m", "foot_travel_max_m", "com_speed_end_rms_mps")
for variant in ("claw_arm_x3", "ts_1ms"):
    for noise, label in ((0.05, "the 46 travel-flagged spawns at 0.05"), (0.02, "calibration seeds 0-29 at 0.02")):
        rs = [r for r in rows if r["variant"] == variant and r["noise"] == noise]
        b = [base[(noise, r["seed"])] for r in rs]
        print(f"== {variant} on {label} ({len(rs)}):")
        print(f"   stood {sum(r['stood'] for r in rs)}/{len(rs)}; flags speed {sum(r['flag_speed'] for r in rs)} travel {sum(r['flag_travel'] for r in rs)} "
              f"(base {sum(x['flag_travel'] for x in b)}) oscill {sum(r['flag_oscill'] for r in rs)}; creep {sum(r['creep'] for r in rs)}; "
              f"failing a spread bar {sum(1 for r in rs if r['n_problems'])} (base {sum(1 for x in b if x['n_problems'])})")
        d = lambda k, s=1e3: np.array([s * (r[k] - x[k]) for r, x in zip(rs, b)])
        for k, unit in (("foot_travel_after1s_m", "mm"), ("com_travel_window_mm", "mm")):
            sc = 1e3 if k.endswith("_m") else 1.0
            dd = d(k, sc)
            print(f"   {k}: variant minus base, median {np.median(dd):+.4f}, range {dd.min():+.4f} to {dd.max():+.4f} {unit}; "
                  f"base median {np.median([sc * x[k] for x in b]):.4f}")
        for k in STANCE:
            sc = 1.0 if (k.endswith("_deg") or k == "left_load_share_mean") else 1e3
            dd = np.array([sc * (r["metrics"][k] - x["metrics"][k]) for r, x in zip(rs, b)])
            spread = np.ptp([sc * x["metrics"][k] for x in b])
            unit = "deg" if k.endswith("_deg") else ("" if k == "left_load_share_mean" else ("mm/s" if k.endswith("_mps") else "mm"))
            print(f"   {k:30s} max |variant - base| {np.abs(dd).max():.4f} {unit} (base seed-to-seed range {spread:.4f})")
