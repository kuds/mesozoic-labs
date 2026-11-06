"""Read the tolerance stage's three baseline records and print every number the frozen candidate quotes from them,
plus a few the text does not quote (W0 - D range over the whole panel, channels seen at 651). Read only.
Usage: python -I facts_baseline.py TOL_VAL_FINAL_DIR OUT_JSON"""
import json
import math
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])


def num(v):
    return math.inf if v == "inf" else v


def pct(vals, p):
    x = np.asarray([num(v) for v in vals], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        r = float(np.percentile(x, p))
    return math.inf if math.isnan(r) else r


facts = {}
for seed in (42, 52, 60):
    d = json.loads((src / f"baseline_s{seed}.json").read_text())
    p = d["parts"]
    eps = d["panel"]["episodes"]
    full = [e for e in eps if e["full"] and e["used"]]
    first20 = [e for e in full if 3042 <= e["reset_seed"] <= 3061]
    yaw20 = [e["yaw_change_deg"] for e in eps if 3042 <= e["reset_seed"] <= 3061 and e["yaw_change_deg"] is not None]
    gap = [e["W0_deg"] - e["D_deg"] for e in full]
    gap20 = [e["W0_deg"] - e["D_deg"] for e in first20]
    rec = d["recovery"]
    labels = d.get("labels")
    f = {
        "tool_sha256": d["tool_sha256"], "head": d["head"], "constants_twin": d["constants"]["TWIN_HEADING_TOL_DEG"],
        "a": {k: p["a"].get(k) for k in ("verdict", "n", "median_signed_deg", "p95_abs_deg", "n_full_of_panel")},
        "b": {k: p["b"].get(k) for k in ("verdict", "n", "D_p95_deg", "W0_p95_deg", "D_median_deg", "D_max_deg",
                                         "W0_minus_D_min_deg", "W0_minus_D_max_deg", "report_R_median_deg",
                                         "report_R_p95_deg", "report_crab_median_deg",
                                         "report_end_cross_track_median_m")},
        "c": {k: p["c"].get(k) for k in ("verdict", "n_episodes", "n_counting", "left_out_ended_before_nudge",
                                         "left_out_not_yet_passed", "fell_after_nudge", "within_11_5", "within_5",
                                         "share_removed_median", "median_r_deg", "p95_r_deg")},
        "straight": p["straight"],
        "checks": d["checks"],
        "prey_in_place": rec["prey_in_place"]["summary"],
        "turned_at_reset": rec["turned_at_reset"]["summary"],
        "rows_3042_3061": {
            "n_full": len(first20), "yaw_n": len(yaw20),
            "yaw_median": pct(yaw20, 50), "yaw_p95_abs": pct([abs(v) for v in yaw20], 95),
            "D_p95": pct([e["D_deg"] for e in first20], 95), "D_med": pct([e["D_deg"] for e in first20], 50),
            "D_max": max(e["D_deg"] for e in first20),
            "R_med": pct([e["R_deg"] for e in first20], 50), "R_p95": pct([e["R_deg"] for e in first20], 95),
            "W0_p95": pct([e["W0_deg"] for e in first20], 95),
            "W0_minus_D_min": min(gap20), "W0_minus_D_max": max(gap20),
            "crab_median": pct([e["crab_median_deg"] for e in first20], 50),
            "cross_track_median": pct([e["end_cross_track_m"] for e in first20], 50),
        },
        "W0_minus_D_whole_panel": [min(gap), max(gap)],
        "short_episodes": [(e["reset_seed"], e["T"]) for e in eps if not e["full"]],
        "labels": labels if not isinstance(labels, dict) else {k: v for k, v in labels.items() if k != "rows"},
        "psi_cmd": d["psi_cmd"],
    }
    # the three 1e-5 exceedances (dev per nudge)
    devs = []
    for e in rec["episodes"]:
        h = e.get("heading_minus_twin_at_1000_deg")
        if h is not None:
            devs.append((e["reset_seed"], e["angle_deg"], h - e["angle_deg"]))
    f["twin_devs_over_1e-5"] = sorted([x for x in devs if abs(x[2]) > 1e-5], key=lambda x: -abs(x[2]))
    f["twin_n"] = len(devs)
    facts[str(seed)] = f
out.write_text(json.dumps(facts, indent=1, default=str))
print(json.dumps(facts, indent=1, default=str))
