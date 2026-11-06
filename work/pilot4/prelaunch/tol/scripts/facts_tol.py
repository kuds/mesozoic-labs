"""Tolerance lane: the numbers the final text quotes from the entry baseline (v3's pre-launch ENTRY BASELINE and the
corrections C5, C6, C9a, C9b, C12a, C12b), read from this lane's validation records (val_final/baseline_s*.json,
written by the edited pilot4_straight.py, TWIN_HEADING_TOL_DEG 1e-4), with the fix lane's formatting
(make_text_corrections.py's e1 / f1). Also the 40-episode panel's labels by pilot 3's rule and (a), (b), (c) per seed.
JSON only; standard library only.

usage: python -I -B facts_tol.py VAL_FINAL_DIR OUT.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

VAL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (42, 52, 60)
B = {s: json.loads((VAL / f"baseline_s{s}.json").read_text()) for s in SEEDS}


def e1(x):
    return f"{float(x):.2e}".replace("e-0", "e-")


def per_seed(fn, sep=" / "):
    return sep.join(fn(s) for s in SEEDS)


def tw(s):
    return B[s]["checks"]["turned_twin"]


f = {"tool_sha256": sorted({B[s]["tool_sha256"] for s in SEEDS}),
     "constants_TWIN_HEADING_TOL_DEG": sorted({B[s]["constants"]["TWIN_HEADING_TOL_DEG"] for s in SEEDS}),
     "head": sorted({B[s]["head"] for s in SEEDS}), "per_seed": {}}
for s in SEEDS:
    b = B[s]
    eps = b["panel"]["episodes"]
    lab = b.get("labels") or {}
    p = b["parts"]
    t = tw(s)
    f["per_seed"][s] = {
        "mode": b["mode"], "partial": b["partial"], "validation": b["validation"],
        "pilot4_env": b["env"].get("pilot4_env"), "environments_file": b.get("environments_file"),
        "checkpoint_sha256": b.get("checkpoint_sha256"), "vecnorm_sha256": b.get("vecnorm_sha256"),
        "panel_episodes": len(eps), "full_length": sum(1 for e in eps if e["full"]),
        "panel_labels": dict(Counter(e.get("label") for e in eps)),
        "seed_outcome": lab.get("seed_outcome"), "labels_block": {k: v for k, v in lab.items() if k != "labels"},
        "a": {k: p["a"].get(k) for k in ("verdict", "median_signed_deg", "p95_abs_deg", "n")},
        "b": {k: p["b"].get(k) for k in ("verdict", "D_p95_deg", "W0_p95_deg", "n", "report_crab_median_deg",
                                         "report_end_cross_track_median_m")},
        "c": {k: p["c"].get(k) for k in ("verdict", "n_counting", "median_r_deg", "p95_r_deg", "within_11_5",
                                         "within_5")},
        "straight": p["straight"],
        "turned_twin": {"episodes": t["episodes"], "flagged": len(t["bad"]), "max_abs_dev_deg": t["max_abs_dev_deg"],
                        "max_action_diff_after_651": t.get("max_action_diff_after_651")},
        "prefix": f"{b['checks']['prefix']['equal']}/{b['checks']['prefix']['episodes']}",
        "obs651": f"{b['checks']['obs651']['only_command_channels']}/{b['checks']['obs651']['episodes']}",
        "zero_nudge": f"{b['checks']['zero_nudge']['equal']}/{b['checks']['zero_nudge']['episodes']}",
        "action_delta_vs_blind_max": b["checks"]["action_delta_vs_blind"]["max"],
        "prey_in_place": b["recovery"]["prey_in_place"]["summary"],
        "turned_at_reset": b["recovery"]["turned_at_reset"]["summary"],
    }
twin_max = max(tw(s)["max_abs_dev_deg"] for s in SEEDS)
f["text"] = {
    "C5_crab_deg": per_seed(lambda s: f"{B[s]['parts']['b']['report_crab_median_deg']:+.1f}"),
    "C5_cross_m": per_seed(lambda s: f"{B[s]['parts']['b']['report_end_cross_track_median_m']:+.2f}"),
    "C6_twin_episodes": sum(tw(s)["episodes"] for s in SEEDS),
    "C6_twin_max_deg": e1(twin_max),
    "C6_flagged_by_the_tool":per_seed(lambda s: f"s{s} {len(tw(s)['bad'])} of {tw(s)['episodes']}", ", "),
    "C6_beyond_1e-5_report_only": per_seed(
        lambda s: f"s{s} {sum(1 for e in B[s]['recovery']['episodes'] if e.get('variant', 'c') == 'c' and e.get('heading_minus_twin_at_1000_deg') is not None and abs(e['heading_minus_twin_at_1000_deg'] - e['angle_deg']) > 1e-5)}",
        ", "),
    "C9a_actions_within": e1(max(tw(s)["max_action_diff_after_651"] for s in SEEDS)),
    "C9a_prefix": per_seed(lambda s: f"{B[s]['checks']['prefix']['equal']}/{B[s]['checks']['prefix']['episodes']}"),
    "C9a_obs651": per_seed(
        lambda s: f"{B[s]['checks']['obs651']['only_command_channels']}/{B[s]['checks']['obs651']['episodes']}"),
    "C9_c_median_r": per_seed(lambda s: f"{B[s]['parts']['c']['median_r_deg']:.1f}"),
    "C9_c_p95_r": per_seed(lambda s: f"{B[s]['parts']['c']['p95_r_deg']:.1f}"),
    "C9_c_within_11_5": per_seed(lambda s: f"{B[s]['parts']['c']['within_11_5']}/{B[s]['parts']['c']['n_counting']}",
                                 ", "),
    "C9_c_within_5": per_seed(lambda s: f"{B[s]['parts']['c']['within_5']}/{B[s]['parts']['c']['n_counting']}", ", "),
    "a_median_signed": per_seed(lambda s: f"{B[s]['parts']['a']['median_signed_deg']:+.1f}"),
    "a_p95_abs": per_seed(lambda s: f"{B[s]['parts']['a']['p95_abs_deg']:.1f}"),
    "b_D_p95": per_seed(lambda s: f"{B[s]['parts']['b']['D_p95_deg']:.1f}"),
    "b_W0_p95": per_seed(lambda s: f"{B[s]['parts']['b']['W0_p95_deg']:.1f}"),
}
OUT.write_text(json.dumps(f, indent=1, default=str) + "\n")
print(json.dumps(f, indent=1, default=str))
