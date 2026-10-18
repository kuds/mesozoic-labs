"""Per-bar margins of each offline certificate: policy vs statue over all 240 (seed, heading) episodes."""
import json, sys, statistics
from pathlib import Path
O = Path(sys.argv[1])
BARS = [("max_sole_cop_fore_aft", "max", 0.80), ("max_sole_tilt_deg", "max", 2.0), ("max_sole_corner_lift_m", "max", 0.004),
        ("min_sole_contacts", "min", 1.5), ("min_foot_load_share", "min", 0.40), ("min_foot_load_share_windowed", "min", 0.35),
        ("settle_peak_floor_force_bw", "max", 1.5), ("settle_stance_width_change_m", "max", 0.08), ("settle_touchdowns", "max", 2),
        ("episode_yaw_change_deg", "max", 25.0), ("window_displacement_m", "max", 0.10), ("max_actuator_saturation_fraction", "max", 0.10),
        ("all_feet_support", "min", 0.98)]
out = {}
lines = []
for d in sorted(O.glob("*_opt*")):
    p = d / "stance_heading_report.json"
    if not p.exists():
        continue
    rep = json.loads(p.read_text())
    pol = [r for h in rep["headings"] for r in h["episode_evidence"]]
    sta = [r for h in rep["headings"] for r in (h["statue"] or {}).get("episode_evidence", [])]
    res = {}
    for key, kind, bar in BARS:
        def stats(rows):
            v = [float(r[key]) for r in rows if r.get(key) is not None and r.get("length", 0) >= 1000]
            if not v:
                return None
            worst = max(v) if kind == "max" else min(v)
            near = sum(1 for x in v if (x > 0.8 * bar if kind == "max" else x < 1.25 * bar))
            over = sum(1 for x in v if (x > bar if kind == "max" else x < bar))
            return {"worst": worst, "median": statistics.median(v), "n": len(v), "past_bar": over, "within_20pct_of_bar": near}
        res[key] = {"bar": bar, "kind": kind, "policy": stats(pol), "statue": stats(sta)}
    out[d.name] = res
    lines.append(f"== {d.name}  (240 policy and 240 statue episodes; full-horizon only)")
    lines.append(f"   {'bar':34s} {'limit':>7s} | {'policy worst':>12s} {'median':>8s} {'past':>5s} {'near':>5s} | {'statue worst':>12s} {'median':>8s} {'past':>5s}")
    for key, v in res.items():
        p_, s_ = v["policy"], v["statue"]
        f = lambda x: "-" if x is None else f"{x:.4g}"
        lines.append(f"   {key:34s} {v['bar']:>7g} | {f(p_ and p_['worst']):>12s} {f(p_ and p_['median']):>8s} {(p_ or {}).get('past_bar','-'):>5} {(p_ or {}).get('within_20pct_of_bar','-'):>5} | {f(s_ and s_['worst']):>12s} {f(s_ and s_['median']):>8s} {(s_ or {}).get('past_bar','-'):>5}")
(O / "cert_margins.json").write_text(json.dumps(out, indent=1))
(O / "cert_margins.txt").write_text("\n".join(lines) + "\n\n'near' = past 80% of a max bar or under 125% of a min bar.\n")
print("\n".join(lines))
