"""E2(a) supporting reading, pre-registered as "shown beside" (prereg section 3 item 6): the speed check with the
fore knees' X6-60 at its own allowance (rl/analysis/gait_pipe1.py RPM_ALLOW "front elbow": 108.0 rpm, about 0.70 x 176 rpm x 42 / 48,
peak 1.1 x that) and every X4 at 83 rpm. DESK arithmetic on e2a_summary.json's per-drive p99 and peak (already x r).
Not judged; the judged verdict stays the 83 rpm one. Writes e2a_speed_x6_allowance_supporting.csv beside it."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, "/home/user/mesozoic-labs/hardware/triceratops_build/sim")
sys.path.insert(0, "/home/user/mesozoic-labs/hardware/triceratops_build/rl/analysis")
import candidate_checks as C
import gait_pipe1 as G
RES = C.RESULTS / "E2"
x6 = float(G.RPM_ALLOW["front elbow"]) if isinstance(G.RPM_ALLOW, dict) and "front elbow" in G.RPM_ALLOW else None
print("RPM_ALLOW:", G.RPM_ALLOW)
s = json.loads((RES / "e2a_summary.json").read_text())
grid = list(csv.DictReader((RES / "e2a_grid.csv").open()))
heat = {(r["transmission"], float(r["r"]), r["placement"], r["policy"], r["block"]): (r["C1_pass"] == "True", r["C2_pass"] == "True") for r in grid}
rows = []
for sc in s["speed_checks"]:
    ok = True
    worst = []
    for d, p, k in zip(sc["drives"], sc["p99_rpm"], sc["peak_rpm"]):
        cap = x6 if C.drive_of(d) == "X6-60" else C.CAP_RPM
        good = p <= cap and k <= C.PEAK_FACTOR * cap
        ok &= good
        if not good:
            worst.append(f"{d} p99 {p:.1f} peak {k:.1f} (cap {cap:.1f})")
    rows.append({"policy": sc["policy"], "block": sc["block"], "r": sc["r"], "placement": sc["placement"],
                 "x6_cap_rpm": round(x6, 2), "speed_pass_83_judged": sc["pass"], "speed_pass_x6_allowance": ok,
                 "failing_drives": "; ".join(worst)})
with (RES / "e2a_speed_x6_allowance_supporting.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
# which judged (transmission, r, placement) would pass C1, C2 and this supporting speed check for all 8 policy-blocks
by = {}
for r in rows:
    by.setdefault((r["r"], r["placement"]), []).append(r)
for tr in ("rod", "belt", "cable"):
    for (rr, pl), rs in sorted(by.items(), key=lambda x: (x[0][1], x[0][0])):
        allp = all(heat[(tr, float(rr), pl, x["policy"], x["block"])][0] and heat[(tr, float(rr), pl, x["policy"], x["block"])][1]
                   and x["speed_pass_x6_allowance"] for x in rs)
        fails = sorted({x["policy"] for x in rs if not x["speed_pass_x6_allowance"]})
        print(f"{tr} r {rr} {pl}: C1+C2+speed(X6 at allowance) all 8 pass: {allp}; speed fails (X6 allowance) on {fails}")
