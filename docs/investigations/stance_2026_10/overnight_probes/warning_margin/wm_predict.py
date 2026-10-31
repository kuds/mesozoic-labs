"""Pre-registered predictions (no simulation): from early_warning/ew_series.csv (G_own, c0_own, own14 on the 0-1 deg
sweep, seeds 5001-5010 own states), verify_early_warning/vew_analyze.txt section G (G_L, the local slope at the
operating point) and wm_static.json (static network curve, computed before any nudged episode).  Writes
wm_predict.json and prints the table that PREREG.txt quotes."""
from __future__ import annotations

import csv
import json
import re

import wm_common as C

PLANT = 5.16
ser = {}
with open(C.SCR / "com_revisit/notes/early_warning/ew_series.csv") as fh:
    for r in csv.DictReader(fh):
        ser[(r["run"], int(r["steps"]))] = r
GL = {}
for line in open(C.SCR / "com_revisit/notes/verify_early_warning/vew_analyze.txt"):
    m = re.match(r"\s+(s\d\d)\s+(\d+) \|\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+) \|\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s*$", line)
    if m:
        GL[(m.group(1), int(m.group(2)))] = {"G_P_v": float(m.group(3)), "c0_v": float(m.group(4)), "bstar_P_v": float(m.group(5)),
                                             "G_L": float(m.group(6)), "own_v": float(m.group(7)), "bstar_L": float(m.group(8))}
st = json.loads((C.HERE / "wm_static.json").read_text())
out = {}
print(f"{'ckpt':14s} {'G_P':>7s} {'c0':>7s} {'own14':>7s} {'FP':>6s} | {'b*_P':>6s} {'b*_L':>6s} {'b*_S':>6s} {'1-own':>6s} | "
      f"{'cap':>5s} {'th_need_P':>9s} {'D*_P(deg)':>9s} | {'S max':>6s} {'D*_S':>5s}")
for run, steps in C.CKPTS:
    r = ser[(run, steps)]
    G, c0, own, FP = float(r["G_own"]), float(r["c0_own"]), float(r["own14"]), float(r["FP_own"])
    bP = 1.0 - G - c0
    gl = GL[(run, steps)]
    assert abs(gl["bstar_P_v"] - bP) < 1.5e-3, (run, steps, gl, bP)
    s = st[f"{run}_{steps}"]
    cap = s["t3_clip_deg"]
    th_need = PLANT * (1.0 - c0) / G if G > 0 else None
    dP = PLANT * bP / G if (th_need is not None and th_need <= cap) else None
    out[f"{run}_{steps}"] = {"G_P": G, "c0": c0, "own14": own, "FP": FP, "bstar_P": bP, "G_L": gl["G_L"],
                             "bstar_L": gl["bstar_L"], "bstar_S": s["bstar_S"], "gap_open": 1.0 - own, "cap_deg": cap,
                             "theta_need_P": th_need, "delta_star_P": dP, "static_max_cmd": s["max_cmd_any_theta"],
                             "delta_star_S": s["delta_star_S"]}
    tn = f"{th_need:9.2f}" if th_need is not None else f"{'G<=0':>9s}"
    print(f"{run} {steps:>10d} {G:+7.3f} {c0:+7.3f} {own:+7.3f} {FP:+6.3f} | {bP:6.3f} {gl['bstar_L']:6.3f} {s['bstar_S']:6.3f} "
          f"{1 - own:6.3f} | {cap:5.2f} {tn} {('%9.2f' % dP) if dP is not None else '     none':>9s} | "
          f"{s['max_cmd_any_theta']:+6.3f} {'none' if s['delta_star_S'] is None else '%.2f' % s['delta_star_S']:>5s}")
(C.HERE / "wm_predict.json").write_text(json.dumps(out, indent=1))
