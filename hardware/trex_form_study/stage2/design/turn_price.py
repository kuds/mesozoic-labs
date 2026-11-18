"""Scratch Stage 0 pricing (prescribed motion, inverse dynamics, the sizing margins) of a steady pi/4 rad/s turn from
the 0.40 m/s walk on 75-M-H-10 with quicker steps (shorter stride at the same speed).  Not a record."""
import math, sys, time, json
from dataclasses import replace
import numpy as np
from designlab import gait, turning, statics, sizing, envelope
from designlab.stage1 import refs
from designlab.stage2 import look
gait.set_accuracy("r2")
body = sys.argv[1]
src = refs.source(body); M, q0, H, design = src["M"], src["q_home"], src["H"], src["design"]
bus = src["record"]["selection"]["bus"]
base = look.speed_spec(H)
res = []
for f, w in [(1.0, 0.0), (0.8, 0.0), (0.8, math.pi / 4), (0.7, 0.0), (0.7, math.pi / 4), (0.9, math.pi / 4)]:
    t0 = time.time()
    sp = replace(base, stride_rel=base.stride_rel * f)
    tr = turning.generate(M, q0, H, sp, yaw_rate=w)
    idr = turning.inverse_dynamics(M, tr, turning.MU, split="exact", ds_friction=False)
    ms = sizing._motor_space(M.names, idr["tau"], idr["omega"], src["layout"])
    rat = envelope._ratios(design, bus, ms)
    wg, wv = max(rat.items(), key=lambda kv: kv[1]["worst"])
    s = turning.summarize(M, tr, idr)
    yaw_use = max(s["range_use"][f"{sd}_hip_yaw"] for sd in statics.SIDES)
    row = dict(stride_scale=f, stride_H=round(sp.stride_rel, 4), period_s=round(float(sp.describe(H)["period"]), 4), yaw_rate=round(w, 4),
               ik_ok=bool(np.all(tr["ik_ok"])), hip_yaw_use=round(yaw_use, 3), worst_group=wg, worst=round(wv["worst"], 3),
               ratios={g: {k: round(float(v[k]), 3) for k in ("cont_ratio", "peak_ratio", "speed_ratio") if k in v} for g, v in rat.items()},
               seconds=round(time.time() - t0, 1))
    res.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "ratios"}), flush=True)
    print("   ", {g: v for g, v in row["ratios"].items()}, flush=True)
json.dump(res, open(f"/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/turn_price_{body}.json", "w"), indent=1)
