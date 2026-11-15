"""Lead's scratch scan, 15 Nov: the quick turn's joint margins other than the thigh twist, by stride factor.
Kinematic pass only (no ZMP correction), V0's 0.40 m/s walk on 75-M-H-10, both directions. Prescribed motion; no record."""
import json, math, sys
import numpy as np
from designlab import statics
from designlab.stage1 import plant, refs
from designlab.stage2 import look
from designlab.stage2 import zz_moves_scratch as moves

OUT, walk = sys.argv[1], sys.argv[2]
src = refs.source("75-M-H-10")
H = src["H"]
spec = look.speed_spec(H)
ref = refs.Reference("v0.4", refs.Reference.load(walk).arrays)
a = moves.GaitX.of("v0.4", spec, H)
m = src["M"].m
rng = plant.usable_ranges(src["body"])
res = {}
with look.accuracy("r2"):
    for f in (0.80, 0.75, 0.70, 0.65, 0.60):
        for d, sgn in (("left", 1.0), ("right", -1.0)):
            k = moves.kinematic(src, moves.turn(f"t_{d}", a, math.pi / 2, sgn * math.pi / 4, f), ref)
            q = np.asarray(k["q"]) if "q" in k else None
            row = dict(ik_ok=k["ik_ok"], twist_use=k["hip_yaw_range_use"])
            if q is not None:
                mar = {}
                for jn, (lo, hi) in rng.items():
                    qv = q[:, m.jnt_qposadr[statics.jid(m, jn)]]
                    mar[jn] = float(min(hi - qv.max(), qv.min() - lo))
                other = {j: v for j, v in mar.items() if not j.endswith("_hip_yaw")}
                jmin = min(other, key=other.get)
                row.update(tightest_other=jmin, tightest_other_rad=other[jmin],
                           hip_roll={j: round(v, 4) for j, v in mar.items() if "hip_roll" in j})
            else:
                row["keys"] = sorted(k.keys())
            res[f"{f:g}_{d}"] = row
            print(f, d, row, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
