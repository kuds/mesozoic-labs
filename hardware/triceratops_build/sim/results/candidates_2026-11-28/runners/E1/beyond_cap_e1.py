"""E1 supporting check, ADDED AFTER A RESULT (not part of the pre-registered rule): every logged pose read at the
frozen geometry's 50 mm cap (ClashGeometry, check_packaging60.CAP_M), so the actual least clearance is unknown beyond
">= 50 mm". This brackets it, with the same parts and poses:
  lower bound : the least, over every logged pose and part pair, of the containing-capsule bound
                (check_packaging60.bound_capsule / seg_dist, as ClashGeometry's prefilter; a true lower bound)
  measured    : mj_geomDistance with distmax 0.5 m over every part pair at the K poses with the smallest bound
                (an upper bound on the true least over all poses)
Writes candidates/E1/beyond_cap.json (scratch) and sim/results/candidates_2026-11-28/E1/e1_beyond_cap.json."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/mesozoic-labs/hardware/triceratops_build/sim")
import candidate_checks as C  # noqa: E402
import run_e1 as R  # noqa: E402

K = 30
DMAX = 0.5


def bounds_for(geom, Q):
    """(N x keys) the least capsule bound per pose and key."""
    CP = geom.CP
    out = {k: np.zeros(len(Q)) for k in C.CLASH_PAIRS}
    for n, q in enumerate(Q):
        geom.pose(q)
        d = geom.d
        X = d.geom_xpos[geom.allg]
        Rm = d.geom_xmat[geom.allg].reshape(-1, 3, 3)
        hw = np.einsum("nij,nj->ni", Rm, geom.half)
        p0, p1 = X - hw, X + hw
        for key, (ga, gb, _) in geom.pairs.items():
            b = CP.seg_dist(p0[ga], p1[ga], p0[gb], p1[gb]) - geom.rad[ga] - geom.rad[gb]
            out[key][n] = float(b.min())
    return out


def exact(geom, q, key):
    geom.pose(q)
    a, b = C.CLASH_PAIRS[key]
    best, parts = math.inf, None
    ft = np.zeros(6)
    for g in geom.parts[a]:
        for h in geom.parts[b]:
            x = float(geom.mj.mj_geomDistance(geom.m, geom.d, g, h, DMAX, ft))
            if x < best:
                best, parts = x, (geom.m.geom(g).name, geom.m.geom(h).name)
    return best, parts


def job(ctrl):
    t0 = time.time()
    geom = C.ClashGeometry()
    names, Q, B, ep_of = R.pooled(ctrl)
    ts = np.concatenate([np.load(R.EPI / f"{n}.npz")["t"] for n in names])
    bd = bounds_for(geom, Q)
    res = {}
    for key in C.CLASH_PAIRS:
        order = np.argsort(bd[key])[:K]
        ex = [(exact(geom, Q[i], key), i) for i in order]
        (d, parts), i = min(ex, key=lambda z: z[0][0])
        res[key] = {"lower_bound_mm": round(1000 * float(bd[key].min()), 2),
                    "measured_least_mm": round(1000 * d, 2), "measured_parts": parts,
                    "measured_at": {"episode": names[ep_of[i]], "t_s": round(float(ts[i]), 3),
                                    "pose_deg": {j: round(math.degrees(float(v)), 2) for j, v in zip(C.LOG_JOINTS, Q[i])}},
                    "poses_measured_exactly": K, "samples": int(len(Q)),
                    "frozen_geometry_least_mm": round(1000 * float(B[key].min()), 3)}
    res["wall_s"] = round(time.time() - t0, 1)
    return ctrl, res


def main():
    import multiprocessing as mp
    out = {}
    with mp.get_context("spawn").Pool(2) as pool:
        for ctrl, res in pool.imap_unordered(job, R.CONTROLLERS):
            out[ctrl] = res
            print(ctrl, {k: (v["lower_bound_mm"], v["measured_least_mm"]) for k, v in res.items() if k != "wall_s"},
                  flush=True)
    doc = {"label": "SUPPORTING, ADDED AFTER A RESULT (not part of E1's pre-registered rule): the frozen geometry caps "
                    "every clearance at 50 mm and every logged pose read 50 mm; this brackets the actual least",
           "method": {"lower_bound": "least containing-capsule bound over all logged poses and part pairs "
                                     "(check_packaging60.bound_capsule, seg_dist; ClashGeometry's prefilter)",
                      "measured": f"mj_geomDistance, distmax {DMAX} m, all part pairs, at the {K} poses with the "
                                  "smallest bound per controller and pair"},
           "units": "mm", "parts": "as ClashGeometry: solid parts, limb covers left out, full capsule soles in, probe rods",
           "per_controller": {c: out[c] for c in R.CONTROLLERS},
           "provenance": C.provenance(runner=Path(__file__).resolve(),
                                      extra={"run_e1.py": R.RUNNER})}
    (R.SCR / "beyond_cap.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (R.OUT / "e1_beyond_cap.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
