"""Nominal walk alone lifted to 0.19 / 0.195 H on the carpet-ready 60 cm bodies (brisk walk at its reference lift).

Run from hardware/trex_form_study: PYTHONPATH=. python round2/refine/nominal_lift.py <scratch dir>; writes nominal_lift.json there
(copied to round2/refine/carpet_lift_carpet_60_nominal_only.json)."""
import json, sys
from dataclasses import replace
from pathlib import Path
from designlab import gait, round2, sizing
from designlab.pipeline import build_candidate
gait.set_accuracy(round2.GAIT_ACCURACY)
out = {}
for cid in ("60-M-T-10", "60-M-H-10"):
    rec = json.loads(Path("round2/refine/carpet_60/candidates", f"{cid}.json").read_text())
    res = build_candidate(round2.candidate(cid, tuple(tuple(o) for o in rec["overrides"])), Path(sys.argv[1]),
                          selection=dict(rec["selection"]), B=rec["B"], write=False)
    M, q0, d = res["model"], res["q_home"], res["design"]
    n0 = sizing.NOMINAL
    for lift in (0.19, 0.195):
        sizing.NOMINAL = replace(n0, lift_frac=lift)
        req = sizing.requirements(M, q0, d.sk.H, d.cand.leg_layout)
        act = d.act("knee"); spd = act.speed_at(rec["selection"]["bus"])
        r = req["groups"]["knee"]
        out[f"{cid} {lift}"] = dict(knee_speed=r["speed_req"] / spd, knee_cont=r["cont"] / act.sustained, knee_peak=r["peak_req"] / act.peak)
        print(cid, lift, out[f"{cid} {lift}"], flush=True)
    sizing.NOMINAL = n0
Path(sys.argv[1], "nominal_lift.json").write_text(json.dumps(out, indent=1))
