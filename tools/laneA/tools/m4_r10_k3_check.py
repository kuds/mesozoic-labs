"""M4 on r10, end to end (validation only): compare judge_r10ltsb.py's two K3 noise tests on an r10 checkpoint with the
in-run K3 panel that the branch's own training wrote at the same step (TEL: "the replication judge's two K3 noise
tests, arithmetic unchanged"), per test and spawn seed: episode length, fall, termination reason and return (exact),
and the sigma vectors' sha256 (own: exp(log_std); joint: the seed's node).

    PY -I -B m4_r10_k3_check.py JUDGE_JSON K3_PANEL_JSON K3_PANEL_CSV OUT.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
VERSION = "m4_r10_k3_check 1.0 (2026-10-26)"


def sha_f64(v) -> str:
    return hashlib.sha256(np.asarray(v, dtype="<f8").tobytes()).hexdigest()


def main() -> int:
    print(VERSION, flush=True)
    jp, kp, cp, out = sys.argv[1:5]
    j = json.load(open(jp))
    k = json.load(open(kp))
    steps = j["checks"]["checkpoint_num_timesteps"]
    [panel] = [p for p in k["panels"] if p["timesteps"] == steps]
    rows = [r for r in csv.DictReader(open(cp)) if int(r["timesteps"]) == steps and r["panel"] == str(panel["panel"])]
    res = {"tool": VERSION, "judge": jp, "k3_panel": kp, "steps": steps, "tests": {}, "problems": []}
    for test, key in (("own", "noise_matched_own"), ("joint", "noise_matched_control_joint")):
        jrows = {r["seed"]: r for r in j[key]["episodes_detail"]}
        prow = {int(r["seed"]): r for r in rows if r["test"] == test}
        cmp = []
        for seed in range(3042, 3062):
            a, b = jrows.get(seed), prow.get(seed)
            if a is None or b is None:
                res["problems"].append(f"{test} {seed}: missing on one side")
                continue
            same = (a["length"] == int(b["length"]) and int(not a["reached_horizon"]) == int(b["fall"])
                    and a["reward"] == float(b["return"]) and str(a["termination_reason"]) == b["termination_reason"])
            cmp.append({"seed": seed, "judge_length": a["length"], "panel_length": int(b["length"]),
                        "judge_return": a["reward"], "panel_return": float(b["return"]), "equal": same})
        res["tests"][test] = {"judge_falls": j[key]["falls"], "panel_falls": panel["tests"][test]["falls"],
                              "episodes_equal": sum(c["equal"] for c in cmp), "of": len(cmp), "rows": cmp}
    res["own_sigma_sha256"] = {"judge": sha_f64(j["noise_matched_own"]["sigma_vector"]), "panel": panel["own_sigma_sha256"]}
    res["joint_sigma_sha256"] = {"judge": sha_f64(j["noise_matched_control_joint"]["sigma_vector"]),
                                 "judge_k3_panel_sigma_record": (j.get("k3_panel_sigma") or {}).get("sha256"),
                                 "panel": panel["joint_sigma_sha256"]}
    res["PASS"] = (not res["problems"]
                   and all(t["episodes_equal"] == t["of"] == 20 and t["judge_falls"] == t["panel_falls"]
                           for t in res["tests"].values())
                   and res["own_sigma_sha256"]["judge"] == res["own_sigma_sha256"]["panel"]
                   and res["joint_sigma_sha256"]["judge"] == res["joint_sigma_sha256"]["panel"]
                   == res["joint_sigma_sha256"]["judge_k3_panel_sigma_record"])
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"PASS": res["PASS"], "steps": steps,
                      **{t: {kk: v[kk] for kk in ("judge_falls", "panel_falls", "episodes_equal", "of")}
                         for t, v in res["tests"].items()},
                      "own_sigma_sha256": res["own_sigma_sha256"], "joint_sigma_sha256": res["joint_sigma_sha256"],
                      "problems": res["problems"]}))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
