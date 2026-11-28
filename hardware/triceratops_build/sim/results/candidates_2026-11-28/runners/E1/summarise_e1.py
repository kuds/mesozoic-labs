"""E1 summary (design/PREREG_CANDIDATES_2026_11_28.md, E1): reads the scratch logs, clearances and stop evaluations
written by run_e1.py and writes sim/results/candidates_2026-11-28/E1/ (e1_summary.json, e1_joint_percentiles.csv,
e1_stops.csv, e1_by_condition.csv, E1_SUMMARY.md). The verdicts come from candidate_checks.e1_verdict,
evaluate_stop (run_e1.py stops) and coupled_limit only."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import candidate_checks as C
import run_e1 as R

HERE = Path(__file__).resolve()


def _deg(q):
    return {j: round(math.degrees(float(v)), 3) for j, v in zip(C.LOG_JOINTS, q)}


def least_info(geom, names, Q, B, ep_of, key_list, ts):
    """Least over the keys, with the pose, the episode and time, the parts."""
    v = np.min(np.vstack([B[k] for k in key_list]), axis=0)
    i = int(np.argmin(v))
    side = key_list[int(np.argmin([B[k][i] for k in key_list]))]
    c = geom.clearance(Q[i])
    # ClashGeometry.clearance names the parts of the least with the soles in only; with the soles aside: None
    parts = list(c[side][2]) if side in c else None
    at_cap = bool(float(v.min()) >= C._cp().CAP_M)
    return {"least_mm": round(1000 * float(v[i]), 3), "at_cap": at_cap,
            "note": ("every sample reads the geometry's 50 mm cap, so this is the first logged sample, not a nearest "
                     "pose; see e1_beyond_cap.json") if at_cap else None, "side_key": side, "episode": names[ep_of[i]],
            "t_s": round(float(ts[i]), 3), "parts": parts, "pose_deg": _deg(Q[i])}


def shares(B, key_list):
    v = np.min(np.vstack([B[k] for k in key_list]), axis=0)
    return {"n": int(len(v)), "under_0": int((v < 0).sum()), "under_0_share": float((v < 0).mean()),
            "under_5mm": int((v < 0.005).sum()), "under_5mm_share": float((v < 0.005).mean())}


def times(names):
    return np.concatenate([np.load(R.EPI / f"{n}.npz")["t"] for n in names])


def summarise():
    t0 = time.time()
    R.OUT.mkdir(parents=True, exist_ok=True)
    geom = C.ClashGeometry()
    metas = {}
    for job in R.jobs():
        n = R.ep_name(job[0], job[1], job[3])
        metas[n] = json.loads((R.EPI / f"{n}.json").read_text())
    stops = C.candidate_stops()
    keys9, keys10 = ["9L", "9R"], ["10"]
    per_controller, reported, perc_rows, stop_rows, cond_rows = {}, {}, [], [], []
    for ctrl in R.CONTROLLERS:
        rep = {}
        for subset, fo in (("all", False), ("fallen_left_out", True)):
            names, Q, B, ep_of = R.pooled(ctrl, fallen_out=fo)
            ts = times(names)
            r = {"episodes": len(names), "samples": int(len(Q))}
            for clash, kl in (("9", keys9), ("10", keys10)):
                r[f"#{clash}"] = {"least": least_info(geom, names, Q, B, ep_of, kl, ts), **shares(B, kl)}
                kl_s = [f"{k}_soles_aside" for k in kl]
                r[f"#{clash}_soles_aside"] = {"least": least_info(geom, names, Q, B, ep_of, kl_s, ts),
                                              **shares(B, kl_s)}
            for clash, kl in (("9L", ["9L"]), ("9R", ["9R"])):
                r[f"#9 side {clash}"] = {"least_mm": round(1000 * float(B[kl[0]].min()), 3), **shares(B, kl)}
            deg = np.degrees(Q)
            for j, jn in enumerate(C.LOG_JOINTS):
                p1, p50, p99 = np.percentile(deg[:, j], [1, 50, 99])
                perc_rows.append({"controller": ctrl, "subset": subset, "joint": jn, "samples": len(Q),
                                  "p1_deg": round(float(p1), 3), "p50_deg": round(float(p50), 3),
                                  "p99_deg": round(float(p99), 3), "min_deg": round(float(deg[:, j].min()), 3),
                                  "max_deg": round(float(deg[:, j].max()), 3)})
            rep[subset] = r
            if subset == "all":
                sres = {}
                for s in stops:
                    f = R.STP / f"{ctrl}__{s['id'].replace(':', '_').replace(' ', '_').replace('/', '_')}.json"
                    e = json.loads(f.read_text())
                    assert e["stop"] == s["id"]
                    e.pop("wall_s", None)
                    sres[s["id"]] = e
                for clash in ("9", "10"):
                    cl = C.coupled_limit(B, clash)
                    sres[cl["stop"]] = cl
                least = {"9": 1000 * float(C.clash_least(B, "9").min()),
                         "10": 1000 * float(C.clash_least(B, "10").min())}
                per_controller[ctrl] = {"least_mm": least, "stops": sres}
                # by condition (reported)
                conds = sorted({metas[n]["condition"] for n in names}, key=lambda c: list(
                    C.POLICY_CONDITIONS).index(c) if c in C.POLICY_CONDITIONS else 10 + list(R.TROT_CONDS).index(c))
                for cond in conds:
                    sel = np.isin(ep_of, [k for k, n in enumerate(names) if metas[n]["condition"] == cond])
                    l9 = C.clash_least({k: B[k][sel] for k in B}, "9")
                    l10 = C.clash_least({k: B[k][sel] for k in B}, "10")
                    eps = [n for n in names if metas[n]["condition"] == cond]
                    cond_rows.append({"controller": ctrl, "condition": cond, "episodes": len(eps),
                                      "falls": sum(metas[n]["fell"] for n in eps),
                                      "unstable": sum(metas[n]["unstable"] for n in eps), "samples": int(sel.sum()),
                                      "least_9_mm": round(1000 * float(l9.min()), 3),
                                      "share_9_under_0": round(float((l9 < 0).mean()), 6),
                                      "share_9_under_5mm": round(float((l9 < 0.005).mean()), 6),
                                      "least_10_mm": round(1000 * float(l10.min()), 3),
                                      "share_10_under_0": round(float((l10 < 0).mean()), 6),
                                      "share_10_under_5mm": round(float((l10 < 0.005).mean()), 6)})
        reported[ctrl] = rep
    verdict = C.e1_verdict(per_controller)
    stop_meta = {s["id"]: s for s in stops}
    for sid, v in verdict["acceptable_stops"].items():
        s = stop_meta.get(sid)
        for ctrl in R.CONTROLLERS:
            e = per_controller[ctrl]["stops"][sid]
            stop_rows.append({"stop": sid, "label": s["label"] if s else sid, "clash": e["clash"],
                              "record_least_mm_full_range": s["record_least_mm_full_range"] if s else "",
                              "controller": ctrl, "samples": e["samples"], "clipped": e["clipped"],
                              "clipped_share": round(e["clipped_share"], 6),
                              "least_mm_after": round(e["least_mm_after"], 3) if e["least_mm_after"] is not None
                              else "", "acceptable_on_controller": e["acceptable"],
                              "ACCEPTABLE_all_five": v["ACCEPTABLE"]})
    # episode table (falls, unstable, delays, poses)
    eps = []
    for n, m in metas.items():
        rec = m["record"]
        eps.append({"episode": n, "controller": m["controller"], "condition": m["condition"], "block": m["block"],
                    "seed": m["seed"], "fell": m["fell"], "termination": m["termination"], "unstable": m["unstable"],
                    "n_poses": m["n_poses"], "t_last_pose_s": m["t_last_pose_s"],
                    "q_nonfinite_dropped": m["q_nonfinite_dropped"],
                    "delay_steps": rec.get("delay_steps"), "physics_dt": rec.get("physics_dt"),
                    "forward_speed_mps": rec.get("forward_speed_mps")})
    wall_eps = sum(m["wall_s"] for m in metas.values())
    stop_wall = 0.0
    for f in R.STP.glob("*.json"):
        stop_wall += json.loads(f.read_text()).get("wall_s", 0.0)
    out = {"experiment": "E1 clash reach", "date": C.DATE, "prereg": "design/PREREG_CANDIDATES_2026_11_28.md",
           "evidence": "SIMULATION logs (MuJoCo 3.10.0, rl/trike_env.py through rl/robustness/sweep.py and the "
                       "trot's judged and push episodes); DESK geometry (sim/check_packaging60.py's parts via "
                       "candidate_checks.ClashGeometry)",
           "model": str(C.MODEL.relative_to(C.BUILD)), "model_sha256": C.sha256_file(C.MODEL),
           "controllers": R.CONTROLLERS,
           "seeds": {"policy_blocks": C.E1_BLOCKS, "trot": list(C.E1_TROT_SEEDS)},
           "conditions": {"policies": C.POLICY_CONDITIONS, "trot": list(R.TROT_CONDS)},
           "units": "clearance mm (signed; < 0 overlap), angles deg, shares 0-1",
           "cap_mm": 1000 * C._cp().CAP_M,
           "cap_note": "ClashGeometry reads every clearance up to check_packaging60.CAP_M (50 mm); a value of 50.0 "
                       "means >= 50 mm. The actual least beyond the cap is bracketed in e1_beyond_cap.json "
                       "(supporting, added after a result)",
           "rule": {"REACHED": "a clash is REACHED if at least two of the five controllers have a logged pose with "
                                "clearance < 0 (#9: either side)",
                    "ACCEPTABLE": "a stop is ACCEPTABLE if, for each of the five controllers separately, the least "
                                  "clearance over all its samples with the stop applied is >= 5 mm and at most 1 % "
                                  "of its samples are clipped"},
           "verdict": verdict, "per_controller_judged": per_controller, "reported": reported,
           "falls": {c: {"fell": sum(m["fell"] for m in metas.values() if m["controller"] == c),
                         "unstable": sum(m["unstable"] for m in metas.values() if m["controller"] == c),
                         "episodes": sum(1 for m in metas.values() if m["controller"] == c)} for c in R.CONTROLLERS},
           "episodes": eps,
           "wall_time_s": {"episodes_sum_of_workers": round(wall_eps, 1), "stops_sum_of_workers": round(stop_wall, 1),
                           "summarise": None},
           "provenance": C.provenance(runner=R.RUNNER, extra={"summarise_e1.py": HERE})}
    out["wall_time_s"]["summarise"] = round(time.time() - t0, 1)
    (R.OUT / "e1_summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for fname, rows in (("e1_joint_percentiles.csv", perc_rows), ("e1_stops.csv", stop_rows),
                        ("e1_by_condition.csv", cond_rows)):
        with open(R.OUT / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({"reached": verdict["reached"],
                      "acceptable": {k: v for k, v in verdict["acceptable_stops"].items()}}, indent=1))
    return out
