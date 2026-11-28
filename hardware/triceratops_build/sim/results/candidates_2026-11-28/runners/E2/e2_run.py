"""E2 runner: transmission and speed cap (design/PREREG_CANDIDATES_2026_11_28.md section 4 E2, section 3 items 3-6,
13, 14). Everything numeric comes from sim/candidate_checks.py (frozen, sha256 008aa308debdf1b6...); this file only
schedules episodes, stores them and tabulates the pre-registered rules. SIMULATION (MuJoCo 3.10.0 episodes of the
build's env) and DESK (the build's thermal model and arithmetic on simulated torques); nothing is measured.

Subcommands (run from anywhere; at most 2 worker processes):
  prepare            write the E2(b) start variants' models (scratch) and list them
  harness            harness check only (not a result): every start variant, policy fore128-s0, seed 12345 (not an
                     evaluation seed), a 3 s episode; checks the delay, the logger and the cap run; nothing judged
  run --stage S      S = e1eq (B1 walk 0.25 / push left / push right and the E1 episodes stand and walk 0.15 at 1 ms,
                     which E2(a) needs), demo (new demo courses at seed0 710000), start (B025, rod, belt, cable, R,
                     belt_1span at 0.25 ms), fallback (item 4(b): only what e2b_judged_runs asks for)
  desk               E2(a) tables and verdicts
  judge              E2(b) comparisons and verdicts
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

BUILD = Path("/home/user/mesozoic-labs/hardware/triceratops_build")
sys.path.insert(0, str(BUILD / "sim"))
import candidate_checks as C  # noqa: E402

RUNNER = Path(__file__).resolve()
OUT = C.CAND_SCRATCH / "E2"
EP = OUT / "episodes"
VAR = OUT / "variants"
DEMO = OUT / "demo"
HARN = OUT / "harness_check"
RES = C.RESULTS / "E2"
WORKERS = 2


def jdump(x):
    return json.dumps(x, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"


def run_id(variant: str, dt: float) -> str:
    return f"{variant}_dt{dt:g}"


# ---------------------------------------------------------------------------------------------------- extra hooks
class Deflection:
    """Largest |q_rotor - q_leg| of each series spring over the episode (rad, joint side); a measurement only."""

    def __init__(self, env):
        m = env.model
        self.pairs = np.array(env.sync_pairs, dtype=np.int64).reshape(-1, 2)
        adr = {int(m.jnt_qposadr[j]): m.joint(j).name for j in range(m.njnt)}
        self.names = [adr[int(a)] for a in self.pairs[:, 0]]
        self.max = np.zeros(len(self.pairs))

    def on_reset(self, env):
        self.max[:] = 0.0

    def __call__(self, env):
        q = env.data.qpos
        np.maximum(self.max, np.abs(q[self.pairs[:, 0]] - q[self.pairs[:, 1]]), out=self.max)


# ---------------------------------------------------------------------------------------------------- one task
def _init_worker():
    import torch
    torch.set_num_threads(1)


def _ep_paths(rid, policy, condition, seed, root=EP):
    d = root / rid / policy
    return d / f"{condition}__{seed}.json", d / f"{condition}__{seed}.npz"


def episode_task(task):
    """(rid, variant, model_path, dt, policy, condition, seeds[, root, seconds]) -> list of written paths."""
    rid, variant, model_path, dt, policy, condition, seeds = task[:7]
    root = Path(task[7]) if len(task) > 7 else EP
    seconds = task[8] if len(task) > 8 else None
    todo = [s for s in seeds if not _ep_paths(rid, policy, condition, s, root)[0].exists()]
    if not todo:
        return []
    ec = C.run_env_config(policy)
    cfg = C.policy_env_config(ec, model_path, dt)
    if seconds is not None:
        cfg = cfg.replace(episode_seconds=float(seconds))
    env = C.make_env(cfg)
    stats, pol = C.load_policy(policy, env)
    spec = C.E2B_VARIANTS[variant]
    r, eta = C.variant_r_eta(variant)
    names = C.output_joint_names(env.model)
    written = []
    for s in todo:
        lg = C.StepLogger(env, joint_names=names, r=r or None, eta=eta or None)
        hooks = [lg]
        cap = C.SpeedCap(env, r) if spec["cap"] else None
        if cap is not None:
            hooks.append(cap)
        dfl = Deflection(env) if env.sync_pairs else None
        if dfl is not None:
            hooks.append(dfl)
        t0 = time.time()
        rec = C.policy_episode(env, s, condition, policy=pol, stats=stats, hooks=hooks)
        wall = time.time() - t0
        res = lg.result()
        row = {"run": rid, "variant": variant, "model": str(model_path), "policy": policy, "condition": condition,
               "seed": int(s), "fell": bool(rec["fell"]), "termination": rec["termination"],
               "unstable": bool(rec["unstable"]), "speed": rec.get("forward_speed_mps"), "t_end_s": rec["t_end_s"],
               "delay_steps": rec["delay_steps"], "delay_s": rec["delay_s"], "physics_dt": rec["physics_dt"],
               "control_dt": rec["control_dt"], "window_samples": int(res["window_samples"]),
               "q_nonfinite_dropped": res["q_nonfinite_dropped"], "actuators": res["actuators"],
               "r": res["r"].tolist(), "eta": res["eta"].tolist(), "drive_sq": res["drive_sq"].tolist(),
               "sq_motor": res["sq_motor"].tolist(), "sq_brake": res["sq_brake"].tolist(),
               "rpm_peak": res["rpm_peak"].tolist(), "rpm_bin": res["rpm_bin"],
               "at_cap_share": (cap.at_cap / max(cap.steps, 1)).tolist() if cap else None,
               "cap_actuators": cap.names if cap else None, "cap_steps": cap.steps if cap else None,
               "deflection_max_rad": dict(zip(dfl.names, dfl.max.tolist())) if dfl else None,
               "wall_s": wall, "sweep_record": rec}
        jp, npz = _ep_paths(rid, policy, condition, s, root)
        jp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, rpm_hist=res["rpm_hist"], q=res["q"], t=res["t"])
        tmp = jp.with_suffix(".tmp")
        tmp.write_text(jdump(row))
        tmp.replace(jp)
        written.append(str(jp))
        print(f"[{time.strftime('%H:%M:%S')}] {rid} {policy} {condition} {s}: fell={row['fell']} "
              f"term={row['termination']!r} v={row['speed']} wall={wall:.1f}s", flush=True)
    return written


def demo_task(task):
    policy, seed0 = task
    out = DEMO / policy / f"demo_course_seed{seed0}.json"
    if out.exists():
        return str(out)
    import demo_course
    make, label = C.demo_make(policy)
    t0 = time.time()
    res = demo_course.run(make, label, [("walk_0.25", 90.0), ("stand", 30.0)], None, seed0)
    res["_wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] demo {policy} seed0 {seed0}: valid={res['valid']} "
          f"minutes={res['minutes']:.2f} wall={res['_wall_s']:.1f}s", flush=True)
    return str(out)


def _dispatch(task):
    kind, payload = task
    return episode_task(payload) if kind == "ep" else demo_task(payload)


def run_pool(tasks):
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS, initializer=_init_worker) as pool:
        for _ in pool.imap_unordered(_dispatch, tasks):
            pass


# ---------------------------------------------------------------------------------------------------- models
def models(variants, dt=None, out=VAR) -> dict:
    """{variant: (path, record)} written by candidate_checks.e2b_model (B1: the adopted model itself)."""
    return {v: C.e2b_model(v, out, dt) for v in variants}


def seeds_all():
    return tuple(C.E1_BLOCKS["A"]) + tuple(C.E1_BLOCKS["B"])


def cmd_prepare(a):
    ms = models([v for v in C.E2B_RUNS_AT_START if v != "B1"])
    for v, (p, rec) in ms.items():
        print(v, p, rec["xml_sha256"][:16], rec.get("physics_dt"))


def cmd_harness(a):
    ms = models(C.E2B_RUNS_AT_START)
    tasks = [("ep", (run_id(v, C.E2B_VARIANTS[v]["physics_dt"]), v, str(p), C.E2B_VARIANTS[v]["physics_dt"],
                     "fore128-s0", "push_right_0.5", (12345,), str(HARN), 3.0)) for v, (p, _) in ms.items()]
    run_pool(tasks)
    for v in ms:
        rid = run_id(v, C.E2B_VARIANTS[v]["physics_dt"])
        row = json.loads(_ep_paths(rid, "fore128-s0", "push_right_0.5", 12345, HARN)[0].read_text())
        print(v, {k: row[k] for k in ("fell", "termination", "delay_steps", "delay_s", "physics_dt", "control_dt",
                                       "window_samples", "wall_s")},
              "deflection", None if row["deflection_max_rad"] is None else
              max(row["deflection_max_rad"].values()), "cap_share",
              None if row["at_cap_share"] is None else max(row["at_cap_share"]))


def cmd_run(a):
    tasks = []
    if a.stage == "e1eq":
        p, _ = C.e2b_model("B1", VAR)
        for pol in C.E1_POLICIES:
            for cond in C.E2B_CONDITIONS + ("stand", "walk_0.15"):
                tasks.append(("ep", (run_id("B1", 0.001), "B1", str(p), 0.001, pol, cond, seeds_all())))
    elif a.stage == "demo":
        tasks = [("demo", (pol, 710000)) for pol in C.E1_POLICIES]
    elif a.stage == "start":
        ms = models([v for v in C.E2B_RUNS_AT_START if v != "B1"])
        order = a.only.split(",") if a.only else ["B025", "rod", "belt", "cable", "R", "belt_1span"]
        for v in order:
            p, _ = ms[v]
            dt = C.E2B_VARIANTS[v]["physics_dt"]
            for pol in C.E1_POLICIES:
                for cond in C.E2B_CONDITIONS:
                    tasks.append(("ep", (run_id(v, dt), v, str(p), dt, pol, cond, seeds_all())))
    elif a.stage == "fallback":
        jr = C.e2b_judged_runs(unstable_runs_at_start())
        need = [v for v, s in jr.items() if not v.startswith("_") and s["rerun_at_fallback"]]
        if jr["_B0125_needed"]:
            need = ["B0125"] + need
        print("item 4(b): rerun at 0.125 ms:", need or "nothing")
        for v in need:
            p, _ = C.e2b_model(v, VAR, C.FALLBACK_DT)
            for pol in C.E1_POLICIES:
                for cond in C.E2B_CONDITIONS:
                    tasks.append(("ep", (run_id(v, C.FALLBACK_DT), v, str(p), C.FALLBACK_DT, pol, cond,
                                         seeds_all())))
    else:
        raise SystemExit(a.stage)
    t0 = time.time()
    run_pool(tasks)
    print(f"stage {a.stage}: {len(tasks)} tasks, wall {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------------- loading
def load_ep(rid, policy, condition, seed, with_npz=False):
    jp, npz = _ep_paths(rid, policy, condition, seed)
    row = json.loads(jp.read_text())
    if with_npz:
        z = np.load(npz)
        row["rpm_hist"] = z["rpm_hist"]
    return row


def unstable_runs_at_start() -> set:
    bad = set()
    for v in C.E2B_RUNS_AT_START:
        rid = run_id(v, C.E2B_VARIANTS[v]["physics_dt"])
        for pol in C.E1_POLICIES:
            for cond in C.E2B_CONDITIONS:
                for s in seeds_all():
                    jp, _ = _ep_paths(rid, pol, cond, s)
                    if not jp.exists():
                        raise FileNotFoundError(f"start run incomplete: {jp}")
                    if json.loads(jp.read_text())["unstable"]:
                        bad.add(v)
    return bad


# ---------------------------------------------------------------------------------------------------- E2(a) desk
def cmd_desk(a):
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b1 = run_id("B1", 0.001)
    grid_rows, per = [], {}
    speed = {}
    for pol in C.E1_POLICIES:
        for blk, seeds in C.E1_BLOCKS.items():
            s0 = seeds[0]
            if blk == "A":
                demo, demo_src = C.stored_demo(pol), str(C.DEMO_DIR / pol / "demo_course.json")
            else:
                p = DEMO / pol / f"demo_course_seed{s0}.json"
                demo, demo_src = json.loads(p.read_text()), str(p)
            assert demo["seed0"] == s0, (pol, blk, demo["seed0"])
            joints = demo["segments"]["stand"]["joints"]
            segs = C.stored_segments(demo)
            ew = load_ep(b1, pol, "walk_0.25", s0)
            es = load_ep(b1, pol, "stand", s0 + 2)
            assert ew["actuators"] == joints and es["actuators"] == joints
            split = {"walk_0.25": (np.array(ew["sq_motor"]), np.array(ew["sq_brake"])),
                     "stand": (np.array(es["sq_motor"]), np.array(es["sq_brake"]))}
            prof = C.reduced_profile(segs, joints, "both", 1.0, 0.95, split)
            rows = C.e2a_grid(demo, segs, split)
            base = C.heat_rules(demo, segs["walk_0.25"], segs["stand"], demo["minutes"])
            per[(pol, blk)] = {"demo": demo_src, "demo_valid": demo["valid"], "minutes": demo["minutes"],
                               "split_from": {"walk_0.25": f"{b1} walk_0.25 seed {s0}",
                                              "stand": f"{b1} stand seed {s0 + 2}"},
                               "method": prof["method"],
                               "reproduction_rel_max": {k: prof[f"{k}_reproduction_rel_max"]
                                                        for k in ("walk_0.25", "stand")},
                               "today": {k: base[k] for k in ("C1_c", "C1_joint", "C1_pass", "C2", "C2_joint",
                                                              "C2_pass", "C1_by_tc_c")}}
            # speed check: walk 0.25 and stand pooled over the block's four episodes, weighted 0.75 / 0.25
            hists, pk = {}, []
            for cond in ("walk_0.25", "stand", "walk_0.15"):
                hs = []
                for s in seeds:
                    e = load_ep(b1, pol, cond, s, with_npz=True)
                    assert e["actuators"] == joints and not e["unstable"]
                    hs.append(e["rpm_hist"])
                    if cond != "walk_0.15":
                        pk.append(np.array(e["rpm_peak"]))
                    else:
                        hists.setdefault("_pk15", []).append(np.array(e["rpm_peak"]))
                    per[(pol, blk)].setdefault("falls_in_speed_episodes", []).extend(
                        [f"{cond} {s}"] if e["fell"] else [])
                hists[cond] = np.sum(hs, axis=0)
            bin_ = float(e["rpm_bin"])
            for r in C.REDUCTIONS:
                for pl in C.PLACEMENTS:
                    sc = C.e2a_speed_check([hists["walk_0.25"], hists["stand"]], [0.75, 0.25], pk, bin_, joints,
                                           pl, r)
                    s15 = C.e2a_speed_check([hists["walk_0.15"]], [1.0], hists["_pk15"], bin_, joints, pl, r)
                    speed[(pol, blk, r, pl)] = {"judged": sc, "walk_0.15_beside": s15}
            for row in rows:
                sc = speed[(pol, blk, row["r"], row["placement"])]["judged"]
                row.update(policy=pol, block=blk, seed0=s0, speed_pass=sc["pass"],
                           speed_p99_max_rpm=max(sc["p99_rpm"]), speed_peak_max_rpm=max(sc["peak_rpm"]),
                           speed_worst_p99_drive=sc["drives"][int(np.argmax(sc["p99_rpm"]))],
                           speed_worst_peak_drive=sc["drives"][int(np.argmax(sc["peak_rpm"]))],
                           all_pass=bool(row["C1_pass"] and row["C2_pass"] and sc["pass"]))
                grid_rows.append(row)
    # verdicts per judged (transmission, r, placement): the worst of the four policies on both blocks
    verdicts = {}
    keys = sorted({(r["transmission"], r["r"], r["placement"]) for r in grid_rows},
                  key=lambda k: (k[0], k[2], k[1]))
    for k in keys:
        rs = [r for r in grid_rows if (r["transmission"], r["r"], r["placement"]) == k]
        assert len(rs) == 8, (k, len(rs))
        judged = rs[0]["judged"]
        v = {"transmission": k[0], "r": k[1], "placement": k[2], "judged": judged, "eta": rs[0]["eta"],
             "C1_worst_c": max(r["C1_c"] for r in rs), "C1_worst_at": max(rs, key=lambda r: r["C1_c"])["policy"] + " "
             + max(rs, key=lambda r: r["C1_c"])["block"],
             "C2_worst": max(r["C2"] for r in rs), "C2_worst_at": max(rs, key=lambda r: r["C2"])["policy"] + " "
             + max(rs, key=lambda r: r["C2"])["block"],
             "C1_all_pass": all(r["C1_pass"] for r in rs), "C2_all_pass": all(r["C2_pass"] for r in rs),
             "speed_all_pass": all(r["speed_pass"] for r in rs),
             "speed_p99_worst_rpm": max(r["speed_p99_max_rpm"] for r in rs),
             "speed_peak_worst_rpm": max(r["speed_peak_max_rpm"] for r in rs),
             "methods": sorted({r["method"] for r in rs}),
             "per_policy_all_pass": {p: all(r["all_pass"] for r in rs if r["policy"] == p) for p in C.E1_POLICIES},
             "per_policy_heat_pass": {p: all(r["C1_pass"] and r["C2_pass"] for r in rs if r["policy"] == p)
                                      for p in C.E1_POLICIES}}
        v["PASSES"] = bool(v["C1_all_pass"] and v["C2_all_pass"] and v["speed_all_pass"]) if judged else None
        verdicts["|".join(map(str, k))] = v
    # outputs
    cols = ["transmission", "kind", "eta", "judged", "r", "placement", "policy", "block", "seed0", "method", "C1_c",
            "C1_joint", "C1_pass", "C2", "C2_joint", "C2_pass", "C1_c_low", "C2_low", "speed_p99_max_rpm",
            "speed_worst_p99_drive", "speed_peak_max_rpm", "speed_worst_peak_drive", "speed_pass", "all_pass",
            "C1_4min_c", "C1_15min_c"]
    with (RES / "e2a_grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in grid_rows:
            tc = r["C1_by_tc_c"]
            r2 = dict(r)
            r2["C1_4min_c"] = tc["tau_240s"]
            r2["C1_15min_c"] = tc["tau_900s"]
            for c in ("C1_c", "C2", "C1_c_low", "C2_low", "C1_4min_c", "C1_15min_c", "speed_p99_max_rpm",
                      "speed_peak_max_rpm"):
                if r2.get(c) is not None:
                    r2[c] = round(float(r2[c]), 4)
            w.writerow(r2)
    sp_rows = []
    for (pol, blk, r, pl), v in speed.items():
        sp_rows.append({"policy": pol, "block": blk, "r": r, "placement": pl, **{
            k: v["judged"][k] for k in ("drives", "p99_rpm", "peak_rpm", "pass")},
            "walk_0.15_p99_rpm": v["walk_0.15_beside"]["p99_rpm"],
            "walk_0.15_peak_rpm": v["walk_0.15_beside"]["peak_rpm"], "walk_0.15_pass": v["walk_0.15_beside"]["pass"]})
    summary = {"experiment": "E2(a) transmission and speed cap, DESK on SIMULATED torques",
               "prereg": "design/PREREG_CANDIDATES_2026_11_28.md section 4 E2(a); section 3 items 3, 6, 13, 14",
               "rule": "PASSES if C1 (hottest winding <= 90 C at 35 C, 8 min), C2 (worst drive loop RMS / continuous "
                       "<= 0.80) and the speed check (each reduced drive's p99 <= 83 rpm and peak <= 91.3 rpm, "
                       "walk 0.25 and stand weighted 0.75 / 0.25) all pass for the worst of the four fore128 policies "
                       "on both seed blocks",
               "verdicts": verdicts, "per_policy_block": {f"{p}|{b}": v for (p, b), v in per.items()},
               "speed_checks": sp_rows,
               "seeds": {"blocks": C.E1_BLOCKS, "demo_seed0": {"A": 700000, "B": 710000}},
               "wall_s": time.time() - t0, "provenance": C.provenance(RUNNER)}
    (RES / "e2a_summary.json").write_text(jdump(summary))
    print("E2(a) judged verdicts:")
    for k, v in verdicts.items():
        if v["judged"]:
            print(f"  {k}: PASSES={v['PASSES']} C1 {v['C1_worst_c']:.1f} C ({v['C1_worst_at']}) C2 {v['C2_worst']:.3f} "
                  f"({v['C2_worst_at']}) speed p99 {v['speed_p99_worst_rpm']:.1f} peak {v['speed_peak_worst_rpm']:.1f}"
                  f" rpm; methods {v['methods']}")
    for (p, b), v in per.items():
        print(f"  {p} {b}: method {v['method']} repro {v['reproduction_rel_max']} today C1 {v['today']['C1_c']:.1f} "
              f"C2 {v['today']['C2']:.3f} falls {v['falls_in_speed_episodes']}")


# ---------------------------------------------------------------------------------------------------- E2(b) judge
def rows_of(rid, policy, condition, eta_override=None):
    out = {}
    for s in seeds_all():
        e = load_ep(rid, policy, condition, s)
        sq = np.array(e["drive_sq"])
        if eta_override:
            sq = C.relog_drive_sq({"actuators": e["actuators"], "eta": np.array(e["eta"]), "r": np.array(e["r"]),
                                   "sq_motor": np.array(e["sq_motor"]), "sq_brake": np.array(e["sq_brake"])},
                                  eta_override)
        out[s] = {"fell": e["fell"], "termination": e["termination"], "speed": e["speed"], "drive_sq": sq,
                  "samples": e["window_samples"], "actuators": e["actuators"], "_ep": e}
    return out


def _clean(cmp):
    return {k: v for k, v in cmp.items()}


def judge_variant(v, dt, baselines, eta_override=None):
    rid = run_id(v, dt)
    per_policy, detail = {}, {}
    for pol in C.E1_POLICIES:
        per_policy[pol] = {}
        for b in baselines:
            bdt = C.E2B_VARIANTS[b]["physics_dt"]
            per_policy[pol][b] = {}
            for cond in C.E2B_CONDITIONS:
                cmp = C.e2b_compare(rows_of(rid, pol, cond, eta_override), rows_of(run_id(b, bdt), pol, cond))
                per_policy[pol][b][cond] = cmp
    verdict = C.e2b_verdict(per_policy)
    return per_policy, verdict


def run_stats(rid, policy, condition):
    eps = [load_ep(rid, policy, condition, s, with_npz=True) for s in seeds_all()]
    fell = [e["seed"] for e in eps if e["fell"]]
    uns = [e["seed"] for e in eps if e["unstable"]]
    out = {"falls": fell, "unstable": uns, "terminations": {e["seed"]: e["termination"] for e in eps if e["fell"]},
           "speed_mean_all": float(np.mean([e["speed"] for e in eps if e["speed"] is not None])),
           "delay_ok": all(abs(e["delay_s"] - 0.010) < 1e-12 for e in eps),
           "delay_steps": sorted({e["delay_steps"] for e in eps}), "physics_dt": sorted({e["physics_dt"] for e in eps}),
           "wall_s_mean": float(np.mean([e["wall_s"] for e in eps]))}
    acts = eps[0]["actuators"]
    if eps[0]["at_cap_share"] is not None:
        sh = np.array([e["at_cap_share"] for e in eps])
        out["at_cap_share_max"] = float(sh.max())
        out["at_cap_share_mean_by_drive"] = dict(zip(eps[0]["cap_actuators"], sh.mean(0).round(6).tolist()))
    if eps[0]["deflection_max_rad"] is not None:
        d = {k: max(e["deflection_max_rad"][k] for e in eps) for k in eps[0]["deflection_max_rad"]}
        out["deflection_max_mrad"] = {k: round(1000 * x, 4) for k, x in d.items()}
    # drive speed (drive side, r applied), pooled over the 8 episodes: p99 and peak per drive
    h = np.sum([e["rpm_hist"] for e in eps], axis=0)
    p99 = C.weighted_percentile_from_hist([h], [1.0], float(eps[0]["rpm_bin"]))
    pk = np.max([e["rpm_peak"] for e in eps], axis=0)
    out["drive_rpm_p99"] = dict(zip(acts, np.round(p99, 2).tolist()))
    out["drive_rpm_peak"] = dict(zip(acts, np.round(pk, 2).tolist()))
    return out


def cmd_judge(a):
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bad = unstable_runs_at_start()
    jr = C.e2b_judged_runs(bad)
    out = {"experiment": "E2(b) transmission and speed cap, SIMULATION",
           "prereg": "design/PREREG_CANDIDATES_2026_11_28.md section 4 E2(b); section 3 items 4-6",
           "rule": "COMPATIBLE WITH TODAY'S POLICIES if at least 3 of the 4 fore128 policies pass every condition "
                   "(walk 0.25, push left 0.5, push right 0.5) against both judged baselines on the same seeds: no "
                   "new falls, |mean speed - baseline| <= 0.02 m/s, worst-drive RMS / continuous (drive side) - "
                   "baseline's <= 0.03; NOT COMPATIBLE once 2 or more fail; NOT JUDGED otherwise",
           "unstable_runs_at_start": sorted(bad), "judged_runs": jr, "variants": {}, "run_stats": {},
           "models": {}}
    for v in list(C.E2B_RUNS_AT_START) + (["B0125"] if jr["_B0125_needed"] else []):
        dt = C.E2B_VARIANTS[v]["physics_dt"]
        p, rec = C.e2b_model(v, VAR, dt) if v != "B1" else C.e2b_model("B1", VAR)
        out["models"][run_id(v, dt)] = {"path": str(p), "xml_sha256": rec["xml_sha256"]}
    for v, spec in jr.items():
        if v.startswith("_"):
            continue
        if spec["rerun_at_fallback"]:
            p, rec = C.e2b_model(v, VAR, C.FALLBACK_DT)
            out["models"][run_id(v, C.FALLBACK_DT)] = {"path": str(p), "xml_sha256": rec["xml_sha256"]}
    runs = sorted({k for k in out["models"]})
    for rid in runs:
        out["run_stats"][rid] = {pol: {cond: run_stats(rid, pol, cond) for cond in C.E2B_CONDITIONS}
                                 for pol in C.E1_POLICIES}
    for v, spec in jr.items():
        if v.startswith("_"):
            continue
        dt = spec["physics_dt"]
        per_policy, verdict = judge_variant(v, dt, spec["baselines"])
        entry = {"role": spec["role"], "physics_dt": dt, "baselines": list(spec["baselines"]),
                 "verdict": verdict, "comparisons": per_policy}
        if spec["rerun_at_fallback"]:      # the 0.25 ms run beside, supporting
            pp, vv = judge_variant(v, C.SERIES_DT, ("B1", "B025"))
            entry["supporting_0.25ms_run"] = {"verdict": vv, "comparisons": pp}
        if v == "rod":                     # supporting: the same episodes re-mapped to the drive at 0.90
            red = {j: 0.90 for j in C.reduced_joints("both")}
            pp = {}
            for pol in C.E1_POLICIES:
                pp[pol] = {}
                for b in spec["baselines"]:
                    bdt = C.E2B_VARIANTS[b]["physics_dt"]
                    pp[pol][b] = {cond: C.e2b_compare(rows_of(run_id(v, dt), pol, cond, red),
                                                      rows_of(run_id(b, bdt), pol, cond))
                                  for cond in C.E2B_CONDITIONS}
            entry["supporting_rod_eta0.9"] = {"verdict": C.e2b_verdict(pp), "comparisons": pp}
        out["variants"][v] = entry
    out["wall_s"] = time.time() - t0
    out["seeds"] = {"blocks": C.E1_BLOCKS, "conditions": list(C.E2B_CONDITIONS)}
    out["provenance"] = C.provenance(RUNNER, {f"model:{k}": v["path"] for k, v in out["models"].items()
                                              if Path(v["path"]) != C.MODEL})
    (RES / "e2b_summary.json").write_text(jdump(out))
    # compact CSV: one row per variant, policy, baseline, condition
    with (RES / "e2b_comparisons.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "role", "physics_dt", "reading", "policy", "baseline", "condition", "pass",
                    "new_falls", "unstable", "paired_up", "speed_variant", "speed_baseline", "d_speed",
                    "worst_ratio_variant", "worst_joint_variant", "worst_ratio_baseline", "d_ratio"])
        for v, e in out["variants"].items():
            readings = [("judged" if e["role"] == "judged" else "supporting", e["physics_dt"], e["comparisons"])]
            if "supporting_0.25ms_run" in e:
                readings.append(("supporting 0.25 ms run", C.SERIES_DT, e["supporting_0.25ms_run"]["comparisons"]))
            if "supporting_rod_eta0.9" in e:
                readings.append(("supporting eta 0.90", e["physics_dt"], e["supporting_rod_eta0.9"]["comparisons"]))
            for name, dt, comp in readings:
                for pol, bb in comp.items():
                    for b, cc in bb.items():
                        for cond, c in cc.items():
                            sv, sb = c.get("speed_variant"), c.get("speed_baseline")
                            rv, rb = c.get("worst_ratio_variant"), c.get("worst_ratio_baseline")
                            w.writerow([v, e["role"], dt, name, pol, b, cond, c["pass"], len(c["new_falls"]),
                                        len(c["unstable"]), len(c["paired_up"]),
                                        None if sv is None else round(sv, 4), None if sb is None else round(sb, 4),
                                        None if sv is None else round(sv - sb, 4),
                                        None if rv is None else round(rv, 4), c.get("worst_joint_variant"),
                                        None if rb is None else round(rb, 4),
                                        None if rv is None else round(rv - rb, 4)])
    print("unstable runs at start:", sorted(bad))
    for v, e in out["variants"].items():
        print(f"{v} ({e['role']}, {e['physics_dt']:g} s vs {e['baselines']}): COMPATIBLE={e['verdict']['COMPATIBLE']} "
              f"per policy {e['verdict']['per_policy']}")
        for extra in ("supporting_0.25ms_run", "supporting_rod_eta0.9"):
            if extra in e:
                print(f"   {extra}: COMPATIBLE={e[extra]['verdict']['COMPATIBLE']} {e[extra]['verdict']['per_policy']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("harness")
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, choices=["e1eq", "demo", "start", "fallback"])
    p.add_argument("--only", default=None)
    sub.add_parser("desk")
    sub.add_parser("judge")
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "harness": cmd_harness, "run": cmd_run, "desk": cmd_desk, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    main()
