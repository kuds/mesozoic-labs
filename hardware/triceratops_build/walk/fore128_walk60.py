"""The frozen model-based trot on the adopted 128 mm capsule model (D25, D41). Simulation only; reported, not judged.

The controller, its judge and its sweep are used as they are; nothing in ``walk60.py``, ``judge_walk60.py``,
``sweep_walk60.py`` or ``rl/`` changes. This file swaps the robot model and adds, as parameters of this driver whose
defaults are the existing tools' own, the measures and points those tools lack:

- ``judged``: the conditions of ``PREREG_WALK60.md`` (``judge_walk60.JUDGED`` J1-J4 and ``REPORTED`` R1-R5) on the
  chosen model, plus J1 at extra servo delays (``--extra-delays``, none by default). Each episode runs exactly as
  ``walk60.run_episode`` runs it (same env, controller, ``gait_metrics.record_episode`` and ``measures``) and also
  records each drive's peak speed. J1-J4 are run a second time with the sole's slip taken at its contacts
  (``slip_at_contact=True``, as the foot study measured the capsule sole, ``sim/feet60_study.py``); the second run's
  trajectory is the first's (checked: same speeds).
- ``sweep``: ``sweep_walk60``'s sweep (``rl/robustness/sweep.py``'s points, physics changes, push, measures, break
  rule; the controller built in each worker by ``sweep_walk60._init_worker``) on the chosen model, with
  ``--delays`` (the delay families' points; default the sweep's own), ``--push-axis`` (``sideways``, the sweep's own,
  or ``fore_aft``: the same push along the chassis's forward axis at the onset, + forward, - back), ``--seed0`` (the
  first episode seed; default the sweep's 200000) and peak drive speed added to each episode record. Terrain is not
  run here (the demo is on flat ground, D28).
- ``summarise``: the small committed summary (summary.json, RESULTS.md) from the raw files; ``--slow-gait DIR`` adds,
  beside the sweep rows it measured (the push rows, and the rows of a point setting such as kp x0.8) and the judged
  conditions, the 10-20 s windows that ``fore128_slow_gait.py`` measured on the same episodes, checked seed by seed
  (the sweep's speeds are 2-20 s means and its push comes at 8 s, so they dilute a slower gait after it).

Drive speed is the joint speed (MuJoCo ``actuator_velocity``; gear 1, rl/trike_env.py), which is the drive's own
speed on this build: every leg drive of option C is at 1:1 (``sim/build_model60.py`` ``AXIS_DRIVE``).

Usage (from the repository root):
  python hardware/triceratops_build/walk/fore128_walk60.py judged --seeds 101-110 --out DIR [--extra-delays 16 17]
  python hardware/triceratops_build/walk/fore128_walk60.py sweep --seed0 200000 --seeds 10 --out DIR
      [--delays 0 2 5 10 15 16 17 18 19 20 30 40 60] [--push-axis fore_aft] [--sweeps delay push ...]
  python hardware/triceratops_build/walk/fore128_walk60.py summarise --raw DIR --out RESULTS_DIR [--slow-gait DIR]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL = HERE.parent / "rl"
for _p in (HERE, RL / "robustness", RL / "analysis", RL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sweep_walk60 as SW  # noqa: E402  (imports rl/robustness/sweep.py as SW.S and walk60)
import judge_walk60 as JW  # noqa: E402
import gait_metrics  # noqa: E402

S = SW.S
walk60 = SW.walk60
G = S.G

MODEL = "trike60_C_closed_capsule_fore128.xml"  # the adopted design's model (D25, D41), in sim/generated/
GEN = HERE.parent / "sim" / "generated"
# The budget's drive-speed allowances (desk), rpm, as rl/analysis/gait_pipe1.py RPM_ALLOW holds them: the X4's 83 rpm
# rated-speed cap above 24 V (DECISION_LOG.md E1, reseller text quoting the maker); the X6 elbow 0.70 x its 176 rpm
# no-load speed at 48 V scaled to 42 V, 176 x 42 / 48 x 0.70 = 108 (budget/size_budget.py DRIVES; sim/CRAWL_REPORT.md
# "speed = 0.70 x no-load at 42 V").
RPM_ALLOW = dict(G.RPM_ALLOW)
ALLOW_SOURCE = ("X4 83 rpm: DECISION_LOG.md E1 (rated-speed cap above 24 V); X6 elbow 108 rpm: 0.70 x 176 rpm no-load "
                "at 48 V x 42/48 (budget/size_budget.py DRIVES, sim/CRAWL_REPORT.md); both as rl/analysis/gait_pipe1.py "
                "RPM_ALLOW holds them")
RECOVERY_NOTE_FORE_AFT = ("not defined for fore-aft pushes: the sweep counts an episode as recovered when the 0.5 s "
                          "mean speed along the push stays within 0.05 m/s for 1 s (rl/robustness/sweep.py "
                          "RECOVER_VLAT), and along the forward axis the robot walks at 0.25 m/s")
_ORIG_RECORD = S.episode_record
_ORIG_LATERAL = S.lateral_axis


# ---------------------------------------------------------------- shared
def rpm_stats(t: np.ndarray, omega: np.ndarray, window_s: float = G.WINDOW_S) -> dict | None:
    """Per drive (columns of ``omega``, rad/s): the peak and the 99th percentile of |speed| in rpm over t >= window_s;
    None when the window has fewer than 100 samples (as the sweep's measures)."""
    w = np.asarray(t) >= window_s
    if w.sum() < 100:
        return None
    om = np.abs(np.asarray(omega)[w]) * 60.0 / (2.0 * math.pi)
    return {"peak_rpm": [round(float(x), 2) for x in om.max(0)],
            "p99_rpm": [round(float(x), 2) for x in np.percentile(om, 99, axis=0)]}


def group_peaks(names: list, peaks: list, p99: list) -> dict:
    """Per drive group (roll, pitch, front elbow, hind knee): the largest peak speed over its joints (rpm) and that
    joint, and the largest 99th-percentile speed (rpm)."""
    out = {"peak_rpm": {}, "peak_joint": {}, "p99_rpm": {}}
    for g in RPM_ALLOW:
        cols = [j for j, nm in enumerate(names) if G.group_of(nm) == g]
        j = max(cols, key=lambda c: peaks[c])
        out["peak_rpm"][g], out["peak_joint"][g] = peaks[j], names[j]
        out["p99_rpm"][g] = max(p99[c] for c in cols)
    return out


def git_state() -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(HERE), str(RL)], capture_output=True,
                                text=True, cwd=HERE).stdout.strip())
    return head, dirty


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seeds_arg(text: str) -> list[int]:
    a, b = text.split("-") if "-" in text else (text, text)
    return list(range(int(a), int(b) + 1))


def write_gz(path: Path, obj) -> None:
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write((json.dumps(obj, default=str, separators=(",", ":")) + "\n").encode())


def read_gz(path: Path):
    return json.loads(gzip.decompress(Path(path).read_bytes()).decode())


# ---------------------------------------------------------------- judged conditions
def judged_conditions(extra_delays=()) -> dict:
    """name: (label, plant overrides, impratio, command, delay ms): judge_walk60's judged (J1-J4, 0.25 m/s, 10 ms) and
    reported (R1-R5) conditions, then J1 at each extra delay."""
    out = {n: (lab, plant, imp, JW.COMMAND, 10) for n, (lab, plant, imp) in JW.JUDGED.items()}
    out.update({n: v for n, v in JW.REPORTED.items()})
    lab, plant, imp = JW.JUDGED["J1"]
    for d in extra_delays:
        out[f"J1d{int(d)}"] = (f"J1 at {int(d)} ms delay", plant, imp, JW.COMMAND, int(d))
    return out


def judged_episode(job) -> tuple:
    """One episode, as ``walk60.run_episode`` runs it (the same lines), plus each drive's peak speed and the slip
    option. ``job``: (name, seed, plant overrides incl. model_path, impratio, command, delay, slip_at_contact)."""
    name, seed, plant, impratio, command, delay, slip_at_contact = job
    env = walk60.TrikeEnv(walk60.plant_config(**plant))
    if impratio is not None:
        env.model.opt.impratio = float(impratio)
    ctrl = walk60.TrotController(env, walk60.WalkParams())
    log = gait_metrics.record_episode(env, ctrl.act, seed=seed, command=command, delay_steps=delay,
                                      slip_at_contact=slip_at_contact)
    ms = gait_metrics.measures(log, command)
    r = {"seed": seed, "command_mps": command, "delay_steps": delay, "fell": bool(log["fell"]),
         "termination": log["termination"], "length_s": float(log["time_s"]),
         "speed_mps": float(log["episode_forward_speed"]), "ik_err_max_m": ctrl.ik_err_max,
         "friction": env.factors.get("friction") if env.factors else None, "measures": ms}
    r["drive_speed"] = rpm_stats(log["t"], log["omega"])
    r["names"] = list(log["names"])
    return name, bool(slip_at_contact), r


def run_judged(model: str, seeds: list[int], extra_delays, workers: int, out: Path) -> dict:
    conds = judged_conditions(extra_delays)
    mpath = str(GEN / model)
    jobs = [(n, s, {**plant, "model_path": mpath}, imp, cmd, d, False) for n, (_, plant, imp, cmd, d) in conds.items()
            for s in seeds]
    jobs += [(n, s, {**JW.JUDGED[n][1], "model_path": mpath}, JW.JUDGED[n][2], JW.COMMAND, 10, True)
             for n in JW.JUDGED for s in seeds]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        done = pool.map(judged_episode, jobs, chunksize=1)
    by, by_contact = {}, {}
    for name, contact, r in done:
        (by_contact if contact else by).setdefault(name, []).append(r)
    head, dirty = git_state()
    res = {"simulation_only": True, "tool": "walk/fore128_walk60.py judged", "commit": head, "dirty": dirty,
           "model": model, "model_sha256": sha(GEN / model), "walk60_sha256": sha(HERE / "walk60.py"),
           "judge_walk60_sha256": sha(HERE / "judge_walk60.py"), "driver_sha256": sha(Path(__file__)),
           "trike_env_sha256": sha(Path(sys.modules[walk60.TrikeEnv.__module__].__file__)),
           "params": walk60.dataclasses.asdict(walk60.WalkParams()), "seeds": seeds,
           "extra_delays_ms": [int(d) for d in extra_delays], "workers": workers, "wall_s": round(time.time() - t0, 1),
           "conditions": {}}
    for n, (label, plant, imp, cmd, d) in conds.items():
        eps = sorted(by[n], key=lambda e: e["seed"])
        c = {"label": label, "plant": plant, "impratio": imp, "command_mps": cmd, "delay_ms": d, "episodes": eps}
        if n in by_contact:
            ce = sorted(by_contact[n], key=lambda e: e["seed"])
            c["contact_slip_episodes"] = [{"seed": e["seed"], "fell": e["fell"],
                                           "forward_speed_mps": e["measures"]["forward_speed_mps"],
                                           "slip_ratio": e["measures"]["slip_ratio"]} for e in ce]
        res["conditions"][n] = c
    out.mkdir(parents=True, exist_ok=True)
    write_gz(out / f"judged_{seeds[0]}-{seeds[-1]}.json.gz", res)
    return res


def judged_summary(c: dict) -> dict:
    """One condition, one seed set: the measures the report quotes, beside PREREG_WALK60.md's thresholds (written for
    the 188 mm roll-ankle model; shown, not judged)."""
    eps = c["episodes"]
    ms = [e["measures"] for e in eps]
    rep = JW.reported(eps)
    names = eps[0]["names"]
    ok = [e for e in eps if not e["fell"]]
    rms_mean = {nm: float(np.mean([e["measures"]["drives"][nm]["rms_over_continuous"] for e in ok])) for nm in names} \
        if ok else {}
    worst_mean = max(rms_mean.items(), key=lambda kv: kv[1]) if rms_mean else (None, None)
    peaks = None
    if ok and all(e["drive_speed"] for e in ok):
        pk = np.max([e["drive_speed"]["peak_rpm"] for e in ok], axis=0).tolist()
        p99 = np.max([e["drive_speed"]["p99_rpm"] for e in ok], axis=0).tolist()
        peaks = group_peaks(names, pk, p99)
    speeds = [m["forward_speed_mps"] for m in ms]
    feet = ms[0]["slip_ratio"].keys()
    slip = {f: (lambda v: float(np.mean(v)) if v else None)([m["slip_ratio"][f] for m in ms
                                                             if m["slip_ratio"][f] is not None]) for f in feet}
    out = {"label": c["label"], "command_mps": c["command_mps"], "delay_ms": c["delay_ms"], "episodes": len(eps),
           "falls": rep["falls"], "speed_mean_mps": float(np.mean(speeds)), "speed_min_mps": float(np.min(speeds)),
           "heat_worst_episode": {"rms_over_continuous": rep["heat_worst"][0], "joint": rep["heat_worst"][1]},
           "heat_worst_joint_mean": {"rms_over_continuous": worst_mean[1], "joint": worst_mean[0]},
           "at_limit_episode_worst": {"fraction": rep["at_limit_episode_worst"][0],
                                      "joint": rep["at_limit_episode_worst"][1]},
           "at_limit_nominal_worst": {"fraction": rep["at_limit_nominal_worst"][0],
                                      "joint": rep["at_limit_nominal_worst"][1]},
           "heading_change_abs_worst_deg": rep["heading_change_abs_worst_deg"],
           "sideways_drift_abs_worst_m": rep["sideways_drift_abs_worst_m"], "power_w_mean": rep["power_w_mean"],
           "touchdowns_per_s_mean": rep["touchdowns_per_s_mean"], "ik_err_max_m": rep["ik_err_max_m"],
           "slip_ratio_mean_by_foot_at_origin": slip, "drive_speed": peaks}
    cmd = c["command_mps"]
    out["episodes_under_90pct_command"] = sum(v < 0.9 * cmd for v in speeds) if cmd > 0 else None
    out["episodes_worst_drive_at_or_over_1"] = sum(max(d["rms_over_continuous"] for d in m["drives"].values()) >= 1.0
                                                   for m in ms)
    out["hind_touchdowns_per_s_episode_max"] = max(max(m["touchdowns_per_s"]["left_hind"],
                                                       m["touchdowns_per_s"]["right_hind"]) for m in ms)
    if "contact_slip_episodes" in c:
        ce = c["contact_slip_episodes"]
        out["slip_ratio_mean_by_foot_at_contacts"] = {
            f: (lambda v: float(np.mean(v)) if v else None)([e["slip_ratio"][f] for e in ce
                                                             if e["slip_ratio"][f] is not None]) for f in feet}
        out["contact_rerun_same_speeds"] = [e["forward_speed_mps"] for e in ce] == speeds
    if c["command_mps"] == JW.COMMAND:  # the pre-registration's thresholds are for the 0.25 m/s command
        w4 = max(v for v in slip.values() if v is not None)
        out["prereg_188mm_thresholds"] = {
            "W1_no_fall": out["falls"] == 0,
            "W2_mean_at_least_0.225_none_under_0.20": out["speed_mean_mps"] >= JW.W2_MEAN and out["speed_min_mps"] >= JW.W2_MIN,
            "W3_every_episode_under_1.0": rep["heat_worst"][0] < JW.W3_HEAT,
            "W4_slip_at_origin_at_most_0.25": w4 <= JW.W4_SLIP}
        if "slip_ratio_mean_by_foot_at_contacts" in out:
            w4c = max(v for v in out["slip_ratio_mean_by_foot_at_contacts"].values() if v is not None)
            out["prereg_188mm_thresholds"]["W4_slip_at_contacts_at_most_0.25"] = w4c <= JW.W4_SLIP
    return out


# ---------------------------------------------------------------- the sweep
def env_config(model: str | None = None) -> dict:
    """``sweep_walk60.env_config()`` with the robot model swapped (None: walk60's own model, i.e. exactly
    ``sweep_walk60.env_config()``)."""
    ec = walk60.plant_config(**({} if model is None else {"model_path": str(GEN / model)})).to_dict()
    ec["model_path"] = Path(ec["model_path"]).name
    return json.loads(json.dumps(ec, default=list))


def forward_axis(xmat) -> np.ndarray:
    """The body's forward (+x) axis projected on the floor, unit length (``sweep.lateral_axis`` for +x)."""
    x = np.asarray(xmat, dtype=np.float64).reshape(3, 3)[:, 0].copy()
    x[2] = 0.0
    return x / np.linalg.norm(x)


def record_with_speed(arr: dict, p: dict, seed: int, pusher) -> dict:
    """``sweep.episode_record`` plus each drive's peak speed (rpm, from t = 2 s) and, after a push's onset, the body's
    largest roll and pitch."""
    rec = _ORIG_RECORD(arr, p, seed, pusher)
    ds = rpm_stats(arr["t"], arr["omega"])
    if ds:
        rec.update(ds)
    if pusher is not None and pusher.onset is not None:
        t = arr["t"]
        i0 = int(np.searchsorted(t, pusher.onset - pusher.half))
        rp = np.abs(np.asarray(arr["rp"])[i0:])
        rec.setdefault("push", {})["rp_peak_after_onset_deg"] = [round(float(rp[:, 0].max()), 3),
                                                                 round(float(rp[:, 1].max()), 3)]
    return rec


def _init_worker(ec: dict, push_axis: str) -> None:
    """The sweep driver's worker (``sweep_walk60._init_worker``: base env and controller on ``ec``'s model), with the
    peak-speed record and, for ``fore_aft``, the push along the forward axis. ``sideways`` restores the sweep's own
    lateral axis."""
    S.episode_record = record_with_speed
    S.lateral_axis = forward_axis if push_axis == "fore_aft" else _ORIG_LATERAL
    SW._init_worker(ec)


def families(delays=None, push_axis: str = "sideways", sweeps=None) -> list:
    """The sweep's families (``sweep.FAMILIES``) without terrain. ``delays``: the delay families' points (None: the
    sweep's own). ``push_axis`` "fore_aft": the push family only, relabelled (its points are the sideways family's,
    applied along the forward axis by the worker)."""
    out = []
    for sweep, fam, xk, xs, xnom, fixed, extra in S.FAMILIES:
        if sweep == "terrain" or (sweeps and sweep not in sweeps):
            continue
        if push_axis == "fore_aft":
            if sweep != "push":
                continue
            fam = "fore-aft push (along the forward axis at onset; + forward, - back)"
        if sweep == "delay" and delays is not None:
            xs = tuple(sorted(set(int(d) for d in delays) | {int(xnom)}))
        out.append((sweep, fam, xk, xs, xnom, fixed, extra))
    return out


def run_sweep(model: str, seed0: int, n_seeds: int, delays, push_axis: str, sweeps, workers: int, out: Path,
              date: str) -> dict:
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    ec = env_config(model)
    env = S.build_env(ec, None, False)
    names = [env.model.actuator(int(i)).name for i in env._leg_act]
    del env
    fams, points = S.plan(families=families(delays, push_axis, sweeps))
    seeds = [seed0 + i for i in range(n_seeds)]
    tasks = [(k, p, s) for k, p in points.items() for s in seeds]
    print(f"{push_axis}: {len(points)} points x {len(seeds)} seeds = {len(tasks)} episodes on {workers} workers",
          flush=True)
    results_eps: dict = {k: [] for k in points}
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(ec, push_axis)) as pool:
        for done, (k, rec) in enumerate(pool.imap_unordered(SW._task, tasks, chunksize=1), start=1):
            results_eps[k].append(rec)
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t_start
                print(f"  {done}/{len(tasks)} episodes, {el / 60:.1f} min", flush=True)
    head, dirty = git_state()
    run = f"walk60_fore128_{push_axis}_{seed0}"
    res = {"run": run, "run_label": f"model-based trot (walk60) on {model}, {push_axis} pushes, seeds from {seed0}",
           "date": date, "simulation_only": True,
           "checkpoint": {"zip_name": "walk/walk60.py (a controller; no checkpoint)", "zip_sha256": sha(HERE / "walk60.py"),
                          "vecnormalize_name": None, "vecnormalize_sha256": None},
           "controller": {"file": "walk/walk60.py", "sha256": sha(HERE / "walk60.py"),
                          "params": walk60.dataclasses.asdict(walk60.WalkParams())},
           "driver": {"file": "walk/fore128_walk60.py", "sha256": sha(Path(__file__)), "push_axis": push_axis,
                      "delays_ms": None if delays is None else [int(d) for d in delays]},
           "run_commit": head, "dirty": dirty,
           "env": {"trike_env_path": os.path.relpath(S.trike_env.__file__, RL.parents[2]),
                   "trike_env_sha256": S.sha256(Path(S.trike_env.__file__)), "model": ec["model_path"],
                   "model_sha256": S.sha256(G.GEN / ec["model_path"]), "plant_config": ec},
           "seeds": seeds, "seeds_per_point": len(seeds), "workers": workers,
           "settings": {"nominal_delay_ms": S.NOMINAL_DELAY_MS, "command_mps": S.COMMAND, "push_t_s": S.PUSH_T,
                        "push_duration_s": S.PUSH_S, "push_look_s": S.PUSH_LOOK_S, "speed_drop_break": S.DROP,
                        "window_from_s": G.WINDOW_S, "continuous_rating_nm": G.THERMAL, "nominal_limit_nm": G.LIMIT,
                        "rpm_allowance": RPM_ALLOW, "training_ranges": S.TRAINING},
           "joint_names": names, "points": {}, "families": []}
    S.pair_pushes(points, results_eps)
    for k, p in points.items():
        eps = sorted(results_eps[k], key=lambda e: e["seed"])
        res["points"][k] = {"point": p, "summary": S.summarise_point(p, eps, names), "episodes": eps}
    res["families"] = fams
    S.add_breaks(res)
    res["episodes_run"] = len(tasks)
    res["wall_s"] = time.time() - t_start
    S.write_outputs(res, out)  # sweep_<run>.json.gz and summary_<run>.md (the sweep's own Markdown)
    print(f"wrote {out} in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return res


def point_summary(res: dict, key: str) -> dict:
    """One sweep point: the sweep's summary figures the report quotes, plus peak drive speed and the worst episode's
    joint. Push figures: ``extra_travel_along_push_m_*`` is the sweep's ``lateral_extra_peak_m_*`` (the largest travel
    along the push within 3 s of the onset, minus the same seed's unpushed episode's), named for either axis; for a
    fore-aft push the sweep's recovery count is set to None (``RECOVERY_NOTE_FORE_AFT``)."""
    v = res["points"][key]
    s, eps, names = v["summary"], v["episodes"], res["joint_names"]
    ok = [e for e in eps if not e["fell"] and "forward_speed_mps" in e]
    out = {"episodes": s["episodes"], "falls": s["falls"], "fall_terminations": s["fall_terminations"]}
    for k in ("speed_mps", "speed_min_mps", "drift_abs_mean_m", "drift_abs_worst_m", "heading_change_abs_worst_deg",
              "power_w", "heat_worst", "heat_worst_joint", "heat_worst_episode", "at_nominal_limit_worst",
              "at_nominal_limit_worst_joint", "at_point_limit_worst", "at_point_limit_worst_joint", "slip_worst",
              "slip_worst_foot"):
        if k in s:
            out[k] = s[k]
    cmd = v["point"]["command"]
    out["episodes_under_90pct_command"] = sum(e["forward_speed_mps"] < 0.9 * cmd for e in ok)  # of those that did not fall
    out["episodes_worst_drive_at_or_over_1"] = sum(max(e["heat"]) >= 1.0 for e in ok)
    if ok:
        he = [(max(e["heat"]), names[int(np.argmax(e["heat"]))]) for e in ok]
        out["heat_worst_episode_joint"] = max(he)[1]
        if all("peak_rpm" in e for e in ok):
            pk = np.max([e["peak_rpm"] for e in ok], axis=0).tolist()
            p99 = np.max([e["p99_rpm"] for e in ok], axis=0).tolist()
            out["drive_speed"] = group_peaks(names, pk, p99)
    q = s.get("push")
    if q:
        out["push"] = {k: q.get(k) for k in ("force_n", "impulse_ns", "com_dv_achieved_mps_mean")}
        out["push"]["extra_travel_along_push_m_mean"] = q.get("lateral_extra_peak_m_mean")
        out["push"]["extra_travel_along_push_m_max"] = q.get("lateral_extra_peak_m_max")
        out["push"]["roll_peak_deg_max"] = q.get("roll_peak_deg_max")
        if res["driver"]["push_axis"] == "fore_aft":
            out["push"].update(recovered=None, recovery_s_median=None, recovery_s_max=None,
                               recovery_note=RECOVERY_NOTE_FORE_AFT)
        else:
            out["push"].update({k: q.get(k) for k in ("recovered", "recovery_s_median", "recovery_s_max")})
        rp = [e["push"]["rp_peak_after_onset_deg"] for e in eps if "rp_peak_after_onset_deg" in e.get("push", {})
              and not e["fell"]]
        if rp:
            out["push"]["roll_pitch_peak_after_onset_deg_max"] = np.max(rp, axis=0).tolist()
    return out


def sweep_summary(res: dict) -> dict:
    fams = []
    for fam in res["families"]:
        rows = [{"x": r["x"], **point_summary(res, r["key"])} for r in fam["rows"]]
        fams.append({"sweep": fam["sweep"], "family": fam["family"], "x_key": fam["x_key"],
                     "nominal_x": fam["nominal_x"], "nominal_speed_mps": fam.get("nominal_speed"),
                     "breaks": fam["breaks"], "rows": rows})
    return {"run": res["run"], "seeds": [res["seeds"][0], res["seeds"][-1]], "model": res["env"]["model"],
            "push_axis": res["driver"]["push_axis"], "run_commit": res["run_commit"], "dirty": res["dirty"],
            "wall_s": round(res["wall_s"], 1), "episodes_run": res["episodes_run"], "families": fams}


# ---------------------------------------------------------------- the small summary (committed) and its Markdown

def _slow_gait_files(slow_gait) -> list:
    """(file name, record) for every fore128_slow_gait.py output in ``slow_gait`` (None: none)."""
    if slow_gait is None:
        return []
    return [(f.name, json.loads(f.read_text())) for f in sorted(Path(slow_gait).glob("*.json"))]


def _window_cell(g: dict, source: str) -> dict:
    """One group of fore128_slow_gait.py's summary: the 10-20 s figures beside a row."""
    return {"source": source, "episodes_upright": g["episodes"] - g["falls"], "slow_episodes": g["slow_after"],
            "slow_hot_episodes": g.get("slow_hot_after"),
            "speed_10_20_range_mps": g["speed_10_20_range_mps"], "hind_knee_10_20_range": g["hind_knee_10_20_range"],
            "speed_10_20_range_mps_slow": g.get("speed_10_20_range_mps_slow"),
            "hind_knee_10_20_range_slow": g.get("hind_knee_10_20_range_slow"),
            "pitch_10_20_mean_deg_slow": g["pitch_10_20_mean_deg_slow"],
            "pitch_10_20_range_deg_slow": g.get("pitch_10_20_range_deg_slow"),
            "speed_10_20_range_mps_slow_hot": g.get("speed_10_20_range_mps_slow_hot"),
            "hind_knee_10_20_range_slow_hot": g.get("hind_knee_10_20_range_slow_hot"),
            "pitch_10_20_mean_deg_slow_hot": g.get("pitch_10_20_mean_deg_slow_hot"),
            "speed_2_8_range_mps": g["speed_2_8_range_mps"]}


def _episode_key(e: dict) -> tuple:
    """What must match, seed by seed, for a slow-gait episode to be the sweep's: whether it fell and the sweep's own
    2-20 s forward speed (None in a slow-gait file written before that field was kept, which then never matches)."""
    return bool(e["fell"]), e.get("forward_speed_mps")


def attach_sweep_windows(ss: dict, raw: dict, slow: list) -> None:
    """Beside the rows of ``ss`` (sweep_summary of ``raw``): the 10-20 s windows of each sweep-mode slow-gait file with
    the same model, push axis and seeds, on the row whose sweep point is the file's (push dv plus the file's further
    point settings). A file with no further settings goes beside the push family's rows; one with settings (e.g.
    kp_scale 0.8) beside the rows of the families those settings vary. Each carries a check that its episodes are the
    sweep's: the same seeds, falls and 2-20 s forward speeds."""
    for name, sg in slow:
        if (sg.get("mode", "sweep") != "sweep" or sg["model"] != raw["env"]["model"]
                or sg["push_axis"] != raw["driver"]["push_axis"]
                or sg["seeds"] != [raw["seeds"][0], raw["seeds"][-1]]):
            continue
        extra = sg.get("point") or {}
        for fam, rfam in zip(ss["families"], raw["families"]):
            if (fam["sweep"] != "push") if not extra else (fam["x_key"] not in extra):
                continue
            for row, rrow in zip(fam["rows"], rfam["rows"]):
                g = next((g for g in sg["summary"]
                          if S.make_point(push_dv=g["dv"], **extra) == raw["points"][rrow["key"]]["point"]), None)
                if g is None:
                    continue
                theirs = {e["seed"]: _episode_key(e) for e in raw["points"][rrow["key"]]["episodes"]}
                mine = {e["seed"]: _episode_key(e) for e in sg["episodes"] if e["dv"] == g["dv"]}
                row["windows_10_20s"] = {**_window_cell(g, name), "same_episodes_as_sweep": mine == theirs}


def attach_judged_windows(js: dict, raw: dict, slow: list) -> None:
    """Beside the judged conditions of ``js`` (one seed set of ``raw``): the 10-20 s windows of the judged-mode
    slow-gait file with the same model and seeds, with a check that every episode's whole-episode speed (the env's
    ``episode_forward_speed``, distance over 0-20 s) and 2-20 s speed (``gait_metrics``'s ``forward_speed_mps``) are
    the stored ones."""
    for name, sg in slow:
        if sg.get("mode") != "judged" or sg["model"] != raw["model"] or sg["seeds"] != [raw["seeds"][0],
                                                                                         raw["seeds"][-1]]:
            continue
        for g in sg["summary"]:
            n = g["condition"]
            if n not in js["conditions"]:
                continue
            stored = {e["seed"]: (e["speed_mps"], e["measures"]["forward_speed_mps"])
                      for e in raw["conditions"][n]["episodes"]}
            mine = {e["seed"]: (e["speed_mps"], e.get("forward_speed_mps")) for e in sg["episodes"]
                    if e["condition"] == n}
            js["conditions"][n]["windows_10_20s"] = {**_window_cell(g, name), "same_episodes": mine == stored}


def summarise(raw: Path, out: Path, slow_gait=None) -> dict:
    slow = _slow_gait_files(slow_gait)
    summ = {"simulation_only": True, "tool": "walk/fore128_walk60.py summarise", "model": None,
            "thresholds_note": "PREREG_WALK60.md's W1-W4 were written for the 188 mm roll-ankle model "
                               "(trike60_C_closed_ankleroll.xml); shown beside the measurements, not judged.",
            "rpm_allowance": RPM_ALLOW, "rpm_allowance_source": ALLOW_SOURCE, "judged": {}, "sweep": {}}
    if slow:
        summ["windows_note"] = ("windows_10_20s: fore128_slow_gait.py on the same episodes (checked seed by seed: "
                                "falls and 2-20 s speeds for sweep rows, whole-episode and 2-20 s speeds for judged "
                                "conditions); slow = forward speed over 10-20 s under 0.225 m/s; slow and hot = slow "
                                "and the larger hind knee's RMS torque over 10-20 s at or over its 6.75 N m "
                                "continuous rating; hind knee = that RMS / 6.75 N m; pitch + nose down. The sweep's "
                                "own speeds are 2-20 s means and its push comes at 8 s.")
    for f in sorted(raw.glob("judged_*.json.gz")):
        r = read_gz(f)
        tag = f"{r['seeds'][0]}-{r['seeds'][-1]}"
        summ["judged"][tag] = {"commit": r["commit"], "dirty": r["dirty"], "model": r["model"],
                               "model_sha256": r["model_sha256"],
                               "walk60_sha256": r["walk60_sha256"], "wall_s": r["wall_s"],
                               "conditions": {n: judged_summary(c) for n, c in r["conditions"].items()}}
        attach_judged_windows(summ["judged"][tag], r, slow)
    for f in sorted(raw.glob("*/sweep_walk60_fore128_*.json.gz")):
        r = read_gz(f)
        for v in r["points"].values():
            v["point"]["terrain"] = S._tuples(v["point"]["terrain"])
        summ["sweep"][r["run"]] = sweep_summary(r)
        attach_sweep_windows(summ["sweep"][r["run"]], r, slow)
    models = sorted({j["model"] for j in summ["judged"].values()} | {r["model"] for r in summ["sweep"].values()})
    summ["model"] = models[0] if len(models) == 1 else models
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S._round(summ, 6), indent=None, separators=(",", ":"),
                                                 default=str) + "\n")
    (out / "RESULTS.md").write_text(markdown(summ))
    return summ


def _f(x, n=3):
    return "" if x is None else f"{x:.{n}f}"


def _rng(v, n=3):
    return "" if not v else f"{v[0]:.{n}f}-{v[1]:.{n}f}"


def _speed_cell(s):
    ds = s.get("drive_speed")
    if not ds:
        return ""
    return " / ".join(f"{ds['peak_rpm'][k]:.0f} ({ds['p99_rpm'][k]:.0f})" for k in ("roll", "pitch", "front elbow",
                                                                                   "hind knee"))


def _rules_cell(name: str, s: dict) -> str:
    """PREREG_WALK60.md's thresholds by name, with the value each was set against; J1-J4 were the judged conditions
    on the 188 mm model, every other case is for orientation only."""
    th = s.get("prereg_188mm_thresholds")
    if not th:
        return "none (not at the 0.25 m/s command)"
    met = lambda ok: "met" if ok else "**not met**"  # noqa: E731
    w4 = max(v for v in s["slip_ratio_mean_by_foot_at_origin"].values() if v is not None)
    cell = (f"W1 {met(th['W1_no_fall'])} ({s['falls']} falls); W2 {met(th['W2_mean_at_least_0.225_none_under_0.20'])} "
            f"({s['speed_mean_mps']:.3f}, min {s['speed_min_mps']:.3f}); W3 {met(th['W3_every_episode_under_1.0'])} "
            f"({s['heat_worst_episode']['rms_over_continuous']:.2f}); W4 {met(th['W4_slip_at_origin_at_most_0.25'])} "
            f"({w4:.2f})")
    return cell if name in JW.JUDGED else "orientation only: " + cell


X_HEAD = {"delay_ms": "Delay, ms", "friction": "Floor friction", "mass_scale": "Mass, x every body",
          "payload_kg": "Payload on the chassis, kg", "force_scale": "Force limit, x nominal",
          "kp_scale": "Position gain kp, x nominal"}


def _x_head(fam: dict, axis: str) -> str:
    if fam["x_key"] == "push_dv":
        return "Push, free-body dv, m/s (" + ("+ forward, - back" if axis == "fore_aft" else "+ left, - right") + ")"
    return X_HEAD.get(fam["x_key"], fam["x_key"])


def _x_cell(fam: dict, x) -> str:
    if fam["x_key"] == "friction" and x is None:
        return "the model's own"
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def markdown(summ: dict) -> str:
    L = [f"# Model-based trot on `{summ['model']}`: results (SIMULATION)", "",
         f"Written by `walk/fore128_walk60.py summarise`. Model `{summ['model']}`. Every number here is simulation, "
         "of a model whose masses and contact are estimates; nothing is measured. Reported, not judged: no "
         "pre-registration covers this model. " + summ["thresholds_note"] + " PREREG_WALK60.md judged only J1-J4; "
         "its thresholds on every other case here are for orientation only.", "",
         "- Speed: the mean forward speed from 2 s to the end of the 20 s episode, m/s, over the episodes that did "
         "not fall. In the sweep the push comes at 8 s, so a slower gait after it is averaged with the 6 s before "
         "it, and the sweep's break rule (a fall, or a mean more than 20 % below the nominal point's) uses this mean."
         + (" Where `fore128_slow_gait.py` measured the same episodes (checked seed by seed: falls and 2-20 s "
            "speeds for sweep rows, whole-episode and 2-20 s speeds for judged conditions), the 10-20 s speed and "
            "the larger hind knee's load over 10-20 s are shown beside the rows. Slow: under 0.225 m/s over 10-20 s. "
            "Slow and hot: slow, and the larger hind knee's RMS torque over 10-20 s at or over its continuous rating "
            "(1.0); the cut-offs are `fore128_slow_gait.py`'s, not a pre-registration's."
            if "windows_note" in summ else ""),
         "- Drive load: RMS torque / continuous rating at the joint (roll and pitch 7.5, front elbow 10.8, hind knee "
         "6.75 N m: rl/trike_env.py cont_torque_*; desk ratings).",
         "- Peak drive speed: the largest |joint speed| over the episodes that did not fall, from t = 2 s, rpm, per "
         "group roll / pitch / front elbow / hind knee, with the largest 99th percentile in brackets. The budget's "
         "allowances (desk): " + ", ".join(f"{k} {v:g}" for k, v in summ["rpm_allowance"].items()) + " rpm"
         + f" ({summ['rpm_allowance_source']}).",
         "- Torque-limit time: the share of 1 ms physics steps with a drive's torque at or above 99 % of its limit "
         "(the model clamps each drive at 70 % of the maker's peak).", ""]
    for tag, j in summ["judged"].items():
        L += [f"## PREREG_WALK60's conditions, reported only on this model, seeds {tag} (commit {j['commit'][:8]}"
              f"{', dirty' if j['dirty'] else ''})", "",
              "Randomisation and sensor noise on, 10 episodes per condition.", "",
              "| Condition | Falls | Speed mean (min), m/s | Episodes under 90 % of the command | Worst drive RMS / "
              "rating, worst episode (joint) | Episodes with a drive at or over 1.0 | Worst joint's mean RMS / rating "
              "| Heading change, worst episode | Sideways drift, worst episode | Slip ratio, worst foot's mean (at the "
              "sole origin / at the contacts) | Peak drive speed, rpm | Steps at the episode's own torque limit, "
              "worst joint-episode | Hind touchdowns per s, worst episode | PREREG_WALK60's thresholds (written for "
              "the 188 mm roll-ankle model) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, s in j["conditions"].items():
            so = max((v for v in s["slip_ratio_mean_by_foot_at_origin"].values() if v is not None), default=None)
            sc = s.get("slip_ratio_mean_by_foot_at_contacts")
            sc = max((v for v in sc.values() if v is not None), default=None) if sc else None
            hm = s["heat_worst_joint_mean"]
            u = s["episodes_under_90pct_command"]
            L.append(f"| {n} {s['label']} | {s['falls']}/{s['episodes']} | {_f(s['speed_mean_mps'])} "
                     f"({_f(s['speed_min_mps'])}) | {'' if u is None else u} | "
                     f"{_f(s['heat_worst_episode']['rms_over_continuous'], 2)} "
                     f"({s['heat_worst_episode']['joint']}) | {s['episodes_worst_drive_at_or_over_1']} | "
                     f"{_f(hm['rms_over_continuous'], 2)} ({hm['joint']}) | "
                     f"{s['heading_change_abs_worst_deg']:.1f} deg | {s['sideways_drift_abs_worst_m']:.2f} m | "
                     f"{_f(so, 2)} / {_f(sc, 2) or 'not run'} | {_speed_cell(s)} | "
                     f"{100 * s['at_limit_episode_worst']['fraction']:.1f} % ({s['at_limit_episode_worst']['joint']}) "
                     f"| {s['hind_touchdowns_per_s_episode_max']:.1f} | {_rules_cell(n, s)} |")
        L.append("")
        win = [(n, s["windows_10_20s"]) for n, s in j["conditions"].items() if "windows_10_20s" in s]
        if win:
            L += [f"### 10-20 s windows of the same episodes, seeds {tag} (`{win[0][1]['source']}`)", "",
                  "| Condition | Slow over 10-20 s / upright | Slow and hot (hind knee at or over 1.0) | 10-20 s speed, "
                  "range, m/s | Larger hind knee RMS / rating over 10-20 s, range | Slow episodes: 10-20 s speed; "
                  "hind knee; mean pitch (range), deg (+ nose down) | Whole-episode and 2-20 s speeds match the "
                  "stored episodes |",
                  "|---|---|---|---|---|---|---|"]
            for n, w in win:
                pr = w.get("pitch_10_20_range_deg_slow")
                sl = ("" if not w["slow_episodes"] else f"{_rng(w['speed_10_20_range_mps_slow'])}; "
                      f"{_rng(w['hind_knee_10_20_range_slow'], 2)}; {w['pitch_10_20_mean_deg_slow']:.2f}"
                      + (f" ({pr[0]:.2f} to {pr[1]:.2f})" if pr else ""))
                hot = "" if w.get("slow_hot_episodes") is None else str(w["slow_hot_episodes"])
                L.append(f"| {n} | {w['slow_episodes']}/{w['episodes_upright']} | {hot} | "
                         f"{_rng(w['speed_10_20_range_mps'])} | {_rng(w['hind_knee_10_20_range'], 2)} | {sl} | "
                         f"{'yes' if w['same_episodes'] else 'NO'} |")
            L.append("")
    for run, r in summ["sweep"].items():
        L += [f"## Sweep `{run}`: {r['push_axis'].replace('_', '-')} pushes, seeds {r['seeds'][0]}-{r['seeds'][1]} "
              f"(commit {r['run_commit'][:8]}{', dirty' if r['dirty'] else ''}; {r['episodes_run']} episodes, "
              f"{r['wall_s'] / 60:.1f} min)", "",
              "Nominal model, observation noise on, no randomisation, 10 ms delay and 0.25 m/s unless the row says "
              f"otherwise; {r['seeds'][1] - r['seeds'][0] + 1} episodes per point. A push lasts 0.1 s from 8 s; its "
              "free-body dv is its impulse over the robot's whole mass.", ""]
        for fam in r["families"]:
            b = fam["breaks"]
            bl = "; ".join(f"{side}: " + ("none" if v is None else f"{_x_cell(fam, v['x'])} ({', '.join(v['reasons'])})")
                           for side, v in b.items())
            push = fam["x_key"] == "push_dv"
            win = any("windows_10_20s" in row for row in fam["rows"])
            L += [f"### {fam['family']} (the sweep's break rule: {bl})", "",
                  f"| {_x_head(fam, r['push_axis'])} | Falls | Speed mean 2-20 s (min), m/s | Episodes under 90 % of "
                  "the command | Sideways drift, worst, m | Heading change, worst, deg | Worst drive RMS / rating: "
                  "worst joint's mean (worst episode; joint) | Episodes with a drive at or over 1.0 | Peak drive speed, "
                  "rpm | Steps at the nominal torque limit, worst joint's mean |"
                  + (" Extra travel along the push, mean (max), m |" if push else "")
                  + (" 10-20 s: slow / upright | 10-20 s: slow and hot | 10-20 s speed, range, m/s | 10-20 s larger "
                     "hind knee RMS / rating, range |" if win else ""),
                  "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if push else "")
                  + ("---|---|---|---|" if win else "")]
            for row in fam["rows"]:
                q = row.get("push") or {}
                w = row.get("windows_10_20s")
                tail = ""
                if win:
                    tail = (" | | | |" if not w else
                            f" {w['slow_episodes']}/{w['episodes_upright']}"
                            f"{'' if w['same_episodes_as_sweep'] else ' (NOT the sweep episodes)'} | "
                            f"{'' if w.get('slow_hot_episodes') is None else w['slow_hot_episodes']} | "
                            f"{_rng(w['speed_10_20_range_mps'])} | {_rng(w['hind_knee_10_20_range'], 2)} |")
                if "speed_mps" not in row:
                    L.append(f"| {_x_cell(fam, row['x'])} | {row['falls']}/{row['episodes']} | all fell | | | | | | | |"
                             + (" |" if push else "") + tail)
                    continue
                ext = ("" if q.get("extra_travel_along_push_m_mean") is None else
                       f"{q['extra_travel_along_push_m_mean']:.3f} ({q['extra_travel_along_push_m_max']:.3f})")
                L.append(f"| {_x_cell(fam, row['x'])} | {row['falls']}/{row['episodes']} | {_f(row['speed_mps'])} "
                         f"({_f(row['speed_min_mps'])}) | {row['episodes_under_90pct_command']} | "
                         f"{_f(row['drift_abs_worst_m'], 2)} | "
                         f"{row['heading_change_abs_worst_deg']:.1f} | {_f(row['heat_worst'], 2)} "
                         f"({_f(row['heat_worst_episode'], 2)}; {row['heat_worst_joint']}) | "
                         f"{row['episodes_worst_drive_at_or_over_1']} | {_speed_cell(row)} | "
                         f"{100 * row['at_nominal_limit_worst']:.1f} % |" + (f" {ext} |" if push else "") + tail)
            L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("judged")
    a1.add_argument("--model", default=MODEL)
    a1.add_argument("--seeds", default="101-110")
    a1.add_argument("--extra-delays", type=int, nargs="*", default=[])
    a1.add_argument("--workers", type=int, default=4)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("sweep")
    a2.add_argument("--model", default=MODEL)
    a2.add_argument("--seed0", type=int, default=S.SEED0)
    a2.add_argument("--seeds", type=int, default=8)
    a2.add_argument("--delays", type=int, nargs="*", default=None)
    a2.add_argument("--push-axis", choices=("sideways", "fore_aft"), default="sideways")
    a2.add_argument("--sweeps", nargs="*", default=None)
    a2.add_argument("--workers", type=int, default=4)
    a2.add_argument("--out", required=True)
    a2.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a3 = sub.add_parser("summarise")
    a3.add_argument("--raw", required=True)
    a3.add_argument("--out", required=True)
    a3.add_argument("--slow-gait", default=None, help="folder of fore128_slow_gait.py outputs to put beside the rows")
    a = ap.parse_args(argv)
    if a.cmd == "judged":
        r = run_judged(a.model, seeds_arg(a.seeds), a.extra_delays, a.workers, Path(a.out))
        print(f"judged: {len(r['conditions'])} conditions, seeds {a.seeds}, {r['wall_s'] / 60:.1f} min")
    elif a.cmd == "sweep":
        run_sweep(a.model, a.seed0, a.seeds, a.delays, a.push_axis, a.sweeps, a.workers, Path(a.out), a.date)
    else:
        print(markdown(summarise(Path(a.raw), Path(a.out), a.slow_gait)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
