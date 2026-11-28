"""The trot's slow gait: speed and hind-knee load before and after a push, or over a judged episode. Simulation only;
reported, not judged.

Sweep mode (the default). For each push and seed, the frozen trot (``walk60.py``) runs one sweep episode exactly as
``fore128_walk60.py sweep`` runs it (same worker, env, controller and push; nominal model, observation noise on, 10 ms
delay, 0.25 m/s, push at 8 s for 0.1 s; ``--point`` adds the sweep's other point settings, e.g. kp_scale), and this
records, from the episode's 1 ms log:
  - the chassis's forward speed along the start heading over 2-8 s (before the push) and 10-20 s (after it), m/s;
  - the larger hind knee's RMS torque / continuous rating (6.75 N m) over each window;
  - the chassis's mean pitch over each window, deg (+ nose down, so negative is nose up).
``--dv all`` runs every point of the sweep's push family (rl/robustness/sweep.py ``_PUSHES``, free-body dv in m/s).
Each record also keeps the sweep's own 2-20 s forward speed of the episode (``forward_speed_mps``, from
``sweep.episode_record``), so ``fore128_walk60.py summarise`` can check, seed by seed, that these are the sweep's
episodes.

Judged mode (``--judged J1,J3,...``). The same windows over episodes of ``fore128_walk60.py judged``'s conditions
(randomisation and noise on, as that tool runs them), with each episode's whole-episode speed (``speed_mps``: the
env's ``episode_forward_speed``, distance over 0-20 s) and 2-20 s speed (``forward_speed_mps``: ``gait_metrics``'s
measure), so it can be matched to the stored judged episode.

Two counts per group:
  - "slow": the 10-20 s speed is under 0.225 m/s (90 % of the 0.25 m/s command);
  - "slow and hot": slow, and the larger hind knee's RMS torque over 10-20 s is at or over its continuous rating
    (1.0). This is the rule the trot's report uses for "the slow gait"; the cut-offs are this file's, not a
    pre-registration's.

Usage (from the repository root):
  python hardware/triceratops_build/walk/fore128_slow_gait.py --model trike60_C_closed_capsule_fore128.xml \
      --push-axis fore_aft --dv all --seed0 200000 --seeds 10 --out FILE.json
  python hardware/triceratops_build/walk/fore128_slow_gait.py --push-axis sideways --dv 0 \
      --point '{"payload_kg": 0.5}' --seed0 200000 --seeds 10 --out FILE.json
  python hardware/triceratops_build/walk/fore128_slow_gait.py --judged J1,J3,J1d16 --seeds-judged 101-110 --out FILE.json
  (--model walk60: the walk's own roll-ankle model.)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fore128_walk60 as F  # noqa: E402

S, SW = F.S, F.SW
walk60, gait_metrics = F.walk60, F.gait_metrics
_STASH: dict = {}
HIND_KNEE_RATING_NM = 6.75  # rl/trike_env.py cont_torque_hind_flexion
SLOW_MPS = 0.225
HOT_RATIO = 1.0  # the larger hind knee's 10-20 s RMS / rating at or over this: "slow and hot" when also slow
PUSHES = [x for f in S.FAMILIES if f[0] == "push" for x in f[3]]  # the sweep's push points, free-body dv, m/s


def _record(arr, p, seed, pusher):
    _STASH["arr"] = arr
    return F.record_with_speed(arr, p, seed, pusher)


def _init(model: str, axis: str) -> None:
    F._init_worker(F.env_config(None if model == "walk60" else model), axis)
    S.episode_record = _record


def windows(a: dict) -> dict:
    t, x = a["t"], a["pos"][:, :2]
    h = np.array([np.cos(a["heading0"]), np.sin(a["heading0"])])
    names = list(a["names"])
    hk = [j for j, n in enumerate(names) if "hind_flexion" in n]
    pitch = np.asarray(a["rp"])[:, 1]
    out = {}
    for tag, (t0, t1) in (("2_8", (2.0, 8.0)), ("10_20", (10.0, 20.0))):
        w = (t >= t0) & (t < t1)
        if w.sum() < 100:
            out.update({f"speed_{tag}_mps": None, f"hind_knee_rms_over_rating_{tag}": None, f"pitch_{tag}_deg": None})
            continue
        i0, i1 = int(np.argmax(w)), int(len(w) - 1 - np.argmax(w[::-1]))
        out[f"speed_{tag}_mps"] = round(float((x[i1] - x[i0]) @ h / (t[i1] - t[i0])), 5)
        out[f"hind_knee_rms_over_rating_{tag}"] = round(
            float((np.sqrt((a["tau"][w][:, hk] ** 2).mean(0)) / HIND_KNEE_RATING_NM).max()), 4)
        out[f"pitch_{tag}_deg"] = round(float(pitch[w].mean()), 3)
    return out


def _job(args):
    dv, seed, extra = args
    env, ctrl = SW.env_and_controller(None)
    rec = S.run_episode(env, None, None, seed, S.make_point(push_dv=dv, **extra), controller=ctrl.act)
    return {"dv": dv, "seed": seed, "fell": rec["fell"], "forward_speed_mps": rec.get("forward_speed_mps"),
            **windows(_STASH["arr"])}


def _judged_job(job):
    """One episode as ``fore128_walk60.judged_episode`` runs it (the same lines), windowed."""
    name, seed, plant, impratio, command, delay = job
    env = walk60.TrikeEnv(walk60.plant_config(**plant))
    if impratio is not None:
        env.model.opt.impratio = float(impratio)
    ctrl = walk60.TrotController(env, walk60.WalkParams())
    log = gait_metrics.record_episode(env, ctrl.act, seed=seed, command=command, delay_steps=delay,
                                      slip_at_contact=False)
    return {"condition": name, "seed": seed, "fell": bool(log["fell"]),
            "speed_mps": float(log["episode_forward_speed"]),
            "forward_speed_mps": float(gait_metrics.measures(log, command)["forward_speed_mps"]), **windows(log)}


def group_summary(rs: list) -> dict:
    """Over a group of episodes: falls, slow episodes (10-20 s under 0.225 m/s), slow and hot episodes (also the
    larger hind knee at or over 1.0 of its rating over 10-20 s) and the windows' ranges."""
    ok = [r for r in rs if not r["fell"] and r["speed_10_20_mps"] is not None]
    slow = [r for r in ok if r["speed_10_20_mps"] < SLOW_MPS]
    hot = [r for r in slow if r["hind_knee_rms_over_rating_10_20"] >= HOT_RATIO]

    def rng(key, rows):
        return [min(r[key] for r in rows), max(r[key] for r in rows)] if rows else None

    def pitch(rows):
        return float(np.mean([r["pitch_10_20_deg"] for r in rows])) if rows else None

    return {"episodes": len(rs), "falls": sum(r["fell"] for r in rs), "slow_after": len(slow),
            "slow_hot_after": len(hot),
            "speed_2_8_range_mps": rng("speed_2_8_mps", ok), "speed_10_20_range_mps": rng("speed_10_20_mps", ok),
            "hind_knee_2_8_max": max(r["hind_knee_rms_over_rating_2_8"] for r in ok) if ok else None,
            "hind_knee_10_20_range": rng("hind_knee_rms_over_rating_10_20", ok),
            "pitch_2_8_mean_deg": float(np.mean([r["pitch_2_8_deg"] for r in ok])) if ok else None,
            "pitch_10_20_mean_deg_slow": pitch(slow),
            "speed_10_20_range_mps_slow": rng("speed_10_20_mps", slow),
            "hind_knee_10_20_range_slow": rng("hind_knee_rms_over_rating_10_20", slow),
            "pitch_10_20_range_deg_slow": rng("pitch_10_20_deg", slow),
            "pitch_10_20_mean_deg_slow_hot": pitch(hot),
            "speed_10_20_range_mps_slow_hot": rng("speed_10_20_mps", hot),
            "hind_knee_10_20_range_slow_hot": rng("hind_knee_rms_over_rating_10_20", hot)}


def provenance(model: str, argv) -> dict:
    """The command line and the files the episodes depend on, hashed before any episode runs."""
    head, dirty = F.git_state()
    mfile = Path(walk60.plant_config().model_path) if model == "walk60" else F.GEN / model
    return {"argv": list(argv), "commit": head, "dirty": dirty,
            "fore128_slow_gait_sha256": F.sha(Path(__file__)), "fore128_walk60_sha256": F.sha(HERE / "fore128_walk60.py"),
            "walk60_sha256": F.sha(HERE / "walk60.py"), "trike_env_sha256": F.sha(Path(S.trike_env.__file__)),
            "model_file": mfile.name, "model_sha256": F.sha(mfile)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=F.MODEL)
    ap.add_argument("--push-axis", choices=("sideways", "fore_aft"), default="fore_aft")
    ap.add_argument("--dv", default="0,-0.3,-1.0", help="comma-separated free-body dv, m/s, or 'all'")
    ap.add_argument("--point", default="{}", help="JSON of further sweep point settings, e.g. '{\"kp_scale\": 0.8}'")
    ap.add_argument("--seed0", type=int, default=S.SEED0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--judged", default=None, help="comma-separated fore128_walk60 judged condition names")
    ap.add_argument("--seeds-judged", default="101-110")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prov = provenance(a.model, argv)
    t0 = time.time()
    if a.judged:
        names = a.judged.split(",")
        ds = sorted({int(n[3:]) for n in names if n.startswith("J1d")})
        conds = F.judged_conditions(ds)
        mpath = str(F.GEN / a.model) if a.model != "walk60" else walk60.plant_config().model_path
        seeds = F.seeds_arg(a.seeds_judged)
        jobs = [(n, s, {**conds[n][1], "model_path": mpath}, conds[n][2], conds[n][3], conds[n][4])
                for n in names for s in seeds]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            rows = pool.map(_judged_job, jobs, chunksize=1)
        summ = [{"condition": n, **group_summary([r for r in rows if r["condition"] == n])} for n in names]
        res = {"simulation_only": True, "tool": "walk/fore128_slow_gait.py", "mode": "judged", **prov,
               "model": a.model, "seeds": [seeds[0], seeds[-1]], "wall_s": round(time.time() - t0, 1),
               "summary": summ, "episodes": rows}
    else:
        dvs = PUSHES if a.dv == "all" else [float(v) for v in a.dv.split(",")]
        extra = json.loads(a.point)
        seeds = [a.seed0 + i for i in range(a.seeds)]
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.model, a.push_axis)) as pool:
            rows = pool.map(_job, [(dv, s, extra) for dv in dvs for s in seeds], chunksize=1)
        summ = [{"dv": dv, **group_summary([r for r in rows if r["dv"] == dv])} for dv in dvs]
        res = {"simulation_only": True, "tool": "walk/fore128_slow_gait.py", "mode": "sweep", **prov,
               "model": a.model, "push_axis": a.push_axis, "point": extra, "seeds": [seeds[0], seeds[-1]],
               "wall_s": round(time.time() - t0, 1), "summary": summ, "episodes": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    for s in summ:
        print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
