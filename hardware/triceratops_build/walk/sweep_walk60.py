"""The robustness sweep of the model-based trot, per PLAN_SWEEP_WALK60.md. Simulation only.

Runs ``rl/robustness/sweep.py``'s points, physics changes, push, terrain, logging, summaries, break rule and charts,
with ``walk60.TrotController`` (frozen ``WalkParams()``) as the source of the actions instead of a policy. Reported,
not judged.

Usage (from the repository root):
  python hardware/triceratops_build/walk/sweep_walk60.py [--workers 3] [--seeds 8] [--sweeps delay push ...]
Output: hardware/triceratops_build/walk/results/sweep_walk60/ (sweep_walk60.json.gz, summary_walk60.md, charts).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parent / "rl"
for p in (HERE, RL / "robustness", RL / "analysis", RL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import sweep as S  # noqa: E402  (imports gait_pipe1, which imports trike_env)
import walk60  # noqa: E402

RUN = "walk60"
LABEL = "model-based trot (walk60)"
_W: dict = {}


def env_config() -> dict:
    """The walk's plant config as a run record stores it (model by file name, JSON-shaped)."""
    ec = walk60.plant_config().to_dict()
    ec["model_path"] = Path(ec["model_path"]).name
    return json.loads(json.dumps(ec, default=list))


def _init_worker(ec: dict) -> None:
    _W["ec"], _W["envs"] = ec, {}
    base = S.build_env(ec, None, False)
    _W["base"] = base
    _W["envs"][json.dumps(None)] = (base, walk60.TrotController(base, walk60.WalkParams()))


def env_and_controller(terrain):
    key = json.dumps(terrain)
    if key not in _W["envs"]:
        env = S.build_env(_W["ec"], terrain, False)
        # the controller's kinematics come from the robot model; a terrain env's model file is a temporary copy
        # with one box added, so the controller is built on the base env and then pointed at the terrain env
        ctrl = walk60.TrotController(_W["base"], walk60.WalkParams())
        ctrl.env = env
        _W["envs"][key] = (env, ctrl)
    return _W["envs"][key]


def _task(args):
    key, p, seed = args
    env, ctrl = env_and_controller(p["terrain"])
    t0 = time.time()
    rec = S.run_episode(env, None, None, seed, p, controller=ctrl.act)
    rec["wall_s"] = round(time.time() - t0, 2)
    return key, rec


def walk_markdown(res: dict) -> str:
    text = S._markdown_policy(res)
    return (text.replace("One policy, so this is a report", "One controller (the model-based trot, "
                         "`walk/walk60.py`, frozen parameters), so this is a report")
            .replace("- Policy `", "- Controller `")
            .replace("Training ranges:", "Shaded as training ranges on the charts: the learned policies'. The walk "
                     "was developed and judged with the same randomisation, on friction 0.4 and 1.0. Training "
                     "ranges:"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(HERE / "results" / "sweep_walk60"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sweeps", nargs="*", default=None)
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args(argv)
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ec = env_config()
    env = S.build_env(ec, None, False)
    names = [env.model.actuator(int(i)).name for i in env._leg_act]
    del env
    fams, points = S.plan(sweeps=a.sweeps)
    seeds = [S.SEED0 + i for i in range(a.seeds)]
    tasks = [(k, p, s) for k, p in points.items() for s in seeds]
    print(f"{len(points)} points x {len(seeds)} seeds = {len(tasks)} episodes on {a.workers} workers", flush=True)
    results_eps: dict = {k: [] for k in points}
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(ec,)) as pool:
        for done, (k, rec) in enumerate(pool.imap_unordered(_task, tasks, chunksize=1), start=1):
            results_eps[k].append(rec)
            if done % 40 == 0 or done == len(tasks):
                el = time.time() - t_start
                print(f"  {done}/{len(tasks)} episodes, {el / 60:.1f} min, about "
                      f"{el / done * (len(tasks) - done) / 60:.1f} min left", flush=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(HERE), str(RL)], capture_output=True,
                                text=True, cwd=HERE).stdout.strip())
    walk_sha = S.sha256(HERE / "walk60.py")
    res = {"run": RUN, "run_label": LABEL, "date": a.date, "simulation_only": True,
           "checkpoint": {"zip_name": "walk/walk60.py (a controller; no checkpoint)", "zip_sha256": walk_sha,
                          "vecnormalize_name": None, "vecnormalize_sha256": None},
           "controller": {"file": "walk/walk60.py", "sha256": walk_sha,
                          "params": walk60.dataclasses.asdict(walk60.WalkParams())},
           "run_commit": head, "dirty": dirty,
           "env": {"trike_env_path": os.path.relpath(S.trike_env.__file__, RL.parents[2]),
                   "trike_env_sha256": S.sha256(Path(S.trike_env.__file__)),
                   "model": ec["model_path"], "model_sha256": S.sha256(S.G.GEN / ec["model_path"]),
                   "TRIKE_ENV_DIR": os.environ.get("TRIKE_ENV_DIR"), "plant_config": ec},
           "seeds": seeds, "seeds_per_point": len(seeds), "workers": a.workers,
           "settings": {"nominal_delay_ms": S.NOMINAL_DELAY_MS, "command_mps": S.COMMAND, "push_t_s": S.PUSH_T,
                        "push_duration_s": S.PUSH_S, "push_look_s": S.PUSH_LOOK_S,
                        "recover_vlat_mps": S.RECOVER_VLAT, "recover_window_s": S.RECOVER_WIN_S,
                        "recover_hold_s": S.RECOVER_HOLD_S, "terrain_edge_m": S.TERRAIN_EDGE_M,
                        "speed_drop_break": S.DROP, "window_from_s": S.G.WINDOW_S,
                        "continuous_rating_nm": S.G.THERMAL, "nominal_limit_nm": S.G.LIMIT,
                        "training_ranges": S.TRAINING},
           "joint_names": names, "points": {}, "families": []}
    S.pair_pushes(points, results_eps)
    for k, p in points.items():
        eps = sorted(results_eps[k], key=lambda e: e["seed"])
        res["points"][k] = {"point": p, "summary": S.summarise_point(p, eps, names), "episodes": eps}
    res["families"] = fams
    S.add_breaks(res)
    res["episodes_run"] = len(tasks)
    res["wall_s"] = time.time() - t_start
    S.write_outputs(res, out)
    if not a.no_charts:
        import charts

        for fam, path in charts.make_charts(res, out).items():
            for f in res["families"]:
                if f["sweep"] == fam:
                    f["chart"] = path.name
        S.write_outputs(res, out)
    print(S.markdown(res))
    print(f"wrote {out} in {(time.time() - t_start) / 60:.1f} min")
    return 0


# the sweep's Markdown, worded for a controller (write_outputs calls S.markdown)
S._markdown_policy = S.markdown
S.markdown = walk_markdown

if __name__ == "__main__":
    raise SystemExit(main())
