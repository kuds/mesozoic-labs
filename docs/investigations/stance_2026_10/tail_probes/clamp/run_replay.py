"""Open-loop decomposition of the tail clamp (exploratory, report-only; added beyond the brief's conditions).

PYTHONPATH=<policies/code_copy> python -B run_replay.py OUT_DIR --group s42|s80 --lens LENS_RUN_DIR [--force V]
                                                        [--seeds 3042-3061] [--only name,name]
PYTHONPATH=<policies/code_copy> python -B run_replay.py OUT_DIR --statue [--force V] [--seeds 3042-3061]

Replay: the unforced episode's own applied commands, step for step, as the lens recorded them
(com_revisit/notes/policies/run/tracks/<who>.npz action_<i>, float32, which is the dtype model.predict returns, so the
replay is exact), with entry 14 (tail_3_pitch_act) replaced by V.  No policy runs: the other 14 commands are frozen at
what the policy did with the rail, so (replay forced - unforced) is the plant's own response to the tail_3 command
under the policy's command history, and (closed-loop forced - replay forced) is what the policy's feedback adds.  With
no --force the replay must reproduce the lens's track bit for bit (checked; compare.jsonl).

Statue: the neutral action (zeros) at the same spawns, with entry 14 replaced by V when --force is given: the plant
alone at the home controls.

Same env constructor, fresh env per seed, env.reset(seed), 1,000 steps and the same per-episode record as run_clamp.py
(its roll() and window_means(), imported from this folder).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import run_clamp as rc
from environments.shared.config import load_stage_config
from environments.shared.species_registry import env_class
from environments.shared.tests.stance_track_helpers import stance_track_quality, stance_track_series


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--group", choices=("s42", "s80"))
    parser.add_argument("--lens", type=Path)
    parser.add_argument("--statue", action="store_true")
    parser.add_argument("--force", type=float, default=None)
    parser.add_argument("--seeds", default="3042-3061")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if args.statue:
        names = ["statue"]
    else:
        only = [w for w in args.only.split(",") if w]
        names = only or [n for n in rc.NAMES if n.startswith(args.group)]
    (args.out / "tracks").mkdir(parents=True, exist_ok=True)
    ctor = dict(load_stage_config(rc.MID, "stance")["env_kwargs"])
    factory = lambda: env_class(rc.MID)(**ctor)  # noqa: E731
    steps = 1000
    step_windows = {"0_end": (0, steps), "200_end": (200, steps), "500_end": (500, steps)}
    (args.out / "meta.json").write_text(json.dumps({"argv": sys.argv, "force": args.force, "seeds": seeds,
                                                    "mode": "statue" if args.statue else "replay"}, indent=1))
    out = open(args.out / "episodes.jsonl", "a")
    cmp_out = open(args.out / "compare.jsonl", "a")
    for who in names:
        if (args.out / "tracks" / f"{who}.done").exists():
            print(who, "done already", flush=True)
            continue
        t_who = time.time()
        lens_npz = None if args.statue else np.load(args.lens / "tracks" / f"{who}.npz")
        lens_seeds = None if lens_npz is None else [int(s) for s in lens_npz["seeds"]]
        save: dict[str, np.ndarray] = {"seeds": np.array(seeds)}
        for i, seed in enumerate(seeds):
            if lens_npz is None:
                zeros = np.zeros(15, dtype=np.float32)
                act = lambda obs, step, zeros=zeros: zeros  # noqa: E731
            else:
                j = lens_seeds.index(seed)
                recorded = lens_npz[f"action_{j}"]
                act = lambda obs, step, recorded=recorded: recorded[step - 1]  # noqa: E731
            env = factory()
            t0 = time.time()
            ep, arrays, infos, tail_cols = rc.roll(env, seed=seed, act=act, force=args.force, horizon=steps)
            env.close()
            tr = ep.track
            row: dict = {"who": who, "seed": seed, "force": args.force, "mode": "statue" if args.statue else "replay",
                         "stood": ep.stood, "steps": ep.steps, "terminated": ep.terminated,
                         "reason": ep.termination_reason, "end_time_s": float(tr.t[-1] - tr.t[0]),
                         "wall_s": round(time.time() - t0, 2), "return": float(np.sum(arrays["reward"])),
                         "metrics": {}, "win": {}}
            for key, (a, b) in step_windows.items():
                row["win"][key] = rc.window_means(arrays, infos, a, b, None)
            if ep.stood:
                series = stance_track_series(tr)
                for w in (2.0, 5.0, 9.0):
                    row["metrics"][str(w)] = stance_track_quality(tr, window_start_s=w, series=series)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if lens_npz is not None and args.force is None:
                checks = {
                    "com": np.array_equal(tr.com.astype(np.float32), lens_npz[f"com_{j}"]),
                    "quat": np.array_equal(tr.root_quat.astype(np.float32), lens_npz[f"quat_{j}"]),
                    "feet": np.array_equal(tr.foot_pos.astype(np.float32), lens_npz[f"feet_{j}"]),
                    "action": np.array_equal(arrays["applied"].astype(np.float32), lens_npz[f"action_{j}"]),
                }
                cmp_out.write(json.dumps({"who": who, "seed": seed, "all": all(checks.values()), **checks}) + "\n")
                cmp_out.flush()
            save[f"t_{i}"] = tr.t
            save[f"com_{i}"] = tr.com.astype(np.float32)
            save[f"quat_{i}"] = tr.root_quat.astype(np.float32)
            save[f"feet_{i}"] = tr.foot_pos.astype(np.float32)
            save[f"reward_{i}"] = arrays["reward"].astype(np.float32)
            save[f"tailq_{i}"] = arrays["qpos"][:, tail_cols].astype(np.float32)
            save[f"tip_{i}"] = arrays["tip"].astype(np.float32)
        np.savez_compressed(args.out / "tracks" / f"{who}.npz", **save)
        (args.out / "tracks" / f"{who}.done").write_text("ok\n")
        print(f"{who}: {len(seeds)} episodes, {time.time() - t_who:.1f} s", flush=True)
    out.close()
    cmp_out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
