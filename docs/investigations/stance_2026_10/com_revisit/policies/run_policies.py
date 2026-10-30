"""CoM stance tracks of the ECS T. rex stance policies and the neutral-action statue at the same spawns.

PYTHONPATH=<code_copy> python -B run_policies.py OUT_DIR [--seeds 3042-3061] [--only who,who]

Rolls, one after the other in this one process, in the scratch copy of the sealed run-branch code (8737cd3d) with
PR #650's reduction files copied in:
  statue            the neutral action (zeros) at the same spawns (the control)
  start_s42/s80     the two start checkpoints (stance6 s42 3,145,728; s80 4,194,304)
  <run name>        the 24 end checkpoints (stage1_<E>_steps, E 4,718,592 for s42_*, 5,767,168 for s80_*)
each deterministic (``model.predict(normalizer.normalize_obs(obs), deterministic=True)``), on the stance config's
own env constructor (``env_class(mid)(**load_stage_config(mid, "stance")["env_kwargs"])``, reset noise 0.05),
1,000 steps, on the M1 seeds.  Every track is kept (tracks/<who>.npz, float32, no support points) so a kinematic window can be chosen afterwards;
per episode the metrics at windows from 2, 3, 4, 5 and 9 s and the policy's window command means go to
episodes.jsonl.  Read only: checkpoints are opened, never written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class
from environments.shared.tests.com_stance_helpers import run_tracked_episode
from environments.shared.tests.stance_track_helpers import stance_track_quality, stance_track_series

RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs")
NAMES = (
    [f"s42_A_{n}" for n in range(1101, 1152, 10)]
    + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
    + [f"s80_A_{n}" for n in range(1301, 1352, 10)]
    + [f"s80_B_{n}" for n in range(1401, 1452, 10)]
)
END = {"s42": 4_718_592, "s80": 5_767_168}
START = {"s42": (3_145_728, "s42_A_1101"), "s80": (4_194_304, "s80_A_1301")}
WINDOWS = (2.0, 3.0, 4.0, 5.0, 9.0)
MID = ModelId("tyrannosaurus_rex", "anatomical")


def pairs() -> list[tuple[str, Path | None, Path | None]]:
    out: list[tuple[str, Path | None, Path | None]] = [("statue", None, None)]
    for start, (steps, run) in START.items():
        models = RUNS / run / "01_stance" / "models"
        out.append((f"start_{start}", models / f"stage1_{steps}_steps.zip", models / f"stage1_vecnormalize_{steps}_steps.pkl"))
    for name in NAMES:
        steps = END[name[:3]]
        models = RUNS / name / "01_stance" / "models"
        out.append((name, models / f"stage1_{steps}_steps.zip", models / f"stage1_vecnormalize_{steps}_steps.pkl"))
    return out


def save_tracks(path: Path, tracks: list, seeds: list[int], actions: list[np.ndarray]) -> None:
    """Each track at full rate in float32 (t in float64), without its support points (their keys are reduced in
    the run at every window of WINDOWS); the actions taken in float32."""
    arrays: dict[str, np.ndarray] = {"seeds": np.array(seeds), "labels": np.array(tracks[0].labels)}
    for i, tr in enumerate(tracks):
        arrays[f"t_{i}"] = tr.t
        arrays[f"com_{i}"] = tr.com.astype(np.float32)
        arrays[f"quat_{i}"] = tr.root_quat.astype(np.float32)
        arrays[f"feet_{i}"] = tr.foot_pos.astype(np.float32)
        arrays[f"load_{i}"] = tr.foot_load.astype(np.float32)
        arrays[f"heading_{i}"] = tr.foot_heading.astype(np.float32)
        arrays[f"action_{i}"] = actions[i].astype(np.float32)
    arrays["floor_z"] = np.array(tracks[0].floor_z)
    arrays["gravity"] = np.array(tracks[0].gravity)
    np.savez_compressed(path, **arrays)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--seeds", default="3042-3061")
    parser.add_argument("--only", default="")
    parser.add_argument("--pair", nargs=3, metavar=("NAME", "ZIP", "PKL"), help="roll this one checkpoint instead")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    only = {w for w in args.only.split(",") if w}
    (args.out / "tracks").mkdir(parents=True, exist_ok=True)
    ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
    factory = lambda: env_class(MID)(**ctor)  # noqa: E731
    out = open(args.out / "episodes.jsonl", "a")
    todo = [(args.pair[0], Path(args.pair[1]), Path(args.pair[2]))] if args.pair else pairs()
    for who, zip_path, pkl_path in todo:
        if only and who not in only:
            continue
        if (args.out / "tracks" / f"{who}.npz").exists():
            print(who, "done already", flush=True)
            continue
        t_who = time.time()
        if zip_path is None:
            act = None
        else:
            model, normalizer, _ = load_sb3_checkpoint(str(zip_path), str(pkl_path), factory)

            def act(obs, step, model=model, normalizer=normalizer):
                return model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]

        tracks, actions = [], []
        for seed in seeds:
            env = factory()
            taken: list[np.ndarray] = []
            t0 = time.time()
            ep = run_tracked_episode(
                env, MID, seed=seed, action_fn=act, post_step=lambda e, s, a: taken.append(np.asarray(a, dtype=np.float64))
            )
            env.close()
            acts = np.asarray(taken)
            row: dict = {
                "who": who, "seed": seed, "stood": ep.stood, "steps": ep.steps, "reason": ep.termination_reason,
                "wall_s": round(time.time() - t0, 2), "zip": str(zip_path) if zip_path else None,
                "cmd_mean_from_2s": acts[200:].mean(axis=0).round(6).tolist() if len(acts) > 200 else None,
                "cmd_sat_share_from_2s": (np.abs(acts[200:]) >= 0.99).mean(axis=0).round(4).tolist() if len(acts) > 200 else None,
                "metrics": {},
            }
            if ep.stood:
                series = stance_track_series(ep.track)
                for w in WINDOWS:
                    row["metrics"][str(w)] = stance_track_quality(ep.track, window_start_s=w, series=series)
            out.write(json.dumps(row) + "\n")
            out.flush()
            tracks.append(ep.track)
            actions.append(acts)
        save_tracks(args.out / "tracks" / f"{who}.npz", tracks, seeds, actions)
        print(f"{who}: {sum(1 for tr in tracks if len(tr.t) == 1001)}/{len(seeds)} full tracks, {time.time() - t_who:.1f} s", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
