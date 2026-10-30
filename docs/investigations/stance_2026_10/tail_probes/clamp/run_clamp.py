"""Tail clamp test: the ECS end policies with the tail_3_pitch_act command replaced, against the same policies unforced.

PYTHONPATH=<policies/code_copy> python -B run_clamp.py OUT_DIR --group s42|s80 [--force V] [--steps 1000]
                                                       [--seeds 3042-3061] [--only name,name] [--compare LENS_RUN_DIR]

Exploratory and report-only (not pre-registered).  No training: checkpoints are opened read only.

The rollout is the policies lens's (com_revisit/notes/policies/run_policies.py, copied here, not imported): the stance
config's own env constructor ``env_class(mid)(**load_stage_config(mid, "stance")["env_kwargs"])`` (reset noise 0.05),
a fresh env per seed, ``env.reset(seed=seed)``, the deterministic ``model.predict(normalizer.normalize_obs(obs),
deterministic=True)[0]`` every step, one sample of PR #650's EnvStanceSampler after the reset and after each step.  The
loop is run_tracked_episode's own loop (com_stance_helpers.py) written out here so that the per-step reward and info can
be kept; the --compare check shows the tracks bit-identical to the lens's.

--force V: the override D-9 used (episode_trace.trace_episode's ``holds``): the policy's own action every step, with
only entry 14 (tail_3_pitch_act) replaced by V before env.step.  The env then clips it, low-passes it (10 Hz), and maps
it as a residual about the home control (0 -> home ctrl 0 rad, +1 -> ctrl +0.2094 rad); the reward's raw-command terms
(saturation, jerk, smoothness: action_penalty_source = "raw") price the replaced command, as they did in D-9.

--steps N other than 1000 overrides max_episode_steps (the observation has no clock).

Per episode (episodes.jsonl): termination, return (full, steps 200-end as D-9's window return, steps 500-end), the
stance_track_quality metrics at windows from 2, 5, 9 s (and 10, 15 s on long runs), per-window means of every reward_*
info term and of selected info diagnostics, joint angles, applied ctrl, actuator force, the applied command and the
policy's own (pre-override) command, the tail tip's height and fore-aft offset.  Tracks (tracks/<who>.npz, float32) as
the lens saved them plus the per-step reward, the four tail joint angles and the policy's own tail_3 command; with
--compare, the unforced tracks are not saved (they are the lens's, as the check shows) and compare.jsonl records the
check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class
from environments.shared.tests.com_stance_helpers import EnvStanceSampler, Morphology, TrackedEpisode
from environments.shared.tests.stance_track_helpers import stance_track_quality, stance_track_series

RUNS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs")
NAMES = (
    [f"s42_A_{n}" for n in range(1101, 1152, 10)]
    + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
    + [f"s80_A_{n}" for n in range(1301, 1352, 10)]
    + [f"s80_B_{n}" for n in range(1401, 1452, 10)]
)
END = {"s42": 4_718_592, "s80": 5_767_168}
MID = ModelId("tyrannosaurus_rex", "anatomical")
TAIL3 = 14  # tail_3_pitch_act (checked against the model's actuator names below)
TAIL_JOINTS = ("tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch")
DIAG = ("tail_instability", "tail_home_pose_error", "action_saturation", "action_jerk", "action_delta", "tilt_angle",
        "forward_z", "pelvis_height", "leg_home_pose_error", "neck_posture_error", "com_lateral_offset_m",
        "foot_load_imbalance", "lateral_lean_quality", "drift_distance", "abs_speed", "pelvis_angular_vel",
        "r_foot_contact", "l_foot_contact", "stance_width", "foot_stagger_m", "peak_foot_force_bw")


def checkpoint(name: str) -> tuple[Path, Path]:
    steps = END[name[:3]]
    models = RUNS / name / "01_stance" / "models"
    return models / f"stage1_{steps}_steps.zip", models / f"stage1_vecnormalize_{steps}_steps.pkl"


def roll(env, *, seed: int, act, force: float | None, horizon: int):
    """run_tracked_episode's loop (com_stance_helpers.py:307-336), keeping reward, info and per-step extras."""
    unwrapped = getattr(env, "unwrapped", env)
    model, data = unwrapped.model, unwrapped.data
    sampler = EnvStanceSampler(env, Morphology.from_env(env, MID))
    tail_q = np.array([model.jnt_qposadr[model.joint(n).id] for n in TAIL_JOINTS])
    hinge_q = np.arange(7, model.nq)
    tip = int(unwrapped.tail_tip_site_id)
    pelvis = int(unwrapped.pelvis_id)
    obs, _ = env.reset(seed=seed)
    sampler.sample(env)
    terminated = truncated = False
    reason = None
    step = 0
    rec: dict[str, list] = {k: [] for k in ("reward", "own", "applied", "ctrl", "force", "qpos", "tip", "info")}
    for step in range(1, horizon + 1):
        own = np.asarray(act(obs, step))
        if force is None:
            action = own
        else:
            action = own.copy()
            action[TAIL3] = force
        obs, reward, terminated, truncated, info = env.step(action)
        sampler.sample(env)
        rec["reward"].append(float(reward))
        rec["own"].append(np.asarray(own, dtype=np.float64))
        rec["applied"].append(np.asarray(action, dtype=np.float64))
        rec["ctrl"].append(np.asarray(data.ctrl, dtype=np.float64).copy())
        rec["force"].append(np.asarray(data.actuator_force, dtype=np.float64).copy())
        rec["qpos"].append(np.asarray(data.qpos[hinge_q], dtype=np.float64).copy())
        # tail tip: height above the floor, and its offset from the pelvis along the pelvis's horizontal forward axis
        fwd = data.xmat[pelvis].reshape(3, 3)[:2, 0]
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-12)
        d = data.site_xpos[tip] - data.xpos[pelvis]
        rec["tip"].append(np.array([float(data.site_xpos[tip][2]), float(d[:2] @ fwd)]))
        rec["info"].append({k: float(v) for k, v in info.items() if isinstance(v, (bool, int, float, np.number))})
        if terminated:
            reason = str(info.get("termination_reason", "unknown"))
        if terminated or truncated:
            break
    ep = TrackedEpisode(track=sampler.track(), steps=step, horizon=horizon, terminated=bool(terminated),
                        truncated=bool(truncated), termination_reason=reason, dt=float(unwrapped.dt))
    arrays = {k: np.asarray(v) for k, v in rec.items() if k != "info"}
    return ep, arrays, rec["info"], tail_q - 7


def window_means(arrays, infos, lo: int, hi: int, names: list[str]) -> dict:
    """Means over steps lo..hi-1 (0-based step index; step index i is the state after step i+1)."""
    out: dict = {}
    n = len(infos)
    lo, hi = min(lo, n), min(hi, n)
    if hi - lo < 1:
        return out
    sl = slice(lo, hi)
    keys = sorted(k for k in infos[0] if k.startswith("reward_"))
    out["reward_sum"] = float(np.sum(arrays["reward"][sl]))
    out["reward_terms_sum"] = {k: float(sum(infos[i][k] for i in range(lo, hi))) for k in keys}
    out["diag_mean"] = {k: float(np.mean([infos[i][k] for i in range(lo, hi)])) for k in DIAG if k in infos[0]}
    out["applied_mean"] = arrays["applied"][sl].mean(axis=0).round(7).tolist()
    out["own_mean"] = arrays["own"][sl].mean(axis=0).round(7).tolist()
    out["own_sat_share"] = (np.abs(arrays["own"][sl]) >= 0.99).mean(axis=0).round(4).tolist()
    out["ctrl_mean"] = arrays["ctrl"][sl].mean(axis=0).round(7).tolist()
    out["force_mean"] = arrays["force"][sl].mean(axis=0).round(5).tolist()
    out["qpos_mean"] = arrays["qpos"][sl].mean(axis=0).round(7).tolist()
    out["qpos_sd"] = arrays["qpos"][sl].std(axis=0).round(7).tolist()
    out["tip_mean"] = arrays["tip"][sl].mean(axis=0).round(6).tolist()
    out["steps"] = hi - lo
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--group", required=True, choices=("s42", "s80"))
    parser.add_argument("--force", type=float, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", default="3042-3061")
    parser.add_argument("--only", default="")
    parser.add_argument("--compare", type=Path, default=None, help="the lens's run dir: check bit identity, keep no tracks")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    only = [w for w in args.only.split(",") if w]
    names = only or [n for n in NAMES if n.startswith(args.group)]
    if args.compare is not None and (args.force is not None or args.steps != 1000):
        raise SystemExit("--compare is for the unforced 1,000-step arm only")
    (args.out / "tracks").mkdir(parents=True, exist_ok=True)
    ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
    if args.steps != 1000:
        ctor["max_episode_steps"] = int(args.steps)
    factory = lambda: env_class(MID)(**ctor)  # noqa: E731
    probe = factory()
    anames = [probe.unwrapped.model.actuator(i).name for i in range(probe.unwrapped.model.nu)]
    jnames = [probe.unwrapped.model.joint(j).name for j in range(1, probe.unwrapped.model.njnt)]
    probe.close()
    if anames[TAIL3] != "tail_3_pitch_act":
        raise SystemExit(f"actuator {TAIL3} is {anames[TAIL3]}")
    lens = {}
    if args.compare is not None:
        for line in open(args.compare / "episodes.jsonl"):
            r = json.loads(line)
            lens[(r["who"], r["seed"])] = r
    windows = (2.0, 5.0, 9.0) if args.steps == 1000 else (2.0, 5.0, 9.0, 10.0, 15.0)
    step_windows = {"0_end": (0, args.steps), "200_end": (200, args.steps), "500_end": (500, args.steps),
                    "500_1000": (500, 1000)}
    if args.steps > 1000:
        step_windows.update({"1000_end": (1000, args.steps), "1500_end": (1500, args.steps)})
    meta = {"actuators": anames, "hinge_joints": jnames, "force": args.force, "steps": args.steps, "seeds": seeds,
            "group": args.group, "ctor_max_episode_steps": ctor.get("max_episode_steps"), "windows_s": windows,
            "step_windows": step_windows, "argv": sys.argv}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    out = open(args.out / "episodes.jsonl", "a")
    cmp_out = open(args.out / "compare.jsonl", "a") if args.compare is not None else None
    for who in names:
        if (args.out / "tracks" / f"{who}.done").exists():
            print(who, "done already", flush=True)
            continue
        t_who = time.time()
        zip_path, pkl_path = checkpoint(who)
        model, normalizer, _ = load_sb3_checkpoint(str(zip_path), str(pkl_path), factory)

        def act(obs, step, model=model, normalizer=normalizer):
            return model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]

        lens_npz = np.load(args.compare / "tracks" / f"{who}.npz") if args.compare is not None else None
        save: dict[str, np.ndarray] = {"seeds": np.array(seeds)}
        n_full = 0
        for i, seed in enumerate(seeds):
            env = factory()
            t0 = time.time()
            ep, arrays, infos, tail_cols = roll(env, seed=seed, act=act, force=args.force, horizon=args.steps)
            env.close()
            tr = ep.track
            n_full += int(len(tr.t) == args.steps + 1)
            row: dict = {"who": who, "seed": seed, "force": args.force, "stood": ep.stood, "steps": ep.steps,
                         "terminated": ep.terminated, "reason": ep.termination_reason,
                         "end_time_s": float(tr.t[-1] - tr.t[0]), "wall_s": round(time.time() - t0, 2),
                         "return": float(np.sum(arrays["reward"])), "metrics": {}, "win": {}}
            for key, (a, b) in step_windows.items():
                row["win"][key] = window_means(arrays, infos, a, b, anames)
            if ep.stood:
                series = stance_track_series(tr)
                for w in windows:
                    row["metrics"][str(w)] = stance_track_quality(tr, window_start_s=w, series=series)
                if args.steps > 1000:
                    # the 5-10 s window of a long episode: the track's first 1,001 samples
                    from dataclasses import replace

                    cut = replace(tr, t=tr.t[:1001], com=tr.com[:1001], root_quat=tr.root_quat[:1001],
                                  foot_pos=tr.foot_pos[:1001], foot_load=None if tr.foot_load is None else tr.foot_load[:1001],
                                  foot_heading=None if tr.foot_heading is None else tr.foot_heading[:1001],
                                  support=None if tr.support is None else tr.support[:1001],
                                  com_vel=None if tr.com_vel is None else tr.com_vel[:1001])
                    row["metrics"]["5.0_to10"] = stance_track_quality(cut, window_start_s=5.0)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if lens_npz is not None:
                ref = lens.get((who, seed))
                checks = {
                    "t": np.array_equal(tr.t, lens_npz[f"t_{i}"]),
                    "com": np.array_equal(tr.com.astype(np.float32), lens_npz[f"com_{i}"]),
                    "quat": np.array_equal(tr.root_quat.astype(np.float32), lens_npz[f"quat_{i}"]),
                    "feet": np.array_equal(tr.foot_pos.astype(np.float32), lens_npz[f"feet_{i}"]),
                    "load": np.array_equal(tr.foot_load.astype(np.float32), lens_npz[f"load_{i}"]),
                    "heading": np.array_equal(tr.foot_heading.astype(np.float32), lens_npz[f"heading_{i}"]),
                    "action": np.array_equal(arrays["applied"].astype(np.float32), lens_npz[f"action_{i}"]),
                    "seed_match": int(lens_npz["seeds"][i]) == seed,
                    "stood": ref is not None and ref["stood"] == ep.stood and ref["steps"] == ep.steps,
                    "metrics_exact": ref is not None and json.dumps(ref["metrics"].get("5.0"), sort_keys=True)
                    == json.dumps(row["metrics"].get("5.0"), sort_keys=True)
                    and json.dumps(ref["metrics"].get("2.0"), sort_keys=True) == json.dumps(row["metrics"].get("2.0"), sort_keys=True)
                    and json.dumps(ref["metrics"].get("9.0"), sort_keys=True) == json.dumps(row["metrics"].get("9.0"), sort_keys=True),
                    "cmd_mean_from_2s": ref is not None
                    and ref["cmd_mean_from_2s"] == arrays["applied"][200:].mean(axis=0).round(6).tolist(),
                }
                cmp_out.write(json.dumps({"who": who, "seed": seed, "all": all(checks.values()), **checks}) + "\n")
                cmp_out.flush()
            else:
                save[f"t_{i}"] = tr.t
                save[f"com_{i}"] = tr.com.astype(np.float32)
                save[f"quat_{i}"] = tr.root_quat.astype(np.float32)
                save[f"feet_{i}"] = tr.foot_pos.astype(np.float32)
                save[f"load_{i}"] = tr.foot_load.astype(np.float32)
                save[f"heading_{i}"] = tr.foot_heading.astype(np.float32)
                save[f"action_{i}"] = arrays["applied"].astype(np.float32)
            # the extras, kept in both modes
            save[f"reward_{i}"] = arrays["reward"].astype(np.float32)
            save[f"tailq_{i}"] = arrays["qpos"][:, tail_cols].astype(np.float32)
            save[f"own_tail3_{i}"] = arrays["own"][:, TAIL3].astype(np.float32)
            save[f"tip_{i}"] = arrays["tip"].astype(np.float32)
        save["labels"] = np.array(tr.labels)
        save["floor_z"] = np.array(tr.floor_z)
        save["gravity"] = np.array(tr.gravity)
        np.savez_compressed(args.out / "tracks" / f"{who}.npz", **save)
        (args.out / "tracks" / f"{who}.done").write_text("ok\n")
        print(f"{who}: {n_full}/{len(seeds)} full tracks, {time.time() - t_who:.1f} s", flush=True)
    out.close()
    if cmp_out is not None:
        cmp_out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
