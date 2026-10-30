"""LONG-EPISODE VARIANT of run_base.py (identical loop; adds --steps N, max_episode_steps overridden as the clamp
runner did, and window means over the last 500 steps).  Written after the 10 s illusion results showed the s42 command
still creeping up at 10 s.

Closed-loop deterministic rollouts of the ECS T. rex stance policies, optionally with a pitch illusion.

PYTHONPATH=<policies/code_copy>:<this folder> python -B run_base.py OUT_DIR [--who a,b,...] [--seeds 3042-3061]
                                                                     [--pitch DEG] [--channels grav,acc,prey]

Simulation and network evaluation only; no training; checkpoints opened read only.

The rollout is the policies lens's harness (com_revisit/notes/policies/run_policies.py, copied here as
run_policies_copy.py): the stance config's own constructor, reset noise 0.05, a fresh env per seed, env.reset(seed),
deterministic model.predict(normalizer.normalize_obs(obs)) every step, 1,000 steps, PR #650's EnvStanceSampler after the
reset and after each step.  run_tracked_episode's loop (com_stance_helpers.py) is written out here, as the clamp runner
did, so reward and info can be kept.  With --pitch 0 the observation reaches the policy untouched (the returns are
checked against the clamp runner's unforced arm in analyze.py).

--pitch DEG (test 2, closed loop): every observation the policy acts on is replaced by pp_common.pitch_illusion(obs,
DEG): the projected gravity, accelerometer and prey direction (or only --channels) as if the pelvis were pitched DEG
further nose-up; nothing else changes and the plant is untouched.

Writes OUT_DIR/episodes.jsonl (one row per episode) and OUT_DIR/arrays/<who>.npz (per seed: raw env observations after
steps 500, 600, ..., 1000; the window means of qpos, ctrl and the action filter state; and, at --pitch 0, the policy's
own action at every step, for the open-loop replays of test 3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from environments.shared.tests.com_stance_helpers import EnvStanceSampler, Morphology
from environments.shared.tests.stance_track_helpers import stance_track_quality
from pp_common import ALL_WHO, MID, TAIL3, Pol, make_env, pitch_deg_from_obs, pitch_illusion, stance_ctor

DIAG = ("tail_instability", "tail_home_pose_error", "action_saturation", "forward_z", "tilt_angle", "pelvis_height",
        "neck_posture_error", "leg_home_pose_error", "head_tip_z", "com_lateral_offset_m")
SNAP = (500, 600, 700, 800, 900, 1000)
W0, W1 = 500, 1000  # window: states after steps 501..1000 (0-based index 500..999)


def roll(env, seed: int, pol: Pol | None, pitch: float, channels, horizon: int = 1000):
    u = env.unwrapped
    m, d = u.model, u.data
    sampler = EnvStanceSampler(env, Morphology.from_env(env, MID))
    q3 = int(m.jnt_qposadr[m.joint("tail_3_pitch").id])
    pelvis = int(u.pelvis_id)
    obs, _ = env.reset(seed=seed)
    sampler.sample(env)
    neutral = np.zeros(env.action_space.shape, dtype=np.float32)
    rec = {k: [] for k in ("reward", "own", "raw14", "tail3", "fwdz", "seen_pitch", "qpos", "ctrl", "filt")}
    infos = []
    snaps = {}
    terminated = truncated = False
    step = 0
    for step in range(1, horizon + 1):
        seen = obs if pitch == 0.0 else pitch_illusion(obs, pitch, channels)
        if pol is None:
            own = neutral
            r14 = 0.0
        else:
            own = pol.act(seen)
            r14 = float(pol.raw_mean(seen[None, :])[0, TAIL3]) if (step % 10 == 0) else float("nan")
        rec["seen_pitch"].append(float(pitch_deg_from_obs(seen)))
        obs, reward, terminated, truncated, info = env.step(own)
        sampler.sample(env)
        rec["reward"].append(float(reward))
        rec["own"].append(np.asarray(own, dtype=np.float64).copy())
        rec["raw14"].append(r14)
        rec["tail3"].append(float(d.qpos[q3]))
        rec["fwdz"].append(float(d.xmat[pelvis].reshape(3, 3)[2, 0]))
        rec["qpos"].append(d.qpos.copy())
        rec["ctrl"].append(d.ctrl.copy())
        rec["filt"].append(np.asarray(u._action_filter_state, dtype=np.float64).copy())
        infos.append({k: float(v) for k, v in info.items() if isinstance(v, (bool, int, float, np.number))})
        if step in SNAP:
            snaps[step] = np.array(obs, dtype=np.float32, copy=True)
        if terminated or truncated:
            break
    return sampler.track(), step, bool(terminated), {k: np.asarray(v) for k, v in rec.items()}, infos, snaps


def bins(x: np.ndarray, edges) -> list[float]:
    return [float(np.nanmean(x[a:b])) if b <= len(x) else float("nan") for a, b in edges]


EDGES = [(i, i + 25) for i in range(0, 400, 25)] + [(i, i + 100) for i in range(400, 1000, 100)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--who", default=",".join(ALL_WHO))
    ap.add_argument("--seeds", default="3042-3061")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--channels", default="grav,acc,prey")
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    channels = tuple(c for c in a.channels.split(",") if c)
    (a.out / "arrays").mkdir(parents=True, exist_ok=True)
    ctor = stance_ctor()
    ctor["max_episode_steps"] = int(a.steps)
    factory = lambda: make_env(ctor)  # noqa: E731
    (a.out / "meta.json").write_text(json.dumps({"argv": sys.argv, "seeds": seeds, "pitch": a.pitch,
                                                 "channels": channels, "snap_steps": SNAP, "window": [W0, W1],
                                                 "bins": EDGES}, indent=1))
    out = open(a.out / "episodes.jsonl", "a")
    for who in a.who.split(","):
        npz = a.out / "arrays" / f"{who}.npz"
        if npz.exists():
            print(who, "done already", flush=True)
            continue
        if who == "statue" and a.pitch != 0.0:
            continue
        t0 = time.time()
        pol = None if who == "statue" else Pol(who, factory)
        save = {"seeds": np.array(seeds), "snap_steps": np.array(SNAP)}
        snaps_all, qm, cm, fm, acts = [], [], [], [], []
        for seed in seeds:
            env = factory()
            tr, steps, term, rec, infos, snaps = roll(env, seed, pol, a.pitch, channels, horizon=a.steps)
            env.close()
            w = slice(W0, min(W1, steps))
            keys = sorted(k for k in infos[0] if k.startswith("reward_"))
            q = rec["qpos"][w].mean(axis=0)
            q[3:7] /= np.linalg.norm(q[3:7])
            row = {
                "who": who, "seed": seed, "pitch_illusion_deg": a.pitch, "channels": list(channels), "steps": steps,
                "terminated": term, "stood": (not term) and steps == 1000,
                "return": float(np.sum(rec["reward"])), "return_200_1000": float(np.sum(rec["reward"][200:1000])),
                "terms_500_1000": {k: float(sum(infos[i][k] for i in range(W0, min(W1, steps)))) for k in keys},
                "terms_200_1000": {k: float(sum(infos[i][k] for i in range(200, min(1000, steps)))) for k in keys},
                "diag_500_1000": {k: float(np.mean([infos[i][k] for i in range(W0, min(W1, steps))]))
                                  for k in DIAG if k in infos[0]},
                "own_mean": rec["own"][w].mean(axis=0).round(6).tolist(),
                "own14_sat_share": float((rec["own"][w, TAIL3] >= 0.99).mean()),
                "raw14_mean": float(np.nanmean(rec["raw14"][w])),
                "tail3_deg": math.degrees(float(rec["tail3"][w].mean())),
                "pitch_deg": math.degrees(math.asin(float(rec["fwdz"][w].mean()))),
                "seen_pitch_deg": float(np.mean(rec["seen_pitch"][w])),
                "own14_bins": bins(rec["own"][:, TAIL3], EDGES),
                "tail3_deg_bins": [math.degrees(x) for x in bins(rec["tail3"], EDGES)],
                "pitch_deg_bins": [math.degrees(math.asin(x)) for x in bins(rec["fwdz"], EDGES)],
                "late_window": [a.steps - 500, a.steps],
                "own14_late": float(rec["own"][a.steps - 500:, TAIL3].mean()) if steps == a.steps else None,
                "own14_sat_late": float((rec["own"][a.steps - 500:, TAIL3] >= 0.99).mean()) if steps == a.steps else None,
                "tail3_deg_late": math.degrees(float(rec["tail3"][a.steps - 500:].mean())) if steps == a.steps else None,
                "pitch_deg_late": math.degrees(math.asin(float(rec["fwdz"][a.steps - 500:].mean()))) if steps == a.steps else None,
                "seen_pitch_deg_late": float(np.mean(rec["seen_pitch"][a.steps - 500:])) if steps == a.steps else None,
                "own14_bins_1s": [float(rec["own"][i:i + 100, TAIL3].mean()) for i in range(0, steps, 100)],
                "tail3_deg_bins_1s": [math.degrees(float(rec["tail3"][i:i + 100].mean())) for i in range(0, steps, 100)],
                "pitch_deg_bins_1s": [math.degrees(math.asin(float(rec["fwdz"][i:i + 100].mean()))) for i in range(0, steps, 100)],
                "seen_pitch_deg_bins_1s": [float(np.mean(rec["seen_pitch"][i:i + 100])) for i in range(0, steps, 100)],
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            snaps_all.append(np.stack([snaps.get(s, np.full(63, np.nan, np.float32)) for s in SNAP]))
            qm.append(q)
            cm.append(rec["ctrl"][w].mean(axis=0))
            fm.append(rec["filt"][w].mean(axis=0))
            pass
        save["NOTE"] = np.array("long variant: obs snaps at steps 500..1000 only")
        save["obs_snaps"] = np.stack(snaps_all).astype(np.float32)
        save["qpos_mean"] = np.stack(qm)
        save["ctrl_mean"] = np.stack(cm)
        save["filt_mean"] = np.stack(fm)
        if acts:
            save["own_actions"] = np.stack(acts)
        np.savez_compressed(npz, **save)
        print(f"{who} pitch {a.pitch:+.2f}: {len(seeds)} episodes {time.time() - t0:.1f} s", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
