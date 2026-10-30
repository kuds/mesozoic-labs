"""Verifier's own rollouts, in a pristine git-archive copy of main 378ae2f8 (main_tree/), never in the lens's trees.

PYTHONPATH=main_tree python -B run_verify.py MODE OUT
MODE
  equiv     s42_B_1251 3050, s80_A_1311 3059, start_s80 3059, statue 3059: 1,000 steps on main; saves the float64
            track, the actions and the per-sample support analysis (for the lens's POL-3 and POL-8).
  panel     the 26 policies on seeds 3072-3081 (the gate panel's last 10, never run by the lens) and r9 on 3062-3081:
            per-episode metrics at 5 s (stance_track_quality and this file's own creep), jsonl.
  long      2,000-step episodes (max_episode_steps override; the obs has no clock) on the lens's largest-creep and
            largest-lean spawns, with the statue at the same spawns.
Deterministic model.predict(normalizer.normalize_obs(obs)); main's own stance config constructor; checkpoints read only.
"""

from __future__ import annotations

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
from environments.shared.tests.stance_track_helpers import convex_hull, stance_track_quality, stance_track_series

S = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
RUNS = Path(f"{S}/r10round/ecs/runs")
R9 = (f"{S}/prs/r10_work/scope/ckpt/stage1_4194304_steps.zip", f"{S}/prs/r10_work/scope/ckpt/stage1_vecnormalize_4194304_steps.pkl")
MID = ModelId("tyrannosaurus_rex", "anatomical")
END = {"s42": 4_718_592, "s80": 5_767_168}
NAMES = ([f"s42_A_{n}" for n in range(1101, 1152, 10)] + [f"s42_B_{n}" for n in range(1201, 1252, 10)]
         + [f"s80_A_{n}" for n in range(1301, 1352, 10)] + [f"s80_B_{n}" for n in range(1401, 1452, 10)])


def ckpt(who: str) -> tuple[str, str] | None:
    if who == "statue":
        return None
    if who == "r9":
        return R9
    if who.startswith("start_"):
        run, steps = {"start_s42": ("s42_A_1101", 3_145_728), "start_s80": ("s80_A_1301", 4_194_304)}[who]
    else:
        run, steps = who, END[who[:3]]
    m = RUNS / run / "01_stance" / "models"
    return str(m / f"stage1_{steps}_steps.zip"), str(m / f"stage1_vecnormalize_{steps}_steps.pkl")


def factory(**over):
    kw = dict(load_stage_config(MID, "stance")["env_kwargs"])
    kw.update(over)
    return lambda: env_class(MID)(**kw)


_loaded: dict = {}


def policy(who: str, fac):
    pair = ckpt(who)
    if pair is None:
        return None
    if who not in _loaded:
        _loaded.clear()
        model, norm, _ = load_sb3_checkpoint(pair[0], pair[1], fac)
        _loaded[who] = (model, norm)
    model, norm = _loaded[who]
    return lambda obs, step: model.predict(norm.normalize_obs(obs), deterministic=True)[0]


def roll(who: str, seed: int, fac):
    env = fac()
    acts: list = []
    ep = run_tracked_episode(env, MID, seed=seed, action_fn=policy(who, fac),
                             post_step=lambda e, s, a: acts.append(np.asarray(a, dtype=np.float64)))
    env.close()
    return ep, np.asarray(acts)


def creep(series, W: float) -> dict:
    t = series["t"]
    w = t >= t[0] + W - 1e-9
    feet = series["foot_xy"][w].mean(axis=1)
    tw = t[w] - t[w].mean()
    slope = np.linalg.lstsq(np.vstack([tw, np.ones_like(tw)]).T, feet, rcond=None)[0][0]
    f = series["forward"][w].mean(axis=0)
    f /= np.linalg.norm(f)
    left = np.array([-f[1], f[0]])
    return {"creep_mmps": 1000 * float(np.linalg.norm(slope)), "creep_fa_mmps": 1000 * float(slope @ f),
            "creep_lat_mmps": 1000 * float(slope @ left)}


def main() -> int:
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if mode == "equiv":
        fac = factory()
        arrays = {}
        for who, seed in (("s42_B_1251", 3050), ("s80_A_1311", 3059), ("start_s80", 3059), ("statue", 3059)):
            ep, acts = roll(who, seed, fac)
            tr = ep.track
            k = f"{who}_{seed}"
            arrays.update({f"{k}_t": tr.t, f"{k}_com": tr.com, f"{k}_quat": tr.root_quat, f"{k}_feet": tr.foot_pos,
                           f"{k}_load": tr.foot_load, f"{k}_heading": tr.foot_heading, f"{k}_act": acts})
            s = stance_track_series(tr)
            # per-sample support analysis: points, hull area, how many points lie nearer each foot, margins
            npts, area, near_r, near_l = [], [], [], []
            for i, pts in enumerate(tr.support):
                npts.append(len(pts))
                h = convex_hull(pts) if len(pts) else np.zeros((0, 2))
                if h.shape[0] >= 3:
                    x, y = h[:, 0], h[:, 1]
                    area.append(0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))
                else:
                    area.append(0.0)
                d = np.linalg.norm(pts[:, None, :] - tr.foot_pos[i, None, :, :2], axis=2) if len(pts) else np.zeros((0, 2))
                lab = np.argmin(d, axis=1) if len(pts) else np.zeros(0, int)
                near_r.append(int((lab == list(tr.labels).index("r")).sum()))
                near_l.append(int((lab == list(tr.labels).index("l")).sum()))
            arrays.update({f"{k}_npts": np.array(npts), f"{k}_area": np.array(area), f"{k}_near_r": np.array(near_r),
                           f"{k}_near_l": np.array(near_l), f"{k}_com_margin": s["com_margin"], f"{k}_xcom_margin": s["xcom_margin"],
                           f"{k}_com_speed": s["com_speed"], f"{k}_xcom": s["xcom"]})
            m2 = stance_track_quality(tr, window_start_s=2.0, series=s)
            m5 = stance_track_quality(tr, window_start_s=5.0, series=s)
            print(json.dumps({"who": who, "seed": seed, "stood": ep.stood, "labels": tr.labels,
                              "w2": {x: m2[x] for x in ("com_margin_min_m", "xcom_margin_min_m", "xcom_margin_lateral_min_m", "com_lateral_mean_m")},
                              "w5": {x: m5[x] for x in ("com_margin_min_m", "xcom_margin_min_m", "com_lateral_mean_m", "root_roll_mean_deg")}}), flush=True)
        np.savez_compressed(out / "equiv_main.npz", **arrays)
    elif mode == "panel":
        fac = factory()
        f = open(out / "panel.jsonl", "a")
        todo = [(w, list(range(3072, 3082))) for w in ["start_s42", "start_s80"] + NAMES] + [("r9", list(range(3062, 3082)))]
        for who, seeds in todo:
            tw = time.time()
            for seed in seeds:
                ep, acts = roll(who, seed, fac)
                row = {"who": who, "seed": seed, "stood": ep.stood, "steps": ep.steps}
                if ep.stood:
                    s = stance_track_series(ep.track)
                    m = stance_track_quality(ep.track, window_start_s=5.0, series=s)
                    row.update({k: m[k] for k in ("com_lateral_mean_m", "root_roll_mean_deg", "left_load_share_mean", "root_turn_deg",
                                                  "root_turn_from_spawn_deg", "com_speed_rms_mps", "com_margin_min_m", "xcom_margin_min_m")})
                    row.update(creep(s, 5.0))
                    row["hiproll_cmd_r_l"] = [float(acts[200:, 4].mean()), float(acts[200:, 8].mean())]
                f.write(json.dumps(row) + "\n")
                f.flush()
            print(f"{who}: {len(seeds)} episodes {time.time() - tw:.1f} s", flush=True)
        f.close()
    elif mode == "long":
        fac = factory(max_episode_steps=2000)
        f = open(out / "long.jsonl", "a")
        arrays = {}
        for who, seed in (("s80_A_1301", 3047), ("s80_A_1301", 3059), ("s80_B_1431", 3059), ("s80_A_1311", 3059),
                          ("statue", 3047), ("statue", 3059)):
            ep, acts = roll(who, seed, fac)
            row = {"who": who, "seed": seed, "stood": ep.stood, "steps": ep.steps}
            if ep.steps >= 1000:
                s = stance_track_series(ep.track)
                t = s["t"]
                for a, b in ((5, 10), (10, 20), (15, 20)):
                    w = (t >= a - 1e-9) & (t <= b + 1e-9)
                    feet = s["foot_xy"][w].mean(axis=1)
                    tw = t[w] - t[w].mean()
                    slope = np.linalg.lstsq(np.vstack([tw, np.ones_like(tw)]).T, feet, rcond=None)[0][0]
                    row[f"creep_{a}_{b}_mmps"] = 1000 * float(np.linalg.norm(slope))
                    row[f"com_lat_{a}_{b}_mm"] = 1000 * float(s["com_lateral"][w].mean())
                    row[f"roll_{a}_{b}_deg"] = float(np.degrees(s["root_roll"][w].mean()))
                    row[f"speed_rms_{a}_{b}_mmps"] = 1000 * float(np.sqrt(np.mean(s["com_speed"][w] ** 2)))
                row["feet_travel_10_20_mm"] = 1000 * float(np.linalg.norm(s["foot_xy"][t >= 20 - 1e-9].mean(axis=1)[-1] - s["foot_xy"][np.argmin(abs(t - 10))].mean(axis=0)))
                k = f"{who}_{seed}"
                arrays.update({f"{k}_com": ep.track.com[:1001], f"{k}_feet": ep.track.foot_pos[:1001], f"{k}_act": acts[:1000]})
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)
        np.savez_compressed(out / "long_first1000.npz", **arrays)
        f.close()
    print(f"{mode} done in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
