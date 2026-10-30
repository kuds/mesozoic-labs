"""Does this folder's override reproduce D-9?  Read only; no training.

PYTHONPATH=<policies/code_copy> python -B d9_repro.py > d9_repro.txt

Runs run_clamp.roll (the policy's own deterministic action every step with entry 14, tail_3_pitch_act, replaced by 0)
on the checkpoint D-9 held at its onset (stance6_r10LTSB_s42 stage1_3,670,016, the copy the round's D-9 read), seeds
3042-3061, fresh env per seed, and compares each seed's full return and window return (steps 200-999) with D-9's
d9_summary.json (r10round/f2/d9_d10/stance6_r10LTSB_s42/d9_3670016_hold_tail_3_pitch_act_0, read only).
"""

import json
from pathlib import Path

import numpy as np

import run_clamp as rc
from environments.shared.config import load_stage_config
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class

M = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2/"
         "stance6_r10LTSB_s42/copies/01_stance/models")
D9 = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/f2/d9_d10/"
          "stance6_r10LTSB_s42/d9_3670016_hold_tail_3_pitch_act_0/d9_summary.json")


def main() -> None:
    ref = {e["seed"]: e for e in json.loads(D9.read_text())["window_return_vs_deterministic_m1"]["per_episode"]}
    ctor = dict(load_stage_config(rc.MID, "stance")["env_kwargs"])
    factory = lambda: env_class(rc.MID)(**ctor)  # noqa: E731
    model, norm, _ = load_sb3_checkpoint(str(M / "stage1_3670016_steps.zip.copy1"),
                                         str(M / "stage1_vecnormalize_3670016_steps.pkl.copy1"), factory)
    act = lambda obs, step: model.predict(norm.normalize_obs(obs), deterministic=True)[0]  # noqa: E731
    dmax_full = dmax_win = 0.0
    diffs = []
    for seed in range(3042, 3062):
        env = factory()
        ep, arrays, infos, tail_cols = rc.roll(env, seed=seed, act=act, force=0.0, horizon=1000)
        env.close()
        full = float(np.sum(arrays["reward"]))
        win = float(np.sum(arrays["reward"][200:]))
        r = ref[seed]
        dmax_full = max(dmax_full, abs(full - r["return"]))
        dmax_win = max(dmax_win, abs(win - r["window_return"]))
        diffs.append(win - r["baseline_window_return"])
        print(f"seed {seed}: return {full:.6f} (D-9 {r['return']:.6f}); window {win:.6f} (D-9 {r['window_return']:.6f}); "
              f"held tail_3 joint {np.mean(arrays['qpos'][200:, tail_cols[3]]):.6f} rad (D-9 {r['held_actuator_joint_pos_window_mean']:.6f})",
              flush=True)
    print(f"max |return - D-9| {dmax_full:.3e}; max |window return - D-9| {dmax_win:.3e}")
    print(f"window return held - D-9's unheld baseline: mean {np.mean(diffs):+.2f} (D-9: +78.29), {sum(d > 0 for d in diffs)}/20 positive")


if __name__ == "__main__":
    main()
