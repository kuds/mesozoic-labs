"""Verifier: the r9 checkpoint (stance5_LTSB_s42, 4,194,304 steps) on its OWN training plant, physics r9.

PYTHONPATH=r9_tree python -B run_r9_plant.py OUT.jsonl
r9_tree is a git archive of 3ffc18e9 (main before physics r10, physics_revision 9 for the T. rex) with PR #650's two
reduction files copied in from main 378ae2f8.  The r9 checkpoint and the neutral-action statue, deterministic, that
tree's own stance config constructor (reset noise 0.05), 1,000 steps, seeds 3042-3061; window from 5 s (and 2 s).
Tests POL-12's explanation of why r9 leans and the ECS policies do not.
"""

import json
import sys
import time

import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class
from environments.shared.tests.com_stance_helpers import run_tracked_episode
from environments.shared.tests.stance_track_helpers import stance_track_quality, stance_track_series

S = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ZIP = f"{S}/prs/r10_work/scope/ckpt/stage1_4194304_steps.zip"
PKL = f"{S}/prs/r10_work/scope/ckpt/stage1_vecnormalize_4194304_steps.pkl"
MID = ModelId("tyrannosaurus_rex", "anatomical")
kw = dict(load_stage_config(MID, "stance")["env_kwargs"])
fac = lambda: env_class(MID)(**kw)  # noqa: E731
model, norm, _ = load_sb3_checkpoint(ZIP, PKL, fac)
out = open(sys.argv[1], "w")
probe = fac()
print("plant physics_revision in this tree:", getattr(probe.unwrapped, "physics_revision", "n/a"), flush=True)
probe.close()
for who in ("r9_on_r9_plant", "statue_on_r9_plant"):
    t0 = time.time()
    for seed in range(3042, 3062):
        env = fac()
        act = None if who.startswith("statue") else (lambda obs, step: model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        ep = run_tracked_episode(env, MID, seed=seed, action_fn=act)
        env.close()
        row = {"who": who, "seed": seed, "stood": ep.stood, "steps": ep.steps}
        if ep.stood:
            s = stance_track_series(ep.track)
            for w in (2.0, 5.0):
                m = stance_track_quality(ep.track, window_start_s=w, series=s)
                row[str(w)] = {k: m[k] for k in ("com_lateral_mean_m", "root_roll_mean_deg", "left_load_share_mean", "com_speed_rms_mps")}
        out.write(json.dumps(row) + "\n")
        out.flush()
    print(who, f"{time.time() - t0:.1f} s", flush=True)
out.close()
