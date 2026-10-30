"""Does main 378ae2f8's T. rex stance env run the same episode as the sealed run-branch copy (8737cd3d)?

PYTHONPATH=<tree> python -B equiv.py OUT.npz
In whichever tree is on PYTHONPATH, with that tree's own stance config (env_class(mid)(**env_kwargs)): the statue and
two end policies (s42_A_1101, rail; s80_B_1401, no rail) on seeds 3042 and 3043, 1,000 steps, deterministic.
Saves the float64 CoM, root quaternion, feet, the observations' sha256 per episode and the actions, for a
bitwise comparison between the two trees (compare_equiv.py).
"""

import hashlib
import sys

import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class
from environments.shared.tests.com_stance_helpers import run_tracked_episode

RUNS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/runs"
MID = ModelId("tyrannosaurus_rex", "anatomical")
ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
factory = lambda: env_class(MID)(**ctor)  # noqa: E731
policies = {
    "statue": None,
    "s42_A_1101": (f"{RUNS}/s42_A_1101/01_stance/models/stage1_4718592_steps.zip", f"{RUNS}/s42_A_1101/01_stance/models/stage1_vecnormalize_4718592_steps.pkl"),
    "s80_B_1401": (f"{RUNS}/s80_B_1401/01_stance/models/stage1_5767168_steps.zip", f"{RUNS}/s80_B_1401/01_stance/models/stage1_vecnormalize_5767168_steps.pkl"),
}
arrays = {}
for who, pair in policies.items():
    if pair is None:
        act = None
    else:
        model, normalizer, _ = load_sb3_checkpoint(pair[0], pair[1], factory)

        def act(obs, step, model=model, normalizer=normalizer):
            return model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]

    for seed in (3042, 3043):
        env = factory()
        obs_hash = hashlib.sha256()
        acts = []

        def post(e, s, a):
            acts.append(np.asarray(a, dtype=np.float64))

        def wrapped(obs, step):
            obs_hash.update(np.asarray(obs, dtype=np.float64).tobytes())
            return act(obs, step) if act is not None else np.zeros(env.action_space.shape, dtype=np.float32)

        ep = run_tracked_episode(env, MID, seed=seed, action_fn=wrapped, post_step=post)
        env.close()
        key = f"{who}_{seed}"
        arrays[f"{key}_com"] = ep.track.com
        arrays[f"{key}_quat"] = ep.track.root_quat
        arrays[f"{key}_feet"] = ep.track.foot_pos
        arrays[f"{key}_load"] = ep.track.foot_load
        arrays[f"{key}_act"] = np.asarray(acts)
        arrays[f"{key}_obs_sha256"] = np.array(obs_hash.hexdigest())
        arrays[f"{key}_steps"] = np.array(ep.steps)
        print(key, ep.steps, ep.stood, obs_hash.hexdigest()[:16], flush=True)
np.savez(sys.argv[1], **arrays)
