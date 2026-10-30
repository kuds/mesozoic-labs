"""Exploration (no training): the policy's structure and the mirror map at observation 6."""
import json
import numpy as np
import torch
torch.set_num_threads(1)
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class
from environments.shared.symmetry import MirrorMap
F = ("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2/"
     "stance6_r10LTSB_s42/copies/01_stance/")
mid = ModelId("tyrannosaurus_rex", "anatomical")
cfg = load_stage_config(mid, "stance")
factory = lambda: env_class(mid)(**dict(cfg["env_kwargs"]))  # noqa: E731
model, norm, _ = load_sb3_checkpoint(F + "models/stage1_1048576_steps.zip.copy1",
                                     F + "models/stage1_vecnormalize_1048576_steps.pkl.copy1", factory)
p = model.policy
print(type(model).__name__, type(p).__name__, p.share_features_extractor, type(p.features_extractor).__name__,
      type(p.mlp_extractor).__name__, p.mlp_extractor.value_net, p.value_net)
print("gae_lambda", model.gae_lambda, "gamma", model.gamma, "ent", model.ent_coef, "n_updates", model._n_updates)
rec = json.load(open(F + "stage_config.json.copy1"))
from environments.shared.plant_contract import current_plant_identity
mm = MirrorMap.load(mid, current_plant_identity(mid))
print("obs perm[6]", mm.obs.perm[6], "sign[6]", mm.obs.sign[6], "act perm[14]", mm.action.perm[14], mm.action.sign[14])
print("obs names?", [a for a in dir(mm.obs) if not a.startswith('_')])
env = factory()
print(env.unwrapped.model.actuator(14).name)
print("norm eps", norm.epsilon, "clip", norm.clip_obs, "t3 mean/std deg", np.degrees(norm.obs_rms.mean[6]), np.degrees(np.sqrt(norm.obs_rms.var[6] + norm.epsilon)))
