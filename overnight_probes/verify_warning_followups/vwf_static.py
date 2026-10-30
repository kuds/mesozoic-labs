"""Verifier: b*_S = 1 - median static tail_3 command at tail input 5.19 deg (warning_margin wm_static definition),
own forward pass (manual normalisation, mlp_extractor + action_net), two checkpoints.  Network evaluation only."""
import math, sys
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
ST = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_warning/states")
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.policy_loading import load_sb3_checkpoint
from environments.shared.species_registry import env_class
mid = ModelId("tyrannosaurus_rex", "anatomical"); kw = dict(load_stage_config(mid, "stance")["env_kwargs"])
fac = lambda: env_class(mid)(**kw)
for run, n in (("s80", 12_001_280), ("s42", 3_145_728), ("s60", 7_340_032)):
    m = F2 / f"stance6_r10LTSB_{run}/copies/01_stance/models"
    z, p = ((m / "stage1_final.zip.copy1", m / "stage1_final_vecnorm.pkl.copy1") if n == 12_001_280 else
            (m / f"stage1_{n}_steps.zip.copy1", m / f"stage1_vecnormalize_{n}_steps.pkl.copy1"))
    model, norm, _ = load_sb3_checkpoint(str(z), str(p), fac)
    mu_, var_ = np.asarray(norm.obs_rms.mean, float), np.asarray(norm.obs_rms.var, float)
    X = np.load(ST / f"{run}_{n}.npz")["X"].astype(float)
    def cmd(theta):
        Y = X.copy(); Y[:, 6] = math.radians(theta)
        xn = np.clip((Y - mu_) / np.sqrt(var_ + norm.epsilon), -norm.clip_obs, norm.clip_obs).astype(np.float32)
        with torch.no_grad():
            f = model.policy.extract_features(torch.as_tensor(xn), model.policy.pi_features_extractor)
            a = model.policy.action_net(model.policy.mlp_extractor.forward_actor(f)).numpy()[:, 14]
        return float(np.median(np.clip(a, -1, 1)))
    c0 = cmd(0.0); c1 = cmd(1.0)
    print(f"{run} {n}: n states {len(X)}; c(0) {c0:+.4f}; c(5.19) {cmd(5.19):+.4f} -> b*_S {1 - cmd(5.19):.4f}; c(12) {cmd(12.0):+.4f}")
