"""Inspect one checkpoint's policy distribution (report-only)."""
import sys, numpy as np, torch
import environments
assert "main_ro" in environments.__file__, environments.__file__
print("environments:", environments.__file__)
from environments.shared.policy_loading import load_sb3_model
m = load_sb3_model(sys.argv[1], device="cpu")
p = m.policy
print(type(m).__mro__[:3], type(p).__mro__[:3])
print("use_sde", getattr(p, "use_sde", None), "squash", getattr(p, "squash_output", None), "dist", type(p.action_dist))
print("action_space", m.action_space)
print("log_std", p.log_std.detach().numpy())
obs = torch.zeros((1,) + m.observation_space.shape)
with torch.no_grad():
    d = p.get_distribution(obs)
print("dist std", d.distribution.stddev.numpy())
print("exp(log_std)", np.exp(p.log_std.detach().numpy()))
