"""Exploration: load one checkpoint as training would; print its settings; check the bank mu against the probe row."""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_bias")
import eb_common as E  # noqa: E402

run, steps = sys.argv[1], int(sys.argv[2])
t0 = time.time()
b = E.build(run, steps, env_seed=777)
print("config problems:", b.config_problems)
m = b.model
for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "vf_coef", "max_grad_norm", "target_kl",
          "normalize_advantage", "ent_coef", "clip_range_vf", "n_envs", "num_timesteps", "_n_updates", "use_sde"):
    print(k, getattr(m, k, None))
print("clip_range(1)", m.clip_range(1.0), "lr_schedule(rem)", m.lr_schedule(1 - steps / 12_000_000))
print("sym", b.sym)
print(m.policy)
print("share_features_extractor", m.policy.share_features_extractor)
opt = m.policy.optimizer
print(opt)
st = opt.state_dict()
print("opt state keys", list(st["state"].keys())[:5], len(st["state"]))
s0 = st["state"][0]
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in s0.items()})
print("vecnorm", b.vn.training, b.vn.norm_obs, b.vn.norm_reward, b.vn.clip_obs, b.vn.clip_reward, b.vn.gamma, b.vn.epsilon,
      "ret var", float(b.vn.ret_rms.var), "obs count", b.vn.obs_rms.count)
pr = E.probe(run)
print(E.set_probe_schedule(b, pr))
i = b.probe_row
raw = pr["bank_raw_obs"]
nb = b.vn.normalize_obs(raw.astype(np.float32))
with torch.no_grad():
    mu = m.policy.get_distribution(torch.as_tensor(np.asarray(nb, dtype=np.float32))).mode().numpy()
    v = m.policy.predict_values(torch.as_tensor(np.asarray(nb, dtype=np.float32))).numpy().ravel()
print("bank mu14 ckpt ", np.round(mu[:, 14], 5))
print("bank mu14 probe", np.round(pr["bank_action_now"][i, :, 14], 5))
print("max |mu - probe| all actuators", float(np.abs(mu - pr["bank_action_now"][i]).max()))
print("max |V - probe|", float(np.abs(v - pr["bank_value_now"][i]).max()))
print("probe obs mean diff", float(np.abs(b.vn.obs_rms.mean - pr["obs_rms_mean"][i]).max()),
      "var diff", float(np.abs(b.vn.obs_rms.var - pr["obs_rms_var"][i]).max()),
      "ret var probe", float(pr["ret_rms_var"][i]))
print("std ckpt", np.round(torch.exp(m.policy.log_std).detach().numpy(), 4))
print("std probe", np.round(pr["std_per_actuator"][i], 4))
print("seconds", time.time() - t0)
