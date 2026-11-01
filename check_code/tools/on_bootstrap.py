"""No-update check of SB3's time-limit bootstrap with the phase: collect_rollouts (no train()) over a 1000-step
truncation on the real stance env; the bootstrapped reward must use V(terminal obs with phi(1000))."""
import sys

WT = "/home/user/work/pc_branch"
sys.path.insert(0, WT)
import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

import environments.shared.train_base as tb  # noqa: E402
from environments.shared.config import load_all_stages  # noqa: E402
from environments.shared.critic_phase import CriticPhaseSpec  # noqa: E402
from environments.shared.critic_phase_ppo import new_model_kwargs  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.species_names import require_model_id  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest  # noqa: E402

mid = require_model_id("tyrannosaurus_rex", "anatomical")
cfg = get_species_config(mid)
st = load_all_stages(mid)
ref = load_stage_manifest(mid).resolve("stance").reference
env = tb.create_vec_env(cfg, st, ref, 1, seed=11, algorithm="ppo", gamma=0.98, plant_identity=current_plant_identity(mid),
                        critic_phase_gamma=0.98)
spec = CriticPhaseSpec(200, 0.98)
model = PPO(env=env, n_steps=1010, batch_size=101, gamma=0.98, seed=3, device="cpu",
            **new_model_kwargs(spec, {"net_arch": [512, 256], "log_std_init": -20.0}))
with th.no_grad():
    w = model.policy.mlp_extractor.value_net[0].weight
    w[:, 63:] = th.randn(w.shape[0], 3, generator=th.Generator().manual_seed(0))
seen = {}


class Grab(BaseCallback):
    def _on_step(self):
        for i, (d, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if d:
                seen.setdefault("t", self.locals["n_steps"])
                seen.setdefault("info", dict(info))
                seen.setdefault("raw_reward", float(self.locals["rewards"][i]))
        return True


model._setup_learn(1010, Grab())
cb = Grab()
cb.init_callback(model)
model.collect_rollouts(env, cb, model.rollout_buffer, 1010)
t, info = seen["t"], seen["info"]
term = np.asarray(info["terminal_observation"], dtype=np.float32)
print("truncated at step index", t, "info step", info["step"], "TimeLimit.truncated", info.get("TimeLimit.truncated"))
print("terminal obs phase", term[63:].tolist())
with th.no_grad():
    def V(x):
        return float(model.policy.predict_values(th.as_tensor(x[None])).item())
    v1000 = V(term)
    alt = term.copy(); alt[63:] = spec.matrix([0])[0]
    v0 = V(alt)
    alt2 = term.copy(); alt2[63:] = spec.matrix([999])[0]
    v999 = V(alt2)
buf_r = float(model.rollout_buffer.rewards[t, 0])
exp = np.float32(seen["raw_reward"]) + np.float32(0.98) * np.float32(v1000)
print("buffer reward", buf_r, "expected r + gamma*V(phi(1000))", float(exp), "with phi(0)", seen["raw_reward"] + 0.98 * v0,
      "with phi(999)", seen["raw_reward"] + 0.98 * v999)
print("CHECK bootstrap uses phi(1000):", abs(buf_r - float(exp)) <= 1e-5 and abs(v1000 - v0) > 1e-4)
print("next obs after truncation phase (k=0):", model._last_obs[0, 63:].tolist() if t == 1009 else "n/a")
