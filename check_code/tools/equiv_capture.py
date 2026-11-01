"""Run train_base.train on the real T. rex stance stage (tiny config) from TREE and capture, bitwise, every
rollout buffer, every pre-rollout policy/optimizer/VecNormalize state and the final state.
usage: equiv_capture.py TREE OUTDIR on|off SEED N_ENVS N_STEPS BATCH EPOCHS TOTAL"""
import json
import sys

tree, out, switch = sys.argv[1], sys.argv[2], sys.argv[3]
seed, n_envs, n_steps, batch, epochs, total = (int(x) for x in sys.argv[4:10])
load = sys.argv[10] if len(sys.argv) > 10 else None  # optional: resume_same_stage from this zip (+ its _vecnorm.pkl)
sys.path.insert(0, tree)

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm  # noqa: E402

import environments.shared.train_base as tb  # noqa: E402

assert tb.__file__.startswith(tree), tb.__file__
from environments.shared.config import load_all_stages  # noqa: E402
from environments.shared.species_names import require_model_id  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest  # noqa: E402

caps = {}
count = {"r": 0}


def snap(model, tag):
    for k, v in model.policy.state_dict().items():
        caps[f"{tag}/param/{k}"] = v.detach().cpu().numpy().copy()
    opt = model.policy.optimizer.state_dict()
    for pid, st in opt["state"].items():
        for kk, vv in st.items():
            caps[f"{tag}/opt/{pid}/{kk}"] = vv.detach().cpu().numpy().copy() if th.is_tensor(vv) else np.asarray(vv)
    caps[f"{tag}/opt_groups"] = np.frombuffer(
        json.dumps(opt["param_groups"], sort_keys=True, default=str).encode(), dtype=np.uint8
    ).copy()
    vn = model.get_vec_normalize_env()
    for name in ("obs_rms", "ret_rms"):
        rms = getattr(vn, name)
        caps[f"{tag}/{name}/mean"] = np.asarray(rms.mean).copy()
        caps[f"{tag}/{name}/var"] = np.asarray(rms.var).copy()
        caps[f"{tag}/{name}/count"] = np.asarray(rms.count).copy()
    caps[f"{tag}/num_timesteps"] = np.asarray(model.num_timesteps)
    caps[f"{tag}/n_updates"] = np.asarray(getattr(model, "_n_updates", -1))


orig_collect = OnPolicyAlgorithm.collect_rollouts
orig_learn = OnPolicyAlgorithm.learn


def collect(self, env, callback, rollout_buffer, n_rollout_steps):
    i = count["r"]
    snap(self, f"pre{i}")
    result = orig_collect(self, env, callback, rollout_buffer, n_rollout_steps)
    for f in ("observations", "actions", "rewards", "returns", "episode_starts", "values", "log_probs", "advantages"):
        caps[f"roll{i}/{f}"] = np.asarray(getattr(rollout_buffer, f)).copy()
    caps[f"roll{i}/last_obs"] = np.asarray(self._last_obs).copy()
    count["r"] += 1
    return result


def learn(self, *args, **kwargs):
    result = orig_learn(self, *args, **kwargs)
    snap(self, "final")
    caps["final/class"] = np.frombuffer(
        f"{type(self).__name__}|{type(self.policy).__name__}|{type(self.rollout_buffer).__name__}|"
        f"{type(self.env).__name__}|{type(self.get_vec_normalize_env()).__name__}".encode(), dtype=np.uint8
    ).copy()
    return result


OnPolicyAlgorithm.collect_rollouts = collect
OnPolicyAlgorithm.learn = learn

model_id = require_model_id("tyrannosaurus_rex", "anatomical")
cfg = get_species_config(model_id)
stages = load_all_stages(model_id)
ref = load_stage_manifest(model_id).resolve("stance").reference
stages[ref]["ppo_kwargs"].update(n_steps=n_steps, batch_size=batch, n_epochs=epochs)
if switch == "on":
    stages[ref]["ppo_kwargs"]["critic_phase"] = True
model = tb.train(
    cfg, stages, ref, total, n_envs=n_envs, seed=seed, output_dir=f"{out}/run", use_tensorboard=False, verbose=0,
    report_metrics=False, save_on_interrupt=False, post_eval_episodes=0,
    **({"load_path": load, "vecnorm_path": load[:-4] + "_vecnorm.pkl", "task_load_mode": "resume_same_stage"} if load else {}),
)
np.savez(f"{out}/caps.npz", **caps)
print("rollouts", count["r"], "keys", len(caps), "final", bytes(caps["final/class"]).decode())
