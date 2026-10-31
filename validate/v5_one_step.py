"""PS V5: ONE optimizer step on a throwaway in-memory copy of the start pair under the critic_phase arm, to show where
the gradient goes.  Nothing is saved: the model, its env and the buffer live in this process only and are discarded
at exit;
the only file written is the JSON evidence on stdout (captured by the caller under <PS>/validate/logs/).  TMPDIR is a
throwaway folder under <PS>/validate/ that the caller deletes afterwards.

What it does (the reader's construction of a loaded pair, eb_common.build: a 4-env DummyVecEnv, VecNormalize.load of the
start pkl with training on, load_sb3_model with the stage's [ppo] kwargs, enable_mirror_kl with the run's mirror map):
  1. install_arm("critic_phase") on it (bs_hooks, the same function the run uses; verified transform);
  2. one collect_rollouts of 2,048 steps per env (seed 9441, a PS validation seed) with the hooked envs (the phase in
     the observations), the buffer's advantages GAE(0.95);
  3. n_epochs := 1 and batch_size := 8,192 (the whole buffer as one minibatch), then MirrorKLPPO.train() itself: the
     clipped surrogate, the entropy term, the value loss and the mirrored value loss, the mirror KL, the gradient clip
     and ONE Adam step (the target_kl check precedes the step and cannot stop the first minibatch: ratio = 1, KL 0);
  4. records: the Adam step count before / after (+1 exactly); the gradients left on the first layers (p.grad after
     train(): the one minibatch's): the actor's phase columns' gradient max |g| (must be exactly 0) and its other
     columns' (nonzero), the critic's phase columns' (nonzero); the weights after the step: the actor's phase columns
     bitwise unchanged, the critic's phase columns moved from 0; the Adam moments of the actor's phase columns still 0,
     the critic's nonzero; the actor still masked (its mean unchanged by perturbing the slots) and the critic now
     responding to the slots.
    PY -I -B v5_one_step.py
"""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
TOOLS = PS / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("bs_common")
H = _load("bs_hooks")
SEED = 9441
assert SEED in C.VALIDATION_SEEDS
tmp = PS / "validate" / "v5_throwaway"
(tmp / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TMPDIR": str(tmp),
                   "MPLCONFIGDIR": str(tmp / "mpl"), "GITHUB_SHA": C.COMMIT})
code = os.path.realpath(C.CODE)
sys.path.insert(0, code)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import Logger  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

import environments  # noqa: E402
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.mirror_kl_ppo import MirrorKLPPO, enable_mirror_kl  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.policy_loading import load_sb3_model  # noqa: E402
from environments.shared.species_registry import env_class  # noqa: E402
from environments.shared.symmetry import MirrorMap, pop_symmetry_config  # noqa: E402
from environments.shared.train_base import _prepare_alg_kwargs, create_vec_env  # noqa: E402

assert os.path.realpath(environments.__file__).startswith(code + os.sep)
zip_path, pkl_path = C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"])
assert C.sha256_file(zip_path) == C.START["zip_sha256"] and C.sha256_file(pkl_path) == C.START["pkl_sha256"]
mid = ModelId("tyrannosaurus_rex", "anatomical")
cfg = copy.deepcopy(load_stage_config(mid, "stance"))


class _Species:
    model_id = mid
    env_class = None


species = _Species()
species.env_class = env_class(mid)
ppo = copy.deepcopy(cfg["ppo_kwargs"])
sym = pop_symmetry_config(ppo)
venv = create_vec_env(species, {1: {"env_kwargs": dict(cfg["env_kwargs"])}}, 1, C.N_ENVS, seed=SEED,
                      use_subproc=False, algorithm="ppo", gamma=float(ppo["gamma"]))
vn = VecNormalize.load(str(pkl_path), venv.venv)
vn.training = True                                     # as the run's training env: the statistics update
vn.norm_reward = True
alg_kwargs, _tb, _g = _prepare_alg_kwargs({"ppo_kwargs": copy.deepcopy(cfg["ppo_kwargs"])}, "ppo", 0,
                                          tmp / "_tb_unused", use_tensorboard=False)
pop_symmetry_config(alg_kwargs)
alg_kwargs.pop("policy_kwargs", None)
model = load_sb3_model(str(zip_path), algorithm="ppo", env=vn, device="cpu", **alg_kwargs)
model = enable_mirror_kl(model, sym, MirrorMap.load(mid, current_plant_identity(mid)))
assert isinstance(model, MirrorKLPPO) and model.gae_lambda == 0.95
rec = H.install_arm("critic_phase", model, vn)
assert rec["transform_verified"] is True
policy = model.policy
w_pi = policy.mlp_extractor.policy_net[0].weight
w_vf = policy.mlp_extractor.value_net[0].weight
LO, HI = H.PHASE_LO, H.PHASE_HI
pi_cols0 = w_pi[:, LO:HI].detach().clone()
pi_other0 = w_pi[:, :LO].detach().clone()
steps0 = H._adam_steps(policy)
vn.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
model.set_logger(Logger(folder=None, output_formats=[]))
model._last_obs = None


class Nop(BaseCallback):
    def _on_step(self):
        return True


_t, cb = model._setup_learn(model.n_steps * model.n_envs, Nop(), False, "ps_v5", False)
assert model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps)
buf = model.rollout_buffer
obs = np.asarray(buf.observations)
out = {"seed": SEED, "arm_applied": {k: rec[k] for k in ("transform_verified", "value_phase_cols_max_abs_before",
                                                        "obs_rms_class", "mlp_extractor_class", "train_env_class")},
       "buffer_rows": int(obs.shape[0] * obs.shape[1]),
       "buffer_slots_min_max": [float(obs[..., LO:HI].min()), float(obs[..., LO:HI].max())],
       "buffer_rows_presettle": int((obs[..., LO + 1] < 0.5).sum()),
       "adam_steps_before": steps0, "n_updates_before": int(model._n_updates)}
model.n_epochs = 1
model.batch_size = model.n_steps * model.n_envs      # one minibatch: exactly one optimizer step
model.train()
g_pi, g_vf = w_pi.grad, w_vf.grad
out.update({
    "adam_steps_after": H._adam_steps(policy), "n_updates_after": int(model._n_updates),
    "grad_actor_phase_cols_max_abs": float(g_pi[:, LO:HI].abs().max()),
    "grad_actor_other_cols_max_abs": float(g_pi[:, :LO].abs().max()),
    "grad_critic_phase_cols_max_abs": float(g_vf[:, LO:HI].abs().max()),
    "grad_critic_other_cols_max_abs": float(g_vf[:, :LO].abs().max()),
    "actor_phase_cols_bitwise_unchanged": bool(torch.equal(w_pi[:, LO:HI].detach(), pi_cols0)),
    "actor_other_cols_changed": bool(not torch.equal(w_pi[:, :LO].detach(), pi_other0)),
    "critic_phase_cols_max_abs_after": float(w_vf[:, LO:HI].detach().abs().max()),
    "adam_moments_actor_phase_cols_max_abs": H._adam_phase_max_abs(policy, w_pi),
    "adam_moments_critic_phase_cols_max_abs": H._adam_phase_max_abs(policy, w_vf),
})
x = torch.as_tensor(obs.reshape(-1, obs.shape[-1])[:256], dtype=torch.float32)
acts = torch.as_tensor(np.asarray(buf.actions).reshape(-1, buf.actions.shape[-1])[:256], dtype=torch.float32)
dep = H.phase_dependence(policy, x, acts, H._mirror_obs_map(model))
out["phase_dependence_after_one_step"] = dep
ok = (out["adam_steps_after"] == steps0 + 1 and out["grad_actor_phase_cols_max_abs"] == 0.0
      and out["grad_actor_other_cols_max_abs"] > 0.0 and out["grad_critic_phase_cols_max_abs"] > 0.0
      and out["actor_phase_cols_bitwise_unchanged"] and out["actor_other_cols_changed"]
      and out["critic_phase_cols_max_abs_after"] > 0.0 and out["adam_moments_actor_phase_cols_max_abs"] == 0.0
      and out["adam_moments_critic_phase_cols_max_abs"] > 0.0
      and max(dep[k] for k in H.PHASE_ACTOR_KEYS if dep[k] is not None) == 0.0
      and all(dep[k] > 0.0 for k in H.PHASE_CRITIC_KEYS))
out["V5"] = "PASS" if ok else "FAIL"
vn.close()
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
