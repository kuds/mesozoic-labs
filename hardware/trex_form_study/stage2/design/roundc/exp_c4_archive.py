"""Round C scratch experiment C4 (not a record): an SB3 archive saved with module-level schedule objects holds no
bytecode (the repository's inspector), loads through the repository's safe loader with the schedules passed, and its
policy fingerprint (256 seeded raw observations -> deterministic float32 actions through the saved VecNormalize
statistics) reproduces after a save and a load.  Also: minibatch steps per update read from the logger."""
import os
import sys
import tempfile
import time

os.environ["MUJOCO_GL"] = "disable"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/user/mesozoic-labs")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402
from s2sched_proto import ConstantSchedule, LinearSchedule  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from designlab.stage1.policy import DESIGN2, model_kwargs  # noqa: E402
from environments.shared.policy_loading import inspect_sb3_archive, load_sb3_model  # noqa: E402

ACTOR, OBS, ACT = 97, 140, 17  # an -X sized layout


class Dummy(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (ACT,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        return self.np_random.standard_normal(OBS).astype(np.float32), {}

    def step(self, a):
        self.k += 1
        return self.np_random.standard_normal(OBS).astype(np.float32), float(-np.square(a).sum()), False, self.k >= 50, {}


torch.set_num_threads(1)
venv = VecNormalize(DummyVecEnv([Dummy for _ in range(8)]), norm_obs=True, norm_reward=True, clip_obs=10.0,
                    clip_reward=50.0, gamma=0.99)
kw = model_kwargs(ACTOR, log_std_init=float(np.log(0.1)))
kw.update(learning_rate=LinearSchedule(1e-4, 3e-5), clip_range=ConstantSchedule(0.2), n_steps=64, batch_size=64)
model = PPO(env=venv, seed=42, device="cpu", verbose=0, **kw)
print("sb3", __import__("stable_baselines3").__version__, "torch", torch.__version__, "target_kl", model.target_kl,
      "n_epochs", model.n_epochs)


class Steps:
    """Counts the minibatch steps PPO.train really ran: n_updates advances by n_epochs whatever happens, so the count
    comes from the optimizer."""
    def __init__(self, opt):
        self.n = 0
        self._step = opt.step
        opt.step = self.step

    def step(self, *a, **k):
        self.n += 1
        return self._step(*a, **k)


cnt = Steps(model.policy.optimizer)
t0 = time.perf_counter()
model.learn(total_timesteps=8 * 64 * 3)
print(f"3 updates in {time.perf_counter() - t0:.1f} s; optimizer steps {cnt.n} of {3 * model.n_epochs * (8 * 64 // 64)} "
      f"possible; logger approx_kl {model.logger.name_to_value.get('train/approx_kl')}, n_updates {model._n_updates}")

rng = np.random.default_rng([20261115, 42])
fp_obs = (3.0 * rng.standard_normal((256, OBS))).astype(np.float32)


def fingerprint(m, vn):
    a, _ = m.predict(vn.normalize_obs(fp_obs), deterministic=True)
    return np.asarray(a, np.float32)


fp0 = fingerprint(model, venv)
with tempfile.TemporaryDirectory(dir=HERE) as d:
    mp, vp = os.path.join(d, "model_1536.zip"), os.path.join(d, "vecnormalize_1536.pkl")
    model.save(mp)
    venv.save(vp)
    insp = inspect_sb3_archive(mp)
    print("saved by Python", insp.saved_python_text, "| serialized members", sorted(insp.serialized_members))
    print("bytecode members:", sorted(insp.bytecode_members) or "none", "| size", os.path.getsize(mp), "bytes")
    m2 = load_sb3_model(mp, algorithm="ppo", device="cpu",
                        custom_objects=dict(learning_rate=LinearSchedule(1e-4, 3e-5), lr_schedule=LinearSchedule(1e-4, 3e-5),
                                            clip_range=ConstantSchedule(0.2)))
    import pickle
    with open(vp, "rb") as f:
        vn2 = pickle.load(f)
    vn2.training = False
    fp1 = fingerprint(m2, vn2)
    print("fingerprint max |difference| after save and load:", float(np.max(np.abs(fp1 - fp0))),
          "| lr at progress 0.5:", m2.lr_schedule(0.5), "| clip:", m2.clip_range(0.5), type(m2.clip_range).__name__)
    torch.set_num_threads(2)
    fp2 = fingerprint(m2, vn2)
    print("with 2 torch threads:", float(np.max(np.abs(fp2 - fp0))))
