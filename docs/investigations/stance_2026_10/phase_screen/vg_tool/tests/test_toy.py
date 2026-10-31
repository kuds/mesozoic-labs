"""vg_sigma's episode loop (collect) and G (episode_rows / _stats_for) on a TOY env with terminations (adapted from the
checker's check1/toy_collect.py): SB3's DummyVecEnv + Monitor + VecNormalize (frozen, norm_reward True, ret var 4,
clip_reward 10); per env a schedule of episode ends: the time limit at 1000 (truncated: bootstrapped with V(terminal)),
a fall at k < 1000, and a fall at exactly k = 1000 (terminated AND truncated, which SB3 does not bootstrap).  A toy
policy returns zero actions and V = 1 + 2 obs[0].  G is checked against a direct computation; no real env or model."""
import gymnasium as gym
import numpy as np
import pytest
import torch
import vg_sigma as T
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SCHED = [[None, None], [300, None], [1000, 50], [None, 999]]    # per env: the fall step per episode (None: none)


class Toy(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (63,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (15,), np.float32)

    def __init__(self, sched):
        self.sched = sched
        self.ep = -1
        self._step_count = 0

    def _obs(self):
        o = np.zeros(63, np.float32)
        o[0] = self._step_count / 1000.0
        return o

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep += 1
        self._step_count = 0
        return self._obs(), {}

    def step(self, a):
        self._step_count += 1
        k = self._step_count
        r = 1.0 + 0.01 * k + (150.0 if k == 77 else 0.0)        # one reward hits the clip
        fall = self.sched[self.ep % len(self.sched)]
        term = fall is not None and k == fall
        if term:
            r -= 10.0                                            # a fall penalty
        return self._obs(), float(r), bool(term), bool(k >= T.HORIZON), {"step": k}


class _Policy:
    def set_training_mode(self, mode):
        pass

    def __call__(self, obs):
        n = obs.shape[0]
        return torch.zeros(n, 15), 1.0 + 2.0 * obs[:, :1].double(), torch.zeros(n)


class _Model:
    policy = _Policy()
    action_space = Toy.action_space


def _vfun(x):
    return 1.0 + 2.0 * np.asarray(x, np.float64)[..., 0]


@pytest.fixture(scope="module")
def toy():
    venv = DummyVecEnv([(lambda s=s: Monitor(Toy(s))) for s in SCHED])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=T.GAMMA, clip_reward=10.0)
    vn.training = False
    vn.ret_rms.var = np.float64(4.0)
    done, _cur, _steps = T.collect(vn, _Model(), 2)
    eps = [ep for i in range(4) for ep in done[i][:2]]
    TX = [ep["term_X"] for ep in eps if ep["trunc"]]
    VT = _vfun(vn.normalize_obs(np.stack(TX))) if TX else np.zeros(0)
    V = np.concatenate([_vfun(vn.normalize_obs(ep["X"])) for ep in eps])
    tobs = np.zeros((1, 63), np.float32)
    tobs[0, 0] = 1.0
    v_end = float(_vfun(vn.normalize_obs(tobs))[0])
    vn.close()
    return eps, V, VT, v_end


def _want_G(fall, v_end):
    """Direct G of one scheduled episode: normalised rewards (float32, clip 10), bootstrapped only at a time-limit end
    that is not also a fall."""
    L = T.HORIZON if fall is None else fall
    ks = np.arange(1, L + 1)
    r = 1.0 + 0.01 * ks + np.where(ks == 77, 150.0, 0.0) - np.where(ks == fall, 10.0, 0.0)
    rn = np.clip(r.astype(np.float32) / np.sqrt(4.0 + 1e-8), -10.0, 10.0).astype(np.float32).astype(np.float64)
    acc = v_end if fall is None else 0.0
    G = np.zeros(L)
    for t in range(L - 1, -1, -1):
        acc = rn[t] + T.GAMMA * acc
        G[t] = acc
    return G


def test_collect_flags_and_counters(toy):
    eps, _V, _VT, _v = toy
    falls = [f for s in SCHED for f in s]
    assert len(eps) == 8
    for ep, fall in zip(eps, falls, strict=True):
        L = T.HORIZON if fall is None else fall
        assert np.array_equal(ep["k"], np.arange(L)) and ep["monitor_l"] == L
        assert ep["trunc"] == (fall is None) and ep["term"] == (fall is not None)
        if ep["trunc"]:
            assert ep["k_term"] == T.HORIZON and ep["term_X"] is not None and ep["term_X"][0] == np.float32(1.0)


def test_G_bootstraps_only_time_limit_ends(toy):
    eps, V, VT, v_end = toy
    rows = T.episode_rows(eps, V, VT, T.GAMMA)
    falls = [f for s in SCHED for f in s]
    want_rows = []
    for row, ep, fall in zip(rows, eps, falls, strict=True):
        want = _want_G(fall, v_end)
        assert len(row["G"]) == len(want)
        np.testing.assert_allclose(row["G"], want, rtol=0, atol=1e-9)
        want_rows.append({"k": ep["k"], "V": row["V"], "G": want})
    got = T._stats_for(eps, V, VT, T.GAMMA)
    ref = T.bucket_stats(want_rows)
    for g, w in zip(got, ref, strict=True):
        assert (g["n_episodes"], g["n_states"]) == (w["n_episodes"], w["n_states"])
        assert g["mean_err"] == pytest.approx(w["mean_err"], abs=1e-9)
        assert g["mean_G"] == pytest.approx(w["mean_G"], abs=1e-9)


def test_stats_for_refuses_another_gamma(toy):
    eps, V, VT, _v = toy
    for gamma in (0.99, 0.95, 1.0):
        with pytest.raises(T.Refused):
            T._stats_for(eps, V, VT, gamma)
