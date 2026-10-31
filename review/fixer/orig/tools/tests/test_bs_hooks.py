"""Unit tests of the ARM HOOK (bs_hooks.py) on SB3 2.9.0's own classes: no environment steps, no training.

    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS>/tools/tests/test_bs_hooks.py

The critic-blind hook on a real ActorCriticPolicy (small and the start pair's own), the frozen statistics (stock RMS
and the code copy's SymmetricRunningMeanStd: no-op update, pickling and deep-copying as the stock class, kept by
MirrorKLPPO's install), install_arm's refusals, the independent GAE / Monte Carlo recomputation against SB3's
RolloutBuffer, rollout_evidence and evidence_problems for every arm (and mutants that must fail).
PS: critic_phase's pieces (the phase features against bs_common, the env's _get_obs on the real T. rex env in lockstep
with a plain one incl. the terminal observation, the actor's mask and its zero gradient, the critic's zero columns,
the frozen statistics' slots incl. pickling and MirrorKLPPO's install, the mirror map on the real map file, the probe
feeding the bank's phase, install_arm's refusals), its per-rollout evidence and its mutants, and the real start pair's
actor and critic bitwise unchanged at every phase value after the transform.  BS2's arms stay in bs_hooks (unreachable
in PS) and keep their tests: the shared machinery is the same.
"""
from __future__ import annotations

import copy
import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("bs_common")
H = _load("bs_hooks")
CODE = C.CODE
if sys.path[:1] != [CODE]:
    sys.path.insert(0, CODE)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.buffers import RolloutBuffer  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.torch_layers import MlpExtractor  # noqa: E402

torch.set_num_threads(1)
OBS, ACT = 63, 15
OBS_SPACE = gym.spaces.Box(-10.0, 10.0, (OBS,), np.float32)
ACT_SPACE = gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32)


def small_policy(seed=0):
    torch.manual_seed(seed)
    return ActorCriticPolicy(OBS_SPACE, ACT_SPACE, lambda _: 1e-3, net_arch=[16, 8], activation_fn=torch.nn.Tanh)


def fake_model(policy, lam=0.95, perm6=6, sign6=1.0, mirror=True):
    perm = np.arange(OBS)
    sign = np.ones(OBS)
    if perm6 != 6:
        perm[6], perm[perm6] = perm6, 6
    sign[6] = sign6
    buf = RolloutBuffer(32, OBS_SPACE, ACT_SPACE, device="cpu", gamma=0.98, gae_lambda=lam, n_envs=4)
    m = SimpleNamespace(policy=policy, gae_lambda=lam, rollout_buffer=buf, observation_space=OBS_SPACE, gamma=0.98,
                        num_timesteps=C.S + C.ROLLOUT)
    if mirror:
        m._mirror = SimpleNamespace(mirror_map=SimpleNamespace(obs=SimpleNamespace(perm=perm, sign=sign)))
    return m


def fake_env(rms=None):
    return SimpleNamespace(obs_rms=rms if rms is not None else RunningMeanStd(shape=(OBS,)),
                           ret_rms=RunningMeanStd(shape=()), training=True, norm_obs=True, norm_reward=True)


def perturbed(x, how):
    y = x.clone()
    if how == "add":
        y[:, 6] += 2.0
    elif how == "rand":
        y[:, 6] = torch.randn(len(y))
    else:
        y[:, 6] = 0.0
    return y


# ------------------------------------------------------------------------------------------------ critic_blind
def test_critic_blind_value_paths_ignore_obs6_actor_does_not():
    p = small_policy()
    sd_before = {k: v.clone() for k, v in p.state_dict().items()}
    x = torch.randn(64, OBS)
    with torch.no_grad():
        v_raw = p.predict_values(x)
        mu_raw = p.get_distribution(x).mode()
    rec = H.install_arm("critic_blind_tail3", fake_model(p), fake_env())
    assert rec["mlp_extractor_class"] == "CriticBlindTail3MlpExtractor" and rec["blind_index"] == 6
    assert type(p.mlp_extractor).__mro__[1] is MlpExtractor
    sd_after = p.state_dict()
    assert list(sd_before) == list(sd_after) and all(sd_before[k].equal(sd_after[k]) for k in sd_before)
    acts = torch.randn(64, ACT)
    with torch.no_grad():
        for how in ("add", "rand", "zero"):
            y = perturbed(x, how)
            assert torch.equal(p.predict_values(x), p.predict_values(y))
            assert torch.equal(p.forward(x, deterministic=True)[1], p.forward(y, deterministic=True)[1])
            assert torch.equal(p.evaluate_actions(x, acts)[0], p.evaluate_actions(y, acts)[0])
            assert not torch.equal(p.get_distribution(x).mode(), p.get_distribution(y).mode())
        # the actor is untouched, the critic is the raw one at obs6 = 0
        assert torch.equal(p.get_distribution(x).mode(), mu_raw)
        x0 = x.clone()
        x0[:, 6] = 0.0
        p.mlp_extractor.__class__ = MlpExtractor
        assert torch.equal(p.predict_values(x0), p.predict_values(x0))
        v_raw0 = p.predict_values(x0)
        p.mlp_extractor.__class__ = H.critic_blind_class(MlpExtractor)
        assert torch.equal(p.predict_values(x), v_raw0)
        assert not torch.equal(p.predict_values(x), v_raw)
    # the caller's tensor is never written
    xc = x.clone()
    p.predict_values(x)
    assert torch.equal(x, xc)


def test_critic_blind_gradient_on_obs6_is_zero_and_raw_values_bypass():
    p = small_policy(1)
    H.install_arm("critic_blind_tail3", fake_model(p), fake_env())
    z = torch.randn(32, OBS, requires_grad=True)
    g = torch.autograd.grad(p.predict_values(z).sum(), z)[0]
    assert torch.all(g[:, 6] == 0) and torch.any(g[:, 5] != 0)
    with torch.no_grad():
        x = torch.randn(8, OBS)
        y = perturbed(x, "add")
        assert not torch.equal(H.raw_values(p, x), H.raw_values(p, y))


def test_install_refusals():
    with pytest.raises(H.HookError):
        H.install_arm("critic_blind_tail3", fake_model(small_policy(), perm6=7), fake_env())
    with pytest.raises(H.HookError):
        H.install_arm("critic_blind_tail3", fake_model(small_policy(), sign6=-1.0), fake_env())
    with pytest.raises(H.HookError):
        H.install_arm("control", fake_model(small_policy(), lam=1.0), fake_env())
    with pytest.raises(H.HookError):
        H.install_arm("gae_lambda_1", fake_model(small_policy(), lam=0.95), fake_env())
    with pytest.raises(H.HookError):
        H.install_arm("nonsense", fake_model(small_policy()), fake_env())
    p = small_policy()
    H.install_arm("critic_blind_tail3", fake_model(p), fake_env())
    with pytest.raises(H.HookError):                     # twice: the extractor is no longer the stock class
        H.install_arm("critic_blind_tail3", fake_model(p), fake_env())
    env = fake_env()
    env.training = False
    with pytest.raises(H.HookError):
        H.install_arm("frozen_obs_norm", fake_model(small_policy()), env)


def test_control_and_gae_change_nothing_on_the_objects():
    for arm, lam in (("control", 0.95), ("gae_lambda_1", 1.0)):
        p = small_policy()
        env = fake_env()
        rms = env.obs_rms
        rec = H.install_arm(arm, fake_model(p, lam=lam), env)
        assert type(p.mlp_extractor) is MlpExtractor and type(env.obs_rms) is RunningMeanStd and env.obs_rms is rms
        assert rec["gae_lambda"] == lam and rec["rollout_buffer_gae_lambda"] == lam
        assert H.arm_state_problems(arm, fake_model(p, lam=lam), env) == []
        other = "critic_blind_tail3"
        assert H.arm_state_problems(other, fake_model(p, lam=0.95), env) != []


# ------------------------------------------------------------------------------------------------ frozen
def _symmetric_rms():
    from environments.shared.mirror_kl_ppo import SymmetricRunningMeanStd

    omap = SimpleNamespace(perm=np.arange(OBS), sign=np.ones(OBS), dim=OBS)
    r = SymmetricRunningMeanStd(omap, shape=(OBS,))
    r.update(np.random.default_rng(0).normal(size=(50, OBS)))
    return r


@pytest.mark.parametrize("make", [lambda: RunningMeanStd(shape=(OBS,)), _symmetric_rms])
def test_frozen_rms_noop_counts_and_pickles_stock(make):
    rms = make()
    rms.update(np.random.default_rng(1).normal(size=(20, OBS))) if type(rms) is RunningMeanStd else None
    env = fake_env(rms)
    base = type(rms)
    before = (rms.mean.copy(), rms.var.copy(), float(rms.count))
    d0 = H.rms_digest(rms)
    rec = H.install_arm("frozen_obs_norm", fake_model(small_policy()), env)
    assert rec["obs_rms_class"] == f"FrozenObs{base.__name__}" and rec["obs_rms_digest_at_freeze"] == d0
    assert isinstance(env.obs_rms, base)
    c0 = H.frozen_counter(env.obs_rms)["calls"]
    for _ in range(3):
        env.obs_rms.update(np.random.default_rng(2).normal(size=(7, OBS)))
    assert H.frozen_counter(env.obs_rms)["calls"] == c0 + 3
    assert H.rms_digest(env.obs_rms) == d0
    assert np.array_equal(env.obs_rms.mean, before[0]) and env.obs_rms.count == before[2]
    for clone in (pickle.loads(pickle.dumps(env.obs_rms)), copy.deepcopy(env.obs_rms)):
        assert type(clone) is RunningMeanStd
        assert np.array_equal(clone.mean, before[0]) and np.array_equal(clone.var, before[1])
        assert clone.count == before[2]
    assert H.arm_state_problems("frozen_obs_norm", fake_model(small_policy()), env) == []
    assert H.arm_state_problems("control", fake_model(small_policy()), env) != []


def test_frozen_symmetric_rms_is_kept_by_mirror_install():
    from environments.shared.mirror_kl_ppo import install_symmetric_obs_rms

    rms = _symmetric_rms()
    env = fake_env(rms)
    H.install_arm("frozen_obs_norm", fake_model(small_policy()), env)
    frozen = env.obs_rms
    vn = SimpleNamespace(obs_rms=frozen, norm_obs=True)
    install_symmetric_obs_rms(vn, SimpleNamespace(obs=SimpleNamespace(perm=np.arange(OBS), sign=np.ones(OBS),
                                                                      dim=OBS)))
    assert vn.obs_rms is frozen                        # MirrorKLPPO.learn()'s _mirror_install keeps it


# ------------------------------------------------------------------------------------------------ GAE / MC
def _filled_buffer(lam, seed=0, policy=None):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(64, OBS_SPACE, ACT_SPACE, device="cpu", gamma=0.98, gae_lambda=lam, n_envs=4)
    starts = np.ones(4, dtype=np.float32)
    for _t in range(64):
        obs = rng.normal(size=(4, OBS)).astype(np.float32)
        if policy is not None:
            with torch.no_grad():
                v = policy.predict_values(torch.as_tensor(obs)).flatten()
        else:
            v = torch.as_tensor(rng.normal(size=4).astype(np.float32))
        buf.add(obs, rng.normal(size=(4, ACT)).astype(np.float32), rng.normal(size=4).astype(np.float32), starts, v,
                torch.zeros(4))
        starts = (rng.random(4) < 0.05).astype(np.float32)
    last = torch.as_tensor(rng.normal(size=4).astype(np.float32))
    dones = rng.random(4) < 0.3
    buf.compute_returns_and_advantage(last_values=last, dones=dones)
    return buf, last.numpy(), dones


@pytest.mark.parametrize("lam", [0.95, 1.0, 0.0])
def test_gae_recomputation_matches_sb3(lam):
    buf, last, dones = _filled_buffer(lam)
    ref = H.gae(buf.rewards, buf.values, buf.episode_starts, last, dones, 0.98, lam)
    scale = max(1.0, float(np.abs(ref).max()))
    assert np.max(np.abs(buf.advantages - ref)) <= H.ADV_REL_TOL * scale
    mc = H.mc_return(buf.rewards, buf.episode_starts, last, dones, 0.98) - buf.values
    if lam == 1.0:
        assert np.max(np.abs(buf.advantages - mc)) <= H.ADV_REL_TOL * scale
    else:
        assert np.max(np.abs(buf.advantages - mc)) > 10 * H.ADV_REL_TOL * scale
    assert np.max(np.abs(H.gae(buf.rewards, buf.values, buf.episode_starts, last, dones, 0.98, 1.0) - mc)) < 1e-9


def test_mc_return_by_hand():
    r = np.array([[1.0], [2.0], [3.0]])
    st = np.array([[1.0], [0.0], [1.0]])                # an episode boundary before step 2
    G = H.mc_return(r, st, np.array([10.0]), np.array([False]), 0.5)
    assert G[:, 0].tolist() == [1.0 + 0.5 * 2.0, 2.0, 3.0 + 0.5 * 10.0]
    G = H.mc_return(r, st, np.array([10.0]), np.array([True]), 0.5)
    assert G[2, 0] == 3.0


# ------------------------------------------------------------------------------------------------ evidence
def _evidence(arm, n=3, mutate=None):
    if arm == "critic_phase":
        return _phase_evidence(n, mutate)
    lam = H.GAE_LAMBDA[arm]
    p = small_policy(3)
    env = fake_env(_symmetric_rms())
    m = fake_model(p, lam=lam)
    H.install_arm(arm, m, env)
    start = H.rms_digest(env.obs_rms)
    out = []
    for k in range(n):
        buf, last, dones = _filled_buffer(lam, seed=10 + k, policy=p)
        if mutate == "values_raw" and arm == "critic_blind_tail3":
            with torch.no_grad():
                obs = torch.as_tensor(buf.observations.reshape(-1, OBS))
                buf.values[:] = H.raw_values(p, obs).numpy().reshape(buf.values.shape)
        if mutate == "adv":
            buf.advantages[0, 0] += 1.0
        m.rollout_buffer = buf
        m.num_timesteps = C.S + C.ROLLOUT * (k + 1)
        env.obs_rms.update(np.random.default_rng(20 + k).normal(size=(9, OBS)))
        env.ret_rms.update(np.random.default_rng(30 + k).normal(size=9))
        if mutate == "unfreeze" and arm == "frozen_obs_norm" and k == 1:
            env.obs_rms.mean = env.obs_rms.mean + 1.0
        if mutate == "unhook" and arm == "critic_blind_tail3" and k == 2:
            p.mlp_extractor.__class__ = MlpExtractor
        out.append(H.rollout_evidence(arm, m, env, {"values": torch.as_tensor(last), "dones": dones},
                                      start_obs_digest=start))
    return out


@pytest.mark.parametrize("arm", H.ARMS)
def test_evidence_passes_for_each_arm(arm):
    ev = _evidence(arm)
    s = H.summarize(ev)
    assert H.evidence_problems(arm, s, n_expected=3) == []
    for other in H.ARMS:
        if other != arm and not {arm, other} <= {"control", "gae_lambda_1"} | {arm}:
            assert H.evidence_problems(other, s, n_expected=3) != [], (arm, other)
    assert H.evidence_problems(arm, s, n_expected=4) != []


def test_evidence_control_vs_gae_distinguished():
    s_c, s_g = H.summarize(_evidence("control")), H.summarize(_evidence("gae_lambda_1"))
    assert H.evidence_problems("gae_lambda_1", s_c, n_expected=3) != []
    assert H.evidence_problems("control", s_g, n_expected=3) != []


@pytest.mark.parametrize("arm,mutate", [("critic_blind_tail3", "values_raw"), ("critic_blind_tail3", "unhook"),
                                        ("frozen_obs_norm", "unfreeze"), ("control", "adv"), ("gae_lambda_1", "adv")])
def test_evidence_mutants_fail(arm, mutate):
    s = H.summarize(_evidence(arm, mutate=mutate))
    assert H.evidence_problems(arm, s, n_expected=3) != []


def test_value_dependence_restores_action_dist():
    p = small_policy()
    x = torch.randn(16, OBS)
    with torch.no_grad():
        p.get_distribution(x[:4])
    before = dict(vars(p.action_dist))
    H.value_dependence(p, x, torch.randn(16, ACT), None)
    after = dict(vars(p.action_dist))
    assert before.keys() == after.keys() and all(before[k] is after[k] for k in before)


# ------------------------------------------------------------------------------------------------ the start pair
def test_start_pair_policy_blinded_values_bank():
    """The real start pair (read only): the hook on its policy; bank values independent of obs 6, actions unchanged."""
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    model, norm, _ = load_sb3_checkpoint(C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"]),
                                         lambda: env_class(mid)(**ctor))
    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        bank = np.asarray(z["bank_raw_obs"], dtype=np.float32)
    x = torch.as_tensor(np.asarray(norm.normalize_obs(bank), dtype=np.float32))
    p = model.policy
    with torch.no_grad():
        mu0, v0 = p.get_distribution(x).mode().clone(), p.predict_values(x).clone()
        p.mlp_extractor.__class__ = H.critic_blind_class(MlpExtractor)
        mu1, v1 = p.get_distribution(x).mode(), p.predict_values(x)
        v2 = p.predict_values(perturbed(x, "add"))
    assert torch.equal(mu0, mu1) and torch.equal(v1, v2) and not torch.equal(v0, v1)


# ------------------------------------------------------------------------------------------------ critic_phase (PS)
SL = slice(H.PHASE_LO, H.PHASE_HI)


class FakeEnv:
    """A training env stand-in: a step counter, a zero command, an observation with zero command slots."""

    def __init__(self, k=0):
        self._step_count = k
        self._command = np.zeros(3, dtype=np.float32)
        self.direction_controller = None

    def _get_obs(self):
        return np.ones(OBS, dtype=np.float32)


def _slot_free_rms(symmetric=True):
    """Statistics whose slots 60-62 carried only zeros (mean exactly 0), like the start pkl's."""
    data = np.random.default_rng(5).normal(size=(50, OBS))
    data[:, SL] = 0.0
    if symmetric:
        from environments.shared.mirror_kl_ppo import SymmetricRunningMeanStd

        r = SymmetricRunningMeanStd(SimpleNamespace(perm=np.arange(OBS), sign=np.ones(OBS), dim=OBS), shape=(OBS,))
    else:
        r = RunningMeanStd(shape=(OBS,))
    r.update(data)
    return r


def _phase_setup(seed=3, n_env=4):
    p = small_policy(seed)
    with torch.no_grad():                               # nonzero first-layer slot columns, like the start pair's
        p.mlp_extractor.value_net[0].weight[:, SL] = 0.3
        p.mlp_extractor.policy_net[0].weight[:, SL] = -0.2
    env = fake_env(_slot_free_rms())
    env.epsilon, env.clip_obs = 1e-8, 10.0
    env.venv = SimpleNamespace(envs=[FakeEnv() for _ in range(n_env)])
    m = fake_model(p, lam=0.95)
    m._n_updates = 100
    return p, env, m


def _phase_buffer(p, env, k0, seed):
    """A 64-step buffer whose slots carry the phase of counters that start at k0 and reset at episode starts, as
    VecNormalize writes them (slots mean 0 / var 1); the envs' counters set to where the rollout leaves them."""
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(64, OBS_SPACE, ACT_SPACE, device="cpu", gamma=0.98, gae_lambda=0.95, n_envs=4)
    k = np.asarray(k0, dtype=np.int64).copy()
    starts = (k == 0).astype(np.float32)
    for t in range(64):
        if t:
            k = np.where(starts > 0, 0, k + 1)
        obs = rng.normal(size=(4, OBS)).astype(np.float32)
        raw = H.phase_matrix(k).astype(np.float32)
        obs[:, SL] = np.clip((raw - 0.0) / np.sqrt(1.0 + env.epsilon), -10, 10).astype(np.float32)
        with torch.no_grad():
            v = p.predict_values(torch.as_tensor(obs)).flatten()
        buf.add(obs, rng.normal(size=(4, ACT)).astype(np.float32), rng.normal(size=4).astype(np.float32), starts, v,
                torch.zeros(4))
        starts = (rng.random(4) < 0.03).astype(np.float32)
    dones = starts > 0                                 # the last step's dones are the next start flags
    for e, ke in zip(env.venv.envs, np.where(dones, 0, k + 1), strict=True):
        e._step_count = int(ke)
    with torch.no_grad():
        last = p.predict_values(torch.as_tensor(rng.normal(size=(4, OBS)).astype(np.float32))).flatten()
    buf.compute_returns_and_advantage(last_values=last, dones=dones)
    return buf, last.numpy(), dones


def _phase_evidence(n=3, mutate=None):
    p, env, m = _phase_setup()
    H.install_arm("critic_phase", m, env)
    ref = H.phase_reference(m)
    start = H.rms_digest(env.obs_rms)
    out = []
    k0 = [0, 0, 150, 990]
    for e, k in zip(env.venv.envs, k0, strict=True):
        e._step_count = k
    for j in range(n):
        if j >= 1:                                        # an "update": the critic's phase columns move
            m._n_updates += 7
            if mutate != "no_learning":
                with torch.no_grad():
                    p.mlp_extractor.value_net[0].weight[:, SL] += 0.01 * (j + 1)
        if mutate == "nonzero_init" and j == 0:
            with torch.no_grad():
                p.mlp_extractor.value_net[0].weight[0, H.PHASE_LO] = 1e-3
        if mutate == "unmask" and j == 1:
            p.mlp_extractor.__class__ = MlpExtractor
        k_start = H.env_step_counts(env)
        buf, last, dones = _phase_buffer(p, env, k_start, seed=40 + j)
        if mutate == "slot_error" and j == 2:
            buf.observations[5, 1, H.PHASE_LO + 2] += 1e-3
        if mutate == "counter" and j == 1:
            env.venv.envs[0]._step_count += 1
        if mutate == "unfreeze" and j == 1:               # the slots' statistics update again (the base class)
            env.obs_rms.__class__ = type(env.obs_rms).bs_base
        if mutate == "mirror" and j == 2:
            m._mirror.mirror_map.obs.sign[H.PHASE_LO + 1] = -1.0
        if mutate == "actor_cols" and j == 2:
            with torch.no_grad():
                p.mlp_extractor.policy_net[0].weight[0, H.PHASE_LO] += 1e-3
        m.rollout_buffer = buf
        m.num_timesteps = C.S + C.ROLLOUT * (j + 1)
        env.obs_rms.update(np.random.default_rng(20 + j).normal(size=(9, OBS)))
        env.ret_rms.update(np.random.default_rng(30 + j).normal(size=9))
        out.append(H.rollout_evidence("critic_phase", m, env, {"values": torch.as_tensor(last), "dones": dones},
                                      start_obs_digest=start, phase_ctx={"k0": k_start, "ref": ref}))
    return out


def test_phase_features_and_env_obs_on_the_real_env():
    """The phase in the observation is the env's own counter's (0 at reset, k after k steps; incremented before
    _get_obs, as the reward gates read it), on a real T. rex env class-swapped, in lockstep with a plain twin: every
    other value equal; the terminal observation at the 1,000-step limit carries k = 1000 (settled); VecNormalize's
    formula with the frozen slots gives back the float32 phase exactly at every k."""
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    a, b = env_class(mid)(**ctor), env_class(mid)(**ctor)
    a.__class__ = H.phase_env_class(type(a))
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert np.array_equal(oa[:60], ob[:60]) and np.all(ob[SL] == 0.0) and tuple(oa[SL]) == tuple(
        np.float32(x) for x in H.phase_tuple(0))
    zero = np.zeros(a.action_space.shape, dtype=np.float32)
    seen = {}

    def lockstep(n):
        for _ in range(n):
            oa, ra, ta, tra, _ = a.step(zero)
            ob, rb, tb, trb, _ = b.step(zero)
            k = a._step_count
            assert k == b._step_count and np.array_equal(oa[:60], ob[:60]) and ra == rb and (ta, tra) == (tb, trb)
            assert oa[SL].tolist() == [float(np.float32(x)) for x in H.phase_tuple(k)] and np.all(ob[SL] == 0.0)
            seen[k] = (oa[SL].tolist(), ta, tra)
            if ta or tra:
                return
    lockstep(210)                                       # across the gate: k = 199 / 200 / 201
    assert seen[199][0] == [float(np.float32(0.995)), 0.0, float(np.float32(0.98))]
    assert seen[200][0] == [1.0, 1.0, 1.0] and seen[201][0] == [1.0, 1.0, 1.0] and 1 in seen
    a._step_count = b._step_count = 998                 # the time limit: the terminal observation at k = 1000
    lockstep(2)
    assert seen[1000] == ([1.0, 1.0, 1.0], False, True)
    a.close()
    b.close()
    raw = H.phase_matrix(np.arange(0, 1001)).astype(np.float32)
    normed = np.clip((raw - 0.0) / np.sqrt(1.0 + 1e-8), -10.0, 10.0).astype(np.float32)
    assert np.array_equal(normed, raw)


def test_phase_env_refuses_a_live_command():
    e = FakeEnv()
    e.__class__ = H.phase_env_class(FakeEnv)
    assert e._get_obs()[SL].tolist() == list(H.phase_tuple(0))[:2] + [float(np.float32(H.phase_tuple(0)[2]))]
    e._command = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(H.HookError):
        e._get_obs()


def test_critic_phase_install_transform_mask_and_zero_init():
    p, env, m = _phase_setup()
    x = torch.randn(64, OBS)
    with torch.no_grad():
        x0 = x.clone()
        x0[:, SL] = 0.0
        mu_load, v_load = p.get_distribution(x0).mode().clone(), p.predict_values(x0).clone()
    sd = {k: v.clone() for k, v in p.state_dict().items()}
    rms_before = (env.obs_rms.mean.copy(), env.obs_rms.var.copy(), float(env.obs_rms.count))
    rec = H.install_arm("critic_phase", m, env)
    assert rec["transform_verified"] is True and rec["mlp_extractor_class"] == "CriticPhaseMlpExtractor"
    assert rec["obs_rms_class"] == "PhaseSlotsSymmetricRunningMeanStd" and rec["n_train_envs"] == 4
    assert rec["value_phase_cols_max_abs_before"] == pytest.approx(0.3)
    after = p.state_dict()
    for k in sd:
        if k == "mlp_extractor.value_net.0.weight":
            assert torch.all(after[k][:, SL] == 0) and torch.equal(after[k][:, :60], sd[k][:, :60])
        else:
            assert torch.equal(after[k], sd[k]), k
    assert np.all(env.obs_rms.mean[SL] == 0.0) and np.all(env.obs_rms.var[SL] == 1.0)
    assert np.array_equal(env.obs_rms.mean[:60], rms_before[0][:60]) and env.obs_rms.count == rms_before[2]
    assert np.all(rms_before[1][SL] != 1.0)            # the reseed changed them
    assert all(type(e).__name__ == "PhaseObsFakeEnv" for e in env.venv.envs)
    assert H.mirror_phase_self_map(m._mirror.mirror_map.obs) and H.arm_state_problems("critic_phase", m, env) == []
    assert H.arm_state_problems("control", m, env) != []
    acts = torch.randn(64, ACT)
    with torch.no_grad():
        for fill in (0.0, 1.0, 0.5, -3.0, 7.0):
            y = x.clone()
            y[:, SL] = fill
            # the actor: bitwise the loaded actor (slots zero) whatever the slots hold; the critic at the load too
            assert torch.equal(p.get_distribution(y).mode(), mu_load)
            assert torch.equal(p.forward(y, deterministic=True)[0], mu_load)
            assert torch.equal(p.evaluate_actions(y, acts)[1], p.evaluate_actions(x0, acts)[1])
            assert torch.equal(p.predict_values(y), v_load)
    # the caller's tensor is never written
    xc = x.clone()
    p.get_distribution(x)
    assert torch.equal(x, xc)


def test_critic_phase_gradients_actor_zero_critic_nonzero():
    """One backward pass of a value + policy loss: the actor's first-layer phase columns get exactly zero gradient,
    the critic's (zero-initialised) get a nonzero one; autograd on the input: d mu / d slots = 0 exactly."""
    p, env, m = _phase_setup(4)
    H.install_arm("critic_phase", m, env)
    x = torch.randn(32, OBS)
    x[:, SL] = torch.rand(32, 3)
    acts = torch.randn(32, ACT)
    values, logp, _ = p.evaluate_actions(x, acts)
    loss = ((values.flatten() - 1.0) ** 2).mean() - logp.mean()
    p.optimizer.zero_grad()
    loss.backward()
    g_pi = p.mlp_extractor.policy_net[0].weight.grad
    g_vf = p.mlp_extractor.value_net[0].weight.grad
    assert torch.all(g_pi[:, SL] == 0) and torch.any(g_pi[:, :60] != 0)
    assert torch.any(g_vf[:, SL] != 0)
    z = x.clone().requires_grad_(True)
    gz = torch.autograd.grad(p.get_distribution(z).mode().sum(), z)[0]
    assert torch.all(gz[:, SL] == 0) and torch.any(gz[:, :60] != 0)
    p.optimizer.step()
    assert torch.equal(p.mlp_extractor.policy_net[0].weight[:, SL], torch.full_like(g_pi[:, SL], -0.2))
    assert float(p.mlp_extractor.value_net[0].weight[:, SL].detach().abs().max()) > 0.0
    assert H._adam_phase_max_abs(p, p.mlp_extractor.policy_net[0].weight) == 0.0
    assert H._adam_phase_max_abs(p, p.mlp_extractor.value_net[0].weight) > 0.0


def test_critic_phase_install_refusals():
    p, env, m = _phase_setup()
    env.obs_rms.mean = env.obs_rms.mean.copy()
    env.obs_rms.mean[H.PHASE_LO] = 1e-3                  # the slots carried a value
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    env.venv.envs[2]._command = np.array([0.0, 0.2, 0.0], dtype=np.float32)
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    env.venv = SimpleNamespace(envs=[])
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    perm = np.arange(OBS)
    perm[60], perm[59] = 59, 60
    m._mirror.mirror_map.obs.perm = perm
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    del m._mirror
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    x = torch.randn(8, OBS)
    loss = p.predict_values(x).sum() + p.get_distribution(x).mode().sum()
    p.optimizer.zero_grad()
    loss.backward()
    p.optimizer.step()                                  # nonzero Adam moments on the slot columns (x had values)
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    H.install_arm("critic_phase", m, env)
    with pytest.raises(H.HookError):                     # twice
        H.install_arm("critic_phase", m, env)


def test_phase_rms_keeps_slots_other_dims_as_control_and_pickles_stock():
    from environments.shared.mirror_kl_ppo import install_symmetric_obs_rms

    a, b = _slot_free_rms(), _slot_free_rms()            # b: control's statistics, the same updates
    H.reseed_phase_slots(a)
    a.__class__ = H.phase_rms_class(type(a))
    c0 = H.phase_counter(a)["calls"]
    for j in range(4):
        data = np.random.default_rng(60 + j).normal(size=(13, OBS))
        b.update(np.where(np.arange(OBS) >= 60, 0.0, data))
        data[:, SL] = H.phase_matrix(np.arange(13) * 37)            # the phase in the slots
        a.update(data)
    assert H.phase_counter(a)["calls"] == c0 + 4 and H.phase_slots_frozen(a)
    assert np.array_equal(a.mean[:60], b.mean[:60]) and np.array_equal(a.var[:60], b.var[:60]) and a.count == b.count
    for clone in (pickle.loads(pickle.dumps(a)), copy.deepcopy(a)):
        assert type(clone) is RunningMeanStd and H.phase_slots_frozen(clone) and np.array_equal(clone.mean, a.mean)
    vn = SimpleNamespace(obs_rms=a, norm_obs=True)
    install_symmetric_obs_rms(vn, SimpleNamespace(obs=SimpleNamespace(perm=np.arange(OBS), sign=np.ones(OBS),
                                                                      dim=OBS)))
    assert vn.obs_rms is a                              # MirrorKLPPO.learn()'s _mirror_install keeps it


def test_phase_mirror_map_on_the_real_map():
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.symmetry import MirrorMap, SignedPermutation

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    mm = MirrorMap.load(mid, current_plant_identity(mid), verify=False)
    assert mm.obs.sign[61] == -1.0 and mm.obs.sign[62] == -1.0 and mm.obs.sign[60] == 1.0
    new = H.phase_mirror_map(mm)
    assert isinstance(new.obs, SignedPermutation) and new.action is mm.action and new.sha256 == mm.sha256
    assert np.array_equal(new.obs.perm, mm.obs.perm) and np.all(new.obs.sign[SL] == 1.0)
    assert np.array_equal(new.obs.sign[:60], mm.obs.sign[:60]) and new.obs.names[60] == "critic_phase.ramp"
    assert mm.obs.sign[61] == -1.0                      # the run's map object is not written
    x = np.random.default_rng(1).normal(size=(5, OBS)).astype(np.float32)
    x[:, SL] = [0.3, 1.0, 0.7]
    assert np.array_equal(new.obs.apply(x)[:, SL], x[:, SL])


class PolicyProbeCallback:                              # a stand-in with the real class's name and bank shape
    def __init__(self, bank):
        self.bank = bank
        self.seen = []

    def _row(self):
        self.seen.append(np.array(self.bank.raw_obs, copy=True))
        return {"row": len(self.seen)}


def _bank():
    from typing import NamedTuple

    class Bank(NamedTuple):
        labels: tuple
        steps: np.ndarray
        raw_obs: np.ndarray

    return Bank(labels=("statue_settled", "statue", "ankle_minus_0.05_at400"), steps=np.array([199, 599, 400]),
                raw_obs=np.zeros((3, OBS)))


def test_probe_phase_feeds_the_bank_phase_and_restores_the_bank():
    probe = PolicyProbeCallback(_bank())
    rec = H.probe_phase("critic_phase", [SimpleNamespace(), probe])
    assert rec["installed"] and rec["k"] == [200, 600, 401] and rec["phase_fed"] == [[1.0, 1.0, 1.0]] * 3
    assert rec["class"] == "PhaseFedPolicyProbeCallback" and all(rec["settled"])
    bank = probe.bank
    assert probe._row() == {"row": 1} and probe.bank is bank and np.all(bank.raw_obs == 0.0)
    assert np.all(probe.seen[0][:, SL] == 1.0) and np.all(probe.seen[0][:, :60] == 0.0)
    with pytest.raises(H.HookError):
        H.probe_phase("critic_phase", [probe])          # already feeding
    plain = PolicyProbeCallback(_bank())
    rec = H.probe_phase("control", [plain])
    assert rec["installed"] is False and type(plain) is PolicyProbeCallback and rec["phase_fed"] is None
    with pytest.raises(H.HookError):
        H.probe_phase("control", [plain, PolicyProbeCallback(_bank())])
    with pytest.raises(H.HookError):
        H.probe_phase("control", [])
    with pytest.raises(H.HookError):
        H.probe_phase("critic_phase", [PolicyProbeCallback(None)])


def test_phase_evidence_zero_init_then_response():
    ev = _phase_evidence(3)
    s = H.summarize(ev)
    assert H.evidence_problems("critic_phase", s, n_expected=3) == [], H.evidence_problems("critic_phase", s)
    assert s["phase_n_zero_init"] == 1 and s["phase_n_updated"] == 2 and s["max_dmu_phase"] == 0.0
    assert s["zero_init_max_dv_phase"] == 0.0 and s["zero_init_max_value_phase_cols"] == 0.0
    assert all(v > 0 for v in s["updated_min_dv_phase"].values()) and s["max_phase_buffer_max_err"] == 0.0
    assert ev[0]["n_updates_since_load"] == 0 and ev[2]["n_updates_since_load"] == 14
    assert ev[0]["phase_k_start_consistent"] and ev[0]["phase_rows_presettle"] > 0
    assert ev[0]["dv_predict_values"] > 0                # the critic still sees observation 6
    assert H.evidence_problems("control", s, n_expected=3) != []


@pytest.mark.parametrize("mutate", ["unmask", "nonzero_init", "no_learning", "slot_error", "counter", "unfreeze",
                                    "mirror", "actor_cols"])
def test_phase_evidence_mutants_fail(mutate):
    s = H.summarize(_phase_evidence(3, mutate=mutate))
    assert H.evidence_problems("critic_phase", s, n_expected=3) != [], mutate


def test_phase_evidence_needs_its_context():
    p, env, m = _phase_setup()
    H.install_arm("critic_phase", m, env)
    buf, last, dones = _phase_buffer(p, env, [0, 0, 0, 0], seed=1)
    m.rollout_buffer = buf
    with pytest.raises(H.HookError):
        H.rollout_evidence("critic_phase", m, env, {"values": torch.as_tensor(last), "dones": dones},
                           start_obs_digest=None, phase_ctx=None)


def test_start_pair_actor_and_critic_bitwise_after_the_phase_transform():
    """The real start pair (read only): after the critic's phase columns are zeroed and the slots reseeded, the bank's
    actions and values with ANY phase fed equal the loaded pair's on the bank as recorded, bitwise."""
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    model, norm, _ = load_sb3_checkpoint(C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"]),
                                         lambda: env_class(mid)(**ctor))
    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        bank = np.asarray(z["bank_raw_obs"], dtype=np.float64)
    p = model.policy
    rms = norm.obs_rms

    def outs(raw, r):
        t = torch.as_tensor(np.clip((raw - r.mean) / np.sqrt(r.var + norm.epsilon), -10, 10), dtype=torch.float32)
        with torch.no_grad():
            return p.get_distribution(t).mode().clone(), p.predict_values(t).clone()

    mu0, v0 = outs(bank, rms)
    assert float(p.mlp_extractor.value_net[0].weight[:, SL].detach().abs().max()) > 0.1
    with torch.no_grad():
        p.mlp_extractor.value_net[0].weight[:, SL] = 0.0
    r2 = copy.deepcopy(rms)
    H.reseed_phase_slots(r2)
    p.mlp_extractor.__class__ = H.critic_phase_class(MlpExtractor)
    for k in (0, 50, 199, 200, 600, 1000):
        fed = bank.copy()
        fed[:, SL] = H.phase_matrix([k] * len(bank))
        mu1, v1 = outs(fed, r2)
        assert torch.equal(mu0, mu1) and torch.equal(v0, v1), k
