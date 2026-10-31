"""Unit tests of the ARM HOOK (bs_hooks.py) on SB3 2.9.0's own classes: no environment steps, no training.

    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS>/tools/tests/test_bs_hooks.py

The critic-blind hook on a real ActorCriticPolicy (small and the start pair's own), the frozen statistics (stock RMS
and the code copy's SymmetricRunningMeanStd: no-op update, pickling and deep-copying as the stock class, kept by
MirrorKLPPO's install), install_arm's refusals, the independent GAE / Monte Carlo recomputation against SB3's
RolloutBuffer, rollout_evidence and evidence_problems for every arm (and mutants that must fail).
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
    lam = C.ARM_GAE_LAMBDA[arm]
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


@pytest.mark.parametrize("arm", C.ARMS)
def test_evidence_passes_for_each_arm(arm):
    ev = _evidence(arm)
    s = H.summarize(ev)
    assert H.evidence_problems(arm, s, n_expected=3) == []
    for other in C.ARMS:
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
