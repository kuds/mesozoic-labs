"""MirrorKLPPO (environments/shared/mirror_kl_ppo.py): the symmetric observation statistics, the tied log_std, the
mirror loss, the copied PPO.train, bitwise equality with stock PPO when the loss and the mirrored critic are off,
stock checkpoints, and the train_base wiring of the [ppo] symmetry keys.  Needs stable-baselines3 (the SB3 job)."""

from __future__ import annotations

import copy
import json
import pickle
import zipfile
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared import mirror_kl_ppo as MK  # noqa: E402
from environments.shared import symmetry as SY  # noqa: E402
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")


@pytest.fixture(scope="module")
def plant():
    return current_plant_identity(TYRANNOSAURUS_REX)


@pytest.fixture(scope="module")
def mirror_map(plant):
    return SY.MirrorMap.load(TYRANNOSAURUS_REX, plant)


# ---------------------------------------------------------------------------------------------------- statistics
def test_symmetric_statistics_are_bitwise_symmetric_and_pickle_as_stock(mirror_map):
    rng = np.random.default_rng(0)
    rms = MK.SymmetricRunningMeanStd(mirror_map.obs, shape=(63,))
    reference = RunningMeanStd(shape=(63,))
    for _ in range(300):
        x = rng.normal(0.5, 2.0, size=(4, 63)) * np.linspace(0.1, 5, 63)
        rms.update(x)
        reference.update(np.concatenate([x, mirror_map.obs.apply(x)]))
        assert rms.residual() == (0.0, 0.0)
    # The moments of [x; M_o x], as stock statistics would compute them on the doubled batch.
    assert np.max(np.abs(rms.mean - reference.mean)) < 1e-12
    assert np.max(np.abs(rms.var - reference.var) / reference.var) < 1e-10
    o = rng.normal(0, 40, size=(2000, 63))

    def normalize(v):
        return np.clip((v - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)

    np.testing.assert_array_equal(normalize(mirror_map.obs.apply(o)), mirror_map.obs.apply(normalize(o)))
    for clone in (pickle.loads(pickle.dumps(rms)), copy.deepcopy(rms), rms.copy()):
        assert type(clone) is RunningMeanStd
        assert np.array_equal(clone.mean, rms.mean) and np.array_equal(clone.var, rms.var)
        assert clone.count == rms.count
    assert b"SymmetricRunningMeanStd" not in pickle.dumps(rms)
    lopsided = RunningMeanStd(shape=(63,))
    lopsided.mean[int(np.flatnonzero(mirror_map.obs.perm != np.arange(63))[0])] = 1.0
    with pytest.raises(SY.SymmetryError, match="not mirror-symmetric"):
        MK.SymmetricRunningMeanStd.from_stock(lopsided, mirror_map.obs)


# ---------------------------------------------------------------------------------------------------- tie and loss
def _policy():
    from stable_baselines3.common.policies import ActorCriticPolicy

    torch.manual_seed(0)
    return ActorCriticPolicy(
        gym.spaces.Box(-np.inf, np.inf, (63,), np.float32),
        gym.spaces.Box(-1, 1, (15,), np.float32),
        lr_schedule=lambda _: 1e-3,
        net_arch=[64, 32],
        log_std_init=-1.5,
    )


def test_the_log_std_tie_survives_the_gradient_clip_and_adam(mirror_map):
    policy = _policy()
    handle = MK.register_log_std_tie(policy, mirror_map)
    weights = torch.arange(15, dtype=torch.float32) + 1.0
    for _ in range(3):
        policy.optimizer.zero_grad()
        (torch.exp(policy.log_std) * weights).sum().backward()
        grad = policy.log_std.grad.numpy()
        assert np.array_equal(grad, grad[mirror_map.action.perm])
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        grad = policy.log_std.grad.numpy()
        assert np.array_equal(grad, grad[mirror_map.action.perm])
        policy.optimizer.step()
        assert MK.log_std_pair_gap(policy, mirror_map) == 0.0
    assert not np.all(policy.log_std.detach().numpy() == -1.5)  # it did train, pairwise
    handle.remove()
    with torch.no_grad():
        policy.log_std[3] += 0.1
    with pytest.raises(SY.SymmetryError, match="not equal"):
        MK.register_log_std_tie(policy, mirror_map)


def test_the_mirror_loss_is_the_kl_with_sigma_detached_and_floored(mirror_map):
    rng = np.random.default_rng(1)
    mu = torch.tensor(rng.normal(size=(128, 15)), dtype=torch.float64)
    mu_mirrored = torch.tensor(rng.normal(size=(128, 15)), dtype=torch.float64, requires_grad=True)
    log_std = rng.normal(-2.5, 1.0, size=15)
    log_std[0] = -5.0  # sigma 0.0067: the floor binds on this channel
    log_std_t = torch.tensor(log_std, dtype=torch.float64, requires_grad=True)
    perm, sign = torch.as_tensor(mirror_map.action.perm), torch.as_tensor(mirror_map.action.sign)
    loss, asymmetry = MK.mirror_kl_loss(mu_mirrored, mu, log_std_t, perm, sign, 0.01)
    sigma = np.maximum(np.exp(log_std), 0.01)
    deviation = mu_mirrored.detach().numpy() - mirror_map.action.apply(mu.numpy())
    assert abs(loss.item() - np.mean(np.sum(deviation**2 / (2 * sigma**2), axis=1))) <= 1e-12 * loss.item()
    assert abs(asymmetry.item() - np.mean(np.max(np.abs(deviation), axis=1)) / 2) <= 1e-12
    grad_mu, grad_log_std = torch.autograd.grad(loss, [mu_mirrored, log_std_t], allow_unused=True)
    assert grad_log_std is None  # dL_K / dlog_std == 0: sigma is held fixed
    assert torch.all(torch.isfinite(grad_mu))


def test_an_equivariant_policy_pays_no_mirror_loss(mirror_map):
    """``(f(s) + M_a f(M_o s)) / 2`` is exactly equivariant; a common-mode hip roll is not."""
    policy = _policy()
    obs = torch.tensor(np.random.default_rng(2).normal(size=(256, 63)), dtype=torch.float32)
    obs_perm, obs_sign = torch.as_tensor(mirror_map.obs.perm), torch.as_tensor(mirror_map.obs.sign, dtype=torch.float32)
    act_perm, act_sign = (
        torch.as_tensor(mirror_map.action.perm),
        torch.as_tensor(mirror_map.action.sign, dtype=torch.float32),
    )

    def symmetric_mean(o):
        f = policy.get_distribution(o).mode()
        f_mirrored = policy.get_distribution(o[:, obs_perm] * obs_sign).mode()
        return (f + f_mirrored[:, act_perm] * act_sign) / 2

    with torch.no_grad():
        mirrored_obs = obs[:, obs_perm] * obs_sign
        loss, _ = MK.mirror_kl_loss(
            symmetric_mean(mirrored_obs), symmetric_mean(obs), policy.log_std, act_perm, act_sign, 0.01
        )
        assert loss.item() == 0.0
        lean = torch.zeros(15)
        lean[4] = lean[8] = 0.01  # the same hip roll on both legs: odd under M_a
        loss, _ = MK.mirror_kl_loss(
            symmetric_mean(mirrored_obs) + lean, symmetric_mean(obs) + lean, policy.log_std, act_perm, act_sign, 0.01
        )
        assert loss.item() > 0.0


def test_train_is_stock_ppo_train_plus_four_marked_lines():
    MK.check_sb3_pin()
    assert MK.stock_ppo_train_sha256() == MK.SB3_PPO_TRAIN_SHA256
    diff = MK.train_source_diff()
    assert diff["removed"] == []
    assert len(diff["added"]) == 4 and all(MK.MIRROR_MARKER in line for line in diff["added"])


# ---------------------------------------------------------------------------------------------------- training
class ToyEnv(gym.Env):
    """The T. rex interface's shapes (63 observations, 15 actions), random dynamics, 50-step episodes."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (63,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (15,), np.float32)

    def __init__(self):
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.np_random.normal(0.3, 1.5, 63).astype(np.float32), {}

    def step(self, action):
        self.t += 1
        obs = self.np_random.normal(0.3, 1.5, 63).astype(np.float32)
        reward = float(-np.sum(np.square(action)) + obs[3] * action[4] - 0.1 * obs[7])
        return obs, reward, False, self.t >= 50, {}


def _venv(seed: int) -> VecNormalize:
    venv = VecNormalize(DummyVecEnv([lambda: Monitor(ToyEnv()) for _ in range(2)]), gamma=0.98)
    venv.seed(seed)
    return venv


_PPO_KWARGS: dict[str, Any] = {
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 2,
    "learning_rate": 3e-4,
    "target_kl": None,
    "policy_kwargs": {"net_arch": [32, 32], "log_std_init": -1.5},
    "device": "cpu",
}


def _make(seed: int = 3) -> PPO:
    return PPO("MlpPolicy", _venv(seed), seed=seed, **_PPO_KWARGS)


def _parameters(model) -> dict:
    return {key: value.detach().numpy().copy() for key, value in model.policy.state_dict().items()}


def _adam(model) -> dict:
    state = model.policy.optimizer.state_dict()["state"]
    return {key: {name: np.asarray(value[name]).copy() for name in value} for key, value in state.items()}


def test_with_the_loss_and_the_mirrored_critic_off_it_trains_bitwise_as_stock_ppo(mirror_map):
    """kappa 0 and no mirrored critic: MirrorKLPPO.train equals stock PPO.train, given the same symmetric
    statistics and log_std tie, in every parameter and Adam moment after two updates."""
    stock = _make(seed=13)  # each model is built (SB3 reseeds the global RNGs) right before it learns
    MK.install_symmetric_obs_rms(stock.get_vec_normalize_env(), mirror_map)
    MK.register_log_std_tie(stock.policy, mirror_map)
    stock.learn(total_timesteps=256)
    mirrored = MK.enable_mirror_kl(_make(seed=13), SY.SymmetryConfig(kappa=0.0, mirror_critic=False), mirror_map)
    mirrored.learn(total_timesteps=256)
    a, b = _parameters(stock), _parameters(mirrored)
    assert a.keys() == b.keys() and all(np.array_equal(a[key], b[key]) for key in a)
    a, b = _adam(stock), _adam(mirrored)
    assert a.keys() == b.keys() and all(np.array_equal(a[k][n], b[k][n]) for k in a for n in a[k])
    assert stock.num_timesteps == mirrored.num_timesteps == 256
    # The comparison can fail: either addition on, the parameters differ.
    for config in (SY.SymmetryConfig(mirror_critic=False), SY.SymmetryConfig(kappa=0.0)):
        other = MK.enable_mirror_kl(_make(seed=13), config, mirror_map)
        other.learn(total_timesteps=256)
        assert not all(
            np.array_equal(a_value, b_value)
            for a_value, b_value in zip(_parameters(stock).values(), _parameters(other).values())
        )


def test_mirror_kl_ppo_trains_symmetrically_and_saves_a_stock_checkpoint(mirror_map, tmp_path):
    model = MK.enable_mirror_kl(_make(), SY.SymmetryConfig(), mirror_map)
    assert type(model) is MK.MirrorKLPPO
    venv = model.get_vec_normalize_env()
    model.learn(total_timesteps=256)
    logged = model.logger.name_to_value
    assert {"train/mirror_kl_loss", "train/mirror_asymmetry", "train/mirror_value_loss"} <= set(logged)
    assert np.isfinite(logged["train/mirror_kl_loss"]) and logged["train/mirror_kl_loss"] >= 0.0
    assert isinstance(venv.obs_rms, MK.SymmetricRunningMeanStd) and venv.obs_rms.residual() == (0.0, 0.0)
    assert MK.log_std_pair_gap(model.policy, mirror_map) == 0.0
    model.save(tmp_path / "mirror.zip")
    venv.save(tmp_path / "vecnormalize.pkl")
    _make().save(tmp_path / "stock.zip")
    data = json.loads(zipfile.ZipFile(tmp_path / "mirror.zip").read("data"))
    stock_keys = set(json.loads(zipfile.ZipFile(tmp_path / "stock.zip").read("data")))
    assert set(data) == stock_keys | {SY.CHECKPOINT_ATTRIBUTE}
    assert data[SY.CHECKPOINT_ATTRIBUTE] == SY.SymmetryConfig().checkpoint_record(mirror_map)
    with open(tmp_path / "vecnormalize.pkl", "rb") as handle:
        assert type(pickle.load(handle).obs_rms) is RunningMeanStd
    loaded = PPO.load(tmp_path / "mirror.zip", device="cpu")
    assert type(loaded) is PPO
    obs = np.random.default_rng(5).normal(size=(8, 63)).astype(np.float32)
    np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0])


def test_enable_refuses_what_it_cannot_mirror(mirror_map):
    from stable_baselines3 import A2C

    with pytest.raises(SY.SymmetryError, match="stock PPO"):
        MK.enable_mirror_kl(A2C("MlpPolicy", _venv(0), device="cpu"), SY.SymmetryConfig(), mirror_map)
    with pytest.raises(SY.SymmetryError, match="Gaussian"):
        MK.enable_mirror_kl(PPO("MlpPolicy", _venv(0), use_sde=True, device="cpu"), SY.SymmetryConfig(), mirror_map)
    lopsided = _make()
    lopsided.get_vec_normalize_env().obs_rms.update(np.random.default_rng(0).normal(size=(8, 63)))
    with pytest.raises(SY.SymmetryError, match="not mirror-symmetric"):
        MK.enable_mirror_kl(lopsided, SY.SymmetryConfig(), mirror_map)
    untied = _make()
    with torch.no_grad():
        untied.policy.log_std[4] += 0.1
    with pytest.raises(SY.SymmetryError, match="not equal"):
        MK.enable_mirror_kl(untied, SY.SymmetryConfig(), mirror_map)


# ---------------------------------------------------------------------------------------------------- train_base
def test_the_ppo_symmetry_keys_select_mirror_kl_ppo_on_fresh_and_resumed_models(plant, mirror_map, tmp_path):
    from environments.shared.policy_loading import _ensure_sb3
    from environments.shared.train_base import _create_or_load_model

    sb3 = _ensure_sb3()
    kwargs = {key: value for key, value in _PPO_KWARGS.items()}
    config = SY.SymmetryConfig()

    def create(load_path=None, symmetry=None, mode="resume_same_stage"):
        return _create_or_load_model(
            sb3, "ppo", dict(kwargs), _venv(3), load_path, plant_identity=plant, task_load_mode=mode, symmetry=symmetry
        )

    stock = create()
    assert type(stock) is PPO and SY.CHECKPOINT_ATTRIBUTE not in vars(stock)
    fresh = create(symmetry=config)
    assert type(fresh) is MK.MirrorKLPPO
    assert getattr(fresh, SY.CHECKPOINT_ATTRIBUTE) == config.checkpoint_record(mirror_map)
    fresh.save(tmp_path / "mirror.zip")
    stock.save(tmp_path / "stock.zip")
    resumed = create(str(tmp_path / "mirror.zip"), symmetry=config)
    assert type(resumed) is MK.MirrorKLPPO
    # A same-stage resume continues the training its checkpoint records, both ways.
    with pytest.raises(SY.SymmetryError, match="declares no symmetry"):
        create(str(tmp_path / "mirror.zip"))
    with pytest.raises(SY.SymmetryError, match="records none"):
        create(str(tmp_path / "stock.zip"), symmetry=config)
    with pytest.raises(SY.SymmetryError, match="the stage"):
        create(str(tmp_path / "mirror.zip"), symmetry=SY.SymmetryConfig(kappa=0.5))
    # A warm start into a stage trained without symmetry does not inherit the record.
    child = create(str(tmp_path / "mirror.zip"), mode="initialize_next_stage")
    assert type(child) is PPO and SY.CHECKPOINT_ATTRIBUTE not in vars(child)


def test_no_stage_config_enables_symmetry():
    from environments.shared.species_names import model_ids
    from environments.shared.stage_manifest import load_stage_manifest

    for model_id in model_ids():
        for entry in load_stage_manifest(model_id).stages:
            ppo = dict(load_stage_config(model_id, entry.reference).get("ppo_kwargs", {}))
            assert not set(SY.SYMMETRY_KEYS) & set(ppo), (model_id, entry.id)
            assert SY.pop_symmetry_config(ppo) is None
