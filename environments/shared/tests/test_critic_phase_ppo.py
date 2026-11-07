"""The critic's episode phase for SB3 PPO (environments/shared/critic_phase_ppo.py): the training env's phase
columns and its stock sidecar, the time-limit bootstrap, the actor's exact blindness on every path, the critic's
dependence after one update, the mirror, save / load / resume through train_base, and the switch off.  Needs
stable-baselines3 (the SB3 job)."""

from __future__ import annotations

import json
import pickle
import zipfile
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.buffers import RolloutBuffer  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared import critic_phase as CP  # noqa: E402
from environments.shared import critic_phase_ppo as CPP  # noqa: E402
from environments.shared import mirror_kl_ppo as MK  # noqa: E402
from environments.shared import symmetry as SY  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
SETTLE, HORIZON, GAMMA = 5, 12, 0.98
SPEC = CP.CriticPhaseSpec(settle_steps=SETTLE, gamma=GAMMA)


@pytest.fixture(scope="module")
def plant():
    return current_plant_identity(TYRANNOSAURUS_REX)


@pytest.fixture(scope="module")
def mirror_map(plant):
    return SY.MirrorMap.load(TYRANNOSAURUS_REX, plant)


class PhaseToyEnv(gym.Env):
    """The T. rex interface's shapes (63 observations, 15 actions) with a BaseDinoEnv-style step counter: info's
    ``step`` is ``_step_count`` after the increment, a reward gated on ``_step_count > settle`` and a time limit."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (63,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (15,), np.float32)

    def __init__(self, settle: int = SETTLE, horizon: int = HORIZON, info_step: bool = True):
        self.stance_width_settle_steps = settle
        self.horizon = horizon
        self.info_step = info_step
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self.np_random.normal(0.3, 1.5, 63).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self.np_random.normal(0.3, 1.5, 63).astype(np.float32)
        gate = 1.0 if self._step_count > self.stance_width_settle_steps else 0.0
        reward = float(gate - 0.1 * np.sum(np.square(action)) + obs[3] * action[4])
        info = {"step": self._step_count} if self.info_step else {}
        return obs, reward, False, self._step_count >= self.horizon, info


def _envs(n: int = 2, **kwargs: Any) -> DummyVecEnv:
    return DummyVecEnv([lambda: Monitor(PhaseToyEnv(**kwargs)) for _ in range(n)])


def _venv(seed: int, *, phase: bool = True, gamma: float = GAMMA, **kwargs: Any) -> VecNormalize:
    venv = (
        CPP.CriticPhaseVecNormalize(_envs(**kwargs), critic_phase_gamma=gamma, gamma=gamma)
        if phase
        else VecNormalize(_envs(**kwargs), gamma=gamma)
    )
    venv.seed(seed)
    return venv


_PPO_KWARGS: dict[str, Any] = {
    "n_steps": 32,
    "batch_size": 32,
    "n_epochs": 2,
    "learning_rate": 3e-4,
    "gamma": GAMMA,
    "target_kl": None,
    "device": "cpu",
}
_POLICY_KWARGS = {"net_arch": [32, 32], "log_std_init": -1.5}


def _make(seed: int = 3, **venv_kwargs: Any) -> PPO:
    venv = _venv(seed, **venv_kwargs)
    return PPO(env=venv, seed=seed, **CPP.new_model_kwargs(venv.critic_phase, _POLICY_KWARGS), **_PPO_KWARGS)


def _parameters(model) -> dict:
    return {key: value.detach().numpy().copy() for key, value in model.policy.state_dict().items()}


def _adam(model) -> dict:
    state = model.policy.optimizer.state_dict()["state"]
    return {key: {name: np.asarray(value[name]).copy() for name in value} for key, value in state.items()}


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[key], b[key]) for key in a)


# ---------------------------------------------------------------------------------------------------- 1. the env
def test_the_training_env_appends_each_env_s_own_phase_and_normalizes_as_stock():
    phase, stock = _venv(7, settle=20), _venv(7, settle=20, phase=False)
    assert phase.observation_space == stock.observation_space == PhaseToyEnv.observation_space
    assert phase.critic_phase == CP.CriticPhaseSpec(20, GAMMA)
    assert phase.critic_observation_space.shape == (66,)
    spec = phase.critic_phase
    rng = np.random.default_rng(0)
    obs, reference = phase.reset(), stock.reset()
    assert obs.dtype == np.float32 and obs.shape == (2, 66)
    np.testing.assert_array_equal(obs[:, :63], reference)
    np.testing.assert_array_equal(obs[:, 63:], spec.matrix([0, 0]).astype(np.float32))
    truncations = 0
    for _ in range(3 * HORIZON + 1):
        actions = rng.uniform(-1, 1, (2, 15)).astype(np.float32)
        obs, rewards, dones, infos = phase.step(actions)
        reference, stock_rewards, stock_dones, stock_infos = stock.step(actions)
        np.testing.assert_array_equal(obs[:, :63], reference)
        np.testing.assert_array_equal(rewards, stock_rewards)
        np.testing.assert_array_equal(dones, stock_dones)
        # k is each env's own counter now (0 after the VecEnv reset a finished env).
        counters = [env.unwrapped._step_count for env in phase.venv.envs]
        np.testing.assert_array_equal(obs[:, 63:], spec.matrix(counters).astype(np.float32))
        for index in np.flatnonzero(dones):
            assert counters[index] == 0
            terminal = infos[index]["terminal_observation"]
            assert terminal.shape == (66,) and infos[index]["TimeLimit.truncated"]
            np.testing.assert_array_equal(terminal[:63], stock_infos[index]["terminal_observation"])
            # The finished episode's last observation: phase(HORIZON), not the reset's phase(0).
            np.testing.assert_array_equal(terminal[63:], np.float32(spec.features(HORIZON)))
            assert infos[index]["step"] == HORIZON
            truncations += 1
    assert truncations == 6
    # The statistics are the env's width and the stock ones, bitwise; the phase is never normalised.
    for name in ("mean", "var", "count"):
        assert np.array_equal(getattr(phase.obs_rms, name), getattr(stock.obs_rms, name))
        assert np.array_equal(getattr(phase.ret_rms, name), getattr(stock.ret_rms, name))
    assert phase.obs_rms.mean.shape == (63,)


def test_the_phase_is_bounded_where_running_statistics_would_blow_up():
    # settle 20 > horizon 12: `settled` is 0 in every observation; normalising it would divide by sqrt(1e-8).
    venv = _venv(1, settle=20)
    venv.reset()
    for _ in range(30):
        obs, *_ = venv.step(np.zeros((2, 15), np.float32))
        assert np.all(obs[:, 63:] >= 0.0) and np.all(obs[:, 63:] <= 1.0) and np.all(obs[:, 64] == 0.0)
    # The settled flag's first 1 is exactly 1.0, never clipped to 10.
    venv = _venv(1, settle=3, horizon=6)
    venv.reset()
    seen = [venv.step(np.zeros((2, 15), np.float32))[0][0, 63:].copy() for _ in range(5)]
    assert [row[1] for row in seen] == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_the_sidecar_is_the_stock_vecnormalize_pickle_byte_for_byte(tmp_path):
    phase, stock = _venv(5), _venv(5, phase=False)
    phase.reset(), stock.reset()
    rng = np.random.default_rng(3)
    for _ in range(20):
        actions = rng.uniform(-1, 1, (2, 15)).astype(np.float32)
        phase.step(actions), stock.step(actions)
    phase.save(tmp_path / "phase.pkl")
    stock.save(tmp_path / "stock.pkl")
    payload = (tmp_path / "phase.pkl").read_bytes()
    assert payload == (tmp_path / "stock.pkl").read_bytes()
    assert b"critic_phase" not in payload and b"CriticPhase" not in payload
    loaded = VecNormalize.load(str(tmp_path / "phase.pkl"), DummyVecEnv([lambda: PhaseToyEnv()]))
    assert type(loaded) is VecNormalize and loaded.observation_space.shape == (63,)
    assert type(loaded.obs_rms) is RunningMeanStd and np.array_equal(loaded.obs_rms.mean, phase.obs_rms.mean)
    # The live wrapper still carries its phase after saving.
    assert phase.critic_phase == SPEC and phase.reset().shape == (2, 66)


def test_the_training_env_refuses_an_env_it_cannot_phase():
    class NoSettle(PhaseToyEnv):
        def __init__(self):
            super().__init__()
            del self.stance_width_settle_steps

    with pytest.raises(CP.CriticPhaseError, match="stance_width_settle_steps"):
        CPP.CriticPhaseVecNormalize(DummyVecEnv([lambda: Monitor(NoSettle())]), critic_phase_gamma=GAMMA)
    mixed = DummyVecEnv([lambda: PhaseToyEnv(settle=5), lambda: PhaseToyEnv(settle=6)])
    with pytest.raises(CP.CriticPhaseError, match="disagree"):
        CPP.CriticPhaseVecNormalize(mixed, critic_phase_gamma=GAMMA)
    silent = _venv(0, info_step=False)
    silent.reset()
    with pytest.raises(CP.CriticPhaseError, match="info\\['step'\\]"):
        silent.step(np.zeros((2, 15), np.float32))


def test_the_tyrannosaurus_rex_env_s_counter_gates_and_horizon_through_create_vec_env():
    """The real stance env through train_base.create_vec_env: k is the env's _step_count, settled turns 1 at the
    observation just before the first gated reward, and the horizon's truncation carries phase(1000)."""
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config
    from environments.shared.train_base import create_vec_env

    species = get_species_config(TYRANNOSAURUS_REX)
    configs = load_all_stages(TYRANNOSAURUS_REX)
    venv = create_vec_env(species, configs, 1, 1, seed=3042, algorithm="ppo", gamma=0.98, critic_phase_gamma=0.98)
    assert type(venv) is CPP.CriticPhaseVecNormalize and venv.critic_phase == CP.CriticPhaseSpec(200, 0.98)
    assert type(create_vec_env(species, configs, 1, 1, seed=3042, algorithm="ppo", gamma=0.98)) is VecNormalize
    env = venv.venv.envs[0].unwrapped
    obs = venv.reset()
    np.testing.assert_array_equal(obs[0, 63:], np.float32(venv.critic_phase.features(0)))
    first_paid = first_settled = None
    for step in range(1, 1001):
        obs, _rewards, dones, infos = venv.step(np.zeros((1, env.action_space.shape[0]), np.float32))
        info = infos[0]
        assert info["step"] == step
        if not dones[0]:
            assert env._step_count == step
            np.testing.assert_array_equal(obs[0, 63:], np.float32(venv.critic_phase.features(step)))
            if first_settled is None and obs[0, 64] == 1.0:
                first_settled = step
        if first_paid is None and info["reward_stance_width"] != 0.0:
            first_paid = step
    assert dones[0] and info["TimeLimit.truncated"], "the zero-action statue stands the 1000-step horizon"
    np.testing.assert_array_equal(info["terminal_observation"][63:], np.ones(3, np.float32))
    assert env._step_count == 0 and obs[0, 63:].tolist() == [0.0, 0.0, float(np.float32(0.98**200))]
    assert (first_settled, first_paid) == (200, 201)


# ---------------------------------------------------------------------------------------------------- 2. bootstrap
class _Rewards(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards: list[np.ndarray] = []

    def _on_step(self) -> bool:
        self.rewards.append(np.array(self.locals["rewards"], copy=True))  # before PPO adds the bootstrap
        return True


def test_a_time_limit_truncation_bootstraps_on_the_phase_of_the_episode_s_last_step():
    model = _make(settle=20)  # phase(12) = (0.6, 0, 0.98 ** 8): neither the reset's phase nor the settled one
    policy = model.policy
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original = policy.predict_values

    def recording(observations):
        values = original(observations)
        calls.append((observations.detach().numpy().copy(), values.detach().numpy().copy()))
        return values

    policy.predict_values = recording
    rewards = _Rewards()
    model.learn(total_timesteps=64, callback=rewards)
    del policy.predict_values
    terminal = [(obs, value) for obs, value in calls if obs.shape == (1, 66)]
    assert len(terminal) == 4  # two truncations per env in 32 steps (steps 12 and 24)
    for obs, _value in terminal:
        np.testing.assert_array_equal(obs[0, 63:], np.float32(CP.CriticPhaseSpec(20, GAMMA).features(HORIZON)))
    # ...and the bootstrap is gamma * V(terminal observation with its phase), added to the truncated step's reward.
    buffer = model.rollout_buffer
    added = [
        (step, env)
        for step in range(32)
        for env in range(2)
        if buffer.rewards[step, env] != np.float32(rewards.rewards[step][env])
    ]
    assert [step for step, _env in added] == [11, 11, 23, 23]
    for (step, env), (_obs, value) in zip(added, terminal):
        assert buffer.rewards[step, env] == np.float32(rewards.rewards[step][env] + GAMMA * value[0, 0])
    # The last observation's value (the GAE bootstrap) is taken on the phase-carrying observation too.
    assert calls[-1][0].shape == (2, 66)


# ---------------------------------------------------------------------------------------------------- 3. the actor
def _phases(rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    return {
        "own": SPEC.matrix(rng.integers(0, 40, rows)),
        "zero": np.zeros((rows, 3)),
        "huge": np.full((rows, 3), 1e6),
        "noise": rng.normal(0, 100, (rows, 3)),
        "nan": np.full((rows, 3), np.nan),
    }


def _with(obs: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return np.concatenate([obs, phase], axis=1).astype(np.float32)


def test_the_actor_is_exactly_blind_to_the_phase_on_every_path(mirror_map):
    from environments.shared.k3_panel import policy_mean_action

    model = MK.enable_mirror_kl(_make(), SY.SymmetryConfig(), mirror_map)
    model.learn(total_timesteps=64)  # one update: trained weights, the mirror installed
    policy = model.policy
    extractor = policy.mlp_extractor
    assert extractor.policy_net[0].weight.shape == (32, 63) and extractor.value_net[0].weight.shape == (32, 66)
    rng = np.random.default_rng(4)
    obs = rng.normal(0, 1.5, (16, 63)).astype(np.float32)
    actions = torch.as_tensor(rng.uniform(-1, 1, (16, 15)), dtype=torch.float32)
    with torch.no_grad():
        mode = policy.get_distribution(torch.as_tensor(obs)).mode().numpy()  # the deploy path: 63 inputs
        log_prob = policy.get_distribution(torch.as_tensor(obs)).log_prob(actions).numpy()
    deploy = model.predict(obs, deterministic=True)[0]
    values, kls = {}, {}
    for name, phase in _phases(16).items():
        o66 = torch.as_tensor(_with(obs, phase))
        with torch.no_grad():
            dist = policy.get_distribution(o66)
            np.testing.assert_array_equal(dist.mode().numpy(), mode)
            np.testing.assert_array_equal(dist.log_prob(actions).numpy(), log_prob)
            forward_actions, forward_values, forward_log_prob = policy(o66, deterministic=True)
            np.testing.assert_array_equal(forward_actions.numpy(), mode)
            v, lp, entropy = policy.evaluate_actions(o66, actions)
            np.testing.assert_array_equal(lp.numpy(), log_prob)
            np.testing.assert_array_equal(
                entropy.numpy(), policy.get_distribution(torch.as_tensor(obs)).entropy().numpy()
            )
            values[name] = v.numpy()
            np.testing.assert_array_equal(forward_values.numpy(), v.numpy())
        np.testing.assert_array_equal(model.predict(o66.numpy(), deterministic=True)[0], deploy)
        np.testing.assert_array_equal(
            model.predict(o66.numpy()[0], deterministic=True)[0], model.predict(obs[0], deterministic=True)[0]
        )
        # The K3 panel's (and the judge's) forward pass on the 63 inputs, one row as it runs it.
        with torch.no_grad():
            single = policy.get_distribution(o66[:1]).mode().numpy()[0].astype(np.float64)
        np.testing.assert_array_equal(policy_mean_action(policy, obs[0]), single)
        # MirrorKLPPO's mirrored KL term and mirrored critic on the stored rows.
        update = model._mirror_begin_update()
        rollout = SimpleNamespace(observations=o66, returns=torch.zeros(16))
        model._mirror_value_loss(update, rollout, torch.tensor(0.0))
        model._mirror_loss(update, rollout, torch.tensor(0.0))
        kls[name] = update["kl"][-1]
        assert torch.equal(update["observations"][:, 63:], o66[:, 63:]) or name == "nan"
    assert len(set(kls.values())) == 1, kls
    # The actor's gradient never reaches the phase; the critic's does.
    o66 = torch.as_tensor(_with(obs, _phases(16)["own"])).requires_grad_(True)
    policy.get_distribution(o66).mode().sum().backward()
    assert torch.count_nonzero(o66.grad[:, 63:]) == 0 and torch.count_nonzero(o66.grad[:, :63]) > 0
    o66.grad = None
    policy.predict_values(o66).sum().backward()
    assert torch.count_nonzero(o66.grad[:, 63:]) > 0
    # The critic reads the phase (and refuses to guess one).
    assert not np.array_equal(values["own"], values["zero"]) and np.all(np.isnan(values["nan"]))
    with pytest.raises(CP.CriticPhaseError, match="66 inputs"):
        policy.predict_values(torch.as_tensor(obs))


def test_a_fresh_model_is_the_stock_model_of_its_seed_plus_zero_phase_columns():
    # Each model is built (SB3 reseeds the global generators) right before it learns.
    phase = _make(seed=13)
    a = _parameters(phase)
    phase.learn(total_timesteps=64)
    stock = PPO("MlpPolicy", _venv(13, phase=False), seed=13, policy_kwargs=_POLICY_KWARGS, **_PPO_KWARGS)
    b = _parameters(stock)
    stock.learn(total_timesteps=64)
    assert a.keys() == b.keys()
    for key in a:
        if key == "mlp_extractor.value_net.0.weight":
            assert a[key].shape == (32, 66) and np.array_equal(a[key][:, :63], b[key])
            assert not np.any(a[key][:, 63:])
        else:
            assert np.array_equal(a[key], b[key]), key
    # So the first rollout is the stock one's: the same observations and actions (the values differ at most by
    # the rounding of the zero columns' terms).
    np.testing.assert_array_equal(phase.rollout_buffer.observations[..., :63], stock.rollout_buffer.observations)
    np.testing.assert_array_equal(phase.rollout_buffer.actions, stock.rollout_buffer.actions)
    np.testing.assert_allclose(phase.rollout_buffer.values, stock.rollout_buffer.values, rtol=1e-5, atol=1e-6)


def test_one_throwaway_update_teaches_the_critic_the_phase():
    model = _make()
    critic = model.policy.mlp_extractor.value_net[0].weight
    before = critic.detach().clone()
    assert not torch.any(before[:, 63:])  # it starts out reading no phase
    model.learn(total_timesteps=64)
    assert model._n_updates == 2  # n_epochs 2, one rollout
    assert torch.count_nonzero(critic.detach()[:, 63:] - before[:, 63:]) > 0
    moments = model.policy.optimizer.state[critic]["exp_avg"]
    assert torch.count_nonzero(moments[:, 63:]) > 0
    obs = torch.as_tensor(np.random.default_rng(2).normal(size=(8, 63)), dtype=torch.float32)
    early = torch.cat([obs, torch.as_tensor(SPEC.matrix([0] * 8), dtype=torch.float32)], dim=1)
    settled = torch.cat([obs, torch.as_tensor(SPEC.matrix([SETTLE] * 8), dtype=torch.float32)], dim=1)
    with torch.no_grad():
        assert not torch.equal(model.policy.predict_values(early), model.policy.predict_values(settled))
        assert torch.equal(model.policy.get_distribution(early).mode(), model.policy.get_distribution(settled).mode())


# ---------------------------------------------------------------------------------------------------- 4. the mirror
def test_the_phase_is_its_own_mirror_image_and_the_statistics_stay_the_env_s(mirror_map):
    model = MK.enable_mirror_kl(_make(), SY.SymmetryConfig(), mirror_map)
    model.learn(total_timesteps=64)
    tensors = model._mirror.tensors
    assert tensors["obs_perm"].tolist() == [*mirror_map.obs.perm.tolist(), 63, 64, 65]
    assert tensors["obs_sign"].tolist() == [*mirror_map.obs.sign.tolist(), 1.0, 1.0, 1.0]
    assert np.isfinite(model.logger.name_to_value["train/mirror_kl_loss"])
    rms = model.get_vec_normalize_env().obs_rms
    assert isinstance(rms, MK.SymmetricRunningMeanStd) and rms.mean.shape == (63,) and rms.residual() == (0.0, 0.0)
    assert MK.log_std_pair_gap(model.policy, mirror_map) == 0.0
    # A stock model's maps are untouched.
    stock = PPO("MlpPolicy", _venv(3, phase=False), seed=3, policy_kwargs=_POLICY_KWARGS, **_PPO_KWARGS)
    stock = MK.enable_mirror_kl(stock, SY.SymmetryConfig(), mirror_map)
    stock._mirror_begin_update()
    assert stock._mirror.tensors["obs_perm"].tolist() == mirror_map.obs.perm.tolist()


# ---------------------------------------------------------------------------------------------------- 5. the probe
def test_the_policy_probe_gives_each_bank_state_its_own_phase():
    from environments.shared.policy_probe import PolicyProbeCallback, StateBank

    model = _make()
    model.learn(total_timesteps=64)
    rng = np.random.default_rng(9)
    steps = np.array([199, 599, 3], dtype=np.int64)
    bank = StateBank(
        labels=("statue_settled", "statue", "early"),
        seeds=np.array([1, 1, 2]),
        steps=steps,
        ankle_offset=np.zeros(3),
        raw_obs=rng.normal(0.3, 1.5, (3, 63)),
        com_fore_aft_m=np.zeros(3),
        ankle_actuators=np.zeros((0, 2), dtype=np.int64),
        ankle_signs=np.zeros((0, 2)),
        horizon=1000,
        build_seconds=0.0,
    )
    probe = PolicyProbeCallback(
        model_id=TYRANNOSAURUS_REX,
        env_factory=lambda horizon: None,
        stage_horizon=1000,
        probe_dir=None,
        flush_every_calls=1,
    )
    probe.init_callback(model)
    probe.bank = bank
    row = probe._row()
    venv = model.get_vec_normalize_env()
    normalized = np.clip((bank.raw_obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + venv.epsilon), -10, 10)
    obs = torch.as_tensor(normalized, dtype=torch.float32)
    critic = torch.cat([obs, torch.as_tensor(SPEC.matrix(steps + 1), dtype=torch.float32)], dim=1)
    with torch.no_grad():
        np.testing.assert_array_equal(row["bank_value_now"], model.policy.predict_values(critic).flatten().numpy())
        np.testing.assert_array_equal(row["bank_action_now"], model.policy.get_distribution(obs).mode().numpy())
    # The buffer's 66-wide rows went through the actor.
    assert int(row["rollout_rows"]) > 0 and np.all(np.isfinite(row["rollout_mu_mean"]))


# ---------------------------------------------------------------------------------------------------- 6. checkpoints
def test_the_checkpoint_loads_on_the_deploy_path_with_the_env_s_own_observation(tmp_path):
    from environments.shared.policy_loading import load_sb3_checkpoint

    model = _make()
    model.learn(total_timesteps=64)
    model.save(tmp_path / "stage1_64_steps.zip")
    model.get_vec_normalize_env().save(tmp_path / "stage1_64_steps_vecnorm.pkl")
    data = json.loads(zipfile.ZipFile(tmp_path / "stage1_64_steps.zip").read("data"))
    assert data["policy_kwargs"]["critic_phase_settle_steps"] == SETTLE
    assert data["observation_space"]["_shape"] == [63]
    loaded, normalizer, _ = load_sb3_checkpoint(
        str(tmp_path / "stage1_64_steps.zip"), str(tmp_path / "stage1_64_steps_vecnorm.pkl"), lambda: PhaseToyEnv()
    )
    assert type(loaded.policy) is CPP.CriticPhaseActorCriticPolicy and loaded.policy.critic_phase == SPEC
    assert type(normalizer) is VecNormalize and normalizer.observation_space.shape == (63,)
    raw = np.random.default_rng(1).normal(0.3, 1.5, (6, 63)).astype(np.float32)
    deploy = loaded.predict(normalizer.normalize_obs(raw), deterministic=True)[0]
    for phase in _phases(6).values():
        trained = model.predict(_with(model.get_vec_normalize_env().normalize_obs(raw), phase), deterministic=True)[0]
        np.testing.assert_array_equal(deploy, trained)


def _create(sb3, plant, venv, load_path=None, *, mode="resume_same_stage", spec=SPEC, symmetry=None, seed=3):
    from environments.shared.train_base import _create_or_load_model

    # The trainer passes the stage's seed (D-D11): a load re-seeds the global generators and the env with it.
    kwargs = {**_PPO_KWARGS, "policy_kwargs": dict(_POLICY_KWARGS), "seed": seed}
    return _create_or_load_model(
        sb3,
        "ppo",
        kwargs,
        venv,
        load_path,
        plant_identity=plant,
        task_load_mode=mode,
        symmetry=symmetry,
        critic_phase=spec,
    )


def test_save_load_and_resume_keep_the_phase_and_continue_identically(plant, mirror_map, tmp_path):
    from environments.shared.curriculum import load_vecnorm_stats
    from environments.shared.policy_loading import _ensure_sb3

    sb3 = _ensure_sb3()
    config = SY.SymmetryConfig()
    model = _create(sb3, plant, _venv(3), symmetry=config)
    assert type(model) is MK.MirrorKLPPO and type(model.policy) is CPP.CriticPhaseActorCriticPolicy
    assert getattr(model, CP.CHECKPOINT_ATTRIBUTE) == SPEC.record()
    model.learn(total_timesteps=64)
    stem = tmp_path / "stage1_64_steps"
    model.save(f"{stem}.zip")
    model.get_vec_normalize_env().save(f"{stem}_vecnorm.pkl")
    saved, saved_adam = _parameters(model), _adam(model)
    saved_rms = model.get_vec_normalize_env().obs_rms

    runs = []
    for _ in range(2):
        venv = _venv(21)
        assert load_vecnorm_stats(
            f"{stem}_vecnorm.pkl", venv, current_plant=None, unsafe_skip_plant_validation=True, carry_ret_rms=True
        )
        resumed = _create(sb3, plant, venv, f"{stem}.zip", symmetry=config, seed=21)
        assert type(resumed) is MK.MirrorKLPPO and type(resumed.policy) is CPP.CriticPhaseActorCriticPolicy
        assert resumed.policy.critic_phase == SPEC and getattr(resumed, CP.CHECKPOINT_ATTRIBUTE) == SPEC.record()
        assert type(resumed.rollout_buffer) is CPP.CriticPhaseRolloutBuffer and resumed.rollout_buffer.obs_shape == (
            66,
        )
        assert _same(_parameters(resumed), saved) and resumed.num_timesteps == 64
        restored = _adam(resumed)
        assert restored.keys() == saved_adam.keys()
        assert all(np.array_equal(restored[k][n], saved_adam[k][n]) for k in restored for n in restored[k])
        assert np.array_equal(venv.obs_rms.mean, saved_rms.mean) and np.array_equal(venv.obs_rms.var, saved_rms.var)
        resumed.learn(total_timesteps=64, reset_num_timesteps=False)
        assert resumed.num_timesteps == 128 and resumed._mirror.tensors["obs_perm"].tolist()[63:] == [63, 64, 65]
        runs.append((_parameters(resumed), _adam(resumed), resumed.get_vec_normalize_env().obs_rms.mean.copy()))
    (a, a_adam, a_mean), (b, b_adam, b_mean) = runs
    assert _same(a, b) and np.array_equal(a_mean, b_mean)
    assert all(np.array_equal(a_adam[k][n], b_adam[k][n]) for k in a_adam for n in a_adam[k])
    assert not _same(a, saved)  # it trained on


def test_every_load_whose_critic_reads_another_phase_is_refused_both_ways(plant, tmp_path):
    from environments.shared.policy_loading import _ensure_sb3

    sb3 = _ensure_sb3()
    _create(sb3, plant, _venv(3)).save(tmp_path / "phase.zip")
    _create(sb3, plant, _venv(3, phase=False), spec=None).save(tmp_path / "stock.zip")
    # _create passes no crosses_edge, so initialize_next_stage here is a load that crosses no node's edge (a root
    # re-initialised from itself, or a direct caller): refused both ways.  The warm start across an edge into a stage
    # without the option drops the phase instead (test_critic_phase_drop.py).
    for mode in ("resume_same_stage", "initialize_next_stage"):
        with pytest.raises(CP.CriticPhaseError, match="declares no"):
            _create(sb3, plant, _venv(3, phase=False), str(tmp_path / "phase.zip"), mode=mode, spec=None)
        with pytest.raises(CP.CriticPhaseError, match="reads no phase"):
            _create(sb3, plant, _venv(3), str(tmp_path / "stock.zip"), mode=mode)
        other = CP.CriticPhaseSpec(SETTLE, 0.99)
        with pytest.raises(CP.CriticPhaseError, match="reads"):
            _create(sb3, plant, _venv(3, gamma=0.99), str(tmp_path / "phase.zip"), mode=mode, spec=other)
    # The env must append the phase the model is built for.
    with pytest.raises(CP.CriticPhaseError, match="training env"):
        _create(sb3, plant, _venv(3, phase=False))
    # A stock checkpoint on a stage without the option loads as before.
    assert (
        type(_create(sb3, plant, _venv(3, phase=False), str(tmp_path / "stock.zip"), spec=None).policy)
        is ActorCriticPolicy
    )


# ---------------------------------------------------------------------------------------------------- 7. switch off
def test_the_switch_off_builds_and_trains_the_stock_model_bitwise(plant):
    """critic_phase None: train_base builds SB3's MlpPolicy and RolloutBuffer, and the model trains bitwise as a
    PPO built directly (the off path changes nothing)."""
    from environments.shared.policy_loading import _ensure_sb3

    sb3 = _ensure_sb3()
    off = _create(sb3, plant, _venv(13, phase=False), spec=None, seed=13)
    assert type(off.policy) is ActorCriticPolicy and type(off.rollout_buffer) is RolloutBuffer
    assert CP.CHECKPOINT_ATTRIBUTE not in vars(off) and off.rollout_buffer.obs_shape == (63,)
    off.learn(total_timesteps=128)
    direct = PPO("MlpPolicy", _venv(13, phase=False), seed=13, policy_kwargs=_POLICY_KWARGS, **_PPO_KWARGS)
    direct.learn(total_timesteps=128)
    assert _same(_parameters(off), _parameters(direct))
    a, b = _adam(off), _adam(direct)
    assert all(np.array_equal(a[k][n], b[k][n]) for k in a for n in a[k])
    np.testing.assert_array_equal(off.rollout_buffer.observations, direct.rollout_buffer.observations)
    np.testing.assert_array_equal(off.rollout_buffer.values, direct.rollout_buffer.values)
    np.testing.assert_array_equal(off.rollout_buffer.rewards, direct.rollout_buffer.rewards)


def test_the_trainer_reads_the_switch_from_the_ppo_table():
    from environments.shared.config import load_all_stages

    configs = load_all_stages(TYRANNOSAURUS_REX)
    ppo = dict(configs[1]["ppo_kwargs"])
    assert CP.pop_critic_phase(ppo) is False
    configs[1]["ppo_kwargs"]["critic_phase"] = "yes"
    with pytest.raises(CP.CriticPhaseError):
        CP.read_critic_phase(configs[1]["ppo_kwargs"])
    assert pickle.loads(pickle.dumps(SPEC)) == SPEC  # the spec travels with a pickled policy


@pytest.mark.parametrize(
    "switch, symmetric", [(False, False), (True, False), (True, True)], ids=["off", "on", "on-symmetric"]
)
def test_the_trainer_wires_the_switch_end_to_end(switch, symmetric, tmp_path, monkeypatch):
    """train_base.train (the notebook's train_stage call) on the real T. rex stance stage, one tiny update: off, the
    stock VecNormalize, MlpPolicy and buffer and no record; on, the phase env, policy, buffer and record.  The
    committed stance trains mirror-KL PPO with the mirrored critic (the r10 LTSB recipe, the stance-landing decision
    of 2026-11-06), the configuration the phase-critic stance runs trained under: the symmetric case keeps it, and the
    other two drop its four [ppo] symmetry keys to train stock PPO."""
    import copy

    from environments.shared import train_base
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    from .critic_phase_off_scenario import LTSB_PPO_KEYS

    configs = copy.deepcopy(load_all_stages(TYRANNOSAURUS_REX))
    ppo = configs[1]["ppo_kwargs"]
    assert ppo["symmetry"] == "mirror_kl" and ppo["symmetry_mirror_critic"] is True  # main's stance is mirror-KL
    ppo.update(n_steps=64, batch_size=64, n_epochs=1)
    if switch:
        ppo["critic_phase"] = True
    if not symmetric:
        for key in LTSB_PPO_KEYS:
            ppo.pop(key)
    seen = {}
    original = train_base._create_or_load_model

    def spy(sb3, algorithm, alg_kwargs, train_env, *args, **kwargs):
        seen["env"] = type(train_env)
        seen["alg_kwargs"] = dict(alg_kwargs)
        return original(sb3, algorithm, alg_kwargs, train_env, *args, **kwargs)

    monkeypatch.setattr(train_base, "_create_or_load_model", spy)
    model = train_base.train(
        get_species_config(TYRANNOSAURUS_REX),
        configs,
        1,
        64,
        n_envs=1,
        seed=42,
        eval_freq=10**9,
        save_freq=10**9,
        verbose=0,
        output_dir=str(tmp_path / "stage"),
        use_tensorboard=False,
        report_metrics=False,
        save_on_interrupt=False,
    )
    assert type(model) is (MK.MirrorKLPPO if symmetric else PPO)
    assert model.num_timesteps == 64 and model._n_updates == 1
    assert "critic_phase" not in seen["alg_kwargs"]  # popped: never a PPO constructor argument
    recorded = json.loads((tmp_path / "stage" / "stage_config.json").read_text())
    if switch:
        assert seen["env"] is CPP.CriticPhaseVecNormalize
        assert type(model.policy) is CPP.CriticPhaseActorCriticPolicy and model.rollout_buffer.obs_shape == (66,)
        assert model.policy.critic_phase == CP.CriticPhaseSpec(200, 0.98)
        assert recorded["hyperparameters"]["critic_phase"] is True
    else:
        assert seen["env"] is VecNormalize
        assert type(model.policy) is ActorCriticPolicy and type(model.rollout_buffer) is RolloutBuffer
        assert model.rollout_buffer.obs_shape == (63,) and CP.CHECKPOINT_ATTRIBUTE not in vars(model)
        assert "critic_phase" not in recorded["hyperparameters"]
    with zipfile.ZipFile(tmp_path / "stage" / "models" / "stage1_final.zip") as archive:
        data = json.loads(archive.read("data"))
    assert (CP.CHECKPOINT_ATTRIBUTE in data) is switch and data["observation_space"]["_shape"] == [63]
    with open(tmp_path / "stage" / "models" / "stage1_final_vecnorm.pkl", "rb") as handle:
        assert type(pickle.load(handle)) is VecNormalize
