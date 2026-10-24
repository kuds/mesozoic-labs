"""The policy probe (``policy_probe``): the policy at every PPO update on a fixed state bank, report-only.

The bank is built on the T. rex stance (ten states, two seeds), and the
probe is attached to a small PPO on a real T. rex env: it draws no random
number and changes no parameter, optimizer state, normalizer or training
flag; its values and actions are the policy's own forward passes; its
rollout clip shares are the rollout buffer's own samples and the policy's
means at the buffer's observations, read with a full buffer without moving
anything; its learning rate is the one the coming ``train()`` applies; and
its file is written on the checkpoint cadence and seeded on a resume (a
file written before the rollout arrays included).
"""

from __future__ import annotations

import copy
import logging
import math
import random
import types
from typing import Any

import numpy as np
import pytest

from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco")


def _factory(model_id=TYRANNOSAURUS_REX, stage=None, record=None):
    """``factory(horizon)``: a bare env of the model's first (stance) stage; *record* collects its actions."""
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    config = get_species_config(model_id)
    stages = load_all_stages(model_id)
    kwargs = dict(stages[stage if stage is not None else next(iter(stages))]["env_kwargs"])

    def factory(horizon):
        env = config.env_class(**{**kwargs, "max_episode_steps": horizon})
        if record is not None:
            step = env.step

            def recording(action):
                record.append(np.array(action))
                return step(action)

            env.step = recording
        return env

    return factory


def _global_states():
    import torch

    return np.random.get_state(), random.getstate(), torch.get_rng_state()


def _same_states(first, second):
    import torch

    (np_a, py_a, torch_a), (np_b, py_b, torch_b) = first, second
    return (
        np_a[0] == np_b[0]
        and np.array_equal(np_a[1], np_b[1])
        and tuple(np_a[2:]) == tuple(np_b[2:])
        and py_a == py_b
        and torch.equal(torch_a, torch_b)
    )


@pytest.fixture(scope="module")
def bank():
    """The T. rex stance bank at a 240-step stage horizon, built once without the guard; and whether it drew."""
    import torch

    from environments.shared.policy_probe import build_state_bank

    np.random.seed(5)
    random.seed(5)
    torch.manual_seed(5)
    before = _global_states()
    built = build_state_bank(_factory(), TYRANNOSAURUS_REX, stage_horizon=240)
    return built, _same_states(before, _global_states())


def _ppo(*, normalize=True, n_steps=64, **kwargs):
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    factory = _factory()
    vec = DummyVecEnv([lambda: Monitor(factory(240))])
    if normalize:
        vec = VecNormalize(vec)
    model = PPO("MlpPolicy", vec, n_steps=n_steps, batch_size=64, n_epochs=1, seed=7, device="cpu", **kwargs)
    model.set_logger(Logger(None, []))  # learn() would set one; the hooks are called directly here
    return model


def _probe(model, *, bank=None, monkeypatch=None, probe_dir=None, flush=10**9, factory=None):
    """A probe on *model*, initialised; with *bank*, the build is replaced by that bank (saving its 5 s)."""
    from environments.shared import policy_probe

    if bank is not None:
        monkeypatch.setattr(policy_probe, "build_state_bank", lambda *args, **kwargs: bank)
    callback = policy_probe.PolicyProbeCallback(
        model_id=TYRANNOSAURUS_REX,
        env_factory=factory or _factory(),
        stage_horizon=240,
        probe_dir=probe_dir,
        flush_every_calls=flush,
    )
    callback.init_callback(model)
    model._total_timesteps = 10_000
    return callback


def _update(model, callback, steps=64):
    model.num_timesteps += steps
    callback.on_step()
    callback.on_rollout_end()


class TestTheBank:
    def test_it_is_whole_at_a_short_stage_horizon_and_draws_no_random_number(self, bank):
        from environments.shared.policy_probe import BANK_SPEC, bank_horizon

        built, undrawn = bank
        assert undrawn, "building the bank moved a global generator"
        assert len(built.labels) == 10 and built.labels == tuple(label for label, *_rest in BANK_SPEC)
        assert built.horizon == 601 and built.raw_obs.shape == (10, 63)
        assert (bank_horizon(240), bank_horizon(400), bank_horizon(1000)) == (601, 601, 1000)
        assert built.steps.tolist() == [199, 599, 599, 599, 400] * 2
        assert built.ankle_actuators.shape == (1, 2) and built.ankle_signs.tolist() == [[1.0, 1.0]]
        assert np.isfinite(built.com_fore_aft_m).all()
        # A forward lean of the shank (negative ankle action on the T. rex) carries the CoM forward.
        com = dict(zip(zip(built.labels, built.seeds.tolist()), built.com_fore_aft_m))
        for seed in (3042, 3043):
            assert com[("ankle_minus_0.03", seed)] > com[("statue", seed)] > com[("ankle_plus_0.03", seed)]

    def test_it_is_deterministic_and_the_horizon_changes_nothing_before_it(self, bank):
        from environments.shared.policy_probe import build_state_bank

        built, _undrawn = bank
        again = build_state_bank(_factory(), TYRANNOSAURUS_REX, stage_horizon=1000)
        assert again.horizon == 1000
        assert np.array_equal(again.raw_obs, built.raw_obs)
        assert np.array_equal(again.com_fore_aft_m, built.com_fore_aft_m)

    @pytest.mark.parametrize(
        ("model_id", "pairs"),
        [(ModelId("compsognathus_longipes", "robot"), 1), (ModelId("dibothrosuchus_elaphros", "anatomical"), 2)],
        ids=str,
    )
    def test_the_offset_follows_each_ankle_pairs_signs(self, model_id, pairs):
        from environments.shared.policy_probe import _mirror_even, build_state_bank

        actions: list[np.ndarray] = []
        built = build_state_bank(
            _factory(model_id, record=actions), model_id, (("lean", 7, 0.03, 300),), stage_horizon=1000
        )
        assert built.ankle_actuators.shape == (pairs, 2) and len(built.labels) == 1
        held = actions[300].astype(np.float64)
        others = np.ones(held.shape, dtype=bool)
        for (left, right), (left_sign, right_sign) in zip(built.ankle_actuators, built.ankle_signs):
            assert held[left] == np.float32(left_sign * 0.03) and held[right] == np.float32(right_sign * 0.03)
            others[[left, right]] = False
        assert not held[others].any() and not actions[199].any()
        assert actions[250][built.ankle_actuators[0, 0]] == np.float32(built.ankle_signs[0, 0] * 0.015)
        assert _mirror_even(held[None], built.ankle_actuators, built.ankle_signs)[0] == pytest.approx(0.03)
        if model_id.embodiment == "robot":
            assert built.ankle_signs.tolist() == [[-1.0, -1.0]]


#: Three rows on seed 3043, where the T. rex recovery stage's pushes topple the statue at step 364 and the
#: -0.05 lean at step 347: the settled statue (step 199) survives them, the other two do not.
_PUSHED_SPEC = (
    ("statue_settled", 3043, 0.0, 199),
    ("statue", 3043, 0.0, 599),
    ("ankle_minus_0.05_at400", 3043, -0.05, 400),
)


class TestAStageWithPushes:
    """A recovery stage pushes the animal on a schedule; the bank env runs with the pushes off."""

    def test_the_bank_env_keywords_switch_off_the_pushes_and_nothing_else(self):
        from environments.shared.config import load_stage_config
        from environments.shared.policy_probe import PUSH_MULTIPLE_KWARG, bank_env_kwargs

        recovery = load_stage_config(TYRANNOSAURUS_REX, "recovery")["env_kwargs"]
        assert recovery[PUSH_MULTIPLE_KWARG] > 0.0
        kwargs = bank_env_kwargs(recovery, 601)
        assert kwargs == {**recovery, "max_episode_steps": 601, PUSH_MULTIPLE_KWARG: 0.0}
        assert recovery[PUSH_MULTIPLE_KWARG] > 0.0  # the stage's own mapping is untouched
        # A stage that declares no pushes is handed no push keyword at all.
        stance = load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"]
        assert PUSH_MULTIPLE_KWARG not in stance
        assert bank_env_kwargs(stance, 601) == {**stance, "max_episode_steps": 601}

    def test_with_the_pushes_off_every_row_survives_and_with_them_on_the_lost_rows_warn_once(self, caplog):
        from environments.shared.config import load_stage_config
        from environments.shared.policy_probe import bank_env_kwargs, build_state_bank
        from environments.shared.species_registry import get_species_config

        env_class = get_species_config(TYRANNOSAURUS_REX).env_class
        recovery = load_stage_config(TYRANNOSAURUS_REX, "recovery")["env_kwargs"]
        horizon = int(recovery["max_episode_steps"])
        with caplog.at_level(logging.WARNING, logger="environments.shared.policy_probe"):
            whole = build_state_bank(
                lambda h: env_class(**bank_env_kwargs(recovery, h)),
                TYRANNOSAURUS_REX,
                _PUSHED_SPEC,
                stage_horizon=horizon,
            )
        assert whole.labels == tuple(label for label, *_rest in _PUSHED_SPEC)
        assert not [r for r in caplog.records if "bank rows dropped" in r.getMessage()]
        with caplog.at_level(logging.WARNING, logger="environments.shared.policy_probe"):
            pushed = build_state_bank(
                lambda h: env_class(**{**recovery, "max_episode_steps": h}),
                TYRANNOSAURUS_REX,
                _PUSHED_SPEC,
                stage_horizon=horizon,
            )
        assert pushed.labels == ("statue_settled",)
        dropped = [r.getMessage() for r in caplog.records if "bank rows dropped" in r.getMessage()]
        assert len(dropped) == 1 and dropped[0].startswith("Policy probe: 2 of 3 bank rows dropped")
        assert "'statue' (seed 3043" in dropped[0] and "'ankle_minus_0.05_at400' (seed 3043" in dropped[0]


class TestNeutrality:
    def test_the_probe_draws_no_random_number_and_writes_nothing(self):
        """The real bank build and three rows leave the generators, the policy and the normalizer bit-equal."""
        import torch

        model = _ppo()
        policy, vec_normalize = model.policy, model.get_vec_normalize_env()
        torch.manual_seed(3)
        np.random.seed(3)
        random.seed(3)
        states = _global_states()
        parameters = copy.deepcopy(policy.state_dict())
        optimizer = copy.deepcopy(policy.optimizer.state_dict())
        training = policy.training
        statistics = copy.deepcopy((vec_normalize.obs_rms, vec_normalize.ret_rms))
        distribution = dict(vars(policy.action_dist))  # what get_distribution() would overwrite
        callback = _probe(model)
        callback.on_training_start({}, {})
        assert callback.bank is not None and len(callback.bank.labels) == 10
        for _ in range(3):
            _update(model, callback)
        assert len(callback.rows) == 3
        assert _same_states(states, _global_states())
        assert policy.training == training
        assert set(vars(policy.action_dist)) == set(distribution)
        assert all(vars(policy.action_dist)[key] is value for key, value in distribution.items())
        for key, tensor in policy.state_dict().items():
            assert torch.equal(tensor, parameters[key]), key
        assert repr(policy.optimizer.state_dict()) == repr(optimizer)
        for now, before in zip((vec_normalize.obs_rms, vec_normalize.ret_rms), statistics):
            assert np.array_equal(now.mean, before.mean) and np.array_equal(now.var, before.var)
            assert now.count == before.count

    def test_the_guard_restores_a_generator_the_build_moved_and_warns_once(self, bank, monkeypatch, caplog):
        import torch

        from environments.shared import policy_probe

        def drawing(*args, **kwargs):
            np.random.rand()
            random.random()
            torch.rand(1)
            return bank[0]

        monkeypatch.setattr(policy_probe, "build_state_bank", drawing)
        model = _ppo()
        states = _global_states()
        callback = _probe(model)
        with caplog.at_level(logging.WARNING, logger="environments.shared.policy_probe"):
            callback.on_training_start({}, {})
        assert _same_states(states, _global_states())
        assert len([r for r in caplog.records if "moved a global random generator" in r.getMessage()]) == 1
        assert callback.bank is bank[0]


class TestTheRows:
    def test_values_and_actions_are_the_policys_own_forward_passes(self, bank, monkeypatch):
        import torch

        model = _ppo()
        callback = _probe(model, bank=bank[0], monkeypatch=monkeypatch)
        callback.on_training_start({}, {})
        vec_normalize = model.get_vec_normalize_env()
        frozen = copy.deepcopy(vec_normalize.obs_rms)
        vec_normalize.obs_rms.mean = vec_normalize.obs_rms.mean + 0.25  # the normalizer moves after the freeze
        vec_normalize.obs_rms.var = vec_normalize.obs_rms.var * 2.0
        _update(model, callback)
        (row,) = callback.rows
        policy = model.policy

        def direct(observations):
            tensor = torch.as_tensor(observations, dtype=torch.float32)
            with torch.no_grad():
                values = policy.predict_values(tensor).flatten().numpy()
                actions = policy.get_distribution(tensor).distribution.mean.numpy()
            return values, actions

        values, actions = direct(vec_normalize.normalize_obs(bank[0].raw_obs))
        assert np.array_equal(row["bank_value_now"], values) and np.array_equal(row["bank_action_now"], actions)
        live = vec_normalize.obs_rms
        vec_normalize.obs_rms = frozen
        try:
            values, actions = direct(vec_normalize.normalize_obs(bank[0].raw_obs))
        finally:
            vec_normalize.obs_rms = live
        assert np.array_equal(row["bank_value_frozen"], values) and np.array_equal(row["bank_action_frozen"], actions)
        assert not np.array_equal(row["bank_value_now"], row["bank_value_frozen"])
        assert np.array_equal(row["std_per_actuator"], torch.exp(policy.log_std).detach().numpy())
        assert np.array_equal(row["obs_rms_mean"], live.mean.astype(np.float32))
        assert row["frozen_reference_timesteps"] == 0 and row["update_timesteps"] == 64

    def test_the_learning_rate_is_the_one_train_applies(self, bank, monkeypatch):
        """Each row's rate equals the optimizer's after the ``train()`` that followed it."""
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList

        applied: list[float] = []

        class Reader(BaseCallback):
            def _on_rollout_start(self) -> None:
                applied.append(float(self.model.policy.optimizer.param_groups[0]["lr"]))

            def _on_training_end(self) -> None:
                applied.append(float(self.model.policy.optimizer.param_groups[0]["lr"]))

            def _on_step(self) -> bool:
                return True

        model = _ppo(learning_rate=lambda remaining: 3e-4 * remaining)
        probe = _probe(model, bank=bank[0], monkeypatch=monkeypatch)
        model.learn(3 * 64, callback=CallbackList([probe, Reader()]))
        rates = [float(row["learning_rate"]) for row in probe.rows]
        assert len(rates) == 3 and len(set(rates)) == 3
        assert rates == pytest.approx(applied[1:], rel=1e-6)  # applied[0] precedes the first train()

    def test_without_vecnormalize_the_raw_observations_go_in_and_the_normalizer_columns_are_nan(
        self, bank, monkeypatch
    ):
        import torch

        model = _ppo(normalize=False)
        callback = _probe(model, bank=bank[0], monkeypatch=monkeypatch)
        callback.on_training_start({}, {})
        _update(model, callback)
        (row,) = callback.rows
        with torch.no_grad():
            values = model.policy.predict_values(torch.as_tensor(bank[0].raw_obs, dtype=torch.float32))
        assert np.array_equal(row["bank_value_now"], values.flatten().numpy())
        for key in ("obs_rms_mean", "obs_rms_var", "obs_rms_count", "ret_rms_var", "bank_value_frozen"):
            assert np.isnan(row[key]).all(), key
        arrays = callback.arrays()
        assert np.isnan(arrays["frozen_obs_mean"]).all() and np.isnan(arrays["bank_action_frozen"]).all()
        assert math.isnan(callback.latest()["value_normalization_shift_statue"])

    def test_sac_gets_no_bank_and_no_rows(self, caplog):
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        def never(horizon):
            raise AssertionError("the bank env must not be built without a rollout buffer")

        factory = _factory()
        model = SAC("MlpPolicy", DummyVecEnv([lambda: Monitor(factory(240))]), buffer_size=100, seed=7, device="cpu")
        callback = _probe(model, factory=never)
        with caplog.at_level(logging.INFO, logger="environments.shared.policy_probe"):
            callback.on_training_start({}, {})
            _update(model, callback)
        assert callback.bank is None and callback.rows == []
        assert len([r for r in caplog.records if "no rollout buffer" in r.getMessage()]) == 1

    def test_a_non_diagonal_distribution_reads_nan_std_and_warns_once(self, bank, monkeypatch, caplog):
        model = _ppo(use_sde=True)
        callback = _probe(model, bank=bank[0], monkeypatch=monkeypatch)
        callback.on_training_start({}, {})
        with caplog.at_level(logging.WARNING, logger="environments.shared.policy_probe"):
            _update(model, callback)
            _update(model, callback)
        assert len(callback.rows) == 2 and np.isnan(callback.rows[0]["std_per_actuator"]).all()
        assert np.isfinite(callback.rows[0]["bank_value_now"]).all()
        assert len([r for r in caplog.records if "not a diagonal" in r.getMessage()]) == 1

    def test_tensorboard_gets_the_scalars_without_the_console(self, bank, monkeypatch):
        model = _ppo()
        records = []
        model.set_logger(types.SimpleNamespace(record=lambda *args, **kwargs: records.append((args, kwargs))))
        callback = _probe(model, bank=bank[0], monkeypatch=monkeypatch)
        callback.on_training_start({}, {})
        _update(model, callback)
        tags = {args[0] for args, _kwargs in records}
        assert {
            "policy_probe/std_min",
            "policy_probe/std_max_over_min",
            "policy_probe/ent_coef",
            "policy_probe/obs_var_max",
            "policy_probe/value_statue",
            "policy_probe/value_ankle_minus_0.05_minus_statue",
            "policy_probe/value_normalization_shift_statue",
            "policy_probe/statue_ankle_action_mean",
            "policy_probe/ankle_gain_per_m",
        } <= tags
        assert all(tag.startswith("policy_probe/") for tag in tags)
        assert all(kwargs == {"exclude": ("stdout", "log")} for _args, kwargs in records)


class TestTheRolloutShares:
    def test_the_shares_of_a_hand_made_rollout(self):
        from environments.shared.policy_probe import rollout_action_shares

        actions = np.array([[1.5, -0.2], [0.995, -1.3], [-2.0, 0.5], [0.2, -0.99]], dtype=np.float32)
        means = np.array([[1.2, 0.0], [0.8, -1.1], [0.4, 0.0], [0.0, -0.5]], dtype=np.float32)
        shares = rollout_action_shares(actions, means, np.full(2, -1.0, np.float32), np.full(2, 1.0, np.float32))
        assert int(shares["rollout_rows"]) == 4
        assert shares["rollout_action_clip_high_share"].tolist() == [0.25, 0.0]
        assert shares["rollout_action_clip_low_share"].tolist() == [0.25, 0.25]
        # |clip(a)| >= 0.99: 1.5 -> 1, 0.995 and -2.0 -> -1 on the first; -1.3 -> -1 and -0.99 on the second.
        assert shares["rollout_action_saturated_share"].tolist() == [0.75, 0.5]
        assert shares["rollout_mu_beyond_high_share"].tolist() == [0.25, 0.0]
        assert shares["rollout_mu_beyond_low_share"].tolist() == [0.0, 0.25]
        assert shares["rollout_mu_mean"] == pytest.approx([0.6, -0.4])
        assert shares["rollout_mu_q50"] == pytest.approx([0.6, -0.25])
        assert shares["rollout_mu_q95"] == pytest.approx(np.quantile(means.astype(np.float64), 0.95, axis=0))
        assert all(value.dtype == np.float32 for key, value in shares.items() if key != "rollout_rows")
        empty = rollout_action_shares(np.zeros((0, 2)), np.zeros((0, 2)), -np.ones(2), np.ones(2))
        assert int(empty["rollout_rows"]) == 0 and np.isnan(empty["rollout_mu_q95"]).all()

    def test_a_real_rollout_reads_the_buffer_and_the_policys_means_and_moves_nothing(self, bank, monkeypatch):
        """Between the rollout's end and train(): the buffer, the generators, the parameters and the distribution
        are the same after the probe as before it, and its shares are the buffer's own."""
        import torch
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList

        model = _ppo(n_steps=128, policy_kwargs={"log_std_init": 0.5})
        with torch.no_grad():  # two actuators' means past a bound everywhere: +1.5 and -1.2 before the first update
            model.policy.action_net.bias[0] = 1.5
            model.policy.action_net.bias[1] = -1.2
        probe = _probe(model, bank=bank[0], monkeypatch=monkeypatch)
        seen: list[dict] = []

        def snapshot(model) -> dict:
            buffer = model.rollout_buffer
            return {
                "states": _global_states(),
                "parameters": copy.deepcopy(model.policy.state_dict()),
                "buffer": [np.copy(getattr(buffer, key)) for key in ("observations", "actions", "log_probs", "values")],
                "distribution": dict(vars(model.policy.action_dist)),
                "training": model.policy.training,
            }

        class Before(BaseCallback):
            def _on_rollout_end(self) -> None:
                seen.append(snapshot(self.model))

            def _on_step(self) -> bool:
                return True

        class After(Before):
            def _on_rollout_end(self) -> None:
                before, after = seen[-1], snapshot(self.model)
                assert _same_states(before["states"], after["states"])
                for key, tensor in after["parameters"].items():
                    assert torch.equal(tensor, before["parameters"][key]), key
                assert all(np.array_equal(a, b) for a, b in zip(after["buffer"], before["buffer"]))
                assert after["training"] == before["training"]
                assert all(after["distribution"][key] is value for key, value in before["distribution"].items())
                model: Any = self.model
                policy, buffer = model.policy, model.rollout_buffer
                with torch.no_grad():
                    observations = torch.as_tensor(buffer.observations.reshape(-1, buffer.observations.shape[-1]))
                    means = policy.get_distribution(observations).distribution.mean.numpy()
                vars(policy.action_dist).update(before["distribution"])
                seen[-1]["means"], seen[-1]["actions"] = means, buffer.actions.reshape(means.shape).copy()

        model.learn(2 * 128, callback=CallbackList([Before(), probe, After()]))
        assert len(probe.rows) == 2
        for row, captured in zip(probe.rows, seen):
            actions, means = captured["actions"], captured["means"]
            assert int(row["rollout_rows"]) == 128
            assert np.array_equal(
                row["rollout_action_clip_high_share"], (actions > 1.0).mean(axis=0).astype(np.float32)
            )
            assert np.array_equal(
                row["rollout_action_clip_low_share"], (actions < -1.0).mean(axis=0).astype(np.float32)
            )
            assert np.array_equal(
                row["rollout_action_saturated_share"],
                (np.abs(np.clip(actions, -1.0, 1.0)) >= 0.99).mean(axis=0).astype(np.float32),
            )
            assert np.array_equal(row["rollout_mu_beyond_high_share"], (means > 1.0).mean(axis=0).astype(np.float32))
            assert np.array_equal(row["rollout_mu_beyond_low_share"], (means < -1.0).mean(axis=0).astype(np.float32))
            assert row["rollout_mu_q95"] == pytest.approx(np.quantile(means.astype(np.float64), 0.95, axis=0), abs=1e-6)
        first = probe.rows[0]
        # The first rollout's policy: the biased means beyond their bounds at every step, std 1.65 clipping often.
        assert first["rollout_mu_beyond_high_share"][0] == 1.0 and first["rollout_mu_beyond_low_share"][1] == 1.0
        assert 0.1 < float(first["rollout_action_clip_high_share"][2]) < 0.5
        latest = probe.latest()
        assert latest["rollout_mu_beyond_share_max"] == pytest.approx(
            max(probe.rows[-1]["rollout_mu_beyond_high_share"] + probe.rows[-1]["rollout_mu_beyond_low_share"])
        )
        assert math.isfinite(latest["rollout_action_clip_share_max"])
        assert math.isfinite(latest["rollout_action_saturated_share_max"])

    def test_a_resume_from_a_file_written_before_the_rollout_arrays_keeps_its_rows_as_nan(
        self, bank, monkeypatch, tmp_path
    ):
        from environments.shared.policy_probe import _ROLLOUT_KEYS, POLICY_PROBE_FILENAME

        TestTheFileAndTheResume()._first_session(bank[0], monkeypatch, tmp_path)
        path = tmp_path / POLICY_PROBE_FILENAME
        with np.load(path, allow_pickle=False) as data:
            older = {key: data[key] for key in data.files if key not in _ROLLOUT_KEYS}
        np.savez(path, **older)
        model = _ppo()
        model.num_timesteps = 128
        resumed = _probe(model, bank=bank[0], monkeypatch=monkeypatch, probe_dir=tmp_path)
        assert resumed.seed_prior_record(tmp_path, max_timesteps=128) == 2
        resumed.on_training_start({}, {})
        _update(model, resumed)
        assert math.isnan(resumed.latest()["rollout_mu_beyond_share_max"])  # this row's buffer is empty here
        resumed.on_training_end()
        with np.load(path, allow_pickle=False) as data:
            assert data["update_timesteps"].tolist() == [64, 128, 192]
            assert data["rollout_rows"].tolist() == [0, 0, 0]
            assert data["rollout_mu_q95"].shape == (3, 15) and np.isnan(data["rollout_mu_q95"]).all()

    def test_a_failing_buffer_pass_costs_the_rollout_arrays_only(self, bank, monkeypatch, caplog):
        from environments.shared.policy_probe import PolicyProbeCallback

        def failing(self, policy, nu):
            raise RuntimeError("out of memory")

        model = _ppo()
        probe = _probe(model, bank=bank[0], monkeypatch=monkeypatch)
        probe.on_training_start({}, {})
        stored = dict(vars(model.policy.action_dist))
        monkeypatch.setattr(PolicyProbeCallback, "_rollout_means", failing)
        with caplog.at_level(logging.WARNING, logger="environments.shared.policy_probe"):
            _update(model, probe)
            _update(model, probe)
        assert len(probe.rows) == 2  # the row stands: its std, normalizer and bank values and actions
        row = probe.rows[-1]
        assert np.isfinite(row["std_per_actuator"]).all() and np.isfinite(row["bank_value_now"]).all()
        assert int(row["rollout_rows"]) == 0 and np.isnan(row["rollout_action_clip_high_share"]).all()
        assert all(vars(model.policy.action_dist)[key] is value for key, value in stored.items())
        warned = [r.getMessage() for r in caplog.records if "could not read the rollout buffer" in r.getMessage()]
        assert len(warned) == 1


def test_the_scalars_read_the_rows_they_name():
    from environments.shared.policy_probe import StateBank, probe_scalars

    labels = ("statue", "ankle_plus_0.03", "ankle_minus_0.03", "statue", "ankle_plus_0.03", "ankle_minus_0.03")
    bank = StateBank(
        labels=labels,
        seeds=np.array([1, 1, 1, 2, 2, 2]),
        steps=np.full(6, 599),
        ankle_offset=np.array([0.0, 0.03, -0.03] * 2),
        raw_obs=np.zeros((6, 2)),
        com_fore_aft_m=np.array([0.0, -0.04, 0.01, 0.0, -0.05, 0.02]),
        ankle_actuators=np.array([[0, 1]]),
        ankle_signs=np.array([[1.0, -1.0]]),
        horizon=601,
        build_seconds=0.0,
    )
    actions = np.zeros((6, 2))
    actions[:, 0], actions[:, 1] = [0.1, 0.2, 0.0, 0.3, 0.4, 0.1], [-0.1, -0.2, 0.0, -0.3, -0.4, -0.1]
    row = {
        "std_per_actuator": np.array([0.1, 0.5]),
        "obs_rms_var": np.array([2.0, 8.0]),
        "ent_coef": 0.01,
        "bank_value_now": np.array([10.0, 11.0, 7.0, 20.0, 23.0, 15.0]),
        "bank_value_frozen": np.array([9.0, 0.0, 0.0, 18.0, 0.0, 0.0]),
        "bank_action_now": actions,
    }
    out = probe_scalars(bank, row)
    assert (out["std_min"], out["std_max"], out["std_max_over_min"]) == pytest.approx((0.1, 0.5, 5.0))
    assert (out["obs_var_min"], out["obs_var_max"], out["ent_coef"]) == pytest.approx((2.0, 8.0, 0.01))
    assert out["value_statue"] == pytest.approx(15.0)
    assert out["value_ankle_plus_minus_statue"] == pytest.approx(((11 - 10) + (23 - 20)) / 2)
    assert out["value_ankle_minus_minus_statue"] == pytest.approx(((7 - 10) + (15 - 20)) / 2)
    assert math.isnan(out["value_ankle_minus_0.05_minus_statue"])  # no such row in this bank
    assert out["value_normalization_shift_statue"] == pytest.approx(((10 - 9) + (20 - 18)) / 2)
    # The mirror-even ankle action: (1 * a_0 + (-1) * a_1) / 2 = a_0 on these rows.
    assert out["statue_ankle_action_mean"] == pytest.approx((0.1 + 0.3) / 2)
    assert out["ankle_gain_per_m"] == pytest.approx(((0.2 - 0.0) / (-0.05) + (0.4 - 0.1) / (-0.07)) / 2)
    assert math.isnan(out["rollout_action_clip_share_max"])  # a row without the rollout arrays
    row.update(
        rollout_action_clip_high_share=np.array([0.1, 0.0]),
        rollout_action_clip_low_share=np.array([0.05, 0.12]),
        rollout_action_saturated_share=np.array([0.2, np.nan]),
        rollout_mu_beyond_high_share=np.array([np.nan, np.nan]),
        rollout_mu_beyond_low_share=np.array([np.nan, np.nan]),
    )
    out = probe_scalars(bank, row)
    assert out["rollout_action_clip_share_max"] == pytest.approx(0.15)  # both bounds of one actuator
    assert out["rollout_action_saturated_share_max"] == pytest.approx(0.2)
    assert math.isnan(out["rollout_mu_beyond_share_max"])


class TestTheFileAndTheResume:
    def test_it_is_written_on_the_checkpoint_cadence_and_at_training_end(self, bank, monkeypatch, tmp_path):
        from environments.shared import file_io

        model = _ppo()
        callback = _probe(model, bank=bank[0], monkeypatch=monkeypatch, probe_dir=tmp_path, flush=5)
        written = []
        original = file_io.atomic_savez
        monkeypatch.setattr(
            file_io,
            "atomic_savez",
            lambda path, **arrays: (written.append(callback.n_calls), original(path, **arrays))[1],
        )
        callback.on_training_start({}, {})
        for call in range(1, 13):
            model.num_timesteps += 1
            callback.on_step()
            if call in (3, 7, 11):
                callback.on_rollout_end()
        callback.on_training_end()
        assert written == [5, 10, 12]  # the calls a checkpoint is saved on with a new row, then the end
        with np.load(tmp_path / "policy_probe.npz", allow_pickle=False) as data:
            assert data["update_timesteps"].tolist() == [3, 7, 11]
            assert data["bank_labels"].tolist() == list(bank[0].labels)
            assert data["bank_action_now"].shape == (3, 10, 15) and data["bank_value_now"].dtype == np.float32
            assert int(data["bank_horizon"]) == 601
            # The rollout arrays beside them; these rows were taken with an empty buffer, so read none.
            assert data["rollout_rows"].tolist() == [0, 0, 0] and data["rollout_mu_mean"].shape == (3, 15)
            assert data["rollout_action_clip_high_share"].dtype == np.float32

    def _first_session(self, bank, monkeypatch, directory):
        model = _ppo()
        callback = _probe(model, bank=bank, monkeypatch=monkeypatch, probe_dir=directory)
        callback.on_training_start({}, {})
        for _ in range(3):
            _update(model, callback)
        callback.on_training_end()
        return callback

    def test_a_resume_holds_the_prior_rows_and_keeps_them_behind_an_equal_bank(self, bank, monkeypatch, tmp_path):
        self._first_session(bank[0], monkeypatch, tmp_path)
        model = _ppo()
        model.num_timesteps = 128
        resumed = _probe(model, bank=bank[0], monkeypatch=monkeypatch, probe_dir=tmp_path)
        assert resumed.seed_prior_record(tmp_path, max_timesteps=128) == 2
        assert resumed.bank is None  # held, not compared: no bank exists before learn()
        resumed.on_training_start({}, {})
        _update(model, resumed)
        resumed.on_training_end()
        with np.load(tmp_path / "policy_probe.npz", allow_pickle=False) as data:
            assert data["update_timesteps"].tolist() == [64, 128, 192]
            assert data["frozen_reference_timesteps"].tolist() == [0, 0, 128]

    def test_a_different_bank_drops_the_prior_rows_with_one_warning(self, bank, monkeypatch, tmp_path, caplog):
        self._first_session(bank[0], monkeypatch, tmp_path)
        other = bank[0]._replace(raw_obs=bank[0].raw_obs + 1.0)
        model = _ppo()
        model.num_timesteps = 192
        resumed = _probe(model, bank=other, monkeypatch=monkeypatch, probe_dir=tmp_path)
        assert resumed.seed_prior_record(tmp_path, max_timesteps=None) == 3
        with caplog.at_level(logging.WARNING, logger="environments.shared.policy_probe"):
            resumed.on_training_start({}, {})
        assert len([r for r in caplog.records if "differs from this one's" in r.getMessage()]) == 1
        _update(model, resumed)
        resumed.on_training_end()
        with np.load(tmp_path / "policy_probe.npz", allow_pickle=False) as data:
            assert data["update_timesteps"].tolist() == [256]


def _core_callbacks(tmp_path, monkeypatch, *, factory):
    import tempfile

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from environments.shared.config import load_stage_config
    from environments.shared.train_base import _build_core_callbacks

    from .tiny_env_helpers import tiny_env_class

    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    eval_env = DummyVecEnv([tiny_env_class()])
    try:
        callbacks, _, _ = _build_core_callbacks(
            {"CheckpointCallback": CheckpointCallback},
            eval_env,
            tmp_path / "models",
            tmp_path / "logs",
            1,
            4,
            100,
            400,
            0,
            load_stage_config(TYRANNOSAURUS_REX, "stance"),
            model_id=TYRANNOSAURUS_REX,
            policy_probe_env_factory=factory,
        )
    finally:
        eval_env.close()
    return callbacks


class TestTheTrainerWiring:
    def test_the_probe_follows_the_training_ledger_on_the_checkpoint_cadence(self, tmp_path, monkeypatch):
        from stable_baselines3.common.callbacks import CheckpointCallback

        from environments.shared.policy_probe import PolicyProbeCallback
        from environments.shared.rollout_ledger import TrainingEpisodeLedgerCallback

        factory = _factory()
        callbacks = _core_callbacks(tmp_path, monkeypatch, factory=factory)
        kinds = [type(callback) for callback in callbacks]
        assert kinds.count(PolicyProbeCallback) == 1
        probe = callbacks[kinds.index(PolicyProbeCallback)]
        assert kinds.index(TrainingEpisodeLedgerCallback) < kinds.index(PolicyProbeCallback)
        assert probe.flush_every_calls == callbacks[kinds.index(CheckpointCallback)].save_freq == 100
        assert probe.env_factory is factory and probe.stage_horizon == 1000
        assert probe.path == tmp_path / "logs" / "policy_probe.npz"
        assert not [cb for cb in _core_callbacks(tmp_path, monkeypatch, factory=None) if isinstance(cb, type(probe))]

    def test_the_stage_body_hands_it_a_bare_env_of_the_stage(self):
        import inspect

        from environments.shared import train_base

        source = inspect.getsource(train_base._train_stage_body)
        call = source[source.index("_build_core_callbacks(") :]
        assert "policy_probe_env_factory=lambda horizon: species_cfg.env_class(" in call
        assert 'bank_env_kwargs(config["env_kwargs"], horizon)' in call
