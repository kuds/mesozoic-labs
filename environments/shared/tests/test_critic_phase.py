"""The critic's episode phase (environments/shared/critic_phase.py): the features, the spec and the [ppo] switch.
Numpy only; the SB3 pieces are tested in test_critic_phase_ppo.py."""

from __future__ import annotations

import numpy as np
import pytest

from environments.shared import critic_phase as CP
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId


def test_the_features_at_the_gate_and_the_horizon():
    # The screen's definition at S 200, gamma 0.98, written out.
    spec = CP.CriticPhaseSpec(settle_steps=200, gamma=0.98)
    assert spec.features(0) == (0.0, 0.0, 0.98**200)
    assert spec.features(199) == (199 / 200, 0.0, 0.98)
    assert spec.features(200) == (1.0, 1.0, 1.0)
    assert spec.features(1000) == (1.0, 1.0, 1.0)
    assert spec.features(1) == (1 / 200, 0.0, 0.98**199)
    np.testing.assert_array_equal(
        spec.matrix([0, 199, 200, 1000]),
        np.array([[0.0, 0.0, 0.98**200], [0.995, 0.0, 0.98], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    )
    assert spec.matrix(np.zeros(0, dtype=np.int64)).shape == (0, 3)
    # Monotone in k, each feature in [0, 1].
    phase = spec.matrix(np.arange(1001))
    assert np.all(np.diff(phase, axis=0) >= 0.0) and phase.min() >= 0.0 and phase.max() == 1.0


def test_settled_says_the_next_reward_is_gated_in():
    # BaseDinoEnv.step increments _step_count before the observation and the reward: the observation with
    # _step_count k is followed by the reward of step k + 1, which the stance gates pay iff k + 1 > S.
    spec = CP.CriticPhaseSpec(settle_steps=200, gamma=0.98)
    for k in range(0, 400):
        assert spec.features(k)[1] == (1.0 if k + 1 > 200 else 0.0)
        # The first gated reward (step S + 1) is max(0, S - k) steps after the next one.
        assert spec.features(k)[2] == 0.98 ** max(0, (200 + 1) - (k + 1))


@pytest.mark.parametrize("k", [-1, 1.0, True, "3", None])
def test_a_step_counter_is_a_non_negative_integer(k):
    with pytest.raises(CP.CriticPhaseError):
        CP.CriticPhaseSpec(settle_steps=200, gamma=0.98).features(k)


@pytest.mark.parametrize(
    "settle, gamma", [(0, 0.98), (True, 0.98), (200.0, 0.98), (200, 0.0), (200, 1.5), (200, float("nan")), (200, True)]
)
def test_the_spec_refuses_what_is_not_a_phase(settle, gamma):
    with pytest.raises(CP.CriticPhaseError):
        CP.CriticPhaseSpec(settle_steps=settle, gamma=gamma)


def test_the_record_round_trips_and_names_the_features():
    spec = CP.CriticPhaseSpec(settle_steps=np.int64(200), gamma=0.98)
    assert type(spec.settle_steps) is int and spec == CP.CriticPhaseSpec(200, 0.98)
    record = spec.record()
    assert record["schema"] == CP.CHECKPOINT_SCHEMA and record["features"] == ["ramp", "settled", "discount"]
    assert CP.CriticPhaseSpec.from_record(record) == spec
    assert spec.policy_kwargs() == {"critic_phase_settle_steps": 200, "critic_phase_gamma": 0.98}
    assert set(spec.policy_kwargs()) == set(CP.POLICY_KWARGS)
    with pytest.raises(CP.CriticPhaseError):
        CP.CriticPhaseSpec.from_record({**record, "features": ["ramp"]})


def test_the_switch_is_a_boolean_popped_from_the_ppo_table():
    assert CP.read_critic_phase({}) is False and CP.read_critic_phase(None) is False
    table = {"gamma": 0.98, "critic_phase": True}
    assert CP.read_critic_phase(table) is True and "critic_phase" in table
    assert CP.pop_critic_phase(table) is True and table == {"gamma": 0.98}
    off = {"critic_phase": False}
    assert CP.pop_critic_phase(off) is False and off == {}
    for value in (1, "true", 1.0, None):
        with pytest.raises(CP.CriticPhaseError, match="true or false"):
            CP.pop_critic_phase({"critic_phase": value})


def test_no_stage_config_turns_the_critic_phase_on():
    # Off by default: the switch is a launch choice (the notebook's CRITIC_PHASE, or --override
    # ppo.critic_phase=true), never a committed stage key.
    from environments.shared.species_names import model_ids
    from environments.shared.stage_manifest import load_stage_manifest

    for model_id in model_ids():
        for entry in load_stage_manifest(model_id).stages:
            ppo = dict(load_stage_config(model_id, entry.reference).get("ppo_kwargs", {}))
            assert CP.CRITIC_PHASE_KEY not in ppo, (model_id, entry.id)


def test_the_tyrannosaurus_rex_stance_gates_are_the_spec_s_settle_step():
    config = load_stage_config(ModelId("tyrannosaurus_rex", "anatomical"), 1)
    assert config["env_kwargs"][CP.SETTLE_ATTRIBUTE] == 200
    assert config["ppo_kwargs"]["gamma"] == 0.98
