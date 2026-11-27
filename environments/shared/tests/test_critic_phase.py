"""The critic's episode phase (environments/shared/critic_phase.py): the features, the spec and the [ppo] switch.
Numpy only; the SB3 pieces are tested in test_critic_phase_ppo.py."""

from __future__ import annotations

import numpy as np
import pytest

from environments.shared import critic_phase as CP
from environments.shared.config import hyperparameters_sha256, load_stage_config
from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
#: The T. rex recipe digests of main at 4c9d5e9b, before the option existed (configs/digest_snapshot.generated.txt
#: there, lines 384-406).  Written out here, never read from a golden a later --write could move.
#: The T. rex recipe digests on main.  The stance and recovery PPO digests moved with the r10 LTSB stance recipe
#: (the stance-landing decision of 2026-11-06: mirror-KL PPO in stance's [ppo], which recovery inherits); the
#: critic-phase option still adds no committed key.  The SAC digests read only [sac] and [curriculum] keys that did not
#: move.  The bite hunt's two digests left with its stage (decision D-D59); the other six are unmoved by that removal.
MAIN_RECIPE_DIGESTS = {
    ("stance", "PPO"): "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c",
    ("stance", "SAC"): "sha256:c6eb8d73dfdde463350306ba49070b031cbe327407cfdaf6aee270eb52d8fd26",
    ("recovery", "PPO"): "sha256:de9c9af0411e29a3a00e01f7fe096b1a2d0d12b2e64bbcbbbafe57c4c4e7eb65",
    ("recovery", "SAC"): "sha256:09b1e8034671fa666c74e2e4faac98fa751288a4ead8dc3473d1c702b7116b31",
    # The walk's moved with decision D-D62, and only by its [curriculum] losing the forward-velocity ramp
    # (ramp_timesteps 2000000, ramp_start_value 0.2), a shaping key of the recipe: with the two restored, the walk
    # gives main's sha256:fde660ac...33f6980c (PPO) and sha256:49ef875a...ab1d1a799 (SAC).
    ("locomotion", "PPO"): "sha256:1703f55481fc4a88d2b89ceb5aa693a881d54ff990b0d94c9b4240993bc33e38",
    ("locomotion", "SAC"): "sha256:02e78fc9e0e5686bf689ea202fd6050722987cda43e3177cc1df60f94d620bcc",
}


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


def test_the_record_and_keys_are_the_ones_the_phase_critic_runs_recorded():
    # The archives the phase-critic stance runs wrote (the screen branch's code) carry these literals: a change here
    # would stop them loading, or dropping at a warm start.  The record is the chosen walking parent's, verbatim.
    assert CP.CRITIC_PHASE_KEY == "critic_phase" and CP.CHECKPOINT_ATTRIBUTE == "mesozoic_critic_phase"
    assert CP.CHECKPOINT_SCHEMA == "mesozoic.critic-phase/v1" and CP.SETTLE_ATTRIBUTE == "stance_width_settle_steps"
    assert CP.POLICY_KWARGS == ("critic_phase_settle_steps", "critic_phase_gamma")
    assert CP.PHASE_FEATURES == ("ramp", "settled", "discount") and CP.PHASE_WIDTH == 3
    parent = {
        "schema": "mesozoic.critic-phase/v1",
        "features": ["ramp", "settled", "discount"],
        "settle_steps": 200,
        "gamma": 0.98,
        "width": 3,
        "step_counter": "the env's _step_count when the observation was made (0 at reset)",
        "input": "the critic's only: appended after the normalized observation, never normalized",
    }
    assert CP.CriticPhaseSpec(200, 0.98).record() == parent
    assert CP.CriticPhaseSpec.from_record(parent) == CP.CriticPhaseSpec(200, 0.98)
    assert CP.CriticPhaseSpec(200, 0.98).policy_kwargs() == {
        "critic_phase_settle_steps": 200,
        "critic_phase_gamma": 0.98,
    }


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
    # ppo.critic_phase=true), never a committed stage key.  No stage declares a warm_start_ curriculum key either.
    from environments.shared.species_names import model_ids
    from environments.shared.stage_manifest import load_stage_manifest

    for model_id in model_ids():
        for entry in load_stage_manifest(model_id).stages:
            config = load_stage_config(model_id, entry.reference)
            ppo = dict(config.get("ppo_kwargs", {}))
            assert CP.CRITIC_PHASE_KEY not in ppo, (model_id, entry.id)
            assert not [key for key in config.get("curriculum_kwargs", {}) if key.startswith("warm_start_")], (
                model_id,
                entry.id,
            )


@pytest.mark.parametrize("stage, algorithm", sorted(MAIN_RECIPE_DIGESTS))
def test_the_tyrannosaurus_rex_recipe_digests_are_main_s(stage, algorithm):
    # The option adds no committed key, so no recorded run.hyperparameters_sha256 moves.
    assert (
        hyperparameters_sha256(load_stage_config(TYRANNOSAURUS_REX, stage), algorithm)
        == (MAIN_RECIPE_DIGESTS[(stage, algorithm)])
    )


def test_the_recipe_digest_records_the_switch_when_a_table_writes_it():
    import copy

    stance = load_stage_config(TYRANNOSAURUS_REX, "stance")
    digests = {"absent": hyperparameters_sha256(stance, "PPO")}
    for name, value in (("on", True), ("off", False)):
        written = copy.deepcopy(stance)
        written["ppo_kwargs"][CP.CRITIC_PHASE_KEY] = value
        digests[name] = hyperparameters_sha256(written, "PPO")
    assert digests["absent"] == MAIN_RECIPE_DIGESTS[("stance", "PPO")]
    # Each moves the digest, an explicit false too: "absent" is the default, and the record says what the table said.
    assert len(set(digests.values())) == 3
    # With the switch written on (the notebook's CRITIC_PHASE = True writes exactly this), the committed stance is
    # the recipe the r10 phase-critic screens recorded (stance7_r10LTSBpc_*, run.hyperparameters_sha256).
    assert digests["on"] == "sha256:c64211e8f84782e97a38141d3d2b6a7cc8f0e455b1a0524a32afb6b4a137a954"


def test_the_off_path_scenario_restores_the_pre_landing_recipe_on_each_plant():
    """The same pre-LTSB stance settings identify distinct tasks on the archived and current plants."""
    import copy

    from environments.shared.config import load_all_stages
    from environments.shared.task_fingerprint import stage_task_fingerprint

    from .critic_phase_off_scenario import archived_plant_identity, pre_landing_stance

    committed = load_all_stages(TYRANNOSAURUS_REX)
    configs = copy.deepcopy(committed)
    pre_landing_stance(configs)
    stance = configs[1]
    assert hyperparameters_sha256(stance, "PPO") == (
        "sha256:9d1598a6b2a96a1f87412bb0398abb49d1f23430aa72462f03cbcd68ab5d7a2e"
    )
    # The critic-off golden captured physics r10 / policy interface r15 at 4c9d5e9b, where this task read
    # 533fff20 (e9d349f8 on the current plant); both moved when the T. rex hunt arguments left its constructor
    # defaults (decision D-D59's amendment), which changes no reward the stance computes (cab8c221 / df516f13),
    # and again with plant_versions note 24 (D-D59: the prey kwargs gone from the constructor, the reset's prey
    # draws and the heading-free observation in the code the task hashes; the current plant physics r12).
    archived = stage_task_fingerprint(
        TYRANNOSAURUS_REX, 1, stage_config=stance, plant_identity=archived_plant_identity()
    )
    assert archived["task_sha256"] == "sha256:a2c78b967ec8b4f8b26d2b2903f0ec70876565930589942bd851ff513175e86d"
    assert stage_task_fingerprint(TYRANNOSAURUS_REX, 1, stage_config=stance)["task_sha256"] == (
        "sha256:4a9821b5fa120d66c8879c797a5ec637164069e82a1be8b92d09345f46fe553b"
    )
    assert stance["curriculum_kwargs"]["collapse_peak_floor_reference"] == 3766.6
    assert configs[2] == committed[2]


def test_the_recipe_digest_records_a_warm_start_log_std_when_a_stage_declares_one():
    import copy

    locomotion = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    declared = copy.deepcopy(locomotion)
    declared["curriculum_kwargs"]["warm_start_log_std"] = -1.5
    assert hyperparameters_sha256(locomotion, "PPO") == MAIN_RECIPE_DIGESTS[("locomotion", "PPO")]
    assert hyperparameters_sha256(declared, "PPO") != MAIN_RECIPE_DIGESTS[("locomotion", "PPO")]


def test_the_tyrannosaurus_rex_stance_gates_are_the_spec_s_settle_step():
    config = load_stage_config(TYRANNOSAURUS_REX, 1)
    assert config["env_kwargs"][CP.SETTLE_ATTRIBUTE] == 200
    assert config["ppo_kwargs"]["gamma"] == 0.98
