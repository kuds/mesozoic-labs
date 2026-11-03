"""The screen-only cadence keys: ``[curriculum] eval_freq``, ``eval_episodes`` and ``checkpoint_freq``.

``train_base._stage_cadence`` and ``_eval_episodes_for_stage``: a declared key replaces the caller's value, an
absent one keeps it, only a positive integer is accepted, and the keys pass the gate schema while moving only the
recorded stage config, never the task, gate or hyperparameter digests.
"""

from __future__ import annotations

import copy

import pytest

from environments.shared import train_base as TB
from environments.shared.config import hyperparameters_sha256, load_all_stages
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view, validate_gate_config
from environments.shared.harnesses.digest_snapshot import _stage_config_view_sha256
from environments.shared.model_id import ModelId
from environments.shared.task_fingerprint import stage_task_fingerprint

CADENCE = {"eval_freq": 131_072, "eval_episodes": 10, "checkpoint_freq": 524_288}


def _stage(**curriculum):
    return {"curriculum_kwargs": dict(curriculum)}


def test_a_declared_frequency_replaces_the_callers_and_an_absent_one_keeps_it():
    assert TB._stage_cadence(_stage(eval_freq=131_072), 50_000, 100_000) == (131_072, 100_000)
    assert TB._stage_cadence(_stage(checkpoint_freq=524_288), 50_000, 100_000) == (50_000, 524_288)
    assert TB._stage_cadence(_stage(), 50_000, 100_000) == (50_000, 100_000)
    assert TB._stage_cadence({}, 50_000, 100_000) == (50_000, 100_000)


def test_a_declared_eval_episodes_is_used_exactly_even_below_min_eval_episodes():
    assert TB._eval_episodes_for_stage(_stage(eval_episodes=10, min_eval_episodes=40)) == 10
    assert TB._eval_episodes_for_stage(_stage(min_eval_episodes=40)) == 40


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
@pytest.mark.parametrize("key", sorted(CADENCE))
def test_only_a_positive_integer_is_accepted(key, bad):
    with pytest.raises(ValueError, match=key):
        if key == "eval_episodes":
            TB._eval_episodes_for_stage(_stage(eval_episodes=bad))
        else:
            TB._stage_cadence(_stage(**{key: bad}), 50_000, 100_000)


def test_the_keys_pass_the_gate_schema_and_move_only_the_recorded_stage_config(tmp_path):
    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    without = copy.deepcopy(load_all_stages(model_id)[1])
    for key in CADENCE:
        without["curriculum_kwargs"].pop(key, None)
    with_keys = copy.deepcopy(without)
    with_keys["curriculum_kwargs"].update(CADENCE)
    validate_gate_config("stance", with_keys["curriculum_kwargs"])

    def digests(config, name):
        return {
            "task": stage_task_fingerprint(model_id, 1, stage_config=config)["task_sha256"],
            "gate": gate_config_sha256(gate_config_view(config["curriculum_kwargs"])),
            "hyperparameters": hyperparameters_sha256(config, "PPO"),
            "view": _stage_config_view_sha256(model_id, 1, config, "PPO", tmp_path / name),
        }

    before, after = digests(without, "without"), digests(with_keys, "with")
    assert {name for name in before if before[name] != after[name]} == {"view"}
