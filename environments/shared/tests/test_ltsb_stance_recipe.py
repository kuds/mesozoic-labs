"""The committed T. rex stance is the recipe the r10 phase-critic screens trained (the maintainer's
stance-landing decision of 2026-11-06).

``stance7_r10LTSBpc_s42``, ``_s60`` and ``_s80`` passed the six-heading certificate on a branch commit; the recipe
landed on ``main`` so that two new seeds can certify it.  These tests compare this checkout's derivation of the
committed stance with what one of those runs RECORDED (``fixtures/ltsb_stance/stance7_r10LTSBpc_s42.json``, a copy of
its ``01_stance/stage_config.json`` and its verdict's gate digest): the task fingerprint and its env block, the
recipe digest with the critic-phase launch switch on (the notebook's ``CRITIC_PHASE = True`` writes ``[ppo]
critic_phase = true``) and with it off, the gate digest, the plant identity, and the reward weights,
hyperparameters and curriculum ``save_stage_config`` records.  The only recorded differences are the screens'
evaluation and checkpoint cadence keys, which main does not take (it trains at its production cadence) and which
enter no digest.  The screen runs themselves never count (``run_exclusions.py``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from environments.shared.config import hyperparameters_sha256, load_stage_config, save_stage_config
from environments.shared.critic_phase import CRITIC_PHASE_KEY
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
from environments.shared.model_id import ModelId
from environments.shared.task_fingerprint import stage_task_fingerprint

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ltsb_stance" / "stance7_r10LTSBpc_s42.json"
#: The recipe the three switch-off screens recorded (stance7_r10LTSBoff_*): the same stance without the critic phase.
SWITCH_OFF_RECIPE = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"
#: The screens' evaluation and checkpoint cadence, which main does not take.
SCREEN_CADENCE_KEYS = {"eval_freq", "eval_episodes", "checkpoint_freq", "max_checkpoints"}


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    record: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return record


@pytest.fixture(scope="module")
def stance() -> dict[str, Any]:
    return load_stage_config(TYRANNOSAURUS_REX, "stance")


def _switched_on(stance: dict[str, Any]) -> dict[str, Any]:
    """The stance as the notebook's ``CRITIC_PHASE = True`` hands it to the trainer."""
    config = copy.deepcopy(stance)
    config["ppo_kwargs"][CRITIC_PHASE_KEY] = True
    return config


def test_the_committed_stance_is_the_recorded_task(recorded):
    fingerprint = stage_task_fingerprint(TYRANNOSAURUS_REX, "stance")
    want = recorded["stage_config"]["task_fingerprint"]
    assert fingerprint["task_sha256"] == want["task_sha256"]
    assert fingerprint["task_sha256"].startswith("sha256:1e5964b6")
    assert fingerprint["env"] == want["env"]


def test_with_the_critic_phase_switch_the_recipe_is_the_recorded_recipe(recorded, stance):
    want = recorded["stage_config"]["run"]["hyperparameters_sha256"]
    assert hyperparameters_sha256(_switched_on(stance), "PPO") == want
    assert want.startswith("sha256:c64211e8")
    # A fresh launch with the switch off trains the switch-off screens' recipe: the certification steps check it.
    assert hyperparameters_sha256(stance, "PPO") == SWITCH_OFF_RECIPE


def test_the_gate_is_the_recorded_gate(recorded, stance):
    assert gate_config_sha256(gate_config_view(stance["curriculum_kwargs"])) == recorded["gate_sha256"]
    assert recorded["gate_sha256"].startswith("sha256:d26c34e6")


def test_the_plant_is_the_recorded_plant(recorded):
    from environments.shared.plant_contract import current_plant_identity

    assert current_plant_identity(TYRANNOSAURUS_REX).to_dict() == recorded["stage_config"]["plant_identity"]


def test_the_recorded_blocks_are_the_screens_but_for_the_cadence(tmp_path, recorded, stance):
    from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

    path = save_stage_config(
        tmp_path,
        "stance",
        _switched_on(stance),
        "PPO",
        env_class=TyrannosaurusRexAnatomicalEnv,
        model_id=TYRANNOSAURUS_REX,
    )
    written = json.loads(Path(path).read_text(encoding="utf-8"))
    want = recorded["stage_config"]
    assert written["reward_weights"] == want["reward_weights"]
    assert written["hyperparameters"] == want["hyperparameters"]
    differing = {
        key
        for key in set(written["curriculum"]) | set(want["curriculum"])
        if written["curriculum"].get(key) != want["curriculum"].get(key)
    }
    assert differing <= SCREEN_CADENCE_KEYS
    assert {key: want["curriculum"][key] for key in differing} == {
        "eval_freq": 131072,
        "eval_episodes": 10,
        "checkpoint_freq": 262144,
        "max_checkpoints": 0,
    }
