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

import ast
import copy
import json
import re
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
REPO_ROOT = Path(__file__).resolve().parents[3]
NEXT_STEPS = REPO_ROOT / "docs" / "NEXT_STEPS.md"
NOTEBOOK = REPO_ROOT / "notebooks" / "sb3_training.ipynb"
#: The seeds the r10 stance screens trained (they never count): a certification seed is another.
SCREEN_SEEDS = {42, 60, 80}
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


# ---------------------------------------------------------------------------------- the certification steps' settings
def _certification_settings() -> dict[str, Any]:
    """The first certification session's notebook settings, as NEXT_STEPS.md section 3 gives them (a python block)."""
    text = NEXT_STEPS.read_text(encoding="utf-8")
    start = text.index("LTSB certification")
    body = text[text.index("```python\n", start) + len("```python\n") :]
    tree = ast.parse(body[: body.index("```")])
    settings: dict[str, Any] = {}
    for node in tree.body:
        assert isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name), ast.dump(node)
        settings[node.targets[0].id] = ast.literal_eval(node.value)
    return settings


def _notebook_configuration() -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """The SB3 notebook's configuration cell: each top-level setting's default, and each ``@param`` menu."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    (source,) = [src for src in sources if "\nCRITIC_PHASE = " in src and "\nBEHAVIOR = " in src]
    lines = [line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))]
    defaults: dict[str, Any] = {}
    for node in ast.parse("\n".join(lines)).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                defaults[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                defaults[node.targets[0].id] = None
    menus: dict[str, list[Any]] = {}
    for line in lines:
        match = re.match(r"^([A-Z_]+) = .*# @param (\[[^\]]*\])", line)
        if match:
            menus[match.group(1)] = json.loads(match.group(2))
    return defaults, menus


def test_the_certification_steps_name_only_settings_the_notebook_has():
    settings = _certification_settings()
    defaults, menus = _notebook_configuration()
    assert set(settings) <= set(defaults), sorted(set(settings) - set(defaults))
    for name, value in settings.items():
        if defaults[name] is not None:
            assert type(value) is type(defaults[name]), name
    for name in ("SPECIES", "EMBODIMENT", "BEHAVIOR"):
        assert settings[name] in menus[name], name
    # "stand" was the stance's label before D-D38; the notebook refuses it.
    assert settings["BEHAVIOR"] == "stance" and "stand" not in menus["BEHAVIOR"]


def test_the_certification_steps_train_the_landed_recipe_on_fresh_seeds():
    settings = _certification_settings()
    text = NEXT_STEPS.read_text(encoding="utf-8")
    assert (settings["SPECIES"], settings["EMBODIMENT"], settings["ALGORITHM"]) == (
        "tyrannosaurus_rex",
        "anatomical",
        "ppo",
    )
    # The phase-critic recipe c64211e8 is the stance with the switch on; off, a fresh launch trains 054ab483.
    assert settings["CRITIC_PHASE"] is True
    # An explicit run id per seed: "" re-enters the run the runtime resolved last.
    assert isinstance(settings["RUN_ID"], str) and settings["RUN_ID"].strip()
    assert settings["TRUNK_FROM"] == "" and settings["RETRAIN_FROM"] == "" and settings["QUICK_TEST"] is False
    second = re.search(r"`SEED = (\d+)` and\s+`RUN_ID = \"([^\"]+)\"`", text)
    assert second is not None
    seeds = {settings["SEED"], int(second.group(1))}
    assert len(seeds) == 2 and not seeds & SCREEN_SEEDS
    assert second.group(2) != settings["RUN_ID"]
