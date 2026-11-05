"""Walking pilot 4's command route (pilot-only, never merged): a canonical stage's [env] carries a live command.

The maintainer's decisions (23) and (25) of 2026-11-05: pilot 4 keeps the T. rex walking stage (03_locomotion,
stage 2) and feeds a "point straight" command through the reserved command inputs, set in the stage TOML.  The hybrid
check of 2026-11-05 measured that config alone cannot do it; these tests pin the pilot-only edits that let it:

- (1) ``BaseDinoEnv.__init__`` builds the ``DirectionCommandConfig`` from a TOML ``command_config`` table (a mapping).
- (2) ``task_fingerprint.derive_stage_task_fingerprint`` derives a live stage's command manifest from that table when
  no manifest is passed, so ``train()``'s ``stage_task_fingerprint(stage_config=...)`` and the digest harness hash it.

Nothing here trains.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.compsognathus_longipes.anatomical.env import CompsognathusLongipesAnatomicalEnv
from environments.shared.config import load_stage_config
from environments.shared.direction_commands import DirectionCommandConfig, DirectionCommandController, wrap_angle
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.shared.task_fingerprint import (
    TaskFingerprintError,
    compute_task_fingerprint,
    derive_stage_task_fingerprint,
    stage_task_fingerprint,
)
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
REPOSITORY = Path(__file__).resolve().parents[3]
#: The pilot's straight command as a TOML table would give it (decisions (21)-(28)): no turn, top-of-scale speed,
#: never a stop, every other field at follow_direction's value.
STRAIGHT_TABLE: dict[str, Any] = {
    "turn_increment_max": 0.0,
    "cruise_speed": 1.5,
    "speed_scale": 1.5,
    "lateral_speed_scale": 1.0,
    "yaw_rate_scale": 0.6,
    "yaw_gain": 1.5,
    "yaw_rate_max": 0.3,
    "turn_slowdown": True,
    "minimum_turn_speed_fraction": 0.25,
    "stop_probability": 0.0,
    "straight_probability": 1.0,
}


def _rollout(env: Any, seed: int, steps: int = 40) -> list[tuple[Any, ...]]:
    obs, info = env.reset(seed=seed)
    rows: list[tuple[Any, ...]] = [(obs.copy(), 0.0, False, False, info)]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-0.05, 0.05, env.action_space.shape))
        rows.append((obs.copy(), reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return rows


# ======================================================================================== (1) the table builds the config
@pytest.mark.parametrize("env_class", [TyrannosaurusRexAnatomicalEnv, CompsognathusLongipesAnatomicalEnv])
@pytest.mark.parametrize("mode", ["heading", "heading_and_speed"])
def test_a_command_table_builds_the_dataclass_and_rolls_as_the_dataclass_does(env_class, mode):
    """A mapping is the dataclass it names: the same config, controller manifest and trajectory, bit for bit."""
    config = DirectionCommandConfig(**STRAIGHT_TABLE)
    from_table = env_class(command_mode=mode, command_config=dict(STRAIGHT_TABLE))
    from_class = env_class(command_mode=mode, command_config=config)
    try:
        assert isinstance(from_table.command_config, DirectionCommandConfig)
        assert from_table.command_config == config
        assert from_table.direction_controller is not None
        assert from_table.direction_controller.config == config
        assert (
            from_table.command_manifest()
            == from_class.command_manifest()
            == DirectionCommandController(config).manifest()
        )
        for seed in (3042, 3043):
            rows_table, rows_class = _rollout(from_table, seed), _rollout(from_class, seed)
            assert len(rows_table) == len(rows_class)
            for (obs_a, reward_a, te_a, tr_a, info_a), (obs_b, reward_b, te_b, tr_b, info_b) in zip(
                rows_table, rows_class
            ):
                np.testing.assert_array_equal(obs_a, obs_b)
                assert (reward_a, te_a, tr_a) == (reward_b, te_b, tr_b)
                assert list(info_a) == list(info_b)
    finally:
        from_table.close()
        from_class.close()


def test_a_partial_table_takes_the_dataclass_defaults_for_the_rest():
    env = TyrannosaurusRexAnatomicalEnv(command_mode="heading", command_config={"turn_increment_max": 0.0})
    try:
        assert env.command_config == DirectionCommandConfig(turn_increment_max=0.0)
    finally:
        env.close()


def test_a_toml_table_through_load_stage_config_builds_the_env(tmp_path):
    """The real route: a stage TOML's [env] table, read by config.load_stage_config (which passes [env] through, lists
    to tuples at the top level only), reaches the constructor as a dict and builds."""
    path = tmp_path / "locomotion.toml"
    lines = ["[env]", 'command_mode = "heading"', "max_episode_steps = 20", "[env.command_config]"]
    for key, value in STRAIGHT_TABLE.items():
        lines.append(f"{key} = {'true' if value is True else repr(value)}")
    path.write_text("\n".join(lines) + "\n")
    env_kwargs = load_stage_config(TYRANNOSAURUS_REX, "locomotion", config_path=str(path))["env_kwargs"]
    assert env_kwargs["command_config"] == STRAIGHT_TABLE and isinstance(env_kwargs["command_config"], dict)
    assert tomllib.loads(path.read_text())["env"]["command_config"] == STRAIGHT_TABLE
    env = TyrannosaurusRexAnatomicalEnv(**env_kwargs)
    try:
        obs, _ = env.reset(seed=3042)
        assert env.command_config == DirectionCommandConfig(**STRAIGHT_TABLE)
        # Straight at reset: the desired heading is the spawn heading, the command is top-of-scale speed, no turn.
        assert env._command_state is not None and env._command_state.desired_heading == wrap_angle(env._heading())
        np.testing.assert_array_equal(obs[-3:], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    finally:
        env.close()


def test_a_bad_table_fails_at_construction():
    """The dataclass's own checks: an unknown field is a TypeError, an invalid value a ValueError, and a table under
    command_mode "none" is refused as the dataclass is."""
    with pytest.raises(TypeError, match="cruise"):
        TyrannosaurusRexAnatomicalEnv(command_mode="heading", command_config={"cruise": 1.5})
    with pytest.raises(ValueError, match="cruise_speed must lie in"):
        TyrannosaurusRexAnatomicalEnv(command_mode="heading", command_config={"cruise_speed": 2.0})
    with pytest.raises(ValueError, match="turn_slowdown must be a bool"):
        TyrannosaurusRexAnatomicalEnv(command_mode="heading", command_config={"turn_slowdown": 1})
    with pytest.raises(ValueError, match="command_config needs a live command_mode"):
        TyrannosaurusRexAnatomicalEnv(command_config=dict(STRAIGHT_TABLE))
    with pytest.raises(ValueError, match="'heading' holds cruise_speed"):
        TyrannosaurusRexAnatomicalEnv(command_mode="heading", command_config={"stop_probability": 0.1})


def test_a_table_speed_range_list_is_the_dataclass_tuple():
    """TOML has arrays, not tuples, and no null: a speed_range list is the dataclass's tuple of floats."""
    env = TyrannosaurusRexAnatomicalEnv(command_mode="heading_and_speed", command_config={"speed_range": [0.5, 1]})
    try:
        assert env.command_config.speed_range == (0.5, 1.0)
        assert env.command_config == DirectionCommandConfig(speed_range=(0.5, 1.0))
    finally:
        env.close()


def test_the_straight_table_names_real_fields_and_holds_the_heading():
    """Every key of the pilot's table is a dataclass field, and under it the schedule never turns: a controller run
    over 60 s at 0.01 s keeps the reset heading as its request (switch_interval_s and switch_jitter_s, left at their
    defaults, only time the switches, at which straight_probability 1.0 draws no turn)."""
    names = {field.name for field in fields(DirectionCommandConfig)}
    assert set(STRAIGHT_TABLE) <= names
    assert set(names) - set(STRAIGHT_TABLE) == {"speed_range", "switch_interval_s", "switch_jitter_s"}
    config = DirectionCommandConfig(**STRAIGHT_TABLE)
    assert asdict(config)["speed_range"] is None
    for seed in range(5):
        controller = DirectionCommandController(config)
        start = controller.reset(np.random.default_rng(seed), 0.25)
        assert start.desired_heading == 0.25 and start.desired_speed == 1.5
        for k in range(1, 6001):
            state = controller.update(k * 0.01, 0.25 + 0.3 * math.sin(k * 0.01))
            assert state.desired_heading == 0.25 and state.desired_speed == 1.5 and state.heading_active
        assert len(controller.events) > 1, "the switches fired"
        assert {(event["desired_heading"], event["desired_speed"]) for event in controller.events} == {(0.25, 1.5)}


# ======================================================================================== (2) the derived command section
def _command_free_locomotion() -> dict[str, Any]:
    """The committed locomotion [env] without its command keys (the pilot-3 [env] until pilot 4's TOML sets them)."""
    env_kwargs = dict(load_stage_config(TYRANNOSAURUS_REX, "locomotion")["env_kwargs"])
    return {key: value for key, value in env_kwargs.items() if key not in ("command_mode", "command_config")}


def _live(table: dict[str, Any], mode: str = "heading") -> dict[str, Any]:
    return {**_command_free_locomotion(), "command_mode": mode, "command_config": dict(table)}


def test_a_live_toml_table_is_hashed_with_the_manifest_its_env_reports():
    """No manifest passed: the command section is derived from the table, and it is the one the env built from the
    same [env] reports; the fingerprint equals the one hashed with that manifest passed explicitly."""
    identity = current_plant_identity(TYRANNOSAURUS_REX).to_dict()
    env_kwargs = _live(STRAIGHT_TABLE)
    env = TyrannosaurusRexAnatomicalEnv(**env_kwargs)
    try:
        reported = env.command_manifest()
    finally:
        env.close()
    assert reported == DirectionCommandController(DirectionCommandConfig(**STRAIGHT_TABLE)).manifest()
    derived = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=env_kwargs, plant_identity=identity)
    passed = stage_task_fingerprint(
        TYRANNOSAURUS_REX, "locomotion", env_kwargs=env_kwargs, plant_identity=identity, command_manifest=reported
    )
    assert derived == passed
    assert derived["command"] == json.loads(json.dumps(reported))
    assert derived["env"]["command_mode"] == "heading" and derived["env"]["command_config"] == STRAIGHT_TABLE
    command_free = stage_task_fingerprint(
        TYRANNOSAURUS_REX, "locomotion", env_kwargs=_command_free_locomotion(), plant_identity=identity
    )
    assert "command" not in command_free and derived["task_sha256"] != command_free["task_sha256"]
    low_level = derive_stage_task_fingerprint(
        model_id=TYRANNOSAURUS_REX,
        stage=2,
        backend="stable-baselines3",
        env_kwargs=env_kwargs,
        plant_identity=identity,
    )
    assert low_level == derived


def test_train_s_call_hashes_a_live_stage_toml(tmp_path):
    """train() derives the task as stage_task_fingerprint(model_id, stage, stage_config=config): a stage TOML whose
    [env] carries the command table, read by load_stage_config, is hashed (it was refused at 31d02e42)."""
    path = tmp_path / "locomotion.toml"
    lines = ["[env]", 'command_mode = "heading"', "[env.command_config]"]
    lines += [f"{key} = {'true' if value is True else repr(value)}" for key, value in STRAIGHT_TABLE.items()]
    path.write_text("\n".join(lines) + "\n")
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion", config_path=str(path))
    identity = current_plant_identity(TYRANNOSAURUS_REX)
    fingerprint = stage_task_fingerprint(TYRANNOSAURUS_REX, 2, stage_config=config, plant_identity=identity)
    assert fingerprint["command"]["config"] == asdict(DirectionCommandConfig(**STRAIGHT_TABLE))
    assert fingerprint["env"]["command_config"] == STRAIGHT_TABLE


def test_the_env_section_hashes_the_table_as_written():
    """The same command written as two different tables is two tasks: the command sections agree (the dataclass,
    field by field) while the env sections, and so the digests, differ.  Pilot 4's table is written once."""
    identity = current_plant_identity(TYRANNOSAURUS_REX).to_dict()
    one_key = {"turn_increment_max": 0.0}
    spelled_out = {
        field.name: getattr(DirectionCommandConfig(turn_increment_max=0.0), field.name)
        for field in fields(DirectionCommandConfig)
        if field.name != "speed_range"  # TOML has no null
    }
    assert DirectionCommandConfig(**one_key) == DirectionCommandConfig(**spelled_out)
    a = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=_live(one_key), plant_identity=identity)
    b = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=_live(spelled_out), plant_identity=identity)
    assert a["command"] == b["command"]
    assert a["env"] != b["env"] and a["task_sha256"] != b["task_sha256"]
    # Key order within the table does not matter (the payload is hashed with sorted keys).
    reordered = dict(reversed(list(spelled_out.items())))
    c = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=_live(reordered), plant_identity=identity)
    assert c["task_sha256"] == b["task_sha256"]


def test_a_passed_manifest_is_hashed_verbatim_and_nothing_is_derived():
    identity = current_plant_identity(TYRANNOSAURUS_REX).to_dict()
    manifest = DirectionCommandController(DirectionCommandConfig(**STRAIGHT_TABLE)).manifest()
    bumped = {**manifest, "adapter": manifest["adapter"] + "+next"}
    passed = stage_task_fingerprint(
        TYRANNOSAURUS_REX,
        "locomotion",
        env_kwargs=_live(STRAIGHT_TABLE),
        plant_identity=identity,
        command_manifest=bumped,
    )
    assert passed["command"]["adapter"] == bumped["adapter"]


def test_what_the_derivation_leaves_refused_or_carved_out():
    """A live mode with the dataclass object or with no command_config still needs the manifest passed (refused
    without it); a table under command_mode "none" is carved out as before, so the command-free digest holds; and
    the low-level compute_task_fingerprint derives nothing."""
    identity = current_plant_identity(TYRANNOSAURUS_REX).to_dict()
    for env_kwargs in (
        {**_command_free_locomotion(), "command_mode": "heading"},
        {**_command_free_locomotion(), "command_mode": "heading", "command_config": DirectionCommandConfig()},
    ):
        with pytest.raises(TaskFingerprintError, match="exactly when command_mode is not 'none'"):
            stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=env_kwargs, plant_identity=identity)
    carved = stage_task_fingerprint(
        TYRANNOSAURUS_REX,
        "locomotion",
        env_kwargs={**_command_free_locomotion(), "command_mode": "none", "command_config": dict(STRAIGHT_TABLE)},
        plant_identity=identity,
    )
    command_free = stage_task_fingerprint(
        TYRANNOSAURUS_REX, "locomotion", env_kwargs=_command_free_locomotion(), plant_identity=identity
    )
    assert carved == command_free
    with pytest.raises(TaskFingerprintError, match="exactly when command_mode is not 'none'"):
        compute_task_fingerprint(
            model_id=TYRANNOSAURUS_REX,
            stage=2,
            backend="stable-baselines3",
            env_kwargs=_live(STRAIGHT_TABLE),
            plant_identity=identity,
            perturbation_manifest=None,
        )
