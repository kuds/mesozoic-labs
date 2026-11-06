"""Walking pilot 4's command route (pilot-only, never merged): a canonical stage's [env] carries a live command.

The maintainer's decisions (23) and (25) of 2026-11-05: pilot 4 keeps the T. rex walking stage (03_locomotion,
stage 2) and feeds a "point straight" command through the reserved command inputs, set in the stage TOML.  The hybrid
check of 2026-11-05 measured that config alone cannot do it; these tests pin the pilot-only edits that let it:

- (1) ``BaseDinoEnv.__init__`` builds the ``DirectionCommandConfig`` from a TOML ``command_config`` table (a mapping);
  a field that should be a number and is not is refused, not coerced.
- (2) ``task_fingerprint.derive_stage_task_fingerprint`` derives a live stage's command manifest from that table when
  no manifest is passed, so ``train()``'s ``stage_task_fingerprint(stage_config=...)`` and the digest harness hash it;
  it builds and checks the table as the env does, the mode rule included, so no table the env refuses is hashed.
- (3) ``task_fingerprint.validate_declared_parent`` accepts a non-root node's own earlier COMMAND-FREE checkpoint
  only when the child stage's ``[curriculum]`` declares ``initialize_from_own_checkpoint = true`` (decision (25):
  this one stage); every other node of every model keeps the edge rule, which the tests here walk exhaustively.
- (4) ``train_behaviors.read_recipe`` leaves pilot 4's T. rex locomotion keys (the command, the prey line, the heading
  price) out of the [env] the T. rex behavior recipes inherit, so the 11 recipes keep their task digests; every other
  model's recipes inherit their locomotion [env] whole, as before.
- (5) the committed T. rex locomotion TOML is pilot 4's: the straight command's one table, the prey line, the
  heading price, the fine-tune's [curriculum]; without them it is pilot 3's stage digest for digest.

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


@pytest.mark.parametrize(
    "name, value",
    [
        ("cruise_speed", "1.5"),
        ("cruise_speed", True),
        ("speed_scale", True),
        ("yaw_gain", "1.5"),
        ("stop_probability", False),
        ("minimum_turn_speed_fraction", None),
        ("switch_interval_s", [4.0]),
    ],
)
def test_a_table_number_must_be_a_number(name, value):
    """The dataclass coerces its numbers through float(), so a string or a bool would build silently, while the env
    section hashes the table as written (the code review measured cruise_speed "1.5" as another task with the same
    command): a table field that should be a number and is not is a TypeError, at construction and in the task
    fingerprint's derivation alike."""
    with pytest.raises(TypeError, match=rf"command_config\.{name} must be a number"):
        TyrannosaurusRexAnatomicalEnv(command_mode="heading", command_config={name: value})
    identity = current_plant_identity(TYRANNOSAURUS_REX).to_dict()
    with pytest.raises(TypeError, match=rf"command_config\.{name} must be a number"):
        stage_task_fingerprint(
            TYRANNOSAURUS_REX, "locomotion", env_kwargs=_live({name: value}), plant_identity=identity
        )


def test_a_table_takes_ints_and_floats_and_keeps_its_bool_and_its_range():
    """TOML integers are numbers (1 is the dataclass's 1.0); turn_slowdown stays a bool and speed_range a pair."""
    env = TyrannosaurusRexAnatomicalEnv(
        command_mode="heading_and_speed",
        command_config={"cruise_speed": 1, "speed_scale": 2, "turn_slowdown": False, "speed_range": [0, 1.5]},
    )
    try:
        assert env.command_config == DirectionCommandConfig(
            cruise_speed=1.0, speed_scale=2.0, turn_slowdown=False, speed_range=(0.0, 1.5)
        )
    finally:
        env.close()


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
#: Every locomotion [env] key pilot 4's TOML adds (the command, the prey line, the heading price).
PILOT4_ENV_KEYS = (
    "command_mode",
    "command_config",
    "prey_lateral_range",
    "walk_heading_price_weight",
    "walk_heading_price_scale_rad",
)


def _command_free_locomotion() -> dict[str, Any]:
    """The committed locomotion [env] without pilot 4's keys: walking pilot 3's [env] (R6, no command)."""
    env_kwargs = dict(load_stage_config(TYRANNOSAURUS_REX, "locomotion")["env_kwargs"])
    return {key: value for key, value in env_kwargs.items() if key not in PILOT4_ENV_KEYS}


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


def test_the_derivation_applies_the_env_s_mode_rule():
    """'heading' holds cruise_speed: a heading table with stops or a speed range is refused by the env built from the
    [env] and, now, by the derivation too (the code review measured such a table hashed while its env refused it),
    while the same table under 'heading_and_speed' builds and is hashed with the manifest its env reports."""
    identity = current_plant_identity(TYRANNOSAURUS_REX).to_dict()
    for table in ({**STRAIGHT_TABLE, "stop_probability": 0.1}, {**STRAIGHT_TABLE, "speed_range": [0.5, 1.5]}):
        with pytest.raises(ValueError, match="'heading' holds cruise_speed"):
            TyrannosaurusRexAnatomicalEnv(**_live(table))
        with pytest.raises(TaskFingerprintError, match="cannot be hashed: command_mode='heading' holds cruise_speed"):
            stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=_live(table), plant_identity=identity)
        live = _live(table, mode="heading_and_speed")
        env = TyrannosaurusRexAnatomicalEnv(**live)
        try:
            reported = env.command_manifest()
        finally:
            env.close()
        derived = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=live, plant_identity=identity)
        assert derived["command"] == json.loads(json.dumps(reported))


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


# ======================================================================================== (3) the node's own checkpoint
#: The (model, stage id) pairs whose committed [curriculum] declares initialize_from_own_checkpoint.
DECLARING_OWN_CHECKPOINT: set[tuple[ModelId, str]] = {(TYRANNOSAURUS_REX, "locomotion")}


def test_only_the_pilot_stage_declares_the_own_checkpoint_key():
    """Decision (25): the self-load is limited to one stage; no other node of any model declares the key."""
    from environments.shared.config import load_all_stages
    from environments.shared.curriculum.gate_schema import declared_initialize_from_own_checkpoint
    from environments.shared.species_names import model_ids
    from environments.shared.stage_manifest import load_stage_manifest

    declaring = set()
    for model_id in model_ids():
        configs = load_all_stages(model_id)
        for entry in load_stage_manifest(model_id).stages:
            curriculum = configs[entry.reference]["curriculum_kwargs"]
            if "initialize_from_own_checkpoint" in curriculum:
                assert declared_initialize_from_own_checkpoint(curriculum, stage=entry.id) is True
                declaring.add((model_id, entry.id))
    assert declaring == DECLARING_OWN_CHECKPOINT
    for path in sorted((REPOSITORY / "configs").rglob("*.toml")):
        if "initialize_from_own_checkpoint" in path.read_text(encoding="utf-8"):
            assert path.relative_to(REPOSITORY).as_posix() == "configs/tyrannosaurus_rex/anatomical/locomotion.toml"


def _every_edge_case() -> list[tuple[ModelId, Any, Any, list[Any]]]:
    """Per model and node: (model, node reference, declared parent reference or None, every node's reference)."""
    from environments.shared.species_names import model_ids
    from environments.shared.stage_manifest import load_stage_manifest

    cases = []
    for model_id in model_ids():
        manifest = load_stage_manifest(model_id)
        references = [entry.reference for entry in manifest.stages]
        for entry in manifest.stages:
            parent = manifest.parent_of(entry.id)
            cases.append((model_id, entry.reference, parent.reference if parent is not None else None, references))
    return cases


def _accepted(model_id: ModelId, child: Any, parent: Any, recorded: dict[str, Any] | None, **flag: Any) -> bool:
    from environments.shared.task_fingerprint import validate_declared_parent

    try:
        validate_declared_parent(
            recorded, declared_parent=parent, model_id=model_id, child_stage=child, artifact="x", **flag
        )
    except TaskFingerprintError:
        return False
    return True


def test_without_the_key_every_node_of_every_model_keeps_the_edge_rule():
    """The default (the keyword left out, as every caller but train() leaves it, and False): a non-root accepts
    exactly its declared parent and a root exactly itself, against a record of every node of its own model; another
    model's record, a bool stage and a record without the model's fields are refused."""
    from environments.shared.species_names import model_ids

    cases = _every_edge_case()
    others = list(model_ids())
    assert {case[0] for case in cases} == set(others) and any(parent is not None for _, _, parent, _ in cases)
    for model_id, child, parent, references in cases:
        for flag in ({}, {"accept_own_checkpoint": False}):
            for reference in references:
                expected = reference == (child if parent is None else parent)
                record = {**model_id.record(), "stage": reference}
                assert _accepted(model_id, child, parent, record, **flag) is expected, (model_id, child, reference)
            for other in others:
                if other != model_id:
                    assert not _accepted(model_id, child, parent, {**other.record(), "stage": child}, **flag)
            assert not _accepted(model_id, child, parent, {**model_id.record(), "stage": True}, **flag)
            assert not _accepted(model_id, child, parent, {"stage": child}, **flag)


def test_the_key_adds_exactly_the_node_itself():
    """accept_own_checkpoint=True: a non-root accepts its parent and itself and nothing else; a root is unchanged."""
    for model_id, child, parent, references in _every_edge_case():
        for reference in references:
            expected = reference == child or (parent is not None and reference == parent)
            record = {**model_id.record(), "stage": reference}
            assert _accepted(model_id, child, parent, record, accept_own_checkpoint=True) is expected
    with pytest.raises(TypeError, match="accept_own_checkpoint must be a bool"):
        _accepted(TYRANNOSAURUS_REX, 2, 1, None, accept_own_checkpoint="yes")


def test_the_own_checkpoint_path_admits_only_a_command_free_checkpoint():
    """The key brings a COMMAND-FREE earlier checkpoint of the node into its live task (pilot 3's walkers, whose
    records carry no command section).  A command-trained checkpoint of the node itself (a pilot-4 checkpoint) is
    refused even with the key: train()'s initialize_next_stage load would reseed its learned command statistics and
    reset ret_rms, so it is resumed instead.  The declared parent's records and every root are unchanged, command
    section or not, and without the key a command-trained own checkpoint is refused by the edge rule as before."""
    from environments.shared.task_fingerprint import validate_declared_parent

    command = {"command": {"schema": "mesozoic.direction-command/v1"}}
    for model_id, child, parent, _ in _every_edge_case():
        own = {**model_id.record(), "stage": child}
        if parent is None:
            for flag in (True, False):
                assert _accepted(model_id, child, parent, {**own, **command}, accept_own_checkpoint=flag)
            continue
        assert _accepted(model_id, child, parent, own, accept_own_checkpoint=True)
        assert _accepted(model_id, child, parent, {**model_id.record(), "stage": parent, **command})
        assert _accepted(
            model_id, child, parent, {**model_id.record(), "stage": parent, **command}, accept_own_checkpoint=True
        )
        assert not _accepted(model_id, child, parent, {**own, **command}, accept_own_checkpoint=True)
        assert not _accepted(model_id, child, parent, {**own, **command})
    with pytest.raises(TaskFingerprintError, match="is a command-trained checkpoint of .* itself") as refusal:
        validate_declared_parent(
            {**TYRANNOSAURUS_REX.record(), "stage": 2, **command},
            declared_parent=1,
            model_id=TYRANNOSAURUS_REX,
            child_stage=2,
            artifact="stage2_250000_steps.zip",
            accept_own_checkpoint=True,
        )
    assert "continue it with resume_same_stage" in str(refusal.value)


def test_the_key_is_a_bool_in_no_digest():
    """The schema knows the key and types it; it enters neither the gate view nor the recipe digest."""
    import copy

    from environments.shared.config import hyperparameters_sha256
    from environments.shared.curriculum.gate_schema import (
        GateSchemaError,
        declared_initialize_from_own_checkpoint,
        gate_config_view,
        validate_gate_config,
    )

    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    curriculum = {k: v for k, v in config["curriculum_kwargs"].items() if k != "initialize_from_own_checkpoint"}
    assert declared_initialize_from_own_checkpoint(curriculum) is False
    for value in (True, False):
        declared = {**curriculum, "initialize_from_own_checkpoint": value}
        assert validate_gate_config(2, declared) == validate_gate_config(2, curriculum)
        assert declared_initialize_from_own_checkpoint(declared) is value
        assert gate_config_view(declared) == gate_config_view(curriculum)
        with_key = copy.deepcopy(config)
        with_key["curriculum_kwargs"] = declared
        without = copy.deepcopy(config)
        without["curriculum_kwargs"] = curriculum
        for algorithm in ("ppo", "sac"):
            assert hyperparameters_sha256(with_key, algorithm) == hyperparameters_sha256(without, algorithm)
    for bad in ("true", 1, 0, None, 1.0):
        with pytest.raises(GateSchemaError, match="initialize_from_own_checkpoint must be true or false"):
            validate_gate_config(2, {**curriculum, "initialize_from_own_checkpoint": bad})


class _SB3Reached(RuntimeError):
    """train() passed the declared-parent check and went on to import the backend."""


def _train_from(
    tmp_path: Path, monkeypatch: Any, stage_configs: dict[Any, Any], *, recorded: Any, child: Any, command: bool = False
) -> str:
    """train()'s launch path up to the backend import, on a checkpoint whose fingerprint records stage *recorded*
    (the pattern of test_train_base.py's TestTrainRefusesAnUndeclaredParent); *command* adds a command section to
    the record, as a command-trained checkpoint carries."""
    import zipfile
    from types import SimpleNamespace

    from environments.shared import task_fingerprint, train_base
    from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE

    monkeypatch.setattr(train_base, "current_plant_identity", lambda model_id: SimpleNamespace(to_dict=dict))
    monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
    monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: (_ for _ in ()).throw(_SB3Reached()))
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / f"parent_{recorded}.zip"
    record = {**TYRANNOSAURUS_REX.record(), "stage": recorded}
    if command:
        record["command"] = {"schema": "mesozoic.direction-command/v1"}
    with zipfile.ZipFile(parent, "w") as archive:
        archive.writestr("data", json.dumps({MODEL_TASK_ATTRIBUTE: record}))
    output_dir = tmp_path / f"out_{recorded}_{child}"
    output_dir.mkdir()
    species: Any = SimpleNamespace(model_id=TYRANNOSAURUS_REX, env_class=object)
    try:
        train_base.train(
            species,
            stage_configs,
            child,
            total_timesteps=1,
            load_path=str(parent),
            task_load_mode="initialize_next_stage",
            output_dir=str(output_dir),
            use_tensorboard=False,
            verbose=0,
        )
    except _SB3Reached:
        assert not any(output_dir.iterdir())
        return "passed-the-check"
    return "returned"


def test_train_reads_the_key_of_the_stage_it_trains_and_nothing_else(tmp_path, monkeypatch):
    """train() passes THIS stage's key: set on locomotion, locomotion accepts its own checkpoint and still refuses a
    foreign node's; recovery and behavior, which do not declare it, still refuse their own; cleared, locomotion
    refuses its own again.  Every refusal comes before a directory exists."""
    import copy

    from environments.shared.config import load_all_stages

    committed = load_all_stages(TYRANNOSAURUS_REX)
    on = copy.deepcopy(committed)
    on[2]["curriculum_kwargs"]["initialize_from_own_checkpoint"] = True
    off = copy.deepcopy(committed)
    off[2]["curriculum_kwargs"].pop("initialize_from_own_checkpoint", None)
    assert _train_from(tmp_path / "a", monkeypatch, on, recorded=2, child=2) == "passed-the-check"
    # The committed locomotion stage declares the key itself (pilot 4's TOML).
    assert _train_from(tmp_path / "a0", monkeypatch, committed, recorded=2, child=2) == "passed-the-check"
    # ... for a command-free checkpoint of itself only: a command-trained one (a pilot-4 checkpoint) is resumed.
    with pytest.raises(TaskFingerprintError, match="is a command-trained checkpoint of .* itself"):
        _train_from(tmp_path / "a1", monkeypatch, committed, recorded=2, child=2, command=True)
    assert _train_from(tmp_path / "a2", monkeypatch, committed, recorded=1, child=2, command=True) == (
        "passed-the-check"
    )
    assert _train_from(tmp_path / "b", monkeypatch, on, recorded=1, child=2) == "passed-the-check"
    with pytest.raises(TaskFingerprintError, match=r"declared parent \(or, by its initialize_from_own_checkpoint"):
        _train_from(tmp_path / "c", monkeypatch, on, recorded=3, child=2)
    for configs in (on, off):
        for node in ("recovery", 3):
            with pytest.raises(TaskFingerprintError, match=r"declared parent\. Load"):
                _train_from(tmp_path / f"d{node}{id(configs)}", monkeypatch, configs, recorded=node, child=node)
    with pytest.raises(TaskFingerprintError, match=r"declared parent\. Load"):
        _train_from(tmp_path / "e", monkeypatch, off, recorded=2, child=2)
    bad = copy.deepcopy(committed)
    bad[2]["curriculum_kwargs"]["initialize_from_own_checkpoint"] = "true"
    from environments.shared.curriculum.gate_schema import GateSchemaError

    with pytest.raises(GateSchemaError, match="initialize_from_own_checkpoint"):
        _train_from(tmp_path / "f", monkeypatch, bad, recorded=2, child=2)


# ======================================================================================== (4) the recipes' inherited [env]
#: What pilot 4's T. rex locomotion [env] adds (decisions (23)-(28)): the live straight command, the prey on the spawn
#: line and the heading price (its keys are read by name; the env learns them in the price commit).
PILOT_LOCOMOTION_ENV: dict[str, Any] = {
    "command_mode": "heading",
    "command_config": dict(STRAIGHT_TABLE),
    "prey_lateral_range": [0.0, 0.0],
    "walk_heading_price_weight": 0.2,
    "walk_heading_price_scale_rad": 0.5,
}


def _recipe_paths(model_id: ModelId) -> list[Path]:
    return sorted((model_id.path(REPOSITORY / "configs") / "behaviors").glob("*.toml"))


def _models_with_recipes() -> list[ModelId]:
    from environments.shared.species_names import model_ids

    return [model_id for model_id in sorted(model_ids()) if _recipe_paths(model_id)]


def _locomotion_env(model_id: ModelId) -> dict[str, Any]:
    from environments.shared.stage_manifest import load_stage_manifest

    path = model_id.path(REPOSITORY / "configs") / load_stage_manifest(model_id).by_id("locomotion").config_file
    return dict(tomllib.loads(path.read_text(encoding="utf-8"))["env"])


class _LocomotionEnvAs:
    """Stands in for train_behaviors' tomllib: the model's locomotion TOML parses with its [env] replaced by *env*;
    every other file (the recipe itself) parses as written."""

    def __init__(self, model_id: ModelId, env: dict[str, Any]) -> None:
        from environments.shared.stage_manifest import load_stage_manifest

        config_file = load_stage_manifest(model_id).by_id("locomotion").config_file
        self.path = (model_id.path(REPOSITORY / "configs") / config_file).resolve()
        self.env = env
        self.reads = 0

    def load(self, stream: Any) -> dict[str, Any]:
        data = tomllib.load(stream)
        if Path(stream.name).resolve() == self.path:
            self.reads += 1
            data = {**data, "env": dict(self.env)}
        return data


def _inherited(recipe: Path, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """read_recipe's env kwargs without the terrain sampler it adds for a recipe with [terrain_sampler]."""
    has_sampler = "terrain_sampler" in tomllib.loads(recipe.read_text(encoding="utf-8"))
    assert ("terrain_sampler" in env_kwargs) is has_sampler, recipe.name
    return {key: value for key, value in env_kwargs.items() if key != "terrain_sampler"}


def _read(model_id: ModelId, recipe: Path, monkeypatch: Any, locomotion_env: dict[str, Any]) -> dict[str, Any]:
    from environments.shared import train_behaviors

    stand_in = _LocomotionEnvAs(model_id, locomotion_env)
    monkeypatch.setattr(train_behaviors, "tomllib", stand_in)
    try:
        env_kwargs = train_behaviors.read_recipe(recipe, model_id)[3]
    finally:
        monkeypatch.undo()
    assert stand_in.reads == 1, "read_recipe read the locomotion TOML through the stand-in"
    return _inherited(recipe, env_kwargs)


def test_the_tyrannosaurus_recipes_leave_the_pilot_locomotion_keys_behind(monkeypatch):
    """Every T. rex recipe resolves the same [env] whether or not the locomotion [env] carries pilot 4's keys: the
    command, the prey line and the price stay with the locomotion stage; R6 and the rest are inherited as before."""
    from environments.shared.train_behaviors import LOCOMOTION_ONLY_ENV_KEYS

    assert set(LOCOMOTION_ONLY_ENV_KEYS) == {TYRANNOSAURUS_REX}
    assert set(LOCOMOTION_ONLY_ENV_KEYS[TYRANNOSAURUS_REX]) == set(PILOT_LOCOMOTION_ENV)
    committed = _locomotion_env(TYRANNOSAURUS_REX)
    without = {key: value for key, value in committed.items() if key not in PILOT_LOCOMOTION_ENV}
    with_keys = {**without, **PILOT_LOCOMOTION_ENV}
    recipes = _recipe_paths(TYRANNOSAURUS_REX)
    assert len(recipes) == 11
    for recipe in recipes:
        own = tomllib.loads(recipe.read_text(encoding="utf-8")).get("env", {})
        assert not set(own) & set(PILOT_LOCOMOTION_ENV), recipe.name
        expected = {**without, "max_episode_steps": 2500, **own}
        assert _read(TYRANNOSAURUS_REX, recipe, monkeypatch, with_keys) == expected, recipe.name
        assert _read(TYRANNOSAURUS_REX, recipe, monkeypatch, without) == expected, recipe.name
        assert _read(TYRANNOSAURUS_REX, recipe, monkeypatch, committed) == expected, recipe.name


def test_a_recipe_s_own_env_keeps_a_key_the_locomotion_drop_leaves_behind(tmp_path):
    """The drop applies to the INHERITED locomotion [env] only: a T. rex recipe that sets a dropped key in its own
    [env] keeps it.  No committed recipe sets one, so a drop applied to the merged [env] behaved the same on all 11
    (the code review's surviving mutant); this recipe, a committed one with prey_lateral_range [-1, 1] written into
    its own [env], tells them apart."""
    from environments.shared.train_behaviors import read_recipe

    recipe = _recipe_paths(TYRANNOSAURUS_REX)[0]
    text = recipe.read_text(encoding="utf-8")
    line = "prey_lateral_range = [-1.0, 1.0]\n"
    text = text.replace("[env]\n", "[env]\n" + line, 1) if "[env]\n" in text else text + "\n[env]\n" + line
    own_env = tomllib.loads(text)["env"]
    assert own_env["prey_lateral_range"] == [-1.0, 1.0]
    copied = tmp_path / recipe.name
    copied.write_text(text, encoding="utf-8")
    resolved = _inherited(copied, read_recipe(copied, TYRANNOSAURUS_REX)[3])
    committed = _locomotion_env(TYRANNOSAURUS_REX)
    without = {key: value for key, value in committed.items() if key not in PILOT_LOCOMOTION_ENV}
    assert resolved == {**without, "max_episode_steps": 2500, **own_env}
    assert resolved["prey_lateral_range"] == [-1.0, 1.0]
    assert not {"command_mode", "command_config", "walk_heading_price_weight"} & set(resolved)


def _golden_recipe_tasks(model_id: ModelId) -> dict[str, str]:
    golden = (REPOSITORY / "configs" / "digest_snapshot.generated.txt").read_text(encoding="utf-8").splitlines()
    tasks = {}
    for line in golden:
        fields_ = line.split("\t")
        if fields_[:3] == ["behavior", model_id.species, model_id.embodiment] and fields_[4] == "task_sha256":
            tasks[fields_[3]] = fields_[5]
    return tasks


def _recipe_task(model_id: ModelId, recipe: Path, monkeypatch: Any, locomotion_env: dict[str, Any]) -> str:
    """The recipe's task_sha256 as the digest harness computes it, under a locomotion [env] of *locomotion_env*."""
    from environments.shared import train_behaviors
    from environments.shared.harnesses.digest_snapshot import _behavior_task

    monkeypatch.setattr(train_behaviors, "tomllib", _LocomotionEnvAs(model_id, locomotion_env))
    try:
        return str(_behavior_task(recipe, model_id)["task_sha256"])
    finally:
        monkeypatch.undo()


def test_the_tyrannosaurus_recipe_tasks_stay_the_golden_s_under_the_pilot_keys(monkeypatch):
    """The digest harness's own recipe task derivation, with pilot 4's keys in the locomotion [env]: all 11 T. rex
    recipe tasks equal the committed golden's lines.  Without the drop (an empty key list) the command keys reach the
    behavior env, which refuses them, and the prey line alone moves every recipe's task: the drop is what holds them."""
    from environments.shared import train_behaviors

    golden = _golden_recipe_tasks(TYRANNOSAURUS_REX)
    recipes = _recipe_paths(TYRANNOSAURUS_REX)
    assert sorted(golden) == [recipe.stem for recipe in recipes]
    committed = _locomotion_env(TYRANNOSAURUS_REX)
    with_keys = {**committed, **PILOT_LOCOMOTION_ENV}
    for recipe in recipes:
        assert _recipe_task(TYRANNOSAURUS_REX, recipe, monkeypatch, with_keys) == golden[recipe.stem], recipe.name
    no_drop = {**train_behaviors.LOCOMOTION_ONLY_ENV_KEYS, TYRANNOSAURUS_REX: ()}
    prey_only = {key: value for key, value in committed.items() if key not in PILOT_LOCOMOTION_ENV}
    prey_only["prey_lateral_range"] = [0.0, 0.0]
    recipe = recipes[0]
    with monkeypatch.context() as patch:
        patch.setattr(train_behaviors, "LOCOMOTION_ONLY_ENV_KEYS", no_drop)
        with pytest.raises(ValueError, match="a behavior env runs command_mode 'heading_and_speed' on commands="):
            _recipe_task(TYRANNOSAURUS_REX, recipe, monkeypatch, with_keys)
        assert _recipe_task(TYRANNOSAURUS_REX, recipe, monkeypatch, prey_only) != golden[recipe.stem]


def test_every_other_model_s_recipes_inherit_their_locomotion_env_whole(monkeypatch):
    """No other model changes: each recipe of every other model resolves {**locomotion [env], 2,500 steps, **recipe
    [env]} exactly as at 31d02e42, prey_lateral_range included where its locomotion [env] sets one (compsognathus
    [0, 0], dibothrosuchus [-1, 1]); and a pilot key written into another model's locomotion [env] is inherited, so
    the drop is keyed to the T. rex alone."""
    from environments.shared.train_behaviors import read_recipe

    others = [model_id for model_id in _models_with_recipes() if model_id != TYRANNOSAURUS_REX]
    assert len(others) == 5
    carried = set()
    for model_id in others:
        locomotion = _locomotion_env(model_id)
        for recipe in _recipe_paths(model_id):
            own = tomllib.loads(recipe.read_text(encoding="utf-8")).get("env", {})
            resolved = _inherited(recipe, read_recipe(recipe, model_id)[3])
            assert resolved == {**locomotion, "max_episode_steps": 2500, **own}, (model_id, recipe.name)
            if "prey_lateral_range" in locomotion and "prey_lateral_range" not in own:
                assert resolved["prey_lateral_range"] == locomotion["prey_lateral_range"]
                carried.add(str(model_id))
        recipe = _recipe_paths(model_id)[0]
        priced = {**locomotion, "prey_lateral_range": [0.0, 0.0], "walk_heading_price_weight": 0.2}
        resolved = _read(model_id, recipe, monkeypatch, priced)
        assert resolved["prey_lateral_range"] == [0.0, 0.0] and resolved["walk_heading_price_weight"] == 0.2
    assert carried == {
        "compsognathus_longipes/anatomical",
        "compsognathus_longipes/robot",
        "dibothrosuchus_elaphros/anatomical",
    }


# ======================================================================================== (5) the committed pilot-4 TOML
#: Walking pilot 3's locomotion digests at 31d02e42 (configs/digest_snapshot.generated.txt there): its task and its
#: PPO and SAC recipes.
PILOT3_TASK_SHA256 = "sha256:782d95b607d0086d0cefe475c9caf8ac420c1f2b74d23b786808aaee98232a6a"
PILOT3_HYPERPARAMETERS_SHA256 = {
    "ppo": "sha256:85881fc441355fc58f353a089d665c857b377aa1e7a9b6a5a5f89ef64271af4c",
    "sac": "sha256:05cce74dfc1365857f67ed187f8be9309f99146ea69641559fa5579f5351424e",
}
#: Pilot 3's three entry-shaping lines, which pilot 4 switches off (decision (26)).
PILOT3_SHAPING = {"warmup_timesteps": 300000, "ramp_start_value": 0.2, "warm_start_log_std": -2.0}
LOCOMOTION_TOML = REPOSITORY / "configs" / "tyrannosaurus_rex" / "anatomical" / "locomotion.toml"


def test_the_committed_locomotion_toml_carries_pilot_4_s_values():
    """The TOML as written (decisions (23)-(28)): the straight command's one table, value for value and type for type
    (the env section hashes it as written), the prey line, the price, and the [curriculum] of the fine-tune."""
    raw = tomllib.loads(LOCOMOTION_TOML.read_text(encoding="utf-8"))
    env, curriculum = raw["env"], raw["curriculum"]
    assert env["command_mode"] == "heading"
    assert env["command_config"] == STRAIGHT_TABLE
    assert {key: type(value) for key, value in env["command_config"].items()} == {
        key: type(value) for key, value in STRAIGHT_TABLE.items()
    }
    assert env["prey_lateral_range"] == [0.0, 0.0]
    assert (env["walk_heading_price_weight"], env["walk_heading_price_scale_rad"]) == (0.2, 0.5)
    expected = {
        "timesteps": 2_000_000,
        "eval_freq": 65_536,
        "eval_episodes": 10,
        "checkpoint_freq": 250_000,
        "max_checkpoints": 0,
        "warmup_timesteps": 0,
        "ramp_start_value": 2.0,
        "ramp_timesteps": 2_000_000,
        "initialize_from_own_checkpoint": True,
        "collapse_peak_floor_reference": 1064.0,
        "collapse_peak_floor_fraction": 0.45,
    }
    assert {key: curriculum[key] for key in expected} == expected
    assert "warm_start_log_std" not in curriculum
    assert env["forward_vel_weight"] == 2.0, "the constant forward weight the ramp starts and ends at"


def test_pilot_4_changes_nothing_else_in_pilot_3_s_locomotion_stage():
    """Without pilot 4's five [env] keys the stage's task is pilot 3's task exactly, and with pilot 3's three shaping
    lines put back its PPO and SAC recipes are pilot 3's: R6, [ppo], [sac] and the rest of [env] did not move."""
    import copy

    from environments.shared.config import hyperparameters_sha256

    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    identity = current_plant_identity(TYRANNOSAURUS_REX)
    pilot3_env = _command_free_locomotion()
    task = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=pilot3_env, plant_identity=identity)
    assert task["task_sha256"] == PILOT3_TASK_SHA256
    pilot3 = copy.deepcopy(config)
    pilot3["env_kwargs"] = pilot3_env
    pilot3["curriculum_kwargs"].update(PILOT3_SHAPING)
    for algorithm, digest in PILOT3_HYPERPARAMETERS_SHA256.items():
        assert hyperparameters_sha256(pilot3, algorithm) == digest
        assert hyperparameters_sha256(config, algorithm) != digest


def test_the_committed_stage_is_hashed_as_the_golden_records_it_and_builds_straight():
    """train()'s derivation of the committed live stage is the golden's task line; the env built from the stage config
    starts on a straight command at top-of-scale speed with the prey on the spawn line, and the price on."""
    golden = {}
    for line in (REPOSITORY / "configs" / "digest_snapshot.generated.txt").read_text(encoding="utf-8").splitlines():
        fields_ = line.split("\t")
        if fields_[:4] == ["stage", "tyrannosaurus_rex", "anatomical", "locomotion"]:
            golden[fields_[4]] = fields_[-1]
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    identity = current_plant_identity(TYRANNOSAURUS_REX)
    fingerprint = stage_task_fingerprint(TYRANNOSAURUS_REX, 2, stage_config=config, plant_identity=identity)
    assert fingerprint["task_sha256"] == golden["task_sha256"] != PILOT3_TASK_SHA256
    assert fingerprint["command"]["config"] == asdict(DirectionCommandConfig(**STRAIGHT_TABLE))
    env = TyrannosaurusRexAnatomicalEnv(**config["env_kwargs"])
    try:
        for seed in (3042, 3043, 3044):
            obs, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(obs[-3:], np.array([1.0, 0.0, 0.0], dtype=np.float32))
            state = env._command_state
            assert state is not None and state.desired_heading == wrap_angle(env._heading())
            assert env._initial_prey_dir_2d[1] == 0.0 and env._initial_prey_dir_2d[0] == pytest.approx(1.0)
            _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert {"walk_heading_error_rad", "reward_walk_heading_price"} <= set(info)
    finally:
        env.close()
