"""Walking pilot 4's command route (pilot-only, never merged): a canonical stage's [env] carries a live command.

The maintainer's decisions (23) and (25) of 2026-11-05: pilot 4 keeps the T. rex walking stage (03_locomotion,
stage 2) and feeds a "point straight" command through the reserved command inputs, set in the stage TOML.  The hybrid
check of 2026-11-05 measured that config alone cannot do it; these tests pin the pilot-only edits that let it:

- (1) ``BaseDinoEnv.__init__`` builds the ``DirectionCommandConfig`` from a TOML ``command_config`` table (a mapping).
- (2) ``task_fingerprint.derive_stage_task_fingerprint`` derives a live stage's command manifest from that table when
  no manifest is passed, so ``train()``'s ``stage_task_fingerprint(stage_config=...)`` and the digest harness hash it.
- (3) ``task_fingerprint.validate_declared_parent`` accepts a non-root node's own earlier checkpoint only when the
  child stage's ``[curriculum]`` declares ``initialize_from_own_checkpoint = true`` (decision (25): this one stage);
  every other node of every model keeps the edge rule, which the tests here walk exhaustively.

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


# ======================================================================================== (3) the node's own checkpoint
#: The (model, stage id) pairs whose committed [curriculum] declares initialize_from_own_checkpoint.
DECLARING_OWN_CHECKPOINT: set[tuple[ModelId, str]] = set()


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


def _train_from(tmp_path: Path, monkeypatch: Any, stage_configs: dict[Any, Any], *, recorded: Any, child: Any) -> str:
    """train()'s launch path up to the backend import, on a checkpoint whose fingerprint records stage *recorded*
    (the pattern of test_train_base.py's TestTrainRefusesAnUndeclaredParent)."""
    import zipfile
    from types import SimpleNamespace

    from environments.shared import task_fingerprint, train_base
    from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE

    monkeypatch.setattr(train_base, "current_plant_identity", lambda model_id: SimpleNamespace(to_dict=dict))
    monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
    monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: (_ for _ in ()).throw(_SB3Reached()))
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = tmp_path / f"parent_{recorded}.zip"
    with zipfile.ZipFile(parent, "w") as archive:
        archive.writestr("data", json.dumps({MODEL_TASK_ATTRIBUTE: {**TYRANNOSAURUS_REX.record(), "stage": recorded}}))
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
