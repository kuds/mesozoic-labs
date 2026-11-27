"""The command plumbing of decision D-D60: manifest stages can run a live command.

* **[commands]** in a stage TOML builds the stage's ``command_mode`` and
  ``DirectionCommandConfig`` (``config.load_stage_config``);
  ``save_stage_config`` records every field so the frozen config rebuilds
  exactly; unknown keys, a non-live ``mode``, ``[env]`` command keys and a
  table on the stance are refused; ``extends`` inherits it key by key.
* **Task fingerprint**: a live stage's command section is its controller
  manifest, derived from the config the env is built from and equal to the
  env's ``command_manifest()``; every committed ("none") stage keeps the
  ``task_sha256`` the digest snapshot records, and ``train_base.train``
  fingerprints, trains and records a live stage under PPO and SAC.
* **Tracking in the base env**: the tracking reward, metrics, info keys and
  dwell are ``BaseDinoEnv``'s, live only.  The pilot behavior recipes
  reproduce, exactly, the rollouts captured before the move
  (``pilot_rollout_golden.py``), in the numerics environment they were
  captured in, and their first steps' rewards, info key order and tracking
  and command values within a tolerance on any machine with the pinned
  MuJoCo.
* **The command stream**: a live mode draws nothing from the reset stream,
  so its joint, velocity and push-schedule draws equal the "none" twin's on a
  seeded reset and every unseeded one after it, and the Phase C reset golden
  holds under a live command.
* **initial_heading_offset_max**: validated, drawn from the controller's own
  stream, nothing drawn at 0, and recorded only when set.
"""

from __future__ import annotations

import functools
import json
import math
import re
import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.base_env import BaseDinoEnv
from environments.shared.command_frame import COMMAND_STREAM_DOMAIN, COMMAND_WIDTH
from environments.shared.config import FORWARD_VELOCITY_RAMP_KEYS, build_env, load_stage_config, save_stage_config
from environments.shared.direction_commands import (
    DirectionCommandConfig,
    DirectionCommandController,
    gaussian_tracking_reward,
    wrap_angle,
)
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.species_names import model_ids
from environments.shared.species_registry import get_species_config
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import (
    TaskFingerprintError,
    check_env_command_manifest,
    stage_command_manifest,
    stage_task_fingerprint,
)
from environments.shared.tests import pilot_rollout_golden, reset_golden
from environments.shared.tests.stage_toml_helpers import replace_commands_table

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
MODEL_IDS = model_ids(backend="stable-baselines3")

#: The fields DirectionCommandConfig had before decision D-D60; a default config records exactly these.
PRE_D_D60_FIELDS = (
    "cruise_speed",
    "speed_range",
    "stop_probability",
    "turn_increment_max",
    "straight_probability",
    "switch_interval_s",
    "switch_jitter_s",
    "speed_scale",
    "lateral_speed_scale",
    "yaw_rate_scale",
    "yaw_gain",
    "yaw_rate_max",
    "turn_slowdown",
    "minimum_turn_speed_fraction",
)

#: A walk-like narrow command (the design's walk: heading hold, reset-only stops, no turns).
WALK_COMMANDS = """
[commands]
mode = "heading_and_speed"
cruise_speed = 0.5
speed_range = [0.25, 0.5]
stop_probability = 0.15
turn_increment_max = 0.0
switch_interval_s = 100.0
initial_heading_offset_max = 0.26
tracking_weight = 2.0
"""


def _model_dir(tmp_path: Path, model_id: ModelId) -> Path:
    """A copy of *model_id*'s committed config directory (stage manifest beside its stage files)."""
    target = tmp_path / model_id.species / model_id.embodiment
    shutil.copytree(model_id.path(REPOSITORY_ROOT / "configs"), target)
    return target


def _with_table(path: Path, text: str) -> Path:
    """*path* with its own [commands] (the committed walk's, decision D-D62) replaced by *text*, and without the
    forward-velocity ramp a live stage refuses."""
    return replace_commands_table(path, text)


def _live_config(tmp_path: Path, model_id: ModelId, stage: str = "locomotion", text: str = WALK_COMMANDS) -> dict:
    """*stage*'s committed config with its [commands] table replaced by *text* (or *text* added, on a stage with
    none), loaded from a copy of its directory."""
    directory = _model_dir(tmp_path, model_id)
    entry = load_stage_manifest(model_id).by_id(stage)
    return load_stage_config(model_id, stage, config_path=str(_with_table(directory / entry.config_file, text)))


# ---------------------------------------------------------------------------
# (a) The [commands] table
# ---------------------------------------------------------------------------


def test_a_commands_table_builds_the_stage_command_kwargs(tmp_path):
    config = _live_config(tmp_path, TYRANNOSAURUS_REX)
    committed = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    env_kwargs = dict(config["env_kwargs"])
    assert env_kwargs.pop("command_mode") == "heading_and_speed"
    command_config = env_kwargs.pop("command_config")
    committed_env = {k: v for k, v in committed["env_kwargs"].items() if k not in ("command_mode", "command_config")}
    assert env_kwargs == committed_env, "the table sets the two command kwargs and nothing else"
    assert command_config == DirectionCommandConfig(
        cruise_speed=0.5,
        speed_range=(0.25, 0.5),
        stop_probability=0.15,
        turn_increment_max=0.0,
        switch_interval_s=100.0,
        initial_heading_offset_max=0.26,
        tracking_weight=2.0,
    )
    assert {key: config[key] for key in ("ppo_kwargs", "sac_kwargs")} == {
        key: committed[key] for key in ("ppo_kwargs", "sac_kwargs")
    }
    # The forward-velocity ramp is dropped: a live stage's entry ramp is the cruise ramp (decision D-D62).
    assert config["curriculum_kwargs"] == {
        key: value for key, value in committed["curriculum_kwargs"].items() if key not in FORWARD_VELOCITY_RAMP_KEYS
    }


def test_every_stance_and_recovery_runs_command_mode_none_and_every_walk_a_live_command():
    """Every stance and recovery declares no [commands] (command_mode "none"); every locomotion node, the walk,
    declares the live walk command (decision D-D62)."""
    for model_id in MODEL_IDS:
        stages = {entry.id: entry for entry in load_stage_manifest(model_id).stages}
        assert set(stages) <= {"stance", "recovery", "locomotion"}, (model_id, sorted(stages))
        for stage_id, entry in stages.items():
            env_kwargs = load_stage_config(model_id, entry.reference)["env_kwargs"]
            if stage_id == "locomotion":
                assert env_kwargs["command_mode"] == "heading_and_speed", model_id
                assert isinstance(env_kwargs["command_config"], DirectionCommandConfig), model_id
            else:
                assert "command_mode" not in env_kwargs and "command_config" not in env_kwargs, (model_id, stage_id)


def test_save_stage_config_records_every_field_and_the_frozen_config_rebuilds_exactly(tmp_path):
    config = _live_config(tmp_path, TYRANNOSAURUS_REX)
    env_class = get_species_config(TYRANNOSAURUS_REX).env_class
    path = save_stage_config(
        tmp_path / "stage", "locomotion", config, "PPO", env_class=env_class, model_id=TYRANNOSAURUS_REX
    )
    recorded = json.loads(path.read_text())["reward_weights"]
    command_config = config["env_kwargs"]["command_config"]
    assert recorded["command_mode"] == "heading_and_speed"
    assert recorded["command_config"] == json.loads(json.dumps(command_config.field_record()))
    assert set(recorded["command_config"]) == {field.name for field in fields(DirectionCommandConfig)}
    assert DirectionCommandConfig(**recorded["command_config"]) == command_config
    # The recorded block re-derives the stage's task: a JSON object hashes as the dataclass it was saved from.
    assert (
        stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=recorded)["task_sha256"]
        == stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", stage_config=config)["task_sha256"]
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('[commands]\nmode = "heading_and_speed"\ncruise_sped = 0.5', "unknown \\[commands\\] key"),
        ("[commands]\ncruise_speed = 0.5", "mode must be one of"),
        ('[commands]\nmode = "none"', "declares no \\[commands\\] table"),
        ('[commands]\nmode = "walk"', "mode must be one of"),
        ('[commands]\nmode = "heading_and_speed"\nstop_probability = 2.0', "stop_probability must lie"),
        ('[commands]\nmode = "heading_and_speed"\ninitial_heading_offset_max = 4.0', "initial_heading_offset_max"),
        ('[commands]\nmode = "heading"\nspeed_range = [0.25, 0.5]', "holds cruise_speed"),
        ('[commands]\nmode = "heading_and_speed"\ntracking_weight = 0.0', "tracking_weight must be positive"),
    ],
)
def test_a_commands_table_is_refused_naming_its_file(tmp_path, text, match):
    with pytest.raises(ValueError, match=match) as raised:
        _live_config(tmp_path, TYRANNOSAURUS_REX, text=text)
    assert "locomotion.toml" in str(raised.value)


@pytest.mark.parametrize("key", ["command_mode", "command_config"])
def test_env_never_states_the_command_task(tmp_path, key):
    directory = _model_dir(tmp_path, TYRANNOSAURUS_REX)
    path = directory / "locomotion.toml"
    text = path.read_text().replace("[env]\n", f'[env]\n{key} = "heading_and_speed"\n', 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="states its command task in a \\[commands\\] table"):
        load_stage_config(TYRANNOSAURUS_REX, "locomotion", config_path=str(path))


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_the_stance_declares_no_commands_table(tmp_path, model_id):
    with pytest.raises(ValueError, match="declares no \\[commands\\] table: it runs command_mode"):
        _live_config(tmp_path, model_id, stage="stance")


def test_the_stance_refusal_covers_an_inherited_table_and_the_legacy_reference(tmp_path):
    directory = _model_dir(tmp_path, TYRANNOSAURUS_REX)
    _with_table(directory / "locomotion.toml", WALK_COMMANDS)
    stance = directory / "stance.toml"
    stance.write_text('extends = { stage = "locomotion", tables = ["commands"] }\n' + stance.read_text())
    for reference in ("stance", 1):
        with pytest.raises(ValueError, match="declares no \\[commands\\] table"):
            load_stage_config(TYRANNOSAURUS_REX, reference, config_path=str(stance))


def test_extends_inherits_commands_key_by_key(tmp_path):
    """A stage that widens its parent's command distribution states only the keys it changes."""
    directory = _model_dir(tmp_path, TYRANNOSAURUS_REX)
    _with_table(directory / "locomotion.toml", WALK_COMMANDS)
    child = directory / "recovery.toml"
    text = child.read_text().replace(
        'extends = { stage = "stance", tables = ["stage", "env", "ppo"] }',
        'extends = { stage = "locomotion", tables = ["commands"] }',
    )
    child.write_text(text + "\n[commands]\nstop_probability = 0.3\nturn_increment_max = 0.5\n")
    config = load_stage_config(TYRANNOSAURUS_REX, "recovery", config_path=str(child))
    assert config["env_kwargs"]["command_mode"] == "heading_and_speed"
    assert config["env_kwargs"]["command_config"] == replace(
        _live_config(tmp_path / "parent", TYRANNOSAURUS_REX)["env_kwargs"]["command_config"],
        stop_probability=0.3,
        turn_increment_max=0.5,
    )


def test_extends_refuses_a_commands_table_the_parent_does_not_declare(tmp_path):
    directory = _model_dir(tmp_path, TYRANNOSAURUS_REX)
    child = directory / "recovery.toml"
    child.write_text(child.read_text().replace('"stage", "env", "ppo"]', '"stage", "env", "ppo", "commands"]'))
    with pytest.raises(ValueError, match="extends lists \\[commands\\], which .*stance.toml does not declare"):
        load_stage_config(TYRANNOSAURUS_REX, "recovery", config_path=str(child))


# ---------------------------------------------------------------------------
# (b) The task fingerprint
# ---------------------------------------------------------------------------


def _snapshot_stage_digests() -> dict[tuple[str, str, str], str]:
    digests = {}
    for line in (REPOSITORY_ROOT / "configs" / "digest_snapshot.generated.txt").read_text().splitlines():
        fields_ = line.split("\t")
        if fields_[0] == "stage" and fields_[4] == "task_sha256":
            digests[(fields_[1], fields_[2], fields_[3])] = fields_[5]
    return digests


def test_every_committed_stage_keeps_its_snapshot_fingerprint():
    """The command plumbing moved no "none" stage: each stance's and recovery's task_sha256 is the digest snapshot's
    line.  The walk runs its live command since decision D-D62; its line is test_task_fingerprint_stage.py's."""
    snapshot = _snapshot_stage_digests()
    seen = set()
    for model_id in MODEL_IDS:
        for entry in load_stage_manifest(model_id).stages:
            if entry.id == "locomotion":
                continue
            fingerprint = stage_task_fingerprint(model_id, entry.reference)
            key = (model_id.species, model_id.embodiment, entry.id)
            assert fingerprint["task_sha256"] == snapshot[key], key
            assert "command" not in fingerprint and "command_mode" not in fingerprint["env"]
            assert stage_command_manifest(model_id, load_stage_config(model_id, entry.reference)["env_kwargs"]) is None
            seen.add(key)
    assert seen == {key for key in snapshot if key[2] != "locomotion"}


def test_a_live_stage_fingerprints_with_the_command_manifest_its_env_reports(tmp_path):
    config = _live_config(tmp_path, TYRANNOSAURUS_REX)
    live = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", stage_config=config)
    committed = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    none_kwargs = {k: v for k, v in committed["env_kwargs"].items() if k not in ("command_mode", "command_config")}
    none = stage_task_fingerprint(
        TYRANNOSAURUS_REX, "locomotion", stage_config={**committed, "env_kwargs": none_kwargs}
    )
    controller = DirectionCommandController(config["env_kwargs"]["command_config"])
    assert live["command"] == json.loads(json.dumps(controller.manifest()))
    assert live["env"]["command_mode"] == "heading_and_speed" and live["task_sha256"] != none["task_sha256"]
    assert {key: value for key, value in live["env"].items() if not key.startswith("command_")} == none["env"]
    env = get_species_config(TYRANNOSAURUS_REX).env_class(**config["env_kwargs"])
    plain = get_species_config(TYRANNOSAURUS_REX).env_class(**none_kwargs)
    try:
        check_env_command_manifest(env.command_manifest(), live, artifact="the live env")
        check_env_command_manifest(plain.command_manifest(), none, artifact="the none env")
        with pytest.raises(TaskFingerprintError, match="does not run the command task the stage declares"):
            check_env_command_manifest(plain.command_manifest(), live, artifact="the none env")
        with pytest.raises(TaskFingerprintError, match="does not run the command task the stage declares"):
            check_env_command_manifest(env.command_manifest(), none, artifact="the live env")
        assert live == stage_task_fingerprint(
            TYRANNOSAURUS_REX, "locomotion", stage_config=config, command_manifest=env.command_manifest()
        )
    finally:
        env.close()
        plain.close()


def _harness_stage_lines(
    model_id: ModelId, stage: str, config: dict[str, Any], scratch: Path
) -> tuple[dict[str, str], list[str], int]:
    """The digest snapshot's lines for one stage config: {name: value}, the raw lines and the ERROR count."""
    from environments.shared.harnesses import digest_snapshot
    from environments.shared.plant_contract import current_plant_identity

    out = digest_snapshot._Snapshot(REPOSITORY_ROOT, keep=True)
    entry = load_stage_manifest(model_id).by_id(stage)
    identity = current_plant_identity(model_id, verify_generated=False)
    digest_snapshot._stage_lines(out, model_id, entry, config, identity, scratch)
    named = {line.split("\t")[4]: line.split("\t")[5] for line in out.kept if line.startswith("stage\t")}
    return named, out.kept, out.errors


def test_the_digest_snapshot_fingerprints_a_live_stage_as_training_does(tmp_path, monkeypatch):
    """The harness's task_sha256 line is stage_task_fingerprint's, so a stage with a [commands] table gets one.

    An ERROR line there would make ``--write`` refuse and ``--check`` fail, so the first committed live stage could
    not record its fingerprint.  The line stays an oracle of the helper (test_task_fingerprint_stage.py): it hashes
    the command manifest the stage's env reports, without the helper's derivation from the config.  The committed
    ("none") stage's lines stay the golden's, and the stage's reward capture runs under the live command too.
    """
    from environments.shared import task_fingerprint
    from environments.shared.harnesses import digest_snapshot

    model_id = COMPSOGNATHUS_LONGIPES
    golden = {
        line.split("\t")[4]: line.split("\t")[5]
        for line in (REPOSITORY_ROOT / digest_snapshot.GOLDEN).read_text().splitlines()
        if line.startswith(f"stage\t{model_id.species}\t{model_id.embodiment}\tlocomotion\t")
    }
    # The committed walk is live (decision D-D62): its lines are the helper's, in the golden's order (the golden's
    # values are test_task_fingerprint_stage.py's and the digest check's to compare).
    committed, _, errors = _harness_stage_lines(
        model_id, "locomotion", load_stage_config(model_id, "locomotion"), tmp_path / "committed"
    )
    assert errors == 0 and list(committed) == list(golden)
    assert committed["task_sha256"] == stage_task_fingerprint(model_id, "locomotion")["task_sha256"]

    config = _live_config(tmp_path / "configs", model_id)
    live, lines, errors = _harness_stage_lines(model_id, "locomotion", config, tmp_path / "live")
    assert errors == 0, [line for line in lines if "\tERROR\t" in line]
    expected = stage_task_fingerprint(model_id, "locomotion", stage_config=config)
    assert "command" in expected
    assert live["task_sha256"] == expected["task_sha256"] != committed["task_sha256"]
    assert list(live) == list(committed), "every stage line, in the golden's order"
    assert live["gate_sha256"] == committed["gate_sha256"], "the table changes the task, not the gate"

    def no_helper(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the harness derived the command section through the helper")

    monkeypatch.setattr(task_fingerprint, "stage_command_manifest", no_helper)
    independent, lines, errors = _harness_stage_lines(model_id, "locomotion", config, tmp_path / "oracle")
    assert errors == 0 and independent["task_sha256"] == expected["task_sha256"], lines
    monkeypatch.undo()

    env = get_species_config(model_id).env_class(**config["env_kwargs"])
    try:
        assert env.command_manifest() is not None
        rewards = digest_snapshot._reward_lines(digest_snapshot._capture(env, digest_snapshot._stage_parts(env)), False)
    finally:
        env.close()
    names = ["summary", "poses", "shape_sha256", "rounded_values_sha256"]
    assert [line[:2] for line in rewards] == [("reward", name) for name in names]


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_train_fingerprints_trains_and_records_a_live_stage(tmp_path, algorithm):
    """Both train_base call sites derive a live stage's fingerprint; the run trains a few hundred steps and records
    the command section and the whole command config."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from environments.shared import train_base
    from environments.shared.config import load_all_stages

    configs = load_all_stages(COMPSOGNATHUS_LONGIPES)
    key = next(ref for ref in configs if load_stage_manifest(COMPSOGNATHUS_LONGIPES).resolve(ref).id == "locomotion")
    live = _live_config(tmp_path / "configs", COMPSOGNATHUS_LONGIPES)
    live["env_kwargs"]["max_episode_steps"] = 64
    if algorithm == "ppo":
        live["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=1)
    else:
        live["sac_kwargs"].update(learning_starts=64, batch_size=32, buffer_size=1_000)
    configs[key] = live
    output = tmp_path / "run"
    model = train_base.train(
        get_species_config(COMPSOGNATHUS_LONGIPES),
        configs,
        key,
        total_timesteps=128,
        n_envs=1,
        seed=3,
        eval_freq=1_000,
        save_freq=1_000,
        verbose=0,
        algorithm=algorithm,
        output_dir=str(output),
        use_tensorboard=False,
        post_eval_episodes=0,
        report_metrics=False,
        save_on_interrupt=False,
    )
    assert model.num_timesteps >= 128
    expected = stage_task_fingerprint(COMPSOGNATHUS_LONGIPES, "locomotion", stage_config=live)
    assert "command" in expected
    assert model.mesozoic_task_fingerprint == expected
    (fingerprint_path,) = output.rglob("task_fingerprint.json")
    assert json.loads(fingerprint_path.read_text()) == json.loads(json.dumps(expected))
    (record_path,) = output.rglob("stage_config.json")
    recorded = json.loads(record_path.read_text())["reward_weights"]
    assert DirectionCommandConfig(**recorded["command_config"]) == live["env_kwargs"]["command_config"]


# ---------------------------------------------------------------------------
# (c) Tracking in the base env
# ---------------------------------------------------------------------------


def _numerics_mismatch(fixture: dict[str, Any]) -> list[str]:
    recorded, running = fixture["numerics_environment"], pilot_rollout_golden.numerics_environment()
    return [key for key in sorted(set(recorded) | set(running)) if recorded.get(key) != running.get(key)]


GOLDEN = pilot_rollout_golden.load_fixture()
PILOT_IDS = [pilot_rollout_golden.pilot_key(*spec[:3]) for spec in pilot_rollout_golden.PILOT_SPECS]


@functools.lru_cache(maxsize=None)
def _pilot_capture(spec: tuple[Any, ...]) -> dict[str, Any]:
    """One pilot's capture on this tree, shared by the exact and the portable replay."""
    return pilot_rollout_golden.capture_pilot(*spec)


@pytest.mark.parametrize("spec", pilot_rollout_golden.PILOT_SPECS, ids=PILOT_IDS)
def test_the_pilot_recipes_reproduce_the_rollouts_captured_before_the_move(spec):
    """Every reward, info dict, observation, command, flag and reset state, exactly, on the same seeds.

    The fixture was captured at the commit before the tracking code moved into BaseDinoEnv; its digests are
    exact only in the numerics environment they were taken in (pilot_rollout_golden.py), so elsewhere this skips
    and the portable replay below is the guard.
    """
    if mismatch := _numerics_mismatch(GOLDEN):
        pytest.skip(f"the golden's numerics environment differs here in {mismatch}; its digests are exact only there")
    key = pilot_rollout_golden.pilot_key(*spec[:3])
    expected, actual = GOLDEN["captures"][key], _pilot_capture(spec)
    assert actual["task_sha256"] == expected["task_sha256"], "the recipe's task moved"
    for index, (want, got) in enumerate(zip(expected["episodes"], actual["episodes"], strict=True)):
        assert (got["length"], got["ended_by"]) == (want["length"], want["ended_by"]), index
        assert got["reset_sha256"] == want["reset_sha256"], f"episode {index}: the reset moved"
        for step, (reward, wanted) in enumerate(zip(got["rewards"], want["rewards"], strict=True)):
            assert reward == wanted, f"episode {index} step {step}: reward {reward!r} != {wanted!r}"
        for step, (digest, wanted) in enumerate(zip(got["step_sha256"], want["step_sha256"], strict=True)):
            assert digest == wanted, f"episode {index} step {step}: the info, observation or command moved"
        assert got["episode_sha256"] == want["episode_sha256"]
        assert got["portable"] == want["portable"], f"episode {index}: the portable rows, exactly here"


@pytest.mark.parametrize("spec", pilot_rollout_golden.PILOT_SPECS, ids=PILOT_IDS)
def test_the_pilot_recipes_reproduce_their_first_steps_on_any_machine(spec):
    """The portable companion of the exact replay: each episode's first steps, on any Python, C library, CPU and
    numpy 2.x, with the fixture's MuJoCo (the repository pins it).

    Per step: the reward, both flags and the info's key order, and the tracking reward and total, the dwell, the
    command state and the tracking metrics; floats within a relative 1e-9, the rest exactly.  The task digest is
    compared too (it hashes the recipe's config, not a rollout).
    """
    recorded = pilot_rollout_golden.portable_environment(GOLDEN["numerics_environment"])
    running = pilot_rollout_golden.portable_environment()
    if mismatch := sorted(key for key in recorded if recorded[key] != running[key]):
        pytest.skip(f"the golden's {mismatch} differ here ({recorded} vs {running}); a step moves beyond tolerance")
    key = pilot_rollout_golden.pilot_key(*spec[:3])
    expected, actual = GOLDEN["captures"][key], _pilot_capture(spec)
    assert actual["task_sha256"] == expected["task_sha256"], "the recipe's task moved"
    for index, (want, got) in enumerate(zip(expected["episodes"], actual["episodes"], strict=True)):
        differences = pilot_rollout_golden.portable_differences(want["portable"], got["portable"])
        assert not differences, f"episode {index}: {differences[:5]}"


def test_the_portable_comparison_tolerates_the_last_bits_and_nothing_more():
    """The tolerance passes a last-bit float change and refuses a semantic one, a type change, a flipped flag and a
    reordered info."""
    (row, *_) = GOLDEN["captures"][PILOT_IDS[0]]["episodes"][0]["portable"]
    columns = pilot_rollout_golden.PORTABLE_COLUMNS
    tracking = columns.index("reward_tracking")
    error_v = columns.index("tracking_error_v")

    def changed(column: int, value: Any) -> list[list[Any]]:
        return [[value if index == column else item for index, item in enumerate(row)]]

    assert pilot_rollout_golden.portable_differences([row], [list(row)]) == []
    assert pilot_rollout_golden.portable_differences([row], changed(error_v, math.nextafter(row[error_v], 9.0))) == []
    assert pilot_rollout_golden.portable_differences([row], changed(error_v, row[error_v] * (1 + 1e-7)))
    assert pilot_rollout_golden.portable_differences([row], changed(tracking, row[tracking] * 2.0 + 1e-6))
    assert pilot_rollout_golden.portable_differences([row], changed(columns.index("terminated"), True))
    assert pilot_rollout_golden.portable_differences([row], changed(columns.index("info_keys_sha256"), "0" * 16))
    assert pilot_rollout_golden.portable_differences([row], changed(columns.index("command_event_id"), 0.0))
    assert pilot_rollout_golden.portable_differences([row], [])


def test_the_golden_was_captured_before_the_move():
    assert GOLDEN["schema"] == pilot_rollout_golden.FIXTURE_SCHEMA and GOLDEN["source_commit"] == "769a657"
    assert set(GOLDEN["captures"]) == set(PILOT_IDS)
    assert GOLDEN["portable_columns"] == list(pilot_rollout_golden.PORTABLE_COLUMNS)
    for key, capture in GOLDEN["captures"].items():
        for episode in capture["episodes"]:
            assert len(episode["portable"]) == min(pilot_rollout_golden.PORTABLE_STEPS, episode["length"]), key


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_the_replay_holds_only_the_walk_stand_term_settings_at_their_capture_values(model_id):
    """The keys the replay builds as at the capture (``WALK_STAND_SETTINGS_SINCE_CAPTURE``) are exactly the walk
    ``[env]`` settings decision D-D62 copied from the stance for its stand block (each line cites
    ``# stance.toml:N``), and each reaches a pilot through ``read_recipe``.  Their terms are at weight 0 in the walk's
    ``[env]`` (test_walk_stand_block.py), so a pilot pays nothing for them; every other inherited key is as captured,
    which the replay's task digest proves."""
    from environments.shared.train_behaviors import read_recipe

    path = model_id.path(REPOSITORY_ROOT / "configs") / "locomotion.toml"
    cited, table = set(), None
    for line in path.read_text().splitlines():
        if line.startswith("["):
            table = line.strip()
        elif table == "[env]" and re.match(r"^\w+\s*=[^#]*#\s*stance\.toml:\d+", line):
            cited.add(line.split("=", 1)[0].strip())
    restored = pilot_rollout_golden.WALK_STAND_SETTINGS_SINCE_CAPTURE.get((model_id.species, model_id.embodiment), ())
    assert set(restored) == cited and len(restored) == len(cited)
    recipe = model_id.path(REPOSITORY_ROOT / "configs") / "behaviors" / "follow_direction.toml"
    _, _, _, env_kwargs = read_recipe(recipe, model_id)
    assert cited <= set(env_kwargs)


@pytest.mark.parametrize(
    ("model_id", "recipe"),
    [
        (ModelId("velociraptor_mongoliensis", "anatomical"), "follow_direction"),
        (ModelId("tyrannosaurus_rex", "anatomical"), "follow_direction_speed"),
        (ModelId("compsognathus_longipes", "anatomical"), "follow_direction"),
    ],
    ids=str,
)
def test_the_inherited_walk_stand_term_settings_move_no_recipe_reward(model_id, recipe):
    """D-D62 (14): a recipe built through ``read_recipe`` / ``create_behavior_env`` with the walk's stand-term
    settings and one built without them (as the pilot golden replays it) step identically: the same reward, the
    same observation and the same flags on every step of a seeded random-action rollout, falls and resets included.
    Only those terms' own diagnostics, at weight 0 in the recipe, may differ."""
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    path = model_id.path(REPOSITORY_ROOT / "configs") / "behaviors" / f"{recipe}.toml"
    parsed, commands, terrain, env_kwargs = read_recipe(path, model_id)
    settings = pilot_rollout_golden.WALK_STAND_SETTINGS_SINCE_CAPTURE[(model_id.species, model_id.embodiment)]
    assert set(settings) <= set(env_kwargs) and not set(settings) & set(parsed.get("env", {}))
    without = {key: value for key, value in env_kwargs.items() if key not in settings}
    envs = [
        create_behavior_env(model_id, commands=commands, terrain=terrain, run_seed=0, **kwargs)
        for kwargs in (env_kwargs, without)
    ]
    try:
        nu = int(np.prod(envs[0].action_space.shape))
        actions = np.random.default_rng(20261124).uniform(-0.3, 0.3, size=(150, nu)).astype(np.float32)
        observations = [env.reset(seed=3042)[0] for env in envs]
        assert np.array_equal(*observations)
        for action in actions:
            steps = [env.step(action) for env in envs]
            (obs_a, reward_a, term_a, trunc_a, _), (obs_b, reward_b, term_b, trunc_b, _) = steps
            assert reward_a == reward_b and (term_a, trunc_a) == (term_b, trunc_b)
            assert np.array_equal(obs_a, obs_b)
            if term_a or trunc_a:
                assert np.array_equal(*(env.reset()[0] for env in envs))
    finally:
        for env in envs:
            env.close()


TRACKING_KEYS = {"reward_tracking", "tracking_dwell_s", "tracking_in_tolerance", "command_event_id", "desired_heading"}


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_a_none_stage_scores_no_tracking(model_id):
    env = build_env(model_id, "stance")
    try:
        env.reset(seed=3)
        _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert TRACKING_KEYS.isdisjoint(info)
        assert not hasattr(env, "tracking_weight") and env._command_metrics == {}
    finally:
        env.close()


def _none_twin(model_id: ModelId, stage: str = "locomotion") -> BaseDinoEnv:
    """*stage*'s committed env under command_mode "none": the walk's [env] without its live command (decision D-D62),
    the "none" twin a live env is compared with."""
    env: BaseDinoEnv = build_env(model_id, stage, command_mode="none", command_config=None)
    return env


def _live_env(model_id: ModelId, stage: str = "locomotion", **config: Any) -> BaseDinoEnv:
    env: BaseDinoEnv = build_env(
        model_id, stage, command_mode="heading_and_speed", command_config=DirectionCommandConfig(**config)
    )
    return env


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_a_live_stage_scores_the_command_config_tracking_term(model_id):
    """reward = the species reward + tracking_weight * gaussian(errors; sigmas), the parameters the config's."""
    env = _live_env(model_id, tracking_weight=1.7, tracking_velocity_sigma=0.4, tracking_yaw_rate_sigma=0.2)
    plain = _none_twin(model_id)
    try:
        assert (env.tracking_weight, env.tracking_velocity_sigma, env.tracking_yaw_rate_sigma) == (1.7, 0.4, 0.2)
        env.reset(seed=3)
        plain.reset(seed=3)
        action = np.full(env.action_space.shape, 0.05, dtype=np.float32)
        for _ in range(5):
            _, reward, terminated, truncated, info = env.step(action)
            _, plain_reward, _, _, _ = plain.step(action)
            expected = 1.7 * gaussian_tracking_reward(
                info["tracking_error_v"], info["tracking_error_yaw"], velocity_sigma=0.4, yaw_rate_sigma=0.2
            )
            assert info["reward_tracking"] == pytest.approx(expected, rel=1e-12)
            assert reward == pytest.approx(plain_reward + expected, rel=1e-12, abs=1e-12)
            assert info["reward_total"] == reward and TRACKING_KEYS <= set(info)
            if terminated or truncated:
                break
    finally:
        env.close()
        plain.close()


def test_the_dwell_counts_seconds_in_tolerance_and_restarts_on_a_new_target():
    """With tolerances no motion can leave, the dwell grows by dt per step; set_direction restarts it."""
    wide = dict(
        tracking_velocity_tolerance=100.0,
        tracking_yaw_rate_tolerance=100.0,
        tracking_heading_tolerance=math.pi,
        switch_interval_s=100.0,
    )
    env = _live_env(TYRANNOSAURUS_REX, **wide)
    try:
        env.reset(seed=4)
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        dwell = [env.step(zero)[4]["tracking_dwell_s"] for _ in range(4)]
        assert dwell == pytest.approx([env.dt * (index + 1) for index in range(4)])
        observation = env.set_direction(1.0, 0.5)
        assert env._tracking_dwell_s == 0.0 and env._command_state.target_source == "external"
        np.testing.assert_array_equal(observation[-COMMAND_WIDTH:], env._command_state.normalized)
        assert env.step(zero)[4]["tracking_dwell_s"] == pytest.approx(env.dt)
    finally:
        env.close()


def test_set_direction_and_step_need_a_live_command_and_a_reset():
    plain = build_env(TYRANNOSAURUS_REX, "stance")
    live = _live_env(TYRANNOSAURUS_REX)
    try:
        with pytest.raises(RuntimeError, match="live command_mode"):
            plain.set_direction(0.0, 0.5)
        with pytest.raises(RuntimeError, match="reset must be called before step"):
            live.step(np.zeros(live.action_space.shape, dtype=np.float32))
    finally:
        plain.close()
        live.close()


def test_a_behavior_env_keeps_its_tracking_weight_kwarg():
    from environments.shared.behavior_env import get_behavior_env_class

    env_class = get_behavior_env_class(TYRANNOSAURUS_REX)
    with pytest.raises(ValueError, match="weights tracking by its tracking_weight kwarg"):
        env_class(commands=DirectionCommandConfig(tracking_weight=3.0))
    env = env_class(tracking_weight=3.0, commands=DirectionCommandConfig(tracking_velocity_sigma=0.3))
    try:
        assert env.tracking_weight == 3.0 and env.tracking_velocity_sigma == 0.3
        assert env.task_fingerprint["env"]["tracking_weight"] == 3.0
        assert env.task_fingerprint["command"]["config"]["tracking_velocity_sigma"] == 0.3
    finally:
        env.close()


# ---------------------------------------------------------------------------
# (d) The command stream
# ---------------------------------------------------------------------------


def _state(env: BaseDinoEnv) -> tuple[Any, ...]:
    starts, directions = env._push_schedule_starts, env._push_schedule_directions
    return (
        env.data.qpos.copy(),
        env.data.qvel.copy(),
        None if starts is None else np.array(starts),
        None if directions is None else np.array(directions),
        env.np_random.bit_generator.state,
    )


def _assert_same(first: tuple[Any, ...], second: tuple[Any, ...], where: str) -> None:
    for name, left, right in zip(("qpos", "qvel", "push starts", "push directions", "rng state"), first, second):
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=f"{where}: {name}")
        else:
            assert left == right, f"{where}: {name}"


@pytest.mark.parametrize(
    ("model_id", "stage"),
    [
        (TYRANNOSAURUS_REX, "recovery"),
        (ModelId("compsognathus_longipes", "robot"), "recovery"),
        (ModelId("velociraptor_mongoliensis", "anatomical"), "locomotion"),
        (ModelId("brachiosaurus_altithorax", "anatomical"), "locomotion"),
    ],
    ids=lambda value: str(value),
)
def test_a_live_command_leaves_every_reset_draw_as_none_draws_it(model_id, stage):
    """A seeded reset, then unseeded resets with zero-action steps between: the same qpos, qvel, push schedule and
    generator state as the "none" twin, and the same observation but for the command; the commands vary."""
    live = _live_env(model_id, stage, stop_probability=0.2, speed_range=(0.1, 0.4), initial_heading_offset_max=0.3)
    none = _none_twin(model_id, stage)
    zero = np.zeros(none.action_space.shape, dtype=np.float32)
    commands = []
    try:
        for seed in (3042, 7):
            for index in range(4):
                where = f"seed {seed} reset {index}"
                live_obs, _ = live.reset(seed=seed if index == 0 else None)
                none_obs, _ = none.reset(seed=seed if index == 0 else None)
                _assert_same(_state(live), _state(none), where)
                np.testing.assert_array_equal(live_obs[:-COMMAND_WIDTH], none_obs[:-COMMAND_WIDTH], err_msg=where)
                np.testing.assert_array_equal(none_obs[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH, dtype=np.float32))
                commands.append(live_obs[-COMMAND_WIDTH:].copy())
                for _ in range(10):
                    live.step(zero)
                    none.step(zero)
                np.testing.assert_array_equal(live.data.qpos, none.data.qpos, err_msg=where)
        assert len({tuple(command) for command in commands}) > 1
    finally:
        live.close()
        none.close()


def test_the_command_stream_restarts_with_each_seed_and_advances_per_episode():
    env = _live_env(TYRANNOSAURUS_REX)
    try:
        seeds = []
        for seed in (11, 11, 12):
            env.reset(seed=seed)
            episode = [env.direction_controller.schedule_seed]
            for _ in range(2):
                env.reset()
                episode.append(env.direction_controller.schedule_seed)
            seeds.append(episode)
        assert seeds[0] == seeds[1] != seeds[2] and len(set(seeds[0])) == 3
        for seed, episode in zip((11, 12), (seeds[0], seeds[2])):
            for index, schedule_seed in enumerate(episode):
                stream = np.random.default_rng(np.random.SeedSequence([seed, index, COMMAND_STREAM_DOMAIN]))
                assert schedule_seed == int(stream.integers(0, 2**32, dtype=np.uint64))
        env.np_random = np.random.default_rng(5)
        with pytest.raises(RuntimeError, match="derives from the reset seed"):
            env.reset()
    finally:
        env.close()


def test_a_live_command_keeps_the_phase_c_reset_golden():
    """Each golden capture's reset and second-reset halves (the seeded draw stream) re-take under a live command."""
    fixture = reset_golden.load_fixture()
    config = DirectionCommandConfig(stop_probability=0.2, speed_range=(0.1, 0.4), initial_heading_offset_max=0.3)
    for model_id, stage, seeds, steps in reset_golden.DEFAULT_SPEC:
        stored = fixture["captures"][model_id.species][model_id.embodiment][stage]
        env = build_env(model_id, stage, command_mode="heading_and_speed", command_config=config)
        try:
            zero = np.zeros(env.action_space.shape, dtype=np.float32)
            for seed in seeds:
                env.reset(seed=int(seed))
                assert reset_golden._reset_record(env) == stored["seeds"][str(seed)]["reset"], (model_id, stage, seed)
                for _ in range(steps):
                    env.step(zero)
                env.reset()
                second = reset_golden._reset_record(env)
                assert second == stored["seeds"][str(seed)]["second_reset"], (model_id, stage, seed)
        finally:
            env.close()


# ---------------------------------------------------------------------------
# (e) initial_heading_offset_max and the record of the new fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-0.1, math.pi + 1e-9, math.nan, math.inf])
def test_the_initial_heading_offset_is_validated(value):
    with pytest.raises(ValueError, match="initial_heading_offset_max"):
        DirectionCommandConfig(initial_heading_offset_max=value)
    assert DirectionCommandConfig(initial_heading_offset_max=math.pi).initial_heading_offset_max == math.pi


def test_a_zero_offset_draws_nothing_and_a_positive_one_turns_the_first_request():
    still = DirectionCommandConfig(turn_increment_max=0.0, switch_interval_s=100.0)
    turned = replace(still, initial_heading_offset_max=0.26)
    heading = 0.4
    passed = {}
    for name, config in (("still", still), ("turned", turned)):
        controller = DirectionCommandController(config)
        rng = np.random.default_rng(9)
        state = controller.reset(rng, heading)
        passed[name] = (controller, state, rng.bit_generator.state)
    controller, state, rng_state = passed["still"]
    # Cruise speed, no jitter, no offset: the controller's own stream is untouched at reset.
    assert controller._rng.bit_generator.state == np.random.PCG64(controller.schedule_seed).state
    assert state.desired_heading == wrap_angle(heading) and state.heading_error == 0.0 and state.physical[2] == 0.0
    turned_controller, turned_state, turned_rng_state = passed["turned"]
    assert turned_rng_state == rng_state, "the passed generator gives one schedule seed either way"
    offset = float(np.random.Generator(np.random.PCG64(turned_controller.schedule_seed)).uniform(-0.26, 0.26))
    assert turned_state.desired_heading == wrap_angle(heading + offset)
    assert turned_controller.events[0]["desired_heading"] == turned_state.desired_heading
    assert turned_state.heading_error == pytest.approx(offset)
    assert np.sign(turned_state.physical[2]) == np.sign(offset)


def test_the_new_fields_are_recorded_only_when_set():
    default = DirectionCommandConfig()
    assert tuple(default.task_record()) == PRE_D_D60_FIELDS
    assert set(DirectionCommandConfig.UNRECORDED_VALUES) == {
        field.name for field in fields(DirectionCommandConfig)
    } - set(PRE_D_D60_FIELDS)
    for name, value in DirectionCommandConfig.UNRECORDED_VALUES.items():
        assert getattr(default, name) == value, name
    assert DirectionCommandController(default).manifest()["config"] == default.task_record()
    for name, value in (
        ("initial_heading_offset_max", 0.2),
        ("tracking_weight", 2.0),
        ("tracking_velocity_sigma", 0.3),
    ):
        record = replace(default, **{name: value}).task_record()
        assert record == {**default.task_record(), name: value}


def test_the_tracking_parameters_scale_with_the_speed_scale_unless_stated():
    config = DirectionCommandConfig(speed_scale=0.6, cruise_speed=0.3)
    ratio = 0.6 / 1.5
    assert config.tracking_parameters() == {
        "weight": 2.5,
        "velocity_sigma": 0.25 * ratio,
        "yaw_rate_sigma": 0.15,
        "tolerances": {
            "velocity_tolerance": 0.2 * ratio,
            "yaw_rate_tolerance": 0.15,
            "heading_tolerance": math.pi / 12,
            "stop_speed_tolerance": 0.1 * ratio,
            "stop_yaw_rate_tolerance": 0.15,
        },
    }
    stated = replace(config, tracking_velocity_sigma=0.05, tracking_stop_speed_tolerance=0.02)
    assert stated.tracking_parameters()["velocity_sigma"] == 0.05
    assert stated.tracking_parameters()["tolerances"]["stop_speed_tolerance"] == 0.02
    for name in ("tracking_velocity_sigma", "tracking_yaw_rate_sigma", "tracking_heading_tolerance"):
        with pytest.raises(ValueError, match=name):
            DirectionCommandConfig(**{name: -0.1})


def test_a_live_env_runs_the_offset_from_its_stage_config(tmp_path):
    """The [commands] offset reaches the env: the first request is turned from the spawn heading by at most it."""
    config = _live_config(tmp_path, TYRANNOSAURUS_REX)
    env = get_species_config(TYRANNOSAURUS_REX).env_class(**config["env_kwargs"])
    try:
        offsets = []
        for seed in range(6):
            env.reset(seed=seed)
            state = env._command_state
            if state.heading_active:
                offsets.append(wrap_angle(state.desired_heading - env._heading()))
        assert offsets and all(abs(offset) <= 0.26 for offset in offsets) and len(set(offsets)) == len(offsets)
    finally:
        env.close()
