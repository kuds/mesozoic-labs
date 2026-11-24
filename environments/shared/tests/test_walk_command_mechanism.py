"""The live walk's shared mechanism (decision D-D62), each inert under command_mode "none".

* **The forward reference**: under a live command the forward, heading and
  lateral terms read the commanded heading (``_forward_reference_2d``) and
  the drift terms the command event's start (``_drift_anchor_2d``); under
  "none" both are the spawn's own arrays.
* **The stand block**: ``[commands.stand_weights]`` becomes the command
  config's ``stand_weights``, rides in the command manifest and the task
  fingerprint (live stages only), turns opt-in task options on, and is held
  over every step taken under an exactly zero live command, never at a small
  non-zero one and never under "none".  T. rex's three standing terms are
  stand-block terms like every other species'.
* **The idle penalty** is not paid at a live zero command.
* **The Allosaurus course** runs along the commanded heading at reset under a
  live command, is an event rather than a termination, and a stand episode
  has none.
* **The cruise ramp**: ``cruise_ramp_start_fraction`` / ``cruise_ramp_timesteps``
  scale the commanded speeds on the training workers only, keep the stand
  draw, record their position in every checkpoint and resume from it.
* **The committed walks**: on every model's ``locomotion.toml`` as committed,
  the forward reference is the reset's commanded heading and follows a later
  request, the whole stand block is in force exactly at a zero command, the
  Allosaurus course runs along each reset heading, and a same-stage resume
  continues the cruise ramp from its checkpoint's position (PPO and SAC).
  The warm start from a stance, command-blind at entry and reading the
  command after training, is ``test_walk_species_training.py``'s.
* **The digest harness** scores a live stage's state probes as ``step()``
  does.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.command_frame import COMMAND_STREAM_DOMAIN
from environments.shared.config import build_env, load_stage_config, save_stage_config
from environments.shared.curriculum.cruise_ramp import (
    CRUISE_RAMP_UPDATE_STEPS,
    MODEL_CRUISE_RAMP_ATTRIBUTE,
    CruiseRampError,
    cruise_ramp_fraction,
    cruise_ramp_record,
    ramped_command_config,
    resume_ramp_steps,
)
from environments.shared.direction_commands import (
    DirectionCommandConfig,
    DirectionCommandController,
    stand_weights_of,
)
from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import inert_opt_in_env_kwargs, stage_task_fingerprint
from environments.shared.tests.stage_toml_helpers import replace_commands_table

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
ALLOSAURUS_FRAGILIS = ModelId("allosaurus_fragilis", "anatomical")
MODEL_IDS = model_ids(backend="stable-baselines3")

#: A walk-like command with stops drawn often enough to find stand seeds fast.
WALK: dict[str, Any] = dict(
    cruise_speed=0.5,
    speed_range=(0.25, 0.5),
    stop_probability=0.4,
    turn_increment_max=0.0,
    switch_interval_s=1000.0,
    initial_heading_offset_max=0.26,
)


def _family(config: DirectionCommandConfig, seed: int) -> str:
    """The reset draw of *seed* under *config*: "stand" (an exactly zero command) or "cruise"."""
    from environments.shared.command_tracking_panel import reset_command_family

    return reset_command_family(config, seed)


def _reset_state(config: DirectionCommandConfig, seed: int) -> Any:
    """The command state ``reset(seed=seed)`` draws, replayed from the env's command stream (spawn heading 0)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0, COMMAND_STREAM_DOMAIN]))
    return DirectionCommandController(config).reset(rng, 0.0)


def _seed(config: DirectionCommandConfig, family: str, start: int = 3042) -> int:
    return next(seed for seed in range(start, start + 500) if _family(config, seed) == family)


def _live(model_id: ModelId, stage: str = "locomotion", env: dict[str, Any] | None = None, **config: Any) -> Any:
    command = DirectionCommandConfig(**{**WALK, **config})
    return build_env(model_id, stage, command_mode="heading_and_speed", command_config=command, **(env or {}))


def _zero(env: Any) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# The command config's new fields
# ---------------------------------------------------------------------------


def test_stand_weights_are_held_as_sorted_pairs_and_recorded_as_an_object():
    config = DirectionCommandConfig(stand_weights={"posture_weight": 1, "drift_penalty_weight": 0.3})
    assert config.stand_weights == (("drift_penalty_weight", 0.3), ("posture_weight", 1.0))
    assert isinstance(config.stand_weights[1][1], float) and hash(config) is not None
    assert config == DirectionCommandConfig(stand_weights=[("posture_weight", 1.0), ("drift_penalty_weight", 0.3)])
    assert config.task_record()["stand_weights"] == {"drift_penalty_weight": 0.3, "posture_weight": 1.0}
    assert config.field_record()["stand_weights"] == {"drift_penalty_weight": 0.3, "posture_weight": 1.0}
    assert DirectionCommandConfig(**json.loads(json.dumps(config.field_record()))) == config
    assert stand_weights_of(config) == stand_weights_of(config.field_record()) == config.stand_weight_map()
    assert stand_weights_of(None) == {} and stand_weights_of({"cruise_speed": 1.0}) == {}


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"posture_weight": -0.1}, "at least 0"),
        ({"posture_weight": True}, "must be a number"),
        ({"posture_weight": math.inf}, "finite"),
        ({"not a name": 1.0}, "reward kwarg name"),
        ([("posture_weight", 1.0), ("posture_weight", 2.0)], "twice"),
        (["posture_weight"], "table of reward weights"),
    ],
)
def test_stand_weights_are_validated(value, match):
    with pytest.raises(ValueError, match=match):
        DirectionCommandConfig(stand_weights=value)


@pytest.mark.parametrize(
    ("fraction", "timesteps", "match"),
    [
        (0.25, 0, "needs both"),
        (1.0, 1000, "needs both"),
        (0.0, 1000, "lie in \\(0, 1\\]"),
        (1.5, 1000, "lie in \\(0, 1\\]"),
        (0.25, -1, "whole number"),
        (0.25, 2.5, "whole number"),
        (0.25, True, "whole number"),
    ],
)
def test_the_cruise_ramp_fields_are_both_on_or_both_off(fraction, timesteps, match):
    with pytest.raises(ValueError, match=match):
        DirectionCommandConfig(cruise_ramp_start_fraction=fraction, cruise_ramp_timesteps=timesteps)


def test_the_new_fields_are_recorded_only_when_set():
    """A config at the new fields' unrecorded values records what it recorded before decision D-D62."""
    base = DirectionCommandConfig(**WALK)
    record = base.task_record()
    assert not {"stand_weights", "cruise_ramp_start_fraction", "cruise_ramp_timesteps"} & set(record)
    walk = DirectionCommandConfig(
        **WALK,
        stand_weights={"drift_penalty_weight": 0.3},
        cruise_ramp_start_fraction=0.25,
        cruise_ramp_timesteps=2_000_000,
    )
    assert walk.task_record() == {
        **record,
        "stand_weights": {"drift_penalty_weight": 0.3},
        "cruise_ramp_start_fraction": 0.25,
        "cruise_ramp_timesteps": 2_000_000,
    }


def test_a_ramped_config_scales_only_the_commanded_speeds():
    config = DirectionCommandConfig(**WALK, stand_weights={"drift_penalty_weight": 0.3})
    ramped = config.ramped(0.25)
    assert (ramped.cruise_speed, ramped.speed_range) == (0.125, (0.0625, 0.125))
    assert replace(ramped, cruise_speed=config.cruise_speed, speed_range=config.speed_range) == config
    assert ramped.tracking_parameters() == config.tracking_parameters()
    assert config.ramped(1.0) == config
    with pytest.raises(ValueError, match="lie in"):
        config.ramped(0.0)


def test_the_ramp_keeps_every_stand_draw_and_scales_every_cruise_speed():
    """The stop is drawn before the speed, so the ramp moves no seed between the stand and cruise families."""
    config = DirectionCommandConfig(**WALK)
    for fraction in (0.25, 0.6):
        ramped = config.ramped(fraction)
        for seed in range(3042, 3242):
            rng = np.random.default_rng(seed)
            end = DirectionCommandController(config).reset(np.random.default_rng(seed), 0.0)
            start = DirectionCommandController(ramped).reset(rng, 0.0)
            assert start.desired_heading == end.desired_heading
            assert (start.desired_speed == 0.0) == (end.desired_speed == 0.0)
            assert start.desired_speed == pytest.approx(fraction * end.desired_speed, rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# [commands] in a stage TOML, the fingerprint and the record
# ---------------------------------------------------------------------------

WALK_TABLE = """
[commands]
mode = "heading_and_speed"
cruise_speed = 0.5
speed_range = [0.25, 0.5]
stop_probability = 0.15
turn_increment_max = 0.0
switch_interval_s = 100.0
initial_heading_offset_max = 0.26
cruise_ramp_start_fraction = 0.25
cruise_ramp_timesteps = 2000000

[commands.stand_weights]
drift_penalty_weight = 0.3
lateral_lean_weight = 0.25
"""


def _walk_stage(tmp_path: Path, model_id: ModelId = TYRANNOSAURUS_REX, table: str = WALK_TABLE) -> Path:
    """A copy of *model_id*'s config directory whose locomotion.toml declares *table* in place of its own [commands]
    (and no forward ramp)."""
    directory = tmp_path / model_id.species / model_id.embodiment
    shutil.copytree(model_id.path(REPOSITORY_ROOT / "configs"), directory)
    path = directory / load_stage_manifest(model_id).by_id("locomotion").config_file
    return replace_commands_table(path, table)


def test_a_commands_table_declares_the_stand_block_and_the_ramp(tmp_path):
    path = _walk_stage(tmp_path)
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion", config_path=str(path))
    command = config["env_kwargs"]["command_config"]
    assert command.stand_weight_map() == {"drift_penalty_weight": 0.3, "lateral_lean_weight": 0.25}
    assert (command.cruise_ramp_start_fraction, command.cruise_ramp_timesteps) == (0.25, 2_000_000)
    from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

    record = save_stage_config(
        tmp_path / "run",
        "locomotion",
        config,
        "PPO",
        env_class=TyrannosaurusRexAnatomicalEnv,
        model_id=TYRANNOSAURUS_REX,
    )
    recorded = json.loads(Path(record).read_text())["reward_weights"]["command_config"]
    assert recorded["stand_weights"] == {"drift_penalty_weight": 0.3, "lateral_lean_weight": 0.25}
    assert DirectionCommandConfig(**recorded) == command


def test_a_live_stage_refuses_the_forward_velocity_ramp_keys(tmp_path):
    path = _walk_stage(tmp_path)
    path.write_text(path.read_text().replace("[curriculum]\n", "[curriculum]\nramp_timesteps = 2000000\n", 1))
    with pytest.raises(ValueError, match="trains no forward-velocity ramp"):
        load_stage_config(TYRANNOSAURUS_REX, "locomotion", config_path=str(path))


def test_extends_inherits_the_stand_block_whole_unless_the_child_states_its_own(tmp_path):
    path = _walk_stage(tmp_path)
    child = path.parent / "recovery.toml"
    text = child.read_text().replace(
        'extends = { stage = "stance", tables = ["stage", "env", "ppo"] }',
        'extends = { stage = "locomotion", tables = ["commands"] }',
    )
    child.write_text(
        text + "\n[commands]\nstop_probability = 0.3\ncruise_ramp_start_fraction = 1.0\ncruise_ramp_timesteps = 0\n"
    )
    inherited = load_stage_config(TYRANNOSAURUS_REX, "recovery", config_path=str(child))["env_kwargs"]["command_config"]
    assert inherited.stand_weight_map() == {"drift_penalty_weight": 0.3, "lateral_lean_weight": 0.25}
    assert not inherited.has_cruise_ramp
    child.write_text(child.read_text() + "\n[commands.stand_weights]\nposture_weight = 1.0\n")
    replaced = load_stage_config(TYRANNOSAURUS_REX, "recovery", config_path=str(child))["env_kwargs"]["command_config"]
    assert replaced.stand_weight_map() == {"posture_weight": 1.0}


def test_the_stand_block_and_the_ramp_are_part_of_a_live_task_only(tmp_path):
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion", config_path=str(_walk_stage(tmp_path)))
    fingerprint = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", stage_config=config)
    recorded = fingerprint["command"]["config"]
    assert recorded["stand_weights"] == {"drift_penalty_weight": 0.3, "lateral_lean_weight": 0.25}
    assert (recorded["cruise_ramp_start_fraction"], recorded["cruise_ramp_timesteps"]) == (0.25, 2_000_000)
    command = config["env_kwargs"]["command_config"]
    moved = []
    for change in (
        {"stand_weights": {"drift_penalty_weight": 0.31, "lateral_lean_weight": 0.25}},
        {"cruise_ramp_timesteps": 1_000_000},
        {"cruise_ramp_start_fraction": 0.3},
    ):
        other = {**config, "env_kwargs": {**config["env_kwargs"], "command_config": replace(command, **change)}}
        moved.append(stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", stage_config=other)["task_sha256"])
    assert len({fingerprint["task_sha256"], *moved}) == 4


def test_a_stand_weight_turns_an_opt_in_option_on_and_hashes_its_kwargs():
    """T. rex's lateral lean is off in a walk's [env]; paid by the stand block, its tolerance is hashed too."""
    from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv as Env

    off = DirectionCommandConfig(**WALK)
    on = DirectionCommandConfig(**WALK, stand_weights={"lateral_lean_weight": 0.25})
    carved = inert_opt_in_env_kwargs(Env, {"command_mode": "heading_and_speed", "command_config": off})
    assert {"lateral_lean_weight", "lateral_lean_tolerance_m"} <= set(carved)
    kept = inert_opt_in_env_kwargs(Env, {"command_mode": "heading_and_speed", "command_config": on})
    assert not {"lateral_lean_weight", "lateral_lean_tolerance_m"} & set(kept)
    assert inert_opt_in_env_kwargs(Env, {"command_config": on.field_record()}) == kept
    configs = [
        {"env_kwargs": {"command_mode": "heading_and_speed", "command_config": on, "lateral_lean_tolerance_m": tol}}
        for tol in (0.015, 0.02)
    ]
    digests = {stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", stage_config=c)["task_sha256"] for c in configs}
    assert len(digests) == 2


# ---------------------------------------------------------------------------
# The forward reference and the drift anchor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_a_none_stage_keeps_the_spawn_anchors_themselves(model_id):
    env = build_env(model_id, "stance")
    try:
        env.reset(seed=3042)
        assert env._forward_reference_2d() is env._reference_heading_2d
        assert env._drift_anchor_2d() is env._initial_pos_2d
        assert not env._live_zero_command() and not np.any(env._command), "the command is zero, yet not live"
        with env._stand_block() as held:
            assert held == {}
    finally:
        env.close()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_a_live_stage_reads_forward_along_the_commanded_heading(model_id):
    env = _live(model_id)
    try:
        seed = _seed(env.command_config, "cruise")
        env.reset(seed=seed)
        heading = env._command_state.desired_heading
        spawn = math.atan2(env._reference_heading_2d[1], env._reference_heading_2d[0])
        assert abs(heading - spawn) > 1e-3, "the initial heading offset turned the command off the spawn"
        np.testing.assert_allclose(env._forward_reference_2d(), [math.cos(heading), math.sin(heading)], atol=0)
        np.testing.assert_array_equal(env._drift_anchor_2d(), env._initial_pos_2d)
        _, _, _, _, info = env.step(_zero(env))
        expected = float(np.dot(env.data.qvel[0:2], [math.cos(heading), math.sin(heading)]))
        assert info["forward_vel"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    finally:
        env.close()


def test_a_new_command_event_moves_the_drift_anchor_to_where_it_began():
    env = _live(COMPSOGNATHUS_LONGIPES)
    try:
        env.reset(seed=_seed(env.command_config, "cruise"))
        for _ in range(20):
            env.step(np.full(env.action_space.shape, 0.05, dtype=np.float32))
        here = env.data.qpos[0:2].copy()
        assert np.linalg.norm(here - env._initial_pos_2d) > 0.0
        env.set_direction(0.3, 0.0)
        np.testing.assert_array_equal(env._drift_anchor_2d(), here)
        _, _, _, _, info = env.step(_zero(env))
        assert info["drift_distance"] == pytest.approx(float(np.linalg.norm(env.data.xpos[env.pelvis_id, :2] - here)))
    finally:
        env.close()


# ---------------------------------------------------------------------------
# The stand block
# ---------------------------------------------------------------------------

#: T. rex's three standing terms, stance-only kwargs a walk's [env] leaves at 0.
TYRANNOSAURUS_REX_STAND = {"lateral_lean_weight": 0.25, "stance_stagger_weight": 0.30, "stance_cop_band_weight": 0.30}


def _rewards_by_key(info: dict[str, Any]) -> set[str]:
    return {key for key in info if key.startswith("reward_")}


def test_the_stand_block_holds_at_a_live_zero_command_and_only_there():
    env = _live(TYRANNOSAURUS_REX, stand_weights={**TYRANNOSAURUS_REX_STAND, "drift_penalty_weight": 0.3})
    try:
        assert env._cop_band_reader is not None, "a stand-block CoP band builds its reader at construction"
        env.reset(seed=_seed(env.command_config, "stand"))
        assert env._live_zero_command()
        for _ in range(env.stance_width_settle_steps + 2):
            _, _, _, _, stand = env.step(_zero(env))
        assert {"reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band"} <= _rewards_by_key(stand)
        assert stand["reward_stance_stagger"] != 0.0 or stand["foot_stagger_m"] == 0.0
        assert stand["reward_drift"] == pytest.approx(-0.3 * (stand["drift_distance"] / 2.0) ** 2)
        assert (env.lateral_lean_weight, env.drift_penalty_weight) == (0.0, 0.0), "the constructor's weights are back"
        env.reset(seed=_seed(env.command_config, "cruise"))
        _, _, _, _, cruise = env.step(_zero(env))
        assert not {"reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band"} & _rewards_by_key(cruise)
        assert cruise["reward_drift"] == 0.0
    finally:
        env.close()


def test_the_tyrannosaurus_rex_standing_terms_no_longer_read_the_command():
    """Their zero-command gates became the stand block: a positive [env] weight pays them on a cruise step too."""
    env = _live(TYRANNOSAURUS_REX, env={"stance_stagger_weight": 0.3})
    try:
        env.reset(seed=_seed(env.command_config, "cruise"))
        assert not env._live_zero_command()
        for _ in range(env.stance_width_settle_steps + 1):
            _, _, terminated, _, info = env.step(_zero(env))
            assert not terminated
        assert info["reward_stance_stagger"] == pytest.approx(-0.3 * (1.0 - info["stance_stagger_quality"]))
    finally:
        env.close()


@pytest.mark.parametrize("speed", [1e-3, 0.0])
def test_the_stand_block_switches_exactly_at_zero(speed):
    """A small non-zero command is not a stand: the block holds at exactly 0 and nowhere else."""
    env = _live(VELOCIRAPTOR_MONGOLIENSIS, stand_weights={"drift_penalty_weight": 0.5, "tail_home_pose_weight": 0.2})
    try:
        env.reset(seed=_seed(env.command_config, "cruise"))
        env.set_direction(env._command_state.desired_heading, speed)
        assert env._live_zero_command() == (speed == 0.0)
        _, _, _, _, info = env.step(_zero(env))
        assert ("reward_tail_home_pose" in info) == (speed == 0.0)
        expected = -0.5 * (info["drift_distance"] / 2.0) ** 2 if speed == 0.0 else 0.0
        assert info["reward_drift"] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ({"posture_wieght": 1.0}, "not a numeric reward kwarg"),
        ({"reset_noise_scale": 0.01}, "swaps reward weights"),
        ({"forward_vel_max": 1.0}, "swaps reward weights"),
        ({"idle_penalty_weight": 0.2}, "never paid at a zero command"),
        ({"support_conditioned_alive_fraction": 1.5}, "must lie in \\[0, 1\\]"),
    ],
)
def test_the_env_refuses_a_stand_weight_it_cannot_swap(weights, match):
    with pytest.raises(ValueError, match=match):
        _live(TYRANNOSAURUS_REX, stand_weights=weights)


def test_a_stand_block_may_zero_the_idle_penalty_and_name_the_alive_terms():
    env = _live(VELOCIRAPTOR_MONGOLIENSIS, stand_weights={"idle_penalty_weight": 0.0, "alive_bonus": 1.75})
    try:
        assert env._stand_weights == {"alive_bonus": 1.75, "idle_penalty_weight": 0.0}
    finally:
        env.close()


@pytest.mark.parametrize("model_id", [BRACHIOSAURUS_ALTITHORAX, DIBOTHROSUCHUS_ELAPHROS], ids=str)
def test_the_idle_penalty_is_not_paid_at_a_live_zero_command(model_id):
    env = _live(model_id)
    try:
        assert env.idle_penalty_weight > 0.0
        env.reset(seed=_seed(env.command_config, "stand"))
        _, _, _, _, stand = env.step(_zero(env))
        assert stand["reward_idle"] == 0.0
        env.reset(seed=_seed(env.command_config, "cruise"))
        _, _, _, _, cruise = env.step(_zero(env))
        assert cruise["reward_idle"] < 0.0, "a statue on a cruise command still pays it"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# The Allosaurus course
# ---------------------------------------------------------------------------


def _displace(env: Any, distance: float, direction: np.ndarray) -> None:
    env.data.qpos[0:2] += distance * np.asarray(direction, dtype=np.float64)
    import mujoco

    mujoco.mj_forward(env.model, env.data)


def test_a_none_course_ends_the_episode_along_the_spawn_heading():
    """The forward walk before decision D-D62, under "none": the stance with the forward weight on."""
    env = build_env(ALLOSAURUS_FRAGILIS, "stance", forward_vel_weight=1.5, max_episode_steps=3000)
    try:
        assert env.forward_vel_weight > 0.0 and env.command_mode == "none"
        env.reset(seed=3042)
        _displace(env, env.COURSE_LENGTH_M, env._reference_heading_2d)
        _, _, terminated, _, info = env.step(_zero(env))
        assert terminated and info["course_complete"] == 1.0 and info["is_success"]
        assert info["termination_reason"] == "course_complete"
    finally:
        env.close()


def test_a_live_course_runs_along_the_commanded_heading_and_does_not_end_the_episode():
    env = _live(ALLOSAURUS_FRAGILIS, env={"forward_vel_weight": 0.0}, initial_heading_offset_max=0.26)
    try:
        assert env.max_episode_steps == 3000
        config = env.command_config
        seed = next(
            s
            for s in range(3042, 3542)
            if _family(config, s) == "cruise" and abs(_reset_state(config, s).desired_heading) > 0.2
        )
        env.reset(seed=seed)
        course = env._course_direction_2d()
        heading = env._reset_command_state.desired_heading
        np.testing.assert_allclose(course, [math.cos(heading), math.sin(heading)], atol=0)
        # Along the spawn heading the pelvis falls short of the commanded course's line.
        _displace(env, env.COURSE_LENGTH_M - env.COURSE_TOLERANCE_M + 0.1, env._reference_heading_2d)
        _, _, terminated, _, info = env.step(_zero(env))
        assert not terminated and info["course_complete"] == 0.0 and "success" not in info
        _displace(env, env.COURSE_LENGTH_M, course)
        _, _, terminated, truncated, info = env.step(_zero(env))
        assert not terminated and info["course_complete"] == 1.0 and info["success"] is True
        _, _, terminated, _, info = env.step(_zero(env))
        assert not terminated and info["course_complete"] == 0.0 and info["success"] is True
        env.max_episode_steps = env._step_count + 1
        _, _, _, truncated, info = env.step(_zero(env))
        assert truncated and info["is_success"]
    finally:
        env.close()


def test_a_stand_episode_has_no_course():
    env = _live(ALLOSAURUS_FRAGILIS, env={"forward_vel_weight": 0.0})
    try:
        env.reset(seed=_seed(env.command_config, "stand"))
        assert env._course_direction_2d() is None
        _displace(env, env.COURSE_LENGTH_M, env._reference_heading_2d)
        _, _, _, _, info = env.step(_zero(env))
        assert info["course_complete"] == 0.0 and "success" not in info
    finally:
        env.close()


# ---------------------------------------------------------------------------
# The cruise ramp
# ---------------------------------------------------------------------------

RAMP = DirectionCommandConfig(**WALK, cruise_ramp_start_fraction=0.25, cruise_ramp_timesteps=100_000)


def test_the_ramp_fraction_rises_linearly_in_held_blocks_to_exactly_one():
    assert cruise_ramp_fraction(RAMP, 0) == 0.25
    assert cruise_ramp_fraction(RAMP, CRUISE_RAMP_UPDATE_STEPS - 1) == 0.25
    assert cruise_ramp_fraction(RAMP, CRUISE_RAMP_UPDATE_STEPS) == pytest.approx(0.25 + 0.75 * 0.1)
    assert cruise_ramp_fraction(RAMP, 50_000) == pytest.approx(0.625)
    assert cruise_ramp_fraction(RAMP, 99_999) < 1.0
    assert cruise_ramp_fraction(RAMP, 100_000) == 1.0 and cruise_ramp_fraction(RAMP, 10**9) == 1.0
    assert ramped_command_config(RAMP, 100_000) is RAMP
    assert cruise_ramp_fraction(DirectionCommandConfig(**WALK), 0) == 1.0


def test_set_command_config_swaps_only_the_commanded_speeds_and_keeps_the_declared_manifest():
    env = _live(TYRANNOSAURUS_REX, cruise_ramp_start_fraction=0.25, cruise_ramp_timesteps=100_000)
    try:
        declared, manifest = env.command_config, env.command_manifest()
        tracking = (env.tracking_velocity_sigma, dict(env.tracking_tolerances))
        env.set_command_config(declared.ramped(0.25))
        assert env.command_config.cruise_speed == 0.125 and env.command_manifest() == manifest
        assert (env.tracking_velocity_sigma, env.tracking_tolerances) == tracking
        env.reset(seed=_seed(declared, "cruise"))
        assert 0.0 < env._command_state.desired_speed <= 0.125, "the next reset draws from the swapped config"
        for change in ({"stop_probability": 0.2}, {"speed_scale": 2.0}, {"stand_weights": {"posture_weight": 1.0}}):
            with pytest.raises(ValueError, match="only cruise_speed and speed_range"):
                env.set_command_config(replace(declared, **change))
    finally:
        env.close()
    plain = build_env(TYRANNOSAURUS_REX, "stance")
    try:
        with pytest.raises(RuntimeError, match="needs a live command_mode"):
            plain.set_command_config(declared)
    finally:
        plain.close()


class _Venv:
    """The env_method surface of a vector env over real envs."""

    def __init__(self, envs: list[Any]):
        self.envs = envs
        self.calls: list[DirectionCommandConfig] = []

    def env_method(self, name: str, *args: Any) -> list[Any]:
        assert name == "set_command_config"
        self.calls.append(args[0])
        return [getattr(env, name)(*args) for env in self.envs]


class _Model:
    logger = None

    def __init__(self, venv: _Venv, num_timesteps: int = 0):
        self.venv, self.num_timesteps = venv, num_timesteps

    def get_env(self) -> _Venv:
        return self.venv


def _callback(start: int = 0) -> Any:
    pytest.importorskip("stable_baselines3")
    from environments.shared.curriculum import CruiseRampCallback

    return CruiseRampCallback(RAMP, start_ramp_steps=start)


def _drive(callback: Any, model: _Model, steps: list[int]) -> None:
    callback.init_callback(model)
    callback.on_training_start({}, {})
    for n in steps:
        model.num_timesteps = n
        callback.on_step()


def test_the_callback_primes_every_worker_and_updates_only_at_block_boundaries():
    envs = [_live(TYRANNOSAURUS_REX, cruise_ramp_start_fraction=0.25, cruise_ramp_timesteps=100_000) for _ in range(2)]
    try:
        venv = _Venv(envs)
        callback = _callback()
        callback.prime(venv)
        assert [env.command_config.cruise_speed for env in envs] == [0.125, 0.125]
        model = _Model(venv)
        _drive(callback, model, [8, 16, 9_992, 10_000, 10_008, 99_992, 100_000, 100_008])
        fractions = [config.cruise_speed / RAMP.cruise_speed for config in venv.calls]
        assert fractions == pytest.approx([0.25, 0.325, 0.925, 1.0])
        assert venv.calls[-1] is RAMP and all(env.command_config is RAMP for env in envs)
        assert getattr(model, MODEL_CRUISE_RAMP_ATTRIBUTE) == cruise_ramp_record(
            RAMP, steps_before_session=0, session_start_num_timesteps=0
        )
    finally:
        for env in envs:
            env.close()


def _checkpoint(path: Path, data: dict[str, Any]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
    return path


def test_a_resume_continues_from_the_position_its_checkpoint_records(tmp_path):
    """A SAC or curated checkpoint restarts the step counter; the record places the ramp all the same."""
    record = cruise_ramp_record(RAMP, steps_before_session=30_000, session_start_num_timesteps=5_000)
    path = _checkpoint(tmp_path / "model.zip", {MODEL_CRUISE_RAMP_ATTRIBUTE: record, "num_timesteps": 27_000})
    assert resume_ramp_steps(path, RAMP) == 52_000
    venv = _Venv([])
    callback = _callback(start=52_000)
    model = _Model(venv, num_timesteps=0)
    _drive(callback, model, [8, 8_000])
    assert callback.ramp_steps(8_000) == 60_000
    assert [c.cruise_speed / RAMP.cruise_speed for c in venv.calls] == pytest.approx(
        [0.25 + 0.75 * 0.5, 0.25 + 0.75 * 0.6]
    )
    assert getattr(model, MODEL_CRUISE_RAMP_ATTRIBUTE)["steps_before_session"] == 52_000


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"num_timesteps": 10}, "records no cruise-ramp position"),
        ({MODEL_CRUISE_RAMP_ATTRIBUTE: {"schema": "other"}, "num_timesteps": 10}, "records no cruise-ramp position"),
        (
            {
                MODEL_CRUISE_RAMP_ATTRIBUTE: cruise_ramp_record(
                    RAMP, steps_before_session=0, session_start_num_timesteps=0
                )
            },
            "no step counter",
        ),
        (
            {
                MODEL_CRUISE_RAMP_ATTRIBUTE: cruise_ramp_record(
                    replace(RAMP, cruise_ramp_timesteps=50_000), steps_before_session=0, session_start_num_timesteps=0
                ),
                "num_timesteps": 10,
            },
            "not the stage's",
        ),
    ],
)
def test_a_resume_refuses_a_checkpoint_without_this_ramp(tmp_path, data, match):
    with pytest.raises(CruiseRampError, match=match):
        resume_ramp_steps(_checkpoint(tmp_path / "model.zip", data), RAMP)


def test_train_base_builds_the_ramp_on_every_load_mode_and_refuses_it_from_a_live_parent(tmp_path):
    pytest.importorskip("stable_baselines3")
    from environments.shared import train_base

    stage = {"env_kwargs": {"command_mode": "heading_and_speed", "command_config": RAMP, "forward_vel_weight": 2.0}}
    none = train_base._cruise_ramp_callback({"env_kwargs": {}}, task_load_mode="resume_same_stage", load_path=None)
    assert none is None
    fresh = train_base._cruise_ramp_callback(stage, task_load_mode="resume_same_stage", load_path=None)
    assert fresh.start_ramp_steps == 0
    stance = _checkpoint(tmp_path / "stance.zip", {"mesozoic_task_fingerprint": {"task_sha256": "sha256:0"}})
    entry = train_base._cruise_ramp_callback(stage, task_load_mode="initialize_next_stage", load_path=str(stance))
    assert entry.start_ramp_steps == 0
    walker = _checkpoint(tmp_path / "walk.zip", {"mesozoic_task_fingerprint": {"command": {"schema": "x"}}})
    with pytest.raises(CruiseRampError, match="already tracked a command"):
        train_base._cruise_ramp_callback(stage, task_load_mode="initialize_next_stage", load_path=str(walker))
    record = cruise_ramp_record(RAMP, steps_before_session=0, session_start_num_timesteps=0)
    resumed = _checkpoint(tmp_path / "ckpt.zip", {MODEL_CRUISE_RAMP_ATTRIBUTE: record, "num_timesteps": 40_000})
    resume = train_base._cruise_ramp_callback(stage, task_load_mode="resume_same_stage", load_path=str(resumed))
    assert resume.start_ramp_steps == 40_000
    # A live stage enters with no forward-velocity ramp, whatever its forward_vel_weight.
    shaping = train_base._stage_entry_shaping_callbacks(
        stage, task_load_mode="initialize_next_stage", parent_id="stance", load_path=str(stance)
    )
    assert [type(callback).__name__ for callback in shaping] == ["StageWarmupCallback"]


def test_a_live_stage_trains_under_its_ramp_and_resumes_from_its_position(tmp_path):
    """train() primes the workers before the first reset, the evaluation env keeps the end state, every checkpoint
    records the position, and a same-stage resume continues it."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from environments.shared import train_base
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    configs = load_all_stages(COMPSOGNATHUS_LONGIPES)
    key = next(ref for ref in configs if load_stage_manifest(COMPSOGNATHUS_LONGIPES).resolve(ref).id == "locomotion")
    live = load_stage_config(
        COMPSOGNATHUS_LONGIPES,
        "locomotion",
        config_path=str(
            _walk_stage(
                tmp_path / "configs", COMPSOGNATHUS_LONGIPES, WALK_TABLE.replace("lateral_lean_weight = 0.25\n", "")
            )
        ),
    )
    live["env_kwargs"]["max_episode_steps"] = 64
    live["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=1)
    configs[key] = live
    seen: dict[str, Any] = {}
    original = train_base._cruise_ramp_callback

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["callback"] = original(*args, **kwargs)
        return seen["callback"]

    common = dict(
        n_envs=1,
        seed=3,
        eval_freq=1_000,
        save_freq=1_000,
        verbose=0,
        algorithm="ppo",
        use_tensorboard=False,
        post_eval_episodes=0,
        report_metrics=False,
        save_on_interrupt=False,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_base, "_cruise_ramp_callback", spy)
        model = train_base.train(
            get_species_config(COMPSOGNATHUS_LONGIPES),
            configs,
            key,
            total_timesteps=128,
            output_dir=str(tmp_path / "run"),
            **common,
        )
        assert seen["callback"].fraction == 0.25 and seen["callback"].start_ramp_steps == 0
        assert model.get_env().get_attr("command_config")[0].cruise_speed == 0.125
        record = getattr(model, MODEL_CRUISE_RAMP_ATTRIBUTE)
        assert record["steps_before_session"] == 0 and record["session_start_num_timesteps"] == 0
        (final,) = (tmp_path / "run").rglob("*final.zip")
        assert resume_ramp_steps(final, live["env_kwargs"]["command_config"]) == model.num_timesteps
        resumed = train_base.train(
            get_species_config(COMPSOGNATHUS_LONGIPES),
            configs,
            key,
            total_timesteps=64,
            output_dir=str(tmp_path / "resume"),
            load_path=str(final),
            task_load_mode="resume_same_stage",
            **common,
        )
        assert seen["callback"].start_ramp_steps == model.num_timesteps
        assert getattr(resumed, MODEL_CRUISE_RAMP_ATTRIBUTE)["steps_before_session"] == model.num_timesteps


# ---------------------------------------------------------------------------
# The committed walks: each mechanism on every model's locomotion.toml as committed
# ---------------------------------------------------------------------------


def _cruise_seed_off_the_spawn(config: DirectionCommandConfig, minimum: float = 0.2) -> int:
    """The first cruise seed from 3042 whose reset heading lies more than *minimum* rad off the spawn heading."""
    return next(
        seed
        for seed in range(3042, 3542)
        if _family(config, seed) == "cruise" and abs(_reset_state(config, seed).desired_heading) > minimum
    )


def _unit(heading: float) -> list[float]:
    return [math.cos(heading), math.sin(heading)]


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_every_committed_walk_reads_forward_along_its_commanded_heading_and_follows_a_new_one(model_id):
    """The walk's forward is the reset's commanded heading, off the spawn by the initial heading offset; a later
    request (a follow node's turn) moves it, a stop keeps the last heading, and the stance keeps the spawn's."""
    env = build_env(model_id, "locomotion")
    try:
        command = env.command_config
        seed = _cruise_seed_off_the_spawn(command)
        env.reset(seed=seed)
        heading = env._command_state.desired_heading
        spawn = math.atan2(env._reference_heading_2d[1], env._reference_heading_2d[0])
        assert 0.2 < abs(math.remainder(heading - spawn, math.tau)) <= command.initial_heading_offset_max
        np.testing.assert_allclose(env._forward_reference_2d(), _unit(heading), atol=0)
        _, _, _, _, info = env.step(_zero(env))
        assert info["forward_vel"] == pytest.approx(float(np.dot(env.data.qvel[0:2], _unit(heading))), abs=1e-12)
        turned = heading + 0.5
        env.set_direction(turned, command.cruise_speed)
        np.testing.assert_allclose(env._forward_reference_2d(), _unit(turned), atol=0)
        _, _, _, _, info = env.step(_zero(env))
        assert info["forward_vel"] == pytest.approx(float(np.dot(env.data.qvel[0:2], _unit(turned))), abs=1e-12)
        env.set_direction(turned, 0.0)
        assert env._live_zero_command()
        np.testing.assert_allclose(env._forward_reference_2d(), _unit(turned), atol=0)
    finally:
        env.close()
    stance = build_env(model_id, "stance")
    try:
        stance.reset(seed=seed)
        assert stance._forward_reference_2d() is stance._reference_heading_2d
    finally:
        stance.close()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_every_committed_walk_swaps_in_its_whole_stand_block_exactly_at_a_zero_command(model_id):
    """Every committed stand weight is in force through a step taken under an exactly zero command (a reset stand
    draw, or a stop requested mid-episode) and none through one under a small non-zero command; the constructor's
    weights are back after every step."""
    env = build_env(model_id, "locomotion")
    try:
        stand = env.command_config.stand_weight_map()
        assert stand and env._stand_weights == stand
        constructor = {name: getattr(env, name) for name in stand}
        assert all(constructor[name] != weight for name, weight in stand.items()), "a block restates differences"
        held: list[dict[str, float]] = []
        body = env._step_body

        def spy(action: np.ndarray) -> Any:
            held.append({name: getattr(env, name) for name in stand})
            return body(action)

        env._step_body = spy
        env.reset(seed=_seed(env.command_config, "stand"))
        for _ in range(5):
            env.step(_zero(env))
            assert {name: getattr(env, name) for name in stand} == constructor
        assert held == [stand] * 5, "a reset stand draw holds the block on every step"
        env.reset(seed=_seed(env.command_config, "cruise"))
        for speed, block in ((0.0, stand), (1e-3 * env.command_config.cruise_speed, constructor), (0.0, stand)):
            # Along the body's own heading, so the small request is small in every component (no yaw rate).
            env.set_direction(env._heading(), speed)
            assert env._live_zero_command() is (speed == 0.0) is (not np.any(env._command))
            assert float(np.max(np.abs(env._command))) < 1e-2
            env.step(_zero(env))
            assert held[-1] == block, speed
            assert {name: getattr(env, name) for name in stand} == constructor
    finally:
        env.close()


def test_the_committed_allosaurus_walk_runs_each_course_along_its_reset_heading():
    """The committed walk zeroes forward_vel_weight, yet every cruise episode has a course along its reset
    heading (both signs of the offset), a later request leaves it there, completing it is an event, and a stand
    episode has none."""
    env = build_env(ALLOSAURUS_FRAGILIS, "locomotion")
    try:
        assert env.forward_vel_weight == 0.0 and env.max_episode_steps == 3000
        config = env.command_config
        offsets = []
        for seed in range(3042, 3082):
            env.reset(seed=seed)
            if _family(config, seed) == "stand":
                assert env._course_direction_2d() is None
                continue
            heading = env._reset_command_state.desired_heading
            np.testing.assert_allclose(env._course_direction_2d(), _unit(heading), atol=0)
            spawn = math.atan2(env._reference_heading_2d[1], env._reference_heading_2d[0])
            offsets.append(math.remainder(heading - spawn, math.tau))
        assert max(offsets) > 0.2 and min(offsets) < -0.2
        assert max(abs(offset) for offset in offsets) <= config.initial_heading_offset_max
        env.reset(seed=_cruise_seed_off_the_spawn(config))
        course = env._course_direction_2d()
        env.set_direction(env._command_state.desired_heading + 1.0, config.cruise_speed)
        np.testing.assert_array_equal(env._course_direction_2d(), course)
        # The line across the commanded course: the spawn heading's 8 m reach falls short of it.
        _displace(env, env.COURSE_LENGTH_M - env.COURSE_TOLERANCE_M + 0.05, env._reference_heading_2d)
        _, _, terminated, _, info = env.step(_zero(env))
        assert not terminated and info["course_complete"] == 0.0 and "success" not in info
        _displace(env, env.COURSE_LENGTH_M, course)
        _, _, terminated, _, info = env.step(_zero(env))
        assert not terminated and info["course_complete"] == 1.0 and info["success"] is True
    finally:
        env.close()


def _move_ramp_record(archive: Path, steps: int) -> None:
    """Rewrite *archive*'s cruise-ramp record *steps* further on: a checkpoint saved that far into the ramp."""
    with zipfile.ZipFile(archive) as opened:
        members = [(info, opened.read(info.filename)) for info in opened.infolist()]
    with zipfile.ZipFile(archive, "w") as rewritten:
        for info, payload in members:
            if info.filename == "data":
                data = json.loads(payload)
                data[MODEL_CRUISE_RAMP_ATTRIBUTE]["steps_before_session"] += steps
                payload = json.dumps(data).encode()
            rewritten.writestr(info, payload)


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_the_committed_walk_resumes_its_cruise_ramp_where_its_checkpoint_left_it(tmp_path, algorithm):
    """A ``resume_same_stage`` of the committed walk (from scratch, then from its final pair moved 1M steps into
    the 2M ramp) primes its workers at the recorded position (0.625 of the end-state speeds, not the 0.25
    start), and its own final pair records the position it continued from."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from environments.shared import train_base
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    model_id = COMPSOGNATHUS_LONGIPES
    configs = load_all_stages(model_id)
    key = load_stage_manifest(model_id).by_id("locomotion").reference
    config = configs[key]
    command = config["env_kwargs"]["command_config"]
    assert (command.cruise_ramp_start_fraction, command.cruise_ramp_timesteps) == (0.25, 2_000_000)
    config[f"{algorithm}_kwargs"].update(
        {"n_steps": 64, "batch_size": 64, "n_epochs": 1}
        if algorithm == "ppo"
        else {"learning_starts": 32, "buffer_size": 1_000, "batch_size": 32, "train_freq": 16, "gradient_steps": 1}
    )
    config["curriculum_kwargs"].update(min_cruise_episodes=1, min_stand_episodes=1)
    seen: dict[str, Any] = {}
    original = train_base._cruise_ramp_callback

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["callback"] = original(*args, **kwargs)
        return seen["callback"]

    common: dict[str, Any] = dict(
        n_envs=1,
        seed=5,
        eval_freq=1_000_000,
        save_freq=1_000_000,
        verbose=0,
        algorithm=algorithm,
        use_tensorboard=False,
        post_eval_episodes=0,
        report_metrics=False,
        save_on_interrupt=False,
    )
    species = get_species_config(model_id)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_base, "_cruise_ramp_callback", spy)
        first = train_base.train(species, configs, key, total_timesteps=64, output_dir=str(tmp_path / "run"), **common)
        assert seen["callback"].start_ramp_steps == 0
        assert first.get_env().get_attr("command_config")[0] == command.ramped(0.25)
        (final,) = (tmp_path / "run" / "models").glob("*_final.zip")
        _move_ramp_record(final, 1_000_000)
        position = resume_ramp_steps(final, command)
        assert position == 1_000_000 + first.num_timesteps
        assert cruise_ramp_fraction(command, position) == pytest.approx(0.625)
        resumed = train_base.train(
            species,
            configs,
            key,
            total_timesteps=64,
            output_dir=str(tmp_path / "resume"),
            load_path=str(final.with_suffix("")),
            task_load_mode="resume_same_stage",
            **common,
        )
    assert seen["callback"].start_ramp_steps == position
    worker = resumed.get_env().get_attr("command_config")[0]
    assert worker == ramped_command_config(command, position)
    assert worker.cruise_speed == pytest.approx(0.625 * command.cruise_speed)
    assert worker.speed_range == pytest.approx(tuple(0.625 * bound for bound in command.speed_range))
    record = getattr(resumed, MODEL_CRUISE_RAMP_ATTRIBUTE)
    assert record["steps_before_session"] == position
    (again,) = (tmp_path / "resume" / "models").glob("*_final.zip")
    assert resume_ramp_steps(again, command) == position + resumed.num_timesteps - record["session_start_num_timesteps"]


# ---------------------------------------------------------------------------
# The in-training evaluation of a live stage
# ---------------------------------------------------------------------------

#: Per-episode returns of the velociraptor walk's zero-action statue (D-D62 (18)): cruise about 500, stand about
#: 5720; a checkpoint that stands worse is the same statue earning 3000 on a stand episode.
_CRUISE_RETURN, _GOOD_STAND_RETURN, _POOR_STAND_RETURN = 500.0, 5720.0, 3000.0


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=str)
def test_the_evaluation_plan_is_the_command_stream_of_an_env_reseeded_at_its_seed(model_id):
    """The plan's families are the ones an env rolls on consecutive episodes after ``reset(seed=S)``, as the
    reseeded evaluation env does, so every evaluation of a live stage rolls the same stand count."""
    from environments.shared.command_tracking_panel import reset_command_family
    from environments.shared.curriculum.command_evaluation import command_evaluation_plan, episode_command_family
    from environments.shared.train_base import eval_env_seed

    env = build_env(model_id, "locomotion")
    try:
        command = env.command_config
        seed = eval_env_seed(42)
        plan = command_evaluation_plan(command, seed, 30)
        for _ in range(2):  # a second pass from the same seed is the next evaluation
            env.reset(seed=seed)
            rolled = []
            for index in range(30):
                if index:
                    env.reset()
                rolled.append("stand" if env._reset_command_state.desired_speed == 0.0 else "cruise")
            assert tuple(rolled) == plan.families
        # The command stream is the model's command config's alone: the default run's evaluation draws 4 stands.
        assert plan.family_counts() == {"cruise": 26, "stand": 4}
        assert plan.family_weights() == {"cruise": pytest.approx(0.85), "stand": pytest.approx(0.15)}
        for probe in range(3042, 3062):
            assert reset_command_family(command, probe) == episode_command_family(command, probe, 0)
    finally:
        env.close()


def test_the_family_aware_score_ranks_standing_where_the_pooled_score_reads_the_family_gap_as_risk():
    from environments.shared.curriculum.command_evaluation import family_robust_score, family_statistics

    families = ["cruise"] * 26 + ["stand"] * 4
    rng = np.random.default_rng(0)
    cruise = _CRUISE_RETURN + rng.normal(0.0, 1.0, 26)
    good = np.concatenate([cruise, _GOOD_STAND_RETURN + rng.normal(0.0, 25.0, 4)])
    poor = np.concatenate([cruise, _POOR_STAND_RETURN + rng.normal(0.0, 25.0, 4)])
    weights = {"cruise": 0.85, "stand": 0.15}

    def pooled(row: np.ndarray) -> float:
        return float(row.mean() - row.std())

    assert pooled(poor) > pooled(good), "pooled, the family gap reads as risk"
    assert family_robust_score(good, families, weights) > family_robust_score(poor, families, weights)
    stats = family_statistics(good, families)
    assert list(stats) == ["cruise", "stand"] and stats["stand"][0] == 4
    expected = 0.85 * (stats["cruise"][1] - stats["cruise"][2]) + 0.15 * (stats["stand"][1] - stats["stand"][2])
    assert family_robust_score(good, families, weights) == pytest.approx(expected)
    # One family present: scored on it alone (the weights renormalized), so a run's evaluations stay comparable.
    assert family_robust_score(cruise, ["cruise"] * 26, weights) == pytest.approx(cruise.mean() - cruise.std())
    with pytest.raises(ValueError, match="30 returns but 29"):
        family_robust_score(good, families[:-1], weights)
    with pytest.raises(ValueError, match="unknown command families"):
        family_robust_score(good, ["cruise"] * 29 + ["turn"], weights)
    with pytest.raises(ValueError, match="carry no weight"):
        family_robust_score(cruise, ["cruise"] * 26, {"cruise": 0.0, "stand": 1.0})


def test_the_evaluation_plan_is_built_for_a_live_stage_only():
    from environments.shared.curriculum.command_evaluation import CommandEvaluationPlan, command_evaluation_plan
    from environments.shared.train_base import _command_evaluation_plan, eval_env_seed

    for model_id in MODEL_IDS:
        walk = load_stage_config(model_id, "locomotion")
        plan = _command_evaluation_plan(walk, 7)
        assert plan == command_evaluation_plan(walk["env_kwargs"]["command_config"], eval_env_seed(7), 30)
        assert _command_evaluation_plan(load_stage_config(model_id, "stance"), 7) is None
        with pytest.raises(ValueError, match="pass the run's seed"):
            _command_evaluation_plan(walk, None)
    with pytest.raises(ValueError, match="unknown command families"):
        CommandEvaluationPlan(seed=1, families=("cruise", "hop"), stop_probability=0.15)
    with pytest.raises(ValueError, match="stop_probability"):
        CommandEvaluationPlan(seed=1, families=("cruise",), stop_probability=1.0)


def _robust_callback(tmp_path: Path, plan: Any, results: list[Any], families: list[list[str]]) -> Any:
    from types import SimpleNamespace

    from environments.shared.curriculum import RobustBestModelCallback

    eval_callback = SimpleNamespace(
        command_evaluation=plan, evaluations_results=results, evaluations_command_families=families
    )
    return RobustBestModelCallback(eval_callback, model_dir=tmp_path)


def test_the_robust_tracker_scores_a_live_stage_by_family_and_refuses_an_evaluation_without_families(tmp_path):
    pytest.importorskip("stable_baselines3")
    from environments.shared.curriculum.command_evaluation import CommandEvaluationPlan, family_robust_score

    families = ("cruise",) * 26 + ("stand",) * 4
    plan = CommandEvaluationPlan(seed=1042, families=families, stop_probability=0.15)
    row = [_CRUISE_RETURN] * 26 + [_GOOD_STAND_RETURN] * 4
    callback = _robust_callback(tmp_path, plan, [row], [list(families)])
    assert callback.score(row, families) == pytest.approx(family_robust_score(row, families, plan.family_weights()))
    assert callback.prior_score(row) == callback.score(row, families)
    assert callback.prior_score(row[:20]) is None, "another episode count is not this plan's"
    with pytest.raises(ValueError, match="needs each evaluation episode's command family"):
        callback.score(row)
    unaligned = _robust_callback(tmp_path, plan, [row], [])
    with pytest.raises(RuntimeError, match="recorded 0 evaluation"):
        unaligned._on_step()
    # Command-free: the pooled score, as before.
    pooled = _robust_callback(tmp_path, None, [row], [])
    assert pooled.score(row) == pytest.approx(np.mean(row) - np.std(row))


def test_a_resumed_robust_tracker_scores_the_prior_record_on_the_plan(tmp_path):
    pytest.importorskip("stable_baselines3")
    from types import SimpleNamespace

    from environments.shared.curriculum import seed_resume_eval_state
    from environments.shared.curriculum.command_evaluation import CommandEvaluationPlan

    families = ("cruise",) * 26 + ("stand",) * 4
    plan = CommandEvaluationPlan(seed=1042, families=families, stop_probability=0.15)
    good = [_CRUISE_RETURN] * 26 + [_GOOD_STAND_RETURN] * 4
    poor = [_CRUISE_RETURN + 50.0] * 26 + [_POOR_STAND_RETURN] * 4
    record = tmp_path / "evaluations.npz"
    np.savez(record, timesteps=[100, 200], results=[good, poor], ep_lengths=[[1] * 30, [1] * 30])
    eval_callback = SimpleNamespace(command_evaluation=plan, evaluations_command_families=[], n_eval_episodes=30)
    callback = _robust_callback(tmp_path, plan, [], [])
    callback.eval_callback = eval_callback
    seed_resume_eval_state(eval_callback, [callback], record, max_timesteps=200)
    assert callback.best_score == pytest.approx(max(callback.score(good, families), callback.score(poor, families)))
    assert callback.best_score == pytest.approx(callback.score(good, families)), "the better stander's score"
    # A prior record of another panel size scores nothing into the robust tracker.
    short = tmp_path / "short" / "evaluations.npz"
    short.parent.mkdir()
    np.savez(short, timesteps=[100], results=[good[:20]], ep_lengths=[[1] * 20])
    fresh = _robust_callback(tmp_path, plan, [], [])
    seed_resume_eval_state(SimpleNamespace(command_evaluation=plan, n_eval_episodes=30), [fresh], short)
    assert fresh.best_score == -np.inf


def test_a_live_stage_reseeds_every_evaluation_and_scores_its_robust_pair_by_family(tmp_path):
    """Trained through ``train_base.train`` on the committed walk (episodes cut to 4 steps so an evaluation of
    30 is cheap): both evaluations roll the plan's families, from the plan's seed, and the robust tracker holds
    the better family-aware score of the two."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from environments.shared import train_base
    from environments.shared.config import load_all_stages
    from environments.shared.curriculum import RobustBestModelCallback
    from environments.shared.curriculum.command_evaluation import command_evaluation_plan
    from environments.shared.eval_diagnostics import StageAwareEvalCallback
    from environments.shared.species_registry import get_species_config

    model_id = BRACHIOSAURUS_ALTITHORAX
    configs = load_all_stages(model_id)
    key = load_stage_manifest(model_id).by_id("locomotion").reference
    config = configs[key]
    config["env_kwargs"]["max_episode_steps"] = 4
    config["ppo_kwargs"].update({"n_steps": 32, "batch_size": 32, "n_epochs": 1})
    config["curriculum_kwargs"].update(min_cruise_episodes=1, min_stand_episodes=1)
    built: dict[str, Any] = {}
    original = train_base._build_core_callbacks

    def spy(*args: Any, **kwargs: Any) -> Any:
        built["callbacks"] = original(*args, **kwargs)
        return built["callbacks"]

    seeds: list[tuple[int, int]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_base, "_build_core_callbacks", spy)
        original_on_step = StageAwareEvalCallback.__dict__["_on_step"]

        def on_step(self: Any) -> bool:
            result = bool(original_on_step(self))
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                env = self.eval_env.venv.envs[0].unwrapped
                seeds.append((env.np_random_seed, env._command_stream_episode))
            return result

        patch.setattr(StageAwareEvalCallback, "_on_step", on_step)
        train_base.train(
            get_species_config(model_id),
            configs,
            key,
            total_timesteps=64,
            output_dir=str(tmp_path / "run"),
            n_envs=1,
            seed=5,
            eval_freq=32,
            save_freq=1_000_000,
            verbose=0,
            algorithm="ppo",
            use_tensorboard=False,
            post_eval_episodes=0,
            report_metrics=False,
            save_on_interrupt=False,
        )
    callbacks, eval_callback, _ = built["callbacks"]
    plan = command_evaluation_plan(config["env_kwargs"]["command_config"], train_base.eval_env_seed(5), 30)
    assert isinstance(eval_callback, StageAwareEvalCallback) and eval_callback.command_evaluation == plan
    assert len(eval_callback.evaluations_results) == 2
    assert eval_callback.evaluations_command_families == [list(plan.families)] * 2
    # Each evaluation restarted the command stream at a reset seeded with the plan's seed: its 30 episodes are
    # stream episodes 0-29, and the reset after the last is episode 30 (an env seeded once would count on).
    assert seeds == [(plan.seed, 30)] * 2
    (robust,) = [callback for callback in callbacks if isinstance(callback, RobustBestModelCallback)]
    assert robust.best_score == pytest.approx(
        max(robust.score(row, plan.families) for row in eval_callback.evaluations_results)
    )
    assert (tmp_path / "run").exists()


# ---------------------------------------------------------------------------
# The digest harness
# ---------------------------------------------------------------------------


def test_the_state_probes_score_a_live_stage_as_its_step_does():
    from environments.shared.harnesses import digest_snapshot

    env = _live(VELOCIRAPTOR_MONGOLIENSIS, stand_weights={"tail_home_pose_weight": 0.2})
    try:
        for family in ("stand", "cruise"):
            env.reset(seed=_seed(env.command_config, family))
            reward, info = digest_snapshot._posed_reward(env, _zero(env))
            assert "reward_tracking" in info and info["reward_total"] == reward
            assert ("reward_tail_home_pose" in info) == (family == "stand")
            assert env.tail_home_pose_weight == 0.0
            assert {"tracking_in_tolerance", "tracking_dwell_s"} <= set(digest_snapshot._posed_command_info(env))
    finally:
        env.close()
    plain = build_env(VELOCIRAPTOR_MONGOLIENSIS, "stance")
    try:
        plain.reset(seed=3042)
        reward, info = digest_snapshot._posed_reward(plain, _zero(plain))
        plain.reset(seed=3042)
        assert (reward, info) == plain._get_reward_info(_zero(plain))
        assert digest_snapshot._posed_command_info(plain) == {}
    finally:
        plain.close()


# ---------------------------------------------------------------------------
# The recipe env keeps the rules before the live walk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", [TYRANNOSAURUS_REX, ALLOSAURUS_FRAGILIS], ids=str)
def test_a_recipe_env_keeps_the_spawn_anchors_and_the_none_course(model_id):
    """The pilot recipes' rollouts are the Phase D golden's: their terms and the Allosaurus course read the spawn."""
    from environments.shared.train_behaviors import create_behavior_env

    env = create_behavior_env(model_id, commands=DirectionCommandConfig(**WALK), terrain=None, run_seed=0)
    try:
        env.reset(seed=3042)
        assert env.direction_controller is not None and not env._rewards_follow_command()
        assert env._forward_reference_2d() is env._reference_heading_2d
        assert env._drift_anchor_2d() is env._initial_pos_2d
        env.set_direction(0.0, 0.0)
        assert not env._live_zero_command()
        if model_id == ALLOSAURUS_FRAGILIS:
            assert env._course_direction_2d() is None, "a recipe zeroes forward_vel_weight: no course, as before"
    finally:
        env.close()


@pytest.mark.parametrize(
    "commands",
    [
        {"stand_weights": {"drift_penalty_weight": 0.3}},
        {"cruise_ramp_start_fraction": 0.25, "cruise_ramp_timesteps": 1000},
    ],
)
def test_a_recipe_declares_no_stand_block_and_no_cruise_ramp(commands):
    from environments.shared.train_behaviors import create_behavior_env

    with pytest.raises(ValueError, match="recipe pipeline runs neither"):
        create_behavior_env(
            TYRANNOSAURUS_REX, commands=DirectionCommandConfig(**WALK, **commands), terrain=None, run_seed=0
        )
