"""Phase C interface pins (BEHAVIOR_RECIPES_PLAN §4.6, WS-C1): the SB3 / backend-neutral half.

The one Phase C interface revision appends a 3-dim body-relative command
segment to every species' observation.  These tests pin what that must and
must not have changed on the Gymnasium (SB3) side and in the backend-neutral
contract code:

* The **golden fixture** (``fixtures/phase_c_reset_golden.json``, captured at
  7db7f8e by ``reset_golden.py``) is replayed on the current tree.  Its
  ``reset`` half — qpos / qvel after ``reset(seed)``, the mocap target rows,
  the push-schedule starts and directions, and the generator state the
  reset leaves behind — is THE CONTRACT (decision D3: the seeded draw
  stream is unchanged; decision D-C2: the compsognathus recovery
  calibrations were restamped, not re-measured) and is compared exactly at
  6 decimals; so is its ``second_reset`` record (an unseeded ``reset()``
  after the trajectory, the way every later training episode starts), which
  catches a draw the hook consumed and discarded.  Its ``observation``
  record is a prefix contract (byte-inert except the trailing zeros).  Its
  ``trajectory`` half (qpos after each zero-action step) crosses the
  solver, so it is ADVISORY: ``assert_allclose(atol=1e-5)`` and skipped
  unless the running MuJoCo is the fixture's (amendment A4).  The trex
  captures were re-taken for trex physics r8 (plant_versions note 13) and
  again for r9 (note 16; the fixture's ``recaptures`` record): each time
  their reset halves were unchanged, which is what proves the revision drew
  nothing extra, and only their observation and trajectory halves moved, as
  dynamics edits must.  They and the anatomical compsognathus captures were
  re-taken once more for the heading-free observation (note 18: trex policy
  interface r14, compsognathus r4), an interface-only revision: the reset,
  second-reset and trajectory halves re-took byte-identical and only the
  observation half moved, and each record names the policy interface and
  observation schema it was taken under.
* The command hook itself is pinned directly: under ``command_mode =
  "none"`` it touches ``np_random`` not at all, and it runs after every
  reset draw (the push schedule included) and before ``_get_obs``.
* Every species' observation ends with three zeros under ``command_mode =
  "none"`` and matches the plant identity width; the segment is the last
  ``observation_segments`` entry on all six species (decision D-C1).
* ``command_mode = "none"`` is inert and deterministic.  Since Phase D
  (consolidation PR-9) the SB3 backend builds the direction controller for
  both live modes from ONE ``command_config`` (decision D-D2): seeded in the
  hook with one draw after every reset draw, advanced at the end of
  ``step()``; a setting a mode would ignore is refused at construction.
* The two command kwargs are carved out of the task fingerprint while the
  effective ``command_mode`` is ``"none"`` (amendment A1), and the
  ``command`` payload section is a live mode's controller manifest, passed
  exactly when the mode is live (amendment A9, as PR-9 carried it out).
* The recovery-calibration restamp tool is idempotent and refuses a physics
  change; the committed compsognathus calibrations load (decision D-C2).  The
  robot's still carries its Phase C restamp; the anatomical one was
  re-measured on compsognathus physics r2 (D-D26), the physics change a
  restamp refuses, and carries one interface restamp since: the heading-free
  policy interface r4 (plant_versions note 18, D-D31), an interface-only
  revision on the physics it was measured on.  Both then carry the identity
  migration of decision D-D33 (``--migrate-identity``, plant_versions note
  19), which the tool reproduces byte for byte from the schema-v1 profile.

Nothing here imports stable_baselines3, torch or jax, so the module runs in
full in the shared legs of the test matrix (``.[test]`` only); the SB3 job
does not repeat it (CU-14a).
The dual-backend probe parity pin lives in ``test_plant_contract_phase_c.py``;
the MJX runtime half left with the JAX runtime (D-D17).
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv
from environments.compsognathus_longipes.anatomical.env import CompsognathusLongipesAnatomicalEnv
from environments.compsognathus_longipes.robot.env import CompsognathusLongipesRobotEnv
from environments.dibothrosuchus_elaphros.anatomical.env import DibothrosuchusElaphrosAnatomicalEnv
from environments.shared.base_env import BaseDinoEnv
from environments.shared.command_frame import (
    COMMAND_COMPONENTS,
    COMMAND_ENV_KEYS,
    COMMAND_MODES,
    COMMAND_SEGMENT_NAME,
    COMMAND_WIDTH,
    RETIRED_COMMAND_ENV_KEYS,
)
from environments.shared.config import load_all_stages, load_stage_config, save_stage_config
from environments.shared.direction_commands import DirectionCommandConfig, DirectionCommandController, wrap_angle
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity, load_plant_versions
from environments.shared.plant_contract.digests import _canonical_float
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.shared.species_names import model_ids
from environments.shared.species_registry import get_species_config
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import (
    TaskFingerprintError,
    compute_task_fingerprint,
    derive_stage_task_fingerprint,
    stage_task_fingerprint,
)
from environments.shared.tests import reset_golden
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
MODEL_ENVS = [
    pytest.param(env_class, model_id, id=f"{model_id.species}-{model_id.embodiment}")
    for env_class, model_id in (
        (VelociraptorMongoliensisAnatomicalEnv, ModelId("velociraptor_mongoliensis", "anatomical")),
        (TyrannosaurusRexAnatomicalEnv, TYRANNOSAURUS_REX),
        (BrachiosaurusAltithoraxAnatomicalEnv, ModelId("brachiosaurus_altithorax", "anatomical")),
        (DibothrosuchusElaphrosAnatomicalEnv, ModelId("dibothrosuchus_elaphros", "anatomical")),
        (CompsognathusLongipesAnatomicalEnv, COMPSOGNATHUS_LONGIPES),
        (CompsognathusLongipesRobotEnv, COMPSOGNATHUS_LONGIPES_ROBOT),
    )
]

_FAKE_PLANT = {
    "physics_sha256": "sha256:aaaa",
    "policy_interface_sha256": "sha256:bbbb",
    "model_path": "environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml",
}
_TYRANNOSAURUS_REX_ENV = {"alive_bonus": 1.0, "healthy_z_range": (0.70, 1.30), "max_episode_steps": 1000}
COMMAND_ENV_KEYS_SET = frozenset(COMMAND_ENV_KEYS)


# ---------------------------------------------------------------------------
# Golden fixture replay (decision D3 / D-C2, amendment A4)
# ---------------------------------------------------------------------------

GOLDEN = reset_golden.load_fixture()


def _flatten(records: dict[str, Any]) -> dict[tuple[ModelId, str], Any]:
    """``records[species][embodiment][stage]`` (the fixture's nesting) keyed by ``(model, stage)``."""
    return {
        (ModelId(species, embodiment), stage): record
        for species, embodiments in records.items()
        for embodiment, stages in embodiments.items()
        for stage, record in stages.items()
    }


CAPTURES = _flatten(GOLDEN["captures"])
RECAPTURES = _flatten(GOLDEN.get("recaptures", {}))
GOLDEN_KEYS = sorted(CAPTURES)
GOLDEN_IDS = [f"{model_id.species}-{model_id.embodiment}-{stage}" for model_id, stage in GOLDEN_KEYS]
_REPLAYS: dict[tuple[ModelId, str], dict[str, Any]] = {}


def _replay(key: tuple[ModelId, str]) -> dict[str, Any]:
    """Replay one capture on the current tree (cached: the fixture halves share it)."""
    if key not in _REPLAYS:
        golden = CAPTURES[key]
        assert (ModelId.from_record(golden), golden["stage"]) == key
        _REPLAYS[key] = reset_golden.capture(
            ModelId.from_record(golden),
            golden["stage"],
            [int(seed) for seed in golden["seeds"]],
            int(golden["steps"]),
        )
    return _REPLAYS[key]


def test_golden_fixture_records_its_provenance():
    assert GOLDEN["schema"] == reset_golden.FIXTURE_SCHEMA
    assert GOLDEN["source_commit"] == "7db7f8e"
    assert GOLDEN["quantization_decimals"] == reset_golden.QUANTIZATION_DECIMALS == 6
    assert re.fullmatch(r"\d+\.\d+\.\d+", GOLDEN["mujoco_version"])
    assert GOLDEN["platform"] and GOLDEN["numpy_version"]
    assert [
        (ModelId.from_record(entry), entry["stage"], tuple(entry["seeds"]), entry["steps"]) for entry in GOLDEN["spec"]
    ] == [(model_id, stage, tuple(seeds), steps) for model_id, stage, seeds, steps in reset_golden.DEFAULT_SPEC]
    assert set(GOLDEN_KEYS) == {(model_id, stage) for model_id, stage, _, _ in reset_golden.DEFAULT_SPEC}


#: Captures re-taken after 7db7f8e, each by a named plant revision of its model: the physics revision it was
#: re-taken on, and the policy interface and observation schema its observation half was taken under (the last
#: recapture, plant_versions note 18's heading-free observation, an interface-only revision).  Re-pinned 14, 14,
#: 4 -> 15, 15, 5 by the identity migration (plant_versions note 19, decision D-D33), which refreshed each
#: recapture's interface revision; no capture was re-taken.
RECAPTURED = {
    (TYRANNOSAURUS_REX, "stance"): 9,
    (TYRANNOSAURUS_REX, "recovery"): 9,
    (COMPSOGNATHUS_LONGIPES, "recovery"): 2,
}
RECAPTURED_INTERFACES = {
    (TYRANNOSAURUS_REX, "stance"): (15, "bipedal-target-heading-free/v1"),
    (TYRANNOSAURUS_REX, "recovery"): (15, "bipedal-target-heading-free/v1"),
    (COMPSOGNATHUS_LONGIPES, "recovery"): (5, "bipedal-target-heading-free/v1"),
}


def test_recaptured_goldens_record_the_plant_they_were_taken_on():
    """A recapture is bound to its species' current physics: a later physics move fails HERE, by name.

    Without the binding the next dynamics edit would surface only as an
    unexplained trajectory mismatch; with it the failure says which capture
    to re-take (``reset_golden.py --recapture``) after the reset half has
    been shown to hold.
    """
    assert {key: record["physics_revision"] for key, record in RECAPTURES.items()} == RECAPTURED
    assert {
        key: (record["policy_interface_revision"], record["observation_schema"]) for key, record in RECAPTURES.items()
    } == RECAPTURED_INTERFACES
    versions = load_plant_versions()[1]
    for key, record in RECAPTURES.items():
        model_id = key[0]
        plant = current_plant_identity(model_id)
        assert (record["physics_revision"], record["physics_sha256"]) == (
            plant.physics_revision,
            plant.physics_sha256,
        ), f"{key} was recaptured on another {model_id} plant; re-take it with reset_golden.py --recapture"
        # The observation half is bound to its layout the same way: a later interface revision fails here.
        assert (record["policy_interface_revision"], record["observation_schema"]) == (
            plant.policy_interface_revision,
            versions[model_id].observation_schema,
        ), f"{key}'s observation was taken under another {model_id} interface; re-take it with --recapture"
        assert record["observation_includes_command_segment"] is True
        assert record["mujoco_version"] == GOLDEN["mujoco_version"] and record["reason"]


@pytest.mark.parametrize("key", GOLDEN_KEYS, ids=GOLDEN_IDS)
def test_reset_draw_stream_matches_the_pre_bump_golden(key):
    """THE CONTRACT: qpos/qvel/targets/push schedules after reset(seed), exact at 6 decimals.

    A reset draw added before the existing draws breaks this by design; the
    Phase C command hook draws nothing under ``command_mode = "none"``.
    """
    golden = CAPTURES[key]
    current = _replay(key)
    assert current["seeds"].keys() == golden["seeds"].keys()
    for seed, record in golden["seeds"].items():
        assert record["reset"]["rng_state"]["bit_generator"] == "PCG64"
        assert current["seeds"][seed]["reset"] == record["reset"], f"{key} seed {seed}: reset draw stream moved"
        # The unseeded follow-up reset: a draw consumed anywhere in reset()
        # -- even one whose value is discarded -- shifts every later episode.
        assert current["seeds"][seed]["second_reset"] == record["second_reset"], (
            f"{key} seed {seed}: the unseeded second reset moved (an extra RNG draw in reset?)"
        )
        assert record["second_reset"]["rng_state"] != record["reset"]["rng_state"]
        if key[1] == "recovery":
            assert record["reset"]["push_schedule_starts"], f"{key} must exercise the push schedule"
            assert record["second_reset"]["push_schedule_starts"] != record["reset"]["push_schedule_starts"]


@pytest.mark.parametrize("key", GOLDEN_KEYS, ids=GOLDEN_IDS)
def test_reset_observation_is_byte_inert_except_the_trailing_zeros(key):
    golden = CAPTURES[key]
    current = _replay(key)
    for seed, record in golden["seeds"].items():
        before = np.asarray(record["observation"], dtype=np.float64)
        after = np.asarray(current["seeds"][seed]["observation"], dtype=np.float64)
        if key in RECAPTURES:
            # Re-taken after the bump, so the capture already ends in the
            # command segment: the whole observation must reproduce.
            np.testing.assert_array_equal(after, before, err_msg=f"{key} seed {seed}")
            np.testing.assert_array_equal(before[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH))
            continue
        assert after.shape == (before.shape[0] + COMMAND_WIDTH,)
        np.testing.assert_array_equal(after[: before.shape[0]], before, err_msg=f"{key} seed {seed}")
        np.testing.assert_array_equal(after[before.shape[0] :], np.zeros(COMMAND_WIDTH))


@pytest.mark.parametrize("key", GOLDEN_KEYS, ids=GOLDEN_IDS)
def test_zero_action_trajectories_match_the_pre_bump_golden(key):
    """ADVISORY (amendment A4): stepping crosses the solver, so only the fixture's MuJoCo compares."""
    if mujoco.__version__ != GOLDEN["mujoco_version"]:
        pytest.skip(
            f"trajectory half is advisory: running mujoco {mujoco.__version__} != fixture {GOLDEN['mujoco_version']}"
        )
    golden = CAPTURES[key]
    current = _replay(key)
    for seed, record in golden["seeds"].items():
        np.testing.assert_allclose(
            np.asarray(current["seeds"][seed]["trajectory"], dtype=np.float64),
            np.asarray(record["trajectory"], dtype=np.float64),
            rtol=0.0,
            atol=1e-5,
            err_msg=f"{key} seed {seed}",
        )


# ---------------------------------------------------------------------------
# Observation layout on every species (decision D-C1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("env_class", "model_id"), MODEL_ENVS)
def test_every_model_observation_ends_with_three_zero_command_dims(env_class, model_id):
    identity = current_plant_identity(model_id, verify_generated=False)
    env = env_class()
    try:
        obs, _ = env.reset(seed=3)
        assert obs.shape == (identity.observation_dim,)
        assert obs.dtype == np.float32
        np.testing.assert_array_equal(obs[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH, dtype=np.float32))
        stepped, _, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        np.testing.assert_array_equal(stepped[-COMMAND_WIDTH:], np.zeros(COMMAND_WIDTH, dtype=np.float32))

        space = env.observation_space
        assert space.shape == (identity.observation_dim,)
        assert space.dtype == np.float32
        assert np.all(np.isneginf(space.low)) and np.all(np.isposinf(space.high))
        assert env.command_mode == "none"
        assert env._command.dtype == np.float32 and env._command.shape == (COMMAND_WIDTH,)
    finally:
        env.close()


def test_observation_segments_end_with_the_command_segment_for_every_model():
    versions = load_plant_versions()[1]
    for env_class, model_id in ((param.values[0], param.values[1]) for param in MODEL_ENVS):
        env = env_class(reset_noise_scale=0.0)
        try:
            payload = _policy_interface_payload(env.model, env, versions[model_id])
        finally:
            env.close()
        segments = payload["observation_segments"]
        assert segments[-1] == {
            "name": COMMAND_SEGMENT_NAME,
            "width": COMMAND_WIDTH,
            "components": list(COMMAND_COMPONENTS),
            "frame": "body-relative",
            "range": [_canonical_float(-1.0), _canonical_float(1.0)],
        }, model_id
        assert [float(bound) for bound in segments[-1]["range"]] == [-1.0, 1.0]
        assert [segment["name"] for segment in segments].count(COMMAND_SEGMENT_NAME) == 1
        assert segments[-2]["name"].endswith("_distance") and segments[-2]["width"] == 1
        assert payload["observation"]["shape"] == [
            current_plant_identity(model_id, verify_generated=False).observation_dim
        ]


# ---------------------------------------------------------------------------
# command_mode on the SB3 backend (decisions D-C5, D-C7)
# ---------------------------------------------------------------------------


def test_command_mode_none_is_inert_and_deterministic_on_sb3():
    explicit_a = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0, command_mode="none")
    explicit_b = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0, command_mode="none")
    plain = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0)
    try:
        for env in (explicit_a, explicit_b, plain):
            env.reset(seed=7)
        action = np.zeros(plain.action_space.shape, dtype=np.float32)
        for _ in range(30):
            for env in (explicit_a, explicit_b, plain):
                env.step(action)
        np.testing.assert_array_equal(explicit_a.data.qpos, explicit_b.data.qpos)
        np.testing.assert_array_equal(explicit_a.data.qpos, plain.data.qpos)
        for env in (explicit_a, explicit_b, plain):
            assert env.command_manifest() is None
            assert env.command_mode == "none"
            np.testing.assert_array_equal(env._command, np.zeros(COMMAND_WIDTH, dtype=np.float32))
            np.testing.assert_array_equal(env._draw_episode_command(), np.zeros(COMMAND_WIDTH, dtype=np.float32))
    finally:
        for env in (explicit_a, explicit_b, plain):
            env.close()


def test_command_hook_draws_no_rng_and_runs_after_every_reset_draw():
    """Decision D-C5, pinned on the hook itself rather than through its effects.

    Phase D replaced the hook body with a real draw under a live mode, so the
    ordering half is what keeps the seeded draw stream (the golden fixture) intact: the
    hook must see the NEW push schedule and the generator state the reset
    ends with, and its value must reach the reset observation.
    """
    from environments.shared.config import load_stage_config

    marker = np.asarray([0.125, -0.25, 0.5], dtype=np.float32)
    seen: dict[str, Any] = {}

    class RecordingEnv(TyrannosaurusRexAnatomicalEnv):
        def _draw_episode_command(self) -> np.ndarray:
            seen["rng_state"] = self.np_random.bit_generator.state
            seen["push_starts"] = np.array(self._push_schedule_starts, copy=True)
            seen["qpos"] = np.array(self.data.qpos, copy=True)
            seen["mocap_pos"] = np.array(self.data.mocap_pos, copy=True)
            command: np.ndarray = marker.copy()
            return command

    env = RecordingEnv(**load_stage_config(TYRANNOSAURUS_REX, "recovery")["env_kwargs"])
    try:
        env.reset(seed=5)
        previous_starts = np.array(env._push_schedule_starts, copy=True)
        seen.clear()
        obs, _ = env.reset(seed=11)
        # After every reset draw: nothing is drawn once the hook has run ...
        assert seen["rng_state"] == env.np_random.bit_generator.state
        # ... and the push schedule (the last draw) was already regenerated.
        np.testing.assert_array_equal(seen["push_starts"], env._push_schedule_starts)
        assert not np.array_equal(seen["push_starts"], previous_starts)
        np.testing.assert_array_equal(seen["qpos"], env.data.qpos)
        np.testing.assert_array_equal(seen["mocap_pos"], env.data.mocap_pos)
        # Before _get_obs: the hook's value is what the reset observation carries.
        np.testing.assert_array_equal(obs[-COMMAND_WIDTH:], marker)
        np.testing.assert_array_equal(env._command, marker)
    finally:
        env.close()

    # Under "none" the hook never touches the generator: every attribute
    # access on np_random while it runs is an error.
    class _NoDraws:
        def __getattr__(self, name):
            raise AssertionError(f"_draw_episode_command touched np_random.{name} under command_mode 'none'")

    plain = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0)
    try:
        plain.reset(seed=3)
        real_generator = plain._np_random
        plain._np_random = _NoDraws()
        try:
            np.testing.assert_array_equal(plain._draw_episode_command(), np.zeros(COMMAND_WIDTH, dtype=np.float32))
        finally:
            plain._np_random = real_generator
    finally:
        plain.close()


@pytest.mark.parametrize(("env_class", "model_id"), MODEL_ENVS)
def test_heading_is_the_free_root_body_yaw(env_class, model_id):
    """The command frame's heading: a world yaw applied to the free root turns ``_heading()`` by that yaw."""
    env = env_class(reset_noise_scale=0.0)
    try:
        env.reset(seed=0)
        before = env._heading()
        yaw = 0.7
        turned = np.empty(4)
        mujoco.mju_mulQuat(turned, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), env.data.qpos[3:7].copy())
        env.data.qpos[3:7] = turned
        mujoco.mj_forward(env.model, env.data)
        assert (env._heading() - before - yaw + np.pi) % (2 * np.pi) - np.pi == pytest.approx(0.0, abs=1e-9)
    finally:
        env.close()


@pytest.mark.parametrize(("env_class", "model_id"), MODEL_ENVS)
def test_the_controller_reads_the_root_heading_at_reset_and_every_step(env_class, model_id):
    """Turned well away from zero, the root's yaw is what the hook seeds the controller at (its current and first
    desired heading) and what every step's update reads, so a command is body-relative on every species."""
    config = DirectionCommandConfig(switch_interval_s=0.05, straight_probability=0.0, turn_increment_max=0.6)
    env = env_class(command_mode="heading_and_speed", command_config=config)
    try:
        env.reset(seed=3)
        yaw = 1.1
        turned = np.empty(4)
        mujoco.mju_mulQuat(turned, np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), env.data.qpos[3:7].copy())
        env.data.qpos[3:7] = turned
        mujoco.mj_forward(env.model, env.data)
        env._command = env._draw_episode_command()  # the hook, as reset() runs it after the pose
        heading = wrap_angle(env._heading())
        assert abs(heading) > 0.5
        assert env._command_state.current_heading == heading == env._command_state.desired_heading
        for _ in range(5):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert env._command_state.current_heading == wrap_angle(env._heading())
    finally:
        env.close()


@pytest.mark.parametrize("env_class", [BaseDinoEnv] + [param.values[0] for param in MODEL_ENVS])
def test_every_constructor_takes_command_mode_and_one_command_config(env_class):
    """Decision D-D2 on every signature: a constructor that kept a Phase C kwarg would put it into one species'
    effective [env] and move that species' task_sha256 alone."""
    parameters = inspect.signature(env_class).parameters
    assert parameters["command_mode"].default == "none" and parameters["command_config"].default is None
    assert not set(RETIRED_COMMAND_ENV_KEYS) & set(parameters)


@pytest.mark.parametrize("retired", RETIRED_COMMAND_ENV_KEYS)
@pytest.mark.parametrize(("env_class", "model_id"), MODEL_ENVS)
def test_every_model_constructor_refuses_each_retired_command_kwarg(env_class, model_id, retired):
    """D-D2 strictly (the maintainer's choice of 2026-10-05): a retired kwarg is a TypeError on every species, at
    its Phase C default too, so an env rebuilt from a record written before PR-9 fails loudly."""
    value = [0.0, 0.0] if retired.endswith("_range") else 0.0
    with pytest.raises(TypeError, match=retired):
        env_class(**{retired: value})


def test_live_modes_build_the_controller_and_refuse_settings_they_would_ignore():
    """Phase D (consolidation PR-9, D-D2): both live modes build the controller from ONE command_config; a config
    without a live mode, a live mode without a config (the dataclass defaults are T. rex-scale) and speed variation
    under "heading" (which holds cruise_speed) are refused at construction rather than ignored."""
    config = DirectionCommandConfig(cruise_speed=0.5)
    for env_class, mode in (
        (TyrannosaurusRexAnatomicalEnv, "heading"),
        (CompsognathusLongipesAnatomicalEnv, "heading_and_speed"),
    ):
        env = env_class(command_mode=mode, command_config=config)
        try:
            assert env.command_mode == mode and env.command_config is config
            assert env.direction_controller is not None and env.direction_controller.config is config
            assert env.command_manifest() == DirectionCommandController(config).manifest()
        finally:
            env.close()
    plain = TyrannosaurusRexAnatomicalEnv()
    try:
        assert plain.command_config is None and plain.direction_controller is None
        assert plain.command_manifest() is None
    finally:
        plain.close()
    with pytest.raises(ValueError, match="command_config needs a live command_mode"):
        TyrannosaurusRexAnatomicalEnv(command_config=config)
    for mode in ("heading", "heading_and_speed"):
        with pytest.raises(ValueError, match=f"command_mode='{mode}' needs a command_config"):
            TyrannosaurusRexAnatomicalEnv(command_mode=mode)
    for varied in (replace(config, speed_range=(0.25, 0.5)), replace(config, stop_probability=0.1)):
        with pytest.raises(ValueError, match="'heading' holds cruise_speed"):
            TyrannosaurusRexAnatomicalEnv(command_mode="heading", command_config=varied)
        TyrannosaurusRexAnatomicalEnv(command_mode="heading_and_speed", command_config=varied).close()
    with pytest.raises(TypeError, match="DirectionCommandConfig"):
        TyrannosaurusRexAnatomicalEnv(command_mode="heading_and_speed", command_config={"cruise_speed": 0.5})
    with pytest.raises(ValueError, match=re.escape(f"command_mode='bogus' is not one of {COMMAND_MODES}")):
        TyrannosaurusRexAnatomicalEnv(command_mode="bogus")
    for removed in RETIRED_COMMAND_ENV_KEYS:
        with pytest.raises(TypeError, match=removed):
            TyrannosaurusRexAnatomicalEnv(**{removed: 0.0})


def test_a_live_mode_draws_after_the_push_schedule():
    """Decision D-C5 on a pushed stage: the schedule seed is the reset stream's draw after the push schedule's, so
    the pushes, the pose and every other reset draw equal the "none" twin's."""
    kwargs = dict(load_stage_config(TYRANNOSAURUS_REX, "recovery")["env_kwargs"])
    config = DirectionCommandConfig(switch_interval_s=0.05, straight_probability=0.0, turn_increment_max=0.6)
    live = TyrannosaurusRexAnatomicalEnv(command_mode="heading_and_speed", command_config=config, **kwargs)
    none = TyrannosaurusRexAnatomicalEnv(**kwargs)
    try:
        for seed in (11, 12):
            live.reset(seed=seed)
            none.reset(seed=seed)
            assert len(none._push_schedule_starts) > 0, "the recovery stage schedules pushes"
            np.testing.assert_array_equal(live._push_schedule_starts, none._push_schedule_starts)
            np.testing.assert_array_equal(live._push_schedule_directions, none._push_schedule_directions)
            np.testing.assert_array_equal(live.data.qpos, none.data.qpos)
            expected = copy.deepcopy(none.np_random)
            assert live.direction_controller.schedule_seed == int(expected.integers(0, 2**32, dtype=np.uint64))
            assert live.np_random.bit_generator.state == expected.bit_generator.state
    finally:
        live.close()
        none.close()


@pytest.mark.parametrize(("env_class", "model_id"), MODEL_ENVS)
def test_a_live_mode_seeds_at_reset_and_changes_only_the_command_segment(env_class, model_id):
    """The controller is seeded once per reset from ONE draw appended to the reset stream, the observation
    carries its normalised command, step() rewards before it advances and returns the next command; physics,
    reward and termination match the same seeds under "none"."""
    config = DirectionCommandConfig(switch_interval_s=0.05, straight_probability=0.0, turn_increment_max=0.6)
    live = env_class(command_mode="heading_and_speed", command_config=config)
    none = env_class()
    rng = np.random.default_rng(5)
    actions = rng.uniform(-0.05, 0.05, (12, *live.action_space.shape)).astype(np.float32)
    try:
        streams = []
        for seed in (11, 11, 12):
            obs, info = live.reset(seed=seed)
            plain_obs, plain_info = none.reset(seed=seed)
            expected = copy.deepcopy(none.np_random)
            schedule_seed = int(expected.integers(0, 2**32, dtype=np.uint64))
            assert live.direction_controller.schedule_seed == schedule_seed
            assert live.np_random.bit_generator.state == expected.bit_generator.state
            assert info == plain_info
            np.testing.assert_array_equal(live.data.qpos, none.data.qpos)
            np.testing.assert_array_equal(obs[:-COMMAND_WIDTH], plain_obs[:-COMMAND_WIDTH])
            np.testing.assert_array_equal(obs[-COMMAND_WIDTH:], live._command_state.normalized)
            assert obs[-COMMAND_WIDTH] > 0.0 and np.all(np.abs(obs[-COMMAND_WIDTH:]) <= 1.0)
            assert live._command.flags.writeable and live._command is not live._command_state.normalized
            rows = [obs.copy()]
            for action in actions:
                executed = live._command_state
                stepped, reward, terminated, truncated, step_info = live.step(action)
                plain, plain_reward, plain_terminated, plain_truncated, plain_step_info = none.step(action)
                assert (reward, terminated, truncated) == (plain_reward, plain_terminated, plain_truncated)
                assert step_info == plain_step_info
                np.testing.assert_array_equal(stepped[:-COMMAND_WIDTH], plain[:-COMMAND_WIDTH])
                assert live._command_state.time_s == pytest.approx(live._step_count * live.dt)
                assert live._command_state.event_id >= executed.event_id
                np.testing.assert_array_equal(stepped[-COMMAND_WIDTH:], live._command_state.normalized)
                assert live._command.flags.writeable and live._command is not live._command_state.normalized
                rows.append(stepped.copy())
                if terminated or truncated:
                    break
            assert live._command_state.event_id > 0, "the switch schedule fired"
            streams.append((schedule_seed, rows))
        assert streams[0][0] == streams[1][0] != streams[2][0]
        for first, second in zip(streams[0][1], streams[1][1], strict=True):
            np.testing.assert_array_equal(first, second)
    finally:
        live.close()
        none.close()


# ---------------------------------------------------------------------------
# Task fingerprint (amendments A1, A9)
# ---------------------------------------------------------------------------


def _fingerprint(**overrides):
    kwargs = dict(
        model_id=TYRANNOSAURUS_REX,
        stage=1,
        backend="stable-baselines3",
        env_kwargs=_TYRANNOSAURUS_REX_ENV,
        plant_identity=_FAKE_PLANT,
        perturbation_manifest=None,
    )
    kwargs.update(overrides)
    return compute_task_fingerprint(**kwargs)


def test_committed_stages_carry_no_command_keys_and_no_command_section():
    """(i): under the committed TOMLs every stage's env section is command-free."""
    for model_id in model_ids():
        identity = current_plant_identity(model_id, verify_generated=False).to_dict()
        for stage_key, stage_config in load_all_stages(model_id).items():
            assert not COMMAND_ENV_KEYS_SET & set(stage_config["env_kwargs"]), (model_id, stage_key)
            payload = derive_stage_task_fingerprint(
                model_id=model_id,
                stage=stage_key,
                backend="stable-baselines3",
                env_kwargs=stage_config["env_kwargs"],
                plant_identity=identity,
            )
            assert not COMMAND_ENV_KEYS_SET & set(payload["env"]), (model_id, stage_key)
            assert "command" not in payload, (model_id, stage_key)


def test_command_kwargs_enter_the_task_fingerprint_and_a_live_mode_has_a_command_section():
    base = _fingerprint()
    assert not COMMAND_ENV_KEYS_SET & set(base["env"])
    assert "command" not in base
    # An explicit "none" is carved out too, whatever config it names: same hash.
    config = DirectionCommandConfig(cruise_speed=0.5, speed_range=(0.25, 0.75))
    explicit_none = _fingerprint(
        env_kwargs={**_TYRANNOSAURUS_REX_ENV, "command_mode": "none", "command_config": config}
    )
    assert explicit_none["task_sha256"] == base["task_sha256"]

    # (ii): a live mode keeps both keys, the config field by field (decision D-D2), moves the hash, and its
    # controller's manifest is the command section.
    live_env = {**_TYRANNOSAURUS_REX_ENV, "command_mode": "heading_and_speed", "command_config": config}
    manifest = DirectionCommandController(config).manifest()
    live = _fingerprint(env_kwargs=live_env, command_manifest=manifest)
    assert COMMAND_ENV_KEYS_SET <= set(live["env"])
    assert live["env"]["command_mode"] == "heading_and_speed"
    assert live["env"]["command_config"] == {**asdict(config), "speed_range": [0.25, 0.75]}
    assert live["command"] == json.loads(json.dumps(manifest))
    assert live["task_sha256"] != base["task_sha256"]
    retuned = replace(config, cruise_speed=0.6)
    other = _fingerprint(
        env_kwargs={**live_env, "command_config": retuned},
        command_manifest=DirectionCommandController(retuned).manifest(),
    )
    assert other["task_sha256"] != live["task_sha256"]
    # The manifest alone moves the hash: its schema and adapter version direction_commands.py.
    bumped = _fingerprint(env_kwargs=live_env, command_manifest={**manifest, "adapter": manifest["adapter"] + "+next"})
    assert bumped["env"] == live["env"] and bumped["task_sha256"] != live["task_sha256"]
    derived = derive_stage_task_fingerprint(
        model_id=TYRANNOSAURUS_REX,
        stage=1,
        backend="stable-baselines3",
        env_kwargs=live_env,
        plant_identity=_FAKE_PLANT,
        command_manifest=manifest,
    )
    assert derived == live

    # (iii): a manifest goes with a live mode and only with one: neither half is hashed without the other.
    for env_kwargs, command_manifest in ((_TYRANNOSAURUS_REX_ENV, manifest), (live_env, None)):
        with pytest.raises(TaskFingerprintError, match="exactly when command_mode is not 'none'"):
            _fingerprint(env_kwargs=env_kwargs, command_manifest=command_manifest)

    # (iv): no manifest while command_mode is "none"; under a live mode, the controller's, with the same config.
    env = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0)
    live_env = TyrannosaurusRexAnatomicalEnv(
        reset_noise_scale=0.0, command_mode="heading_and_speed", command_config=config
    )
    try:
        assert env.command_manifest() is None
        manifest = live_env.command_manifest()
        assert manifest == DirectionCommandController(config).manifest()
        assert json.loads(json.dumps(manifest["config"])) == live["env"]["command_config"]
        assert manifest["schema"] == "mesozoic.direction-commands/v1"
    finally:
        env.close()
        live_env.close()


@pytest.mark.parametrize("mode", ["none", "heading_and_speed"])
@pytest.mark.parametrize("retired", RETIRED_COMMAND_ENV_KEYS)
def test_the_task_fingerprint_refuses_each_retired_command_kwarg_by_name(retired, mode):
    """A record written before PR-9 carries the five at their inert values; the fingerprint refuses each by name,
    in every mode, rather than hash it into a task that matches nothing (the reader drops them first)."""
    value = [0.0, 0.0] if retired.endswith("_range") else 0.0
    env_kwargs: dict[str, Any] = {**_TYRANNOSAURUS_REX_ENV, "command_mode": mode, retired: value}
    manifest = None
    if mode != "none":
        config = DirectionCommandConfig(cruise_speed=0.5)
        env_kwargs["command_config"] = config
        manifest = DirectionCommandController(config).manifest()
    message = (
        rf"^\[env\] {retired}: retired by decision D-D2 \(consolidation PR-9\).*a record written before PR-9 "
        r"carries them, so drop command_frame\.RETIRED_COMMAND_ENV_KEYS from its env kwargs first$"
    )
    with pytest.raises(TaskFingerprintError, match=message):
        _fingerprint(env_kwargs=env_kwargs, command_manifest=manifest)
    with pytest.raises(TaskFingerprintError, match=message):
        stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=env_kwargs, command_manifest=manifest)


#: What ``save_stage_config`` recorded for the five retired kwargs before PR-9: their inert defaults.
_PRE_PR9_RETIRED_RECORD = {
    "command_speed_range": [0.0, 0.0],
    "command_lateral_range": [0.0, 0.0],
    "command_yaw_rate_max": 0.0,
    "command_switch_interval": 0.0,
    "command_switch_jitter": 0.0,
}
#: The compsognathus push knobs the quiet-task encoding drops only when none is configured (task_fingerprint.py).
_COMPSOGNATHUS_LONGIPES_PUSH_KEYS = (
    "perturbation_capture_velocity_multiple",
    "perturbation_interval",
    "perturbation_jitter",
    "perturbation_duration",
    "perturbation_direction",
)
_STAGES = [
    pytest.param(model_id, entry.id, id=f"{model_id.species}-{model_id.embodiment}-{entry.id}")
    for model_id in model_ids()
    for entry in load_stage_manifest(model_id).stages
]


@pytest.mark.parametrize(("model_id", "stage"), _STAGES)
def test_a_record_written_before_pr9_rebuilds_its_stage_once_the_retired_keys_are_dropped(tmp_path, model_id, stage):
    """Runs recorded before PR-9 are not kept rebuildable from their recorded constructor kwargs (the maintainer's
    choice of 2026-10-05): the constructors refuse the five, and the fingerprint, by this PR's default, refuses them
    by name.  A reader that drops RETIRED_COMMAND_ENV_KEYS builds the env and re-derives the task the record names,
    as before PR-9."""
    env_class = get_species_config(model_id).env_class
    path = save_stage_config(
        tmp_path, stage, load_stage_config(model_id, stage), "PPO", env_class=env_class, model_id=model_id
    )
    record = json.loads(path.read_text())["reward_weights"]
    assert record.pop("command_config") is None and record["command_mode"] == "none"
    recorded = {**record, **_PRE_PR9_RETIRED_RECORD}  # this stage's record as save_stage_config wrote it before PR-9
    with pytest.raises(TypeError, match="command_"):
        env_class(**recorded)
    with pytest.raises(TaskFingerprintError, match="RETIRED_COMMAND_ENV_KEYS"):
        stage_task_fingerprint(model_id, stage, env_kwargs=recorded)
    kept = {key: value for key, value in recorded.items() if key not in RETIRED_COMMAND_ENV_KEYS}
    env_class(**kept).close()
    committed = stage_task_fingerprint(model_id, stage)["task_sha256"]
    derived = stage_task_fingerprint(model_id, stage, env_kwargs=kept)["task_sha256"]
    if model_id.species == "compsognathus_longipes" and kept["perturbation_capture_velocity_multiple"] == 0.0:
        # Unchanged by PR-9: the record writes out compsognathus's disabled push knobs, which the quiet-task
        # encoding keeps only when none is configured, so it names this task with those knobs explicit.
        explicit = {key: kept[key] for key in _COMPSOGNATHUS_LONGIPES_PUSH_KEYS}
        toml_env = load_stage_config(model_id, stage)["env_kwargs"]
        assert derived == stage_task_fingerprint(model_id, stage, env_kwargs={**toml_env, **explicit})["task_sha256"]
        kept = {key: value for key, value in kept.items() if key not in _COMPSOGNATHUS_LONGIPES_PUSH_KEYS}
        derived = stage_task_fingerprint(model_id, stage, env_kwargs=kept)["task_sha256"]
    assert derived == committed


# ---------------------------------------------------------------------------
# Recovery-calibration restamp (decision D-C2, amendment A13)
# ---------------------------------------------------------------------------

CALIBRATED_MODELS = (COMPSOGNATHUS_LONGIPES, COMPSOGNATHUS_LONGIPES_ROBOT)
#: Every committed profile carries two restamps.  The first is its interface restamp: the robot's is the Phase C
#: restamp; the anatomical profile was re-measured on compsognathus physics r2 (D-D26; ``calibrate_recovery.py``),
#: a physics change being exactly what a restamp refuses, and restamped once since, for its heading-free policy
#: interface r4 (plant_versions note 18, D-D31), an interface-only revision on the physics it was measured on.  The
#: second is the identity migration of decision D-D33 (plant_versions note 19, ``--migrate-identity``): an
#: interface-only revision that also renamed the profile's model, so its entry records the identity it replaced.
RESTAMPED_MODELS = (COMPSOGNATHUS_LONGIPES, COMPSOGNATHUS_LONGIPES_ROBOT)
#: Each first restamp's reason (the tool's default is the Phase C one).
RESTAMP_REASONS = {
    COMPSOGNATHUS_LONGIPES: (
        "compsognathus policy interface r3 -> r4 (configs/plant_versions.toml note 18, D-D31): heading-free "
        "observations bipedal-target-heading-free/v1; physics r2 and the recovery [env] block unchanged; the "
        "fixed-command nulls do not read the observation and the reset draw stream is unchanged "
        "(tests/fixtures/phase_c_reset_golden.json)"
    ),
}
#: The identity migration's recorded reason, and the one schema it migrated from (decision D-D33).
IDENTITY_MIGRATION_REASON = "plant_versions note 19 (D-D33)"
PRE_D_D33_PROFILE_SCHEMA = "mesozoic.recovery-calibration/v1"
CONFIGS_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _committed_profile(model_id: ModelId) -> tuple[Path, str, dict[str, Any]]:
    path = model_id.path(CONFIGS_ROOT) / "recovery_calibration.json"
    text = path.read_text(encoding="utf-8")
    return path, text, json.loads(text)


def _render(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, ensure_ascii=False) + "\n"


def _rolled_back(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """*profile* with its last restamp undone: the plant identity's interface fields and the task hash it replaced.

    The D-D33 migration moved no observation width, so ``observation_dim`` stays.
    """
    profile = copy.deepcopy(profile)
    history = profile["restamp_history"]
    entry = history.pop()
    identity = dict(profile["plant_identity"])
    identity["policy_interface_revision"] = entry["previous_policy_interface_revision"]
    identity["policy_interface_sha256"] = entry["previous_policy_interface_sha256"]
    profile["plant_identity"] = identity
    profile["task_sha256"] = entry["previous_task_sha256"]
    if not history:
        del profile["restamp_history"]
    return profile, entry


def _write(model_id: ModelId, root: Path, profile: dict[str, Any]) -> Path:
    path = model_id.path(root) / "recovery_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(profile), encoding="utf-8")
    return path


def _pre_restamp_copy(model_id: ModelId, root: Path) -> tuple[Path, dict[str, Any]]:
    """The committed profile one interface revision behind, still at the current schema (what a plain restamp
    finds after an interface-only revision)."""
    profile, entry = _rolled_back(_committed_profile(model_id)[2])
    return _write(model_id, root, profile), entry


def _pre_migration_copy(model_id: ModelId, root: Path) -> tuple[Path, dict[str, Any]]:
    """The committed profile rolled back to the schema-v1 profile ``--migrate-identity`` found (decision D-D33).

    The schema, species and ``calibration_id`` come from the migration entry's ``previous_identity``; the
    measurement artifact is the recorded path under the recorded species' directory.  Of the replaced plant
    identity, the tool reads only the physics digest and the interface fields its entry records.
    """
    rolled, entry = _rolled_back(_committed_profile(model_id)[2])
    previous = entry["previous_identity"]
    profile: dict[str, Any] = {}
    for key, value in rolled.items():
        if key == "schema":
            profile[key] = previous["schema"]
        elif key == "species":
            profile[key] = previous["species"]
        elif key == "embodiment":
            continue
        elif key == "calibration_id":
            profile[key] = previous["calibration_id"]
        elif key == "provenance":
            artifact = Path("environments", previous["species"], "data", Path(value["measurement_artifact"]).name)
            profile[key] = {**value, "measurement_artifact": artifact.as_posix()}
        else:
            profile[key] = value
    return _write(model_id, root, profile), entry


def _committed_copy(model_id: ModelId, root: Path) -> Path:
    path = model_id.path(root) / "recovery_calibration.json"
    path.parent.mkdir(parents=True)
    shutil.copy(model_id.path(CONFIGS_ROOT) / "recovery_calibration.json", path)
    return path


@pytest.mark.parametrize("model_id", RESTAMPED_MODELS, ids=lambda m: f"{m.species}-{m.embodiment}")
def test_committed_recovery_calibrations_load_and_record_their_two_restamps(model_id):
    from environments.shared.recovery_calibration import load_recovery_calibration
    from environments.shared.scripts.restamp_recovery_calibration import RESTAMP_REASON

    calibration = load_recovery_calibration(model_id)
    _, _, profile = _committed_profile(model_id)
    assert ModelId.from_record(calibration.profile) == model_id
    assert calibration.profile["restamp_history"] == profile["restamp_history"]
    identity = current_plant_identity(model_id, verify_generated=False)
    assert profile["plant_identity"] == identity.to_dict()
    interface, migration = profile["restamp_history"]
    assert interface["previous_policy_interface_revision"] == identity.policy_interface_revision - 2
    assert interface["reason"] == RESTAMP_REASONS.get(model_id, RESTAMP_REASON)
    assert "previous_identity" not in interface
    # The identity migration (decision D-D33): one interface revision, the replaced identity recorded whole.
    assert migration["previous_policy_interface_revision"] == identity.policy_interface_revision - 1
    assert migration["previous_policy_interface_sha256"] != identity.policy_interface_sha256
    assert migration["previous_task_sha256"] != profile["task_sha256"]
    assert migration["reason"] == IDENTITY_MIGRATION_REASON
    previous = migration["previous_identity"]
    assert set(previous) == {"schema", "species", "calibration_id"}
    assert previous["schema"] == PRE_D_D33_PROFILE_SCHEMA and previous["species"] != model_id.species
    assert profile["calibration_id"] == (
        f"{model_id.species}-{model_id.embodiment}-"
        + previous["calibration_id"].removeprefix(previous["species"] + "-")
    )
    assert profile["provenance"]["measurement_artifact"].startswith(f"environments/{model_id.species}/data/")
    for entry in (interface, migration):
        assert re.fullmatch(r"[0-9a-f]{40}", entry["restamped_at_commit"])


def test_the_anatomical_recovery_calibration_is_a_physics_r2_measurement_with_one_heading_interface_restamp(
    tmp_path,
):
    """Measured on the current physics, then restamped once for the heading-free interface and once for the
    identity migration (decision D-D33): it loads, names the physics it was measured on, measured the committed
    recovery [env] block, carries the restamp from policy interface r3, and the restamp tool leaves it
    byte-identical."""
    from environments.shared.recovery_calibration import load_recovery_calibration
    from environments.shared.scripts import restamp_recovery_calibration as tool

    calibration = load_recovery_calibration(COMPSOGNATHUS_LONGIPES)
    _, text, profile = _committed_profile(COMPSOGNATHUS_LONGIPES)
    identity = current_plant_identity(COMPSOGNATHUS_LONGIPES, verify_generated=False)
    assert profile["plant_identity"] == identity.to_dict()
    assert profile["plant_identity"]["physics_revision"] == 2
    entry, migration = calibration.profile["restamp_history"]
    # Re-pinned (decision D-D33): r3 -> r4 heading-free, then r4 -> r5 for the identity migration.
    assert (entry["previous_policy_interface_revision"], identity.policy_interface_revision) == (3, 5)
    assert migration["previous_policy_interface_revision"] == 4
    assert "heading-free" in entry["reason"] and "note 18" in entry["reason"]
    measured_env = json.loads(json.dumps(load_stage_config(COMPSOGNATHUS_LONGIPES, "recovery")["env_kwargs"]))
    assert profile["recovery_env_kwargs"] == measured_env
    path = _committed_copy(COMPSOGNATHUS_LONGIPES, tmp_path / "configs")
    assert tool.restamp_recovery_calibration(COMPSOGNATHUS_LONGIPES, path=path) is False
    assert path.read_text(encoding="utf-8") == text


def test_restamp_recovery_calibration_is_idempotent_and_refuses_a_physics_change(tmp_path):
    from environments.shared.scripts import restamp_recovery_calibration as tool

    root = tmp_path / "configs"
    # Roll the committed profile back one interface revision: a plain restamp
    # must reproduce the committed bytes, except that its entry names no
    # replaced identity (only the D-D33 migration renamed the model).  The
    # recorded commit is passed explicitly so the replay does not depend on
    # the checkout's HEAD.
    for model_id in RESTAMPED_MODELS:
        path, entry = _pre_restamp_copy(model_id, root)
        assert (
            tool.restamp_recovery_calibration(
                model_id, path=path, commit=entry["restamped_at_commit"], reason=entry["reason"]
            )
            is True
        )
        expected = _committed_profile(model_id)[2]
        del expected["restamp_history"][-1]["previous_identity"]
        assert path.read_text(encoding="utf-8") == _render(expected), model_id
    for model_id in (model_id for model_id in CALIBRATED_MODELS if model_id not in RESTAMPED_MODELS):
        _committed_copy(model_id, root)

    # Second run through the CLI, over both default models: a byte no-op, no history append.
    before = {
        model_id: (model_id.path(root) / "recovery_calibration.json").read_bytes() for model_id in CALIBRATED_MODELS
    }
    assert tool.main(["--configs-root", str(root)]) == 0
    for model_id in CALIBRATED_MODELS:
        after = (model_id.path(root) / "recovery_calibration.json").read_bytes()
        assert after == before[model_id], model_id
        assert len(json.loads(after).get("restamp_history", [])) == 2 * int(model_id in RESTAMPED_MODELS)

    # A physics change needs a real recalibration, not a restamp.
    path = COMPSOGNATHUS_LONGIPES.path(root) / "recovery_calibration.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["plant_identity"]["physics_sha256"] = "sha256:" + "0" * 64
    profile["task_sha256"] = "sha256:" + "1" * 64
    foreign = json.dumps(profile, indent=2, ensure_ascii=False) + "\n"
    path.write_text(foreign, encoding="utf-8")
    with pytest.raises(tool.RestampError, match="physics change"):
        tool.restamp_recovery_calibration(COMPSOGNATHUS_LONGIPES, path=path)
    assert tool.main(["--model-id", "compsognathus_longipes", "anatomical", "--configs-root", str(root)]) == 1
    assert path.read_text(encoding="utf-8") == foreign  # never writes on a refusal

    # A changed task (recovery_env_kwargs) is refused the same way.
    shutil.copy(COMPSOGNATHUS_LONGIPES.path(CONFIGS_ROOT) / "recovery_calibration.json", path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["recovery_env_kwargs"]["perturbation_interval"] = 99.0
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(tool.RestampError, match="recovery_env_kwargs differ"):
        tool.restamp_recovery_calibration(COMPSOGNATHUS_LONGIPES, path=path)

    # An unknown model is a refusal (the function's RestampError; the CLI's usage error, exit 2), not a traceback.
    # (The function checks the profile names the model before it reads the model's stage config.)
    with pytest.raises(tool.RestampError, match="is not a recovery calibration of stegosaurus_stenops/anatomical"):
        tool.restamp_recovery_calibration(ModelId("stegosaurus_stenops", "anatomical"), path=path)
    with pytest.raises(SystemExit) as usage:
        tool.main(["--model-id", "stegosaurus_stenops", "anatomical", "--configs-root", str(root)])
    assert usage.value.code == 2


def test_restamp_refuses_the_other_embodiments_profile_and_one_that_names_half_its_model(tmp_path):
    """Decision D-D33 (review H3): the restamp target is the model, both fields; never matched on species."""
    from environments.shared.scripts import restamp_recovery_calibration as tool

    robot_profile = _committed_profile(COMPSOGNATHUS_LONGIPES_ROBOT)[2]
    path = _write(COMPSOGNATHUS_LONGIPES, tmp_path, robot_profile)
    with pytest.raises(tool.RestampError, match="is not a recovery calibration of compsognathus_longipes/anatomical"):
        tool.restamp_recovery_calibration(COMPSOGNATHUS_LONGIPES, path=path)
    halved = {key: value for key, value in _committed_profile(COMPSOGNATHUS_LONGIPES)[2].items() if key != "embodiment"}
    path.write_text(_render(halved), encoding="utf-8")
    with pytest.raises(tool.RestampError, match="does not name its model.*embodiment"):
        tool.restamp_recovery_calibration(COMPSOGNATHUS_LONGIPES, path=path)
    # A schema-v1 profile is the migration's, never a plain restamp's.
    path, _ = _pre_migration_copy(COMPSOGNATHUS_LONGIPES, tmp_path / "v1")
    with pytest.raises(tool.RestampError, match="--migrate-identity migrates it"):
        tool.restamp_recovery_calibration(COMPSOGNATHUS_LONGIPES, path=path)


@pytest.mark.parametrize("model_id", CALIBRATED_MODELS, ids=lambda m: f"{m.species}-{m.embodiment}")
def test_migrate_identity_reproduces_the_committed_profile_from_its_schema_v1_form(model_id, tmp_path):
    """``--migrate-identity`` (decision D-D33, plan section 3.5) rebuilds the committed profile byte for byte from
    the schema-v1 profile it found, and a second run is a byte no-op."""
    from environments.shared.scripts import restamp_recovery_calibration as tool

    path, entry = _pre_migration_copy(model_id, tmp_path / "configs")
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == PRE_D_D33_PROFILE_SCHEMA
    assert (
        tool.migrate_recovery_calibration_identity(
            model_id, path=path, commit=entry["restamped_at_commit"], reason=entry["reason"]
        )
        is True
    )
    committed = _committed_profile(model_id)[1]
    assert path.read_text(encoding="utf-8") == committed
    assert tool.migrate_recovery_calibration_identity(model_id, path=path, reason=entry["reason"]) is False
    assert path.read_text(encoding="utf-8") == committed


def test_migrate_identity_refuses_another_models_profile_and_a_changed_plant_or_task(tmp_path):
    from environments.shared.scripts import restamp_recovery_calibration as tool

    def refused(model_id: ModelId, path: Path, match: str) -> None:
        before = path.read_bytes()
        with pytest.raises(tool.RestampError, match=match):
            tool.migrate_recovery_calibration_identity(model_id, path=path, reason=IDENTITY_MIGRATION_REASON)
        assert path.read_bytes() == before  # never writes on a refusal

    # The robot's schema-v1 profile offered for the anatomical model: the identity is the directory's, checked by
    # the measured task and the physics digest, so the other embodiment's profile is refused (review H3).
    path, _ = _pre_migration_copy(COMPSOGNATHUS_LONGIPES_ROBOT, tmp_path / "robot")
    refused(COMPSOGNATHUS_LONGIPES, path, "recovery_env_kwargs differ|not this model's")
    path, _ = _pre_migration_copy(COMPSOGNATHUS_LONGIPES, tmp_path / "physics")
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["plant_identity"]["physics_sha256"] = "sha256:" + "0" * 64
    path.write_text(_render(profile), encoding="utf-8")
    refused(COMPSOGNATHUS_LONGIPES, path, "not this model's")
    path, _ = _pre_migration_copy(COMPSOGNATHUS_LONGIPES, tmp_path / "task")
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["recovery_env_kwargs"]["perturbation_interval"] = 99.0
    path.write_text(_render(profile), encoding="utf-8")
    refused(COMPSOGNATHUS_LONGIPES, path, "recovery_env_kwargs differ")
    # A schema-v1 profile that already names an embodiment is not guessed at.
    path, _ = _pre_migration_copy(COMPSOGNATHUS_LONGIPES, tmp_path / "half")
    profile = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(_render({**profile, "embodiment": "anatomical"}), encoding="utf-8")
    refused(COMPSOGNATHUS_LONGIPES, path, "already records an embodiment")
    # A current-schema profile is migrated only when it is already the model's current one (a byte no-op).
    path, _ = _pre_restamp_copy(COMPSOGNATHUS_LONGIPES, tmp_path / "behind")
    refused(COMPSOGNATHUS_LONGIPES, path, "only a schema-v1 profile")
    # The CLI mode records its own reason, never the Phase C default.
    with pytest.raises(SystemExit) as usage:
        tool.main(["--migrate-identity", "--configs-root", str(tmp_path / "robot")])
    assert usage.value.code == 2


def test_restamp_recovery_calibration_records_the_given_reason(tmp_path):
    """A later interface-only revision must name its own reason; the Phase C text is only the default."""
    from environments.shared.scripts import restamp_recovery_calibration as tool
    from environments.shared.scripts.restamp_recovery_calibration import RESTAMP_REASON

    root = tmp_path / "named"
    path, entry = _pre_restamp_copy(COMPSOGNATHUS_LONGIPES_ROBOT, root)
    assert (
        tool.main(
            ["--model-id", "compsognathus_longipes", "robot", "--configs-root", str(root), "--reason", "r99 bump"]
        )
        == 0
    )
    history = json.loads(path.read_text(encoding="utf-8"))["restamp_history"]
    assert [item["reason"] for item in history] == [RESTAMP_REASON, "r99 bump"]
    assert history[-1]["previous_task_sha256"] == entry["previous_task_sha256"]

    path, _ = _pre_restamp_copy(COMPSOGNATHUS_LONGIPES_ROBOT, tmp_path / "blank")
    with pytest.raises(tool.RestampError, match="non-empty"):
        tool.restamp_recovery_calibration(COMPSOGNATHUS_LONGIPES_ROBOT, path=path, reason="  ")
