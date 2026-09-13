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
  calibrations are restamped, not re-measured) and is compared exactly at
  6 decimals; so is its ``second_reset`` record (an unseeded ``reset()``
  after the trajectory, the way every later training episode starts), which
  catches a draw the hook consumed and discarded.  Its ``observation``
  record is a prefix contract (byte-inert except the trailing zeros).  Its
  ``trajectory`` half (qpos after each zero-action step) crosses the
  solver, so it is ADVISORY: ``assert_allclose(atol=1e-5)`` and skipped
  unless the running MuJoCo is the fixture's (amendment A4).
* The command hook itself is pinned directly: under ``command_mode =
  "none"`` it touches ``np_random`` not at all, and it runs after every
  reset draw (the push schedule included) and before ``_get_obs``.
* Every species' observation ends with three zeros under ``command_mode =
  "none"`` and matches the plant identity width; the segment is the last
  ``observation_segments`` entry on all six species (decision D-C1).
* ``command_mode = "none"`` is inert and deterministic; every live mode is
  refused on the SB3 backend until Phase D (decision D-C7).
* The six command kwargs are carved out of the task fingerprint while the
  effective ``command_mode`` is ``"none"`` (amendment A1) and a ``command``
  payload section exists only when a manifest is passed (amendment A9).
* The recovery-calibration restamp tool is idempotent and refuses a physics
  change; the committed compsognathus calibrations load (decision D-C2).
* The velociraptor mjlab registration's ``obs_dim`` tracks the plant
  identity (amendment A5).

Nothing here imports stable_baselines3, torch or jax, so the module runs in
full under the shared matrix job (``.[dev]`` only) as well as the SB3 job.
The MJX counterparts live in ``test_mjx_phase_c_interface.py`` and the
dual-backend probe parity pin in ``test_plant_contract_phase_c.py``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.compsognathus.envs.compsognathus_env import CompsognathusEnv, CompsognathusRobotEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.shared.command_frame import (
    COMMAND_COMPONENTS,
    COMMAND_ENV_KEYS,
    COMMAND_MODES,
    COMMAND_SEGMENT_NAME,
    COMMAND_WIDTH,
)
from environments.shared.config import SPECIES_NAMES, load_all_stages
from environments.shared.plant_contract import current_plant_identity, load_plant_versions
from environments.shared.plant_contract.digests import _canonical_float
from environments.shared.plant_contract.policy_layer import _policy_interface_payload
from environments.shared.task_fingerprint import compute_task_fingerprint, derive_stage_task_fingerprint
from environments.shared.tests import reset_golden
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

SPECIES_ENVS = [
    pytest.param(RaptorEnv, "velociraptor", id="velociraptor"),
    pytest.param(TRexEnv, "trex", id="trex"),
    pytest.param(BrachioEnv, "brachiosaurus", id="brachiosaurus"),
    pytest.param(DibothrosuchusEnv, "dibothrosuchus", id="dibothrosuchus"),
    pytest.param(CompsognathusEnv, "compsognathus", id="compsognathus"),
    pytest.param(CompsognathusRobotEnv, "compsognathus_robot", id="compsognathus_robot"),
]

_FAKE_PLANT = {
    "physics_sha256": "sha256:aaaa",
    "policy_interface_sha256": "sha256:bbbb",
    "model_path": "environments/trex/assets/trex.xml",
}
_TREX_ENV = {"alive_bonus": 1.0, "healthy_z_range": (0.70, 1.30), "max_episode_steps": 1000}
COMMAND_ENV_KEYS_SET = frozenset(COMMAND_ENV_KEYS)


# ---------------------------------------------------------------------------
# Golden fixture replay (decision D3 / D-C2, amendment A4)
# ---------------------------------------------------------------------------

GOLDEN = reset_golden.load_fixture()
GOLDEN_KEYS = sorted(GOLDEN["captures"])
_REPLAYS: dict[str, dict[str, Any]] = {}


def _replay(key: str) -> dict[str, Any]:
    """Replay one capture on the current tree (cached: the fixture halves share it)."""
    if key not in _REPLAYS:
        golden = GOLDEN["captures"][key]
        _REPLAYS[key] = reset_golden.capture(
            golden["species"],
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
    assert [(entry["species"], entry["stage"], tuple(entry["seeds"]), entry["steps"]) for entry in GOLDEN["spec"]] == [
        (species, stage, tuple(seeds), steps) for species, stage, seeds, steps in reset_golden.DEFAULT_SPEC
    ]
    assert set(GOLDEN_KEYS) == {reset_golden.capture_key(s, st) for s, st, _, _ in reset_golden.DEFAULT_SPEC}


@pytest.mark.parametrize("key", GOLDEN_KEYS)
def test_reset_draw_stream_matches_the_pre_bump_golden(key):
    """THE CONTRACT: qpos/qvel/targets/push schedules after reset(seed), exact at 6 decimals.

    A reset draw added before the existing draws breaks this by design; the
    Phase C command hook draws nothing under ``command_mode = "none"``.
    """
    golden = GOLDEN["captures"][key]
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
        if key.endswith("/recovery"):
            assert record["reset"]["push_schedule_starts"], f"{key} must exercise the push schedule"
            assert record["second_reset"]["push_schedule_starts"] != record["reset"]["push_schedule_starts"]


@pytest.mark.parametrize("key", GOLDEN_KEYS)
def test_reset_observation_is_byte_inert_except_the_trailing_zeros(key):
    golden = GOLDEN["captures"][key]
    current = _replay(key)
    for seed, record in golden["seeds"].items():
        before = np.asarray(record["observation"], dtype=np.float64)
        after = np.asarray(current["seeds"][seed]["observation"], dtype=np.float64)
        assert after.shape == (before.shape[0] + COMMAND_WIDTH,)
        np.testing.assert_array_equal(after[: before.shape[0]], before, err_msg=f"{key} seed {seed}")
        np.testing.assert_array_equal(after[before.shape[0] :], np.zeros(COMMAND_WIDTH))


@pytest.mark.parametrize("key", GOLDEN_KEYS)
def test_zero_action_trajectories_match_the_pre_bump_golden(key):
    """ADVISORY (amendment A4): stepping crosses the solver, so only the fixture's MuJoCo compares."""
    if mujoco.__version__ != GOLDEN["mujoco_version"]:
        pytest.skip(
            f"trajectory half is advisory: running mujoco {mujoco.__version__} != fixture {GOLDEN['mujoco_version']}"
        )
    golden = GOLDEN["captures"][key]
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


@pytest.mark.parametrize(("env_class", "species"), SPECIES_ENVS)
def test_every_species_observation_ends_with_three_zero_command_dims(env_class, species):
    identity = current_plant_identity(species, verify_generated=False)
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


def test_observation_segments_end_with_the_command_segment_for_every_species():
    versions = load_plant_versions()[1]
    for env_class, species in ((param.values[0], param.values[1]) for param in SPECIES_ENVS):
        env = env_class(reset_noise_scale=0.0)
        try:
            payload = _policy_interface_payload(env.model, env, versions[species])
        finally:
            env.close()
        segments = payload["observation_segments"]
        assert segments[-1] == {
            "name": COMMAND_SEGMENT_NAME,
            "width": COMMAND_WIDTH,
            "components": list(COMMAND_COMPONENTS),
            "frame": "body-relative",
            "range": [_canonical_float(-1.0), _canonical_float(1.0)],
        }, species
        assert [float(bound) for bound in segments[-1]["range"]] == [-1.0, 1.0]
        assert [segment["name"] for segment in segments].count(COMMAND_SEGMENT_NAME) == 1
        assert segments[-2]["name"].endswith("_distance") and segments[-2]["width"] == 1
        assert payload["observation"]["shape"] == [
            current_plant_identity(species, verify_generated=False).observation_dim
        ]


# ---------------------------------------------------------------------------
# command_mode on the SB3 backend (decisions D-C5, D-C7)
# ---------------------------------------------------------------------------


def test_command_mode_none_is_inert_and_deterministic_on_sb3():
    explicit_a = TRexEnv(reset_noise_scale=0.0, command_mode="none")
    explicit_b = TRexEnv(reset_noise_scale=0.0, command_mode="none")
    plain = TRexEnv(reset_noise_scale=0.0)
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

    Phase D replaces the hook body with a real draw, so the ordering half is
    what keeps the seeded draw stream (the golden fixture) intact then: the
    hook must see the NEW push schedule and the generator state the reset
    ends with, and its value must reach the reset observation.
    """
    from environments.shared.config import load_stage_config

    marker = np.asarray([0.125, -0.25, 0.5], dtype=np.float32)
    seen: dict[str, Any] = {}

    class RecordingEnv(TRexEnv):
        def _draw_episode_command(self) -> np.ndarray:
            seen["rng_state"] = self.np_random.bit_generator.state
            seen["push_starts"] = np.array(self._push_schedule_starts, copy=True)
            seen["qpos"] = np.array(self.data.qpos, copy=True)
            seen["mocap_pos"] = np.array(self.data.mocap_pos, copy=True)
            command: np.ndarray = marker.copy()
            return command

    env = RecordingEnv(**load_stage_config("trex", "recovery")["env_kwargs"])
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

    plain = TRexEnv(reset_noise_scale=0.0)
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


def test_sb3_refuses_command_mode_other_than_none_in_phase_c():
    with pytest.raises(ValueError, match="Phase D"):
        TRexEnv(command_mode="heading")
    with pytest.raises(ValueError, match="Phase D"):
        CompsognathusEnv(command_mode="heading_and_speed")
    with pytest.raises(ValueError, match=re.escape(f"command_mode='bogus' is not one of {COMMAND_MODES}")):
        TRexEnv(command_mode="bogus")


# ---------------------------------------------------------------------------
# Task fingerprint (amendments A1, A9)
# ---------------------------------------------------------------------------


def _fingerprint(**overrides):
    kwargs = dict(
        species="trex",
        stage=1,
        backend="stable-baselines3",
        env_kwargs=_TREX_ENV,
        plant_identity=_FAKE_PLANT,
        perturbation_manifest=None,
    )
    kwargs.update(overrides)
    return compute_task_fingerprint(**kwargs)


def test_committed_stages_carry_no_command_keys_and_no_command_section():
    """(i): under the committed TOMLs every stage's env section is command-free."""
    for species in SPECIES_NAMES:
        identity = current_plant_identity(species, verify_generated=False).to_dict()
        for stage_key, stage_config in load_all_stages(species).items():
            assert not COMMAND_ENV_KEYS_SET & set(stage_config["env_kwargs"]), (species, stage_key)
            payload = derive_stage_task_fingerprint(
                species=species,
                stage=stage_key,
                backend="stable-baselines3",
                env_kwargs=stage_config["env_kwargs"],
                plant_identity=identity,
            )
            assert not COMMAND_ENV_KEYS_SET & set(payload["env"]), (species, stage_key)
            assert "command" not in payload, (species, stage_key)


def test_command_kwargs_enter_the_task_fingerprint_and_the_payload_has_a_command_section():
    base = _fingerprint()
    assert not COMMAND_ENV_KEYS_SET & set(base["env"])
    assert "command" not in base
    # An explicit "none" (with inert ranges) is carved out too: same hash.
    explicit_none = _fingerprint(env_kwargs={**_TREX_ENV, "command_mode": "none", "command_yaw_rate_max": 0.0})
    assert explicit_none["task_sha256"] == base["task_sha256"]

    # (ii): a live mode keeps all six keys and moves the hash.  compute_task_
    # fingerprint is pure (no env is constructed), so the SB3 refusal does
    # not fire here.
    live = _fingerprint(env_kwargs={**_TREX_ENV, "command_mode": "heading", "command_yaw_rate_max": 0.5})
    assert COMMAND_ENV_KEYS_SET <= set(live["env"])
    assert live["env"]["command_mode"] == "heading"
    assert live["env"]["command_yaw_rate_max"] == 0.5
    assert live["env"]["command_speed_range"] == [0.0, 0.0]
    assert live["task_sha256"] != base["task_sha256"]
    assert "command" not in live

    # (iii): a command manifest adds the payload section and moves the hash.
    with_manifest = _fingerprint(command_manifest={"schema": "x"})
    assert with_manifest["command"] == {"schema": "x"}
    assert with_manifest["task_sha256"] != base["task_sha256"]
    assert with_manifest["env"] == base["env"]
    derived = derive_stage_task_fingerprint(
        species="trex",
        stage=1,
        backend="stable-baselines3",
        env_kwargs=_TREX_ENV,
        plant_identity=_FAKE_PLANT,
        command_manifest={"schema": "x"},
    )
    assert derived["command"] == {"schema": "x"}

    # (iv), SB3 half: no manifest while command_mode is "none".  The MJX
    # half lives in test_mjx_phase_c_interface.py.
    env = TRexEnv(reset_noise_scale=0.0)
    try:
        assert env.command_manifest() is None
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Recovery-calibration restamp (decision D-C2, amendment A13)
# ---------------------------------------------------------------------------

CALIBRATED_SPECIES = ("compsognathus", "compsognathus_robot")
CONFIGS_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _committed_profile(species: str) -> tuple[Path, str, dict[str, Any]]:
    path = CONFIGS_ROOT / species / "recovery_calibration.json"
    text = path.read_text(encoding="utf-8")
    return path, text, json.loads(text)


def _pre_restamp_copy(species: str, root: Path) -> tuple[Path, dict[str, Any]]:
    """The committed profile rolled back to what the tool found before the bump."""
    _, _, profile = _committed_profile(species)
    entry = profile["restamp_history"][-1]
    identity = dict(profile["plant_identity"])
    identity["policy_interface_revision"] = entry["previous_policy_interface_revision"]
    identity["policy_interface_sha256"] = entry["previous_policy_interface_sha256"]
    identity["observation_dim"] = identity["observation_dim"] - COMMAND_WIDTH
    profile["plant_identity"] = identity
    profile["task_sha256"] = entry["previous_task_sha256"]
    del profile["restamp_history"]
    path = root / species / "recovery_calibration.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, entry


@pytest.mark.parametrize("species", CALIBRATED_SPECIES)
def test_committed_recovery_calibrations_load_and_record_one_restamp(species):
    from environments.shared.recovery_calibration import load_recovery_calibration
    from environments.shared.scripts.restamp_recovery_calibration import RESTAMP_REASON

    calibration = load_recovery_calibration(species)
    _, _, profile = _committed_profile(species)
    assert calibration.profile["species"] == species
    assert calibration.profile["restamp_history"] == profile["restamp_history"]
    identity = current_plant_identity(species, verify_generated=False)
    assert profile["plant_identity"] == identity.to_dict()
    history = profile["restamp_history"]
    assert len(history) == 1
    entry = history[0]
    assert entry["previous_policy_interface_revision"] == identity.policy_interface_revision - 1
    assert entry["previous_policy_interface_sha256"] != identity.policy_interface_sha256
    assert entry["previous_task_sha256"] != profile["task_sha256"]
    assert entry["reason"] == RESTAMP_REASON
    assert re.fullmatch(r"[0-9a-f]{40}", entry["restamped_at_commit"])


def test_restamp_recovery_calibration_is_idempotent_and_refuses_a_physics_change(tmp_path):
    from environments.shared.scripts import restamp_recovery_calibration as tool

    root = tmp_path / "configs"
    # Roll both committed profiles back to their pre-bump identity: the tool
    # must reproduce the committed bytes (the recorded commit is passed
    # explicitly so the replay does not depend on the checkout's HEAD).
    for species in CALIBRATED_SPECIES:
        path, entry = _pre_restamp_copy(species, root)
        assert tool.restamp_recovery_calibration(species, path=path, commit=entry["restamped_at_commit"]) is True
        assert path.read_text(encoding="utf-8") == _committed_profile(species)[1], species

    # Second run through the CLI: a byte no-op, no history append.
    before = {species: (root / species / "recovery_calibration.json").read_bytes() for species in CALIBRATED_SPECIES}
    assert tool.main(["--configs-root", str(root)]) == 0
    for species in CALIBRATED_SPECIES:
        after = (root / species / "recovery_calibration.json").read_bytes()
        assert after == before[species], species
        assert len(json.loads(after)["restamp_history"]) == 1

    # A physics change needs a real recalibration, not a restamp.
    path = root / "compsognathus" / "recovery_calibration.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["plant_identity"]["physics_sha256"] = "sha256:" + "0" * 64
    profile["task_sha256"] = "sha256:" + "1" * 64
    foreign = json.dumps(profile, indent=2, ensure_ascii=False) + "\n"
    path.write_text(foreign, encoding="utf-8")
    with pytest.raises(tool.RestampError, match="physics change"):
        tool.restamp_recovery_calibration("compsognathus", path=path)
    assert tool.main(["--species", "compsognathus", "--configs-root", str(root)]) == 1
    assert path.read_text(encoding="utf-8") == foreign  # never writes on a refusal

    # A changed task (recovery_env_kwargs) is refused the same way.
    shutil.copy(CONFIGS_ROOT / "compsognathus" / "recovery_calibration.json", path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["recovery_env_kwargs"]["perturbation_interval"] = 99.0
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(tool.RestampError, match="recovery_env_kwargs differ"):
        tool.restamp_recovery_calibration("compsognathus", path=path)

    # An unknown species is a refusal (exit 1), not a traceback.
    with pytest.raises(tool.RestampError, match="Unknown species"):
        tool.restamp_recovery_calibration("stegosaurus", path=path)
    assert tool.main(["--species", "stegosaurus", "--configs-root", str(root)]) == 1


def test_restamp_recovery_calibration_records_the_given_reason(tmp_path):
    """A later interface-only revision must name its own reason; the Phase C text is only the default."""
    from environments.shared.scripts import restamp_recovery_calibration as tool

    root = tmp_path / "configs"
    path, entry = _pre_restamp_copy("compsognathus", root)
    assert tool.main(["--species", "compsognathus", "--configs-root", str(root), "--reason", "r99 bump"]) == 0
    history = json.loads(path.read_text(encoding="utf-8"))["restamp_history"]
    assert [item["reason"] for item in history] == ["r99 bump"]
    assert history[0]["previous_task_sha256"] == entry["previous_task_sha256"]

    path, _ = _pre_restamp_copy("compsognathus_robot", root)
    with pytest.raises(tool.RestampError, match="non-empty"):
        tool.restamp_recovery_calibration("compsognathus_robot", path=path, reason="  ")


# ---------------------------------------------------------------------------
# mjlab registration (amendment A5)
# ---------------------------------------------------------------------------


def test_velociraptor_mjlab_obs_dim_matches_the_plant_identity():
    import environments.velociraptor.mjlab_config  # noqa: F401  (registers the species)
    from environments.shared.mjlab_env import get_species_mjlab

    config = get_species_mjlab("velociraptor")
    identity = current_plant_identity("velociraptor", verify_generated=False)
    assert config.obs_dim == identity.observation_dim == 70
    assert config.action_dim == identity.action_dim == 22
