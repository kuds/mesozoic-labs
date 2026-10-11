"""The plant contract on the ``bipedal-target-heading-free/v1`` schema (plant_versions note 18, D-D31).

The T-Rex (policy interface r14) and the anatomical Compsognathus (r4) are the
schema's two species.  Its policy-interface payload differs from the legacy
schemas' in four ways, each pinned here: the observation segments name their
frames, sources and thresholds and describe the feet per foot, so their widths
sum to the observation width (checked, with the command last); the frame
helpers ``_get_obs`` calls are hashed beside it
(``interface_implementations.heading_free_observation``), so editing one moves
the digest; the schema is SB3-only, and a jax-mjx declaration is refused; and
the synthetic probe's non-unit quaternion is normalized, so the probe still
exercises a real tilt.  The observation-ablation script's T-Rex slices are
checked against the same segments.  The legacy schemas keep their payloads
(the digest snapshot pins every other species' lines).
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.compsognathus.envs.compsognathus_env import CompsognathusBiologicalEnv, CompsognathusRobotEnv
from environments.shared import heading_observation
from environments.shared.command_frame import COMMAND_SEGMENT_NAME
from environments.shared.heading_observation import HEADING_FREE_FUNCTIONS, HEADING_FREE_OBSERVATION_SCHEMA
from environments.shared.plant_contract import (
    POLICY_INTERFACE_SCHEMA,
    PlantContractError,
    load_plant_versions,
    policy_layer,
)
from environments.shared.plant_contract.digests import _semantic_digest
from environments.shared.scripts import observation_ablation_report
from environments.trex.envs.trex_env import TRexEnv

HEADING_FREE = (
    pytest.param("trex", TRexEnv, id="trex"),
    pytest.param("compsognathus", CompsognathusBiologicalEnv, id="compsognathus"),
)
#: Each species' per-foot touch-sensor groups, right foot first: the T-Rex sums its plantar pad and three digits.
FOOT_GROUPS = {
    "trex": [
        ["r_foot_touch", "r_toe_d2_touch", "r_toe_d3_touch", "r_toe_d4_touch"],
        ["l_foot_touch", "l_toe_d2_touch", "l_toe_d3_touch", "l_toe_d4_touch"],
    ],
    "compsognathus": [["r_foot_touch"], ["l_foot_touch"]],
}


def _payload(species: str, env, **kwargs):
    return policy_layer._policy_interface_payload(env.model, env, load_plant_versions()[1][species], **kwargs)


def _offsets(payload) -> dict[str, slice]:
    layout: dict[str, slice] = {}
    start = 0
    for segment in payload["observation_segments"]:
        width = len(segment["components"]) if "components" in segment else int(segment["width"])
        layout[segment["name"]] = slice(start, start + width)
        start += width
    return layout


@pytest.fixture(params=[pytest.param(tuple(param.values), id=param.id) for param in HEADING_FREE])
def animal(request):
    species, env_class = request.param
    env = env_class(reset_noise_scale=0.0)
    yield species, env
    env.close()


def test_heading_free_payload_records_segments_frames_and_helper(animal):
    species, env = animal
    assert load_plant_versions()[1][species].observation_schema == HEADING_FREE_OBSERVATION_SCHEMA
    payload = _payload(species, env, require_backend_parity=True)
    assert payload["observation_schema"] == HEADING_FREE_OBSERVATION_SCHEMA
    segments = {segment["name"]: segment for segment in payload["observation_segments"]}
    assert list(segments)[-1] == COMMAND_SEGMENT_NAME
    assert "pelvis_orientation_quaternion" not in segments and "prey_direction" not in segments
    assert segments["pelvis_projected_gravity"] == {
        "name": "pelvis_projected_gravity",
        "width": 3,
        "frame": "pelvis",
        "source": "qpos[3:7]",
        "convention": "unit-down",
        "normalization_floor": "1e-12",
    }
    assert segments["pelvis_heading_linear_velocity"] == {
        "name": "pelvis_heading_linear_velocity",
        "width": 3,
        "components": ["forward", "left", "up"],
        "frame": "gravity-aligned-pelvis-heading",
        "source": "qvel[0:3]",
        "fallback": "horizontal-pelvis-y-rotated-clockwise-90",
        "fallback_threshold": "1e-08",
    }
    assert segments["pelvis_angular_velocity"]["frame"] == "gyro-site"
    assert segments["pelvis_linear_acceleration"]["frame"] == "accelerometer-site"
    assert segments["prey_direction_body"]["frame"] == "pelvis"
    assert segments["foot_contact"] == {"name": "foot_contact", "width": 2, "groups": FOOT_GROUPS[species]}
    widths = [stop.stop - stop.start for stop in _offsets(payload).values()]
    assert sum(widths) == env.observation_space.shape[0] == payload["observation"]["shape"][0]

    implementations = payload["interface_implementations"]
    assert set(implementations["heading_free_observation"]) == set(HEADING_FREE_FUNCTIONS)
    assert implementations["supported_training_backends"] == ["stable-baselines3"]
    assert "sb3_sensor_cache" in implementations
    assert set(implementations["home_reset"]) == {"sb3"}
    assert not {"backend_neutral_observation", "jax_action_mapping", "jax_observation_callers"} & set(implementations)
    assert payload["jax_interface"] == {"supported": False, "backend": "jax-mjx"}
    assert payload["backend_observation_equal"] is None
    # The T-Rex's low-pass filter pins stay in its interface after it left the frozen MJX core.
    assert ("action_low_pass_filter" in implementations) is (species == "trex")


def test_the_legacy_schemas_keep_their_layout():
    """The robot (and every other species) stays on its world-frame layout and hashes no heading helper."""
    with CompsognathusRobotEnv(reset_noise_scale=0.0) as env:
        payload = _payload("compsognathus_robot", env)
    names = [segment["name"] for segment in payload["observation_segments"]]
    assert names[2] == "pelvis_orientation_quaternion" and "prey_direction" in names
    assert "heading_free_observation" not in payload["interface_implementations"]
    assert payload["observation"]["shape"] == [46]


def test_editing_a_heading_helper_moves_the_policy_digest(animal, monkeypatch):
    species, env = animal
    before = _payload(species, env)

    def projected_gravity(rotation: np.ndarray) -> np.ndarray:
        down = np.array([0.0, 0.0, -1.0])
        return np.asarray(rotation.T @ down, dtype=np.float64)

    monkeypatch.setattr(heading_observation, "projected_gravity", projected_gravity)
    after = _payload(species, env)
    # The same values (R^T (0, 0, -1) is minus R's third row), so only the hashed tokens can tell them apart.
    np.testing.assert_array_equal(after["observation_probe"]["values"], before["observation_probe"]["values"])
    moved = {
        name
        for name in HEADING_FREE_FUNCTIONS
        if after["interface_implementations"]["heading_free_observation"][name]
        != before["interface_implementations"]["heading_free_observation"][name]
    }
    assert moved == {"projected_gravity"}
    assert _semantic_digest(POLICY_INTERFACE_SCHEMA, after) != _semantic_digest(POLICY_INTERFACE_SCHEMA, before)


def test_heading_free_schema_refuses_the_jax_backend(animal, monkeypatch):
    species, env = animal
    monkeypatch.setattr(env, "supported_training_backends", ("stable-baselines3", "jax-mjx"), raising=False)
    with pytest.raises(PlantContractError, match="has no MJX builder"):
        _payload(species, env)


def test_the_segments_must_sum_to_the_observation_width(animal, monkeypatch):
    import gymnasium as gym

    species, env = animal
    width = env.observation_space.shape[0]
    monkeypatch.setattr(env, "observation_space", gym.spaces.Box(-np.inf, np.inf, shape=(width + 1,)))
    with pytest.raises(PlantContractError, match=f"must sum to the observation width {width + 1}"):
        _payload(species, env)


def test_foot_groups_must_name_touch_sensors(animal, monkeypatch):
    species, env = animal
    monkeypatch.setattr(env, "_foot_sensor_groups", ((env._sensor_gyro_start,), (env._sensor_gyro_start + 1,)))
    with pytest.raises(PlantContractError, match="start no touch sensor"):
        _payload(species, env)


def test_heading_free_probe_normalizes_the_synthetic_quaternion(animal, monkeypatch):
    """The probe's linspace qpos gives a non-unit quaternion; the helper normalizes it, so scaling changes nothing
    and the probe still reads a real tilt rather than the upright (0, 0, -1)."""
    species, env = animal
    before = _payload(species, env)
    probe_data = policy_layer._deterministic_probe_data

    def scaled(model):
        data = probe_data(model)
        data.qpos[3:7] *= -7.5
        return data

    monkeypatch.setattr(policy_layer, "_deterministic_probe_data", scaled)
    after = _payload(species, env)
    np.testing.assert_array_equal(after["observation_probe"]["values"], before["observation_probe"]["values"])
    gravity = np.asarray(before["observation_probe"]["values"])[_offsets(before)["pelvis_projected_gravity"]]
    assert np.linalg.norm(gravity) == pytest.approx(1.0, abs=1e-5)
    assert np.max(np.abs(gravity - [0.0, 0.0, -1.0])) > 0.5


def test_observation_ablation_slices_match_segments():
    with TRexEnv(reset_noise_scale=0.0) as env:
        layout = _offsets(_payload("trex", env))
    table = observation_ablation_report.SLICES["trex"]
    expected = {
        "root_linvel": "pelvis_heading_linear_velocity",
        "target_dir": "prey_direction_body",
        "target_dist": "prey_distance",
        "command": COMMAND_SEGMENT_NAME,
    }
    assert set(table) == set(expected)
    for key, segment in expected.items():
        start, stop, quaternion_index = table[key]
        assert slice(start, stop) == layout[segment], key
        # No world-frame slice and no quaternion on this layout: the "body" intervention has nothing to rotate.
        assert quaternion_index is None, key
