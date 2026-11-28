"""The plant contract on the heading-free schemas (plant_versions note 18, D-D31; every model since the target-free
revision).

The T-Rex, the anatomical Compsognathus, the velociraptor and the Allosaurus
carry the biped schema ``bipedal-heading-free/v1`` (before the target-free
revision ``bipedal-target-heading-free/v1``, which also carried the target
direction and distance); the robot joined them in that revision, and the two
quadrupeds took ``quadrupedal-heading-free/v1``, the same segments on the
torso.  The
policy-interface payload is pinned here in four ways: the observation segments
name their frames, sources and thresholds and describe the feet per foot (the
velociraptor's per site since its policy interface r14, plant_versions note 22;
the robot's as floor-contact force since its r4), so their widths sum to the
observation width (checked, with the command last); the frame helpers
``_get_obs`` calls are hashed beside it
(``interface_implementations.heading_free_observation``), so editing one moves
the digest; the schemas are SB3-only, and a jax-mjx declaration is refused; and
the synthetic probe's non-unit quaternion is normalized, so the probe still
exercises a real tilt.  The observation-ablation script's T-Rex slices are
checked against the same segments.  A world-frame or target schema, a plant
with a mocap body, and floor-contact foot channels that are not disjoint groups
of the animal's geoms, are refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.allosaurus_fragilis.anatomical.env import AllosaurusFragilisAnatomicalEnv
from environments.brachiosaurus_altithorax.anatomical.env import BrachiosaurusAltithoraxAnatomicalEnv
from environments.compsognathus_longipes.anatomical.env import CompsognathusLongipesAnatomicalEnv
from environments.compsognathus_longipes.robot.env import CompsognathusLongipesRobotEnv
from environments.dibothrosuchus_elaphros.anatomical.env import DibothrosuchusElaphrosAnatomicalEnv
from environments.shared import heading_observation
from environments.shared.command_frame import COMMAND_SEGMENT_NAME
from environments.shared.heading_observation import (
    BIPEDAL_HEADING_FREE_SCHEMA,
    HEADING_FREE_FUNCTIONS,
    HEADING_FREE_ROOT_LABELS,
    QUADRUPEDAL_HEADING_FREE_SCHEMA,
)
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import (
    POLICY_INTERFACE_SCHEMA,
    PlantContractError,
    load_plant_versions,
    policy_layer,
)
from environments.shared.plant_contract.digests import _semantic_digest
from environments.shared.scripts import observation_ablation_report
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
ALLOSAURUS_FRAGILIS = ModelId("allosaurus_fragilis", "anatomical")
BRACHIOSAURUS_ALTITHORAX = ModelId("brachiosaurus_altithorax", "anatomical")
DIBOTHROSUCHUS_ELAPHROS = ModelId("dibothrosuchus_elaphros", "anatomical")
QUADRUPEDS = {BRACHIOSAURUS_ALTITHORAX, DIBOTHROSUCHUS_ELAPHROS}

#: Every model whose feet read touch sensors (the robot's read floor contact: its own test below).
HEADING_FREE = (
    pytest.param(TYRANNOSAURUS_REX, TyrannosaurusRexAnatomicalEnv, id="tyrannosaurus_rex-anatomical"),
    pytest.param(COMPSOGNATHUS_LONGIPES, CompsognathusLongipesAnatomicalEnv, id="compsognathus_longipes-anatomical"),
    pytest.param(
        VELOCIRAPTOR_MONGOLIENSIS, VelociraptorMongoliensisAnatomicalEnv, id="velociraptor_mongoliensis-anatomical"
    ),
    pytest.param(ALLOSAURUS_FRAGILIS, AllosaurusFragilisAnatomicalEnv, id="allosaurus_fragilis-anatomical"),
    pytest.param(
        BRACHIOSAURUS_ALTITHORAX, BrachiosaurusAltithoraxAnatomicalEnv, id="brachiosaurus_altithorax-anatomical"
    ),
    pytest.param(DIBOTHROSUCHUS_ELAPHROS, DibothrosuchusElaphrosAnatomicalEnv, id="dibothrosuchus_elaphros-anatomical"),
)
#: Each model's foot-contact channels' touch-sensor groups, right foot first (the quadrupeds' front right, front
#: left, rear right, rear left): the T-Rex sums its plantar pad and three digits per foot, the brachiosaurus its
#: pad and metapodial per leg; the velociraptor observes each of its six sensors as its own channel.
FOOT_GROUPS = {
    TYRANNOSAURUS_REX: [
        ["r_foot_touch", "r_toe_d2_touch", "r_toe_d3_touch", "r_toe_d4_touch"],
        ["l_foot_touch", "l_toe_d2_touch", "l_toe_d3_touch", "l_toe_d4_touch"],
    ],
    COMPSOGNATHUS_LONGIPES: [["r_foot_touch"], ["l_foot_touch"]],
    # One channel per load-bearing site since policy interface r14 (plant_versions note 22): the MTP pad (the foot
    # site's own sensor), digit III and digit IV, right foot first (r11-r13 summed d3, metatarsus and d4 per foot).
    VELOCIRAPTOR_MONGOLIENSIS: [
        ["r_foot_touch"],
        ["r_toe_d3_touch"],
        ["r_toe_d4_touch"],
        ["l_foot_touch"],
        ["l_toe_d3_touch"],
        ["l_toe_d4_touch"],
    ],
    ALLOSAURUS_FRAGILIS: [["r_foot_touch"], ["l_foot_touch"]],
    BRACHIOSAURUS_ALTITHORAX: [
        ["fr_foot_touch", "fr_meta_touch"],
        ["fl_foot_touch", "fl_meta_touch"],
        ["rr_foot_touch", "rr_meta_touch"],
        ["rl_foot_touch", "rl_meta_touch"],
    ],
    DIBOTHROSUCHUS_ELAPHROS: [["fr_foot_touch"], ["fl_foot_touch"], ["rr_foot_touch"], ["rl_foot_touch"]],
}


def _schema(model_id: ModelId) -> str:
    return QUADRUPEDAL_HEADING_FREE_SCHEMA if model_id in QUADRUPEDS else BIPEDAL_HEADING_FREE_SCHEMA


def _payload(model_id: ModelId, env, **kwargs):
    return policy_layer._policy_interface_payload(env.model, env, load_plant_versions()[1][model_id], **kwargs)


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
    model_id, env_class = request.param
    env = env_class(reset_noise_scale=0.0)
    yield model_id, env
    env.close()


def test_heading_free_payload_records_segments_frames_and_helper(animal):
    model_id, env = animal
    schema = _schema(model_id)
    root = HEADING_FREE_ROOT_LABELS[schema]
    assert load_plant_versions()[1][model_id].observation_schema == schema
    payload = _payload(model_id, env)
    assert payload["observation_schema"] == schema
    segments = {segment["name"]: segment for segment in payload["observation_segments"]}
    assert list(segments) == [
        "joint_position",
        "joint_velocity",
        f"{root}_projected_gravity",
        f"{root}_angular_velocity",
        f"{root}_heading_linear_velocity",
        f"{root}_linear_acceleration",
        "foot_contact",
        COMMAND_SEGMENT_NAME,
    ]
    assert segments[f"{root}_projected_gravity"] == {
        "name": f"{root}_projected_gravity",
        "width": 3,
        "frame": root,
        "source": "qpos[3:7]",
        "convention": "unit-down",
        "normalization_floor": "1e-12",
    }
    assert segments[f"{root}_heading_linear_velocity"] == {
        "name": f"{root}_heading_linear_velocity",
        "width": 3,
        "components": ["forward", "left", "up"],
        "frame": f"gravity-aligned-{root}-heading",
        "source": "qvel[0:3]",
        "fallback": f"horizontal-{root}-y-rotated-clockwise-90",
        "fallback_threshold": "1e-08",
    }
    assert segments[f"{root}_angular_velocity"]["frame"] == "gyro-site"
    assert segments[f"{root}_linear_acceleration"]["frame"] == "accelerometer-site"
    groups = FOOT_GROUPS[model_id]
    assert segments["foot_contact"] == {"name": "foot_contact", "width": len(groups), "groups": groups}
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
    assert ("action_low_pass_filter" in implementations) is (model_id == TYRANNOSAURUS_REX)
    # Only an embodiment whose feet read floor contact hashes the floor-contact reading.
    assert "foot_floor_contact_observation" not in implementations


def test_the_robot_reads_the_biped_schema_with_floor_contact_feet():
    """The robot left its world-frame layout in the target-free revision (policy interface r4): the biped schema's
    segments, each foot the normal force of its sole against the floor alone (a sole resting on the other foot reads
    0 N), with that reading hashed beside the heading helpers."""
    with CompsognathusLongipesRobotEnv(reset_noise_scale=0.0) as env:
        payload = _payload(COMPSOGNATHUS_LONGIPES_ROBOT, env)
    assert payload["observation_schema"] == BIPEDAL_HEADING_FREE_SCHEMA
    segments = {segment["name"]: segment for segment in payload["observation_segments"]}
    assert list(segments)[2:] == [
        "pelvis_projected_gravity",
        "pelvis_angular_velocity",
        "pelvis_heading_linear_velocity",
        "pelvis_linear_acceleration",
        "foot_contact",
        COMMAND_SEGMENT_NAME,
    ]
    assert segments["foot_contact"] == {
        "name": "foot_contact",
        "width": 2,
        "source": "floor-contact-normal-force",
        "floor_geom": "floor",
        "geom_groups": [["right_sole"], ["left_sole"]],
    }
    implementations = payload["interface_implementations"]
    assert set(implementations["heading_free_observation"]) == set(HEADING_FREE_FUNCTIONS)
    assert implementations["foot_floor_contact_observation"]["qualname"] == "BaseDinoEnv._foot_floor_contact_forces"
    assert payload["observation"]["shape"] == [41]


def _no_floor_channel(model, groups):
    return ()


def _an_empty_floor_channel(model, groups):
    return (*groups, ())


def _a_geom_off_the_model(model, groups):
    return (groups[0], (model.ngeom,))


def _the_floor_itself(model, groups):
    return (groups[0], (model.geom("floor").id,))


def _a_sole_in_both_channels(model, groups):
    return (groups[0], (*groups[1], *groups[0]))


@pytest.mark.parametrize(
    "edit",
    [_no_floor_channel, _an_empty_floor_channel, _a_geom_off_the_model, _the_floor_itself, _a_sole_in_both_channels],
    ids=["none", "empty", "off-model", "floor", "shared"],
)
def test_the_floor_contact_channels_must_be_disjoint_groups_of_the_animals_geoms(monkeypatch, edit):
    """``_foot_floor_contact_forces`` credits a geom named in two channels to the last of them only, and a channel
    naming the floor geom reads nothing for it, so the contract refuses those groups, as it refuses no channel, an
    empty channel or a geom the model does not have."""
    with CompsognathusLongipesRobotEnv(reset_noise_scale=0.0) as env:
        monkeypatch.setattr(env, "_foot_observation_floor_geoms", edit(env.model, env._foot_observation_floor_geoms))
        with pytest.raises(PlantContractError, match="needs disjoint, non-empty groups of the model's geoms"):
            _payload(COMPSOGNATHUS_LONGIPES_ROBOT, env)


@pytest.mark.parametrize("schema", ["bipedal-target/v1", "bipedal-target-heading-free/v1", "quadrupedal-target/v1"])
def test_a_world_frame_or_target_schema_is_refused(schema):
    """The world-frame and target schemas went with every scene's target: a plant declaring one is refused."""
    import dataclasses

    with TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0) as env:
        version = dataclasses.replace(load_plant_versions()[1][TYRANNOSAURUS_REX], observation_schema=schema)
        with pytest.raises(PlantContractError, match=f"declares observation schema '{schema}'"):
            policy_layer._policy_interface_payload(env.model, env, version)


def test_a_plant_with_a_mocap_body_is_refused():
    """No scene carries a mocap body: one added back (a target marker, say) is refused by name."""
    import mujoco

    with TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0) as env:
        spec = mujoco.MjSpec.from_file(str(env.MODEL_PATH))
        marker = spec.worldbody.add_body(name="marker", mocap=True, pos=[8.0, 0.0, 0.5])
        marker.add_site(name="marker_site")
        model = spec.compile()
        assert model.nmocap == 1
        with pytest.raises(PlantContractError, match=r"defines 1 mocap bodies \['marker'\]"):
            policy_layer._policy_interface_payload(model, env, load_plant_versions()[1][TYRANNOSAURUS_REX])


def test_editing_a_heading_helper_moves_the_policy_digest(animal, monkeypatch):
    model_id, env = animal
    before = _payload(model_id, env)

    def projected_gravity(rotation: np.ndarray) -> np.ndarray:
        down = np.array([0.0, 0.0, -1.0])
        return np.asarray(rotation.T @ down, dtype=np.float64)

    monkeypatch.setattr(heading_observation, "projected_gravity", projected_gravity)
    after = _payload(model_id, env)
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
    model_id, env = animal
    monkeypatch.setattr(env, "supported_training_backends", ("stable-baselines3", "jax-mjx"), raising=False)
    # Refused on every schema and mapping since the backend was removed (plant_versions note 23).
    monkeypatch.setattr(env, "action_mapping", "home-keyframe-residual/v1", raising=False)
    with pytest.raises(PlantContractError, match="declares the jax-mjx training backend, which was removed"):
        _payload(model_id, env)


def test_the_segments_must_sum_to_the_observation_width(animal, monkeypatch):
    import gymnasium as gym

    model_id, env = animal
    width = env.observation_space.shape[0]
    monkeypatch.setattr(env, "observation_space", gym.spaces.Box(-np.inf, np.inf, shape=(width + 1,)))
    with pytest.raises(PlantContractError, match=f"must sum to the observation width {width + 1}"):
        _payload(model_id, env)


def test_foot_groups_must_name_touch_sensors(animal, monkeypatch):
    model_id, env = animal
    monkeypatch.setattr(env, "_foot_sensor_groups", ((env._sensor_gyro_start,), (env._sensor_gyro_start + 1,)))
    with pytest.raises(PlantContractError, match="start no touch sensor"):
        _payload(model_id, env)


@pytest.fixture
def velociraptor_mongoliensis_env():
    env = VelociraptorMongoliensisAnatomicalEnv(reset_noise_scale=0.0)
    yield env
    env.close()


def _drop_the_last_site(env):
    return {"_foot_observation_groups": env._foot_observation_groups[:-1]}


def _read_a_site_twice(env):
    return {"_foot_observation_groups": (*env._foot_observation_groups, env._foot_observation_groups[0])}


def _read_a_site_no_foot_reads(env):
    # The per-foot groups lose the left digit IV; the per-site channels still read it.
    right, left = env._foot_sensor_groups
    return {"_foot_sensor_groups": (right, left[:-1])}


def _read_a_gyro_channel(env):
    return {"_foot_observation_groups": (*env._foot_observation_groups[:-1], (env._sensor_gyro_start,))}


def _add_an_empty_channel(env):
    return {"_foot_observation_groups": (*env._foot_observation_groups, ())}


def _declare_no_channel(env):
    return {"_foot_observation_groups": ()}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(_drop_the_last_site, "must read exactly the touch sensors of _foot_sensor_groups", id="drop"),
        pytest.param(_read_a_site_twice, "must read exactly the touch sensors of _foot_sensor_groups", id="twice"),
        pytest.param(
            _read_a_site_no_foot_reads, "must read exactly the touch sensors of _foot_sensor_groups", id="add"
        ),
        pytest.param(
            _read_a_gyro_channel, r"_foot_observation_groups=.*addresses that start no touch sensor: \[\d+\]", id="gyro"
        ),
        pytest.param(_add_an_empty_channel, r"none empty; _foot_observation_groups=.*\[\]\], addresses", id="empty"),
        pytest.param(_declare_no_channel, r"none empty; _foot_observation_groups=\[\], addresses", id="none"),
    ],
)
def test_the_per_site_channels_must_read_exactly_the_feets_touch_sensors(
    velociraptor_mongoliensis_env, monkeypatch, edit, message
):
    """The velociraptor's ``_foot_observation_groups`` (policy interface r14): each channel a non-empty group of
    touch sensors, together reading every sensor of the per-foot ``_foot_sensor_groups`` once and nothing else.
    The per-foot check alone (test_foot_groups_must_name_touch_sensors) cannot see a per-site channel dropped,
    doubled or added."""
    env = velociraptor_mongoliensis_env
    for name, value in edit(env).items():
        monkeypatch.setattr(env, name, value)
    with pytest.raises(PlantContractError, match=message):
        _payload(VELOCIRAPTOR_MONGOLIENSIS, env)


def test_heading_free_probe_normalizes_the_synthetic_quaternion(animal, monkeypatch):
    """The probe's linspace qpos gives a non-unit quaternion; the helper normalizes it, so scaling changes nothing
    and the probe still reads a real tilt rather than the upright (0, 0, -1)."""
    model_id, env = animal
    before = _payload(model_id, env)
    probe_data = policy_layer._deterministic_probe_data

    def scaled(model):
        data = probe_data(model)
        data.qpos[3:7] *= -7.5
        return data

    monkeypatch.setattr(policy_layer, "_deterministic_probe_data", scaled)
    after = _payload(model_id, env)
    np.testing.assert_array_equal(after["observation_probe"]["values"], before["observation_probe"]["values"])
    root = HEADING_FREE_ROOT_LABELS[_schema(model_id)]
    gravity = np.asarray(before["observation_probe"]["values"])[_offsets(before)[f"{root}_projected_gravity"]]
    assert np.linalg.norm(gravity) == pytest.approx(1.0, abs=1e-5)
    assert np.max(np.abs(gravity - [0.0, 0.0, -1.0])) > 0.5


def test_observation_ablation_slices_match_segments():
    with TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0) as env:
        layout = _offsets(_payload(TYRANNOSAURUS_REX, env))
    table = observation_ablation_report.SLICES[TYRANNOSAURUS_REX]
    expected = {
        "root_linvel": "pelvis_heading_linear_velocity",
        "command": COMMAND_SEGMENT_NAME,
    }
    assert set(table) == set(expected)
    for key, segment in expected.items():
        start, stop, quaternion_index = table[key]
        assert slice(start, stop) == layout[segment], key
        # No world-frame slice and no quaternion on this layout: the "body" intervention has nothing to rotate.
        assert quaternion_index is None, key
