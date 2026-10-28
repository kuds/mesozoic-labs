"""M1 model, sensing, Gymnasium behavior and revision provenance regressions."""

import csv
import hashlib
import json
import tomllib

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.tyrannosaurus_rex.robot.env import TyrannosaurusRexRobotEnv
from examples.mini_tyrannosaurus_rex.skeleton.build import OUTPUT, ROOT, SOURCE, model_text

MODEL_ID = ModelId("tyrannosaurus_rex", "robot")
HARDWARE = ROOT / "hardware/mini_tyrannosaurus_rex/m01_skeleton"


def test_generated_asset_is_standalone_and_reproducible():
    assert OUTPUT.read_text() == model_text()
    model = mujoco.MjModel.from_xml_string(OUTPUT.read_text())
    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    assert (model.nq, model.nv, model.nu) == (17, 16, 10)
    np.testing.assert_array_equal(model.jnt_range, source.jnt_range)
    np.testing.assert_array_equal(model.actuator_forcerange, source.actuator_forcerange)
    np.testing.assert_array_equal(model.body_mass[: source.nbody], source.body_mass)
    np.testing.assert_allclose(model.actuator_forcerange, np.tile([-0.1, 0.1], (10, 1)))
    assert model.body_subtreemass[model.body("pelvis").id] == pytest.approx(0.7541)
    cases = [g for g in range(model.ngeom) if model.geom(g).name.endswith("_servo_case")]
    assert len(cases) == 10
    for g in cases:
        np.testing.assert_allclose(sorted(model.geom_size[g] * 2000), [20, 26, 34])


def test_future_loads_are_fixed_and_target_cannot_support_robot():
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    assert model.neq == 0 and not np.any(model.body_gravcomp)
    for name in ("future_head_and_neck", "future_fixed_tail", "future_fixed_arms"):
        assert model.body_jntnum[model.body(name).id] == 0
    mocap = np.flatnonzero(model.body_mocapid >= 0)
    assert [model.body(int(i)).name for i in mocap] == ["prey"]
    assert model.geom_contype[model.geom("prey_geom").id] == 0
    assert model.geom_conaffinity[model.geom("prey_geom").id] == 0


def test_gym_api_and_same_seed_replay():
    env = TyrannosaurusRexRobotEnv()
    try:
        check_env(env, skip_render_check=True)
        assert env.observation_space.shape == (41,) and env.action_space.shape == (10,)
        assert env.dt == pytest.approx(0.02)
        samples = []
        for _ in range(2):
            observation, _ = env.reset(seed=62)
            states = [observation]
            for _ in range(40):
                observation, reward, _, _, _ = env.step(np.full(10, 0.01))
                assert np.isfinite(reward) and np.all(np.isfinite(observation))
                states.append(observation)
            samples.append(np.asarray(states))
        np.testing.assert_array_equal(*samples)
    finally:
        env.close()


def test_touch_sensors_capture_entire_weight_and_distinguish_feet():
    env = TyrannosaurusRexRobotEnv(reset_noise_scale=0)
    try:
        env.reset(seed=0)
        for _ in range(250):
            _, _, terminated, _, _ = env.step(np.zeros(10))
            assert not terminated
        right, left = env._foot_contact_forces()
        assert right > 0 and left > 0
        assert right + left == pytest.approx(env.body_mass * 9.81, rel=0.001)
        # Loading one sensor must never appear in the other observation channel.
        env.data.sensordata[10:12] = [1.2, 3.4]
        np.testing.assert_allclose(env._get_obs()[32:34], [1.2, 3.4])
    finally:
        env.close()


def test_home_action_and_asymmetric_knee_endpoints():
    env = TyrannosaurusRexRobotEnv()
    try:
        np.testing.assert_array_equal(env._scale_action(np.zeros(10)), env._home_ctrl)
        np.testing.assert_array_equal(env._scale_action(np.full(10, -1)), env.model.actuator_ctrlrange[:, 0])
        np.testing.assert_array_equal(env._scale_action(np.ones(10)), env.model.actuator_ctrlrange[:, 1])
    finally:
        env.close()


@pytest.mark.parametrize("stage", ["stance", "recovery", "locomotion", "behavior"])
def test_every_stage_constructs_and_steps(stage):
    config = load_stage_config(MODEL_ID, stage)
    env = TyrannosaurusRexRobotEnv(**config["env_kwargs"])
    try:
        env.reset(seed=0)
        observation, reward, terminated, truncated, _ = env.step(np.zeros(10))
        assert not terminated and not truncated
        assert np.isfinite(reward) and observation.shape == (41,)
    finally:
        env.close()


def test_target_reach_requires_upright_slow_arrival():
    env = TyrannosaurusRexRobotEnv(reset_noise_scale=0, target_reach_bonus=25)
    try:
        env.reset(seed=0)
        env.data.mocap_pos[0, :2] = env.data.qpos[:2]
        env.data.qvel[:] = 0
        assert env._is_terminated()[1]["success"]
        env.data.qvel[0] = 0.2
        assert not env._is_terminated()[0]
        env.data.qvel[:] = 0
        env.data.qpos[2] = 0.04
        mujoco.mj_forward(env.model, env.data)
        terminated, info = env._is_terminated()
        assert terminated and not info.get("success", False)
    finally:
        env.close()


def test_registered_model_and_hardware_record_share_identity():
    with gym.make(MODEL_ID.gym_id) as env:
        assert isinstance(env.unwrapped, TyrannosaurusRexRobotEnv)
    revision = tomllib.loads((HARDWARE / "revision.toml").read_text())
    identity = current_plant_identity(MODEL_ID)
    for name in ("physics_revision", "policy_interface_revision", "visual_revision"):
        assert revision["model"][name] == getattr(identity, name)
    assert identity.observation_dim == 41 and identity.action_dim == 10
    with (HARDWARE / "bom.csv").open() as handle:
        mass = sum(int(row["quantity"]) * float(row["unit_mass_g"]) for row in csv.DictReader(handle))
    assert mass / 1000 == pytest.approx(revision["model"]["total_allocation_kg"])


def test_validation_evidence_is_fresh_and_scoped():
    report = json.loads((HARDWARE / "validation.json").read_text())
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    assert report["packing"]["pose_count"] == 1050
    assert report["packing"]["status"] == "pass"
    assert len(report["standing"]) == 8
    assert all(row["status"] == "pass" for row in report["standing"])
    assert report["crouch"]["status"] == "pass"
    assert "No trained walking" in report["limitations"]
