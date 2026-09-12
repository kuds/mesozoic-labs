"""Push mechanics for both plants; these checks do not establish learned recovery."""

import math

import mujoco
import numpy as np
import pytest

from environments.compsognathus import MODEL_PATHS
from environments.compsognathus.envs import CompsognathusEnv, CompsognathusRobotEnv
from environments.compsognathus.model import robot_body_ids
from environments.shared.perturbation import derive_push_parameters
from environments.shared.task_fingerprint import compute_task_fingerprint, derive_stage_task_fingerprint


@pytest.fixture(params=[CompsognathusEnv, CompsognathusRobotEnv], ids=["anatomical", "robot"])
def env_class(request):
    return request.param


PUSH_KWARGS = {
    "perturbation_capture_velocity_multiple": 0.2,
    "perturbation_interval": 0.20,
    "perturbation_jitter": 0.04,
    "perturbation_duration": 0.04,
    "perturbation_direction": "uniform_horizontal",
}


def test_default_and_explicitly_disabled_pushes_are_identical(env_class):
    with env_class() as default, env_class(perturbation_capture_velocity_multiple=0.0) as off:
        first, _ = default.reset(seed=37)
        other, _ = off.reset(seed=37)
        np.testing.assert_array_equal(first, other)
        assert default.perturbation_manifest() is None
        assert off.perturbation_manifest() is None
        assert default._push_schedule_starts is None
        assert off._push_schedule_starts is None
        for _ in range(30):
            action = np.zeros(default.action_space.shape)
            actual = default.step(action)
            expected = off.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
            np.testing.assert_array_equal(default.data.qpos, off.data.qpos)
            np.testing.assert_array_equal(default.data.qvel, off.data.qvel)
            assert not np.any(default.data.xfrc_applied)


def test_enabled_push_preserves_reset_and_dynamics_until_its_first_window(env_class):
    with env_class() as off, env_class(**PUSH_KWARGS) as pushed:
        reset_off, _ = off.reset(seed=37)
        reset_pushed, _ = pushed.reset(seed=37)
        np.testing.assert_array_equal(reset_off, reset_pushed)
        np.testing.assert_array_equal(off.data.mocap_pos, pushed.data.mocap_pos)
        assert off.action_space == pushed.action_space
        assert off.observation_space == pushed.observation_space
        first_push = int(pushed._push_schedule_starts[0])
        action = np.zeros(pushed.action_space.shape)
        for _ in range(first_push):
            off.step(action)
            pushed.step(action)
            np.testing.assert_array_equal(off.data.qpos, pushed.data.qpos)
            np.testing.assert_array_equal(off.data.qvel, pushed.data.qvel)
        off.step(action)
        pushed.step(action)
        assert not np.array_equal(off.data.qvel, pushed.data.qvel)


def test_pulse_is_applied_during_physics_with_calibrated_impulse_and_clears(env_class):
    with env_class(reset_noise_scale=0, **PUSH_KWARGS) as env:
        env.reset(seed=13)
        first_push = int(env._push_schedule_starts[0])
        push_body = env._push_root_body
        direction = env._push_schedule_directions[0]
        observed = []
        env._substep_probe_hook = lambda: observed.append(env.data.xfrc_applied.copy())
        action = np.zeros(env.action_space.shape)
        for step in range(first_push + env._push_duration_steps + 1):
            obs, reward, terminated, truncated, _ = env.step(action)
            assert np.all(np.isfinite(obs)) and np.isfinite(reward)
            assert not terminated and not truncated
            expected = np.zeros_like(env.data.xfrc_applied)
            if first_push <= step < first_push + env._push_duration_steps:
                expected[push_body, :2] = direction * env._push_force_n
            # Every physics substep sees the force on the selected trunk body,
            # with no vertical force or externally applied torque.
            for sample in observed[-env.frame_skip :]:
                np.testing.assert_allclose(sample, expected, rtol=1e-6, atol=1e-12)
        assert not np.any(env.data.xfrc_applied)
        measured_impulse = np.sum(np.asarray(observed)[:, push_body, :2], axis=0) * env.model.opt.timestep
        manifest = env.perturbation_manifest()
        np.testing.assert_allclose(measured_impulse, direction * manifest["impulse_ns"], rtol=1e-6)
        assert env._push_duration_steps * env.dt == pytest.approx(manifest["duration_s"])


def test_reset_clears_an_active_force_and_reproduces_seeded_schedules_and_motion(env_class):
    with env_class(**PUSH_KWARGS) as first, env_class(**PUSH_KWARGS) as second:
        first.reset(seed=7)
        action = np.zeros(first.action_space.shape)
        for _ in range(int(first._push_schedule_starts[0]) + 1):
            first.step(action)
        assert np.any(first.data.xfrc_applied)
        reset_first, _ = first.reset(seed=123)
        reset_second, _ = second.reset(seed=123)
        assert not np.any(first.data.xfrc_applied)
        np.testing.assert_array_equal(reset_first, reset_second)
        np.testing.assert_array_equal(first._push_schedule_starts, second._push_schedule_starts)
        np.testing.assert_array_equal(first._push_schedule_directions, second._push_schedule_directions)
        for _ in range(25):
            actual = first.step(action)
            expected = second.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
            np.testing.assert_array_equal(first.data.xfrc_applied, second.data.xfrc_applied)
            if actual[2] or actual[3]:
                break
        second.reset(seed=124)
        assert not np.array_equal(first._push_schedule_directions, second._push_schedule_directions)


def test_push_magnitude_uses_this_plants_mass_geometry_and_duration(env_class):
    with (
        env_class(perturbation_capture_velocity_multiple=0.5, perturbation_duration=0.20) as nominal,
        env_class(perturbation_capture_velocity_multiple=1.0, perturbation_duration=0.20) as stronger,
        env_class(perturbation_capture_velocity_multiple=0.5, perturbation_duration=0.10) as shorter,
    ):
        params = nominal.perturbation_manifest()
        mass = float(nominal.model.body_mass[robot_body_ids(nominal.model)].sum())
        assert params["root_body_id"] == nominal.pelvis_id
        assert params["subtree_mass_kg"] == pytest.approx(mass)
        assert params["support_radius_m"] > 0
        assert params["com_height_m"] > 0
        gravity = abs(float(nominal.model.opt.gravity[2]))
        capture_velocity = params["support_radius_m"] * math.sqrt(gravity / params["com_height_m"])
        expected_impulse = mass * 0.5 * capture_velocity
        assert params["impulse_ns"] == pytest.approx(expected_impulse)
        assert params["force_n"] == pytest.approx(expected_impulse / 0.20)
        assert stronger.perturbation_manifest()["force_n"] == pytest.approx(2 * params["force_n"])
        assert shorter.perturbation_manifest()["force_n"] == pytest.approx(2 * params["force_n"])
        assert shorter.perturbation_manifest()["impulse_ns"] == pytest.approx(params["impulse_ns"])
        assert params["direction"] == "uniform_horizontal"
        assert params["interval_s"] == 2.0
        assert params["jitter_s"] == 0.5


def test_robot_force_targets_the_physical_core_com_instead_of_the_massless_root():
    with CompsognathusRobotEnv(reset_noise_scale=0, **PUSH_KWARGS) as env:
        env.reset(seed=0)
        params = env.perturbation_manifest()
        core = env.model.body("core").id
        assert env.model.body_mass[env.pelvis_id] == 0
        assert params["root_body_id"] == env.pelvis_id
        assert params["push_body_id"] == core == env._push_root_body
        assert env.model.body_mass[core] == pytest.approx(0.505)
        assert env.model.body_weldid[core] == env.pelvis_id
        assert env.data.xipos[core, 2] == pytest.approx(0.288, abs=1e-9)
        # A force on the root's synthetic inertial point would act 15 cm
        # above the trunk COM and introduce an artificial tipping moment.
        assert env.data.xipos[env.pelvis_id, 2] - env.data.xipos[core, 2] == pytest.approx(0.15)
        first_push = int(env._push_schedule_starts[0])
        for _ in range(first_push + 1):
            env.step(np.zeros(env.action_space.shape))
        assert np.any(env.data.xfrc_applied[core, :2])
        assert not np.any(env.data.xfrc_applied[env.pelvis_id])


def _model_with_a_massless_root_and_heavy_articulated_limb():
    return mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 .1"/>
            <body name="root" pos="0 0 .20">
              <freejoint/>
              <body name="torso" pos="0 0 .08">
                <inertial pos="0 0 0" mass=".5" diaginertia=".01 .01 .01"/>
              </body>
              <body name="smaller_fixed_segment" pos="0 0 .03">
                <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
              </body>
              <body name="jointed_limb">
                <joint axis="0 1 0"/>
                <inertial pos="0 0 -.08" mass="4" diaginertia=".01 .01 .01"/>
                <body name="heavy_fixed_foot" pos="0 0 -.19">
                  <inertial pos="0 0 0" mass="20" diaginertia=".01 .01 .01"/>
                  <geom type="sphere" pos="-.05 -.05 0" size=".011"/>
                  <geom type="sphere" pos="-.05 .05 0" size=".011"/>
                  <geom type="sphere" pos=".05 -.05 0" size=".011"/>
                  <geom type="sphere" pos=".05 .05 0" size=".011"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
    """)


def test_massless_root_fallback_never_crosses_an_articulated_joint():
    model = _model_with_a_massless_root_and_heavy_articulated_limb()
    params = derive_push_parameters(model, capture_velocity_multiple=1, duration_s=0.2)
    assert params["root_body_id"] == model.body("root").id
    assert params["push_body_id"] == model.body("torso").id
    assert params["subtree_mass_kg"] == pytest.approx(24.6)
    # Even the fixed foot is excluded: it is welded to the jointed limb,
    # not to the free root, despite outweighing the torso forty to one.
    assert model.body_mass[model.body("heavy_fixed_foot").id] == 20


def test_massless_root_without_a_massive_rigid_segment_is_rejected():
    model = _model_with_a_massless_root_and_heavy_articulated_limb()
    model.body_mass[model.body("torso").id] = 0
    model.body_mass[model.body("smaller_fixed_segment").id] = 0
    with pytest.raises(ValueError, match="no massive rigidly attached body"):
        derive_push_parameters(model, capture_velocity_multiple=1, duration_s=0.2)


def test_massive_anatomical_root_keeps_its_original_force_target_and_manifest():
    with CompsognathusEnv(**PUSH_KWARGS) as env:
        assert env.model.body_mass[env.pelvis_id] > 0
        assert env._push_root_body == env.pelvis_id
        assert "push_body_id" not in env.perturbation_manifest()


def test_task_fingerprint_records_the_resolved_force_target(env_class):
    with env_class(**PUSH_KWARGS) as env:
        species = "compsognathus_robot" if env.variant == "robot" else "compsognathus"
        plant = {"model_path": str(MODEL_PATHS[env.variant])}
        kwargs = {
            "species": species,
            "stage": "recovery",
            "backend": "stable-baselines3",
            "env_kwargs": PUSH_KWARGS,
            "plant_identity": plant,
        }
        runtime = compute_task_fingerprint(**kwargs, perturbation_manifest=env.perturbation_manifest())
        recorded = derive_stage_task_fingerprint(**kwargs)
        assert recorded == runtime
        if env.variant == "robot":
            assert runtime["perturbation"]["push_body_id"] == env.model.body("core").id
            old_target = dict(env.perturbation_manifest())
            old_target.pop("push_body_id")
            previous = compute_task_fingerprint(**kwargs, perturbation_manifest=old_target)
            assert previous["task_sha256"] != runtime["task_sha256"]


@pytest.mark.parametrize(
    "invalid",
    [
        {"perturbation_capture_velocity_multiple": -1.0},
        {"perturbation_capture_velocity_multiple": np.nan},
        {"perturbation_capture_velocity_multiple": np.inf},
        {"perturbation_interval": np.nan},
        {"perturbation_interval": np.inf},
        {"perturbation_interval": 0.0},
        {"perturbation_jitter": np.nan},
        {"perturbation_jitter": np.inf},
        {"perturbation_jitter": -0.01},
        {"perturbation_duration": np.nan},
        {"perturbation_duration": np.inf},
        {"perturbation_duration": 0.0},
        {"perturbation_duration": 0.005},
        {"perturbation_interval": 0.20, "perturbation_jitter": 0.09, "perturbation_duration": 0.04},
        {"perturbation_direction": "vertical"},
    ],
)
def test_invalid_pushes_fail_before_an_episode_can_start(env_class, invalid):
    with pytest.raises(ValueError, match="perturbation"):
        env_class(**{**PUSH_KWARGS, **invalid})
