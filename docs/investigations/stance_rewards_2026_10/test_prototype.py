"""Independent physics and measurement checks for the research wrapper."""

import mujoco
import numpy as np
import pytest
from prototype import VARIANTS, BudgetSchedule, ExperimentalReward, Variant, cop_margin_cost

from environments.shared.config import load_stage_config
from environments.shared.curriculum.schedules import CosineSchedule, LinearSchedule
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

FACTORIES = {"trex": TRexEnv, "velociraptor": RaptorEnv}


def environment(species, **overrides):
    cfg = load_stage_config(species, "stance")
    return FACTORIES[species](**(cfg["env_kwargs"] | overrides))


def test_margin_prices_the_worst_foot_and_missing_support():
    assert cop_margin_cost(np.array([0.3, 0.6]), np.array([True, True])) == 0
    expected = ((0.9 - 0.65) / 0.35) ** 2
    assert cop_margin_cost(np.array([0.3, 0.9]), np.array([True, True])) == pytest.approx(expected)
    assert cop_margin_cost(np.array([-0.9, 0.3]), np.array([True, True])) == pytest.approx(expected)
    assert cop_margin_cost(np.array([np.nan, 0.3]), np.array([False, True])) == 1
    assert cop_margin_cost(np.array([0.0, 0.3]), np.array([False, True])) == 1
    assert cop_margin_cost(np.array([1.2, 0.3]), np.array([True, True])) == 1


@pytest.mark.parametrize("schedule", [LinearSchedule(3e-5, 1e-5), CosineSchedule(3e-5, 5e-6)])
def test_short_pilot_does_not_accelerate_the_production_learning_rate_decay(schedule):
    anchored = BudgetSchedule(schedule, 1_048_576, 11_000_000)
    for elapsed in [0, 8192, 524_288, 1_048_576]:
        assert anchored(1 - elapsed / 1_048_576) == pytest.approx(schedule(1 - elapsed / 11_000_000))


@pytest.mark.parametrize("species", FACTORIES)
def test_streaming_measurements_match_the_independent_episode_recorder(species):
    env = ExperimentalReward(environment(species), VARIANTS[species, "control"], settle_steps=100)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, species)).attach()
    rows = []
    try:
        env.reset(seed=3042)
        for _ in range(320):
            action = np.zeros(env.action_space.shape)
            _, reward, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated
            recorder.end_step(action, reward)
            rows.append(info)
        trace = recorder.end_episode(terminated=False, truncated=False, termination_reason="measurement_check")
        np.testing.assert_allclose(
            [r["experiment_cop"] for r in rows], trace.sole_cop[:, :, 0], atol=1e-12, rtol=1e-12, equal_nan=True
        )
        np.testing.assert_array_equal([r["experiment_sole_loaded"] for r in rows], trace.sole_contacts_mean > 0)
        np.testing.assert_allclose(
            [r["experiment_peak_floor_force_bw"] for r in rows], trace.total_floor_max / trace.body_weight_n, atol=1e-12
        )
        np.testing.assert_allclose(
            [r["experiment_airborne_fraction"] for r in rows],
            trace.feet_airborne_substeps / trace.frame_skip,
            atol=1e-12,
        )
        support_foot = np.asarray(trace.support_geom_foot)
        coverage = np.min(
            np.stack([trace.support_loaded_frac[:, support_foot == i].mean(axis=1) for i in range(trace.n_feet)]),
            axis=0,
        )
        np.testing.assert_allclose([r["experiment_support_coverage"] for r in rows], coverage, atol=1e-12)
    finally:
        recorder.detach()
        env.close()


@pytest.mark.parametrize("species", FACTORIES)
def test_control_is_bit_identical_to_the_unwrapped_environment(species):
    base = environment(species)
    wrapped = ExperimentalReward(environment(species), VARIANTS[species, "control"], settle_steps=200)
    rng = np.random.default_rng(991)
    try:
        a, _ = base.reset(seed=3048)
        b, _ = wrapped.reset(seed=3048)
        np.testing.assert_array_equal(a, b)
        for _ in range(120):
            action = rng.normal(0, 0.06, size=base.action_space.shape)
            a, ar, at, ax, _ = base.step(action)
            b, br, bt, bx, _ = wrapped.step(action)
            np.testing.assert_array_equal(base.data.qpos, wrapped.unwrapped.data.qpos)
            np.testing.assert_array_equal(base.data.qvel, wrapped.unwrapped.data.qvel)
            np.testing.assert_array_equal(a, b)
            assert ar == br and at == bt and ax == bx
            if at or ax:
                base.reset(seed=3048)
                wrapped.reset(seed=3048)
    finally:
        base.close()
        wrapped.close()


def test_support_delta_matches_a_real_env_configuration_change():
    reference = environment("velociraptor", support_conditioned_alive_fraction=0.5)
    wrapped = ExperimentalReward(
        environment("velociraptor"), VARIANTS["velociraptor", "support_fraction"], settle_steps=100
    )
    rng = np.random.default_rng(444)
    try:
        reference.reset(seed=3042)
        wrapped.reset(seed=3042)
        for _ in range(100):
            action = rng.normal(0, 0.15, size=reference.action_space.shape)
            a, ar, at, ax, _ = reference.step(action)
            b, br, bt, bx, _ = wrapped.step(action)
            np.testing.assert_array_equal(a, b)
            assert ar == pytest.approx(br, abs=1e-12)
            assert at == bt and ax == bx
            if at or ax:
                break
    finally:
        reference.close()
        wrapped.close()


def test_probe_normal_force_decoder_matches_mujoco_api():
    env = ExperimentalReward(environment("trex"), VARIANTS["trex", "control"], settle_steps=200)
    try:
        env.reset(seed=3042)
        for _ in range(20):
            env.step(np.zeros(env.action_space.shape))
            decoded = env.probe.decoder._normal_forces(env.unwrapped.data.contact, env.unwrapped.data.efc_force)
            expected = []
            for index in range(env.unwrapped.data.ncon):
                force = np.empty(6)
                mujoco.mj_contactForce(env.unwrapped.model, env.unwrapped.data, index, force)
                expected.append(force[0])
            np.testing.assert_allclose(decoded, expected, atol=1e-12, rtol=1e-12)
    finally:
        env.close()


def test_probe_fails_if_its_hook_is_displaced():
    env = ExperimentalReward(environment("trex"), VARIANTS["trex", "control"], settle_steps=200)
    hook = env.unwrapped._substep_probe_hook
    try:
        env.reset(seed=3042)
        env.unwrapped._substep_probe_hook = None
        with pytest.raises(RuntimeError, match="hook was displaced"):
            env.step(np.zeros(env.action_space.shape))
    finally:
        env.unwrapped._substep_probe_hook = hook
        env.close()


@pytest.mark.parametrize(
    "kwargs", [{"cop_weight": -1}, {"airborne_weight": np.nan}, {"cop_safe_fraction": 1}, {"support_fraction": 1.1}]
)
def test_invalid_variants_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Variant("invalid", "trex", **kwargs)
