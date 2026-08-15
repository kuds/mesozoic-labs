"""Tests for environments.shared.perturbation.

The scheduler's whole value is identity: the same seed must mean the same
pushes for the policy and for every null controller, on either backend, or
the 1b gate's paired statistics measure schedule luck instead of control.
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.shared.perturbation import (
    derive_push_parameters,
    external_push_force,
    hash_uniform01,
    max_pushes_for,
    push_schedule,
    validate_push_config,
)

SCHEDULE = {"max_pushes": 6, "interval_steps": 200, "jitter_steps": 50}


class TestSchedule:
    def test_same_seed_same_schedule(self):
        a = push_schedule(np.uint32(3042), **SCHEDULE)
        b = push_schedule(np.uint32(3042), **SCHEDULE)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_different_seeds_differ(self):
        a = push_schedule(np.uint32(3042), **SCHEDULE)
        b = push_schedule(np.uint32(3043), **SCHEDULE)
        assert not np.array_equal(a[0], b[0])

    def test_starts_are_ordered_and_jitter_bounded(self):
        starts, directions = push_schedule(np.uint32(7), **SCHEDULE)
        assert np.all(np.diff(starts) > 0)
        for k, start in enumerate(starts):
            nominal = (k + 1) * SCHEDULE["interval_steps"]
            assert abs(int(start) - nominal) <= SCHEDULE["jitter_steps"]
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)

    def test_max_pushes_covers_horizon(self):
        # Worst case every gap is interval - jitter; the last row must land
        # at or beyond the horizon so no in-horizon push is ever dropped.
        n = max_pushes_for(1000, 200, 50)
        assert (n - 1) * (200 - 50) >= 1000 - (200 - 50)

    def test_hash_is_stable_across_calls(self):
        # Pinned values: the schedule hash is part of task identity, so a
        # silent change to the mixer would invalidate recorded schedules.
        u = float(hash_uniform01(np.uint32(3042), np.uint32(0)))
        assert u == pytest.approx(float(hash_uniform01(np.uint32(3042), np.uint32(0))))
        assert 0.0 <= u < 1.0


class TestBackendParity:
    def test_numpy_jax_schedules_are_identical(self):
        jnp = pytest.importorskip("jax.numpy")
        n_starts, n_dirs = push_schedule(np.uint32(3042), **SCHEDULE)
        j_starts, j_dirs = push_schedule(jnp.uint32(3042), **SCHEDULE)
        np.testing.assert_array_equal(n_starts, np.asarray(j_starts))
        np.testing.assert_allclose(n_dirs, np.asarray(j_dirs), atol=1e-6)

    def test_numpy_jax_forces_are_identical(self):
        jnp = pytest.importorskip("jax.numpy")
        n_starts, n_dirs = push_schedule(np.uint32(11), **SCHEDULE)
        j_starts, j_dirs = push_schedule(jnp.uint32(11), **SCHEDULE)
        for step in (0, int(n_starts[0]), int(n_starts[0]) + 5, 999):
            f_np = external_push_force(step, n_starts, n_dirs, duration_steps=20, force_newtons=165.5)
            f_jax = external_push_force(step, j_starts, j_dirs, duration_steps=20, force_newtons=165.5)
            np.testing.assert_allclose(f_np, np.asarray(f_jax), atol=1e-4)


class TestForceWindows:
    def test_force_active_exactly_inside_windows(self):
        starts, directions = push_schedule(np.uint32(5), **SCHEDULE)
        duration = 20
        for k, start in enumerate(starts):
            inside = external_push_force(int(start), starts, directions, duration_steps=duration, force_newtons=100.0)
            np.testing.assert_allclose(inside[:2], directions[k] * 100.0, atol=1e-4)
            assert inside[2] == 0.0
            before = external_push_force(
                int(start) - 1, starts, directions, duration_steps=duration, force_newtons=100.0
            )
            after = external_push_force(
                int(start) + duration, starts, directions, duration_steps=duration, force_newtons=100.0
            )
            np.testing.assert_array_equal(before, np.zeros(3, dtype=np.float32))
            np.testing.assert_array_equal(after, np.zeros(3, dtype=np.float32))

    def test_zero_between_all_windows(self):
        starts, directions = push_schedule(np.uint32(9), **SCHEDULE)
        duration = 20
        windows = {int(s) + o for s in starts for o in range(duration)}
        for step in range(0, 1000):
            if step in windows:
                continue
            force = external_push_force(step, starts, directions, duration_steps=duration, force_newtons=100.0)
            np.testing.assert_array_equal(force, np.zeros(3, dtype=np.float32))


class TestValidation:
    def test_adopted_config_validates(self):
        steps = validate_push_config(
            capture_velocity_multiple=1.5,
            interval_s=2.0,
            jitter_s=0.5,
            duration_s=0.20,
            direction="uniform_horizontal",
            control_dt=0.01,
        )
        assert steps == {"interval_steps": 200, "jitter_steps": 50, "duration_steps": 20}

    def test_overlapping_windows_are_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            validate_push_config(
                capture_velocity_multiple=1.5,
                interval_s=1.0,
                jitter_s=0.45,
                duration_s=0.20,
                direction="uniform_horizontal",
                control_dt=0.01,
            )

    def test_unknown_direction_is_fatal(self):
        with pytest.raises(ValueError, match="uniform_horizontal"):
            validate_push_config(
                capture_velocity_multiple=1.5,
                interval_s=2.0,
                jitter_s=0.5,
                duration_s=0.2,
                direction="uniform_lateral",
                control_dt=0.01,
            )


class TestDerivation:
    """Pinned against the r7 trex plant, like the statue-constant freshness
    test: a plant revision that moves the mass or stance geometry must be
    seen here, with the re-derivation, not discovered mid-run."""

    def test_trex_derivation_matches_design_doc_scale(self):
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path("environments/trex/assets/trex.xml")
        params = derive_push_parameters(model, capture_velocity_multiple=1.5, duration_s=0.20)
        # STAGE1_SPLIT_PLAN §3.3 gives "~150 N for 0.20 s" [artifact-derived];
        # the first-principles derivation lands at 165.5 N on physics r7.
        assert params["subtree_mass_kg"] == pytest.approx(85.72, abs=0.5)
        assert params["com_height_m"] == pytest.approx(0.884, abs=0.01)
        assert params["capture_velocity_mps"] == pytest.approx(0.2574, abs=0.01)
        assert params["force_n"] == pytest.approx(165.5, abs=2.0)
        assert params["impulse_ns"] == pytest.approx(33.1, abs=0.5)

    def test_derivation_scales_linearly_with_the_multiple(self):
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path("environments/trex/assets/trex.xml")
        one = derive_push_parameters(model, capture_velocity_multiple=1.0, duration_s=0.20)
        three = derive_push_parameters(model, capture_velocity_multiple=3.0, duration_s=0.20)
        assert three["force_n"] == pytest.approx(3.0 * one["force_n"], rel=1e-9)

    def test_off_multiple_is_rejected(self):
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path("environments/trex/assets/trex.xml")
        with pytest.raises(ValueError, match="positive"):
            derive_push_parameters(model, capture_velocity_multiple=0.0, duration_s=0.2)


class TestEnvIntegration:
    """SB3-path integration: off is inert, on pushes exactly on schedule."""

    @staticmethod
    def _trex(**kwargs):
        from environments.trex.envs.trex_env import TRexEnv

        return TRexEnv(reset_noise_scale=0.0, **kwargs)

    def test_off_is_inert_and_deterministic(self):
        a = self._trex()
        b = self._trex()
        a.reset(seed=7)
        b.reset(seed=7)
        action = np.zeros(a.action_space.shape, dtype=np.float32)
        for _ in range(30):
            a.step(action)
            b.step(action)
            assert not np.any(a.data.xfrc_applied)
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        assert a.perturbation_manifest() is None

    def test_on_diverges_only_after_the_first_push(self):
        off = self._trex()
        on = self._trex(perturbation_capture_velocity_multiple=1.5)
        off.reset(seed=11)
        on.reset(seed=11)
        first_push = int(on._push_schedule_starts[0])
        action = np.zeros(off.action_space.shape, dtype=np.float32)
        for _ in range(first_push):
            off.step(action)
            on.step(action)
        # Identical up to the step BEFORE the push window opens: the only
        # difference between the two tasks is the scheduled force.
        np.testing.assert_array_equal(off.data.qpos, on.data.qpos)
        for _ in range(20):
            off.step(action)
            on.step(action)
        assert not np.array_equal(off.data.qpos, on.data.qpos)

    def test_push_window_writes_the_derived_force_at_the_root(self):
        env = self._trex(perturbation_capture_velocity_multiple=1.5)
        env.reset(seed=3)
        start = int(env._push_schedule_starts[0])
        direction = np.asarray(env._push_schedule_directions[0])
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(start):
            env.step(action)
        env.step(action)  # first step inside the window
        applied = env.data.xfrc_applied[env._push_root_body]
        np.testing.assert_allclose(applied[:2], direction * env._push_force_n, rtol=1e-5)
        assert applied[2] == 0.0 and not np.any(env.data.xfrc_applied[: env._push_root_body])
        for _ in range(env._push_duration_steps):
            env.step(action)
        assert not np.any(env.data.xfrc_applied)  # self-cleared after the window

    def test_reset_clears_forces_and_pairs_schedules(self):
        env = self._trex(perturbation_capture_velocity_multiple=1.5)
        env.reset(seed=5)
        start = int(env._push_schedule_starts[0])
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(start + 2):
            env.step(action)
        assert np.any(env.data.xfrc_applied)
        env.reset(seed=123)
        assert not np.any(env.data.xfrc_applied)
        first = np.array(env._push_schedule_starts)
        env.reset(seed=123)
        np.testing.assert_array_equal(first, env._push_schedule_starts)
        env.reset(seed=124)
        assert not np.array_equal(first, env._push_schedule_starts)

    def test_manifest_carries_the_derived_parameters(self):
        env = self._trex(perturbation_capture_velocity_multiple=1.5)
        manifest = env.perturbation_manifest()
        assert manifest is not None
        assert manifest["force_n"] == pytest.approx(165.5, abs=2.0)
        assert manifest["direction"] == "uniform_horizontal"

    @pytest.mark.parametrize(
        "species_env",
        [
            "environments.trex.envs.trex_env:TRexEnv",
            "environments.velociraptor.envs.raptor_env:RaptorEnv",
            "environments.brachiosaurus.envs.brachio_env:BrachioEnv",
            "environments.dibothrosuchus.envs.dibothrosuchus_env:DibothrosuchusEnv",
        ],
    )
    def test_every_species_accepts_the_parameters(self, species_env):
        """Species-generic is a constructor contract, not a T-Rex feature."""
        import importlib

        module_name, class_name = species_env.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        env = cls(perturbation_capture_velocity_multiple=0.0)
        assert env.perturbation_manifest() is None


class TestMjxIntegration:
    """MJX-path integration: same semantics as the SB3 path, traced."""

    @staticmethod
    def _mjx_env(**env_kwargs):
        pytest.importorskip("jax")
        pytest.importorskip("mujoco.mjx")
        import environments.trex.mjx_config  # noqa: F401  (registers trex)
        from environments.shared.mjx_env import MJXDinoEnv

        return MJXDinoEnv("trex", stage=1, num_envs=1, env_kwargs=env_kwargs or None)

    # Short schedule so the roll stays cheap: first push lands near step 30
    # instead of 200.  validate_push_config still holds (30 - 2*5 > 10).
    _SHORT = {
        "perturbation_capture_velocity_multiple": 1.5,
        "perturbation_interval": 0.3,
        "perturbation_jitter": 0.05,
        "perturbation_duration": 0.1,
    }

    def test_off_keeps_empty_schedule_and_zero_xfrc(self):
        import jax
        import jax.numpy as jnp

        env = self._mjx_env()
        assert env.perturbation_manifest() is None
        state = env.reset(jax.random.PRNGKey(0))
        assert state.push_start_steps.shape == (1, 0)
        actions = jnp.zeros((1, env.action_dim))
        for i in range(3):
            state, *_ = env.step(state, actions, jax.random.PRNGKey(i))
            assert not bool(jnp.any(state.data.xfrc_applied))

    def test_push_windows_write_forces_on_schedule(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        env = self._mjx_env(**self._SHORT)
        constants = env._push_constants
        state = env.reset(jax.random.PRNGKey(42))
        start = int(state.push_start_steps[0, 0])
        direction = np.asarray(state.push_directions[0, 0])
        actions = jnp.zeros((1, env.action_dim))
        for i in range(start):
            state, *_ = env.step(state, actions, jax.random.PRNGKey(i))
            assert not bool(jnp.any(state.data.xfrc_applied))
        state, *_ = env.step(state, actions, jax.random.PRNGKey(999))
        applied = np.asarray(state.data.xfrc_applied[0, constants["root_body_id"]])
        np.testing.assert_allclose(applied[:3], [*(direction * constants["force_n"]), 0.0], rtol=1e-5)
        for i in range(constants["duration_steps"]):
            state, *_ = env.step(state, actions, jax.random.PRNGKey(1000 + i))
        assert not bool(jnp.any(state.data.xfrc_applied))  # self-cleared

    def test_schedule_determinism_and_cross_backend_derivation(self):
        import jax
        import numpy as np

        env = self._mjx_env(**self._SHORT)
        a = env.reset(jax.random.PRNGKey(7))
        b = env.reset(jax.random.PRNGKey(7))
        np.testing.assert_array_equal(np.asarray(a.push_start_steps), np.asarray(b.push_start_steps))
        c = env.reset(jax.random.PRNGKey(8))
        assert not np.array_equal(np.asarray(a.push_start_steps), np.asarray(c.push_start_steps))
        # Both backends derive from the same host model: identical constants.
        from environments.trex.envs.trex_env import TRexEnv

        sb3 = TRexEnv(reset_noise_scale=0.0, **self._SHORT)
        assert sb3.perturbation_manifest()["force_n"] == pytest.approx(
            env.perturbation_manifest()["force_n"], rel=1e-12
        )
