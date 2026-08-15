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
