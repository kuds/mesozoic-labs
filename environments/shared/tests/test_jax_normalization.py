"""Tests for jax_normalization (RunningMeanStd / Welford updates)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def _jnp():
    jnp = pytest.importorskip("jax.numpy")
    return jnp


class TestRunningMeanStdCreate:
    def test_initial_state(self, _jnp):
        from environments.shared.jax_normalization import RunningMeanStd

        stats = RunningMeanStd.create(5)
        assert stats.mean.shape == (5,)
        assert stats.var.shape == (5,)
        # Initial mean=0, var=1, count=tiny so the first update dominates.
        assert float(_jnp.max(_jnp.abs(stats.mean))) == pytest.approx(0.0)
        assert float(_jnp.max(_jnp.abs(stats.var - 1.0))) == pytest.approx(0.0)
        assert stats.count < 1.0


class TestUpdateRunningStats:
    """Welford updates must converge to the true distribution stats."""

    def test_converges_to_true_mean_and_var(self, _jnp):
        import numpy as np

        from environments.shared.jax_normalization import RunningMeanStd, update_running_stats

        rng = np.random.default_rng(42)
        true_mean = np.array([3.0, -2.0, 7.0])
        true_std = np.array([0.5, 1.5, 2.5])

        stats = RunningMeanStd.create(3)
        for _ in range(20):
            batch = rng.normal(true_mean, true_std, size=(512, 3))
            stats = update_running_stats(stats, _jnp.array(batch))

        # Tolerance: with 20 batches of 512 the Welford estimate should
        # be within ~1% of the true value.
        assert float(_jnp.max(_jnp.abs(stats.mean - _jnp.array(true_mean)))) < 0.1
        assert float(_jnp.max(_jnp.abs(_jnp.sqrt(stats.var) - _jnp.array(true_std)))) < 0.1

    def test_does_not_collapse_when_fed_raw_obs(self, _jnp):
        """Bug #4 regression: feeding RAW obs to update_running_stats
        and then normalizing must NOT make the distribution drift
        toward N(0, 1) — that's the failure mode of feeding back
        already-normalized obs.  We verify that after several updates,
        the running mean still tracks the raw distribution mean.
        """
        import numpy as np

        from environments.shared.jax_normalization import (
            RunningMeanStd,
            update_running_stats,
        )

        rng = np.random.default_rng(0)
        # Heavily off-zero distribution
        stats = RunningMeanStd.create(2)
        for _ in range(10):
            batch = rng.normal([10.0, 5.0], 0.3, size=(256, 2))
            stats = update_running_stats(stats, _jnp.array(batch))

        # Mean should clearly NOT be ~0 — that'd indicate normalized
        # data was being fed back.
        assert float(stats.mean[0]) > 5.0
        assert float(stats.mean[1]) > 2.5


class TestNormalizeObs:
    def test_zero_distribution_normalizes_to_zero(self, _jnp):
        """An obs equal to the running mean normalizes to 0."""
        from environments.shared.jax_normalization import (
            RunningMeanStd,
            normalize_obs,
            update_running_stats,
        )

        # Update with constant data so mean is exactly 5.0 and var is small.
        stats = RunningMeanStd.create(1)
        batch = _jnp.full((100, 1), 5.0)
        stats = update_running_stats(stats, batch)

        normed = normalize_obs(_jnp.array([5.0]), stats)
        assert float(_jnp.abs(normed[0])) < 1e-3

    def test_clipping(self, _jnp):
        from environments.shared.jax_normalization import RunningMeanStd, normalize_obs

        stats = RunningMeanStd.create(1)  # mean=0, var=1
        # Far-out value normalizes to its raw value (since var=1) and
        # then gets clipped to ±10.
        normed = normalize_obs(_jnp.array([50.0]), stats, clip=10.0)
        assert float(normed[0]) == pytest.approx(10.0, abs=1e-5)
        normed_neg = normalize_obs(_jnp.array([-50.0]), stats, clip=10.0)
        assert float(normed_neg[0]) == pytest.approx(-10.0, abs=1e-5)


class TestDecayRunningStats:
    """Curriculum stage transitions need to make new data dominate."""

    def test_decay_reduces_count(self, _jnp):
        from environments.shared.jax_normalization import (
            RunningMeanStd,
            decay_running_stats,
            update_running_stats,
        )

        stats = RunningMeanStd.create(2)
        batch = _jnp.ones((1000, 2)) * 3.0
        stats = update_running_stats(stats, batch)
        pre_count = stats.count
        decayed = decay_running_stats(stats, decay_factor=0.01)
        # Mean/var preserved
        assert float(_jnp.max(_jnp.abs(decayed.mean - stats.mean))) == pytest.approx(0.0)
        assert float(_jnp.max(_jnp.abs(decayed.var - stats.var))) == pytest.approx(0.0)
        # Count reduced
        assert decayed.count < pre_count
        assert decayed.count == pytest.approx(pre_count * 0.01, rel=1e-3)
