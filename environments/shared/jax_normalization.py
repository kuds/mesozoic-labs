"""Running-mean observation normalisation in JAX.

Equivalent to SB3's ``VecNormalize`` but implemented as a pure
JAX pytree so it can be used inside ``jax.jit``/``jax.vmap``.

Usage::

    stats = RunningMeanStd.create(obs_dim)
    stats = update_running_stats(stats, obs_batch)
    normed = normalize_obs(obs, stats)
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .mjx_utils import check_jax


class RunningMeanStd(NamedTuple):
    """Welford online mean/variance tracker as a JAX-compatible NamedTuple."""

    mean: Any  # jnp.ndarray (lazy import)
    var: Any  # jnp.ndarray (lazy import)
    count: float  # type: ignore[assignment]  # shadows tuple.count; intentional for NamedTuple field

    @classmethod
    def create(cls, shape: int | tuple[int, ...]) -> "RunningMeanStd":
        """Create a fresh running statistics tracker."""
        check_jax()
        import jax.numpy as jnp

        if isinstance(shape, int):
            shape = (shape,)
        return cls(
            mean=jnp.zeros(shape),
            var=jnp.ones(shape),
            count=1e-4,
        )


def update_running_stats(
    stats: RunningMeanStd,
    batch: Any,
) -> RunningMeanStd:
    """Update running mean/variance with a new batch of observations.

    Uses Welford's parallel algorithm for numerical stability.

    Args:
        stats: Current running statistics.
        batch: New observation batch of shape ``(batch_size, obs_dim)``.

    Returns:
        Updated ``RunningMeanStd``.
    """
    import jax.numpy as jnp

    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    delta = batch_mean - stats.mean
    total_count = stats.count + batch_count

    new_mean = stats.mean + delta * batch_count / total_count
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * stats.count * batch_count / total_count
    new_var = m2 / total_count

    return RunningMeanStd(mean=new_mean, var=new_var, count=total_count)


def decay_running_stats(
    stats: RunningMeanStd,
    decay_factor: float = 0.01,
) -> RunningMeanStd:
    """Reduce the effective sample count so new data adapts the stats faster.

    Useful when transitioning between curriculum stages: the obs distribution
    shifts (e.g. near-zero velocity in balance → sustained velocity in
    locomotion) but a count of millions from the prior stage makes
    ``update_running_stats`` nearly a no-op.  Decaying the count lets the
    mean/var track the new distribution within a few updates.

    The mean and variance are preserved — only ``count`` is reduced so the
    next ``update_running_stats`` call gives proportionally more weight to
    incoming data.

    Args:
        stats: Current running statistics (from prior stage checkpoint).
        decay_factor: Multiply count by this factor.  ``0.01`` means new
            batches of 131K samples (2048 envs × 64 steps) will dominate
            the statistics within 1-2 updates.  Set to ``1.0`` to keep
            the prior stats unchanged (no decay).

    Returns:
        ``RunningMeanStd`` with reduced count.
    """
    return RunningMeanStd(
        mean=stats.mean,
        var=stats.var,
        count=max(stats.count * decay_factor, 1e-4),
    )


def normalize_obs(
    obs: Any,
    stats: RunningMeanStd,
    clip: float = 10.0,
) -> Any:
    """Normalise observations using running mean and variance.

    Args:
        obs: Observation array (single or batched).
        stats: Running statistics.
        clip: Clipping range for normalised values.

    Returns:
        Normalised observation array.
    """
    import jax.numpy as jnp

    normed = (obs - stats.mean) / jnp.sqrt(stats.var + 1e-8)
    return jnp.clip(normed, -clip, clip)
