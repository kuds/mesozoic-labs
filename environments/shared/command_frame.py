"""The body-relative command frame (BEHAVIOR_RECIPES_PLAN §4.6, decision D2).

Species-generic by construction: nothing here names a species, a backend
array library, or a learning framework.  The module is importable with
numpy alone (amendment A3) — the plant contract, the MJX kernels and the
checkpoint tooling all read these constants, and none of them may pull
stable_baselines3, torch or jax in as a side effect.

Design contract:

* **The segment is appended LAST.**  Three dims — ``v_x_cmd`` (forward
  speed), ``v_y_cmd`` (lateral speed), ``yaw_rate_cmd`` — sit at the END
  of every species' observation, so every existing slice offset is
  unchanged and ``obs[-COMMAND_WIDTH:]`` is the command on every plant.
* **Pre-scaled to [-1, 1].**  The environment emits each component already
  divided by the stage's declared range (``command_speed_range``,
  ``command_lateral_range``, ``command_yaw_rate_max``), so a live command
  normalises to O(1) from the first step after a reseeded load (plan §4.6
  "Normalization of the command slice", invariant 8).
* **``"none"`` is byte-inert.**  Under ``command_mode = "none"`` the
  segment is constant zero, no RNG is drawn, and every trajectory is
  identical to the pre-Phase-C environment (the seeded draw stream is
  pinned by ``tests/fixtures/phase_c_reset_golden.json``).
* **Fail closed on every backend.**  The MJX backend refuses any live mode
  until its command path lands (invariant 9); the SB3 backend refuses
  every live mode until Phase D implements the sampler (decision D-C7).
  Phase D deletes only the SB3 branch of :func:`validate_command_mode`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only; SB3 is never imported at runtime here
    from stable_baselines3.common.running_mean_std import RunningMeanStd

COMMAND_SEGMENT_NAME = "command"
COMMAND_WIDTH = 3
COMMAND_COMPONENTS = ("v_x_cmd", "v_y_cmd", "yaw_rate_cmd")
COMMAND_RANGE = (-1.0, 1.0)
COMMAND_MODE_NONE = "none"
COMMAND_MODES = ("none", "heading", "heading_and_speed")
#: The six ``[env]`` keys / constructor kwargs (both backends, identical
#: names and defaults).  Task-level like ``perturbation_*``: they enter the
#: task fingerprint, never the plant interface.
COMMAND_ENV_KEYS = (
    "command_mode",
    "command_speed_range",
    "command_lateral_range",
    "command_yaw_rate_max",
    "command_switch_interval",
    "command_switch_jitter",
)
#: Injected into the plant-contract observation probes on BOTH backends so
#: SB3/MJX parity is asserted on a non-zero command (decision D-C4).
COMMAND_PROBE_VECTOR = (0.25, -0.5, 0.75)

COMMAND_BACKENDS = ("stable-baselines3", "jax-mjx")

MJX_COMMAND_REFUSAL = (
    "command_mode={mode!r} is not implemented on the jax-mjx backend; only 'none' is accepted until the MJX "
    "command path lands (BEHAVIOR_RECIPES_PLAN §4.6, A7: SB3 is the evidence backend and MJX fails closed on "
    "command-mode configs)"
)
SB3_COMMAND_REFUSAL = (
    "command_mode={mode!r} is reserved for BEHAVIOR_RECIPES_PLAN §4.6 Phase D; only 'none' is implemented"
)


def zero_command(xp: Any = np) -> Any:
    """The inert command: ``COMMAND_WIDTH`` float32 zeros on *xp* (numpy or jax.numpy)."""
    return xp.zeros(COMMAND_WIDTH, dtype=xp.float32)


def validate_command_mode(mode: str, *, backend: str) -> str:
    """Return *mode* when *backend* implements it; raise ``ValueError`` otherwise.

    An unknown mode is refused on every backend, naming the valid set.  The
    jax-mjx backend refuses every live mode (:data:`MJX_COMMAND_REFUSAL`);
    the stable-baselines3 backend refuses them until Phase D
    (:data:`SB3_COMMAND_REFUSAL`) — Phase D deletes only that branch.
    """
    if mode not in COMMAND_MODES:
        raise ValueError(f"command_mode={mode!r} is not one of {COMMAND_MODES}")
    if backend not in COMMAND_BACKENDS:
        raise ValueError(f"unknown training backend {backend!r} for command_mode validation; known: {COMMAND_BACKENDS}")
    if mode != COMMAND_MODE_NONE:
        if backend == "jax-mjx":
            raise ValueError(MJX_COMMAND_REFUSAL.format(mode=mode))
        raise ValueError(SB3_COMMAND_REFUSAL.format(mode=mode))
    return mode


def command_slice(observation_dim: int) -> slice:
    """The trailing command slice of an observation of *observation_dim* dims."""
    if observation_dim < COMMAND_WIDTH:
        raise ValueError(f"an observation of {observation_dim} dims cannot carry a {COMMAND_WIDTH}-dim command segment")
    return slice(observation_dim - COMMAND_WIDTH, observation_dim)


def _running_stats(obs_rms: Any) -> tuple[np.ndarray, np.ndarray]:
    """The ``mean`` / ``var`` arrays of a RunningMeanStd-like object, validated."""
    if isinstance(obs_rms, Mapping):
        raise ValueError("obs_rms must be a RunningMeanStd-like object with ndarray mean/var attributes, not a dict")
    mean = getattr(obs_rms, "mean", None)
    var = getattr(obs_rms, "var", None)
    if not isinstance(mean, np.ndarray) or not isinstance(var, np.ndarray):
        raise ValueError("obs_rms must expose ndarray mean and var attributes")
    if mean.ndim != 1 or mean.shape != var.shape:
        raise ValueError(f"obs_rms mean/var must be matching 1-D arrays, got {mean.shape} and {var.shape}")
    if mean.shape[0] < COMMAND_WIDTH:
        raise ValueError(
            f"obs_rms holds {mean.shape[0]} dims, fewer than the {COMMAND_WIDTH}-dim command segment it should end with"
        )
    return mean, var


def reseed_command_slice(obs_rms: Any) -> None:
    """Reset the trailing command slice of *obs_rms* to mean 0 / var 1, in place.

    Invariant 8: a slice that was constant zero during the parent run has
    var ≈ 1e-11 after millions of samples, and a live command of 1.0 would
    normalise to ≈ 1e4 and clip at ``clip_obs``.  The scalar ``count`` is
    left untouched so the carried statistics keep their weight; commands
    are emitted pre-scaled, so unit variance is the right prior.
    """
    mean, var = _running_stats(obs_rms)
    mean[-COMMAND_WIDTH:] = 0.0
    var[-COMMAND_WIDTH:] = 1.0


def pad_running_stats(obs_rms: Any, width: int = COMMAND_WIDTH) -> "RunningMeanStd":
    """A NEW running-statistics object with *width* reseeded dims appended.

    The widen tool's primitive: ``mean`` gains zeros, ``var`` gains ones,
    ``count`` is carried — the appended slice is born reseeded, exactly as
    :func:`reseed_command_slice` would leave it.  The padded object is built
    from the input's own class (``type(obs_rms)(shape=...)``) so this module
    never imports stable_baselines3 at module level.
    """
    mean, var = _running_stats(obs_rms)
    if width < 1:
        raise ValueError("width must be a positive number of dims")
    padded_shape = (mean.shape[0] + int(width),)
    try:
        padded = type(obs_rms)(shape=padded_shape)
    except TypeError:
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        padded = RunningMeanStd(shape=padded_shape)
    padded.mean = np.concatenate([np.asarray(mean, dtype=np.float64), np.zeros(int(width), dtype=np.float64)])
    padded.var = np.concatenate([np.asarray(var, dtype=np.float64), np.ones(int(width), dtype=np.float64)])
    padded.count = obs_rms.count
    return padded
