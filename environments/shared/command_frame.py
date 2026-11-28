"""The body-relative command frame (BEHAVIOR_RECIPES_PLAN §4.6, decision D2).

Species-generic by construction: nothing here names a species, a backend
array library, or a learning framework.  The module is importable with
numpy alone (amendment A3) — the plant contract and the checkpoint tooling
read these constants, and neither may pull stable_baselines3 or torch in as
a side effect.

Design contract:

* **The segment is appended LAST.**  Three dims — ``v_x_cmd`` (forward
  speed), ``v_y_cmd`` (lateral speed), ``yaw_rate_cmd`` — sit at the END
  of every species' observation, so every existing slice offset is
  unchanged and ``obs[-COMMAND_WIDTH:]`` is the command on every plant.
* **Pre-scaled to [-1, 1].**  The environment emits each component already
  divided by the task's declared scale (``DirectionCommandConfig.speed_scale``,
  ``lateral_speed_scale``, ``yaw_rate_scale``).
* **Identity normalization (decision D-D60).**  Every stage's training
  (``train_base``: the CLI ``train`` and ``curriculum`` and the notebook)
  holds the slice's ``VecNormalize`` statistics at mean 0 / variance 1
  (:func:`hold_command_identity`, applied by
  ``curriculum.command_identity.CommandIdentityCallback`` and before every
  sidecar save), so a normalised command equals the raw one in float32 and
  means the same in every stage: zero under ``"none"``, the controller's
  request under a live mode.  A load refuses statistics whose slice is not at
  identity where a live command would read them
  (:func:`require_command_identity`) and re-imposes identity where the stage
  is ``"none"``, whose zero command normalises to zero under either.  The
  recipe pipeline (``train_behaviors``) keeps decision D-D3's rule: its
  preparation sets the slice to identity and its statistics keep updating;
  decision D-D9 keeps its artifacts out of every stage.
* **``"none"`` is byte-inert.**  Under ``command_mode = "none"`` the
  segment is constant zero, no RNG is drawn, and every trajectory is
  identical to the pre-Phase-C environment (the seeded draw stream is
  pinned by ``tests/fixtures/phase_c_reset_golden.json``).
* **One backend.**  Stable-Baselines3 implements both live modes since
  Phase D (consolidation PR-9).  The jax-mjx backend, which refused every
  live mode (invariant 9), was removed with the frozen MJX interface core
  (configs/plant_versions.toml note 23), so :func:`validate_command_mode`
  refuses it as an unknown backend.
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
#: The two ``[env]`` keys / constructor kwargs: the mode and ONE
#: ``DirectionCommandConfig`` (decision D-D2 replaced five numeric kwargs).
#: Task-level like ``perturbation_*``: they enter the task fingerprint, never
#: the plant interface.
COMMAND_ENV_KEYS = ("command_mode", "command_config")
#: The five numeric kwargs decision D-D2 retired (consolidation PR-9).  The
#: constructors refuse them (``TypeError``) and the task fingerprint refuses
#: them by name; a ``stage_config.json`` written before PR-9 records them
#: (``reward_weights``), so a reader that rebuilds an env or re-derives a
#: task from such a record drops these keys first.
RETIRED_COMMAND_ENV_KEYS = (
    "command_speed_range",
    "command_lateral_range",
    "command_yaw_rate_max",
    "command_switch_interval",
    "command_switch_jitter",
)
#: The domain-separation word of every command stream's ``SeedSequence``:
#: ``BaseDinoEnv._command_rng`` (reset seed, episode index) and the behavior
#: env's (run seed, episode seed, episode index).  A live command never draws
#: from the reset stream, so it moves no joint, velocity or push draw.
COMMAND_STREAM_DOMAIN = 0xC044
#: Injected into the plant-contract observation probe so it exercises the
#: appended slot on a non-zero command (decision D-C4).
COMMAND_PROBE_VECTOR = (0.25, -0.5, 0.75)

COMMAND_BACKENDS = ("stable-baselines3",)


def zero_command(xp: Any = np) -> Any:
    """The inert command: ``COMMAND_WIDTH`` float32 zeros on the array module *xp* (numpy by default)."""
    return xp.zeros(COMMAND_WIDTH, dtype=xp.float32)


def validate_command_mode(mode: str, *, backend: str) -> str:
    """Return *mode* when *backend* implements it; raise ``ValueError`` otherwise.

    An unknown mode is refused, naming the valid set, and so is any backend
    but :data:`COMMAND_BACKENDS` (the removed jax-mjx included); the
    stable-baselines3 backend accepts every mode.
    """
    if mode not in COMMAND_MODES:
        raise ValueError(f"command_mode={mode!r} is not one of {COMMAND_MODES}")
    if backend not in COMMAND_BACKENDS:
        raise ValueError(f"unknown training backend {backend!r} for command_mode validation; known: {COMMAND_BACKENDS}")
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


class CommandNormalizationError(ValueError):
    """``VecNormalize`` statistics whose command slice is not at identity, where a live command would read them."""


def command_identity_gap(obs_rms: Any) -> float:
    """``max(|mean|, |var - 1|)`` over the trailing command slice of *obs_rms*: ``0.0`` exactly at identity.

    ``nan`` when a statistic is not finite, which no comparison treats as identity.
    """
    mean, var = _running_stats(obs_rms)
    deviations = np.concatenate([np.abs(mean[-COMMAND_WIDTH:]), np.abs(var[-COMMAND_WIDTH:] - 1.0)])
    if not np.all(np.isfinite(deviations)):
        return float("nan")
    return float(np.max(deviations))


def hold_command_identity(obs_rms: Any) -> bool:
    """Set the trailing command slice of *obs_rms* to mean 0 / var 1, in place; return whether a value changed.

    Decision D-D60: the slice is held at identity in every stage, so a
    normalised command is the raw, pre-scaled command (``x / sqrt(1 + eps)``
    rounds back to ``x`` in float32) and means the same to the stance, the
    walk and every follow node.  The scalar ``count`` is shared by every dim
    and left untouched, so the other dims' statistics are exactly as they
    were.  Under ``"none"`` the command is constant zero and normalises to
    zero under any statistics with mean 0, so holding the slice changes no
    output of a command-free stage.
    """
    mean, var = _running_stats(obs_rms)
    changed = bool(np.any(mean[-COMMAND_WIDTH:] != 0.0) or np.any(var[-COMMAND_WIDTH:] != 1.0))
    mean[-COMMAND_WIDTH:] = 0.0
    var[-COMMAND_WIDTH:] = 1.0
    return changed


def require_command_identity(obs_rms: Any, *, artifact: str, command_mode: str) -> None:
    """Refuse (:class:`CommandNormalizationError`) statistics whose command slice is not at identity.

    *artifact* names the statistics and *command_mode* the live mode that
    would read commands through them; both go into the message.  Statistics
    saved by training since decision D-D60 are at identity; others were saved
    before it (a ``"none"`` stage's slice then decayed toward variance 0 as
    ``1e-4 / count``, a recipe's followed its commands) or were edited.
    """
    gap = command_identity_gap(obs_rms)
    if gap == 0.0:
        return
    mean, var = _running_stats(obs_rms)
    raise CommandNormalizationError(
        f"{artifact}: the command slice of these observation statistics is not at identity (mean "
        f"{mean[-COMMAND_WIDTH:].tolist()}, variance {var[-COMMAND_WIDTH:].tolist()}), and a {command_mode!r} stage "
        "would read live commands through them. Training holds the slice at mean 0 / variance 1 since decision "
        "D-D60, so these statistics were saved before it, or edited; their commands would not mean what they mean "
        "to every other stage's policy. Use a checkpoint saved under the identity rule."
    )


def pad_running_stats(obs_rms: Any, width: int = COMMAND_WIDTH) -> "RunningMeanStd":
    """A NEW running-statistics object with *width* identity dims appended.

    The widen tool's primitive: ``mean`` gains zeros, ``var`` gains ones,
    ``count`` is carried — the appended slice is born at identity, exactly as
    :func:`hold_command_identity` leaves it.  The padded object is built
    from the input's own class (``type(obs_rms)(shape=...)``) so this module
    never imports stable_baselines3 at module level.
    """
    mean, var = _running_stats(obs_rms)
    if width < 1:
        raise ValueError("width must be a positive number of dims")
    padded_shape = (mean.shape[0] + int(width),)
    padded: RunningMeanStd
    try:
        padded = type(obs_rms)(shape=padded_shape)
    except TypeError:
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        padded = RunningMeanStd(shape=padded_shape)
    padded.mean = np.concatenate([np.asarray(mean, dtype=np.float64), np.zeros(int(width), dtype=np.float64)])
    padded.var = np.concatenate([np.asarray(var, dtype=np.float64), np.ones(int(width), dtype=np.float64)])
    padded.count = obs_rms.count
    return padded
