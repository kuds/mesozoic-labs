"""The cruise ramp: a live stage's commanded speeds ramped in over training (decision D-D62).

The command-tracking reward is near zero far from the commanded speed, so a
walker warm-started from a stance (which stood still and is command-blind at
the start of the walk) would see almost no tracking signal at full cruise.
A stage whose ``[commands]`` declares ``cruise_ramp_start_fraction`` and
``cruise_ramp_timesteps`` therefore trains on the declared command config
with ``cruise_speed`` and ``speed_range`` scaled by a fraction that starts at
the start fraction and rises linearly to 1 over the ramp's timesteps
(``DirectionCommandConfig.ramped``).  The rules:

* **What moves.**  Only the two commanded speeds.  The stop probability is
  not scaled and the stop is drawn before the speed from the command stream,
  so the same seeds are stand episodes at every fraction and the stand
  fraction never changes along the ramp; each cruise episode is commanded the
  fraction of its end-state speed (to rounding), at the same heading offset.
  The tracking sigmas and tolerances are the end state's (their speed
  defaults follow ``speed_scale``, which the ramp keeps): the ramp works
  because a standing walker's speed error shrinks with the command while the
  Gaussian's width does not, so scaling them with the ramp would undo it.
* **Where.**  Every training worker, through ``env_method`` and
  ``BaseDinoEnv.set_command_config``, from before ``learn()``'s first reset
  (:meth:`CruiseRampCallback.prime`), so the first episodes are drawn at the
  session's fraction.  The evaluation env, the selection and certification
  panels, the policy-probe and K3 envs keep the stage's declared config: the
  handoff is chosen and certified on the end state, which is the task the
  fingerprint names, and an evaluation that followed the ramp would rank
  checkpoints on an easier task than the one they are chosen on.
* **The record.**  The ramp spec rides in the command config, so the task
  fingerprint's command section records it with the end state; the
  environment's ``command_manifest()`` reports the declared config whatever
  has been swapped in.  The ramp position rides in every checkpoint: the
  callback attaches :data:`MODEL_CRUISE_RAMP_ATTRIBUTE` to the model at the
  start of each session (the ramp steps before the session and the model's
  step counter there), which SB3 saves with every checkpoint, so a checkpoint
  at step counter ``n`` was saved at ramp step
  ``steps_before_session + n - session_start_num_timesteps``.
* **Resume.**  A ``resume_same_stage`` load continues the ramp from the
  checkpoint's own position (:func:`resume_ramp_steps`), for a PPO
  continuation, whose step counter carries on, and for a SAC or curated
  checkpoint, whose counter restarts, alike; a checkpoint without the record
  is refused.  The ramp is a function of ramp steps alone, held in blocks of
  :data:`CRUISE_RAMP_UPDATE_STEPS`, so a resumed session commands exactly
  what the uninterrupted one would have.  A stage load across an edge starts
  the ramp at 0, and a ramp is refused on a stage warm-started from a parent
  that already tracked a command (``train_base``): that warm start is an
  exact continuation, and the parent was trained at the full cruise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from environments.shared.direction_commands import DirectionCommandConfig

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)

#: The schema of the ramp position a checkpoint records.
CRUISE_RAMP_SCHEMA = "mesozoic.cruise-ramp/v1"
#: The model attribute SB3 persists the ramp position in, inside the checkpoint ZIP.
MODEL_CRUISE_RAMP_ATTRIBUTE = "mesozoic_cruise_ramp"
#: The ramp holds each fraction for this many ramp steps: each change is one
#: ``env_method`` round trip to every worker (RewardRampCallback's cadence).
CRUISE_RAMP_UPDATE_STEPS = 10_000


class CruiseRampError(ValueError):
    """A cruise ramp cannot run or resume as declared."""


def cruise_ramp_fraction(config: DirectionCommandConfig, ramp_steps: int) -> float:
    """The fraction of the end-state speeds commanded *ramp_steps* steps into *config*'s ramp.

    The start fraction, raised linearly to 1 over ``cruise_ramp_timesteps``
    and held over each block of :data:`CRUISE_RAMP_UPDATE_STEPS`: exactly 1.0
    once the held step reaches the budget, and always for a config without a
    ramp.
    """
    if not config.has_cruise_ramp:
        return 1.0
    held = (max(0, int(ramp_steps)) // CRUISE_RAMP_UPDATE_STEPS) * CRUISE_RAMP_UPDATE_STEPS
    if held >= config.cruise_ramp_timesteps:
        return 1.0
    start = config.cruise_ramp_start_fraction
    return start + (1.0 - start) * held / config.cruise_ramp_timesteps


def ramped_command_config(config: DirectionCommandConfig, ramp_steps: int) -> DirectionCommandConfig:
    """The command config the training workers draw from *ramp_steps* into the ramp: *config* itself at 1."""
    fraction = cruise_ramp_fraction(config, ramp_steps)
    return config if fraction == 1.0 else config.ramped(fraction)


def cruise_ramp_record(
    config: DirectionCommandConfig, *, steps_before_session: int, session_start_num_timesteps: int
) -> dict[str, Any]:
    """The ramp position a session attaches to its model (:data:`MODEL_CRUISE_RAMP_ATTRIBUTE`)."""
    return {
        "schema": CRUISE_RAMP_SCHEMA,
        "start_fraction": float(config.cruise_ramp_start_fraction),
        "timesteps": int(config.cruise_ramp_timesteps),
        "update_steps": CRUISE_RAMP_UPDATE_STEPS,
        "steps_before_session": int(steps_before_session),
        "session_start_num_timesteps": int(session_start_num_timesteps),
    }


def ramp_steps_at(record: Mapping[str, Any], num_timesteps: int) -> int:
    """The ramp step a model holding *record* had reached at step counter *num_timesteps*."""
    return int(record["steps_before_session"]) + max(0, int(num_timesteps) - int(record["session_start_num_timesteps"]))


def resume_ramp_steps(checkpoint: "str | Path", config: DirectionCommandConfig) -> int:
    """The ramp step a ``resume_same_stage`` load of *checkpoint* continues *config*'s ramp from.

    Read from the checkpoint ZIP without loading it: its
    :data:`MODEL_CRUISE_RAMP_ATTRIBUTE` record and its saved step counter.
    Refused (:class:`CruiseRampError`): a checkpoint that records no ramp
    position or no step counter, and one whose ramp is not *config*'s (its
    start fraction, budget or update cadence differ).
    """
    from environments.shared.task_fingerprint import read_checkpoint_attribute

    record = read_checkpoint_attribute(checkpoint, MODEL_CRUISE_RAMP_ATTRIBUTE)
    num_timesteps = read_checkpoint_attribute(checkpoint, "num_timesteps")
    if not isinstance(record, Mapping) or record.get("schema") != CRUISE_RAMP_SCHEMA:
        raise CruiseRampError(
            f"{checkpoint} records no cruise-ramp position ({MODEL_CRUISE_RAMP_ATTRIBUTE}), so a resume cannot "
            "restore the commanded speeds it trained at; resume from a checkpoint this stage's ramp saved"
        )
    if isinstance(num_timesteps, bool) or not isinstance(num_timesteps, int):
        raise CruiseRampError(f"{checkpoint} records no step counter (num_timesteps) to place its cruise ramp")
    expected = cruise_ramp_record(config, steps_before_session=0, session_start_num_timesteps=0)
    if any(record.get(key) != expected[key] for key in ("start_fraction", "timesteps", "update_steps")):
        raise CruiseRampError(
            f"{checkpoint} was trained under the cruise ramp {dict(record)!r}, not the stage's "
            f"(start fraction {config.cruise_ramp_start_fraction!r} over {config.cruise_ramp_timesteps} steps)"
        )
    return ramp_steps_at(record, num_timesteps)


class CruiseRampCallback(BaseCallback):  # type: ignore[misc]
    """Ramp a live stage's commanded speeds in over training (see the module docstring).

    *command_config* is the stage's declared config (its ramp's end state);
    *start_ramp_steps* the ramp step this session starts at (0 for a new
    stage, :func:`resume_ramp_steps` for a resume).  The ramp step at the
    model's step counter ``n`` is ``start_ramp_steps + n - n0``, ``n0`` the
    counter at the start of the session.  Call :meth:`prime` on the training
    env before ``learn()``, whose first reset draws the first commands.
    """

    def __init__(self, command_config: DirectionCommandConfig, *, start_ramp_steps: int = 0, verbose: int = 0):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for CruiseRampCallback.")
        if not isinstance(command_config, DirectionCommandConfig) or not command_config.has_cruise_ramp:
            raise CruiseRampError("CruiseRampCallback needs a command config that declares a cruise ramp")
        super().__init__(verbose)
        self.command_config = command_config
        self.start_ramp_steps = int(start_ramp_steps)
        self._session_start: int | None = None
        self._fraction: float | None = None

    def _apply(self, venv: Any, ramp_steps: int) -> None:
        """Swap the ramped config into every worker of *venv* when its fraction changed."""
        fraction = cruise_ramp_fraction(self.command_config, ramp_steps)
        if fraction == self._fraction:
            return
        venv.env_method("set_command_config", ramped_command_config(self.command_config, ramp_steps))
        self._fraction = fraction
        logger.info("CruiseRampCallback: ramp step %d, commanded speeds at %.4f of the end state", ramp_steps, fraction)

    def prime(self, venv: Any) -> None:
        """Set the session's first fraction on every worker of the training env *venv*, before ``learn()``."""
        self._apply(venv, self.start_ramp_steps)

    @property
    def fraction(self) -> float | None:
        """The fraction the workers were last given (``None`` before :meth:`prime`)."""
        return self._fraction

    def ramp_steps(self, num_timesteps: int) -> int:
        """The ramp step at the model's step counter *num_timesteps* in this session."""
        start = self._session_start if self._session_start is not None else int(num_timesteps)
        return self.start_ramp_steps + max(0, int(num_timesteps) - start)

    def _on_training_start(self) -> None:
        self._session_start = int(self.model.num_timesteps)
        setattr(
            self.model,
            MODEL_CRUISE_RAMP_ATTRIBUTE,
            cruise_ramp_record(
                self.command_config,
                steps_before_session=self.start_ramp_steps,
                session_start_num_timesteps=self._session_start,
            ),
        )
        self._apply(self.model.get_env(), self.start_ramp_steps)
        logger.info(
            "CruiseRampCallback: commanded speeds from %.4f to 1 of the end state over %d steps, from ramp step %d",
            self.command_config.cruise_ramp_start_fraction,
            self.command_config.cruise_ramp_timesteps,
            self.start_ramp_steps,
        )

    def _on_step(self) -> bool:
        self._apply(self.model.get_env(), self.ramp_steps(self.num_timesteps))
        return True
