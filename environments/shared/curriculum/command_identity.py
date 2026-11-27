"""The command slice held at identity normalization while a stage trains (decision D-D60).

Every observation ends with the 3-wide command (``command_frame``), emitted
pre-scaled to [-1, 1].  Its ``VecNormalize`` statistics are held at mean 0 /
variance 1 in every stage, so a normalised command is the raw command and
means the same to the stance (where it is always zero), the walk and every
follow node, and a warm start between two live stages is an exact
continuation.  No ``VecNormalize`` subclass is pickled for it (decision D-D3
deleted the pilots' one): the sidecar stays a stock ``VecNormalize``, and the
slice is re-imposed in place.

Where the statistics move and where they are written, and how each is held:

* ``VecNormalize.reset`` updates them once inside ``learn()``'s setup, before
  any callback runs: :meth:`CommandIdentityCallback._on_training_start`
  re-imposes the slice on the training env (and on the evaluation env, which
  never updates but is held for any statistics loaded into it).
* ``VecNormalize.step_wait`` updates them on every vector step, before the
  observation is normalised and before ``callback.on_step``:
  :meth:`CommandIdentityCallback._on_step` re-imposes the slice after every
  update.  The observation of that one step is normalised with statistics
  one update away from identity; under ``"none"`` the command is exactly 0
  and normalises to exactly 0 either way, and under a live mode the
  difference is at most about ``n_envs / count`` (0.6 on a fresh normaliser's
  first step, below 1e-6 once a stance's count is carried).
* The terminal observation of an env that finished at that step is
  normalised by ``step_wait`` with the same updated statistics and read by
  the algorithm AFTER the callbacks: an off-policy algorithm (SAC) stores
  ``unnormalize_obs(terminal)`` in its replay buffer, and PPO bootstraps a
  truncated episode's value on it.  Un-normalising it under the held
  statistics would store a command one update away from the raw one, so
  :meth:`CommandIdentityCallback._on_step` re-expresses the command slice of
  each terminal observation under the held statistics as it holds them: the
  replay buffer stores the raw terminal command (to the float32 rounding of
  the round trip, as without the hold), and PPO's bootstrap reads it at
  identity.  Under ``"none"`` the slice is exactly 0 and stays 0.
* Every sidecar save inside ``learn()`` (the best-model pair the
  ``EvalCallback`` writes, ``RobustBestModelCallback``, SB3's periodic
  ``CheckpointCallback``) runs in a callback AFTER this one in the same
  ``on_step``: ``train_base`` puts it first in the list, and
  ``SaveVecNormalizeCallback`` and ``RobustBestModelCallback`` also hold the
  slice themselves before they save.  The ``EvalCallback`` copies the
  training statistics into the evaluation env at every evaluation
  (``sync_envs_normalization``), after this callback, so the evaluation env
  carries identity too.
* The final save after ``learn()`` (and after a ``KeyboardInterrupt``, which
  can land between an update and the next ``on_step``) holds the slice itself
  (``train_base._save_final_and_sync_tb``).
* ``CriticPhaseVecNormalize.save`` writes the training env's own statistics
  object, so it is held by its callers above.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from environments.shared.command_frame import COMMAND_WIDTH, command_slice, hold_command_identity

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)


def _vec_normalize(vec_env: Any) -> Any:
    """The ``VecNormalize`` *vec_env* is or wraps; ``None`` for ``None``, an unnormalised VecEnv, or no SB3."""
    if vec_env is None or not sb3_compat._SB3_AVAILABLE:
        return None
    from stable_baselines3.common.vec_env import unwrap_vec_normalize

    return unwrap_vec_normalize(vec_env)


def hold_vec_normalize_command_identity(vec_env: Any) -> bool:
    """Hold the command slice of *vec_env*'s ``VecNormalize`` statistics at identity; return whether one changed.

    *vec_env* is a ``VecNormalize`` or a VecEnv wrapping one; ``None`` or a
    VecEnv without one (an unnormalised env) is left alone and returns
    ``False``.
    """
    normalizer = _vec_normalize(vec_env)
    if normalizer is None:
        return False
    return hold_command_identity(normalizer.obs_rms)


def reexpress_terminal_commands(
    normalizer: Any,
    mean: np.ndarray,
    var: np.ndarray,
    dones: "Sequence[Any] | None",
    infos: "Sequence[dict[str, Any]] | None",
) -> int:
    """Re-normalise the command slice of each finished env's terminal observation under the held statistics.

    ``VecNormalize.step_wait`` normalised it with the command statistics
    *mean* / *var* (the slice as that step's update left it, before the hold).
    The raw command is recovered exactly as ``unnormalize_obs`` would recover
    it under them, then normalised as ``normalize_obs`` does under mean 0 /
    variance 1, so ``unnormalize_obs`` under the held statistics returns the
    raw command.  The command columns are the last ``COMMAND_WIDTH`` of the
    statistics' width (a critic-phase env appends its phase after them); every
    other column is left as it is.  Returns how many terminal observations were
    rewritten.
    """
    if dones is None or infos is None:
        return 0
    columns = command_slice(int(normalizer.obs_rms.mean.shape[0]))
    stored_scale = np.sqrt(np.asarray(var, dtype=np.float64) + normalizer.epsilon)
    identity_scale = np.sqrt(1.0 + normalizer.epsilon)
    rewritten = 0
    for done, info in zip(dones, infos):
        terminal = info.get("terminal_observation") if done else None
        if not isinstance(terminal, np.ndarray):
            continue
        command = terminal[columns] * stored_scale + mean
        held = terminal.copy()
        held[columns] = np.clip(command / identity_scale, -normalizer.clip_obs, normalizer.clip_obs)
        info["terminal_observation"] = held
        rewritten += 1
    return rewritten


class CommandIdentityCallback(BaseCallback):  # type: ignore[misc]
    """Re-impose the command slice's identity statistics at training start and after every step (decision D-D60).

    Place it FIRST in the callback list, so every callback after it in the
    same step, and every save they make, sees the slice at identity (the
    module docstring lists where the statistics move and are written).
    ``eval_env`` is the stage's evaluation env (``None``: none to hold).
    Every update moves the slice off identity, under ``"none"`` too: the
    zero command keeps its mean at exactly 0, but its variance decays as
    ``count / (count + n_envs)`` per step.  Each step also re-expresses the
    command slice of the step's terminal observations under the held
    statistics (:func:`reexpress_terminal_commands`), so SAC's replay buffer
    stores the raw terminal command.
    """

    def __init__(self, eval_env: Any = None, verbose: int = 0):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for CommandIdentityCallback.")
        super().__init__(verbose)
        self.eval_env = eval_env

    def _hold(self) -> None:
        hold_vec_normalize_command_identity(self.model.get_vec_normalize_env())

    def _on_training_start(self) -> None:
        self._hold()
        hold_vec_normalize_command_identity(self.eval_env)
        logger.info("Command slice held at identity normalization (mean 0, variance 1) for this stage (D-D60)")

    def _on_step(self) -> bool:
        normalizer = _vec_normalize(self.model.get_vec_normalize_env())
        if normalizer is None:
            return True
        mean = np.array(normalizer.obs_rms.mean[-COMMAND_WIDTH:], dtype=np.float64)
        var = np.array(normalizer.obs_rms.var[-COMMAND_WIDTH:], dtype=np.float64)
        if hold_command_identity(normalizer.obs_rms) and normalizer.norm_obs:
            # The terminal observations step_wait normalised with this step's update (module docstring).
            reexpress_terminal_commands(normalizer, mean, var, self.locals.get("dones"), self.locals.get("infos"))
        return True

    def _on_training_end(self) -> None:
        self._hold()
