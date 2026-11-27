"""Where each opt-in per-episode feature begins its episode, the streams it draws from, and who sees it.

Three pieces every opt-in feature of the shared env uses (design record
``docs/TRICERATOPS_ROBOT_PORT_2026_11.md`` §4); none is fingerprinted, and
with every feature off none of them reads a generator or writes anything:

* **The hook.** :class:`EpisodeStartHook` sits between ``gymnasium.Env`` and
  ``BaseDinoEnv`` in the MRO.  ``BaseDinoEnv.reset``'s own
  ``super().reset(seed=seed)`` lands in it: gymnasium seeds ``np_random``,
  then :meth:`EpisodeStartHook._begin_episode` runs, before the keyframe is
  loaded.  Before gymnasium seeds anything,
  :meth:`EpisodeStartHook._check_episode_may_begin` may refuse the reset
  (``BaseDinoEnv``'s refuses one inside a ``nominal_physics()`` block), so a
  refused reset reseeds nothing.  ``BaseDinoEnv.reset``'s source is
  unchanged (the policy layer hashes it as ``home_reset``), and nothing
  hashes class bases, so no digest moves.
* **The streams.** Each feature draws from its own PCG64 stream,
  :func:`feature_stream`, never from ``np_random``, so the reset draws (joint,
  velocity and height noise, the target, the push schedule, the command) are
  bit-identical with any feature on or off.  A feature rebuilds its stream on
  every seeded reset and on its first, and continues it across unseeded
  resets (SB3's auto-reset), as ``np_random`` is continued.  The stream keys
  are spawn keys of the episode's seed, so no stream coincides with
  gymnasium's ``SeedSequence(seed)``.
* **The role.** ``BaseDinoEnv.training_rollouts`` is ``False`` unless the
  trainer marks the env with :func:`mark_training_rollouts`: every env is an
  evaluation env unless a trainer says otherwise.  The trainers' rollout envs
  are marked in exactly two places, ``train_base.make_env`` (for
  ``create_vec_env(training=True)``) and ``train_behaviors.main`` (which unmarks
  its env again before scoring); ``test_episode_start.py`` pins them.

Numpy and gymnasium only, so the module imports no training backend.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

#: Spawn keys of the per-feature streams (:func:`feature_stream`), one per
#: feature and all distinct: the per-episode physics randomisation, the
#: actuation delay (``actuation_delay.py``) and the observation noise.
EPISODE_PHYSICS_STREAM_KEY = 0xD0A1
ACTUATION_DELAY_STREAM_KEY = 0xDE1A
OBSERVATION_NOISE_STREAM_KEY = 0x0B5E4015

#: Every stream key above, for the distinctness check.
FEATURE_STREAM_KEYS = (EPISODE_PHYSICS_STREAM_KEY, ACTUATION_DELAY_STREAM_KEY, OBSERVATION_NOISE_STREAM_KEY)


class EpisodeStartHook(gym.Env):
    """A ``gymnasium.Env`` whose reset calls :meth:`_begin_episode` right after gymnasium seeds ``np_random``."""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        self._check_episode_may_begin()  # before gymnasium reseeds anything
        result = super().reset(seed=seed, options=options)  # gymnasium seeds np_random
        self._begin_episode(seed)
        return result

    def _check_episode_may_begin(self) -> None:
        """Raise if no episode may begin now, before anything of the reset runs. Nothing by default."""

    def _begin_episode(self, seed: int | None) -> None:
        """Where each opt-in feature begins its episode. Nothing by default."""


def feature_stream(env: Any, seed: int | None, key: int) -> np.random.Generator:
    """A feature's own PCG64 stream for the episode a reset with *seed* begins, keyed by *key*.

    The entropy is the reset's *seed* when given; else the env's
    ``np_random_seed`` (gymnasium 1.0 and later); else, when ``np_random`` was
    assigned directly (``np_random_seed`` is -1) or the attribute is missing
    (gymnasium before 1.0, which ``pyproject.toml`` still allows), the state of
    ``np_random``'s PCG64 bit generator, read and never drawn.  Any other bit
    generator is refused: its state has no one integer form here.  The stream
    is ``SeedSequence(entropy, spawn_key=(key,))``, which no gymnasium seeding
    produces (gymnasium uses ``SeedSequence(seed)``, with no spawn key).
    """
    if seed is not None:
        entropy = int(seed)
    else:
        entropy = int(getattr(env, "np_random_seed", -1))  # gymnasium >= 1.0
        if entropy < 0:  # np_random assigned directly
            bit_generator = env.np_random.bit_generator
            if not isinstance(bit_generator, np.random.PCG64):
                raise ValueError(
                    "an opt-in episode feature without a reset seed needs a seeded reset or a PCG64 np_random; "
                    f"got {type(bit_generator).__name__}"
                )
            state = bit_generator.state["state"]  # read, never drawn
            entropy = (int(state["state"]) << 128) | int(state["inc"])
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(int(key),))))


def mark_training_rollouts(env: Any, enabled: bool) -> None:
    """Mark *env* as a trainer's rollout env (*enabled* true) or as an evaluation env (false).

    Calls ``set_training_rollouts`` on ``env.unwrapped`` (the env itself when
    it has no ``unwrapped``), so a wrapped env is marked at its base; an env
    without the method (a plain ``gymnasium.Env`` test double) is left as it
    is.  The mark takes effect at the env's next reset, for every feature
    alike.

    A vectorised env (one with ``num_envs`` or ``env_method``: an SB3
    ``VecEnv`` or ``VecNormalize``, or a gymnasium vector env) is refused with
    ``TypeError``: its ``unwrapped`` is the vector env itself, which has no
    ``set_training_rollouts``, so the rule above would leave every env it
    holds an evaluation env without a word.  Mark each env before it is
    vectorised, as ``train_base.make_env(..., training=True)`` does.
    """
    if hasattr(env, "num_envs") or hasattr(env, "env_method"):
        raise TypeError(
            f"mark_training_rollouts marks one env, not the vectorised {type(env).__name__}: mark each env before "
            "it is vectorised, as train_base.make_env(..., training=True) does for create_vec_env(training=True)"
        )
    base = getattr(env, "unwrapped", env)
    setter = getattr(base, "set_training_rollouts", None)
    if setter is not None:
        setter(enabled)
