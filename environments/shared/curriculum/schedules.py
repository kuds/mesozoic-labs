"""Learning-rate / coefficient schedules and the schedule-driving callback.

``_ConstantSchedule`` exists because SB3 expects a callable schedule wherever
it accepts a learning rate or clip range, so temporarily pinning one to a fixed
value means swapping in a callable rather than assigning a float."""

from __future__ import annotations

import logging

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)


class _ConstantSchedule:
    """Picklable callable that returns a constant value.

    Replaces inline lambdas (e.g. ``lambda _: 0.02``) which capture the
    notebook cell's ``__globals__`` and fail to pickle in Colab/Jupyter
    because of ``zmq.Context`` objects in that namespace.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, _progress: float) -> float:
        return self.value


class EntCoefDecayCallback(BaseCallback):  # type: ignore[misc]
    """Linearly decay PPO's entropy coefficient during a stage.

    In run 20260709_185946 the policy's action std grew 1.18 → 1.49 under
    a constant ``ent_coef`` while the eval failure fraction climbed: the
    entropy bonus keeps paying for exploration noise long after the gait
    needs consolidation. This callback decays ``model.ent_coef`` from its
    initial value to ``end_value`` over ``decay_timesteps``, then holds.

    PPO only — SAC's auto-tuned entropy already adapts on its own.

    Args:
        end_value: Final entropy coefficient.
        decay_timesteps: Timesteps over which to decay (typically the
            stage budget).
        verbose: Verbosity level.
    """

    def __init__(self, end_value: float = 0.0005, decay_timesteps: int = 4_000_000, verbose: int = 0):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for EntCoefDecayCallback.")
        super().__init__(verbose)
        self.end_value = end_value
        self.decay_timesteps = max(1, decay_timesteps)
        self._initial: float | None = None

    def _on_training_start(self) -> None:
        self._initial = float(self.model.ent_coef)
        logger.info(
            "EntCoefDecay: ent_coef %.4f → %.4f over %d timesteps",
            self._initial,
            self.end_value,
            self.decay_timesteps,
        )

    def _on_step(self) -> bool:
        if self._initial is None:
            return True
        frac = min(1.0, self.num_timesteps / self.decay_timesteps)
        self.model.ent_coef = self._initial + frac * (self.end_value - self._initial)
        return True
