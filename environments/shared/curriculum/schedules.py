"""Learning-rate / coefficient schedules and the schedule-driving callback.

``_ConstantSchedule`` exists because SB3 expects a callable schedule wherever
it accepts a learning rate or clip range, so temporarily pinning one to a fixed
value means swapping in a callable rather than assigning a float."""

from __future__ import annotations

import logging

from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)

#: Attribute set on the *model* by ``StageWarmupCallback`` while its PPO
#: warm-up owns ``ent_coef``, and read by ``EntCoefDecayCallback`` to stand
#: aside. On the model rather than between the callbacks because the two are
#: constructed independently in three launch paths (``train``,
#: ``train_curriculum``, and the notebook) and in no guaranteed order — the
#: model is the one object both are guaranteed to share. Cleared when the
#: warm-up restores ``ent_coef``. A checkpoint saved inside a warm-up window
#: pickles the flag as ``True``; every stage>1 ``--load`` path re-adds the
#: warm-up callback, which re-stamps it, so a stale flag only matters for a
#: run that loads a mid-warm-up checkpoint *without* a warm-up — and the
#: decay callback logs whenever it defers, so that state is visible.
ENT_COEF_WARMUP_MARKER = "_ent_coef_warmup_active"


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

    Defers to ``StageWarmupCallback`` whenever :data:`ENT_COEF_WARMUP_MARKER`
    is set on the model. Without that guard the per-step assignment here
    silently overwrote the configured stage-entry entropy boost on the first
    step after the warm-up set it — every PPO stage-2/3 transition logged
    "warm-up active … ent_coef=0.020" and then trained at the decaying base
    value, with the warm-up's restore clobbered the same way. The base value
    for the decay is captured on the first step after the warm-up releases
    ``ent_coef``, so the schedule continues exactly as configured from there.

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
        self._started = False

    def _capture_initial(self, deferred: bool) -> None:
        self._initial = float(self.model.ent_coef)
        logger.info(
            "EntCoefDecay: ent_coef %.4f → %.4f over %d timesteps%s",
            self._initial,
            self.end_value,
            self.decay_timesteps,
            " (captured after the stage warm-up released ent_coef)" if deferred else "",
        )

    def _on_training_start(self) -> None:
        self._started = True
        if getattr(self.model, ENT_COEF_WARMUP_MARKER, False):
            # StageWarmupCallback owns ent_coef; the value on the model right
            # now is (or is about to be) the warm-up boost, not the base this
            # schedule should decay from.
            logger.info("EntCoefDecay: deferring to the stage warm-up before capturing the base ent_coef")
            return
        self._capture_initial(deferred=False)

    def _on_step(self) -> bool:
        if not self._started:
            return True
        if getattr(self.model, ENT_COEF_WARMUP_MARKER, False):
            return True
        if self._initial is None:
            self._capture_initial(deferred=True)
        frac = min(1.0, self.num_timesteps / self.decay_timesteps)
        self.model.ent_coef = self._initial + frac * (self.end_value - self._initial)
        return True
