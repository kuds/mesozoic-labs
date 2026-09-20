"""Learning-rate / coefficient schedules and the schedule-driving callback.

``_ConstantSchedule`` exists because SB3 expects a callable schedule wherever
it accepts a learning rate or clip range, so temporarily pinning one to a fixed
value means swapping in a callable rather than assigning a float.
``LinearSchedule`` / ``CosineSchedule`` are the decays ``train_base`` builds
from a stage's ``learning_rate_end`` / ``clip_range_end``; they are classes
so that a saved archive pickles them by reference instead of embedding
interpreter-specific bytecode (see :class:`LinearSchedule`)."""

from __future__ import annotations

import logging
from typing import Any, Mapping

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"


class LinearSchedule:
    """Linear decay from *initial* (``progress_remaining = 1``) to *final* (``0``).

    A module-level class rather than a closure on purpose. SB3 stores a
    model's ``learning_rate`` / ``lr_schedule`` / ``clip_range`` members in
    the archive through cloudpickle, which pickles a closure BY VALUE, code
    object and all, and that bytecode belongs to the interpreter that saved
    it: loading such an archive under another Python minor version
    executes foreign bytecode inside ``PPO.load`` and kills the process
    (KNOWN_ISSUES, "SB3 archives are bound to the interpreter that saved
    them"). An instance of an importable class pickles by reference, so an
    archive saved with it carries no bytecode at all.
    """

    def __init__(self, initial: float, final: float) -> None:
        self.initial = float(initial)
        self.final = float(final)

    def __call__(self, progress_remaining: float) -> float:
        return self.final + progress_remaining * (self.initial - self.final)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.initial!r}, {self.final!r})"


class CosineSchedule:
    """Cosine decay from *initial* (``progress_remaining = 1``) to *final* (``0``).

    Decays faster in mid-training than :class:`LinearSchedule`, then
    flattens near the end, which better protects converged policies from
    late-training destabilisation. A class for the same reason as
    :class:`LinearSchedule`.
    """

    def __init__(self, initial: float, final: float) -> None:
        self.initial = float(initial)
        self.final = float(final)

    def __call__(self, progress_remaining: float) -> float:
        import math

        # progress_remaining goes from 1.0 -> 0.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return self.final + cosine_decay * (self.initial - self.final)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.initial!r}, {self.final!r})"


def schedule_members_from_hyperparameters(
    algorithm: str, hyperparameters: "Mapping[str, Any] | None"
) -> dict[str, Any]:
    """The schedule members a stage's recorded algorithm block trains under, as picklable-by-reference objects.

    Mirrors the mapping ``train_base._prepare_alg_kwargs`` applies to a
    stage TOML's algorithm table. For PPO, ``learning_rate`` with
    ``learning_rate_end`` becomes a :class:`LinearSchedule` (or
    :class:`CosineSchedule` when ``lr_schedule = "cosine"``), ``clip_range``
    with ``clip_range_end`` a :class:`LinearSchedule`, and ``clip_range_vf``
    is passed through; the trainer applies no decay mapping to a ``[sac]``
    table, so neither does this (a SAC ``learning_rate`` stays a float). A
    value without an end stays a float, and an absent key is absent from the
    result. This is what the widen tool stamps into a widened archive in
    place of the parent's cloudpickled closures, read from the parent's
    ``stage_config.json`` ``"hyperparameters"`` block (the block
    ``save_stage_config`` records), or from the current stage config's block
    when the parent recorded none.
    """
    block: dict[str, Any] = dict(hyperparameters or {})
    members: dict[str, Any] = {}
    is_ppo = algorithm.lower() == "ppo"
    if block.get("learning_rate") is not None:
        lr_end = block.get("learning_rate_end") if is_ppo else None
        if lr_end is None:
            members["learning_rate"] = float(block["learning_rate"])
        elif block.get("lr_schedule", "linear") == "cosine":
            members["learning_rate"] = CosineSchedule(block["learning_rate"], lr_end)
        else:
            members["learning_rate"] = LinearSchedule(block["learning_rate"], lr_end)
    if is_ppo:
        if block.get("clip_range") is not None:
            clip_end = block.get("clip_range_end")
            members["clip_range"] = (
                float(block["clip_range"]) if clip_end is None else LinearSchedule(block["clip_range"], clip_end)
            )
        if "clip_range_vf" in block:
            members["clip_range_vf"] = block["clip_range_vf"]
    return members


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

    def _capture_initial(self, deferred: bool) -> float:
        initial = float(self.model.ent_coef)
        self._initial = initial
        logger.info(
            "EntCoefDecay: ent_coef %.4f → %.4f over %d timesteps%s",
            initial,
            self.end_value,
            self.decay_timesteps,
            " (captured after the stage warm-up released ent_coef)" if deferred else "",
        )
        return initial

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
        initial = self._initial
        if initial is None:
            initial = self._capture_initial(deferred=True)
        frac = min(1.0, self.num_timesteps / self.decay_timesteps)
        self.model.ent_coef = initial + frac * (self.end_value - initial)
        return True
