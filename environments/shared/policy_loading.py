"""Load an SB3 checkpoint together with the VecNormalize statistics it trained under.

The one loading path the offline report scripts and the stance gate report
share.  Four copies of this block had drifted apart -- in how they found the
sidecar, in what they did when it was missing, and in whether they checked
the plant -- and every one of those differences changes *which policy* gets
scored: a policy evaluated on raw observations is a different policy, and
the report would blame the joints, the actions or the gate for a loading
mistake.  Keeping the block in one place is what makes the four reports
comparable.

Not the trainer's :func:`~environments.shared.train_base.load_vecnorm_stats`,
which loads statistics INTO a live training wrapper, and not the
Monitor-wrapped evaluator in :mod:`~environments.shared.evaluation`: this is
the read-only, normalise-then-predict loader an offline rollout wants.

SB3 is imported inside the functions, per the repository's lazy-SB3
convention, so the module stays importable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

UNNORMALIZED_BANNER = "UNNORMALIZED EVAL — results are not comparable to training-time metrics"


class PolicyLoadError(RuntimeError):
    """The checkpoint's normalisation statistics could not be resolved or read.

    A plain ``Exception`` subclass rather than ``SystemExit`` on purpose: the
    stance gate report runs inside the training pipeline's artifact guard,
    which catches ``Exception`` so a diagnostic cannot sink a finished run,
    and ``SystemExit`` (a ``BaseException``) would sail straight through it.
    Each CLI converts this to ``SystemExit`` at its own boundary, which is
    where an exit status belongs.
    """


def resolve_vecnorm_path(model_path: str, vecnorm_arg: str | None, allow_unnormalized: bool) -> str | None:
    """The VecNormalize sidecar to evaluate with, or ``None`` for a deliberately unnormalised run.

    An explicit *vecnorm_arg* (the CLI's ``--vecnorm``) wins.  Otherwise the
    trainer's own resolver probes both sidecar conventions -- the
    ``<stem>_vecnorm.pkl`` guess the report scripts used to make can never
    match SB3's periodic ``<prefix>_vecnormalize_<steps>_steps.pkl``, so
    every periodic checkpoint was silently scored on raw observations.  No
    sidecar is fatal unless *allow_unnormalized*: a policy evaluated
    unnormalised is a different policy, and the report would blame the
    policy for a loading mistake.
    """
    if vecnorm_arg is not None:
        return vecnorm_arg
    from environments.shared.train_base import _resolve_vecnorm_sidecar

    candidate = _resolve_vecnorm_sidecar(model_path)
    if Path(candidate).exists():
        return candidate
    if allow_unnormalized:
        return None
    raise PolicyLoadError(
        f"no VecNormalize sidecar found for {model_path} (probed {candidate}). A policy evaluated on "
        "unnormalised observations is a different policy; pass --vecnorm, or --allow-unnormalized "
        "to proceed deliberately."
    )


def load_sb3_checkpoint(
    model_path: str,
    vecnorm_path: str | None,
    env_factory: Callable[[], Any],
    *,
    guess_sidecar: bool = True,
    allow_unnormalized: bool = False,
    plant_identity: Any = None,
    allow_legacy_plant: bool = False,
) -> tuple[Any, Any, str | None]:
    """Return ``(model, normalizer, resolved_vecnorm_path)`` for a saved SB3 checkpoint.

    ``PPO.load`` on the CPU, then the statistics through ``VecNormalize.load``
    over a throwaway ``DummyVecEnv`` built from *env_factory* -- the loader
    needs a live env to rebuild the wrapper, and hand-unpickling reconstructs
    a partial object whose ``__setstate__`` expectations drift with the SB3
    version.  The wrapper comes back frozen (``training = False``,
    ``norm_reward = False``) so a rollout can call ``normalize_obs`` without
    moving the statistics.

    A *vecnorm_path* of ``None`` means: when *guess_sidecar*, probe the
    checkpoint's sidecar under both naming conventions; when that finds
    nothing, or when guessing is off, run unnormalised only if
    *allow_unnormalized*.  Fail-closed by default because the failure is
    silent -- a policy scored on raw observations is a different policy.
    ``normalizer`` and the resolved path are both ``None`` for an
    unnormalised run, and the caller prints :data:`UNNORMALIZED_BANNER`.

    With *plant_identity*, both artifacts are validated against it in the
    order they load -- the model right after ``PPO.load``, the statistics
    right after ``VecNormalize.load`` -- so a checkpoint from another plant
    is refused before anything is scored; *allow_legacy_plant* admits one
    that predates the contract.  An unreadable sidecar raises
    :class:`PolicyLoadError` naming the file; plant refusals propagate as
    ``PlantCompatibilityError``.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(model_path, device="cpu")
    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(model, plant_identity, artifact=model_path, allow_legacy=allow_legacy_plant)

    if vecnorm_path is None:
        if guess_sidecar:
            vecnorm_path = resolve_vecnorm_path(model_path, None, allow_unnormalized)
        elif not allow_unnormalized:
            raise PolicyLoadError(
                f"no VecNormalize statistics given for {model_path}. A policy evaluated on unnormalised "
                "observations is a different policy; name the checkpoint's sidecar, or allow an "
                "unnormalised run deliberately."
            )
    if vecnorm_path is None:
        return model, None, None

    try:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([env_factory]))
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        # A truncated or text-mode-copied .pkl is the usual cause, and the raw
        # UnpicklingError/KeyError gives no hint that the file rather than the
        # code is at fault.
        raise PolicyLoadError(
            f"cannot read VecNormalize statistics from {vecnorm_path}: "
            f"{type(exc).__name__}: {exc}. Re-copy the file in binary mode. "
            "Running without --vecnorm would evaluate the policy on unnormalised "
            "observations — a different policy — so this is fatal, not a warning."
        ) from exc
    normalizer.training = False
    normalizer.norm_reward = False

    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(normalizer, plant_identity, artifact=vecnorm_path, allow_legacy=allow_legacy_plant)

    return model, normalizer, vecnorm_path
