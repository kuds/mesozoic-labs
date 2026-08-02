"""Checkpoint selection, publication, and VecNormalize statistics.

Keeping the normalisation statistics beside the policy they were trained with
is what makes a checkpoint replayable; :func:`load_vecnorm_stats` refuses to
load statistics whose plant identity does not match the current plant."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import sb3_compat
from .sb3_compat import BaseCallback

if TYPE_CHECKING:
    from environments.shared.plant_contract import PlantIdentity

logger = logging.getLogger(__name__)


class RobustBestModelCallback(BaseCallback):  # type: ignore[misc]
    """Save the checkpoint with the best risk-adjusted evaluation score.

    SB3's ``EvalCallback`` saves ``best_model.zip`` by mean eval reward.
    In run 20260709_185946 that selected the 800k-step checkpoint whose
    eval distribution was already bimodal — a high mean propped up by
    27/30 good episodes while the catastrophic-fall fraction was growing
    — when the genuinely robust checkpoint (~450k steps, zero failures)
    scored a slightly lower mean. This callback scores each evaluation as
    ``mean - risk_coef * std`` and saves ``robust_best_model.zip`` (plus
    its matching VecNormalize stats) whenever the score improves, so
    next-stage training and gating can prefer consistently-good policies
    over great-on-average ones with a fat failure tail.

    Place after the ``EvalCallback`` in the callback list.

    Args:
        eval_callback: The ``EvalCallback`` whose in-memory eval results
            to score.
        model_dir: Directory to save ``robust_best_model.zip`` into.
        risk_coef: Weight on the per-episode std in the score
            (default 1.0, i.e. one standard deviation below the mean).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        eval_callback: Any,
        model_dir: "str | Path",
        risk_coef: float = 1.0,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RobustBestModelCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.model_dir = Path(model_dir)
        self.risk_coef = risk_coef
        self.best_score = -np.inf
        self._last_seen_n = 0

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None) or []
        n = len(results)
        if n <= self._last_seen_n:
            return True
        self._last_seen_n = n

        latest = np.asarray(results[-1], dtype=float)
        score = float(latest.mean() - self.risk_coef * latest.std())
        if score <= self.best_score:
            return True
        self.best_score = score

        path = self.model_dir / "robust_best_model"
        self.model.save(str(path))
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(str(path) + "_vecnorm.pkl")
        logger.info(
            "RobustBestModel: new best risk-adjusted score %.1f (mean %.1f - %.1f*std %.1f) at step %d",
            score,
            latest.mean(),
            self.risk_coef,
            latest.std(),
            self.num_timesteps,
        )
        return True


class PublishEvalArtifactsCallback(BaseCallback):  # type: ignore[misc]
    """Atomically publish EvalCallback's ``evaluations.npz`` to the stage dir.

    The paired ``EvalCallback`` writes its npz to fast local scratch;
    this callback copies it to the (possibly Drive/GCS-FUSE mounted)
    stage directory after every new evaluation and again at training
    end, via copy-to-temp + rename, so the published file is never
    observed truncated. Place it *after* the ``EvalCallback`` in the
    callback list so it runs on the same step a new eval completes.

    Args:
        eval_callback: The ``EvalCallback`` whose npz to publish.
        publish_dir: Directory to publish ``evaluations.npz`` into
            (typically the stage directory).
        verbose: Verbosity level.
    """

    def __init__(self, eval_callback: Any, publish_dir: "str | Path", verbose: int = 0):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PublishEvalArtifactsCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.publish_dir = Path(publish_dir)
        self._last_published_n = 0

    def _source_npz(self) -> "Path | None":
        # SB3 stores log_path as the "<dir>/evaluations" file prefix.
        prefix = getattr(self.eval_callback, "log_path", None)
        return Path(str(prefix) + ".npz") if prefix else None

    def _publish(self) -> None:
        from ..file_io import atomic_copy

        src = self._source_npz()
        if src is not None and src.exists():
            atomic_copy(src, self.publish_dir / "evaluations.npz")

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None) or []
        if len(results) > self._last_published_n:
            self._last_published_n = len(results)
            self._publish()
        return True

    def _on_training_end(self) -> None:
        self._publish()


class SaveVecNormalizeCallback(BaseCallback):  # type: ignore[misc]
    """Save VecNormalize stats whenever triggered.

    Intended for use as ``callback_on_new_best`` in SB3's ``EvalCallback``
    so that the VecNormalize wrapper is saved alongside ``best_model.zip``.
    This ensures the observation normalization statistics match the policy
    weights when the best model is loaded for evaluation or next-stage
    curriculum training.

    Example::

        save_vecnorm_cb = SaveVecNormalizeCallback(
            save_path=str(model_dir / "best_model_vecnorm.pkl"),
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=save_vecnorm_cb,
            ...
        )

    Args:
        save_path: Destination path for the VecNormalize ``.pkl`` file.
        verbose: Verbosity level.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for SaveVecNormalizeCallback.")
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(self.save_path)
            logger.info("VecNormalize saved to: %s", self.save_path)
        return True


def load_vecnorm_stats(
    vecnorm_path: str,
    train_env,
    eval_env=None,
    *,
    current_plant: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    unsafe_skip_plant_validation: bool = False,
) -> bool:
    """Load VecNormalize running statistics from a previous stage into new envs.

    Only observation normalization (``obs_rms``) is carried forward.  Return
    normalization (``ret_rms``) is deliberately **reset** because the reward
    distribution changes between curriculum stages (new reward components,
    different weight magnitudes).  Carrying stale ``ret_rms`` produces badly
    scaled normalized rewards that destabilise policy gradients during the
    critical first updates of a new stage.

    Args:
        vecnorm_path: Path to a ``_vecnorm.pkl`` file saved by a previous stage.
        train_env: The new stage's training ``VecNormalize`` wrapper.
            Its existing ``training`` / ``norm_reward`` flags are preserved
            (set by ``create_vec_env`` based on the algorithm — SAC keeps
            ``norm_reward=False`` to avoid replay-buffer reward-scale drift).
        eval_env: Optional evaluation ``VecNormalize`` wrapper.
            ``training`` is set to ``False``; ``norm_reward`` is disabled.
        current_plant: Expected plant identity. Required unless the explicit
            unsafe inspection-only escape hatch is enabled.
        allow_legacy_plant: Explicitly allow a pre-contract normalization
            artifact with no identity.  Incompatible tagged artifacts are
            always rejected.
        unsafe_skip_plant_validation: Deliberately load without checking the
            plant. This is for low-level inspection/tests only and cannot be
            combined with either validation option.

    Returns:
        ``True`` if stats were loaded, ``False`` if the file was not found.
    """
    from pathlib import Path as _Path

    from environments.shared.plant_contract import PlantCompatibilityError

    if unsafe_skip_plant_validation:
        if current_plant is not None or allow_legacy_plant:
            raise ValueError("unsafe_skip_plant_validation cannot be combined with current_plant or allow_legacy_plant")
        logger.warning(
            "UNSAFE: loading %s without plant compatibility validation; "
            "do not use these statistics for training or evaluation until verified",
            vecnorm_path,
        )
    elif current_plant is None:
        raise PlantCompatibilityError(
            f"refusing to load {vecnorm_path} without current_plant; pass the current PlantIdentity "
            "or use unsafe_skip_plant_validation=True only for deliberate low-level inspection"
        )

    if not sb3_compat._SB3_AVAILABLE:
        logger.warning("stable-baselines3 not available; skipping VecNormalize load.")
        return False

    from stable_baselines3.common.vec_env import VecNormalize

    path = _Path(vecnorm_path)
    if not path.exists():
        logger.debug("VecNormalize file not found: %s", vecnorm_path)
        return False

    logger.info("Loading VecNormalize stats from: %s", vecnorm_path)
    prev_norm = VecNormalize.load(str(path), train_env.venv)
    if not unsafe_skip_plant_validation:
        from environments.shared.plant_contract import validate_model_plant

        assert current_plant is not None
        # Validate before copying obs_rms so an incompatible artifact cannot
        # partially mutate either destination environment.
        validate_model_plant(
            prev_norm,
            current_plant,
            artifact=str(path),
            allow_legacy=allow_legacy_plant,
        )

    # Carry forward observation statistics — the observation space is identical
    # across stages, so the running mean/var remain valid.  ret_rms is
    # intentionally NOT copied: reward distribution changes between stages, so
    # stale return statistics would produce incorrectly scaled normalised
    # rewards for PPO.  train_env.training / norm_reward are left as configured
    # by create_vec_env (algorithm-aware).
    train_env.obs_rms = prev_norm.obs_rms
    logger.info("obs_rms carried forward; ret_rms reset (reward distribution changed)")

    if eval_env is not None:
        eval_env.obs_rms = prev_norm.obs_rms.copy()
        eval_env.training = False
        eval_env.norm_reward = False

    return True


#: Periodic checkpoints kept by default.  Matches the JAX backend's
#: ``JaxTrainerConfig.max_checkpoints``, so a stage keeps the same amount of
#: rollback history whichever backend produced it.
DEFAULT_MAX_CHECKPOINTS = 5

#: The three artifact families SB3's ``CheckpointCallback`` emits, as
#: ``(infix, extension)``.  Its filenames are
#: ``f"{name_prefix}_{infix}{num_timesteps}_steps.{ext}"``
#: (``stable_baselines3.common.callbacks.CheckpointCallback._checkpoint_path``).
_CHECKPOINT_KINDS: tuple[tuple[str, str], ...] = (
    ("", "zip"),
    ("vecnormalize_", "pkl"),
    ("replay_buffer_", "pkl"),
)


def prune_periodic_checkpoints(
    model_dir: "str | Path",
    name_prefix: str,
    max_checkpoints: int,
) -> list[Path]:
    """Delete all but the newest *max_checkpoints* periodic checkpoints.

    Matches only SB3's periodic ``{prefix}_{steps}_steps.*`` files and their
    ``vecnormalize_`` / ``replay_buffer_`` siblings.  ``best_model``,
    ``robust_best_model`` and ``stage<N>_final`` do not fit that pattern and
    are therefore unreachable from here by construction rather than by an
    exclusion list that could fall out of date.

    Ordering is by the **step count parsed from the filename**, not by mtime:
    on a Drive or GCS-FUSE mount the modification times reflect when the
    bytes finished uploading, which is not the order they were produced in.

    A checkpoint whose model ``.zip`` is pruned takes its matched
    ``vecnormalize_`` statistics with it — keeping orphaned statistics would
    leave a ``.pkl`` that no longer belongs to any policy in the directory.

    Returns the paths actually deleted, oldest step first.  A file that
    cannot be unlinked is logged and skipped rather than aborting the rest:
    these are archival copies, and one locked file must not leave the
    directory half-pruned.  ``max_checkpoints <= 0`` disables pruning
    entirely and returns ``[]``.
    """
    directory = Path(model_dir)
    if max_checkpoints <= 0 or not directory.is_dir():
        return []

    # Group every artifact by the step count in its name, so a step is only
    # ever dropped as a complete set.
    by_step: dict[int, list[Path]] = {}
    for infix, extension in _CHECKPOINT_KINDS:
        pattern = f"{name_prefix}_{infix}*_steps.{extension}"
        for path in directory.glob(pattern):
            remainder = path.name[len(f"{name_prefix}_{infix}") : -len(f"_steps.{extension}")]
            if not remainder.isdigit():
                # The remainder between the prefix and `_steps.<ext>` must be
                # the step count and nothing else. Anything unparsed is not a
                # file this function put here, so it is left alone rather than
                # deleted on a guess.
                continue
            by_step.setdefault(int(remainder), []).append(path)

    doomed_steps = sorted(sorted(by_step, reverse=True)[max_checkpoints:])
    removed: list[Path] = []
    for step in doomed_steps:
        for path in by_step[step]:
            try:
                path.unlink()
            except OSError:
                logger.warning("Could not prune checkpoint %s", path, exc_info=True)
                continue
            removed.append(path)
    return removed


class CheckpointRetentionCallback(BaseCallback):  # type: ignore[misc]
    """Keep only the newest *max_checkpoints* periodic checkpoints.

    SB3's ``CheckpointCallback`` never deletes anything.  At the shipped
    ``save_freq = 500_000`` a 6M-step stage 1 leaves **twelve** 4.02 MB
    policy zips plus their VecNormalize sidecars — measured on run
    ``20260801_021545``, where ``models/`` totals 60 MB of which 48 MB is
    periodic history, and a three-stage run is ~200 MB. Those files live on
    a Drive mount and nothing in this repository reads them: they exist for
    manual rollback, which needs recent history, not all of it.

    Place this **after** the ``CheckpointCallback`` in the callback list so
    it prunes on the same step a new checkpoint lands.  It never touches
    ``best_model``, ``robust_best_model`` or ``stage<N>_final`` — see
    :func:`prune_periodic_checkpoints` for why that is structural rather
    than an exclusion list.

    Args:
        model_dir: Directory the paired ``CheckpointCallback`` saves into.
        name_prefix: Its ``name_prefix`` (``f"stage{N}"``).
        save_freq: The paired callback's ``save_freq``, already divided by
            ``n_envs``. Pruning fires on the same ``n_calls`` cadence, which
            matters: ``_on_step`` runs on *every* training step, and globbing
            a Drive mount six million times a stage would cost far more than
            the storage it reclaims.
        max_checkpoints: How many step-points to keep. ``0`` disables
            pruning, which is how a stage opts out of retention entirely.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        model_dir: "str | Path",
        name_prefix: str,
        save_freq: int,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for CheckpointRetentionCallback.")
        if save_freq < 1:
            raise ValueError("save_freq must be at least 1")
        super().__init__(verbose)
        self.model_dir = Path(model_dir)
        self.name_prefix = name_prefix
        self.save_freq = int(save_freq)
        self.max_checkpoints = int(max_checkpoints)

    def _prune(self) -> None:
        removed = prune_periodic_checkpoints(self.model_dir, self.name_prefix, self.max_checkpoints)
        if removed:
            logger.info(
                "Pruned %d periodic checkpoint file(s), keeping the newest %d step-points: %s",
                len(removed),
                self.max_checkpoints,
                ", ".join(sorted(path.name for path in removed)),
            )

    def _on_step(self) -> bool:
        # Same trigger CheckpointCallback uses, so pruning happens on exactly
        # the steps a new checkpoint was written and never in between.
        if self.n_calls % self.save_freq == 0:
            self._prune()
        return True

    def _on_training_end(self) -> None:
        # A stage can stop between save points (EvalCollapseEarlyStopCallback,
        # or a budget that is not a multiple of save_freq), leaving the last
        # prune stale. Cheap once, at the end.
        self._prune()
