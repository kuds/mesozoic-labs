"""
Shared training infrastructure for all dinosaur species.

Provides the common training, curriculum, and evaluation logic used by every
species.  Species-specific entry points (``environments/<species>/scripts/train_sb3.py``)
define a :class:`SpeciesConfig` and delegate to :func:`main`.

The original monolithic module has been split into focused submodules for
maintainability:

- :mod:`~environments.shared.diagnostics` -- ``DiagnosticsCallback``
- :mod:`~environments.shared.eval_diagnostics` -- stage-aware SB3 evaluation
  and plateau diagnostics
- :mod:`~environments.shared.evaluation` -- ``eval_policy``, ``evaluate``,
  ``record_stage_video``
- :mod:`~environments.shared.cli` -- ``main``, ``_apply_overrides``,
  ``_cast_value``
- :mod:`~environments.shared.tb_sync` -- ``_is_gcs_path``,
  ``_make_local_tb_dir``, ``_sync_tb_to_gcs``

All public names are re-exported here so existing ``from
environments.shared.train_base import ...`` statements continue to work.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import DEFAULT_CLIP_OBS, DEFAULT_CLIP_REWARD, DEFAULT_NORM_OBS, DEFAULT_NORM_REWARD
from .plant_contract import (
    PlantIdentity,
    attach_plant_identity,
    current_plant_identity,
    validate_environment_plant,
    validate_model_plant,
)
from .stage_manifest import stage_label
from .tb_sync import _is_gcs_path, _make_local_tb_dir, _sync_tb_to_gcs  # noqa: F401  (re-exported for backward compat)

logger = logging.getLogger(__name__)

# Suppress noisy tensorboardX NaN/Inf warnings (handled by _sanitize in diagnostics.py)
logging.getLogger("tensorboardX").setLevel(logging.ERROR)


def _ensure_sb3():
    """Import SB3 or exit with a helpful error."""
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecNormalize,
        )

        return {
            "PPO": PPO,
            "SAC": SAC,
            "CallbackList": CallbackList,
            "CheckpointCallback": CheckpointCallback,
            "EvalCallback": EvalCallback,
            "Monitor": Monitor,
            "set_random_seed": set_random_seed,
            "DummyVecEnv": DummyVecEnv,
            "SubprocVecEnv": SubprocVecEnv,
            "VecNormalize": VecNormalize,
        }
    except ImportError:
        logger.error("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")
        sys.exit(1)


@dataclasses.dataclass
class SpeciesConfig:
    """Species-specific parameters that differ across training scripts.

    Each species defines one of these in its thin ``train_sb3.py`` wrapper.
    """

    species: str
    """Species identifier used for directory names and config lookup
    (e.g. ``"velociraptor"``, ``"trex"``, ``"brachiosaurus"``)."""

    env_class: type
    """The Gymnasium environment class (e.g. ``RaptorEnv``)."""

    stage_descriptions: str
    """Short stage legend for ``--stage`` argparse help
    (e.g. ``"1=balance, 2=locomotion, 3=strike"``)."""

    height_label: str
    """Label used in evaluation log output (``"Pelvis height"`` or ``"Torso height"``)."""

    stage3_section_label: str
    """Section header for stage-3 eval results (``"Hunting"`` or ``"Food Reaching"``)."""

    success_keys: list
    """Info-dict keys that signal a successful episode
    (e.g. ``["strike_success", "bite_success"]``)."""


# ── Utility helpers ──────────────────────────────────────────────────────


def linear_schedule(initial_lr: float, final_lr: float):
    """Return a callable that linearly decays learning rate."""

    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)

    return schedule


def cosine_schedule(initial_lr: float, final_lr: float):
    """Return a callable that decays learning rate on a cosine curve.

    Decays faster in mid-training than linear, then flattens near the end.
    This better protects converged policies from late-training destabilisation.
    """
    import math

    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 → 0.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return final_lr + cosine_decay * (initial_lr - final_lr)

    return schedule


# TensorBoard local-buffer helpers live in ``tb_sync.py`` (re-exported above).


# ── Environment creation ─────────────────────────────────────────────────


#: Evaluation panel size when a stage's gate does not demand a specific one.
_DEFAULT_EVAL_EPISODES = 30


def _eval_episodes_for_stage(stage_config: dict[str, Any]) -> int:
    """Episodes per evaluation, never fewer than the stage's gate requires.

    A gate that demands ``min_eval_episodes = 40`` against a 30-episode panel
    can never pass — the panel-size criterion fails at every evaluation and
    the stage runs to its timestep budget looking like a training failure.
    Taking the maximum couples the two deliberately: the evaluation is sized
    to the claim the gate intends to certify.

    It matters for ``stance_quality/v1``, whose bound and whose 28%-rejection
    analysis are both specified at n=40; at n=30 neither describes the gate
    being run.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    return max(_DEFAULT_EVAL_EPISODES, int(curriculum.get("min_eval_episodes", 0)))


def make_env(
    species_cfg: SpeciesConfig,
    stage_configs: "dict[int | str, dict[str, Any]]",
    stage: "int | str",
    rank: int,
    seed: int = 0,
    plant_identity: PlantIdentity | None = None,
):
    """Create a single environment instance."""
    sb3 = _ensure_sb3()

    def _init():
        sb3["set_random_seed"](seed + rank)
        env_kwargs = stage_configs[stage]["env_kwargs"].copy()
        env = species_cfg.env_class(**env_kwargs)
        if plant_identity is not None:
            validate_environment_plant(
                env,
                plant_identity,
                artifact=f"{species_cfg.species} stage {stage} environment",
            )
        env = sb3["Monitor"](env)
        env.reset(seed=seed + rank)
        return env

    return _init


def create_vec_env(
    species_cfg: SpeciesConfig,
    stage_configs: "dict[int | str, dict[str, Any]]",
    stage: "int | str",
    n_envs: int,
    seed: int = 0,
    use_subproc: bool = False,
    *,
    algorithm: str | None = None,
    gamma: float | None = None,
    plant_identity: PlantIdentity | None = None,
):
    """Create vectorized environment with observation/reward normalization.

    ``norm_reward`` is forced off for off-policy SAC: its replay buffer stores
    rewards at write-time, but VecNormalize's running return statistics keep
    drifting during training, so old buffer samples end up with an
    inconsistent reward scale relative to new samples — which also destabilises
    SAC's auto entropy coefficient. PPO (on-policy) is unaffected and keeps
    reward normalization on.

    ``gamma`` is forwarded to ``VecNormalize`` so its discounted-return
    statistics match the algorithm's discount factor; otherwise it silently
    drifts from SB3's hard-coded default of 0.99.
    """
    sb3 = _ensure_sb3()

    env_fns = [make_env(species_cfg, stage_configs, stage, i, seed, plant_identity) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        env = sb3["SubprocVecEnv"](env_fns)
    else:
        env = sb3["DummyVecEnv"](env_fns)

    norm_reward = DEFAULT_NORM_REWARD
    if algorithm is not None and algorithm.lower() == "sac":
        norm_reward = False

    vecnorm_kwargs: dict[str, Any] = dict(
        norm_obs=DEFAULT_NORM_OBS,
        norm_reward=norm_reward,
        clip_obs=DEFAULT_CLIP_OBS,
        clip_reward=DEFAULT_CLIP_REWARD,
    )
    if gamma is not None:
        vecnorm_kwargs["gamma"] = gamma

    env = sb3["VecNormalize"](env, **vecnorm_kwargs)
    if plant_identity is not None:
        # VecNormalize is pickled separately from the SB3 model.  Embedding the
        # same contract in both artifacts prevents a policy from being paired
        # with normalization statistics from a different physics plant.
        attach_plant_identity(env, plant_identity)
    return env


# ── Shared setup helpers (used by both train() and train_curriculum()) ──


def _prepare_alg_kwargs(
    config: dict[str, Any],
    algorithm: str,
    verbose: int,
    log_path: Path,
    use_tensorboard: bool,
) -> tuple[dict[str, Any], Path | None, Path]:
    """Build algorithm kwargs with LR schedule, clip annealing, and TB setup.

    Returns ``(alg_kwargs, local_tb_dir, gcs_tb_path)`` where *local_tb_dir*
    is ``None`` when the output is not on a GCS FUSE mount.
    """
    alg_kwargs = config["sac_kwargs"].copy() if algorithm == "sac" else config["ppo_kwargs"].copy()
    alg_kwargs["verbose"] = verbose

    # TensorBoard buffering
    local_tb_dir = None
    gcs_tb_path = log_path / "tensorboard"
    if use_tensorboard:
        if _is_gcs_path(log_path):
            local_tb_dir = _make_local_tb_dir(gcs_tb_path)
            alg_kwargs["tensorboard_log"] = str(local_tb_dir)
            logger.info(
                "TensorBoard buffering locally at %s (will sync to GCS after training)",
                local_tb_dir,
            )
        else:
            alg_kwargs["tensorboard_log"] = str(gcs_tb_path)
    else:
        logger.info("TensorBoard logging disabled")

    # PPO-specific schedule setup
    if algorithm == "ppo":
        # Callback-driven schedule keys — not PPO constructor kwargs.
        # Callers read them from config["ppo_kwargs"] to build
        # EntCoefDecayCallback (see _maybe_ent_coef_decay_callback).
        alg_kwargs.pop("ent_coef_end", None)
        alg_kwargs.pop("ent_coef_decay_timesteps", None)
        lr_end = alg_kwargs.pop("learning_rate_end", None)
        lr_schedule_type = alg_kwargs.pop("lr_schedule", "linear")
        if lr_end is not None:
            lr_start = alg_kwargs["learning_rate"]
            if lr_schedule_type == "cosine":
                alg_kwargs["learning_rate"] = cosine_schedule(lr_start, lr_end)
            else:
                alg_kwargs["learning_rate"] = linear_schedule(lr_start, lr_end)
            logger.info("Using %s LR schedule: %s -> %s", lr_schedule_type, lr_start, lr_end)

        clip_range_end = alg_kwargs.pop("clip_range_end", None)
        if clip_range_end is not None:
            clip_start = alg_kwargs["clip_range"]
            alg_kwargs["clip_range"] = linear_schedule(clip_start, clip_range_end)
            logger.info("Using clip_range schedule: %s -> %s", clip_start, clip_range_end)

    return alg_kwargs, local_tb_dir, gcs_tb_path


#: SB3's ``CheckpointCallback`` names its periodic checkpoints
#: ``{prefix}_{steps}_steps.zip`` (``CheckpointCallback._checkpoint_path``;
#: mirrored by ``curriculum.checkpoints._CHECKPOINT_KINDS``).
_PERIODIC_CHECKPOINT_RE = re.compile(r"(.+)_(\d+)_steps$")


def _is_remote_mount_path(path: "Path | str") -> bool:
    """Whether *path* sits on a FUSE-mounted remote filesystem (GCS or Drive).

    Direct streaming writes to these mounts can be observed — or permanently
    left — truncated when the runtime dies mid-write (see ``file_io``); paths
    that match get the local-stage + atomic-publish treatment.  Colab mounts
    Google Drive at ``/content/drive`` (older tooling used ``/gdrive``).
    """
    text = str(path)
    return _is_gcs_path(text) or text.startswith(("/content/drive", "/gdrive"))


def _resolve_vecnorm_sidecar(load_path: str) -> str:
    """Resolve the VecNormalize sidecar path for a checkpoint being loaded.

    Two sidecar naming conventions coexist: this repository's curated
    checkpoints (``best_model``, ``robust_best_model``, ``stage<N>_final``)
    save ``<base>_vecnorm.pkl``, while SB3's
    ``CheckpointCallback(save_vecnormalize=True)`` writes
    ``<prefix>_vecnormalize_<steps>_steps.pkl`` for its periodic
    ``<prefix>_<steps>_steps.zip``.  Probing only the curated name made a
    ``--load stage2_5000000_steps.zip`` resume warn and then train the loaded
    policy under fresh normalization statistics — silently (review F3).

    A ``load_path`` that already names a ``.pkl`` file is returned unchanged:
    ``train_curriculum`` hands the sidecar path itself, and appending
    ``_vecnorm.pkl`` to it would probe a file that cannot exist.

    Returns the first existing candidate; when none exists, the curated
    ``<base>_vecnorm.pkl`` name, so the caller's warning names the primary
    probe.
    """
    if load_path.endswith(".pkl"):
        return load_path
    base = load_path[:-4] if load_path.endswith(".zip") else load_path
    curated = base + "_vecnorm.pkl"
    if Path(curated).exists():
        return curated
    match = _PERIODIC_CHECKPOINT_RE.match(Path(base).name)
    if match:
        periodic = Path(base).parent / f"{match.group(1)}_vecnormalize_{match.group(2)}_steps.pkl"
        if periodic.exists():
            return str(periodic)
    return curated


def _load_vecnorm_into_envs(
    load_path: str | None,
    train_env,
    eval_env,
    *,
    plant_identity: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    task_load_mode: str,
    allow_fresh_vecnorm: bool = False,
) -> None:
    """Carry forward VecNormalize stats from a prior stage or reset eval env.

    A ``load_path`` whose VecNormalize sidecar is missing **fails closed**:
    training the loaded policy under fresh mean-0/var-1 statistics feeds it
    wildly mis-scaled observations and collapses it within the first updates
    (review TC5 — the notebook resume cell already refused this; the CLI
    warned and continued).  ``allow_fresh_vecnorm=True`` is the explicit
    escape hatch, and its warning names both affected envs.

    On a same-stage resume (``task_load_mode == "resume_same_stage"``) the
    return-normalization statistics are carried forward too — the reward
    distribution is unchanged, so resetting ``ret_rms`` only distorts the
    first post-resume updates (review TC6).
    """
    from .curriculum import load_vecnorm_stats

    if load_path:
        _vecnorm_path = _resolve_vecnorm_sidecar(load_path)
        load_kwargs: dict[str, Any]
        if plant_identity is not None:
            load_kwargs = {
                "current_plant": plant_identity,
                "allow_legacy_plant": allow_legacy_plant,
            }
        else:
            load_kwargs = {"unsafe_skip_plant_validation": True}
        if task_load_mode == "resume_same_stage":
            load_kwargs["carry_ret_rms"] = True
        if not load_vecnorm_stats(_vecnorm_path, train_env, eval_env, **load_kwargs):
            if not allow_fresh_vecnorm:
                raise FileNotFoundError(
                    f"VecNormalize sidecar not found for {load_path!r} (probed {_vecnorm_path!r}). "
                    "Refusing to train the loaded policy under fresh normalization statistics — "
                    "pass --allow-fresh-vecnorm (allow_fresh_vecnorm=True) to proceed deliberately."
                )
            logger.warning(
                "VecNormalize file not found: %s — BOTH the train and eval envs will use fresh "
                "normalization statistics for the loaded policy (allow_fresh_vecnorm)",
                _vecnorm_path,
            )
            eval_env.training = False
            eval_env.norm_reward = False
    else:
        eval_env.training = False
        eval_env.norm_reward = False


def _create_or_load_model(
    sb3: dict,
    algorithm: str,
    alg_kwargs: dict[str, Any],
    train_env,
    load_path: str | None = None,
    *,
    plant_identity: PlantIdentity | None = None,
    allow_legacy_plant: bool = False,
    task_fingerprint: "dict[str, Any] | None" = None,
    task_load_mode: str = "resume_same_stage",
) -> Any:
    """Create a new model or load from checkpoint.

    Pops ``policy_kwargs`` from *alg_kwargs* (mutating it) so that network
    architecture is only applied to new models, not loaded ones.

    ``task_fingerprint``/``task_load_mode`` validate the TASK the checkpoint
    was trained on (STAGE1_SPLIT_PLAN §3.2) — a layer above the plant
    contract, which cannot see a ``step()``-level task change like the
    scheduled pushes.  ``resume_same_stage`` requires an exact match;
    ``initialize_next_stage`` records the boundary as lineage on the new
    checkpoint instead of forbidding it.
    """
    from .task_fingerprint import attach_task_fingerprint, attach_task_lineage, validate_model_task

    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]
    policy_kwargs = alg_kwargs.pop("policy_kwargs", None)

    task_lineage = None
    if load_path:
        logger.info("Loading model from: %s", load_path)
        model = alg_cls.load(load_path, env=train_env, **alg_kwargs)
        if plant_identity is not None:
            validate_model_plant(
                model,
                plant_identity,
                artifact=str(load_path),
                allow_legacy=allow_legacy_plant,
            )
        if task_fingerprint is not None:
            # allow_unfingerprinted: transition valve — no checkpoint minted
            # before 2026-08-15 carries a task fingerprint, so a missing one
            # warns instead of failing.  Tighten to fail-closed once
            # fingerprinted checkpoints are the norm (planned with the gate
            # resolver, plan §W5).
            task_lineage = validate_model_task(
                model,
                task_fingerprint,
                mode=task_load_mode,
                artifact=str(load_path),
                allow_unfingerprinted=True,
            )
        if task_load_mode == "resume_same_stage":
            # A checkpoint saved inside a stage-entry warm-up window pickles
            # ENT_COEF_WARMUP_MARKER=True.  The resume path deliberately
            # attaches no warm-up callback, so nothing would ever clear the
            # restored marker and EntCoefDecayCallback would defer forever —
            # the configured entropy decay silently never runs, and the stale
            # marker re-pickles into every descendant checkpoint (review EE3).
            from .curriculum.schedules import ENT_COEF_WARMUP_MARKER

            if getattr(model, ENT_COEF_WARMUP_MARKER, False):
                setattr(model, ENT_COEF_WARMUP_MARKER, False)
                logger.warning(
                    "Loaded checkpoint was saved inside a stage-entry warm-up window; "
                    "cleared the warm-up marker so entropy decay resumes. The remainder "
                    "of that warm-up is NOT re-applied on a same-stage resume."
                )
    else:
        logger.info("Creating new %s model...", algorithm.upper())
        model = alg_cls("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **alg_kwargs)

    if plant_identity is not None:
        # Attach after validation so an explicit legacy migration is tagged on
        # its next save, while an incompatible checkpoint is never relabelled.
        attach_plant_identity(model, plant_identity)
    if task_fingerprint is not None:
        # Same ordering contract as the plant identity above.
        attach_task_fingerprint(model, task_fingerprint)
        if task_lineage is not None:
            attach_task_lineage(model, task_lineage)

    return model


def _build_core_callbacks(
    sb3: dict,
    eval_env,
    model_dir: Path,
    log_path: Path,
    stage: "int | str",
    n_envs: int,
    eval_freq: int,
    save_freq: int,
    verbose: int,
    stage_config: dict[str, Any],
    use_wandb: bool = False,
    local_tb_dir: Path | None = None,
    gcs_tb_path: Path | None = None,
    species: str | None = None,
) -> tuple[list, Any, Any]:
    """Build the standard callback set shared by train() and train_curriculum().

    Returns ``(callbacks, eval_callback, save_vecnorm_cb)`` so callers can
    append additional stage-specific callbacks.
    """
    from .curriculum import (
        DEFAULT_MAX_CHECKPOINTS,
        CheckpointRetentionCallback,
        PublishEvalArtifactsCallback,
        RobustBestModelCallback,
        SaveVecNormalizeCallback,
        build_baseline_progress_callback,
        build_eval_collapse_early_stop_callback,
    )
    from .diagnostics import DiagnosticsCallback as _DiagCB
    from .eval_diagnostics import build_stage_evaluation_callbacks
    from .tb_sync import PeriodicTbSyncCallback
    from .wandb_integration import WandbCallback

    callbacks: list[Any] = []

    save_vecnorm_cb = SaveVecNormalizeCallback(
        save_path=str(model_dir / "best_model_vecnorm.pkl"),
    )

    # EvalCallback rewrites evaluations.npz in place on every eval; on a
    # Drive/GCS FUSE mount that streaming rewrite can be observed — or
    # permanently left — truncated. Write to local scratch instead and
    # publish atomically to the stage dir after each eval.
    import tempfile as _tempfile

    local_eval_dir = _tempfile.mkdtemp(prefix=f"eval_{stage_label(stage)}_")
    eval_callback, plateau_callback = build_stage_evaluation_callbacks(
        eval_env,
        stage=stage,
        stage_config=stage_config,
        diagnostics_verbose=verbose,
        # The stage dir, not the local scratch eval dir: gate_progress.npz is
        # the artifact a human (or a Drive reader) checks mid-run, so it has
        # to land beside diagnostics.npz rather than in a temp directory.
        gate_progress_dir=log_path,
        best_model_save_path=str(model_dir),
        log_path=local_eval_dir,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=_eval_episodes_for_stage(stage_config),
        deterministic=True,
        render=False,
        verbose=max(verbose, 1),
        callback_on_new_best=save_vecnorm_cb,
    )
    callbacks.append(eval_callback)
    callbacks.append(plateau_callback)
    callbacks.append(PublishEvalArtifactsCallback(eval_callback, publish_dir=log_path))
    # Risk-adjusted (mean - std) checkpoint alongside SB3's mean-reward
    # best_model; next-stage loading prefers it when present.
    callbacks.append(RobustBestModelCallback(eval_callback, model_dir=model_dir, verbose=verbose))

    checkpoint_stride = max(1, save_freq // n_envs)
    # On a Drive/GCS FUSE mount, stream the periodic checkpoint pair to fast
    # local scratch and publish it atomically (copy-to-temp + rename) — the
    # same treatment evaluations.npz gets above, and for the same reason: an
    # ungraceful runtime reclaim inside the direct-to-mount write (or the
    # DriveFS sync window) can leave the newest pair — the sole artifact the
    # resume path depends on — truncated or orphaned.
    checkpoint_save_dir = model_dir
    if _is_remote_mount_path(model_dir):
        checkpoint_save_dir = Path(_tempfile.mkdtemp(prefix=f"ckpt_{stage_label(stage)}_"))
        logger.info(
            "Periodic checkpoints staged locally at %s and published atomically to %s",
            checkpoint_save_dir,
            model_dir,
        )
    checkpoint_callback = sb3["CheckpointCallback"](
        save_freq=checkpoint_stride,
        save_path=str(checkpoint_save_dir),
        name_prefix=stage_label(stage),
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    if checkpoint_save_dir != model_dir:
        from .curriculum import PublishCheckpointPairCallback

        # Immediately after the CheckpointCallback so the pair is published
        # on the same step it was written.
        callbacks.append(
            PublishCheckpointPairCallback(
                local_dir=checkpoint_save_dir,
                publish_dir=model_dir,
                name_prefix=stage_label(stage),
                save_freq=checkpoint_stride,
                verbose=verbose,
            )
        )
    # Immediately after, so a new checkpoint and the prune of the one it
    # displaced land on the same step. SB3's CheckpointCallback never deletes
    # anything: at the former save_freq = 500k a 6M-step stage kept twelve
    # 4.02 MB policy zips plus their VecNormalize sidecars (48 MB of the 60 MB
    # `models/` measured on run 20260801_021545), on a Drive mount, for files
    # nothing in this repository reads. Retention (keep 5) is what makes the
    # denser 100k cadence — chosen so a Colab session cap discards at most
    # ~30 min of training instead of ~2.2 h (review CO4) — storage-neutral.
    # Set `max_checkpoints = 0` in [curriculum] to keep every one.
    callbacks.append(
        CheckpointRetentionCallback(
            model_dir=model_dir,
            name_prefix=stage_label(stage),
            save_freq=checkpoint_stride,
            max_checkpoints=int(
                stage_config.get("curriculum_kwargs", {}).get("max_checkpoints", DEFAULT_MAX_CHECKPOINTS)
            ),
            verbose=verbose,
        )
    )

    callbacks.append(_DiagCB(log_dir=str(log_path), verbose=verbose))

    # Collapse early-stop is a backstop against a genuinely diverging stage.
    # Its thresholds are configurable via the stage's [curriculum] section;
    # defaults are lenient (looser than the former hardcoded 8 / 5 / 0.3) so a
    # normal early-training dip does not abort a stage before entropy decay /
    # LR annealing have had a chance to engage. The peak uses a rolling median
    # of per-eval mean rewards (collapse_smoothing_window), while each raw
    # per-eval mean contributes one patience observation. Detection only arms
    # once that robust peak clears the stage's own reward gate (overridable via
    # collapse_peak_floor), so neither a variance-inflated early eval nor a
    # healthy bimodal transition can trip the backstop.
    collapse_cfg = stage_config.get("curriculum_kwargs", {})
    callbacks.append(
        build_eval_collapse_early_stop_callback(
            eval_callback,
            collapse_cfg,
            verbose=verbose,
        )
    )

    # Advisory only: reports each evaluation against the run's captured
    # zero-action baseline. Run 20260804_143747 spent its whole 10M budget
    # (8h 13m) below the statue on every major reward term and nothing said
    # so -- reward was climbing, full-horizon was 100%, the backstop correctly
    # never fired. The comparison costs one float already on disk. See
    # `curriculum.baseline_watch` for why this warns rather than stops.
    # `species` is optional so the existing positional callers keep working,
    # but a caller that omits it silently loses the watch -- so the notebook
    # passes it and a test pins that it still does.
    baseline_cb = (
        build_baseline_progress_callback(
            eval_callback,
            run_dir=Path(log_path).parent if log_path is not None else None,
            species=species,
            stage_config=stage_config,
            verbose=verbose,
        )
        if species
        else None
    )
    if baseline_cb is not None:
        callbacks.append(baseline_cb)

    if use_wandb:
        callbacks.append(WandbCallback())

    # Flush buffered TensorBoard events to GCS on the checkpoint cadence so
    # a preempted worker (spot VMs) doesn't lose the whole stage's logs.
    if local_tb_dir is not None and gcs_tb_path is not None:
        callbacks.append(PeriodicTbSyncCallback(local_tb_dir, gcs_tb_path, sync_freq=save_freq, verbose=verbose))

    return callbacks, eval_callback, save_vecnorm_cb


def _select_handoff_checkpoint(model_dir: Path) -> "tuple[str, str, str] | None":
    """The checkpoint the curriculum actually promotes to the next stage.

    Preference order matches next-stage loading: the risk-adjusted
    ``robust_best_model``, then SB3's mean-reward ``best_model`` — each
    only when its matched VecNormalize stats exist, so obs normalization
    matches the policy weights. Returns ``(name, model_path_without_ext,
    vecnorm_path)`` or ``None`` when neither candidate is complete.
    """
    for candidate in ("robust_best_model", "best_model"):
        cand_zip = model_dir / f"{candidate}.zip"
        cand_vecnorm = model_dir / f"{candidate}_vecnorm.pkl"
        if cand_zip.exists() and cand_vecnorm.exists():
            return candidate, str(model_dir / candidate), str(cand_vecnorm)
    return None


def _maybe_ent_coef_decay_callback(config: dict, algorithm: str, total_timesteps: int):
    """Build an EntCoefDecayCallback when the PPO config asks for one.

    Reads ``ent_coef_end`` (and optional ``ent_coef_decay_timesteps``,
    defaulting to the stage budget) from ``config["ppo_kwargs"]``.
    Returns ``None`` for SAC or when no decay is configured.
    """
    if algorithm != "ppo":
        return None
    ppo_kwargs = config.get("ppo_kwargs", {})
    end_value = ppo_kwargs.get("ent_coef_end")
    if end_value is None:
        return None
    from .curriculum import EntCoefDecayCallback

    return EntCoefDecayCallback(
        end_value=float(end_value),
        decay_timesteps=int(ppo_kwargs.get("ent_coef_decay_timesteps", total_timesteps)),
    )


def _stage_entry_shaping_callbacks(
    stage_config: dict[str, Any],
    *,
    task_load_mode: str,
    stage_position: int,
    load_path: str | None,
) -> list:
    """Stage-entry shaping (warm-up + reward ramp) for a boundary-crossing load.

    Applies only when a loaded checkpoint ENTERS a new non-first stage —
    ``task_load_mode == "initialize_next_stage"``, the mode
    ``_create_or_load_model`` records as lineage.  A same-stage resume
    (``resume_same_stage``) just passed an exact task-fingerprint identity
    check and must resume the exact task: warm-up clamps and a
    forward-velocity ramp would train it for ~500k steps on a task the
    fingerprint claims is unchanged (review F4).

    ``stage_position`` comes from the stage manifest, so a semantic reference
    ("recovery", position 2) gets the same warm-up an integer one does; for
    legacy integers this is behaviourally identical to the old ``stage > 1``.

    Both launch paths (:func:`train` and :func:`train_curriculum`) MUST build
    their shaping here rather than inline — the notebook's inline copy is how
    the ramp guard below was lost once already (the 20260821 recovery pilot).
    """
    if task_load_mode != "initialize_next_stage" or stage_position <= 1 or not load_path:
        return []

    from .curriculum import RewardRampCallback, StageWarmupCallback

    cur_kwargs = stage_config.get("curriculum_kwargs", {})
    shaping: list = [
        StageWarmupCallback(
            warmup_timesteps=cur_kwargs.get("warmup_timesteps", 100_000),
            warmup_clip_range=cur_kwargs.get("warmup_clip_range", 0.02),
            warmup_ent_coef=cur_kwargs.get("warmup_ent_coef", 0.02),
            warmup_lr_scale=cur_kwargs.get("warmup_lr_scale", 0.1),
        )
    ]
    target_fwd_weight = stage_config["env_kwargs"].get("forward_vel_weight", 1.0)
    # Ramping forward_vel_weight only makes sense when the stage USES it:
    # recovery mirrors stance and sets it to 0.0, and ramping 0.1 -> 0.0
    # would inject a walk incentive the task fingerprint says is absent.
    if target_fwd_weight > 0.0:
        shaping.append(
            RewardRampCallback(
                attr_name="forward_vel_weight",
                start_value=cur_kwargs.get("ramp_start_value", 0.1),
                end_value=target_fwd_weight,
                ramp_timesteps=cur_kwargs.get("ramp_timesteps", 500_000),
            )
        )
    return shaping


def _save_final_and_sync_tb(
    model,
    train_env,
    model_dir: Path,
    stage: "int | str",
    local_tb_dir: Path | None,
    gcs_tb_path: Path,
) -> Path:
    """Save the final model checkpoint and sync TensorBoard events to GCS.

    Returns the final model path (without ``.zip`` extension).
    """
    final_path = model_dir / f"{stage_label(stage)}_final"
    model.save(str(final_path))
    train_env.save(str(final_path) + "_vecnorm.pkl")

    if local_tb_dir is not None:
        try:
            _sync_tb_to_gcs(local_tb_dir, gcs_tb_path)
        except Exception:
            logger.warning("TensorBoard sync to GCS failed.", exc_info=True)

    return final_path


# ── Single-stage training ────────────────────────────────────────────────


def train(
    species_cfg: SpeciesConfig,
    stage_configs: "dict[int | str, dict[str, Any]]",
    stage: "int | str",
    total_timesteps: int,
    n_envs: int = 4,
    seed: int = 42,
    load_path: str | None = None,
    eval_freq: int = 50000,
    save_freq: int = 100000,
    log_dir: str | None = None,
    use_subproc: bool = False,
    verbose: int = 1,
    algorithm: str = "ppo",
    use_wandb: bool = False,
    output_dir: str | None = None,
    use_tensorboard: bool = True,
    allow_legacy_plant: bool = False,
    task_load_mode: str = "resume_same_stage",
    allow_fresh_vecnorm: bool = False,
):
    """Train a single stage of the curriculum.

    ``task_load_mode`` governs how a ``load_path`` checkpoint's recorded
    task fingerprint is validated: ``resume_same_stage`` (default)
    requires an exact task match; ``initialize_next_stage`` records the
    crossing as lineage — the mode for warm-starting a NEW stage from a
    previous stage's checkpoint, e.g. recovery from a stance checkpoint.

    A ``resume_same_stage`` load is treated as a **continuation** of the
    interrupted run: the SB3 step counter keeps counting from the
    checkpoint (``reset_num_timesteps=False``), so ``total_timesteps``
    means "train this many MORE steps", periodic checkpoints keep their
    cumulative numbering (retention would otherwise delete each fresh
    checkpoint while keeping stale pre-crash ones — review TC1),
    progress-anchored LR/clip schedules continue instead of snapping back
    to their starts (review TC3), and the best-model trackers and
    evaluation history are seeded from the stage directory's existing
    ``evaluations.npz`` so a post-resume evaluation can only overwrite
    ``best_model``/``robust_best_model`` by genuinely beating the
    pre-interruption best (review TC2).

    ``allow_fresh_vecnorm`` is the explicit escape hatch for resuming a
    checkpoint whose VecNormalize sidecar is lost; by default that load
    fails closed (review TC5).
    """
    from .config import save_stage_config
    from .task_fingerprint import derive_stage_task_fingerprint
    from .wandb_integration import init_wandb

    sb3 = _ensure_sb3()

    config = stage_configs[stage]
    species = species_cfg.species
    plant_identity = current_plant_identity(species)
    task_fingerprint = derive_stage_task_fingerprint(
        species=species,
        stage=stage,
        backend="stable-baselines3",
        env_kwargs=config.get("env_kwargs", {}),
        plant_identity=plant_identity.to_dict(),
    )

    logger.info("=" * 60)
    logger.info("Training stage %s: %s", stage, config["name"])
    logger.info("Description: %s", config["description"])
    logger.info("=" * 60)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Path
    if output_dir is not None:
        log_path = Path(output_dir)
    elif log_dir is None:
        from .stage_manifest import stage_dirname

        log_path = (
            Path(__file__).parent.parent / species / "logs" / species / f"{stage_dirname(species, stage)}_{timestamp}"
        )
    else:
        log_path = Path(log_dir)

    log_path.mkdir(parents=True, exist_ok=True)
    model_dir = log_path / "models"
    model_dir.mkdir(exist_ok=True)

    logger.info("Log directory: %s", log_path)
    logger.info("Model directory: %s", model_dir)

    save_stage_config(
        log_path,
        stage,
        config,
        algorithm.upper(),
        extra={"seed": seed, "n_envs": n_envs, "timesteps": total_timesteps},
        env_class=species_cfg.env_class,
        species=species_cfg.species,
        plant_identity=plant_identity,
        task_fingerprint=task_fingerprint,
    )

    # Create environments
    # SAC benefits from SubprocVecEnv: MuJoCo is CPU-bound and SAC's off-policy
    # nature means env collection and gradient updates can overlap better when
    # envs run in separate processes.
    effective_subproc = use_subproc or (algorithm == "sac" and n_envs > 1)
    if effective_subproc and not use_subproc:
        logger.info("Auto-enabling SubprocVecEnv for SAC (use --subproc to make explicit)")
    alg_kwargs_key = f"{algorithm}_kwargs"
    alg_gamma = config.get(alg_kwargs_key, {}).get("gamma")
    logger.info("Creating %d training environments...", n_envs)
    train_env = create_vec_env(
        species_cfg,
        stage_configs,
        stage,
        n_envs,
        seed,
        effective_subproc,
        algorithm=algorithm,
        gamma=alg_gamma,
        plant_identity=plant_identity,
    )

    logger.info("Creating evaluation environment...")
    eval_env = create_vec_env(
        species_cfg,
        stage_configs,
        stage,
        1,
        seed + 1000,
        use_subproc=False,
        algorithm=algorithm,
        gamma=alg_gamma,
        plant_identity=plant_identity,
    )

    _load_vecnorm_into_envs(
        load_path,
        train_env,
        eval_env,
        plant_identity=plant_identity,
        allow_legacy_plant=allow_legacy_plant,
        task_load_mode=task_load_mode,
        allow_fresh_vecnorm=allow_fresh_vecnorm,
    )

    alg_kwargs, local_tb_dir, gcs_tb_path = _prepare_alg_kwargs(
        config,
        algorithm,
        verbose,
        log_path,
        use_tensorboard,
    )

    wandb_run = None
    if use_wandb:
        wandb_run = init_wandb(species=species, stage=stage, config=config, run_dir=str(log_path))

    model = _create_or_load_model(
        sb3,
        algorithm,
        alg_kwargs,
        train_env,
        load_path,
        plant_identity=plant_identity,
        allow_legacy_plant=allow_legacy_plant,
        task_fingerprint=task_fingerprint,
        # Default resume_same_stage: a user --load continues the same task.
        # The CLI's --load-mode initialize_next_stage is the deliberate
        # boundary-crossing path (e.g. recovery warm-started from stance).
        task_load_mode=task_load_mode,
    )

    logger.info("Model architecture:")
    logger.info("  Policy: %s", model.policy)
    logger.info("  Learning rate: %s", model.learning_rate)
    logger.info("  Batch size: %s", alg_kwargs.get("batch_size", "N/A"))

    # A same-stage resume continues the interrupted run: the loaded step
    # counter is preserved (reset_num_timesteps=False below), so
    # total_timesteps means "this many MORE steps" and every step-anchored
    # mechanism (checkpoint numbering, retention, schedule progress, entropy
    # decay, eval history timesteps) stays on the run's cumulative axis.
    resuming = bool(load_path) and task_load_mode == "resume_same_stage"
    loaded_steps = int(getattr(model, "num_timesteps", 0)) if resuming else 0
    target_timesteps = loaded_steps + total_timesteps
    if resuming and loaded_steps > 0:
        logger.info(
            "Resuming at %s cumulative steps; training %s more (target %s).",
            f"{loaded_steps:,}",
            f"{total_timesteps:,}",
            f"{target_timesteps:,}",
        )

    callbacks, eval_callback, _ = _build_core_callbacks(
        sb3,
        eval_env,
        model_dir,
        log_path,
        stage,
        n_envs,
        eval_freq,
        save_freq,
        verbose,
        config,
        use_wandb,
        local_tb_dir=local_tb_dir,
        gcs_tb_path=gcs_tb_path,
        species=species,
    )

    if resuming:
        from .curriculum import seed_resume_eval_state

        # Seed best-model trackers and the in-memory eval history from the
        # stage directory's published record, so the first post-resume eval
        # cannot overwrite a better pre-interruption best_model /
        # robust_best_model, and evaluations.npz keeps the whole run.
        seed_resume_eval_state(eval_callback, callbacks, Path(log_path) / "evaluations.npz")

    # The decay anchor is an ABSOLUTE step count.  On a resume the default
    # (when the TOML sets no ent_coef_decay_timesteps) must be the cumulative
    # target, not this call's remaining budget — otherwise a late resume
    # decays over a compressed horizon.
    ent_decay_cb = _maybe_ent_coef_decay_callback(config, algorithm, target_timesteps)
    if ent_decay_cb is not None:
        callbacks.append(ent_decay_cb)

    # Stage-entry shaping is keyed on the load MODE, not on stage position
    # alone: a resume_same_stage --load of a non-first stage passes the exact
    # task-fingerprint check above and must resume un-warmed and un-ramped.
    from .stage_manifest import load_stage_manifest

    stage_position = load_stage_manifest(species).resolve(stage).position
    callbacks.extend(
        _stage_entry_shaping_callbacks(
            config,
            task_load_mode=task_load_mode,
            stage_position=stage_position,
            load_path=load_path,
        )
    )

    callback_list = sb3["CallbackList"](callbacks)

    # Train
    logger.info("Starting training for %s timesteps...", f"{total_timesteps:,}")
    logger.info("-" * 60)

    train_start = time.monotonic()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=verbose >= 1,
            # On resume, keep the checkpoint's cumulative counter: SB3 then
            # trains total_timesteps MORE steps, checkpoint filenames stay
            # cumulative (so retention keeps the newest, not the stale
            # pre-crash set), and progress-based schedules continue from
            # where the interrupted run left off.
            reset_num_timesteps=not resuming,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    training_duration = time.monotonic() - train_start

    # Actual CUMULATIVE steps trained — differs from the target when a
    # callback (e.g. EvalCollapseEarlyStopCallback) or Ctrl-C ended training
    # early.  On a resume this includes the loaded checkpoint's steps.
    actual_timesteps = int(model.num_timesteps)
    if actual_timesteps < target_timesteps:
        logger.warning(
            "Training ended early at %s of %s timesteps.",
            f"{actual_timesteps:,}",
            f"{target_timesteps:,}",
        )

    if wandb_run is not None:
        wandb_run.finish()

    # Save the final checkpoint *before* the post-training evaluation so a
    # failure (or second Ctrl-C) during the ~80 serial eval episodes can't
    # lose the model.
    final_path = _save_final_and_sync_tb(
        model,
        train_env,
        model_dir,
        stage,
        local_tb_dir,
        gcs_tb_path,
    )

    # Report metrics to Vertex AI HPT (no-op when cloudml-hypertune not installed)
    _report_hpt_metrics(
        species_cfg,
        model,
        eval_env,
        eval_callback,
        log_path,
        model_dir,
        stage,
        actual_timesteps,
        algorithm,
        training_duration_seconds=training_duration,
        stage_config=config,
        seed=seed,
        plant_identity=plant_identity,
    )

    train_env.close()
    eval_env.close()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Final model: %s.zip", final_path)
    logger.info("VecNormalize stats: %s_vecnorm.pkl", final_path)
    logger.info("=" * 60)

    return model


# ── HPT metric reporting ─────────────────────────────────────────────────


def _report_hpt_metrics(
    species_cfg: SpeciesConfig,
    model,
    eval_env,
    eval_callback,
    log_path: Path,
    model_dir: Path,
    stage: "int | str",
    total_timesteps: int,
    algorithm: str,
    training_duration_seconds: float = 0.0,
    stage_config: dict[str, Any] | None = None,
    seed: int | None = None,
    plant_identity: PlantIdentity | None = None,
):
    """Report metrics to Vertex AI Hypertune and write a local JSON sidecar.

    Only ``best_mean_reward`` is declared in the HPT ``metric_spec`` (it
    is the sole optimisation target).  All auxiliary metrics are written
    to ``<log_path>/metrics.json`` so they can be collected from GCS
    after the sweep completes — without polluting the HPT objective.

    For forward velocity and success rate (stages 2+), the **best model**
    checkpoint is loaded with its matched VecNormalize stats so the
    reported metrics reflect the checkpoint that will be handed off to
    the next stage — not the final model which may have regressed.
    """
    import json as _json

    import numpy as _np

    sb3 = _ensure_sb3()

    from .config import get_library_version

    # Accumulate all metrics for the JSON sidecar.  Run identity + effective
    # seed make each trial reproducible from the collected CSV alone.
    aux_metrics: dict[str, Any] = {
        "species": species_cfg.species,
        "algorithm": algorithm,
        "library_version": get_library_version(),
        "best_mean_reward": float(eval_callback.best_mean_reward),
        "training_duration_seconds": round(training_duration_seconds, 1),
        # Actual steps trained (callers pass model.num_timesteps), so
        # early-stopped runs are visible in offline result collection.
        "timesteps": int(total_timesteps),
    }
    if plant_identity is not None:
        aux_metrics["plant_identity"] = plant_identity.to_dict()
    if seed is not None:
        aux_metrics["seed"] = seed

    # Report the primary optimisation metric to HPT (if available).
    try:
        import hypertune as _hypertune

        _hypertune.HyperTune().report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="best_mean_reward",
            metric_value=eval_callback.best_mean_reward,
            global_step=total_timesteps,
        )
        logger.info(
            "HPT metric reported: best_mean_reward=%.4f",
            eval_callback.best_mean_reward,
        )
    except ImportError:
        logger.info(
            "cloudml-hypertune not installed — HPT metric not reported (best_mean_reward=%.4f)",
            eval_callback.best_mean_reward,
        )

    eval_npz_path = Path(log_path) / "evaluations.npz"
    if eval_npz_path.exists():
        eval_data = _np.load(str(eval_npz_path))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        mean_rewards_per_eval = eval_rewards.mean(axis=1)

        best_eval_idx = int(mean_rewards_per_eval.argmax())
        best_mean_ep_length = float(eval_lengths[best_eval_idx].mean())
        aux_metrics["best_mean_episode_length"] = best_mean_ep_length
        logger.info(
            "HPT metric reported: best_mean_episode_length=%.1f",
            best_mean_ep_length,
        )

        last_mean_reward = float(mean_rewards_per_eval[-1])
        last_mean_ep_length = float(eval_lengths[-1].mean())
        aux_metrics["last_mean_reward"] = last_mean_reward
        aux_metrics["last_mean_episode_length"] = last_mean_ep_length
        logger.info(
            "HPT metric reported: last_mean_reward=%.4f, last_mean_episode_length=%.1f",
            last_mean_reward,
            last_mean_ep_length,
        )

    # ── Post-training quality evaluation (all stages) ──────────────────
    # Run evaluation rollouts with LocomotionMetrics to collect spinning
    # detection signals, heading stability, and reward component breakdown.
    # These metrics enable model selection beyond raw reward.
    from .curriculum import load_vecnorm_stats

    best_model_zip = model_dir / "best_model.zip"
    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]

    # Evaluate the same checkpoint the curriculum hands to the next stage
    # (robust_best_model before mean-reward best_model, via
    # _select_handoff_checkpoint), so metrics.json describes the promoted
    # policy rather than a possibly-degenerate lucky-peak mean-best one
    # (run 20260720_203454's best_model was the 50k checkpoint whose eval
    # was 261.79 ± 261.72). The chosen name is recorded in the sidecar as
    # quality_eval_checkpoint.
    handoff = _select_handoff_checkpoint(model_dir)
    if handoff is not None:
        ckpt_name, ckpt_path, ckpt_vecnorm = handoff
        eval_model = alg_cls.load(ckpt_path, env=eval_env)
        if plant_identity is not None:
            validate_model_plant(eval_model, plant_identity, artifact=ckpt_path + ".zip")
        load_kwargs: dict[str, Any]
        if plant_identity is not None:
            load_kwargs = {"current_plant": plant_identity}
        else:
            load_kwargs = {"unsafe_skip_plant_validation": True}
        load_vecnorm_stats(ckpt_vecnorm, eval_env, **load_kwargs)
        eval_env.training = False
        eval_env.norm_reward = False
        aux_metrics["quality_eval_checkpoint"] = ckpt_name
        logger.info("HPT eval: using %s + matched VecNormalize", ckpt_name)
    elif best_model_zip.exists():
        # Legacy fallback: a best_model saved without matched VecNormalize
        # stats. Evaluate it rather than nothing, but flag the mismatch.
        eval_model = alg_cls.load(str(model_dir / "best_model"), env=eval_env)
        if plant_identity is not None:
            validate_model_plant(eval_model, plant_identity, artifact=str(best_model_zip))
        eval_env.training = False
        eval_env.norm_reward = False
        aux_metrics["quality_eval_checkpoint"] = "best_model"
        logger.warning(
            "HPT eval: best_model has no matched VecNormalize stats — quality eval "
            "normalization may not match the policy weights"
        )
    else:
        eval_model = model
        eval_env.training = False
        eval_env.norm_reward = False
        aux_metrics["quality_eval_checkpoint"] = "final_model"
        logger.warning("HPT eval: no saved checkpoint found, falling back to final model")

    # Quality evaluation with full LocomotionMetrics (spinning detection,
    # heading alignment, reward breakdown, etc.)
    from .evaluation import eval_policy_quality

    try:
        quality_metrics = eval_policy_quality(eval_model, eval_env, species_cfg.success_keys, n_episodes=50)
        aux_metrics.update(quality_metrics)
        logger.info(
            "Quality eval complete: %d metrics collected (angular_vel=%.3f, heading_align=%.3f)",
            len(quality_metrics),
            quality_metrics.get("eval_mean_pelvis_angular_velocity", float("nan")),
            quality_metrics.get("eval_mean_heading_alignment", float("nan")),
        )
    except Exception:
        logger.warning("Quality evaluation failed — skipping quality metrics.", exc_info=True)

    # Forward velocity, distance, and success rate evaluation.
    # Run for all stages so mean_distance_traveled is always captured.
    # Guarded so a mid-eval failure still writes the metrics.json sidecar
    # with whatever was collected above.
    try:
        _, _, fwd_vels, success_flags, distances = eval_policy(
            eval_model,
            eval_env,
            species_cfg.success_keys,
            n_episodes=30,
        )
    except Exception:
        logger.warning("Post-training eval_policy failed — skipping velocity/success metrics.", exc_info=True)
        fwd_vels, success_flags, distances = [], [], []
    if fwd_vels:
        mean_fwd = float(_np.mean(fwd_vels))
        std_fwd = float(_np.std(fwd_vels))
        aux_metrics["mean_forward_vel"] = mean_fwd
        aux_metrics["std_forward_vel"] = std_fwd
        # Keep backward-compat alias used by existing sweep analysis.
        aux_metrics["best_mean_forward_vel"] = mean_fwd
        logger.info("HPT metric reported: mean_forward_vel=%.4f (std=%.4f)", mean_fwd, std_fwd)
    if distances:
        mean_dist = float(_np.mean(distances))
        aux_metrics["mean_distance_traveled"] = mean_dist
        logger.info("HPT metric reported: mean_distance_traveled=%.4f", mean_dist)
    if success_flags:
        mean_success = float(_np.mean(success_flags))
        aux_metrics["mean_success_rate"] = mean_success
        # Keep backward-compat alias used by existing sweep analysis.
        aux_metrics["best_mean_success_rate"] = mean_success
        logger.info("HPT metric reported: mean_success_rate=%.4f", mean_success)

    # Include key hyperparameters in the sidecar so offline result
    # collection works even when stage_config.json is missing.
    if stage_config is not None:
        algo_key = "sac_kwargs" if algorithm == "sac" else "ppo_kwargs"
        algo_kwargs = stage_config.get(algo_key, {})
        _metric_keys = (
            ("learning_rate", "batch_size", "gamma", "n_steps", "ent_coef")
            if algorithm == "ppo"
            else ("learning_rate", "batch_size", "gamma", "tau", "buffer_size", "ent_coef")
        )
        for k in _metric_keys:
            if k in algo_kwargs:
                val = algo_kwargs[k]
                # Skip callable schedules — store the initial value description
                if not callable(val):
                    aux_metrics[f"{algorithm}_{k}"] = val
        net_arch = algo_kwargs.get("policy_kwargs", {}).get("net_arch")
        if net_arch is not None:
            aux_metrics[f"{algorithm}_net_arch"] = str(net_arch)

    # Write all metrics to a JSON sidecar so they can be collected from
    # GCS without relying on the HPT metric_spec.
    metrics_path = Path(log_path) / "metrics.json"
    with open(metrics_path, "w") as f:
        _json.dump(aux_metrics, f, indent=2)
    logger.info("Metrics sidecar written to: %s", metrics_path)


# ── Curriculum training ──────────────────────────────────────────────────


def train_curriculum(
    species_cfg: SpeciesConfig,
    stage_configs: "dict[int | str, dict[str, Any]]",
    n_envs: int = 4,
    seed: int = 42,
    eval_freq: int = 50000,
    save_freq: int = 100000,
    log_dir: str | None = None,
    use_subproc: bool = False,
    verbose: int = 1,
    algorithm: str = "ppo",
    use_wandb: bool = False,
    output_dir: str | None = None,
    gcs_bucket: str | None = None,
    gcs_project: str | None = None,
    use_tensorboard: bool = True,
):
    """Run the full 3-stage curriculum with automatic advancement."""
    from .config import (
        save_stage_config,
        upload_curriculum_artifacts,
    )
    from .curriculum import (
        CurriculumCallback,
        CurriculumManager,
        thresholds_from_configs,
    )
    from .task_fingerprint import derive_stage_task_fingerprint
    from .wandb_integration import init_wandb

    sb3 = _ensure_sb3()
    species = species_cfg.species
    plant_identity = current_plant_identity(species)

    thresholds = thresholds_from_configs(stage_configs)
    manager = CurriculumManager(species=species, stage_thresholds=thresholds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is not None:
        base_dir = Path(output_dir)
    elif log_dir is None:
        base_dir = Path(__file__).parent.parent / species / "logs" / species / f"curriculum_{timestamp}"
    else:
        base_dir = Path(log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    from .plant_contract import write_plant_identity

    write_plant_identity(base_dir / "plant_identity.json", plant_identity)

    logger.info("=" * 60)
    logger.info("Starting automated curriculum training (stages 1-3)")
    logger.info("Base directory: %s", base_dir)
    logger.info("=" * 60)

    model = None
    load_path = None
    prev_vecnorm_path = None

    for stage in range(1, 4):
        config = stage_configs[stage]
        cur_kwargs = config.get("curriculum_kwargs", {})
        total_timesteps = cur_kwargs.get("timesteps", 500000)

        from .stage_manifest import stage_dirname

        stage_dir = base_dir / stage_dirname(species, stage)
        stage_dir.mkdir(exist_ok=True)
        model_dir = stage_dir / "models"
        model_dir.mkdir(exist_ok=True)

        logger.info("=" * 60)
        logger.info("Curriculum Stage %d/%d: %s", stage, 3, config["name"])
        logger.info("Description: %s", config["description"])
        logger.info("Timesteps: %s", f"{total_timesteps:,}")
        logger.info("=" * 60)

        task_fingerprint = derive_stage_task_fingerprint(
            species=species_cfg.species,
            stage=stage,
            backend="stable-baselines3",
            env_kwargs=config.get("env_kwargs", {}),
            plant_identity=plant_identity.to_dict(),
        )
        save_stage_config(
            stage_dir,
            stage,
            config,
            algorithm.upper(),
            extra={"seed": seed, "n_envs": n_envs, "timesteps": total_timesteps},
            env_class=species_cfg.env_class,
            species=species_cfg.species,
            plant_identity=plant_identity,
            task_fingerprint=task_fingerprint,
        )

        effective_subproc = use_subproc or (algorithm == "sac" and n_envs > 1)
        alg_kwargs_key = f"{algorithm}_kwargs"
        alg_gamma = config.get(alg_kwargs_key, {}).get("gamma")
        train_env = create_vec_env(
            species_cfg,
            stage_configs,
            stage,
            n_envs,
            seed,
            effective_subproc,
            algorithm=algorithm,
            gamma=alg_gamma,
            plant_identity=plant_identity,
        )
        eval_env = create_vec_env(
            species_cfg,
            stage_configs,
            stage,
            1,
            seed + 1000,
            use_subproc=False,
            algorithm=algorithm,
            gamma=alg_gamma,
            plant_identity=plant_identity,
        )

        _load_vecnorm_into_envs(
            prev_vecnorm_path,
            train_env,
            eval_env,
            plant_identity=plant_identity,
            # Stage boundary: obs_rms carries, ret_rms resets (the reward
            # distribution changes with the new stage's terms).
            task_load_mode="initialize_next_stage",
        )

        alg_kwargs, local_tb_dir, gcs_tb_path = _prepare_alg_kwargs(
            config,
            algorithm,
            verbose,
            stage_dir,
            use_tensorboard,
        )

        wandb_run = None
        if use_wandb:
            wandb_run = init_wandb(species=species, stage=stage, config=config, run_dir=str(stage_dir))

        model = _create_or_load_model(
            sb3,
            algorithm,
            alg_kwargs,
            train_env,
            load_path,
            plant_identity=plant_identity,
            task_fingerprint=task_fingerprint,
            # Inside the curriculum loop, load_path is only ever the previous
            # stage's promoted checkpoint (it starts None and is assigned
            # exclusively by the stage handoff), so every load here crosses a
            # stage/task boundary deliberately and is recorded as lineage.
            task_load_mode="initialize_next_stage",
        )

        callbacks, eval_callback, _ = _build_core_callbacks(
            sb3,
            eval_env,
            model_dir,
            stage_dir,
            stage,
            n_envs,
            eval_freq,
            save_freq,
            verbose,
            config,
            use_wandb,
            local_tb_dir=local_tb_dir,
            gcs_tb_path=gcs_tb_path,
            species=species,
        )

        ent_decay_cb = _maybe_ent_coef_decay_callback(config, algorithm, total_timesteps)
        if ent_decay_cb is not None:
            callbacks.append(ent_decay_cb)

        curriculum_cb = CurriculumCallback(
            curriculum_manager=manager,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=_eval_episodes_for_stage(config),
            eval_callback=eval_callback,
            supplementary_episodes=cur_kwargs.get("supplementary_episodes", 10),
        )
        callbacks.append(curriculum_cb)

        # Every stage after the first enters on the previous stage's promoted
        # checkpoint — the same initialize_next_stage boundary recorded above.
        # The shared helper also applies train()'s forward_vel_weight > 0 ramp
        # guard: a stage that sets the weight to 0.0 (recovery mirrors stance)
        # must not have a walk incentive ramped through it.
        callbacks.extend(
            _stage_entry_shaping_callbacks(
                config,
                task_load_mode="initialize_next_stage",
                stage_position=stage,
                load_path=load_path,
            )
        )

        interrupted = False
        stage_start = time.monotonic()
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=sb3["CallbackList"](callbacks),
                progress_bar=verbose >= 1,
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            interrupted = True
        stage_duration = time.monotonic() - stage_start

        # Actual steps trained — differs from total_timesteps when a
        # callback (e.g. EvalCollapseEarlyStopCallback) or Ctrl-C ended
        # training early.
        actual_timesteps = int(model.num_timesteps)
        if actual_timesteps < total_timesteps:
            logger.warning(
                "Stage %d ended early at %s of %s timesteps.",
                stage,
                f"{actual_timesteps:,}",
                f"{total_timesteps:,}",
            )

        if wandb_run is not None:
            wandb_run.finish()

        final_path = _save_final_and_sync_tb(
            model,
            train_env,
            model_dir,
            stage,
            local_tb_dir,
            gcs_tb_path,
        )

        # Prefer loading the risk-adjusted robust_best_model (highest
        # mean - std eval, saved by RobustBestModelCallback), then SB3's
        # mean-reward best_model, then the final checkpoint — each with
        # its matched VecNormalize so obs normalization matches the
        # policy weights.
        load_path = str(final_path)
        prev_vecnorm_path = str(final_path) + "_vecnorm.pkl"
        handoff = _select_handoff_checkpoint(model_dir)
        if handoff is not None:
            candidate, load_path, prev_vecnorm_path = handoff
            logger.info(
                "Next stage will load %s (%s) with VecNormalize: %s",
                candidate,
                load_path,
                prev_vecnorm_path,
            )

        train_env.close()
        eval_env.close()

        # Record stage result to CSV
        _record_stage_result(
            species,
            algorithm,
            timestamp,
            base_dir,
            stage,
            config,
            cur_kwargs,
            eval_callback,
            stage_dir,
            seed,
            n_envs,
            actual_timesteps,
            curriculum_cb,
            training_duration_seconds=stage_duration,
            plant_identity=plant_identity,
        )

        if interrupted:
            break

        if curriculum_cb and curriculum_cb.ready_to_advance and not manager.is_final_stage:
            manager.advance()
            logger.info("Auto-advanced to stage %d", manager.current_stage)
        elif stage < 3:
            logger.warning(
                "Stage %d timestep budget exhausted without meeting advancement "
                "thresholds. Stopping curriculum — advancing with a weak policy "
                "causes catastrophic forgetting.",
                stage,
            )
            break

    upload_curriculum_artifacts(
        base_dir,
        species=species,
        algorithm=algorithm,
        bucket=gcs_bucket,
        project=gcs_project,
    )

    logger.info("=" * 60)
    logger.info("Curriculum training complete!")
    logger.info("Results directory: %s", base_dir)
    logger.info("=" * 60)


def _record_stage_result(
    species,
    algorithm,
    timestamp,
    base_dir,
    stage,
    config,
    cur_kwargs,
    eval_callback,
    stage_dir,
    seed,
    n_envs,
    total_timesteps,
    curriculum_cb,
    training_duration_seconds: float | None = None,
    plant_identity: PlantIdentity | None = None,
):
    """Record stage hyperparameters and outcome to CSV."""
    import numpy as _np

    from .config import append_stage_result_csv

    algo_prefix = "sac" if algorithm == "sac" else "ppo"
    algo_key = f"{algo_prefix}_kwargs"
    algo_kwargs = config[algo_key]
    env_kwargs = config["env_kwargs"]

    best_mean_reward: float | str = eval_callback.best_mean_reward
    best_mean_episode_length: float | str = ""
    last_mean_reward: float | str = ""
    last_mean_episode_length: float | str = ""
    eval_npz = stage_dir / "evaluations.npz"
    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        mean_rewards_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_rewards_per_eval.argmax())
        best_mean_reward = round(float(mean_rewards_per_eval[best_idx]), 2)
        best_mean_episode_length = round(float(eval_lengths[best_idx].mean()), 1)
        # Last eval as "final" metrics
        last_mean_reward = round(float(mean_rewards_per_eval[-1]), 2)
        last_mean_episode_length = round(float(eval_lengths[-1].mean()), 1)

    net_arch_val = algo_kwargs.get("policy_kwargs", {}).get("net_arch", "")
    if isinstance(net_arch_val, (list, tuple)):
        net_arch_str = str(list(net_arch_val))
    else:
        net_arch_str = str(net_arch_val) if net_arch_val else ""

    # Use canonical column names matching CSV_METRIC_COLUMNS and prefixed
    # hyperparameter conventions from the sweep CSV format.
    result_row: dict = {
        "species": species,
        "algorithm": algorithm.upper(),
        "run_date": timestamp,
        "run_dir": base_dir.name,
        "stage": stage,
        "stage_name": config["name"],
        "seed": seed,
        "n_envs": n_envs,
        # Prefixed hyperparameters (matching sweep CSV conventions)
        f"{algo_prefix}_learning_rate": algo_kwargs.get("learning_rate", ""),
        f"{algo_prefix}_batch_size": algo_kwargs.get("batch_size", ""),
        f"{algo_prefix}_gamma": algo_kwargs.get("gamma", ""),
        f"{algo_prefix}_net_arch": net_arch_str,
        "env_alive_bonus": env_kwargs.get("alive_bonus", ""),
        "env_energy_penalty_weight": env_kwargs.get("energy_penalty_weight", ""),
        "env_forward_vel_weight": env_kwargs.get("forward_vel_weight", ""),
        "env_posture_weight": env_kwargs.get("posture_weight", ""),
        # Canonical metric columns
        "best_mean_reward": best_mean_reward,
        "best_mean_episode_length": best_mean_episode_length,
        "last_mean_reward": last_mean_reward,
        "last_mean_episode_length": last_mean_episode_length,
        "training_duration_seconds": (
            round(training_duration_seconds, 1) if training_duration_seconds is not None else ""
        ),
        "reward_threshold": cur_kwargs.get("min_avg_reward", ""),
        "ep_length_threshold": cur_kwargs.get("min_avg_episode_length", ""),
        "forward_vel_threshold": cur_kwargs.get("min_avg_forward_vel", ""),
        "success_rate_threshold": cur_kwargs.get("min_success_rate", ""),
        # stance_quality/v1. A stance-gated stage otherwise records only its
        # reward rail, which reads as though reward were the gate.
        "gate_kind": cur_kwargs.get("gate_kind", ""),
        "full_horizon_fraction_threshold": cur_kwargs.get("min_full_horizon_fraction", ""),
        "unsupported_duty_ceiling": cur_kwargs.get("max_unsupported_duty", ""),
        "unsupported_duty_ucb_ceiling": cur_kwargs.get("max_unsupported_duty_ucb", ""),
        "stage_passed": bool(curriculum_cb is not None and curriculum_cb.ready_to_advance),
    }
    if plant_identity is not None:
        result_row.update({f"plant_{key}": value for key, value in plant_identity.to_dict().items()})
    append_stage_result_csv(base_dir / "curriculum_results.csv", result_row)
    logger.info(
        "Stage %d result appended to: %s",
        stage,
        base_dir / "curriculum_results.csv",
    )


# ── Backward-compatible re-exports ──────────────────────────────────────
# These were extracted to dedicated modules but are re-exported here so
# existing ``from environments.shared.train_base import ...`` continues
# to work without changes.

from .cli import _apply_overrides, _cast_value, main  # noqa: E402, F401
from .diagnostics import DiagnosticsCallback  # noqa: E402, F401
from .evaluation import eval_policy, evaluate, record_stage_video  # noqa: E402, F401
