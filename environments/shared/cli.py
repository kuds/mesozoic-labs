"""CLI entry point for species training scripts.

Extracted from ``train_base.py`` for maintainability.  Provides the
``main()`` function used by each species' ``train_sb3.py`` wrapper, along
with argument parsing and config-override helpers.
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _cast_value(v: str):
    """Auto-cast a string value to bool, int, float, or keep as string.

    Handles float-encoded integers (e.g. ``"128.0"`` -> ``128``) which
    Vertex AI HPT sends for ``DiscreteParameterSpec`` values, and
    ``true``/``false`` literals (otherwise ``bool("false")`` truthiness
    bites anyone overriding a boolean kwarg).
    """
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            f = float(v)
            if f.is_integer():
                return int(f)
            return f
        except ValueError:
            return v


def _resolve_override_stage_token(token: str, configs: dict, species: "str | None") -> "int | str | None":
    """Map an override's leading token onto its ``configs`` key, if it is one.

    Digits mean a legacy stage number; a string token can be a semantic
    config key directly (``recovery``) or, when *species* is known, any
    manifest stage id (``stance``, ``locomotion``) resolved through the
    manifest.
    Returns ``None`` when the token does not name a stage — the override is
    then an all-stages ``section.key`` form.
    """
    if token.isdigit():
        return int(token)
    if token in configs:
        return token
    if species is not None:
        from .stage_manifest import StageManifestError, load_stage_manifest

        try:
            entry = load_stage_manifest(species).by_id(token)
        except StageManifestError:
            return None
        if entry.id in configs:
            return entry.id
        if entry.legacy_number is not None and entry.legacy_number in configs:
            return entry.legacy_number
    return None


def _apply_overrides(configs: dict, overrides: list | None, species: "str | None" = None) -> None:
    """Apply dot-notation ``key=value`` overrides to stage configs.

    Two formats are supported:

    - ``section.key=value``       -- applies to **all** stages
    - ``STAGE.section.key=value`` -- applies to one stage only, where STAGE
      is a legacy number (``2``), a semantic config key (``recovery``), or —
      when *species* is given — any manifest stage id (``stance``,
      ``locomotion``)

    Unknown stages and unknown config sections raise instead of silently
    no-opping or crashing with a bare ``KeyError``: a typo'd override used to
    train the full multi-hour stage budget at unmodified hyperparameters
    with nothing in the log but the absence of an "override" line, and the
    semantic ``recovery`` stage — the active training focus — could not be
    targeted at all (gap review TC10/CI6).
    """
    if not overrides:
        return
    for item in overrides:
        key, _, raw_value = item.partition("=")
        value = _cast_value(raw_value)
        parts = key.split(".")
        stage_key = _resolve_override_stage_token(parts[0], configs, species) if len(parts) == 3 else None
        if len(parts) == 3 and (stage_key is None or stage_key not in configs):
            # The token was meant as a stage whenever the MIDDLE token is a
            # real config section: a typo'd semantic id ('recvery.env.x')
            # must get the stage-flavored error with the available choices,
            # not a misattributed unknown-section one.
            middle_kwargs = "env_kwargs" if parts[1] == "env" else f"{parts[1]}_kwargs"
            if parts[0].isdigit() or any(middle_kwargs in cfg for cfg in configs.values()):
                raise ValueError(
                    f"--override {key!r} names unknown stage {parts[0]!r}; available: {sorted(map(str, configs))}"
                )
        if stage_key is not None and stage_key in configs:
            section, param = parts[1], parts[2]
            kwargs_key = "env_kwargs" if section == "env" else f"{section}_kwargs"
            if kwargs_key not in configs[stage_key]:
                raise ValueError(f"--override {key!r} names unknown config section {section!r}")
            configs[stage_key][kwargs_key][param] = value
            logger.info(
                "Stage %s override: %s.%s = %r",
                stage_key,
                section,
                param,
                value,
            )
        else:
            section, _, param = key.partition(".")
            kwargs_key = "env_kwargs" if section == "env" else f"{section}_kwargs"
            for stage_config in configs.values():
                if kwargs_key not in stage_config:
                    raise ValueError(f"--override {key!r} names unknown config section {section!r}")
                stage_config[kwargs_key][param] = value
            logger.info("Override applied: %s.%s = %r", section, param, value)


def _non_negative_int(value: str) -> int:
    """argparse type for a count that may be zero but never negative.

    ``--post-eval-episodes`` was a bare ``int``, so ``-1`` sailed through
    to train() and ran both post-training panels over empty ranges.
    """
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}") from None
    if count < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {count}")
    return count


def _parse_stage_ref(value: str) -> "int | str":
    """argparse type for --stage: digits mean a legacy stage number (their
    historical meaning, per the stage manifest), anything else a semantic
    stage ID such as "recovery"."""
    return int(value) if value.isdigit() else value


def _default_resume_timesteps(load_path: "str | None", load_mode: str, stage_budget: int) -> "int | None":
    """Remaining-budget default for a same-stage resume of a periodic checkpoint.

    With no explicit ``--timesteps``, a ``--load`` resume used to default to
    the stage's FULL TOML budget on top of the checkpoint's steps — an
    interrupted 11M stance stage resumed at 8M trained 11M more (review TC4).
    When the loaded file is one of SB3's periodic
    ``<prefix>_<steps>_steps.zip`` checkpoints and the mode is
    ``resume_same_stage``, default to ``max(budget - steps, 0)`` instead —
    the same arithmetic the notebook resume cell performs.

    Returns ``None`` when the default should stay the full stage budget
    (no load, a boundary-crossing load, or a curated checkpoint whose
    cumulative step count is not in its filename).
    """
    if not load_path or load_mode != "resume_same_stage":
        return None
    from .train_base import _PERIODIC_CHECKPOINT_RE

    stem = Path(load_path).name
    if stem.endswith(".zip"):
        stem = stem[: -len(".zip")]
    match = _PERIODIC_CHECKPOINT_RE.match(stem)
    if not match:
        return None
    return max(stage_budget - int(match.group(2)), 0)


def _resolve_stage_ref(stage_ref: "int | str", stage_configs: dict, species: str) -> "int | str | None":
    """Map a --stage reference onto its ``stage_configs`` key.

    A ref that is already a config key passes through: legacy numbers
    (1-3) and semantic ids without a numeric history ("recovery").  A
    semantic id naming a LEGACY stage resolves to its historical number,
    so ``--stage locomotion`` works wherever ``--stage 2`` does.  Returns
    ``None`` for a ref this species cannot satisfy — the caller owns the
    friendly ``parser.error``.  Resolution must run before ANY
    ``stage_configs[...]`` lookup: the default-timesteps lookup used to
    run first and turned unknown refs into raw KeyErrors (F10).
    """
    if stage_ref in stage_configs:
        return stage_ref
    if isinstance(stage_ref, str):
        from .stage_manifest import StageManifestError, load_stage_manifest

        try:
            entry = load_stage_manifest(species).by_id(stage_ref)
        except StageManifestError:
            return None
        if entry.legacy_number is not None and entry.legacy_number in stage_configs:
            return entry.legacy_number
    return None


def main(species_cfg):
    """Parse arguments and dispatch to train/curriculum/evaluate."""
    from .config import load_all_stages
    from .evaluation import DEFAULT_EVAL_SEED, evaluate
    from .train_base import train, train_curriculum

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stage_configs = load_all_stages(species_cfg.species)

    parser = argparse.ArgumentParser(description=f"Train {species_cfg.species.title()} with SB3 PPO")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # -- train ---------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument(
        "--stage",
        type=_parse_stage_ref,
        default=1,
        help=(
            f"Curriculum stage: a legacy number ({species_cfg.stage_descriptions}) "
            "or a semantic stage id from the species' manifest (e.g. 'recovery')"
        ),
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: the stage's curriculum.timesteps from its TOML config)",
    )
    train_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--load", type=str, default=None, help="Path to model to continue from")
    train_parser.add_argument(
        "--load-mode",
        choices=["resume_same_stage", "initialize_next_stage"],
        default="resume_same_stage",
        help=(
            "How the loaded checkpoint's task fingerprint is validated: resume_same_stage "
            "requires an exact task match; initialize_next_stage records the boundary as "
            "lineage (use it to warm-start a new stage, e.g. --stage recovery from a "
            "stance checkpoint)"
        ),
    )
    train_parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="Explicitly allow an untagged pre-contract model/VecNormalize artifact",
    )
    train_parser.add_argument(
        "--allow-fresh-vecnorm",
        action="store_true",
        help="Explicitly allow --load to proceed when its VecNormalize sidecar is missing "
        "(the loaded policy then trains under fresh normalization statistics — normally a "
        "fatal mistake, so this fails closed by default)",
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--eval-freq", type=int, default=50000, help="Evaluation frequency")
    train_parser.add_argument(
        "--save-freq",
        type=int,
        default=100000,
        help="Checkpoint frequency (default 100k: a lost Colab session then discards at most "
        "~30 min of training; retention keeps the newest 5 step-points either way)",
    )
    train_parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    train_parser.add_argument("--subproc", action="store_true", help="Use subprocess vectorization")
    train_parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbose level: 0=eval only, 1=progress bar (default), 2=debug",
    )
    train_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm",
    )
    train_parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    train_parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Override config values, e.g. ppo.learning_rate=1e-4",
    )
    train_parser.add_argument("--output-dir", type=str, default=None, help="Base output directory")
    train_parser.add_argument(
        "--post-eval-episodes",
        type=_non_negative_int,
        default=None,
        help="Episodes per post-training evaluation panel written to metrics.json (default: 50 for the "
        "quality panel and 30 for the velocity/success panel; 0 skips both panels — for smoke and CI "
        "runs, where they otherwise take roughly half the wall time)",
    )

    # -- curriculum ----------------------------------------------------
    cur_parser = subparsers.add_parser(
        "curriculum",
        help="Run the automated end-to-end curriculum over the species' advancing stages in manifest "
        "order (a non-advancing pilot stage such as trex 'recovery' is skipped with a log line; "
        "train it on its own with `train --stage recovery`)",
    )
    cur_parser.add_argument("--n-envs", type=int, default=4)
    cur_parser.add_argument("--seed", type=int, default=42)
    cur_parser.add_argument("--eval-freq", type=int, default=50000)
    cur_parser.add_argument("--save-freq", type=int, default=100000)
    cur_parser.add_argument("--log-dir", type=str, default=None)
    cur_parser.add_argument("--subproc", action="store_true")
    cur_parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    cur_parser.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    cur_parser.add_argument("--wandb", action="store_true")
    cur_parser.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE")
    cur_parser.add_argument(
        "--allow-fresh-vecnorm",
        action="store_true",
        help="Explicitly allow a stage handoff to proceed when its VecNormalize sidecar is missing",
    )
    cur_parser.add_argument("--output-dir", type=str, default=None)
    cur_parser.add_argument("--gcs-bucket", type=str, default=None)
    cur_parser.add_argument("--gcs-project", type=str, default=None)

    # -- eval ----------------------------------------------------------
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument("model_path", type=str, help="Path to trained model")
    eval_parser.add_argument(
        "--stage",
        type=_parse_stage_ref,
        default=None,
        help=(
            "Curriculum stage (auto-detected if omitted): a legacy number "
            f"({species_cfg.stage_descriptions}) or a semantic stage id from "
            "the species' manifest (e.g. 'recovery')"
        ),
    )
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    eval_parser.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    eval_parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="Explicitly allow untagged pre-contract evaluation artifacts",
    )
    eval_parser.add_argument(
        "--allow-unnormalized",
        action="store_true",
        help="Explicitly allow evaluation to proceed when the model's VecNormalize sidecar is missing "
        "(the policy is then scored on raw observations — a different policy — so this fails "
        "closed by default)",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Evaluation environment seed (default: a fixed constant independent of any training "
        "seed, so evaluations of different checkpoints are paired)",
    )

    # -- dispatch ------------------------------------------------------
    args = parser.parse_args()

    # SAC benefits from more envs (CPU-bound MuJoCo + off-policy replay).
    # Bump n_envs from default 4→8 when using SAC, unless user overrode it.
    _SAC_DEFAULT_N_ENVS = 8

    if args.command == "train" or args.command is None:
        if args.command is None:
            args.stage = 1
            args.timesteps = None
            args.n_envs = 4
            args.load = None
            args.allow_legacy_plant = False
            args.allow_fresh_vecnorm = False
            args.seed = 42
            args.eval_freq = 50000
            args.save_freq = 100000
            args.log_dir = None
            args.subproc = False
            args.verbose = 1
            args.algorithm = "ppo"
            args.wandb = False
            args.override = None
            args.output_dir = None
            args.post_eval_episodes = None

        if args.algorithm == "sac" and args.n_envs == 4:
            args.n_envs = _SAC_DEFAULT_N_ENVS
            logger.info("SAC: defaulting to %d parallel envs (override with --n-envs)", _SAC_DEFAULT_N_ENVS)

        _apply_overrides(stage_configs, args.override, species_cfg.species)

        stage_ref = _resolve_stage_ref(args.stage, stage_configs, species_cfg.species)
        if stage_ref is None:
            parser.error(f"unknown stage {args.stage!r} for this species; available: {sorted(map(str, stage_configs))}")

        # Resolve after overrides so --override curriculum.timesteps=... wins
        if args.timesteps is None:
            stage_budget = stage_configs[stage_ref].get("curriculum_kwargs", {}).get("timesteps", 500_000)
            remaining = _default_resume_timesteps(
                args.load, getattr(args, "load_mode", "resume_same_stage"), stage_budget
            )
            if remaining is not None:
                args.timesteps = remaining
                logger.info(
                    "No --timesteps given: resuming a periodic checkpoint, so training the "
                    "REMAINING stage budget (%s of %s). Pass --timesteps to override.",
                    f"{remaining:,}",
                    f"{stage_budget:,}",
                )
                if remaining == 0:
                    logger.warning(
                        "Stage budget already exhausted by the loaded checkpoint; no further "
                        "training will run (artifacts and the post-training evaluation still will)."
                    )
            else:
                args.timesteps = stage_budget
                logger.info(
                    "No --timesteps given: using stage %s config value (%s)",
                    args.stage,
                    f"{args.timesteps:,}",
                )

        train(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            stage=stage_ref,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            load_path=args.load,
            task_load_mode=getattr(args, "load_mode", "resume_same_stage"),
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            use_subproc=args.subproc,
            verbose=args.verbose,
            algorithm=args.algorithm,
            use_wandb=args.wandb,
            output_dir=args.output_dir,
            allow_legacy_plant=args.allow_legacy_plant,
            allow_fresh_vecnorm=getattr(args, "allow_fresh_vecnorm", False),
            post_eval_episodes=args.post_eval_episodes,
        )

    elif args.command == "curriculum":
        if args.algorithm == "sac" and args.n_envs == 4:
            args.n_envs = _SAC_DEFAULT_N_ENVS
            logger.info("SAC: defaulting to %d parallel envs (override with --n-envs)", _SAC_DEFAULT_N_ENVS)

        _apply_overrides(stage_configs, args.override, species_cfg.species)
        train_curriculum(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            n_envs=args.n_envs,
            seed=args.seed,
            allow_fresh_vecnorm=getattr(args, "allow_fresh_vecnorm", False),
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            log_dir=args.log_dir,
            use_subproc=args.subproc,
            verbose=args.verbose,
            algorithm=args.algorithm,
            use_wandb=args.wandb,
            output_dir=args.output_dir,
            gcs_bucket=args.gcs_bucket,
            gcs_project=args.gcs_project,
        )

    elif args.command == "eval":
        # A misdetected checkpoint (a recovery model in an unrecognized
        # layout) must be overridable, so eval --stage accepts the full
        # manifest vocabulary — not int choices=[1, 2, 3] (F11).
        eval_stage = args.stage
        if eval_stage is not None:
            eval_stage = _resolve_stage_ref(args.stage, stage_configs, species_cfg.species)
            if eval_stage is None:
                parser.error(
                    f"unknown stage {args.stage!r} for this species; available: {sorted(map(str, stage_configs))}"
                )
        evaluate(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render,
            stage=eval_stage,
            algorithm=args.algorithm,
            allow_legacy_plant=args.allow_legacy_plant,
            allow_unnormalized=args.allow_unnormalized,
            seed=args.seed,
        )
