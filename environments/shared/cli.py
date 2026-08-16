"""CLI entry point for species training scripts.

Extracted from ``train_base.py`` for maintainability.  Provides the
``main()`` function used by each species' ``train_sb3.py`` wrapper, along
with argument parsing and config-override helpers.
"""

import argparse
import logging

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


def _apply_overrides(configs: dict, overrides: list | None) -> None:
    """Apply dot-notation ``key=value`` overrides to stage configs.

    Two formats are supported:

    - ``section.key=value``   -- applies to **all** stages
    - ``N.section.key=value`` -- applies to stage *N* only
    """
    if not overrides:
        return
    for item in overrides:
        key, _, raw_value = item.partition("=")
        value = _cast_value(raw_value)
        parts = key.split(".")
        if len(parts) == 3 and parts[0].isdigit():
            stage_num, section, param = int(parts[0]), parts[1], parts[2]
            kwargs_key = "env_kwargs" if section == "env" else f"{section}_kwargs"
            if stage_num in configs:
                configs[stage_num][kwargs_key][param] = value
                logger.info(
                    "Stage %d override: %s.%s = %r",
                    stage_num,
                    section,
                    param,
                    value,
                )
        else:
            section, _, param = key.partition(".")
            kwargs_key = "env_kwargs" if section == "env" else f"{section}_kwargs"
            for stage_config in configs.values():
                stage_config[kwargs_key][param] = value
            logger.info("Override applied: %s.%s = %r", section, param, value)


def _parse_stage_ref(value: str) -> "int | str":
    """argparse type for --stage: digits mean a legacy stage number (their
    historical meaning, per the stage manifest), anything else a semantic
    stage ID such as "recovery"."""
    return int(value) if value.isdigit() else value


def main(species_cfg):
    """Parse arguments and dispatch to train/curriculum/evaluate."""
    from .config import load_all_stages
    from .evaluation import evaluate
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
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--eval-freq", type=int, default=50000, help="Evaluation frequency")
    train_parser.add_argument("--save-freq", type=int, default=500000, help="Checkpoint frequency")
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

    # -- curriculum ----------------------------------------------------
    cur_parser = subparsers.add_parser("curriculum", help="Run automated end-to-end curriculum (stages 1-3)")
    cur_parser.add_argument("--n-envs", type=int, default=4)
    cur_parser.add_argument("--seed", type=int, default=42)
    cur_parser.add_argument("--eval-freq", type=int, default=50000)
    cur_parser.add_argument("--save-freq", type=int, default=500000)
    cur_parser.add_argument("--log-dir", type=str, default=None)
    cur_parser.add_argument("--subproc", action="store_true")
    cur_parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    cur_parser.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    cur_parser.add_argument("--wandb", action="store_true")
    cur_parser.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE")
    cur_parser.add_argument("--output-dir", type=str, default=None)
    cur_parser.add_argument("--gcs-bucket", type=str, default=None)
    cur_parser.add_argument("--gcs-project", type=str, default=None)

    # -- eval ----------------------------------------------------------
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument("model_path", type=str, help="Path to trained model")
    eval_parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Curriculum stage (auto-detected if omitted)",
    )
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    eval_parser.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    eval_parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="Explicitly allow untagged pre-contract evaluation artifacts",
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
            args.seed = 42
            args.eval_freq = 50000
            args.save_freq = 500000
            args.log_dir = None
            args.subproc = False
            args.verbose = 1
            args.algorithm = "ppo"
            args.wandb = False
            args.override = None
            args.output_dir = None

        if args.algorithm == "sac" and args.n_envs == 4:
            args.n_envs = _SAC_DEFAULT_N_ENVS
            logger.info("SAC: defaulting to %d parallel envs (override with --n-envs)", _SAC_DEFAULT_N_ENVS)

        _apply_overrides(stage_configs, args.override)

        # Resolve after overrides so --override curriculum.timesteps=... wins
        if args.timesteps is None:
            args.timesteps = stage_configs[args.stage].get("curriculum_kwargs", {}).get("timesteps", 500_000)
            logger.info(
                "No --timesteps given: using stage %d config value (%s)",
                args.stage,
                f"{args.timesteps:,}",
            )

        stage_ref = args.stage
        if stage_ref not in stage_configs and isinstance(stage_ref, str):
            # A semantic id naming a LEGACY stage resolves to its historical
            # number, so --stage locomotion works wherever --stage 2 does.
            from .stage_manifest import StageManifestError, load_stage_manifest

            try:
                entry = load_stage_manifest(species_cfg.species).by_id(stage_ref)
            except StageManifestError:
                entry = None
            if entry is not None and entry.legacy_number is not None:
                stage_ref = entry.legacy_number
        if stage_ref not in stage_configs:
            parser.error(f"unknown stage {args.stage!r} for this species; available: {sorted(map(str, stage_configs))}")
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
        )

    elif args.command == "curriculum":
        if args.algorithm == "sac" and args.n_envs == 4:
            args.n_envs = _SAC_DEFAULT_N_ENVS
            logger.info("SAC: defaulting to %d parallel envs (override with --n-envs)", _SAC_DEFAULT_N_ENVS)

        _apply_overrides(stage_configs, args.override)
        train_curriculum(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            n_envs=args.n_envs,
            seed=args.seed,
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
        evaluate(
            species_cfg=species_cfg,
            stage_configs=stage_configs,
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render,
            stage=args.stage,
            algorithm=args.algorithm,
            allow_legacy_plant=args.allow_legacy_plant,
        )
