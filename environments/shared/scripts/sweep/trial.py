"""Trial execution — runs a single training trial with HPT-injected hyperparameters."""

import argparse
import logging
import os
import sys
from typing import Any

from .constants import NET_ARCH_PRESETS

logger = logging.getLogger(__name__)


def _hpt_arg_to_override(key: str, value: str) -> str:
    """Convert a Vertex AI HPT arg name to ``--override`` dot notation.

    Examples::

        "ppo_learning_rate", "0.0003"          → "ppo.learning_rate=0.0003"
        "env_alive_bonus",   "2.0"             → "env.alive_bonus=2.0"
        "curriculum_warmup_timesteps", "50000"  → "curriculum.warmup_timesteps=50000"
    """
    for prefix in ("ppo", "sac", "env", "curriculum"):
        if key.startswith(prefix + "_"):
            param = key[len(prefix) + 1 :]
            return f"{prefix}.{param}={value}"
    # Unrecognised prefix — pass through as-is (best-effort)
    return f"{key}={value}"


def _parse_hpt_extra_args(extra_args: list[str]) -> list[str]:
    """Parse HPT-injected extra args into override strings.

    Vertex AI HPT injects hyperparameters as either:
      ``--param_id=value``   (equals sign format)
      ``--param_id value``   (space-separated format)

    Returns a list of override strings like ``["ppo.learning_rate=0.0003", ...]``.
    """
    overrides: list[str] = []
    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if token.startswith("--"):
            # Handle --key=value format (used by Vertex AI HPT)
            if "=" in token:
                key, value = token[2:].split("=", 1)
                i += 1
            # Handle --key value format (space-separated)
            elif i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                key = token[2:]
                value = extra_args[i + 1]
                i += 2
            else:
                # Boolean flag — skip (shouldn't appear for HPT numeric params)
                i += 1
                continue
            overrides.append(_hpt_arg_to_override(key, value))
        else:
            i += 1
    return overrides


def run_trial(args: argparse.Namespace, extra_args: list[str]) -> None:
    """Run a single training trial.

    ``extra_args`` contains the hyperparameter values injected by Vertex AI
    HPT (e.g. ``['--ppo_learning_rate', '0.0003', '--ppo_ent_coef', '0.01']``
    or ``['--ppo_learning_rate=0.0003', '--ppo_ent_coef=0.01']``).
    They are converted to ``--override`` format before calling ``train()``.

    Each Vertex AI worker gets a unique ``CLOUD_ML_TRIAL_ID`` environment
    variable.  When ``--output-dir`` is set, this ID is appended as a
    subdirectory so that every trial's checkpoint is written to its own
    path — which is how ``launch-all`` identifies the best trial later.
    """
    overrides = _parse_hpt_extra_args(extra_args)

    # Extract net_arch overrides — these need special handling because
    # net_arch lives inside a nested ``policy_kwargs`` dict rather than at
    # the top level of ``ppo_kwargs`` / ``sac_kwargs``.
    net_arch_preset: str | None = None
    net_arch_algo: str | None = None
    remaining_overrides: list[str] = []
    for ovr in overrides:
        key, _, value = ovr.partition("=")
        section, _, param = key.partition(".")
        if param == "net_arch" and section in ("ppo", "sac"):
            if value not in NET_ARCH_PRESETS:
                logger.error(
                    "Unknown net_arch preset %r. Valid presets: %s",
                    value,
                    list(NET_ARCH_PRESETS.keys()),
                )
                sys.exit(1)
            net_arch_preset = value
            net_arch_algo = section
            logger.info("Net-arch preset: %s → %s", value, NET_ARCH_PRESETS[value])
        else:
            remaining_overrides.append(ovr)
    overrides = remaining_overrides

    if overrides:
        logger.info("Trial overrides from HPT: %s", overrides)
    else:
        logger.info("No HPT overrides received — using TOML defaults")

    # Make output unique per trial so each worker keeps its own checkpoint.
    # Vertex AI sets CLOUD_ML_TRIAL_ID on every worker container.
    trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", "local")
    output_dir = f"{args.output_dir}/{trial_id}" if args.output_dir else None
    if output_dir:
        logger.info("Trial output directory: %s", output_dir)

    # Import shared training infrastructure
    from environments.shared.config import load_all_stages
    from environments.shared.train_base import _apply_overrides, train

    # Load species config and stage configs
    if args.species == "velociraptor":
        from environments.velociraptor.scripts.train_sb3 import SPECIES_CONFIG
    elif args.species == "brachiosaurus":
        from environments.brachiosaurus.scripts.train_sb3 import SPECIES_CONFIG
    elif args.species == "trex":
        from environments.trex.scripts.train_sb3 import SPECIES_CONFIG
    else:
        logger.error("Unknown species: %s", args.species)
        sys.exit(1)

    STAGE_CONFIGS = load_all_stages(args.species)

    if overrides:
        _apply_overrides(STAGE_CONFIGS, overrides)

    # Apply net_arch preset to the nested policy_kwargs dict
    if net_arch_preset is not None:
        arch = NET_ARCH_PRESETS[net_arch_preset]
        algo_key = f"{net_arch_algo}_kwargs"
        for stage_config in STAGE_CONFIGS.values():
            stage_config[algo_key].setdefault("policy_kwargs", {})["net_arch"] = arch
        logger.info("Applied net_arch=%s (%s) to all stages", net_arch_preset, arch)

    model = train(
        species_cfg=SPECIES_CONFIG,
        stage_configs=STAGE_CONFIGS,
        stage=args.stage,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        load_path=args.load,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_subproc=False,
        verbose=0,
        algorithm=args.algorithm,
        use_wandb=args.wandb,
        output_dir=output_dir,
    )

    # Generate stage summary and replay videos (same artifacts the notebook produces)
    if output_dir is not None:
        _generate_trial_artifacts(
            model=model,
            species_cfg=SPECIES_CONFIG,
            stage_configs=STAGE_CONFIGS,
            stage=args.stage,
            algorithm=args.algorithm,
            output_dir=output_dir,
            timesteps=args.timesteps,
            seed=args.seed,
        )


def _generate_trial_artifacts(
    model,
    species_cfg,
    stage_configs,
    stage: int,
    algorithm: str,
    output_dir: str,
    timesteps: int,
    seed: int,
) -> None:
    """Generate stage summary and replay videos for a completed trial.

    Mirrors what the training notebook produces so sweep trials leave the
    same artifacts: ``stage_summary.txt`` and best/final ``.mp4`` videos.
    """
    from pathlib import Path

    import numpy as _np

    from environments.shared.evaluation import record_stage_video
    from environments.shared.reporting import write_stage_summary
    from environments.shared.train_base import _ensure_sb3

    sb3 = _ensure_sb3()
    log_path = Path(output_dir)
    model_dir = log_path / "models"
    config = stage_configs[stage]
    species = species_cfg.species

    # ── Build a results dict from on-disk eval data ─────────────────────
    eval_npz = log_path / "evaluations.npz"
    mean_reward = 0.0
    std_reward = 0.0
    mean_length = 0.0
    std_length = 0.0
    best_eval_reward: float | str = ""
    best_eval_std: float | str = ""
    best_eval_length: float | str = ""
    best_eval_std_length: float | str = ""
    best_eval_timestep: int | str = ""

    if eval_npz.exists():
        eval_data = _np.load(str(eval_npz))
        eval_rewards = eval_data["results"]
        eval_lengths = eval_data["ep_lengths"]
        eval_timesteps = eval_data["timesteps"]

        mean_per_eval = eval_rewards.mean(axis=1)
        best_idx = int(mean_per_eval.argmax())

        best_eval_reward = round(float(mean_per_eval[best_idx]), 2)
        best_eval_std = round(float(eval_rewards[best_idx].std()), 2)
        best_eval_length = round(float(eval_lengths[best_idx].mean()), 1)
        best_eval_std_length = round(float(eval_lengths[best_idx].std()), 1)
        best_eval_timestep = int(eval_timesteps[best_idx])

        # Use last eval as "final" metrics
        mean_reward = float(mean_per_eval[-1])
        std_reward = float(eval_rewards[-1].std())
        mean_length = float(eval_lengths[-1].mean())
        std_length = float(eval_lengths[-1].std())

    # Read duration from metrics.json if available
    metrics_path = log_path / "metrics.json"
    duration = 0.0
    if metrics_path.exists():
        import json as _json

        metrics = _json.loads(metrics_path.read_text())
        duration = metrics.get("training_duration_seconds", 0.0)

    best_model_path = model_dir / "best_model"
    final_path = model_dir / f"stage{stage}_final"
    vecnorm_path = str(model_dir / "best_model_vecnorm.pkl")
    final_vecnorm_path = str(final_path) + "_vecnorm.pkl"

    sim_dt = config.get("env_kwargs", {}).get("sim_dt", 0.01)

    stage_results = {
        "stage": stage,
        "name": config["name"],
        "description": config["description"],
        "timesteps": timesteps,
        "duration_seconds": duration,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "mean_forward_vel": 0.0,
        "std_forward_vel": 0.0,
        "mean_success_rate": 0.0,
        "best_eval_reward": best_eval_reward,
        "best_eval_std": best_eval_std,
        "best_eval_length": best_eval_length,
        "best_eval_std_length": best_eval_std_length,
        "best_eval_timestep": best_eval_timestep,
        "sim_dt": sim_dt,
        "model_path": str(best_model_path),
        "vecnorm_path": vecnorm_path,
    }

    write_stage_summary(log_path, stage_results, species, algorithm)
    logger.info("Stage summary written to: %s", log_path / "stage_summary.txt")

    # ── Record replay videos for best and final models ──────────────────
    env_kwargs = config["env_kwargs"].copy()
    alg_cls = sb3["SAC"] if algorithm == "sac" else sb3["PPO"]

    if (model_dir / "best_model.zip").exists():
        best_model = alg_cls.load(str(best_model_path))
        record_stage_video(
            best_model,
            env_class=species_cfg.env_class,
            env_kwargs=env_kwargs,
            stage=stage,
            stage_dir=log_path,
            species=species,
            algorithm=algorithm,
            seed=seed,
            vecnorm_path=vecnorm_path,
            label="best",
        )

    if (Path(str(final_path) + ".zip")).exists():
        final_model = alg_cls.load(str(final_path))
        record_stage_video(
            final_model,
            env_class=species_cfg.env_class,
            env_kwargs=env_kwargs,
            stage=stage,
            stage_dir=log_path,
            species=species,
            algorithm=algorithm,
            seed=seed,
            vecnorm_path=final_vecnorm_path,
            label="final",
        )


def _build_parameter_spec(search_space: dict, hpt_module: Any) -> dict:
    """Convert a search-space dict to Vertex AI parameter spec objects."""
    hpt = hpt_module
    parameter_spec: dict = {}
    for param_id, spec in search_space.items():
        kind = spec.get("type", "double")
        if kind == "double":
            parameter_spec[param_id] = hpt.DoubleParameterSpec(
                min=float(spec["min"]),
                max=float(spec["max"]),
                scale=spec.get("scale", "linear"),
            )
        elif kind == "discrete":
            parameter_spec[param_id] = hpt.DiscreteParameterSpec(
                values=[float(v) for v in spec["values"]], scale="linear"
            )
        elif kind == "categorical":
            parameter_spec[param_id] = hpt.CategoricalParameterSpec(values=spec["values"])
        else:
            logger.warning("Unknown parameter type %r for %s — skipping", kind, param_id)
    return parameter_spec
