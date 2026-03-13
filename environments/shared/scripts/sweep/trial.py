"""Trial execution — runs a single training trial with HPT-injected hyperparameters."""

from __future__ import annotations

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
    # ── Log system info for debugging failed Vertex AI trials ──────────
    trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", "local")
    logger.info("=" * 60)
    logger.info(
        "TRIAL START  |  species=%s  stage=%d  algorithm=%s  trial_id=%s",
        args.species,
        args.stage,
        args.algorithm,
        trial_id,
    )
    logger.info("  timesteps=%s  n_envs=%d  seed=%d", f"{args.timesteps:,}", args.n_envs, args.seed)
    try:
        import multiprocessing

        logger.info("  CPU cores: %d", multiprocessing.cpu_count())
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("  GPU: %s (CUDA %s)", torch.cuda.get_device_name(0), torch.version.cuda)
            logger.info("  GPU memory: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9)
        else:
            logger.info("  GPU: none (CPU-only training)")
    except ImportError:
        logger.info("  GPU: torch not installed (CPU-only training)")
    logger.info("=" * 60)

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

    # Use SubprocVecEnv when running multiple envs on Vertex AI to
    # exploit multi-core machines.  DummyVecEnv runs envs sequentially
    # which wastes CPU on n1-standard-8 and above.
    use_subproc = args.n_envs > 1

    train(
        species_cfg=SPECIES_CONFIG,
        stage_configs=STAGE_CONFIGS,
        stage=args.stage,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        load_path=args.load,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_subproc=use_subproc,
        verbose=0,
        algorithm=args.algorithm,
        use_wandb=args.wandb,
        output_dir=output_dir,
        use_tensorboard=not args.no_tensorboard,
    )

    # Generate stage summary and replay videos (same artifacts the notebook produces)
    if output_dir is not None:
        from environments.shared.reporting import generate_stage_artifacts

        generate_stage_artifacts(
            species_cfg=SPECIES_CONFIG,
            stage_config=STAGE_CONFIGS[args.stage],
            stage=args.stage,
            algorithm=args.algorithm,
            stage_dir=output_dir,
            seed=args.seed,
            timesteps=args.timesteps,
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
