#!/usr/bin/env python3
"""Hyperparameter sweep tool for Mesozoic Labs.

Three modes:

  **launch**      — Submit a Vertex AI Hyperparameter Tuning job for **one stage**.
                    Vertex AI uses Bayesian optimisation to run N parallel trials
                    and automatically find the best hyperparameter combination.
                    Returns immediately (non-blocking).
                    Requires: ``pip install google-cloud-aiplatform``

  **launch-all**  — Submit Stage 1, Stage 2, and Stage 3 HPT jobs sequentially in a
                    **single command**.  Each stage waits for the previous to finish,
                    automatically picks the best trial's checkpoint, and passes it as
                    the warm-start model for the next stage.

  **trial**       — Entry point used by each Vertex AI HPT trial worker running
                    inside Docker. Receives hyperparameter values injected by the
                    HPT service as ``--ppo_learning_rate X --ppo_ent_coef Y``
                    style args, converts them to ``--override`` dot notation, and
                    runs the training script.  Output is written to a per-trial
                    subdirectory (using the ``CLOUD_ML_TRIAL_ID`` env var) so that
                    each worker's checkpoint is preserved separately.

Usage examples::

    # ── single command: sweep all three stages end-to-end ───────────────────
    python environments/shared/scripts/sweep.py launch-all \\
        --species velociraptor --algorithm ppo \\
        --project YOUR_GCP_PROJECT --bucket YOUR_GCS_BUCKET \\
        --image us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest \\
        --trials 20 --parallel 5 \\
        --timesteps-stage1 500000 --timesteps-stage2 1000000 --timesteps-stage3 1500000

    # ── launch a single-stage PPO sweep (from your local machine) ───────────
    python environments/shared/scripts/sweep.py launch \\
        --species velociraptor --stage 1 --algorithm ppo \\
        --project YOUR_GCP_PROJECT --bucket YOUR_GCS_BUCKET \\
        --image us-central1-docker.pkg.dev/YOUR_PROJECT/mesozoic-labs/trainer:latest \\
        --trials 20 --parallel 5 --timesteps 500000

    # ── run a single trial locally (test before launching) ──────────────────
    python environments/shared/scripts/sweep.py trial \\
        --species velociraptor --stage 1 --algorithm ppo \\
        --timesteps 10000 --n-envs 1 \\
        --ppo_learning_rate 1e-4 --ppo_ent_coef 0.02 --ppo_batch_size 128

    # ── customise the search space with JSON ────────────────────────────────
    python environments/shared/scripts/sweep.py launch \\
        --species trex --stage 1 --algorithm ppo \\
        --project YOUR_PROJECT --bucket YOUR_BUCKET --image IMAGE_URI \\
        --trials 30 --parallel 5 \\
        --search-space '{
            "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
            "ppo_ent_coef":      {"type": "double", "min": 0.001, "max": 0.05, "scale": "log"},
            "ppo_batch_size":    {"type": "discrete", "values": [64, 128, 256]}
        }'

Metric optimised: ``best_mean_reward`` (maximised).
This is reported by each trial's ``train()`` call via ``cloudml-hypertune``.

Search space parameter naming convention:
  ``{algo}_{param}``  e.g. ``ppo_learning_rate``, ``sac_batch_size``
  ``env_{param}``     e.g. ``env_alive_bonus``, ``env_forward_vel_weight``

These are auto-converted to ``--override`` dot notation inside ``trial`` mode:
  ``ppo_learning_rate`` → ``ppo.learning_rate=<value>``
  ``env_alive_bonus``   → ``env.alive_bonus=<value>``
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add repo root to Python path so environments.* imports work
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SweepStageError(Exception):
    """Raised when a sweep stage fails and cannot proceed."""


class _SweepJobFailed(SweepStageError):
    """A submitted HPT job failed but may contain partial trial results.

    Attributes:
        hpt_job: The Vertex AI ``HyperparameterTuningJob`` object, which may
            still expose ``.trials`` for completed trial data even when the
            overall job failed.
    """

    def __init__(self, message: str, hpt_job: Any = None):
        super().__init__(message)
        self.hpt_job = hpt_job


# ── Default search spaces ────────────────────────────────────────────────────
# Each entry: parameter_id -> {"type": ..., ...}
# parameter_id uses underscore notation to match Vertex AI HPT arg injection.
# The trial entry point converts them to dot notation for --override.

_DEFAULT_PPO_SEARCH_SPACE = {
    "ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "ppo_ent_coef": {"type": "double", "min": 1e-4, "max": 0.05, "scale": "log"},
    "ppo_batch_size": {"type": "discrete", "values": [64, 128, 256, 512]},
    "ppo_gamma": {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
    "ppo_n_steps": {"type": "discrete", "values": [1024, 2048, 4096]},
}

_DEFAULT_SAC_SEARCH_SPACE = {
    "sac_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"},
    "sac_batch_size": {"type": "discrete", "values": [128, 256, 512]},
    "sac_gamma": {"type": "double", "min": 0.97, "max": 0.999, "scale": "linear"},
}

_DEFAULT_SEARCH_SPACES = {
    "ppo": _DEFAULT_PPO_SEARCH_SPACE,
    "sac": _DEFAULT_SAC_SEARCH_SPACE,
}

# ── Net-arch presets ─────────────────────────────────────────────────────────
# Categorical values for the ``ppo_net_arch`` / ``sac_net_arch`` sweep param.
# Each preset maps to a ``policy_kwargs.net_arch`` list for SB3.
NET_ARCH_PRESETS: dict[str, list[int]] = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [512, 512],
    "deep": [256, 256, 256],
    "tapered": [512, 256],
    "deep_tapered": [512, 512, 256],
}


def _load_search_space_file(path: str) -> dict:
    """Load a search space definition from a JSON file.

    The file can be either:

    * **Flat** — a single dict of parameter specs applied to all stages::

        {"ppo_learning_rate": {"type": "double", "min": 1e-5, "max": 3e-4, "scale": "log"}, ...}

    * **Per-stage** — top-level keys ``"stage1"``, ``"stage2"``, ``"stage3"``
      each mapping to a stage-specific search space dict::

        {
          "stage1": {"ppo_learning_rate": ..., "env_alive_bonus": ...},
          "stage2": {"ppo_learning_rate": ...},
          "stage3": {"ppo_learning_rate": ...}
        }

    Returns the parsed dict as-is; the caller detects the format by checking
    for ``"stage1"`` / ``"stage2"`` / ``"stage3"`` keys.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.error("Search space file not found: %s", file_path)
        sys.exit(1)
    try:
        result: dict = json.loads(file_path.read_text())
        return result
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in search space file %s: %s", file_path, exc)
        sys.exit(1)


def _resolve_search_space(
    search_space_json: str | None,
    search_space_file: str | None,
    algorithm: str,
) -> dict:
    """Resolve the search space from CLI args.

    Priority: ``--search-space`` (inline JSON) > ``--search-space-file`` >
    algorithm default.

    Returns either a flat search-space dict (all stages share the same space)
    or a dict with ``"stage1"``/``"stage2"``/``"stage3"`` keys for per-stage
    spaces.
    """
    if search_space_json:
        try:
            result: dict = json.loads(search_space_json)
            logger.info("Search space resolved from inline --search-space JSON")
            return result
        except json.JSONDecodeError as exc:
            logger.error("Invalid --search-space JSON: %s", exc)
            sys.exit(1)

    if search_space_file:
        logger.info("Search space resolved from file: %s", search_space_file)
        return _load_search_space_file(search_space_file)

    logger.info("Using default %s search space", algorithm)
    return _DEFAULT_SEARCH_SPACES.get(algorithm, _DEFAULT_PPO_SEARCH_SPACE)


def _is_per_stage(config: dict) -> bool:
    """Return True if *config* uses per-stage keys (stage1/stage2/stage3)."""
    return "stage1" in config or "stage2" in config or "stage3" in config


def _split_stage_block(block: dict) -> tuple[dict, dict]:
    """Split a stage block into (search_space, settings).

    Within a stage block, any key whose value is a dict containing a
    ``"type"`` field is a search-space parameter.  All other keys
    (``trials``, ``timesteps``, ``parallel``, ``n_envs``) are job settings.
    """
    search_space: dict = {}
    settings: dict = {}
    for key, value in block.items():
        if isinstance(value, dict) and "type" in value:
            search_space[key] = value
        else:
            settings[key] = value
    return search_space, settings


def _search_space_for_stage(config: dict, stage: int) -> dict:
    """Extract the search space for a specific stage.

    If *config* has ``"stage1"``/``"stage2"``/``"stage3"`` keys, return
    the search-space parameters for the requested stage (job settings like
    ``trials`` and ``timesteps`` are filtered out).  Otherwise return
    *config* as-is (flat format — same space for all stages).
    """
    if _is_per_stage(config):
        key = f"stage{stage}"
        if key not in config:
            logger.error("Per-stage search space file is missing '%s' key", key)
            sys.exit(1)
        search_space, _ = _split_stage_block(config[key])
        return search_space
    return config


def _settings_for_stage(config: dict, stage: int) -> dict:
    """Extract job settings (trials, timesteps, parallel, n_envs) for a stage.

    Returns an empty dict for flat configs or stages without settings.
    """
    if _is_per_stage(config):
        key = f"stage{stage}"
        if key in config:
            _, settings = _split_stage_block(config[key])
            return settings
    return {}


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


def run_trial(args: argparse.Namespace, extra_args: list[str]) -> None:
    """Run a single training trial.

    ``extra_args`` contains the hyperparameter values injected by Vertex AI
    HPT (e.g. ``['--ppo_learning_rate', '0.0003', '--ppo_ent_coef', '0.01']``).
    They are converted to ``--override`` format before calling ``train()``.

    Each Vertex AI worker gets a unique ``CLOUD_ML_TRIAL_ID`` environment
    variable.  When ``--output-dir`` is set, this ID is appended as a
    subdirectory so that every trial's checkpoint is written to its own
    path — which is how ``launch-all`` identifies the best trial later.
    """
    # Convert HPT-style args (--ppo_learning_rate 0.0003) to override format
    overrides: list[str] = []
    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if token.startswith("--"):
            key = token[2:]  # strip leading "--"
            # Peek at the next token — is it a value or another flag?
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                value = extra_args[i + 1]
                i += 2
            else:
                # Boolean flag — skip (shouldn't appear for HPT numeric params)
                i += 1
                continue
            overrides.append(_hpt_arg_to_override(key, value))
        else:
            i += 1

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
        use_subproc=False,
        verbose=0,
        algorithm=args.algorithm,
        use_wandb=args.wandb,
        output_dir=output_dir,
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


def _collect_trial_results(hpt_job: Any, stage: int, stage_config: dict) -> list[dict]:
    """Extract per-trial hyperparameters and outcomes from a completed HPT job.

    Each returned dict contains:

    * ``trial_id`` — Vertex AI trial identifier
    * ``stage`` — curriculum stage number
    * one key per hyperparameter (e.g. ``ppo_learning_rate``)
    * ``best_mean_reward`` — final metric reported by the trial
    * ``best_mean_episode_length`` — episode length from the best eval
    * ``reward_threshold`` — ``min_avg_reward`` from the stage TOML config
    * ``ep_length_threshold`` — ``min_avg_episode_length`` from config
    * ``forward_vel_threshold`` — ``min_avg_forward_vel`` from config
    * ``success_rate_threshold`` — ``min_success_rate`` from config
    * ``stage_passed`` — ``True`` when all curriculum criteria are met
    """
    cur = stage_config.get("curriculum_kwargs", {})
    reward_threshold = cur.get("min_avg_reward")
    ep_length_threshold = cur.get("min_avg_episode_length")
    forward_vel_threshold = cur.get("min_avg_forward_vel")
    success_rate_threshold = cur.get("min_success_rate")

    rows: list[dict] = []
    for trial in hpt_job.trials:
        row: dict = {"trial_id": trial.id, "stage": stage}
        if hasattr(trial, "parameters") and trial.parameters:
            for param in trial.parameters:
                row[param.parameter_id] = param.value

        # Extract all reported metrics from the trial
        metrics: dict[str, float | None] = {}
        if trial.final_measurement and trial.final_measurement.metrics:
            for metric in trial.final_measurement.metrics:
                metrics[metric.metric_id] = metric.value

        best_reward = metrics.get("best_mean_reward")
        best_ep_length = metrics.get("best_mean_episode_length")
        last_reward = metrics.get("last_mean_reward")
        last_ep_length = metrics.get("last_mean_episode_length")

        row["best_mean_reward"] = best_reward
        row["best_mean_episode_length"] = best_ep_length
        row["last_mean_reward"] = last_reward
        row["last_mean_episode_length"] = last_ep_length
        row["reward_threshold"] = reward_threshold
        row["ep_length_threshold"] = ep_length_threshold
        row["forward_vel_threshold"] = forward_vel_threshold
        row["success_rate_threshold"] = success_rate_threshold

        # Check all curriculum criteria.  When no thresholds are defined
        # (e.g. Stage 3 hunting stages), the trial passes by default as
        # long as it produced a valid reward.
        passed = best_reward is not None
        fail_reasons: list[str] = []
        if best_reward is None:
            fail_reasons.append("no reward reported (trial may have crashed)")
        if reward_threshold is not None and (best_reward is None or best_reward < reward_threshold):
            passed = False
            fail_reasons.append(f"reward {best_reward} < threshold {reward_threshold}")
        if passed and ep_length_threshold is not None:
            if best_ep_length is None or best_ep_length < ep_length_threshold:
                passed = False
                fail_reasons.append(f"ep_length {best_ep_length} < threshold {ep_length_threshold}")
        if passed and forward_vel_threshold is not None:
            trial_fwd_vel = metrics.get("best_mean_forward_vel")
            if trial_fwd_vel is None or trial_fwd_vel < forward_vel_threshold:
                passed = False
                fail_reasons.append(f"forward_vel {trial_fwd_vel} < threshold {forward_vel_threshold}")
        if passed and success_rate_threshold is not None:
            trial_success_rate = metrics.get("best_mean_success_rate")
            if trial_success_rate is None or trial_success_rate < success_rate_threshold:
                passed = False
                fail_reasons.append(f"success_rate {trial_success_rate} < threshold {success_rate_threshold}")
        row["stage_passed"] = passed

        # Log per-trial diagnostic summary
        if passed:
            logger.info(
                "  Trial %s stage %d: PASSED (reward=%.2f)",
                trial.id,
                stage,
                best_reward,
            )
        else:
            logger.warning(
                "  Trial %s stage %d: FAILED — %s",
                trial.id,
                stage,
                "; ".join(fail_reasons),
            )

        rows.append(row)
    return rows


def write_results_csv(rows: list[dict], path: str | Path) -> Path:
    """Write sweep trial results to a CSV file.

    Each row records the trial ID, stage, all hyperparameter values,
    performance metrics, curriculum thresholds, and whether the trial
    met all stage advancement criteria.

    Args:
        rows: List of result dicts from :func:`_collect_trial_results`.
        path: Output CSV path (parent directories are created as needed).

    Returns:
        Path to the written CSV file.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        logger.warning("No trial rows to write — skipping CSV")
        return path

    fixed_cols = ["trial_id", "stage"]
    metric_cols = [
        "best_mean_reward",
        "best_mean_episode_length",
        "last_mean_reward",
        "last_mean_episode_length",
        "reward_threshold",
        "ep_length_threshold",
        "forward_vel_threshold",
        "success_rate_threshold",
        "stage_passed",
    ]
    # Collect all hyperparameter column names across all rows (union, sorted)
    hparam_cols: list[str] = sorted({k for row in rows for k in row if k not in fixed_cols + metric_cols})
    fieldnames = fixed_cols + hparam_cols + metric_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Sweep results written to: %s", path)
    return path


def plot_sweep_results(csv_path: str | Path, species: str, algorithm: str, save_dir: str | Path | None = None) -> None:
    """Generate visualisation graphs from sweep results CSV.

    Produces two PNG figures saved alongside the CSV (or to *save_dir*):

    **sweep_trial_metrics.png** — 2x2 grid:
      - [0,0] Best Mean Reward per trial (grouped by stage)
      - [0,1] Best Mean Episode Length per trial (grouped by stage)
      - [1,0] Best vs Last Mean Reward (training stability)
      - [1,1] Stage Pass/Fail summary (stacked bar)

    **sweep_hyperparameter_analysis.png** — Nx1 column:
      - One scatter plot per hyperparameter vs best_mean_reward (colour = stage)

    Args:
        csv_path: Path to the sweep results CSV.
        species: Species name for titles.
        algorithm: Algorithm name for titles.
        save_dir: Directory to save PNGs. Defaults to the CSV's parent directory.
    """
    import csv as _csv

    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend for headless environments
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib or numpy not installed — skipping sweep visualisations")
        return

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("Sweep CSV not found: %s — skipping visualisations", csv_path)
        return

    if save_dir is None:
        save_dir = csv_path.parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV into list of dicts
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("Sweep CSV is empty — skipping visualisations")
        return

    # Parse numeric values
    def _float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    stages = sorted({int(r["stage"]) for r in rows if r.get("stage")})
    stage_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    # ── Figure 1: Trial Metrics (2x2) ────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    title = f"{species.capitalize()} {algorithm.upper()} Sweep"
    fig1.suptitle(title, fontsize=14, fontweight="bold")

    for stage in stages:
        stage_rows = [r for r in rows if int(r["stage"]) == stage]
        trial_ids = [r["trial_id"] for r in stage_rows]
        label = f"Stage {stage}"
        color = stage_colors.get(stage, "#333333")

        # [0,0] Best Mean Reward
        rewards = [_float(r.get("best_mean_reward")) for r in stage_rows]
        valid = [(tid, rw) for tid, rw in zip(trial_ids, rewards) if rw is not None]
        if valid:
            tids, rws = zip(*valid)
            x = np.arange(len(tids))
            axes1[0, 0].bar(x, rws, color=color, alpha=0.7, label=label)
            axes1[0, 0].set_xticks(x)
            axes1[0, 0].set_xticklabels(tids, rotation=45, fontsize=7)
            # Draw threshold line if available
            threshold = _float(stage_rows[0].get("reward_threshold"))
            if threshold is not None:
                axes1[0, 0].axhline(y=threshold, color=color, linestyle="--", alpha=0.5)

        # [0,1] Best Mean Episode Length
        ep_lengths = [_float(r.get("best_mean_episode_length")) for r in stage_rows]
        valid_ep = [(tid, el) for tid, el in zip(trial_ids, ep_lengths) if el is not None]
        if valid_ep:
            tids_ep, els = zip(*valid_ep)
            x = np.arange(len(tids_ep))
            axes1[0, 1].bar(x, els, color=color, alpha=0.7, label=label)
            axes1[0, 1].set_xticks(x)
            axes1[0, 1].set_xticklabels(tids_ep, rotation=45, fontsize=7)
            ep_threshold = _float(stage_rows[0].get("ep_length_threshold"))
            if ep_threshold is not None:
                axes1[0, 1].axhline(y=ep_threshold, color=color, linestyle="--", alpha=0.5)

        # [1,0] Best vs Last Mean Reward (training stability)
        best_last = [(_float(r.get("best_mean_reward")), _float(r.get("last_mean_reward"))) for r in stage_rows]
        valid_bl = [(b, last) for b, last in best_last if b is not None and last is not None]
        if valid_bl:
            bests, lasts = zip(*valid_bl)
            axes1[1, 0].scatter(bests, lasts, color=color, alpha=0.7, label=label, edgecolors="white", s=50)

    axes1[0, 0].set_xlabel("Trial ID")
    axes1[0, 0].set_ylabel("Best Mean Reward")
    axes1[0, 0].set_title("Best Mean Reward per Trial")
    axes1[0, 0].legend()
    axes1[0, 0].grid(True, alpha=0.3)

    axes1[0, 1].set_xlabel("Trial ID")
    axes1[0, 1].set_ylabel("Best Mean Episode Length")
    axes1[0, 1].set_title("Best Mean Episode Length per Trial")
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)

    axes1[1, 0].set_xlabel("Best Mean Reward")
    axes1[1, 0].set_ylabel("Last Mean Reward")
    axes1[1, 0].set_title("Training Stability (Best vs Last Reward)")
    # Draw y=x reference line
    all_rewards = [_float(r.get("best_mean_reward")) for r in rows]
    all_rewards = [v for v in all_rewards if v is not None]
    if all_rewards:
        lo, hi = min(all_rewards), max(all_rewards)
        axes1[1, 0].plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y=x")
    axes1[1, 0].legend()
    axes1[1, 0].grid(True, alpha=0.3)

    # [1,1] Stage Pass/Fail Summary
    ax_pf = axes1[1, 1]
    pass_counts = []
    fail_counts = []
    stage_labels = []
    for stage in stages:
        stage_rows = [r for r in rows if int(r["stage"]) == stage]
        passed = sum(1 for r in stage_rows if str(r.get("stage_passed", "")).lower() == "true")
        failed = len(stage_rows) - passed
        pass_counts.append(passed)
        fail_counts.append(failed)
        stage_labels.append(f"Stage {stage}")

    x_pf = np.arange(len(stages))
    ax_pf.bar(x_pf, pass_counts, color="#2ca02c", alpha=0.8, label="Passed")
    ax_pf.bar(x_pf, fail_counts, bottom=pass_counts, color="#d62728", alpha=0.8, label="Failed")
    ax_pf.set_xticks(x_pf)
    ax_pf.set_xticklabels(stage_labels)
    ax_pf.set_ylabel("Number of Trials")
    ax_pf.set_title("Stage Pass/Fail Summary")
    ax_pf.legend()
    ax_pf.grid(True, alpha=0.3, axis="y")

    fig1.tight_layout()
    fig1_path = save_dir / "sweep_trial_metrics.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info("Sweep trial metrics graph saved to: %s", fig1_path)

    # ── Figure 2: Hyperparameter Analysis ─────────────────────────────────────
    # Identify hyperparameter columns (not fixed or metric columns)
    fixed_cols = {
        "trial_id",
        "stage",
        "best_mean_reward",
        "best_mean_episode_length",
        "last_mean_reward",
        "last_mean_episode_length",
        "reward_threshold",
        "ep_length_threshold",
        "forward_vel_threshold",
        "success_rate_threshold",
        "stage_passed",
    }
    hparam_cols = [k for k in rows[0].keys() if k not in fixed_cols]
    # Filter to columns with numeric, varying values
    numeric_hparams = []
    for col in hparam_cols:
        vals = [_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2 and len(set(vals)) > 1:
            numeric_hparams.append(col)

    if numeric_hparams:
        n_params = len(numeric_hparams)
        fig2, axes2 = plt.subplots(n_params, 1, figsize=(10, 4 * n_params), squeeze=False)
        fig2.suptitle(f"{title} — Hyperparameter vs Reward", fontsize=14, fontweight="bold")

        for idx, hparam in enumerate(numeric_hparams):
            ax = axes2[idx, 0]
            for stage in stages:
                stage_rows = [r for r in rows if int(r["stage"]) == stage]
                xs = [_float(r.get(hparam)) for r in stage_rows]
                ys = [_float(r.get("best_mean_reward")) for r in stage_rows]
                valid_hp = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
                if valid_hp:
                    hx, hy = zip(*valid_hp)
                    ax.scatter(
                        hx,
                        hy,
                        color=stage_colors.get(stage, "#333"),
                        alpha=0.7,
                        label=f"Stage {stage}",
                        edgecolors="white",
                        s=50,
                    )
            ax.set_xlabel(hparam)
            ax.set_ylabel("Best Mean Reward")
            ax.set_title(f"{hparam} vs Best Mean Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2_path = save_dir / "sweep_hyperparameter_analysis.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("Sweep hyperparameter analysis graph saved to: %s", fig2_path)
    else:
        logger.info("No varying numeric hyperparameters found — skipping hyperparameter analysis graph")


def _best_trial_model_path(stage_rows: list[dict], bucket: str, species: str, stage: int) -> tuple[str, dict]:
    """Return the GCS path of the best trial's best model and its result row.

    Each trial saves its best model (highest eval reward) to::

        /gcs/<bucket>/sweeps/<species>/stage<N>/<trial_id>/models/best_model.zip

    alongside a matched VecNormalize snapshot at ``best_model_vecnorm.pkl``.

    This function inspects the completed trial rows, identifies the trial with
    the highest ``best_mean_reward`` among those that passed the stage, and
    returns the best model checkpoint path so the next stage's sweep can
    warm-start from it.

    Returns:
        A tuple of ``(model_path, best_row)`` where ``best_row`` is the full
        result dict for the winning trial.  This allows callers to inspect
        hyperparameters (e.g. ``net_arch``) that must be propagated forward.
    """
    best_row: dict | None = None
    best_value = float("-inf")

    for row in stage_rows:
        if row.get("stage_passed"):
            best_reward = row.get("best_mean_reward")
            if best_reward is not None and best_reward > best_value:
                best_value = best_reward
                best_row = row

    if best_row is None:
        raise SweepStageError(
            f"No trials passed stage {stage} criteria. Re-run with adjusted thresholds or more trials."
        )

    best_trial_id = best_row["trial_id"]
    logger.info("Best passing trial for stage %d: id=%s  best_mean_reward=%.4f", stage, best_trial_id, best_value)

    # Use a pre-computed model_path if available (e.g. from a partial-resume
    # merge where trials come from different GCS output directories).
    if "model_path" in best_row:
        path = best_row["model_path"]
    else:
        path = f"/gcs/{bucket}/sweeps/{species}/stage{stage}/{best_trial_id}/models/best_model.zip"
    return path, best_row


def _best_trial_model_path_any(stage_rows: list[dict], bucket: str, species: str, stage: int) -> tuple[str, dict]:
    """Return the GCS path of the best trial's model, ignoring pass/fail status.

    Like :func:`_best_trial_model_path` but selects the trial with the highest
    ``best_mean_reward`` regardless of whether it met the curriculum gate.
    Used by ``--force-continue`` to chain stages even when no trial passes.

    Raises:
        SweepStageError: If no trial reported a valid reward.
    """
    best_row: dict | None = None
    best_value = float("-inf")

    for row in stage_rows:
        best_reward = row.get("best_mean_reward")
        if best_reward is not None and best_reward > best_value:
            best_value = best_reward
            best_row = row

    if best_row is None:
        raise SweepStageError(
            f"No trials reported a valid reward for stage {stage}. All trials may have crashed."
        )

    best_trial_id = best_row["trial_id"]
    passed_str = "PASSED" if best_row.get("stage_passed") else "FAILED gate"
    logger.info(
        "Best trial for stage %d (force-continue): id=%s  best_mean_reward=%.4f (%s)",
        stage, best_trial_id, best_value, passed_str,
    )

    if "model_path" in best_row:
        path = best_row["model_path"]
    else:
        path = f"/gcs/{bucket}/sweeps/{species}/stage{stage}/{best_trial_id}/models/best_model.zip"
    return path, best_row


def _sweep_state_local_path(species: str, algorithm: str) -> Path:
    """Return the local path for a sweep state file."""
    return Path(f"sweep_state_{species}_{algorithm}.json")


def _save_sweep_state(
    state: dict,
    species: str,
    algorithm: str,
    bucket: str | None = None,
    project: str | None = None,
) -> None:
    """Persist sweep progress to local disk and (best-effort) GCS.

    The state dict is written as JSON to a local file and, if *bucket* is
    provided, also uploaded to
    ``gs://<bucket>/sweeps/<species>/_sweep_state.json``.
    """
    local_path = _sweep_state_local_path(species, algorithm)
    local_path.write_text(json.dumps(state, indent=2))
    logger.info("Sweep state saved locally: %s", local_path)

    if bucket:
        gcs_key = f"sweeps/{species}/_sweep_state_{algorithm}.json"
        try:
            from google.cloud import storage as _gcs

            _client = _gcs.Client(project=project)
            _bkt = _client.bucket(bucket)
            _bkt.blob(gcs_key).upload_from_string(json.dumps(state, indent=2))
            logger.info("Sweep state uploaded to: gs://%s/%s", bucket, gcs_key)
        except Exception as exc:
            logger.warning("Failed to upload sweep state to GCS: %s", exc)


def _load_sweep_state(
    species: str,
    algorithm: str,
    bucket: str | None = None,
    project: str | None = None,
) -> dict | None:
    """Load a previously saved sweep state file.

    Checks GCS first (if *bucket* is provided), then falls back to the
    local file.  Returns ``None`` if no state exists or if the state does
    not match *species* and *algorithm*.
    """
    state: dict | None = None

    # Try GCS first
    if bucket:
        gcs_key = f"sweeps/{species}/_sweep_state_{algorithm}.json"
        try:
            from google.cloud import storage as _gcs

            _client = _gcs.Client(project=project)
            _bkt = _client.bucket(bucket)
            blob = _bkt.blob(gcs_key)
            if blob.exists():
                state = json.loads(blob.download_as_text())
                logger.info("Loaded sweep state from: gs://%s/%s", bucket, gcs_key)
        except Exception as exc:
            logger.warning("Could not read sweep state from GCS: %s", exc)

    # Fall back to local file
    if state is None:
        local_path = _sweep_state_local_path(species, algorithm)
        if local_path.exists():
            try:
                state = json.loads(local_path.read_text())
                logger.info("Loaded sweep state from: %s", local_path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read local sweep state: %s", exc)

    # Validate species + algorithm match
    if state is not None:
        if state.get("species") != species or state.get("algorithm") != algorithm:
            logger.warning(
                "Sweep state species/algorithm (%s/%s) does not match current (%s/%s) — ignoring",
                state.get("species"),
                state.get("algorithm"),
                species,
                algorithm,
            )
            return None

    return state


def _is_retryable_gcp_error(exc: Exception) -> bool:
    """Return ``True`` if *exc* is a transient GCP error worth retrying.

    Matches by class name so we don't need ``google-cloud-aiplatform``
    installed at import time (the exceptions live in ``google.api_core``).
    """
    retryable_names = {
        "ResourceExhausted",
        "ServiceUnavailable",
        "GoogleAPICallError",
        "TooManyRequests",
        "InternalServerError",
        "GatewayTimeout",
    }
    return type(exc).__name__ in retryable_names


def _submit_stage_sweep(
    *,
    aiplatform,
    hpt_module,
    species: str,
    stage: int,
    algorithm: str,
    timesteps: int,
    n_envs: int,
    trials: int,
    parallel: int,
    bucket: str,
    image: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    search_space: dict,
    load_path: str | None = None,
    fixed_trial_args: list[str] | None = None,
    wandb: bool = False,
    sync: bool = False,
    resume_run: int = 0,
):
    """Build and submit a single-stage HPT job. Returns the job object.

    Args:
        fixed_trial_args: Extra CLI args appended verbatim to every trial's
            command line.  Used to inject hyperparameters that are *not* part
            of the search space but must match a prior stage's winning trial
            (e.g. ``["--ppo_net_arch", "medium"]``).
        resume_run: When resuming a partially completed stage, set to a
            positive integer so the new job writes to a separate GCS
            sub-directory (``stage{N}_r{resume_run}``) to avoid overwriting
            checkpoints from previous runs.
    """
    parameter_spec = _build_parameter_spec(search_space, hpt_module)
    if not parameter_spec:
        logger.error("No valid parameters in search space for stage %d — aborting", stage)
        sys.exit(1)

    output_base = f"/gcs/{bucket}/sweeps/{species}/stage{stage}"
    if resume_run:
        output_base = f"{output_base}_r{resume_run}"

    trial_args = [
        "environments/shared/scripts/sweep.py",
        "trial",
        "--species",
        species,
        "--stage",
        str(stage),
        "--algorithm",
        algorithm,
        "--timesteps",
        str(timesteps),
        "--n-envs",
        str(n_envs),
        "--output-dir",
        output_base,
    ]
    if load_path:
        trial_args += ["--load", load_path]
    if fixed_trial_args:
        trial_args += fixed_trial_args
    if wandb:
        trial_args.append("--wandb")

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": image,
                "command": ["python"],
                "args": trial_args,
            },
        }
    ]

    display_name = f"{species}-stage{stage}-{algorithm}-sweep"

    logger.info("Submitting HPT job: %s", display_name)
    logger.info("  Trials: %d  |  Parallel: %d", trials, parallel)
    logger.info("  Search space:")
    for k, v in search_space.items():
        logger.info("    %-30s %s", k, v)
    if load_path:
        logger.info("  Warm-start model: %s", load_path)

    custom_job = aiplatform.CustomJob(
        display_name=f"{display_name}-trial",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"gs://{bucket}/sweeps/{species}/stage{stage}",
    )

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name=display_name,
        custom_job=custom_job,
        metric_spec={"best_mean_reward": "maximize"},
        parameter_spec=parameter_spec,
        max_trial_count=trials,
        parallel_trial_count=parallel,
    )

    # Retry loop for transient Vertex AI / quota errors.
    _RETRY_DELAYS = [60, 180, 300]  # seconds between retries
    last_exc: Exception | None = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            hpt_job.run(sync=sync)
            break  # success
        except Exception as exc:
            # Only retry on transient / quota errors from the Google API.
            _retryable = _is_retryable_gcp_error(exc)
            if not _retryable or attempt >= len(_RETRY_DELAYS):
                raise _SweepJobFailed(str(exc), hpt_job=hpt_job) from exc
            last_exc = exc
            delay = _RETRY_DELAYS[attempt]
            logger.warning(
                "Vertex AI error on attempt %d/%d for stage %d: %s. Retrying in %ds …",
                attempt + 1,
                len(_RETRY_DELAYS) + 1,
                stage,
                exc,
                delay,
            )
            time.sleep(delay)
    else:
        # All retries exhausted — should not reach here because we re-raise
        # above, but guard defensively.
        raise _SweepJobFailed(
            f"Job submission for stage {stage} failed after {len(_RETRY_DELAYS) + 1} attempts",
            hpt_job=hpt_job,
        ) from last_exc

    logger.info("Job submitted: %s", hpt_job.resource_name)
    logger.info("Monitor at: https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs")
    logger.info("Results will be written to: gs://%s/sweeps/%s/stage%d/", bucket, species, stage)
    return hpt_job


def launch_sweep(args: argparse.Namespace) -> None:
    """Submit a Vertex AI Hyperparameter Tuning job for a single stage.

    Each trial runs ``sweep.py trial`` inside the Docker container. The
    HPT service injects the trial's parameter values as additional CLI args.
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
    except ImportError:
        logger.error("google-cloud-aiplatform is not installed.\nInstall it with:  pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(
        project=args.project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}",
    )

    # Load search space: inline JSON > file > algorithm default
    resolved = _resolve_search_space(args.search_space, args.search_space_file, args.algorithm)
    search_space = _search_space_for_stage(resolved, args.stage)

    _submit_stage_sweep(
        aiplatform=aiplatform,
        hpt_module=hpt,
        species=args.species,
        stage=args.stage,
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        trials=args.trials,
        parallel=args.parallel,
        bucket=args.bucket,
        image=args.image,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        search_space=search_space,
        wandb=args.wandb,
        sync=False,  # fire-and-forget for single-stage launch
    )


def launch_all_stages(args: argparse.Namespace) -> None:
    """Run Stage 1 → Stage 2 → Stage 3 HPT sweeps sequentially in one command.

    Workflow:

    1. Submit Stage 1 sweep and wait for it to complete.
    2. Identify the best Stage 1 trial (by ``best_mean_reward``).
    3. Submit Stage 2 sweep, loading the best Stage 1 checkpoint.
    4. Identify the best Stage 2 trial.
    5. Submit Stage 3 sweep, loading the best Stage 2 checkpoint.

    Each stage can have its own budget via ``--trials-stageN``,
    ``--parallel-stageN``, and ``--timesteps-stageN`` flags.  When a
    per-stage flag is omitted, the shared ``--trials`` / ``--parallel``
    default is used.

    The search space can be customised per stage using a JSON file with
    ``"stage1"`` / ``"stage2"`` / ``"stage3"`` top-level keys (see
    ``--search-space-file``).  A flat JSON file or inline ``--search-space``
    applies the same space to all three stages.
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import hyperparameter_tuning as hpt
    except ImportError:
        logger.error("google-cloud-aiplatform is not installed.\nInstall it with:  pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(
        project=args.project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}",
    )

    # Load search space: inline JSON > file > algorithm default
    # If the file uses per-stage keys ("stage1", "stage2", "stage3"), each
    # stage gets its own search space.  Otherwise the same space is reused.
    resolved = _resolve_search_space(args.search_space, args.search_space_file, args.algorithm)

    # Per-stage budgets: CLI flags > search-space file settings > shared defaults
    # The search-space file can include "trials", "timesteps", "parallel",
    # "n_envs" alongside the search-space params.  CLI flags always win.
    cli_timesteps = [args.timesteps_stage1, args.timesteps_stage2, args.timesteps_stage3]
    cli_trials = [
        args.trials_stage1,
        args.trials_stage2,
        args.trials_stage3,
    ]
    cli_parallel = [
        args.parallel_stage1,
        args.parallel_stage2,
        args.parallel_stage3,
    ]

    load_path: str | None = None
    fixed_trial_args: list[str] | None = None
    all_rows: list[dict] = []
    sweep_start_time = time.monotonic()

    # Determine the net_arch HPT key for this algorithm (e.g. "ppo_net_arch")
    net_arch_key = f"{args.algorithm}_net_arch"

    # ── Resume: restore state from a previous (possibly interrupted) run ─────
    completed_stages: set[int] = set()
    partial_stages: dict[int, dict] = {}
    sweep_state: dict = {
        "species": args.species,
        "algorithm": args.algorithm,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }
    logger.info("Resume mode: %s", "enabled" if args.resume else "disabled")
    if args.resume:
        prev_state = _load_sweep_state(args.species, args.algorithm, bucket=args.bucket, project=args.project)
        if prev_state and prev_state.get("stages"):
            sweep_state = prev_state
            for stg_key, stg_data in prev_state["stages"].items():
                stg_num = int(stg_key)
                if stg_data.get("status") == "completed":
                    completed_stages.add(stg_num)
                    # Restore load_path and fixed_trial_args from the last completed stage
                    if stg_data.get("best_model_path"):
                        load_path = stg_data["best_model_path"]
                    if stg_data.get("fixed_trial_args"):
                        fixed_trial_args = stg_data["fixed_trial_args"]
                    if stg_data.get("trial_rows"):
                        all_rows.extend(stg_data["trial_rows"])
                elif stg_data.get("status") == "partial":
                    partial_stages[stg_num] = stg_data
                    if stg_data.get("fixed_trial_args"):
                        fixed_trial_args = stg_data["fixed_trial_args"]
            if completed_stages:
                logger.info(
                    "Resuming sweep: stages %s already completed. load_path=%s",
                    sorted(completed_stages),
                    load_path,
                )
            if partial_stages:
                for ps, pd in sorted(partial_stages.items()):
                    logger.info(
                        "Resuming sweep: stage %d has %d partial trial results from a previous run.",
                        ps,
                        len(pd.get("trial_rows", [])),
                    )
    # ── End resume ───────────────────────────────────────────────────────────

    for stage in range(1, 4):
        # Skip stages that were already completed in a previous run
        if stage in completed_stages:
            logger.info("=" * 60)
            logger.info("ALL-STAGES SWEEP  —  Stage %d / 3  (SKIPPED — already completed)", stage)
            logger.info("=" * 60)
            continue

        stage_start_time = time.monotonic()
        search_space = _search_space_for_stage(resolved, stage)
        file_settings = _settings_for_stage(resolved, stage)

        # Resolve each setting: CLI flag > file setting > shared CLI default
        timesteps = cli_timesteps[stage - 1]
        if timesteps is None:
            timesteps = file_settings.get("timesteps", cli_timesteps[stage - 1])
        if timesteps is None:
            # Fall back to the argparse defaults (500k/1M/1.5M)
            timesteps = [500_000, 1_000_000, 1_500_000][stage - 1]

        trials = cli_trials[stage - 1]
        if trials is None:
            trials = file_settings.get("trials", args.trials)

        parallel = cli_parallel[stage - 1]
        if parallel is None:
            parallel = file_settings.get("parallel", args.parallel)

        n_envs = file_settings.get("n_envs", args.n_envs)

        # ── Partial recovery: check for trial results from a previous
        #    interrupted run of this stage ──────────────────────────────────
        partial_data = partial_stages.get(stage)
        partial_rows: list[dict] = []
        resume_run = 0
        if partial_data:
            partial_rows = [r for r in partial_data.get("trial_rows", []) if r.get("best_mean_reward") is not None]
            if partial_rows:
                resume_run = partial_data.get("resume_run", 0) + 1

        logger.info("=" * 60)
        logger.info("ALL-STAGES SWEEP  —  Stage %d / 3", stage)
        logger.info(
            "  Timesteps: %s  |  Trials: %d  |  Parallel: %d  |  n_envs: %d", timesteps, trials, parallel, n_envs
        )
        logger.info("=" * 60)

        try:
            remaining_trials = trials - len(partial_rows)
            job_resource_name = None

            if remaining_trials > 0:
                if partial_rows:
                    logger.info(
                        "Resuming stage %d: %d trials recovered from previous run, %d remaining.",
                        stage,
                        len(partial_rows),
                        remaining_trials,
                    )

                hpt_job = _submit_stage_sweep(
                    aiplatform=aiplatform,
                    hpt_module=hpt,
                    species=args.species,
                    stage=stage,
                    algorithm=args.algorithm,
                    timesteps=timesteps,
                    n_envs=n_envs,
                    trials=remaining_trials,
                    parallel=parallel,
                    bucket=args.bucket,
                    image=args.image,
                    machine_type=args.machine_type,
                    accelerator_type=args.accelerator_type,
                    accelerator_count=args.accelerator_count,
                    search_space=search_space,
                    load_path=load_path,
                    fixed_trial_args=fixed_trial_args,
                    wandb=args.wandb,
                    sync=True,  # block until this stage's sweep is done
                    resume_run=resume_run,
                )
                job_resource_name = getattr(hpt_job, "resource_name", None)

                # Collect per-trial results and append to the running list
                from environments.shared.config import load_stage_config as _load_stage_config

                stage_config = _load_stage_config(args.species, stage)
                new_rows = _collect_trial_results(hpt_job, stage, stage_config)

                # Tag new rows with model_path so _best_trial_model_path can
                # locate the correct checkpoint even when trials span multiple
                # GCS output directories (original run vs resumed run).
                output_base = f"/gcs/{args.bucket}/sweeps/{args.species}/stage{stage}"
                if resume_run:
                    output_base = f"{output_base}_r{resume_run}"
                for row in new_rows:
                    row["model_path"] = f"{output_base}/{row['trial_id']}/models/best_model.zip"

                stage_rows = partial_rows + new_rows
            else:
                # All requested trials were already completed in a previous
                # partial run — no need to submit a new job.
                logger.info(
                    "Stage %d: all %d trials already completed in previous run. Skipping job submission.",
                    stage,
                    len(partial_rows),
                )
                stage_rows = list(partial_rows)

            all_rows.extend(stage_rows)

            stage_elapsed = time.monotonic() - stage_start_time
            stage_mins = stage_elapsed / 60
            logger.info(
                "Stage %d finished in %.1f min (%.0f s). Trials: %d total, %d passed.",
                stage,
                stage_mins,
                stage_elapsed,
                len(stage_rows),
                sum(1 for r in stage_rows if r.get("stage_passed")),
            )

            # Identify the best passing trial for this stage (used for
            # chaining stages 1→2→3 and for reporting in the saved state).
            try:
                best_model_path, best_row = _best_trial_model_path(stage_rows, args.bucket, args.species, stage)
            except SweepStageError:
                # No trials passed the curriculum gate.
                if args.force_continue and stage < 3:
                    # --force-continue: pick the best trial regardless of
                    # gate status and chain it into the next stage.
                    logger.warning(
                        "Stage %d: no trials passed curriculum gate. "
                        "--force-continue is set — selecting best trial anyway.",
                        stage,
                    )
                    best_model_path, best_row = _best_trial_model_path_any(
                        stage_rows, args.bucket, args.species, stage,
                    )
                else:
                    best_model_path = None
                    best_row = None
                    if stage < 3:
                        raise  # stages 1-2 must pass to chain forward

            if best_row is not None:
                logger.info(
                    "Stage %d best passing trial: id=%s  reward=%.4f",
                    stage,
                    best_row["trial_id"],
                    best_row.get("best_mean_reward", 0),
                )

            if stage < 3 and best_model_path is not None:
                # Chain the best checkpoint into the next stage
                load_path = best_model_path
                logger.info("Stage %d complete. Best passing model: %s", stage, load_path)

                # Propagate the winning trial's net_arch to subsequent stages so
                # every trial loads the checkpoint with a matching architecture.
                # net_arch is only searched in stage 1; stages 2+ inherit it.
                best_net_arch = best_row.get(net_arch_key) if best_row else None
                if best_net_arch is not None:
                    fixed_trial_args = [f"--{net_arch_key}", str(best_net_arch)]
                    logger.info("Propagating %s=%s from best trial to stage %d", net_arch_key, best_net_arch, stage + 1)

            # Save state after each stage completes successfully
            sweep_state["stages"][str(stage)] = {
                "status": "completed",
                "job_resource_name": job_resource_name,
                "best_trial_id": best_row["trial_id"] if best_row else None,
                "best_mean_reward": best_row.get("best_mean_reward") if best_row else None,
                "best_model_path": best_model_path if stage < 3 else None,
                "fixed_trial_args": fixed_trial_args,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "trial_rows": stage_rows,
            }
            _save_sweep_state(
                sweep_state,
                args.species,
                args.algorithm,
                bucket=args.bucket,
                project=args.project,
            )

        except Exception as exc:
            if isinstance(exc, SweepStageError) or _is_retryable_gcp_error(exc):
                # ── Partial trial recovery ────────────────────────────────
                # Try to salvage completed trials from the failed job so they
                # can be reused on the next ``--resume`` run.
                if isinstance(exc, _SweepJobFailed) and exc.hpt_job is not None:
                    try:
                        from environments.shared.config import load_stage_config as _lsc

                        _sc = _lsc(args.species, stage)
                        _new_partial = _collect_trial_results(exc.hpt_job, stage, _sc)
                        # Keep only trials that actually reported metrics
                        _new_partial = [r for r in _new_partial if r.get("best_mean_reward") is not None]
                        if _new_partial:
                            # Tag each row with its model_path for correct
                            # checkpoint resolution on resume.
                            _out = f"/gcs/{args.bucket}/sweeps/{args.species}/stage{stage}"
                            if resume_run:
                                _out = f"{_out}_r{resume_run}"
                            for row in _new_partial:
                                row["model_path"] = f"{_out}/{row['trial_id']}/models/best_model.zip"

                            all_partial = partial_rows + _new_partial
                            sweep_state["stages"][str(stage)] = {
                                "status": "partial",
                                "job_resource_name": getattr(exc.hpt_job, "resource_name", None),
                                "trial_rows": all_partial,
                                "fixed_trial_args": fixed_trial_args,
                                "resume_run": resume_run,
                            }
                            logger.info(
                                "Recovered %d completed trials from failed stage %d "
                                "(total partial: %d). These will be reused on resume.",
                                len(_new_partial),
                                stage,
                                len(all_partial),
                            )
                    except Exception as collect_exc:
                        logger.warning("Could not collect partial trial results: %s", collect_exc)

                logger.error(
                    "Stage %d failed: %s. Saving progress — completed stages: %s. "
                    "Re-run the same command to resume from where you left off.",
                    stage,
                    exc,
                    sorted(int(k) for k, v in sweep_state["stages"].items() if v.get("status") == "completed"),
                )
                _save_sweep_state(
                    sweep_state,
                    args.species,
                    args.algorithm,
                    bucket=args.bucket,
                    project=args.project,
                )
                sys.exit(1)
            raise

    # Write a combined CSV of every trial across all three stages
    csv_path = Path(f"sweep_results_{args.species}_{args.algorithm}.csv")
    write_results_csv(all_rows, csv_path)

    # Persist the CSV and graphs to GCS alongside the trial artifacts
    _gcs_bucket = None
    try:
        from google.cloud import storage as _gcs

        _gcs_client = _gcs.Client(project=args.project)
        _gcs_bucket = _gcs_client.bucket(args.bucket)
    except Exception as exc:
        logger.warning("Could not initialise GCS client for uploads: %s", exc)

    if _gcs_bucket is not None:
        gcs_csv_path = f"sweeps/{args.species}/{csv_path.name}"
        try:
            _gcs_bucket.blob(gcs_csv_path).upload_from_filename(str(csv_path))
            logger.info("Sweep CSV uploaded to: gs://%s/%s", args.bucket, gcs_csv_path)
        except Exception as exc:
            logger.warning("Failed to upload sweep CSV to GCS: %s. Local copy at: %s", exc, csv_path)

    # Generate sweep visualisation graphs
    plot_sweep_results(csv_path, args.species, args.algorithm)

    # Upload graphs to GCS alongside the CSV
    if _gcs_bucket is not None:
        for graph_name in ("sweep_trial_metrics.png", "sweep_hyperparameter_analysis.png"):
            graph_path = csv_path.parent / graph_name
            if graph_path.exists():
                gcs_graph_path = f"sweeps/{args.species}/{graph_name}"
                try:
                    _gcs_bucket.blob(gcs_graph_path).upload_from_filename(str(graph_path))
                    logger.info("Sweep graph uploaded to: gs://%s/%s", args.bucket, gcs_graph_path)
                except Exception as exc:
                    logger.warning("Failed to upload %s to GCS: %s", graph_name, exc)

    total_elapsed = time.monotonic() - sweep_start_time
    total_mins = total_elapsed / 60
    logger.info("=" * 60)
    logger.info(
        "ALL-STAGES SWEEP COMPLETE for %s (%s) in %.1f min (%.0f s)",
        args.species,
        args.algorithm,
        total_mins,
        total_elapsed,
    )
    logger.info("All results at: gs://%s/sweeps/%s/", args.bucket, args.species)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mesozoic Labs hyperparameter sweep tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode")

    # ── trial mode ────────────────────────────────────────────────────────────
    trial = subparsers.add_parser(
        "trial",
        help="Run one sweep trial (used by Vertex AI HPT workers inside Docker)",
    )
    trial.add_argument("--species", required=True, choices=["velociraptor", "brachiosaurus", "trex"])
    trial.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    trial.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    trial.add_argument("--timesteps", type=int, default=500000, help="Training timesteps per trial")
    trial.add_argument("--n-envs", type=int, default=4, help="Parallel environments per trial")
    trial.add_argument("--seed", type=int, default=42)
    trial.add_argument("--eval-freq", type=int, default=10000)
    trial.add_argument("--save-freq", type=int, default=50000)
    trial.add_argument("--load", type=str, default=None, help="Path to model checkpoint to warm-start from")
    trial.add_argument("--output-dir", type=str, default=None, help="Base output dir (GCS mount path on Vertex AI)")
    trial.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # ── launch mode (single stage) ────────────────────────────────────────────
    launch = subparsers.add_parser(
        "launch",
        help="Submit a Vertex AI Hyperparameter Tuning job for one stage",
    )
    launch.add_argument("--species", required=True, choices=["velociraptor", "brachiosaurus", "trex"])
    launch.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Curriculum stage to sweep")
    launch.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    launch.add_argument("--timesteps", type=int, default=500000, help="Training timesteps per trial")
    launch.add_argument("--n-envs", type=int, default=4, help="Parallel environments per trial")
    launch.add_argument("--project", required=True, help="GCP project ID")
    launch.add_argument("--location", default="us-central1", help="GCP region")
    launch.add_argument("--bucket", required=True, help="GCS bucket name (without gs:// prefix)")
    launch.add_argument("--image", required=True, help="Docker image URI for trial workers")
    launch.add_argument("--trials", type=int, default=20, help="Maximum number of trials")
    launch.add_argument("--parallel", type=int, default=5, help="Parallel trials running at once")
    launch.add_argument("--machine-type", default="n1-standard-8", help="Vertex AI machine type")
    launch.add_argument(
        "--accelerator-type",
        default="NVIDIA_TESLA_T4",
        help="Vertex AI accelerator type (use 'None' for CPU-only)",
    )
    launch.add_argument("--accelerator-count", type=int, default=1)
    launch.add_argument(
        "--search-space",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "JSON search space override (inline string). If omitted, the default space for the chosen "
            "algorithm is used. See module docstring for format details."
        ),
    )
    launch.add_argument(
        "--search-space-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file defining the search space. Supports per-stage "
            "sections (keys: stage1, stage2, stage3) or a flat dict for all stages."
        ),
    )
    launch.add_argument("--wandb", action="store_true", help="Enable W&B logging in each trial")

    # ── launch-all mode (all stages, sequential) ──────────────────────────────
    launch_all = subparsers.add_parser(
        "launch-all",
        help="Sweep all three curriculum stages end-to-end with a single command",
    )
    launch_all.add_argument("--species", required=True, choices=["velociraptor", "brachiosaurus", "trex"])
    launch_all.add_argument("--algorithm", type=str, choices=["ppo", "sac"], default="ppo")
    launch_all.add_argument("--n-envs", type=int, default=4, help="Parallel environments per trial")
    launch_all.add_argument("--project", required=True, help="GCP project ID")
    launch_all.add_argument("--location", default="us-central1", help="GCP region")
    launch_all.add_argument("--bucket", required=True, help="GCS bucket name (without gs:// prefix)")
    launch_all.add_argument("--image", required=True, help="Docker image URI for trial workers")
    launch_all.add_argument(
        "--trials", type=int, default=20, help="Default max trials per stage (overridden by --trials-stageN)"
    )
    launch_all.add_argument(
        "--trials-stage1", type=int, default=None, help="Max trials for Stage 1 (defaults to --trials)"
    )
    launch_all.add_argument(
        "--trials-stage2", type=int, default=None, help="Max trials for Stage 2 (defaults to --trials)"
    )
    launch_all.add_argument(
        "--trials-stage3", type=int, default=None, help="Max trials for Stage 3 (defaults to --trials)"
    )
    launch_all.add_argument(
        "--parallel", type=int, default=5, help="Default parallel trials per stage (overridden by --parallel-stageN)"
    )
    launch_all.add_argument(
        "--parallel-stage1", type=int, default=None, help="Parallel trials for Stage 1 (defaults to --parallel)"
    )
    launch_all.add_argument(
        "--parallel-stage2", type=int, default=None, help="Parallel trials for Stage 2 (defaults to --parallel)"
    )
    launch_all.add_argument(
        "--parallel-stage3", type=int, default=None, help="Parallel trials for Stage 3 (defaults to --parallel)"
    )
    launch_all.add_argument(
        "--timesteps-stage1", type=int, default=None, help="Timesteps per Stage 1 trial (default: 500k)"
    )
    launch_all.add_argument(
        "--timesteps-stage2", type=int, default=None, help="Timesteps per Stage 2 trial (default: 1M)"
    )
    launch_all.add_argument(
        "--timesteps-stage3", type=int, default=None, help="Timesteps per Stage 3 trial (default: 1.5M)"
    )
    launch_all.add_argument("--machine-type", default="n1-standard-8", help="Vertex AI machine type")
    launch_all.add_argument(
        "--accelerator-type",
        default="NVIDIA_TESLA_T4",
        help="Vertex AI accelerator type (use 'None' for CPU-only)",
    )
    launch_all.add_argument("--accelerator-count", type=int, default=1)
    launch_all.add_argument(
        "--search-space",
        type=str,
        default=None,
        metavar="JSON",
        help="JSON search space override (inline string, applied to all stages). Defaults to the algorithm's default space.",
    )
    launch_all.add_argument(
        "--search-space-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file defining the search space. Use per-stage keys "
            "(stage1, stage2, stage3) to give each stage its own search space, "
            "or a flat dict to apply the same space to all stages."
        ),
    )
    launch_all.add_argument("--wandb", action="store_true", help="Enable W&B logging in each trial")
    launch_all.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from the last completed stage if a sweep state file exists "
            "(default: --resume). Use --no-resume to start fresh."
        ),
    )
    launch_all.add_argument(
        "--force-continue",
        action="store_true",
        default=False,
        help=(
            "Continue to the next stage even when no trials pass the curriculum "
            "gate. The best trial (by reward) is selected regardless of gate "
            "status. Useful for running all three stages to completion."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    # parse_known_args so Vertex AI HPT can inject extra --param value pairs
    args, extra_args = parser.parse_known_args()

    if args.mode == "trial":
        run_trial(args, extra_args)
    elif args.mode == "launch":
        if extra_args:
            logger.warning("Ignoring unexpected args in launch mode: %s", extra_args)
        launch_sweep(args)
    elif args.mode == "launch-all":
        if extra_args:
            logger.warning("Ignoring unexpected args in launch-all mode: %s", extra_args)
        launch_all_stages(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
