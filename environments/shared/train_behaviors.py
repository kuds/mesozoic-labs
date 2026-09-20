"""Train and evaluate direction and randomized-terrain PPO behaviors for all species.

Example: python -m environments.shared.train_behaviors --help
Artifacts are self-contained behavior bundles, separate from canonical gates.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import secrets
import subprocess
import time
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from environments.shared.behavior_env import canonical_env_parameters, get_behavior_env_class
from environments.shared.direction_commands import DirectionCommandConfig
from environments.shared.species_names import resolve_species_id, species_display_names
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.terrain import TerrainConfig
from environments.shared.terrain_sampling import TerrainSamplerConfig, get_sampled_behavior_env_class

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_recipe(
    path: Path, species: str | None = None
) -> tuple[dict[str, Any], DirectionCommandConfig, TerrainConfig | None, dict[str, Any]]:
    """Load a species behavior recipe; historical T-Rex recipes remain readable."""
    with path.open("rb") as stream:
        recipe = tomllib.load(stream)
    unknown = set(recipe) - {"behavior", "pilot", "commands", "terrain", "terrain_sampler", "env", "ppo"}
    if "behavior" in recipe and "pilot" in recipe:
        raise ValueError("Choose one behavior metadata section, not both behavior and pilot")
    section = "behavior" if "behavior" in recipe else "pilot"
    metadata = recipe.get(section, {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{section} must be a table")
    if section == "behavior":
        if not {"species", "name", "parent"} <= metadata.keys():
            raise ValueError("behavior requires species, name and parent")
        if not isinstance(metadata["name"], str) or not metadata["name"].strip():
            raise ValueError("behavior.name must be a nonempty string")
        if metadata["parent"] != "locomotion":
            raise ValueError("behavior.parent must be locomotion")
        declared_species = resolve_species_id(metadata["species"])
    else:
        declared_species = "trex"
    species = resolve_species_id(species) if species is not None else declared_species
    if species != declared_species:
        raise ValueError("Recipe species differs from the requested species")
    if species not in species_display_names(backend="stable-baselines3"):
        raise ValueError(f"Species {species!r} has no supported SB3 behaviors")
    if unknown:
        raise ValueError(f"Unknown recipe sections: {sorted(unknown)}")
    for section, allowed in (
        ("pilot", {"name", "timesteps"}),
        ("behavior", {"species", "name", "parent", "timesteps"}),
        ("ppo", {"learning_rate", "ent_coef", "target_kl", "warmup_timesteps", "warmup_clip_range"}),
    ):
        unknown = set(recipe.get(section, {})) - allowed
        if unknown:
            raise ValueError(f"Unknown {section} fields: {sorted(unknown)}")
    env_class = get_behavior_env_class(species)
    allowed_env = set(canonical_env_parameters(species)) | set(inspect.signature(env_class.__init__).parameters)
    allowed_env -= {"self", "env_kwargs", "commands", "terrain", "run_seed"}
    if unknown := set(recipe.get("env", {})) - allowed_env:
        raise ValueError(f"Unknown env fields: {sorted(unknown)}")
    recipe["ppo"] = {
        "learning_rate": 5e-5,
        "ent_coef": 0.005,
        "target_kl": 0.03,
        "warmup_timesteps": 100_000,
        "warmup_clip_range": 0.02,
        **recipe.get("ppo", {}),
    }
    for key, allow_zero in (
        ("learning_rate", False),
        ("ent_coef", True),
        ("target_kl", False),
        ("warmup_clip_range", False),
    ):
        value = recipe["ppo"][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)
        ):
            raise ValueError(f"Invalid ppo.{key}")
    if recipe["ppo"]["warmup_clip_range"] > 0.2:
        raise ValueError("ppo.warmup_clip_range must not exceed 0.2")
    metadata_section = "behavior" if "behavior" in recipe else "pilot"
    for section, key, default in ((metadata_section, "timesteps", 3_000_000), ("ppo", "warmup_timesteps", 100_000)):
        value = recipe.setdefault(section, {}).setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{section}.{key} must be a nonnegative integer")
    for section, cls in (("commands", DirectionCommandConfig), ("terrain", TerrainConfig)):
        allowed = {f.name for f in fields(cls)} | ({"enabled"} if section == "terrain" else set())
        unknown = set(recipe.get(section, {})) - allowed
        if unknown:
            raise ValueError(f"Unknown {section} fields: {sorted(unknown)}")
    commands = DirectionCommandConfig(**recipe.get("commands", {}))
    terrain_values = dict(recipe.get("terrain", {}))
    enabled = terrain_values.pop("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("terrain.enabled must be a boolean")
    terrain = TerrainConfig(**terrain_values) if enabled else None
    sampler = None
    if "terrain_sampler" in recipe:
        values = recipe["terrain_sampler"]
        if not isinstance(values, dict):
            raise ValueError("terrain_sampler must be a table")
        if unknown := set(values) - {f.name for f in fields(TerrainSamplerConfig)}:
            raise ValueError(f"Unknown terrain_sampler fields: {sorted(unknown)}")
        sampler = TerrainSamplerConfig(**values)
        if terrain is None or terrain.mode != "gentle":
            raise ValueError("terrain_sampler requires enabled gentle terrain")
        if recipe.get("env", {}).get("flat_probability", 0.0) != 0:
            raise ValueError("terrain_sampler owns flat sampling; env.flat_probability must be zero")
    # Keep the measured walker posture/control settings and override only the
    # declared behavior settings. The resolved values enter the task identity.
    parent_stage = load_stage_manifest(species).by_id("locomotion")
    baseline = REPO_ROOT / "configs" / species / parent_stage.config_file
    with baseline.open("rb") as stream:
        defaults = tomllib.load(stream)
    env_kwargs = {**defaults["env"], "max_episode_steps": 2500, **recipe.get("env", {})}
    if sampler is not None:
        env_kwargs["terrain_sampler"] = sampler
    return recipe, commands, terrain, env_kwargs


def create_behavior_env(
    species: str,
    *,
    commands: DirectionCommandConfig,
    terrain: TerrainConfig | None,
    run_seed: int,
    **env_kwargs: Any,
) -> Any:
    """Construct the fixed-template or sampled behavior declared by a recipe."""
    factory = get_sampled_behavior_env_class if "terrain_sampler" in env_kwargs else get_behavior_env_class
    return factory(species)(commands=commands, terrain=terrain, run_seed=run_seed, **env_kwargs)


class EpisodeManifestRecorder(gym.Wrapper):
    """Append each actual reset recipe, including automatically reset episodes."""

    def __init__(self, env: gym.Env, path: Path):
        super().__init__(env)
        self.path = path

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        with self.path.open("a") as stream:
            stream.write(json.dumps(info, allow_nan=False) + "\n")
        return obs, info


def _verify_bundle(model: Path, normalizer: Path, recipe: dict[str, Any] | None = None) -> None:
    manifest_path = model.parent / "bundle.json"
    if not manifest_path.is_file():
        raise ValueError("Resume requires bundle.json beside the checkpoint")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_sha256") != _sha(model) or manifest.get("normalizer_sha256") != _sha(normalizer):
        raise ValueError("Checkpoint or normalization file does not match the saved bundle hashes")
    if recipe is not None and manifest.get("training_recipe") != recipe:
        raise ValueError("Exact resume requires the saved training recipe, including PPO settings and stage budget")


def _save_bundle(model: Any, normalizer: Any, output: Path, identity: dict, recipe: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_path, norm_path = output / "model.zip", output / "vecnormalize.pkl"
    model.save(str(model_path))
    normalizer.save(str(norm_path))
    _write(
        output / "bundle.json",
        {
            "schema": "mesozoic.behavior-bundle/v1",
            "model": model_path.name,
            "normalizer": norm_path.name,
            "model_sha256": _sha(model_path),
            "normalizer_sha256": _sha(norm_path),
            "behavior_identity": identity,
            "training_recipe": recipe,
            "num_timesteps": model.num_timesteps,
            "canonical_certification": False,
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", help="Registered species ID or display name (defaults to recipe species)")
    parser.add_argument(
        "--recipe",
        "--config",
        dest="config",
        type=Path,
        required=True,
        help="Behavior TOML in configs/<species>/behaviors (--config remains an alias)",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vecnormalize", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New or empty local output directory")
    parser.add_argument("--seed", type=int, help="Omit to generate a fresh recorded run seed")
    parser.add_argument("--steps", type=int, help="Additional adaptation steps (rounded up to a PPO rollout)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true", help="Prepare/load and evaluate without learning")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save scored episode videos with matching terrain maps, raw heights and paths",
    )
    parser.add_argument("--video-fps", type=float, default=25.0, help="Replay frame rate (default: 25)")
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument(
        "--resume", action="store_true", help="Continue this exact behavior without re-zeroing commands"
    )
    loading.add_argument(
        "--adapt", action="store_true", help="Adapt a learned behavior to the next compatible behavior recipe"
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None or args.vecnormalize is None:
        parser.error(
            "Set both --checkpoint and --vecnormalize: every mode (prepare, --resume, --adapt, --eval-only) takes an "
            "explicit matched model/normalization pair; nothing is selected automatically"
        )
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.eval_episodes < 0:
        parser.error("--eval-episodes must be nonnegative")
    if not np.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("--video-fps must be finite and positive")
    if args.record_video and args.eval_episodes == 0:
        parser.error("--record-video requires at least one evaluation episode")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    recipe, commands, terrain, env_kwargs = read_recipe(args.config, species=args.species)
    metadata_section = "behavior" if "behavior" in recipe else "pilot"
    metadata = recipe[metadata_section]
    species = resolve_species_id(metadata.get("species", "trex"))
    if args.resume or args.adapt:
        _verify_bundle(args.checkpoint, args.vecnormalize, recipe if args.resume else None)
    run_seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    steps = args.steps if args.steps is not None else int(metadata["timesteps"])
    if steps < 0:
        parser.error(f"{metadata_section}.timesteps must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)

    import mujoco
    import stable_baselines3
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    from environments.shared.behavior_checkpoint import (
        adapt_behavior_checkpoint,
        load_behavior_checkpoint,
        prepare_behavior_checkpoint,
    )
    from environments.shared.behavior_evaluation import evaluate_behavior
    from environments.shared.curriculum.advancement import StageWarmupCallback

    torch.set_num_threads(1)
    env = create_behavior_env(species, commands=commands, terrain=terrain, run_seed=run_seed, **env_kwargs)
    if args.record_video:
        try:
            from environments.shared.behavior_replay import require_replay_dependencies

            if args.video_fps > 1.0 / env.dt + 1e-9:
                raise ValueError("--video-fps must not exceed the environment control frequency")
            require_replay_dependencies()
        except Exception:
            env.close()
            raise
    behavior_identity = env.behavior_identity
    wrapped: gym.Env = Monitor(EpisodeManifestRecorder(env, args.output / "training_episodes.jsonl"))
    learning_rate = float(recipe.get("ppo", {}).get("learning_rate", 5e-5))
    loader = (
        adapt_behavior_checkpoint
        if args.adapt
        else load_behavior_checkpoint
        if args.resume
        else prepare_behavior_checkpoint
    )
    try:
        model, normalizer, preparation = loader(
            args.checkpoint,
            args.vecnormalize,
            wrapped,
            learning_rate=learning_rate,
            behavior_identity=behavior_identity,
            species=species,
        )
    except BaseException:
        wrapped.close()
        raise
    # Explicitly seed action sampling, PPO shuffles and the environment. Model
    # loading otherwise restores the parent's seed, making --seed misleading.
    model.set_random_seed(run_seed)
    # Continuing one optimization lineage with another sampling seed does not
    # create an independent training replicate. Adaptation starts a new stage
    # from its exact immediate parent pair; exact resume retains that stage.
    if not args.resume:
        setattr(model, "mesozoic_behavior_training_seed", run_seed)
        setattr(model, "mesozoic_behavior_training_run_id", args.output.name)
        setattr(model, "mesozoic_behavior_training_parent_sha256", "sha256:" + _sha(args.checkpoint))
        setattr(model, "mesozoic_behavior_training_parent_normalization_sha256", "sha256:" + _sha(args.vecnormalize))
    certification_training_seed = getattr(model, "mesozoic_behavior_training_seed", None)
    certification_training_parent = getattr(model, "mesozoic_behavior_training_parent_sha256", None)
    certification_training_parent_normalization = getattr(
        model, "mesozoic_behavior_training_parent_normalization_sha256", None
    )
    normalizer.norm_reward = not args.eval_only
    model.ent_coef = float(recipe.get("ppo", {}).get("ent_coef", 0.005))
    model.target_kl = float(recipe.get("ppo", {}).get("target_kl", 0.03))
    if not args.resume:
        setattr(model, "mesozoic_behavior_stage_start", model.num_timesteps)
        setattr(model, "mesozoic_behavior_stage_start_updates", model._n_updates)
    stage_start = int(getattr(model, "mesozoic_behavior_stage_start"))
    if args.resume and args.steps is None:
        steps = max(0, int(metadata["timesteps"]) - (model.num_timesteps - stage_start))
    warmup_steps = int(recipe.get("ppo", {}).get("warmup_timesteps", 100_000))
    warmup_clip = float(recipe.get("ppo", {}).get("warmup_clip_range", 0.02))
    if warmup_steps < 0 or not 0 < warmup_clip <= 0.2:
        raise ValueError("Invalid PPO warmup configuration")
    # The shared callback compares absolute model steps, including the parent's
    # 8M. Anchor it to this adaptation stage and retain the anchor on resume.
    callback = (
        StageWarmupCallback(
            warmup_timesteps=stage_start + warmup_steps,
            warmup_clip_range=warmup_clip,
            warmup_ent_coef=model.ent_coef,
        )
        if model.num_timesteps < stage_start + warmup_steps
        else None
    )
    model.set_logger(configure(str(args.output), ["stdout", "csv"]))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True)
    manifest: dict[str, Any] = {
        "schema": "mesozoic.behavior-run/v1",
        "species": species,
        "parent_behavior": "walk",
        "parent_stage": "locomotion",
        "status": "running",
        "run_seed": run_seed,
        "recipe": recipe,
        "recipe_sha256": _sha(args.config),
        "behavior_identity": behavior_identity,
        "preparation": preparation,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
        "training": {
            "requested_additional_steps": 0 if args.eval_only else steps,
            "learning_rate": learning_rate,
            "schedule": "constant",
            "clip_range": 0.2,
            "stage_start_timesteps": stage_start,
            "warmup_timesteps": warmup_steps,
            "warmup_clip_range": warmup_clip,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "ent_coef": model.ent_coef,
            "target_kl": model.target_kl,
            "device": str(model.device),
            "observation_normalization": "proprioception only; command passthrough",
            "reward_normalization": bool(normalizer.norm_reward),
        },
        "canonical_certification": False,
        "replay": {"record_video": args.record_video, "video_fps": args.video_fps},
        "certification_training_seed": certification_training_seed,
        "certification_training_parent_sha256": certification_training_parent,
        "certification_training_parent_normalization_sha256": certification_training_parent_normalization,
    }
    _write(args.output / "run.json", manifest)
    print(f"Behavior run seed: {run_seed}; output: {args.output}")
    start_steps, started = model.num_timesteps, time.monotonic()
    interrupted = False
    bundle_saved = False

    class SaveCompletedRollout(BaseCallback):
        """Save matched periodic bundles after completed optimizer updates."""

        def __init__(self) -> None:
            super().__init__()
            self.last_saved = model.num_timesteps

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            if self.model.num_timesteps - self.last_saved < 100_000:
                return
            directory = args.output / "checkpoints" / f"step-{self.model.num_timesteps}"
            temporary = directory.with_name(directory.name + ".tmp")
            _save_bundle(self.model, normalizer, temporary, behavior_identity, recipe)
            temporary.replace(directory)
            _write(
                args.output / "latest_checkpoint.json",
                {"directory": str(directory), "num_timesteps": self.model.num_timesteps},
            )
            self.last_saved = self.model.num_timesteps

    try:
        if not args.eval_only and steps:
            try:
                callbacks: list[BaseCallback] = [SaveCompletedRollout()]
                if callback is not None:
                    callbacks.insert(0, callback)
                model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callbacks)
            except KeyboardInterrupt:
                interrupted = True
        _save_bundle(model, normalizer, args.output, behavior_identity, recipe)
        bundle_saved = True
        manifest["training"]["actual_additional_steps"] = model.num_timesteps - start_steps
        manifest["training"]["elapsed_seconds"] = time.monotonic() - started
        # Disjoint deterministic streams for evaluation; never select a policy
        # using these as a training curriculum signal.
        seeds = [int(s) for s in np.random.SeedSequence([run_seed, 0xE7A1]).generate_state(args.eval_episodes)]
        if seeds and not interrupted:
            manifest["evaluation"] = evaluate_behavior(
                model,
                normalizer,
                episode_seeds=seeds,
                output_dir=args.output,
                record_video=args.record_video,
                video_fps=args.video_fps,
                replay_context={
                    "model_sha256": "sha256:" + _sha(args.output / "model.zip"),
                    "normalizer_sha256": "sha256:" + _sha(args.output / "vecnormalize.pkl"),
                    "behavior_identity": behavior_identity,
                },
            )
        manifest["status"] = "interrupted" if interrupted else "complete"
        _write(args.output / "run.json", manifest)
    except KeyboardInterrupt:
        # Scoring and video export can take longer than a short training run.
        # Keep their already-saved checkpoint, and finish a save interrupted
        # before its matched bundle was published, without restarting scoring.
        if not bundle_saved:
            _save_bundle(model, normalizer, args.output, behavior_identity, recipe)
        manifest["training"]["actual_additional_steps"] = model.num_timesteps - start_steps
        manifest["training"]["elapsed_seconds"] = time.monotonic() - started
        manifest["status"] = "interrupted"
        _write(args.output / "run.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write(args.output / "run.json", manifest)
        raise
    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
