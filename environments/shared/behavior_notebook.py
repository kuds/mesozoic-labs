"""Notebook routing for opt-in SB3 behavior pilots, using the CLI runner.

Canonical curriculum training stays in the notebook's existing chain. Pilot
artifacts live in their own directory and never enter a canonical result bundle.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PILOT_RECIPES = {
    "follow_direction": "trex_follow_direction.toml",
    "follow_direction_speed": "trex_follow_direction_speed.toml",
    "terrain_contact": "trex_terrain_contact.toml",
    "sloped_terrain": "trex_gentle_terrain.toml",
    "bumps_terrain": "trex_bumps_terrain.toml",
    "depressions_terrain": "trex_depressions_terrain.toml",
    "mixed_terrain": "trex_mixed_terrain.toml",
    "combined_terrain": "trex_combined_terrain.toml",
}


def validate_pilot_selection(
    *,
    species: str,
    algorithm: str,
    behavior: str,
    checkpoint: str,
    vecnormalize: str,
    load_mode: str,
    trunk_from: str = "",
    widen_from: str = "",
    retrain_from: str = "",
) -> None:
    """Validate selection before Drive mounts or output-directory creation."""
    if behavior not in PILOT_RECIPES:
        raise ValueError(f"Unknown behavior pilot: {behavior!r}")
    if species != "trex" or algorithm.lower() != "ppo":
        raise ValueError("Behavior pilots require SPECIES='Tyrannosaurus Rex' and ALGORITHM='ppo'.")
    if load_mode not in {"prepare", "resume", "adapt"}:
        raise ValueError("PILOT_LOAD_MODE must be prepare, resume, or adapt.")
    if not checkpoint.strip() or not vecnormalize.strip():
        raise ValueError("Set both PILOT_CHECKPOINT and PILOT_VECNORMALIZE to the explicit matched source files.")
    if trunk_from or widen_from or retrain_from:
        raise ValueError(
            "Clear TRUNK_FROM, WIDEN_FROM and RETRAIN_FROM for a pilot; use PILOT_LOAD_MODE and its source pair."
        )


@dataclass(frozen=True)
class NotebookPilotPlan:
    behavior: str
    recipe_path: Path
    checkpoint_path: Path
    vecnormalize_path: Path
    output_dir: Path
    load_mode: str
    seed: int
    steps: int | None
    eval_only: bool
    eval_episodes: int
    record_video: bool
    video_fps: float

    def argv(self) -> list[str]:
        args = [
            "--config",
            str(self.recipe_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--vecnormalize",
            str(self.vecnormalize_path),
            "--output",
            str(self.output_dir),
            "--seed",
            str(self.seed),
            "--eval-episodes",
            str(self.eval_episodes),
            "--video-fps",
            str(self.video_fps),
        ]
        if self.load_mode != "prepare":
            args.append("--" + self.load_mode)
        if self.steps is not None:
            args.extend(("--steps", str(self.steps)))
        if self.eval_only:
            args.append("--eval-only")
        if self.record_video:
            args.append("--record-video")
        return args


def build_notebook_pilot_plan(
    *,
    repo_root: Path,
    log_base: Path,
    species: str,
    algorithm: str,
    behavior: str,
    checkpoint: str,
    vecnormalize: str,
    load_mode: str = "prepare",
    seed: int | None = None,
    steps: int | None = None,
    quick_test: bool = False,
    eval_only: bool = False,
    eval_episodes: int = 5,
    record_video: bool = True,
    video_fps: float = 25.0,
    run_id: str = "",
) -> NotebookPilotPlan:
    """Resolve a recipe/source pair after Drive is mounted, without writing files."""
    import math

    validate_pilot_selection(
        species=species,
        algorithm=algorithm,
        behavior=behavior,
        checkpoint=checkpoint,
        vecnormalize=vecnormalize,
        load_mode=load_mode,
    )
    for name, value in (("PILOT_SEED", seed), ("PILOT_STEPS", steps), ("PILOT_EVAL_EPISODES", eval_episodes)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer, or None where supported.")
    if seed is not None and seed >= 2**32:
        raise ValueError("PILOT_SEED must be less than 2**32 (the PPO/NumPy seed range).")
    if eval_episodes is None or (record_video and eval_episodes == 0):
        raise ValueError("Recorded pilot replays require at least one evaluation episode.")
    if isinstance(video_fps, bool) or not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("PILOT_VIDEO_FPS must be finite and positive.")
    if run_id and (run_id in {".", ".."} or Path(run_id).name != run_id or "\\" in run_id):
        raise ValueError("PILOT_RUN_ID must be a directory name, not a path.")
    root = Path(repo_root).resolve()
    recipe = root / "configs" / "trex" / "behavior_pilots" / PILOT_RECIPES[behavior]
    if not recipe.is_file():
        raise FileNotFoundError(
            f"Pilot recipe missing in this checkout: {recipe}. Check REPO_REF and restart the runtime after changing code."
        )

    def source_path(value: str) -> Path:
        path = Path(value).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pilot source file not found: {path}. Mount Drive before selecting its files.")
        return path

    model, normalizer = source_path(checkpoint), source_path(vecnormalize)
    actual_seed = secrets.randbelow(2**32) if seed is None else seed
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(log_base).resolve() / "trex" / "ppo" / "behavior_pilots" / behavior / identifier
    return NotebookPilotPlan(
        behavior,
        recipe,
        model,
        normalizer,
        output,
        load_mode,
        actual_seed,
        steps if steps is not None else 4096 if quick_test and not eval_only else None,
        eval_only,
        eval_episodes,
        record_video,
        float(video_fps),
    )


def run_notebook_pilot(plan: NotebookPilotPlan) -> dict[str, Any]:
    """Run the shared CLI path, including exact bundle/identity validation."""
    from environments.trex.scripts.train_behaviors import main

    main(plan.argv())
    report = json.loads((plan.output_dir / "run.json").read_text())
    if report.get("schema") != "mesozoic.behavior-pilot-run/v1" or report.get("canonical_certification") is not False:
        raise ValueError("Pilot runner did not write the expected diagnostic run manifest.")
    return dict(report)


def _replay_display_paths(root: Path, episode: dict[str, Any]) -> dict[str, str]:
    """Find the matched assets in the supplied run, including relocated runs."""
    replay = episode["replay"]
    keys = ("video", "full_map", "local_map")
    directory = root / "replays" / f"episode_{episode['episode']:03d}_seed_{episode['episode_seed']}"
    manifest_path = directory / "manifest.json"
    if "manifest" in replay or manifest_path.is_file():
        # The index records the original output location, but the episode
        # manifest already stores portable filenames. Always use this run's
        # manifest, even when an original copy still exists elsewhere.
        manifest = json.loads(manifest_path.read_text())
        paths = {}
        for key in keys:
            relative = Path(manifest["files"][key]["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Replay {key} must be relative to its episode directory: {relative}")
            paths[key] = directory / relative
    else:
        # Retain support for reports with direct paths and no episode manifest.
        paths = {key: root / Path(replay[key]) for key in keys}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Replay {key} is missing from saved run: {path}")
    return {key: str(path) for key, path in paths.items()}


def display_notebook_pilot(output_dir: Path) -> None:
    """Show saved summaries, videos at their encoded rate, and matching maps."""
    from IPython.display import Image, Video, display

    root = Path(output_dir).resolve()
    report = json.loads((root / "run.json").read_text())
    print(f"Behavior pilot: {report['status']} · seed {report['run_seed']} · {root}")
    training = report.get("training", {})
    print(f"Actual additional steps: {training.get('actual_additional_steps', 0):,}")
    print("Artifacts are behavior diagnostics; no canonical certification is claimed.")
    evaluation = report.get("evaluation", {})
    if evaluation:
        print(
            f"Full horizon: {evaluation['full_horizon_count']}/{evaluation['episode_count']}; falls: {evaluation['fall_count']}"
        )
        for episode in evaluation.get("episodes", []):
            replay = episode.get("replay")
            if not replay:
                continue
            print(f"Episode {episode['episode']} · seed {episode['episode_seed']}")
            paths = _replay_display_paths(root, episode)
            display(Video(filename=paths["video"], embed=True))
            display(Image(filename=paths["full_map"]))
            display(Image(filename=paths["local_map"]))
