"""Read existing mounted-Drive runs and replay an explicitly recorded model pair.

Historical reports remain references, separate from the matched training study.
Existing models, normalizers and source bundles are never modified or relabeled.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _saved_metrics(text: str) -> dict:
    fields = {
        "full_horizon_fraction": "full_horizon_fraction",
        "mean_unsupported_duty": "mean_unsupported_duty",
        "unsupported_duty_ucb": "unsupported_duty_ucb",
        "bilateral_control_window_support": "bilateral support",
        "single_control_window_support": "single support",
        "mean_reward": "reward",
    }
    result = {}
    for key, label in fields.items():
        match = re.search(r"^\s*" + re.escape(label) + r"\s+([-+\d.eE]+)(?:\s|$)", text, re.MULTILINE)
        if match:
            result[key] = float(match.group(1))
    panel = re.search(r"^panel\s+(\d+) episodes, seeds (\d+)-(\d+)", text, re.MULTILINE)
    if panel:
        result.update(n_episodes=int(panel[1]), seed_start=int(panel[2]), seed_end=int(panel[3]))
    gate = re.search(r"^GATE: (PASS|FAIL)\s*$", text, re.MULTILINE)
    if gate:
        result["saved_gate_passed"] = gate[1] == "PASS"
    return result


def _recorded_pair(stage: Path, text: str) -> dict | None:
    paths = {}
    for key, label in (("model_path", "Best model"), ("normalization_path", "VecNormalize")):
        match = re.search(r"^" + label + r":\s*(.+?)\s*$", text, re.MULTILINE)
        if not match:
            return None
        # Saved Colab absolute paths are remapped to this mounted copy by the
        # explicit stage/models suffix; arbitrary sibling-file guesses are not.
        declared = match[1].replace("\\", "/")
        marker = "/" + stage.name + "/"
        relative = declared.rsplit(marker, 1)[-1] if marker in declared else declared
        path = (stage / relative).resolve()
        if not path.is_relative_to((stage / "models").resolve()) or not path.is_file():
            return None
        paths[key] = str(path)
        paths[key + "_sha256"] = _hash(path)
    paths["config_path"] = str(stage / "stage_config.json")
    paths["config_sha256"] = _hash(stage / "stage_config.json")
    paths["pair_declaration"] = str(stage / "stage_summary.txt")
    paths["declaration_sha256"] = _hash(stage / "stage_summary.txt")
    paths["compatibility"] = "Not checked until strict canonical replay"
    return paths


def discover_balance_runs(project_drive_root: Path) -> list[dict]:
    """Inspect the existing mesozoic-labs/logs layout; read only small sidecars."""
    root = Path(project_drive_root).resolve()
    runs = []
    for species in ("compsognathus", "trex"):
        for stage in sorted((root / "logs" / species / "ppo").glob("*/01_stance"), reverse=True):
            config_path = stage / "stage_config.json"
            report_path = stage / "stance_gate_report.txt"
            summary_path = stage / "stage_summary.txt"
            row: dict[str, Any] = {
                "species": species,
                "run_id": stage.parent.name,
                "run_dir": str(stage.parent),
                "research_reference_only": True,
                "source_files": {},
                "checkpoint_pair": None,
            }
            try:
                config = json.loads(config_path.read_text())
                if config.get("species") != species or config.get("stage") not in (1, "stance"):
                    raise ValueError("Saved species/stage does not match the stance directory")
                row["source_commit"] = config.get("git_commit")
                row["training_seed"] = config.get("run", {}).get("seed")
                row["saved_environment"] = config.get("reward_weights")
                row["saved_curriculum"] = config.get("curriculum")
                for path in (config_path, report_path, summary_path):
                    if path.is_file():
                        if path.stat().st_size > 2_000_000:
                            raise ValueError("Unexpectedly large report sidecar: " + path.name)
                        row["source_files"][str(path)] = _hash(path)
                report = report_path.read_text() if report_path.exists() else ""
                row["saved_metrics"] = _saved_metrics(report)
                row["saved_gate_report"] = report
                if summary_path.exists():
                    row["checkpoint_pair"] = _recorded_pair(stage, summary_path.read_text())
                row["status"] = "available" if report else "report_missing"
            except (OSError, ValueError, TypeError) as exc:
                row.update(status="unreadable", error=str(exc))
            runs.append(row)
    return runs


def save_historical_baselines(runs: list[dict], output_path: Path) -> dict:
    result = {
        "schema": "compsognathus.balance-drive-references/v1",
        "historical_reference_only": True,
        "note": "Saved control-window support is not the new physics-sampled 20%-body-weight bilateral criterion.",
        "runs": runs,
    }
    _write(Path(output_path), result)
    return result


def evaluate_saved_baseline(
    pair: dict, output_dir: Path, *, seeds: tuple[int, ...] = tuple(range(11042, 11082))
) -> dict:
    """Evaluate the recorded canonical Compy pair, with all plant checks intact.

    Hashes bind this call to the discovered pair; plant identity is checked by
    the existing replay loader before inference. No old checkpoint initializes
    a changed-filter or changed-reward training arm. Incompatibility propagates.
    """
    from environments.compsognathus.envs.compsognathus_env import CompsognathusEnv
    from environments.compsognathus.experiments.balance_metrics import (
        evaluate_balance_episode,
        summarize_balance_panel,
    )
    from environments.compsognathus.scripts.probe_feet import _load_policy
    from environments.shared.curriculum.stance_gate import StanceGateThresholds
    from environments.shared.plant_contract import validate_environment_plant

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Baseline reset seeds must be nonempty and distinct")
    config_path = Path(pair["config_path"])
    config = json.loads(config_path.read_text())
    if config.get("species") != "compsognathus" or config.get("stage") not in (1, "stance"):
        raise ValueError("Replay requires a saved canonical Compsognathus stance config")
    for key in ("model_path", "normalization_path"):
        if _hash(Path(pair[key])) != pair[key + "_sha256"]:
            raise ValueError("Drive artifact changed after discovery: " + key)
    # Recheck the actual sidecar declaration, not only the supplied dictionary.
    recorded = _recorded_pair(config_path.parent, Path(pair["pair_declaration"]).read_text())
    if recorded != pair:
        raise ValueError("Model pair no longer matches its saved stage summary")
    output_dir = Path(output_dir)
    if output_dir.resolve().is_relative_to(config_path.parent.parent.resolve()):
        raise ValueError("Baseline output must not modify the existing run")
    output_dir.mkdir(parents=True, exist_ok=False)
    normalizer = None
    env = None
    try:
        kwargs = config["reward_weights"]
        model, normalizer, identity = _load_policy(Path(pair["model_path"]), Path(pair["normalization_path"]), kwargs)
        env = CompsognathusEnv(**kwargs)
        validate_environment_plant(env, identity, artifact="saved canonical balance baseline")

        def predict(observation):
            return model.predict(normalizer.normalize_obs(observation), deterministic=True)[0]

        settle = config["curriculum"]["settle_steps"]
        rows = [
            evaluate_balance_episode(
                env, predict, seed, output_dir / "trace.csv" if i == 0 else None, settle_steps=settle
            )
            for i, seed in enumerate(seeds)
        ]
        result = {
            "schema": "compsognathus.saved-balance-replay/v1",
            "historical_reference_only": True,
            "training_steps": 0,
            "pair": pair,
            "config_sha256": _hash(config_path),
            "plant_identity": identity.to_dict(),
            "episodes": rows,
            "summary": summarize_balance_panel(
                rows,
                horizon=kwargs["max_episode_steps"],
                stance_thresholds=StanceGateThresholds.from_curriculum(config["curriculum"]),
            ),
        }
        _write(output_dir / "baseline.json", result)
        return result
    except Exception as exc:
        _write(output_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc), "pair": pair})
        raise
    finally:
        if env is not None:
            env.close()
        if normalizer is not None:
            normalizer.close()
