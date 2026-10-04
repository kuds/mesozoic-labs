"""Bind a gait panel to recorded training and selection seed exclusions.

Only JSON archive metadata is read; this module never deserializes a saved
policy. These known exclusions cannot prove that a human never used the
panel while tuning. A certification panel must also remain locked and unused
for development, as required by the evaluation protocol.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from ..file_io import read_json_object
from ..result_bundle.hashing import canonical_json_sha256

SEED_PROVENANCE_SCHEMA = "mesozoic.gait-seed-provenance/v1"


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"gait seed provenance {name} must be an integer >= {minimum}")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = read_json_object(path)
    if value is None:
        raise ValueError(f"gait seed provenance {path.name} must contain a JSON object")
    return value


def stage_replay_seeds(training_seed: int, stage: int | str, *, species: str | None = None) -> list[int]:
    """Seeds of the stage replay videos a developer watches (``evaluation.replay_seed``).

    ``generate_stage_artifacts`` rolls them at ``replay_seed(run.seed, stage)``
    for whichever spelling of the stage its caller passed, so every spelling
    the species' manifest gives the stage (legacy number and id) is excluded.
    """
    from ..evaluation import replay_seed
    from ..stage_manifest import StageManifestError, load_stage_manifest

    spellings: set[int | str] = {stage}
    if species is not None:
        try:
            entry = load_stage_manifest(species).resolve(stage)
        except (StageManifestError, OSError, ValueError):
            pass
        else:
            spellings |= {entry.reference, entry.id}
    return sorted({replay_seed(training_seed, spelling) for spelling in spellings})


def refuse_known_seed_overlaps(
    seed_start: int, episodes: int, *, training_seed: int, training_envs: int, stage: int | str, species: str | None
) -> list[int]:
    """Refuse a panel that intersects the run's own training, selection or replay seeds.

    Shared by the certification-time binding below and the pre-training
    check, so a seed choice that overlaps the panel is refused before a
    training budget is spent. Returns the stage's replay seeds.
    """
    panel_stop = seed_start + episodes
    if max(seed_start, training_seed) < min(panel_stop, training_seed + training_envs):
        raise ValueError("gait certification panel overlaps the recorded training environment seed range")
    if seed_start <= training_seed + 1000 < panel_stop:
        raise ValueError("gait certification panel overlaps the checkpoint-selection environment seed")
    replay_seeds = stage_replay_seeds(training_seed, stage, species=species)
    if any(seed_start <= seed < panel_stop for seed in replay_seeds):
        raise ValueError("gait certification panel overlaps the stage replay video seed")
    return replay_seeds


def _checkpoint_seed_fields(model: Path) -> tuple[int, int]:
    """Read only scalar JSON metadata, without executing serialized members."""
    try:
        with zipfile.ZipFile(model) as archive:
            if archive.namelist().count("data") != 1:
                raise ValueError("checkpoint must contain exactly one SB3 data JSON member")
            data = json.loads(archive.read("data"))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"gait seed provenance cannot read checkpoint JSON metadata: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("gait seed provenance checkpoint data must be a JSON object")
    return _integer(data.get("seed"), "checkpoint seed"), _integer(data.get("n_envs"), "checkpoint n_envs", minimum=1)


def checkpoint_seed_provenance(model_path: str | Path, *, seed_start: int, episodes: int) -> dict[str, Any]:
    """Refuse missing seed evidence and overlaps with known used seed roles.

    The stage's recorded ``run.seed`` and ``run.n_envs`` describe reset seeds
    ``seed+rank``. Both repository SB3 trainers use one selection environment
    at ``seed+1000``. The checkpoint's safe JSON metadata must corroborate the
    seed and environment count. A nearby optional run provenance contributes
    any explicitly recorded training, selection, calibration or development
    seed roles. Publication and certification roles may refer to this panel.

    The returned payload and its digest are portable across copied bundles;
    unrelated configuration/provenance changes do not alter this seed binding.
    """
    seed_start = _integer(seed_start, "panel seed_start")
    episodes = _integer(episodes, "panel episodes", minimum=1)
    model = Path(model_path)
    if not model.is_file() and not model.name.endswith(".zip"):
        model = model.with_name(model.name + ".zip")
    config = next(
        (
            candidate
            for candidate in (model.parent / "stage_config.json", model.parent.parent / "stage_config.json")
            if candidate.is_file()
        ),
        None,
    )
    if config is None:
        raise ValueError("gait seed provenance requires the checkpoint's saved stage_config.json")
    run = _read_object(config).get("run")
    if not isinstance(run, dict):
        raise ValueError("gait seed provenance requires stage_config.json run seed and n_envs")
    training_seed = _integer(run.get("seed"), "run.seed")
    training_envs = _integer(run.get("n_envs"), "run.n_envs", minimum=1)
    model_seed, model_envs = _checkpoint_seed_fields(model)
    if model_seed != training_seed or model_envs != training_envs:
        raise ValueError("gait seed provenance checkpoint seed/n_envs do not match the recorded stage run")

    panel_stop = seed_start + episodes
    training_stop = training_seed + training_envs
    if max(seed_start, training_seed) < min(panel_stop, training_stop):
        raise ValueError("gait certification panel overlaps the recorded training environment seed range")
    selection_seed = training_seed + 1000
    if seed_start <= selection_seed < panel_stop:
        raise ValueError("gait certification panel overlaps the checkpoint-selection environment seed")

    additional_roles: dict[str, int] = {}
    provenance = next(
        (
            candidate
            for candidate in (config.parent / "provenance.json", config.parent.parent / "provenance.json")
            if candidate.is_file()
        ),
        None,
    )
    if provenance is not None:
        roles = _read_object(provenance).get("seed_roles")
        if roles is not None and not isinstance(roles, dict):
            raise ValueError("gait seed provenance seed_roles must be a JSON object")
        for role, value in (roles or {}).items():
            if not isinstance(role, str):
                raise ValueError("gait seed provenance role names must be strings")
            if role == "training" or any(
                token in role.lower() for token in ("selection", "calibration", "development")
            ):
                role_seed = _integer(value, f"seed_roles.{role}")
                if role == "training" and role_seed != training_seed:
                    raise ValueError("gait seed provenance training role does not match the recorded stage run")
                if seed_start <= role_seed < panel_stop:
                    raise ValueError(f"gait certification panel overlaps the recorded {role} seed")
                # A later-created ordinary run provenance must not change a
                # binding merely by repeating exclusions already proved from
                # the stage and checkpoint. Distinct known used seeds remain
                # part of the portable evidence.
                if not training_seed <= role_seed < training_stop and role_seed != selection_seed:
                    additional_roles[role] = role_seed

    payload = {
        "schema": SEED_PROVENANCE_SCHEMA,
        "training_seed": training_seed,
        "training_envs": training_envs,
        "checkpoint_model_seed": model_seed,
        "checkpoint_n_envs": model_envs,
        "selection_seed": selection_seed,
        "panel_seed_start": seed_start,
        "panel_episodes": episodes,
        "additional_seed_roles": additional_roles,
        "seed_fields_sha256": canonical_json_sha256(
            {
                "run": {"seed": training_seed, "n_envs": training_envs},
                "checkpoint": {"seed": model_seed, "n_envs": model_envs},
            }
        ),
    }
    return {**payload, "seed_provenance_sha256": canonical_json_sha256(payload)}
