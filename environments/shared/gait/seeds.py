"""Bind a gait panel to recorded training, selection and replay seed exclusions.

Only JSON archive metadata is read; this module never deserializes a saved
policy. These known exclusions cannot prove that a human never used the
panel while tuning. A certification panel must also remain locked and unused
for development, as required by the evaluation protocol: report-only panels
roll the separate development block (``constants.DEVELOPMENT_GAIT_SEED_START``).
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..file_io import read_json_object
from ..result_bundle.hashing import canonical_json_sha256

SEED_PROVENANCE_SCHEMA = "mesozoic.gait-seed-provenance/v2"
#: Roles a certification panel may share seeds with: the certification panel
#: itself and the publication protocol's evaluations of the selected
#: checkpoint (``result_bundle.provenance`` writes exactly these names). Every
#: other recorded role is a used seed (deny by default), whatever its name.
_PUBLICATION_ROLES = frozenset({"certification_panel", "publication_evaluation"})
_ADDITIONAL_EVALUATION_ROLE = re.compile(r"additional_evaluation_[0-9]+")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"gait seed provenance {name} must be an integer >= {minimum}")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = read_json_object(path)
    if value is None:
        raise ValueError(f"gait seed provenance {path.name} must contain a JSON object")
    return value


def _role_block(role: str, value: Any) -> tuple[int, int]:
    """``[start, stop)`` of a recorded role: an integer is one seed, ``{start, episodes}`` a block.

    ``initialize_result_bundle`` records integers only, so an in-repo bundle
    never holds a block; one written by other tooling is excluded whole.
    """
    if isinstance(value, Mapping):
        if set(value) != {"start", "episodes"}:
            raise ValueError(f"gait seed provenance seed_roles.{role} block must hold exactly start and episodes")
        start = _integer(value["start"], f"seed_roles.{role}.start")
        return start, start + _integer(value["episodes"], f"seed_roles.{role}.episodes", minimum=1)
    seed = _integer(value, f"seed_roles.{role}")
    return seed, seed + 1


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
    seed_start: int,
    episodes: int,
    *,
    training_seed: int,
    training_envs: int,
    stage: int | str,
    species: str | None,
    policy_seed: int | None = None,
) -> list[int]:
    """Refuse a panel that intersects the run's own training, selection or replay seeds.

    Shared by the certification-time binding below and the pre-training
    check, so a seed choice that overlaps the panel is refused before a
    training budget is spent. *policy_seed* is a seed the stage's algorithm
    block names (decision D-D11, ``--override ppo.seed=N``): the policy is
    built under it, and Stable-Baselines3 then resets the training
    environments at ``policy_seed+rank`` as well as the run's ``seed+rank``.
    Returns the stage's replay seeds.
    """
    panel_stop = seed_start + episodes
    if max(seed_start, training_seed) < min(panel_stop, training_seed + training_envs):
        raise ValueError("gait certification panel overlaps the recorded training environment seed range")
    if policy_seed is not None and max(seed_start, policy_seed) < min(panel_stop, policy_seed + training_envs):
        raise ValueError("gait certification panel overlaps the training environment seeds of the policy seed")
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


def checkpoint_seed_provenance(
    model_path: str | Path,
    *,
    seed_start: int,
    episodes: int,
    stage: int | str,
    species: str | None = None,
) -> dict[str, Any]:
    """Refuse missing seed evidence and overlaps with known used seed roles.

    The stage's recorded ``run.seed`` and ``run.n_envs`` describe reset seeds
    ``seed+rank``. Both repository SB3 trainers use one selection environment
    at ``seed+1000``, and the stage replay videos roll at
    ``evaluation.replay_seed(seed, stage)``. The checkpoint's safe JSON
    metadata must corroborate the environment count and the policy seed: the
    seed the recorded algorithm block (``hyperparameters``) names, or else
    ``run.seed``. A block seed's ``seed+rank`` resets are used seeds too. A
    nearby optional run provenance contributes every other recorded seed
    role, as a single seed or a ``{start, episodes}`` block; only the
    certification panel and the publication evaluations may share the
    panel's seeds. An unrecognised role is a used seed (deny by default),
    whatever its name.

    The returned payload and its digest are portable across copied bundles;
    unrelated configuration/provenance changes do not alter this seed binding.
    """
    seed_start = _integer(seed_start, "panel seed_start")
    episodes = _integer(episodes, "panel episodes", minimum=1)
    if isinstance(stage, bool) or not isinstance(stage, (int, str)):
        raise ValueError("gait seed provenance requires the panel's stage reference")
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
    record = _read_object(config)
    run = record.get("run")
    if not isinstance(run, dict):
        raise ValueError("gait seed provenance requires stage_config.json run seed and n_envs")
    training_seed = _integer(run.get("seed"), "run.seed")
    training_envs = _integer(run.get("n_envs"), "run.n_envs", minimum=1)
    # D-D11: a seed the stage's algorithm block names builds the policy, and
    # the checkpoint records it; the run seed still seeds the environments.
    hyperparameters = record.get("hyperparameters")
    block_seed = hyperparameters.get("seed") if isinstance(hyperparameters, Mapping) else None
    policy_seed = training_seed if block_seed is None else _integer(block_seed, "hyperparameters.seed")
    model_seed, model_envs = _checkpoint_seed_fields(model)
    if model_seed != policy_seed or model_envs != training_envs:
        raise ValueError("gait seed provenance checkpoint seed/n_envs do not match the recorded stage run")

    replay_seeds = refuse_known_seed_overlaps(
        seed_start,
        episodes,
        training_seed=training_seed,
        training_envs=training_envs,
        stage=stage,
        species=species,
        policy_seed=policy_seed,
    )
    panel_stop = seed_start + episodes
    training_stop = training_seed + training_envs
    selection_seed = training_seed + 1000

    def overlaps(start: int, stop: int) -> bool:
        return max(seed_start, start) < min(panel_stop, stop)

    additional_roles: dict[str, Any] = {}
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
            start, stop = _role_block(role, value)
            if role in _PUBLICATION_ROLES or _ADDITIONAL_EVALUATION_ROLE.fullmatch(role):
                continue
            if role == "training" and (isinstance(value, Mapping) or start != training_seed):
                raise ValueError("gait seed provenance training role does not match the recorded stage run")
            if overlaps(start, stop):
                raise ValueError(f"gait certification panel overlaps the recorded {role} seed")
            # A later-created ordinary run provenance must not change a
            # binding merely by repeating exclusions already proved from
            # the stage and checkpoint. Distinct known used seeds remain
            # part of the portable evidence.
            proven = training_seed <= start and stop <= training_stop
            proven |= policy_seed <= start and stop <= policy_seed + training_envs
            proven |= stop - start == 1 and (start == selection_seed or start in replay_seeds)
            if not proven:
                additional_roles[role] = dict(value) if isinstance(value, Mapping) else start

    payload = {
        "schema": SEED_PROVENANCE_SCHEMA,
        "training_seed": training_seed,
        "training_envs": training_envs,
        "checkpoint_model_seed": model_seed,
        "checkpoint_n_envs": model_envs,
        "selection_seed": selection_seed,
        "replay_seeds": replay_seeds,
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
