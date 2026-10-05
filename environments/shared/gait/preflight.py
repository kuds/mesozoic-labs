"""Check declared gait certifications against this build before any training.

A ``locomotion_gait/v2`` stage declares the ``measurement_protocol_sha256``
its certification panel must reproduce. The panel writer refuses a different
hash, but only after the whole training budget has been spent;
:func:`check_gait_stage` recomputes the planned digest (and, given the run's
seeds, the panel's overlap with its training, selection and replay seeds) up
front. ``train_curriculum`` and the training notebook call it for every
gait-gated node of the run's chain before anything is trained, and

    python -m environments.shared.gait.preflight --check

checks every committed stage (CI): a stage whose declared digest no longer
matches the build fails, naming the command that re-plans it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from ..curriculum.gait_gate import GAIT_GATE_KIND
from ..curriculum.gate_schema import GateSchemaError, validate_gate_config
from .identity import protocol_sha256, stage_measurement_protocol
from .seeds import refuse_known_seed_overlaps


class GaitPreflightError(ValueError):
    """A declared gait certification cannot be produced by this build or run."""


def check_gait_stage(
    species_cfg: Any,
    stage: int | str,
    stage_config: Mapping[str, Any],
    *,
    training_seed: int | None = None,
    n_envs: int | None = None,
) -> str | None:
    """The planned protocol digest of a gait-gated stage; ``None`` for any other kind.

    Raises :class:`GaitPreflightError` when the stage's gate is malformed, its
    declared ``measurement_protocol_sha256`` differs from the digest this
    build plans for its panel, or (given ``training_seed`` and ``n_envs``)
    the panel overlaps the run's own training, selection or replay seeds.
    """
    curriculum = stage_config.get("curriculum_kwargs", {})
    if curriculum.get("gate_kind") != GAIT_GATE_KIND:
        return None
    species = species_cfg.species
    try:
        validate_gate_config(stage, curriculum)
        episodes, seed_start = curriculum["min_eval_episodes"], curriculum["gait_panel_seed_start"]
        planned = protocol_sha256(
            stage_measurement_protocol(species_cfg, dict(stage_config), episodes=episodes, seed_start=seed_start)
        )
    except (GateSchemaError, ValueError, OSError, RuntimeError) as error:
        raise GaitPreflightError(f"{species} stage {stage}: gait certification cannot be planned: {error}") from error
    declared = curriculum["measurement_protocol_sha256"]
    if planned != declared:
        raise GaitPreflightError(
            f"{species} stage {stage} declares measurement_protocol_sha256 {declared}, but this build plans "
            f"{planned} for its panel (measurement code, detector, registry, sampling or panel changed); re-plan "
            f"it before training: python -m environments.shared.scripts.gait_report {species} --stage {stage} "
            f"--protocol-only --episodes {episodes} --seed {seed_start}"
        )
    if training_seed is not None:
        try:
            refuse_known_seed_overlaps(
                seed_start,
                episodes,
                training_seed=training_seed,
                training_envs=1 if n_envs is None else n_envs,
                stage=stage,
                species=species,
            )
        except ValueError as error:
            raise GaitPreflightError(f"{species} stage {stage}: {error}; choose another run seed") from error
    return planned


def check_gait_stages(
    species_cfg: Any,
    stage_configs: Mapping[int | str, Mapping[str, Any]],
    stages: Iterable[int | str],
    *,
    training_seed: int | None = None,
    n_envs: int | None = None,
) -> dict[int | str, str]:
    """Check every gait-gated stage of *stages*; one error names every failure."""
    planned: dict[int | str, str] = {}
    failures: list[str] = []
    for stage in stages:
        try:
            digest = check_gait_stage(
                species_cfg, stage, stage_configs[stage], training_seed=training_seed, n_envs=n_envs
            )
        except GaitPreflightError as error:
            failures.append(str(error))
            continue
        if digest is not None:
            planned[stage] = digest
    if failures:
        raise GaitPreflightError("; ".join(failures))
    return planned


def main(argv: list[str] | None = None) -> int:
    from ..config import load_all_stages
    from ..species_registry import SPECIES_FACTORIES, get_species_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        required=True,
        help="verify every committed stage declaring the gait gate against this build",
    )
    parser.add_argument("species", nargs="*", help="limit the check to these species (default: all)")
    args = parser.parse_args(argv)
    # Canonical ids, each once (the registry also maps legacy aliases).
    trainable = sorted({get_species_config(name).species for name in SPECIES_FACTORIES})
    try:
        selected = sorted({get_species_config(name).species for name in args.species}) or trainable
    except ValueError as error:
        parser.error(f"unknown species: {error}; known: {trainable}")
    checked = 0
    failures: list[str] = []
    for species in selected:
        species_cfg = get_species_config(species)
        configs = load_all_stages(species)
        for stage, config in configs.items():
            try:
                digest = check_gait_stage(species_cfg, stage, config)
            except GaitPreflightError as error:
                failures.append(str(error))
                continue
            if digest is not None:
                checked += 1
                print(f"{species} stage {stage}: {GAIT_GATE_KIND} protocol {digest} matches this build")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if not failures and not checked:
        print(f"No committed stage declares {GAIT_GATE_KIND}; nothing to check.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
