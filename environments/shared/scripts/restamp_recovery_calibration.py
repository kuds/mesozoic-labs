"""Restamp a recovery calibration across an interface-only plant revision.

Decision D-C2 (BEHAVIOR_RECIPES_PLAN §4.6, Phase C): the 3-dim command
segment appended to every species observation moves each plant's
``policy_interface_sha256`` and therefore every stage's ``task_sha256``
(``task_fingerprint.py`` folds the interface hash in), so
``load_recovery_calibration`` refuses the committed compsognathus
calibrations with "plant identity changed; recalibrate this plant".  A
recalibration would re-measure a quiet reference the change cannot have
moved: the fixed-command nulls the profile was measured from never read
the observation, and the seeded reset draw stream is byte-identical
(pinned by ``environments/shared/tests/fixtures/phase_c_reset_golden.json``).
This tool RESTAMPS the profile instead — plant identity, task hash and a
``restamp_history`` entry naming what was replaced and why — and refuses
everything a restamp cannot honestly cover:

* the measured ``recovery_env_kwargs`` must still equal the stage's
  committed ``[env]`` block (the task itself is unchanged);
* the current plant's ``physics_sha256`` must equal the profile's (a
  physics change moves the quiet reference and needs a real
  recalibration, not a restamp);
* the profile must be a valid recovery calibration for the named model
  (its ``species`` and ``embodiment`` both name it; decision D-D33).

Idempotent: a profile whose identity and task hash are already current is
left untouched — a second run is a byte no-op and appends no history.
``restamped_at_commit`` records ``git rev-parse HEAD`` of the repository
the tool runs in, or ``"unknown"`` (said so in the log) when git is
unavailable — never a refusal on that account (amendment A13).

The recorded ``reason`` defaults to :data:`RESTAMP_REASON` -- the Phase C
justification.  A later interface-only revision that reuses this tool MUST
pass its own ``--reason`` (the entry names the revision it covers; nothing
else in the history distinguishes one restamp from the next).

``--migrate-identity --reason TEXT`` is the decision D-D33 transition, and
nothing else: it migrates ONLY a schema-v1 profile (named by the species ID
of its date, e.g. ``compsognathus_robot``) found at the model's
``configs/<species>/<embodiment>/recovery_calibration.json``.  It refuses
unless the recorded ``physics_sha256`` equals the model's current physics
digest (the identity moved, the plant did not) and ``recovery_env_kwargs``
equal the committed recovery ``[env]``, as a restamp does; then it writes
schema v2 with ``species`` and ``embodiment``, the current plant identity
(v2) and task hash, and the ``calibration_id`` renamed from
``<recorded species>-...`` to ``<species>-<embodiment>-...`` (an identifier
copied into every recovery-evaluation spec); it points
``provenance.measurement_artifact`` at the measurement's file under
``environments/<species>/data/`` (which must exist; the artifact keeps its
recorded content); and it appends a ``restamp_history`` entry whose
``previous_identity`` keeps the recorded schema, species and
``calibration_id``.  No old-to-new table: the model is the directory the
file sits in, checked by the physics digest.  On a profile already migrated
and current it is a byte no-op; any other profile is refused.

Run: ``python -m environments.shared.scripts.restamp_recovery_calibration
[--model-id compsognathus_longipes anatomical --model-id compsognathus_longipes robot] [--configs-root DIR]
[--reason TEXT] [--migrate-identity]``.  With no ``--model-id``, every model whose
``<configs-root>/<species>/<embodiment>/recovery_calibration.json`` exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT

logger = logging.getLogger(__name__)

RESTAMP_REASON = (
    "BEHAVIOR_RECIPES_PLAN §4.6 Phase C interface bump; fixed-command nulls do not read the observation "
    "and the reset draw stream is unchanged (tests/fixtures/phase_c_reset_golden.json)"
)


class RestampError(RuntimeError):
    """The profile cannot honestly be restamped; it needs a real recalibration."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _render(profile: dict[str, Any]) -> str:
    """The committed on-disk form: two-space indent, insertion order, trailing newline."""
    return json.dumps(profile, indent=2, ensure_ascii=False) + "\n"


def current_commit(repository: Path = _REPOSITORY_ROOT) -> str:
    """``git rev-parse HEAD`` of *repository*, or ``"unknown"`` when git cannot answer."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning(
            "git rev-parse HEAD failed in %s (%s); recording restamped_at_commit = 'unknown'", repository, exc
        )
        return "unknown"


def restamp_recovery_calibration(
    model_id: ModelId,
    *,
    path: str | Path | None = None,
    stage: str = "recovery",
    commit: str | None = None,
    reason: str = RESTAMP_REASON,
) -> bool:
    """Restamp *model_id*'s recovery calibration in place; ``True`` when the file changed.

    *path* defaults to ``configs/<species>/<embodiment>/recovery_calibration.json``;
    *reason* is what the appended ``restamp_history`` entry records.
    Raises :class:`RestampError` on every refusal (a profile of another
    model, or one that does not name its model whole, included); never
    writes on one.
    """
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.recovery_calibration import CONFIGS_ROOT, PROFILE_SCHEMA
    from environments.shared.task_fingerprint import stage_task_fingerprint

    if not isinstance(reason, str) or not reason.strip():
        raise RestampError("the restamp reason must be a non-empty string")
    profile_path = Path(path) if path is not None else model_id.path(CONFIGS_ROOT) / "recovery_calibration.json"
    try:
        original_text = profile_path.read_text(encoding="utf-8")
        profile = json.loads(original_text)
    except (OSError, ValueError) as exc:
        raise RestampError(f"cannot load recovery calibration {profile_path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise RestampError(f"{profile_path} must hold a JSON object")
    if profile.get("schema") != PROFILE_SCHEMA:
        raise RestampError(
            f"{profile_path} has the wrong recovery calibration schema {profile.get('schema')!r}; expected "
            f"{PROFILE_SCHEMA!r} (a schema-v1 profile predates decision D-D33: --migrate-identity migrates it)"
        )
    try:
        same_model = model_id.matches(profile)
    except ValueError as exc:
        raise RestampError(f"{profile_path} does not name its model: {exc}") from exc
    if not same_model:
        raise RestampError(f"{profile_path} is not a recovery calibration of {model_id}")

    measured_env = profile.get("recovery_env_kwargs")
    stage_env = load_stage_config(model_id, stage)["env_kwargs"]
    if not isinstance(measured_env, dict) or _canonical(measured_env) != _canonical(stage_env):
        raise RestampError(
            f"{profile_path} recovery_env_kwargs differ from the committed {model_id} {stage!r} [env] block; "
            "a changed task needs a real recalibration, not a restamp"
        )

    recorded_identity = profile.get("plant_identity")
    if not isinstance(recorded_identity, dict):
        raise RestampError(f"{profile_path} records no plant_identity to restamp from")
    current_identity = current_plant_identity(model_id).to_dict()
    if recorded_identity.get("physics_sha256") != current_identity["physics_sha256"]:
        raise RestampError(
            f"{profile_path} was measured on physics {recorded_identity.get('physics_sha256')!r} but the current "
            f"{model_id} plant is {current_identity['physics_sha256']!r}: a physics change moves the quiet reference "
            "and needs a real recalibration, not a restamp"
        )

    fingerprint = stage_task_fingerprint(model_id, stage, env_kwargs=measured_env, plant_identity=current_identity)
    task_sha256 = fingerprint["task_sha256"]

    if recorded_identity == current_identity and profile.get("task_sha256") == task_sha256:
        logger.info("%s already carries the current plant identity and task hash; nothing to restamp", profile_path)
        return False

    history = profile.get("restamp_history")
    if history is None:
        history = []
    if not isinstance(history, list):
        raise RestampError(f"{profile_path} restamp_history must be a list")
    history.append(
        {
            "restamped_at_commit": current_commit() if commit is None else commit,
            "previous_policy_interface_revision": recorded_identity.get("policy_interface_revision"),
            "previous_policy_interface_sha256": recorded_identity.get("policy_interface_sha256"),
            "previous_task_sha256": profile.get("task_sha256"),
            "reason": reason,
        }
    )
    profile["plant_identity"] = current_identity
    profile["task_sha256"] = task_sha256
    profile["restamp_history"] = history

    rendered = _render(profile)
    if rendered == original_text:  # pragma: no cover - identity/hash differed above, so this cannot hold
        return False
    profile_path.write_text(rendered, encoding="utf-8")
    logger.info(
        "Restamped %s: policy interface r%s -> r%s, task %s -> %s",
        profile_path,
        recorded_identity.get("policy_interface_revision"),
        current_identity["policy_interface_revision"],
        history[-1]["previous_task_sha256"],
        task_sha256,
    )
    return True


#: The one schema ``--migrate-identity`` migrates from (decision D-D33).
_PRE_D_D33_PROFILE_SCHEMA = "mesozoic.recovery-calibration/v1"


def migrate_recovery_calibration_identity(
    model_id: ModelId,
    *,
    reason: str,
    path: str | Path | None = None,
    stage: str = "recovery",
    commit: str | None = None,
    repository_root: Path = _REPOSITORY_ROOT,
) -> bool:
    """Migrate *model_id*'s schema-v1 profile to v2 in place (decision D-D33); ``True`` when the file changed.

    The module docstring lists every guard.  A profile already at the current
    schema is handed to :func:`restamp_recovery_calibration`'s checks and
    must need nothing: a byte no-op, or a refusal.  Raises
    :class:`RestampError` on every refusal; never writes on one.
    """
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.recovery_calibration import CONFIGS_ROOT, PROFILE_SCHEMA
    from environments.shared.task_fingerprint import stage_task_fingerprint

    if not isinstance(reason, str) or not reason.strip():
        raise RestampError("the migration reason must be a non-empty string")
    profile_path = Path(path) if path is not None else model_id.path(CONFIGS_ROOT) / "recovery_calibration.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RestampError(f"cannot load recovery calibration {profile_path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise RestampError(f"{profile_path} must hold a JSON object")
    if profile.get("schema") == PROFILE_SCHEMA:
        # Already migrated: a byte no-op when it is this model's current
        # profile, refused otherwise (a plain restamp, or a recalibration).
        try:
            same_model = model_id.matches(profile)
        except ValueError as exc:
            raise RestampError(f"{profile_path} does not name its model: {exc}") from exc
        measured_env = profile.get("recovery_env_kwargs")
        identity = current_plant_identity(model_id).to_dict()
        if (
            same_model
            and isinstance(measured_env, dict)
            and _canonical(measured_env) == _canonical(load_stage_config(model_id, stage)["env_kwargs"])
            and profile.get("plant_identity") == identity
            and profile.get("task_sha256")
            == stage_task_fingerprint(model_id, stage, env_kwargs=measured_env, plant_identity=identity)["task_sha256"]
        ):
            logger.info("%s is already migrated and current; nothing to migrate", profile_path)
            return False
        raise RestampError(
            f"{profile_path} is already schema {PROFILE_SCHEMA!r} but is not {model_id}'s current profile; "
            "--migrate-identity migrates only a schema-v1 profile"
        )
    if profile.get("schema") != _PRE_D_D33_PROFILE_SCHEMA:
        raise RestampError(
            f"{profile_path} declares schema {profile.get('schema')!r}; --migrate-identity migrates only "
            f"{_PRE_D_D33_PROFILE_SCHEMA!r}"
        )
    if "embodiment" in profile:
        raise RestampError(f"{profile_path} is schema v1 but already records an embodiment; refusing to guess")
    recorded_species = profile.get("species")
    recorded_id = profile.get("calibration_id")
    if not isinstance(recorded_species, str) or not recorded_species:
        raise RestampError(f"{profile_path} records no species to migrate from")
    if not isinstance(recorded_id, str) or not recorded_id.startswith(f"{recorded_species}-"):
        raise RestampError(
            f"{profile_path} calibration_id {recorded_id!r} does not start with its recorded species "
            f"{recorded_species!r}; refusing to rename it"
        )

    measured_env = profile.get("recovery_env_kwargs")
    stage_env = load_stage_config(model_id, stage)["env_kwargs"]
    if not isinstance(measured_env, dict) or _canonical(measured_env) != _canonical(stage_env):
        raise RestampError(
            f"{profile_path} recovery_env_kwargs differ from the committed {model_id} {stage!r} [env] block; "
            "a changed task needs a real recalibration, not an identity migration"
        )
    recorded_identity = profile.get("plant_identity")
    if not isinstance(recorded_identity, dict):
        raise RestampError(f"{profile_path} records no plant_identity to migrate from")
    current_identity = current_plant_identity(model_id).to_dict()
    if recorded_identity.get("physics_sha256") != current_identity["physics_sha256"]:
        raise RestampError(
            f"{profile_path} was measured on physics {recorded_identity.get('physics_sha256')!r} but the current "
            f"{model_id} plant is {current_identity['physics_sha256']!r}: this profile is not this model's"
        )

    provenance = profile.get("provenance")
    artifact = provenance.get("measurement_artifact") if isinstance(provenance, dict) else None
    if not isinstance(artifact, str) or not artifact:
        raise RestampError(f"{profile_path} records no provenance.measurement_artifact")
    moved_artifact = Path("environments") / model_id.species / "data" / Path(artifact).name
    if not (repository_root / moved_artifact).is_file():
        raise RestampError(f"{profile_path} measurement artifact {artifact!r} is not at {moved_artifact.as_posix()}")

    fingerprint = stage_task_fingerprint(model_id, stage, env_kwargs=measured_env, plant_identity=current_identity)
    history = profile.get("restamp_history")
    if history is None:
        history = []
    if not isinstance(history, list):
        raise RestampError(f"{profile_path} restamp_history must be a list")
    history.append(
        {
            "restamped_at_commit": current_commit() if commit is None else commit,
            "previous_identity": {
                "schema": profile["schema"],
                "species": recorded_species,
                "calibration_id": recorded_id,
            },
            "previous_policy_interface_revision": recorded_identity.get("policy_interface_revision"),
            "previous_policy_interface_sha256": recorded_identity.get("policy_interface_sha256"),
            "previous_task_sha256": profile.get("task_sha256"),
            "reason": reason,
        }
    )
    migrated: dict[str, Any] = {}
    for key, value in profile.items():
        if key == "schema":
            migrated[key] = PROFILE_SCHEMA
        elif key == "species":
            migrated.update(model_id.record())
        elif key == "calibration_id":
            migrated[key] = (
                f"{model_id.species}-{model_id.embodiment}-{recorded_id.removeprefix(recorded_species + '-')}"
            )
        elif key == "plant_identity":
            migrated[key] = current_identity
        elif key == "task_sha256":
            migrated[key] = fingerprint["task_sha256"]
        elif key == "provenance":
            migrated[key] = {**value, "measurement_artifact": moved_artifact.as_posix()}
        elif key == "restamp_history":
            migrated[key] = history
        else:
            migrated[key] = value
    migrated.setdefault("restamp_history", history)
    profile_path.write_text(_render(migrated), encoding="utf-8")
    logger.info(
        "Migrated %s to %s: %s -> %s, calibration_id %s -> %s, policy interface r%s -> r%s",
        profile_path,
        PROFILE_SCHEMA,
        recorded_species,
        model_id,
        recorded_id,
        migrated["calibration_id"],
        recorded_identity.get("policy_interface_revision"),
        current_identity["policy_interface_revision"],
    )
    return True


def main(argv: "list[str] | None" = None) -> int:
    from environments.shared.recovery_calibration import CONFIGS_ROOT
    from environments.shared.species_names import model_ids, require_model_id

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--model-id",
        nargs=2,
        action="append",
        metavar=("SPECIES", "EMBODIMENT"),
        dest="model_ids",
        help="a model whose recovery calibration to restamp; repeatable (default: every model with a "
        "recovery_calibration.json under the configs root)",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=None,
        help="read/write <configs-root>/<species>/<embodiment>/recovery_calibration.json instead of the "
        "committed configs/",
    )
    parser.add_argument(
        "--reason",
        default=RESTAMP_REASON,
        help="the restamp_history reason to record (default: the Phase C justification; a later "
        "interface-only revision must name its own)",
    )
    parser.add_argument(
        "--migrate-identity",
        action="store_true",
        help="migrate a schema-v1 profile to the model's identity (decision D-D33; requires --reason); "
        "a byte no-op on a migrated, current profile",
    )
    args = parser.parse_args(argv)
    if args.migrate_identity and (args.reason == RESTAMP_REASON or not args.reason.strip()):
        parser.error("--migrate-identity requires its own --reason")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configs_root = CONFIGS_ROOT if args.configs_root is None else args.configs_root
    targets: list[ModelId] = []
    for species, embodiment in args.model_ids or ():
        try:
            targets.append(require_model_id(species, embodiment))
        except ValueError as exc:
            parser.error(str(exc))
    if not args.model_ids:
        # Built from the manifest's models (review L2): a calibration file no
        # model owns is never picked up by a glob.
        targets = [
            model_id
            for model_id in model_ids()
            if (model_id.path(configs_root) / "recovery_calibration.json").is_file()
        ]
    changed = 0
    for model_id in targets:
        path = None if args.configs_root is None else model_id.path(args.configs_root) / "recovery_calibration.json"
        try:
            if args.migrate_identity:
                changed += migrate_recovery_calibration_identity(model_id, path=path, reason=args.reason)
            elif restamp_recovery_calibration(model_id, path=path, reason=args.reason):
                changed += 1
        except RestampError as exc:
            logger.error("Refusing to %s %s: %s", "migrate" if args.migrate_identity else "restamp", model_id, exc)
            return 1
    print(f"{'migrated' if args.migrate_identity else 'restamped'} {changed} of {len(targets)} calibration(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
