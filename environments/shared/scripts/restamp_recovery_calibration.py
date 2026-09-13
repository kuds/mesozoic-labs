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
* the profile must be a valid recovery calibration for the named species.

Idempotent: a profile whose identity and task hash are already current is
left untouched — a second run is a byte no-op and appends no history.
``restamped_at_commit`` records ``git rev-parse HEAD`` of the repository
the tool runs in, or ``"unknown"`` (said so in the log) when git is
unavailable — never a refusal on that account (amendment A13).

The recorded ``reason`` defaults to :data:`RESTAMP_REASON` -- the Phase C
justification.  A later interface-only revision that reuses this tool MUST
pass its own ``--reason`` (the entry names the revision it covers; nothing
else in the history distinguishes one restamp from the next).

Run: ``python -m environments.shared.scripts.restamp_recovery_calibration
[--species compsognathus --species compsognathus_robot] [--configs-root DIR]
[--reason TEXT]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SPECIES = ("compsognathus", "compsognathus_robot")
RESTAMP_REASON = (
    "BEHAVIOR_RECIPES_PLAN §4.6 Phase C interface bump; fixed-command nulls do not read the observation "
    "and the reset draw stream is unchanged (tests/fixtures/phase_c_reset_golden.json)"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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
    species: str,
    *,
    path: str | Path | None = None,
    stage: str = "recovery",
    commit: str | None = None,
    reason: str = RESTAMP_REASON,
) -> bool:
    """Restamp *species*' recovery calibration in place; ``True`` when the file changed.

    *path* defaults to ``configs/<species>/recovery_calibration.json``;
    *reason* is what the appended ``restamp_history`` entry records.
    Raises :class:`RestampError` on every refusal (an unknown species
    included); never writes on one.
    """
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.recovery_calibration import CONFIGS_ROOT, PROFILE_SCHEMA
    from environments.shared.species_names import resolve_species_id
    from environments.shared.task_fingerprint import derive_stage_task_fingerprint

    try:
        species = resolve_species_id(species)
    except ValueError as exc:
        raise RestampError(str(exc)) from exc
    if not isinstance(reason, str) or not reason.strip():
        raise RestampError("the restamp reason must be a non-empty string")
    profile_path = Path(path) if path is not None else CONFIGS_ROOT / species / "recovery_calibration.json"
    try:
        original_text = profile_path.read_text(encoding="utf-8")
        profile = json.loads(original_text)
    except (OSError, ValueError) as exc:
        raise RestampError(f"cannot load recovery calibration {profile_path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise RestampError(f"{profile_path} must hold a JSON object")
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("species") != species:
        raise RestampError(f"{profile_path} has the wrong recovery calibration schema or species")

    measured_env = profile.get("recovery_env_kwargs")
    stage_env = load_stage_config(species, stage)["env_kwargs"]
    if not isinstance(measured_env, dict) or _canonical(measured_env) != _canonical(stage_env):
        raise RestampError(
            f"{profile_path} recovery_env_kwargs differ from the committed {species} {stage!r} [env] block; "
            "a changed task needs a real recalibration, not a restamp"
        )

    recorded_identity = profile.get("plant_identity")
    if not isinstance(recorded_identity, dict):
        raise RestampError(f"{profile_path} records no plant_identity to restamp from")
    current_identity = current_plant_identity(species).to_dict()
    if recorded_identity.get("physics_sha256") != current_identity["physics_sha256"]:
        raise RestampError(
            f"{profile_path} was measured on physics {recorded_identity.get('physics_sha256')!r} but the current "
            f"{species} plant is {current_identity['physics_sha256']!r}: a physics change moves the quiet reference "
            "and needs a real recalibration, not a restamp"
        )

    fingerprint = derive_stage_task_fingerprint(
        species=species,
        stage=stage,
        backend="stable-baselines3",
        env_kwargs=measured_env,
        plant_identity=current_identity,
    )
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


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--species",
        action="append",
        default=None,
        help=f"species whose recovery calibration to restamp (repeatable; default: {', '.join(DEFAULT_SPECIES)})",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=None,
        help="read/write <configs-root>/<species>/recovery_calibration.json instead of the committed configs/",
    )
    parser.add_argument(
        "--reason",
        default=RESTAMP_REASON,
        help="the restamp_history reason to record (default: the Phase C justification; a later "
        "interface-only revision must name its own)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    species_list = args.species or list(DEFAULT_SPECIES)
    changed = 0
    for species in species_list:
        path = None if args.configs_root is None else args.configs_root / species / "recovery_calibration.json"
        try:
            if restamp_recovery_calibration(species, path=path, reason=args.reason):
                changed += 1
        except RestampError as exc:
            logger.error("Refusing to restamp %s: %s", species, exc)
            return 1
    print(f"restamped {changed} of {len(species_list)} calibration(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
