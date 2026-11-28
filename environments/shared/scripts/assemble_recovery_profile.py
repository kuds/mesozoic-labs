"""Assemble a model's runtime recovery profile from a ``calibrate_recovery`` measurement.

``calibrate_recovery`` measures a model's push task against fixed-command PD
nulls and writes one JSON measurement; this tool turns that measurement into the
profile ``recovery_calibration.load_recovery_calibration`` reads,
``configs/<species>/<embodiment>/recovery_calibration.json``.  It is the step
the Compsognathus profiles were built by since the anatomical physics r3
calibration (and both re-measurements of ``configs/plant_versions.toml`` note
24), and the step a later re-measurement repeats.

Run from the repository root::

    python -m environments.shared.scripts.assemble_recovery_profile \
        --species compsognathus_longipes --embodiment robot \
        --measurement recovery_robot.json \
        --artifact recovery_calibration_robot_r3.json.gz \
        --calibration-id compsognathus_longipes-robot-fixed-command-recovery-2026-12-v3
    python -m environments.shared.scripts.assemble_recovery_profile --check

The profile, field by field (the selection rule is ``calibrate_recovery``'s: the
first ascending capture-velocity multiple whose zero-command selection panel
succeeds on at most half its episodes and reaches the full horizon on at least
a quarter):

* ``schema``, ``species``, ``embodiment``: the profile schema and the model,
  which the measurement must name;
* ``calibration_id``: ``--calibration-id`` (a new name per measurement; every
  recovery-evaluation spec copies it); ``--check`` keeps the committed one;
* ``status``: :data:`STATUS`;
* ``control_dt_s``: the discrete timing of the zero-command held-out panel at
  the selected multiple;
* ``height_reference_m`` and ``safe_set``: the quiet calibration's;
* ``recovery_env_kwargs``: the recovery stage's committed ``[env]`` block, whose
  ``perturbation_capture_velocity_multiple`` must already equal the selected
  multiple (edit ``recovery.toml`` first when the selection moves);
* ``capability_spec``: the recovery stage's curriculum values of
  :data:`CAPABILITY_SPEC_KEYS`, pre-registered targets rather than measurements;
* ``required_paired_nulls``: ``zero_action`` and ``brace``;
* ``plant_identity`` and ``task_sha256``: the current plant's identity and the
  recovery stage's task fingerprint over that ``[env]`` block; the measurement's
  recorded hash of the model's MJCF must equal the file's, so a profile is
  never stamped with another plant's identity;
* ``derived_push_manifest``: the push the selected zero-command held-out panel
  was rolled under;
* ``provenance``: the measurement's repository path; the method (with the
  selection panels' episode count); the quiet, selection, held-out, stress and
  reserved publication seed ranges the measurement records; the nominal
  recovery and dwell times (the held-out panel's re-entry deadline and dwell
  steps times ``control_dt_s``); :data:`QUALIFICATION_TARGETS`; and the
  sha256 of the uncompressed measurement.

Assembling (``--measurement``, ``--artifact``, ``--calibration-id``) keeps the
measurement under ``environments/<species>/data/<artifact>``, gzip-compressed
with mtime 0 unless ``--measurement`` already is that file (an existing file
there with other content is refused), and writes the profile, replacing the
committed one and any ``restamp_history`` it carried.  A recorded source
(``provenance.source_sha256``) other than the MJCF that differs from the
working tree is reported, not refused: the loader judges the task by its
fingerprint, not by those hashes.  The new profile's bytes differ from the old
one's, so every recovery freeze made from the old profile is refused.

``--check`` re-assembles every committed profile from the measurement it
names, with its committed ``calibration_id`` (and its ``restamp_history``, which
only a restamp writes), and exits 1 unless each equals its file byte for byte.
It fails wherever the loader would refuse the profile (a changed plant, task or
recovery ``[env]``) and wherever the profile was edited by hand.  It is
stricter than the loader in one place: it compares the measurement's recorded
MJCF bytes, while ``restamp_recovery_calibration`` accepts any revision whose
``physics_sha256`` is unchanged.  A restamp that changes the MJCF's bytes but
not its physics (a visual, comment or interface-only edit) therefore needs a
re-measurement, or a change to this check, before ``--check`` passes again.

This module is not a recorded source of any measurement (``calibrate_recovery``
records its own source list) and enters no plant, task or gate digest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT

PROFILE_FILENAME = "recovery_calibration.json"
#: The ``status`` every profile assembled from fixed-command nulls carries.
STATUS = "physical pilot calibration; learned-policy qualification outstanding"
#: The recovery curriculum keys a profile's ``capability_spec`` records, in its order.
CAPABILITY_SPEC_KEYS = (
    "gate_kind",
    "min_recovery_success_lcb",
    "recovery_t_recover_steps",
    "recovery_dwell_steps",
    "min_eval_episodes",
    "min_paired_success_delta_lcb",
)
REQUIRED_PAIRED_NULLS = ("zero_action", "brace")
#: ``provenance.method``; ``{episodes}`` is the selection panels' episode count.
METHOD = (
    "Fixed-command PD quiet reference; max(engineering floor, 1.5 * quiet p99.9). "
    "Force selection on {episodes} separate seeds; no trained-policy calibration."
)
QUALIFICATION_TARGETS = "Pre-registered pilot engineering targets, not measured policy achievement."
#: The measurement schema ``calibrate_recovery`` writes (one model per run, decision D-D33).
MEASUREMENT_SCHEMA_VERSION = 2
ZERO_COMMAND_CONTROLLER = "zero_command_pd"


class AssemblyError(ValueError):
    """A measurement that cannot honestly become the model's profile."""


def render_profile(profile: dict[str, Any]) -> str:
    """The committed on-disk form: two-space indent, insertion order, trailing newline."""
    return json.dumps(profile, indent=2, ensure_ascii=False) + "\n"


def read_measurement(path: Path) -> bytes:
    """The uncompressed measurement bytes of *path* (``.gz`` is decompressed)."""
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _multiple(panel: dict[str, Any]) -> float:
    return float(panel["perturbation"]["capture_velocity_multiple"])


def _selected_multiple(evidence: dict[str, Any]) -> float:
    """The measurement's selected multiple, after checking that the selection rule gives it."""
    from environments.shared.scripts.calibrate_recovery import _candidates

    panels = sorted(evidence["selection_panels"], key=_multiple)
    candidates = _candidates(panels)
    selected = float(evidence["selected_capture_velocity_multiple"])
    if not candidates or _multiple(candidates[0]) != selected:
        first = _multiple(candidates[0]) if candidates else None
        raise AssemblyError(
            f"the measurement selected multiple {selected!r}, but its selection panels give {first!r} under the "
            "selection rule"
        )
    return selected


def _heldout_zero_command_panel(evidence: dict[str, Any], selected: float) -> dict[str, Any]:
    panels: list[dict[str, Any]] = [
        panel
        for panel in evidence["heldout_diagnostic_panels"]
        if panel["controller_id"] == ZERO_COMMAND_CONTROLLER and _multiple(panel) == selected
    ]
    if len(panels) != 1:
        raise AssemblyError(
            f"the measurement has {len(panels)} {ZERO_COMMAND_CONTROLLER} held-out panels at multiple {selected:g}"
        )
    return panels[0]


def _seed_range(panels: Sequence[dict[str, Any]], name: str) -> list[int]:
    ranges = {tuple(panel["seeds"]) for panel in panels}
    if len(ranges) != 1:
        raise AssemblyError(f"the measurement's {name} panels do not share one seed range: {sorted(ranges)}")
    return list(ranges.pop())


def drifted_sources(evidence: dict[str, Any], repository_root: Path = _REPOSITORY_ROOT) -> list[str]:
    """The measurement's recorded sources whose working-tree bytes differ from the recorded hash (or are gone)."""
    drifted = []
    for path, digest in evidence["provenance"]["source_sha256"].items():
        file = repository_root / path
        if not file.is_file() or _sha256(file.read_bytes()) != digest:
            drifted.append(path)
    return drifted


def assemble_recovery_profile(
    model_id: ModelId,
    measurement: bytes,
    *,
    measurement_artifact: str,
    calibration_id: str,
) -> dict[str, Any]:
    """The profile of *model_id* that *measurement* (``calibrate_recovery``'s uncompressed output) gives.

    *measurement_artifact* is the repository path the measurement is kept at; *calibration_id* names the profile.
    Raises :class:`AssemblyError` when the measurement cannot be the model's profile on this tree.
    """
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.recovery_calibration import PROFILE_SCHEMA
    from environments.shared.task_fingerprint import stage_task_fingerprint

    evidence = json.loads(measurement)
    if evidence.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise AssemblyError(
            f"measurement schema_version {evidence.get('schema_version')!r}; expected {MEASUREMENT_SCHEMA_VERSION}"
        )
    if ModelId.from_record(evidence) != model_id:
        raise AssemblyError(f"the measurement is of {ModelId.from_record(evidence)}, not {model_id}")
    if not calibration_id.strip():
        raise AssemblyError("a profile needs a calibration_id")
    selected = _selected_multiple(evidence)
    heldout = _heldout_zero_command_panel(evidence, selected)
    timing = heldout["discrete_timing"]
    control_dt = timing["control_dt_s"]
    config = load_stage_config(model_id, "recovery")
    env_kwargs = json.loads(json.dumps(config["env_kwargs"]))
    if env_kwargs.get("perturbation_capture_velocity_multiple") != selected:
        raise AssemblyError(
            f"the recovery stage's perturbation_capture_velocity_multiple is "
            f"{env_kwargs.get('perturbation_capture_velocity_multiple')!r}; the measurement selected {selected!r} "
            "(set it in recovery.toml first)"
        )
    curriculum = config["curriculum_kwargs"]
    missing = [key for key in CAPABILITY_SPEC_KEYS if key not in curriculum]
    if missing:
        raise AssemblyError(f"the recovery stage's curriculum does not set {missing}")
    identity = current_plant_identity(model_id).to_dict()
    model_path = identity["model_path"]
    recorded = evidence["provenance"]["source_sha256"].get(model_path)
    if recorded != _sha256((_REPOSITORY_ROOT / model_path).read_bytes()):
        raise AssemblyError(f"the measurement was not taken on the current {model_path} (recorded {recorded})")
    task = stage_task_fingerprint(model_id, "recovery", env_kwargs=env_kwargs, plant_identity=identity)
    quiet = evidence["quiet_calibration"]
    selection_episodes = {int(panel["summary"]["episodes"]) for panel in evidence["selection_panels"]}
    if len(selection_episodes) != 1:
        raise AssemblyError(f"the selection panels do not share one episode count: {sorted(selection_episodes)}")
    return {
        "schema": PROFILE_SCHEMA,
        "species": model_id.species,
        "embodiment": model_id.embodiment,
        "calibration_id": calibration_id,
        "status": STATUS,
        "control_dt_s": control_dt,
        "height_reference_m": quiet["height_reference_m"],
        "safe_set": quiet["safe_set"],
        "recovery_env_kwargs": env_kwargs,
        "capability_spec": {key: curriculum[key] for key in CAPABILITY_SPEC_KEYS},
        "required_paired_nulls": list(REQUIRED_PAIRED_NULLS),
        "plant_identity": identity,
        "task_sha256": task["task_sha256"],
        "derived_push_manifest": heldout["perturbation"],
        "provenance": {
            "measurement_artifact": measurement_artifact,
            "method": METHOD.format(episodes=selection_episodes.pop()),
            "quiet_seed_range": list(quiet["calibration_seeds"]),
            "selection_seed_range": _seed_range(evidence["selection_panels"], "selection"),
            "diagnostic_seed_range": _seed_range(evidence["heldout_diagnostic_panels"], "held-out"),
            "stress_seed_range": _seed_range(evidence["off_distribution_diagnostic_panels"], "stress"),
            "reserved_publication_seed_range": list(evidence["reserved_publication_seed_range_untouched"]),
            "nominal_recovery_seconds": round(timing["reentry_deadline_steps"] * control_dt, 12),
            "nominal_dwell_seconds": round(timing["dwell_steps"] * control_dt, 12),
            "qualification_targets": QUALIFICATION_TARGETS,
            "measurement_uncompressed_sha256": "sha256:" + _sha256(measurement),
        },
    }


def profile_path(model_id: ModelId, configs_root: Path | None = None) -> Path:
    from environments.shared.recovery_calibration import CONFIGS_ROOT

    return model_id.path(CONFIGS_ROOT if configs_root is None else configs_root) / PROFILE_FILENAME


def reassemble_committed_profile(
    model_id: ModelId,
    *,
    configs_root: Path | None = None,
    repository_root: Path = _REPOSITORY_ROOT,
) -> tuple[str, str]:
    """``(committed text, re-assembled text)`` of *model_id*'s profile, from the measurement it names."""
    text = profile_path(model_id, configs_root).read_text(encoding="utf-8")
    committed = json.loads(text)
    artifact = committed["provenance"]["measurement_artifact"]
    profile = assemble_recovery_profile(
        model_id,
        read_measurement(repository_root / artifact),
        measurement_artifact=artifact,
        calibration_id=committed["calibration_id"],
    )
    if "restamp_history" in committed:
        profile["restamp_history"] = committed["restamp_history"]
    return text, render_profile(profile)


def stale_profiles(model_ids: Sequence[ModelId], *, configs_root: Path | None = None) -> list[str]:
    """One line per profile that does not re-assemble to its committed bytes (empty when all are current)."""
    problems = []
    for model_id in model_ids:
        try:
            text, assembled = reassemble_committed_profile(model_id, configs_root=configs_root)
        except (AssemblyError, OSError, KeyError, ValueError) as exc:
            problems.append(f"{model_id}: cannot re-assemble the committed profile: {exc}")
            continue
        if text != assembled:
            problems.append(f"{model_id}: the committed profile differs from the one its measurement assembles")
    return problems


def write_recovery_profile(
    model_id: ModelId,
    measurement_path: Path,
    *,
    artifact: str,
    calibration_id: str,
    configs_root: Path | None = None,
    repository_root: Path = _REPOSITORY_ROOT,
) -> tuple[Path, Path, list[str]]:
    """Keep the measurement at ``environments/<species>/data/<artifact>`` under *repository_root* and write the
    profile it assembles under *configs_root*.

    Returns the profile path, the artifact path and the recorded sources that differ from the working tree.
    """
    if Path(artifact).name != artifact or not artifact.endswith(".json.gz"):
        raise AssemblyError(f"--artifact must be a bare file name ending in .json.gz, not {artifact!r}")
    measurement = read_measurement(measurement_path)
    relative = Path("environments", model_id.species, "data", artifact).as_posix()
    target = repository_root / relative
    profile = assemble_recovery_profile(
        model_id, measurement, measurement_artifact=relative, calibration_id=calibration_id
    )
    if target.exists():
        if read_measurement(target) != measurement:
            raise AssemblyError(f"{relative} already holds another measurement; name a new artifact")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(measurement, compresslevel=9, mtime=0))
    path = profile_path(model_id, configs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_profile(profile), encoding="utf-8")
    return path, target, drifted_sources(json.loads(measurement))


def _committed_models(configs_root: Path | None) -> list[ModelId]:
    from environments.shared.species_names import model_ids

    return [model_id for model_id in model_ids() if profile_path(model_id, configs_root).is_file()]


def main(argv: Sequence[str] | None = None) -> int:
    from environments.shared.species_names import require_model_id

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="every committed profile must re-assemble exactly")
    parser.add_argument("--species")
    parser.add_argument("--embodiment")
    parser.add_argument("--measurement", type=Path, help="calibrate_recovery's output (.json, or a .json.gz)")
    parser.add_argument("--artifact", help="the measurement's file name under environments/<species>/data/")
    parser.add_argument("--calibration-id", help="the new profile's calibration_id")
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=None,
        help="read/write <configs-root>/<species>/<embodiment>/recovery_calibration.json instead of configs/",
    )
    args = parser.parse_args(argv)
    model_id = None
    if args.species or args.embodiment:
        if not (args.species and args.embodiment):
            parser.error("--species and --embodiment go together")
        try:
            model_id = require_model_id(args.species, args.embodiment)
        except ValueError as exc:
            parser.error(str(exc))
    if args.check:
        if args.measurement or args.artifact or args.calibration_id:
            parser.error("--check re-assembles the committed profiles; it takes no measurement")
        targets = [model_id] if model_id is not None else _committed_models(args.configs_root)
        problems = stale_profiles(targets, configs_root=args.configs_root)
        for problem in problems:
            print(problem)
        if problems:
            return 1
        print(f"Recovery profiles are current ({', '.join(map(str, targets)) or 'none committed'})")
        return 0
    if model_id is None or not (args.measurement and args.artifact and args.calibration_id):
        parser.error("assembling needs --species, --embodiment, --measurement, --artifact and --calibration-id")
    try:
        path, artifact, drifted = write_recovery_profile(
            model_id,
            args.measurement,
            artifact=args.artifact,
            calibration_id=args.calibration_id,
            configs_root=args.configs_root,
        )
    except AssemblyError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    for source in drifted:
        print(f"note: {source} differs from the hash the measurement recorded", file=sys.stderr)
    print(f"Wrote {path} from {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
