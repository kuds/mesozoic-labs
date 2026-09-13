"""Build the public species catalog from executable and versioned sources.

The hand-authored species manifest contains presentation metadata and pointers
only. Interface dimensions come from the environments/MJCF models, layered plant
identity comes from the committed generated plant manifest, curriculum facts come
from the stage TOML files, and published metrics come from immutable result
summaries. The generated JSON is committed so the Docusaurus site can build
without requiring Python or MuJoCo in its Node.js build environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mujoco

from environments.shared.config import load_all_stages, load_stage_config
from environments.shared.curriculum import StageThreshold
from environments.shared.curriculum.gate_schema import GATE_KINDS, STANCE_GATE_KIND
from environments.shared.curriculum.recovery_gate import RECOVERY_GATE_KIND
from environments.shared.curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND
from environments.shared.plant_contract import (
    GENERATED_MANIFEST_PATH,
    PHYSICS_SCHEMA,
    PLANT_MANIFEST_SCHEMA,
    POLICY_INTERFACE_SCHEMA,
    SOURCE_SCHEMA,
    VISUAL_SCHEMA,
)
from environments.shared.result_schema import (
    ALLOWED_MODEL_REVISION_STATUSES,
    ALLOWED_TRAINING_BACKENDS,
    ALLOWED_VERIFICATION_STATUSES,
    PROVENANCE_IDENTIFIERS,
    ResultSchemaError,
    ordered_stage_entries,
    primary_deliverable_key,
    validate_provenance,
    validate_result_summary,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS_DIR = REPOSITORY_ROOT / "configs"
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "species_manifest.toml"
DEFAULT_PLANT_MANIFEST_PATH = GENERATED_MANIFEST_PATH
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "website" / "src" / "data" / "species.generated.json"
DEFAULT_README_PATH = REPOSITORY_ROOT / "README.md"

README_BLOCKS = {
    "species": ("<!-- BEGIN GENERATED: SPECIES -->", "<!-- END GENERATED: SPECIES -->"),
    "results": ("<!-- BEGIN GENERATED: RESULTS -->", "<!-- END GENERATED: RESULTS -->"),
    "notebooks": ("<!-- BEGIN GENERATED: NOTEBOOKS -->", "<!-- END GENERATED: NOTEBOOKS -->"),
}

ALLOWED_CAPABILITY_STATUSES = {"implemented", "experimental", "in_progress", "planned", "not_started"}
DEFAULT_STAGE_THRESHOLD = StageThreshold()

#: The measured gate kind a ``none/v1`` pilot stage graduates to, keyed by
#: semantic stage id.  ``none/v1`` is the recorded non-advancing placeholder
#: (curriculum.gate_schema): the recovery stage's real gate —
#: ``recovery_quality/v1`` — was frozen by P5 on 2026-08-28, and the
#: committed trex config now declares it, so this map is dormant for trex;
#: it stays so that any stage run as a pilot (a future species' recovery,
#: or a deliberate regression to none/v1) names what is pending instead of
#: rendering the pilot as though it had no gate at all.  Keyed by stage id, not
#: species, so any species that gains a recovery stage inherits the honest
#: description.
_PENDING_GATE_KINDS = {"recovery": RECOVERY_GATE_KIND}

#: The species manifest schema this reader accepts.  2 (decision D-A8,
#: 2026-09-12): ``[[species.deliverable_metrics]]`` beside the per-backend
#: success metrics, and ``[[species.stage_videos]]`` keyed by stage id.
SPECIES_MANIFEST_SCHEMA_VERSION = 2

#: The units a deliverable headline metric can carry; the website formats
#: by unit, so this vocabulary is part of the catalog contract.
HEADLINE_UNITS = frozenset({"percent", "m/s", "ratio"})


class CatalogError(ValueError):
    """Raised when catalog inputs are incomplete or contradictory."""


# ── Deliverable headline metrics (plan §4.3, decision D-A9) ───────────────
#
# Which statistic headlines a published deliverable is chosen by the gate
# kind it was certified under: a stance policy is judged on unsupported duty
# and full-horizon episodes, a recovery policy on its recovery-success lower
# bound, a walk on velocity and a hunt on task success.  Each spec is
# ``(summary key, label, unit)``; the VALUE is read from the published stage
# row under that key, and a statistic the summary does not carry (stance and
# recovery, whose per-stage gate metrics reach summary.json in a later phase
# — decision D-B15) is published with a null value and the key still named,
# so a reader sees what the gate measured rather than a blank.
#
# The registry is keyed by exactly ``set(GATE_KINDS)`` (pinned): a gate kind
# added to the schema without a headline entry is a catalog failure, never a
# silently metric-less deliverable.
_HeadlineSpec = tuple[str, str, str]


def _stance_headline_specs(current_gate: dict[str, Any]) -> list[_HeadlineSpec]:
    return [
        ("unsupported_duty_ucb", "unsupported duty 95% UCB", "ratio"),
        ("full_horizon_fraction", "full-horizon episodes", "percent"),
    ]


def _recovery_headline_specs(current_gate: dict[str, Any]) -> list[_HeadlineSpec]:
    return [("recovery_success_lcb", "recovery success LCB95", "ratio")]


def _reward_and_length_headline_specs(current_gate: dict[str, Any]) -> list[_HeadlineSpec]:
    # The historical gate measures whatever the stage declares: a walk
    # (velocity floor) headlines its velocity, a hunt (success floor) its
    # task success, and a stage gated on both publishes both.
    specs: list[_HeadlineSpec] = []
    if current_gate.get("min_avg_forward_velocity") is not None:
        specs.append(("avg_forward_vel", "avg. forward velocity", "m/s"))
    if current_gate.get("min_success_rate") is not None:
        specs.append(("mean_success_rate", "task success", "percent"))
    return specs


def _task_success_headline_specs(current_gate: dict[str, Any]) -> list[_HeadlineSpec]:
    # The hunting gate certifies the exact binomial lower bound on task
    # success (plan §4.4), which the summary stage row records as
    # selected_model_success_lcb; the raw selected-checkpoint rate follows
    # it so a reader sees both the bound and the fraction it bounds.
    return [
        ("selected_model_success_lcb", "task success LCB95", "ratio"),
        ("selected_model_success_rate", "task success", "percent"),
    ]


def _no_headline_specs(current_gate: dict[str, Any]) -> list[_HeadlineSpec]:
    # none/v1 certifies nothing, so nothing headlines it.
    return []


_HEADLINE_BY_GATE_KIND: dict[str, Callable[[dict[str, Any]], list[_HeadlineSpec]]] = {
    STANCE_GATE_KIND: _stance_headline_specs,
    RECOVERY_GATE_KIND: _recovery_headline_specs,
    "reward_and_length/v1": _reward_and_length_headline_specs,
    TASK_SUCCESS_GATE_KIND: _task_success_headline_specs,
    "none/v1": _no_headline_specs,
}

if set(_HEADLINE_BY_GATE_KIND) != set(GATE_KINDS):  # pragma: no cover - import-time contract
    raise RuntimeError(
        "species_catalog._HEADLINE_BY_GATE_KIND must name exactly the gate kinds in "
        f"curriculum.gate_schema.GATE_KINDS; missing {sorted(set(GATE_KINDS) - set(_HEADLINE_BY_GATE_KIND))}, "
        f"extra {sorted(set(_HEADLINE_BY_GATE_KIND) - set(GATE_KINDS))}"
    )


def _deliverable_headline(
    gate_kind: str | None,
    stage_row: dict[str, Any],
    current_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    """The headline metrics of a deliverable certified under *gate_kind*.

    Each entry is ``{key, label, value, unit}``; ``value`` is the published
    stage row's statistic under ``key`` or null when the summary does not
    record it.  A null *gate_kind* (a record whose verdict and config both
    left the kind unrecorded) headlines nothing: choosing a metric by the
    CURRENT gate would present a statistic the certifying gate never
    measured.  A non-null kind outside the registry is fatal.
    """
    if gate_kind is None:
        return []
    specs = _HEADLINE_BY_GATE_KIND.get(gate_kind)
    if specs is None:
        raise CatalogError(
            f"no headline metric registered for gate kind {gate_kind!r}; "
            f"species_catalog._HEADLINE_BY_GATE_KIND knows {sorted(_HEADLINE_BY_GATE_KIND)}"
        )
    headline: list[dict[str, Any]] = []
    for key, label, unit in specs(current_gate):
        if unit not in HEADLINE_UNITS:  # pragma: no cover - registry contract
            raise CatalogError(f"headline unit {unit!r} for {key} is not one of {sorted(HEADLINE_UNITS)}")
        headline.append(
            {
                "key": key,
                "label": label,
                "value": _optional_number(stage_row.get(key), field=f"headline metric {key}"),
                "unit": unit,
            }
        )
    return headline


def _repo_path(relative_path: str, *, field: str) -> Path:
    """Resolve and validate a repository-relative manifest path."""
    path = (REPOSITORY_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise CatalogError(f"{field} must stay within the repository: {relative_path}") from exc
    if not path.exists():
        raise CatalogError(f"{field} does not exist: {relative_path}")
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    if manifest.get("schema_version") != SPECIES_MANIFEST_SCHEMA_VERSION:
        raise CatalogError(f"species manifest schema_version must be {SPECIES_MANIFEST_SCHEMA_VERSION}")
    return manifest


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_nonempty_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field=field)


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CatalogError(f"{field} must be a positive integer")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CatalogError(f"{field} must be sha256:<64 lowercase hex>")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CatalogError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def _load_plant_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read generated plant manifest {path}: {exc}") from exc
    manifest = _require_mapping(manifest, field="generated plant manifest")
    if manifest.get("schema") != PLANT_MANIFEST_SCHEMA:
        raise CatalogError(f"generated plant manifest must use schema {PLANT_MANIFEST_SCHEMA}")
    _require_positive_int(manifest.get("fingerprint_tool_version"), field="plant fingerprint_tool_version")
    generated_with = _require_mapping(manifest.get("generated_with"), field="plant generated_with")
    _require_nonempty_string(generated_with.get("mujoco"), field="plant generated_with.mujoco")
    _require_positive_int(
        generated_with.get("float_significant_digits"),
        field="plant generated_with.float_significant_digits",
    )
    _require_mapping(manifest.get("plants"), field="plant manifest plants")
    return manifest


def _optional_number(value: Any, *, field: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CatalogError(f"{field} must be null or a finite number")
    # Annotated local: the isinstance guard above narrows the value, but mypy
    # does not carry that narrowing out of an Any-typed parameter, so the bare
    # ``return value`` trips no-any-return.
    number: int | float = value
    return number


def _load_environment(entrypoint: str) -> type[Any]:
    module_name, separator, class_name = entrypoint.partition(":")
    if not separator or not module_name or not class_name:
        raise CatalogError(f"invalid environment entrypoint: {entrypoint}")
    try:
        module = importlib.import_module(module_name)
        environment_class = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise CatalogError(f"cannot import environment entrypoint: {entrypoint}") from exc
    return cast(type[Any], environment_class)


def _space_dimension(space: Any, *, label: str) -> int:
    shape = getattr(space, "shape", None)
    if not shape:
        raise CatalogError(f"{label} space must have a fixed shape")
    return int(math.prod(shape))


def _model_facts(model_path: Path) -> dict[str, int | float]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    dynamic_mass = sum(
        float(mass) for mass, mocap_id in zip(model.body_mass, model.body_mocapid, strict=True) if mocap_id == -1
    )
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "dynamic_mass_kg": round(dynamic_mass, 6),
    }


def _environment_facts(entrypoint: str, expected_nu: int) -> dict[str, Any]:
    environment_class = _load_environment(entrypoint)
    environment = environment_class()
    try:
        observation_dim = _space_dimension(environment.observation_space, label="observation")
        action_dim = _space_dimension(environment.action_space, label="action")
    finally:
        environment.close()
    if action_dim != expected_nu:
        raise CatalogError(f"{entrypoint} action dimension {action_dim} does not match model.nu {expected_nu}")
    return {
        "entrypoint": entrypoint,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }


def _public_plant_contract(
    species_id: str,
    raw_entry: Any,
    *,
    model_relative_path: str,
    model: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Validate one generated plant entry and select its public contract."""
    entry = _require_mapping(raw_entry, field=f"plant contract for {species_id}")
    if entry.get("species") != species_id:
        raise CatalogError(f"plant contract species mismatch for {species_id}: {entry.get('species')}")
    if entry.get("model_path") != model_relative_path:
        raise CatalogError(
            f"plant contract model_path mismatch for {species_id}: expected {model_relative_path}, "
            f"got {entry.get('model_path')}"
        )

    source = _require_mapping(entry.get("source"), field=f"plant source for {species_id}")
    if source.get("schema") != SOURCE_SCHEMA:
        raise CatalogError(f"plant source for {species_id} must use schema {SOURCE_SCHEMA}")
    if source.get("root") != model_relative_path:
        raise CatalogError(f"plant source root mismatch for {species_id}: {source.get('root')}")

    policy = _require_mapping(entry.get("policy_interface"), field=f"plant policy interface for {species_id}")
    if policy.get("schema") != POLICY_INTERFACE_SCHEMA:
        raise CatalogError(f"plant policy interface for {species_id} must use schema {POLICY_INTERFACE_SCHEMA}")
    policy_revision = _require_positive_int(policy.get("revision"), field=f"{species_id} policy revision")
    policy_observation_dim = _require_positive_int(
        policy.get("observation_dim"), field=f"{species_id} plant observation_dim"
    )
    policy_action_dim = _require_positive_int(policy.get("action_dim"), field=f"{species_id} plant action_dim")
    if policy_observation_dim != environment["observation_dim"]:
        raise CatalogError(
            f"plant observation_dim mismatch for {species_id}: contract={policy_observation_dim}, "
            f"environment={environment['observation_dim']}"
        )
    if policy_action_dim != environment["action_dim"]:
        raise CatalogError(
            f"plant action_dim mismatch for {species_id}: contract={policy_action_dim}, "
            f"environment={environment['action_dim']}"
        )

    physics = _require_mapping(entry.get("physics"), field=f"plant physics for {species_id}")
    if physics.get("schema") != PHYSICS_SCHEMA:
        raise CatalogError(f"plant physics for {species_id} must use schema {PHYSICS_SCHEMA}")
    physics_revision = _require_positive_int(physics.get("revision"), field=f"{species_id} physics revision")
    for dimension in ("nq", "nv", "nu"):
        contract_value = _require_positive_int(physics.get(dimension), field=f"{species_id} plant {dimension}")
        if contract_value != model[dimension]:
            raise CatalogError(
                f"plant {dimension} mismatch for {species_id}: contract={contract_value}, model={model[dimension]}"
            )

    visual = _require_mapping(entry.get("visual"), field=f"plant visual for {species_id}")
    if visual.get("schema") != VISUAL_SCHEMA:
        raise CatalogError(f"plant visual for {species_id} must use schema {VISUAL_SCHEMA}")
    visual_revision = _require_positive_int(visual.get("revision"), field=f"{species_id} visual revision")

    return {
        "bundle_sha256": _require_sha256(entry.get("bundle_sha256"), field=f"{species_id} plant bundle"),
        "source_closure_sha256": _require_sha256(source.get("closure_sha256"), field=f"{species_id} source closure"),
        "policy_interface": {
            "schema": POLICY_INTERFACE_SCHEMA,
            "revision": policy_revision,
            "observation_schema": _require_nonempty_string(
                policy.get("observation_schema"), field=f"{species_id} observation_schema"
            ),
            "sha256": _require_sha256(policy.get("sha256"), field=f"{species_id} policy interface"),
        },
        "physics": {
            "schema": PHYSICS_SCHEMA,
            "revision": physics_revision,
            "sha256": _require_sha256(physics.get("sha256"), field=f"{species_id} physics"),
        },
        "visual": {
            "schema": VISUAL_SCHEMA,
            "revision": visual_revision,
            "sha256": _require_sha256(visual.get("sha256"), field=f"{species_id} visual"),
        },
    }


def _configs_root(configs_dir: "Path | str | None") -> Path:
    """The configs directory the catalog reads stage manifests and TOMLs from.

    ``None`` is the repository's ``configs/``; a test may point at a
    synthesized tree (a species directory without ``stages.toml``) without
    monkeypatching the loaders' private constants.
    """
    return DEFAULT_CONFIGS_DIR if configs_dir is None else Path(configs_dir).resolve()


def _load_manifest_and_configs(
    species_id: str, configs_dir: "Path | str | None" = None
) -> "tuple[Any, dict[int | str, dict[str, Any]]]":
    """The species' stage manifest and every stage config it declares, in manifest order."""
    from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

    root = _configs_root(configs_dir)
    try:
        manifest = load_stage_manifest(species_id, root)
    except StageManifestError as exc:
        raise CatalogError(f"cannot load the stage manifest for {species_id}: {exc}") from exc
    if configs_dir is None:
        return manifest, load_all_stages(species_id)
    # A non-default root bypasses load_all_stages (which reads the
    # repository's configs/) by naming each TOML explicitly; the manifest
    # already validated that every config file exists.
    configs: dict[int | str, dict[str, Any]] = {
        entry.reference: load_stage_config(
            species_id, entry.reference, config_path=str(root / species_id / entry.config_file)
        )
        for entry in manifest.stages
    }
    return manifest, configs


def _stage_config_path(species_id: str, stage_ref: "int | str", configs_dir: "Path | str | None" = None) -> Path:
    from environments.shared.stage_manifest import StageManifestError, load_stage_manifest

    root = _configs_root(configs_dir)
    try:
        entry = load_stage_manifest(species_id, root).resolve(stage_ref)
    except StageManifestError as exc:
        raise CatalogError(f"cannot resolve stage {stage_ref} config for {species_id}: {exc}") from exc
    return root / species_id / entry.config_file


def _advancement_gate(entry: Any, curriculum: dict[str, Any]) -> dict[str, Any]:
    """The stage's effective early-advancement gate as the catalog exports it."""
    return {
        # The declared gate KIND drives rendering: a none/v1
        # pilot must read as "non-advancing", never as an empty
        # criteria list that looks like a free pass.
        "gate_kind": curriculum.get("gate_kind"),
        "pending_gate_kind": _PENDING_GATE_KINDS.get(entry.id) if curriculum.get("gate_kind") == "none/v1" else None,
        "min_avg_reward": curriculum.get("min_avg_reward"),
        "min_avg_episode_length": curriculum.get("min_avg_episode_length"),
        "min_avg_forward_velocity": curriculum.get("min_avg_forward_vel"),
        "min_success_rate": curriculum.get("min_success_rate"),
        # stance_quality/v1. Exported so the published gate is the
        # one actually enforced: a stance-gated stage whose only
        # listed criterion was its reward rail would read as gated
        # on a threshold its own statue clears by 68%.
        "min_full_horizon_fraction": curriculum.get("min_full_horizon_fraction"),
        "max_unsupported_duty": curriculum.get("max_unsupported_duty"),
        "max_unsupported_duty_ucb": curriculum.get("max_unsupported_duty_ucb"),
        # recovery_quality/v1. The certifying criteria are frozen
        # in the stage directory's gate_resolution.json
        # (curriculum/gate_resolver); the config declares the same
        # numbers and reporting/gates refuses when the two
        # disagree, so exporting the declared values publishes the
        # enforced gate rather than an episode-count shell.
        "min_recovery_success_lcb": curriculum.get("min_recovery_success_lcb"),
        "min_paired_success_delta_lcb": curriculum.get("min_paired_success_delta_lcb"),
        "recovery_t_recover_steps": curriculum.get("recovery_t_recover_steps"),
        "recovery_dwell_steps": curriculum.get("recovery_dwell_steps"),
        # task_success/v1 (plan §4.4): the LCB bar the hunting deliverable
        # is certified on; null on every other kind.
        "min_success_lcb": curriculum.get("min_success_lcb"),
        "min_eval_episodes": int(curriculum.get("min_eval_episodes", DEFAULT_STAGE_THRESHOLD.min_eval_episodes)),
        "required_consecutive": int(
            curriculum.get("required_consecutive", DEFAULT_STAGE_THRESHOLD.required_consecutive)
        ),
    }


def current_advancement_gates(species_id: str, configs_dir: "Path | str | None" = None) -> dict[str, dict[str, Any]]:
    """The gate each of the species' stages declares TODAY, by stage id."""
    manifest, configs = _load_manifest_and_configs(species_id, configs_dir)
    return {
        entry.id: _advancement_gate(entry, configs[entry.reference]["curriculum_kwargs"]) for entry in manifest.stages
    }


def _public_video_path(relative_path: str) -> str:
    prefix = "website/static"
    if not relative_path.startswith(f"{prefix}/"):
        raise CatalogError(f"stage video must be under {prefix}: {relative_path}")
    return relative_path[len(prefix) :]


def _build_stages(
    species_id: str,
    raw_videos: list[dict[str, Any]],
    *,
    configs_dir: "Path | str | None" = None,
) -> list[dict[str, Any]]:
    from environments.shared.stage_manifest import StageManifestError

    manifest, configs = _load_manifest_and_configs(species_id, configs_dir)
    # Videos are keyed by stage id (species manifest schema 2); the legacy
    # number stays an accepted alias because the stage manifest resolves
    # both spellings an entry can carry, and fails closed on anything the
    # species does not declare.  Two spellings of one stage are a duplicate.
    video_by_stage: dict[str, dict[str, Any]] = {}
    for raw_video in raw_videos:
        raw_stage_ref = raw_video["stage"]
        try:
            video_entry = manifest.resolve(raw_stage_ref)
        except StageManifestError as exc:
            raise CatalogError(f"stage video for {species_id} names an unknown stage {raw_stage_ref!r}: {exc}") from exc
        stage_number = video_entry.reference
        if video_entry.id in video_by_stage:
            raise CatalogError(f"duplicate stage video for {species_id} stage {stage_number}")
        relative_path = str(raw_video["path"])
        video_path = _repo_path(relative_path, field=f"{species_id} stage {stage_number} video")
        poster_path = REPOSITORY_ROOT / "website" / "static" / "img" / "posters" / f"{video_path.stem}.jpg"
        if not poster_path.exists():
            raise CatalogError(f"poster does not exist for {relative_path}: {poster_path.relative_to(REPOSITORY_ROOT)}")
        algorithm = _require_nonempty_string(
            raw_video.get("algorithm"), field=f"{species_id} stage {stage_number} video algorithm"
        )
        backend = raw_video.get("backend")
        backend_version = _optional_nonempty_string(
            raw_video.get("backend_version"), field=f"{species_id} stage {stage_number} video backend_version"
        )
        model_revision_status = raw_video.get("model_revision_status")
        verification_status = raw_video.get("verification_status")
        if backend not in ALLOWED_TRAINING_BACKENDS:
            raise CatalogError(f"invalid stage-video backend for {species_id} stage {stage_number}: {backend}")
        if model_revision_status not in ALLOWED_MODEL_REVISION_STATUSES:
            raise CatalogError(
                f"invalid stage-video model_revision_status for {species_id} stage {stage_number}: "
                f"{model_revision_status}"
            )
        if verification_status not in ALLOWED_VERIFICATION_STATUSES:
            raise CatalogError(
                f"invalid stage-video verification_status for {species_id} stage {stage_number}: {verification_status}"
            )
        identifiers = {key: raw_video.get(key) for key in PROVENANCE_IDENTIFIERS}
        for key, value in identifiers.items():
            identifiers[key] = _optional_nonempty_string(value, field=f"{species_id} stage {stage_number} video {key}")
        if model_revision_status == "current" or verification_status == "verified":
            missing = [key for key, value in identifiers.items() if not value]
            if backend_version is None:
                missing.append("backend_version")
            if missing:
                raise CatalogError(
                    f"current or verified stage video for {species_id} stage {stage_number} "
                    f"is missing identifiers: {missing}"
                )
        video_by_stage[video_entry.id] = {
            "path": _public_video_path(relative_path),
            "algorithm": algorithm,
            "backend": str(backend),
            "backend_version": backend_version,
            "model_revision_status": str(model_revision_status),
            "verification_status": str(verification_status),
            **identifiers,
        }

    # config_path is rendered relative to the configs root's parent: the
    # repository root for the committed tree (unchanged), a temporary root
    # for a synthesized fixture.
    path_root = _configs_root(configs_dir).parent
    stages: list[dict[str, Any]] = []
    # Every stage the manifest declares, in manifest (curriculum) order —
    # not the retired hardcoded (1, 2, 3), which silently dropped the
    # recovery stage from the generated tables.
    for entry in manifest.stages:
        stage_config = configs[entry.reference]
        curriculum = stage_config["curriculum_kwargs"]
        config_path = _stage_config_path(species_id, entry.reference, configs_dir)
        name = str(stage_config["name"])
        stages.append(
            {
                "id": entry.id,
                "position": entry.position,
                # The legacy number, or null for semantic-only stages
                # (recovery).  Display goes through "label": rendering a
                # position here would renumber history, and inventing "1b"
                # would mint a number the manifest never declared.
                "number": entry.legacy_number,
                "label": entry.key,
                "name": name,
                "title": name.replace("_", " ").title(),
                "description": str(stage_config["description"]),
                "config_path": config_path.relative_to(path_root).as_posix(),
                "timesteps": int(curriculum["timesteps"]),
                # The recipe DAG (stage manifest v2, plan §4.1): whether this
                # node's certified checkpoint is a published policy, the
                # EARLIER stage id it warm-starts from (null for a root), and
                # the behavior label it belongs to.  Top-level rather than
                # inside advancement_gate, which stays the gate alone.  A v1
                # or synthesized manifest derives these as the loader did
                # before v2 (last advancing entry the only deliverable, no
                # labels), so the catalog never invents a recipe.
                "deliverable": entry.deliverable,
                "warm_start_from": entry.warm_start_from,
                "recipe": entry.recipe,
                "advancement_gate": _advancement_gate(entry, curriculum),
                "video": video_by_stage.get(entry.id),
            }
        )
    return stages


def _validate_provenance(provenance: Any, *, result_path: str) -> dict[str, Any]:
    """Validate result provenance while preserving the catalog error API."""
    try:
        return validate_provenance(provenance, result_path=result_path)
    except ResultSchemaError as exc:
        raise CatalogError(str(exc)) from exc


def _validate_result_summary(summary: Any, *, species_id: str, relative_path: str) -> dict[str, Any]:
    """Validate public results while preserving the catalog error API.

    Publishable, not complete (decision D-A2): a schema-4 run that certified
    walk but failed hunt publishes its certified deliverables as ``partial``.
    Below schema 4 the two flags are synonyms, so the four committed v2
    summaries validate exactly as before.
    """
    try:
        return validate_result_summary(
            summary,
            expected_species=species_id,
            relative_path=relative_path,
            require_complete=False,
            require_publishable=True,
        )
    except ResultSchemaError as exc:
        raise CatalogError(str(exc)) from exc


def current_gate_kinds(species_id: str, configs_dir: "Path | str | None" = None) -> dict[str, "str | None"]:
    """The gate kind each of the species' stages declares TODAY, by stage id.

    What a published verdict is compared against: a ``stage_passed`` earned
    under a different (or unrecorded) gate is a pass of a retired gate, and
    the catalog says so instead of re-serving the bare boolean beneath the
    current gate's description (review SS5).
    """
    manifest, configs = _load_manifest_and_configs(species_id, configs_dir)
    return {entry.id: configs[entry.reference]["curriculum_kwargs"].get("gate_kind") for entry in manifest.stages}


def _max_reported_velocity(stage_summaries: list[dict[str, Any]]) -> int | float | None:
    reported_velocities = [
        stage["avg_forward_vel"] for stage in stage_summaries if stage["avg_forward_vel"] is not None
    ]
    return max(reported_velocities) if reported_velocities else None


def _build_result(species_id: str, relative_path: str) -> dict[str, Any]:
    result_path = _repo_path(relative_path, field=f"{species_id} result summary")
    with result_path.open(encoding="utf-8") as result_file:
        summary = _validate_result_summary(json.load(result_file), species_id=species_id, relative_path=relative_path)

    # Manifest order, manifest vocabulary: a recovery-bearing summary renders
    # its stage between stance and locomotion, and sorting never calls int()
    # on a key (the semantic id has no integer).  The summary was validated
    # above, so resolution cannot fail here except through the same
    # CatalogError channel.
    try:
        stage_entries = ordered_stage_entries(summary["stages"], species=species_id, field=f"stages in {relative_path}")
    except ResultSchemaError as exc:
        raise CatalogError(str(exc)) from exc
    declared_gates = current_gate_kinds(species_id)
    stage_summaries: list[dict[str, Any]] = []
    for stage_key, entry in stage_entries:
        raw_stage = summary["stages"][stage_key]
        recorded_gate = raw_stage.get("gate_kind")
        current_gate = declared_gates.get(entry.id)
        stage_summaries.append(
            {
                "id": entry.id,
                "position": entry.position,
                "number": entry.legacy_number,
                "label": entry.key,
                # The recipe DAG as the CURRENT manifest declares it for this
                # stage, so a ladder row says which behavior it now belongs
                # to; certification is per deliverable below, never inferred
                # from these two keys.
                "recipe": entry.recipe,
                "deliverable": entry.deliverable,
                "name": raw_stage.get("name", entry.id),
                "description": raw_stage.get("description", ""),
                "timesteps": raw_stage.get("timesteps"),
                "best_eval_reward": raw_stage.get("best_eval_reward"),
                "final_eval_reward": raw_stage.get("final_eval_reward"),
                "avg_forward_vel": raw_stage.get("avg_forward_vel"),
                "avg_episode_length": raw_stage.get("avg_episode_length"),
                "mean_success_rate": raw_stage.get("mean_success_rate"),
                "training_time": raw_stage.get("training_time"),
                "stage_passed": raw_stage.get("stage_passed"),
                # Provenance for the verdict: the gate the summary recorded
                # (None for a producer that predates the key), the gate the
                # species declares now, and whether the two differ.
                "gate_kind": recorded_gate,
                "current_gate_kind": current_gate,
                "gate_retired": recorded_gate is None or recorded_gate != current_gate,
            }
        )

    validated_provenance = _validate_provenance(summary["provenance"], result_path=relative_path)
    raw_stages_by_id = {entry.id: summary["stages"][stage_key] for stage_key, entry in stage_entries}
    deliverables, primary_deliverable, target_deliverable = _build_result_deliverables(
        species_id, summary, validated_provenance, raw_stages_by_id, relative_path=relative_path
    )
    # The exported provenance is the identity/status surface alone.  A v4
    # block also validates to its deliverables map, ancestors, primary and
    # target (and a canonical one to run_id, seeds, ...); those are published
    # ONLY through the per-deliverable rows below, so every schema exports
    # the same six keys and the website adapter's RawResult.provenance stays
    # honest for the first committed schema-4 summary.
    provenance = {
        key: validated_provenance[key]
        for key in ("model_revision_status", "verification_status", "evaluation_episodes", *PROVENANCE_IDENTIFIERS)
    }
    max_average_forward_velocity = _max_reported_velocity(stage_summaries)
    stage_three = next((stage for stage in stage_summaries if stage["number"] == 3), None)
    return {
        "summary_path": relative_path,
        "algorithm": summary["algorithm"],
        "backend": summary["backend"],
        "backend_version": summary.get("backend_version"),
        "date": summary["date"],
        "hardware": summary.get("hardware"),
        "seed": summary.get("seed"),
        "parallel_envs": summary.get("parallel_envs"),
        "total_timesteps": summary.get(
            "total_timesteps", sum(stage.get("timesteps") or 0 for stage in stage_summaries)
        ),
        "total_training_time": summary.get("total_training_time"),
        "final_avg_reward": summary.get("final_avg_reward"),
        "max_average_forward_velocity": max_average_forward_velocity,
        # The historical ladder headline (plan §4.3): unchanged for every
        # committed row; a v4 result's headline comes from its primary
        # deliverable below, and a v4 run without a behavior stage is null.
        "stage3_success_rate": stage_three.get("mean_success_rate") if stage_three else None,
        "provenance": provenance,
        # Per-deliverable publication (schema 4): [] and nulls for every
        # schema-2/3 summary — never synthesized from a ladder pass.
        "deliverables": deliverables,
        "primary_deliverable": primary_deliverable,
        "target_deliverable": target_deliverable,
        "stages": stage_summaries,
    }


def _build_result_deliverables(
    species_id: str,
    summary: dict[str, Any],
    provenance: dict[str, Any],
    raw_stages_by_id: dict[str, dict[str, Any]],
    *,
    relative_path: str,
) -> "tuple[list[dict[str, Any]], str | None, str | None]":
    """The published deliverables of one result, in manifest order.

    *provenance* is the VALIDATED provenance block (its ``deliverables``
    map, primary and target); *raw_stages_by_id* maps each recorded stage's
    id to its summary row as written, so a headline reads the statistic the
    summary records (``unsupported_duty_ucb``, ``full_horizon_fraction``,
    ``recovery_success_lcb`` once a later phase exports them — D-A9,
    deferred by D-B15; ``selected_model_success_lcb`` for a task_success/v1
    hunt is exported now) rather than the ladder projection, which carries
    only the fixed ladder columns.

    A schema-2/3 ladder summary publishes NO deliverable: its ``stage_passed``
    was a pass of the retired reward gate, and relabelling it "certified"
    would mint a certification nothing measured.  From schema 4 on each
    ``provenance.deliverables`` record becomes a row carrying its gate kind,
    certification, model hash, replication count and gate-kind headline; the
    primary and target come from the explicit provenance keys, and a primary
    that is absent, unknown, uncertified or not the one the deliverables
    imply is a catalog failure.
    """
    schema_version = int(summary.get("schema_version") or 0)
    if schema_version < 4:
        return [], None, None
    raw_records = provenance.get("deliverables")
    if not raw_records:
        raise CatalogError(f"schema-{schema_version} result {relative_path} publishes no provenance.deliverables")
    try:
        entries = ordered_stage_entries(
            raw_records, species=species_id, field=f"provenance.deliverables in {relative_path}"
        )
    except ResultSchemaError as exc:
        raise CatalogError(str(exc)) from exc
    current_gates = current_advancement_gates(species_id)
    deliverables: list[dict[str, Any]] = []
    for key, entry in entries:
        record = raw_records[key]
        stage_row = raw_stages_by_id.get(entry.id)
        if stage_row is None:
            raise CatalogError(
                f"provenance.deliverables in {relative_path} names stage {key!r}, which the summary does not record"
            )
        deliverables.append(
            {
                "id": entry.id,
                "stage_key": key,
                "label": entry.key,
                "recipe": entry.recipe,
                "gate_kind": record["gate_kind"],
                "certified": bool(record["certified"]),
                "model_hash": record["model_hash"],
                "replication_count": int(record["replication"]["count"]),
                "headline": _deliverable_headline(record["gate_kind"], stage_row, current_gates[entry.id]),
            }
        )
    certified_by_key = {row["stage_key"]: row["certified"] for row in deliverables}
    target_deliverable = provenance.get("target_deliverable")
    primary_deliverable = provenance.get("primary_deliverable")
    if primary_deliverable is None:
        raise CatalogError(f"schema-{schema_version} result {relative_path} names no primary deliverable")
    if primary_deliverable not in certified_by_key:
        raise CatalogError(
            f"primary deliverable {primary_deliverable!r} in {relative_path} is not among the published "
            f"deliverables {sorted(certified_by_key)}"
        )
    if not certified_by_key[primary_deliverable]:
        raise CatalogError(
            f"primary deliverable {primary_deliverable!r} in {relative_path} is not certified; the catalog "
            "publishes only a certified primary"
        )
    try:
        expected_primary = primary_deliverable_key(raw_records, species=species_id, target=target_deliverable)
    except ResultSchemaError as exc:
        raise CatalogError(str(exc)) from exc
    if primary_deliverable != expected_primary:
        raise CatalogError(
            f"primary deliverable {primary_deliverable!r} in {relative_path} is not the one its deliverables "
            f"and target {target_deliverable!r} imply ({expected_primary!r})"
        )
    return deliverables, str(primary_deliverable), None if target_deliverable is None else str(target_deliverable)


def _build_success_metrics(
    species_id: str, raw_metrics: Any, expected_backends: set[str] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise CatalogError(f"{species_id} must define at least one backend-scoped success metric")

    covered_backends: set[str] = set()
    metrics: list[dict[str, Any]] = []
    for index, raw_metric_value in enumerate(raw_metrics, start=1):
        raw_metric = _require_mapping(raw_metric_value, field=f"{species_id} success metric {index}")
        backends = raw_metric.get("backends")
        if not isinstance(backends, list) or not backends:
            raise CatalogError(f"{species_id} success metric {index} must name at least one backend")
        if any(backend not in ALLOWED_TRAINING_BACKENDS for backend in backends):
            raise CatalogError(f"{species_id} success metric {index} has invalid backends: {backends}")
        if len(set(backends)) != len(backends):
            raise CatalogError(f"{species_id} success metric {index} repeats a backend")
        duplicate_backends = covered_backends.intersection(backends)
        if duplicate_backends:
            raise CatalogError(f"{species_id} defines more than one success metric for {sorted(duplicate_backends)}")
        covered_backends.update(backends)
        metrics.append(
            {
                "backends": list(backends),
                "key": _require_nonempty_string(
                    raw_metric.get("key"), field=f"key for {species_id} success metric {index}"
                ),
                "label": _require_nonempty_string(
                    raw_metric.get("label"), field=f"label for {species_id} success metric {index}"
                ),
                "definition": _require_nonempty_string(
                    raw_metric.get("definition"), field=f"definition for {species_id} success metric {index}"
                ),
            }
        )

    expected = ALLOWED_TRAINING_BACKENDS if expected_backends is None else expected_backends
    if covered_backends != expected:
        missing = sorted(expected - covered_backends)
        extra = sorted(covered_backends - expected)
        raise CatalogError(f"{species_id} success metric backends mismatch; missing={missing}, unsupported={extra}")
    return metrics


def _build_deliverable_metrics(
    species_id: str,
    raw_metrics: Any,
    expected_backends: set[str] | None = None,
    *,
    configs_dir: "Path | str | None" = None,
) -> list[dict[str, Any]]:
    """Validate ``[[species.deliverable_metrics]]`` against the species' stage manifest.

    Each entry names a deliverable (a recipe label or a deliverable stage
    id, resolved exactly as the notebook's ``BEHAVIOR`` knob is — a label
    means its deepest deliverable), the backends the definition holds for,
    and the metric's key, label and definition.  Fail-closed: an unknown
    deliverable or label, a backend the species does not train, and two
    entries for one (stage, backend) are all fatal.  Coverage: every
    deliverable the manifest declares has a ``stable-baselines3``
    definition, the evidence backend (plan §4.9), so a newly declared
    deliverable cannot publish with no stated semantics.
    """
    from environments.shared.stage_manifest import StageManifestError

    manifest, _configs = _load_manifest_and_configs(species_id, configs_dir)
    if not isinstance(raw_metrics, list):
        raise CatalogError(f"{species_id} deliverable_metrics must be a list")
    expected = ALLOWED_TRAINING_BACKENDS if expected_backends is None else expected_backends

    covered: set[tuple[str, str]] = set()
    metrics: list[dict[str, Any]] = []
    for index, raw_metric_value in enumerate(raw_metrics, start=1):
        raw_metric = _require_mapping(raw_metric_value, field=f"{species_id} deliverable metric {index}")
        deliverable = _require_nonempty_string(
            raw_metric.get("deliverable"), field=f"deliverable for {species_id} deliverable metric {index}"
        )
        try:
            entry = manifest.resolve_behavior(deliverable)
        except StageManifestError as exc:
            raise CatalogError(
                f"{species_id} deliverable metric {index} names unknown deliverable {deliverable!r}: {exc}"
            ) from exc
        backends = raw_metric.get("backends")
        if not isinstance(backends, list) or not backends:
            raise CatalogError(f"{species_id} deliverable metric {index} must name at least one backend")
        unknown_backends = [backend for backend in backends if backend not in ALLOWED_TRAINING_BACKENDS]
        if unknown_backends:
            raise CatalogError(f"{species_id} deliverable metric {index} has unknown backends: {unknown_backends}")
        untrained = sorted(set(backends) - expected)
        if untrained:
            raise CatalogError(
                f"{species_id} deliverable metric {index} names backends the species does not train: {untrained}"
            )
        if len(set(backends)) != len(backends):
            raise CatalogError(f"{species_id} deliverable metric {index} repeats a backend")
        for backend in backends:
            scope = (entry.id, str(backend))
            if scope in covered:
                raise CatalogError(
                    f"{species_id} defines more than one deliverable metric for stage {entry.id!r} on {backend}"
                )
            covered.add(scope)
        metrics.append(
            {
                "deliverable": deliverable,
                "stage_id": entry.id,
                "backends": [str(backend) for backend in backends],
                "key": _require_nonempty_string(
                    raw_metric.get("key"), field=f"key for {species_id} deliverable metric {index}"
                ),
                "label": _require_nonempty_string(
                    raw_metric.get("label"), field=f"label for {species_id} deliverable metric {index}"
                ),
                "definition": _require_nonempty_string(
                    raw_metric.get("definition"), field=f"definition for {species_id} deliverable metric {index}"
                ),
            }
        )

    uncovered = [entry.id for entry in manifest.deliverables if (entry.id, "stable-baselines3") not in covered]
    if uncovered:
        raise CatalogError(
            f"{species_id} deliverable_metrics must define a stable-baselines3 metric for every manifest "
            f"deliverable; missing {uncovered}"
        )
    return metrics


def build_catalog(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    plant_manifest_path: Path = DEFAULT_PLANT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Build and validate the deterministic public species catalog."""
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    plant_manifest_path = plant_manifest_path.resolve()
    plant_manifest = _load_plant_manifest(plant_manifest_path)
    raw_plant_entries = _require_mapping(plant_manifest["plants"], field="plant manifest plants")

    raw_capabilities = manifest.get("project_capabilities", {})
    capabilities: dict[str, dict[str, str]] = {}
    for capability_id, capability in raw_capabilities.items():
        status = capability.get("status")
        if status not in ALLOWED_CAPABILITY_STATUSES:
            raise CatalogError(f"invalid capability status for {capability_id}: {status}")
        capabilities[capability_id] = {"label": str(capability["label"]), "status": str(status)}

    raw_notebooks = sorted(manifest.get("notebooks", []), key=lambda item: int(item["display_order"]))
    notebook_ids: set[str] = set()
    notebooks: list[dict[str, Any]] = []
    for raw_notebook in raw_notebooks:
        notebook_id = str(raw_notebook["id"])
        if notebook_id in notebook_ids:
            raise CatalogError(f"duplicate notebook id: {notebook_id}")
        notebook_ids.add(notebook_id)
        notebook_path = str(raw_notebook["path"])
        _repo_path(notebook_path, field=f"notebook {notebook_id}")
        notebooks.append(
            {
                "id": notebook_id,
                "label": str(raw_notebook["label"]),
                "description": str(raw_notebook["description"]),
                "path": notebook_path,
            }
        )

    species_entries = sorted(manifest.get("species", []), key=lambda item: int(item["display_order"]))
    species_ids: set[str] = set()
    aliases: set[str] = set()
    manifested_result_paths: set[str] = set()
    species_catalog: list[dict[str, Any]] = []
    for raw_species in species_entries:
        species_id = str(raw_species["id"])
        if species_id in species_ids:
            raise CatalogError(f"duplicate species id: {species_id}")
        species_ids.add(species_id)

        species_aliases = [str(alias) for alias in raw_species.get("aliases", [])]
        duplicate_aliases = aliases.intersection(species_aliases)
        if duplicate_aliases:
            raise CatalogError(f"duplicate species aliases: {sorted(duplicate_aliases)}")
        aliases.update(species_aliases)

        model_relative_path = str(raw_species["model_path"])
        model_path = _repo_path(model_relative_path, field=f"{species_id} model")
        model: dict[str, Any] = {"path": model_relative_path, **_model_facts(model_path)}
        environment = _environment_facts(str(raw_species["env_entrypoint"]), int(model["nu"]))
        model["plant_contract"] = _public_plant_contract(
            species_id,
            raw_plant_entries.get(species_id),
            model_relative_path=model_relative_path,
            model=model,
            environment=environment,
        )

        training_notebook_ids = [str(notebook_id) for notebook_id in raw_species.get("training_notebooks", [])]
        training_backends = set(raw_species.get("training_backends", ALLOWED_TRAINING_BACKENDS))
        if not training_backends or training_backends - ALLOWED_TRAINING_BACKENDS:
            raise CatalogError(f"{species_id} has invalid training backends: {training_backends}")
        for notebook_id, backend in (("sb3_training", "stable-baselines3"), ("jax_training", "jax-mjx")):
            if notebook_id in training_notebook_ids and backend not in training_backends:
                raise CatalogError(f"{species_id} advertises unsupported notebook {notebook_id}")
        unknown_notebooks = sorted(set(training_notebook_ids) - notebook_ids)
        if unknown_notebooks:
            raise CatalogError(f"{species_id} references unknown notebooks: {unknown_notebooks}")

        result_summary_paths = [str(path) for path in raw_species.get("result_summary_paths", [])]
        duplicate_result_paths = manifested_result_paths.intersection(result_summary_paths)
        if duplicate_result_paths:
            raise CatalogError(f"result summaries are listed more than once: {sorted(duplicate_result_paths)}")
        manifested_result_paths.update(result_summary_paths)

        species_catalog.append(
            {
                "id": species_id,
                "display_name": str(raw_species["display_name"]),
                "tagline": str(raw_species["tagline"]),
                "gait": str(raw_species["gait"]),
                "specialty": str(raw_species["specialty"]),
                "aliases": species_aliases,
                "environment": environment,
                "model": model,
                "training_notebooks": training_notebook_ids,
                "success_metrics": _build_success_metrics(
                    species_id, raw_species.get("success_metrics"), training_backends
                ),
                "deliverable_metrics": _build_deliverable_metrics(
                    species_id, raw_species.get("deliverable_metrics", []), training_backends
                ),
                "stages": _build_stages(species_id, raw_species.get("stage_videos", [])),
                "historical_results": [_build_result(species_id, result_path) for result_path in result_summary_paths],
            }
        )

    discovered_result_paths = {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in (REPOSITORY_ROOT / "results").glob("*/*/summary.json")
    }
    if manifested_result_paths != discovered_result_paths:
        missing = sorted(discovered_result_paths - manifested_result_paths)
        unknown = sorted(manifested_result_paths - discovered_result_paths)
        raise CatalogError(f"manifest/result summary coverage mismatch; unlisted={missing}, unknown={unknown}")

    implemented_species = {
        path.name
        for path in (REPOSITORY_ROOT / "environments").iterdir()
        if path.is_dir() and (path / "envs").is_dir() and (path / "assets").is_dir()
    }
    variants = {str(entry["id"]): str(entry["variant_of"]) for entry in species_entries if "variant_of" in entry}
    for entry in species_entries:
        if entry["id"] in variants:
            parent = variants[entry["id"]]
            if parent not in species_ids - variants.keys() or parent not in implemented_species:
                raise CatalogError(f"invalid parent species for variant {entry['id']}: {parent}")
            if not str(entry["env_entrypoint"]).startswith(f"environments.{parent}.envs."):
                raise CatalogError(f"variant {entry['id']} must use its parent species' environment package")
    primary_species = species_ids - variants.keys()
    if primary_species != implemented_species:
        missing = sorted(implemented_species - primary_species)
        unknown = sorted(primary_species - implemented_species)
        raise CatalogError(f"manifest/implemented species coverage mismatch; unlisted={missing}, unknown={unknown}")

    plant_species = set(raw_plant_entries)
    if species_ids != plant_species:
        missing = sorted(species_ids - plant_species)
        unknown = sorted(plant_species - species_ids)
        raise CatalogError(f"plant/species manifest coverage mismatch; missing={missing}, unknown={unknown}")

    return {
        # v3 (2026-08-23): stage rows carry manifest identity (id / position /
        # label, nullable legacy number) and gate kinds, and species with a
        # semantic stage (trex recovery) list it in curriculum order.  The
        # nullable "number" is shape-breaking for a consumer that assumed an
        # integer, hence the bump.
        # v4 (2026-09-12, decision D-A8): stage rows carry the recipe DAG
        # (deliverable / warm_start_from / recipe), result rows publish
        # per-deliverable certification with a gate-kind headline
        # (deliverables / primary_deliverable / target_deliverable, [] and
        # nulls for every schema-2/3 ladder summary), per-stage result rows
        # carry recipe / deliverable, and species carry deliverable_metrics
        # beside the untouched success_metrics.  The website adapter guards
        # on this number, hence the bump.
        "schema_version": 4,
        "manifest_path": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "plant_manifest": {
            "path": plant_manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "schema": PLANT_MANIFEST_SCHEMA,
            "fingerprint_tool_version": int(plant_manifest["fingerprint_tool_version"]),
            "generated_with": dict(plant_manifest["generated_with"]),
        },
        "project_capabilities": capabilities,
        "notebooks": notebooks,
        "species": species_catalog,
    }


def render_catalog_json(catalog: dict[str, Any]) -> str:
    """Serialize a catalog deterministically for source control."""
    return json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"


def _format_millions(value: int | float | None) -> str:
    if value is None:
        return "—"
    millions = value / 1_000_000
    return f"{millions:g}M"


def _format_number(value: int | float | None, *, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.2f}{suffix}"


def _format_percent(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _format_verdict(stage: dict[str, Any]) -> str:
    """Render a stage verdict with its gate provenance (mirrored in the site's TSX).

    A verdict earned under a gate the species no longer declares — or under
    no recorded gate at all, which every pre-provenance summary is — reads as
    a pass/fail of that RETIRED gate, never as a bare "Yes" beneath the
    current gate's description.
    """
    passed = stage["stage_passed"]
    if passed is None:
        return "—"
    if not stage["gate_retired"]:
        return "Yes" if passed else "No"
    gate = stage["gate_kind"] or "reward gate"
    return f"{'passed' if passed else 'failed'} retired gate ({gate})"


def _format_evaluation_episodes(value: int | None) -> str:
    if value is None:
        return "evaluation episode count not recorded"
    return f"{value} evaluation episodes"


def _format_backend(backend: str, version: str | None = None) -> str:
    label = {"stable-baselines3": "Stable-Baselines3", "jax-mjx": "JAX/MJX"}.get(backend, backend)
    return f"{label} {version}" if version else f"{label} (version not recorded)"


def _success_metric_for_backend(species: dict[str, Any], backend: str) -> dict[str, Any]:
    for metric in species["success_metrics"]:
        if backend in metric["backends"]:
            return cast(dict[str, Any], metric)
    raise CatalogError(f"{species['id']} has no success metric for backend {backend}")


def _stage_heading(stage: dict[str, Any]) -> str:
    """``1 — Balance`` / ``recovery — Recovery``: the label the tables address a stage by."""
    return f"{stage['label']} — {stage['title']}"


def _format_recipe(stage: dict[str, Any]) -> str:
    """The behavior label a stage belongs to, tagged when its checkpoint is published."""
    recipe = stage.get("recipe")
    if recipe is None:
        return "—"
    return f"{recipe} (deliverable)" if stage.get("deliverable") else str(recipe)


def _format_warm_start(stage: dict[str, Any], stage_headings: dict[str, str]) -> str:
    """The parent row's heading, or a dash for a root node."""
    parent = stage.get("warm_start_from")
    if parent is None:
        return "—"
    try:
        return stage_headings[parent]
    except KeyError:
        # The stage manifest only accepts edges to declared earlier entries,
        # so this is a catalog bug, not a manifest error.
        raise CatalogError(
            f"stage {stage['id']} warm-starts from {parent!r}, which the species does not list"
        ) from None


def _format_headline_metric(metric: dict[str, Any]) -> str:
    """``task success 96.7%`` / ``unsupported duty 95% UCB not recorded`` (D-A9)."""
    value = metric["value"]
    if value is None:
        return f"{metric['label']} not recorded"
    unit = metric["unit"]
    if unit == "percent":
        rendered = _format_percent(value)
    elif unit == "m/s":
        rendered = _format_number(value, suffix=" m/s")
    else:
        rendered = _format_number(value)
    return f"{metric['label']} {rendered}"


def _format_deliverable(deliverable: dict[str, Any], *, primary: bool) -> str:
    """One published deliverable with its certification, gate, replication and headline."""
    status = "certified" if deliverable["certified"] else "not certified"
    if primary:
        status += ", primary"
    details = [
        status,
        f"gate {deliverable['gate_kind'] or 'not recorded'}",
        f"{deliverable['replication_count']} run" + ("" if deliverable["replication_count"] == 1 else "s"),
        *(_format_headline_metric(metric) for metric in deliverable["headline"]),
    ]
    behavior = deliverable["recipe"] or deliverable["id"]
    return f"{deliverable['label']} — {behavior} ({'; '.join(details)})"


def _format_advancement_gate(gate: dict[str, Any]) -> str:
    # A none/v1 stage is a recorded NON-ADVANCING pilot: it has no criteria
    # to list, and rendering the episode/consecutive defaults below would
    # dress the placeholder up as a permissive gate.  Say what it is, and —
    # when the stage vocabulary knows its measured successor (recovery →
    # recovery_quality/v1) — what it is waiting on.
    if gate.get("gate_kind") == "none/v1":
        text = "non-advancing pilot (gate_kind none/v1); never advances"
        if gate.get("pending_gate_kind"):
            text += f"; gate {gate['pending_gate_kind']} pending calibration (P5)"
        return text
    # A recovery_quality/v1 verdict is produced once, post-stage, from the
    # frozen gate_resolution.json (curriculum/gate_resolver) -- the
    # in-training scheduler refuses the kind outright -- so the generic
    # consecutive-passes tail below would publish hysteresis that never
    # applies to it. Render the frozen criteria and say where the verdict
    # comes from.
    if gate.get("gate_kind") == "recovery_quality/v1":
        recovery_criteria = [f"recovery success LCB95 ≥ {gate['min_recovery_success_lcb']:g}"]
        if gate.get("min_paired_success_delta_lcb") is not None:
            recovery_criteria.append(
                f"paired Δ vs each required frozen null LCB95 ≥ {gate['min_paired_success_delta_lcb']:g}"
            )
        recovery_criteria.extend(
            [
                f"re-entry ≤ {gate['recovery_t_recover_steps']:g} steps + {gate['recovery_dwell_steps']:g}-step dwell",
                f"≥ {gate['min_eval_episodes']} episodes/evaluation",
                "verdict from the frozen gate_resolution.json (post-stage; fail-closed when absent or stale)",
            ]
        )
        return "; ".join(recovery_criteria)
    # A task_success/v1 verdict is produced once, post-stage, from the
    # selected checkpoint's per-episode evaluation_selected.csv (plan §4.4):
    # the exact binomial LCB95 on task success against the declared bar at
    # the declared panel size, with min_avg_reward as a collapse RAIL, never
    # the gate -- rendered as "reward rail" so it cannot be read as the
    # reward_and_length/v1 criterion. required_consecutive is in-training
    # scheduler hysteresis only (D-B3), so the generic consecutive-passes
    # tail is not rendered here either.
    if gate.get("gate_kind") == "task_success/v1":
        task_criteria = [f"task success LCB95 ≥ {gate['min_success_lcb']:g}"]
        if gate["min_avg_reward"] is not None:
            task_criteria.append(f"reward rail ≥ {gate['min_avg_reward']:g}")
        if gate["min_avg_episode_length"] is not None:
            task_criteria.append(f"episode length ≥ {gate['min_avg_episode_length']:g}")
        task_criteria.extend(
            [
                f"≥ {gate['min_eval_episodes']} episodes/evaluation",
                "verdict from the selected checkpoint's evaluation_selected.csv (post-stage; fail-closed when absent)",
            ]
        )
        return "; ".join(task_criteria)
    criteria: list[str] = []
    if gate["min_avg_reward"] is not None:
        criteria.append(f"reward ≥ {gate['min_avg_reward']:g}")
    if gate["min_avg_episode_length"] is not None:
        criteria.append(f"episode length ≥ {gate['min_avg_episode_length']:g}")
    if gate["min_avg_forward_velocity"] is not None:
        criteria.append(f"avg. velocity ≥ {gate['min_avg_forward_velocity']:g} m/s")
    if gate["min_success_rate"] is not None:
        criteria.append(f"task success ≥ {_format_percent(gate['min_success_rate'])}")
    if gate.get("min_full_horizon_fraction") is not None:
        criteria.append(f"full-horizon episodes ≥ {_format_percent(gate['min_full_horizon_fraction'])}")
    if gate.get("max_unsupported_duty") is not None:
        criteria.append(f"unsupported duty ≤ {gate['max_unsupported_duty']:g}")
    if gate.get("max_unsupported_duty_ucb") is not None:
        criteria.append(f"unsupported duty 95% upper bound ≤ {gate['max_unsupported_duty_ucb']:g}")
    criteria.append(f"≥ {gate['min_eval_episodes']} episodes/evaluation")
    criteria.append(f"{gate['required_consecutive']} consecutive passes")
    return "; ".join(criteria)


def render_readme_species(catalog: dict[str, Any]) -> str:
    """Render active species facts and current stages for the root README."""
    lines = [
        "The active-species tables are generated from `configs/species_manifest.toml`, the layered plant contract,",
        "the executable Gymnasium environments, compiled MJCF models, and current stage TOML files. The budgets and",
        "gates shown are for the Stable-Baselines3 curriculum path. Do not edit the generated block by hand.",
    ]
    for species in catalog["species"]:
        environment = species["environment"]
        model = species["model"]
        plant = model["plant_contract"]
        lines.extend(
            [
                "",
                *[f'<a id="{anchor}"></a>' for anchor in [species["id"], *species["aliases"]]],
                "",
                f"### {species['display_name']}",
                "",
                f"{species['tagline']}. **Specialty:** {species['specialty']}.",
                "",
                "| Generated specification | Value |",
                "|---|---|",
                f"| Observation dimension | {environment['observation_dim']} |",
                f"| Action dimension / actuators | {environment['action_dim']} |",
                f"| Generalized coordinates / velocities | nq={model['nq']}, nv={model['nv']} |",
                f"| Compiled dynamic model mass | {model['dynamic_mass_kg']:.1f} kg |",
                "| Plant contract revisions | "
                f"policy r{plant['policy_interface']['revision']}; "
                f"physics r{plant['physics']['revision']}; visual r{plant['visual']['revision']} "
                "([details](docs/PLANT_CONTRACT.md)) |",
                f"| Model | `{model['path']}` |",
                "",
                "| Current stage | Recipe | Warm-start from | Objective | SB3 configured budget | "
                "SB3 early-advancement gate |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        stage_headings = {stage["id"]: _stage_heading(stage) for stage in species["stages"]}
        for stage in species["stages"]:
            # The row label is the stage's canonical reference: the legacy
            # number where one exists, the semantic id (recovery) where none
            # does — the manifest is the authority, and no "1b" is invented.
            # The recipe column says which behavior the node belongs to and
            # whether its checkpoint is a published deliverable; the
            # warm-start column names the parent row by ITS label, so an
            # edge reads as the table reads.
            lines.append(
                f"| {stage_headings[stage['id']]} | {_format_recipe(stage)} | "
                f"{_format_warm_start(stage, stage_headings)} | {stage['description']} | "
                f"{_format_millions(stage['timesteps'])} | "
                f"{_format_advancement_gate(stage['advancement_gate'])} |"
            )
        lines.extend(["", "**Backend-specific success semantics:**"])
        for metric in species["success_metrics"]:
            scopes = " / ".join(
                _format_backend(backend).replace(" (version not recorded)", "") for backend in metric["backends"]
            )
            lines.append(f"- **{scopes} — {metric['label']}:** {metric['definition']}")
        lines.extend(["", "**Per-deliverable success semantics:**"])
        for metric in species["deliverable_metrics"]:
            scopes = " / ".join(
                _format_backend(backend).replace(" (version not recorded)", "") for backend in metric["backends"]
            )
            lines.append(
                f"- **{metric['deliverable']} ({stage_headings[metric['stage_id']]}) · {scopes} — "
                f"{metric['label']}:** {metric['definition']}"
            )
        model_package = Path(species["model"]["path"]).parent.parent.as_posix()
        lines.extend(["", f"[Full documentation →]({model_package}/README.md)"])
        if species["id"] == "velociraptor":
            lines.extend(
                [
                    "",
                    "[Hugging Face models →](https://huggingface.co/kuds/mesozoic-labs-velocipastor)",
                ]
            )
    return "\n".join(lines)


def render_readme_results(catalog: dict[str, Any]) -> str:
    """Render provenance-labelled historical experiment summaries."""
    lines = [
        "The summaries below are historical experiment records generated from the versioned JSON files under",
        "`results/`. They are not evidence for the current model revision unless provenance is marked both current",
        "and verified. Current stage budgets may therefore differ from the steps reported here.",
    ]
    for species in catalog["species"]:
        for result in species["historical_results"]:
            provenance = result["provenance"]
            result_metric = _success_metric_for_backend(species, result["backend"])
            lines.extend(
                [
                    "",
                    f"### {species['display_name']} ({result['algorithm'].upper()} · "
                    f"{_format_backend(result['backend']).replace(' (version not recorded)', '')}) — {result['date']}",
                    "",
                    f"**Provenance:** {provenance['model_revision_status'].title()} model; "
                    f"{provenance['verification_status']}; "
                    f"{_format_evaluation_episodes(provenance['evaluation_episodes'])}; "
                    f"{_format_backend(result['backend'], result['backend_version'])}. "
                    f"**Run total:** {_format_millions(result['total_timesteps'])} steps; "
                    f"{result['total_training_time'] or 'time not recorded'}. "
                    f"[Source summary]({result['summary_path']}).",
                    "",
                    "| Stage | Best eval reward | Avg. forward velocity | Task success | Trained steps | Passed |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for stage in result["stages"]:
                lines.append(
                    f"| {stage['label']} — {str(stage['name']).replace('_', ' ').title()} | "
                    f"{_format_number(stage['best_eval_reward'])} | "
                    f"{_format_number(stage['avg_forward_vel'], suffix=' m/s')} | "
                    f"{_format_percent(stage['mean_success_rate'])} | "
                    f"{_format_millions(stage['timesteps'])} | {_format_verdict(stage)} |"
                )
            # Only a schema-4 result publishes deliverables; the four
            # committed ladder summaries publish none, and this block is
            # byte-identical to its pre-Phase-A rendering for them (D-A10).
            if result["deliverables"]:
                lines.extend(
                    [
                        "",
                        "**Deliverables:** "
                        + " · ".join(
                            _format_deliverable(
                                deliverable, primary=deliverable["stage_key"] == result["primary_deliverable"]
                            )
                            for deliverable in result["deliverables"]
                        ),
                    ]
                )
            lines.extend(
                [
                    "",
                    f"**Current {_format_backend(result['backend']).replace(' (version not recorded)', '')} "
                    "catalog definition for this task label:** "
                    f"{result_metric['definition']}",
                ]
            )
    return "\n".join(lines)


def render_readme_notebooks(catalog: dict[str, Any]) -> str:
    """Render the public notebook inventory."""
    lines = ["| Notebook | Description |", "|---|---|"]
    for notebook in catalog["notebooks"]:
        lines.append(f"| [`{notebook['path']}`]({notebook['path']}) | {notebook['description']} |")
    return "\n".join(lines)


def _replace_generated_block(source: str, block_name: str, content: str) -> str:
    begin, end = README_BLOCKS[block_name]
    if source.count(begin) != 1 or source.count(end) != 1:
        raise CatalogError(f"README must contain exactly one {block_name} generated block")
    before, remainder = source.split(begin, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{begin}\n{content.rstrip()}\n{end}{after}"


def render_readme(source: str, catalog: dict[str, Any]) -> str:
    """Replace all generator-managed README sections deterministically."""
    rendered = source
    rendered = _replace_generated_block(rendered, "species", render_readme_species(catalog))
    rendered = _replace_generated_block(rendered, "results", render_readme_results(catalog))
    rendered = _replace_generated_block(rendered, "notebooks", render_readme_notebooks(catalog))
    return rendered


def write_catalog(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    readme_path: Path = DEFAULT_README_PATH,
    plant_manifest_path: Path = DEFAULT_PLANT_MANIFEST_PATH,
) -> None:
    """Build and write the generated catalog JSON and README blocks."""
    catalog = build_catalog(manifest_path, plant_manifest_path)
    readme_source = readme_path.read_text(encoding="utf-8")
    rendered_readme = render_readme(readme_source, catalog)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_catalog_json(catalog), encoding="utf-8")
    readme_path.write_text(rendered_readme, encoding="utf-8")


def check_catalog(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    readme_path: Path = DEFAULT_README_PATH,
    plant_manifest_path: Path = DEFAULT_PLANT_MANIFEST_PATH,
) -> None:
    """Raise when committed catalog outputs are missing or stale."""
    catalog = build_catalog(manifest_path, plant_manifest_path)
    expected = render_catalog_json(catalog)
    if not output_path.exists():
        raise CatalogError(f"generated catalog is missing: {output_path.relative_to(REPOSITORY_ROOT)}")
    actual = output_path.read_text(encoding="utf-8")
    if actual != expected:
        raise CatalogError(
            "generated species catalog is stale; run "
            "`python -m environments.shared.species_catalog` and commit the result"
        )
    readme_source = readme_path.read_text(encoding="utf-8")
    if render_readme(readme_source, catalog) != readme_source:
        raise CatalogError(
            "generated README species data is stale; run "
            "`python -m environments.shared.species_catalog` and commit the result"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--plant-manifest", type=Path, default=DEFAULT_PLANT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README_PATH)
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when generated data is stale")
    args = parser.parse_args(argv)

    try:
        if args.check:
            check_catalog(args.output, args.manifest, args.readme, args.plant_manifest)
            print(f"Species catalog is current: {args.output} and {args.readme}")
        else:
            write_catalog(args.output, args.manifest, args.readme, args.plant_manifest)
            print(f"Wrote species catalog: {args.output} and {args.readme}")
    except CatalogError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
