"""Versioned, layered safety contract for the MuJoCo physics plants.

The contract separates four concerns that used to be conflated as "the XML":

* source closure -- exact MJCF and referenced asset bytes;
* policy interface -- ordered observations, actions, sensors, and controls;
* physics -- compiled dynamics, contacts, actuators, and reset state; and
* visual presentation -- render geometry, materials, cameras, and lights.

This separation lets a render-only model improvement invalidate videos without
silently invalidating a policy, while any control or physics change fails closed
until its human revision and generated manifest are deliberately updated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import io
import json
import logging
import math
import textwrap
import token
import tokenize
import tomllib
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import mujoco
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPECIES_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "species_manifest.toml"
PLANT_VERSIONS_PATH = REPOSITORY_ROOT / "configs" / "plant_versions.toml"
GENERATED_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "plant_manifest.generated.json"
BUNDLED_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "plant_manifest.generated.json"

PLANT_MANIFEST_SCHEMA = "mesozoic.plant-manifest/v1"
PLANT_IDENTITY_SCHEMA = "mesozoic.plant-identity/v1"
POLICY_INTERFACE_SCHEMA = "mesozoic.policy-interface/v1"
PHYSICS_SCHEMA = "mesozoic.mujoco-physics/v1"
VISUAL_SCHEMA = "mesozoic.mujoco-visual/v1"
SOURCE_SCHEMA = "mesozoic.source-closure/v1"
FINGERPRINT_TOOL_VERSION = 2
PORTABLE_FLOAT_SIGNIFICANT_DIGITS = 12
MODEL_IDENTITY_ATTRIBUTE = "_mesozoic_plant_identity"
_MISSING_IDENTITY = object()

_ACTION_MAPPING_MIDPOINT = "midpoint/v1"
_ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL = "home-keyframe-residual/v1"
_MIDPOINT_ACTION_MAPPING_DESCRIPTION = "clip[-1,1]-then-affine-to-ordered-ctrlrange/v1"

_logger = logging.getLogger(__name__)


class PlantContractError(ValueError):
    """Raised when a plant manifest is missing, stale, or internally invalid."""


class PlantCompatibilityError(PlantContractError):
    """Raised when a checkpoint was created for an incompatible plant."""


@dataclass(frozen=True)
class PlantVersion:
    """Human-reviewed revision counters and interface semantics for one species."""

    species: str
    physics_revision: int
    policy_interface_revision: int
    visual_revision: int
    observation_schema: str


@dataclass(frozen=True)
class PlantIdentity:
    """Portable identity embedded in checkpoints and training artifacts."""

    species: str
    model_path: str
    physics_revision: int
    policy_interface_revision: int
    visual_revision: int
    source_closure_sha256: str
    policy_interface_sha256: str
    physics_sha256: str
    visual_sha256: str
    nq: int
    nv: int
    nu: int
    observation_dim: int
    action_dim: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema": PLANT_IDENTITY_SCHEMA,
            "species": self.species,
            "model_path": self.model_path,
            "physics_revision": self.physics_revision,
            "policy_interface_revision": self.policy_interface_revision,
            "visual_revision": self.visual_revision,
            "source_closure_sha256": self.source_closure_sha256,
            "policy_interface_sha256": self.policy_interface_sha256,
            "physics_sha256": self.physics_sha256,
            "visual_sha256": self.visual_sha256,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PlantIdentity:
        """Parse and validate checkpoint metadata."""
        if value.get("schema") != PLANT_IDENTITY_SCHEMA:
            raise PlantCompatibilityError(
                f"unsupported plant identity schema: {value.get('schema')!r}; expected {PLANT_IDENTITY_SCHEMA}"
            )
        try:
            identity = cls(
                species=str(value["species"]),
                model_path=str(value["model_path"]),
                physics_revision=int(value["physics_revision"]),
                policy_interface_revision=int(value["policy_interface_revision"]),
                visual_revision=int(value["visual_revision"]),
                source_closure_sha256=str(value["source_closure_sha256"]),
                policy_interface_sha256=str(value["policy_interface_sha256"]),
                physics_sha256=str(value["physics_sha256"]),
                visual_sha256=str(value["visual_sha256"]),
                nq=int(value["nq"]),
                nv=int(value["nv"]),
                nu=int(value["nu"]),
                observation_dim=int(value["observation_dim"]),
                action_dim=int(value["action_dim"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlantCompatibilityError(f"invalid plant identity metadata: {exc}") from exc
        for field_name in (
            "source_closure_sha256",
            "policy_interface_sha256",
            "physics_sha256",
            "visual_sha256",
        ):
            try:
                _validate_digest(getattr(identity, field_name), field=field_name)
            except PlantContractError as exc:
                raise PlantCompatibilityError(f"invalid plant identity metadata: {exc}") from exc
        return identity

    def compatibility_errors(self, recorded: PlantIdentity) -> list[str]:
        """Describe policy- or physics-level incompatibilities.

        Source and visual revisions are intentionally excluded: a policy can be
        replayed after a render-only change when its interface and physics are
        identical.
        """
        checks = (
            "species",
            "physics_revision",
            "policy_interface_revision",
            "policy_interface_sha256",
            "physics_sha256",
            "nq",
            "nv",
            "nu",
            "observation_dim",
            "action_dim",
        )
        return [
            f"{name}: checkpoint={getattr(recorded, name)!r}, current={getattr(self, name)!r}"
            for name in checks
            if getattr(recorded, name) != getattr(self, name)
        ]


def _validate_digest(value: str, *, field: str) -> None:
    prefix = "sha256:"
    digest = value[len(prefix) :] if value.startswith(prefix) else ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PlantContractError(f"{field} must be sha256:<64 lowercase hex>, got {value!r}")


def _canonical_float(value: float) -> str:
    """Encode floats portably while preserving meaningful model changes.

    MuJoCo's compiler can differ by a few ULPs across CPU architectures for
    derived inertias, quaternions, and geometry sizes.  Twelve significant
    decimal digits leave ample headroom for that numerical noise while exact
    MJCF and referenced-asset bytes remain protected by the source closure.
    """
    value = float(value)
    if math.isnan(value):
        raise PlantContractError("plant fingerprint cannot encode NaN")
    if math.isinf(value):
        # Gymnasium spaces deliberately use infinite observation bounds.
        # Encode them explicitly instead of relying on non-standard JSON.
        return "+inf" if value > 0 else "-inf"
    if value == 0.0:
        return "0"
    return format(value, f".{PORTABLE_FLOAT_SIGNIFICANT_DIGITS}g")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise PlantContractError(f"unsupported fingerprint value: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    normalized = _canonical_value(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _semantic_digest(schema: str, payload: Any) -> str:
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json(payload))
    return f"sha256:{digest.hexdigest()}"


def _normalized_python_tokens(source: str) -> list[list[str]]:
    """Return a Python-version-stable semantic token stream.

    Comments, blank lines, and indentation width are ignored. Structural
    INDENT/DEDENT/NEWLINE markers are retained, so block structure remains
    part of the interface. Unlike ``ast.dump``, this representation does not
    change when Python adds fields to its AST nodes.
    """
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    normalized: list[list[str]] = []
    dedented = textwrap.dedent(source)
    try:
        tree = ast.parse(dedented)
    except SyntaxError as exc:
        raise PlantContractError(f"cannot parse policy-interface implementation: {exc}") from exc
    if any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree)):
        # Python 3.12 changed f-string tokenization (PEP 701).  Rejecting them
        # keeps the manifest portable across supported Python versions until
        # the contract defines a dedicated cross-version representation.
        raise PlantContractError("f-strings are not supported in fingerprinted policy-interface callables")
    docstring_ranges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            value = body[0].value
            docstring_ranges.add(
                ((value.lineno, value.col_offset), (value.end_lineno or value.lineno, value.end_col_offset or 0))
            )

    reader = io.StringIO(dedented).readline
    skip_docstring_newline = False
    try:
        tokens = tokenize.generate_tokens(reader)
        for item in tokens:
            if item.type in ignored:
                continue
            if item.type == tokenize.STRING and (item.start, item.end) in docstring_ranges:
                skip_docstring_newline = True
                continue
            if skip_docstring_newline and item.type == tokenize.NEWLINE:
                skip_docstring_newline = False
                continue
            token_name = token.tok_name[item.type]
            if item.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE}:
                normalized.append([token_name, ""])
            elif item.type == tokenize.STRING:
                try:
                    literal = ast.literal_eval(item.string)
                except (SyntaxError, ValueError):
                    literal = item.string
                normalized.append([token_name, repr(literal)])
            else:
                normalized.append([token_name, item.string])
    except (IndentationError, tokenize.TokenError) as exc:
        raise PlantContractError(f"cannot tokenize policy-interface implementation: {exc}") from exc
    return normalized


def _callable_semantics(callable_object: Any) -> dict[str, str]:
    """Fingerprint executable interface code without importing optional backends."""
    try:
        source = inspect.getsource(callable_object)
    except (OSError, TypeError) as exc:
        raise PlantContractError(f"cannot inspect policy-interface callable {callable_object!r}: {exc}") from exc
    return {
        "qualname": str(getattr(callable_object, "__qualname__", type(callable_object).__qualname__)),
        "tokens_sha256": _semantic_digest(
            "mesozoic.python-interface-tokens/v1",
            _normalized_python_tokens(source),
        ),
    }


def _module_function_semantics(module_name: str, function_names: Sequence[str]) -> dict[str, str]:
    """Fingerprint selected functions from an optional-backend-safe module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PlantContractError(f"cannot import policy-interface module {module_name}: {exc}") from exc
    functions = {name: getattr(module, name, None) for name in function_names}
    missing = sorted(name for name, value in functions.items() if not callable(value))
    if missing:
        raise PlantContractError(f"{module_name} is missing policy-interface functions: {missing}")
    return {name: _callable_semantics(functions[name])["tokens_sha256"] for name in function_names}


def _action_mapping_contract(
    model: mujoco.MjModel,
    env: Any,
) -> tuple[str | dict[str, Any], tuple[str, ...]]:
    """Describe the species' normalized-action mapping and JAX implementation."""
    mode = str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT))
    if mode == _ACTION_MAPPING_MIDPOINT:
        return _MIDPOINT_ACTION_MAPPING_DESCRIPTION, ("scale_action_jax",)
    if mode != _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
        raise PlantContractError(f"unsupported action mapping mode: {mode!r}")

    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_keyframe_id < 0:
        raise PlantContractError("home-keyframe residual mapping requires a named 'home' keyframe")
    nominal_ctrl = np.asarray(model.key_ctrl[home_keyframe_id], dtype=np.float64)
    ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    if nominal_ctrl.shape != (model.nu,) or not np.all(np.isfinite(nominal_ctrl)):
        raise PlantContractError("home keyframe controls must be finite and match the actuator dimension")
    if np.any(nominal_ctrl < ctrl_range[:, 0]) or np.any(nominal_ctrl > ctrl_range[:, 1]):
        raise PlantContractError("home keyframe controls must remain inside every actuator control range")

    return (
        {
            "schema": "mesozoic.action-mapping/v1",
            "mode": mode,
            "input_clip": [-1.0, 1.0],
            "negative_endpoint": "ordered-ctrlrange-minimum",
            "origin": {
                "keyframe": "home",
                "keyframe_id": home_keyframe_id,
                "ctrl": nominal_ctrl,
            },
            "positive_endpoint": "ordered-ctrlrange-maximum",
            "interpolation": "piecewise-affine/v1",
        },
        ("scale_action_around_nominal_jax",),
    )


def _array_digest(schema: str, array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "sha256": _semantic_digest(schema, {"shape": list(array.shape), "values": array}),
    }


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PlantContractError(f"cannot read {path.relative_to(REPOSITORY_ROOT)}: {exc}") from exc


def _species_entries() -> dict[str, dict[str, Any]]:
    if not SPECIES_MANIFEST_PATH.is_file():
        generated = _load_generated_manifest()
        bundled_entries = {
            str(species): {
                "id": str(species),
                "model_path": entry.get("model_path"),
                "env_entrypoint": entry.get("env_entrypoint"),
            }
            for species, entry in generated.get("plants", {}).items()
        }
        if not bundled_entries or any(
            not entry.get("model_path") or not entry.get("env_entrypoint") for entry in bundled_entries.values()
        ):
            raise PlantContractError("bundled plant manifest lacks runtime species pointers")
        return bundled_entries

    raw = _read_toml(SPECIES_MANIFEST_PATH)
    entries: dict[str, dict[str, Any]] = {}
    for entry in raw.get("species", []):
        species = str(entry.get("id", ""))
        if not species or species in entries:
            raise PlantContractError(f"invalid or duplicate species id in {SPECIES_MANIFEST_PATH.name}: {species!r}")
        entries[species] = dict(entry)
    if not entries:
        raise PlantContractError("species manifest contains no species")
    return entries


def load_plant_versions() -> tuple[str, dict[str, PlantVersion]]:
    """Load human revision counters and require complete species coverage."""
    if not PLANT_VERSIONS_PATH.is_file():
        generated = _load_generated_manifest()
        canonical_mujoco = str(generated.get("generated_with", {}).get("mujoco", ""))
        bundled_versions: dict[str, PlantVersion] = {}
        for species, entry in generated.get("plants", {}).items():
            policy = entry.get("policy_interface", {})
            physics = entry.get("physics", {})
            visual = entry.get("visual", {})
            try:
                bundled_versions[str(species)] = PlantVersion(
                    species=str(species),
                    physics_revision=int(physics["revision"]),
                    policy_interface_revision=int(policy["revision"]),
                    visual_revision=int(visual["revision"]),
                    observation_schema=str(policy["observation_schema"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PlantContractError(f"invalid bundled plant version for {species}: {exc}") from exc
        if not canonical_mujoco or not bundled_versions:
            raise PlantContractError("bundled plant manifest lacks canonical version metadata")
        return canonical_mujoco, bundled_versions

    raw = _read_toml(PLANT_VERSIONS_PATH)
    if raw.get("schema_version") != 1:
        raise PlantContractError("plant_versions.toml schema_version must be 1")
    if raw.get("fingerprint_tool_version") != FINGERPRINT_TOOL_VERSION:
        raise PlantContractError(f"plant_versions.toml fingerprint_tool_version must be {FINGERPRINT_TOOL_VERSION}")
    canonical_mujoco = str(raw.get("canonical_mujoco_version", ""))
    if not canonical_mujoco:
        raise PlantContractError("plant_versions.toml must define canonical_mujoco_version")

    versions: dict[str, PlantVersion] = {}
    for species, value in raw.get("plants", {}).items():
        try:
            version = PlantVersion(
                species=str(species),
                physics_revision=int(value["physics_revision"]),
                policy_interface_revision=int(value["policy_interface_revision"]),
                visual_revision=int(value["visual_revision"]),
                observation_schema=str(value["observation_schema"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlantContractError(f"invalid version entry for {species}: {exc}") from exc
        if min(version.physics_revision, version.policy_interface_revision, version.visual_revision) < 1:
            raise PlantContractError(f"all plant revisions must be positive for {species}")
        if not version.observation_schema:
            raise PlantContractError(f"observation_schema must be non-empty for {species}")
        versions[species] = version

    species_ids = set(_species_entries())
    if set(versions) != species_ids:
        raise PlantContractError(
            f"plant version coverage mismatch: missing={sorted(species_ids - set(versions))}, "
            f"unknown={sorted(set(versions) - species_ids)}"
        )
    return canonical_mujoco, versions


def _resolve_repo_path(relative_path: str, *, field: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise PlantContractError(f"{field} must stay inside the repository: {relative_path}") from exc
    if not path.is_file():
        raise PlantContractError(f"{field} does not exist: {relative_path}")
    return path


def _source_dependencies(model_path: Path) -> list[Path]:
    """Return the recursive MJCF/include/asset dependency closure."""
    pending = [model_path.resolve()]
    visited: set[Path] = set()
    while pending:
        xml_path = pending.pop()
        if xml_path in visited:
            continue
        try:
            xml_path.relative_to(REPOSITORY_ROOT)
        except ValueError as exc:
            raise PlantContractError(f"MJCF dependency is outside the repository: {xml_path}") from exc
        visited.add(xml_path)
        try:
            root = ET.parse(xml_path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise PlantContractError(f"cannot parse MJCF dependency {xml_path}: {exc}") from exc

        compiler = root.find("compiler")
        asset_dir = xml_path.parent
        mesh_dir = asset_dir
        texture_dir = asset_dir
        if compiler is not None:
            if compiler.get("assetdir"):
                asset_dir = (xml_path.parent / str(compiler.get("assetdir"))).resolve()
            mesh_dir = (
                (xml_path.parent / str(compiler.get("meshdir"))).resolve() if compiler.get("meshdir") else asset_dir
            )
            texture_dir = (
                (xml_path.parent / str(compiler.get("texturedir"))).resolve()
                if compiler.get("texturedir")
                else asset_dir
            )

        for element in root.iter():
            file_name = element.get("file")
            if not file_name:
                continue
            if element.tag == "include":
                dependency = (xml_path.parent / file_name).resolve()
                pending.append(dependency)
                continue
            base_dir = (
                mesh_dir if element.tag in {"mesh", "skin"} else texture_dir if element.tag == "texture" else asset_dir
            )
            dependency = (base_dir / file_name).resolve()
            try:
                dependency.relative_to(REPOSITORY_ROOT)
            except ValueError as exc:
                raise PlantContractError(f"MJCF asset is outside the repository: {dependency}") from exc
            if not dependency.is_file():
                raise PlantContractError(f"MJCF asset does not exist: {dependency}")
            visited.add(dependency)
    return sorted(visited, key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix())


def _source_payload(model_path: Path) -> dict[str, Any]:
    dependencies = []
    for path in _source_dependencies(model_path):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        dependencies.append(
            {
                "logical_path": relative,
                "kind": "mjcf" if path.suffix.lower() == ".xml" else "asset",
                "content_sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            }
        )
    return {"root": model_path.relative_to(REPOSITORY_ROOT).as_posix(), "dependencies": dependencies}


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


def _fields(source: Any, names: Sequence[str], indices: np.ndarray | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in names:
        value = np.asarray(getattr(source, name))
        if indices is not None:
            value = value[indices]
        payload[name] = value
    return payload


def _option_payload(model: mujoco.MjModel) -> dict[str, Any]:
    # MuJoCo adds option fields over time.  The contract pins an exact MuJoCo
    # version, so fingerprint every public data attribute instead of relying
    # on an allow-list that can silently miss contact overrides or solver
    # controls introduced by a newer release.
    return {
        name: value
        for name in dir(model.opt)
        if not name.startswith("_") and not callable(value := getattr(model.opt, name))
    }


def _reject_unclassified_features(model: mujoco.MjModel) -> None:
    unsupported = []
    for count_name in ("nflex", "nplugin"):
        if int(getattr(model, count_name, 0)):
            unsupported.append(f"{count_name}={getattr(model, count_name)}")
    for field in ("body_plugin", "geom_plugin", "actuator_plugin", "sensor_plugin"):
        values = np.asarray(getattr(model, field, []))
        if values.size and np.any(values >= 0):
            unsupported.append(field)
    if unsupported:
        raise PlantContractError(
            "plant fingerprint does not yet classify these MuJoCo features: " + ", ".join(unsupported)
        )


def _physics_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    """Return geoms that can affect contacts, fluids, or tendon dynamics."""
    geom_ids = set(np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0)).tolist())
    if model.npair:
        geom_ids.update(int(value) for value in model.pair_geom1)
        geom_ids.update(int(value) for value in model.pair_geom2)
    fluid_rows = np.flatnonzero(np.any(np.asarray(model.geom_fluid) != 0.0, axis=1))
    geom_ids.update(int(value) for value in fluid_rows)
    if float(model.opt.density) != 0.0 or float(model.opt.viscosity) != 0.0 or model.ntendon:
        # The default fluid model and tendon wrapping can depend on otherwise
        # collision-disabled geometry, so classify all geoms conservatively.
        geom_ids.update(range(model.ngeom))
    return np.asarray(sorted(geom_ids), dtype=np.int64)


def _joint_component_names(model: mujoco.MjModel, *, velocity: bool) -> list[str]:
    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6 if velocity else 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 3 if velocity else 4,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    components: list[str] = []
    for index, joint_type in enumerate(model.jnt_type):
        if int(joint_type) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}"
        width = widths[int(joint_type)]
        components.extend(f"{name}[{component}]" for component in range(width))
    return components


def _space_payload(space: Any) -> dict[str, Any]:
    return {
        "shape": list(space.shape),
        "dtype": str(space.dtype),
        "low": np.asarray(space.low),
        "high": np.asarray(space.high),
    }


def _deterministic_probe_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Create synthetic observation state without platform-sensitive physics."""
    probe_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, probe_data)
    if model.nq:
        probe_data.qpos[:] = np.linspace(-0.2, 0.3, model.nq)
    if model.nv:
        probe_data.qvel[:] = np.linspace(-0.25, 0.35, model.nv)
    if model.nsensordata:
        probe_data.sensordata[:] = np.linspace(0.05, 0.05 * model.nsensordata, model.nsensordata)
    for index in range(model.nbody):
        probe_data.xpos[index] = (0.1 + 0.03 * index, -0.4 + 0.02 * index, 0.6 + 0.01 * index)
    for index in range(model.nmocap):
        probe_data.mocap_pos[index] = (1.25 + 0.1 * index, -0.45 + 0.05 * index, 0.8 + 0.02 * index)
    return probe_data


def _portable_probe_values(observation: np.ndarray) -> np.ndarray:
    """Quantize executable probe results across supported CPU architectures."""
    return np.round(np.asarray(observation, dtype=np.float64), decimals=6)


def _observation_probe(model: mujoco.MjModel, env: Any) -> dict[str, Any]:
    """Execute the observation ABI against deterministic, non-degenerate state.

    Source fingerprints catch formula edits; this probe also catches changes to
    cached body IDs, sensor offsets, and other runtime mappings used by the
    observation method.
    """
    probe_data = _deterministic_probe_data(model)

    original_model = env.model
    original_data = env.data
    try:
        env.model = model
        env.data = probe_data
        observation = np.asarray(env._get_obs())
    finally:
        env.model = original_model
        env.data = original_data

    expected_shape = tuple(env.observation_space.shape)
    if observation.shape != expected_shape or not np.all(np.isfinite(observation)):
        raise PlantContractError(
            f"observation probe produced invalid output: shape={observation.shape}, expected={expected_shape}"
        )
    return {
        "schema": "mesozoic.observation-probe/v1",
        "dtype": str(observation.dtype),
        "shape": list(observation.shape),
        "quantization_decimals": 6,
        "values": _portable_probe_values(observation),
    }


def _jax_policy_interface_payload(
    model: mujoco.MjModel,
    env: Any,
    version: PlantVersion,
) -> dict[str, Any]:
    """Resolve the MJX observation/control ABI without requiring JAX.

    Species registration modules only populate ordinary Python data.  Running
    their observation builder with NumPy catches stale numeric body IDs and
    sensor offsets while keeping the canonical contract usable in CPU-only CI.
    """
    module_name = f"environments.{version.species}.mjx_config"
    try:
        registration_module = importlib.import_module(module_name)
        mjx_env_module = importlib.import_module("environments.shared.mjx_env")
    except ImportError as exc:
        raise PlantContractError(f"cannot import MJX plant registration {module_name}: {exc}") from exc

    registry = getattr(mjx_env_module, "_SPECIES_CONFIGS", {})
    if version.species not in registry:
        # A test or embedding process may have cleared the registry after the
        # module's first import.  Reloading deterministically re-registers it.
        importlib.reload(registration_module)
    raw_config = getattr(mjx_env_module, "_SPECIES_CONFIGS", {}).get(version.species)
    if not isinstance(raw_config, Mapping):
        raise PlantContractError(f"MJX plant registration is missing for {version.species}")

    from .obs_functions import SensorLayout

    body_ids = {str(name): int(body_id) for name, body_id in dict(raw_config.get("body_ids", {})).items()}
    if not body_ids:
        raise PlantContractError(f"MJX plant registration has no root body mapping for {version.species}")
    resolved_bodies: dict[str, dict[str, Any]] = {}
    for body_name, registered_id in body_ids.items():
        resolved_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if resolved_id < 0:
            raise PlantContractError(f"MJX root body {body_name!r} does not exist for {version.species}")
        if registered_id != resolved_id:
            raise PlantContractError(
                f"MJX root body ID for {body_name!r} is stale: registered={registered_id}, resolved={resolved_id}"
            )
        resolved_bodies[body_name] = {"registered_id": registered_id, "resolved_id": resolved_id}

    root_name = "torso" if version.observation_schema == "quadrupedal-target/v1" else "pelvis"
    if root_name not in body_ids:
        raise PlantContractError(f"MJX plant registration lacks required {root_name!r} body mapping")

    sensor_layout = SensorLayout(
        gyro_start=int(raw_config.get("sensor_gyro_start", 0)),
        accel_start=int(raw_config.get("sensor_accel_start", 3)),
        quat_start=int(raw_config.get("sensor_quat_start", 6)),
        foot_indices=tuple(int(index) for index in raw_config.get("sensor_foot_indices", ())),
    )
    used_sensor_indices = (
        *range(sensor_layout.gyro_start, sensor_layout.gyro_start + 3),
        *range(sensor_layout.accel_start, sensor_layout.accel_start + 3),
        *range(sensor_layout.quat_start, sensor_layout.quat_start + 4),
        *sensor_layout.foot_indices,
    )
    if any(index < 0 or index >= model.nsensordata for index in used_sensor_indices):
        raise PlantContractError(
            f"MJX sensor layout for {version.species} indexes outside nsensordata={model.nsensordata}"
        )

    registered_frame_skip = int(raw_config.get("frame_skip", -1))
    if registered_frame_skip <= 0:
        raise PlantContractError(f"MJX frame_skip must be positive for {version.species}")
    registered_action_mapping = str(raw_config.get("action_mapping", _ACTION_MAPPING_MIDPOINT))
    sb3_action_mapping = str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT))
    if registered_action_mapping != sb3_action_mapping:
        raise PlantContractError(
            f"SB3 and MJX action mappings diverge for {version.species}: "
            f"sb3={sb3_action_mapping!r}, mjx={registered_action_mapping!r}"
        )

    probe_data = _deterministic_probe_data(model)
    target_pos = probe_data.mocap_pos[0] if model.nmocap else np.array([1.25, -0.45, 0.8])
    observation = np.asarray(
        mjx_env_module.build_mjx_observation(
            probe_data,
            target_pos,
            {
                "species": version.species,
                "body_ids": body_ids,
                "sensor_gyro_start": sensor_layout.gyro_start,
                "sensor_accel_start": sensor_layout.accel_start,
                "sensor_quat_start": sensor_layout.quat_start,
                "sensor_foot_indices": sensor_layout.foot_indices,
            },
        )
    )
    expected_shape = tuple(env.observation_space.shape)
    if observation.shape != expected_shape or not np.all(np.isfinite(observation)):
        raise PlantContractError(
            f"MJX observation probe produced invalid output: shape={observation.shape}, expected={expected_shape}"
        )

    payload = {
        "registration_module": module_name,
        "frame_skip": registered_frame_skip,
        "control_timestep": model.opt.timestep * registered_frame_skip,
        "body_ids": resolved_bodies,
        "sensor_layout": {
            "gyro_start": sensor_layout.gyro_start,
            "accel_start": sensor_layout.accel_start,
            "quat_start": sensor_layout.quat_start,
            "foot_indices": sensor_layout.foot_indices,
        },
        "observation_builder": (
            "build_quadruped_obs" if version.observation_schema == "quadrupedal-target/v1" else "build_bipedal_obs"
        ),
        "observation_probe": {
            "schema": "mesozoic.mjx-observation-probe/v1",
            "dtype": str(observation.dtype),
            "shape": list(observation.shape),
            "quantization_decimals": 6,
            "values": _portable_probe_values(observation),
        },
    }
    if registered_action_mapping != _ACTION_MAPPING_MIDPOINT:
        payload["action_mapping"] = registered_action_mapping
    return payload


def _policy_interface_payload(
    model: mujoco.MjModel,
    env: Any,
    version: PlantVersion,
    *,
    require_backend_parity: bool = False,
) -> dict[str, Any]:
    touch_sensor_names = [
        name
        for index, name in enumerate(_names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor))
        if int(model.sensor_type[index]) == int(mujoco.mjtSensor.mjSENS_TOUCH)
    ]
    root_label = "torso" if version.species == "brachiosaurus" else "pelvis"
    target_label = "food" if version.species == "brachiosaurus" else "prey"
    observation_functions = (
        ("_array_mod", "build_bipedal_obs", "build_quadruped_obs")
        if version.observation_schema == "quadrupedal-target/v1"
        else ("_array_mod", "build_bipedal_obs")
    )
    observation_segments = [
        {"name": "joint_position", "components": _joint_component_names(model, velocity=False)},
        {"name": "joint_velocity", "components": _joint_component_names(model, velocity=True)},
        {"name": f"{root_label}_orientation_quaternion", "width": 4},
        {"name": f"{root_label}_angular_velocity", "width": 3},
        {"name": f"{root_label}_linear_velocity", "width": 3},
        {"name": f"{root_label}_linear_acceleration", "width": 3},
        {"name": "foot_contact", "components": touch_sensor_names},
        {"name": f"{target_label}_direction", "width": 3, "normalization_epsilon": _canonical_float(1e-8)},
        {"name": f"{target_label}_distance", "width": 1},
    ]
    collision_ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    sb3_observation_probe = _observation_probe(model, env)
    jax_interface = _jax_policy_interface_payload(model, env, version)
    action_mapping, jax_action_functions = _action_mapping_contract(model, env)
    mjx_env_module = importlib.import_module("environments.shared.mjx_env")
    jax_setup_module = importlib.import_module("environments.shared.jax_setup")
    backend_observation_equal = np.array_equal(
        sb3_observation_probe["values"],
        jax_interface["observation_probe"]["values"],
    )
    if require_backend_parity and not backend_observation_equal:
        raise PlantContractError(
            f"SB3 and MJX observation probes diverge for {version.species}; "
            "check cached body IDs, sensor offsets, and observation builders"
        )
    interface_implementations = {
        "sb3_observation": _callable_semantics(env._get_obs),
        "sb3_action_mapping": _callable_semantics(env._scale_action),
        "backend_neutral_observation": _module_function_semantics(
            "environments.shared.obs_functions",
            observation_functions,
        ),
        "jax_action_mapping": _module_function_semantics(
            "environments.shared.mjx_utils",
            jax_action_functions,
        ),
        # These are intentionally production observation callables, not a
        # parallel test implementation. Reward/termination code stays out
        # of the interface fingerprint.
        "jax_observation_callers": {
            "training_reset_and_step": _callable_semantics(mjx_env_module.build_mjx_observation),
            "cpu_evaluation": _callable_semantics(jax_setup_module.make_obs_fn),
        },
    }
    if str(getattr(env, "action_mapping", _ACTION_MAPPING_MIDPOINT)) == _ACTION_MAPPING_HOME_KEYFRAME_RESIDUAL:
        interface_implementations["home_reset"] = {
            "sb3": _callable_semantics(env.reset),
            "jax": _module_function_semantics(
                "environments.shared.mjx_utils",
                ("reset_mujoco_data_to_home",),
            ),
        }
    return {
        "observation_schema": version.observation_schema,
        "observation": _space_payload(env.observation_space),
        "observation_probe": sb3_observation_probe,
        "observation_segments": observation_segments,
        "action": _space_payload(env.action_space),
        "action_mapping": action_mapping,
        "interface_implementations": interface_implementations,
        "jax_interface": jax_interface,
        "backend_observation_equal": backend_observation_equal,
        "frame_skip": int(env.frame_skip),
        "physics_timestep": model.opt.timestep,
        "control_timestep": model.opt.timestep * int(env.frame_skip),
        "dimensions": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
            "nsensor": int(model.nsensor),
            "nsensordata": int(model.nsensordata),
        },
        "bodies": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "joints": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            **_fields(model, ("jnt_type", "jnt_qposadr", "jnt_dofadr")),
        },
        "collision_geoms": [_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)[index] for index in collision_ids],
        "sites": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            **_fields(model, ("site_bodyid", "site_type", "site_pos", "site_quat", "site_size")),
        },
        "actuators": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
            **_fields(
                model,
                (
                    "actuator_trntype",
                    "actuator_trnid",
                    "actuator_ctrllimited",
                    "actuator_ctrlrange",
                ),
            ),
        },
        "sensors": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
            **_fields(
                model,
                (
                    "sensor_type",
                    "sensor_datatype",
                    "sensor_needstage",
                    "sensor_objtype",
                    "sensor_objid",
                    "sensor_reftype",
                    "sensor_refid",
                    "sensor_dim",
                    "sensor_adr",
                    "sensor_cutoff",
                    "sensor_noise",
                    "sensor_delay",
                    "sensor_interval",
                    "sensor_intprm",
                    "sensor_user",
                ),
            ),
        },
    }


def _physics_payload(model: mujoco.MjModel, version: PlantVersion) -> dict[str, Any]:
    _reject_unclassified_features(model)
    dynamics_geom_ids = _physics_geom_ids(model)
    payload: dict[str, Any] = {
        "options": _option_payload(model),
        "dimensions": {
            name: int(getattr(model, name))
            for name in (
                "nq",
                "nv",
                "nu",
                "na",
                "nbody",
                "njnt",
                "ngeom",
                "nsite",
                "nsensor",
                "ntendon",
                "neq",
                "npair",
                "nexclude",
                "nkey",
                "nmocap",
            )
        },
        "bodies": _fields(
            model,
            (
                "body_parentid",
                "body_mocapid",
                "body_pos",
                "body_quat",
                "body_ipos",
                "body_iquat",
                "body_mass",
                "body_inertia",
                "body_gravcomp",
            ),
        ),
        "joints": _fields(
            model,
            (
                "jnt_type",
                "jnt_bodyid",
                "jnt_pos",
                "jnt_axis",
                "jnt_limited",
                "jnt_range",
                "jnt_stiffness",
                "jnt_stiffnesspoly",
                "jnt_actfrclimited",
                "jnt_actfrcrange",
                "jnt_actgravcomp",
                "jnt_margin",
                "jnt_solref",
                "jnt_solimp",
            ),
        ),
        "dofs": _fields(
            model,
            (
                "dof_jntid",
                "dof_armature",
                "dof_damping",
                "dof_dampingpoly",
                "dof_frictionloss",
                "dof_solref",
                "dof_solimp",
            ),
        ),
        "dynamics_geoms": {
            "ids": dynamics_geom_ids,
            **_fields(
                model,
                (
                    "geom_bodyid",
                    "geom_type",
                    "geom_dataid",
                    "geom_pos",
                    "geom_quat",
                    "geom_size",
                    "geom_contype",
                    "geom_conaffinity",
                    "geom_condim",
                    "geom_friction",
                    "geom_priority",
                    "geom_solmix",
                    "geom_solref",
                    "geom_solimp",
                    "geom_margin",
                    "geom_gap",
                    "geom_fluid",
                ),
                dynamics_geom_ids,
            ),
        },
        "actuators": _fields(
            model,
            (
                "actuator_trntype",
                "actuator_trnid",
                "actuator_dyntype",
                "actuator_gaintype",
                "actuator_biastype",
                "actuator_actearly",
                "actuator_dynprm",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_gear",
                "actuator_cranklength",
                "actuator_length0",
                "actuator_armature",
                "actuator_damping",
                "actuator_dampingpoly",
                "actuator_delay",
                "actuator_ctrllimited",
                "actuator_ctrlrange",
                "actuator_forcelimited",
                "actuator_forcerange",
                "actuator_actlimited",
                "actuator_actrange",
                "actuator_lengthrange",
            ),
        ),
        "initialization": {
            "qpos0": model.qpos0,
            "qpos_spring": model.qpos_spring,
            **_fields(model, ("key_time", "key_qpos", "key_qvel", "key_act", "key_ctrl", "key_mpos", "key_mquat")),
        },
    }
    if model.npair:
        payload["contact_pairs"] = _fields(
            model,
            (
                "pair_geom1",
                "pair_geom2",
                "pair_dim",
                "pair_friction",
                "pair_solref",
                "pair_solreffriction",
                "pair_solimp",
                "pair_margin",
                "pair_gap",
            ),
        )
    if model.nexclude:
        payload["contact_exclusions"] = _fields(model, ("exclude_signature",))
    if model.neq:
        payload["equalities"] = _fields(
            model,
            ("eq_type", "eq_objtype", "eq_obj1id", "eq_obj2id", "eq_active0", "eq_solref", "eq_solimp", "eq_data"),
        )
    if model.ntendon:
        payload["tendons"] = _fields(
            model,
            (
                "tendon_adr",
                "tendon_num",
                "tendon_limited",
                "tendon_range",
                "tendon_stiffness",
                "tendon_stiffnesspoly",
                "tendon_damping",
                "tendon_dampingpoly",
                "tendon_frictionloss",
                "tendon_armature",
                "tendon_actfrclimited",
                "tendon_actfrcrange",
                "tendon_margin",
                "tendon_solref_lim",
                "tendon_solimp_lim",
                "tendon_solref_fri",
                "tendon_solimp_fri",
                "tendon_lengthspring",
                "wrap_type",
                "wrap_objid",
                "wrap_prm",
            ),
        )
    if model.nhfield:
        payload["heightfields"] = {
            **_fields(
                model,
                ("hfield_adr", "hfield_nrow", "hfield_ncol", "hfield_size"),
            ),
            "data": _array_digest("mesozoic.physics-heightfield-data/v1", model.hfield_data),
        }

    collision_mesh_ids = sorted(
        {
            int(model.geom_dataid[index])
            for index in dynamics_geom_ids
            if int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        }
    )
    if collision_mesh_ids:
        mesh_payload = []
        for mesh_id in collision_mesh_ids:
            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            face_adr = int(model.mesh_faceadr[mesh_id])
            face_num = int(model.mesh_facenum[mesh_id])
            mesh_payload.append(
                {
                    "mesh_id": mesh_id,
                    "vertices": _array_digest(
                        "mesozoic.collision-mesh-vertices/v1", model.mesh_vert[vert_adr : vert_adr + vert_num]
                    ),
                    "faces": _array_digest(
                        "mesozoic.collision-mesh-faces/v1", model.mesh_face[face_adr : face_adr + face_num]
                    ),
                    "scale": model.mesh_scale[mesh_id],
                    "pos": model.mesh_pos[mesh_id],
                    "quat": model.mesh_quat[mesh_id],
                }
            )
        payload["collision_meshes"] = mesh_payload
    return payload


def _visual_options(model: mujoco.MjModel) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name in ("global_", "quality", "headlight", "map", "scale", "rgba"):
        group = getattr(model.vis, group_name)
        result[group_name] = {
            name: getattr(group, name)
            for name in dir(group)
            if not name.startswith("_") and not callable(getattr(group, name))
        }
    return result


def _visual_payload(model: mujoco.MjModel, version: PlantVersion) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "geoms": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
            **_fields(
                model,
                (
                    "geom_bodyid",
                    "geom_type",
                    "geom_dataid",
                    "geom_pos",
                    "geom_quat",
                    "geom_size",
                    "geom_group",
                    "geom_matid",
                    "geom_rgba",
                ),
            ),
        },
        "sites": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            **_fields(
                model,
                (
                    "site_bodyid",
                    "site_type",
                    "site_pos",
                    "site_quat",
                    "site_size",
                    "site_group",
                    "site_matid",
                    "site_rgba",
                ),
            ),
        },
        "materials": _fields(
            model,
            (
                "mat_texid",
                "mat_texrepeat",
                "mat_texuniform",
                "mat_emission",
                "mat_specular",
                "mat_shininess",
                "mat_reflectance",
                "mat_metallic",
                "mat_roughness",
                "mat_rgba",
            ),
        ),
        "cameras": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
            **_fields(
                model,
                (
                    "cam_bodyid",
                    "cam_targetbodyid",
                    "cam_mode",
                    "cam_pos",
                    "cam_quat",
                    "cam_fovy",
                    "cam_ipd",
                    "cam_intrinsic",
                    "cam_projection",
                    "cam_resolution",
                    "cam_sensorsize",
                    "cam_output",
                    "cam_user",
                ),
            ),
        },
        "lights": {
            "names": _names(model, mujoco.mjtObj.mjOBJ_LIGHT, model.nlight),
            **_fields(
                model,
                (
                    "light_type",
                    "light_bodyid",
                    "light_targetbodyid",
                    "light_mode",
                    "light_pos",
                    "light_dir",
                    "light_active",
                    "light_castshadow",
                    "light_bulbradius",
                    "light_attenuation",
                    "light_cutoff",
                    "light_exponent",
                    "light_intensity",
                    "light_range",
                    "light_texid",
                    "light_ambient",
                    "light_diffuse",
                    "light_specular",
                ),
            ),
        },
        "visual_options": _visual_options(model),
    }
    if model.ntex:
        payload["textures"] = {
            **_fields(
                model,
                ("tex_type", "tex_height", "tex_width", "tex_nchannel", "tex_adr", "tex_colorspace"),
            ),
            "data": _array_digest("mesozoic.texture-data/v1", model.tex_data),
        }
    if model.nhfield:
        payload["heightfields"] = {
            **_fields(model, ("hfield_adr", "hfield_nrow", "hfield_ncol", "hfield_size")),
            "data": _array_digest("mesozoic.visual-heightfield-data/v1", model.hfield_data),
        }
    if model.nmesh:
        payload["meshes"] = {
            **_fields(
                model,
                ("mesh_vertadr", "mesh_vertnum", "mesh_faceadr", "mesh_facenum", "mesh_scale", "mesh_pos", "mesh_quat"),
            ),
            "vertices": _array_digest("mesozoic.visual-mesh-vertices/v1", model.mesh_vert),
            "faces": _array_digest("mesozoic.visual-mesh-faces/v1", model.mesh_face),
            "normals": _array_digest("mesozoic.visual-mesh-normals/v1", model.mesh_normal),
            "texcoords": _array_digest("mesozoic.visual-mesh-texcoords/v1", model.mesh_texcoord),
        }
    if model.nskin:
        payload["skins"] = {
            **_fields(
                model,
                (
                    "skin_matid",
                    "skin_group",
                    "skin_rgba",
                    "skin_inflate",
                    "skin_vertadr",
                    "skin_vertnum",
                    "skin_faceadr",
                    "skin_facenum",
                ),
            ),
            "vertices": _array_digest("mesozoic.skin-vertices/v1", model.skin_vert),
            "faces": _array_digest("mesozoic.skin-faces/v1", model.skin_face),
        }
    return payload


def _load_environment(entrypoint: str) -> type[Any]:
    module_name, separator, class_name = entrypoint.partition(":")
    if not separator:
        raise PlantContractError(f"invalid environment entrypoint: {entrypoint}")
    try:
        return cast(type[Any], getattr(importlib.import_module(module_name), class_name))
    except (ImportError, AttributeError) as exc:
        raise PlantContractError(f"cannot import environment entrypoint {entrypoint}: {exc}") from exc


def fingerprint_model_layers(
    model: mujoco.MjModel,
    env: Any,
    version: PlantVersion,
    *,
    require_backend_parity: bool = False,
) -> dict[str, str]:
    """Fingerprint a compiled model's policy, physics, and visual layers.

    This small public seam also supports mutation tests and model-authoring
    tools without exposing the large canonical payloads as persisted APIs.
    """
    return {
        "policy_interface_sha256": _semantic_digest(
            POLICY_INTERFACE_SCHEMA,
            _policy_interface_payload(
                model,
                env,
                version,
                require_backend_parity=require_backend_parity,
            ),
        ),
        "physics_sha256": _semantic_digest(PHYSICS_SCHEMA, _physics_payload(model, version)),
        "visual_sha256": _semantic_digest(VISUAL_SCHEMA, _visual_payload(model, version)),
    }


def _manifest_entry_for_identity(
    species: str,
    entry: Mapping[str, Any],
    version: PlantVersion,
) -> tuple[dict[str, Any], PlantIdentity]:
    model_relative_path = str(entry["model_path"])
    model_path = _resolve_repo_path(model_relative_path, field=f"{species} model")
    env_class = _load_environment(str(entry["env_entrypoint"]))
    env = env_class(reset_noise_scale=0.0)
    try:
        model = env.model
        source_payload = _source_payload(model_path)
        source_digest = _semantic_digest(SOURCE_SCHEMA, source_payload)
        layer_digests = fingerprint_model_layers(model, env, version, require_backend_parity=True)
        declared_model = mujoco.MjModel.from_xml_path(str(model_path))
        declared_digests = fingerprint_model_layers(declared_model, env, version, require_backend_parity=True)
        if declared_digests != layer_digests:
            raise PlantContractError(
                f"{species} environment model does not match the declared source {model_relative_path}"
            )
        policy_digest = layer_digests["policy_interface_sha256"]
        physics_digest = layer_digests["physics_sha256"]
        visual_digest = layer_digests["visual_sha256"]
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        identity = PlantIdentity(
            species=species,
            model_path=model_relative_path,
            physics_revision=version.physics_revision,
            policy_interface_revision=version.policy_interface_revision,
            visual_revision=version.visual_revision,
            source_closure_sha256=source_digest,
            policy_interface_sha256=policy_digest,
            physics_sha256=physics_digest,
            visual_sha256=visual_digest,
            nq=int(model.nq),
            nv=int(model.nv),
            nu=int(model.nu),
            observation_dim=observation_dim,
            action_dim=action_dim,
        )
    finally:
        env.close()

    bundle_digest = _semantic_digest(
        PLANT_MANIFEST_SCHEMA,
        {
            "species": species,
            "source": source_digest,
            "policy_interface": policy_digest,
            "physics": physics_digest,
            "visual": visual_digest,
        },
    )
    manifest_entry = {
        "species": species,
        "model_path": model_relative_path,
        "env_entrypoint": str(entry["env_entrypoint"]),
        "source": {"schema": SOURCE_SCHEMA, **source_payload, "closure_sha256": source_digest},
        "policy_interface": {
            "schema": POLICY_INTERFACE_SCHEMA,
            "revision": version.policy_interface_revision,
            "observation_schema": version.observation_schema,
            "sha256": policy_digest,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        },
        "physics": {
            "schema": PHYSICS_SCHEMA,
            "revision": version.physics_revision,
            "sha256": physics_digest,
            "nq": identity.nq,
            "nv": identity.nv,
            "nu": identity.nu,
        },
        "visual": {"schema": VISUAL_SCHEMA, "revision": version.visual_revision, "sha256": visual_digest},
        "bundle_sha256": bundle_digest,
    }
    return manifest_entry, identity


def build_plant_manifest(*, require_canonical_mujoco: bool = False) -> dict[str, Any]:
    """Build the current generated plant manifest from executable sources."""
    canonical_mujoco, versions = load_plant_versions()
    if require_canonical_mujoco and mujoco.__version__ != canonical_mujoco:
        raise PlantContractError(
            f"plant manifest generation requires MuJoCo {canonical_mujoco}, found {mujoco.__version__}"
        )
    entries = _species_entries()
    plants = {
        species: _manifest_entry_for_identity(species, entries[species], versions[species])[0]
        for species in sorted(entries)
    }
    return {
        "schema": PLANT_MANIFEST_SCHEMA,
        "fingerprint_tool_version": FINGERPRINT_TOOL_VERSION,
        "generated_with": {
            "mujoco": mujoco.__version__,
            "float_significant_digits": PORTABLE_FLOAT_SIGNIFICANT_DIGITS,
        },
        "plants": plants,
    }


def render_plant_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize the generated manifest deterministically for source control."""
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _load_generated_manifest(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = GENERATED_MANIFEST_PATH if GENERATED_MANIFEST_PATH.is_file() else BUNDLED_MANIFEST_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlantContractError(f"cannot read generated plant manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != PLANT_MANIFEST_SCHEMA:
        raise PlantContractError(f"generated plant manifest must use schema {PLANT_MANIFEST_SCHEMA}")
    return value


def _enforce_revision_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    old_plants = old.get("plants", {})
    new_plants = new.get("plants", {})
    for species in sorted(set(old_plants) & set(new_plants)):
        old_entry = old_plants[species]
        new_entry = new_plants[species]
        for layer, revision_field in (
            ("policy_interface", "policy_interface_revision"),
            ("physics", "physics_revision"),
            ("visual", "visual_revision"),
        ):
            old_layer = old_entry[layer]
            new_layer = new_entry[layer]
            old_revision = int(old_layer["revision"])
            new_revision = int(new_layer["revision"])
            if new_revision < old_revision:
                raise PlantContractError(
                    f"{species} {revision_field} cannot decrease from {old_revision} to {new_revision}"
                )
            if old_layer["sha256"] == new_layer["sha256"]:
                continue
            if new_revision <= old_revision:
                raise PlantContractError(
                    f"{species} {layer} fingerprint changed without increasing {revision_field} "
                    f"above {old_layer['revision']} in {PLANT_VERSIONS_PATH.relative_to(REPOSITORY_ROOT)}"
                )


def write_plant_manifest() -> Path:
    """Regenerate the committed manifest after enforcing revision increments."""
    if not SPECIES_MANIFEST_PATH.is_file() or not PLANT_VERSIONS_PATH.is_file():
        raise PlantContractError("plant manifest generation requires a source checkout with configs/")
    new_manifest = build_plant_manifest(require_canonical_mujoco=True)
    if GENERATED_MANIFEST_PATH.exists():
        _enforce_revision_changes(_load_generated_manifest(), new_manifest)
    rendered = render_plant_manifest(new_manifest)
    for output_path in (GENERATED_MANIFEST_PATH, BUNDLED_MANIFEST_PATH):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".json.tmp")
        temp_path.write_text(rendered, encoding="utf-8")
        temp_path.replace(output_path)
    return GENERATED_MANIFEST_PATH


def check_plant_manifest(*, baseline_path: Path | None = None) -> dict[str, Any]:
    """Fail when the committed generated manifest is stale."""
    expected = render_plant_manifest(build_plant_manifest(require_canonical_mujoco=True))
    source_path = GENERATED_MANIFEST_PATH if GENERATED_MANIFEST_PATH.is_file() else BUNDLED_MANIFEST_PATH
    try:
        actual = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlantContractError(f"generated plant manifest is missing: {exc}") from exc
    if actual != expected:
        raise PlantContractError(
            "generated plant manifest is stale; if semantic layers changed, bump affected revisions in "
            "configs/plant_versions.toml, then run `python -m environments.shared.plant_contract --write`"
        )
    if GENERATED_MANIFEST_PATH.is_file():
        try:
            bundled = BUNDLED_MANIFEST_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise PlantContractError(f"bundled plant manifest is missing: {exc}") from exc
        if bundled != actual:
            raise PlantContractError("bundled plant manifest is stale; regenerate with `--write`")
    current = cast(dict[str, Any], json.loads(actual))
    if baseline_path is not None:
        _enforce_revision_changes(_load_generated_manifest(baseline_path), current)
    return current


def current_plant_identity(species: str, *, verify_generated: bool = True) -> PlantIdentity:
    """Return the executable identity for one species and optionally verify it."""
    _, versions = load_plant_versions()
    entries = _species_entries()
    if species not in entries:
        raise PlantContractError(f"unknown species: {species}")
    manifest_entry, identity = _manifest_entry_for_identity(species, entries[species], versions[species])
    if verify_generated:
        generated = _load_generated_manifest().get("plants", {}).get(species)
        if generated != manifest_entry:
            raise PlantContractError(
                f"generated plant manifest is stale for {species}; run the plant-contract check before training"
            )
    return identity


def validate_compiled_plant(
    model: mujoco.MjModel,
    current: PlantIdentity,
    *,
    artifact: str = "runtime MuJoCo model",
) -> None:
    """Prove that a backend's compiled model matches the tagged plant.

    This is intentionally independent of Gymnasium and JAX.  It prevents a
    backend from loading a different MJCF and then stamping checkpoints with
    the identity computed from the canonical species entrypoint.
    """
    _, versions = load_plant_versions()
    version = versions.get(current.species)
    if version is None:
        raise PlantCompatibilityError(f"{artifact} uses unknown species {current.species!r}")
    actual_physics = _semantic_digest(PHYSICS_SCHEMA, _physics_payload(model, version))
    errors = []
    for field_name in ("nq", "nv", "nu"):
        actual = int(getattr(model, field_name))
        expected = int(getattr(current, field_name))
        if actual != expected:
            errors.append(f"{field_name}: runtime={actual}, current={expected}")
    if actual_physics != current.physics_sha256:
        errors.append(f"physics_sha256: runtime={actual_physics!r}, current={current.physics_sha256!r}")
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is not the current {current.species} plant:\n- " + "\n- ".join(errors)
        )


def validate_environment_plant(
    env: Any,
    current: PlantIdentity,
    *,
    artifact: str = "runtime environment",
) -> None:
    """Validate the actual Gymnasium environment before tagging artifacts."""
    model = getattr(env, "model", None)
    if not isinstance(model, mujoco.MjModel):
        raise PlantCompatibilityError(f"{artifact} exposes no MuJoCo model for plant validation")
    _, versions = load_plant_versions()
    version = versions.get(current.species)
    if version is None:
        raise PlantCompatibilityError(f"{artifact} uses unknown species {current.species!r}")
    layers = fingerprint_model_layers(model, env, version, require_backend_parity=True)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    errors = []
    checks = {
        "physics_sha256": layers["physics_sha256"],
        "policy_interface_sha256": layers["policy_interface_sha256"],
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }
    for field_name, actual in checks.items():
        expected = getattr(current, field_name)
        if actual != expected:
            errors.append(f"{field_name}: runtime={actual!r}, current={expected!r}")
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is not the current {current.species} plant:\n- " + "\n- ".join(errors)
        )


def validate_mjx_environment_plant(
    env: Any,
    current: PlantIdentity,
    *,
    artifact: str = "runtime MJX environment",
) -> None:
    """Bind an actual MJX model and observation/control config to identity."""
    model = getattr(env, "mj_model", None)
    if not isinstance(model, mujoco.MjModel):
        raise PlantCompatibilityError(f"{artifact} exposes no compiled mj_model for plant validation")
    validate_compiled_plant(model, current, artifact=artifact)

    try:
        importlib.import_module(f"environments.{current.species}.mjx_config")
        mjx_env_module = importlib.import_module("environments.shared.mjx_env")
        expected = mjx_env_module._SPECIES_CONFIGS[current.species]
    except (ImportError, KeyError) as exc:
        raise PlantCompatibilityError(f"{artifact} has no canonical MJX registration: {exc}") from exc
    config = getattr(env, "config", None)
    if config is None:
        raise PlantCompatibilityError(f"{artifact} exposes no MJX config for plant validation")

    fields = (
        "frame_skip",
        "body_ids",
        "sensor_foot_indices",
        "sensor_gyro_start",
        "sensor_accel_start",
        "sensor_quat_start",
        "action_mapping",
    )
    errors = []
    if str(getattr(config, "species", "")) != current.species:
        errors.append(f"species: runtime={getattr(config, 'species', None)!r}, current={current.species!r}")
    for field_name in fields:
        actual_value = getattr(config, field_name, None)
        expected_value = (
            expected.get(field_name, _ACTION_MAPPING_MIDPOINT)
            if field_name == "action_mapping"
            else expected.get(field_name)
        )
        if actual_value != expected_value:
            errors.append(f"{field_name}: runtime={actual_value!r}, registered={expected_value!r}")
    action_dim = int(getattr(env, "action_dim", -1))
    if action_dim != current.action_dim:
        errors.append(f"action_dim: runtime={action_dim}, current={current.action_dim}")
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is not the current {current.species} MJX interface:\n- " + "\n- ".join(errors)
        )


def write_plant_identity(path: str | Path, identity: PlantIdentity) -> Path:
    """Atomically write a checkpoint/run identity sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return path


def attach_plant_identity(model: Any, identity: PlantIdentity) -> None:
    """Attach identity metadata that Stable-Baselines3 persists in its ZIP."""
    setattr(model, MODEL_IDENTITY_ATTRIBUTE, identity.to_dict())


def validate_recorded_identity(
    recorded: Mapping[str, Any] | None,
    current: PlantIdentity,
    *,
    artifact: str = "checkpoint",
    allow_legacy: bool = False,
) -> None:
    """Validate persisted metadata, failing closed unless legacy use is explicit."""
    if recorded is None:
        if allow_legacy:
            _logger.warning(
                "%s has no plant identity; loading only because allow_legacy_plant was explicitly enabled",
                artifact,
            )
            return
        raise PlantCompatibilityError(
            f"{artifact} has no plant identity. It predates the plant-safety contract and cannot be loaded "
            "silently. Re-run with allow_legacy_plant=True only for deliberate historical migration/evaluation."
        )
    checkpoint_identity = PlantIdentity.from_mapping(recorded)
    errors = current.compatibility_errors(checkpoint_identity)
    if errors:
        raise PlantCompatibilityError(
            f"{artifact} is incompatible with the current {current.species} plant:\n- " + "\n- ".join(errors)
        )


def validate_model_plant(
    model: Any,
    current: PlantIdentity,
    *,
    artifact: str = "checkpoint",
    allow_legacy: bool = False,
) -> None:
    """Validate identity embedded in a loaded SB3 model."""
    raw = getattr(model, MODEL_IDENTITY_ATTRIBUTE, _MISSING_IDENTITY)
    if raw is _MISSING_IDENTITY:
        validate_recorded_identity(None, current, artifact=artifact, allow_legacy=allow_legacy)
        return
    if not isinstance(raw, Mapping):
        raise PlantCompatibilityError(
            f"{artifact} contains invalid plant identity metadata of type {type(raw).__name__}"
        )
    validate_recorded_identity(raw, current, artifact=artifact, allow_legacy=allow_legacy)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify the layered MuJoCo plant manifest")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify the committed manifest")
    action.add_argument("--write", action="store_true", help="regenerate after deliberate revision bumps")
    action.add_argument("--show", action="store_true", help="print the currently computed manifest")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="with --check, enforce revision monotonicity against a base manifest",
    )
    args = parser.parse_args()
    if args.baseline is not None and not args.check:
        parser.error("--baseline requires --check")
    try:
        if args.write:
            path = write_plant_manifest()
            print(f"Wrote {path.relative_to(REPOSITORY_ROOT)}")
        elif args.check:
            check_plant_manifest(baseline_path=args.baseline)
            print("Plant manifest is current")
        else:
            print(render_plant_manifest(build_plant_manifest()))
    except PlantContractError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
