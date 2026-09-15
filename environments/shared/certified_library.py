"""Immutable certified evidence, independent replication and conservative selection.

Adapters validate the domain-specific certificate before calling this module.
This layer verifies complete copied evidence, counts independent passing runs,
and selects only candidates with comparable paired benchmark measurements.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

_SCHEMA = "mesozoic.certified-candidate/v1"
_POLICY: dict[str, Any] = {"version": "paired-bootstrap-priority/v1", "resamples": 10000, "confidence": 0.95}
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CertifiedLibraryError(ValueError):
    """Invalid publication or evidence that no longer matches its manifest."""


class NoRecommendedCandidate(LookupError):
    """No verified, sufficiently replicated recommendation exists for this key."""


def _encode(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CertifiedLibraryError("Library metadata must contain finite JSON values") from exc


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CertifiedLibraryError("JSON object keys must be strings")
        value = {key: _json_copy(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_json_copy(item) for item in value]
    return json.loads(_encode(value))


def _digest(value: Any) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _hash_file(path: Path) -> str:
    _check_path(path)
    if not path.is_file():
        raise CertifiedLibraryError(f"Evidence is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_path(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise CertifiedLibraryError(f"Symlinks are not allowed in certified evidence paths: {path}")


def _mkdir(path: Path) -> None:
    _check_path(path)
    path.mkdir(parents=True, exist_ok=True)
    _check_path(path)


def _relative(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or PurePosixPath(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise CertifiedLibraryError(f"Unsafe evidence-relative path: {name!r}")
    return name


def _key(key: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(key)
    if not isinstance(result, dict):
        raise CertifiedLibraryError("Compatibility key must be an object")
    for name in ("species", "algorithm", "behavior"):
        value = result.get(name)
        if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
            raise CertifiedLibraryError(f"Invalid compatibility key component: {name}")
    return result


def _scope(library: Path, key: Mapping[str, Any]) -> Path:
    path = Path(library).absolute().joinpath(key["species"], key["algorithm"], key["behavior"], _digest(key))
    _check_path(path)
    return path


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    _mkdir(directory)
    lock_path = directory / ".library.lock"
    _check_path(lock_path)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_atomic(path: Path, value: Any) -> None:
    _check_path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encode(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    _check_path(path)
    try:
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        _encode(value)
        return value
    except (OSError, ValueError) as exc:
        raise CertifiedLibraryError(f"Cannot read certified manifest: {path}") from exc


def _comparison(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result = _json_copy(value)
    if (
        not isinstance(result, dict)
        or not {"protocol", "metrics"} <= result.keys()
        or set(result) - {"protocol", "metrics", "model_sha256", "normalization_sha256"}
    ):
        raise CertifiedLibraryError("Comparison requires protocol and ordered metrics")
    for name in ("model_sha256", "normalization_sha256"):
        if name in result:
            digest = result[name]
            if not isinstance(digest, str) or not _SHA.fullmatch(digest.removeprefix("sha256:")):
                raise CertifiedLibraryError(f"Invalid comparison {name}")
            result[name] = digest.removeprefix("sha256:")
    protocol, metrics = result["protocol"], result["metrics"]
    if not isinstance(protocol, dict) or not isinstance(protocol.get("version"), str) or not protocol["version"]:
        raise CertifiedLibraryError("Comparison protocol requires a version")
    seeds = protocol.get("episode_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise CertifiedLibraryError("Comparison protocol requires at least two distinct episode seeds")
    if not isinstance(metrics, list) or not metrics:
        raise CertifiedLibraryError("Comparison requires at least one metric")
    names = set()
    for metric in metrics:
        if not isinstance(metric, dict) or set(metric) - {"name", "values", "direction", "margin", "sample_keys"}:
            raise CertifiedLibraryError("Invalid comparison metric fields")
        metric_name, values = metric.get("name"), metric.get("values")
        margin = metric.get("margin")
        if not isinstance(metric_name, str) or not metric_name or metric_name in names:
            raise CertifiedLibraryError("Comparison metric names must be nonempty and unique")
        names.add(metric_name)
        if metric.get("direction") not in ("higher", "lower"):
            raise CertifiedLibraryError("Metric direction must be higher or lower")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not np.isfinite(margin) or margin < 0:
            raise CertifiedLibraryError("Metric margin must be finite and nonnegative")
        if (
            not isinstance(values, list)
            or len(values) < 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) or not np.isfinite(item) for item in values
            )
        ):
            raise CertifiedLibraryError("Metric values must contain at least two finite numbers")
        sample_keys = metric.setdefault("sample_keys", [str(seed) for seed in seeds])
        if (
            not isinstance(sample_keys, list)
            or len(sample_keys) != len(values)
            or any(not isinstance(item, str) or not item for item in sample_keys)
            or len(set(sample_keys)) != len(sample_keys)
        ):
            raise CertifiedLibraryError("Metric sample_keys must uniquely identify every value")
    return result


def _verified(directory: Path, key: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    payload = {name: value for name, value in manifest.items() if name != "version"}
    if (
        manifest.get("schema") != _SCHEMA
        or manifest.get("version") != directory.name
        or _digest(payload) != directory.name
        or manifest.get("key") != key
    ):
        raise CertifiedLibraryError(f"Candidate identity mismatch: {directory}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise CertifiedLibraryError("Candidate evidence list is missing")
    actual = set()
    for path in directory.rglob("*"):
        _check_path(path)
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(files) | {"manifest.json"}:
        raise CertifiedLibraryError("Candidate contains missing or unrecorded evidence files")
    for name, description in files.items():
        path = directory / _relative(name)
        if not isinstance(description, dict) or description.get("sha256") != _hash_file(path):
            raise CertifiedLibraryError(f"Candidate evidence hash mismatch: {path}")
        if description.get("size_bytes") != path.stat().st_size:
            raise CertifiedLibraryError(f"Candidate evidence size mismatch: {path}")
    if manifest.get("model_path") not in files or manifest.get("normalization_path") not in files:
        raise CertifiedLibraryError("Candidate checkpoint pair is missing")
    _comparison(manifest.get("comparison"))
    return manifest


def _candidates(scope: Path, key: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    candidates, invalid = {}, []
    versions = scope / "versions"
    _check_path(versions)
    if versions.exists():
        for directory in sorted(versions.iterdir()):
            if not _SHA.fullmatch(directory.name):
                continue
            try:
                candidates[directory.name] = _verified(directory, key)
            except (CertifiedLibraryError, OSError):
                invalid.append(directory.name)
    return candidates, invalid


def _replication(candidate: Mapping[str, Any], candidates: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    group = [item for item in candidates.values() if item["recipe_sha256"] == candidate["recipe_sha256"]]
    required = max(item["required_seeds"] for item in group)
    by_seed: dict[int, dict[str, str]] = {}
    for item in sorted(group, key=lambda value: value["version"]):
        if item["certificate"]["passed"]:
            model_hash = item["files"][item["model_path"]]["sha256"]
            by_seed.setdefault(item["training_seed"], {}).setdefault(model_hash, item["version"])
    # Maximum bipartite matching prevents duplicate seeds or copied models
    # from counting, including when one seed has multiple legitimate models.
    assigned: dict[str, int] = {}

    def match(seed: int, visited: set[str]) -> bool:
        for model_hash in sorted(by_seed[seed]):
            if model_hash in visited:
                continue
            visited.add(model_hash)
            if model_hash not in assigned or match(assigned[model_hash], visited):
                assigned[model_hash] = seed
                return True
        return False

    for seed in sorted(by_seed):
        match(seed, set())
    return {
        "distinct_seeds": len(assigned),
        "required_seeds": required,
        "training_seeds": sorted(assigned.values()),
        "contributing_versions": sorted(by_seed[seed][model_hash] for model_hash, seed in assigned.items()),
        "eligible": bool(candidate["certificate"]["passed"] and len(assigned) >= required),
    }


def _paired_comparison(challenger: dict[str, Any], incumbent: dict[str, Any]) -> dict[str, Any]:
    left, right = challenger.get("comparison"), incumbent.get("comparison")
    if left is None or right is None:
        return {"promote": False, "reason": "missing_comparison"}
    if left["protocol"] != right["protocol"]:
        return {"promote": False, "reason": "different_comparison_protocol"}
    if len(left["metrics"]) != len(right["metrics"]):
        return {"promote": False, "reason": "different_comparison_metrics"}
    results = []
    for new, old in zip(left["metrics"], right["metrics"], strict=True):
        if any(new[name] != old[name] for name in ("name", "direction", "margin")) or set(new["sample_keys"]) != set(
            old["sample_keys"]
        ):
            return {"promote": False, "reason": "different_comparison_metrics"}
        previous = dict(zip(old["sample_keys"], old["values"], strict=True))
        following = dict(zip(new["sample_keys"], new["values"], strict=True))
        differences = np.asarray([following[key] - previous[key] for key in sorted(previous)], dtype=np.float64)
        if not np.all(np.isfinite(differences)):
            return {"promote": False, "reason": "nonfinite_paired_difference"}
        if new["direction"] == "lower":
            differences *= -1
        seed = int(_digest({"protocol": left["protocol"], "metric": new["name"], "policy": _POLICY})[:16], 16)
        rng = np.random.default_rng(seed)
        estimates = np.empty(int(_POLICY["resamples"]))
        for start in range(0, len(estimates), 256):
            count = min(256, len(estimates) - start)
            indices = rng.integers(0, len(differences), size=(count, len(differences)))
            estimates[start : start + count] = (differences[indices] / len(differences)).sum(axis=1)
        lower, upper = np.quantile(estimates, [0.05, 0.95])
        if not np.isfinite(lower) or not np.isfinite(upper):
            return {"promote": False, "reason": "nonfinite_confidence_interval"}
        results.append(
            {
                "name": new["name"],
                "mean_difference": float((differences / len(differences)).sum()),
                "lower": float(lower),
                "upper": float(upper),
                "margin": new["margin"],
            }
        )
    if any(item["lower"] < -item["margin"] for item in results):
        return {"promote": False, "reason": "regression_or_uncertain_safety", "metrics": results}
    for item in results:
        if item["lower"] >= -item["margin"] and item["upper"] <= item["margin"]:
            continue
        if item["lower"] > item["margin"]:
            return {
                "promote": True,
                "reason": "paired_improvement",
                "priority_metric": item["name"],
                "metrics": results,
            }
        return {"promote": False, "reason": "uncertain_priority_metric", "metrics": results}
    return {"promote": False, "reason": "equivalent_within_margins", "metrics": results}


def _bind_comparison(comparison: dict[str, Any], candidate: Mapping[str, Any], *, required: bool) -> None:
    for field, path in (("model_sha256", "model_path"), ("normalization_sha256", "normalization_path")):
        if required or field in comparison:
            if comparison.get(field) != candidate["files"][candidate[path]]["sha256"]:
                raise CertifiedLibraryError(f"Comparison {field} does not match the selected candidate")


def _effective_comparison(scope: Path, pointer: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    evidence_id = pointer.get("comparison_evidence")
    if evidence_id is None:
        return cast(dict[str, Any] | None, candidate["comparison"])
    if not isinstance(evidence_id, str) or not _SHA.fullmatch(evidence_id):
        raise CertifiedLibraryError("Invalid recommendation comparison reference")
    evidence = _read_json(scope / "comparisons" / f"{evidence_id}.json")
    if _digest(evidence) != evidence_id or evidence.get("version") != candidate["version"]:
        raise CertifiedLibraryError("Recommendation comparison evidence identity mismatch")
    comparison = _comparison(evidence.get("comparison"))
    if comparison is None:
        raise CertifiedLibraryError("Recommendation comparison evidence is missing")
    _bind_comparison(comparison, candidate, required=True)
    return comparison


def _select(
    scope: Path,
    key: Mapping[str, Any],
    trigger_version: str,
    incumbent_comparison: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates, invalid = _candidates(scope, key)
    replication = {version: _replication(item, candidates) for version, item in candidates.items()}
    pointer = scope / "recommendation.json"
    old_pointer = _read_json(pointer) if pointer.exists() else {}
    old = old_pointer.get("version")
    comparison_evidence = old_pointer.get("comparison_evidence")
    if old in candidates:
        candidates[old] = {**candidates[old], "comparison": _effective_comparison(scope, old_pointer, candidates[old])}
    if incumbent_comparison is not None:
        if old not in candidates:
            raise CertifiedLibraryError("Cannot attach comparison without a verified incumbent")
        _bind_comparison(incumbent_comparison, candidates[old], required=True)
        requested_comparison = candidates[trigger_version]["comparison"]
        if requested_comparison is None or incumbent_comparison["protocol"] != requested_comparison["protocol"]:
            raise CertifiedLibraryError("Incumbent comparison must use the candidate comparison protocol")
        evidence = {"version": old, "comparison": incumbent_comparison, "key": key}
        comparison_evidence = _digest(evidence)
        _mkdir(scope / "comparisons")
        evidence_path = scope / "comparisons" / f"{comparison_evidence}.json"
        if not evidence_path.exists():
            _write_atomic(evidence_path, evidence)
        candidates[old] = {**candidates[old], "comparison": incumbent_comparison}
    eligible = {
        version: item
        for version, item in candidates.items()
        if replication[version]["eligible"] and item["comparison"] is not None
    }
    selected = old if old in eligible else None
    decisions = []
    for version in sorted(eligible):
        if version == selected:
            continue
        decision = (
            {"promote": True, "reason": "first_eligible_benchmarked_candidate"}
            if selected is None
            else _paired_comparison(eligible[version], eligible[selected])
        )
        decisions.append({"candidate": version, "incumbent": selected, **decision})
        if decision["promote"]:
            selected = version
    report = {
        "schema": "mesozoic.certified-library-decision/v1",
        "key": key,
        "trigger_version": trigger_version,
        "previous_version": old,
        "version": selected,
        "source_run_id": candidates[selected]["source_run_id"] if selected else None,
        "policy": _POLICY,
        "decisions": decisions,
        "replication": replication,
        "invalid_versions": invalid,
        "incumbent_comparison_evidence": comparison_evidence,
    }
    decision_id = _digest(report)
    _mkdir(scope / "decisions")
    decision_path = scope / "decisions" / f"{decision_id}.json"
    if not decision_path.exists():
        _write_atomic(decision_path, report)
    _write_atomic(
        pointer,
        {
            "schema": "mesozoic.certified-recommendation/v1",
            "key": key,
            "version": selected,
            "decision": decision_id,
            "comparison_evidence": comparison_evidence if selected == old else None,
        },
    )
    return report, replication[trigger_version]


def publish_candidate(
    library: Path,
    *,
    key: dict,
    recipe_sha256: str,
    training_seed: int,
    source_run_id: str,
    files: Mapping[str, Path],
    model_path: str,
    normalization_path: str,
    certificate: dict,
    comparison: dict | None = None,
    incumbent_comparison: dict | None = None,
    required_seeds: int = 1,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Copy a passing or failed run, then reconsider verified recommendations."""
    key = _key(key)
    if not isinstance(recipe_sha256, str) or not _SHA.fullmatch(recipe_sha256.removeprefix("sha256:")):
        raise CertifiedLibraryError("recipe_sha256 must be a SHA-256 digest")
    recipe_sha256 = recipe_sha256.removeprefix("sha256:")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise CertifiedLibraryError("training_seed must be a nonnegative integer")
    if isinstance(required_seeds, bool) or not isinstance(required_seeds, int) or not 1 <= required_seeds <= 100:
        raise CertifiedLibraryError("required_seeds must be an integer in [1, 100]")
    if not isinstance(source_run_id, str) or not source_run_id.strip():
        raise CertifiedLibraryError("source_run_id must be nonempty")
    certificate = _json_copy(certificate)
    if not isinstance(certificate, dict) or type(certificate.get("passed")) is not bool:
        raise CertifiedLibraryError("certificate.passed must be an explicit boolean")
    comparison = _comparison(comparison)
    incumbent_comparison = _comparison(incumbent_comparison)
    evidence = {_relative(name): Path(path).absolute() for name, path in files.items()}
    if "manifest.json" in evidence or any(name.startswith("manifest.json/") for name in evidence):
        raise CertifiedLibraryError("manifest.json is reserved for the certified manifest")
    if model_path not in evidence or normalization_path not in evidence or model_path == normalization_path:
        raise CertifiedLibraryError("Distinct model and normalization paths must both occur in files")
    scope = _scope(library, key)
    with _locked(scope):
        versions = scope / "versions"
        _mkdir(versions)
        temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=scope))
        try:
            descriptions = {}
            for name, source in sorted(evidence.items()):
                source_hash = _hash_file(source)
                destination = temporary / name
                _mkdir(destination.parent)
                shutil.copyfile(source, destination, follow_symlinks=False)
                if _hash_file(destination) != source_hash or _hash_file(source) != source_hash:
                    raise CertifiedLibraryError(f"Evidence changed during publication: {source}")
                descriptions[name] = {"sha256": source_hash, "size_bytes": destination.stat().st_size}
            payload = {
                "schema": _SCHEMA,
                "key": key,
                "recipe_sha256": recipe_sha256,
                "training_seed": training_seed,
                "source_run_id": source_run_id,
                "files": descriptions,
                "model_path": model_path,
                "normalization_path": normalization_path,
                "certificate": certificate,
                "comparison": comparison,
                "required_seeds": required_seeds,
                "metadata": _json_copy(metadata or {}),
            }
            version = _digest(payload)
            manifest = {**payload, "version": version}
            if comparison is not None:
                _bind_comparison(comparison, manifest, required=False)
            _write_atomic(temporary / "manifest.json", manifest)
            directory = versions / version
            if directory.exists():
                if _verified(directory, key) != manifest:
                    raise CertifiedLibraryError("Existing immutable candidate differs from publication")
            else:
                os.rename(temporary, directory)
            decision, replication = _select(scope, key, version, incumbent_comparison)
            status = "failed" if not certificate["passed"] else "eligible" if replication["eligible"] else "provisional"
            return {
                "version": version,
                "directory": str(directory),
                "status": status,
                **replication,
                "recommended": decision["version"] == version,
                "recommended_version": decision["version"],
                "decision_reason": next(
                    (item["reason"] for item in reversed(decision["decisions"]) if item["candidate"] == version),
                    "incumbent_retained"
                    if decision["version"] == version
                    else "awaiting_comparison"
                    if comparison is None and replication["eligible"]
                    else status,
                ),
                "decision": decision,
            }
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _resolved(manifest: dict[str, Any], directory: Path, replication: dict[str, Any]) -> dict[str, Any]:
    return {
        **manifest,
        "manifest": manifest,
        "directory": str(directory),
        "model": str(directory / manifest["model_path"]),
        "normalizer": str(directory / manifest["normalization_path"]),
        "replication": replication,
        "distinct_seeds": replication["distinct_seeds"],
        "required_seeds": replication["required_seeds"],
    }


def _resolve(library: Path, key: dict, version: str | None = None) -> dict[str, Any]:
    key = _key(key)
    scope = _scope(library, key)
    if not scope.exists() or (version is None and not (scope / "recommendation.json").exists()):
        raise NoRecommendedCandidate("No certified recommendation exists for this compatibility key")
    with _locked(scope):
        pointer = _read_json(scope / "recommendation.json") if (scope / "recommendation.json").exists() else {}
        if version is None:
            if pointer.get("key") != key:
                raise NoRecommendedCandidate("Recommendation compatibility key differs")
            version = pointer.get("version")
        if not isinstance(version, str) or not _SHA.fullmatch(version):
            raise NoRecommendedCandidate("No eligible benchmarked recommendation exists for this compatibility key")
        if pointer.get("version") != version:
            pointer = {}
        candidates, _ = _candidates(scope, key)
        if version not in candidates:
            raise CertifiedLibraryError("Recommended candidate evidence failed integrity verification")
        manifest = candidates[version]
        replication = _replication(manifest, candidates)
        if not replication["eligible"] or manifest["comparison"] is None:
            raise NoRecommendedCandidate("The recommendation no longer has sufficient verified passing replicas")
        result = _resolved(manifest, scope / "versions" / version, replication)
        result["comparison"] = _effective_comparison(scope, pointer, manifest)
        evidence_id = pointer.get("comparison_evidence")
        result["comparison_evidence"] = (
            {"id": evidence_id, "path": str(scope / "comparisons" / f"{evidence_id}.json")}
            if evidence_id is not None
            else None
        )
        return result


def resolve_recommended(library: Path, key: dict) -> dict[str, Any]:
    """Resolve an exact compatible version, rechecking all replication evidence."""
    return _resolve(library, key)


def copy_recommended(library: Path, key: dict, run_dir: Path) -> dict[str, Any]:
    """Materialize the entire selected bundle within a portable training run."""
    return _copy_selected(resolve_recommended(library, key), run_dir)


def copy_version(library: Path, key: dict, version: str, run_dir: Path) -> dict[str, Any]:
    """Copy an explicit eligible version independently of later recommendation changes."""
    if not isinstance(version, str) or not _SHA.fullmatch(version):
        raise CertifiedLibraryError("version must be a complete candidate SHA-256 identifier")
    return _copy_selected(_resolve(library, key, version), run_dir)


def _copy_selected(selected: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    manifest = selected["manifest"]
    parent = Path(run_dir).absolute() / "certified_inputs" / manifest["key"]["behavior"]
    with _locked(parent):
        destination = parent / manifest["version"]
        if destination.exists():
            if _verified(destination, manifest["key"]) != manifest:
                raise CertifiedLibraryError("Existing run input differs from the immutable recommendation")
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
            try:
                for name in manifest["files"]:
                    source = Path(selected["directory"]) / name
                    _check_path(source)
                    target = temporary / name
                    _mkdir(target.parent)
                    shutil.copyfile(source, target, follow_symlinks=False)
                _write_atomic(temporary / "manifest.json", manifest)
                # Verify under its content-addressed name before exposing it.
                verification = temporary / manifest["version"]
                verification.mkdir()
                for child in list(temporary.iterdir()):
                    if child != verification:
                        child.rename(verification / child.name)
                _verified(verification, manifest["key"])
                verification.rename(destination)
            finally:
                shutil.rmtree(temporary)
        result = _resolved(manifest, destination, selected["replication"])
        result["comparison"] = selected["comparison"]
        evidence = selected["comparison_evidence"]
        if evidence is not None:
            value = _read_json(Path(evidence["path"]))
            if _digest(value) != evidence["id"]:
                raise CertifiedLibraryError("Comparison evidence changed while copying the recommendation")
            _mkdir(parent / "comparisons")
            path = parent / "comparisons" / f"{evidence['id']}.json"
            if path.exists():
                if _read_json(path) != value:
                    raise CertifiedLibraryError("Existing run comparison evidence differs from its identity")
            else:
                _write_atomic(path, value)
            result["comparison_evidence"] = {"id": evidence["id"], "path": str(path)}
        else:
            result["comparison_evidence"] = None
        selection = {
            "version": manifest["version"],
            "key": manifest["key"],
            "source_run_id": manifest["source_run_id"],
            "comparison_evidence": None if evidence is None else evidence["id"],
            "replication": selected["replication"],
        }
        _mkdir(parent / "selections")
        selection_path = parent / "selections" / f"{_digest(selection)}.json"
        if selection_path.exists():
            if _read_json(selection_path) != selection:
                raise CertifiedLibraryError("Existing run selection evidence differs from its identity")
        else:
            _write_atomic(selection_path, selection)
        result["selection_record"] = str(selection_path)
        return result
