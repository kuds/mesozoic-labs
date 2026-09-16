"""Evidence preservation, replication and paired recommendation selection."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from environments.shared.certified_library import (
    CertifiedLibraryError,
    NoRecommendedCandidate,
    copy_recommended,
    copy_version,
    publish_candidate,
    resolve_recommended,
)

KEY = {"species": "trex", "algorithm": "ppo", "behavior": "walk", "plant": "fixed", "parent": None}
RECIPE = "a" * 64


def comparison(values=None, *, count=6, second=None, protocol="paired/v1"):
    values = [0.5] * count if values is None else values
    result = {
        "protocol": {"version": protocol, "episode_seeds": list(range(count))},
        "metrics": [{"name": "survival", "values": values, "direction": "higher", "margin": 0.01}],
    }
    if second is not None:
        result["metrics"].append({"name": "falls", "values": second, "direction": "lower", "margin": 0.01})
    return result


@pytest.fixture
def publish(tmp_path):
    counter = 0

    def run(*, seed=1, model=None, passed=True, benchmark=True, **kwargs):
        nonlocal counter
        counter += 1
        source = tmp_path / "sources" / str(counter)
        source.mkdir(parents=True)
        (source / "model.zip").write_bytes(model if model is not None else f"model-{seed}".encode())
        (source / "norm.pkl").write_bytes(b"normalizer")
        (source / "report.json").write_text(json.dumps({"passed": passed}))
        arguments = {
            "key": KEY,
            "recipe_sha256": RECIPE,
            "training_seed": seed,
            "source_run_id": f"run-{seed}",
            "files": {
                "snapshots/walk/model.zip": source / "model.zip",
                "snapshots/walk/norm.pkl": source / "norm.pkl",
                "evidence/report.json": source / "report.json",
            },
            "model_path": "snapshots/walk/model.zip",
            "normalization_path": "snapshots/walk/norm.pkl",
            "certificate": {"passed": passed},
            "comparison": comparison() if benchmark else None,
        }
        arguments.update(kwargs)
        return publish_candidate(tmp_path / "library", **arguments)

    return run


def test_first_candidate_full_copy_is_portable_and_idempotent(tmp_path, publish):
    first = publish()
    assert first["status"] == "eligible" and first["recommended"]
    original = Path(first["directory"]) / "manifest.json"
    original_bytes = original.read_bytes()
    assert publish()["version"] == first["version"]
    assert original.read_bytes() == original_bytes
    resolved = resolve_recommended(tmp_path / "library", KEY)
    assert resolved["distinct_seeds"] == 1
    copied = copy_recommended(tmp_path / "library", KEY, tmp_path / "training-run")
    assert copied["version"] == first["version"]
    assert Path(copied["directory"]).is_relative_to(tmp_path / "training-run")
    assert copied["manifest"] == resolved["manifest"]
    for name in copied["files"]:
        assert (Path(copied["directory"]) / name).read_bytes() == (Path(resolved["directory"]) / name).read_bytes()
    assert copy_recommended(tmp_path / "library", KEY, tmp_path / "training-run")["version"] == first["version"]


def test_failed_outcome_retained_and_missing_comparison_not_recommended(tmp_path, publish):
    failed = publish(passed=False, required_seeds=2)
    assert failed["status"] == "failed" and not failed["recommended"]
    assert Path(failed["directory"], "manifest.json").exists()
    with pytest.raises(NoRecommendedCandidate):
        resolve_recommended(tmp_path / "library", KEY)
    unbenchmarked = publish(seed=2, benchmark=False, required_seeds=2)
    assert unbenchmarked["status"] == "provisional"
    eligible = publish(seed=3, benchmark=False, required_seeds=2)
    assert eligible["eligible"] and not eligible["recommended"]
    assert eligible["decision_reason"] == "awaiting_comparison"


def test_replication_requires_distinct_verified_passing_seeds_and_models(tmp_path, publish):
    first = publish(seed=1, required_seeds=3)
    assert first["status"] == "provisional"
    duplicate = publish(seed=2, model=b"model-1", required_seeds=3)
    assert duplicate["distinct_seeds"] == 1
    same_seed = publish(seed=1, model=b"different-model", required_seeds=3)
    assert same_seed["distinct_seeds"] == 2  # seed1 has another independent checkpoint; seed2 uses model1
    failed = publish(seed=3, passed=False, required_seeds=3)
    assert failed["distinct_seeds"] == 2
    third = publish(seed=3, required_seeds=3)
    assert third["distinct_seeds"] == 3 and third["eligible"]
    selected = resolve_recommended(tmp_path / "library", KEY)
    assert selected["distinct_seeds"] == 3
    # Corrupt a necessary contributor without changing its manifest. It stops
    # counting even when a different immutable version was recommended.
    Path(third["directory"], "snapshots/walk/model.zip").write_bytes(b"corrupted")
    with pytest.raises((NoRecommendedCandidate, CertifiedLibraryError)):
        resolve_recommended(tmp_path / "library", KEY)


def test_recipes_and_exact_compatibility_keys_never_share_replicas(tmp_path, publish):
    publish(required_seeds=2)
    other = publish(seed=2, recipe_sha256="b" * 64, required_seeds=2)
    assert other["distinct_seeds"] == 1 and not other["eligible"]
    with pytest.raises(NoRecommendedCandidate):
        resolve_recommended(tmp_path / "library", {**KEY, "plant": "changed"})
    distinct = publish(seed=3, key={**KEY, "plant": "changed"}, required_seeds=2)
    assert distinct["distinct_seeds"] == 1
    # A caller cannot silently lower the recipe's established requirement.
    lowered = publish(seed=4, required_seeds=1)
    assert lowered["required_seeds"] == 2


def test_clear_paired_improvement_promotes_and_keeps_all_versions(tmp_path, publish):
    first = publish(comparison=comparison([0.5] * 6, second=[0.2] * 6))
    better = publish(seed=2, comparison=comparison([0.8] * 6, second=[0.1] * 6))
    assert better["recommended"] and better["decision_reason"] == "paired_improvement"
    assert resolve_recommended(tmp_path / "library", KEY)["version"] == better["version"]
    assert Path(first["directory"]).exists()
    assert list(Path(first["directory"]).parents[1].joinpath("decisions").glob("*.json"))


@pytest.mark.parametrize("kind", ["equal", "regression", "noisy", "different_protocol", "priority_uncertain"])
def test_uncertainty_regression_or_incomparable_evidence_retains_incumbent(tmp_path, publish, kind):
    first = publish(comparison=comparison([0.5] * 6, second=[0.2] * 6))
    challenger = comparison([0.8] * 6, second=[0.1] * 6)
    if kind == "equal":
        challenger = comparison([0.5] * 6, second=[0.2] * 6)
    elif kind == "regression":
        challenger["metrics"][1]["values"] = [0.4] * 6
    elif kind == "noisy":
        challenger["metrics"][0]["values"] = [0, 1, 0, 1, 1, 1]
    elif kind == "different_protocol":
        challenger["protocol"]["version"] = "paired/v2"
    else:
        challenger["metrics"][0]["values"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.6]
    result = publish(seed=2, comparison=challenger)
    assert not result["recommended"]
    assert resolve_recommended(tmp_path / "library", KEY)["version"] == first["version"]


def test_metric_subset_keys_pair_by_identity_not_position(tmp_path, publish):
    benchmark = comparison()
    benchmark["metrics"][0].update(values=[0.1, 0.9], sample_keys=["family:seed1", "family:seed2"])
    first = publish(comparison=benchmark)
    following = copy.deepcopy(benchmark)
    following["metrics"][0].update(values=[0.9, 0.1], sample_keys=["family:seed2", "family:seed1"])
    result = publish(seed=2, comparison=following)
    assert not result["recommended"]
    assert resolve_recommended(tmp_path / "library", KEY)["version"] == first["version"]


def test_incumbent_can_be_rebenchmarked_without_mutating_original(tmp_path, publish):
    first = publish()
    old_manifest = Path(first["directory"], "manifest.json").read_bytes()
    requested = comparison([0.4] * 8, count=8)
    incumbent = comparison([0.5] * 8, count=8)
    resolved = resolve_recommended(tmp_path / "library", KEY)
    incumbent["model_sha256"] = resolved["files"][resolved["model_path"]]["sha256"]
    incumbent["normalization_sha256"] = resolved["files"][resolved["normalization_path"]]["sha256"]
    result = publish(seed=2, comparison=requested, incumbent_comparison=incumbent)
    assert not result["recommended"]
    assert Path(first["directory"], "manifest.json").read_bytes() == old_manifest
    updated = resolve_recommended(tmp_path / "library", KEY)
    assert len(updated["comparison"]["protocol"]["episode_seeds"]) == 8
    assert len(updated["manifest"]["comparison"]["protocol"]["episode_seeds"]) == 6
    copied = copy_recommended(tmp_path / "library", KEY, tmp_path / "run")
    evidence_path = Path(copied["comparison_evidence"]["path"])
    assert evidence_path.is_relative_to(tmp_path / "run")
    assert json.loads(evidence_path.read_text())["comparison"] == updated["comparison"]
    assert Path(copied["selection_record"]).is_file()
    bad_binding = {**incumbent, "model_sha256": "0" * 64}
    with pytest.raises(CertifiedLibraryError, match="selected candidate"):
        publish(seed=3, comparison=requested, incumbent_comparison=bad_binding)


@pytest.mark.parametrize("name", ["../bad", "/absolute", "a/../bad", "a//bad", "a\\bad", "manifest.json"])
def test_unsafe_evidence_names_rejected(tmp_path, publish, name):
    source = tmp_path / "source"
    source.write_bytes(b"model")
    with pytest.raises(CertifiedLibraryError):
        publish(files={name: source}, model_path=name, normalization_path="norm")


def test_symlink_sources_and_symlink_library_are_rejected(tmp_path, publish):
    source = tmp_path / "source"
    source.write_bytes(b"model")
    link = tmp_path / "linked"
    link.symlink_to(source)
    with pytest.raises(CertifiedLibraryError, match="Symlink"):
        publish(files={"model": link, "norm": source}, model_path="model", normalization_path="norm")
    (tmp_path / "library").rename(tmp_path / "real-library")
    (tmp_path / "library").symlink_to(tmp_path / "real-library", target_is_directory=True)
    with pytest.raises(CertifiedLibraryError, match="Symlink"):
        publish()


@pytest.mark.parametrize("field", ["nan", "duplicate_keys", "short", "negative_margin", "missing_version"])
def test_invalid_comparison_refuses_publication(publish, field):
    value = comparison()
    if field == "nan":
        value["metrics"][0]["values"][0] = float("nan")
    elif field == "duplicate_keys":
        value["metrics"][0]["sample_keys"] = ["same"] * 6
    elif field == "short":
        value["metrics"][0]["values"] = [0.5]
    elif field == "negative_margin":
        value["metrics"][0]["margin"] = -0.01
    else:
        del value["protocol"]["version"]
    with pytest.raises(CertifiedLibraryError):
        publish(comparison=value)


def test_tampered_copied_inputs_are_not_overwritten(tmp_path, publish):
    publish()
    copied = copy_recommended(tmp_path / "library", KEY, tmp_path / "run")
    Path(copied["normalizer"]).write_bytes(b"changed")
    with pytest.raises(CertifiedLibraryError):
        copy_recommended(tmp_path / "library", KEY, tmp_path / "run")
    assert Path(copied["normalizer"]).read_bytes() == b"changed"


def test_parallel_publications_count_both_runs_without_losing_versions(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    norm = source / "norm"
    norm.write_bytes(b"norm")
    paths = []
    for seed in (1, 2):
        model = source / f"model-{seed}"
        model.write_bytes(str(seed).encode())
        paths.append(model)

    def publish_one(seed):
        return publish_candidate(
            tmp_path / "library",
            key=KEY,
            recipe_sha256=RECIPE,
            training_seed=seed,
            source_run_id=str(seed),
            files={"model": paths[seed - 1], "norm": norm},
            model_path="model",
            normalization_path="norm",
            certificate={"passed": True},
            comparison=comparison(),
            required_seeds=2,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(publish_one, (1, 2)))
    assert len({item["version"] for item in reports}) == 2
    assert resolve_recommended(tmp_path / "library", KEY)["distinct_seeds"] == 2
    assert all(Path(item["directory"], "manifest.json").is_file() for item in reports)


def test_candidate_comparison_hash_must_match_evidence(publish):
    value = comparison()
    value["model_sha256"] = hashlib.sha256(b"wrong-model").hexdigest()
    with pytest.raises(CertifiedLibraryError, match="model_sha256"):
        publish(comparison=value)


def test_explicit_version_copy_is_pinned_and_refuses_uncertified_candidates(tmp_path, publish):
    first = publish()
    publish(seed=2, comparison=comparison([0.9] * 6))
    copied = copy_version(tmp_path / "library", KEY, first["version"], tmp_path / "run")
    assert copied["version"] == first["version"]
    assert Path(copied["model"]).read_bytes() == b"model-1"
    failed = publish(seed=3, passed=False)
    with pytest.raises(NoRecommendedCandidate):
        copy_version(tmp_path / "library", KEY, failed["version"], tmp_path / "run")
    provisional = publish(seed=4, recipe_sha256="c" * 64, required_seeds=3)
    with pytest.raises(NoRecommendedCandidate):
        copy_version(tmp_path / "library", KEY, provisional["version"], tmp_path / "run")


def test_extreme_finite_metric_values_cannot_promote_via_overflow(tmp_path, publish):
    first = publish(comparison=comparison([-1e308] * 6))
    result = publish(seed=2, comparison=comparison([1e308] * 6))
    assert not result["recommended"]
    assert result["decision_reason"] == "nonfinite_paired_difference"
    assert resolve_recommended(tmp_path / "library", KEY)["version"] == first["version"]
