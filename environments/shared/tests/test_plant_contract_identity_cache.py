"""Tests for the per-process plant-identity cache in plant_contract.manifest.

``current_plant_identity`` reuses one build per species while the build's key
is unchanged.  Every test starts from an empty cache and hands the rest of the
session its cache back afterwards.  Most tests replace the builder with a
stand-in (``builds``), so a key test costs milliseconds; three tests run real
builds of the cheapest plant.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from types import SimpleNamespace

import mujoco
import pytest

from environments.shared.plant_contract import (
    PlantContractError,
    build_plant_manifest,
    clear_plant_identity_cache,
    constants,
    current_plant_identity,
    manifest,
    physics_layer,
    source_layer,
)

SPECIES = "compsognathus"  # one MJCF file and no meshes: the cheapest real build
COMMITTED = json.loads(constants.GENERATED_MANIFEST_PATH.read_text(encoding="utf-8"))["plants"]


@pytest.fixture(autouse=True)
def _empty_identity_cache():
    saved = dict(manifest._IDENTITY_CACHE)
    clear_plant_identity_cache()
    yield
    clear_plant_identity_cache()
    manifest._IDENTITY_CACHE.update(saved)


@pytest.fixture
def builds(monkeypatch):
    """Stand in for the uncached builder: return the committed entry, count the calls.

    The identity each call returns is a marker naming the build that served it.
    """
    calls = []

    def build(species, entry, version):
        calls.append(species)
        return copy.deepcopy(COMMITTED[species]), SimpleNamespace(build=len(calls))

    monkeypatch.setattr(manifest, "_manifest_entry_for_identity", build)
    monkeypatch.setattr(manifest, "_policy_interface_unchanged", lambda *args: True)
    return calls


@pytest.fixture
def physics_payloads(monkeypatch):
    """Count physics fingerprints of real builds (two per build), the work a hit skips."""
    calls = []
    real = physics_layer._physics_payload

    def counting(model, version):
        calls.append(version.species)
        return real(model, version)

    monkeypatch.setattr(physics_layer, "_physics_payload", counting)
    return calls


@pytest.fixture
def checkout_copy(tmp_path):
    """A second repository root holding a byte-identical copy of the plant's source closure."""
    root = tmp_path.resolve() / "checkout"
    model_path = constants.REPOSITORY_ROOT / COMMITTED[SPECIES]["model_path"]
    for dependency in source_layer._source_dependencies(model_path):
        copied = root / dependency.relative_to(constants.REPOSITORY_ROOT)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dependency, copied)
    return root


@pytest.fixture
def stale_manifest(tmp_path):
    stale = json.loads(constants.GENERATED_MANIFEST_PATH.read_text(encoding="utf-8"))
    stale["plants"][SPECIES]["physics"]["sha256"] = "sha256:" + "0" * 64
    path = tmp_path / "plant_manifest.generated.json"
    path.write_text(json.dumps(stale), encoding="utf-8")
    return path


def test_repeat_calls_reuse_one_real_build(physics_payloads):
    identity = current_plant_identity(SPECIES)

    assert current_plant_identity(SPECIES) is identity
    assert current_plant_identity(SPECIES, verify_generated=False) is identity
    assert len(physics_payloads) == 2

    clear_plant_identity_cache()
    assert current_plant_identity(SPECIES) == identity
    assert len(physics_payloads) == 4


def test_a_hit_still_checks_the_generated_manifest(builds, monkeypatch, stale_manifest):
    identity = current_plant_identity(SPECIES)
    monkeypatch.setattr(constants, "GENERATED_MANIFEST_PATH", stale_manifest)

    with pytest.raises(PlantContractError, match=f"generated plant manifest is stale for {SPECIES}"):
        current_plant_identity(SPECIES)
    assert current_plant_identity(SPECIES, verify_generated=False) is identity
    assert len(builds) == 1


def test_a_build_that_differs_from_the_committed_manifest_is_never_stored(builds, monkeypatch, stale_manifest):
    committed_path = constants.GENERATED_MANIFEST_PATH
    monkeypatch.setattr(constants, "GENERATED_MANIFEST_PATH", stale_manifest)
    assert current_plant_identity(SPECIES, verify_generated=False).build == 1
    assert current_plant_identity(SPECIES, verify_generated=False).build == 2

    monkeypatch.setattr(constants, "GENERATED_MANIFEST_PATH", committed_path)
    assert current_plant_identity(SPECIES).build == 3
    assert current_plant_identity(SPECIES).build == 3


def test_a_failed_build_is_never_stored(builds, monkeypatch):
    stand_in = manifest._manifest_entry_for_identity

    def fails_once(species, entry, version):
        if not builds:
            builds.append(species)
            raise PlantContractError("first build fails")
        return stand_in(species, entry, version)

    monkeypatch.setattr(manifest, "_manifest_entry_for_identity", fails_once)
    with pytest.raises(PlantContractError, match="first build fails"):
        current_plant_identity(SPECIES)
    assert current_plant_identity(SPECIES).build == 2
    assert current_plant_identity(SPECIES).build == 2


def test_a_changed_species_entry_rebuilds(builds, monkeypatch):
    assert current_plant_identity(SPECIES).build == 1
    real_entries = manifest._species_entries

    def edited_entries():
        entries = real_entries()
        entries[SPECIES]["display_name"] = "Edited"
        return entries

    monkeypatch.setattr(manifest, "_species_entries", edited_entries)
    assert current_plant_identity(SPECIES).build == 2
    assert current_plant_identity(SPECIES).build == 2


def test_a_changed_plant_version_rebuilds(builds, monkeypatch):
    assert current_plant_identity(SPECIES).build == 1
    real_versions = manifest.load_plant_versions

    def bumped_visual_revision():
        canonical, versions = real_versions()
        version = versions[SPECIES]
        versions[SPECIES] = dataclasses.replace(version, visual_revision=version.visual_revision + 1)
        return canonical, versions

    monkeypatch.setattr(manifest, "load_plant_versions", bumped_visual_revision)
    assert current_plant_identity(SPECIES).build == 2


def test_a_different_repository_root_rebuilds(builds, monkeypatch, checkout_copy):
    assert current_plant_identity(SPECIES).build == 1
    monkeypatch.setattr(constants, "REPOSITORY_ROOT", checkout_copy)

    assert current_plant_identity(SPECIES).build == 2
    assert current_plant_identity(SPECIES).build == 2


def test_an_edited_source_closure_rebuilds(builds, monkeypatch, checkout_copy):
    monkeypatch.setattr(constants, "REPOSITORY_ROOT", checkout_copy)
    assert current_plant_identity(SPECIES).build == 1
    assert current_plant_identity(SPECIES).build == 1

    model_copy = checkout_copy / COMMITTED[SPECIES]["model_path"]
    model_copy.write_text(model_copy.read_text(encoding="utf-8") + "<!-- edited -->\n", encoding="utf-8")
    assert current_plant_identity(SPECIES).build == 2


def test_a_different_environment_class_rebuilds(builds, monkeypatch):
    assert current_plant_identity(SPECIES).build == 1
    env_class = manifest._load_environment(COMMITTED[SPECIES]["env_entrypoint"])
    reloaded = type(env_class.__name__, (env_class,), {"__module__": env_class.__module__})
    monkeypatch.setattr(manifest, "_load_environment", lambda entrypoint: reloaded)

    assert current_plant_identity(SPECIES).build == 2


def test_a_different_mujoco_version_rebuilds(builds, monkeypatch):
    assert current_plant_identity(SPECIES).build == 1
    monkeypatch.setattr(mujoco, "__version__", "0.0.0")

    assert current_plant_identity(SPECIES).build == 2


def test_the_bundled_manifest_fallback_is_its_own_build(builds, monkeypatch, tmp_path):
    # Bundled entries carry only id, model_path and env_entrypoint, so the
    # fallback never reuses a build made from configs/, and
    # test_runtime_identity_falls_back_to_bundled_manifest keeps building it.
    assert current_plant_identity("velociraptor").build == 1
    for name in ("SPECIES_MANIFEST_PATH", "PLANT_VERSIONS_PATH", "GENERATED_MANIFEST_PATH"):
        monkeypatch.setattr(constants, name, tmp_path / f"missing-{name.lower()}")

    assert current_plant_identity("velociraptor").build == 2


def test_an_unreadable_source_closure_is_left_to_the_uncached_build(builds, monkeypatch):
    def unreadable(model_path):
        raise PlantContractError("unreadable source closure")

    monkeypatch.setattr(source_layer, "_source_payload", unreadable)
    assert current_plant_identity(SPECIES).build == 1
    assert current_plant_identity(SPECIES).build == 2


def test_the_plant_manifest_build_never_reads_the_cache(monkeypatch, physics_payloads):
    current_plant_identity(SPECIES)
    real_versions, real_entries = manifest.load_plant_versions, manifest._species_entries

    def one_species_versions():
        canonical, versions = real_versions()
        return canonical, {SPECIES: versions[SPECIES]}

    monkeypatch.setattr(manifest, "load_plant_versions", one_species_versions)
    monkeypatch.setattr(manifest, "_species_entries", lambda: {SPECIES: real_entries()[SPECIES]})

    assert build_plant_manifest()["plants"] == {SPECIES: COMMITTED[SPECIES]}
    assert len(physics_payloads) == 4


def test_patched_policy_code_after_a_hit_is_rebuilt(monkeypatch):
    import numpy as np

    env_class = manifest._load_environment(COMMITTED[SPECIES]["env_entrypoint"])
    committed = current_plant_identity(SPECIES)
    original = env_class._get_obs
    monkeypatch.setattr(
        env_class, "_get_obs", lambda self: np.concatenate([original(self)[:1] * 2.0, original(self)[1:]])
    )

    with pytest.raises(PlantContractError, match=f"generated plant manifest is stale for {SPECIES}"):
        current_plant_identity(SPECIES)
    assert (
        current_plant_identity(SPECIES, verify_generated=False).policy_interface_sha256
        != committed.policy_interface_sha256
    )


def test_a_build_of_other_bytes_than_the_key_read_is_never_stored(builds, monkeypatch, checkout_copy):
    # The stand-in returns the committed entry, as a build does that read the committed
    # bytes after the key had read an edit (a file restored mid-build).
    monkeypatch.setattr(constants, "REPOSITORY_ROOT", checkout_copy)
    model_copy = checkout_copy / COMMITTED[SPECIES]["model_path"]
    model_copy.write_text(model_copy.read_text(encoding="utf-8") + "<!-- edited -->\n", encoding="utf-8")

    assert current_plant_identity(SPECIES).build == 1
    assert current_plant_identity(SPECIES).build == 2
