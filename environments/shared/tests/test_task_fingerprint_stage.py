"""The one stage-level task fingerprint derivation (cleanup CU-8a).

``task_fingerprint.stage_task_fingerprint`` is how every site derives a
stage's task digest: ``train_base.train`` and ``train_curriculum``,
``ancestors.select_trunk``, the recovery freeze, the widen tool, the recovery
calibration and its restamp script, and the SB3 notebook's chain loop.  These
tests hold it to the committed digests (the golden snapshot and both recovery
profiles) in every form a caller uses, keep ``FINGERPRINT_BACKEND`` defined
once, and keep ``derive_stage_task_fingerprint`` referenced only by the helper
and the digest harness, which stays an independent oracle of the golden.
"""

from __future__ import annotations

import ast
import json
from functools import cache
from pathlib import Path
from typing import Any, Iterator

import pytest

from environments.shared import task_fingerprint
from environments.shared.config import SPECIES_NAMES, load_all_stages, load_stage_config
from environments.shared.plant_contract import PlantIdentity, current_plant_identity
from environments.shared.stage_manifest import load_stage_manifest

from .notebook_cells import code_cells, strip_magics

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPO_ROOT / "configs" / "digest_snapshot.generated.txt"
ENVIRONMENTS = REPO_ROOT / "environments"
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
#: The only non-test code that may name ``derive_stage_task_fingerprint``: its own module (the
#: definition and the helper's call) and the digest harness, whose literal-backend call is the
#: oracle the golden's ``task_sha256`` lines pin.
DERIVATION_MODULES = {
    "environments/shared/task_fingerprint.py",
    "environments/shared/harnesses/digest_snapshot.py",
}


def _golden_task_digests() -> dict[tuple[str, str], str]:
    """``(species, stage id) -> task_sha256`` from the committed digest snapshot."""
    digests: dict[tuple[str, str], str] = {}
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) == 5 and fields[0] == "stage" and fields[3] == "task_sha256":
            digests[(fields[1], fields[2])] = fields[4]
    return digests


GOLDEN_TASKS = _golden_task_digests()
STAGES = [(species, entry.id) for species in SPECIES_NAMES for entry in load_stage_manifest(species).stages]


@cache
def _identity(species: str) -> PlantIdentity:
    return current_plant_identity(species)


@cache
def _stage_configs(species: str) -> dict[Any, dict[str, Any]]:
    return load_all_stages(species)


def _non_test_trees() -> Iterator[tuple[str, ast.AST]]:
    """``(name, tree)`` for every non-test module under ``environments/`` and every notebook code cell."""
    for path in sorted(ENVIRONMENTS.rglob("*.py")):
        if "tests" in path.relative_to(ENVIRONMENTS).parts:
            continue
        yield path.relative_to(REPO_ROOT).as_posix(), ast.parse(path.read_text(encoding="utf-8"))
    for notebook in NOTEBOOKS:
        for index, source in code_cells(notebook):
            yield f"{notebook.relative_to(REPO_ROOT).as_posix()} cell {index}", ast.parse(strip_magics(source))


def test_the_backend_constant_is_the_committed_value_defined_once():
    """The value enters every ``task_sha256``; one definition means no second copy can drift."""
    assert task_fingerprint.FINGERPRINT_BACKEND == "stable-baselines3"
    definitions = []
    for name, tree in _non_test_trees():
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if (isinstance(leaf, ast.Name) and leaf.id == "FINGERPRINT_BACKEND") or (
                        isinstance(leaf, ast.Attribute) and leaf.attr == "FINGERPRINT_BACKEND"
                    ):
                        definitions.append(name)
    assert definitions == ["environments/shared/task_fingerprint.py"]


def test_only_the_helper_and_the_digest_oracle_name_the_low_level_derivation():
    """One derivation: every site calls ``stage_task_fingerprint``.  A reference of any kind counts
    (a call, a bare name passed as a callable as the digest harness does, an attribute, an import),
    so a site that goes back to ``derive_stage_task_fingerprint`` directly fails here."""
    name = "derive_stage_task_fingerprint"
    referencing = set()
    for where, tree in _non_test_trees():
        for node in ast.walk(tree):
            if (
                (isinstance(node, ast.Name) and node.id == name)
                or (isinstance(node, ast.Attribute) and node.attr == name)
                or (isinstance(node, ast.ImportFrom) and any(alias.name == name for alias in node.names))
                or (isinstance(node, ast.Constant) and node.value == name)
            ):
                referencing.add(where.split(" cell ")[0])
    assert referencing == DERIVATION_MODULES


def test_the_golden_names_every_manifest_stage():
    """The equality test below covers every stage of every species' manifest (21 stages of six species
    today), each exactly once and each with a committed digest."""
    assert len(STAGES) == len(set(STAGES)) == len(GOLDEN_TASKS)
    assert set(STAGES) == set(GOLDEN_TASKS)
    assert {species for species, _ in STAGES} == set(SPECIES_NAMES)


@pytest.mark.parametrize(("species", "stage_id"), STAGES, ids=[f"{s}-{i}" for s, i in STAGES])
def test_every_form_of_the_helper_is_the_committed_digest(species, stage_id):
    """An id or a reference, an in-memory config or the committed one, the identity object or its
    dict: all give the digest the golden records, which is the low-level derivation's under the
    literal backend, the committed ``[env]`` block and the current plant.  (A helper that hashed the
    id would move every numbered stage's digest.)"""
    entry = load_stage_manifest(species).by_id(stage_id)
    reference = entry.reference
    identity = _identity(species)
    helper = task_fingerprint.stage_task_fingerprint
    by_id = helper(species, stage_id)
    by_reference = helper(species, reference)
    in_memory = helper(species, reference, stage_config=_stage_configs(species)[reference], plant_identity=identity)
    as_mapping = helper(species, stage_id, plant_identity=identity.to_dict())
    derived = task_fingerprint.derive_stage_task_fingerprint(
        species=species,
        stage=reference,
        backend="stable-baselines3",
        env_kwargs=load_stage_config(species, reference).get("env_kwargs", {}),
        plant_identity=identity.to_dict(),
    )
    assert by_id == by_reference == in_memory == as_mapping == derived
    assert derived["task_sha256"] == GOLDEN_TASKS[(species, stage_id)]
    assert derived["stage"] == reference and derived["backend"] == task_fingerprint.FINGERPRINT_BACKEND


@pytest.mark.parametrize("species", ["compsognathus", "compsognathus_robot"])
def test_the_measured_recovery_env_gives_each_committed_profile_digest(species):
    """The recovery calibration and its restamp derive from the profile's MEASURED ``[env]`` block
    (``env_kwargs=``); on the committed profiles that is the digest each one records."""
    profile = json.loads((REPO_ROOT / "configs" / species / "recovery_calibration.json").read_text(encoding="utf-8"))
    measured = profile["recovery_env_kwargs"]
    helper = task_fingerprint.stage_task_fingerprint
    assert helper(species, "recovery", env_kwargs=measured)["task_sha256"] == profile["task_sha256"]
    assert (
        helper(species, "recovery", env_kwargs=measured, plant_identity=profile["plant_identity"])["task_sha256"]
        == profile["task_sha256"]
    )


def test_the_helper_hands_the_derivation_exactly_its_five_inputs(monkeypatch):
    """The reference, the one backend, the chosen ``[env]`` block and the identity as a dict — and no
    ``command_manifest``: the low-level derivation is looked up when the helper runs, so a test that
    replaces it sees the call."""
    derived: list[dict[str, Any]] = []

    def derive(**kwargs):
        derived.append(kwargs)
        return {"task_sha256": "sha256:derived"}

    monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", derive)
    identity = {"physics_sha256": "sha256:plant"}
    assert task_fingerprint.stage_task_fingerprint(
        "trex", "stance", stage_config={"env_kwargs": {"push_prob": 0.25}}, plant_identity=identity
    ) == {"task_sha256": "sha256:derived"}
    task_fingerprint.stage_task_fingerprint("trex", "recovery", env_kwargs={"a": 1}, plant_identity=identity)
    task_fingerprint.stage_task_fingerprint("trex", 1, stage_config={"name": "no env block"}, plant_identity=identity)
    assert derived == [
        {
            "species": "trex",
            "stage": 1,
            "backend": "stable-baselines3",
            "env_kwargs": {"push_prob": 0.25},
            "plant_identity": identity,
        },
        {
            "species": "trex",
            "stage": "recovery",
            "backend": "stable-baselines3",
            "env_kwargs": {"a": 1},
            "plant_identity": identity,
        },
        {
            "species": "trex",
            "stage": 1,
            "backend": "stable-baselines3",
            "env_kwargs": {},
            "plant_identity": identity,
        },
    ]
    assert derived[0]["plant_identity"] is not identity, "the identity mapping is copied, never aliased"


def test_a_stage_config_and_a_measured_env_together_are_refused(monkeypatch):
    """The task has one ``[env]`` block: naming two sources is a caller error, refused before any lookup."""

    def never(*args, **kwargs):
        raise AssertionError("refused before anything is derived")

    monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", never)
    with pytest.raises(TypeError, match="stage_config or env_kwargs, not both"):
        task_fingerprint.stage_task_fingerprint(
            "trex", "recovery", stage_config={"env_kwargs": {}}, env_kwargs={}, plant_identity={}
        )


@pytest.mark.parametrize("stage", ["no-such-stage", "1", True, 99])
def test_a_stage_the_manifest_does_not_name_is_refused_not_hashed(stage):
    """Canonicalising through the manifest refuses what it does not name (``"1"`` is no id and a
    bool is no legacy number), instead of hashing it into a digest no run records."""
    from environments.shared.stage_manifest import StageManifestError

    with pytest.raises(StageManifestError):
        task_fingerprint.stage_task_fingerprint("trex", stage, stage_config={}, plant_identity={})
