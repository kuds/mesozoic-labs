"""Every model is the pair (species, embodiment), and every name is built from the pair (decision D-D33).

One rule names everything a model owns (docs/SPECIES_NAMING.md): its
embodiment package ``environments.<species>.<embodiment>`` and class
``<Species><Embodiment>Env``, its MJCF
``environments/<species>/<embodiment>/assets/<species>_<embodiment>.xml``, its
Gymnasium ID, its configs ``configs/<species>/<embodiment>/`` and its plant
table ``[plants.<species>.<embodiment>]``.  These tests hold the manifest
(``configs/species_manifest.toml``, the one list of models) and the tree to
that rule, so a model added in one place and forgotten in another fails here,
naming what is missing:

* each manifest model's entrypoint and model path are the derived names, its
  class's ``MODEL_PATH`` is that file, and the class carries its own
  ``embodiment``, ``FOOT_SUPPORT`` and ``REWARD_PROBE`` (the per-model body
  data that lives on the embodiment class, review H2);
* its ``stages.toml`` exists, every stage file is named by its stage with no
  number (``config`` minus ``.toml`` is the stage id or its ``[stage] name``;
  decision D) and the ``[stage]`` names are unique (review L1);
* the Gymnasium registry, the stage-manifest directories, the plant tables and
  (for the Stable-Baselines3 models) the behavior-recipe directories hold
  exactly the manifest's models;
* the directory names at depth 1 (species) and depth 2 (embodiment) under
  ``configs/`` and ``environments/`` are the manifest's; nothing deeper is
  constrained, so a later ``configs/<species>/robot/hardware.toml`` needs no
  edit here;
* only the two frozen MJX registrations declare ``jax-mjx`` (review L5);
* the website's species folders and the intro's model list follow the
  manifest (review L9), and so do the hand-written model lists of
  ``docs/SPECIES_NAMING.md`` and the website's API overview.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import gymnasium
import pytest

import environments  # noqa: F401  (registers the manifest's models with Gymnasium)
from environments.shared.model_id import EMBODIMENTS, ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.shared.species_names import (
    embodiments_of,
    model_display_name,
    model_entry,
    model_ids,
    species_display_name,
    species_ids,
)
from environments.shared.species_registry import env_class
from environments.shared.stage_manifest import load_stage_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPOSITORY_ROOT / "configs"
ENVIRONMENTS = REPOSITORY_ROOT / "environments"
WEBSITE_DOCS = REPOSITORY_ROOT / "website" / "docs"

MODEL_IDS = model_ids()


def _model_label(model_id: ModelId) -> str:
    return f"{model_id.species}-{model_id.embodiment}"


#: The two frozen D-D17 registrations, the only models that may declare jax-mjx (review L5).
FROZEN_MJX_MODELS = {
    ModelId("brachiosaurus_altithorax", "anatomical"),
    ModelId("dibothrosuchus_elaphros", "anatomical"),
}
#: Species-level directories of ``environments/<species>/`` (plan section 1.1 placement rules): the species' data,
#: scripts and tests.  Every other directory at that depth is an embodiment package.
SPECIES_LEVEL_DIRECTORIES = frozenset({"data", "scripts", "tests"})


def _directories(path: Path) -> set[str]:
    """Subdirectory names, without caches (``__pycache__``) or hidden directories."""
    return {child.name for child in path.iterdir() if child.is_dir() and not child.name.startswith(("_", "."))}


def _pairs(paths: Any) -> set[tuple[str, str]]:
    """The (species, embodiment) of each ``configs/<species>/<embodiment>/<name>``."""
    return {(path.parent.parent.name, path.parent.name) for path in paths}


def _manifest() -> dict[str, Any]:
    with (CONFIGS / "species_manifest.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_the_manifest_lists_models() -> None:
    """Every test below iterates the manifest; an empty list would pass them all vacuously."""
    assert MODEL_IDS, "configs/species_manifest.toml lists no model"
    assert set(MODEL_IDS) == {
        ModelId(species, embodiment) for species in species_ids() for embodiment in embodiments_of(species)
    }


# ── names built from the pair ─────────────────────────────────────────────


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_model_label)
def test_entrypoint_and_model_path_are_the_derived_names(model_id: ModelId) -> None:
    entry = model_entry(model_id)
    assert entry["env_entrypoint"] == f"{model_id.package}.env:{model_id.env_class_name}"
    assert entry["model_path"] == model_id.asset_path(Path("environments")).as_posix()
    cls: Any = env_class(model_id)
    assert cls.__module__ == f"{model_id.package}.env" and cls.__name__ == model_id.env_class_name
    assert Path(cls.MODEL_PATH) == model_id.asset_path(ENVIRONMENTS_ROOT) == REPOSITORY_ROOT / entry["model_path"]
    assert Path(cls.MODEL_PATH).is_file()


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_model_label)
def test_the_embodiment_class_carries_its_own_identity_and_body_data(model_id: ModelId) -> None:
    """``embodiment``, ``MODEL_PATH``, ``FOOT_SUPPORT`` and ``REWARD_PROBE`` sit on the embodiment class itself, not
    on the species base it shares with the species' other embodiments (review H2): a new robot adds its own."""
    cls: Any = env_class(model_id)
    for attribute in ("embodiment", "MODEL_PATH", "FOOT_SUPPORT", "REWARD_PROBE"):
        assert attribute in vars(cls), f"{cls.__qualname__} does not define {attribute}"
    assert cls.embodiment == model_id.embodiment
    # The ankle and its support span are declared together, where the whole-foot centre of pressure is measured.
    assert set(cls.FOOT_SUPPORT) in (
        {"support", "sole", "foot_site"},
        {"support", "sole", "foot_site", "ankle", "heel_line", "toe_line"},
    )
    # The species base names the species, never an embodiment.
    base = cls.__mro__[1]
    assert base.__module__ == f"environments.{model_id.species}.base_env"
    assert base.__name__ == f"{model_id.species_pascal}Env"


def test_the_gymnasium_registry_holds_exactly_the_manifest_models() -> None:
    registered = {key for key in gymnasium.registry if key.startswith("MesozoicLabs/")}
    assert registered == {model_id.gym_id for model_id in MODEL_IDS}
    for model_id in MODEL_IDS:
        assert gymnasium.registry[model_id.gym_id].entry_point == model_entry(model_id)["env_entrypoint"]


# ── stage files ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_id", MODEL_IDS, ids=_model_label)
def test_stage_files_are_named_by_their_stage_with_unique_names(model_id: ModelId) -> None:
    directory = model_id.path(CONFIGS)
    assert (directory / "stages.toml").is_file(), f"{model_id} has no {directory / 'stages.toml'}"
    manifest = load_stage_manifest(model_id)
    assert manifest.stages, f"{model_id}'s stages.toml lists no stage"
    names: list[str] = []
    for entry in manifest.stages:
        with (directory / entry.config_file).open("rb") as handle:
            name = tomllib.load(handle)["stage"]["name"]
        names.append(name)
        stem = entry.config_file.removesuffix(".toml")
        # Decision D: a stage file carries no number and is named by its stage.
        assert stem in {entry.id, name}, (
            f"{model_id}: {entry.config_file} names neither stage {entry.id!r} nor {name!r}"
        )
        assert not re.search(r"\d", stem), f"{model_id}: {entry.config_file} carries a number"
    assert len(names) == len(set(names)), f"{model_id}: [stage] names repeat: {names}"


# ── one set of models everywhere ──────────────────────────────────────────


def test_configs_manifest_plants_and_behaviors_hold_the_same_models() -> None:
    manifest_pairs = {(model_id.species, model_id.embodiment) for model_id in MODEL_IDS}
    with (CONFIGS / "plant_versions.toml").open("rb") as handle:
        plants = tomllib.load(handle)["plants"]
    plant_pairs = {(species, embodiment) for species, table in plants.items() for embodiment in table}
    stage_pairs = _pairs(CONFIGS.glob("*/*/stages.toml"))
    behavior_pairs = _pairs(path for path in CONFIGS.glob("*/*/behaviors") if path.is_dir())
    sb3_pairs = {(model_id.species, model_id.embodiment) for model_id in model_ids(backend="stable-baselines3")}
    assert stage_pairs == manifest_pairs, "configs/<species>/<embodiment>/stages.toml"
    assert plant_pairs == manifest_pairs, "configs/plant_versions.toml [plants.<species>.<embodiment>]"
    assert behavior_pairs == sb3_pairs, "configs/<species>/<embodiment>/behaviors/ (Stable-Baselines3 models)"


def test_directory_names_at_depth_one_and_two_are_the_manifests() -> None:
    """Depth 1 is the species, depth 2 the embodiment; nothing deeper is constrained (design review section 4)."""
    species = set(species_ids())
    assert _directories(CONFIGS) == species
    assert _directories(ENVIRONMENTS) - {"shared"} == species
    for name in species:
        embodiments = set(embodiments_of(name))
        loose = [child.name for child in (CONFIGS / name).iterdir() if not child.is_dir()]
        assert not loose, f"configs/{name}/ holds files outside an embodiment directory: {loose}"
        assert _directories(CONFIGS / name) == embodiments, f"configs/{name}/"
        packages = _directories(ENVIRONMENTS / name) - SPECIES_LEVEL_DIRECTORIES
        assert packages == embodiments, f"environments/{name}/"
        for embodiment in embodiments:
            assert embodiment in EMBODIMENTS
            assert (ENVIRONMENTS / name / embodiment / "env.py").is_file()
            assert (ENVIRONMENTS / name / embodiment / "__init__.py").is_file()


def test_only_the_frozen_mjx_registrations_declare_jax() -> None:
    declared = {model_id for model_id in MODEL_IDS if "jax-mjx" in model_entry(model_id)["training_backends"]}
    assert declared == FROZEN_MJX_MODELS
    assert set(model_ids(backend="jax-mjx")) == FROZEN_MJX_MODELS
    for model_id in FROZEN_MJX_MODELS:
        assert (ENVIRONMENTS / model_id.species / model_id.embodiment / "mjx_config.py").is_file()


# ── the website follows the manifest (review L9) ──────────────────────────


@pytest.mark.parametrize("species", species_ids())
def test_each_species_website_category_follows_the_manifest(species: str) -> None:
    category = WEBSITE_DOCS / "models" / species / "_category_.json"
    assert category.is_file(), f"website/docs/models/{species}/_category_.json is missing"
    recorded = json.loads(category.read_text(encoding="utf-8"))
    table = _manifest()["species"][species]
    assert recorded["label"] == species_display_name(species) == table["display_name"]
    assert recorded["position"] == table["display_order"]
    pages = {path.stem for path in (WEBSITE_DOCS / "models" / species).glob("*.mdx")}
    assert pages == set(embodiments_of(species)), f"website/docs/models/{species}/"


def test_the_intro_links_exactly_the_model_pages() -> None:
    intro = (WEBSITE_DOCS / "intro.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\(/docs/models/([a-z_]+/[a-z_]+)\)", intro))
    assert linked == {f"{model_id.species}/{model_id.embodiment}" for model_id in MODEL_IDS}


def test_the_naming_guide_tables_exactly_the_models() -> None:
    """``docs/SPECIES_NAMING.md``'s model table: one row per model, with its display name and Gymnasium ID."""
    naming = (REPOSITORY_ROOT / "docs" / "SPECIES_NAMING.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([a-z_]+)` \| `([a-z_]+)` \| ([^|]+?) \| `([^`]+)` \|", naming, flags=re.MULTILINE)
    assert rows == [
        (model_id.species, model_id.embodiment, model_display_name(model_id), model_id.gym_id) for model_id in MODEL_IDS
    ]


def test_the_api_overview_lists_exactly_the_gymnasium_ids() -> None:
    overview = (WEBSITE_DOCS / "api" / "overview.md").read_text(encoding="utf-8")
    listed = re.findall(r"^- `(MesozoicLabs/[^`]+)` - ", overview, flags=re.MULTILINE)
    assert listed == [model_id.gym_id for model_id in MODEL_IDS]
