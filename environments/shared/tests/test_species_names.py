"""The manifest's species and models: exact IDs, display names, and the notebook's model selection.

Decision D-D33: a model is the pair (species, embodiment), named by the exact
manifest IDs.  The refusal of every retired ID, alias and display-name form
lives in ``test_retired_species_identifiers.py``.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from environments.shared.config import load_all_stages
from environments.shared.model_id import EMBODIMENTS, ModelId
from environments.shared.species_names import (
    model_display_name,
    model_ids,
    species_display_name,
    species_ids,
)
from environments.shared.species_registry import get_species_config
from environments.shared.stage_manifest import load_stage_manifest

from .notebook_cells import cell_sources, first_code_cell

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECIES_DISPLAY_NAMES = {
    "velociraptor_mongoliensis": "Velociraptor mongoliensis",
    "tyrannosaurus_rex": "Tyrannosaurus rex",
    "brachiosaurus_altithorax": "Brachiosaurus altithorax",
    "dibothrosuchus_elaphros": "Dibothrosuchus elaphros",
    "compsognathus_longipes": "Compsognathus longipes",
    "allosaurus_fragilis": "Allosaurus fragilis",
}
MODEL_DISPLAY_NAMES = {
    ModelId("velociraptor_mongoliensis", "anatomical"): "Velociraptor mongoliensis (Anatomical)",
    ModelId("tyrannosaurus_rex", "anatomical"): "Tyrannosaurus rex (Anatomical)",
    ModelId("brachiosaurus_altithorax", "anatomical"): "Brachiosaurus altithorax (Anatomical)",
    ModelId("dibothrosuchus_elaphros", "anatomical"): "Dibothrosuchus elaphros (Anatomical)",
    ModelId("compsognathus_longipes", "anatomical"): "Compsognathus longipes (Anatomical)",
    ModelId("compsognathus_longipes", "robot"): "Compsognathus longipes (Robot)",
    ModelId("allosaurus_fragilis", "anatomical"): "Allosaurus fragilis (Anatomical)",
}


def _notebook_cell(marker):
    return first_code_cell(REPO_ROOT / "notebooks/sb3_training.ipynb", marker)


def _selection_code(model_id):
    """The selection cell with ``SPECIES`` and ``EMBODIMENT`` set to *model_id*'s IDs through the AST."""
    tree = ast.parse(_notebook_cell("# ===== SPECIES SELECTION ====="))
    values = {"SPECIES": model_id.species, "EMBODIMENT": model_id.embodiment}
    if model_id.species == "allosaurus_fragilis":
        values["BEHAVIOR"] = "walk"
    assigned = set()
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in values
            and statement.targets[0].id not in assigned
        ):
            statement.value = ast.Constant(values[statement.targets[0].id])
            assigned.add(statement.targets[0].id)
    assert assigned == set(values), f"the selection cell assigns {sorted(assigned)}, not SPECIES and EMBODIMENT"
    return compile(ast.fix_missing_locations(tree), "sb3_training.ipynb", "exec")


@pytest.mark.parametrize("model_id", model_ids(), ids=lambda m: f"{m.species}-{m.embodiment}")
def test_the_notebook_selection_runs_the_model_and_its_environment(model_id, capsys):
    # The environment cell also resolves BEHAVIOR's chain through the manifest
    # (Phase A WS5); the setup cell that imports load_stage_manifest is not
    # executed here, so it is supplied like load_all_stages.
    namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
    exec(_selection_code(model_id), namespace)
    exec(compile(_notebook_cell("EnvClass = SPECIES_CFG.env_class"), "sb3_training.ipynb", "exec"), namespace)

    assert namespace["MODEL_ID"] == model_id
    assert namespace["STAGE_CONFIGS"] == load_all_stages(model_id)
    assert namespace["SPECIES_CFG"].success_keys == get_species_config(model_id).success_keys
    assert model_display_name(model_id) in capsys.readouterr().out
    env = namespace["EnvClass"]()
    try:
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        obs, reward, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
    finally:
        env.close()


def test_manifest_display_names_cover_every_model():
    assert model_ids() == tuple(MODEL_DISPLAY_NAMES)
    assert species_ids() == tuple(SPECIES_DISPLAY_NAMES)
    for species, label in SPECIES_DISPLAY_NAMES.items():
        assert species_display_name(species) == label
    for model_id, label in MODEL_DISPLAY_NAMES.items():
        assert model_display_name(model_id) == label


def test_backend_specific_models_do_not_advertise_unimplemented_jax():
    assert model_ids(backend="stable-baselines3") == tuple(MODEL_DISPLAY_NAMES)
    # Velociraptor left jax-mjx at its policy-interface revision 11 (plant_versions note 14), the T-Rex at its
    # revision 14 (note 18, the heading-free observation), and brachiosaurus and dibothrosuchus at revisions 11 and
    # 10 (note 23), when the backend was removed: it is now an unknown backend.
    for backend in ("jax-mjx", "unsupported"):
        with pytest.raises(ValueError, match="Unknown training backend"):
            model_ids(backend=backend)


def test_an_unknown_species_is_refused_for_training_and_for_labels():
    with pytest.raises(ValueError, match="Unknown species"):
        get_species_config(ModelId("stegosaurus_stenops", "anatomical"))
    with pytest.raises(ValueError, match="Unknown species"):
        species_display_name("My custom dinosaur")


def _selector_options(source, name):
    for line in source.splitlines():
        if line.startswith(f"{name} = ") and "# @param [" in line:
            return json.loads(line.split("# @param ", 1)[1])
    return None


def test_colab_model_selectors_match_the_manifest():
    """The notebook selects a model with two dropdowns: the species IDs and the embodiments."""
    for source in cell_sources(REPO_ROOT / "notebooks" / "sb3_training.ipynb"):
        species = _selector_options(source, "SPECIES")
        if species is not None:
            assert species == list(species_ids())
            assert _selector_options(source, "EMBODIMENT") == list(EMBODIMENTS)
            break
    else:
        pytest.fail("sb3_training.ipynb has no SPECIES selector")
