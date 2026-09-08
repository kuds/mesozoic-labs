"""Full-name selection must keep configs and saved species identities stable."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from environments.shared.config import SPECIES_NAMES, load_all_stages
from environments.shared.species_names import resolve_species_id, species_display_name, species_display_names
from environments.shared.species_registry import get_species_config

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_NAMES = {
    "velociraptor": "Velociraptor Mongoliensis",
    "trex": "Tyrannosaurus Rex",
    "brachiosaurus": "Brachiosaurus Altithorax",
    "dibothrosuchus": "Dibothrosuchus Elaphros",
    "compsognathus": "Compsognathus Longipes",
    "compsognathus_robot": "Compsognathus Longipes (Robot)",
}


def _notebook_cell(marker):
    notebook = json.loads((REPO_ROOT / "notebooks/sb3_training.ipynb").read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


def _selection_code(name):
    tree = ast.parse(_notebook_cell("# ===== SPECIES SELECTION ====="))
    selection = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "SPECIES"
    )
    selection.value = ast.Constant(name)
    return compile(ast.fix_missing_locations(tree), "sb3_training.ipynb", "exec")


@pytest.mark.parametrize("species_id,label", FULL_NAMES.items())
def test_full_name_runs_notebook_selection_and_environment(species_id, label, capsys):
    namespace = {"load_all_stages": load_all_stages}
    exec(_selection_code(label), namespace)
    exec(compile(_notebook_cell("EnvClass = SPECIES_CFG.env_class"), "sb3_training.ipynb", "exec"), namespace)

    assert namespace["SPECIES"] == species_id
    assert namespace["SPECIES_DISPLAY_NAME"] == label
    assert namespace["STAGE_CONFIGS"] == load_all_stages(species_id)
    assert namespace["SPECIES_CFG"].success_keys == get_species_config(species_id).success_keys
    assert f"Species: {label}" in capsys.readouterr().out
    env = namespace["EnvClass"]()
    try:
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        obs, reward, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["Tyrannosaurus Rex", "tyrannosaurus_rex", " TYRANNOSAURUS-REX ", "T-Rex", "trex"])
def test_full_names_and_legacy_aliases_keep_checkpoint_identity(name):
    assert resolve_species_id(name) == "trex"
    assert get_species_config(name).species == "trex"
    assert species_display_name(name) == "Tyrannosaurus Rex"


def test_manifest_labels_cover_every_training_species():
    assert species_display_names() == FULL_NAMES
    assert set(species_display_names()) == set(SPECIES_NAMES)
    assert species_display_names(include_prototypes=True)["compsognathus"] == "Compsognathus Longipes"


@pytest.mark.parametrize("name", ["Compsognathus Longipes", "compsognathus", "compso"])
def test_compsognathus_aliases_are_trainable(name):
    assert get_species_config(name).species == "compsognathus"


def test_backend_specific_names_do_not_advertise_unimplemented_jax():
    assert species_display_names(backend="stable-baselines3") == FULL_NAMES
    assert set(species_display_names(backend="jax-mjx")) == set(FULL_NAMES) - {"compsognathus", "compsognathus_robot"}
    with pytest.raises(ValueError, match="Unknown training backend"):
        species_display_names(backend="unsupported")


def test_unknown_training_name_is_rejected_but_custom_plot_label_is_preserved():
    with pytest.raises(ValueError, match="Unknown species"):
        get_species_config("My custom dinosaur")
    assert species_display_name("My custom dinosaur") == "My custom dinosaur"


def test_colab_species_selectors_match_the_manifest():
    for filename in ("sb3_training.ipynb", "jax_training.ipynb", "ray_tune_sweep.ipynb"):
        notebook = json.loads((REPO_ROOT / "notebooks" / filename).read_text())
        for cell in notebook["cells"]:
            for line in "".join(cell["source"]).splitlines():
                if line.startswith("SPECIES = ") and "# @param [" in line:
                    labels = json.loads(line.split("# @param ", 1)[1])
                    backend = "jax-mjx" if filename == "jax_training.ipynb" else "stable-baselines3"
                    assert labels == list(species_display_names(backend=backend).values())
                    break
            else:
                continue
            break
        else:
            pytest.fail(f"{filename} has no full-name species selector")
