"""Regression tests for the species MJCF assets."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import pytest

from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids


@dataclass(frozen=True)
class ModelExpectations:
    """Compiled dimensions and animal mass for a species model."""

    nq: int
    nv: int
    nu: int
    dynamic_mass: float


ENVIRONMENTS_DIR = Path(__file__).resolve().parents[2]

#: One pin per model, keyed by the model (review M4); its MJCF is ``ModelId.asset_path`` (decision D-D33).
EXPECTED_MODELS = {
    ModelId("brachiosaurus_altithorax", "anatomical"): ModelExpectations(38, 37, 30, 175.3),
    ModelId("compsognathus_longipes", "anatomical"): ModelExpectations(24, 23, 14, 1.0),
    ModelId("compsognathus_longipes", "robot"): ModelExpectations(19, 18, 12, 1.5856),
    ModelId("dibothrosuchus_elaphros", "anatomical"): ModelExpectations(35, 34, 27, 8.65),
    ModelId("tyrannosaurus_rex", "anatomical"): ModelExpectations(28, 27, 15, 85.72),
    ModelId("tyrannosaurus_rex", "robot"): ModelExpectations(17, 16, 10, 0.7541),
    ModelId("velociraptor_mongoliensis", "anatomical"): ModelExpectations(31, 30, 22, 13.5),
}


def _discover_mjcf_assets() -> tuple[Path, ...]:
    """Return every model's MJCF asset (``<species>/<embodiment>/assets/``) in a deterministic order."""
    return tuple(sorted(ENVIRONMENTS_DIR.glob("*/*/assets/*.xml")))


MJCF_ASSETS = _discover_mjcf_assets()
MODEL_BY_ASSET = {model_id.asset_path(ENVIRONMENTS_DIR): model_id for model_id in EXPECTED_MODELS}


def _asset_id(asset_path: Path) -> str:
    return asset_path.relative_to(ENVIRONMENTS_DIR).as_posix()


def test_mjcf_inventory_matches_expected_models():
    """Require new or renamed model assets to add explicit regression pins."""
    assert set(EXPECTED_MODELS) == set(model_ids()), "EXPECTED_MODELS must pin every manifest model"
    # Coverage equality with the files: no MJCF sits under an embodiment's assets/ that no model names.
    assert MJCF_ASSETS and set(MJCF_ASSETS) == set(MODEL_BY_ASSET)


@pytest.mark.parametrize("asset_path", MJCF_ASSETS, ids=_asset_id)
def test_mjcf_is_standard_xml(asset_path: Path):
    """All MJCF files must also be well-formed, standards-compliant XML."""
    ElementTree.parse(asset_path)


@pytest.mark.parametrize("asset_path", MJCF_ASSETS, ids=_asset_id)
def test_mjcf_compiled_model_is_unchanged(asset_path: Path):
    """Pin the compiled interface and dynamic animal mass for every species."""
    model = mujoco.MjModel.from_xml_path(str(asset_path))
    expected = EXPECTED_MODELS[MODEL_BY_ASSET[asset_path]]

    assert (model.nq, model.nv, model.nu) == (expected.nq, expected.nv, expected.nu)

    # The target is a mocap body, so exclude it (and the massless world body)
    # from the mass of the dynamic animal.
    dynamic_mass = float(model.body_mass[model.body_mocapid == -1].sum())
    assert dynamic_mass == pytest.approx(expected.dynamic_mass)
