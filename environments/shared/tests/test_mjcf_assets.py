"""Regression tests for the species MJCF assets."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import pytest


@dataclass(frozen=True)
class ModelExpectations:
    """Compiled dimensions and animal mass for a species model."""

    nq: int
    nv: int
    nu: int
    dynamic_mass: float


ENVIRONMENTS_DIR = Path(__file__).resolve().parents[2]

EXPECTED_MODELS = {
    "brachiosaurus/assets/brachiosaurus.xml": ModelExpectations(38, 37, 30, 175.3),
    "trex/assets/trex.xml": ModelExpectations(40, 37, 21, 85.44),
    "velociraptor/assets/raptor.xml": ModelExpectations(31, 30, 22, 13.5),
}


def _discover_mjcf_assets() -> tuple[Path, ...]:
    """Return every species MJCF asset in a deterministic order."""
    return tuple(sorted(ENVIRONMENTS_DIR.glob("*/assets/*.xml")))


MJCF_ASSETS = _discover_mjcf_assets()


def _asset_id(asset_path: Path) -> str:
    return asset_path.relative_to(ENVIRONMENTS_DIR).as_posix()


def test_mjcf_inventory_matches_expected_models():
    """Require new or renamed species assets to add explicit regression pins."""
    discovered = {_asset_id(asset_path) for asset_path in MJCF_ASSETS}
    assert discovered == set(EXPECTED_MODELS)


@pytest.mark.parametrize("asset_path", MJCF_ASSETS, ids=_asset_id)
def test_mjcf_is_standard_xml(asset_path: Path):
    """All MJCF files must also be well-formed, standards-compliant XML."""
    ElementTree.parse(asset_path)


@pytest.mark.parametrize("asset_path", MJCF_ASSETS, ids=_asset_id)
def test_mjcf_compiled_model_is_unchanged(asset_path: Path):
    """Pin the compiled interface and dynamic animal mass for every species."""
    model = mujoco.MjModel.from_xml_path(str(asset_path))
    expected = EXPECTED_MODELS[_asset_id(asset_path)]

    assert (model.nq, model.nv, model.nu) == (expected.nq, expected.nv, expected.nu)

    # The target is a mocap body, so exclude it (and the massless world body)
    # from the mass of the dynamic animal.
    dynamic_mass = float(model.body_mass[model.body_mocapid == -1].sum())
    assert dynamic_mass == pytest.approx(expected.dynamic_mass)
