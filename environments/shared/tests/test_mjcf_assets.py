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
    ModelId("allosaurus_fragilis", "anatomical"): ModelExpectations(19, 18, 10, 1482.2),
    ModelId("brachiosaurus_altithorax", "anatomical"): ModelExpectations(38, 37, 30, 175.3),
    ModelId("compsognathus_longipes", "anatomical"): ModelExpectations(24, 23, 14, 1.0),
    ModelId("compsognathus_longipes", "robot"): ModelExpectations(19, 18, 12, 1.5856),
    # Physics r4 / r12 removed the dibothrosuchus's 0.02 kg snap box and the T-Rex's 0.2 kg bite box (r3 8.65 kg,
    # r11 85.72 kg).
    ModelId("dibothrosuchus_elaphros", "anatomical"): ModelExpectations(35, 34, 27, 8.63),
    ModelId("tyrannosaurus_rex", "anatomical"): ModelExpectations(28, 27, 15, 85.52),
    # The life-size plant of physics r5 (IGM 100/986, 15 kg): 23 hinges, 17 servos (r4: 31, 30, 22, 13.5).
    ModelId("velociraptor_mongoliensis", "anatomical"): ModelExpectations(30, 29, 17, 15.0),
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

    # No scene carries a mocap body (the prey, food and course targets left in the target-free revision), so
    # every body but the massless world body is the dynamic animal.
    assert model.nmocap == 0
    dynamic_mass = float(model.body_mass.sum())
    assert dynamic_mass == pytest.approx(expected.dynamic_mass)


#: The hunt's objects and its hunt-only body parts, removed in the target-free revision (plant_versions note 24,
#: decision D-D59), by MuJoCo object type: every scene's mocap prey, food or course target, the velociraptor's
#: prey-only claw collision capsules with their tip sites and position sensors, the T-Rex's 0.2 kg head_bite box and
#: the dibothrosuchus's 0.02 kg snout_snap box.
REMOVED_OBJECTS = {
    mujoco.mjtObj.mjOBJ_BODY: ("prey", "food", "target"),
    mujoco.mjtObj.mjOBJ_GEOM: (
        "prey_geom",
        "food_geom",
        "r_claw_geom",
        "l_claw_geom",
        "head_bite",
        "snout_snap",
    ),
    mujoco.mjtObj.mjOBJ_SITE: ("target_site", "r_claw_tip", "l_claw_tip"),
    mujoco.mjtObj.mjOBJ_SENSOR: ("r_claw_pos", "l_claw_pos"),
}
#: Kept by the owner's decision: the T-Rex's head clearance site, the dibothrosuchus's snout-prop termination site
#: and the brachiosaurus's head tip with its position sensor.
KEPT_SITES = {
    ModelId("tyrannosaurus_rex", "anatomical"): ("head_tip",),
    ModelId("dibothrosuchus_elaphros", "anatomical"): ("snout_tip",),
    ModelId("brachiosaurus_altithorax", "anatomical"): ("head_tip",),
}


@pytest.mark.parametrize("asset_path", MJCF_ASSETS, ids=_asset_id)
def test_no_scene_keeps_the_hunts_objects_or_its_hunt_only_parts(asset_path: Path):
    model = mujoco.MjModel.from_xml_path(str(asset_path))
    model_id = MODEL_BY_ASSET[asset_path]
    assert model.nmocap == 0
    for kind, names in REMOVED_OBJECTS.items():
        for name in names:
            assert mujoco.mj_name2id(model, kind, name) == -1, (model_id, name)
    # The velociraptor's claw pair lost its contact exclude with its collision capsules.
    bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("r_toe_claw", "l_toe_claw")}
    excluded = {frozenset((int(signature) >> 16, int(signature) & 0xFFFF)) for signature in model.exclude_signature}
    assert frozenset(bodies) not in excluded
    for name in KEPT_SITES.get(model_id, ()):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) != -1, (model_id, name)
    if model_id == ModelId("brachiosaurus_altithorax", "anatomical"):
        sensor = model.sensor("head_tip_pos")
        assert sensor.type == mujoco.mjtSensor.mjSENS_FRAMEPOS and sensor.objid == model.site("head_tip").id
