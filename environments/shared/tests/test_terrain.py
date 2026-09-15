"""Terrain geometry is checked against MuJoCo's own ray/collision queries."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environments.shared.terrain import (
    TerrainConfig,
    TerrainRealization,
    apply_terrain,
    build_terrain_model,
    generate_terrain,
)

_TREX_XML = Path(__file__).resolve().parents[2] / "trex" / "assets" / "trex.xml"


@pytest.fixture
def small_config():
    return TerrainConfig(extent=8, nrow=81, ncol=81)


@pytest.fixture
def sphere_xml(tmp_path):
    source = tmp_path / "sphere.xml"
    source.write_text(
        '<mujoco><worldbody><geom name="floor" type="plane" size="10 10 .1"/>'
        '<body pos="0 0 .2"><freejoint/><geom name="ball" type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    return source


def test_generation_is_seeded_local_and_episode_variation_is_bounded(small_config):
    np.random.seed(918)
    before = np.random.get_state()
    original = generate_terrain(small_config, run_seed=42, episode_index=7)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    repeated = generate_terrain(small_config, run_seed=42, episode_index=7)
    np.testing.assert_array_equal(original.heights, repeated.heights)
    assert original.manifest() == repeated.manifest()
    changed_episode = generate_terrain(small_config, run_seed=42, episode_index=8)
    changed_run = generate_terrain(small_config, run_seed=43, episode_index=7)
    assert original.manifest()["samples_sha256"] != changed_episode.manifest()["samples_sha256"]
    assert original.manifest()["samples_sha256"] != changed_run.manifest()["samples_sha256"]
    # Variation changes an existing course modestly instead of resampling it.
    assert np.corrcoef(original.heights.ravel(), changed_episode.heights.ravel())[0, 1] > 0.9


def test_zero_episode_variation_preserves_run_geometry(small_config):
    config = replace(small_config, episode_variation=0)
    first = generate_terrain(config, run_seed=9, episode_index=0)
    later = generate_terrain(config, run_seed=9, episode_index=39)
    np.testing.assert_array_equal(first.heights, later.heights)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123, 991])
def test_realized_grade_and_height_are_bounded_with_both_elevation_signs(seed):
    config = TerrainConfig()
    terrain = generate_terrain(config, run_seed=seed)
    manifest = terrain.manifest()
    assert 0 < manifest["maximum_grade_degrees"] <= config.max_slope_degrees
    assert -config.max_height <= terrain.heights.min() < -0.1
    assert 0.1 < terrain.heights.max() <= config.max_height


def test_apron_is_exactly_flat_between_grid_vertices(small_config):
    terrain = generate_terrain(small_config, run_seed=12)
    rng = np.random.default_rng(17)
    angles = rng.uniform(-np.pi, np.pi, 1000)
    radii = np.r_[rng.uniform(0, small_config.apron_radius, 990), np.full(10, small_config.apron_radius)]
    np.testing.assert_array_equal(terrain.height_at(radii * np.cos(angles), radii * np.sin(angles)), 0)


def test_flat_terrain_and_finite_boundaries(small_config):
    terrain = generate_terrain(replace(small_config, mode="flat"), run_seed=1)
    np.testing.assert_array_equal(terrain.heights, 0)
    assert terrain.height_at(-8, 8) == 0
    assert terrain.height_at(8, -8) == 0
    assert np.isnan(terrain.height_at(8.01, 0))
    assert np.isnan(terrain.height_at(np.nan, 0))
    assert terrain.height_at(0, np.inf, outside=-999) == -999
    np.testing.assert_array_equal(terrain.contains([0, 8, 8.01], [0, -8, 0]), [True, True, False])
    assert terrain.bounds == (-8, 8, -8, 8)


def test_height_sampler_matches_mujoco_diagonal_triangles(sphere_xml):
    config = TerrainConfig(
        extent=2,
        nrow=9,
        ncol=9,
        apron_radius=0.3,
        blend_width=0.4,
        wavelength_min=2,
        wavelength_max=4,
    )
    rng = np.random.default_rng(64)
    terrain = TerrainRealization(config, 64, 0, rng.uniform(-0.4, 0.4, (9, 9)))
    model = build_terrain_model(sphere_xml, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    points = np.vstack([rng.uniform(-1.9999, 1.9999, (200, 2)), [[-2, -2], [2, 2], [-2, 2], [2, -2]]])
    actual = []
    for x, y in points:
        distance = mujoco.mj_rayHfield(model, data, floor, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]))
        assert distance >= 0
        actual.append(3 - distance)
    np.testing.assert_allclose(terrain.height_at(points[:, 0], points[:, 1]), actual, rtol=0, atol=1e-10)
    # A single raised cell corner specifically distinguishes triangles from
    # bilinear interpolation: the cell center has height .5, not .25.
    corners = np.zeros((9, 9))
    corners[1, 1] = 1
    corner_terrain = TerrainRealization(config, 0, 0, corners)
    assert corner_terrain.height_at(-1.75, -1.75) == pytest.approx(0.5)


def test_model_replaces_floor_preserves_dinosaur_and_does_not_mutate_source(small_config):
    source_before = _TREX_XML.read_bytes()
    canonical = mujoco.MjModel.from_xml_path(str(_TREX_XML))
    terrain = generate_terrain(small_config, run_seed=41)
    model = build_terrain_model(_TREX_XML, terrain)
    assert model.nhfield == 1
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert (model.nq, model.nv, model.nu, model.nsensor, model.nbody, model.ngeom) == (
        canonical.nq,
        canonical.nv,
        canonical.nu,
        canonical.nsensor,
        canonical.nbody,
        canonical.ngeom,
    )
    for name in ("body_mass", "body_inertia", "jnt_range", "actuator_ctrlrange", "key_qpos", "key_ctrl"):
        np.testing.assert_array_equal(getattr(model, name), getattr(canonical, name))
    assert _TREX_XML.read_bytes() == source_before
    assert canonical.nhfield == 0


def test_apply_changes_rays_and_contacts_without_rebuilding(sphere_xml, small_config):
    flat = generate_terrain(replace(small_config, mode="flat"), run_seed=1)
    raised = TerrainRealization(flat.config, 1, 1, np.full(flat.heights.shape, 0.15))
    model = build_terrain_model(sphere_xml, flat)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0  # ball bottom at .1, flat ground at zero
    size_before, bounds_before = model.hfield_size.copy(), model.geom_aabb.copy()
    apply_terrain(model, raised, data)
    assert data.ncon > 0  # same ball now penetrates raised ground
    ray = mujoco.mj_rayHfield(model, data, 0, np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]))
    assert 3 - ray == pytest.approx(raised.height_at(0, 0), abs=1e-10)
    np.testing.assert_array_equal(model.hfield_size, size_before)
    np.testing.assert_array_equal(model.geom_aabb, bounds_before)
    apply_terrain(model, flat, data)
    assert data.ncon == 0


def test_build_resolves_relative_includes(tmp_path, small_config):
    (tmp_path / "body.xml").write_text('<mujoco><body><freejoint/><geom size=".1" mass="1"/></body></mujoco>')
    source = tmp_path / "scene.xml"
    source.write_text(
        '<mujoco><worldbody><geom name="floor" type="plane" size="10 10 .1"/>'
        '<include file="body.xml"/></worldbody></mujoco>'
    )
    assert build_terrain_model(source, generate_terrain(small_config, run_seed=2)).nq == 7


def test_extra_plane_is_rejected(tmp_path, small_config):
    source = tmp_path / "extra.xml"
    source.write_text(
        '<mujoco><worldbody><geom name="floor" type="plane" size="10 10 .1"/>'
        '<geom name="hidden" type="plane" size="10 10 .1" pos="0 0 -1"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="additional plane"):
        build_terrain_model(source, generate_terrain(small_config, run_seed=2))


def test_apply_rejects_changed_shape_and_canonical_model(sphere_xml, small_config):
    terrain = generate_terrain(small_config, run_seed=4)
    model = build_terrain_model(sphere_xml, terrain)
    changed = generate_terrain(replace(small_config, nrow=83), run_seed=4)
    with pytest.raises(ValueError, match="dimensions differ"):
        apply_terrain(model, changed)
    with pytest.raises(ValueError, match="behavioral terrain"):
        apply_terrain(mujoco.MjModel.from_xml_path(str(sphere_xml)), terrain)


def test_configuration_and_heights_are_immutable(small_config):
    with pytest.raises(FrozenInstanceError):
        small_config.extent = 10
    terrain = generate_terrain(small_config, run_seed=1)
    with pytest.raises(ValueError, match="read-only"):
        terrain.heights[0, 0] = 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "rocky"},
        {"nrow": 1},
        {"ncol": True},
        {"extent": float("nan")},
        {"extent": 2},
        {"max_height": 0},
        {"max_slope_degrees": 45},
        {"roughness_amplitude": -1},
        {"episode_variation": 2},
        {"wavelength_min": 0.01},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TerrainConfig(**kwargs)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_rejected(seed, small_config):
    with pytest.raises(ValueError, match="run_seed"):
        generate_terrain(small_config, run_seed=seed)
