"""Terrain geometry is checked against MuJoCo's own ray/collision queries."""

import json
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


def test_original_sloped_samples_and_recipe_hash_remain_compatible():
    # Frozen output from the shipped v1 implementation, not a second copy of
    # the generator. Old recorded terrain recipes must still replay exactly.
    manifest = generate_terrain(TerrainConfig(), run_seed=42).manifest()
    assert manifest["samples_sha256"] == "sha256:bc0108a462d2d616eeca007b97b5c23f30abfbfff6f4c7bc9e42ac7abc83a2db"
    assert manifest["terrain_sha256"] == "sha256:c438e649fe70f3233ca40b1fc0c3f997d3398a8021db49bb7ca9b8bc2d983589"
    assert manifest["schema"] == "mesozoic.gentle-terrain/v1"


def test_legacy_small_encoding_height_ignores_unused_feature_defaults():
    config = TerrainConfig(max_height=0.01, roughness_amplitude=0.005)
    terrain = generate_terrain(config, run_seed=42)
    manifest = terrain.manifest()
    assert manifest["samples_sha256"] == "sha256:e9ba3b46aa1cd11e5fc7853049749e1455fd9948d7c9320d1c7d86983b283709"
    assert manifest["terrain_sha256"] == "sha256:2467eb6a341b5efb9319a30aad680322af602f9469892d06c32e7c75afcc22d4"
    recorded = json.loads(json.dumps(manifest, allow_nan=False))
    replay = generate_terrain(TerrainConfig(**recorded["config"]), run_seed=recorded["run_seed"])
    np.testing.assert_array_equal(terrain.heights, replay.heights)
    # Cross-field constraints apply only to the active feature generator.
    with pytest.raises(ValueError, match="feature_height"):
        replace(config, template="mixed")


@pytest.mark.parametrize("template", ["bumps", "depressions", "mixed"])
@pytest.mark.parametrize("seed", [0, 42, 43])
def test_feature_templates_have_bounded_local_relief_on_the_walk_corridor(template, seed):
    config = TerrainConfig(template=template, extent=16, nrow=161, ncol=161)
    terrain = generate_terrain(config, run_seed=seed)
    manifest = terrain.manifest()
    assert manifest["schema"] == "mesozoic.terrain-templates/v2"
    assert 0 < manifest["maximum_grade_degrees"] <= config.max_slope_degrees
    assert np.abs(terrain.heights).max() < 0.05  # Centimetre features, not metre-scale grading.
    x, y = np.meshgrid(np.linspace(3, 15, 121), np.linspace(-3, 3, 61))
    sampled = terrain.height_at(x, y)
    if template == "bumps":
        assert np.all(sampled >= 0)
        assert sampled.max() > 0.005
    elif template == "depressions":
        assert np.all(sampled <= 0)
        assert sampled.min() < -0.005
    else:
        assert sampled.max() > 0.005
        assert sampled.min() < -0.005
    # A plane fit must leave substantial local relief; a tilted flat map
    # would have negligible residual and cannot satisfy this requirement.
    design = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    fitted_plane = design @ np.linalg.lstsq(design, sampled.ravel(), rcond=None)[0]
    assert np.sqrt(np.mean((sampled.ravel() - fitted_plane) ** 2)) > 0.002
    assert len(terrain.features) > 20
    assert all(abs(feature.height) <= config.feature_height for feature in terrain.features)
    # Also sample between grid vertices at the circular spawn boundary.
    angles = np.linspace(-np.pi, np.pi, 361)
    np.testing.assert_array_equal(
        terrain.height_at(config.apron_radius * np.cos(angles), config.apron_radius * np.sin(angles)),
        0,
    )


@pytest.mark.parametrize("template", ["bumps", "depressions", "mixed"])
def test_local_features_repeat_and_episode_changes_stay_small(template, small_config):
    config = replace(small_config, template=template)
    first = generate_terrain(config, run_seed=41, episode_index=4)
    again = generate_terrain(config, run_seed=41, episode_index=4)
    later = generate_terrain(config, run_seed=41, episode_index=5)
    different_run = generate_terrain(config, run_seed=42, episode_index=4)
    np.testing.assert_array_equal(first.heights, again.heights)
    assert first.manifest(include_features=True) == again.manifest(include_features=True)
    assert first.manifest()["samples_sha256"] != later.manifest()["samples_sha256"]
    assert np.corrcoef(first.heights.ravel(), later.heights.ravel())[0, 1] > 0.9
    assert first.manifest()["samples_sha256"] != different_run.manifest()["samples_sha256"]
    no_variation = replace(config, episode_variation=0)
    np.testing.assert_array_equal(
        generate_terrain(no_variation, run_seed=41, episode_index=4).heights,
        generate_terrain(no_variation, run_seed=41, episode_index=5).heights,
    )


def test_feature_manifest_is_compact_and_optional_details_report_post_cap_heights(small_config):
    terrain = generate_terrain(replace(small_config, template="mixed", feature_height=0.1), run_seed=3)
    manifest = terrain.manifest()
    details = terrain.manifest(include_features=True)["localized_features"]
    assert "features" not in manifest["localized_features"]
    assert 0 < details["height_scale_after_bounds"] < 1
    assert details["count"] == details["positive_count"] + details["negative_count"] == len(terrain.features)
    assert details["positive_count"] > 0 and details["negative_count"] > 0
    assert len(details["features"]) == len(terrain.features)
    assert details["scaled_component_height_range_m"][1] < 0.1
    for feature in details["features"]:
        assert feature["surface_height_at_center_m"] == terrain.height_at(feature["center_x_m"], feature["center_y_m"])


@pytest.mark.parametrize("template", ["sloped", "bumps", "depressions", "mixed"])
@pytest.mark.parametrize("include_features", [False, True])
def test_terrain_manifest_is_json_serializable(template, include_features, small_config):
    terrain = generate_terrain(
        replace(small_config, template=template, feature_height=np.float32(0.04)),
        run_seed=np.int64(42),
        episode_index=np.int64(1),
    )
    manifest = terrain.manifest(include_features=include_features)
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest


def test_depressions_are_real_solid_surface_in_mujoco(sphere_xml, small_config):
    terrain = generate_terrain(replace(small_config, template="depressions"), run_seed=17)
    model = build_terrain_model(sphere_xml, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    row, col = np.unravel_index(np.argmin(terrain.heights), terrain.heights.shape)
    x = -terrain.config.extent + col * terrain.config.dx
    y = -terrain.config.extent + row * terrain.config.dy
    assert terrain.height_at(x, y) < -0.005
    distance = mujoco.mj_rayHfield(model, data, 0, np.array([x, y, 1.0]), np.array([0.0, 0.0, -1.0]))
    assert distance > 1  # Below the original zero-height plane, still solid.
    assert 1 - distance == pytest.approx(terrain.height_at(x, y), abs=1e-10)
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)


def test_feature_template_flat_mode_has_no_features(small_config):
    terrain = generate_terrain(replace(small_config, template="mixed", mode="flat"), run_seed=3)
    np.testing.assert_array_equal(terrain.heights, 0)
    assert terrain.features == ()
    assert terrain.manifest()["localized_features"]["count"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"template": "caverns"},
        {"template": "mixed", "feature_radius_min": 0.1},
        {"feature_height": -0.01},
        {"feature_density": -1},
        {"template": "mixed", "feature_density": 100},
        {"template": "mixed", "feature_radius_min": 2, "feature_radius_max": 1},
    ],
)
def test_invalid_feature_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TerrainConfig(**kwargs)
