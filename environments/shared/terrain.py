"""Deterministic, opt-in gentle terrain for behavioral environments.

This module does not change the canonical species plants.  A terrain scene
replaces its horizontal floor with one finite heightfield; the caller owns
terrain-aware reset, termination, and out-of-bounds behavior.  The circular
spawn apron is exactly flat, including every triangle intersecting it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import mujoco

_HFIELD_NAME = "behavior_terrain"
_SCHEMA = "mesozoic.gentle-terrain/v1"


@dataclass(frozen=True)
class TerrainConfig:
    """Terrain dimensions in metres and maximum triangular grade in degrees.

    ``max_height`` bounds the encoding, not the requested roughness. Keeping
    it fixed lets new episodes replace samples without changing model bounds.
    ``extent`` is a half-extent: the default scene covers [-25, 25] on both axes.
    """

    mode: Literal["flat", "gentle"] = "gentle"
    extent: float = 25.0
    nrow: int = 251
    ncol: int = 251
    apron_radius: float = 3.0
    blend_width: float = 1.5
    max_height: float = 2.0
    max_slope_degrees: float = 3.0
    roughness_amplitude: float = 0.015
    wavelength_min: float = 1.5
    wavelength_max: float = 4.0
    episode_variation: float = 0.15
    base_thickness: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in ("flat", "gentle"):
            raise ValueError("terrain mode must be 'flat' or 'gentle'")
        for name in ("nrow", "ncol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 3:
                raise ValueError(f"{name} must be an integer >= 3")
        for name in (
            "extent",
            "apron_radius",
            "blend_width",
            "max_height",
            "wavelength_min",
            "wavelength_max",
            "base_thickness",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_slope_degrees", "roughness_amplitude", "episode_variation"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_slope_degrees >= 45:
            raise ValueError("gentle terrain requires max_slope_degrees < 45")
        if self.episode_variation > 1:
            raise ValueError("episode_variation must be <= 1")
        if self.wavelength_min > self.wavelength_max:
            raise ValueError("wavelength_min must not exceed wavelength_max")
        if self.roughness_amplitude > self.max_height:
            raise ValueError("roughness_amplitude must not exceed max_height")
        if self.apron_radius + self.cell_diagonal + self.blend_width >= self.extent:
            raise ValueError("terrain extent must contain the apron and its blend region")
        if self.wavelength_min < 4 * max(self.dx, self.dy):
            raise ValueError("terrain wavelengths require at least four samples per cycle")

    @property
    def dx(self) -> float:
        return 2 * self.extent / (self.ncol - 1)

    @property
    def dy(self) -> float:
        return 2 * self.extent / (self.nrow - 1)

    @property
    def cell_diagonal(self) -> float:
        return float(np.hypot(self.dx, self.dy))


def _maximum_gradient(heights: np.ndarray, dx: float, dy: float) -> float:
    """Largest gradient norm over MuJoCo's two triangles in each cell."""
    h00, h10 = heights[:-1, :-1], heights[:-1, 1:]
    h01, h11 = heights[1:, :-1], heights[1:, 1:]
    first = np.hypot((h10 - h00) / dx, (h11 - h10) / dy)
    second = np.hypot((h11 - h01) / dx, (h01 - h00) / dy)
    return float(max(first.max(), second.max()))


@dataclass(frozen=True, eq=False)
class TerrainRealization:
    """An immutable world-height grid; row zero is the negative-y edge.

    Samples are quantized through MuJoCo's float32 heightfield representation
    on construction. ``height_at`` therefore queries the physical samples,
    rather than an unquantized precursor. Outside the finite map it returns
    NaN by default; callers must handle the missing surface explicitly.
    """

    config: TerrainConfig
    run_seed: int
    episode_index: int
    heights: np.ndarray

    def __post_init__(self) -> None:
        _validate_seed(self.run_seed, "run_seed")
        _validate_seed(self.episode_index, "episode_index")
        heights = np.asarray(self.heights, dtype=np.float64)
        if heights.shape != (self.config.nrow, self.config.ncol):
            raise ValueError("terrain grid shape does not match its configuration")
        if not np.isfinite(heights).all():
            raise ValueError("terrain heights must be finite")
        if np.max(np.abs(heights)) > self.config.max_height:
            raise ValueError("terrain heights exceed the fixed encoding bounds")
        encoded = ((heights + self.config.max_height) / self.elevation_scale).astype(np.float32)
        heights = encoded.astype(np.float64) * self.elevation_scale + self.geom_z
        heights.setflags(write=False)
        object.__setattr__(self, "heights", heights)

    @property
    def geom_z(self) -> float:
        return -self.config.max_height

    @property
    def elevation_scale(self) -> float:
        return 2 * self.config.max_height

    @property
    def normalized_heights(self) -> np.ndarray:
        return ((self.heights - self.geom_z) / self.elevation_scale).astype(np.float32)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        e = self.config.extent
        return -e, e, -e, e

    def contains(self, x: Any, y: Any) -> Any:
        x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        result = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= self.config.extent)
        result &= np.abs(y) <= self.config.extent
        return bool(result) if result.ndim == 0 else result

    def height_at(self, x: Any, y: Any, *, outside: float = float("nan")) -> Any:
        """Sample MuJoCo's diagonal triangulation, never bilinear interpolation.

        The cell diagonal connects (x0,y0) to (x1,y1). Scalars and broadcastable
        arrays are accepted. Map boundaries belong to the surface.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        valid = self.contains(x, y)
        safe_x, safe_y = np.where(valid, x, 0.0), np.where(valid, y, 0.0)
        fx = np.clip((safe_x + self.config.extent) / self.config.dx, 0, self.config.ncol - 1)
        fy = np.clip((safe_y + self.config.extent) / self.config.dy, 0, self.config.nrow - 1)
        ix = np.minimum(np.floor(fx).astype(int), self.config.ncol - 2)
        iy = np.minimum(np.floor(fy).astype(int), self.config.nrow - 2)
        u, v = fx - ix, fy - iy
        h00, h10 = self.heights[iy, ix], self.heights[iy, ix + 1]
        h01, h11 = self.heights[iy + 1, ix], self.heights[iy + 1, ix + 1]
        lower = h00 * (1 - u) + h10 * (u - v) + h11 * v
        upper = h00 * (1 - v) + h01 * (v - u) + h11 * u
        result = np.where(valid, np.where(u >= v, lower, upper), outside)
        return float(result) if result.ndim == 0 else result

    def manifest(self) -> dict[str, Any]:
        """Replay recipe, exact sample hash, and measured geometric difficulty."""
        sample_hash = hashlib.sha256(self.normalized_heights.astype("<f4").tobytes()).hexdigest()
        recipe = {
            "schema": _SCHEMA,
            "config": asdict(self.config),
            "run_seed": self.run_seed,
            "episode_index": self.episode_index,
            "samples_sha256": f"sha256:{sample_hash}",
        }
        recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {
            **recipe,
            "terrain_sha256": f"sha256:{recipe_hash}",
            "minimum_height_m": float(self.heights.min()),
            "maximum_height_m": float(self.heights.max()),
            "maximum_grade_degrees": float(
                np.degrees(
                    np.arctan(
                        _maximum_gradient(
                            self.heights,
                            self.config.dx,
                            self.config.dy,
                        )
                    )
                )
            ),
            "grid_order": "rows increase with world y; columns increase with world x",
            "interpolation": "triangles with diagonal (x0,y0)-(x1,y1)",
        }


def _validate_seed(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def generate_terrain(config: TerrainConfig, *, run_seed: int, episode_index: int = 0) -> TerrainRealization:
    """Generate a run-specific smooth grade with small seeded episode changes.

    Random draws are local, independent of command/reset RNGs. A padded apron
    keeps every triangle crossing the requested spawn disk exactly at z=0.
    Both elevation and maximum triangle gradient are bounded after generation.
    """
    _validate_seed(run_seed, "run_seed")
    _validate_seed(episode_index, "episode_index")
    run_rng = np.random.default_rng(np.random.SeedSequence([int(run_seed), 0x54455252]))
    episode_rng = np.random.default_rng(np.random.SeedSequence([int(run_seed), int(episode_index), 0x45504953]))
    x, y = np.meshgrid(
        np.linspace(-config.extent, config.extent, config.ncol),
        np.linspace(-config.extent, config.extent, config.nrow),
    )
    heights = np.zeros_like(x)
    if config.mode == "gentle" and config.max_slope_degrees > 0:
        radius = np.hypot(x, y)
        # Any cell touching the requested disk has all vertices in this pad.
        distance = np.maximum(radius - config.apron_radius - config.cell_diagonal, 0)
        blend = np.clip(distance / config.blend_width, 0, 1)
        envelope = blend * blend * (3 - 2 * blend)
        grade_distance = np.where(
            distance < config.blend_width,
            distance**2 / (2 * config.blend_width),
            distance - config.blend_width / 2,
        )
        variation = config.episode_variation
        angle = run_rng.uniform(-np.pi, np.pi) + episode_rng.uniform(-variation, variation)
        grade = np.tan(np.radians(config.max_slope_degrees)) * run_rng.uniform(0.5, 0.85)
        grade *= 1 + episode_rng.uniform(-variation, variation)
        along = np.divide(x * np.cos(angle) + y * np.sin(angle), radius, out=np.zeros_like(x), where=radius > 0)
        heights = grade * along * grade_distance
        roughness = np.zeros_like(x)
        for _ in range(6):
            direction = run_rng.uniform(-np.pi, np.pi)
            wavelength = run_rng.uniform(config.wavelength_min, config.wavelength_max)
            phase = run_rng.uniform(-np.pi, np.pi) + episode_rng.uniform(-variation, variation)
            amplitude = run_rng.uniform(0.5, 1.0) * (1 + episode_rng.uniform(-variation, variation))
            coordinate = x * np.cos(direction) + y * np.sin(direction)
            roughness += amplitude * np.sin(2 * np.pi * coordinate / wavelength + phase)
        roughness *= config.roughness_amplitude / max(float(np.max(np.abs(roughness))), 1e-12)
        heights += envelope * roughness
        height_ratio = config.max_height / max(float(np.max(np.abs(heights))), 1e-12)
        grade_ratio = np.tan(np.radians(config.max_slope_degrees)) / max(
            _maximum_gradient(heights, config.dx, config.dy),
            1e-12,
        )
        # Small margin absorbs float32 heightfield quantization in the cap.
        heights *= min(1.0, height_ratio, grade_ratio) * (1 - 1e-4)
    return TerrainRealization(config, int(run_seed), int(episode_index), heights)


def build_terrain_model(model_path: str | PathLike[str] | mujoco.MjSpec, realization: TerrainRealization) -> Any:
    """Compile a new scene from its source, preserving relative asset paths.

    Only a world-body horizontal plane named ``floor`` may be replaced. Extra
    planes are rejected so an invisible fallback floor cannot fill depressions.
    The canonical model/source are not changed. This is scene construction,
    not an assertion that the resulting model has the canonical plant identity.
    """
    import mujoco

    spec = model_path.copy() if isinstance(model_path, mujoco.MjSpec) else mujoco.MjSpec.from_file(str(model_path))
    floor = spec.geom("floor")
    if floor is None or floor.type != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("terrain source must have a plane geom named 'floor'")
    if floor not in list(spec.worldbody.geoms):
        raise ValueError("terrain floor must belong directly to the world body")
    if not np.allclose(floor.quat, [1, 0, 0, 0]) or not np.allclose(floor.pos, [0, 0, 0]):
        raise ValueError("terrain source floor must be unrotated at the world origin")
    if any(g.type == mujoco.mjtGeom.mjGEOM_PLANE and g.name != "floor" for g in spec.geoms):
        raise ValueError("terrain source must not contain additional plane geoms")
    config = realization.config
    spec.add_hfield(
        name=_HFIELD_NAME,
        nrow=config.nrow,
        ncol=config.ncol,
        size=[config.extent, config.extent, realization.elevation_scale, config.base_thickness],
        userdata=np.zeros(config.nrow * config.ncol).tolist(),
    )
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = _HFIELD_NAME
    floor.pos = [0, 0, realization.geom_z]
    model = spec.compile()
    apply_terrain(model, realization)
    return model


def apply_terrain(model: Any, realization: TerrainRealization, data: Any = None) -> None:
    """Replace samples in an existing terrain model without recompilation.

    Extents, scale, topology and geom transform must match the compiled scene;
    no derived model constants change, so ``mj_setConst`` is unnecessary.
    If data is supplied, refresh kinematics/contacts with ``mj_forward``.
    A live renderer additionally needs ``mjr_uploadHField`` on its GL context.
    Never use ``mj_geomDistance`` on a heightfield for spawn settling: it is
    not a reliable distance-to-terrain query. Use the known flat apron instead.
    """
    import mujoco

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, _HFIELD_NAME)
    if geom_id < 0 or hfield_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_HFIELD:
        raise ValueError("model was not constructed with a behavioral terrain floor")
    if model.geom_dataid[geom_id] != hfield_id:
        raise ValueError("floor references a different heightfield")
    config = realization.config
    expected_size = [config.extent, config.extent, realization.elevation_scale, config.base_thickness]
    if (
        model.hfield_nrow[hfield_id] != config.nrow
        or model.hfield_ncol[hfield_id] != config.ncol
        or not np.allclose(model.hfield_size[hfield_id], expected_size, rtol=0, atol=1e-12)
        or not np.allclose(model.geom_pos[geom_id], [0, 0, realization.geom_z], rtol=0, atol=1e-12)
        or not np.allclose(model.geom_quat[geom_id], [1, 0, 0, 0], rtol=0, atol=1e-12)
    ):
        raise ValueError("terrain dimensions differ from the fixed compiled scene")
    start = int(model.hfield_adr[hfield_id])
    model.hfield_data[start : start + config.nrow * config.ncol] = realization.normalized_heights.ravel()
    if data is not None:
        mujoco.mj_forward(model, data)
