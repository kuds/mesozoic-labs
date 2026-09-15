"""Matched behavior videos and terrain maps from the scored trajectory.

The recorder sits inside the local VecEnv's automatic reset boundary.  A
terminal pose, physics heightfield and image are captured before the next
episode can replace them.  Each replay directory is published atomically
only after its video, maps, raw samples, path and manifest are complete.
Visualization dependencies are imported only when recording is requested.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import gymnasium as gym
import numpy as np

_REPLAY_LIGHTING = {"ambient": [0.35, 0.35, 0.35], "diffuse": [0.8, 0.8, 0.8], "specular": [0.1, 0.1, 0.1]}


class ReplayExportError(RuntimeError):
    """An explicitly requested replay could not be exported completely."""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def require_replay_dependencies() -> None:
    """Fail before rollout if the explicitly requested exporter is unavailable."""
    try:
        import imageio_ffmpeg
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401

        imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise ReplayExportError(
            "Video replay requires matplotlib and imageio-ffmpeg with its encoder; install mesozoic-labs[train,viz]"
        ) from exc


@dataclass(frozen=True)
class TerrainReplaySnapshot:
    kind: str
    heights_m: np.ndarray
    normalized_heights: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    manifest: dict[str, Any]


def capture_terrain_snapshot(raw_env: Any, reset_info: Mapping[str, Any]) -> TerrainReplaySnapshot:
    """Copy the compiled surface and verify its samples and physical geometry."""
    import mujoco

    model = raw_env.model
    floor = int(raw_env.floor_geom_id)
    if model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE:
        if model.geom_bodyid[floor] != 0 or not np.allclose(model.geom_quat[floor, 1:3], 0, rtol=0, atol=1e-12):
            raise ReplayExportError("Flat reference maps require a horizontal plane attached to the world")
        # There is no physical heightfield on the infinite flat plane.  The
        # 2x2 reference grid is solely a map extent, explicitly labelled so.
        extent = max(10.0, float(getattr(raw_env, "course_distance", 10.0)) * 2.0)
        elevation = float(model.geom_pos[floor, 2])
        return TerrainReplaySnapshot(
            kind="flat_plane",
            heights_m=np.full((2, 2), elevation, dtype=np.float64),
            normalized_heights=np.empty((0, 0), dtype=np.float32),
            x_m=np.array([-extent, extent]),
            y_m=np.array([-extent, extent]),
            manifest={
                "family": "flat_plane",
                "height_m": elevation,
                "physical_heightfield": False,
                "reference_grid_only": True,
            },
        )
    if model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_HFIELD:
        raise ReplayExportError("Behavior replay expects a horizontal plane or a heightfield floor")
    if not np.allclose(model.geom_quat[floor], [1, 0, 0, 0], rtol=0, atol=1e-12):
        raise ReplayExportError("Rotated heightfield maps are not supported by this exporter")
    field = int(model.geom_dataid[floor])
    nrow, ncol = int(model.hfield_nrow[field]), int(model.hfield_ncol[field])
    address = int(model.hfield_adr[field])
    normalized = np.array(model.hfield_data[address : address + nrow * ncol], dtype="<f4", copy=True).reshape(
        nrow, ncol
    )
    size = np.array(model.hfield_size[field], dtype=np.float64)
    position = np.array(model.geom_pos[floor], dtype=np.float64)
    heights = normalized.astype(np.float64) * size[2] + position[2]
    if not np.isfinite(heights).all():
        raise ReplayExportError("The replay physics heightfield contains non-finite heights")
    manifest = copy.deepcopy(dict(reset_info.get("terrain", {})))
    if not manifest and getattr(raw_env, "terrain", None) is not None:
        manifest = copy.deepcopy(raw_env.terrain.manifest())
    samples_hash = "sha256:" + hashlib.sha256(normalized.tobytes()).hexdigest()
    if manifest.get("samples_sha256") != samples_hash:
        raise ReplayExportError("The reset terrain manifest does not match the actual physics heightfield samples")
    # Normalized samples alone cannot identify a physical surface: changing
    # its scale or translation preserves those bytes while moving the ground.
    try:
        config = manifest["config"]
        extent, max_height = float(config["extent"]), float(config["max_height"])
        expected_size = [extent, extent, 2 * max_height, float(config["base_thickness"])]
        geometry_matches = (
            config["nrow"] == nrow
            and config["ncol"] == ncol
            and model.geom_bodyid[floor] == 0
            and np.isfinite(expected_size).all()
            and np.allclose(size, expected_size, rtol=0, atol=1e-12)
            and np.allclose(position, [0, 0, -max_height], rtol=0, atol=1e-12)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayExportError("The reset terrain manifest lacks valid physical geometry") from exc
    if not geometry_matches:
        raise ReplayExportError("The reset terrain manifest does not match the heightfield's physical geometry")
    live_terrain = getattr(raw_env, "terrain", None)
    if live_terrain is not None:
        live_manifest = live_terrain.manifest()
        recipe_keys = ("schema", "config", "run_seed", "episode_index", "samples_sha256", "terrain_sha256")
        if any(manifest.get(key) != live_manifest[key] for key in recipe_keys):
            raise ReplayExportError("The reset terrain manifest does not match the live terrain recipe")
    manifest.update(
        physical_heightfield=True,
        physics_samples_sha256=samples_hash,
        physics_height_scale_m=float(size[2]),
        physics_geom_z_m=float(position[2]),
        captured_from="model.hfield_data before the scored episode",
    )
    return TerrainReplaySnapshot(
        kind="heightfield",
        heights_m=heights,
        normalized_heights=normalized,
        x_m=np.linspace(position[0] - size[0], position[0] + size[0], ncol),
        y_m=np.linspace(position[1] - size[1], position[1] + size[1], nrow),
        manifest=manifest,
    )


def _local_bounds(snapshot: TerrainReplaySnapshot, path: np.ndarray) -> tuple[float, float, float, float]:
    center = path[-1] if len(path) else np.zeros(2)
    limits: list[float] = []
    for coordinates, value in ((snapshot.x_m, center[0]), (snapshot.y_m, center[1])):
        width = min(8.0, float(coordinates[-1] - coordinates[0]))
        low = max(float(coordinates[0]), min(float(value) - width / 2, float(coordinates[-1]) - width))
        limits.extend((low, low + width))
    return limits[0], limits[1], limits[2], limits[3]


def write_terrain_maps(
    snapshot: TerrainReplaySnapshot,
    path_xy_m: np.ndarray,
    destination: Path,
    *,
    episode_seed: int,
    terminated: bool,
) -> dict[str, Any]:
    """Draw exact-sample full/local maps, with centimetres on both colorbars."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    full_bounds = (snapshot.x_m[0], snapshot.x_m[-1], snapshot.y_m[0], snapshot.y_m[-1])
    local_bounds = _local_bounds(snapshot, path_xy_m)
    files = {}
    scales = {}
    flat = snapshot.kind == "flat_plane"
    prefix = "flat_plane" if flat else "terrain"
    for view, bounds in (("full", full_bounds), ("local", local_bounds)):
        figure = Figure(figsize=(8.6, 7.2), dpi=160, layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        values = snapshot.heights_m * 100.0
        if view == "local":
            region = values[
                np.ix_(
                    (snapshot.y_m >= bounds[2]) & (snapshot.y_m <= bounds[3]),
                    (snapshot.x_m >= bounds[0]) & (snapshot.x_m <= bounds[1]),
                )
            ]
            # A two-point flat-plane reference can have no vertices inside
            # the crop even though the infinite plane covers it entirely.
            region = region if region.size else values
            low, high = float(region.min()), float(region.max())
            padding = max(0.05, (high - low) * 0.03)
            low, high = low - padding, high + padding
        else:
            amplitude = max(0.05, float(np.max(np.abs(values))))
            low, high = -amplitude, amplitude
        surface = axis.pcolormesh(
            snapshot.x_m,
            snapshot.y_m,
            values,
            shading="nearest",
            cmap="RdBu_r",
            vmin=low,
            vmax=high,
            rasterized=True,
        )
        if len(path_xy_m):
            axis.plot(path_xy_m[:, 0], path_xy_m[:, 1], color="white", linewidth=3.0, zorder=3)
            axis.plot(path_xy_m[:, 0], path_xy_m[:, 1], color="#202735", linewidth=1.3, label="Actual path", zorder=4)
            axis.scatter(
                float(path_xy_m[0, 0]),
                float(path_xy_m[0, 1]),
                c="#008450",
                edgecolors="white",
                s=65,
                marker="o",
                label="Start",
                zorder=5,
            )
            axis.scatter(
                float(path_xy_m[-1, 0]),
                float(path_xy_m[-1, 1]),
                c="#ed6b00",
                s=85,
                marker="X",
                edgecolors="white",
                label="Terminated" if terminated else "End",
                zorder=6,
            )
        axis.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="World x (m)", ylabel="World y (m)", aspect="equal")
        template = snapshot.manifest.get("config", {}).get("template", "sloped")
        subject = "Flat plane reference (no heightfield)" if flat else f"Physics terrain height · {template}"
        subtitle = (
            "Full map · color scale centered on zero"
            if view == "full"
            else "Local view at final position · local color scale"
        )
        axis.set_title(f"{subject}\n{subtitle} · episode seed {episode_seed}", loc="left", fontsize=12, pad=14)
        axis.grid(color="#425066", alpha=0.18, linewidth=0.6)
        axis.legend(loc="upper right", framealpha=0.95, fontsize=9)
        colorbar = figure.colorbar(surface, ax=axis, shrink=0.88, pad=0.04)
        colorbar.set_label("Surface height (cm)")
        name = f"{prefix}_{view}_map.png"
        figure.savefig(destination / name, dpi=160)
        figure.clear()
        files[f"{view}_map"] = name
        scales[view] = {"bounds_m": [float(value) for value in bounds], "color_min_cm": low, "color_max_cm": high}
    return {"files": files, "map_scales": scales, "height_units": "cm", "coordinate_units": "m"}


def _open_video_writer(path: Path, frame: np.ndarray, fps: float) -> Any:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path),
        (frame.shape[1], frame.shape[0]),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    return writer


def _verify_video(path: Path, expected_frames: int, fps: float) -> dict[str, float | int]:
    """Decode the completed stream before publishing an apparently valid MP4."""
    import imageio_ffmpeg

    try:
        frames, duration = imageio_ffmpeg.count_frames_and_secs(path)
    except RuntimeError as exc:
        raise ReplayExportError("The encoded replay could not be decoded") from exc
    if frames != expected_frames or not math.isfinite(duration):
        raise ReplayExportError("The encoded replay frame count does not match the captured trajectory")
    # FFmpeg's diagnostic clock has two decimal places. A frame's timestamp
    # can denote its start or end across encoder versions, so allow one frame.
    if abs(duration - expected_frames / fps) > max(0.02, 1.0 / fps) + 1e-9:
        raise ReplayExportError("The encoded replay duration does not match its frame rate")
    return {"decoded_frame_count": int(frames), "decoded_duration_s": float(duration)}


class BehaviorReplayRecorder(gym.Wrapper):
    """Capture the underlying Gym trajectory before vectorized auto-reset."""

    def __init__(self, env: gym.Env, output_dir: Path, *, fps: float = 25.0, context: Mapping[str, Any] | None = None):
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.raw_env: Any = env.unwrapped
        self.dt = float(getattr(self.raw_env, "dt"))
        if not math.isfinite(self.fps) or not 0 < self.fps <= 1.0 / self.dt + 1e-9:
            raise ReplayExportError("video_fps must be finite, positive and no greater than the control frequency")
        self.context = copy.deepcopy(dict(context or {}))
        self._armed: tuple[int, int] | None = None
        self._temporary: Path | None = None
        self._writer: Any = None
        self._pending_frame: np.ndarray | None = None
        self._pending_time_s = 0.0
        self._frame_times: list[float] = []
        self._path: list[np.ndarray] = []
        self._path_times: list[float] = []
        self._snapshot: TerrainReplaySnapshot | None = None
        self._reset_info: dict[str, Any] = {}
        self._episode = 0
        self._seed = 0
        self.completed: list[dict[str, Any]] = []

    def arm_episode(self, episode_index: int, seed: int) -> None:
        if self._temporary is not None:
            raise ReplayExportError("Finish the active replay before arming another episode")
        self._armed = (int(episode_index), int(seed))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        observation, info = self.env.reset(**kwargs)
        if self._armed is not None:
            self._episode, self._seed = self._armed
            self._armed = None  # An automatic reset must never begin another recording.
            self._snapshot = capture_terrain_snapshot(self.raw_env, info)
            self._reset_info = copy.deepcopy(info)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"episode_{self._episode:03d}_seed_{self._seed}"
            if (self.output_dir / stem).exists():
                raise ReplayExportError(f"Replay destination already exists: {self.output_dir / stem}")
            self._temporary = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=self.output_dir))
            self._path = [np.array(self.raw_env.data.qpos[:2], dtype=np.float64, copy=True)]
            self._path_times = [0.0]
            self._frame_times = []
            self._pending_time_s = 0.0
            self._pending_frame = self._render_frame()
            self._writer = _open_video_writer(self._temporary / "replay.mp4", self._pending_frame, self.fps)
        return observation, info

    def _render_frame(self) -> np.ndarray:
        # An episode reset may activate a different precompiled terrain or
        # plane model. Apply the visual settings to the model actually being
        # rendered and restore them immediately, including on renderer error.
        headlight = self.raw_env.model.vis.headlight
        previous = {name: np.array(getattr(headlight, name), copy=True) for name in _REPLAY_LIGHTING}
        try:
            for name, values in _REPLAY_LIGHTING.items():
                getattr(headlight, name)[:] = values
            frame = np.asarray(self.raw_env.render())
        finally:
            for name, restored_values in previous.items():
                getattr(headlight, name)[:] = restored_values
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ReplayExportError("RGB replay rendering must return an HxWx3 uint8 image")
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            raise ReplayExportError("MP4 replay dimensions must be even")
        return np.array(frame, copy=True)

    def _flush_pending(self) -> None:
        if self._writer is None or self._pending_frame is None:
            raise ReplayExportError("Replay writer has no pending frame")
        self._writer.send(self._pending_frame)
        self._frame_times.append(self._pending_time_s)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._temporary is not None:
            self._path.append(np.array(self.raw_env.data.qpos[:2], dtype=np.float64, copy=True))
            time_s = (len(self._path) - 1) * self.dt
            self._path_times.append(time_s)
            if terminated or truncated or len(self._path) - 1 >= self.raw_env.max_episode_steps:
                # Replace the last pending display interval with the terminal
                # frame.  No extra end frame stretches or slows the replay.
                self.finish(terminated=terminated, truncated=truncated, terminal_info=info)
            elif math.floor(time_s * self.fps + 1e-9) > len(self._frame_times):
                self._flush_pending()
                self._pending_frame = self._render_frame()
                self._pending_time_s = time_s
        return observation, float(reward), terminated, truncated, info

    def finish(
        self, *, terminated: bool = False, truncated: bool = False, terminal_info: Mapping[str, Any] | None = None
    ) -> None:
        """Finalize before auto-reset, or at an evaluator-enforced horizon."""
        if self._temporary is None:
            return
        temporary = self._temporary
        assert self._snapshot is not None and self._pending_frame is not None
        duration = self._path_times[-1]
        self._pending_frame = self._render_frame()
        self._pending_time_s = duration
        self._flush_pending()
        self._writer.close()
        self._writer = None
        expected_frames = max(1, math.ceil(duration * self.fps - 1e-9))
        if len(self._frame_times) != expected_frames:
            raise ReplayExportError("Video frame count does not match the fixed-FPS episode duration")
        video_validation = _verify_video(temporary / "replay.mp4", expected_frames, self.fps)
        path = np.asarray(self._path)
        if self._snapshot.kind == "flat_plane":
            # The flat reference map represents an infinite physical plane;
            # unlike a finite heightfield, its plot can grow to fit the route.
            extent = max(float(np.max(np.abs(self._snapshot.x_m))), float(np.max(np.abs(path))) + 2.0)
            self._snapshot = replace(self._snapshot, x_m=np.array([-extent, extent]), y_m=np.array([-extent, extent]))
        maps = write_terrain_maps(self._snapshot, path, temporary, episode_seed=self._seed, terminated=terminated)
        np.savez_compressed(
            temporary / "terrain_and_path.npz",
            heights_m=self._snapshot.heights_m,
            normalized_heights=self._snapshot.normalized_heights,
            x_m=self._snapshot.x_m,
            y_m=self._snapshot.y_m,
            path_xy_m=path,
            path_time_s=np.asarray(self._path_times),
            frame_sample_time_s=np.asarray(self._frame_times),
            frame_display_time_s=np.arange(len(self._frame_times)) / self.fps,
        )
        files = {"video": "replay.mp4", "raw_terrain_and_path": "terrain_and_path.npz", **maps["files"]}
        for name in files.values():
            if not (temporary / name).is_file() or (temporary / name).stat().st_size == 0:
                raise ReplayExportError(f"Replay output was not written: {name}")
        manifest = {
            "schema": "mesozoic.behavior-replay/v1",
            "status": "complete",
            "episode": self._episode,
            "episode_seed": self._seed,
            "same_scored_trajectory": True,
            "context": self.context,
            "reset_info": self._reset_info,
            "terrain": self._snapshot.manifest,
            "surface_kind": self._snapshot.kind,
            "step_count": len(self._path) - 1,
            "simulation_duration_s": duration,
            "video_fps": self.fps,
            "video_frame_count": len(self._frame_times),
            "video_duration_s": len(self._frame_times) / self.fps,
            "video_validation": video_validation,
            "frame_width": self._pending_frame.shape[1],
            "frame_height": self._pending_frame.shape[0],
            "terminal_frame_before_autoreset": True,
            "terminal_frame_rule": "replace final pending display interval; duration rounded up by less than one frame",
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": (terminal_info or {}).get("termination_reason"),
            "maps": {key: value for key, value in maps.items() if key != "files"},
            "files": {key: {"path": name, "sha256": _sha(temporary / name)} for key, name in files.items()},
        }
        (temporary / "manifest.json").write_text(json.dumps(_json_value(manifest), indent=2, allow_nan=False) + "\n")
        destination = self.output_dir / f"episode_{self._episode:03d}_seed_{self._seed}"
        temporary.replace(destination)
        self.completed.append(
            {
                "episode": self._episode,
                "episode_seed": self._seed,
                "directory": str(destination),
                "manifest": str(destination / "manifest.json"),
                **{key: str(destination / name) for key, name in files.items()},
            }
        )
        self._temporary = None
        self._pending_frame = None

    def abort(self) -> None:
        """Remove unpublished partial outputs without closing the caller's env."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                # Abort never publishes the partial encoder output; preserve
                # the inference/render/export error that triggered cleanup.
                pass
            finally:
                self._writer = None
        if self._temporary is not None:
            shutil.rmtree(self._temporary, ignore_errors=True)
            self._temporary = None
        self._armed = None


@contextmanager
def record_evaluation_replays(
    vec_env: Any, output_dir: Path, *, fps: float = 25.0, context: Mapping[str, Any] | None = None
) -> Iterator[BehaviorReplayRecorder]:
    """Temporarily record inside the local single-environment VecEnv boundary."""
    require_replay_dependencies()
    inner = vec_env
    while hasattr(inner, "venv"):
        inner = inner.venv
    if not hasattr(inner, "envs") or len(inner.envs) != 1:
        raise ReplayExportError("Video replay requires one local DummyVecEnv environment")
    original = inner.envs[0]
    recorder = BehaviorReplayRecorder(original, output_dir, fps=fps, context=context)
    raw_env = original.unwrapped
    original_render_mode = raw_env.render_mode
    # A fresh context uploads the current episode's heightfield when it is
    # first rendered.  A cached context can otherwise show old GPU samples.
    if getattr(raw_env, "_renderer", None) is not None:
        raw_env._renderer.close()
        raw_env._renderer = None
    raw_env.render_mode = "rgb_array"
    recorder.context["replay_lighting"] = copy.deepcopy(_REPLAY_LIGHTING)
    inner.envs[0] = recorder
    try:
        yield recorder
        if recorder._temporary is not None:
            raise ReplayExportError("Evaluation exited before the active replay was finalized")
    finally:
        active_error = sys.exc_info()[1]
        inner.envs[0] = original
        raw_env.render_mode = original_render_mode
        try:
            if getattr(raw_env, "_renderer", None) is not None:
                raw_env._renderer.close()
        except Exception as cleanup_error:
            if active_error is None:
                raise
            active_error.add_note(f"Replay renderer cleanup also failed: {cleanup_error}")
        finally:
            raw_env._renderer = None
            recorder.abort()
