"""Replay/heightfield matching before VecEnv autoreset, with bounded frame timing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from environments.shared import behavior_replay
from environments.shared.behavior_evaluation import evaluate_behavior
from environments.shared.behavior_replay import (
    ReplayExportError,
    capture_terrain_snapshot,
    record_evaluation_replays,
    write_terrain_maps,
)
from environments.shared.direction_commands import DirectionCommandController
from environments.shared.terrain import TerrainConfig, apply_terrain, build_terrain_model, generate_terrain


class _RenderEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, horizon=10, *, terrain=True, fail_at=None):
        self.dt = 0.01
        self.max_episode_steps = horizon
        self.fail_at = fail_at
        self.course_distance = 10.0
        self.render_mode = None
        self._renderer = None
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        spec = mujoco.MjSpec.from_string(
            '<mujoco><worldbody><geom name="floor" type="plane" size="0 0 .1"/>'
            '<body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>'
            "</body></worldbody></mujoco>"
        )
        self.config = TerrainConfig(extent=6.0, nrow=61, ncol=61, apron_radius=1.0, blend_width=0.5)
        self.terrain = generate_terrain(self.config, run_seed=0) if terrain else None
        self.model = build_terrain_model(spec, self.terrain) if terrain else spec.compile()
        self.data = mujoco.MjData(self.model)
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.direction_controller = DirectionCommandController()
        self._step = 0
        self._seed = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = self._seed + 1000 if seed is None else seed
        self._step = 0
        mujoco.mj_resetData(self.model, self.data)
        if self.terrain is not None:
            self.terrain = generate_terrain(self.config, run_seed=self._seed)
            apply_terrain(self.model, self.terrain, self.data)
        self.direction_controller.reset(np.random.default_rng(self._seed), 0.0)
        return np.zeros(3, dtype=np.float32), {
            "episode_seed": self._seed,
            "terrain": self.terrain.manifest() if self.terrain is not None else {"family": "flat_plane"},
        }

    def step(self, action):
        state = self.direction_controller.update(self._step * self.dt, 0.0)
        self._step += 1
        self.data.qpos[0] = self._step * 0.1
        terminated = self._step == self.fail_at
        truncated = self._step >= self.max_episode_steps
        info = {
            **state.as_info(),
            "actual_speed": 1.05,
            "actual_yaw_rate": 0.0,
            "tracking_error_v": 0.0,
            "tracking_error_yaw": 0.0,
            "tracking_in_tolerance": True,
        }
        if terminated:
            info["termination_reason"] = "fallen"
        return np.zeros(3, dtype=np.float32), 1.0, terminated, truncated, info

    def render(self):
        assert self.render_mode == "rgb_array"
        return np.full((32, 48, 3), self._step % 255, dtype=np.uint8)


class _Model:
    def predict(self, observation, deterministic=True):
        return np.zeros((1, 1), dtype=np.float32), None


@pytest.fixture
def fake_media(monkeypatch):
    frames = []

    class Writer:
        def __init__(self, path):
            self.path = path
            self.current = []
            frames.append(self.current)

        def send(self, frame):
            self.current.append(np.array(frame, copy=True))

        def close(self):
            self.path.write_bytes(b"fake-mp4:" + bytes([len(self.current)]))

    def maps(snapshot, path, destination, **kwargs):
        prefix = "flat_plane" if snapshot.kind == "flat_plane" else "terrain"
        files = {"full_map": f"{prefix}_full_map.png", "local_map": f"{prefix}_local_map.png"}
        for name in files.values():
            (destination / name).write_bytes(b"test-map")
        return {"files": files, "height_units": "cm", "coordinate_units": "m"}

    monkeypatch.setattr(behavior_replay, "require_replay_dependencies", lambda: None)
    monkeypatch.setattr(behavior_replay, "_open_video_writer", lambda path, frame, fps: Writer(path))
    monkeypatch.setattr(
        behavior_replay,
        "_verify_video",
        lambda path, count, fps: {"decoded_frame_count": count, "decoded_duration_s": count / fps},
    )
    monkeypatch.setattr(behavior_replay, "write_terrain_maps", maps)
    return frames


def _vec(raw):
    sb3_vec = pytest.importorskip("stable_baselines3.common.vec_env")
    return sb3_vec.VecNormalize(sb3_vec.DummyVecEnv([lambda: raw]))


def test_snapshot_copies_exact_physics_samples_and_rejects_stale_manifest():
    raw = _RenderEnv()
    _, info = raw.reset(seed=42)
    snapshot = capture_terrain_snapshot(raw, info)
    assert snapshot.manifest["physics_samples_sha256"] == info["terrain"]["samples_sha256"]
    np.testing.assert_array_equal(snapshot.heights_m, raw.terrain.heights)
    raw.model.hfield_data[0] += 0.01
    assert snapshot.normalized_heights.flat[0] != raw.model.hfield_data[0]
    with pytest.raises(ReplayExportError, match="actual physics"):
        capture_terrain_snapshot(raw, info)


@pytest.mark.parametrize("parameter", ["height_scale", "extent", "vertical_position", "horizontal_position"])
def test_snapshot_rejects_changed_physical_geometry_even_with_identical_samples(parameter):
    raw = _RenderEnv()
    _, info = raw.reset(seed=42)
    samples = raw.model.hfield_data.copy()
    if parameter == "height_scale":
        raw.model.hfield_size[0, 2] *= 2
    elif parameter == "extent":
        raw.model.hfield_size[0, 0] *= 2
    elif parameter == "vertical_position":
        raw.model.geom_pos[raw.floor_geom_id, 2] += 0.01
    else:
        raw.model.geom_pos[raw.floor_geom_id, 0] += 0.01
    np.testing.assert_array_equal(samples, raw.model.hfield_data)
    with pytest.raises(ReplayExportError, match="physical geometry"):
        capture_terrain_snapshot(raw, info)


def test_snapshot_does_not_label_a_tilted_plane_as_a_flat_reference():
    raw = _RenderEnv(terrain=False)
    _, info = raw.reset(seed=42)
    raw.model.geom_quat[raw.floor_geom_id] = [math.cos(0.05), math.sin(0.05), 0, 0]
    with pytest.raises(ReplayExportError, match="horizontal plane"):
        capture_terrain_snapshot(raw, info)


def test_snapshot_rejects_mutated_recipe_metadata_with_an_unchanged_hash():
    raw = _RenderEnv()
    _, info = raw.reset(seed=42)
    info["terrain"]["run_seed"] = 43
    with pytest.raises(ReplayExportError, match="live terrain recipe"):
        capture_terrain_snapshot(raw, info)


@pytest.mark.parametrize("steps", [1, 4, 6, 8, 10])
def test_fixed_fps_keeps_terminal_frame_without_stretching_episode(tmp_path, fake_media, steps):
    raw = _RenderEnv(horizon=steps)
    vec = _vec(raw)
    try:
        report = evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path, record_video=True)
        replay = report["episodes"][0]["replay"]
        manifest = json.loads(Path(replay["manifest"]).read_text())
        expected = math.ceil(steps * raw.dt * 25 - 1e-9)
        assert manifest["video_frame_count"] == len(fake_media[0]) == expected
        assert manifest["simulation_duration_s"] == steps * raw.dt
        assert 0 <= manifest["video_duration_s"] - manifest["simulation_duration_s"] < 1 / 25 + 1e-9
        assert np.all(fake_media[0][-1] == steps)  # Old terminal state, not autoreset frame 0.
        assert raw._step == 0
        assert raw._seed == 1042  # DummyVecEnv already reset to a different map.
        npz = np.load(replay["raw_terrain_and_path"])
        expected_terrain = generate_terrain(raw.config, run_seed=42)
        np.testing.assert_array_equal(npz["normalized_heights"], expected_terrain.normalized_heights)
        assert npz["path_xy_m"][-1, 0] == pytest.approx(steps * 0.1)
        assert npz["frame_sample_time_s"][-1] == steps * raw.dt
        assert manifest["reset_info"]["episode_seed"] == 42
        for record in manifest["files"].values():
            path = Path(replay["directory"]) / record["path"]
            assert record["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert raw.render_mode is None
        assert vec.training and vec.norm_reward
        assert Path(report["outputs"]["replay_index"]).is_file()
    finally:
        vec.close()


def test_fall_saves_terminal_state_and_matching_terrain_before_autoreset(tmp_path, fake_media):
    raw = _RenderEnv(horizon=20, fail_at=6)
    vec = _vec(raw)
    try:
        report = evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path, record_video=True)
        manifest = json.loads(Path(report["episodes"][0]["replay"]["manifest"]).read_text())
        assert manifest["terminated"] and manifest["termination_reason"] == "fallen"
        assert manifest["step_count"] == 6
        assert report["fall_count"] == 1
        assert manifest["terminal_frame_before_autoreset"]
        assert np.all(fake_media[0][-1] == 6)
    finally:
        vec.close()


def test_map_failure_never_publishes_a_partial_replay_and_restores_wrapper_and_lighting(
    tmp_path, fake_media, monkeypatch
):
    raw = _RenderEnv(horizon=4)
    vec = _vec(raw)
    original = vec.venv.envs[0]
    original_ambient = raw.model.vis.headlight.ambient.copy()

    def failed_maps(*args, **kwargs):
        raise RuntimeError("map export failed")

    monkeypatch.setattr(behavior_replay, "write_terrain_maps", failed_maps)
    try:
        with pytest.raises(RuntimeError, match="map export failed"):
            evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path, record_video=True)
        assert list((tmp_path / "replays").iterdir()) == []
        assert vec.venv.envs[0] is original
        assert raw.render_mode is None
        np.testing.assert_array_equal(raw.model.vis.headlight.ambient, original_ambient)
        assert vec.training and vec.norm_reward
    finally:
        vec.close()


def test_invalid_encoded_video_is_not_published(tmp_path, fake_media, monkeypatch):
    def invalid_video(*args):
        raise ReplayExportError("encoded replay frame count mismatch")

    monkeypatch.setattr(behavior_replay, "_verify_video", invalid_video)
    vec = _vec(_RenderEnv(horizon=4))
    try:
        with pytest.raises(ReplayExportError, match="frame count mismatch"):
            evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path, record_video=True)
        assert list((tmp_path / "replays").iterdir()) == []
    finally:
        vec.close()


def test_missing_visualization_dependency_is_explicit_and_unrecorded_evaluation_still_works(tmp_path, monkeypatch):
    def missing():
        raise ReplayExportError("install mesozoic-labs[train,viz]")

    monkeypatch.setattr(behavior_replay, "require_replay_dependencies", missing)
    raw = _RenderEnv(horizon=4)
    vec = _vec(raw)
    try:
        evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path / "plain")
        with pytest.raises(ReplayExportError, match="train,viz"):
            evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path / "video", record_video=True)
        assert vec.training and vec.norm_reward
    finally:
        vec.close()


def test_inference_failure_keeps_original_error_and_discards_unpublished_video(tmp_path, fake_media):
    class BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("original inference failure")

    raw = _RenderEnv()
    vec = _vec(raw)
    try:
        with pytest.raises(RuntimeError, match="original inference failure"):
            evaluate_behavior(BrokenModel(), vec, episode_seeds=[42], output_dir=tmp_path, record_video=True)
        assert list((tmp_path / "replays").iterdir()) == []
        assert raw.render_mode is None
        assert vec.training and vec.norm_reward
    finally:
        vec.close()


def test_renderer_cleanup_failure_keeps_original_error_and_still_aborts_replay(tmp_path, fake_media):
    class BrokenRenderer:
        def close(self):
            raise RuntimeError("renderer cleanup failed")

    raw = _RenderEnv()

    class BrokenModel:
        def predict(self, *args, **kwargs):
            raw._renderer = BrokenRenderer()
            raise RuntimeError("original inference failure")

    vec = _vec(raw)
    original = vec.venv.envs[0]
    try:
        with pytest.raises(RuntimeError, match="original inference failure"):
            evaluate_behavior(BrokenModel(), vec, episode_seeds=[42], output_dir=tmp_path, record_video=True)
        assert list((tmp_path / "replays").iterdir()) == []
        assert raw._renderer is None
        assert raw.render_mode is None
        assert vec.venv.envs[0] is original
        assert vec.training and vec.norm_reward
    finally:
        raw._renderer = None
        vec.close()


def test_visual_lighting_uses_and_restores_each_model_selected_on_reset(tmp_path, fake_media, monkeypatch):
    raw = _RenderEnv(horizon=4)
    first_model = raw.model
    second_model = copy.copy(first_model)
    first_model.vis.headlight.ambient[:] = [0.11, 0.11, 0.11]
    second_model.vis.headlight.ambient[:] = [0.22, 0.22, 0.22]
    original_reset, original_render = raw.reset, raw.render
    rendered_models = []

    def reset(**kwargs):
        raw.model = first_model if kwargs.get("seed") == 1 else second_model
        raw.data = mujoco.MjData(raw.model)
        return original_reset(**kwargs)

    def render():
        np.testing.assert_allclose(raw.model.vis.headlight.ambient, [0.35, 0.35, 0.35])
        rendered_models.append(id(raw.model))
        return original_render()

    monkeypatch.setattr(raw, "reset", reset)
    monkeypatch.setattr(raw, "render", render)
    vec = _vec(raw)
    try:
        evaluate_behavior(_Model(), vec, episode_seeds=[1, 2], output_dir=tmp_path, record_video=True)
        assert set(rendered_models) == {id(first_model), id(second_model)}
        np.testing.assert_allclose(first_model.vis.headlight.ambient, [0.11, 0.11, 0.11])
        np.testing.assert_allclose(second_model.vis.headlight.ambient, [0.22, 0.22, 0.22])
    finally:
        vec.close()


def test_real_map_render_uses_seeded_physics_grid_and_meter_axes(tmp_path):
    pytest.importorskip("matplotlib")
    raw = _RenderEnv()
    _, info = raw.reset(seed=42)
    snapshot = capture_terrain_snapshot(raw, info)
    result = write_terrain_maps(
        snapshot, np.array([[0.0, 0.0], [1.0, 0.5]]), tmp_path, episode_seed=42, terminated=False
    )
    assert result["height_units"] == "cm" and result["coordinate_units"] == "m"
    assert result["map_scales"]["full"]["bounds_m"] == [-6.0, 6.0, -6.0, 6.0]
    assert result["map_scales"]["full"]["color_min_cm"] == -result["map_scales"]["full"]["color_max_cm"]
    for path in result["files"].values():
        assert (tmp_path / path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_flat_plane_is_labelled_as_reference_and_bounds_expand_to_contain_path(tmp_path, fake_media):
    raw = _RenderEnv(horizon=4, terrain=False)
    vec = _vec(raw)
    try:
        with record_evaluation_replays(vec, tmp_path, fps=25) as recorder:
            recorder.arm_episode(0, 42)
            vec.seed(42)
            vec.reset()
            # A synthetic long path checks the infinite-plane plotting extent.
            recorder._path = [np.zeros(2), np.array([26.0, 0.0])]
            recorder._path_times = [0.0, 0.01]
            recorder.finish(truncated=True)
        replay = recorder.completed[0]
        assert Path(replay["full_map"]).name == "flat_plane_full_map.png"
        values = np.load(replay["raw_terrain_and_path"])
        assert values["normalized_heights"].size == 0
        assert values["x_m"][-1] >= 28
        manifest = json.loads(Path(replay["manifest"]).read_text())
        assert manifest["surface_kind"] == "flat_plane"
        assert not manifest["terrain"]["physical_heightfield"]
    finally:
        vec.close()


def test_video_frame_rate_cannot_exceed_control_frequency(tmp_path, fake_media):
    vec = _vec(_RenderEnv())
    try:
        with pytest.raises(ReplayExportError, match="control frequency"):
            with record_evaluation_replays(vec, tmp_path, fps=101):
                pass
    finally:
        vec.close()


class _SamplingRenderEnv(_RenderEnv):
    terrain_families = ("flat", "sloped", "bumps", "depressions", "mixed")

    def __init__(self):
        super().__init__(horizon=4)
        self._field_model = self.model
        self._flat_model = _RenderEnv(terrain=False).model

    def reset(self, *, seed=None, options=None):
        family = (options or {}).get("terrain_family", "mixed")
        self.model = self._flat_model if family == "flat" else self._field_model
        self.data = mujoco.MjData(self.model)
        self.config = replace(self.config, template="sloped" if family == "flat" else family)
        self.terrain = None if family == "flat" else generate_terrain(self.config, run_seed=0)
        observation, info = super().reset(seed=seed, options=options)
        info["terrain_sampling"] = {"family": family, "mode": "forced" if options else "balanced_shuffle"}
        return observation, info


def test_sampler_replays_keep_each_selected_family_and_its_physics_map_before_autoreset(tmp_path, fake_media):
    raw = _SamplingRenderEnv()
    vec = _vec(raw)
    seeds = [11, 22, 33, 44, 55]
    try:
        report = evaluate_behavior(_Model(), vec, episode_seeds=seeds, output_dir=tmp_path, record_video=True)
        index = json.loads((tmp_path / "replays/index.json").read_text())
        assert [entry["terrain_family"] for entry in index["episodes"]] == list(raw.terrain_families)
        for family, seed, episode in zip(raw.terrain_families, seeds, report["episodes"], strict=True):
            replay = episode["replay"]
            manifest = json.loads(Path(replay["manifest"]).read_text())
            assert manifest["terrain_family"] == replay["terrain_family"] == episode["terrain_family"] == family
            assert manifest["reset_info"]["terrain_sampling"]["family"] == family
            assert manifest["reset_info"]["episode_seed"] == seed
            data = np.load(replay["raw_terrain_and_path"])
            if family == "flat":
                assert manifest["surface_kind"] == "flat_plane"
                assert data["normalized_heights"].size == 0
            else:
                config = replace(raw.config, template=family)
                expected = generate_terrain(config, run_seed=seed)
                np.testing.assert_array_equal(data["normalized_heights"], expected.normalized_heights)
                assert manifest["terrain"]["config"].get("template", "sloped") == family
            assert Path(replay["full_map"]).is_file()
            assert Path(replay["local_map"]).is_file()
        assert raw.terrain.config.template == "mixed"
        assert raw._seed == 1055
    finally:
        vec.close()


def test_replay_rejects_sampler_label_that_disagrees_with_recorded_physics(tmp_path, fake_media, monkeypatch):
    raw = _RenderEnv(horizon=4, terrain=False)
    original_reset = raw.reset

    def reset(**kwargs):
        observation, info = original_reset(**kwargs)
        info["terrain_sampling"] = {"family": "bumps"}
        return observation, info

    monkeypatch.setattr(raw, "reset", reset)
    vec = _vec(raw)
    try:
        with pytest.raises(ValueError, match="disagrees with the reset surface"):
            evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path, record_video=True)
        assert not list((tmp_path / "replays").glob("episode_*"))
        assert vec.training and vec.norm_reward
    finally:
        vec.close()
