"""The report-only heading probe (decision D-D27): spawns turned about vertical.

Both physics-r8 trex stances fell when spawned 90 degrees off where the
statue stands; the stance stage always spawns facing +x, so no gate input
can see it.  The probe turns the spawn after ``reset`` -- the animal alone,
or with the prey and the episode's reference direction -- and these tests
pin that the turn is exactly a rotation about vertical through the root and
that it changes nothing else.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from environments.shared.config import load_stage_config
from environments.shared.curriculum.gate_schema import _DIAGNOSTIC_KEYS
from environments.shared.reporting.stance_report import (
    SpawnYaw,
    _apply_spawn_yaw,
    _root_yaw,
    build_stance_gate_report,
    probe_stem,
    render_heading_probe,
    spawn_yaw_variants,
    write_heading_probe,
)
from environments.trex.envs.trex_env import TRexEnv


@pytest.fixture
def env():
    e = TRexEnv(prey_distance_range=(10.0, 15.0), reset_noise_scale=0.05)
    yield e
    e.close()


def _obs_slices(env) -> dict[str, slice]:
    """Where TRexEnv._get_obs puts the world-frame channels: after the joint positions and velocities."""
    start = (env.model.nq - 7) + (env.model.nv - 6)
    return {
        "pelvis_quat": slice(start, start + 4),
        "pelvis_linvel": slice(start + 7, start + 10),
        "prey_direction": slice(start + 15, start + 18),
    }


def _turned_quat(yaw_deg: float, quat: np.ndarray) -> np.ndarray:
    half = math.radians(yaw_deg) / 2.0
    turned = np.empty(4)
    mujoco.mju_mulQuat(turned, np.array([math.cos(half), 0.0, 0.0, math.sin(half)]), np.asarray(quat, dtype=np.float64))
    return turned


def _turned_xy(yaw_deg: float, vector: np.ndarray) -> np.ndarray:
    theta = math.radians(yaw_deg)
    x, y, z = (float(value) for value in vector)
    return np.array([math.cos(theta) * x - math.sin(theta) * y, math.sin(theta) * x + math.cos(theta) * y, z])


class TestTurn:
    @pytest.mark.parametrize("yaw_deg", [-90.0, 45.0])
    def test_the_animal_turns_about_vertical_and_nothing_else_moves(self, env, yaw_deg):
        env.reset(seed=3042)
        joints = env.data.qpos[7:].copy()
        root = env.data.qpos[0:3].copy()
        heights = env.data.xpos[:, 2].copy()
        prey = env.data.mocap_pos[0].copy()
        yaw_before = _root_yaw(env)
        slices = _obs_slices(env)
        obs_before = env._get_obs()
        obs = _apply_spawn_yaw(env, SpawnYaw(yaw_deg))
        assert math.degrees(math.remainder(_root_yaw(env) - yaw_before, 2 * math.pi)) == pytest.approx(yaw_deg)
        np.testing.assert_array_equal(env.data.qpos[7:], joints)
        np.testing.assert_array_equal(env.data.qpos[0:3], root)
        # A rotation about vertical moves no height: the spawn stays on the floor exactly as it was.
        np.testing.assert_allclose(env.data.xpos[:, 2], heights, atol=1e-12)
        np.testing.assert_array_equal(env.data.mocap_pos[0], prey)
        np.testing.assert_array_equal(obs, env._get_obs())
        # The observation reads the TURNED state (the derived sensors recomputed, not the reset's): the
        # pelvis quaternion and the world-frame velocity turn, and the prey, left behind, does not.
        expected_quat = _turned_quat(yaw_deg, obs_before[slices["pelvis_quat"]])
        np.testing.assert_allclose(obs[slices["pelvis_quat"]], expected_quat, atol=1e-6)
        np.testing.assert_allclose(
            obs[slices["pelvis_linvel"]], _turned_xy(yaw_deg, obs_before[slices["pelvis_linvel"]]), atol=1e-6
        )
        np.testing.assert_allclose(obs[slices["prey_direction"]], obs_before[slices["prey_direction"]], atol=1e-6)
        # The turned state was posed out of band, so the step-tagged substep aggregates are dropped.
        assert env._substep_contact_step == -1

    def test_the_whole_scene_turns_the_prey_and_the_reference_direction_with_it(self, env):
        env.reset(seed=3042)
        root_xy = env.data.qpos[0:2].copy()
        prey_rel = env.data.mocap_pos[0, :2] - root_xy
        reference = env._initial_prey_dir_2d.copy()
        slices = _obs_slices(env)
        obs_before = env._get_obs()
        obs = _apply_spawn_yaw(env, SpawnYaw(90.0, whole_scene=True))
        np.testing.assert_allclose(env.data.mocap_pos[0, :2] - root_xy, [-prey_rel[1], prey_rel[0]], atol=1e-12)
        np.testing.assert_allclose(env._initial_prey_dir_2d, [-reference[1], reference[0]], atol=1e-12)
        # The observation's prey direction turns with the scene; the quaternion with the animal.
        np.testing.assert_allclose(
            obs[slices["prey_direction"]], _turned_xy(90.0, obs_before[slices["prey_direction"]]), atol=1e-6
        )
        np.testing.assert_allclose(
            obs[slices["pelvis_quat"]], _turned_quat(90.0, obs_before[slices["pelvis_quat"]]), atol=1e-6
        )

    def test_the_statue_scores_the_same_episode_from_a_turned_scene(self):
        """Physics and reward are rotation-invariant for a constant command when the prey turns too."""
        stage_config = load_stage_config("trex", "stance")

        def roll(spawn_yaw):
            report = build_stance_gate_report(
                "trex", 1, stage_config=stage_config, zero_action=True, episodes=1, seed=3042, spawn_yaw=spawn_yaw
            )
            return report

        straight = roll(SpawnYaw(0.0))
        turned = roll(SpawnYaw(90.0, whole_scene=True))
        assert turned["metrics"]["episode_length_mean"] == straight["metrics"]["episode_length_mean"] == 1000
        assert turned["metrics"]["reward_mean"] == pytest.approx(straight["metrics"]["reward_mean"], rel=1e-6)
        assert turned["episode_yaw_change_deg"][0] == pytest.approx(straight["episode_yaw_change_deg"][0], abs=1e-3)
        assert probe_stem(turned) == "stance_gate_probe_heading"
        assert turned["spawn_yaw"] == {"kind": "spawn_yaw/v1", "yaw_deg": 90.0, "whole_scene": True}


class TestYawChange:
    def test_a_turn_past_180_degrees_is_accumulated_not_folded_back(self):
        """Three 100-degree steps are a 300-degree turn; read off the wrapped yaw they would be 60."""
        from environments.shared.reporting.stance_report import _roll_episodes

        class _Spinning:
            """Turns its root 100 degrees about vertical on every step, for three steps."""

            def __init__(self) -> None:
                self.unwrapped = self
                self.data = type("Data", (), {})()
                self.yaw_deg = 0.0

            def _pose(self) -> None:
                half = math.radians(self.yaw_deg) / 2.0
                self.data.qpos = np.array([0.0, 0.0, 1.0, math.cos(half), 0.0, 0.0, math.sin(half)])

            def reset(self, seed=None):
                self.yaw_deg = 0.0
                self._pose()
                return np.zeros(1), {}

            def step(self, action):
                self.yaw_deg += 100.0
                self._pose()
                return np.zeros(1), 1.0, False, self.yaw_deg >= 300.0, {}

        yaw_changes: list[float] = []
        _roll_episodes(
            _Spinning(),
            predict=lambda obs: np.zeros(1),
            episodes=1,
            seed=0,
            settle_steps=10,
            lengths=[],
            rewards=[],
            duties=[],
            bilateral_duties=[],
            single_duties=[],
            terminations={},
            components={},
            action_stats={},
            yaw_changes=yaw_changes,
        )
        assert yaw_changes == [pytest.approx(300.0)]


class TestProbeTable:
    def test_the_rows_are_the_control_then_each_offset_both_ways(self):
        variants = spawn_yaw_variants([90.0, -90.0, 0.0, 45.0])
        assert variants[0] == SpawnYaw(0.0)
        assert [(v.yaw_deg, v.whole_scene) for v in variants[1:]] == [
            (-90.0, False),
            (-90.0, True),
            (45.0, False),
            (45.0, True),
            (90.0, False),
            (90.0, True),
        ]

    def test_render_and_write(self, tmp_path):
        def report(yaw_deg, whole_scene, length, reward):
            return {
                "spawn_yaw": SpawnYaw(yaw_deg, whole_scene).as_dict(),
                "metrics": {
                    "full_horizon_fraction": length / 1000,
                    "reward_mean": reward,
                    "episode_length_mean": length,
                },
                "episode_yaw_change_deg": [3.0, 5.0],
                "terminations": {},
                "horizon": 1000,
                "policy": "robust_best_model.zip",
                "seed": 3042,
            }

        policy = [report(0.0, False, 1000, 3770.0), report(90.0, False, 200, 400.0), report(90.0, True, 200, 410.0)]
        statue = [report(0.0, False, 1000, 3765.0), report(90.0, False, 1000, 3680.0), report(90.0, True, 1000, 3765.0)]
        text, payload = render_heading_probe(policy, statue, probe_episodes=2)
        assert "REPORT ONLY" in payload["note"] and "not a gate result" in text
        assert [row["whole_scene"] for row in payload["rows"]] == [False, False, True]
        assert payload["rows"][1]["policy"]["full_horizon_fraction"] == 0.2
        written = write_heading_probe(tmp_path, policy, statue, probe_episodes=2)
        assert written["heading_probe_txt"].name == "stance_heading_probe.txt"
        assert written["heading_probe_json"].exists()

    def test_the_probe_key_is_a_diagnostic_declared_by_trex_stance(self):
        assert "stance_probe_spawn_yaw_deg" in _DIAGNOSTIC_KEYS
        curriculum = load_stage_config("trex", "stance")["curriculum_kwargs"]
        assert curriculum["stance_probe_spawn_yaw_deg"] == [-90.0, -45.0, 45.0, 90.0]
