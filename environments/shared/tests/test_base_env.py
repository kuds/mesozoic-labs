import contextlib
import math
import sys
import types

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.shared import reward_functions as rf
from environments.shared.base_env import BaseDinoEnv
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv


def test_base_env_lifecycle():
    env = RaptorEnv(render_mode="rgb_array")

    # Test reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    # Test step
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)

    # Test render (may fail in headless environments without GPU/OpenGL)
    try:
        frame = env.render()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
    except Exception:
        # Rendering requires a valid OpenGL context which may not be
        # available in CI or headless environments.
        pass

    # Test close
    env.close()


def test_human_render_loads_the_viewer_itself(monkeypatch):
    """`import mujoco` does not load `mujoco.viewer`, so human rendering must.

    The submodule is removed from `mujoco` and a fake stands in for it in
    `sys.modules`: code that reaches `mujoco.viewer` as an attribute fails as
    the eval CLI did, and the lazy import finds the fake instead of opening a
    window. The fake camera records every attribute assigned on it while the
    viewer's lock is not held, since the real viewer's render thread reads the
    camera on every frame.
    """
    launches = []

    class LockCheckedCamera:
        def __init__(self, viewer):
            object.__setattr__(self, "_viewer", viewer)
            object.__setattr__(self, "_cam", mujoco.MjvCamera())
            object.__setattr__(self, "unlocked_writes", [])

        def __getattr__(self, name):
            return getattr(self._cam, name)

        def __setattr__(self, name, value):
            if not self._viewer.locked:
                self.unlocked_writes.append(name)
            setattr(self._cam, name, value)

    class FakeViewer:
        def __init__(self):
            self.locked = False
            self.cam = LockCheckedCamera(self)
            self.syncs = 0
            self.closed = False

        @contextlib.contextmanager
        def lock(self):
            self.locked = True
            try:
                yield
            finally:
                self.locked = False

        def sync(self):
            self.syncs += 1

        def close(self):
            self.closed = True

    viewer = FakeViewer()

    def launch_passive(model, data):
        launches.append((model, data))
        return viewer

    monkeypatch.delattr(mujoco, "viewer", raising=False)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", types.SimpleNamespace(launch_passive=launch_passive))

    env = RaptorEnv(render_mode="human")
    env.reset(seed=0)
    for _ in range(2):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    assert launches == [(env.model, env.data)], "the viewer launches once, on the first step"
    assert viewer.syncs == 2
    expected = env._make_camera()
    for field in ("type", "trackbodyid", "distance", "azimuth", "elevation"):
        assert getattr(viewer.cam, field) == getattr(expected, field), field
    assert viewer.cam.unlocked_writes == [], "the camera is aimed under the viewer's lock"
    env.close()
    assert viewer.closed and env._viewer is None


def test_rgb_array_render_does_not_depend_on_the_human_branch(monkeypatch):
    """Guards the lazy viewer import: a bare `import mujoco.viewer` inside
    `render` would make `mujoco` local to it and break this branch."""

    class FakeRenderer:
        def __init__(self, model, height, width):
            self.shape = (height, width, 3)

        def update_scene(self, data, camera):
            self.camera = camera

        def render(self):
            return np.zeros(self.shape, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    env = RaptorEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (480, 640, 3)
    env.close()


def test_base_env_scale_action():
    env = RaptorEnv()

    # Test action scaling
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    scaled = env._scale_action(action)
    assert scaled.shape == action.shape

    env.close()


# ── _quat_to_tilt ────────────────────────────────────────────────────────


class TestQuatToTilt:
    """Test the quaternion-to-tilt-angle static method on BaseDinoEnv."""

    def test_upright_quaternion_is_zero_tilt(self):
        """Identity quaternion (w=1, x=0, y=0, z=0) means perfectly upright."""
        tilt = BaseDinoEnv._quat_to_tilt(np.array([1.0, 0.0, 0.0, 0.0]))
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_pitch(self):
        """Quaternion representing 90-degree pitch forward around Y axis."""
        # q = [cos(45°), 0, sin(45°), 0] = [0.7071, 0, 0.7071, 0]
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi / 2, abs=0.01)

    def test_90_degree_roll(self):
        """Quaternion representing 90-degree roll around X axis."""
        angle = np.pi / 2
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi / 2, abs=0.01)

    def test_upside_down(self):
        """180-degree flip: body Z-axis points downward."""
        # Rotation of pi around X axis: q = [0, 1, 0, 0]
        quat = np.array([0.0, 1.0, 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(np.pi, abs=0.01)

    def test_small_tilt(self):
        """A small tilt angle should be close to the rotation angle."""
        angle = 0.1  # ~5.7 degrees
        quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(angle, abs=0.01)

    def test_yaw_only_is_zero_tilt(self):
        """Pure yaw rotation (around Z) should produce zero tilt."""
        angle = np.pi / 4
        quat = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        tilt = BaseDinoEnv._quat_to_tilt(quat)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_return_type_is_float(self):
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        assert isinstance(BaseDinoEnv._quat_to_tilt(quat), float)


# ── set_reward_weight ─────────────────────────────────────────────────────


class TestSetRewardWeight:
    """Test dynamic reward weight mutation via set_reward_weight()."""

    @pytest.fixture
    def env(self):
        e = RaptorEnv()
        yield e
        e.close()

    def test_set_existing_weight(self, env):
        env.set_reward_weight("forward_vel_weight", 2.5)
        assert env.forward_vel_weight == 2.5

    def test_set_alive_bonus(self, env):
        env.set_reward_weight("alive_bonus", 0.0)
        assert env.alive_bonus == 0.0

    def test_nonexistent_attribute_raises(self, env):
        with pytest.raises(AttributeError, match="has no attribute"):
            env.set_reward_weight("nonexistent_weight", 1.0)

    def test_set_to_zero(self, env):
        env.set_reward_weight("energy_penalty_weight", 0.0)
        assert env.energy_penalty_weight == 0.0

    def test_set_negative_value(self, env):
        env.set_reward_weight("forward_vel_weight", -1.0)
        assert env.forward_vel_weight == -1.0


# ── truncation ────────────────────────────────────────────────────────────


class TestTruncation:
    """Test episode truncation at max steps."""

    def test_truncates_at_max_steps(self):
        env = RaptorEnv(max_episode_steps=5)
        env.reset(seed=42)
        for i in range(5):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated:
                break
        # If not already terminated, the 5th step should truncate
        if not terminated:
            assert truncated
        env.close()

    def test_not_truncated_before_max_steps(self):
        env = RaptorEnv(max_episode_steps=1000)
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, truncated, _ = env.step(action)
        assert not truncated
        env.close()


# ── action scaling ────────────────────────────────────────────────────────


class TestActionScaling:
    """Test ``BaseDinoEnv``'s shared midpoint action mapping.

    This used to exercise the mapping through ``BrachioEnv``, which was the
    last species inheriting it.  Brachiosaurus moved to the home-keyframe
    residual mapping (plant_versions note 7), so **no species overrides
    ``_scale_action`` with the base implementation any more** and there is no
    env to reach it through.  The base implementation is still live code --
    it is what a newly added species gets before it authors a home keyframe,
    and it is the mapping the plant contract records as
    ``clip[-1,1]-then-affine-to-ordered-ctrlrange/v1`` -- so it is called
    explicitly here rather than deleted along with its last caller.
    """

    @pytest.fixture
    def env(self):
        e = BrachioEnv()
        yield e
        e.close()

    @staticmethod
    def _base_scale(env, action):
        """Invoke the base mapping, bypassing the species override."""
        return BaseDinoEnv._scale_action(env, action)

    def test_zero_action_maps_to_midpoint(self, env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        scaled = self._base_scale(env, action)
        ctrl_range = env.model.actuator_ctrlrange
        midpoint = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2
        np.testing.assert_allclose(scaled, midpoint, atol=1e-6)

    def test_plus_one_maps_to_max(self, env):
        action = np.ones(env.action_space.shape, dtype=np.float32)
        scaled = self._base_scale(env, action)
        ctrl_max = env.model.actuator_ctrlrange[:, 1]
        np.testing.assert_allclose(scaled, ctrl_max, atol=1e-6)

    def test_minus_one_maps_to_min(self, env):
        action = -np.ones(env.action_space.shape, dtype=np.float32)
        scaled = self._base_scale(env, action)
        ctrl_min = env.model.actuator_ctrlrange[:, 0]
        np.testing.assert_allclose(scaled, ctrl_min, atol=1e-6)

    def test_species_override_preserves_the_endpoints(self, env):
        """The residual override must keep -1/+1 at the actuator limits.

        Only the ZERO point moves between the two mappings; if an override
        ever narrowed the reachable control range, the policy would silently
        lose authority at the extremes.
        """
        ctrl_range = env.model.actuator_ctrlrange
        for action, expected in (
            (np.ones(env.action_space.shape, dtype=np.float32), ctrl_range[:, 1]),
            (-np.ones(env.action_space.shape, dtype=np.float32), ctrl_range[:, 0]),
        ):
            np.testing.assert_allclose(env._scale_action(action), expected, atol=1e-6)


# ── distance tracking ────────────────────────────────────────────────────


class TestDistanceTracking:
    """Test cumulative XY distance traveled tracking."""

    @pytest.fixture
    def env(self):
        e = RaptorEnv(max_episode_steps=100)
        yield e
        e.close()

    def test_distance_starts_at_zero(self, env):
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # First step distance should be very small (near zero with zero action)
        assert "distance_traveled" in info
        assert info["distance_traveled"] >= 0.0

    def test_distance_is_non_negative(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        for _ in range(10):
            _, _, terminated, _, info = env.step(action)
            assert info["distance_traveled"] >= 0.0
            if terminated:
                break

    def test_distance_is_monotonically_increasing(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        prev_dist = 0.0
        for _ in range(10):
            _, _, terminated, _, info = env.step(action)
            assert info["distance_traveled"] >= prev_dist
            prev_dist = info["distance_traveled"]
            if terminated:
                break

    def test_distance_resets_on_reset(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        for _ in range(5):
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # After reset, distance should be near zero again
        assert info["distance_traveled"] < 0.1


# ── gait symmetry ────────────────────────────────────────────────────────


class TestGaitSymmetry:
    """Test the shared gait symmetry helper on BaseDinoEnv."""

    @pytest.fixture
    def env(self):
        e = RaptorEnv(gait_symmetry_weight=1.0)
        e.reset(seed=42)
        yield e
        e.close()

    def test_no_touchdowns_gives_zero(self, env):
        """With no foot contacts, alternation ratio should be 0."""
        reward, ratio = env._compute_gait_symmetry(0.0, 0.0, 1.0)
        assert ratio == 0.0
        assert reward == 0.0

    def test_single_touchdown_gives_zero(self, env):
        """A single touchdown cannot produce alternation."""
        # First call: no contact
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Second call: right foot lands (off→on)
        reward, ratio = env._compute_gait_symmetry(1.0, 0.0, 1.0)
        assert ratio == 0.0

    def test_perfect_alternation(self, env):
        """L→R→L should produce alternation_ratio = 1.0."""
        # Start: no contact
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Left touchdown
        env._compute_gait_symmetry(0.0, 1.0, 1.0)
        # Left lifts, right lands
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        # Right lifts, left lands
        reward, ratio = env._compute_gait_symmetry(0.0, 1.0, 1.0)
        assert ratio == pytest.approx(1.0)
        assert reward == pytest.approx(1.0)

    def test_same_foot_repeated(self, env):
        """R→R→R (same foot) should produce alternation_ratio = 0.0."""
        # Start: no contact
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Right touchdown
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        # Right lifts
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Right lands again
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        # Right lifts
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        # Right lands again
        reward, ratio = env._compute_gait_symmetry(1.0, 0.0, 1.0)
        assert ratio == pytest.approx(0.0)
        assert reward == pytest.approx(0.0)

    def test_weight_scales_reward(self, env):
        """Reward should scale linearly with weight."""
        # Perfect alternation
        env._compute_gait_symmetry(0.0, 0.0, 1.0)
        env._compute_gait_symmetry(0.0, 1.0, 1.0)
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        _, ratio = env._compute_gait_symmetry(0.0, 1.0, 1.0)
        # Reset and redo with weight=0.5
        env._reset_gait_state()
        env._compute_gait_symmetry(0.0, 0.0, 0.5)
        env._compute_gait_symmetry(0.0, 1.0, 0.5)
        env._compute_gait_symmetry(1.0, 0.0, 0.5)
        reward, _ = env._compute_gait_symmetry(0.0, 1.0, 0.5)
        assert reward == pytest.approx(0.5 * ratio)

    def test_reset_clears_state(self, env):
        """After _reset_gait_state, history should be empty."""
        env._compute_gait_symmetry(0.0, 1.0, 1.0)
        env._compute_gait_symmetry(1.0, 0.0, 1.0)
        env._reset_gait_state()
        reward, ratio = env._compute_gait_symmetry(0.0, 0.0, 1.0)
        assert ratio == 0.0

    def test_trex_uses_shared_helper(self):
        """TRexEnv should use the shared gait symmetry (not the old buggy version)."""
        from environments.trex.envs.trex_env import TRexEnv

        env = TRexEnv(gait_symmetry_weight=1.0)
        env.reset(seed=42)
        # Verify gait state was initialised
        assert hasattr(env, "_touchdown_sequence")
        assert env._touchdown_sequence == []
        env.close()


class TestQuadrupedGaitSymmetry:
    """Test the quadrupedal diagonal pair gait symmetry helper."""

    @pytest.fixture
    def env(self):
        from environments.brachiosaurus.envs.brachio_env import BrachioEnv

        e = BrachioEnv(gait_symmetry_weight=1.0)
        e.reset(seed=42)
        yield e
        e.close()

    def test_no_touchdowns_gives_zero(self, env):
        """With no foot contacts, alternation ratio should be 0."""
        reward, ratio = env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        assert ratio == 0.0
        assert reward == 0.0

    def test_single_diagonal_touchdown_gives_zero(self, env):
        """A single diagonal touchdown cannot produce alternation."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A (FR + RL) lands
        reward, ratio = env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        assert ratio == 0.0

    def test_perfect_diagonal_alternation(self, env):
        """A→B→A should produce alternation_ratio = 1.0."""
        # Start: no contact
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A: FR + RL land
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        # All off
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal B: FL + RR land
        env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 1.0)
        # All off
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A again
        reward, ratio = env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        assert ratio == pytest.approx(1.0)
        assert reward == pytest.approx(1.0)

    def test_same_diagonal_repeated(self, env):
        """A→A→A (same diagonal) should produce alternation_ratio = 0.0."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A lands
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A again
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Diagonal A again
        reward, ratio = env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        assert ratio == pytest.approx(0.0)
        assert reward == pytest.approx(0.0)

    def test_partial_diagonal_triggers_pair(self, env):
        """A single foot from a diagonal pair should trigger that pair."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Only FR lands (partial diagonal A)
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 0.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        # Only FL lands (partial diagonal B)
        reward, ratio = env._compute_quadruped_gait_symmetry(0.0, 1.0, 0.0, 0.0, 1.0)
        assert ratio == pytest.approx(1.0)

    def test_weight_scales_reward(self, env):
        """Reward should scale linearly with weight."""
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        _, ratio = env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 1.0)
        # Reset and redo with weight=0.5
        env._reset_quadruped_gait_state()
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 0.5)
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 0.5)
        env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 0.5)
        reward, _ = env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 0.5)
        assert reward == pytest.approx(0.5 * ratio)

    def test_reset_clears_state(self, env):
        """After _reset_quadruped_gait_state, history should be empty."""
        env._compute_quadruped_gait_symmetry(1.0, 0.0, 0.0, 1.0, 1.0)
        env._compute_quadruped_gait_symmetry(0.0, 1.0, 1.0, 0.0, 1.0)
        env._reset_quadruped_gait_state()
        reward, ratio = env._compute_quadruped_gait_symmetry(0.0, 0.0, 0.0, 0.0, 1.0)
        assert ratio == 0.0

    def test_brachio_uses_quadruped_helper(self, env):
        """BrachioEnv should use quadrupedal gait state."""
        assert hasattr(env, "_quad_touchdown_sequence")
        assert env._quad_touchdown_sequence == []

    def test_brachio_info_has_all_foot_contacts(self, env):
        """BrachioEnv step info should include all 4 foot contacts."""
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "r_foot_contact" in info
        assert "l_foot_contact" in info
        assert "rr_foot_contact" in info
        assert "rl_foot_contact" in info


class TestIsSuccessInfo:
    """info["is_success"] is reported at episode end (SB3 EvalCallback
    consumes it to record per-episode success rates in evaluations.npz)."""

    def test_truncated_episode_reports_is_success_false(self):
        env = RaptorEnv(max_episode_steps=3)
        try:
            env.reset(seed=42)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(3):
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            assert terminated or truncated
            assert info["is_success"] is False
        finally:
            env.close()

    def test_mid_episode_steps_omit_is_success(self):
        env = RaptorEnv(max_episode_steps=1000)
        try:
            env.reset(seed=42)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            if not (terminated or truncated):
                assert "is_success" not in info
        finally:
            env.close()


class TestScaleActionClipping:
    """_scale_action clips out-of-range actions before mapping to ctrl."""

    def test_out_of_range_action_clipped_to_ctrl_range(self):
        env = RaptorEnv()
        try:
            big = np.full(env.action_space.shape, 5.0, dtype=np.float32)
            scaled = env._scale_action(big)
            ctrl_max = env.model.actuator_ctrlrange[:, 1]
            np.testing.assert_allclose(scaled, ctrl_max, atol=1e-6)
        finally:
            env.close()


class TestResetHeightTruncation:
    """Reset must never generate an already-terminal state.

    The root-height jitter was the only unbounded term in the reset — every
    other one is a bounded uniform — so an unbounded Gaussian eventually placed
    the root outside ``healthy_z_range`` before the policy acted.  An episode
    that ends on step 1 whatever the action is not a policy failure, and
    counting it as one puts an unreachable ceiling on any reliability gate.

    SUPERSEDED in effect by ground settling, which overwrites the root height
    from the sampled joint pose; the bound and its draw survive only for
    RNG-stream compatibility (see TestHeightJitterIsInertSinceGroundSettling).
    These tests keep pinning the function's arithmetic until the height
    channel is removed outright.
    """

    def test_delta_is_bounded_by_distance_to_the_floor(self):
        env = RaptorEnv()
        try:
            low, high = env.healthy_z_range
            base_z = 0.5 * (low + high)
            headroom = min(base_z - low, high - base_z)
            huge = 10.0 * (high - low)

            assert env._bounded_reset_height_delta(base_z, huge) < headroom
            assert env._bounded_reset_height_delta(base_z, -huge) > -headroom
        finally:
            env.close()

    def test_bound_is_symmetric_so_the_mean_spawn_height_is_unchanged(self):
        env = RaptorEnv()
        try:
            low, high = env.healthy_z_range
            base_z = 0.5 * (low + high)
            for delta in (0.05, 0.2, 5.0):
                up = env._bounded_reset_height_delta(base_z, delta)
                down = env._bounded_reset_height_delta(base_z, -delta)
                assert up == pytest.approx(-down)
        finally:
            env.close()

    def test_small_deltas_pass_through_untouched(self):
        """Species with ample headroom must be unaffected in practice."""
        env = RaptorEnv()
        try:
            low, high = env.healthy_z_range
            base_z = 0.5 * (low + high)
            tiny = 1e-4
            assert env._bounded_reset_height_delta(base_z, tiny) == pytest.approx(tiny)
        finally:
            env.close()

    def test_no_headroom_pins_the_spawn_to_the_keyframe(self):
        """A base height already at the floor yields a deterministic spawn."""
        env = RaptorEnv()
        try:
            low, _ = env.healthy_z_range
            assert env._bounded_reset_height_delta(low, 1.0) == pytest.approx(0.0)
            assert env._bounded_reset_height_delta(low, -1.0) == pytest.approx(0.0)
        finally:
            env.close()

    @pytest.mark.parametrize("env_cls", [RaptorEnv, BrachioEnv])
    def test_reset_never_spawns_outside_the_healthy_range(self, env_cls):
        env = env_cls(reset_noise_scale=0.5)
        try:
            low, high = env.healthy_z_range
            for seed in range(200):
                env.reset(seed=seed)
                root_z = float(env.data.qpos[2])
                assert low < root_z < high, f"seed {seed} spawned at {root_z:.4f}, outside [{low}, {high}]"
        finally:
            env.close()

    def test_zero_noise_still_gives_a_deterministic_reset(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=1)
            first = env.data.qpos.copy()
            env.reset(seed=999)
            np.testing.assert_allclose(env.data.qpos, first)
        finally:
            env.close()


class TestHeightJitterIsInertSinceGroundSettling:
    """The reset's root-height draw must stay drawn, and must stay inert.

    ``_settle_root_on_ground`` overwrites the root height as a pure function
    of the sampled joint pose, so ``reset_height_noise_scale`` no longer
    reaches the post-reset state.  The draw is deliberately KEPT so the reset
    consumes the same RNG sequence — removing it would shift every subsequent
    draw and re-anchor all seeded baselines.  These tests pin both halves of
    that contract: the knob changes nothing, and the stream stays aligned.
    """

    def test_height_scale_does_not_change_the_post_reset_state(self):
        from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv

        quiet = DibothrosuchusEnv(reset_noise_scale=0.30, reset_height_noise_scale=0.0)
        loud = DibothrosuchusEnv(reset_noise_scale=0.30, reset_height_noise_scale=0.5)
        try:
            for seed in (0, 17, 41):
                quiet.reset(seed=seed)
                loud.reset(seed=seed)
                # Identical up to the one-ULP roundoff of settling from a
                # different pre-settle height; anything larger means the knob
                # has grown a real effect again.
                np.testing.assert_allclose(quiet.data.qpos, loud.data.qpos, rtol=0, atol=1e-12)
                np.testing.assert_allclose(quiet.data.qvel, loud.data.qvel, rtol=0, atol=0)
        finally:
            quiet.close()
            loud.close()

    def test_height_scale_does_not_desync_the_rng_stream(self):
        from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv

        quiet = DibothrosuchusEnv(reset_noise_scale=0.30, reset_height_noise_scale=0.0)
        loud = DibothrosuchusEnv(reset_noise_scale=0.30, reset_height_noise_scale=0.5)
        try:
            quiet.reset(seed=3)
            loud.reset(seed=3)
            assert quiet.np_random.bit_generator.state == loud.np_random.bit_generator.state
        finally:
            quiet.close()
            loud.close()


class TestLowestGroundClearanceDataArgument:
    """The probe must measure the ``MjData`` it is handed, not ``self.data``.

    The parameter used to be accepted and silently ignored — the probe always
    read ``self.data``, and ``home_ground_clearance`` had to swap ``self.data``
    out to use a scratch buffer.  A caller passing a scratch pose would get
    the live pose's clearance with no error.
    """

    def test_probe_reads_the_passed_data(self):
        import mujoco

        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            mujoco.mj_forward(env.model, env.data)
            settled = env.lowest_ground_clearance()

            # Home pose lowered by 0.1 m: over a plane floor the clearance
            # must drop by exactly the shift.
            buried = mujoco.MjData(env.model)
            mujoco.mj_resetDataKeyframe(env.model, buried, int(getattr(env, "_reset_keyframe_id", 0)))
            buried.qpos[2] -= 0.1
            mujoco.mj_forward(env.model, buried)
            probed = env.lowest_ground_clearance(buried)

            assert probed == pytest.approx(env.home_ground_clearance() - 0.1, abs=1e-9), (
                f"clearance({probed:.4f}) should reflect the buried scratch pose, "
                f"not the settled live pose ({settled:.4f})"
            )
            # And the live probe is unaffected by having probed other data.
            assert env.lowest_ground_clearance() == settled
        finally:
            env.close()

    def test_home_clearance_no_longer_depends_on_live_state(self):
        env = RaptorEnv(reset_noise_scale=0.1)
        try:
            env.reset(seed=5)
            home_after_reset = env.home_ground_clearance()
            fresh = RaptorEnv(reset_noise_scale=0.1)
            try:
                assert home_after_reset == fresh.home_ground_clearance()
            finally:
                fresh.close()
        finally:
            env.close()


class TestRewardTermHelpers:
    """The reward-term helpers' contract with the species' reward chains.

    ``_progress_terms``, ``_nosedive_term``, ``_heading_terms`` and ``_speed_terms`` record their
    info keys in a fixed order, pass the pure functions' NumPy scalars through unconverted and
    return their terms unsummed, in the order every dual species destructures them into its
    left-associated ``+`` chain.  The digest golden rounds rewards and gives float and np.float64
    one shape, so it sees neither a swapped pair (heading and lateral sit next to each other in
    every chain, and swapping them moves some step rewards by an ulp) nor a ``float()`` wrap.

    The species defaults leave most of these weights at 0.0, which makes the terms -0.0 and a swap
    invisible, so every weight is set nonzero and the speed threshold is put under the idle one.
    The state then makes every term nonzero and distinct: a planar velocity of speed 0.03 pointing
    backward along the reference, a body yawed off the reference and pitched nose-down, and a root
    displaced from its spawn point.
    """

    WEIGHTS = {
        "forward_vel_weight": 1.5,
        "forward_vel_max": 2.0,
        "backward_vel_penalty_weight": 0.7,
        "drift_penalty_weight": 0.4,
        "nosedive_weight": 0.9,
        "heading_weight": 0.3,
        "lateral_penalty_weight": 0.6,
        "speed_penalty_weight": 0.5,
        "speed_penalty_threshold": 0.01,
        "idle_penalty_weight": 0.35,
        "idle_velocity_threshold": 0.05,
    }

    @staticmethod
    def _assert_recorded(info, expected):
        assert list(info) == [key for key, _ in expected]
        for key, value in expected:
            assert type(value) is np.float64, key
            assert type(info[key]) is np.float64, f"{key} was converted to {type(info[key]).__name__}"
            assert info[key] == value, key

    @pytest.mark.parametrize("env_cls", [TRexEnv, RaptorEnv, BrachioEnv, DibothrosuchusEnv])
    def test_terms_are_recorded_unconverted_and_returned_in_chain_order(self, env_cls):
        env = env_cls(reset_noise_scale=0.0, **self.WEIGHTS)
        try:
            env.reset(seed=0)
            # Yawed 0.4 rad about z, then pitched 0.6 rad nose-down about the body's y.
            half_yaw, half_pitch = 0.2, 0.3
            quat = np.array(
                [
                    math.cos(half_yaw) * math.cos(half_pitch),
                    -math.sin(half_yaw) * math.sin(half_pitch),
                    math.cos(half_yaw) * math.sin(half_pitch),
                    math.sin(half_yaw) * math.cos(half_pitch),
                ]
            )
            forward_ref_2d = np.array([0.6, 0.8])
            vel_2d = np.array([-0.024, -0.018])
            root_pos_2d = env._initial_pos_2d + np.array([0.3, -0.4])

            reward_forward, forward_vel = rf.reward_forward_velocity(
                vel_2d, forward_ref_2d, env.forward_vel_max, env.forward_vel_weight
            )
            reward_backward, backward_vel = rf.reward_backward_penalty(
                forward_vel, env.forward_vel_max, env.backward_vel_penalty_weight
            )
            reward_drift, drift_distance = rf.reward_drift_penalty(
                root_pos_2d, env._initial_pos_2d, env.drift_penalty_weight
            )
            reward_nosedive, forward_z = rf.reward_nosedive(quat, env.nosedive_weight, env._natural_forward_z)
            body_forward_2d = np.asarray(rf.quat_to_forward_2d(quat))
            reward_heading, heading_alignment = rf.reward_heading_alignment(
                body_forward_2d, forward_ref_2d, env.heading_weight
            )
            reward_lateral, lateral_vel = rf.reward_lateral_velocity_penalty(
                vel_2d, body_forward_2d, env.lateral_penalty_weight
            )
            reward_speed, abs_speed = rf.reward_speed_penalty(
                vel_2d, env.speed_penalty_weight, env.speed_penalty_threshold
            )
            reward_idle, _ = rf.reward_idle_penalty(vel_2d, env.idle_penalty_weight, env.idle_velocity_threshold)
            terms = (
                reward_forward,
                reward_backward,
                reward_drift,
                reward_nosedive,
                reward_heading,
                reward_lateral,
                reward_speed,
                reward_idle,
            )
            assert all(term != 0.0 for term in terms), terms
            assert len(set(terms)) == len(terms), terms

            info = {}
            returned = env._progress_terms(info, vel_2d, forward_ref_2d, root_pos_2d)
            self._assert_recorded(
                info,
                [
                    ("forward_vel", forward_vel),
                    ("reward_forward", reward_forward),
                    ("backward_vel", backward_vel),
                    ("reward_backward", reward_backward),
                    ("drift_distance", drift_distance),
                    ("reward_drift", reward_drift),
                ],
            )
            assert returned == (info["reward_forward"], info["reward_backward"], info["reward_drift"])

            info = {}
            returned = env._nosedive_term(info, quat)
            self._assert_recorded(info, [("forward_z", forward_z), ("reward_nosedive", reward_nosedive)])
            assert returned == info["reward_nosedive"]

            info = {}
            returned = env._heading_terms(info, quat, forward_ref_2d, vel_2d)
            self._assert_recorded(
                info,
                [
                    ("heading_alignment", heading_alignment),
                    ("reward_heading", reward_heading),
                    ("lateral_vel", lateral_vel),
                    ("reward_lateral", reward_lateral),
                ],
            )
            assert returned == (info["reward_heading"], info["reward_lateral"])

            info = {}
            returned = env._speed_terms(info, vel_2d)
            self._assert_recorded(
                info, [("abs_speed", abs_speed), ("reward_speed", reward_speed), ("reward_idle", reward_idle)]
            )
            assert returned == (info["reward_speed"], info["reward_idle"])
        finally:
            env.close()


class TestContactGeom:
    """``_contact_geom``: the one contact query behind the bite, strike and snap checks and the floor scan.

    It scans ``data.contact`` in order and returns the geom of *geoms* from the first contact that
    pairs one of them with *other_geom*, from either side of the pair, or None.  The first match
    decides which body part a categorised floor contact reports.
    """

    @staticmethod
    def _query(pairs, geoms, other_geom):
        contacts = [types.SimpleNamespace(geom1=geom1, geom2=geom2) for geom1, geom2 in pairs]
        holder = types.SimpleNamespace(data=types.SimpleNamespace(ncon=len(contacts), contact=contacts))
        return BaseDinoEnv._contact_geom(holder, geoms, other_geom)

    def test_no_contact_is_none(self):
        assert self._query([], {3, 4}, 0) is None

    def test_contacts_without_the_other_geom_are_none(self):
        assert self._query([(3, 4), (5, 3), (7, 8)], {3, 4}, 0) is None

    def test_contacts_with_the_other_geom_but_not_the_geoms_are_none(self):
        assert self._query([(0, 5), (6, 0)], {3, 4}, 0) is None

    def test_first_matching_contact_wins_from_either_side(self):
        assert self._query([(5, 9), (0, 4), (3, 0)], {3, 4}, 0) == 4
        assert self._query([(5, 9), (3, 0), (0, 4)], {3, 4}, 0) == 3

    def test_a_live_airborne_body_has_no_floor_contact(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            env.data.qpos[2] += 1.0
            mujoco.mj_forward(env.model, env.data)
            assert env._contact_geom(set(range(env.model.ngeom)), env.floor_geom_id) is None
        finally:
            env.close()

    def test_a_live_standing_body_reports_its_first_floor_contact(self):
        env = RaptorEnv(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            mujoco.mj_forward(env.model, env.data)
            floor = env.floor_geom_id
            pairs = [(int(env.data.contact[i].geom1), int(env.data.contact[i].geom2)) for i in range(env.data.ncon)]
            touching = [geom1 if geom2 == floor else geom2 for geom1, geom2 in pairs if floor in (geom1, geom2)]
            assert touching, "the settled raptor should stand on the floor"
            assert env._contact_geom(set(touching), floor) == touching[0]
            assert env._contact_geom(set(touching), -2) is None
        finally:
            env.close()


class TestRootTermination:
    """``_root_termination``: the height/tilt(/nosedive) opening every dual species' termination shares.

    The digest harness hashes the order of info keys, so the order is part of the contract: the
    height key, ``tilt_angle``, ``forward_z`` only when a nosedive threshold is given, then
    ``termination_reason`` only on termination.
    """

    @pytest.fixture
    def env(self):
        e = RaptorEnv(reset_noise_scale=0.0)
        e.reset(seed=0)
        yield e
        e.close()

    def test_healthy_with_a_threshold_records_height_tilt_then_forward_z(self, env):
        terminated, info = env._root_termination(env.pelvis_id, "pelvis_height", 0.5)
        assert not terminated
        assert list(info) == ["pelvis_height", "tilt_angle", "forward_z"]

    def test_healthy_without_a_threshold_records_no_forward_z(self, env):
        terminated, info = env._root_termination(env.pelvis_id, "some_height")
        assert not terminated
        assert list(info) == ["some_height", "tilt_angle"]

    def test_a_height_failure_ends_before_the_nosedive_check(self, env):
        env.data.qpos[2] = env.healthy_z_range[0] - 0.1
        mujoco.mj_forward(env.model, env.data)
        terminated, info = env._root_termination(env.pelvis_id, "pelvis_height", 0.5)
        assert terminated
        assert info["termination_reason"] == "fallen"
        assert list(info) == ["pelvis_height", "tilt_angle", "termination_reason"]

    @pytest.mark.parametrize(
        "module, class_name, prefix",
        [
            ("environments.trex.envs.trex_env", "TRexEnv", ["pelvis_height", "tilt_angle", "forward_z"]),
            ("environments.velociraptor.envs.raptor_env", "RaptorEnv", ["pelvis_height", "tilt_angle", "forward_z"]),
            ("environments.brachiosaurus.envs.brachio_env", "BrachioEnv", ["torso_height", "tilt_angle"]),
            (
                "environments.dibothrosuchus.envs.dibothrosuchus_env",
                "DibothrosuchusEnv",
                ["torso_height", "tilt_angle", "forward_z"],
            ),
        ],
    )
    def test_each_species_termination_opens_with_the_prefix(self, module, class_name, prefix):
        import importlib

        env = getattr(importlib.import_module(module), class_name)(reset_noise_scale=0.0)
        try:
            env.reset(seed=0)
            terminated, info = env._is_terminated()
            assert not terminated, info
            assert list(info)[: len(prefix)] == prefix
            if "forward_z" not in prefix:
                assert "forward_z" not in info
        finally:
            env.close()

    @pytest.mark.parametrize("env_cls", [TRexEnv, DibothrosuchusEnv])
    @pytest.mark.parametrize(
        "offset, reason", [pytest.param(-0.01, "nosedive", id="just_past"), pytest.param(0.01, None, id="just_short")]
    )
    def test_the_nosedive_threshold_is_the_species_knob(self, env_cls, offset, reason):
        """T-Rex and Dibothrosuchus pass ``nosedive_termination_threshold`` to the prefix.

        The digest probes that reach ``nosedive`` end there under the knob and a nearby literal
        alike, so they cannot tell the two apart.  A non-default knob of 0.3 is pinned from
        both sides instead: the root is lifted 0.1 above its settled height (inside
        ``healthy_z_range``) and pitched nose-down about +y to 0.01 past or 0.01 short of
        ``_natural_forward_z - 0.3``; the tilt then stays well inside ``max_tilt_angle``.
        """
        env = env_cls(reset_noise_scale=0.0, nosedive_termination_threshold=0.3)
        try:
            env.reset(seed=0)
            forward_z = env._natural_forward_z - 0.3 + offset
            pitch = math.asin(-forward_z)
            env.data.qpos[2] += 0.1
            env.data.qpos[3:7] = (math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0)
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            terminated, info = env._is_terminated()

            assert info["forward_z"] == pytest.approx(forward_z, abs=1e-9)
            assert info["tilt_angle"] < env.max_tilt_angle
            assert info.get("termination_reason") == reason, info
            assert terminated == (reason is not None)
        finally:
            env.close()


class TestCacheHomeKeyframe:
    """``_cache_home_keyframe``: the ``home`` keyframe lookup the trex, velociraptor, brachiosaurus and
    dibothrosuchus ``_cache_ids`` share."""

    _XML = """
    <mujoco>
      <worldbody>
        <body name="b"><joint name="j" type="hinge"/><geom size="0.1"/></body>
      </worldbody>
      <actuator><motor joint="j"/></actuator>
      <keyframe>{keys}</keyframe>
    </mujoco>
    """

    @classmethod
    def _holder(cls, keys):
        return types.SimpleNamespace(model=mujoco.MjModel.from_xml_string(cls._XML.format(keys=keys)))

    def test_a_model_without_home_is_refused_with_the_species_label(self):
        holder = self._holder('<key name="other" ctrl="0.5"/>')
        with pytest.raises(ValueError, match="^Velociraptor model must define a named 'home' keyframe$"):
            BaseDinoEnv._cache_home_keyframe(holder, "Velociraptor")

    def test_home_is_the_reset_keyframe_and_its_controls_are_a_copy(self):
        holder = self._holder('<key name="other" ctrl="0.5"/><key name="home" ctrl="0.25"/>')
        BaseDinoEnv._cache_home_keyframe(holder, "Test")
        assert holder.home_keyframe_id == holder._reset_keyframe_id == 1
        np.testing.assert_array_equal(holder._home_ctrl, [0.25])
        holder._home_ctrl[0] = 9.0
        assert holder.model.key_ctrl[1, 0] == 0.25
        # Re-callable, as the behavior env's model swap requires.
        BaseDinoEnv._cache_home_keyframe(holder, "Test")
        np.testing.assert_array_equal(holder._home_ctrl, [0.25])
