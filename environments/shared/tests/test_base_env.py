import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.shared.base_env import BaseDinoEnv
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
    """Test the shared midpoint action mapping retained by other species."""

    @pytest.fixture
    def env(self):
        e = BrachioEnv()
        yield e
        e.close()

    def test_zero_action_maps_to_midpoint(self, env):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_range = env.model.actuator_ctrlrange
        midpoint = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2
        np.testing.assert_allclose(scaled, midpoint, atol=1e-6)

    def test_plus_one_maps_to_max(self, env):
        action = np.ones(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_max = env.model.actuator_ctrlrange[:, 1]
        np.testing.assert_allclose(scaled, ctrl_max, atol=1e-6)

    def test_minus_one_maps_to_min(self, env):
        action = -np.ones(env.action_space.shape, dtype=np.float32)
        scaled = env._scale_action(action)
        ctrl_min = env.model.actuator_ctrlrange[:, 0]
        np.testing.assert_allclose(scaled, ctrl_min, atol=1e-6)


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
