"""Tests for jax_ppo module — focuses on fixes from the JAX review."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fix #1: sample_action returns raw (unclipped) action with matching log_prob
# ---------------------------------------------------------------------------


class TestSampleActionLogProb:
    """Verify that sample_action returns raw actions with consistent log_prob."""

    @pytest.fixture()
    def _jax(self):
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        return jax, jnp

    def test_log_prob_independent_of_clip_boundary(self, _jax):
        """Actions near ±1 should have log_probs consistent with the Gaussian,
        not distorted by the clip operation."""
        jax, jnp = _jax
        from environments.shared.jax_ppo import make_actor_critic, sample_action

        network = make_actor_critic(action_dim=2, hidden_dims=(8,))
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros(4)
        params = network.init(rng, dummy)

        # Sample many actions and collect log_probs
        rng, *rngs = jax.random.split(rng, 101)
        log_probs = []
        actions = []
        for r in rngs:
            a, lp, _ = sample_action(params, network, dummy, r)
            log_probs.append(float(lp))
            actions.append(np.array(a))

        # log_probs should be finite and not all identical
        assert all(np.isfinite(lp) for lp in log_probs)
        assert np.std(log_probs) > 0, "log_probs should vary across samples"

    def test_raw_action_not_clipped(self, _jax):
        """sample_action should return raw (unclipped) actions that may exceed [-1, 1]."""
        jax, jnp = _jax
        from environments.shared.jax_ppo import make_actor_critic, sample_action

        network = make_actor_critic(action_dim=8, hidden_dims=(8,))
        rng = jax.random.PRNGKey(99)
        dummy = jnp.zeros(4)
        params = network.init(rng, dummy)

        # With std=1.0 (log_std init zeros) and 8 dims, some samples should exceed [-1, 1]
        found_outside = False
        rng, *rngs = jax.random.split(rng, 201)
        for r in rngs:
            action, _, _ = sample_action(params, network, dummy, r)
            if float(jnp.max(jnp.abs(action))) > 1.0:
                found_outside = True
                break
        assert found_outside, "Raw actions should sometimes exceed [-1, 1] bounds"

    def test_log_prob_matches_manual_gaussian(self, _jax):
        """log_prob should match the analytical diagonal-Gaussian formula
        applied to the raw (unclipped) action."""
        jax, jnp = _jax
        from environments.shared.jax_ppo import make_actor_critic, sample_action

        network = make_actor_critic(action_dim=2, hidden_dims=(8,))
        rng = jax.random.PRNGKey(42)
        dummy = jnp.zeros(4)
        params = network.init(rng, dummy)

        rng, sample_rng = jax.random.split(rng)
        action, log_prob, _ = sample_action(params, network, dummy, sample_rng)

        # Recompute what the network outputs
        mean, log_std, _ = network.apply(params, dummy)
        std = jnp.exp(log_std)

        # action is raw (unclipped), so manual Gaussian log_prob must match exactly
        manual_lp = -0.5 * jnp.sum(jnp.square((action - mean) / (std + 1e-8)) + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
        assert float(log_prob) == pytest.approx(float(manual_lp), abs=1e-5)


# ---------------------------------------------------------------------------
# Fix #12: hidden_dims default consistency
# ---------------------------------------------------------------------------


class TestActorCriticDefaults:
    def test_default_hidden_dims_match(self):
        """The inner class default should match the outer function default."""
        pytest.importorskip("jax")
        from environments.shared.jax_ppo import make_actor_critic

        network = make_actor_critic(action_dim=3)
        # The network's hidden_dims attribute should be (512, 256)
        assert network.hidden_dims == (512, 256)

    def test_custom_hidden_dims_propagated(self):
        """Custom hidden_dims should be passed through correctly."""
        pytest.importorskip("jax")
        from environments.shared.jax_ppo import make_actor_critic

        network = make_actor_critic(action_dim=3, hidden_dims=(64, 32))
        assert network.hidden_dims == (64, 32)


# ---------------------------------------------------------------------------
# PPOConfig: learning rate schedule total_steps
# ---------------------------------------------------------------------------


class TestMakeOptimizer:
    def test_constant_lr(self):
        pytest.importorskip("optax")
        from environments.shared.jax_ppo import PPOConfig, make_optimizer

        cfg = PPOConfig(learning_rate=1e-3, learning_rate_end=None)
        opt = make_optimizer(cfg)
        assert opt is not None

    def test_linear_decay_lr(self):
        pytest.importorskip("optax")
        from environments.shared.jax_ppo import PPOConfig, make_optimizer

        cfg = PPOConfig(
            learning_rate=3e-4,
            learning_rate_end=1e-5,
            total_updates=100,
            n_epochs=4,
        )
        opt = make_optimizer(cfg)
        assert opt is not None


# ---------------------------------------------------------------------------
# compute_gae: verify done masking distinguishes termination from truncation
# ---------------------------------------------------------------------------


class TestComputeGAE:
    """The trainer feeds compute_gae a done mask that is 1 ONLY for natural
    termination — truncation must keep the bootstrap value alive."""

    @pytest.fixture()
    def _jnp(self):
        jnp = pytest.importorskip("jax.numpy")
        return jnp

    def test_terminal_done_zeros_bootstrap(self, _jnp):
        """When done=1 at step T-1, GAE must not propagate V(s_T)."""
        from environments.shared.jax_ppo import compute_gae

        jnp = _jnp
        # 1 env, 3 steps; done at the last step.
        rewards = jnp.array([[0.0], [0.0], [1.0]])
        # values = [V(s_0), V(s_1), V(s_2), V(s_3)]; bootstrap at index 3.
        values = jnp.array([[0.0], [0.0], [0.0], [10.0]])
        dones = jnp.array([[0.0], [0.0], [1.0]])

        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        # Step 2 done=1 → δ_2 = r_2 + 0 - V(s_2) = 1.0; A_2 = 1.0
        # The bootstrap V(s_3)=10 must NOT enter A_2.
        assert float(adv[-1, 0]) == pytest.approx(1.0, abs=1e-5)
        assert float(ret[-1, 0]) == pytest.approx(1.0, abs=1e-5)

    def test_no_done_bootstraps_value(self, _jnp):
        """When done=0 (truncation, in our convention), bootstrap V(s_T)."""
        from environments.shared.jax_ppo import compute_gae

        jnp = _jnp
        rewards = jnp.array([[1.0]])
        values = jnp.array([[0.0], [10.0]])  # V(s_0)=0, bootstrap V(s_1)=10
        dones = jnp.array([[0.0]])

        adv, _ = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        # δ_0 = r + γ * V(s_1) - V(s_0) = 1 + 0.99*10 - 0 = 10.9; A_0 = 10.9
        assert float(adv[0, 0]) == pytest.approx(10.9, abs=1e-4)

    def test_truncation_vs_termination_diverge(self, _jnp):
        """Same trajectory, two done patterns — bootstrap behavior must differ."""
        from environments.shared.jax_ppo import compute_gae

        jnp = _jnp
        rewards = jnp.array([[0.0]])
        values = jnp.array([[0.0], [5.0]])

        adv_truncated, _ = compute_gae(
            rewards, values, jnp.array([[0.0]]), gamma=1.0, gae_lambda=1.0
        )
        adv_terminated, _ = compute_gae(
            rewards, values, jnp.array([[1.0]]), gamma=1.0, gae_lambda=1.0
        )
        # Truncation bootstraps: A_0 = 0 + 5 - 0 = 5
        # Termination zeros bootstrap: A_0 = 0 + 0 - 0 = 0
        assert float(adv_truncated[0, 0]) == pytest.approx(5.0, abs=1e-5)
        assert float(adv_terminated[0, 0]) == pytest.approx(0.0, abs=1e-5)

    def test_chronological_order(self, _jnp):
        """Output advantages must be in chronological (forward) order."""
        from environments.shared.jax_ppo import compute_gae

        jnp = _jnp
        rewards = jnp.array([[1.0], [2.0], [3.0]])
        values = jnp.array([[0.0], [0.0], [0.0], [0.0]])
        dones = jnp.array([[0.0], [0.0], [0.0]])

        adv, ret = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
        # With γ=λ=1, advantage at step t = sum of future rewards.
        # A_0 = r_0 + r_1 + r_2 = 6; A_1 = 5; A_2 = 3
        assert float(adv[0, 0]) == pytest.approx(6.0, abs=1e-5)
        assert float(adv[1, 0]) == pytest.approx(5.0, abs=1e-5)
        assert float(adv[2, 0]) == pytest.approx(3.0, abs=1e-5)
