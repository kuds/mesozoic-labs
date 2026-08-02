"""Tests for environments.shared.curriculum.manager."""

import pytest

from environments.shared.curriculum import (
    CurriculumManager,
    StageThreshold,
    thresholds_from_configs,
)
from environments.shared.curriculum.gate_schema import (
    GATE_KINDS,
    GATE_SCHEMA_VERSION,
    GateSchemaError,
    validate_gate_config,
    validate_gate_configs,
)

#: The gate declaration every real stage config carries. Tests that exercise a
#: working gate must include it, because an undeclared gate is now fatal.
_GATE = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "reward_and_length/v1"}

#: The explicit, recorded non-advancing mode.
_PILOT = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "none/v1"}


class TestStageThreshold:
    """Test StageThreshold defaults."""

    def test_default_values(self):
        t = StageThreshold()
        assert t.min_avg_reward == float("-inf")
        assert t.min_avg_episode_length == 0.0
        assert t.min_avg_forward_vel == 0.0
        assert t.min_eval_episodes == 10
        assert t.required_consecutive == 3

    def test_custom_values(self):
        t = StageThreshold(min_avg_reward=50.0, required_consecutive=5)
        assert t.min_avg_reward == 50.0
        assert t.required_consecutive == 5

    def test_forward_vel_threshold(self):
        t = StageThreshold(min_avg_forward_vel=0.5)
        assert t.min_avg_forward_vel == 0.5


class TestCurriculumManager:
    """Test CurriculumManager lifecycle."""

    @pytest.fixture
    def manager(self):
        return CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_eval_episodes": 3,
                    "required_consecutive": 2,
                },
                2: {
                    "min_avg_reward": 50.0,
                    "min_avg_episode_length": 200,
                    "min_eval_episodes": 3,
                    "required_consecutive": 2,
                },
            },
            start_stage=1,
        )

    def test_initial_stage(self, manager):
        assert manager.current_stage == 1
        assert not manager.is_final_stage

    def test_current_config_returns_dict(self, manager):
        config = manager.current_config()
        assert "name" in config
        assert "env_kwargs" in config
        assert "ppo_kwargs" in config

    def test_current_threshold_returns_effective_override(self):
        manager = CurriculumManager(
            species="velociraptor",
            stage_thresholds={1: {"min_success_rate": 0.75}},
            start_stage=1,
        )

        assert manager.current_threshold.min_success_rate == 0.75

    def test_should_not_advance_without_data(self, manager):
        assert not manager.should_advance()

    def test_should_not_advance_below_threshold(self, manager):
        # Reward below threshold
        rewards = [5.0, 5.0, 5.0]
        lengths = [100.0, 100.0, 100.0]
        assert not manager.should_advance(rewards, lengths)

    def test_should_advance_after_consecutive_passes(self, manager):
        rewards = [15.0, 15.0, 15.0]
        lengths = [100.0, 100.0, 100.0]

        # First pass
        assert not manager.should_advance(rewards, lengths)
        # Second consecutive pass -> should advance
        assert manager.should_advance(rewards, lengths)

    def test_consecutive_resets_on_failure(self, manager):
        good_rewards = [15.0, 15.0, 15.0]
        bad_rewards = [5.0, 5.0, 5.0]
        lengths = [100.0, 100.0, 100.0]

        # First pass
        manager.should_advance(good_rewards, lengths)
        # Failure resets counter
        manager.should_advance(bad_rewards, lengths)
        # First pass again (not enough consecutive)
        assert not manager.should_advance(good_rewards, lengths)
        # Second consecutive -> now passes
        assert manager.should_advance(good_rewards, lengths)

    def test_advance_increments_stage(self, manager):
        new_stage = manager.advance()
        assert new_stage == 2
        assert manager.current_stage == 2

    def test_advance_to_final_stage(self, manager):
        manager.advance()
        manager.advance()
        assert manager.current_stage == 3
        assert manager.is_final_stage

    def test_advance_past_final_raises(self, manager):
        manager.advance()
        manager.advance()
        with pytest.raises(RuntimeError, match="Cannot advance past final stage"):
            manager.advance()

    def test_should_not_advance_on_final_stage(self, manager):
        manager.advance()
        manager.advance()
        # Even with good data, can't advance past final
        rewards = [100.0] * 10
        lengths = [500.0] * 10
        assert not manager.should_advance(rewards, lengths)

    def test_record_eval_returns_summary(self, manager):
        summary = manager.record_eval([10.0, 20.0], [100.0, 200.0])
        assert summary["mean_reward"] == 15.0
        assert summary["mean_length"] == 150.0
        assert summary["n_episodes"] == 2

    def test_summary_contains_history(self, manager):
        manager.record_eval([10.0, 20.0], [100.0, 200.0])
        s = manager.summary()
        assert s["species"] == "velociraptor"
        assert s["current_stage"] == 1
        assert len(s["eval_history"][1]) == 1

    def test_min_eval_episodes_enforced(self):
        """Threshold requires 5 episodes but we only provide 3."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {"min_avg_reward": 0.0, "min_eval_episodes": 5, "required_consecutive": 1},
            },
        )
        # Only 3 episodes provided
        assert not mgr.should_advance([100.0, 100.0, 100.0], [500.0, 500.0, 500.0])

    def test_forward_vel_gate_blocks_without_velocity(self):
        """Stage with forward velocity threshold should block if velocity is too low."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_avg_forward_vel": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        # Good reward/length but no forward velocity data -> defaults to 0.0
        assert not mgr.should_advance(rewards, lengths)

    def test_forward_vel_gate_blocks_low_velocity(self):
        """Stage with forward velocity threshold should block if velocity is below threshold."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_avg_forward_vel": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        low_vels = [0.1, 0.2, 0.1]
        assert not mgr.should_advance(rewards, lengths, low_vels)

    def test_forward_vel_gate_passes_with_good_velocity(self):
        """Stage with forward velocity threshold should pass when all metrics met."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_avg_forward_vel": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        good_vels = [1.0, 1.2, 0.8]
        assert mgr.should_advance(rewards, lengths, good_vels)

    def test_success_rate_gate_blocks_low_rate(self):
        """Success rate below threshold should block advancement."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_success_rate": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        low_success = [0.0, 0.0, 1.0]  # mean = 0.33, below 0.5
        assert not mgr.should_advance(rewards, lengths, success_rates=low_success)

    def test_success_rate_gate_passes_high_rate(self):
        """Success rate above threshold should allow advancement."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_success_rate": 0.5,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        rewards = [100.0, 100.0, 100.0]
        lengths = [500.0, 500.0, 500.0]
        high_success = [1.0, 1.0, 0.0]  # mean = 0.67, above 0.5
        assert mgr.should_advance(rewards, lengths, success_rates=high_success)

    def test_record_eval_with_forward_vel(self):
        """record_eval should include forward velocity in summary."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={1: {"min_avg_reward": 0.0, "required_consecutive": 1}},
        )
        summary = mgr.record_eval([10.0, 20.0], [100.0, 200.0], forward_velocities=[1.0, 2.0])
        assert summary["mean_forward_vel"] == 1.5

    def test_record_eval_with_success_rate(self):
        """record_eval should include success rate in summary."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={1: {"min_avg_reward": 0.0, "required_consecutive": 1}},
        )
        summary = mgr.record_eval([10.0, 20.0], [100.0, 200.0], success_rates=[1.0, 0.0])
        assert summary["mean_success_rate"] == 0.5


class TestThresholdsFromConfigs:
    """Test extracting thresholds from loaded TOML configs."""

    def test_extracts_reward_threshold(self):
        configs = {
            1: {"curriculum_kwargs": dict(_GATE, min_avg_reward=10.0, required_consecutive=2)},
            2: {"curriculum_kwargs": dict(_GATE, min_avg_reward=50.0)},
            3: {"curriculum_kwargs": dict(_PILOT)},
        }
        thresholds = thresholds_from_configs(configs, advancement_enabled=False)
        assert thresholds[1]["min_avg_reward"] == 10.0
        assert thresholds[1]["required_consecutive"] == 2
        assert thresholds[2]["min_avg_reward"] == 50.0
        # A declared non-advancing pilot now gets an explicit entry carrying its
        # gate kind, rather than no entry at all. The old "no entry" shape left
        # it on StageThreshold's permissive defaults (min_avg_reward = -inf),
        # which pass on any evaluation; only the schema's refusal to accept
        # "none/v1" under advancement kept that from being reachable.
        assert thresholds[3]["gate_kind"] == "none/v1"

    def test_extracts_all_threshold_fields(self):
        configs = {
            1: {
                "curriculum_kwargs": dict(
                    _GATE,
                    min_avg_reward=10.0,
                    min_avg_episode_length=100,
                    min_avg_forward_vel=0.5,
                    min_success_rate=0.3,
                    min_eval_episodes=12,
                    required_consecutive=3,
                ),
            },
        }
        thresholds = thresholds_from_configs(configs)
        assert thresholds[1]["min_avg_forward_vel"] == 0.5
        assert thresholds[1]["min_success_rate"] == 0.3
        assert thresholds[1]["min_eval_episodes"] == 12

    def test_empty_configs(self):
        thresholds = thresholds_from_configs({})
        assert thresholds == {}

    def test_undeclared_gate_is_fatal_when_advancement_is_enabled(self):
        """Fail closed: a stage with no gate declaration must not advance.

        A composite-only gate config used to have its unknown fields silently
        discarded here, after which StageThreshold's permissive defaults
        (min_avg_reward = -inf, length and success floors 0) advanced the stage
        on any evaluation at all. See docs/STAGE1_SPLIT_PLAN.md section 5.2.
        """
        configs = {1: {"curriculum_kwargs": {"min_avg_reward": 10.0}}}
        with pytest.raises(GateSchemaError, match="no gate_kind declared"):
            thresholds_from_configs(configs)

    def test_unknown_field_is_fatal_rather_than_silently_dropped(self):
        configs = {1: {"curriculum_kwargs": dict(_GATE, min_stance_success_lcb=0.90)}}
        with pytest.raises(GateSchemaError, match="unrecognised"):
            thresholds_from_configs(configs)

    def test_threshold_belonging_to_another_gate_kind_is_fatal(self):
        """A leftover threshold implies a gate that is not actually enforced."""
        configs = {1: {"curriculum_kwargs": dict(_PILOT, min_avg_reward=10.0)}}
        with pytest.raises(GateSchemaError, match="does not consume"):
            thresholds_from_configs(configs, advancement_enabled=False)

    def test_non_advancing_pilot_is_rejected_when_advancement_is_enabled(self):
        configs = {1: {"curriculum_kwargs": dict(_PILOT)}}
        with pytest.raises(GateSchemaError, match="non-advancing pilot"):
            thresholds_from_configs(configs)

    def test_declared_gate_without_required_thresholds_is_fatal(self):
        """A reward gate with no reward threshold must raise, not default open.

        This shape used to pass the schema (which only rejected *misplaced*
        threshold keys), yield no threshold_fields, and drop through to
        StageThreshold's permissive defaults (min_avg_reward = -inf) — the SB3
        path advancing on any evaluation while the JAX path raised. The schema
        now requires each gate kind's core field, so both backends reject it.
        """
        configs = {1: {"curriculum_kwargs": dict(_GATE)}}
        with pytest.raises(GateSchemaError, match="missing required threshold"):
            thresholds_from_configs(configs)
        # Malformed is malformed even when the run cannot advance.
        with pytest.raises(GateSchemaError, match="missing required threshold"):
            thresholds_from_configs(configs, advancement_enabled=False)

    def test_every_committed_stage_config_declares_a_valid_gate(self):
        """The shipped configs must satisfy the schema on every species.

        Asserts every declared kind is one the registry knows and both
        backends evaluate, rather than pinning the specific kind each stage
        uses — pinning the literal made adopting stance_quality/v1 for T-Rex
        stage 1a look like a regression instead of the intended change.
        """
        from environments.shared.config import load_all_stages

        for species in ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus"):
            kinds = validate_gate_configs(load_all_stages(species))
            assert set(kinds.values()) <= set(GATE_KINDS), species
            # "none/v1" refuses to advance, so a shipped curriculum stage must
            # never declare it.
            assert "none/v1" not in set(kinds.values()), species

    def test_trex_stage1_gates_on_stance_quality(self):
        """T-Rex 1a must not be gated on return: a statue is the reward optimum."""
        from environments.shared.config import load_all_stages

        curriculum = load_all_stages("trex")[1]["curriculum_kwargs"]
        assert curriculum["gate_kind"] == "stance_quality/v1"
        # The statue scores 3271.8 at 1000.0 steps, so the retired reward and
        # length criteria were both cleared by doing nothing.
        assert curriculum["min_avg_reward"] < 3271.8, "reward must be a rail below the statue, not a gate"
        assert "min_avg_episode_length" not in curriculum
        # The bound's power is specified at this panel size.
        assert curriculum["min_eval_episodes"] == 40

    def test_with_real_configs(self):
        """Integration test: extract thresholds from actual TOML configs."""
        from environments.shared.config import load_all_stages

        configs = load_all_stages("velociraptor")
        thresholds = thresholds_from_configs(configs)
        assert isinstance(thresholds, dict)


class TestRetentionKeys:
    """``max_checkpoints`` configures artifact retention, not the gate."""

    def test_the_schema_accepts_it(self):
        # It is only settable from a TOML, and the fail-closed unknown-key
        # check would otherwise make it unreachable there.
        assert (
            validate_gate_config(
                1,
                {**_GATE, "min_avg_reward": 100.0, "max_checkpoints": 3},
            )
            == "reward_and_length/v1"
        )

    def test_it_is_not_carried_onto_the_threshold(self):
        thresholds = thresholds_from_configs(
            {1: {"curriculum_kwargs": {**_GATE, "min_avg_reward": 100.0, "max_checkpoints": 3}}}
        )
        assert "max_checkpoints" not in thresholds[1]
