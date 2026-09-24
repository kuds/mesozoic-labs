"""Tests for environments.shared.curriculum.manager."""

from pathlib import Path

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
    declared_certification_seeds,
    gate_config_sha256,
    gate_config_view,
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

    def test_min_success_lcb_defaults_fail_closed(self):
        """No binomial bound reaches +inf, so an unpopulated bar refuses every panel."""
        assert StageThreshold().min_success_lcb == float("inf")


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

    def test_task_success_gate_blocks_low_lcb(self):
        """19/30 bounds at 0.467 < 0.5: refused, whatever the reward."""
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "gate_kind": "task_success/v1",
                    "min_success_lcb": 0.5,
                    "min_eval_episodes": 30,
                    "min_avg_reward": 100.0,
                    "required_consecutive": 1,
                }
            },
        )
        assert not mgr.should_advance([1e6] * 30, [1000.0] * 30, success_rates=[1.0] * 19 + [0.0] * 11)
        latest = mgr.summary()["eval_history"][1][-1]
        assert (latest["success_count"], latest["n_success_samples"]) == (19, 30)

    def test_task_success_gate_passes_high_lcb(self):
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "gate_kind": "task_success/v1",
                    "min_success_lcb": 0.5,
                    "min_eval_episodes": 30,
                    "min_avg_reward": 100.0,
                    "required_consecutive": 2,
                }
            },
        )
        panel = [1.0] * 20 + [0.0] * 10
        assert not mgr.should_advance([500.0] * 30, [1000.0] * 30, success_rates=panel)
        assert mgr.should_advance([500.0] * 30, [1000.0] * 30, success_rates=panel)

    def test_task_success_gate_keeps_the_reward_rail_as_a_conjunct(self):
        mgr = CurriculumManager(
            species="velociraptor",
            stage_thresholds={
                1: {
                    "gate_kind": "task_success/v1",
                    "min_success_lcb": 0.5,
                    "min_eval_episodes": 30,
                    "min_avg_reward": 361.0,
                    "required_consecutive": 1,
                }
            },
        )
        assert not mgr.should_advance([200.0] * 30, [1000.0] * 30, success_rates=[1.0] * 30)

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

    def test_extracts_task_success_thresholds(self):
        """task_success/v1's bar is copied (the manager judges it in-training), unlike recovery's."""
        configs = {
            1: {
                "curriculum_kwargs": {
                    "gate_schema_version": GATE_SCHEMA_VERSION,
                    "gate_kind": "task_success/v1",
                    "min_success_lcb": 0.5,
                    "min_eval_episodes": 30,
                    "min_avg_reward": 361.0,
                    "required_consecutive": 1,
                }
            }
        }
        thresholds = thresholds_from_configs(configs)
        assert thresholds[1] == {
            "gate_kind": "task_success/v1",
            "min_success_lcb": 0.5,
            "min_eval_episodes": 30,
            "min_avg_reward": 361.0,
            "required_consecutive": 1,
        }
        assert StageThreshold(**thresholds[1]).min_success_lcb == 0.5

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

        The advancing chain and semantic-only stages are validated
        separately, mirroring how they run: advancement walks the legacy
        integer chain (thresholds_from_configs filters to int keys), so
        those stages must carry advancing-valid gates; a semantic-only
        stage (today: trex "recovery") is reachable only by ID as an
        explicit pilot, and its fail-closed "none/v1" placeholder is the
        DESIGNED state until P5 lands measured recovery_quality/v1
        thresholds — gate_schema's own tests pin that it refuses to
        advance.
        """
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import _CONFIGS_DIR

        # Every species with a stage manifest, compsognathus and its robot
        # included: D-B14 keeps them on reward_and_length/v1, and the loop
        # must see them so a kind drift there fails here too.
        species_ids = sorted(path.parent.name for path in _CONFIGS_DIR.glob("*/stages.toml"))
        assert {"trex", "velociraptor", "brachiosaurus", "dibothrosuchus", "compsognathus"} <= set(species_ids)
        for species in species_ids:
            stages = load_all_stages(species)
            chain = {stage: cfg for stage, cfg in stages.items() if isinstance(stage, int)}
            kinds = validate_gate_configs(chain)
            assert set(kinds.values()) <= set(GATE_KINDS), species
            # "none/v1" refuses to advance, so a stage on the advancing
            # chain must never declare it.
            assert "none/v1" not in set(kinds.values()), species
            pilots = {stage: cfg for stage, cfg in stages.items() if isinstance(stage, str)}
            pilot_kinds = validate_gate_configs(pilots, advancement_enabled=False)
            assert set(pilot_kinds.values()) <= set(GATE_KINDS), species

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

    def test_trex_behavior_gates_on_task_success(self):
        """The trex hunt is certified on the binomial LCB95, not on speed or a raw mean (plan §4.4, WS-B2).

        Review CF2 / plan D1 retired ``min_avg_forward_vel`` (a bite episode
        cannot average 2.0 m/s) and SS2 retired the raw-mean
        ``min_success_rate`` (43% false-block at threshold); the schema's
        misplaced-key check would reject either if it crept back.  The panel
        size is REQUIRED for this kind because the bound's power is a
        function of the declared n, and it is coupled to the notebook's
        provenance ``evaluation_episodes`` (result_bundle/evidence.py
        refuses any other row count) -- decision D-B1 keeps all of them at
        30, so a change to one must move the rest.
        """
        import re

        from environments.shared.config import load_all_stages
        from environments.shared.curriculum import TASK_SUCCESS_GATE_KIND

        curriculum = load_all_stages("trex")[3]["curriculum_kwargs"]
        assert curriculum["gate_kind"] == TASK_SUCCESS_GATE_KIND
        assert "min_avg_forward_vel" not in curriculum
        assert "min_success_rate" not in curriculum
        # D-B2: 0.5 is PROVISIONAL until the first Phase-B pilot re-freezes it.
        assert curriculum["min_success_lcb"] == 0.5
        # D-B1: the declared panel n; every notebook evaluation_episodes= must agree.
        assert curriculum["min_eval_episodes"] == 30
        notebook = Path(__file__).resolve().parents[3] / "notebooks" / "sb3_training.ipynb"
        notebook_text = notebook.read_text(encoding="utf-8")
        declared = {int(n) for n in re.findall(r"evaluation_episodes=(\d+)", notebook_text)}
        assert declared == {curriculum["min_eval_episodes"]}, declared
        # The number the judge actually counts is the ROW COUNT of
        # evaluation_selected.csv, which the notebook's selected-checkpoint
        # panels (reporting.evaluate_stage_checkpoints since consolidation
        # PR-14c) roll with a literal n_episodes=; raising min_eval_episodes
        # without moving these would make every notebook hunt verdict refuse.
        import inspect

        from environments.shared.reporting import evaluate_stage_checkpoints

        evaluation = inspect.getsource(evaluate_stage_checkpoints)
        rolled = {int(n) for n in re.findall(r"n_episodes=(\d+)", notebook_text + evaluation)}
        assert rolled == {curriculum["min_eval_episodes"]}, rolled
        # The scheduler's hysteresis stays as an allowed key (D-B3); the
        # thresholds the manager extracts carry the bar.
        assert curriculum["required_consecutive"] == 3
        thresholds = thresholds_from_configs(load_all_stages("trex"))[3]
        assert thresholds["gate_kind"] == TASK_SUCCESS_GATE_KIND
        assert thresholds["min_success_lcb"] == 0.5
        assert thresholds["min_eval_episodes"] == 30
        assert StageThreshold(**thresholds).min_success_lcb == 0.5

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


class TestPublicationKeys:
    """``certification_seeds`` configures publication, not the gate (plan §4.5, decision D-B9)."""

    def test_it_is_accepted_beside_every_registered_gate_kind(self):
        """Every kind trex declares (stance, reward_and_length, task_success, recovery) plus the pilot."""
        from environments.shared.config import load_all_stages

        stages = load_all_stages("trex")
        seen = set()
        for stage, cfg in stages.items():
            block = {**cfg["curriculum_kwargs"], "certification_seeds": 3}
            kind = validate_gate_config(stage, block, advancement_enabled=isinstance(stage, int))
            seen.add(kind)
        assert validate_gate_config(1, {**_PILOT, "certification_seeds": 3}, advancement_enabled=False) == "none/v1"
        seen.add("none/v1")
        assert seen == set(GATE_KINDS)

    @pytest.mark.parametrize(
        "value", [0, -1, 1.5, True, "2", None], ids=["zero", "negative", "float", "bool", "str", "null"]
    )
    def test_it_must_be_a_positive_integer(self, value):
        with pytest.raises(GateSchemaError, match="stage 1: certification_seeds must be a positive integer"):
            validate_gate_config(1, {**_GATE, "min_avg_reward": 100.0, "certification_seeds": value})
        with pytest.raises(GateSchemaError, match="certification_seeds must be a positive integer"):
            declared_certification_seeds({"certification_seeds": value}, stage=1)

    def test_it_defaults_to_one(self):
        assert declared_certification_seeds({}) == 1
        assert declared_certification_seeds({**_GATE, "min_avg_reward": 100.0}) == 1
        assert declared_certification_seeds({"certification_seeds": 2}) == 2

    def test_it_is_not_a_threshold_and_enters_no_digest(self):
        """Never in a kind's threshold set, never in the gate view, never carried onto the threshold."""
        assert all("certification_seeds" not in keys for keys in GATE_KINDS.values())
        block = {**_GATE, "min_avg_reward": 100.0, "certification_seeds": 2}
        assert "certification_seeds" not in gate_config_view(block)["thresholds"]
        assert gate_config_sha256(gate_config_view(block)) == gate_config_sha256(
            gate_config_view({**_GATE, "min_avg_reward": 100.0})
        )
        thresholds = thresholds_from_configs({1: {"curriculum_kwargs": block}})
        assert "certification_seeds" not in thresholds[1]

    def test_trex_stance_declares_two_certification_seeds_outside_every_digest(self):
        """configs/trex/stance.toml declares 2 (plan A8; the seed 42/43/44 record in KNOWN_ISSUES) and
        every other committed stage keeps the default 1; the stance gate digest AND its recipe digest
        are exactly what they were without the key, so the certified stance trunk stays reusable."""
        from environments.shared.config import hyperparameters_sha256, load_all_stages
        from environments.shared.stage_manifest import _CONFIGS_DIR

        stance = load_all_stages("trex")[1]
        block = stance["curriculum_kwargs"]
        assert block["certification_seeds"] == 2 == declared_certification_seeds(block, stage=1)
        without = {key: value for key, value in block.items() if key != "certification_seeds"}
        assert gate_config_view(block) == gate_config_view(without)
        assert gate_config_sha256(gate_config_view(block)) == gate_config_sha256(gate_config_view(without))
        assert hyperparameters_sha256(stance, "PPO") == hyperparameters_sha256(
            {**stance, "curriculum_kwargs": without}, "PPO"
        )
        for species in sorted(path.parent.name for path in _CONFIGS_DIR.glob("*/stages.toml")):
            for stage, cfg in load_all_stages(species).items():
                if (species, stage) == ("trex", 1):
                    continue
                assert "certification_seeds" not in cfg["curriculum_kwargs"], (species, stage)
                assert declared_certification_seeds(cfg["curriculum_kwargs"], stage=stage) == 1
