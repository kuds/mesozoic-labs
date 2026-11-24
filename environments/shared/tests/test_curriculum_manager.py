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
    declared_certification_seeds,
    gate_config_sha256,
    gate_config_view,
    validate_gate_config,
    validate_gate_configs,
)
from environments.shared.model_id import ModelId

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

    @pytest.mark.parametrize("field", ["min_success_rate", "min_success_lcb"])
    def test_the_retired_success_fields_are_gone(self, field):
        """min_success_rate and task_success/v1's min_success_lcb left with the hunt nodes (decision D-D59)."""
        from dataclasses import fields

        assert field not in {f.name for f in fields(StageThreshold)}
        with pytest.raises(TypeError, match=field):
            StageThreshold(**{field: 0.5})


class TestCurriculumManager:
    """Test CurriculumManager lifecycle."""

    @pytest.fixture
    def manager(self):
        return CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            stage_thresholds={1: {"min_avg_forward_vel": 0.75}},
            start_stage=1,
        )

        assert manager.current_threshold.min_avg_forward_vel == 0.75

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

    def test_the_default_stage_count_is_the_manifests_advancing_ladder(self, manager):
        """Locomotion is the last advancing stage of every committed manifest since decision D-D59, so the
        default ``total_stages`` is 2, read from the manifest (it was a fixed 3, the hunt's stage included)."""
        assert manager.total_stages == 2
        assert CurriculumManager(model_id=ModelId("tyrannosaurus_rex", "anatomical")).total_stages == 2
        assert (
            CurriculumManager(model_id=ModelId("velociraptor_mongoliensis", "anatomical"), total_stages=1).total_stages
            == 1
        )

    def test_advance_to_final_stage(self, manager):
        manager.advance()
        assert manager.current_stage == 2
        assert manager.is_final_stage

    def test_advance_past_final_raises(self, manager):
        manager.advance()
        with pytest.raises(RuntimeError, match="Cannot advance past final stage"):
            manager.advance()

    def test_should_not_advance_on_final_stage(self, manager):
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
        assert (s["species"], s["embodiment"]) == ("velociraptor_mongoliensis", "anatomical")
        assert s["current_stage"] == 1
        assert len(s["eval_history"][1]) == 1

    def test_min_eval_episodes_enforced(self):
        """Threshold requires 5 episodes but we only provide 3."""
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            stage_thresholds={
                1: {"min_avg_reward": 0.0, "min_eval_episodes": 5, "required_consecutive": 1},
            },
        )
        # Only 3 episodes provided
        assert not mgr.should_advance([100.0, 100.0, 100.0], [500.0, 500.0, 500.0])

    def test_forward_vel_gate_blocks_without_velocity(self):
        """Stage with forward velocity threshold should block if velocity is too low."""
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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

    def test_no_success_sample_is_taken(self):
        """No gate kind consumes task success since decision D-D59: the manager takes no success argument."""
        import inspect

        for method in (CurriculumManager.should_advance, CurriculumManager.record_eval):
            assert "success_rates" not in inspect.signature(method).parameters
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            stage_thresholds={
                1: {
                    "min_avg_reward": 10.0,
                    "min_avg_episode_length": 50,
                    "min_eval_episodes": 3,
                    "required_consecutive": 1,
                },
            },
        )
        assert mgr.should_advance([100.0] * 3, [500.0] * 3)
        assert "mean_success_rate" not in mgr.summary()["eval_history"][1][-1]

    def test_record_eval_with_forward_vel(self):
        """record_eval should include forward velocity in summary."""
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            stage_thresholds={1: {"min_avg_reward": 0.0, "required_consecutive": 1}},
        )
        summary = mgr.record_eval([10.0, 20.0], [100.0, 200.0], forward_velocities=[1.0, 2.0])
        assert summary["mean_forward_vel"] == 1.5


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
                    min_eval_episodes=12,
                    required_consecutive=3,
                ),
            },
        }
        thresholds = thresholds_from_configs(configs)
        assert thresholds[1]["min_avg_forward_vel"] == 0.5
        assert thresholds[1]["min_eval_episodes"] == 12

    def test_a_retired_task_success_config_is_refused_naming_the_decision(self):
        """task_success/v1 and its bar were deleted with the hunt nodes (decision D-D59); no threshold is built."""
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
        with pytest.raises(GateSchemaError, match=r"'task_success/v1' was removed with the hunt nodes .*D-D59"):
            thresholds_from_configs(configs)

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
        path advancing on any evaluation while the JAX path (retired by D-D17)
        raised. The schema now requires each gate kind's core field, so it is refused.
        """
        configs = {1: {"curriculum_kwargs": dict(_GATE)}}
        with pytest.raises(GateSchemaError, match="missing required threshold"):
            thresholds_from_configs(configs)
        # Malformed is malformed even when the run cannot advance.
        with pytest.raises(GateSchemaError, match="missing required threshold"):
            thresholds_from_configs(configs, advancement_enabled=False)

    def test_every_committed_stage_config_declares_a_valid_gate(self):
        """The shipped configs must satisfy the schema on every species.

        Asserts every declared kind is one the registry knows and SB3
        evaluates, rather than pinning the specific kind each stage
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
        from environments.shared.species_names import model_ids

        # Every model, both Compsognathus embodiments included: D-B14 keeps
        # them on reward_and_length/v1, and the loop must see them so a kind
        # drift there fails here too.
        assert len(model_ids()) == 7
        for model_id in model_ids():
            stages = load_all_stages(model_id)
            chain = {stage: cfg for stage, cfg in stages.items() if isinstance(stage, int)}
            kinds = validate_gate_configs(chain)
            assert set(kinds.values()) <= set(GATE_KINDS), model_id
            # "none/v1" refuses to advance, so a stage on the advancing
            # chain must never declare it.
            assert "none/v1" not in set(kinds.values()), model_id
            pilots = {stage: cfg for stage, cfg in stages.items() if isinstance(stage, str)}
            pilot_kinds = validate_gate_configs(pilots, advancement_enabled=False)
            assert set(pilot_kinds.values()) <= set(GATE_KINDS), model_id

    def test_tyrannosaurus_rex_stage1_gates_on_stance_quality(self):
        """T-Rex 1a must not be gated on return: a statue is the reward optimum."""
        from environments.shared.config import load_all_stages

        curriculum = load_all_stages(ModelId("tyrannosaurus_rex", "anatomical"))[1]["curriculum_kwargs"]
        # The six-heading certificate since D-D32, stance_quality/v2 from physics r8 (D-D24) and
        # stance_quality/v1 before it: a stance kind every time.
        assert curriculum["gate_kind"] == "stance_quality_heading/v1"
        # The statue scores 4372.7 at 1000.0 steps on the r12 plant under the LTSB stance task with heading_weight 0
        # (plant_versions note 24; 4470.6 on r11, the stance-landing decision of 2026-11-06; 3766.6 on the D-D27 task and
        # physics r10, 3756.8 through r9, 3766.1 on the r8 task, 3271.8 when the reward gate was retired), so the retired
        # reward and length criteria were both cleared by doing nothing.
        assert curriculum["min_avg_reward"] < curriculum["collapse_peak_floor_reference"] == 4372.7, (
            "reward must be a rail below the statue, not a gate"
        )
        assert "min_avg_episode_length" not in curriculum
        # The bound's power is specified at this panel size.
        assert curriculum["min_eval_episodes"] == 40

    def test_no_committed_stage_declares_a_retired_kind_or_key(self):
        """task_success/v1 and min_success_rate left with every hunt (decision D-D59), and no stage names them."""
        from environments.shared.config import load_all_stages
        from environments.shared.curriculum.gate_schema import (
            GATE_KINDS,
            RETIRED_CURRICULUM_KEYS,
            RETIRED_GATE_KINDS,
            retired_gate_declarations,
        )
        from environments.shared.species_names import model_ids

        assert set(RETIRED_GATE_KINDS) == {"task_success/v1"}
        assert set(RETIRED_CURRICULUM_KEYS) == {"min_success_rate", "min_success_lcb", "task_success_panel_episodes"}
        # A retired name is no longer registered, so nothing can declare it as live.
        assert not set(RETIRED_GATE_KINDS) & set(GATE_KINDS)
        assert not set(RETIRED_CURRICULUM_KEYS) & frozenset().union(*GATE_KINDS.values())
        for model_id in model_ids():
            for stage, cfg in load_all_stages(model_id).items():
                assert retired_gate_declarations(cfg["curriculum_kwargs"]) == [], (model_id, stage)

    def test_with_real_configs(self):
        """Integration test: extract thresholds from actual TOML configs."""
        from environments.shared.config import load_all_stages

        configs = load_all_stages(ModelId("velociraptor_mongoliensis", "anatomical"))
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
        """Every kind a committed stage declares, plus a synthetic block for each kind none declares yet.

        The committed blocks of every species (stance v1, reward_and_length,
        recovery) are validated as they stand; the pilot,
        ``stance_quality/v2`` -- registered by D-D23 before any stage adopts
        it -- the six-heading certificate (D-D32) and ``command_tracking/v1``
        (D-D61, both profiles at their provisional values, each with a biped's
        and a quadruped's provisional gait clause) get a minimal
        schema-valid block each, so the loop still covers exactly
        ``GATE_KINDS`` whichever kinds the TOMLs happen to declare.
        """
        from environments.shared.config import load_all_stages
        from environments.shared.curriculum.command_tracking_gate import (
            COMMAND_TRACKING_GATE_KIND,
            PROVISIONAL_BIPED_GAIT_THRESHOLDS,
            PROVISIONAL_FOLLOW_THRESHOLDS,
            PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS,
            PROVISIONAL_WALK_THRESHOLDS,
        )
        from environments.shared.curriculum.stance_gate_v2 import STANCE_GATE_V2_KIND, STANCE_V2_REQUIRED_KEYS
        from environments.shared.curriculum.stance_heading_gate import STANCE_HEADING_GATE_KIND
        from environments.shared.species_names import model_ids

        seen = set()
        for model_id in model_ids():
            for stage, cfg in load_all_stages(model_id).items():
                block = {**cfg["curriculum_kwargs"], "certification_seeds": 3}
                kind = validate_gate_config(stage, block, advancement_enabled=isinstance(stage, int))
                seen.add(kind)
        assert validate_gate_config(1, {**_PILOT, "certification_seeds": 3}, advancement_enabled=False) == "none/v1"
        seen.add("none/v1")
        synthetic = {
            kind: {
                "gate_schema_version": GATE_SCHEMA_VERSION,
                "gate_kind": kind,
                **{
                    key: 200 if key == "settle_steps" else 40 if key == "min_eval_episodes" else 0.5
                    for key in STANCE_V2_REQUIRED_KEYS
                },
            }
            for kind in (STANCE_GATE_V2_KIND, STANCE_HEADING_GATE_KIND)
        }
        for kind, block in synthetic.items():
            if kind not in seen:
                assert validate_gate_config(1, {**block, "certification_seeds": 3}) == kind
                seen.add(kind)
        for profile_block in (PROVISIONAL_WALK_THRESHOLDS, PROVISIONAL_FOLLOW_THRESHOLDS):
            for gait_block in (PROVISIONAL_BIPED_GAIT_THRESHOLDS, PROVISIONAL_QUADRUPED_GAIT_THRESHOLDS):
                block = {
                    "gate_schema_version": GATE_SCHEMA_VERSION,
                    "gate_kind": COMMAND_TRACKING_GATE_KIND,
                    **profile_block,
                    **gait_block,
                    "certification_seeds": 3,
                }
                assert validate_gate_config(1, block) == COMMAND_TRACKING_GATE_KIND
        seen.add(COMMAND_TRACKING_GATE_KIND)
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

    def test_the_two_biped_stances_declare_two_certification_seeds_outside_every_digest(self):
        """configs/tyrannosaurus_rex/anatomical/stance.toml declares 2 (plan A8; the seed 42/43/44 record in KNOWN_ISSUES),
        and so does the velociraptor's since 2026-11-06; every other committed stage keeps the default 1. Each stance's
        gate digest AND its recipe digest are exactly what they were without the key, so a certified stance trunk
        stays reusable."""
        from environments.shared.config import hyperparameters_sha256, load_all_stages
        from environments.shared.species_names import model_ids

        declaring = {
            (ModelId("tyrannosaurus_rex", "anatomical"), 1),
            (ModelId("velociraptor_mongoliensis", "anatomical"), 1),
        }
        for model_id, stage in sorted(declaring):
            stance = load_all_stages(model_id)[stage]
            block = stance["curriculum_kwargs"]
            assert block["certification_seeds"] == 2 == declared_certification_seeds(block, stage=stage)
            without = {key: value for key, value in block.items() if key != "certification_seeds"}
            assert gate_config_view(block) == gate_config_view(without)
            assert gate_config_sha256(gate_config_view(block)) == gate_config_sha256(gate_config_view(without))
            assert hyperparameters_sha256(stance, "PPO") == hyperparameters_sha256(
                {**stance, "curriculum_kwargs": without}, "PPO"
            )
        models = model_ids()
        assert models, "the manifest lists no model"
        for model_id in models:
            for stage, cfg in load_all_stages(model_id).items():
                if (model_id, stage) in declaring:
                    continue
                assert "certification_seeds" not in cfg["curriculum_kwargs"], (model_id, stage)
                assert declared_certification_seeds(cfg["curriculum_kwargs"], stage=stage) == 1
