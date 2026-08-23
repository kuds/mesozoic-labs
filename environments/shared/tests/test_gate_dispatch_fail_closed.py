"""Both backends must refuse gate kinds they cannot evaluate in-training.

TREX_REVIEW_2026_08 §3.1 (F1/F2): ``CurriculumManager.should_advance`` and the
JAX ``check_stage_gate`` both treated the reward-and-length evaluator as the
fall-through for every non-stance kind.  A schema-valid ``recovery_quality/v1``
stage therefore advanced on reward alone under ``StageThreshold``'s permissive
defaults (``min_avg_reward = -inf``) on SB3, and on JAX either crashed on a
missing threshold key after the stage's whole budget or advanced on its
optional reward rail — while ``reporting/gates.py`` explicitly refused the
same fall-through.  These tests pin the refusal on both backends, and pin that
the JAX path refuses BEFORE any training compute is spent.
"""

from __future__ import annotations

import logging

import pytest

from environments.shared.curriculum import CurriculumManager, thresholds_from_configs
from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, GateSchemaError
from environments.shared.jax_curriculum import check_stage_gate, run_curriculum

#: A schema-valid recovery gate declaration.  Threshold values are stand-ins:
#: they are irrelevant here because the dispatch must refuse before reading
#: any of them — the real ones land with the P3/P5 calibration.
_RECOVERY = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": "recovery_quality/v1",
    "min_recovery_success_lcb": 0.70,
    "recovery_t_recover_steps": 66,
    "recovery_dwell_steps": 66,
}

#: Evaluation results no criterion could reject — if the stage advances on
#: these, it advanced on evidence nobody checked.
_SKY_HIGH_REWARDS = [1e9] * 10
_FULL_LENGTHS = [1000.0] * 10


def _manager(gate_kind: str) -> CurriculumManager:
    return CurriculumManager(
        species="velociraptor",
        stage_thresholds={
            1: {
                "gate_kind": gate_kind,
                # The laxest shared settings the manager accepts, so only the
                # dispatch itself can be what refuses.
                "min_eval_episodes": 1,
                "required_consecutive": 1,
            },
        },
    )


class TestManagerRefusesUnevaluatableKinds:
    """F1: the SB3 advancement engine must fail closed, not fall through."""

    def test_recovery_quality_never_advances_on_sky_high_reward(self):
        mgr = _manager("recovery_quality/v1")
        for _ in range(5):
            assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)
        # The refusal must not even count toward the consecutive-pass streak.
        assert mgr.summary()["consecutive_passes"][1] == 0

    def test_the_refusal_is_logged_naming_the_kind(self, caplog):
        mgr = _manager("recovery_quality/v1")
        with caplog.at_level(logging.ERROR, logger="environments.shared.curriculum.manager"):
            assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)
        assert "recovery_quality/v1" in caplog.text
        assert "cannot evaluate" in caplog.text

    def test_a_made_up_kind_also_refuses(self):
        mgr = _manager("made_up/v9")
        for _ in range(5):
            assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)

    def test_a_schema_valid_recovery_config_still_refuses_end_to_end(self):
        """The exact F1 scenario: P5 flips a config's gate kind to recovery.

        The schema validates it (recovery_quality/v1 is a known kind with its
        required fields present), ``thresholds_from_configs`` carries the kind
        onto the threshold, and the manager must then refuse to advance rather
        than evaluate the reward gate its defaults would trivially pass.
        """
        configs = {1: {"curriculum_kwargs": dict(_RECOVERY, min_eval_episodes=1, required_consecutive=1)}}
        thresholds = thresholds_from_configs(configs, advancement_enabled=True)
        assert thresholds[1]["gate_kind"] == "recovery_quality/v1"

        mgr = CurriculumManager(species="velociraptor", stage_thresholds=thresholds)
        for _ in range(5):
            assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)


class TestJaxCheckStageGateRefusesUnevaluatableKinds:
    """F2: the JAX gate must refuse explicitly, not fall through to reward."""

    def test_recovery_quality_is_refused_even_with_generous_metrics(self):
        config = {"stage": 1, "curriculum_kwargs": dict(_RECOVERY)}
        with pytest.raises(GateSchemaError, match="cannot evaluate"):
            check_stage_gate({"mean_episode_return": 1e9, "mean_episode_length": 1000.0}, config)

    def test_the_optional_reward_rail_does_not_convert_it_into_a_reward_gate(self):
        """With the rail present the old code advanced on reward alone."""
        config = {"stage": 1, "curriculum_kwargs": dict(_RECOVERY, min_avg_reward=100.0)}
        with pytest.raises(GateSchemaError, match="cannot evaluate"):
            check_stage_gate({"mean_episode_return": 1e9, "mean_episode_length": 1000.0}, config)

    def test_a_made_up_kind_is_refused_by_the_schema(self):
        bogus = {"curriculum_kwargs": {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "made_up/v9"}}
        with pytest.raises(GateSchemaError, match="unknown gate_kind"):
            check_stage_gate({"mean_episode_return": 1e9}, bogus)


class TestRunCurriculumFailsFastBeforeTraining:
    """F2's expensive half: the refusal must land before the budget is spent.

    ``check_stage_gate`` runs only after a stage trains, so without a
    pre-flight check a recovery-gated stage burned its full budget before the
    verdict turned out to be uncomputable.
    """

    @staticmethod
    def _patch_loader(monkeypatch, curriculum_kwargs):
        import environments.shared.jax_curriculum as jc

        def mock_load(species, stage):
            return {
                "stage": stage,
                "jax_kwargs": {},
                "env_kwargs": {},
                "curriculum_kwargs": dict(curriculum_kwargs),
            }

        monkeypatch.setattr(jc, "load_stage_config", mock_load)

    def test_a_recovery_gated_stage_is_rejected_before_any_training(self, monkeypatch):
        self._patch_loader(monkeypatch, _RECOVERY)
        trained: list[int] = []

        def mock_train(species, stage, **kwargs):
            trained.append(stage)
            return {"w": 1.0}, {"mean_episode_return": 1e9}

        with pytest.raises(GateSchemaError, match="cannot evaluate"):
            run_curriculum("trex", mock_train, stages=(1, 2))
        assert trained == []

    def test_a_none_gate_on_a_gated_stage_is_rejected_before_any_training(self, monkeypatch):
        self._patch_loader(monkeypatch, {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "none/v1"})
        trained: list[int] = []

        def mock_train(species, stage, **kwargs):
            trained.append(stage)
            return {"w": 1.0}, {"mean_episode_return": 1e9}

        with pytest.raises(GateSchemaError, match="non-advancing pilot"):
            run_curriculum("trex", mock_train, stages=(1, 2))
        assert trained == []

    def test_a_single_stage_pilot_still_trains(self, monkeypatch):
        """The final stage's gate is never evaluated, so it is not pre-checked.

        A one-stage recovery pilot (the DESIGNED state until P5 lands measured
        thresholds) must keep training; only stages whose gate would actually
        be checked are validated up front.
        """
        self._patch_loader(monkeypatch, _RECOVERY)
        trained: list[int] = []

        def mock_train(species, stage, **kwargs):
            trained.append(stage)
            return {"w": 1.0}, {"mean_episode_return": 1e9}

        results = run_curriculum("trex", mock_train, stages=(1,))
        assert trained == [1]
        assert 1 in results
