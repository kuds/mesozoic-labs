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

import ast
import inspect
import logging
from pathlib import Path

import pytest

from environments.shared.curriculum import CurriculumManager, thresholds_from_configs
from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, GateSchemaError
from environments.shared.curriculum.task_success_gate import TASK_SUCCESS_GATE_KIND
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

#: A schema-valid task_success/v1 declaration at the plan's provisional bar
#: (D-B2: 0.5 at n=30, 20/30 clears it and 19/30 does not) with the reward
#: rail the hunting statue clears.
_TASK_SUCCESS = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": TASK_SUCCESS_GATE_KIND,
    "min_success_lcb": 0.5,
    "min_eval_episodes": 30,
    "min_avg_reward": 100.0,
}

#: Evaluation results no criterion could reject — if the stage advances on
#: these, it advanced on evidence nobody checked.
_SKY_HIGH_REWARDS = [1e9] * 10
_FULL_LENGTHS = [1000.0] * 10
_SKY_HIGH_PANEL = [1e9] * 30
_FULL_PANEL = [1000.0] * 30


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

    def test_task_success_never_advances_on_sky_high_reward(self):
        """Plan §4.4: the hunting gate is the binomial bound, never the reward.

        10/30 successes bound at 0.21 and must refuse beside a reward of 1e9;
        20/30 (bound 0.5006) with hysteresis 1 advances.
        """
        thresholds = thresholds_from_configs({1: {"curriculum_kwargs": dict(_TASK_SUCCESS, required_consecutive=1)}})
        mgr = CurriculumManager(species="velociraptor", stage_thresholds=thresholds)
        low = [1.0] * 10 + [0.0] * 20
        for _ in range(3):
            assert not mgr.should_advance(_SKY_HIGH_PANEL, _FULL_PANEL, success_rates=low)
        assert mgr.summary()["consecutive_passes"][1] == 0
        latest = mgr.summary()["eval_history"][1][-1]
        assert (latest["success_count"], latest["n_success_samples"]) == (10, 30)

        mgr = CurriculumManager(species="velociraptor", stage_thresholds=thresholds)
        assert mgr.should_advance(_SKY_HIGH_PANEL, _FULL_PANEL, success_rates=[1.0] * 20 + [0.0] * 10)

    def test_task_success_refuses_without_a_success_sample(self, caplog):
        thresholds = thresholds_from_configs({1: {"curriculum_kwargs": dict(_TASK_SUCCESS, required_consecutive=1)}})
        mgr = CurriculumManager(species="velociraptor", stage_thresholds=thresholds)
        with caplog.at_level(logging.WARNING, logger="environments.shared.curriculum.manager"):
            assert not mgr.should_advance(_SKY_HIGH_PANEL, _FULL_PANEL)
        assert "no per-episode success sample" in caplog.text
        assert mgr.summary()["consecutive_passes"][1] == 0

    def test_task_success_refuses_a_panel_below_the_declared_size(self):
        """19/19 bounds at 0.85 but the bound's power is specified at n=30."""
        thresholds = thresholds_from_configs({1: {"curriculum_kwargs": dict(_TASK_SUCCESS, required_consecutive=1)}})
        mgr = CurriculumManager(species="velociraptor", stage_thresholds=thresholds)
        assert not mgr.should_advance([1e9] * 19, [1000.0] * 19, success_rates=[1.0] * 19)

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


class TestTaskSuccessSchemaRequiresTheBarAndThePanelSize:
    """min_eval_episodes is REQUIRED for the kind (unlike every other), because the bound's power is
    a function of the declared n: without the pin, thresholds_from_configs would leave the manager
    at DEFAULT_MIN_EVAL_EPISODES = 10 and a 9/10 panel (bound 0.61) would clear a 0.5 bar in-training
    while the post-stage judge refused the same config."""

    @pytest.mark.parametrize("missing", ["min_eval_episodes", "min_success_lcb"])
    def test_the_schema_names_the_missing_key(self, missing):
        from environments.shared.curriculum.gate_schema import validate_gate_config

        block = {key: value for key, value in _TASK_SUCCESS.items() if key != missing}
        with pytest.raises(GateSchemaError, match=rf"missing required threshold field\(s\) \['{missing}'\]"):
            validate_gate_config(3, block, advancement_enabled=True)
        with pytest.raises(GateSchemaError, match=missing):
            thresholds_from_configs({3: {"curriculum_kwargs": block}}, advancement_enabled=True)


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


class TestJaxEvalGateRefusesUnevaluatableKinds:
    """F2 on the OTHER JAX path (review J #10).

    ``jax_eval.check_stage_gate_for_config`` is what ``run_stage_evaluation``
    writes into ``publication_gate_passed``.  It refused nothing the schema
    accepted, so a recovery stage -- reachable once ``jax_setup`` took the
    semantic id -- fell through to the scalar check with every threshold
    unset and passed on evidence nobody checked.  It must refuse exactly what
    ``check_stage_gate`` refuses, for exactly the same reason.
    """

    def _results(self, reward):
        from environments.shared.jax_eval import EvalResults

        results = EvalResults()
        results.rewards = [reward] * 10
        results.lengths = [1000] * 10
        results.forward_vels = [0.0] * 10
        results.distances = [0.0] * 10
        results.successes = [False] * 10
        return results

    def test_recovery_quality_is_refused_with_the_reason_as_the_verdict(self):
        from environments.shared.jax_eval import check_stage_gate_for_config

        config = {"stage": 1, "curriculum_kwargs": dict(_RECOVERY)}
        passed, failures = check_stage_gate_for_config(self._results(1e9), config)
        assert passed is False
        assert len(failures) == 1
        assert "recovery_quality/v1" in failures[0]
        assert "cannot evaluate" in failures[0]

    def test_the_optional_reward_rail_does_not_convert_it_into_a_reward_gate(self):
        from environments.shared.jax_eval import check_stage_gate_for_config

        config = {"stage": 1, "curriculum_kwargs": dict(_RECOVERY, min_avg_reward=100.0)}
        passed, _ = check_stage_gate_for_config(self._results(1e9), config)
        assert passed is False

    def test_both_jax_gates_share_one_refusal(self):
        """One predicate, one message: jax_curriculum.unevaluable_gate_kind_reason."""
        from environments.shared.jax_eval import check_stage_gate_for_config

        config = {"stage": 1, "curriculum_kwargs": dict(_RECOVERY)}
        _, failures = check_stage_gate_for_config(self._results(1e9), config)
        with pytest.raises(GateSchemaError) as excinfo:
            check_stage_gate({"mean_episode_return": 1e9, "mean_episode_length": 1000.0}, config)
        assert failures == [str(excinfo.value)]

    def test_the_evaluable_kinds_are_one_set_for_both_paths(self):
        from environments.shared.curriculum.stance_gate import STANCE_GATE_KIND
        from environments.shared.jax_curriculum import unevaluable_gate_kind_reason

        assert unevaluable_gate_kind_reason(1, "reward_and_length/v1") is None
        assert unevaluable_gate_kind_reason(1, STANCE_GATE_KIND) is None
        reason = unevaluable_gate_kind_reason(1, "recovery_quality/v1")
        assert reason is not None and "recovery_quality/v1" in reason
        # task_success/v1 stays OUT of the evaluable set (plan A7): its
        # verdict comes from the SB3 evidence CSV, and no MJX hunting panel
        # exists.  The refusal names the kind and where the verdict comes from.
        reason = unevaluable_gate_kind_reason(3, TASK_SUCCESS_GATE_KIND)
        assert reason is not None and TASK_SUCCESS_GATE_KIND in reason
        assert "evaluation_selected.csv" in reason and "cannot evaluate" in reason
        # The generic sentence is kind-neutral: the recovery-specific text
        # travels only with the recovery refusal.
        assert "gate resolver" not in reason


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

    def test_a_task_success_final_stage_is_rejected_before_any_training(self, monkeypatch):
        """Decision D-B13: a FINAL stage no JAX path can judge refuses up front.

        A JAX trex chain ending on task_success/v1 would otherwise spend the
        hunting stage's 8M steps for the CPU eval to record a refusal.
        """
        self._patch_loader(monkeypatch, _TASK_SUCCESS)
        trained: list[int] = []

        def mock_train(species, stage, **kwargs):
            trained.append(stage)
            return {"w": 1.0}, {"mean_episode_return": 1e9}

        with pytest.raises(GateSchemaError, match="D-B13") as excinfo:
            run_curriculum("trex", mock_train, stages=(3,))
        assert TASK_SUCCESS_GATE_KIND in str(excinfo.value) and "cannot evaluate" in str(excinfo.value)
        assert trained == []

    def test_the_final_stage_block_is_not_schema_validated_by_the_preflight(self, monkeypatch):
        """D-B13 refuses on the declared KIND only: a single-stage pilot whose final block would not
        validate under advancement (an evaluable kind missing its threshold, an unregistered kind, no
        declaration at all) keeps training exactly as it did before the decision."""
        for block in (
            {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "reward_and_length/v1"},
            {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "stance_quality/v1"},
            {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "not_a_kind/v9"},
            {},
        ):
            self._patch_loader(monkeypatch, block)
            trained: list[int] = []

            def mock_train(species, stage, **kwargs):
                trained.append(stage)
                return {"w": 1.0}, {"mean_episode_return": 1e9}

            results = run_curriculum("trex", mock_train, stages=(1,))
            assert trained == [1] and 1 in results, block

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


def _selected_evidence(stage_dir: Path, successes: list[bool], *, checkpoint: str = "robust_best_model") -> Path:
    """A handoff pair plus an evaluation_selected.csv bound to it, as the trainers write them."""
    from environments.shared.reporting import save_evaluation_episodes

    models = stage_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    zip_path = models / f"{checkpoint}.zip"
    zip_path.write_bytes(f"{checkpoint} weights".encode())
    vecnorm = models / f"{checkpoint}_vecnorm.pkl"
    vecnorm.write_bytes(b"statistics")
    n = len(successes)
    return save_evaluation_episodes(
        stage_dir,
        rewards=[600.0] * n,
        lengths=[1000] * n,
        forward_velocities=[1.5] * n,
        distances=[10.0] * n,
        successes=successes,
        evaluation_seed=3042,
        checkpoint_label="selected",
        checkpoint_path=zip_path,
        normalization_path=vecnorm,
    )


class TestTaskSuccessGateIsConsulted:
    """Plan §8 invariant 10 for task_success/v1: the fail-closed dispatch test.

    Registering the kind in GATE_KINDS without the judge arm routed a
    hunting stage to the reward conjunction, which certifies on
    ``min_avg_reward`` — a collapse rail the ~600 statue clears.  These pin
    that the bound is consulted, that the reward cannot substitute for it,
    and that the NEXT registered kind fails closed by construction.
    """

    STAGE_RESULTS = {"best_model_reward": 1e9, "best_model_length": 1000.0}

    def _judge(self, stage_dir, curriculum=None):
        from environments.shared.reporting import evaluate_stage_gate

        return evaluate_stage_gate(curriculum or dict(_TASK_SUCCESS), self.STAGE_RESULTS, stage=3, stage_dir=stage_dir)

    def test_sky_high_reward_with_no_success_evidence_is_refused(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        (models / "best_model.zip").write_bytes(b"w")
        (models / "best_model_vecnorm.pkl").write_bytes(b"s")
        passed, failures = self._judge(tmp_path)
        assert passed is False
        assert any("evaluation_selected.csv" in failure and "absent" in failure for failure in failures)

    def test_no_stage_dir_is_refused(self):
        passed, failures = self._judge(None)
        assert passed is False
        assert any("no stage_dir was given" in failure for failure in failures)

    def test_a_low_lcb_is_refused_despite_reward_far_above_the_rail(self, tmp_path):
        _selected_evidence(tmp_path, [True] * 19 + [False] * 11)
        passed, failures = self._judge(tmp_path)
        assert passed is False
        assert any("task_success_lcb" in failure and "19/30" in failure for failure in failures)

    def test_twenty_of_thirty_passes_at_the_provisional_bar(self, tmp_path):
        _selected_evidence(tmp_path, [True] * 20 + [False] * 10)
        assert self._judge(tmp_path) == (True, [])

    def test_the_plan_sizing_table_pins_binomial_lcb(self):
        """Plan §4.4: ≥ 0.5 needs 20/30 or 26/40; the committed 29/30 bounds at 0.851."""
        from environments.shared.curriculum.recovery_gate import binomial_lcb

        assert binomial_lcb(20, 30) >= 0.5
        assert binomial_lcb(20, 30) == pytest.approx(0.5006, abs=1e-3)
        assert binomial_lcb(19, 30) < 0.5
        assert binomial_lcb(26, 40) >= 0.5
        assert binomial_lcb(25, 40) < 0.5
        assert binomial_lcb(29, 30) == pytest.approx(0.851, abs=1e-3)

    def test_a_registered_kind_without_an_evaluator_is_refused_not_routed_to_the_reward_arm(self, monkeypatch):
        from environments.shared.curriculum import gate_schema
        from environments.shared.reporting import evaluate_stage_gate

        monkeypatch.setitem(gate_schema.GATE_KINDS, "made_up/v1", frozenset({"min_avg_reward"}))
        passed, failures = evaluate_stage_gate(
            {"gate_kind": "made_up/v1", "gate_schema_version": 1, "min_avg_reward": 100.0},
            self.STAGE_RESULTS,
            stage=3,
        )
        assert passed is False
        assert any("has no evaluator" in failure and "made_up/v1" in failure for failure in failures)

    def test_what_would_have_to_be_deleted_for_the_lcb_to_stop_being_consulted(self, tmp_path, monkeypatch):
        """The consulted test (plan §8 invariant 10): the call chain, by AST and at runtime.

        evaluate_stage_gate dispatches on TASK_SUCCESS_GATE_KIND into
        _task_success_stage_gate BEFORE the reward_and_length return;
        _task_success_stage_gate calls evaluate_task_success_gate; and that
        calls binomial_lcb.  Then the runtime half: with binomial_lcb raising,
        judging a valid hunting panel raises — the bound was consulted.
        """
        from environments.shared.curriculum import task_success_gate
        from environments.shared.reporting import gates

        module = ast.parse(inspect.getsource(gates))
        functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        dispatch = functions["evaluate_stage_gate"]
        task_success_branch = None
        reward_return = None
        for index, node in enumerate(ast.walk(dispatch)):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                if {"gate_kind", "TASK_SUCCESS_GATE_KIND"} <= names:
                    calls = {
                        c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    }
                    assert "_task_success_stage_gate" in calls
                    task_success_branch = node.lineno
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "_reward_and_length_stage_gate":
                    reward_return = node.lineno
        assert task_success_branch is not None and reward_return is not None
        assert task_success_branch < reward_return

        arm_calls = {
            c.func.id
            for c in ast.walk(functions["_task_success_stage_gate"])
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        assert "evaluate_task_success_gate" in arm_calls

        gate_module = ast.parse(inspect.getsource(task_success_gate))
        evaluator = next(
            node
            for node in gate_module.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_task_success_gate"
        )
        evaluator_calls = {
            c.func.id for c in ast.walk(evaluator) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        assert "binomial_lcb" in evaluator_calls

        _selected_evidence(tmp_path, [True] * 20 + [False] * 10)

        def consulted(*args, **kwargs):
            raise RuntimeError("consulted")

        monkeypatch.setattr(task_success_gate, "binomial_lcb", consulted)
        with pytest.raises(RuntimeError, match="consulted"):
            self._judge(tmp_path)
