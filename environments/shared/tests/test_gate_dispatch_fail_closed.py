"""The in-training gate must refuse gate kinds it cannot evaluate.

TREX_REVIEW_2026_08 §3.1 (F1/F2): ``CurriculumManager.should_advance`` and the
JAX ``check_stage_gate`` both treated the reward-and-length evaluator as the
fall-through for every non-stance kind.  A schema-valid ``recovery_quality/v1``
stage therefore advanced on reward alone under ``StageThreshold``'s permissive
defaults (``min_avg_reward = -inf``) on SB3, and on JAX either crashed on a
missing threshold key after the stage's whole budget or advanced on its
optional reward rail — while ``reporting/gates.py`` explicitly refused the
same fall-through.  These tests pin the SB3 refusal; the JAX half left with
the JAX runtime (decision D-D17, cleanup PR-B).
"""

from __future__ import annotations

import ast
import inspect
import logging
import re

import pytest

from environments.shared.curriculum import CurriculumManager, thresholds_from_configs
from environments.shared.curriculum.gate_schema import GATE_SCHEMA_VERSION, GateSchemaError
from environments.shared.model_id import ModelId

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

#: The T. rex bite's frozen gate block as its stage_config.json recorded it
#: (task_success/v1 at the provisional D-B2 bar, with the reward rail its
#: hunting statue cleared).  The kind and its bar were removed with the hunt
#: nodes (decision D-D59); every reader must refuse the record naming that
#: decision, never judge it as an unknown kind or on its reward rail.
_RETIRED_TASK_SUCCESS = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": "task_success/v1",
    "min_success_lcb": 0.5,
    "min_eval_episodes": 30,
    "min_avg_reward": 100.0,
}

#: A hunt stage's frozen reward_and_length/v1 block with its success floor
#: (the velociraptor strike's shape), every other threshold trivially met.
_RETIRED_SUCCESS_FLOOR = {
    "gate_schema_version": GATE_SCHEMA_VERSION,
    "gate_kind": "reward_and_length/v1",
    "min_avg_reward": 100.0,
    "min_success_rate": 0.5,
}

#: The refusal every reader gives for the retired kind and the retired key.
_RETIRED_KIND_MESSAGE = r"gate_kind 'task_success/v1' was removed with the hunt nodes \(decision D-D59\)"
_RETIRED_KEY_MESSAGE = r"'min_success_rate' was removed with the hunt nodes \(decision D-D59\)"

#: Evaluation results no criterion could reject — if the stage advances on
#: these, it advanced on evidence nobody checked.
_SKY_HIGH_REWARDS = [1e9] * 10
_FULL_LENGTHS = [1000.0] * 10
_SKY_HIGH_PANEL = [1e9] * 30
_FULL_PANEL = [1000.0] * 30


def _manager(gate_kind: str) -> CurriculumManager:
    return CurriculumManager(
        model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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

    def test_a_retired_task_success_threshold_never_advances_and_names_the_decision(self, caplog):
        """A threshold built by hand past the schema still refuses, naming D-D59 rather than an unknown kind."""
        mgr = _manager("task_success/v1")
        with caplog.at_level(logging.ERROR, logger="environments.shared.curriculum.manager"):
            for _ in range(3):
                assert not mgr.should_advance(_SKY_HIGH_PANEL, _FULL_PANEL)
        assert mgr.summary()["consecutive_passes"][1] == 0
        assert "decision D-D59" in caplog.text

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

        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"), stage_thresholds=thresholds
        )
        for _ in range(5):
            assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)


class TestRetiredTaskSuccessGateFailsClosed:
    """task_success/v1 and min_success_rate were deleted with the hunt nodes (decision D-D59).

    A stage TOML or a frozen record (a ``stage_config.json``) that still
    names either must be refused by every reader, naming the decision: the
    schema at config load, the in-training thresholds, the post-stage judge
    and the recorded-history reading.  The failure mode this guards is the
    fall-through: an unregistered kind read as merely unknown, and a
    reward_and_length/v1 block judged on the floors it has left once its
    success floor is no longer read — a hunt certified on its reward rail.
    """

    STAGE_RESULTS = {
        "best_model_reward": 1e9,
        "best_model_length": 1000.0,
        "best_model_success_rate": 1.0,
    }

    def test_the_schema_refuses_the_kind_naming_the_decision(self):
        from environments.shared.curriculum.gate_schema import validate_gate_config

        with pytest.raises(GateSchemaError, match=_RETIRED_KIND_MESSAGE) as caught:
            validate_gate_config(3, dict(_RETIRED_TASK_SUCCESS), advancement_enabled=True)
        # Its bar is retired with it, and the refusal names that too.
        assert "'min_success_lcb' was removed" in str(caught.value)
        assert "unknown gate_kind" not in str(caught.value)
        with pytest.raises(GateSchemaError, match=_RETIRED_KIND_MESSAGE):
            thresholds_from_configs({3: {"curriculum_kwargs": dict(_RETIRED_TASK_SUCCESS)}}, advancement_enabled=True)

    def test_the_schema_refuses_min_success_rate_naming_the_decision(self):
        from environments.shared.curriculum.gate_schema import validate_gate_config

        with pytest.raises(GateSchemaError, match=_RETIRED_KEY_MESSAGE) as caught:
            validate_gate_config(3, dict(_RETIRED_SUCCESS_FLOOR), advancement_enabled=True)
        assert "unrecognised" not in str(caught.value)
        # Even a disabled (0.0) floor: a key no reader evaluates is refused, not ignored.
        with pytest.raises(GateSchemaError, match=_RETIRED_KEY_MESSAGE):
            thresholds_from_configs(
                {3: {"curriculum_kwargs": dict(_RETIRED_SUCCESS_FLOOR, min_success_rate=0.0)}},
                advancement_enabled=True,
            )

    def test_the_schema_refuses_the_panel_size_key(self):
        from environments.shared.curriculum.gate_schema import validate_gate_config

        block = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "none/v1", "task_success_panel_episodes": 30}
        with pytest.raises(GateSchemaError, match=r"'task_success_panel_episodes' was removed .*D-D59"):
            validate_gate_config("pilot", block, advancement_enabled=False)

    @pytest.mark.parametrize(
        ("curriculum", "message"),
        [(_RETIRED_TASK_SUCCESS, _RETIRED_KIND_MESSAGE), (_RETIRED_SUCCESS_FLOOR, _RETIRED_KEY_MESSAGE)],
        ids=["task_success-v1", "min_success_rate"],
    )
    def test_the_judge_refuses_a_frozen_record_naming_the_decision(self, tmp_path, curriculum, message):
        from environments.shared.reporting import evaluate_stage_gate

        passed, failures = evaluate_stage_gate(dict(curriculum), self.STAGE_RESULTS, stage=3, stage_dir=tmp_path)
        assert passed is False
        assert failures and all(failure.startswith("stage 3: ") for failure in failures)
        assert any(re.search(message, failure) for failure in failures)

    @pytest.mark.parametrize("curriculum", [_RETIRED_TASK_SUCCESS, _RETIRED_SUCCESS_FLOOR], ids=["kind", "key"])
    def test_the_recorded_history_reading_proves_nothing(self, curriculum):
        from environments.shared.reporting import evaluate_recorded_gate

        evaluations = [
            {"mean_reward": 1e9, "mean_episode_length": 1000.0, "n_episodes": 30, "mean_success_rate": 1.0}
        ] * 5
        assert evaluate_recorded_gate(dict(curriculum, required_consecutive=1), evaluations) is None

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

    def test_what_would_have_to_be_deleted_for_the_retired_refusal_to_stop_being_consulted(self):
        """The consulted test (plan §8 invariant 10) for the refusal, by AST.

        evaluate_stage_gate calls retired_gate_declarations, and returns on
        what it finds, BEFORE its first comparison of the gate kind and so
        before the reward_and_length return; validate_gate_config calls it
        before the unknown-key check.
        """
        from environments.shared.curriculum import gate_schema
        from environments.shared.reporting import gates

        module = ast.parse(inspect.getsource(gates))
        dispatch = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_stage_gate"
        )
        retired_call = min(
            node.lineno
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "retired_gate_declarations"
        )
        first_kind_compare = min(
            node.lineno
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Compare)
            and any(isinstance(n, ast.Name) and n.id == "gate_kind" for n in ast.walk(node))
        )
        assert retired_call < first_kind_compare

        schema = ast.parse(inspect.getsource(gate_schema))
        validator = next(
            node for node in schema.body if isinstance(node, ast.FunctionDef) and node.name == "validate_gate_config"
        )
        calls = [
            node.lineno
            for node in ast.walk(validator)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "retired_gate_declarations"
        ]
        unknown = min(
            node.lineno
            for node in ast.walk(validator)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "unknown" for t in node.targets)
        )
        assert calls and min(calls) < unknown


class TestStanceV2GateIsConsulted:
    """Plan §8 invariant 10 for ``stance_quality/v2`` (decision D-D23): fail-closed in every dispatch.

    Registering the kind without its arms would route a v2 stage to the
    reward conjunction at JUDGE and to the generic arm of the recorded-gate
    reader, both of which certify on ``min_avg_reward`` -- the rail the
    statue clears -- and, naively wired into the v1 stance arm, would accept
    a v1-scored report labelled v2.  These pin that the manager refuses it,
    that the judge's v2 arm is what is consulted (before the reward return,
    and down to the binomial bound), and that the substitutes are refused.
    """

    @staticmethod
    def _curriculum():
        from .stance_v2_helpers import V2_CURRICULUM

        return dict(V2_CURRICULUM)

    def test_the_manager_never_advances_on_sky_high_reward(self, caplog):
        mgr = _manager("stance_quality/v2")
        with caplog.at_level(logging.ERROR, logger="environments.shared.curriculum.manager"):
            for _ in range(5):
                assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)
        assert mgr.summary()["consecutive_passes"][1] == 0
        assert "stance_quality/v2" in caplog.text and "cannot evaluate" in caplog.text
        assert "handoff pair" in caplog.text and "3042-3081" in caplog.text

    def test_a_schema_valid_v2_config_still_refuses_end_to_end(self):
        thresholds = thresholds_from_configs(
            {1: {"curriculum_kwargs": dict(self._curriculum(), required_consecutive=1)}}
        )
        assert thresholds[1]["gate_kind"] == "stance_quality/v2"
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"), stage_thresholds=thresholds
        )
        for _ in range(3):
            assert not mgr.should_advance(_SKY_HIGH_PANEL * 2, _FULL_PANEL * 2)

    def test_the_v1_arm_is_never_used_for_a_v2_stage(self, monkeypatch):
        from environments.shared.curriculum.manager import CurriculumManager as Manager

        monkeypatch.setattr(Manager, "_stance_gate_passes", lambda *a, **k: pytest.fail("v1 arm consulted"))
        assert not _manager("stance_quality/v2").should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)

    def test_the_recorded_history_reader_never_reads_reward_for_v2(self):
        from environments.shared.reporting import evaluate_recorded_gate

        history = [{"mean_reward": 1e9, "mean_episode_length": 1000.0, "n_episodes": 40}] * 5
        assert evaluate_recorded_gate({**self._curriculum(), "min_avg_reward": 1.0}, history) is None
        counted = [{"stance_clean_count": 40, "n_stance_samples": 40}] * 3
        assert evaluate_recorded_gate(self._curriculum(), counted) is True
        assert (
            evaluate_recorded_gate(self._curriculum(), [{"stance_clean_count": 36, "n_stance_samples": 40}] * 3)
            is False
        )

    def test_the_judge_certifies_a_valid_report_and_refuses_its_substitutes(self, tmp_path):
        from environments.shared.reporting import evaluate_stage_gate

        from .stance_v2_helpers import v2_report

        report = v2_report(tmp_path)

        def judge(stance_report, **curriculum):
            return evaluate_stage_gate(
                {**self._curriculum(), **curriculum},
                {"best_model_reward": 1e9},
                stage=1,
                stance_report=stance_report,
                stage_dir=tmp_path,
            )

        assert judge(report) == (True, [])
        assert judge(None)[0] is False
        v1_scored = {**report, "scored_gate_kind": "stance_quality/v1"}
        assert any("was scored by 'stance_quality/v1'" in f for f in judge(v1_scored)[1])
        assert any("scored under other thresholds" in f for f in judge(report, min_all_feet_support=0.99)[1])
        (tmp_path / "models" / "robust_best_model.zip").write_bytes(b"another checkpoint")
        assert any("not the handoff robust_best_model's" in f for f in judge(report)[1])

    def test_a_recomputed_verdict_that_disagrees_is_refused(self, tmp_path):
        from environments.shared.reporting import evaluate_stage_gate

        from .stance_v2_helpers import v2_report

        report = v2_report(tmp_path, n_clean=36, defect={"settle_airborne_substeps": 4.0})
        assert report["passed"] is False
        laundered = {**report, "passed": True, "failures": [], "result": {**report["result"], "n_clean": 40}}
        laundered["episode_evidence"] = [{**row, "clean": True, "reasons": []} for row in report["episode_evidence"]]
        passed, failures = evaluate_stage_gate(
            self._curriculum(), {}, stage=1, stance_report=laundered, stage_dir=tmp_path
        )
        assert passed is False and any("disagrees with the one re-derived" in failure for failure in failures)

    def test_what_would_have_to_be_deleted_for_the_v2_bound_to_stop_being_consulted(self, tmp_path, monkeypatch):
        """The consulted test (plan §8 invariant 10): the call chain, by AST and at runtime.

        evaluate_stage_gate dispatches on STANCE_GATE_V2_KIND into
        _stance_v2_stage_gate BEFORE the reward_and_length return; that arm
        and stance_v2_report_refusals (the backfill tool's check) both read
        _admit_stance_v2_report, which re-derives through
        _rederive_stance_v2_verdict; that calls evaluate_stance_v2_gate; and
        that calls binomial_lcb.  Then the runtime half: with binomial_lcb
        raising, judging a valid v2 report raises -- the bound was consulted.
        """
        from environments.shared.curriculum import stance_gate_v2
        from environments.shared.reporting import evaluate_stage_gate, gates

        from .stance_v2_helpers import v2_report

        module = ast.parse(inspect.getsource(gates))
        functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        dispatch = functions["evaluate_stage_gate"]
        v2_branch = reward_return = None
        for node in ast.walk(dispatch):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                if {"gate_kind", "STANCE_GATE_V2_KIND"} <= names:
                    calls = {
                        c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    }
                    assert "_stance_v2_stage_gate" in calls
                    v2_branch = node.lineno
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "_reward_and_length_stage_gate":
                    reward_return = node.lineno
        assert v2_branch is not None and reward_return is not None
        assert v2_branch < reward_return

        def calls_in(name: str) -> set[str]:
            return {
                c.func.id for c in ast.walk(functions[name]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }

        assert "_admit_stance_v2_report" in calls_in("_stance_v2_stage_gate")
        assert "_admit_stance_v2_report" in calls_in("stance_v2_report_refusals")
        assert "_rederive_stance_v2_verdict" in calls_in("_admit_stance_v2_report")
        assert "evaluate_stance_v2_gate" in calls_in("_rederive_stance_v2_verdict")

        gate_module = ast.parse(inspect.getsource(stance_gate_v2))
        evaluator = next(
            node
            for node in gate_module.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_stance_v2_gate"
        )
        assert "binomial_lcb" in {
            c.func.id for c in ast.walk(evaluator) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }

        report = v2_report(tmp_path)

        def consulted(*args, **kwargs):
            raise RuntimeError("consulted")

        monkeypatch.setattr(stance_gate_v2, "binomial_lcb", consulted)
        with pytest.raises(RuntimeError, match="consulted"):
            evaluate_stage_gate(self._curriculum(), {}, stage=1, stance_report=report, stage_dir=tmp_path)

    def test_publication_has_a_v2_arm_before_the_legacy_reward_loop(self):
        """Without it a recorded v2 PASS takes the legacy loop and publishes on min_avg_reward alone."""
        from environments.shared.result_bundle import evidence

        source = inspect.getsource(evidence.validate_evaluation_evidence)
        v2_arm = source.index("elif recorded_pass and gate_kind == STANCE_GATE_V2_KIND:")
        legacy_loop = source.index("elif recorded_pass:")
        assert v2_arm < legacy_loop
        assert "_validate_stance_v2_panel_evidence(" in source[v2_arm:legacy_loop]


class TestStanceHeadingGateIsConsulted:
    """Plan §8 invariant 10 for the six-heading certificate (``stance_quality_heading/v1``, decision D-D32).

    Registered without its arms, the kind would fall to the closed refusal
    at JUDGE (fail-closed, but never certifiable), to the generic arm of the
    recorded-gate reader (the reward rail), and -- naively wired into the v2
    arm -- would accept a yaw-zero v2 report.  These pin that the manager
    refuses it, that the judge's heading arm is consulted before the reward
    return and down to the grouped statistic, that publication has its arm
    before the legacy reward loop, and that a history never certifies it.
    """

    @staticmethod
    def _curriculum():
        from .stance_heading_helpers import HEADING_CURRICULUM

        return dict(HEADING_CURRICULUM)

    def test_the_manager_never_advances_on_sky_high_reward(self, caplog):
        mgr = _manager("stance_quality_heading/v1")
        with caplog.at_level(logging.ERROR, logger="environments.shared.curriculum.manager"):
            for _ in range(5):
                assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)
        assert mgr.summary()["consecutive_passes"][1] == 0
        assert "stance_quality_heading/v1" in caplog.text and "cannot evaluate" in caplog.text

    def test_a_schema_valid_heading_config_still_refuses_end_to_end(self):
        thresholds = thresholds_from_configs(
            {1: {"curriculum_kwargs": dict(self._curriculum(), required_consecutive=1)}}
        )
        assert thresholds[1]["gate_kind"] == "stance_quality_heading/v1"
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"), stage_thresholds=thresholds
        )
        for _ in range(3):
            assert not mgr.should_advance(_SKY_HIGH_PANEL * 2, _FULL_PANEL * 2)

    def test_the_recorded_history_reader_never_certifies_the_grid(self):
        from environments.shared.reporting import evaluate_recorded_gate

        reward_history = [{"mean_reward": 1e9, "mean_episode_length": 1000.0, "n_episodes": 40}] * 5
        assert evaluate_recorded_gate({**self._curriculum(), "min_avg_reward": 1.0}, reward_history) is None
        counted = [{"stance_clean_count": 40, "n_stance_samples": 40}] * 3
        assert evaluate_recorded_gate(self._curriculum(), counted) is None

    def test_what_would_have_to_be_deleted_for_the_grouped_bound_to_stop_being_consulted(self, tmp_path, monkeypatch):
        """The call chain by AST, then at runtime: with the grouped bound raising, judging a certificate raises."""
        from environments.shared.curriculum import stance_heading_gate
        from environments.shared.reporting import evaluate_stage_gate, gates

        from .stance_heading_helpers import heading_report

        module = ast.parse(inspect.getsource(gates))
        functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        dispatch = functions["evaluate_stage_gate"]
        heading_branch = v2_branch = reward_return = None
        for node in ast.walk(dispatch):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                calls = {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
                if {"gate_kind", "STANCE_HEADING_GATE_KIND"} <= names:
                    assert "_stance_heading_stage_gate" in calls and "_stance_v2_stage_gate" not in calls
                    heading_branch = node.lineno
                if {"gate_kind", "STANCE_GATE_V2_KIND"} <= names:
                    v2_branch = node.lineno
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "_reward_and_length_stage_gate":
                    reward_return = node.lineno
        assert heading_branch is not None and v2_branch is not None and reward_return is not None
        assert heading_branch < reward_return

        def calls_in(name: str) -> set[str]:
            return {
                c.func.id for c in ast.walk(functions[name]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }

        assert "_admit_stance_heading_report" in calls_in("_stance_heading_stage_gate")
        assert "_admit_stance_heading_report" in calls_in("stance_heading_report_refusals")
        assert "_rederive_stance_heading_verdict" in calls_in("_admit_stance_heading_report")
        assert "evaluate_stance_heading_gate" in calls_in("_rederive_stance_heading_verdict")
        gate_module = ast.parse(inspect.getsource(stance_heading_gate))
        evaluator = next(
            node
            for node in gate_module.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_stance_heading_gate"
        )
        called = {c.func.id for c in ast.walk(evaluator) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert {"binomial_lcb", "evaluate_stance_v2_gate", "grid_refusals"} <= called

        report = heading_report(tmp_path)

        def consulted(*args, **kwargs):
            raise RuntimeError("consulted")

        monkeypatch.setattr(stance_heading_gate, "binomial_lcb", consulted)
        with pytest.raises(RuntimeError, match="consulted"):
            evaluate_stage_gate(self._curriculum(), {}, stage=1, stance_report=report, stage_dir=tmp_path)

    def test_publication_has_a_heading_arm_before_the_legacy_reward_loop(self):
        from environments.shared.result_bundle import evidence

        source = inspect.getsource(evidence.validate_evaluation_evidence)
        heading_arm = source.index("elif recorded_pass and gate_kind == STANCE_HEADING_GATE_KIND:")
        legacy_loop = source.index("elif recorded_pass:")
        assert heading_arm < legacy_loop
        assert "_validate_stance_heading_panel_evidence(" in source[heading_arm:legacy_loop]


class TestCommandTrackingGateIsConsulted:
    """Plan §8 invariant 10 for the command-tracking certificate (``command_tracking/v1``, decision D-D61).

    The kind's verdict is a post-stage panel on the handoff chosen by the
    gated statistic.  These pin that the manager refuses it on any reward,
    that a history never certifies it, that the judge's command arm is
    consulted before the reward return and down to the exact binomial bound,
    and that publication has its arm before the legacy reward loop.
    """

    @staticmethod
    def _curriculum():
        from .test_command_tracking_wiring import WALK_CURRICULUM

        return dict(WALK_CURRICULUM)

    def test_the_manager_never_advances_on_sky_high_reward(self, caplog):
        """The normal path of every walk: refused at every evaluation, said once per stage at INFO, never an ERROR."""
        mgr = _manager("command_tracking/v1")
        with caplog.at_level(logging.DEBUG, logger="environments.shared.curriculum.manager"):
            for _ in range(5):
                assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)
        assert mgr.summary()["consecutive_passes"][1] == 0
        refusals = [record for record in caplog.records if "command_tracking/v1" in record.getMessage()]
        assert [record.levelno for record in refusals] == [logging.INFO], "one record over five evaluations"
        assert "command_tracking_report.json" in refusals[0].getMessage()
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]

    def test_the_refusal_is_said_once_per_stage(self, caplog):
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            stage_thresholds={
                stage: {"gate_kind": "command_tracking/v1", "min_eval_episodes": 1, "required_consecutive": 1}
                for stage in (1, 2)
            },
        )
        with caplog.at_level(logging.INFO, logger="environments.shared.curriculum.manager"):
            for _ in range(3):
                assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)
            mgr.advance()
            for _ in range(3):
                assert not mgr.should_advance(_SKY_HIGH_REWARDS, _FULL_LENGTHS)
        refusals = [record.getMessage() for record in caplog.records if "command_tracking/v1" in record.getMessage()]
        assert [message.split(" declares", 1)[0] for message in refusals] == ["Stage 1", "Stage 2"]

    def test_a_schema_valid_command_config_still_refuses_end_to_end(self):
        thresholds = thresholds_from_configs(
            {1: {"curriculum_kwargs": dict(self._curriculum(), required_consecutive=1)}}
        )
        assert thresholds[1]["gate_kind"] == "command_tracking/v1"
        mgr = CurriculumManager(
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"), stage_thresholds=thresholds
        )
        for _ in range(3):
            assert not mgr.should_advance(_SKY_HIGH_PANEL * 2, _FULL_PANEL * 2)

    def test_the_judge_and_the_history_reader_never_certify_on_reward(self, tmp_path):
        from environments.shared.reporting import evaluate_recorded_gate, evaluate_stage_gate

        reward_history = [{"mean_reward": 1e9, "mean_episode_length": 1000.0, "n_episodes": 40}] * 5
        assert evaluate_recorded_gate({**self._curriculum(), "min_avg_reward": 1.0}, reward_history) is None
        results = {"mean_reward": 1e9, "mean_episode_length": 1000.0, "n_episodes": 40}
        passed, failures = evaluate_stage_gate(self._curriculum(), results, stage=2, stage_dir=tmp_path)
        assert not passed and "no command-tracking report" in failures[0]

    def test_what_would_have_to_be_deleted_for_the_panel_bound_to_stop_being_consulted(self, tmp_path, monkeypatch):
        """The call chain by AST, then at runtime: with the exact bound raising, judging an admitted report raises."""
        from environments.shared.curriculum import command_tracking_gate
        from environments.shared.reporting import evaluate_stage_gate, gates

        from .test_command_tracking_wiring import _choose, _panel, _report, _stage_dir

        module = ast.parse(inspect.getsource(gates))
        functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        command_branch = reward_return = None
        for node in ast.walk(functions["evaluate_stage_gate"]):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                calls = {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
                if {"gate_kind", "COMMAND_TRACKING_GATE_KIND"} <= names:
                    assert "_command_tracking_stage_gate" in calls and "_reward_and_length_stage_gate" not in calls
                    command_branch = node.lineno
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "_reward_and_length_stage_gate":
                    reward_return = node.lineno
        assert command_branch is not None and reward_return is not None and command_branch < reward_return

        def calls_in(name: str) -> set[str]:
            return {
                c.func.id for c in ast.walk(functions[name]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }

        assert "_admit_command_tracking_report" in calls_in("_command_tracking_stage_gate")
        assert "_admit_command_tracking_report" in calls_in("command_tracking_report_refusals")
        assert "_rederive_command_tracking_verdict" in calls_in("_admit_command_tracking_report")
        assert "evaluate_command_tracking_gate" in calls_in("_rederive_command_tracking_verdict")
        gate_module = ast.parse(inspect.getsource(command_tracking_gate))
        gate_functions = {node.name: node for node in gate_module.body if isinstance(node, ast.FunctionDef)}

        def gate_calls(name: str) -> set[str]:
            node = gate_functions[name]
            return {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}

        assert {"evaluate_walk_profile", "evaluate_follow_profile"} <= gate_calls("evaluate_command_tracking_gate")
        assert "_lcb" in gate_calls("evaluate_walk_profile") and "_lcb" in gate_calls("evaluate_follow_profile")
        assert "binomial_lcb" in gate_calls("_lcb")

        stage_dir = _stage_dir(tmp_path)
        _choose(stage_dir)
        report = _report(stage_dir, _panel())
        assert evaluate_stage_gate(
            self._curriculum(), {}, stage=2, stage_dir=stage_dir, command_tracking_report=report
        )[0]

        def consulted(*args, **kwargs):
            raise RuntimeError("consulted")

        monkeypatch.setattr(command_tracking_gate, "binomial_lcb", consulted)
        with pytest.raises(RuntimeError, match="consulted"):
            evaluate_stage_gate(self._curriculum(), {}, stage=2, stage_dir=stage_dir, command_tracking_report=report)

    def test_publication_has_a_command_arm_before_the_legacy_reward_loop(self):
        from environments.shared.result_bundle import evidence

        source = inspect.getsource(evidence.validate_evaluation_evidence)
        command_arm = source.index("elif recorded_pass and gate_kind == COMMAND_TRACKING_GATE_KIND:")
        legacy_loop = source.index("elif recorded_pass:")
        assert command_arm < legacy_loop
        assert "_validate_command_tracking_panel_evidence(" in source[command_arm:legacy_loop]
