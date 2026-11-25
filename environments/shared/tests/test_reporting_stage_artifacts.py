"""Tests for environments.shared.reporting.stage_artifacts."""

import json
import logging
import re

import pytest

from environments.shared.model_id import ModelId
from environments.shared.reporting import build_stage_results_from_eval_data

from .reporting_helpers import plant_identity

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")


class TestBuildStageResultsFromEvalData:
    """Tests for build_stage_results_from_eval_data."""

    def test_builds_from_evaluations_npz(self, tmp_path):
        import numpy as np

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        rewards = np.array([[10.0, 12.0], [20.0, 22.0], [15.0, 17.0]])
        lengths = np.array([[100, 110], [200, 210], [150, 160]])
        timesteps = np.array([50000, 100000, 150000])
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=rewards,
            ep_lengths=lengths,
            timesteps=timesteps,
        )

        config = {
            "name": "Balance",
            "description": "Stand up",
            "env_kwargs": {"sim_dt": 0.02},
        }

        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=1,
            stage_config=config,
            timesteps=150_000,
        )

        assert result["stage"] == 1
        assert result["name"] == "Balance"
        assert result["timesteps"] == 150_000
        assert result["sim_dt"] == 0.02
        # Best eval is at index 1 (mean 21.0)
        assert result["best_eval_reward"] == 21.0
        assert result["best_eval_timestep"] == 100000
        # Last eval used as final metrics (mean 16.0)
        assert result["mean_reward"] == 16.0
        # No post-training panel ran: the velocity keys are absent, never a
        # fabricated 0.0 the gate would then compare against (review ER4).
        assert "mean_forward_vel" not in result
        assert "mean_success_rate" not in result

    def test_an_explicit_sim_dt_wins_over_the_fallback(self, tmp_path):
        """Callers pass the env's `dt`; the 0.01 s fallback is wrong for the compsognathus pair."""
        config = {"name": "Locomotion", "description": "Walk", "env_kwargs": {}}
        default = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=config, timesteps=1)
        explicit = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=config, timesteps=1, sim_dt=0.02)
        assert default["sim_dt"] == 0.01
        assert explicit["sim_dt"] == 0.02

    def test_reads_duration_from_metrics_json(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        (tmp_path / "metrics.json").write_text(json.dumps({"training_duration_seconds": 123.4}))

        config = {"name": "Loco", "description": "Walk", "env_kwargs": {}}
        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=2,
            stage_config=config,
            timesteps=50_000,
        )
        assert result["duration_seconds"] == 123.4

    def test_reads_plant_identity_from_metrics_json(self, tmp_path):
        (tmp_path / "models").mkdir()
        identity = plant_identity().to_dict()
        (tmp_path / "metrics.json").write_text(json.dumps({"plant_identity": identity}))

        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=1,
            stage_config={"name": "Balance", "description": "Stand", "env_kwargs": {}},
            timesteps=100,
        )

        assert result["plant_identity"] == identity

    def test_explicit_duration_overrides_metrics_json(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        (tmp_path / "metrics.json").write_text(json.dumps({"training_duration_seconds": 123.4}))

        config = {"name": "Loco", "description": "Walk", "env_kwargs": {}}
        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=2,
            stage_config=config,
            timesteps=50_000,
            duration_seconds=999.0,
        )
        assert result["duration_seconds"] == 999.0

    def test_no_eval_data_returns_defaults(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        config = {"name": "Balance", "description": "Stand", "env_kwargs": {}}
        result = build_stage_results_from_eval_data(
            tmp_path,
            stage=1,
            stage_config=config,
            timesteps=100_000,
        )
        assert result["mean_reward"] == 0.0
        assert result["best_eval_reward"] == ""
        assert result["duration_seconds"] == 0.0

    # Review ER4: a results dict built here (the sweep workers' then, retired
    # by D-D17; the backfill tool's now) takes the velocity and success
    # numbers the gate judges from here — and they were hardcoded to 0.0
    # although metrics.json (already opened for the duration) carries the
    # real panel.

    _LOCO = {"name": "Loco", "description": "Walk", "env_kwargs": {}}

    def test_reads_the_velocity_panel_from_metrics_json(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "mean_forward_vel": 1.75,
                    "std_forward_vel": 0.2,
                    "mean_distance_traveled": 17.5,
                    "mean_success_rate": 0.9,
                }
            )
        )

        result = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=self._LOCO, timesteps=10)

        assert result["mean_forward_vel"] == 1.75
        assert result["std_forward_vel"] == 0.2
        assert result["mean_distance_traveled"] == 17.5
        assert result["mean_success_rate"] == 0.9

    def test_a_skipped_panel_leaves_the_metrics_unmeasured(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "post_eval_skipped": True,
                    "quality_eval_checkpoint": None,
                    "mean_forward_vel": None,
                    "mean_success_rate": None,
                }
            )
        )

        result = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=self._LOCO, timesteps=10)

        for key in ("mean_forward_vel", "std_forward_vel", "mean_distance_traveled", "mean_success_rate"):
            assert key not in result

    def _velocity_gated(self):
        return {
            **self._LOCO,
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "gate_schema_version": 1,
                "min_avg_reward": -1.0,
                "min_avg_forward_vel": 1.0,
            },
        }

    def test_an_unmeasured_velocity_fails_the_gate_by_name_not_at_zero(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        (tmp_path / "models").mkdir()
        (tmp_path / "metrics.json").write_text(json.dumps({"training_duration_seconds": 1.0}))
        config = self._velocity_gated()
        results = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=config, timesteps=10)

        _apply_stage_gate(
            stage=2, stage_config=config, stage_results=results, stance_report=None, model_id=VELOCIRAPTOR_MONGOLIENSIS
        )

        assert results["gate_passed"] is False
        assert any("no forward-velocity measurement" in failure for failure in results["gate_failures"])
        assert not any("0.00 m/s" in failure for failure in results["gate_failures"])

    def test_a_measured_velocity_reaches_the_gate(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        (tmp_path / "models").mkdir()
        (tmp_path / "metrics.json").write_text(json.dumps({"mean_forward_vel": 1.5, "std_forward_vel": 0.1}))
        config = self._velocity_gated()
        results = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=config, timesteps=10)

        _apply_stage_gate(
            stage=2, stage_config=config, stage_results=results, stance_report=None, model_id=VELOCIRAPTOR_MONGOLIENSIS
        )

        assert (results["gate_passed"], results["gate_failures"]) == (True, [])

    def test_the_stage_summary_says_not_measured_instead_of_zero(self, tmp_path):
        from environments.shared.reporting import write_stage_summary

        (tmp_path / "models").mkdir()
        results = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=self._LOCO, timesteps=10)

        text = write_stage_summary(tmp_path, results, VELOCIRAPTOR_MONGOLIENSIS, "PPO").read_text()

        assert "Avg fwd vel:    not measured" in text
        assert "0.00 +/- 0.00 m/s" not in text


class TestSaveStageConfigKeepsTheCurriculumTable:
    """The SB3 config writer records ``[curriculum]`` verbatim, a sub-table included."""

    def test_the_sb3_config_writer_still_records_the_shared_table_verbatim(self, tmp_path):
        """generate_stage_artifacts hands save_stage_config the raw config, and the writer filters
        nothing: refusing a key is the stage loader's, the CLI's and the gate schema's job (each refuses this
        retired ``[curriculum.jax]`` table), so the record shows what the stage declared, sub-table and all."""
        from environments.shared.config import save_stage_config

        stage_config = {
            "name": "Stage 1",
            "description": "Curriculum stage 1",
            "env_kwargs": {"forward_vel_weight": 1.0},
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "gate_schema_version": 1,
                "min_avg_reward": 100.0,
                "min_avg_episode_length": 100,
                "jax": {"min_avg_reward": 40.0},
            },
        }
        path = save_stage_config(
            tmp_path,
            stage=1,
            stage_config=stage_config,
            algorithm="PPO",
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        saved = json.loads(path.read_text())["curriculum"]
        assert saved["min_avg_reward"] == 100.0
        assert saved["jax"] == {"min_avg_reward": 40.0}


class TestApplyStageGate:
    """The wiring: generate_stage_artifacts must record a verdict, always.

    The notebook reads `publication_gate_passed` off the dict this writes. If
    it were absent the notebook's `if not results_1["publication_gate_passed"]`
    would raise KeyError; if it were reward-derived we would be back to run
    20260802_203215, which advanced a stance-gated stage on its rail alone.
    """

    STANCE_CONFIG = {
        "name": "Balance",
        "description": "Stand",
        "curriculum_kwargs": {
            "gate_kind": "stance_quality/v1",
            "gate_schema_version": 1,
            "min_full_horizon_fraction": 0.95,
            "max_unsupported_duty": 0.02,
            "max_unsupported_duty_ucb": 0.02,
            "min_eval_episodes": 40,
            "min_avg_reward": 1950.0,
        },
    }

    def _apply(self, stage_config, stage_results, stance_report):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        _apply_stage_gate(
            stage=1,
            stage_config=stage_config,
            stage_results=stage_results,
            stance_report=stance_report,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        return stage_results

    def test_a_missing_stance_panel_records_a_failure_not_a_pass(self):
        results = self._apply(self.STANCE_CONFIG, {"best_model_reward": 2297.0}, None)
        assert results["publication_gate_passed"] is False
        assert results["gate_passed"] is False
        assert results["gate_failures"]

    def test_records_the_stance_panel_verdict(self):
        report = {"gate_kind": "stance_quality/v1", "passed": True, "failures": []}
        results = self._apply(self.STANCE_CONFIG, {"best_model_reward": 2297.0}, report)
        assert results["publication_gate_passed"] is True
        assert results["gate_failures"] == []

    def test_verdict_keys_are_always_written(self):
        """Absent keys would turn a gate failure into a KeyError at the check."""
        results = self._apply(
            {"name": "Run", "description": "Go", "curriculum_kwargs": {}},
            {},
            None,
        )
        assert set(results) >= {"gate_passed", "publication_gate_passed", "gate_failures"}
        assert results["publication_gate_passed"] is False

    def test_a_malformed_config_records_a_failure_instead_of_raising(self):
        """The docstring promises "never raises"; enforce it.

        `evaluate_stage_gate` coerces config values, so a non-numeric
        threshold reaches it as a ValueError. Propagating would cost a
        finished multi-hour stage the artifacts written around this call.
        """
        config = {
            "name": "Balance",
            "description": "Stand",
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "gate_schema_version": 1,
                "min_avg_reward": "not-a-number",
            },
        }
        results = self._apply(config, {"best_model_reward": 2297.0}, None)
        assert results["publication_gate_passed"] is False
        assert any("gate evaluation raised" in failure for failure in results["gate_failures"])

    def test_a_non_iterable_failures_field_records_a_failure_instead_of_raising(self):
        report = {"gate_kind": "stance_quality/v1", "passed": False, "failures": 7}
        results = self._apply(self.STANCE_CONFIG, {"best_model_reward": 2297.0}, report)
        assert results["publication_gate_passed"] is False
        assert results["gate_failures"]

    def test_records_the_gate_kind_beside_the_verdict(self):
        """Review SS5: a verdict without its gate kind is re-served as a bare boolean."""
        results = self._apply(self.STANCE_CONFIG, {"best_model_reward": 2297.0}, None)
        assert results["gate_kind"] == "stance_quality/v1"
        assert results["gate_schema_version"] == 1

    def test_an_undeclared_gate_kind_is_recorded_as_null(self):
        results = self._apply({"name": "Run", "description": "Go", "curriculum_kwargs": {}}, {}, None)
        assert results["gate_kind"] is None
        assert results["gate_schema_version"] is None


class TestStageSummaryRecordsTheVerdict:
    """The reasons must outlive the Colab cell that raised them.

    On gate failure the notebook calls `halt`, which releases the runtime and raises,
    so the failure text is the first thing lost. `collected_results.csv`
    carries the boolean but not the criteria, and before this the stage
    summary carried neither.
    """

    @staticmethod
    def _results(**overrides):
        base = {
            "stage": 1,
            "name": "Balance",
            "description": "Stand",
            "timesteps": 1_450_000,
            "duration_seconds": 3114.1,
            "mean_reward": 281.85,
            "std_reward": 154.5,
            "mean_episode_length": 313.0,
            "std_episode_length": 142.0,
        }
        base.update(overrides)
        return base

    def test_failing_stage_summary_names_every_criterion(self, tmp_path):
        from environments.shared.reporting import text_summaries

        text_summaries.write_stage_summary(
            tmp_path,
            self._results(
                publication_gate_passed=False,
                gate_failures=[
                    "mean_unsupported_duty 0.2120 > 0.0200",
                    "unsupported_duty_ucb 0.2153 > 0.0200",
                ],
            ),
            TYRANNOSAURUS_REX,
            "PPO",
        )
        summary = (tmp_path / "stage_summary.txt").read_text(encoding="utf-8")
        assert "Curriculum Gate" in summary
        assert "Verdict:      FAIL" in summary
        assert "mean_unsupported_duty 0.2120 > 0.0200" in summary
        assert "unsupported_duty_ucb 0.2153 > 0.0200" in summary

    def test_passing_stage_summary_states_the_pass(self, tmp_path):
        from environments.shared.reporting import text_summaries

        text_summaries.write_stage_summary(
            tmp_path,
            self._results(publication_gate_passed=True, gate_failures=[]),
            TYRANNOSAURUS_REX,
            "PPO",
        )
        summary = (tmp_path / "stage_summary.txt").read_text(encoding="utf-8")
        assert "Verdict:      PASS" in summary

    def test_a_summary_with_no_verdict_omits_the_section(self, tmp_path):
        """Pre-gate callers must still work."""
        from environments.shared.reporting import text_summaries

        text_summaries.write_stage_summary(tmp_path, self._results(), TYRANNOSAURUS_REX, "PPO")
        assert "Curriculum Gate" not in (tmp_path / "stage_summary.txt").read_text(encoding="utf-8")

    def test_generate_stage_artifacts_writes_the_summary_after_the_gate(self):
        """Ordering regression: the summary must see the verdict.

        `write_stage_summary` used to run before `_apply_stage_gate`, which
        would print a stage summary with no gate section on every SB3 run.
        """
        import inspect

        from environments.shared.reporting import stage_artifacts

        source = inspect.getsource(stage_artifacts.generate_stage_artifacts)
        assert source.index("_apply_stage_gate(") < source.index("write_stage_summary(")


class TestStageGateVerdictRecord:
    """The post-stage judge writes gate_verdict.json beside the handoff pair it judged
    (decision D-A5): hash-bound, task-bound, and never at the cost of the artifacts."""

    STANCE_CONFIG = TestApplyStageGate.STANCE_CONFIG
    TASK = "sha256:" + "7" * 64

    def _stage_dir(self, tmp_path, *, with_handoff=True):
        stage_dir = tmp_path / "01_stance"
        (stage_dir / "models").mkdir(parents=True)
        if with_handoff:
            (stage_dir / "models" / "best_model.zip").write_bytes(b"policy weights")
            (stage_dir / "models" / "best_model_vecnorm.pkl").write_bytes(b"normalisation statistics")
        (stage_dir / "stage_config.json").write_text(
            json.dumps({"task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": self.TASK}}),
            encoding="utf-8",
        )
        return stage_dir

    def _apply(self, stage_dir, stance_report, *, model_id=VELOCIRAPTOR_MONGOLIENSIS, stage=1):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        results = {"best_model_reward": 2297.0, "mean_reward": 2200.0, "timesteps": 10}
        _apply_stage_gate(
            stage=stage,
            stage_config=self.STANCE_CONFIG,
            stage_results=results,
            stance_report=stance_report,
            stage_dir=stage_dir,
            model_id=model_id,
        )
        return results

    def test_a_pass_is_recorded_hash_bound_to_the_handoff_it_judged(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import GATE_VERDICT_JUDGED_BY
        from environments.shared.result_bundle import read_gate_verdict, sha256_file, verdict_is_reusable

        stage_dir = self._stage_dir(tmp_path)
        results = self._apply(stage_dir, {"gate_kind": "stance_quality/v1", "passed": True, "failures": []})

        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["passed"] is True and verdict["failures"] == []
        assert (verdict["species"], verdict["embodiment"], verdict["stage"], verdict["stage_id"]) == (
            "velociraptor_mongoliensis",
            "anatomical",
            1,
            "stance",
        )
        assert verdict["gate_kind"] == "stance_quality/v1" and verdict["gate_schema_version"] == 1
        assert verdict["checkpoint"] == "models/best_model.zip"
        assert verdict["checkpoint_sha256"] == sha256_file(stage_dir / "models" / "best_model.zip")
        assert verdict["normalization"] == "models/best_model_vecnorm.pkl"
        assert verdict["normalization_sha256"] == sha256_file(stage_dir / "models" / "best_model_vecnorm.pkl")
        assert verdict["task_sha256"] == self.TASK
        assert verdict["judged_by"] == GATE_VERDICT_JUDGED_BY
        # The numbers the gate was judged on travel with the verdict.
        assert verdict["stage_result"]["publication_gate_passed"] is True
        assert verdict["stage_result"]["best_model_reward"] == results["best_model_reward"]
        assert verdict_is_reusable(verdict)
        # D-A22: so does the gate it was judged under — the CURRENT block's
        # thresholds (the kind's keys only), digested.
        from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view

        block = self.STANCE_CONFIG["curriculum_kwargs"]
        assert verdict["gate"] == gate_config_view(block)
        assert verdict["gate"]["thresholds"] == {
            key: block[key]
            for key in (
                "max_unsupported_duty",
                "max_unsupported_duty_ucb",
                "min_avg_reward",
                "min_eval_episodes",
                "min_full_horizon_fraction",
            )
        }
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", verdict["gate_sha256"])
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(block))

    def test_judging_under_a_block_the_directory_did_not_train_under_is_logged(self, tmp_path, caplog):
        """D-B8: the notebook's JUDGE branch judges under the session's config; when that differs from
        the block stage_config.json recorded, it is a WARNING naming the thresholds — never a refusal,
        and the verdict (recording the gate judged under) is still written."""
        from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
        from environments.shared.result_bundle import read_gate_verdict

        stage_dir = self._stage_dir(tmp_path)
        trained_under = {**self.STANCE_CONFIG["curriculum_kwargs"], "max_unsupported_duty_ucb": 0.05, "timesteps": 5}
        (stage_dir / "stage_config.json").write_text(
            json.dumps(
                {
                    "task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": self.TASK},
                    "curriculum": trained_under,
                }
            ),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            self._apply(stage_dir, {"gate_kind": "stance_quality/v1", "passed": True, "failures": []})

        warnings = [r for r in caplog.records if "judged under a gate configuration that differs" in r.message]
        assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
        assert "max_unsupported_duty_ucb: judged at 0.05, configured 0.02 now" in warnings[0].message
        assert "timesteps" not in warnings[0].message
        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["passed"] is True
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(self.STANCE_CONFIG["curriculum_kwargs"]))

        # The same block, retyped (timesteps is not the gate; 0.02 == 0.02): quiet.
        caplog.clear()
        (stage_dir / "stage_config.json").write_text(
            json.dumps(
                {
                    "task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": self.TASK},
                    "curriculum_kwargs": {**self.STANCE_CONFIG["curriculum_kwargs"], "min_eval_episodes": 40.0},
                }
            ),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            self._apply(stage_dir, {"gate_kind": "stance_quality/v1", "passed": True, "failures": []})
        assert not [r for r in caplog.records if "judged under a gate configuration" in r.message]

        # A DIFFERING block under the loaded-config spelling is read, not skipped: the warning names it.
        caplog.clear()
        (stage_dir / "stage_config.json").write_text(
            json.dumps(
                {
                    "task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": self.TASK},
                    "curriculum_kwargs": {**self.STANCE_CONFIG["curriculum_kwargs"], "max_unsupported_duty_ucb": 0.05},
                }
            ),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            self._apply(stage_dir, {"gate_kind": "stance_quality/v1", "passed": True, "failures": []})
        warnings = [r for r in caplog.records if "judged under a gate configuration that differs" in r.message]
        assert len(warnings) == 1
        assert "max_unsupported_duty_ucb: judged at 0.05, configured 0.02 now" in warnings[0].message

    def test_a_failure_is_recorded_too_and_is_not_reusable(self, tmp_path):
        from environments.shared.result_bundle import read_gate_verdict, verdict_is_reusable

        stage_dir = self._stage_dir(tmp_path)
        results = self._apply(stage_dir, None)

        assert results["publication_gate_passed"] is False
        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["passed"] is False
        assert verdict["failures"] == results["gate_failures"] and verdict["failures"]
        assert not verdict_is_reusable(verdict)

    def test_a_stage_without_a_handoff_pair_records_an_unreusable_verdict(self, tmp_path):
        from environments.shared.result_bundle import read_gate_verdict, verdict_is_reusable

        stage_dir = self._stage_dir(tmp_path, with_handoff=False)
        self._apply(stage_dir, {"gate_kind": "stance_quality/v1", "passed": True, "failures": []})

        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["passed"] is True
        assert verdict["checkpoint_sha256"] is None and verdict["normalization_sha256"] is None
        assert not verdict_is_reusable(verdict)

    def test_a_stage_the_manifest_does_not_declare_is_recorded_by_reference(self, tmp_path, caplog):
        from environments.shared.result_bundle import read_gate_verdict

        stage_dir = self._stage_dir(tmp_path)
        with caplog.at_level(logging.WARNING):
            self._apply(stage_dir, None, stage="warp")

        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["stage_id"] == "warp"
        assert any(
            "is not in the velociraptor_mongoliensis/anatomical manifest" in record.message for record in caplog.records
        )

    def test_no_stage_dir_means_no_verdict_and_the_gate_still_records(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        results: dict = {"best_model_reward": 2297.0}
        _apply_stage_gate(
            stage=1,
            stage_config=self.STANCE_CONFIG,
            stage_results=results,
            stance_report=None,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        assert results["publication_gate_passed"] is False
        assert not list(tmp_path.iterdir())

    def test_a_verdict_write_failure_never_costs_the_artifacts(self, tmp_path, caplog, monkeypatch):
        from environments.shared import result_bundle

        def explode(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(result_bundle, "write_gate_verdict", explode)
        stage_dir = self._stage_dir(tmp_path)
        with caplog.at_level(logging.WARNING):
            results = self._apply(stage_dir, {"gate_kind": "stance_quality/v1", "passed": True, "failures": []})
        assert results["publication_gate_passed"] is True
        assert not (stage_dir / "gate_verdict.json").exists()
        assert any("gate verdict could not be written" in record.message for record in caplog.records)


class TestRetiredTaskSuccessGate:
    """task_success/v1 was deleted with the hunt nodes (decision D-D59), with the evidence roll it needed.

    generate_stage_artifacts no longer rolls a task-success panel before the
    gate, and a frozen stage config that still declares the kind is recorded
    as a refusal naming the decision, with none of the kind's numbers copied
    onto the results or the verdict, whatever evidence the directory holds.
    """

    TASK_SUCCESS_CONFIG = {
        "name": "Hunt",
        "description": "Hunt prey",
        "env_kwargs": {"max_episode_steps": 1000},
        "ppo_kwargs": {"gamma": 0.99},
        "curriculum_kwargs": {
            "gate_kind": "task_success/v1",
            "gate_schema_version": 1,
            "min_success_lcb": 0.5,
            "min_eval_episodes": 30,
            "min_avg_reward": 361.0,
        },
    }

    def test_generate_stage_artifacts_rolls_no_task_success_evidence(self):
        import inspect

        from environments.shared.reporting import stage_artifacts

        assert not hasattr(stage_artifacts, "_write_task_success_evidence")
        assert "task_success" not in inspect.getsource(stage_artifacts.generate_stage_artifacts)

    def test_a_frozen_task_success_config_is_recorded_as_a_refusal_naming_the_decision(self, tmp_path):
        from environments.shared.reporting import save_evaluation_episodes
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate
        from environments.shared.result_bundle import read_gate_verdict

        models = tmp_path / "models"
        models.mkdir()
        (models / "robust_best_model.zip").write_bytes(b"weights")
        (models / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")
        save_evaluation_episodes(
            tmp_path,
            rewards=[600.0] * 30,
            lengths=[1000] * 30,
            forward_velocities=[1.0] * 30,
            distances=[5.0] * 30,
            successes=[True] * 29 + [False],
            evaluation_seed=3042,
            checkpoint_label="selected",
            checkpoint_path=models / "robust_best_model.zip",
            normalization_path=models / "robust_best_model_vecnorm.pkl",
        )
        results: dict = {"best_model_reward": 602.1, "best_model_length": 1000.0}
        _apply_stage_gate(
            stage=3,
            stage_config=self.TASK_SUCCESS_CONFIG,
            stage_results=results,
            stance_report=None,
            stage_dir=tmp_path,
            model_id=TYRANNOSAURUS_REX,
        )
        assert results["publication_gate_passed"] is False
        assert any(
            "'task_success/v1' was removed" in failure and "D-D59" in failure for failure in results["gate_failures"]
        )
        assert not {"best_model_success_count", "best_model_n_episodes", "best_model_success_lcb"} & set(results)
        verdict = read_gate_verdict(tmp_path)
        assert verdict is not None and verdict["passed"] is False
        assert not {"best_model_success_count", "best_model_success_lcb"} & set(verdict["stage_result"])


def test_evaluate_stage_checkpoints_loads_a_missing_model_through_the_loader_before_any_rollout():
    """The notebook's evaluation (its infrastructure cell's until consolidation PR-14c): ``model=None`` (the chain
    loop's JUDGE branch) loads ``<final_path>.zip`` through ``load_sb3_model`` and validates its plant before any
    rollout, and the function returns the 5-tuple the chain loop's JUDGE branch unpacks."""
    import ast
    import inspect

    from environments.shared.reporting import stage_artifacts

    function = ast.parse(inspect.getsource(stage_artifacts.evaluate_stage_checkpoints)).body[0]
    assert isinstance(function, ast.FunctionDef)
    load_if = next(
        node for node in function.body if isinstance(node, ast.If) and ast.unparse(node.test) == "model is None"
    )
    assert "load_sb3_model(" in ast.unparse(load_if) and "validate_model_plant(" in ast.unparse(load_if)
    rolls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_eval_forward_vel"
    ]
    assert len(rolls) == 2 and load_if.lineno < min(node.lineno for node in rolls)
    returned = function.body[-1]
    assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Tuple) and len(returned.value.elts) == 5


def test_evaluate_stage_checkpoints_takes_every_session_fact_explicitly():
    """Only ``model`` has a default (``None``: the JUDGE branch loads the final checkpoint); the plant and the
    publication seed are required, because ``EVALUATION_SEED`` equals ``PUBLICATION_SEED_START`` only at SEED 42."""
    import inspect

    from environments.shared.reporting import stage_artifacts

    parameters = inspect.signature(stage_artifacts.evaluate_stage_checkpoints).parameters
    assert list(parameters) == [
        "species_cfg",
        "stage_config",
        "stage",
        "algorithm",
        "stage_dir",
        "final_path",
        "final_vecnorm_path",
        "timesteps",
        "duration_seconds",
        "plant_identity",
        "evaluation_seed",
        "model",
    ]
    assert parameters["model"].default is None
    assert all(p.default is inspect.Parameter.empty for name, p in parameters.items() if name != "model")


class TestStanceV2StageGate:
    """stance_quality/v2 through the post-stage entry points (decision D-D23).

    ``_write_stance_gate_report`` rolls the report for a v2 stage too (the
    stance-kind predicate), and ``_apply_stage_gate`` judges it through the
    v2 arm and copies the clean count, panel size, bound and statue reward
    onto ``stage_results`` -- and into ``gate_verdict.json`` -- only from a
    report the judge admits.
    """

    @staticmethod
    def _config():
        from .stance_v2_helpers import V2_CURRICULUM

        return {"name": "Balance", "description": "Stand", "curriculum_kwargs": dict(V2_CURRICULUM), "env_kwargs": {}}

    def test_the_report_is_rolled_for_a_v2_stage(self, tmp_path, monkeypatch):
        import environments.shared.reporting.stance_report as stance_report
        from environments.shared.reporting.stage_artifacts import _write_stance_gate_report

        from .stance_v2_helpers import handoff_pair, v2_report

        handoff_pair(tmp_path)
        seen = {}

        def fake_build(species, stage, *, stage_config, model_path, vecnorm_path, episodes):
            seen.update(model_path=model_path, episodes=episodes)
            return v2_report(tmp_path)

        monkeypatch.setattr(stance_report, "build_stance_gate_report", fake_build)
        report = _write_stance_gate_report(
            model_id=TYRANNOSAURUS_REX,
            stage=1,
            stage_config=self._config(),
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
        )
        assert report is not None and report["scored_gate_kind"] == "stance_quality/v2"
        assert seen == {"model_path": str(tmp_path / "models" / "robust_best_model.zip"), "episodes": 40}
        assert (tmp_path / "stance_gate_report.json").is_file() and (tmp_path / "stance_panel_selected.csv").is_file()

    def test_an_admitted_report_passes_and_its_numbers_travel_with_the_verdict(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate
        from environments.shared.result_bundle import read_gate_verdict

        from .stance_v2_helpers import v2_report

        report = v2_report(tmp_path)
        results: dict = {"stage": 1, "best_model_reward": 1e9}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=report,
            stage_dir=tmp_path,
            model_id=TYRANNOSAURUS_REX,
        )
        assert results["gate_passed"] is True and results["gate_failures"] == []
        assert results["stance_clean_count"] == 40 and results["stance_n_episodes"] == 40
        assert results["stance_clean_fraction"] == 1.0
        assert results["stance_clean_lcb"] == pytest.approx(0.9278, abs=1e-4)
        assert results["stance_statue_mean_reward"] == pytest.approx(1000.0)
        verdict = read_gate_verdict(tmp_path)
        assert verdict is not None and verdict["passed"] is True
        assert verdict["stage_result"]["stance_clean_count"] == 40
        assert verdict["stage_result"]["stance_clean_fraction"] == 1.0

    def test_a_refused_report_fails_and_copies_none_of_its_numbers(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        from .stance_v2_helpers import v2_report

        report = v2_report(tmp_path)
        (tmp_path / "models" / "robust_best_model.zip").write_bytes(b"another checkpoint")
        results: dict = {"stage": 1}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=report,
            stage_dir=tmp_path,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        assert results["gate_passed"] is False
        assert any("not the handoff robust_best_model's" in failure for failure in results["gate_failures"])
        assert not {"stance_clean_count", "stance_clean_lcb"} & set(results)

    def test_a_v2_stage_without_a_report_fails_closed(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        results: dict = {"stage": 1, "best_model_reward": 1e9}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=None,
            stage_dir=tmp_path,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        assert results["publication_gate_passed"] is False
        assert any("no stance gate report was produced" in failure for failure in results["gate_failures"])


class TestStanceHeadingStageGate:
    """The six-heading certificate (decision D-D32) through the post-stage entry points.

    ``_write_stance_gate_report`` rolls the certificate -- never a yaw-zero
    report -- for a heading-kind stage, ``_apply_stage_gate`` judges it and
    copies its grouped numbers from an admitted certificate only, a stage
    without one fails, and the probes run after the verdict and annotate the
    certificate's 0 degree cell, with the heading probe left out.
    """

    @staticmethod
    def _config():
        from .stance_heading_helpers import HEADING_CURRICULUM

        return {
            "name": "Balance",
            "description": "Stand",
            "curriculum_kwargs": dict(HEADING_CURRICULUM),
            "env_kwargs": {"max_episode_steps": 1000},
        }

    def test_the_certificate_is_rolled_for_a_heading_stage(self, tmp_path, monkeypatch):
        import environments.shared.reporting.stance_report as stance_report
        from environments.shared.reporting.stage_artifacts import _write_stance_gate_report

        from .stance_heading_helpers import heading_report

        seen = {}

        def fake_build(species, stage, *, stage_config, model_path, vecnorm_path, episodes):
            seen.update(model_path=model_path, episodes=episodes)
            return heading_report(tmp_path)

        monkeypatch.setattr(stance_report, "build_stance_heading_report", fake_build)
        monkeypatch.setattr(
            stance_report, "build_stance_gate_report", lambda *a, **k: pytest.fail("a yaw-zero report was rolled")
        )
        from .stance_v2_helpers import handoff_pair

        handoff_pair(tmp_path)
        report = _write_stance_gate_report(
            model_id=TYRANNOSAURUS_REX,
            stage=1,
            stage_config=self._config(),
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
        )
        # Re-pinned from /v1 when the report began naming the embodiment (decision D-D33).
        assert report is not None and report["schema"] == "mesozoic.stance-heading-report/v2"
        assert seen == {"model_path": str(tmp_path / "models" / "robust_best_model.zip"), "episodes": 40}
        assert (tmp_path / "stance_heading_report.json").is_file()
        assert (tmp_path / "stance_heading_panel_selected.csv").is_file()
        assert (
            not (tmp_path / "stance_gate_report.json").exists()
            and not (tmp_path / "stance_panel_selected.csv").exists()
        )

    def test_an_admitted_certificate_passes_and_its_numbers_travel_with_the_verdict(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate
        from environments.shared.result_bundle import read_gate_verdict

        from .stance_heading_helpers import heading_report

        results: dict = {"stage": 1, "best_model_reward": 1e9}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=heading_report(tmp_path),
            stage_dir=tmp_path,
            model_id=TYRANNOSAURUS_REX,
        )
        assert results["gate_passed"] is True and results["gate_failures"] == []
        assert results["stance_heading_n_seeds"] == 40 and results["stance_heading_clean_seeds"] == 40
        assert results["stance_heading_clean_seed_fraction"] == 1.0
        assert results["stance_heading_clean_seed_lcb"] == pytest.approx(0.9278, abs=1e-4)
        assert results["stance_heading_statue_mean_reward_min"] == pytest.approx(1000.0)
        assert not {"stance_clean_count", "stance_clean_lcb"} & set(results)
        verdict = read_gate_verdict(tmp_path)
        assert verdict is not None and verdict["passed"] is True
        assert verdict["stage_result"]["stance_heading_clean_seeds"] == 40

    def test_the_numbers_that_travel_are_the_re_derived_ones(self, tmp_path):
        """The judge cross-checks the verdict and clean counts, not the bound: a bound, fraction or seed count
        edited in the report's result block is never what the verdict records."""
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate
        from environments.shared.result_bundle import read_gate_verdict

        from .stance_heading_helpers import FALL, heading_report

        report = heading_report(tmp_path, defects={(index, 90.0): FALL for index in range(2)})
        honest = dict(report["result"])
        assert honest["n_clean_seeds"] == 38 and honest["clean_seed_lcb"] < 0.9
        report["result"].update(n_seeds=400, clean_seed_fraction=1.0, clean_seed_lcb=0.99)
        results: dict = {"stage": 1, "best_model_reward": 1e9}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=report,
            stage_dir=tmp_path,
            model_id=TYRANNOSAURUS_REX,
        )
        assert results["gate_passed"] is True
        assert results["stance_heading_n_seeds"] == 40 and results["stance_heading_clean_seeds"] == 38
        assert results["stance_heading_clean_seed_fraction"] == 38 / 40
        assert results["stance_heading_clean_seed_lcb"] == honest["clean_seed_lcb"]
        verdict = read_gate_verdict(tmp_path)
        assert verdict is not None
        assert verdict["stage_result"]["stance_heading_clean_seed_lcb"] == honest["clean_seed_lcb"]

    def test_a_heading_stage_without_a_certificate_fails_closed(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        from .stance_v2_helpers import V2_CURRICULUM, v2_report

        results: dict = {"stage": 1, "best_model_reward": 1e9}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=None,
            stage_dir=tmp_path,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        assert results["publication_gate_passed"] is False
        assert any("no six-heading certificate" in failure for failure in results["gate_failures"])
        # A refused report copies none of its numbers: here a yaw-zero v2 PASS handed in its place.
        results = {"stage": 1}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=v2_report(tmp_path / "v2", V2_CURRICULUM),
            stage_dir=tmp_path,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        assert results["gate_passed"] is False and not any(key.startswith("stance_") for key in results)
        # Nor does a heading certificate the judge refuses: here one whose handoff was retrained after rolling.
        from .stance_heading_helpers import heading_report

        refused_dir = tmp_path / "refused"
        report = heading_report(refused_dir)
        assert report["passed"] is True
        (refused_dir / "models" / "robust_best_model.zip").write_bytes(b"retrained after judging")
        results = {"stage": 1}
        _apply_stage_gate(
            stage=1,
            stage_config=self._config(),
            stage_results=results,
            stance_report=report,
            stage_dir=refused_dir,
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        assert results["gate_passed"] is False
        assert any("scored checkpoint" in failure for failure in results["gate_failures"])
        assert not any(key.startswith("stance_") for key in results)

    @staticmethod
    def _record_probes(monkeypatch) -> list:
        from environments.shared.reporting import stage_artifacts

        calls: list = []
        for name in (
            "_write_filtered_action_probe",
            "_write_constant_hold_probe",
            "_write_constant_hold_ablation",
            "_write_impulse_probe",
            "_write_heading_probe",
        ):
            monkeypatch.setattr(stage_artifacts, name, lambda _name=name, **kwargs: calls.append((_name, kwargs)))
        return calls

    def test_probes_annotate_the_zero_heading_cell(self, tmp_path, monkeypatch):
        from environments.shared.reporting.stage_artifacts import _run_stance_probes

        from .stance_heading_helpers import heading_report

        calls = self._record_probes(monkeypatch)
        report = heading_report(tmp_path)
        _run_stance_probes(
            model_id=TYRANNOSAURUS_REX,
            stage=1,
            stage_config=self._config(),
            stage_dir=tmp_path,
            model_dir=tmp_path / "models",
            report=report,
        )
        # The heading probe is not rolled: the certificate judged every heading.
        assert [name for name, _ in calls] == [
            "_write_filtered_action_probe",
            "_write_constant_hold_probe",
            "_write_constant_hold_ablation",
            "_write_impulse_probe",
        ]
        measured = dict(calls)["_write_constant_hold_probe"]["measured"]
        assert measured["heading_cell_deg"] == 0.0 and measured["scored_gate_kind"] == "stance_quality/v2"
        zero = next(heading for heading in report["headings"] if heading["yaw_deg"] == 0.0)
        assert measured["episode_evidence"] is zero["episode_evidence"]
        assert dict(calls)["_write_impulse_probe"]["settle_steps"] == 200

    def test_the_certificate_runs_before_the_verdict_and_the_probes_after(self):
        import inspect

        from environments.shared.reporting import stage_artifacts

        source = inspect.getsource(stage_artifacts.generate_stage_artifacts)
        assert source.index("_write_stance_gate_report(") < source.index("_apply_stage_gate(")
        assert source.index("_apply_stage_gate(") < source.index("_run_stance_probes(")
        # The report-only training history is drawn after the verdict, and the probes still run after it.
        assert source.index("_apply_stage_gate(") < source.index("_write_training_history(")
        assert source.index("_write_training_history(") < source.index("_run_stance_probes(")
        dispatch = inspect.getsource(stage_artifacts._write_stance_gate_report)
        # The heading kind is dispatched before the v1/v2 builder is reached, after the shared handoff selection.
        assert dispatch.index("select_handoff_checkpoint(") < dispatch.index("_write_stance_heading_certificate(")
        assert dispatch.index("_write_stance_heading_certificate(") < dispatch.index("build_stance_gate_report(")


def test_collected_results_carry_the_heading_bar_and_grouped_count_on_a_heading_row_only():
    from environments.shared.reporting.csv_output import build_results_csv_rows

    from .stance_heading_helpers import HEADING_CURRICULUM

    heading_config = {"name": "Balance", "curriculum_kwargs": dict(HEADING_CURRICULUM)}
    other_config = {"name": "Walk", "curriculum_kwargs": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 1.0}}
    base = {"timesteps": 1, "publication_gate_passed": True}
    rows = build_results_csv_rows(
        [
            {**base, "stage": 1, "stance_heading_clean_seeds": 37, "stance_heading_clean_seed_lcb": 0.817388},
            {**base, "stage": 2},
        ],
        {1: heading_config, 2: other_config},
        TYRANNOSAURUS_REX,
        "PPO",
        42,
    )
    assert rows[0]["clean_stance_lcb_threshold"] == 0.8 and rows[0]["stance_eval_episodes"] == 40
    assert rows[0]["stance_heading_clean_seeds"] == 37 and rows[0]["stance_heading_clean_seed_lcb"] == 0.8174
    assert "stance_clean_count" not in rows[0]
    assert not {"clean_stance_lcb_threshold", "stance_heading_clean_seeds"} & set(rows[1])


def test_collected_results_carry_the_v2_bar_on_a_v2_row_only():
    """csv_output: the certifying bar and what was measured on a stance_quality/v2 row; no new column elsewhere."""
    from environments.shared.reporting.csv_output import build_results_csv_rows

    from .stance_v2_helpers import V2_CURRICULUM

    v2_config = {"name": "Balance", "curriculum_kwargs": dict(V2_CURRICULUM)}
    other_config = {"name": "Walk", "curriculum_kwargs": {"gate_kind": "reward_and_length/v1", "min_avg_reward": 1.0}}
    base = {"timesteps": 1, "publication_gate_passed": True}
    rows = build_results_csv_rows(
        [
            {**base, "stage": 1, "stance_clean_count": 38, "stance_clean_lcb": 0.85079},
            {**base, "stage": 2},
        ],
        {1: v2_config, 2: other_config},
        TYRANNOSAURUS_REX,
        "PPO",
        42,
    )
    assert rows[0]["clean_stance_lcb_threshold"] == 0.8 and rows[0]["stance_eval_episodes"] == 40
    assert rows[0]["stance_clean_count"] == 38 and rows[0]["stance_clean_lcb"] == 0.8508
    assert not {"clean_stance_lcb_threshold", "stance_clean_count"} & set(rows[1])


@pytest.mark.parametrize(
    ("env_kwargs", "mode"),
    [({}, "none"), ({"command_mode": "heading_and_speed"}, "heading_and_speed")],
    ids=["none", "live"],
)
def test_evaluate_stage_checkpoints_loads_the_statistics_under_the_stage_command_mode(
    tmp_path, monkeypatch, env_kwargs, mode
):
    """The notebook's evaluation loads the final sidecar through decision D-D60's load rule under the stage's own
    command mode (``load_vecnorm_stats(command_mode=)``): refused off identity for a live stage, re-imposed for a
    "none" one, never judged under another stage's mode."""
    from types import SimpleNamespace

    from environments.shared import curriculum, evaluation, train_base
    from environments.shared.reporting import stage_artifacts

    stage_dir = tmp_path / "stage2"
    (stage_dir / "models").mkdir(parents=True)
    final_path = stage_dir / "models" / "stage2_final"
    final_path.with_suffix(".zip").write_bytes(b"weights")
    final_vecnorm = stage_dir / "models" / "stage2_final_vecnorm.pkl"
    final_vecnorm.write_bytes(b"stats")
    loads: list[tuple[str, str]] = []

    def load_vecnorm_stats(path, *_envs, command_mode, **_kwargs):
        loads.append((str(path), command_mode))
        return True

    class Env:
        dt = 0.01

        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(train_base, "create_vec_env", lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(curriculum, "load_vecnorm_stats", load_vecnorm_stats)
    monkeypatch.setattr(
        evaluation,
        "eval_policy",
        lambda model, env, keys, *, n_episodes: (
            [1.0] * n_episodes,
            [10] * n_episodes,
            [0.1] * n_episodes,
            [],
            [0.5] * n_episodes,
        ),
    )
    monkeypatch.setattr(stage_artifacts, "select_handoff_checkpoint", lambda _model_dir: None)
    monkeypatch.setattr(
        stage_artifacts,
        "build_stage_results_from_eval_data",
        lambda *a, **k: {"best_eval_reward": "", "best_eval_std": "", "best_eval_timestep": ""},
    )
    species_cfg = SimpleNamespace(model_id=VELOCIRAPTOR_MONGOLIENSIS, env_class=Env, success_keys=[])
    stage_artifacts.evaluate_stage_checkpoints(
        species_cfg,
        {"env_kwargs": env_kwargs},
        2,
        "ppo",
        stage_dir,
        final_path=final_path,
        final_vecnorm_path=final_vecnorm,
        timesteps=1,
        duration_seconds=1.0,
        plant_identity=None,
        evaluation_seed=101,
        model=object(),
    )
    assert loads == [(str(final_vecnorm), mode)]


@pytest.mark.parametrize(
    ("success_keys", "successes", "expected"),
    [([], [], None), (["course_complete"], [1.0] * 6 + [0.0] * 24, 0.2)],
    ids=["no-success-keys", "course-complete"],
)
def test_evaluate_stage_checkpoints_records_a_success_rate_only_for_a_model_with_success_keys(
    tmp_path, monkeypatch, success_keys, successes, expected
):
    """Decision D-D59's amendment: without success keys the panels record no flag, so no rate (never a zero) and no
    task_success column; with them both rates come from the panel. The panel is EVALUATION_PANEL_EPISODES long."""
    import csv
    from types import SimpleNamespace

    from environments.shared import curriculum, evaluation, train_base
    from environments.shared.reporting import stage_artifacts

    stage_dir = tmp_path / "stage1"
    (stage_dir / "models").mkdir(parents=True)
    final_path = stage_dir / "models" / "stage1_final"
    final_path.with_suffix(".zip").write_bytes(b"weights")
    final_vecnorm = stage_dir / "models" / "stage1_final_vecnorm.pkl"
    final_vecnorm.write_bytes(b"stats")
    panels: list[int] = []

    def eval_policy(model, env, keys, *, n_episodes):
        panels.append(n_episodes)
        assert list(keys) == success_keys
        return [1.0] * n_episodes, [10] * n_episodes, [0.1] * n_episodes, list(successes), [0.5] * n_episodes

    class Env:
        dt = 0.01

        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(train_base, "create_vec_env", lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(curriculum, "load_vecnorm_stats", lambda *a, **k: True)
    monkeypatch.setattr(evaluation, "eval_policy", eval_policy)
    monkeypatch.setattr(stage_artifacts, "select_handoff_checkpoint", lambda _model_dir: None)
    monkeypatch.setattr(
        stage_artifacts,
        "build_stage_results_from_eval_data",
        lambda *a, **k: {"best_eval_reward": "", "best_eval_std": "", "best_eval_timestep": ""},
    )
    species_cfg = SimpleNamespace(model_id=VELOCIRAPTOR_MONGOLIENSIS, env_class=Env, success_keys=success_keys)
    *_, results = stage_artifacts.evaluate_stage_checkpoints(
        species_cfg,
        {"env_kwargs": {}},
        1,
        "ppo",
        stage_dir,
        final_path=final_path,
        final_vecnorm_path=final_vecnorm,
        timesteps=1,
        duration_seconds=1.0,
        plant_identity=None,
        evaluation_seed=101,
        model=object(),
    )
    assert panels == [stage_artifacts.EVALUATION_PANEL_EPISODES]
    with (stage_dir / "evaluation_final.csv").open(newline="", encoding="utf-8") as source:
        fieldnames = csv.DictReader(source).fieldnames or []
    if expected is None:
        assert "mean_success_rate" not in results and results["best_model_success_rate"] == ""
        assert "task_success" not in fieldnames
    else:
        assert results["mean_success_rate"] == pytest.approx(expected)
        assert results["best_model_success_rate"] == pytest.approx(expected)
        assert "task_success" in fieldnames
