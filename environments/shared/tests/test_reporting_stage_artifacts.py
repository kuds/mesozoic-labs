"""Tests for environments.shared.reporting.stage_artifacts."""

import json
import logging
import re

import pytest

from environments.shared.reporting import build_stage_results_from_eval_data

from .reporting_helpers import plant_identity


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

        _apply_stage_gate(stage=2, stage_config=config, stage_results=results, stance_report=None)

        assert results["gate_passed"] is False
        assert any("no forward-velocity measurement" in failure for failure in results["gate_failures"])
        assert not any("0.00 m/s" in failure for failure in results["gate_failures"])

    def test_a_measured_velocity_reaches_the_gate(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        (tmp_path / "models").mkdir()
        (tmp_path / "metrics.json").write_text(json.dumps({"mean_forward_vel": 1.5, "std_forward_vel": 0.1}))
        config = self._velocity_gated()
        results = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=config, timesteps=10)

        _apply_stage_gate(stage=2, stage_config=config, stage_results=results, stance_report=None)

        assert (results["gate_passed"], results["gate_failures"]) == (True, [])

    def test_the_stage_summary_says_not_measured_instead_of_zero(self, tmp_path):
        from environments.shared.reporting import write_stage_summary

        (tmp_path / "models").mkdir()
        results = build_stage_results_from_eval_data(tmp_path, stage=2, stage_config=self._LOCO, timesteps=10)

        text = write_stage_summary(tmp_path, results, "velociraptor", "PPO").read_text()

        assert "Avg fwd vel:    not measured" in text
        assert "0.00 +/- 0.00 m/s" not in text


class TestSaveStageConfigKeepsTheCurriculumTable:
    """The SB3 config writer records ``[curriculum]`` verbatim, a recorded backend sub-table included."""

    def test_the_sb3_config_writer_still_records_the_shared_table_verbatim(self, tmp_path):
        """generate_stage_artifacts hands save_stage_config the raw config and the SB3 gate judges the
        shared bar (thresholds_from_configs ignores the table), so the SB3 artifacts keep recording it,
        sub-table and all."""
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
        path = save_stage_config(tmp_path, stage=1, stage_config=stage_config, algorithm="PPO", species="velociraptor")
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
            "trex",
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
            "trex",
            "PPO",
        )
        summary = (tmp_path / "stage_summary.txt").read_text(encoding="utf-8")
        assert "Verdict:      PASS" in summary

    def test_a_summary_with_no_verdict_omits_the_section(self, tmp_path):
        """Pre-gate callers must still work."""
        from environments.shared.reporting import text_summaries

        text_summaries.write_stage_summary(tmp_path, self._results(), "trex", "PPO")
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

    def _apply(self, stage_dir, stance_report, *, species="velociraptor", stage=1):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        results = {"best_model_reward": 2297.0, "mean_reward": 2200.0, "timesteps": 10}
        _apply_stage_gate(
            stage=stage,
            stage_config=self.STANCE_CONFIG,
            stage_results=results,
            stance_report=stance_report,
            stage_dir=stage_dir,
            species=species,
        )
        return results

    def test_a_pass_is_recorded_hash_bound_to_the_handoff_it_judged(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import GATE_VERDICT_JUDGED_BY
        from environments.shared.result_bundle import read_gate_verdict, sha256_file, verdict_is_reusable

        stage_dir = self._stage_dir(tmp_path)
        results = self._apply(stage_dir, {"gate_kind": "stance_quality/v1", "passed": True, "failures": []})

        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["passed"] is True and verdict["failures"] == []
        assert (verdict["species"], verdict["stage"], verdict["stage_id"]) == ("velociraptor", 1, "stance")
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
        assert any("is not in the velociraptor manifest" in record.message for record in caplog.records)

    def test_no_stage_dir_means_no_verdict_and_the_gate_still_records(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        results: dict = {"best_model_reward": 2297.0}
        _apply_stage_gate(stage=1, stage_config=self.STANCE_CONFIG, stage_results=results, stance_report=None)
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


class TestTaskSuccessEvidence:
    """generate_stage_artifacts gives task_success/v1 the stance treatment (D-B12 amendment).

    The kind is judged from evaluation_selected.csv; a caller that rolled no
    bound panel wrote none, so before the gate the directory
    must hold one bound to the handoff — kept when the trainer's panel
    wrote it, rolled from the handoff pair on the publication seed otherwise.
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

    @staticmethod
    def _handoff(stage_dir):
        models = stage_dir / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "robust_best_model.zip").write_bytes(b"weights")
        (models / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")
        return models

    @staticmethod
    def _stub_rollout(monkeypatch, successes, *, seen):
        """Stand in for the SB3 env/model/rollout so the roller's plumbing is what is tested."""
        from types import SimpleNamespace

        from environments.shared import curriculum as curriculum_package
        from environments.shared import evaluation, plant_contract, train_base

        class _Env:
            training = True
            norm_reward = True

            def close(self):
                seen["closed"] = True

        def create_vec_env(species_cfg, stage_configs, stage, n_envs, seed, **kwargs):
            seen["seed"] = seed
            seen["n_envs"] = n_envs
            return _Env()

        class _Alg:
            @staticmethod
            def load(path, env=None, **load_kwargs):
                seen["loaded"] = path
                seen["load_seed"] = load_kwargs.get("seed", "absent")
                return object()

        def eval_policy(model, env, success_keys, n_episodes):
            seen["episodes"] = n_episodes
            n = n_episodes
            return [600.0] * n, [1000] * n, [1.0] * n, list(successes[:n]), [5.0] * n

        monkeypatch.setattr(train_base, "create_vec_env", create_vec_env)
        monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: {"PPO": _Alg, "SAC": _Alg})
        monkeypatch.setattr(evaluation, "eval_policy", eval_policy)
        monkeypatch.setattr(plant_contract, "current_plant_identity", lambda species: SimpleNamespace(species=species))
        monkeypatch.setattr(plant_contract, "validate_model_plant", lambda *a, **k: None)
        monkeypatch.setattr(curriculum_package, "load_vecnorm_stats", lambda *a, **k: True)

    def _roll(self, stage_dir, stage_config=None):
        from types import SimpleNamespace

        from environments.shared.reporting.stage_artifacts import _write_task_success_evidence

        return _write_task_success_evidence(
            species_cfg=SimpleNamespace(species="trex", success_keys=["bite_success"]),
            stage=3,
            stage_config=stage_config or self.TASK_SUCCESS_CONFIG,
            stage_dir=stage_dir,
            model_dir=stage_dir / "models",
            algorithm="ppo",
        )

    def test_other_kinds_roll_nothing(self, tmp_path, monkeypatch):
        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 30, seen=seen)
        self._handoff(tmp_path)
        config = dict(self.TASK_SUCCESS_CONFIG, curriculum_kwargs={"gate_kind": "reward_and_length/v1"})
        assert self._roll(tmp_path, config) is None
        assert seen == {} and not (tmp_path / "evaluation_selected.csv").exists()

    def test_rolls_the_handoff_pair_on_the_publication_seed_at_min_eval_episodes(self, tmp_path, monkeypatch):
        import csv

        from environments.shared.result_bundle import sha256_file

        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 20 + [False] * 10, seen=seen)
        models = self._handoff(tmp_path)
        written = self._roll(tmp_path)
        assert written == tmp_path / "evaluation_selected.csv"
        assert seen["seed"] == 3042 and seen["n_envs"] == 1 and seen["episodes"] == 30
        assert seen["loaded"] == str(models / "robust_best_model") and seen["closed"] is True
        assert seen["load_seed"] is None, "a seeded archive would re-seed the env with its training seed on load"
        with written.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 30
        assert {row["checkpoint_sha256"] for row in rows} == {sha256_file(models / "robust_best_model.zip")}
        assert {row["evaluation_seed"] for row in rows} == {"3042"}
        assert sum(row["task_success"] == "True" for row in rows) == 20

    def test_evidence_already_bound_to_the_handoff_is_kept(self, tmp_path, monkeypatch):
        from environments.shared.reporting import save_evaluation_episodes

        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 30, seen=seen)
        models = self._handoff(tmp_path)
        # At the declared panel size: a smaller bound file is re-rolled (see below).
        existing = save_evaluation_episodes(
            tmp_path,
            rewards=[1.0] * 30,
            lengths=[10] * 30,
            forward_velocities=[0.0] * 30,
            distances=[0.0] * 30,
            successes=[False] * 30,
            evaluation_seed=3042,
            checkpoint_label="selected",
            checkpoint_path=models / "robust_best_model.zip",
            normalization_path=models / "robust_best_model_vecnorm.pkl",
        )
        before = existing.read_bytes()
        assert self._roll(tmp_path) == existing
        assert seen == {} and existing.read_bytes() == before

    def test_evidence_for_another_checkpoint_is_re_rolled(self, tmp_path, monkeypatch, caplog):
        from environments.shared.reporting import save_evaluation_episodes

        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 30, seen=seen)
        models = self._handoff(tmp_path)
        other = tmp_path / "other.zip"
        other.write_bytes(b"another policy")
        save_evaluation_episodes(
            tmp_path,
            rewards=[1.0],
            lengths=[10],
            forward_velocities=[0.0],
            distances=[0.0],
            successes=[False],
            evaluation_seed=3042,
            checkpoint_label="selected",
            checkpoint_path=other,
            normalization_path=models / "robust_best_model_vecnorm.pkl",
        )
        with caplog.at_level(logging.WARNING):
            assert self._roll(tmp_path) == tmp_path / "evaluation_selected.csv"
        assert seen["episodes"] == 30
        assert "is not bound to the handoff" in caplog.text

    def test_evidence_under_another_vecnorm_sidecar_is_re_rolled(self, tmp_path, monkeypatch, caplog):
        """Bound means the pair: the judge and the backfill tool refuse a panel rolled under other
        observation statistics, so keeping it would leave the stage unjudgeable."""
        from environments.shared.reporting import save_evaluation_episodes

        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 30, seen=seen)
        models = self._handoff(tmp_path)
        other_stats = tmp_path / "other_vecnorm.pkl"
        other_stats.write_bytes(b"other statistics")
        save_evaluation_episodes(
            tmp_path,
            rewards=[1.0] * 30,
            lengths=[10] * 30,
            forward_velocities=[0.0] * 30,
            distances=[0.0] * 30,
            successes=[False] * 30,
            evaluation_seed=3042,
            checkpoint_label="selected",
            checkpoint_path=models / "robust_best_model.zip",
            normalization_path=other_stats,
        )
        with caplog.at_level(logging.WARNING):
            assert self._roll(tmp_path) == tmp_path / "evaluation_selected.csv"
        assert seen["episodes"] == 30 and "is not bound to the handoff" in caplog.text

    def test_a_bound_panel_below_min_eval_episodes_is_re_rolled(self, tmp_path, monkeypatch, caplog):
        """A 10-row trainer panel (--post-eval-episodes 10) bound to the
        handoff would make the judge refuse 'n_episodes 10 < min_eval_episodes 30' although a fresh
        roll makes the stage judgeable; it is re-rolled at the declared size."""
        from environments.shared.reporting import save_evaluation_episodes

        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 30, seen=seen)
        models = self._handoff(tmp_path)
        save_evaluation_episodes(
            tmp_path,
            rewards=[1.0] * 10,
            lengths=[10] * 10,
            forward_velocities=[0.0] * 10,
            distances=[0.0] * 10,
            successes=[True] * 10,
            evaluation_seed=3042,
            checkpoint_label="selected",
            checkpoint_path=models / "robust_best_model.zip",
            normalization_path=models / "robust_best_model_vecnorm.pkl",
        )
        with caplog.at_level(logging.WARNING):
            written = self._roll(tmp_path)
        assert written == tmp_path / "evaluation_selected.csv"
        assert seen["episodes"] == 30 and "fewer than min_eval_episodes 30" in caplog.text
        from environments.shared.reporting.gates import evaluate_stage_gate

        assert evaluate_stage_gate(self.TASK_SUCCESS_CONFIG["curriculum_kwargs"], {}, stage=3, stage_dir=tmp_path) == (
            True,
            [],
        )

    def test_zero_panel_episodes_skips_and_an_override_is_warned_about(self, tmp_path, monkeypatch, caplog):
        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 30, seen=seen)
        self._handoff(tmp_path)
        config = dict(self.TASK_SUCCESS_CONFIG)
        config["curriculum_kwargs"] = dict(config["curriculum_kwargs"], task_success_panel_episodes=0)
        assert self._roll(tmp_path, config) is None and seen == {}
        config["curriculum_kwargs"] = dict(config["curriculum_kwargs"], task_success_panel_episodes=8)
        with caplog.at_level(logging.WARNING):
            assert self._roll(tmp_path, config) is not None
        assert seen["episodes"] == 8 and "does not certify what the gate claims" in caplog.text

    def test_no_handoff_pair_rolls_nothing(self, tmp_path, monkeypatch, caplog):
        seen: dict = {}
        self._stub_rollout(monkeypatch, [True] * 30, seen=seen)
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "best_model.zip").write_bytes(b"w")  # no matched sidecar
        with caplog.at_level(logging.WARNING):
            assert self._roll(tmp_path) is None
        assert seen == {} and "matched _vecnorm.pkl" in caplog.text

    def test_a_failed_roll_never_costs_the_artifacts_and_the_gate_refuses(self, tmp_path, monkeypatch, caplog):
        from environments.shared import train_base
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate

        self._handoff(tmp_path)

        def explode(*args, **kwargs):
            raise RuntimeError("no simulator here")

        monkeypatch.setattr(train_base, "create_vec_env", explode)
        monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: {"PPO": object, "SAC": object})
        with caplog.at_level(logging.WARNING):
            assert self._roll(tmp_path) is None
        assert "could not be rolled" in caplog.text
        results: dict = {"best_model_reward": 1e9}
        _apply_stage_gate(
            stage=3,
            stage_config=self.TASK_SUCCESS_CONFIG,
            stage_results=results,
            stance_report=None,
            stage_dir=tmp_path,
        )
        assert results["publication_gate_passed"] is False
        assert any("absent" in failure for failure in results["gate_failures"])

    def test_generate_stage_artifacts_rolls_the_evidence_before_the_gate(self):
        import inspect

        from environments.shared.reporting import stage_artifacts

        source = inspect.getsource(stage_artifacts.generate_stage_artifacts)
        assert source.index("_write_task_success_evidence(") < source.index("_apply_stage_gate(")

    def test_the_gate_copies_the_judged_numbers_onto_the_results_and_the_verdict(self, tmp_path):
        from environments.shared.reporting import save_evaluation_episodes
        from environments.shared.reporting.stage_artifacts import _apply_stage_gate
        from environments.shared.result_bundle import read_gate_verdict

        models = self._handoff(tmp_path)
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
            species="trex",
        )
        assert results["publication_gate_passed"] is True
        assert (results["best_model_success_count"], results["best_model_n_episodes"]) == (29, 30)
        assert results["best_model_success_lcb"] == pytest.approx(0.851, abs=1e-3)
        verdict = read_gate_verdict(tmp_path)
        assert verdict is not None and verdict["passed"] is True
        assert verdict["stage_result"]["best_model_success_count"] == 29
        assert verdict["stage_result"]["best_model_success_lcb"] == pytest.approx(0.851, abs=1e-3)


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
