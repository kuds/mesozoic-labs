"""Tests for environments.shared.reporting.stage_artifacts."""

import csv
import json
import os
from pathlib import Path

import pytest

from environments.shared.plant_contract import MODEL_IDENTITY_ATTRIBUTE
from environments.shared.reporting import build_stage_results_from_eval_data, save_jax_stage_artifacts

from .reporting_helpers import make_stage_result, plant_identity


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

    # Review ER4: the sweep workers pass no stage_results, so the velocity and
    # success numbers the gate judges come from here — and they were
    # hardcoded to 0.0 although metrics.json (already opened for the
    # duration) carries the real panel.

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


class _FakeEvalResults:
    """Minimal stand-in for jax_eval.EvalResults."""

    def __init__(self):
        self.rewards = [50.0, 55.0, 60.0]
        self.lengths = [200, 210, 220]
        self.forward_vels = [0.5, 0.6, 0.55]
        self.distances = [1.0, 1.2, 1.1]
        self.successes = [True, True, True]
        self.diag_tilt = [0.1, 0.2, 0.15]
        self.diag_fwd_vel = [0.5, 0.6, 0.55]
        self.diag_pelvis_h = [0.7, 0.72, 0.71]
        self.diag_energy = [0.01, 0.02, 0.015]
        self.diag_l_foot = [1.0, 0.0, 1.0]
        self.diag_r_foot = [0.0, 1.0, 0.0]
        self.diag_reward_components = {
            "forward": [0.3, 0.4, 0.35],
            "alive": [0.1, 0.1, 0.1],
        }
        self.diag_reward_diagnostics = {
            "bilateral_support_quality": [0.4, 0.6, 0.8],
            "alive_gate": [0.7, 0.8, 0.9],
        }


class _FakeFinalEvalResults:
    """Terminal-policy evidence deliberately different from the selected policy."""

    def __init__(self):
        self.rewards = [10.0, 15.0, 20.0]
        self.lengths = [100, 110, 120]
        self.forward_vels = [0.1, 0.2, 0.3]
        self.distances = [0.5, 1.0, 1.5]
        self.successes = [False, False, True]


class TestSaveJaxStageArtifacts:
    """Tests for save_jax_stage_artifacts."""

    @pytest.fixture(autouse=True)
    def _clean_repository_state(self, monkeypatch):
        from environments.shared.result_bundle import provenance

        monkeypatch.setattr(
            provenance,
            "_repository_state",
            lambda _root: {
                "repository_url": "https://github.com/kuds/mesozoic-labs.git",
                "repository_commit": "a" * 40,
                "repository_dirty": False,
                "repository_patch_sha256": None,
            },
        )

    def _call(self, tmp_path, stage=1, species="velociraptor", **overrides):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        stage_dir = run_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_config = {
            "name": f"Stage {stage}",
            "description": f"Curriculum stage {stage}",
            "env_kwargs": {"forward_vel_weight": float(stage)},
            "jax_kwargs": {"learning_rate": 3e-4},
            "curriculum_kwargs": {"min_avg_reward": 50.0},
        }
        stage_results = make_stage_result(
            stage=stage,
            model_path=str(stage_dir / "models" / "best_model.pkl"),
            best_eval_reward=55.0,
            best_eval_timestep=50000,
            mean_distance_traveled=2.5,
        )

        kwargs = dict(
            species=species,
            stage=stage,
            stage_config=stage_config,
            stage_results=stage_results,
            stage_dir=stage_dir,
            run_dir=run_dir,
            eval_results=_FakeEvalResults(),
            params={"dense": [1.0, 2.0]},
            obs_rms=None,
            seed=42,
            num_envs=2048,
            reward_cfg={"forward_vel_weight": 1.0},
            best_params={"dense": [3.0, 4.0]},
            best_reward=55.0,
            best_update=10,
            plant_identity=plant_identity(),
        )
        kwargs.update(overrides)
        return save_jax_stage_artifacts(**kwargs), stage_dir, run_dir

    def test_returns_all_artifact_paths(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        expected_keys = {
            "stage_summary",
            "stage_config",
            "evaluation_episodes",
            "final_evaluation_episodes",
            "collected_results_csv",
            "diagnostics",
            "best_model",
            "final_model",
            "stage_result",
            "training_summary",
            "provenance",
            "artifact_manifest",
        }
        assert set(paths.keys()) == expected_keys

    def test_all_files_exist(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        for name, path in paths.items():
            assert path.exists(), f"{name} not found at {path}"

    def test_stage_summary_content(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        text = paths["stage_summary"].read_text()
        assert "Velociraptor" in text
        assert "Stage 1" in text

    def test_stage_config_json(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        data = json.loads(paths["stage_config"].read_text())
        assert data["species"] == "velociraptor"
        assert data["stage"] == 1
        assert data["algorithm"] == "JAX_PPO"
        assert data["plant_identity"] == plant_identity().to_dict()
        assert (paths["stage_config"].parent / "plant_identity.json").exists()

    def test_collected_results_csv(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        with open(paths["collected_results_csv"]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["species"] == "velociraptor"
        assert rows[0]["algorithm"] == "PPO"
        assert rows[0]["backend"] == "jax-mjx"
        assert rows[0]["plant_physics_sha256"] == plant_identity().physics_sha256

    def test_diagnostics_npz(self, tmp_path):
        import numpy as np

        paths, _, _ = self._call(tmp_path)
        data = np.load(paths["diagnostics"])
        assert "tilt_angle" in data
        assert "timesteps" in data
        assert "forward_vel" in data
        assert "l_foot_contact" in data
        assert "reward_forward" in data
        assert "bilateral_support_quality" in data
        assert "alive_gate" in data
        assert all(len(data[key]) == len(data["timesteps"]) for key in data.files)

    def test_trex_stage1_diagnostics_derive_support_duty_and_load_share(self, tmp_path):
        from dataclasses import replace

        import numpy as np

        trex_identity = replace(
            plant_identity(),
            species="trex",
            model_path="environments/trex/assets/trex.xml",
            physics_revision=5,
            policy_interface_revision=7,
            nq=28,
            nv=27,
            nu=21,
            observation_dim=61,
            action_dim=21,
        )
        paths, _, _ = self._call(
            tmp_path,
            species="trex",
            plant_identity=trex_identity,
        )
        data = np.load(paths["diagnostics"])

        np.testing.assert_array_equal(data["r_foot_contact_duty"], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(data["l_foot_contact_duty"], [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(data["bilateral_support_duty"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(data["single_support_duty"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(data["unsupported_duty"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data["r_foot_load_share"], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(data["l_foot_load_share"], [1.0, 0.0, 1.0])
        np.testing.assert_allclose(data["foot_load_balance"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data["foot_load_asymmetry"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(data["foot_load_imbalance"], [1.0, 1.0, 1.0])

    def test_diagnostics_npz_no_foot_data(self, tmp_path):
        import numpy as np

        eval_results = _FakeEvalResults()
        eval_results.diag_l_foot = []
        eval_results.diag_r_foot = []

        paths, _, _ = self._call(tmp_path, eval_results=eval_results)
        data = np.load(paths["diagnostics"])
        assert "tilt_angle" in data
        assert "l_foot_contact" not in data

    def test_best_model_checkpoint(self, tmp_path):
        import pickle

        paths, _, _ = self._call(tmp_path)
        with open(paths["best_model"], "rb") as f:
            ckpt = pickle.load(f)  # noqa: S301
        assert ckpt["params"] == {"dense": [3.0, 4.0]}
        assert ckpt["best_reward"] == 55.0
        assert ckpt[MODEL_IDENTITY_ATTRIBUTE] == plant_identity().to_dict()

    def test_final_model_checkpoint(self, tmp_path):
        import pickle

        paths, _, _ = self._call(tmp_path)
        with open(paths["final_model"], "rb") as f:
            ckpt = pickle.load(f)  # noqa: S301
        assert ckpt["params"] == {"dense": [1.0, 2.0]}
        assert ckpt[MODEL_IDENTITY_ATTRIBUTE] == plant_identity().to_dict()

    def test_training_summary_content(self, tmp_path):
        paths, _, _ = self._call(tmp_path)
        text = paths["training_summary"].read_text()
        assert "Velociraptor" in text
        assert "JAX/MJX PPO" in text
        assert (paths["training_summary"].parent / "plant_identity.json").exists()

    def test_partial_bundle_has_no_public_summary(self, tmp_path):
        paths, _, run_dir = self._call(tmp_path)
        from environments.shared.result_bundle import validate_result_bundle

        assert "summary" not in paths
        assert not (run_dir / "summary.json").exists()
        assert validate_result_bundle(run_dir, require_complete=False)["status"] == "partial"

    def test_csv_upserts_across_stages(self, tmp_path):
        """Each stage is represented once in the regenerated canonical CSV."""
        _, _, run_dir = self._call(tmp_path)
        self._call(tmp_path, stage=2)
        with open(run_dir / "collected_results.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["stage"] == "1"
        assert rows[1]["stage"] == "2"

    def test_repeated_stage_save_is_idempotent(self, tmp_path):
        _, _, run_dir = self._call(tmp_path)
        self._call(tmp_path)

        with open(run_dir / "collected_results.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["stage"] == "1"

    def test_rejects_a_skipped_prior_stage_before_writing(self, tmp_path):
        from environments.shared.result_bundle import ResultBundleError

        with pytest.raises(ResultBundleError, match="contiguous prefix"):
            self._call(tmp_path, stage=2)
        assert not (tmp_path / "run" / "provenance.json").exists()

    def test_preserves_preinitialized_colab_hardware_identity(self, tmp_path):
        from environments.shared.result_bundle import initialize_result_bundle

        run_dir = tmp_path / "run"
        initialize_result_bundle(
            run_dir,
            species="velociraptor",
            algorithm="JAX_PPO",
            backend="jax-mjx",
            seed=42,
            # The writer's default publication seed; it must differ from the
            # training seed or the provenance validator refuses the bundle.
            evaluation_seeds=[3042],
            evaluation_episodes=3,
            parallel_envs=2048,
            hardware="Google Colab (NVIDIA A100-SXM4-40GB)",
            plant_identity=plant_identity().to_dict(),
            run_id="preinitialized-jax-run",
        )

        paths, _, _ = self._call(tmp_path)
        provenance = json.loads(paths["provenance"].read_text())
        assert provenance["hardware"] == "Google Colab (NVIDIA A100-SXM4-40GB)"

    def test_three_stage_run_emits_valid_jax_summary(self, tmp_path):
        from environments.shared.result_bundle import validate_result_bundle
        from environments.shared.result_schema import validate_result_summary

        self._call(tmp_path, stage=1, backend_version="0.11.0")
        self._call(tmp_path, stage=2, backend_version="0.11.0")
        paths, _, run_dir = self._call(tmp_path, stage=3, backend_version="0.11.0")

        summary = json.loads(paths["summary"].read_text())
        validate_result_summary(
            summary,
            expected_species="velociraptor",
            require_complete=True,
            canonical_provenance=True,
        )
        assert summary["algorithm"] == "PPO"
        assert summary["backend"] == "jax-mjx"
        assert summary["backend_version"] == "0.11.0"
        assert set(summary["stages"]) == {"1", "2", "3"}
        assert validate_result_bundle(run_dir)["status"] == "canonical-valid"

    def test_selected_and_terminal_jax_metrics_remain_distinct(self, tmp_path):
        for stage in (1, 2, 3):
            self._call(
                tmp_path,
                stage=stage,
                backend_version="0.11.0",
                final_eval_results=_FakeFinalEvalResults(),
            )

        summary = json.loads((tmp_path / "run" / "summary.json").read_text())
        assert summary["stages"]["3"]["selected_model_reward"] == 55.0
        assert summary["stages"]["3"]["final_eval_reward"] == 15.0

    def test_completed_jax_bundle_rejects_stage_rewrite_before_mutation(self, tmp_path):
        from environments.shared.result_bundle import ResultBundleError

        self._call(tmp_path, stage=1, backend_version="0.11.0")
        self._call(tmp_path, stage=2, backend_version="0.11.0")
        _, _, run_dir = self._call(tmp_path, stage=3, backend_version="0.11.0")
        before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

        with pytest.raises(ResultBundleError, match="completed result bundle is immutable"):
            self._call(tmp_path, stage=3, backend_version="0.11.0")

        after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
        assert after == before

    def test_default_publication_seed_differs_from_the_training_default(self, tmp_path):
        """Review ER6: the validator refuses a publication seed shared with training."""
        paths, _, _ = self._call(tmp_path)
        seed_roles = json.loads(paths["provenance"].read_text())["seed_roles"]
        assert seed_roles["training"] == 42
        assert seed_roles["publication_evaluation"] == 3042

    def test_evaluation_csvs_bind_to_the_checkpoint_they_were_rolled_from(self, tmp_path):
        """Review RP3: every evidence row names the checkpoint by sha256."""
        from environments.shared.result_bundle import sha256_file

        paths, _, _ = self._call(tmp_path)
        for csv_key, model_key in (("evaluation_episodes", "best_model"), ("final_evaluation_episodes", "final_model")):
            with open(paths[csv_key]) as f:
                rows = list(csv.DictReader(f))
            assert rows, csv_key
            assert {row["checkpoint_sha256"] for row in rows} == {sha256_file(paths[model_key])}

    @staticmethod
    def _gated_config(stage):
        return {
            "name": f"Stage {stage}",
            "description": f"Curriculum stage {stage}",
            "env_kwargs": {"forward_vel_weight": float(stage)},
            "jax_kwargs": {"learning_rate": 3e-4},
            "curriculum_kwargs": {
                "gate_kind": "reward_and_length/v1",
                "gate_schema_version": 1,
                "min_avg_reward": 50.0,
            },
        }

    def test_persists_gate_kind_beside_the_verdict(self, tmp_path):
        """Review SS5: the stage record says which gate its verdict was earned under."""
        paths, _, _ = self._call(tmp_path, stage_config=self._gated_config(1))
        record = json.loads(paths["stage_result"].read_text())
        assert record["gate_kind"] == "reward_and_length/v1"
        assert record["gate_schema_version"] == 1

    def test_summary_carries_gate_kind_for_every_stage_and_still_validates(self, tmp_path):
        from environments.shared.result_schema import validate_result_summary

        for stage in (1, 2, 3):
            self._call(tmp_path, stage=stage, backend_version="0.11.0", stage_config=self._gated_config(stage))

        summary = json.loads((tmp_path / "run" / "summary.json").read_text())
        assert {stage["gate_kind"] for stage in summary["stages"].values()} == {"reward_and_length/v1"}
        assert {stage["gate_schema_version"] for stage in summary["stages"].values()} == {1}
        validate_result_summary(
            summary, expected_species="velociraptor", require_complete=True, canonical_provenance=True
        )

    def _crash_publishing(self, monkeypatch, filename):
        """Fail the atomic publish of one artifact, as a lost mount would."""
        real_replace = os.replace

        def replace_unless_target(src, dst):
            if Path(dst).name == filename:
                raise OSError("mount went away")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", replace_unless_target)
        return real_replace

    def test_a_crash_publishing_stage_result_keeps_the_previous_record(self, tmp_path, monkeypatch):
        """Review ER5: a truncated stage_result.json wedged every later save for the run."""
        paths, stage_dir, _ = self._call(tmp_path)
        before = paths["stage_result"].read_bytes()

        real_replace = self._crash_publishing(monkeypatch, "stage_result.json")
        with pytest.raises(OSError, match="mount went away"):
            self._call(tmp_path)

        assert paths["stage_result"].read_bytes() == before
        assert not list(stage_dir.glob(".stage_result.json.*"))
        # The record was never truncated, so the next save still goes through.
        monkeypatch.setattr(os, "replace", real_replace)
        self._call(tmp_path)

    def test_a_crash_publishing_diagnostics_keeps_the_previous_archive(self, tmp_path, monkeypatch):
        import numpy as np

        paths, stage_dir, _ = self._call(tmp_path)
        before = paths["diagnostics"].read_bytes()

        self._crash_publishing(monkeypatch, "diagnostics.npz")
        with pytest.raises(OSError, match="mount went away"):
            self._call(tmp_path)

        assert paths["diagnostics"].read_bytes() == before
        assert np.load(str(paths["diagnostics"]))["tilt_angle"].shape == (3,)
        assert not list(stage_dir.glob(".diagnostics.npz.*"))


class TestApplyStageGate:
    """The wiring: generate_stage_artifacts must record a verdict, always.

    Both trainers read `publication_gate_passed` off the dict this writes. If
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

    On gate failure the notebook calls `disconnect_runtime` and then raises,
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
        """Pre-gate callers (and the JAX path's own ordering) must still work."""
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


class TestStageResultDiscoveryAcrossLayouts:
    """The prior-stage scans must see the same names the save accepts."""

    def test_iter_finds_records_in_both_generations(self, tmp_path):
        from environments.shared.reporting.stage_artifacts import _iter_stage_result_paths

        (tmp_path / "stage1").mkdir()
        (tmp_path / "stage1" / "stage_result.json").write_text("{}")
        (tmp_path / "02_locomotion").mkdir()
        (tmp_path / "02_locomotion" / "stage_result.json").write_text("{}")
        found = sorted(path.parent.name for path in _iter_stage_result_paths(tmp_path))
        assert found == ["02_locomotion", "stage1"]
