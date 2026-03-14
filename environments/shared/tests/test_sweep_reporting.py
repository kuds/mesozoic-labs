"""Tests for reporting integration used by sweep trial artifacts."""

from unittest.mock import MagicMock, patch


class TestGenerateTrialArtifacts:
    """Test generate_stage_artifacts produces stage summary and videos.

    This tests the shared function that both the sweep trial worker
    and the training notebook use to produce consistent artifacts.
    """

    def test_writes_stage_summary_and_records_videos(self, tmp_path):
        """After a trial, stage_summary.txt is written and videos are attempted."""
        import numpy as np

        from environments.shared.reporting import generate_stage_artifacts

        # Setup directory structure matching what train() produces
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "best_model.zip").write_bytes(b"fake")
        (model_dir / "best_model_vecnorm.pkl").write_bytes(b"fake")
        (model_dir / "stage1_final.zip").write_bytes(b"fake")
        (model_dir / "stage1_final_vecnorm.pkl").write_bytes(b"fake")

        # Create evaluations.npz
        rewards = np.array([[10.0, 12.0], [20.0, 22.0]])
        lengths = np.array([[100, 110], [200, 210]])
        timesteps = np.array([50000, 100000])
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=rewards,
            ep_lengths=lengths,
            timesteps=timesteps,
        )

        mock_species_cfg = MagicMock()
        mock_species_cfg.species = "velociraptor"
        mock_species_cfg.env_class = MagicMock

        stage_config = {
            "name": "Balance",
            "description": "Stand up",
            "env_kwargs": {"sim_dt": 0.01},
            "ppo_kwargs": {},
        }

        mock_sb3 = {
            "PPO": MagicMock(),
            "SAC": MagicMock(),
        }

        with (
            patch(
                "environments.shared.train_base._ensure_sb3",
                return_value=mock_sb3,
            ),
            patch(
                "environments.shared.evaluation.record_stage_video",
            ) as mock_video,
        ):
            results = generate_stage_artifacts(
                species_cfg=mock_species_cfg,
                stage_config=stage_config,
                stage=1,
                algorithm="ppo",
                stage_dir=tmp_path,
                seed=42,
                timesteps=100_000,
            )

        # stage_summary.txt should exist
        summary = tmp_path / "stage_summary.txt"
        assert summary.exists()
        text = summary.read_text()
        assert "Balance" in text
        assert "Velociraptor" in text

        # Videos should be recorded for both best and final models
        assert mock_video.call_count == 2
        labels = [call.kwargs["label"] for call in mock_video.call_args_list]
        assert "best" in labels
        assert "final" in labels

        # Returned results should have best eval metrics from evaluations.npz
        assert results["best_eval_reward"] == 21.0
        assert results["best_eval_timestep"] == 100000

        # Training graphs should be generated when matplotlib is available
        try:
            import matplotlib  # noqa: F401

            assert (tmp_path / "training_curves.png").exists()
            assert (tmp_path / "locomotion_health.png").exists()
            assert (tmp_path / "behavioral_metrics.png").exists()
        except ImportError:
            pass  # graphs are skipped gracefully without matplotlib
