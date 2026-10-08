"""Tests for ``generate_stage_artifacts`` on a trainer-shaped stage directory."""

from unittest.mock import MagicMock, patch

import pytest

from environments.shared.species_names import species_display_names


@pytest.mark.parametrize(
    "species,stage",
    [(species, 1) for species in species_display_names()]
    + [(species, "recovery") for species in ("trex", "compsognathus", "compsognathus_robot")],
)
def test_stance_replay_producer_enables_the_same_outputs_and_matched_checkpoints(tmp_path, species, stage):
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
    from environments.shared.reporting.stage_artifacts import _record_stage_replays
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import stage_label

    models = tmp_path / "models"
    models.mkdir()
    final = f"{stage_label(stage)}_final"
    for name in ("best_model", "robust_best_model", final):
        (models / f"{name}.zip").write_bytes(b"model")
        (models / f"{name}_vecnorm.pkl").write_bytes(b"normalization")
    model = MagicMock()
    attach_plant_identity(model, current_plant_identity(species))
    algorithm = MagicMock()
    algorithm.load.return_value = model
    replays = tmp_path / "replays"
    with (
        patch("environments.shared.policy_loading._ensure_sb3", return_value={"PPO": algorithm, "SAC": algorithm}),
        patch("environments.shared.evaluation.record_stage_video") as video,
    ):
        _record_stage_replays(
            species_cfg=get_species_config(species),
            stage_config=load_stage_config(species, stage),
            stage=stage,
            algorithm="ppo",
            stage_dir=tmp_path,
            replays_out=replays,
            model_dir=models,
            seed=42,
            stage_results={},
            allow_legacy_plant=False,
        )
    calls = [call.kwargs for call in video.call_args_list]
    assert [call["label"] for call in calls] == ["selected", "final"]
    for call, name in zip(calls, ("robust_best_model", final)):
        assert call["vecnorm_path"] == str(models / f"{name}_vecnorm.pkl")
        assert call["output_dir"] == replays
        assert call["collect_stance_diagnostics"] is True
        assert set(call["camera_views"]) == {"side", "front"}
        # Other species retain their own camera distance, rather than the
        # 3.4 m framing that makes the small Compsognathus barely visible.
        if species not in ("trex", "brachiosaurus"):
            assert all("distance" not in preset for preset in call["camera_views"].values())
        if species == "brachiosaurus":
            assert call["camera_views"]["front"]["distance"] > call["env_class"]._camera_distance


class TestGenerateTrialArtifacts:
    """Test generate_stage_artifacts produces stage summary and videos.

    This tests the shared function the training notebook uses to
    produce its stage artifacts.
    """

    def test_writes_stage_summary_and_records_videos(self, tmp_path):
        """After a stage, stage_summary.txt is written and videos are attempted."""
        import numpy as np

        from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
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

        fake_model = MagicMock()
        attach_plant_identity(fake_model, current_plant_identity("velociraptor", verify_generated=False))
        mock_algorithm = MagicMock()
        mock_algorithm.load.return_value = fake_model
        mock_sb3 = {"PPO": mock_algorithm, "SAC": mock_algorithm}

        with (
            patch(
                "environments.shared.policy_loading._ensure_sb3",
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
        assert "Stance" in text
        assert "Velociraptor" in text

        # Videos should be recorded for the selected and final checkpoints.
        # "selected", not "best": the replay comes from the same selector that
        # decides the next-stage handoff and that evaluation_selected.csv is
        # evidence for, so the two describe one policy.
        assert mock_video.call_count == 2
        labels = [call.kwargs["label"] for call in mock_video.call_args_list]
        assert "selected" in labels
        assert "final" in labels

        # Returned results should have best eval metrics from evaluations.npz
        assert results["best_eval_reward"] == 21.0
        assert results["best_eval_timestep"] == 100000

        # Training graphs should be generated when matplotlib is available,
        # into stage_dir/figures/ rather than loose in the stage root.
        try:
            import matplotlib  # noqa: F401

            from environments.shared.reporting import stage_layout

            figures = stage_layout.figures_dir(tmp_path)
            assert (figures / "training_curves.png").exists()
            assert (figures / "locomotion_health.png").exists()
            assert (figures / "behavioral_metrics.png").exists()
            assert not list(tmp_path.glob("*.png")), "figures must not be loose in the stage root"
        except ImportError:
            pass  # graphs are skipped gracefully without matplotlib


def test_generate_stage_artifacts_takes_sim_dt_from_the_nodes_env(tmp_path):
    """Without precomputed results, sim time uses a bare env of this node's task."""
    from environments.shared.reporting import stage_artifacts

    class ProbeEnv:
        dt = 0.02
        created_with: dict = {}
        closed = False

        def __init__(self, **kwargs):
            ProbeEnv.created_with = kwargs

        def close(self):
            ProbeEnv.closed = True

    class Stop(Exception):
        pass

    captured = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        raise Stop

    species_cfg = MagicMock()
    species_cfg.species = "compsognathus"
    species_cfg.env_class = ProbeEnv
    stage_config = {"name": "Locomotion", "description": "Walk", "env_kwargs": {"alive_bonus": 0.5}}
    with patch.object(stage_artifacts, "build_stage_results_from_eval_data", side_effect=fake_build):
        with pytest.raises(Stop):
            stage_artifacts.generate_stage_artifacts(
                species_cfg=species_cfg,
                stage_config=stage_config,
                stage=2,
                algorithm="ppo",
                stage_dir=tmp_path,
                seed=42,
                timesteps=1,
            )
    assert captured["sim_dt"] == 0.02
    assert ProbeEnv.created_with == {"alive_bonus": 0.5}
    assert ProbeEnv.closed


@pytest.mark.parametrize("env_path", ["CompsognathusBiologicalEnv", "CompsognathusRobotEnv"])
def test_the_compsognathus_pair_steps_at_twenty_milliseconds(env_path):
    """The premise of the probe: a 0.01 s default halves this pair's printed sim time."""
    from environments.compsognathus.envs import compsognathus_env

    env = getattr(compsognathus_env, env_path)()
    try:
        assert env.dt == pytest.approx(0.02)
    finally:
        env.close()
