"""Tests for the training visualization utilities."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")


class TestPlotTrainingCurves:
    """Tests for plot_training_curves."""

    def test_saves_png_from_evaluations_npz(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_training_curves

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

        stage_configs = {1: {"name": "Balance", "curriculum_kwargs": {}}}
        save_path = tmp_path / "training_curves.png"

        fig = plot_training_curves(
            [(1, tmp_path)],
            stage_configs,
            species="velociraptor",
            algorithm="ppo",
            save_path=save_path,
        )
        import matplotlib.pyplot as plt

        plt.close(fig)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_missing_eval_log(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_training_curves

        stage_configs = {1: {"name": "Balance"}}
        save_path = tmp_path / "training_curves.png"

        fig = plot_training_curves(
            [(1, tmp_path)],
            stage_configs,
            species="velociraptor",
            algorithm="ppo",
            save_path=save_path,
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        # File is still saved (just empty plots)
        assert save_path.exists()


class TestPlotDiagnosticsGraphs:
    """Tests for plot_diagnostics_graphs."""

    def test_saves_locomotion_health_and_behavioral_metrics(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_diagnostics_graphs

        # Create minimal diagnostics.npz
        ts = np.array([50000, 100000])
        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=ts,
            tilt_angle=np.array([0.1, 0.05]),
            forward_vel=np.array([0.5, 1.0]),
            pelvis_height=np.array([0.8, 0.85]),
            reward_energy=np.array([-0.1, -0.2]),
            reward_forward=np.array([0.5, 1.0]),
        )

        stage_configs = {1: {"name": "Balance"}}

        fig1, fig2 = plot_diagnostics_graphs(
            [(1, tmp_path)],
            stage_configs,
            species="velociraptor",
            algorithm="ppo",
            save_dir=tmp_path,
            show=False,
        )

        assert (tmp_path / "locomotion_health.png").exists()
        assert (tmp_path / "behavioral_metrics.png").exists()

    def test_plots_distance_traveled_and_drift(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_diagnostics_graphs

        ts = np.array([50000, 100000])
        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=ts,
            tilt_angle=np.array([0.1, 0.05]),
            forward_vel=np.array([0.5, 1.0]),
            pelvis_height=np.array([0.8, 0.85]),
            reward_energy=np.array([-0.1, -0.2]),
            distance_traveled=np.array([0.5, 2.3]),
            drift_distance=np.array([0.1, 0.3]),
        )

        stage_configs = {1: {"name": "Balance"}}

        fig1, fig2 = plot_diagnostics_graphs(
            [(1, tmp_path)],
            stage_configs,
            species="velociraptor",
            algorithm="ppo",
            save_dir=tmp_path,
            show=False,
        )

        assert (tmp_path / "behavioral_metrics.png").exists()
        # Verify the figure has 3 rows (3x2 grid)
        assert fig2.axes[0] is not None  # Should have at least 6 axes
        assert len(fig2.axes) == 6

    def test_handles_empty_diagnostics(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_diagnostics_graphs

        stage_configs = {1: {"name": "Balance"}}

        fig1, fig2 = plot_diagnostics_graphs(
            [(1, tmp_path)],
            stage_configs,
            species="velociraptor",
            algorithm="ppo",
            save_dir=tmp_path,
            show=False,
        )

        # Files still created (with empty/"no data" plots)
        assert (tmp_path / "locomotion_health.png").exists()
        assert (tmp_path / "behavioral_metrics.png").exists()
