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
        # No hunting data → 2x2 grid (4 axes)
        assert fig2.axes[0] is not None
        assert len(fig2.axes) == 4

    def test_expands_to_3x2_with_hunting_data(self, tmp_path):
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
            prey_distance=np.array([5.0, 4.5]),
            strike_success=np.array([0.0, 0.1]),
        )

        stage_configs = {3: {"name": "Hunting"}}

        fig1, fig2 = plot_diagnostics_graphs(
            [(3, tmp_path)],
            stage_configs,
            species="velociraptor",
            algorithm="ppo",
            save_dir=tmp_path,
            show=False,
        )

        assert (tmp_path / "behavioral_metrics.png").exists()
        # Hunting data present → 3x2 grid (6 axes)
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


class TestPlotFootContacts:
    """Tests for plot_foot_contacts."""

    def test_bipedal_saves_png(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_foot_contacts

        ts = np.array([50000, 100000, 150000])
        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=ts,
            r_foot_contact=np.array([0.5, 0.8, 0.3]),
            l_foot_contact=np.array([0.3, 0.6, 0.7]),
        )

        stage_configs = {1: {"name": "Balance"}}
        save_path = tmp_path / "foot_contacts.png"

        fig = plot_foot_contacts(
            [(1, tmp_path)],
            stage_configs,
            species="velociraptor",
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0
        # Bipedal: single axes (no diagonal pair subplot)
        assert len(fig.axes) == 1

    def test_quadrupedal_saves_png_with_diagonal_pairs(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_foot_contacts

        ts = np.array([50000, 100000, 150000])
        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=ts,
            r_foot_contact=np.array([0.5, 0.8, 0.3]),
            l_foot_contact=np.array([0.3, 0.6, 0.7]),
            rr_foot_contact=np.array([0.4, 0.7, 0.2]),
            rl_foot_contact=np.array([0.6, 0.5, 0.8]),
        )

        stage_configs = {2: {"name": "Locomotion"}}
        save_path = tmp_path / "foot_contacts.png"

        fig = plot_foot_contacts(
            [(2, tmp_path)],
            stage_configs,
            species="brachiosaurus",
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0
        # Quadrupedal: 2 subplots (individual feet + diagonal pairs)
        assert len(fig.axes) == 2

    def test_handles_missing_diagnostics(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_foot_contacts

        stage_configs = {1: {"name": "Balance"}}
        save_path = tmp_path / "foot_contacts.png"

        plot_foot_contacts(
            [(1, tmp_path)],
            stage_configs,
            species="trex",
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()
