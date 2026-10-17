"""Tests for the training visualization utilities."""

import numpy as np
import pytest

from environments.shared.model_id import ModelId

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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
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
            model_id=ModelId("velociraptor_mongoliensis", "anatomical"),
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0
        # Bipedal: single axes (no diagonal pair subplot)
        assert len(fig.axes) == 1

    def test_biped_with_nan_quadruped_keys_gets_no_diagonal_panel(self, tmp_path):
        """The real biped npz CONTAINS rr/rl keys, as all-NaN series.

        DiagnosticsCallback saves every info key for every species, so key
        presence is not evidence of quadrupedal data. Keying the layout off
        ``key in diag`` rendered an empty "diagonal pair" panel for bipeds and
        routed their two lines through the four-foot branch under FR/FL
        labels (narrow-tolerance postmortem, item 6). The fixture above omits
        the keys entirely, which is why it never caught this.
        """
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_foot_contacts

        ts = np.array([50000, 100000, 150000])
        nan_series = np.full(3, np.nan)
        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=ts,
            r_foot_contact=np.array([0.5, 0.8, 0.3]),
            l_foot_contact=np.array([0.3, 0.6, 0.7]),
            rr_foot_contact=nan_series,
            rl_foot_contact=nan_series,
        )

        stage_configs = {1: {"name": "Balance"}}
        save_path = tmp_path / "foot_contacts.png"

        fig = plot_foot_contacts(
            [(1, tmp_path)],
            stage_configs,
            model_id=ModelId("tyrannosaurus_rex", "anatomical"),
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()
        assert len(fig.axes) == 1
        # And the two real feet are plotted under biped labels, not FR/FL.
        labels = [line.get_label() for line in fig.axes[0].get_lines()]
        assert any("Right" in label for label in labels)
        assert not any("FR" in label for label in labels)

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
            model_id=ModelId("brachiosaurus_altithorax", "anatomical"),
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
            model_id=ModelId("tyrannosaurus_rex", "anatomical"),
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )

        assert save_path.exists()


class TestPlotStanceDiagnostics:
    def test_saves_stance_dashboard_when_fields_exist(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_stance_diagnostics

        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=np.array([50000, 100000]),
            bilateral_support_duty=np.array([0.5, 0.9]),
            single_support_duty=np.array([0.5, 0.1]),
            foot_load_imbalance=np.array([0.3, 0.1]),
            head_tip_z=np.array([0.6, 0.7]),
            pelvis_height=np.array([0.8, 0.9]),
            drift_distance=np.array([0.2, 0.1]),
            pelvis_yaw_vel=np.array([0.1, 0.05]),
        )
        save_path = tmp_path / "stance_diagnostics.png"
        fig = plot_stance_diagnostics(
            [(1, tmp_path)],
            {1: {"name": "Balance"}},
            model_id=ModelId("tyrannosaurus_rex", "anatomical"),
            algorithm="ppo",
            save_path=save_path,
            show=False,
        )

        assert fig is not None
        assert save_path.exists()
        assert len(fig.axes) == 6

    def test_returns_none_without_stance_fields(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_stance_diagnostics

        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=np.array([50000, 100000]),
            r_foot_contact=np.array([1.0, 1.0]),
            l_foot_contact=np.array([1.0, 1.0]),
            pelvis_height=np.array([0.8, 0.9]),
            drift_distance=np.array([0.2, 0.1]),
        )
        result = plot_stance_diagnostics(
            [(1, tmp_path)],
            {1: {"name": "Balance"}},
            model_id=ModelId("tyrannosaurus_rex", "anatomical"),
            algorithm="ppo",
            save_path=tmp_path / "stance_diagnostics.png",
            show=False,
        )

        assert result is None
        assert not (tmp_path / "stance_diagnostics.png").exists()

    def test_returns_none_for_other_species_even_with_support_fields(self, tmp_path):
        matplotlib.use("Agg")

        from environments.shared.visualization import plot_stance_diagnostics

        np.savez(
            str(tmp_path / "diagnostics.npz"),
            timesteps=np.array([50000, 100000]),
            bilateral_support_duty=np.array([0.5, 0.9]),
            foot_load_imbalance=np.array([0.3, 0.1]),
        )
        result = plot_stance_diagnostics(
            [(1, tmp_path)],
            {1: {"name": "Balance"}},
            model_id=ModelId("brachiosaurus_altithorax", "anatomical"),
            algorithm="ppo",
            save_path=tmp_path / "stance_diagnostics.png",
            show=False,
        )

        assert result is None
        assert not (tmp_path / "stance_diagnostics.png").exists()


def _gate_progress(stage_dir, *, every_group=True):
    """A synthetic ``gate_progress.npz`` of five evaluations: every panel's keys, or the reward keys only."""
    import numpy as np

    steps = np.arange(1, 6) * 131_072
    ramp = np.linspace(0.1, 0.5, 5)
    keys = {"timesteps": steps, "mean_reward": 100.0 * ramp, "full_horizon_fraction": np.ones(5)}
    if every_group:
        keys.update(
            {
                "r_sole_cop_fore_aft": ramp,
                "l_sole_cop_fore_aft": ramp - 0.05,
                "r_foot_cop_fore_aft": ramp + 0.1,
                "max_sole_cop_fore_aft_mean": ramp,  # a panel key, never a foot
                "max_sole_cop_fore_aft_over_bar_fraction": np.array([0.0, 0.0, 0.1, 0.3, np.nan]),
                "screen_v2_clean_lcb": np.array([0.1, 0.3, 0.5, 0.6, np.nan]),
                "com_fore_aft_mean_m": -0.01 + 0.01 * ramp,
                "xcom_toe_margin_min_m": 0.06 - 0.05 * ramp,
                "leg_dc_l_minus_r_ankle_deg": ramp,
                "leg_dc_l_r_mean_ankle_deg": -ramp,
                "leg_q_l_minus_r_ankle_deg": ramp / 2,
                "action_dc_per_actuator": np.zeros((5, 3)),  # a matrix: not a series
            }
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez(stage_dir / "gate_progress.npz", **keys)
    return stage_dir


class TestPlotGateProgress:
    def test_one_panel_per_group_with_the_declared_bars(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import _gate_progress_groups, plot_gate_progress

        stage_dir = _gate_progress(tmp_path / "run" / "01_stance")
        out = tmp_path / "figure.png"
        figure = plot_gate_progress(
            stage_dir,
            curriculum_kwargs={"max_sole_cop_fore_aft": 0.8, "min_clean_stance_lcb": 0.8},
            model_id=ModelId("tyrannosaurus_rex", "anatomical"),
            save_path=out,
        )
        assert out.stat().st_size > 0
        import numpy as np

        with np.load(stage_dir / "gate_progress.npz") as data:
            groups = _gate_progress_groups(tuple(key for key in data.files if data[key].ndim == 1))
        assert list(groups) == ["reward", "cop", "over_bar", "screen", "balance", "leg_dc"]
        assert "max_sole_cop_fore_aft_mean" not in groups["cop"]
        assert len([axis for axis in figure.axes if axis.get_ylabel() != "full-horizon fraction"]) == 6

    def test_a_reward_only_file_draws_one_panel_and_no_file_draws_none(self, tmp_path):
        pytest.importorskip("matplotlib")
        from environments.shared.visualization import plot_gate_progress

        stage_dir = _gate_progress(tmp_path / "plain", every_group=False)
        figure = plot_gate_progress(stage_dir, save_path=tmp_path / "plain.png")
        assert len(figure.axes) == 2  # the reward panel and its full-horizon twin
        assert plot_gate_progress(tmp_path / "missing") is None

    def test_the_script_writes_to_the_current_directory_and_refuses_the_run_directory(
        self, tmp_path, monkeypatch, capsys
    ):
        pytest.importorskip("matplotlib")
        import json

        from environments.shared.scripts import plot_gate_progress as script

        run = tmp_path / "run"
        stage_dir = _gate_progress(run / "01_stance")
        (run / "provenance.json").write_text(json.dumps({"run_id": "test"}), encoding="utf-8")
        (stage_dir / "stage_config.json").write_text(
            json.dumps(
                {
                    "species": "tyrannosaurus_rex",
                    "embodiment": "anatomical",
                    "curriculum": {"max_sole_cop_fore_aft": 0.8},
                }
            ),
            encoding="utf-8",
        )
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert script.main([str(stage_dir)]) == 0
        assert (elsewhere / "gate_progress_01_stance.png").stat().st_size > 0
        before = sorted(path.relative_to(run) for path in run.rglob("*"))
        for inside in (run / "figure.png", stage_dir / "figures" / "gate_progress.png"):
            assert script.main([str(stage_dir), "--out", str(inside)]) == 2
            assert "result_bundle/manifest.py" in capsys.readouterr().err
        assert sorted(path.relative_to(run) for path in run.rglob("*")) == before  # nothing written into the run
        assert script.main([str(tmp_path / "missing")]) == 1
