"""Tests for jax_viz module (visualization utilities for JAX training)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from environments.shared.jax_eval import EvalResults
from environments.shared.jax_viz import _smooth, plot_locomotion_diagnostics, plot_training_curves


class TestSmooth:
    def test_short_array(self):
        values = [1.0, 2.0, 3.0]
        result = _smooth(values, window=50)
        np.testing.assert_array_equal(result, values)

    def test_identity_window_1(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _smooth(values, window=1)
        np.testing.assert_array_almost_equal(result, values)

    def test_smoothing(self):
        values = [0.0] * 50 + [10.0] * 50
        result = _smooth(values, window=10)
        assert len(result) == 100
        # Middle values should be smoothed (between 0 and 10)
        assert 0.0 < result[50] < 10.0

    def test_preserves_length(self):
        values = np.random.randn(200)
        result = _smooth(values, window=20)
        assert len(result) == len(values)

    def test_empty_array(self):
        result = _smooth([], window=10)
        assert len(result) == 0


class TestPlotTrainingCurves:
    def _make_diagnostics(self, n):
        return [
            {
                "grad_norm": np.random.rand(),
                "policy_loss": np.random.rand(),
                "value_loss": np.random.rand(),
                "entropy": np.random.rand(),
                "approx_kl": np.random.rand() * 0.01,
                "fall_rate": np.random.rand() * 0.5,
                "episode_length": float(np.random.randint(50, 500)),
                "clip_fraction": np.random.rand() * 0.3,
                "mean_std": np.random.rand() * 0.5,
            }
            for _ in range(n)
        ]

    def test_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")

        n = 50
        fig = plot_training_curves(
            reward_history=list(np.random.randn(n)),
            loss_history=list(np.random.randn(n)),
            episode_return_history=list(np.random.randn(n)),
            diagnostics_history=self._make_diagnostics(n),
            species="trex",
            stage=1,
            show=False,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saves_to_file(self):
        import matplotlib

        matplotlib.use("Agg")

        n = 30
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "curves.png"
            fig = plot_training_curves(
                reward_history=list(np.random.randn(n)),
                loss_history=list(np.random.randn(n)),
                episode_return_history=list(np.random.randn(n)),
                diagnostics_history=self._make_diagnostics(n),
                output_path=path,
                show=False,
            )
            assert path.exists()
            assert path.stat().st_size > 0
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_handles_nan_episode_returns(self):
        import matplotlib

        matplotlib.use("Agg")

        n = 20
        returns = [float("nan")] * n
        fig = plot_training_curves(
            reward_history=list(np.random.randn(n)),
            loss_history=list(np.random.randn(n)),
            episode_return_history=returns,
            diagnostics_history=self._make_diagnostics(n),
            show=False,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_handles_small_history(self):
        import matplotlib

        matplotlib.use("Agg")

        n = 3
        fig = plot_training_curves(
            reward_history=list(np.random.randn(n)),
            loss_history=list(np.random.randn(n)),
            episode_return_history=list(np.random.randn(n)),
            diagnostics_history=self._make_diagnostics(n),
            show=False,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotLocomotionDiagnostics:
    def _make_eval_results(self, n_steps=200):
        results = EvalResults()
        results.diag_tilt = list(np.random.rand(n_steps) * 0.5)
        results.diag_fwd_vel = list(np.random.randn(n_steps))
        results.diag_pelvis_h = list(np.random.rand(n_steps) * 0.5 + 0.5)
        results.diag_l_foot = list(np.random.rand(n_steps) * 10)
        results.diag_r_foot = list(np.random.rand(n_steps) * 10)
        results.diag_energy = list(np.random.rand(n_steps) * 0.1)
        results.diag_reward_components = {
            "forward": list(np.random.randn(n_steps)),
            "alive": [0.1] * n_steps,
            "energy": list(-np.random.rand(n_steps) * 0.01),
            "posture": list(-np.random.rand(n_steps) * 0.1),
        }
        return results

    def test_returns_figures(self):
        import matplotlib

        matplotlib.use("Agg")

        results = self._make_eval_results()
        fig_main, fig_foot = plot_locomotion_diagnostics(
            results,
            species="trex",
            stage=1,
            show=False,
        )
        assert fig_main is not None
        assert fig_foot is not None
        import matplotlib.pyplot as plt

        plt.close(fig_main)
        plt.close(fig_foot)

    def test_saves_to_directory(self):
        import matplotlib

        matplotlib.use("Agg")

        results = self._make_eval_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            fig_main, fig_foot = plot_locomotion_diagnostics(
                results,
                output_dir=tmpdir,
                show=False,
            )
            assert (Path(tmpdir) / "locomotion_health.png").exists()
            assert (Path(tmpdir) / "foot_contacts.png").exists()
            import matplotlib.pyplot as plt

            plt.close(fig_main)
            if fig_foot:
                plt.close(fig_foot)

    def test_no_foot_data(self):
        import matplotlib

        matplotlib.use("Agg")

        results = self._make_eval_results()
        results.diag_l_foot = []
        results.diag_r_foot = []
        fig_main, fig_foot = plot_locomotion_diagnostics(
            results,
            show=False,
        )
        assert fig_main is not None
        assert fig_foot is None
        import matplotlib.pyplot as plt

        plt.close(fig_main)
