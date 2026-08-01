"""Tests for environments.shared.curriculum.early_stopping."""

import inspect
import math
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.curriculum import (
    EvalCollapseEarlyStopCallback,
    build_eval_collapse_early_stop_callback,
)
from environments.shared.curriculum.early_stopping import collapse_settings_from_config


class TestEvalCollapseEarlyStopCallback:
    """Test EvalCollapseEarlyStopCallback early stopping logic."""

    def test_raises_without_sb3(self):
        with patch("environments.shared.curriculum.sb3_compat._SB3_AVAILABLE", False):
            with pytest.raises(ImportError, match="stable-baselines3"):
                EvalCollapseEarlyStopCallback(eval_callback=MagicMock())

    def test_preserves_verbose_positional_slot(self):
        params = inspect.signature(EvalCollapseEarlyStopCallback.__init__).parameters

        assert list(params) == [
            "self",
            "eval_callback",
            "drop_fraction",
            "patience",
            "min_evals",
            "peak_floor",
            "verbose",
            "smoothing_window",
        ]
        assert params["verbose"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params["smoothing_window"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_returns_true_without_eval_results(self):
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock(spec=[])  # no evaluations_results
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 0.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 1
        assert cb._on_step() is True

    def test_returns_true_when_eval_results_empty(self):
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = []
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 0.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 1
        assert cb._on_step() is True

    def test_returns_true_before_min_evals(self, tmp_path):

        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = [[10.0, 20.0], [15.0, 25.0]]
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 0.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 1
        cb.min_evals = 5  # Need 5 evals, only have 2
        cb.drop_fraction = 0.3
        cb.patience = 3
        assert cb._on_step() is True

    def test_returns_true_no_new_eval(self, tmp_path):

        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = [[10.0]]
        cb._last_seen_n_evals = 1  # Already seen this eval
        cb._peak_score = float("-inf")
        cb.peak_floor = 0.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 1
        assert cb._on_step() is True

    def test_stops_after_patience_drops(self, tmp_path):

        # 6 evals: peak at eval 3 (50.0), then drops to 20.0
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = [
            [10.0, 10.0],
            [30.0, 30.0],
            [50.0, 50.0],  # peak
            [20.0, 20.0],  # drop 1
            [15.0, 15.0],  # drop 2
            [10.0, 10.0],  # drop 3 -> stop
        ]
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 0.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 1
        cb.min_evals = 3
        cb.drop_fraction = 0.3
        cb.patience = 1  # Stop after 1 drop
        cb.num_timesteps = 1000

        result = cb._on_step()
        # latest_mean = 10.0, peak = 50.0, threshold = 35.0
        # 10.0 < 35.0, so consecutive_drops=1 >= patience=1 -> stop
        assert result is False

    def test_resets_drops_on_recovery(self, tmp_path):

        # Evals: peak, drop, recovery
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = [
            [50.0, 50.0],
            [40.0, 40.0],
            [30.0, 30.0],
            [20.0, 20.0],
            [10.0, 10.0],
            [45.0, 45.0],  # recovery
        ]
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 0.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 1
        cb.min_evals = 5
        cb.drop_fraction = 0.3
        cb.patience = 5  # High patience
        cb.num_timesteps = 1000

        result = cb._on_step()
        # latest_mean = 45.0, peak = 50.0, threshold = 35.0
        # 45.0 >= 35.0 -> consecutive_drops reset to 0
        assert result is True
        assert cb._consecutive_drops == 0

    def test_rolling_median_ignores_lone_high_outlier(self):
        # A single variance-inflated eval must not raise the peak above the
        # surrounding pre-convergence grind.
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = [[80.0], [80.0], [520.0], [80.0], [80.0]]
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 100.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 5
        cb.min_evals = 3
        cb.drop_fraction = 0.5
        cb.patience = 1
        cb.num_timesteps = 1000

        assert cb._on_step() is True
        assert cb._peak_score == pytest.approx(80.0)
        assert cb._consecutive_drops == 0

    def test_early_spike_cannot_arm_a_false_collapse(self):
        # This is the false-abort shape the moving-mean peak still allowed:
        # one high eval followed by a below-gate grind. The rolling median
        # remains 80, so the detector never arms.
        trace = [520.0] + [80.0] * 30
        fired = self._drive(
            trace,
            min_evals=20,
            patience=10,
            drop_fraction=0.5,
            smoothing_window=5,
            peak_floor=100.0,
        )
        assert fired is None, f"early spike caused a false abort at step {fired}"

    def test_single_low_outlier_counts_once_then_recovery_resets(self):
        # A rolling current value would repeat the -500 outlier across five
        # overlapping windows and exhaust patience. Raw per-eval means count
        # it once; the immediately recovered eval resets the counter.
        fired = self._drive(
            [200.0] * 5 + [-500.0] + [200.0] * 5,
            min_evals=5,
            patience=3,
            drop_fraction=0.4,
            smoothing_window=5,
            peak_floor=100.0,
        )
        assert fired is None

    def test_waits_for_full_window_even_when_min_evals_is_smaller(self):
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 100.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 5
        cb.min_evals = 3
        cb.drop_fraction = 0.5
        cb.patience = 1
        cb.num_timesteps = 1000

        cb.eval_callback.evaluations_results = [[520.0], [80.0], [80.0]]
        assert cb._on_step() is True
        assert cb._peak_score == float("-inf")

        cb.eval_callback.evaluations_results.append([80.0])
        assert cb._on_step() is True
        assert cb._peak_score == float("-inf")

        cb.eval_callback.evaluations_results.append([80.0])
        assert cb._on_step() is True
        assert cb._peak_score == pytest.approx(80.0)
        assert cb._consecutive_drops == 0

    def test_peak_floor_disarms_below_gate_peaks(self):
        # A pre-convergence grind (rolling-median peak below the curriculum
        # reward gate) can never register collapse drops.
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = [[80.0], [80.0], [10.0], [10.0], [10.0]]
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 100.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 5
        cb.min_evals = 3
        cb.drop_fraction = 0.4
        cb.patience = 1
        cb.num_timesteps = 1000

        result = cb._on_step()
        assert result is True
        assert cb._consecutive_drops == 0

    def _drive(self, trace, *, min_evals, patience, drop_fraction, smoothing_window, peak_floor, eval_freq=50000):
        """Feed a per-eval mean-reward trace one eval at a time.

        Each eval is a single-episode list, so the callback's per-eval
        ``np.mean`` reproduces the trace value exactly. Returns the step at
        which the backstop stopped training, or ``None`` if it never did.
        """
        return self._drive_results(
            [[mean_reward] for mean_reward in trace],
            min_evals=min_evals,
            patience=patience,
            drop_fraction=drop_fraction,
            smoothing_window=smoothing_window,
            peak_floor=peak_floor,
            eval_freq=eval_freq,
        )

    def _drive_results(
        self,
        evaluations,
        *,
        min_evals,
        patience,
        drop_fraction,
        smoothing_window,
        peak_floor,
        eval_freq=50000,
    ):
        """Feed complete per-episode evaluation results one eval at a time."""
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = []
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb._consecutive_drops = 0
        cb.min_evals = min_evals
        cb.patience = patience
        cb.drop_fraction = drop_fraction
        cb.smoothing_window = smoothing_window
        cb.peak_floor = peak_floor
        for i in range(len(evaluations)):
            cb.eval_callback.evaluations_results = evaluations[: i + 1]
            cb.num_timesteps = (i + 1) * eval_freq
            if cb._on_step() is False:
                return cb.num_timesteps
        return None

    # Real velociraptor stage-2 per-eval mean rewards (deterministic runs).
    # Round 1's mean-std current-side rule aborted run 20260722_124556 at
    # 1.75M during a healthy bimodal gait transition (the dip at evals
    # 25-34); run 20260721_141523 is the identical trace continued, which
    # recovered to 2707 at ~1.95M. TODAY is an exact prefix of RECOVER.
    _STAGE2_TODAY_KILLED = [
        750.2,
        754.5,
        770.3,
        801.8,
        801.6,
        791.3,
        878.0,
        901.1,
        885.3,
        925.0,
        1074.6,
        1223.1,
        1199.0,
        1328.9,
        1319.0,
        1405.1,
        1074.8,
        1388.1,
        1147.9,
        1289.8,
        1334.2,
        1049.1,
        1274.0,
        1551.4,
        1385.3,
        728.7,
        387.1,
        403.7,
        557.9,
        944.2,
        1150.7,
        680.4,
        531.0,
        1101.9,
        672.0,
    ]
    _STAGE2_RECOVER = _STAGE2_TODAY_KILLED + [
        392.5,
        990.0,
        2371.3,
        2580.4,
        2618.7,
    ]

    def test_bimodal_stage2_transition_does_not_abort(self):
        # The exact trace round 1 wrongly aborted. Under the merged
        # stage-2 knobs (min_evals=20, patience=10, drop=0.5) with
        # a rolling-median peak and the gate floor, it must survive.
        fired = self._drive(
            self._STAGE2_TODAY_KILLED,
            min_evals=20,
            patience=10,
            drop_fraction=0.5,
            smoothing_window=5,
            peak_floor=100.0,
        )
        assert fired is None, f"backstop aborted a healthy transition at step {fired}"

    def test_bimodal_episode_arrays_use_healthy_mean_for_patience(self):
        import numpy as np

        # Real transition shape: half the episodes still score ~1300 while
        # half fail. This array reproduces the recorded 728.7 eval mean.
        stable_eval = [1000.0] * 30
        bimodal_eval = [1300.0] * 15 + [157.4] * 15
        peak = float(np.mean(stable_eval))
        threshold = 0.5 * peak

        assert np.mean(bimodal_eval) == pytest.approx(728.7)
        assert np.mean(bimodal_eval) > threshold
        assert np.mean(bimodal_eval) - np.std(bimodal_eval) < threshold

        fired = self._drive_results(
            [stable_eval] * 5 + [bimodal_eval] * 10,
            min_evals=5,
            patience=10,
            drop_fraction=0.5,
            smoothing_window=5,
            peak_floor=100.0,
        )
        assert fired is None

    def test_stage2_recovery_trace_reaches_high_reward_without_aborting(self):
        # Continuing the same trace recovers to ~2600; the backstop must
        # not have fired anywhere along the way.
        fired = self._drive(
            self._STAGE2_RECOVER,
            min_evals=20,
            patience=10,
            drop_fraction=0.5,
            smoothing_window=5,
            peak_floor=100.0,
        )
        assert fired is None

    def test_genuine_sustained_collapse_still_stops(self):
        # Climb to a healthy converged plateau, then a genuine sustained
        # collapse (every eval near 300). The backstop must fire.
        import numpy as np

        climb = list(np.linspace(300.0, 2700.0, 25))
        plateau = [2700.0] * 10
        collapse = [300.0] * 25
        fired = self._drive(
            climb + plateau + collapse,
            min_evals=20,
            patience=10,
            drop_fraction=0.5,
            smoothing_window=5,
            peak_floor=100.0,
        )
        assert fired is not None, "backstop failed to catch a genuine sustained collapse"


class TestCollapsePeakFloorIsDecoupledFromTheRewardGate:
    """``collapse_peak_floor`` must not inherit ``min_avg_reward``.

    The builder used to chain ``collapse_peak_floor`` -> ``min_avg_reward`` ->
    ``0.0``, coupling an early-stop backstop to an unrelated advancement
    threshold. Removing the reward gate from a stage -- which a state-capability
    gate would do -- silently dropped the floor to ``0.0``, arming collapse
    detection after any positive robust peak. See docs/STAGE1_SPLIT_PLAN.md
    section 7.4.
    """

    def test_reward_gate_no_longer_leaks_into_the_floor(self):
        settings = collapse_settings_from_config({"min_avg_reward": 1840.0})
        assert settings["peak_floor"] != 1840.0

    def test_missing_floor_never_arms_rather_than_arming_eagerly(self):
        """An unconfigured backstop must not abort a run."""
        assert collapse_settings_from_config({})["peak_floor"] == float("inf")

    def test_explicit_floor_is_honoured(self):
        settings = collapse_settings_from_config({"collapse_peak_floor": 2200.0, "min_avg_reward": 1840.0})
        assert settings["peak_floor"] == 2200.0

    def test_builder_passes_the_resolved_floor_through(self):
        """Pin the wiring too, where stable-baselines3 is available."""
        pytest.importorskip("stable_baselines3")
        cb = build_eval_collapse_early_stop_callback(
            eval_callback=None,
            curriculum_kwargs={"collapse_peak_floor": 2200.0, "min_avg_reward": 1840.0},
        )
        assert cb.peak_floor == 2200.0

    def test_every_committed_stage_config_sets_the_floor_explicitly(self):
        """Every stage must configure the floor, by either mechanism.

        Checked on the RESOLVED value rather than on the presence of
        ``collapse_peak_floor``, because a stage may now set the floor
        relatively (``collapse_peak_floor_fraction`` x
        ``collapse_peak_floor_reference``) so it re-anchors when the reward
        changes -- see PLANT_VALIDATION section 14 item 3 and the two runs
        where an absolute floor silently never armed.  A stage that configures
        neither resolves to ``inf``, i.e. never arms, which is what this test
        exists to prevent.
        """
        from environments.shared.config import load_all_stages

        for species in ("trex", "velociraptor", "brachiosaurus", "dibothrosuchus"):
            for stage, cfg in load_all_stages(species).items():
                cur = cfg.get("curriculum_kwargs", {})
                floor = collapse_settings_from_config(cur)["peak_floor"]
                assert math.isfinite(floor), (
                    f"{species} stage {stage} configures no collapse floor (neither collapse_peak_floor "
                    "nor the fraction/reference pair), so the backstop can never arm"
                )


class TestRelativeCollapsePeakFloor:
    """A relative floor re-anchors when the reward function changes.

    An absolute floor failed to arm TWICE for the same reason: 2200 against
    run ``20260731_132102`` (peak 1934.1, watched a -59% collapse), then 2450
    against run ``20260801_021545`` (best eval 2347.67, watched
    2347.67 -> 1666.33).  Deriving the number from the statue was only half a
    fix -- the statue bounds what is achievable, not what a learning policy
    passes through -- so the floor is now a fraction of a declared reference.
    """

    def test_fraction_times_reference_resolves_the_floor(self):
        settings = collapse_settings_from_config(
            {"collapse_peak_floor_fraction": 0.45, "collapse_peak_floor_reference": 3271.8}
        )
        assert settings["peak_floor"] == pytest.approx(0.45 * 3271.8)

    def test_explicit_absolute_floor_still_wins(self):
        """Existing configs keep their behaviour exactly."""
        settings = collapse_settings_from_config(
            {
                "collapse_peak_floor": 2450.0,
                "collapse_peak_floor_fraction": 0.45,
                "collapse_peak_floor_reference": 3271.8,
            }
        )
        assert settings["peak_floor"] == pytest.approx(2450.0)

    def test_incomplete_relative_pair_never_arms(self):
        """Half a declaration must not silently become a low floor."""
        for partial in ({"collapse_peak_floor_fraction": 0.45}, {"collapse_peak_floor_reference": 3271.8}):
            assert collapse_settings_from_config(partial)["peak_floor"] == float("inf")

    def test_trex_stage_one_floor_would_have_armed_on_the_failing_run(self):
        """Regression against the measured miss, not against a chosen number."""
        from environments.shared.config import load_stage_config

        floor = collapse_settings_from_config(load_stage_config("trex", 1)["curriculum_kwargs"])["peak_floor"]
        best_eval_20260801 = 2347.67
        collapse_bottom = 888.0
        assert best_eval_20260801 >= floor, "the detector must arm on a run that reached this peak"
        assert floor > collapse_bottom, "the floor must sit above the measured collapse bottom"
