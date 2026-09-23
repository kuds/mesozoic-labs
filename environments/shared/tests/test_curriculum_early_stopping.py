"""Tests for environments.shared.curriculum.early_stopping."""

import inspect
import logging
import math
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.curriculum import (
    EvalCollapseEarlyStopCallback,
    build_eval_collapse_early_stop_callback,
    early_stopping,
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
            "peak_warmup_timesteps",
        ]
        assert params["verbose"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params["smoothing_window"].kind is inspect.Parameter.KEYWORD_ONLY
        # Keyword-only for the same reason: every new knob goes after the
        # `*`, so `verbose` keeps its historical positional slot forever.
        assert params["peak_warmup_timesteps"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["peak_warmup_timesteps"].default == 0.0

    def test_returns_true_without_eval_results(self):
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock(spec=[])  # no evaluations_results
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb.peak_floor = 0.0
        cb._consecutive_drops = 0
        cb.smoothing_window = 1
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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
        cb.peak_warmup_timesteps = 0.0
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

        Every species with a stage manifest is covered (D-B14 enumerates
        compsognathus and its robot too).  Compsognathus and its robot
        declare no floor on any stage -- the plant has no learned-policy
        history to calibrate one against.  Only their stance TOMLs record
        that as a decision ("Deliberately omit a collapse floor: the shared
        detector stays unarmed until this plant has a learning history"; the
        recovery TOMLs say "Keep collapse detection unarmed"); their
        locomotion and behavior TOMLs omit the floor WITHOUT a comment, so
        for those stage rows this assertion holds by the species-level
        exemption alone (follow-up: a maintainer config decision to add the
        pair or the unarmed statement there would let the exemption narrow
        to per-stage statements).  Any other species' omission, and a
        compsognathus omission whose stance TOML no longer records the
        decision, still fails here.
        """
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import _CONFIGS_DIR, resolve_stage_key

        unarmed_by_recorded_decision = {"compsognathus", "compsognathus_robot"}
        species_ids = sorted(path.parent.name for path in _CONFIGS_DIR.glob("*/stages.toml"))
        assert {"trex", "velociraptor", "brachiosaurus", "dibothrosuchus"} | unarmed_by_recorded_decision <= set(
            species_ids
        )
        for species in species_ids:
            if species in unarmed_by_recorded_decision:
                stance_toml = _CONFIGS_DIR / species / resolve_stage_key(species, "stance").config_file
                assert "Deliberately omit a collapse floor" in stance_toml.read_text(encoding="utf-8"), (
                    f"{stance_toml} no longer records the unarmed-detector decision this exemption rests on"
                )
            for stage, cfg in load_all_stages(species).items():
                cur = cfg.get("curriculum_kwargs", {})
                floor = collapse_settings_from_config(cur)["peak_floor"]
                assert math.isfinite(floor) or species in unarmed_by_recorded_decision, (
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


class TestTighteningDropAndPatienceIsRefuted:
    """Pins the measured reason NOT to tighten the collapse thresholds.

    PLANT_VALIDATION section 14 item 3 asked for ``drop_fraction`` and
    ``patience`` to be tightened.  Simulating the detector against run
    ``20260801_021545``'s real 120-evaluation series refutes it: stage-1
    evaluation reward is so noisy that the endgame dip is milder than 53
    separate mid-training excursions, so every setting that catches the
    endgame first aborts the run long before its best model.

    These tests use the measured summary statistics rather than the raw
    series, so they document the finding without vendoring 120 evaluations
    of data into the repository.
    """

    # Measured from the run's evaluations.npz (120 evals, peak at eval 114).
    PEAK = 2347.67
    FINAL = 1630.7
    EVALS_BELOW_FINAL = 76  # of the 119 evaluations preceding the last one
    PRE_PEAK_BELOW_HALF_PEAK = 45  # of 94 pre-peak evaluations

    def test_the_endgame_dip_is_inside_the_run_s_own_noise(self):
        """The final eval beat most of the run, so there was no collapse."""
        assert self.EVALS_BELOW_FINAL / 119 > 0.5, (
            "the final evaluation was higher than most of the run; calling the "
            "best-to-final gap a collapse misreads ordinary eval variance"
        )

    def test_shipped_drop_fraction_correctly_declines_to_fire(self):
        """0.5 requires a 50% drop; the observed gap was 29%."""
        observed_drop = (self.PEAK - self.FINAL) / self.PEAK
        assert observed_drop < 0.5, f"observed drop {observed_drop:.0%} is below the 0.5 threshold"

    def test_catching_the_endgame_would_require_catching_mid_training_noise(self):
        """Any threshold catching a 29% dip also fires on the grind."""
        observed_drop = (self.PEAK - self.FINAL) / self.PEAK
        assert observed_drop < 0.5 <= 1.0, "sanity"
        assert self.PRE_PEAK_BELOW_HALF_PEAK > 0, (
            "45 pre-peak evaluations already sat below HALF the running peak, so a "
            "threshold tight enough for the endgame aborts the run mid-training"
        )


class TestPeakWarmup:
    """Evaluations before the warm-up must not set the peak.

    Regression for run `20260803_012355`: trex stage 1 peaked at 2469.4 on its
    SECOND evaluation (100k steps), because with `home-keyframe-residual/v1`
    and `log_std_init = 0` an untrained policy already commands the nominal
    stance. That armed the 0.45 x 3271.8 = 1472.3 floor on initialisation, and
    the ordinary exploration dip that follows read as a collapse -- stopping at
    1.45M of a 10M budget, after which stages 2 and 3 spent 9 1/4 hours on the
    near-statue checkpoint it left behind.

    No `peak_floor` value separates those two cases: set it above
    initialisation (>0.75x the baseline) and it sits above what a learning
    policy passes through, which is how both absolute floors in
    `collapse_settings_from_config` failed. The warm-up is a different axis.
    """

    STATUE_PEAK = 2469.4
    PEAK_FLOOR = 0.45 * 3271.8  # 1472.31, as configured for trex stage 1

    def _drive(self, trace, *, warmup, eval_freq=50_000, min_evals=20, patience=10, drop_fraction=0.5):
        """Feed a per-eval mean trace one eval at a time, tracking timesteps.

        Returns ``(stopped_at_step_or_None, callback)`` so a test can assert on
        the peak as well as the verdict.
        """
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb.eval_callback.evaluations_results = []
        cb.eval_callback.evaluations_timesteps = []
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb._consecutive_drops = 0
        cb.min_evals = min_evals
        cb.patience = patience
        cb.drop_fraction = drop_fraction
        cb.smoothing_window = 5
        cb.peak_floor = self.PEAK_FLOOR
        cb.peak_warmup_timesteps = warmup
        for i in range(len(trace)):
            cb.eval_callback.evaluations_results = [[m] for m in trace[: i + 1]]
            cb.eval_callback.evaluations_timesteps = [(j + 1) * eval_freq for j in range(i + 1)]
            cb.num_timesteps = (i + 1) * eval_freq
            if cb._on_step() is False:
                return cb.num_timesteps, cb
        return None, cb

    # The REAL per-evaluation mean rewards of run 20260803_012355 stage 1, read
    # from its evaluations.npz: 29 evaluations at 50k spacing, ending at the
    # 1.45M step where the backstop actually stopped it. Full-horizon fraction
    # runs 50.0% -> 70.0% -> 37.5% -> 7.5% -> 0.0% across the first five, so
    # the initialisation advantage is spent by 200k; the rise from evaluation
    # 14 (59.4) to 27 (350.7) is the recovery that the stop cut short.
    RUN_20260803 = [
        2007.3, 2469.4, 1506.5, 580.5, 314.6,
        323.9, 498.0, 522.1, 585.7, 539.9,
        238.1, 189.7, 112.6, 59.4, 80.8,
        111.0, 152.6, 150.6, 172.9, 226.5,
        236.2, 217.2, 270.4, 346.7, 349.5,
        379.4, 350.7, 275.1, 281.2,
    ]  # fmt: skip

    def test_without_warmup_it_reproduces_the_real_stopping_point(self):
        """Not an approximation of the failure — the failure itself."""
        stopped_at, cb = self._drive(self.RUN_20260803, warmup=0.0)
        assert stopped_at == 1_450_000
        # Armed on the first window's median by 34.2, a 2.3% margin. Every
        # other window in the run maxes out at 580.5, far under the floor.
        assert cb._peak_score == pytest.approx(1506.5)

    def test_the_configured_warmup_lets_the_real_run_continue(self):
        stopped_at, cb = self._drive(self.RUN_20260803, warmup=1_000_000)
        assert stopped_at is None
        assert cb._peak_score < self.PEAK_FLOOR  # never armed

    @pytest.mark.parametrize("warmup", [150_000, 250_000, 500_000, 1_000_000, 2_500_000])
    def test_every_warmup_past_the_initialisation_spike_works(self, warmup):
        """The spike is spent by 200k, so the choice is not knife-edge."""
        assert self._drive(self.RUN_20260803, warmup=warmup)[0] is None

    def test_a_genuine_late_collapse_still_stops(self):
        """The warm-up delays the peak; it must not disarm the backstop."""
        trace = [400.0] * 50 + [2400.0] * 10 + [300.0] * 12
        stopped_at, cb = self._drive(trace, warmup=2_500_000)
        assert stopped_at is not None
        assert cb._peak_score == pytest.approx(2400.0)

    def test_a_window_straddling_the_warmup_is_excluded_whole(self):
        """Eligibility is by window START, so no kept window spans the line.

        Judging by the window's END would keep the very first window here --
        five consecutive spikes, median 2469.4 -- and a median survives one
        contaminating sample out of five, not five.
        """
        trace = [self.STATUE_PEAK] * 5 + [400.0] * 40
        _, cb = self._drive(trace, warmup=250_000)
        assert cb._peak_score == pytest.approx(400.0)

    def test_missing_eval_timesteps_stays_disarmed_rather_than_aborting(self):
        """A backstop that cannot tell early from late must not end the run."""
        trace = self.RUN_20260803
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock(spec=["evaluations_results"])
        cb.eval_callback.evaluations_results = [[m] for m in trace]
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb._consecutive_drops = 0
        cb.min_evals = 20
        cb.patience = 10
        cb.drop_fraction = 0.5
        cb.smoothing_window = 5
        cb.peak_floor = self.PEAK_FLOOR
        cb.peak_warmup_timesteps = 1_000_000
        cb.num_timesteps = 1_450_000
        assert cb._on_step() is True

    def test_default_config_leaves_every_existing_stage_unchanged(self):
        assert collapse_settings_from_config({})["peak_warmup_timesteps"] == 0.0

    def test_config_key_is_read(self):
        settings = collapse_settings_from_config({"collapse_peak_warmup_timesteps": 1_000_000})
        assert settings["peak_warmup_timesteps"] == 1_000_000.0

    def test_the_builder_forwards_the_warmup_to_the_callback(self, monkeypatch):
        """The wiring, asserted without stable-baselines3.

        `collapse_settings_from_config` resolving the value and the callback
        honouring it were both covered; the step BETWEEN them -- the builder
        passing it through -- was asserted nowhere, and the one builder test
        that exists is `importorskip("stable_baselines3")`, so it is skipped
        in CI. A dropped kwarg here silently restores the old behaviour on
        every run while every other test stays green.
        """
        captured: dict[str, object] = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return "callback"

        monkeypatch.setattr(early_stopping, "EvalCollapseEarlyStopCallback", _capture)
        built = early_stopping.build_eval_collapse_early_stop_callback(
            eval_callback=None,
            curriculum_kwargs={
                "collapse_peak_warmup_timesteps": 1_000_000,
                "collapse_peak_floor": 1472.31,
            },
        )
        assert built == "callback"
        assert captured["peak_warmup_timesteps"] == 1_000_000.0
        assert captured["peak_floor"] == 1472.31

    def test_a_warmup_past_the_stage_budget_is_announced(self, monkeypatch, caplog):
        """Silently disabling the backstop for a whole run is the bad outcome.

        An extra zero on a step count in the millions is a plausible typo, and
        its only symptom is a run that never stops early — indistinguishable
        from a run that simply never collapsed.
        """
        monkeypatch.setattr(early_stopping, "EvalCollapseEarlyStopCallback", lambda **kwargs: "callback")
        with caplog.at_level(logging.WARNING, logger=early_stopping.logger.name):
            early_stopping.build_eval_collapse_early_stop_callback(
                eval_callback=None,
                curriculum_kwargs={
                    "collapse_peak_warmup_timesteps": 20_000_000,
                    "timesteps": 10_000_000,
                    "collapse_peak_floor": 1472.31,
                },
            )
        assert any("collapse backstop is disabled for the whole run" in record.message for record in caplog.records)

    def test_a_reachable_warmup_is_not_warned_about(self, monkeypatch, caplog):
        monkeypatch.setattr(early_stopping, "EvalCollapseEarlyStopCallback", lambda **kwargs: "callback")
        with caplog.at_level(logging.WARNING, logger=early_stopping.logger.name):
            early_stopping.build_eval_collapse_early_stop_callback(
                eval_callback=None,
                curriculum_kwargs={
                    "collapse_peak_warmup_timesteps": 1_000_000,
                    "timesteps": 10_000_000,
                    "collapse_peak_floor": 1472.31,
                },
            )
        assert not any("disabled for the whole run" in record.message for record in caplog.records)

    def test_every_committed_stage_can_still_arm_within_its_budget(self):
        """A declared warm-up must leave room for the backstop to work.

        Pairs with `test_every_committed_stage_config_sets_the_floor_explicitly`:
        that one proves the floor is configured, this one proves the warm-up
        does not push the peak past the end of the stage, which would disarm
        the backstop just as completely.
        """
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import _CONFIGS_DIR

        # Every species with a stage manifest (D-B14 enumerates compsognathus
        # and its robot too).
        species_ids = sorted(path.parent.name for path in _CONFIGS_DIR.glob("*/stages.toml"))
        assert {"trex", "velociraptor", "brachiosaurus", "dibothrosuchus", "compsognathus"} <= set(species_ids)
        for species in species_ids:
            for stage, cfg in load_all_stages(species).items():
                cur = cfg.get("curriculum_kwargs", {})
                warmup = collapse_settings_from_config(cur)["peak_warmup_timesteps"]
                if warmup <= 0.0:
                    continue
                budget = float(cur["timesteps"])
                assert warmup < budget, (
                    f"{species} stage {stage} sets collapse_peak_warmup_timesteps={warmup:.0f} "
                    f"at or beyond its timesteps={budget:.0f}, so no window can ever set the "
                    "peak and the collapse backstop can never arm"
                )
                # Not merely reachable: reachable with enough of the run left
                # for `patience` consecutive evaluations to accumulate.
                assert warmup <= 0.5 * budget, (
                    f"{species} stage {stage} spends {100 * warmup / budget:.0f}% of its budget "
                    "in collapse warm-up, leaving too little of the run protected"
                )


class TestDibothrosuchusAndBrachiosaurusWarmups:
    """The peak warm-ups on dibothrosuchus and brachiosaurus stages 1-2.

    Regression for run `20260923_020654` (dibothrosuchus, seed 42), which hit
    both shapes of the `TestPeakWarmup` failure and stopped each node at 1.45M.
    Stance: the untrained policy is the statue (2598.3), above the absolute
    0.75x-statue floor of 1950, so the backstop armed on initialisation and
    the exploration dip read as a collapse. Locomotion: EvalCallback scores the
    full forward weight from step 0, so the warm-started policy, still
    standing, evaluated at ~22x the absolute floor of 100 until the clip
    release at 800k; the lunge-and-fall that followed read as a collapse.
    Brachiosaurus has the same two shapes and no current-plant series, so
    only its configuration is pinned.
    """

    # The REAL per-evaluation mean rewards of run 20260923_020654, read from
    # its two evaluations.npz files: 29 evaluations of 30 episodes each at 50k
    # spacing, ending at the 1.45M step where the backstop stopped each node.
    # Stance: the statue plateau lasts to 450k (2514.0), then the policy falls
    # (mean episode length 116.8 at the stop). Locomotion: the standing level
    # holds to 800k (1857.5, the clip release), then 557.6 at 850k and 39.3 at
    # the stop (mean episode length 28.4).
    STANCE_20260923 = [
        2597.6, 2596.7, 2592.9, 2285.4, 2588.2,
        2581.2, 2591.0, 2516.1, 2514.0, 1655.2,
        1395.8, 1355.8, 424.1, 334.8, 663.3,
        231.5, 444.7, 618.5, 318.7, 551.0,
        113.6, 115.3, 64.3, 52.9, 59.6,
        177.1, 44.7, 134.2, 158.1,
    ]  # fmt: skip
    LOCOMOTION_20260923 = [
        2182.9, 2182.3, 2248.9, 2248.8, 2246.8,
        2246.7, 2246.9, 2214.1, 2178.4, 2148.3,
        2082.6, 2201.0, 2157.0, 2181.5, 1842.4,
        1857.5, 557.6, 192.4, 165.8, 134.6,
        150.2, 114.8, 106.8, 82.3, 78.9,
        47.9, 66.6, 48.9, 39.3,
    ]  # fmt: skip

    @staticmethod
    def _settings(species, stage, **overrides):
        from environments.shared.config import load_all_stages

        settings = collapse_settings_from_config(dict(load_all_stages(species)[stage]["curriculum_kwargs"]))
        settings.update(overrides)
        return settings

    @staticmethod
    def _drive(trace, settings, *, eval_freq=50_000):
        """`TestPeakWarmup._drive`, parameterised by resolved settings."""
        cb = object.__new__(EvalCollapseEarlyStopCallback)
        cb.eval_callback = MagicMock()
        cb._last_seen_n_evals = 0
        cb._peak_score = float("-inf")
        cb._consecutive_drops = 0
        cb.min_evals = settings["min_evals"]
        cb.patience = settings["patience"]
        cb.drop_fraction = settings["drop_fraction"]
        cb.smoothing_window = settings["smoothing_window"]
        cb.peak_floor = settings["peak_floor"]
        cb.peak_warmup_timesteps = settings["peak_warmup_timesteps"]
        for i in range(len(trace)):
            cb.eval_callback.evaluations_results = [[m] for m in trace[: i + 1]]
            cb.eval_callback.evaluations_timesteps = [(j + 1) * eval_freq for j in range(i + 1)]
            cb.num_timesteps = (i + 1) * eval_freq
            if cb._on_step() is False:
                return cb.num_timesteps, cb
        return None, cb

    @pytest.mark.parametrize(
        ("stage", "trace_name", "peak"),
        [(1, "STANCE_20260923", 2592.9), (2, "LOCOMOTION_20260923", 2246.9)],
    )
    def test_without_the_warmup_the_committed_config_reproduces_the_stop(self, stage, trace_name, peak):
        """Not an approximation of the failure — the failure itself, on both nodes."""
        settings = self._settings("dibothrosuchus", stage, peak_warmup_timesteps=0.0)
        stopped_at, cb = self._drive(getattr(self, trace_name), settings)
        assert stopped_at == 1_450_000
        assert cb._peak_score == pytest.approx(peak)

    @pytest.mark.parametrize(("stage", "trace_name"), [(1, "STANCE_20260923"), (2, "LOCOMOTION_20260923")])
    def test_the_committed_warmup_lets_the_real_run_continue(self, stage, trace_name):
        settings = self._settings("dibothrosuchus", stage)
        stopped_at, cb = self._drive(getattr(self, trace_name), settings)
        assert stopped_at is None
        # Never armed: no window past the warm-up reaches the floor.
        assert not cb._peak_score >= settings["peak_floor"]

    def test_a_shorter_locomotion_warmup_arms_on_the_fallen_plateau(self):
        """Why stage 2's warm-up spans the whole stage-entry window.

        Just past the clip release, the eligible windows are the lunge-and-fall
        regime itself (peak ~115), which clears the absolute floor of 100 and
        arms the backstop on a level that was never good.
        """
        settings = self._settings("dibothrosuchus", 2, peak_warmup_timesteps=1_000_000)
        _, cb = self._drive(self.LOCOMOTION_20260923, settings)
        assert settings["peak_floor"] <= cb._peak_score < 200.0

    def test_a_genuine_collapse_after_the_locomotion_warmup_still_stops(self):
        """The warm-up delays the peak; it must not disarm the backstop."""
        settings = self._settings("dibothrosuchus", 2)
        trace = [2245.0] * 66 + [2500.0] * 10 + [300.0] * 12
        stopped_at, cb = self._drive(trace, settings)
        assert stopped_at is not None
        assert cb._peak_score == pytest.approx(2500.0)

    @pytest.mark.parametrize("species", ["dibothrosuchus", "brachiosaurus"])
    def test_stance_warmup_is_the_trex_value(self, species):
        assert self._settings(species, 1)["peak_warmup_timesteps"] == 1_000_000

    @pytest.mark.parametrize("species", ["dibothrosuchus", "brachiosaurus"])
    def test_locomotion_warmup_covers_the_stage_entry_window(self, species):
        """The D-B5 bound trex behavior already carries, applied to locomotion."""
        from environments.shared.config import load_all_stages

        cur = load_all_stages(species)[2]["curriculum_kwargs"]
        warmup = collapse_settings_from_config(dict(cur))["peak_warmup_timesteps"]
        assert warmup >= cur["warmup_timesteps"] + cur["ramp_timesteps"]


class TestTrexBehaviorCollapseFloor:
    """The hunting stage's backstop is the measured relative pair (WS-B2; review CF3).

    The absolute ``collapse_peak_floor = 100.0`` sat 6x below the measured
    do-nothing reward (602.13 +/- 175.35, n = 40, seed 3042, 40/40 full
    horizon, physics r7), so it armed on the first qualifying evaluation of
    every run and encoded nothing.  ``collapse_settings_from_config`` lets an
    explicit absolute floor win over the pair, so the key must be ABSENT,
    not shadowed.
    """

    def _behavior_curriculum(self) -> dict:
        from environments.shared.config import load_all_stages

        return dict(load_all_stages("trex")[3]["curriculum_kwargs"])

    def test_trex_behavior_collapse_floor_is_the_pair_not_an_absolute(self):
        cur = self._behavior_curriculum()
        assert "collapse_peak_floor" not in cur
        settings = collapse_settings_from_config(cur)
        assert settings["peak_floor"] == pytest.approx(0.45 * cur["collapse_peak_floor_reference"])
        assert cur["collapse_peak_floor_reference"] == pytest.approx(602.0)
        assert settings["peak_floor"] == pytest.approx(270.9)
        # D-B5: a judgment, not a replay -- bounded by the 600k entry window
        # below and the half-budget pin above.
        assert settings["peak_warmup_timesteps"] == 1_000_000
        assert settings["peak_warmup_timesteps"] >= cur["warmup_timesteps"] + cur["ramp_timesteps"]
        # Explicit, at locomotion's tuned values rather than the 12/8/0.4 defaults.
        assert (settings["min_evals"], settings["patience"], settings["drop_fraction"]) == (20, 10, 0.5)
        # The freshness pin that lets test_statue_constant_freshness.py guard it.
        assert cur["statue_constants_physics_revision"] == 7

    def test_trex_behavior_rail_sits_below_the_statue(self):
        """D-B4: the rail is round(0.6 x reference); a statue must clear it, a collapse must not."""
        cur = self._behavior_curriculum()
        assert 0 < cur["min_avg_reward"] < cur["collapse_peak_floor_reference"]
        assert cur["min_avg_reward"] == round(0.6 * 602.13)
        assert cur["min_avg_reward"] == 361
