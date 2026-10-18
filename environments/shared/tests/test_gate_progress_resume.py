"""A same-stage resume continues ``gate_progress.npz`` rather than truncating it.

Found by the 6M screen.  On a same-stage resume ``seed_resume_eval_state``
seeds SB3's evaluation history from ``evaluations.npz`` and
``StageAwareEvalCallback.seed_prior_history`` pads the capture series with an
empty row per seeded evaluation, so the series stay indexed by the absolute
evaluation index.  ``StageGatePlateauCallback`` started its own rows from
nothing, though, and its writer took the per-actuator rows as the first
*n* of those series for its session-local row count *n*: every resumed write
read the padding, found it empty and left out ``action_dc_per_actuator`` and
``action_ac_rms_per_actuator``, and the file it wrote held the resumed
session's evaluations only.  These tests resume a recorded run and hold its
file to the one the uninterrupted run writes.
"""

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.shared.curriculum.checkpoints import seed_resume_eval_state
from environments.shared.curriculum.hop_watch import HopRegimeWatchCallback
from environments.shared.eval_diagnostics import SagittalLegPair, StageAwareEvalCallback, StageGatePlateauCallback

sb3 = pytest.importorskip("stable_baselines3")

EPISODES = 4
STEP = 50_000
#: #620's keys: two box-soled feet, the right one with digits, and one sagittal pair.
PAD_KEYS = (
    "r_sole_cop_fore_aft",
    "r_sole_cop_fore_aft_gate",
    "r_digit_share",
    "l_sole_cop_fore_aft",
    "l_sole_cop_fore_aft_gate",
)
PAIR = SagittalLegPair("leg_dc_l_minus_r_ankle_deg", 1, 0, 1.0, 1.0)
#: Declares the pad rail's bar, so the panel's bar fraction is recorded as well.
CURRICULUM = {"min_avg_reward": 0.0, "max_sole_cop_fore_aft": 0.2}
MATRICES = ("action_dc_per_actuator", "action_ac_rms_per_actuator")


def _session(stage_dir: Path) -> tuple[StageAwareEvalCallback, StageGatePlateauCallback]:
    """The capture and plateau callbacks one training session builds, before its first evaluation."""
    eval_callback = object.__new__(StageAwareEvalCallback)
    eval_callback.n_eval_episodes = EPISODES
    eval_callback.best_mean_reward = -np.inf
    for name in (
        "evaluations_timesteps",
        "evaluations_results",
        "evaluations_length",
        "evaluations_successes",
        "evaluations_forward_velocities",
        "evaluations_unsupported_duties",
        "evaluations_bilateral_duties",
        "evaluations_action_dc",
        "evaluations_action_ac_rms",
        "evaluations_action_delta",
        "evaluations_action_jerk",
        "evaluations_reward_terms",
        "evaluations_foot_pad",
        "evaluations_leg_dc",
    ):
        setattr(eval_callback, name, [])
    # What the first evaluation resolves from the model.
    eval_callback.foot_pad_keys = PAD_KEYS
    eval_callback.leg_dc_pairs = (PAIR,)
    plateau = StageGatePlateauCallback(
        eval_callback, stage="stance", curriculum_kwargs=CURRICULUM, plateau_window=3, gate_progress_dir=stage_dir
    )
    model = MagicMock()
    model.logger = MagicMock()
    plateau.init_callback(model)
    return eval_callback, plateau


def _evaluate(eval_callback: Any, plateau: StageGatePlateauCallback, number: int, *, seed: int | None = None) -> None:
    """Evaluation *number* of the run, as SB3 and the capture callback publish it, then the plateau callback's step."""
    rng = np.random.default_rng(number if seed is None else seed)
    eval_callback.evaluations_timesteps.append(number * STEP)
    eval_callback.evaluations_results.append([float(v) for v in rng.normal(100.0, 5.0, EPISODES)])
    eval_callback.evaluations_length.append([1000] * EPISODES)
    eval_callback.evaluations_forward_velocities.append([float(v) for v in rng.normal(0.0, 0.01, EPISODES)])
    eval_callback.evaluations_unsupported_duties.append([float(v) for v in rng.uniform(0.01, 0.05, EPISODES)])
    eval_callback.evaluations_bilateral_duties.append([float(v) for v in rng.uniform(0.9, 1.0, EPISODES)])
    eval_callback.evaluations_action_dc.append([float(v) for v in rng.normal(0.0, 0.1, 3)])
    eval_callback.evaluations_action_ac_rms.append([float(v) for v in rng.uniform(0.0, 0.1, 3)])
    eval_callback.evaluations_action_delta.append(float(rng.uniform(0.5, 1.0)))
    eval_callback.evaluations_action_jerk.append(float(rng.uniform(1.0, 2.0)))
    eval_callback.evaluations_reward_terms.append({"reward_energy": float(rng.normal(-50.0, 1.0))})
    eval_callback.evaluations_foot_pad.append(
        {key: [float(v) for v in rng.uniform(-0.4, 0.4, EPISODES)] for key in PAD_KEYS}
    )
    eval_callback.evaluations_leg_dc.append({PAIR.key: float(rng.normal(0.0, 3.0))})
    plateau.num_timesteps = number * STEP
    assert plateau._on_step() is True


def _publish_evaluations(eval_callback: Any, stage_dir: Path) -> None:
    """The ``evaluations.npz`` SB3 saves (and the publisher copies into the stage directory)."""
    np.savez(
        stage_dir / "evaluations.npz",
        timesteps=np.array(eval_callback.evaluations_timesteps),
        results=np.array(eval_callback.evaluations_results),
        ep_lengths=np.array(eval_callback.evaluations_length),
    )


def _recorded(plateau: Any) -> list[tuple]:
    """The per-evaluation TensorBoard scalars, less the plateau's own (its window restarts on a resume)."""
    return [
        call.args
        for call in plateau.logger.record.call_args_list
        if not call.args[0].startswith(("diagnostics/eval_gate_", "diagnostics/eval_plateau_"))
    ]


def _first_session(stage_dir: Path, evaluations: int) -> dict[str, np.ndarray]:
    """Run *evaluations* evaluations, publish ``evaluations.npz`` and return the ``gate_progress.npz`` written."""
    eval_callback, plateau = _session(stage_dir)
    for number in range(1, evaluations + 1):
        _evaluate(eval_callback, plateau, number)
    _publish_evaluations(eval_callback, stage_dir)
    with np.load(stage_dir / "gate_progress.npz") as data:
        return dict(data)


@pytest.mark.parametrize("fresh_stage_dir", [False, True], ids=["same-stage-dir", "fresh-stage-dir"])
def test_a_resumed_run_writes_the_uninterrupted_runs_gate_progress(tmp_path, fresh_stage_dir):
    """Prior rows exactly as the first session wrote them, the resumed ones appended, for every key."""
    prior, total = 3, 5
    reference_dir = tmp_path / "uninterrupted"
    reference_dir.mkdir()
    reference_eval, reference = _session(reference_dir)
    for number in range(1, total + 1):
        if number == prior + 1:
            reference.logger.reset_mock()
        _evaluate(reference_eval, reference, number)
    with np.load(reference_dir / "gate_progress.npz") as data:
        expected = dict(data)
    assert {
        *MATRICES,
        *PAD_KEYS,
        PAIR.key,
        "max_sole_cop_fore_aft_mean",
        "max_sole_cop_fore_aft_over_bar_fraction",
    } <= set(expected)

    first_dir = tmp_path / "first"
    first_dir.mkdir()
    before = _first_session(first_dir, prior)
    # The default CLI resume mints a fresh stage directory and seeds from the interrupted one's records.
    resumed_dir = tmp_path / "resumed" if fresh_stage_dir else first_dir
    resumed_dir.mkdir(exist_ok=True)
    resumed_eval, resumed = _session(resumed_dir)
    seeded = seed_resume_eval_state(
        resumed_eval, [resumed_eval, resumed], first_dir / "evaluations.npz", max_timesteps=prior * STEP
    )
    assert seeded == prior
    for number in range(prior + 1, total + 1):
        _evaluate(resumed_eval, resumed, number)

    with np.load(resumed_dir / "gate_progress.npz") as data:
        written = dict(data)
    assert set(written) == set(expected)
    for key, values in expected.items():
        np.testing.assert_array_equal(written[key], values, err_msg=key)
        np.testing.assert_array_equal(written[key][:prior], before[key], err_msg=key)
    # Row for row with the evaluation history SB3 republishes as evaluations.npz.
    assert written["timesteps"].tolist() == [int(step) for step in resumed_eval.evaluations_timesteps]
    # The TensorBoard scalars of the resumed evaluations are the uninterrupted run's: they index the capture series
    # by the absolute evaluation index, which the padding keeps aligned.
    assert _recorded(resumed) == _recorded(reference)


def test_rows_after_the_resumed_checkpoint_are_dropped_with_the_evaluations(tmp_path):
    """A crash after the checkpoint's last evaluation: that row goes, as it goes from the evaluation history."""
    before = _first_session(tmp_path, 3)
    resumed_eval, resumed = _session(tmp_path)
    assert seed_resume_eval_state(resumed_eval, [resumed], tmp_path / "evaluations.npz", max_timesteps=2 * STEP) == 2
    _evaluate(resumed_eval, resumed, 3, seed=33)  # evaluation 3 again, from the checkpoint before it

    with np.load(tmp_path / "gate_progress.npz") as written:
        assert written["timesteps"].tolist() == [STEP, 2 * STEP, 3 * STEP]
        assert written["timesteps"].tolist() == [int(step) for step in resumed_eval.evaluations_timesteps]
        for key in before:
            np.testing.assert_array_equal(written[key][:2], before[key][:2], err_msg=key)
        assert written["action_dc_per_actuator"][2].tolist() == resumed_eval.evaluations_action_dc[2]
        assert written[PAIR.key][2] == resumed_eval.evaluations_leg_dc[2][PAIR.key]
        assert written[PAIR.key][2] != before[PAIR.key][2]


def test_a_prior_file_without_the_pad_keys_reads_nan_on_its_rows(tmp_path):
    """A run interrupted before #620 resumed after it: the new keys are kept, NaN where nothing measured them."""
    before = _first_session(tmp_path, 2)
    np.savez(
        tmp_path / "gate_progress.npz",
        **{key: values for key, values in before.items() if key in ("timesteps", "unsupported_duty", *MATRICES)},
    )
    resumed_eval, resumed = _session(tmp_path)
    assert seed_resume_eval_state(resumed_eval, [resumed], tmp_path / "evaluations.npz", max_timesteps=2 * STEP) == 2
    _evaluate(resumed_eval, resumed, 3)

    with np.load(tmp_path / "gate_progress.npz") as written:
        assert len(written["timesteps"]) == 3
        assert written["unsupported_duty"][:2].tolist() == before["unsupported_duty"].tolist()
        for key in (*PAD_KEYS, PAIR.key, "action_dc_rms"):
            assert all(math.isnan(value) for value in written[key][:2]), key
            assert math.isfinite(written[key][2]), key
        assert all(written[key].shape == (3, 3) for key in MATRICES)


def test_without_a_prior_file_the_resumed_session_still_writes_its_matrices(tmp_path):
    """Nothing to seed: the file starts over, and the per-actuator rows come from the resumed evaluations, not
    from the seeded evaluations' empty padding."""
    _first_session(tmp_path, 3)
    (tmp_path / "gate_progress.npz").unlink()
    resumed_eval, resumed = _session(tmp_path)
    assert seed_resume_eval_state(resumed_eval, [resumed], tmp_path / "evaluations.npz", max_timesteps=3 * STEP) == 3
    _evaluate(resumed_eval, resumed, 4)
    _evaluate(resumed_eval, resumed, 5)

    with np.load(tmp_path / "gate_progress.npz") as written:
        assert written["timesteps"].tolist() == [4 * STEP, 5 * STEP]
        assert written["action_dc_per_actuator"].tolist() == resumed_eval.evaluations_action_dc[3:]
        assert written["action_ac_rms_per_actuator"].tolist() == resumed_eval.evaluations_action_ac_rms[3:]


def test_the_hop_watch_judges_the_resumed_evaluations_only(tmp_path):
    """The seeded rows are the interrupted session's: its watch already judged them, like every eval cursor."""
    _first_session(tmp_path, 3)
    resumed_eval, resumed = _session(tmp_path)
    watch = object.__new__(HopRegimeWatchCallback)
    watch.plateau_callback = resumed
    watch.max_unsupported_duty = 0.0  # every evaluation measured here is above it
    watch.after_timesteps = 0
    watch.stop = True
    watch.stage_dir = None
    watch.num_timesteps = 3 * STEP
    seed_resume_eval_state(resumed_eval, [resumed, watch], tmp_path / "evaluations.npz", max_timesteps=3 * STEP)
    assert len(resumed.gate_progress_series("unsupported_duty")[0]) == 3
    assert watch._on_step() is True
    _evaluate(resumed_eval, resumed, 4)
    watch.num_timesteps = 4 * STEP
    assert watch._on_step() is False


def test_the_eval_episode_ledger_keeps_its_rows_at_or_before_the_resumed_step(tmp_path):
    """The ledger beside the record is seeded too (``seed_prior_record``), as gate_progress.npz is."""
    from environments.shared.episode_ledger import EpisodeLedger
    from environments.shared.eval_diagnostics import EVAL_LEDGER_FIXED_COLUMNS

    _first_session(tmp_path, 3)
    prior = EpisodeLedger(tmp_path / "eval_episodes.csv", fixed_columns=EVAL_LEDGER_FIXED_COLUMNS)
    prior.extend(
        {"evaluation": number - 1, "timesteps": number * STEP, "episode": episode, "return": 1.0}
        for number in range(1, 4)
        for episode in range(EPISODES)
    )
    prior.write()
    resumed_eval, resumed = _session(tmp_path)
    resumed_eval._ledger = EpisodeLedger(tmp_path / "eval_episodes.csv", fixed_columns=EVAL_LEDGER_FIXED_COLUMNS)
    seed_resume_eval_state(resumed_eval, [resumed_eval, resumed], tmp_path / "evaluations.npz", max_timesteps=2 * STEP)
    assert resumed_eval._ledger.column("timesteps") == [str(STEP)] * EPISODES + [str(2 * STEP)] * EPISODES
