"""Independent event windows, stop masking and VecEnv auto-reset evidence."""

from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from environments.shared.behavior_evaluation import evaluate_behavior, summarize_episode
from environments.shared.direction_commands import DirectionCommandConfig, DirectionCommandController


def _rows(n=40, dt=0.1, *, good=True, speed=1.05, desired=0.0, actual=0.0):
    return [
        {
            "time_s": (index + 1) * dt,
            "command_event_id": 0,
            "command_event_start_s": 0.0,
            "desired_heading": desired,
            "actual_heading": actual,
            "desired_speed": speed,
            "actual_speed": speed if good else 0.4,
            "actual_yaw_rate": 0.0,
            "tracking_error_v": 0.0 if good else 0.65,
            "tracking_error_yaw": 0.0,
            "tracking_in_tolerance": good,
            "course_progress_m": index * 0.1,
            "course_reached": False,
            "reward": 1.0,
        }
        for index in range(n)
    ]


def test_correct_stable_command_acquires_and_dwells_within_complete_window():
    rows = _rows()
    summary = summarize_episode(rows, 0.1, horizon_steps=40, initial_heading=0.0)
    assert summary["full_horizon"]
    assert summary["settled_event_count"] == 1
    event = summary["events"][0]
    assert event["eligible"] and event["settled"]
    assert event["acquisition_time_s"] == 0.0
    assert event["dwell_complete_time_s"] == pytest.approx(1.0)
    assert summary["active_heading_mae_deg_after_1s"] == 0.0
    assert summary["mean_actual_speed"] == pytest.approx(1.05)
    assert "success" not in summary and "certified" not in summary


def test_radial_progress_does_not_make_wrong_direction_a_success():
    rows = _rows(good=False, desired=0.3)
    rows[-1]["course_reached"] = True
    summary = summarize_episode(rows, 0.1, horizon_steps=40, initial_heading=0.0)
    assert summary["course_reached_radial_diagnostic"]
    assert summary["full_horizon"]
    assert summary["settled_event_count"] == 0
    assert summary["active_heading_mae_deg_after_1s"] == pytest.approx(math.degrees(0.3))


def test_dwell_must_be_contiguous_and_start_before_the_acquisition_deadline():
    rows = _rows()
    for index, row in enumerate(rows):
        row["tracking_in_tolerance"] = index % 10 != 9
    summary = summarize_episode(rows, 0.1, horizon_steps=40, initial_heading=0.0)
    assert summary["settled_event_count"] == 0
    for index, row in enumerate(rows):
        row["tracking_in_tolerance"] = index >= 12
    late = summarize_episode(rows, 0.1, horizon_steps=40, initial_heading=0.0)
    assert late["settled_event_count"] == 0
    assert late["events"][0]["status"] == "not_settled"


def test_short_final_event_cannot_pass_even_if_it_already_looks_good():
    rows = _rows(n=15)
    summary = summarize_episode(rows, 0.1, horizon_steps=15)
    event = summary["events"][0]
    assert not event["eligible"] and not event["settled"]
    assert event["status"] == "window_too_short"
    assert event["acquisition_time_s"] is None
    assert summary["eligible_event_settle_fraction"] is None


def test_stop_heading_is_masked_and_drift_or_spin_prevents_settling():
    rows = _rows(speed=0.0, desired=math.pi, actual=0.0)
    summary = summarize_episode(rows, 0.1, horizon_steps=40, initial_heading=0.0)
    assert summary["settled_stop_event_count"] == 1
    assert summary["active_heading_mae_deg_after_1s"] is None
    assert summary["active_heading_samples_after_1s"] == 0
    assert summary["stop_mean_speed"] == 0.0
    assert summary["events"][0]["initial_heading_error_rad"] == 0.0
    for row in rows:
        row.update(actual_speed=0.2, tracking_error_v=0.2, tracking_in_tolerance=False)
    drifting = summarize_episode(rows, 0.1, horizon_steps=40)
    assert drifting["settled_stop_event_count"] == 0
    assert drifting["stop_mean_speed"] == pytest.approx(0.2)


def test_fall_preserves_planned_unobserved_events_as_failures():
    rows = _rows(n=5)
    rows[-1].update(terminated=True, termination_reason="fallen")
    planned = [
        {"event_id": 0, "start_time_s": 0.0, "desired_heading": 0.0, "desired_speed": 1.05},
        {"event_id": 1, "start_time_s": 4.0, "desired_heading": 0.3, "desired_speed": 1.05},
    ]
    summary = summarize_episode(rows, 0.1, planned_events=planned, horizon_steps=80, initial_heading=0.0)
    assert summary["fall"] and summary["early_termination"]
    assert not summary["full_horizon"]
    assert summary["event_count"] == 2
    assert summary["eligible_event_count"] == 2
    assert summary["settled_event_count"] == 0
    assert summary["unobserved_event_count"] == 1
    assert summary["events"][0]["status"] == "observation_incomplete"
    assert summary["events"][1]["status"] == "not_observed"
    assert summary["events"][1]["eligibility_basis"] == "nominal_request_change_for_unobserved_event"


def test_heading_exclusion_uses_measured_event_error_and_control_seconds():
    rows = _rows(n=60, desired=0.3, actual=0.0)
    rows[0]["event_initial_heading_error_rad"] = 0.3
    for row in rows:
        if row["time_s"] >= 2.0:
            row["actual_heading"] = 0.3
    summary = summarize_episode(rows, 0.1, horizon_steps=60, yaw_rate_max=0.3)
    assert summary["events"][0]["settling_allowance_s"] == pytest.approx(2.0)
    assert summary["active_heading_mae_deg_after_1s"] > 0.0
    assert summary["active_heading_mae_deg_after_settle"] == 0.0


def test_missing_samples_do_not_create_fake_continuous_dwell_and_nan_cannot_pass():
    rows = _rows(n=12)
    for row in rows[6:]:
        row["time_s"] += 0.2
    summary = summarize_episode(rows, 0.1, horizon_steps=40)
    assert summary["settled_event_count"] == 0
    rows = _rows()
    for row in rows:
        row["actual_speed"] = math.nan
    summary = summarize_episode(rows, 0.1, horizon_steps=40)
    assert summary["settled_event_count"] == 0
    assert summary["mean_actual_speed"] is None


class _FakeVecNormalize:
    """Small VecEnv protocol double that deliberately auto-resets on done."""

    num_envs = 1

    def __init__(self, *, fall_step=None):
        self.training = True
        self.norm_reward = True
        self.raw = SimpleNamespace(dt=0.1, max_episode_steps=80)
        self.raw.direction_controller = DirectionCommandController(DirectionCommandConfig(straight_probability=1.0))
        self.raw.unwrapped = self.raw
        self.venv = SimpleNamespace(reset_infos=[{}])
        self._seed = 0
        self._step = 0
        self.fall_step = fall_step
        self.seeds = []
        self.reset()

    def get_attr(self, name, indices=0):
        assert name == "unwrapped" and indices == 0
        return [self.raw]

    def seed(self, seed):
        self._seed = seed
        self.seeds.append(seed)

    def reset(self):
        self._step = 0
        self.raw.direction_controller.reset(np.random.default_rng(self._seed), 0.0)
        self.venv.reset_infos = [{"terrain": {"family": "test_surface", "seed": self._seed}}]
        return np.zeros((1, 3))

    def step(self, action):
        assert not self.training and not self.norm_reward
        state = self.raw.direction_controller.update(self._step * self.raw.dt, 0.0)
        self._step += 1
        info = {
            **_rows(n=1)[0],
            **state.as_info(),
            "actual_speed": 1.05,
            "tracking_error_v": 0.0,
            "tracking_error_yaw": 0.0,
            "tracking_in_tolerance": True,
        }
        fell = self._step == self.fall_step
        done = fell or self._step >= self.raw.max_episode_steps
        info["TimeLimit.truncated"] = done and not fell
        if fell:
            info["termination_reason"] = "fallen"
        if done:
            self._seed += 1000
            self.reset()  # Overwrites live controller and terrain metadata.
        return np.zeros((1, 3)), np.array([1.0]), np.array([done]), [info]


class _Model:
    def predict(self, observation, deterministic=True):
        assert deterministic
        return np.zeros((1, 1)), None


def test_evaluation_writes_replayable_seeded_traces_and_restores_flags(tmp_path):
    vec = _FakeVecNormalize()
    report = evaluate_behavior(_Model(), vec, episode_seeds=[11, 22], output_dir=tmp_path)
    assert report["episode_count"] == 2
    assert report["full_horizon_fraction"] == 1.0
    assert report["fall_count"] == 0
    assert vec.training and vec.norm_reward
    assert vec.seeds == [11, 22]
    reset = json.loads((tmp_path / "episodes/episode_000_seed_11_reset.json").read_text())
    assert reset["terrain"]["seed"] == 11  # Not the auto-reset's 1011.
    assert len(reset["planned_events"]) == 2  # The event at exactly horizon is excluded.
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert "no acceptance or certification" in summary["purpose"]
    with (tmp_path / "episodes/episode_000_seed_11_steps.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 80
    assert float(rows[-1]["time_s"]) == 8.0
    assert rows[-1]["truncated"] == "True"
    assert report["episodes"][0]["events"][1]["settled"]


def test_evaluation_fall_does_not_lose_scheduled_events_to_auto_reset(tmp_path):
    vec = _FakeVecNormalize(fall_step=5)
    report = evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path)
    assert report["fall_count"] == 1
    assert report["full_horizon_fraction"] == 0.0
    assert report["unobserved_event_count"] == 1
    assert report["eligible_event_count"] == 2
    assert report["settled_event_count"] == 0


def test_normalization_flags_are_restored_when_inference_raises(tmp_path):
    class BrokenModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("inference failed")

    vec = _FakeVecNormalize()
    vec.norm_reward = False
    with pytest.raises(RuntimeError, match="inference failed"):
        evaluate_behavior(BrokenModel(), vec, episode_seeds=[42], output_dir=tmp_path)
    assert vec.training and not vec.norm_reward


@pytest.mark.parametrize(
    "artifact", ["episodes", "replays", "evaluation_summary.json", "episodes.csv", "command_events.csv"]
)
def test_evaluation_refuses_to_mix_existing_evidence_with_a_new_run(tmp_path, artifact):
    destination = tmp_path / artifact
    if artifact in ("episodes", "replays"):
        destination.mkdir()
        destination /= "existing.json"
    destination.write_text("previous evidence\n")
    vec = _FakeVecNormalize()
    with pytest.raises(FileExistsError, match="existing evaluation artifacts"):
        evaluate_behavior(_Model(), vec, episode_seeds=[42], output_dir=tmp_path)
    assert destination.read_text() == "previous evidence\n"
    assert vec.training and vec.norm_reward


def test_evaluation_accepts_existing_training_and_checkpoint_context(tmp_path):
    (tmp_path / "model.zip").write_bytes(b"checkpoint")
    (tmp_path / "run_manifest.json").write_text("{}\n")
    report = evaluate_behavior(_Model(), _FakeVecNormalize(), episode_seeds=[42], output_dir=tmp_path)
    assert report["episode_count"] == 1
    assert (tmp_path / "model.zip").read_bytes() == b"checkpoint"


@pytest.mark.parametrize("seeds", [[], [-1], [1, 1], [True], [0.5]])
def test_evaluation_refuses_invalid_or_duplicate_seed_panels(tmp_path, seeds):
    with pytest.raises(ValueError, match="episode_seeds"):
        evaluate_behavior(_Model(), _FakeVecNormalize(), episode_seeds=seeds, output_dir=tmp_path)
