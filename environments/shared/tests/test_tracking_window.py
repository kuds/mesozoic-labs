"""The stride-averaged tracking window (decision D-D64), and that it is inert where it is not declared.

* **The window** (``direction_commands.TrackingWindow``): a trailing boxcar
  that expands from its first sample (never zero-padded), holds the last
  ``size`` samples, cancels a sway of its own period and is emptied by
  ``clear``.
* **The windowed metrics** (``windowed_tracking_metrics``): at a moving
  command the errors are differences of window means, and the heading error
  and the ``actual_*`` keys stay per step; at a stop it is the identity.
* **The field** (``DirectionCommandConfig.tracking_window_s``): validated,
  recorded in the task and the command manifest only off its unrecorded 0,
  inherited key by key through ``extends``, and a record written before it
  rebuilds as the per-step task.
* **The env**: per step at 0 (no window is built), a one-step window scores
  exactly as per step, a longer one matches an offline recomputation from
  the step rows, the window is emptied at every reset and at no command
  event, a stand episode is scored exactly as per step, a window that is
  not a whole number of control steps is refused, and command_mode "none"
  builds nothing.
* **The committed walks** each declare the window of D-D64's table.
* **The panel** refuses an env whose window is not its measurement's.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.command_tracking_panel import reset_command_family
from environments.shared.config import load_stage_config
from environments.shared.direction_commands import (
    TRACKING_WINDOW_CHANNELS,
    DirectionCommandConfig,
    DirectionCommandController,
    TrackingWindow,
    gaussian_tracking_reward,
    tracking_window_sample,
    tracking_window_steps,
    windowed_tracking_metrics,
)
from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids
from environments.shared.species_registry import env_class
from environments.shared.task_fingerprint import stage_task_fingerprint

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")

#: D-D64's per-model windows (seconds) and the control steps each spans.
WALK_WINDOWS: dict[str, tuple[float, int]] = {
    "tyrannosaurus_rex/anatomical": (1.12, 112),
    "allosaurus_fragilis/anatomical": (1.31, 131),
    "brachiosaurus_altithorax/anatomical": (1.40, 140),
    "dibothrosuchus_elaphros/anatomical": (0.80, 80),
    "velociraptor_mongoliensis/anatomical": (0.70, 70),
    "compsognathus_longipes/anatomical": (1.22, 61),
    "compsognathus_longipes/robot": (1.50, 75),
}

#: The keys the window replaces at a moving command; every other metric stays the step's own.
WINDOWED_KEYS = frozenset(
    {
        "tracking_error_v",
        "tracking_error_yaw",
        "tracking_error_requested_speed",
        "tracking_command_in_tolerance",
        "tracking_in_tolerance",
    }
)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def _samples(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, len(TRACKING_WINDOW_CHANNELS)))


def test_the_first_sample_comes_back_exactly_and_the_window_expands_then_trails():
    window = TrackingWindow(5)
    samples = _samples(12)
    first = window.push(samples[0])
    assert np.array_equal(first, samples[0]), "one sample's mean is the sample, bit for bit"
    for count in range(2, 13):
        means = window.push(samples[count - 1])
        held = samples[max(0, count - 5) : count]
        assert len(window) == min(count, 5)
        np.testing.assert_allclose(means, held.mean(axis=0), rtol=0, atol=1e-15)


def test_a_sway_of_the_window_period_averages_to_zero():
    steps, dt = 112, 0.01
    window = TrackingWindow(steps)
    times = np.arange(1, 4 * steps + 1) * dt
    period = steps * dt
    sway = np.zeros((times.size, len(TRACKING_WINDOW_CHANNELS)))
    sway[:, 1] = 0.2 * np.sin(2 * math.pi * times / period + 0.3)  # lateral sway, and its second harmonic
    sway[:, 1] += 0.05 * np.sin(4 * math.pi * times / period)
    sway[:, 2] = 0.37 * np.cos(2 * math.pi * times / period)  # yaw-rate sway
    sway[:, 0] = 0.6  # a steady forward speed
    for row in sway[: steps - 1]:
        window.push(row)
    for row in sway[steps - 1 :]:
        means = window.push(row)
        assert abs(means[1]) < 1e-12 and abs(means[2]) < 1e-12
        assert means[0] == pytest.approx(0.6, abs=1e-12)


def test_clear_empties_the_window():
    window = TrackingWindow(3)
    samples = _samples(5, seed=1)
    for row in samples[:4]:
        window.push(row)
    window.clear()
    assert len(window) == 0
    assert np.array_equal(window.push(samples[4]), samples[4])


@pytest.mark.parametrize("size", [0, -1, 1.5, True, None])
def test_a_window_holds_a_whole_positive_number_of_samples(size):
    with pytest.raises(ValueError, match="whole number of samples"):
        TrackingWindow(size)


@pytest.mark.parametrize("sample", [np.zeros(6), np.full(7, np.nan), np.zeros((1, 7))])
def test_a_sample_is_seven_finite_values(sample):
    with pytest.raises(ValueError, match="7 finite values"):
        TrackingWindow(4).push(sample)


def test_the_window_spans_a_whole_number_of_control_steps():
    assert tracking_window_steps(0.0, 0.01) == 0
    assert tracking_window_steps(1.12, 0.01) == 112
    assert tracking_window_steps(1.22, 0.02) == 61
    assert tracking_window_steps(0.01, 0.01) == 1
    for window_s, dt in ((1.125, 0.01), (1.23, 0.02), (0.004, 0.01)):
        with pytest.raises(ValueError, match=r"whole number of control steps of dt = "):
            tracking_window_steps(window_s, dt)


# ---------------------------------------------------------------------------
# The windowed metrics
# ---------------------------------------------------------------------------


def _metrics(active: bool, **overrides: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "actual_v_x": 0.5,
        "actual_v_y": 0.3,
        "actual_speed": math.hypot(0.5, 0.3),
        "actual_heading": 0.1,
        "actual_yaw_rate": 0.4,
        "tracking_error_v": 0.3,
        "tracking_error_yaw": 0.4,
        "tracking_error_heading": 0.05,
        "tracking_heading_active": active,
        "tracking_error_requested_speed": 0.1,
        "tracking_command_in_tolerance": False,
        "tracking_in_tolerance": False,
    }
    metrics.update(overrides)
    return metrics


def test_a_moving_command_is_scored_on_differences_of_window_means():
    # Window means: v (0.62, 0.03), yaw rate 0.02; command (0.6, 0, 0.01); requested speed 0.6.
    means = np.array([0.62, 0.03, 0.02, 0.6, 0.0, 0.01, 0.6])
    windowed = windowed_tracking_metrics(_metrics(True), means)
    assert windowed["tracking_error_v"] == pytest.approx(math.hypot(0.02, 0.03))
    assert windowed["tracking_error_yaw"] == pytest.approx(0.01)
    assert windowed["tracking_error_requested_speed"] == pytest.approx(math.hypot(0.62, 0.03) - 0.6)
    assert windowed["tracking_command_in_tolerance"] is True and windowed["tracking_in_tolerance"] is True
    per_step = _metrics(True)
    assert {key: value for key, value in windowed.items() if key not in WINDOWED_KEYS} == {
        key: value for key, value in per_step.items() if key not in WINDOWED_KEYS
    }, "the actual_* keys and the heading error stay the step's own"
    # The heading error is judged per step: off heading, in tolerance on the window's velocity alone.
    turned = windowed_tracking_metrics(_metrics(True, tracking_error_heading=0.3), means)
    assert turned["tracking_command_in_tolerance"] is True and turned["tracking_in_tolerance"] is False
    tight = windowed_tracking_metrics(_metrics(True), means, velocity_tolerance=0.01)
    assert tight["tracking_command_in_tolerance"] is False and tight["tracking_in_tolerance"] is False


def test_a_stop_is_scored_per_step():
    stop = _metrics(False, tracking_in_tolerance=True)
    assert windowed_tracking_metrics(stop, np.zeros(7)) == stop


def test_tolerances_and_means_are_checked():
    with pytest.raises(ValueError, match="velocity_tolerance"):
        windowed_tracking_metrics(_metrics(True), np.zeros(7), velocity_tolerance=0.0)
    with pytest.raises(ValueError, match="7 finite values"):
        windowed_tracking_metrics(_metrics(True), np.zeros(5))


# ---------------------------------------------------------------------------
# The field: validation, the record, the fingerprint and extends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf, True, "1.0x", None])
def test_the_field_is_finite_and_at_least_zero(value):
    with pytest.raises(ValueError, match=r"tracking_window_s must be finite and at least 0 \(0 scores every step"):
        DirectionCommandConfig(tracking_window_s=value)
    assert DirectionCommandConfig(tracking_window_s=1).tracking_window_s == 1.0


def test_the_field_is_last_and_recorded_only_off_zero():
    names = [spec.name for spec in fields(DirectionCommandConfig)]
    assert names[-1] == "tracking_window_s"
    assert DirectionCommandConfig.UNRECORDED_VALUES["tracking_window_s"] == 0.0
    default = DirectionCommandConfig()
    assert "tracking_window_s" not in default.task_record()
    windowed = replace(default, tracking_window_s=1.12)
    assert windowed.task_record() == {**default.task_record(), "tracking_window_s": 1.12}
    assert DirectionCommandController(windowed).manifest()["config"]["tracking_window_s"] == 1.12
    assert "tracking_window_s" not in DirectionCommandController(default).manifest()["config"]
    # The cruise ramp's configs keep it; set_command_config refuses any change but the speeds.
    assert windowed.ramped(0.25).tracking_window_s == 1.12


def test_the_record_round_trips_and_a_record_from_before_it_rebuilds_per_step():
    windowed = DirectionCommandConfig(cruise_speed=0.5, speed_range=(0.25, 0.5), tracking_window_s=1.4)
    record = windowed.field_record()
    assert record["tracking_window_s"] == 1.4
    assert DirectionCommandConfig(**record) == windowed
    before = {key: value for key, value in record.items() if key != "tracking_window_s"}
    rebuilt = DirectionCommandConfig(**before)
    assert rebuilt.tracking_window_s == 0.0 and rebuilt == replace(windowed, tracking_window_s=0.0)


def test_the_walk_task_records_its_window_and_the_per_step_task_does_not():
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    command = config["env_kwargs"]["command_config"]
    assert command.tracking_window_s == 1.12
    fingerprint = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", stage_config=config)
    assert fingerprint["command"]["config"]["tracking_window_s"] == 1.12
    assert fingerprint["env"]["command_config"]["tracking_window_s"] == 1.12
    per_step = {
        **config,
        "env_kwargs": {**config["env_kwargs"], "command_config": replace(command, tracking_window_s=0.0)},
    }
    before = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", stage_config=per_step)
    assert "tracking_window_s" not in before["command"]["config"]
    assert "tracking_window_s" not in before["env"]["command_config"]
    assert before["task_sha256"] != fingerprint["task_sha256"]


def test_extends_inherits_the_window_unless_the_child_restates_it(tmp_path: Path):
    import shutil

    from environments.shared.paths import REPOSITORY_ROOT
    from environments.shared.stage_manifest import load_stage_manifest

    directory = tmp_path / TYRANNOSAURUS_REX.species / TYRANNOSAURUS_REX.embodiment
    shutil.copytree(TYRANNOSAURUS_REX.path(REPOSITORY_ROOT / "configs"), directory)
    child = directory / load_stage_manifest(TYRANNOSAURUS_REX).by_id("recovery").config_file
    text = child.read_text()
    assert 'extends = { stage = "stance", tables = ["stage", "env", "ppo"] }' in text
    child.write_text(
        text.replace(
            'extends = { stage = "stance", tables = ["stage", "env", "ppo"] }',
            'extends = { stage = "locomotion", tables = ["commands"] }',
        )
        + "\n[commands]\ncruise_ramp_start_fraction = 1.0\ncruise_ramp_timesteps = 0\n"
    )

    def window() -> float:
        loaded = load_stage_config(TYRANNOSAURUS_REX, "recovery", config_path=str(child))
        value: float = loaded["env_kwargs"]["command_config"].tracking_window_s
        return value

    assert window() == 1.12, "a node that continues the walk inherits its window"
    child.write_text(
        child.read_text().replace("cruise_ramp_timesteps = 0\n", "cruise_ramp_timesteps = 0\ntracking_window_s = 0.5\n")
    )
    assert window() == 0.5


@pytest.mark.parametrize("key", sorted(WALK_WINDOWS))
def test_every_committed_walk_declares_its_window(key):
    species, embodiment = key.split("/")
    model_id = ModelId(species, embodiment)
    window_s, steps = WALK_WINDOWS[key]
    config = load_stage_config(model_id, "locomotion")
    command = config["env_kwargs"]["command_config"]
    assert command.tracking_window_s == window_s
    dt = float(config["env_kwargs"].get("timestep", 0.002)) * int(config["env_kwargs"].get("frame_skip", 5))
    assert tracking_window_steps(command.tracking_window_s, dt) == steps


def test_the_windows_cover_every_walk():
    assert set(WALK_WINDOWS) == {
        f"{model_id.species}/{model_id.embodiment}" for model_id in model_ids(backend="stable-baselines3")
    }


# ---------------------------------------------------------------------------
# The env
# ---------------------------------------------------------------------------


def _walk_env(window_s: float | None = None) -> Any:
    """The committed T. rex walk's env, with its window replaced by *window_s* when given."""
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    kwargs = dict(config["env_kwargs"])
    if window_s is not None:
        kwargs["command_config"] = replace(kwargs["command_config"], tracking_window_s=window_s)
    return env_class(TYRANNOSAURUS_REX)(**kwargs)


def _seed(family: str) -> int:
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")["env_kwargs"]["command_config"]
    return next(seed for seed in range(3042, 3542) if reset_command_family(config, seed) == family)


def _roll(env: Any, seed: int, steps: int) -> list[tuple[float, dict[str, Any]]]:
    """*steps* zero-action steps from ``reset(seed=seed)``: each step's reward and info."""
    env.reset(seed=seed)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    rolled = []
    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(zero)
        rolled.append((float(reward), dict(info)))
        if terminated or truncated:
            break
    return rolled


def _same(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether two infos hold the same keys and values, NaN equal to NaN."""
    return repr(sorted(left.items())) == repr(sorted(right.items()))


@pytest.fixture(scope="module")
def envs():
    """The per-step walk (window 0), the committed one (1.12 s) and a one-step window (0.01 s)."""
    built = {"per_step": _walk_env(0.0), "walk": _walk_env(), "one_step": _walk_env(0.01)}
    yield built
    for env in built.values():
        env.close()


def test_a_per_step_config_builds_no_window_and_scores_as_tracking_metrics(envs):
    env = envs["per_step"]
    assert env._tracking_window is None
    sigmas = (env.tracking_velocity_sigma, env.tracking_yaw_rate_sigma)
    for reward, info in _roll(env, _seed("cruise"), 30):
        error_v = math.hypot(info["actual_v_x"] - info["command_v_x"], info["actual_v_y"] - info["command_v_y"])
        assert info["tracking_error_v"] == pytest.approx(error_v, rel=0, abs=1e-12)
        assert info["tracking_error_yaw"] == pytest.approx(abs(info["actual_yaw_rate"] - info["command_yaw_rate"]))
        expected = env.tracking_weight * gaussian_tracking_reward(
            info["tracking_error_v"], info["tracking_error_yaw"], velocity_sigma=sigmas[0], yaw_rate_sigma=sigmas[1]
        )
        assert info["reward_tracking"] == expected


def test_a_one_step_window_scores_exactly_as_per_step(envs):
    assert envs["one_step"]._tracking_window.size == 1
    seed = _seed("cruise")
    for (reward, info), (reward_1, info_1) in zip(_roll(envs["per_step"], seed, 60), _roll(envs["one_step"], seed, 60)):
        assert reward == reward_1 and _same(info, info_1)


def test_the_walk_window_matches_an_offline_recomputation_and_keeps_the_step_values(envs):
    env, size = envs["walk"], 112
    assert env._tracking_window.size == size
    seed = _seed("cruise")
    per_step = _roll(envs["per_step"], seed, 200)
    windowed = _roll(env, seed, 200)
    assert len(per_step) == len(windowed) == 200, "the statue stands 2 s; the reward never moves the physics"
    tolerances = env.tracking_tolerances
    samples = []
    for step, ((_, step_info), (_, info)) in enumerate(zip(per_step, windowed)):
        # Every key but the windowed ones and what they feed (the tracking term, the totals, the dwell) is the step's.
        fed = WINDOWED_KEYS | {"reward_tracking", "reward_total", "tracking_dwell_s"}
        assert _same(
            {k: v for k, v in info.items() if k not in fed}, {k: v for k, v in step_info.items() if k not in fed}
        )
        samples.append(tracking_window_sample(step_info))
        means = np.mean(samples[max(0, step + 1 - size) :], axis=0)
        offline = windowed_tracking_metrics(step_info, means, **tolerances)
        for key in ("tracking_error_v", "tracking_error_yaw", "tracking_error_requested_speed"):
            assert info[key] == pytest.approx(offline[key], rel=0, abs=1e-9), (step, key)
        assert info["tracking_in_tolerance"] == offline["tracking_in_tolerance"]
        expected = env.tracking_weight * gaussian_tracking_reward(
            info["tracking_error_v"],
            info["tracking_error_yaw"],
            velocity_sigma=env.tracking_velocity_sigma,
            yaw_rate_sigma=env.tracking_yaw_rate_sigma,
        )
        assert info["reward_tracking"] == pytest.approx(expected, rel=0, abs=1e-12)
    # The first step's window holds one sample: the per-step values exactly.
    assert _same(windowed[0][1], per_step[0][1])
    assert any(
        info["tracking_error_v"] != step_info["tracking_error_v"]
        for (_, step_info), (_, info) in zip(per_step, windowed)
    )


def test_the_window_is_emptied_at_every_reset_and_at_no_command_event(envs):
    env = envs["walk"]
    cruise = _seed("cruise")
    _roll(env, cruise, 150)
    assert len(env._tracking_window) == 112
    # The next episode's first step is scored on its own sample (an unseeded reset, then a seeded one).
    env.reset()
    assert len(env._tracking_window) == 0
    _roll(env, cruise + 1, 5)
    env.reset(seed=cruise + 1)
    assert len(env._tracking_window) == 0
    first_windowed = _roll(env, cruise, 1)[0]
    first_per_step = _roll(envs["per_step"], cruise, 1)[0]
    assert first_windowed[0] == first_per_step[0] and _same(first_windowed[1], first_per_step[1])
    # set_direction is a command event, not a reset: the window keeps filling.
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(20):
        env.step(zero)
    held = len(env._tracking_window)
    env.set_direction(env._heading() + 0.4, 0.6)
    assert len(env._tracking_window) == held
    env.step(zero)
    assert len(env._tracking_window) == held + 1


def test_a_stand_episode_is_scored_exactly_as_per_step(envs):
    stand = _seed("stand")
    per_step = _roll(envs["per_step"], stand, 150)
    windowed = _roll(envs["walk"], stand, 150)
    assert len(per_step) == len(windowed) == 150
    assert all(info["desired_speed"] == 0.0 for _, info in windowed)
    for (reward, info), (reward_w, info_w) in zip(per_step, windowed):
        assert reward == reward_w and _same(info, info_w)
    assert len(envs["walk"]._tracking_window) == 112, "a stop's samples enter the window too"


def test_a_window_that_is_not_a_whole_number_of_control_steps_is_refused():
    with pytest.raises(ValueError, match=r"tracking_window_s = 1\.125 s is not a whole number of control steps of dt"):
        _walk_env(1.125)


def test_command_mode_none_builds_no_window():
    config = load_stage_config(TYRANNOSAURUS_REX, "stance")
    assert config["env_kwargs"].get("command_mode", "none") == "none"
    env = env_class(TYRANNOSAURUS_REX)(**config["env_kwargs"])
    try:
        assert env.direction_controller is None and env._tracking_window is None
        env.reset(seed=3042)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not any(key.startswith("tracking_") for key in info)
    finally:
        env.close()


def test_the_panel_refuses_an_env_whose_window_is_not_the_measurements(monkeypatch: pytest.MonkeyPatch):
    """The gate must read the signal the env's metrics read: a built env on another window is refused before it rolls."""
    import environments.shared.species_registry as species_registry
    from environments.shared.command_tracking_panel import CommandPanelError, roll_command_panel

    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    original = species_registry.env_class

    def per_step_env(model_id: ModelId) -> Any:
        def build(**kwargs: Any) -> Any:
            command = replace(kwargs["command_config"], tracking_window_s=0.0)
            return original(model_id)(**{**kwargs, "command_config": command})

        return build

    monkeypatch.setattr(species_registry, "env_class", per_step_env)
    zero = np.zeros(1, dtype=np.float32)
    with pytest.raises(CommandPanelError, match=r"a 0 s tracking window and the panel's measurement records 1\.12 s"):
        roll_command_panel(TYRANNOSAURUS_REX, config, predict=lambda _obs: zero)
