"""The per-step episode trace (``reporting/episode_trace.py``, ``scripts/episode_trace_report.py``): report-only.

The property that matters most is that the default trace IS the stance
judge's rollout: the same command at every step, the same rewards and
lengths, for the v1 and v2 panels and the heading certificate's 0 degree
cell.  The rest checks what the columns and the summary mean, the two
probes (the stochastic draw and the hold), and the refusals.  The rollouts
use the tiny real T. rex checkpoint of :mod:`.episode_trace_helpers`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from .episode_trace_helpers import (
    BIASED_ACTUATOR,
    HORIZON,
    SETTLE_STEPS,
    TYRANNOSAURUS_REX,
    tiny_checkpoint,
)

pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")

from environments.shared.gait.constants import SATURATION_ABS  # noqa: E402
from environments.shared.reporting import episode_trace  # noqa: E402
from environments.shared.reporting.episode_trace import (  # noqa: E402
    TRACE_SCHEMA,
    TRACE_SUMMARY_JSON,
    ActuatorHold,
    EpisodeTraceError,
    build_episode_trace,
    refuse_run_directory_output,
    render_episode_trace,
    write_episode_trace,
)

SEEDS = (3042, 3043)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp("episode-trace"))


@pytest.fixture(scope="module")
def deterministic(checkpoint):
    return build_episode_trace(
        TYRANNOSAURUS_REX,
        1,
        stage_config=checkpoint.stage_config,
        model_path=str(checkpoint.model),
        vecnorm_path=str(checkpoint.vecnorm),
        seeds=SEEDS,
    )


def _judge_commands(checkpoint, **panel_kwargs):
    """The stance report's ``run_panel`` on the same checkpoint and seeds, with every command it sent recorded."""
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.reporting import stance_report
    from environments.shared.species_registry import env_class

    env_kwargs = checkpoint.stage_config["env_kwargs"]
    cls = env_class(TYRANNOSAURUS_REX)
    identity = current_plant_identity(TYRANNOSAURUS_REX)
    predict, _note = stance_report._load_policy(
        str(checkpoint.model), str(checkpoint.vecnorm), lambda: cls(**env_kwargs), plant_identity=identity
    )
    sent: list[np.ndarray] = []

    def recording(obs):
        action = predict(obs)
        sent.append(np.asarray(action, dtype=np.float64).ravel().copy())
        return action

    result = stance_report.run_panel(
        TYRANNOSAURUS_REX,
        predict=recording,
        episodes=len(SEEDS),
        seed=SEEDS[0],
        settle_steps=SETTLE_STEPS,
        horizon=HORIZON,
        env_kwargs=env_kwargs,
        plant_identity=identity,
        **panel_kwargs,
    )
    return result, np.stack(sent)


@pytest.mark.parametrize("panel", ["v1", "heading_zero"])
def test_the_deterministic_trace_is_the_judges_rollout_step_for_step(checkpoint, deterministic, panel):
    from environments.shared.reporting.stance_report import SpawnYaw

    kwargs = {} if panel == "v1" else {"floor_truth": True, "spawn_yaw": SpawnYaw(0.0, whole_scene=True)}
    result, sent = _judge_commands(checkpoint, **kwargs)
    report, traces = deterministic
    traced = np.concatenate([trace.values["command"] for trace in traces])
    np.testing.assert_array_equal(traced, sent)
    assert [trace.length for trace in traces] == [int(length) for length in result["lengths"]]
    assert [trace.episode_return for trace in traces] == [float(reward) for reward in result["rewards"]]
    assert [episode["return"] for episode in report["episodes"]] == [float(reward) for reward in result["rewards"]]
    assert report["seeds"] == list(SEEDS) and report["deterministic"] is True and report["holds"] == []


def test_the_columns_are_the_raw_mean_the_clipped_command_and_the_plants_response(checkpoint, deterministic):
    from environments.shared.gait.balance_telemetry import actuator_force_limits
    from environments.shared.species_registry import env_class

    report, traces = deterministic
    names = report["actuator_names"]
    tail = names.index(BIASED_ACTUATOR)
    env = env_class(TYRANNOSAURUS_REX)(**checkpoint.stage_config["env_kwargs"])
    try:
        plant = env.unwrapped.model
        scaled_one = float(env.unwrapped._scale_action(np.ones(plant.nu))[tail])
        limits = actuator_force_limits(plant)
    finally:
        env.close()
    for trace in traces:
        values = trace.values
        # predict(deterministic=True) is the Gaussian mean clipped to the bound, bit for bit.
        np.testing.assert_array_equal(values["command"], np.clip(values["raw_mean"], -1.0, 1.0))
        assert np.all(values["raw_mean"][:, tail] > 1.0) and np.all(values["command"][:, tail] == 1.0)
        assert np.all(values["ctrl"][:, tail] == scaled_one)
        assert np.all(np.isnan(values["sample"]))
        force = values["force"][:, tail]
        side = np.where(force >= 0.0, limits[tail, 1], -limits[tail, 0])
        np.testing.assert_allclose(values["force_frac"][:, tail], np.abs(force) / side)
        assert np.all(np.isfinite(values["joint_pos"])) and np.all(np.isfinite(values["joint_vel"]))


def test_the_summary_reads_the_gates_window_and_saturation(deterministic):
    from environments.shared.gait.stance_metrics import _saturation_metrics

    report, traces = deterministic
    assert report["schema"] == TRACE_SCHEMA
    assert report["settle_steps"] == SETTLE_STEPS and report["saturation_abs"] == SATURATION_ABS
    for episode, trace in zip(report["episodes"], traces):
        assert episode["length"] == HORIZON and episode["window_steps"] == HORIZON - SETTLE_STEPS
        gate = _saturation_metrics(trace.values["command"][SETTLE_STEPS:])
        assert episode["max_actuator_saturation_fraction"] == gate["max_actuator_saturation_fraction"] == 1.0
        assert episode["mean_actuator_saturation_fraction"] == gate["mean_actuator_saturation_fraction"]
        tail = episode["actuators"][BIASED_ACTUATOR]
        assert tail["saturated_fraction"] == tail["near_saturated_fraction"] == 1.0
        assert tail["raw_mean_beyond_bound_fraction"] == 1.0
        assert tail["command_window_mean"] == 1.0 and tail["raw_mean_window_mean"] > 1.0
        column = trace.values["raw_mean"][:, tail["index"]]
        assert tail["raw_mean_max"] == pytest.approx(column[SETTLE_STEPS:].max())
        assert tail["raw_mean_min"] == pytest.approx(column[SETTLE_STEPS:].min())
        assert tail["raw_mean_episode_max"] == pytest.approx(column.max())
        assert tail["raw_mean_episode_min"] == pytest.approx(column.min())
        assert tail["first_saturated_step"] == 0 and tail["first_saturated_window_step"] == SETTLE_STEPS
        assert 0.0 < tail["force_frac_window_mean"] <= tail["force_frac_window_max"] < 1.0


def test_the_raw_means_extremes_are_the_windows_and_the_reset_transient_is_kept_apart():
    """A raw mean that peaks before the window opens moves the episode's extremes, not the window's."""
    settle, length = 4, 10
    raw = np.full((length, 1), 1.02)
    raw[1, 0], raw[2, 0], raw[7, 0] = 1.30, -0.40, 1.05
    values = {name: np.zeros((length, 1)) for name in episode_trace.TRACE_FIELDS}
    values["raw_mean"], values["command"] = raw, np.clip(raw, -1.0, 1.0)
    trace = episode_trace.EpisodeTrace(seed=0, reward=np.zeros(length), values=values)
    entry = episode_trace.summarize_episode(trace, names=["a"], settle_steps=settle, horizon=length, selected=[0])
    actuator = entry["actuators"]["a"]
    assert actuator["raw_mean_max"] == 1.05 and actuator["raw_mean_min"] == 1.02
    assert actuator["raw_mean_episode_max"] == 1.30 and actuator["raw_mean_episode_min"] == -0.40
    short = episode_trace.summarize_episode(trace, names=["a"], settle_steps=length, horizon=length, selected=[0])
    assert short["actuators"]["a"]["raw_mean_max"] is None and short["actuators"]["a"]["raw_mean_min"] is None
    assert short["actuators"]["a"]["raw_mean_episode_max"] == 1.30


def test_a_hold_replaces_the_command_and_keeps_the_policys_raw_mean(checkpoint, deterministic):
    report, traces = build_episode_trace(
        TYRANNOSAURUS_REX,
        1,
        stage_config=checkpoint.stage_config,
        model_path=str(checkpoint.model),
        vecnorm_path=str(checkpoint.vecnorm),
        seeds=SEEDS[:1],
        actuators=["tail_3"],
        holds=[ActuatorHold.parse(f"{BIASED_ACTUATOR}=0")],
    )
    tail = report["actuator_names"].index(BIASED_ACTUATOR)
    values = traces[0].values
    assert np.all(values["command"][:, tail] == 0.0) and np.all(values["raw_mean"][:, tail] > 1.0)
    # The other actuators still follow the policy: their commands are its clipped mean.
    others = [index for index in range(values["command"].shape[1]) if index != tail]
    np.testing.assert_array_equal(values["command"][:, others], np.clip(values["raw_mean"][:, others], -1.0, 1.0))
    assert report["holds"] == [{"actuator": BIASED_ACTUATOR, "value": 0.0}]
    assert report["traced_actuators"] == [BIASED_ACTUATOR]
    summary = report["episodes"][0]["actuators"][BIASED_ACTUATOR]
    assert summary["saturated_fraction"] == 0.0 and summary["first_saturated_step"] is None
    assert summary["raw_mean_beyond_bound_fraction"] == 1.0
    assert "held" in render_episode_trace(report)


def test_the_stochastic_trace_draws_the_policys_own_noise_from_its_seed(checkpoint):
    from environments.shared.policy_loading import load_sb3_model

    def run():
        return build_episode_trace(
            TYRANNOSAURUS_REX,
            1,
            stage_config=checkpoint.stage_config,
            model_path=str(checkpoint.model),
            vecnorm_path=str(checkpoint.vecnorm),
            seeds=SEEDS,
            stochastic_seed=SEEDS[0],
        )

    report, traces = run()
    _again, repeated = run()
    sigma = np.exp(load_sb3_model(str(checkpoint.model), device="cpu").policy.log_std.detach().numpy().astype(float))
    assert report["deterministic"] is False and report["stochastic_seed"] == SEEDS[0]
    for index, (trace, twin) in enumerate(zip(traces, repeated)):
        np.testing.assert_array_equal(trace.values["command"], twin.values["command"])
        rng = np.random.default_rng(SEEDS[0] + index)
        z = np.stack([rng.standard_normal(sigma.shape[0]) for _ in range(trace.length)])
        expected = trace.values["raw_mean"] + sigma * z
        np.testing.assert_array_equal(trace.values["sample"], expected)
        np.testing.assert_array_equal(trace.values["command"], np.clip(expected, -1.0, 1.0))


def test_policy_raw_mean_is_predicts_mean_before_the_clip_and_draws_nothing(checkpoint):
    """``stance_diagnostics.policy_raw_mean``, the pass the trace and the replay's stance CSV share."""
    import torch

    from environments.shared.policy_loading import load_sb3_model
    from environments.shared.stance_diagnostics import policy_raw_mean

    model = load_sb3_model(str(checkpoint.model), device="cpu")
    obs = np.random.default_rng(1).normal(size=model.observation_space.shape).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    stored = dict(vars(model.policy.action_dist))
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    raw = policy_raw_mean(model, obs)
    assert raw is not None and raw.dtype == np.float64 and raw.shape == action.shape
    np.testing.assert_array_equal(np.clip(raw, -1.0, 1.0), action.astype(np.float64))
    assert raw.max() > 1.0  # the biased tail actuator, beyond the bound the command is clipped to
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert vars(model.policy.action_dist) == stored


def test_the_writer_puts_one_csv_per_episode_and_a_strict_json_summary(deterministic, tmp_path):
    report, traces = deterministic
    written = write_episode_trace(tmp_path, report, traces)
    assert set(written) == {"episode_trace_seed3042.csv", "episode_trace_seed3043.csv", TRACE_SUMMARY_JSON}
    summary = json.loads((tmp_path / TRACE_SUMMARY_JSON).read_text(), parse_constant=lambda name: pytest.fail(name))
    assert summary["episodes"][0]["actuators"][BIASED_ACTUATOR]["saturated_fraction"] == 1.0
    with (tmp_path / "episode_trace_seed3042.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == HORIZON
    assert [row["step"] for row in rows[:2]] == ["0", "1"]
    assert rows[SETTLE_STEPS - 1]["in_window"] == "0" and rows[SETTLE_STEPS]["in_window"] == "1"
    header = list(rows[0])
    for field in ("raw_mean", "command", "ctrl", "joint_pos", "joint_vel", "force", "force_frac"):
        assert f"{BIASED_ACTUATOR}_{field}" in header
    assert not any(name.endswith("_sample") for name in header)
    assert float(rows[0][f"{BIASED_ACTUATOR}_command"]) == 1.0
    tail = report["actuator_names"].index(BIASED_ACTUATOR)
    assert float(rows[0][f"{BIASED_ACTUATOR}_raw_mean"]) == traces[0].values["raw_mean"][0, tail]


def test_the_refusals(checkpoint, tmp_path):
    run = tmp_path / "run"
    (run / "01_stance" / "models").mkdir(parents=True)
    (run / "provenance.json").write_text("{}")
    model = run / "01_stance" / "models" / "model.zip"
    with pytest.raises(EpisodeTraceError, match="inside the run directory"):
        refuse_run_directory_output(run / "trace", [model, None])
    refuse_run_directory_output(tmp_path / "elsewhere", [model])

    def build(**overrides):
        kwargs = {
            "stage_config": checkpoint.stage_config,
            "model_path": str(checkpoint.model),
            "vecnorm_path": str(checkpoint.vecnorm),
            "seeds": SEEDS[:1],
        }
        kwargs.update(overrides)
        return build_episode_trace(TYRANNOSAURUS_REX, 1, **kwargs)

    with pytest.raises(EpisodeTraceError, match="no VecNormalize statistics"):
        build(vecnorm_path=None)
    with pytest.raises(EpisodeTraceError, match="no actuator name contains"):
        build(actuators=["wing"])
    with pytest.raises(EpisodeTraceError, match="names no actuator"):
        build(holds=[ActuatorHold("tail_9_pitch_act", 0.0)])
    with pytest.raises(EpisodeTraceError, match="outside the action bound"):
        build(holds=[ActuatorHold(BIASED_ACTUATOR, 1.5)])
    with pytest.raises(EpisodeTraceError, match="twice"):
        build(holds=[ActuatorHold(BIASED_ACTUATOR, 0.0), ActuatorHold(BIASED_ACTUATOR, 0.5)])
    with pytest.raises(ValueError, match="at least one"):
        build(seeds=())
    for text in ("tail_3_pitch_act", "=0", "tail_3_pitch_act=x", "tail_3_pitch_act=nan"):
        with pytest.raises(ValueError):
            ActuatorHold.parse(text)
    assert ActuatorHold.parse(" tail_3_pitch_act = -0.5") == ActuatorHold("tail_3_pitch_act", -0.5)


def test_an_unnormalized_trace_is_allowed_only_explicitly_and_recorded(checkpoint):
    report, traces = build_episode_trace(
        TYRANNOSAURUS_REX,
        1,
        stage_config=checkpoint.stage_config,
        model_path=str(checkpoint.model),
        vecnorm_path=None,
        seeds=SEEDS[:1],
        allow_unnormalized=True,
    )
    assert report["normalization"] == {"mode": "none"} and report["vecnorm"] is None
    assert traces[0].length == HORIZON


class TestScript:
    """The CLI: its seeds, its refusals as exits, and what it writes."""

    @staticmethod
    def _main(argv, monkeypatch, checkpoint):
        import environments.shared.config as config_module
        from environments.shared.scripts import episode_trace_report

        monkeypatch.setattr(config_module, "load_stage_config", lambda *args, **kwargs: checkpoint.stage_config)
        base = ["--species", "tyrannosaurus_rex", "--embodiment", "anatomical", "--model", str(checkpoint.model)]
        return episode_trace_report.main([*base, *argv])

    def test_it_traces_and_writes(self, checkpoint, monkeypatch, tmp_path, capsys):
        out = tmp_path / "out"
        argv = ["--vecnorm", str(checkpoint.vecnorm), "--seeds", "3043", "--actuators", "tail_3", "--out-dir", str(out)]
        assert self._main(argv, monkeypatch, checkpoint) == 0
        printed = capsys.readouterr().out
        assert "Episode trace" in printed and BIASED_ACTUATOR in printed and "written:" in printed
        summary = json.loads((out / TRACE_SUMMARY_JSON).read_text())
        assert summary["seeds"] == [3043] and summary["traced_actuators"] == [BIASED_ACTUATOR]
        assert (out / "episode_trace_seed3043.csv").is_file()

    def test_it_refuses_an_output_inside_the_run(self, checkpoint, monkeypatch, tmp_path):
        (tmp_path / "provenance.json").write_text("{}")
        vecnorm = tmp_path / "models" / "model_vecnorm.pkl"
        argv = ["--vecnorm", str(vecnorm), "--out-dir", str(tmp_path / "trace")]
        with pytest.raises(SystemExit, match="inside the run directory"):
            self._main(argv, monkeypatch, checkpoint)
        assert not (tmp_path / "trace").exists()

    @pytest.mark.parametrize(
        "argv",
        [["--seeds", "3042,x"], ["--seeds", ","], ["--seeds", "3042,3042"], ["--episodes", "0"], ["--hold", "tail"]],
    )
    def test_usage_errors(self, checkpoint, monkeypatch, argv):
        with pytest.raises(SystemExit) as excinfo:
            self._main(["--vecnorm", str(checkpoint.vecnorm), *argv], monkeypatch, checkpoint)
        assert excinfo.value.code == 2

    def test_seeds_default_to_consecutive_ones_from_seed(self):
        import argparse

        from environments.shared.scripts.episode_trace_report import _seeds

        parser = argparse.ArgumentParser()
        assert _seeds(parser, argparse.Namespace(seeds=None, seed=7, episodes=3)) == [7, 8, 9]
        assert _seeds(parser, argparse.Namespace(seeds="5, 3", seed=7, episodes=3)) == [5, 3]


def test_the_module_is_not_reexported_from_the_reporting_package():
    import environments.shared.reporting as reporting

    assert not hasattr(reporting, "build_episode_trace")
    assert Path(episode_trace.__file__).name == "episode_trace.py"
