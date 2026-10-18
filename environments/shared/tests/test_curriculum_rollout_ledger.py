"""The training-episode ledger (``rollout_ledger``): one row per training episode, report-only.

A real T. rex ``DummyVecEnv`` of two Monitor-wrapped envs is stepped by a
script (1,050 steps at a 300-step horizon; env 0 stands at zero action, env
1 is pushed over after its 230th step), and the callback's rows are checked
against reads taken directly from the same envs after every step, reduced
with the per-sample ``heading_observation`` helpers.  The column set is
checked on all six models, and the failure paths on the same envs.
"""

from __future__ import annotations

import csv
import logging
import math
import types

import numpy as np
import pytest

from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
SETTLE = 200
HORIZON = 300
STEPS = 1050
PUSH_AFTER = 230

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco")


def _vec_env(model_id=TYRANNOSAURUS_REX, n_envs=2, horizon=HORIZON, stage=None):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    config = get_species_config(model_id)
    stages = load_all_stages(model_id)
    kwargs = dict(stages[stage if stage is not None else next(iter(stages))]["env_kwargs"])
    kwargs["max_episode_steps"] = horizon

    def make(rank):
        def init():
            env = Monitor(config.env_class(**kwargs))
            env.reset(seed=11 + rank)
            return env

        return init

    vec = VecNormalize(DummyVecEnv([make(rank) for rank in range(n_envs)]))
    vec.seed(11)
    vec.reset()
    return vec


def _callback(vec, tmp_path=None, *, model_id=TYRANNOSAURUS_REX, settle=SETTLE, flush=10**9, **kwargs):
    from stable_baselines3.common.logger import Logger

    from environments.shared.rollout_ledger import TrainingEpisodeLedgerCallback

    model = types.SimpleNamespace(get_env=lambda: vec, logger=Logger(None, []), num_timesteps=0)
    callback = TrainingEpisodeLedgerCallback(
        model_id=model_id, settle_steps=settle, ledger_dir=tmp_path, flush_every_calls=flush, **kwargs
    )
    callback.init_callback(model)
    return callback, model


def _direct_reads(unwrapped, layout):
    """The reads the ledger copies, taken directly from the env, per-sample (no vectorised frame)."""
    from environments.shared.heading_observation import body_to_world, world_to_heading

    data = unwrapped.data
    com = np.array(data.subtree_com[layout.root_body])
    rotation = body_to_world(np.array(data.xquat[layout.root_body]))
    forward = world_to_heading(rotation)[0, :2]
    feet = np.array(data.site_xpos[layout.sites]).mean(axis=0)
    return {
        "com_ahead": float(forward @ (com - feet)[:2]),
        "pitch": math.degrees(math.asin(max(-1.0, min(1.0, rotation[2, 0])))),
        "touch": np.array(data.sensordata[layout.touch_addresses]),
        "force": np.array(data.actuator_force),
    }


def _run(vec, callback, model, *, steps=STEPS, push_env=1, push_after=PUSH_AFTER, parallel=True):
    """Step *vec* with the callback attached; returns the per-env episodes of direct reads and their infos."""
    callback.on_training_start({}, {})
    nu = vec.action_space.shape[0]
    envs = vec.venv.envs
    layouts = callback._layouts
    current = [[] for _ in envs]
    finished: list[list[tuple[list, dict]]] = [[] for _ in envs]
    in_episode = [0] * len(envs)
    for _ in range(steps):
        actions = np.zeros((len(envs), nu), dtype=np.float32)
        if push_env is not None and in_episode[push_env] >= push_after:
            actions[push_env, :] = -1.0
        _obs, _rewards, dones, infos = vec.step(actions)
        model.num_timesteps += len(envs)
        callback.update_locals({"dones": dones, "infos": infos})
        assert callback.on_step() is True
        for index, done in enumerate(dones):
            if done:
                finished[index].append((current[index], infos[index]))
                current[index] = []
                in_episode[index] = 0
            else:
                in_episode[index] += 1
                if parallel:
                    current[index].append(_direct_reads(envs[index].unwrapped, layouts[index]))
    callback.on_rollout_end()
    callback.on_training_end()
    return finished


@pytest.fixture(scope="module")
def scripted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ledger")
    vec = _vec_env()
    try:
        callback, model = _callback(vec, directory, flush=500)
        finished = _run(vec, callback, model)
    finally:
        vec.close()
    with open(directory / "train_episodes.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return callback, finished, rows


def _rows_of(rows, env):
    return [row for row in rows if int(row["env"]) == env]


class TestTheRows:
    def test_one_row_per_episode_with_monitors_length_return_and_the_cause(self, scripted):
        _callback_, finished, rows = scripted
        assert len(rows) == sum(len(episodes) for episodes in finished)
        causes = set()
        for env, episodes in enumerate(finished):
            mine = _rows_of(rows, env)
            assert len(mine) == len(episodes) >= 3
            for row, (_reads, info) in zip(mine, episodes):
                assert int(row["length"]) == info["episode"]["l"]
                assert float(row["return"]) == pytest.approx(info["episode"]["r"])
                expected = "truncated" if info.get("TimeLimit.truncated") else info.get("termination_reason")
                assert row["cause"] == expected
                causes.add(row["cause"])
        assert "truncated" in causes and len(causes) >= 2, causes
        assert all(row["cause"] == "truncated" for row in _rows_of(rows, 0))

    def test_the_posture_columns_match_direct_reads_reduced_per_sample(self, scripted):
        _callback_, finished, rows = scripted
        checked = 0
        for env, episodes in enumerate(finished):
            for row, (reads, _info) in zip(_rows_of(rows, env), episodes):
                window = reads[SETTLE:]
                assert len(reads) == int(row["length"]) - 1  # the terminal reading is not one of them
                assert float(row["com_fore_aft_last_m"]) == pytest.approx(reads[-1]["com_ahead"], abs=1e-12)
                assert float(row["root_pitch_last_deg"]) == pytest.approx(reads[-1]["pitch"], abs=1e-12)
                if not window:
                    assert math.isnan(float(row["com_fore_aft_mean_m"]))
                    assert math.isnan(float(row["root_pitch_mean_deg"]))
                    continue
                checked += 1
                assert float(row["com_fore_aft_mean_m"]) == pytest.approx(
                    np.mean([read["com_ahead"] for read in window]), abs=1e-12
                )
                assert float(row["root_pitch_mean_deg"]) == pytest.approx(
                    np.mean([read["pitch"] for read in window]), abs=1e-12
                )
        assert checked >= 4

    def test_the_last_reading_precedes_the_reset(self, scripted):
        """At ``done`` the env has reset: the row's last values are the pre-terminal state's, not the spawn's."""
        _callback_, finished, rows = scripted
        falls = [
            (row, reads) for row, (reads, _i) in zip(_rows_of(rows, 1), finished[1]) if row["cause"] != "truncated"
        ]
        assert falls
        for row, reads in falls:
            assert abs(float(row["root_pitch_last_deg"]) - reads[0]["pitch"]) > 1.0

    def test_the_touch_shares_are_window_ratios_of_the_sensor_readings(self, scripted):
        callback, finished, rows = scripted
        touch = callback._layouts[0].touch
        assert [foot.label for foot in touch.feet] == ["r", "l"]
        for env, episodes in enumerate(finished):
            for row, (reads, _info) in zip(_rows_of(rows, env), episodes):
                window = reads[SETTLE:]
                if not window:
                    assert all(math.isnan(float(row[column])) for column in touch.columns)
                    continue
                sums = np.sum([read["touch"] for read in window], axis=0)
                offset = 0
                for foot in touch.feet:
                    foot_sums = sums[offset : offset + len(foot.addresses)]
                    offset += len(foot.addresses)
                    shares = [float(row[column]) for column in foot.columns]
                    expected = [*(foot_sums / foot_sums.sum()), foot_sums[1:].sum() / foot_sums.sum()]
                    assert shares == pytest.approx(expected, abs=1e-12)
                    pad, digits = shares[0], shares[-1]
                    assert pad + digits == pytest.approx(1.0, abs=1e-12)
                    assert digits == pytest.approx(sum(shares[1:-1]), abs=1e-12)

    def test_the_limited_force_fraction_is_the_evaluation_keys_definition(self, scripted):
        from environments.shared.gait.balance_telemetry import FORCE_LIMIT_FRACTION

        callback, finished, rows = scripted
        limits = callback._layouts[0].force_limits
        assert np.isfinite(limits).all()  # every T. rex actuator is force-limited
        checked = 0
        for env, episodes in enumerate(finished):
            for row, (reads, _info) in zip(_rows_of(rows, env), episodes):
                window = reads[SETTLE:]
                if not window:
                    continue
                at_limit = []
                for read in window:
                    force = read["force"]
                    limit = np.where(force >= 0.0, limits[:, 1], -limits[:, 0])
                    at_limit.append(bool(np.any(np.abs(force) / limit >= FORCE_LIMIT_FRACTION)))
                assert float(row["actuator_force_limited_fraction"]) == np.mean(at_limit)
                checked += 1
        assert checked >= 4

    def test_the_helper_counts_the_samples_at_either_limit(self):
        """``actuator_limited_fraction`` on synthetic forces (the scripted push falls before any limit binds)."""
        from environments.shared.gait.balance_telemetry import actuator_limited_fraction

        limits = np.array([[-10.0, 20.0], [-5.0, 5.0], [np.nan, np.nan]])
        force = np.array(
            [
                [19.9, 0.0, 1e9],  # 0.995 of the upper limit (the unlimited actuator never counts)
                [-9.8, 0.0, 0.0],  # 0.98 of the lower one: below
                [0.0, -4.95, 0.0],  # 0.99 of the lower one: at it
                [0.0, 0.0, 0.0],
            ]
        )
        assert actuator_limited_fraction(force, limits) == 0.5
        assert math.isnan(actuator_limited_fraction(force[:0], limits))
        assert math.isnan(actuator_limited_fraction(force, np.full((3, 2), np.nan)))

    def test_the_tensorboard_scalars_cover_the_rollouts_episodes(self, tmp_path):
        vec = _vec_env()
        try:
            callback, model = _callback(vec)
            records = []
            model.logger = types.SimpleNamespace(record=lambda *args, **kwargs: records.append((args, kwargs)))
            _run(vec, callback, model, steps=320, push_after=40, parallel=False)
        finally:
            vec.close()
        tags = {args[0]: (args[1], kwargs) for args, kwargs in records}
        assert {"rollout_balance/episodes", "rollout_balance/falls"} <= set(tags)
        assert tags["rollout_balance/episodes"][0] == len(callback.ledger)
        assert tags["rollout_balance/falls"][0] == sum(
            cause != "truncated" for cause in callback.ledger.column("cause")
        )
        assert "rollout_balance/com_fore_aft_last_falls_m" in tags
        assert all(kwargs == {"exclude": ("stdout", "log")} for _value, kwargs in tags.values())


class TestTheColumnSet:
    @pytest.mark.parametrize(
        "model_id",
        [
            TYRANNOSAURUS_REX,
            ModelId("compsognathus_longipes", "anatomical"),
            ModelId("compsognathus_longipes", "robot"),
            ModelId("dibothrosuchus_elaphros", "anatomical"),
            ModelId("velociraptor_mongoliensis", "anatomical"),
            ModelId("brachiosaurus_altithorax", "anatomical"),
        ],
        ids=str,
    )
    def test_the_touch_shares_exist_on_the_tyrannosaurus_rex_only(self, model_id, tmp_path):
        from environments.shared.rollout_ledger import TRAIN_LEDGER_FIXED_COLUMNS

        vec = _vec_env(model_id, n_envs=1, horizon=6)
        try:
            callback, model = _callback(vec, tmp_path, model_id=model_id, settle=0)
            _run(vec, callback, model, steps=7, push_env=None, parallel=False)
        finally:
            vec.close()
        with open(tmp_path / "train_episodes.csv", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header, rows = list(reader.fieldnames or []), list(reader)
        touch = [column for column in header if "_touch_share_" in column]
        if model_id == TYRANNOSAURUS_REX:
            assert touch == [
                f"{label}_touch_share_{stem}"
                for label in ("r", "l")
                for stem in ("pad", "toe_d2", "toe_d3", "toe_d4", "digits")
            ]
        else:
            assert touch == []
        assert header == [*TRAIN_LEDGER_FIXED_COLUMNS, *touch]
        assert not [column for column in header if column.endswith("_digit_share")]  # a foot rail's suffix
        assert len(rows) == 1 and math.isfinite(float(rows[0]["actuator_force_limited_fraction"]))
        assert math.isfinite(float(rows[0]["com_fore_aft_mean_m"]))


class TestFailuresCostTheTelemetryOnly:
    def test_a_raising_read_gives_that_envs_rows_nan_once_and_the_run_goes_on(self, tmp_path, caplog):
        vec = _vec_env()
        try:
            callback, model = _callback(vec, tmp_path, settle=0)
            original = callback._read

            def flaky(index):
                if index == 1:
                    raise RuntimeError("mjData read failed")
                original(index)

            callback._read = flaky
            with caplog.at_level(logging.WARNING, logger="environments.shared.rollout_ledger"):
                _run(vec, callback, model, steps=620, push_env=None, parallel=False)
        finally:
            vec.close()
        assert len([r for r in caplog.records if "could not read" in r.getMessage()]) == 1
        ledger = callback.ledger
        assert ledger.column("env") == [0, 1, 0, 1]
        com = ledger.column("com_fore_aft_mean_m")
        assert math.isfinite(com[0]) and math.isfinite(com[2])
        assert math.isnan(com[1]) and math.isnan(com[3])
        assert ledger.column("length") == [HORIZON] * 4 and ledger.column("cause") == ["truncated"] * 4

    def test_envs_in_other_processes_keep_the_outcome_columns(self, tmp_path, caplog):
        """A SubprocVecEnv's envs cannot be read: the rows keep the outcome and read NaN elsewhere, one warning."""
        from environments.shared.rollout_ledger import TRAIN_LEDGER_FIXED_COLUMNS

        vec = _vec_env(n_envs=1, horizon=5)
        try:
            stub = types.SimpleNamespace(num_envs=1)  # no envs, no venv: what _in_process_envs cannot see into
            callback, model = _callback(vec, tmp_path, settle=0)
            model.get_env = lambda: stub
            with caplog.at_level(logging.WARNING, logger="environments.shared.rollout_ledger"):
                callback.on_training_start({}, {})
                for _ in range(11):
                    _obs, _rewards, dones, infos = vec.step(np.zeros((1, vec.action_space.shape[0]), np.float32))
                    model.num_timesteps += 1
                    callback.update_locals({"dones": dones, "infos": infos})
                    callback.on_step()
                callback.on_training_end()
        finally:
            vec.close()
        assert len([r for r in caplog.records if "not in this process" in r.getMessage()]) == 1
        ledger = callback.ledger
        assert ledger.columns == TRAIN_LEDGER_FIXED_COLUMNS
        assert ledger.column("length") == [5, 5] and ledger.column("cause") == ["truncated", "truncated"]
        assert all(math.isnan(value) for value in ledger.column("com_fore_aft_mean_m"))

    def test_it_writes_nothing_to_the_env_and_returns_true(self):
        vec = _vec_env(n_envs=1, horizon=50)
        try:
            env = vec.venv.envs[0].unwrapped
            callback, model = _callback(vec, settle=0)
            callback.on_training_start({}, {})
            before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.act.copy(), env.data.time)
            callback.update_locals({"dones": np.array([False]), "infos": [{}]})
            assert callback.on_step() is True
            after = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.act.copy(), env.data.time)
        finally:
            vec.close()
        assert all(np.array_equal(a, b) for a, b in zip(before[:3], after[:3])) and before[3] == after[3]


class TestTheFileAndTheResume:
    def test_the_file_is_written_on_the_checkpoint_cadence_and_at_training_end(self, tmp_path, monkeypatch):
        from environments.shared import episode_ledger

        vec = _vec_env(n_envs=1, horizon=5)
        written = []
        try:
            callback, model = _callback(vec, tmp_path, settle=0, flush=7)
            original = episode_ledger.EpisodeLedger.write

            def spy(ledger):
                wrote = original(ledger)
                written.append((callback.n_calls, wrote))
                return wrote

            monkeypatch.setattr(episode_ledger.EpisodeLedger, "write", spy)
            _run(vec, callback, model, steps=30, push_env=None, parallel=False)
        finally:
            vec.close()
        assert [calls for calls, _wrote in written] == [7, 14, 21, 28, 30]
        assert all(wrote for _calls, wrote in written)  # an episode ended between every two writes
        with open(tmp_path / "train_episodes.csv", newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 6

    def test_a_resume_keeps_the_prior_rows_at_or_before_the_resumed_step(self, tmp_path):
        vec = _vec_env(n_envs=1, horizon=5)
        try:
            first, model = _callback(vec, tmp_path, settle=0)
            _run(vec, first, model, steps=20, push_env=None, parallel=False)
            assert first.ledger.column("timesteps") == [5, 10, 15, 20]
            second, model = _callback(vec, tmp_path, settle=0)
            assert second.seed_prior_record(tmp_path, max_timesteps=12) == 2
            model.num_timesteps = 12
            _run(vec, second, model, steps=5, push_env=None, parallel=False)
        finally:
            vec.close()
        with open(tmp_path / "train_episodes.csv", newline="", encoding="utf-8") as handle:
            assert [int(row["timesteps"]) for row in csv.DictReader(handle)] == [5, 10, 17]

    def test_the_summary_covers_its_window_and_the_session(self, scripted):
        callback, _finished, rows = scripted
        summary = callback.summary(timesteps=2 * STEPS)
        assert summary["episodes"] == len(rows) and summary["episodes_total"] == len(rows)
        falls = [row for row in rows if row["cause"] != "truncated"]
        assert summary["falls"] == summary["falls_total"] == len(falls) > 0
        assert summary["fall_fraction"] == pytest.approx(len(falls) / len(rows))
        assert summary["com_fore_aft_last_falls_m"] == pytest.approx(
            np.mean([float(row["com_fore_aft_last_m"]) for row in falls])
        )
        assert set(summary["touch_shares"]) == {"r", "l"}
        assert set(summary["touch_shares"]["r"]) == {"pad", "toe_d2", "toe_d3", "toe_d4", "digits"}
        later = callback.summary(timesteps=2 * STEPS + callback.window_timesteps)
        assert later["episodes"] == 0 and later["episodes_total"] == len(rows)


def _core_callbacks(tmp_path, monkeypatch, *, model_id, n_envs=4, save_freq=400):
    """``train_base._build_core_callbacks`` for the T. rex stance (its mkdtemp dirs kept in tmp_path)."""
    import tempfile

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from environments.shared.config import load_stage_config
    from environments.shared.train_base import _build_core_callbacks

    from .tiny_env_helpers import tiny_env_class

    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    eval_env = DummyVecEnv([tiny_env_class()])
    try:
        callbacks, _, _ = _build_core_callbacks(
            {"CheckpointCallback": CheckpointCallback},
            eval_env,
            tmp_path / "models",
            tmp_path / "logs",
            1,
            n_envs,
            100,
            save_freq,
            0,
            load_stage_config(TYRANNOSAURUS_REX, "stance"),
            model_id=model_id,
        )
    finally:
        eval_env.close()
    return callbacks


class TestTheTrainerWiring:
    def test_the_ledger_runs_after_the_checkpoint_and_diagnostics_callbacks_on_their_cadence(
        self, tmp_path, monkeypatch
    ):
        from stable_baselines3.common.callbacks import CheckpointCallback

        from environments.shared.diagnostics import DiagnosticsCallback
        from environments.shared.rollout_ledger import TrainingEpisodeLedgerCallback

        callbacks = _core_callbacks(tmp_path, monkeypatch, model_id=TYRANNOSAURUS_REX)
        kinds = [type(callback) for callback in callbacks]
        assert kinds.count(TrainingEpisodeLedgerCallback) == 1
        ledger = callbacks[kinds.index(TrainingEpisodeLedgerCallback)]
        assert kinds.index(CheckpointCallback) < kinds.index(DiagnosticsCallback) < kinds.index(type(ledger))
        assert ledger.flush_every_calls == callbacks[kinds.index(CheckpointCallback)].save_freq == 100
        assert ledger.settle_steps == 200 and ledger.model_id == TYRANNOSAURUS_REX
        assert ledger.ledger.path == tmp_path / "logs" / "train_episodes.csv"

    def test_without_a_model_there_is_no_ledger(self, tmp_path, monkeypatch):
        from environments.shared.rollout_ledger import TrainingEpisodeLedgerCallback, build_training_ledger_callback

        callbacks = _core_callbacks(tmp_path, monkeypatch, model_id=None)
        assert not [callback for callback in callbacks if isinstance(callback, TrainingEpisodeLedgerCallback)]
        assert (
            build_training_ledger_callback(
                model_id=TYRANNOSAURUS_REX, stage_config={}, log_dir=None, flush_every_calls=1
            )
            is None
        )
