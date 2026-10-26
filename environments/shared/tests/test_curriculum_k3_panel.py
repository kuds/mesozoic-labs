"""The in-run K3 noise panel (``k3_panel``): its sigma, its launch option, and its callback on a real T. rex env.

The sigma reader, the launch option and the command line need no SB3.  The
callback tests build an untrained PPO (no ``learn()``, no gradient step) on
the T. rex stance env with a short horizon and roll the panel on two spawns:
the panel leaves the policy's training flag, the normalizer's arrays, the
global generators and torch's thread count as they were; its rows are an
offline :func:`run_k3_panel` on the same policy bit for bit; a resumed
session keeps the prior panels, copies the prior files aside and rolls a
missing grid panel on the restored statistics; a failing panel is recorded
and costs nothing else.  The proof that training is bit-identical with the
panel on is ``test_curriculum_balance_telemetry.py``.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.shared import k3_panel as K
from environments.shared.model_id import ModelId

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
SIDECAR = (
    Path(__file__).resolve().parents[3] / "docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json"
)
#: The panel's horizon here: 20 steps past the stance settle window.
HORIZON_PAST_SETTLE = 20


# ---------------------------------------------------------------------------------------------- the sigma
def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class TestJointSigma:
    def test_a_bare_seed_reads_its_own_vector_then_the_mean(self):
        sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
        own = K.read_joint_sigma(SIDECAR, "42")
        mean = K.read_joint_sigma(SIDECAR, "80")
        assert own.path_in_file == "source_vectors.42" and mean.path_in_file == "control_sigma.80"
        assert np.array_equal(own.vector, np.asarray(sidecar["source_vectors"]["42"]["vector"], dtype=np.float64))
        assert np.array_equal(mean.vector, np.asarray(sidecar["control_sigma"]["80"]["vector"], dtype=np.float64))
        assert mean.file_sha256 == hashlib.sha256(SIDECAR.read_bytes()).hexdigest()
        assert mean.sha256 == hashlib.sha256(mean.vector.astype("<f8").tobytes()).hexdigest()
        record = mean.record()
        assert record["mean"] == pytest.approx(0.1607125300684153, abs=1e-15) and record["problem"] is None

    def test_a_dotted_path_and_a_bare_list(self, tmp_path):
        path = _write(tmp_path / "sigma.json", {"r10": {"s60": [0.1, 0.2]}, "plain": {"vector": [0.3]}})
        assert K.read_joint_sigma(path, "r10.s60").vector.tolist() == [0.1, 0.2]
        assert K.read_joint_sigma(path, "plain").path_in_file == "plain"

    @pytest.mark.parametrize(
        ("content", "key", "reason"),
        [
            ({"control_sigma": {"80": {"vector": [0.1, -0.2]}}}, "80", "not finite and positive"),
            ({"control_sigma": {"80": {"vector": [0.1, "x"]}}}, "80", "non-number"),
            ({"control_sigma": {}}, "80", "no sigma vector"),
            ([0.1, 0.2], "80", "not a JSON object"),
            ({"a": [0.1]}, None, "no sigma key"),
        ],
    )
    def test_an_unusable_sigma_says_why(self, tmp_path, content, key, reason):
        sigma = K.read_joint_sigma(_write(tmp_path / "sigma.json", content), key)
        assert sigma.vector is None and reason in sigma.problem and sigma.sha256 is None

    def test_no_file_or_an_unreadable_one(self, tmp_path):
        assert "no joint-space sigma file" in K.read_joint_sigma(None, "80").problem
        assert "cannot be read" in K.read_joint_sigma(tmp_path / "missing.json", "80").problem
        (tmp_path / "bad.json").write_text("{", encoding="utf-8")
        assert "not a JSON object" in K.read_joint_sigma(tmp_path / "bad.json", "80").problem

    def test_a_vector_of_another_width_is_withdrawn(self):
        sigma = K.read_joint_sigma(SIDECAR, "80").checked(12)
        assert sigma.vector is None and "15 entries, the action 12" in sigma.problem
        assert K.read_joint_sigma(SIDECAR, "80").checked(15).vector is not None


class TestLaunchOption:
    def test_the_defaults_are_the_judges_k3(self):
        config = K.K3PanelConfig()
        assert config.every == 1_048_576 and config.seeds == tuple(range(3042, 3062)) and config.stage == "stance"
        assert config.sigma_file is None and config.sigma_key is None

    @pytest.mark.parametrize("kwargs", [{"every": 0}, {"every": True}, {"seeds": ()}, {"seeds": (3042, "x")}])
    def test_a_bad_option_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            K.K3PanelConfig(**kwargs)

    def test_train_takes_it_off_by_default(self):
        from environments.shared import train_base

        for function in (train_base.train, train_base._train_stage_body, train_base._build_core_callbacks):
            assert inspect.signature(function).parameters["k3_panel"].default is None, function.__name__

    def test_the_command_line_passes_it_only_with_k3_panel(self):
        from environments.shared.cli import main

        species_cfg = MagicMock()
        species_cfg.model_id = TYRANNOSAURUS_REX
        species_cfg.stage_descriptions = "1=stance"
        stages = MagicMock(return_value={n: {"env_kwargs": {}, "ppo_kwargs": {}} for n in (1, 2, 3)})
        base = ["prog", "train", "--stage", "1", "--timesteps", "1000"]
        for extra, expected in (
            ([], None),
            (["--k3-panel"], K.K3PanelConfig()),
            (
                ["--k3-panel", "--k3-panel-sigma-file", "s.json", "--k3-panel-sigma-key", "60"],
                K.K3PanelConfig(sigma_file="s.json", sigma_key="60"),
            ),
        ):
            trainer = MagicMock()
            with (
                patch("environments.shared.config.load_all_stages", stages),
                patch("environments.shared.train_base.train", trainer),
                patch("sys.argv", [*base, *extra]),
            ):
                main(species_cfg)
            assert trainer.call_args.kwargs["k3_panel"] == expected
        with (
            patch("environments.shared.config.load_all_stages", stages),
            patch("environments.shared.train_base.train", MagicMock()),
            patch("sys.argv", [*base, "--k3-panel-sigma-file", "s.json"]),
            pytest.raises(SystemExit),
        ):
            main(species_cfg)

    def test_the_builder_runs_on_the_stance_stage_only(self, tmp_path):
        stance = {"env_kwargs": {"max_episode_steps": 1000}}
        config = K.K3PanelConfig()
        common = {"model_id": TYRANNOSAURUS_REX, "stage_config": stance, "env_factory": lambda: None}
        assert K.build_k3_panel_callback(None, stage=1, log_dir=tmp_path, **common) is None
        assert K.build_k3_panel_callback(config, stage="recovery", log_dir=tmp_path, **common) is None
        assert K.build_k3_panel_callback(config, stage=1, log_dir=None, **common) is None
        pytest.importorskip("stable_baselines3")
        callback = K.build_k3_panel_callback(config, stage=1, log_dir=tmp_path, **common)
        assert isinstance(callback, K.K3PanelCallback) and callback.horizon == 1000


# ---------------------------------------------------------------------------------------------- the callback
@pytest.fixture(scope="module")
def stance():
    """The T. rex stance env factory with a horizon 20 steps past the settle window, and that horizon."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    config = load_all_stages(TYRANNOSAURUS_REX)[1]
    kwargs = dict(config["env_kwargs"])
    kwargs["max_episode_steps"] = int(config["curriculum_kwargs"]["settle_steps"]) + HORIZON_PAST_SETTLE
    env_class = get_species_config(TYRANNOSAURUS_REX).env_class
    return (lambda: env_class(**kwargs)), int(kwargs["max_episode_steps"])


def _model(factory, steps: int = 12):
    """An untrained PPO on a VecNormalize of the stance env, its statistics moved by a few zero-action steps."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = VecNormalize(DummyVecEnv([factory]))
    venv.seed(5)
    venv.reset()
    width = int(np.prod(venv.action_space.shape or ()))
    for _ in range(steps):
        venv.step(np.zeros((1, width), dtype=np.float32))
    torch.manual_seed(3)
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, seed=3, device="cpu")
    return model, venv


def _state() -> tuple:
    import torch

    numpy_state: Any = np.random.get_state()
    return (
        hashlib.sha256(np.ascontiguousarray(numpy_state[1]).tobytes()).hexdigest(),
        numpy_state[2:],
        repr(random.getstate()),
        hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        torch.get_num_threads(),
    )


def _callback(factory, horizon: int, stage_dir: Path, **config) -> K.K3PanelCallback:
    options = {"sigma_file": str(SIDECAR), "sigma_key": "80", "every": 64, "seeds": (3042, 3043), **config}
    return K.K3PanelCallback(
        config=K.K3PanelConfig(**options),
        env_factory=factory,
        horizon=horizon,
        stage_dir=stage_dir,
        model_id=TYRANNOSAURUS_REX,
    )


def test_a_panel_reads_the_live_policy_and_statistics_and_moves_nothing(stance, tmp_path):
    import torch

    from environments.shared.policy_probe import GlobalRngGuard

    factory, horizon = stance
    model, venv = _model(factory)
    callback = _callback(factory, horizon, tmp_path)
    callback.init_callback(model)
    model.num_timesteps = 0
    callback.on_training_start({}, {})
    assert callback._twin is not None and callback._twin is not model.policy and not callback.panels
    mean, var, count = (np.array(getattr(venv.obs_rms, name), copy=True) for name in ("mean", "var", "count"))
    training, before = model.policy.training, _state()

    model.num_timesteps = 63
    assert callback.on_step() is True and not callback.panels  # before the grid step
    model.num_timesteps = 64
    assert callback.on_step() is True
    model.num_timesteps = 100
    assert callback.on_step() is True  # the next grid step is 128
    assert _state() == before and model.policy.training is training
    assert np.array_equal(venv.obs_rms.mean, mean) and np.array_equal(venv.obs_rms.var, var)
    assert np.array_equal(venv.obs_rms.count, count) and venv.training is True

    (panel,) = callback.panels
    assert panel["kind"] == "grid" and panel["timesteps"] == 64 and panel["grid_point"] == 64
    assert panel["failure"] is None and panel["global_rng_moved"] is False
    assert panel["own_episodes"] == panel["joint_episodes"] == 2
    assert panel["joint_sigma_sha256"] == K.read_joint_sigma(SIDECAR, "80").sha256
    assert panel["own_sigma"] == K.own_sigma(model.policy).tolist()
    assert panel["obs_mean_sha256"] == hashlib.sha256(mean.astype("<f8").tobytes()).hexdigest()
    assert panel["policy_state_sha256"] == K.state_dict_sha256(model.policy.state_dict())

    # The same arithmetic on the live policy itself (no twin), one thread as the panel runs it: bit for bit.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with GlobalRngGuard():
            offline = K.run_k3_panel(
                model.policy,
                K.copy_normalization(venv),
                factory,
                joint_sigma=K.read_joint_sigma(SIDECAR, "80").vector,
                seeds=(3042, 3043),
                horizon=horizon,
            )
    finally:
        torch.set_num_threads(threads)
    assert [(row["test"], row["seed"]) for row in offline] == [
        ("own", 3042),
        ("own", 3043),
        ("joint", 3042),
        ("joint", 3043),
    ]
    with open(tmp_path / K.K3_PANEL_CSV_FILENAME, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [
        (row["test"], int(row["seed"]), int(row["length"]), float(row["return"]), int(row["fall"])) for row in rows
    ] == [(row["test"], row["seed"], row["length"], row["return"], row["fall"]) for row in offline]
    assert all(row["sigma_sha256"] for row in rows)

    callback.on_training_end()  # inside learn(), before the final pair is saved: nothing
    assert [p["kind"] for p in callback.panels] == ["grid"]
    model.num_timesteps = 120
    callback.roll_final_panel()  # what train_base calls once the final pair is saved
    assert [p["kind"] for p in callback.panels] == ["grid", "final"] and _state() == before
    assert callback.panels[1]["timesteps"] == 120 and callback.panels[1]["grid_point"] is None
    document = json.loads((tmp_path / K.K3_PANEL_JSON_FILENAME).read_text(encoding="utf-8"))
    assert document["schema"] == K.K3_PANEL_SCHEMA and [p["panel"] for p in document["panels"]] == [0, 1]
    assert document["spec"]["joint_sigma"]["path_in_file"] == "control_sigma.80"
    assert document["spec"]["horizon"] == horizon and document["spec"]["seeds"] == [3042, 3043]


def test_a_resumed_session_keeps_the_prior_panels_and_rolls_a_missing_grid_panel(stance, tmp_path):
    factory, horizon = stance
    model, venv = _model(factory)
    first = _callback(factory, horizon, tmp_path)
    first.init_callback(model)
    model.num_timesteps = 0
    first.on_training_start({}, {})
    for step in (64, 128, 192):
        model.num_timesteps = step
        first.on_step()
    assert [p["timesteps"] for p in first.panels] == [64, 128, 192]

    # A resume from the checkpoint at 128: the prior panels after it are dropped, the grid panel at 128 is kept, so
    # nothing is rolled at training start.
    again = _callback(factory, horizon, tmp_path)
    assert again.seed_prior_record(tmp_path, max_timesteps=128) == 2
    again.init_callback(model)
    model.num_timesteps = 128
    again.on_training_start({}, {})
    assert [p["timesteps"] for p in again.panels] == [64, 128]

    # A resume on a grid step whose panel is missing: rolled at training start on the restored statistics, held
    # before learn() would reset (and so move) the training env's.
    resumed = _callback(factory, horizon, tmp_path)
    assert resumed.seed_prior_record(tmp_path, max_timesteps=64) == 1
    resumed.hold_restored_normalization(venv)
    held = np.array(venv.obs_rms.mean, copy=True)
    venv.reset()  # what learn()'s setup does: the statistics move
    assert not np.array_equal(venv.obs_rms.mean, held)
    resumed.init_callback(model)
    model.num_timesteps = 128
    resumed.on_training_start({}, {})
    panels = resumed.panels
    assert [(p["timesteps"], p["kind"]) for p in panels] == [(64, "grid"), (128, "resume")]
    assert panels[1]["obs_mean_sha256"] == hashlib.sha256(held.astype("<f8").tobytes()).hexdigest()
    assert "restored checkpoint pair" in panels[1]["normalization"] and panels[1]["panel"] == 1
    priors = sorted(path.name for path in tmp_path.glob("k3_panel.prior_*"))
    assert len(priors) == 2 and priors[0].endswith(".csv") and priors[1].endswith(".json")
    with open(tmp_path / K.K3_PANEL_CSV_FILENAME, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted({(int(row["panel"]), int(row["timesteps"])) for row in rows}) == [(0, 64), (1, 128)]
    model.num_timesteps = 150
    resumed.on_step()
    assert len(resumed.panels) == 2  # the next grid step is 192


def test_a_failing_panel_is_recorded_and_costs_nothing_else(stance, tmp_path):
    factory, horizon = stance
    model, _venv = _model(factory)

    def broken():
        raise RuntimeError("no env today")

    callback = _callback(factory, horizon, tmp_path)
    callback.env_factory = broken
    callback.init_callback(model)
    model.num_timesteps = 0
    callback.on_training_start({}, {})
    before = _state()
    model.num_timesteps = 64
    assert callback.on_step() is True
    (panel,) = callback.panels
    assert panel["failure"] == "RuntimeError: no env today" and _state() == before
    assert json.loads((tmp_path / K.K3_PANEL_JSON_FILENAME).read_text(encoding="utf-8"))["panels"][0]["failure"]


def test_without_a_joint_sigma_the_own_test_runs_alone(stance, tmp_path):
    factory, horizon = stance
    model, _venv = _model(factory)
    callback = _callback(factory, horizon, tmp_path, sigma_file=None, sigma_key=None, seeds=(3042,))
    callback.init_callback(model)
    model.num_timesteps = 64
    callback.on_training_start({}, {})  # 64 is on the grid: a fresh callback with no prior record rolls it
    (panel,) = callback.panels
    assert panel["own_episodes"] == 1 and panel["joint_episodes"] is None
    assert "no joint-space sigma file" in panel["tests"]["joint"]["skipped"]


def test_a_policy_without_log_std_gets_no_panel(tmp_path):
    pytest.importorskip("stable_baselines3")
    callback = K.K3PanelCallback(
        config=K.K3PanelConfig(), env_factory=lambda: None, horizon=1000, stage_dir=tmp_path, model_id=None
    )
    model = SimpleNamespace(policy=object(), num_timesteps=0, get_env=lambda: None, logger=None)
    callback.init_callback(model)
    callback.on_training_start({}, {})
    model.num_timesteps = 2_000_000
    assert callback.on_step() is True and not callback.panels
    callback.on_training_end()
    callback.roll_final_panel()
    assert not callback.panels
    document = json.loads((tmp_path / K.K3_PANEL_JSON_FILENAME).read_text(encoding="utf-8"))
    assert "not a Gaussian actor-critic" in document["session_problem"] and document["panels"] == []


def test_train_base_rolls_the_final_panel_after_the_final_save_and_survives_a_failure(caplog):
    """``_train_stage_body`` calls this right after ``_save_final_and_sync_tb`` (and not after an interruption):
    every callback with a ``roll_final_panel`` rolls it, a failing one is logged, and other callbacks are left alone."""
    from environments.shared import train_base

    calls: list[str] = []

    class Rolls:
        def roll_final_panel(self) -> None:
            calls.append("rolled")

    class Fails:
        def roll_final_panel(self) -> None:
            raise RuntimeError("no panel today")

    with caplog.at_level("WARNING", logger=train_base.logger.name):
        train_base._roll_final_k3_panels([object(), Fails(), Rolls()])
    assert calls == ["rolled"] and "the final panel could not be rolled" in caplog.text
    source = inspect.getsource(train_base._train_stage_body)
    assert source.index("_save_final_and_sync_tb(") < source.index("_roll_final_k3_panels(callbacks)")
