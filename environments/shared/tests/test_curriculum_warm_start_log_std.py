"""``[curriculum] warm_start_log_std`` (walking pilot 2; pilot-only): the warm start's exploration-noise reset.

The key writes one value into every entry of a PPO policy's ``log_std`` after the parent's parameters load and before
learning, on a load across the node's edge under ``initialize_next_stage`` only: never on a same-stage resume, a
fresh start or a root, and never when the stage declares no key.  The gate schema validates it (its one reader,
``declared_warm_start_log_std``); ``stage_config.json`` records it in the ``curriculum`` block, as
``run.warm_start_log_std`` (kept across a resume, like the edge) and in ``run.hyperparameters_sha256``; and the
optimizer keeps the same parameter with the parent's Adam moments.

The schema and the rule run everywhere; the policy and ``train`` cases need stable-baselines3 and skip without it
(the SB3 job runs this file through its ``test_curriculum_*.py`` glob).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from environments.shared.config import (
    hyperparameters_sha256,
    load_all_stages,
    load_stage_config,
    recorded_hyperparameters_sha256,
)
from environments.shared.curriculum.gate_schema import (
    GATE_SCHEMA_VERSION,
    WARM_START_LOG_STD_KEY,
    GateSchemaError,
    declared_warm_start_log_std,
    validate_gate_config,
)
from environments.shared.model_id import ModelId
from environments.shared.train_base import _warm_start_log_std

from .reporting_helpers import make_plant_identity

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
_GATE = {"gate_schema_version": GATE_SCHEMA_VERSION, "gate_kind": "reward_and_length/v1", "min_avg_reward": 100.0}
#: The parent's std on the pilot branch's committed checkpoint is 0.018-0.164 (mean 0.095); this is its mean.
_PARENT_LOG_STD = math.log(0.095)


# ---------------------------------------------------------------------------------------------------- the schema
class TestSchema:
    @pytest.mark.parametrize("value", [-1.5, -2, 0.0])
    def test_a_finite_number_is_accepted(self, value):
        assert validate_gate_config("locomotion", dict(_GATE, warm_start_log_std=value)) == "reward_and_length/v1"
        assert declared_warm_start_log_std({WARM_START_LOG_STD_KEY: value}) == float(value)

    @pytest.mark.parametrize("value", [True, "-1.5", float("nan"), float("inf"), None, [-1.5]])
    def test_anything_else_is_refused(self, value):
        with pytest.raises(GateSchemaError, match="warm_start_log_std must be a finite number"):
            validate_gate_config("locomotion", dict(_GATE, warm_start_log_std=value))

    def test_absent_reads_none(self):
        assert declared_warm_start_log_std(_GATE) is None

    def test_a_misspelling_is_refused_as_unknown(self):
        with pytest.raises(GateSchemaError, match="unrecognised"):
            validate_gate_config("locomotion", dict(_GATE, warm_start_logstd=-1.5))

    def test_the_committed_locomotion_stage_declares_the_stance_noise_and_no_other_stage_declares_any(self):
        from environments.shared.species_names import model_ids
        from environments.shared.stage_manifest import load_stage_manifest

        declared = {}
        for model_id in model_ids():
            for entry in load_stage_manifest(model_id).stages:
                curriculum = load_stage_config(model_id, entry.reference)["curriculum_kwargs"]
                validate_gate_config(entry.id, curriculum)
                if WARM_START_LOG_STD_KEY in curriculum:
                    declared[(str(model_id), entry.id)] = curriculum[WARM_START_LOG_STD_KEY]
        assert declared == {(str(TYRANNOSAURUS_REX), "locomotion"): -1.5}
        stance = load_stage_config(TYRANNOSAURUS_REX, "stance")
        assert stance["ppo_kwargs"]["policy_kwargs"]["log_std_init"] == -1.5  # the stance's own starting noise

    def test_the_recipe_digest_covers_it(self):
        config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
        without = copy.deepcopy(config)
        del without["curriculum_kwargs"][WARM_START_LOG_STD_KEY]
        other = copy.deepcopy(config)
        other["curriculum_kwargs"][WARM_START_LOG_STD_KEY] = -1.514
        digests = {hyperparameters_sha256(c, "PPO") for c in (config, without, other)}
        assert len(digests) == 3
        # A record without a run block derives the same digest from its blocks (decision D-B16).
        recorded = {
            "algorithm": "PPO",
            "hyperparameters": config["ppo_kwargs"],
            "curriculum": config["curriculum_kwargs"],
        }
        assert recorded_hyperparameters_sha256(recorded) == hyperparameters_sha256(config, "PPO")


# ---------------------------------------------------------------------------------------------------- when it applies
def _config(value=-1.5):
    return {"curriculum_kwargs": {} if value is None else {WARM_START_LOG_STD_KEY: value}}


class TestWhenItApplies:
    def _resolve(self, value=-1.5, *, algorithm="ppo", mode="initialize_next_stage", parent="stance", load="p.zip"):
        return _warm_start_log_std(
            _config(value),
            stage="locomotion",
            algorithm=algorithm,
            task_load_mode=mode,
            parent_id=parent,
            load_path=load,
        )

    def test_a_warm_start_across_the_edge_applies_it(self):
        assert self._resolve() == -1.5

    @pytest.mark.parametrize(
        "case",
        [
            {"mode": "resume_same_stage"},  # a same-stage resume continues the noise its checkpoint learned
            {"mode": "resume_same_stage", "load": None},  # train()'s fresh start
            {"load": None},  # nothing loaded
            {"parent": None},  # a root has no edge, as stage-entry shaping
            {"value": None},  # the key is absent
        ],
    )
    def test_nothing_else_applies_it(self, case):
        assert self._resolve(**case) is None

    def test_a_malformed_value_fails_a_fresh_start_too(self):
        with pytest.raises(GateSchemaError):
            self._resolve("-1.5", mode="resume_same_stage", load=None)

    def test_sac_is_refused_only_where_it_would_apply(self):
        with pytest.raises(ValueError, match="SAC has no state-independent log_std"):
            self._resolve(algorithm="sac")
        assert self._resolve(algorithm="sac", mode="resume_same_stage") is None


# ---------------------------------------------------------------------------------------------------- the policy
@pytest.fixture
def th():
    """torch, after a skip without stable-baselines3 (the ``test`` extra brings neither)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    return torch


class ToyEnv(gym.Env):
    """The T. rex interface's shapes (63 observations, 15 actuators), random dynamics, 32-step episodes."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (63,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (15,), np.float32)

    def __init__(self):
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.np_random.normal(size=63).astype(np.float32), {}

    def step(self, action):
        self.t += 1
        obs = self.np_random.normal(size=63).astype(np.float32)
        return obs, float(obs[3] * action[4] - np.sum(np.square(action))), False, self.t >= 32, {}


def _model(**kwargs):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    policy_kwargs = {"net_arch": [16, 16], "log_std_init": _PARENT_LOG_STD}
    return PPO(
        "MlpPolicy",
        DummyVecEnv([ToyEnv]),
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        seed=5,
        device="cpu",
        policy_kwargs=policy_kwargs,
        **kwargs,
    )


def _adam_state(model, parameter) -> dict:
    import torch

    return {name: torch.as_tensor(value).clone() for name, value in model.policy.optimizer.state[parameter].items()}


class TestTheReset:
    def test_every_actuator_is_set_in_place_and_the_optimizer_keeps_its_parameter_and_moments(self, th, caplog):
        from environments.shared.train_base import _apply_warm_start_log_std

        model = _model()
        model.learn(total_timesteps=128)  # Adam moments exist, as on a loaded parent
        log_std = model.policy.log_std
        moments = _adam_state(model, log_std)
        others = {k: v.clone() for k, v in model.policy.state_dict().items() if k != "log_std"}
        parent = log_std.detach().exp()

        with caplog.at_level(logging.WARNING, logger="environments.shared.train_base"):
            _apply_warm_start_log_std(model, -1.5)

        assert model.policy.log_std is log_std
        assert th.equal(log_std.detach(), th.full((15,), -1.5))
        assert any(p is log_std for group in model.policy.optimizer.param_groups for p in group["params"])
        kept = _adam_state(model, log_std)
        assert kept.keys() == moments.keys() and all(th.equal(kept[n], moments[n]) for n in moments)
        assert all(th.equal(v, others[k]) for k, v in model.policy.state_dict().items() if k != "log_std")
        (record,) = [r for r in caplog.records if r.getMessage().startswith("Warm start:")]
        assert record.levelno == logging.WARNING
        assert record.getMessage().startswith("Warm start: policy log_std set to -1.5 (std 0.223) on all 15 actuators")
        assert f"mean {float(parent.mean()):.3f}" in record.getMessage()

        model.learn(total_timesteps=128, reset_num_timesteps=False)  # learning continues from the new value
        assert not th.equal(log_std.detach(), th.full((15,), -1.5))
        assert float(log_std.detach().exp().mean()) == pytest.approx(math.exp(-1.5), abs=0.01)

    def test_a_policy_without_a_state_independent_log_std_is_refused(self, th):
        from environments.shared.train_base import _apply_warm_start_log_std

        with pytest.raises(ValueError, match="gSDE"):
            _apply_warm_start_log_std(_model(use_sde=True), -1.5)


# ---------------------------------------------------------------------------------------------------- train()
def _zip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({}))
    return path


class TestThroughTrain:
    """The real ``train`` (the notebook's manual and RESUME cells train through it) on the committed T. rex stages,
    with the environments, the model load and the final save stubbed: what the policy holds when ``learn`` starts,
    and what ``stage_config.json`` records."""

    def _train(
        self, tmp_path, monkeypatch, caplog, *, stage_dir, load_path=None, mode="resume_same_stage", configs=None
    ):
        from environments.shared import plant_contract, task_fingerprint, train_base

        seen = {}
        model = _model()

        def create_or_load_model(sb3, algorithm, alg_kwargs, train_env, load, **kwargs):
            seen["loaded"] = load
            return model

        def learn(**kwargs):
            seen["std_at_learn"] = model.policy.log_std.detach().exp().clone()

        model.learn = learn

        def save_final(model, train_env, model_dir, stage, *rest):
            _zip(model_dir / "stage2_final.zip")
            return model_dir / "stage2_final"

        monkeypatch.setattr(train_base, "current_plant_identity", lambda model_id: make_plant_identity())
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
        monkeypatch.setattr(task_fingerprint, "read_checkpoint_task_fingerprint", lambda path: None)
        monkeypatch.setattr(task_fingerprint, "validate_declared_parent", lambda *a, **k: None)
        monkeypatch.setattr(plant_contract, "write_plant_identity", lambda path, identity: None)
        monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: {"CallbackList": list})
        monkeypatch.setattr(train_base, "create_vec_env", lambda *args, **kwargs: MagicMock())
        monkeypatch.setattr(train_base, "_load_vecnorm_into_envs", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_base, "_create_or_load_model", create_or_load_model)
        monkeypatch.setattr(train_base, "_build_core_callbacks", lambda *a, **k: (["core"], MagicMock(), None))
        monkeypatch.setattr(train_base, "_maybe_ent_coef_decay_callback", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_base, "_stage_entry_shaping_callbacks", lambda *args, **kwargs: [])
        monkeypatch.setattr(train_base, "_save_final_and_sync_tb", save_final)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="environments.shared.train_base"):
            train_base.train(
                SimpleNamespace(model_id=TYRANNOSAURUS_REX, env_class=object),
                load_all_stages(TYRANNOSAURUS_REX) if configs is None else configs,
                2,
                total_timesteps=64,
                load_path=None if load_path is None else str(load_path),
                task_load_mode=mode,
                output_dir=str(stage_dir),
                use_tensorboard=False,
                verbose=0,
                report_metrics=False,
            )
        seen["record"] = json.loads((stage_dir / "stage_config.json").read_text())
        seen["lines"] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Warm start:")]
        return seen

    def test_the_warm_start_sets_it_and_a_resume_keeps_the_record_without_setting_it(
        self, th, tmp_path, monkeypatch, caplog
    ):
        configs = load_all_stages(TYRANNOSAURUS_REX)
        parent = _zip(tmp_path / "parent" / "stage1_6291456_steps.zip")
        stage_dir = tmp_path / "run" / "03_locomotion"

        entered = self._train(
            tmp_path, monkeypatch, caplog, stage_dir=stage_dir, load_path=parent, mode="initialize_next_stage"
        )
        assert entered["loaded"] == str(parent)
        assert th.allclose(entered["std_at_learn"], th.full((15,), math.exp(-1.5)))
        assert entered["lines"] == [
            "Warm start: policy log_std set to -1.5 (std 0.223) on all 15 actuators (the parent's std: mean 0.095, "
            "min 0.095, max 0.095; Adam moments kept)"
        ]
        run = entered["record"]["run"]
        assert run[WARM_START_LOG_STD_KEY] == -1.5 and run["load_mode"] == "initialize_next_stage"
        assert entered["record"]["curriculum"][WARM_START_LOG_STD_KEY] == -1.5
        assert run["hyperparameters_sha256"] == hyperparameters_sha256(configs[2], "PPO")

        # The notebook's RESUME cell: the same stage directory, a periodic checkpoint, resume_same_stage.
        periodic = _zip(stage_dir / "models" / "stage2_2097152_steps.zip")
        resumed = self._train(tmp_path, monkeypatch, caplog, stage_dir=stage_dir, load_path=periodic)
        assert resumed["loaded"] == str(periodic)
        assert th.allclose(resumed["std_at_learn"], th.full((15,), 0.095))  # as loaded, not reset
        assert resumed["lines"] == []
        run = resumed["record"]["run"]
        assert run[WARM_START_LOG_STD_KEY] == -1.5  # the stage entered with it; kept like the edge
        assert run["load_mode"] == "initialize_next_stage" and run["resume_load_path"] == str(periodic)

    def test_a_fresh_start_does_not_set_it(self, th, tmp_path, monkeypatch, caplog):
        fresh = self._train(tmp_path, monkeypatch, caplog, stage_dir=tmp_path / "fresh")
        assert fresh["loaded"] is None
        assert th.allclose(fresh["std_at_learn"], th.full((15,), 0.095))
        assert fresh["lines"] == [] and WARM_START_LOG_STD_KEY not in fresh["record"]["run"]

    def test_a_stage_without_the_key_does_not_set_it(self, th, tmp_path, monkeypatch, caplog):
        configs = load_all_stages(TYRANNOSAURUS_REX)
        del configs[2]["curriculum_kwargs"][WARM_START_LOG_STD_KEY]
        parent = _zip(tmp_path / "parent.zip")
        absent = self._train(
            tmp_path,
            monkeypatch,
            caplog,
            stage_dir=tmp_path / "absent",
            load_path=parent,
            mode="initialize_next_stage",
            configs=configs,
        )
        assert th.allclose(absent["std_at_learn"], th.full((15,), 0.095))
        assert absent["lines"] == [] and WARM_START_LOG_STD_KEY not in absent["record"]["run"]
        assert WARM_START_LOG_STD_KEY not in absent["record"]["curriculum"]

    def test_a_malformed_value_is_refused_before_the_stage_directory_exists(self, th, tmp_path, monkeypatch, caplog):
        configs = load_all_stages(TYRANNOSAURUS_REX)
        configs[2]["curriculum_kwargs"][WARM_START_LOG_STD_KEY] = "-1.5"
        with pytest.raises(GateSchemaError):
            self._train(tmp_path, monkeypatch, caplog, stage_dir=tmp_path / "bad", configs=configs)
        assert not (tmp_path / "bad").exists()
