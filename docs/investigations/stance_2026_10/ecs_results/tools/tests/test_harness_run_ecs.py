"""Tests of run_ecs.py (build list v2 item 9: StopAtEnd, the arm edit and V5's comparison, each preflight refusal,
the folder-exists refusal, the no-update mode, the load wrapper's refusals).

No optimizer step of any kind: the SB3 tests use a toy gymnasium env and a PPO whose train() is replaced by a
counting no-op before learn() (the same replacement --no-update makes); nothing loads the code copy or MuJoCo.
Run: OMP_NUM_THREADS=1 nice -n 15 PY -I -B -m pytest -q -p no:cacheprovider tests/test_harness_run_ecs.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RE = _load("run_ecs")
C = RE.C
F2_PARENT_SC = {s: C.f2_record(s, "stage_config.json") for s in C.STARTS}


# ---------------------------------------------------------------------------------------------------- the arm edit
def _config() -> dict:
    return {"ppo_kwargs": {"ent_coef": 0.005, "ent_coef_end": 0.0, "ent_coef_decay_timesteps": 7000000,
                           "learning_rate": 3e-5, "policy_kwargs": {"net_arch": [64]}},
            "curriculum_kwargs": {"eval_freq": 131072}, "env_kwargs": {"a": 1}}


def test_arm_a_changes_nothing():
    cfg = _config()
    out = RE.arm_config(cfg, "A")
    assert out == cfg and out is not cfg and out["ppo_kwargs"] is not cfg["ppo_kwargs"]


def test_arm_b_changes_exactly_ent_coef_and_ent_coef_end():
    cfg = _config()
    before = copy.deepcopy(cfg)
    out = RE.arm_config(cfg, "B")
    assert cfg == before  # in memory, on a copy
    po, pc = out["ppo_kwargs"], cfg["ppo_kwargs"]
    diff = {k for k in set(po) | set(pc) if po.get(k) != pc.get(k)}
    assert diff == {"ent_coef"}  # ent_coef_end was 0.0 already
    assert out["ppo_kwargs"]["ent_coef"] == 0.0 and out["ppo_kwargs"]["ent_coef_end"] == 0.0
    assert {k: v for k, v in out.items() if k != "ppo_kwargs"} == {k: v for k, v in cfg.items() if k != "ppo_kwargs"}


def test_arm_edit_refuses_a_missing_key_and_an_unknown_arm():
    cfg = _config()
    del cfg["ppo_kwargs"]["ent_coef_end"]
    with pytest.raises(ValueError):
        RE.arm_config(cfg, "B")
    with pytest.raises(ValueError):
        RE.arm_config(_config(), "C")


# ---------------------------------------------------------------------------------------------------- V5's comparison
def _parent(start: str = "s42") -> dict:
    with open(F2_PARENT_SC[start]) as fh:
        return json.load(fh)


def _written(parent: dict, arm: str, *, seed: int = 1101, S: int = 3145728, label: str = "ECS s42 A 1101",
             hp_sha: str | None = None) -> dict:
    w = copy.deepcopy(parent)
    w.pop("gpu", None)
    w["hyperparameters"].update(C.ARM_PPO_EDIT[arm])
    run = w["run"]
    run.update({"seed": seed, "timesteps": C.TOTAL_TIMESTEPS - S, "label": label, "duration_seconds": 12.5,
                "hyperparameters_sha256": hp_sha or C.HYPERPARAMETERS_SHA256_A,
                "load_path": "/x/stage1_3145728_steps.zip",
                "load_mode": "resume_same_stage", "parent_checkpoint_sha256": "sha256:" + C.STARTS["s42"]["zip_sha256"],
                "parent_task_sha256": parent["task_fingerprint"]["task_sha256"]})
    return w


def _kw(arm: str, hp_sha: str | None = None) -> dict:
    return {"arm": arm, "S": 3145728, "seed": 1101, "label": "ECS s42 A 1101",
            "load_path": "/x/stage1_3145728_steps.zip",
            "zip_sha256": C.STARTS["s42"]["zip_sha256"], "hp_sha256_expected": hp_sha or C.HYPERPARAMETERS_SHA256_A}


def test_parent_records_the_pinned_digest():
    for start in C.STARTS:
        p = _parent(start)
        assert p["run"]["hyperparameters_sha256"] == C.HYPERPARAMETERS_SHA256_A
        assert p["git_commit"] == C.COMMIT and p["run"]["timesteps"] == C.TOTAL_TIMESTEPS


def test_v5_arm_a_passes_and_the_expected_to_differ_keys_pass():
    p = _parent()
    assert RE.recipe_problems(_written(p, "A"), p, **_kw("A")) == []


def test_v5_arm_b_passes_with_exactly_its_two_keys_and_its_own_digest():
    p = _parent()
    b_sha = "sha256:" + "b" * 64
    assert RE.recipe_problems(_written(p, "B", hp_sha=b_sha), p, **_kw("B", b_sha)) == []
    # B with A's digest, or A with another digest, is refused
    assert RE.recipe_problems(_written(p, "B"), p, **_kw("B"))
    assert RE.recipe_problems(_written(p, "A", hp_sha=b_sha), p, **_kw("A", b_sha))


@pytest.mark.parametrize("mutate", [
    lambda w: w["reward_weights"].update({next(iter(w["reward_weights"])): "changed"}),
    lambda w: w["curriculum"].update({"eval_freq": 65536}),
    lambda w: w["hyperparameters"].update({"learning_rate": 1.0}),
    lambda w: w["hyperparameters"].update({"ent_coef": 0.004}),
    lambda w: w["plant_identity"].update({"physics_revision": 9}),
    lambda w: w["task_fingerprint"].update({"task_sha256": "sha256:" + "0" * 64}),
    lambda w: w.update({"git_commit": "unknown"}),
    lambda w: w.update({"name": "other"}),
    lambda w: w["run"].update({"n_envs": 8}),
    lambda w: w["run"].update({"seed": 1102}),
    lambda w: w["run"].update({"timesteps": 12000000}),
    lambda w: w["run"].update({"label": "x"}),
    lambda w: w["run"].update({"load_mode": "initialize_next_stage"}),
    lambda w: w["run"].update({"parent_checkpoint_sha256": "sha256:" + "0" * 64}),
    lambda w: w["run"].update({"parent_checkpoint_sha256": C.STARTS["s42"]["zip_sha256"]}),  # unprefixed
    lambda w: w["run"].update({"parent_run_id": "stance6_r10LTSB_s42"}),
    lambda w: w["run"].update({"resume_load_path": "/x"}),
    lambda w: w["run"].update({"duration_seconds": -1}),
    lambda w: w.update({"extra_key": 1}),
])
def test_v5_any_other_difference_is_refused(mutate):
    p = _parent()
    w = _written(p, "A")
    mutate(w)
    assert RE.recipe_problems(w, p, **_kw("A"))


def test_v5_arm_b_refuses_a_third_hyperparameter_change():
    p = _parent()
    b_sha = "sha256:" + "b" * 64
    w = _written(p, "B", hp_sha=b_sha)
    w["hyperparameters"]["ent_coef_decay_timesteps"] = 1
    assert RE.recipe_problems(w, p, **_kw("B", b_sha))


# ---------------------------------------------------------------------------------------------------- pure checks
def _rms(mean, var, count):
    import numpy as np

    return SimpleNamespace(mean=np.asarray(mean, dtype=np.float64), var=np.asarray(var, dtype=np.float64), count=count)


def _stats(**over):
    base = {"obs_rms": _rms([0.1, 0.2], [1.0, 2.0], 100.0), "ret_rms": _rms(0.0, 3.0, 50.0)}
    base.update(over)
    return SimpleNamespace(**base)


def test_stats_equal_pass_and_any_difference_is_refused():
    assert RE.stats_problems(_stats(), _stats()) == []
    assert RE.stats_problems(_stats(obs_rms=_rms([0.1, 0.2000001], [1.0, 2.0], 100.0)), _stats())
    assert RE.stats_problems(_stats(ret_rms=_rms(0.0, 3.1, 50.0)), _stats())
    assert RE.stats_problems(_stats(obs_rms=_rms([0.1, 0.2], [1.0, 2.0], 101.0)), _stats())
    assert RE.stats_problems(_stats(obs_rms=None), _stats())


def test_stop_reason_table():
    assert RE.stop_reason(flag=True, T=10, target=10, collapse_tripped=False) == "end"
    assert RE.stop_reason(flag=True, T=10, target=10, collapse_tripped=True) == "end"
    assert RE.stop_reason(flag=False, T=8, target=10, collapse_tripped=True) == "collapse"
    assert RE.stop_reason(flag=False, T=8, target=10, collapse_tripped=False) == "other"
    assert RE.stop_reason(flag=False, T=10, target=10, collapse_tripped=False) == "other"
    # review H5 (mutant M09): a collapse trip at the target without StopAtEnd's flag is neither "end" nor "collapse"
    assert RE.stop_reason(flag=False, T=10, target=10, collapse_tripped=True) == "other"


def test_post_run_status_is_not_a_crash():
    """Review H3: a failed post-run check exits EXIT_POST_CHECK (5, status "failed"), never the crash code 1."""
    assert RE.post_run_status([], [], []) == ("done", 0)
    for args in ((["params digest changed"], [], []), ([], ["environments: /elsewhere"], []),
                 ([], [], ["code: changed"])):
        assert RE.post_run_status(*args) == ("failed", C.EXIT_POST_CHECK) and C.EXIT_POST_CHECK == 5


def test_no_update_problems():
    d = {"params": "sha256:a", "adam": "sha256:b"}
    assert RE.no_update_problems(d, dict(d), train_calls=31, expected_calls=31) == []
    assert RE.no_update_problems(d, {**d, "params": "sha256:c"}, train_calls=31, expected_calls=31)
    assert RE.no_update_problems(d, {**d, "adam": "sha256:c"}, train_calls=31, expected_calls=31)
    assert RE.no_update_problems(d, dict(d), train_calls=30, expected_calls=31)
    assert RE.no_update_problems({}, {}, train_calls=0, expected_calls=0)  # no digest is not a pass


def test_adam_digest_sees_a_moment_change():
    import torch

    st = {"state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.zeros(4), "exp_avg_sq": torch.ones(4)}}}
    a = RE._hash_arrays(RE.adam_arrays(st))
    st["state"][0]["exp_avg"][1] = 1e-9
    assert RE._hash_arrays(RE.adam_arrays(st)) != a


def test_expected_rows_and_windows():
    assert C.expected_update_rows(3145728, 4718592) == 191
    assert C.expected_update_rows(4194304, 4194304 + 262144) == 31
    assert C.window("s42") == (3145728, 4718592) and C.window("s80") == (4194304, 5767168)
    with pytest.raises(ValueError):
        C.expected_update_rows(0, 100)


# ---------------------------------------------------------------------------------- SB3 toy (no optimizer step)
def _toy_env_cls():
    import gymnasium as gym
    import numpy as np

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-10.0, 10.0, (3,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return np.zeros(3, np.float32), {}

        def step(self, action):
            self.t += 1
            return np.full(3, 0.1 * self.t, np.float32), float(action[0]), self.t >= 7, False, {}

    return Toy


@pytest.fixture()
def toy(tmp_path):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    Toy = _toy_env_cls()
    venv = VecNormalize(DummyVecEnv([Toy, Toy]))
    model = PPO("MlpPolicy", venv, n_steps=4, batch_size=8, n_epochs=1, device="cpu", seed=0, verbose=0,
                policy_kwargs={"net_arch": [8]})
    holder: dict = {}
    RE.install_no_update(model, holder)  # before any learn(): no optimizer step can happen
    model.num_timesteps = 64  # a resume at S = 64, as the periodic zip carries it
    return SimpleNamespace(model=model, venv=venv, holder=holder, Toy=Toy, tmp=tmp_path)


def _callbacks(toy, target, *, collapse_at=None):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class RolloutEnds(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.ends = []

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            self.ends.append(int(self.model.num_timesteps))

    class EvalCollapseEarlyStopCallback(BaseCallback):  # the name collapse_state looks for
        def __init__(self):
            super().__init__(0)
            self.patience, self._consecutive_drops = 3, 0

        def _on_step(self):
            if collapse_at is not None and self.model.num_timesteps == collapse_at:
                self._consecutive_drops = self.patience
                return False
            return True

    eval_env = VecNormalize(DummyVecEnv([toy.Toy]), training=False, norm_reward=False)
    stop = RE.make_stop_at_end_class(BaseCallback)(target)
    ends = RolloutEnds()
    ev = EvalCallback(eval_env, eval_freq=8, n_eval_episodes=1, deterministic=True, log_path=str(toy.tmp / "eval"),
                      warn=False, verbose=0)
    ck = CheckpointCallback(save_freq=16, save_path=str(toy.tmp / "ckpt"), name_prefix="stage1",
                            save_vecnormalize=True)
    # StopAtEnd first: every other callback still runs on the stop step (CallbackList runs all of them).
    return stop, ends, ev, [stop, ends, ev, ck, EvalCollapseEarlyStopCallback()]


def test_stop_at_end_stops_at_the_step_with_the_evaluation_and_checkpoint_there_and_no_last_update(toy):
    target = 96  # S + 4 rollouts of 8 steps
    stop, ends, ev, cbs = _callbacks(toy, target)
    start = RE.state_digest(toy.model, toy.venv)
    toy.model.learn(total_timesteps=12_000, callback=cbs, reset_num_timesteps=False)
    assert toy.model.num_timesteps == target and stop.flag and stop.step == target
    assert list(ev.evaluations_timesteps) == [80, 96]  # the evaluation at the stop step ran
    assert (toy.tmp / "ckpt" / "stage1_96_steps.zip").is_file()
    assert (toy.tmp / "ckpt" / "stage1_vecnormalize_96_steps.pkl").is_file()
    assert ends.ends == [72, 80, 88]  # no rollout end (and no update) on the rollout ending at the stop step
    assert toy.holder["train_calls"] == 3  # (96 - 64) / 8 - 1
    assert stop.last_step == target
    end = RE.state_digest(toy.model, toy.venv)
    calls = toy.holder["train_calls"]
    assert RE.no_update_problems(start, end, train_calls=calls, expected_calls=(96 - 64) // 8 - 1) == []
    assert end["vecnormalize"] != start["vecnormalize"]  # the statistics moved (scratch), the parameters did not
    assert RE.collapse_state(stop.siblings) == {"found": 1, "consecutive_drops": 0, "patience": 3, "tripped": False}
    assert RE.stop_reason(flag=stop.flag, T=96, target=target, collapse_tripped=False) == "end"
    # The checkpoint at the stop step carries the policy as it is (no update on the last rollout).
    assert RE.zip_digests(toy.tmp / "ckpt" / "stage1_96_steps.zip")["params"] == end["params"]
    assert RE.stats_problems(toy.venv, RE.pkl_stats(toy.tmp / "ckpt" / "stage1_vecnormalize_96_steps.pkl")) == []


def test_stop_at_end_heartbeat_at_each_rollout_end_before_its_update(toy):
    """Review H4: the heartbeat (run_meta's rollout_end) fires at each rollout's end, never for the rollout that ends
    at the stop step, and changes nothing the training does."""
    from stable_baselines3.common.callbacks import BaseCallback

    beats = []
    stop, ends, _ev, cbs = _callbacks(toy, 96)
    cbs[0] = stop = RE.make_stop_at_end_class(BaseCallback)(96, heartbeat=beats.append)
    toy.model.learn(total_timesteps=12_000, callback=cbs, reset_num_timesteps=False)
    assert beats == ends.ends == [72, 80, 88] and stop.last_step == 96 and toy.holder["train_calls"] == 3


def test_write_meta_first_exclusive_then_atomic(tmp_path):
    p = tmp_path / "run_meta.json"
    RE._write_meta(p, {"status": "starting"}, "x")
    with pytest.raises(FileExistsError):
        RE._write_meta(p, {"status": "starting"}, "x")
    RE._write_meta(p, {"status": "running", "heartbeat": {"rollout_end": 8192}}, "w")
    assert json.loads(p.read_text())["heartbeat"]["rollout_end"] == 8192
    assert sorted(x.name for x in tmp_path.iterdir()) == ["run_meta.json"]  # no temporary file left


def test_stop_at_end_refuses_a_step_past_the_target(toy):
    stop, _, _, cbs = _callbacks(toy, 97)  # never equal on a 2-env grid: 98 > 97
    with pytest.raises(RuntimeError, match="passed the end step"):
        toy.model.learn(total_timesteps=12_000, callback=cbs, reset_num_timesteps=False)
    assert not stop.flag


def test_a_collapse_stop_before_the_target_reads_collapse(toy):
    stop, _, _, cbs = _callbacks(toy, 96, collapse_at=80)
    toy.model.learn(total_timesteps=12_000, callback=cbs, reset_num_timesteps=False)
    coll = RE.collapse_state(stop.siblings)
    assert toy.model.num_timesteps == 80 and not stop.flag and coll["tripped"]
    assert RE.stop_reason(flag=stop.flag, T=80, target=96, collapse_tripped=coll["tripped"]) == "collapse"


def test_no_update_mode_detects_a_parameter_change(toy):
    import torch

    start = RE.state_digest(toy.model, toy.venv)
    with torch.no_grad():  # a direct edit, standing in for a step the mode must catch (no optimizer involved)
        next(toy.model.policy.parameters()).add_(1.0)
    end = RE.state_digest(toy.model, toy.venv)
    assert RE.no_update_problems(start, end, train_calls=0, expected_calls=0)


# ---------------------------------------------------------------------------------------------------- the load wrapper
def _wrapper(toy, *, arm="B", mode="no_update", start_stats=None, save_loaded=None, holder=None):
    from stable_baselines3 import PPO

    toy.model.__class__ = PPO  # the wrapper checks the loaded class; the fixture's no-update class is undone here
    toy.model._total_timesteps = C.TOTAL_TIMESTEPS
    holder = {} if holder is None else holder
    if start_stats is None:
        toy.venv.save(str(toy.tmp / "start.pkl"))
        start_stats = RE.pkl_stats(toy.tmp / "start.pkl")
    load = RE.make_checked_load(lambda *a, **k: toy.model, mirror_cls=PPO, S=64, arm=arm, start_stats=start_stats,
                                mode=mode, holder=holder, save_loaded=save_loaded)
    return load, holder


def test_wrapper_passes_and_installs_the_no_update_train(toy):
    from stable_baselines3 import PPO

    toy.model.__class__ = PPO  # undo the fixture's install: the wrapper installs it here
    load, holder = _wrapper(toy)
    model = load(None, "ppo", {}, toy.venv, "zip")
    assert model is toy.model and holder["train_calls"] == 0 and holder["ent_coef_after_load"] == 0.0
    model.train()
    assert holder["train_calls"] == 1 and isinstance(model, PPO) and type(model).train is not PPO.train
    assert holder["digest_start"]["num_timesteps"] == 64


def test_no_update_model_saves_without_the_counter(toy):
    """Regression (lane H smoke 1): SB3 serialises instance attributes into the zip; the no-update train must not be
    one.  The holder carries an unpicklable object, as the real run's holder does (StopAtEnd and its siblings)."""
    import threading
    import zipfile

    toy.holder["lock"] = threading.Lock()
    assert "train" not in toy.model.__dict__
    toy.model.save(str(toy.tmp / "m.zip"))
    data = json.loads(zipfile.ZipFile(toy.tmp / "m.zip").read("data"))
    assert "train" not in data and "holder" not in json.dumps(data)
    toy.model.train()
    assert toy.holder["train_calls"] == 1


def test_wrapper_refuses_statistics_that_differ_from_the_pkl(toy):
    toy.venv.save(str(toy.tmp / "s.pkl"))
    stats = RE.pkl_stats(toy.tmp / "s.pkl")
    stats.obs_rms.mean = stats.obs_rms.mean + 1e-6
    load, _ = _wrapper(toy, start_stats=stats)
    with pytest.raises(RE.Refused, match="obs_rms.mean"):
        load(None, "ppo", {}, toy.venv, "zip")
    stats = RE.pkl_stats(toy.tmp / "s.pkl")
    stats.ret_rms.var = stats.ret_rms.var * 2
    load, _ = _wrapper(toy, start_stats=stats)
    with pytest.raises(RE.Refused, match="ret_rms.var"):
        load(None, "ppo", {}, toy.venv, "zip")


@pytest.mark.parametrize("breakit,match", [
    (lambda t: setattr(t.model, "num_timesteps", 65), "num_timesteps"),
    (lambda t: setattr(t.model, "ent_coef", 0.005), "ent_coef"),
    (lambda t: setattr(t.venv, "training", False), "does not update"),
    # review H5: the three refusals no test reached (mutants M03, M04, M06)
    (lambda t: setattr(t.model, "device", "cuda"), "device cuda != cpu"),
    (lambda t: setattr(t.model, "_total_timesteps", C.TOTAL_TIMESTEPS - 1), "_total_timesteps"),
    (lambda t: setattr(t.model, "get_vec_normalize_env", lambda: object()), "is not the training env"),
])
def test_wrapper_refuses_each_load_check(toy, breakit, match):
    load, _ = _wrapper(toy)
    breakit(toy)
    with pytest.raises(RE.Refused, match=match):
        load(None, "ppo", {}, toy.venv, "zip")


def test_wrapper_refuses_a_thread_count_other_than_one(toy, monkeypatch):
    """Review H6: the load wrapper reads torch's intra-op threads and refuses any count but 1; both counts are kept."""
    load, holder = _wrapper(toy)
    monkeypatch.setattr(RE, "torch_threads", lambda: (4, 4))
    with pytest.raises(RE.Refused, match="intra-op threads 4"):
        load(None, "ppo", {}, toy.venv, "zip")
    assert holder["torch_threads"] == 4 and holder["torch_interop_threads"] == 4
    monkeypatch.setattr(RE, "torch_threads", lambda: (1, 4))
    load, holder = _wrapper(toy)
    load(None, "ppo", {}, toy.venv, "zip")
    assert holder["torch_threads"] == 1 and holder["torch_interop_threads"] == 4


def test_wrapper_problems_guard_before_learn(toy):
    """Review I1: the stage_callback factory refuses unless the wrapper ran and, in --no-update, installed the no-op."""
    from stable_baselines3 import PPO

    assert RE.wrapper_problems({}, mode="continuation", update_classes=(PPO,))
    assert RE.wrapper_problems({}, mode="no_update", update_classes=(PPO,))
    load, holder = _wrapper(toy)  # no_update
    load(None, "ppo", {}, toy.venv, "zip")
    assert RE.wrapper_problems(holder, mode="no_update", update_classes=(PPO,)) == []
    bad = dict(holder, model_class=PPO)  # the stepping class: refused
    assert any("no-op train" in p for p in RE.wrapper_problems(bad, mode="no_update", update_classes=(PPO,)))
    toy.model.__class__ = PPO
    load, holder = _wrapper(toy, mode="continuation")
    load(None, "ppo", {}, toy.venv, "zip")
    assert RE.wrapper_problems(holder, mode="continuation", update_classes=(PPO,)) == []
    assert RE.wrapper_problems(holder, mode="no_update", update_classes=(PPO,))  # no no-op installed


def test_wrapper_refuses_another_class_and_another_arm(toy):
    from stable_baselines3 import PPO

    class Other(PPO):
        pass

    toy.model._total_timesteps = C.TOTAL_TIMESTEPS
    toy.venv.save(str(toy.tmp / "s.pkl"))
    stats = RE.pkl_stats(toy.tmp / "s.pkl")
    load = RE.make_checked_load(lambda *a, **k: toy.model, mirror_cls=Other, S=64, arm="B", start_stats=stats,
                                mode="continuation", holder={})
    with pytest.raises(RE.Refused, match="not Other"):
        load(None, "ppo", {}, toy.venv, "zip")
    load, _ = _wrapper(toy, arm="A")  # arm A loads 0.005; the toy has 0.0
    with pytest.raises(RE.Refused, match="ent_coef after load"):
        load(None, "ppo", {}, toy.venv, "zip")


def test_wrapper_save_loaded_only_saves_and_leaves_before_learn(toy):
    calls = []
    load, holder = _wrapper(toy, mode="save_loaded_only", save_loaded=lambda m: calls.append(m) or ["saved"])
    with pytest.raises(RE._LoadedOnly):
        load(None, "ppo", {}, toy.venv, "zip")
    assert calls == [toy.model] and holder["saved"] == ["saved"]


def test_wrapper_continuation_leaves_train_alone(toy):
    from stable_baselines3 import PPO

    toy.model.__class__ = PPO  # undo the fixture's install
    load, holder = _wrapper(toy, mode="continuation")
    model = load(None, "ppo", {}, toy.venv, "zip")
    assert "train" not in model.__dict__ and type(model) is PPO and "train_calls" not in holder


# ---------------------------------------------------------------------------------------------------- preflight
def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    import hashlib

    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def root(tmp_path, monkeypatch):
    """A synthetic <ECS> (APPROVED, criteria, tools, a sealed code copy) and F2 (start pairs and records)."""
    rc = RE._load("t_r10m_common", C.M1_COMMON)
    ecs = tmp_path / "ecs"
    (ecs / "validate").mkdir(parents=True)
    code = ecs / "code"
    _write(code / "environments" / "__init__.py", b"")
    _write(code / "environments" / "shared" / "x.py", b"x = 1\n")
    digest, n = rc.tree_digest(code)
    (code / C.SEAL_NAME).write_text(json.dumps({"commit": C.COMMIT, "tree_digest": digest, "files": n}))
    starts = copy.deepcopy(C.STARTS)
    f2 = tmp_path / "f2"
    for s, st in starts.items():
        base = f2 / st["run_id"] / "copies"
        st["zip_sha256"] = _write(base / (st["zip_rel"] + ".copy1"), f"zip {s}".encode())
        st["pkl_sha256"] = _write(base / (st["pkl_rel"] + ".copy1"), f"pkl {s}".encode())
        for name in st["records"]:
            st["records"][name] = _write(base / "01_stance" / (name + ".copy1"), f"{name} {s}".encode())
    tools = tmp_path / "tools"
    pins = {name: _write(tools / name, f"tool {name}".encode()) for name in C.TOOL_FILES}
    crit = _write(ecs / "criteria.txt", b"the approved text\n")
    approved = {"criteria_sha256": crit, "runlog_line": "x", "code_tree_digest": C.TREE_DIGEST, "tool_sha256": pins}
    (ecs / "APPROVED").write_text(json.dumps(approved))
    for k, v in {"ECS": str(ecs), "CODE": str(code), "F2_ROOT": str(f2), "STARTS": starts}.items():
        monkeypatch.setattr(C, k, v)
    monkeypatch.setattr(C, "check_stack", lambda: [])
    monkeypatch.setattr(C, "check_pins", lambda: [])
    monkeypatch.setattr(RE, "TOOLS", tools)
    monkeypatch.setattr(RE, "free_bytes", lambda p: 10 * 1024**3)
    # The seal pins the real copy's digest; the synthetic copy's stands in for it here.
    monkeypatch.setattr(C, "TREE_DIGEST", digest)
    monkeypatch.setattr(C, "TREE_FILES", n)
    approved["code_tree_digest"] = digest
    (ecs / "APPROVED").write_text(json.dumps(approved))
    return SimpleNamespace(ecs=ecs, code=code, f2=f2, tools=tools, approved=approved, starts=starts)


def _ns(**over):
    base = {"start": "s42", "arm": "A", "seed": 1101, "save_loaded_only": False, "no_update": False, "until": None,
            "out": None, "preflight_only": True}
    base.update(over)
    return SimpleNamespace(**base)


def test_preflight_passes_on_the_synthetic_root(root):
    assert RE.preflight(_ns()) == []
    assert RE.preflight(_ns(start="s80", arm="B", seed=1451)) == []


def test_validation_modes_pass_without_approval(root):
    (root.ecs / "APPROVED").unlink()
    assert RE.preflight(_ns(save_loaded_only=True)) == []
    assert RE.preflight(_ns(no_update=True, until=3145728 + 262144)) == []
    assert RE.preflight(_ns(no_update=True, until=3145728 + 262144, out=str(root.ecs / "validate" / "mine"))) == []
    assert any("APPROVED" in p for p in RE.preflight(_ns()))


def _expect(problems, text):
    assert any(text in p for p in problems), problems


@pytest.mark.parametrize("over,text", [
    ({"start": "s60"}, "args: start"),
    ({"arm": "C"}, "args: arm"),
    ({"seed": 1201}, "args: seed"),  # arm B's seed on arm A
    ({"seed": 1100}, "args: seed"),
    ({"seed": True}, "args: seed"),
    ({"until": 3145728 + 262144}, "--until is only for --no-update"),
    ({"no_update": True}, "needs --until"),
    ({"no_update": True, "until": 3145728}, "not in (S, E]"),
    ({"no_update": True, "until": 4718592 + 8192}, "not in (S, E]"),
    ({"no_update": True, "until": 3145728 + 4096}, "not in (S, E]"),
    ({"no_update": True, "save_loaded_only": True, "until": 3145728 + 8192}, "exclude each other"),
    ({"out": "/tmp/x"}, "--out is only for"),
    ({"save_loaded_only": True, "out": "/tmp/elsewhere"}, "is not under"),
])
def test_each_argument_refusal(root, over, text):
    _expect(RE.preflight(_ns(**over)), text)


def test_approval_refusals(root):
    (root.ecs / "APPROVED").write_text("{not json")
    _expect(RE.preflight(_ns()), "APPROVED: does not parse")
    (root.ecs / "APPROVED").write_text(json.dumps({k: v for k, v in root.approved.items() if k != "runlog_line"}))
    _expect(RE.preflight(_ns()), "missing keys")
    (root.ecs / "APPROVED").write_text(json.dumps({**root.approved, "criteria_sha256": "0" * 64}))
    _expect(RE.preflight(_ns()), "criteria_sha256")
    (root.ecs / "APPROVED").write_text(json.dumps({**root.approved, "code_tree_digest": "0" * 64}))
    _expect(RE.preflight(_ns()), "code_tree_digest")
    (root.ecs / "APPROVED").write_text(json.dumps(root.approved))
    (root.tools / "score_ecs.py").write_bytes(b"changed")
    _expect(RE.preflight(_ns()), "tool_sha256['score_ecs.py']")
    (root.ecs / "criteria.txt").unlink()
    _expect(RE.preflight(_ns()), "criteria.txt missing")


def test_the_run_folder_exists_refusal(root):
    (root.ecs / "runs" / "s42_A_1101").mkdir(parents=True)
    _expect(RE.preflight(_ns()), "exists")
    (root.ecs / "validate" / "v1" / "s42_A_1101").mkdir(parents=True)
    _expect(RE.preflight(_ns(save_loaded_only=True)), "exists")
    assert RE.preflight(_ns(seed=1111)) == []  # another seed's folder is free


def test_input_refusals(root):
    st = root.starts["s42"]
    p = root.f2 / st["run_id"] / "copies" / (st["zip_rel"] + ".copy1")
    p.write_bytes(b"other")
    _expect(RE.preflight(_ns()), "input: sha256")
    p.unlink()
    _expect(RE.preflight(_ns()), "missing")
    rec = root.f2 / st["run_id"] / "copies" / "01_stance" / "policy_probe.npz.copy1"
    rec.write_bytes(b"other")
    _expect(RE.preflight(_ns(start="s42", arm="B", seed=1201)), "policy_probe.npz.copy1")


def test_seal_refusals(root):
    (root.code / "environments" / "shared" / "x.py").write_text("x = 2\n")
    _expect(RE.preflight(_ns()), "changed")
    (root.code / "environments" / "shared" / "x.py").write_text("x = 1\n")
    seal = json.loads((root.code / C.SEAL_NAME).read_text())
    (root.code / C.SEAL_NAME).write_text(json.dumps({**seal, "commit": "0" * 40}))
    _expect(RE.preflight(_ns()), "not the pinned")


def test_stack_pins_and_disk_refusals(root, monkeypatch):
    monkeypatch.setattr(C, "check_stack", lambda: ["torch: running 2.0 != pinned 2.13.0+cpu"])
    _expect(RE.preflight(_ns()), "stack: torch")
    monkeypatch.setattr(C, "check_stack", lambda: [])
    monkeypatch.setattr(C, "check_pins", lambda: ["m5: sha256 x != pinned y"])
    _expect(RE.preflight(_ns()), "pins: m5")
    monkeypatch.setattr(C, "check_pins", lambda: [])
    monkeypatch.setattr(RE, "free_bytes", lambda p: C.DISK_FLOOR_BYTES - 1)
    _expect(RE.preflight(_ns()), "disk:")


def test_main_preflight_only_creates_nothing(root, capsys):
    before = sorted(str(p) for p in root.ecs.rglob("*"))
    assert RE.main(["--start", "s42", "--arm", "A", "--seed", "1101", "--preflight-only"]) == 0
    assert RE.main(["--start", "s42", "--arm", "A", "--seed", "1201", "--preflight-only"]) == 2
    assert RE.main(["--start", "s42", "--arm", "A", "--seed", "1201"]) == 2  # refused: never reaches the run
    assert sorted(str(p) for p in root.ecs.rglob("*")) == before
    out = capsys.readouterr().out
    assert '"preflight": "refused"' in out and '"preflight": "pass"' in out


def test_an_error_inside_preflight_is_a_refusal_not_a_crash(root, monkeypatch, capsys):
    """Review H4: an unexpected exception in preflight exits 2 (refused; nothing made), not 1 (a counted crash)."""
    def boom(ns):
        raise OSError("disk read error")

    monkeypatch.setattr(RE, "preflight", boom)
    before = sorted(str(p) for p in root.ecs.rglob("*"))
    assert RE.main(["--start", "s42", "--arm", "A", "--seed", "1101"]) == 2
    assert "preflight error: OSError('disk read error')" in capsys.readouterr().out
    assert sorted(str(p) for p in root.ecs.rglob("*")) == before


def test_the_real_pins_hold():
    """The real F2 inputs, the M1/M5 pins and the stack, read only (the code copy's seal is checked in
    LANE_H.txt by check-copy)."""
    for start in C.STARTS:
        assert RE.input_problems(start) == []
    assert C.check_pins() == []
    assert C.check_stack() == []
    assert len(C.QUEUE) == 24 and len({C.run_name(*q) for q in C.QUEUE}) == 24
    seeds = [s for q in C.QUEUE for s in C.derived_seeds(q[2])]
    assert len(seeds) == len(set(seeds)) == 120
    assert not set(seeds) & set(range(3042, 3062)) and not {42, 60, 80} & set(seeds)


def test_pickle_of_vecnormalize_roundtrip(toy):
    toy.venv.save(str(toy.tmp / "v.pkl"))
    with open(toy.tmp / "v.pkl", "rb") as fh:
        obj = pickle.load(fh)
    assert RE.stats_problems(toy.venv, obj) == []


def test_module_imports_without_torch_in_sys_modules():
    """Importing run_ecs (and its preflight) pulls in neither torch nor the code copy."""
    import subprocess

    code = (f"import importlib.util,sys; s=importlib.util.spec_from_file_location('r', {str(TOOLS / 'run_ecs.py')!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(sorted(k for k in ('torch','numpy','environments','stable_baselines3') if k in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert out.stdout.strip() == "[]"
