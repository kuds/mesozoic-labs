"""The generic training telemetry (rl/telemetry/; decision D21): exercised on a toy env that knows nothing of the
Triceratops, so the modules' robot-agnostic claim is tested, plus their wiring into train.py."""

from __future__ import annotations

import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
from conftest import RL_DIR  # noqa: F401  (puts rl/ on sys.path)

from telemetry.checks import PeriodicChecks, first_multiple_above, read_checks
from telemetry.health import TrainingHealth, grad_norm, latest_branch, read_health


class Toy(gym.Env):
    """A 2-D point that is rewarded for staying near zero; ends after 7 steps (a fall) or at 10 (time limit). Its
    info carries two reward terms, a "right foot" force named like the species envs' (``r_foot_contact``, not a
    reward), a non-reward value and a termination reason under ``reason_key``."""

    reason_key = "termination"

    observation_space = gym.spaces.Box(-10, 10, (2,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.k, self.x = 0, np.zeros(2, np.float32)
        return self.x.copy(), {}

    def step(self, a):
        self.k += 1
        self.x = np.clip(self.x + 0.1 * np.asarray(a, np.float32), -10, 10)
        r_near, r_const = -float(np.sum(self.x ** 2)), 0.5
        term = self.k == 7 and self.np_random.random() < 0.5
        trunc = self.k >= 10 and not term
        info = {"reward_near": r_near, "reward_const": r_const, "r_foot_contact": 98.1, "speed": 1.0,
                self.reason_key: "floor_contact:head" if term else ""}
        return self.x.copy(), r_near + r_const, term, trunc, info


def _ppo(n_envs=2, n_steps=32, env=Toy):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = VecNormalize(DummyVecEnv([env for _ in range(n_envs)]))
    return PPO("MlpPolicy", venv, n_steps=n_steps, batch_size=32, n_epochs=2, seed=0, verbose=0,
               policy_kwargs={"net_arch": [16]})


def test_health_writes_one_line_per_rollout_with_the_update_that_followed(tmp_path):
    path = tmp_path / "train_health.jsonl"
    model = _ppo()
    model.learn(32 * 2 * 3, callback=TrainingHealth(path))
    rows = read_health(path)
    assert [r["timesteps"] for r in rows] == [64, 128, 192]
    for r in rows:
        assert {"approx_kl", "clip_fraction", "entropy_loss", "explained_variance", "value_loss"} <= set(r["algo"])
        assert r["grad_norm"] is not None and r["action_std"] is not None
        # found by the default prefix "reward_"; neither "speed" nor the force "r_foot_contact" is a reward term
        assert set(r["rewards"]) == {"reward_near", "reward_const"}
        assert r["rewards"]["reward_const"] == pytest.approx(0.5)
        assert r["episodes"] == sum(r["endings"].values())
        assert set(r["endings"]) <= {"floor_contact:head", "time_limit"}
        assert r["obs_var"]["min"] <= r["obs_var"]["median"] <= r["obs_var"]["max"]
    assert sum(r["episodes"] for r in rows) >= 18  # 384 steps of 7- and 10-step episodes over 2 envs


def test_health_endings_by_key_and_unlabelled_terminations(tmp_path):
    """The species envs' "termination_reason" is read by default; a termination without a reason is "unlabelled",
    told apart from a truncation, which SB3 marks ``TimeLimit.truncated``."""

    class Species(Toy):
        reason_key = "termination_reason"

    class Silent(Toy):
        reason_key = "unused"

    for env, want in ((Species, {"floor_contact:head", "time_limit"}), (Silent, {"unlabelled", "time_limit"})):
        path = tmp_path / f"{env.__name__}.jsonl"
        _ppo(env=env).learn(32 * 2 * 3, callback=TrainingHealth(path))
        endings = set().union(*(r["endings"] for r in read_health(path)))
        assert endings == want, (env.__name__, endings)


def test_health_prefixes_are_the_callers(tmp_path):
    path = tmp_path / "h.jsonl"
    _ppo().learn(64, callback=TrainingHealth(path, reward_prefixes=("r_", "reward_")))
    assert "r_foot_contact" in read_health(path)[0]["rewards"]  # what the default protects against


def test_latest_branch_and_torn_line_recovery(tmp_path):
    rows = [{"t": 1}, {"t": 2}, {"t": 3}, {"t": 4}, {"t": 2.5}, {"t": 3}, {"x": 9}, {"t": 5}]
    assert [r["t"] for r in latest_branch(rows, "t")] == [1, 2, 2.5, 3, 5]  # an older file: cut at the first row
    # the review's kill-and-resume: a pair at 1500, the killed session ran on to 2560, the resume starts at 1500
    a = [{"session": "A", "session_start": 0, "t": n} for n in (512, 1024, 1536, 2048, 2560)]
    b = [{"session": "B", "session_start": 1500, "t": n} for n in (2012, 2524, 3036)]
    assert [r["t"] for r in latest_branch(a + b, "t")] == [512, 1024, 2012, 2524, 3036]
    # a second kill and resume from the same pair
    c = [{"session": "C", "session_start": 1500, "t": n} for n in (2012, 2524)]
    assert [(r["session"], r["t"]) for r in latest_branch(a + b + c, "t")] == [("A", 512), ("A", 1024), ("C", 2012),
                                                                              ("C", 2524)]
    path = tmp_path / "h.jsonl"
    path.write_text('{"timesteps": 7}\n{"timesteps": 8, "algo')  # a session killed mid-write
    _ppo().learn(64, callback=TrainingHealth(path))
    rows = read_health(path)
    assert [r["timesteps"] for r in rows] == [7, 64] and rows[1]["session_start"] == 0 and rows[1]["session"]


def test_health_reads_only(tmp_path):
    """The same seed trains to the same parameters with and without the callback."""
    import torch

    a = _ppo()  # each model is built just before it learns: building one reseeds the global generators
    a.learn(128)
    b = _ppo()
    b.learn(128, callback=TrainingHealth(tmp_path / "h.jsonl"))
    for pa, pb in zip(a.policy.parameters(), b.policy.parameters()):
        assert torch.equal(pa, pb)


def test_health_survives_an_unwritable_file(tmp_path):
    said = []
    model = _ppo()
    model.learn(64, callback=TrainingHealth(tmp_path / "missing_dir" / "h.jsonl", say=said.append))
    assert len(said) == 1 and "could not be written" in said[0]


def test_grad_norm_and_torn_lines(tmp_path):
    import torch

    p = torch.nn.Parameter(torch.ones(3))
    assert grad_norm([p]) is None
    p.grad = torch.tensor([3.0, 4.0, 0.0])
    assert grad_norm([p]) == pytest.approx(5.0)
    f = tmp_path / "x.jsonl"
    f.write_text('{"timesteps": 1}\n{"timesteps": 2}\n{"timest')
    assert [r["timesteps"] for r in read_health(f)] == [1, 2] and read_health(tmp_path / "none") == []


def test_checks_run_every_n_steps_with_the_deterministic_policy(tmp_path):
    calls = []

    def run_point(name, spec, seed, act):
        a = act(np.array([0.3, -0.2], np.float32))
        calls.append((name, seed))
        return {"fell": spec["fall"], "action": [float(v) for v in a]}

    path = tmp_path / "checks.jsonl"
    model = _ppo()
    said = []

    def run_or_fail(name, spec, seed, act):
        if name == "broken":
            raise RuntimeError("no such floor")
        return run_point(name, spec, seed, act)

    cb = PeriodicChecks(path, every=100, points={"flat": {"fall": False}, "cliff": {"fall": True}, "broken": {}},
                        run_point=run_or_fail, seeds=(7, 8), say=said.append)
    model.learn(64 * 4, callback=cb)
    rows = read_checks(path)
    assert [r["steps"] for r in rows] == [100, 200] and first_multiple_above(256, 100) == 300
    assert calls[:4] == [("flat", 7), ("flat", 8), ("cliff", 7), ("cliff", 8)]
    r = rows[0]["points"]
    assert [e["fell"] for e in r["cliff"]] == [True, True] and r["flat"][0]["action"] == r["flat"][1]["action"]
    assert all("no such floor" in e["error"] for e in r["broken"])  # a failing point never stops training
    assert sum("WARNING" in s for s in said) == 2 and rows[0]["session_start"] == 0
    from telemetry import health_dashboard as hd

    s = hd.checks_series(rows + [{"steps": 300}])  # a row without points (a foreign file) is skipped
    assert s["points"]["broken"]["failed_episodes"] == 4 and s["points"]["broken"]["fall_share"] == [None, None]
    with pytest.raises(ValueError):
        PeriodicChecks(path, every=0, points={}, run_point=run_point)


def test_dashboard_draws_from_both_files(tmp_path):
    from telemetry import health_dashboard as hd

    model = _ppo()
    model.learn(64 * 4, callback=[TrainingHealth(tmp_path / "h.jsonl"),
                                  PeriodicChecks(tmp_path / "c.jsonl", 64, {"a": 1},
                                                 lambda n, s, seed, act: {"fell": False, "forward_speed_mps": 0.1},
                                                 say=lambda s: None)])
    assert hd.main(["--label", "toy", "--health", str(tmp_path / "h.jsonl"), "--checks", str(tmp_path / "c.jsonl"),
                    "--out", str(tmp_path / "out")]) == 0
    files = json.loads((tmp_path / "out" / "training_health_toy.json").read_text())["charts"]
    assert files == ["training_health_toy.png", "training_checks_toy.png"]
    assert all((tmp_path / "out" / f).stat().st_size < 500 * 1024 for f in files)


def test_train_records_health_and_checks_are_not_kept_settings(tmp_path, monkeypatch):
    """train.py writes train_health.jsonl. The checks are monitoring, not among the settings a run keeps, so a
    resume may turn them on or off and older records compare equal. Unknown points, a model outside sim/generated
    and a TRIKE_ENV_DIR export are refused before anything starts."""
    import train

    args = train.parse_args(["--out", str(tmp_path / "r"), "--total-steps", "1024"])
    cfg = train.env_config(args)
    args2 = train.parse_args(["--out", str(tmp_path / "r"), "--total-steps", "1024", "--checks-every", "1000000"])
    assert train.run_settings(args2, cfg) == train.run_settings(args, cfg)
    with pytest.raises(SystemExit):
        train.check_points("step_3cm,moon")
    import shutil

    outside = tmp_path / "trike60_C_closed.xml"
    shutil.copy(cfg.model_path, outside)
    with pytest.raises(SystemExit, match="sim/generated"):
        train.make_checks(args2, cfg.replace(model_path=str(outside)), tmp_path)
    # through main(): refused before any file is written (the review's N1)
    for extra, env in ((["--checks", "step_3cm,moon"], None), ([], str(tmp_path)),
                       (["--model", str(outside)], None)):
        run = tmp_path / f"refused{len(extra)}{bool(env)}"
        if env:
            monkeypatch.setenv("TRIKE_ENV_DIR", env)
        else:
            monkeypatch.delenv("TRIKE_ENV_DIR", raising=False)
        assert train.main(["--out", str(run), "--total-steps", "1024", "--checks-every", "512", *extra]) == 2
        assert not run.exists()
    monkeypatch.setenv("TRIKE_ENV_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="TRIKE_ENV_DIR"):
        train.make_checks(args2, cfg, tmp_path)
    out = tmp_path / "run"
    p = subprocess.run([sys.executable, str(RL_DIR / "train.py"), "--out", str(out), "--total-steps", "1024",
                        "--n-envs", "2", "--n-steps", "256", "--batch-size", "512", "--checkpoint-every", "1024",
                        "--log-every", "1024", "--eval-every", "0", "--seed", "3"],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = read_health(out / "train_health.jsonl")
    assert [r["timesteps"] for r in rows] == [512, 1024] and "r_track" in rows[0]["rewards"]
    assert "reward_total" in rows[0]["rewards"]
    assert not (out / "checks_during_training.jsonl").exists()
