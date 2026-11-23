"""The generic telemetry (rl/telemetry/; decision D21). Training health and periodic checks run on a toy env that
knows nothing of the Triceratops; the loads recorder on small MuJoCo models built here (one and two hinges on a body,
a ball joint) as well as the Triceratops; heat and battery on plain numbers; plus their wiring into train.py and the
demo course."""

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
    # a pair exactly at a rollout's end (2048): the killed session's 2048 row carries an update the pair lacks
    e = [{"session": "E", "session_start": 0, "t": n} for n in (512, 1024, 1536, 2048, 2560)]
    f = [{"session": "F", "session_start": 2048, "t": n} for n in (2560, 3072)]
    assert [(r["session"], r["t"]) for r in latest_branch(e + f, "t")] == [("E", 512), ("E", 1024), ("E", 1536),
                                                                          ("F", 2560), ("F", 3072)]
    # a completed run extended from its last pair keeps its last row
    g = [{"session": "G", "session_start": 2048, "t": n} for n in (2560,)]
    assert [r["t"] for r in latest_branch(e[:4] + g, "t")] == [512, 1024, 1536, 2048, 2560]
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
        run = tmp_path / f"refused_{len(list((tmp_path).glob('*')))}"
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


def _steer_infos(segments, dt=0.02, fall_at=None):
    """Synthetic step infos: segments of (seconds, desired heading, desired speed, actual heading fn(t), forward
    speed, yaw rate, command v_x, command yaw rate, turn_in_place); the base moves at the forward speed along the
    actual heading; ``fall_at`` ends the episode there with a fall."""
    import math

    infos, t, eid, x, y = [], 0.0, 0, 0.0, 0.0
    for seconds, desired, speed, heading, v, yr, cv, cy, tip in segments:
        for k in range(int(round(seconds / dt))):
            t += dt
            h = heading(k * dt)
            x, y = x + v * math.cos(h) * dt, y + v * math.sin(h) * dt
            done = fall_at is not None and t >= fall_at
            infos.append({"time_s": t, "command_event_id": eid, "desired_heading": desired, "desired_speed": speed,
                          "actual_heading": h, "heading_active": speed > 0, "turn_in_place": tip,
                          "forward_speed": v, "lateral_speed": 0.0, "yaw_rate": yr, "command_v_x": cv,
                          "command_yaw_rate": cy, "tracking_error_v": abs(v - cv), "tracking_error_yaw": abs(yr - cy),
                          "base_x": x, "base_y": y, "termination": "tilt" if done else ""})
            if done:
                return infos
        eid += 1
    return infos


def test_steering_measures_on_a_known_episode():
    import math

    from telemetry.steering import episode_measures, summarise

    tol = dict(velocity_tolerance=0.05, yaw_rate_tolerance=0.1, heading_tolerance=math.radians(10))
    infos = _steer_infos([
        (4.0, 0.0, 0.25, lambda s: 0.0, 0.25, 0.0, 0.25, 0.0, False),                     # straight, perfect
        (5.0, math.pi / 2, 0.25, lambda s: min(math.pi / 2, 0.4 * s), 0.2, 0.4, 0.2, 0.4, False),  # 90 deg turn
        (5.0, math.pi, 0.0, lambda s: min(math.pi, math.pi / 2 + 0.6 * s), 0.0, 0.6, 0.0, 0.6, True),  # on the spot
        (3.0, math.pi, 0.0, lambda s: math.pi, 0.0, 0.0, 0.0, 0.0, False),                 # stand
    ])
    m = episode_measures(infos, **tol)
    ev = {r["event_id"]: r for r in m["events"]}
    assert ev[1]["is_turn"] and ev[1]["initial_error_rad"] == pytest.approx(math.pi / 2)
    assert ev[1]["settle_time_s"] == pytest.approx((math.pi / 2 - math.radians(10)) / 0.4, abs=0.03)
    assert ev[2]["turn_in_place"] and ev[2]["is_turn"] and ev[2]["heading_error_at_settle_rad"] < math.radians(10)
    assert m["turning"]["radius_m_median"] == pytest.approx(0.2 / 0.4)
    assert m["turn_in_place"]["achieved_over_commanded"]["median"] == pytest.approx(1.0)
    assert m["turn_in_place"]["net_drift_mps"]["median"] == pytest.approx(0.0)
    assert m["stops"]["steps"] > 0 and m["stops"]["speed_mps"]["median"] == 0.0
    s = summarise([m, m])
    assert s["episodes"] == 2 and s["falls"] == 0 and s["turns"] == 4 and s["turns_settled_share"] == 1.0


def test_steering_measures_do_not_flatter():
    """The review's cases: a robot that never turns settles no turn, even when a request returns to its heading;
    a turn cut short by a fall counts as failed; the heading error wraps at +-pi; sway is not drift."""
    import math

    from telemetry.steering import episode_measures

    tol = dict(velocity_tolerance=0.05, yaw_rate_tolerance=0.1, heading_tolerance=math.radians(10))
    still = _steer_infos([(4.0, 0.0, 0.25, lambda s: 0.0, 0.0, 0.0, 0.25, 0.0, False),
                          (4.0, math.pi / 2, 0.25, lambda s: 0.0, 0.0, 0.0, 0.06, 0.4, False),
                          (4.0, 0.0, 0.25, lambda s: 0.0, 0.0, 0.0, 0.25, 0.0, False)])
    m = episode_measures(still, **tol)
    turns = [r for r in m["events"] if r["is_turn"]]
    assert len(turns) == 1 and turns[0]["settle_time_s"] is None  # the return to 0 is not a turn, not a success
    assert m["events_summary"]["settled_share_of_turns"] == 0.0
    fall = _steer_infos([(4.0, 0.0, 0.25, lambda s: 0.0, 0.25, 0.0, 0.25, 0.0, False),
                         (4.0, math.pi / 2, 0.25, lambda s: 0.3 * s, 0.2, 0.3, 0.2, 0.4, False)], fall_at=5.0)
    m = episode_measures(fall, **tol)
    t = [r for r in m["events"] if r["is_turn"]]
    assert len(t) == 1 and t[0]["fell_during"] and t[0]["settle_time_s"] is None
    assert m["events_summary"]["turns_failed_by_fall"] == 1 and m["events_summary"]["settled_share_of_turns"] == 0.0
    wrap = _steer_infos([(4.0, math.pi - 0.02, 0.25, lambda s: -math.pi + 0.02, 0.25, 0.0, 0.25, 0.0, False)])
    r = episode_measures(wrap, **tol)["events"][0]
    assert r["initial_error_rad"] == pytest.approx(0.04, abs=1e-9) and not r["is_turn"]
    sway = [dict(i, base_y=0.03 * math.sin(2 * math.pi * 1.4 * i["time_s"]))
            for i in _steer_infos([(6.0, math.pi, 0.0, lambda s: 0.5 * s, 0.0, 0.5, 0.0, 0.5, True)])]
    assert episode_measures(sway, **tol)["turn_in_place"]["net_drift_mps"]["median"] < 0.01


# ---------------------------------------------------------------- heat, battery, loads (step 3)
def test_thermal_calibration_reproduces_the_makers_figure_and_cools():
    import math

    from telemetry.thermal import DriveThermal, demo_temperatures

    from telemetry.thermal import TIME_CONSTANTS_S, tc_key

    d = DriveThermal("x4", kt_nm_per_a=1.8, r_phase_ohm=0.2, thermal_eq_nm=7.5)
    for tc in TIME_CONSTANTS_S + (2400.0,):
        n = d.calibrate(tc)
        assert n.run([(90 * 60.0, 7.5 ** 2)])[-1][1] == pytest.approx(100.0, abs=0.05)
        assert n.time_constant_s == pytest.approx(tc)
        ss = n.steady_state_c(7.5 ** 2)
        assert ss is not None and ss >= 100.0 - 0.1  # held longer than 90 min, it ends at or above 100 C
        assert n.run([(10 * tc, 7.5 ** 2)])[-1][1] == pytest.approx(ss, abs=0.2)
        assert n.run([(5 * tc, 0.0)], start_c=80.0)[-1][1] == pytest.approx(25.0 + 55.0 * math.exp(-5), abs=0.1)
    out = demo_temperatures({"x4": d}, {"x4": [(1200.0, 7.5 ** 2), (600.0, 0.0)]}, limit_c=60.0)
    assert set(out["x4"]) == {"tau_240s", "tau_480s", "tau_900s"} and tc_key(610.5) == "tau_610.5s"
    r = out["x4"]["tau_480s"]
    assert r["peak_c"] > r["end_c"] > 25.0 and r["minutes_to_limit"] is not None and r["cooldown_minutes"] > 0
    assert out["x4"]["tau_240s"]["peak_c"] > out["x4"]["tau_900s"]["peak_c"]  # a short time constant heats faster
    # continuous: the steady state at the time-weighted mean square (2/3 of 7.5^2 here)
    assert r["continuous_c"] == pytest.approx(d.calibrate(480.0).steady_state_c(7.5 ** 2 * 2 / 3))
    assert 100.0 <= r["steady_at_thermal_eq_c"] < 102.0  # 90 min is about steady at an 8 min time constant
    with pytest.raises(ValueError):
        demo_temperatures({"x4": d}, {"x4": [(60.0, 1.0)]}, time_constants_s=(300.0, 300.0))


def test_thermal_default_range_brackets_the_makers_overload_figures():
    """The maker's overload times (thermal.py's docstring) fall within what the default time constants predict."""
    from telemetry.thermal import TIME_CONSTANTS_S, DriveThermal

    def minutes_to_100(d, tc, tau):
        return next(t for t, T in d.calibrate(tc).run([(3600.0, tau ** 2)]) if T >= 100.0) / 60.0

    x4, x6 = DriveThermal("x4", 1.0, 0.2, 7.5), DriveThermal("x6", 1.0, 0.2, 12.0)
    lo, hi = min(TIME_CONSTANTS_S), max(TIME_CONSTANTS_S)
    assert minutes_to_100(x4, lo, 10.5) <= 5.4 and minutes_to_100(x4, hi, 10.5) >= 12.4 * 0.95  # 5.4-12.4 min
    assert minutes_to_100(x4, lo, 21.0) <= 44 / 60 <= minutes_to_100(x4, hi, 21.0)  # 44 s
    assert minutes_to_100(x6, lo, 20.0) <= 4.15 <= minutes_to_100(x6, hi, 20.0)  # 4 min 9 s


def test_battery_arithmetic():
    from telemetry.power import demo_battery

    b = demo_battery([{"seconds": 90, "mean_draw_w": 40.0, "peak_draw_w": 400.0, "peak_surplus_w": 40.0,
                       "surplus_share": 0.1, "mean_sq_draw_w2": 2000.0, "brake_event_j_max": 1.5},
                      {"seconds": 30, "mean_draw_w": 8.0, "peak_draw_w": 20.0, "peak_surplus_w": 0.0,
                       "surplus_share": 0.0, "mean_sq_draw_w2": 100.0, "brake_event_j_max": 0.0}],
                     pack_wh=100.0, usable_share=0.8, v_nominal=48.0, v_low=40.0, hotel_w=30.0,
                     hotel_range_w=(25.0, 40.0))
    assert b["mean_drive_draw_w"] == pytest.approx(32.0) and b["mean_total_w"] == pytest.approx(62.0)
    assert b["runtime_minutes"] == pytest.approx(60 * 80 / 62.0)
    assert b["runtime_minutes_by_hotel_w"] == pytest.approx({"25": 60 * 80 / 57.0, "40": 60 * 80 / 72.0})
    assert b["current_peak_a"]["low_v"] == pytest.approx(430.0 / 40.0)
    assert b["braking_surplus_share"] == pytest.approx(0.075)
    assert b["braking_peak_beyond_hotel_w"] == pytest.approx(10.0) and b["braking_event_j_max"] == 1.5
    ms = (90 * (2000 + 60 * 40 + 900) + 30 * (100 + 60 * 8 + 900)) / 120  # E[(p + h)^2]
    assert b["current_rms_a"]["nominal_v"] == pytest.approx(ms ** 0.5 / 48.0)
    for bad in ([], [{"seconds": 0, "mean_draw_w": 1, "peak_draw_w": 1, "peak_surplus_w": 0, "surplus_share": 0}]):
        with pytest.raises(ValueError):
            demo_battery(bad, pack_wh=1, usable_share=1, v_nominal=1, v_low=1, hotel_w=0)


def test_loads_balance_the_weight_standing():
    from trike_env import CAPSULE_MODEL, CORNERS, TrikeEnv, TrikeEnvConfig

    from telemetry.loads import LoadRecorder

    env = TrikeEnv(TrikeEnvConfig(model_path=str(CAPSULE_MODEL), domain_randomization=False,
                                  observation_noise=False))
    m = env.model
    joints = [m.joint(int(m.actuator(int(i)).trnid[0])).name for i in env._leg_act]
    rec = LoadRecorder(m, joints, [f"{c}_sole" for c in CORNERS])
    env.reset(seed=0)
    env.substep_hook = lambda e: rec(e.model, e.data)
    for _ in range(50):
        env.step(np.zeros(12))
    s = rec.summary(env.nominal_weight)
    feet = sum(v["normal_n"]["median"] for v in s["feet"].values())
    assert feet == pytest.approx(env.nominal_weight, rel=0.01)
    for c in CORNERS:  # the top joint carries its foot's load less the leg's own weight
        top = [j for j in joints if j.startswith(c)][0]
        leg = float(m.body_subtreemass[m.jnt_bodyid[m.joint(top).id]]) * 9.81
        assert s["joints"][top]["force_n"]["median"] == pytest.approx(
            s["feet"][f"{c}_sole"]["normal_n"]["median"] - leg, abs=1.0)


LINK_XML = """<mujoco><option gravity="0 0 -9.81" timestep="0.001"/><worldbody>
<body name="base" pos="0 0 1"><geom type="sphere" size="0.02" pos="0.3 0 0" mass="10"/>
<body name="link">{joints}<geom type="sphere" size="0.01" pos="0 {L} 0" mass="2"/></body></body>
</worldbody>{actuators}</mujoco>"""
GIMBAL_XML = """<mujoco><option gravity="0 0 -9.81" timestep="0.001"/><worldbody>
<body name="base" pos="0 0 1"><geom type="sphere" size="0.02" pos="0.3 0 0" mass="10"/>
<body name="yoke"><joint name="pitch" type="hinge" axis="0 1 0"/><inertial pos="0 0 0" mass="1e-4"
diaginertia="1e-8 1e-8 1e-8"/>
<body name="thigh"><joint name="roll" type="hinge" axis="1 0 0" damping="5"/>
<geom type="sphere" size="0.01" pos="0 0.05 0" mass="2"/></body></body></body>
</worldbody><actuator><position joint="roll" kp="2000"/></actuator></mujoco>"""


def _link(joints, actuators="", L=0.05):
    """A 2 kg link hinged under a heavy jointless base whose centre of mass (the reference cfrc_int is given at) is
    away from the anchor, so an error in moving the moment to the anchor shows."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(LINK_XML.format(joints=joints, actuators=actuators, L=L))
    return m, mujoco.MjData(m)


def test_loads_single_hinge_bending_is_analytic():
    """A 2 kg mass 0.05 m out along a hinge's own axis: the hinge carries no torque, the bearing 19.62 N and a
    bending moment m g L = 0.981 N m. The reference point is the base's subtree centre of mass, 0.25 m away, so a
    sign error in the moment shift shows (it reads 0.654 N m)."""
    import mujoco

    from telemetry.loads import LoadRecorder

    m, d = _link('<joint name="pitch" type="hinge" axis="0 1 0"/>')
    mujoco.mj_forward(m, d)
    assert np.linalg.norm(d.subtree_com[m.body("base").id] - d.xanchor[0]) > 0.2
    rec = LoadRecorder(m, ["pitch"])
    rec(m, d)
    assert rec.force[0][0] == pytest.approx(19.62, abs=1e-3) and rec.moment[0][0] == pytest.approx(0.981, abs=1e-4)
    # the same mass on a hinge across it (axis x): m g L is that hinge's torque, not bending
    m, d = _link('<joint name="roll" type="hinge" axis="1 0 0" damping="5"/>',
                 '<actuator><position joint="roll" kp="2000"/></actuator>')
    for _ in range(3000):
        mujoco.mj_step(m, d)
    rec = LoadRecorder(m, ["roll"])
    rec(m, d)
    assert rec.moment[0][0] < 0.02


def test_loads_two_hinges_on_one_body_are_a_serial_gimbal():
    """Pitch (y) and roll (x) hinges on one body, the roll held by a stiff servo against m g L. MuJoCo chains them
    in order, a serial gimbal: the pitch bearing carries the roll servo's torque as bending; the roll bearing carries
    none here. The same hip built with an explicit near-massless yoke body gives the same figures."""
    import mujoco

    from telemetry.loads import LoadRecorder

    m, d = _link('<joint name="pitch" type="hinge" axis="0 1 0"/><joint name="roll" type="hinge" axis="1 0 0" '
                 'damping="5"/>', '<actuator><position joint="roll" kp="2000"/></actuator>')
    g = mujoco.MjModel.from_xml_string(GIMBAL_XML)
    gd = mujoco.MjData(g)
    for _ in range(3000):
        mujoco.mj_step(m, d)
        mujoco.mj_step(g, gd)
    rec, grec = LoadRecorder(m, ["pitch", "roll"]), LoadRecorder(g, ["pitch", "roll"])
    rec(m, d)
    grec(g, gd)
    assert abs(d.actuator_force[0]) == pytest.approx(0.981, rel=0.02)  # the servo holds m g L
    assert rec.force[0][0] == pytest.approx(19.62, rel=0.01) and rec.force[0][1] == rec.force[0][0]
    assert rec.moment[0][0] == pytest.approx(0.981, rel=0.02) and rec.moment[0][1] < 0.02
    assert rec.moment[0] == pytest.approx(grec.moment[0], abs=0.01)  # the explicit gimbal agrees
    assert rec.force[0] == pytest.approx(grec.force[0], rel=1e-3)
    s = rec.summary(19.62)
    assert s["joints"]["pitch"]["shares_body_with"] == ["roll"]
    ball_m, ball_d = _link('<joint name="ball" type="ball"/>')
    mujoco.mj_forward(ball_m, ball_d)
    rec = LoadRecorder(ball_m, ["ball"])
    rec(ball_m, ball_d)
    assert rec.summary(19.62)["joints"]["ball"]["bending_nm"] is None
    free_m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><freejoint name="base"/>'
                                            '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match="free joint"):
        LoadRecorder(free_m, ["base"])


def test_loads_hook_does_not_change_the_simulation():
    from trike_env import CAPSULE_MODEL, CORNERS, TrikeEnv, TrikeEnvConfig

    from telemetry.loads import LoadRecorder

    def run(with_hook):
        env = TrikeEnv(TrikeEnvConfig(model_path=str(CAPSULE_MODEL)))
        env.reset(seed=3)
        m = env.model
        if with_hook:
            joints = [m.joint(int(m.actuator(int(i)).trnid[0])).name for i in env._leg_act]
            rec = LoadRecorder(m, joints, [f"{c}_sole" for c in CORNERS])
            env.substep_hook = lambda e: rec(e.model, e.data)
        rng, obs_all = np.random.default_rng(3), []
        for _ in range(100):
            obs, *_ = env.step(rng.uniform(-0.5, 0.5, 12))
            obs_all.append(obs)
        return np.array(obs_all), env.data.qpos.copy()

    (o1, q1), (o2, q2) = run(False), run(True)
    assert np.array_equal(o1, o2) and np.array_equal(q1, q2)


def test_demo_loop_is_validated():
    import demo_course

    assert demo_course.build_sequence([("walk", 0.0), ("stand", 30.0)], 1.5) == [("stand", 30.0)] * 3
    assert demo_course.build_sequence([("walk", 1e-10), ("stand", 30.0)], 1.0) == [("stand", 30.0)] * 2
    assert demo_course.build_sequence([("walk", 60.0), ("stand", 30.0)], 2.0) == [("walk", 60.0), ("stand", 30.0),
                                                                                   ("walk", 30.0)]
    for loop in ([("walk", 0.0), ("stand", 0.0)], [("walk", -5.0), ("stand", 30.0)], [("walk", 1e-10)]):
        with pytest.raises(ValueError):
            demo_course.build_sequence(loop, 1.0)


def test_demo_course_runs_end_to_end(tmp_path):
    import demo_course

    make, walker = demo_course.trot_factory()
    res = demo_course.run(make, walker, [("walk_0.25", 60.0), ("stand", 30.0)], None, 700_000, segment_s=4.0)
    assert set(res["segments"]) == set(demo_course.SEGMENTS) and not any(s["fell"] for s in res["segments"].values())
    assert res["valid"] and res["minutes_is_one_pack"]
    b = res["battery"]
    # one pack on a whole loop's mix; the demo ends part-way through a loop, so its own mix differs slightly
    assert res["minutes"] == pytest.approx(b["runtime_minutes"], rel=2e-3)
    assert b["runtime_minutes"] > 0 and res["segments"]["stand"]["mean_draw_w"] < \
        res["segments"]["walk_0.25"]["mean_draw_w"]
    assert b["current_peak_all_segments_a"]["low_v"] >= b["current_peak_a"]["low_v"]
    assert b["current_rms_a"]["low_v"] >= b["current_mean_a"]["low_v"]
    assert set(next(iter(res["temperatures"].values()))) == {"tau_240s", "tau_480s", "tau_900s"}
    h = res["loads_headline"]
    assert {"joint_force_p99_bw", "joint_force_max_bw", "bending_p99_nm", "bending_max_nm", "foot_max_bw"} <= set(h)
    worst = max(j["force_bw"]["p99"] for s in res["segments"].values() for j in s["loads"]["joints"].values())
    assert h["joint_force_p99_bw"]["value"] == worst
    assert res["segments"]["walk_0.25"]["drive_energy_per_m_j"] > 0 and \
        res["segments"]["stand"]["drive_energy_per_m_j"] is None
    assert demo_course.chart(res, tmp_path, "test") == "demo_temperatures.png"


def test_demo_course_marks_a_fall_invalid(tmp_path):
    """A walker that falls: the segment says so and the demo is INVALID; one that falls before the window gives no
    demo at all rather than a crash."""
    import demo_course

    make, _ = demo_course.trot_factory()

    def falling(command):
        env, _ = make(command)
        return env, lambda obs: np.full(12, 1.0)  # hard against the limits: down within a second or two

    s = demo_course.measure_segment(falling, 0.25, 700_000, seconds=6.0)
    assert s["fell"]
    res = demo_course.run(falling, "falls", [("walk_0.25", 60.0), ("stand", 30.0)], 2.0, 700_000, segment_s=6.0)
    assert not res["valid"] and res["fallen_segments"]
    # an episode that ends inside the first 2 s (here by its time limit) has nothing measured: no demo, no crash
    short = demo_course.run(make, "short", [("walk_0.25", 60.0), ("stand", 30.0)], 2.0, 700_000, segment_s=1.0)
    assert not any(x["measured"] for x in short["segments"].values())
    assert short["battery"] is None and short["temperatures"] is None and short["loads_headline"] == {}
