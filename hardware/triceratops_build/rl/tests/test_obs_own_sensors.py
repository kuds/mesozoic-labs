"""The robot's-own-sensors observation (decision D46): ``obs_foot_loads`` beside ``obs_ankle_angles`` and
``obs_privileged_lin_vel``. Simulation only. Run by hand:
    python -m pytest hardware/triceratops_build/rl/tests/test_obs_own_sensors.py

- With every flag at its default the environment is byte-identical to the code before ``obs_foot_loads``: the
  digests below, of seeded rollouts' observations, rewards, endings and complete infos, were recorded from
  aa150f89's ``trike_env.py`` (the code without the flag; mujoco 3.10.0, this container, Linux x86-64). On the
  default and D22 cases f84a6919's ``trike_env.py``, the commit the four fore128 runs trained at, gave the same
  observations, rewards and endings bit for bit and the same infos except the ``r_heat_peak`` key added since
  (0 there), compared step by step when the digests were recorded.
- With all three off the observation loses exactly 11 channels (4 foot loads, 4 ankle angles, 3 body velocity) and
  nothing else changes: physics, rewards, infos and every other channel, noise included, are bit for bit those of
  the full observation (the dropped channels' noise is still drawn, so the random stream does not shift).
- Nothing outside the observation reads these slots: rewards, endings and the analysis tools read the simulator's
  state (``data.sensordata``, ``data.qpos``), which the side-by-side test covers for every reward term and info key.
  The model-based trot (``walk/walk60.py``) reads ``foot_load_over_weight`` from the observation and needs the
  default.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from trike_env import TrikeEnv, TrikeEnvConfig

RL_DIR = Path(__file__).resolve().parents[1]
GEN = RL_DIR.parent / "sim" / "generated"
CONFIGS = RL_DIR / "configs"
DEFAULT_MODEL = GEN / "trike60_C_closed.xml"
FORE128 = GEN / "trike60_C_closed_capsule_fore128.xml"
OWN = {"obs_ankle_angles": False, "obs_privileged_lin_vel": False, "obs_foot_loads": False}
DROPPED = {"ankle_pitch_pos": 4, "base_lin_vel_body_PRIVILEGED": 3, "foot_load_over_weight": 4}
FLAG_SLOT = {"obs_ankle_angles": "ankle_pitch_pos", "obs_privileged_lin_vel": "base_lin_vel_body_PRIVILEGED",
             "obs_foot_loads": "foot_load_over_weight"}
OWN_PRESET = "run3_terrain_ownsensors"
D22_RECORD = RL_DIR / "analysis" / "results" / "d22" / "run_record_fore128-s0.json"

# name: (model, preset file in configs/ or None, first episode seed (then +1 at each reset), control steps in all)
CASES = {
    "default": (DEFAULT_MODEL, None, 11, 400),
    "d22_fore128": (FORE128, "run3_terrain", 11, 600),  # seed 11: flat floor, a push at 3.69 s
    "steer1_dev_heat1": (FORE128, "steer1_dev_heat1", 14, 600),  # seed 14: a push at 3.75 s
}
# recorded with mujoco 3.10.0 from aa150f89's trike_env.py (the code without obs_foot_loads)
DIGESTS = {
    "default": "76aa0863ce1ec5c9ff5363bed9c0d15fbb287ef02c13d1612e5b4dc665447a8c",
    "d22_fore128": "12910bbd7deea3deed42d1db354918678332d9371d0c0424899d9bd33a9f0a46",
    "steer1_dev_heat1": "c26ef46422c10400f936b57859faf0bbd8f9ab9331b77f97bf18cf95f341fc79",
}


def config_for(cfg_cls, model: Path, preset: str | None, **changes):
    """As train.py builds it: the model's defaults, then the preset file's fields, then ``changes``."""
    d = cfg_cls(model_path=str(model)).to_dict()
    if preset:
        d.update(json.loads((CONFIGS / f"{preset}.json").read_text()))
    d.update(changes)
    return cfg_cls.from_dict(d)


def canon(info: dict) -> bytes:
    """An info dict as bytes, every key and the exact value and type of every entry."""
    def one(v):
        if isinstance(v, np.ndarray):
            return ("ndarray", str(v.dtype), v.shape, v.tobytes().hex())
        if isinstance(v, np.generic):
            return (type(v).__name__, repr(v.item()))
        return (type(v).__name__, repr(v))
    return repr(sorted((k, one(v)) for k, v in info.items())).encode()


def action(rng, k: int) -> np.ndarray:
    """At control step ``k`` of an episode: small random actions, then standing (the D22 case's push lands at
    3.69 s), then hard alternating actions that end the episode, so the termination and reset path is covered."""
    if k < 30:
        return rng.uniform(-0.3, 0.3, 12)
    if k < 220:
        return np.zeros(12)
    return np.full(12, 1.0 if (k // 10) % 2 else -1.0)


def rollout(env, seed: int, steps: int):
    """Yields ("reset", obs, info) and ("step", obs, reward, terminated, truncated, info) along a seeded rollout of
    ``steps`` control steps, resetting with the next seed (seed + 1, ...) whenever an episode ends."""
    obs, info = env.reset(seed=seed)
    yield ("reset", obs, info)
    rng, k_ep = np.random.default_rng(11), 0
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(action(rng, k_ep))
        k_ep += 1
        yield ("step", obs, r, term, trunc, info)
        if term or trunc:
            seed, k_ep = seed + 1, 0
            obs, info = env.reset(seed=seed)
            yield ("reset", obs, info)


def case_digest(env_cls, cfg_cls, name: str, **changes):
    """(sha256 of the rollout's observations, rewards, endings and infos, coverage counts) for a case of CASES."""
    import hashlib

    model, preset, seed, steps = CASES[name]
    env = env_cls(config_for(cfg_cls, model, preset, **changes))
    h = hashlib.sha256()
    cov = {"resets": 0, "push_steps": 0, "terminations": 0}
    for ev in rollout(env, seed, steps):
        if ev[0] == "reset":
            h.update(b"R" + ev[1].tobytes() + canon(ev[2]))
            cov["resets"] += 1
        else:
            _, obs, r, term, trunc, info = ev
            h.update(b"S" + obs.tobytes() + repr((r, term, trunc)).encode() + canon(info))
            cov["push_steps"] += int(bool(info.get("push_active")))
            cov["terminations"] += int(term)
    env.close()
    return h.hexdigest(), cov


# ======================================================================================== the default is unchanged
@pytest.mark.parametrize("name", sorted(CASES))
def test_default_flags_leave_the_env_byte_identical(name):
    """The default env, the D22 fore128 setup (run 3's terrain preset on the 128 mm model) and the steering preset,
    against the digests recorded from the code before obs_foot_loads. Skipped for another mujoco version; on another
    platform re-record from aa150f89's trike_env.py before trusting a failure."""
    if mujoco.__version__ != "3.10.0":
        pytest.skip("digests recorded with mujoco 3.10.0")
    digest, cov = case_digest(TrikeEnv, TrikeEnvConfig, name)
    assert cov["terminations"] >= 1 and cov["resets"] >= 2  # the ending and reset path is in the digest
    if name != "default":
        assert cov["push_steps"] > 0  # so is a push
    assert digest == DIGESTS[name]
    assert case_digest(TrikeEnv, TrikeEnvConfig, name, obs_foot_loads=True)[0] == digest


def test_foot_loads_default_on_and_old_records_load_with_it():
    """A run record written before the flag (the D22 runs') rebuilds the 68-float observation."""
    assert TrikeEnvConfig().obs_foot_loads is True
    ec = json.loads(D22_RECORD.read_text())["settings"]["env_config"]
    assert "obs_foot_loads" not in ec
    cfg = TrikeEnvConfig.from_dict({**ec, "model_path": str(GEN / Path(ec["model_path"]).name)})
    assert (cfg.obs_foot_loads, cfg.obs_ankle_angles, cfg.obs_privileged_lin_vel) == (True, True, True)
    e = TrikeEnv(cfg)
    assert e.observation_space.shape == (68,)
    assert [n for n, _ in e.obs_layout][7] == "foot_load_over_weight"  # where it always was
    e.close()


# ======================================================================================== the three flags off
@pytest.mark.parametrize("name", ["d22_fore128", "steer1_dev_heat1"])
def test_own_sensors_drop_exactly_the_eleven_channels_and_change_nothing_else(name):
    """Side by side, same seeds and actions: the own-sensors env's observation is the full one without the 11
    channels, bit for bit, and every reward, ending and info is identical (randomisation, noise, terrain, pushes on;
    the D22 fore128 setup and the steering preset)."""
    model, preset, seed, steps = CASES[name]
    full = TrikeEnv(config_for(TrikeEnvConfig, model, preset))
    own = TrikeEnv(config_for(TrikeEnvConfig, model, preset, **OWN))
    n_full = 71 if preset.startswith("steer1_") else 68
    assert full.observation_space.shape == (n_full,) and own.observation_space.shape == (n_full - 11,)
    assert sum(DROPPED.values()) == 11
    assert own.obs_layout == tuple((n, w) for n, w in full.obs_layout if n not in DROPPED)
    assert {n: w for n, w in full.obs_layout if n in DROPPED} == DROPPED
    keep = [n for n, _ in own.obs_layout]
    pushes = 0
    for a, b in zip(rollout(full, seed, steps), rollout(own, seed, steps), strict=True):
        assert a[0] == b[0]
        assert own.observation_space.shape == b[1].shape
        expected = np.concatenate([full.obs_slots(a[1])[n] for n in keep])
        assert expected.dtype == b[1].dtype == np.float32 and expected.tobytes() == b[1].tobytes()
        assert canon(a[-1]) == canon(b[-1])  # every reward term, the measurements, the command, the ending
        if a[0] == "step":
            assert repr(a[2:5]) == repr(b[2:5])
            pushes += int(bool(a[-1].get("push_active")))
    assert pushes > 0
    full.close()
    own.close()


@pytest.mark.parametrize("flag", sorted(FLAG_SLOT))
def test_each_flag_removes_only_its_own_slot(flag):
    full = TrikeEnv(config_for(TrikeEnvConfig, FORE128, "run3_terrain"))
    one = TrikeEnv(config_for(TrikeEnvConfig, FORE128, "run3_terrain", **{flag: False}))
    slot = FLAG_SLOT[flag]
    assert one.obs_layout == tuple((n, w) for n, w in full.obs_layout if n != slot)
    assert one.observation_space.shape == (68 - DROPPED[slot],)
    o_full, _ = full.reset(seed=5)
    o_one, _ = one.reset(seed=5)
    keep = [n for n, _ in one.obs_layout]
    assert np.concatenate([full.obs_slots(o_full)[n] for n in keep]).tobytes() == o_one.tobytes()
    full.close()
    one.close()


def test_own_sensor_slots_read_their_quantities():
    """With the three flags off, the remaining slots still read what they name (noise off)."""
    e = TrikeEnv(config_for(TrikeEnvConfig, FORE128, None, observation_noise=False, domain_randomization=False,
                            **OWN))
    e.reset(seed=0)
    for _ in range(5):
        obs, *_ = e.step(np.full(12, 0.2))
    s, d = e.obs_slots(obs), e.data
    assert set(s) == {"leg_joint_pos", "leg_joint_vel", "gravity_body", "heading_error_sin_cos_ESTIMATE", "gyro",
                      "previous_action_filtered", "previous_action_raw", "command_speed_mps"}
    assert np.allclose(s["leg_joint_pos"], d.qpos[e._leg_qadr] - e._q_neutral, atol=1e-5)
    assert np.allclose(s["gyro"], d.sensordata[e._gyro_adr:e._gyro_adr + 3], atol=1e-4)
    assert np.allclose(s["previous_action_raw"], 0.2)
    e.close()


# ======================================================================================== the run's preset
def test_ownsensors_preset_is_the_d22_fore128_setup_with_three_inputs_off():
    """configs/run3_terrain_ownsensors.json is run3_terrain.json plus the three flags off; with train.py's defaults
    and the notebook's arguments it reproduces the four fore128 runs' recorded settings (run_record_fore128-s0.json,
    seed 0) except those three inputs. The only other keys are the ones added since their commit, at their off
    defaults (the steering run's heat-peak cost and whole-robot draws, D26, D29)."""
    import train

    run3 = json.loads((CONFIGS / "run3_terrain.json").read_text())
    own = json.loads((CONFIGS / f"{OWN_PRESET}.json").read_text())
    assert own == {**run3, **OWN}
    rec = json.loads(D22_RECORD.read_text())["settings"]
    args = train.parse_args(["--out", "x", "--total-steps", "10000000", "--n-envs", "8", "--seed", "0", "--model",
                             "trike60_C_closed_capsule_fore128", "--resume", "--env-config",
                             str(CONFIGS / f"{OWN_PRESET}.json")])
    mine = train.run_settings(args, train.env_config(args))
    assert {k: v for k, v in mine.items() if k != "env_config"} == {k: v for k, v in rec.items() if k != "env_config"}
    ec, rec_ec = mine["env_config"], rec["env_config"]
    differ = {k for k in rec_ec if ec[k] != rec_ec[k]}
    assert differ == {"obs_ankle_angles", "obs_privileged_lin_vel"}
    added = {k: ec[k] for k in ec if k not in rec_ec}
    assert added == {"obs_foot_loads": False, "w_heat_peak": 0.0, "heat_peak_margin": 0.8, "heat_peak_tau_s": 4.0,
                     "heat_peak_start_s": 2.0, "mass_scale_whole_range": None, "payload_kg_range": None}
    e = TrikeEnv(train.env_config(args))
    assert e.observation_space.shape == (57,)
    e.close()


def test_notebook_offers_the_ownsensors_preset_only_with_the_128_mm_model(tmp_path, monkeypatch):
    nb = json.loads((RL_DIR / "trike60_training.ipynb").read_text())
    src = "".join(nb["cells"][1]["source"])
    menu = src[src.index("ENV_PRESET = "):].splitlines()[0]
    assert OWN_PRESET in json.loads(menu[menu.index("["):menu.index("]") + 1])
    params = tmp_path / "params.json"
    for model, ok in (("trike60_C_closed_capsule_fore128", True), ("trike60_C_closed_capsule", False)):
        params.write_text(json.dumps({"REPO_REF": "a" * 40, "RUN_ID": "x", "MODEL": model, "ENV_PRESET": OWN_PRESET}))
        monkeypatch.setenv("TRIKE60_PARAMS", str(params))
        if ok:
            exec(compile(src, "cell1", "exec"), {})
        else:
            with pytest.raises(ValueError, match="128 mm"):
                exec(compile(src, "cell1", "exec"), {})


def test_a_short_own_sensors_run_loads_in_the_tools(tmp_path):
    """train.py with the preset (1,024 steps, a test, not a run), then what reads a run: its record rebuilds the
    57-float env, VecNormalize's statistics and the policy have 57 inputs, the sweep's consistency check passes,
    gait_pipe1's loaders (used by the sweep and the gait report) run an episode, and evaluate.py --run-dir works."""
    import subprocess
    import sys

    sys.path.insert(0, str(RL_DIR / "robustness"))
    sys.path.insert(0, str(RL_DIR / "analysis"))
    import gait_pipe1 as G
    import sweep

    out = tmp_path / "run"
    args = ["--out", out, "--total-steps", 1024, "--n-envs", 2, "--n-steps", 256, "--batch-size", 512,
            "--checkpoint-every", 1024, "--log-every", 512, "--seed", 0, "--eval-every", 0,
            "--model", "trike60_C_closed_capsule_fore128", "--env-config", CONFIGS / f"{OWN_PRESET}.json"]
    p = subprocess.run([sys.executable, RL_DIR / "train.py", *map(str, args)], capture_output=True, text=True,
                       timeout=900, cwd=str(RL_DIR))
    assert p.returncode == 0, p.stdout + p.stderr
    rec = json.loads((out / "run_record.json").read_text())
    ec = rec["settings"]["env_config"]
    assert {k: ec[k] for k in OWN} == OWN and ec["terrain_boxes"] == 8
    zp, pk = sweep.find_checkpoint(out / "checkpoints")
    env = sweep.build_env(ec, randomise=True)
    assert env.observation_space.shape == (57,) and sweep.config_diffs(ec, env.cfg) == {}
    stats = G.load_obs_stats(pk)
    assert stats[0].shape == env.observation_space.shape
    policy = G.build_policy(env, zp, rec["settings"]["net"], rec["settings"]["log_std_init"])
    assert policy.observation_space.shape == (57,)
    arr, _ = G.run_episode(env, policy, stats, seed=100000, command=0.25)
    assert len(arr["t"]) > 0 and "fell" in arr
    p = subprocess.run([sys.executable, RL_DIR / "evaluate.py", "--run-dir", str(out), "--episodes", "1",
                        "--seconds", "1", "--out", str(tmp_path / "eval.json")],
                       capture_output=True, text=True, timeout=600, cwd=str(RL_DIR))
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads((tmp_path / "eval.json").read_text())
    env.close()
