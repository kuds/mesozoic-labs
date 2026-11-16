"""Track B's loader (``designlab/stage2/s2load.py``): the design's "cross-version loading" rows (``STAGE2_PLAN.md``
S2-12, S2-14; Track B rules K2, K4, K9).

The checkpoints are those of a tiny PPO on a stub env with an ``-X`` sized layout (observation 140, actor 102, action
17), built exactly as the trainer builds its model: ``policy.model_kwargs`` updated with ``learning_rate=
LinearSchedule(...)``.  No plant, no reference set and no episode of the study is run.  Everything is written under
pytest's temporary directories.  Needs torch, stable-baselines3 and gymnasium (skipped without them).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import operator
import pickle
import shutil
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
gym = pytest.importorskip("gymnasium")

from designlab.stage2 import s2io, s2load, schedules, tbenv  # noqa: E402
from designlab.stage2.s2load import CheckpointError  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
OBS, ACTOR, ACT = 140, 102, 17
#: A hand-made layout of the design's section 3.1 (16 joints, heading and head command), with one privileged block.
LAYOUT = {"blocks": {"q": (0, 16), "qd": (16, 32), "gravity": (32, 35), "gyro": (35, 38), "foot_load": (38, 40),
                     "phase": (40, 42), "support": (42, 44), "command": (44, 46), "heading": (46, 47),
                     "prev_actions": (47, 98), "head_command": (98, 102), "privileged": (102, 140)},
          "actor_dim": ACTOR, "obs_dim": OBS, "action_dim": ACT, "nj": 16, "head_x": True}
RUN = {"run_uuid": "0f0e0d0c-1111-4222-8333-444455556666", "run_id": "t-run", "job": "R5-stub-s7",
       "settings_sha256": "ab" * 32}
STATE = {"task_weights": {"walk": 0.4, "turn": 0.6}, "fix": None, "monitor": [], "gates": {"g8a": "pass"},
         "no_gain_count": 0, "checks_done": [], "stopped": None, "flags": {"kl_cut": False}}
LR = (1e-4, 3e-5)


class StubEnv(gym.Env):
    """Seeded noise with a per-column offset and scale, so that the statistics are not trivial; counts its calls."""

    def __init__(self, seed: int = 0) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32)
        self._rng, self._t, self.resets, self.steps = np.random.default_rng(seed), 0, 0, 0
        self._offset, self._scale = np.linspace(-1.0, 1.0, OBS), np.linspace(0.2, 3.0, OBS)

    def _obs(self):
        return (self._offset + self._scale * self._rng.standard_normal(OBS)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t, self.resets = 0, self.resets + 1
        return self._obs(), {}

    def step(self, action):
        self._t, self.steps = self._t + 1, self.steps + 1
        return self._obs(), float(-np.square(action).mean()), False, self._t >= 24, {}


def _venv():
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([lambda: StubEnv(1), lambda: StubEnv(2)])


def _build(closure: bool = False):
    """A model as ``train2`` builds it (design 3.6); with ``closure`` Stage 1's own schedule is left in."""
    from designlab.stage1 import policy
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(1)
    venv = VecNormalize(_venv(), norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0, gamma=0.99)
    kw = policy.model_kwargs(ACTOR, log_std_init=math.log(0.1))     # returns a closure learning_rate
    kw.update(n_steps=16, batch_size=16, clip_range=0.2)
    if not closure:
        kw.update(learning_rate=schedules.LinearSchedule(*LR))
    return PPO(env=venv, seed=7, device="cpu", **kw), venv


def scratch_checkpoints(job):
    """Write the checkpoints of steps 32 and 64 of one short run on the stub env into job directory ``job``; the
    live model and wrapper of step 64 with the two records.  Also this file's command line (see its last lines):
    the scratch checkpoint that ``python -m designlab.stage2.s2load check | pack | import`` are tried on."""
    job = Path(job)
    model, venv = _build()
    model.learn(32)
    first = s2load.save_checkpoint(job, 32, model, venv, LAYOUT, RUN, STATE)
    model.learn(32, reset_num_timesteps=False)
    second = s2load.save_checkpoint(job, 64, model, venv, LAYOUT, RUN, dict(STATE, checks_done=[32]))
    return types.SimpleNamespace(job=job, model=model, venv=venv, records={32: first, 64: second})


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return scratch_checkpoints(tmp_path_factory.mktemp("s2load") / "jobs" / RUN["job"])


def _copy(saved, tmp_path, name: str = "job") -> Path:
    return Path(shutil.copytree(saved.job, tmp_path / name))


def _reseal(job, n, **changes) -> dict:
    """Make the record of step n agree with the files as they now are, with ``changes`` to its entries (a test's
    way to get past the hash check)."""
    files = s2load.checkpoint_files(job, n)
    doc = s2io.read_json(files["record"])
    doc.pop("sha256")
    for kind in s2load.FILE_KINDS:
        doc["files"][kind].update(bytes=files[kind].stat().st_size, sha256=s2io.sha256_file(files[kind]))
    doc.update(changes)
    doc["sha256"] = s2io.sha256_json(doc)
    s2io.write_json(files["record"], doc)
    return doc


def _edit_fingerprint(job, n, drop=(), **edits) -> None:
    """Rewrite ``fingerprint_<n>.npz`` with ``edits`` (array name -> a function of a copy of the array that returns
    the new one) and without the arrays named in ``drop``, then make the record agree with the files."""
    path = s2load.checkpoint_files(job, n)["fingerprint"]
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k not in drop}
    for key, change in edits.items():
        arrays[key] = change(arrays[key].copy())
    s2io.write_npz(path, arrays)
    _reseal(job, n)


def _cli(*args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "designlab.stage2.s2load", *map(str, args)], cwd=STUDY,
                          env=env or tbenv.child_env("disable"), capture_output=True, text=True, timeout=300)


# ------------------------------------------------------------------------------------------------
# Imports, versions, names, keep
# ------------------------------------------------------------------------------------------------
def test_importing_s2load_imports_neither_torch_nor_sb3():
    code = ("import sys\nfrom designlab.stage2 import s2load\n"
            "print([m for m in ('torch', 'stable_baselines3', 'gymnasium', 'mujoco', 'environments')"
            " if m in sys.modules])\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=STUDY, env=tbenv.child_env("disable"),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "[]"


def test_versions_and_comparable():
    here = s2load.versions()
    assert tuple(here) == s2load.VERSION_KEYS and all(isinstance(v, str) and v for v in here.values())
    assert here["torch"] == str(torch.__version__)                       # the build label is kept in a record
    assert s2load.comparable(here)["torch"] == str(torch.__version__).split("+")[0]
    cpu, cuda = dict(here, torch="2.15.0+cpu"), dict(here, torch="2.15.0+cu126")
    assert s2load.comparable(cpu) == s2load.comparable(cuda)             # S2-12: the label is dropped on both sides
    assert s2load.comparable(cpu)["torch"] == "2.15.0"
    assert s2load.comparable(dict(here, torch="2.15.1+cpu")) != s2load.comparable(cpu)     # the number is compared
    assert s2load.comparable(dict(here, python="3.12.11")) != s2load.comparable(here)
    assert s2load.comparable(dict(here, numpy="2+x")) == dict(s2load.comparable(here), numpy="2+x")   # torch's only
    assert tuple(s2load.comparable(dict(reversed(list(here.items())), extra="1"))) == s2load.VERSION_KEYS
    lacking = {k: v for k, v in here.items() if k != "cloudpickle"}
    assert s2load.comparable(lacking) != s2load.comparable(here)         # a set that lacks a key is another set
    assert s2load.comparable({"torch": "2.15.0+cpu"}) == {"torch": "2.15.0"}               # a table of pins


def test_names_and_listing(saved, tmp_path):
    files = s2load.checkpoint_files(saved.job, 64)
    assert {k: p.name for k, p in files.items()} == {"model": "model_64.zip", "vecnormalize": "vecnormalize_64.pkl",
                                                     "fingerprint": "fingerprint_64.npz",
                                                     "record": "checkpoint_64.json"}
    assert all(p.parent == saved.job / "checkpoints" and p.is_file() for p in files.values())
    assert s2load.list_checkpoints(saved.job) == [32, 64]
    assert s2load.list_checkpoints(tmp_path / "nothing") == []
    ck = tmp_path / "job" / "checkpoints"
    ck.mkdir(parents=True)
    for name in ("checkpoint_1000.json", "checkpoint_250.json", "checkpoint_x.json", "checkpoint_7.json.tmp",
                 "model_9.zip"):
        (ck / name).write_text("{}")
    assert s2load.list_checkpoints(tmp_path / "job") == [250, 1000]      # by the number, not by the name
    assert s2load.tar_name(250) == "ckpt_250.tar"
    for bad in (-1, 2.5, "64"):
        with pytest.raises(ValueError):
            s2load.checkpoint_files(saved.job, bad)


def test_keep():
    assert s2load.keep([0, 1024, 2048, 3072, 4096], 2048) == [2048, 3072, 4096]
    assert s2load.keep([4096, 0, 2048, 1024, 3072, 2048], 2048) == [2048, 3072, 4096]      # any order, repeats
    assert s2load.keep([], 2048) == [] and s2load.keep([0], 2048) == [0] and s2load.keep([0, 1024], 2048) == [0, 1024]
    assert s2load.keep([0, 1024, 2048], 2048) == [1024, 2048]            # step 0 is no check's
    # the trainer's own step counts: rollout starts (8,192 steps each) at or past the multiples of 250,000
    every, done = 250_000, []
    for n in range(0, 7_700_000, 8192):
        if n >= every * len(done):
            done.append(n)
    assert done[:3] == [0, 253_952, 507_904] and 2_506_752 in done and 2_498_560 not in done
    kept = s2load.keep(done, 2_500_000)
    assert (done[-2], done[-1]) == (7_258_112, 7_503_872)
    assert kept == [2_506_752, 5_005_312, 7_258_112, 7_503_872]          # three checks' and the newest two
    assert s2load.keep(kept, 2_500_000) == kept
    live = []                                    # deleting after every save keeps what the whole history would
    for n in done:
        live = s2load.keep(live + [n], 2_500_000)
    assert live == kept
    assert s2load.keep([2_506_752], 2_500_000) == [2_506_752]
    assert s2load.keep([0, 2_490_000, 2_506_752, 2_760_000, 3_000_000], 2_500_000) == [2_506_752, 2_760_000, 3_000_000]
    for bad in (0, -5):
        with pytest.raises(ValueError):
            s2load.keep([0, 1], bad)


# ------------------------------------------------------------------------------------------------
# The fingerprint's observations and the plain statistics
# ------------------------------------------------------------------------------------------------
def test_fingerprint_obs():
    obs = s2load.fingerprint_obs(LAYOUT)
    assert obs.shape == (256, OBS) and obs.dtype == np.float32 and np.all(np.isfinite(obs))
    assert np.array_equal(obs, s2load.fingerprint_obs(dict(LAYOUT)))     # seeded: the same table every time
    assert not obs[:, ACTOR:].any()                                      # the critic's columns are zero
    b = {k: obs[:, i0:i1].astype(np.float64) for k, (i0, i1) in LAYOUT["blocks"].items()}
    rng = np.random.default_rng([20261115, OBS, ACTOR])                  # K9's seed; the first block is drawn first
    assert np.array_equal(obs[:, 0:16], (0.3 * rng.standard_normal((256, 16))).astype(np.float32))
    assert np.array_equal(obs[:, 16:32], (2.0 * rng.standard_normal((256, 16))).astype(np.float32))
    assert 0.2 < b["q"].std() < 0.4 and 1.7 < b["qd"].std() < 2.3 and 0.4 < b["gyro"].std() < 0.6
    assert np.allclose(np.linalg.norm(b["gravity"], axis=1), 1.0, atol=1e-6) and np.all(b["gravity"][:, 2] < -0.9)
    assert b["foot_load"].min() >= 0.0 and b["foot_load"].max() <= 1.0 and b["foot_load"].std() > 0.2
    assert np.allclose(np.linalg.norm(b["phase"], axis=1), 1.0, atol=1e-6) and b["phase"].std() > 0.5
    assert set(np.unique(b["support"])) == {0.0, 1.0}
    assert b["command"][:, 0].min() >= 0.0 and b["command"][:, 0].max() <= 1.05 + 1e-6
    assert b["command"][:, 1].min() < -0.5 and np.abs(b["command"][:, 1]).max() <= 1.05 + 1e-6
    assert 0.2 < b["heading"].std() < 0.4 and 0.08 < b["prev_actions"].std() < 0.12
    assert np.abs(b["head_command"]).max() <= 1.0 and b["head_command"].min() < -0.5
    # another layout is another table: 14 joints, no head (88 / 130 / 15)
    plain = {"blocks": {"q": (0, 14), "qd": (14, 28), "gravity": (28, 31), "gyro": (31, 34), "foot_load": (34, 36),
                        "phase": (36, 38), "support": (38, 40), "command": (40, 42), "heading": (42, 43),
                        "prev_actions": (43, 88), "root_vel": (88, 91), "rest": (91, 130)},
             "actor_dim": 88, "obs_dim": 130, "action_dim": 15}
    other = s2load.fingerprint_obs(plain)
    assert other.shape == (256, 130) and not other[:, 88:].any() and not np.array_equal(other[:, :14], obs[:, :14])
    # the whole table is part of the rule every record carries (FINGERPRINT_RULE): both tables to the bit, so that
    # no constant of FINGERPRINT_BLOCKS, no kind of draw and no order of the draws changes without this line changing
    assert s2load._obs_sha256(obs) == "6fc5c509d17682ad1cd5839ccd162fe92f19deb1c71fcaf21c913c72305fd4b6"
    assert s2load._obs_sha256(other) == "0a9aa56cd5df322d815cb088add1cead8cca01d05f31e67a40c1a86910bc7527"
    assert s2load._obs_sha256(obs) == hashlib.sha256(obs.astype("<f4").tobytes()).hexdigest()   # what the hash is of


def test_fingerprint_obs_refuses_a_layout_it_cannot_draw():
    def layout(**blocks):
        return dict(LAYOUT, blocks=dict(LAYOUT["blocks"], **blocks))

    with pytest.raises(CheckpointError, match="no fingerprint distribution for the actor block 'mystery'"):
        s2load.fingerprint_obs(dict(LAYOUT, blocks={("mystery" if k == "gyro" else k): v
                                                    for k, v in LAYOUT["blocks"].items()}))
    with pytest.raises(CheckpointError, match="do not tile"):
        s2load.fingerprint_obs(layout(heading=(45, 47)))                   # it overlaps the command block
    with pytest.raises(CheckpointError, match="across actor_dim"):
        s2load.fingerprint_obs(layout(head_command=(98, 104), privileged=(104, 140)))
    with pytest.raises(CheckpointError, match="gravity vector"):
        s2load.fingerprint_obs(layout(gravity=(32, 34), gyro=(34, 38)))
    with pytest.raises(CheckpointError, match="no blocks"):
        s2load.fingerprint_obs({k: v for k, v in LAYOUT.items() if k != "blocks"})
    with pytest.raises(CheckpointError, match="obs_dim"):
        s2load.fingerprint_obs({"blocks": LAYOUT["blocks"], "actor_dim": ACTOR})


def test_normalize_is_vecnormalizes_formula(saved):
    stats = s2load.stats_of(saved.venv)
    assert tuple(stats) == s2load.STATS_KEYS and all(v.dtype == np.float64 for v in stats.values())
    assert stats["obs_mean"].shape == (OBS,) and float(stats["clip_obs"]) == 10.0 and float(stats["obs_count"]) > 60
    assert stats["obs_mean"] is not saved.venv.obs_rms.mean              # copies
    rng = np.random.default_rng(5)
    obs = (rng.standard_normal((64, OBS)) * np.linspace(0.1, 40.0, OBS)).astype(np.float32)
    want, got = saved.venv.normalize_obs(obs), s2load.normalize(stats, obs)
    assert got.dtype == np.float32 and np.array_equal(got, want)         # to the bit
    assert np.count_nonzero(np.abs(got) == 10.0) > 50                    # the clip was reached
    assert np.array_equal(s2load.normalize(stats, obs[3]), saved.venv.normalize_obs(obs[3]))          # one row
    assert np.array_equal(s2load.normalize(stats, obs[:, :ACTOR]), want[:, :ACTOR])     # the actor's columns alone
    with pytest.raises(CheckpointError, match="wide"):
        s2load.normalize(stats, np.zeros((2, OBS + 1), np.float32))


# ------------------------------------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------------------------------------
def test_a_saved_archive_holds_no_bytecode_member(saved):
    tbenv.repo_on_path()
    from environments.shared.policy_loading import inspect_sb3_archive

    for n in (32, 64):
        seen = inspect_sb3_archive(s2load.checkpoint_files(saved.job, n)["model"])
        assert seen.bytecode_members == frozenset()
        assert {"learning_rate", "lr_schedule", "clip_range"} <= seen.serialized_members   # held by reference
    assert saved.model.learning_rate == schedules.LinearSchedule(*LR)
    assert schedules.describe(saved.model.lr_schedule) == {"class": "LinearSchedule", "start": LR[0], "end": LR[1]}


def test_the_record_and_the_four_files(saved):
    files, rec = s2load.checkpoint_files(saved.job, 64), saved.records[64]
    assert s2io.read_json(files["record"]) == rec
    assert rec["schema"] == "s2-ckpt-2" and rec["steps"] == 64 and rec["num_timesteps"] == 64
    assert {k: rec[k] for k in s2load.RUN_KEYS} == RUN
    assert rec["run_state"] == dict(STATE, checks_done=[32]) and saved.records[32]["run_state"] == STATE   # as given
    assert sorted(rec["files"]) == sorted(s2load.FILE_KINDS)
    for kind in s2load.FILE_KINDS:
        assert rec["files"][kind] == {"name": files[kind].name, "bytes": files[kind].stat().st_size,
                                      "sha256": s2io.sha256_file(files[kind])}
    assert rec["versions"] == s2load.versions()
    assert {k: rec["layout"][k] for k in ("actor_dim", "obs_dim", "action_dim", "nj", "heading")} == {
        "actor_dim": ACTOR, "obs_dim": OBS, "action_dim": ACT, "nj": 16, "heading": True}
    assert rec["layout"]["blocks"]["q"] == [0, 16] and rec["layout"]["head_x"] is True
    fp = rec["fingerprint"]
    assert (fp["rows"], fp["seed"], fp["tol"]) == (256, [20261115, OBS, ACTOR], 1e-5)
    assert fp["rule"] == s2load.FINGERPRINT_RULE
    body ={k: v for k, v in rec.items() if k != "sha256"}
    assert rec["sha256"] == s2io.sha256_json(body)
    # the order of the writes, no temporary file left
    times = [files[k].stat().st_mtime_ns for k in ("model", "vecnormalize", "fingerprint", "record")]
    assert times == sorted(times)
    assert sorted(p.name for p in files["record"].parent.iterdir()) == sorted(
        s2load.FILE_NAMES[k] % n for k in s2load.FILE_NAMES for n in (32, 64))
    # the fingerprint file: the seeded table (drawn again from the record's own layout), the actions, plain statistics
    with np.load(files["fingerprint"], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    assert sorted(arrays) == sorted(("obs", "actions") + s2load.STATS_KEYS)
    assert arrays["obs"].dtype == np.float32 and np.array_equal(arrays["obs"], s2load.fingerprint_obs(rec["layout"]))
    assert fp["obs_sha256"] == s2load._obs_sha256(arrays["obs"])
    assert arrays["actions"].shape == (256, ACT) and arrays["actions"].dtype == np.float32
    assert np.abs(arrays["actions"]).max() <= 1.0 and arrays["actions"].std() > 0.0
    live = s2load.stats_of(saved.venv)
    assert all(np.array_equal(arrays[k], live[k]) and arrays[k].dtype == np.float64 for k in s2load.STATS_KEYS)
    assert np.array_equal(arrays["actions"], s2load.fingerprint(saved.model, live, arrays["obs"]))
    # S2-12's definition, by stable-baselines3's own code and none of this module's: the deterministic actions of
    # the policy on the observations as the SAVED statistics normalise them (not on the observations as drawn)
    normalised = saved.venv.normalize_obs(arrays["obs"])
    want, _ = saved.model.predict(normalised, deterministic=True)
    assert np.array_equal(arrays["actions"], want)
    raw, _ = saved.model.predict(arrays["obs"], deterministic=True)
    assert not np.array_equal(arrays["actions"], raw) and np.abs(want - raw).max() > 1e-3
    # the rows the normalisation clips in an actor column, counted from VecNormalize's own output
    clipped = int(np.any(np.abs(normalised[:, :ACTOR]) >= 10.0, axis=1).sum())
    assert fp["clipped_rows"] == clipped and 0 < clipped < 256
    # the two checkpoints differ (the policy and the statistics moved between them)
    with np.load(s2load.checkpoint_files(saved.job, 32)["fingerprint"]) as z:
        assert np.array_equal(z["obs"], arrays["obs"]) and not np.array_equal(z["actions"], arrays["actions"])


def test_an_archive_with_stage1s_closure_schedule_is_refused(tmp_path):
    model, venv = _build(closure=True)
    job = tmp_path / "job"
    with pytest.raises(CheckpointError, match=r"learning_rate, lr_schedule hold bytecode"):
        s2load.save_checkpoint(job, 0, model, venv, LAYOUT, RUN, STATE)
    assert list((job / "checkpoints").iterdir()) == []                   # none of the files, no temporary file
    # ... also when a good checkpoint of that step stood there before: its record does not survive a refused save
    good, good_venv = _build()
    s2load.save_checkpoint(job, 0, good, good_venv, LAYOUT, RUN, STATE)
    assert s2load.pair_problem(job, 0) is None
    with pytest.raises(CheckpointError, match="bytecode"):
        s2load.save_checkpoint(job, 0, model, venv, LAYOUT, RUN, STATE)
    assert list((job / "checkpoints").iterdir()) == [] and s2load.list_checkpoints(job) == []


def test_saving_draws_no_random_number_and_touches_no_env(tmp_path):
    import random

    model, venv = _build()
    model.learn(32)
    before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate(),
              [(e.resets, e.steps) for e in venv.venv.envs], model._last_obs.copy(), s2load.stats_of(venv),
              venv.returns.copy(), [t.clone() for t in model.policy.state_dict().values()])
    s2load.save_checkpoint(tmp_path / "job", 32, model, venv, LAYOUT, RUN, STATE)
    assert torch.equal(torch.get_rng_state(), before[0])
    now = np.random.get_state()
    assert now[0] == before[1][0] and np.array_equal(now[1], before[1][1]) and now[2:] == before[1][2:]
    assert random.getstate() == before[2]
    assert [(e.resets, e.steps) for e in venv.venv.envs] == before[3]
    assert np.array_equal(model._last_obs, before[4]) and np.array_equal(venv.returns, before[6])
    assert all(np.array_equal(s2load.stats_of(venv)[k], before[5][k]) for k in s2load.STATS_KEYS)
    assert all(torch.equal(a, b) for a, b in zip(model.policy.state_dict().values(), before[7]))


def test_save_refuses_before_it_writes(saved, tmp_path):
    from stable_baselines3.common.vec_env import VecNormalize

    job = tmp_path / "job"
    # statistics that are not one flat observation's (normalisation off; a dict observation's), or of another width
    with pytest.raises(CheckpointError, match="plain statistics need a VecNormalize that normalises one flat Box"):
        s2load.save_checkpoint(job, 64, saved.model, VecNormalize(_venv(), norm_obs=False), LAYOUT, RUN, STATE)
    with pytest.raises(CheckpointError, match="plain statistics need"):
        s2load.stats_of(types.SimpleNamespace(obs_rms={"camera": None}, norm_obs=True))
    rms = types.SimpleNamespace(mean=np.zeros(130), var=np.ones(130), count=1.0)
    narrow_stats = types.SimpleNamespace(obs_rms=rms, norm_obs=True, clip_obs=10.0, epsilon=1e-8)
    with pytest.raises(CheckpointError, match="the statistics are 130 wide, the layout 140"):
        s2load.save_checkpoint(job, 64, saved.model, narrow_stats, LAYOUT, RUN, STATE)
    for run, said in ((dict(RUN, mode="R5"), r"run holds \['mode'\]"),
                      ({k: v for k, v in RUN.items() if k != "run_uuid"}, "run_uuid"),
                      (dict(RUN, settings_sha256=""), "settings_sha256")):
        with pytest.raises(CheckpointError, match=said):
            s2load.save_checkpoint(job, 64, saved.model, saved.venv, LAYOUT, run, STATE)
    narrow = dict(LAYOUT, obs_dim=130, blocks={k: v for k, v in LAYOUT["blocks"].items() if k != "privileged"})
    with pytest.raises(CheckpointError, match="spaces"):
        s2load.save_checkpoint(job, 64, saved.model, saved.venv, narrow, RUN, STATE)
    other_actor = dict(LAYOUT, actor_dim=98, blocks={k: v for k, v in LAYOUT["blocks"].items() if k != "head_command"})
    with pytest.raises(CheckpointError, match="actor_dim"):
        s2load.save_checkpoint(job, 64, saved.model, saved.venv, other_actor, RUN, STATE)
    with pytest.raises(TypeError):
        s2load.save_checkpoint(job, 64, saved.model, saved.venv, LAYOUT, RUN, dict(STATE, bad={1, 2}))
    assert not job.exists()


def test_a_save_whose_fingerprint_is_not_finite_leaves_nothing(saved, tmp_path, monkeypatch):
    job, real = tmp_path / "job", s2load.fingerprint
    s2load.save_checkpoint(job, 64, saved.model, saved.venv, LAYOUT, RUN, STATE)       # a good one stands there
    assert s2load.pair_problem(job, 64) is None
    for bad in (np.inf, np.nan):                  # a policy that diverged: one action column is no number
        monkeypatch.setattr(s2load, "fingerprint", lambda model, stats, obs, bad=bad: np.where(
            np.arange(ACT) == 3, np.float32(bad), real(model, stats, obs)))
        with pytest.raises(CheckpointError, match="checkpoint 64: the fingerprint actions are not 256 finite rows"):
            s2load.save_checkpoint(job, 64, saved.model, saved.venv, LAYOUT, RUN, STATE)
        assert list((job / "checkpoints").iterdir()) == []                # no record, no binary, no temporary file
    monkeypatch.setattr(s2load, "fingerprint", lambda model, stats, obs: real(model, stats, obs)[:, :-1])
    with pytest.raises(CheckpointError, match="not 256 finite rows of 17"):
        s2load.save_checkpoint(job, 64, saved.model, saved.venv, LAYOUT, RUN, STATE)
    assert list((job / "checkpoints").iterdir()) == []
    monkeypatch.undo()
    s2load.save_checkpoint(job, 64, saved.model, saved.venv, LAYOUT, RUN, STATE)
    assert s2load.load_checkpoint(job, 64)["fingerprint_max_abs_diff"] == 0.0


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
def test_an_inference_load_reproduces_the_fingerprint(saved):
    for n in (32, 64):
        ck = s2load.load_checkpoint(saved.job, n, expect_run=RUN)
        assert sorted(ck) == sorted(("model", "stats", "record", "versions_saved", "versions_here", "cross_version",
                                     "fingerprint_max_abs_diff", "tol"))
        assert ck["fingerprint_max_abs_diff"] == 0.0 and ck["tol"] == 1e-5
        assert ck["cross_version"] is False and ck["versions_saved"] == ck["versions_here"] == s2load.versions()
        assert ck["record"] == saved.records[n]
        model = ck["model"]
        assert model.get_env() is None and model.num_timesteps == n
        assert all(getattr(model, name) is None for name in s2load.INFERENCE_NONE)
        assert model.learning_rate == schedules.ConstantSchedule(0.0) and model.clip_range(1.0) == 0.0
        assert model.observation_space.shape == (OBS,) and model.action_space.shape == (ACT,)
        assert float(model.action_space.low.max()) == -1.0 and float(model.action_space.high.min()) == 1.0
    live = s2load.stats_of(saved.venv)                                   # the statistics come from the plain arrays
    assert all(np.array_equal(ck["stats"][k], live[k]) for k in s2load.STATS_KEYS)
    for a, b in zip(ck["model"].policy.state_dict().values(), saved.model.policy.state_dict().values()):
        assert torch.equal(a, b)
    # the deploy path: the actor's columns alone give the same actions as the full rows
    obs = s2load.fingerprint_obs(LAYOUT)
    full, _ = ck["model"].predict(s2load.normalize(ck["stats"], obs), deterministic=True)
    actor, _ = ck["model"].predict(s2load.normalize(ck["stats"], obs[:, :ACTOR]), deterministic=True)
    assert np.array_equal(full, actor)


def test_the_fingerprint_is_reproduced_exactly_in_a_fresh_process(saved):
    done = _cli("check", saved.job, 64)
    assert done.returncode == 0, done.stdout + done.stderr[-2000:]
    assert "S2 CHECKPOINT 64: intact; fingerprint 0.0 " in done.stdout and "cross_version False" in done.stdout


def test_the_cli_assigns_mujoco_gl_itself(saved):
    # started under OSMesa, main() still builds its policy in a process with no GL library loaded
    env = tbenv.child_env("disable")
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    code = ("import json, os, sys\n"
            "from designlab.stage2 import s2load\n"
            f"rc = s2load.main(['check', {str(saved.job)!r}, '32'])\n"
            "print(json.dumps([rc, os.environ.get('MUJOCO_GL'), os.environ.get('PYOPENGL_PLATFORM'),"
            " 'mujoco' in sys.modules, 'OpenGL' in sys.modules]))\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=STUDY, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr[-2000:]
    assert "S2 CHECKPOINT 32: intact; fingerprint 0.0 " in done.stdout
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [0, "disable", None, True, False]
    # ... and an inherited thread count other than one is refused, not silently kept (round A's rule for a CLI)
    done = _cli("check", saved.job, 32, env=dict(tbenv.child_env("disable"), OMP_NUM_THREADS="4"))
    assert done.returncode != 0 and "one BLAS thread is required" in done.stderr


class _Boom:
    """Unpickling it divides by zero: a stand-in for a member another library version cannot read."""

    def __reduce__(self):
        return operator.truediv, (1, 0)


def _doctor(src: Path, dst: Path, members) -> None:
    import cloudpickle

    bad = base64.b64encode(cloudpickle.dumps(_Boom())).decode()
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            payload = zin.read(item.filename)
            if item.filename == "data":
                doc = json.loads(payload)
                for name in members:
                    assert ":serialized:" in doc[name], name
                    doc[name][":serialized:"] = bad
                payload = json.dumps(doc).encode()
            zout.writestr(item, payload)


def test_an_inference_load_unpickles_neither_space_nor_last_observation(saved, tmp_path):
    from stable_baselines3 import PPO

    tbenv.repo_on_path()
    from environments.shared.policy_loading import inspect_sb3_archive, load_sb3_model

    src = s2load.checkpoint_files(saved.job, 64)["model"]
    # every pickled member of the archive itself, less the two class references (K9: "only tensors and the policy
    # class cross library versions"): the list is read from the archive, not from the loader's own constants
    replaced = tuple(sorted(inspect_sb3_archive(src).serialized_members - {"policy_class", "rollout_buffer_class"}))
    assert set(replaced) == {"observation_space", "action_space", "learning_rate", "lr_schedule", "clip_range",
                             "_last_obs", "_last_original_obs", "_last_episode_starts", "ep_info_buffer",
                             "ep_success_buffer"}
    assert set(s2load.INFERENCE_NONE) == set(replaced) - {"observation_space", "action_space", "learning_rate",
                                                          "lr_schedule", "clip_range"}
    for name in replaced:                         # each member alone stops a plain load ...
        one = tmp_path / f"one_{name}.zip"
        _doctor(src, one, (name,))
        with pytest.raises(ZeroDivisionError):
            PPO.load(str(one), device="cpu")
    doctored = tmp_path / "doctored.zip"
    _doctor(src, doctored, replaced)
    assert inspect_sb3_archive(doctored).bytecode_members == frozenset()   # nothing the repository's loader replaces
    with pytest.raises(ZeroDivisionError):
        load_sb3_model(str(doctored), algorithm="ppo", device="cpu")
    model = s2load.load_model(doctored, LAYOUT)                           # ... and the inference load reads none
    with np.load(s2load.checkpoint_files(saved.job, 64)["fingerprint"]) as z:
        stats = {k: z[k] for k in s2load.STATS_KEYS}
        assert np.array_equal(s2load.fingerprint(model, stats, z["obs"]), z["actions"])
    with pytest.raises(ZeroDivisionError):        # a training load does read the spaces (to check the env's)
        s2load.load_model(doctored, LAYOUT, env=_venv(), lr=LR)


def test_a_load_for_judging_does_not_unpickle_the_statistics_file(saved, tmp_path):
    """K9: a load for judging reads the statistics from the plain arrays; ``vecnormalize_<n>.pkl`` is checked by its
    size and sha256 and is not unpickled, so a file this machine's libraries cannot read does not stop a checkpoint
    from being judged.  A training resume uses the file, and is refused."""
    job = _copy(saved, tmp_path)
    path = s2load.checkpoint_files(job, 64)["vecnormalize"]
    path.write_bytes(pickle.dumps(_Boom()))                              # unpickling it divides by zero
    with pytest.raises(ZeroDivisionError):
        pickle.loads(path.read_bytes())
    _refused(job, 64, "vecnormalize_64.pkl has")                         # still one of the checkpoint's three files
    _reseal(job, 64)
    ck = s2load.load_checkpoint(job, 64, expect_run=RUN)
    assert ck["fingerprint_max_abs_diff"] == 0.0 and "vecnorm" not in ck
    packed =s2load.pack(job, 64, tmp_path / "out")                       # ... and it comes through an import
    got = s2load.import_tar(s2load.checkpoint_files(job, 64)["record"], packed["tar"], tmp_path / "here")
    assert got["ok"] is True and got["fingerprint_max_abs_diff"] == 0.0
    assert "does not unpickle vecnormalize_<n>.pkl" in got["rules"]["loader"]
    # what a resume starts from: the repository's own check of the pair, which unpickles the file
    why = "VecNormalize sidecar vecnormalize_64.pkl does not unpickle (division by zero)"
    assert s2load.pair_problem(job, 64) == why
    assert s2load.newest_intact(job) == (32, [{"steps": 64, "reason": why}])
    venv = _venv()
    assert _refused(job, 64, "does not unpickle", env=venv, lr=LR).endswith(why)
    assert [(e.resets, e.steps) for e in venv.envs] == [(0, 0), (0, 0)]


def test_the_zip_test_of_a_judging_load_is_the_repositorys(saved, tmp_path):
    """The inference load's zip test is this module's copy of the first half of the repository's
    ``checkpoint_pair_problem`` (the second half unpickles the ``.pkl``): the same verdict in the same words."""
    tbenv.repo_on_path()
    from environments.shared.curriculum.checkpoints import checkpoint_pair_problem

    files = s2load.checkpoint_files(saved.job, 64)
    whole = files["model"].read_bytes()
    flipped = bytearray(whole)
    flipped[1000] ^= 0xFF                         # inside a stored member: its CRC no longer matches
    cases = {"whole": whole, "cut": whole[: len(whole) * 2 // 3], "tail": whole[-(len(whole) // 3):], "empty": b"",
             "noise": b"not a zip at all" * 200, "flipped": bytes(flipped)}
    with zipfile.ZipFile(files["model"]) as zin:
        names = zin.namelist()
        for drop in ("data", "policy.pth"):       # an archive that opens and tests clean, without one SB3 member
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zout:
                for item in zin.infolist():
                    if item.filename != drop:     # by name: writestr(item, ...) would move zin's own offsets
                        zout.writestr(item.filename, zin.read(item.filename))
            cases["without " + drop] = buf.getvalue()
    assert {"data", "policy.pth"} <= set(names)
    path, said = tmp_path / "model_64.zip", {}
    for name, payload in cases.items():
        path.write_bytes(payload)
        said[name] = s2load._archive_problem(path)
        assert said[name] == checkpoint_pair_problem(path, files["vecnormalize"]), name
    assert said.pop("whole") is None
    assert all(why.startswith("bad/truncated checkpoint zip model_64.zip (") for why in said.values()), said
    assert all("outer archive lacks SB3 members" in said[k] for k in ("without data", "without policy.pth"))
    assert "corrupt archive member" in said["flipped"]
    assert s2load._archive_problem(tmp_path / "absent.zip").startswith("bad/truncated checkpoint zip absent.zip (")


def test_a_training_load_keeps_the_last_observation_and_resets_nothing(saved):
    venv = _venv()
    ck = s2load.load_checkpoint(saved.job, 64, env=venv, lr=LR, expect_run=RUN)
    model, vecnorm = ck["model"], ck["vecnorm"]
    assert ck["fingerprint_max_abs_diff"] == 0.0 and ck["cross_version"] is False
    assert vecnorm.venv is venv and model.get_env() is vecnorm and model.get_vec_normalize_env() is vecnorm
    assert [(e.resets, e.steps) for e in venv.envs] == [(0, 0), (0, 0)]  # nothing reset or stepped the vec env
    # force_reset=False reached SB3: the archive's last observation is still the model's
    assert model._last_obs is not None and np.array_equal(model._last_obs, saved.model._last_obs)
    assert np.array_equal(model._last_original_obs, saved.model._last_original_obs)
    assert np.array_equal(model._last_episode_starts, saved.model._last_episode_starts)
    assert model.num_timesteps == 64 and model._n_updates == saved.model._n_updates
    # the schedules are the objects passed, not the archive's
    assert model.learning_rate == schedules.LinearSchedule(*LR)
    assert schedules.describe(model.lr_schedule) == {"class": "LinearSchedule", "start": LR[0], "end": LR[1]}
    assert schedules.describe(model.clip_range) == {"class": "ConstantSchedule", "value": 0.2}
    # weights, optimizer and statistics are the saved ones
    for a, b in zip(model.policy.state_dict().values(), saved.model.policy.state_dict().values()):
        assert torch.equal(a, b)
    mine, theirs = model.policy.optimizer.state_dict(), saved.model.policy.optimizer.state_dict()
    assert mine["param_groups"][0]["lr"] == theirs["param_groups"][0]["lr"]
    for key, slot in theirs["state"].items():
        assert all(torch.equal(slot[name], mine["state"][key][name]) for name in ("exp_avg", "exp_avg_sq", "step"))
    live = s2load.stats_of(saved.venv)
    assert all(np.array_equal(s2load.stats_of(vecnorm)[k], live[k]) for k in s2load.STATS_KEYS)
    assert np.array_equal(vecnorm.ret_rms.var, saved.venv.ret_rms.var)
    assert vecnorm.ret_rms.count == saved.venv.ret_rms.count
    assert vecnorm.training is True and vecnorm.norm_reward is True and vecnorm.clip_reward == 50.0


def _run_state(model, vecnorm) -> dict:
    out = {f"policy.{k}": v.clone() for k, v in model.policy.state_dict().items()}
    for i, slot in model.policy.optimizer.state_dict()["state"].items():
        out.update({f"optimizer.{i}.{k}": torch.as_tensor(v).clone() for k, v in slot.items()})
    for k, v in (("obs_mean", vecnorm.obs_rms.mean), ("obs_var", vecnorm.obs_rms.var), ("returns", vecnorm.returns),
                 ("obs_count", vecnorm.obs_rms.count), ("ret_mean", vecnorm.ret_rms.mean),
                 ("ret_var", vecnorm.ret_rms.var), ("ret_count", vecnorm.ret_rms.count), ("last_obs", model._last_obs),
                 ("steps", model.num_timesteps), ("updates", model._n_updates)):
        out[k] = torch.as_tensor(np.asarray(v, dtype=np.float64))
    return out


def test_a_training_load_then_the_same_boundary_continues_bit_for_bit(tmp_path):
    """The loader's half of the kill-and-resume rule (K1, K3), on the stub env: a run that saves and then reseeds
    and resets at every rollout start, against the same run loaded from each of its checkpoints and continued."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed

    budget, every, job = 96, 32, tmp_path / "job"

    class Boundaries(BaseCallback):
        def __init__(self, vecnorm, save: bool) -> None:
            super().__init__()
            self.vecnorm, self.save, self.due = vecnorm, save, None

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            n = self.model.num_timesteps
            if self.due is None:                  # a session's first rollout start is a boundary
                self.due = n
            if n < self.due:
                return
            if self.save:
                s2load.save_checkpoint(job, n, self.model, self.vecnorm, LAYOUT, RUN, {"at": n})
            seed = 100_000 * 7 + 100 * (n // every)
            set_random_seed(seed)
            self.model.action_space.seed(seed)
            self.vecnorm.seed(seed)
            self.model._last_obs = self.vecnorm.reset()
            self.vecnorm.returns[:] = 0.0
            self.model._last_episode_starts = np.ones((self.vecnorm.num_envs,), dtype=bool)
            self.model._last_original_obs = self.vecnorm.get_original_obs()
            self.due = (n // every + 1) * every

    model, venv = _build()
    model.learn(budget, callback=Boundaries(venv, save=True))
    assert s2load.list_checkpoints(job) == [0, 32, 64] and model.num_timesteps == budget
    whole = _run_state(model, venv)
    obs = s2load.fingerprint_obs(LAYOUT)
    for n in (64, 32, 0):
        ck = s2load.load_checkpoint(job, n, env=_venv(), lr=LR, expect_run=RUN)
        ck["model"].learn(budget - n, reset_num_timesteps=False, callback=Boundaries(ck["vecnorm"], save=False))
        resumed = _run_state(ck["model"], ck["vecnorm"])
        assert sorted(resumed) == sorted(whole) and len(whole) > 40
        assert [k for k in whole if not torch.equal(whole[k], resumed[k])] == [], f"resumed from {n}"
        assert np.array_equal(s2load.fingerprint(ck["model"], s2load.stats_of(ck["vecnorm"]), obs),
                              s2load.fingerprint(model, s2load.stats_of(venv), obs))


def test_load_arguments(saved):
    from stable_baselines3.common.vec_env import VecNormalize

    with pytest.raises(ValueError, match="lr="):
        s2load.load_checkpoint(saved.job, 64, env=_venv())
    with pytest.raises(ValueError, match="training load"):
        s2load.load_checkpoint(saved.job, 64, lr=LR)
    with pytest.raises(ValueError, match="pass the vec env itself"):
        s2load.load_checkpoint(saved.job, 64, env=VecNormalize(_venv()), lr=LR)
    with pytest.raises(ValueError, match="expect_run"):
        s2load.load_checkpoint(saved.job, 64, expect_run={"run_uuid": RUN["run_uuid"]})
    with pytest.raises(ValueError, match="lr="):
        s2load.load_model(s2load.checkpoint_files(saved.job, 64)["model"], LAYOUT, env=_venv())
    with pytest.raises(ValueError, match="an inference load has no schedule"):
        s2load.load_model(s2load.checkpoint_files(saved.job, 64)["model"], LAYOUT, lr=LR)


# ------------------------------------------------------------------------------------------------
# Refusals, each with its reason
# ------------------------------------------------------------------------------------------------
def _refused(job, n, said, **kw) -> str:
    with pytest.raises(CheckpointError, match=said) as err:
        s2load.load_checkpoint(job, n, **kw)
    return str(err.value)


def test_a_changed_byte_is_refused(saved, tmp_path):
    job = _copy(saved, tmp_path)
    for kind in s2load.FILE_KINDS:
        path = s2load.checkpoint_files(job, 64)[kind]
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        path.write_bytes(bytes(raw))
        why = s2load.pair_problem(job, 64)
        assert why.startswith(f"{path.name} has sha256 ") and "the record says" in why
        _refused(job, 64, f"checkpoint 64 of .*{path.name} has sha256")
        shutil.copy(s2load.checkpoint_files(saved.job, 64)[kind], path)
        assert s2load.pair_problem(job, 64) is None


def test_a_missing_file_is_refused(saved, tmp_path):
    job = _copy(saved, tmp_path)
    for kind in s2load.FILE_KINDS:                # the statistics sidecar, the archive, the fingerprint
        path = s2load.checkpoint_files(job, 64)[kind]
        path.unlink()
        assert s2load.pair_problem(job, 64) == f"missing file {path.name}"
        _refused(job, 64, f"missing file {path.name}")
        shutil.copy(s2load.checkpoint_files(saved.job, 64)[kind], path)
    s2load.checkpoint_files(job, 64)["record"].unlink()
    assert s2load.pair_problem(job, 64) == "missing record checkpoint_64.json"
    assert s2load.pair_problem(job, 96) == "missing record checkpoint_96.json"
    _refused(job, 64, "missing record")


def test_a_truncated_zip_is_refused(saved, tmp_path):
    job = _copy(saved, tmp_path)
    path = s2load.checkpoint_files(job, 64)["model"]
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) * 2 // 3])
    assert s2load.pair_problem(job, 64) == f"model_64.zip has {len(whole) * 2 // 3} bytes, the record says {len(whole)}"
    _refused(job, 64, "bytes, the record says")
    # were the record cut with it (a pair written straight to a mount), the repository's own check still says no,
    # and so does a load for judging, whose zip test is the same without the .pkl being unpickled
    _reseal(job, 64)
    why = s2load.pair_problem(job, 64)
    assert why.startswith("bad/truncated checkpoint zip model_64.zip")
    assert _refused(job, 64, "bad/truncated checkpoint zip").endswith(why)
    assert _refused(job, 64, "bad/truncated checkpoint zip", env=_venv(), lr=LR).endswith(why)
    path.write_bytes(whole)
    _reseal(job, 64)
    stats = s2load.checkpoint_files(job, 64)["vecnormalize"]
    stats.write_bytes(stats.read_bytes()[:100])
    _reseal(job, 64)
    assert "does not unpickle" in s2load.pair_problem(job, 64)


def test_a_changed_record_is_refused(saved, tmp_path):
    job = _copy(saved, tmp_path)
    path = s2load.checkpoint_files(job, 64)["record"]
    doc = s2io.read_json(path)
    s2io.write_json(path, dict(doc, run_uuid="another"))                  # edited by hand: its own hash says so
    assert "does not hash to its own sha256" in s2load.pair_problem(job, 64)
    path.write_text(json.dumps(doc)[:200])
    assert "is not JSON" in s2load.pair_problem(job, 64)
    old = dict({k: v for k, v in doc.items() if k != "sha256"}, schema="s2-ckpt-1")
    s2io.write_json(path, dict(old, sha256=s2io.sha256_json(old)))
    assert "schema 's2-ckpt-1'" in s2load.pair_problem(job, 64)
    shutil.copy(s2load.checkpoint_files(job, 32)["record"], path)         # another step's record under this name
    assert s2load.pair_problem(job, 64) == "record checkpoint_64.json is of step 32"
    # a record whose own hash is right but which names another file, or does not list the three
    other = _copy(saved, tmp_path, "names")
    entries = s2io.read_json(s2load.checkpoint_files(other, 64)["record"])["files"]
    _reseal(other, 64, files=dict(entries, model=dict(entries["model"], name="model_32.zip")))
    assert s2load.pair_problem(other, 64) == "record names 'model_32.zip' as its model, not model_64.zip"
    _refused(other, 64, "record names 'model_32.zip' as its model")
    with pytest.raises(CheckpointError, match="record names 'model_32.zip'.*not packed"):
        s2load.pack(other, 64, tmp_path / "out")
    _reseal(other, 64, files={k: v for k, v in entries.items() if k != "vecnormalize"})
    assert s2load.pair_problem(other, 64) == "record checkpoint_64.json does not list model, vecnormalize, fingerprint"
    _refused(other, 64, "does not list model, vecnormalize, fingerprint")


def test_a_foreign_run_is_refused(saved):
    said = _refused(saved.job, 64, "run_uuid", expect_run=dict(RUN, run_uuid="11111111-2222-4333-8444-555566667777"))
    assert RUN["run_uuid"] in said and "11111111-2222" in said and "not judged" in said
    said = _refused(saved.job, 64, "settings_sha256", expect_run=dict(RUN, settings_sha256="cd" * 32))
    assert "not judged" in said
    assert s2load.load_checkpoint(saved.job, 64, expect_run={"run_uuid": RUN["run_uuid"],
                                                             "settings_sha256": RUN["settings_sha256"]})["model"]


def _perturb(job, n, by: float) -> None:
    path = s2load.checkpoint_files(job, n)["fingerprint"]
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["actions"] = arrays["actions"].copy()
    arrays["actions"][17, 3] += np.float32(by)
    s2io.write_npz(path, arrays)
    _reseal(job, n)


def test_a_wrong_fingerprint_is_refused(saved, tmp_path):
    job = _copy(saved, tmp_path)
    _perturb(job, 64, 1e-3)
    assert s2load.pair_problem(job, 64) is None                          # every file is intact: only the load can tell
    with pytest.raises(CheckpointError, match="fingerprint not reproduced.*not judged") as err:
        s2load.load_checkpoint(job, 64)
    assert err.value.fingerprint_max_abs_diff == pytest.approx(1e-3, rel=1e-3)
    with pytest.raises(CheckpointError, match="fingerprint not reproduced"):
        s2load.load_checkpoint(job, 64, env=_venv(), lr=LR)               # a resume checks it too (K4)
    done = _cli("check", job, 64)
    assert done.returncode == 1 and "NOT USABLE" in done.stdout and "fingerprint not reproduced" in done.stdout
    # inside the tolerance the difference is reported, not hidden
    _perturb(job, 32, 4e-6)
    ck = s2load.load_checkpoint(job, 32)
    assert 0.0 < ck["fingerprint_max_abs_diff"] < 1e-5
    with pytest.raises(CheckpointError, match="fingerprint not reproduced"):
        s2load.load_checkpoint(job, 32, tol=1e-6)
    # observations other than the seeded table are refused before any load
    path = s2load.checkpoint_files(job, 32)["fingerprint"]
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["obs"] = arrays["obs"] * np.float32(0.5)
    s2io.write_npz(path, arrays)
    _reseal(job, 32)
    _refused(job, 32, "do not hash to the record's obs_sha256")
    # a difference that is no number is not "within the tolerance"
    nan = _copy(saved, tmp_path, "nan")
    _perturb(nan, 64, float("nan"))
    with pytest.raises(CheckpointError, match="fingerprint not reproduced: max .difference. nan > 1e-05") as err:
        s2load.load_checkpoint(nan, 64)
    assert math.isnan(err.value.fingerprint_max_abs_diff)
    with pytest.raises(CheckpointError, match="fingerprint not reproduced"):
        s2load.load_checkpoint(nan, 64, env=_venv(), lr=LR)


def _set(index, value):
    def change(array):
        array[index] = value(array[index]) if callable(value) else value
        return array
    return change


def test_a_changed_statistic_or_weight_is_refused_by_the_fingerprint(saved, tmp_path):
    """The stored actions untouched, every hash right: what changed is what the policy is.  The fingerprint goes
    through the saved statistics and the saved weights, so each of these is "not judged" (S2-12, K9)."""
    def diff(job) -> float:
        with pytest.raises(CheckpointError, match="fingerprint not reproduced.*not judged") as err:
            s2load.load_checkpoint(job, 64, expect_run=RUN)
        return err.value.fingerprint_max_abs_diff

    seen = {}
    for name, key, change in (("mean", "obs_mean", _set(0, lambda x: x + 1e-2)),
                              ("var", "obs_var", _set(5, lambda x: x + 0.1)),
                              ("clip", "clip_obs", _set((), 1.0)),
                              ("epsilon", "epsilon", _set((), 1e-2))):
        job = _copy(saved, tmp_path, name)
        _edit_fingerprint(job, 64, **{key: change})
        assert s2load._inference_problem(job, 64) is None                # every file is intact
        seen[name] = diff(job)
    assert all(1e-5 < d < 0.1 for d in seen.values()), seen
    # a weight: the archive saved again from an inference load (as it stands: reproduced), then with one output
    # weight moved by 1e-3
    job = _copy(saved, tmp_path, "weight")
    path = s2load.checkpoint_files(job, 64)["model"]
    model = s2load.load_model(path, LAYOUT)
    model.save(path)
    _reseal(job, 64)
    assert s2load.load_checkpoint(job, 64)["fingerprint_max_abs_diff"] == 0.0
    with torch.no_grad():
        model.policy.action_net.weight[0, 0] += 1e-3
    model.save(path)
    _reseal(job, 64)
    assert 1e-5 < diff(job) < 1e-2


def test_a_fingerprint_file_that_does_not_read_is_refused(saved, tmp_path):
    """A file that was already cut when its record hashed it: the hashes agree, numpy cannot read it, and the
    checkpoint is refused by rule (``CheckpointError``), not by a traceback out of numpy or zipfile."""
    whole = s2load.checkpoint_files(saved.job, 64)["fingerprint"].read_bytes()
    plain_npy = io.BytesIO()
    np.save(plain_npy, np.zeros(3))
    for name, payload in (("half", whole[: len(whole) // 2]), ("nearly", whole[: len(whole) * 98 // 100]),
                          ("empty", b""), ("noise", b"\x00" * 4096), ("npy", plain_npy.getvalue())):
        job = _copy(saved, tmp_path, name)
        s2load.checkpoint_files(job, 64)["fingerprint"].write_bytes(payload)
        _reseal(job, 64)
        _refused(job, 64, "checkpoint 64 of .*fingerprint_64.npz does not read")
        _refused(job, 64, "checkpoint 64 of .*fingerprint_64.npz does not read", env=_venv(), lr=LR)
        done = _cli("check", job, 64) if name == "half" else None
        assert done is None or (done.returncode == 1 and "NOT USABLE" in done.stdout and "Traceback" not in done.stderr)
    # an array that is not there, or of another shape
    job = _copy(saved, tmp_path, "lacking")
    _edit_fingerprint(job, 64, drop=("epsilon",))
    _refused(job, 64, "checkpoint 64 of .*fingerprint_64.npz lacks epsilon")
    _edit_fingerprint(job, 32, drop=("actions", "obs_count"))
    _refused(job, 32, "checkpoint 32 of .*fingerprint_32.npz lacks actions, obs_count")
    job = _copy(saved, tmp_path, "shape")
    _edit_fingerprint(job, 64, actions=lambda a: a[:, :-1])
    _refused(job, 64, r"actions \(256, 16\); the layout asks \(256, 140\) and \(256, 17\)")


def test_an_archive_that_does_not_load_is_refused(saved, tmp_path):
    """A clean zip with every hash right whose weights are not the policy's: whatever the library raises on it, the
    checkpoint is refused by rule."""
    job = _copy(saved, tmp_path)
    path = s2load.checkpoint_files(job, 64)["model"]
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(buf, "w") as zout:
        for item in zin.infolist():
            payload = zin.read(item.filename)
            if item.filename == "policy.pth":
                state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                state["action_net.weight"] = state["action_net.weight"][:, :-1].clone()     # another layer width
                out = io.BytesIO()
                torch.save(state, out)
                payload = out.getvalue()
            zout.writestr(item, payload)
    path.write_bytes(buf.getvalue())
    _reseal(job, 64)
    assert s2load.pair_problem(job, 64) is None                          # the zip and the .pkl are whole
    said = _refused(job, 64, "checkpoint 64 of .*: does not load \\(RuntimeError: ")
    assert "action_net.weight" in said
    venv = _venv()
    _refused(job, 64, "does not load", env=venv, lr=LR)
    assert [(e.resets, e.steps) for e in venv.envs] == [(0, 0), (0, 0)]


def test_statistics_that_are_not_the_pkls_are_refused_on_a_resume(saved, tmp_path):
    job = _copy(saved, tmp_path)
    path = s2load.checkpoint_files(job, 64)["fingerprint"]
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["obs_mean"] = arrays["obs_mean"] + 1e-9
    s2io.write_npz(path, arrays)
    _reseal(job, 64)
    with pytest.raises(CheckpointError, match="statistics are not the fingerprint file's .obs_mean."):
        s2load.load_checkpoint(job, 64, env=_venv(), lr=LR)


def test_newest_intact_skips_a_broken_newest_checkpoint_and_says_why(saved, tmp_path):
    job = _copy(saved, tmp_path)
    assert s2load.newest_intact(job) == (64, [])
    path = s2load.checkpoint_files(job, 64)["model"]
    raw = bytearray(path.read_bytes())
    raw[1000] ^= 0xFF
    path.write_bytes(bytes(raw))
    n, skipped = s2load.newest_intact(job)
    assert n == 32 and [s["steps"] for s in skipped] == [64]
    assert skipped[0]["reason"].startswith("model_64.zip has sha256 ")
    # a save that was cut before its record: its binaries are there, the step is reported and passed over
    (job / "checkpoints" / "model_96.zip").write_bytes(b"PK")
    (job / "checkpoints" / "vecnormalize_96.pkl.tmp").write_bytes(b"")
    n, skipped = s2load.newest_intact(job)
    assert n == 32 and [s["steps"] for s in skipped] == [96, 64]
    assert skipped[0]["reason"] == "no record checkpoint_96.json (a save that was cut)"
    s2load.checkpoint_files(job, 32)["vecnormalize"].unlink()
    n, skipped = s2load.newest_intact(job)
    assert n is None and [s["steps"] for s in skipped] == [96, 64, 32]
    assert skipped[2]["reason"] == "missing file vecnormalize_32.pkl"
    assert s2load.newest_intact(tmp_path / "no job") == (None, [])


def test_remove_checkpoint(saved, tmp_path):
    job = _copy(saved, tmp_path)
    gone = s2load.remove_checkpoint(job, 32)
    assert gone[0] == "checkpoint_32.json"
    assert sorted(gone) == sorted(s2load.FILE_NAMES[k] % 32 for k in s2load.FILE_NAMES)
    assert s2load.list_checkpoints(job) == [64] and s2load.newest_intact(job) == (64, [])
    assert s2load.remove_checkpoint(job, 32) == []


# ------------------------------------------------------------------------------------------------
# The tar and its import
# ------------------------------------------------------------------------------------------------
def test_pack_gives_identical_bytes_twice(saved, tmp_path):
    a, b = s2load.pack(saved.job, 64, tmp_path / "a"), s2load.pack(saved.job, 64, tmp_path / "b")
    one, two = Path(a["tar"]), Path(b["tar"])
    assert one.name == two.name == "ckpt_64.tar" and one.read_bytes() == two.read_bytes()
    assert a["sha256"] == b["sha256"] == s2io.sha256_file(one) and a["bytes"] == one.stat().st_size
    assert list((tmp_path / "a").iterdir()) == [one]                     # no temporary file left
    rec = saved.records[64]
    assert a["members"] == sorted(({"name": rec["files"][k]["name"], "bytes": rec["files"][k]["bytes"],
                                    "sha256": rec["files"][k]["sha256"]} for k in s2load.FILE_KINDS),
                                  key=lambda m: m["name"])
    assert a["record"] == {"name": "checkpoint_64.json",
                           "sha256": s2io.sha256_file(s2load.checkpoint_files(saved.job, 64)["record"])}
    # the bundle's tar rules: USTAR, sorted names, mtime 0, uid and gid 0, no names, mode 0644, no compression
    with tarfile.open(one, "r:") as tar:
        infos = tar.getmembers()
    raw = one.read_bytes()
    assert raw[257:265] == b"ustar\x0000" and b"PaxHeader" not in raw and b"././@LongLink" not in raw
    assert [i.name for i in infos] == ["fingerprint_64.npz", "model_64.zip", "vecnormalize_64.pkl"]
    assert all((i.mtime, i.uid, i.gid, i.uname, i.gname, i.mode, i.isreg()) == (0, 0, 0, "", "", 0o644, True)
               for i in infos)
    assert one.stat().st_size % tarfile.RECORDSIZE == 0 and one.stat().st_size < sum(i.size for i in infos) + 20480
    # a later copy of the same files (other inode, other times) packs to the same bytes
    job = _copy(saved, tmp_path)
    assert Path(s2load.pack(job, 64, tmp_path / "c")["tar"]).read_bytes() == one.read_bytes()
    # a checkpoint that is not intact is not packed
    s2load.checkpoint_files(job, 32)["fingerprint"].write_bytes(b"x")
    with pytest.raises(CheckpointError, match="not packed"):
        s2load.pack(job, 32, tmp_path / "c")
    assert not (tmp_path / "c" / "ckpt_32.tar").exists()


def test_import_tar_round_trips(saved, tmp_path):
    packed = s2load.pack(saved.job, 64, tmp_path / "out")
    record = s2load.checkpoint_files(saved.job, 64)["record"]
    into = tmp_path / "here" / "jobs" / "imported"
    got = s2load.import_tar(record, packed["tar"], into)
    assert got["ok"] is True and got["reason"] is None and got["schema"] == "s2-ckpt-import-1"
    assert got["fingerprint_max_abs_diff"] == 0.0 and got["tol"] == 1e-5 and got["cross_version"] is False
    assert got["record_sha256"] == s2io.sha256_file(record) and got["files"] == saved.records[64]["files"]
    assert got["versions_saved"] == saved.records[64]["versions"] and got["versions_here"] == s2load.versions()
    assert got["tar"] == {"name": "ckpt_64.tar", "bytes": packed["bytes"], "sha256": packed["sha256"]}
    assert (got["steps"], got["run_uuid"], got["job"], got["settings_sha256"]) == (
        64, RUN["run_uuid"], RUN["job"], RUN["settings_sha256"])
    assert got["rules"] == {"fingerprint": s2load.FINGERPRINT_RULE, "loader": s2load.LOADER_RULE}
    theirs, mine = s2load.checkpoint_files(saved.job, 64), s2load.checkpoint_files(into, 64)
    assert all(mine[k].read_bytes() == theirs[k].read_bytes() for k in mine)
    assert sorted(p.name for p in (into / "checkpoints").iterdir()) == sorted(p.name for p in mine.values())
    assert s2load.newest_intact(into) == (64, []) and s2load.load_checkpoint(into, 64, expect_run=RUN)["model"]
    assert s2load.import_tar(record, packed["tar"], into)["ok"] is True  # again, over itself
    own = s2load.checkpoint_files(into, 64)["record"]                     # ... and with the record that stands there
    assert s2load.import_tar(own, packed["tar"], into)["ok"] is True and s2load.newest_intact(into) == (64, [])
    json.dumps(s2io.plain(got))                                           # the validation record is JSON as it stands


def test_a_checkpoint_saved_under_other_versions_is_flagged(saved, tmp_path):
    job = _copy(saved, tmp_path)
    here = s2load.versions()
    colab = dict(here, python="3.12.11", torch=here["torch"].split("+")[0] + "+cu126")
    _reseal(job, 64, versions=colab)
    ck = s2load.load_checkpoint(job, 64)
    assert ck["cross_version"] is True and ck["versions_saved"] == colab and ck["versions_here"] == here
    assert ck["fingerprint_max_abs_diff"] == 0.0                          # flagged, and judged by its fingerprint
    packed = s2load.pack(job, 64, tmp_path / "out")
    got = s2load.import_tar(s2load.checkpoint_files(job, 64)["record"], packed["tar"], tmp_path / "here")
    assert got["ok"] is True and got["cross_version"] is True and got["versions_saved"] == colab
    _reseal(job, 32, versions=dict(here, torch=here["torch"].split("+")[0] + "+cu126"))
    assert s2load.load_checkpoint(job, 32)["cross_version"] is False     # only torch's build label differs


def _tar(path: Path, members) -> Path:
    """A plain tar of ``members``: a dict name -> bytes, or a list of (name, bytes) pairs (a name may then repeat)."""
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, payload in (members.items() if isinstance(members, dict) else members):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def test_import_tar_refuses_what_is_not_the_records_checkpoint(saved, tmp_path):
    files = s2load.checkpoint_files(saved.job, 64)
    good = {files[k].name: files[k].read_bytes() for k in s2load.FILE_KINDS}

    def failed(members, said: str, name: str) -> dict:
        into = tmp_path / name
        got = s2load.import_tar(files["record"], _tar(tmp_path / f"{name}.tar", members), into)
        assert got["ok"] is False and said in got["reason"], got["reason"]
        assert got["fingerprint_max_abs_diff"] is None
        assert not into.exists() or list((into / "checkpoints").iterdir()) == []      # nothing left behind
        return got

    changed = dict(good)
    changed["model_64.zip"] = good["model_64.zip"][:-1] + bytes([good["model_64.zip"][-1] ^ 1])
    failed(changed, "model_64.zip has", "changed")
    failed(dict(good, **{"extra.bin": b"x"}), "holds ['extra.bin', 'fingerprint_64.npz'", "extra")
    failed({k: v for k, v in good.items() if "pkl" not in k}, "the record lists", "lacking")
    failed({("../" + k if "npz" in k else k): v for k, v in good.items()}, "is not a plain file name", "leaving")
    failed({("sub/" + k if "npz" in k else k): v for k, v in good.items()}, "is not a plain file name", "nested")
    # the same name twice: the right three names, and a second model_64.zip that would be unpacked over the first
    failed(list(good.items()) + [("model_64.zip", changed["model_64.zip"])], "member 'model_64.zip' twice", "twice")
    failed([("model_64.zip", changed["model_64.zip"])] + list(good.items()), "member 'model_64.zip' twice", "twice2")
    got = s2load.import_tar(files["record"], tmp_path / "absent.tar", tmp_path / "absent")
    assert got["ok"] is False and "missing tar" in got["reason"]
    (tmp_path / "junk.tar").write_bytes(b"not a tar at all" * 100)
    got = s2load.import_tar(files["record"], tmp_path / "junk.tar", tmp_path / "junk")
    assert got["ok"] is False and "not a readable plain tar" in got["reason"]
    with pytest.raises(CheckpointError, match="missing record"):          # without a record there is nothing to check
        s2load.import_tar(tmp_path / "checkpoint_64.json", _tar(tmp_path / "g.tar", good), tmp_path / "g")
    # a failed import does not take down a good checkpoint that stood there
    into = tmp_path / "standing"
    assert s2load.import_tar(files["record"], _tar(tmp_path / "good.tar", good), into)["ok"] is True
    assert s2load.import_tar(files["record"], _tar(tmp_path / "bad.tar", changed), into)["ok"] is False
    assert s2load.newest_intact(into) == (64, [])


def test_an_import_whose_fingerprint_fails_says_so(saved, tmp_path):
    job = _copy(saved, tmp_path)
    _perturb(job, 64, 2e-3)
    packed = s2load.pack(job, 64, tmp_path / "out")
    got = s2load.import_tar(s2load.checkpoint_files(job, 64)["record"], packed["tar"], tmp_path / "here")
    assert got["ok"] is False and "fingerprint not reproduced" in got["reason"]
    assert got["fingerprint_max_abs_diff"] == pytest.approx(2e-3, rel=1e-3)


def test_the_cli_packs_imports_and_writes_the_validation_record(saved, tmp_path):
    done = _cli("pack", saved.job, 64, "--out", tmp_path / "out")
    assert done.returncode == 0, done.stdout + done.stderr[-2000:]
    tar = tmp_path / "out" / "ckpt_64.tar"
    assert f"S2 CHECKPOINT 64: packed {tar} " in done.stdout
    assert tar.read_bytes() == Path(s2load.pack(saved.job, 64, tmp_path / "again")["tar"]).read_bytes()
    record, out = s2load.checkpoint_files(saved.job, 64)["record"], tmp_path / "validation" / "colab_archive.json"
    done = _cli("import", "--record", record, "--tar", tar, "--into", tmp_path / "here", "--write", out)
    assert done.returncode == 0, done.stdout + done.stderr[-2000:]
    assert "S2 CHECKPOINT 64: imported into" in done.stdout and "intact; fingerprint 0.0 " in done.stdout
    doc = s2io.read_json(out)
    assert doc["ok"] is True and doc["fingerprint_max_abs_diff"] == 0.0 and doc["cross_version"] is False
    assert doc["sha256"] == s2io.sha256_json({k: v for k, v in doc.items() if k != "sha256"})
    assert set(doc["provenance"]) == {"commit", "dirty", "dirty_paths"}
    assert s2load.newest_intact(tmp_path / "here") == (64, [])
    # a failed import still writes its record and exits 1
    bad = tmp_path / "bad.tar"
    bad.write_bytes(tar.read_bytes()[:-20000])
    done = _cli("import", "--record", record, "--tar", bad, "--into", tmp_path / "there", "--write", out)
    assert done.returncode == 1 and "IMPORT FAILED" in done.stdout
    assert s2io.read_json(out)["ok"] is False
    done = _cli("pack", saved.job, 96, "--out", tmp_path / "out")
    assert done.returncode == 1 and "NOT PACKED" in done.stdout and "missing record" in done.stdout


_CHECKPOINTS = "stage2/tb/work/run/jobs/R5-x-s42/checkpoints/"


def _git_ignored(paths) -> list:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    done = subprocess.run(["git", "check-ignore", *paths], cwd=STUDY, capture_output=True, text=True)
    if done.returncode not in (0, 1):
        pytest.skip(f"not a git checkout: {done.stderr.strip()}")
    return done.stdout.split()


def test_a_checkpoints_binaries_are_ignored_by_git():
    binaries = [_CHECKPOINTS + s2load.FILE_NAMES[k] % 250000 for k in s2load.FILE_KINDS]
    assert _git_ignored(binaries) == binaries     # the repository's own rules (*.zip, *.pkl, *.npz), at any depth
    record = ["stage2/tb/validation/colab_archive.json"]                  # a committed record is not ignored
    assert _git_ignored(record) == []


def test_the_record_the_temporary_names_and_the_tar_are_ignored_by_git():
    """Every other path this module writes that is no committed record.  The rules that cover them are Track B's own
    ``stage2/tb/.gitignore``, which is the launcher package's file (CB), not this one's: without it this test is
    reported as skipped, never as passed, and it asserts as soon as that file is in the checkout."""
    rest = [_CHECKPOINTS + "checkpoint_250000.json", _CHECKPOINTS + "checkpoint_250000.json.tmp"]
    rest += [_CHECKPOINTS + s2load.FILE_NAMES[k] % 250000 + ".tmp" for k in s2load.FILE_KINDS]
    rest += ["stage2/tb/work/run/ckpt_250000.tar", "stage2/tb/work/run/ckpt_250000.tar.tmp",
             "stage2/tb/validation/colab_archive.json.tmp"]
    if not (STUDY / "stage2" / "tb" / ".gitignore").is_file():
        covered = _git_ignored(rest)
        missing = [p for p in rest if p not in covered]
        pytest.skip(f"stage2/tb/.gitignore (work package CB) is not in this checkout: git does not yet ignore "
                    f"{len(missing)} of the {len(rest)} paths ({', '.join(missing)})")
    assert _git_ignored(rest) == rest


def test_the_scratch_helper_assigns_mujoco_gl_itself(tmp_path):
    # the acceptance commands' scratch checkpoint: started under OSMesa, it still builds its policy with no GL library
    env = tbenv.child_env("disable")
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path / "job")], cwd=STUDY, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr[-2000:]
    said = done.stdout.strip().splitlines()[-1]
    assert said.startswith(f"scratch checkpoints [32, 64] in {tmp_path / 'job'}; MUJOCO_GL=disable, mujoco imported, ")
    if Path("/proc/self/maps").is_file():
        assert said.endswith(", no GL library")
    assert s2load.newest_intact(tmp_path / "job") == (64, [])


def _gl_libraries():
    """The OpenGL libraries mapped into this process (Linux: ``/proc/self/maps``), or None where that cannot be read."""
    try:
        with open("/proc/self/maps") as fh:
            return sorted({line.rsplit("/", 1)[-1].strip() for line in fh
                           if any(lib in line for lib in ("libOSMesa", "libEGL", "libGL.", "libGLX", "libglfw"))})
    except OSError:
        return None


if __name__ == "__main__":                        # python tests/test_stage2_s2load.py JOBDIR: a scratch job for the CLI
    import os

    tbenv.assign_gl("disable")                    # this process builds a policy; nothing has imported mujoco yet
    _made = scratch_checkpoints(sys.argv[1])
    _gl = _gl_libraries()
    print(f"scratch checkpoints {sorted(_made.records)} in {_made.job}; MUJOCO_GL={os.environ['MUJOCO_GL']}, "
          f"mujoco {'imported' if 'mujoco' in sys.modules else 'not imported'}, "
          + ("GL libraries not read" if _gl is None else "GL libraries " + ", ".join(_gl) if _gl else "no GL library"))
