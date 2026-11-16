"""``designlab.stage2.train2``: the trainer of R5 and R6 (round C design 3.6 and section 4's rows "Kill and resume",
"Lesson 16", "``MUJOCO_GL``"; Track B rules K1-K5, M1, M4, O3).

What is held here, on a stub env and a stub monitor (no plant, no reference set):

* **kill and resume**: a run killed with SIGKILL between two boundaries and resumed with another worker count ends
  with the uninterrupted run's policy tensors, optimizer state, normalisation statistics (their count included,
  also at the first checkpoint after the resume) and fingerprint, bit for bit; so does a run killed while its
  monitor child runs (the attempt is logged before the child starts and is an interruption), a run killed after
  G8b's fix and before the next checkpoint (the fix is recorded once), a run whose ``run.json`` is ahead of its
  newest intact checkpoint (later entries move to ``discarded_after_resume``; ``updates.jsonl`` holds no step
  twice), and a job that has a ``run.json`` and no checkpoint (it restarts from step 0 in place);
* **what a resume refuses**, in the rule's order: no ``run.json``, a stopped run, other settings, other inputs (a
  changed file refused; a later freeze of the set with unchanged files accepted, both freezes recorded), other
  sources and other library versions (each refused without its flag, accepted and flagged with it), a checkpoint
  of another run; a broken newest checkpoint is skipped with its reason;
* **a check's attempts** (rule M1): logged before the monitor starts and closed with its exit code; after two closed
  attempts without a record the check counts as a failed G8b reading; after two attempts that were never closed
  the run halts;
* **lesson 16**: with a learning rate that trips ``target_kl`` ``minibatch_steps_run < minibatch_steps_possible``,
  and the ``kl_cut`` flag is set from the median share over a check interval;
* **``MUJOCO_GL``**: a trainer started with ``MUJOCO_GL=osmesa`` runs a rollout with ``disable`` in itself and in
  its worker, and the worker holds no torch; an archive the trainer saves has no bytecode member; the schedule's
  value follows the step count across a resume;
* the probes on the stub (rule O3's stops at the first rate that passes both sigmas; rule O4's sigma).

Slow (``STAGE2_SLOW=1``; the Stage 1 view ``S1`` of 75-M-H-10, 2 workers, minutes): a tiny R5 run killed after its
first checkpoint and resumed to its budget, equal to the uninterrupted tiny run; a tiny R6 run killed and resumed.

The stub env, the stub seams and the stub monitor are defined here and named to the workers and to the trainer's
children as ``tests.test_stage2_train2``: a child's working directory is the study directory.  This module
therefore imports neither torch nor stable-baselines3 at its top.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

STUDY = Path(__file__).resolve().parents[1]
HERE = "tests.test_stage2_train2"
PY = sys.executable
BODY = "75-M-H-10"
slow = pytest.mark.skipif(not os.environ.get("STAGE2_SLOW"), reason="real env on the S1 view; set STAGE2_SLOW=1")

# ---------------------------------------------------------------------------------------------------------------
# The stub env (imported by the workers under the name tests.test_stage2_train2)
# ---------------------------------------------------------------------------------------------------------------
STUB_LAYOUT = dict(blocks={"q": (0, 4), "qd": (4, 8), "root_vel": (8, 11), "height": (11, 12)}, actor_dim=8,
                   obs_dim=12, action_dim=3, nj=4, heading=False, head_x=False)
STUB_TASKS = dict(stand=1.0, walk_nominal=2.0, **{"walk_v0.4": 3.0}, turn_q90_left=1.5, turn_q90_right=1.5,
                  stop=2.0, start_stop=1.0)


class StubEnv(gymnasium.Env):
    """12 observations (the first 8 the actor's), 3 actions.  Everything an episode draws comes from the env's own
    generator, rebuilt whole at a seeded reset (history-free); an episode plays a task drawn by the task weights,
    lasts 5 to 40 steps and ends terminated or truncated.  ``describe`` reads a small JSON "set file" each time
    it is called, so a test can freeze the set again or change a file's hash between two sessions."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict):
        if os.environ.get("STUB_INPUT_ERROR"):  # (as the real env refuses its inputs when it is built, rule K4)
            from designlab.stage1 import tune

            raise tune.InputError("stub: a used file is not the one the named freeze holds")
        self.cfg = dict(cfg)
        self.observation_space = spaces.Box(-np.inf, np.inf, (STUB_LAYOUT["obs_dim"],), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (STUB_LAYOUT["action_dim"],), np.float32)
        self.tasks = list(cfg["tasks"])
        self._mix = np.random.default_rng(7).normal(size=(12, 3))
        self._rng = np.random.default_rng(1000 + int(cfg["tag"]))
        self._built = False
        self.set_task_weights(cfg.get("task_weights") or [STUB_TASKS[t] for t in self.tasks])
        self._built = True
        self._t, self._n, self._end, self._x, self.task = 0, 1, "terminated", np.zeros(12), self.tasks[0]

    def set_task_weights(self, w) -> None:
        if isinstance(w, dict):
            cur = dict(zip(self.tasks, getattr(self, "task_w", np.ones(len(self.tasks))), strict=True))
            w = [float(w.get(t, cur[t])) for t in self.tasks]
        self.task_w = np.asarray(w, float)
        if self._built and os.environ.get("STUB_WEIGHTS_TRIP") and self.cfg["tag"] == 5:
            raise ValueError("stub env 5 trips in set_task_weights")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.task = str(self._rng.choice(self.tasks, p=self.task_w / self.task_w.sum()))
        self._t = 0
        self._n = int(self._rng.integers(5, 41))
        self._end = "terminated" if self._rng.random() < 0.5 else "truncated"
        self._x = self._rng.normal(size=12) + 0.3 * self.tasks.index(self.task)
        return self._x.astype(np.float32), {"task": self.task, "seed": seed}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        self._t += 1
        self._x = 0.9 * self._x + self._mix @ a + 0.05 * self._rng.normal(size=12)
        reward = float(np.tanh(self._x[0]) - 0.1 * (a @ a) + 0.05 * self.tasks.index(self.task))
        done = self._t >= self._n
        return (self._x.astype(np.float32), reward, bool(done and self._end == "terminated"),
                bool(done and self._end == "truncated"), {"task": self.task})

    def describe(self) -> dict:
        doc = json.loads(Path(self.cfg["set_file"]).read_text())
        named = self.cfg.get("set_freeze")
        started = next((f for f in doc["freezes"] if f["manifest_sha256"] == named), None)
        return dict(config=dict(self.cfg), layout=dict(STUB_LAYOUT), task_names=list(self.tasks),
                    tasks={t: dict(plan=[["hold", 1.0]], weight=float(STUB_TASKS[t]), available=True)
                           for t in self.tasks},
                    task_files=doc["files"], inputs=dict(files=doc["files"], plant_inputs=doc["plant_inputs"]),
                    set_freeze_started=started, set_freeze_now=doc["freezes"][-1])

    def process_cpu(self):
        return os.getpid(), time.process_time()

    def where(self) -> dict:
        return dict(pid=os.getpid(), mujoco_gl=os.environ.get("MUJOCO_GL"), torch="torch" in sys.modules,
                    argv=list(sys.orig_argv), weights=self.task_w.tolist())


def make_stub(cfg: dict) -> StubEnv:
    return StubEnv(cfg)


# ---------------------------------------------------------------------------------------------------------------
# The stub seams (installed in the trainer's process: this pytest process, or a ``drive`` child)
# ---------------------------------------------------------------------------------------------------------------
def write_set(path, freezes=("f0",), files=None) -> None:
    files = files or {"refs/nominal.npz": dict(sha256="a" * 64, entry_sha256="b" * 64),
                      "transitions/start.npz": dict(sha256="c" * 64, entry_sha256="d" * 64)}
    Path(path).write_text(json.dumps(dict(
        freezes=[dict(index=i, manifest_sha256=f) for i, f in enumerate(freezes)], files=files,
        plant_inputs=dict(mjcf="e" * 64))))


def stub_configs(mode, body, set_name, overrides) -> list:
    path = os.environ["STUB_SET_FILE"]
    latest = json.loads(Path(path).read_text())["freezes"][-1]["manifest_sha256"]
    return [dict(dict(tag=i, body=body, set=set_name, mode=mode, floor="hard" if i % 4 < 3 else "carpet",
                      tasks=list(STUB_TASKS), task_weights=None, set_freeze=latest, set_file=path, filter_hz=10.0),
                 **dict(overrides or {})) for i in range(8)]


def stub_specs(mode, cfgs) -> list:
    return [(f"{HERE}:make_stub", {"cfg": dict(c)}) for c in cfgs]


def stub_g8a(body, set_name, tiny) -> dict:
    ok = os.environ.get("STUB_G8A_FAIL") != "1"
    return dict(form="stub", body=body, set=set_name, episodes=[], identical=2 if ok else 1, of=2, passes=ok)


def stub_monitor_cmd() -> list:
    return [PY, "-m", HERE, "monitor"]


def stub_validation(body, set_name) -> dict:
    return dict(lr_probe=None, filter=None)


def install(module=None):
    """Put the stub seams on ``train2`` (in this process)."""
    from designlab.stage2 import train2

    t = module or train2
    t._env_configs, t._env_specs, t._g8a_short = stub_configs, stub_specs, stub_g8a
    t._monitor_cmd, t._validation = stub_monitor_cmd, stub_validation
    return t


def _steps_env(name: str) -> list:
    return [int(x) for x in os.environ.get(name, "").split(",") if x]


def stub_episodes(body, set_name, row, pairs, policies=None, traces=None) -> list:
    """The stub of ``lookbar.episodes``: R0L reads every episode (survival 10 s); a policy row reads them too and
    survives 10.5 s, but ``walk/nominal`` falls at the steps of ``STUB_MON_FAIL_AT``."""
    n = None if not policies else int(next(iter(policies.values()))["steps"])
    down = n is not None and n in _steps_env("STUB_MON_FAIL_AT")
    out = []
    for cond, seed in pairs:
        fell = bool(down and cond == "walk/nominal")
        surv = 10.0 if n is None else (3.0 if fell else 10.5)
        out.append(dict(condition=cond, seed=int(seed), outcome="fell" if fell else "read", reason=None, fell=fell,
                        survival_s=surv, planned_s=12.0, succeeded=False, succeeded_but_A1=False,
                        succeeded_look=False, succeeded_but_A1_look=False, failed_terms=["tail_servo"], look=None,
                        clock=dict(mean_factor=1.0, min_factor=1.0, t_ref_reached=1.05 * surv,
                                   t_ref_planned=1.05 * 12.0), error=None))
    return out


def _monitor_main(argv) -> int:
    """The stub monitor child: ``monitor2``'s own CLI with ``_episodes`` replaced.  ``STUB_MON_CRASH`` names a file
    holding a count: while it is above zero the child takes one off and leaves with code 7 and no record.
    ``STUB_MON_HOLD`` names a file: while it exists, a ``run`` writes ``<file>.started`` and waits to be killed."""
    crash = os.environ.get("STUB_MON_CRASH")
    if crash and argv[:1] == ["run"] and Path(crash).is_file() and int(Path(crash).read_text() or 0) > 0:
        Path(crash).write_text(str(int(Path(crash).read_text()) - 1))
        return 7
    hold = os.environ.get("STUB_MON_HOLD")
    if hold and argv[:1] == ["run"] and Path(hold).is_file():
        Path(hold + ".started").write_text(str(os.getpid()))
        time.sleep(300)
        return 9
    from designlab.stage2 import monitor2

    monitor2._episodes = stub_episodes
    return monitor2.main(argv)


def _drive_main(argv) -> int:
    """A trainer on the stub in its own process: ``drive <JSON kwargs of train2.train>``.  ``STUB_KILL_AT=n``: the
    process kills itself with SIGKILL at the rollout start with n steps done, after everything that rollout start
    does (the checkpoint, the check, the boundary, ``progress.json``)."""
    t = install()
    kill_at = os.environ.get("STUB_KILL_AT")
    if kill_at:
        inner = t._Run._progress

        def progress(self, n, rate=None, last=None):
            inner(self, n, rate, last)
            if int(n) == int(kill_at) and self.cb is not None and self.session["ended_at"] is None:
                os.kill(os.getpid(), signal.SIGKILL)

        t._Run._progress = progress
    try:
        doc = t.train(**json.loads(argv[0]))
    except t.TrainError as exc:
        print(f"drive: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return exc.code
    print(json.dumps(dict(status=doc["status"], steps=doc["num_timesteps"])), flush=True)
    return 0


def _main(argv) -> int:
    return _monitor_main(argv[1:]) if argv[0] == "monitor" else _drive_main(argv[1:])


# ---------------------------------------------------------------------------------------------------------------
# Helpers of the tests
# ---------------------------------------------------------------------------------------------------------------
torch = None


def _need_torch():
    global torch
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")


#: The stub runs: rollouts of 8 x 16 steps, a checkpoint every 2 rollouts, a check every 4, 12 rollouts.
ROLLOUT = 128
KW = dict(mode="R5", body="stub", seed=42, set_name="STUB", steps=1536, checkpoint_every=256, monitor_every=512,
          ppo=dict(n_steps=16, batch_size=64), tiny=True)
FINAL = 1536


def _env(tmp_path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")}
    env.update(PYTHONPATH=".", MUJOCO_GL="disable", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
               STUB_SET_FILE=str(tmp_path / "set.json"))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _set_file(tmp_path, **kw) -> Path:
    path = tmp_path / "set.json"
    if not path.exists() or kw:
        write_set(path, **kw)
    return path


def drive(tmp_path, out, env=None, **kw) -> subprocess.CompletedProcess:
    """One trainer process on the stub; its own process group, so that a kill takes its children with it."""
    _set_file(tmp_path)
    args = dict(KW, out=str(out), **kw)
    return subprocess.run([PY, "-m", HERE, "drive", json.dumps(args)], cwd=STUDY, env=_env(tmp_path, **(env or {})),
                          capture_output=True, text=True, start_new_session=True, timeout=600)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """``train2`` with the stub seams in this process (envs in this process: workers 0 unless a test says)."""
    _need_torch()
    from designlab.stage2 import train2

    for name, fn in (("_env_configs", stub_configs), ("_env_specs", stub_specs), ("_g8a_short", stub_g8a),
                     ("_monitor_cmd", stub_monitor_cmd), ("_validation", stub_validation)):
        monkeypatch.setattr(train2, name, fn)
    for k in list(os.environ):
        if k.startswith("STUB_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("STUB_SET_FILE", str(_set_file(tmp_path)))
    monkeypatch.chdir(STUDY)  # (the stub factory is named tests.test_stage2_train2)
    return train2


def run(train2, out, **kw) -> dict:
    args = dict(KW, out=str(out), workers=0, gates=False, log=None)
    args.update(kw)
    return train2.train(**args)


class _Cut(Exception):
    pass


def cut_at(monkeypatch, train2, n: int) -> None:
    """Make the trainer of this process raise ``_Cut`` at the rollout start with ``n`` steps done (after its
    checkpoint, check and boundary): a session that ends early, in process."""
    inner = train2._Run._progress

    def progress(self, steps, rate=None, last=None):
        inner(self, steps, rate, last)
        if int(steps) == int(n) and self.session["ended_at"] is None:
            raise _Cut(f"cut at {steps}")

    monkeypatch.setattr(train2._Run, "_progress", progress)


def job(out, name="R5-stub-s42") -> Path:
    return Path(out) / "jobs" / name


def doc_of(out, name="R5-stub-s42") -> dict:
    return json.loads((job(out, name) / "run.json").read_text())


def tensors(job_dir, n: int) -> dict:
    """Everything a checkpoint holds that training made: policy and optimizer tensors, the normalisation
    statistics with their counts, the fingerprint's arrays."""
    ck = Path(job_dir) / "checkpoints"
    out = {}
    with zipfile.ZipFile(ck / f"model_{n}.zip") as z:
        for member in ("policy.pth", "policy.optimizer.pth"):
            state = torch.load(io.BytesIO(z.read(member)), map_location="cpu", weights_only=False)
            flat = {}

            def walk(prefix, x, flat=flat):
                if isinstance(x, dict):
                    for k, v in x.items():
                        walk(f"{prefix}/{k}", v)
                elif isinstance(x, (list, tuple)):
                    for i, v in enumerate(x):
                        walk(f"{prefix}/{i}", v)
                elif hasattr(x, "numpy"):
                    flat[prefix] = x.numpy().copy()
                else:
                    flat[prefix] = np.asarray(x)

            walk(member, state)
            out.update(flat)
    with open(ck / f"vecnormalize_{n}.pkl", "rb") as fh:
        vn = pickle.load(fh)
    out.update(obs_mean=vn.obs_rms.mean, obs_var=vn.obs_rms.var, obs_count=np.asarray(vn.obs_rms.count),
               ret_mean=np.asarray(vn.ret_rms.mean), ret_var=np.asarray(vn.ret_rms.var),
               ret_count=np.asarray(vn.ret_rms.count))
    with np.load(ck / f"fingerprint_{n}.npz") as z:
        out.update({f"fingerprint/{k}": z[k] for k in z.files})
    return out


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return a.tolist() == b.tolist()


def assert_same_checkpoint(a, b, n: int) -> None:
    x, y = tensors(a, n), tensors(b, n)
    assert sorted(x) == sorted(y)
    differ = [k for k in x if not _same(x[k], y[k])]
    assert not differ, f"checkpoint {n} differs in {differ[:5]} ({len(differ)} of {len(x)})"
    assert any(k.startswith("policy.optimizer.pth") for k in x) and "fingerprint/actions" in x


def updates(job_dir) -> list:
    return [json.loads(x) for x in (Path(job_dir) / "updates.jsonl").read_text().splitlines()]


def same_updates(a, b) -> None:
    drop = ("session", "wall_s")
    ua = [{k: v for k, v in u.items() if k not in drop} for u in updates(a)]
    ub = [{k: v for k, v in u.items() if k not in drop} for u in updates(b)]
    assert ua == ub


# ---------------------------------------------------------------------------------------------------------------
# The reference runs (uninterrupted; made once per module)
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref_fix(tmp_path_factory):
    """The uninterrupted run with gates: the stub monitor fails ``walk/nominal`` at the first check (512 steps),
    so G8b fails there, the one fix is made, and the second reading (1,024) passes.  Envs in the trainer."""
    _need_torch()
    tmp = tmp_path_factory.mktemp("ref_fix")
    r = drive(tmp, tmp / "out", env=dict(STUB_MON_FAIL_AT="512"), workers=0)
    assert r.returncode == 0, r.stderr[-3000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == dict(status="done", steps=FINAL)
    return tmp


@pytest.fixture(scope="module")
def ref_plain(tmp_path_factory):
    """The uninterrupted run without gates (no monitor child): envs in the trainer."""
    _need_torch()
    tmp = tmp_path_factory.mktemp("ref_plain")
    r = drive(tmp, tmp / "out", workers=0, gates=False)
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp


# ---------------------------------------------------------------------------------------------------------------
# The run's records
# ---------------------------------------------------------------------------------------------------------------
def test_the_uninterrupted_run_and_its_records(ref_fix):
    d = doc_of(ref_fix / "out")
    jd = job(ref_fix / "out")
    assert d["schema"] == "s2-run-1" and d["status"] == "done" and d["budget"] == FINAL == d["num_timesteps"]
    assert d["settings_sha256"] and d["head"] == "current" and d["settings"]["tiny"] is True
    # every checkpoint was a boundary: seeds 100,000 x 42 + 100 x b, b the multiple's index
    seeds = d["sessions"][0]["boundary_seeds"]
    assert [s["steps"] for s in seeds] == [0, 256, 512, 768, 1024, 1280]
    assert [s["seed"] for s in seeds] == [4_200_000 + 100 * b for b in range(6)]
    # K2: each check's checkpoint and the newest two
    assert d["checkpoints"] == [0, 256, 512, 768, 1024, 1280, 1536] and d["kept"] == [512, 1024, 1280, 1536]
    assert sorted(int(p.stem.split("_")[1]) for p in (jd / "checkpoints").glob("checkpoint_*.json")) == d["kept"]
    # the checks: 512 and 1,024 at rollout starts, 1,536 on the final checkpoint; each attempt logged and closed
    assert [e["at_steps"] for e in d["monitor"]] == [512, 1024, 1536]
    for e in d["monitor"]:
        assert len(e["attempts"]) == 1 and e["attempts"][0]["exit_code"] == 0 and e["attempts"][0]["record"] is True
        rec = json.loads((jd / e["record"]).read_text())
        assert rec["checkpoint_sha256"] == e["checkpoint_sha256"] == e["attempts"][0]["checkpoint_sha256"]
        assert rec["sha256"] == e["record_sha256"] and rec["sources"]["given"] == rec["sources"]["own"]
        assert rec["sources"]["own"] == d["sessions"][0]["sources"]["sha256"]
    # G8b failed at the first check: one fix, the mapped task's weight doubled, in place
    assert d["gates"]["g8b"]["pass"] is False and d["gates"]["g8b"]["at_steps"] == 512
    assert d["fix"]["at_steps"] == 512 and d["fix"]["tasks"] == ["walk_nominal"]
    assert d["fix"]["weights"]["walk_nominal"] == 2 * d["fix"]["weights_before"]["walk_nominal"] == 4.0
    assert {t: w for t, w in d["fix"]["weights"].items() if t != "walk_nominal"} == {
        t: w for t, w in d["fix"]["weights_before"].items() if t != "walk_nominal"}
    assert d["gates"]["g8b_after_fix"]["pass"] is True and d["stopped"] is None
    # the settings keep the configs the run started with; the fixed ones are under the fix
    assert all(c["task_weights"] is None for c in d["settings"]["env_configs"])
    assert d["fix"]["env_configs"][0]["task_weights"][1] == 4.0
    # the checkpoint's run state is the state before that step's check
    c512 = json.loads((jd / "checkpoints" / "checkpoint_512.json").read_text())
    c1024 = json.loads((jd / "checkpoints" / "checkpoint_1024.json").read_text())
    assert c512["run_state"]["fix"] is None and c512["run_state"]["checks_done"] == []
    assert c512["run_state"]["task_weights"]["walk_nominal"] == 2.0
    assert c1024["run_state"]["fix"]["at_steps"] == 512 and c1024["run_state"]["checks_done"] == [512]
    assert c1024["run_state"]["task_weights"]["walk_nominal"] == 4.0
    assert c512["run_uuid"] == d["run_uuid"] and c512["settings_sha256"] == d["settings_sha256"]
    # one line per update, each step once
    up = updates(jd)
    assert [u["steps"] for u in up] == list(range(ROLLOUT, FINAL + 1, ROLLOUT))
    assert all(u["minibatch_steps_possible"] == 10 and 0 < u["minibatch_steps_run"] <= 10 for u in up)
    # the baseline, the gate records, progress
    assert d["baseline"]["record"] == "monitor/r0l.json" and (jd / "monitor" / "r0l.json").is_file()
    assert d["gates"]["g8a"]["passes"] is True and (jd / "gates" / "g8a.json").is_file()
    rate = json.loads((jd / "gates" / "rate.json").read_text())["sessions"][0]
    assert rate["first_rollout"]["env_steps"] == ROLLOUT and rate["first_three"]["env_steps"] == 3 * ROLLOUT
    prog = json.loads((jd / "progress.json").read_text())
    assert prog["steps"] == FINAL and prog["status"] == "done" and prog["last_monitor"]["at_steps"] == FINAL


def test_an_archive_saved_by_the_trainer_has_no_bytecode(ref_plain):
    from designlab.stage2 import tbenv

    _need_torch()
    tbenv.repo_on_path()
    from environments.shared.policy_loading import inspect_sb3_archive

    for path in sorted((job(ref_plain / "out") / "checkpoints").glob("model_*.zip")):
        assert sorted(inspect_sb3_archive(path).bytecode_members) == []
    d = doc_of(ref_plain / "out")
    # no schedule is a function: the rate is this study's class, the float clip range SB3's own module-level one
    assert d["sessions"][0]["schedules"] == dict(
        lr={"class": "LinearSchedule", "start": 3e-5, "end": 0.3 * 3e-5},
        clip_range={"class": "stable_baselines3.common.utils.ConstantSchedule", "value": 0.2})
    from designlab.stage2 import train2

    with pytest.raises(TypeError, match="is a function"):
        train2.describe_schedule(lambda progress: 1e-4)


def test_no_gates_runs_no_monitor(ref_plain):
    d, jd = doc_of(ref_plain / "out"), job(ref_plain / "out")
    assert d["status"] == "done" and d["settings"]["gates"] is False
    assert d["monitor"] == [] and d["baseline"] is None and d["gates"]["g8a"] is None
    assert not (jd / "monitor").exists() and not (jd / "gates" / "g8a.json").exists()


# ---------------------------------------------------------------------------------------------------------------
# Kill and resume (SIGKILL; a trainer process of its own)
# ---------------------------------------------------------------------------------------------------------------
def test_a_killed_run_resumes_bit_for_bit(ref_fix, tmp_path):
    """SIGKILL between two boundaries (896 steps: boundaries at 768 and 1,024), with 2 workers; resumed with 1."""
    out = tmp_path / "out"
    r = drive(tmp_path, out, env=dict(STUB_MON_FAIL_AT="512", STUB_KILL_AT="896"), workers=2)
    assert r.returncode == -signal.SIGKILL, (r.returncode, r.stderr[-2000:])
    d = doc_of(out)
    assert d["status"] == "running" and d["sessions"][0]["ended_at"] is None and d["kept"] == [512, 768]
    assert len(updates(job(out))) == 7  # (updates of 128 .. 896: three of them past the last checkpoint)
    r = drive(tmp_path, out, env=dict(STUB_MON_FAIL_AT="512"), workers=1, resume=True)
    assert r.returncode == 0, r.stderr[-3000:]
    d = doc_of(out)
    s = d["sessions"][1]
    assert d["status"] == "done" and s["resumed_from"] == 768 and s["restarted_from_zero"] is False
    assert s["fingerprint_max_abs_diff"] == 0.0 and s["workers"] == 1 and d["sessions"][0]["workers"] == 2
    assert s["boundary_seeds"][0] == dict(b=3, steps=768, seed=4_200_300)  # the same boundary, before its rollout
    assert s["discarded_after_resume"]["updates"]["lines"] == 1  # (the update of 896 steps, made again)
    a, b = job(ref_fix / "out"), job(out)
    for n in (1024, 1280, FINAL):  # 1,024: the first checkpoint after the resume (obs_rms with its count)
        assert_same_checkpoint(a, b, n)
    same_updates(a, b)
    assert d["fix"]["weights"] == doc_of(ref_fix / "out")["fix"]["weights"]
    assert not list((tmp_path / "out").rglob("*.tmp"))


def test_kill_during_the_monitor(ref_fix, tmp_path):
    """The whole process group is killed while the monitor child of the first check runs: the attempt was logged
    before the child started and was never closed; the resume runs that check again and it does not count."""
    out, hold = tmp_path / "out", tmp_path / "hold"
    hold.write_text("")
    _set_file(tmp_path)
    args = dict(KW, out=str(out), workers=2)
    p = subprocess.Popen([PY, "-m", HERE, "drive", json.dumps(args)], cwd=STUDY, start_new_session=True,
                         env=_env(tmp_path, STUB_MON_FAIL_AT="512", STUB_MON_HOLD=hold),
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 240
        while not Path(str(hold) + ".started").exists():
            assert p.poll() is None, p.stderr.read().decode()[-3000:]
            assert time.monotonic() < deadline, "the monitor child did not start"
            time.sleep(0.05)
        d = doc_of(out)
        (e,) = d["monitor"]
        assert e["at_steps"] == 512 and len(e["attempts"]) == 1
        assert e["attempts"][0]["closed_at"] is None and e["attempts"][0]["exit_code"] is None
        assert e["attempts"][0]["checkpoint_sha256"] == e["checkpoint_sha256"]
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        p.stderr.close()
    hold.unlink()
    assert not (job(out) / "monitor" / "monitor_512.json").exists()
    r = drive(tmp_path, out, env=dict(STUB_MON_FAIL_AT="512"), workers=0, resume=True)
    assert r.returncode == 0, r.stderr[-3000:]
    d = doc_of(out)
    e = d["monitor"][0]
    assert d["status"] == "done" and d["sessions"][1]["resumed_from"] == 512
    assert [bool(a.get("interrupted")) for a in e["attempts"]] == [True, False]
    assert e["attempts"][0]["closed_at"] is None and e["attempts"][1]["exit_code"] == 0
    assert e["counted_without_record"] is False and d["fix"]["at_steps"] == 512
    for n in (1024, FINAL):
        assert_same_checkpoint(job(ref_fix / "out"), job(out), n)
    same_updates(job(ref_fix / "out"), job(out))


def test_kill_after_the_fix(ref_fix, tmp_path):
    """SIGKILL one rollout after the check that made the fix (640 steps; the next checkpoint is 768): the resume
    starts from checkpoint 512, whose run state has no fix, reads the check's record again (no episode is run
    twice), makes the same fix, and ends as the uninterrupted run."""
    out = tmp_path / "out"
    r = drive(tmp_path, out, env=dict(STUB_MON_FAIL_AT="512", STUB_KILL_AT="640"), workers=0)
    assert r.returncode == -signal.SIGKILL, (r.returncode, r.stderr[-2000:])
    before = doc_of(out)
    assert before["fix"]["at_steps"] == 512 and before["kept"] == [256, 512]
    record = (job(out) / "monitor" / "monitor_512.json").read_bytes()
    r = drive(tmp_path, out, env=dict(STUB_MON_FAIL_AT="512"), workers=0, resume=True)
    assert r.returncode == 0, r.stderr[-3000:]
    d = doc_of(out)
    assert d["status"] == "done" and d["sessions"][1]["resumed_from"] == 512
    assert (job(out) / "monitor" / "monitor_512.json").read_bytes() == record  # read again, not run again
    assert len(d["monitor"][0]["attempts"]) == 1 and d["monitor"][0]["attempts"][0]["session"] == 0
    assert {k: v for k, v in d["fix"].items()} == {k: v for k, v in before["fix"].items()}
    assert d["sessions"][1]["discarded_after_resume"]["fix"] == before["fix"]  # (rebuilt from the records)
    assert d["sessions"][1]["discarded_after_resume"]["monitor_at_resume"]["at_steps"] == 512
    for n in (1024, 1280, FINAL):
        assert_same_checkpoint(job(ref_fix / "out"), job(out), n)
    same_updates(job(ref_fix / "out"), job(out))
    fixes = [e for e in d["monitor"] if "g8b" in e]
    assert len(fixes) == 1 and fixes[0]["at_steps"] == 512


# ---------------------------------------------------------------------------------------------------------------
# Resume, in process (a session cut by an exception at a rollout start)
# ---------------------------------------------------------------------------------------------------------------
def test_resume_when_run_json_is_ahead_of_the_checkpoint(ref_fix, stub, tmp_path, monkeypatch):
    """The session is cut at 1,152 steps, after the check at 1,024; then checkpoint 1,024 loses a file (a copy
    that was cut short), so the newest intact checkpoint is 768 while ``run.json`` holds the check at 1,024."""
    out = tmp_path / "out"
    monkeypatch.setenv("STUB_MON_FAIL_AT", "512")
    cut_at(monkeypatch, stub, 1152)
    with pytest.raises(_Cut):
        run(stub, out, gates=True)
    monkeypatch.undo()
    d = doc_of(out)
    assert d["status"] == "partial" and [e["at_steps"] for e in d["monitor"]] == [512, 1024]
    assert d["kept"] == [512, 768, 1024]  # (K2: each check's and the newest two)
    jd = job(out)
    old_record = (jd / "monitor" / "monitor_1024.json").read_bytes()
    with open(jd / "checkpoints" / "fingerprint_1024.npz", "r+b") as fh:
        fh.truncate(100)
    stub2 = install_again(monkeypatch, tmp_path)
    monkeypatch.setenv("STUB_MON_FAIL_AT", "512")
    d = run(stub2, out, gates=True, resume=True)
    s = d["sessions"][1]
    assert d["status"] == "done" and s["resumed_from"] == 768
    assert [r["steps"] for r in s["skipped_checkpoints"]] == [1024]
    assert "fingerprint_1024.npz" in s["skipped_checkpoints"][0]["reason"]
    disc = s["discarded_after_resume"]
    assert [e["at_steps"] for e in disc["monitor"]] == [1024] and disc["checkpoints"] == [1024]
    assert disc["updates"]["lines"] == 3 and disc["num_timesteps"] == 1152  # (the updates of 896 .. 1,152)
    assert (jd / "updates.discarded.s1.jsonl").read_text().count("\n") == 3
    assert d["fix"]["at_steps"] == 512 and "fix" not in disc  # (the fix is in checkpoint 768's run state)
    steps = [u["steps"] for u in updates(jd)]
    assert steps == sorted(set(steps)) == list(range(ROLLOUT, FINAL + 1, ROLLOUT))  # no step twice
    # the check at 1,024 was made again on the checkpoint saved again; the old record is kept beside
    assert [e["at_steps"] for e in d["monitor"]] == [512, 1024, 1536]
    assert (jd / "monitor" / "monitor_1024.discarded.s1.json").read_bytes() == old_record
    assert d["monitor"][1]["attempts"][0]["session"] == 1
    for n in (1024, FINAL):
        assert_same_checkpoint(job(ref_fix / "out"), jd, n)
    same_updates(job(ref_fix / "out"), jd)


def install_again(monkeypatch, tmp_path):
    """The stub seams again after ``monkeypatch.undo()`` (which took them off with the cut)."""
    from designlab.stage2 import train2

    for name, fn in (("_env_configs", stub_configs), ("_env_specs", stub_specs), ("_g8a_short", stub_g8a),
                     ("_monitor_cmd", stub_monitor_cmd), ("_validation", stub_validation)):
        monkeypatch.setattr(train2, name, fn)
    monkeypatch.setenv("STUB_SET_FILE", str(tmp_path / "set.json"))
    monkeypatch.chdir(STUDY)
    return train2


def test_restart_from_zero_is_the_fresh_run(ref_plain, stub, tmp_path, monkeypatch):
    """A job with a ``run.json`` and no checkpoint restarts from step 0 in place and is the same run."""
    out = tmp_path / "out"
    cut_at(monkeypatch, stub, 384)
    with pytest.raises(_Cut):
        run(stub, out)
    monkeypatch.undo()
    stub = install_again(monkeypatch, tmp_path)
    jd = job(out)
    uuid = doc_of(out)["run_uuid"]
    for p in (jd / "checkpoints").iterdir():
        p.unlink()
    d = run(stub, out, resume=True)
    s = d["sessions"][1]
    assert d["status"] == "done" and d["run_uuid"] == uuid
    assert s["restarted_from_zero"] is True and s["resumed_from"] is None
    assert s["boundary_seeds"][0] == dict(b=0, steps=0, seed=4_200_000)
    assert s["discarded_after_resume"]["updates"]["lines"] == 3
    for n in (1280, FINAL):
        assert_same_checkpoint(job(ref_plain / "out"), jd, n)
    same_updates(job(ref_plain / "out"), jd)


def test_a_resume_from_checkpoint_zero(ref_plain, stub, tmp_path, monkeypatch):
    """A session cut before its second checkpoint resumes from checkpoint 0: the same code path as a fresh start
    (the boundary of step 0 is made again, the checkpoint is not saved again)."""
    out = tmp_path / "out"
    cut_at(monkeypatch, stub, 128)
    with pytest.raises(_Cut):
        run(stub, out)
    monkeypatch.undo()
    stub = install_again(monkeypatch, tmp_path)
    assert doc_of(out)["kept"] == [0]
    zero = (job(out) / "checkpoints" / "model_0.zip").read_bytes()
    seen = {}
    inner = stub._Run.boundary

    def boundary(self, n):
        if n == 0:  # (what checkpoint 0 holds when the resumed session makes its boundary)
            seen[0] = (job(out) / "checkpoints" / "model_0.zip").read_bytes()
        inner(self, n)

    monkeypatch.setattr(stub._Run, "boundary", boundary)
    d = run(stub, out, resume=True)
    s = d["sessions"][1]
    assert d["status"] == "done" and s["resumed_from"] == 0 and s["restarted_from_zero"] is False
    assert s["boundary_seeds"][0] == dict(b=0, steps=0, seed=4_200_000) and seen[0] == zero  # not saved again
    assert s["discarded_after_resume"]["updates"]["lines"] == 1
    for n in (1280, FINAL):
        assert_same_checkpoint(job(ref_plain / "out"), job(out), n)
    same_updates(job(ref_plain / "out"), job(out))


def test_the_schedule_follows_the_step_count_across_a_resume(ref_plain, stub, tmp_path, monkeypatch):
    out = tmp_path / "out"
    cut_at(monkeypatch, stub, 640)
    with pytest.raises(_Cut):
        run(stub, out)
    monkeypatch.undo()
    stub = install_again(monkeypatch, tmp_path)
    d = run(stub, out, resume=True)
    assert d["sessions"][1]["resumed_from"] == 512
    up = updates(job(out))
    start, end = d["settings"]["lr"]["start"], d["settings"]["lr"]["end"]
    assert (start, end) == (3e-5, 0.3 * 3e-5)
    for u in up:  # linear in the steps done over the budget, whatever the session
        assert u["lr"] == pytest.approx(end + (start - end) * (1.0 - u["steps"] / FINAL), rel=1e-12)
    assert up[-1]["lr"] == pytest.approx(end) and [u["session"] for u in up] == [0] * 4 + [1] * 8
    same_updates(job(ref_plain / "out"), job(out))
    assert_same_checkpoint(job(ref_plain / "out"), job(out), FINAL)


# ---------------------------------------------------------------------------------------------------------------
# What a resume refuses
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture
def cut_run(stub, tmp_path, monkeypatch):
    """A run cut at 640 steps (checkpoints 256 and 512 kept), and ``train2`` with the stub seams."""
    out = tmp_path / "out"
    cut_at(monkeypatch, stub, 640)
    with pytest.raises(_Cut):
        run(stub, out)
    monkeypatch.undo()
    return install_again(monkeypatch, tmp_path), out


def _refused(train2, out, match, **kw):
    before = (job(out) / "run.json").read_bytes() if (job(out) / "run.json").exists() else None
    with pytest.raises(train2.Refused, match=match) as info:
        run(train2, out, resume=True, **kw)
    assert info.value.code == train2.EXIT_REFUSED
    if before is not None:  # a refusal writes nothing
        assert (job(out) / "run.json").read_bytes() == before
    return str(info.value)


def test_resume_refusals_settings(cut_run, stub, tmp_path):
    train2, out = cut_run
    with pytest.raises(train2.Refused, match="nothing to resume"):
        run(train2, tmp_path / "nowhere", resume=True)
    with pytest.raises(train2.Refused, match="pass --resume"):
        run(train2, out)
    for kw, word in ((dict(seed=43, job="R5-stub-s42"), "seed"), (dict(steps=2048), "budget"),
                     (dict(checkpoint_every=128), "checkpoint_every"), (dict(monitor_every=1024), "monitor_every"),
                     (dict(lr=1e-4), "lr"), (dict(tiny=False), "tiny"), (dict(gates=True), "gates"),
                     (dict(config=dict(filter_hz=20.0)), "config"), (dict(ppo=dict(n_steps=16)), "ppo_overrides"),
                     (dict(monitor_also_k=(0,)), "monitor_also_k")):
        assert word in _refused(train2, out, "other settings", **kw)
    # nothing given but the run's own: accepted (the budget and the rate are the run's)
    d = run(train2, out, resume=True, steps=None)
    assert d["status"] == "done" and d["sessions"][1]["resumed_from"] == 512
    # a run that is done: a resume has nothing to do and writes nothing
    before = (job(out) / "run.json").read_bytes()
    assert run(train2, out, resume=True)["status"] == "done" and (job(out) / "run.json").read_bytes() == before
    assert len(doc_of(out)["sessions"]) == 2


def test_resume_refusals_inputs_and_a_later_freeze(cut_run, tmp_path, monkeypatch):
    train2, out = cut_run
    files = json.loads((tmp_path / "set.json").read_text())["files"]
    # a later freeze of the set, every used file and entry unchanged: accepted, both freezes recorded
    write_set(tmp_path / "set.json", freezes=("f0", "f1"), files=files)
    changed = json.loads(json.dumps(files))
    changed["transitions/start.npz"]["sha256"] = "9" * 64
    write_set(tmp_path / "set.json", freezes=("f0", "f1"), files=changed)
    text = _refused(train2, out, "other inputs")
    assert "transitions/start.npz" in text and "no flag" in text
    _refused(train2, out, "other inputs", allow_source_change=True, allow_version_change=True)  # no flag helps
    entry = json.loads(json.dumps(files))
    entry["refs/nominal.npz"]["entry_sha256"] = "8" * 64
    write_set(tmp_path / "set.json", freezes=("f0", "f1"), files=entry)
    assert "refs/nominal.npz" in _refused(train2, out, "other inputs")
    write_set(tmp_path / "set.json", freezes=("f0", "f1"), files=files)
    # an env that refuses its inputs when it is built from the run's stored config (the real env's own check of
    # rule K4) is a refused resume too
    monkeypatch.setenv("STUB_INPUT_ERROR", "1")
    assert "not the one the named freeze holds" in _refused(train2, out, "other inputs")
    monkeypatch.delenv("STUB_INPUT_ERROR")
    d = run(train2, out, resume=True)
    s = d["sessions"][1]
    assert d["status"] == "done" and s["resumed_from"] == 512
    assert s["freeze_index"] == dict(started=dict(index=0, manifest_sha256="f0"),
                                     now=dict(index=1, manifest_sha256="f1"))
    assert d["sessions"][0]["freeze_index"]["now"] == dict(index=0, manifest_sha256="f0")
    assert d["settings"]["set_freeze"] == dict(index=0, manifest_sha256="f0")
    assert all(c["set_freeze"] == "f0" for c in d["settings"]["env_configs"])  # the run's stored configs


def test_resume_refusals_sources_and_versions(cut_run, monkeypatch):
    from designlab.stage2 import s2load

    train2, out = cut_run
    real = train2.sources()
    moved = dict(real, files=dict(real["files"], **{"designlab/stage2/r5core.py": "0" * 64}), sha256="1" * 64)
    monkeypatch.setattr(train2, "sources", lambda: moved)
    text = _refused(train2, out, "other sources")
    assert "--allow-source-change" in text and "designlab/stage2/r5core.py" in text
    vers = dict(s2load.versions(), numpy="9.9.9")
    monkeypatch.setattr(s2load, "versions", lambda: dict(vers))
    # both differ: the sources are refused first, then the versions
    _refused(train2, out, "other sources", allow_version_change=True)
    text = _refused(train2, out, "other library versions", allow_source_change=True)
    assert "--allow-version-change" in text and "9.9.9" in text
    d = run(train2, out, resume=True, allow_source_change=True, allow_version_change=True)
    s = d["sessions"][1]
    assert d["status"] == "done" and s["source_change"] is True and s["version_change"] is True
    assert s["sources_before"]["sha256"] == real["sha256"] and s["sources"]["sha256"] == "1" * 64
    assert s["versions_before"]["numpy"] != "9.9.9" and s["versions"]["numpy"] == "9.9.9"
    assert d["flags"]["source_change"] is True and d["flags"]["version_change"] is True
    # every later record of the run carries the flag: the checkpoints' run state
    last = json.loads((job(out) / "checkpoints" / f"checkpoint_{FINAL}.json").read_text())
    assert last["run_state"]["flags"]["source_change"] is True and last["run_state"]["flags"]["version_change"] is True
    # torch's build label is no version change
    assert s2load.comparable(dict(torch="2.15.0+cu126")) == s2load.comparable(dict(torch="2.15.0+cpu"))


def test_resume_refusals_checkpoints(cut_run, tmp_path):
    from designlab.stage2 import s2io

    train2, out = cut_run
    jd = job(out)
    # the newest checkpoint is another run's (its record says so, and hashes to itself): refused
    path = jd / "checkpoints" / "checkpoint_512.json"
    keep = path.read_bytes()
    rec = json.loads(keep)
    rec["run_uuid"] = "f" * 32
    rec.pop("sha256")
    rec["sha256"] = s2io.sha256_json(rec)
    path.write_text(json.dumps(rec, indent=1))
    assert "another run's checkpoint" in _refused(train2, out, "run_uuid")
    # a fingerprint that is not reproduced: refused (the stored actions moved; the record follows the file)
    path.write_bytes(keep)
    fp = jd / "checkpoints" / "fingerprint_512.npz"
    keep_fp = fp.read_bytes()
    with np.load(fp) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["actions"] = arrays["actions"] + np.float32(1e-3)
    s2io.write_npz(fp, arrays)
    rec = json.loads(keep)
    rec["files"]["fingerprint"].update(bytes=fp.stat().st_size, sha256=s2io.sha256_file(fp))
    rec.pop("sha256")
    rec["sha256"] = s2io.sha256_json(rec)
    path.write_text(json.dumps(rec, indent=1))
    assert "not judged" in _refused(train2, out, "fingerprint not reproduced")
    fp.write_bytes(keep_fp)
    path.write_bytes(keep)
    # a broken newest checkpoint is skipped with its reason; the run resumes from the one before
    with open(jd / "checkpoints" / "model_512.zip", "r+b") as fh:
        fh.truncate(1000)
    d = run(train2, out, resume=True)
    s = d["sessions"][1]
    assert d["status"] == "done" and s["resumed_from"] == 256
    assert s["skipped_checkpoints"] == [dict(steps=512, reason=s["skipped_checkpoints"][0]["reason"])]
    assert "model_512.zip" in s["skipped_checkpoints"][0]["reason"]
    # a stopped run is not resumed
    doc = doc_of(out)
    doc.update(status="stopped_g8b", stopped=dict(at_steps=1024, status="stopped_g8b", reason="G8b failed again"))
    (jd / "run.json").write_text(json.dumps(doc))
    assert "G8b failed again" in _refused(train2, out, "was stopped")


def test_set_s1_needs_tiny_and_a_run_names_its_budget(stub, tmp_path):
    with pytest.raises(stub.Refused, match="only with --tiny"):
        run(stub, tmp_path / "a", set_name="S1", tiny=False)
    with pytest.raises(stub.Refused, match="names its budget"):
        run(stub, tmp_path / "b", steps=None, tiny=False, lr=1e-4)
    with pytest.raises(stub.Refused, match="no learning rate"):
        run(stub, tmp_path / "c", tiny=False)
    with pytest.raises(stub.Refused, match="multiple of checkpoint_every"):
        run(stub, tmp_path / "d", monitor_every=300)
    with pytest.raises(stub.Refused, match="names its own job"):
        run(stub, tmp_path / "e", parent=(tmp_path / "x", 512))
    assert not (tmp_path / "a").exists() and not (tmp_path / "c" / "jobs").exists()


# ---------------------------------------------------------------------------------------------------------------
# G8a, the checks' attempts, the stop
# ---------------------------------------------------------------------------------------------------------------
def test_g8a_fails_no_run(stub, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("STUB_G8A_FAIL", "1")
    d = run(stub, out, gates=True)
    assert d["status"] == "stopped_g8a" and d["stopped"]["at_steps"] == 0 and "G8a fails: no run" in d[
        "stopped"]["reason"]
    assert d["gates"]["g8a"]["passes"] is False and d["settings_sha256"] is None  # no env was built
    assert not (job(out) / "checkpoints").exists() and (job(out) / "gates" / "g8a.json").is_file()
    with pytest.raises(stub.Refused, match="was stopped"):
        run(stub, out, gates=True, resume=True)
    # the CLI's exit code for it is 3
    monkeypatch.setattr(stub, "train", lambda *a, **k: d)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(v, "1")
    assert stub.main(["run", "--mode", "R5", "--body", "stub", "--seed", "42", "--out", str(out), "--tiny"]) == 3
    monkeypatch.setattr(stub, "train", lambda *a, **k: dict(d, status="stopped_g8b"))
    assert stub.main(["run", "--mode", "R5", "--body", "stub", "--seed", "42", "--out", str(out), "--tiny"]) == 0
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    assert stub.main(["run", "--mode", "R5", "--body", "stub", "--seed", "42", "--out", str(out), "--tiny"]) == 4


def test_the_cli_hands_its_arguments_to_train(stub, tmp_path, monkeypatch):
    """Every option of ``run`` reaches ``train``; ``--config`` takes any env config field by name (a further
    named option of the env needs no change to the trainer)."""
    seen = {}

    def train(mode, body, seed, **kw):
        seen.update(kw, mode=mode, body=body, seed=seed)
        return dict(status="done")

    monkeypatch.setattr(stub, "train", train)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(v, "1")
    code = stub.main(["run", "--mode", "R6", "--body", "75-M-H-10-X", "--seed", "43", "--set", "V6", "--out",
                      str(tmp_path), "--job", "ft", "--steps", "5000000", "--workers", "3", "--resume",
                      "--allow-version-change", "--allow-source-change", "--from-checkpoint", "/runs/jobs/p:40001536",
                      "--lr", "3e-5", "--monitor-every", "500000", "--monitor-also-k", "0,2", "--no-gates",
                      "--run-id", "s2-4", "--config", '{"filter_hz": 20.0, "servo_model": "m1"}'])
    log = seen.pop("log")
    assert code == 0 and callable(log) and seen == dict(
        mode="R6", body="75-M-H-10-X", seed=43, set_name="V6", out=str(tmp_path), job="ft", steps=5_000_000,
        resume=True, workers=3, monitor_every=500_000, monitor_also_k=(0, 2), lr=3e-5,
        config=dict(filter_hz=20.0, servo_model="m1"), gates=False, allow_version_change=True,
        allow_source_change=True, parent="/runs/jobs/p:40001536", tiny=False, run_id="s2-4")
    assert stub._parent_of("/runs/jobs/p:40001536") == (Path("/runs/jobs/p"), 40_001_536)
    # the overrides reach every env config untouched (the stub's configs take any field)
    monkeypatch.setenv("STUB_SET_FILE", str(tmp_path / "set.json"))
    assert all(c["servo_model"] == "m1" and c["filter_hz"] == 20.0
               for c in stub._env_configs("R5", "stub", "STUB", dict(filter_hz=20.0, servo_model="m1")))
    # defaults: the design's intervals, one worker per vCPU, gates on
    seen.clear()
    assert stub.main(["run", "--mode", "R5", "--body", "b", "--seed", "42", "--out", str(tmp_path)]) == 0
    assert (seen["monitor_every"], seen["workers"], seen["gates"], seen["tiny"], seen["set_name"]) == (
        2_500_000, None, True, False, "V6")
    assert stub.CHECKPOINT_EVERY == 250_000 and stub.N_ENVS == 8 and stub.N_STEPS == 1024


def test_a_check_without_a_record_counts_after_two_closed_attempts(stub, tmp_path, monkeypatch):
    """Rule M1: the monitor leaves no record twice at the first check: two attempts, each logged and closed with
    its exit code; the check then counts as without a gain and as a failed G8b reading (every episode an error:
    every mapped task doubled).  At the second reading the record is there and passes."""
    out, crash = tmp_path / "out", tmp_path / "crash"
    crash.write_text("2")
    monkeypatch.setenv("STUB_MON_CRASH", str(crash))
    d = run(stub, out, gates=True)
    e = d["monitor"][0]
    assert d["status"] == "done" and crash.read_text() == "0"
    assert [(a["exit_code"], a["record"]) for a in e["attempts"]] == [(7, False), (7, False)]
    assert all(a["closed_at"] and a["checkpoint_sha256"] == e["checkpoint_sha256"] for a in e["attempts"])
    assert e["counted_without_record"] is True and e["record"] is None and e["no_record"] is True
    assert e["gain"]["gain"] is False and e["outcomes"]["error"] == 6 and e["N"] == 0 and e["S"] == 0.0
    assert d["gates"]["g8b"]["pass"] is False and d["gates"]["g8b"]["up"] == 0
    assert d["fix"]["tasks"] == ["walk_nominal", "walk_v0.4", "turn_q90_left", "turn_q90_right", "stop",
                                 "start_stop"]
    assert d["fix"]["monitor"] is None and d["gates"]["g8b_after_fix"]["pass"] is True
    assert len(d["monitor"][1]["attempts"]) == 1 and d["monitor"][1]["counted_without_record"] is False


def test_two_attempts_never_closed_halt_the_run(stub, tmp_path, monkeypatch):
    """The lead's ruling of 16 Nov 2026: after two attempts at one check that were never closed, the run halts."""
    out = tmp_path / "out"
    cut_at(monkeypatch, stub, 640)
    with pytest.raises(_Cut):
        run(stub, out, gates=True)
    monkeypatch.undo()
    stub = install_again(monkeypatch, tmp_path)
    jd = job(out)
    d = doc_of(out)
    e = d["monitor"][0]
    never = dict(e["attempts"][0], closed_at=None, exit_code=None, record=None)
    d["monitor"] = [dict(at_steps=512, multiple=512, checkpoint_sha256=e["checkpoint_sha256"],
                         attempts=[dict(never), dict(never)])]
    d["fix"] = None
    (jd / "run.json").write_text(json.dumps(d))
    (jd / "monitor" / "monitor_512.json").unlink()
    with pytest.raises(stub.Halt, match="never closed") as info:
        run(stub, out, gates=True, resume=True)
    assert info.value.code == stub.EXIT_HALT == 5
    d = doc_of(out)
    assert d["status"] == "partial" and "never closed" in d["sessions"][1]["error"]
    assert [a.get("interrupted") for a in d["monitor"][0]["attempts"]] == [True, True]


def test_a_second_g8b_failure_stops_the_run(stub, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("STUB_MON_FAIL_AT", "512,1024")
    d = run(stub, out, gates=True)
    assert d["status"] == "stopped_g8b" and d["num_timesteps"] == 1024
    assert d["stopped"]["at_steps"] == 1024 and d["stopped"]["status"] == "stopped_g8b"
    assert d["gates"]["g8b"]["pass"] is False and d["gates"]["g8b_after_fix"]["pass"] is False
    assert d["kept"][-1] == 1024 and [u["steps"] for u in updates(job(out))][-1] == 1024  # no step after the stop
    assert [e["at_steps"] for e in d["monitor"]] == [512, 1024]
    prog = json.loads((job(out) / "progress.json").read_text())
    assert prog["status"] == "stopped_g8b" and prog["stopped"]["at_steps"] == 1024
    with pytest.raises(stub.Refused, match="was stopped"):
        run(stub, out, gates=True, resume=True)


def test_the_final_check_does_not_stop_a_run_its_budget_ended(stub, tmp_path, monkeypatch):
    """The check on the final checkpoint is read like any other, but the budget ended the run: a second G8b
    failure there is recorded and is no stop; a first failure there makes no fix (no training follows)."""
    monkeypatch.setenv("STUB_MON_FAIL_AT", "768,1536")
    d = run(stub, tmp_path / "a", gates=True, monitor_every=768)
    assert d["status"] == "done" and d["stopped"] is None and d["num_timesteps"] == FINAL
    assert [e["at_steps"] for e in d["monitor"]] == [768, 1536] and d["fix"]["at_steps"] == 768
    assert d["gates"]["g8b_after_fix"]["pass"] is False and d["gates"]["g8b_after_fix"]["at_steps"] == FINAL
    assert d["final_check"]["would_have_stopped"]["status"] == "stopped_g8b"
    assert d["final_check"]["ended_by"] == "the budget" and "fix_not_applied" not in d["final_check"]
    monkeypatch.setenv("STUB_MON_FAIL_AT", "1536")
    d = run(stub, tmp_path / "b", gates=True, monitor_every=1536)
    assert d["status"] == "done" and [e["at_steps"] for e in d["monitor"]] == [1536]
    assert d["gates"]["g8b"]["pass"] is False and d["fix"] is None
    assert d["final_check"]["fix_not_applied"]["tasks"] == ["walk_nominal"]
    last = json.loads((job(tmp_path / "b") / "checkpoints" / f"checkpoint_{FINAL}.json").read_text())
    assert last["run_state"]["task_weights"]["walk_nominal"] == 2.0


def test_a_failed_baseline_halts_before_the_first_rollout(stub, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(stub, "_monitor_cmd", lambda: [PY, "-c", "import sys; sys.exit(6)"])
    with pytest.raises(stub.Halt, match="R0L baseline could not be made"):
        run(stub, out, gates=True)
    d = doc_of(out)
    assert d["status"] == "partial" and d["baseline"] is None and d["checkpoints"] == []
    assert d["baseline_attempts"][0]["exit_code"] == 6 and d["baseline_attempts"][0]["record"] is False
    assert not (job(out) / "checkpoints").exists() and not (job(out) / "updates.jsonl").exists()


def test_a_failed_set_task_weights_ends_the_session(stub, tmp_path, monkeypatch):
    """After an ``env_method`` that changes env state failed in one env, the trainer does not carry on."""
    out = tmp_path / "out"
    monkeypatch.setenv("STUB_MON_FAIL_AT", "512")
    monkeypatch.setenv("STUB_WEIGHTS_TRIP", "1")
    with pytest.raises(stub.TrainError, match="set_task_weights failed"):
        run(stub, out, gates=True)
    d = doc_of(out)
    assert d["status"] == "partial" and d["num_timesteps"] == 512 and d["fix"]["at_steps"] == 512


# ---------------------------------------------------------------------------------------------------------------
# Lesson 16
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hot(tmp_path_factory):
    """A run with a learning rate that trips ``target_kl`` (3e-2 against the tiny run's 3e-5), with its checks."""
    _need_torch()
    tmp = tmp_path_factory.mktemp("hot")
    r = drive(tmp, tmp / "out", workers=0, lr=3e-2)
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp / "out"


def test_minibatch_steps_are_counted(hot):
    """Lesson 16: PPO stops an update early without a message when the KL passes 1.5 x ``target_kl``; the wrapped
    optimizer step counts what really ran, while SB3's own count goes on as if all had."""
    up = updates(job(hot))
    assert len(up) == 12 and all(u["minibatch_steps_possible"] == 10 for u in up)
    assert all(0 < u["minibatch_steps_run"] < u["minibatch_steps_possible"] for u in up[1:])
    assert any(u["approx_kl"] > 1.5 * 0.03 for u in up)
    assert doc_of(hot)["settings"]["lr"]["start"] == 3e-2 and doc_of(hot)["settings"]["ppo"]["target_kl"] == 0.03


def test_kl_cut_flag(hot):
    """Rule O3: the median share of minibatch steps run over a check interval under one half sets the flag; it
    stops nothing."""
    out, d = hot, doc_of(hot)
    up = updates(job(out))
    share = float(np.median([u["minibatch_steps_run"] / 10 for u in up[8:]]))
    last = d["monitor"][-1]["kl_cut"]
    assert last == dict(after_steps=1024, to_steps=1536, updates=4, median_share=share, cut=share < 0.5)
    assert d["flags"]["kl_cut"] is last["cut"] is True and d["flags"]["kl_cut_ever"] is True
    assert d["status"] == "done"  # (it stops nothing)
    prog = json.loads((job(out) / "progress.json").read_text())
    assert prog["flags"]["kl_cut"] is True and prog["minibatch_steps_last"] == [up[-1]["minibatch_steps_run"], 10]


def test_kl_share_and_small_helpers(tmp_path):
    from designlab.stage2 import train2

    rows = [dict(steps=128 * i, minibatch_steps_run=r, minibatch_steps_possible=10)
            for i, r in enumerate((10, 4, 4, 10, 10, 3), start=1)]
    assert train2.kl_share(rows, 0, 384) == dict(after_steps=0, to_steps=384, updates=3, median_share=0.4, cut=True)
    assert train2.kl_share(rows, 384, 768)["cut"] is False and train2.kl_share(rows, 768, 900)["median_share"] is None
    # the first rollout start at or past each multiple, 0 included
    starts = [n for n in range(0, 2049, 128) if train2.is_boundary(n, 500, 128)]
    assert starts == [0, 512, 1024, 1536, 2048]
    assert train2.boundary_seed(42, 3) == 4_200_300 and train2.job_name("R5", "75-M-H-10-X", 42) == "R5-75-M-H-10-X-s42"
    assert train2.base_body("75-M-H-10-X") == "75-M-H-10" == train2.base_body("75-M-H-10")
    # updates.jsonl is cut back to a step; a line cut by a kill goes with the others
    path = tmp_path / "updates.jsonl"
    path.write_text("".join(json.dumps(dict(steps=128 * i)) + "\n" for i in range(1, 6)) + '{"steps": 7')
    assert train2._cut_updates(path, 384, 2) == 3
    assert [u["steps"] for u in train2.read_updates(path)] == [128, 256, 384]
    assert (tmp_path / "updates.discarded.s2.jsonl").read_text().splitlines()[-1] == '{"steps": 7'
    assert train2.filter_sigma(20.0) == pytest.approx(0.081, abs=5e-4) and train2.filter_sigma(10.0) == 0.1


def test_sources():
    from designlab.stage2 import train2

    s = train2.sources()
    assert sorted(s["files"]) == sorted(train2.SOURCE_FILES) and len(s["sha256"]) == 64
    for m in ("tbenv", "schedules", "s2load", "r5lib", "r5core", "r5env", "g8a", "vecpool", "train2", "monitor2",
              "clip2", "bar", "readings", "setio", "s2io", "motion"):
        assert len(s["files"][f"designlab/stage2/{m}.py"]) == 64, m
    for m in ("direction_commands", "command_frame", "policy_loading", "curriculum/checkpoints"):
        assert len(s["files"][f"../../environments/shared/{m}.py"]) == 64, m
    assert train2.sources() == s


# ---------------------------------------------------------------------------------------------------------------
# MUJOCO_GL, the worker entry
# ---------------------------------------------------------------------------------------------------------------
def test_the_trainer_assigns_mujoco_gl(tmp_path):
    """A trainer started with ``MUJOCO_GL=osmesa`` builds a policy and runs one rollout: exit 0, ``disable`` in it
    and in its worker, and the worker (this module's ``worker`` entry) holds no torch."""
    _need_torch()
    env = _env(tmp_path, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    r = subprocess.run([PY, "-m", "designlab.stage2.train2", "preflight"], cwd=STUDY, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["ok"] is True and out["mujoco_gl"] == "disable" and out["pyopengl_platform"] is None
    w = out["worker"]
    assert w["mujoco_gl"] == "disable" and w["pyopengl_platform"] is None
    assert w["torch"] is False and w["stable_baselines3"] is False
    assert w["argv"][-4:-1] == ["-m", "designlab.stage2.train2", "worker"]
    assert w["threads"] == dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    assert w["pythonhashseed"] == "0" and out["bytecode_members"] == [] and out["torch_threads"] == 1
    assert out["steps"] == 32 and out["minibatch_steps_run"] > 0 and out["fingerprint_max_abs_diff"] == 0.0


def test_the_workers_of_a_run_are_the_trainers_worker_entry(stub, tmp_path):
    from designlab.stage2 import vecpool

    cfgs = stub_configs("R5", "stub", "STUB", None)
    venv = vecpool.PoolVecEnv(stub_specs("R5", cfgs), 2, worker_cmd=[PY, *stub.WORKER_CMD])
    try:
        rows = venv.env_method("where")
    finally:
        venv.close()
    assert len({r["pid"] for r in rows}) == 2 and os.getpid() not in {r["pid"] for r in rows}
    for r in rows:
        assert r["argv"][-4:-1] == ["-m", "designlab.stage2.train2", "worker"]
        assert r["mujoco_gl"] == "disable" and r["torch"] is False
    assert stub.main(["worker"]) == 2


def test_importing_the_trainer_imports_no_torch():
    code = ("import sys; from designlab.stage2 import train2, monitor2; "
            "print(sorted(m for m in ('torch', 'stable_baselines3', 'mujoco') if m in sys.modules))")
    r = subprocess.run([PY, "-c", code], cwd=STUDY, env=_env(Path(".")), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "[]", r.stderr[-2000:]


# ---------------------------------------------------------------------------------------------------------------
# The probes, the fine-tune
# ---------------------------------------------------------------------------------------------------------------
def test_probe_lr_stops_at_the_first_rate_that_passes(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(stub, "N_STEPS", 16)
    doc = stub.probe_lr(["stub", "stub-X"], "STUB", tmp_path / "lr_probe.json", rates=(1e-1, 1e-6, 1e-7), updates=2,
                        workers=0)
    assert json.loads((tmp_path / "lr_probe.json").read_text())["sha256"] == doc["sha256"]
    for body in ("stub", "stub-X"):
        row = doc["bodies"][body]
        assert [t["rate"] for t in row["tried"]] == [1e-1, 1e-6] and row["passed"] == 1e-6  # 1e-7 is not tried
        assert [t["passes"] for t in row["tried"]] == [False, True]
        for t in row["tried"]:
            assert [c["sigma"] for c in t["cells"]] == [0.1, 0.03]
            assert all(len(c["updates"]) == 2 for c in t["cells"])
        cut = row["tried"][0]["cells"]
        assert any(u["minibatch_steps_run"] < u["minibatch_steps_possible"] for c in cut for u in c["updates"])
        low = row["tried"][1]["cells"][1]["updates"]
        assert all(u["std_mean"] == pytest.approx(0.03, rel=1e-3) for u in low)
    assert doc["rates_by_body"] == dict(stub=dict(rate=1e-6, heads={"stub": 1e-6, "stub-X": 1e-6}))
    assert doc["value"] == 1e-6 and doc["seed"] == 42 and doc["updates"] == 2


def test_a_fine_tune_starts_from_its_parents_checkpoint(ref_plain, stub, tmp_path, monkeypatch):
    """Rule K8: a new job directory and run id from the parent's checkpoint and statistics, a constant rate."""
    monkeypatch.setenv("STUB_SET_FILE", str(ref_plain / "set.json"))
    parent = job(ref_plain / "out")
    out = tmp_path / "out"
    d = run(stub, out, job="R5-stub-s42-ft", parent=(parent, FINAL), steps=512, lr=1e-5)
    jd = job(out, "R5-stub-s42-ft")
    pdoc = doc_of(ref_plain / "out")
    assert d["status"] == "done" and d["run_uuid"] != pdoc["run_uuid"] and d["num_timesteps"] == 512
    assert d["parent"]["job"] == "R5-stub-s42" and d["parent"]["run_uuid"] == pdoc["run_uuid"]
    assert d["parent"]["steps"] == FINAL and d["parent"]["fingerprint_max_abs_diff"] == 0.0
    assert d["settings"]["lr"] == dict(start=1e-5, end=1e-5, source="given (--lr)", schedule="constant")
    assert all(u["lr"] == pytest.approx(1e-5) for u in updates(jd))
    assert d["sessions"][0]["boundary_seeds"][0]["steps"] == 0 and d["checkpoints"] == [0, 256, 512]
    # the fine-tune went on from the parent's statistics (updated further) and wrote checkpoints of its own run
    a, b = tensors(parent, FINAL), tensors(jd, 512)
    c512 = json.loads((jd / "checkpoints" / "checkpoint_512.json").read_text())
    assert c512["run_uuid"] == d["run_uuid"] and c512["settings_sha256"] == d["settings_sha256"]
    assert float(b["obs_count"]) == pytest.approx(float(a["obs_count"]) + 512 + 2 * 8)  # (512 steps, 2 boundaries)
    assert float(b["policy.pth/log_std"].mean()) == pytest.approx(float(a["policy.pth/log_std"].mean()), abs=1e-3)
    # without --lr the rate is rule K8's probe from the parent's checkpoint, run once and kept
    d2 = run(stub, tmp_path / "out2", job="ft2", parent=(parent, 1280), steps=256)
    p = 0.3 * 3e-5 + 0.7 * 3e-5 * (1.0 - 1280 / FINAL)
    assert d2["status"] == "done" and d2["settings"]["lr"]["start"] == pytest.approx(p) == d2["settings"]["lr"]["end"]
    assert d2["settings"]["lr"]["source"].startswith("rule K8") and d2["parent"]["lr_probe"]["rate"] == pytest.approx(p)
    probe = json.loads((job(tmp_path / "out2", "ft2") / d2["parent"]["lr_probe"]["record"]).read_text())
    assert probe["sha256"] == d2["parent"]["lr_probe"]["sha256"] and probe["updates"] == 6
    assert probe["bodies"]["stub"]["tried"][0]["passes"] is True and len(probe["bodies"]["stub"]["tried"]) == 1
    # rule K8's probe from the parent's checkpoint: p first, then the listed rates below it
    monkeypatch.setattr(stub, "FINE_TUNE_RATES", (1e-1, 1e-7))
    probe = stub.probe_lr(["stub"], "STUB", tmp_path / "probe.json", updates=2, workers=0, parent=(parent, 1280))
    p = probe["parent"]["p"]
    assert p == pytest.approx(0.3 * 3e-5 + 0.7 * 3e-5 * (1.0 - 1280 / FINAL))
    assert probe["rates"] == [p, 1e-7] and probe["sigmas"] == ["the parent's"]
    assert probe["bodies"]["stub"]["passed"] == p and probe["bodies"]["stub"]["tried"][0]["rate"] == p


# ---------------------------------------------------------------------------------------------------------------
# Slow: the real env on the Stage 1 view
# ---------------------------------------------------------------------------------------------------------------
def _have_s1() -> bool:
    return (STUDY / "stage1" / "runs" / "2026-11-08" / BODY / "refs" / "nominal.npz").is_file()


def _real_env(tmp_path) -> dict:
    return _env(tmp_path)


def _tiny_cmd(mode, out, *extra) -> list:
    return [PY, "-m", "designlab.stage2.train2", "run", "--mode", mode, "--body", BODY, "--seed", "42", "--set",
            "S1", "--tiny", "--workers", "2", "--out", str(out), *extra]


def _kill_after_first_checkpoint(mode, out, tmp_path) -> None:
    """Start a tiny run and kill its process group with SIGKILL once checkpoint 1,024 is whole."""
    name = f"{mode}-{BODY}-s42"
    p = subprocess.Popen(_tiny_cmd(mode, out), cwd=STUDY, env=_real_env(tmp_path), start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 900
        while not (job(out, name) / "checkpoints" / "checkpoint_1024.json").exists():
            assert p.poll() is None, p.stderr.read().decode()[-3000:]
            assert time.monotonic() < deadline, "no first checkpoint"
            time.sleep(0.2)
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        p.stderr.close()


@slow
def test_tiny_r5_run_resumes(tmp_path):
    """A tiny R5 run on the S1 view killed after its first checkpoint and resumed to its budget: its final
    checkpoint is the uninterrupted tiny run's, bit for bit (2 workers)."""
    _need_torch()
    if not _have_s1():
        pytest.skip("Stage 1's arrays are not restored")
    name = f"R5-{BODY}-s42"
    r = subprocess.run(_tiny_cmd("R5", tmp_path / "ref"), cwd=STUDY, env=_real_env(tmp_path), capture_output=True,
                       text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    ref = doc_of(tmp_path / "ref", name)
    assert ref["status"] == "done" and ref["num_timesteps"] == 4096 and ref["gates"]["g8a"]["passes"] is True
    assert [e["at_steps"] for e in ref["monitor"]] == [2048, 4096] and ref["baseline"] is not None
    _kill_after_first_checkpoint("R5", tmp_path / "out", tmp_path)
    r = subprocess.run(_tiny_cmd("R5", tmp_path / "out", "--resume"), cwd=STUDY, env=_real_env(tmp_path),
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    d = doc_of(tmp_path / "out", name)
    assert d["status"] == "done" and d["num_timesteps"] == 4096 and d["sessions"][1]["resumed_from"] >= 1024
    assert d["sessions"][1]["fingerprint_max_abs_diff"] == 0.0
    assert_same_checkpoint(job(tmp_path / "ref", name), job(tmp_path / "out", name), 4096)
    same_updates(job(tmp_path / "ref", name), job(tmp_path / "out", name))
    assert d["fix"] == ref["fix"] and d["gates"]["g8b"] == ref["gates"]["g8b"]


@slow
def test_tiny_r6_run_resumes(tmp_path):
    """A tiny R6 run on the S1 view killed after its first checkpoint and resumed to its budget."""
    _need_torch()
    if not _have_s1():
        pytest.skip("Stage 1's arrays are not restored")
    name = f"R6-{BODY}-s42"
    _kill_after_first_checkpoint("R6", tmp_path / "out", tmp_path)
    r = subprocess.run(_tiny_cmd("R6", tmp_path / "out", "--resume"), cwd=STUDY, env=_real_env(tmp_path),
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    d = doc_of(tmp_path / "out", name)
    assert d["mode"] == "R6" and d["status"] in ("done", "stopped_g8b") and d["sessions"][1]["resumed_from"] >= 1024
    assert d["num_timesteps"] == (4096 if d["status"] == "done" else d["stopped"]["at_steps"])
    assert len(d["sessions"]) == 2 and d["sessions"][1]["fingerprint_max_abs_diff"] == 0.0
    steps = [u["steps"] for u in updates(job(tmp_path / "out", name))]
    assert steps == sorted(set(steps))


@slow
def test_tiny_probes_on_the_s1_view(tmp_path, monkeypatch):
    """Rule O2's and rule O4's episodes run on the real env (one episode per task, two sigmas)."""
    _need_torch()
    if not _have_s1():
        pytest.skip("Stage 1's arrays are not restored")
    from designlab.stage2 import train2

    monkeypatch.setattr(train2, "PROBE_EPISODES", (7000,))
    doc = train2.probe_sigma([BODY], "S1", tmp_path, sigmas=(0.0, 0.1))
    per = doc["bodies"][BODY]["sigmas"]
    assert set(per) == {"0", "0.1"} and per["0"]["episodes"] == 2 == per["0.1"]["episodes"]
    assert per["0.1"]["mean_survival_s"] <= per["0"]["mean_survival_s"]
    assert (tmp_path / "sigma_probe.json").is_file()
    doc = train2.filter_choice([BODY], "S1", tmp_path)
    cut = doc["bodies"][BODY]["cutoffs"]
    assert cut["10"]["sigma"] == 0.1 and cut["20"]["sigma"] == pytest.approx(0.081, abs=5e-4)
    assert doc["choice"]["body"] == BODY and doc["choice"]["filter_hz"] in (10.0, 20.0)
    assert doc["choice"]["filter_hz"] == (20.0 if doc["bodies"][BODY]["ratio"] >= 0.9 else 10.0)
    # rule O3's probe on the real env: one rate, one update of 8 x 64 steps at each sigma, 2 workers
    monkeypatch.setattr(train2, "N_STEPS", 64)
    monkeypatch.chdir(STUDY)
    doc = train2.probe_lr([BODY], "S1", tmp_path, rates=(3e-5,), updates=1, workers=2)
    row = doc["bodies"][BODY]
    assert [c["sigma"] for c in row["tried"][0]["cells"]] == [0.1, 0.03] and row["floors"].count("hard") == 6
    assert all(len(c["updates"]) == 1 and c["updates"][0]["minibatch_steps_possible"] == 10
               for c in row["tried"][0]["cells"])
    assert row["tried"][0]["cells"][1]["updates"][0]["std_mean"] == pytest.approx(0.03, rel=1e-2)
    assert doc["rates_by_body"][BODY]["rate"] == row["passed"] and row["task_names"] == ["stand", "walk", "brisk",
                                                                                        "stop"]


if __name__ == "__main__":
    from tests import test_stage2_train2 as _named  # the importable copy owns the stub env's class

    sys.exit(_named._main(sys.argv[1:]))
