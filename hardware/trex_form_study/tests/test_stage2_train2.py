"""``designlab.stage2.train2``: the trainer of R5 and R6 (round C design 3.6 and section 4's rows "Kill and resume",
"Lesson 16", "``MUJOCO_GL``"; Track B rules K1-K5, M1, M4, O3).

What is held here, on a stub env and a stub monitor (no plant, no reference set):

* **kill and resume**: a run killed with SIGKILL between two boundaries and resumed with another worker count ends
  with the uninterrupted run's policy tensors, optimizer state, normalisation statistics (their count included,
  also at the first checkpoint after the resume) and fingerprint, bit for bit; so does a run killed while its
  monitor child runs (the attempt is logged before the child starts and is an interruption), a run killed after
  G8b's fix and before the next checkpoint (the fix is recorded once), a run whose ``run.json`` is ahead of its
  newest intact checkpoint (later entries move to ``discarded_after_resume``; ``updates.jsonl`` holds no step
  twice), and a job that has a ``run.json`` and no checkpoint (it restarts from step 0 in place); a trainer killed
  alone while its monitor child runs (the resume ends the child that outlived it); a kill between a checkpoint's
  last file and ``run.json``; a ``monitor_<n>.json`` or ``r0l.json`` that was cut (no record: kept beside, made
  again); a lost baseline made again (it must be the one ``run.json`` names, else the run halts);
* **what a resume refuses**, in the rule's order: no ``run.json``, a stopped run, other settings, other inputs (a
  changed file refused; a later freeze of the set with unchanged files accepted, both freezes recorded), other
  sources and other library versions (each refused without its flag, accepted and flagged with it), a checkpoint
  of another run; a broken newest checkpoint is skipped with its reason; the stored rule text names that order;
  the trainer refuses to start without ``PYTHONHASHSEED=0``;
* **a check's attempts** (rule M1): logged before the monitor starts and closed with its exit code; after two closed
  attempts without a record the check counts as a failed G8b reading; after two attempts that were never closed
  the run halts; ``progress.json`` names a check counted without a record, and a halt;
* **lesson 16**: with a learning rate that trips ``target_kl`` ``minibatch_steps_run < minibatch_steps_possible``,
  and the ``kl_cut`` flag is set from the median share over a check interval;
* **``MUJOCO_GL``**: a trainer started with ``MUJOCO_GL=osmesa`` runs a rollout with ``disable`` in itself and in
  its worker, and the worker holds no torch; an archive the trainer saves has no bytecode member; the schedule's
  value follows the step count across a resume;
* the probes on the stub (rule O3's stops at the first rate that passes both sigmas; rule O4's sigma); a later
  probe adds its heads, or its bodies, to the validation record and probes none again; a run that is not tiny and
  rule O3's probe refuse without rule O4's filter; rule K8's probe runs on the fine-tune's own envs.

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


def with_filter(body, set_name) -> dict:
    """A validation directory that holds rule O4's choice and no rate."""
    return dict(lr_probe=None, filter=dict(path="stage2/tb/validation/filter.json", sha256="f" * 64, filter_hz=10.0))


#: ``train2``'s own ``_validation`` (the ``stub`` fixture replaces it).
REAL: dict = {}


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
    """The stub of ``lookbar.episodes``: R0L reads every episode (survival 10 s, or ``STUB_R0L_SURVIVAL``); a
    policy row reads them too and survives 10.5 s, but ``walk/nominal`` falls at the steps of
    ``STUB_MON_FAIL_AT``."""
    n = None if not policies else int(next(iter(policies.values()))["steps"])
    down = n is not None and n in _steps_env("STUB_MON_FAIL_AT")
    out = []
    for cond, seed in pairs:
        fell = bool(down and cond == "walk/nominal")
        surv = float(os.environ.get("STUB_R0L_SURVIVAL", 10.0)) if n is None else (3.0 if fell else 10.5)
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

    REAL.setdefault("_validation", train2._validation)
    for name, fn in (("_env_configs", stub_configs), ("_env_specs", stub_specs), ("_g8a_short", stub_g8a),
                     ("_monitor_cmd", stub_monitor_cmd), ("_validation", stub_validation)):
        monkeypatch.setattr(train2, name, fn)
    for k in list(os.environ):
        if k.startswith("STUB_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("STUB_SET_FILE", str(_set_file(tmp_path)))
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # (what train() asks of its process; the children get it anyway)
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


def process_alive(pid: int) -> bool:
    """Whether the process runs (a process that has ended and waits to be collected does not)."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rindex(")") + 2:].split()[0] != "Z"


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
    assert prog["last_monitor"]["counted_without_record"] is False and prog["checks_without_record"] == []
    assert prog["halt"] is None
    # every attempt holds its monitor child's pid and start time; the baseline's episodes are in run.json
    atts = d["baseline_attempts"] + [a for e in d["monitor"] for a in e["attempts"]]
    assert len(atts) == 4 and all(a["pid"] > 0 and a["pid_start"] > 0 and a["closed_at"] for a in atts)
    assert len({a["pid"] for a in atts}) == 4 and "ended_children" not in d["sessions"][0]
    r0l = json.loads((jd / "monitor" / "r0l.json").read_text())
    assert d["baseline"]["sha256"] == r0l["sha256"] and "remade" not in d["baseline"]
    assert d["baseline"]["episodes"] == [dict(condition=e["condition"], seed=e["seed"], outcome="read",
                                              survival_s=10.0) for e in r0l["episodes"]]
    assert [e["seed"] for e in d["baseline"]["episodes"]] == [6001, 6004, 6007, 6010, 6013, 6016]
    assert set(d["rules"]) >= {"resume", "baseline", "children"}


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


def test_a_monitor_child_that_outlived_its_trainer_is_ended_at_the_resume(ref_fix, tmp_path):
    """The trainer alone is killed while the monitor child of the first check runs (not its process group): the
    child lives on.  The attempt holds the child's pid; the resume ends that child before it runs the check again,
    so two monitors never write one record, and the run ends as the uninterrupted one."""
    out, hold = tmp_path / "out", tmp_path / "hold"
    hold.write_text("")
    _set_file(tmp_path)
    args = dict(KW, out=str(out), workers=0)
    p = subprocess.Popen([PY, "-m", HERE, "drive", json.dumps(args)], cwd=STUDY, start_new_session=True,
                         env=_env(tmp_path, STUB_MON_FAIL_AT="512", STUB_MON_HOLD=hold),
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 240
        while not Path(str(hold) + ".started").exists():
            assert p.poll() is None, p.stderr.read().decode()[-3000:]
            assert time.monotonic() < deadline, "the monitor child did not start"
            time.sleep(0.05)
        child = int(Path(str(hold) + ".started").read_text())
        while "pid" not in doc_of(out)["monitor"][0]["attempts"][0]:
            assert time.monotonic() < deadline, "the attempt never took its child's pid"
            time.sleep(0.05)
        att = doc_of(out)["monitor"][0]["attempts"][0]
        assert att["pid"] == child and att["pid_start"] > 0 and att["closed_at"] is None
        os.kill(p.pid, signal.SIGKILL)  # the trainer alone
        p.wait()
        time.sleep(1.0)
        assert process_alive(child)  # (the monitor child outlived its trainer)
        hold.unlink()
        r = drive(tmp_path, out, env=dict(STUB_MON_FAIL_AT="512"), workers=0, resume=True)
        assert r.returncode == 0, r.stderr[-3000:]
        assert not process_alive(child)
        assert f"the monitor child {child} of session 0 (check at 512 steps) outlived its trainer" in r.stdout
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        p.stderr.close()
    d = doc_of(out)
    e = d["monitor"][0]
    assert d["status"] == "done" and d["sessions"][1]["resumed_from"] == 512
    assert d["sessions"][1]["ended_children"] == [dict(kind="check", at_steps=512, pid=child, session=0)]
    assert [bool(a.get("interrupted")) for a in e["attempts"]] == [True, False]
    assert e["attempts"][0]["child_ended"]["session"] == 1 and e["attempts"][0]["closed_at"] is None
    assert e["attempts"][1]["exit_code"] == 0 and e["attempts"][1]["pid"] != child
    assert e["counted_without_record"] is False and d["fix"]["at_steps"] == 512
    for n in (1024, FINAL):
        assert_same_checkpoint(job(ref_fix / "out"), job(out), n)
    same_updates(job(ref_fix / "out"), job(out))


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
    monkeypatch.setenv("PYTHONHASHSEED", "0")
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


def test_a_kill_between_a_checkpoints_last_file_and_run_json(ref_plain, stub, tmp_path, monkeypatch):
    """The session ends when checkpoint 768 has its four files and ``run.json`` does not name it yet: the resume
    starts from it, and ``run.json``'s list of checkpoints is the uninterrupted run's."""
    from designlab.stage2 import s2load

    out = tmp_path / "out"
    inner = s2load.save_checkpoint

    def save_checkpoint(job_dir, steps, *a, **k):
        rec = inner(job_dir, steps, *a, **k)
        if int(steps) == 768:
            raise _Cut("cut after checkpoint 768's last file")
        return rec

    monkeypatch.setattr(s2load, "save_checkpoint", save_checkpoint)
    with pytest.raises(_Cut):
        run(stub, out)
    monkeypatch.undo()
    stub = install_again(monkeypatch, tmp_path)
    assert doc_of(out)["checkpoints"] == [0, 256, 512]
    assert (job(out) / "checkpoints" / "checkpoint_768.json").is_file()
    d = run(stub, out, resume=True)
    ref = doc_of(ref_plain / "out")
    assert d["status"] == "done" and d["sessions"][1]["resumed_from"] == 768
    assert d["checkpoints"] == ref["checkpoints"] == [0, 256, 512, 768, 1024, 1280, 1536]
    assert d["kept"] == ref["kept"]
    assert_same_checkpoint(job(ref_plain / "out"), job(out), FINAL)
    same_updates(job(ref_plain / "out"), job(out))


def _cut_with_gates(stub, out, tmp_path, monkeypatch, at=640):
    """A run with its checks cut at ``at`` steps; ``train2`` with the stub seams again."""
    monkeypatch.setenv("STUB_MON_FAIL_AT", "512")
    cut_at(monkeypatch, stub, at)
    with pytest.raises(_Cut):
        run(stub, out, gates=True)
    monkeypatch.undo()
    stub = install_again(monkeypatch, tmp_path)
    monkeypatch.setenv("STUB_MON_FAIL_AT", "512")
    return stub


def test_a_record_that_was_cut_is_no_record(ref_fix, stub, tmp_path, monkeypatch):
    """A ``monitor_<n>.json`` or an ``r0l.json`` that cannot be read (a copy that was cut short) is no record
    (rule M1): the resume does not fail on it; the file is kept beside, the check of the resumed step is run again
    and R0L's episodes are made again and compared with what ``run.json`` holds."""
    out = tmp_path / "out"
    stub = _cut_with_gates(stub, out, tmp_path, monkeypatch)
    jd = job(out)
    before = doc_of(out)["baseline"]
    for name in ("monitor_512.json", "r0l.json"):
        with open(jd / "monitor" / name, "r+b") as fh:
            fh.truncate(200)
    d = run(stub, out, gates=True, resume=True)
    s = d["sessions"][1]
    assert d["status"] == "done" and s["resumed_from"] == 512
    assert (jd / "monitor" / "monitor_512.discarded.s1.json").stat().st_size == 200
    assert (jd / "monitor" / "r0l.unreadable.s1.json").stat().st_size == 200
    e = d["monitor"][0]
    assert [(a["session"], a["exit_code"], a["record"]) for a in e["attempts"]] == [(0, 0, True), (1, 0, True)]
    assert e["counted_without_record"] is False and d["fix"]["at_steps"] == 512
    assert [(a["session"], a["record"]) for a in d["baseline_attempts"]] == [(0, True), (1, True)]
    assert d["baseline"]["remade"] == [dict(session=1, was_sha256=before["sha256"], equal=True, differ=[],
                                            episodes_compared=6, digests_compared=False)]
    assert d["baseline"]["sha256"] != before["sha256"] and d["baseline"]["episodes"] == before["episodes"]
    for n in (1024, FINAL):
        assert_same_checkpoint(job(ref_fix / "out"), jd, n)
    same_updates(job(ref_fix / "out"), jd)
    # an earlier check's record that cannot be read: the rules read records, and the reason is named
    with open(jd / "monitor" / "monitor_512.json", "r+b") as fh:
        fh.truncate(200)
    with pytest.raises(stub.TrainError, match="cannot be read"):
        stub._Run._history(_history_of(stub, d, jd), 1024)


def _history_of(train2, doc, jd):
    """A bare ``_Run`` holding a job's document: enough for ``_history``."""
    r = object.__new__(train2._Run)
    r.doc, r.jd, r.set, r.tiny = doc, jd, "STUB", True
    return r


def test_a_lost_baseline_made_again_must_be_the_runs(stub, tmp_path, monkeypatch):
    """R0L's record is lost between two sessions.  Made again, it must hold the episodes ``run.json`` names: the
    run's checks were read against them.  If it does not, nothing is replaced and the run halts for the lead."""
    out = tmp_path / "out"
    stub = _cut_with_gates(stub, out, tmp_path, monkeypatch)
    jd = job(out)
    before = doc_of(out)["baseline"]
    (jd / "monitor" / "r0l.json").unlink()
    monkeypatch.setenv("STUB_R0L_SURVIVAL", "9.0")
    with pytest.raises(stub.Halt, match="is not the one the run's checks were read against") as info:
        run(stub, out, gates=True, resume=True)
    assert info.value.code == stub.EXIT_HALT and "walk/nominal seed 6001: survival_s" in str(info.value)
    d = doc_of(out)
    assert d["baseline"] == before and d["halt"]["session"] == 1 and d["status"] == "partial"
    assert not (jd / "monitor" / "r0l.json").exists()
    kept = json.loads((jd / "monitor" / "r0l.differs.s1.json").read_text())
    assert kept["episodes"][0]["survival_s"] == 9.0 and d["num_timesteps"] == 512  # (no step was made)
    assert json.loads((jd / "progress.json").read_text())["halt"]["session"] == 1
    # the same episodes again: the new record takes the lost one's place, and the session is named
    monkeypatch.delenv("STUB_R0L_SURVIVAL")
    d = run(stub, out, gates=True, resume=True)
    assert d["status"] == "done" and d["halt"] is None and d["sessions"][2]["resumed_from"] == 512
    assert d["baseline"]["remade"] == [dict(session=2, was_sha256=before["sha256"], equal=True, differ=[],
                                            episodes_compared=6, digests_compared=False)]
    now = json.loads((jd / "monitor" / "r0l.json").read_text())
    assert d["baseline"]["sha256"] == now["sha256"] != before["sha256"]
    assert {k: v for k, v in d["baseline"].items() if k not in ("sha256", "remade")} == {
        k: v for k, v in before.items() if k != "sha256"}
    # a run.json of before this rule (no episodes under baseline) is compared in its summary
    old = {k: v for k, v in before.items() if k != "episodes"}
    assert stub.baseline_same(old, now) == dict(equal=True, differ=[], episodes_compared=None, digests_compared=None)
    assert stub.baseline_same(dict(old, S=9.5), now)["differ"] == ["S: 9.5 -> 10.0"]


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


def test_the_resume_rule_names_the_order_the_code_refuses_in(cut_run, tmp_path, monkeypatch):
    """``RESUME_RULE`` is stored in every ``run.json``: its order is the code's (and the design's): the settings,
    the inputs, the sources, the versions, the checkpoint."""
    from designlab.stage2 import s2load

    train2, out = cut_run
    text = train2.RESUME_RULE
    assert doc_of(out)["rules"]["resume"] == text
    at = [text.index(w) for w in ("no run.json", "a stopped run", "other settings", "other inputs", "other sources",
                                  "other library versions", "a checkpoint whose run_uuid",
                                  "a fingerprint that is not reproduced")]
    assert at == sorted(at)
    # a job where an argument, a used file, the sources and the versions all differ is refused in that order
    files = json.loads((tmp_path / "set.json").read_text())["files"]
    changed = json.loads(json.dumps(files))
    changed["transitions/start.npz"]["sha256"] = "9" * 64
    write_set(tmp_path / "set.json", files=changed)
    real = train2.sources()
    monkeypatch.setattr(train2, "sources", lambda: dict(real, sha256="1" * 64))
    vers = dict(s2load.versions(), numpy="9.9.9")
    monkeypatch.setattr(s2load, "versions", lambda: dict(vers))
    _refused(train2, out, "other settings", lr=1e-4)
    _refused(train2, out, "other inputs")
    write_set(tmp_path / "set.json", files=files)
    _refused(train2, out, "other sources")
    _refused(train2, out, "other library versions", allow_source_change=True)


def test_the_trainer_refuses_without_pythonhashseed_0(stub, tmp_path, monkeypatch):
    """Design D16: the trainer and its workers run with ``PYTHONHASHSEED=0``.  ``train()`` asks it of its own
    process, whoever calls it (not the command line alone)."""
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    with pytest.raises(stub.Refused, match="PYTHONHASHSEED=0") as info:
        run(stub, tmp_path / "out")
    assert info.value.code == stub.EXIT_REFUSED and "'7'" in str(info.value)
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(stub.Refused, match="PYTHONHASHSEED=0"):
        run(stub, tmp_path / "out")
    # the probes too, called or started from the command line (exit code 4: nothing was run)
    for probe in (lambda: stub.probe_lr(["stub"], "STUB", tmp_path / "lr.json", workers=0),
                  lambda: stub.probe_sigma(["stub"], "STUB", tmp_path / "sigma.json"),
                  lambda: stub.filter_choice(["stub"], "STUB", tmp_path / "filter.json")):
        with pytest.raises(stub.Refused, match="PYTHONHASHSEED=0"):
            probe()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(v, "1")
    for cmd in ("probe-lr", "probe-sigma", "filter"):
        assert stub.main([cmd, "--bodies", "stub", "--set", "STUB", "--out", str(tmp_path / "x.json")]) == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["set.json"]  # (the fixture's set file; nothing written)


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
                      "--run-id", "s2-4", "--retry-check", "--config", '{"filter_hz": 20.0, "servo_model": "m1"}'])
    log = seen.pop("log")
    assert code == 0 and callable(log) and seen == dict(
        mode="R6", body="75-M-H-10-X", seed=43, set_name="V6", out=str(tmp_path), job="ft", steps=5_000_000,
        resume=True, workers=3, monitor_every=500_000, monitor_also_k=(0, 2), lr=3e-5,
        config=dict(filter_hz=20.0, servo_model="m1"), gates=False, allow_version_change=True,
        allow_source_change=True, parent="/runs/jobs/p:40001536", tiny=False, run_id="s2-4", retry_check=True)
    assert stub._parent_of("/runs/jobs/p:40001536") == (Path("/runs/jobs/p"), 40_001_536)
    # the overrides reach every env config untouched (the stub's configs take any field)
    monkeypatch.setenv("STUB_SET_FILE", str(tmp_path / "set.json"))
    assert all(c["servo_model"] == "m1" and c["filter_hz"] == 20.0
               for c in stub._env_configs("R5", "stub", "STUB", dict(filter_hz=20.0, servo_model="m1")))
    # probe-lr: rule K8's probe takes the fine-tune's mode and its env config fields
    got = {}
    monkeypatch.setattr(stub, "probe_lr", lambda bodies, set_name, out, **kw: got.update(
        kw, bodies=bodies, set_name=set_name, out=out) or dict(rates_by_body={}))
    assert stub.main(["probe-lr", "--bodies", "a,b", "--set", "V6", "--out", "x.json", "--workers", "2", "--parent",
                      "/runs/jobs/p:512", "--mode", "R6", "--config", '{"filter_hz": 20.0}']) == 0
    assert got == dict(bodies=["a", "b"], set_name="V6", out="x.json", workers=2, parent="/runs/jobs/p:512",
                       mode="R6", config=dict(filter_hz=20.0))
    got.clear()
    assert stub.main(["probe-lr", "--bodies", "a", "--out", "x.json"]) == 0
    assert (got["mode"], got["config"], got["parent"], got["set_name"]) == ("R5", None, None, "V6")
    # defaults: the design's intervals, one worker per vCPU, gates on
    seen.clear()
    assert stub.main(["run", "--mode", "R5", "--body", "b", "--seed", "42", "--out", str(tmp_path)]) == 0
    assert (seen["monitor_every"], seen["workers"], seen["gates"], seen["tiny"], seen["set_name"],
            seen["retry_check"]) == (2_500_000, None, True, False, "V6", False)
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
    assert len(e["episodes"]) == 6 and {x["outcome"] for x in e["episodes"]} == {"error"}  # entered as errors
    assert [x["seed"] for x in e["episodes"]] == [6001, 6004, 6007, 6010, 6013, 6016]
    assert e["gain"]["gain"] is False and e["outcomes"]["error"] == 6 and e["N"] == 0 and e["S"] == 0.0
    assert d["gates"]["g8b"]["pass"] is False and d["gates"]["g8b"]["up"] == 0
    assert d["fix"]["tasks"] == ["walk_nominal", "walk_v0.4", "turn_q90_left", "turn_q90_right", "stop",
                                 "start_stop"]
    assert d["fix"]["monitor"] is None and d["gates"]["g8b_after_fix"]["pass"] is True
    assert len(d["monitor"][1]["attempts"]) == 1 and d["monitor"][1]["counted_without_record"] is False
    # the launcher reads it in progress.json (rule M1: the lead is told), also after later checks
    prog = json.loads((job(out) / "progress.json").read_text())
    assert prog["checks_without_record"] == [512] and prog["last_monitor"]["at_steps"] == FINAL
    assert prog["last_monitor"]["counted_without_record"] is False


def test_progress_names_a_check_counted_without_a_record(stub, tmp_path, monkeypatch):
    """What the launcher sees one rollout after such a check: not only N 0 and S 0."""
    out, crash = tmp_path / "out", tmp_path / "crash"
    crash.write_text("2")
    monkeypatch.setenv("STUB_MON_CRASH", str(crash))
    cut_at(monkeypatch, stub, 640)
    with pytest.raises(_Cut):
        run(stub, out, gates=True)
    prog = json.loads((job(out) / "progress.json").read_text())
    last = prog["last_monitor"]
    assert last["at_steps"] == 512 and (last["N"], last["S"]) == (0, 0.0) and last["counted_without_record"] is True
    assert prog["checks_without_record"] == [512] and prog["halt"] is None and prog["fix"] == 512


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
    assert d["halt"]["session"] == 1 and "--retry-check" in d["halt"]["reason"] and d["sessions"][1]["retry_check"] is False
    prog = json.loads((jd / "progress.json").read_text())  # (the launcher reads the halt there)
    assert prog["status"] == "partial" and prog["halt"] == d["halt"] and "never closed" in prog["halt"]["reason"]
    # the lead's fifth list of 16 Nov 2026: with the fault repaired, a resume with the recorded parameter lets
    # the check run once more per resume.  A retry that leaves no record halts again ...
    crash = tmp_path / "crash"
    crash.write_text("1")
    monkeypatch.setenv("STUB_MON_CRASH", str(crash))
    with pytest.raises(stub.Halt, match="never closed"):
        run(stub, out, gates=True, resume=True, retry_check=True)
    d = doc_of(out)
    atts = d["monitor"][0]["attempts"]
    assert len(atts) == 3 and atts[2]["after_halt"] is True and (atts[2]["exit_code"], atts[2]["record"]) == (7, False)
    assert d["sessions"][2]["retry_check"] is True and d["halt"]["session"] == 2
    # ... and the next one, whose monitor works, goes on to the run's end
    d = run(stub, out, gates=True, resume=True, retry_check=True)
    atts = d["monitor"][0]["attempts"]
    assert d["status"] == "done" and d["halt"] is None and len(atts) == 4
    assert atts[3]["after_halt"] is True and atts[3]["record"] is True and atts[3]["session"] == 3
    assert d["monitor"][0]["counted_without_record"] is False and len(d["monitor"][1]["attempts"]) == 1
    assert "after_halt" not in d["monitor"][1]["attempts"][0] and d["sessions"][3]["retry_check"] is True
    assert json.loads((jd / "progress.json").read_text())["halt"] is None


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
    prog = json.loads((job(out) / "progress.json").read_text())
    assert prog["status"] == "partial" and "R0L baseline could not be made" in prog["halt"]["reason"]


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


def test_what_a_resume_compares_is_the_envs_own_keys(stub, tmp_path, monkeypatch):
    """Rule K4: of the description of the envs as built the trainer compares ``r5env.RESUME_KEYS``; ``config`` is
    the stored config the env was built from."""
    from designlab.stage2 import r5env

    assert r5env.RESUME_KEYS == ("config", "task_names", "tasks", "task_files", "set_freeze_started")
    desc = make_stub(stub_configs("R5", "stub", "STUB", None)[0]).describe()
    view = stub._resume_view(desc)
    assert set(view) == set(r5env.RESUME_KEYS) - {"config"} | {"plant_inputs"}
    import importlib

    named = importlib.import_module(HERE)  # (the copy of this module the trainer builds its envs from)
    monkeypatch.setattr(named.StubEnv, "describe",
                        lambda self: dict(desc, config=dict(desc["config"], filter_hz=99.0)))
    with pytest.raises(stub.TrainError, match="not the config it was built from"):
        run(stub, tmp_path / "out")


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


def test_the_packages_processes_take_no_lock_on_the_checkouts_index(tmp_path):
    """``s2io.git_state`` runs ``git status``, which rewrites the index under ``index.lock``; a SIGKILL of the
    job's process group in that instant leaves the lock behind, and later git commands in the checkout refuse to
    run.  The trainer, the monitor and the clip tool set ``GIT_OPTIONAL_LOCKS=0`` unless the caller set it, and
    their children inherit it; with it ``git status`` leaves the index alone."""
    env = {k: v for k, v in _env(tmp_path).items() if k != "GIT_OPTIONAL_LOCKS"}
    for module, gl in (("train2", "disable"), ("monitor2", "disable"), ("clip2", "osmesa")):
        code = (f"import os; from designlab.stage2 import {module}, tbenv; "
                f"print(os.environ.get('GIT_OPTIONAL_LOCKS'), tbenv.child_env('{gl}').get('GIT_OPTIONAL_LOCKS'))")
        r = subprocess.run([PY, "-c", code], cwd=STUDY, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and r.stdout.split() == ["0", "0"], (module, r.stdout, r.stderr[-2000:])
        r = subprocess.run([PY, "-c", code], cwd=STUDY, env=dict(env, GIT_OPTIONAL_LOCKS="1"), capture_output=True,
                           text=True, timeout=120)
        assert r.stdout.split() == ["1", "1"], module  # (the caller's own setting stands)
    # what the variable buys, in a scratch repository: a file whose time moved makes a plain status rewrite the
    # index, and a status with the variable not
    import shutil

    if shutil.which("git") is None:
        pytest.skip("no git on PATH")
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, **extra):
        e = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        e.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, **extra)
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                               *args], env=e, capture_output=True, text=True, check=True).stdout

    git("init", "-q", ".")
    (repo / "f").write_text("a")
    git("add", "f")
    git("commit", "-q", "-m", "one file")
    index = repo / ".git" / "index"
    for i, (extra, rewritten) in enumerate(((dict(GIT_OPTIONAL_LOCKS="0"), False), ({}, True))):
        t = time.time() + 100 * (i + 1)
        os.utime(repo / "f", (t, t))
        before = index.read_bytes()
        assert git("status", "--porcelain", **extra) == ""
        assert (index.read_bytes() != before) is rewritten


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
    # rule O3's probe runs with rule O4's filter, which is measured first: without it nothing is probed
    built = []
    monkeypatch.setattr(stub, "_env_configs", lambda *x: built.append(x) or stub_configs(*x))
    with pytest.raises(stub.Refused, match="probe-lr stub: no action filter"):
        stub.probe_lr(["stub", "stub-X"], "STUB", tmp_path / "lr_probe.json", rates=(1e-1, 1e-6, 1e-7), updates=2,
                      workers=0)
    assert not built and not (tmp_path / "lr_probe.json").exists()
    monkeypatch.setattr(stub, "_validation", with_filter)
    doc = stub.probe_lr(["stub", "stub-X"], "STUB", tmp_path / "lr_probe.json", rates=(1e-1, 1e-6, 1e-7), updates=2,
                        workers=0)
    assert json.loads((tmp_path / "lr_probe.json").read_text())["sha256"] == doc["sha256"]
    assert doc["added"] == ["stub", "stub-X"] and doc["mode"] == "R5" and doc["parent"] is None
    for body in ("stub", "stub-X"):
        row = doc["bodies"][body]
        assert (row["mode"], row["config"], row["filter_hz"]) == ("R5", {}, 10.0)
        assert row["validation"]["filter"]["filter_hz"] == 10.0 and row["probed"]["workers"] == 0
        assert set(row["probed"]) == {"versions", "provenance", "at", "workers"}
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


def test_a_later_probe_adds_its_heads_to_the_record(ref_plain, stub, tmp_path, monkeypatch):
    """Rule O3 as settled: the heads of a body are probed at different times, each once and never again; a body's
    rate is the smallest passing rate over every head the record holds.  A later call must not drop what an
    earlier one found."""
    from designlab.stage2 import s2io

    passing = {"at": 1e-6}

    def cell(venv, layout, rate, sigma, updates, n_steps, parent=None):  # (no PPO: the largest rate that "passes")
        return dict(sigma=sigma, updates=[], passes=bool(rate <= passing["at"]))

    monkeypatch.setattr(stub, "_probe_cell", cell)
    monkeypatch.setattr(stub, "_validation", with_filter)
    val = tmp_path / "validation"
    kw = dict(rates=(1e-5, 1e-6, 1e-7), workers=0)
    a = stub.probe_lr(["stub"], "STUB", val, **kw)
    assert a["rates_by_body"] == dict(stub=dict(rate=1e-6, heads={"stub": 1e-6})) and a["added"] == ["stub"]
    passing["at"] = 1e-7
    b = stub.probe_lr(["stub-X"], "STUB", val, **kw)  # the -X head, later, into the same record
    assert b["bodies"]["stub"] == a["bodies"]["stub"] and list(b["bodies"]) == ["stub", "stub-X"]
    assert b["rates_by_body"] == dict(stub=dict(rate=1e-7, heads={"stub": 1e-6, "stub-X": 1e-7}))
    assert b["value"] == 1e-7 and b["added"] == ["stub-X"]
    assert json.loads((val / "lr_probe.json").read_text()) == b
    # what a run of the body takes when it starts, and the heads its rate rests on (the record read back)
    monkeypatch.setattr(stub, "VALIDATION", val)
    for body in ("stub", "stub-X"):
        got = REAL["_validation"](body, "STUB")["lr_probe"]
        assert got["rate"] == 1e-7 and got["heads"] == {"stub": 1e-6, "stub-X": 1e-7}
        assert got["sha256"] == s2io.sha256_file(val / "lr_probe.json")
    # a head the record holds is not probed again; a record of other settings is not added to; no env is built
    built = []
    monkeypatch.setattr(stub, "_env_configs", lambda *x: built.append(x) or stub_configs(*x))
    keep = (val / "lr_probe.json").read_bytes()
    with pytest.raises(stub.Refused, match="already probed: stub-X"):
        stub.probe_lr(["other", "stub-X"], "STUB", val, **kw)
    with pytest.raises(stub.Refused, match="other settings.*rates"):
        stub.probe_lr(["other"], "STUB", val, rates=(1e-5,), workers=0)
    with pytest.raises(stub.Refused, match="other settings.*set"):
        stub.probe_lr(["other"], "S9", val, **kw)
    with pytest.raises(stub.Refused, match="not the record of this parent's probe"):  # rule K8's probe is its job's
        stub.probe_lr(["stub"], "STUB", val, workers=0, parent=(job(ref_plain / "out"), 1280))
    assert not built and (val / "lr_probe.json").read_bytes() == keep
    # another body is its own group; a head without a passing rate leaves its body without one
    passing["at"] = 1e-5
    c = stub.probe_lr(["other"], "STUB", val, **kw)
    assert c["rates_by_body"] == dict(b["rates_by_body"], other=dict(rate=1e-5, heads={"other": 1e-5}))
    assert c["value"] == 1e-7 and len(built) == 1
    passing["at"] = 0.0
    e = stub.probe_lr(["other-X"], "STUB", val, **kw)
    assert e["rates_by_body"]["other"] == dict(rate=None, heads={"other": 1e-5, "other-X": None})
    assert e["value"] is None and e["rates_by_body"]["stub"] == b["rates_by_body"]["stub"]
    assert REAL["_validation"]("other", "STUB")["lr_probe"] is None  # (nothing trains on that body: the lead)
    # a record that was changed by hand is not added to and not replaced
    doc = json.loads((val / "lr_probe.json").read_text())
    doc["rates_by_body"]["stub"]["rate"] = 1e-4
    (val / "lr_probe.json").write_text(json.dumps(doc))
    with pytest.raises(stub.Refused, match="hashes to its own sha256"):
        stub.probe_lr(["third"], "STUB", val, **kw)
    assert json.loads((val / "lr_probe.json").read_text()) == doc
    with pytest.raises(stub.Refused, match="fresh R5 model"):
        stub.probe_lr(["third"], "STUB", tmp_path / "r6.json", mode="R6", **kw)


class _NoEnv:
    def close(self) -> None:
        pass


def test_a_later_filter_measurement_keeps_the_choice(stub, tmp_path, monkeypatch):
    """Rule O4: measured once on 75-M-H-10, which decides; the 60 cm body's measurement comes later, is reported
    and decides nothing.  The later call must not take the choice out of the record, and a record without the
    choice gives a run no filter."""
    from designlab.stage2 import s2io

    mean = {("75-M-H-10", 10.0): 10.0, ("75-M-H-10", 20.0): 9.5, ("60-N-T-10", 10.0): 8.0, ("60-N-T-10", 20.0): 4.0}
    asked = []

    def episodes(body, set_name, filter_hz, sigma, episodes=None, k=5, env=None):  # (no env: the survival asked for)
        asked.append((body, filter_hz))
        return [dict(task=t, episode_seed=e, survived=False, env_steps=1, of=1, survival_s=mean[(body, filter_hz)],
                     fall_cause=None) for t in ("walk_nominal", "walk_v0.4") for e in range(7000, 7010)], _NoEnv()

    monkeypatch.setattr(stub, "_noise_episodes", episodes)
    val = tmp_path / "validation"
    a = stub.filter_choice(["75-M-H-10"], "STUB", val)
    assert a["choice"] == dict(body="75-M-H-10", ratio=0.95, keep=0.9, filter_hz=20.0) and a["added"] == ["75-M-H-10"]
    b = stub.filter_choice(["60-N-T-10"], "STUB", val)  # the 60 cm body, later, into the same record
    assert b["choice"] == a["choice"] and b["bodies"]["75-M-H-10"] == a["bodies"]["75-M-H-10"]
    assert b["bodies"]["60-N-T-10"]["role"] == "reported; decides nothing" and b["bodies"]["60-N-T-10"]["ratio"] == 0.5
    assert b["added"] == ["60-N-T-10"] and json.loads((val / "filter.json").read_text()) == b
    assert set(b["bodies"]["60-N-T-10"]["measured"]) == {"versions", "provenance", "at"}
    monkeypatch.setattr(stub, "VALIDATION", val)
    for body in ("75-M-H-10", "60-N-T-10", "60-N-T-10-X"):
        got = REAL["_validation"](body, "STUB")["filter"]
        assert got["filter_hz"] == 20.0 and got["sha256"] == s2io.sha256_file(val / "filter.json")
    # measured once: neither body again; another pair of cut-offs is another measurement
    keep, n = (val / "filter.json").read_bytes(), len(asked)
    for body in ("75-M-H-10", "60-N-T-10"):
        with pytest.raises(stub.Refused, match=f"already measured: {body}"):
            stub.filter_choice([body], "STUB", val)
    with pytest.raises(stub.Refused, match="other settings.*cutoffs"):
        stub.filter_choice(["60-N-T-10-X"], "STUB", val, cutoffs=(10.0, 30.0))
    assert (val / "filter.json").read_bytes() == keep and len(asked) == n
    # the other order: while the rule's body is not measured there is no choice, and a run is given no filter
    other = tmp_path / "other"
    c = stub.filter_choice(["60-N-T-10"], "STUB", other)
    assert c["choice"] is None
    monkeypatch.setattr(stub, "VALIDATION", other)
    assert REAL["_validation"]("60-N-T-10", "STUB")["filter"] is None
    d = stub.filter_choice(["75-M-H-10"], "STUB", other)
    assert d["choice"] == a["choice"] and list(d["bodies"]) == ["60-N-T-10", "75-M-H-10"]
    assert REAL["_validation"]("60-N-T-10", "STUB")["filter"]["filter_hz"] == 20.0


def test_a_run_that_is_not_tiny_needs_the_filter(stub, tmp_path, monkeypatch):
    """Rule O4 is measured before any run: without its record a run that is not tiny is refused, as it is without
    a rate, unless the caller names the filter; it does not start on the env's default without a word."""
    with pytest.raises(stub.Refused, match="no action filter.*rule O4") as info:
        run(stub, tmp_path / "a", tiny=False, lr=1e-4)
    assert info.value.code == stub.EXIT_REFUSED and not (tmp_path / "a").exists()
    with pytest.raises(stub.Refused, match="no learning rate"):  # (the rate is asked for first)
        run(stub, tmp_path / "a", tiny=False)
    d = run(stub, tmp_path / "b", tiny=False, lr=1e-4, config=dict(filter_hz=20.0))  # named by the caller
    assert d["status"] == "done" and {c["filter_hz"] for c in d["settings"]["env_configs"]} == {20.0}
    assert d["settings"]["validation"]["filter"] is None and d["settings"]["config"] == dict(filter_hz=20.0)
    assert "rule O4 is measured before any run" in d["rules"]["filter"]
    monkeypatch.setattr(stub, "_validation", lambda body, set_name: dict(
        lr_probe=None, filter=dict(path="p", sha256="0" * 64, filter_hz=20.0)))
    d = run(stub, tmp_path / "c", tiny=False, lr=1e-4)  # the record's choice
    assert d["status"] == "done" and {c["filter_hz"] for c in d["settings"]["env_configs"]} == {20.0}
    assert d["settings"]["validation"]["filter"]["filter_hz"] == 20.0 and d["settings"]["config"] == {}


def test_rule_k8s_probe_runs_on_the_fine_tunes_own_envs(stub, tmp_path, monkeypatch):
    """Rule K8's probe asks whether every minibatch step runs from the parent's checkpoint: on the envs the
    fine-tune will train on, its mode and its config fields, not on R5 envs with today's defaults."""
    assert run(stub, tmp_path / "p", mode="R6")["status"] == "done"
    parent = job(tmp_path / "p", "R6-stub-s42")
    calls = []
    monkeypatch.setattr(stub, "_env_configs", lambda mode, body, set_name, overrides: calls.append(
        ("configs", mode, dict(overrides or {}))) or stub_configs(mode, body, set_name, overrides))
    monkeypatch.setattr(stub, "_env_specs", lambda mode, cfgs: calls.append(
        ("specs", mode, sorted({(c["mode"], c["filter_hz"]) for c in cfgs}))) or stub_specs(mode, cfgs))
    d = run(stub, tmp_path / "ft", mode="R6", job="ft", parent=(parent, 1280), steps=256,
            config=dict(filter_hz=20.0))
    assert d["status"] == "done" and d["settings"]["lr"]["source"].startswith("rule K8")
    assert calls == [("configs", "R6", dict(filter_hz=20.0)),            # the job's configs
                     ("configs", "R6", dict(filter_hz=20.0)),            # the probe's
                     ("specs", "R6", [("R6", 20.0)]),                    # the probe's pool
                     ("specs", "R6", [("R6", 20.0)])]                    # the job's pool
    probe = json.loads((job(tmp_path / "ft", "ft") / d["parent"]["lr_probe"]["record"]).read_text())
    row = probe["bodies"]["stub"]
    assert probe["mode"] == "R6" and (row["mode"], row["config"], row["filter_hz"]) == ("R6", dict(filter_hz=20.0), 20.0)
    assert probe["parent"]["job"] == "R6-stub-s42" and probe["sha256"] == d["parent"]["lr_probe"]["sha256"]
    # the validation record's filter reaches the probe as it reaches the job, unless the caller names one
    calls.clear()
    monkeypatch.setattr(stub, "_validation", with_filter)
    stub.probe_lr(["stub"], "STUB", tmp_path / "k8.json", updates=1, workers=0, parent=(parent, 1280), mode="R6")
    assert calls == [("configs", "R6", dict(filter_hz=10.0)), ("specs", "R6", [("R6", 10.0)])]


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

    monkeypatch.setenv("PYTHONHASHSEED", "0")
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
