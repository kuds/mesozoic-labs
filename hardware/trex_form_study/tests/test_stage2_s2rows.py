"""Tests for the policy rows of the bar runner (``designlab/stage2/s2rows.py``; round C design section 3.11; Track
B rules E1, E2, E5, K4, K9; ``STAGE2_PLAN.md`` S2-12).

A helper (``make_job``) writes a job directory as the trainer leaves one: ``run.json`` with the settings, the env
configs and a session's recorded source hashes, and one checkpoint of a **freshly initialised** policy saved
through ``s2load``.  Nothing is trained.  The closed-loop tests run on the S1 view (Stage 1's own run directory) and
on a synthetic set ``T0`` in a temporary directory (rule X1: no Stage 2 reference set is touched); they need
torch, the Stage 1 arrays, the round-two models and one BLAS thread, and skip without them.  Slow
(``STAGE2_SLOW=1``): ``equivalence_check`` over whole episodes at 10 ms on ``75-M-H-10`` and, when its model is
restored, on ``75-M-H-10-X`` (also with the head script H1), and the R6 row against ``R6Env``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import panels, plant  # noqa: E402
from designlab.stage2 import bar, gates2, lookbar, r5lib, s2io, s2load, s2rows, setio  # noqa: E402

BODY, BODY_X = "75-M-H-10", "75-M-H-10-X"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)
SLOW = os.environ.get("STAGE2_SLOW") == "1"


def _have(body: str) -> bool:
    try:
        d = setio._anchored(r5lib.s1_root(body)) / body
        return (d / "transitions" / "stop.npz").exists() and all(Path(f).exists() for f in plant.model_files(body))
    except Exception:  # noqa: BLE001  (no model record: the same skip)
        return False


try:
    import stable_baselines3  # noqa: F401
    import torch  # noqa: F401

    TORCH = True
except Exception:  # noqa: BLE001
    TORCH = False
needs_run = pytest.mark.skipif(not (TORCH and _have(BODY) and ONE_THREAD), reason=(
    "needs torch and stable-baselines3, the Stage 1 arrays and the round-two models (tools/restore_inputs.py) and "
    "one BLAS thread (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)"))
slow = pytest.mark.skipif(not SLOW, reason="whole closed-loop episodes of a policy row and of the env (STAGE2_SLOW=1)")
SET = "T0"
W = setio.walk_name(setio.BAR_SPEED)


def quiet(*a, **k):
    pass


def plain(x):
    return json.loads(json.dumps(s2io.plain(x)))


# ------------------------------------------------------------------------------------------------
# A job directory as the trainer leaves one, with a freshly initialised policy
# ------------------------------------------------------------------------------------------------
def make_job(job_dir, body: str, set_name: str = "S1", mode: str = "R5", seed: int = 42, root=None, scale: float = 1.0,
             clock: float | None = None, steps: int = 0, servo: str | None = None) -> dict:
    """``run.json`` and checkpoint ``steps`` of a freshly initialised policy of the body's layout (nothing is
    trained).  ``scale`` multiplies the action layer's weights (1: actions of about 0.02; 20: about 0.4, which
    fells the robot); ``clock`` sets the clock action's bias (with ``scale`` 0 the policy then holds that clock
    action and moves no joint); ``servo`` turns the env configs' servo option on (``R5Config.servo``, the layer's
    joint factors fixed at 1).  The observation statistics are made non-trivial.  Returns ``run.json``."""
    import gymnasium as gym
    from designlab.stage1 import policy
    from designlab.stage2 import r5env, schedules, train2
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    job_dir = Path(job_dir)
    kw = {} if root is None else dict(root=root)
    cfgs = r5env.env_configs(mode, body, set_name, 8, **kw)
    if servo is not None:
        cfgs = [dataclasses.replace(c, servo=servo, joint_damping=(1.0, 1.0), joint_friction=(1.0, 1.0)) for c in cfgs]
    e = (r5env.R6Env if mode == "R6" else r5env.R5Env)(cfgs[0], **kw)
    desc, layout = e.describe(), e.layout
    e.close()

    class Spaces(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (layout["obs_dim"],), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (layout["action_dim"],), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(layout["obs_dim"], np.float32), {}

        def step(self, action):
            return np.zeros(layout["obs_dim"], np.float32), 0.0, False, False, {}

    torch.set_num_threads(1)
    venv = VecNormalize(DummyVecEnv([Spaces]), norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0, gamma=0.99)
    k = policy.model_kwargs(int(layout["actor_dim"]), log_std_init=math.log(0.1))
    k.update(learning_rate=schedules.LinearSchedule(1e-4, 3e-5), n_steps=16, batch_size=16, clip_range=0.2)
    model = PPO(env=venv, seed=seed, device="cpu", verbose=0, **k)
    venv.obs_rms.update(0.5 + 0.3 * np.random.default_rng(7).standard_normal((64, layout["obs_dim"])))
    with torch.no_grad():
        model.policy.action_net.weight.mul_(float(scale))
        if clock is not None:
            model.policy.action_net.bias[-1] = float(clock)
    settings = s2io.plain(dict(mode=mode, body=body, seed=seed, set=set_name, set_freeze=desc["set_freeze_started"],
                               inputs=dict(files=desc["task_files"], plant_inputs=desc["inputs"]["plant_inputs"]),
                               tasks=desc["tasks"], task_names=desc["task_names"],
                               env_configs=[c.describe() for c in cfgs], layout=layout))
    src = train2.sources()
    run = dict(schema="s2-run-1", run_uuid=uuid.uuid4().hex, run_id="test", job=job_dir.name, mode=mode, body=body,
               seed=seed, set=set_name, settings=settings, settings_sha256=s2io.sha256_json(settings),
               sessions=[dict(index=0, versions=s2load.versions(),
                              sources=dict(sha256=src["sha256"], files=src["files"],
                                           plant_sources=src["plant_sources"], panel_sources=src["panel_sources"]))],
               flags=dict(kl_cut=False), status="running")
    s2io.write_json(job_dir / "run.json", run)
    s2load.save_checkpoint(job_dir, steps, model, venv, layout,
                           dict(run_uuid=run["run_uuid"], run_id="test", job=job_dir.name,
                                settings_sha256=run["settings_sha256"]), {})
    return run


@pytest.fixture(scope="module")
def job(tmp_path_factory) -> Path:
    """A job of mode R5 on the S1 view of the 75 cm body: one checkpoint (step 0) of a freshly initialised policy."""
    d = tmp_path_factory.mktemp("jobs") / "R5-75-M-H-10-s42"
    make_job(d, BODY)
    return d


@pytest.fixture(autouse=True)
def _table(monkeypatch):
    """Every test starts and ends with an empty table of policies and without the two environment variables."""
    monkeypatch.delenv(s2rows.POLICIES_ENV, raising=False)
    monkeypatch.delenv(s2rows.SOURCE_CHANGE_ENV, raising=False)
    s2rows.configure(None)
    yield
    s2rows.configure(None)


def copy_job(job: Path, to: Path) -> Path:
    return Path(shutil.copytree(job, to))


def reseal(job: Path, steps: int = 0, **changes) -> dict:
    """Checkpoint ``steps``' record with ``changes`` and its own sha256 again."""
    path = s2load.checkpoint_files(job, steps)["record"]
    doc = s2io.read_json(path)
    doc.pop("sha256")
    doc.update(changes)
    doc["sha256"] = s2io.sha256_json(doc)
    s2io.write_json(path, doc)
    return doc


def edit_run(job: Path, fn) -> dict:
    doc = s2io.read_json(job / "run.json")
    fn(doc)
    s2io.write_json(job / "run.json", doc)
    return doc


def stock_cut(monkeypatch, seconds: float = 1.0, keys=("walk/brisk",)):
    conds = [dataclasses.replace(c, seconds=float(seconds)) for c in panels.conditions() if c.key in keys]
    monkeypatch.setattr(bar, "stock_conditions", lambda: conds)
    monkeypatch.setattr(bar, "stock_default", lambda: conds)
    return conds


# ------------------------------------------------------------------------------------------------
# The module and the table of policies
# ------------------------------------------------------------------------------------------------
def test_importing_s2rows_imports_neither_the_env_nor_the_trainer_nor_torch():
    code = ("import sys\nfrom designlab.stage2 import s2rows\nfrom designlab.stage1 import panels\n"
            "bad = [m for m in ('designlab.stage2.r5env', 'designlab.stage2.train2', 'torch', 'stable_baselines3') "
            "if m in sys.modules]\n"
            "print(bad, sorted(c for c in panels.ROW_FACTORIES if c in ('R0L', 'R5', 'R5G', 'R6')), "
            "__import__('os').environ['MUJOCO_GL'])\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env=dict(os.environ, PYTHONPATH="."))
    assert out.stdout.strip() == "[] ['R0L', 'R5', 'R5G', 'R6'] disable"
    assert s2rows.CONTROLLERS == ("R5", "R6", "R5G") and s2rows.S2Policy.every == 10
    assert all(panels.ROW_FACTORIES[c] is s2rows._factory for c in s2rows.CONTROLLERS)


def test_the_table_of_policies(tmp_path, monkeypatch):
    assert s2rows.policies() == {}
    with pytest.raises(s2rows.RowError, match="no checkpoint is configured"):
        s2rows.policy_entry("R5/42")
    s2rows.configure({"R5/42": dict(job_dir=tmp_path / "a", steps=500000.0), "R6/42": dict(job_dir="b", steps=0, root="r")})
    assert s2rows.policies() == {"R5/42": dict(job_dir=str(tmp_path / "a"), steps=500000),
                                 "R6/42": dict(job_dir="b", steps=0, root="r")}
    assert s2rows.policy_entry("R5G/42") == s2rows.policy_entry("R5/42")  # the head-script row of the same checkpoint
    with pytest.raises(s2rows.RowError, match="R5/43"):
        s2rows.policy_entry("R5/43")
    with pytest.raises(s2rows.RowError, match="job_dir"):
        s2rows.configure({"R5/42": dict(steps=1)})
    # the JSON file the environment names is read when nothing is configured
    s2rows.configure({})
    path = tmp_path / "policies.json"
    s2io.write_json(path, {"R5/7": dict(job_dir="c", steps=250000), "R5G/7": dict(job_dir="g", steps=1)})
    monkeypatch.setenv(s2rows.POLICIES_ENV, str(path))
    assert s2rows.policy_entry("R5/7") == dict(job_dir="c", steps=250000)
    assert s2rows.policy_entry("R5G/7") == dict(job_dir="g", steps=1)  # its own entry wins
    s2io.write_json(path, [1, 2])
    with pytest.raises(s2rows.RowError, match="JSON object"):
        s2rows.policies()
    with pytest.raises(s2rows.RowError, match="<seed>"):
        s2rows.build("R5", BODY, tmp_path, 0)
    with pytest.raises(s2rows.RowError, match="no run.json"):
        s2rows.build("R5/42", BODY, tmp_path, 0)


# ------------------------------------------------------------------------------------------------
# Building a row
# ------------------------------------------------------------------------------------------------
KEYS = ["action", "body", "checkpoint", "controller", "cross_version", "deterministic", "head", "layout", "mode",
        "ppo_seed", "row", "run", "source_change", "versions_judged", "versions_trained"]


def floats_of(x, path="") -> list:
    if isinstance(x, dict):
        return [f for k, v in x.items() for f in floats_of(v, f"{path}.{k}")]
    if isinstance(x, list):
        return [f for i, v in enumerate(x) for f in floats_of(v, f"{path}[{i}]")]
    return [path] if isinstance(x, float) else []


@needs_run
def test_a_row_describes_what_an_episode_executes_and_no_measured_value(job):
    s2rows.configure({"R5/42": dict(job_dir=job, steps=0)})
    look = panels.tracker_rows(BODY, RUN / BODY, ("R5/42",))["R5/42"]
    assert isinstance(look, lookbar.LookRow) and isinstance(look.inner, s2rows.R5Row) and look.name == "R5/42"
    row = look.inner
    assert isinstance(row, panels.TrackerRow) and (row.mode, row.gains, row.terms, row.ankle_cap) == ("R2", None, (), ())
    assert row.rate == panels.RATE and isinstance(row.policy, s2rows.S2Policy) and row.hard_only is False
    d = row.describe()
    run, record = s2io.read_json(job / "run.json"), s2io.read_json(s2load.checkpoint_files(job, 0)["record"])
    assert sorted(d) == KEYS and d == plain(d)
    assert (d["row"], d["mode"], d["body"], d["ppo_seed"], d["head"], d["deterministic"]) == ("R5/42", "R5", BODY, 42, "neutral", True)
    assert d["controller"].startswith("R5: the tracker on the estimate with no feedback term")
    assert d["checkpoint"] == dict(steps=0, files={k: dict(name=v["name"], sha256=v["sha256"]) for k, v in record["files"].items()})
    assert d["run"] == dict(run_uuid=run["run_uuid"], job=job.name, set="S1", set_freeze=None,
                            settings_sha256=run["settings_sha256"])
    assert d["layout"] == record["layout"] and (d["layout"]["actor_dim"], d["layout"]["action_dim"]) == (88, 15)
    assert d["action"] == dict(scales=dict(leg_rad=0.35, other_rad=0.15, clock=0.25), rate=1.05, filter_hz=10.0)
    # both version sets, torch's build label dropped (S2-12): they are executed, and are in the record's key
    assert d["versions_trained"] == s2load.comparable(record["versions"]) == d["versions_judged"]
    assert sorted(d["versions_judged"]) == sorted(s2load.VERSION_KEYS) and "+" not in d["versions_judged"]["torch"]
    assert d["cross_version"] is False and d["source_change"] is False
    # no measured value: the only floats are the action's settings
    assert sorted(floats_of(d)) == [".action.filter_hz", ".action.rate", ".action.scales.clock", ".action.scales.leg_rad",
                                    ".action.scales.other_rad"]
    assert "fingerprint_max_abs_diff" not in json.dumps(d) and "tol" not in d
    # the measured fingerprint difference and its tolerance are in the provenance (the episode's block)
    p = row.provenance()
    assert p["fingerprint_max_abs_diff"] == 0.0 and p["fingerprint_tol"] == 1e-5 and p["cross_version"] is False
    assert p["versions_trained"] == record["versions"] and p["versions_judged"] == s2load.versions()
    assert p["checkpoint"]["record_sha256"] == record["sha256"] and p["source_change"] is False and p["source_hashes"] is None
    assert p["run"] == d["run"] and (p["set_freeze_now"], p["later_freeze"], p["mode"], p["head"]) == (None, False, "R5", "neutral")
    assert look.provenance() == p
    # the sources: what a policy row's episodes run, by path
    assert sorted(row.sources()) == ["designlab/stage1/env.py", "designlab/stage1/policy.py"] + [
        f"designlab/stage2/{m}.py" for m in ("lookbar", "lookread", "r5core", "r5lib", "s2load", "s2rows")]
    assert row.sources()["designlab/stage2/r5core.py"] == s2io.sha256_file(ROOT / "designlab/stage2/r5core.py")
    # the bar's rows wrap it like any registered row; the wrapper's description is the row's with the look reading
    delayed = lookbar.rows(BODY, ["R5/42@10ms"])["R5/42@10ms"]
    assert bar.controller_of("R5/42@10ms") == "R5" and delayed.inner.inner.describe() == d
    assert {k: v for k, v in delayed.inner.describe().items() if k != "look_reading"} == d
    assert sorted(set(delayed.sources()) - set(row.sources())) == ["designlab/stage1/speedfit.py", "designlab/stage2/bar.py"]
    assert bar.rows(BODY, ["R5/42@6ms"])["R5/42@6ms"].inner.describe() == delayed.inner.describe()


@needs_run
def test_a_spec_rebuilds_the_row_and_refuses_a_changed_file(job, tmp_path):
    mine = copy_job(job, tmp_path / job.name)
    s2rows.configure({"R5/42": dict(job_dir=mine, steps=0)})
    delayed = lookbar.rows(BODY, ["R5/42@10ms"], sidecar=tmp_path / "side")["R5/42@10ms"]
    spec = plain(delayed.spec())
    assert spec["inner"]["inner"]["cls"] == "designlab.stage2.s2rows.R5Row" and spec["inner"]["inner"]["ck"]["job_dir"] == str(mine)
    again = panels.row_from_spec(spec)
    assert type(again.inner.inner) is s2rows.R5Row and again.describe() == delayed.describe()
    assert again.sources() == delayed.sources() and again.inner.sidecar_row == "R5/42@10ms"
    assert again.inner.inner.policy._loaded is None  # a worker loads on first use
    assert again.inner.inner.provenance()["fingerprint_max_abs_diff"] == 0.0
    # a file that changed since the row was built is refused when a worker rebuilds the row
    model = s2load.checkpoint_files(mine, 0)["model"]
    model.write_bytes(model.read_bytes() + b"\0")
    with pytest.raises(s2rows.RowError, match="not the file the row"):
        panels.row_from_spec(spec)


@needs_run
def test_a_record_that_names_other_versions_gives_cross_version(job, tmp_path):
    mine = copy_job(job, tmp_path / job.name)
    record = s2io.read_json(s2load.checkpoint_files(mine, 0)["record"])
    reseal(mine, versions=dict(record["versions"], torch="2.99.0+cu126", numpy="9.9.9"))
    row = s2rows.policy_row("R5/42", BODY, mine, 0)
    d = row.describe()
    assert d["cross_version"] is True and d["versions_trained"]["torch"] == "2.99.0" and d["versions_trained"]["numpy"] == "9.9.9"
    assert d["versions_judged"] == s2load.comparable(s2load.versions()) and d["versions_judged"] != d["versions_trained"]
    p = row.provenance()
    assert p["cross_version"] is True and p["versions_trained"]["torch"] == "2.99.0+cu126" and p["fingerprint_max_abs_diff"] == 0.0
    # torch's build label alone is no other version (S2-12)
    reseal(mine, versions=dict(record["versions"], torch=record["versions"]["torch"].split("+")[0] + "+cu126"))
    assert s2rows.policy_row("R5/42", BODY, mine, 0).describe()["cross_version"] is False


@needs_run
def test_a_foreign_run_a_perturbed_fingerprint_and_other_settings_are_refused(job, tmp_path, monkeypatch, capsys):
    # another run's checkpoint in this run's directory
    a = copy_job(job, tmp_path / "a" / job.name)
    reseal(a, run_uuid="0" * 32)
    with pytest.raises(s2load.CheckpointError, match="another run's checkpoint is not used and not judged"):
        s2rows.policy_row("R5/42", BODY, a, 0)
    # a fingerprint that is not reproduced within 1e-5
    b = copy_job(job, tmp_path / "b" / job.name)
    files = s2load.checkpoint_files(b, 0)
    with np.load(files["fingerprint"]) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["actions"] = arrays["actions"] + np.float32(1e-3)
    sha = s2io.write_npz(files["fingerprint"], arrays)
    rec = s2io.read_json(files["record"])
    rec["files"]["fingerprint"].update(sha256=sha, bytes=files["fingerprint"].stat().st_size)
    reseal(b, files=rec["files"])
    with pytest.raises(s2load.CheckpointError, match="fingerprint not reproduced") as exc:
        s2rows.policy_row("R5/42", BODY, b, 0)
    assert exc.value.fingerprint_max_abs_diff == pytest.approx(1e-3, rel=1e-3)
    # a run.json whose settings were changed, another mode, body or seed, a clock that is not the bar's
    c = copy_job(job, tmp_path / "c" / job.name)
    edit_run(c, lambda d: d["settings"].update(seed=43))
    with pytest.raises(s2rows.RowError, match="do not hash"):
        s2rows.policy_row("R5/42", BODY, c, 0)
    for name, body, match in (("R6/42", BODY, "the run's mode is 'R5'"), ("R5/43", BODY, "the run's seed is 42"),
                              ("R5/42", "60-N-T-10", "the run's body is '75-M-H-10'")):
        with pytest.raises(s2rows.RowError, match=match):
            s2rows.policy_row(name, body, job, 0)
    with pytest.raises(s2rows.RowError, match="H1 needs a body with a neck yaw and a jaw"):
        s2rows.policy_row("R5G/42", BODY, job, 0)
    with pytest.raises(s2load.CheckpointError, match="missing record"):
        s2rows.policy_row("R5/42", BODY, job, 250000)
    # at the command line each is a refusal (exit code 1), before any episode
    monkeypatch.setenv(s2rows.POLICIES_ENV, "")  # (the command sets it: restored after the test)
    for i, (d, steps) in enumerate(((a, 0), (c, 0), (job, 250000))):
        table = tmp_path / f"policies{i}.json"
        s2io.write_json(table, {"R5/42": dict(job_dir=str(d), steps=steps)})
        out = tmp_path / f"out{i}"
        assert lookbar.main(["run", "--set", "S1", "--body", BODY, "--rows", "R5/42@10ms", "--policies", str(table),
                             "--out", str(out)]) == 1
        assert not out.exists()
    assert "not used and not judged" in capsys.readouterr().err


@needs_run
def test_a_changed_source_is_refused_or_flagged(job, tmp_path, monkeypatch):
    today = s2rows.source_hashes()
    assert sorted(today["files"]) == [f"designlab/stage2/{m}.py" for m in ("lookread", "r5core", "r5lib")]
    assert today["stage1"] == plant.sources_sha256() and today["panels"] == panels.panel_sources()["sha256"]
    run = s2io.read_json(job / "run.json")
    assert s2rows.source_check(run)["changed"] == [] and s2rows.source_check(dict(run, sessions=[]))["changed"]
    for i, (what, change) in enumerate((
            ("designlab/stage2/r5core.py", lambda s: s["files"].update({"designlab/stage2/r5core.py": "0" * 64})),
            ("designlab/stage2/lookread.py", lambda s: s["files"].update({"designlab/stage2/lookread.py": "1" * 64})),
            ("the Stage 1 sources", lambda s: s.update(plant_sources="2" * 64)),
            ("the panels' sources", lambda s: s.update(panel_sources="3" * 64)))):
        mine = copy_job(job, tmp_path / str(i) / job.name)
        edit_run(mine, lambda d: change(d["sessions"][0]["sources"]))  # noqa: B023
        monkeypatch.delenv(s2rows.SOURCE_CHANGE_ENV, raising=False)
        with pytest.raises(s2rows.RowError, match="not judged") as exc:
            s2rows.policy_row("R5/42", BODY, mine, 0)
        assert what in str(exc.value) and "session 0" in str(exc.value)
        # under the environment variable it runs, flagged, with both hash sets in its description
        monkeypatch.setenv(s2rows.SOURCE_CHANGE_ENV, "1")
        row = s2rows.policy_row("R5/42", BODY, mine, 0)
        d = row.describe()
        assert d["source_change"] is True and sorted(d) == sorted(KEYS + ["source_hashes"])
        assert d["source_hashes"]["judged"] == plain(today) and len(d["source_hashes"]["changed"]) == 1
        assert d["source_hashes"]["trained"][0] != dict(session=0, **plain(today))
        assert row.provenance()["source_change"] is True and row.provenance()["source_hashes"] == d["source_hashes"]
    # an unchanged run is not flagged by the variable
    assert s2rows.policy_row("R5/42", BODY, job, 0).describe()["source_change"] is False
    # a second session on other sources is a run on two sets of sources: every session is compared
    mine = copy_job(job, tmp_path / "two" / job.name)
    edit_run(mine, lambda d: d["sessions"].append(dict(copy.deepcopy(d["sessions"][0]), index=1,
                                                      sources=dict(d["sessions"][0]["sources"], panel_sources="4" * 64))))
    monkeypatch.delenv(s2rows.SOURCE_CHANGE_ENV)
    with pytest.raises(s2rows.RowError, match="session 1: the panels' sources"):
        s2rows.policy_row("R5/42", BODY, mine, 0)


@needs_run
def test_the_runs_set_as_it_is_today_rule_k4(job, tmp_path):
    run = s2io.read_json(job / "run.json")
    assert s2rows.set_check(run, BODY) == dict(started=None, now=None, later=False)
    files = run["settings"]["inputs"]["files"]
    assert "refs/nominal.npz" in files and all(sorted(v) == ["entry_sha256", "sha256"] for v in files.values())

    def changed(fn, match):
        doc = copy.deepcopy(run)
        fn(doc["settings"])
        with pytest.raises(s2rows.RowError, match=match):
            s2rows.set_check(doc, BODY)

    changed(lambda s: s["inputs"]["files"]["refs/brisk.npz"].update(sha256="0" * 64), "refs/brisk.npz: sha256")
    changed(lambda s: s["inputs"]["files"]["refs/brisk.npz"].update(entry_sha256="0" * 64), "refs/brisk.npz: entry_sha256")
    changed(lambda s: s["inputs"]["files"].update({"refs/v0.4.npz": dict(sha256="0", entry_sha256="0")}), "not in the set")
    changed(lambda s: s["task_names"].append("turn_q90_left"), "task turn_q90_left was in the run's table")
    changed(lambda s: s["inputs"].update(plant_inputs=dict(mjcf="0" * 64)), "the plant's inputs")
    changed(lambda s: s["inputs"].update(files={}), "records no input file")
    # a run.json that holds no task names or no plant inputs has nothing to compare: refused, not passed
    changed(lambda s: s.pop("task_names"), "names no task")
    changed(lambda s: s.update(task_names=[]), "names no task")
    changed(lambda s: s.update(task_names="walk_nominal"), "names no task")
    changed(lambda s: s["inputs"].pop("plant_inputs"), "records no plant inputs")
    changed(lambda s: s["inputs"].update(plant_inputs={}), "records no plant inputs")
    # the settings are hashed, so a run.json edited this way is refused before the set is read at all
    mine = copy_job(job, tmp_path / job.name)
    edit_run(mine, lambda d: d["settings"]["inputs"]["files"]["refs/brisk.npz"].update(sha256="0" * 64))
    with pytest.raises(s2rows.RowError, match="do not hash"):
        s2rows.policy_row("R5/42", BODY, mine, 0)
    # ... and with the hash made again, by rule K4
    edit_run(mine, lambda d: d.update(settings_sha256=s2io.sha256_json(d["settings"])))
    reseal(mine, settings_sha256=s2io.read_json(mine / "run.json")["settings_sha256"])
    with pytest.raises(s2rows.RowError, match="rule K4"):
        s2rows.policy_row("R5/42", BODY, mine, 0)


# ------------------------------------------------------------------------------------------------
# Running a row
# ------------------------------------------------------------------------------------------------
@needs_run
def test_building_and_running_a_row_imports_neither_r5env_nor_train2(job):
    code = (
        "import dataclasses, json, sys\n"
        "from designlab.stage1 import panels\n"
        "from designlab.stage2 import bar, lookbar, s2rows\n"
        "conds = [dataclasses.replace(c, seconds=1.0) for c in panels.conditions() if c.key == 'walk/brisk']\n"
        "bar.stock_conditions = lambda: conds\n"
        f"s2rows.configure({{'R5/42': dict(job_dir={str(job)!r}, steps=0)}})\n"
        f"rows = lookbar.rows({BODY!r}, ['R5/42@10ms'])\n"
        f"e = lookbar.episodes({BODY!r}, 'S1', rows, [('walk/brisk', 3042)])[0]\n"
        "bad = [m for m in ('designlab.stage2.r5env', 'designlab.stage2.train2', 'designlab.stage2.schedules') "
        "if m in sys.modules]\n"
        "print(json.dumps(dict(bad=bad, torch='torch' in sys.modules, outcome=e['outcome'], steps=e['steps'], "
        "diff=e['policy_check']['fingerprint_max_abs_diff'], factor=e['clock']['mean_factor'], error=e['error'])))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH="."))
    assert out.returncode == 0, out.stderr[-2000:]
    doc = json.loads(out.stdout.strip().splitlines()[-1])
    # (the loader's schedule objects are instances of ``schedules``' classes: that module is the loader's, not a row's)
    assert doc["bad"] in ([], ["designlab.stage2.schedules"]) and doc["torch"] is True
    assert (doc["outcome"], doc["steps"], doc["diff"], doc["error"]) == ("read", 500, 0.0, None)
    assert doc["factor"] != 1.0 and abs(doc["factor"] - 1.0) < 0.01  # the policy moves the clock a little


@needs_run
@pytest.mark.parametrize("k,seconds", [(5, 1.013), (0, 1.0)])
def test_a_row_episode_is_the_envs_on_the_s1_view(job, k, seconds):
    """A cut episode of the row as the bar runs it against the training env with the same policy: digest for
    digest (the slow tests run whole episodes)."""
    r = s2rows.equivalence_check(job, 0, BODY, "S1", "walk/brisk", 3042, k, seconds=seconds)
    n = round(seconds / 0.002)
    assert r["identical"] and r["ends_alike"] and r["first_difference"] is None
    assert r["reference_steps"] == r["compared_steps"] == n and r["env_steps"] == -(-n // 10)
    assert (r["row"], r["set"], r["condition"], r["seed"], r["k"]) == (f"R5/42@{2 * k}ms", "S1", "walk/brisk", 3042, k)
    assert 0.005 < r["action_abs_max"] < 0.1 and r["clock_factor_min"] < 1.0 < r["clock_factor_max"]  # it acts
    assert r["fingerprint_max_abs_diff"] == 0.0 and r["cross_version"] is False
    assert r["look"]["policy_check"]["fingerprint_tol"] == 1e-5 and r["look"]["steps"] == n
    # the env side of the comparison is the policy's: the bar's R0 row is another episode
    from designlab.stage2 import g8a

    opened = lookbar.open_set(BODY, "S1", ["walk/brisk"])
    cond = dataclasses.replace(opened["conds"]["walk/brisk"], seconds=seconds)
    r0 = g8a.row_episode(bar.rows(BODY, [f"R0@{2 * k}ms"])[f"R0@{2 * k}ms"], opened["ctx"], cond, 3042)
    pol = g8a.row_episode(bar.Delayed(lookbar.LookRow(s2rows.policy_row("R5/42", BODY, job, 0)), k), opened["ctx"], cond, 3042)
    assert len(r0["digests"]) == len(pol["digests"]) == n and r0["digests"] != pol["digests"]
    # (the first action reaches the servos k physics steps after the start: until then the two are one episode)
    first = next(i for i in range(n) if r0["digests"][i] != pol["digests"][i])
    assert first == k


@needs_run
def test_a_row_runs_conditions_of_its_runs_set_only(job, tmp_path):
    make_t0(tmp_path)
    s2rows.configure({"R5/42": dict(job_dir=job, steps=0)})
    rows = lookbar.rows(BODY, ["R5/42@10ms"])
    e = lookbar.episodes(BODY, SET, rows, [("walk/nominal", 6001)], root=tmp_path)[0]
    assert e["outcome"] == "error" and e["clock"] is None and not e["succeeded"]
    assert "the condition walk/nominal is of set T0, the policy's run of S1" in e["error"] and "RowError" in e["error"]


@needs_run
def test_run_through_the_bars_runner_on_the_s1_view(job, tmp_path, monkeypatch):
    stock_cut(monkeypatch, 1.0)
    s2rows.configure({"R5/42": dict(job_dir=job, steps=0)})
    out = tmp_path / "out"
    doc = lookbar.run("S1", BODY, out, ["R5/42@10ms", "R0L@10ms"], ["walk/brisk"], episodes=2, procs=1, stage1_run=RUN,
                      log=quiet)
    assert [c["row"] for c in doc["records"]] == ["R0L@10ms", "R5/42@10ms"]
    bdir = bar.block_dir(out, "S1", BODY, bar.HEADLINE_BLOCK)
    rec = s2io.read_json(bdir / "eval" / "R5" / "42@10ms" / "walk" / "brisk.json")
    comp = s2io.read_json(bdir / "look" / "R5" / "42@10ms" / "walk" / "brisk.json")
    row = lookbar.rows(BODY, ["R5/42@10ms"])["R5/42@10ms"]
    # the record's controller is the row's description: both version sets are in the record and in its key
    assert rec["controller"] == plain(s2io.portable(row.describe())) and rec["key"]["row"] == rec["controller"]
    assert rec["controller"]["versions_trained"] == rec["controller"]["versions_judged"] and rec["counts_look"] is None
    assert "designlab/stage2/s2rows.py" in rec["key"]["row_sources"] and "designlab/stage1/policy.py" in rec["key"]["row_sources"]
    # the companion holds the measured side: the fingerprint difference, its tolerance and the versions as recorded
    pol = comp["policy"]
    assert (pol["fingerprint_max_abs_diff"], pol["fingerprint_tol"], pol["cross_version"], pol["source_change"]) == (0.0, 1e-5, False, False)
    assert pol["versions_judged"] == s2load.versions() and pol["episodes_checked"] == 2 and pol["same_checkpoint"] is True
    assert pol["checkpoint"]["files"]["model"]["sha256"] == rec["controller"]["checkpoint"]["files"]["model"]["sha256"]
    assert [e["outcome"] for e in comp["episodes"]] == ["read", "read"] and comp["counts"] == rec["counts"]
    for e, b in zip(comp["episodes"], rec["episodes"], strict=True):
        side = s2io.read_json(lookbar.sidecar_path(bdir / "look_partial", "R5/42@10ms", "walk/brisk", e["seed"]))
        assert side["policy_check"]["fingerprint_max_abs_diff"] == 0.0 and side["survival_s"] == b["survival_s"]
        assert e["clock"]["mean_factor"] != 1.0 and e["clock"]["t_ref_reached"] != pytest.approx(1.05, abs=1e-6)
    # the open-loop row beside it has no policy block, and ran another episode
    r0l = s2io.read_json(bdir / "look" / "R0L@10ms" / "walk" / "brisk.json")
    assert r0l["policy"] is None and r0l["episodes"][0]["state_digest"] != comp["episodes"][0]["state_digest"]


@needs_run
def test_the_monitor_runs_a_checkpoints_row(job, tmp_path):
    """``monitor2.run`` on the helper's job (the S1 view, tiny): the row is built from the checkpoint's files, and
    every episode comes back with the clock times its credited survival needs."""
    m2 = pytest.importorskip("designlab.stage2.monitor2")
    doc = m2.run(job, 0, BODY, "S1", tmp_path / "monitor", tiny=True)
    assert doc["row"] == "R5/42@10ms" and doc["episodes_by"] == "lookbar.episodes" and len(doc["episodes"]) == 2
    assert m2.is_record(doc, doc["checkpoint_sha256"], m2.pairs_for("S1", tiny=True))
    for e in doc["episodes"]:
        assert e["outcome"] in ("read", "fell") and m2.has_clock(e) and e["planned_s"] > 10
        assert e["policy_check"]["fingerprint_max_abs_diff"] == 0.0 and e["row"] == "R5/42@10ms"
        assert 0.0 < e["survival_credit"] <= e["survival_s"] + 1e-12
        assert e["clock"]["mean_factor"] != 1.0
    assert doc["mean_clock_factor"] == pytest.approx(np.mean([e["clock"]["mean_factor"] for e in doc["episodes"]]))
    assert (tmp_path / "monitor" / "traces" / "0" / "R5" / "42@10ms" / "walk" / "brisk" / "6004.npz").is_file()


# ------------------------------------------------------------------------------------------------
# A bar item that a slowed clock leaves unread (a synthetic set in a temporary directory)
# ------------------------------------------------------------------------------------------------
T0_REFS = (("nominal", "nominal"), (W, "brisk"))
T0_MOVES = (("start", "start"), (setio.change_name("nominal", W), "nominal_to_brisk"),
            (setio.stop_name(2, W), "stop"), (setio.turn_name("pi4", "left"), "enter_walk_0.35_left"))


def make_t0(root: Path, refs=T0_REFS) -> Path:
    """``T0``: Stage 1's arrays of the 75 cm body under the bar's names, frozen (as the bar's own tests build it).
    ``refs``: which of Stage 1's walks stands under each of the set's walk names."""
    src = RUN / BODY
    t0_refs = tuple(refs)
    d = setio.set_dir(SET, BODY, root)
    refs1, trs1 = s2io.read_json(src / "refs.json"), s2io.read_json(src / "transitions.json")
    refs, trs, manifest = {}, {}, {}
    for name, old in t0_refs:
        e = copy.deepcopy(refs1["references"][old])
        e.update(name=name, file=f"refs/{name}.npz", set=SET, label=None, strides=1, window_s=e["period_s"],
                 crouch_frac=0.03, margin_rule="a fixture's")
        refs[name] = e
    for name, old in T0_MOVES:
        e = copy.deepcopy(trs1["transitions"][old])
        e.update(name=name, file=f"transitions/{name}.npz", a1_over=None)
        p = setio.parse_name(name)
        if p["kind"] == "stop":
            e["move"] = dict(kind="stop", builder="fixture", causal=True, command_phase=-0.5)
        elif p["kind"] == "turn":
            e["move"] = dict(kind="turn", builder="fixture", causal=True, command_phase=0.0)
            e["turn"] = dict(angle_rad=p["angle_rad"], rate_rad_s=p["sign"] * p["rate_rad_s"], direction=p["direction"])
        else:
            e["move"] = dict(kind=p["kind"], builder="fixture", causal=False, command_phase=None)
        trs[name] = e
    for doc, sub, kind in ((refs, "refs", "walk"), (trs, "transitions", "move")):
        for name, e in doc.items():
            (d / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / sub / f"{dict(t0_refs + T0_MOVES)[name]}.npz", d / e["file"])
            manifest[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256(kind, e))
    s2io.write_json(d / "refs.json", dict(body=BODY, set=SET, schema=1, references=refs))
    s2io.write_json(d / "transitions.json", dict(body=BODY, set=SET, schema=1, transitions=trs))
    freeze = dict(index=0, at="fixture", commit=None, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest))
    s2io.write_json(d / "set.json", dict(set=SET, body=BODY, role="fixture", crouch_frac=0.03, not_available={},
                                         stand_in_walks=True, rules=dict(fixture="Stage 1's arrays"),
                                         freezes=[freeze]))
    return d


@needs_run
def test_a_slowed_clock_leaves_a_bar_item_unread_and_a_later_freeze_is_accepted_by_k4(tmp_path, monkeypatch):
    """A policy that moves no joint and holds its clock action at -0.4 (a clock 10 % slow) on the fixture set: the
    open loop walks on, the bar cannot read the item, and the episode is *unread* with the reason ``clock_behind``
    (rules U6, M1), on a look set and elsewhere."""
    m2 = pytest.importorskip("designlab.stage2.monitor2")
    sets = tmp_path / "sets"
    d = make_t0(sets)
    job = tmp_path / "jobs" / "R5-slow-s42"
    run = make_job(job, BODY, SET, root=sets, scale=0.0, clock=-0.4)
    fz = run["settings"]["set_freeze"]
    assert fz["index"] == 0 and run["settings"]["task_names"] and "refs/v0.4.npz" in run["settings"]["inputs"]["files"]
    s2rows.configure({"R5/42": dict(job_dir=job, steps=0, root=sets)})
    rows = lookbar.rows(BODY, ["R5/42@10ms"])
    inner = rows["R5/42@10ms"].inner.inner
    assert inner.describe()["run"]["set_freeze"] == fz and inner.provenance()["later_freeze"] is False
    pairs = [("walk/v0.4", 6004), ("walk/nominal", 6001), ("turn/q90_left", 6007)]
    monkeypatch.setattr(gates2, "GATES", tmp_path / "gates")
    for look_sets in (lookbar.LOOK_SETS, ("V6", SET)):
        monkeypatch.setattr(lookbar, "LOOK_SETS", look_sets)
        monkeypatch.setattr(gates2, "LOOK_SETS", look_sets)  # (lookbar calls the gates' own check of rule X1)
        if SET in look_sets:  # a look set: the row runs there only once its G9-ref has passed (lookbar.RULES["look_sets"])
            with pytest.raises(lookbar.LookSetRefused, match=f"{SET}/{BODY}: there is no G9-ref record"):
                lookbar.episodes(BODY, SET, rows, pairs, root=sets)
            gates2.finish(dict(schema=gates2.SCHEMAS["g9_ref"], set=SET, body=BODY, pass_=True, at="fixture",
                               set_freeze=setio.frozen(s2io.read_json(d / "set.json"))),
                          gates2.gate_path(SET, BODY, "g9_ref"))
        eps = lookbar.episodes(BODY, SET, rows, pairs, root=sets)
        walk, nominal, turn = eps
        assert (walk["outcome"], walk["reason"], walk["fell"], walk["error"]) == ("unread", "clock_behind", False, None)
        assert "not all in the walk's steady part" in walk["unreadable"] and walk["survival_s"] == pytest.approx(walk["planned_s"], abs=0.01)
        assert (turn["outcome"], turn["reason"]) == ("unread", "clock_behind") and "before the command" in turn["unreadable"]
        for e in (walk, turn):
            assert not any(e[k] for k in ("succeeded", "succeeded_but_A1", "succeeded_look", "succeeded_but_A1_look"))
            assert e["clock"]["mean_factor"] == pytest.approx(0.9, abs=1e-3) and e["failed_terms"] == []
            # the reference time reached is recorded, and the monitor's credit is the survival times the clock's share
            assert e["clock"]["t_ref_reached"] == pytest.approx(0.9 * 1.05 * e["survival_s"], rel=1e-3)
            assert m2.has_clock(e) and m2.survival_credit(e) == pytest.approx(0.9 * e["survival_s"], rel=1e-3)
        # the slow walk whose window is still read fails the bar's distance, and is no unread episode
        assert (nominal["outcome"], nominal["succeeded"]) == ("read", False) and "along" in nominal["failed_terms"]
        assert nominal["look"]["own"]["steps"] < nominal["look"]["reference"]["steps"]
        assert [e["outcome"] for e in m2._checked(eps, pairs, policy=True)] == ["unread", "read", "unread"]
    # rule K4: a later freeze of the set that changes nothing the run used is accepted, and both freezes are named
    doc = s2io.read_json(d / "set.json")
    manifest = dict(doc["freezes"][0]["manifest"], extra=dict(kind="walk", sha256="0" * 64, entry_sha256="0" * 64))
    doc["freezes"].append(dict(index=1, at="fixture", commit=None, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest)))
    s2io.write_json(d / "set.json", doc)
    later = s2rows.policy_row("R5/42", BODY, job, 0, root=sets)
    p = later.provenance()
    assert p["later_freeze"] is True and p["set_freeze_now"]["index"] == 1 and p["run"]["set_freeze"] == fz
    assert later.describe()["run"]["set_freeze"] == fz
    # ... and one that changes a file the run used is refused
    doc["freezes"][1]["manifest"]["v0.4"] = dict(doc["freezes"][1]["manifest"]["v0.4"], sha256="1" * 64)
    doc["freezes"][1]["manifest_sha256"] = s2io.sha256_json(doc["freezes"][1]["manifest"])
    s2io.write_json(d / "set.json", doc)
    with pytest.raises(s2rows.RowError, match="rule K4"):
        s2rows.policy_row("R5/42", BODY, job, 0, root=sets)


@needs_run
def test_the_equivalence_check_plays_a_look_sets_turn_at_j1s_leave_point(job, tmp_path, monkeypatch):
    """The lead's ruling 37 (b) of 17 Nov 2026: ``equivalence_check`` on a look set (a synthetic one under
    ``tmp_path``, built from Stage 1's arrays; a freshly initialised policy; the episode cut to 1 s) runs inside
    ``lookbar.look_set_placement``: the row's built condition, whose plan the env is given, plays the turn at join
    rule J1's leave point (``lookbar.leave_resolve``), and the row and the env stay digest for digest.  On the S1
    view nothing is placed: the row's plan is the bar's own and the comparison is as before."""
    from designlab.stage2 import g8a
    from tests import test_stage2_r5lib as R

    sets = tmp_path / "sets"
    R.make_look(sets, set_name="V6", stand_in_walks=True)
    look_job = tmp_path / "jobs" / "R5-look-s42"
    make_job(look_job, BODY, "V6", root=sets)
    seen, real = [], g8a.row_episode

    def spy(row, ctx, cond, seed):
        r = real(row, ctx, cond, seed)
        seen.append((cond, [list(x) for x in r["built"].plan], sorted(lookbar.placed())))
        return r

    monkeypatch.setattr(g8a, "row_episode", spy)
    r = s2rows.equivalence_check(look_job, 0, BODY, "V6", "turn/q90_right", 3042, 0, root=sets, seconds=1.0)
    ((cond, plan, placed),) = seen
    d = setio.set_dir("V6", BODY, sets)
    docs = setio.load(d)
    planner = bar.BarPlanner(d, docs["refs.json"]["references"], docs["transitions.json"]["transitions"],
                             docs["set.json"])
    want = [list(x) for x in lookbar.leave_resolve(planner, cond)["plan"]]
    assert plan == want and want != [list(x) for x in planner.resolve(cond)["plan"]]
    assert placed == [lookbar.placement_key("V6", BODY, d)] and lookbar.placed() == set()
    assert (r["set"], r["condition"], r["identical"], r["first_difference"]) == ("V6", "turn/q90_right", True, None)
    assert r["reference_steps"] == r["compared_steps"] == 500
    # the S1 view: nothing placed
    seen.clear()
    r = s2rows.equivalence_check(job, 0, BODY, "S1", "walk/brisk", 3042, 0, seconds=1.0)
    ((cond, plan, placed),) = seen
    assert placed == [] and r["identical"] and r["reference_steps"] == r["compared_steps"] == 500
    assert vars(bar.BarPlanner)["_resolve"] is lookbar._BAR_RESOLVE


@needs_run
def test_rule_k4_at_the_episode_the_files_the_bar_reads_are_the_runs(tmp_path):
    """A row is built, and checked by rule K4, against the set under one root; the bar is then given the same set
    name under another root (``--set-root``, ``episodes(root=)``), where the 0.40 m/s walk is another file under a
    freeze of its own.  The row's check at the episode (``S2Policy._check_files``) is the one guard there: the
    episode is an error before anything is simulated."""
    a, b = tmp_path / "A", tmp_path / "B"
    make_t0(a)
    make_t0(b, refs=(("nominal", "nominal"), (W, "nominal")))  # (Stage 1's nominal walk's file under the walk's name)
    man = [s2io.read_json(setio.set_dir(SET, BODY, r) / "set.json")["freezes"][0]["manifest"] for r in (a, b)]
    assert man[0][W]["sha256"] != man[1][W]["sha256"] and man[0]["nominal"] == man[1]["nominal"]
    job = tmp_path / "jobs" / "R5-k4-s42"
    run = make_job(job, BODY, SET, root=a)
    assert f"refs/{W}.npz" in run["settings"]["inputs"]["files"]
    s2rows.configure({"R5/42": dict(job_dir=job, steps=0, root=a)})  # built against root A: rule K4 holds there
    rows = lookbar.rows(BODY, ["R5/42@10ms"])
    policy = rows["R5/42@10ms"].inner.inner.policy
    key = f"walk/{W}"
    # the same row, the same condition: prepared on root A, refused on root B, by the file the run recorded
    on_a = lookbar.open_set(BODY, SET, [key], root=a)
    policy.prepare(on_a["ctx"], on_a["conds"][key], on_a["ctx"].built(on_a["conds"][key]))
    on_b = lookbar.open_set(BODY, SET, [key], root=b)
    with pytest.raises(s2rows.RowError, match=rf"R5/42: refs/{W}\.npz in .* are not the files the run recorded \(rule K4\)"):
        policy.prepare(on_b["ctx"], on_b["conds"][key], on_b["ctx"].built(on_b["conds"][key]))
    # through the monitor's entry: an error, and no step of the episode was simulated
    e = lookbar.episodes(BODY, SET, rows, [(key, 6004)], root=b)[0]
    assert e["outcome"] == "error" and "RowError" in e["error"] and "rule K4" in e["error"]
    assert f"refs/{W}.npz" in e["error"] and str(setio.set_dir(SET, BODY, b)) in e["error"]
    assert e["steps"] is None and e["clock"] is None and e["fell"] is None and not e["succeeded"]


# ------------------------------------------------------------------------------------------------
# Slow: whole episodes against the env
# ------------------------------------------------------------------------------------------------
WHOLE = [("walk/nominal", 3042), ("walk/brisk", 3043), ("start_stop/twice", 3042), ("push/walking_left", 3044),
         ("turn/walking_0.35_left", 3042), ("carpet/robot_floor_carpet", 3042)]


def whole(job: Path, body: str, pairs, k: int = 5, row: str | None = None) -> list:
    out = []
    for key, seed in pairs:
        r = s2rows.equivalence_check(job, 0, body, "S1", key, seed, k, row=row)
        assert r["identical"] and r["ends_alike"] and r["first_difference"] is None, (key, seed, r)
        assert r["compared_steps"] == r["reference_steps"] > 500 and r["action_abs_max"] > 0.0
        assert r["k"] == k and (r["reference_fell_step"] is None) == (r["env_terminated_step"] is None)
        out.append(r)
    return out


@slow
@needs_run
@pytest.mark.parametrize("scale", [1.0, 20.0])
def test_equivalence_at_10_ms_on_the_s1_view_of_the_75_cm_body(job, tmp_path, scale):
    """Whole stock episodes at k = 5: a policy that moves the joints a little, and one whose actions fell the robot
    (the row at a physics step, the env in the env step that holds it)."""
    if scale != 1.0:
        job = tmp_path / "R5-75-M-H-10-s42"
        make_job(job, BODY, scale=scale)
    rs = whole(job, BODY, WHOLE)
    if scale == 1.0:
        assert max(r["action_abs_max"] for r in rs) < 0.2
    else:
        assert max(r["action_abs_max"] for r in rs) > 0.2 and any(r["reference_fell_step"] is not None for r in rs)
    assert all(r["row"] == "R5/42@10ms" for r in rs)


@slow
@pytest.mark.skipif(not (TORCH and _have(BODY_X) and ONE_THREAD), reason="needs the -X body's model and Stage 1 arrays "
                    "(tools/restore_inputs.py --group x), torch and one BLAS thread")
def test_equivalence_at_10_ms_on_the_x_body_with_the_head_neutral_and_with_the_head_script(tmp_path):
    job = tmp_path / "R5-75-M-H-10-X-s42"
    make_job(job, BODY_X)
    lay = s2io.read_json(s2load.checkpoint_files(job, 0)["record"])["layout"]
    assert (lay["actor_dim"], lay["action_dim"], lay["head_x"]) == (102, 17, True)
    pairs = [("walk/nominal", 3042), ("walk/brisk", 3043)]
    neutral = whole(job, BODY_X, pairs)
    h1 = whole(job, BODY_X, pairs, row="R5G/42")
    assert [r["row"] for r in h1] == ["R5G/42@10ms"] * 2
    # the head script moves the neck and the jaw: another episode than the head neutral's, and the same in row and env
    row = s2rows.policy_row("R5G/42", BODY_X, job, 0)
    assert row.describe()["head"] == "H1" and s2rows.policy_row("R5/42", BODY_X, job, 0).describe()["head"] == "neutral"
    assert h1[0]["look"]["state_digest"] != neutral[0]["look"]["state_digest"]
    from designlab.stage2 import r5core

    a, b = r5core.HeadScript.fixed(3042, 18.0), r5core.HeadScript.fixed(3042, 18.0)
    assert a.describe() == b.describe() and {e[0] for e in a.events} == {"gaze", "jaw"}


@slow
@needs_run
def test_two_spawned_workers_rebuild_the_policy_row_from_its_spec(job, tmp_path, monkeypatch):
    """The bar's runner with two worker processes: each rebuilds the rows from their specs, checks the files'
    hashes, loads the checkpoint itself and writes its episodes' sidecars; the records are those of one process."""
    stock_cut(monkeypatch, 1.0)
    s2rows.configure({"R5/42": dict(job_dir=job, steps=0)})
    docs = {}
    for procs in (2, 1):
        out = tmp_path / f"procs{procs}"
        doc = lookbar.run("S1", BODY, out, ["R5/42@10ms", "R0L@10ms"], ["walk/brisk"], episodes=4, procs=procs,
                          stage1_run=RUN, log=quiet)
        assert doc["bar"]["sessions"][-1]["episodes_run"] == 8 and not doc["bar"]["sessions"][-1]["incomplete"]
        bdir = bar.block_dir(out, "S1", BODY, bar.HEADLINE_BLOCK)
        docs[procs] = {r: s2io.read_json(bdir / "look" / r / "walk" / "brisk.json") for r in ("R5/42@10ms", "R0L@10ms")}
    for r in ("R5/42@10ms", "R0L@10ms"):
        two, one = docs[2][r], docs[1][r]
        assert [e["state_digest"] for e in two["episodes"]] == [e["state_digest"] for e in one["episodes"]]
        assert two["bar_record"]["key_sha256"] == one["bar_record"]["key_sha256"] and two["counts"] == one["counts"]
    pol = docs[2]["R5/42@10ms"]["policy"]
    assert pol["episodes_checked"] == 4 and pol["fingerprint_max_abs_diff"] == 0.0 and pol["same_checkpoint"] is True


@slow
@needs_run
def test_r6_row_against_the_r6_env(tmp_path):
    job = tmp_path / "R6-75-M-H-10-s42"
    make_job(job, BODY, mode="R6")
    row = s2rows.policy_row("R6/42", BODY, job, 0)
    d = row.describe()
    assert isinstance(row, s2rows.R6Row) and not isinstance(row, panels.TrackerRow) and d["mode"] == "R6"
    assert d["controller"].startswith("R6: no reference controller") and "span_rule" in d["action"]["scales"]
    assert sorted(d) == KEYS and panels.row_from_spec(plain(row.spec())).describe() == d
    rs = whole(job, BODY, [("walk/nominal", 3042), ("walk/brisk", 3043), ("stand/hard", 3042)])
    assert [r["row"] for r in rs] == ["R6/42@10ms"] * 3
    r0 = s2rows.equivalence_check(job, 0, BODY, "S1", "walk/brisk", 3042, 0, seconds=2.0)
    assert r0["identical"] and r0["row"] == "R6/42@0ms" and r0["reference_steps"] == 1000
    # the bar judges an R6 episode like any other row's, and the monitor's table holds its clock times
    m2 = pytest.importorskip("designlab.stage2.monitor2")
    s2rows.configure({"R6/42": dict(job_dir=job, steps=0)})
    pairs = m2.pairs_for("S1", tiny=True)
    eps = m2._checked(lookbar.episodes(BODY, "S1", lookbar.rows(BODY, ["R6/42@10ms"]), pairs), pairs, policy=True)
    assert [e["outcome"] in ("read", "fell") for e in eps] == [True, True] and all(e["error"] is None for e in eps)
    assert all(e["policy_check"]["mode"] == "R6" and e["survival_credit"] > 0 for e in eps)


@needs_run
def test_the_equivalence_check_gives_a_servo_draw_run_an_episode_seed(job, tmp_path):
    """The lead's fourth list (the builder's (f) of S2): a run whose config has the servo option's draw is given the
    panel seed as the env's episode seed, so the env draws ``servo.training_draw(seed)`` instead of refusing the
    reset; the result names the set.  The row as the bar runs it is not served, so the two run different models.  A
    run without the option is checked as before (no episode seed given, no ``servo`` key)."""
    from designlab.stage2 import servo

    draw = tmp_path / "R5-75-M-H-10-s42-draw"
    run = make_job(draw, BODY, servo="draw")
    assert {c["servo"] for c in run["settings"]["env_configs"]} == {"draw"}
    r = s2rows.equivalence_check(draw, 0, BODY, "S1", "walk/brisk", 3043, 0, seconds=0.1)
    assert r["servo"] == dict(option="draw", episode_seed=3043, setting="train/3043")
    assert servo.training_draw(3043).name == "train/3043"
    assert r["reference_steps"] == 50 and not r["identical"] and r["first_difference"] is not None
    plain_ = s2rows.equivalence_check(job, 0, BODY, "S1", "walk/brisk", 3043, 0, seconds=0.1)
    assert "servo" not in plain_ and plain_["identical"] and plain_["reference_steps"] == 50
