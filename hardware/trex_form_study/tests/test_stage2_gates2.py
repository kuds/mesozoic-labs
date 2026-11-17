"""Track B's gates and validation records (``designlab/stage2/gates2.py``) and the run chain
(``stage2/tb/run_trackb.sh``): round C design 3.7, sections 4 and 6; rules Z1, Z2, K3, X1 of
``STAGE2_TRACKB_PLAN.md``; ``TRACKB_RULINGS6_16NOV.md`` ruling 2; ``ROUND_C_RULINGS_16NOV.md``.

No look set is frozen yet, so nothing here runs on one: the closed-loop gates are tested on Stage 1's own run
directory (the view ``S1``) and on a synthetic frozen set ``T0`` built in ``tmp_path`` from Stage 1's arrays
(``tests/test_stage2_g8a.py``'s fixture); G9-ref's kinematic parts on the look ladder's arrays, on the S1 view and on
the frozen 75 cm set V0 (a check that leaves its record in ``tmp_path`` only); kill and resume and the rates on the
stub env of the trainer's tests (``tests/test_stage2_train2.py``).  Every record goes to ``tmp_path``.  Tests that
need Stage 1's arrays, the look ladder's arrays, the frozen sets' arrays, the round-two models or torch skip without
them (``python tools/restore_inputs.py restore``).

Run from hardware/trex_form_study:
    PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \\
        python -m pytest -p no:cacheprovider tests/test_stage2_gates2.py

Slow (``STAGE2_SLOW=1``): R6's start on the S1 view (an R6 env stepped, a tiny R6 run of 2 rollouts in 2 workers).
"""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage2 import gates2, s2io  # noqa: E402

PY = sys.executable
BODY = "75-M-H-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
CHAIN = ROOT / "stage2" / "tb" / "run_trackb.sh"
SLOW = os.environ.get("STAGE2_SLOW") == "1"
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)


def _have_models(body: str = BODY) -> bool:
    try:
        from designlab.stage1 import plant

        return all(Path(f).exists() for f in plant.model_files(body))
    except Exception:  # noqa: BLE001  (no model record: the same skip)
        return False


HAVE_S1 = (RUN / BODY / "transitions" / "stop.npz").exists() and _have_models()
HAVE_LADDER = _have_models() and all((ROOT / "stage2" / "look" / "refs" / BODY / s / "V6.npz").exists()
                                     for s in ("nominal", "v040"))
HAVE_V0 = (ROOT / "stage2" / "sets" / "V0" / BODY / "refs" / "v0.4.npz").exists() and _have_models()
needs_s1 = pytest.mark.skipif(not (HAVE_S1 and ONE_THREAD), reason=(
    "needs the Stage 1 arrays and the round-two models (tools/restore_inputs.py restore --group picks-models "
    "picks-arrays) and one BLAS thread"))
needs_ladder = pytest.mark.skipif(not (HAVE_LADDER and ONE_THREAD), reason=(
    "needs the look ladder's arrays and the round-two models (tools/restore_inputs.py restore --group picks-models "
    "look) and one BLAS thread"))
needs_v0 = pytest.mark.skipif(not (HAVE_V0 and ONE_THREAD), reason=(
    "needs the frozen sets' arrays (tools/restore_inputs.py restore --group sets) and one BLAS thread"))
slow = pytest.mark.skipif(not SLOW, reason="real env and a tiny training run on the S1 view; set STAGE2_SLOW=1")


def _env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")}
    env.update(PYTHONPATH=".", MUJOCO_GL="disable", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _record(path) -> dict:
    doc = json.loads(Path(path).read_text())
    assert gates2.record_sha_ok(doc), "the record's sha256 is not its own"
    return doc


PROVENANCE = ("commit", "dirty", "dirty_paths", "versions", "blas_threads", "at", "sha256", "sources", "rule")


# ------------------------------------------------------------------------------------------------
# The module: constants, imports, refusals (no plant)
# ------------------------------------------------------------------------------------------------
def test_constants_are_the_rules():
    assert gates2.G8A_CONDITIONS[:8] == ("walk/v0.4", "turn/q90_left", "turn/q90_right", "nudge/forward",
                                         "nudge/backward", "nudge/left", "nudge/right", "stop/two_step")
    assert gates2.G8A_CONDITIONS[8:] == ("walk/v0.4_mu0.54", "walk/nominal") and len(set(gates2.G8A_CONDITIONS)) == 10
    assert (gates2.G8A_SEEDS, gates2.G8A_DELAYS) == ((3042, 3043), (0, 5))
    k3 = gates2.K3
    assert (k3["mode"], k3["seed"], k3["steps"], k3["checkpoint_every"], k3["kill_at"], k3["workers"]) == (
        "R5", 42, 131_072, 32_768, 81_920, (2, 3))
    assert k3["n_envs"] == 8 and k3["n_steps"] == 1024 and k3["monitor_every"] % k3["checkpoint_every"] == 0
    # the kill lands between two checkpoints, at a rollout start
    assert 2 * 32_768 < 81_920 < 3 * 32_768 and 81_920 % (8 * 1024) == 0
    t = gates2.K3_TINY
    assert t["checkpoint_every"] * 2 < t["kill_at"] < t["checkpoint_every"] * 3 and t["kill_at"] % (8 * 64) == 0
    assert gates2.TAIL_SCALE == 0.1 and gates2.JOIN_TOL_RAD == 1e-6 and gates2.ZERO_ACTION_S == 2.0
    for key in ("z1", "z2_ref", "z2_run", "k3", "rates", "gl_start", "r6_start", "x1"):
        assert gates2.RULES[key]
    assert "a tenth about its mean" in gates2.RULES["z2_ref"] and "toe-off and its side of 0.02 rad" in gates2.RULES["z2_ref"]
    for key in ("z1", "z2_ref", "z2_run", "k3"):
        assert gates2.MODULE_READING in gates2.RULES[key]


def test_the_look_sets_are_lookbars():
    lookbar = pytest.importorskip("designlab.stage2.lookbar")
    assert gates2.LOOK_SETS == lookbar.LOOK_SETS


def test_the_module_assigns_mujoco_gl_and_imports_nothing_heavy(tmp_path):
    code = ("import sys, os; from designlab.stage2 import gates2; "
            "print(os.environ['MUJOCO_GL'], os.environ.get('PYOPENGL_PLATFORM'), os.environ.get('GIT_OPTIONAL_LOCKS'), "
            "sorted(m for m in ('torch', 'stable_baselines3', 'mujoco', 'designlab.stage2.bar', "
            "'designlab.stage2.r5env', 'designlab.stage2.train2') if m in sys.modules))")
    env = _env(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    env.pop("GIT_OPTIONAL_LOCKS", None)
    r = subprocess.run([PY, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "disable None 0 []"


def _write_g9(path: Path, set_name="V6", body=BODY, pass_=True, freeze="f" * 64, schema=None) -> Path:
    doc = dict(schema=schema or gates2.SCHEMAS["g9_ref"], gate="G9-ref", body=body, set=set_name, pass_=pass_,
               set_freeze=dict(index=0, manifest_sha256=freeze), at="2026-11-17T00:00:00Z")
    gates2.finish(doc, path)
    return path


def _look_set(root: Path, set_name="V6", body=BODY, freeze="f" * 64) -> Path:
    d = root / set_name / body
    d.mkdir(parents=True)
    s2io.write_json(d / "set.json", dict(set=set_name, body=body, freezes=[dict(index=0, manifest={},
                                                                               manifest_sha256=freeze)]))
    return d


def test_closed_loop_gates_refuse_a_look_set_without_a_passed_g9_ref(tmp_path, capsys):
    """Rule X1: g8a, rates, kill-resume and r6-start on a look set need a passed G9-ref of that set, body and
    freeze; each refusal is exit 2 and nothing is run (no env is built: these calls take milliseconds)."""
    missing = tmp_path / "none.json"
    root = tmp_path / "sets"
    _look_set(root)
    failed = _write_g9(tmp_path / "failed.json", pass_=False)
    other_freeze = _write_g9(tmp_path / "old.json", freeze="0" * 64)
    other_body = _write_g9(tmp_path / "body.json", body="60-N-T-10")
    edited = _write_g9(tmp_path / "edited.json")
    doc = json.loads(edited.read_text())
    doc["pass_"] = True
    doc["body"] = BODY
    doc["note"] = "changed after its hash"
    edited.write_text(json.dumps(doc))
    cmds = dict(g8a=[], rates=[], **{"kill-resume": ["--out", str(tmp_path / "kr")]}, **{"r6-start": []})
    for g9, why in ((missing, "there is no G9-ref record"), (failed, "G9-ref did not pass"),
                    (other_freeze, "another freeze"), (other_body, "of V6/60-N-T-10"),
                    (edited, "its sha256 is not its own")):
        for cmd, extra in cmds.items():
            code = gates2.main([cmd, "--body", BODY, "--set", "V6", "--g9-ref", str(g9), "--set-root", str(root)]
                               + extra)
            err = capsys.readouterr().err
            assert code == gates2.EXIT_ERROR and "refused" in err and why in err and "rule X1" in err, (cmd, err)
    assert not (tmp_path / "kr").exists()
    good = _write_g9(tmp_path / "good.json")
    c = gates2.require_g9_ref(BODY, "V6", good, root)
    assert c["pass_"] is True and c["sha256"] == s2io.sha256_file(good) and c["set_freeze"]["manifest_sha256"] == "f" * 64
    # not a look set: nothing is asked
    assert gates2.require_g9_ref(BODY, "V0", missing, root) is None and gates2.require_g9_ref(BODY, "S1") is None
    # a look set that is not here (or never frozen) is refused too
    with pytest.raises(gates2.Refused, match="not here or not frozen"):
        gates2.require_g9_ref(BODY, "V6noF5", _write_g9(tmp_path / "nof5.json", set_name="V6noF5"), root)


def test_a_record_holds_its_own_sha256(tmp_path):
    doc = gates2.finish(dict(gate="x", value=1.5, nested=dict(a=[1, 2])), tmp_path / "r.json")
    assert doc == json.loads((tmp_path / "r.json").read_text()) and gates2.record_sha_ok(doc)
    assert doc["sha256"] == s2io.sha256_json({k: v for k, v in doc.items() if k != "sha256"})
    assert not gates2.record_sha_ok(dict(doc, value=2.0)) and not gates2.record_sha_ok({"a": 1})


def test_the_dev_episode_list_is_keyed_by_step_set_and_body(tmp_path):
    f = tmp_path / "dev_episodes.json"
    entry = dict(step="V5-filter", command="train2 filter ...", set="V6", body="75-M-H-10", rows_or_mode="R5",
                 count=80, reason="rule O4")
    gates2.dev_episode(f, dict(entry, state="listed"))
    gates2.dev_episode(f, dict(entry, step="V4-rates", count=3, state="listed"))
    doc = gates2.dev_episode(f, dict(entry, state="done"))
    assert doc["schema"] == gates2.SCHEMAS["dev_episodes"] and doc["rule"] == gates2.RULES["x1"]
    eps = json.loads(f.read_text())["episodes"]
    assert [e["step"] for e in eps] == ["V5-filter", "V4-rates"]
    assert eps[0]["state"] == "done" and eps[0]["listed_at"] and eps[0]["done_at"] and eps[1]["state"] == "listed"
    assert eps[0]["count"] == 80 and eps[0]["command"] == "train2 filter ..." and "commit" in eps[0]
    assert gates2.main(["dev-episode", "--file", str(f), "--step", "V4-rates", "--command", "c", "--set", "V6",
                        "--body", BODY, "--what", "R5", "--count", "3", "--reason", "r", "--state", "done"]) == 0
    assert [e["state"] for e in json.loads(f.read_text())["episodes"]] == ["done", "done"]


# ------------------------------------------------------------------------------------------------
# G8a (rule Z1): the S1 view's stock form and the bar form on a synthetic set
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def _release():
    yield
    from designlab.stage2 import g8a

    g8a.release()


@needs_s1
def test_g8a_record_and_exit_codes_on_the_s1_view(tmp_path, monkeypatch, capsys, _release):
    """The stock form on the S1 view (2 episodes cut to 1.013 s): the record's schema; exit 0; then one action
    entry forced to 1e-6 in the env: exit 1, and the record and the printed line name the condition, seed, delay and
    the first differing physics step."""
    from designlab.stage2 import g8a

    rec = tmp_path / "g8a.json"

    def argv(delays):
        return ["g8a", "--body", BODY, "--set", "S1", "--conditions", "walk/brisk", "--seeds", "3042", "--delays",
                delays, "--cut-s", "1.013", "--record", str(rec)]

    assert gates2.main(argv("0,3")) == 0
    doc = _record(rec)
    for key in ("schema", "gate", "body", "set", "form", "set_freeze", "conditions", "seeds", "delays", "cut_s",
                "complete", "episodes", "identical", "of", "not_run", "differences", "pass_", "key_context",
                "g9_ref") + PROVENANCE:
        assert key in doc, key
    assert doc["schema"] == "s2-gate-g8a-1" and doc["form"] == "stock" and doc["complete"] is False
    assert (doc["identical"], doc["of"], doc["pass_"], doc["differences"]) == (2, 2, True, [])
    for e in doc["episodes"]:
        for key in gates2.EPISODE_KEYS:
            assert key in e, key
        assert e["identical"] and e["compared_steps"] == e["reference_steps"] == 506 and e["env_steps"] == 51
    assert [(e["condition"], e["seed"], e["k"]) for e in doc["episodes"]] == [("walk/brisk", 3042, 0),
                                                                              ("walk/brisk", 3042, 3)]
    assert doc["sources"]["track_b"]["designlab/stage2/gates2.py"] == s2io.sha256_file(gates2.__file__)
    capsys.readouterr()
    # a forced difference: one action entry at 1e-6 (rule Z1: a zero action is not "almost" zero)
    inner = g8a.env_episode

    def nudged(e, *a, **k):
        act = np.zeros(e.layout["action_dim"], np.float32)
        act[0] = 1e-6
        return inner(e, *a, **dict(k, action=act))

    monkeypatch.setattr(g8a, "env_episode", nudged)
    assert gates2.main(argv("0")) == 1
    out = capsys.readouterr().out
    doc = _record(rec)
    assert doc["pass_"] is False and doc["identical"] == 0 and doc["of"] == 1
    (d,) = doc["differences"]
    assert (d["condition"], d["seed"], d["k"], d["delay_ms"]) == ("walk/brisk", 3042, 0, 0)
    assert 0 <= d["first_difference"] < 10  # (inside the first env step)
    assert f"walk/brisk seed 3042 k 0 (0 ms): first difference at physics step {d['first_difference']}" in out
    assert "FAIL" in out


@needs_s1
def test_g8a_bar_form_on_a_synthetic_set_with_a_condition_not_run(tmp_path, _release):
    """``g8a.bar_form`` through the gate on the frozen synthetic set ``T0``: an episode of ``walk/v0.4`` at 0 and
    10 ms (cut to 1.013 s) is identical; ``turn/q90_left``, which ``T0`` holds no reference for, is not run with the
    set's reason.  Rule Z1's ten keys are the conditions ``g8a.conditions`` finds on ``T0``."""
    from designlab.stage2 import g8a, setio
    from tests import test_stage2_g8a as G

    G.make_t0(tmp_path)
    assert tuple(g8a.conditions(setio.load(setio.set_dir("T0", BODY, tmp_path)))) == gates2.G8A_CONDITIONS
    rec = tmp_path / "rec" / "g8a.json"
    doc = gates2.g8a(BODY, "T0", rec, conditions=("walk/v0.4", "turn/q90_left"), seeds=(3042,), delays=(0, 5),
                     cut_s=1.013, root=tmp_path)
    assert doc == _record(rec)
    assert doc["form"] == "bar" and doc["complete"] is False and doc["pass_"] is True and doc["g9_ref"] is None
    assert (doc["identical"], doc["of"]) == (2, 2) and [e["row"] for e in doc["episodes"]] == ["R0@0ms", "R0@10ms"]
    assert [n["condition"] for n in doc["not_run"]] == ["turn/q90_left"] and doc["not_run"][0]["reason"]
    assert doc["set_freeze"]["index"] == 0 and doc["conditions_of_the_rule"] == list(gates2.G8A_CONDITIONS)
    assert doc["sources"]["bar"] and doc["sources"]["panels"] and doc["key_context"]


def _fake_bar_form(calls: list, compared: int = 506):
    """A stand-in of ``g8a.bar_form``: every episode identical, with ``compared`` of 506 physics steps compared."""

    def bar_form(body, set_name, conds, seeds, delays, cut_s=None, root=None):
        calls.append((body, set_name, tuple(conds), tuple(seeds), tuple(delays), cut_s))
        eps = [dict(condition=c, seed=s, k=k, reference_steps=506, env_steps=51, compared_steps=compared,
                    first_difference=None, reference_fell_step=None, env_terminated_step=None, ends_alike=True,
                    identical=True, interference_first_env_step=None) for c in conds for s in seeds for k in delays]
        return dict(rule="stand-in", episodes=eps, identical=len(eps), of=len(eps), not_run=[], set_freeze=None,
                    key_context=None)

    return bar_form


def test_g8a_counts_a_short_comparison_as_a_difference(tmp_path, monkeypatch):
    """Rule Z1's check: compared_steps == reference_steps in every episode.  An episode flagged identical that
    compared fewer physics steps than the row ran is a difference (exit 1), not a pass."""
    from designlab.stage2 import g8a

    calls = []
    monkeypatch.setattr(g8a, "bar_form", _fake_bar_form(calls, compared=500))
    argv = ["g8a", "--body", BODY, "--set", "T9", "--conditions", "walk/v0.4", "--seeds", "3042", "--delays", "0",
            "--record", str(tmp_path / "g8a.json")]
    assert gates2.main(argv) == 1
    doc = _record(tmp_path / "g8a.json")
    assert doc["pass_"] is False and doc["identical"] == doc["of"] == 1
    assert [(d["condition"], d["seed"], d["k"]) for d in doc["differences"]] == [("walk/v0.4", 3042, 0)]
    monkeypatch.setattr(g8a, "bar_form", _fake_bar_form(calls, compared=506))
    assert gates2.main(argv) == 0 and _record(tmp_path / "g8a.json")["pass_"] is True


def test_g8a_writes_no_reduced_record_where_the_gates_record_lies(tmp_path, monkeypatch):
    """A run of fewer conditions, seeds or delays, a cut, or the Stage 1 view's form is a check of the command: it is
    refused (exit 2, nothing run or written) when its record would lie under stage2/tb/gates/ (here a stand-in
    directory), and runs anywhere else; the gate's own form is written there."""
    from designlab.stage2 import g8a

    gates = tmp_path / "gates"
    monkeypatch.setattr(gates2, "GATES", gates)
    calls = []
    monkeypatch.setattr(g8a, "bar_form", _fake_bar_form(calls))
    rec = gates / "T9" / BODY / "g8a.json"
    for extra in (["--conditions", "walk/v0.4"], ["--seeds", "3042"], ["--delays", "0"], ["--cut-s", "1.0"]):
        assert gates2.main(["g8a", "--body", BODY, "--set", "T9", "--record", str(rec)] + extra) == 2, extra
    assert gates2.main(["g8a", "--body", BODY, "--set", "S1", "--record", str(gates / "S1" / BODY / "g8a.json")]) == 2
    assert calls == [] and not gates.exists()
    with pytest.raises(gates2.Refused, match="never rule Z1's record"):
        gates2.g8a(BODY, "T9", rec, seeds=(3042,))
    assert gates2.main(["g8a", "--body", BODY, "--set", "T9", "--record", str(tmp_path / "check.json"),
                        "--seeds", "3042"]) == 0
    assert gates2.main(["g8a", "--body", BODY, "--set", "T9", "--record", str(rec)]) == 0
    doc = _record(rec)
    assert doc["complete"] is True and doc["of"] == 40 and calls[-1][2:] == (gates2.G8A_CONDITIONS, (3042, 3043),
                                                                             (0, 5), None)


@needs_v0
def test_rule_z1s_keys_are_the_conditions_g8a_finds_on_v0():
    from designlab.stage2 import g8a, setio

    assert tuple(g8a.conditions(setio.load(setio.set_dir("V0", BODY)))) == gates2.G8A_CONDITIONS


# ------------------------------------------------------------------------------------------------
# G9-ref (rule Z2): (c) on the look ladder's arrays, (f) on the S1 view, the whole gate on V0
# ------------------------------------------------------------------------------------------------
def _ladder_arrays(tmp_path, speed: str, body: str = BODY, variant: str = "V6") -> tuple:
    from designlab.stage2 import lookread
    from tests import test_stage2_lookread as L

    p = L.plant_of(body)
    z = L.load(L._rung(body, speed, variant))
    d = tmp_path / f"{body}_{speed}_{variant}"
    d.mkdir()
    track = L._rung_as_a_run_dir(d, z, lookread.weight(p), 20)
    return p, lookread.played(p, track, d, gates2.WALK_TA_REF, gates2.WALK_TA_REF + 1.05 * gates2.WALK_WINDOW_S,
                              lead_s=lookread.LEAD_S)


@needs_ladder
@pytest.mark.parametrize("speed", ("nominal", "v040"))
def test_g9_ref_part_c_on_the_look_ladders_arrays(tmp_path, speed):
    """Each degraded playback fails exactly its term, the unchanged one passes all four, the 3-step cut reads
    too_few_steps, and the fifth (the tail scaled to a tenth about its mean: ruling 2 of TRACKB_RULINGS6_16NOV) fails
    the tail term alone, through the tail's own amplitude (0.012 rad under the 0.02 rad band)."""
    from designlab.stage2 import lookread

    p, a = _ladder_arrays(tmp_path, speed)
    c = gates2.controls(p, a, flat_feet=True)
    assert c["ok"] and c["kinds"] == ["bob_half", "neck_3deg", "tail_late", "flat_feet", "tail_tenth"]
    b = c["brief"]
    assert b["unchanged"] and b["cut"]
    for kind, term in (("bob_half", "bob"), ("neck_3deg", "head"), ("tail_late", "tail"), ("flat_feet", "toe_off"),
                       ("tail_tenth", "tail")):
        assert b[kind]["failed"] == [term] and b[kind]["ok"], kind
    t = c["tail_tenth"]
    assert t["own_amp_rad"] == pytest.approx(0.1 * t["reference_amp_rad"], rel=0.02)
    assert t["own_amp_rad"] < lookread.TAIL_AMP_MIN_RAD <= t["reference_amp_rad"]
    assert t["compare"]["tail_phase_diff_cycles"] == pytest.approx(0.0, abs=1e-6)  # (the phase is unchanged)
    # the degraded copy changes the tail joint only
    q = gates2.tail_tenth(p, a["poses"])
    i = lookread._ids(p.m)["tail"]
    assert np.array_equal(np.delete(q, i, axis=1), np.delete(np.asarray(a["poses"]), i, axis=1))
    assert np.std(q[:, i]) == pytest.approx(0.1 * np.std(a["poses"][:, i]), rel=1e-9)
    # without flat feet (a set other than V6): four controls, still all ok
    c2 = gates2.controls(p, a, flat_feet=False)
    assert c2["ok"] and "flat_feet" not in c2["kinds"] and "flat_feet" not in c2["brief"]


@needs_ladder
def test_the_fifth_control_fails_nothing_but_the_tail_where_the_tail_is_not_read(tmp_path):
    """A control is only a control where its term is read: on the 60 cm fr05 rung the toe-off is under 0.02 rad, so
    flat feet cannot fail it, and the gate makes its controls on nominal and v0.4 only (``CONTROL_WALKS``)."""
    if not _have_models("60-N-T-10") or not (ROOT / "stage2" / "look" / "refs" / "60-N-T-10" / "fr05" / "V6.npz").exists():
        pytest.skip("needs the 60 cm look ladder")
    p, a = _ladder_arrays(tmp_path, "fr05", body="60-N-T-10")
    c = gates2.controls(p, a, flat_feet=True)
    assert c["brief"]["flat_feet"]["failed"] == [] and not c["ok"] and c["brief"]["tail_tenth"]["ok"]
    assert gates2.CONTROL_WALKS == ("nominal", "v0.4")


def _seg(kind, S, **kw):
    return types.SimpleNamespace(kind=kind, name=kind, t0=kw.pop("t0", 0.0), S=np.asarray(S, float), **kw)


def test_the_join_rule_on_tables():
    """G9-ref (f): the jump at a join is between the joint targets segment i gives at its own end and segment
    i + 1's first frame; within 1e-6 rad passes."""
    walk = np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3]])
    per = _seg("periodic", walk, duration=2 * 4 * 0.01, step=0.01, n=4)  # two whole strides: it ends at frame 0
    np.testing.assert_array_equal(gates2._segment_end(per), walk[0])
    half = _seg("periodic", walk, duration=2.5 * 4 * 0.01, step=0.01, n=4)  # two and a half: frame 2
    np.testing.assert_allclose(gates2._segment_end(half), walk[2], atol=1e-12)
    for jump, ok in ((0.0, True), (9e-7, True), (1e-6, True), (2e-6, False), (3e-5, False)):
        move = _seg("finite", [walk[0] + [jump, 0.0], walk[1]], t0=0.08)
        track = types.SimpleNamespace(segments=[per, move])
        (j,) = gates2.joins(track, 2)
        assert j["jump_rad"] == pytest.approx(jump, abs=1e-15) and j["ok"] is ok and j["at_ref_s"] == 0.08
    # a finite segment ends at its last row (the next segment's first frame, appended by control.Track); a hold
    # holds its one frame
    fin = _seg("finite", [[5.0, 5.0], [6.0, 6.0]])
    hold = _seg("hold", [[6.0, 6.0]])
    assert gates2.joins(types.SimpleNamespace(segments=[fin, hold]), 2)[0]["jump_rad"] == 0.0
    # only the first nj columns (the joint targets) are read
    wide = _seg("hold", [[6.0, 6.0, 99.0]])
    assert gates2.joins(types.SimpleNamespace(segments=[fin, wide]), 2)[0]["jump_rad"] == 0.0


@needs_s1
def test_g9_ref_part_f_on_the_s1_view():
    """Every task of the S1 view's library: its joins measured and 2 s of the zero action's targets finite.  Stage
    1's own moves do not join their walks to 1e-6 rad (the walk to stop: 3.8e-5 rad), so (f) fails there by the
    rule as written; the measurement is what the gate records (for the lead)."""
    v = gates2._SetView(BODY, "S1")
    f = gates2.part_f(v)
    assert sorted(f["tasks"]) == ["brisk", "stand", "stop", "walk"] and f["screen"] is None
    for t, x in f["tasks"].items():
        assert x["available"] and x["zero_action_finite"] is True and x["zero_action_s"] == 2.0, t
        assert all(set(j) == {"at_ref_s", "before", "after", "jump_rad", "ok"} for j in x["joins"])
    stand = f["tasks"]["stand"]
    assert stand["ok"] and [j["jump_rad"] for j in stand["joins"]] == [0.0]
    walk = f["tasks"]["walk"]
    assert [j["after"] for j in walk["joins"]] == ["periodic nominal", "finite stop"]
    assert walk["joins"][0]["jump_rad"] == 0.0 and walk["joins"][1]["jump_rad"] == pytest.approx(3.77e-5, rel=0.01)
    assert walk["ok"] is False and f["ok"] is False and f["max_jump_rad"] > gates2.JOIN_TOL_RAD


@needs_v0
def test_g9_ref_on_the_frozen_set_v0(tmp_path):
    """The whole gate, kinematic only, on V0 (no look feature; a check whose record stays in tmp_path): (a) cites
    the refcheck record the set's latest freeze cites, by path and sha256; (b) holds the ladder's two rungs of the
    body by their twins (V0's) and fails on L1_FREEZE's V6 hashes, lists every walk and turn item with its steps,
    lift-offs and its reference toe-off with its side of 0.02 rad; (c) is made without flat feet (not V6); (f) cites
    the screen record and measures every task.  Exit 1: G9-ref does not pass on V0."""
    rec = tmp_path / "gates" / "g9_ref.json"
    code = gates2.main(["g9-ref", "--body", BODY, "--set", "V0", "--record", str(rec)])
    assert code == 1
    doc = _record(rec)
    assert doc["schema"] == "s2-gate-g9-ref-1" and doc["form"] == "plain" and doc["kinematic_only"] is True
    assert doc["pass_"] is False and doc["set_freeze"]["index"] == 0
    a = doc["parts"]["a"]
    fz = s2io.read_json(ROOT / "stage2" / "sets" / "V0" / BODY / "set.json")["freezes"][-1]["refcheck"]
    assert a["ok"] is True and a["refcheck"]["path"] == fz["file"] and a["refcheck"]["sha256"] == fz["sha256"]
    assert a["blocked"] == [] and a["not_available"] == [] and "_doc" not in a
    assert not (tmp_path / "gates" / "refcheck").exists()  # (cited, not run again)
    b = doc["parts"]["b"]
    assert set(b["rungs"]) == {"nominal", "v040", "fr05"} and b["rungs"]["fr05"]["in_ladder"] is False
    for speed, walk in (("nominal", "nominal"), ("v040", "v0.4")):
        r = b["rungs"][speed]
        assert r["walk"] == walk and r["held"] and r["ladder_variant"] == "V0" and r["l1_variant"] == "V6"
        assert r["hash_ok"] is False and r["ok"] is False  # (V0's look files are not the owner's V6 pick)
        assert r["against_ladder"]["terms"]["head"] and r["against_ladder"]["terms"]["tail"]
    assert sorted(b["walks"]) == ["nominal", "v0.3", "v0.35", "v0.4", "v0.45", "v0.5"]
    assert all(w["ok"] and w["finite"] and w["steps"] >= 4 for w in b["walks"].values())
    items = b["items"]
    assert {"walk/v0.4", "walk/nominal", "turn/q90_left", "turn/q90_right"} <= set(items)
    assert items["turn/t90_1_left"]["status"] == "not run"
    for k, x in items.items():
        if x["status"] == "played":
            assert x["steps"] >= 4 and x["lift_offs"] >= 4 and x["ok"], k
            assert x["side"] == "under 0.02 rad" and x["toe_off_term"] == "not read", k  # (V0 has flat feet)
    c = doc["parts"]["c"]
    assert c["flat_feet"] is False and set(c["walks"]) == {"nominal", "v0.4"}
    assert all("flat_feet" not in w["brief"] for w in c["walks"].values())
    f = doc["parts"]["f"]
    assert f["screen"]["path"] == "stage2/sets/V0/75-M-H-10/screen.json" and f["unplayed"] == {}
    assert f["tasks"]["turn_1_left"]["available"] is False and f["tasks"]["walk_v0.4"]["zero_action_finite"]
    assert doc["followed"].startswith("no closed-loop episode runs on this set and body")
    # the record holds every file it played, each with the latest freeze's sha256, and those (f)'s library read
    manifest = s2io.read_json(ROOT / "stage2" / "sets" / "V0" / BODY / "set.json")["freezes"][-1]["manifest"]
    played = doc["inputs"]["played"]
    assert played["freeze"] == doc["set_freeze"] and doc["inputs"]["comparand"] is None
    assert {"refs/v0.4.npz", "refs/v0.5.npz", "transitions/start.npz", "transitions/turn90_pi4_left.npz"} <= set(
        played["files"])
    for fname, x in list(played["files"].items()) + list(doc["inputs"]["library"]["files"].items()):
        assert x["sha256"] == manifest[fname.split("/", 1)[1][:-len(".npz")]]["sha256"], fname
    assert doc["inputs"]["library"]["freeze"] == doc["set_freeze"] and "library_inputs" not in f


@needs_s1
def test_g9_ref_runs_round_as_refcheck_where_no_record_is_cited(tmp_path):
    """On the synthetic frozen set ``T0`` (no refcheck record anywhere) part (a) runs ``bar.refcheck`` and cites the
    record it wrote beside the gate's own, by path and sha256; the gate's other parts run and are recorded."""
    from tests import test_stage2_g8a as G

    G.make_t0(tmp_path)
    rec = tmp_path / "rec" / "g9_ref.json"
    doc = gates2.g9_ref(BODY, "T0", rec, root=tmp_path)
    assert doc == _record(rec)
    a = doc["parts"]["a"]
    written = tmp_path / "rec" / "refcheck" / "T0" / BODY / "refcheck.json"
    assert a["refcheck"]["source"] == "bar.refcheck, run by this gate" and written.is_file()
    assert a["refcheck"]["sha256"] == s2io.sha256_file(written) and a["schema"] == "bar-refcheck-1"
    assert a["same_set_and_body"] is True and set(a["items"]) >= {"walk/v0.4", "walk/nominal", "stop/two_step"}
    assert doc["parts"]["f"]["screen"] is None and doc["parts"]["f"]["unplayed"] == {}
    assert set(doc["parts_ok"]) == {"a", "b", "c", "f"} and doc["form"] == "plain"


@needs_v0
def test_g9_ref_part_b_against_a_comparand(monkeypatch):
    """The form of (b) for an ``-X`` body and for V6noF5: each walk's playback against the comparand set's playback
    of the walk of the same name within (b)'s limits (here V0 against itself: every difference is zero); the
    comparand's files are cited by path and sha256."""
    v = gates2._SetView(BODY, "V0")
    monkeypatch.setattr(v, "walks", lambda: ["nominal", "v0.4"])
    # (the comparand's files are cited by their own sha256, not by what their entries say: one entry is changed)
    v.refs = copy.deepcopy(v.refs)
    v.refs["v0.4"]["sha256"] = "0" * 64
    b = gates2.part_b(v, comparand=v)
    assert set(b["compared"]) == {"nominal", "v0.4"} and b["rungs"] == {}
    for n, x in b["compared"].items():
        d = x["against"]["differences"]
        assert x["against"]["terms"]["head"] and x["against"]["terms"]["tail"], n
        assert d["head_deg"] == 0.0 and d["tail_cycles"] == 0.0 and d["toe_off_rad"] == 0.0, n
    own = s2io.sha256_file(v.dir / "refs" / "v0.4.npz")
    assert b["comparand"]["set"] == "V0" and b["comparand"]["files"]["v0.4"]["sha256"] == own != "0" * 64
    assert b["comparand"]["files"]["v0.4"]["entry_sha256"] == "0" * 64
    assert b["comparand"]["files"]["v0.4"]["path"] == "stage2/sets/V0/75-M-H-10/refs/v0.4.npz"


V0_REFCHECK = ROOT / "stage2" / "sets" / "V0" / BODY / "refcheck.json"


def _doctored_refcheck(tmp_path, name: str, fail=None, set_name=None, file_sha=None) -> Path:
    """V0's own refcheck record with items made to fail (``fail``: ``{item: [terms]}``), another set named, or one
    file it read given another sha256 (``file_sha``)."""
    doc = s2io.read_json(V0_REFCHECK)
    for key, terms in (fail or {}).items():
        x = doc["items"][key]
        x["pass_"] = False
        for t in terms:
            x["ok"][t] = False
        doc["pass_"] = False
    if set_name is not None:
        doc["set"] = set_name
    if file_sha is not None:
        doc["inputs"]["files"][file_sha]["sha256"] = "0" * 64
    p = tmp_path / f"{name}.json"
    s2io.write_json(p, doc)
    return p


@needs_v0
def test_g9_ref_part_a_on_a_doctored_reference_check(tmp_path):
    """G9-ref (a) on V0 with its reference check doctored: a failing turn item names its move as not available and
    does not block; a failing walk/v0.4 item blocks; a record of another set, or one that read a file the latest
    freeze holds with another sha256 (given by path as with --refcheck), is not read and blocks."""
    v = gates2._SetView(BODY, "V0")
    a = gates2.part_a(v, refcheck=V0_REFCHECK)
    assert a["ok"] is True and a["blocks"] is False and a["files_match_freeze"] is True and a["not_available"] == []
    a = gates2.part_a(v, refcheck=_doctored_refcheck(tmp_path, "turn", fail={"turn/q90_left": ["heading"]}))
    assert a["ok"] is False and a["blocks"] is False and a["blocked"] == []
    assert a["not_available"] == [dict(item="turn/q90_left", reference="transitions/turn90_pi4_left.npz",
                                       failing=["heading"])]
    assert "treated as not available" in a["followed"]
    a = gates2.part_a(v, refcheck=_doctored_refcheck(tmp_path, "walk", fail={"walk/v0.4": []}))
    assert a["ok"] is False and a["blocks"] is True and a["blocked"] == ["walk/v0.4"]
    assert a["not_available"] == [dict(item="walk/v0.4", reference="refs/v0.4.npz", failing=[])]
    assert a["followed"].startswith("blocked")
    for name, kw in (("other_set", dict(set_name="V1")), ("other_file", dict(file_sha="refs/v0.4.npz"))):
        a = gates2.part_a(v, refcheck=_doctored_refcheck(tmp_path, name, **kw))
        assert a["ok"] is False and a["blocks"] is True and "(a) is not read" in a["followed"], name
    assert a["files_match_freeze"] is False and a["refcheck"]["source"] == "given"


@needs_v0
def test_g9_ref_passes_only_when_every_part_does(tmp_path, monkeypatch):
    """The whole gate on V0 with parts (b), (c) and (f) stood in: it passes (exit 0) only when (a), (b), (c) and
    (f) all pass; a blocking (a) fails it.  A failing reference that does not block is listed as not available and
    the gate can pass: the builder's reading, which this fix round lists for the lead (it is not changed here)."""
    outcome = dict(b=True, c=True, f=True)
    monkeypatch.setattr(gates2, "part_b", lambda v, comparand=None: dict(ok=outcome["b"], items={}, walks={}))
    monkeypatch.setattr(gates2, "part_c", lambda v: dict(ok=outcome["c"], walks={}))
    monkeypatch.setattr(gates2, "part_f", lambda v, refcheck_doc=None: dict(ok=outcome["f"], tasks={}, unplayed={}))
    rec = tmp_path / "g9_ref.json"

    def run(refcheck=V0_REFCHECK) -> tuple[int, dict]:
        code = gates2.main(["g9-ref", "--body", BODY, "--set", "V0", "--record", str(rec), "--refcheck", str(refcheck)])
        return code, _record(rec)

    code, doc = run()
    assert code == 0 and doc["pass_"] is True and doc["parts_ok"] == dict(a=True, b=True, c=True, f=True)
    for part in ("b", "c", "f"):
        outcome[part] = False
        code, doc = run()
        assert code == 1 and doc["pass_"] is False and doc["parts_ok"][part] is False, part
        outcome[part] = True
    code, doc = run(_doctored_refcheck(tmp_path, "walk", fail={"walk/v0.4": ["along"]}))
    assert code == 1 and doc["pass_"] is False and doc["parts"]["a"]["blocked"] == ["walk/v0.4"]
    code, doc = run(_doctored_refcheck(tmp_path, "turn", fail={"turn/q90_left": ["heading"]}))
    assert doc["parts_ok"]["a"] is False and doc["parts"]["a"]["blocks"] is False
    assert code == 0 and doc["pass_"] is True and "treated as not available" in doc["followed"]


@needs_s1
def test_g9_ref_refuses_a_set_that_is_not_frozen_or_whose_files_are_not_the_freezes(tmp_path):
    """G9-ref reads a frozen set: on the synthetic set T0 without its freeze it refuses (exit 2), and with a walk's
    file changed after the freeze it refuses before anything is played, naming the file."""
    from tests import test_stage2_g8a as G

    d = G.make_t0(tmp_path)
    sd = s2io.read_json(d / "set.json")
    s2io.write_json(d / "set.json", dict(sd, freezes=[]))
    with pytest.raises(gates2.Refused, match="the set is not frozen"):
        gates2.g9_ref(BODY, "T0", None, root=tmp_path)
    assert gates2.main(["g9-ref", "--body", BODY, "--set", "T0", "--set-root", str(tmp_path)]) == 2
    s2io.write_json(d / "set.json", sd)
    with open(d / "refs" / "v0.4.npz", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(gates2.Refused, match="refs/v0.4.npz"):
        gates2.g9_ref(BODY, "T0", tmp_path / "rec" / "g9_ref.json", root=tmp_path)
    assert not (tmp_path / "rec").exists()


# ------------------------------------------------------------------------------------------------
# G9-run (rule Z2): synthetic companion records over a real 10 ms bar record
# ------------------------------------------------------------------------------------------------
BAR_V0 = ROOT / "stage2" / "runs" / "2026-11-15-75cm" / "V0" / BODY / "seeds3042" / "eval" / "R0@10ms" / "walk" / "v0.4.json"


def _companion_run(out: Path, bar_rec: dict) -> Path:
    """A bar record of R0L@10ms (a committed R0@10ms record's episodes) and its companion under ``out``, as
    ``lookbar.run`` and ``collect`` lay them out: the companion names the bar record by path and sha256 and holds, per
    episode, the outcome and the look reading."""
    from designlab.stage2 import bar

    bdir = bar.block_dir(out, "V0", BODY, "seeds3042")
    bpath = bdir / "eval" / "R0L@10ms" / "walk" / "v0.4.json"
    s2io.write_json(bpath, bar_rec)
    eps = []
    for e in bar_rec["episodes"]:
        own = dict(bob_pp_m=0.01, head_world_pitch_rms_deg=0.5, tail=dict(phase_cycles=0.1, amp_rad=0.12),
                   toe_off_rad=0.1, steps=20, lift_offs=20)
        eps.append(dict(condition="walk/v0.4", seed=e["seed"], outcome="fell" if e["fell"] else "read", look=dict(own=own)))
    doc = dict(schema="s2-look-1", bar_record=dict(path=os.path.relpath(bpath, out), sha256=s2io.sha256_file(bpath)),
               set="V0", body=BODY, row="R0L@10ms", condition="walk/v0.4", judge="bar_walk", episodes=eps)
    cpath = bdir / "look" / "R0L@10ms" / "walk" / "v0.4.json"
    gates2.finish(doc, cpath)
    return cpath


@pytest.mark.skipif(not BAR_V0.exists(), reason="needs the committed 75 cm round's records")
def test_g9_run_part_d_and_e(tmp_path, monkeypatch):
    rec = s2io.read_json(BAR_V0)
    out = tmp_path / "bar"
    cpath = _companion_run(out, rec)
    monkeypatch.setattr(gates2, "_expected_conditions", lambda body, set_name, root=None: ["walk/v0.4"])
    d = gates2.part_d(out, BODY, "V0", "seeds3042")
    assert d["ok"] and d["records"]["walk/v0.4"]["ok"] and d["missing"] == []
    (b10,) = d["bar_records_10ms"].values()
    assert b10["ok"] and b10["reached_window"] == 20 and b10["missing_toe_readings"] == []
    # (e) applies to the 60 cm body and head only, and needs round A's reread record
    assert gates2.part_e(BODY)["applies"] is False
    assert gates2.part_e("60-N-T-10")["ok"] is False and "no reread record" in gates2.part_e("60-N-T-10-X")["reason"]
    rr = tmp_path / "reread.json"
    good = dict(schema="bar-reread-1", body="60-N-T-10", complete=True, pass_=True, traces=40, traces_equal=40,
                rows=["R2s/42@10ms", "R2@10ms"], conditions=["turn/walking_0.35_left", "turn/in_place_0.50_right"])
    s2io.write_json(rr, good)
    e = gates2.part_e("60-N-T-10-X", rr)
    assert e["ok"] and e["record"]["sha256"] == s2io.sha256_file(rr) and e["of_body"] == "60-N-T-10"
    assert all(e["checks"].values()) and set(e["checks"]) >= {"rows_10ms", "turn_conditions"}
    # what fails (e): a failed re-read; rows that are not all 10 ms rows; conditions that are not all turns; another
    # body; a record that is not complete
    for change, check in ((dict(pass_=False, traces_equal=39), "traces_equal"),
                          (dict(rows=["R2@10ms", "R2@0ms"]), "rows_10ms"), (dict(rows=[]), "rows_10ms"),
                          (dict(conditions=["turn/walking_0.35_left", "walk/nominal"]), "turn_conditions"),
                          (dict(conditions=None), "turn_conditions"), (dict(body="75-M-H-10"), "body"),
                          (dict(complete=False), "complete")):
        s2io.write_json(rr, dict(good, **change))
        e = gates2.part_e("60-N-T-10", rr)
        assert e["ok"] is False and [k for k, v in e["checks"].items() if not v] == [check], change
    # the whole record
    g = gates2.g9_run(BODY, "V0", out, None, tmp_path / "g9_run.json")
    assert g == _record(tmp_path / "g9_run.json") and g["pass_"] and g["schema"] == "s2-gate-g9-run-1"
    assert gates2.main(["g9-run", "--body", BODY, "--set", "V0", "--bar-out", str(out)]) == 0
    # what fails (d): an episode that is an error; a look number that is not finite; a changed bar record; a
    # missing toe-clearance reading; a missing record
    comp = s2io.read_json(cpath)
    bad = copy.deepcopy(comp)
    bad["episodes"][3]["outcome"] = "error"
    gates2.finish(bad, cpath)
    d = gates2.part_d(out, BODY, "V0", "seeds3042")
    assert not d["ok"] and "seed 3045: error" in d["records"]["walk/v0.4"]["problems"][0]
    bad = copy.deepcopy(comp)
    bad["episodes"][0]["look"]["own"]["bob_pp_m"] = float("nan")
    gates2.finish(bad, cpath)
    assert "look reading is not finite" in gates2.part_d(out, BODY, "V0", "seeds3042")["records"]["walk/v0.4"]["problems"][0]
    gates2.finish(comp, cpath)
    edited = copy.deepcopy(rec)
    del edited["episodes"][2]["measures"]["bar"]["stage1_toe_clearance_m"]
    edited["episodes"][5]["success"].pop("duty")
    s2io.write_json(out / comp["bar_record"]["path"], edited)
    d = gates2.part_d(out, BODY, "V0", "seeds3042")
    probs = d["records"]["walk/v0.4"]["problems"]
    assert not d["ok"] and "sha256 is not the one it names" in probs[0] and "seed 3047: its readings" in probs[1]
    assert list(d["bar_records_10ms"].values())[0]["missing_toe_readings"] == [3044]
    # an expected record that is absent fails (d) by itself: every record there is sound
    s2io.write_json(out / comp["bar_record"]["path"], rec)
    assert gates2.part_d(out, BODY, "V0", "seeds3042")["ok"] is True
    monkeypatch.setattr(gates2, "_expected_conditions", lambda body, set_name, root=None: ["walk/v0.4", "turn/q90_left"])
    d = gates2.part_d(out, BODY, "V0", "seeds3042")
    assert d["missing"] == ["turn/q90_left"] and d["ok"] is False
    assert d["records"]["walk/v0.4"]["ok"] and all(x["ok"] for x in d["bar_records_10ms"].values()) and not d["problems"]


@needs_v0
def test_g9_run_expects_every_gated_condition_and_walk_nominal():
    keys = gates2._expected_conditions(BODY, "V0")
    assert len(keys) == 17 and keys[-1] == "walk/nominal" and "walk/v0.4_mu0.54" in keys
    assert gates2._expected_conditions(BODY, "S1") == []


# ------------------------------------------------------------------------------------------------
# Rates and memory; kill and resume (the trainer's stub env)
# ------------------------------------------------------------------------------------------------
def _stub():
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from tests import test_stage2_train2 as T

    return T


def test_rates_record_on_a_stub_env_with_2_workers(tmp_path, monkeypatch):
    T = _stub()
    T.write_set(tmp_path / "set.json")
    monkeypatch.setenv("STUB_SET_FILE", str(tmp_path / "set.json"))
    monkeypatch.chdir(ROOT)  # (the workers import the stub env as tests.test_stage2_train2)
    cfgs = T.stub_configs("R5", "stub", "STUB", None)
    monkeypatch.setattr(gates2, "_rate_specs", lambda body, set_name: (cfgs, T.stub_specs("R5", cfgs)))
    calls = []

    def throughput(body, set_name, floor, steps):
        calls.append((body, set_name, floor, steps))
        return dict(floor=floor, env_steps=steps, env_steps_per_wall_s=1234.0, rss_mb=100.0)

    monkeypatch.setattr(gates2, "_throughput", throughput)
    rec = tmp_path / "rates.json"
    doc = gates2.rates("stub", "STUB", floors=("hard", "robot_floor_carpet"), workers=(2,), steps=256, record=rec,
                       n_steps=16)
    assert doc == _record(rec) and doc["schema"] == "s2-val-rates-1" and doc["pass_"] is True
    assert calls == [("stub", "STUB", "hard", 256), ("stub", "STUB", "robot_floor_carpet", 256)]
    (r,) = doc["pool"]
    assert r["workers"] == 2 and r["n_envs"] == 8 and r["env_steps"] >= 256 and r["env_steps"] % 128 == 0
    assert r["rollouts"] == 2 and r["env_steps_per_wall_s"] > 0 and r["build_s"] > 0
    assert len(r["worker_pids"]) == 2 and len(r["rss_mb"]["workers"]) == 2 and r["rss_mb"]["parent"] > 0
    assert all(x is None or x > 0 for x in r["rss_mb"]["workers"])  # (a worker that has ended reads None)
    assert r["cpu_s_parent"] > 0 and r["cpu_s_workers"] > 0
    assert doc["provisional"] is (not doc["load_average_before"][0] < 0.5)
    assert doc["env_floors"] == [c["floor"] for c in cfgs] and doc["throughput"]["hard"]["env_steps"] == 256


STUB_K3 = dict(steps=1536, checkpoint_every=256, monitor_every=512, kill_at=896, workers=(2, 3), tiny=True,
               n_steps=16, ppo=dict(n_steps=16, batch_size=64))
STUB_CHILD = ("import sys; from tests import test_stage2_train2 as t; t.install(); "
              "from designlab.stage2 import gates2; sys.exit(gates2.main(['_train', 'designlab.stage2.train2', sys.argv[1]]))")


def _stub_child(kw: dict) -> list:
    return [PY, "-c", STUB_CHILD, json.dumps(kw)]


@pytest.fixture(scope="module")
def stub_kr(tmp_path_factory):
    """Rule K3's two runs on the stub (rollouts of 8 x 16 steps, a checkpoint every 256 steps, 1,536 in all): the
    second killed (the whole process group, SIGKILL) when its progress reads 896, resumed with 3 workers.  The
    settings asked are read with the stub's validation records (none), as its children read them."""
    T = _stub()
    from designlab.stage2 import train2

    tmp = tmp_path_factory.mktemp("kr")
    T.write_set(tmp / "set.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        mp.setattr(train2, "_validation", T.stub_validation)
        doc = gates2.kill_resume("stub", "STUB", tmp / "out", tmp / "kill_resume.json", settings=STUB_K3,
                                 child_cmd=_stub_child, child_env=dict(STUB_SET_FILE=str(tmp / "set.json")), hold=True)
    return tmp, doc


@pytest.fixture
def stub_seams(monkeypatch):
    """The stub's validation records (none) in this process, and the study as the working directory."""
    T = _stub()
    from designlab.stage2 import train2

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(train2, "_validation", T.stub_validation)
    return T


def test_kill_resume_on_the_stub_env(stub_kr):
    tmp, doc = stub_kr
    assert doc == _record(tmp / "kill_resume.json") and doc["schema"] == "s2-val-kill-resume-1"
    assert doc["pass_"] is True and doc["equal"] is True and doc["first_difference"] is None
    assert all(doc["settings_ok"].values()) and doc["form"] == gates2.K3_CHECK
    k = doc["kill"]
    assert k["progress_steps_read"] == 896 and k["as_ruled"] and k["exit_code"] == -9 and k["group_gone"]
    assert k["whole_group"] and k["signal"] == "SIGKILL"
    assert doc["resumed_from"] == 768 and doc["checkpoints_compared"] == [1024, 1280, 1536] and doc["final_steps"] == 1536
    ru, rk = doc["runs"]["uninterrupted"], doc["runs"]["killed"]
    assert [s["workers"] for s in ru["sessions"]] == [2] and [s["workers"] for s in rk["sessions"]] == [2, 3]
    assert rk["sessions"][1]["resumed_from"] == 768 and rk["sessions"][1]["fingerprint_max_abs_diff"] == 0.0
    assert ru["run_uuid"] != rk["run_uuid"] and ru["status"] == rk["status"] == "done"
    assert ru["filter_hz"] == rk["filter_hz"] == [10.0] and ru["lr"] == rk["lr"]
    final = doc["compared"][-1]
    assert final["steps"] == 1536 and final["equal"] and final["intact"] == {"a": None, "b": None}
    kinds = final["by_kind"]
    assert set(kinds) == {"model", "vecnormalize", "fingerprint"} and all(v["equal"] for v in kinds.values())
    names = [r["name"] for r in final["compared"]]
    assert any(n.startswith("model/policy.pth/") for n in names)
    assert any(n.startswith("model/policy.optimizer.pth/state/") for n in names)
    for n in ("vecnormalize/obs_rms/count", "vecnormalize/ret_rms/count", "vecnormalize/obs_rms/mean",
              "fingerprint/actions", "fingerprint/obs_count"):
        assert n in names, n
    # (the archives' bytes need not be equal: what is compared is what they hold)
    assert final["files_identical"]["fingerprint"] is True
    # the record's provenance and the logs of the three children
    for key in PROVENANCE:
        assert key in doc, key
    assert {p.name for p in (tmp / "out").glob("*.log")} == {"uninterrupted.log", "killed.log", "resumed.log"}


def test_kill_resume_is_resumable(stub_kr, stub_seams):
    """A second call reuses both finished runs (nothing is run again) and gives the same comparison; a killed run
    whose kill was not as ruled is moved aside and made again."""
    tmp, first = stub_kr
    logs = {p.name: p.stat().st_size for p in (tmp / "out").glob("*.log")}
    doc = gates2.kill_resume("stub", "STUB", tmp / "out", None, settings=STUB_K3, child_cmd=_stub_child,
                             child_env=dict(STUB_SET_FILE=str(tmp / "set.json")), hold=True)
    assert doc["made"] == {} and doc["equal"] and doc["compared"] == first["compared"]
    assert {p.name: p.stat().st_size for p in (tmp / "out").glob("*.log")} == logs


def test_the_difference_report_names_the_first_tensor_that_differs(stub_kr, tmp_path):
    """A perturbed policy tensor and a perturbed fingerprint action: ``compare_items`` names the first difference
    in name order with its index and both values; ``compare_checkpoints`` refuses a checkpoint that is not intact
    (its file no longer has its record's sha256) before comparing anything."""
    import torch

    tmp, _ = stub_kr
    a = tmp / "out" / "uninterrupted" / "jobs" / "R5-stub-s42"
    b = tmp_path / "R5-stub-s42"
    shutil.copytree(tmp / "out" / "killed" / "jobs" / "R5-stub-s42", b)
    model = b / "checkpoints" / "model_1536.zip"
    with zipfile.ZipFile(model) as z:
        members = {n: z.read(n) for n in z.namelist()}
    state = torch.load(io.BytesIO(members["policy.pth"]), map_location="cpu", weights_only=True)
    key = sorted(k for k in state if state[k].ndim == 2)[0]
    state[key][1, 2] += 1e-6
    buf = io.BytesIO()
    torch.save(state, buf)
    members["policy.pth"] = buf.getvalue()
    with zipfile.ZipFile(model, "w") as z:
        for n, data in members.items():
            z.writestr(n, data)
    x, y = gates2.checkpoint_items(a, 1536), gates2.checkpoint_items(b, 1536)
    c = gates2.compare_items(x, y)
    f = c["first_difference"]
    assert c["equal"] is False and f["name"] == f"model/policy.pth/{key}" and f["index"] == [1, 2]
    assert f["differing"] == 1 and f["b"] - f["a"] == pytest.approx(1e-6, rel=1e-2) and f["max_abs_diff"] > 0
    assert c["by_kind"]["model"]["equal"] is False and c["by_kind"]["fingerprint"]["equal"] is True
    y2 = dict(y)
    y2["fingerprint/actions"] = y["fingerprint/actions"].copy()
    y2["fingerprint/actions"][7, 1] += 1e-7
    c2 = gates2.compare_items(x, dict(y2, **{f"model/policy.pth/{key}": x[f"model/policy.pth/{key}"]}))
    assert c2["first_difference"]["name"] == "fingerprint/actions" and c2["first_difference"]["index"] == [7, 1]
    assert gates2.compare_items(x, {k: v for k, v in x.items() if k != "fingerprint/obs"})["first_difference"] == dict(
        name="fingerprint/obs", missing_in="b")
    cc = gates2.compare_checkpoints(a, b, 1536)
    assert cc["equal"] is False and cc["intact"]["b"] and cc["first_difference"]["name"] == "integrity"


def test_a_kill_that_was_not_as_ruled_is_made_again(stub_kr, stub_seams, tmp_path):
    tmp, _ = stub_kr
    out = tmp_path / "out"
    shutil.copytree(tmp / "out", out)
    marker = out / "killed" / "KILLED.json"
    s2io.write_json(marker, dict(s2io.read_json(marker), as_ruled=False, progress_steps_read=1024))
    doc = gates2.kill_resume("stub", "STUB", out, None, settings=STUB_K3, child_cmd=_stub_child,
                             child_env=dict(STUB_SET_FILE=str(tmp / "set.json")), hold=True)
    assert doc["made"]["killed_discarded"].endswith("killed.discarded.0") and (out / "killed.discarded.0").is_dir()
    assert doc["pass_"] and doc["kill"]["progress_steps_read"] == 896 and doc["equal"]


def test_the_comparison_has_tolerance_zero():
    """Rule K3: equal is tolerance zero.  One ulp in a float32 or a float64 array is a difference, named with its
    index and both values; so is the same value in another dtype; NaN equals NaN in the same place."""
    for dtype in (np.float32, np.float64):
        a = np.array([[0.1, 0.2], [0.3, 1e3]], dtype)
        b = a.copy()
        b[1, 1] = np.nextafter(b[1, 1], dtype(np.inf))
        assert not gates2._same(a, b) and gates2._same(a, a.copy())
        c = gates2.compare_items({"model/w": a, "fingerprint/actions": a}, {"model/w": a.copy(), "fingerprint/actions": b})
        f = c["first_difference"]
        assert c["equal"] is False and f["name"] == "fingerprint/actions" and f["index"] == [1, 1] and f["differing"] == 1
        assert f["a"] == float(a[1, 1]) and f["b"] == float(b[1, 1]) and f["b"] != f["a"]
        assert c["by_kind"] == {"fingerprint": dict(items=1, equal=False), "model": dict(items=1, equal=True)}
    x = np.array([1.0, np.nan], np.float32)
    assert gates2._same(x, x.copy()) and not gates2._same(x, x.astype(np.float64))
    assert not gates2._same(np.array([1, 2]), np.array([1, 3])) and gates2._same(np.array(5), np.array(5))


#: A killed run's stand-in whose first progress reading is past the kill step (1,024 > 896): it writes that reading
#: and waits to be killed.
PAST_CHILD = ("import json, sys, time; from pathlib import Path; kw = json.loads(sys.argv[1]); "
              "p = Path(kw['out']) / 'jobs' / 'R5-stub-s42'; p.mkdir(parents=True, exist_ok=True); "
              "(p / 'progress.json').write_text(json.dumps(dict(steps=1024))); time.sleep(60)")


def test_a_kill_whose_first_progress_reading_is_past_the_kill_step_is_no_kill_as_ruled(stub_kr, stub_seams, tmp_path):
    """Rule K3's kill is at 81,920 steps (here 896): a first reading past it (1,024) is killed but is no kill as
    ruled; the marker says so, the gate stops (an error, not a pass), and the next call makes that run again."""
    tmp, _ = stub_kr
    out = tmp_path / "out"
    shutil.copytree(tmp / "out", out)
    shutil.rmtree(out / "killed")
    with pytest.raises(gates2.GateError, match="no kill as ruled"):
        gates2.kill_resume("stub", "STUB", out, None, settings=STUB_K3,
                           child_cmd=lambda kw: [PY, "-c", PAST_CHILD, json.dumps(kw)])
    k = s2io.read_json(out / "killed" / "KILLED.json")
    assert k["as_ruled"] is False and k["progress_steps_read"] == 1024 and k["kill_at"] == 896
    assert k["exit_code"] == -9 and k["group_gone"]


def test_a_run_made_with_other_settings_is_not_reused(stub_kr, stub_seams, tmp_path):
    """Kill-resume reuses a run on disk only when its run.json's settings are the ones asked: another starting rate
    or filter (asked by --lr and --config) moves both runs aside and makes them again; a killed run of another env
    count, the uninterrupted one being right, is moved aside alone.  (The stand-in child exits 7 at once: only what
    the gate does before a run is looked at.)"""
    tmp, first = stub_kr
    started = []

    def child(kw):
        started.append(kw)
        return [PY, "-c", "import sys; sys.exit(7)"]

    out = tmp_path / "a"
    shutil.copytree(tmp / "out", out)
    with pytest.raises(gates2.GateError, match="uninterrupted run ended with exit 7"):
        gates2.kill_resume("stub", "STUB", out, None, settings=STUB_K3, lr=1e-3, config={"filter_hz": 20.0},
                           child_cmd=child)
    assert (out / "uninterrupted.discarded.0" / "jobs" / "R5-stub-s42" / "run.json").is_file()
    assert started[0]["lr"] == 1e-3 and started[0]["config"] == {"filter_hz": 20.0}
    out = tmp_path / "b"
    shutil.copytree(tmp / "out", out)
    p = out / "killed" / "jobs" / "R5-stub-s42" / "run.json"
    s2io.write_json(p, dict(s2io.read_json(p), settings=dict(s2io.read_json(p)["settings"], n_envs=4)))
    started.clear()
    with pytest.raises(gates2.GateError, match="killed run ended .exit 7. before its progress read 896"):
        gates2.kill_resume("stub", "STUB", out, None, settings=STUB_K3, child_cmd=child)
    assert len(started) == 1 and started[0]["out"] == str(out / "killed") and "resume" not in started[0]
    assert (out / "killed.discarded.0" / "KILLED.json").is_file() and not (out / "uninterrupted.discarded.0").exists()
    # the record of the first call holds what was asked and that both runs' settings were it
    assert first["settings_problems"] == {"uninterrupted": [], "killed": []} and first["settings_asked"]["n_envs"] == 8
    assert first["settings_ok"]["same_settings"] and first["settings_ok"]["killed_settings_as_asked"]


def test_what_a_runs_settings_are_compared_with(stub_kr, stub_seams):
    """``run_settings_problems`` names every setting of rule K3's list that is not the one asked, from the run's
    run.json as the trainer wrote it; the trainer's own env count and rollout length are refused before any run when
    they are not the settings'."""
    tmp, _ = stub_kr
    doc = s2io.read_json(tmp / "out" / "uninterrupted" / "jobs" / "R5-stub-s42" / "run.json")
    asked = gates2.k3_asked(dict(gates2.K3_TINY, **STUB_K3), "stub", "STUB")
    assert gates2.run_settings_problems(doc, asked) == []
    assert asked["lr_start"] == 3e-5 and asked["filter_hz"] is None and asked["budget"] == 1536
    for key, value in (("seed", 43), ("set", "V6"), ("body", "75-M-H-10"), ("n_envs", 4), ("n_steps", 32),
                       ("checkpoint_every", 512), ("monitor_every", 1024), ("budget", 2048), ("gates", True),
                       ("tiny", False), ("config", {"filter_hz": 20.0}), ("ppo_overrides", {})):
        (why,) = gates2.run_settings_problems(doc, dict(asked, **{key: value}))
        assert why.startswith(f"{key}: "), (key, why)
    (why,) = gates2.run_settings_problems(doc, dict(asked, lr_start=1e-3))
    assert why == "lr_start: 3e-05 (asked 0.001)"
    (why,) = gates2.run_settings_problems(doc, dict(asked, filter_hz=20.0))
    assert why == "filter_hz: [10.0] (asked 20.0)"
    assert gates2.run_settings_problems(doc, dict(asked, filter_hz=10.0)) == []
    assert "no starting rate is asked" in gates2.run_settings_problems(doc, dict(asked, lr_start=None))[-1]
    with pytest.raises(gates2.Refused, match="the trainer runs 8 envs"):
        gates2.k3_asked(dict(gates2.K3_TINY, **STUB_K3, n_envs=4), "stub", "STUB")
    with pytest.raises(gates2.Refused, match="of 16"):
        gates2.k3_asked(dict(gates2.K3_TINY, **dict(STUB_K3, n_steps=64)), "stub", "STUB")
    # rule K3's own settings are the trainer's defaults
    k3 = gates2.k3_asked(gates2.K3, "75-M-H-10", "V6", lr=1e-4, config={"filter_hz": 10.0})
    assert (k3["n_envs"], k3["n_steps"], k3["budget"], k3["checkpoint_every"]) == (8, 1024, 131_072, 32_768)


def test_r6_starts_tiny_run_is_reused_only_when_it_is_this_ones(tmp_path):
    """r6-start's tiny R6 run: a finished run of another set under the work directory is moved aside and made again
    (fresh, not resumed); a finished run of this body and set is reused; a run whose trainer is alive is refused."""
    _stub()
    from designlab.stage2 import train2

    work = tmp_path / "work"
    jd = work / "jobs" / "R6-75-M-H-10-s42"
    calls = []

    def fake_train(mode, body, seed, **kw):
        calls.append(dict(kw, mode=mode, body=body, seed=seed))
        doc = dict(status="done", num_timesteps=kw["steps"], settings=dict(
            mode=mode, body=body, seed=seed, set=kw["set_name"], tiny=kw["tiny"], gates=kw["gates"],
            budget=kw["steps"]))
        s2io.write_json(jd / "run.json", doc)
        return doc

    steps = 2 * train2.N_ENVS * train2.TINY["n_steps"]
    other = dict(status="done", num_timesteps=steps, settings=dict(mode="R6", body=BODY, seed=42, set="V0", tiny=True,
                                                                   gates=False, budget=steps))
    s2io.write_json(jd / "run.json", other)
    t = gates2.r6_training(BODY, "S1", work, 2, train=fake_train)
    assert len(calls) == 1 and calls[0]["resume"] is False and calls[0]["set_name"] == "S1"
    assert t["made"]["discarded"].endswith("R6-75-M-H-10-s42.discarded.0")
    assert t["made"]["discarded_settings"] == ["set: 'V0' (asked 'S1')"] and t["settings_problems"] == []
    assert s2io.read_json(work / "jobs" / "R6-75-M-H-10-s42.discarded.0" / "run.json") == other
    calls.clear()
    t = gates2.r6_training(BODY, "S1", work, 2, train=fake_train)
    assert calls == [] and t["made"] == {} and t["status"] == "done" and t["steps"] == steps
    assert gates2.r6_run_problems(dict(settings=dict(other["settings"], budget=steps + 1)), BODY, "V0", steps) == [
        f"budget: {steps + 1} (asked {steps})"]
    p = subprocess.Popen([PY, "-c", "import time; time.sleep(60)", "designlab.stage2.train2"])
    try:
        s2io.write_json(jd / "progress.json", dict(pid=p.pid, steps=0))
        with pytest.raises(gates2.Refused, match="still running"):
            gates2.r6_training(BODY, "S1", work, 2, train=fake_train)
    finally:
        p.kill()
        p.wait()


# ------------------------------------------------------------------------------------------------
# The training start with MUJOCO_GL set against it; R6's start
# ------------------------------------------------------------------------------------------------
def test_gl_start(tmp_path):
    pytest.importorskip("torch")
    doc = gates2.gl_start(tmp_path, tmp_path / "gl_start.json")
    assert doc == _record(tmp_path / "gl_start.json") and doc["schema"] == "s2-val-gl-start-1"
    assert doc["pass_"] and all(doc["checks"].values()) and doc["exit_code"] == 0
    assert doc["started_with"] == dict(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    assert doc["preflight"]["mujoco_gl"] == "disable" and doc["preflight"]["worker"]["mujoco_gl"] == "disable"
    assert (tmp_path / "gl_start.log").is_file()


@slow
@needs_s1
def test_r6_start_on_the_s1_view(tmp_path):
    pytest.importorskip("torch")
    doc = gates2.r6_start(BODY, "S1", tmp_path / "r6_start.json", out=tmp_path / "work", workers=2)
    assert doc == _record(tmp_path / "r6_start.json") and doc["pass_"], doc["checks"]
    assert doc["zero_action"]["steps"] == doc["random_actions"]["steps"] == 200
    assert doc["mid_start"]["history_is_reference"] and doc["mid_start"]["targets_are_reference"]
    t = doc["training"]
    assert t["status"] == "done" and t["steps"] == t["budget"] == 1024 and t["updates"] == 2
    assert t["fingerprint_max_abs_diff"] == 0.0
    # resumable: a second call reuses the finished tiny run
    again = gates2.r6_start(BODY, "S1", None, out=tmp_path / "work", workers=2)
    assert again["training"]["steps"] == 1024 and again["pass_"]


# ------------------------------------------------------------------------------------------------
# The chain
# ------------------------------------------------------------------------------------------------
def _chain(tmp_path, **extra) -> subprocess.CompletedProcess:
    env = _env(TB_WORK=str(tmp_path / "work"), TB_DEV=str(tmp_path / "dev.json"), **extra)
    return subprocess.run(["sh", str(CHAIN)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def _steps(text: str) -> list[str]:
    return [line[1:line.index("]")] for line in text.splitlines() if line.startswith("[")]


ORDER = ("V1-", "V2-", "V3-", "V5-", "V6-", "V7-", "V4-", "V8-", "V9-", "V10-")


def test_the_chain_parses_and_its_dry_run_prints_the_plan_in_order(tmp_path):
    assert subprocess.run(["sh", "-n", str(CHAIN)], capture_output=True).returncode == 0
    r = _chain(tmp_path, TB_DRY="1")
    assert r.returncode == 0, r.stderr
    steps = _steps(r.stdout)
    assert len(steps) == len(set(steps)) == 46
    # rule M3: S2-4 starts only after the check that its baseline is committed, which follows the baseline's copy
    i = steps.index("V8-baseline-committed")
    assert steps[i - 2:i + 2] == ["V8-baseline", "V8-baseline-copy", "V8-baseline-committed", "V8-s2-4"]
    firsts = [next(i for i, s in enumerate(steps) if s.startswith(p)) for p in ORDER]
    lasts = [max(i for i, s in enumerate(steps) if s.startswith(p)) for p in ORDER]
    assert firsts == sorted(firsts) and all(lasts[i] < firsts[i + 1] for i in range(len(ORDER) - 1))
    text = r.stdout
    for b in ("75-M-H-10", "75-M-H-10-X", "60-N-T-10", "60-N-T-10-X"):
        assert f"g9-ref --body {b} --set V6 --record stage2/tb/gates/V6/{b}/g9_ref.json" in text
        assert f"g8a --body {b} --set V6 --record stage2/tb/gates/V6/{b}/g8a.json" in text
    assert "r6-start --body 75-M-H-10 --set S1" in text and "kill-resume --body 75-M-H-10 --set V6" in text
    assert "rates --body 75-M-H-10 --set V6 --record stage2/tb/validation/rates.json" in text
    # rule X1: the probes, kill-resume and the rates are listed; the gates' own steps are not
    listed = [line.split()[2] for line in text.splitlines() if line.startswith("  (rule X1:")]
    assert sorted(set(listed)) == ["V4-rates", "V5-filter", "V5-sigma-probe", "V6-lr-probe", "V7-kill-resume"]
    assert "nothing was run or written" in text
    # nothing was written: no marker, no log, no list
    assert not (tmp_path / "work").exists() and not (tmp_path / "dev.json").exists()
    # with the reread record named, G9-run of the 60 cm body and head cites it
    r = _chain(tmp_path, TB_DRY="1", TB_REREAD="x/reread.json")
    assert r.stdout.count("--stage1b-reread x/reread.json") == 2


def test_a_steps_marker_makes_a_second_run_a_no_op(tmp_path):
    done = tmp_path / "work" / "done"
    done.mkdir(parents=True)
    r = _chain(tmp_path, TB_DRY="1")
    steps = _steps(r.stdout)
    for name in steps[:10] + ["V7-kill-resume", "V7-kill-resume.x1"]:
        (done / name).touch()
    r2 = _chain(tmp_path, TB_DRY="1")
    left = _steps(r2.stdout)
    assert left == [s for s in steps if s not in steps[:10] + ["V7-kill-resume"]]
    assert "skip V1-g9ref-V6-75-M-H-10 (done)" in r2.stdout and "skip V7-kill-resume (done)" in r2.stdout
    assert "V7-kill-resume in" not in r2.stdout  # (a done step is not listed again)
    # a listed step that ran but whose entry was not marked done: it is marked done, and not run or listed again
    (done / "V7-kill-resume.x1").unlink()
    r2b = _chain(tmp_path, TB_DRY="1")
    assert _steps(r2b.stdout) == left and "skip V7-kill-resume (done)" in r2b.stdout
    assert r2b.stdout.count("V7-kill-resume in") == 1
    assert "V7-kill-resume in " + str(tmp_path / "dev.json") + " as done" in r2b.stdout
    devsteps = ("V4-rates", "V5-filter", "V5-sigma-probe", "V6-lr-probe", "V7-kill-resume")
    for name in steps + [f"{d}.x1" for d in devsteps]:
        (done / name).touch()
    r3 = _chain(tmp_path, TB_DRY="1")
    assert r3.returncode == 0 and _steps(r3.stdout) == [] and r3.stdout.count("skip ") == len(steps)
    assert "(rule X1:" not in r3.stdout
    # a real run with every marker present runs nothing and exits 0
    r4 = _chain(tmp_path, PY="/nonexistent/python")
    assert r4.returncode == 0 and "all steps of the chain are done" in r4.stdout


def _fake_py(tmp_path) -> tuple[Path, Path]:
    """A python that records its arguments, one call per line, and exits 0."""
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "py.sh"
    fake.write_text(f'#!/bin/sh\necho "$*" >> "{calls}"\nexit 0\n')
    fake.chmod(0o755)
    return fake, calls


def test_a_listed_step_whose_entry_was_not_marked_done_is_marked_on_the_next_run(tmp_path):
    """Rule X1's second listing failed after kill-resume ran (its step's marker is there, the listing's is not): the
    next real run lists it as done once, writes the second marker, and does not run kill-resume again."""
    fake, calls = _fake_py(tmp_path)
    steps = _steps(_chain(tmp_path, TB_DRY="1").stdout)
    done = tmp_path / "work" / "done"
    done.mkdir(parents=True)
    devsteps = ("V4-rates", "V5-filter", "V5-sigma-probe", "V6-lr-probe")
    for name in steps + [f"{d}.x1" for d in devsteps]:
        (done / name).touch()
    r = _chain(tmp_path, PY=str(fake))
    assert r.returncode == 0, r.stdout + r.stderr
    (line,) = calls.read_text().splitlines()
    assert "gates2 dev-episode" in line and "--state done --step V7-kill-resume" in line and "--set V6" in line
    assert (done / "V7-kill-resume.x1").exists()
    r = _chain(tmp_path, PY=str(fake))
    assert r.returncode == 0 and len(calls.read_text().splitlines()) == 1


def test_s2_4_waits_until_rule_m3s_baseline_is_committed(tmp_path):
    """Rule M3: R0L's baseline is committed before S2-4 starts.  In a copy of the chain inside a scratch git
    repository, with every step before the check done: the chain stops before S2-4 (exit 1, a message for the lead,
    no marker) while the baseline is absent, untracked, only staged, committed but changed since, or committed with
    other bytes than the baseline made; once it is committed with those bytes the check passes and S2-4 starts."""
    repo = tmp_path / "repo"
    study = repo / "hardware" / "trex_form_study"
    (study / "stage2" / "tb").mkdir(parents=True)
    (study / "tools").mkdir()
    chain = study / "stage2" / "tb" / "run_trackb.sh"
    shutil.copyfile(CHAIN, chain)
    shutil.copyfile(ROOT / "tools" / "lowprio.sh", study / "tools" / "lowprio.sh")

    def git(*a):
        subprocess.run(["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "-c",
                        "commit.gpgsign=false", *a], cwd=study, check=True, capture_output=True, timeout=60)

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "the chain")
    fake, calls = _fake_py(tmp_path)
    work = tmp_path / "work"
    env = _env(PY=str(fake), TB_WORK=str(work), TB_DEV=str(tmp_path / "dev.json"))

    def run(**extra):
        return subprocess.run(["sh", str(chain)], cwd=tmp_path, env=dict(env, **extra), capture_output=True, text=True,
                              timeout=120)

    steps = _steps(run(TB_DRY="1").stdout)
    done = work / "done"
    done.mkdir(parents=True)
    for name in steps[:steps.index("V8-s2-4")]:
        if name != "V8-baseline-committed":
            (done / name).touch()
            (done / f"{name}.x1").touch()
    made = work / "s2-4-baseline" / "r0l.json"
    made.parent.mkdir(parents=True)
    made.write_text('{"episodes": 18}\n')
    bl = study / "stage2" / "tb" / "runs" / "s2-4" / "R0L_BASELINE.json"
    bl.parent.mkdir(parents=True)

    def stops(why):
        r = run()
        assert r.returncode == 1, (why, r.stdout, r.stderr)
        assert "STOP V8-baseline-committed: rule M3" in r.stdout and "goes to the lead" in r.stdout, why
        assert not (done / "V8-baseline-committed").exists(), why
        assert not calls.exists() or "designlab.stage2.train2 run" not in calls.read_text(), why

    stops("absent")
    shutil.copyfile(made, bl)
    stops("untracked")
    git("add", str(bl))
    stops("staged, not committed")
    bl.write_text('{"episodes": 17}\n')
    git("add", str(bl))
    git("commit", "-q", "-m", "another baseline")
    shutil.copyfile(made, bl)
    stops("committed with other bytes, changed since to the baseline's")
    git("checkout", "--", str(bl))
    stops("committed with other bytes than the baseline made")
    shutil.copyfile(made, bl)
    git("commit", "-q", "-am", "the baseline")
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "done V8-baseline-committed (stage2/tb/runs/s2-4/R0L_BASELINE.json is committed: " in r.stdout
    assert (done / "V8-baseline-committed").exists() and "designlab.stage2.train2 run --mode R5" in calls.read_text()


def test_a_failing_step_stops_the_chain_and_leaves_no_marker(tmp_path):
    """A real (not dry) run with a python that fails: the first step's exit code stops the chain, its log holds the
    command and the exit code, and no marker is written."""
    fake = tmp_path / "py.sh"
    fake.write_text("#!/bin/sh\nexit 3\n")
    fake.chmod(0o755)
    r = _chain(tmp_path, PY=str(fake))
    assert r.returncode == 3 and "FAILED V1-g9ref-V6-75-M-H-10 (exit 3)" in r.stdout
    log = (tmp_path / "work" / "logs" / "V1-g9ref-V6-75-M-H-10.log").read_text()
    assert "g9-ref --body 75-M-H-10 --set V6" in log and "exit 3" in log
    assert not any((tmp_path / "work" / "done").iterdir())


def test_every_output_path_of_the_chain_is_ignored_or_a_record():
    """``git check-ignore``: the chain's live paths are ignored; the records it writes (gates, validation, the
    collected runs) are not (the lead commits them)."""
    live = ["stage2/tb/work/done/V1-x", "stage2/tb/work/logs/V1-x.log", "stage2/tb/work/kill_resume/killed/KILLED.json",
            "stage2/tb/work/kill_resume/uninterrupted/jobs/R5-75-M-H-10-s42/run.json",
            "stage2/tb/work/r6_start/jobs/R6-75-M-H-10-s42/checkpoints/model_1024.zip", "stage2/tb/work/gl_start/gl_start.log",
            "stage2/tb/gates/V6/75-M-H-10/refcheck/V6/75-M-H-10/refcheck.json.tmp"]
    records = ["stage2/tb/gates/V6/75-M-H-10/g9_ref.json", "stage2/tb/gates/V6/75-M-H-10/g8a.json",
               "stage2/tb/validation/rates.json", "stage2/tb/validation/kill_resume.json", "stage2/tb/dev_episodes.json"]
    r = subprocess.run(["git", "check-ignore", *live], cwd=ROOT, capture_output=True, text=True)
    assert sorted(r.stdout.split()) == sorted(live)
    r = subprocess.run(["git", "check-ignore", *records], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout == ""
