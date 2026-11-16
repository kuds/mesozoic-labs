"""Gate G8a's comparison (``designlab/stage2/g8a.py``; round C design 3.7; rule Z1).

``compare`` is tested on tables.  The closed-loop tests run the stock form on the S1 view (Stage 1's own run
directory) and the bar form on a synthetic set ``T0`` that a fixture builds in a temporary directory from Stage
1's arrays: no Stage 2 reference set is touched (rule X1).
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import panels, plant, tune  # noqa: E402
from designlab.stage2 import g8a, r5lib, s2io, setio  # noqa: E402

BODY = "75-M-H-10"
RUN = ROOT / "stage1" / "runs" / "2026-11-08"
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)
HAVE = (RUN / BODY / "transitions" / "stop.npz").exists()
try:
    HAVE = HAVE and all(Path(f).exists() for f in plant.model_files(BODY))
except Exception:  # noqa: BLE001  (no model record: the same skip)
    HAVE = False
needs_run = pytest.mark.skipif(not (HAVE and ONE_THREAD), reason=(
    "needs the Stage 1 arrays and the round-two models (tools/restore_inputs.py) and one BLAS thread "
    "(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)"))


# ------------------------------------------------------------------------------------------------
# The rule, on tables
# ------------------------------------------------------------------------------------------------
def _d(n: int, tag: str = "") -> list:
    return [f"{tag}{i}".encode() for i in range(n)]


def test_compare_rule():
    # the row ran its 1,013 steps and did not fall; the env ran 102 env steps and did not terminate
    c = g8a.compare(_d(1013), None, _d(1020), None)
    assert c == dict(reference_steps=1013, env_physics_steps=1020, compared_steps=1013, first_difference=None,
                     reference_fell_step=None, env_terminated_step=None, ends_alike=True, identical=True)
    # the env's last, partial step (up to 9 physics steps past the row's end) is not read: a termination there is none
    assert g8a.compare(_d(1013), None, _d(1020), 102)["identical"]
    # an env termination the row did not have, in an env step wholly inside the row's steps
    for step in (1, 57, 101):
        c = g8a.compare(_d(1013), None, _d(10 * step), step)
        assert not c["ends_alike"] and not c["identical"] and c["first_difference"] == 10 * step
    c = g8a.compare(_d(1010), None, _d(1010), 101)  # a whole last step: it is inside
    assert not c["ends_alike"] and c["first_difference"] is None and not c["identical"]
    # a row fall at physics step j: the env terminates in the env step that holds j
    for j, step in ((1, 1), (10, 1), (11, 2), (2232, 224), (1012, 102)):  # (1012: inside the env's last, partial step)
        assert g8a.compare(_d(j), j, _d(10 * step), step)["identical"]
        for other in (step - 1, step + 1, None):
            n = 10 * (other if other else step)
            c = g8a.compare(_d(j), j, _d(n), other)
            assert not c["ends_alike"] and not c["identical"] and c["reference_fell_step"] == j
    # one digest differs: the first place is named
    env_ = _d(1020)
    env_[407] = b"x"
    c = g8a.compare(_d(1013), None, env_, None)
    assert c["first_difference"] == 407 and not c["identical"] and c["ends_alike"] and c["compared_steps"] == 1013
    past = _d(1020)
    past[1015] = b"y"  # past the row's end: not compared
    assert g8a.compare(_d(1013), None, past, None)["identical"]
    # the env stopped before the row did: every step the row ran must be compared
    c = g8a.compare(_d(1013), None, _d(1000), None)
    assert c["first_difference"] == 1000 and c["compared_steps"] == 1000 and not c["identical"]
    assert g8a.compare([], None, [], None)["identical"]


def test_module_assigns_mujoco_gl_and_names_the_rules_constants():
    assert os.environ["MUJOCO_GL"] == "disable"
    src = (ROOT / "designlab" / "stage2" / "g8a.py").read_text()
    head = src.split('"""', 2)[2]
    assert head.lstrip().startswith("from __future__ import annotations\n\nfrom designlab.stage2 import tbenv\n\n"
                                    'tbenv.assign_gl("disable")')
    assert "import bar" not in src.split("def conditions")[0]  # the stock form runs without the bar
    assert (g8a.SEEDS, g8a.DELAYS, g8a.SHORT_CONDITION) == ((3042, 3043), (0, 5), "walk/v0.4")
    assert g8a.EXTRA_CONDITIONS == ("walk/v0.4_mu0.54", "walk/nominal") and g8a.SUBSTEPS == 10


# ------------------------------------------------------------------------------------------------
# The stock form on the S1 view
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _release():
    yield
    g8a.release()


@needs_run
@pytest.mark.parametrize("k", [0, 3])
def test_zero_action_is_r0_on_the_s1_view(k):
    r = g8a.stock(BODY, "walk/brisk", 3042, k, 1.0)
    assert r["identical"] and r["ends_alike"] and r["first_difference"] is None
    assert r["reference_steps"] == r["compared_steps"] == r["env_physics_steps"] == 500 and r["env_steps"] == 50
    assert (r["set"], r["row"], r["condition"]) == ("S1", "R0", "walk/brisk")
    assert (r["seed"], r["k"], r["seconds"]) == (3042, k, 1.0)
    assert r["reference_fell_step"] is None and r["env_terminated_step"] is None and r["rule"] == g8a.RULE
    assert r["interference_first_env_step"] is None and not r["reference_warned"]
    parts = g8a._STOCK[BODY]
    assert "command" not in parts["ctx"].plant("hard", None).__dict__  # both plants got their own command back
    assert "command" not in parts["envs"]["hard"].p.__dict__
    assert parts["ctx"].plant("hard", None) is not parts["envs"]["hard"].p  # two plants, built apart


@needs_run
def test_a_length_that_is_no_multiple_of_20_ms():
    """The bar's lengths are no multiples of 20 ms: the env covers every physics step the row ran."""
    r = g8a.stock(BODY, "walk/brisk", 3043, 5, 1.013)
    assert r["reference_steps"] == round(1.013 / 0.002) == 506 and r["env_steps"] == 51
    assert r["compared_steps"] == r["reference_steps"] and r["env_physics_steps"] == 510 and r["identical"]


@needs_run
def test_a_fall_ends_both_alike_on_carpet():
    """On robot_floor_carpet the open loop falls: the row at a physics step, the env in the env step that holds it."""
    r = g8a.stock(BODY, "carpet/robot_floor_carpet", 3042, 0, 8.0)
    j = r["reference_fell_step"]
    assert j is not None and r["reference_steps"] == j < 4000 and r["reference_fall_cause"] == r["env_fall_cause"]
    assert r["env_terminated_step"] == -(-j // 10) == r["env_steps"] and r["compared_steps"] == j
    assert r["identical"] and r["ends_alike"]


@needs_run
def test_any_other_action_breaks_the_comparison_and_the_recorder_is_given_back():
    g8a.stock(BODY, "walk/brisk", 3042, 0, 1.0)
    parts = g8a._STOCK[BODY]
    ctx, row, e = parts["ctx"], parts["row"], parts["envs"]["hard"]
    cond = next(c for c in ctx._built if c.key == "walk/brisk" and c.seconds == 1.0)
    b = ctx.built(cond)
    ref = g8a.row_episode(row, ctx, cond, 3042)
    assert "recorder" not in b.__dict__ and len(ref["digests"]) == 500 and ref["fell_step"] is None
    seeds = panels.episode_seeds(cond, 3042)
    a = np.zeros(15, np.float32)
    a[7] = 1e-6
    for action, identical in ((None, True), (a, False)):
        x = g8a.env_episode(e, b.plan, b.seconds, seeds, cond.friction, 0, b.push(3042), action=action)
        c = g8a.compare(ref["digests"], ref["fell_step"], x["digests"], x["terminated_step"])
        assert c["identical"] is identical
        assert (c["first_difference"] is None) is identical and (identical or c["first_difference"] < 10)
    # other seeds are another episode
    other = g8a.env_episode(e, b.plan, b.seconds, dict(seeds, start=seeds["start"] + 1), cond.friction, 0, None)
    assert g8a.compare(ref["digests"], None, other["digests"], None)["first_difference"] == 0
    # the lent recorder is an instance attribute for the one episode, removed whatever happens in it

    class Boom(RuntimeError):
        pass

    class Raising:
        @staticmethod
        def episode(ctx_, c, s, relax):
            assert "recorder" in ctx_.built(c).__dict__
            assert isinstance(ctx_.built(c).recorder(), g8a._Digests)
            raise Boom

    with pytest.raises(Boom):
        g8a.row_episode(Raising, ctx, cond, 3042)
    assert "recorder" not in b.__dict__ and type(b.recorder()).__name__ == "Recorder"
    with pytest.raises(ValueError, match="condition"):
        g8a.stock(BODY, "walk/v0.4")


# ------------------------------------------------------------------------------------------------
# The bar form on a synthetic set
# ------------------------------------------------------------------------------------------------
SET = "T0"
W = setio.walk_name(setio.BAR_SPEED)
T0_REFS = (("nominal", "nominal"), (W, "brisk"))
T0_MOVES = (("start", "start"), (setio.change_name("nominal", W), "nominal_to_brisk"), (setio.stop_name(2, W), "stop"))
ABSENT = setio.turn_name("0.35", "left")
REASON = "no reference within the thigh twist and the leg drives"


def make_t0(root: Path) -> Path:
    """A frozen set of Stage 1's arrays under the bar's names: two walks, the start, the change and a stop; one
    turn recorded as not available."""
    src = RUN / BODY
    d = setio.set_dir(SET, BODY, root)
    refs1, trs1 = s2io.read_json(src / "refs.json"), s2io.read_json(src / "transitions.json")
    refs, trs, manifest = {}, {}, {}
    for name, old in T0_REFS:
        e = copy.deepcopy(refs1["references"][old])
        e.update(name=name, file=f"refs/{name}.npz", set=SET, label=None, strides=1, window_s=e["period_s"],
                 crouch_frac=0.03, margin_rule="a fixture's")
        refs[name] = e
    for name, old in T0_MOVES:
        e = copy.deepcopy(trs1["transitions"][old])
        kind = setio.parse_name(name)["kind"]
        e.update(name=name, file=f"transitions/{name}.npz", a1_over=None,
                 move=dict(kind=kind, builder="fixture", causal=kind == "stop",
                           command_phase=-0.5 if kind == "stop" else None))
        trs[name] = e
    trs[ABSENT] = dict(name=ABSENT, kind="turn", available=False, reason=REASON, numbers=None)
    for doc, sub, kind in ((refs, "refs", "walk"), (trs, "transitions", "move")):
        for name, e in doc.items():
            if not e.get("file"):
                continue
            (d / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / sub / f"{dict(T0_REFS + T0_MOVES)[name]}.npz", d / e["file"])
            manifest[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256(kind, e))
    s2io.write_json(d / "refs.json", dict(body=BODY, set=SET, schema=1, references=refs))
    s2io.write_json(d / "transitions.json", dict(body=BODY, set=SET, schema=1, transitions=trs))
    freeze = dict(index=0, at="fixture", commit=None, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest))
    # (Stage 1's brisk walk stands in for the 0.40 m/s rung: the bar refuses that unless the set says so)
    s2io.write_json(d / "set.json", dict(set=SET, body=BODY, role="fixture", crouch_frac=0.03, not_available={},
                                         stand_in_walks=True, rules=dict(fixture="Stage 1's arrays"),
                                         freezes=[freeze]))
    return d


@needs_run
def test_bar_form_on_a_synthetic_set(tmp_path):
    """The env against the bar's own rows ``R0@0ms`` and ``R0@6ms`` (``bar.Delayed``) on a frozen set."""
    bar = pytest.importorskip("designlab.stage2.bar")
    d = make_t0(tmp_path)
    doc = g8a.bar_form(BODY, SET, ("walk/v0.4", "walk/nominal", "turn/t90_0.35_left"), (3042,), (0, 3), cut_s=1.013,
                       root=tmp_path)
    assert (doc["identical"], doc["of"], doc["passes"]) == (4, 4, True) and doc["rule"] == g8a.RULE
    assert [(e["condition"], e["k"], e["row"]) for e in doc["episodes"]] == [
        ("walk/v0.4", 0, "R0@0ms"), ("walk/v0.4", 3, "R0@6ms"), ("walk/nominal", 0, "R0@0ms"),
        ("walk/nominal", 3, "R0@6ms")]
    for e in doc["episodes"]:
        assert e["identical"] and e["compared_steps"] == e["reference_steps"] == 506 and e["env_steps"] == 51
        assert e["seed"] == 3042 and e["seconds"] == 1.013 and e["interference_first_env_step"] is None
    assert doc["not_run"] == [dict(condition="turn/t90_0.35_left", reference=f"transitions/{ABSENT}", reason=REASON,
                                   numbers=None)]
    freeze = s2io.read_json(d / "set.json")["freezes"][0]
    assert doc["set_freeze"] == dict(index=0, manifest_sha256=freeze["manifest_sha256"]) and doc["set"] == SET
    assert doc["sources"]["bar"] == bar.bar_sources()["sha256"]
    assert doc["sources"]["panels"] == panels.panel_sources()["sha256"]
    mine = ("g8a", "r5core", "r5env", "r5lib")
    assert sorted(doc["sources"]["track_b"]) == [f"designlab/stage2/{n}.py" for n in mine]
    assert doc["key_context"] == panels.key_context(BODY) and doc["blas_threads"] == 1 and "commit" in doc
    # rule Z1's conditions on a set: the gated mu 0.9 ones it defines, then the two named
    keys = g8a.conditions(setio.load(d))
    assert keys[0] == "walk/v0.4" and keys[-2:] == ["walk/v0.4_mu0.54", "walk/nominal"] and len(keys) == 10
    assert all("_mu" not in k for k in keys[:8]) and len(set(keys)) == 10
    # an unfrozen set is refused before any episode
    doc2 = s2io.read_json(d / "set.json")
    doc2["freezes"] = []
    s2io.write_json(d / "set.json", doc2)
    with pytest.raises(tune.InputError, match="not frozen"):
        g8a.bar_form(BODY, SET, ("walk/v0.4",), (3042,), (0,), cut_s=0.5, root=tmp_path)
    assert r5lib.STOCK_SET == bar.STOCK_SET == "S1"
