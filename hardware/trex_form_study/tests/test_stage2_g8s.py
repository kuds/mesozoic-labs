"""Gate G8s and the servo option's keys (``designlab/stage2/g8s.py``; ``stage2/design/servo_model/
SERVO_MODEL_DESIGN.md`` 5.1, tests T1, T3 and T13; package S2).

The closed-loop tests run the stock form on the S1 view (Stage 1's own run directory) and the bar form on a
synthetic set ``T0`` that ``tests/test_stage2_g8a.py``'s fixture builds in a temporary directory from Stage 1's
arrays: no Stage 2 reference set is touched (rule X1).  Simulation only: the servo model's values are a model, none
is measured, and no episode here is a record.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import drive, env, panels, plant  # noqa: E402
from designlab.stage2 import g8a, g8s, r5env, r5lib, regress, s2io, servo  # noqa: E402

BODY = servo.BODY
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
#: The registered pair (design T1): the bar's and the panels' source hashes, in every record key.
BAR_SOURCES = "45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d"
PANEL_SOURCES = "c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1"


@pytest.fixture(scope="module", autouse=True)
def _release():
    yield
    g8s.release()


# ------------------------------------------------------------------------------------------------
# T1: no record key moves
# ------------------------------------------------------------------------------------------------
def test_t1_the_record_keys_hashes_stay_and_the_regress_check_passes():
    """T1: the bar's and the panels' source hashes are the registered pair with the option, the layer and this gate
    in the code; none of their files is in either hash; the regress record's four hashes are today's."""
    bar = pytest.importorskip("designlab.stage2.bar")
    b, pn = bar.bar_sources(), panels.panel_sources()
    assert (b["sha256"], pn["sha256"]) == (BAR_SOURCES, PANEL_SOURCES)
    assert not {"servo.py", "g8s.py", "r5env.py"} & {Path(f).name for f in b["files"]}
    assert not {"servo.py", "g8s.py", "r5env.py"} & set(pn["modules"] + pn["designlab_modules"])
    assert regress.check() == []


def test_module_assigns_mujoco_gl_and_names_t13s_cases():
    assert os.environ["MUJOCO_GL"] == "disable"
    src = (ROOT / "designlab" / "stage2" / "g8s.py").read_text()
    head = src.split('"""', 2)[2]
    assert head.lstrip().startswith("from __future__ import annotations\n\nfrom designlab.stage2 import tbenv\n\n"
                                    'tbenv.assign_gl("disable")')
    assert "import bar" not in src.split("def bar_form")[0]  # the stock form runs without the bar
    assert g8s.RULE == g8a.RULE and g8s.SUBSTEPS == 10 and g8s.DELAYS == (0, 5)
    assert g8s.SETTINGS == ("N", "F+", "P+", "V-", "J+", "M-", "M+") and g8s.D_SEEDS == (3042, 3043, 3044)
    assert g8s.cases() == [(s, 3042) for s in g8s.SETTINGS] + [("D", 3042), ("D", 3043), ("D", 3044)]
    assert g8s.params_of("D", 3043) == servo.panel_draw(3043) and g8s.params_of("M+", 1) is servo.setting("M+")
    doc = " ".join(g8s.__doc__.split())
    assert "Simulation only" in doc and "S3's row ``servo.Served`` and its names are not built yet" in doc


# ------------------------------------------------------------------------------------------------
# T13: the env and the served R0 row run one model
# ------------------------------------------------------------------------------------------------
@needs_run
@pytest.mark.parametrize("k", g8s.DELAYS)
@pytest.mark.parametrize("case", g8s.cases(), ids=lambda c: f"{c[0]}@{c[1]}")
def test_t13_the_env_is_the_served_r0_row(case, k):
    """T13: the env with the zero action and the setting against the served R0 row of the same setting, at 0 and 10
    ms: every physics step's state digest is equal (G8a's rule), the trunk mass of M- and M+ applied before the
    start on both sides; both plants are given back as they were."""
    setting, seed = case
    r = g8s.stock(setting, "walk/brisk", seed, k, 1.0)
    assert r["identical"] and r["ends_alike"] and r["first_difference"] is None, r
    assert r["reference_steps"] == r["compared_steps"] == r["env_physics_steps"] == 500 and r["env_steps"] == 50
    want = servo.setting(setting, seed) if setting == "D" else servo.setting(setting)
    assert r["setting"] == want.name and r["trunk_mass"] == want.mass and r["table_sha256"] == servo.TABLE_SHA256
    assert (r["set"], r["row"], r["condition"], r["seed"], r["k"]) == ("S1", "R0", "walk/brisk", seed, k)
    sr = r["servo_readings"]
    assert sr["row"] == sr["env"] and sr["row"]["physics_steps"] == 500  # (500 = a whole number of env steps)
    parts = g8s._STOCK[BODY]
    e, p_row = parts["envs"]["hard"], parts["ctx"].plant("hard", None)
    for p in (p_row, e.p):
        assert not {"step", "command", "reset"} & set(p.__dict__) and type(p.envelope) is drive.Envelope
    # the row's plant has its built masses back; the env keeps an episode's factors until its next reset (today's)
    assert np.array_equal(p_row.m.body_mass, e.nominal_model.body_mass)
    assert e.p.m.body_mass[e.p.trunk] == want.mass * e.nominal_model.body_mass[e.p.trunk]
    assert p_row is not e.p  # two plants, built apart


@needs_run
def test_g8s_tells_another_model_apart():
    """The gate is not blind: the env given another setting, or the same play on the other side of a reset, differs
    from the served row at the first physics step; ``episode`` refuses a row that is not served with the env's
    parameter object; the layer refuses a mass setting on a plant whose trunk is as built."""
    g8s.stock("N", "walk/brisk", 3042, 0, 0.2)
    parts = g8s._STOCK[BODY]
    ctx, row, e = parts["ctx"], parts["row"], parts["envs"]["hard"]
    cond = next(c for c in ctx._built if c.key == "walk/brisk" and c.seconds == 0.2)
    b = ctx.built(cond)
    seeds = panels.episode_seeds(cond, 3042)
    for mine, other in (("N", "P+"), ("N", "M+"), ("F-", "F+"), ("D/3042", "D/3043")):
        ref = g8a.row_episode(g8s._Delayed(g8s.ServedRow(row, r5env.servo_params(mine)), 0), ctx, cond, 3042)
        x = g8s.env_episode(e, b.plan, b.seconds, seeds, cond.friction, 0, b.push(3042), r5env.servo_params(other))
        c = g8a.compare(ref["digests"], ref["fell_step"], x["digests"], x["terminated_step"])
        assert not c["identical"] and c["first_difference"] is not None and c["first_difference"] < 10, (mine, other)
    with pytest.raises(ValueError, match="not the served row"):
        g8s.episode(g8s._Delayed(g8s.ServedRow(row, servo.setting("N")), 0), ctx, cond, 3042, 0, e,
                    servo.setting("F+"))
    with pytest.raises(servo.ServoError, match="trunk"):
        with servo.served(b.p, servo.setting("M+")):
            pass


@needs_run
@pytest.mark.parametrize("k", [0, 3])
def test_t3_today_adds_nothing_on_either_side(k):
    """T3: with the setting ``today`` the R0 row's and the env's state digests equal those without the layer, at 0
    and 6 ms (and so the env with the option off is the served row of ``today``)."""
    r = g8s.stock("today", "walk/brisk", 3042, k, 1.0)
    assert r["identical"] and r["reference_steps"] == 500
    parts = g8s._STOCK[BODY]
    ctx, row = parts["ctx"], parts["row"]
    cond = next(c for c in ctx._built if c.key == "walk/brisk" and c.seconds == 1.0)
    b = ctx.built(cond)
    plain = g8a.row_episode(g8s._Delayed(row, k), ctx, cond, 3042)
    served = g8a.row_episode(g8s._Delayed(g8s.ServedRow(row, servo.setting("today")), k), ctx, cond, 3042)
    assert served["digests"] == plain["digests"] and len(plain["digests"]) == 500
    off = r5env.R5Env(r5env.R5Config(body=BODY, set=r5lib.STOCK_SET, floor="hard", tasks=("stand",), randomize=False,
                                     push_prob=0.0, head_events=False, mid_start=0.0))
    seeds = panels.episode_seeds(cond, 3042)
    x_off = g8a.env_episode(off, b.plan, b.seconds, seeds, cond.friction, k, b.push(3042))
    x_on = g8s.env_episode(parts["envs"]["hard"], b.plan, b.seconds, seeds, cond.friction, k, b.push(3042),
                           servo.setting("today"))
    assert x_on["digests"] == x_off["digests"] and len(x_off["digests"]) == 500
    assert x_off["digests"][:500] == plain["digests"]


@needs_run
def test_the_served_row_gives_the_plant_back_and_trunk_mass_undoes_itself():
    """``ServedRow``: the trunk mass of a setting is applied before the inner episode and undone after (the model
    bit for bit as before), the layer's own undo inside it; a factor of 1 touches nothing."""
    g8s.stock("N", "walk/brisk", 3042, 0, 0.2)
    parts = g8s._STOCK[BODY]
    ctx, row = parts["ctx"], parts["row"]
    cond = next(c for c in ctx._built if c.key == "walk/brisk" and c.seconds == 0.2)
    p = ctx.built(cond).p
    saved = copy.deepcopy(p.m)
    seen = {}

    class Probe:
        name = "probe"

        @staticmethod
        def episode(ctx_, c, s, relax):
            seen["mass"] = float(p.m.body_mass[p.trunk])
            seen["step"] = "step" in p.__dict__
            return row.episode(ctx_, c, s, relax)

    sr = g8s.ServedRow(Probe, servo.setting("M-"))
    g8a.row_episode(g8s._Delayed(sr, 2), ctx, cond, 3042)
    assert seen == dict(mass=0.9 * float(saved.body_mass[p.trunk]), step=True)
    assert env.model_differences(p.m, saved) == [] and "step" not in p.__dict__
    assert sr.readings["physics_steps"] == 100 and sr.name == "probe/servo:M-"
    with g8s.trunk_mass(p, 1.0):
        assert env.model_differences(p.m, saved) == []

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with g8s.trunk_mass(p, 1.1):
            assert p.m.body_mass[p.trunk] == 1.1 * saved.body_mass[p.trunk]
            raise Boom
    assert env.model_differences(p.m, saved) == []


# ------------------------------------------------------------------------------------------------
# The bar form on a synthetic set
# ------------------------------------------------------------------------------------------------
@needs_run
def test_bar_form_on_a_synthetic_set(tmp_path):
    """The env against ``bar.Delayed(ServedRow(the bar's own R0 row))`` on a frozen set, for a named setting and the
    drawn panel of the run's seed."""
    bar = pytest.importorskip("designlab.stage2.bar")
    from tests import test_stage2_g8a as G

    G.make_t0(tmp_path)
    doc = g8s.bar_form(BODY, G.SET, ("walk/v0.4", "turn/t90_0.35_left"), (3042,), (0, 3), ("M+", "D"),
                       cut_s=1.013, root=tmp_path)
    assert (doc["identical"], doc["of"], doc["passes"]) == (4, 4, True)
    assert [(e["setting"], e["k"], e["row"]) for e in doc["episodes"]] == [
        ("M+", 0, "R0@0ms/servo:M+"), ("M+", 3, "R0@6ms/servo:M+"), ("D/3042", 0, "R0@0ms/servo:D/3042"),
        ("D/3042", 3, "R0@6ms/servo:D/3042")]
    for e in doc["episodes"]:
        assert e["identical"] and e["compared_steps"] == e["reference_steps"] == 506 and e["env_steps"] == 51
    assert [n["condition"] for n in doc["not_run"]] == ["turn/t90_0.35_left"]
    assert doc["sources"]["bar"] == bar.bar_sources()["sha256"] == BAR_SOURCES
    assert doc["sources"]["panels"] == panels.panel_sources()["sha256"] == PANEL_SOURCES
    mine = ("g8a", "g8s", "r5core", "r5env", "r5lib", "servo")
    assert sorted(doc["sources"]["track_b"]) == [f"designlab/stage2/{n}.py" for n in mine]
    assert doc["table_sha256"] == servo.TABLE_SHA256 and doc["align_rule"] == g8s.ALIGN_RULE
