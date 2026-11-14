"""Tests for what Stage 2's expressive head costs (``stage2/x/x_costs.py``, ``x_costs.json``, ``X_COSTS.md``;
``STAGE2_PLAN.md`` S2-2, DECISION_LOG P29).

* The committed ``x_costs.json`` came from the committed script (its ``script_sha256``).
* Its statics reproduce the records they cite: every check within 1e-9 (relative), and the masses, holds and ratios
  equal to the -X records' ``head_x`` blocks.
* Its Stage 0 walking screens read arrays equal to the regenerated references, and the -X bodies' new joints held
  still in them.
* Its character moves' inverse dynamics recompute exactly on the -X bodies rebuilt from their records.
* The minimum-jerk profile starts and ends at rest and peaks at 1.875 A / T.

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_x_costs.py -q
About 20 s (each -X body is rebuilt once, 4 s, and its moves recomputed).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "stage2" / "x"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUN))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco renders)
import x_costs  # noqa: E402
from designlab import gait  # noqa: E402
from designlab.stage1 import plant, refs  # noqa: E402

DOC = json.loads((RUN / "x_costs.json").read_text())
X_BODIES = ("60-N-T-10-X", "75-M-H-10-X")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_costs_record_is_the_committed_scripts():
    assert DOC["script_sha256"] == _sha(RUN / "x_costs.py"), "x_costs.json is stale: run stage2/x/x_costs.py"
    assert set(DOC["bodies"]) == set(X_BODIES)
    assert [p[1] for p in x_costs.PAIRS] == list(X_BODIES)


@pytest.mark.parametrize("body", X_BODIES)
def test_the_statics_reproduce_the_records(body):
    b = DOC["bodies"][body]
    st = b["statics"]
    rr = st["records_reproduced"]
    assert rr["all_match"] and rr["max_rel_diff"] <= 1e-9 and rr["n"] >= 40
    hx = plant.record(body)["head_x"]
    # the compiled models' masses differ by the added parts to the MJCF's six significant digits (2e-6 of the body's
    # mass, as tests/test_stage2_x.py allows); the parts themselves add 2 x (0.055 servo + 0.020 horn and bracket)
    assert hx["added_mass_kg"] == pytest.approx(0.150, abs=1e-12)
    assert st["mass_kg"]["added"] == pytest.approx(hx["added_mass_kg"], abs=2e-6 * st["mass_kg"]["x"])
    for g in ("hip_pitch", "neck"):
        ab = hx["against_base"][g]
        assert st["loads"][g]["today"]["hold_Nm"] == ab["base_hold_Nm"]
        assert st["loads"][g]["x"]["hold_Nm"] == ab["hold_Nm"]
        assert st["loads"][g]["x"]["hold_over_A1"] == pytest.approx(ab["hold_over_sustained"], rel=1e-12)
        assert st["loads"][g]["today"]["hold_over_A1"] == pytest.approx(ab["base_hold_over_sustained"], rel=1e-12)
    ahead = hx["com_ahead_of_hip_m"]
    assert st["com_m"]["shift_same_pose_mm"][0] == pytest.approx(1e3 * ahead["change_at_pick_pose"], abs=1e-9)
    assert st["com_m"]["shift_own_homes_mm"][0] == pytest.approx(1e3 * ahead["change"], abs=1e-9)
    rec_t, rec_x = plant.record(DOC["bodies"][body]["today"]), plant.record(body)
    rc = st["records_costs"]
    assert rc["cost_usd"]["nominal"] == [rec_t["cost"]["nominal"], rec_x["cost"]["nominal"]]
    assert rc["walking_power"]["x"]["total_w"] == rec_x["power"]["total_w"]
    assert rc["runtime_min"] == [rec_t["runtime"]["minutes"], rec_x["runtime"]["minutes"]]


@pytest.mark.parametrize("body", X_BODIES)
def test_the_walk_screens_read_the_regenerated_references_with_the_new_joints_still(body):
    walks = DOC["bodies"][body]["walks"]
    for name in x_costs.WALKS:
        for head in ("today", "x"):
            assert walks[name][head]["arrays"]["all_equal"], (body, name, head)
        for jn, s in walks[name]["x"]["new_joints_still"].items():
            assert s["q_min"] == s["q_max"] == 0.0 and s["rate_max"] == 0.0, (body, name, jn)
        moved = walks[name]["moves"]
        assert all(abs(r["change"]) >= x_costs.MOVE_THRESHOLD for r in moved["moved"])
        assert moved["largest_unmoved"] < x_costs.MOVE_THRESHOLD


@pytest.mark.parametrize("body", X_BODIES)
def test_the_moves_inverse_dynamics_recompute_exactly(body):
    gait.set_accuracy(refs.ACCURACY)
    src = refs.source(body)
    assert src["model"]["matches_record"] and src["model"]["matches_plan"]
    hd = x_costs.HeadDynamics(src["M"], src["q_home"])
    rec = DOC["bodies"][body]["moves"]["inverse_dynamics"]
    for name, mv in x_costs.MOVES.items():
        got = x_costs.move_inverse_dynamics(hd, src["design"], mv)
        for key in ("peak_Nm", "rms_Nm", "rigid_peak_Nm"):
            assert got[key] == pytest.approx(rec[name][key], rel=1e-12, abs=1e-15), (name, key)
        assert got["ratios"] == pytest.approx(rec[name]["ratios"], rel=1e-12)
    m = hd.m
    for jn in ("neck_yaw", "jaw"):  # the MJCF's default class (designlab/mjcf.py), unsourced
        dof = hd.joint(jn)[1]
        assert (m.dof_damping[dof], m.dof_frictionloss[dof]) == (0.2, 0.15)


def test_the_minimum_jerk_profile():
    T, A = 0.5, 0.6
    t = np.linspace(0.0, T, 2001)
    q, qd, qdd = x_costs.min_jerk(t, T, A)
    assert q[0] == 0.0 and q[-1] == pytest.approx(A, abs=1e-15)
    assert qd[0] == 0.0 and qd[-1] == pytest.approx(0.0, abs=1e-12)
    assert qdd[0] == 0.0 and qdd[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.max(qd) == pytest.approx(x_costs.MIN_JERK_PEAK_SPEED * A / T, rel=1e-12)
    assert float(np.argmax(qd)) == 1000  # at mid-move
    q2, qd2, qdd2 = x_costs.min_jerk(np.array([-0.1, T + 0.1]), T, A)  # at rest outside the move
    assert list(q2) == [0.0, A] and list(qd2) == [0.0, 0.0] and list(qdd2) == [0.0, 0.0]


def test_the_report_cites_its_record():
    md = (RUN / "X_COSTS.md").read_text()
    assert "x_costs.json" in md and "x_costs.py" in md
