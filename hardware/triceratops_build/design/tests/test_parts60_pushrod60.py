"""Tests for the parts list generator (design/parts60.py) and the pushrod sizing (design/pushrod60.py). Run by hand:
    python -m pytest hardware/triceratops_build/design/tests/test_parts60_pushrod60.py
"""

import json
import math
import sys
from pathlib import Path

import pytest

DESIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESIGN))
GEN = DESIGN.parent / "sim" / "generated"

import parts60  # noqa: E402


def test_listed_derivation_reproduces_the_188_mm_listed_ledger():
    closed = json.loads((GEN / "mass_ledger_trike60_C_closed.json").read_text())
    listed = json.loads((GEN / "mass_ledger_trike60_C_listed.json").read_text())
    got = parts60.listed_from_closed(closed)
    assert sum(got.values()) == pytest.approx(listed["total_mass_kg"], abs=2e-3)
    for c, kg in listed["category_kg"].items():
        assert got[c] == pytest.approx(kg, abs=2e-3)


def test_committed_parts_list_is_current_and_from_the_adopted_model(tmp_path):
    out = tmp_path / "PARTS_60CM.md"
    parts60.main(out)
    assert out.read_text() == (DESIGN / "PARTS_60CM.md").read_text(), "run design/parts60.py and commit the result"
    text = out.read_text()
    adopted = json.loads((GEN / parts60.LEDGER).read_text())
    assert parts60.LEDGER == "mass_ledger_trike60_C_closed_capsule_fore128.json"
    assert f"**{adopted['total_mass_kg']:.3f} kg**" in text
    assert "Orion BMS Jr 2" in text and "| $440 |" in text and "| $100 |" in text


def test_battery_cut_off_adds_540_of_hardware_and_647_landed():
    cfg = next(c for c in parts60.sb.CONFIGS if c[0].startswith("C:"))
    hw0, l0 = parts60.costs(cfg, 0.0, False)
    hw1, l1 = parts60.costs(cfg, 0.0, True)
    assert round(hw0) == 16776 and round(l0["Fremont stock, duty paid (0 %)"]) == 20105   # size_budget_out.txt
    assert hw1 - hw0 == pytest.approx(540.0)
    for k in l0:
        assert l1[k] - l0[k] == pytest.approx(540.0 * 1.12 * 1.07)


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")  # pushrod60 imports sim/check_packaging60.py, which needs scipy


def test_pushrod_forces_reproduce_the_review_and_size_the_rod(tmp_path):
    import pushrod60
    out = pushrod60.main(tmp_path / "out.json")
    fore = out["cases"]["fore"]["forces_n"]
    assert fore["joint_limit"] == pytest.approx(1979, abs=1)   # REVIEW_60CM.md (2ac57608) line 117
    assert fore["peak"] == pytest.approx(3142, abs=1)
    assert out["geometry"]["fore"]["pin_to_pin_m"] == pytest.approx(0.0815, abs=1e-5)
    assert out["geometry"]["hind"]["pin_to_pin_m"] == pytest.approx(0.0980, abs=1e-5)
    assert out["geometry"]["fore"]["min_transmission_angle_deg"] == pytest.approx(43.0, abs=0.05)
    best = [r for r in out["candidates"] if r["kind"] == "fore" and r["passes"] and r["passes_fatigue_bound"]]
    assert [(r["material"], r["thread"]) for r in best] == [("class 8.8 steel stud, rolled threads", "M8")]
    m5 = [r for r in out["candidates"] if r["kind"] == "fore" and r["thread"] == "M5"]
    assert m5 and not any(r["passes"] for r in m5)


def test_fore_and_hind_parts_with_different_bases_have_their_own_rows():
    led = json.loads((GEN / parts60.LEDGER).read_text())
    rows = parts60.mass_table(led)
    by = {(c, f): (n, kg, basis) for c, f, n, kg, _, _, basis in rows}
    assert ("frame and feet", "upper_strut") not in by
    n, kg, basis = by[("frame and feet", "upper_strut, hind")]
    assert n == 2 and kg == pytest.approx(2 * 0.086488, abs=1e-4) and basis.startswith("PR #669 part")
    assert by[("frame and feet", "upper_strut, fore")][2].startswith("the 188 mm part's mass")
    for c, f in (("leg transmissions", "pushrod"), ("shells", "upper_limb_cover"), ("frame and feet", "elbow_axle")):
        assert (c, f + ", fore") in by and (c, f + ", hind") in by
    assert sum(kg for _, _, _, kg, _, _, _ in rows) == pytest.approx(led["total_mass_kg"], abs=1e-3)   # ledger rounding


def test_cut_off_estimate_grows_in_the_closed_column():
    text = (DESIGN / "PARTS_60CM.md").read_text()
    led = json.loads((GEN / parts60.LEDGER).read_text())
    grown = 0.25 * parts60.growth_factor(led)
    assert f"+{grown:.3f} kg" in text and f"**{led['total_mass_kg'] + grown:.3f} kg**" in text


def test_clevis_pin_counts_bending_and_fatigue(tmp_path):
    import pushrod60
    out = pushrod60.main(tmp_path / "out.json")
    p8 = next(p for p in out["crank_pins"]["fore"] if p["pin_mm"] == 8.0 and p["lever_mm"] == 6.0)
    c0 = next(c for c in p8["clevis"] if c["gap_each_side_mm"] == 0.0)
    assert c0["moment_nm"] == pytest.approx(3142 / 2 * 4 / 1000, rel=1e-3)   # F/2 x (t/2 + b/4), b 8, t 4 mm
    assert c0["bending_mpa"] == pytest.approx(125.0, abs=0.5)
    assert c0["combined_von_mises_mpa"] > c0["bending_mpa"] > 3 * c0["shear_mpa"]   # bending, not shear, governs
    assert c0["fatigue_n"]["F1 reversed +-limit"] >= pushrod60.SF_FATIGUE
    assert p8["cantilever_fatigue_n"]["F1 reversed +-limit"] < pushrod60.SF_FATIGUE
    assert out["rod_end_sensitivity"]["M8 at C0 7.9 kN, front static case"] == pytest.approx(2.51, abs=0.01)


def test_static_case_is_the_largest_torque_drives_json_gives(tmp_path):
    """The X4-36-L's stall torque (36 N m) is above its peak (30 N m), so the hind static case takes it; the X6-60 has no
    stall figure, so the front takes its peak (sim/data/drives.json). The hind recommendations hold at 1.9 kN."""
    import pushrod60
    drv = json.loads((DESIGN.parent / "sim" / "data" / "drives.json").read_text())["leg_drives"]
    assert drv["X4-36-L"]["stall_nm"] > drv["X4-36-L"]["peak_nm"] and drv["X6-60"]["stall_nm"] is None
    out = pushrod60.main(tmp_path / "out.json")
    hind, fore = out["cases"]["hind"], out["cases"]["fore"]
    assert hind["torques_nm"]["static"] == 36 and fore["torques_nm"]["static"] == 60
    assert hind["forces_n"]["static"] == pytest.approx(36 / (0.028 * math.sin(math.radians(43.0))), rel=1e-3)
    assert fore["forces_n"]["static"] == fore["forces_n"]["peak"]
    ok = {(r["material"], r["thread"]) for r in out["candidates"] if r["kind"] == "hind" and r["passes"]
          and r["passes_fatigue_bound"] and r["passes_fatigue_walking"]}
    assert {("class 8.8 steel stud, rolled threads", "M6"), ("1018 CD steel, cut threads", "M8")} <= ok
    assert out["x6_stall_sensitivity"]["class 8.8 M8 rod"]["passes"]
