"""Tests for Stage 2's Jetson Orin Nano Super + Partial PLEM + walking speed study
(``stage2/jetson_plem/jetson_plem_study.py``, ``jetson_plem_study.json``, ``JETSON_PLEM_AND_SPEED.md``).

* The committed ``jetson_plem_study.json`` came from the committed script (matching ``script_sha256``).
* Both bodies (``60-N-T-10`` and ``75-M-H-10``) reproduce their committed Round 2 records and Stage 1 MJCF
  hashes within 1e-9 relative without modifying ``round2/`` or ``designlab/stage1/``.
* Packaging checks verify that ``75-M-H-10`` accepts ``nvidia_orin_nano_super_devkit`` with zero internal
  hardware overlaps and zero chest profile flare, while ``60-N-T-10`` exhibits the expected -6.95 mm
  ``power_distribution`` vs ``hip_roll_motor`` overlap at the 110 mm default PDB length and resolves to
  > +8.5 mm clearance with the 94 mm PDB layout fix.
* Power and 6S->12V buck regulator checks verify catalog parts ``pololu_d36v50f12`` (15 W mode) and
  ``pololu_d42v110f12`` (25 W Super mode) clear 25.2 V 6S full charge and keep walking runtime >= 20 min
  on ``lipo_6s_3300``.
* Speed and latency checks verify that ``F25 + F9`` speed-tuning + counter-phased tail lowers both drive
  speed ratios and required floor friction ``mu_req`` at >= 0.40 m/s while satisfying mid-swing pad
  clearance and joint-range margins, and that Jetson + Partial PLEM expands the Chest ToF safe stopping
  speed ceiling over the baseline 200 ms perception pipeline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "stage2" / "jetson_plem"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUN))

import designlab  # noqa: E402,F401
import jetson_plem_study  # noqa: E402
from designlab import catalog  # noqa: E402

DOC = json.loads((RUN / "jetson_plem_study.json").read_text())
BODIES = ("60-N-T-10", "75-M-H-10")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_study_record_matches_committed_script():
    assert DOC["script_sha256"] == _sha(RUN / "jetson_plem_study.py"), (
        "jetson_plem_study.json is stale: run stage2/jetson_plem/jetson_plem_study.py"
    )
    assert set(DOC["bodies"]) == set(BODIES)
    assert [p[0] for p in jetson_plem_study.PAIRS] == list(BODIES)


@pytest.mark.parametrize("body", BODIES)
def test_baseline_reproduces_round2_records_and_stage1_hashes(body):
    rep = DOC["bodies"][body]["baseline_reproduction"]
    assert rep["xml_matches_record"] is True
    assert rep["xml_matches_stage1_plan"] is True
    assert rep["all_match"] is True
    assert rep["max_rel_diff"] <= 1e-9
    assert rep["n_checks"] >= 10


def test_jetson_packaging_dropin_and_60cm_pdb_fix():
    pkg75 = DOC["bodies"]["75-M-H-10"]["jetson_packaging"]["drop_in_devkit"]
    base75 = DOC["bodies"]["75-M-H-10"]["baseline_reproduction"]["baseline_exterior_mm"]
    assert pkg75["same_body_overlap_count"] == 0
    assert pkg75["exterior_mm"]["torso_width_mm"] == pytest.approx(base75["torso_width_mm"], abs=0.05)
    assert pkg75["exterior_mm"]["max_width_mm"] == pytest.approx(base75["max_width_mm"], abs=0.05)
    assert pkg75["added_mass_over_pi5_g"] == pytest.approx(78.7, abs=1.5)

    pkg60_drop = DOC["bodies"]["60-N-T-10"]["jetson_packaging"]["drop_in_devkit"]
    pkg60_fix = DOC["bodies"]["60-N-T-10"]["jetson_packaging"]["shortened_94mm_pdb_fix"]
    base60 = DOC["bodies"]["60-N-T-10"]["baseline_reproduction"]["baseline_exterior_mm"]
    assert pkg60_drop["same_body_overlap_count"] >= 1
    assert pkg60_drop["min_pdb_gap_mm"] == pytest.approx(-6.95, abs=0.25)
    assert pkg60_drop["exterior_mm"]["torso_width_mm"] == pytest.approx(base60["torso_width_mm"], abs=0.05)
    assert pkg60_drop["exterior_mm"]["max_width_mm"] == pytest.approx(base60["max_width_mm"], abs=0.05)

    assert pkg60_fix is not None
    assert pkg60_fix["pdb_length_mm"] == pytest.approx(94.0, abs=0.1)
    assert pkg60_fix["same_body_overlap_count"] == 0
    assert pkg60_fix["min_pdb_gap_mm"] >= 8.5


@pytest.mark.parametrize("body", BODIES)
def test_power_regulators_and_runtime_margins(body):
    pwr = DOC["bodies"][body]["power_and_regulators"]
    regs = pwr["regulators"]
    assert regs["mode_15w"]["id"] == "pololu_d36v50f12"
    assert regs["mode_25w"]["id"] == "pololu_d42v110f12"
    assert catalog.part(regs["mode_15w"]["id"]).voltage == 50.0
    assert "12V" in catalog.part(regs["mode_15w"]["id"]).model
    assert catalog.part(regs["mode_25w"]["id"]).voltage == 60.0
    assert "12V" in catalog.part(regs["mode_25w"]["id"]).model

    stacks = pwr["stacks"]
    assert stacks["jetson_15w_partial_plem"]["total_walking_w"] < stacks["jetson_15w_teensy"]["total_walking_w"]
    assert stacks["jetson_15w_partial_plem"]["runtime_6s_3300_min"] > stacks["jetson_15w_teensy"]["runtime_6s_3300_min"]
    for st in stacks.values():
        assert st["meets_20min_spec_on_3300"] is True
        assert st["runtime_6s_3300_min"] >= 28.0


@pytest.mark.parametrize("body", BODIES)
def test_speed_tuning_f9_friction_and_stopping_bounds(body):
    sl = DOC["bodies"][body]["speed_and_latency"]
    std_map = {r["speed_ms"]: r for r in sl["stage0_standard"]}
    tun_map = {r["speed_ms"]: r for r in sl["stage0_tuned_f9"]}

    s40 = std_map[0.4]
    t40 = tun_map[0.4]
    assert t40["gates_ok"] is True
    assert t40["worst_planning_ratio_24v"] < s40["worst_planning_ratio_24v"]
    assert t40["mu_req_max"] < s40["mu_req_max"]
    assert t40["tail_cont_ratio_total"] < s40["tail_cont_ratio_total"]

    lp = sl["closed_loop_latency_probe_040ms"]["results"]
    assert lp["2ms"]["survived"] >= lp["10ms"]["survived"]

    sh = sl["stopping_horizon"]
    assert sh["camera_near_floor_ahead_of_toes_m"] > 2.30
    assert sh["usable_tof_stopping_budget_m"] > 0.80
    assert sh["max_tof_safe_speed_jetson_plem_worst_ms"] > sh["max_tof_safe_speed_pi5_worst_ms"]
    assert sh["max_tof_safe_speed_jetson_plem_mean_ms"] > sh["max_tof_safe_speed_pi5_mean_ms"]


def test_reports_cite_study_artifacts():
    md = (RUN / "JETSON_PLEM_AND_SPEED.md").read_text()
    assert "jetson_plem_study.json" in md
    assert "jetson_plem_study.py" in md
    assert "BUILD_BUDGET_75_JETSON.md" in md
    budget_md = (RUN / "BUILD_BUDGET_75_JETSON.md").read_text()
    assert "75-M-H-10.json" in budget_md
    assert "pololu_d42v110f12" in budget_md
    assert "PCAN-USB FD" in budget_md
    compute_md = (ROOT / "COMPUTE.md").read_text()
    assert "stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md" in compute_md
    assert "stage2/jetson_plem/BUILD_BUDGET_75_JETSON.md" in compute_md
