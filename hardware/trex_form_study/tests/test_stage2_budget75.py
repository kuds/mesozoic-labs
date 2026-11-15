"""Tests for the 75 cm build's price record (``stage2/jetson75/budget_75.py``, ``budget_75.json``, ``BUDGET_75.md``;
``JETSON_PLEM_REVIEW.md`` section 5, task 9).  The file names keep the word "budget" from the first version.

* The committed ``budget_75.json`` came from the committed script and regenerates from it exactly.
* The baseline reproduces the round-two record's cost block for ``75-M-H-10`` exactly: every line and the totals
  ($13,576 / $18,521 / $25,658), with the unrounded drivers to 12 digits; a record that does not reproduce stops the
  script.
* Every catalog id the record names, and every id the price record prices, exists in ``designlab/data``; the two
  ids PR #668 cited do not.
* The Jetson line uses the catalog's price ($399), not the stale $249.
* Every total is its lines plus the cost model's contingency; the blocks add up to the scenarios; P21's items are not
  counted twice; the optional lines stay out of the main totals; every low is at most its nominal and its high.
* A line that a kept lot may already hold (the regulator, the supervisor) is priced both ways, and the
  Jetson-against-Pi difference is a range read from the catalog and the compute survey.
* The lines are ranked by the gap between their low and high cases (where the prices are least certain), and what
  a quote would cover is described without telling anyone to act.
* ``BUDGET_75.md`` embeds the tables the script prints from the record, quotes the record's figures and none of
  PR #668's totals.
* Decision-log P31: prices are documented and not weighed against P4.  Outside the dated note at its top, the
  document names P4 once, to say that it is not applied; the document, the record and the script hold no verdict
  against P4, no option to cut a line and no request for a budget decision.

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_budget75.py -q      (about 2 s)
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "stage2" / "jetson75"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUN))

import budget_75  # noqa: E402
from designlab import catalog, cost  # noqa: E402

DOC = json.loads((RUN / "budget_75.json").read_text())
RECORD = json.loads((ROOT / budget_75.RECORD).read_text())
HEAD = json.loads((ROOT / budget_75.HEAD_RECORD).read_text())
TIERS = ("low", "nominal", "high")
PR668_IDS = ("feetech_sts3215_12v", "robotis_xm540_w270")
#: PR #668's (unverified) totals, self-assembled and turnkey; none may appear in this document.  Its turnkey high
#: equals this budget's ``pi_p21`` high by coincidence (other lines), so that one cannot be tested by its digits.
PR668_TOTALS = ("$12,586", "$17,216", "$24,077", "$14,282", "$20,030")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _all_ids() -> set:
    return set(catalog.actuators()) | set(catalog.parts())


def _same_cost(est: dict, recorded: dict) -> None:
    """Every line (to the cent) and the three totals equal; the unrounded drivers to 12 digits."""
    assert est["lines"] == recorded["lines"] and len(est["lines"]) == 16
    assert [est[t] for t in TIERS] == [recorded[t] for t in TIERS]
    assert set(est["drivers"]) == set(recorded["drivers"])
    for name, value in recorded["drivers"].items():
        assert est["drivers"][name] == pytest.approx(value, rel=1e-12, abs=0.0)


def _with_contingency(usd) -> list:
    return [round(usd[i] * (1 + cost.CONTINGENCY[i]), 0) for i in range(3)]


def _prose() -> tuple:
    text = (RUN / "BUDGET_75.md").read_text()
    begin, end = "<!-- budget_75.py md: begin -->\n", "<!-- budget_75.py md: end -->"
    assert text.count(begin) == 1 and text.count(end) == 1
    return text, text.split(begin)[0] + text.split(end)[1], text.split(begin)[1].split(end)[0]


def _note_and_rest() -> tuple:
    """The dated note at the top of the document (it lists what the first version held, in that version's words) and
    the document without it."""
    text = (RUN / "BUDGET_75.md").read_text()
    begin, end = "**Changed, 14 Nov 2026 (P31).**", "\n14 Nov 2026. A price record"
    assert text.count(begin) == 1 and text.count(end) == 1
    head, rest = text.split(begin)
    note, tail = rest.split(end)
    return begin + note, head + end + tail


def test_the_record_is_the_committed_scripts_and_regenerates_exactly():
    stale = "budget_75.json is stale: run stage2/jetson75/budget_75.py"
    assert DOC["script_sha256"] == _sha(RUN / "budget_75.py"), stale
    for rel, sha in DOC["inputs"].items():
        assert _sha(ROOT / rel) == sha, f"{rel} changed since budget_75.json was written: {stale}"
    # every file a number depends on is hashed: the records, the cost model, the modules it reads and the data
    for rel in (budget_75.RECORD, budget_75.HEAD_RECORD, budget_75.LEVERS, budget_75.LEVER_REFERENCE,
                budget_75.SURVEY, "designlab/cost.py", "designlab/build.py", "designlab/refine.py",
                "designlab/catalog.py", "designlab/data/parts.json", "designlab/data/actuators.json"):
        assert rel in DOC["inputs"], rel
    fresh = budget_75.build()
    assert json.loads(json.dumps(fresh)) == DOC
    assert json.dumps(fresh, indent=1) + "\n" == (RUN / "budget_75.json").read_text()


def test_the_baseline_reproduces_the_round_two_record_exactly():
    # the record's own totals (LESSONS_LEARNED.md section 7: nominal $18.5k, high $25.7k)
    assert [RECORD["cost"][t] for t in TIERS] == [13576.0, 18521.0, 25658.0]
    est = cost.estimate(budget_75.RecordDesign(RECORD), budget_75._rows(RECORD), RECORD["selection"])
    _same_cost(est, RECORD["cost"])
    rep = budget_75.reproduce(RECORD)
    assert rep["exact"] and rep["differences"] == []
    b = DOC["baseline"]
    assert b["reproduces_exactly"] and b["differences"] == []
    assert b["recorded"] == b["recomputed"] == {t: RECORD["cost"][t] for t in TIERS}
    assert b["lines_recomputed"] == b["lines_recorded"] == RECORD["cost"]["lines"]
    assert b["record_sha256"] == _sha(ROOT / budget_75.RECORD) == HEAD["head_x"]["base_record_sha256"]
    # the scenario that changes nothing is the record, and its totals rebuild from the rounded lines
    assert DOC["scenarios"]["record"]["turnkey"]["total"] == [RECORD["cost"][t] for t in TIERS]
    assert DOC["scenarios"]["record"]["vs_record"]["lines"] == {}


def test_a_record_that_does_not_reproduce_stops_the_script(monkeypatch):
    bad = copy.deepcopy(RECORD)
    bad["cost"]["lines"]["frame"][1] += 1.0
    bad["cost"]["nominal"] += 1.0
    rep = budget_75.reproduce(bad)
    assert not rep["exact"]
    assert {d["where"] for d in rep["differences"]} == {"total nominal", "line frame"}
    real = budget_75._load
    monkeypatch.setattr(budget_75, "_load", lambda rel: bad if rel == budget_75.RECORD else real(rel))
    with pytest.raises(SystemExit) as stop:
        budget_75.build()
    assert "does not reproduce" in str(stop.value) and "line frame" in str(stop.value)


def test_every_catalog_id_the_record_names_exists():
    ids = _all_ids()
    sel = RECORD["selection"]
    named = [v for v in sel.values() if isinstance(v, str)] + list(sel["joints"].values())
    assert len(named) == 13 and set(named) == set(DOC["baseline"]["record_catalog_ids"])
    missing = [i for i in named if i not in ids]
    assert missing == []
    # and every id the budget itself prices, at the catalog's price, with the catalog's own basis for that price
    for i, entry in DOC["catalog"].items():
        assert i in ids
        item = catalog.actuators().get(i) or catalog.parts()[i]
        assert entry["price_usd"] == item.price and item.price > 0
        assert entry["basis"], i  # an actuator's basis is keyed "price"; a part's is one string
        if i in catalog.actuators():
            assert entry["basis"] == item.basis["price"] == "sourced"
    for spec in budget_75.EXTRA.values():
        for pid in spec.get("parts", ()):
            assert pid in catalog.parts()


def test_the_ids_pr668_cited_do_not_exist():
    ids = _all_ids()
    for wrong in PR668_IDS:
        assert wrong not in ids
    rows = {r["what"]: r for r in DOC["pr668_slips"]}
    assert rows["The tail servo's catalog id"]["right_id"] == RECORD["selection"]["tail"] == "feetech_sts3215_c018"
    assert rows["The neck servo's catalog id"]["right_id"] == RECORD["selection"]["neck"] == "robotis_xm540_w270_t"
    for r in DOC["pr668_slips"]:
        if "right_id" in r:
            assert r["pr668_id_exists"] is False and r["right_id_exists"] is True


def test_the_jetson_line_uses_the_catalogs_price():
    jetson, pi = catalog.part(budget_75.JETSON), catalog.part(budget_75.PI5)
    assert jetson.price == 399 and pi.price == 290  # the catalog's prices (the review; COMPUTE.md)
    assert RECORD["selection"]["compute"] == budget_75.PI5
    for name in ("jetson_mcu_p21", "jetson_direct_p21", "jetson_mcu_p21_optional"):
        s = DOC["scenarios"][name]
        assert s["compute"] == budget_75.JETSON
        assert s["lines"]["compute"] == [jetson.price + 60, jetson.price + 120, jetson.price + 200]
    for name in ("record", "pi_p21", "pi_p21_optional"):
        s = DOC["scenarios"][name]
        assert s["compute"] == budget_75.PI5
        assert s["lines"]["compute"] == [pi.price + 60, pi.price + 120, pi.price + 200]
    swap = DOC["blocks"]["jetson"]["lines"]["compute"]["usd"]
    assert swap == [jetson.price - pi.price] * 3 == [109.0] * 3
    slip = DOC["pr668_slips"][0]
    assert slip["catalog_price_usd"] == jetson.price and slip["pr668_price_usd"] == 249
    # the regulator the dev kit's carrier needs is a catalog part at its catalog price; which part is an assumption
    reg = DOC["blocks"]["jetson"]["lines"]["regulator_12v"]
    assert reg["kind"] == "c+d" and reg["caution"]
    assert reg["usd"] == [catalog.part(c["id"]).price * c["qty"] for c in reg["catalog"]]
    assert [c["id"] for c in reg["catalog"]] == ["pololu_d36v50f12", "pololu_d42v110f12", "pololu_d42v110f12"]


def test_a_line_a_kept_lot_may_hold_is_priced_both_ways():
    """The regulator (against the power lot) and the supervisor (against the kept control lot): every total that
    holds one is also given without it, and what the Jetson adds is stated for the kit alone and with the
    regulator."""
    assert set(DOC["conditions"]) == {"regulator_12v", "supervisor"}
    assert budget_75.BLOCKS["jetson"]["conditional"] == ("regulator_12v",)
    assert budget_75.BLOCKS["jetson_direct_parts"]["conditional"] == ("supervisor",)
    s = DOC["scenarios"]
    assert "lower" not in s["record"] and "lower" not in s["pi_p21"] and "lower" not in s["pi_p21_optional"]
    for name, without in (("jetson_mcu_p21", ["regulator_12v"]), ("jetson_direct_p21", ["regulator_12v", "supervisor"]),
                          ("jetson_mcu_p21_optional", ["regulator_12v"])):
        lo = s[name]["lower"]
        assert lo["without"] == without
        for which, skip in (("turnkey", ()), ("self_assembled", ("integration",))):
            for i in range(3):
                sub = sum(v[i] for k, v in s[name]["lines"].items() if k not in skip and k not in without)
                assert lo[which]["subtotal"][i] == pytest.approx(sub, abs=0.011)
                assert lo[which]["total"][i] == pytest.approx(sub * (1 + cost.CONTINGENCY[i]), abs=0.51)
                assert lo[which]["total"][i] < s[name][which]["total"][i]
    # the kit alone is the catalog difference with the model's contingency: like for like with the Pi 5 set, whose
    # own regulator has no line either
    steps = DOC["derived"]["steps"]
    assert steps["jetson_kit_alone_on_pi"] == _with_contingency([109.0] * 3) == [115.0, 117.0, 118.0]
    assert DOC["blocks"]["jetson"]["lower"]["turnkey"] == steps["jetson_kit_alone_on_pi"]
    reg = DOC["blocks"]["jetson"]["lines"]["regulator_12v"]["usd"]
    assert steps["jetson_with_regulator_on_pi"] == _with_contingency([109.0 + r for r in reg])
    assert DOC["derived"]["regulator_upgrade_usd"] == pytest.approx(
        catalog.part("pololu_d42v110f12").price - catalog.part("pololu_d36v50f12").price)
    # the Pi 5 set's own 5 V regulator is not in its catalog price and has no catalog entry
    assert "5 V/5 A buck" in catalog.part(budget_75.PI5).notes
    assert not [p.id for p in catalog.parts().values() if p.category == "dcdc" and re.search(r"\b5 ?V\b", p.model)]
    # the direct option: without a separate supervisor it is the adapters alone
    direct = DOC["blocks"]["jetson_direct_parts"]
    assert direct["lower"]["without"] == ["supervisor"]
    sup = direct["lines"]["supervisor"]["usd"]
    for i in range(3):
        assert direct["subtotal"][i] - direct["lower"]["subtotal"][i] == pytest.approx(sup[i], abs=0.011)
    assert direct["lines"]["supervisor"]["caution"] and direct["lines"]["servo_bus_adapter"]["caution"]
    kept = {u["line"]: u["note"] for u in direct["unchanged"]}
    assert "overlap" in kept["control"] and "conservative" in kept["control"]


def test_the_computer_difference_is_a_range_read_from_the_catalog_and_the_survey():
    cd = DOC["derived"]["compute_difference"]
    jetson, pi = catalog.part(budget_75.JETSON), catalog.part(budget_75.PI5)
    assert cd["catalog_usd"]["difference"] == jetson.price - pi.price == 109
    # the Pi 5 set's catalog price already holds its storage; the dev kit's holds none
    assert "$175 + $5 + $70 + $40 = $290" in pi.notes and "USB 3 SSD" in pi.model
    survey = json.loads((ROOT / budget_75.SURVEY).read_text())["synthesis"]["architectures"]
    sets = {a["id"]: a for a in survey}
    a, c = sets[budget_75.SURVEY_SETS[budget_75.JETSON]], sets[budget_75.SURVEY_SETS[budget_75.PI5]]
    assert (a["price_usd"], c["price_usd"]) == (439, 290)
    assert any("NVMe SSD" in p and "$40" in p for p in a["parts"])
    assert any("12 V" in p and "not counted" in p for p in a["parts"])  # the survey: the 12 V converter exists already
    assert any("5 V/5 A" in p and "not priced" in p for p in c["parts"])
    assert cd["survey_set_usd"]["difference"] == 149 and cd["survey_set_usd"]["jetson_set_minus_kit"] == 40
    assert cd["survey_set_usd"]["pi5_set_equals_catalog"] is True
    compute_md = (ROOT / "COMPUTE.md").read_text()
    assert "$439" in compute_md and "$290" in compute_md
    assert cd["pi5_street_usd"] == 323.0 and "~$323 at observed street prices" in pi.notes
    for key, usd in cd["before_contingency"].items():
        assert cd["turnkey"][key] == _with_contingency(usd)
    assert cd["nominal_as_priced"] == [117.0, 192.0]
    assert cd["nominal_all_readings"] == [81.0, 235.0]
    # the record's compute line is the catalog price plus the allowance that also stands for storage
    assert RECORD["cost"]["lines"]["compute"] == [pi.price + 60, pi.price + 120, pi.price + 200]
    assert "NVMe" in (ROOT / "designlab" / "build.py").read_text()


def test_every_total_is_its_lines_plus_the_models_contingency():
    for name, s in DOC["scenarios"].items():
        assert "contingency" not in s["lines"]
        for which, skip in (("turnkey", ()), ("self_assembled", ("integration",))):
            for i in range(3):
                sub = sum(v[i] for k, v in s["lines"].items() if k not in skip)
                assert s[which]["subtotal"][i] == pytest.approx(sub, abs=0.011)
                assert s[which]["total"][i] == pytest.approx(sub * (1 + cost.CONTINGENCY[i]), abs=0.51)
        # self-assembled is turnkey less the integration line and the contingency on it
        for i in range(3):
            gap = s["turnkey"]["total"][i] - s["self_assembled"]["total"][i]
            assert gap == pytest.approx(s["lines"]["integration"][i] * (1 + cost.CONTINGENCY[i]), abs=1.01)
    assert DOC["contingency_fractions"] == list(cost.CONTINGENCY)


def test_every_low_is_at_most_its_nominal_and_its_high():
    def ordered(v, where):
        assert v[0] <= v[1] <= v[2], (where, v)

    for name, s in DOC["scenarios"].items():
        for key, usd in s["lines"].items():
            ordered(usd, (name, key))
        for which in ("turnkey", "self_assembled"):
            ordered(s[which]["total"], (name, which))
            ordered(s[which]["subtotal"], (name, which, "subtotal"))
            if "lower" in s:
                ordered(s["lower"][which]["total"], (name, "lower", which))
    for name, blk in DOC["blocks"].items():
        for key, line in blk["lines"].items():
            ordered(line["usd"], (name, key))
        ordered(blk["turnkey"], (name, "turnkey"))
        if "lower" in blk:
            ordered(blk["lower"]["turnkey"], (name, "lower"))
    for key, spec in budget_75.EXTRA.items():
        if "usd" in spec:
            ordered(spec["usd"], key)
    for key, usd in DOC["baseline"]["lines_recorded"].items():
        ordered(usd, key)
    ordered(DOC["derived"]["head_yaw_only"]["turnkey_with_bearing"], "head_yaw_only")


def test_the_blocks_add_up_to_the_scenarios():
    base = DOC["scenarios"]["record"]["turnkey"]
    for name, sc in budget_75.SCENARIOS.items():
        s = DOC["scenarios"][name]
        assert s["blocks"] == list(sc["blocks"])
        for i in range(3):
            sub = base["subtotal"][i] + sum(DOC["blocks"][b]["subtotal"][i] for b in sc["blocks"])
            assert s["turnkey"]["subtotal"][i] == pytest.approx(sub, abs=0.011)
            tot = base["total"][i] + sum(DOC["blocks"][b]["turnkey"][i] for b in sc["blocks"])
            assert s["turnkey"]["total"][i] == pytest.approx(tot, abs=len(sc["blocks"]) + 0.01)  # whole-dollar rounding
    assert DOC["checks"]["additivity"]["max_abs_diff_usd"] == 0.0
    # the direct option is the microcontroller option plus its own parts, and nothing else
    a, b = DOC["scenarios"]["jetson_mcu_p21"]["lines"], DOC["scenarios"]["jetson_direct_p21"]["lines"]
    assert set(b) - set(a) == {"usb_can", "supervisor", "servo_bus_adapter"}
    assert all(b[k] == a[k] for k in a)
    assert b["usb_can"] == [catalog.part("pcan_usb_fd").price * q for q in (1, 2, 2)]
    assert a["control"] == RECORD["cost"]["lines"]["control"]  # the microcontroller stack stays
    # the steps the prose quotes are differences of two printed totals, so its sums work to the dollar
    s, steps = DOC["scenarios"], DOC["derived"]["steps"]
    tk = {n: s[n]["turnkey"]["total"] for n in s}
    lo = {n: s[n]["lower"]["turnkey"]["total"] for n in s if "lower" in s[n]}
    for i in range(3):
        assert tk["record"][i] + steps["p21_on_the_record"][i] == tk["pi_p21"][i]
        assert tk["pi_p21"][i] + steps["jetson_kit_alone_on_pi"][i] == lo["jetson_mcu_p21"][i]
        assert tk["pi_p21"][i] + steps["jetson_with_regulator_on_pi"][i] == tk["jetson_mcu_p21"][i]
        assert lo["jetson_mcu_p21"][i] + steps["direct_on_jetson_lower"][i] == lo["jetson_direct_p21"][i]
        assert tk["jetson_mcu_p21"][i] + steps["direct_on_jetson"][i] == tk["jetson_direct_p21"][i]
        assert tk["pi_p21"][i] + steps["optional_on_pi"][i] == tk["pi_p21_optional"][i]
        # a block's own figure can be $1 off a step, because totals are rounded after summing
        assert abs(steps["direct_on_jetson"][i] - DOC["blocks"]["jetson_direct_parts"]["turnkey"][i]) <= 1.0
    assert DOC["derived"]["optional_together_turnkey"] == steps["optional_on_pi"] == steps["optional_on_jetson"]


def test_p21_items_are_not_counted_twice():
    ck = DOC["checks"]
    over = dict(map(tuple, RECORD["overrides"]))
    assert set(over) == {"toe_scale", "pad_scale"}  # P20's bigger feet only: no material or power lever
    assert ck["frame_6061"]["in_the_record"] is True and ck["frame_6061"]["thin_plate_lever_cost_equals_reference"]
    assert ck["covers_pa11"]["in_the_record"] is False and ck["covers_pa11"]["driver_matches"]
    assert ck["dc_input_and_boom"]["in_the_record"] is False
    assert "tether supply" in ck["dc_input_and_boom"]["lot_scope"] and "soles" in ck["pad_treads"]["lot_scope"]
    # cost.py gives the power lot no scope, so what it holds (a regulator, a clamp) cannot be checked: said, not hidden
    assert ck["dc_input_and_boom"]["power_lot_scope"] is None and ck["dc_input_and_boom"]["not_checked"]
    assert "power24" not in budget_75._ref_comments()
    p21 = DOC["blocks"]["p21"]["lines"]
    assert set(p21) == {"covers", "pad_treads", "dc_input", "cable_boom"}  # the frame and the battery do not move
    assert "clamp" in p21["dc_input"]["caution"] and "cable" in p21["dc_input"]["basis"]
    assert {u["line"] for u in DOC["blocks"]["p21"]["unchanged"]} == {"frame", "battery", "bench"}
    for key in ("pad_treads", "dc_input", "cable_boom"):
        assert key not in RECORD["cost"]["lines"] and p21[key]["new"] and p21[key]["kind"] == "d"
    # the PA11 re-pricing goes through cost.py's own material path, and reproduces the committed lever it copies
    lever = ck["covers_pa11"]["lever"]
    assert (lever["material"], lever["wall_scale"]) == ("PA11_MJF", 0.75)
    est = cost.estimate(budget_75.RecordDesign(RECORD, cover="PA11_MJF"),
                        budget_75._rows(RECORD, lever["mass_factor"]), RECORD["selection"])
    assert est["lines"]["covers"] == DOC["scenarios"]["pi_p21"]["lines"]["covers"]
    same = [k for k in est["lines"] if k not in ("covers", "contingency")]
    assert all(est["lines"][k] == RECORD["cost"]["lines"][k] for k in same)
    rule = ck["covers_pa11"]["rule_on_the_recorded_lever"]
    assert rule["max_abs_diff_usd"] <= 0.25
    assert max(abs(a - b) for a, b in zip(rule["totals_rule"], rule["totals_recorded"])) <= 1.0
    levers = {r["case"]: r for r in json.loads((ROOT / budget_75.LEVERS).read_text())}
    assert rule["totals_recorded"] == [levers["mat75_pa11"]["cost"][t] for t in TIERS]


def test_the_optional_lines_stay_out_of_the_main_totals():
    for name in budget_75.MAIN:
        lines = DOC["scenarios"][name]["lines"]
        assert "yaw_bearing" not in lines and "audio" not in lines
        assert lines["expression_actuators"] == RECORD["cost"]["lines"]["expression_actuators"]
    assert DOC["blocks"]["head_x"]["optional"] and DOC["blocks"]["audio"]["optional"]
    # the expressive head's servos and brackets are its committed record's, reproduced exactly
    assert DOC["checks"]["head_record"]["exact"]
    est = cost.estimate(budget_75.RecordDesign(HEAD), budget_75._rows(HEAD), HEAD["selection"])
    _same_cost(est, HEAD["cost"])
    hx = DOC["blocks"]["head_x"]["lines"]
    assert set(hx) == {"expression_actuators", "frame", "yaw_bearing"}
    servo = catalog.actuator("feetech_sts3215_c018").price
    assert HEAD["selection"]["neck_yaw"] == HEAD["selection"]["jaw"] == "feetech_sts3215_c018"
    assert hx["expression_actuators"]["usd"][:2] == [2 * servo, 2 * servo]
    # with the jaw deferred (P2, P29) the head is one servo and one bracket: an estimate, about half of the two
    yaw = DOC["derived"]["head_yaw_only"]
    assert [p["name"] for p in yaw["removed"]] == ["jaw_motor", "jaw_bracket"]
    assert yaw["lines"]["expression_actuators"][:2] == [servo, servo]
    for i in range(3):
        assert 0.45 * hx["frame"]["usd"][i] <= yaw["lines"]["frame"][i] <= 0.55 * hx["frame"]["usd"][i]
        assert yaw["turnkey_with_bearing"][i] < DOC["blocks"]["head_x"]["turnkey"][i]
    # every build with the optional lines is its main build plus the two optional blocks, and nothing else
    for main, optional in (("pi_p21", "pi_p21_optional"), ("jetson_mcu_p21", "jetson_mcu_p21_optional")):
        a, b = DOC["scenarios"][main]["lines"], DOC["scenarios"][optional]["lines"]
        assert set(b) - set(a) == {"yaw_bearing", "audio"}
        assert {k for k in a if a[k] != b[k]} == {"expression_actuators", "frame"}


def test_every_line_has_a_kind_and_the_lines_are_ranked_by_their_spread():
    for blk in DOC["blocks"].values():
        for key, line in blk["lines"].items():
            assert line["kind"] in DOC["kinds"] and line["basis"]
            if "catalog" in line:  # a catalog price with an assumed count or choice of part: never a plain (c)
                assert line["kind"] == "c+d"
                assert all(c["id"] in catalog.parts() for c in line["catalog"])
            elif key in budget_75.EXTRA:
                assert line["kind"] == "d"
    # the lines are ranked on the adopted build; the lines only a Jetson build adds are listed apart
    q = DOC["least_certain"]
    assert q["scenario"] == "pi_p21" and q["jetson_only_scenario"] == "jetson_direct_p21"
    rows = q["rows"]
    assert [r["spread"] for r in rows] == sorted((r["spread"] for r in rows), reverse=True)
    assert [r["line"] for r in rows[:3]] == ["frame", "covers", "integration"]
    assert {r["line"] for r in rows} == set(DOC["scenarios"]["pi_p21"]["lines"])
    assert {r["line"] for r in q["jetson_only_rows"]} == {"regulator_12v", "usb_can", "supervisor",
                                                          "servo_bus_adapter"}
    assert set(q["direct_only_lines"]) == {"usb_can", "supervisor", "servo_bus_adapter"}
    assert q["jetson_compute_line"] == DOC["scenarios"]["jetson_mcu_p21"]["lines"]["compute"]
    cls = {r["line"]: r["cls"] for r in rows}
    assert {k for k, v in cls.items() if v == "a8"} == {"frame", "covers", "integration", "harness", "foot_sensing"}
    assert {k for k, v in cls.items() if v == "carried"} == {"control", "power", "bench"}
    # what a quote would cover is a description of the line, not a step for anyone to take (decision-log P31)
    scopes = {r["line"]: r["quote_scope"] for r in rows + q["jetson_only_rows"]}
    assert all(scopes.values()) and set(scopes) == set(budget_75.QUOTE_SCOPE)
    for line, scope in scopes.items():
        assert not re.search(r"\b(ask|asks|asked|buy|decide|confirm|approve)\b", scope.lower()), (line, scope)
    # the clamp and the stop circuit are counted once, with the power bay
    assert "clamp" in scopes["power"] and "stop circuit" in scopes["power"] and "regulator" in scopes["power"]
    assert "belong to the power bay's price" in scopes["dc_input"]
    # the leg drives are the largest line only in the low case; the frame is at nominal and high
    assert DOC["derived"]["largest_lines"] == {"low": "leg_actuators", "nominal": "frame", "high": "frame"}
    # the scalings the table names are the ones cost.py applies
    drv, size = RECORD["cost"]["drivers"], RECORD["H"] / 0.39
    lines = RECORD["cost"]["lines"]
    assert lines["frame"] == [round(v * drv["frame_factor"], 2) for v in cost.REF["frame"]]
    assert lines["covers"] == [round(v * drv["cover_factor"], 2) for v in cost.REF["covers"]]
    assert lines["integration"] == [round(v * size ** 0.7, 2) for v in cost.REF["integration"]]
    assert lines["harness"] == [round(v * size, 2) for v in cost.REF["harness"]]
    assert lines["foot_sensing"] == [round(v * size ** 0.5, 2) for v in cost.REF["foot_sensing"]]
    assert lines["control"] == list(cost.REF["control"]) and lines["power"] == list(cost.REF["power24"])
    assert lines["bench"] == list(cost.REF["bench"])


def test_the_document_embeds_the_tables_and_quotes_the_record():
    text, prose, tables = _prose()
    assert tables == budget_75.md(DOC)
    s = DOC["scenarios"]
    for name in ("record", "pi_p21", "jetson_mcu_p21", "jetson_direct_p21"):
        for v in s[name]["turnkey"]["total"]:
            assert f"${v:,.0f}" in prose, (name, v)
    # the sums and shares the prose quotes are the record's (the prose is wrapped, so compare with spaces collapsed)
    dv = DOC["derived"]
    flat = " ".join(prose.split())

    def three(values):
        return " / ".join(f"${v:,.0f}" for v in values)

    for values in (*dv["steps"].values(), *dv["p21_lines_with_contingency"].values(),
                   dv["optional_together_turnkey"], dv["head_record_minus_baseline"], dv["self_assembled_gap_record"],
                   dv["regulator_with_contingency"], dv["supervisor_with_contingency"],
                   dv["head_yaw_only"]["turnkey_servo_and_bracket"], dv["head_yaw_only"]["turnkey_with_bearing"],
                   *dv["compute_difference"]["turnkey"].values(),
                   *(DOC["blocks"][b]["turnkey"] for b in ("p21", "head_x", "audio")),
                   *(s[n]["self_assembled"]["total"] for n in ("pi_p21", "jetson_mcu_p21")),
                   s["jetson_mcu_p21"]["lower"]["self_assembled"]["total"],
                   *(s[n]["turnkey"]["total"] for n in ("pi_p21_optional", "jetson_mcu_p21_optional")),
                   s["jetson_mcu_p21_optional"]["lower"]["turnkey"]["total"],
                   DOC["checks"]["head_record"]["totals"], s["pi_p21"]["lines"]["integration"],
                   *(budget_75.EXTRA[k]["usd"] for k in ("supervisor", "servo_bus_adapter", "pad_treads", "dc_input",
                                                         "cable_boom", "yaw_bearing", "audio"))):
        assert three(values) in flat, values
    for name in ("jetson_mcu_p21", "jetson_direct_p21"):  # each Jetson total both ways
        low, nom, high = s[name]["lower"]["turnkey"]["total"]
        assert f"${low:,.0f} low, ${nom:,.0f} nominal, ${high:,.0f} high" in flat, name
    for v in (dv["spread_top3"], dv["spread_all"], dv["spread_compute"], dv["baseline_subtotal_nominal"],
              dv["adopted_subtotal_nominal"], *dv["baseline_split_nominal"].values(),
              dv["adopted_split_nominal"]["a8"], *dv["compute_difference"]["nominal_as_priced"],
              dv["regulator_upgrade_usd"]):
        assert f"${v:,.0f}" in flat, v
    assert [round(100 * v) for v in dv["baseline_split_share"].values()] == [55, 8, 36]
    assert "second largest line at nominal after the frame" in flat  # the drives are not the largest line
    assert "In brief, for the owner" in prose and "Draft decision-log rows" in prose
    # the brief leads with the two nominal totals, in dollars, and says what kind of number they are
    brief = " ".join(prose.split("## In brief, for the owner")[1].split("\n## ")[0].split())
    pi, jetson = s["pi_p21"]["turnkey"]["total"], s["jetson_mcu_p21"]["lower"]["turnkey"]["total"]
    assert brief.startswith(f"One prototype of the 75 cm build is priced at **${pi[1]:,.0f}**")
    assert f"**${jetson[1]:,.0f}** with the Jetson dev kit" in brief and "US dollars" in brief
    for values in (pi, jetson, s["record"]["turnkey"]["total"], dv["steps"]["p21_on_the_record"],
                   dv["steps"]["jetson_kit_alone_on_pi"], dv["steps"]["jetson_with_regulator_on_pi"],
                   dv["steps"]["direct_on_jetson_lower"], dv["steps"]["direct_on_jetson"],
                   s["pi_p21"]["self_assembled"]["total"], s["jetson_mcu_p21"]["lower"]["self_assembled"]["total"],
                   DOC["blocks"]["head_x"]["turnkey"], DOC["blocks"]["audio"]["turnkey"]):
        assert three(values) in brief, values
    assert "a cost-model estimate (d) or a catalog price (c). None is a quote." in brief
    assert "nvidia_orin_nano_super_devkit" in prose and "$399" in prose
    for word in ("H is the hip height", "TPU is", "Shore A is", "sha256 is"):  # every abbreviation explained once
        assert word in flat, word
    assert max(len(line) for line in prose.splitlines() if not line.startswith("|")) <= 120  # tables cannot wrap


def test_the_document_names_no_total_of_pr668():
    """PR #668's (unverified) totals are not copied, in the prose or in the tables; its stale Jetson price appears
    only where it is corrected."""
    text, prose, tables = _prose()
    for stale in PR668_TOTALS:
        assert stale not in text, stale
    for line in tables.splitlines():
        if "$249" in line:
            assert line.startswith("| The Jetson dev kit's price"), line  # Table 6, the slip's own row
    flat = " ".join(prose.split())
    corrected = ("launch price of $249", "the $249 Jetson price", "from $249 to $399")
    assert flat.count("$249") == sum(flat.count(phrase) for phrase in corrected) == 3
    # the one equal figure is a coincidence, and the document says so
    assert DOC["scenarios"]["pi_p21"]["turnkey"]["total"][2] == 28215.0
    assert "happens to equal PR #668's (unverified) turnkey high" in " ".join(prose.split())


def test_no_total_is_set_against_p4():
    """Decision-log P31 (the owner, 14 Nov 2026): "remove any budget concerns at this point and just document the
    prices".  The document is a price record: a dated note at its very top lists what changed, and outside that note
    neither it, nor the record, nor the script weighs a total against P4, offers a line to cut or asks for a budget
    decision."""
    text = (RUN / "BUDGET_75.md").read_text()
    note, rest = _note_and_rest()
    # the note is the first thing under the title, and it names the decision and the owner's words
    assert text.split("\n\n")[1].startswith("**Changed, 14 Nov 2026 (P31).**")
    flat_note = " ".join(note.split())
    assert "remove any budget concerns at this point and just document the prices" in flat_note
    for place in ("Title", "In brief, for the owner", "Table 4", "Table 5", "Table 6", "Table 7", "Section 4",
                  "Section 6", "draft decision-log rows", "`budget_75.json`"):
        assert place in flat_note, place
    # P4 is named once outside the note, to say that it is not applied
    flat = " ".join(rest.split())
    assert flat.count("P4") == 1
    assert ("P4's $10-20k is not applied at this point (decision-log row P31), so this document sets no total "
            "against it.") in flat
    assert flat.index("P4's $10-20k is not applied") < flat.index("## In brief, for the owner")
    # no verdict against P4, no option to cut a line, no request for a budget decision
    record_text = (RUN / "budget_75.json").read_text()
    script_text = (RUN / "budget_75.py").read_text()
    verdicts = ("over p4", "under p4", "above p4", "below p4", "trim", "ceiling", "$20,000", "$20k", "exceed",
                "afford", "over budget", "within budget", "prototype budget", "what to do with this",
                "the decision this prepares", "owner's decision)", "awaiting the owner", "accept about", "defer the")
    for where, body in (("document", rest), ("tables", budget_75.md(DOC)), ("record", record_text),
                        ("script", script_text)):
        low = " ".join(body.split()).lower()
        for word in verdicts:
            assert word not in low, (where, word)
        # nor a dollar amount said to be over or under something
        assert not re.search(r"\$[\d,.]+k? (over|above|under|below)\b", low), where
    assert DOC["p31"]["row"] == "P31" and "not weighed against P4" in DOC["p31"]["applied_as"]
    assert "p4" not in DOC and "trims" not in DOC and "quotes" not in DOC
    for name, s in DOC["scenarios"].items():
        assert set(s) <= {"blocks", "compute", "cover", "lines", "extra_lines", "turnkey", "self_assembled", "label",
                          "vs_record", "lower"}, name
        if "lower" in s:
            assert set(s["lower"]) == {"without", "condition", "turnkey", "self_assembled"}, name
    assert not [k for k in DOC["derived"] if "ceiling" in k or "p4" in k.lower()]
    # every slip that is left is a price or a catalog id
    assert [r["what"] for r in DOC["pr668_slips"]] == [
        "The Jetson dev kit's price", "The tail servo's catalog id", "The neck servo's catalog id"]
    # the totals table has the three cases, turnkey and self-assembled, and no column against a figure
    tables = budget_75.md(DOC)
    header = next(line for line in tables.splitlines() if line.startswith("| Build |"))
    assert header == ("| Build | Turnkey low | Turnkey nominal | Turnkey high | Self-assembled low | "
                      "Self-assembled nominal | Self-assembled high |")
    assert [line[4:] for line in tables.splitlines() if line.startswith("### Table")] == [
        "Table 1. The round-two record's cost block, recomputed",
        "Table 2. The variants, each as a difference from the record",
        "Table 3. What the record already holds of P21's items",
        "Table 4. Totals: turnkey and self-assembled",
        "Table 5. Where the prices are least certain, the widest lines first",
        "Table 6. PR #668's price slips, corrected"]
    # the draft rows record prices and assumptions; none proposes a budget decision
    drafts = [line.split("|")[1].strip() for line in rest.split("Draft decision-log rows")[1].splitlines()
              if line.startswith("|") and not line.startswith(("| Draft", "|---"))]
    assert drafts == ["Price finding", "Lines outside the cost model", "A8, made explicit"]
