"""Gate L1's freeze record (``designlab/stage2/l1freeze.py``): it is the committed look records, restated; nothing in
it is computed anew."""

import json

import pytest
from designlab.stage2 import l1freeze


def test_the_record_regenerates_from_the_committed_look_records():
    assert l1freeze.RECORD.read_text() == l1freeze.text(l1freeze.build())


def test_every_rung_is_v6_with_the_ladders_hash():
    doc = json.loads(l1freeze.RECORD.read_text())
    refs = json.loads((l1freeze.LOOK / "look_refs.json").read_text())
    assert sorted(doc["rungs"]) == sorted(refs["bodies"]) == ["60-N-T-10", "75-M-H-10"]
    for body, speeds in doc["rungs"].items():
        assert list(speeds) == list(refs["speeds"])
        for speed, r in speeds.items():
            assert r["variant"] == "V6"
            assert r["sha256"] == refs["bodies"][body][speed]["V6"]["sha256"]
            assert r["in_ladder"] is r["within_limits"]


def test_the_owner_decisions_it_carries():
    doc = json.loads(l1freeze.RECORD.read_text())
    d = doc["decided"]
    assert (d["pick"], d["decision_log"], d["date"]) == ("V6", "P30", "2026-11-14")
    assert d["feature_may_cost_a_speed_rung"] is True
    assert d["duty_band"] == {"below_0.30_m_s": [0.55, 0.72], "from_0.30_m_s": [0.55, 0.65]}


def test_the_one_rung_over_a_limit_is_out_of_the_ladder_and_every_duty_is_in_its_band():
    doc = json.loads(l1freeze.RECORD.read_text())
    assert doc["not_in_ladder"] == ["75-M-H-10 fr05"]
    over = doc["rungs"]["75-M-H-10"]["fr05"]["over"]
    assert [o["limit"] for o in over] == ["knee speed"] and 1.0 < over[0]["value"] < 1.02
    band = doc["decided"]["duty_band"]
    for speeds in doc["rungs"].values():
        for r in speeds.values():
            lo, hi = band["below_0.30_m_s"] if r["speed_m_s"] < 0.30 else band["from_0.30_m_s"]
            assert lo <= r["duty"] <= hi, r


def test_arrays_on_disk_have_the_frozen_hashes():
    problems = l1freeze.check()
    if any(p.startswith("missing array") for p in problems):
        pytest.skip("the look ladder's arrays are not restored here (tools/restore_inputs.py)")
    assert problems == []
