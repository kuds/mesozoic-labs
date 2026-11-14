"""Tests for the 75 cm track's latency budget, bus plan and delay bench sheet (``stage2/jetson75/latency_bus.py``,
``latency_bus.json``, ``LATENCY_AND_BUS.md``, ``BENCH_SHEET.md``; ``JETSON_PLEM_REVIEW.md`` section 5, tasks 7 and 8).

* The committed ``latency_bus.json`` regenerates from the committed script exactly, and names its script's sha256.
* The bus-load arithmetic is recomputed here from the frame's fields, without the script's functions, for the Teensy
  with two buses (A), with three (A3) and for the one-bus Jetson layouts (C1, D); the worst-case stuffing is checked
  with a bit-stuffing simulation.
* Every input and every delay term has a kind in {a, b, c, d}; every (c) names sources that exist; every (d) says
  whose estimate it is and how it was derived; every study source is a file of the repository.  A rate or a bus term
  that rests on an assumption carries (d): EtherCAT's rate, the 250 Hz of the one-bus layouts, the order of frames.
* The totals, the feedback ages and the simulated delay they map onto are recomputed from the terms for two
  architectures.
* The model's file and line references point at lines that hold the quoted text.
* The ledger rows equal Stage 1's records, and the bench parts' prices equal the catalog's; the bench lot quoted is
  the round-two record's; run 1's parts can carry the rows the sheet asks of it (two USB-CAN channels, a transceiver
  for every bus on both Teensys, a level-shifting buffer for each servo bus on the Teensy).
* The delay steps' bench gains, the stall torque behind a supply's current limit, the neck bus at its factory bit
  rate and the statistics of "all of 50" are recomputed here.
* The tables embedded in the two documents are the script's, and the figures of each "In brief" and of the sheet's
  hand-written sentences are the record's.

Run from hardware/trex_form_study:  python -m pytest -p no:cacheprovider tests/test_stage2_latency_bus.py -q
About a second: no MuJoCo, no simulation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "stage2" / "jetson75"
sys.path.insert(0, str(RUN))

import latency_bus  # noqa: E402

DOC = json.loads((RUN / "latency_bus.json").read_text())
BUDGET = (RUN / "LATENCY_AND_BUS.md").read_text()
SHEET = (RUN / "BENCH_SHEET.md").read_text()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ the record
def test_the_record_regenerates_from_the_script_exactly():
    assert DOC["script_sha256"] == _sha(RUN / "latency_bus.py"), "stale: run python stage2/jetson75/latency_bus.py"
    assert latency_bus.dumps(latency_bus.build()) == (RUN / "latency_bus.json").read_text()
    assert latency_bus.check(latency_bus.build()) == []


def test_the_record_says_what_it_is():
    assert DOC["body"] == "75-M-H-10"
    assert "nothing here is simulated anew, built or measured" in DOC["status"]
    assert set(DOC["kinds"]) == set("abcd")
    by_part = DOC["drives"]["by_part"]
    assert {k: v["count"] for k, v in by_part.items()} == {
        "myactuator_rmd_x6_p20_60_e": 2, "myactuator_rmd_x4_p36_36_e_l": 10, "robotis_xm540_w270_t": 1,
        "feetech_sts3215_c018": 1}
    assert by_part["myactuator_rmd_x6_p20_60_e"]["joints"] == ["hip_pitch"]


# ------------------------------------------------------------------------------------------------ kinds and sources
def test_every_input_and_term_has_a_kind_and_every_sourced_fact_a_source():
    sources = DOC["sources"]
    entries = dict(DOC["inputs"])
    entries.update({f"term:{k}": v for k, v in DOC["terms"].items()})
    assert len(DOC["inputs"]) >= 40 and len(DOC["terms"]) == 10
    for name, e in entries.items():
        assert e["kind"] in ("a", "b", "c", "d"), name
        assert all(s in sources for s in e["source"]), name
        if e["kind"] == "c":
            assert e["source"], f"{name}: a sourced fact without a source"
        if e["kind"] == "d":
            assert len(e["basis"]) > 20, f"{name}: an estimate without its basis"
            assert "author" in e["basis"] or "disagree" in e["basis"], f"{name}: whose estimate?"
    for a in DOC["architectures"].values():
        for t in a["terms"].values():
            assert t["kind"] in ("a", "b", "c", "d") and t["basis"]
        for block in ("tick_to_torque_ms", "sensor_age_ms", "total_ms", "maps_to"):
            assert a[block]["kind"] == "d", "every delay total is an estimate"


def test_an_assumption_is_never_labelled_a_sourced_fact():
    arch = DOC["architectures"]
    # EtherCAT's 500 Hz is assumed: the frame is an assumed layout and the drives' cycle was not found
    e = arch["E"]["rate_held"]
    assert e["kind"] == "d" and "assumed (d)" in e["text"] and "not found" in e["text"]
    for name in ("ethercat_process_bytes_per_drive", "ethercat_overhead_bytes", "ethercat_bit_rate_bps"):
        assert DOC["inputs"][name]["kind"] == "d"
    # the one-bus layouts: that 500 Hz does not fit is arithmetic (c); the 250 Hz is the author's choice (d)
    assert DOC["inputs"]["one_bus_rate_hz"]["kind"] == "d"
    for k in ("C1", "D"):
        r = arch[k]["rate_held"]
        assert r["kind"] == "d" and "does not fit** (c" in r["text"] and "250 Hz chosen (d)" in r["text"]
    # where 500 Hz fits, the claim is about the bus time only, and its basis names what was not found
    for k in ("A", "A3", "B", "C2"):
        r = arch[k]["rate_held"]
        assert r["kind"] == "c" and r["text"] == "500 Hz fits the bus (c)"
        assert "neither of which was found" in r["basis"]
    for name in ("drive_reply_turnaround_ms", "drive_max_command_rate_hz"):
        assert DOC["inputs"][name]["value"] is None and DOC["inputs"][name]["kind"] == "d"
    # the bus-transmit term rests on an assumed frame order
    for k, a in arch.items():
        t = a["terms"]["bus_transmit"]
        assert t["kind"] == "d", k
        if a["bus"] == "can":
            assert "sourced (c)" in t["basis"] and "assumption (d)" in t["basis"]
    table = latency_bus.block(DOC, "architectures")
    assert "(c: bus arithmetic)" not in table and "kind in each cell" in table
    assert "(c)" not in latency_bus.block(DOC, "terms"), "no delay term is a sourced fact"
    assert "nobody has measured" not in BUDGET + SHEET, "an absolute claim a search cannot support"


def test_every_source_is_a_file_of_the_study_or_a_web_page_with_what_it_says():
    for key, s in DOC["sources"].items():
        assert len(s["says"]) > 15, key
        if s["type"] == "web":
            assert all(u.startswith("https://") for u in [s["ref"]] + s.get("also", [])), key
        else:
            assert s["type"] in ("study", "catalog") and (ROOT / s["ref"]).is_file(), key
    assert "search tool" in DOC["web_read"] and "confirm" in DOC["web_read"]


# ------------------------------------------------------------------------------------------------ frames and bus load
def _stuff_bits(bits) -> int:
    """Stuff bits a CAN transmitter inserts: after five equal bits (stuff bits count) comes one opposite bit."""
    n, run, last = 0, 0, None
    for b in bits:
        run = run + 1 if b == last else 1
        last = b
        if run == 5:
            n += 1
            last, run = 1 - b, 1
    return n


def _classic_frame():
    """(bits of a classic CAN data frame with an 11-bit identifier and 8 data bytes, its stuffed part)."""
    stuffed = dict(sof=1, identifier=11, rtr=1, ide=1, r0=1, dlc=4, data=8 * 8, crc=15)
    rest = dict(crc_delimiter=1, ack=1, ack_delimiter=1, eof=7, intermission=3)
    return sum(stuffed.values()) + sum(rest.values()), sum(stuffed.values())


def test_the_classic_can_frame_is_recounted_field_by_field():
    plain, stuffed_span = _classic_frame()
    assert (plain, stuffed_span) == (111, 98)
    worst = [1] * 5 + [b for i in range(stuffed_span) for b in [i % 2] * 4]
    worst = worst[:stuffed_span]
    assert _stuff_bits(worst) == 24, "five equal bits, then runs of four: the worst case"
    rng = random.Random(0)
    assert max(_stuff_bits([rng.getrandbits(1) for _ in range(stuffed_span)]) for _ in range(2000)) <= 24
    c = DOC["frames"]["classic_can"]
    assert (c["bits_plain"], c["stuff_bits_max"], c["bits_stuffed"]) == (plain, 24, plain + 24)
    assert DOC["inputs"]["can_frame_bits_plain"]["value"] == plain
    assert DOC["inputs"]["can_frame_bits_stuffed"]["value"] == plain + 24
    rate = DOC["inputs"]["can_bit_rate_bps"]["value"]
    assert rate == 1_000_000
    assert c["us_plain"] == pytest.approx(plain / rate * 1e6) and c["us_stuffed"] == pytest.approx(135.0)


@pytest.mark.parametrize("arch, buses, rate_hz, load, fits_500", [
    ("A", 2, 500, (0.666, 0.810), True),
    ("A3", 3, 500, (0.444, 0.540), True),
    ("B", 2, 500, (0.666, 0.810), True),
    ("C1", 1, 250, (0.666, 0.810), False),
    ("C2", 2, 500, (0.666, 0.810), True),
    ("D", 1, 250, (0.666, 0.810), False),
])
def test_the_bus_load_is_recomputed_independently(arch, buses, rate_hz, load, fits_500):
    plain, _ = _classic_frame()
    drives, frames_per_drive, bit_s = 12, 2, 1e-6
    frames = frames_per_drive * drives // buses
    lo, hi = frames * plain * bit_s, frames * (plain + 24) * bit_s  # seconds of bus time per cycle
    a = DOC["architectures"][arch]
    row = a["bus_at_planned_rate"]
    assert (a["buses"], a["drives_on_fullest_bus"], row["frames_per_cycle"]) == (buses, drives // buses, frames)
    assert a["planned_rate_hz"] == rate_hz and row["cycle_ms"] == pytest.approx(1000.0 / rate_hz)
    assert row["ms_per_cycle_plain"] == pytest.approx(lo * 1e3, abs=1e-4)
    assert row["ms_per_cycle_stuffed"] == pytest.approx(hi * 1e3, abs=1e-4)
    assert row["load_plain"] == pytest.approx(lo * rate_hz, abs=1e-4)
    assert row["load_stuffed"] == pytest.approx(hi * rate_hz, abs=1e-4)
    assert (lo * rate_hz, hi * rate_hz) == pytest.approx(load, abs=1e-3), "the hand figures"
    assert a["holds_500_hz"] is fits_500 is (hi * 500 <= 1.0)
    assert a["load_at_500_hz"]["stuffed"] == pytest.approx(hi * 500, abs=1e-4)
    assert row["max_rate_hz_stuffed"] == pytest.approx(1.0 / hi, abs=0.06)
    assert row["max_rate_hz_plain"] == pytest.approx(1.0 / lo, abs=0.06)


def test_one_bus_cannot_carry_twelve_drives_at_500_hz_and_no_layout_carries_1000_hz():
    rows = {(r["buses"], r["rate_hz"]): r for r in DOC["bus_load"] if r["bus"].startswith("classic")}
    assert len(rows) == 9
    one = rows[(1, 500)]
    assert (one["frames_per_cycle"], one["fits"]) == (24, False)
    assert one["load_plain"] == pytest.approx(1.332) and one["load_stuffed"] == pytest.approx(1.62)
    assert one["max_rate_hz_stuffed"] == pytest.approx(308.6, abs=0.1)
    assert [rows[(b, 1000)]["fits"] for b in (1, 2, 3)] == [False, False, False]
    assert [rows[(b, 500)]["fits"] for b in (1, 2, 3)] == [False, True, True]
    assert all(r["fits"] == (r["ms_per_cycle_stuffed"] <= r["cycle_ms"]) for r in DOC["bus_load"])


def test_the_can_fd_and_ethercat_rows_are_labelled_hypothetical_estimates():
    fd, e = DOC["frames"]["can_fd_hypothetical"], DOC["frames"]["ethercat"]
    assert fd["kind"] == e["kind"] == "d" and fd["available_on_leg_drives"] is False
    assert (fd["arbitration"], fd["data"]) == (30, 96)
    assert fd["us_plain"] == pytest.approx(30 * 1.0 + 96 / 5.0)
    assert fd["us_stuffed"] == pytest.approx((30 + 4) * 1.0 + (96 + 17) / 5.0)
    assert e["bytes"] == 52 + 12 * 24 and e["us"] == pytest.approx(e["bytes"] * 8 / 100.0 + 12 * 1.0)
    assert DOC["inputs"]["ethercat_offered"]["value"] is True and DOC["inputs"]["ethercat_offered"]["kind"] == "c"
    assert DOC["architectures"]["E"]["bus_at_planned_rate"]["load_stuffed"] < 0.05


# ------------------------------------------------------------------------------------------------ the delay budget
def _ages(pre, frame, back, n_bus, cycle):
    """Feedback ages of the first and the last drive on a bus: sampled when its command arrives, used at the first
    tick after its reply is back (the commands go first, then the replies in the same order)."""
    out = []
    for i in (1, n_bus):
        sampled = pre + i * frame
        returned = pre + n_bus * frame + i * frame + back
        out.append(math.ceil(returned / cycle - 1e-9) * cycle - sampled)
    return out


@pytest.mark.parametrize("arch, expect", [
    ("A", dict(cmd=(0.311, 2.880), age=(1.234, 2.620), total=(0.311, 3.500), k=(1, 2), k_slow=3)),
    ("C2", dict(cmd=(0.451, 4.010), age=(1.094, 3.490), total=(0.451, 5.500), k=(1, 3), k_slow=4)),
])
def test_the_totals_are_the_sum_of_their_terms(arch, expect):
    a = DOC["architectures"][arch]
    t = a["terms"]
    names = ("compute", "host_jitter", "host_path", "bus_transmit", "drive", "rate_hold")
    cmd = [sum(t[n][end] for n in names) for end in ("lo_ms", "hi_ms")]
    n_bus, cycle = a["drives_on_fullest_bus"], a["planned_cycle_ms"]
    frame = (0.111, 0.135)
    assert t["bus_transmit"]["lo_ms"] == pytest.approx(frame[0]) and t["bus_transmit"]["hi_ms"] == n_bus * frame[1]
    assert t["rate_hold"]["hi_ms"] == 0.0
    ages = []
    for i, end in enumerate(("lo_ms", "hi_ms")):
        pre = sum(t[n][end] for n in ("compute", "host_jitter", "host_path"))
        ages += _ages(pre, frame[i], t["host_path"][end], n_bus, cycle)
    age = (min(ages), max(ages))
    total = [cmd[i] + max(0.0, age[i] - 2.0) for i in (0, 1)]
    for got, mine, hand in ((a["tick_to_torque_ms"], cmd, expect["cmd"]), (a["sensor_age_ms"], age, expect["age"]),
                            (a["total_ms"], total, expect["total"])):
        assert (got["lo"], got["hi"]) == pytest.approx(tuple(mine), abs=1e-3)
        assert tuple(mine) == pytest.approx(hand, abs=1e-3), "the hand figures"
    k = tuple(max(1, math.ceil(x / 2.0 - 1e-9)) for x in total)
    assert (a["maps_to"]["k_lo"], a["maps_to"]["k_hi"]) == k == expect["k"]
    assert (a["maps_to"]["delay_ms_lo"], a["maps_to"]["delay_ms_hi"]) == (2.0 * k[0], 2.0 * k[1])
    # the soft spot: a 500 Hz outer loop in the drive, 1 ms more at the upper end
    slow = a["if_drive_loop_500_hz"]
    assert slow["kind"] == "d" and slow["total_hi_ms"] == pytest.approx(total[1] + 1.0, abs=1e-3)
    assert slow["k_hi"] == math.ceil((total[1] + 1.0) / 2.0 - 1e-9) == expect["k_slow"]
    assert f"{slow['total_hi_ms']:.1f} ms (2–{slow['delay_ms_hi']:g} ms)" in BUDGET


def test_the_delay_maps_onto_whole_steps_rounded_up():
    assert [latency_bus.k_of(x) for x in (0.01, 2.0, 2.01, 4.0, 5.9, 6.0, 9.99, 10.0, 10.01)] == \
        [1, 1, 2, 2, 3, 3, 5, 5, 6]


def test_what_separates_the_architectures():
    arch = DOC["architectures"]
    assert list(arch) == ["A", "A3", "B", "C1", "C2", "D", "E"]
    # the host does not change the Teensy paths' budget
    for block in ("terms", "tick_to_torque_ms", "sensor_age_ms", "total_ms", "maps_to", "bus_at_planned_rate"):
        assert arch["A"][block] == arch["B"][block]
    assert arch["A"]["host"] != arch["B"]["host"] and arch["A"]["realtime_layer"] == arch["B"]["realtime_layer"]
    # two buses can overrun the 2 ms cycle at worst; three end inside it
    assert arch["A"]["exchange_fits_cycle"] is False and arch["A"]["exchange_end_hi_ms"] == pytest.approx(2.21)
    assert arch["A3"]["exchange_fits_cycle"] is True and arch["A3"]["sensor_age_beyond_model_ms"]["hi"] == 0.0
    # the one-bus layouts run below the model's target rate, and their k says so
    for k in ("C1", "D"):
        assert arch[k]["planned_rate_hz"] == 250 and arch[k]["maps_to"]["target_rate_matches_model"] is False
        assert arch[k]["terms"]["rate_hold"]["hi_ms"] == 2.0
    # only the microcontroller paths end a Linux crash in a controlled stop
    for k, a in arch.items():
        stop = a["failure"]["linux_crash"]["short"].startswith("Controlled stop")
        assert stop == (a["realtime_layer"] == "Teensy 4.1"), k
        assert "fall" in a["failure"]["linux_crash"]["detail"] or stop
    assert "unwritten" in arch["A"]["failure"]["linux_crash"]["detail"]
    assert "fall" in arch["A"]["failure"]["microcontroller_crash"]["detail"]


def test_the_neck_and_tail_buses():
    aux = DOC["aux"]
    assert aux["neck"]["part"] == "robotis_xm540_w270_t" and aux["tail"]["part"] == "feetech_sts3215_c018"
    for s, nbytes in ((aux["neck"], 16 + 11 + 14 + 21), (aux["tail"], 13 + 6 + 8 + 12)):
        assert s["kind"] == "d" and "TTL half-duplex" in s["catalog_interface"]
        wire_ms = nbytes * 10 / 1e6 * 1e3
        assert s["exchange_ms"]["lo"] == pytest.approx(wire_ms, abs=1e-4)
        assert s["exchange_ms"]["hi"] == pytest.approx(wire_ms + 2 * 0.5, abs=1e-4)
        assert s["own_bus_occupancy"]["hi"] == pytest.approx(s["exchange_ms"]["hi"] * 1e-3 * 100, abs=1e-4)
    # the neck servo as delivered: 57,600 bit/s, at which a write and a read do not fit the 10 ms period of 100 Hz
    neck, tail = aux["neck"], aux["tail"]
    assert neck["factory_baud_bps"] == DOC["inputs"]["xm540_baud_default_bps"]["value"] == 57_600
    ms = (16 + 11 + 14 + 21) * 10 / 57_600 * 1e3
    assert neck["exchange_at_factory_baud_ms"]["lo"] == pytest.approx(ms, abs=0.006) and ms > 10.0
    assert neck["exchange_at_factory_baud_ms"]["hi"] == pytest.approx(ms + 1.0, abs=0.006)
    assert neck["fits_period_at_factory_baud"] is False and tail["fits_period_at_factory_baud"] is True
    quoted = f"{neck['exchange_at_factory_baud_ms']['lo']:.1f}–{neck['exchange_at_factory_baud_ms']['hi']:.1f} ms"
    assert quoted == "10.8–11.8 ms" and quoted in BUDGET and quoted in SHEET
    assert "57,600 bit/s" in BUDGET and "57,600 bit/s" in SHEET and "value 3" in SHEET


# ------------------------------------------------------------------------------------------------ the model and records
def test_the_model_references_point_at_the_quoted_lines():
    ids = [m["id"] for m in DOC["model"]]
    assert {"queue", "ledger_queue", "range", "sensor_age", "feedback", "gains", "passive", "n8"} <= set(ids)
    n = 0
    for m in DOC["model"]:
        for r in m["refs"]:
            line = (ROOT / r["file"]).read_text().splitlines()[r["line"] - 1]
            assert r["text"] in line, f"{r['file']}:{r['line']} moved: run latency_bus.py sync"
            n += 1
    assert n >= 22
    from_code = {m["id"]: m for m in DOC["model"]}
    # the ledger's readings were made by the ledger's own queue, and the range is the ledger's as well as the env's
    assert from_code["ledger_queue"]["refs"][0]["file"] == "designlab/stage1/ledger.py"
    assert [r["file"] for r in from_code["range"]["refs"]] == ["designlab/stage1/env.py", "designlab/stage1/ledger.py"]
    assert "the leg's, not the bare shaft's" in from_code["gains"]["says"]
    assert "queue[0]" in from_code["queue"]["refs"][2]["text"] and "0.002" in from_code["rate"]["refs"][1]["text"]
    assert DOC["inputs"]["model_step_ms"]["value"] == 2.0 and DOC["inputs"]["model_feedback_every_steps"]["value"] == 5


def test_the_ledger_rows_are_stage_1s_records():
    led = DOC["ledger"]
    assert led["kind"] == "b" and led["never_run_k"] == [1, 6, 7, 8, 9]
    for row, block in led["rows"].items():
        assert block["sha256"] == _sha(ROOT / block["record"])
        rec = json.loads((ROOT / block["record"]).read_text())
        assert rec["body"] == "75-M-H-10" and rec["row"] == row
        seen = {}
        for ev in rec["evaluations"].values():
            k = ev["variant"]["value"] or 0
            seen[str(k)] = (sum(not e["fell"] for e in ev["episodes"]), sum(e["succeeded"] for e in ev["episodes"]))
        assert {k: (v["up"], v["meets_every_limit"]) for k, v in block["by_k"].items()} == seen
        # the correction note's reading: every limit to 6 ms, up to 8 ms, falls at 10 ms, up again at 20 ms
        assert seen == {"0": (3, 3), "2": (3, 3), "3": (3, 3), "4": (3, 0), "5": (0, 0), "10": (3, 0)}
    assert led["stiffness"]["meets_every_limit_down_to"] == pytest.approx(0.7129, abs=1e-4)
    assert led["stiffness"]["stays_up_down_to"] == pytest.approx(0.5547, abs=1e-4)


def test_the_delay_map_never_reads_between_delays():
    rows = DOC["bench"]["delay_map"]
    assert [r["k"] for r in rows] == [1, 2, 3, 4, 5, None, 10, None]
    assert [r["simulated_delay_ms"] for r in rows] == [2, 4, 6, 8, 10, None, 20, None]
    assert rows[-1]["measured"] == "above 20 ms" and "outside the simulation's range" in rows[-1]["rerun"]
    for r in rows:
        ran = all(v is not None for v in r["model"].values())
        assert ran == (r["k"] in (2, 3, 4, 5, 10))
        assert (r["rerun"] is None) == (r["k"] is not None)
        if not ran:
            assert "simulated" in r["reading"]
        assert "rules out" not in r["reading"], "three episodes of one walk do not rule a controller out"
    assert "falls" in rows[4]["reading"] and "P29" in rows[4]["reading"] and "count against" in rows[4]["reading"]
    # the sheet's own words never read between steps either
    assert "rules out" not in SHEET and "at or above 10 ms" not in SHEET
    assert "a reading of 2.1 ms is run at 4 ms; it is not estimated from results at other delays" in SHEET
    assert "always that of its worst load condition" in SHEET
    assert "A 250 Hz reading cannot be mapped" in SHEET
    leg = [r for r in DOC["bench"]["results"]["rows"] if r["drive"].startswith("X")]
    assert sorted({r["drive"] for r in leg if r["target_rate_hz"] == 250}) == ["X4-P36-36-E-L", "X6-P20-60-E"]
    for line in latency_bus.block(DOC, "results").splitlines()[3:-1]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert (cells[-2:] == ["n/a", "n/a"]) == (cells[4] != "500"), "only a 500 Hz reading maps onto k"


# ------------------------------------------------------------------------------------------------ the bench tables
def test_the_bench_parts_are_priced_from_the_catalog():
    acts = {a["id"]: a for a in json.loads((ROOT / "designlab/data/actuators.json").read_text())["actuators"]}
    parts = {p["id"]: p for p in json.loads((ROOT / "designlab/data/parts.json").read_text())["parts"]}
    block = DOC["bench"]["parts"]
    for r in block["rows"]:
        if r["catalog_id"]:
            entry = acts.get(r["catalog_id"]) or parts[r["catalog_id"]]
            assert r["price_usd"] == entry["price_usd"] and r["kind"] == "c"
        else:
            assert r["kind"] in ("c", "d") and r["note"]
            assert r["kind"] == "d" or r["source"], "a priced part outside the catalog names its source"
    priced = [r for r in block["rows"] if r["price_usd"] is not None]
    for q, total, cat in (("run1_qty", "run1_total_usd", "run1_catalog_usd"),
                          ("run2_qty", "run2_total_usd", "run2_catalog_usd")):
        assert block[total] == pytest.approx(sum(r["price_usd"] * r[q] for r in priced), abs=0.005)
        assert block[cat] == pytest.approx(sum(r["price_usd"] * r[q] for r in priced if r["catalog_id"]), abs=0.005)
    assert block["leg_drives_usd"] == pytest.approx(2 * 428 + 10 * 269)
    kit = [r for r in block["rows"] if r["catalog_id"] == "nvidia_orin_nano_super_devkit"][0]
    assert (kit["run1_qty"], kit["run2_qty"]) == (0, 0) and "on hand" in kit["note"], "no compute purchase (P30)"
    assert not any(r["catalog_id"] == "rpi5_8gb_ai_hat_plus_13tops" for r in block["rows"])
    assert block["run1_catalog_usd"] == pytest.approx(1997.79) and block["run2_catalog_usd"] == pytest.approx(4866.74)
    assert block["run1_share_of_run2"] == pytest.approx(block["run1_total_usd"] / block["run2_total_usd"], abs=1e-3)
    assert f"({block['run1_share_of_run2']:.2f} of it)" in SHEET and block["run1_share_of_run2"] < 0.5


def _part(fragment: str) -> dict:
    hits = [r for r in DOC["bench"]["parts"]["rows"] if fragment in r["what"]]
    assert len(hits) == 1, fragment
    return hits[0]


def test_the_bench_lot_quoted_is_this_bodys():
    record = json.loads((ROOT / "round2/refine/feet_75/candidates/75-M-H-10.json").read_text())
    assert record["id"] == "75-M-H-10" and record["cost"]["lines"]["bench"] == [550, 850, 1300]
    lot = DOC["bench"]["parts"]["bench_lot_usd"]
    assert (lot["low"], lot["target"], lot["high"]) == (550, 850, 1300) and lot["kind"] == "c"
    supply = _part("24 V bench supply")
    assert supply["price_usd"] is None and "$550 / 850 / 1,300" in supply["note"]
    assert supply["what"] in DOC["bench"]["parts"]["unpriced"]
    assert "$900" not in SHEET and "2,200" not in SHEET, "that lot is for bodies whose H is above 0.6 m"
    plant = json.loads((ROOT / "stage1/runs/2026-11-08/75-M-H-10/plant.json").read_text())
    assert plant["H_m"] < 0.6
    assert "not priced" in _brief(SHEET) and "bench supply" in _brief(SHEET)


def test_run_1s_parts_can_carry_the_rows_the_sheet_asks_of_it():
    res = DOC["bench"]["results"]
    assert len(res["rows"]) == 16 and res["leg_rows"] == 12 and len(res["soak"]) == 5 and res["commands"] == 1000
    # path 2 at 500 Hz needs two buses, so two one-channel adapters in run 1 as in run 2
    assert DOC["inputs"]["pcan_usb_fd_channels"]["value"] == 1
    adapters = [r for r in DOC["bench"]["parts"]["rows"] if r["catalog_id"] == "pcan_usb_fd"][0]
    buses_path2 = max(int(r["buses"]) for r in res["rows"] if r["path"].startswith("2") and r["buses"] != "own"
                      and r["target_rate_hz"] == 500)
    assert buses_path2 == 2 and adapters["run1_qty"] == adapters["run2_qty"] == buses_path2
    assert "for the owner to confirm" in adapters["note"]
    assert DOC["bench"]["parts"]["usb_can_adapters_usd"] == pytest.approx(2 * 368)
    # path 1 goes to three buses: the loop Teensy and the stopwatch each need a transceiver per bus
    buses_path1 = max(int(r["buses"]) for r in res["rows"] if r["path"].startswith("1") and r["buses"] != "own")
    assert buses_path1 == 3 == DOC["inputs"]["teensy_can_buses"]["value"]
    assert _part("transceiver")["run1_qty"] == _part("transceiver")["run2_qty"] == 2 * buses_path1
    # the neck and tail servos on both paths: a USB adapter each for path 2, a level-shifting buffer each for path 1
    assert DOC["inputs"]["teensy_pin_max_v"]["value"] == 3.3 < DOC["inputs"]["dxl_ttl_logic_v"]["value"] == 5.0
    assert _part("U2D2")["run1_qty"] == 1 and _part("FE-URT-1")["run1_qty"] == 1
    assert _part("level-shifting")["run1_qty"] == 2 and " or a half-duplex buffer" not in SHEET
    assert SHEET.count("Never wire a servo's data line straight to a Teensy pin") == 2, "the set-up and the safety list"
    # the Teensy's price is the listing that could be read
    teensy = _part("Teensy 4.1: one runs")
    assert teensy["price_usd"] == 31.50 and teensy["kind"] == "c" and "teensy41_price" in teensy["source"]
    assert "$31.50" in DOC["sources"]["teensy41_price"]["says"] and "35.50" not in latency_bus.block(DOC, "bench_parts")
    # the things the steps need are on the list
    for fragment in ("set-up software", "piezo disc", "dial indicator", "D-Sub 9", "tread sample", "emergency-stop"):
        assert _part(fragment)["run1_qty"] >= 1
    assert all(r["run1_qty"] <= r["run2_qty"] for r in DOC["bench"]["parts"]["rows"])


def test_the_delay_steps_bench_gains_are_recomputed():
    plant = json.loads((ROOT / "stage1/runs/2026-11-08/75-M-H-10/plant.json").read_text())
    acts = {a["id"]: a for a in json.loads((ROOT / "designlab/data/actuators.json").read_text())["actuators"]}
    rows = {g["part"]: g for g in DOC["bench"]["step_gains"]}
    assert set(rows) == {"myactuator_rmd_x6_p20_60_e", "myactuator_rmd_x4_p36_36_e_l"}
    hand = {"myactuator_rmd_x6_p20_60_e": dict(kp=1140.0, kv=4.43, ratio=(6.9, 6.9), stall=18.6),
            "myactuator_rmd_x4_p36_36_e_l": dict(kp=598.5, kv=2.79, ratio=(2.7, 8.8), stall=16.1)}
    for part, g in rows.items():
        assert g["kind"] == "d" and "author" in g["basis"]
        gains = [plant["gains"]["drives"][d["actuator"]] for d in plant["drives"]
                 if d["part"] == part and not d["group"].startswith("ankle")]
        kp, inertia = g["kp"], acts[part]["armature_kg_m2"]
        assert all(x["kp"] == pytest.approx(kp, abs=0.01) for x in gains) and kp == hand[part]["kp"]
        kv = 2.0 * math.sqrt(kp * inertia)  # critical damping on the bare shaft
        assert g["bench_kv"] == pytest.approx(kv, abs=0.006) and kv == pytest.approx(hand[part]["kv"], abs=0.006)
        # the model's kv is sized on the leg's inertia: several times critical on the bare shaft
        model = sorted(x["kv"] for x in gains)
        got = (g["model_kv_over_bench_kv"]["lo"], g["model_kv_over_bench_kv"]["hi"])
        assert got == pytest.approx((model[0] / kv, model[-1] / kv), abs=0.06) and got == hand[part]["ratio"]
        assert min(x["inertia_kg_m2"] for x in gains) > 5 * inertia
        for x in gains:
            assert x["kv"] == pytest.approx(2.0 * math.sqrt(x["kp"] * x["inertia_kg_m2"]), rel=1e-6)
        # the step: x0 (1 - (1 + w t) e^(-w t)) of travel after t, with w = sqrt(kp / I)
        w = math.sqrt(kp / inertia)
        assert g["time_constant_ms"] == pytest.approx(1e3 / w, abs=0.006)
        t = g["frame_ms"] / 1e3
        share = 1.0 - (1.0 + w * t) * math.exp(-w * t)
        assert g["tip_mm_in_one_frame"] == pytest.approx(share * 0.01 * 0.20 * 1e3, abs=0.006)
        assert g["torque_demand_nm"] == pytest.approx(kp * 0.01, abs=0.006)
        assert g["torque_demand_nm"] == pytest.approx(g["sustained_nm"], abs=0.01), "the step asks the sustained torque"
        # a supply's current limit is not a torque limit: 2 A x 24 V as heat in three phase resistances
        amps = math.sqrt(2.0 * 24.0 / (3.0 * acts[part]["phase_resistance_ohm"]))
        assert g["stall_torque_at_supply_limit_nm"] == pytest.approx(acts[part]["kt_nm_per_a"] * amps, abs=0.06)
        assert g["stall_torque_at_supply_limit_nm"] == hand[part]["stall"] > g["sustained_nm"]
    x4, x6 = rows["myactuator_rmd_x4_p36_36_e_l"], rows["myactuator_rmd_x6_p20_60_e"]
    safety = SHEET.split("## 9. Safety")[1].split("\n## ")[0]
    assert f"about {x4['stall_torque_at_supply_limit_nm']:.0f} N·m on the X4" in safety
    assert f"{x6['stall_torque_at_supply_limit_nm']:.0f} N·m on the X6" in safety and "does not limit torque" in safety
    assert "80–93 N" in safety and (round(16.1 / 0.2), round(18.6 / 0.2)) == (80, 93)
    assert "4.4 A and 2.7 A" in safety and "7.1 A together" in safety
    assert "3 to 9 times critical damping" in SHEET and "always in the same direction" in SHEET
    assert "will snap" not in SHEET and "alternating direction" not in SHEET


def test_the_statistics_the_sheet_claims():
    # 50 of 50 reads showing the change bound the true share only above 94 % at 95 % confidence; 300 of 300 above 99 %
    assert 0.05 ** (1 / 50) == pytest.approx(0.9418, abs=1e-4) and 0.05 ** (1 / 300) > 0.99
    assert "at least 94 % of the delays are shorter (95 % confidence)" in SHEET and "98 % level" not in SHEET
    assert "300 of 300 shows at least 99 %" in SHEET
    assert "Call a figure a p99\n   only with 300 or more trials" in SHEET
    # 1,000 steps in bins of 0.2 ms over 4 ms, 50 a bin
    assert 4.0 / 0.2 * 50 == DOC["bench"]["results"]["commands"]
    # the frames: a frame alone is 108-132 bit times; the spacing of back-to-back frames is 111-135
    plain, _ = _classic_frame()
    assert plain - 3 == 108 and "start to start: 111–135 µs" in SHEET and "3 bit times shorter" in SHEET
    assert (round(5 * 0.111, 2), round(5 * 0.135, 2)) == (0.56, 0.68) and SHEET.count("0.56–0.68 ms") == 2
    assert "0.56–0.68 ms" in BUDGET
    assert (round(2 * 0.111, 2), round(2 * 0.135, 2)) == (0.22, 0.27) and "(0.22–0.27 ms)" in SHEET
    frame = DOC["inputs"]["video_frame_ms"]["value"]
    assert frame == [pytest.approx(1e3 / 240, abs=0.006), pytest.approx(1e3 / 120, abs=0.006)]
    assert "4.2 ms at 240 frames a second, 8.3 ms at 120" in SHEET


def test_the_follow_on_sheets_figures():
    plant = json.loads((ROOT / "stage1/runs/2026-11-08/75-M-H-10/plant.json").read_text())
    acts = {a["id"]: a for a in json.loads((ROOT / "designlab/data/actuators.json").read_text())["actuators"]}
    # B: every constant speed is inside the drive's usable speed on the build's supply
    env = {e["part"]: e for e in plant["envelope"]["drives"].values()}
    assert len(DOC["bench"]["friction"]) == 4
    for r in DOC["bench"]["friction"]:
        assert max(r["speeds_rad_s"]) <= 0.9 * env[r["part"]]["rated_speed_rad_s"] and r["kind"] == "c"
        assert r["no_load_speed_rad_s"] == pytest.approx(env[r["part"]]["no_load_speed_rad_s"], abs=0.006)
    neck = [r for r in DOC["bench"]["friction"] if r["part"] == "robotis_xm540_w270_t"][0]
    assert neck["no_load_speed_rad_s"] == pytest.approx(30 * math.pi / 30, abs=0.006) and max(neck["speeds_rad_s"]) < 2
    assert "per ampere rms" in SHEET and "1.4-fold error" in SHEET
    # D: the prescribed walks' needs are the look screen's record (a)
    screen = json.loads((ROOT / "stage2/look/screen.json").read_text())
    grip = DOC["bench"]["grip"]
    assert grip["sha256"] == _sha(ROOT / "stage2/look/screen.json") and len(grip["rows"]) == 3
    for r in grip["rows"]:
        ss = screen["tail_phase"]["75-M-H-10"][r["speed"]]["walks"]["V0"]["single_support"]
        assert r["kind"] == "a" and r["peak_twist_nm"] == pytest.approx(ss["mz_abs_max_nm"], abs=0.006)
        assert r["friction_needed"] == pytest.approx(ss["mu_req_max"], abs=0.006)
    assert [r["friction_needed"] for r in grip["rows"]] == [0.36, 0.63, 0.79]
    assert grip["sample"]["mass_kg"] == pytest.approx(41e3 * 0.04 * 0.04 / 9.81, abs=0.006)
    assert grip["sample"]["mass_kg"] == pytest.approx(6.69, abs=0.006)
    assert "41 kPa" in (ROOT / "round2/refine/MATERIALS.md").read_text() and "41 kPa" in SHEET
    assert "under 6.7 kg" in SHEET and "above 0.63" in SHEET
    # E: the planning torque is 0.167 x the 12 V stall torque, not 0.57 x a vendor rating
    th = DOC["bench"]["thermal"]
    servo = acts["feetech_sts3215_c018"]
    assert "10 kg·cm" in servo["vendor_rated_definition"] and "30 kg·cm = 2.94 N·m" in servo["notes"]
    assert th["vendor_rated_nm"] == pytest.approx(10 * 0.0980665) and th["stall_nm"] == 2.94
    assert th["sustained_of_stall"] * th["stall_nm"] == pytest.approx(th["rows"][0]["torque_nm"], abs=1e-3)
    assert 0.57 * servo["rated_torque_nm"] == pytest.approx(th["rows"][0]["torque_nm"], abs=1e-3)
    assert "derived" in servo["basis"]["rated_torque"]
    assert [r["ratio_to_vendor_rated"] for r in th["rows"]] == [0.5, 0.53, 0.65]
    assert "0.57 x the vendor's rating" not in SHEET + (RUN / "latency_bus.py").read_text()
    assert "0.167 × the vendor's 12 V\nstall torque of 2.94 N·m" in SHEET and "0.98 N·m, 10 kg·cm at 0.9 A" in SHEET
    assert th["holds"] == 6 and "Six holds in all" in SHEET and "is safe" not in SHEET
    # the time table counts what the procedure asks for
    tm = DOC["bench"]["time"]
    soak = [r for r in tm["rows"] if r["what"].startswith("Soak")][0]
    assert tm["soak_hours"] == 5 * 30 / 60 <= soak["hours_lo"] and "5 rows of the soak table" in soak["what"]
    holds = [r for r in tm["rows"] if r["what"].startswith("E.")][0]
    assert tm["thermal_hold_hours"] == 6 * 20 / 60 <= holds["hours_lo"] and "6 holds of 20 min" in holds["what"]
    assert "unattended" not in holds["what"] and "attended" in holds["what"]
    assert "16 rows of the results table" in [r for r in tm["rows"] if r["what"].startswith("Delay runs")][0]["what"]
    assert "Soak, once per row of the soak table" in SHEET


def test_the_holding_gains_are_the_plants():
    plant = json.loads((ROOT / "stage1/runs/2026-11-08/75-M-H-10/plant.json").read_text())
    assert DOC["drives"]["sha256"] == _sha(ROOT / "stage1/runs/2026-11-08/75-M-H-10/plant.json")
    rows = DOC["bench"]["gains"]
    assert sum(g["count"] for g in rows) == 14 == len(plant["drives"])
    for g in rows:
        dr = [d for d in plant["drives"] if d["group"] == g["group"]][0]
        gains = plant["gains"]["drives"][dr["actuator"]]
        assert g["part"] == dr["part"] and g["kp"] == pytest.approx(gains["kp"], abs=0.005)
        assert g["kv"] == pytest.approx(gains["kv"], abs=0.005)
        # the rule: the sustained planning torque at 0.01 rad (each ankle motor at half the joint's stiffness)
        per_joint = 2.0 if g["group"].startswith("ankle") else 1.0
        assert g["kp"] * per_joint == pytest.approx(dr["sustained"] / 0.01, rel=1e-4)
        assert g["mass_kg_on_lever"] == pytest.approx(dr["sustained"] / (9.81 * 0.20), abs=0.005)
    th = DOC["bench"]["thermal"]
    assert th["part"] == "feetech_sts3215_c018" and [r["torque_nm"] for r in th["rows"]] == [0.4913, 0.52, 0.64]
    assert [r["kind"] for r in th["rows"]] == ["c", "b", "b"]
    tm = DOC["bench"]["time"]
    for g in ("run1", "run2", "follow"):
        assert tm["totals"][g]["lo"] == pytest.approx(sum(r["hours_lo"] for r in tm["rows"] if r["group"] == g))
        assert tm["totals"][g]["hi"] == pytest.approx(sum(r["hours_hi"] for r in tm["rows"] if r["group"] == g))
    assert all(r["kind"] == "d" for r in tm["rows"])


# ------------------------------------------------------------------------------------------------ the documents
@pytest.mark.parametrize("text, names", [
    (BUDGET, ["frames", "bus_load", "architectures", "terms", "term_basis", "model", "ledger", "aux", "failure",
              "sources"]),
    (SHEET, ["bench_parts", "step_gains", "results", "soak", "delay_map", "ledger", "bench_gains", "friction",
             "grip", "thermal", "time"]),
])
def test_the_documents_tables_are_the_scripts(text, names):
    found = latency_bus.embedded(text)
    assert list(found) == names
    for name, block in found.items():
        assert block == latency_bus.block(DOC, name), f"table {name} is stale: run latency_bus.py sync"
        assert len(block.splitlines()) >= 5
    assert set(latency_bus.TABLES) == set(latency_bus.embedded(BUDGET)) | set(latency_bus.embedded(SHEET))


def _brief(text: str) -> str:
    return text.split("## In brief, for the owner")[1].split("\n## ")[0]


def test_the_budgets_brief_states_the_records_figures():
    brief, arch = _brief(BUDGET), DOC["architectures"]
    rows = {(r["buses"], r["rate_hz"]): r for r in DOC["bus_load"] if r["bus"].startswith("classic")}
    for buses in (1, 2, 3):
        r = rows[(buses, 500)]
        assert f"{100 * r['load_plain']:.0f}–{100 * r['load_stuffed']:.0f} %" in brief
    one = rows[(1, 500)]
    assert f"{one['ms_per_cycle_plain']:.2f}–{one['ms_per_cycle_stuffed']:.2f} ms" in brief
    assert f"{one['max_rate_hz_stuffed']:.0f}–{one['max_rate_hz_plain']:.0f} Hz" in brief
    for k in ("A", "C2", "C1", "D", "E"):
        a = arch[k]
        assert f"{a['total_ms']['lo']:.1f}–{a['total_ms']['hi']:.1f} ms" in brief, k
        assert f"{a['maps_to']['delay_ms_lo']:g}–{a['maps_to']['delay_ms_hi']:g} ms" in brief, k
    assert f"{arch['A']['exchange_end_hi_ms']:.1f} ms" in brief
    assert "meets every limit to 6 ms" in brief and "falls at 10 ms" in brief and "2 ms was never run" in brief
    assert "PR #668's layout\n     (unverified)" in brief or "PR #668's layout (unverified)" in brief
    assert "Only the bench test can\n   replace them" in brief
    assert "Every delay term except the bus arithmetic" not in brief
    assert "the order of\n   the frames on the bus" in brief
    slow_a, slow_c2 = arch["A"]["if_drive_loop_500_hz"], arch["C2"]["if_drive_loop_500_hz"]
    assert f"(A, B) would map onto 2–{slow_a['delay_ms_hi']:g} ms and C2 onto 2–{slow_c2['delay_ms_hi']:g} ms" in brief
    assert "were not found (d)" in brief
    # the 8 ms figure's provenance is the survey's record of PR #662's platform, not the sim-to-real plan
    assert "sim-to-real plan" not in BUDGET and "PR #662's 60 cm platform" in BUDGET
    survey = (ROOT / "designlab/data/research/compute_2026-11-07.json").read_text()
    assert "position servos with an 8 ms command delay and 50 Hz zero-order hold" in survey


def test_the_sheets_brief_states_the_records_figures():
    brief, parts, tm = _brief(SHEET), DOC["bench"]["parts"], DOC["bench"]["time"]["totals"]
    for key in ("run1_catalog_usd", "run1_total_usd", "run2_catalog_usd", "run2_total_usd"):
        assert f"${parts[key]:,.0f}" in brief, key
    for g in ("run1", "run2", "follow"):
        assert f"{tm[g]['lo']:g}–{tm[g]['hi']:g} hours" in brief, g
    a, c2 = DOC["architectures"]["A"]["maps_to"], DOC["architectures"]["C2"]["maps_to"]
    assert f"{a['delay_ms_lo']:g}–{a['delay_ms_hi']:g} ms" in brief
    assert f"{c2['delay_ms_lo']:g}–{c2['delay_ms_hi']:g} ms" in brief
    assert "Never read between steps" in brief and "1,000 commands" in brief
    assert f"${parts['leg_drives_usd']:,.0f} is the 12 leg drives" in brief
    assert f"two USB-CAN adapters (${parts['usb_can_adapters_usd']:,.0f})" in brief and "for you to confirm" in brief
    slow_a = DOC["architectures"]["A"]["if_drive_loop_500_hz"]
    slow_c2 = DOC["architectures"]["C2"]["if_drive_loop_500_hz"]
    assert f"at 2–{slow_a['delay_ms_hi']:g} and 2–{slow_c2['delay_ms_hi']:g} ms" in brief
    assert "worst load condition" in brief and "compute step" in brief
    # H is defined so that two people measure the same thing, and it does not claim the compute step
    assert "It contains the\n    compute step" not in SHEET
    assert "the loop program of\n  this bench computes nothing" in SHEET
    assert "the end of the last command frame on the bus − the tick" in SHEET and "fixed, written order" in SHEET
    assert "the stopwatch never sends on a bus under\n  test" in SHEET
    assert "Loaded means all of these at once" in SHEET and "any continuous GPU benchmark" not in SHEET
    for drive in ("X6-P20-60-E", "X4-P36-36-E-L", "XM540-W270", "STS3215"):
        assert drive in SHEET
    for method in ("M1, coarse", "M1, fine", "M2", "M3"):
        assert f"**{method}**" in SHEET


@pytest.mark.parametrize("name, text", [("LATENCY_AND_BUS.md", BUDGET), ("BENCH_SHEET.md", SHEET)])
def test_the_documents_keep_the_studys_rules(name, text):
    assert "## In brief, for the owner" in text and "Draft" in text
    assert "Nothing is adopted here" in text
    for kind in ("(a)", "(b)", "(c)", "(d)"):
        assert kind in text
    for n, line in enumerate(text.splitlines(), 1):
        if not line.startswith(("|", "<!--")):
            assert len(line) <= 120, f"{name}:{n} is {len(line)} characters wide"
    # PR #668's figures are cited only as unverified
    for m in re.finditer(r"PR #668[^\n]*", text):
        ahead = text[m.start():m.start() + 160]
        assert "unverified" in ahead or "review" in ahead.lower() or "JETSON_PLEM_REVIEW" in ahead, ahead
