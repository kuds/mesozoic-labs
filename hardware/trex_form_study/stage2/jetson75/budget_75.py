"""The 75 cm build's prices, by the study's cost model ``designlab/cost.py`` (``JETSON_PLEM_REVIEW.md`` section 5,
task 9).

Planning only.  Nothing here is built, quoted or measured, and nothing is adopted (decision-log P14, P21, P30).  It
is a price record.  By the owner's decision of 14 Nov 2026 (decision-log P31) prices are documented and, at this
point, are not weighed against P4's $10-20k, so nothing here sets a total against that figure.  It writes
``stage2/jetson75/budget_75.json``; ``python stage2/jetson75/budget_75.py md`` prints the tables that
``stage2/jetson75/BUDGET_75.md`` embeds.

Kinds of number (the study's labels):

* (c) a sourced fact: a catalog price, read from ``designlab/data`` at run time and named by its id.  The two set
  prices of the study's compute survey are read from its record too; the dev kit's holds the survey's own estimate
  for a drive;
* (d) an estimate: every lot, allowance, scaling and total of the cost model (``designlab/cost.py``, decision-log A8),
  and every line added here that the catalog does not hold (each names whose estimate it is and how it was derived);
* (c+d) catalog prices with an estimate on top, or with an assumed count or choice of part.

No dollar figure here is a simulation result or a hardware measurement.  The non-dollar inputs read from the records
(hip height, masses, cover volume) are static model results (a); the parts' ratings are catalog facts (c).

What it does, in order:

1. **Reproduces the round-two record first.**  ``cost.estimate`` is called with what it reads of a design (the
   selected parts, the hip height and the mass by category), taken from the committed record of ``75-M-H-10``
   (``round2/refine/feet_75/candidates/75-M-H-10.json``).  The result must equal the record's ``cost`` block: every
   line and the three totals exactly, and the drivers to 12 digits.  If it does not, the script stops and prints what
   differs.
2. **Prices the variants as differences from that baseline**, each with its own lines.  Every line that
   ``cost.estimate`` covers comes from a second call of it with the changed input (the compute part's id, the cover
   material and volume, or the expressive head's record); lines it does not cover are added from ``EXTRA`` and take the
   model's own contingency fractions (``cost.CONTINGENCY``).
3. **Gives each total two ways**: turnkey (the cost model's total, which holds the subcontract integration line) and
   self-assembled (the same without that line and without the contingency on it).
4. **Checks what the record already holds of P21's items**, so that nothing is counted twice (``checks``).
5. **Prices twice the lines a kept lot may already hold** (``conditional``: the 12 V regulator against the power lot,
   the supervisor against the kept control lot): each Jetson total is given with the line and without it.
6. **Ranks the lines by the gap between their low and high cases** (``least_certain``): where the prices are least
   certain, and what a quote for each line would have to cover.  No quote has been asked for.

Run from ``hardware/trex_form_study``::

    PYTHONPATH=. python stage2/jetson75/budget_75.py        # writes budget_75.json
    PYTHONPATH=. python stage2/jetson75/budget_75.py md     # prints the tables from the record on disk
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

STUDY = Path(__file__).resolve().parents[2]
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

from designlab import catalog, cost, refine  # noqa: E402
from designlab.build import COVER_MATERIALS, cover_material  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "budget_75.json"
BODY = "75-M-H-10"
RECORD = "round2/refine/feet_75/candidates/75-M-H-10.json"
HEAD_RECORD = "round2/refine/head_x_75/candidates/75-M-H-10.json"
LEVER_REFERENCE = "round2/refine/targeted_75/candidates/75-M-T-10.json"
LEVERS = "round2/refine/fixed_drives.json"
SURVEY = "designlab/data/research/compute_2026-11-07.json"  # the compute survey behind COMPUTE.md
#: Every file a number here depends on: the records, the cost model, the modules it reads (build.py: the covering
#: materials; refine.py: the PA11 lever; catalog.py: the loader) and the data.
INPUT_FILES = (RECORD, HEAD_RECORD, LEVER_REFERENCE, LEVERS, "designlab/cost.py", "designlab/build.py",
               "designlab/refine.py", "designlab/catalog.py", "designlab/data/parts.json",
               "designlab/data/actuators.json", SURVEY)

#: Decision-log P31 (the owner, 14 Nov 2026), as this record applies it.
P31 = dict(row="P31", date="14 Nov 2026",
           applied_as="prices are documented and, at this point, are not weighed against P4's $10-20k: this record "
                      "sets no total against that figure, and the Jetson hardware is priced for the record whatever "
                      "its extra cost")
PI5 = "rpi5_8gb_ai_hat_plus_13tops"  # decision-log P14 (adopted); the record's compute part
JETSON = "nvidia_orin_nano_super_devkit"  # studied as a refine case (P30) and explored whatever it adds (P31)
SURVEY_SETS = {JETSON: "A_orin_nano_super_devkit", PI5: "C_pi5_hailo8l_oak_depth"}  # the survey's ids for the two sets
TIERS = ("low", "nominal", "high")

KINDS = {
    "c": "a sourced fact: a catalog price in designlab/data, named by its id",
    "d": "an estimate: the cost model (designlab/cost.py, decision-log A8) or a line estimated here",
    "c+d": "catalog prices (c) with an estimate (d) on top or in them: the cost model's allowance, or an assumed "
           "count or choice of part",
}

#: What each of cost.py's 16 lines is made of.  ``a8`` marks the lots that are PR #662's 60 cm values multiplied by a
#: size driver (decision-log A8 names the frame, the covers, the integration and the contingency fractions; cost.py
#: also scales the harness and the foot sensing); ``carried`` marks the 60 cm lots used unchanged at 75 cm.
LINE_BASIS = {
    "leg_actuators": dict(kind="c+d", cls="catalog", basis="12 leg drives at catalog prices; the high case adds 5 %"),
    "expression_actuators": dict(kind="c+d", cls="catalog",
                                 basis="neck and tail servos at catalog prices; the high case adds 5 %"),
    "compute": dict(kind="c+d", cls="catalog",
                    basis="the computer's catalog price + $60 / $120 / $200 for storage, fan duct, mounts and leads"),
    "rgbd_camera": dict(kind="c+d", cls="catalog", basis="catalog price + $15 / $35 / $80 for the lead and mount"),
    "tof": dict(kind="c+d", cls="catalog", basis="two sensors at catalog price + $0 / $10 / $30"),
    "battery": dict(kind="c+d", cls="catalog", basis="one pack at catalog price + $30 / $60 / $100 for the cassette"),
    "control": dict(kind="d", cls="carried",
                    basis="PR #662's 60 cm lot, not scaled (Teensy 4.1, two CAN transceivers, IMU; the 60 cm "
                          "hardware basis also names a TTL servo-bus interface)"),
    "foot_sensing": dict(kind="d", cls="a8", basis="PR #662's 60 cm lot x (H / 0.39 m)^0.5"),
    "power": dict(kind="d", cls="carried", basis="PR #662's 60 cm lot for a 24 V bus, not scaled"),
    "harness": dict(kind="d", cls="a8", basis="PR #662's 60 cm lot x (H / 0.39 m)"),
    "frame": dict(kind="d", cls="a8", basis="PR #662's 60 cm lot x (structure mass / 1.79 kg)^0.6"),
    "covers": dict(kind="d", cls="a8", basis="PR #662's 60 cm lot x (cover volume / 480 cm3)^0.85"),
    "bench": dict(kind="d", cls="carried", basis="PR #662's 60 cm lot, not scaled (H is not above 0.6 m)"),
    "spares": dict(kind="c+d", cls="catalog",
                   basis="two of the dearest leg drive and one pack at catalog prices + $150 / $350 / $650"),
    "integration": dict(kind="d", cls="a8", basis="PR #662's 60 cm lot x (H / 0.39 m)^0.7"),
    "contingency": dict(kind="d", cls="a8", basis="5.3 / 7.2 / 8.4 % of the other lines"),
}
CLASS_LABEL = {
    "a8": "scaled from the 60 cm estimate",
    "carried": "60 cm lot, carried over unscaled",
    "catalog": "catalog price + allowance",
    "new": "new line, estimated here",
    "new_c": "new line, catalog price; the count or the part is assumed",
}

#: Lines the cost model does not hold.  ``parts``/``qty`` lines are catalog prices read at run time (kind c+d: the
#: price is the catalog's, the count or the choice of part per case is an assumption); ``usd`` lines are estimates
#: (kind d) with their owner and derivation.  A range from a study document takes its midpoint as the nominal value.
#: "A web search summary" is a search result's summary read on 14 Nov 2026; the pages were not opened (BUDGET_75.md
#: section 6 lists them).
EXTRA = {
    "regulator_12v": dict(
        label="6S-to-12 V regulator for the dev kit's carrier (decision-log E7), if it is bought separately",
        parts=("pololu_d36v50f12", "pololu_d42v110f12", "pololu_d42v110f12"), qty=(1, 1, 1), kind="c+d",
        basis="the prices are the catalog's (c); the choice of part per case is an assumption (d).  Low: the 4.5 A "
              "part; nominal and high: the 9 A part (no power mode is chosen, and the larger part covers both modes "
              "with room for USB loads).  Both accept the 6S bus; E7 rules the D36V50F12 out only on 12S",
        caution="probably already in the power lot, and not a cost of the Jetson alone (section 3.2).  The lot is the "
                "60 cm robot's, whose hardware basis lists a D36V50F12 for its computer "
                "(r0_posture_study_2026-11-07/study/hardware/component_basis.md); E7 calls that part the incumbent; "
                "the compute survey lists the dev kit's 12 V converter as existing and not counted, and the Pi 5 "
                "set's 5 V / 5 A regulator as not priced.  The lot's own breakdown is not in this repository"),
    "usb_can": dict(
        label="USB-CAN FD adapter(s), PCAN-USB FD (one CAN channel each)",
        parts=("pcan_usb_fd",) * 3, qty=(1, 2, 2), kind="c+d",
        basis="the price is the catalog's (c); the count is an assumption (d): one adapter in the low case (PR "
              "#668's layout, unverified); two in the nominal and high cases, to keep the two CAN buses of the "
              "study's build (designlab/build.py, control stack: 'CAN x2'; JETSON_PLEM_REVIEW.md section 3).  The "
              "bus plan (review task 7) decides"),
    "supervisor": dict(
        label="Supervisor, if it is a separate board: a small microcontroller board with a CAN transceiver and a "
              "stop output",
        usd=(40.0, 70.0, 120.0), kind="d",
        basis="no catalog entry; this document's estimate from the price class of a Teensy 4.1 (about $34-36 at its "
              "maker's store, a web search summary) plus a transceiver, a relay driver and wiring (a guess, no "
              "source)",
        caution="overlaps the kept control lot, whose Teensy 4.1 and CAN transceivers are such a board: if that stack "
                "serves as the supervisor, this line is not needed"),
    "servo_bus_adapter": dict(
        label="USB adapter(s) for the neck and tail servos' TTL buses",
        usd=(45.0, 60.0, 90.0), kind="d",
        basis="no catalog entry; this document's estimate: a ROBOTIS U2D2 (about $44, a web search summary) plus a "
              "Feetech bus adapter (about $15, a guess, no source)",
        caution="the kept control lot's 60 cm basis also names a TTL servo-bus interface; if the kept stack keeps "
                "driving the servo buses, this line is not needed"),
    "pad_treads": dict(
        label="Cast polyurethane pad treads: casting rubber, printed moulds and the grip test on the owner's floors",
        usd=(100.0, 200.0, 400.0), kind="d",
        basis="low: MATERIALS.md section 5's estimate for the floor test alone (about one day and $100), with a "
              "printed TPU trial tread.  Nominal adds $100 and high adds $300 for casting rubber, release agent and "
              "printed moulds: this document's guess, with no source beyond the price class of a trial unit of 60 "
              "Shore A urethane rubber (about $25-45, a web search summary); the high case allows for several "
              "compounds and remakes.  The soles themselves are in the frame lot"),
    "dc_input": dict(
        label="External DC input: the input, the 24 V priority power path, a regeneration clamp and the dummy puck",
        usd=(200.0, 300.0, 400.0), kind="d",
        basis="MATERIALS.md section 6's estimate, $0.2-0.4k 'for the input, power path, clamp and puck'; the "
              "nominal value is its midpoint.  That estimate does not name the 5 m cable or the stop tie-in; this "
              "document assumes they fit inside it",
        caution="the clamp may already be in the power lot: the 60 cm robot's hardware basis names a regenerative "
                "clamp with its regulators (component_basis.md), and cost.py gives that lot no scope, so its "
                "contents could not be checked"),
    "cable_boom": dict(
        label="Overhead boom for the power cable, with a balancer under 0.5 kg",
        usd=(300.0, 900.0, 1500.0), kind="d",
        basis="MATERIALS.md section 6's estimate, $0.3-1.5k 'if the bench lot's catch frame cannot carry it'; the "
              "nominal value is its midpoint; $0 if the catch frame can"),
    "yaw_bearing": dict(
        label="Optional, unmodelled: the neck-yaw stage's bearing (expressive head)",
        usd=(20.0, 165.0, 300.0), kind="d",
        basis="no catalog entry and not in any record (stage2/x/X_COSTS.md section 5).  This document's estimate; "
              "low: two thin-section ball bearings in a machined seat (a guess, no source); nominal: a 20 mm-bore "
              "polymer slewing ring at the cheapest listing found (about $164); high: the same ring at the dearest "
              "listing found (about $300); the listings are from a web search summary"),
    "audio": dict(
        label="Optional, unmodelled: audio (amplifier, speaker, microphones)",
        usd=(25.0, 70.0, 150.0), kind="d",
        basis="no catalog entry and not in any record.  This document's estimate; low: a small amplifier board "
              "(about $6, a web search summary) with a small speaker and wiring (a guess, no source); high: with a "
              "far-field USB microphone array (about $60-78 in US listings, a web search summary) and a larger "
              "speaker (a guess); nominal: a guess between the two"),
}

#: Lines a kept lot of the record may already hold.  Each is priced in its block, and every total that holds one is
#: also given without it ("lower").
CONDITIONS = {
    "regulator_12v": "the power lot already holds the 12 V regulator",
    "supervisor": "the kept Teensy stack is the supervisor",
}

#: The variants, each a set of changes to the baseline.  ``compute`` swaps the selected computer's catalog id;
#: ``cover`` re-prices the coverings through cost.py's material path; ``record`` prices another committed record.
BLOCKS = {
    "jetson": dict(
        group="compute", optional=False,
        label="Jetson Orin Nano Super dev kit in place of the Raspberry Pi 5 + AI HAT+, with the walking loop on "
              "the microcontroller (the baseline of the owner's 14 Nov decision, P30)",
        compute=JETSON, extras=("regulator_12v",), conditional=("regulator_12v",),
        keeps=(("control", "the microcontroller stack stays: it is the record's control lot (Teensy 4.1, two CAN "
                           "transceivers, IMU).  The catalog has no Teensy entry"),)),
    "jetson_direct_parts": dict(
        group="compute", optional=False,
        label="Added only if the Jetson drives the leg drives directly over USB-CAN (the alternative the bench test "
              "measures), on top of `jetson`",
        extras=("usb_can", "supervisor", "servo_bus_adapter"), conditional=("supervisor",),
        keeps=(("control", "kept in full as a conservative choice, because the lot has no breakdown here.  In the "
                           "direct option its Teensy and two CAN transceivers overlap the supervisor line, and its "
                           "servo-bus interface overlaps the USB servo-bus adapters; only the IMU is certainly "
                           "still needed"),)),
    "p21": dict(
        group="p21", optional=False,
        label="P21's adopted materials and power items that the record does not hold",
        cover="PA11_MJF", extras=("pad_treads", "dc_input", "cable_boom"),
        keeps=(("frame", "the 6061-T6 frame is already the record's frame line, and the 2.54 mm plates cost the "
                         "same in the model (checks, `frame_6061`)"),
               ("battery", "the removable pack stays (P21: the battery plus an external DC input)"),
               ("bench", "the bench supply for the tether is already in the bench lot"))),
    "head_x": dict(
        group="optional", optional=True,
        label="Optional: the expressive head's two extra servos (one for the neck yaw, one for the jaw), their "
              "brackets and the yaw-stage bearing.  The servos and brackets are modelled (the head's committed "
              "record); the bearing is not.  P2's deferral of a jaw for the build stands (P29), so the build's head "
              "as decided today has the neck-yaw servo only",
        record=HEAD_RECORD, extras=("yaw_bearing",),
        line_notes=dict(
            expression_actuators="two more STS3215 servos (`feetech_sts3215_c018`), one for the neck yaw and one "
                                 "for the jaw, at the catalog price, as the head's committed record holds them "
                                 "(round2/refine/head_x_75); the high case adds 5 %",
            frame="the two servos' brackets: the head's record carries 0.040 kg more structure (a), which the frame "
                  "lot's mass scaling prices")),
    "audio": dict(group="optional", optional=True, label="Optional: audio", extras=("audio",)),
}
SCENARIOS = {
    "record": dict(blocks=(), label="The round-two record: Raspberry Pi 5 + AI HAT+ (P14), P21's items not priced"),
    "pi_p21": dict(blocks=("p21",), label="Raspberry Pi 5 + AI HAT+ (P14, adopted) with P21's items"),
    "jetson_mcu_p21": dict(blocks=("jetson", "p21"),
                           label="Jetson dev kit, walking loop on the microcontroller, with P21's items"),
    "jetson_direct_p21": dict(blocks=("jetson", "jetson_direct_parts", "p21"),
                              label="Jetson dev kit driving the legs directly, with P21's items"),
    "pi_p21_optional": dict(blocks=("p21", "head_x", "audio"),
                            label="Raspberry Pi 5 + AI HAT+ with P21's items and every optional line"),
    "jetson_mcu_p21_optional": dict(blocks=("jetson", "p21", "head_x", "audio"),
                                    label="Jetson dev kit, microcontroller loop, P21's items and every optional line"),
}
#: The three builds the document leads with; the optional lines stay outside their totals.
MAIN = ("pi_p21", "jetson_mcu_p21", "jetson_direct_p21")

#: What a quote for each line would have to cover (the lines are ranked by their high-minus-low gap: where the prices
#: are least certain).  A description of the line's scope, not a step to take: no quote has been asked for.
QUOTE_SCOPE = {
    "frame": "Laser-cut and machined 6061-T6 plates and brackets, bearings, hubs, soles and fasteners, from "
             "drawings, at three shops; the plates at 2.54 mm and at 3 mm",
    "covers": "MJF PA11 shells from the exported cover shapes with graded 1.4 / 1.2 / 1.0 mm walls, inserts and "
              "finish, at three print services",
    "integration": "Subcontract fit-up of one prototype: hours and rate for mechanical assembly, harness routing "
                   "and bench checkout",
    "cable_boom": "Whether the bench frame can carry the 5 m cable (the line is then $0); if not, a jib or gantry "
                  "boom with a balancer rated under 0.5 kg",
    "bench": "24 V bench supply (about 31 A, RSP-750-24 class), a 6S charger, a cradle or catch frame and "
             "instrumentation",
    "spares": "Two spare hip-pitch drives and one spare pack, at the prices that quotes for the drives and the "
              "pack would give",
    "power": "The power bay as one assembled module: distribution board, fuses, the stop circuit, the regeneration "
             "clamp and its regulators, each regulator named (the computer's, the logic rail's, the servos')",
    "harness": "The leg and trunk harness with locking connectors and the neck service loop, from a wiring list",
    "pad_treads": "A urethane casting kit (50-60 Shore A polyether), printed moulds and the grip test on the "
                  "owner's floors",
    "foot_sensing": "Foot contact or load sensors for two feet with their conditioning boards",
    "dc_input": "The input connector, the priority power-path board (LTC4417 class), the dummy puck and the cable; "
                "the clamp and the stop tie-in belong to the power bay's price",
    "leg_actuators": "Twelve drives landed: unit price, shipping, duty and lead time (10 x RMD-X4-P36-36-E-L, 2 x "
                     "RMD-X6-P20-60-E), and which X6 variant ships at the listed price",
    "control": "Teensy 4.1, two CAN transceivers, an IMU and the servo-bus interface as a board stack",
    "compute": "The computer at today's price, its storage and its mounting tray",
    "rgbd_camera": "The camera and a locking lead, and the remaining stock of the discontinued camera chip",
    "battery": "One pack and the tail-base cassette",
    "expression_actuators": "The neck and tail servos at today's prices",
    "regulator_12v": "A stock part at its list price; an extra only if the power bay holds no 12 V regulator for "
                     "the computer",
    "tof": "Stock parts at their list prices",
    "usb_can": "The adapter at today's price, and the two-channel version's price",
    "supervisor": "No firm price until the supervisor's design exists; not needed if the kept Teensy stack is the "
                  "supervisor",
    "servo_bus_adapter": "Stock parts at their list prices",
}

#: PR #668's price slips (JETSON_PLEM_REVIEW.md, caveat 5), each checked against the catalog at run time.
PR668_SLIPS = (
    dict(what="The Jetson dev kit's price", pr668="$249 in its low and nominal cases",
         catalog_id=JETSON, wrong_price=249),
    dict(what="The tail servo's catalog id", pr668="`feetech_sts3215_12v`", wrong_id="feetech_sts3215_12v",
         right_id_from=("selection", "tail")),
    dict(what="The neck servo's catalog id", pr668="`robotis_xm540_w270`", wrong_id="robotis_xm540_w270",
         right_id_from=("selection", "neck")),
)


def _sha(rel: str) -> str:
    return hashlib.sha256((STUDY / rel).read_bytes()).hexdigest()


def _load(rel: str) -> dict:
    return json.loads((STUDY / rel).read_text())


def _r2(values) -> list:
    return [round(float(v), 2) for v in values]


class RecordDesign:
    """What ``cost.estimate`` reads of a design, taken from a committed candidate record: the selected actuators, the
    hip height and, for the expressive head, its two drive groups.  Round-two records carry no material or power
    override unless ``cover`` is given here, so the model's defaults apply."""

    def __init__(self, record: dict, cover: str | None = None):
        self.sel = record["selection"]
        self.sk = SimpleNamespace(H=record["H"])
        self.head_x_groups = tuple(g for g in ("neck_yaw", "jaw") if g in self.sel)
        if cover is not None:
            self.cover_mat_override = cover

    def act(self, role: str):
        joints = self.sel["joints"]
        return catalog.actuator(joints[role] if role in joints else self.sel[role])


def _rows(record: dict, cover_mass_factor: float = 1.0) -> list[dict]:
    """The record's mass by category, as the rows ``cost.estimate`` sums (it reads only category and mass)."""
    return [dict(category=k, mass_kg=v * (cover_mass_factor if k == "cover" else 1.0))
            for k, v in record["mass_by_category"].items()]


def _pa11_lever() -> dict:
    """The study's own PA11 lever (``refine.MAT_LEVERS``: the material and the wall scale), and the factor it puts
    on the cover mass when the walls thin uniformly: wall scale x PA11 density / the body's cover density."""
    over = dict(next(lv[1] for lv in refine.MAT_LEVERS if lv[0] == "pa11"))
    mat, scale = over["cover_material"], over["cover_wall_scale"]
    base = cover_material(int(BODY.split("-")[0]))
    return dict(material=mat, base_material=base, wall_scale=scale,
                mass_factor=scale * COVER_MATERIALS[mat]["density"] / COVER_MATERIALS[base]["density"])


def _estimate(record: dict, compute: str | None = None, cover: str | None = None) -> dict:
    sel = dict(record["selection"])
    if compute is not None:
        sel["compute"] = compute
    if cover is None:
        return cost.estimate(RecordDesign(dict(record, selection=sel)), _rows(record), sel)
    lever = _pa11_lever()
    assert cover == lever["material"]
    return cost.estimate(RecordDesign(dict(record, selection=sel), cover=cover), _rows(record, lever["mass_factor"]),
                         sel)


def reproduce(record: dict) -> dict:
    """``cost.estimate`` on the record's own inputs against the record's ``cost`` block; every difference listed.
    The lines (rounded to the cent) and the totals (to the dollar) must be equal.  The drivers are unrounded floats, so
    they are compared to 12 digits: three of them come from a power function, whose last bit may differ between
    machines."""
    est, rec, diffs = _estimate(record), record["cost"], []
    for tier in TIERS:
        if est[tier] != rec[tier]:
            diffs.append(dict(where=f"total {tier}", recorded=rec[tier], recomputed=est[tier]))
    for name in sorted(set(est["lines"]) | set(rec["lines"])):
        if est["lines"].get(name) != rec["lines"].get(name):
            diffs.append(dict(where=f"line {name}", recorded=rec["lines"].get(name),
                              recomputed=est["lines"].get(name)))
    for name in sorted(set(est["drivers"]) | set(rec["drivers"])):
        a, b = est["drivers"].get(name), rec["drivers"].get(name)
        same = a == b
        if not same and all(isinstance(v, float) for v in (a, b)):
            same = math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
        if not same:
            diffs.append(dict(where=f"driver {name}", recorded=rec["drivers"].get(name),
                              recomputed=est["drivers"].get(name)))
    return dict(recomputed=est, exact=not diffs, differences=diffs)


def extra_line(key: str) -> dict:
    spec = EXTRA[key]
    out = dict(key=key, label=spec["label"], kind=spec["kind"], basis=spec["basis"])
    if "parts" in spec:
        parts = catalog.parts()
        out["catalog"] = [dict(id=pid, qty=q, price_usd=parts[pid].price) for pid, q in zip(spec["parts"], spec["qty"])]
        out["usd"] = _r2(q * parts[pid].price for pid, q in zip(spec["parts"], spec["qty"]))
    else:
        out["usd"] = _r2(spec["usd"])
    if "caution" in spec:
        out["caution"] = spec["caution"]
    return out


def totals(lines: dict) -> dict:
    """Totals of a set of lines (without a contingency line) the way cost.py totals them: the contingency fractions
    on the sum of the other lines.  Turnkey holds every line; self-assembled leaves out the subcontract integration
    line and the contingency on it."""
    out = {}
    for name, skip in (("turnkey", ()), ("self_assembled", ("integration",))):
        sub = [sum(v[i] for k, v in lines.items() if k not in skip) for i in range(3)]
        cont = [sub[i] * cost.CONTINGENCY[i] for i in range(3)]
        out[name] = dict(subtotal=_r2(sub), contingency=_r2(cont), total=[round(sub[i] + cont[i], 0) for i in range(3)])
    return out


def price(record: dict, head_record: dict, blocks: tuple, drop: tuple = ()) -> dict:
    """One scenario: the baseline with the named blocks applied, lines and totals."""
    specs = [BLOCKS[b] for b in blocks]
    rec = head_record if any("record" in s for s in specs) else record
    compute = next((s["compute"] for s in specs if "compute" in s), None)
    cover = next((s["cover"] for s in specs if "cover" in s), None)
    est = _estimate(rec, compute=compute, cover=cover)
    lines = {k: list(v) for k, v in est["lines"].items() if k != "contingency"}
    extras = []
    for s in specs:
        for key in s.get("extras", ()):
            if key not in drop and key not in lines:
                e = extra_line(key)
                lines[key] = e["usd"]
                extras.append(key)
    return dict(blocks=list(blocks), compute=compute or rec["selection"]["compute"], cover=cover or "as recorded",
                lines={k: _r2(v) for k, v in lines.items()}, extra_lines=extras, **totals(lines))


def _delta(a: dict, b: dict) -> dict:
    """b minus a: the lines that differ (changed or added), and the two totals' differences."""
    lines = {}
    for name in list(b["lines"]):
        base = a["lines"].get(name, [0.0, 0.0, 0.0])
        d = _r2(b["lines"][name][i] - base[i] for i in range(3))
        if any(d) or name not in a["lines"]:
            lines[name] = dict(usd=d, new=name not in a["lines"])
    return dict(lines=lines,
                turnkey=[b["turnkey"]["total"][i] - a["turnkey"]["total"][i] for i in range(3)],
                self_assembled=[b["self_assembled"]["total"][i] - a["self_assembled"]["total"][i] for i in range(3)],
                subtotal=_r2(b["turnkey"]["subtotal"][i] - a["turnkey"]["subtotal"][i] for i in range(3)))


def _ref_comments() -> dict:
    """The scope comment cost.py gives each reference lot (read from its source, so the scopes quoted here are the
    model's own words)."""
    out = {}
    for line in Path(cost.__file__).read_text().splitlines():
        m = re.match(r"\s{4}(\w+)=\([^)]*\),?\s*#\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def checks(record: dict, head_record: dict, base: dict) -> dict:
    """What the record already holds of P21's items, and the checks on this script's own rules."""
    over = dict(map(tuple, record["overrides"]))
    drv = record["cost"]["drivers"]
    scopes = _ref_comments()
    lever = _pa11_lever()
    levers = {r["case"]: r for r in _load(LEVERS)}
    # 6061-T6 frame: the record's frame line is the reference lot times the mass factor alone
    frame_plain = _r2(v * drv["frame_factor"] for v in cost.REF["frame"])
    thinal = levers["mat75_thinal"]
    # coverings: priced as the body's default material at full wall thickness
    base_mat = COVER_MATERIALS[lever["base_material"]]
    cover_cm3 = record["mass_by_category"]["cover"] / (base_mat["density"] * 1e-6)
    # the PA11 rule, applied to the lever's own reference body and set against the lever's recorded cost
    ref = _load(LEVER_REFERENCE)
    rule = _estimate(ref, cover=lever["material"])
    recd = levers["mat75_pa11"]["cost"]
    pa11 = price(record, head_record, ("p21",), drop=("pad_treads", "dc_input", "cable_boom"))
    # additivity: every block priced alone and summed against the scenario priced in one call
    worst = 0.0
    for name, sc in SCENARIOS.items():
        whole = price(record, head_record, sc["blocks"])
        parts_sum = [base["turnkey"]["subtotal"][i] + sum(
            _delta(base, price(record, head_record, (b,)))["subtotal"][i] for b in sc["blocks"]) for i in range(3)]
        worst = max(worst, max(abs(whole["turnkey"]["subtotal"][i] - parts_sum[i]) for i in range(3)))
    head = reproduce(head_record)
    return dict(
        frame_6061=dict(
            in_the_record=("structure_material" not in over and record["cost"]["lines"]["frame"] == frame_plain),
            how="the record has no structure_material override, and its frame line equals cost.REF['frame'] x its "
                "frame_factor with no material factor; the committed 2.54 mm plate lever on 75-M-T-10 "
                "(fixed_drives.json, mat75_thinal) costs exactly what its reference does",
            record_overrides=record["overrides"], frame_line=record["cost"]["lines"]["frame"],
            ref_lot_times_factor=frame_plain,
            thin_plate_lever_cost_equals_reference=all(
                thinal["cost"][t] == thinal["reference_cost"][t] for t in TIERS),
            lot_scope=scopes.get("frame")),
        covers_pa11=dict(
            in_the_record=("cover_material" in over or "cover_wall_scale" in over),
            how="the record has no cover_material or cover_wall_scale override, and its cover volume driver equals "
                "its cover mass over the default material's density, so its covers line is PA12 at full walls",
            default_material=lever["base_material"], cover_mass_kg=record["mass_by_category"]["cover"],
            cover_cm3_from_mass=cover_cm3, cover_cm3_driver=drv["cover_cm3"],
            driver_matches=abs(cover_cm3 - drv["cover_cm3"]) < 1e-9, lever=lever,
            covers_line_recorded=record["cost"]["lines"]["covers"], covers_line_pa11=pa11["lines"]["covers"],
            rule_on_the_recorded_lever=dict(
                what="the same rule on 75-M-T-10 (targeted_75), against the lever's recorded cost (mat75_pa11)",
                covers_line_rule=rule["lines"]["covers"], covers_line_recorded=recd["lines"]["covers"],
                max_abs_diff_usd=round(max(abs(a - b) for a, b in zip(rule["lines"]["covers"],
                                                                      recd["lines"]["covers"])), 2),
                totals_rule=[rule[t] for t in TIERS], totals_recorded=[recd[t] for t in TIERS],
                cover_cm3_rule=rule["drivers"]["cover_cm3"], cover_cm3_recorded=recd["drivers"]["cover_cm3"])),
        pad_treads=dict(
            in_the_record="soles only",
            how="the frame lot of cost.py names soles in its scope, and the record's foot mass (with P20's bigger "
                "feet: toe_scale 1.15, pad_scale 1.25) is inside the structure mass that scales that lot; no line "
                "names a tread compound or a floor test",
            lot_scope=scopes.get("frame"), foot_mass_kg=record["mass_by_category"]["foot"],
            structure_kg_driver=drv["structure_kg"],
            structure_kg_from_mass=sum(record["mass_by_category"][k] for k in ("bearings", "structure", "linkage",
                                                                                "foot"))),
        dc_input_and_boom=dict(
            in_the_record=False,
            how="the record has no power_mode override and its battery line is the pack plus the cassette allowance; "
                "the bench lot's scope names the tether supply and the catch frame, so the bench supply is not added "
                "again; no line names an onboard DC input, a power path or a boom",
            not_checked="cost.py gives the power lot no scope, so its contents could not be checked: the 60 cm "
                        "robot's hardware basis names a regenerative clamp with its regulators, and the DC input "
                        "line prices a clamp too",
            battery_line=record["cost"]["lines"]["battery"], lot_scope=scopes.get("bench"),
            power_lot_scope=scopes.get("power24")),
        compute_allowance=dict(
            how="cost.py adds $60 / $120 / $200 to the computer's price; build.py's compute tray names what it "
                "stands for (NVMe storage, fan duct, mounts, leads), so no separate storage line is added for the "
                "Jetson.  The Pi 5 set's catalog price already holds a USB SSD, so the record's compute line counts "
                "storage twice and the Jetson's once (compute_difference)",
            allowance=[60, 120, 200]),
        head_record=dict(
            what="cost.estimate on the expressive head's record against that record's cost block",
            record=HEAD_RECORD, exact=head["exact"], differences=head["differences"],
            totals=[head_record["cost"][t] for t in TIERS],
            base_record_sha256_in_it=head_record["head_x"]["base_record_sha256"]),
        totals_from_rounded_lines=dict(
            what="the totals rebuilt from the record's rounded lines against the record's totals",
            rebuilt=base["turnkey"]["total"], recorded=[record["cost"][t] for t in TIERS],
            equal=base["turnkey"]["total"] == [record["cost"][t] for t in TIERS]),
        additivity=dict(
            what="each scenario priced in one call against the baseline plus its blocks priced alone",
            max_abs_diff_usd=round(worst, 2)),
    )


def catalog_entries(record: dict, head_record: dict) -> dict:
    """Every catalog id this record prices, with its price (kind c) and the catalog's own basis and sources."""
    acts, parts, out = catalog.actuators(), catalog.parts(), {}
    sel = dict(head_record["selection"])
    ids = [sel[k] for k in ("leg", "neck", "tail", "camera", "tof", "compute", "battery", "neck_yaw", "jaw")]
    ids += list(sel["joints"].values()) + [JETSON]
    ids += [pid for spec in EXTRA.values() for pid in spec.get("parts", ())]
    for i in dict.fromkeys(ids):
        item = acts.get(i) or parts[i]
        # an actuator's basis is a dict keyed by quantity ("price"); a part's basis is one string for the whole entry
        price_basis = (item.basis.get("price") or item.basis.get("price_usd")) if isinstance(item.basis, dict) \
            else item.basis
        out[i] = dict(table="actuators.json" if i in acts else "parts.json", model=f"{item.vendor} {item.model}",
                      price_usd=item.price, basis=price_basis,
                      price_sources=[s for s in item.sources if "price" in s.lower() or "USD" in s or "$" in s])
    return out


def record_ids(record: dict) -> list:
    """Every catalog id the round-two record names (its selection)."""
    sel = record["selection"]
    ids = [v for k, v in sel.items() if isinstance(v, str)] + list(sel["joints"].values())
    return list(dict.fromkeys(ids))


def spreads(scn: dict) -> list:
    """The lines of one scenario ranked by their high-minus-low spread: where its prices are least certain."""
    rows = []
    for name, usd in scn["lines"].items():
        b = LINE_BASIS.get(name)
        if b is None:
            e = EXTRA[name]
            b = dict(kind=e["kind"], cls="new_c" if "parts" in e else "new", basis=e["basis"])
        rows.append(dict(line=name, usd=usd, spread=round(usd[2] - usd[0], 2), kind=b["kind"], cls=b["cls"],
                         basis=b["basis"], quote_scope=QUOTE_SCOPE.get(name, "")))
    rows.sort(key=lambda r: (-r["spread"], r["line"]))
    return rows


def compute_difference(blocks: dict) -> dict:
    """The Jetson-against-Pi difference read four ways, so that it is stated as a range.  The cost model's own
    difference is the two catalog prices' ($109).  The compute survey behind COMPUTE.md prices the two sets like for
    like, each with its storage: its dev-kit set is the kit plus an NVMe SSD, and its Pi 5 set, which is the catalog
    entry, already holds a USB SSD.  The catalog's note also gives a street price for the Pi 5 set."""
    parts = catalog.parts()
    arch = {a["id"]: a["price_usd"] for a in _load(SURVEY)["synthesis"]["architectures"]}
    kit, pi = parts[JETSON].price, parts[PI5].price
    set_j, set_p = arch[SURVEY_SETS[JETSON]], arch[SURVEY_SETS[PI5]]
    street = float(re.search(r"~\$(\d+) at observed street prices", parts[PI5].notes).group(1))
    reg = blocks["jetson"]["lines"]["regulator_12v"]["usd"]
    readings = dict(
        kit_alone=[kit - pi] * 3,
        with_regulator=[kit - pi + r for r in reg],
        storage_once_each=[set_j - set_p] * 3,
        storage_once_each_with_regulator=[set_j - set_p + r for r in reg],
        pi5_at_street_price=[kit - street] * 3)
    turnkey = {k: _with_contingency(v) for k, v in readings.items()}
    return dict(
        catalog_usd={JETSON: kit, PI5: pi, "difference": kit - pi},
        survey_set_usd={JETSON: set_j, PI5: set_p, "difference": set_j - set_p, "source": SURVEY,
                        "jetson_set_minus_kit": set_j - kit, "pi5_set_equals_catalog": set_p == pi},
        pi5_street_usd=street, pi5_street_source=f"designlab/data/parts.json, the notes of `{PI5}`",
        before_contingency={k: _r2(v) for k, v in readings.items()},
        turnkey=turnkey,
        nominal_as_priced=[turnkey["kit_alone"][1], turnkey["with_regulator"][1]],
        nominal_all_readings=[min(v[1] for v in turnkey.values()), max(v[1] for v in turnkey.values())])


def head_yaw_only(record: dict, head_record: dict) -> dict:
    """The expressive head as the build would have it with the jaw deferred (P2, P29): the head's record less the
    jaw servo and its bracket, priced by the cost model.  An estimate: no record of that body exists."""
    jaw = [p for p in head_record["head_x"]["parts"] if p["group"] == "jaw"]
    mass = dict(head_record["mass_by_category"])
    for p in jaw:
        mass[p["category"]] -= p["mass_kg"]
    sel = {k: v for k, v in head_record["selection"].items() if k != "jaw"}
    est = _estimate(dict(head_record, selection=sel, mass_by_category=mass))
    base = record["cost"]["lines"]
    lines = {k: _r2(est["lines"][k][i] - base[k][i] for i in range(3)) for k in ("expression_actuators", "frame")}
    both = [sum(v[i] for v in lines.values()) for i in range(3)]
    bearing = EXTRA["yaw_bearing"]["usd"]
    return dict(removed=[dict(name=p["name"], category=p["category"], mass_kg=p["mass_kg"]) for p in jaw],
                servo=head_record["selection"]["neck_yaw"],
                servo_price_usd=catalog.actuator(head_record["selection"]["neck_yaw"]).price, lines=lines,
                turnkey_servo_and_bracket=_with_contingency(both),
                turnkey_with_bearing=_with_contingency([both[i] + bearing[i] for i in range(3)]))


def slips(record: dict) -> list:
    acts, parts, out = catalog.actuators(), catalog.parts(), []
    for s in PR668_SLIPS:
        row = dict(what=s["what"], pr668_unverified=s["pr668"])
        if "catalog_id" in s:
            row.update(catalog_id=s["catalog_id"], catalog_price_usd=parts[s["catalog_id"]].price,
                       pr668_price_usd=s["wrong_price"],
                       used_here=f"${parts[s['catalog_id']].price:,.0f}, the price of `{s['catalog_id']}`")
        else:
            right = record[s["right_id_from"][0]][s["right_id_from"][1]]
            row.update(pr668_id_exists=s["wrong_id"] in acts or s["wrong_id"] in parts, right_id=right,
                       right_id_exists=right in acts, right_id_price_usd=acts[right].price,
                       used_here=f"`{right}` (${acts[right].price:,.2f}), the record's own id")
        out.append(row)
    return out


def _with_contingency(usd) -> list:
    return [round(usd[i] * (1 + cost.CONTINGENCY[i]), 0) for i in range(3)]


def derived(record: dict, head_record: dict, blocks: dict, scenarios: dict, spread_rows: list) -> dict:
    """The figures the document's prose quotes that are sums, differences or shares of the lines above (so the prose
    is checked against the record, not typed by hand).  ``steps`` are differences of two printed totals, so the prose
    adds up to the dollar; a block's own figure can be $1 off a step, because totals are rounded after summing."""
    lines = record["cost"]["lines"]
    sub = sum(v[1] for k, v in lines.items() if k != "contingency")
    split = {c: round(sum(v[1] for k, v in lines.items() if LINE_BASIS[k]["cls"] == c and k != "contingency"), 2)
             for c in ("a8", "carried", "catalog")}
    s_split = {c: round(sum(r["usd"][1] for r in spread_rows if r["cls"] == c), 2)
               for c in ("a8", "carried", "catalog", "new", "new_c")}

    def upper(s):
        return scenarios[s]["turnkey"]["total"]

    def lower(s):
        return scenarios[s]["lower"]["turnkey"]["total"]

    def minus(a, b):
        return [a[i] - b[i] for i in range(3)]

    steps = dict(
        p21_on_the_record=minus(upper("pi_p21"), upper("record")),
        jetson_kit_alone_on_pi=minus(lower("jetson_mcu_p21"), upper("pi_p21")),
        jetson_with_regulator_on_pi=minus(upper("jetson_mcu_p21"), upper("pi_p21")),
        direct_on_jetson_lower=minus(lower("jetson_direct_p21"), lower("jetson_mcu_p21")),
        direct_on_jetson=minus(upper("jetson_direct_p21"), upper("jetson_mcu_p21")),
        optional_on_pi=minus(upper("pi_p21_optional"), upper("pi_p21")),
        optional_on_jetson=minus(upper("jetson_mcu_p21_optional"), upper("jetson_mcu_p21")))
    by_line = {r["line"]: r for r in spread_rows}
    return dict(
        steps=steps,
        p21_lines_with_contingency={k: _with_contingency(v["usd"]) for k, v in blocks["p21"]["lines"].items()},
        regulator_with_contingency=_with_contingency(blocks["jetson"]["lines"]["regulator_12v"]["usd"]),
        regulator_upgrade_usd=round(blocks["jetson"]["lines"]["regulator_12v"]["usd"][1]
                                    - blocks["jetson"]["lines"]["regulator_12v"]["usd"][0], 2),
        supervisor_with_contingency=_with_contingency(blocks["jetson_direct_parts"]["lines"]["supervisor"]["usd"]),
        integration_line=lines["integration"],
        baseline_subtotal_nominal=round(sub, 2),
        baseline_split_nominal=split,
        baseline_split_share={c: round(v / sub, 4) for c, v in split.items()},
        adopted_subtotal_nominal=round(sum(r["usd"][1] for r in spread_rows), 2),
        adopted_split_nominal=s_split,
        spread_top3=round(sum(r["spread"] for r in spread_rows[:3]), 2),
        spread_all=round(sum(r["spread"] for r in spread_rows), 2),
        spread_compute=by_line["compute"]["spread"],
        largest_lines={t: max(((k, v[i]) for k, v in scenarios["pi_p21"]["lines"].items()), key=lambda kv: kv[1])[0]
                       for i, t in enumerate(TIERS)},
        optional_together_turnkey=steps["optional_on_pi"],
        head_record_minus_baseline=[head_record["cost"][t] - record["cost"][t] for t in TIERS],
        head_yaw_only=head_yaw_only(record, head_record),
        self_assembled_gap_record=[scenarios["record"]["turnkey"]["total"][i]
                                   - scenarios["record"]["self_assembled"]["total"][i] for i in range(3)],
        compute_difference=compute_difference(blocks),
    )


def build() -> dict:
    record, head_record = _load(RECORD), _load(HEAD_RECORD)
    rep = reproduce(record)
    if not rep["exact"]:
        raise SystemExit("STOP: cost.estimate does not reproduce the record's cost block for " + BODY + ":\n"
                         + json.dumps(rep["differences"], indent=1))
    base = price(record, head_record, ())

    def conditional(names: tuple) -> tuple:
        return tuple(k for b in names for k in BLOCKS[b].get("conditional", ()))

    def condition(keys: tuple) -> str:
        return " and ".join(CONDITIONS[k] for k in keys)

    scenarios = {}
    for name, sc in SCENARIOS.items():
        p = price(record, head_record, sc["blocks"])
        p.update(label=sc["label"], vs_record=_delta(base, p))
        cond = conditional(sc["blocks"])
        if cond:  # the same build if the kept lots already hold these lines
            lo = price(record, head_record, sc["blocks"], drop=cond)
            p["lower"] = dict(without=list(cond), condition=condition(cond), turnkey=lo["turnkey"],
                              self_assembled=lo["self_assembled"])
        scenarios[name] = p
    blocks = {}
    for name, spec in BLOCKS.items():
        p = price(record, head_record, (name,))
        d = _delta(base, p)
        for key, line in d["lines"].items():
            if key in EXTRA:
                line.update({k: v for k, v in extra_line(key).items() if k != "usd"})
            else:
                line.update(label=key, kind=LINE_BASIS[key]["kind"],
                            basis=spec.get("line_notes", {}).get(key, LINE_BASIS[key]["basis"]))
        kept = [dict(line=key, usd=base["lines"][key], kind=LINE_BASIS[key]["kind"], note=note)
                for key, note in spec.get("keeps", ())]
        blocks[name] = dict(label=spec["label"], group=spec["group"], optional=spec["optional"], unchanged=kept, **d)
        cond = conditional((name,))
        if cond:
            lo = _delta(base, price(record, head_record, (name,), drop=cond))
            blocks[name]["lower"] = dict(without=list(cond), condition=condition(cond), turnkey=lo["turnkey"],
                                         self_assembled=lo["self_assembled"], subtotal=lo["subtotal"])
    rc = record["cost"]
    spread_rows = spreads(scenarios["pi_p21"])
    jetson_rows = [r for r in spreads(scenarios["jetson_direct_p21"])
                   if r["line"] not in scenarios["pi_p21"]["lines"]]
    return dict(
        study="The 75 cm build's prices, by designlab/cost.py (JETSON_PLEM_REVIEW.md section 5, task 9)",
        status="planning estimate; nothing is built, quoted, measured or adopted",
        p31=P31,
        body=BODY,
        script="stage2/jetson75/budget_75.py",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={rel: _sha(rel) for rel in INPUT_FILES},
        kinds=KINDS,
        tiers=list(TIERS),
        contingency_fractions=list(cost.CONTINGENCY),
        baseline=dict(
            record=RECORD, record_sha256=_sha(RECORD), reproduces_exactly=rep["exact"],
            differences=rep["differences"],
            recorded=dict(low=rc["low"], nominal=rc["nominal"], high=rc["high"]),
            recomputed=dict(low=rep["recomputed"]["low"], nominal=rep["recomputed"]["nominal"],
                            high=rep["recomputed"]["high"]),
            lines_recorded=rc["lines"], lines_recomputed=rep["recomputed"]["lines"],
            drivers=rc["drivers"], size_factor=record["H"] / 0.39, hip_height_m=record["H"],
            selection=record["selection"], record_catalog_ids=record_ids(record),
            self_assembled=base["self_assembled"], turnkey=base["turnkey"],
            line_basis=LINE_BASIS, class_labels=CLASS_LABEL),
        checks=checks(record, head_record, base),
        catalog=catalog_entries(record, head_record),
        blocks=blocks,
        scenarios=scenarios,
        conditions=CONDITIONS,
        least_certain=dict(scenario="pi_p21", rows=spread_rows, jetson_only_scenario="jetson_direct_p21",
                           jetson_only_rows=jetson_rows,
                           direct_only_lines=[r["line"] for r in jetson_rows
                                              if r["line"] not in scenarios["jetson_mcu_p21"]["lines"]],
                           jetson_compute_line=scenarios["jetson_mcu_p21"]["lines"]["compute"]),
        derived=derived(record, head_record, blocks, scenarios, spread_rows),
        pr668_slips=slips(record),
    )


# ---------------------------------------------------------------------------------------------------------------
# Tables (``md``): printed from the record on disk, so the document's tables are the record's.

def _usd(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v:,.0f}" if abs(v - round(v)) < 0.005 else f"{sign}${v:,.2f}"


def _plus(v: float) -> str:
    return ("+" if v >= 0 else "") + _usd(v)


def _three(values, f=_usd) -> str:
    return " / ".join(f(v) for v in values)


def _table(head: list, rows: list) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return re.sub(r"(?<=\S)  +(?=\S)", " ", "\n".join(out))


def _para(text: str) -> str:
    """A paragraph wrapped at 120 characters, like the study's documents, followed by a blank line."""
    return textwrap.fill(re.sub(r"  +", " ", text.strip()), width=120, break_long_words=False,
                         break_on_hyphens=False) + "\n"


def md(doc: dict) -> str:
    b, ck, out = doc["baseline"], doc["checks"], []
    out.append("### Table 1. The round-two record's cost block, recomputed\n")
    out.append(_para(f"`{b['record']}` (sha256 `{b['record_sha256'][:12]}`). The recorded and the recomputed values "
                     "are equal on every line, so one set of numbers is shown. Kind c+d: catalog prices (c) plus the "
                     "model's allowance (d). Kind d: the cost model."))
    rows = []
    for name, usd in b["lines_recorded"].items():
        lb = b["line_basis"][name]
        same = "yes" if b["lines_recomputed"][name] == usd else "NO"
        rows.append([f"`{name}`", *[_usd(v) for v in usd], lb["kind"], same, lb["basis"]])
    rows.append(["**Total (turnkey)**", *[f"**{_usd(b['recorded'][t])}**" for t in doc["tiers"]], "d",
                 "yes" if b["reproduces_exactly"] else "NO", "the sum of the lines above, as recorded"])
    rows.append(["Self-assembled", *[_usd(v) for v in b["self_assembled"]["total"]], "d", "",
                 "the same without `integration` and the contingency on it"])
    out.append(_table(["Line", "Low", "Nominal", "High", "Kind", "Equal", "What it is"], rows))

    out.append("\n### Table 2. The variants, each as a difference from the record\n")
    out.append(_para("Each row is one line's change. A block's row \"Block, with contingency\" is the change in the "
                     "record's turnkey total when that block alone is applied: its lines with the cost model's "
                     "contingency (x 1.053 / 1.072 / 1.084). Totals are rounded to the dollar after summing, so a "
                     "block's figure can be $1 off its lines times those factors, and a difference of two totals in "
                     "Table 4 can be $1 off a block's figure. Kind c+d on an added line: the price is the catalog's "
                     "(c), the count or the choice of part is an assumption (d)."))
    for name, blk in doc["blocks"].items():
        out.append(_para(f"**`{name}`: {blk['label']}.**"))
        rows = []
        for key, line in blk["lines"].items():
            src = line["basis"]
            if "catalog" in line:
                ids = dict.fromkeys(f"`{c['id']}` at ${c['price_usd']:,.2f}" for c in line["catalog"])
                src = "Catalog: " + ", ".join(ids) + ". " + src[0].upper() + src[1:]
            if key == "compute":
                cj, cp = doc["catalog"][JETSON], doc["catalog"][PI5]
                cd = doc["derived"]["compute_difference"]["survey_set_usd"]
                src = (f"Catalog: `{JETSON}` at ${cj['price_usd']:,.0f} in place of `{PI5}` at "
                       f"${cp['price_usd']:,.0f}; the model's allowance for storage, fan duct, mounts and leads is "
                       f"unchanged. Not like for like on storage: the Pi 5 set's price already holds a USB SSD and "
                       f"the dev kit's holds none; the compute survey's set prices, each with its storage, are "
                       f"${cd[JETSON]:,.0f} and ${cd[PI5]:,.0f} (COMPUTE.md; the dev kit's holds the survey's "
                       f"${cd['jetson_set_minus_kit']:,.0f} estimate for a drive), a difference of "
                       f"${cd['difference']:,.0f}")
            if key == "covers":
                lv = ck["covers_pa11"]["lever"]
                src = (f"cost.py's material path: PA11 with walls x{lv['wall_scale']:g} in place of PA12 at full "
                       "walls (the study's own lever, re-priced on this body; an estimate, since the lever was "
                       "never run on it)")
            if "caution" in line:
                src += ". Caution: " + line["caution"]
            label = line["label"] if key in EXTRA else f"`{key}` (a line of the record, changed)"
            rows.append([label, *[_plus(v) for v in line["usd"]], line["kind"], src[0].upper() + src[1:]])
        for u in blk["unchanged"]:
            rows.append([f"`{u['line']}` (a line of the record, unchanged at {_three(u['usd'])})", "+$0", "+$0",
                         "+$0", u["kind"], u["note"][0].upper() + u["note"][1:]])
        rows.append(["**Block, with contingency (turnkey)**", *[f"**{_plus(v)}**" for v in blk["turnkey"]], "d",
                     "Every line above" if "lower" in blk else ""])
        if "lower" in blk:
            lo = blk["lower"]
            rows.append([f"**The same if {lo['condition']}**", *[f"**{_plus(v)}**" for v in lo["turnkey"]], "d",
                         "Without " + " and ".join(f"`{k}`" for k in lo["without"])])
        if name == "head_x":
            hy = doc["derived"]["head_yaw_only"]
            rows.append(["**The same with the jaw deferred (P2, P29): the neck-yaw servo, its bracket and the "
                         "bearing**", *[f"**{_plus(v)}**" for v in hy["turnkey_with_bearing"]], "d",
                         f"An estimate: the head's record less the jaw servo (${hy['servo_price_usd']:,.0f}, c) and "
                         "its bracket, priced by the cost model; no record of that body exists. Without the bearing: "
                         + _three(hy["turnkey_servo_and_bracket"], _plus)])
        out.append(_table(["Line", "Low", "Nominal", "High", "Kind", "Source and derivation"], rows))
        out.append("")

    out.append("### Table 3. What the record already holds of P21's items\n")
    p21 = doc["blocks"]["p21"]["lines"]
    fr, cv, tr, dc = ck["frame_6061"], ck["covers_pa11"], ck["pad_treads"], ck["dc_input_and_boom"]
    rl = cv["rule_on_the_recorded_lever"]
    rows = [
        ["6061-T6 frame (2.54 mm plates where a leg bench test allows)", "yes" if fr["in_the_record"] else "NO",
         fr["how"][0].upper() + fr["how"][1:], "none ($0)"],
        ["MJF PA11 coverings, walls about a quarter thinner",
         "no: PA12 at full walls" if not cv["in_the_record"] else "YES",
         cv["how"][0].upper() + cv["how"][1:] + f". The re-pricing rule, run on the body the lever was recorded on, "
         f"gives its recorded covers line within ${rl['max_abs_diff_usd']:.2f} and its totals "
         f"{_three(rl['totals_rule'])} against {_three(rl['totals_recorded'])} recorded",
         "`covers` " + _three(p21["covers"]["usd"], _plus)],
        ["Cast polyurethane pad treads, measured on the owner's floors", tr["in_the_record"],
         tr["how"][0].upper() + tr["how"][1:], "`pad_treads` " + _three(p21["pad_treads"]["usd"], _plus)],
        ["The battery, with an external DC input", "the battery: yes; the input: no; the clamp: not known",
         dc["how"][0].upper() + dc["how"][1:] + ". Not checked: " + dc["not_checked"],
         "`dc_input` " + _three(p21["dc_input"]["usd"], _plus)],
        ["The cable hung from a boom", "no (the bench supply: yes)",
         f"cost.py's bench lot covers \"{dc['lot_scope']}\"; MATERIALS.md section 6 prices a boom only \"if the "
         "bench lot's catch frame cannot carry it\"", "`cable_boom` " + _three(p21["cable_boom"]["usd"], _plus)],
    ]
    out.append(_table(["P21 item", "In the record's totals?", "How it was checked", "Line added (low / nominal / "
                       "high)"], rows))

    out.append("\n### Table 4. Totals: turnkey and self-assembled\n")
    out.append(_para("Turnkey is the cost model's total. Self-assembled leaves out the subcontract integration line "
                     "and the contingency on it. Every total is kind d. A Jetson build has two rows: with every "
                     "line of its blocks, and without the lines a kept lot may already hold (Table 2)."))
    rows = []
    for name, s in doc["scenarios"].items():
        rows.append([f"`{name}`: {s['label']}", *[_usd(v) for v in s["turnkey"]["total"]],
                     *[_usd(v) for v in s["self_assembled"]["total"]]])
        if "lower" in s:
            lo = s["lower"]
            rows.append([f"`{name}`, if {lo['condition']}", *[_usd(v) for v in lo["turnkey"]["total"]],
                         *[_usd(v) for v in lo["self_assembled"]["total"]]])
    out.append(_table(["Build", "Turnkey low", "Turnkey nominal", "Turnkey high", "Self-assembled low",
                       "Self-assembled nominal", "Self-assembled high"], rows))

    out.append("\n### Table 5. Where the prices are least certain, the widest lines first\n")
    q = doc["least_certain"]
    out.append(_para(f"The lines of `{q['scenario']}`, the adopted build (P14) with P21's items, ranked by high minus "
                     "low: the wider that gap, the less the estimate says about the line's price. The contingency "
                     "is not ranked: it is 5.3 / 7.2 / 8.4 % of whatever the other lines come to. A Jetson build "
                     "has the same lines in the same order, with its `compute` line at "
                     f"{_three(q['jetson_compute_line'])} and the lines of the second table. The last column says "
                     "what a quote for the line would have to cover; no quote has been asked for."))

    def spread_table(lines: list) -> str:
        rows = [[f"`{r['line']}`", _usd(r["spread"]), _three(r["usd"]), b["class_labels"][r["cls"]],
                 r["quote_scope"]] for r in lines]
        return _table(["Line", "High minus low", "Low / nominal / high", "How it was priced",
                       "What a quote would cover"], rows)

    out.append(spread_table(q["rows"]))
    out.append("")
    out.append(_para(f"Lines only a Jetson build adds (`{q['jetson_only_scenario']}`). "
                     + ", ".join(f"`{k}`" for k in q["direct_only_lines"])
                     + " are added only if the Jetson drives the legs directly."))
    out.append(spread_table(q["jetson_only_rows"]))

    out.append("\n### Table 6. PR #668's price slips, corrected\n")
    rows = []
    for s in doc["pr668_slips"]:
        if "catalog_price_usd" in s:
            fact = f"`{s['catalog_id']}` is ${s['catalog_price_usd']:,.0f} in `designlab/data/parts.json` (c)"
        else:
            fact = (("That id is in" if s["pr668_id_exists"] else "No such id in") + " `designlab/data`; the "
                    f"record names `{s['right_id']}`")
        rows.append([s["what"], s["pr668_unverified"], fact, s["used_here"]])
    out.append(_table(["Slip", "PR #668 (unverified)", "The catalog", "Used here"], rows))
    return "\n".join(out) + "\n"


def main(argv: list) -> None:
    if argv[1:] == ["md"]:
        print(md(json.loads(OUT.read_text())), end="")
        return
    doc = build()
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    b, s = doc["baseline"], doc["scenarios"]
    print(f"baseline reproduced exactly: {b['recorded']}")
    for name in s:
        print(f"{name:26s} turnkey {s[name]['turnkey']['total']}  self-assembled {s[name]['self_assembled']['total']}")
        if "lower" in s[name]:
            lo = s[name]["lower"]
            print(f"{'  without ' + '+'.join(lo['without']):26s} turnkey {lo['turnkey']['total']}  self-assembled "
                  f"{lo['self_assembled']['total']}")
    print(f"wrote {OUT.relative_to(STUDY)}")


if __name__ == "__main__":
    main(sys.argv)
