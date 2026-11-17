"""The servo model of gate 1: its parameter table, parameter object, draw, named settings and run-time layer
(``stage2/design/servo_model/SERVO_MODEL_DESIGN.md``, package S1; the lead's rulings ``LEAD_RULINGS.md``, both
lists; the owner's condition 2, ``DECISION_LOG.md`` P41 and ``BUY_GATE_DRAFT.md``).

Simulation only.  This module is a prospective design built in code: no value in it is measured on a part of this
build, and nothing it runs is a result.  The parameter table (``TABLE``) gives every value its source and kind as the
design note's section 3 gives it; ``describe_lines`` prints them.

What it holds
-------------
* ``TABLE``: the design's 39 rows (section 3), each with today's value, the nominal, the range per run, the
  distribution, the source and its kind (``KINDS``: the maker's catalogue value; third-party; the study's
  assumption; estimate; no source) and what would replace it.  ``CONSTANTS``: the layer's fixed numbers, with theirs.
  ``TABLE_SHA256`` hashes the table, the constants, the draw rule and the named settings.
* ``ServoParams``: one run's values (per drive class, per drive, per run).  ``setting(name)`` gives a named setting
  (``NAMED``: section 4's table; ``JUDGED`` and ``REPORTED`` as P41 has them); ``panel_draw(s)`` the drawn panel
  ``D`` of bar seed ``s`` (``SeedSequence([s, 3])``, the rotor inertia and the bus resistance at nominal);
  ``training_draw(e)`` a training episode's draw (``SeedSequence([e, 2])``; refused without an episode seed).
* ``served(p, params)``: the run-time layer on a Stage 1 plant of body ``75-M-H-10`` (``LAYER_RULE``), a context
  manager that restores the plant exactly on exit (``HOOKS_RULE``).  The model file on disk, the screened model,
  the references and every record are untouched: the layer changes arrays of the plant's model in memory.

Not here (other packages of section 6): the env option and ``g8s.py`` (S2), the served row and gate 1's runner (S3),
any measurement with the layer (S4).  The trunk mass of ``M-``, ``M+``, ``W1`` and ``W2`` is not the layer's: the
setting carries the factor (``ServoParams.mass``) for the caller, who applies it before the reset on both sides
(design 5.1; ``env.set_trunk_mass``).
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace

import mujoco
import numpy as np

from .. import carpet, catalog, power
from ..stage1 import drive
from ..stage1 import plant as plant_mod

# ------------------------------------------------------------------------------------------------
# The body, its drive classes and today's values (the catalog's, as the plant has them)
# ------------------------------------------------------------------------------------------------
#: The one body the table is for (design, status line): 14 drives, the Raspberry Pi 5 as computer.  Not the 60 cm
#: bodies (their own table: choice 14), not the Jetson body (another auxiliary input), not the expressive heads.
BODY = "75-M-H-10"
#: The record's bus: today's envelope is rated on it (``plant.build``), and ``today`` keeps it.
BUS_V = 24.0
CLASSES = ("X4", "X6", "neck", "tail")
LEG_CLASSES = ("X4", "X6")
BOX_CLASSES = ("neck", "tail")
PARTS = {"X4": "myactuator_rmd_x4_p36_36_e_l", "X6": "myactuator_rmd_x6_p20_60_e",
         "neck": "robotis_xm540_w270_t", "tail": "feetech_sts3215_c018"}
_GROUP_CLASS = dict(hip_roll="X4", hip_pitch="X6", hip_yaw="X4", knee="X4", ankle_a="X4", ankle_b="X4")
#: The body's 14 drives (the plant's actuators, in its order) and the class of each.
DRIVES = (("neck_pitch_servo", "neck"), ("tail_yaw_servo", "tail")) + tuple(
    (f"{s}_{g}_servo", _GROUP_CLASS[g]) for s in ("left", "right")
    for g in ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b"))
ACTUATORS = tuple(a for a, _ in DRIVES)
LEG_DRIVES = tuple(a for a, c in DRIVES if c in LEG_CLASSES)

_ACT = {c: catalog.actuator(pid) for c, pid in PARTS.items()}


def _bus_of(c: str) -> float | None:
    return None if _ACT[c].box is not None else BUS_V


#: Today's corner (the speed up to which the peak is available), zero-torque speed, planning peak and reflected
#: inertia of each class, by the rules the plant builds with (``drive.drives``: ``Actuator.speed_at``,
#: ``drive.no_load_speed_at``, ``Actuator.peak``, the catalog armature with a 1e-5 floor; assumptions A2 and A7).
TODAY_CORNER = {c: float(_ACT[c].speed_at(_bus_of(c))) for c in CLASSES}
TODAY_ZERO = {c: float(drive.no_load_speed_at(_ACT[c], _bus_of(c))) for c in CLASSES}
PLANNING_PEAK = {c: float(_ACT[c].peak) for c in CLASSES}
A7 = {c: float(max(_ACT[c].armature, 1e-5)) for c in CLASSES}
#: The leg drives' nominal voltage [M] (RES:68, 136): the zero-torque speed scales with the bus below it.
V_NOMINAL = {c: float(_ACT[c].nominal_voltage) for c in LEG_CLASSES}

# ------------------------------------------------------------------------------------------------
# The parameter table (design section 3)
# ------------------------------------------------------------------------------------------------
KINDS = ("maker's catalogue value", "third-party", "the study's assumption", "estimate", "no source")
MAKER, THIRD, ASSUMED, ESTIMATE, UNSOURCED = KINDS
UNIFORM, PER_DRIVE, LOG_UNIFORM, INTEGER = "uniform", "uniform, per drive", "log-uniform", "integer uniform"
FIXED, ABOVE_CORNER, SETTING_ONLY, NOT_LAYER = ("fixed", "uniform, above the drawn corner", "a setting, not drawn",
                                                "not the layer's")
#: A cell that holds today's value (the plant's own, as found): ``today`` in the table, ``None`` in a ServoParams.
TODAY = "today"


def _row(param, classes, unit, today, nominal, low, high, distribution, kind, source, replaced_by,
         maker_related=False) -> dict:
    classes = (classes,) if isinstance(classes, str) else tuple(classes)

    def per(v):
        return tuple(v) if isinstance(v, (tuple, list)) else (v,) * len(classes)

    return dict(param=param, classes=classes, unit=unit, today=per(today), nominal=per(nominal), low=per(low),
                high=per(high), distribution=distribution, kind=kind, source=source, replaced_by=replaced_by,
                maker_figure_for_related_quantity=bool(maker_related))


#: Design section 3, row by row (39 rows; "neck, tail" rows are one row for two classes, as there).  Sources and
#: kinds as the table gives them; marks: [M] the maker's catalogue value as the research record transcribes it, [T]
#: a third party's measurement or fit of its own unit ("file": the saved file was read; "summary": a web summary
#: only), [A] the study's assumption, [E] an estimate (arithmetic), "no source" a placeholder.  RES =
#: designlab/data/research/actuators_2026-11-07.json, BENCH = stage2/jetson75/BENCH_SHEET.md, LAT =
#: stage2/jetson75/LATENCY_AND_BUS.md, MAT = round2/refine/MATERIALS.md, N1-N6 the design's proposed bench additions.
TABLE = (
    _row("fric_c", "X4", "N m", 0.15, 0.2, 0.1, 0.6, UNIFORM, ASSUMED,
         "nominal [A, this note]: the maker's backdrive torque, 0.2 N m [M] (RES:85): another quantity (the torque "
         "that turns the unpowered output), taken as the dry term; range [E]; the V4 sibling's 1.14 N m [M] (RES:52) "
         "is outside it", "sheet B step 1 (BENCH:564-567)", maker_related=True),
    _row("fric_c", "X6", "N m", 0.15, 1.6, 0.2, 2.4, UNIFORM, ASSUMED,
         "nominal [A, this note]: the maker's backdrive torque, 1.6 N m [M] (RES:154), taken as on the X4; low end: "
         "the X6-50-L's backdrive 0.2 N m [M] (RES:186); high end [E]", "sheet B step 1", maker_related=True),
    _row("fric_c", "neck", "N m", 0.15, 0.12, 0.03, 0.30, UNIFORM, THIRD,
         "no source for this model; nominal [T, file, another model]: the MX-106 fit's low-speed (Stribeck) term, "
         "0.122 N m (its constant term is 1.0e-5 N m); range [E]", "sheet B steps 1 and 3"),
    _row("fric_c", "tail", "N m", 0.15, 0.05, 0.03, 0.15, UNIFORM, THIRD,
         "the 7.4 V version's fit, 0.052 N m [T, file]; top = A18; no source for the 12 V part",
         "sheet B steps 1 and 3"),
    _row("fric_v", "X4", "N m s/rad", 0.2, 0.10, 0.0, 0.25, UNIFORM, UNSOURCED,
         "no source; nominal [E], set so that the total at the maker's rated point equals today's (1.9 N m)",
         "sheet B step 2 (BENCH:568-572); it stops at 4 rad/s, the model uses 8.7"),
    _row("fric_v", "X6", "N m s/rad", 0.2, 0.05, 0.0, 0.10, UNIFORM, UNSOURCED,
         "no source; nominal [E], the same rule (3.4 against today's 3.35 N m)", "sheet B step 2"),
    _row("fric_v", "neck", "N m s/rad", 0.2, 0.05, 0.01, 0.20, UNIFORM, THIRD,
         "another model's fit, 0.051 [T, file]; top = A18", "sheet B step 3"),
    _row("fric_v", "tail", "N m s/rad", 0.2, 0.06, 0.03, 0.20, UNIFORM, THIRD,
         "the 7.4 V version's fit, 0.059 [T, file]; top = A18", "sheet B step 3"),
    _row("fric_load", "X4", "share of motor torque", 0.0, 0.08, 0.0, 0.15, UNIFORM, UNSOURCED,
         "no source; [E]; upper bound 0.30 [E from M]", "proposed N5"),
    _row("fric_load", "X6", "share of motor torque", 0.0, 0.05, 0.0, 0.10, UNIFORM, UNSOURCED,
         "no source; [E]; upper bound 0.20 [E from M]", "proposed N5"),
    _row("fric_load", "neck", "share of motor torque", 0.0, 0.20, 0.0, 0.30, UNIFORM, THIRD,
         "other models' fits: 0.198, 0.311 [T, file]", "proposed N5"),
    _row("fric_load", "tail", "share of motor torque", 0.0, 0.0, 0.0, 0.25, UNIFORM, UNSOURCED,
         "no source (the third-party fit has no load term)", "proposed N5"),
    _row("fric_joint", "all", "x", 1.0, 1.0, 0.8, 1.25, PER_DRIVE, UNSOURCED,
         "no source; [E]: left and right differ (today the env draws one factor 0.5 to 1.5 for all hinges, "
         "r5env.py:139-140; with the layer on, those factors are 1)", "sheet B on each unit; step 4 on the leg"),
    _row("play_deg", "X4", "deg (total)", 0.0, 0.25, 0.0, 1.0, PER_DRIVE, ASSUMED,
         "nominal [A, this note]: the maker's upper limit, 15 arcmin [M] (RES:75), taken as the value; top [E from "
         "T, summary]: 4.0 x that limit, the ratio one third-party measurement found on an older, larger sibling "
         "(0.334 deg against 0.083 deg advertised), also the buy gate draft's 1 deg; ankle push rods: no source, "
         "not included", "proposed N1", maker_related=True),
    _row("play_deg", "X6", "deg (total)", 0.0, 0.25, 0.0, 1.0, PER_DRIVE, ASSUMED,
         "as the X4; the limit is [M] (RES:143; the store page says 10 arcmin, RES:154)", "proposed N1",
         maker_related=True),
    _row("play_deg", "neck", "deg (total)", 0.0, 0.0, None, None, FIXED, ASSUMED,
         "not modelled: left out until it is known which side of the play the sensor is on (design 2b); for that "
         "day: 15 arcmin [M] (RES:1502); [T, summary]: 0.11 to 0.29 deg on a family member", "proposed N6, then N1"),
    _row("play_deg", "tail", "deg (total)", 0.0, 0.0, None, None, FIXED, ASSUMED,
         "not modelled, as the neck; for that day: <= 0.5 deg [M] (RES:1566); [T, summary]: 0.6 to 0.9 deg "
         "unloaded, 1.3 deg loaded", "proposed N6, then N1"),
    _row("ts_corner", "X4", "rad/s at 24 V", TODAY_CORNER["X4"], 5.76, 5.76, TODAY, UNIFORM, ESTIMATE,
         "nominal [E from M]: the line through the maker's rated and no-load points (RES:63-67); top: today's rule "
         "(Actuator.speed_at(24 V), 83 rpm; the design prints 8.69)", "proposed N2"),
    _row("ts_corner", "X6", "rad/s at 24 V", TODAY_CORNER["X6"], 4.16, 2.15, TODAY, UNIFORM, ESTIMATE,
         "nominal [E from M]: the line through the maker's two 48 V points (RES:131-135), kept at its slope and "
         "shifted to the halved no-load speed; low end [E from M; R, checked]: the corner on the back-EMF reading, "
         "24 V over 16 Vdc/krpm (RES:154) with the copper slope Kt^2 / 3R; top: today's corner (153 rpm halved; the "
         "design prints 8.01); no 24 V data exist and the maker's figures do not fit each other (question 3)",
         "proposed N2"),
    _row("ts_corner", ("neck", "tail"), "rad/s", (TODAY_CORNER["neck"], TODAY_CORNER["tail"]), TODAY, None, None,
         FIXED, ASSUMED, "today's lines are inside the maker's stall-to-no-load lines [E from M]", "none planned"),
    _row("ts_zero", "X6", "rad/s at 24 V", TODAY_ZERO["X6"], TODAY, 8.01, TODAY, ABOVE_CORNER, ASSUMED,
         "ruled by the lead on 16 Nov 2026: drawn while the maker has not answered question 3; nominal and top [A]: "
         "today's rule, the maker's 48 V no-load speed of 176 rpm halved (RES:135; catalog.py:77-82); low end [E "
         "from M]: the no-load speed the back-EMF figure gives at 24 V, 8.01 rad/s (RES:154); drawn after the corner, "
         "uniform between the larger of 8.01 rad/s and the corner + 1.20 rad/s, and today's (the lead's second list)",
         "the maker's answer (question 3); proposed N2"),
    _row("armature", "X4", "kg m^2", A7["X4"], TODAY, TODAY, 0.106, LOG_UNIFORM, ASSUMED,
         "nominal and low end [A7]; high [E from M]: 0.000082 kg m^2 x 36^2 (RES:85), if rotor-side; between them the "
         "V4 sibling's 0.3 kg cm^2 x 36^2 = 0.039 [E from M, RES:52]; the nominal is at the bottom of a 33-fold range "
         "(question 1 to the maker)", "the maker's answer (question 1); proposed N5"),
    _row("armature", "X6", "kg m^2", A7["X6"], TODAY, TODAY, 0.0254, LOG_UNIFORM, ASSUMED,
         "nominal and low end [A7]; high [E from M]: 0.66 kg cm^2 x 19.612^2 (RES:154), if rotor-side",
         "the maker's answer (question 1); proposed N5"),
    _row("armature", "neck", "kg m^2", A7["neck"], TODAY, 0.0002, 0.026, LOG_UNIFORM, ASSUMED,
         "no source for this model; nominal [A]: the study's rule (A7); high: other models [T]: 0.0058 (summary), "
         "0.026 (file); the nominal keeps the study's rule although the friction nominals come from another model's "
         "fit: flagged (design section 8, choice 16)", "proposed N5"),
    _row("armature", "tail", "kg m^2", A7["tail"], 0.028, 0.0002, 0.028, LOG_UNIFORM, THIRD,
         "nominal [T, file, the 7.4 V version]: the fit's 0.0284 kg m^2, the fit that gives this servo's friction "
         "nominals; another project's model of that version has 0.027 [T, file]; low end: today's rule; no source "
         "for the 12 V part; the nominal is the top of the range (ruled by the lead on 16 Nov 2026), so J+ equals N "
         "for the tail; with it the tail's nominal kv is 3.59 N m s/rad, not today's 2.72", "proposed N5"),
    _row("cmd_lag_ms", "X4", "ms", 0.0, 0.5, 0.0, 4.0, UNIFORM, ESTIMATE,
         "nominal [E from M]: the winding's L/R, 0.45 ms (RES:85); top [E]; no source for the response; the budget's "
         "drive-side delay D already holds this rise (design 2d)", "proposed N4"),
    _row("cmd_lag_ms", "X6", "ms", 0.0, 1.2, 0.0, 4.0, UNIFORM, ESTIMATE,
         "nominal [E from M]: the winding's L/R, 1.24 ms (RES:154); top [E]", "proposed N4"),
    _row("cmd_lag_ms", ("neck", "tail"), "ms", 0.0, 0.0, 0.0, 20.0, UNIFORM, UNSOURCED,
         "no source (LAT:346-347)", "BENCH:347-370"),
    _row("cmd_hold", ("neck", "tail"), "phase (physics steps)", "every step", "every 5th step, phase 0", 0, 4,
         INTEGER, ASSUMED,
         "the bus plan: neck and tail targets at 100 Hz on their own serial port [A] (LAT:335-336); the phase is "
         "drawn", "BENCH:347-370"),
    _row("kp_scale", "neck", "x", 1.0, 1.0, 0.3, 1.2, UNIFORM, UNSOURCED,
         "no source (register gains); the nominal keeps the model's stiffness until sheet A (choice 17)",
         "sheet A steps 1 to 3 (BENCH:519-535)"),
    _row("kp_scale", "tail", "x", 1.0, 1.0, 0.3, 1.2, UNIFORM, ASSUMED,
         "nominal [A]: the model's stiffness kept until sheet A (choice 17); low end [E from T (file) and M]: 15.6 N "
         "m/rad at the default register, 0.32 of the model's", "sheet A"),
    _row("kv_scale", "neck", "x", 1.0, 1.0, 0.5, 1.2, UNIFORM, ASSUMED,
         "nominal [A]: the model's damping kept; low end [E from M]: stall / no-load = 3.4 N m s/rad, 0.81 of the "
         "model's", "sheet A step 4"),
    _row("kv_scale", "tail", "x of the nominal kv under the new inertia", 1.0, 1.0, 0.15, 1.2, UNIFORM, ASSUMED,
         "nominal [A]: 1 of the nominal kv under the new inertia, 3.59 N m s/rad (ruled by the lead on 16 Nov 2026); "
         "low end [E from M]: stall / no-load = 0.62 N m s/rad, 0.17 of it; top 4.3 N m s/rad", "sheet A step 4"),
    _row("gain_cap", ("X4", "X6"), "-", "none", "none", "none", "kp <= 500 N m/rad, kv <= 5 N m s/rad",
         SETTING_ONLY, ASSUMED,
         "nominal 'none' [A, this note]: the caps (KP 0 to 500, KD 0 to 5) are from a sibling's manual [M via search "
         "extract] (LAT:442) and their units are not confirmed, so they are not taken as this part's until the "
         "maker's document is read (question 2)", "sheet A step 1 (BENCH:519-525); the maker's protocol document"),
    _row("enc_offset", "all", "rad", 0.0, 0.0, -0.015, 0.015, PER_DRIVE, ASSUMED,
         "nominal 0 [A, this note]: an offset has no preferred sign; range: a third-party default [T, code: R3 "
         "section 2b]; the maker's accuracy on the X4, 0.4 deg = 0.007 rad [M] (RES:85), is inside it",
         "proposed N6"),
    _row("sense_steps", "all", "physics steps", 0, 0, 0, 2, INTEGER, ASSUMED,
         "the budget's feedback age S [A] (LAT:196-199); a measured feedback age goes here (ruled by the lead on 16 "
         "Nov 2026)", "bench S (BENCH:93-94)"),
    _row("bus_v0", "bus", "V", BUS_V, 22.2, 21.0, 25.2, UNIFORM, MAKER,
         "nominal: the pack's nominal voltage [M] (designlab/data/parts.json:207); range [A11]",
         "not on the sheet (the pack is an optional part there, BENCH:186): a log of the pack's voltage over a run, "
         "with proposed N3"),
    _row("bus_r", "bus", "ohm", 0.0, 0.05, 0.02, 0.15, UNIFORM, UNSOURCED,
         "no source (the pack's entry has none, parts.json:193-216); [E]; for scale: 5 m of tether cable is 0.13 ohm "
         "[E from MAT:233]; the top of the range takes a 21.0 V bus below the drives' 20 V limit at the walk's summed "
         "peak", "proposed N3"),
    _row("latency_steps", "all", "physics steps", "training 0 to 12; judged 0 to 5", "unchanged", None, None,
         NOT_LAYER, ASSUMED,
         "not the layer's: the env's draw (r5env.py:137, rules J1) and the record's delay (bar.py:134-136), through "
         "speedfit.servo_delay; the drawn panel D keeps the record's delay", "bench T (BENCH:84)"),
)
#: (param, class) -> (row, index of the class in the row); "all" rows answer for every class.
_CELL = {}
for _r_ in TABLE:
    for _i, _c in enumerate(_r_["classes"]):
        _CELL[(_r_["param"], _c)] = (_r_, _i)
del _r_, _i, _c


def cell(param: str, cls: str, column: str):
    """The table's ``column`` ("today", "nominal", "low", "high") of ``param`` for class ``cls``."""
    hit = _CELL.get((param, cls)) or _CELL.get((param, "all"))
    if hit is None:
        raise KeyError(f"the table has no row {param!r} for {cls!r}")
    row, i = hit
    return row[column][i]


def _number(param: str, cls: str, column: str) -> float:
    """A cell as a number: ``TODAY`` read as today's value of the class (for the draw's range ends)."""
    v = cell(param, cls, column)
    if v != TODAY:
        return float(v)
    return float({"ts_corner": TODAY_CORNER, "ts_zero": TODAY_ZERO, "armature": A7}[param][cls])


def kind_counts() -> dict:
    """How many rows have each kind, and how many take a maker's figure for a related quantity as the value."""
    out = {k: sum(r["kind"] == k for r in TABLE) for k in KINDS}
    out["of which a maker's figure for a related quantity, taken as the value"] = sum(
        r["maker_figure_for_related_quantity"] for r in TABLE)
    return out


#: The layer's fixed numbers (not drawn), each with its source and kind.
CONSTANTS = dict(
    copper_loss_W_per_Nm2=dict(
        value={c: 3.0 * _ACT[c].phase_resistance / _ACT[c].kt ** 2 for c in LEG_CLASSES}, kind=ESTIMATE,
        source="c_cu = 3 R_phase / Kt^2 [E from M] (RES:71-72, 139-140; catalog kt and phase_resistance)"),
    drive_efficiency=dict(value=float(power.ETA_DRIVE), kind=ASSUMED, source="A6 (designlab/power.py ETA_DRIVE)"),
    auxiliary_input_W=dict(value="the body's record, power.aux.input_w (20.3 W on 75-M-H-10: the Raspberry Pi 5's)",
                           kind=ASSUMED, source="the study's power budget in the body's record [S] (REC:298); no credit "
                                                "for braking power, as A6"),
    supply_floor_V=dict(value=min(float(_ACT[c].voltage_min) for c in LEG_CLASSES), kind=MAKER,
                        source="the leg drives' lower supply limit [M] (RES:69, 137)"),
    nominal_voltage_V=dict(value=dict(V_NOMINAL), kind=MAKER, source="RES:68, 136"),
    load_filter_ms=dict(value=10.0, kind=ASSUMED,
                        source="the design's choice (2a): it breaks the loop between friction and torque"),
    box_update_steps=dict(value=5, kind=ASSUMED, source="100 Hz targets at 2 ms physics steps [A] (LAT:335-336)"),
    ts_zero_gap_rad_s=dict(value=1.20, kind=ASSUMED,
                           source="the lead's second list: today's gap between corner and zero (9.215 - 8.011)"),
    gain_cap=dict(value=dict(kp_N_m_per_rad=500.0, kv_N_m_s_per_rad=5.0), kind=MAKER,
                  source="a sibling's manual, KP 0 to 500 and KD 0 to 5, units not confirmed [M via search extract] "
                         "(LAT:442)"),
    x6l=dict(value=dict(peak_N_m=0.70 * 45.0, fric_c_N_m=0.2), kind=ASSUMED,
             source="the X6-50-L: A2's 0.70 x the maker's 45 N m peak [M] (RES:167), and its 0.2 N m backdrive [M] "
                    "(RES:186) taken as the dry term as on the other leg drives"),
    collapse=dict(value="V = V0 / 2 for the step", kind=ASSUMED,
                  source="design 2c: no root of the sag; the step is a fault and the run has left the supply window"),
)
#: No answer to question 3 to the maker (the X6 at 24 V) is in the model: the X6's zero-torque speed is drawn (ruled by
#: the lead on 16 Nov 2026; design section 3, row ts_zero).  An answer ends that draw.
X6_ZERO_DRAWN = True
TRAINING_TAG, PANEL_TAG = 2, 3
GAIN_CAP_KP, GAIN_CAP_KV = 500.0, 5.0
X6L_PEAK, X6L_FRIC_C = 0.70 * 45.0, 0.2
ZERO_GAP = 1.20
LOAD_FILTER_S = 0.010
HOLD_STEPS = 5
DT = 0.002

DRAW_RULE = (
    "one numpy Generator from SeedSequence([seed, tag]) (tag 2: a training episode's seed; tag 3: the drawn panel D's "
    "bar seed), drawn in this order: fric_c, fric_v, fric_load (each X4, X6, neck, tail, uniform); fric_joint (the 14 "
    "drives in the plant's actuator order, uniform); play_deg (the 12 leg drives in that order, uniform; the neck and "
    "tail are fixed at 0); ts_corner X4 then X6 (uniform); ts_zero X6 (uniform between the larger of 8.01 rad/s and "
    "the drawn corner + 1.20 rad/s, and today's 9.215; drawn while the maker has not answered question 3); armature "
    "X4, X6, neck, tail (log-uniform); cmd_lag_ms X4, X6, neck, tail (uniform); the box servos' hold phase (integer "
    "0 to 4); kp_scale neck, tail; kv_scale neck, tail (uniform); enc_offset (the 14 drives, uniform); sense_steps "
    "(integer 0 to 2); bus_v0; bus_r (uniform).  Every one is drawn every time, so the stream never shifts; the drawn "
    "panel D then sets armature and bus_r back to the nominal (design section 4).  Not drawn: latency_steps (the "
    "env's and the record's), gain_cap (a setting), the neck's and tail's play and torque-speed lines (fixed), the "
    "trunk mass (the env's factor; the panel keeps 1).  A training draw needs the episode's seed: none, none drawn")

# ------------------------------------------------------------------------------------------------
# The parameter object
# ------------------------------------------------------------------------------------------------
_PER_CLASS = ("fric_c", "fric_v", "fric_load", "ts_corner", "ts_zero", "peak", "armature", "cmd_lag_ms",
              "kp_scale", "kv_scale")
_PER_DRIVE = ("fric_joint", "play_deg", "enc_offset")


@dataclass(frozen=True)
class ServoParams:
    """One run's servo values (design sections 2 to 4).  Per class tuples follow ``CLASSES``, per drive tuples
    ``ACTUATORS``.  ``None`` in ``ts_corner``, ``ts_zero``, ``peak`` or ``armature`` is today's value of the plant as
    found (its envelope's corner, zero-torque speed and peak; its model's armature, rule A7).  ``friction`` False
    leaves today's friction arrays (A18, the ankle's on its joints) as found: the setting ``today``.  ``mass`` is
    the trunk mass factor, which the caller applies before the reset (design 5.1), never the layer."""

    name: str
    friction: bool
    fric_c: tuple
    fric_v: tuple
    fric_load: tuple
    fric_joint: tuple
    play_deg: tuple
    ts_corner: tuple
    ts_zero: tuple
    peak: tuple
    armature: tuple
    cmd_lag_ms: tuple
    cmd_hold: bool
    hold_phase: int
    kp_scale: tuple
    kv_scale: tuple
    gain_cap: bool
    enc_offset: tuple
    sense_steps: int
    bus_v0: float
    bus_r: float
    mass: float = 1.0
    seed: int | None = None
    tag: int | None = None

    def __post_init__(self):
        for f in _PER_CLASS:
            if len(getattr(self, f)) != len(CLASSES):
                raise ValueError(f"{f}: one value per class {CLASSES}")
        for f in _PER_DRIVE:
            if len(getattr(self, f)) != len(DRIVES):
                raise ValueError(f"{f}: one value per drive ({len(DRIVES)})")
        for (a, c), b in zip(DRIVES, self.play_deg, strict=True):
            if b < 0 or (c in BOX_CLASSES and b != 0):
                raise ValueError(f"{a}: play {b!r} deg (at least 0; the neck and tail have none until N6, design 2b)")
        if any(v < 0 for v in self.cmd_lag_ms):
            raise ValueError("cmd_lag_ms: at least 0")
        if any(v <= 0 for v in self.kp_scale + self.kv_scale + self.fric_joint):
            raise ValueError("kp_scale, kv_scale and fric_joint: above 0")
        if not (isinstance(self.hold_phase, int) and 0 <= self.hold_phase < HOLD_STEPS):
            raise ValueError(f"hold_phase {self.hold_phase!r}: an integer 0 to {HOLD_STEPS - 1}")
        if not (isinstance(self.sense_steps, int) and self.sense_steps >= 0):
            raise ValueError(f"sense_steps {self.sense_steps!r}: a whole number of physics steps, at least 0")
        if not (self.bus_v0 > 0 and self.bus_r >= 0):
            raise ValueError("bus_v0 above 0 and bus_r at least 0")
        for i, c in enumerate(CLASSES):
            if c in BOX_CLASSES and (self.ts_zero[i] is not None or self.ts_corner[i] is not None):
                raise ValueError(f"{c}: its torque-speed line is today's (design 2c: the box servos keep theirs)")
            if c in LEG_CLASSES and (self.kp_scale[i] != 1.0 or self.kv_scale[i] != 1.0):
                raise ValueError(f"{c}: kp_scale and kv_scale are the box servos' (design 2d)")
            z = TODAY_ZERO[c] if self.ts_zero[i] is None else self.ts_zero[i]
            k = TODAY_CORNER[c] if self.ts_corner[i] is None else self.ts_corner[i]
            if not z > k:
                raise ValueError(f"{c}: zero-torque speed {z!r} is not above the corner {k!r}")

    def as_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in _PER_CLASS:
                v = dict(zip(CLASSES, v, strict=True))
            elif f.name in _PER_DRIVE:
                v = dict(zip(ACTUATORS, v, strict=True))
            out[f.name] = v
        return out


def _col(param: str, column: str) -> tuple:
    """A per-class tuple of a table column, ``TODAY`` as None."""
    return tuple(None if (v := cell(param, c, column)) == TODAY else v for c in CLASSES)


def _box_col(param: str) -> tuple:
    """The box servos' nominal of a per-class scale (the leg drives have none: 1)."""
    return tuple(float(cell(param, c, "nominal")) if c in BOX_CLASSES else 1.0 for c in CLASSES)


def _per_drive(values_by_class: tuple) -> tuple:
    return tuple(float(values_by_class[CLASSES.index(c)]) for _, c in DRIVES)


def nominal() -> ServoParams:
    """``N``: the Nominal column of the table (condition 1's values)."""
    return ServoParams(
        name="N", friction=True, fric_c=_col("fric_c", "nominal"), fric_v=_col("fric_v", "nominal"),
        fric_load=_col("fric_load", "nominal"), fric_joint=(1.0,) * len(DRIVES),
        play_deg=_per_drive(_col("play_deg", "nominal")), ts_corner=_col("ts_corner", "nominal"),
        ts_zero=(None,) * len(CLASSES), peak=(None,) * len(CLASSES), armature=_col("armature", "nominal"),
        cmd_lag_ms=_col("cmd_lag_ms", "nominal"), cmd_hold=True, hold_phase=0, kp_scale=_box_col("kp_scale"),
        kv_scale=_box_col("kv_scale"), gain_cap=False, enc_offset=(0.0,) * len(DRIVES),
        sense_steps=int(cell("sense_steps", "all", "nominal")), bus_v0=float(cell("bus_v0", "bus", "nominal")),
        bus_r=float(cell("bus_r", "bus", "nominal")))


def _friction_set(which: str) -> dict:
    return {p: _col(p, which) for p in ("fric_c", "fric_v", "fric_load")}


def _today() -> ServoParams:
    return ServoParams(
        name="today", friction=False, fric_c=_col("fric_c", "today"), fric_v=_col("fric_v", "today"),
        fric_load=_col("fric_load", "today"), fric_joint=(1.0,) * len(DRIVES), play_deg=(0.0,) * len(DRIVES),
        ts_corner=(None,) * 4, ts_zero=(None,) * 4, peak=(None,) * 4, armature=(None,) * 4,
        cmd_lag_ms=(0.0,) * 4, cmd_hold=False, hold_phase=0, kp_scale=(1.0,) * 4, kv_scale=(1.0,) * 4,
        gain_cap=False, enc_offset=(0.0,) * len(DRIVES), sense_steps=0, bus_v0=BUS_V, bus_r=0.0)


def _named() -> dict:
    n = nominal()
    leg = tuple(c in LEG_CLASSES for _, c in DRIVES)
    x6 = CLASSES.index("X6")
    out = dict(today=_today(), N=n)
    out["F-"] = replace(n, name="F-", **_friction_set("low"))
    out["F+"] = replace(n, name="F+", **_friction_set("high"))
    out["P0"] = replace(n, name="P0", play_deg=tuple(0.0 for _ in DRIVES))
    out["P+"] = replace(n, name="P+", play_deg=tuple(1.0 if g else 0.0 for g in leg))
    out["M-"] = replace(n, name="M-", mass=0.9)
    out["M+"] = replace(n, name="M+", mass=1.1)
    out["V-"] = replace(n, name="V-", bus_v0=21.0, bus_r=0.15)
    out["J+"] = replace(n, name="J+", armature=tuple(_number("armature", c, "high") for c in CLASSES))
    out["G5"] = replace(n, name="G5", gain_cap=True)
    out["T+"] = replace(n, name="T+", ts_corner=(None,) * 4, ts_zero=(None,) * 4)
    fc = list(n.fric_c)
    fc[x6] = X6L_FRIC_C
    pk = [None] * 4
    pk[x6] = X6L_PEAK
    out["X6L"] = replace(n, name="X6L", fric_c=tuple(fc), peak=tuple(pk))
    w1 = replace(out["F+"], bus_v0=21.0, bus_r=0.15, armature=out["J+"].armature, mass=1.1)
    out["W1"] = replace(w1, name="W1")
    out["W2"] = replace(out["F-"], name="W2", play_deg=out["P+"].play_deg, mass=0.9)
    return out


#: Section 4's named settings (each corner moves one factor and leaves the rest at N), and their use under P41.
NAMED_USE = {
    "today": "every part neutral: today's plant, for the identity test and a link to today's records",
    "N": "the Nominal column: gate 1's condition 1",
    "F-": "every class at its low friction set: condition 2, judged (stay up only, P41)",
    "F+": "every class at its high friction set, above the estimate's bound on the leg drives: condition 2, judged",
    "P0": "no play on any leg drive: condition 2, judged",
    "P+": "1.0 deg of play on every leg drive (none on the neck and tail): condition 2, judged",
    "M-": "trunk mass and inertia x 0.9, applied by the caller before the reset (not the layer): condition 2, judged",
    "M+": "trunk mass and inertia x 1.1, applied by the caller before the reset (not the layer): condition 2, judged",
    "V-": "bus_v0 21.0 V with bus_r 0.15 ohm: reported (it leaves the supply window at the walk's summed peak)",
    "J+": "armature at the top of each class's range, kv by the plant's rule (the tail's equals N): reported",
    "G5": "the gain cap on (kp <= 500, kv <= 5 on the leg drives): reported",
    "T+": "today's torque-speed corners and the X6's today's zero-torque speed: reported, the easy end",
    "X6L": "the hip pitch as if the X6-50-L ships: force range 31.5 N m and dry friction 0.2 N m; the judge's "
           "registered limits unchanged: reported",
    "W1": "F+, V-, J+ and M+ together: reported",
    "W2": "F-, P+ and M- together: reported",
    "D": "the drawn panel (panel_draw): condition 2, judged by the bar's whole rule; a passing record is run again "
         "on the second block (5042 to 5061) and the repeat must pass too (P41)",
}
NAMED = tuple(NAMED_USE)
#: P41: the six extremes and the drawn panel can stop the purchase; N is condition 1.
JUDGED = ("F-", "F+", "P0", "P+", "M-", "M+", "D")
REPORTED = ("V-", "J+", "G5", "T+", "X6L", "W1", "W2")
_SETTINGS = _named()


def setting(name: str, seed: int | None = None) -> ServoParams:
    """A named setting of section 4 (``NAMED``); ``D`` needs the bar seed of the run (``panel_draw``)."""
    if name == "D":
        if seed is None:
            raise ValueError("the drawn panel D draws from the run's bar seed: setting('D', seed)")
        return panel_draw(seed)
    if seed is not None:
        raise ValueError(f"the named setting {name!r} draws nothing: no seed")
    if name not in _SETTINGS:
        raise ValueError(f"setting {name!r}: choose from {', '.join(NAMED)}")
    return _SETTINGS[name]


def draw(seed: int, tag: int, name: str | None = None) -> ServoParams:
    """``DRAW_RULE``'s parameter set of ``seed`` under ``tag``: the same seed and tag give the same set."""
    if seed is None:
        raise ValueError("no seed: the servo draw needs one (with options['seeds'] and no episode_seed the env's "
                         "episode seed is None; give a named setting instead, design section 4)")
    seed, tag = int(seed), int(tag)
    rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))

    def uni(param, c):
        return float(rng.uniform(_number(param, c, "low"), _number(param, c, "high")))

    fric_c = tuple(uni("fric_c", c) for c in CLASSES)
    fric_v = tuple(uni("fric_v", c) for c in CLASSES)
    fric_load = tuple(uni("fric_load", c) for c in CLASSES)
    fric_joint = tuple(uni("fric_joint", c) for _, c in DRIVES)
    play = tuple(uni("play_deg", c) if c in LEG_CLASSES else 0.0 for _, c in DRIVES)
    corner = {c: uni("ts_corner", c) for c in LEG_CLASSES}
    zero_x6 = None
    if X6_ZERO_DRAWN:
        lo = max(_number("ts_zero", "X6", "low"), corner["X6"] + ZERO_GAP)
        zero_x6 = float(rng.uniform(lo, _number("ts_zero", "X6", "high")))
    arm = tuple(float(math.exp(rng.uniform(math.log(_number("armature", c, "low")),
                                           math.log(_number("armature", c, "high"))))) for c in CLASSES)
    lag = tuple(uni("cmd_lag_ms", c) for c in CLASSES)
    phase = int(rng.integers(int(cell("cmd_hold", "neck", "low")), int(cell("cmd_hold", "neck", "high")) + 1))
    kps = {c: uni("kp_scale", c) for c in BOX_CLASSES}
    kvs = {c: uni("kv_scale", c) for c in BOX_CLASSES}
    offs = tuple(uni("enc_offset", c) for _, c in DRIVES)
    sense = int(rng.integers(int(cell("sense_steps", "all", "low")), int(cell("sense_steps", "all", "high")) + 1))
    v0 = uni("bus_v0", "bus")
    r = uni("bus_r", "bus")
    return ServoParams(
        name=name or f"draw/{seed}/{tag}", friction=True, fric_c=fric_c, fric_v=fric_v, fric_load=fric_load,
        fric_joint=fric_joint, play_deg=play, ts_corner=(corner["X4"], corner["X6"], None, None),
        ts_zero=(None, zero_x6, None, None), peak=(None,) * 4, armature=arm, cmd_lag_ms=lag, cmd_hold=True,
        hold_phase=phase, kp_scale=tuple(kps.get(c, 1.0) for c in CLASSES),
        kv_scale=tuple(kvs.get(c, 1.0) for c in CLASSES), gain_cap=False, enc_offset=offs, sense_steps=sense,
        bus_v0=v0, bus_r=r, seed=seed, tag=tag)


def training_draw(episode_seed: int | None) -> ServoParams:
    """A training episode's draw: ``SeedSequence([episode seed, 2])`` (design section 4)."""
    return draw(episode_seed, TRAINING_TAG, None if episode_seed is None else f"train/{int(episode_seed)}")


def panel_draw(bar_seed: int) -> ServoParams:
    """The drawn panel ``D`` of the run with bar seed ``s``: ``SeedSequence([s, 3])``, then the rotor inertia and the
    bus resistance at nominal (design section 4: the maker's question 1 and N5; no source and its top leaves the
    supply window); the delay is the record's (not drawn), the trunk mass 1."""
    p = draw(bar_seed, PANEL_TAG, f"D/{int(bar_seed)}")
    n = nominal()
    return replace(p, armature=n.armature, bus_r=n.bus_r)


def _canonical(x):
    if isinstance(x, dict):
        return {str(k): _canonical(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canonical(v) for v in x]
    if isinstance(x, float):
        return repr(x)
    return x


def table_doc() -> dict:
    """What ``TABLE_SHA256`` hashes: the table, the constants, the draw rule and the named settings."""
    return dict(body=BODY, classes=list(CLASSES), parts=PARTS, drives=[list(d) for d in DRIVES], table=list(TABLE),
                constants=CONSTANTS, draw_rule=DRAW_RULE, x6_zero_drawn=X6_ZERO_DRAWN,
                tags=dict(training=TRAINING_TAG, panel=PANEL_TAG),
                settings={k: v.as_dict() for k, v in _SETTINGS.items()})


def table_sha256() -> str:
    return hashlib.sha256(json.dumps(_canonical(table_doc()), sort_keys=True).encode()).hexdigest()


TABLE_SHA256 = table_sha256()

# ------------------------------------------------------------------------------------------------
# The model's arithmetic (pure functions; the layer applies them per physics step)
# ------------------------------------------------------------------------------------------------
def bus_voltage(v0: float, r: float, p_w: float) -> tuple[float, bool]:
    """(V, collapsed): the higher root of V = V0 - R P / V, never above V0; with no root (4 R P > V0^2) the bus has
    collapsed and V is V0 / 2 for the step (design 2c)."""
    if r == 0.0:
        return float(v0), False
    disc = v0 * v0 - 4.0 * r * p_w
    if disc < 0.0:
        return 0.5 * v0, True
    return min(float(v0), 0.5 * (v0 + math.sqrt(disc))), False


def zero_speed(zero24, v_nominal, v):
    """The zero-torque speed on a bus of ``v`` volts: today's rule each step, ω0(V) = ω0(24 V) x min(V, V_nominal) /
    24 (``catalog.Actuator.speed_at``: derated below the nominal voltage, no credit above it; design 2c)."""
    return np.asarray(zero24, float) * (np.minimum(v, v_nominal) / BUS_V)


def line_limits(w, peak, corner24, zero24, zero_v) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) force range at the signed drive speeds ``w``: driving side clip(s (ω0(V) - |w|), 0, peak) with
    the slope s = peak / (ω0(24 V) - corner) fixed by the 24 V line (the line shifts with the voltage), braking side
    the peak (design 2c)."""
    w = np.asarray(w, float)
    s = np.asarray(peak, float) / (np.asarray(zero24, float) - np.asarray(corner24, float))
    lim = np.clip(s * (np.asarray(zero_v, float) - np.abs(w)), 0.0, peak)
    full = np.broadcast_to(np.asarray(peak, float), lim.shape)
    return np.where(w < 0, -lim, -full), np.where(w > 0, lim, full)


def friction_loss(mj, fric_c, fric_load, tau_bar):
    """The dry friction of each drive, F = m_j (T_c + μ_L τ̄) (design 2a)."""
    return np.asarray(mj, float) * (np.asarray(fric_c, float) + np.asarray(fric_load, float) * np.asarray(tau_bar))


def dead_zone(c_eff, length, w, kp, kv, b) -> tuple[np.ndarray, np.ndarray]:
    """(command sent, the clip): with u = kp (c - L) - kv w the torque the servo asks for and b half the play, the
    servo is sent c - clip(u / kp, -b, b), so MuJoCo's servo gives u - kp b above kp b, u + kp b below -kp b and 0 in
    between (design 2b).  The clip is also the motor-side reading's error."""
    c_eff = np.asarray(c_eff, float)
    x = np.clip((np.asarray(kp, float) * (c_eff - length) - np.asarray(kv, float) * w) / kp, -np.asarray(b),
                np.asarray(b))
    return c_eff - x, x


def lag_gain(lag_ms) -> np.ndarray:
    """1 - exp(-dt / T_r) per drive (1 for no lag): y <- y + gain (c - y) each physics step (design 2d)."""
    t = np.asarray(lag_ms, float) * 1e-3
    return np.where(t > 0, -np.expm1(-DT / np.where(t > 0, t, 1.0)), 1.0)


def kv_rule(kp, inertia_home, j, j_today):
    """The plant's own damping rule on a drawn inertia: kv = 2 ζ sqrt(kp (I_home + J - J_today)) (``plant._gains``;
    design 2f; ζ = ``carpet.SERVO_ZETA``)."""
    return 2.0 * carpet.SERVO_ZETA * np.sqrt(np.asarray(kp) * (np.asarray(inertia_home) + np.asarray(j)
                                                              - np.asarray(j_today)))


# ------------------------------------------------------------------------------------------------
# The run-time layer
# ------------------------------------------------------------------------------------------------
LAYER_RULE = (
    "Before every physics step, on each drive's own coordinate (the joint, or the ankle motor's tendon): (1) the bus: "
    "P = sum over the 12 leg drives of [max(tau w, 0) + c_cu tau^2] / 0.92 + the record's auxiliary input, from the "
    "last physics step's motor torques and speeds, and V the higher root of V = V0 - R P / V (V0 / 2 and a fault "
    "when there is none); (2) the force range: driving side clip(s (w0(V) - |w|), 0, peak), braking side the peak, "
    "with s = peak / (w0(24 V) - corner) and w0(V) = w0(24 V) x min(V, V_nominal) / 24 on the leg drives; the box "
    "servos keep today's lines; (3) friction: dry F = m_j (T_c + mu_L tau_bar), tau_bar the motor torque's magnitude "
    "through a 10 ms low-pass, written to dof_frictionloss on the joints and tendon_frictionloss on the ankle motors "
    "(the ankle joints' own damping and friction 0), viscous m_j b_v written once per run to dof_damping and "
    "tendon_damping; (4) the command c that reaches the drive (after the delay): the box servos take a new target "
    "every 5th physics step from the drawn phase (and at the first step), a first-order lag y <- y + (1 - exp(-dt / "
    "T_r)) (target - y) (none at T_r 0), the encoder's zero error o (the drive holds L + o = y, so it is sent y - o) "
    "and on the leg drives the gear play's dead zone (sent y - o - clip(u / kp, -b, b), u = kp (y - o - L) - kv w, b "
    "half the play), in d.ctrl for the step.  After it: the controller's own command back into d.ctrl, and the "
    "encoder readings patched: each drive's error o + clip mapped to the joints (at the ankle pitch reads q_p + (e_a "
    "+ e_b) / 2 and roll q_r + (e_a - e_b) / 2), the readings of sense_steps physics steps earlier, and the neck's "
    "and tail's refreshed only when their target is.  Per run: the reflected inertia (dof_armature, tendon_armature, "
    "then mj_setConst on a scratch MjData), kv by the plant's rule on it, the gain cap, the box servos' kp and kv "
    "scales.  The judge reads the nominal peak, corner and zero-torque speeds (the registered limits do not move); "
    "the controller's feedforward keeps today's friction (A18).  Not modelled: the rotor's crossing of the play, "
    "heat, flex, the drive's own feedback sampling (design section 7)")
HOOKS_RULE = (
    "The layer arms at entry: it replaces p.envelope (an object with the old one's nominal moment, peak, rated, "
    "no_load, bind, restore, limits, enabled, active and describe) and p.step (an instance attribute; refused when "
    "one is there already), and sets the plant's own gain copies p.kp and p.kv where the run's inertia or the gain cap "
    "changes them (the box servos' kp and kv scales leave them nominal on purpose, design 2d): every command folded "
    "after entry is folded with the gains it is served with.  It acts from the first physics step after entry or "
    "after a reset (the envelope's bind): then it writes the run's model gains, damping, friction and armature, so "
    "the reset's forward pass, its first reading and every tracker's copy of the friction are today's (traps 1 and "
    "9).  A reset inside the layer restores those arrays and arms again.  Exit restores, bit for bit, every model "
    "array the layer or mj_setConst writes, the envelope object, the step and the gain copies.  The trunk mass is "
    "not the layer's (the caller's, before the reset; a setting with a mass factor is refused while the trunk is as "
    "built).  The layer's states (the lag, the hold, the load filter, the readings' age) are in no snapshot: it is not "
    "offered to control.run(confirm=True) or to the ledger (trap 6)")
#: Every model array the layer writes, and every one mj_setConst overwrites (found on this plant by perturbing each
#: float array of a copy and seeing which ones mj_setConst writes back; tests/test_stage2_servo.py repeats it), with
#: the model's ``stat`` fields.
LAYER_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_forcelimited", "dof_damping",
                "dof_frictionloss", "dof_armature", "tendon_damping", "tendon_frictionloss", "tendon_armature")
SETCONST_FIELDS = ("actuator_acc0", "actuator_length0", "body_invweight0", "body_subtreemass", "dof_M0",
                   "dof_invweight0", "dof_length", "light_dir0", "light_pos0", "light_poscom0", "tendon_invweight0",
                   "tendon_length0", "cam_mat0", "cam_pos0", "cam_poscom0")
STAT_FIELDS = ("meaninertia", "meanmass", "meansize", "extent", "center")


def _span(idx):
    """A slice for consecutive indices (a view, the same values as the index array's copy), else the array."""
    idx = np.asarray(idx, int)
    if len(idx) and np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
        return slice(int(idx[0]), int(idx[0]) + len(idx))
    return idx


class ServoError(RuntimeError):
    """The plant is not one the layer can act on, or a hook it set was changed under it."""


class ServoEnvelope:
    """``p.envelope`` while the layer is in: the judge's reading of the drives (the nominal ``peak``, ``rated``,
    ``no_load`` and ``moment``) with the layer's force ranges.  ``update`` is the layer's own step's (refused from
    anywhere else: the class ``Plant.step`` would skip the layer's after-step); ``active`` is True whenever the
    layer's line is not today's (the judge's clamp test also asks that a range be below the peak)."""

    def __init__(self, layer: "Layer", old: drive.Envelope):
        self._layer, self._old = layer, old
        self.drives = old.drives
        self.moment = old.moment
        self.speed_scale, self.torque_scale, self.braking = old.speed_scale, old.torque_scale, old.braking
        self.peak, self.rated, self.no_load = old.peak.copy(), old.rated.copy(), old.no_load.copy()
        self._enabled = True
        self._m = self._d = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        if self._layer._today_env is not None:
            self._layer._today_env.enabled = self._enabled
        self.restore()

    @property
    def active(self) -> bool:
        if self._layer._today_env is not None:
            return self._layer._today_env.active
        return True

    def bind(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """A reset (``Plant.reset``, ``set_state``, ``settle``): the layer arms again (``HOOKS_RULE``)."""
        self._m, self._d = m, d
        self._layer._bind(m, d)
        m.actuator_forcelimited[:] = 1
        self.restore()

    def bound_to(self, m, d) -> bool:
        return self._m is m and self._d is d

    def restore(self) -> None:
        """Every force range back to the run's peak (a plant at rest)."""
        lay = self._layer
        if lay._today_env is not None:
            lay._today_env.restore()
        elif lay.m is not None:
            lay.m.actuator_forcerange[:, 1] = lay.peak_run
            lay.m.actuator_forcerange[:, 0] = -lay.peak_run
            lay._narrowed = False

    def limits(self, qvel) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) torque limits at the joint velocities ``qvel`` and the layer's last bus voltage."""
        lay = self._layer
        if not self._enabled:
            return -lay.peak_run.copy(), lay.peak_run.copy()
        if lay._today_env is not None:
            return lay._today_env.limits(qvel)
        return line_limits(self.moment @ np.asarray(qvel, float), lay.peak_run, lay.corner24, lay.zero24,
                           lay.zero_at(lay.V))

    def update(self) -> None:
        if not self._layer._in_step:
            raise ServoError("the servo layer's envelope was updated outside the layer's own physics step (its step "
                             "hook was removed or bypassed)")

    def describe(self) -> dict:
        return dict(self._old.describe(), servo_layer=dict(setting=self._layer.params.name,
                                                           table_sha256=TABLE_SHA256, rule=LAYER_RULE))


class Layer:
    """The servo layer of ``params`` on the Stage 1 plant ``p`` (``LAYER_RULE``, ``HOOKS_RULE``).  Use ``served``;
    ``enter`` and ``exit`` are its two halves.  ``readings`` holds the run's bus and play counts."""

    def __init__(self, p: plant_mod.Plant, params: ServoParams):
        if p.body != BODY:
            raise ServoError(f"{p.body}: the servo table is for {BODY} only (design, status line)")
        if tuple(d.actuator for d in p.drives) != ACTUATORS:
            raise ServoError(f"{p.body}: the plant's drives are not the table's {ACTUATORS}")
        for d, (a, c) in zip(p.drives, DRIVES, strict=True):
            if d.part != PARTS[c]:
                raise ServoError(f"{a}: part {d.part}, the table's class {c} is {PARTS[c]}")
        env = p.envelope
        if type(env) is not drive.Envelope:
            raise ServoError(f"{p.body}: the plant's envelope is a {type(env).__name__} (another layer or variant)")
        if (env.speed_scale, env.torque_scale, env.braking, env.enabled) != (1.0, 1.0, "peak", True):
            raise ServoError("the layer owns the torque-speed line: the plant's envelope must be its own (scales 1, "
                             "braking at peak, enabled)")
        if float(p.meta["options"]["bus_V"]) != BUS_V:
            raise ServoError(f"the plant's envelope is rated on {p.meta['options']['bus_V']} V, not the record's "
                             f"{BUS_V} V: the layer models the bus itself")
        if float(p.m.opt.timestep) != DT:
            raise ServoError(f"physics step {p.m.opt.timestep} s: the layer's hold and lag are for {DT} s")
        if params.mass != 1.0 and float(p.m.body_subtreemass[p.trunk]) == float(p.meta["mass_kg"]):
            raise ServoError(f"setting {params.name}: its trunk mass x {params.mass:g} is the caller's, applied before "
                             "the reset (env.set_trunk_mass), and the plant's trunk is as built")
        self.p, self.params = p, params
        m = p.m
        self.m = self.d = None
        n = len(DRIVES)
        cls = [c for _, c in DRIVES]
        ci = np.array([CLASSES.index(c) for c in cls])
        self.is_leg = np.array([c in LEG_CLASSES for c in cls])
        self.is_box = ~self.is_leg
        self.box_idx = np.flatnonzero(self.is_box)
        # the drives' coordinates: a joint's dof or an ankle motor's tendon
        trn = m.actuator_trntype
        self.on_tendon = trn == mujoco.mjtTrn.mjTRN_TENDON
        if not np.all(self.on_tendon | (trn == mujoco.mjtTrn.mjTRN_JOINT)):
            raise ServoError("every drive acts on a joint or on a fixed tendon")
        tid = m.actuator_trnid[:, 0]
        self.joint_idx = np.flatnonzero(~self.on_tendon)
        self.tendon_idx = np.flatnonzero(self.on_tendon)
        self.joint_dofs = np.array([m.jnt_dofadr[tid[i]] for i in self.joint_idx])
        self.tendons = tid[self.tendon_idx].copy()
        self.ankle_joint_dofs = np.array([p.dofs[p.joints.index(f"{s}_{a}")] for s in ("left", "right")
                                          for a in ("ankle_pitch", "ankle_roll")])
        # per drive values of the run
        pc = lambda f: np.array([getattr(params, f)[i] for i in ci], float)  # noqa: E731
        self.mj = np.array(params.fric_joint, float)
        self.damp = self.mj * pc("fric_v")
        self.mT = self.mj * pc("fric_c")
        self.mmu = self.mj * pc("fric_load")
        self.b = np.deg2rad(np.array(params.play_deg, float)) / 2.0
        self.offset = np.array(params.enc_offset, float)
        self.beta = lag_gain(pc("cmd_lag_ms"))
        self.nolag = self.beta == 1.0
        old = env
        pk = [params.peak[i] for i in ci]
        self.peak_run = np.array([old.peak[k] if v is None else v for k, v in enumerate(pk)], float)
        cr = [params.ts_corner[i] for i in ci]
        zr = [params.ts_zero[i] for i in ci]
        self.corner24 = np.array([old.rated[k] if v is None else v for k, v in enumerate(cr)], float)
        self.zero24 = np.array([old.no_load[k] if v is None else v for k, v in enumerate(zr)], float)
        self.v_nominal = np.array([V_NOMINAL[c] if c in LEG_CLASSES else np.inf for c in cls])
        self.slope = self.peak_run / (self.zero24 - self.corner24)
        self.V0, self.R = float(params.bus_v0), float(params.bus_r)
        self._zero_v0 = self.zero_at(self.V0)
        self._zl, self._zb = np.where(self.is_leg, self.zero24, 0.0), np.where(self.is_box, self.zero24, 0.0)
        self._v_fast = min(V_NOMINAL.values())
        self._neg_peak = -self.peak_run
        self._neg_b = -self.b
        self._moment = env.moment
        self._G = p.G
        self._box_sl = _span(self.box_idx)
        self.aux_w = float(plant_mod.record(p.body)["power"]["aux"]["input_w"])
        self.eta = float(power.ETA_DRIVE)
        ccu = CONSTANTS["copper_loss_W_per_Nm2"]["value"]
        self.c_cu = np.array([ccu[c] if c in LEG_CLASSES else 0.0 for c in cls])
        self.legmask = self.is_leg.astype(float)
        self.v_floor = float(CONSTANTS["supply_floor_V"]["value"])
        self.alpha = -math.expm1(-DT / LOAD_FILTER_S)
        # gains: the plant's copies follow the inertia and the cap; the model's also take the box servos' scales
        kp_m0, kv_m0 = m.actuator_gainprm[:, 0].copy(), -m.actuator_biasprm[:, 2].copy()
        kp_c0, kv_c0 = p.kp.copy(), p.kv.copy()
        j_found = np.array([m.tendon_armature[tid[i]] if self.on_tendon[i] else m.dof_armature[m.jnt_dofadr[tid[i]]]
                            for i in range(n)], float)
        j_drive = np.array([d.armature for d in p.drives], float)
        if not np.allclose(j_found, j_drive, rtol=1e-4, atol=0.0):  # (the MJCF writes it to 5 significant digits)
            raise ServoError("the plant's armature is not the one it was built with (another change is in)")
        arm = [params.armature[i] for i in ci]
        self.j_found = j_found
        self.j_run = np.array([j_found[k] if v is None else v for k, v in enumerate(arm)], float)
        self.arm_changed = self.j_run != j_found
        inertia = np.array([p.meta["gains"]["drives"][a]["inertia_kg_m2"] for a in ACTUATORS], float)
        kv_nom = np.where(self.arm_changed, kv_rule(kp_c0, inertia, self.j_run, j_found), kv_c0)
        capped = self.is_leg & bool(params.gain_cap)
        kp_c = np.where(capped, np.minimum(kp_c0, GAIN_CAP_KP), kp_c0)
        kv_c = np.where(capped, np.minimum(kv_nom, GAIN_CAP_KV), kv_nom)
        kp_scale, kv_scale = pc("kp_scale"), pc("kv_scale")
        kp_m = np.where(capped, np.minimum(kp_m0, GAIN_CAP_KP), kp_m0)
        kp_m = np.where(kp_scale != 1.0, kp_scale * kp_m, kp_m)
        kv_base = np.where(kv_c != kv_c0, kv_c, kv_m0)  # (the rule's kv where the inertia or the cap moved it)
        kv_m = np.where(kv_scale != 1.0, kv_scale * kv_base, kv_base)
        self.kp_copy0, self.kv_copy0 = kp_c0, kv_c0
        self.kp_copy, self.kv_copy = kp_c, kv_c
        self.kp_model0, self.kv_model0 = kp_m0, kv_m0
        self.kp_model, self.kv_model = kp_m, kv_m
        self.kp_set = np.flatnonzero(kp_m != kp_m0)
        self.kv_set = np.flatnonzero(kv_m != kv_m0)
        self.kv_over_kp = kv_m / kp_m
        # which parts act
        self._line_is_today = (self.R == 0.0 and self.V0 == BUS_V and all(v is None for v in params.ts_corner)
                               and all(v is None for v in params.ts_zero) and all(v is None for v in params.peak))
        self._sag = self.R != 0.0
        self._friction = bool(params.friction)
        self._friction_dynamic = self._friction and bool(np.any(self.mmu != 0.0))
        self._hold = bool(params.cmd_hold)
        self._phase = int(params.hold_phase)
        self._has_play = bool(np.any(self.b > 0.0))
        self._has_offset = bool(np.any(self.offset != 0.0))
        self._has_lag = not bool(np.all(self.nolag))
        self._any_nolag = bool(np.any(self.nolag))
        self._acts_on_command = self._hold or self._has_lag or self._has_play or self._has_offset
        self._errors = self._has_play or self._has_offset
        self.sense = int(params.sense_steps)
        self._patch = self._errors or self.sense > 0 or self._hold
        self._today_env = None
        self._entered = False
        self._in_step = False
        self._acting = False
        self.V = self.V0
        self._reset_readings()

    # ---------------------------------------------------------------- readings of the run
    def _reset_readings(self) -> None:
        self._steps = 0
        self._v_min = math.inf
        self._v_sum = 0.0
        self._below = 0
        self._collapse = 0
        self._inside = np.zeros(len(DRIVES), int)

    def readings(self) -> dict:
        """The run's bus and play counts: physics steps, the bus's lowest and mean voltage, the steps below the leg
        drives' 20 V supply floor and the collapsed ones (a run with any has left the supply window, design 2c;
        reported, not failed, the lead's ruling of 16 Nov 2026), and each leg drive's share of steps inside its
        play."""
        n = self._steps
        return dict(physics_steps=n, bus_v_min=None if n == 0 else float(self._v_min),
                    bus_v_mean=None if n == 0 else self._v_sum / n, supply_floor_v=self.v_floor,
                    steps_below_supply_floor=int(self._below), collapsed_steps=int(self._collapse),
                    left_supply_window=bool(self._below or self._collapse),
                    inside_play_share={a: (float(self._inside[i]) / n if n else 0.0)
                                       for i, a in enumerate(ACTUATORS) if self.is_leg[i]})

    def zero_at(self, v: float) -> np.ndarray:
        return np.where(self.is_leg, zero_speed(self.zero24, self.v_nominal, v), self.zero24)

    # ---------------------------------------------------------------- entry, reset, exit
    def enter(self) -> "Layer":
        p = self.p
        if self._entered:
            raise ServoError("this layer is already in")
        if "step" in p.__dict__:
            raise ServoError(f"{p.body}: the plant's step is already replaced (a digest or another layer)")
        m = p.m
        self._saved = {k: getattr(m, k).copy() for k in LAYER_FIELDS + SETCONST_FIELDS if hasattr(m, k)}
        self._saved_stat = {k: np.array(getattr(m.stat, k), copy=True) for k in STAT_FIELDS}
        self._old_env = p.envelope
        self._env = ServoEnvelope(self, self._old_env)
        if self._line_is_today:
            o = self._old_env
            self._today_env = drive.Envelope(o.drives, o.moment, enabled=True, speed_scale=o.speed_scale,
                                             torque_scale=o.torque_scale, braking=o.braking)
        self._scratch = None  # (mj_setConst's MjData, made at the first armature write)
        self._step_fn = self._step
        p.envelope = self._env
        p.step = self._step_fn
        self._entered = True
        try:
            self._env.bind(m, p.d)  # (arms: nothing of the run is written before the first physics step)
            p.kp[:] = self.kp_copy
            p.kv[:] = self.kv_copy
        except BaseException:
            self.exit()
            raise
        return self

    def _bind(self, m, d) -> None:
        if m is not self.p.m:
            raise ServoError("the layer's envelope was bound to another model")
        self.m, self.d = m, d
        if self._acting:
            self._restore_model()
        self._acting = False
        self._views(d)
        if self._today_env is not None:
            self._today_env.bind(m, d)
        self._reset_readings()

    def _views(self, d) -> None:
        p, nd = self.p, len(DRIVES)
        self._qvel, self._f, self._ctrl, self._sd = d.qvel, d.actuator_force, d.ctrl, d.sensordata
        qa = _span(p.qadr)
        self._qh = d.qpos[qa] if isinstance(qa, slice) else None
        sens = p._sens
        self._q_adr = np.array([sens[f"enc_q_{jn}"][0] for jn in p.joints])
        self._qd_adr = np.array([sens[f"enc_qd_{jn}"][0] for jn in p.joints])
        self._q_sl, self._qd_sl = _span(self._q_adr), _span(self._qd_adr)
        # the box servos' joints are their drives' (index for index: only the ankle's coordinates differ)
        self._box_q, self._box_qd = self._q_adr[self.box_idx], self._qd_adr[self.box_idx]
        fr = self.m.actuator_forcerange  # (views kept: each MjModel attribute access makes a new array object)
        self._fr_hi, self._fr_lo = fr[:, 1], fr[:, 0]
        self._dof_fl, self._ten_fl = self.m.dof_frictionloss, self.m.tendon_frictionloss
        self._w, self._w_prev, self._L, self._z = np.zeros(nd), np.zeros(nd), np.zeros(nd), np.zeros(nd)
        self._b1, self._b2, self._b3, self._b4 = np.zeros(nd), np.zeros(nd), np.zeros(nd), np.zeros(nd)
        self._m1, self._m2 = np.zeros(nd, bool), np.zeros(nd, bool)
        self._fj, self._ft = np.zeros(len(self.joint_idx)), np.zeros(len(self.tendon_idx))
        self._tbar = np.zeros(nd)
        self._c, self._held, self._tgt, self._y = np.zeros(nd), np.zeros(nd), np.zeros(nd), np.zeros(nd)
        self._sent, self._e, self._eq = np.zeros(nd), np.zeros(nd), np.zeros(nd)
        self._ring = np.zeros((self.sense + 1, 2 * nd))
        self._ring_i = 0
        self._held_r = np.zeros(2 * len(self.box_idx))
        self._Ginv = np.linalg.inv(p.G)
        self._narrowed = True

    def _restore_model(self) -> None:
        m = self.m if self.m is not None else self.p.m
        for k, v in self._saved.items():
            getattr(m, k)[...] = v
        for k, v in self._saved_stat.items():
            setattr(m.stat, k, v.copy() if v.ndim else v.item())

    def exit(self) -> None:
        """Restore the plant exactly (``HOOKS_RULE``); raises after restoring if the step hook was changed under it."""
        if not self._entered:
            return
        p = self.p
        step_ok = p.__dict__.get("step") is self._step_fn
        p.__dict__.pop("step", None)
        p.envelope = self._old_env
        if not self._old_env.bound_to(p.m, p.d):
            self._old_env.bind(p.m, p.d)
        self._restore_model()
        p.kp[:] = self.kp_copy0
        p.kv[:] = self.kv_copy0
        self._entered = self._acting = False
        self._today_env = None
        if not step_ok:
            raise ServoError(f"{p.body}: the plant's step was replaced inside the servo layer; restored on exit")

    # ---------------------------------------------------------------- the physics step
    def _start(self) -> None:
        """The first physics step after arming: the run's model arrays, then the states from the plant as it is."""
        m = self.m
        if len(self.kp_set):
            m.actuator_gainprm[self.kp_set, 0] = self.kp_model[self.kp_set]
            m.actuator_biasprm[self.kp_set, 1] = -self.kp_model[self.kp_set]
        if len(self.kv_set):
            m.actuator_biasprm[self.kv_set, 2] = -self.kv_model[self.kv_set]
        if self._friction:
            m.dof_damping[self.ankle_joint_dofs] = 0.0
            m.dof_frictionloss[self.ankle_joint_dofs] = 0.0
            m.dof_damping[self.joint_dofs] = self.damp[self.joint_idx]
            m.tendon_damping[self.tendons] = self.damp[self.tendon_idx]
        if np.any(self.arm_changed):
            m.dof_armature[self.joint_dofs] = self.j_run[self.joint_idx]
            m.tendon_armature[self.tendons] = self.j_run[self.tendon_idx]
            if self._scratch is None:
                self._scratch = mujoco.MjData(m)
            mujoco.mj_setConst(m, self._scratch)
        np.abs(self._f, out=self._tbar)
        np.dot(self._moment, self._qvel, out=self._w_prev)
        if self._friction:
            self._write_friction()
        sd, nd = self._sd, len(DRIVES)
        self._ring[:, :nd] = sd[self._q_sl]  # the reset's own reading: exact (the readings' age starts from it)
        self._ring[:, nd:] = sd[self._qd_sl]
        self._ring_i = 0
        self._k = 0
        self._acting = True

    def _write_friction(self) -> None:
        F = np.multiply(self.mmu, self._tbar, out=self._b3)
        F += self.mT
        self._dof_fl[self.joint_dofs] = np.take(F, self.joint_idx, out=self._fj)
        self._ten_fl[self.tendons] = np.take(F, self.tendon_idx, out=self._ft)

    def _line(self, w, V) -> None:
        """The force ranges at the drive speeds ``w`` and the bus voltage ``V`` (``line_limits``' arithmetic: the
        zero-torque speed as ``zero_speed`` gives it, element for element); rewritten only while some drive is past its
        corner (and once more on the way back), as the envelope's."""
        if not self._sag:
            z = self._zero_v0
        elif V <= self._v_fast:  # every leg drive below its nominal voltage: w0(V) = w0(24 V) x V / 24
            z = np.multiply(self._zl, V / BUS_V, out=self._z)
            z += self._zb
        else:
            z = self.zero_at(V)
        a = np.abs(w, out=self._b1)
        np.subtract(z, a, out=a)
        a *= self.slope
        peak = self.peak_run
        np.minimum(a, peak, out=a)
        narrowed = bool(np.less(a, peak, out=self._m1).any())
        if not (narrowed or self._narrowed):
            return
        np.maximum(a, 0.0, out=a)
        hi, lo = self._fr_hi, self._fr_lo
        hi[:] = peak
        np.copyto(hi, a, where=np.greater(w, 0.0, out=self._m1))
        lo[:] = self._neg_peak
        np.copyto(lo, np.negative(a, out=a), where=np.less(w, 0.0, out=self._m2))
        self._narrowed = narrowed

    def _before(self) -> None:
        first = not self._acting
        if first:
            self._start()
        k = self._k
        w = np.dot(self._moment, self._qvel, out=self._w)
        V = self.V0
        if self._sag:
            f = self._f
            tw = np.multiply(f, self._w_prev, out=self._b1)
            np.maximum(tw, 0.0, out=tw)
            P = (float(np.dot(tw, self.legmask)) + float(np.dot(np.multiply(f, f, out=self._b2), self.c_cu))) \
                / self.eta + self.aux_w
            V, collapsed = bus_voltage(self.V0, self.R, P)
            self._collapse += collapsed
        self.V = V
        if V < self._v_min:
            self._v_min = V
        self._v_sum += V
        if V < self.v_floor:
            self._below += 1
        if self._today_env is not None:
            self._today_env.update()
        elif self._env._enabled:
            self._line(w, V)
        if self._friction_dynamic and not first:
            a = np.abs(self._f, out=self._b2)
            a -= self._tbar
            a *= self.alpha
            self._tbar += a
            self._write_friction()
        ctrl, c = self._ctrl, self._c
        c[:] = ctrl
        self._refresh = refresh = self._hold and (k == 0 or (k - self._phase) % HOLD_STEPS == 0)
        if self._acts_on_command:
            if self._hold:
                tgt, bs = self._tgt, self._box_sl
                tgt[:] = c
                if refresh:
                    self._held[bs] = c[bs]
                else:
                    tgt[bs] = self._held[bs]
            else:
                tgt = c
            if self._has_lag:
                y = self._y
                if first:
                    y[:] = tgt
                else:
                    dy = np.subtract(tgt, y, out=self._b2)
                    dy *= self.beta
                    y += dy
                    if self._any_nolag:
                        np.copyto(y, tgt, where=self.nolag)
            else:
                y = tgt
            c_eff = self._sent
            if self._has_offset:
                np.subtract(y, self.offset, out=c_eff)
            else:
                c_eff[:] = y
            if self._has_play:
                L = np.dot(self._G, self._qh if self._qh is not None else self.d.qpos[self.p.qadr], out=self._L)
                x = np.subtract(c_eff, L, out=self._b3)
                x -= np.multiply(self.kv_over_kp, w, out=self._b4)
                np.less(np.abs(x, out=self._b4), self.b, out=self._m1)
                self._inside += self._m1
                np.minimum(x, self.b, out=x)
                np.maximum(x, self._neg_b, out=x)
                c_eff -= x
                if self._has_offset:
                    np.add(self.offset, x, out=self._e)
                else:
                    self._e[:] = x
            elif self._has_offset:
                self._e[:] = self.offset
            ctrl[:] = c_eff
        np.copyto(self._w_prev, w)

    def _after(self) -> None:
        if self._patch:
            sd, nd = self._sd, len(DRIVES)
            r = self._ring[self._ring_i]
            r[:nd] = sd[self._q_sl]
            r[nd:] = sd[self._qd_sl]
            if self._errors:
                r[:nd] += np.dot(self._Ginv, self._e, out=self._eq)
            if self.sense:
                self._ring_i = (self._ring_i + 1) % (self.sense + 1)
                r = self._ring[self._ring_i]
            sd[self._q_sl] = r[:nd]
            sd[self._qd_sl] = r[nd:]
            if self._hold:  # the neck's and tail's readings refresh with their targets
                nb, hr = len(self.box_idx), self._held_r
                if self._refresh:
                    hr[:nb] = sd[self._box_q]
                    hr[nb:] = sd[self._box_qd]
                else:
                    sd[self._box_q] = hr[:nb]
                    sd[self._box_qd] = hr[nb:]
        self._steps += 1
        self._k += 1

    def _step(self, n: int = 1) -> None:
        """``n`` physics steps under the layer (``Plant.step``'s place while the layer is in)."""
        m, d = self.m, self.d
        for _ in range(n):
            self._in_step = True
            try:
                self._before()
                try:
                    mujoco.mj_step(m, d)
                finally:
                    self._ctrl[:] = self._c
            finally:
                self._in_step = False
            self._after()

    # ---------------------------------------------------------------- description
    def describe(self) -> dict:
        """The setting, its values with each parameter's source and kind (``describe_rows``), the table's sha256,
        the rules and the constants; the run's gains as the layer sets them."""
        return dict(body=BODY, setting=self.params.name, table_sha256=TABLE_SHA256, kinds=list(KINDS),
                    kind_counts=kind_counts(), params=self.params.as_dict(), rows=describe_rows(self.params),
                    constants=CONSTANTS, rules=dict(layer=LAYER_RULE, hooks=HOOKS_RULE, draw=DRAW_RULE),
                    gains={a: dict(kp_model=float(self.kp_model[i]), kv_model=float(self.kv_model[i]),
                                   kp_copy=float(self.kp_copy[i]), kv_copy=float(self.kv_copy[i]),
                                   armature=float(self.j_run[i]))
                           for i, a in enumerate(ACTUATORS)},
                    simulation_only="a prospective design built in code; no value is measured on this build's "
                                    "hardware, and nothing the layer runs is a result")


@contextmanager
def served(p: plant_mod.Plant, params: ServoParams):
    """The servo layer of ``params`` on ``p`` for the block (``LAYER_RULE``); restores ``p`` exactly when the block
    ends, whatever happens in it (``HOOKS_RULE``).  Yields the ``Layer``."""
    lay = Layer(p, params).enter()
    try:
        yield lay
    finally:
        lay.exit()


# ------------------------------------------------------------------------------------------------
# Description
# ------------------------------------------------------------------------------------------------
def _value_of(params: ServoParams, param: str, cls: str):
    if param == "fric_joint":
        return dict(zip(ACTUATORS, params.fric_joint, strict=True))
    if param == "enc_offset":
        return dict(zip(ACTUATORS, params.enc_offset, strict=True))
    if param == "play_deg":
        return {a: v for (a, c), v in zip(DRIVES, params.play_deg, strict=True) if c == cls}
    if param == "cmd_hold":
        return f"every {HOLD_STEPS}th step, phase {params.hold_phase}" if params.cmd_hold else "every step"
    if param == "gain_cap":
        return "kp <= 500 N m/rad, kv <= 5 N m s/rad" if params.gain_cap else "none"
    if param == "sense_steps":
        return params.sense_steps
    if param in ("bus_v0", "bus_r"):
        return getattr(params, param)
    if param == "latency_steps":
        return "the env's draw or the record's delay (not the layer's)"
    v = getattr(params, param)[CLASSES.index(cls)]
    if v is None:
        return f"today ({_number(param, cls, 'today') if param in ('ts_corner', 'ts_zero', 'armature') else '-'})"
    if param in ("fric_c", "fric_v", "fric_load") and not params.friction:
        return f"today's arrays as found (A18: 0.15 N m and 0.2 N m s/rad on every hinge; {v})"
    return v


def describe_rows(params: ServoParams) -> list[dict]:
    """Each table row with this setting's value: the value, the nominal, the range, the distribution, the source,
    its kind (``KINDS``) and what would replace it."""
    out = []
    for r in TABLE:
        for i, c in enumerate(r["classes"]):
            out.append(dict(param=r["param"], cls=c, unit=r["unit"], value=_value_of(params, r["param"], c),
                            today=r["today"][i], nominal=r["nominal"][i], low=r["low"][i], high=r["high"][i],
                            distribution=r["distribution"], kind=r["kind"],
                            unsourced=r["kind"] == UNSOURCED, source=r["source"], replaced_by=r["replaced_by"]))
    return out


def _short(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, dict):
        vals = list(v.values())
        if vals and all(isinstance(x, float) for x in vals):
            return f"{len(vals)} values {min(vals):.4g} to {max(vals):.4g}"
        return "; ".join(f"{k} {_short(x)}" for k, x in v.items())
    return str(v)


def describe_lines(params: ServoParams) -> list[str]:
    """The description the layer prints: the setting, then every parameter with its value, source and kind."""
    head = [f"servo model, setting {params.name} (body {BODY}; table sha256 {TABLE_SHA256[:16]}...): simulation "
            "only; no value is measured on this build's hardware",
            "kinds of the table's 39 rows: " + ", ".join(f"{k} {n}" for k, n in kind_counts().items())]
    if params.mass != 1.0:
        head.append(f"trunk mass x {params.mass:g}: the caller's, applied before the reset (not the layer's)")
    rows = []
    for r in describe_rows(params):
        rng = "fixed" if r["low"] is None else f"{_short(r['low'])} to {_short(r['high'])}"
        tag = " UNSOURCED" if r["unsourced"] else ""
        rows.append(f"{r['param']} [{r['cls']}, {r['unit']}] = {_short(r['value'])} (nominal {_short(r['nominal'])}; "
                    f"range {rng}, {r['distribution']}) | {r['kind']}{tag}: {r['source']} | replaced by: "
                    f"{r['replaced_by']}")
    return head + rows
