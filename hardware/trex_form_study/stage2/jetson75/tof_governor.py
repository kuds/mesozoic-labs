"""A speed governor from the chest ToF stopping horizon (``JETSON_PLEM_REVIEW.md`` section 5, task 10).

Simulation and planning only: nothing here is a hardware measurement, and nothing here is a speed the robot has shown.
The two chest time-of-flight (ToF) ranging sensors look down and forward over each foot path (``DECISION_LOG.md`` F4,
P22).  Autonomous walking must be able to stop for a drop or an obstacle that only they see.  That sets a top speed for
autonomous walking, a governor, per stop path and per ToF mode.

Usage (from ``hardware/trex_form_study`` with ``PYTHONPATH=.``)::

    python stage2/jetson75/tof_governor.py        # computes and writes stage2/jetson75/tof_governor.json
    python stage2/jetson75/tof_governor.py md     # prints the tables of TOF_GOVERNOR.md from that record

Kinds of evidence, as in the study's reviews: (a) a prescribed-walk or static model result of this study, (b) a
closed-loop simulation result (none is used here), (c) a sourced fact (a catalog id, a document of this study or a
vendor document), (d) an estimate or assumption (whose, and how derived).  Every input in ``inputs()`` carries its kind
and its source; a (d) input carries a range, so the governor is a range of speeds.

Method
------
For a body, a ToF mode, a stop path and a stop reading, the governor speed v is the largest speed with

    d_see - v * T_react >= d_stop(v)

* ``d_see``: the farthest floor point ahead of the toe tips that the sensors see on the foot path.  It comes from the
  body's committed sensor screen (``designlab/sensors.py``): the zone-centre rays of the far row on a flat floor,
  inside the foot path (the feet's outer edges plus 0.05 m).  The record holds the 8x8 zones at the home pose; this
  module recomputes them with the screen's own lens model (``sensors.Lens``), checks that it reproduces the 64
  recorded zones, and then lowers the trunk by the reference walk's crouch (0.03 H, ``designlab/gait.py``), because
  the walk never stands at the home height.  The 4x4 zones come from the same computation.  The range runs from the
  nearest far-row zone centre on the foot path (every far-row zone there has reached the hazard) to the farthest
  (the first has).
* ``T_react``: the sum of the path's timing terms (``PATHS``): the ToF frames needed to detect, the I2C read, the stop
  logic's period, the walking loop's period and the command delay to the drives; the host path adds the planner and
  its link.  Each term has a low and a high value.
* ``d_stop(v)``: how far the front of the feet travels in the stop, measured from the home-stance toe line at the
  instant the stop begins.  Six readings, all from what the study has (``READINGS``).  Two are decelerations held from
  the first instant, v^2 / (2 a): the behaviour spec's limit (1.5 m/s^2) and the body's Stage 0 one-foot stopping
  hold.  Four are step counts of the study's reference walk at speed v: one step and two steps (the agility bar's
  "a step or two"), today's stop builder at its worst phase (2.4 steps, from the Stage 1 record) and the behaviour
  spec's own stop limit (1.5 strides, three steps).  Two floors apply (``stop_parts``): a step reading is never
  shorter than v^2 / (2 x 1.5), since the spec's deceleration limit binds every stop (the bar gates both), and no
  reading is shorter than the lead of a foot already on the ground, duty / 2 x stride(v) (a stance foot lands that
  far ahead of the toe line, ``gait._foot_x``).  The stride and duty at speed v are the reference walks' own: linear
  in speed between the nominal, brisk and Froude 0.05 walks (``designlab.stage2.look.speed_spec``); outside them the
  nearest walk's period and duty are held, which is an extrapolation and is flagged.

The low end of each range takes the nearest ``d_see`` with every timing term high; the high end takes the farthest
``d_see`` with every term low.  The sensitivity (``_sensitivity``) moves one term at a time about the reference
values and prices other readings of the same case, each at the midpoint of its own sight range, as the reference is;
the stacked case (``_stress``) takes every adverse reading at once.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from designlab import envelope, sensors, sizing  # noqa: E402
from designlab.stage2 import look  # noqa: E402

OUT = HERE / "tof_governor.json"
WRITTEN = "2026-11-14"

#: body -> its round-two refine case (the committed records) and its Stage 1 run (the stop reference's record).
BODIES = {
    "75-M-H-10": dict(case="feet_75", stage1_run="stage1/runs/2026-11-08/75-M-H-10"),
    "60-N-T-10": dict(case="narrow_carpet_60", stage1_run="stage1/runs/2026-11-08/60-N-T-10"),
}

KINDS = {
    "a": "a prescribed-walk or static model result of this study",
    "b": "a closed-loop simulation result",
    "c": "a sourced fact (a catalog id, a document of this study or a vendor document)",
    "d": "an estimate or assumption (whose, and how derived)",
}

# ------------------------------------------------------------------------------------------------
# Sourced facts and assumptions
# ------------------------------------------------------------------------------------------------
TOF_PART = "st_vl53l5cx_carrier"
#: The two catalog entries this document reads (the part selected and the newer part its range note sits on).
TOF_CATALOG_IDS = ("st_vl53l5cx_carrier", "pololu_vl53l8cx")
#: The mount (designlab/build.py, "Two ToF sensors under the chest"): 45 deg down, each 12 deg outward.
TOF_YAW_DEG = 12.0
#: Ranging modes: zones per side and the ranging frequency.  ST UM2884, table 2: 1 to 15 Hz at 8x8, 1 to 60 Hz at
#: 4x4.  The vendor states range per resolution and rate (datasheet tables 17 and 18); those values were not read.
MODES = {
    "8x8_15hz": dict(label="8x8 zones at 15 Hz", zones=8, rate_hz=15.0,
                     rate_basis="the fastest rate ST allows at 8x8 (UM2884 table 2)",
                     vendor_range="not read (datasheet tables 17, 18)"),
    "4x4_60hz": dict(label="4x4 zones at 60 Hz", zones=4, rate_hz=60.0,
                     rate_basis="the fastest rate ST allows at 4x4 (UM2884 table 2)",
                     vendor_range="none found for this rate"),
    "4x4_30hz": dict(label="4x4 zones at 30 Hz", zones=4, rate_hz=30.0,
                     rate_basis="inside ST's 1 to 60 Hz at 4x4 (UM2884 table 2); the rate the datasheet's 4x4 range "
                                "table is given at, as the review's search extract has it (d)",
                     vendor_range="not read (datasheet tables 17, 18)"),
}
#: I2C bus rates the sensor accepts (vendor datasheet) and bits on the wire per byte (8 data bits and the acknowledge).
I2C_HZ = (400e3, 1e6)
I2C_BITS_PER_BYTE = 9
#: Result packet per frame, bytes: (smallest, largest) in ST's I2C bandwidth table (UM2887).  The largest of each
#: resolution was read with its row: every output, four targets per zone.  The smallest, 124 bytes, was read as 4x4
#: with one target per zone and distance, status and target count only; 316 bytes is taken to be the same outputs at
#: 8x8, which the extract did not show.  That the stop needs only those outputs is this document's assumption, so the
#: sizes used for the timing are kind (d).
PACKET_BYTES = {4: (124, 1008), 8: (316, 3360)}
#: ToF frames from the hazard reaching d_see to a frame that reports it: 1 = one frame, best phase; 2 = one frame,
#: worst phase (the hazard arrives just after a frame began); 3 = a two-frame confirmation at the worst phase.
FRAMES = (1.0, 2.0, 3.0)
#: Loop rates fixed in COMPUTE.md ("What the onboard computer has to do"), Hz.
MCU_LOOP_HZ = (200.0, 1000.0)
WALK_LOOP_HZ = (50.0, 100.0)
PLANNER_HZ = (10.0, 15.0)
#: BEHAVIOR_SPEC.md section 3: at most 200 ms from the oldest sensor sample to a motion request (a deadline).  The
#: low end is 0: a bound, since nothing sourced says how little a planner adds.
HOST_PLANNER_S = (0.0, 0.200)
HOST_LINK_S = (0.0, 0.005)
#: The command delay to the drives: the agility bar is judged at 10 ms until a bench value exists (P29, P30); Stage 1
#: simulated 0-20 ms.
SERVO_DELAY_S = (0.0, 0.010, 0.020)
#: The stop readings.
SPEC_DECEL = envelope.STOP_DECEL  # 1.5 m/s^2 (BEHAVIOR_SPEC.md section 1; STAGE1_PLAN.md decision 10)
SPEC_STOP_STRIDES = 1.5  # BEHAVIOR_SPEC.md section 1: stop within 1.5 strides (STAGE1_PLAN.md decision 18)
DECEL_READINGS = ("spec_decel", "one_foot_hold")
#: Step counts; ``stop_builder_worst`` is read from each body's Stage 1 stop record (``as_built_stop``).
STEP_READINGS = {"one_step": 1.0, "two_steps": 2.0, "spec_strides": 2.0 * SPEC_STOP_STRIDES}
READINGS = ("spec_decel", "one_foot_hold", "one_step", "two_steps", "stop_builder_worst", "spec_strides")
READING_LABELS = {
    "spec_decel": "spec limit, 1.5 m/s^2",
    "one_foot_hold": "one-foot hold",
    "one_step": "one step",
    "two_steps": "two steps",
    "stop_builder_worst": "today's stop builder, worst phase (2.4 steps)",
    "spec_strides": "spec stop limit, 1.5 strides",
}
#: The kind of each stop reading (not of the governor speed, which is (a) with a range from (d) timing).
READING_KINDS = {
    "spec_decel": "c, a target",
    "one_foot_hold": "a",
    "one_step": "count c, a target; step a",
    "two_steps": "count c, a target; step a",
    "stop_builder_worst": "a",
    "spec_strides": "count c, a target; step a",
}
#: The agility bar's walk and the speed ladder (STAGE2_PLAN.md section 5), m/s.
BAR_SPEED = look.AGILITY_SPEED
LADDER = (0.30, 0.35, 0.40, 0.45, 0.50)
#: Trunk pitch limits while walking (BEHAVIOR_SPEC.md section 1), degrees; the walking crouch of the reference walks.
PITCH_RMS_DEG, PITCH_PEAK_DEG = 4.0, 8.0
CROUCH_FRAC = 0.03
STANDOFF_M = 0.10
OBSTACLE_M = 0.10  # BEHAVIOR_SPEC.md section 3: static obstacles at least 0.10 m tall

PATHS = {
    "mcu": dict(
        label="1. Microcontroller reads the ToF pair and stops",
        short="1. Microcontroller",
        terms=("tof_frames", "i2c_read", "mcu_loop", "walk_loop", "servo_delay"),
        design="COMPUTE.md's design and the owner's baseline (P30): the ToF stop runs in the microcontroller's loop",
        if_linux_fails="by design the ToF stop keeps working, since the real-time layer does not need Linux; that "
                       "firmware is not written (COMPUTE.md)"),
    "rt_core": dict(
        label="2. Isolated real-time core on the Jetson reads the ToF pair and stops",
        short="2. Real-time core",
        terms=("tof_frames", "i2c_read", "rt_core_loop", "walk_loop", "servo_delay"),
        design="the alternative the delay bench test measures (P30); PR #668 (unverified) calls it Partial PLEM",
        if_linux_fails="no ToF stop: the walking loop itself runs on Linux, and a Linux failure ends in torque-off "
                       "(JETSON_PLEM_REVIEW.md section 3, Safety)"),
    "host_planner": dict(
        label="3. Host planner decides the stop through the camera pipeline",
        short="3. Host planner",
        terms=("tof_frames", "i2c_read", "host_planner", "host_link", "mcu_loop", "walk_loop", "servo_delay"),
        design="PR #668's (unverified) routing; COMPUTE.md does not route the ToF stop this way",
        if_linux_fails="no ToF stop from the planner; by design the microcontroller's own stops remain "
                       "(velocity-request expiry, watchdogs; that firmware is not written: COMPUTE.md)"),
}

TERM_LABELS = {
    "tof_frames": "ToF frames to detect",
    "i2c_read": "I2C read of the frame",
    "mcu_loop": "microcontroller loop period",
    "rt_core_loop": "real-time core loop period",
    "walk_loop": "walking loop period",
    "host_planner": "host planner, sample to motion request",
    "host_link": "host to real-time layer link",
    "servo_delay": "command delay to the drives",
}

#: The kind of each timing term's value, as the tables show it (the sensor period is (c); the frames needed are (d)).
TERM_KINDS = {"tof_frames": "c period x d frames", "i2c_read": "d", "mcu_loop": "c", "rt_core_loop": "d",
              "walk_loop": "c paths 1 and 3; d path 2, taken equal", "host_planner": "d: low a bound, high the c "
              "deadline assumed met", "host_link": "d", "servo_delay": "d"}


def term_kind(term: str, path: str) -> str:
    """The kind of a timing term on one stop path.  COMPUTE.md fixes the walking loop's rate on the microcontroller
    only; on the Jetson's real-time core the same rate is this document's assumption."""
    if term == "walk_loop":
        return "d" if path == "rt_core" else "c"
    return TERM_KINDS[term]


_READ_HOW = "through the web search tool's extract on 14 Nov 2026"

WEB_SOURCES = [
    dict(id="st_um2884",
         url="https://st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-"
             "ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf",
         mirror="https://core-electronics.com.au/attachments/localcontent/um2884-a-guide_75701dc7b2e.pdf",
         says="ST user manual UM2884 for the VL53L5CX driver. Table 2, minimum and maximum ranging frequencies: 4x4 "
              "1 to 60 Hz, 8x8 1 to 15 Hz; the default is 1 Hz; the resolution is set before the frequency. Continuous "
              "mode keeps the emitter on, so range and ambient immunity are better; autonomous mode is the default and "
              "takes an integration time. Changing the resolution also changes the I2C traffic when results are read.",
         read_how=_READ_HOW + "; the vendor's site could not be opened directly from this session"),
    dict(id="st_vl53l5cx_datasheet",
         url="https://static6.arrow.com/aropdfconversion/65108bc7ab8c2686fe29d25aca394c39e7548213/vl53l5cx.pdf",
         mirror="https://cdn.sparkfun.com/assets/6/e/3/0/6/vl53l5cx-datasheet.pdf",
         says="ST VL53L5CX datasheet, DS13754 (distributors' copies). I2C at 400 kHz to 1 MHz, address 0x52; up to "
              "400 cm ranging; the integration time applies in autonomous mode only; the detection volume depends on "
              "reflectance, ambient light, resolution, sharpener, ranging mode and integration time. Field of view: "
              "revision 2 says 63 deg diagonal; revision 13 lists 45 deg horizontal, 45 deg vertical and 65 deg "
              "diagonal. The field of view is measured with a white 88 % target square to the sensor at 1 m, in the "
              "dark, at 8x8.",
         read_how=_READ_HOW + "; the maximum-range tables (17 and 18) were not in any extract, and the file hosts "
                  "could not be opened directly"),
    dict(id="st_an5894",
         url="https://www.st.com/resource/en/application_note/an5894-description-of-the-fields-of-view-of-"
             "stmicroelectronics-timeofflight-sensors-stmicroelectronics.pdf",
         mirror=None,
         says="ST application note AN5894 on the fields of view of its ToF sensors: for the VL53L5CX the field is "
              "given on the diagonal (65 deg) and by side (45 x 45 deg), and the side cannot be converted to the "
              "diagonal with Pythagoras' theorem.",
         read_how=_READ_HOW),
    dict(id="st_range_tables_thread",
         url="https://community.st.com/imaging-sensors-49/vl53l5cx-datasheet-table-17-18-definition-of-typical-and-"
             "minimum-values-4794",
         mirror=None,
         says="An ST community question on how to read 'typical' and 'minimum' in the VL53L5CX datasheet's tables 17 "
              "and 18 of maximum range capability (table 18 is on page 20). It shows that the tables exist; it gives "
              "no value, and the extract held no answer.",
         read_how=_READ_HOW + "; used only to name the tables that were not read"),
    dict(id="st_tof_presentation_2023",
         url="https://st.com/resource/en/application_presentation/oct24th-2023-latest-time-of-flight-sensor-"
             "solutions-and-applications.pdf",
         mirror=None,
         says="An ST presentation of October 2023 lists the VL53L5CX at 400 cm in the dark and 170 cm in ambient "
              "light, with no target, resolution or rate beside the figures.",
         read_how=_READ_HOW),
    dict(id="pololu_3419",
         url="https://www.pololu.com/product/3419",
         mirror=None,
         says="Pololu's page for its VL53L8CX carrier: in 5000 lux the VL53L8CX is expected to detect a large "
              "reflective target out to about 2.8 m, against about 1.7 m for the VL53L5CX; the 4 m maximum of both "
              "assumes an ideal dark environment; effective range and update rate depend on the configuration, the "
              "target and the environment. No resolution or rate is given with the 1.7 m.",
         read_how=_READ_HOW + "; the catalog note on pololu_vl53l8cx cites the same page"),
    dict(id="st_i2c_bandwidth_table",
         url="https://manuals.plus/m/4fea81abf81933d02393c56515009feef4d6015b81bc2f2bccf14262e8fcc3b4.pdf",
         mirror=None,
         says="A copy of an ST guide for the VL53L5CX, which the search tool names UM2887 revision 1, the software "
              "integration guide. Its table of I2C bandwidth during ranging lists packet sizes of 124, 316, 1008 and "
              "3360 bytes and bandwidths of 124, 316, 63000 and 50909 bytes per second. Three rows were read with "
              "their configuration: 1 Hz, 4x4, one target per zone, distance, target status and number of targets "
              "only: 124 bytes; 60 Hz, 4x4, four targets per zone, all output enabled: 1008 bytes, 63000 bytes per "
              "second; 15 Hz, 8x8, four targets per zone, all output enabled: 3360 bytes, 50909 bytes per second. "
              "The 316-byte row did not come back with its configuration.",
         read_how=_READ_HOW + "; the 316-byte row is taken to be the 124-byte row's outputs at 8x8, which is this "
                  "document's reading (d)"),
    dict(id="st_um3109",
         url="https://www.pololu.com/file/0J2030/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-"
             "timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf",
         mirror=None,
         says="ST user manual UM3109 for the VL53L8CX (the catalog's newer part, pololu_vl53l8cx): the same limits, "
              "4x4 1 to 60 Hz and 8x8 1 to 15 Hz.",
         read_how=_READ_HOW),
    dict(id="st_forum_multi_sensor",
         url="https://community.st.com/imaging-sensors-49/vl53l5cx-mutual-influence-of-multiple-sensors-34808",
         mirror=None,
         says="ST community answers by an ST employee: several VL53L5CX sensors interfere a little (an extra target "
              "with an invalid status); ST tested sensors set at different angles and found the effect minimal. Forum "
              "answers, not a specification.",
         read_how=_READ_HOW + "; used only as a note, kind (d)"),
]


# ------------------------------------------------------------------------------------------------
# Records
# ------------------------------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tof_catalog_sha() -> str:
    """sha256 of the canonical JSON of the two ToF catalog entries this document reads (not of the whole catalog)."""
    parts = {p["id"]: p for p in json.loads((ROOT / "designlab/data/parts.json").read_text())["parts"]}
    text = json.dumps([parts[i] for i in TOF_CATALOG_IDS], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def record_paths(body: str) -> dict:
    """The committed records a body's rows are computed from, relative to the study directory."""
    b = BODIES[body]
    scr = f"round2/refine/{b['case']}/review/screening"
    return dict(sensors=f"{scr}/{body}_sensors.json", envelope=f"{scr}/{body}_envelope.json",
                candidate=f"round2/refine/{b['case']}/candidates/{body}.json",
                transitions=f"{b['stage1_run']}/transitions.json")


def load_body(body: str) -> dict:
    """Everything the computation reads for one body, from committed records only."""
    paths = record_paths(body)
    rec = {k: json.loads((ROOT / p).read_text()) for k, p in paths.items()}
    cand, sen, env = rec["candidate"], rec["sensors"], rec["envelope"]
    if cand["selection"]["tof"] != TOF_PART or sen["fov"]["tof"]["part"] != TOF_PART:
        raise ValueError(f"{body}: the record's ToF part is not {TOF_PART}")
    stop = rec["transitions"]["transitions"]["stop"]
    H = float(cand["H"])
    specs = (("nominal", sizing.NOMINAL), ("brisk", sizing.BRISK), ("fr05", envelope.FAST))
    walks = {name: {k: float(v) for k, v in spec.describe(H).items()} for name, spec in specs}
    stop_crouch = float(stop["gaits"]["a"]["crouch"])
    if any(spec.crouch != CROUCH_FRAC for _, spec in specs) or abs(stop_crouch - CROUCH_FRAC * H) > 1e-9:
        raise ValueError(f"{body}: a reference walk does not walk {CROUCH_FRAC} H below the home height")
    ctx = dict(body=body, paths=paths, H=H, walks=walks, sensors=sen, crouch_m=CROUCH_FRAC * H,
               one_foot=float(env["stopping"]["max_one_foot_decel_ms2"]),
               foot_path_half_width=float(sen["tof"]["near_coverage"]["foot_path_half_width_m"]),
               toe_x=float(sen["model"]["toe_tip_x_m"]), hip_height=float(sen["model"]["hip_height_m"]),
               stop=stop)
    ctx["builder_steps"] = as_built_stop(ctx)["worst_phase_steps"]
    if ctx["builder_steps"] != 2.4:  # the reading's label says 2.4 steps
        raise ValueError(f"{body}: the stop builder's worst phase is {ctx['builder_steps']} steps, not 2.4")
    return ctx


# ------------------------------------------------------------------------------------------------
# Geometry: what the far row sees on the foot path
# ------------------------------------------------------------------------------------------------
def _ry(angle: float) -> np.ndarray:
    """Rotation about +y; a positive angle turns +x toward -z (nose down)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def site_rotation(yaw_deg: float, depression_deg: float) -> np.ndarray:
    """Site frame in the world: x the sensing axis, z image up (``sensors.py``'s convention), no roll."""
    return _rz(math.radians(yaw_deg)) @ _ry(math.radians(depression_deg))


def _centres(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Zone-centre image coordinates as in ``sensors._Screen.tof``: columns left to right, row 0 at the top."""
    return -1 + (2 * np.arange(n) + 1) / n, 1 - (2 * np.arange(n) + 1) / n


def _floor(o: np.ndarray, R: np.ndarray, lens: sensors.Lens, a, b) -> tuple[np.ndarray, np.ndarray]:
    """Floor points (n, 3) and slant ranges (n,) of the rays at image coordinates (a, b); nan above the horizon."""
    v = lens.world(R, a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(v[:, 2] < -1e-9, -o[2] / v[:, 2], np.nan)
    return o + v * t[:, None], t


def _sites(ctx: dict, pitch_deg: float = 0.0, drop_m: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Both sensors' positions and rotations, with the trunk pitched nose-down about the hip axis and lowered."""
    out = []
    Rp = _ry(math.radians(pitch_deg))
    c = np.array([0.0, 0.0, ctx["hip_height"]])
    for name, sgn in (("tof_l", 1.0), ("tof_r", -1.0)):
        summ = ctx["sensors"]["tof"][name]["summary"]
        o = np.array(summ["site_pos_m"], float)
        R = site_rotation(TOF_YAW_DEG * sgn, float(summ["axis_depression_deg"]))
        if np.abs(R[:, 0] - np.array(summ["axis"], float)).max() > 1e-3:
            raise ValueError(f"{ctx['body']}: {name}'s recorded axis is not 45 deg down and 12 deg outward")
        o = c + Rp @ (o - c) - np.array([0.0, 0.0, drop_m])
        out.append((o, Rp @ R))
    return out


def reproduction_error(ctx: dict) -> float:
    """Largest difference (m) between this module's 8x8 zone centres and the 64 the screen recorded, per sensor."""
    lens = sensors.Lens(ctx["sensors"]["fov"]["tof"])
    ac, bc = _centres(8)
    A, B = np.meshgrid(ac, bc)
    err = 0.0
    for (o, R), name in zip(_sites(ctx), ("tof_l", "tof_r")):
        p, t = _floor(o, R, lens, A.ravel(), B.ravel())
        for k, z in enumerate(ctx["sensors"]["tof"][name]["zones"]):
            if not z["valid"]:
                raise ValueError(f"{ctx['body']}: zone {k} of {name} is not a valid floor zone in the record")
            err = max(err, abs(p[k, 0] - z["floor_xy"][0]), abs(p[k, 1] - z["floor_xy"][1]), abs(t[k] - z["range_m"]),
                      abs(p[k, 0] - ctx["toe_x"] - z["ahead_of_toes_m"]))
    return float(err)


def row_on_path(ctx: dict, n: int, row: int, edge: str = "centre", pitch_deg: float = 0.0,
                drop_m: float = 0.0) -> dict:
    """Where one zone row meets the floor on the foot path, ahead of the toe tips (m).

    ``edge`` is "centre" (the zone-centre rays, as the screen) or "near" (the row's lower boundary: a hazard edge
    there has the whole zone beyond it).  A column counts when its zone-centre floor point is inside the foot path.
    """
    lens = sensors.Lens(ctx["sensors"]["fov"]["tof"])
    ac, bc = _centres(n)
    b_row = bc[row] if edge == "centre" else bc[row] - 1.0 / n
    xs, ys, slants, angles = [], [], [], []
    for o, R in _sites(ctx, pitch_deg, drop_m):
        pc, _ = _floor(o, R, lens, ac, np.full(n, bc[row]))
        p, t = _floor(o, R, lens, ac, np.full(n, b_row))
        for k in range(n):
            if abs(pc[k, 1]) <= ctx["foot_path_half_width"] and np.isfinite(p[k, 0]):
                xs.append(float(p[k, 0] - ctx["toe_x"]))
                ys.append(float(pc[k, 1]))
                slants.append(float(t[k]))
                angles.append(math.degrees(math.asin(o[2] / t[k])))  # between the ray and the floor
    return dict(ahead_min_m=min(xs), ahead_max_m=max(xs), zones=len(xs), slant_max_m=max(slants),
                floor_angle_deg=[min(angles), max(angles)], lateral_m=sorted(round(y, 4) for y in ys))


def sight(ctx: dict, n: int, row: int = 0, edge: str = "centre", pitch_deg: float = 0.0,
          standing: bool = False) -> list[float]:
    """[nearest, farthest] floor point (m ahead of the toes) of a zone row on the foot path, at the walking height
    (the home pose lowered by the reference walk's crouch) unless ``standing``."""
    r = row_on_path(ctx, n, row, edge=edge, pitch_deg=pitch_deg, drop_m=0.0 if standing else ctx["crouch_m"])
    return [r["ahead_min_m"], r["ahead_max_m"]]


def _mid(pair) -> float:
    return 0.5 * (pair[0] + pair[1])


def far_row_fill_height(ctx: dict, n: int) -> float:
    """Height (m) of the far row's top boundary ray above the row's near-boundary floor point, largest on the path,
    at the walking height.

    An obstacle whose face stands at the far row's near boundary fills that zone only if it is at least this tall.
    """
    lens = sensors.Lens(ctx["sensors"]["fov"]["tof"])
    ac, bc = _centres(n)
    worst = 0.0
    for o, R in _sites(ctx, drop_m=ctx["crouch_m"]):
        pc, _ = _floor(o, R, lens, ac, np.full(n, bc[0]))
        pn, _ = _floor(o, R, lens, ac, np.full(n, bc[0] - 1.0 / n))
        top = lens.world(R, ac, np.full(n, 1.0))
        for k in range(n):
            if abs(pc[k, 1]) <= ctx["foot_path_half_width"]:
                d = float(np.hypot(*(pn[k, :2] - o[:2])))
                worst = max(worst, float(o[2] + top[k, 2] / np.hypot(top[k, 0], top[k, 1]) * d))
    return worst


#: The other readings of what the sensors see (``sight`` arguments), each priced in the sensitivity.
SIGHT_WHAT_IFS = {
    "full_zone": dict(label="the hazard must fill a far-row zone on the foot path (the row's near edge)",
                      args=dict(edge="near")),
    "second_row": dict(label="the far row is not trusted: the second row's zone centres", args=dict(row=1)),
    "pitch_rms": dict(label=f"trunk pitched {PITCH_RMS_DEG:g} deg nose-down (the spec's RMS limit)",
                      args=dict(pitch_deg=PITCH_RMS_DEG)),
    "pitch_peak": dict(label=f"trunk pitched {PITCH_PEAK_DEG:g} deg nose-down (the spec's peak limit)",
                       args=dict(pitch_deg=PITCH_PEAK_DEG)),
    "standing": dict(label=f"standing height: the screen's home pose, {CROUCH_FRAC:g} H above the walk's",
                     args=dict(standing=True)),
}


def horizons(ctx: dict, mode: str) -> dict:
    """The far row's reach on the foot path for one ToF mode, with the alternative readings of the sensitivity."""
    n = MODES[mode]["zones"]
    far = row_on_path(ctx, n, 0, drop_m=ctx["crouch_m"])
    home = sight(ctx, n, standing=True)
    out = dict(
        zone_deg=ctx["sensors"]["fov"]["tof"]["hfov_deg"] / n,
        far_row_zones_on_path=far["zones"],
        far_row_lateral_m=far["lateral_m"],
        home_all_m=home[0],
        home_far_m=home[1],
        all_m=far["ahead_min_m"],
        far_m=far["ahead_max_m"],
        far_row_slant_max_m=far["slant_max_m"],
        far_row_floor_angle_deg=far["floor_angle_deg"],
        alt_m={k: sight(ctx, n, **w["args"]) for k, w in SIGHT_WHAT_IFS.items()},
        far_row_fill_height_m=far_row_fill_height(ctx, n),
    )
    out["spec_obstacle_fills_far_row_zone"] = bool(out["far_row_fill_height_m"] <= OBSTACLE_M)
    out["ref_m"] = _mid((out["all_m"], out["far_m"]))
    if n == 8:
        recorded = max(ctx["sensors"]["summary"]["tof"][s]["floor_ahead_of_toes_m"][1] for s in ("tof_l", "tof_r"))
        if abs(recorded - out["home_far_m"]) > 5e-4:
            raise ValueError(f"{ctx['body']}: the far row does not reproduce the screen's {recorded} m")
        out["screen_record_far_m"] = float(recorded)
    return out


# ------------------------------------------------------------------------------------------------
# Timing: the reaction time of a stop path
# ------------------------------------------------------------------------------------------------
def term_range(term: str, mode: str) -> tuple[float, float, float]:
    """(low, reference, high) of one timing term, seconds.  The reference is the midpoint unless a rule sets it."""
    n, period = MODES[mode]["zones"], 1.0 / MODES[mode]["rate_hz"]
    if term == "tof_frames":
        return FRAMES[0] * period, FRAMES[1] * period, FRAMES[2] * period
    if term == "servo_delay":
        return SERVO_DELAY_S
    if term == "i2c_read":  # one sensor at 1 MHz; two sensors read back to back at 400 kHz; the smallest packet
        byte_s = PACKET_BYTES[n][0] * I2C_BITS_PER_BYTE
        lo, hi = byte_s / I2C_HZ[1], 2 * byte_s / I2C_HZ[0]
    elif term in ("mcu_loop", "rt_core_loop"):
        lo, hi = 1.0 / MCU_LOOP_HZ[1], 1.0 / MCU_LOOP_HZ[0]
    elif term == "walk_loop":
        lo, hi = 1.0 / WALK_LOOP_HZ[1], 1.0 / WALK_LOOP_HZ[0]
    elif term == "host_planner":
        lo, hi = HOST_PLANNER_S
    elif term == "host_link":
        lo, hi = HOST_LINK_S
    else:
        raise KeyError(term)
    return lo, 0.5 * (lo + hi), hi


def reaction(mode: str, path: str, level: str = "ref", overrides: dict | None = None) -> float:
    """Reaction time (s) of a path with every term at one level ("low", "ref", "high"), or as overridden (s)."""
    k = ("low", "ref", "high").index(level)
    over = overrides or {}
    return float(sum(over.get(term, term_range(term, mode)[k]) for term in PATHS[path]["terms"]))


# ------------------------------------------------------------------------------------------------
# Stops and the governor
# ------------------------------------------------------------------------------------------------
def walk_at(ctx: dict, v: float) -> dict:
    """Stride (m) and duty of the study's reference walk at speed v, and the rule that gave them."""
    w = ctx["walks"]
    if v <= w["nominal"]["speed"]:
        return dict(stride=v * w["nominal"]["period"], duty=w["nominal"]["duty"],
                    rule="nominal walk's period and duty held (extrapolation)")
    if v >= w["fr05"]["speed"]:
        return dict(stride=v * w["fr05"]["period"], duty=w["fr05"]["duty"],
                    rule="Froude 0.05 walk's period and duty held (extrapolation)")
    lo, hi = (sizing.NOMINAL, sizing.BRISK) if v < w["brisk"]["speed"] else (sizing.BRISK, envelope.FAST)
    spec = look.speed_spec(ctx["H"], v, lo, hi)
    return dict(stride=float(spec.stride_rel) * ctx["H"], duty=float(spec.duty), rule="look.speed_spec")


def stride_m(ctx: dict, v: float) -> tuple[float, str]:
    """Stride length (m) of the study's reference walk at speed v, and the rule that gave it."""
    w = walk_at(ctx, v)
    return w["stride"], w["rule"]


def foot_lead_m(ctx: dict, v: float) -> float:
    """How far ahead of the home-stance toe line a stance foot lands in the reference walk at speed v (m):
    duty / 2 x stride (``gait._foot_x``: a stance foot is placed duty x stride / 2 ahead of its home position)."""
    w = walk_at(ctx, v)
    return 0.5 * w["duty"] * w["stride"]


def foot_lead_binds_below(ctx: dict, reading: str) -> float:
    """The speed (m/s) below which a deceleration reading's own distance is shorter than the foot's lead."""
    a = SPEC_DECEL if reading == "spec_decel" else ctx["one_foot"]
    lo, hi = 0.05, 5.0  # the lead is the longer of the two at 0.05 m/s and the shorter at 5 m/s
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if foot_lead_m(ctx, mid) >= mid * mid / (2.0 * a):
            lo = mid
        else:
            hi = mid
    return lo


def stop_parts(ctx: dict, reading: str, v: float, steps: float | None = None) -> dict:
    """The stop distance (m) under one reading and what sets it; ``steps`` overrides the reading with a step count.

    ``reading``: the reading's own distance.  ``spec_decel``: v^2 / (2 x 1.5), the floor under a step reading.
    ``foot_lead``: the lead of a foot already on the ground, the floor under every reading.
    """
    w = walk_at(ctx, v)
    parts = {}
    if steps is None and reading in DECEL_READINGS:
        a = SPEC_DECEL if reading == "spec_decel" else ctx["one_foot"]
        parts["reading"] = v * v / (2.0 * a)
    else:
        n = steps if steps is not None else (ctx["builder_steps"] if reading == "stop_builder_worst"
                                             else STEP_READINGS[reading])
        parts["reading"] = n * 0.5 * w["stride"]
        parts["spec_decel"] = v * v / (2.0 * SPEC_DECEL)
    parts["foot_lead"] = 0.5 * w["duty"] * w["stride"]
    set_by = max(parts, key=parts.get)  # the reading itself wins a tie
    return dict(d_stop=parts[set_by], set_by=set_by, step_m=0.5 * w["stride"], parts=parts)


def stop_distance(ctx: dict, reading: str, v: float, steps: float | None = None) -> float:
    """Distance (m) the front of the feet travels in the stop, under one reading (``stop_parts``)."""
    return stop_parts(ctx, reading, v, steps)["d_stop"]


def governor(ctx: dict, reading: str, d_see: float, t_react: float, steps: float | None = None) -> float:
    """Largest speed (m/s) with d_see - v * t_react >= d_stop(v); 0 if there is no room at all."""
    if d_see <= 0.0:
        return 0.0
    lo, hi = 0.0, 10.0  # the spare distance falls as v rises: bisect its zero
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if d_see - mid * t_react - stop_distance(ctx, reading, mid, steps) >= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def as_built_stop(ctx: dict) -> dict:
    """The stop reference the study has built (Stage 1's ``stop``, nominal walk to stand), read from its record."""
    s = ctx["stop"]
    ga = s["gaits"]["a"]
    stride = ga["speed"] * ga["period"]
    r0, r1 = s["ramps"]["path"]
    last = s["steps"][-1]
    return dict(
        from_walk=ga["name"], speed_m_s=_r(ga["speed"]), stride_m=_r(stride), duty=ga["duty"],
        speed_ramp_phase=[r0, r1],
        # a minimum-jerk speed ramp is symmetric, so the path covers half the ramp's length at the walk's speed
        travel_after_command_strides=_r(0.5 * (r1 - r0)),
        travel_after_command_m=_r(0.5 * (r1 - r0) * stride, 5),
        last_step=dict(side=last["side"], lift_off_phase=last["lift_off_phase"], length_m=_r(last["length_m"])),
        single_step=bool(s["stop"]["single_step"]),
        entry="the stop begins at a left lift-off only (no mirrored stop is built), so a request waits up to one "
              "stride for it",
        worst_phase_strides=_r(1.0 + 0.5 * (r1 - r0)),
        worst_phase_steps=_r(2.0 * (1.0 + 0.5 * (r1 - r0))),
        only_stop_built="nominal walk to stand; no stop reference exists from a faster walk (STAGE1_PLAN.md "
                        "decision 31), and Stage 2's two-step stop is deferred (designlab/stage2/look.py DEFERRED)",
    )


# ------------------------------------------------------------------------------------------------
# The record
# ------------------------------------------------------------------------------------------------
def inputs(ctxs: dict) -> list[dict]:
    """Every input with its kind and source.  Ranged inputs carry low, ref and high."""
    rows = []

    def add(id_, kind, unit, source, basis=None, **values):
        rows.append(dict(id=id_, kind=kind, unit=unit, **values, source=source, basis=basis))

    for body, ctx in ctxs.items():
        p = ctx["paths"]
        s = ctx["sensors"]["tof"]["tof_l"]["summary"]
        add(f"{body}.tof_site", "a", "m", f"{p['sensors']} tof.tof_l.summary (tof_r mirrored)",
            value=dict(site_pos_m=s["site_pos_m"], axis=s["axis"], axis_depression_deg=s["axis_depression_deg"]))
        add(f"{body}.screen_far_floor", "a", "m", f"{p['sensors']} summary.tof.tof_l.floor_ahead_of_toes_m[1]",
            value=ctx["sensors"]["summary"]["tof"]["tof_l"]["floor_ahead_of_toes_m"][1])
        add(f"{body}.toe_tip_x", "a", "m", f"{p['sensors']} model.toe_tip_x_m", value=ctx["toe_x"])
        add(f"{body}.foot_path_half_width", "a", "m", f"{p['sensors']} tof.near_coverage.foot_path_half_width_m",
            value=ctx["foot_path_half_width"])
        add(f"{body}.hip_height_screen", "a", "m", f"{p['sensors']} model.hip_height_m", value=ctx["hip_height"])
        add(f"{body}.H", "a", "m", f"{p['candidate']} H", value=ctx["H"])
        add(f"{body}.one_foot_decel", "a", "m/s^2", f"{p['envelope']} stopping.max_one_foot_decel_ms2 "
            "(designlab/envelope.py; DECISION_LOG.md F12, F23)", value=ctx["one_foot"])
        add(f"{body}.stop_reference", "a", "-", f"{p['transitions']} transitions.stop (ramps, gaits.a, steps)",
            value=dict(ramps_path=ctx["stop"]["ramps"]["path"], speed_m_s=ctx["stop"]["gaits"]["a"]["speed"],
                       period_s=ctx["stop"]["gaits"]["a"]["period"], crouch_m=ctx["stop"]["gaits"]["a"]["crouch"]))
    add("tof_part", "c", "-", "designlab/data/parts.json id st_vl53l5cx_carrier (the records' selection.tof)",
        value=TOF_PART)
    add("tof_fov", "c", "deg", "designlab/data/parts.json id st_vl53l5cx_carrier, fov; designlab/sensors.py SPECS. "
        "The datasheet's revision 2 said 63 deg on the diagonal and revision 13 says 65 deg (web sources "
        "st_vl53l5cx_datasheet, st_an5894); the screen's equidistant lens gives 63.6 deg",
        value=dict(hfov_deg=45.0, vfov_deg=45.0, dfov_deg=65.0, range_m=[0.02, 4.0]))
    add("tof_4x4_layout", "d", "-", "none",
        "This document's: the 4x4 zones tile the same 45 x 45 deg field, each one four of the 8x8 zones; the catalog "
        "says only '4x4 or 8x8 zones'", value="same field, zones twice as wide")
    add("tof_mount_aim", "a", "deg", "designlab/build.py (two ToF sensors under the chest); DECISION_LOG.md F4",
        value=dict(down_deg=45.0, outward_deg=TOF_YAW_DEG))
    add("tof_rate_8x8", "c", "Hz", "ST UM2884 table 2 (web source st_um2884): 8x8 ranges at 1 to 15 Hz",
        value=MODES["8x8_15hz"]["rate_hz"])
    add("tof_rate_4x4", "c", "Hz", "ST UM2884 table 2 (web source st_um2884): 4x4 ranges at 1 to 60 Hz", low=1.0,
        high=MODES["4x4_60hz"]["rate_hz"])
    add("tof_rate_4x4_table", "d", "Hz", "none",
        "This document's choice of a second 4x4 rate, allowed by UM2884's 1 to 60 Hz: the review's search extract "
        "has the datasheet's 4x4 range table at 30 Hz; this document's own searches did not return the table",
        value=MODES["4x4_30hz"]["rate_hz"])
    add("tof_range_ambient", "c", "m", "designlab/data/parts.json id pololu_vl53l8cx, fov note; web source "
        "pololu_3419: in 5000 lux about 1.7 m for the VL53L5CX, for a large reflective target, with no resolution or "
        "rate given. The vendor's own range tables, per resolution and rate, were not read", value=1.7)
    add("i2c_rate", "c", "Hz", "ST VL53L5CX datasheet (web source st_vl53l5cx_datasheet)", low=I2C_HZ[0],
        high=I2C_HZ[1])
    add("i2c_packet_bytes", "d", "bytes", "web source st_i2c_bandwidth_table",
        "This document's: the smallest and largest packets in ST's table. Read with their rows: 124 (4x4, one "
        "target, three outputs), 1008 (4x4, every output), 3360 (8x8, every output); 316 is taken as the 8x8 "
        "counterpart of 124. That the stop needs only the smallest set of outputs is assumed",
        value={f"{n}x{n}": list(b) for n, b in PACKET_BYTES.items()})
    add("i2c_read", "d", "s", "derived",
        "This document's: bytes x 9 bits / bus rate, the smallest packet; low = one sensor at 1 MHz, high = both "
        "sensors read back to back at 400 kHz",
        value={m: [_r(x, 6) for x in term_range("i2c_read", m)] for m in MODES})
    add("tof_frames", "d", "frames", "derived",
        "This document's: 1 = one frame at the best phase, 2 = one frame at the worst phase, 3 = a two-frame "
        "confirmation at the worst phase; the firmware is not written", low=FRAMES[0], ref=FRAMES[1], high=FRAMES[2])
    add("mcu_loop", "c", "Hz", "COMPUTE.md, 'What the onboard computer has to do': CAN loop, IMU, ToF stop, "
        "watchdogs at 200-1000 Hz on the microcontroller", low=MCU_LOOP_HZ[0], high=MCU_LOOP_HZ[1])
    add("rt_core_loop", "d", "Hz", "none",
        "This document's: taken equal to the microcontroller's 200-1000 Hz; no document of the study fixes it and "
        "PR #668 (unverified) gives no measured period", low=MCU_LOOP_HZ[0], high=MCU_LOOP_HZ[1])
    add("walk_loop", "c", "Hz", "COMPUTE.md, same table: locomotion policy at 50-100 Hz, on the microcontroller "
        "(stop paths 1 and 3)", low=WALK_LOOP_HZ[0], high=WALK_LOOP_HZ[1])
    add("rt_core_walk_loop", "d", "Hz", "none",
        "This document's: on stop path 2 the walking loop runs on the Jetson's real-time core; its rate is taken "
        "equal to COMPUTE.md's 50-100 Hz for the microcontroller, which no document of the study fixes for the Jetson",
        low=WALK_LOOP_HZ[0], high=WALK_LOOP_HZ[1])
    add("planner_rate", "c", "Hz", "COMPUTE.md, same table: ground-frame tracking, obstacle grid, planner at "
        "10-15 Hz. Not used in the ranges; given for scale (one period is 67-100 ms)", low=PLANNER_HZ[0],
        high=PLANNER_HZ[1])
    add("host_deadline", "c", "s", "BEHAVIOR_SPEC.md section 3, Compute timing: at most 200 ms from the oldest sensor "
        "sample to a motion request. A deadline (a target), not a measured latency", value=HOST_PLANNER_S[1])
    add("host_planner", "d", "s", "derived",
        "This document's: low = 0, a bound (a planner at its best phase waits almost nothing, and no compute time "
        "is sourced); high = the spec's deadline, assumed met. No latency has been measured on the Pi 5 or the "
        "Jetson, so both hosts take the same range", low=HOST_PLANNER_S[0], high=HOST_PLANNER_S[1])
    add("host_link", "d", "s", "none", "This document's: a serial or USB hop from the host to the real-time layer",
        low=HOST_LINK_S[0], high=HOST_LINK_S[1])
    add("servo_delay", "d", "s", "DECISION_LOG.md P29 (and the owner's 14 Nov decision, P30): the bar is judged at "
        "10 ms until a bench value exists; A17",
        "The study's assumption: reference 10 ms, range 0-20 ms as Stage 1 simulated (LESSONS_LEARNED.md 2.3)",
        low=SERVO_DELAY_S[0], ref=SERVO_DELAY_S[1], high=SERVO_DELAY_S[2])
    add("spec_decel", "c", "m/s^2", "BEHAVIOR_SPEC.md section 1, Start / stop: peak horizontal trunk acceleration at "
        "most 1.5 m/s^2; STAGE1_PLAN.md decision 10; STAGE2_PLAN.md section 5 gates it in the bar's stop; "
        "designlab/envelope.py STOP_DECEL. A limit (a target), not a deceleration the robot has shown",
        value=SPEC_DECEL)
    add("stop_steps", "c", "steps", "STAGE2_PLAN.md section 5, the bar: stop within a step or two (at most two "
        "touchdowns after the stop command). A target the bar sets, not a stop the robot has shown", low=1.0,
        high=2.0)
    add("spec_stop_strides", "c", "strides", "BEHAVIOR_SPEC.md section 1, Start / stop: stop within 1.5 strides; "
        "STAGE1_PLAN.md decision 18. A target, not a stop the robot has shown", value=SPEC_STOP_STRIDES)
    add("reference_walks", "a", "-", "designlab/sizing.py NOMINAL, BRISK; designlab/envelope.py FAST; "
        "designlab/stage2/look.py speed_spec", value="period and duty linear in speed between the three walks")
    add("stride_extrapolation", "d", "-", "none",
        "This document's: outside the three walks the nearest walk's period and duty are held",
        value="flagged where used")
    add("foot_lead", "a", "stride", "designlab/gait.py _foot_x: a stance foot is placed duty x stride / 2 ahead of "
        "its home position, so it lands that far ahead of the home-stance toe line", value="duty / 2")
    add("bar_speed", "c", "m/s", "STAGE2_PLAN.md section 5; DECISION_LOG.md P29", value=BAR_SPEED)
    add("speed_ladder", "c", "m/s", "STAGE2_PLAN.md section 5, Ladders", value=list(LADDER))
    add("trunk_pitch_limits", "c", "deg", "BEHAVIOR_SPEC.md section 1, Body motion: at most 4 deg RMS, 8 deg peak",
        value=[PITCH_RMS_DEG, PITCH_PEAK_DEG])
    add("pitch_axis", "d", "-", "none", "This document's: the trunk is pitched about the hip axis, taken at x = 0 and "
        "the screen record's hip height", value="hip axis")
    add("crouch", "a", "H", "designlab/gait.py GaitSpec.crouch default: every reference walk holds the trunk this far "
        "below the home height, so the main table's sight distance is taken there", value=CROUCH_FRAC)
    add("standoff", "d", "m", "none", "This document's what-if: a gap kept between the toes and the hazard; the main "
        "table keeps none", value=STANDOFF_M)
    add("obstacle_height", "c", "m", "BEHAVIOR_SPEC.md section 3: no contact with static obstacles at least 0.10 m "
        "tall", value=OBSTACLE_M)
    return rows


def _end(ctx: dict, reading: str, d_see: float, t_react: float) -> dict:
    """One end of a governor range, with the stop it implies."""
    v = governor(ctx, reading, d_see, t_react)
    sp = stop_parts(ctx, reading, v)
    out = dict(m_s=_r(v), d_see_m=_r(d_see), reaction_s=_r(t_react, 5), d_stop_m=_r(sp["d_stop"]),
               stop_set_by=sp["set_by"], implied_decel_ms2=_r(v * v / (2.0 * sp["d_stop"]), 3))
    if reading not in DECEL_READINGS:
        out["step_m"] = _r(sp["step_m"])
        out["step_over_H"] = _r(sp["step_m"] / ctx["H"], 3)
    return out


def _cell(ctx: dict, mode: str, path: str, reading: str, hz: dict) -> dict:
    lo = _end(ctx, reading, hz["all_m"], reaction(mode, path, "high"))
    hi = _end(ctx, reading, hz["far_m"], reaction(mode, path, "low"))
    top = ctx["walks"]["fr05"]["speed"]
    return dict(low_m_s=lo["m_s"], high_m_s=hi["m_s"], low_above_fastest_reference=bool(lo["m_s"] > top),
                high_above_fastest_reference=bool(hi["m_s"] > top), low=lo, high=hi)


def _ledger(ctx: dict, mode: str, path: str, hz: dict, v: float) -> dict:
    t_lo, t_hi = reaction(mode, path, "low"), reaction(mode, path, "high")
    w = walk_at(ctx, v)
    out = dict(speed_m_s=v, d_see_m=[_r(hz["all_m"]), _r(hz["far_m"])],
               reaction_m=[_r(v * t_lo), _r(v * t_hi)], stride_m=_r(w["stride"]), duty=_r(w["duty"]),
               stride_rule=w["rule"], foot_lead_m=_r(foot_lead_m(ctx, v)), stop_m={}, stop_set_by={}, spare_m={})
    for reading in READINGS:
        sp = stop_parts(ctx, reading, v)
        out["stop_m"][reading] = _r(sp["d_stop"])
        out["stop_set_by"][reading] = sp["set_by"]
        out["spare_m"][reading] = [_r(hz["all_m"] - v * t_hi - sp["d_stop"]), _r(hz["far_m"] - v * t_lo - sp["d_stop"])]
    return out


def _sensitivity(ctx: dict, mode: str, path: str, reading: str, hz: dict) -> dict:
    """One term at a time about the reference values, then the what-ifs, for one body, mode, path and reading.

    The reference takes the midpoint of the sight range and of each timing range (2 frames; 10 ms of command
    delay).  Each what-if changes one thing: a reading of the sight is evaluated at the midpoint of its own range.
    """
    base_t = reaction(mode, path, "ref")
    base = governor(ctx, reading, hz["ref_m"], base_t)
    terms = [dict(term="d_see", label="far row: nearest to farthest zone centre on the foot path", kind="a",
                  range=[_r(hz["all_m"]), _r(hz["far_m"])], unit="m",
                  governor_m_s=[_r(governor(ctx, reading, hz["all_m"], base_t)),
                                _r(governor(ctx, reading, hz["far_m"], base_t))])]
    for term in PATHS[path]["terms"]:
        lo, _, hi = term_range(term, mode)
        terms.append(dict(term=term, label=TERM_LABELS[term], kind=term_kind(term, path),
                          range=[_r(lo, 5), _r(hi, 5)], unit="s",
                          governor_m_s=[_r(governor(ctx, reading, hz["ref_m"],
                                                    reaction(mode, path, "ref", {term: hi}))),
                                        _r(governor(ctx, reading, hz["ref_m"],
                                                    reaction(mode, path, "ref", {term: lo})))]))
    for t in terms:
        t["swing_m_s"] = _r(abs(t["governor_m_s"][1] - t["governor_m_s"][0]))
    terms.sort(key=lambda t: -t["swing_m_s"])
    period = 1.0 / MODES[mode]["rate_hz"]
    n = MODES[mode]["zones"]
    what = []

    def add(key, label, kind, v, **more):
        what.append(dict(what_if=key, label=label, kind=kind, governor_m_s=_r(v), change_m_s=_r(v - base), **more))

    for key, w in SIGHT_WHAT_IFS.items():
        add(key, w["label"], "a", governor(ctx, reading, _mid(hz["alt_m"][key]), base_t),
            d_see_m=[_r(x) for x in hz["alt_m"][key]])
    add("standoff", f"a {STANDOFF_M:.2f} m gap kept between the toes and the hazard", "d",
        governor(ctx, reading, hz["ref_m"] - STANDOFF_M, base_t))
    add("sensors_take_turns", "the two sensors range in turn (each frame period doubles)", "d",
        governor(ctx, reading, hz["ref_m"], reaction(mode, path, "ref", {"tof_frames": 2 * FRAMES[1] * period})))
    add("every_output_read", "every output is read from both sensors at 1 MHz (the largest packet)", "d",
        governor(ctx, reading, hz["ref_m"], reaction(mode, path, "ref", {
            "i2c_read": 2 * PACKET_BYTES[n][1] * I2C_BITS_PER_BYTE / I2C_HZ[1]})))
    for other in READINGS:
        if other != reading:
            add(f"stop_{other}", f"stop reading: {READING_LABELS[other]}", READING_KINDS[other],
                governor(ctx, other, hz["ref_m"], base_t))
    what.sort(key=lambda w: w["change_m_s"])
    return dict(base=dict(d_see_m=_r(hz["ref_m"]), reaction_s=_r(base_t, 5), governor_m_s=_r(base)),
                terms=terms, what_ifs=what)


def _stress(ctx: dict, mode: str, path: str) -> dict:
    """Every adverse reading of the sensitivity at once (this document's stack, kind d; not a likely case).

    Sight: the nearest point of the far row's near edge on the foot path with the trunk at the spec's peak pitch,
    nose-down, at the walking height, less the stand-off.  Timing: every term high.  Stop: each reading.
    """
    n = MODES[mode]["zones"]
    d = sight(ctx, n, edge="near", pitch_deg=PITCH_PEAK_DEG)[0] - STANDOFF_M
    t = reaction(mode, path, "high")
    top, bottom = ctx["walks"]["fr05"]["speed"], ctx["walks"]["nominal"]["speed"]
    out = dict(d_see_m=_r(d), reaction_s=_r(t, 5), governor={})
    for reading in READINGS:
        v = governor(ctx, reading, d, t)
        out["governor"][reading] = dict(governor_m_s=_r(v), above_fastest_reference=bool(v > top),
                                        below_bar=bool(v < BAR_SPEED), below_nominal_walk=bool(v < bottom))
    return out


def compute() -> dict:
    ctxs = {body: load_body(body) for body in BODIES}
    doc = dict(
        study="stage2_jetson75_tof_governor",
        written=WRITTEN,
        status="simulation and planning only; nothing here is a hardware measurement or a speed the robot has shown",
        task="JETSON_PLEM_REVIEW.md section 5, task 10: a speed governor, not a speed claim",
        kinds=KINDS,
        condition="d_see - v * T_react >= d_stop(v); low end: nearest d_see, every timing term high; high end: "
                  "farthest d_see, every timing term low; d_see at the walking height; a step reading is never "
                  "shorter than v^2 / (2 x 1.5 m/s^2), and no reading shorter than the lead of a foot on the ground",
        result_kind="a (a static and kinematic model result of this study); its range comes from the (d) inputs",
        inputs=inputs(ctxs),
        web_sources=WEB_SOURCES,
        modes={m: dict(label=v["label"], zones=v["zones"], rate_hz=v["rate_hz"], period_s=_r(1.0 / v["rate_hz"], 5),
                       rate_basis=v["rate_basis"], vendor_range=v["vendor_range"]) for m, v in MODES.items()},
        paths={p: dict(label=v["label"], short=v["short"], terms=list(v["terms"]), design=v["design"],
                       if_linux_fails=v["if_linux_fails"]) for p, v in PATHS.items()},
        readings={r: READING_LABELS[r] for r in READINGS},
        reading_kinds=READING_KINDS,
        term_labels=TERM_LABELS,
        term_kinds=TERM_KINDS,
        reaction={m: {p: dict(terms={t: dict(kind=term_kind(t, p), label=TERM_LABELS[t],
                                             low_ref_high_s=[_r(x, 5) for x in term_range(t, m)])
                                     for t in PATHS[p]["terms"]},
                              total_low_ref_high_s=[_r(reaction(m, p, lv), 5) for lv in ("low", "ref", "high")])
                      for p in PATHS} for m in MODES},
        bodies={},
        provenance=dict(records={}, tof_catalog_entries=list(TOF_CATALOG_IDS),
                        tof_catalog_entries_sha256=tof_catalog_sha(),
                        note="sha256 of every committed record read, and of the canonical JSON (sorted keys, no "
                             "spaces) of the two ToF entries of designlab/data/parts.json; no git-ignored file is "
                             "read"),
    )
    for body, ctx in ctxs.items():
        doc["provenance"]["records"][body] = {k: dict(path=p, sha256=_sha(ROOT / p)) for k, p in ctx["paths"].items()}
        hz = {m: horizons(ctx, m) for m in MODES}
        b = dict(
            refine_case=BODIES[body]["case"], H_m=_r(ctx["H"], 6), crouch_m=_r(ctx["crouch_m"], 5),
            one_foot_decel_ms2=_r(ctx["one_foot"]), foot_path_half_width_m=ctx["foot_path_half_width"],
            walks={k: dict(speed_m_s=_r(w["speed"]), stride_m=_r(w["stride"]), period_s=_r(w["period"]),
                           duty=w["duty"], foot_lead_m=_r(0.5 * w["duty"] * w["stride"]))
                   for k, w in ctx["walks"].items()},
            fastest_reference_m_s=_r(ctx["walks"]["fr05"]["speed"]),
            foot_lead=dict(at_bar_m=_r(foot_lead_m(ctx, BAR_SPEED)),
                           binds_below_m_s={r: _r(foot_lead_binds_below(ctx, r)) for r in DECEL_READINGS}),
            zone_reproduction_max_error_m=_r(reproduction_error(ctx), 5),
            stop_reference=as_built_stop(ctx),
            horizons={m: _round_tree(h) for m, h in hz.items()},
            governor={m: {p: {r: _cell(ctx, m, p, r, hz[m]) for r in READINGS} for p in PATHS} for m in MODES},
            ledger_at_bar={m: {p: _ledger(ctx, m, p, hz[m], BAR_SPEED) for p in PATHS} for m in MODES},
            ladder={m: {p: {f"{v:.2f}": dict(
                stride_rule=stride_m(ctx, v)[1],
                worst_spare_m={r: _r(hz[m]["all_m"] - v * reaction(m, p, "high") - stop_distance(ctx, r, v))
                               for r in READINGS}) for v in LADDER} for p in PATHS} for m in MODES},
            sensitivity={m: {p: {r: _sensitivity(ctx, m, p, r, hz[m]) for r in ("one_foot_hold", "two_steps")}
                             for p in PATHS} for m in MODES},
            stress={m: {p: _stress(ctx, m, p) for p in PATHS} for m in MODES},
        )
        cells = [(c["low_m_s"], m, p, r) for m in MODES for p in PATHS for r, c in b["governor"][m][p].items()]
        low = min(cells)
        b["lowest_governor_m_s"] = low[0]
        b["lowest_governor_at"] = dict(mode=low[1], path=low[2], reading=low[3])
        b["lowest_by_reading_m_s"] = {r: min(c[0] for c in cells if c[3] == r) for r in READINGS}
        b["highest_governor_m_s"] = max(c["high_m_s"] for m in MODES for p in PATHS
                                        for c in b["governor"][m][p].values())
        b["ladder_fits_worst_case"] = bool(all(s >= 0.0 for m in MODES for p in PATHS
                                               for row in b["ladder"][m][p].values()
                                               for s in row["worst_spare_m"].values()))
        doc["bodies"][body] = b
    return doc


def _round_tree(x):
    """Floats to 4 places through dicts and lists; everything else unchanged."""
    if isinstance(x, float):
        return _r(x)
    if isinstance(x, dict):
        return {k: _round_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_tree(v) for v in x]
    return x


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1) + "\n"


# ------------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------------
#: Marks in the tables.
MARK_ABOVE_REFERENCE = "†"   # the whole range is above the body's fastest reference walk
MARK_EXTRAPOLATED = "‡"      # the stride is extrapolated beyond the reference walks
MARK_SPEC_CAP = "¶"          # the spec's 1.5 m/s^2 sets the stop, not the step count
MARK_FOOT_LEAD = "◊"         # the lead of a foot on the ground sets the stop distance
MARK_BELOW_BAR = "↓"         # below the bar's 0.40 m/s
MARK_BELOW_NOMINAL = "§"     # below the nominal walk


def _ms(x: float) -> str:
    return f"{1000 * x:.1f}"


def _num(x: float) -> str:
    return f"{int(x)}" if float(x).is_integer() else f"{x:.6g}"


def _cap(label: str) -> str:
    """A label as a table shows it: a capital first letter and a typeset square."""
    return (label[:1].upper() + label[1:]).replace("^2", "²")


def _rng(lo: float, hi: float, nd: int = 2) -> str:
    return f"{lo:.{nd}f}–{hi:.{nd}f}"


def cell_text(c: dict) -> str:
    """A governor cell as the tables show it: low–high, each end marked when the spec's limit sets its stop."""
    ends = [f"{c[k]['m_s']:.2f}" + (MARK_SPEC_CAP if c[k]["stop_set_by"] == "spec_decel" else "")
            for k in ("low", "high")]
    return "–".join(ends) + (f" {MARK_ABOVE_REFERENCE}" if c["low_above_fastest_reference"] else "")


def tables(doc: dict) -> dict:
    """The tables of TOF_GOVERNOR.md, each a list of Markdown lines, made from the record."""
    out = {}
    bodies, modes, paths, readings = list(doc["bodies"]), doc["modes"], doc["paths"], doc["readings"]
    kinds = doc["reading_kinds"]
    bar = doc_bar(doc)
    # 1. what the far row sees
    L = ["| Body | ToF mode | Frame period, ms (c) | Zone, deg (c) | d_see at the home pose, the screen's, m (a) | "
         "d_see at the walking height, used here, m (a) | Slant range to it, m (a) | Angle to the floor there, deg "
         "(a) | Vendor's range for this mode | Second row, m (a) | Far-row zone filled, m (a) | Height that fills a "
         "far-row zone, m (a) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for body in bodies:
        for m, mode in modes.items():
            h = doc["bodies"][body]["horizons"][m]
            L.append(f"| `{body}` | {mode['label']} | {_ms(mode['period_s'])} | {h['zone_deg']:.2f} | "
                     f"{_rng(h['home_all_m'], h['home_far_m'], 3)} | {_rng(h['all_m'], h['far_m'], 3)} | "
                     f"{h['far_row_slant_max_m']:.3f} | {_rng(*h['far_row_floor_angle_deg'], 0)} | "
                     f"{_cap(mode['vendor_range'])} | {_rng(*h['alt_m']['second_row'], 3)} | "
                     f"{_rng(*h['alt_m']['full_zone'], 3)} | {h['far_row_fill_height_m']:.3f} |")
    out["horizons"] = L
    # 2. the timing terms
    L = ["| Timing term | Kind | " + " | ".join(f"{mode['label']}: low–high, ms" for mode in modes.values())
         + " | In stop paths |", "|---|---|" + "---|" * len(modes) + "---|"]
    for t, label in doc["term_labels"].items():
        used = [p for p in paths if t in paths[p]["terms"]]
        cells = []
        for m in modes:
            lo, _, hi = doc["reaction"][m][used[0]]["terms"][t]["low_ref_high_s"]
            cells.append(f"{_ms(lo)}–{_ms(hi)}")
        L.append(f"| {_cap(label)} | ({doc['term_kinds'][t]}) | " + " | ".join(cells) + " | "
                 + ", ".join(paths[p]["short"][0] for p in used) + " |")
    out["terms"] = L
    # 3. the reaction time of each path
    L = ["| Stop path | " + " | ".join(f"Reaction time, {mode['label']}: low–high, ms (d)" for mode in modes.values())
         + " | If Linux fails |", "|---|" + "---|" * len(modes) + "---|"]
    for p, path in paths.items():
        cells = []
        for m in modes:
            lo, _, hi = doc["reaction"][m][p]["total_low_ref_high_s"]
            cells.append(f"{_ms(lo)}–{_ms(hi)}")
        L.append(f"| {path['label']} | " + " | ".join(cells) + f" | {_cap(path['if_linux_fails'])} |")
    out["reaction"] = L
    # 4. the governor
    L = ["| Body | ToF mode | Stop path | " + " | ".join(
        f"{_cap(label)} [reading: {kinds[r]}]: governor, m/s (a, range from (d) timing)"
        for r, label in readings.items()) + " |", "|---|---|---|" + "---|" * len(readings)]
    for body in bodies:
        b = doc["bodies"][body]
        for m, mode in modes.items():
            for p, path in paths.items():
                L.append(f"| `{body}` | {mode['label']} | {path['short']} | "
                         + " | ".join(cell_text(b["governor"][m][p][r]) for r in readings) + " |")
    out["governor"] = L
    # 5. the distance ledger at the bar's speed
    L = [f"| Body | ToF mode | Stop path | d_see, m (a) | Travelled while reacting at {bar:.2f} m/s, m (a, from (d) "
         "timing) | Stop distance, m (a): " + "; ".join(_cap(label).lower() for label in readings.values())
         + " | Left over, worst–best, m (a, from (d) timing): two steps | Left over, worst–best, m (a, from (d) "
         "timing): spec stop limit, 1.5 strides |", "|---|---|---|---|---|---|---|---|"]
    for body in bodies:
        for m, mode in modes.items():
            for p, path in paths.items():
                g = doc["bodies"][body]["ledger_at_bar"][m][p]
                stops = "; ".join(f"{g['stop_m'][r]:.3f}" + (MARK_FOOT_LEAD if g["stop_set_by"][r] == "foot_lead"
                                                              else "") for r in readings)
                L.append(f"| `{body}` | {mode['label']} | {path['short']} | {_rng(*g['d_see_m'], 3)} | "
                         f"{_rng(*g['reaction_m'], 3)} | {stops} | {_rng(*g['spare_m']['two_steps'], 3)} | "
                         f"{_rng(*g['spare_m']['spec_strides'], 3)} |")
    out["ledger"] = L
    # 6. the ladder: distance left over in the worst case, two steps and the spec's 1.5 strides
    L = ["| Body | ToF mode | Stop path | " + " | ".join(
        f"Left over at {v:.2f} m/s, m (a, from (d) timing): two steps / 1.5 strides" for v in LADDER) + " |",
         "|---|---|---|" + "---|" * len(LADDER)]
    for body in bodies:
        for m, mode in modes.items():
            for p, path in paths.items():
                row = doc["bodies"][body]["ladder"][m][p]
                cells = []
                for v in LADDER:
                    w = row[f"{v:.2f}"]
                    cells.append(f"{w['worst_spare_m']['two_steps']:.3f} / {w['worst_spare_m']['spec_strides']:.3f}"
                                 + (f" {MARK_EXTRAPOLATED}" if "extrapolation" in w["stride_rule"] else ""))
                L.append(f"| `{body}` | {mode['label']} | {path['short']} | " + " | ".join(cells) + " |")
    out["ladder"] = L
    # 7. sensitivity: per body, mode, path and reading
    for body in bodies:
        for m in modes:
            for p in paths:
                for r, s in doc["bodies"][body]["sensitivity"][m][p].items():
                    head = f"`{body}`, {modes[m]['label']}, path {paths[p]['short']}, stop: {readings[r]}"
                    L = [f"| Term, one at a time ({head}) | Kind of the term | Range | Governor at the range's ends, "
                         "m/s (a) | Swing, m/s |", "|---|---|---|---|---|"]
                    for t in s["terms"]:
                        rng = (f"{t['range'][0]:.3f}–{t['range'][1]:.3f} m" if t["unit"] == "m"
                               else f"{_ms(t['range'][0])}–{_ms(t['range'][1])} ms")
                        L.append(f"| {_cap(t['label'])} | ({t['kind']}) | {rng} | {_rng(*t['governor_m_s'])} | "
                                 f"{t['swing_m_s']:.2f} |")
                    out[f"sensitivity_terms.{body}.{m}.{p}.{r}"] = L
                    L = [f"| What-if ({head}; reference {s['base']['governor_m_s']:.2f} m/s) | Kind of the reading | "
                         "Governor, m/s (a) | Change, m/s |", "|---|---|---|---|"]
                    for w in s["what_ifs"]:
                        L.append(f"| {_cap(w['label'])} | ({w['kind']}) | {w['governor_m_s']:.2f} | "
                                 f"{w['change_m_s']:+.2f} |")
                    out[f"sensitivity_what_ifs.{body}.{m}.{p}.{r}"] = L
    # 8. every adverse reading at once
    L = ["| Body | ToF mode | Stop path | d_see less the gap, m (a, with the (d) gap) | Reaction time, ms (d) | "
         + " | ".join(f"{_cap(label)}: governor, m/s (a, from a (d) stack)" for label in readings.values()) + " |",
         "|---|---|---|---|---|" + "---|" * len(readings)]
    for body in bodies:
        for m, mode in modes.items():
            for p, path in paths.items():
                s = doc["bodies"][body]["stress"][m][p]
                cells = [f"{g['governor_m_s']:.2f}" + (f" {MARK_BELOW_NOMINAL}" if g["below_nominal_walk"]
                                                       else f" {MARK_BELOW_BAR}" if g["below_bar"] else "")
                         for g in s["governor"].values()]
                L.append(f"| `{body}` | {mode['label']} | {path['short']} | {s['d_see_m']:.3f} | "
                         f"{_ms(s['reaction_s'])} | " + " | ".join(cells) + " |")
    out["stress"] = L
    # 9. inputs
    L = ["| Input | Kind | Value | Unit | Source; for (d), whose and how derived |", "|---|---|---|---|---|"]
    for i in doc["inputs"]:
        if "value" in i:
            val = json.dumps(i["value"]) if isinstance(i["value"], (dict, list)) else f"{i['value']}"
        else:
            val = " / ".join(f"{k} {_num(i[k])}" for k in ("low", "ref", "high") if k in i)
        src = i["source"] if not i["basis"] else (i["basis"] if i["source"] in ("none", "derived")
                                                    else f"{i['source']}. {i['basis']}")
        L.append(f"| `{i['id']}` | ({i['kind']}) | {val} | {i['unit']} | {src} |")
    out["inputs"] = L
    return out


def doc_bar(doc: dict) -> float:
    return next(i["value"] for i in doc["inputs"] if i["id"] == "bar_speed")


def markdown(doc: dict) -> str:
    return "\n\n".join(f"<!-- table: {name} -->\n" + "\n".join(lines) for name, lines in tables(doc).items()) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["md"]:
        sys.stdout.write(markdown(json.loads(OUT.read_text())))
        return 0
    if argv:
        raise SystemExit("usage: tof_governor.py [md]")
    doc = compute()
    OUT.write_text(dumps(doc))
    for body, b in doc["bodies"].items():
        print(f"{body}: lowest governor {b['lowest_governor_m_s']:.3f} m/s; fastest reference "
              f"{b['fastest_reference_m_s']:.3f} m/s; ladder fits in the worst case: {b['ladder_fits_worst_case']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
