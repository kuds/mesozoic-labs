"""Versioned measurement options for the physical gait audit.

``GaitProtocol`` holds every setting that shapes a *measured* value: contact
segmentation, swing validity, the continuous-phase construction and the gait
templates whose local tolerances define template coverage and persistence.
All of them are hashed into the measurement identity, so changing one forces
a fresh panel. Pass/fail bars that are compared against stored metrics live
in the separately declared gate (``curriculum/gait_gate.py``) and can be
re-judged from stored evidence without re-rolling a panel.

Every length is a fraction of the declared leg length ``L``, every force a
fraction of body weight, every timing tolerance a fraction of a stride or of
the analysis window. The few absolute times (chatter fill, blip) are below
any real swing or stance and were calibrated from 2 ms to 10 ms sampling.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MEASUREMENT_SCHEMA = "mesozoic.gait-measurement/v2"


@dataclass(frozen=True)
class GaitProtocol:
    """Metric-shaping detector settings; defaults are the calibrated v2 values."""

    # -- contact segmentation (per limb) --------------------------------------
    #: A foot is loaded when its floor normal force exceeds this fraction of
    #: body weight divided by the number of limbs.
    contact_force_bw_per_limb: float = 0.01
    #: Unloads no longer than this are sub-dwell chatter and stay stance.
    chatter_fill_s: float = 0.010
    #: Loads shorter than this (after merging) are blips, not stances.
    blip_s: float = 0.006
    #: An unload that never clears this height and moves the foot less than
    #: ``scuff_travel_over_leg`` (or is short, see below) is a scuff: stance.
    scuff_clearance_over_leg: float = 0.01
    scuff_travel_over_leg: float = 0.03
    #: An unload shorter than this fraction of the limb's reference swing
    #: (upper quartile of its clear swings) and lower than
    #: ``bounce_clearance_fraction`` of the reference clearance is an impact
    #: bounce: stance. The touchdown stays at the first contact.
    bounce_swing_fraction: float = 0.25
    bounce_clearance_fraction: float = 0.30
    #: A valid swing clears this height and repositions the foot this far.
    valid_swing_clearance_over_leg: float = 0.01
    valid_swing_travel_over_leg: float = 0.03
    #: Swing-phase floor contact: a swing sample is at the floor when the foot
    #: carries any floor force or is lower than this (a drag or a skim). It is
    #: the scuff height: a foot below it has not cleared the ground.
    swing_ground_clearance_over_leg: float = 0.01
    # -- continuous limb phase --------------------------------------------------
    #: A stride longer than this multiple of its local reference is a pause:
    #: phase is undefined (coverage loss) for its whole duration. The
    #: reference is the larger of the median cadences of up to
    #: ``pause_neighbour_strides`` strides of the same limb on either side
    #: (the stride itself excluded), so a cadence change is not a pause.
    pause_factor: float = 2.0
    pause_neighbour_strides: int = 3
    #: Centred sliding window for local phase statistics, in pooled strides.
    local_window_strides: float = 1.0
    #: Local mean resultant length below which a pair is locally unlocked.
    local_min_locking: float = 0.5
    # -- alternation template and the off-gait budget (cycles, strides) ---------
    #: A contralateral pair is on template when its local relative phase lies
    #: within this of anti-phase, anchored at touchdown or at mid-stance
    #: (template coverage, a diagnostic) ...
    template_alternation_tolerance: float = 0.15
    #: ... and off-gait when it is locally locked this much further out (a
    #: hop, a bound, a skip or a gallop, whatever the lag).
    off_gait_extra_tolerance: float = 0.05
    #: Off-gait time also counts unlocked (uncoordinated) stepping inside bouts
    #: that last at least this many pooled strides; shorter unlocked moments
    #: (a stumble, a double step) are not off-gait time.
    off_gait_bout_strides: float = 2.0
    # -- footprints, trunk frame and step-through -------------------------------
    #: A footprint is the load-weighted centre of a stance (force x time), so
    #: a light toe touch or a slide during stance does not move it.
    #: Step length is the footprint's advance past the contralateral foot's
    #: previous footprint along the trunk's own axis (the root body's x axis,
    #: from the recorded root quaternion, averaged over a centred window of
    #: this many pooled strides to remove the stride-periodic yaw wobble).
    trunk_axis_window_strides: float = 1.0
    #: A step whose length is below this does not step through: it lands
    #: behind the other foot's footprint. A stride (consecutive steps of the
    #: two feet of a pair) steps through when both of its steps do ...
    step_through_min_over_leg: float = 0.0
    #: ... and at least this many consecutive such steps of one foot are a
    #: step-to bout (the ``step_to_bout_fraction`` diagnostic; a genuine
    #: walker crabbing in its trunk frame has such bouts of its weaker foot,
    #: so they are never off-gait time).
    step_to_bout_steps: int = 2
    #: A stride whose footprint advances less than this along the local
    #: heading is a stride in place (marking time, tapping, freezing): its
    #: duration is off-gait time. Half the stride floor of the gate.
    in_place_stride_over_leg: float = 0.10
    #: Standing: the trunk's horizontal speed over a centred window of this
    #: many pooled strides is below ``standing_speed_fraction`` of the upper
    #: quartile of that speed over the analysis window. Standing time (a
    #: pause, a freeze, a standing start inside the window) is off-gait time.
    standing_window_strides: float = 0.1
    standing_speed_fraction: float = 0.05
    # -- local travel frame ------------------------------------------------------
    #: Stride length and skid travel are measured along the local travel
    #: heading: the trunk displacement over a centred window of this many
    #: pooled strides (a whole number of strides cancels the stride-periodic
    #: lateral sway; three also average out trunk heading wobble at other
    #: frequencies, and a centred chord still follows a steady turn). Where
    #: that displacement is shorter than ``heading_min_travel_over_leg`` the
    #: heading of the whole window is used, and the declared direction only
    #: when the trunk did not travel.
    heading_window_strides: float = 3.0
    heading_min_travel_over_leg: float = 0.05
    # -- stance quality and girdle participation -----------------------------------
    #: A stance is a glide when its foot's median slip speed exceeds this
    #: fraction of the trunk's speed during it (stance travel floored at
    #: 0.05 L; stances of at least three samples).
    glide_skid_ratio: float = 0.6
    #: Light stance: a stance sample whose foot carries less than this fraction
    #: of body weight divided by the number of limbs (merged unloads included).
    light_load_bw_per_limb: float = 0.1
    #: Time-local girdle participation (quadrupeds): the lighter girdle's share
    #: of the foot impulse over a centred one-stride window falls below this.
    girdle_local_min_share: float = 0.12
    # -- other measured quantities ----------------------------------------------
    #: Foot-on-foot contact when the force between feet exceeds this.
    foot_foot_force_bw: float = 0.01
    #: Biped hop-flight diagnostic: a flight's opener/closer search window.
    hop_flight_window_strides: float = 0.25

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"gait protocol option {name} must be a finite number")
            if value < 0.0:
                raise ValueError(f"gait protocol option {name} must be nonnegative")
        for name in (
            "contact_force_bw_per_limb",
            "local_window_strides",
            "foot_foot_force_bw",
            "hop_flight_window_strides",
            "template_alternation_tolerance",
            "trunk_axis_window_strides",
            "heading_window_strides",
            "standing_window_strides",
            "glide_skid_ratio",
        ):
            if values[name] <= 0.0:
                raise ValueError(f"gait protocol option {name} must be positive")
        if self.pause_factor <= 1.0:
            raise ValueError("gait protocol option pause_factor must exceed one stride")
        for name in ("pause_neighbour_strides", "step_to_bout_steps"):
            if isinstance(values[name], float) or values[name] < 1:
                raise ValueError(f"gait protocol option {name} must be a positive integer")
        if self.light_load_bw_per_limb < self.contact_force_bw_per_limb:
            raise ValueError("gait protocol option light_load_bw_per_limb must not be below the contact threshold")
        if self.girdle_local_min_share > 0.5:
            raise ValueError("gait protocol option girdle_local_min_share must lie in [0, 0.5]")
        for name in ("bounce_swing_fraction", "bounce_clearance_fraction", "local_min_locking"):
            if values[name] > 1.0:
                raise ValueError(f"gait protocol option {name} must lie in [0, 1]")
        if self.standing_speed_fraction >= 1.0:
            raise ValueError("gait protocol option standing_speed_fraction must be below one")
        if self.template_alternation_tolerance + self.off_gait_extra_tolerance >= 0.5:
            raise ValueError("gait protocol alternation tolerance plus the off-gait margin must stay below 0.5")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MEASUREMENT_SCHEMA,
            "default_status": "walk-first calibration on the 2026-10 development split; confirm on fresh panels",
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return measurement_protocol_sha256(self)


def measurement_protocol_sha256(protocol: GaitProtocol) -> str:
    """Hash the versioned options; caller also pins implementation/morphology."""
    payload = json.dumps(protocol.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LimbContacts:
    """Merged stance intervals of one limb, in sample indices ``[start, end)``.

    ``touchdowns`` are the first samples of stances that begin after the first
    sample (a trace edge is never an event). ``swing_*`` tuples are aligned
    with ``touchdowns``: the swing that ends at each touchdown.
    """

    stance: NDArray[np.bool_]
    stances: tuple[tuple[int, int], ...]
    touchdowns: tuple[int, ...]
    swing_clearance_m: tuple[float, ...]
    swing_travel_m: tuple[float, ...]
    swing_valid: tuple[bool, ...]
    merged_unloads: int
