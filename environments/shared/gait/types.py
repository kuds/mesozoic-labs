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
    #: carries any floor force or is lower than this (a drag or a skim).
    swing_ground_clearance_over_leg: float = 0.005
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
    # -- gait templates (cycles) shaping coverage and persistence ---------------
    template_alternation_tolerance: float = 0.09
    template_synchrony_tolerance: float = 0.125
    template_walk_band_low: float = 0.125
    template_walk_band_high: float = 0.375
    #: A sample is on-template when its local mean lies inside the target set
    #: widened by this much ...
    template_local_extra_tolerance: float = 0.05
    #: ... and grossly off-template when it lies this much further out, or
    #: the pair is unlocked or a limb's phase is undefined.
    template_gross_extra_tolerance: float = 0.05
    #: Off-gait time counts undefined phase and time locked inside a competing
    #: gait's template, plus uncoordinated (unlocked) stepping only inside
    #: bouts that last at least this many pooled strides.
    off_gait_bout_strides: float = 2.0
    # -- local travel frame ------------------------------------------------------
    #: Fore-aft quantities (lead exchange, step and stride length, skid travel)
    #: are measured along the local travel heading: the trunk displacement over
    #: a centred window of this many pooled strides (one stride cancels the
    #: stride-periodic lateral sway). Where that displacement is shorter than
    #: ``heading_min_travel_over_leg`` the heading of the whole window is used,
    #: and the declared direction only when the trunk did not travel at all.
    heading_window_strides: float = 1.0
    heading_min_travel_over_leg: float = 0.05
    # -- other measured quantities ----------------------------------------------
    #: Lead-limb exchange: the fore-aft order of a contralateral pair must
    #: flip past +-this distance within a stride.
    lead_hysteresis_over_leg: float = 0.01
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
            "lead_hysteresis_over_leg",
            "foot_foot_force_bw",
            "hop_flight_window_strides",
            "template_alternation_tolerance",
            "template_synchrony_tolerance",
        ):
            if values[name] <= 0.0:
                raise ValueError(f"gait protocol option {name} must be positive")
        if self.pause_factor <= 1.0:
            raise ValueError("gait protocol option pause_factor must exceed one stride")
        if isinstance(self.pause_neighbour_strides, float) or self.pause_neighbour_strides < 1:
            raise ValueError("gait protocol option pause_neighbour_strides must be a positive integer")
        if self.heading_window_strides <= 0.0:
            raise ValueError("gait protocol option heading_window_strides must be positive")
        for name in ("bounce_swing_fraction", "bounce_clearance_fraction", "local_min_locking"):
            if values[name] > 1.0:
                raise ValueError(f"gait protocol option {name} must lie in [0, 1]")
        if not 0.0 < self.template_walk_band_low < self.template_walk_band_high < 0.5:
            raise ValueError("gait protocol walk limb-phase band requires 0 < low < high < 0.5")
        widest = (
            max(
                self.template_alternation_tolerance,
                self.template_synchrony_tolerance,
                0.5 * (self.template_walk_band_high - self.template_walk_band_low),
            )
            + self.template_local_extra_tolerance
            + self.template_gross_extra_tolerance
        )
        if widest >= 0.5:
            raise ValueError("gait protocol template tolerances plus local/gross widening must stay below 0.5")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MEASUREMENT_SCHEMA,
            "default_status": "calibrated on the 2026-10 development split; confirm on fresh panels",
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
