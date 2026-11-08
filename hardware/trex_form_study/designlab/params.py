"""Candidate parameters.

A candidate is identified as ``<height>-<width>-<family>-<posture>[-<proportion>]``, e.g.
``100-M-S-10``.  Height classes are design-study points (target crown height in the
reference posture), not separate robot programmes.

Definitions used throughout the study (also in ``DEFINITIONS.md``):

* Crown height: floor to the highest point of the skull covering in the nominal standing pose.
* Hip height ``H``: floor to the hip-pitch axis (acetabulum analogue) in the nominal pose.
* Torso inclination: angle above horizontal of the line from the hip-pitch axis midpoint
  (acetabulum analogue) to the shoulder landmark (glenoid analogue).  Positive is chest-up.
* Pelvis inclination: angle above horizontal of the pelvis/sacral reference axis, which is
  parallel to the torso axis in the level posture.  The difference between the two is a fixed
  structural lumbar bend, not a joint.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Target crown height (m) in the reference posture for each height class.
HEIGHT_CLASSES = {60: 0.600, 75: 0.750, 100: 1.000, 120: 1.200}
#: Study heights (size study, round two): crown targets between and around the classes.  Each takes every
#: per-class rule (wall thickness scale, cover supports and fasteners, starting parts, mass guess, ankle-drive
#: station, cover material, control-stack allowance) from ``SIZE_CLASS``, so only the size changes.
STUDY_HEIGHTS = {54: 0.540, 57: 0.570, 80: 0.800}
SIZE_CLASS = {54: 60, 57: 60, 80: 75}


def size_class(height: int) -> int:
    """The height class whose per-class rules a height uses (itself for a class)."""
    return height if height in HEIGHT_CLASSES else SIZE_CLASS[height]


def crown_target(height: int) -> float:
    return HEIGHT_CLASSES[height] if height in HEIGHT_CLASSES else STUDY_HEIGHTS[height]

#: The posture at which the crown height is matched; other postures keep the same body and
#: report the crown height they reach (controlled comparison, see params docstring).
REFERENCE_POSTURE = 10

#: Postures: torso (chest) inclination and pelvis inclination in degrees, chest-up positive.
#: The pelvis follows half the chest rotation; the rest is a fixed lumbar bend.
POSTURES = {
    0: dict(chest_deg=0.0, pelvis_deg=0.0, neck_rise_deg=0.0, label="Level torso"),
    10: dict(chest_deg=10.0, pelvis_deg=5.0, neck_rise_deg=0.0, label="10° chest-up"),
    20: dict(chest_deg=20.0, pelvis_deg=10.0, neck_rise_deg=0.0, label="20° chest-up"),
}

#: Width variants.  Each scales the *designed* body half-widths and sets the hip module
#: geometry; hardware minima (component envelopes plus clearances) always win.
#:   torso_scale   multiplies the family's designed trunk half-width profile
#:   hip_extra     extra lateral offset of the leg plane beyond the hardware minimum, in units of H
#:   foot_ratio    nominal foot-centre spacing / hip-pitch spacing (<1 means feet drawn inward)
WIDTHS = {
    "N": dict(torso_scale=0.86, hip_extra=0.000, roll_extra=0.000, foot_ratio=0.80, label="Narrow"),
    "M": dict(torso_scale=1.00, hip_extra=0.020, roll_extra=0.010, foot_ratio=0.90, label="Balanced"),
    "W": dict(torso_scale=1.16, hip_extra=0.055, roll_extra=0.030, foot_ratio=0.95, label="Broad"),
}

#: Proportion variants that change length/leg/skull relationships.  ``base`` is the study's
#: recommended engineering adaptation of the anatomical template; the others are explicit
#: alternatives so their effect can be seen in isolation.
PROPORTIONS = {
    "base": dict(tail_scale=1.00, leg_scale=1.00, skull_scale=1.00, neck_scale=1.00, label="Study default"),
    "anat": dict(tail_scale=1.48, leg_scale=1.00, skull_scale=1.00, neck_scale=1.00, label="Anatomical tail length"),
    "compact": dict(tail_scale=0.72, leg_scale=0.90, skull_scale=1.05, neck_scale=0.90, label="Compact (short tail, shorter legs)"),
    "bighead": dict(tail_scale=1.00, leg_scale=1.00, skull_scale=1.15, neck_scale=1.00, label="Large skull"),
}

FAMILIES = {
    "T": "Tapered, lightly covered",
    "S": "Substantial, anatomically balanced",
    "H": "Hybrid, selective covers with exposed structure",
}


@dataclass(frozen=True)
class Candidate:
    height: int  # height class key in HEIGHT_CLASSES
    width: str  # key in WIDTHS
    family: str  # key in FAMILIES
    posture: int  # key in POSTURES
    proportion: str = "base"  # key in PROPORTIONS
    #: Leg actuation layout.  "tandem": proximal actuation (hip-pitch and knee motors in a thigh
    #: pod, ankle motors high on the shank driving push rods) with both hip-roll motors on the body
    #: centreline, one ahead of and one behind the hip axis.  "side": the same legs with the two
    #: roll motors side by side.  "serial": a motor at every joint, as in PR #662.
    leg_layout: str = "tandem"
    #: Battery cassette position: "tailbase" (fixed tail base behind the hip, dorsal hatch; restores
    #: the animal's tail-base mass and moves the centre of mass toward the hip) or "belly" (low in the
    #: abdomen ahead of the pubic boot, as in PR #662).
    battery_pos: str = "tailbase"
    #: Free-form overrides applied after the family/size defaults (used by round-two refinements).
    overrides: tuple = field(default=())

    @property
    def id(self) -> str:
        tag = f"{self.height}-{self.width}-{self.family}-{self.posture:02d}"
        if self.proportion != "base":
            tag += f"-{self.proportion}"
        if self.leg_layout != "tandem":
            tag += f"-{self.leg_layout}"
        if self.battery_pos != "tailbase":
            tag += f"-{self.battery_pos}"
        return tag

    @property
    def crown_target(self) -> float:
        return crown_target(self.height)

    def at_posture(self, posture: int) -> "Candidate":
        return replace(self, posture=posture)

    def override(self, key: str, default=None):
        for k, v in self.overrides:
            if k == key:
                return v
        return default


def parse(cid: str) -> Candidate:
    parts = cid.split("-")
    height, width, family, posture = int(parts[0]), parts[1], parts[2], int(parts[3])
    proportion, layout, bat = "base", "tandem", "tailbase"
    for extra in parts[4:]:
        if extra in PROPORTIONS:
            proportion = extra
        elif extra in ("serial", "side", "tandem"):
            layout = extra
        elif extra in ("belly", "tailbase"):
            bat = extra
        else:
            raise ValueError(f"unknown candidate suffix {extra!r} in {cid}")
    return Candidate(height, width, family, posture, proportion, layout, bat)


def full_matrix() -> list[Candidate]:
    """Stage A: every height x width x family x posture at the default proportions."""
    return [
        Candidate(h, w, f, p)
        for h in HEIGHT_CLASSES
        for w in WIDTHS
        for f in FAMILIES
        for p in POSTURES
    ]
