"""Hardware selection, placement and structure sizing.

Components are placed from catalog dimensions (never scaled with the body).  Every placed item
records its *volume class*:

* A  volume required by the hardware itself (catalog body plus installation allowance);
* B  volume caused by the chosen arrangement (structure, brackets, linkages, routing reserves);
* C  cosmetic covering volume (shells), which can be redesigned freely.

Structure masses use stated beam rules (``size_beam``); they are estimates to be replaced by
CAD and weighing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import catalog
from .geometry import box_inertia, cylinder_inertia, frame_from_axis, rotate_inertia

G = 9.81
AL_DENSITY = 2700.0
AL_E = 69e9
AL_ALLOW = 70e6  # 6061-T6 yield ~276 MPa / ~4 for fatigue and impact
CF_DENSITY = 1550.0


@dataclass
class Item:
    """A ledger line with geometry."""

    name: str
    body: str
    category: str
    mass: float
    pos: np.ndarray  # body frame centre
    rot: np.ndarray  # body frame orientation of the item's local axes
    shape: str  # "box" | "cylinder" | "capsule" | "mesh" | "none"
    size: tuple  # box: full (L, W, H); cylinder: (diameter, length) along local z; capsule: (radius, half-length)
    volume_class: str
    basis: str  # sourced | derived | estimate
    source: str
    cost_nom: float = 0.0
    cost_high: float = 0.0
    power_nom: float = 0.0
    power_max: float = 0.0
    rgba: tuple = (0.2, 0.2, 0.22, 1.0)
    collide: bool = True
    visual: bool = True
    group: int = 2
    inertia: np.ndarray | None = None  # central inertia in body axes; computed if None
    meta: dict = field(default_factory=dict)

    def central_inertia(self) -> np.ndarray:
        if self.inertia is not None:
            return self.inertia
        if self.shape == "box":
            return rotate_inertia(box_inertia(self.mass, self.size), self.rot)
        if self.shape == "cylinder":
            return rotate_inertia(cylinder_inertia(self.mass, self.size[0] / 2, self.size[1], axis=2), self.rot)
        if self.shape == "capsule":
            r, hl = self.size
            return rotate_inertia(cylinder_inertia(self.mass, r, 2 * hl + 2 * r, axis=2), self.rot)
        return np.zeros((3, 3))


# ------------------------------------------------------------------------------------------------
# Hardware selection per size class
# ------------------------------------------------------------------------------------------------
JOINT_GROUPS = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle_a", "ankle_b")

DEFAULT_SELECTION = {
    # Starting points only; ``sizing.select`` replaces the actuator choices from computed loads and
    # ``power.select_battery`` replaces the battery from the computed walking power.
    60: dict(leg="myactuator_rmd_x4_p36_36_e_l", neck="robotis_xm540_w270_t", tail="robotis_xm430_w350_t", bus=24.0,
             camera="luxonis_oak_d_lite", lidar=None, tof="st_vl53l5cx_carrier",
             compute="nvidia_orin_nano_super_devkit", battery="lipo_6s_3300"),
    75: dict(leg="myactuator_rmd_x4_p36_36_e_l", neck="robotis_xm540_w270_t", tail="robotis_xm540_w150_t", bus=24.0,
             camera="luxonis_oak_d_lite", lidar=None, tof="st_vl53l5cx_carrier",
             compute="nvidia_orin_nano_super_devkit", battery="liion_6s2p_p45b"),
    100: dict(leg="robstride_03", neck="robotis_xm540_w270_t", tail="robotis_xm540_w270_t", bus=48.0,
              camera="luxonis_oak_d_pro_w", lidar="ldrobot_stl19p", tof="st_vl53l5cx_carrier",
              compute="nvidia_orin_nano_super_devkit", battery="liion_12s2p_p45b"),
    120: dict(leg="robstride_04", neck="robstride_03", tail="robotis_xm540_w270_t", bus=48.0,
              camera="luxonis_oak_d_pro_w", lidar="ldrobot_stl19p", tof="st_vl53l5cx_carrier",
              compute="nvidia_orin_nano_super_devkit", battery="liion_12s2p_p45b"),
}


def default_selection(height: int) -> dict:
    sel = dict(DEFAULT_SELECTION[height])
    sel["joints"] = {g: sel["leg"] for g in JOINT_GROUPS}
    return sel


# ------------------------------------------------------------------------------------------------
# Structure rules
# ------------------------------------------------------------------------------------------------
def size_beam(length: float, load_n: float, lever_frac: float, width: float, min_t: float = 0.003,
              defl_frac: float = 0.005, fittings: float = 1.35) -> tuple[float, float]:
    """Twin 6061 side plates of ``width`` carrying ``load_n`` at ``lever_frac*length``.

    Thickness satisfies both stress (sigma <= AL_ALLOW) and stiffness (tip deflection of a
    cantilever <= defl_frac*length); mass includes a fittings factor.  Returns (mass, t).
    """
    m_bend = load_n * lever_frac * length
    t_stress = 6 * m_bend / (2 * width**2 * AL_ALLOW)
    i_req = load_n * length**2 / (3 * AL_E * defl_frac)
    t_stiff = 12 * i_req / (2 * width**3)
    t = max(min_t, t_stress, t_stiff)
    return 2 * AL_DENSITY * width * t * length * fittings, t


def bearing_mass(motor_diameter: float) -> float:
    """Output-side support bearing set and housing per joint (estimate, scales with motor size)."""
    return 0.018 * (motor_diameter / 0.055) ** 2.2


# ------------------------------------------------------------------------------------------------
# Items helpers
# ------------------------------------------------------------------------------------------------
MOTOR_RGBA = (0.13, 0.14, 0.15, 1.0)
STRUCT_RGBA = (0.62, 0.64, 0.66, 1.0)
BOARD_RGBA = (0.10, 0.45, 0.32, 1.0)
BATT_RGBA = (0.80, 0.48, 0.12, 1.0)
SENSOR_RGBA = (0.15, 0.20, 0.55, 1.0)
RESERVE_RGBA = (0.9, 0.3, 0.3, 0.25)


def motor_item(name: str, body: str, act: catalog.Actuator, pos, axis, category="leg_actuator") -> Item:
    d, ln = act.envelope
    if act.box is not None:
        size = tuple(float(v) + 0.004 for v in act.box)
        rot = frame_from_axis(axis)
        shape = "box"
    else:
        size = (d, ln)
        rot = frame_from_axis(axis)
        shape = "cylinder"
    basis = "sourced" if act.basis.get("mass") == "sourced" else act.basis.get("mass", "estimate")
    return Item(name, body, category, act.mass, np.asarray(pos, float), rot, shape, size, "A", basis, act.id,
                cost_nom=act.price, cost_high=act.price * 1.08, rgba=MOTOR_RGBA,
                meta=dict(actuator=act.id, axis=list(np.asarray(axis, float))))


def box_item(name, body, category, mass, pos, size, vclass, basis, source, rgba, rot=None, collide=True, **kw) -> Item:
    return Item(name, body, category, mass, np.asarray(pos, float), np.eye(3) if rot is None else rot, "box",
                tuple(size), vclass, basis, source, rgba=rgba, collide=collide, **kw)


def rod_item(name, body, p0, p1, radius, mass, vclass="B", basis="estimate", source="push rod rule",
             rgba=STRUCT_RGBA, category="linkage") -> Item:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = 0.5 * (p0 + p1)
    ax = p1 - p0
    hl = 0.5 * np.linalg.norm(ax)
    return Item(name, body, category, mass, c, frame_from_axis(ax), "capsule", (radius, hl), vclass, basis, source,
                rgba=rgba, meta=dict(fromto=list(p0) + list(p1)))


def plate_link_items(name, body, p0, p1, width, t, mass, lateral, rgba=STRUCT_RGBA, source="twin-plate beam rule"):
    """Two side plates between p0 and p1, offset +-lateral in y (body frame)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    L = np.linalg.norm(ax)
    rot = frame_from_axis(ax, up=(0, 1, 0))
    # local z along link; local x/y span width/thickness
    items = []
    for k, s in enumerate((+1, -1)):
        c = 0.5 * (p0 + p1) + np.array([0, s * lateral, 0])
        r = np.column_stack([rot[:, 0], rot[:, 1], rot[:, 2]])
        items.append(Item(f"{name}_plate{k}", body, "structure", mass / 2, c, r, "box", (width, t, L), "B", "estimate",
                          source, rgba=rgba))
    return items
