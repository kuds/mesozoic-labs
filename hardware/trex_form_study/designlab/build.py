"""Build a candidate: skeleton -> hardware -> structure -> coverings (bodies with items/panels).

``mjcf.write_mjcf`` turns a built ``Design`` into a MuJoCo model; ``pipeline`` adds the stance
keyframe and ledgers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import catalog, shapes
from . import hardware as hw
from . import template as T
from .geometry import frame_from_axis, ry, rz
from .params import REFERENCE_POSTURE, WIDTHS, Candidate
from .skeleton import Skeleton, build_skeleton

#: Camera mount: station along the skull (fraction of skull length from the occipital condyle;
#: the orbits are at 0.20 and the lacrimal horn at 0.27) and pitch above level, in the world.
CAMERA_STATION = 0.24
CAMERA_PITCH_DEG = 5.0
#: Lidar station along the trunk (fraction of B ahead of the hip axis).
LIDAR_STATION = 0.92
#: Revision 2 (round two): the camera box is placed by a search over skull stations and depths so that it
#: sits inside the skull cover (inset by half the wall and ``CAMERA_CLEAR_M``) and clear of the eyes; the
#: highest fitting position wins (round one put it 4 mm under the roof at station 0.24, where the cover
#: cut the box by 2.3-13.7 mm).
CAMERA_STATIONS_R2 = np.arange(0.08, 0.305, 0.005)
CAMERA_CLEAR_M = 0.002
#: Upper ankle motor station along the tibia (fraction of its length below the knee).  Revision 2 value
#: chosen from a knee-flexion sweep at 60 cm (decision log F9).
ANKLE_MOTOR_FRAC = {1: 0.30, 2: 0.30}
ANKLE_MOTOR_FRAC_R2_BY_HEIGHT = {60: 0.46}  # 60 cm: knee usable 0.80-0.86 rad (0.62-0.66 at 0.30)
#: Revision 2: clearance between a pelvis side plate and the inner face of the upper-thigh shell.  The
#: round-one hip spacing ignored the plate thickness, so narrow stances (no extra spacing) put the plates
#: 1.4-2.8 mm into the shells at 60-75 cm and 5 mm at 100 cm (``100-N-S-10``).  3 mm still left a 1.3 mm
#: contact at standing (the stance hip roll and the shell's convex pieces bring it about 4 mm further in);
#: 6 mm clears every 60-75 cm narrow body at standing (decision log F9).
PELVIS_THIGH_CLEAR_M = 0.006
#: Revision 2: minimum vertical gap between the tail motor and the tail-base battery cassette where they
#: overlap in plan (round one left -0.2 mm on 60-M-T-10).
TAIL_MOTOR_CLEAR_M = 0.003

G = 9.81

FAMILY_COLORS = {
    # placeholder finishes for review renders (jaw a lighter skin tone, not bone, so the head reads as
    # one deep skull rather than a beak)
    "T": dict(main=(0.38, 0.45, 0.37, 1), accent=(0.55, 0.57, 0.45, 1), jaw=(0.50, 0.53, 0.43, 1),
              limb=(0.31, 0.38, 0.31, 1)),
    "S": dict(main=(0.34, 0.41, 0.33, 1), accent=(0.50, 0.52, 0.41, 1), jaw=(0.47, 0.50, 0.40, 1),
              limb=(0.28, 0.34, 0.27, 1)),
    "H": dict(main=(0.36, 0.41, 0.42, 1), accent=(0.58, 0.55, 0.44, 1), jaw=(0.50, 0.52, 0.48, 1),
              limb=(0.30, 0.35, 0.36, 1)),
}
TEETH_RGBA = (0.93, 0.91, 0.84, 1)

COVER_MATERIALS = {
    # density kg/m^3; finished-part cost USD/kg (outsourced MJF incl. finishing) - estimate, see COSTS.md
    "PA12_MJF": dict(density=1010.0, cost_per_kg=180.0, basis="estimate"),
    "PA12_MJF_large": dict(density=1010.0, cost_per_kg=150.0, basis="estimate"),
    "AL_6061": dict(density=2700.0, cost_per_kg=60.0, basis="estimate"),
}

MASS_GUESS = {60: 11.0, 75: 16.0, 100: 28.0, 120: 42.0}


def cover_material(height: int) -> str:
    return "PA12_MJF" if height <= 75 else "PA12_MJF_large"


def cover_thickness(height: int, family_mm: float) -> float:
    """Wall grows with panel span: 1.6-1.8 mm at 60 cm to ~2.6-2.9 mm at 120 cm (estimate)."""
    scale = {60: 1.0, 75: 1.12, 100: 1.40, 120: 1.60}[height]
    return family_mm * scale / 1000.0


@dataclass
class Body:
    name: str
    parent: str | None
    pos_world: np.ndarray
    R_world: np.ndarray
    joint: dict | None = None
    items: list = field(default_factory=list)
    panels: list = field(default_factory=list)
    sites: list = field(default_factory=list)
    support: np.ndarray | None = None


class Design:
    """Everything about one built candidate."""

    def __init__(self, cand: Candidate, B: float, selection: dict | None = None):
        self.cand = cand
        self.B = B
        #: generator revision (1: round one, 2: round two) and the matching family shape key
        self.rev = int(cand.override("rev", 1))
        self.fk = T.family_key(cand.family, self.rev)
        self.sel = selection or hw.default_selection(cand.height)
        self.notes: list[str] = []
        self.geometry_checks: dict = {}
        self.sk: Skeleton = build_skeleton(cand, B)
        if cand.battery_pos == "tailbase":
            # The tail-base cassette sits between the rear hip-roll motor and the tail pivot.  If it does
            # not fit, the tail pivot moves back (never forward): a packaging setback, recorded.
            x_lim = self._tailbase_root_limit()
            if x_lim < self.sk.tail_root[0] - 1e-6:
                setback = float(self.sk.tail_root[0] - x_lim)
                self.sk = build_skeleton(replace(cand, overrides=tuple(cand.overrides) + (("tail_root_x_m", x_lim),)), B)
                self.geometry_checks["tail_root_setback_m"] = setback
                self.notes.append(f"tail pivot moved back {1000 * setback:.0f} mm to fit the tail-base battery cassette")
        self.bodies: dict[str, Body] = {}
        self.mass_guess = self.sel.get("mass_guess", MASS_GUESS[cand.height])
        self._build()

    # ------------------------------------------------------------------ helpers
    def _tailbase_root_limit(self) -> float:
        """Most forward tail-pivot x (trunk frame) that leaves the tail-base cassette 10 mm clear."""
        bat = catalog.parts()[self.sel["battery"]]
        cas_x = bat.dims[0] + 0.020
        roll = self.act("hip_roll")
        x_front = -(roll.envelope[1] + 0.012) if self.cand.leg_layout == "tandem" else -0.02 * self.B
        return x_front - cas_x - 0.010

    def _camera_fit(self, cam_size) -> dict:
        """Revision 2: highest stereo-camera position inside the skull cover and clear of the eyes."""
        sk = self.sk
        L = sk.skull_len
        fam = T.FAMILY_SHAPE[self.fk]
        inset = 0.5 * cover_thickness(self.cand.height, fam["shell_mm"]) + CAMERA_CLEAR_M
        ax, up = shapes._skull_frame(sk)
        Rc = ry(np.radians(CAMERA_PITCH_DEG))
        half = 0.5 * np.asarray(cam_size)
        signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        eyes = shapes.eye_boxes(sk, self.fk)
        best = None
        for st in CAMERA_STATIONS_R2:
            top_w, _, _ = shapes.skull_dorsal(sk, self.fk, float(st))
            for drop in np.arange(0.004, 0.0605, 0.0005):
                c = top_w - up * (half[2] + drop) - ax * half[0]
                corners = c + (Rc @ (signs * half).T).T
                if not all(shapes.skull_inside(sk, self.fk, q, inset) for q in corners):
                    continue
                lo, hi = corners.min(0), corners.max(0)
                if any(np.all(lo < e_hi + 0.002) and np.all(e_lo - 0.002 < hi) for e_lo, e_hi in eyes):
                    continue
                if best is None or c[2] > best["z"] + 1e-9:
                    best = dict(z=float(c[2]), station=float(st), drop_m=float(drop))
                break  # the first fitting drop is the highest at this station
        if best is None:
            return dict(fits=False, station=CAMERA_STATION, drop_m=0.004, skull_length_m=float(L))
        return dict(fits=True, station=best["station"], drop_m=best["drop_m"], centre_z_m=best["z"],
                    skull_length_m=float(L), inset_m=float(inset))

    def _tail_motor_lift(self, tail, tr, h_ax) -> float:
        """Revision 2: raise the tail motor so it clears the cassette by TAIL_MOTOR_CLEAR_M where they overlap
        in plan (the motor stays on the yaw axis)."""
        b = getattr(self, "battery", None)
        if b is None:
            return 0.0
        cen, size, R = b["center"], np.asarray(b["size"]), b["R"]
        cas = cen + (R @ (np.array([[a, c, d] for a in (-1, 1) for c in (-1, 1) for d in (-1, 1)]) * 0.5 * size).T).T
        if tail.box is not None:
            hx = 0.5 * max(tail.box[0], tail.box[1])
        else:
            hx = 0.5 * tail.envelope[0]
        if cas[:, 0].max() < tr[0] - hx or cas[:, 0].min() > tr[0] + hx:
            return 0.0
        gap = (tr[2] + 0.006) - cas[:, 2].max()
        return float(max(0.0, TAIL_MOTOR_CLEAR_M - gap))

    def act(self, role: str) -> catalog.Actuator:
        if role in hw.JOINT_GROUPS:
            return catalog.actuator(self.sel["joints"][role])
        return catalog.actuator(self.sel[role])

    def add_body(self, name, parent, pos_world, R_world=None, joint=None):
        b = Body(name, parent, np.asarray(pos_world, float), np.eye(3) if R_world is None else R_world, joint)
        self.bodies[name] = b
        return b

    def to_body(self, body: str, p_world):
        b = self.bodies[body]
        return (b.R_world.T @ (np.asarray(p_world, float).T - b.pos_world[:, None])).T if np.ndim(p_world) > 1 \
            else b.R_world.T @ (np.asarray(p_world, float) - b.pos_world)

    def add(self, item: hw.Item):
        self.bodies[item.body].items.append(item)
        return item

    def at(self, x):
        return self.design_profile.at(x)

    # ------------------------------------------------------------------ build
    def _build(self):
        c, sk = self.cand, self.sk
        B, H = sk.B, sk.H
        wv = WIDTHS[c.width]
        self.design_profile = shapes.trunk_profile(self.fk, B, wv["torso_scale"])

        roll, yaw, pitch, knee = (self.act(r) for r in ("hip_roll", "hip_yaw", "hip_pitch", "knee"))
        d_roll, l_roll = roll.envelope
        d_yaw, l_yaw = yaw.envelope
        d_pitch, l_pitch = pitch.envelope
        gap, wall = 0.004, 0.0035
        layout = c.leg_layout
        # upper thigh lateral half-thickness: hip-pitch motor (axis y) and thigh-twist motor (axis along femur)
        half_upper = max(l_pitch / 2, d_yaw / 2) + wall + gap
        if layout == "tandem":
            y_r = 0.0
            pelvis_half = d_roll / 2 + wall + 0.003
        elif layout == "side":
            y_r = d_roll / 2 + gap + wv["roll_extra"] * B
            pelvis_half = y_r + d_roll / 2 + wall + 0.003
        elif layout == "serial":
            pelvis_half = d_roll / 2 + wall + 0.003 + 0.02
            y_r = None
        else:
            raise ValueError(layout)
        y_leg = pelvis_half + gap + half_upper + wv["hip_extra"] * B
        if self.rev >= 2 and layout in ("tandem", "side"):
            # the pelvis side plate (its beam-rule thickness) and the thigh shell wall must both fit
            _, t_pel = hw.size_beam(0.55 * B, 2.0 * self.mass_guess * G, 0.25, max(0.05, 1.2 * d_roll))
            t_sh = cover_thickness(c.height, T.FAMILY_SHAPE[self.fk]["shell_mm"])
            need = (pelvis_half + 0.5 * max(0.003, t_pel) + 0.5 * t_sh + PELVIS_THIGH_CLEAR_M + half_upper
                    + wv["hip_extra"] * B)
            if need > y_leg:
                self.geometry_checks["hip_spacing_added_for_pelvis_plate_m"] = float(need - y_leg)
                y_leg = need
            add = float(c.override("hip_spacing_add_m", 0.0))  # round two: narrowest walkable stance search
            if add:
                self.geometry_checks["hip_spacing_added_for_walking_m"] = add
                y_leg += add
        if y_r is None:  # PR #662-like: roll axis ~32 mm (at 360 mm hip) inboard of the leg plane
            y_r = y_leg - 0.032 * (H / 0.36)
        z_r = 0.02 * B
        self.hip = dict(layout=layout, y_roll=y_r, z_roll=z_r, y_leg=y_leg, pelvis_half=pelvis_half,
                        half_upper=half_upper, foot_spacing=2 * y_leg * wv["foot_ratio"])

        Rp = sk.R_p
        self.add_body("trunk", None, sk.origin, Rp, joint=dict(type="free", name="root"))
        self.add_body("head", "trunk", sk.to_world(sk.neck_base), np.eye(3),
                      joint=dict(type="hinge", name="neck_pitch", axis=(0, -1, 0), range=(-0.35, 0.35)))
        self.add_body("tail", "trunk", sk.to_world(sk.tail_root), np.eye(3),
                      joint=dict(type="hinge", name="tail_yaw", axis=tuple(Rp[:, 2]), range=(-0.45, 0.45)))
        leg = sk.leg
        ax_f = (leg["knee"] - leg["hip"]) / np.linalg.norm(leg["knee"] - leg["hip"])
        self.femur_axis = ax_f
        for side, s in (("left", 1.0), ("right", -1.0)):
            hip_w = np.array([0.0, s * y_leg, H])
            off = np.array([0.0, s * y_leg, 0.0])
            # chain: hip roll (pelvis) -> hip pitch -> thigh twist (hip yaw about the femur axis) -> knee -> ankle
            self.add_body(f"{side}_roll", "trunk", np.array([0.0, s * y_r, H + z_r]), np.eye(3),
                          joint=dict(type="hinge", name=f"{side}_hip_roll", axis=(1, 0, 0), range=(-0.45, 0.45)))
            self.add_body(f"{side}_thigh", f"{side}_roll", hip_w, np.eye(3),
                          joint=dict(type="hinge", name=f"{side}_hip_pitch", axis=(0, 1, 0), range=(-1.0, 1.0)))
            self.add_body(f"{side}_femur", f"{side}_thigh", hip_w, np.eye(3),
                          joint=dict(type="hinge", name=f"{side}_hip_yaw", axis=tuple(ax_f), range=(-0.30, 0.30)))
            self.add_body(f"{side}_shank", f"{side}_femur", leg["knee"] + off, np.eye(3),
                          joint=dict(type="hinge", name=f"{side}_knee", axis=(0, 1, 0), range=(-1.1, 1.1)))
            self.add_body(f"{side}_ankle", f"{side}_shank", leg["mtp"] + off, np.eye(3),
                          joint=dict(type="hinge", name=f"{side}_ankle_pitch", axis=(0, 1, 0), range=(-0.8, 0.8)))
            self.add_body(f"{side}_foot", f"{side}_ankle", leg["mtp"] + off, np.eye(3),
                          joint=dict(type="hinge", name=f"{side}_ankle_roll", axis=(1, 0, 0), range=(-0.4, 0.4)))
            del hip_w
        self._place_legs()
        self._place_trunk_hardware()
        self._place_head_tail_hardware()
        self._place_structure()
        self._place_covers()
        self._place_lidar()

    # ------------------------------------------------------------------ legs
    def _place_legs(self):
        c, sk = self.cand, self.sk
        H = sk.H
        leg = sk.leg
        y_leg, y_r, z_r = self.hip["y_leg"], self.hip["y_roll"], self.hip["z_roll"]
        roll, yaw, pitch, knee = (self.act(r) for r in ("hip_roll", "hip_yaw", "hip_pitch", "knee"))
        ank_a, ank_b = self.act("ankle_a"), self.act("ankle_b")
        load = 2.0 * self.mass_guess * G  # design load on one leg: 2 x body weight
        layout = c.leg_layout
        d_roll, l_roll = roll.envelope
        d_pitch, l_pitch = pitch.envelope
        d_yaw, l_yaw = yaw.envelope
        d_knee, l_knee = knee.envelope
        ax_f = self.femur_axis
        for side, s in (("left", 1.0), ("right", -1.0)):
            hip_w = np.array([0.0, s * y_leg, H])
            knee_w = leg["knee"] + np.array([0, s * y_leg, 0])
            hock_w = leg["hock"] + np.array([0, s * y_leg, 0])
            mtp_w = leg["mtp"] + np.array([0, s * y_leg, 0])
            femur_len = np.linalg.norm(knee_w - hip_w)
            # --- hip roll motor (fixed in the trunk, axis x)
            xr = (l_roll / 2 + 0.008) * s if layout == "tandem" else -(l_roll / 2 + 0.006)
            p = np.array([xr, s * y_r, H + z_r])
            self.add(hw.motor_item(f"{side}_hip_roll_motor", "trunk", roll, self.to_body("trunk", p),
                                   self.bodies["trunk"].R_world.T @ np.array([1.0, 0, 0])))
            # --- roll output bracket and hip-pitch motor case (roll body)
            br_len = abs(y_leg - y_r) - 0.5 * l_pitch + 0.02
            br_w = max(0.025, 0.55 * d_pitch)
            m_br, t_br = hw.size_beam(max(br_len, 0.02), load, 0.6, br_w)
            # tandem: the bracket leaves the roll output on the centreline, so it sits on that roll motor's
            # side of the hip axis (left ahead, right behind) and the two brackets cannot meet
            x_br = np.sign(xr) * (0.5 * br_w + 0.002) if layout == "tandem" else 0.0
            self.add(hw.box_item(f"{side}_hip_bracket", f"{side}_roll", "structure", m_br,
                                 self.to_body(f"{side}_roll", np.array([x_br, s * (0.5 * (y_leg - 0.5 * l_pitch + y_r)), H])),
                                 (br_w, max(br_len, 0.02), max(0.006, 2 * t_br)), "B", "estimate", "twin-plate beam rule",
                                 hw.STRUCT_RGBA))
            self.add(hw.motor_item(f"{side}_hip_pitch_motor", f"{side}_roll", pitch, self.to_body(f"{side}_roll", hip_w),
                                   (0, 1, 0)))
            self.add(hw.Item(f"{side}_hip_bearings", f"{side}_roll", "bearings", hw.bearing_mass(d_pitch),
                             self.to_body(f"{side}_roll", hip_w), np.eye(3), "none", (0,), "B", "estimate", "bearing rule",
                             visual=False, collide=False))
            # --- thigh-twist (hip yaw) motor case on the thigh body, axis along the femur
            py = hip_w + ax_f * (0.5 * d_pitch + 0.5 * l_yaw + 0.006)
            self.add(hw.motor_item(f"{side}_hip_yaw_motor", f"{side}_thigh", yaw, self.to_body(f"{side}_thigh", py), ax_f))
            m_tb, _ = hw.size_beam(0.5 * d_pitch + l_yaw, load, 0.4, max(0.025, 0.6 * d_pitch))
            self.add(hw.box_item(f"{side}_thigh_bracket", f"{side}_thigh", "structure", m_tb,
                                 self.to_body(f"{side}_thigh", hip_w + ax_f * (0.5 * d_pitch + 0.25 * l_yaw)
                                              + np.array([0, -s * (0.5 * l_pitch + 0.004), 0])),
                                 (0.7 * d_pitch, 0.006, 0.5 * d_pitch + l_yaw), "B", "estimate", "bracket rule",
                                 hw.STRUCT_RGBA, rot=_frame_z(ax_f)))
            self.add(hw.Item(f"{side}_twist_bearings", f"{side}_thigh", "bearings", hw.bearing_mass(d_yaw),
                             self.to_body(f"{side}_thigh", py), np.eye(3), "none", (0,), "B", "estimate", "bearing rule",
                             visual=False, collide=False))
            # --- femur structure and knee drive (femur body, below the twist output)
            f0 = hip_w + ax_f * (0.5 * d_pitch + l_yaw + 0.008)
            width = max(0.022, 0.5 * d_knee)
            m_f, t_f = hw.size_beam(np.linalg.norm(knee_w - f0), load, 0.35, width)
            lat = 0.5 * l_knee + 0.004
            for it in hw.plate_link_items(f"{side}_femur", f"{side}_femur", self.to_body(f"{side}_femur", f0),
                                          self.to_body(f"{side}_femur", knee_w), width, max(t_f, 0.003), m_f, lateral=lat):
                self.add(it)
            if layout in ("tandem", "side"):
                frac = max(0.40, (0.5 * d_pitch + l_yaw + 0.012 + 0.5 * d_knee) / femur_len)
                pk = hip_w + ax_f * (frac * femur_len)
                self.knee_motor_frac = frac
                self.add(hw.motor_item(f"{side}_knee_motor", f"{side}_femur", knee, self.to_body(f"{side}_femur", pk),
                                       (0, 1, 0)))
                crank = 0.42 * d_knee
                back = np.cross([0, 1, 0], ax_f)
                back = back if back[0] < 0 else -back
                rod_a = pk + back * crank
                rod_b = knee_w + back * (0.7 * crank) + ax_f * (-0.3 * crank)
                m_rod = 0.012 * (np.linalg.norm(rod_b - rod_a) / 0.12) * (d_knee / 0.055) ** 2
                self.add(hw.rod_item(f"{side}_knee_rod", f"{side}_femur", self.to_body(f"{side}_femur", rod_a),
                                     self.to_body(f"{side}_femur", rod_b), 0.06 * d_knee, m_rod))
            else:
                self.knee_motor_frac = 1.0
                self.add(hw.motor_item(f"{side}_knee_motor", f"{side}_femur", knee,
                                       self.to_body(f"{side}_femur", knee_w + np.array([0, s * 0.5 * l_knee, 0])), (0, 1, 0)))
            self.add(hw.Item(f"{side}_knee_bearings", f"{side}_femur", "bearings", hw.bearing_mass(d_knee),
                             self.to_body(f"{side}_femur", knee_w), np.eye(3), "none", (0,), "B", "estimate", "bearing rule",
                             visual=False, collide=False))
            # --- shank: tibia (knee->hock) + metatarsus (hock->MTP), fixed hock bend
            tib_len = np.linalg.norm(hock_w - knee_w)
            met_len = np.linalg.norm(mtp_w - hock_w)
            da, la = ank_a.envelope
            wid = max(0.020, 0.40 * da)
            m_t, t_t = hw.size_beam(tib_len, load, 0.30, wid)
            m_m, t_m = hw.size_beam(met_len, load, 0.30, wid)
            # the metatarsus plates stop short of the two-axis ankle block (half its size + 6 mm) so they do
            # not limit ankle pitch
            mtp_stop = mtp_w - (mtp_w - hock_w) / met_len * (0.5 * 0.028 * H / 0.38 + 0.006)
            for nm, a, b_, m_, t_ in (("tibia", knee_w, hock_w, m_t, t_t), ("metatarsus", hock_w, mtp_stop, m_m, t_m)):
                for it in hw.plate_link_items(f"{side}_{nm}", f"{side}_shank", self.to_body(f"{side}_shank", a),
                                              self.to_body(f"{side}_shank", b_), wid, max(t_, 0.003), m_,
                                              lateral=0.008 + 0.10 * wid):
                    self.add(it)
            self.add(hw.Item(f"{side}_ankle_bearings", f"{side}_ankle", "bearings", hw.bearing_mass(da) * 1.3,
                             self.to_body(f"{side}_ankle", mtp_w), np.eye(3), "none", (0,), "B", "estimate",
                             "bearing rule (2-DOF joint x1.3)", visual=False, collide=False))
            if layout in ("tandem", "side"):
                ax_t = (hock_w - knee_w) / tib_len
                back = np.cross([0, 1, 0], ax_t)
                back = back if back[0] < 0 else -back
                pos = []
                for k, act in enumerate((ank_a, ank_b)):
                    f0 = (ANKLE_MOTOR_FRAC_R2_BY_HEIGHT.get(c.height, ANKLE_MOTOR_FRAC[2]) if self.rev >= 2
                          else ANKLE_MOTOR_FRAC[1])
                    frac_a = f0 + k * (da + 0.006) / tib_len  # below the thigh fairing at knee flexion
                    pm = knee_w + ax_t * (frac_a * tib_len) + back * (0.5 * da + 0.010)
                    pos.append(pm)
                    self.add(hw.motor_item(f"{side}_ankle_motor_{'ab'[k]}", f"{side}_shank", act,
                                           self.to_body(f"{side}_shank", pm), (0, 1, 0)))
                    latoff = (0.5 * la + 0.004) * (1 if k == 0 else -1)
                    ra = pm + np.array([0, latoff, 0]) - ax_t * (0.4 * da)
                    rb = mtp_w + np.array([-0.06 * H, latoff * 0.8, 0.035 * H])
                    m_rod = 0.010 * (np.linalg.norm(rb - ra) / 0.15) * (da / 0.055) ** 2
                    self.add(hw.rod_item(f"{side}_ankle_rod_{'ab'[k]}", f"{side}_shank", self.to_body(f"{side}_shank", ra),
                                         self.to_body(f"{side}_shank", rb), 0.05 * da, m_rod))
                self.ankle_motor_pos = pos
            else:
                self.add(hw.motor_item(f"{side}_ankle_motor_a", f"{side}_shank", ank_a,
                                       self.to_body(f"{side}_shank", mtp_w + np.array([0, s * 0.5 * la, 0.0])), (0, 1, 0)))
                self.add(hw.motor_item(f"{side}_ankle_motor_b", f"{side}_ankle", ank_b,
                                       self.to_body(f"{side}_ankle", mtp_w + np.array([-0.5 * ank_b.envelope[1], 0, 0.0])),
                                       (1, 0, 0)))
            self.add(hw.Item(f"{side}_ankle_ujoint", f"{side}_ankle", "structure", 0.02 * (H / 0.38) ** 2,
                             self.to_body(f"{side}_ankle", mtp_w), np.eye(3), "box", (0.028 * H / 0.38,) * 3, "B",
                             "estimate", "universal joint block rule", rgba=hw.STRUCT_RGBA))
            self._place_foot(side, s, mtp_w, load)
            self.add(hw.Item(f"{side}_leg_harness", f"{side}_femur", "harness", 0.05 * (H / 0.38) ** 1.5,
                             self.to_body(f"{side}_femur", 0.5 * (hip_w + knee_w)), np.eye(3), "none", (0,), "B",
                             "estimate", "PR #662 0.05 kg/leg at 360 mm hip, scaled with hip height^1.5", visual=False,
                             collide=False))

    def _place_foot(self, side, s, mtp_w, load):
        sk = self.sk
        H = sk.H
        body = f"{side}_foot"
        toe_len = sk.leg["toe_len"]
        heel = sk.leg["heel_len"]
        h = T.LEG["mtp_height"] * H
        r_toe = 0.015 * H + 0.004
        splay = np.radians(T.LEG["toe_splay_deg"])
        toes, tips = [], []
        for ang, scl in ((0.0, 1.0), (splay, T.LEG["toe_side_scale"]), (-splay, T.LEG["toe_side_scale"])):
            L = toe_len * scl
            dvec = np.array([np.cos(ang), np.sin(ang), 0.0])
            base = mtp_w + np.array([0.012 * H, 0, -h + r_toe + 0.006 * H])
            tip = base + dvec * L + np.array([0, 0, -0.004 * H])
            toes.append((base, tip))
            tips.append((tip, dvec, L))
        m_toe = 0.016 * (H / 0.38) ** 2.3
        for k, (a, b) in enumerate(toes):
            it = hw.rod_item(f"{side}_toe_{k}", body, self.to_body(body, a), self.to_body(body, b), r_toe, m_toe,
                             vclass="B", source="toe rule", rgba=(0.33, 0.33, 0.30, 1), category="foot")
            it.collide = False
            self.add(it)
            dv = (b - a) / np.linalg.norm(b - a)
            claw_tip = b + dv * (0.035 * H) + np.array([0, 0, -0.008 * H])
            claw = hw.rod_item(f"{side}_claw_{k}", body, self.to_body(body, b), self.to_body(body, claw_tip),
                               0.55 * r_toe, 0.0, vclass="C", source="claw cap (mass in toe)", rgba=(0.20, 0.19, 0.17, 1),
                               category="foot")
            claw.collide = False
            self.add(claw)
        # hallux (digit I): small, cosmetic, high on the inside rear
        hb = mtp_w + np.array([-0.01 * H, -s * 0.0, -h + 0.05 * H])
        ht = hb + np.array([-0.06 * H, s * -0.025 * H, -0.02 * H])
        hx = hw.rod_item(f"{side}_hallux", body, self.to_body(body, hb), self.to_body(body, ht), 0.6 * r_toe, 0.004,
                         vclass="C", source="hallux (cosmetic)", rgba=(0.33, 0.33, 0.30, 1), category="foot")
        hx.collide = False
        self.add(hx)
        pad_h = 0.022 * H
        contact = [("heel", mtp_w + np.array([-0.2 * heel, 0, 0]), (0.050 * H, 0.060 * H))]
        for k, (tip, dvec, L) in enumerate(tips):
            c_ = tip - dvec * 0.22 * L
            contact.append((f"toe{k}", c_, (0.050 * H, 0.040 * H)))
        pts = []
        for nm, c_, (lx, ly) in contact:
            cc = np.array([c_[0], c_[1], 0.5 * pad_h])
            it = hw.box_item(f"{side}_pad_{nm}", body, "foot", 0.010 * (H / 0.38) ** 2, self.to_body(body, cc),
                             (lx, ly, pad_h), "B", "estimate", "elastomer pad rule", (0.13, 0.13, 0.12, 1))
            it.meta["floor_contact"] = True
            self.add(it)
            for dx in (-lx / 2, lx / 2):
                for dy in (-ly / 2, ly / 2):
                    pts.append(self.to_body(body, np.array([c_[0] + dx, c_[1] + dy, 0.0])))
        plate_len = toe_len * 0.7 + heel
        m_plate, _ = hw.size_beam(plate_len, load, 0.25, 0.10 * H)
        self.add(hw.box_item(f"{side}_foot_plate", body, "structure", m_plate,
                             self.to_body(body, mtp_w + np.array([0.28 * toe_len, 0, -h + pad_h + 0.004])),
                             (plate_len, 0.5 * toe_len, 0.005), "B", "estimate", "foot plate beam rule", hw.STRUCT_RGBA,
                             collide=False))
        self.add(hw.Item(f"{side}_foot_sensing", body, "sensors", 0.02 * (H / 0.38), self.to_body(body, mtp_w), np.eye(3),
                         "none", (0,), "A", "estimate", "PR #662 foot-load sensing allocation", visual=False, collide=False,
                         cost_nom=60, cost_high=140, power_nom=0.2, power_max=0.3))
        self.bodies[body].support = np.array(pts)
        self.bodies[body].sites.append(dict(name=f"{side}_sole", pos=self.to_body(body, np.array(
            [np.mean([c[1][0] for c in contact]), mtp_w[1], 0.0])), size=0.004))

    # ------------------------------------------------------------------ trunk hardware
    def _chest(self, p, chest=True):
        return self.sk.chest_rigid(p) if chest else np.asarray(p, float)

    def _place_trunk_hardware(self):
        c, sk = self.cand, self.sk
        B = sk.B
        sel = self.sel
        parts = catalog.parts()
        bendR = ry(sk.bend)
        trunk = "trunk"
        piv = sk.pivot[0]
        # Battery cassette, low in the belly ahead of the pubic boot; long axis along x.
        bat = parts[sel["battery"]]
        cas = (bat.dims[0] + 0.020, bat.dims[1] + 0.016, bat.dims[2] + 0.014)
        if c.battery_pos == "tailbase":
            # behind the rear hip-roll motor, ahead of the tail pivot; removed upward through a dorsal hatch
            roll = self.act("hip_roll")
            x_front = -(roll.envelope[1] + 0.012) if c.leg_layout == "tandem" else -0.02 * B
            bx = x_front - 0.5 * cas[0]
            st = self.at(bx)
            # low under the tail base (removed through a ventral hatch), below the tail motor, which sits
            # on the tail pivot; the bottom follows the highest hull point over the cassette length
            v_hi = max(self.at(x)["ventral"] for x in np.linspace(bx - 0.5 * cas[0], bx + 0.5 * cas[0], 7))
            bz = v_hi + 0.5 * cas[2] + 0.012
            chest = False
        else:
            bx = 0.20 * B + 0.5 * cas[0]
            st = self.at(bx)
            bz = st["ventral"] + 0.5 * cas[2] + 0.012 + 0.015 * B
            chest = bx > piv
        pos = self._chest(np.array([bx, 0.0, bz]), chest)
        R = bendR if chest else np.eye(3)
        m_cas = bat.mass * 0.28 + 0.03
        self.add(hw.box_item("battery", trunk, "battery", bat.mass, pos, bat.dims, "A", bat.basis, bat.id, hw.BATT_RGBA,
                             rot=R, cost_nom=bat.price, cost_high=bat.price * 1.15))
        self.add(hw.box_item("battery_cassette", trunk, "battery", m_cas, pos, cas, "B", "estimate",
                             "tray, latch, pads, leads: 28 % of pack + 30 g", (0.55, 0.38, 0.2, 0.25), rot=R, collide=False,
                             cost_nom=90, cost_high=160))
        self.battery = dict(center=pos, size=cas, part=bat.id, R=R)
        # Power distribution: above the battery (belly mode) or low in the abdomen (tail-base mode).
        bus = sel["bus"]
        pdims = (0.110, 0.070, 0.050) if bus <= 24 else (0.150, 0.090, 0.060)
        pmass = 0.30 if bus <= 24 else 0.62
        if c.battery_pos == "tailbase":
            # in the abdomen, kept 10 mm behind the chest compute tray
            comp_ = parts[sel["compute"]]
            tray_x = comp_.dims[0] + 0.017
            px = min(0.22 * B + 0.5 * pdims[0], 0.60 * B - 0.5 * tray_x - 0.5 * pdims[0] - 0.010)
            stp = self.at(px)
            ppos = self._chest(np.array([px, 0.0, stp["ventral"] + 0.5 * pdims[2] + 0.014 + 0.01 * B]), px > piv)
            pR = bendR if px > piv else np.eye(3)
        else:
            ppos = pos + R @ np.array([0.0, 0.0, 0.5 * cas[2] + 0.5 * pdims[2] + 0.010])
            pR = R
        self.add(hw.box_item("power_distribution", trunk, "power", pmass, ppos, pdims, "A", "estimate",
                             "PR #662 power bay 0.30 kg (24 V); 48 V bay with contactor and 48->12 V converter estimated",
                             hw.BOARD_RGBA, rot=pR, cost_nom=260 if bus <= 24 else 520, cost_high=420 if bus <= 24 else 780))
        # Compute tray in the chest.
        comp = parts[sel["compute"]]
        tray = (comp.dims[0] + 0.017, comp.dims[1] + 0.020, comp.dims[2] + 0.030)
        cx = 0.60 * B
        st = self.at(cx)
        cz = min(st["z_wide"] + 0.02 * B, st["dorsal"] - 0.5 * tray[2] - 0.014)
        cpos = self._chest(np.array([cx, 0.0, cz]))
        self.add(hw.box_item("compute", trunk, "compute", comp.mass, cpos, comp.dims, "A", comp.basis, comp.id,
                             hw.BOARD_RGBA, rot=bendR, cost_nom=comp.price, cost_high=comp.price * 1.1,
                             power_nom=comp.power_typ, power_max=comp.power_max))
        self.add(hw.box_item("compute_tray", trunk, "compute", 0.12, cpos, tray, "B", "estimate",
                             "NVMe, fan duct, mounts, leads (PR #662: 0.35 kg installed with module)", (0.3, 0.6, 0.4, 0.25),
                             rot=bendR, collide=False, cost_nom=120, cost_high=200))
        self.compute = dict(center=cpos, size=tray)
        # Control stack (MCU, CAN, IMU) above the hip-roll motors.
        roll = self.act("hip_roll")
        cd = (0.090, 0.065, 0.032)
        cpos2 = np.array([0.0, 0.0, self.hip["z_roll"] + 0.5 * roll.envelope[0] + 0.5 * cd[2] + 0.010])
        self.add(hw.box_item("control_stack", trunk, "control", 0.18 if c.height <= 75 else 0.24, cpos2, cd, "A",
                             "estimate", "PR #662 control allocation 0.18 kg (Teensy 4.1, CAN x2, IMU)", hw.BOARD_RGBA,
                             cost_nom=260, cost_high=420, power_nom=4.0, power_max=6.0))
        self.add(hw.Item("imu", trunk, "sensors", 0.02, cpos2 + np.array([0, 0, 0.5 * cd[2] + 0.005]), np.eye(3), "none",
                         (0,), "A", "estimate", "IMU breakout", visual=False, collide=False, cost_nom=40, cost_high=400))
        self.add(hw.Item("torso_harness", trunk, "harness", 0.08 * (B / 0.38) ** 1.6, np.array([0.25 * B, 0, 0]), np.eye(3),
                         "none", (0,), "B", "estimate", "PR #662 0.08 kg at 360 mm hip, scaled with B^1.6", visual=False,
                         collide=False))
        # Two ToF sensors under the chest, 45 deg down, each turned 12 deg outward over its own foot path
        # (round-one screening: the earlier pair aimed identically and overlapped 82-90 %).
        tof = parts[sel["tof"]]
        for k, sgn in enumerate((1, -1)):
            tx = 0.50 * B
            st = self.at(tx)
            tp = self._chest(np.array([tx, sgn * 0.030 * B, st["ventral"] + 0.014]), tx > piv)
            tR = sk.R_p.T @ rz(np.radians(12.0 * sgn)) @ ry(np.radians(-45.0))
            self.add(hw.box_item(f"tof_{'lr'[k]}", trunk, "sensors", tof.mass + 0.012, tp, (0.008, 0.020, 0.020), "A",
                                 tof.basis, tof.id, hw.SENSOR_RGBA, rot=tR, cost_nom=tof.price, cost_high=tof.price * 1.2,
                                 power_nom=tof.power_typ, power_max=tof.power_max))
            self.bodies[trunk].sites.append(dict(name=f"tof_{'lr'[k]}", pos=tp, rot=tR, size=0.004, fov=tof.fov, part=tof.id))
        # The camera is in the head (_place_head_tail_hardware) and the lidar under the as-built belly
        # (_place_lidar, after the coverings exist).

    def _place_head_tail_hardware(self):
        sk = self.sk
        B = sk.B
        # Stereo camera in the skull roof just ahead of the orbits, looking forward over the snout
        # (round-one screening: a throat camera lost the view straight ahead to the jaw, and its box
        # met the neck motor).  Pitched +5 deg above level in the world; the head keeps its world
        # attitude in every posture, so the view does not tilt with the chest.
        cam = catalog.parts()[self.sel["camera"]]
        cam_size = (cam.dims[2], cam.dims[0], cam.dims[1])
        station, drop = CAMERA_STATION, 0.004
        if self.rev >= 2:
            fit = self._camera_fit(cam_size)
            self.geometry_checks["camera_fit"] = fit
            if fit["fits"]:
                station, drop = fit["station"], fit["drop_m"]
            else:
                self.notes.append("stereo camera does not fit inside the skull cover at any searched station")
        top_w, ax, up = shapes.skull_dorsal(sk, self.fk, station)
        cam_w = top_w - up * (0.5 * cam_size[2] + drop) - ax * 0.5 * cam_size[0]
        cam_p = self.to_body("head", cam_w)
        cam_R = self.bodies["head"].R_world.T @ ry(np.radians(CAMERA_PITCH_DEG))
        self.add(hw.box_item("camera", "head", "sensors", cam.mass + 0.04, cam_p, cam_size, "A", cam.basis, cam.id,
                             hw.SENSOR_RGBA, rot=cam_R, cost_nom=cam.price + 25, cost_high=cam.price + 60,
                             power_nom=cam.power_typ, power_max=cam.power_max))
        self.bodies["head"].sites.append(dict(name="camera", pos=cam_p, rot=cam_R, size=0.006, fov=cam.fov, part=cam.id))
        self.add(hw.Item("camera_cable", "head", "harness", 0.012 + 0.03 * (B / 0.38), cam_p, np.eye(3), "none", (0,), "B",
                         "estimate", "USB/flex camera lead through the neck with a service loop", visual=False,
                         collide=False))
        neck = self.act("neck")
        nb = sk.neck_base
        p_motor = nb + ry(sk.bend) @ np.array([-0.02 * B, 0, -0.02 * B])
        self.add(hw.motor_item("neck_motor", "trunk", neck, p_motor, ry(sk.bend) @ np.array([0, 1.0, 0]),
                               category="expression_actuator"))
        self.add(hw.Item("neck_fork_bearings", "trunk", "bearings", 0.08 * (B / 0.38) ** 2, nb, np.eye(3), "none", (0,),
                         "B", "estimate", "PR #662 neck fork 0.08 kg, scaled with B^2", visual=False, collide=False))
        tail = self.act("tail")
        tr = sk.tail_root
        h_ax = (tail.box[2] + 0.004) if tail.box is not None else tail.envelope[1]
        lift = 0.0
        if self.rev >= 2 and self.cand.battery_pos == "tailbase":
            lift = self._tail_motor_lift(tail, tr, h_ax)
            if lift > 0:
                self.geometry_checks["tail_motor_raised_m"] = lift
        self.add(hw.motor_item("tail_motor", "trunk", tail, tr + np.array([0.0, 0.0, 0.5 * h_ax + 0.006 + lift]),
                               (0, 0, 1), category="expression_actuator"))  # on the yaw axis, above the pivot
        self.add(hw.Item("tail_fork_bearings", "trunk", "bearings", 0.08 * (B / 0.38) ** 2, tr, np.eye(3), "none", (0,),
                         "B", "estimate", "PR #662 tail fork 0.08 kg, scaled with B^2", visual=False, collide=False))

    # ------------------------------------------------------------------ structure
    def _place_structure(self):
        sk = self.sk
        B = sk.B
        mg = self.mass_guess
        roll = self.act("hip_roll")
        plen = 0.55 * B
        ph = self.hip["pelvis_half"]
        m_pel, t_pel = hw.size_beam(plen, 2.0 * mg * G, 0.25, max(0.05, 1.2 * roll.envelope[0]))
        for s in (1, -1):
            self.add(hw.box_item(f"pelvis_plate_{'lr'[s < 0]}", "trunk", "structure", m_pel / 2,
                                 np.array([-0.05 * B, s * ph, self.hip["z_roll"]]),
                                 (plen, max(0.003, t_pel), max(0.05, 1.2 * roll.envelope[0])), "B", "estimate",
                                 "pelvis twin-plate beam rule (2 x body weight)", hw.STRUCT_RGBA))
        front_mass = 0.35 * mg
        a = np.array([0.05 * B, 0, 0.10 * B])
        b = sk.neck_base + ry(sk.bend) @ np.array([-0.06 * B, 0, -0.05 * B])
        slen = np.linalg.norm(b - a)
        m_sp, _ = hw.size_beam(slen, 3.0 * front_mass * G, 0.5, max(0.03, 0.06 * B))
        self.add(hw.rod_item("spine_beam", "trunk", a, b, max(0.008, 0.02 * B), m_sp, source="spine twin-plate rule"))
        self.add(hw.Item("trays_crossmembers", "trunk", "structure", 0.12 + 0.02 * mg, np.array([0.4 * B, 0, -0.02 * B]),
                         np.eye(3), "none", (0,), "B", "estimate", "trays/crossmembers 0.12 kg + 2 % of mass", visual=False,
                         collide=False))
        occ_local = self.to_body("head", sk.occiput)
        head_mass_guess = 0.04 * (B / 0.38) ** 2.3 * 10
        m_ns, _ = hw.size_beam(np.linalg.norm(occ_local), 3.0 * head_mass_guess * G, 0.5, max(0.02, 0.05 * B))
        self.add(hw.rod_item("neck_spar", "head", np.zeros(3), occ_local, max(0.006, 0.015 * B), m_ns,
                             source="neck spar rule"))
        ax = np.array([np.cos(sk.skull_pitch), 0, np.sin(sk.skull_pitch)])
        self.add(hw.Item("skull_mount", "head", "structure", 0.12 * (B / 0.38) ** 2, occ_local + ax * 0.3 * sk.skull_len,
                         np.eye(3), "none", (0,), "B", "estimate", "PR #662 skull mount 0.12 kg, scaled with B^2",
                         visual=False, collide=False))
        path = sk.tail_path - sk.tail_path[0]
        r_out = max(0.006, 0.018 * B)
        wall = max(0.0008, 0.06 * r_out)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        tail_len = float(seg.sum())
        k_end = int(0.8 * (len(path) - 1))
        m_spar = hw.CF_DENSITY * np.pi * (r_out**2 - (r_out - wall) ** 2) * 0.8 * tail_len
        self.add(hw.rod_item("tail_spar", "tail", path[0], path[k_end], r_out, m_spar + 0.02 * (B / 0.38) ** 2,
                             source="CF tube spar rule", rgba=(0.12, 0.12, 0.12, 1)))
        self.add(hw.Item("tail_ribs", "tail", "structure", 0.04 * (B / 0.38) ** 2.2, path[int(0.3 * len(path))], np.eye(3),
                         "none", (0,), "B", "estimate", "printed ribs 40 g at B=0.38, scaled B^2.2", visual=False,
                         collide=False))

    # ------------------------------------------------------------------ covers
    def _place_covers(self):
        c, sk = self.cand, self.sk
        B = sk.B
        fam = T.FAMILY_SHAPE[self.fk]
        colors = FAMILY_COLORS[c.family]
        t = cover_thickness(c.height, fam["shell_mm"])
        mat = cover_material(c.height)
        regions = fam["cover_regions"]
        prof = self._required_profile()
        self.prof = prof
        for p in shapes.trunk_panels(sk, self.fk, prof, regions, t, mat, colors["main"], colors["accent"]):
            self.bodies["trunk"].panels.append(p)
        if "rib_hoops" in regions:
            for p in shapes.rib_hoops(sk, prof, (0.27, 0.36, 0.45, 0.54), t, (0.50, 0.50, 0.46, 1), fam.get("rib_hoops")):
                self.bodies["trunk"].panels.append(p)
        nb_w = self.bodies["head"].pos_world
        self.bodies["head"].panels.append(shapes.neck_panel(sk, self.fk, nb_w, t, mat, colors["main"]))
        for p in shapes.skull_panels(sk, self.fk, nb_w, t, mat, colors["accent"], colors["jaw"], TEETH_RGBA):
            if isinstance(p, shapes.Panel):
                self.bodies["head"].panels.append(p)
            elif p[0] == "teeth":
                for k, (a, b) in enumerate(p[1]):
                    it = hw.rod_item(f"tooth_{k}", "head", a, b, 0.010 * sk.skull_len, 0.0, vclass="C",
                                     source="cosmetic (mass in skull shell)", rgba=p[2], category="cosmetic")
                    it.collide = False
                    self.add(it)
            elif p[0] == "eyes":
                for k, e in enumerate(p[1]):
                    self.add(hw.Item(f"eye_{k}", "head", "cosmetic", 0.0, e, np.eye(3), "box",
                                     (0.035 * sk.skull_len, 0.012 * sk.skull_len, 0.035 * sk.skull_len), "C", "estimate",
                                     "cosmetic", rgba=p[2], collide=False))
        tr_w = self.bodies["tail"].pos_world
        for p in shapes.tail_panels(sk, self.fk, tr_w, prof, t, mat, colors["main"], regions):
            self.bodies["tail"].panels.append(p)
        pitch, knee = self.act("hip_pitch"), self.act("knee")
        yaw = self.act("hip_yaw")
        for side, s in (("left", 1.0), ("right", -1.0)):
            if "thigh" in regions:
                Lf = float(np.linalg.norm(sk.leg["knee"] - sk.leg["hip"]))
                fs_ = T.FAMILY_SHAPE[self.fk]
                r_top = max(0.5 * pitch.envelope[0] + 0.012, 0.19 * B * fs_["thigh_scale"])
                s_seam = (0.5 * pitch.envelope[0] + yaw.envelope[1] + 0.012) / Lf
                r_seam = max(0.5 * knee.envelope[0], 0.5 * knee.envelope[1], 0.5 * yaw.envelope[0]) + 0.010
                kf = getattr(self, "knee_motor_frac", 0.4)
                r_knee = max(0.045 * B, (0.5 * max(knee.envelope) + 0.008) if kf > 0.75 else 0.0)
                half_up = self.hip["half_upper"] + 0.004
                tl = fs_.get("thigh_lower")
                kw = dict(lat_min=0.5 * knee.envelope[1] + tl["lat_extra"], end_s=tl["end_s"]) if tl else {}
                up, lo = shapes.thigh_panels(side, self.femur_axis, Lf, B, self.fk, r_top, half_up, s_seam, r_seam,
                                             r_knee, t, mat, colors["limb"], **kw)
                self.bodies[f"{side}_thigh"].panels.append(up)
                self.bodies[f"{side}_femur"].panels.append(lo)
            if "shank" in regions and c.leg_layout in ("tandem", "side"):
                hock_l = self.to_body(f"{side}_shank", sk.leg["hock"] + np.array([0, s * self.hip["y_leg"], 0]))
                half_t = 0.5 * self.act("ankle_a").envelope[1] + 0.008
                if fam.get("calf") and self.rev >= 2:
                    # revision 2: a calf sized from the ankle drives, covering both of them
                    back = self.act("ankle_a").envelope[0] + fam["calf"]["clear"]
                    low = c.height <= 60  # the upper drive sits at 0.46 of the tibia at 60 cm (F9)
                    f0, f1 = (0.30, 1.0) if low else (0.06, 0.88)
                    panel = shapes.shank_panel(side, f0 * hock_l, f1 * hock_l, B, half_t, t, mat, colors["limb"],
                                               back_m=back, profile="r2_low" if low else "r2")
                else:
                    panel = shapes.shank_panel(side, 0.06 * hock_l, 0.66 * hock_l, B, half_t, t, mat, colors["limb"])
                self.bodies[f"{side}_shank"].panels.append(panel)
        # fixed arms, low at the front of the chest
        from .geometry import sweep

        L = T.SKELETON["arm_len"] * B
        st = self.prof.at(fam.get("arm_x", T.SKELETON["arm_x"]) * B)
        for side, s in (("left", 1.0), ("right", -1.0)):
            root = sk.arm_root + ry(sk.bend) @ np.array([0, s * 0.82 * st["half"], 0])
            pts = np.array([[0, 0, 0], [0.06, s * 0.02, -0.40], [0.30, s * 0.05, -0.50], [0.56, s * 0.06, -0.47],
                            [0.70, s * 0.05, -0.55]]) * L
            pts = (ry(sk.bend) @ pts.T).T + root
            r = np.linspace(0.13, 0.05, len(pts)) * L
            ang = np.radians(np.linspace(-180, 180, 11)[:-1])
            secs = [shapes.egg_ring(r[k], r[k], -r[k], 0.0, 2.0, 2.0, ang) for k in range(len(pts))]
            self.bodies["trunk"].panels.append(shapes.Panel(f"{side}_arm", "trunk", sweep(pts, secs), t, mat, "arm", True,
                                                            False, colors["accent"], mounts=2))

    def _place_lidar(self):
        """2D lidar under the as-built belly, scan plane clear of the hull (round-one screening: placed
        inside the hull it widened the belly around itself and blocked 93-100 % of its own scan)."""
        if not self.sel.get("lidar"):
            return
        sk = self.sk
        B = sk.B
        li = catalog.parts()[self.sel["lidar"]]
        lx = LIDAR_STATION * B
        st = self.prof.at(lx)
        # lowest hull point over the lidar's footprint, in the unbent design frame
        xs = np.linspace(lx - 0.6 * li.dims[0], lx + 0.6 * li.dims[0], 7)
        vent = min(float(self.prof.at(x)["ventral"]) for x in xs)
        lp = self._chest(np.array([lx, 0.0, vent - 0.5 * li.dims[2] - 0.004]))
        lR = sk.R_p.T
        self.add(hw.box_item("lidar", "trunk", "sensors", li.mass + 0.03, lp, li.dims, "A", li.basis, li.id,
                             hw.SENSOR_RGBA, rot=lR, cost_nom=li.price + 30, cost_high=li.price + 80,
                             power_nom=li.power_typ, power_max=li.power_max))
        gap = float(st["ventral"] - vent) + 0.004  # lidar top to the hull above its centre
        bp = self._chest(np.array([lx, 0.0, vent + 0.5 * gap - 0.004]))
        self.add(hw.box_item("lidar_bracket", "trunk", "structure", 0.025 * (B / 0.38) ** 2, bp,
                             (0.6 * li.dims[0], 0.6 * li.dims[1], gap + 0.010), "B", "estimate",
                             "printed throat bracket and guard", hw.STRUCT_RGBA, rot=lR, collide=False))
        self.bodies["trunk"].sites.append(dict(name="lidar", pos=lp, rot=lR, size=0.005, fov=li.fov, part=li.id))
        self.geometry_checks["lidar_below_hull_m"] = float(st["ventral"] - vent)

    def _required_profile(self) -> shapes.TrunkProfile:
        """Widen/deepen the designed trunk profile where hardware needs more room (class A/B)."""
        prof = self.design_profile
        sk = self.sk
        clearance, wall = 0.008, 0.002
        xs = prof.x.copy()
        dors, vent, half = prof.dorsal.copy(), prof.ventral.copy(), prof.half_width.copy()
        hw_half = np.zeros_like(half)
        widened = []
        inv = ry(-sk.bend)
        for it in self.bodies["trunk"].items:
            if it.shape not in ("box", "cylinder"):
                continue
            if it.category in ("leg_actuator",) and self.cand.leg_layout == "serial":
                continue
            if it.category in ("structure",) and it.name.startswith("pelvis_plate"):
                pass
            elif it.volume_class != "A" and it.name != "battery_cassette":
                continue
            corners = _corners(it)
            unb = corners.copy()
            mask = corners[:, 0] > sk.pivot[0]
            unb[mask] = (inv @ (corners[mask] - sk.pivot).T).T + sk.pivot
            for k, x0 in enumerate(xs):
                win = 0.5 * (xs[min(k + 1, len(xs) - 1)] - xs[max(k - 1, 0)])
                sel = unb[np.abs(unb[:, 0] - x0) <= win]
                if not len(sel):
                    continue
                need_up = sel[:, 2].max() + clearance + wall
                need_lo = sel[:, 2].min() - clearance - wall
                need_w = np.abs(sel[:, 1]).max() + clearance + wall
                if need_up > dors[k]:
                    widened.append((it.name, "dorsal", float(x0 / sk.B), float(need_up - dors[k])))
                    dors[k] = need_up
                if need_lo < vent[k]:
                    widened.append((it.name, "ventral", float(x0 / sk.B), float(vent[k] - need_lo)))
                    vent[k] = need_lo
                hw_half[k] = max(hw_half[k], need_w)
                if need_w > half[k]:
                    widened.append((it.name, "half_width", float(x0 / sk.B), float(need_w - half[k])))
                    half[k] = need_w
        # The pelvis covers clear the thigh fairings: within the fairing's reach of the hip axis the trunk
        # half-width stays inside the fairing's inner face, less the wall and a 6 mm gap (round-one
        # clearance screen: pelvis and upper-thigh shells overlapped at home, up to 21 mm at 120 cm).
        fam = T.FAMILY_SHAPE[self.fk]
        if "thigh" in fam["cover_regions"]:
            pitch = self.act("hip_pitch")
            r_top = max(0.5 * pitch.envelope[0] + 0.012, 0.19 * sk.B * fam["thigh_scale"])
            t = cover_thickness(self.cand.height, fam["shell_mm"])
            cap = self.hip["y_leg"] - (self.hip["half_upper"] + 0.004) - t - 0.006
            narrowed, conflicts = [], []
            for k, x0 in enumerate(xs):
                if abs(x0) <= 1.06 * r_top + 0.010 and half[k] > cap:
                    new_h = max(cap, hw_half[k])
                    if hw_half[k] > cap:
                        conflicts.append((float(x0 / sk.B), float(hw_half[k] - cap)))
                    narrowed.append((float(x0 / sk.B), float(half[k] - new_h)))
                    half[k] = new_h
            self.geometry_checks["pelvis_narrowed_for_thighs"] = narrowed
            self.geometry_checks["pelvis_thigh_conflicts"] = conflicts
        self.geometry_checks["trunk_widened_for_hardware"] = widened
        self.geometry_checks["designed_profile"] = dict(x=prof.x.tolist(), dorsal=prof.dorsal.tolist(),
                                                        ventral=prof.ventral.tolist(), half=prof.half_width.tolist())
        return shapes.TrunkProfile(xs, dors, vent, half, prof.wide, prof.n_up, prof.n_lo)


def _frame_z(axis) -> np.ndarray:
    """Rotation whose local z is ``axis`` and local y is world y (for sagittal-plane items)."""
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _corners(it: hw.Item) -> np.ndarray:
    if it.shape == "box":
        L, W, Hh = it.size
        loc = np.array([[sx * L / 2, sy * W / 2, sz * Hh / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif it.shape == "cylinder":
        d, ln = it.size
        ang = np.linspace(0, 2 * np.pi, 13)[:-1]
        loc = np.array([[d / 2 * np.cos(a), d / 2 * np.sin(a), z] for a in ang for z in (-ln / 2, ln / 2)])
    else:
        return np.zeros((0, 3))
    return it.pos + (it.rot @ loc.T).T


# ================================================================================================
# Scale solve
# ================================================================================================
def crown_height(cand: Candidate, B: float) -> float:
    return shapes.skull_crown_world(build_skeleton(cand, B), T.family_key(cand.family, int(cand.override("rev", 1))))


def solve_B(cand: Candidate) -> float:
    """Body unit that puts the crown at the target height in the reference posture."""
    ref = cand.at_posture(REFERENCE_POSTURE)
    return cand.crown_target / crown_height(ref, 1.0)


del frame_from_axis
