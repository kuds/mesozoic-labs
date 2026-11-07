"""World-frame skeleton landmarks for a candidate.

The body unit ``B`` scales every anatomical landmark; the hip height is ``H = leg_scale * B``.
Hardware never scales with ``B``: it is placed afterwards from catalog dimensions.

Posture handling (see ``params``): the pelvis is pitched by ``pelvis_deg`` about the hip-pitch
axis midpoint; the chest is bent about the lumbar pivot by the angle that makes the
acetabulum-glenoid line reach ``chest_deg``; the neck joint's nominal angle keeps the neck and
skull orientation fixed in the world (plus any ``neck_rise_deg``), so a chest-up posture raises
the neck base and the head rather than tipping the skull back; the tail root carries a fixed
built-in pitch so the tail keeps its world orientation.  Legs keep their design geometry; the
stance solver then moves the feet so the centre of mass is centred over the support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import template as T
from .geometry import ry
from .params import POSTURES, PROPORTIONS, REFERENCE_POSTURE, Candidate


@dataclass
class Skeleton:
    cand: Candidate
    B: float
    H: float
    theta_p: float  # pelvis pitch (rad, chest-up positive)
    theta_c: float  # torso (acetabulum-glenoid) inclination (rad)
    bend: float  # lumbar bend (rad) applied to the chest about the pivot (trunk frame)
    R_p: np.ndarray  # trunk frame -> world rotation
    origin: np.ndarray  # world position of the trunk frame origin (hip-pitch axis midpoint)
    pivot: np.ndarray  # lumbar pivot, trunk frame
    glenoid: np.ndarray  # trunk frame (bent)
    neck_base: np.ndarray  # trunk frame (bent)
    neck_path: np.ndarray  # world, from base to occiput
    occiput: np.ndarray  # world
    skull_len: float
    skull_pitch: float  # world pitch of the skull axis (rad)
    tail_root: np.ndarray  # trunk frame
    tail_path: np.ndarray  # world
    leg: dict = field(default_factory=dict)  # world landmarks for the left leg (mirror for right)
    arm_root: np.ndarray = None  # trunk frame (bent)

    # ------------------------------------------------------------------ frame helpers
    def chest_rot(self) -> np.ndarray:
        return ry(self.bend)

    def bend_point(self, p_trunk: np.ndarray, blend_width: float | None = None) -> np.ndarray:
        """Apply the lumbar bend to trunk-frame point(s) with a smooth blend across the pivot."""
        p = np.atleast_2d(np.asarray(p_trunk, float))
        w = blend_width if blend_width is not None else 0.10 * self.B
        x = p[:, 0]
        s = np.clip((x - (self.pivot[0] - w)) / (2 * w), 0.0, 1.0)
        s = s * s * (3 - 2 * s)
        out = p.copy()
        r_full = self.chest_rot()
        for k in range(len(p)):
            if s[k] <= 0:
                continue
            ang = self.bend * s[k]
            r = ry(ang)
            out[k] = self.pivot + r @ (p[k] - self.pivot)
        del r_full
        return out if np.ndim(p_trunk) > 1 else out[0]

    def chest_rigid(self, p_trunk: np.ndarray) -> np.ndarray:
        """Rigid chest transform (for components mounted in the chest)."""
        return self.pivot + ry(self.bend) @ (np.asarray(p_trunk, float) - self.pivot)

    def to_world(self, p_trunk: np.ndarray) -> np.ndarray:
        return self.origin + (self.R_p @ np.asarray(p_trunk, float).T).T

    def to_trunk(self, p_world: np.ndarray) -> np.ndarray:
        return (self.R_p.T @ (np.asarray(p_world, float) - self.origin).T).T


def _posture(cand: Candidate) -> dict:
    return POSTURES[cand.posture]


def _neck_path(base_world: np.ndarray, B: float, neck_scale: float, rise: float) -> np.ndarray:
    pts = [base_world]
    p = base_world.copy()
    for length, ang in T.SKELETON["neck_segments"]:
        a = np.radians(ang) + rise
        n = 6
        for _ in range(1, n + 1):
            step = length * B * neck_scale / n
            p = p + step * np.array([np.cos(a), 0.0, np.sin(a)])
            pts.append(p.copy())
    return np.array(pts)


def _tail_path(root_world: np.ndarray, B: float, tail_scale: float, n: int = 24) -> np.ndarray:
    L = T.SKELETON["tail_len"] * B * tail_scale
    a0 = np.radians(T.SKELETON["tail_pitch_deg"])
    curve = np.radians(T.SKELETON["tail_curve_deg"])
    pts = [root_world]
    p = root_world.copy()
    for k in range(1, n + 1):
        u = k / n
        a = a0 + curve * max(0.0, (u - 0.4) / 0.6) ** 1.5
        p = p + (L / n) * np.array([-np.cos(a), 0.0, np.sin(a)])
        pts.append(p.copy())
    return np.array(pts)


def build_skeleton(cand: Candidate, B: float) -> Skeleton:
    post = _posture(cand)
    prop = PROPORTIONS[cand.proportion]
    S = T.SKELETON
    H = B * prop["leg_scale"]
    theta_p = np.radians(post["pelvis_deg"])
    theta_c = np.radians(post["chest_deg"])
    R_p = ry(theta_p)
    origin = np.array([0.0, 0.0, H])
    pivot = np.array([S["lumbar_x"], 0.0, S["lumbar_z"]]) * B
    g0 = np.array([S["glenoid_x"], 0.0, S["glenoid_z"]]) * B

    # Solve the lumbar bend so the acetabulum->glenoid line has world inclination theta_c.
    bend = theta_c - theta_p
    for _ in range(30):
        g = pivot + ry(bend) @ (g0 - pivot)
        gw = R_p @ g
        incl = np.arctan2(gw[2], gw[0])
        err = theta_c - incl
        if abs(err) < 1e-10:
            break
        bend += err
    glenoid = pivot + ry(bend) @ (g0 - pivot)

    nb0 = np.array([S["neck_base_x"], 0.0, S["neck_base_z"]]) * B
    neck_base = pivot + ry(bend) @ (nb0 - pivot)
    nb_world = origin + R_p @ neck_base
    rise = np.radians(post["neck_rise_deg"])
    neck_path = _neck_path(nb_world, B, prop["neck_scale"], rise)
    occiput = neck_path[-1]
    skull_len = S["skull_len"] * B * prop["skull_scale"]
    skull_pitch = np.radians(S["skull_pitch_deg"]) + rise

    trx = S["tail_root_x_tailbase"] if cand.battery_pos == "tailbase" else S["tail_root_x"]
    tail_root = np.array([trx, 0.0, S["tail_root_z"]]) * B
    trx_m = cand.override("tail_root_x_m")  # packaging setback (build.Design), metres in the trunk frame
    if trx_m is not None:
        tail_root[0] = min(tail_root[0], float(trx_m))
    tail_path = _tail_path(origin + R_p @ tail_root, B, prop["tail_scale"])

    arm0 = np.array([S["arm_x"], 0.0, S["arm_z"]]) * B
    arm_root = pivot + ry(bend) @ (arm0 - pivot)

    # Leg design pose (left leg, before hip-module lateral offsets are known).
    L = T.leg_chain_length(H)
    lf, lt, lm = (T.LEG[k] * L for k in ("femur_frac", "tibia_frac", "meta_frac"))
    af, at, am = (np.radians(T.LEG[k]) for k in ("femur_deg", "tibia_deg", "meta_deg"))
    hip = np.array([0.0, 0.0, H])
    knee = hip + lf * np.array([np.sin(af), 0, -np.cos(af)])
    hock = knee + lt * np.array([np.sin(at), 0, -np.cos(at)])
    mtp = hock + lm * np.array([np.sin(am), 0, -np.cos(am)])
    leg = dict(hip=hip, knee=knee, hock=hock, mtp=mtp, chain=L, femur=lf, tibia=lt, meta=lm,
               toe_len=T.LEG["toe_len"] * H, heel_len=T.LEG["heel_len"] * H)

    return Skeleton(cand=cand, B=B, H=H, theta_p=theta_p, theta_c=theta_c, bend=bend, R_p=R_p,
                    origin=origin, pivot=pivot, glenoid=glenoid, neck_base=neck_base,
                    neck_path=neck_path, occiput=occiput, skull_len=skull_len, skull_pitch=skull_pitch,
                    tail_root=tail_root, tail_path=tail_path, leg=leg, arm_root=arm_root)


def reference_candidate(cand: Candidate) -> Candidate:
    return cand.at_posture(REFERENCE_POSTURE)
