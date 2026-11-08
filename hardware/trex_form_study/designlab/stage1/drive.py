"""Stage 1 drives: the two-motor ankle linkage and the torque-speed envelope (``STAGE1_PLAN.md`` section 2).

Ankle linkage
-------------
The two ankle motors of a leg sit high on the shank and drive the foot through push rods (tandem layout, unit
crank ratios).  In the plant each motor turns a fixed tendon:

    motor a:  L_a = q_p + q_r          motor b:  L_b = q_p - q_r

so motor torques f map to joint torques ``tau_p = f_a + f_b`` and ``tau_r = f_a - f_b``, and back

    f_a = (tau_p + tau_r) / 2,   f_b = (tau_p - tau_r) / 2,   w_a = w_p + w_r,   w_b = w_p - w_r,

which is ``sizing._motor_space`` for the tandem and side layouts.  With each motor held to +-peak the ankle reaches
exactly ``|tau_p| + |tau_r| <= 2 peak``; the screened MJCF gives each ankle joint a +-peak box instead, the largest
box inside that diamond (``mjcf.joint_limits``).  Each motor's rotor inertia (its catalog armature, floored as in
``mjcf.joint_armature``) sits on its own tendon, so pitch and roll each carry both motors' reflected inertia, where
the screened MJCF puts one motor's armature on each joint.

Torque-speed envelope
---------------------
Before every physics step each drive's torque limit is set from its own speed |w| (the motor speed for the ankle
motors, the joint speed for the others):

    |f| <= peak                              for |w| <= w1 = Actuator.speed_at(bus)
    |f| <= peak (w0 - |w|) / (w0 - w1)       for w1 < |w| < w0, and 0 beyond w0

``peak`` is ``Actuator.peak`` (0.70 x the vendor peak, ``catalog.PEAK_FRACTION``).  ``w0`` is the no-load speed on
the bus: ``Actuator.no_load_speed`` derated by bus / nominal below the drive's nominal voltage, with no credit above
it, the rule of ``Actuator.speed_at``; box servos are not on the leg bus and keep their own ratings, as in
``speed_at``.  The limit is the same for driving and braking torque, as the plan states it.  ``speed_scale`` and
``torque_scale`` scale the speed axis and the torque axis of every envelope (the margin sweeps of plan section 5;
1.0 otherwise).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .. import catalog, mjcf, statics

#: The ankle motors and the roll coefficient of each one's tendon (the pitch coefficient is +1).
ANKLE_MOTORS = (("ankle_a", 1.0), ("ankle_b", -1.0))
#: The ankle joint each motor's actuator replaces in the screened MJCF's actuator list.
REPLACES = {"ankle_a": "ankle_pitch", "ankle_b": "ankle_roll"}


@dataclass(frozen=True)
class Drive:
    """One motor of the plant and its planning limits."""

    actuator: str  # actuator name in the plant
    coordinate: str  # the joint it turns, or the tendon (ankle motors)
    group: str  # sizing group (hip_roll, hip_pitch, hip_yaw, knee, ankle_a, ankle_b, neck, tail)
    part: str  # catalog id
    peak: float  # N m, Actuator.peak
    sustained: float  # N m, Actuator.sustained
    rated_speed: float  # rad/s, Actuator.speed_at(bus): the envelope's corner
    no_load_speed: float  # rad/s on the bus: where the envelope reaches zero
    armature: float  # kg m^2 on the drive's own coordinate

    def as_dict(self) -> dict:
        return asdict(self)


def no_load_speed_at(a: catalog.Actuator, bus: float | None) -> float:
    """No-load speed of a drive on a bus nominal, derated like ``Actuator.speed_at`` (box servos keep their own)."""
    if bus is None or a.box is not None:
        return a.no_load_speed
    return a.no_load_speed * min(1.0, bus / a.nominal_voltage)


def torque_limit(speed, peak, rated_speed, no_load_speed):
    """The envelope: ``peak`` up to ``rated_speed``, then linear to zero at ``no_load_speed`` (element-wise)."""
    s = np.abs(np.asarray(speed, float))
    return np.clip(peak * (no_load_speed - s) / (no_load_speed - rated_speed), 0.0, peak)


def tendon_name(side: str, motor: str) -> str:
    return f"{side}_{motor}"


def actuator_name(coordinate: str) -> str:
    """Plant actuator names keep the screened MJCF's ``<joint>_servo`` pattern; the ankle motors are on tendons."""
    return f"{coordinate}_servo"


def drives(selection: dict, joints: list[str]) -> list[Drive]:
    """The plant's drives, one per hinge in ``joints`` order (the screened MJCF's actuator order), with each ankle's
    pitch and roll servos replaced by its motors a and b.  ``selection`` is a record's ``selection``."""
    bus = selection.get("bus")
    out = []
    for jn in joints:
        group = mjcf.joint_role(jn)
        part = selection["joints"][group] if group in selection["joints"] else selection[group]
        a = catalog.actuator(part)
        bus_v = bus if a.box is None else None
        coordinate = jn
        if group in REPLACES:
            coordinate = tendon_name(jn.split("_", 1)[0], group)
        out.append(Drive(actuator=actuator_name(coordinate), coordinate=coordinate, group=group, part=part,
                         peak=a.peak, sustained=a.sustained, rated_speed=a.speed_at(bus_v),
                         no_load_speed=no_load_speed_at(a, bus_v), armature=max(a.armature, 1e-5)))
    for d in out:
        if not d.no_load_speed > d.rated_speed:
            raise ValueError(f"{d.part}: no-load speed {d.no_load_speed:.3f} rad/s is not above its rated speed")
    return out


def motor_matrix(joints: list[str]) -> np.ndarray:
    """G with motor coordinates L = G q over the hinges ``joints`` (rows: the drives of ``drives``, same order).

    Motor speeds are ``G @ qd``, joint torques from motor torques ``G.T @ f``, motor torques for joint torques
    ``solve(G.T, tau)``."""
    idx = {jn: i for i, jn in enumerate(joints)}
    G = np.zeros((len(joints), len(joints)))
    for i, jn in enumerate(joints):
        side, _, rest = jn.partition("_")
        if rest == "ankle_pitch":
            G[i, i], G[i, idx[f"{side}_ankle_roll"]] = 1.0, ANKLE_MOTORS[0][1]
        elif rest == "ankle_roll":
            G[i, idx[f"{side}_ankle_pitch"]], G[i, i] = 1.0, ANKLE_MOTORS[1][1]
        else:
            G[i, i] = 1.0
    return G


def motor_torques(G: np.ndarray, tau) -> np.ndarray:
    """Motor torques that produce joint torques ``tau`` (rows of ``tau`` are frames, columns the hinges)."""
    return np.linalg.solve(G.T, np.asarray(tau, float).T).T


def sizing_motor_values(joints: list[str], tau, omega) -> tuple[np.ndarray, np.ndarray]:
    """Motor torque and speed of every drive (``drives`` order) for one frame of joint torques and speeds, read
    from ``sizing._motor_space`` (tandem layout): the sizing's own linkage map, for checking the plant against."""
    from .. import sizing

    ms = sizing._motor_space(joints, np.atleast_2d(tau), np.atleast_2d(omega), "tandem")
    f, w = np.zeros(len(joints)), np.zeros(len(joints))
    for i, jn in enumerate(joints):
        group = mjcf.joint_role(jn)
        k = statics.SIDES.index(jn.split("_", 1)[0]) if group not in ("neck", "tail") else 0
        f[i], w[i] = ms[group][0][k], ms[group][1][k]
    return f, w


class Envelope:
    """The torque-speed envelope of every drive of a plant, applied to its ``actuator_forcerange``.

    ``moment`` maps qvel to drive speeds (nu x nv; constant for joint and fixed-tendon transmissions).  ``bind``
    attaches a model and its data; ``update`` (before every ``mj_step``) then sets each force range from the
    current drive speeds.  Below every drive's corner speed the limits are the peaks, so the force ranges are only
    rewritten while some drive is past its corner (and once more on the way back)."""

    def __init__(self, drive_list: list[Drive], moment: np.ndarray, enabled: bool = True, speed_scale: float = 1.0,
                 torque_scale: float = 1.0):
        self.drives = list(drive_list)
        self.moment = np.ascontiguousarray(moment, float)
        self.speed_scale = float(speed_scale)
        self.torque_scale = float(torque_scale)
        self.peak = torque_scale * np.array([d.peak for d in drive_list])
        self.rated = speed_scale * np.array([d.rated_speed for d in drive_list])
        self.no_load = speed_scale * np.array([d.no_load_speed for d in drive_list])
        self._slope = self.peak / (self.no_load - self.rated)
        self._w = np.zeros(len(drive_list))
        self._neg = np.zeros(len(drive_list))
        self._fr = self._qvel = None
        self._dirty = False
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        self.restore()

    def limits(self, qvel) -> np.ndarray:
        """Torque limit of each drive at the joint velocities ``qvel``."""
        if not self._enabled:
            return self.peak.copy()
        return torque_limit(self.moment @ np.asarray(qvel, float), self.peak, self.rated, self.no_load)

    def bind(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        """Make every actuator of ``m`` force-limited at its (scaled) peak and keep views for ``update``."""
        m.actuator_forcelimited[:] = 1
        self._fr, self._qvel = m.actuator_forcerange, d.qvel
        self.restore()

    def restore(self) -> None:
        """Set every force range back to the peak (a plant at rest)."""
        if self._fr is not None:
            self._fr[:, 1] = self.peak
            self._fr[:, 0] = -self.peak
        self._dirty = False

    def update(self) -> None:
        """Set the bound model's force ranges from the bound data's drive speeds (call before every ``mj_step``)."""
        if not self._enabled:
            return
        w = np.dot(self.moment, self._qvel, out=self._w)
        np.abs(w, out=w)
        if (w > self.rated).any():
            np.subtract(self.no_load, w, out=w)
            np.multiply(self._slope, w, out=w)
            np.minimum(w, self.peak, out=w)
            np.maximum(w, 0.0, out=w)
            self._fr[:, 1] = w
            np.negative(w, out=self._neg)
            self._fr[:, 0] = self._neg
            self._dirty = True
        elif self._dirty:
            self.restore()

    def describe(self) -> dict:
        return dict(enabled=self.enabled, speed_scale=self.speed_scale, torque_scale=self.torque_scale,
                    rule="peak up to Actuator.speed_at(bus), then linear to zero at the no-load speed on the bus; "
                         "the same for driving and braking; set from each drive's speed before every physics step",
                    drives={d.actuator: dict(group=d.group, part=d.part, peak_Nm=float(p), rated_speed_rad_s=float(r),
                                             no_load_speed_rad_s=float(n))
                            for d, p, r, n in zip(self.drives, self.peak, self.rated, self.no_load, strict=True)})
