"""Stage 1 state estimator: CoM and its velocity from deployable inputs only (``STAGE1_PLAN.md`` section 2,
controller R2).

Inputs, every physics step (2 ms), as ``plant.Plant.sensors`` gives them (the state at the step's start): the IMU's
gyro and accelerometer (the IMU site on the trunk), the joint encoders (angles and rates of the 14 hinges) and the
two foot loads.  The plant's ``imu_quat`` (an exact orientation) is not used.  ``SensorNoise`` corrupts them:
white noise on the gyro, accelerometer and joint rates, a constant bias per episode on the gyro and accelerometer
(drawn uniformly within the given bounds), encoder angles quantised to ``encoder_bits`` per turn, and white noise on
the foot loads.  ``STAGE1_PLAN.md`` randomises IMU noise but sources no values; the defaults are MEMS-class
assumptions (``NOISE_SOURCE``).  The estimator also knows the robot's own kinematic and inertial model (the plant's
MJCF, or the model it is given: the nominal one when a PPO env randomises the plant's masses, a model error) and the
time step.

Method:

* Attitude: the gyro (less its bias estimate, zero by default) is integrated every step; every
  ``update`` (100 Hz) roll and pitch are pulled toward the tilt the loaded stance feet imply when they lie flat on
  the floor (the foot's orientation in the trunk frame from the encoders: the world vertical seen from the trunk), a
  complementary filter with gain ``k_leg_tilt`` (1/s), and weakly toward the accelerometer's gravity direction
  (``k_accel_tilt``).  Yaw is the gyro's alone (no magnetometer).
* Contacts: a foot is in contact while its load exceeds ``contact_on`` x the weight (and until it drops below
  ``contact_off`` x the weight).  The stance-leg odometry works at the loaded pad's contact point
  (``STAGE1_PLAN.md`` decision 19), not at the foot body's centre of mass, which sits about 11 mm above the floor
  and moves when the foot rocks on its pads while the contact does not: the contact point is the mean of the foot's
  pad bottom corners that lie within ``CONTACT_TOL`` of its lowest one, in the estimate (the whole sole when the foot
  lies flat, an edge or a pad when it rocks), a point of the foot.  At touchdown its position is set from the
  current base estimate and the encoders; it is then held fixed while in contact, and when the contact moves to
  other corners (the foot rolling) the held point moves with it along the foot.
* Base position: the load-weighted mean over the contact feet of the held contact point minus its offset from the
  base (orientation estimate and encoders); the feet the controller sees are the foot origins this implies.
* Velocity: the velocity the contact feet imply (a stance foot's contact point does not move: base velocity = minus
  the velocity of that point of the foot the gyro and joint rates produce), load-weighted, at every ``update``.
  An accelerometer-propagated complementary filter is available (``k_leg_velocity`` finite: the IMU point's velocity predicted with the
  accelerometer every step and pulled toward the legs' with that gain) but is off by default, because integrating
  the simulated accelerometer drifts.  Its gravity direction is right: at rest it reads R^T g to within 5e-4 m/s^2.
  But with MuJoCo's implicitfast integrator and the stiff servo damping, each step's change of the IMU point's
  velocity differs from the reading MuJoCo computed for that step (at its start, under its command, as
  ``NoisySensors`` reads it after the step) by 0.06-0.07 m/s^2 on average (forward) and 0.6-1.1 m/s^2 RMS per axis
  along the nominal walk (R1; 60-N-T-10 0.69 / 1.10 / 0.80, 60-N-H-10 0.67 / 1.00 / 0.79, 75-M-H-10 0.57 / 0.78 /
  0.60 m/s^2 x / y / z), so that the readings integrated over 1 s from the true velocity drift by up to 0.07 m/s on
  the 60 cm bodies and 0.15 m/s on 75-M-H-10, whereas the legs' velocity alone is within 4-5 mm/s RMS without noise
  and 8-9 mm/s RMS with the default noise (forward and sideways, the CoM).  (The build first compared each velocity
  change with a reading taken before that step's command was set, under the previous step's command, and read 0.35
  m/s^2 on average and 2-5 m/s^2 RMS.)
* CoM: the model's CoM at the estimated base pose and the encoder angles; its velocity from the same kinematics with
  the estimated base velocity, the gyro and the joint rates.

The controller needs the CoM relative to the stance foot and the feet relative to each other, both in the estimate's
own world frame, so the absolute drift of the position estimate does not enter it.  The world frame is the start's:
``reset`` takes the start pose (and velocity, for a start from a moving reference frame) as known, or ``reset_at_rest``
initialises from the sensors with the body at rest on both feet.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .. import statics
from . import plant as plant_mod

#: Decision 19: a pad bottom corner within this of the foot's lowest one (in the estimate) is in contact.
CONTACT_TOL = 1e-3
#: The stance-leg odometry's point of a contact foot (``EstimatorGains.stance_point``): decision 19's, the loaded
#: pad's contact point (the default), or, for comparison, the foot body's centre of mass, as the build first had it.
STANCE_POINTS = ("loaded_pad_contact_point", "foot_com")
_KIN_IDS = itertools.count(1)  # Estimator._kv: unique per kinematics computation
_E_Z = np.array([0.0, 0.0, 1.0])
_GRAVITY = np.array([0.0, 0.0, -9.81])

NOISE_SOURCE = ("STAGE1_PLAN.md randomises IMU noise (section 2) without values; assumed MEMS-class figures: gyro "
                "white noise 0.003 rad/s per 2 ms sample (0.17 deg/s, about 0.011 deg/s/sqrt(Hz)) and a residual bias "
                "after a standstill calibration within +-2e-4 rad/s (about 40 deg/h); accelerometer 0.03 m/s^2 per "
                "sample (about 0.2 mg/sqrt(Hz)) and a bias within +-0.05 m/s^2; 14-bit absolute joint encoders; "
                "joint-rate noise 0.02 rad/s; foot-load noise 2 % of the weight")


@dataclass
class SensorNoise:
    """Sensor corruption per episode (module docstring).  ``scale`` multiplies every noise and bias term."""

    gyro_sigma: float = 0.003  # rad/s per sample
    gyro_bias: float = 2e-4  # rad/s, per-axis bound
    accel_sigma: float = 0.03  # m/s^2 per sample
    accel_bias: float = 0.05  # m/s^2, per-axis bound
    encoder_bits: int = 14
    joint_rate_sigma: float = 0.02  # rad/s
    foot_load_sigma_frac: float = 0.02  # of the weight
    scale: float = 1.0

    @classmethod
    def none(cls) -> "SensorNoise":
        return cls(scale=0.0, encoder_bits=0)

    def describe(self) -> dict:
        return dict(asdict(self), source=NOISE_SOURCE)


MjViews = plant_mod.MjViews


def _span(adr) -> slice | np.ndarray:
    """A slice for consecutive addresses (a view: the same values as the index array's copy), else the array."""
    adr = np.asarray(adr, int)
    if len(adr) and np.array_equal(adr, np.arange(adr[0], adr[0] + len(adr))):
        return slice(int(adr[0]), int(adr[0]) + len(adr))
    return adr


class NoisySensors:
    """``plant.Plant.sensors`` with ``SensorNoise`` applied, from one seeded generator per episode."""

    def __init__(self, p: plant_mod.Plant, noise: SensorNoise, seed: int, weight: float | None = None):
        """``weight`` (N) scales the foot-load noise (default: the plant model's weight; a controller with a model of
        its own passes that model's: the sensors' noise does not follow a randomised mass)."""
        self.p, self.noise = p, noise
        self.rng = np.random.default_rng(seed)
        s = noise.scale
        self.gyro_b = s * self.rng.uniform(-noise.gyro_bias, noise.gyro_bias, 3)
        self.accel_b = s * self.rng.uniform(-noise.accel_bias, noise.accel_bias, 3)
        self.q_step = 2 * math.pi / 2**noise.encoder_bits if noise.encoder_bits else 0.0
        self.weight = float(p.m.body_subtreemass[p.trunk]) * 9.81 if weight is None else float(weight)
        m, sd = p.m, p._sens
        self._gyro = sd["imu_gyro"][0]
        self._accel = sd["imu_accel"][0]
        self._q = np.array([sd[f"enc_q_{jn}"][0] for jn in p.joints])
        self._qd = np.array([sd[f"enc_qd_{jn}"][0] for jn in p.joints])
        self._load = np.array([sd[f"{side}_foot_load"][0] + 3 for side in statics.SIDES])
        self._sd = p.d.sensordata
        self._qs = _span(self._q)
        nj = len(self._qd)
        #: The noisy channels gathered at once: gyro and accelerometer (with their biases), then rates and loads.
        self._imu_idx = np.r_[np.arange(self._gyro, self._gyro + 3), np.arange(self._accel, self._accel + 3)]
        self._rl_idx = np.r_[self._qd, self._load]
        self._nj = nj
        self._scales = None  # (the noise's scale factors they were made from, the 6- and 16-value scale vectors)
        del m

    def _scale_vectors(self) -> tuple:
        n, s = self.noise, self.noise.scale
        sg, sa = s * n.gyro_sigma, s * n.accel_sigma
        sq, sl = s * n.joint_rate_sigma, s * n.foot_load_sigma_frac * self.weight
        key = (sg, sa, sq, sl)
        if self._scales is None or self._scales[0] != key:
            nj = self._nj
            self._scales = (key, np.array([sg, sg, sg, sa, sa, sa]), np.r_[np.full(nj, sq), sl, sl],
                            np.r_[self.gyro_b, self.accel_b])
        return self._scales

    def read(self) -> dict:
        """The readings, with the same operations as one channel at a time: reading + (bias + (s sigma) z) for the
        gyro and the accelerometer, reading + (s sigma) z for the joint rates and the foot loads (each product and sum
        elementwise, so the same bits), the angles rounded to the encoder step.  The noise is drawn in one call of
        3 + 3 + 14 + 2 normals, which gives the same numbers in the same order (and leaves the generator in the same
        state) as calls of 3, 3, 14 and 2.  (The accelerometer, which meets BLAS, is returned as an array of its own,
        as before.)"""
        d = self._sd
        nj = self._nj
        imu = d[self._imu_idx]
        rl = d[self._rl_idx]
        if self.noise.scale:
            _, c6, c16, bias = self._scale_vectors()
            z = self.rng.standard_normal(8 + nj)
            imu += bias + c6 * z[:6]
            rl += c16 * z[6:]
        q = d[self._qs]
        q = (q / self.q_step).round() * self.q_step if self.q_step else q.copy()  # (np.round's own call)
        return dict(gyro=imu[:3], accel=imu[3:].copy(), q=q, qd=rl[:nj], foot_load=rl[nj:])


def cross3(a, b) -> np.ndarray:
    """``np.cross(a, b)`` of a 3-vector ``a`` and a 3-vector (or rows of 3-vectors) ``b`` by numpy's own formula (each
    component a1 b2 - a2 b1 and so on: the same products and differences, so the same bits), without its overhead."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b = np.asarray(b, float)
    if b.ndim == 1:
        b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
        return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
    return np.column_stack((a1 * b[:, 2] - a2 * b[:, 1], a2 * b[:, 0] - a0 * b[:, 2], a0 * b[:, 1] - a1 * b[:, 0]))


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _quat_integrate(q, w_body, dt):
    out = np.array(q, float)
    mujoco.mju_quatIntegrate(out, np.asarray(w_body, float), dt)
    mujoco.mju_normalize4(out)
    return out


def _mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def sole_corners(m, feet) -> list[np.ndarray]:
    """Per foot body: the bottom corners of its pads (its box geoms named ``*_pad_*``) in the foot's frame."""
    out = []
    for f in feet:
        pts = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if m.geom_bodyid[g] != f or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or "_pad_" not in name:
                continue
            R = _mat(m.geom_quat[g])
            h = m.geom_size[g]
            pts += [m.geom_pos[g] + R @ np.array([sx * h[0], sy * h[1], -h[2]]) for sx in (-1, 1) for sy in (-1, 1)]
        if not pts:
            raise ValueError(f"body {f} has no pads")
        out.append(np.array(pts))
    return out


@dataclass
class EstimatorGains:
    k_leg_tilt: float = 2.0  # 1/s
    k_accel_tilt: float = 0.05  # 1/s
    k_leg_velocity: float = math.inf  # 1/s; inf: the legs' velocity alone (module docstring)
    contact_on: float = 0.20  # of the weight
    contact_off: float = 0.10
    stance_point: str = STANCE_POINTS[0]  # decision 19


class Estimator:
    """CoM and velocity from encoders, IMU and stance-leg kinematics (module docstring).

    ``step(reading)`` every physics step; ``update()`` at the feedback rate; then ``view`` holds the estimated
    kinematic data (``d``: an MjData of the estimator's model at the estimated pose, kinematics, CoM and velocities
    computed) and the estimates.  ``model`` is the robot model the estimator knows (default: the plant's own; it must
    have the plant's structure): given the nominal model while the plant's masses are randomised, its CoM is the
    nominal model's at the estimated pose, a model error."""

    def __init__(self, p: plant_mod.Plant, gains: EstimatorGains | None = None, model: mujoco.MjModel | None = None):
        self.p = p
        self.g = gains or EstimatorGains()
        if self.g.stance_point not in STANCE_POINTS:
            raise ValueError(f"stance point {self.g.stance_point!r}: choose from {', '.join(STANCE_POINTS)}")
        m = p.m if model is None else model
        if model is not None and any(getattr(model, k) != getattr(p.m, k) for k in ("nq", "nv", "nbody", "ngeom",
                                                                                      "nsite", "njnt")):
            raise ValueError("the estimator's model must have the plant's structure (nq, nv, bodies, geoms, sites, "
                             "joints)")
        self.m = m
        self.d = mujoco.MjData(m)
        self._views = MjViews(self.d)
        self._qs, self._vs = _span(p.qadr), _span(p.dofs)
        self._kv = 0  # the kinematics' version: a new number at every _kin (control's cap reuses them between)
        self._Rb = np.zeros(9)  # step's attitude matrix buffer
        self.dt = float(m.opt.timestep)
        self.weight = float(m.body_subtreemass[p.trunk]) * 9.81
        self.trunk, self.feet = p.trunk, p.feet
        imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, plant_mod.IMU_SITE)
        self.r_imu = m.site_pos[imu].copy()  # trunk frame (the site has no rotation)
        self.qadr, self.dofs = p.qadr, p.dofs
        self.sole = sole_corners(m, self.feet)  # per foot: its pads' bottom corners in the foot frame
        self._res = np.zeros(6)
        self._kin(p.q_settled[:3], p.q_settled[3:7], p.q_settled[self.qadr])
        self.foot_z0 = float(np.mean([self.d.xpos[f][2] for f in self.feet])) - p.z_floor  # feet on the floor
        self.reset(p.d.qpos.copy(), np.zeros(m.nv))

    # ------------------------------------------------------------------ initialisation
    def reset(self, qpos, qvel, reading: dict | None = None) -> None:
        """Start from a known pose and velocity (the start's world frame)."""
        self.quat = np.array(qpos[3:7], float)
        self.base = np.array(qpos[:3], float)
        R = _mat(self.quat)
        r_imu_w = R @ self.r_imu
        w_world = R @ np.asarray(qvel[3:6], float)
        self.v_imu = np.asarray(qvel[:3], float) + cross3(w_world, r_imu_w)
        self.q = np.array(qpos[self.qadr], float)
        self.qd = np.array(qvel[self.dofs], float)
        self.gyro = np.asarray(qvel[3:6], float).copy()
        self.accel = None if reading is None else reading["accel"]
        self.load = np.full(2, 0.5 * self.weight) if reading is None else reading["foot_load"]
        self._kin(self.base, self.quat, self.q)
        self.contact = self.load > self.g.contact_on * self.weight
        self.foot = np.array([self.d.xpos[f].copy() for f in self.feet])
        self.foot_yaw = np.array([self._yaw(self.d.xmat[f]) for f in self.feet])
        self.contact_local = np.array([self.contact_point(i) for i in range(2)])
        self.contact_world = np.array([self._point_world(i, self.contact_local[i]) for i in range(2)])
        self._finish()

    def reset_at_rest(self, reading: dict, yaw: float = 0.0, base_xy=(0.0, 0.0)) -> None:
        """Initialise from the sensors with the body at rest on both feet: roll and pitch from the feet's tilt (both
        flat), yaw and the horizontal position given (the world frame), height from the feet on the floor plane."""
        q = np.asarray(reading["q"], float)
        quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        for _ in range(3):  # roll and pitch from the feet lying flat (two passes converge)
            self._kin(np.zeros(3), quat, q)
            corr = np.zeros(3)
            for f in self.feet:
                corr += self._tilt_error(quat, self.d.xmat[f].reshape(3, 3))
            quat = _quat_integrate(quat, 0.5 * corr, 1.0)
        self._kin(np.zeros(3), quat, q)
        z_feet = np.mean([self.d.xpos[f][2] for f in self.feet])
        base = np.array([base_xy[0], base_xy[1], self.p.z_floor + self.foot_z0 - z_feet])
        qpos = np.zeros(self.m.nq)
        qpos[:3], qpos[3:7], qpos[self.qadr] = base, quat, q
        self.reset(qpos, np.zeros(self.m.nv), reading)

    # ------------------------------------------------------------------ helpers
    def contact_point(self, i: int) -> np.ndarray:
        """Decision 19: foot ``i``'s contact point in its own frame at the current estimated pose: the mean of its pad
        bottom corners within ``CONTACT_TOL`` of the lowest one (``stance_point`` "foot_com": the foot body's centre of
        mass, for comparison)."""
        if self.g.stance_point == "foot_com":
            return self.m.body_ipos[self.feet[i]].copy()
        z = self.sole[i] @ self.d.xmat[self.feet[i]].reshape(3, 3)[2]
        return self.sole[i][z <= z.min() + CONTACT_TOL].mean(0)

    def _point_world(self, i: int, local: np.ndarray) -> np.ndarray:
        f = self.feet[i]
        return self.d.xpos[f] + self.d.xmat[f].reshape(3, 3) @ local

    @staticmethod
    def _yaw(xmat) -> float:
        R = np.asarray(xmat).reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def _kin(self, base, quat, q, qvel=None) -> None:
        d, v = self.d, self._views
        v.qpos[:3], v.qpos[3:7], v.qpos[self._qs] = base, quat, q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        if qvel is not None:
            v.qvel[:] = qvel
            mujoco.mj_comVel(self.m, d)
        self._kv = next(_KIN_IDS)

    def _kin_vel(self, base, quat, q, v_base, gyro, qd) -> None:
        """``_kin`` with the velocity [v_base, gyro, qd] (v_base None: zero), written into the data directly."""
        d, v = self.d, self._views
        v.qpos[:3], v.qpos[3:7], v.qpos[self._qs] = base, quat, q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        qv = v.qvel
        qv[:] = 0.0
        if v_base is not None:
            qv[:3] = v_base
        qv[3:6] = gyro
        qv[self._vs] = qd
        mujoco.mj_comVel(self.m, d)
        self._kv = next(_KIN_IDS)

    def _tilt_error(self, quat, R_foot_world_est) -> np.ndarray:
        """Body-frame rotation rate (per unit gain) that turns the estimated vertical toward the vertical a flat foot
        implies (the foot's z axis is the world's): k (z_measured x z_estimated), both in the trunk frame (with
        q' = q (x) w / 2 the estimated vertical then moves as z' = z_measured - z cos(angle))."""
        R = _mat(quat)
        z_est_body = R.T @ np.array([0.0, 0.0, 1.0])
        z_leg_body = R.T @ np.asarray(R_foot_world_est)[:, 2]  # the foot's z axis in the trunk frame
        return cross3(z_leg_body, z_est_body)

    # ------------------------------------------------------------------ propagation
    def step(self, reading: dict) -> None:
        """One physics step's sensors: integrate the gyro and the accelerometer."""
        self.gyro, self.accel = reading["gyro"], reading["accel"]
        self.q, self.qd, self.load = reading["q"], reading["qd"], reading["foot_load"]
        Rb = self._Rb
        mujoco.mju_quat2Mat(Rb, self.quat)  # (_mat into a buffer of the same layout)
        a_world = Rb.reshape(3, 3) @ self.accel + _GRAVITY
        self.v_imu = self.v_imu + self.dt * a_world
        quat = self.quat  # (_quat_integrate in place: the estimator's own array)
        mujoco.mju_quatIntegrate(quat, self.gyro, self.dt)
        mujoco.mju_normalize4(quat)

    def update(self, dt_update: float) -> None:
        """Corrections at the feedback rate (module docstring), then the CoM and the views.

        (The two-element logic runs in Python floats with numpy's semantics -- comparisons, ``np.maximum``, sums of
        two -- and every product with a matrix stays numpy's; the attitude matrix and its products with the gyro and
        the IMU offset, needed again by the view at the same attitude, are computed once.)"""
        g = self.g
        m, d, v = self.m, self.d, self._views
        weight = self.weight
        on_thr, off_thr = g.contact_on * weight, g.contact_off * weight
        loads = self.load.tolist()
        was = [bool(c) for c in self.contact]
        nc = [(not lv < off_thr) if c else (lv > on_thr) for lv, c in zip(loads, was, strict=True)]
        new_contact = np.array(nc)
        # tilt from the loaded flat feet, and the accelerometer
        self._kin(self.base, self.quat, self.q)
        corr = np.zeros(3)
        w = [(lv if (lv >= 0.0 or lv != lv) else 0.0) if c else 0.0 for lv, c in zip(loads, nc, strict=True)]
        ws = w[0] + w[1]
        R = z_est = None
        if ws > 0:
            R = _mat(self.quat)
            z_est = R.T @ _E_Z
            for i, f in enumerate(self.feet):
                if w[i] > 0:
                    z_leg = R.T @ v.xmat[f].reshape(3, 3)[:, 2]  # the foot's z axis in the trunk frame
                    corr += (w[i] / ws) * cross3(z_leg, z_est)
            corr *= g.k_leg_tilt
        if g.k_accel_tilt and self.accel is not None:
            na = np.linalg.norm(self.accel)
            if na > 1e-6:
                if R is None:
                    R = _mat(self.quat)
                    z_est = R.T @ _E_Z
                corr += g.k_accel_tilt * cross3(self.accel / na, z_est)
        self.quat = _quat_integrate(self.quat, corr, dt_update)
        # kinematics at the corrected attitude: feet, base, velocities (zero base translation)
        self._kin_vel(self.base, self.quat, self.q, None, self.gyro, self.qd)
        xmat, xpos = v.xmat, v.xpos
        point_now = np.zeros((2, 3))  # each foot's contact point where the kinematics put it (base at self.base)
        for i in range(2):
            f = self.feet[i]
            local = self.contact_point(i)
            point_now[i] = xpos[f] + xmat[f].reshape(3, 3) @ local
            if nc[i] and was[i]:  # in contact: held, moving along the foot with the contact
                if not np.array_equal(local, self.contact_local[i]):
                    R_f = xmat[f].reshape(3, 3)
                    self.contact_world[i] = self.contact_world[i] + R_f @ (local - self.contact_local[i])
            else:  # touchdown (placed where the estimate puts it) or swing
                self.contact_world[i] = point_now[i]
                self.foot_yaw[i] = math.atan2(xmat[f][3], xmat[f][0])
            self.contact_local[i] = local
        self.contact = new_contact
        w = [(lv if (lv >= 1e-6 or lv != lv) else 1e-6) if c else 0.0 for lv, c in zip(loads, nc, strict=True)]
        R = _mat(self.quat)
        r_imu_w = R @ self.r_imu
        w_world = R @ self.gyro
        ws = w[0] + w[1]
        if ws > 0:
            w = [w[0] / ws, w[1] / ws]
            base, v_base = np.zeros(3), np.zeros(3)
            res = self._res
            for i, f in enumerate(self.feet):
                if w[i] == 0:
                    continue
                base += w[i] * (self.contact_world[i] - (point_now[i] - self.base))
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, f, res, 0)  # at the foot origin
                r = point_now[i] - xpos[f]
                v_base += w[i] * -(res[3:] + cross3(res[:3], r))  # the contact point's velocity
            shift = base - self.base
            self.base = base
            v_imu_leg = v_base + cross3(w_world, r_imu_w)
            gain = 1.0 if math.isinf(g.k_leg_velocity) else min(1.0, g.k_leg_velocity * dt_update)
            self.v_imu = self.v_imu + gain * (v_imu_leg - self.v_imu)
        else:
            shift = np.zeros(3)
        for i, f in enumerate(self.feet):  # the foot origins the controller sees: a contact foot's from its held point
            self.foot[i] = (self.contact_world[i] - xmat[f].reshape(3, 3) @ self.contact_local[i] if nc[i]
                            else xpos[f] + shift)
        self._finish(R, w_world, r_imu_w)

    def _finish(self, R=None, w_world=None, r_imu_w=None) -> None:
        """Kinematics, CoM and velocities at the estimate (the controller's view); ``R``, ``w_world`` and ``r_imu_w``
        are the attitude matrix and its products with the gyro and the IMU offset when the caller has them (the
        same computations at the same attitude)."""
        if R is None:
            R = _mat(self.quat)
            w_world = R @ self.gyro
            r_imu_w = R @ self.r_imu
        v_base = self.v_imu - cross3(w_world, r_imu_w)
        self._kin_vel(self.base, self.quat, self.q, v_base, self.gyro, self.qd)
        mujoco.mj_subtreeVel(self.m, self.d)
        v = self._views
        self.R = R
        self.omega_world = w_world
        self.com = v.subtree_com[self.trunk].copy()
        self.com_vel = v.subtree_linvel[self.trunk].copy()
        self.v_base = v_base
