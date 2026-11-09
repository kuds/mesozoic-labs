"""Stage 1 state estimator: CoM and its velocity from deployable inputs only (``STAGE1_PLAN.md`` section 2,
controller R2).

Inputs, every physics step (2 ms), as ``plant.Plant.sensors`` gives them (the state at the step's start): the IMU's
gyro and accelerometer (the IMU site on the trunk), the joint encoders (angles and rates of the 14 hinges) and the
two foot loads.  The plant's ``imu_quat`` (an exact orientation) is not used.  ``SensorNoise`` corrupts them:
white noise on the gyro, accelerometer and joint rates, a constant bias per episode on the gyro and accelerometer
(drawn uniformly within the given bounds), encoder angles quantised to ``encoder_bits`` per turn, and white noise on
the foot loads.  ``STAGE1_PLAN.md`` randomises IMU noise but sources no values; the defaults are MEMS-class
assumptions (``NOISE_SOURCE``).  The estimator also knows the robot's own kinematic and inertial model (the plant's
MJCF) and the time step.

Method:

* Attitude: the gyro (less its bias estimate, zero by default) is integrated every step; every
  ``update`` (100 Hz) roll and pitch are pulled toward the tilt the loaded stance feet imply when they lie flat on
  the floor (the foot's orientation in the trunk frame from the encoders: the world vertical seen from the trunk), a
  complementary filter with gain ``k_leg_tilt`` (1/s), and weakly toward the accelerometer's gravity direction
  (``k_accel_tilt``).  Yaw is the gyro's alone (no magnetometer).
* Contacts: a foot is in contact while its load exceeds ``contact_on`` x the weight (and until it drops below
  ``contact_off`` x the weight).  At touchdown its position is set from the current base estimate and the encoders;
  it is then held fixed while in contact (leg odometry).
* Base position: the load-weighted mean over the contact feet of foot position minus the foot's offset from the
  base (orientation estimate and encoders).
* Velocity: the velocity the contact feet imply (a stance foot does not move: base velocity = minus the foot
  velocity the gyro and joint rates produce), load-weighted, at every ``update``.  An accelerometer-propagated
  complementary filter is available (``k_leg_velocity`` finite: the IMU point's velocity predicted with the
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

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .. import statics
from . import plant as plant_mod

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


class NoisySensors:
    """``plant.Plant.sensors`` with ``SensorNoise`` applied, from one seeded generator per episode."""

    def __init__(self, p: plant_mod.Plant, noise: SensorNoise, seed: int):
        self.p, self.noise = p, noise
        self.rng = np.random.default_rng(seed)
        s = noise.scale
        self.gyro_b = s * self.rng.uniform(-noise.gyro_bias, noise.gyro_bias, 3)
        self.accel_b = s * self.rng.uniform(-noise.accel_bias, noise.accel_bias, 3)
        self.q_step = 2 * math.pi / 2**noise.encoder_bits if noise.encoder_bits else 0.0
        self.weight = float(p.m.body_subtreemass[p.trunk]) * 9.81
        m, sd = p.m, p._sens
        self._gyro = sd["imu_gyro"][0]
        self._accel = sd["imu_accel"][0]
        self._q = np.array([sd[f"enc_q_{jn}"][0] for jn in p.joints])
        self._qd = np.array([sd[f"enc_qd_{jn}"][0] for jn in p.joints])
        self._load = np.array([sd[f"{side}_foot_load"][0] + 3 for side in statics.SIDES])
        del m

    def read(self) -> dict:
        n, s, rng = self.noise, self.noise.scale, self.rng
        d = self.p.d.sensordata
        gyro = d[self._gyro : self._gyro + 3].copy()
        accel = d[self._accel : self._accel + 3].copy()
        q, qd, load = d[self._q].copy(), d[self._qd].copy(), d[self._load].copy()
        if s:
            gyro += self.gyro_b + s * n.gyro_sigma * rng.standard_normal(3)
            accel += self.accel_b + s * n.accel_sigma * rng.standard_normal(3)
            qd += s * n.joint_rate_sigma * rng.standard_normal(len(qd))
            load += s * n.foot_load_sigma_frac * self.weight * rng.standard_normal(2)
        if self.q_step:
            q = np.round(q / self.q_step) * self.q_step
        return dict(gyro=gyro, accel=accel, q=q, qd=qd, foot_load=load)


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


@dataclass
class EstimatorGains:
    k_leg_tilt: float = 2.0  # 1/s
    k_accel_tilt: float = 0.05  # 1/s
    k_leg_velocity: float = math.inf  # 1/s; inf: the legs' velocity alone (module docstring)
    contact_on: float = 0.20  # of the weight
    contact_off: float = 0.10


class Estimator:
    """CoM and velocity from encoders, IMU and stance-leg kinematics (module docstring).

    ``step(reading)`` every physics step; ``update()`` at the feedback rate; then ``view`` holds the estimated
    kinematic data (``d``: an MjData of the plant's model at the estimated pose, kinematics, CoM and velocities
    computed) and the estimates."""

    def __init__(self, p: plant_mod.Plant, gains: EstimatorGains | None = None):
        self.p = p
        self.g = gains or EstimatorGains()
        self.m = p.m
        self.d = mujoco.MjData(p.m)
        self.dt = float(p.m.opt.timestep)
        self.weight = float(p.m.body_subtreemass[p.trunk]) * 9.81
        self.trunk, self.feet = p.trunk, p.feet
        imu = mujoco.mj_name2id(p.m, mujoco.mjtObj.mjOBJ_SITE, plant_mod.IMU_SITE)
        self.r_imu = p.m.site_pos[imu].copy()  # trunk frame (the site has no rotation)
        self.qadr, self.dofs = p.qadr, p.dofs
        self._res = np.zeros(6)
        self._kin(p.q_settled[:3], p.q_settled[3:7], p.q_settled[self.qadr])
        self.foot_z0 = float(np.mean([self.d.xpos[f][2] for f in self.feet])) - p.z_floor  # feet on the floor
        self.reset(p.d.qpos.copy(), np.zeros(p.m.nv))

    # ------------------------------------------------------------------ initialisation
    def reset(self, qpos, qvel, reading: dict | None = None) -> None:
        """Start from a known pose and velocity (the start's world frame)."""
        self.quat = np.array(qpos[3:7], float)
        self.base = np.array(qpos[:3], float)
        R = _mat(self.quat)
        r_imu_w = R @ self.r_imu
        w_world = R @ np.asarray(qvel[3:6], float)
        self.v_imu = np.asarray(qvel[:3], float) + np.cross(w_world, r_imu_w)
        self.q = np.array(qpos[self.qadr], float)
        self.qd = np.array(qvel[self.dofs], float)
        self.gyro = np.asarray(qvel[3:6], float).copy()
        self.accel = None if reading is None else reading["accel"]
        self.load = np.full(2, 0.5 * self.weight) if reading is None else reading["foot_load"]
        self._kin(self.base, self.quat, self.q)
        self.contact = self.load > self.g.contact_on * self.weight
        self.foot = np.array([self.d.xpos[f].copy() for f in self.feet])
        self.foot_yaw = np.array([self._yaw(self.d.xmat[f]) for f in self.feet])
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
    @staticmethod
    def _yaw(xmat) -> float:
        R = np.asarray(xmat).reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def _kin(self, base, quat, q, qvel=None) -> None:
        d = self.d
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qadr] = base, quat, q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        if qvel is not None:
            d.qvel[:] = qvel
            mujoco.mj_comVel(self.m, d)

    def _tilt_error(self, quat, R_foot_world_est) -> np.ndarray:
        """Body-frame rotation rate (per unit gain) that turns the estimated vertical toward the vertical a flat foot
        implies (the foot's z axis is the world's): k (z_measured x z_estimated), both in the trunk frame (with
        q' = q (x) w / 2 the estimated vertical then moves as z' = z_measured - z cos(angle))."""
        R = _mat(quat)
        z_est_body = R.T @ np.array([0.0, 0.0, 1.0])
        z_leg_body = R.T @ np.asarray(R_foot_world_est)[:, 2]  # the foot's z axis in the trunk frame
        return np.cross(z_leg_body, z_est_body)

    # ------------------------------------------------------------------ propagation
    def step(self, reading: dict) -> None:
        """One physics step's sensors: integrate the gyro and the accelerometer."""
        self.gyro, self.accel = reading["gyro"], reading["accel"]
        self.q, self.qd, self.load = reading["q"], reading["qd"], reading["foot_load"]
        R = _mat(self.quat)
        a_world = R @ self.accel + np.array([0.0, 0.0, -9.81])
        self.v_imu = self.v_imu + self.dt * a_world
        self.quat = _quat_integrate(self.quat, self.gyro, self.dt)

    def update(self, dt_update: float) -> None:
        """Corrections at the feedback rate (module docstring), then the CoM and the views."""
        g = self.g
        on = self.load > g.contact_on * self.weight
        off = self.load < g.contact_off * self.weight
        new_contact = np.where(self.contact, ~off, on)
        # tilt from the loaded flat feet, and the accelerometer
        self._kin(self.base, self.quat, self.q)
        corr = np.zeros(3)
        w = np.where(new_contact, np.maximum(self.load, 0.0), 0.0)
        if w.sum() > 0:
            for i, f in enumerate(self.feet):
                if w[i] > 0:
                    corr += (w[i] / w.sum()) * self._tilt_error(self.quat, self.d.xmat[f].reshape(3, 3))
            corr *= g.k_leg_tilt
        if g.k_accel_tilt and self.accel is not None and np.linalg.norm(self.accel) > 1e-6:
            R = _mat(self.quat)
            z_est = R.T @ np.array([0.0, 0.0, 1.0])
            corr += g.k_accel_tilt * np.cross(self.accel / np.linalg.norm(self.accel), z_est)
        self.quat = _quat_integrate(self.quat, corr, dt_update)
        # kinematics at the corrected attitude: feet, base, velocities (zero base translation)
        qvel = np.zeros(self.m.nv)
        qvel[3:6], qvel[self.dofs] = self.gyro, self.qd
        self._kin(self.base, self.quat, self.q, qvel)
        d = self.d
        feet_now = np.array([d.xpos[f].copy() for f in self.feet])
        yaw_now = np.array([self._yaw(d.xmat[f]) for f in self.feet])
        for i in range(2):
            if new_contact[i] and not self.contact[i]:  # touchdown: the foot is placed where the estimate puts it
                self.foot[i], self.foot_yaw[i] = feet_now[i], yaw_now[i]
            elif not new_contact[i]:
                self.foot[i], self.foot_yaw[i] = feet_now[i], yaw_now[i]
        self.contact = new_contact
        w = np.where(self.contact, np.maximum(self.load, 1e-6), 0.0)
        R = _mat(self.quat)
        r_imu_w = R @ self.r_imu
        w_world = R @ self.gyro
        if w.sum() > 0:
            w = w / w.sum()
            base, v_base = np.zeros(3), np.zeros(3)
            for i, f in enumerate(self.feet):
                if w[i] == 0:
                    continue
                base += w[i] * (self.foot[i] - (feet_now[i] - self.base))
                mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, f, self._res, 0)
                v_base += w[i] * (-self._res[3:])
            self.base = base
            v_imu_leg = v_base + np.cross(w_world, r_imu_w)
            gain = 1.0 if math.isinf(g.k_leg_velocity) else min(1.0, g.k_leg_velocity * dt_update)
            self.v_imu = self.v_imu + gain * (v_imu_leg - self.v_imu)
        self._finish()

    def _finish(self) -> None:
        """Kinematics, CoM and velocities at the estimate (the controller's view)."""
        R = _mat(self.quat)
        w_world = R @ self.gyro
        v_base = self.v_imu - np.cross(w_world, R @ self.r_imu)
        qvel = np.zeros(self.m.nv)
        qvel[:3], qvel[3:6], qvel[self.dofs] = v_base, self.gyro, self.qd
        self._kin(self.base, self.quat, self.q, qvel)
        mujoco.mj_subtreeVel(self.m, self.d)
        self.R = R
        self.omega_world = w_world
        self.com = self.d.subtree_com[self.trunk].copy()
        self.com_vel = self.d.subtree_linvel[self.trunk].copy()
        self.v_base = v_base
