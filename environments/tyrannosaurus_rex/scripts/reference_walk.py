"""Straight, level-ground anatomical T. rex reference walk.

Construct one ReferenceWalk after each reset. Call env.step(walker.action(env.data, env.dt))
at each control interval. The private inverse-kinematics workspace receives foot targets
and the live animal balances under its normal servos.
"""

import numpy as np

from environments.shared.gait.biped_walk import FOOTFALL_PHASE, BipedReferenceWalk


class ReferenceWalk(BipedReferenceWalk):
    """Shift weight between alternating feet and roll the planted foot onto the passive toes."""

    settle = 0.3
    yaw_lead = 0.7
    # The scaled plant targets 0.635 m/s. Period is seconds per same-foot
    # stride, and stride and all offsets below are metres.
    period = 1.04
    stride = 0.66
    stance_fraction = 0.65
    swing_height = 0.05
    walking_crouch = 0.05
    lateral_sway = 0.04
    medial_step = 0.055
    pad_suffix = "plantar_geom"
    cop_min = (-0.065, -0.035)
    cop_max = (0.105, 0.035)
    lateral_accel_gain = 1.35
    position_lead = 0.5
    velocity_lead = 0.25
    orientation_lead = 0.7
    angular_stiffness = 140.0
    angular_damping = 16.0
    lateral_stiffness = 15.0
    lateral_damping = 4.0
    ramp_cycles = 0.65
    rocker_radius = 0.05  # pivot anterior to the plantar pad centre, metres

    def __init__(self, env):
        super().__init__(env)
        self.roll_gain_fraction = 0.0
        self.hip_roll_dofs = np.array([self.model.joint(f"{side}_hip_roll").dofadr[0] for side in ("r", "l")])

    def _sway_amplitude(self, cycles):
        # The wide reset stance needs the larger initial weight shift.
        progress = np.clip((cycles - 1.0) / 4.0, 0.0, 1.0)
        return 0.06 + (self.lateral_sway - 0.06) * progress

    def _foot_pitch(self, phase, cycles):
        # Elevate the heel about the distal contact as the trailing foot unloads.
        progress = np.clip((phase - self.stance_fraction + 0.08) / 0.42, 0.0, 1.0)
        return 0.52 * np.sin(np.pi * progress) ** 2 * min(cycles, 1.0)

    def _targets(self, time):
        feet, root, velocity, lateral_accel, active = super()._targets(time)
        cycles = max(0.0, (time - self.start) / self.period)
        phase = np.remainder(cycles - FOOTFALL_PHASE, 1.0)
        # Drive the trailing leg backward during toe loading. The subsequent
        # swing releases the offset. Only the servo target follows this path.
        rise = np.clip((phase - 0.54) / 0.105, 0.0, 1.0)
        release = np.clip((phase - 0.7) / 0.2, 0.0, 1.0)
        retraction = 0.5 * (1.0 - np.cos(np.pi * rise))
        retraction *= 0.5 * (1.0 + np.cos(np.pi * release))
        feet[:, 0] -= 0.1 * retraction * min(cycles, 1.0)
        self.roll_gain_fraction = np.clip((cycles - 1.0) / 3.0, 0.0, 1.0)
        return feet, root, velocity, lateral_accel, active

    def _support(self, data, active, wrench):
        support = super()._support(data, active, wrench)
        loads = np.asarray(self.env._foot_contact_forces())
        total = loads.sum()
        if total > 12.0:  # N
            # Apply the roll correction through hips in proportion to the
            # measured support. Twice the quaternion x component approximates
            # the small roll angle on this straight walk.
            moment = 330.0 * (2.0 * data.qpos[4]) + 75.0 * data.qvel[3]
            support[self.hip_roll_dofs] -= moment * (loads / total) * self.roll_gain_fraction
        return support

    def _place_legs(self, data, feet, root, velocity):
        # Move the feet toward an incipient lateral fall. Twice the quaternion
        # x component approximates roll near the upright walking posture.
        feet[:, 1] -= 0.2 * (2 * data.qpos[4] + 0.15 * data.qvel[3])
        super()._place_legs(data, feet, root, velocity)

    def _distance(self, cycles):
        cycles = np.maximum(cycles, 0.0)
        return self.stride * (cycles - self.ramp_cycles * (1 - np.exp(-cycles / self.ramp_cycles)))

    def _next_landing(self, touchdown, landing):
        return self._distance(touchdown + 1 + self.stance_fraction / 2)

    def _speed(self, cycles):
        return (self.stride / self.period) * (1 - np.exp(-cycles / self.ramp_cycles))

    def _load_shares(self, planned):
        # The passive digits can keep a leg loaded past scheduled liftoff.
        # Use the measured split whenever the feet carry appreciable weight.
        loads = np.asarray(self.env._foot_contact_forces())
        weight = self.mass * -self.model.opt.gravity[2]
        return loads / loads.sum() if loads.sum() > 0.03 * weight else planned
