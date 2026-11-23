"""Turn shaping using operator requests and the plant's noisy IMU observations."""

import math

import numpy as np


class GentleTurn:
    """Relative yaw starts at zero in the parked robot's initial frame, with no external reference."""

    def __init__(self, env, *, gyro_bias_dps=0.0, stock_yaw=False):
        self.env = env
        self.stock_yaw = stock_yaw
        self.gyro_bias = math.radians(gyro_bias_dps)
        self.yaw = 0.0
        self.rate = 0.0
        self.pivoting = False
        self.previous_gyro_rate = None

    def gyro_target(self, slots, requested_deg, speed):
        _, gy, gz = slots["gravity_body"]
        _, wy, wz = slots["gyro"]
        roll = math.atan2(-gy, -gz)
        cos_pitch = max(0.5, math.hypot(gy, gz))
        # Euler yaw rate from the measured body-frame angular velocity and gravity direction.
        yaw_rate = (math.sin(roll) * wy + math.cos(roll) * (wz + self.gyro_bias)) / cos_pitch
        if self.previous_gyro_rate is not None:
            self.yaw += 0.5 * (self.previous_gyro_rate + yaw_rate) * self.env.dt
        self.previous_gyro_rate = yaw_rate

        error = math.remainder(math.radians(requested_deg) - self.yaw, 2 * math.pi)
        deadband = self.env.cfg.cmd_turn_in_place_deadband
        if speed == 0:
            if abs(error) > deadband:
                self.pivoting = True
            elif abs(error) < deadband / 2:
                self.pivoting = False
        forward = speed * max(self.env.cfg.cmd_min_turn_speed_fraction, math.cos(error))
        slots["command_speed_mps"][0] = forward
        slots["command"][0] = forward / self.env.cfg.cmd_speed_scale
        return self.env.cfg.cmd_yaw_gain * error if speed > 0 or self.pivoting else 0.0

    def apply(self, observation, requested_deg, speed, controller):
        """Return a controller observation with the adapted speed and yaw command slots."""
        obs = observation.copy()
        slots = self.env.obs_slots(obs)
        scale = self.env.cfg.cmd_yaw_rate_scale
        if self.stock_yaw:
            target = float(slots["command"][2]) * scale
        else:
            target = self.gyro_target(slots, requested_deg, speed)
        target = float(np.clip(target, -0.34, 0.34))
        step = 0.65 * self.env.dt
        self.rate += float(np.clip(target - self.rate, -step, step))
        # The inherited gait clock runs only when a speed or yaw command is nonzero.
        # Let the airborne pair finish the swing before parking all four feet.
        rate = self.rate
        if abs(rate) < 1e-8:
            rate = 1e-7 if any(controller.in_swing.values()) else 0.0
        slots["command"][2] = rate / scale
        return obs
