"""Substep model for a delayed, torque/speed/effort-limited position drive.

The MJCF carries the installed envelope per axis. The rolling squared-effort
budget is a derating proxy, not a temperature estimate. Simulated joint damping
and friction act independently of these drive limits.
"""

from __future__ import annotations

import mujoco
import numpy as np


class InstalledDrive:
    """One instance per MjModel/environment; updates its actuator force bounds."""

    def __init__(self, model: mujoco.MjModel, voltage: float = 22.2) -> None:
        self.model = model
        self.voltage = voltage
        if not 20.0 <= voltage <= 25.2:
            raise ValueError("the modeled six-cell bus is supported from 20 to 25.2 V")
        self.peak = model.numeric("drive_peak_nm").data.copy()
        self.cont = model.numeric("drive_continuous_nm").data.copy()
        self.speed = model.numeric("drive_no_load_rad_s").data.copy()
        self.lash = model.numeric("drive_deadband_rad").data.copy() / 2
        self.traction = model.numeric("drive_traction").data.copy().astype(bool)
        self.reference_v, delay, window = model.numeric("drive_timing").data
        if any(len(x) != model.nu for x in (self.peak, self.cont, self.speed, self.lash, self.traction)):
            raise ValueError("one installed drive entry is required per actuator")
        if np.any(self.cont <= 0) or np.any(self.peak <= self.cont) or np.any(self.speed <= 0):
            raise ValueError("invalid continuous, peak or speed envelope")
        self.max_speed = self.speed * np.where(self.traction, voltage / self.reference_v, 1.0)
        self.dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.delay = int(round(delay / model.opt.timestep))
        self.queue = np.zeros((max(1, self.delay), model.nu))
        self.index = 0
        self.dt_over_capacity = model.opt.timestep / (window * (self.peak**2 - self.cont**2))
        self.used = np.zeros(model.nu)
        self.engaged_target = np.zeros(model.nu)

    def reset(self, data: mujoco.MjData) -> None:
        self.queue[:] = data.ctrl
        self.engaged_target[:] = data.ctrl
        self.index = 0
        self.used.fill(0.0)
        self.model.actuator_forcerange[:, 0] = -self.peak
        self.model.actuator_forcerange[:, 1] = self.peak

    def step(self, data: mujoco.MjData, target_rad: np.ndarray) -> None:
        """Apply one physics tick; never mutates the caller's target array."""
        if self.delay:
            delayed = self.queue[self.index].copy()
            self.queue[self.index] = target_rad
            self.index = (self.index + 1) % self.delay
        else:
            delayed = target_rad
        # Hysteretic command play models lost travel when reversing direction.
        self.engaged_target = np.clip(self.engaged_target, delayed - self.lash, delayed + self.lash)
        data.ctrl[:] = self.engaged_target
        velocity_ratio = data.qvel[self.dof] / self.max_speed
        available = self.peak - self.used * (self.peak - self.cont)
        # Motoring torque decreases towards no load. Braking can use the full
        # current limit; a physical bus still needs a separate energy sink.
        self.model.actuator_forcerange[:, 0] = -available * np.clip(1 + velocity_ratio, 0, 1)
        self.model.actuator_forcerange[:, 1] = available * np.clip(1 - velocity_ratio, 0, 1)
        mujoco.mj_step(self.model, data)
        self.used = np.clip(self.used + (data.actuator_force**2 - self.cont**2) * self.dt_over_capacity, 0, 1)
