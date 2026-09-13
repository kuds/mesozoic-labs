"""Physics-step measurements shared by the isolated foot experiments."""

from __future__ import annotations

import mujoco
import numpy as np

from .feet import foot_geom_groups, foot_sensor_groups, spectral_metrics


class FootProbe:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.groups = foot_geom_groups(model)
        self.sensors = foot_sensor_groups(model)
        self.ground = {model.geom("floor").id}
        self.obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_obstacle")
        if self.obstacle >= 0:
            self.ground.add(self.obstacle)
        self.pads = [model.geom(s + "_plantar_pad").id for s in ("r", "l")]
        self.indices = [
            i
            for i in range(model.nu)
            if model.actuator(i).name in ("r_toe_act", "l_toe_act", "r_ankle_act", "l_ankle_act")
        ]
        self.n = 0
        self.total_load = self.pad_load = self.air = self.bilateral = 0.0
        self.meaningful_bilateral = 0.0
        self.body_hit = False
        self.first_body_contact: dict | None = None
        self.max_sensor_error = self.max_self_force = self.max_ground_force = 0.0
        self.obstacle_samples = 0
        self.sensor_feet_seen = np.zeros(2, dtype=bool)
        self.sensor_digits_seen: set[str] = set()
        self.digit_sensors = [
            (model.sensor(i).name, int(model.sensor(i).adr[0]))
            for i in range(model.nsensor)
            if "_digits" in model.sensor(i).name and model.sensor(i).name.endswith("_touch")
        ]
        self.weight = float(model.body_subtreemass[model.body("pelvis").id] * np.linalg.norm(model.opt.gravity))
        self.saturation = np.zeros(len(self.indices))
        self.positive_work = 0.0
        self.pitch: list[list[float]] = []
        self.dominance: list[float] = []
        self.pelvis_omega_squared = 0.0
        self.last_loads = np.zeros(2)

    def sample(self, *, record: bool = True) -> np.ndarray:
        m, d = self.model, self.data
        loads = np.zeros(2)
        pad = 0.0
        obstacle_loaded = False
        for i in range(d.ncon):
            contact = d.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if contact.geom1 in self.ground or contact.geom2 in self.ground:
                other = contact.geom2 if contact.geom1 in self.ground else contact.geom1
                side = next((j for j, ids in enumerate(self.groups) if other in ids), None)
                if side is None:
                    self.body_hit = True
                    if self.first_body_contact is None:
                        self.first_body_contact = {
                            "time_s": float(d.time),
                            "geom": m.geom(other).name,
                            "normal_force_N": float(force[0]),
                            "distance_m": float(contact.dist),
                        }
                else:
                    loads[side] += force[0]
                    if other in self.pads:
                        pad += force[0]
                if self.obstacle in (contact.geom1, contact.geom2) and force[0] > 0.01:
                    obstacle_loaded = True
            elif record:
                self.max_self_force = max(self.max_self_force, float(force[0]))
        sensed = np.array([sum(d.sensordata[i] for i in group) for group in self.sensors])
        self.max_sensor_error = max(self.max_sensor_error, float(np.max(abs(sensed - loads))))
        self.sensor_feet_seen |= sensed > 0.1
        self.obstacle_samples += int(obstacle_loaded)
        for name, address in self.digit_sensors:
            if d.sensordata[address] > 0.01:
                self.sensor_digits_seen.add(name)
        self.last_loads = loads
        if not record:
            return loads
        self.n += 1
        self.total_load += loads.sum()
        self.pad_load += pad
        self.air += float(np.all(loads <= 0.1))
        self.bilateral += float(np.all(loads > 0.1))
        self.meaningful_bilateral += float(np.all(loads > 0.2 * self.weight))
        self.max_ground_force = max(self.max_ground_force, float(loads.sum()))
        self.saturation += abs(d.actuator_force[self.indices]) >= 0.99 * m.actuator_forcerange[self.indices, 1]
        self.positive_work += float(np.maximum(d.actuator_force * d.actuator_velocity, 0).sum() * m.opt.timestep)
        self.pelvis_omega_squared += float(np.dot(d.qvel[3:6], d.qvel[3:6]))
        self.pitch.append(
            [
                float(np.arctan2(-d.geom_xmat[g].reshape(3, 3)[2, 0], d.geom_xmat[g].reshape(3, 3)[0, 0]))
                for g in self.pads
            ]
        )
        self.dominance.append(float(np.sign(loads[0] - loads[1])) if loads.sum() > 0.1 else 0.0)
        return loads

    def results(self) -> dict:
        n = self.n
        dominates = np.asarray(self.dominance)
        switched = np.count_nonzero(dominates[1:] * dominates[:-1] < 0)
        return {
            "physics_samples": n,
            "physics_unsupported": self.air / n if n else None,
            "physics_bilateral": self.bilateral / n if n else None,
            "physics_bilateral_20pct_weight": self.meaningful_bilateral / n if n else None,
            "pad_load_fraction": self.pad_load / self.total_load if self.total_load else None,
            "support_switches_per_s": switched / (n * self.model.opt.timestep) if n else None,
            "sole_pitch_spectrum": spectral_metrics(np.asarray(self.pitch), self.model.opt.timestep),
            "positive_actuator_work_J": self.positive_work if n else None,
            "pelvis_angular_velocity_rms": float(np.sqrt(self.pelvis_omega_squared / n)) if n else None,
            "max_sensor_error_N": self.max_sensor_error,
            "max_self_contact_N": self.max_self_force if n else None,
            "max_ground_force_N": self.max_ground_force if n else None,
            "body_ground_contact": self.body_hit,
            "first_body_contact": self.first_body_contact,
            "obstacle_contact_samples": self.obstacle_samples,
            "both_feet_sensed": bool(np.all(self.sensor_feet_seen)),
            "digit_sensors_with_load": sorted(self.sensor_digits_seen),
            "actuator_saturation": {
                self.model.actuator(a).name: float(self.saturation[j] / n) if n else None
                for j, a in enumerate(self.indices)
            },
        }
