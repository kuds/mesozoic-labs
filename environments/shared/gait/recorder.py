"""Read-only recording at each physical substep, with safe hook restoration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .morphology import GaitMorphology

#: MuJoCo's instability warnings. ``mj_step`` that raises one resets the
#: simulation (``mj_resetData``: time 0, the model's qpos0) and continues, so
#: whatever the env reports afterwards is not this policy's motion.
_DIVERGENCE_WARNINGS = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")


class SubstepContactRecorder:
    """Attach after reset and detach before the next reset.

    Samples include the initial state and every post-``mj_step`` hook. Contact
    wrenches are MuJoCo's latest solved contacts; no ``mj_forward`` or second
    dynamics step is introduced. A previously installed hook (e.g. terrain
    probing) continues to run first. The recorder never changes physics,
    sensors, actions, observations, rewards, or random-generator state.

    A numerical divergence that MuJoCo answers with its automatic reset (an
    instability warning raised inside ``mj_step``) stops the recording at
    the last valid sample and sets ``diverged``; the trace carries
    ``physics_diverged`` so the episode fails as invalid telemetry rather than
    aborting the panel. Resetting the env while attached is still refused.
    """

    def __init__(self, env: Any, morphology: GaitMorphology):
        self.env = env.unwrapped
        self.morphology = morphology
        self._previous_hook: Any = None
        self._attached = False
        self._rows: list[dict[str, Any]] = []
        self._jacobian = np.zeros((3, morphology.model.nv), dtype=np.float64)
        self._contact_wrench = np.zeros(6, dtype=np.float64)
        self._terrain = frozenset(morphology.terrain_geom_ids)
        self._hook = self._record_substep
        self._warning_counts: tuple[int, ...] = ()
        self.diverged = False

    def __enter__(self) -> SubstepContactRecorder:
        if self._attached:
            raise RuntimeError("gait recorder is already attached")
        if self.env.model is not self.morphology.model:
            raise RuntimeError("gait morphology belongs to a different model")
        self._previous_hook = self.env._substep_probe_hook
        self.env._substep_probe_hook = self._hook
        self._attached = True
        self._warning_counts = self._divergence_counts()
        try:
            self.capture()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._attached:
            # A later owner replacing this hook must not be overwritten.
            if self.env._substep_probe_hook is self._hook:
                self.env._substep_probe_hook = self._previous_hook
            self._attached = False

    def _record_substep(self) -> None:
        if self._previous_hook is not None:
            self._previous_hook()
        if not self.diverged:
            self.capture()

    def _divergence_counts(self) -> tuple[int, ...]:
        import mujoco

        warnings = self.env.data.warning
        return tuple(int(warnings[int(getattr(mujoco.mjtWarning, name))].number) for name in _DIVERGENCE_WARNINGS)

    def _point_velocity(self, body: int, point: np.ndarray) -> np.ndarray:
        import mujoco

        if body == 0:
            return np.zeros(3)
        mujoco.mj_jac(self.env.model, self.env.data, self._jacobian, None, point, body)
        return np.asarray(self._jacobian @ self.env.data.qvel)

    def _surface_velocity(self, geom: int, point: np.ndarray) -> np.ndarray:
        model, data = self.env.model, self.env.data
        velocity = self._point_velocity(int(model.geom_bodyid[geom]), point)
        surface = getattr(model, "geom_surfacevel", None)
        if surface is not None:
            local = np.asarray(surface[geom])
            rotation = data.geom_xmat[geom].reshape(3, 3)
            velocity += rotation @ local[:3] + np.cross(rotation @ local[3:], point - data.geom_xpos[geom])
        return velocity

    def capture(self) -> None:
        import mujoco

        morph, model, data = self.morphology, self.env.model, self.env.data
        if model is not morph.model:
            raise RuntimeError("model changed while gait recorder was attached; resolve morphology after reset")
        counts = self._divergence_counts()
        if self._rows and counts != self._warning_counts and any(counts):
            # mj_step reset the state after an instability warning; the
            # samples so far are the episode's valid physics.
            self.diverged = True
            return
        if self._rows and float(data.time) <= self._rows[-1]["time_s"]:
            raise RuntimeError("gait sample time did not increase; detach the recorder before resetting")
        count = len(morph.foot_names)
        floor_force = np.zeros(count)
        weighted_slip_sq = np.zeros(count)
        body_force = 0.0
        foot_foot_force = 0.0
        # Animal contacts with neither terrain nor the animal itself (mocap
        # prey or props): never support, reported as a diagnostic.
        nonterrain_force = 0.0
        for ci in range(data.ncon):
            contact = data.contact[ci]
            if int(contact.efc_address) < 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 < 0 or g2 < 0:
                continue
            if g1 not in morph.animal_geom_ids and g2 not in morph.animal_geom_ids:
                continue
            mujoco.mj_contactForce(model, data, ci, self._contact_wrench)
            force = max(float(self._contact_wrench[0]), 0.0)
            if force <= 0:
                continue
            f1, f2 = int(morph.geom_foot[g1]), int(morph.geom_foot[g2])
            if f1 >= 0 and f2 >= 0 and f1 != f2:
                foot_foot_force += force
            if g1 in self._terrain and g2 in morph.animal_geom_ids:
                terrain, animal, foot = g1, g2, f2
            elif g2 in self._terrain and g1 in morph.animal_geom_ids:
                terrain, animal, foot = g2, g1, f1
            else:
                if (g1 in morph.animal_geom_ids) != (g2 in morph.animal_geom_ids):
                    nonterrain_force += force
                continue
            if foot < 0:
                body_force += force
                continue
            floor_force[foot] += force
            point = np.asarray(contact.pos)
            relative_velocity = self._point_velocity(int(model.geom_bodyid[animal]), point) - self._surface_velocity(
                terrain, point
            )
            normal = np.asarray(contact.frame[:3])
            tangential = relative_velocity - normal * float(relative_velocity @ normal)
            weighted_slip_sq[foot] += force * float(tangential @ tangential)
        slip = np.sqrt(np.divide(weighted_slip_sq, floor_force, out=np.zeros(count), where=floor_force > 0))
        clearance = np.asarray(
            [morph.foot_clearance(data, foot) for foot in range(count)],
            dtype=np.float64,
        )
        root_address = morph.root_qpos_address
        self._rows.append(
            {
                "time_s": float(data.time),
                "floor_force_n": floor_force,
                "foot_position_m": np.asarray(data.site_xpos[list(morph.foot_site_ids)]).copy(),
                "foot_clearance_m": clearance,
                "slip_speed_mps": slip,
                "body_floor_force_n": body_force,
                "foot_foot_force_n": foot_foot_force,
                "nonterrain_contact_force_n": nonterrain_force,
                "root_position_m": np.asarray(data.qpos[root_address : root_address + 3]).copy(),
                # Trunk orientation (free-joint quaternion, w x y z): step length
                # and lead exchange are measured in the trunk's own frame.
                "root_quat_wxyz": np.asarray(data.qpos[root_address + 3 : root_address + 7]).copy(),
                "touch_force_n": np.asarray([sum(data.sensordata[list(group)]) for group in morph.touch_addresses]),
            }
        )

    def trace(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("gait recorder contains no samples")
        arrays = {key: np.asarray([row[key] for row in self._rows]) for key in self._rows[0]}
        arrays["physics_diverged"] = np.asarray(self.diverged)
        return arrays
