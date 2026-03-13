"""Shared helpers and base test classes for static balance tests.

Species-specific test files inherit from these bases, overriding only
the configuration (body names, thresholds, joint lists).
"""

import mujoco
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_foot_contacts_xy(model: mujoco.MjModel, data: mujoco.MjData, foot_geom_names: list[str]) -> np.ndarray:
    """Return (N, 2) array of foot-floor contact positions in the XY plane."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_geom_names}

    points = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 == floor_id and g2 in foot_ids) or (g2 == floor_id and g1 in foot_ids):
            points.append(c.pos[:2].copy())
    return np.array(points) if points else np.empty((0, 2))


def com_xy(model: mujoco.MjModel, data: mujoco.MjData, root_body: str) -> np.ndarray:
    """Return the species' center of mass projected onto the XY plane."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    return np.array(data.subtree_com[body_id, :2])


def species_mass(model: mujoco.MjModel, root_body: str) -> float:
    """Return the total mass of the species (root subtree), excluding worldbody."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    return float(model.body_subtreemass[body_id])


def body_group_mass(model: mujoco.MjModel, body_names: list[str]) -> float:
    """Sum the mass of a list of named bodies."""
    return float(sum(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in body_names))


# ---------------------------------------------------------------------------
# Base test classes
# ---------------------------------------------------------------------------


class JointLimitsAtHomeBase:
    """Verify the home keyframe doesn't violate joint limits.

    Subclasses must provide:
        - ``env`` fixture
        - ``knee_names``: list of knee joint names
        - ``knee_margin_deg``: minimum margin in degrees
    """

    knee_names: list[str] = []
    knee_margin_deg: float = 20.0

    def test_no_joint_limit_violations(self, env):
        """Every joint in the home keyframe should be within its declared range."""
        violations = []
        for j in range(env.model.njnt):
            if env.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                continue

            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            qadr = env.model.jnt_qposadr[j]
            qval = env.data.qpos[qadr]
            limited = env.model.jnt_limited[j]

            if limited:
                lo = env.model.jnt_range[j, 0]
                hi = env.model.jnt_range[j, 1]
                if qval < lo - 1e-4 or qval > hi + 1e-4:
                    violations.append(
                        f"  {name}: qpos={np.degrees(qval):.2f} deg "
                        f"range=[{np.degrees(lo):.1f} deg, {np.degrees(hi):.1f} deg]"
                    )

        assert not violations, "Home keyframe violates joint limits:\n" + "\n".join(violations)

    def test_knees_flexed_at_home(self, env):
        """Knees should be flexed (away from limits) with room to absorb impact."""
        margin_rad = np.radians(self.knee_margin_deg)
        for knee_name in self.knee_names:
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, knee_name)
            qadr = env.model.jnt_qposadr[jid]
            knee_angle = env.data.qpos[qadr]
            lo = env.model.jnt_range[jid, 0]
            hi = env.model.jnt_range[jid, 1]

            extension_margin = hi - knee_angle
            flexion_margin = knee_angle - lo

            assert extension_margin > margin_rad, (
                f"{knee_name} angle {np.degrees(knee_angle):.1f} deg is too close to "
                f"extension limit {np.degrees(hi):.1f} deg — only "
                f"{np.degrees(extension_margin):.1f} deg of extension travel."
            )
            assert flexion_margin > margin_rad, (
                f"{knee_name} angle {np.degrees(knee_angle):.1f} deg is too close to "
                f"flexion limit {np.degrees(lo):.1f} deg — only "
                f"{np.degrees(flexion_margin):.1f} deg of flexion travel."
            )


class MassDistributionBase:
    """Sanity-check the mass distribution.

    Subclasses must provide:
        - ``env`` fixture
        - ``root_body``: name of the root body
        - ``mass_range``: (min, max) total mass in kg
        - ``leg_body_names``: list of leg body names
        - ``min_leg_fraction``: minimum leg mass fraction
        - ``tail_body_names``: list of tail body names
        - ``max_tail_fraction``: maximum tail mass fraction
        - ``symmetry_pairs``: list of (right_name, left_name) tuples
    """

    root_body: str = ""
    mass_range: tuple[float, float] = (0, 0)
    leg_body_names: list[str] = []
    min_leg_fraction: float = 0.15
    tail_body_names: list[str] = []
    max_tail_fraction: float = 0.30
    symmetry_pairs: list[tuple[str, str]] = []

    def test_total_mass_reasonable(self, env):
        total_mass = species_mass(env.model, self.root_body)
        lo, hi = self.mass_range
        assert lo < total_mass < hi, f"Total mass is {total_mass:.1f} kg — expected {lo}-{hi} kg for this scale."

    def test_leg_mass_fraction(self, env):
        total_mass = species_mass(env.model, self.root_body)
        leg_mass = body_group_mass(env.model, self.leg_body_names)
        fraction = leg_mass / total_mass
        assert fraction > self.min_leg_fraction, (
            f"Leg mass fraction is {fraction:.1%} — legs should be at least {self.min_leg_fraction:.0%} of total mass."
        )

    def test_tail_mass_not_excessive(self, env):
        total_mass = species_mass(env.model, self.root_body)
        tail_mass = body_group_mass(env.model, self.tail_body_names)
        fraction = tail_mass / total_mass
        assert fraction < self.max_tail_fraction, (
            f"Tail mass fraction is {fraction:.1%} — exceeds {self.max_tail_fraction:.0%} threshold."
        )

    def test_left_right_symmetry(self, env):
        for rn, ln in self.symmetry_pairs:
            r_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, rn)
            l_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, ln)
            r_mass = env.model.body_mass[r_id]
            l_mass = env.model.body_mass[l_id]
            assert abs(r_mass - l_mass) < 1e-6, f"Mass asymmetry: {rn}={r_mass:.4f} kg vs {ln}={l_mass:.4f} kg"


class ZeroTorqueStabilityBase:
    """The home pose with zero control should not immediately collapse.

    Subclasses must provide:
        - ``env`` fixture
        - ``species_name``: display name for error messages
        - ``root_body_id_attr``: attribute name on env for the root body ID (e.g. "pelvis_id")
        - ``max_height_drop``: max allowed height drop in meters over 30 steps
    """

    species_name: str = ""
    root_body_id_attr: str = "pelvis_id"
    max_height_drop: float = 0.10

    def test_survives_100_zero_torque_steps(self, env):
        env.reset(seed=0)
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for step in range(100):
            _, _, terminated, _, info = env.step(zero_action)
            if terminated:
                reason = info.get("termination_reason", "unknown")
                pelvis_z = info.get("pelvis_height", "?")
                tilt = info.get("tilt_angle", "?")
                pytest.fail(
                    f"{self.species_name} fell at step {step + 1}/100 under zero torque. "
                    f"Reason: {reason}, height: {pelvis_z}, tilt: {tilt}. "
                    "The home keyframe may not be statically balanced."
                )

    def test_survives_30_zero_torque_steps(self, env):
        env.reset(seed=0)
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for step in range(30):
            _, _, terminated, _, info = env.step(zero_action)
            if terminated:
                reason = info.get("termination_reason", "unknown")
                pytest.fail(
                    f"{self.species_name} fell at step {step + 1}/30 under zero torque. "
                    f"Reason: {reason}. "
                    "The model is too unstable for even basic policy learning."
                )

    def test_pelvis_height_stable_short(self, env):
        env.reset(seed=0)
        body_id = getattr(env, self.root_body_id_attr)
        initial_z = env.data.xpos[body_id, 2]
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for _ in range(30):
            _, _, terminated, _, _ = env.step(zero_action)
            if terminated:
                break

        final_z = env.data.xpos[body_id, 2]
        drop = initial_z - final_z
        assert drop < self.max_height_drop, (
            f"Height dropped {drop:.3f} m (from {initial_z:.3f} to {final_z:.3f}) "
            "over 30 zero-torque steps. The model may lack sufficient joint stiffness "
            "to hold the home pose passively."
        )

    def test_tilt_deviation_stays_small(self, env):
        env.reset(seed=0)
        _, _, _, _, initial_info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        initial_tilt = initial_info.get("tilt_angle", 0.0)

        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        max_tilt_seen = initial_tilt

        for _ in range(29):
            _, _, terminated, _, info = env.step(zero_action)
            tilt = info.get("tilt_angle", 0.0)
            max_tilt_seen = max(max_tilt_seen, tilt)
            if terminated:
                break

        tilt_increase = max_tilt_seen - initial_tilt
        assert tilt_increase < np.radians(30), (
            f"Tilt increased by {np.degrees(tilt_increase):.1f} deg from initial "
            f"{np.degrees(initial_tilt):.1f} deg to peak {np.degrees(max_tilt_seen):.1f} deg "
            "over 30 zero-torque steps (limit: 30 deg increase). "
            "The home pose may lack sufficient passive stability."
        )
