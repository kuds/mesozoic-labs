"""Shared helpers and base test classes for static balance tests.

Species-specific test files inherit from these bases, overriding only
the configuration (body names, thresholds, joint lists).

``env.step(np.zeros(...))`` is deliberately described as a *neutral action*,
not zero torque: :class:`~environments.shared.base_env.BaseDinoEnv` maps it to
the midpoint of every actuator control range, so position servos remain active.
True passive characterization uses MuJoCo's ``mjDSBL_ACTUATION`` flag and
checks that both actuator-space and generalized actuator forces stay zero.
"""

from contextlib import contextmanager

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

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


@contextmanager
def actuation_disabled(model: mujoco.MjModel, data: mujoco.MjData):
    """Temporarily disable MuJoCo actuator forces for passive simulation.

    Clearing ``data.ctrl`` is not sufficient for position servos because a
    zero control still commands a zero joint position.  ``mjDSBL_ACTUATION``
    disables actuator force generation while leaving gravity, contacts,
    springs, damping, and other passive plant forces enabled.
    """

    old_flags = int(model.opt.disableflags)
    model.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    mujoco.mj_forward(model, data)
    try:
        yield
    finally:
        model.opt.disableflags = old_flags
        mujoco.mj_forward(model, data)


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


class HomePoseCOMBase:
    """Verify the home keyframe places COM over the support polygon.

    Subclasses must provide:
        - ``env`` fixture
        - ``foot_geom_names``: list of foot geom names for contact detection
        - ``root_body``: root body name for COM computation
        - ``ankle_body_names``: pair of (right, left) ankle/metatarsus body names
        - ``max_ankle_offset``: max distance COM can be behind the ankle midpoint
        - ``max_support_distance``: max distance COM can be from nearest foot contact
        - ``species_label``: display name for error messages
    """

    foot_geom_names: list[str] = []
    root_body: str = ""
    ankle_body_names: tuple[str, str] = ("", "")
    max_ankle_offset: float = 0.15
    max_support_distance: float = 0.20
    species_label: str = "dinosaur"

    def test_foot_contacts_exist(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, self.foot_geom_names)
        assert len(contacts) >= 2, (
            f"Expected at least 2 foot-floor contacts at home pose, got {len(contacts)}. "
            f"The {self.species_label} may be floating or the keyframe places feet above the floor."
        )

    def test_com_inside_support_polygon(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, self.foot_geom_names)
        if len(contacts) < 3:
            if len(contacts) == 0:
                pytest.skip("No foot contacts detected (model may need ground settling)")
            com = com_xy(env.model, env.data, self.root_body)
            bbox_min = contacts.min(axis=0) - 0.05
            bbox_max = contacts.max(axis=0) + 0.05
            assert np.all(com >= bbox_min) and np.all(com <= bbox_max), (
                f"COM XY {com} is outside bounding box of foot contacts [{bbox_min}, {bbox_max}]"
            )
            return

        com = com_xy(env.model, env.data, self.root_body)
        hull = ConvexHull(contacts)
        inside = np.all(hull.equations[:, :2] @ com + hull.equations[:, 2] <= 0)
        assert inside, (
            f"COM XY {com} is outside the convex hull of foot contacts. "
            f"Contact points: {contacts.tolist()}. "
            "The model's mass distribution may not balance over its feet."
        )

    def test_com_not_too_far_back(self, env):
        com = com_xy(env.model, env.data, self.root_body)
        r_name, l_name = self.ankle_body_names
        r_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, r_name)
        l_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, l_name)
        avg_ankle_x = (env.data.xpos[r_id, 0] + env.data.xpos[l_id, 0]) / 2.0
        assert com[0] >= avg_ankle_x - self.max_ankle_offset, (
            f"COM X ({com[0]:.3f}) is more than {self.max_ankle_offset * 100:.0f} cm behind "
            f"the ankle midpoint ({avg_ankle_x:.3f}). The tail mass may be pulling the COM "
            "too far rearward for stable bipedal balance."
        )

    def test_com_distance_from_support(self, env):
        contacts = get_foot_contacts_xy(env.model, env.data, self.foot_geom_names)
        com = com_xy(env.model, env.data, self.root_body)
        if len(contacts) == 0:
            pytest.skip("No foot contacts")
        dists = np.linalg.norm(contacts - com, axis=1)
        min_dist = dists.min()
        assert min_dist < self.max_support_distance, (
            f"COM is {min_dist:.3f} m from nearest foot contact. "
            f"COM: {com}, nearest contact: {contacts[dists.argmin()]}"
        )


class NeutralActionStabilityBase:
    """The reset pose under an active neutral action should remain learnable.

    Subclasses must provide:
        - ``env`` fixture
        - ``species_name``: display name for error messages
        - ``root_body_id_attr``: attribute name on env for the root body ID (e.g. "pelvis_id")
        - ``max_height_drop``: max allowed height drop in meters over 30 steps
    """

    species_name: str = ""
    root_body_id_attr: str = "pelvis_id"
    max_height_drop: float = 0.10
    max_tilt_increase: float = np.radians(30)

    def test_survives_100_neutral_action_steps(self, env):
        env.reset(seed=0)
        neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for step in range(100):
            _, _, terminated, _, info = env.step(neutral_action)
            if terminated:
                reason = info.get("termination_reason", "unknown")
                root_z = info.get("pelvis_height", info.get("torso_height", "?"))
                tilt = info.get("tilt_angle", "?")
                pytest.fail(
                    f"{self.species_name} fell at step {step + 1}/100 under a neutral action. "
                    f"Reason: {reason}, height: {root_z}, tilt: {tilt}. "
                    "The reset pose may be too unstable for basic policy learning."
                )

    def test_root_height_stable_under_neutral_action(self, env):
        env.reset(seed=0)
        body_id = getattr(env, self.root_body_id_attr)
        initial_z = env.data.xpos[body_id, 2]
        neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for _ in range(30):
            _, _, terminated, _, _ = env.step(neutral_action)
            if terminated:
                break

        final_z = env.data.xpos[body_id, 2]
        drop = initial_z - final_z
        assert drop < self.max_height_drop, (
            f"Height dropped {drop:.3f} m (from {initial_z:.3f} to {final_z:.3f}) "
            "over 30 neutral-action steps. The active neutral command may not provide "
            "a sufficiently stable learning start."
        )

    def test_tilt_deviation_stays_small_under_neutral_action(self, env):
        env.reset(seed=0)
        _, _, _, _, initial_info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        initial_tilt = initial_info.get("tilt_angle", 0.0)

        neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)
        max_tilt_seen = initial_tilt

        for _ in range(29):
            _, _, terminated, _, info = env.step(neutral_action)
            tilt = info.get("tilt_angle", 0.0)
            max_tilt_seen = max(max_tilt_seen, tilt)
            if terminated:
                break

        tilt_increase = max_tilt_seen - initial_tilt
        assert tilt_increase < self.max_tilt_increase, (
            f"Tilt increased by {np.degrees(tilt_increase):.1f} deg from initial "
            f"{np.degrees(initial_tilt):.1f} deg to peak {np.degrees(max_tilt_seen):.1f} deg "
            f"over 30 neutral-action steps (limit: {np.degrees(self.max_tilt_increase):.1f} deg increase). "
            "The active neutral command may be too unstable."
        )


class ActuatorDisabledPassiveBase:
    """Characterize the reset pose with actuator forces truly disabled.

    Subclasses configure a short minimum safe horizon. The contract deliberately
    does not require a later fall: improving passive stability should not fail CI.
    """

    species_name: str = ""
    root_body_id_attr: str = "pelvis_id"
    safe_steps: int = 30
    max_height_drop: float = 0.10
    max_tilt_increase: float = 0.20

    @staticmethod
    def _reset_without_noise(env) -> None:
        """Reset to the exact XML keyframe so the plant pin excludes RNG noise."""

        env.reset_noise_scale = 0.0
        env.reset(seed=0)

    def test_disabled_actuators_generate_no_force(self, env):
        self._reset_without_noise(env)
        # Deliberately request maximum controls: the solver-level disable flag,
        # not a coincidental zero command, must be what removes actuation.
        env.data.ctrl[:] = env.model.actuator_ctrlrange[:, 1]

        with actuation_disabled(env.model, env.data):
            mujoco.mj_step(env.model, env.data)
            np.testing.assert_allclose(env.data.actuator_force, 0.0, atol=1e-12)
            np.testing.assert_allclose(env.data.qfrc_actuator, 0.0, atol=1e-12)

    def test_actuator_disabled_minimum_safe_horizon(self, env):
        self._reset_without_noise(env)
        body_id = getattr(env, self.root_body_id_attr)
        initial_z = float(env.data.xpos[body_id, 2])
        initial_terminated, initial_info = env._is_terminated()
        assert not initial_terminated, f"{self.species_name} starts terminated at the exact home keyframe"
        initial_tilt = float(initial_info.get("tilt_angle", 0.0))

        neutral_action = np.zeros(env.action_space.shape, dtype=np.float32)
        max_height_drop = 0.0
        max_tilt_increase = 0.0

        with actuation_disabled(env.model, env.data):
            for step in range(1, self.safe_steps + 1):
                _, _, terminated, truncated, info = env.step(neutral_action)

                np.testing.assert_allclose(env.data.actuator_force, 0.0, atol=1e-12)
                np.testing.assert_allclose(env.data.qfrc_actuator, 0.0, atol=1e-12)
                assert np.all(np.isfinite(env.data.qpos)), f"{self.species_name} produced non-finite passive qpos"
                assert np.all(np.isfinite(env.data.qvel)), f"{self.species_name} produced non-finite passive qvel"
                assert not truncated, f"{self.species_name} unexpectedly truncated during passive characterization"

                height_drop = initial_z - float(env.data.xpos[body_id, 2])
                tilt_increase = float(info.get("tilt_angle", initial_tilt)) - initial_tilt
                max_height_drop = max(max_height_drop, height_drop)
                max_tilt_increase = max(max_tilt_increase, tilt_increase)
                assert not terminated, (
                    f"{self.species_name} terminated at passive step {step}, before the "
                    f"{self.safe_steps}-step safety horizon: {info.get('termination_reason', 'unknown')}"
                )

        assert max_height_drop < self.max_height_drop, (
            f"{self.species_name} passive root height dropped {max_height_drop:.3f} m in "
            f"{self.safe_steps} steps (limit {self.max_height_drop:.3f} m)"
        )
        assert max_tilt_increase < self.max_tilt_increase, (
            f"{self.species_name} passive tilt increased by {np.degrees(max_tilt_increase):.1f} deg in "
            f"{self.safe_steps} steps (limit {np.degrees(self.max_tilt_increase):.1f} deg)"
        )
