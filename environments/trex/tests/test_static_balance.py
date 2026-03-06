"""Tests that the T-Rex model is physically set up for static balance.

Validates the home keyframe: COM projection, support polygon, joint limits,
and zero-torque stability. These catch model regressions (e.g. mass changes
that shift COM behind the feet, or keyframe edits that violate joint limits)
before any RL training is attempted.

Mirrors the velociraptor static balance tests, adapted for T-Rex proportions
(heavier mass, higher pelvis, same digitigrade foot structure).
"""

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from environments.trex.envs.trex_env import TRexEnv


def _trex_mass(model: mujoco.MjModel) -> float:
    """Return the total mass of the T-Rex (pelvis subtree), excluding worldbody/prey."""
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return float(model.body_subtreemass[pelvis_id])


@pytest.fixture
def env():
    e = TRexEnv()
    e.reset(seed=0)
    yield e
    e.close()


def _get_foot_contacts_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return (N, 2) array of foot-floor contact positions in the XY plane."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geom_names = [
        "r_toe_d3_geom",
        "l_toe_d3_geom",
        "r_toe_d4_geom",
        "l_toe_d4_geom",
        "r_metatarsus_geom",
        "l_metatarsus_geom",
    ]
    foot_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_geom_names}

    points = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 == floor_id and g2 in foot_ids) or (g2 == floor_id and g1 in foot_ids):
            points.append(c.pos[:2].copy())
    return np.array(points) if points else np.empty((0, 2))


def _com_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the T-Rex's center of mass projected onto the XY plane."""
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    com: np.ndarray = data.subtree_com[pelvis_id, :2].copy()
    return com


class TestHomePoseCOM:
    """Verify the home keyframe places COM over the support polygon."""

    def test_foot_contacts_exist(self, env):
        """Both feet should be in contact with the floor at the home pose."""
        contacts = _get_foot_contacts_xy(env.model, env.data)
        assert len(contacts) >= 2, (
            f"Expected at least 2 foot-floor contacts at home pose, got {len(contacts)}. "
            "The T-Rex may be floating or the keyframe places feet above the floor."
        )

    def test_com_inside_support_polygon(self, env):
        """COM projection must fall inside the convex hull of foot contacts."""
        contacts = _get_foot_contacts_xy(env.model, env.data)
        if len(contacts) < 3:
            if len(contacts) == 0:
                pytest.skip("No foot contacts detected (model may need ground settling)")
            com = _com_xy(env.model, env.data)
            bbox_min = contacts.min(axis=0) - 0.05
            bbox_max = contacts.max(axis=0) + 0.05
            assert np.all(com >= bbox_min) and np.all(com <= bbox_max), (
                f"COM XY {com} is outside bounding box of foot contacts [{bbox_min}, {bbox_max}]"
            )
            return

        com = _com_xy(env.model, env.data)
        hull = ConvexHull(contacts)
        inside = np.all(hull.equations[:, :2] @ com + hull.equations[:, 2] <= 0)
        assert inside, (
            f"COM XY {com} is outside the convex hull of foot contacts. "
            f"Contact points: {contacts.tolist()}. "
            "The model's mass distribution may not balance over its feet."
        )

    def test_com_not_too_far_back(self, env):
        """COM X should be forward of the ankle (not pulled back by the tail)."""
        com = _com_xy(env.model, env.data)

        r_meta_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "r_metatarsus")
        l_meta_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "l_metatarsus")
        avg_ankle_x = (env.data.xpos[r_meta_id, 0] + env.data.xpos[l_meta_id, 0]) / 2.0

        assert com[0] >= avg_ankle_x - 0.15, (
            f"COM X ({com[0]:.3f}) is more than 15 cm behind the ankle midpoint "
            f"({avg_ankle_x:.3f}). The tail mass may be pulling the COM too far "
            "rearward for stable bipedal balance."
        )

    def test_com_distance_from_support(self, env):
        """Report how far the COM is from the nearest foot contact (diagnostic)."""
        contacts = _get_foot_contacts_xy(env.model, env.data)
        com = _com_xy(env.model, env.data)

        if len(contacts) == 0:
            pytest.skip("No foot contacts")

        dists = np.linalg.norm(contacts - com, axis=1)
        min_dist = dists.min()

        # T-Rex is larger, allow slightly more distance than raptor
        assert min_dist < 0.20, (
            f"COM is {min_dist:.3f} m from nearest foot contact. "
            f"COM: {com}, nearest contact: {contacts[dists.argmin()]}"
        )


class TestZeroTorqueStability:
    """The home pose with zero control should not immediately collapse."""

    def test_survives_100_zero_torque_steps(self, env):
        """T-Rex should remain upright for 100 steps (~1 s) under zero torque."""
        env.reset(seed=0)
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for step in range(100):
            _, _, terminated, _, info = env.step(zero_action)
            if terminated:
                reason = info.get("termination_reason", "unknown")
                pelvis_z = info.get("pelvis_height", "?")
                tilt = info.get("tilt_angle", "?")
                pytest.fail(
                    f"T-Rex fell at step {step + 1}/100 under zero torque. "
                    f"Reason: {reason}, pelvis_z: {pelvis_z}, tilt: {tilt}. "
                    "The home keyframe may not be statically balanced."
                )

    def test_survives_30_zero_torque_steps(self, env):
        """T-Rex should at least survive 30 steps (~0.3 s) — enough for the
        policy to react from the home pose."""
        env.reset(seed=0)
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for step in range(30):
            _, _, terminated, _, info = env.step(zero_action)
            if terminated:
                reason = info.get("termination_reason", "unknown")
                pytest.fail(
                    f"T-Rex fell at step {step + 1}/30 under zero torque. "
                    f"Reason: {reason}. "
                    "The model is too unstable for even basic policy learning."
                )

    def test_pelvis_height_stable_short(self, env):
        """Pelvis height should not drop more than 15 cm over 30 zero-torque steps."""
        env.reset(seed=0)
        pelvis_id = env.pelvis_id
        initial_z = env.data.xpos[pelvis_id, 2]
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for _ in range(30):
            _, _, terminated, _, _ = env.step(zero_action)
            if terminated:
                break

        final_z = env.data.xpos[pelvis_id, 2]
        drop = initial_z - final_z
        assert drop < 0.15, (
            f"Pelvis dropped {drop:.3f} m (from {initial_z:.3f} to {final_z:.3f}) "
            "over 30 zero-torque steps. The model may lack sufficient joint stiffness "
            "to hold the home pose passively."
        )

    def test_tilt_deviation_stays_small(self, env):
        """Tilt should not deviate more than 30 deg from initial lean over 30 steps."""
        env.reset(seed=0)
        initial_obs, _, _, _, initial_info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
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


class TestJointLimitsAtHome:
    """Verify the home keyframe doesn't violate joint limits."""

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
        """Knees should be flexed (away from extension limit) with room to absorb impact."""
        for side in ("r", "l"):
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_knee")
            qadr = env.model.jnt_qposadr[jid]
            knee_angle = env.data.qpos[qadr]
            lo = env.model.jnt_range[jid, 0]
            hi = env.model.jnt_range[jid, 1]

            extension_margin = hi - knee_angle
            flexion_margin = knee_angle - lo

            assert extension_margin > np.radians(20), (
                f"{side}_knee angle {np.degrees(knee_angle):.1f} deg is too close to "
                f"extension limit {np.degrees(hi):.1f} deg — only "
                f"{np.degrees(extension_margin):.1f} deg of extension travel."
            )
            assert flexion_margin > np.radians(20), (
                f"{side}_knee angle {np.degrees(knee_angle):.1f} deg is too close to "
                f"flexion limit {np.degrees(lo):.1f} deg — only "
                f"{np.degrees(flexion_margin):.1f} deg of flexion travel."
            )


class TestMassDistribution:
    """Sanity-check the mass distribution for bipedal balance."""

    def test_total_mass_reasonable(self, env):
        """Total T-Rex mass (pelvis subtree) should be in a plausible range."""
        total_mass = _trex_mass(env.model)
        assert 60.0 < total_mass < 100.0, (
            f"Total T-Rex mass is {total_mass:.1f} kg — expected 60-100 kg for this scale."
        )

    def test_leg_mass_fraction(self, env):
        """Legs should carry a meaningful fraction of total mass (at least 15%)."""
        total_mass = _trex_mass(env.model)
        leg_body_names = [
            "r_thigh",
            "r_tibia",
            "r_metatarsus",
            "r_toe_d2",
            "r_toe_d3",
            "r_toe_d4",
            "l_thigh",
            "l_tibia",
            "l_metatarsus",
            "l_toe_d2",
            "l_toe_d3",
            "l_toe_d4",
        ]
        leg_mass = sum(
            env.model.body_mass[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in leg_body_names
        )
        fraction = leg_mass / total_mass
        assert fraction > 0.15, (
            f"Leg mass fraction is {fraction:.1%} — legs should be at least 15% of total mass for a bipedal model."
        )

    def test_tail_mass_not_excessive(self, env):
        """Tail should not exceed 30% of total mass (would pull COM too far back)."""
        total_mass = _trex_mass(env.model)
        tail_body_names = ["tail_1", "tail_2", "tail_3", "tail_4", "tail_5"]
        tail_mass = sum(
            env.model.body_mass[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in tail_body_names
        )
        fraction = tail_mass / total_mass
        assert fraction < 0.30, (
            f"Tail mass fraction is {fraction:.1%} — exceeds 30% threshold. "
            "This much tail mass would pull the COM behind the feet."
        )

    def test_left_right_symmetry(self, env):
        """Left and right leg masses should be symmetric."""
        r_names = ["r_thigh", "r_tibia", "r_metatarsus", "r_toe_d2", "r_toe_d3", "r_toe_d4"]
        l_names = ["l_thigh", "l_tibia", "l_metatarsus", "l_toe_d2", "l_toe_d3", "l_toe_d4"]

        for rn, ln in zip(r_names, l_names):
            r_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, rn)
            l_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, ln)
            r_mass = env.model.body_mass[r_id]
            l_mass = env.model.body_mass[l_id]
            assert abs(r_mass - l_mass) < 1e-6, f"Mass asymmetry: {rn}={r_mass:.4f} kg vs {ln}={l_mass:.4f} kg"
