"""Tests that the Brachiosaurus model is physically set up correctly.

Validates the home keyframe: joint limits, mass distribution, and foot
contacts. These catch model regressions (e.g. mass changes, keyframe edits
that violate joint limits) before any RL training is attempted.

The Brachiosaurus is quadrupedal with front legs longer than rear legs
(giraffe-like posture). Note: unlike the bipedal species, the Brachiosaurus
does not passively balance at the home pose — it requires active control
from the RL policy. The front feet contact the ground at reset, but the
rear feet may settle over the first few simulation steps.
"""

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv


def _brachio_mass(model: mujoco.MjModel) -> float:
    """Return the total mass of the Brachiosaurus (torso subtree), excluding worldbody/food."""
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    return float(model.body_subtreemass[torso_id])


@pytest.fixture
def env():
    e = BrachioEnv()
    e.reset(seed=0)
    yield e
    e.close()


def _get_foot_contacts_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return (N, 2) array of foot-floor contact positions in the XY plane."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geom_names = [
        "fr_foot_geom",
        "fl_foot_geom",
        "rr_foot_geom",
        "rl_foot_geom",
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
    """Return the Brachiosaurus's center of mass projected onto the XY plane."""
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    com: np.ndarray = data.subtree_com[torso_id, :2].copy()
    return com


class TestHomePoseCOM:
    """Verify the home keyframe COM and foot contact properties."""

    def test_foot_contacts_exist(self, env):
        """At least 2 feet should be in contact with the floor at the home pose.

        The Brachiosaurus front feet contact immediately at reset; rear feet
        may need simulation steps to settle due to the giraffe-like posture.
        """
        contacts = _get_foot_contacts_xy(env.model, env.data)
        assert len(contacts) >= 2, (
            f"Expected at least 2 foot-floor contacts at home pose, got {len(contacts)}. "
            "The Brachiosaurus may be floating or the keyframe places feet above the floor."
        )

    def test_com_centered_laterally(self, env):
        """COM Y should be near zero (centered between left and right feet)."""
        com = _com_xy(env.model, env.data)
        assert abs(com[1]) < 0.15, (
            f"COM Y ({com[1]:.3f}) is off-center by more than 15 cm. The model may have asymmetric mass distribution."
        )

    def test_com_x_between_front_and_rear_hips(self, env):
        """COM X should lie between the front and rear hip attachment points."""
        com = _com_xy(env.model, env.data)
        fr_thigh_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "fr_thigh")
        rr_thigh_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rr_thigh")
        front_x = env.data.xpos[fr_thigh_id, 0]
        rear_x = env.data.xpos[rr_thigh_id, 0]

        min_x = min(front_x, rear_x) - 0.30
        max_x = max(front_x, rear_x) + 0.30
        assert min_x <= com[0] <= max_x, (
            f"COM X ({com[0]:.3f}) is outside hip range [{min_x:.3f}, {max_x:.3f}]. "
            "The mass distribution may be too front- or rear-heavy."
        )


class TestInitialSettling:
    """The Brachiosaurus does not passively balance (requires active control).

    These tests verify the model's settling behavior rather than static stability.
    The torso starts at z=2.0 and settles quickly under gravity.
    """

    def test_torso_starts_in_healthy_range(self, env):
        """Torso z at reset should be within the healthy range."""
        torso_z = env.data.xpos[env.torso_id, 2]
        assert 1.0 < torso_z < 3.5, f"Torso z ({torso_z:.3f}) is outside plausible range at reset."

    def test_initial_tilt_is_small(self, env):
        """Tilt at reset should be near zero (model starts upright)."""
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        tilt = info.get("tilt_angle", 0.0)
        assert tilt < np.radians(15), f"Initial tilt is {np.degrees(tilt):.1f} deg — model should start near upright."

    def test_settling_drops_less_than_1m(self, env):
        """Even without control, torso should not free-fall more than 1m in 10 steps."""
        env.reset(seed=0)
        initial_z = env.data.xpos[env.torso_id, 2]
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        for _ in range(10):
            _, _, terminated, _, _ = env.step(zero_action)
            if terminated:
                break

        final_z = env.data.xpos[env.torso_id, 2]
        drop = initial_z - final_z
        assert drop < 1.0, (
            f"Torso dropped {drop:.3f} m in 10 steps — model may be free-falling "
            f"(from {initial_z:.3f} to {final_z:.3f})."
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
        """All four knees should be flexed with room to absorb impact."""
        knee_names = ["fr_knee", "fl_knee", "rr_knee", "rl_knee"]

        for knee_name in knee_names:
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, knee_name)
            qadr = env.model.jnt_qposadr[jid]
            knee_angle = env.data.qpos[qadr]
            lo = env.model.jnt_range[jid, 0]
            hi = env.model.jnt_range[jid, 1]

            extension_margin = hi - knee_angle
            flexion_margin = knee_angle - lo

            # Quadrupedal knees need at least 5 deg margin (less aggressive than bipedal)
            assert extension_margin > np.radians(5), (
                f"{knee_name} angle {np.degrees(knee_angle):.1f} deg is too close to "
                f"extension limit {np.degrees(hi):.1f} deg — only "
                f"{np.degrees(extension_margin):.1f} deg of extension travel."
            )
            assert flexion_margin > np.radians(5), (
                f"{knee_name} angle {np.degrees(knee_angle):.1f} deg is too close to "
                f"flexion limit {np.degrees(lo):.1f} deg — only "
                f"{np.degrees(flexion_margin):.1f} deg of flexion travel."
            )


class TestMassDistribution:
    """Sanity-check the mass distribution for quadrupedal balance."""

    def test_total_mass_reasonable(self, env):
        """Total Brachiosaurus mass (torso subtree) should be in a plausible range."""
        total_mass = _brachio_mass(env.model)
        assert 150.0 < total_mass < 230.0, (
            f"Total Brachiosaurus mass is {total_mass:.1f} kg — expected 150-230 kg for this scale."
        )

    def test_leg_mass_fraction(self, env):
        """Legs should carry a meaningful fraction of total mass (at least 20% for quadruped)."""
        total_mass = _brachio_mass(env.model)
        leg_body_names = [
            "fr_thigh",
            "fr_shin",
            "fr_foot",
            "fl_thigh",
            "fl_shin",
            "fl_foot",
            "rr_thigh",
            "rr_shin",
            "rr_foot",
            "rl_thigh",
            "rl_shin",
            "rl_foot",
        ]
        leg_mass = sum(
            env.model.body_mass[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in leg_body_names
        )
        fraction = leg_mass / total_mass
        assert fraction > 0.20, (
            f"Leg mass fraction is {fraction:.1%} — four legs should be at least 20% "
            "of total mass for a quadrupedal model."
        )

    def test_tail_mass_not_excessive(self, env):
        """Tail should not exceed 15% of total mass (less critical for quadruped,
        but excessive tail mass would shift COM rearward)."""
        total_mass = _brachio_mass(env.model)
        tail_body_names = ["tail_1", "tail_2", "tail_3", "tail_4"]
        tail_mass = sum(
            env.model.body_mass[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in tail_body_names
        )
        fraction = tail_mass / total_mass
        assert fraction < 0.15, f"Tail mass fraction is {fraction:.1%} — exceeds 15% threshold for a quadruped."

    def test_left_right_symmetry(self, env):
        """Left and right leg masses should be symmetric for both front and rear pairs."""
        pairs = [
            ("fr_thigh", "fl_thigh"),
            ("fr_shin", "fl_shin"),
            ("fr_foot", "fl_foot"),
            ("rr_thigh", "rl_thigh"),
            ("rr_shin", "rl_shin"),
            ("rr_foot", "rl_foot"),
        ]

        for rn, ln in pairs:
            r_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, rn)
            l_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, ln)
            r_mass = env.model.body_mass[r_id]
            l_mass = env.model.body_mass[l_id]
            assert abs(r_mass - l_mass) < 1e-6, f"Mass asymmetry: {rn}={r_mass:.4f} kg vs {ln}={l_mass:.4f} kg"

    def test_front_legs_heavier_than_rear(self, env):
        """Front legs should be heavier than rear legs (longer in Brachiosaurus)."""
        front_names = ["fr_thigh", "fr_shin", "fr_foot", "fl_thigh", "fl_shin", "fl_foot"]
        rear_names = ["rr_thigh", "rr_shin", "rr_foot", "rl_thigh", "rl_shin", "rl_foot"]

        front_mass = sum(
            env.model.body_mass[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in front_names
        )
        rear_mass = sum(
            env.model.body_mass[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in rear_names
        )
        assert front_mass > rear_mass, (
            f"Front leg mass ({front_mass:.2f} kg) should exceed rear leg mass "
            f"({rear_mass:.2f} kg) — Brachiosaurus has characteristically longer front legs."
        )
