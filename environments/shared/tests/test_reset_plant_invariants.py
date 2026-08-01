"""Plant-level invariants that every reset must satisfy, for every species.

These assertions are deliberately independent of the reward function, the
curriculum stage and the animal: a reset either hands the policy a physically
sane starting state or it does not.  Two real defects motivated them, both
invisible to every existing test and to every reward-side metric:

1. Reset placed the root by height alone, ignoring that joint jitter changes
   how far the lowest geom sits below the root.  On T-Rex this spawned the
   model up to 0.198 m *inside* the floor; the contact solver answered with
   ~19x body weight and launched it 0.5 m upward to tumble ballistically for
   ~60 steps before landing nose-down past the nosedive threshold.  Other
   seeds spawned 0.18 m *above* the floor and opened with a free fall.  The
   prior guard checked the root against ``healthy_z_range``, which is a
   termination predicate on the ROOT and says nothing about foot-to-floor
   geometry — a pelvis at 0.739 m is "healthy" with the toes underground.

2. ``tail_1_geom`` overlapped both thigh capsules in the home pose, injecting
   16.3 kN — eleven times body weight — of constant self-contact into every
   reset of every run, identically and independently of the sampled pose.

Both classes are caught here by bounding, at t=0: penetration depth, ground
clearance, and total contact force as a multiple of body weight.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

SPECIES = [
    pytest.param(TRexEnv, id="trex"),
    pytest.param(RaptorEnv, id="velociraptor"),
    pytest.param(BrachioEnv, id="brachiosaurus"),
    pytest.param(DibothrosuchusEnv, id="dibothrosuchus"),
]

# Reset noise large enough to exercise the settling path.  Every shipped stage
# now runs at 0.05 (the stage-1a operating point, PLANT_VALIDATION section 12),
# so this is deliberately 2x the widest configured value: the invariants are
# about reset GEOMETRY, which must hold at any noise a future stage might pick,
# not only at today's.
RESET_NOISE = 0.10
SEEDS = range(48)

# Tolerances.  Reset settles to the home keyframe's authored clearance, which is
# a deliberate shallow contact (0.5 mm on T-Rex), so "no penetration" means "no
# DEEPER than the authored resting contact" plus solver-scale slop.  A tenth of
# that budget again is ample; the defect this catches was 0.198 m.
SETTLE_TOLERANCE = 5e-4
# Depth beyond the authored home contact that counts as interpenetration.
MAX_EXTRA_PENETRATION = 2e-3
# A spawn further above the authored contact than this opens with a free fall.
MAX_GROUND_CLEARANCE = 0.01
# Contact force at t=0, as a multiple of body weight.  A body resting on the
# ground carries ~1x; the solver needs headroom for the settling clearance and
# for whatever velocity jitter the reset applied.  Anything near 10x is a
# geometry defect, not a landing.
MAX_RESET_CONTACT_FORCE_WEIGHTS = 3.0


def make_env(cls):
    return cls(reset_noise_scale=RESET_NOISE)


def total_contact_force(env) -> float:
    """Sum of normal contact-force magnitudes over every active contact."""
    buf = np.zeros(6)
    total = 0.0
    for i in range(env.data.ncon):
        mujoco.mj_contactForce(env.model, env.data, i, buf)
        total += abs(float(buf[0]))
    return total


def body_weight(env) -> float:
    return float(sum(env.model.body_mass) * abs(env.model.opt.gravity[2]))


@pytest.mark.parametrize("env_cls", SPECIES)
def test_reset_never_penetrates_the_floor(env_cls):
    """No reset may start with a body geom inside the ground."""
    env = make_env(env_cls)
    home = env.home_ground_clearance()
    worst, worst_seed = float("inf"), None
    for seed in SEEDS:
        env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)
        clearance = env.lowest_ground_clearance()
        if clearance < worst:
            worst, worst_seed = clearance, seed
    assert worst > home - MAX_EXTRA_PENETRATION, (
        f"{env_cls.__name__} spawned {(home - worst) * 1000:.1f} mm deeper than its authored "
        f"home contact on seed {worst_seed}. The reset must settle the root against floor "
        "geometry, not against healthy_z_range."
    )


@pytest.mark.parametrize("env_cls", SPECIES)
def test_reset_never_spawns_airborne(env_cls):
    """No reset may start the animal dropping from a height."""
    env = make_env(env_cls)
    home = env.home_ground_clearance()
    highest, highest_seed = -float("inf"), None
    for seed in SEEDS:
        env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)
        clearance = env.lowest_ground_clearance()
        if clearance > highest:
            highest, highest_seed = clearance, seed
    assert highest - home < MAX_GROUND_CLEARANCE, (
        f"{env_cls.__name__} spawned {(highest - home) * 100:.1f} cm above its authored home "
        f"contact on seed {highest_seed}; the episode would open with a free fall."
    )


@pytest.mark.parametrize("env_cls", SPECIES)
def test_reset_contact_force_is_physically_plausible(env_cls):
    """t=0 contact force must look like resting, not like a collision.

    This is the assertion that catches a permanent self-collision: a geom pair
    overlapping in the home pose shows up as a large, *pose-independent* force
    that no amount of correct root placement removes.
    """
    env = make_env(env_cls)
    weight = body_weight(env)
    worst, worst_seed = 0.0, None
    for seed in SEEDS:
        env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)
        force = total_contact_force(env)
        if force > worst:
            worst, worst_seed = force, seed
    assert worst < MAX_RESET_CONTACT_FORCE_WEIGHTS * weight, (
        f"{env_cls.__name__} starts with {worst:.0f} N of contact force "
        f"({worst / weight:.1f}x body weight) on seed {worst_seed}. "
        "A resting body carries ~1x; a large multiple means overlapping geometry."
    )


@pytest.mark.parametrize("env_cls", SPECIES)
def test_home_pose_has_no_self_collision(env_cls):
    """The unperturbed keyframe must not have body geoms overlapping.

    Pose-independent by construction — no reset noise — so a failure here is a
    modelling defect in the MJCF, and it will be present in every single step
    of every episode rather than only at spawn.
    """
    env = env_cls(reset_noise_scale=0.0)
    env.reset(seed=0)
    mujoco.mj_forward(env.model, env.data)
    floor = {int(g) for g in env._static_floor_geoms()}

    def geom_name(geom_id: int) -> str:
        return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom{geom_id}"

    offenders = []
    for i in range(env.data.ncon):
        g1, g2 = int(env.data.contact.geom1[i]), int(env.data.contact.geom2[i])
        if g1 in floor or g2 in floor:
            continue
        depth = float(env.data.contact.dist[i])
        if depth < -MAX_EXTRA_PENETRATION:
            offenders.append(f"{geom_name(g1)}<->{geom_name(g2)} ({depth * 1000:.1f} mm)")
    assert not offenders, (
        f"{env_cls.__name__} home pose has overlapping geoms: {', '.join(offenders)}. "
        "Exclude the pair in <contact> or fix the geometry — an overlap here is a "
        "constant force injected into the solver and into every contact-based reward."
    )


@pytest.mark.parametrize("env_cls", SPECIES)
def test_reset_is_not_already_terminal(env_cls):
    """A reset must never hand back a state that terminates on the first step.

    An episode whose outcome no action can influence is not a policy failure,
    and counting it as one puts an unreachable ceiling on any reliability gate.
    """
    env = make_env(env_cls)
    terminal = []
    for seed in SEEDS:
        env.reset(seed=seed)
        _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        if terminated:
            terminal.append(seed)
    assert not terminal, f"{env_cls.__name__} reset is already terminal on seeds {terminal}"


@pytest.mark.parametrize("env_cls", SPECIES)
def test_settling_keeps_reset_deterministic(env_cls):
    """Settling must not consume RNG draws: same seed, same state."""
    a, b = make_env(env_cls), make_env(env_cls)
    for seed in (0, 17, 41):
        a.reset(seed=seed)
        b.reset(seed=seed)
        np.testing.assert_allclose(a.data.qpos, b.data.qpos, rtol=0, atol=0)
        np.testing.assert_allclose(a.data.qvel, b.data.qvel, rtol=0, atol=0)


@pytest.mark.parametrize("env_cls", SPECIES)
def test_noise_free_reset_keeps_the_authored_home_contact(env_cls):
    """With noise off, settling is a no-op: the keyframe contact is preserved.

    This is what keeps the species static-balance suites meaningful — they
    assert the home pose has real foot-floor contact, and settling to an
    arbitrary positive clearance would leave every animal hovering.
    """
    env = env_cls(reset_noise_scale=0.0)
    env.reset(seed=0)
    mujoco.mj_forward(env.model, env.data)
    assert env.lowest_ground_clearance() == pytest.approx(env.home_ground_clearance(), abs=SETTLE_TOLERANCE)


@pytest.mark.parametrize("env_cls", SPECIES)
def test_reset_keeps_feet_on_the_floor(env_cls):
    """Every reset must produce real foot-floor contact, not a hover."""
    env = make_env(env_cls)
    for seed in SEEDS:
        env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)
        floor = set(int(g) for g in env._static_floor_geoms())
        touching = sum(
            1
            for i in range(env.data.ncon)
            if int(env.data.contact.geom1[i]) in floor or int(env.data.contact.geom2[i]) in floor
        )
        assert touching > 0, f"{env_cls.__name__} seed {seed} spawned with no ground contact at all"


@pytest.mark.parametrize("env_cls", SPECIES)
def test_settle_ignores_geoms_that_cannot_touch_the_ground(env_cls):
    """Only collidable geometry may decide where the ground is.

    A geom with ``contype == 0`` and ``conaffinity == 0`` generates no
    contacts, so it can never rest on the floor.  Settling to one would hold
    the animal's real feet above the ground -- the hover these invariants
    exist to catch -- and every species carries some: cosmetic detail
    (``brow_ridge``, ``crest``, ``sagittal_crest``, dibothrosuchus' twelve
    ``scute``\\s) plus the necks that are non-collidable on every species
    except velociraptor, whose neck IS real and terminates the episode on
    ground contact.
    """
    env = make_env(env_cls)
    try:
        probed = {int(g) for g in env._root_subtree_geoms()}
        assert probed, "the settle probe must consider at least one geom"
        for geom_id in probed:
            contype = int(env.model.geom_contype[geom_id])
            conaffinity = int(env.model.geom_conaffinity[geom_id])
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom{geom_id}"
            assert contype != 0 or conaffinity != 0, (
                f"{env_cls.__name__} settle probe includes {name!r}, which cannot collide; "
                "settling to it would hold the real feet off the floor"
            )
    finally:
        env.close()
