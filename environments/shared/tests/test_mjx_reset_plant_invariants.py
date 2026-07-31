"""MJX counterparts of the reset plant invariants.

``test_reset_plant_invariants.py`` validates the Gymnasium plant; the MJX
reset lagged the `ca56f6c` ground settle and kept the same defect class: joint
and yaw jitter change how far the lowest geom sits below the root, but the
root height stayed fixed at the keyframe value.  Measured on T-Rex at the
stage-1 noise (0.10), un-settled MJX spawns ranged from **41.2 mm inside the
floor to 5.2 mm above it** — milder than the SB3 case only because MJX never
jittered root height, and still far beyond the 2 mm the invariants allow.
The midpoint action mapping was worse: its reset base pose is ``qpos0``, which
on brachiosaurus hovers **610 mm** above the floor, so every episode opened
with a half-metre free fall.

The settle mirrors ``BaseDinoEnv._settle_root_on_ground``: shift the root so
the lowest animal geom sits at the clearance the home keyframe was authored
with.  MJX has no ``mj_geomDistance``, so the probe is analytic (exact for
sphere/capsule/cylinder/box/ellipsoid over a horizontal plane); these tests
therefore cross-check every reset with CPU ``mj_geomDistance`` so the
assertion does not share the implementation's formula.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco.mjx")

import environments.brachiosaurus.mjx_config  # noqa: E402,F401
import environments.dibothrosuchus.mjx_config  # noqa: E402,F401
import environments.trex.mjx_config  # noqa: E402,F401
import environments.velociraptor.mjx_config  # noqa: E402,F401
from environments.shared.mjx_env import MJXDinoEnv, _ground_settle_constants  # noqa: E402

SPECIES = ["trex", "velociraptor", "brachiosaurus", "dibothrosuchus"]

#: One batched reset gives this many independent spawn samples per species.
NUM_ENVS = 16
#: Widest reset noise any stage ships, matching the SB3 invariant suite.
RESET_NOISE = 0.10
YAW_NOISE = 0.10
#: Same geometry budgets as the SB3 invariants, which already include solver-
#: scale slop; MJX float32 adds micrometres, not millimetres.
MAX_EXTRA_PENETRATION = 2e-3
MAX_GROUND_CLEARANCE = 0.01

_NO_NOISE = {"reset_noise_scale": 0.0, "init_qpos_noise": 0.0, "init_yaw_noise": 0.0}

# Env construction compiles the reset kernel, which dominates this suite's
# runtime — build each configuration once and share it across tests.
_ENV_CACHE: dict[tuple, MJXDinoEnv] = {}


def make_env(species: str, *, noisy: bool, num_envs: int = NUM_ENVS) -> MJXDinoEnv:
    key = (species, noisy, num_envs)
    if key not in _ENV_CACHE:
        kwargs = {"reset_noise_scale": RESET_NOISE, "init_yaw_noise": YAW_NOISE} if noisy else dict(_NO_NOISE)
        _ENV_CACHE[key] = MJXDinoEnv(species, stage=1, num_envs=num_envs, env_kwargs=kwargs)
    return _ENV_CACHE[key]


def reference_clearances(env: MJXDinoEnv, batched_qpos: np.ndarray) -> np.ndarray:
    """Per-env ground clearance via CPU ``mj_geomDistance`` — independent of
    the analytic support-extent formula the settle itself uses."""
    model = env.mj_model
    data = mujoco.MjData(model)
    settle = _ground_settle_constants(model)
    floors = [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) == 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    clearances = []
    for qpos in batched_qpos:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        clearances.append(
            min(
                mujoco.mj_geomDistance(model, data, int(g), int(f), 10.0, None) for g in settle.geom_ids for f in floors
            )
        )
    return np.asarray(clearances)


@pytest.mark.parametrize("species", SPECIES)
def test_reset_never_penetrates_the_floor(species):
    """No jittered reset may start deeper than the authored home contact."""
    env = make_env(species, noisy=True)
    states = env.reset(jax.random.PRNGKey(42))
    clearances = reference_clearances(env, np.asarray(states.data.qpos))
    worst = float(clearances.min())
    assert worst > env._home_ground_clearance - MAX_EXTRA_PENETRATION, (
        f"{species} MJX reset spawned {(env._home_ground_clearance - worst) * 1000:.1f} mm deeper "
        "than its authored home contact; the settle must run after ALL pose noise."
    )


@pytest.mark.parametrize("species", SPECIES)
def test_reset_never_spawns_airborne(species):
    """No jittered reset may open the episode with a free fall."""
    env = make_env(species, noisy=True)
    states = env.reset(jax.random.PRNGKey(42))
    clearances = reference_clearances(env, np.asarray(states.data.qpos))
    highest = float(clearances.max())
    assert highest - env._home_ground_clearance < MAX_GROUND_CLEARANCE, (
        f"{species} MJX reset spawned {(highest - env._home_ground_clearance) * 100:.1f} cm above "
        "its authored home contact; the episode would open with a free fall."
    )


@pytest.mark.parametrize("species", SPECIES)
def test_settling_keeps_reset_deterministic(species):
    """Settling is a pure function of the sampled pose: same key, same state."""
    env = make_env(species, noisy=True)
    first = env.reset(jax.random.PRNGKey(7))
    second = env.reset(jax.random.PRNGKey(7))
    np.testing.assert_array_equal(np.asarray(first.data.qpos), np.asarray(second.data.qpos))
    np.testing.assert_array_equal(np.asarray(first.data.qvel), np.asarray(second.data.qvel))


@pytest.mark.parametrize("species", SPECIES)
def test_settle_target_matches_the_gymnasium_plant(species):
    """Both backends must settle to the same authored keyframe clearance.

    The float32 MJX probe and the float64 ``mj_geomDistance`` probe measure
    the same authored pose, so they may differ by rounding only.  A larger gap
    means the two plants are settling to different ground contacts and every
    cross-backend baseline comparison is skewed.
    """
    gym_env = None
    try:
        from environments.brachiosaurus.envs.brachio_env import BrachioEnv
        from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
        from environments.trex.envs.trex_env import TRexEnv
        from environments.velociraptor.envs.raptor_env import RaptorEnv

        gym_cls = {
            "trex": TRexEnv,
            "velociraptor": RaptorEnv,
            "brachiosaurus": BrachioEnv,
            "dibothrosuchus": DibothrosuchusEnv,
        }[species]
        gym_env = gym_cls(reset_noise_scale=0.0)
        mjx_env = make_env(species, noisy=True)
        assert mjx_env._home_ground_clearance == pytest.approx(gym_env.home_ground_clearance(), abs=1e-5)
    finally:
        if gym_env is not None:
            gym_env.close()


@pytest.mark.parametrize("species", SPECIES)
def test_noise_free_reset_is_a_bit_exact_noop(species):
    """With noise off, settling must not move the home-keyframe base pose.

    The settle target is probed through the reset's own kinematics path at
    construction, so the shift is exactly ``0.0`` — not merely small — and a
    zero-noise MJX reset stays byte-identical to an un-settled one.  All four
    species now use the home-residual mapping whose base pose IS the keyframe;
    the midpoint mapping this settle also guards (base pose ``qpos0``, which
    hovered 610 mm above the floor on pre-residual brachiosaurus) has no
    remaining user, so the keyframe-targeted branch is the covered path.
    """
    env = make_env(species, noisy=False, num_envs=2)
    states = env.reset(jax.random.PRNGKey(0))
    base = np.asarray(env._default_data.qpos)
    for qpos in np.asarray(states.data.qpos):
        np.testing.assert_array_equal(qpos, base)
