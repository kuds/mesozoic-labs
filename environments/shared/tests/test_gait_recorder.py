"""Real-contact and physics-preservation checks for gait observation.

Small analytic MuJoCo plants distinguish tangential velocity at a loaded
contact point from motion of a foot site, and separate ground support from
self-contact. The shipped plants check the registry and the observer's lack
of side effects without requiring trained policies or long rollout panels.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from environments.brachiosaurus.envs.brachio_env import BrachioEnv
from environments.compsognathus.envs import CompsognathusEnv, CompsognathusRobotEnv
from environments.dibothrosuchus.envs.dibothrosuchus_env import DibothrosuchusEnv
from environments.shared.gait.morphology import FOOT_GEOMETRIES, GaitMorphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.trex.envs.trex_env import TRexEnv
from environments.velociraptor.envs.raptor_env import RaptorEnv

SPECIES = [
    pytest.param("trex", TRexEnv, id="trex"),
    pytest.param("velociraptor", RaptorEnv, id="velociraptor"),
    pytest.param("brachiosaurus", BrachioEnv, id="brachiosaurus"),
    pytest.param("dibothrosuchus", DibothrosuchusEnv, id="dibothrosuchus"),
    pytest.param("compsognathus", CompsognathusEnv, id="compsognathus"),
    pytest.param("compsognathus_robot", CompsognathusRobotEnv, id="compsognathus_robot"),
]


def _noop_hook():
    pass


def _animal_bodies(model):
    """Independent ancestry walk, excluding siblings such as prey and food."""
    free_joint = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    root = int(model.jnt_bodyid[free_joint])
    descendants = set()
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root:
            descendants.add(body)
    return descendants


def _ground_force_oracle(env, species, foot_names):
    """Partition solved force using authored names, not the recorder's map."""
    by_name = {name: foot for foot, names in FOOT_GEOMETRIES[species].items() for name in names}
    animal = _animal_bodies(env.model)
    terrain = set(env._static_floor_geoms())
    feet = dict.fromkeys(foot_names, 0.0)
    nonfoot = 0.0
    wrench = np.zeros(6)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        if contact.efc_address < 0:
            continue
        one, two = int(contact.geom1), int(contact.geom2)
        if one in terrain and int(env.model.geom_bodyid[two]) in animal:
            ground, part = one, two
        elif two in terrain and int(env.model.geom_bodyid[one]) in animal:
            ground, part = two, one
        else:
            continue
        assert ground in terrain
        mujoco.mj_contactForce(env.model, env.data, index, wrench)
        normal = max(float(wrench[0]), 0.0)
        name = env.model.geom(part).name
        if name in by_name:
            feet[by_name[name]] += normal
        else:
            nonfoot += normal
    return np.array([feet[foot] for foot in foot_names]), nonfoot


@pytest.mark.parametrize("species,env_class", SPECIES)
def test_all_six_registries_separate_distal_feet_from_fallen_body_support(species, env_class):
    env = env_class(reset_noise_scale=0.0)
    try:
        env.reset(seed=19)
        morphology = GaitMorphology.from_env(env, species)
        expected_names = ("fr", "fl", "rr", "rl") if species in ("brachiosaurus", "dibothrosuchus") else ("r", "l")
        assert morphology.foot_names == expected_names
        declared = {name for names in FOOT_GEOMETRIES[species].values() for name in names}
        resolved = {env.model.geom(g).name for group in morphology.foot_geom_ids for g in group}
        assert resolved == declared
        # No ancestor body/whole-leg shortcut may turn a thigh, shin, ankle
        # housing, tail or trunk into a registered distal foot.
        for geom in morphology.animal_geom_ids:
            name = env.model.geom(geom).name
            assert (morphology.geom_foot[geom] >= 0) == (name in declared)

        recorder = SubstepContactRecorder(env, morphology)
        recorder.capture()
        feet, nonfoot = _ground_force_oracle(env, species, morphology.foot_names)
        assert feet.sum() > 0
        np.testing.assert_allclose(recorder.trace()["floor_force_n"][0], feet, rtol=1e-13)
        assert recorder.trace()["body_floor_force_n"][0] == pytest.approx(nonfoot, rel=1e-13)

        # Force a fallen pose in this isolated test model. Merely inspecting
        # a healthy standing pose would never exercise nonfoot classification.
        env.data.qpos[morphology.root_qpos_address + 2] -= 1.1 * morphology.leg_length_m
        env.data.time += env.model.opt.timestep
        mujoco.mj_forward(env.model, env.data)
        recorder.capture()
        feet, nonfoot = _ground_force_oracle(env, species, morphology.foot_names)
        assert nonfoot > 0
        np.testing.assert_allclose(recorder.trace()["floor_force_n"][-1], feet, rtol=1e-13)
        assert recorder.trace()["body_floor_force_n"][-1] == pytest.approx(nonfoot, rel=1e-13)
        assert recorder.trace()["floor_force_n"][-1].sum() < feet.sum() + nonfoot
    finally:
        env.close()


@pytest.mark.parametrize("species,env_class", SPECIES)
def test_reference_scales_exclude_prey_and_do_not_touch_live_state_or_rng(species, env_class):
    env = env_class(reset_noise_scale=0.01)
    try:
        env.reset(seed=7)
        env.step(np.zeros(env.action_space.shape))
        live = {name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl", "sensordata", "xpos")}
        rng = copy.deepcopy(env.np_random.bit_generator.state)
        time = env.data.time
        morphology = GaitMorphology.from_env(env, species)
        descendants = _animal_bodies(env.model)
        weight = sum(env.model.body_mass[b] for b in descendants) * np.linalg.norm(env.model.opt.gravity)
        assert morphology.body_weight_n == pytest.approx(weight, rel=1e-13)
        assert morphology.leg_length_m > 0
        outsiders = set(range(1, env.model.nbody)) - descendants
        if species != "compsognathus_robot":
            assert sum(env.model.body_mass[b] for b in outsiders) > 0
        if sum(env.model.body_mass[b] for b in outsiders) > 0:
            assert morphology.body_weight_n < sum(env.model.body_mass) * np.linalg.norm(env.model.opt.gravity)
            # A target's mass is unrelated to animal normalization even when
            # inflated well beyond the animal's mass.
            for body in outsiders:
                env.model.body_mass[body] *= 1000
            assert GaitMorphology.from_env(env, species).body_weight_n == morphology.body_weight_n
        for name, expected in live.items():
            np.testing.assert_array_equal(getattr(env.data, name), expected)
        assert env.data.time == time
        assert env.np_random.bit_generator.state == rng
    finally:
        env.close()


@pytest.mark.parametrize("species,env_class", SPECIES)
def test_recording_preserves_exact_dynamics_observation_reward_and_next_reset(species, env_class):
    plain, observed = env_class(reset_noise_scale=0.01), env_class(reset_noise_scale=0.01)
    try:
        plain_obs, _ = plain.reset(seed=41)
        observed_obs, _ = observed.reset(seed=41)
        np.testing.assert_array_equal(plain_obs, observed_obs)
        morphology = GaitMorphology.from_env(observed, species)
        actions = np.random.default_rng(103).uniform(-0.05, 0.05, size=(3, *plain.action_space.shape))
        with SubstepContactRecorder(observed, morphology) as recorder:
            for action in actions:
                left, right = plain.step(action.copy()), observed.step(action.copy())
                np.testing.assert_array_equal(left[0], right[0])
                assert left[1:4] == right[1:4]
                np.testing.assert_array_equal(plain.data.qpos, observed.data.qpos)
                np.testing.assert_array_equal(plain.data.qvel, observed.data.qvel)
                np.testing.assert_array_equal(plain.data.sensordata, observed.data.sensordata)
                assert plain.np_random.bit_generator.state == observed.np_random.bit_generator.state
        trace = recorder.trace()
        assert len(trace["time_s"]) == 1 + len(actions) * observed.frame_skip
        np.testing.assert_allclose(np.diff(trace["time_s"]), observed.model.opt.timestep, rtol=1e-13)
        assert observed._substep_probe_hook is None
        # Omitting the seed makes reset consume the generator's next draws.
        # This catches RNG use that comparing two freshly seeded resets misses.
        left, right = plain.reset(), observed.reset()
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(plain.data.qpos, observed.data.qpos)
        np.testing.assert_array_equal(plain.data.qvel, observed.data.qvel)
        assert plain.np_random.bit_generator.state == observed.np_random.bit_generator.state
    finally:
        plain.close()
        observed.close()


@pytest.fixture
def real_env():
    env = RaptorEnv(reset_noise_scale=0.0, frame_skip=3)
    env.reset(seed=0)
    try:
        yield env
    finally:
        env.close()


def test_previous_hook_runs_before_every_capture_and_is_restored(real_env, monkeypatch):
    order = []

    def previous():
        order.append("previous")

    real_env._substep_probe_hook = previous
    recorder = SubstepContactRecorder(real_env, GaitMorphology.from_env(real_env, "velociraptor"))
    capture = recorder.capture

    def traced_capture():
        order.append("capture")
        capture()

    monkeypatch.setattr(recorder, "capture", traced_capture)
    with recorder:
        real_env.step(np.zeros(real_env.action_space.shape))
    assert order == ["capture"] + ["previous", "capture"] * real_env.frame_skip
    assert real_env._substep_probe_hook is previous
    recorder.close()  # Idempotent, including with a previous owner.
    assert real_env._substep_probe_hook is previous


@pytest.mark.parametrize("exception_location", ["initial_capture", "previous_hook", "body"])
def test_hook_cleanup_on_exceptions(real_env, monkeypatch, exception_location):
    def failure():
        raise RuntimeError("sentinel")

    previous = failure if exception_location == "previous_hook" else _noop_hook
    real_env._substep_probe_hook = previous
    recorder = SubstepContactRecorder(real_env, GaitMorphology.from_env(real_env, "velociraptor"))
    if exception_location == "initial_capture":
        monkeypatch.setattr(recorder, "capture", failure)
    with pytest.raises(RuntimeError, match="sentinel"):
        with recorder:
            if exception_location == "body":
                failure()
            real_env.step(np.zeros(real_env.action_space.shape))
    assert real_env._substep_probe_hook is previous


def test_reset_boundary_fails_and_restores_previous_hook(real_env):
    previous = _noop_hook
    real_env._substep_probe_hook = previous
    recorder = SubstepContactRecorder(real_env, GaitMorphology.from_env(real_env, "velociraptor"))
    with pytest.raises(RuntimeError, match="detach.*before resetting"):
        with recorder:
            real_env.step(np.zeros(real_env.action_space.shape))
            real_env.reset(seed=0)
            recorder.capture()
    assert real_env._substep_probe_hook is previous


def test_model_mismatch_refused_before_attachment_and_after_model_swap(real_env):
    other = RaptorEnv(reset_noise_scale=0.0)
    try:
        other.reset(seed=0)
        previous = _noop_hook
        real_env._substep_probe_hook = previous
        wrong = SubstepContactRecorder(real_env, GaitMorphology.from_env(other, "velociraptor"))
        with pytest.raises(RuntimeError, match="different model"):
            with wrong:
                pytest.fail("mismatched recorder entered")
        assert real_env._substep_probe_hook is previous
        model = real_env.model
        recorder = SubstepContactRecorder(real_env, GaitMorphology.from_env(real_env, "velociraptor"))
        try:
            with pytest.raises(RuntimeError, match="model changed"):
                with recorder:
                    real_env.model = other.model
                    recorder.capture()
        finally:
            real_env.model = model
        assert real_env._substep_probe_hook is previous
    finally:
        other.close()


def test_later_hook_owner_is_not_overwritten_and_double_attachment_is_refused(real_env):
    def later():
        pass

    recorder = SubstepContactRecorder(real_env, GaitMorphology.from_env(real_env, "velociraptor"))
    with recorder:
        with pytest.raises(RuntimeError, match="already attached"):
            recorder.__enter__()
        real_env._substep_probe_hook = later
    assert real_env._substep_probe_hook is later


def _toy_env(animal_xml, *, root_z=0.09, contact_xml=""):
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco>
          <option timestep=".002"/>
          <worldbody>
            <geom name="floor" type="plane" size="3 3 .1"/>
            <body name="animal" pos="0 0 {root_z}">
              <freejoint/>
              {animal_xml}
            </body>
          </worldbody>
          {contact_xml}
          <sensor><touch name="r" site="rsite"/><touch name="l" site="lsite"/></sensor>
        </mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data, _substep_probe_hook=None)
    env.unwrapped = env
    ids = tuple((int(model.geom(name).id),) for name in ("r", "l"))
    geom_foot = np.full(model.ngeom, -1, dtype=np.int32)
    for foot, group in enumerate(ids):
        geom_foot[list(group)] = foot
    morphology = GaitMorphology(
        species="analytic_test_plant",
        model=model,
        foot_names=("r", "l"),
        foot_geom_ids=ids,
        foot_site_ids=tuple(int(model.site(name).id) for name in ("rsite", "lsite")),
        touch_addresses=tuple((int(model.sensor(name).adr[0]),) for name in ("r", "l")),
        geom_foot=geom_foot,
        terrain_geom_ids=(int(model.geom("floor").id),),
        animal_geom_ids=frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] != 0),
        root_body_id=int(model.body("animal").id),
        root_qpos_address=0,
        body_weight_n=float(sum(model.body_mass) * 9.81),
        leg_length_m=0.5,
    )
    return env, morphology


SPHERE_FEET = """
    <geom name="r" type="sphere" size=".1" mass="1"/>
    <site name="rsite" size=".11"/>
    <geom name="l" type="sphere" size=".03" pos="0 0 .4" mass=".1"/>
    <site name="lsite" pos="0 0 .4" size=".04"/>
"""


@pytest.mark.parametrize("motion", ["stationary", "sliding", "rolling", "spinning"])
def test_slip_measures_loaded_contact_point_instead_of_foot_site(motion):
    env, morphology = _toy_env(SPHERE_FEET)
    assert env.data.ncon == 1
    point = env.data.contact[0].pos.copy()
    lever = point - env.data.xpos[morphology.root_body_id]
    linear, angular = np.zeros(3), np.zeros(3)
    if motion == "sliding":
        linear[0] = 0.2
    elif motion == "rolling":
        angular[1] = 1.0
        linear = -np.cross(angular, lever)
    elif motion == "spinning":
        angular[1] = 1.0
    env.data.qvel[:3], env.data.qvel[3:6] = linear, angular
    mujoco.mj_forward(env.model, env.data)
    recorder = SubstepContactRecorder(env, morphology)
    recorder.capture()
    trace = recorder.trace()
    assert trace["floor_force_n"][0, 0] > 0
    contact_velocity = linear + np.cross(angular, lever)
    expected = np.linalg.norm(contact_velocity[:2])
    assert trace["slip_speed_mps"][0, 0] == pytest.approx(expected, abs=1e-12)
    assert trace["slip_speed_mps"][0, 1] == 0  # The upper, unloaded foot moves too.
    if motion == "rolling":
        assert np.linalg.norm(linear[:2]) > 0.09
        assert expected < 1e-12
    if motion == "spinning":
        assert np.linalg.norm(linear) == 0
        assert expected > 0.09
    assert trace["foot_clearance_m"][0, 0] == pytest.approx(-0.01, abs=1e-12)
    # Geometric clearance is a surface distance; a site at the sphere's
    # center would falsely report 9 cm of clearance while it bears weight.
    assert trace["foot_position_m"][0, 0, 2] == pytest.approx(0.09)


def test_foot_foot_force_is_reported_without_becoming_floor_support_or_slip():
    animal = """
      <geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
      <body pos="-.08 0 0"><joint type="slide" axis="1 0 0"/>
        <geom name="r" type="sphere" size=".1" mass=".1"/><site name="rsite" size=".11"/>
      </body>
      <body pos=".08 0 0"><joint type="slide" axis="1 0 0"/>
        <geom name="l" type="sphere" size=".1" mass=".1"/><site name="lsite" size=".11"/>
      </body>
    """
    env, morphology = _toy_env(animal, root_z=2.0)
    assert env.data.ncon == 1
    recorder = SubstepContactRecorder(env, morphology)
    recorder.capture()
    trace = recorder.trace()
    assert trace["touch_force_n"].sum() > 0  # A touch-only detector would accept this.
    assert trace["foot_foot_force_n"][0] > 0
    np.testing.assert_array_equal(trace["floor_force_n"], [[0, 0]])
    np.testing.assert_array_equal(trace["slip_speed_mps"], [[0, 0]])
    assert trace["body_floor_force_n"][0] == 0


def test_inactive_proximity_contact_never_calls_force_solver_or_creates_support(monkeypatch):
    pair = '<contact><pair geom1="floor" geom2="r" margin=".01" gap=".1"/></contact>'
    env, morphology = _toy_env(SPHERE_FEET, root_z=0.15, contact_xml=pair)
    assert env.data.ncon == 1
    assert env.data.contact[0].efc_address == -1

    def must_not_call(*args):
        pytest.fail("inactive contact was queried as a load-bearing constraint")

    monkeypatch.setattr(mujoco, "mj_contactForce", must_not_call)
    env.data.qvel[0] = 2.0
    recorder = SubstepContactRecorder(env, morphology)
    recorder.capture()
    trace = recorder.trace()
    np.testing.assert_array_equal(trace["floor_force_n"], [[0, 0]])
    np.testing.assert_array_equal(trace["slip_speed_mps"], [[0, 0]])
    assert trace["body_floor_force_n"][0] == trace["foot_foot_force_n"][0] == 0


def test_trace_copies_live_arrays_and_refuses_empty_trace(real_env):
    recorder = SubstepContactRecorder(real_env, GaitMorphology.from_env(real_env, "velociraptor"))
    with pytest.raises(ValueError, match="no samples"):
        recorder.trace()
    with recorder:
        initial = recorder.trace()
        real_env.step(np.zeros(real_env.action_space.shape))
        trace = recorder.trace()
    np.testing.assert_array_equal(trace["root_position_m"][0], initial["root_position_m"][0])
    np.testing.assert_array_equal(trace["foot_position_m"][0], initial["foot_position_m"][0])
    trace["root_position_m"][:] = np.nan
    assert np.isfinite(recorder.trace()["root_position_m"]).all()


def test_contact_with_a_non_terrain_body_is_a_diagnostic_never_support():
    """A prop (prey) pressing on a foot is neither floor support, body support nor foot-on-foot."""
    prop = '<worldbody><geom name="prop" type="box" size=".05 .05 .05" pos=".079 0 .49"/></worldbody>'
    env, morphology = _toy_env(SPHERE_FEET, root_z=0.0899, contact_xml=prop)
    recorder = SubstepContactRecorder(env, morphology)
    recorder.capture()
    trace = recorder.trace()
    assert trace["nonterrain_contact_force_n"][0] > 0
    assert trace["floor_force_n"][0, 1] == 0  # the l foot touches only the prop
    assert trace["body_floor_force_n"][0] == trace["foot_foot_force_n"][0] == 0


def test_divergence_reset_stops_recording_but_a_user_reset_is_still_refused(real_env):
    recorder = SubstepContactRecorder(real_env, GaitMorphology.from_env(real_env, "velociraptor"))
    with recorder:
        real_env.step(np.zeros(real_env.action_space.shape))
        samples = len(recorder.trace()["time_s"])
        real_env.data.qvel[6] = 2e10  # MuJoCo resets the state inside the next mj_step
        real_env.step(np.zeros(real_env.action_space.shape))
        real_env.step(np.zeros(real_env.action_space.shape))
    assert recorder.diverged
    trace = recorder.trace()
    assert len(trace["time_s"]) == samples
    assert bool(trace["physics_diverged"]) is True
    assert np.all(np.diff(trace["time_s"]) > 0)


def test_velociraptor_sickle_claw_is_not_a_foot_and_every_registered_geometry_reaches_the_terrain(monkeypatch):
    from environments.shared.gait import morphology as morphology_module

    assert all("claw" not in name for names in FOOT_GEOMETRIES["velociraptor"].values() for name in names)
    env = RaptorEnv(reset_noise_scale=0.0)
    try:
        env.reset(seed=0)
        claw = int(env.model.geom("r_claw_geom").id)
        floor = int(env.model.geom("floor").id)
        assert not morphology_module._can_collide(env.model, claw, floor)
        registry = dict(FOOT_GEOMETRIES)
        registry["velociraptor"] = {
            side: (*names, f"{side}_claw_geom") for side, names in FOOT_GEOMETRIES["velociraptor"].items()
        }
        monkeypatch.setattr(morphology_module, "FOOT_GEOMETRIES", registry)
        with pytest.raises(ValueError, match="claw_geom cannot collide with the declared terrain"):
            GaitMorphology.from_env(env, "velociraptor")
    finally:
        env.close()


# --- heightfield terrain -----------------------------------------------------


def _surface_oracle(model, geom, count=3000):
    """Independent dense random surface samples of a primitive geometry, in its own frame."""
    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    rng = np.random.default_rng(geom)
    unit = rng.normal(size=(count, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return size[0] * unit
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return unit * size[:3]
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        cap = size[0] * unit
        cap[:, 2] += np.where(cap[:, 2] >= 0, size[1], -size[1])
        angle = rng.uniform(0, 2 * np.pi, count)
        side = np.stack(
            [size[0] * np.cos(angle), size[0] * np.sin(angle), rng.uniform(-size[1], size[1], count)], axis=1
        )
        return np.concatenate([cap, side])
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        points = rng.uniform(-1, 1, size=(count, 3))
        face = rng.integers(0, 3, size=count)
        points[np.arange(count), face] = np.sign(points[np.arange(count), face])
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
        return np.concatenate([points, corners]) * size[:3]
    raise AssertionError(f"oracle has no sampler for geom type {kind}")


def _oracle_clearance(model, data, geoms, height_at):
    gaps = []
    for geom in geoms:
        points = data.geom_xpos[geom] + _surface_oracle(model, geom) @ data.geom_xmat[geom].reshape(3, 3).T
        gaps.append(float(np.min(points[:, 2] - height_at(points[:, 0], points[:, 1]))))
    return min(gaps)


@pytest.fixture(scope="module")
def terrain_env():
    from environments.shared.paths import REPOSITORY_ROOT
    from environments.shared.train_behaviors import create_behavior_env, read_recipe

    _, commands, terrain, env_kwargs = read_recipe(
        REPOSITORY_ROOT / "configs/trex/behaviors/bumps_terrain.toml", "trex"
    )
    env = create_behavior_env("trex", commands=commands, terrain=terrain, run_seed=42, **env_kwargs)
    try:
        yield env
    finally:
        env.close()


def test_heightfield_clearance_follows_the_terrain_surface(terrain_env):
    """mj_geomDistance to a heightfield is not a surface distance; the recorder measures the real gap."""
    env = terrain_env
    env.reset(seed=0, options={"terrain_family": "bumps"})
    unwrapped = env.unwrapped
    morphology = GaitMorphology.from_env(env, "trex")
    height_at = unwrapped.terrain.height_at
    assert morphology.terrain_height is not None
    assert morphology.describe()["terrain_clearance"].startswith("vertical gap")
    data = mujoco.MjData(unwrapped.model)
    data.qpos[:] = unwrapped.data.qpos
    base_z = float(data.qpos[2])
    apron = unwrapped.terrain.config.apron_radius
    checked = 0
    for xy in ((apron + 2.0, 0.5), (apron + 4.0, -1.0), (-apron - 3.0, 2.0)):
        for lift in (0.0, 0.004, 0.02, 0.1):
            data.qpos[0:2] = xy
            data.qpos[2] = base_z + height_at(*xy)
            mujoco.mj_kinematics(unwrapped.model, data)
            lowest = min(_oracle_clearance(unwrapped.model, data, ids, height_at) for ids in morphology.foot_geom_ids)
            data.qpos[2] += lift - lowest
            mujoco.mj_kinematics(unwrapped.model, data)
            for foot, ids in enumerate(morphology.foot_geom_ids):
                reference = _oracle_clearance(unwrapped.model, data, ids, height_at)
                measured = morphology.foot_clearance(data, foot)
                assert measured == pytest.approx(reference, abs=1.5e-3), (xy, lift, foot)
                legacy = min(
                    float(mujoco.mj_geomDistance(unwrapped.model, data, g, t, 2 * morphology.leg_length_m, None))
                    for g in ids
                    for t in morphology.terrain_geom_ids
                )
                assert abs(legacy - reference) > 0.1  # the old reading, about -1 m whatever the height
                checked += 1
    assert checked == 24


def test_recorder_on_heightfield_reports_resting_contact_not_a_metre_underground(terrain_env):
    env = terrain_env
    env.reset(seed=1, options={"terrain_family": "bumps"})
    morphology = GaitMorphology.from_env(env, "trex")
    with SubstepContactRecorder(env, morphology) as recorder:
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape))
    trace = recorder.trace()
    loaded = trace["floor_force_n"] > 0
    assert loaded.any()
    assert np.all(np.abs(trace["foot_clearance_m"][loaded]) < 0.01)


def test_heightfield_without_a_matching_height_map_is_refused(terrain_env):
    from environments.shared.terrain import generate_terrain

    env = terrain_env
    env.reset(seed=2, options={"terrain_family": "bumps"})
    unwrapped = env.unwrapped
    realization = unwrapped.terrain
    try:
        unwrapped.terrain = None
        with pytest.raises(ValueError, match="height map"):
            GaitMorphology.from_env(env, "trex")
        unwrapped.terrain = generate_terrain(realization.config, run_seed=realization.run_seed + 1, episode_index=0)
        with pytest.raises(ValueError, match="does not describe its compiled heightfield"):
            GaitMorphology.from_env(env, "trex")
    finally:
        unwrapped.terrain = realization
    assert GaitMorphology.from_env(env, "trex").terrain_height is not None


def test_plane_clearance_is_unchanged_on_the_flat_family(terrain_env):
    env = terrain_env
    env.reset(seed=3, options={"terrain_family": "flat"})
    morphology = GaitMorphology.from_env(env, "trex")
    assert morphology.terrain_height is None and "terrain_clearance" not in morphology.describe()
    unwrapped = env.unwrapped
    for foot, ids in enumerate(morphology.foot_geom_ids):
        expected = min(
            float(mujoco.mj_geomDistance(unwrapped.model, unwrapped.data, g, t, 2 * morphology.leg_length_m, None))
            for g in ids
            for t in morphology.terrain_geom_ids
        )
        assert morphology.foot_clearance(unwrapped.data, foot) == expected
