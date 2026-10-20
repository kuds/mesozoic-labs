"""The mirror maps (environments/shared/symmetry.py): derivation, the committed T. rex map, its plant binding,
simulator equivariance by paired rollouts, and the [ppo] symmetry keys.  MuJoCo and numpy only; MirrorKLPPO's own
tests, which need stable-baselines3, are in test_mirror_kl_ppo.py."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

from environments.shared import symmetry as SY
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.shared.species_names import model_ids
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
ROBOT = ModelId("compsognathus_longipes", "robot")


@pytest.fixture(scope="module")
def plant():
    return current_plant_identity(TYRANNOSAURUS_REX)


@pytest.fixture(scope="module")
def mirror_map(plant):
    return SY.MirrorMap.load(TYRANNOSAURUS_REX, plant)


@pytest.fixture(scope="module")
def stance_kwargs():
    return dict(load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"])


# ---------------------------------------------------------------------------------------------------- signed maps
def test_a_signed_permutation_must_be_an_involution():
    with pytest.raises(SY.SymmetryError, match="involution"):
        SY.SignedPermutation(perm=np.array([1, 2, 0]), sign=np.ones(3))
    with pytest.raises(SY.SymmetryError, match="involution"):
        SY.SignedPermutation(perm=np.array([1, 0]), sign=np.array([1.0, -1.0]))
    with pytest.raises(SY.SymmetryError, match="permutation"):
        SY.SignedPermutation(perm=np.array([0, 0]), sign=np.ones(2))
    with pytest.raises(SY.SymmetryError, match=r"\+1 or -1"):
        SY.SignedPermutation(perm=np.array([0, 1]), sign=np.array([1.0, 0.5]))
    sp = SY.SignedPermutation(perm=np.array([1, 0, 2]), sign=np.array([-1.0, -1.0, 1.0]), names=("a", "b", "c"))
    x = np.arange(6, dtype=np.float32).reshape(2, 3) + 1
    assert sp.apply(x).dtype == np.float32
    np.testing.assert_array_equal(sp.apply(x), [[-2.0, -1.0, 3.0], [-5.0, -4.0, 6.0]])
    np.testing.assert_array_equal(sp.apply(sp.apply(x)), x)
    again = SY.SignedPermutation.from_entries(sp.entries(), "test")
    assert np.array_equal(again.perm, sp.perm) and np.array_equal(again.sign, sp.sign) and again.names == sp.names


@pytest.mark.parametrize(
    ("name", "partner"),
    [("r_hip_roll", "l_hip_roll"), ("l_toe_d2_joint", "r_toe_d2_joint"), ("fr_hip_roll", "fl_hip_roll"),
     ("rl_knee", "rr_knee"), ("right_ankle", "left_ankle"), ("arm_r", "arm_l"), ("neck_yaw", "neck_yaw")],
)  # fmt: skip
def test_mirror_name_swaps_the_side_token(name, partner):
    assert SY.mirror_name(name) == partner
    assert SY.mirror_name(partner) == name


# ---------------------------------------------------------------------------------------------------- derivation
def _kinematics_error(mm: SY.ModelMirror, model: mujoco.MjModel, n: int, seed: int) -> float:
    """Random configurations: the forward kinematics of the mirrored qpos equals the mirror of the kinematics."""
    rng = np.random.default_rng(seed)
    a, b = mujoco.MjData(model), mujoco.MjData(model)
    worst = 0.0
    for _ in range(n):
        q = model.qpos0.copy()
        for j in range(model.njnt):
            adr = int(model.jnt_qposadr[j])
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                q[adr : adr + 3] = rng.uniform(-2, 2, 3)
                quat = rng.normal(size=4)
                q[adr + 3 : adr + 7] = quat / np.linalg.norm(quat)
            else:
                lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-math.pi, math.pi)
                q[adr] = rng.uniform(lo, hi)
        a.qpos[:] = q
        b.qpos[:] = mm.qpos.apply(q)
        mujoco.mj_kinematics(model, a)
        mujoco.mj_kinematics(model, b)
        rotations_a = a.xmat.reshape(-1, 3, 3)
        rotations_b = b.xmat.reshape(-1, 3, 3)
        worst = max(
            worst,
            float(np.max(np.abs(b.xpos[mm.body_map] - a.xpos * SY.POLAR))),
            float(np.max(np.abs(rotations_b[mm.body_map] - SY.S @ rotations_a @ SY.S @ mm.body_C))),
            float(np.max(np.abs(b.site_xpos[mm.site_map] - a.site_xpos * SY.POLAR))),
            float(np.max(np.abs(b.geom_xpos[mm.geom_map] - a.geom_xpos * SY.POLAR))),
        )
    return worst


@pytest.mark.parametrize("model_id", model_ids(), ids=str)
def test_every_anatomical_plant_derives_a_map_and_the_robot_is_refused(model_id):
    model = mujoco.MjModel.from_xml_path(str(model_id.asset_path(SY.REPOSITORY_ROOT / "environments")))
    if model_id == ROBOT:
        # Not mirror-symmetric as authored: one side's ankle chain sits 37 mm from the mirror of the other's, and
        # both hip-roll assemblies are the same (unmirrored) part.
        with pytest.raises(SY.SymmetryError, match="not mirror-symmetric"):
            SY.derive_model_mirror(model)
        return
    mm = SY.derive_model_mirror(model)
    for sp in (mm.qpos, mm.qvel, mm.ctrl):
        assert np.array_equal(sp.perm[sp.perm], np.arange(sp.dim))
    assert _kinematics_error(mm, model, n=20, seed=1) <= 1e-12
    # Every hinge partner's axis is the mirrored one up to the joint's sign: pitch keeps it, roll and yaw flip it.
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            label = SY._axis_label(model.jnt_axis[j])
            if label in ("pitch", "roll", "yaw"):
                assert mm.joint_sign[j] == (1.0 if label == "pitch" else -1.0), mujoco.mj_id2name(model, 3, j)


def _left_knee(model: mujoco.MjModel) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_knee"))


def _shift_the_left_knee_range(model: mujoco.MjModel) -> None:
    model.jnt_range[_left_knee(model), 0] -= 0.01


def _limit_the_left_knee_actuator_force(model: mujoco.MjModel) -> None:
    model.jnt_actfrclimited[_left_knee(model)] = 1
    model.jnt_actfrcrange[_left_knee(model)] = (-500.0, 500.0)


def _compensate_gravity_on_the_left_shank(model: mujoco.MjModel) -> None:
    model.body_gravcomp[model.jnt_bodyid[_left_knee(model)]] = 0.5


def _give_one_left_geom_fluid_drag(model: mujoco.MjModel) -> None:
    geom = next(g for g in range(model.ngeom) if SY._name(model, mujoco.mjtObj.mjOBJ_GEOM, g).startswith("l_"))
    model.geom_fluid[geom, 0] = 1.0


def _dent_a_colliding_tail_mesh(model: mujoco.MjModel) -> None:
    mesh = model.geom_dataid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tail_1_geom")]
    model.mesh_vert[model.mesh_vertadr[mesh], 1] += 1e-3


@pytest.mark.parametrize(
    ("model_id", "edit", "message"),
    [
        (TYRANNOSAURUS_REX, _shift_the_left_knee_range, "range"),
        (TYRANNOSAURUS_REX, _limit_the_left_knee_actuator_force, "actuator force range"),
        (TYRANNOSAURUS_REX, _compensate_gravity_on_the_left_shank, "gravity compensation"),
        (TYRANNOSAURUS_REX, _give_one_left_geom_fluid_drag, "fluid"),
        # The anatomical Compsognathus's tail segments are colliding meshes, each its own mirror image.
        (ModelId("compsognathus_longipes", "anatomical"), _dent_a_colliding_tail_mesh, "vertices"),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_an_asymmetric_pair_is_refused(model_id, edit, message):
    model = mujoco.MjModel.from_xml_path(str(model_id.asset_path(SY.REPOSITORY_ROOT / "environments")))
    SY.derive_model_mirror(model)
    edit(model)
    with pytest.raises(SY.SymmetryError, match=message):
        SY.derive_model_mirror(model)


def _rotation(axis, degrees: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=np.float64), math.radians(degrees))
    return SY._quat_to_mat(quat)


def _numbers(values) -> str:
    return " ".join(repr(float(v)) for v in np.ravel(values))


def _quat(rotation: np.ndarray) -> str:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=np.float64).reshape(-1))
    return _numbers(quat)


#: The CAD-style frame rotations of the synthetic biped's left links (E) and of its left foot site (F).
_E_THIGH, _E_SHANK, _F_SITE = _rotation((1, 2, 3), 40.0), _rotation((0, 0, 1), 90.0), _rotation((0, 1, 0), -90.0)


def _cad_biped_xml(*, correct_the_left_knee: bool = True) -> str:
    """A free-floating biped whose left links are the mirror of the right ones in a rotated link frame.

    Each left body's world frame is ``S R S E`` (``R`` its partner's, ``E`` a proper rotation, as a CAD export that
    orients one side's link frames differently gives), so every local point, frame and hinge axis on the left is
    the mirror one re-expressed through ``E``; the left foot site carries a further quarter turn.  The sides are
    named by suffix tokens (``_r`` / ``_l``).  ``correct_the_left_knee=False`` authors the left knee axis as if the
    frames were pure mirror images.
    """
    s = SY.S
    thigh_rot, thigh_pos = _rotation((0, 1, 0), 10.0), np.array([0.05, -0.2, -0.1])
    knee_rot, knee_pos = _rotation((1, 0, 0), 15.0), np.array([0.02, -0.03, -0.4])  # shank_r in thigh_r's frame
    tilt, site_rot = _rotation((1, 1, 0), 20.0), _rotation((0, 0, 1), 30.0)
    left_thigh_rot = s @ thigh_rot @ s @ _E_THIGH
    left_knee_rot = left_thigh_rot.T @ s @ thigh_rot @ knee_rot @ s @ _E_SHANK
    left_knee_pos = left_thigh_rot.T @ s @ thigh_rot @ knee_pos

    def side(suffix: str, left: bool) -> str:
        def polar(e: np.ndarray, v) -> np.ndarray:  # a right-local point or polar axis, mirrored into the left frame
            return e.T @ s @ np.asarray(v, dtype=np.float64) if left else np.asarray(v, dtype=np.float64)

        def frame(e: np.ndarray, r: np.ndarray) -> np.ndarray:
            return e.T @ s @ r @ s if left else r

        pitch, roll = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        # A hinge axis is a pseudovector: the mirror of a is -E^T S a; the roll axes are authored the other way round.
        knee_axis = -polar(_E_SHANK, pitch) if left and correct_the_left_knee else pitch
        return f"""
      <body name="thigh_{suffix}" pos="{_numbers(s @ thigh_pos if left else thigh_pos)}"
            quat="{_quat(left_thigh_rot if left else thigh_rot)}">
        <joint name="hip_pitch_{suffix}" axis="{_numbers(-polar(_E_THIGH, pitch) if left else pitch)}" damping="0.5"/>
        <joint name="hip_roll_{suffix}" axis="{_numbers(polar(_E_THIGH, roll))}"
               range="{"-20 30" if left else "-30 20"}"/>
        <geom name="thigh_geom_{suffix}" type="capsule" size="0.05 0.18"
              pos="{_numbers(polar(_E_THIGH, [0.01, -0.02, -0.2]))}" quat="{_quat(frame(_E_THIGH, tilt))}"/>
        <geom name="thigh_box_{suffix}" type="box" size="0.03 0.05 0.07"
              pos="{_numbers(polar(_E_THIGH, [0.04, 0.0, -0.05]))}" quat="{_quat(frame(_E_THIGH, tilt))}"/>
        <body name="shank_{suffix}" pos="{_numbers(left_knee_pos if left else knee_pos)}"
              quat="{_quat(left_knee_rot if left else knee_rot)}">
          <joint name="knee_{suffix}" axis="{_numbers(knee_axis)}" range="-80 5"/>
          <geom name="shank_geom_{suffix}" type="capsule" size="0.04 0.16"
                pos="{_numbers(polar(_E_SHANK, [0.0, 0.01, -0.18]))}" quat="{_quat(frame(_E_SHANK, tilt))}"/>
          <site name="foot_{suffix}" size="0.02 0.02 0.02" pos="{_numbers(polar(_E_SHANK, [0.03, -0.01, -0.36]))}"
                quat="{_quat(frame(_E_SHANK, site_rot) @ _F_SITE if left else site_rot)}"/>
        </body>
      </body>"""

    actuators = "".join(
        f'<motor name="{joint}_motor_{side}" joint="{joint}_{side}" gear="40" ctrlrange="-1 1"/>'
        for side in ("r", "l")
        for joint in ("hip_pitch", "hip_roll", "knee")
    )
    sensors = "".join(
        f'<gyro name="foot_gyro_{side}" site="foot_{side}"/><accelerometer name="foot_acc_{side}" site="foot_{side}"/>'
        f'<framepos name="foot_pos_{side}" objtype="site" objname="foot_{side}"/>'
        f'<jointpos name="knee_pos_{side}" joint="knee_{side}"/>'
        for side in ("r", "l")
    )
    return f"""<mujoco model="cad_biped">
  <worldbody>
    <body name="torso" pos="0 0 2">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.3 0.2 0.1"/>{side("r", False)}{side("l", True)}
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
  <sensor>{sensors}</sensor>
</mujoco>"""


def _sensor_mirror_error(mm: SY.ModelMirror, model: mujoco.MjModel, a: mujoco.MjData, b: mujoco.MjData) -> float:
    """How far *b*'s sensor readings are from the mirror of *a*'s (sensor map, component permutation and signs)."""
    worst = 0.0
    for sensor in range(model.nsensor):
        dim, partner = int(model.sensor_dim[sensor]), int(mm.sensor_map[sensor])
        mine, theirs = int(model.sensor_adr[sensor]), int(model.sensor_adr[partner])
        expected = mm.sensor_signs[sensor] * a.sensordata[theirs : theirs + dim][mm.sensor_comp_perm[sensor]]
        worst = max(worst, float(np.max(np.abs(b.sensordata[mine : mine + dim] - expected))))
    return worst


def test_rotated_link_frames_and_suffix_side_tokens_derive_on_a_synthetic_biped():
    """No committed plant needs a frame correction (``C = I`` on all five that derive) or names its sides by suffix,
    so a small biped exercises both: the correction is ``E``, the paired kinematics, dynamics and site-frame sensors
    are equivariant, and a left axis authored without the correction is refused."""
    model = mujoco.MjModel.from_xml_string(_cad_biped_xml())
    mm = SY.derive_model_mirror(model)

    def index(kind, name: str) -> int:
        return int(mujoco.mj_name2id(model, kind, name))

    body, joint = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT
    for link, rotation in (("thigh", _E_THIGH), ("shank", _E_SHANK)):
        right, left = index(body, f"{link}_r"), index(body, f"{link}_l")
        assert (mm.body_map[right], mm.body_map[left]) == (left, right)
        np.testing.assert_allclose(mm.body_C[right], rotation, atol=1e-12)
        np.testing.assert_allclose(mm.body_C[left], SY.S @ rotation.T @ SY.S, atol=1e-12)
    np.testing.assert_allclose(mm.site_C[index(mujoco.mjtObj.mjOBJ_SITE, "foot_r")], _F_SITE, atol=1e-12)
    signs = {name: int(mm.joint_sign[index(joint, f"{name}_{side}")]) for name in ("hip_pitch", "hip_roll", "knee")
             for side in ("r", "l")}  # fmt: skip
    assert signs == {"hip_pitch": 1, "hip_roll": -1, "knee": 1}
    assert _kinematics_error(mm, model, n=20, seed=4) <= 1e-12
    # A mirrored state stays the mirror of the original under the dynamics, sensors included.  The left thigh's
    # compiled inertia agrees with the mirror of the right's to about 3e-9 (MuJoCo diagonalises it in another
    # frame), which bounds the agreement here.
    rng = np.random.default_rng(11)
    a, b = mujoco.MjData(model), mujoco.MjData(model)
    a.qpos[:3] = rng.uniform(-1.0, 1.0, 3)
    a.qpos[3:7] = rng.normal(size=4)
    a.qpos[3:7] /= np.linalg.norm(a.qpos[3:7])
    a.qpos[7:] = rng.uniform(-0.4, 0.1, model.nq - 7)
    a.qvel[:] = rng.normal(0.0, 1.0, model.nv)
    a.ctrl[:] = rng.uniform(-1.0, 1.0, model.nu)
    SY.mirror_physics_state(mm, model, a, b)
    worst_state = worst_sensor = 0.0
    for _ in range(50):
        mujoco.mj_step(model, a)
        mujoco.mj_step(model, b)
        worst_state = max(
            worst_state,
            float(np.max(np.abs(b.qpos - mm.qpos.apply(a.qpos)))),
            float(np.max(np.abs(b.qvel - mm.qvel.apply(a.qvel)))),
        )
        worst_sensor = max(worst_sensor, _sensor_mirror_error(mm, model, a, b))
    assert worst_state <= 1e-7 and worst_sensor <= 1e-6  # measured 5e-9 and 4e-8, against readings up to 60
    with pytest.raises(SY.SymmetryError, match="joint 'knee_r': 0 mirror-consistent partners"):
        SY.derive_model_mirror(mujoco.MjModel.from_xml_string(_cad_biped_xml(correct_the_left_knee=False)))


# ---------------------------------------------------------------------------------------------------- the T. rex map
def test_the_committed_maps_are_current():
    # The CI form of `python -m environments.shared.symmetry --check`.
    assert SY.committed_mirror_maps() == [TYRANNOSAURUS_REX]
    assert SY.stale_mirror_maps() == []
    assert SY.main(["--check"]) == 0


def test_the_tyrannosaurus_rex_map(mirror_map, plant):
    assert (mirror_map.obs.dim, mirror_map.action.dim) == (63, 15) == (plant.observation_dim, plant.action_dim)
    # Actuators: the leg pairs swap; neck yaw, tail_1 yaw and the hip-roll pair are negated.
    assert [int(mirror_map.action.perm[i]) for i in (3, 4, 5, 6)] == [7, 8, 9, 10]
    assert {i for i in range(15) if mirror_map.action.sign[i] < 0} == {1, 4, 8, 12}
    names = mirror_map.obs.names
    flipped = {names[i] for i in range(63) if mirror_map.obs.sign[i] < 0}
    assert {"projected_gravity.y", "heading_linear_velocity.left", "target_direction.y", "pelvis_gyro.x",
            "pelvis_gyro.z", "pelvis_accel.y", "command.v_y_cmd", "command.yaw_rate_cmd"} <= flipped  # fmt: skip
    foot = [i for i, name in enumerate(names) if name.startswith("foot_contact.")]
    assert [int(mirror_map.obs.perm[i]) for i in foot] == foot[::-1]
    record = json.loads(Path(mirror_map.path).read_text())
    assert record["physics_sha256"] == plant.physics_sha256
    assert record["policy_interface_sha256"] == plant.policy_interface_sha256


@pytest.mark.parametrize("key", ["physics_sha256", "policy_interface_sha256", "observation_dim"])
def test_the_map_refuses_another_plant(plant, key):
    identity = plant.to_dict()
    identity[key] = identity[key] + 1 if isinstance(identity[key], int) else identity[key][:-4] + "0000"
    with pytest.raises(SY.SymmetryError, match="another plant|widths"):
        SY.MirrorMap.load(TYRANNOSAURUS_REX, identity)


def test_the_map_fails_closed(plant, tmp_path):
    with pytest.raises(SY.SymmetryError, match="plant identity"):
        SY.MirrorMap.load(TYRANNOSAURUS_REX, None)
    with pytest.raises(SY.SymmetryError, match="must both name"):
        SY.MirrorMap.load(TYRANNOSAURUS_REX, {**plant.to_dict(), "embodiment": "robot"})
    with pytest.raises(SY.SymmetryError, match="cannot read"):
        SY.MirrorMap.load(ModelId("velociraptor_mongoliensis", "anatomical"), plant, configs_root=tmp_path)
    # A hand edit that is still an involution (neck yaw no longer negated) is refused against the derivation.
    source = SY.mirror_map_path(TYRANNOSAURUS_REX)
    copy = SY.mirror_map_path(TYRANNOSAURUS_REX, tmp_path)
    copy.parent.mkdir(parents=True)
    shutil.copyfile(source, copy)
    assert (
        SY.MirrorMap.load(TYRANNOSAURUS_REX, plant, configs_root=tmp_path).sha256
        == SY.MirrorMap.load(TYRANNOSAURUS_REX, plant).sha256
    )
    record = json.loads(source.read_text())
    record["action"][1]["sign"] = 1
    copy.write_text(SY.render_mirror_map(record))
    with pytest.raises(SY.SymmetryError, match="fresh derivation"):
        SY.MirrorMap.load(TYRANNOSAURUS_REX, plant, configs_root=tmp_path)
    assert SY.stale_mirror_maps(tmp_path) and SY.MirrorMap.load(
        TYRANNOSAURUS_REX, plant, configs_root=tmp_path, verify=False
    )


# ---------------------------------------------------------------------------------------------------- equivariance
def _env_mirror(env) -> SY.EnvMirror:
    return SY.build_env_mirror(env, SY.HEADING_FREE_OBSERVATION_SCHEMA)


def _swap_sides(info: dict) -> dict:
    out = {}
    for key, value in info.items():
        if key.startswith("r_"):
            out["l_" + key[2:]] = value
        elif key.startswith("l_"):
            out["r_" + key[2:]] = value
        else:
            out[key] = value
    return out


#: Info keys that are not mirror-invariant scalars: the pelvis yaw rate, the lean term's lateral offset and the
#: stagger term's fore-aft foot lead (screen-only) flip, and the gait history appends "R" before "L" when both feet
#: touch down in one step (gait_symmetry_weight is 0 on every T. rex stage).
_SIGNED_INFO = {"pelvis_yaw_vel": -1.0, "com_lateral_offset_m": -1.0, "foot_stagger_m": -1.0}
_ORDER_DEPENDENT_INFO = {"alternation_ratio", "contact_asymmetry"}


def _paired_rollout(kwargs, seed, actions, mirror_map):
    """Env A from a seeded reset, env B its mirror image; A takes a, B takes M_a a.  Per-step errors."""
    env_a, env_b = TyrannosaurusRexAnatomicalEnv(**kwargs), TyrannosaurusRexAnatomicalEnv(**kwargs)
    try:
        em = _env_mirror(env_a)
        np.testing.assert_array_equal(em.obs.perm, mirror_map.obs.perm)
        np.testing.assert_array_equal(em.action.sign, mirror_map.action.sign)
        obs_a, _ = env_a.reset(seed=seed)
        env_b.reset(seed=seed + 1)
        SY.mirror_env_state(env_a, env_b, em)
        rows = [{"obs": float(np.max(np.abs(env_b._get_obs() - em.obs.apply(obs_a))))}]
        for action in actions:
            obs_a, reward_a, term_a, trunc_a, info_a = env_a.step(action)
            obs_b, reward_b, term_b, trunc_b, info_b = env_b.step(em.action.apply(action))
            swapped = _swap_sides(info_a)
            info_error = max(
                abs(float(value) - _SIGNED_INFO.get(key, 1.0) * float(swapped[key])) / max(1.0, abs(float(value)))
                for key, value in info_b.items()
                if key in swapped and key not in _ORDER_DEPENDENT_INFO and isinstance(value, (int, float, np.floating))
            )
            rows.append(
                {
                    "obs": float(np.max(np.abs(obs_b.astype(np.float64) - em.obs.apply(obs_a).astype(np.float64)))),
                    "reward": abs(float(reward_a) - float(reward_b)),
                    "info": info_error,
                    "lean_paid": info_a.get("reward_lateral_lean", 0.0) > 0.0,
                    "stagger_paid": info_a.get("reward_stance_stagger", 0.0) < 0.0,
                    "band_paid": info_a.get("reward_stance_cop_band", 0.0) < 0.0,
                    "ended": (bool(term_a), bool(trunc_a), info_a.get("termination_reason")),
                    "ended_b": (bool(term_b), bool(trunc_b), info_b.get("termination_reason")),
                }
            )
            if term_a or trunc_a or term_b or trunc_b:
                break
        return rows
    finally:
        env_a.close()
        env_b.close()


def _smooth_actions(n: int, seed: int, amplitude: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(15)
    out = []
    for _ in range(n):
        x += -0.1 * x + 0.12 * rng.normal(size=15)
        out.append(np.clip(amplitude * x / 0.27, -1.0, 1.0))
    return out


#: The T. rex stance options as the 2026-10 screen trained them (lean term and settled leg target on).
_SCREENED_OPTIONS = {
    "lateral_lean_weight": 0.25,
    "lateral_lean_tolerance_m": 0.015,
    "leg_home_pose_reference": "settled",
}
#: Screen-only (stance4, never merged): the screened options plus the standing foot stagger penalty.
_STAGGER_OPTIONS = {**_SCREENED_OPTIONS, "stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045}
#: Scratch prototype (stance4 copband design, never merged): the screened options plus the per-foot CoP band.
_BAND_OPTIONS = {**_SCREENED_OPTIONS, "stance_cop_band_weight": 0.30}


@pytest.mark.parametrize(
    ("options", "steps"),
    # On, the rollout runs past stance_width_settle_steps (200) so the lean and stagger terms' gates open.
    [({}, 40), (_SCREENED_OPTIONS, 260), (_STAGGER_OPTIONS, 260), (_BAND_OPTIONS, 260)],
    ids=["options off", "screened options on", "screened options and stagger on", "screened options and CoP band on"],
)
@pytest.mark.parametrize("seed", [3047, 3059])
def test_the_simulator_is_mirror_equivariant_on_the_stance(stance_kwargs, mirror_map, seed, options, steps):
    """Zero action (the statue) and small smooth actions from a mirrored reset: the mirrored episode is the
    mirror image of the source, its observation to float32 rounding, its reward and info to 1e-9, with the stance
    options off (as every stage has them), with the lean term and the settled leg target on, and with the stagger
    penalty on as well."""
    kwargs = {**stance_kwargs, **options}
    stagger_live = steps > kwargs.get("stance_width_settle_steps", 200) and kwargs.get("stance_stagger_weight", 0) > 0
    band_on = steps > kwargs.get("stance_width_settle_steps", 200) and kwargs.get("stance_cop_band_weight", 0) > 0
    for actions in ([np.zeros(15)] * steps, _smooth_actions(steps, seed, 0.03)):
        rows = _paired_rollout(kwargs, seed, actions, mirror_map)
        assert rows[0]["obs"] == 0.0
        assert rows[1]["obs"] <= 1e-9, "step 1"
        assert len(rows) == steps + 1
        assert max(row["obs"] for row in rows) <= 1e-6
        assert max(row["reward"] for row in rows[1:]) <= 1e-9
        assert max(row["info"] for row in rows[1:]) <= 1e-9
        assert any(row["lean_paid"] for row in rows[1:]) == bool(options)
        assert any(row["stagger_paid"] for row in rows[1:]) == stagger_live
        if not band_on:  # the band pays only steps a foot leaves it; on, the reward and info checks above cover it
            assert not any(row["band_paid"] for row in rows[1:])


def test_termination_is_mirror_invariant(stance_kwargs, mirror_map):
    rows = _paired_rollout(stance_kwargs, 3042, _smooth_actions(400, 5, 0.6), mirror_map)
    assert rows[-1]["ended"][0], "the source episode must end inside the horizon"
    assert all(row["ended"] == row["ended_b"] for row in rows[1:])
    assert max(row["reward"] for row in rows[1:]) <= 1e-9


def test_the_state_mirror_is_an_involution(stance_kwargs):
    env = TyrannosaurusRexAnatomicalEnv(**stance_kwargs)
    try:
        em = _env_mirror(env)
        env.reset(seed=3050)
        for action in _smooth_actions(20, 1, 0.1):
            env.step(action)
        before = {
            name: np.copy(getattr(env.data, name)) for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")
        }
        filter_state = env._action_filter_state.copy()
        SY.mirror_env_state(env, env, em)
        assert not np.array_equal(env.data.qpos, before["qpos"])
        SY.mirror_env_state(env, env, em)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(env.data, name), value)
        np.testing.assert_array_equal(env._action_filter_state, filter_state)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- [ppo] keys
def test_absent_or_none_symmetry_keys_are_stock_ppo():
    kwargs = {"n_steps": 2048}
    assert SY.pop_symmetry_config(kwargs) is None and kwargs == {"n_steps": 2048}
    kwargs = {"symmetry": "none", "symmetry_kappa": 0.5, "batch_size": 128}
    assert SY.pop_symmetry_config(kwargs) is None and kwargs == {"batch_size": 128}


def test_the_symmetry_keys_select_mirror_kl_with_the_screened_defaults():
    kwargs = {"symmetry": "mirror_kl", "batch_size": 128}
    config = SY.pop_symmetry_config(kwargs)
    assert kwargs == {"batch_size": 128}
    # The values the T. rex stance screen trained: changing a default is a recipe change.
    assert config == SY.SymmetryConfig(kappa=0.3, sigma_floor=0.01, mirror_critic=True)
    explicit = SY.pop_symmetry_config(
        {"symmetry": "mirror_kl", "symmetry_kappa": 0, "symmetry_sigma_floor": 0.02, "symmetry_mirror_critic": False}
    )
    assert explicit == SY.SymmetryConfig(kappa=0.0, sigma_floor=0.02, mirror_critic=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"symmetry_kappa": 0.3}, "without it"),
        ({"symmetry": "augment"}, "one of"),
        ({"symmetry": "mirror_kl", "symmetry_kappa": -0.1}, "kappa >= 0"),
        ({"symmetry": "mirror_kl", "symmetry_sigma_floor": 0.0}, "sigma_floor > 0"),
        ({"symmetry": "mirror_kl", "symmetry_kappa": True}, "finite number"),
        ({"symmetry": "mirror_kl", "symmetry_mirror_critic": "yes"}, "true or false"),
        # Off, the settings a table writes are still checked.
        ({"symmetry": "none", "symmetry_kappa": -0.1}, "kappa >= 0"),
        ({"symmetry": "none", "symmetry_mirror_critic": 1}, "true or false"),
    ],
)
def test_bad_symmetry_keys_are_refused(kwargs, message):
    with pytest.raises(SY.SymmetryError, match=message):
        SY.pop_symmetry_config(dict(kwargs))


def test_the_checkpoint_record_and_its_mismatch(mirror_map):
    record = SY.SymmetryConfig().checkpoint_record(mirror_map)
    assert record == {
        "schema": "mesozoic.mirror-kl/v1",
        "method": "mirror_kl",
        "species": "tyrannosaurus_rex",
        "embodiment": "anatomical",
        "mirror_map_sha256": f"sha256:{mirror_map.sha256}",
        "kappa": 0.3,
        "sigma_floor": 0.01,
        "mirror_critic": True,
    }
    assert SY.symmetry_mismatch(None, None) is None
    assert SY.symmetry_mismatch(record, dict(record)) is None
    assert "declares no symmetry" in SY.symmetry_mismatch(record, None)
    assert "records none" in SY.symmetry_mismatch(None, record)
    assert "the stage" in SY.symmetry_mismatch({**record, "kappa": 0.5}, record)
