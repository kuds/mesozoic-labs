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


def test_an_asymmetric_pair_is_refused():
    model = mujoco.MjModel.from_xml_path(str(TYRANNOSAURUS_REX.asset_path(SY.REPOSITORY_ROOT / "environments")))
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_knee")
    model.jnt_range[joint, 0] -= 0.01
    with pytest.raises(SY.SymmetryError, match="range"):
        SY.derive_model_mirror(model)


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


#: Info keys that are not mirror-invariant scalars: the pelvis yaw rate flips, and the gait history appends "R"
#: before "L" when both feet touch down in one step (gait_symmetry_weight is 0 on every T. rex stage).
_SIGNED_INFO = {"pelvis_yaw_vel": -1.0}
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


@pytest.mark.parametrize("seed", [3047, 3059])
def test_the_simulator_is_mirror_equivariant_on_the_stance(stance_kwargs, mirror_map, seed):
    """Zero action (the statue) and small smooth actions from a mirrored reset: the mirrored episode is the
    mirror image of the source, its observation to float32 rounding, its reward and info to 1e-9."""
    for actions in ([np.zeros(15)] * 40, _smooth_actions(40, seed, 0.03)):
        rows = _paired_rollout(stance_kwargs, seed, actions, mirror_map)
        assert rows[0]["obs"] == 0.0
        assert rows[1]["obs"] <= 1e-9, "step 1"
        assert len(rows) == 41
        assert max(row["obs"] for row in rows) <= 1e-6
        assert max(row["reward"] for row in rows[1:]) <= 1e-9
        assert max(row["info"] for row in rows[1:]) <= 1e-9


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
    # The values the T. rex stance screen trained (arm L): changing a default is a recipe change.
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
