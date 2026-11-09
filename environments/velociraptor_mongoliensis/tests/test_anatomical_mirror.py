"""The life-size raptor is mirror-exact, and its policy interface mirrors with it (physics r5 / policy interface r14).

``environments/shared/symmetry.py`` derives a plant's mirror maps from the compiled model
and its observation layout; the T-Rex's map is committed and its paired rollouts are pinned
in ``environments/shared/tests/test_symmetry.py``.  The raptor commits no map (it trains
with stock PPO), but the plant was rebuilt as the T-Rex's equivalent, the maintainer's
decision of 2026-11-08, so a later mirror-KL recipe (the T-Rex's LTSB, a follow-up) can
derive one: these pin that the derivation holds on the r5 plant (exact r <-> l geometry,
the six per-site foot channels swapping pad for pad, digit for digit) and that the
simulator is mirror-equivariant on the stance config, its reward and info included.
"""

import mujoco
import numpy as np
import pytest

from environments.shared import symmetry as SY
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.velociraptor_mongoliensis.anatomical.env import VelociraptorMongoliensisAnatomicalEnv

MODEL_ID = ModelId("velociraptor_mongoliensis", "anatomical")


@pytest.fixture(scope="module")
def stance_kwargs():
    return load_stage_config(MODEL_ID, "stance")["env_kwargs"]


@pytest.fixture(scope="module")
def env_mirror():
    env = VelociraptorMongoliensisAnatomicalEnv()
    try:
        yield SY.build_env_mirror(env, SY.HEADING_FREE_OBSERVATION_SCHEMA)
    finally:
        env.close()


def test_the_home_pose_is_mirror_exact(env_mirror):
    """At the keyframe every body, geom and site is its partner's mirror image (to 1e-12 m)."""
    model = mujoco.MjModel.from_xml_path(str(VelociraptorMongoliensisAnatomicalEnv.MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_kinematics(model, data)
    mm = env_mirror.model
    assert float(np.max(np.abs(data.xpos[mm.body_map] - data.xpos * SY.POLAR))) <= 1e-12
    assert float(np.max(np.abs(data.geom_xpos[mm.geom_map] - data.geom_xpos * SY.POLAR))) <= 1e-12
    assert float(np.max(np.abs(data.site_xpos[mm.site_map] - data.site_xpos * SY.POLAR))) <= 1e-12
    # The mirror maps every r_ element onto its l_ partner and each midline element onto itself.
    for kind, mapping, count in (
        (mujoco.mjtObj.mjOBJ_BODY, mm.body_map, model.nbody),
        (mujoco.mjtObj.mjOBJ_GEOM, mm.geom_map, model.ngeom),
    ):
        for index in range(count):
            name = mujoco.mj_id2name(model, kind, index) or ""
            partner = mujoco.mj_id2name(model, kind, int(mapping[index])) or ""
            assert partner == SY.mirror_name(name), (name, partner)


def test_the_policy_maps_swap_the_legs_and_the_per_site_feet(env_mirror):
    obs, action = env_mirror.obs, env_mirror.action
    assert (obs.dim, action.dim) == (71, 17)
    for sp in (obs, action):
        assert np.array_equal(sp.perm[sp.perm], np.arange(sp.dim)), "not an involution"
    # Actuators: neck 0-2, right leg 3-7 (hip pitch, hip roll, knee, ankle, claw), left leg 8-12, tail 13-16.  The
    # legs swap; neck yaw, both hip rolls and tail_1 yaw are negated.
    assert [int(action.perm[i]) for i in range(3, 8)] == list(range(8, 13))
    assert {i for i in range(17) if action.sign[i] < 0} == {1, 4, 9, 14}
    assert all(int(action.perm[i]) == i for i in (0, 1, 2, 13, 14, 15, 16))
    # The six per-site foot channels: the right pad, d3, d4 swap with the left pad, d3, d4, each one sensor.
    foot = [i for i, name in enumerate(obs.names) if name.startswith("foot_contact.")]
    assert [obs.names[i] for i in foot] == [
        f"foot_contact.{side}_{site}_touch" for side in "rl" for site in ("foot", "toe_d3", "toe_d4")
    ]
    assert [int(obs.perm[i]) for i in foot] == foot[3:] + foot[:3]
    assert all(obs.sign[i] == 1.0 for i in foot)
    flipped = {obs.names[i] for i in range(71) if obs.sign[i] < 0}
    assert {
        "projected_gravity.y",
        "heading_linear_velocity.left",
        "target_direction.y",
        "pelvis_gyro.x",
        "pelvis_gyro.z",
        "pelvis_accel.y",
        "command.v_y_cmd",
        "command.yaw_rate_cmd",
        "qpos.neck_yaw",
        "qpos.tail_1_yaw",
    } <= flipped


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


#: Info keys that are not mirror-invariant scalars: the pelvis yaw rate flips sign, and the gait history appends "R"
#: before "L" when both feet touch down in one step (gait_symmetry_weight is 0 on the stance).
_SIGNED_INFO = {"pelvis_yaw_vel": -1.0}
_ORDER_DEPENDENT_INFO = {"alternation_ratio", "contact_asymmetry"}


def _paired_rollout(kwargs, seed, actions, em):
    """Env A from a seeded reset, env B its mirror image; A takes a, B takes M_a a.  Per-step errors."""
    env_a = VelociraptorMongoliensisAnatomicalEnv(**kwargs)
    env_b = VelociraptorMongoliensisAnatomicalEnv(**kwargs)
    try:
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
    x = np.zeros(17)
    out = []
    for _ in range(n):
        x += -0.1 * x + 0.12 * rng.normal(size=17)
        out.append(np.clip(amplitude * x / 0.27, -1.0, 1.0))
    return out


@pytest.mark.parametrize("seed", [3047, 3059])
def test_the_simulator_is_mirror_equivariant_on_the_stance(stance_kwargs, env_mirror, seed):
    """Zero action (the statue) and small smooth actions from a mirrored reset, on the stance config (its four
    opt-in terms on): the mirrored episode is the mirror image of the source, its observation to float32 rounding,
    its reward and info to 1e-9."""
    steps = 150
    for actions in ([np.zeros(17)] * steps, _smooth_actions(steps, seed, 0.03)):
        rows = _paired_rollout(stance_kwargs, seed, actions, env_mirror)
        assert rows[0]["obs"] == 0.0
        assert len(rows) == steps + 1
        assert max(row["obs"] for row in rows) <= 1e-6
        assert max(row["reward"] for row in rows[1:]) <= 1e-9
        assert max(row["info"] for row in rows[1:]) <= 1e-9


def test_termination_is_mirror_invariant(stance_kwargs, env_mirror):
    rows = _paired_rollout(stance_kwargs, 3042, _smooth_actions(400, 5, 0.6), env_mirror)
    assert rows[-1]["ended"][0], "the source episode must end inside the horizon"
    assert all(row["ended"] == row["ended_b"] for row in rows[1:])
    assert max(row["reward"] for row in rows[1:]) <= 1e-9
