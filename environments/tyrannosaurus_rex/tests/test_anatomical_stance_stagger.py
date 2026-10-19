"""The T. rex's opt-in standing foot stagger penalty (screen-only, never merged): stance_stagger_weight and
stance_stagger_tolerance_m.

Ported for the stance4 screen from the 2026-10-19 night continuation (SP/foot/stance3/night/design/spec.md section
3.1).  It is off by default, and off it changes nothing: no reward, no info key, no trajectory, no task fingerprint
and no recorded stage config.  On, it is -w (1 - exp(-(s/tol)^2)), s the right foot site's lead over the left along
the gravity-aligned pelvis heading, paid only after stance_width_settle_steps and only while the command is zero,
and exactly mirror-invariant.

Every test starts from the stance stage's env kwargs with the two stagger keys removed, so "off" means off whether
or not the screen's stance.toml turns the term on.
"""

from __future__ import annotations

import inspect
import json
import math

import mujoco
import numpy as np
import pytest

from environments.shared import symmetry as SY
from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.model_id import ModelId
from environments.shared.reward_functions import reward_stance_stagger
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import stage_task_fingerprint
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
OPTION_KEYS = ("stance_stagger_weight", "stance_stagger_tolerance_m")
STAGGER_INFO_KEYS = {"foot_stagger_m", "stance_stagger_quality", "reward_stance_stagger"}
#: The values the stance4 screen trains with (spec.md section 3.1).
SCREENED = {"stance_stagger_weight": 0.30, "stance_stagger_tolerance_m": 0.045}


@pytest.fixture(scope="module")
def off_kwargs():
    """The stance stage's env kwargs with the stagger keys removed."""
    kwargs = dict(load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"])
    for key in OPTION_KEYS:
        kwargs.pop(key, None)
    return kwargs


def _pair(kwargs_a, kwargs_b, steps, seed=3052, amplitude=0.05):
    """Two envs, same seed, same actions: per step (reward, info, qpos) for each."""
    envs = [TyrannosaurusRexAnatomicalEnv(**kwargs_a), TyrannosaurusRexAnatomicalEnv(**kwargs_b)]
    try:
        for env in envs:
            env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(steps):
            action = rng.uniform(-amplitude, amplitude, 15)
            out = []
            for env in envs:
                _, reward, terminated, truncated, info = env.step(action)
                out.append((reward, info, env.data.qpos.copy(), terminated or truncated))
            rows.append(out)
            if out[0][3] or out[1][3]:
                break
        return rows
    finally:
        for env in envs:
            env.close()


def _bytes(value) -> bytes:
    return np.float64(value).tobytes()


def _stagger_by_hand(env) -> float:
    """The stagger recomputed from the rotation matrix MuJoCo keeps (not the quaternion the env reads)."""
    data = env.data
    offset = data.site_xpos[env.r_foot_site_id] - data.site_xpos[env.l_foot_site_id]
    x_axis = data.xmat[env.pelvis_id].reshape(3, 3)[:, 0]
    forward = x_axis[:2] / np.linalg.norm(x_axis[:2])
    return float(offset[:2] @ forward)


def _expected(stagger: float, weight: float = 0.30, tolerance: float = 0.045) -> float:
    return -weight * (1.0 - math.exp(-((stagger / tolerance) ** 2)))


# ---------------------------------------------------------------------------------------------------- off by default
def test_the_option_defaults_to_off_and_is_an_opt_in_task_option():
    parameters = inspect.signature(TyrannosaurusRexAnatomicalEnv).parameters
    assert {key: parameters[key].default for key in OPTION_KEYS} == {
        "stance_stagger_weight": 0.0,
        "stance_stagger_tolerance_m": 0.045,
    }
    assert ("stance_stagger_weight", 0.0, OPTION_KEYS) in TyrannosaurusRexAnatomicalEnv.OPT_IN_TASK_KWARGS


def test_off_it_changes_no_reward_no_info_key_and_no_trajectory(off_kwargs):
    explicit_off = {**off_kwargs, "stance_stagger_weight": 0.0, "stance_stagger_tolerance_m": 0.02}
    rows = _pair(off_kwargs, explicit_off, 240)
    assert len(rows) == 240
    for (reward_a, info_a, qpos_a, _), (reward_b, info_b, qpos_b, _) in rows:
        assert _bytes(reward_a) == _bytes(reward_b) == _bytes(info_b["reward_total"])
        assert list(info_a) == list(info_b) and not STAGGER_INFO_KEYS & set(info_a)
        assert all(_bytes(info_a[k]) == _bytes(info_b[k]) for k in info_a if isinstance(info_a[k], float))
        np.testing.assert_array_equal(qpos_a, qpos_b)


def test_off_it_moves_no_task_fingerprint_and_no_recorded_config(tmp_path, off_kwargs):
    base = stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs=off_kwargs)
    explicit_off = {**off_kwargs, "stance_stagger_weight": 0, "stance_stagger_tolerance_m": 0.02}
    assert stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs=explicit_off) == base
    assert not set(OPTION_KEYS) & set(base["env"])
    # Every stage that leaves it off keeps its committed fingerprint with the keys written out at off values.
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        env_kwargs = load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"]
        if env_kwargs.get("stance_stagger_weight", 0.0) == 0.0:
            written = {**env_kwargs, "stance_stagger_weight": 0.0, "stance_stagger_tolerance_m": 0.07}
            committed = stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference)
            assert stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference, env_kwargs=written) == committed, entry.id
    # On, the switch and its tolerance are hashed, the default tolerance included, and nothing else moves.
    on = stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs={**off_kwargs, "stance_stagger_weight": 0.3})
    assert {k: v for k, v in on["env"].items() if k not in base["env"]} == SCREENED
    assert {k: v for k, v in on["env"].items() if k in base["env"]} == base["env"]
    wider = stage_task_fingerprint(
        TYRANNOSAURUS_REX, "stance", env_kwargs={**off_kwargs, **SCREENED, "stance_stagger_tolerance_m": 0.06}
    )
    assert len({base["task_sha256"], on["task_sha256"], wider["task_sha256"]}) == 3
    # stage_config.json records no off option's defaults; on, both keys.
    config = {**load_stage_config(TYRANNOSAURUS_REX, "stance"), "env_kwargs": off_kwargs}
    path = save_stage_config(
        tmp_path / "off", "stance", config, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    assert not set(OPTION_KEYS) & set(json.loads(path.read_text())["reward_weights"])
    config_on = {**config, "env_kwargs": {**off_kwargs, "stance_stagger_weight": 0.3}}
    path = save_stage_config(
        tmp_path / "on", "stance", config_on, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    recorded = json.loads(path.read_text())["reward_weights"]
    assert (recorded["stance_stagger_weight"], recorded["stance_stagger_tolerance_m"]) == (0.3, 0.045)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stance_stagger_weight": -0.1}, "stance_stagger_weight"),
        ({"stance_stagger_weight": math.inf}, "stance_stagger_weight"),
        ({"stance_stagger_weight": math.nan}, "stance_stagger_weight"),
        ({"stance_stagger_tolerance_m": 0.0}, "stance_stagger_tolerance_m"),
        ({"stance_stagger_tolerance_m": -0.01}, "stance_stagger_tolerance_m"),
        ({"stance_stagger_tolerance_m": math.inf}, "stance_stagger_tolerance_m"),
    ],
)
def test_invalid_values_fail_fast(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TyrannosaurusRexAnatomicalEnv(**kwargs)


# ---------------------------------------------------------------------------------------------------- on
def test_on_it_matches_the_formula_on_a_constructed_state(off_kwargs):
    """A built stance: the right hip swung forward and the left back, the pelvis yawed; the step counter past the
    settle and the command zero.  The stagger is the heading-frame lead (yaw does not move it) and the term is
    -w (1 - exp(-(s/tol)^2)); at the settle step, or with a live command, it pays nothing but still reports s."""
    env = TyrannosaurusRexAnatomicalEnv(**off_kwargs, **SCREENED)
    try:
        free = [j for j in range(env.model.njnt) if env.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert len(free) == 1
        quat = slice(int(env.model.jnt_qposadr[free[0]]) + 3, int(env.model.jnt_qposadr[free[0]]) + 7)
        hips = np.asarray(env._leg_home_qpos_indices)[[0, 4]]  # r_hip_pitch, l_hip_pitch
        settle = env.stance_width_settle_steps
        staggers = []
        for yaw in (0.0, 0.9, -2.3):
            env.reset(seed=3042)
            env.data.qpos[env._leg_home_qpos_indices] = env._LEG_SETTLED_QPOS
            env.data.qpos[hips] += (-0.12, 0.12)
            turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
            rotated = np.zeros(4)
            mujoco.mju_mulQuat(rotated, turn, env.data.qpos[quat].copy())
            env.data.qpos[quat] = rotated
            mujoco.mj_forward(env.model, env.data)
            env._step_count = settle + 1
            env._command = np.zeros_like(env._command)
            _, info = env._get_reward_info(np.zeros(15))
            stagger = info["foot_stagger_m"]
            assert stagger == pytest.approx(_stagger_by_hand(env), abs=1e-12)
            assert info["stance_stagger_quality"] == pytest.approx(math.exp(-((stagger / 0.045) ** 2)), rel=1e-12)
            assert info["reward_stance_stagger"] == pytest.approx(_expected(stagger), rel=1e-12)
            staggers.append(stagger)
            # The gate: not at the settle step itself, and not under a live command.
            env._step_count = settle
            _, info = env._get_reward_info(np.zeros(15))
            assert info["reward_stance_stagger"] == 0.0 and info["foot_stagger_m"] == stagger
            env._step_count = settle + 1
            env._command = np.array([0.0, 0.1, 0.0], dtype=env._command.dtype)
            _, info = env._get_reward_info(np.zeros(15))
            assert info["reward_stance_stagger"] == 0.0 and info["stance_stagger_quality"] < 1.0
        assert abs(staggers[0]) > 0.03  # a real stagger, well into the term's slope
        assert staggers[1] == pytest.approx(staggers[0], abs=1e-9)
        assert staggers[2] == pytest.approx(staggers[0], abs=1e-9)
    finally:
        env.close()


def test_on_it_changes_no_dynamics_and_adds_exactly_its_term(off_kwargs):
    rows = _pair(off_kwargs, {**off_kwargs, **SCREENED}, 320)
    assert len(rows) == 320
    settle = off_kwargs.get("stance_width_settle_steps", 200)
    paid = 0
    for step, ((r_off, i_off, q_off, _), (r_on, i_on, q_on, _)) in enumerate(rows, start=1):
        np.testing.assert_array_equal(q_off, q_on)  # a reward term never moves the trajectory
        assert set(i_on) - set(i_off) == STAGGER_INFO_KEYS
        assert _bytes(r_on) == _bytes(r_off + i_on["reward_stance_stagger"]) == _bytes(i_on["reward_total"])
        if step <= settle:
            assert i_on["reward_stance_stagger"] == 0.0 and _bytes(r_on) == _bytes(r_off)
        else:
            assert i_on["reward_stance_stagger"] == pytest.approx(_expected(i_on["foot_stagger_m"]), rel=1e-12)
            assert i_on["reward_stance_stagger"] < 0.0
            paid += 1
    assert paid == 320 - settle


def test_the_stagger_is_the_heading_frame_lead_and_flips_under_the_mirror(off_kwargs):
    env = TyrannosaurusRexAnatomicalEnv(**off_kwargs, **SCREENED)
    try:
        mirror = SY.build_env_mirror(env, SY.HEADING_FREE_OBSERVATION_SCHEMA)
        env.reset(seed=3052)  # the split diagnosis' seed with a +30 mm spawn stagger
        for action in np.random.default_rng(2).uniform(-0.2, 0.2, (40, 15)):
            env.step(action)
        mujoco.mj_forward(env.model, env.data)
        env._step_count = env.stance_width_settle_steps + 1  # the gate open, so the paid reward is compared
        before = env._foot_stagger()
        assert before == pytest.approx(_stagger_by_hand(env), abs=1e-12)
        _, info_before = env._get_reward_info(np.zeros(15))
        SY.mirror_env_state(env, env, mirror)
        after = env._foot_stagger()
        _, info_after = env._get_reward_info(np.zeros(15))
        assert abs(before) > 1e-3
        assert after == pytest.approx(-before, abs=1e-9)
        assert info_after["foot_stagger_m"] == pytest.approx(-info_before["foot_stagger_m"], abs=1e-9)
        for key in ("stance_stagger_quality", "reward_stance_stagger"):
            assert info_after[key] == pytest.approx(info_before[key], abs=1e-9)
        assert info_before["reward_stance_stagger"] < 0.0
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- pure function
@pytest.mark.parametrize("stagger_m", [0.0, 0.01, -0.02, 0.032, 0.044, -0.058, 0.3])
def test_the_pure_function_is_the_spec_formula(stagger_m):
    reward, quality = reward_stance_stagger(np.asarray(stagger_m), 0.045, 0.30)
    assert float(quality) == pytest.approx(math.exp(-((stagger_m / 0.045) ** 2)), rel=1e-12, abs=1e-15)
    assert float(reward) == pytest.approx(_expected(stagger_m), rel=1e-12, abs=1e-15)
    assert -0.30 <= float(reward) <= 0.0
    assert reward_stance_stagger(np.asarray(-stagger_m), 0.045, 0.30) == (reward, quality)


def test_the_slopes_match_the_spec_table():
    """spec.md 3.1: per step, per mm of |s|, 0.0028 / 0.0049 / 0.0057 / 0.0050 / 0.0033 at 10 / 20 / 32 / 44 / 58 mm."""
    h = 1e-7
    for mm, want in ((10, 0.0028), (20, 0.0049), (32, 0.0057), (44, 0.0050), (58, 0.0033)):
        s = mm / 1000.0
        up = float(reward_stance_stagger(np.asarray(s + h), 0.045, 0.30)[0])
        down = float(reward_stance_stagger(np.asarray(s - h), 0.045, 0.30)[0])
        assert -(up - down) / (2 * h) / 1000.0 == pytest.approx(want, abs=6e-5), mm
