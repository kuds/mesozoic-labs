"""Walking pilot 4's heading price (pilot-only, never merged): the maintainer's decisions (23) and (27) of 2026-11-05.

``walk_heading_price_weight`` (default 0.0, off) and ``walk_heading_price_scale_rad`` (default 0.5) add
``r_H = -w * min(|theta| / s, 1)`` to the T. rex reward, ``theta = wrap(pelvis yaw - the command's desired heading)``
in [-pi, pi), read from the CURRENT pelvis quaternion (the state the step reached), not from the command state's
``heading_error``, which base_env.step computes after the reward and so is one control step stale at reward time.
The price needs a live command mode (its reference is the command's desired heading); a stop masks it.

- A: off, nothing moves: rewards, observations, info, trajectories, every T. rex stage's task fingerprint and the
  recorded stage config; the keys are validated (numbers only, in range); on, both keys are hashed.
- B: the term: the formula and its saturation, a left and a right turn cost the same, the configured weight and
  scale (not the pilot's alone), the wrap at the half turn,
  the current pelvis yaw against the desired heading (not the stale heading error), a stop, the info keys only when
  on, and the price added to the total and nothing else changed.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.direction_commands import wrap_angle
from environments.shared.model_id import ModelId
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import stage_task_fingerprint
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.tyrannosaurus_rex.base_env import TyrannosaurusRexEnv, _walk_heading_price, _walk_pelvis_yaw

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
REPOSITORY = Path(__file__).resolve().parents[3]
PRICE_KEYS = ("walk_heading_price_weight", "walk_heading_price_scale_rad")
PRICE_INFO_KEYS = {"walk_heading_error_rad", "reward_walk_heading_price"}
#: The pilot's price (decision (27)): weight 0.2, scale 0.5 rad.
PRICE = {"walk_heading_price_weight": 0.2, "walk_heading_price_scale_rad": 0.5}
#: The pilot's straight command (decisions (21)-(28)) as its TOML table writes it.
STRAIGHT_TABLE: dict[str, Any] = {
    "turn_increment_max": 0.0,
    "cruise_speed": 1.5,
    "speed_scale": 1.5,
    "lateral_speed_scale": 1.0,
    "yaw_rate_scale": 0.6,
    "yaw_gain": 1.5,
    "yaw_rate_max": 0.3,
    "turn_slowdown": True,
    "minimum_turn_speed_fraction": 0.25,
    "stop_probability": 0.0,
    "straight_probability": 1.0,
}
#: The T. rex stages whose committed [env] turns the price on.
PRICED_STAGES: set[str] = {"locomotion"}
#: Every locomotion [env] key walking pilot 4 adds.
PILOT4_ENV_KEYS = ("command_mode", "command_config", "prey_lateral_range", *PRICE_KEYS)


def _locomotion_without_pilot4() -> dict[str, Any]:
    """The committed locomotion [env] without pilot 4's keys (walking pilot 3's [env])."""
    env_kwargs = load_stage_config(TYRANNOSAURUS_REX, "locomotion")["env_kwargs"]
    return {key: value for key, value in env_kwargs.items() if key not in PILOT4_ENV_KEYS}


def _live(**extra: Any) -> dict[str, Any]:
    """Pilot 3's locomotion [env] with pilot 4's straight command and prey line (and *extra*)."""
    return {
        **_locomotion_without_pilot4(),
        "command_mode": "heading",
        "command_config": dict(STRAIGHT_TABLE),
        "prey_lateral_range": (0.0, 0.0),
        **extra,
    }


def _same(a: Any, b: Any) -> bool:
    """Equal values of one type (arrays element by element; a NaN equals a NaN)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b, equal_nan=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b) and type(a) is type(b)


def _roll(kwargs: dict[str, Any], seed: int, steps: int = 240, amplitude: float = 0.05) -> list[tuple[Any, ...]]:
    env = TyrannosaurusRexAnatomicalEnv(**kwargs)
    try:
        obs, info = env.reset(seed=seed)
        rows: list[tuple[Any, ...]] = [(obs, 0.0, False, False, info, env.data.qpos.copy())]
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-amplitude, amplitude, 15))
            rows.append((obs, reward, terminated, truncated, info, env.data.qpos.copy()))
            if terminated or truncated:
                break
        return rows
    finally:
        env.close()


# ======================================================================================== A. off moves nothing
def test_the_price_keys_default_to_off_and_only_the_priced_stages_set_them():
    defaults = inspect.signature(TyrannosaurusRexAnatomicalEnv).parameters
    assert {key: defaults[key].default for key in PRICE_KEYS} == {
        "walk_heading_price_weight": 0.0,
        "walk_heading_price_scale_rad": 0.5,
    }
    assert ("walk_heading_price_weight", 0.0, PRICE_KEYS) in TyrannosaurusRexEnv.OPT_IN_TASK_KWARGS
    priced = set()
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        env_kwargs = load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"]
        if set(PRICE_KEYS) & set(env_kwargs):
            priced.add(entry.id)
    assert priced == PRICED_STAGES
    for path in sorted((REPOSITORY / "configs").rglob("*.toml")):
        text = path.read_text(encoding="utf-8")
        if any(key in text for key in PRICE_KEYS):
            assert path.relative_to(REPOSITORY).as_posix() == "configs/tyrannosaurus_rex/anatomical/locomotion.toml"


@pytest.mark.parametrize("stage", ["stance", "locomotion_live", "locomotion_pilot3"])
@pytest.mark.parametrize("seed", [3042, 3043])
def test_off_the_price_changes_no_reward_no_info_key_and_no_trajectory(stage, seed):
    """Off (the keys absent) equals an explicit 0.0 weight with any scale, bit for bit, with no price info key."""
    kwargs = {
        "stance": lambda: dict(load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"]),
        "locomotion_live": _live,
        "locomotion_pilot3": _locomotion_without_pilot4,
    }[stage]()
    kwargs = {key: value for key, value in kwargs.items() if key not in PRICE_KEYS}
    default = _roll(kwargs, seed)
    explicit = _roll({**kwargs, "walk_heading_price_weight": 0.0, "walk_heading_price_scale_rad": 0.7}, seed)
    assert len(default) == len(explicit) > 1
    for (obs_a, reward_a, te_a, tr_a, info_a, q_a), (obs_b, reward_b, te_b, tr_b, info_b, q_b) in zip(
        default, explicit
    ):
        assert np.array_equal(obs_a, obs_b) and np.array_equal(q_a, q_b)
        assert reward_a == reward_b and (te_a, tr_a) == (te_b, tr_b)
        assert list(info_a) == list(info_b) and not PRICE_INFO_KEYS & set(info_a)
        assert all(_same(info_a[key], info_b[key]) for key in info_a)


def test_off_the_price_moves_no_task_fingerprint_and_no_recorded_config(tmp_path):
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        env_kwargs = dict(load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"])
        if set(PRICE_KEYS) & set(env_kwargs):
            continue  # a stage that turns the price on (walking pilot 4's locomotion) is checked below
        committed = stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference)
        explicit_off = {**env_kwargs, "walk_heading_price_weight": 0.0, "walk_heading_price_scale_rad": 0.9}
        assert stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference, env_kwargs=explicit_off) == committed
        assert not set(PRICE_KEYS) & set(committed["env"]), entry.id
    # On, both keys are hashed, the scale at its default too: five settings, five tasks.
    digests = [
        stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=_live(**extra))["task_sha256"]
        for extra in (
            {},
            {"walk_heading_price_weight": 0.2},
            {"walk_heading_price_weight": 0.1},
            {"walk_heading_price_weight": 0.2, "walk_heading_price_scale_rad": 0.6},
            {"walk_heading_price_weight": 0.0, "walk_heading_price_scale_rad": 0.6},
        )
    ]
    assert len(set(digests[:4])) == 4 and digests[4] == digests[0]
    on = stage_task_fingerprint(TYRANNOSAURUS_REX, "locomotion", env_kwargs=_live(walk_heading_price_weight=0.2))
    assert {key: on["env"][key] for key in PRICE_KEYS} == PRICE
    # stage_config.json records neither key while off and both when on.
    config = load_stage_config(TYRANNOSAURUS_REX, "locomotion")
    for name, extra, expect in (("off", {}, None), ("on", PRICE, PRICE)):
        env_kwargs = {**_live(), **extra}
        path = save_stage_config(
            tmp_path / name,
            "locomotion",
            {**config, "env_kwargs": env_kwargs},
            "PPO",
            env_class=TyrannosaurusRexAnatomicalEnv,
            model_id=TYRANNOSAURUS_REX,
        )
        recorded = json.loads(path.read_text())["reward_weights"]
        if expect is None:
            assert not set(PRICE_KEYS) & set(recorded)
        else:
            assert {key: recorded[key] for key in PRICE_KEYS} == expect


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"walk_heading_price_weight": -0.1}, "walk_heading_price_weight"),
        ({"walk_heading_price_weight": math.inf}, "walk_heading_price_weight"),
        ({"walk_heading_price_weight": math.nan}, "walk_heading_price_weight"),
        ({"walk_heading_price_scale_rad": 0.0}, "walk_heading_price_scale_rad"),
        ({"walk_heading_price_scale_rad": -0.5}, "walk_heading_price_scale_rad"),
        ({"walk_heading_price_scale_rad": math.inf}, "walk_heading_price_scale_rad"),
        ({"walk_heading_price_scale_rad": math.nan}, "walk_heading_price_scale_rad"),
        ({"walk_heading_price_weight": 0.2}, "needs a live command_mode"),
    ],
)
def test_invalid_price_settings_fail_fast(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TyrannosaurusRexAnatomicalEnv(**kwargs)


@pytest.mark.parametrize("key", PRICE_KEYS)
@pytest.mark.parametrize("value", [True, False, "0.2", None])
def test_a_price_key_must_be_a_number(key, value):
    """A bool passed the range checks as 0 or 1 (the weight True acted as 1.0) and a string failed them with an
    unhelpful TypeError (both measured by the code review of 2026-11-06): each is refused by name."""
    with pytest.raises(ValueError, match=f"{key} must be a number"):
        TyrannosaurusRexAnatomicalEnv(**_live(**{**PRICE, key: value}))


# ======================================================================================== B. the term
def test_the_formula_saturates_at_the_scale_and_is_even():
    """r_H = -w * min(|theta| / s, 1): 0 at 0, linear to -w at |theta| = s, flat at -w beyond, the same at -theta."""
    w, s = PRICE["walk_heading_price_weight"], PRICE["walk_heading_price_scale_rad"]
    assert _walk_heading_price(0.0, w, s) == 0.0
    for theta, expected in ((0.1, -0.04), (0.25, -0.1), (0.5, -0.2), (0.7, -0.2), (math.pi, -0.2), (-math.pi, -0.2)):
        assert _walk_heading_price(theta, w, s) == pytest.approx(expected, abs=1e-15)
        assert _walk_heading_price(-theta, w, s) == _walk_heading_price(theta, w, s)
    grid = np.linspace(-math.pi, math.pi, 4001)
    values = np.array([_walk_heading_price(float(t), w, s) for t in grid])
    mirrored = np.array([_walk_heading_price(-float(t), w, s) for t in grid])
    np.testing.assert_array_equal(values, mirrored)
    assert values.max() == 0.0 and values.min() == -w
    assert np.all(np.diff(values[grid >= 0]) <= 0.0), "the price grows with the error"


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _scored_at_yaw(env: Any, yaw: float) -> dict[str, Any]:
    """Turn the root about the world vertical so the pelvis yaw becomes *yaw*, then score the posed state as step()
    would (the digest harness's state probe) without a physics step and without touching the command."""
    current = _walk_pelvis_yaw(env.data.sensordata[env._sensor_quat_start : env._sensor_quat_start + 4])
    env.data.qpos[3:7] = _quaternion_product(_yaw_quaternion(yaw - current), env.data.qpos[3:7])
    mujoco.mj_forward(env.model, env.data)
    env._invalidate_substep_aggregates()
    info: dict[str, Any] = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))[1]
    return info


def test_a_left_and_a_right_turn_cost_the_same_and_saturate():
    """Posed at +psi (left of the commanded heading) and -psi (right): theta = +-psi, one price for both, the formula's
    value, flat beyond the scale."""
    env = TyrannosaurusRexAnatomicalEnv(**_live(**PRICE))
    try:
        for psi in (0.05, 0.2, 0.5, 0.9, 2.5):
            prices = {}
            for sign in (1.0, -1.0):
                env.reset(seed=3042)
                state = env._command_state
                assert state is not None and state.desired_heading == pytest.approx(0.0, abs=1e-12)
                info = _scored_at_yaw(env, state.desired_heading + sign * psi)
                assert info["walk_heading_error_rad"] == pytest.approx(sign * psi, abs=1e-9)
                prices[sign] = info["reward_walk_heading_price"]
            assert prices[1.0] == pytest.approx(prices[-1.0], abs=1e-12)
            assert prices[1.0] == pytest.approx(-0.2 * min(psi / 0.5, 1.0), abs=1e-9)
    finally:
        env.close()


@pytest.mark.parametrize("weight, scale", [(0.2, 0.5), (0.1, 0.25), (0.35, 1.0)])
def test_the_price_uses_the_configured_weight_and_scale(weight, scale):
    """Every other 'on' test uses the pilot's 0.2 / 0.5, so a price that ignored either key (the weight or the scale
    hard-coded: the code review's two surviving mutants) passed them all: -w * min(psi / s, 1) at other settings,
    below, at and beyond each scale, on both sides."""
    env = TyrannosaurusRexAnatomicalEnv(**_live(walk_heading_price_weight=weight, walk_heading_price_scale_rad=scale))
    try:
        for psi in (0.05, 0.2, 0.25, 0.5, 0.9, 2.5):
            for sign in (1.0, -1.0):
                env.reset(seed=3042)
                state = env._command_state
                assert state is not None
                info = _scored_at_yaw(env, state.desired_heading + sign * psi)
                expected = -weight * min(psi / scale, 1.0)
                assert info["reward_walk_heading_price"] == pytest.approx(expected, abs=1e-9), (weight, scale, psi)
    finally:
        env.close()


def test_the_error_wraps_into_minus_pi_to_pi():
    """The desired heading set just past -pi and the pelvis posed just short of +pi: theta is the short way round,
    -0.2 rad (not 2 pi - 0.2), and the exact half turn is -pi, the controller's own convention."""
    env = TyrannosaurusRexAnatomicalEnv(**_live(**PRICE, command_mode="heading_and_speed"))
    try:
        env.reset(seed=3043)
        assert env.direction_controller is not None
        env.direction_controller.set_target(-math.pi + 0.1, 1.5)
        env._command_state = env.direction_controller.update(0.0, env._heading())
        info = _scored_at_yaw(env, math.pi - 0.1)
        assert info["walk_heading_error_rad"] == pytest.approx(-0.2, abs=1e-9)
        assert info["reward_walk_heading_price"] == pytest.approx(-0.2 * 0.4, abs=1e-9)
        env.direction_controller.set_target(0.0, 1.5)
        env._command_state = env.direction_controller.update(0.0, env._heading())
        info = _scored_at_yaw(env, math.pi)
        assert -math.pi <= info["walk_heading_error_rad"] < math.pi
        assert abs(info["walk_heading_error_rad"]) == pytest.approx(math.pi, abs=1e-9)
        assert info["reward_walk_heading_price"] == pytest.approx(-0.2, abs=1e-12)
    finally:
        env.close()
    assert wrap_angle(math.pi) == -math.pi and wrap_angle(-math.pi) == -math.pi


def test_the_reference_is_the_command_s_desired_heading():
    """Under a turned request (an external target), theta is the pelvis yaw against THAT heading, not the spawn's."""
    env = TyrannosaurusRexAnatomicalEnv(**_live(**PRICE, command_mode="heading_and_speed"))
    try:
        env.reset(seed=3042)
        assert env.direction_controller is not None
        spawn = env._heading()
        env.direction_controller.set_target(spawn + 1.0, 1.5)
        env._command_state = env.direction_controller.update(0.0, env._heading())
        info = _scored_at_yaw(env, spawn + 0.75)
        assert info["walk_heading_error_rad"] == pytest.approx(-0.25, abs=1e-9)
        assert info["reward_walk_heading_price"] == pytest.approx(-0.1, abs=1e-9)
    finally:
        env.close()


def test_the_price_reads_the_current_pelvis_yaw_not_the_stale_heading_error():
    """Posed without a step, the command state still holds the reset's heading error (about 0) while the price reads
    the posed yaw.  Over a real episode, each step's theta is the yaw the step reached against the desired heading,
    which equals minus the heading error of the command state base_env.step computes AFTER the reward, and differs from
    the state's error at reward time (the state before the step)."""
    env = TyrannosaurusRexAnatomicalEnv(**_live(**PRICE))
    try:
        env.reset(seed=3042)
        state = env._command_state
        assert state is not None and abs(state.heading_error) < 1e-12
        info = _scored_at_yaw(env, 0.3)
        assert env._command_state is state and abs(state.heading_error) < 1e-12
        assert info["walk_heading_error_rad"] == pytest.approx(0.3, abs=1e-9)
        assert info["reward_walk_heading_price"] == pytest.approx(-0.12, abs=1e-9)

        env.reset(seed=3043)
        rng = np.random.default_rng(3043)
        stale_gaps, steps = [], 0
        for _ in range(400):
            before = env._command_state
            assert before is not None
            _, _, terminated, truncated, info = env.step(rng.uniform(-0.3, 0.3, 15))
            after = env._command_state
            assert after is not None and after is not before
            theta = info["walk_heading_error_rad"]
            yaw = env._heading()
            assert theta == pytest.approx(wrap_angle(yaw - after.desired_heading), abs=1e-12)
            assert theta == pytest.approx(-after.heading_error, abs=1e-12)
            assert info["reward_walk_heading_price"] == _walk_heading_price(theta, 0.2, 0.5)
            stale_gaps.append(abs(theta + before.heading_error))
            steps += 1
            if terminated or truncated:
                break
        assert steps >= 50
        assert max(stale_gaps) > 1e-6, "the step turned the pelvis, so the stale error is a step behind"
    finally:
        env.close()


def test_a_stop_masks_the_price():
    """A stop request (heading_and_speed, stop_probability 1) has no heading objective: theta 0, price 0."""
    table = {**STRAIGHT_TABLE, "stop_probability": 1.0}
    env = TyrannosaurusRexAnatomicalEnv(**_live(**PRICE, command_mode="heading_and_speed", command_config=table))
    try:
        env.reset(seed=3042)
        assert env._command_state is not None and not env._command_state.heading_active
        info = _scored_at_yaw(env, 1.0)
        assert info["walk_heading_error_rad"] == 0.0 and info["reward_walk_heading_price"] == 0.0
        env._command_state = None
        info = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))[1]
        assert info["walk_heading_error_rad"] == 0.0 and info["reward_walk_heading_price"] == 0.0
    finally:
        env.close()


@pytest.mark.parametrize("seed", [3042, 3043])
def test_on_the_price_is_added_to_the_total_and_changes_nothing_else(seed):
    """On and off (the same live [env]) roll the same trajectory; each step's reward differs by exactly the price,
    the price's two info keys are added at the end of the touchdown block and every other info item is equal."""
    off, on = _roll(_live(), seed, steps=300, amplitude=0.2), _roll(_live(**PRICE), seed, steps=300, amplitude=0.2)
    assert len(off) == len(on) > 1
    nonzero = 0
    for row, ((obs_a, reward_a, te_a, tr_a, info_a, q_a), (obs_b, reward_b, te_b, tr_b, info_b, q_b)) in enumerate(
        zip(off, on)
    ):
        assert np.array_equal(obs_a, obs_b) and np.array_equal(q_a, q_b) and (te_a, tr_a) == (te_b, tr_b)
        if row == 0:
            assert not PRICE_INFO_KEYS & set(info_b)  # the reset's info carries no reward
            continue
        assert set(info_b) - set(info_a) == PRICE_INFO_KEYS and not PRICE_INFO_KEYS & set(info_a)
        keys_b = [key for key in info_b if key not in PRICE_INFO_KEYS]
        assert keys_b == list(info_a)
        position = list(info_b).index("walk_heading_error_rad")
        assert list(info_b)[position - 1] == "reward_walk_touchdown"
        assert list(info_b)[position + 1] == "reward_walk_heading_price"
        assert all(_same(info_a[key], info_b[key]) for key in info_a if key != "reward_total")
        price = info_b["reward_walk_heading_price"]
        assert reward_b - reward_a == pytest.approx(price, abs=1e-12)
        assert info_b["reward_total"] - info_a["reward_total"] == pytest.approx(price, abs=1e-12)
        nonzero += price < 0.0
    assert nonzero > 0


@pytest.mark.parametrize("price", [{}, PRICE])
def test_the_total_is_the_env_order_sum_with_the_price_only_when_on(price):
    """Each step's reward is R6's env-order sum (the 29 legacy terms, then the airborne substeps and the touchdown
    event, as test_anatomical_walk_r6.py rebuilds it) plus the price only while it is on: nothing is added while off."""
    from environments.tyrannosaurus_rex.tests.test_anatomical_walk_r6 import ORDER

    rows = _roll(_live(**price), 3042, steps=300, amplitude=0.2)
    assert len(rows) > 50
    for m, (_, reward, terminated, _, info, _) in enumerate(rows[1:]):
        total = info[ORDER[0]]
        for key in ORDER[1:]:
            total = total + info[key]
        total = total + info["reward_airborne_substeps"] + info["reward_walk_touchdown"]
        if price:
            total = total + info["reward_walk_heading_price"]
        if terminated and not info.get("success"):
            total += -150.0
        assert reward == total, m


def test_the_pelvis_yaw_is_the_heading_the_command_reads():
    """The quaternion yaw the price reads equals the root body's heading the controller reads (BaseDinoEnv._heading)
    on walking states, so the price and the command measure one angle."""
    env = TyrannosaurusRexAnatomicalEnv(**_live(**PRICE))
    try:
        env.reset(seed=3042)
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(rng.uniform(-0.3, 0.3, 15))
            quat = env.data.sensordata[env._sensor_quat_start : env._sensor_quat_start + 4]
            worst = max(worst, abs(wrap_angle(_walk_pelvis_yaw(quat) - env._heading())))
            if terminated or truncated:
                break
        assert worst < 1e-12
    finally:
        env.close()
