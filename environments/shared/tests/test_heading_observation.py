"""The heading-free frame helpers (``bipedal-target-heading-free/v1``; docs/HEADING_INVARIANCE_DESIGN_2026_10.md).

Pure frame algebra, no environment: the helpers build ``R`` from the free
root's quaternion, and these tests pin that ``R`` is MuJoCo's, that a global
yaw leaves all three outputs unchanged to the design's float32 budget
(2e-6), including across the yaw wrap, for ``q`` and ``-q`` and at the
vertical-pelvis fallback, that an invalid quaternion is a state error, and
that tilt and target bearing stay visible.  Part of the module is ported
from the design study's ``docs/investigations/heading_2026_10/test_prototype.py``.
"""

from __future__ import annotations

import ast
import inspect
import math

import mujoco
import numpy as np
import pytest

from environments.shared import heading_observation
from environments.shared.heading_observation import (
    HEADING_FREE_FUNCTIONS,
    ObservationStateError,
    body_to_world,
    heading_free_root_segments,
    projected_gravity,
    root_quaternion,
    world_to_heading,
)

#: The design's acceptance budget for a float32 observation of corresponding states.
FLOAT32_BUDGET = 2e-6
#: Global yaws across the wrap, in degrees.
YAWS_DEG = (-179.9, -90.0, -45.0, 17.0, 90.0, 180.0, 359.0)


def _qpos(quaternion) -> np.ndarray:
    qpos = np.zeros(9)
    qpos[0:3] = (0.4, -1.2, 0.9)
    qpos[3:7] = quaternion
    qpos[7:] = (0.3, -0.2)
    return qpos


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)])


def _euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quaternion = np.empty(4)
    mujoco.mju_euler2Quat(quaternion, np.array([yaw, pitch, roll]), "zyx")
    return quaternion


def _turned(qpos: np.ndarray, velocity: np.ndarray, direction: np.ndarray, yaw_rad: float):
    """The same state with the whole scene turned by *yaw_rad* about world vertical."""
    turn = _yaw_quaternion(yaw_rad)
    turned_qpos = qpos.copy()
    mujoco.mju_mulQuat(turned_qpos[3:7], turn, qpos[3:7])
    world = body_to_world(turn)
    return turned_qpos, world @ velocity, world @ direction


def _as_float32(segments) -> np.ndarray:
    return np.concatenate(segments).astype(np.float32)


def test_body_to_world_matches_mujoco():
    rng = np.random.default_rng(18042)
    worst = 0.0
    for _ in range(200):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        reference = np.empty(9)
        mujoco.mju_quat2Mat(reference, quaternion)
        worst = max(worst, float(np.max(np.abs(body_to_world(quaternion) - reference.reshape(3, 3)))))
    assert worst <= 2e-15


def test_root_quaternion_normalizes_and_ignores_sign():
    rng = np.random.default_rng(3042)
    velocity = np.array([0.3, -0.7, 0.2])
    direction = np.array([0.6, 0.0, -0.8])
    for _ in range(50):
        quaternion = rng.normal(size=4)
        unit = quaternion / np.linalg.norm(quaternion)
        np.testing.assert_allclose(root_quaternion(_qpos(3.7 * quaternion)), unit, atol=1e-15)
        reference = _as_float32(heading_free_root_segments(_qpos(unit), velocity, direction))
        for scaled in (3.7 * quaternion, -unit, -0.02 * quaternion):
            np.testing.assert_allclose(
                _as_float32(heading_free_root_segments(_qpos(scaled), velocity, direction)), reference, atol=1e-7
            )
        np.testing.assert_array_equal(
            _as_float32(heading_free_root_segments(_qpos(-unit), velocity, direction)), reference
        )


@pytest.mark.parametrize(
    "qpos",
    [
        _qpos([0.0, 0.0, 0.0, 0.0]),
        _qpos([np.nan, 1.0, 0.0, 0.0]),
        _qpos([1.0, 0.0, np.inf, 0.0]),
        _qpos([1e-13, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9, 1.0, 0.0]),
        np.zeros((7, 2)),
    ],
    ids=["zero", "nan", "inf", "norm-1e-13", "short", "two-dimensional"],
)
def test_invalid_root_quaternion_is_a_state_error(qpos):
    with pytest.raises(ObservationStateError, match=r"qpos\[3:7\]"):
        root_quaternion(qpos)
    with pytest.raises(ObservationStateError):
        heading_free_root_segments(qpos, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert issubclass(ObservationStateError, ValueError)


def test_projected_gravity_is_unit_down_and_keeps_tilt():
    np.testing.assert_array_equal(projected_gravity(body_to_world(np.array([1.0, 0.0, 0.0, 0.0]))), [0.0, 0.0, -1.0])
    # A pure yaw changes nothing; a 30 degree nose-down pitch (+y in MuJoCo) tips gravity toward pelvis +x.
    np.testing.assert_allclose(projected_gravity(body_to_world(_euler(0.0, 0.0, 1.1))), [0.0, 0.0, -1.0], atol=1e-15)
    pitched = projected_gravity(body_to_world(_euler(0.0, math.radians(30.0), 0.0)))
    np.testing.assert_allclose(pitched, [0.5, 0.0, -math.sqrt(3.0) / 2.0], atol=1e-15)
    rolled = projected_gravity(body_to_world(_euler(math.radians(20.0), 0.0, 0.0)))
    np.testing.assert_allclose(rolled, [0.0, -math.sin(math.radians(20.0)), -math.cos(math.radians(20.0))], atol=1e-15)
    rng = np.random.default_rng(5042)
    for _ in range(100):
        quaternion = rng.normal(size=4)
        gravity = projected_gravity(body_to_world(quaternion / np.linalg.norm(quaternion)))
        assert np.linalg.norm(gravity) == pytest.approx(1.0, abs=1e-14)


def test_heading_velocity_is_forward_left_up():
    velocity = np.array([0.4, -0.3, 0.25])
    facing_north = body_to_world(_yaw_quaternion(math.pi / 2.0))
    # Facing world +y: world +y is forward, world -x is left, vertical is untouched.
    np.testing.assert_allclose(world_to_heading(facing_north) @ velocity, [-0.3, -0.4, 0.25], atol=1e-15)
    # A tilt does not tip the heading frame: it stays gravity-aligned.
    tilted = body_to_world(_euler(math.radians(25.0), math.radians(-35.0), math.pi / 2.0))
    heading = world_to_heading(tilted)
    np.testing.assert_allclose(heading[2], [0.0, 0.0, 1.0], atol=0.0)
    np.testing.assert_allclose(heading @ velocity, [-0.3, -0.4, 0.25], atol=1e-15)
    np.testing.assert_allclose(heading @ heading.T, np.eye(3), atol=1e-15)


@pytest.mark.parametrize("yaw_deg", YAWS_DEG)
def test_global_yaw_leaves_segments_invariant(yaw_deg):
    rng = np.random.default_rng(7042)
    worst = 0.0
    for _ in range(100):
        quaternion = _euler(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-math.pi, math.pi))
        if rng.uniform() < 0.5:
            quaternion = -quaternion
        qpos = _qpos(quaternion)
        velocity = rng.normal(scale=0.7, size=3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        before = _as_float32(heading_free_root_segments(qpos, velocity, direction))
        after = _as_float32(heading_free_root_segments(*_turned(qpos, velocity, direction, math.radians(yaw_deg))))
        worst = max(worst, float(np.max(np.abs(after - before))))
    assert worst <= FLOAT32_BUDGET


@pytest.mark.parametrize("pitch", [-math.pi / 2, math.pi / 2, math.pi / 2 - 1e-10, math.pi / 2 - 1e-7])
@pytest.mark.parametrize("yaw", [-math.pi, -math.pi / 2, 0.73, 2.9])
def test_vertical_pelvis_x_fallback_is_finite_and_scene_covariant(pitch, yaw):
    # Pelvis X vertical (or within 1e-7 rad of it): the heading forward axis falls back to pelvis Y.
    qpos = _qpos(_euler(0.0, pitch, 0.2))
    velocity = np.array([0.2, -0.4, 0.3])
    direction = np.array([0.6, 0.8, 0.0])
    before = heading_free_root_segments(qpos, velocity, direction)
    after = heading_free_root_segments(*_turned(qpos, velocity, direction, yaw))
    assert all(np.all(np.isfinite(segment)) for segment in (*before, *after))
    heading = world_to_heading(body_to_world(root_quaternion(qpos)))
    np.testing.assert_allclose(heading @ heading.T, np.eye(3), atol=1e-12)
    assert float(np.max(np.abs(_as_float32(after) - _as_float32(before)))) <= FLOAT32_BUDGET


def test_vertical_fallback_uses_pelvis_y_turned_clockwise():
    # Nose straight down (pitch +90 deg) facing world +y: pelvis Y points to world -x, so forward is world +y.
    rotation = body_to_world(_euler(0.0, math.pi / 2, math.pi / 2))
    assert np.linalg.norm(rotation[:2, 0]) < 1e-8
    np.testing.assert_allclose(world_to_heading(rotation)[0], [0.0, 1.0, 0.0], atol=1e-15)


def test_tilt_and_bearing_changes_stay_distinguishable():
    velocity = np.array([0.1, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    upright = _as_float32(heading_free_root_segments(_qpos(_euler(0.0, 0.0, 0.0)), velocity, direction))
    pitched = _as_float32(heading_free_root_segments(_qpos(_euler(0.0, 0.2, 0.0)), velocity, direction))
    rolled = _as_float32(heading_free_root_segments(_qpos(_euler(0.2, 0.0, 0.0)), velocity, direction))
    assert np.max(np.abs(pitched[0:3] - upright[0:3])) > 0.15
    assert np.max(np.abs(rolled[0:3] - upright[0:3])) > 0.15
    assert np.max(np.abs(rolled[0:3] - pitched[0:3])) > 0.15
    # The animal turned alone, the target left behind: gravity and heading velocity's magnitude stay, the
    # target's bearing in the pelvis frame turns by the same angle the other way.
    turned = _as_float32(
        heading_free_root_segments(
            _qpos(_yaw_quaternion(math.pi / 2)), body_to_world(_yaw_quaternion(math.pi / 2)) @ velocity, direction
        )
    )
    np.testing.assert_allclose(turned[0:6], upright[0:6], atol=1e-7)
    np.testing.assert_allclose(turned[6:9], [0.0, -1.0, 0.0], atol=1e-7)


def test_every_module_function_is_fingerprinted_and_reads_no_module_constant():
    """The plant contract hashes HEADING_FREE_FUNCTIONS' bodies: every function must be listed, and none may
    read a module-level value the hash cannot see (or use an f-string, which the token digest refuses)."""
    defined = [
        name
        for name, value in vars(heading_observation).items()
        if inspect.isfunction(value) and value.__module__ == heading_observation.__name__
    ]
    assert sorted(defined) == sorted(HEADING_FREE_FUNCTIONS)
    allowed = {"np", "ObservationStateError", *HEADING_FREE_FUNCTIONS}
    module_names = set(vars(heading_observation))
    for name in HEADING_FREE_FUNCTIONS:
        tree = ast.parse(inspect.getsource(getattr(heading_observation, name)))
        assert not any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree)), name
        bound = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
        bound |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        read = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        globals_read = (read - bound) & module_names
        assert globals_read <= allowed, (name, sorted(globals_read - allowed))
