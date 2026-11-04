"""Walking pilot 3's R6 walking reward (pilot-only, never merged): the maintainer's decision (9) of 2026-11-04.

R6 is four parts.  Two are existing knobs (airborne_substep_weight, forward_vel_max).  Two are new and off by
default: ``walk_support_gate_bw`` pays alive, posture and height only on a step whose two feet's substep-MIN touch
exceeds that multiple of body weight (the compsognathus rule), and ``walk_touchdown_weight`` pays the SCREEN stage's
alternating-touchdown event R5p (the frozen gait probe's debounced touchdowns; the time since that foot's previous
touchdown, capped at 1.2 s; scaled by its lead over the other foot in tenths of the probe's leg length; nothing for a
simultaneous landing), one control step later than SCREEN paid it so that no pay reads a step not yet taken.

- A: off, the env is the branch head 3ae1c959 bit for bit (rewards, observations, info, trajectories, the task
  fingerprint and the recorded stage config), against a golden recorded on an export of 3ae1c959.
- B: the support flag, the down rule and the touchdown event on synthetic inputs: the probe's debounced touchdown set
  exhaustively, SCREEN's own events_probe (ported verbatim) one step later, no read of a future step.
- C: the env's R6 terms against an offline recompute of its own episodes, and the R6 returns of SCREEN's recorded
  hand-built episodes (fixtures/walk_r6/screen_cases.npz) against SCREEN's published R6p.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import math
import platform
import struct
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.model_id import ModelId
from environments.shared.reward_functions import reward_forward_velocity
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import _effective_env_kwargs, stage_task_fingerprint
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv
from environments.tyrannosaurus_rex.base_env import _WalkTouchdownEvents

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "walk_r6"
PROBE_PATH = REPOSITORY / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"
R6_KEYS = ("walk_support_gate_bw", "walk_touchdown_weight")
R6_INFO_KEYS = {
    "walk_support",
    "r_foot_down",
    "l_foot_down",
    "walk_touchdowns_registered",
    "walk_touchdown_unit",
    "reward_walk_touchdown",
}
#: The pilot's four [env] values (decision 9 (a)-(d)).
PILOT = {
    "forward_vel_max": 1.5,
    "airborne_substep_weight": 1.0,
    "walk_support_gate_bw": 0.04,
    "walk_touchdown_weight": 0.1,
}
#: The env's 29-term sum, in its order (base_env.py, "Total reward").
ORDER = (
    "reward_forward",
    "reward_backward",
    "reward_drift",
    "reward_alive",
    "reward_energy",
    "reward_tail",
    "reward_bite",
    "reward_approach",
    "reward_head_proximity",
    "reward_head_clearance",
    "reward_neck_posture",
    "reward_tail_home_pose",
    "reward_posture",
    "reward_nosedive",
    "reward_height",
    "reward_bilateral_support",
    "reward_foot_load_balance",
    "reward_leg_home_pose",
    "reward_foot_flatness",
    "reward_stance_width",
    "reward_gait",
    "reward_smoothness",
    "reward_action_jerk",
    "reward_action_saturation",
    "reward_heading",
    "reward_lateral",
    "reward_spin",
    "reward_speed",
    "reward_idle",
)
DT = 0.01
STRIDE = 0.08766


@functools.cache
def _probe() -> Any:
    """The frozen gait probe, loaded by path (read only)."""
    spec = importlib.util.spec_from_file_location("gait_probe_frozen_walk_r6", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def screen_events_probe(down: np.ndarray, dx: np.ndarray, dt: float, ell: float, pay_delay: int) -> np.ndarray:
    """The SCREEN stage's ``events_probe`` (R5p), verbatim but for its inputs and the pay delay (3 = SCREEN).

    *down* holds the (T, 2) down flags (right, left); *dx* the (T + 1, 2) leads per state row (row 0 the reset), each
    ``(fp[m, i, :2] - fp[m, j, :2]) @ fwd2d(quat)[m]`` as SCREEN computed it; *ell* is 0.1 x the probe's leg length.
    """
    probe = _probe()
    T, F = down.shape
    tds = []
    for i in range(F):
        td, _ = probe._events(probe.debounce(down[:, i]))
        tds += [(int(k), i) for k in td]
    tds.sort()
    E = np.zeros(T)
    last: list[int | None] = [None, None]
    last_foot = None
    for k, i in tds:
        j = 1 - i
        alt = last_foot == j
        near = any(ii == j and abs(kk - k) <= 2 for kk, ii in tds)
        previous = last[i]
        Ti = (k - previous) * dt if previous is not None else k * dt
        m_ = min(k + 2, T)
        S = float(np.clip(float(dx[m_, i]) / ell, 0.0, 1.0))
        if alt and not near and k + pay_delay < T:
            E[k + pay_delay] += min(Ti, 1.2) / (2.0 * dt) * S
        last[i] = k
        last_foot = i
    return E


def fwd2d(q: np.ndarray) -> np.ndarray:
    """The SCREEN stage's heading (screen.py ``fwd2d``), verbatim."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    v = np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)], axis=-1)
    return np.asarray(v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9))


def track(
    down: np.ndarray, dx: np.ndarray | None = None, stride: float = STRIDE, tracker: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a whole episode into a fresh tracker: per-step (pay, registered).  *dx* per state row as above."""
    T = len(down)
    dx = np.full((T + 1, 2), 0.2) if dx is None else dx
    tracker = _WalkTouchdownEvents(DT, stride) if tracker is None else tracker
    tracker.reset()
    pays, registered = np.zeros(T), np.zeros(T, dtype=int)
    for m in range(T):
        pays[m], registered[m] = tracker.update(m, bool(down[m, 0]), bool(down[m, 1]), dx[m + 1, 0], dx[m + 1, 1])
    return pays, registered


def _locomotion_as_3ae1c959() -> dict[str, Any]:
    """The committed locomotion [env] with the pilot's four R6 lines as they were at 3ae1c959 (2.5, unset, unset,
    unset)."""
    kwargs = {
        k: v for k, v in load_stage_config(TYRANNOSAURUS_REX, "locomotion")["env_kwargs"].items() if k not in PILOT
    }
    kwargs["forward_vel_max"] = 2.5
    return kwargs


def _same(a: Any, b: Any) -> bool:
    """Equal values of one type (arrays element by element; a NaN equals a NaN)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b, equal_nan=True))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return type(a) is type(b)
    return bool(a == b) and type(a) is type(b)


# ======================================================================================== A. off is the branch head
def test_the_r6_keys_default_to_off_and_no_committed_config_sets_them():
    parameters = inspect.signature(TyrannosaurusRexAnatomicalEnv).parameters
    assert {key: parameters[key].default for key in R6_KEYS} == dict.fromkeys(R6_KEYS, 0.0)
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        env_kwargs = load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"]
        assert not set(R6_KEYS) & set(env_kwargs), entry.id
    for path in sorted((REPOSITORY / "configs" / "tyrannosaurus_rex").rglob("*.toml")):
        assert not [key for key in R6_KEYS if key in path.read_text()], path


def _roll(kwargs: dict[str, Any], seed: int, steps: int = 240, amplitude: float = 0.05) -> list[tuple[Any, ...]]:
    env = TyrannosaurusRexAnatomicalEnv(**kwargs)
    try:
        obs, info = env.reset(seed=seed)
        rows: list[tuple[Any, ...]] = [(obs, 0.0, False, False, info)]
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-amplitude, amplitude, 15))
            rows.append((obs, reward, terminated, truncated, info))
            if terminated or truncated:
                break
        return rows
    finally:
        env.close()


@pytest.mark.parametrize("stage", ["stance", "locomotion_as_3ae1c959"])
@pytest.mark.parametrize("seed", [3042, 3043])
def test_off_r6_changes_no_reward_no_info_key_and_no_trajectory(stage, seed):
    kwargs = (
        _locomotion_as_3ae1c959()
        if stage == "locomotion_as_3ae1c959"
        else dict(load_stage_config(TYRANNOSAURUS_REX, stage)["env_kwargs"])
    )
    default = _roll(kwargs, seed)
    explicit = _roll({**kwargs, **dict.fromkeys(R6_KEYS, 0.0)}, seed)
    assert len(default) == len(explicit) == 241
    for (obs_a, reward_a, te_a, tr_a, info_a), (obs_b, reward_b, te_b, tr_b, info_b) in zip(default, explicit):
        assert np.array_equal(obs_a, obs_b) and reward_a == reward_b and (te_a, tr_a) == (te_b, tr_b)
        assert list(info_a) == list(info_b) and not R6_INFO_KEYS & set(info_a)
        assert all(_same(info_a[key], info_b[key]) for key in info_a), [
            key for key in info_a if not _same(info_a[key], info_b[key])
        ]


def test_off_r6_moves_no_task_fingerprint_and_no_recorded_config(tmp_path):
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        committed = stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference)
        env_kwargs = load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"]
        if set(R6_KEYS) & set(env_kwargs):
            continue  # a stage that turns R6 on (walking pilot 3's locomotion) is checked below
        explicit_off = {**env_kwargs, **dict.fromkeys(R6_KEYS, 0.0)}
        assert stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference, env_kwargs=explicit_off) == committed, (
            entry.id
        )
        assert not set(R6_KEYS) & set(committed["env"]), entry.id
    stance = dict(load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"])
    # On, the switch is hashed; the two switches and two on-values give four different tasks.
    gate_on = _effective_env_kwargs(TYRANNOSAURUS_REX, {**stance, "walk_support_gate_bw": 0.04})
    assert gate_on["walk_support_gate_bw"] == 0.04 and "walk_touchdown_weight" not in gate_on
    digests = {
        stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs={**stance, **extra})["task_sha256"]
        for extra in (
            {},
            {"walk_support_gate_bw": 0.04},
            {"walk_support_gate_bw": 0.05},
            {"walk_touchdown_weight": 0.1},
            {"walk_touchdown_weight": 0.2},
        )
    }
    assert len(digests) == 5
    # stage_config.json records neither key while off, and the key a config sets when on.
    config = load_stage_config(TYRANNOSAURUS_REX, "stance")
    path = save_stage_config(
        tmp_path / "off", "stance", config, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    recorded = json.loads(path.read_text())["reward_weights"]
    assert not set(R6_KEYS) & set(recorded) and "airborne_substep_weight" in recorded
    config_on = {**config, "env_kwargs": {**config["env_kwargs"], **{key: PILOT[key] for key in R6_KEYS}}}
    path = save_stage_config(
        tmp_path / "on", "stance", config_on, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    recorded = json.loads(path.read_text())["reward_weights"]
    assert {key: recorded[key] for key in R6_KEYS} == {key: PILOT[key] for key in R6_KEYS}


# ---------------------------------------------------------------------------------------- ROLL (the golden's generator)
# Verbatim from the implement stage's gen_off_golden.py, which recorded fixtures/walk_r6/off_golden_3ae1c959.json on an
# export of 3ae1c95967547ada569ad71aebaa88999afa02cd (its sha256 is in the fixture).
SEEDS = (3042, 3043)
UNIFORM_STEPS, UNIFORM_AMPLITUDE, ZERO_STEPS, ALT_STEPS = 300, 0.1, 100, 400


def alt_action(k: int, dt: float) -> np.ndarray:
    """The smoke stage's ALT_f1.0_A0.15_lead+ (smoke/scripted/scripted_rollout.py, grid 2) at control step k."""
    t, g, w = k * dt, min(1.0, k / 100), 2.0 * math.pi * 1.0
    a = np.zeros(15)
    for hip, knee, ankle, off in ((3, 5, 6, 0.0), (7, 9, 10, math.pi)):
        ph = w * t + off
        a[hip] = g * 0.15 * math.sin(ph)
        a[knee] = g * 0.3 * math.sin(ph + math.pi / 2)
        a[ankle] = g * 1.0 * 0.3 * math.sin(ph + math.pi / 2)
    return np.clip(a, -1.0, 1.0)


def roll(env: Any, seed: int, kind: str) -> dict[str, Any]:
    """One episode's per-stream sha256 digests (observation float32 bytes, reward float64, flags, every info item's
    repr in key order, qpos and qvel float64 bytes), the reset's included."""
    streams = {name: hashlib.sha256() for name in ("obs", "reward", "terminated", "truncated", "info", "state")}

    def feed(obs: Any, info: dict[str, Any]) -> None:
        streams["obs"].update(np.asarray(obs).dtype.str.encode() + np.asarray(obs).tobytes())
        streams["info"].update(repr(sorted(info.items())).encode())
        streams["state"].update(env.unwrapped.data.qpos.tobytes() + env.unwrapped.data.qvel.tobytes())

    obs, info = env.reset(seed=seed)
    feed(obs, info)
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        actions = [
            rng.uniform(-UNIFORM_AMPLITUDE, UNIFORM_AMPLITUDE, 15).astype(np.float32) for _ in range(UNIFORM_STEPS)
        ]
        actions += [np.zeros(15, dtype=np.float32)] * ZERO_STEPS
    else:
        actions = [alt_action(k, float(env.unwrapped.dt)) for k in range(ALT_STEPS)]
    steps, total = 0, 0.0
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        feed(obs, info)
        streams["reward"].update(struct.pack("<d", float(reward)))
        streams["terminated"].update(bytes([bool(terminated)]))
        streams["truncated"].update(bytes([bool(truncated)]))
        steps, total = steps + 1, total + float(reward)
        if terminated or truncated:
            break
    return {"steps": steps, "return": total, "digests": {name: h.hexdigest() for name, h in streams.items()}}


def environment() -> dict[str, Any]:
    """What the digests bind: interpreter, platform, numpy (and its enabled CPU features), MuJoCo, gymnasium."""
    import gymnasium

    umath = (
        "numpy._core._multiarray_umath" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core._multiarray_umath"
    )
    features: dict[str, bool] = importlib.import_module(umath).__cpu_features__
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "numpy_cpu_features": ",".join(sorted(name for name, on in features.items() if on)),
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
    }


# ---------------------------------------------------------------------------------------- end of ROLL
def test_off_r6_is_the_branch_head_bitwise():
    """Off (the keys absent), every scenario rolls exactly as 3ae1c959 rolled it: stance, recovery (pushes), the
    locomotion [env] as at 3ae1c959 and a default-constructed env, two seeds, seeded noise then zero action and the
    smoke's open-loop alternating pattern."""
    golden = json.loads((FIXTURES / "off_golden_3ae1c959.json").read_text())
    assert golden["source_commit"] == "3ae1c95967547ada569ad71aebaa88999afa02cd"
    configs = {
        "stance": dict(load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"]),
        "recovery": dict(load_stage_config(TYRANNOSAURUS_REX, "recovery")["env_kwargs"]),
        "locomotion": _locomotion_as_3ae1c959(),
        "default": {},
    }
    # The configs are 3ae1c959's: an [env] change anywhere but R6 would show here first.
    assert {name: json.loads(json.dumps(kwargs)) for name, kwargs in configs.items()} == golden["env_kwargs"]
    here = environment()
    differing = sorted(key for key, value in golden["environment"].items() if here.get(key) != value)
    if differing:
        pytest.skip(
            "the golden digests bind the environment they were made in; this one differs in "
            + ", ".join(f"{key} ({here.get(key)!r}, golden {golden['environment'][key]!r})" for key in differing)
        )
    mismatched = {}
    for name, kwargs in configs.items():
        for seed, kind in itertools.product(SEEDS, ("uniform", "alt")):
            env = TyrannosaurusRexAnatomicalEnv(**kwargs)
            try:
                result = roll(env, seed, kind)
            finally:
                env.close()
            if result != golden["scenarios"][f"{name}/{kind}/{seed}"]:
                mismatched[f"{name}/{kind}/{seed}"] = result
    assert not mismatched, sorted(mismatched)


@pytest.mark.parametrize("key", R6_KEYS)
@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_invalid_r6_values_fail_fast(key, value):
    with pytest.raises(ValueError, match=key):
        TyrannosaurusRexAnatomicalEnv(**{key: value})


# ======================================================================================== B. the pieces
def _scored(env: Any) -> dict[str, Any]:
    info: dict[str, Any] = env._get_reward_info(np.zeros(env.action_space.shape, dtype=np.float32))[1]
    return info


def test_the_support_flag_is_strict_and_reads_four_percent_of_body_weight():
    kwargs = _locomotion_as_3ae1c959()
    gated, plain = (TyrannosaurusRexAnatomicalEnv(**kwargs, walk_support_gate_bw=g) for g in (0.04, 0.0))
    try:
        threshold = 0.04 * gated._body_weight_n
        assert threshold == pytest.approx(33.636528) and gated.walk_support_gate_bw * gated._body_weight_n == threshold
        for forces, supported in (
            ((16.0, 17.0), False),
            ((20.0, 20.0), True),
            ((threshold, 0.0), False),
            ((threshold / 2, threshold / 2), False),
            ((float(np.nextafter(threshold, np.inf)), 0.0), True),
        ):
            infos = []
            for env in (gated, plain):
                # A few steps first, so the pelvis tilts and the posture penalty is not zero; then the step's
                # aggregates are dropped, so the support terms read the instantaneous (patched) forces.
                env.__dict__.pop("_foot_contact_forces", None)
                env.reset(seed=3042)
                for _ in range(30):
                    env.step(np.zeros(15, dtype=np.float32))
                env._invalidate_substep_aggregates()
                env._foot_contact_forces = lambda forces=forces: forces
                infos.append(_scored(env))
            on, off = infos
            assert on["walk_support"] == float(supported) and "walk_support" not in off, forces
            for key in ("reward_alive", "reward_posture", "reward_height"):
                assert off[key] != 0.0
                assert on[key] == (off[key] if supported else 0.0), (forces, key)
            for key in set(off) - {"reward_alive", "reward_posture", "reward_height", "reward_total"}:
                assert _same(on[key], off[key]), (forces, key)
            if supported:
                assert on["reward_total"] == off["reward_total"]
            else:
                lost = off["reward_alive"] + off["reward_posture"] + off["reward_height"]
                assert on["reward_total"] == pytest.approx(off["reward_total"] - lost, abs=1e-12)
            assert on["raw_alive"] == off["raw_alive"] and on["tilt_angle"] == off["tilt_angle"]
    finally:
        gated.close()
        plain.close()


def test_a_foot_is_down_on_at_least_three_of_five_substeps():
    env = TyrannosaurusRexAnatomicalEnv(walk_touchdown_weight=0.1)
    try:
        env.reset(seed=3042)
        for block, expected in (
            ([[0.2, 0.0], [0.2, 0.0], [0.2, 0.0], [0.0, 0.2], [0.0, 0.2]], (1.0, 0.0)),
            ([[0.0, 9.0], [0.2, 9.0], [0.0, 0.0], [5.0, 0.0], [0.0, 9.0]], (0.0, 1.0)),
            ([[0.1, 0.1]] * 5, (0.0, 0.0)),
            ([[np.nextafter(0.1, 1.0), 0.0]] * 5, (1.0, 0.0)),
            ([[0.2, 0.05]], (1.0, 0.0)),
        ):
            env._substep_foot_force_block = lambda block=block: np.asarray(block, dtype=np.float64)
            info = _scored(env)
            assert (info["r_foot_down"], info["l_foot_down"]) == expected, block
    finally:
        env.close()


def _touchdowns(registered: np.ndarray) -> list[int]:
    """The registered touchdown steps (k = the registration step - 2) of a one-foot run."""
    return [int(m) - 2 for m in np.flatnonzero(registered) for _ in range(int(registered[m]))]


def test_registration_is_the_probes_debounced_touchdown_set():
    probe = _probe()
    sequences: list[np.ndarray] = [
        np.array(bits, dtype=bool) for length in range(4, 13) for bits in itertools.product((0, 1), repeat=length)
    ]
    assert len(sequences) == 8176
    rng = np.random.default_rng(3042)
    for _ in range(2000):
        runs = np.repeat(rng.integers(0, 2, 60).astype(bool), rng.integers(1, 12, 60))[:200]
        flicker = rng.random(len(runs)) < 0.08
        sequences.append(runs ^ flicker)
    for seq in sequences:
        T = len(seq)
        down = np.stack([seq, np.zeros(T, dtype=bool)], axis=1)
        _, registered = track(down)
        expected = [int(k) for k in probe._events(probe.debounce(seq))[0] if 1 <= k <= T - 3]
        assert _touchdowns(registered) == expected, seq.astype(int).tolist()


def _gait(T: int, *feet: tuple[int, int, int]) -> np.ndarray:
    """(T, 2) down flags: per foot (first touchdown step, steps down, steps up), repeating."""
    down = np.zeros((T, 2), dtype=bool)
    for foot, (start, on, off) in enumerate(feet):
        m = np.arange(T)
        down[:, foot] = (m >= start) & ((m - start) % (on + off) < on)
    return down


def test_the_event_rule_on_hand_written_gaits():
    def unit(interval: float, scale: float = 1.0) -> float:
        return min(interval, 1.2) / (2.0 * DT) * scale

    # (a) a clean alternating walk: R lands at 10, 60, ..., L at 35, 85, ...; every touchdown after the first is
    # paid at k + 4, the first of each foot timed from the reset.
    T = 400
    pays, registered = track(_gait(T, (10, 30, 20), (35, 30, 20)))
    expected = np.zeros(T)
    last = {0: None, 1: None}
    for k, foot in sorted([(k, 0) for k in range(10, T, 50)] + [(k, 1) for k in range(35, T, 50)]):
        previous = last[foot]
        if (k, foot) != (10, 0) and k + 4 < T:
            expected[k + 4] = unit((k - previous) * DT if previous is not None else k * DT)
        last[foot] = k
    assert np.array_equal(pays, expected) and pays[39] == pytest.approx(17.5) and pays[64] == pytest.approx(25.0)
    assert registered.sum() == len(range(10, T - 2, 50)) + len(range(35, T - 2, 50))

    # (b) both feet landing 0-2 steps apart pays neither; 3 apart pays the later (the earlier is the first touchdown).
    for gap in range(5):
        down = np.zeros((60, 2), dtype=bool)
        down[20:, 0] = True
        down[20 + gap :, 1] = True
        pays, registered = track(down)
        assert registered.sum() == 2
        assert np.flatnonzero(pays).tolist() == ([] if gap <= 2 else [20 + gap + 4]), gap
    # ... and with an earlier L touchdown, so that R's would alternate: within 2 steps neither, 3 apart both.
    for gap in range(5):
        down = np.zeros((80, 2), dtype=bool)
        down[5:15, 1] = True
        down[40:, 0] = True
        down[40 + gap :, 1] = True
        pays, _ = track(down)
        assert np.flatnonzero(pays).tolist() == ([] if gap <= 2 else [44, 40 + gap + 4]), gap

    # (c) the same foot twice in a row is not paid the second time.
    down = np.zeros((80, 2), dtype=bool)
    down[5:15, 1] = True
    down[20:30, 0] = True
    down[40:50, 0] = True
    pays, registered = track(down)
    assert _touchdowns(registered) == [5, 20, 40] and np.flatnonzero(pays).tolist() == [24]

    # (d) the lead is read at the state after step k + 1: <= 0 pays 0, half a tenth of L pays half, more pays all.
    for lead, scale in ((-0.3, 0.0), (0.0, 0.0), (STRIDE / 2, 0.5), (STRIDE, 1.0), (0.5, 1.0)):
        dx = np.full((81, 2), -1.0)
        dx[22, 0] = lead  # row k + 2 = the state after step k + 1, for R's touchdown at k = 20
        pays, _ = track(down, dx)
        assert pays[24] == unit(20 * DT, scale) and pays.sum() == pays[24], lead

    # (e) the interval is capped at 1.2 s.
    down = np.zeros((400, 2), dtype=bool)
    down[5:15, 1] = True
    down[300:310, 0] = True
    pays, _ = track(down)
    assert pays[304] == 1.2 / (2.0 * DT) and pays.sum() == pays[304]

    # (f) a one-step bounce before a landing is one touchdown at its first step (the probe's gap fill); a one-step
    # blip is none.
    for pattern, touchdowns in (([0, 0, 1, 0, 1, 1, 1, 1], [2]), ([0, 0, 1, 0, 0, 0, 0, 0], [])):
        down = np.zeros((8, 2), dtype=bool)
        down[:, 0] = pattern
        assert _touchdowns(track(down)[1]) == touchdowns

    # (g) step 0 is never a touchdown: a foot down from the reset does not land.
    down = np.ones((50, 2), dtype=bool)
    assert track(down)[1].sum() == 0

    # (h) a touchdown whose pay step lies at or beyond the episode's last step is never paid ...
    down = _gait(60, (10, 30, 20), (35, 30, 20))
    assert track(down[:39])[0].sum() == 0 and track(down[:40])[0][39] > 0

    # (i) ... and reset() forgets everything: the first touchdown of a new episode does not alternate, and each
    # foot's first interval is timed from that reset.
    tracker = _WalkTouchdownEvents(DT, STRIDE)
    track(_gait(400, (10, 30, 20), (35, 30, 20)), tracker=tracker)
    down = np.zeros((40, 2), dtype=bool)
    down[3:8, 1] = True
    down[12:, 0] = True
    pays, registered = track(down, tracker=tracker)
    assert _touchdowns(registered) == [3, 12] and np.flatnonzero(pays).tolist() == [16] and pays[16] == unit(12 * DT)


def _random_gaits(rng: np.random.Generator, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Two feet, each a random up/down run sequence with flicker; random leads in [-0.1, 0.2] m."""
    down = np.zeros((T, 2), dtype=bool)
    for foot in (0, 1):
        runs = np.repeat(rng.integers(0, 2, T).astype(bool), rng.integers(1, 40, T))[:T]
        down[:, foot] = runs ^ (rng.random(T) < 0.03)
    return down, rng.uniform(-0.1, 0.2, (T + 1, 2))


def test_the_tracker_is_screens_events_probe_one_step_later():
    rng = np.random.default_rng(3043)
    paid = 0
    for _ in range(500):
        down, dx = _random_gaits(rng, 300)
        pays, _ = track(down, dx)
        e3 = screen_events_probe(down, dx, DT, STRIDE, 3)
        e4 = screen_events_probe(down, dx, DT, STRIDE, 4)
        assert np.array_equal(pays, e4)
        assert np.array_equal(e4[1:], e3[:-1])  # SCREEN's pays, one step later; its pay at the last step (k = T-4) lost
        paid += int(np.count_nonzero(pays))
    assert paid > 1000


def test_the_tracker_reads_no_future_step():
    rng = np.random.default_rng(3044)
    for _ in range(200):
        down, dx = _random_gaits(rng, 200)
        pays, registered = track(down, dx)
        cut = int(rng.integers(0, 200))
        other_down, other_dx = _random_gaits(rng, 200)
        down2, dx2 = down.copy(), dx.copy()
        down2[cut + 1 :], dx2[cut + 1 :] = other_down[cut + 1 :], other_dx[cut + 1 :]
        pays2, registered2 = track(down2, dx2)
        assert np.array_equal(pays[: cut + 1], pays2[: cut + 1])
        assert np.array_equal(registered[: cut + 1], registered2[: cut + 1])


def test_rescoring_a_step_is_idempotent_and_a_gap_fails():
    tracker = _WalkTouchdownEvents(DT, STRIDE)
    down = _gait(60, (10, 30, 20), (35, 30, 20))
    outputs = [tracker.update(m, *map(bool, down[m]), 0.2, 0.2) for m in range(45)]
    assert outputs[39][0] > 0
    assert tracker.update(44, False, True, -5.0, -5.0) == outputs[44]
    tracker.update(45, *map(bool, down[45]), 0.2, 0.2)
    with pytest.raises(RuntimeError, match="step 47 after step 45"):
        tracker.update(47, False, False, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="step 3 after step 45"):
        tracker.update(3, False, False, 0.0, 0.0)
    tracker.reset()
    with pytest.raises(RuntimeError, match="step -1 after step -1"):
        tracker.update(-1, False, False, 0.0, 0.0)
    assert tracker.update(0, True, True, 0.0, 0.0) == (0.0, 0)
    # Through the env: a read before the first step (m = -1) registers nothing, pays nothing and leaves the event
    # untouched; scoring a step again out of band repeats its R6 info.
    env = TyrannosaurusRexAnatomicalEnv(**{**_locomotion_as_3ae1c959(), **PILOT})
    try:
        env.reset(seed=3042)
        info = _scored(env)
        assert {key: info[key] for key in R6_INFO_KEYS - {"walk_support", "r_foot_down", "l_foot_down"}} == {
            "walk_touchdowns_registered": 0.0,
            "walk_touchdown_unit": 0.0,
            "reward_walk_touchdown": 0.0,
        }
        assert env._walk_touchdowns is not None and env._walk_touchdowns._last_step == -1
        for _ in range(3):
            _, _, _, _, stepped = env.step(np.zeros(15, dtype=np.float32))
        again = _scored(env)
        assert {key: again[key] for key in R6_INFO_KEYS} == {key: stepped[key] for key in R6_INFO_KEYS}
        assert env._walk_touchdowns._last_step == 2
    finally:
        env.close()


def test_the_stride_scale_is_a_tenth_of_the_probes_leg_length():
    env = TyrannosaurusRexAnatomicalEnv(walk_touchdown_weight=0.1)
    try:
        description = _probe().RobotDescription(env)
        assert env._walk_touchdowns is not None
        assert env._walk_touchdowns.stride_scale_m == 0.1 * description.leg_length
        assert description.leg_length == pytest.approx(0.8766, abs=1e-4) and env._walk_touchdowns.dt == env.dt == DT
        # The leads are read from the probe's own foot sites and quaternion.
        assert description.primary_site == [env.r_foot_site_id, env.l_foot_site_id]
        assert description.quat_sensor_start == env._sensor_quat_start
    finally:
        env.close()
    off = TyrannosaurusRexAnatomicalEnv()
    try:
        assert off._walk_touchdowns is None
    finally:
        off.close()


# ======================================================================================== C. integration
def _alt_or_hop(kind: str, f: float, a_hip: float, a_knee: float, a_ankle: float, lead: float, sign: float) -> Any:
    """The smoke stage's open-loop patterns (smoke/scripted/scripted_rollout.py ``Pattern``), as a function of k."""

    def action(k: int) -> np.ndarray:
        t, g, w = k * DT, min(1.0, k / 100), 2.0 * math.pi * f
        a = np.zeros(15)
        for hip, knee, ankle, off in ((3, 5, 6, 0.0), (7, 9, 10, math.pi if kind == "ALT" else 0.0)):
            ph = w * t + off
            a[hip] = g * a_hip * math.sin(ph)
            a[knee] = g * a_knee * math.sin(ph + lead)
            a[ankle] = g * sign * a_ankle * math.sin(ph + lead)
        return np.clip(a, -1.0, 1.0)

    return action


@pytest.mark.parametrize(
    "pattern",
    [("ALT", 1.5, 0.15, 0.3, 0.3, math.pi / 2, 1.0), ("HOP", 2.0, 0.0, 0.4, 0.4, 0.0, 1.0)],
    ids=["ALT_f1.5_A0.15_lead+", "HOP_f2.0_ankle+"],
)
@pytest.mark.parametrize("seed", [3042, 3043])
def test_the_env_r6_terms_equal_an_offline_recompute_of_its_own_episode(pattern, seed):
    env = TyrannosaurusRexAnatomicalEnv(**{**_locomotion_as_3ae1c959(), **PILOT})
    action = _alt_or_hop(*pattern)
    try:
        env.reset(seed=seed)
        sites = [np.array([env.data.site_xpos[env.r_foot_site_id], env.data.site_xpos[env.l_foot_site_id]])]
        quats = [env.data.sensordata[env._sensor_quat_start : env._sensor_quat_start + 4].copy()]
        rows = []
        for k in range(400):
            _, reward, terminated, truncated, info = env.step(action(k))
            rows.append((reward, terminated, info, env._substep_foot_force_block()))
            sites.append(np.array([env.data.site_xpos[env.r_foot_site_id], env.data.site_xpos[env.l_foot_site_id]]))
            quats.append(env.data.sensordata[env._sensor_quat_start : env._sensor_quat_start + 4].copy())
            if terminated or truncated:
                break
        threshold = 0.04 * env._body_weight_n
    finally:
        env.close()
    T = len(rows)
    fp, hd = np.asarray(sites), fwd2d(np.asarray(quats))
    dx = np.array([[float((fp[m, i, :2] - fp[m, 1 - i, :2]) @ hd[m]) for i in (0, 1)] for m in range(T + 1)])
    down = np.array([(block > 0.1).mean(axis=0) >= 0.5 for _, _, _, block in rows])
    e4 = screen_events_probe(down, dx, DT, STRIDE, 4)
    for m, (reward, terminated, info, _) in enumerate(rows):
        assert (info["r_foot_down"], info["l_foot_down"]) == (float(down[m, 0]), float(down[m, 1])), m
        assert info["walk_support"] == float(info["r_foot_contact"] + info["l_foot_contact"] > threshold), m
        assert info["walk_touchdown_unit"] == e4[m] and info["reward_walk_touchdown"] == 0.1 * e4[m], m
        total = info[ORDER[0]]
        for key in ORDER[1:]:
            total = total + info[key]
        total = total + info["reward_airborne_substeps"] + info["reward_walk_touchdown"]
        if terminated and not info.get("success"):
            total += -150.0
        assert reward == total, m
    # The patterns do what they are here for: a paid alternating touchdown, an unpaid simultaneous landing, steps
    # without support and airborne substeps.
    registered = sum(info["walk_touchdowns_registered"] for _, _, info, _ in rows)
    unsupported = sum(info["walk_support"] == 0.0 for _, _, info, _ in rows)
    if pattern[0] == "ALT":
        assert registered >= 3 and np.count_nonzero(e4) == 1
    else:
        assert registered >= 4 and np.count_nonzero(e4) == 0 and unsupported >= 15


@functools.cache
def _screen_cases() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(FIXTURES / "screen_cases.npz") as data:
        arrays = {name: data[name] for name in data.files}
    return json.loads(arrays.pop("meta_json").tobytes()), arrays


def test_r6_returns_equal_screens_offline_r6p_on_recorded_hand_built_episodes():
    """SCREEN's hand-built episodes (seed 3042: four stable walks, a stepping-in-place, two hops, the statue; and
    two re-rolls whose last pay SCREEN made for a touchdown at k = T-4) through the env's own pieces, the touchdown
    event on the probe's floor-truth flags: each R6 return equals SCREEN's published R6p at forward weights 0.2 and
    2.0 once that pay is removed."""
    meta, arrays = _screen_cases()
    assert (meta["w_td"], meta["T_cap"], meta["ell_frac"], meta["merge"]) == (0.1, 1.2, 0.1, 2)
    env = TyrannosaurusRexAnatomicalEnv(**{**_locomotion_as_3ae1c959(), **PILOT})
    try:
        tracker = env._walk_touchdowns
        assert tracker is not None
        threshold = env.walk_support_gate_bw * env._body_weight_n
        lost_at_end = 0
        for episode in meta["episodes"]:
            name, T = episode["key"], episode["T"]
            a = {key.split("/", 1)[1]: value for key, value in arrays.items() if key.startswith(name + "/")}
            assert episode["body_weight_n"] == env._body_weight_n and episode["dt"] == env.dt
            assert tracker.stride_scale_m == 0.1 * episode["leg_length"]
            down = a["down"].astype(bool)
            pays, _ = track(down, a["dx"], tracker=tracker)
            e3 = screen_events_probe(down, a["dx"], DT, tracker.stride_scale_m, 3)
            e4 = screen_events_probe(down, a["dx"], DT, tracker.stride_scale_m, 4)
            assert np.array_equal(pays, e4) and np.array_equal(e4[1:], e3[:-1]), name
            lost_at_end += int(e3[T - 1] > 0)
            support = (a["contact"] > threshold).astype(float)
            airborne = -env.airborne_substep_weight * a["airborne"].astype(float) / 5.0
            for weight in (0.2, 2.0):
                main_forward = reward_forward_velocity(
                    np.stack([a["fwd_vel"], np.zeros(T)], axis=1), np.array([1.0, 0.0]), 2.5, weight
                )[0]
                r6_forward = reward_forward_velocity(
                    np.stack([a["fwd_vel"], np.zeros(T)], axis=1), np.array([1.0, 0.0]), env.forward_vel_max, weight
                )[0]
                if weight == episode["recorded_forward_vel_weight"]:
                    # The fixture is the recording: main's terms add up to the env's own step reward.
                    assert np.allclose(main_forward + a["other"] + a["gated"] + a["fall"], a["reward"], atol=1e-12)
                r6 = r6_forward + a["other"] + a["gated"] * support + airborne + env.walk_touchdown_weight * pays
                r6 = r6 + a["fall"]
                published = episode["published_R6p"][str(weight)]
                assert float(r6.sum()) == pytest.approx(published - 0.1 * e3[T - 1], abs=1e-9), (name, weight)
        # W02 seed 3044 and V29 seed 3048: SCREEN paid a touchdown at k = T-4 on the last step; the env never does.
        assert lost_at_end == 2
    finally:
        env.close()
