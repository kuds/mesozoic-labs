"""The T. rex's opt-in per-foot CoP band (scratch prototype for the stance4 fallback; never merged, never trained).

stance4 decision line 5 names it: if the stagger penalty does not hold fore-aft from scratch, the next round tests a
per-foot centre-of-pressure band.  Off by default, and off it changes nothing: no reward, no info key, no trajectory,
no task fingerprint, no recorded stage config, and the env never loads the band's reader.  On, per foot k:

    u_k = max(excess(whole-foot CoP, [lower, upper]), excess(pad CoP, [pad_lower, pad_upper]))
    q_k = exp(-(u_k / tolerance)^2), or 0 when the foot is unloaded, carries less than min_load_share of the two
          limbs' floor load, or stands off its pad
    reward = -w * mean_k (1 - q_k), paid only after stance_width_settle_steps and only while the command is zero

The whole-foot CoP is the judge's K2b reading for the step (every floor contact of the limb over the step's
substeps, span half-lengths), computed with the recorder's reduction fused into one pass per step, in the neutral
environments/shared/floor_contact package so the env imports nothing from gait (floor_contact/step_cop.py); the tests
pin it to the recorder bit for bit, to the shared reduction functions bit for bit on synthetic contacts, to an
independent mj_contactForce computation on hand-posed states, and to a golden band-on rollout, so a later change to
the shared reduction or the fused reader cannot move the reward unseen.

Every test starts from the stance stage's env kwargs with the stagger keys removed, so "off" means off.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from environments.shared import symmetry as SY
from environments.shared.config import load_stage_config, save_stage_config
from environments.shared.floor_contact import reduction
from environments.shared.floor_contact.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.shared.gait.stance_metrics import whole_foot_cop
from environments.shared.model_id import ModelId
from environments.shared.reward_functions import band_excess, reward_stance_cop_band
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.task_fingerprint import stage_task_fingerprint
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
OPTION_KEYS = (
    "stance_cop_band_weight",
    "stance_cop_band_lower",
    "stance_cop_band_upper",
    "stance_cop_band_pad_lower",
    "stance_cop_band_pad_upper",
    "stance_cop_band_tolerance",
    "stance_cop_band_min_load_share",
)
DEFAULTS = {
    "stance_cop_band_weight": 0.0,
    "stance_cop_band_lower": -0.6,
    "stance_cop_band_upper": 0.4,
    "stance_cop_band_pad_lower": -0.65,
    "stance_cop_band_pad_upper": 0.65,
    "stance_cop_band_tolerance": 0.357,
    "stance_cop_band_min_load_share": 0.35,
}
PER_FOOT = ("foot_cop", "pad_cop", "load_share", "digit_share", "counted", "quality")
BAND_INFO_KEYS = {f"{side}_cop_band_{name}" for side in "rl" for name in PER_FOOT} | {
    "stance_cop_band_quality",
    "reward_stance_cop_band",
}
#: The proposed dose (copband design.md): the stagger term's maximum price.
ON = {"stance_cop_band_weight": 0.30}
SETTLE_STEPS = 250


def _off_kwargs() -> dict:
    """The stance stage's env kwargs with the stagger keys (and any band key) removed."""
    kwargs = dict(load_stage_config(TYRANNOSAURUS_REX, "stance")["env_kwargs"])
    for key in ("stance_stagger_weight", "stance_stagger_tolerance_m", *OPTION_KEYS):
        kwargs.pop(key, None)
    return kwargs


@pytest.fixture(scope="module")
def off_kwargs():
    return _off_kwargs()


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


def _quality(foot_cop, pad_cop, counted, kwargs=None):
    k = {**DEFAULTS, **(kwargs or {})}
    if not counted:
        return 0.0
    u = max(
        max(k["stance_cop_band_lower"] - foot_cop, 0.0) + max(foot_cop - k["stance_cop_band_upper"], 0.0),
        max(k["stance_cop_band_pad_lower"] - pad_cop, 0.0) + max(pad_cop - k["stance_cop_band_pad_upper"], 0.0),
    )
    return math.exp(-((u / k["stance_cop_band_tolerance"]) ** 2))


def _expected(info, weight=0.30, kwargs=None):
    """The term from the env's own per-foot info readings, by the formula."""
    qualities = [
        _quality(
            info[f"{s}_cop_band_foot_cop"], info[f"{s}_cop_band_pad_cop"], info[f"{s}_cop_band_counted"] > 0, kwargs
        )
        for s in "rl"
    ]
    return -weight * (2.0 - sum(qualities)) / 2.0, qualities


# ---------------------------------------------------------------------------------------------- an independent oracle
def _oracle(env, morphology):
    """Per foot, from the CURRENT data only, by a plain loop over data.contact with mj_contactForce (not the
    recorder's decode or bincount): the whole-foot CoP and its loaded flag, the pad CoP and its loaded flag, and the
    limb's floor force."""
    model, data = env.model, env.data
    out = []
    for index, foot in enumerate(morphology.feet):
        rotation = data.geom_xmat[foot.sole_geom].reshape(3, 3)
        origin = data.geom_xpos[foot.sole_geom]
        half_pad = float(model.geom_size[foot.sole_geom][foot.sole_forward_axis])
        force = moment = pad_force = pad_moment = 0.0
        pad_contact = False
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if morphology.is_floor[g1] == morphology.is_floor[g2]:
                continue
            other = g2 if morphology.is_floor[g1] else g1
            if not morphology.is_animal[other] or morphology.geom_foot[other] != index:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            normal = float(wrench[0])
            forward = foot.sole_forward_sign * float((rotation.T @ (contact.pos - origin))[foot.sole_forward_axis])
            force += normal
            moment += normal * (forward - foot.ankle_forward_m)
            if other == foot.sole_geom:
                pad_force += normal
                pad_moment += normal * forward / half_pad
                pad_contact |= normal > 0.1
        middle = 0.5 * (foot.heel_line_m + foot.toe_line_m)
        half = 0.5 * (foot.toe_line_m - foot.heel_line_m)
        foot_cop = (moment / force - middle) / half if force > 1e-9 else float("nan")
        pad_cop = pad_moment / pad_force if pad_force > 1e-9 else float("nan")
        out.append(
            {
                "foot_cop": foot_cop,
                "foot_loaded": bool(np.isfinite(foot_cop) and force > 0.1),
                "pad_cop": pad_cop,
                "pad_loaded": bool(np.isfinite(pad_cop) and pad_contact),
                "force": force,
            }
        )
    return out


def _settled(kwargs, seed=3042):
    """An env stood SETTLE_STEPS zero-action steps from a seeded reset, and that state's qpos."""
    env = TyrannosaurusRexAnatomicalEnv(**kwargs)
    env.reset(seed=seed)
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(15))
    return env, env.data.qpos.copy()


def _pose(env, base_qpos, edits, step=None):
    """Hand-pose: base_qpos plus joint offsets, zero velocity, forward kinematics; the record dropped and the step
    counter past the settle (or at ``step``) with a zero command.  Returns the reward info."""
    env.data.qpos[:] = base_qpos
    env.data.qvel[:] = 0.0
    for name, offset in edits.items():
        env.data.qpos[int(env.model.jnt_qposadr[env.model.joint(name).id])] += offset
    mujoco.mj_forward(env.model, env.data)
    env._invalidate_substep_aggregates()
    env._step_count = env.stance_width_settle_steps + 1 if step is None else step
    env._command = np.zeros_like(env._command)
    return env._get_reward_info(np.zeros(15))[1]


# ---------------------------------------------------------------------------------------------------- off by default
def test_the_option_defaults_to_off_and_is_an_opt_in_task_option():
    parameters = inspect.signature(TyrannosaurusRexAnatomicalEnv).parameters
    assert {key: parameters[key].default for key in OPTION_KEYS} == DEFAULTS
    assert ("stance_cop_band_weight", 0.0, OPTION_KEYS) in TyrannosaurusRexAnatomicalEnv.OPT_IN_TASK_KWARGS


def test_off_it_changes_no_reward_no_info_key_and_no_trajectory(off_kwargs):
    explicit_off = {**off_kwargs, "stance_cop_band_weight": 0.0, "stance_cop_band_lower": -0.2}
    rows = _pair(off_kwargs, explicit_off, 240)
    assert len(rows) == 240
    for (reward_a, info_a, qpos_a, _), (reward_b, info_b, qpos_b, _) in rows:
        assert _bytes(reward_a) == _bytes(reward_b) == _bytes(info_b["reward_total"])
        assert list(info_a) == list(info_b) and not BAND_INFO_KEYS & set(info_a)
        assert all(_bytes(info_a[k]) == _bytes(info_b[k]) for k in info_a if isinstance(info_a[k], float))
        np.testing.assert_array_equal(qpos_a, qpos_b)


def test_off_the_env_never_loads_the_band_reader(off_kwargs, monkeypatch):
    """The reader's import sits in _cop_band's body and runs only when the band is on."""
    monkeypatch.delitem(sys.modules, "environments.shared.floor_contact.step_cop", raising=False)
    env = TyrannosaurusRexAnatomicalEnv(**off_kwargs)
    try:
        env.reset(seed=3042)
        for _ in range(5):
            env.step(np.zeros(15))
        assert env._cop_band_reader is None
        assert "environments.shared.floor_contact.step_cop" not in sys.modules
    finally:
        env.close()


def test_off_it_moves_no_task_fingerprint_and_no_recorded_config(tmp_path, off_kwargs):
    base = stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs=off_kwargs)
    explicit_off = {**off_kwargs, "stance_cop_band_weight": 0, "stance_cop_band_upper": 0.9}
    assert stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs=explicit_off) == base
    assert not set(OPTION_KEYS) & set(base["env"])
    # Every stage that leaves it off keeps its committed fingerprint with the keys written out at off values.
    for entry in load_stage_manifest(TYRANNOSAURUS_REX).stages:
        env_kwargs = load_stage_config(TYRANNOSAURUS_REX, entry.reference)["env_kwargs"]
        if env_kwargs.get("stance_cop_band_weight", 0.0) == 0.0:
            written = {**env_kwargs, "stance_cop_band_weight": 0.0, "stance_cop_band_tolerance": 0.5}
            committed = stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference)
            assert stage_task_fingerprint(TYRANNOSAURUS_REX, entry.reference, env_kwargs=written) == committed, entry.id
    # On, the switch and every band setting are hashed, defaults included, and nothing else moves.
    on = stage_task_fingerprint(TYRANNOSAURUS_REX, "stance", env_kwargs={**off_kwargs, **ON})
    assert {k: v for k, v in on["env"].items() if k not in base["env"]} == {**DEFAULTS, **ON}
    assert {k: v for k, v in on["env"].items() if k in base["env"]} == base["env"]
    tighter = stage_task_fingerprint(
        TYRANNOSAURUS_REX, "stance", env_kwargs={**off_kwargs, **ON, "stance_cop_band_pad_upper": 0.6}
    )
    assert len({base["task_sha256"], on["task_sha256"], tighter["task_sha256"]}) == 3
    # stage_config.json records no off option's defaults; on, all seven keys.
    config = {**load_stage_config(TYRANNOSAURUS_REX, "stance"), "env_kwargs": off_kwargs}
    path = save_stage_config(
        tmp_path / "off", "stance", config, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    assert not set(OPTION_KEYS) & set(json.loads(path.read_text())["reward_weights"])
    config_on = {**config, "env_kwargs": {**off_kwargs, **ON}}
    path = save_stage_config(
        tmp_path / "on", "stance", config_on, "PPO", env_class=TyrannosaurusRexAnatomicalEnv, model_id=TYRANNOSAURUS_REX
    )
    recorded = json.loads(path.read_text())["reward_weights"]
    assert {key: recorded[key] for key in OPTION_KEYS} == {**DEFAULTS, **ON}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stance_cop_band_weight": -0.1}, "stance_cop_band_weight"),
        ({"stance_cop_band_weight": math.inf}, "stance_cop_band_weight"),
        ({"stance_cop_band_weight": math.nan}, "stance_cop_band_weight"),
        ({"stance_cop_band_lower": 0.5}, "stance_cop_band_lower"),
        ({"stance_cop_band_upper": -math.inf}, "stance_cop_band_lower"),
        ({"stance_cop_band_pad_lower": 0.7}, "stance_cop_band_pad_lower"),
        ({"stance_cop_band_tolerance": 0.0}, "stance_cop_band_tolerance"),
        ({"stance_cop_band_tolerance": math.inf}, "stance_cop_band_tolerance"),
        ({"stance_cop_band_min_load_share": 0.5}, "stance_cop_band_min_load_share"),
        ({"stance_cop_band_min_load_share": -0.1}, "stance_cop_band_min_load_share"),
    ],
)
def test_invalid_values_fail_fast(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TyrannosaurusRexAnatomicalEnv(**kwargs)


# ---------------------------------------------------------------------------------------------------- on: the reading
def test_on_the_reading_is_the_recorders_bit_for_bit(off_kwargs):
    """Every step of a 300-step noisy episode: each foot's whole-foot CoP is the recorder's trace (whole_foot_cop,
    the K2b reading) and the pad CoP its sole_cop, byte for byte, with the loaded flags the gate uses."""
    env = TyrannosaurusRexAnatomicalEnv(**off_kwargs, **ON)
    recorder = SubstepContactRecorder(env, Morphology.from_env(env, TYRANNOSAURUS_REX)).attach()
    try:
        env.reset(seed=3052)
        rng = np.random.default_rng(1)
        rows = []
        for _ in range(300):
            action = rng.uniform(-0.1, 0.1, 15)
            _, reward, terminated, truncated, info = env.step(action)
            recorder.end_step(action, reward)
            rows.append(info)
            if terminated or truncated:
                break
        trace = recorder.end_episode(terminated=terminated, truncated=truncated)
    finally:
        recorder.detach()
        env.close()
    assert len(rows) == 300
    _spanned, cop, loaded = whole_foot_cop(trace)
    pad = trace.sole_cop[:, :, 0]
    pad_loaded = np.isfinite(pad) & (trace.sole_contacts_mean > 0)
    seen_out = 0
    for step, info in enumerate(rows):
        for index, side in enumerate("rl"):
            assert _bytes(info[f"{side}_cop_band_foot_cop"]) == _bytes(cop[step, index] if loaded[step, index] else 0)
            assert _bytes(info[f"{side}_cop_band_pad_cop"]) == _bytes(
                pad[step, index] if pad_loaded[step, index] else 0
            )
            share = trace.leg_floor_mean[step, index] / trace.leg_floor_mean[step].sum()
            assert info[f"{side}_cop_band_load_share"] == pytest.approx(share, rel=1e-12)
            seen_out += not -0.6 <= cop[step, index] <= 0.4
    assert seen_out > 20  # the noisy episode leaves the band, so the comparison covers paid steps


def test_on_it_matches_the_formula_and_an_independent_oracle_on_constructed_states(off_kwargs):
    """Hand-posed states from a settled stance: as stood (inside the band), both ankles opened 0.06 rad (the right
    foot's whole-foot CoP past the toe edge), and the right toes flexed 0.2 rad (the right foot's load onto its
    digits: whole-foot CoP far past the toe edge, pad CoP on its heel edge).  The readings match a plain
    mj_contactForce loop and the reward matches the formula."""
    env, base = _settled({**off_kwargs, **ON})
    try:
        morphology = Morphology.from_env(env, TYRANNOSAURUS_REX)
        toes = {f"r_toe_d{d}_joint": 0.2 for d in (2, 3, 4)}
        rewards = []
        for edits in ({}, {"r_ankle": 0.06, "l_ankle": 0.06}, toes):
            info = _pose(env, base, edits)
            oracle = _oracle(env, morphology)
            for index, side in enumerate("rl"):
                assert info[f"{side}_cop_band_foot_cop"] == pytest.approx(oracle[index]["foot_cop"], abs=1e-9)
                assert info[f"{side}_cop_band_pad_cop"] == pytest.approx(oracle[index]["pad_cop"], abs=1e-9)
                share = oracle[index]["force"] / (oracle[0]["force"] + oracle[1]["force"])
                assert info[f"{side}_cop_band_load_share"] == pytest.approx(share, abs=1e-9)
                assert info[f"{side}_cop_band_counted"] == 1.0
            want, qualities = _expected(info)
            assert info["reward_stance_cop_band"] == pytest.approx(want, rel=1e-12, abs=1e-15)
            assert [info["r_cop_band_quality"], info["l_cop_band_quality"]] == pytest.approx(qualities, rel=1e-12)
            rewards.append(info["reward_stance_cop_band"])
        assert rewards[0] == 0.0  # the settled stance stands inside both bands
        assert -0.3 < rewards[1] < -0.01  # the right foot's CoP just past the toe edge
        assert rewards[2] < rewards[1]  # the digit-loaded right foot: far past it
    finally:
        env.close()


def test_the_gates_pay_nothing_at_or_before_the_settle_and_under_a_command(off_kwargs):
    env, base = _settled({**off_kwargs, **ON})
    try:
        edits = {"r_ankle": 0.06, "l_ankle": 0.06}
        paid = _pose(env, base, edits)
        assert paid["reward_stance_cop_band"] < 0.0
        for step in (0, 1, env.stance_width_settle_steps):
            info = _pose(env, base, edits, step=step)
            assert info["reward_stance_cop_band"] == 0.0
            assert info["stance_cop_band_quality"] == paid["stance_cop_band_quality"] < 1.0  # still reported
        _pose(env, base, edits)
        env._command = np.array([0.0, 0.1, 0.0], dtype=env._command.dtype)
        _, info = env._get_reward_info(np.zeros(15))
        assert info["reward_stance_cop_band"] == 0.0 and info["stance_cop_band_quality"] < 1.0
    finally:
        env.close()


def test_on_it_changes_no_dynamics_and_adds_exactly_its_term(off_kwargs):
    rows = _pair(off_kwargs, {**off_kwargs, **ON}, 320)
    assert len(rows) == 320
    settle = off_kwargs.get("stance_width_settle_steps", 200)
    paid = 0
    for step, ((r_off, i_off, q_off, _), (r_on, i_on, q_on, _)) in enumerate(rows, start=1):
        np.testing.assert_array_equal(q_off, q_on)  # the reader only copies: the trajectory is untouched
        assert set(i_on) - set(i_off) == BAND_INFO_KEYS
        assert _bytes(r_on) == _bytes(r_off + i_on["reward_stance_cop_band"]) == _bytes(i_on["reward_total"])
        if step <= settle:
            assert i_on["reward_stance_cop_band"] == 0.0 and _bytes(r_on) == _bytes(r_off)
        else:
            assert i_on["reward_stance_cop_band"] == pytest.approx(_expected(i_on)[0], rel=1e-12, abs=1e-15)
            paid += i_on["reward_stance_cop_band"] < 0.0
    assert paid > 0


# ---------------------------------------------------------------------------------------------------- on: the golden
#: The reward runs the shared floor-contact reduction (floor_contact/), which no task fingerprint hashes, so a change
#: to it would move the reward with no fingerprint move (copband design section 7, risk 5).  This pins it: a 60-step
#: band-on rollout from a seeded reset under uniform(-0.2, 0.2) actions, with the gate open from step 11
#: (stance_width_settle_steps 10), so the spawn transient's lifted, light and off-pad feet are read and 50 steps are
#: paid.  Per band info key: the sha256 of its per-step values quantized to 6 decimals (the digest snapshot's portable
#: rounding; negative zero folded to zero).  Stepping crosses the solver, so it compares only under this MuJoCo.
#: Re-pin from the failure message's digests only with a change that is meant to move the band's readings, and say
#: which keys moved.
GOLDEN_MUJOCO = "3.10.0"
GOLDEN_STEPS = 60
GOLDEN = {
    "l_cop_band_counted": "2ef683d346aa2ef2c78c774881d1ed74fb4a35b671b08de3e168b895b90e3bb9",
    "l_cop_band_digit_share": "cd6acb09122511149cbf1aaf088cc89f6f03336a967817ce506d075f0b993a1c",
    "l_cop_band_foot_cop": "9c26b615d48fc35e009cddf01e030d3d1467db88dc405834cbd0b37cb42ee921",
    "l_cop_band_load_share": "8beaacd30795cdc5bad6bdc39d9bb493e74c8c55547f012fa2e3058b17f9805e",
    "l_cop_band_pad_cop": "3727b9c534e1bccb3f1bba1684a11393bf7a41877599e10ee7d194dc7cccf1bf",
    "l_cop_band_quality": "01187557e7c64a5ec5961e9418d61020d8852c9815390ee7a460d5960b6f8475",
    "r_cop_band_counted": "3e48d7e29fa41f25ca9e45597f9827d4ade3f242e687605091bc582de0f873a9",
    "r_cop_band_digit_share": "f8ea879b7902755d4d6fd4f99d63c94fb4b14b045fb20a8d9409ac9c7dd7053c",
    "r_cop_band_foot_cop": "0084fee3b6b1ad7853fbe0e7329074446d4b2d982801d69bf001534fafabbc6d",
    "r_cop_band_load_share": "1864e45c647c70b85583434ba32e5ff2faaee8257a56eb4d3dc31639fb083ae9",
    "r_cop_band_pad_cop": "e826fe34cd686ab8e1b6073be86b69fd0278e8aec3b07f6bfe433f7faa253174",
    "r_cop_band_quality": "6e61280822b1afd15d3a34d70c6245253938c2d4484eafef1f914634e25e4b3f",
    "reward_stance_cop_band": "16b413362fb27562100926a39850a7f10336a0274de2e9a748f9c88a2022a3e1",
    "stance_cop_band_quality": "346031ca4b59f5343a33bf6c3b9f01a90677c047e9de9d7eb71ad3f283df949d",
}


def _golden_rows(kwargs) -> list[dict]:
    """The golden rollout's band info, step by step (``kwargs``: the band-off stance kwargs)."""
    env = TyrannosaurusRexAnatomicalEnv(**{**kwargs, **ON, "stance_width_settle_steps": 10})
    try:
        env.reset(seed=3052)
        rng = np.random.default_rng(3052)
        rows = []
        for _ in range(GOLDEN_STEPS):
            _, _, terminated, truncated, info = env.step(rng.uniform(-0.2, 0.2, 15))
            rows.append({key: info[key] for key in sorted(BAND_INFO_KEYS)})
            if terminated or truncated:
                break
        return rows
    finally:
        env.close()


def _golden_digests(rows) -> dict[str, str]:
    digests = {}
    for key in sorted(BAND_INFO_KEYS):
        values = np.round(np.array([row[key] for row in rows], dtype=np.float64), 6) + 0.0
        digests[key] = hashlib.sha256(values.tobytes()).hexdigest()
    return digests


def test_the_shared_reduction_is_pinned_by_a_golden_band_on_rollout(off_kwargs):
    if mujoco.__version__ != GOLDEN_MUJOCO:
        pytest.skip(f"the golden steps the solver: running mujoco {mujoco.__version__} != {GOLDEN_MUJOCO}")
    rows = _golden_rows(off_kwargs)
    assert len(rows) == GOLDEN_STEPS
    paid = [row["reward_stance_cop_band"] for row in rows]
    assert sum(value < 0.0 for value in paid) == 50  # the gate opens after step 10, and every open step pays
    counted = np.array([[row["r_cop_band_counted"], row["l_cop_band_counted"]] for row in rows])
    quality = np.array([[row["r_cop_band_quality"], row["l_cop_band_quality"]] for row in rows])
    assert (counted == 0.0).any() and ((quality > 0.0) & (quality < 1.0)).any()  # guards and the slope both read
    digests = _golden_digests(rows)
    moved = sorted(key for key in GOLDEN.keys() | digests.keys() if GOLDEN.get(key) != digests.get(key))
    assert moved == [], f"the band's readings moved ({moved}): {digests}"


#: The rollout reads no contact near the reduction's thresholds, so it would not see an edit of one (measured: the
#: contact threshold 0.1 -> 0.11 N and the 1e-9 N load rule -> 1e-3 N both leave it unmoved).  This second pin feeds
#: the pure reduction synthetic contacts that straddle each one: no physics step, so no solver dependence.  Per output:
#: the sha256 of its values quantized to 12 decimals, NaN canonicalized.
REDUCTION_GOLDEN = {
    "classify": "9e628d6fd06a3b18eeef39fb43c05f5af10198b47516924d8e529821a9d31a6b",
    "leg_force": "4772050c8eb3258d94f5b3c9001c54a353b524374da604a57ce64f4f21646ee8",
    "support_force": "dd9b9be2153360b564722b91fef8414ab22d120242a300eda04800622940cae8",
    "sole_cop": "610cf0a562370dc9d40a8651d378314b75740fb7a13e9f4ca267042491e4ffcc",
    "foot_cop": "cf0c78b9a3cdc4bffb8853482de99ccf0a11b221a67602bb38132e99b1a50e31",
    "normalized": "51fa6142b8f13b3d451c250a39757b945c4d0897b4d32982787819b333d1840c",
    "decoded": "88cef0ab4c9954e179ca214f5fa2c2a6fda8d04df7abc01db2b239dbf78d40cc",
}


def _synthetic_reduction() -> dict[str, np.ndarray]:
    """Two steps of two substeps on two feet (floor 0; right sole 1 and digit 2; left sole 3 and digit 4; a body
    geom 5; a prop 6).  Step 0: a right sole contact at 0.105 N (CONTACT_THRESHOLD_N 0.1), the left limb's largest
    substep force 0.104 N, a digit contact with its geoms in floor-second order, a body and a prop on the floor and a
    sole on the other sole.  Step 1: the right foot carries nothing, the left only 5e-4 N on its digit (the 1e-9 N
    load rule).  The left sole is turned 90 degrees about z.  Plus the normal-force decode, pyramidal and elliptic."""
    morphology = SimpleNamespace(
        is_floor=np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool),
        is_animal=np.array([0, 1, 1, 1, 1, 1, 0], dtype=bool),
        geom_foot=np.array([-1, 0, 0, 1, 1, -1, -1]),
        geom_support=np.array([-1, 0, 1, 2, 3, -1, -1]),
    )
    feet = [
        SimpleNamespace(
            side=side, sole_forward_axis=0, sole_forward_sign=1.0, sole_lateral_axis=1, sole_lateral_sign=1.0
        )
        for side in "RL"
    ]
    rows = [  # (substep of the episode, geom1, geom2, normal force, world point)
        (0, 0, 1, 0.105, (0.03, 0.01, 0.0)),
        (0, 2, 0, 0.3, (0.09, -0.005, 0.0)),
        (0, 0, 3, 0.104, (0.02, 0.33, 0.0)),
        (0, 0, 5, 1.0, (0.5, 0.0, 0.4)),
        (0, 0, 6, 2.0, (1.0, 1.0, 0.0)),
        (0, 1, 3, 0.5, (0.0, 0.15, 0.0)),
        (1, 0, 1, 0.2, (-0.01, 0.02, 0.0)),
        (1, 4, 0, 0.05, (-0.03, 0.4, 0.0)),
        (2, 0, 4, 3e-4, (-0.02, 0.41, 0.0)),
        (3, 0, 4, 2e-4, (-0.01, 0.41, 0.0)),
    ]
    substep_of = np.array([row[0] for row in rows])
    geoms = np.array([row[1:3] for row in rows])
    normal = np.array([row[3] for row in rows])
    points = np.array([row[4] for row in rows])
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose = np.tile(np.concatenate([[0.0, 0.0, 0.0], np.eye(3).ravel(), [0.0, 0.3, 0.0], turned.ravel()]), (4, 1))
    pose = pose.reshape(4, 2, 12)
    other, on_floor, foot, leg_on_floor = reduction.classify_floor_contacts(morphology, geoms)
    leg = reduction.per_substep_force(substep_of, normal, leg_on_floor, foot, 2, 2, 2)
    support_index = np.where(leg_on_floor, morphology.geom_support[other], -1)
    support = reduction.per_substep_force(substep_of, normal, support_index >= 0, support_index, 4, 2, 2)
    on_sole = leg_on_floor & (other == np.array([1, 3, -2])[foot])
    half_size = [np.array([0.07, 0.04, 0.01])] * 2
    sole_cop = reduction.sole_cop_steps(feet, half_size, pose, substep_of, foot, on_sole, points, normal, 2, 2)
    ankle_forward = np.array([0.02, -0.01])
    foot_cop = reduction.whole_foot_cop_steps(
        feet, ankle_forward, pose, substep_of, foot, leg_on_floor, points, normal, 2, 2
    )
    span = np.array([[-0.05, 0.09], [-0.06, 0.08]])
    spanned, cop, loaded = reduction.normalized_foot_cop(foot_cop, leg.max(axis=1), span)
    contact = SimpleNamespace(efc_address=np.array([0, 4, -1, 8]), dim=np.array([3, 3, 3, 1]))
    efc_force = np.array([0.5, 0.25, 0.125, 0.0625, 1.0, 2.0, 0.5, 0.25, 3.0])
    decoded = []
    for cone, condims in ((0, [3, 3]), (0, [3, 1]), (1, [3, 3])):
        model = SimpleNamespace(opt=SimpleNamespace(cone=cone), geom_condim=np.array(condims), npair=0)
        elliptic, edge_offsets = reduction.decode_settings(model)
        decoded.append(reduction.decode_normal_forces(contact, efc_force, elliptic=elliptic, edge_offsets=edge_offsets))
    return {
        "classify": np.stack([other, on_floor, foot, leg_on_floor]),
        "leg_force": leg,
        "support_force": support,
        "sole_cop": sole_cop,
        "foot_cop": foot_cop,
        "normalized": np.stack([np.broadcast_to(spanned, cop.shape), cop, loaded]),
        "decoded": np.stack(decoded),
    }


def _reduction_digest(values) -> str:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isnan(values), np.nan, np.round(values, 12) + 0.0)
    return hashlib.sha256(values.tobytes()).hexdigest()


def test_the_shared_reduction_is_pinned_on_contacts_that_straddle_its_thresholds():
    outputs = _synthetic_reduction()
    loaded = outputs["normalized"][2]
    assert loaded.tolist() == [[1.0, 1.0], [0.0, 0.0]]  # the 0.104 N limb is loaded at 0.1 N, the 5e-4 N one is not
    assert np.isnan(outputs["foot_cop"][1, 0]) and np.isfinite(outputs["foot_cop"][1, 1])  # 5e-4 N passes 1e-9 N
    digests = {name: _reduction_digest(values) for name, values in outputs.items()}
    moved = sorted(
        name for name in REDUCTION_GOLDEN.keys() | digests.keys() if REDUCTION_GOLDEN.get(name) != digests.get(name)
    )
    assert moved == [], f"the shared reduction's outputs moved ({moved}): {digests}"


#: The band's reader runs the shared reduction's rules and arithmetic in one fused pass per step (floor_contact/
#: step_cop.py), so it is pinned to the shared functions themselves, byte for byte, on synthetic steps of two substeps
#: (the geoms of _synthetic_reduction).  Step 1: a right sole contact at 0.105 N (CONTACT_THRESHOLD_N 0.1), a right
#: digit contact in floor-second order, one without constraint rows (efc_address -1), a left sole contact at 0.0999 N
#: and the left limb's largest substep force 0.104 N (its digit), a body and a prop on the floor, a sole on the other
#: sole.  Step 2: the right foot carries 4e-10 N (the 1e-9 N load rule: no CoP), the left 5e-4 N on its digit (a CoP,
#: not loaded).  Step 3: an empty substep, then the right sole alone at 0.1 N.  Each sole moves between substeps; the
#: left one is turned 90 degrees about z.
FUSED_STEPS = (
    (  # (geom1, geom2, normal force, world point), per substep
        [
            (0, 1, 0.105, (0.03, 0.01, 0.0)),
            (2, 0, 0.3, (0.09, -0.005, 0.0)),
            (0, 2, None, (0.08, 0.0, 0.0)),
            (0, 3, 0.0999, (0.02, 0.33, 0.0)),
            (0, 5, 1.0, (0.5, 0.0, 0.4)),
            (0, 6, 2.0, (1.0, 1.0, 0.0)),
            (1, 3, 0.5, (0.0, 0.15, 0.0)),
        ],
        [(0, 1, 0.2, (-0.01, 0.02, 0.0)), (4, 0, 0.104, (-0.03, 0.4, 0.0))],
    ),
    ([(0, 4, 3e-4, (-0.02, 0.41, 0.0)), (0, 1, 4e-10, (0.01, 0.0, 0.0))], [(0, 4, 2e-4, (-0.01, 0.41, 0.0))]),
    ([], [(0, 1, 0.1, (0.01, 0.005, 0.0))]),
)


def _fused_scene(cone, condims):
    """The synthetic morphology, a model with ``cone`` and per-geom ``condims``, and per step and substep the data the
    reader copies (a contact's constraint rows decode to its normal force; None: no rows)."""
    morphology = SimpleNamespace(
        is_floor=np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool),
        is_animal=np.array([0, 1, 1, 1, 1, 1, 0], dtype=bool),
        geom_foot=np.array([-1, 0, 0, 1, 1, -1, -1]),
        geom_support=np.array([-1, 0, 1, 2, 3, -1, -1]),
        support_geom_ids=(1, 2, 3, 4),
        support_geom_foot=(0, 0, 1, 1),
        n_feet=2,
        feet=[
            SimpleNamespace(
                side=side,
                sole_geom=sole,
                sole_forward_axis=0,
                sole_forward_sign=1.0,
                sole_lateral_axis=1,
                sole_lateral_sign=1.0,
                ankle_forward_m=ankle,
                heel_line_m=heel,
                toe_line_m=toe,
            )
            for side, sole, ankle, heel, toe in (("R", 1, 0.02, -0.05, 0.09), ("L", 3, -0.01, -0.06, 0.08))
        ],
    )
    model = SimpleNamespace(
        opt=SimpleNamespace(cone=cone),
        geom_condim=np.array(condims),
        npair=0,
        geom_size=np.tile([0.07, 0.04, 0.01], (7, 1)),
    )
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    steps = []
    for rows in FUSED_STEPS:
        substeps = []
        for substep, contacts in enumerate(rows):
            geom_xpos = np.zeros((7, 3))
            geom_xpos[1] = (0.002 * substep, 0.0, 0.001)
            geom_xpos[3] = (-0.001 * substep, 0.3, 0.0)
            geom_xmat = np.tile(np.eye(3).ravel(), (7, 1))
            geom_xmat[3] = turned.ravel()
            address, dim, efc_force = [], [], []
            for geom1, geom2, force, _point in contacts:
                condim = int(max(condims[geom1], condims[geom2]))
                dim.append(condim)
                if force is None:
                    address.append(-1)
                    continue
                address.append(len(efc_force))
                edges = 1 if (cone == 1 or condim == 1) else 2 * (condim - 1)
                efc_force += [force] + [0.1 * force] * (condim - 1) if edges == 1 else [force / edges] * edges
            contact = SimpleNamespace(
                geom=np.array([c[:2] for c in contacts], dtype=np.int32).reshape(-1, 2),
                pos=np.array([c[3] for c in contacts], dtype=np.float64).reshape(-1, 3),
                efc_address=np.array(address, dtype=np.int32),
                dim=np.array(dim, dtype=np.int32),
            )
            substeps.append(
                SimpleNamespace(
                    ncon=len(contacts),
                    contact=contact,
                    efc_force=np.array(efc_force),
                    geom_xpos=geom_xpos,
                    geom_xmat=geom_xmat,
                )
            )
        steps.append(substeps)
    return morphology, model, steps


def _shared_reading(morphology, model, substeps) -> dict[str, np.ndarray]:
    """One step read by the shared reduction functions, as the recorder runs them (``SubstepContactRecorder._reduce``
    and pad_telemetry's loaded rule), for the step's ``substeps``."""
    feet, n_feet = morphology.feet, morphology.n_feet
    elliptic, edge_offsets = reduction.decode_settings(model)
    counts = [data.ncon for data in substeps]
    present = [data for data in substeps if data.ncon]
    normal = np.concatenate(
        [np.zeros(0)]
        + [
            reduction.decode_normal_forces(d.contact, d.efc_force, elliptic=elliptic, edge_offsets=edge_offsets)
            for d in present
        ]
    )
    geoms = np.concatenate([np.zeros((0, 2), dtype=np.int64)] + [d.contact.geom.astype(np.int64) for d in present])
    points = np.concatenate([np.zeros((0, 3))] + [d.contact.pos for d in present])
    substep_of = np.repeat(np.arange(len(substeps)), counts)
    other, _on_floor, foot, leg_on_floor = reduction.classify_floor_contacts(morphology, geoms)
    leg = reduction.per_substep_force(substep_of, normal, leg_on_floor, foot, n_feet, 1, len(substeps))
    support_index = np.where(leg_on_floor, morphology.geom_support[other], -1)
    support = reduction.per_substep_force(substep_of, normal, support_index >= 0, support_index, 4, 1, len(substeps))
    on_sole = leg_on_floor & (other == np.array([1, 3, -2])[foot])
    loaded = on_sole & (normal > reduction.CONTACT_THRESHOLD_N)
    sole_contacts = np.bincount(foot[loaded], minlength=n_feet)
    soles = [1, 3]
    pose = np.stack([np.concatenate([d.geom_xpos[soles], d.geom_xmat[soles]], axis=1) for d in substeps])
    half_size = [model.geom_size[sole] for sole in soles]
    sole_cop = reduction.sole_cop_steps(
        feet, half_size, pose, substep_of, foot, on_sole, points, normal, 1, len(substeps)
    )
    ankle_forward = np.array([spec.ankle_forward_m for spec in feet])
    foot_cop_m = reduction.whole_foot_cop_steps(
        feet, ankle_forward, pose, substep_of, foot, leg_on_floor, points, normal, 1, len(substeps)
    )
    span = np.array([[spec.heel_line_m, spec.toe_line_m] for spec in feet])
    _spanned, foot_cop, foot_loaded = reduction.normalized_foot_cop(foot_cop_m, leg.max(axis=1), span)
    limb = leg.mean(axis=1)[0]
    digits = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) @ support.mean(axis=1)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        digit_share = np.where(limb > 0, digits / np.where(limb > 0, limb, 1.0), np.nan)
    return {
        "foot_cop": foot_cop[0],
        "foot_loaded": foot_loaded[0],
        "pad_cop": sole_cop[0, :, 0],
        "pad_loaded": np.isfinite(sole_cop[0, :, 0]) & (sole_contacts > 0),
        "leg_force_mean": limb,
        "leg_force_max": leg.max(axis=1)[0],
        "digit_share": digit_share,
    }


@pytest.mark.parametrize(
    "cone, condims",
    [(1, [3] * 7), (0, [3] * 7), (0, [3, 3, 3, 3, 3, 1, 3])],
    ids=["elliptic", "pyramidal", "per-contact decode"],
)
def test_the_fused_reader_is_the_shared_reduction_on_contacts_that_straddle_its_thresholds(cone, condims):
    from environments.shared.floor_contact.step_cop import StepFootCop

    morphology, model, steps = _fused_scene(cone, condims)
    reader = StepFootCop(SimpleNamespace(model=model, frame_skip=2), morphology)
    readings = []
    for step, substeps in enumerate(steps, start=1):
        for substep, data in enumerate(substeps):
            reader.record(model, data, step, substep)
        readings.append((reader.reading(step, model, substeps[-1]), _shared_reading(morphology, model, substeps)))
    # A state read outside a recorded step: its current data as one substep.
    readings.append((reader.reading(99, model, steps[0][0]), _shared_reading(morphology, model, steps[0][:1])))
    for fused, shared in readings:
        for name, want in shared.items():
            got = getattr(fused, name)
            assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), (name, got, want)
    (first, _), (second, _), (third, _), _fallback = readings
    assert first.pad_loaded.tolist() == [True, False]  # a 0.105 N sole contact passes 0.1 N, a 0.0999 N one does not
    assert first.foot_loaded.tolist() == [True, True]  # the 0.104 N limb is loaded at 0.1 N
    assert np.isnan(second.foot_cop[0]) and np.isnan(second.pad_cop[0])  # 4e-10 N: under the 1e-9 N load rule
    assert np.isfinite(second.foot_cop[1]) and not second.foot_loaded[1]  # 5e-4 N: a CoP, not loaded
    assert np.isfinite(third.foot_cop[0]) and not third.foot_loaded[0] and not third.pad_loaded[0]  # 0.1 N: not above


# ---------------------------------------------------------------------------------------------------- on: the guards
def test_guard_1_a_lifted_foot_pays_the_maximum(off_kwargs):
    """The right hip swung 0.3 rad forward lifts the right foot off the floor: it is not counted and pays the full
    per-foot price (0.30 / 2), whatever its readings; the left foot carries everything and stands in the band."""
    env, base = _settled({**off_kwargs, **ON})
    try:
        info = _pose(env, base, {"r_hip_pitch": -0.3})
        oracle = _oracle(env, Morphology.from_env(env, TYRANNOSAURUS_REX))
        assert oracle[0]["force"] == 0.0 and not oracle[0]["foot_loaded"]
        assert info["r_cop_band_counted"] == 0.0 and info["r_cop_band_quality"] == 0.0
        assert info["r_cop_band_foot_cop"] == 0.0 and info["r_cop_band_load_share"] == 0.0
        assert info["l_cop_band_counted"] == 1.0 and info["l_cop_band_quality"] == 1.0
        assert info["reward_stance_cop_band"] == pytest.approx(-0.15, abs=1e-15)
    finally:
        env.close()


def test_guard_1_a_lightly_loaded_foot_pays_the_maximum(off_kwargs):
    """Opening the right ankle 0.06 rad leaves the right foot about 43% of the limbs' load.  At the default minimum
    share (0.35) it is counted; with a 0.45 minimum it is not, and pays the full per-foot price."""
    for min_share, counted in ((0.35, True), (0.45, False)):
        env, base = _settled({**off_kwargs, **ON, "stance_cop_band_min_load_share": min_share})
        try:
            info = _pose(env, base, {"r_ankle": 0.06})
            assert 0.35 < info["r_cop_band_load_share"] < 0.45 < info["l_cop_band_load_share"]
            assert info["r_cop_band_counted"] == float(counted)
            if not counted:
                assert info["r_cop_band_quality"] == 0.0
                left = info["l_cop_band_quality"]
                assert info["reward_stance_cop_band"] == pytest.approx(-0.30 * (2.0 - left) / 2.0, rel=1e-12)
        finally:
            env.close()


def test_guard_2_a_foot_off_its_pad_pays_the_maximum(off_kwargs):
    """The right toes flexed 0.4 rad lift the right pad: the limb is loaded, on its digits alone.  The pad is
    unloaded, so the foot is not counted, whatever its whole-foot CoP; and the left foot, now under 35% of the
    load, is not counted either."""
    env, base = _settled({**off_kwargs, **ON})
    try:
        info = _pose(env, base, {f"r_toe_d{d}_joint": 0.4 for d in (2, 3, 4)})
        oracle = _oracle(env, Morphology.from_env(env, TYRANNOSAURUS_REX))
        assert oracle[0]["foot_loaded"] and not oracle[0]["pad_loaded"]
        assert info["r_cop_band_digit_share"] == pytest.approx(1.0)
        assert info["r_cop_band_counted"] == 0.0 and info["r_cop_band_pad_cop"] == 0.0
        assert info["l_cop_band_load_share"] < 0.35 and info["l_cop_band_counted"] == 0.0
        assert info["reward_stance_cop_band"] == pytest.approx(-0.30, abs=1e-15)
    finally:
        env.close()


def test_guard_2_the_pad_band_prices_load_moved_onto_the_digits(off_kwargs):
    """The right toes extended 0.6 rad: about 40% of the right foot's load moves onto its digits near their MTP
    joints, which pulls the whole-foot CoP back inside the band (+0.03) while the pad itself sits toe-loaded
    (+0.63).  The whole-foot band alone pays nothing; the pad band prices the pad's own excursion (here with its
    upper edge at 0.5 to show it binding; the default 0.65 sits just past this pose)."""
    toes = {f"r_toe_d{d}_joint": -0.6 for d in (2, 3, 4)}
    env, base = _settled({**off_kwargs, **ON})
    try:
        info = _pose(env, base, toes)
        assert -0.6 < info["r_cop_band_foot_cop"] < 0.4 < info["r_cop_band_pad_cop"] < 0.65
        assert info["r_cop_band_digit_share"] > 0.3
        assert info["reward_stance_cop_band"] == 0.0
    finally:
        env.close()
    tight = {"stance_cop_band_pad_upper": 0.5}
    env, base = _settled({**off_kwargs, **ON, **tight})
    try:
        info = _pose(env, base, toes)
        pad = info["r_cop_band_pad_cop"]
        want = 0.30 * (1.0 - math.exp(-(((pad - 0.5) / 0.357) ** 2))) / 2.0
        assert info["r_cop_band_quality"] < 1.0
        assert info["reward_stance_cop_band"] == pytest.approx(-want, rel=1e-12)
        assert info["reward_stance_cop_band"] == pytest.approx(_expected(info, kwargs=tight)[0], rel=1e-12)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- on: the mirror
def test_the_readings_swap_feet_under_the_mirror_and_the_term_does_not_move(off_kwargs):
    env = TyrannosaurusRexAnatomicalEnv(**off_kwargs, **ON)
    try:
        mirror = SY.build_env_mirror(env, SY.HEADING_FREE_OBSERVATION_SCHEMA)
        env.reset(seed=3052)
        for action in np.random.default_rng(2).uniform(-0.2, 0.2, (40, 15)):
            env.step(action)
        mujoco.mj_forward(env.model, env.data)
        env._invalidate_substep_aggregates()
        env._step_count = env.stance_width_settle_steps + 1  # the gate open, so the paid reward is compared
        _, before = env._get_reward_info(np.zeros(15))
        SY.mirror_env_state(env, env, mirror)
        env._invalidate_substep_aggregates()
        _, after = env._get_reward_info(np.zeros(15))
        assert before["reward_stance_cop_band"] < 0.0
        for name in PER_FOOT:
            assert after[f"r_cop_band_{name}"] == pytest.approx(before[f"l_cop_band_{name}"], abs=1e-9)
            assert after[f"l_cop_band_{name}"] == pytest.approx(before[f"r_cop_band_{name}"], abs=1e-9)
        for key in ("stance_cop_band_quality", "reward_stance_cop_band"):
            assert after[key] == pytest.approx(before[key], abs=1e-9)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------------- pure function
@pytest.mark.parametrize(
    ("foot_cop", "pad_cop", "counted"),
    [
        ((0.0, -0.1), (0.2, 0.1), (True, True)),  # inside both bands
        ((-0.9, 0.1), (-0.95, 0.3), (True, True)),  # a heel-side split foot (K2a-like)
        ((0.6, 0.0), (0.62, 0.2), (True, True)),  # past the toe edge
        ((-0.5, 0.0), (-0.95, 0.2), (True, True)),  # whole foot inside, pad on its heel: the pad band binds
        ((math.nan, 0.0), (math.nan, 0.2), (False, True)),  # an unloaded foot: NaN readings, the maximum
        ((-0.3, 0.0), (0.0, 0.2), (False, False)),  # neither counted: -w
    ],
)
def test_the_pure_function_is_the_design_formula(foot_cop, pad_cop, counted):
    reward, quality = reward_stance_cop_band(
        np.asarray(foot_cop), np.asarray(pad_cop), np.asarray(counted), (-0.6, 0.4), (-0.65, 0.65), 0.357, 0.30
    )
    want = [_quality(f, p, c) for f, p, c in zip(foot_cop, pad_cop, counted)]
    np.testing.assert_allclose(quality, want, rtol=1e-12, atol=1e-15)
    assert float(reward) == pytest.approx(-0.30 * (2.0 - sum(want)) / 2.0, rel=1e-12, abs=1e-15)
    assert -0.30 <= float(reward) <= 0.0 and np.isfinite(float(reward))
    swapped = reward_stance_cop_band(
        np.asarray(foot_cop[::-1]),
        np.asarray(pad_cop[::-1]),
        np.asarray(counted[::-1]),
        (-0.6, 0.4),
        (-0.65, 0.65),
        0.357,
        0.30,
    )
    assert _bytes(swapped[0]) == _bytes(reward)  # exactly invariant when the feet swap
    np.testing.assert_array_equal(swapped[1], np.asarray(quality)[::-1])


def test_band_excess():
    values = np.array([-1.0, -0.6, 0.0, 0.4, 0.7])
    np.testing.assert_allclose(band_excess(values, -0.6, 0.4), [0.4, 0.0, 0.0, 0.0, 0.3], atol=1e-15)


def test_the_slopes_match_the_design_table():
    """design.md: one foot u past the band edge, per step per 0.1 span half-length, w 0.30 and tolerance 0.357:
    0.0360 / 0.0349 / 0.0331 / 0.0297 / 0.0268 at u = 0.26 / 0.30 / 0.33 / 0.37 / 0.40."""
    h = 1e-7
    for u, want in ((0.26, 0.0360), (0.30, 0.0349), (0.33, 0.0331), (0.37, 0.0297), (0.40, 0.0268)):

        def cost(x):
            reward, _ = reward_stance_cop_band(
                np.array([-0.6 - x, 0.0]), np.array([0.0, 0.0]), np.array([True, True]), (-0.6, 0.4), (-0.65, 0.65),
                0.357, 0.30,
            )  # fmt: skip
            return -float(reward)

        slope = (cost(u + h) - cost(u - h)) / (2 * h) * 0.1
        assert slope == pytest.approx(want, abs=6e-5), u
