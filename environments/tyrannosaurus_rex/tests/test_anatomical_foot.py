"""The physics-r10 foot (``configs/plant_versions.toml`` note 20, decision D-D34).

Physics r10 replaced the plantar pad and the near-parallel digits with a sole
box standing for the metatarsal pad and three splayed, load-bearing digits.  These
tests pin what the stance gate and the touch observation rest on:

* the geometry, as DESIGN values (not literature targets: note 20 and
  ``docs/investigations/TREX_FOOT_R10_2026_10.md`` give their sources and
  their ranges), and the left foot as the mirror of the right;
* that the digits carry a share of the statue's load, out along their length
  (a regression pin, not an anatomical target);
* that the summed touch reading is the floor's normal force on every floor
  contact, under a held toe-ward ankle offset, under a forward push that
  moves the load onto the digit tips, and through the one spawn that sets a
  digit on the other foot's;
* that the whole-foot centre of pressure is the vertical load's moment about
  the ankle axis, and how it relates to the ankle's own torque.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np
import pytest

from environments.shared.config import build_env
from environments.shared.gait.morphology import Morphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.shared.gait.stance_metrics import episode_stance_metrics
from environments.shared.model_id import ModelId
from environments.shared.scripts.foot_toe_line import _action_for
from environments.tyrannosaurus_rex.anatomical.env import TyrannosaurusRexAnatomicalEnv

MODEL = ModelId("tyrannosaurus_rex", "anatomical")
SIDES = ("r", "l")
DIGITS = ("d2", "d3", "d4")
#: The touch sensors of a foot, the sole box's first.
SENSORS = ("foot", "toe_d2", "toe_d3", "toe_d4")


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(TyrannosaurusRexAnatomicalEnv.MODEL_PATH))


def _in_body(model: mujoco.MjModel, kind: str, name: str) -> np.ndarray:
    """The local position of a body, geom, site or joint (in its parent body's frame)."""
    return np.asarray(
        {"body": model.body, "geom": model.geom, "site": model.site, "joint": model.joint}[kind](name).pos
    )


def _toe_point(model: mujoco.MjModel, side: str, digit: str, *, cap: bool) -> np.ndarray:
    """A digit capsule's far end in its metatarsus frame: from the toe body's origin (the MTP joint) out along its
    +x, to the segment's end (the tip cap's centre) or, with *cap*, to the claw tip."""
    body = model.body(f"{side}_toe_{digit}")
    geom = model.geom(f"{side}_toe_{digit}_geom")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(body.quat, dtype=np.float64))
    reach = 2.0 * float(geom.size[1]) + (float(geom.size[0]) if cap else 0.0)
    return np.asarray(body.pos) + rotation.reshape(3, 3)[:, 0] * reach


class TestFootGeometry:
    """Design values of the r10 foot, in each metatarsus frame (x forward, z up; the right foot's medial side +y)."""

    def test_digit_lengths_and_divarication(self, model):
        # MTP joint (the toe body's origin) to claw tip: Sue's digit / MT III ratios at this model's MT III.
        for digit, length, yaw in (("d2", 0.171, 25.0), ("d3", 0.212, 0.0), ("d4", 0.178, -33.0)):
            body = model.body(f"r_toe_{digit}")
            geom = model.geom(f"r_toe_{digit}_geom")
            reach = 2.0 * float(geom.size[1]) + float(geom.size[0])
            assert reach == pytest.approx(length, abs=5e-4), digit
            # The capsule runs along its body's +x from the MTP joint, so the body's yaw is the chord angle.
            np.testing.assert_allclose(geom.pos, [float(geom.size[1]), 0.0, 0.0], atol=1e-12)
            quat = np.asarray(body.quat, dtype=np.float64)
            assert math.degrees(2.0 * math.atan2(quat[3], quat[0])) == pytest.approx(yaw, abs=1e-9), digit
        # The three MTP joints 8 cm ahead of the ankle, digit III's 3 mm above II's and IV's (the capsule radii).
        np.testing.assert_allclose(_in_body(model, "body", "r_toe_d2"), [0.08, 0.015, -0.203], atol=1e-12)
        np.testing.assert_allclose(_in_body(model, "body", "r_toe_d3"), [0.08, 0.0, -0.2], atol=1e-12)
        np.testing.assert_allclose(_in_body(model, "body", "r_toe_d4"), [0.08, -0.02, -0.203], atol=1e-12)

    def test_the_sole_box_and_the_contact_line(self, model):
        ankle = _in_body(model, "joint", "r_ankle")
        sole = model.geom("r_plantar_geom")
        heel = float(sole.pos[0] - sole.size[0] - ankle[0])
        front = float(sole.pos[0] + sole.size[0] - ankle[0])
        assert (heel, front) == pytest.approx((0.060, 0.165), abs=1e-12)
        assert 2.0 * float(sole.size[1]) == pytest.approx(0.11, abs=1e-12)
        assert float(sole.pos[1]) == pytest.approx(-0.006, abs=1e-12)  # centred 6 mm lateral
        # The free digits project beyond the box: the contact line through the tip caps' centres.
        centres = {digit: _toe_point(model, "r", digit, cap=False) for digit in DIGITS}
        width = float(centres["d2"][1] - centres["d4"][1])
        length = float(_toe_point(model, "r", "d3", cap=True)[0] - ankle[0]) - heel  # heel edge to digit III's tip
        assert width == pytest.approx(0.180, abs=1e-3)
        assert length == pytest.approx(0.232, abs=1e-3)
        assert width / length == pytest.approx(0.78, abs=0.01)

    def test_the_mtp_springs(self, model):
        """Per-digit passive MTP springs (a mechanical choice, note 20), resting where the keyframe puts them."""
        key = model.key("home")
        for side in SIDES:
            for digit, stiffness, damping in (("d2", 3000.0, 35.0), ("d3", 4500.0, 43.0), ("d4", 3000.0, 35.0)):
                joint = model.joint(f"{side}_toe_{digit}_joint")
                assert (float(joint.stiffness[0]), float(joint.damping[0])) == (stiffness, damping)
                assert float(joint.armature[0]) == pytest.approx(0.1)
                assert float(model.qpos_spring[joint.qposadr[0]]) == pytest.approx(math.radians(12.5))
                # The home keyframe holds each toe at its spring's rest angle, so the reset preloads nothing.
                assert float(key.qpos[joint.qposadr[0]]) == pytest.approx(math.radians(12.5), abs=1e-6)
                assert [float(x) for x in np.degrees(joint.range)] == pytest.approx([-25.0, 50.0])

    def test_the_left_foot_mirrors_the_right(self, model):
        """Every body, geom, site and joint of the left metatarsus subtree is the right one's reflection in the
        sagittal plane: positions (x, -y, z), rotations (w, -x, y, -z), joint axes as pseudo-vectors."""

        def subtree(root: int) -> list[int]:
            return [b for b in range(model.nbody) if b == root or _descends(model, b, root)]

        def mirror(vector) -> np.ndarray:
            return np.asarray(vector, dtype=np.float64) * [1.0, -1.0, 1.0]

        def mirror_quat(quat) -> np.ndarray:
            return np.asarray(quat, dtype=np.float64) * [1.0, -1.0, 1.0, -1.0]

        right = subtree(model.body("r_metatarsus").id)
        assert len(right) == 4
        for body in right:
            name = model.body(body).name
            left = model.body("l" + name[1:])
            np.testing.assert_allclose(left.pos, mirror(model.body(body).pos), atol=1e-12, err_msg=name)
            np.testing.assert_allclose(left.quat, mirror_quat(model.body(body).quat), atol=1e-12, err_msg=name)
            assert float(left.mass[0]) == float(model.body(body).mass[0]), name
            for kind, count, owner in (
                ("geom", model.ngeom, model.geom_bodyid),
                ("site", model.nsite, model.site_bodyid),
                ("joint", model.njnt, model.jnt_bodyid),
            ):
                accessor = {"geom": model.geom, "site": model.site, "joint": model.joint}[kind]
                for element in (i for i in range(count) if int(owner[i]) == body):
                    r = accessor(element)
                    l_ = accessor("l" + r.name[1:])
                    np.testing.assert_allclose(l_.pos, mirror(r.pos), atol=1e-12, err_msg=r.name)
                    if kind == "joint":
                        # A hinge axis is a pseudo-vector: its reflection is (-x, y, -z).
                        np.testing.assert_allclose(l_.axis, -mirror(r.axis), atol=1e-12, err_msg=r.name)
                        for field in ("range", "stiffness", "damping", "armature", "qpos0"):
                            np.testing.assert_array_equal(getattr(l_, field), getattr(r, field), err_msg=r.name)
                    else:
                        np.testing.assert_allclose(l_.quat, mirror_quat(r.quat), atol=1e-12, err_msg=r.name)
                        np.testing.assert_array_equal(l_.size, r.size, err_msg=r.name)
                        np.testing.assert_array_equal(l_.type, r.type, err_msg=r.name)


def _descends(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body > 0:
        body = int(model.body_parentid[body])
        if body == root:
            return True
    return False


# ── the digits' share of the statue's load ───────────────────────────────────


def test_the_digits_carry_a_share_of_the_statue_load_out_along_their_length():
    """The noise-free stance statue, post-settle window: each digit carries 12.7-14.4% of its foot's floor load
    on every step (41% together; on r9 the three carried 18%, at their bases), its load centred 0.45-0.52 of the
    way from the MTP axis to the claw tip.  A regression pin of the r10 plant, not an anatomical target: with
    reset noise the split moves with the stance's roll (``docs/investigations/TREX_FOOT_R10_2026_10.md``)."""
    env = build_env(MODEL, "stance", reset_noise_scale=0.0)
    try:
        recorder = SubstepContactRecorder(env, Morphology.from_env(env, MODEL)).attach()
        env.reset(seed=0)
        for _ in range(1000):
            action = np.zeros(env.model.nu)
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.end_step(action, float(reward))
            assert not terminated
        trace = recorder.end_episode(terminated=False, truncated=bool(truncated))
        recorder.detach()
    finally:
        env.close()
    metrics = episode_stance_metrics(trace, settle_steps=200)
    names = trace.support_geom_names
    for side in SIDES:
        digits = [names.index(f"{side}_toe_{digit}_geom") for digit in DIGITS]
        assert sum(metrics.support_geom_share[i] for i in digits) > 0.35, side
        for column in digits:
            assert metrics.support_geom_share[column] > 0.10, names[column]
            assert metrics.support_geom_duty[column] == 1.0, names[column]
            assert float(np.mean(trace.support_load_pos[200:, column])) > 0.4, names[column]


# ── touch against floor truth ────────────────────────────────────────────────


class _TouchProbe:
    """A substep hook that compares each touch sensor with the floor's normal force on its site's body.

    Floor truth per sensor is the ``mj_contactForce`` normal of every floor contact of a geom on the
    sensor site's body.  A substep with any other loaded contact on those bodies (a foot on the other
    foot) is counted apart: a touch sensor reads every contact in its volume, the floor's or not.
    """

    def __init__(self, env) -> None:
        model = env.model
        self.env = env
        self.floor = model.geom("floor").id
        names = [f"{side}_{sensor}_touch" for side in SIDES for sensor in SENSORS]
        self.address = np.array([int(model.sensor(name).adr[0]) for name in names])
        self.body = np.array([int(model.site_bodyid[int(model.sensor(name).objid[0])]) for name in names])
        self.worst = 0.0
        self.substeps = 0
        self.nonfloor_steps: list[int] = []
        self._force = np.zeros(6)
        self._previous: Callable[[], None] | None = None

    def __enter__(self) -> "_TouchProbe":
        self._previous = self.env._substep_probe_hook
        self.env._substep_probe_hook = self._hook
        return self

    def __exit__(self, *exc: object) -> None:
        self.env._substep_probe_hook = self._previous

    def _hook(self) -> None:
        if self._previous is not None:
            self._previous()
        model, data = self.env.model, self.env.data
        truth = np.zeros(len(self.address))
        nonfloor = False
        for index in range(data.ncon):
            contact = data.contact[index]
            mujoco.mj_contactForce(model, data, index, self._force)
            geoms = (int(contact.geom1), int(contact.geom2))
            bodies = [int(model.geom_bodyid[g]) for g in geoms]
            if self.floor in geoms:
                truth[self.body == bodies[1 - geoms.index(self.floor)]] += self._force[0]
            elif self._force[0] > 0.0 and any(b in self.body for b in bodies):
                nonfloor = True
        if nonfloor:
            self.nonfloor_steps.append(int(self.env._step_count))
            return
        touch = np.asarray(data.sensordata[self.address], dtype=np.float64)
        self.worst = max(self.worst, float(np.max(np.abs(touch - truth))))
        self.substeps += 1


def _stance_env():
    return build_env(MODEL, "stance")


class TestTouchIsFloorTruth:
    def test_under_a_held_toe_ward_ankle_offset(self):
        """The right ankle held 4.5 deg toe-ward (the last level of the toe-line rule at which no seed falls):
        the left foot's load moves onto its digits, and every sensor still reads exactly the floor's force."""
        env = _stance_env()
        try:
            morphology = Morphology.from_env(env, MODEL)
            recorder = SubstepContactRecorder(env, morphology).attach()
            nu = int(env.model.nu)
            actuator = [a for a in range(nu) if env.model.actuator(a).name == "r_ankle_act"][0]
            home = np.asarray(env._scale_action(np.zeros(nu)), dtype=np.float64)
            hold = np.zeros(nu)
            hold[actuator] = _action_for(env._scale_action, nu, actuator, home[actuator] + math.radians(-4.5))
            with _TouchProbe(env) as probe:
                env.reset(seed=3042)
                for t in range(1000):
                    action = hold * min(1.0, (t + 1) / 200)
                    _, reward, terminated, truncated, _ = env.step(action)
                    recorder.end_step(action, float(reward))
                    assert not terminated
            trace = recorder.end_episode(terminated=False, truncated=bool(truncated))
            recorder.detach()
        finally:
            env.close()
        assert probe.nonfloor_steps == [] and probe.substeps == 5000
        assert probe.worst == pytest.approx(0.0, abs=1e-6)
        # The other (left) foot's whole-foot CoP sits 3 cm past the sole box's front edge (0.165 m).
        assert float(np.nanmean(trace.foot_cop[200:, 1])) > 0.195
        metrics = episode_stance_metrics(trace, settle_steps=200)
        left_digits = [trace.support_geom_names.index(f"l_toe_{digit}_geom") for digit in DIGITS]
        assert sum(metrics.support_geom_share[i] for i in left_digits) > 0.45  # about a third on the statue

    def test_under_a_forward_push_with_the_load_on_the_digit_tips(self):
        """The noise-free statue pushed forward at its pelvis, ramped at 20 N/s from step 200 and capped at 58 N
        (the foot proposal's push limit, 58.4 N, found with its own fall criterion; under the env's termination
        the statue nosedives at every cap from 40 to 58 N, and holds at 35 N), until the env's nosedive
        termination: the sole boxes unload with all three digits loaded out at their tips, the load then
        gathers on digit III's tip, and touch is floor truth on every substep of it."""
        env = TyrannosaurusRexAnatomicalEnv(reset_noise_scale=0.0)
        try:
            recorder = SubstepContactRecorder(env, Morphology.from_env(env, MODEL)).attach()
            pelvis = env.model.body("pelvis").id
            with _TouchProbe(env) as probe:
                env.reset(seed=0)
                for t in range(700):
                    env.data.xfrc_applied[pelvis, 0] = 0.0 if t < 200 else min(58.0, 0.2 * (t - 199))
                    action = np.zeros(env.model.nu)
                    _, reward, terminated, truncated, info = env.step(action)
                    recorder.end_step(action, float(reward))
                    if terminated:
                        break
            trace = recorder.end_episode(
                terminated=bool(terminated), truncated=False, termination_reason=str(info.get("termination_reason"))
            )
            recorder.detach()
        finally:
            env.close()
        assert terminated and info["termination_reason"] == "nosedive"
        assert probe.nonfloor_steps == [] and probe.substeps == 5 * trace.length
        assert probe.worst == pytest.approx(0.0, abs=1e-6)
        names = trace.support_geom_names
        soles = [names.index(f"{side}_plantar_geom") for side in SIDES]
        unloaded = int(np.flatnonzero((trace.support_floor_mean[200:, soles] < 1e-9).all(axis=1))[0]) + 200
        assert 0.2 * (unloaded - 199) > 45.0  # the box holds until most of the limit push
        for foot, side in enumerate(SIDES):
            assert float(trace.foot_cop[unloaded, foot]) > 0.213  # past the toe line
            for digit in DIGITS:
                column = names.index(f"{side}_toe_{digit}_geom")
                assert float(trace.support_floor_mean[unloaded, column]) > 20.0, (side, digit)
                assert float(trace.support_load_pos[unloaded, column]) > 0.8, (side, digit)

    def test_the_seed_7068_spawn_sets_digit_ii_on_the_other_foot_for_four_substeps_only(self):
        """The digits' splay leaves 7.7 cm between the two digit-II tips at home (14.0 cm on r9).  Reset noise on
        seed 7068 spawns the right digit II on the left one: both toe-II sensors read the 118-174 N between them
        with no floor load, for the first 4 substeps of control step 0, and never again.  Every other substep's
        touch is floor truth."""
        env = _stance_env()
        try:
            with _TouchProbe(env) as probe:
                env.reset(seed=7068)
                for _ in range(200):
                    env.step(np.zeros(env.model.nu))
        finally:
            env.close()
        assert probe.nonfloor_steps == [0, 0, 0, 0]
        assert probe.substeps == 200 * 5 - 4
        assert probe.worst == pytest.approx(0.0, abs=1e-6)


# ── the whole-foot centre of pressure and the ankle torque ───────────────────


def test_the_whole_foot_cop_is_the_vertical_load_moment_and_the_ankle_torque_adds_the_shear():
    """On the stance statue (seed 3042), per control step and foot, pooled over its substeps as the recorder
    pools ``foot_cop``:

    * the vertical floor load's moment about the ankle axis over the limb's floor force is the whole-foot CoP,
      up to the sole's tilt (PR #622's identity; within 1.5 mm here);
    * the ankle's actuator plus passive torque balances that moment together with the floor's SHEAR times the
      contacts' height below the ankle and the foot's own weight: with both terms the residual is the foot's
      inertia (within 0.05 mm of the CoP here), while the shear term alone is worth several mm, so the ankle
      torque over the floor force is not the centre of pressure.
    """
    env = _stance_env()
    model, data = env.model, env.data
    morphology = Morphology.from_env(env, MODEL)
    recorder = SubstepContactRecorder(env, morphology).attach()
    floor = model.geom("floor").id
    feet = []
    for foot in morphology.feet:
        joint = int(foot.ankle_joint)
        body = int(model.jnt_bodyid[joint])
        feet.append(
            {
                "joint": joint,
                "dof": int(model.jnt_dofadr[joint]),
                "bodies": {b for b in range(model.nbody) if b == body or _descends(model, b, body)},
            }
        )
    rows: list[list[tuple[float, ...]]] = []
    force = np.zeros(6)

    def hook() -> None:
        previous()
        row: list[tuple[float, ...]] = []
        for foot in feet:
            anchor, axis = data.xanchor[foot["joint"]].copy(), data.xaxis[foot["joint"]].copy()
            vertical = shear = load = 0.0
            for index in range(data.ncon):
                contact = data.contact[index]
                geoms = (int(contact.geom1), int(contact.geom2))
                if floor not in geoms:
                    continue
                other = geoms[1 - geoms.index(floor)]
                if int(model.geom_bodyid[other]) not in foot["bodies"]:
                    continue
                mujoco.mj_contactForce(model, data, index, force)
                # The contact frame's rows are its axes in world; the force acts on geom2 from geom1.
                on_foot = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
                if geoms[1] != other:
                    on_foot = -on_foot
                lever = np.asarray(contact.pos) - anchor
                up = np.array([0.0, 0.0, on_foot[2]])
                vertical += float(axis @ np.cross(lever, up))
                shear += float(axis @ np.cross(lever, on_foot - up))
                load += float(on_foot[2])
            weight = 0.0 + sum(
                float(axis @ np.cross(data.xipos[b] - anchor, model.body_mass[b] * model.opt.gravity))
                for b in foot["bodies"]
            )
            torque = float(data.qfrc_actuator[foot["dof"]] + data.qfrc_passive[foot["dof"]])
            row.append((vertical, shear, weight, torque, load))
        rows.append(row)

    previous = env._substep_probe_hook
    env._substep_probe_hook = hook
    try:
        env.reset(seed=3042)
        for _ in range(1000):
            action = np.zeros(model.nu)
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.end_step(action, float(reward))
            assert not terminated
        trace = recorder.end_episode(terminated=False, truncated=bool(truncated))
    finally:
        env._substep_probe_hook = previous
        recorder.detach()
        env.close()
    pooled = np.asarray(rows).reshape(1000, 5, 2, 5).sum(axis=1)  # per step and foot, summed over substeps
    window = slice(200, None)
    for index, foot in enumerate(morphology.feet):
        vertical, shear, weight, torque, load = (pooled[window, index, k] for k in range(5))
        cop = trace.foot_cop[window, index]
        # The ankle axis is the sole's lateral axis, so a load ahead of it turns the foot one way: find the sign.
        sign = float(np.sign(np.median(vertical / load / cop)))
        from_vertical = sign * vertical / load
        from_torque = sign * -(torque + shear + weight) / load
        assert np.max(np.abs(from_vertical - cop)) < 1.5e-3, foot.label
        assert np.max(np.abs(from_torque - from_vertical)) < 5e-5, foot.label
        shear_term = np.max(np.abs(shear / load))
        assert 5e-3 < shear_term < 15e-3, (foot.label, shear_term)
