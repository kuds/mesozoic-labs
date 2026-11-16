"""What R5's env and the policy rows share (``designlab/stage2/r5core.py``; round C design D7-D10, sections 3.1,
3.4; rules E1-E5, W1-W5, J4).

The reward is tested on a **kinematic playback**: a script's own poses (the bar's placement of the files in one
world frame, ``bar._Played``) written into the plant's data at the commanded clock, the foot loads from the
reference's support flags, no physics.  On it every imitation term is 1 and every penalty below 1e-4 (rule W1 as
the lead ruled it on 16 Nov 2026: ``progress`` is not exactly 0, and ``heading`` is charged inside a walking turn);
each term then moves when its cause is injected into the playback.  The S1 view only (Stage 1's run directory).
"""

from __future__ import annotations

import copy
import math
import os
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from designlab.stage1 import HEAD_X_BODIES, control, env, legs, metrics, plant, tune  # noqa: E402
from designlab.stage2 import r5core, r5lib, s2io, setio  # noqa: E402

BODY, BODY_X = "75-M-H-10", "75-M-H-10-X"
RATE = 1.05
ONE_THREAD = all(os.environ.get(v) == "1" for v in s2io.THREAD_VARS)


def _have(body: str) -> bool:
    try:
        return ((r5lib.s1_root(body) / body / "transitions" / "stop.npz").exists()
                and all(Path(f).exists() for f in plant.model_files(body)))
    except Exception:  # noqa: BLE001  (no model record: the same skip)
        return False


needs_run = pytest.mark.skipif(not (_have(BODY) and ONE_THREAD), reason=(
    "needs the Stage 1 arrays and the round-two models (tools/restore_inputs.py) and one BLAS thread"))
needs_x = pytest.mark.skipif(not (_have(BODY_X) and ONE_THREAD), reason=(
    "needs the expressive-head body's models and arrays (tools/restore_inputs.py --group x) and one BLAS thread"))

#: Every body with a head that takes a command (a neck yaw and a jaw): each test on them skips where its models and
#: arrays are not restored.
HEAD_BODIES = [pytest.param(b, marks=pytest.mark.skipif(not (_have(b) and ONE_THREAD), reason=(
    f"needs {b}'s models and arrays (tools/restore_inputs.py --group x) and one BLAS thread"))) for b in HEAD_X_BODIES]

J14 = ["neck_pitch", "tail_yaw"] + [f"{s}_{j}" for s in ("left", "right") for j in control.LEGS]
J16 = ["neck_yaw", "neck_pitch", "jaw", "tail_yaw"] + J14[2:]


# ------------------------------------------------------------------------------------------------
# Layout (section 3.1; rule E3)
# ------------------------------------------------------------------------------------------------
#: Section 3.1's table, written out: block -> (first, last plus one) for 14 and for 16 joints.
LAYOUT_14 = dict(q=(0, 14), qd=(14, 28), gravity=(28, 31), gyro=(31, 34), foot_load=(34, 36), phase=(36, 38),
                 support=(38, 40), command=(40, 42), heading=(42, 43), prev_actions=(43, 88), root_vel=(88, 91),
                 height=(91, 92), roll_pitch=(92, 94), dcm_error=(94, 96), joint_error=(96, 110),
                 heading_error=(110, 111), friction=(111, 112), floor=(112, 114), push=(114, 117),
                 plant_factors=(117, 122), foot_error=(122, 128), ref_contact=(128, 130))
LAYOUT_16 = dict(q=(0, 16), qd=(16, 32), gravity=(32, 35), gyro=(35, 38), foot_load=(38, 40), phase=(40, 42),
                 support=(42, 44), command=(44, 46), heading=(46, 47), prev_actions=(47, 98), head_command=(98, 102),
                 root_vel=(102, 105), height=(105, 106), roll_pitch=(106, 108), dcm_error=(108, 110),
                 joint_error=(110, 126), heading_error=(126, 127), friction=(127, 128), floor=(128, 130),
                 push=(130, 133), plant_factors=(133, 138), foot_error=(138, 144), ref_contact=(144, 146))


def test_layouts():
    for joints, want, dims in ((J14, LAYOUT_14, (88, 130, 15)), (J16, LAYOUT_16, (102, 146, 17))):
        lay = r5core.layout(joints)
        assert lay["blocks"] == want
        assert (lay["actor_dim"], lay["obs_dim"], lay["action_dim"]) == dims and lay["nj"] == len(joints)
        assert lay["head_x"] == (len(joints) == 16) and lay["heading"] is True
        spans = list(lay["blocks"].values())
        assert spans[0][0] == 0 and spans[-1][1] == lay["obs_dim"]
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))  # contiguous
        assert lay["actor_dim"] == lay["blocks"]["root_vel"][0]  # the actor's inputs come first
        # without the heading input: one input fewer, every later block one lower
        no = r5core.layout(joints, heading=False)
        assert (no["actor_dim"], no["obs_dim"], no["action_dim"]) == (dims[0] - 1, dims[1] - 1, dims[2])
        assert "heading" not in no["blocks"] and no["heading"] is False
        for name, (a, b) in no["blocks"].items():
            shift = 1 if want[name][0] > want["heading"][0] else 0
            assert (a + shift, b + shift) == want[name]
    # a head needs both of its joints, whatever their place in the list
    assert r5core.layout(J14 + ["neck_yaw"])["head_x"] is False
    assert r5core.layout(list(reversed(J16)))["head_x"] is True


# ------------------------------------------------------------------------------------------------
# The reward's constants (rule W2's table, written out)
# ------------------------------------------------------------------------------------------------
W2_POSITIVE = {"joint": 0.55, "tracking": 0.35, "foot": 0.15, "contact": 0.10, "head": 0.10}
W2_PENALTIES = {"duty": 0.02, "slip": 0.02, "impact": 0.02, "tilt": 0.02, "flight": 0.05, "clearance": 0.02,
                "action_rate": 0.03, "touchdown": 0.02, "servo": 0.05, "interference": 0.10, "progress": 0.10,
                "heading": 0.05, "range": 0.05}
W2_SCALES = dict(
    joint_rad=0.1, foot_frac_H=0.10, head_deg=3.0, contact_load_frac=0.02, contact_lag_s=0.024,
    duty_band=(0.55, 0.65), duty_band_slow=(0.55, 0.72), duty_slow_below_m_s=0.30, duty_over=0.05,
    slip_load_frac=0.2, slip_free_m_s=0.05, slip_scale_m_s=0.05, impact_free_bw=1.6, impact_scale_bw=1.6,
    clearance_frac_H=0.04, mid_swing=(0.25, 0.75), tilt_free_deg=4.0, tilt_scale_deg=4.0, action_rate=0.25,
    touchdown_free_m_s=0.15, touchdown_scale_m_s=0.30, servo_tau_s=1.0, servo_over=0.5, interference_m=0.003,
    progress_short=0.10, heading_free_deg=3.0, heading_scale_deg=17.0, range_rad=0.02)


def test_reward_weights_and_scales_equal_the_rules_table():
    assert r5core.REWARD_WEIGHTS == {**W2_POSITIVE, **W2_PENALTIES}
    assert r5core.POSITIVE == tuple(W2_POSITIVE) and sorted(r5core.PENALTIES) == sorted(W2_PENALTIES)
    assert dict(r5core.REWARD_WEIGHTS_ITEMS) == r5core.REWARD_WEIGHTS
    assert r5core.REWARD_SCALES == W2_SCALES
    assert sum(W2_POSITIVE.values()) == pytest.approx(1.25)
    # Stage 1's terms keep Stage 1's weights; no new term outweighs Stage 1's joint imitation
    assert (r5core.REWARD_WEIGHTS["joint"], r5core.REWARD_WEIGHTS["tracking"]) == (env.IMITATION_W, env.TRACKING_W)
    assert {k: r5core.REWARD_WEIGHTS[k] for k in env.PENALTIES} == env.PENALTIES
    assert max(v for k, v in r5core.REWARD_WEIGHTS.items() if k != "joint") < r5core.REWARD_WEIGHTS["joint"]
    assert r5core.SUBSTEPS == env.ENV_SUBSTEPS == 10
    for word in ("joint", "tracking", "foot", "contact", "head", *W2_PENALTIES):
        assert word in r5core.REWARD_RULE


def test_the_reward_rule_states_the_readings_as_coded():
    """The lead's ruling of 16 Nov 2026: the readings of duty, progress and clearance stand as coded and the rule
    text follows them.  The rule string an env description and every record carry states each reading, and holds
    no formula of the wording it replaced (the tests of the next sections pin what the code computes)."""
    rule = " ".join(r5core.REWARD_RULE.split())
    for said in (
            "a foot is read only where the reference's own support share for it over the same physics steps, each "
            "at the clock, lies inside the band [lo, hi]",
            "min(1, max(0, (v_ref_bar - v_bar) / (rate x v_cmd)) / 0.10)",
            "the same mean over the same env steps of the reference's own sample",
            "restarted at the segment's first frame",
            "T the planned length, in reference time, of the reference's swing in progress at c",
            "which the simulator computes at that physics step's start, 2 ms before the env step's end",
            "joint angles are those at the env step's end",
            "in a turn-in-place task the reference's own yaw",
            "below 0.30 m/s, else 0.55-0.65",
            # the six details the review of the fix round found unstated (16 Nov 2026), each pinned by a test below
            # or in tests/test_stage2_r5env.py:
            # (a) the first env step's displacement
            "In an episode's first env step the displacement behind tracking's speed and yaw rate and behind "
            "progress's sample runs from the reading at the reset, which is of the episode's start, to the step's "
            "reading 18 ms later, and is divided by 20 ms like every other",
            # (b) what a playback scores on progress
            "progress below 1e-4 and every other penalty 0, except heading inside a finite walking turn",
            # (c) the bounds of duty's n
            "n = round(the stride period at c / the clock rate / 20 ms), kept between 1 and 256 env steps",
            # (d) clearance (a) without a swing length
            "where T is not above zero, (a) is not read",
            # (e) the instant of the readings taken at every physics step
            "What is read at every physics step (the foot loads, and with them a foot's contact, for contact, duty, "
            "flight, impact, touchdown, a foot's own lift-off and clearance (b); the foot bodies' heights for "
            "touchdown's speed; the actuator forces for servo) is likewise the plant's own after that physics step, "
            "computed at its start, and the reference's flags beside it are read at the reference clock after that "
            "physics step"):
        assert said in rule, said
    for gone in ("1 - v_bar", "Stage 1's:", "(Stage 1's)", "the planned swing time", "and every penalty 0"):
        assert gone not in rule, gone
    assert len(r5core.READINGS) == 4 and [x.split(":")[0] for x in r5core.READINGS] == [
        "duty", "progress", "clearance (a)", "positions and foot loads"]
    assert "after every physics step for what is read there" in r5core.READINGS[3]
    assert r5core.HISTORY_STEPS == 256
    # (f) the head rule's first sentence: R5's overlay on the reference, R6's on the settled home pose
    head = " ".join(r5core.HEAD_RULE.split())
    for said in ("in R5 it is added to the reference's neck-yaw and jaw targets before the policy's offsets",
                 "in R6, which has no reference controller, it is added to the settled home pose's neck-yaw and jaw "
                 "angles before the policy's term (the joint targets are clip(home + overlay + action x span, range))",
                 "the neck yaw settles within 1e-7 rad of zero",
                 "an R6 jaw target reaches 0.4515 and 0.4507 rad"):
        assert said in head, said


# ------------------------------------------------------------------------------------------------
# The head script (rules E5, J4)
# ------------------------------------------------------------------------------------------------
def test_head_script_h1_and_the_training_draw():
    h = r5core.HeadScript.fixed(3042, 20.0)
    assert h.events == r5core.HeadScript.fixed(3042, 20.0).events != r5core.HeadScript.fixed(3043, 20.0).events
    gaze = [e for e in h.events if e[0] == "gaze"]
    jaw = [e for e in h.events if e[0] == "jaw"]
    assert len(jaw) == 1 and len(gaze) >= 3 and gaze[0][1] == 0.0  # both always present; the train starts at once
    t = 0.0
    for _, t0, move, hold, target in gaze:
        assert t0 == pytest.approx(t) and 0.8 <= move <= 1.2 and 2.0 <= hold <= 6.0 and abs(target) <= 0.45
        t = t0 + move + hold
    assert t >= 20.0 and gaze[-1][1] < 20.0  # the whole episode
    _, t0, move, hold, opening = jaw[0]
    assert 5.0 <= t0 <= 15.0 and 0.4 <= move <= 0.6 and 1.0 <= hold <= 1.5 and 0.3 <= opening <= 0.45
    # the script's numbers (rules E5, J4 as the lead ruled them on 16 Nov 2026: 0.15 rad from a joint limit)
    assert (r5core.GAZE_YAW_RAD, r5core.JAW_OPEN_RAD, r5core.HEAD_TARGET_MARGIN_RAD) == (0.45, (0.3, 0.45), 0.15)
    assert (r5core.GAZE_MOVE_S, r5core.GAZE_HOLD_S) == ((0.8, 1.2), (2.0, 6.0))
    assert (r5core.JAW_MOVE_S, r5core.JAW_HOLD_S, r5core.JAW_AT, r5core.H1_STREAM) == ((0.4, 0.6), (1.0, 1.5),
                                                                                    (0.25, 0.75), 77)
    for text in ("+-0.45 rad", "0.3-0.45 rad", "0.15 rad from a joint limit", "middle half", "first instant"):
        assert text in r5core.HEAD_RULE
    assert "0.3-0.6 rad" not in r5core.HEAD_RULE and "+-0.5 rad" not in r5core.HEAD_RULE
    # the stream is SeedSequence([seed, 77])
    rng = np.random.default_rng(np.random.SeedSequence([3042, 77]))
    assert gaze[0][4] == float(rng.uniform(-0.45, 0.45))
    assert h.value(0.0) == (0.0, 0.0, gaze[0][4], 0.0)  # at rest at the first instant: exactly zero
    # minimum-jerk moves: continuous, at the target when the move ends, held, the goal known from the move's start
    _, t1, move1, _, x1 = gaze[1]
    assert h.value(t1 - 1e-9)[0] == pytest.approx(gaze[0][4]) and h.value(t1 + move1)[0] == pytest.approx(x1)
    assert h.value(t1 + 1e-9)[2] == x1 and h.value(t1 + 0.5 * move1)[0] == pytest.approx(0.5 * (gaze[0][4] + x1))
    ts = np.arange(0.0, 20.0, 0.002)
    val = np.array([h.value(float(x)) for x in ts])
    assert np.max(np.abs(np.diff(val[:, 0]))) < 0.005 and np.max(np.abs(np.diff(val[:, 1]))) < 0.008
    assert np.max(np.abs(val[:, 0])) <= 0.45 and 0.0 <= val[:, 1].min() and val[:, 1].max() == pytest.approx(opening)
    assert h.value(t0 + move + 0.5 * hold)[1::2] == (opening, opening)
    assert h.value(t0 + 2 * move + hold + 0.1)[1::2] == (0.0, 0.0)  # closed over the same time, goal back at zero
    # training (rule J4): probabilities from the env's generator; neither drawn is no script at all
    rng = np.random.default_rng(0)
    assert r5core.HeadScript.drawn(rng, 20.0, 0.0, 0.0) is None
    both = r5core.HeadScript.drawn(rng, 20.0, 1.0, 1.0)
    assert {e[0] for e in both.events} == {"gaze", "jaw"}
    kinds = [r5core.HeadScript.drawn(rng, 20.0, 0.5, 0.25) for _ in range(400)]
    n_gaze = sum(1 for k in kinds if k is not None and any(e[0] == "gaze" for e in k.events))
    n_jaw = sum(1 for k in kinds if k is not None and any(e[0] == "jaw" for e in k.events))
    assert 160 <= n_gaze <= 240 and 70 <= n_jaw <= 130
    assert r5core.HeadScript(h.describe()["events"]).events == h.events


# ------------------------------------------------------------------------------------------------
# Fixtures: a plant, the S1 library, a kinematic playback
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def p():
    return plant.build(BODY, "hard")


@pytest.fixture(scope="module")
def s1(p):
    return r5lib.Library(p, BODY, "S1", RATE)


@pytest.fixture(scope="module")
def core(p):
    return r5core.Core(p, {}, None)


SET = "T0"
V03_BELOW = float(np.nextafter(0.3, 0.0))
SET_REFS = (("nominal", "nominal"), ("v0.3", "nominal"), ("inplace_0.5_left", "in_place_0.50_left"),
            ("inplace_0.5_right", "in_place_0.50_right"))
SET_MOVES = (("start", "start"),) + tuple((f"{k}_inplace_0.5_{d}", f"{k}_in_place_0.50_{d}")
                                          for d in ("left", "right") for k in ("start", "stop"))


def make_set(root: Path, body: str = BODY) -> Path:
    """A frozen synthetic set ``T0`` under ``root``, of Stage 1's arrays under the names a Stage 2 set gives them:
    ``nominal`` and ``start``; ``v0.3`` (Stage 1's nominal walk again, its entry's speed one ulp below 0.30 m/s);
    the turn in place with its start and its stop under design D5's names."""
    src = r5lib.s1_root(body) / body
    d = setio.set_dir(SET, body, root)
    refs1, trs1 = s2io.read_json(src / "refs.json"), s2io.read_json(src / "transitions.json")
    refs, trs, manifest = {}, {}, {}
    for name, old in SET_REFS:
        e = copy.deepcopy(refs1["references"][old])
        e.update(name=name, file=f"refs/{name}.npz", set=SET, strides=1, window_s=e["period_s"], crouch_frac=0.03)
        if name == "v0.3":
            e["speed_m_s"] = V03_BELOW
        refs[name] = e
    for name, old in SET_MOVES:
        e = copy.deepcopy(trs1["transitions"][old])
        e.update(name=name, file=f"transitions/{name}.npz", a1_over=None,
                 move=dict(kind=name.split("_")[0], builder="fixture", causal=False, command_phase=None))
        trs[name] = e
    for doc, sub, kind in ((refs, "refs", "walk"), (trs, "transitions", "move")):
        for name, e in doc.items():
            (d / sub).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / sub / f"{dict(SET_REFS + SET_MOVES)[name]}.npz", d / e["file"])
            manifest[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256(kind, e))
    s2io.write_json(d / "refs.json", dict(body=body, set=SET, schema=1, references=refs))
    s2io.write_json(d / "transitions.json", dict(body=body, set=SET, schema=1, transitions=trs))
    freeze = dict(index=0, at="fixture", commit=None, manifest=manifest, manifest_sha256=s2io.sha256_json(manifest))
    s2io.write_json(d / "set.json", dict(set=SET, body=body, role="fixture", crouch_frac=0.03, not_available={},
                                         freezes=[freeze]))
    return d


@pytest.fixture(scope="module")
def t0(p, tmp_path_factory):
    root = tmp_path_factory.mktemp("sets")
    make_set(root)
    return r5lib.Library(p, BODY, SET, RATE, root=root)


def played_turning(track, run_dir):
    """``bar._Played`` for a track that holds a turn in place, which ``bar._Played`` itself refuses: the same
    placement of the files in one world frame, a periodic window carried on by the file's own stride motion
    (``turning.move_pose``: a turn about a point), any number of windows."""
    from designlab import turning
    from designlab.stage1.refs import Reference
    from designlab.stage1.transitions import Transition
    from designlab.stage2 import bar

    class Turning(bar._Played):
        def __init__(self, track, run_dir):  # (bar._Played's own lines, less its refusal of a turning gait)
            self.track, run_dir = track, Path(run_dir)
            self.objs = [Reference.load(run_dir / "refs" / f"{seg.name}.npz") if seg.kind == "periodic" else
                         Transition.load(run_dir / "transitions" / f"{seg.name}.npz") if seg.kind == "finite" else None
                         for seg in track.segments]
            self.T, self.held = [None] * len(self.objs), [None] * len(self.objs)
            pose = None
            for i, obj in enumerate(self.objs):
                if obj is None:
                    self.held[i] = np.array(self.objs[1].q[0], float) if i == 0 else self._file_end_frame(i - 1)
                    pose = bar._xy_yaw(self.held[i]) if i == 0 else pose
                    self.T[i] = (0.0, 0.0, 0.0)
                    continue
                self.T[i] = (0.0, 0.0, 0.0) if pose is None else bar._onto(pose, bar._xy_yaw(obj.q[0]))
                pose = bar._xy_yaw(bar._move(self.T[i], self._end_pose(i)))

        def _periodic(self, i: int, k: int) -> np.ndarray:
            ref = self.objs[i]
            q = np.array(ref.q[k % ref.n], float)
            phi, ax, ay, bx, by = (float(x) for x in ref.stride_motion)
            for _ in range(k // ref.n):
                q = turning.move_pose(q, phi, (ax, ay), (bx, by))
            return q

    return Turning(track, run_dir)


class Playback:
    """A script played kinematically on a plant's data with a ``Core`` reading it (module docstring).  What a test
    injects: ``factor`` (the clock's rate over the commanded one), ``pose(q, c)`` (changes the played pose),
    ``load_lag`` (the loads are the reference's this much reference time earlier), ``loads(c)`` (replaces them),
    ``yaw`` (the whole pose turned about the vertical) and ``force`` (actuator forces).  ``played`` places the
    files (default: ``bar._Played``; ``played_turning`` for a turn in place)."""

    def __init__(self, p, core, script, run_dir, t0: float = 0.0, head=None, played=None):
        from designlab.stage2 import bar

        self.p, self.core, self.script, self._move = p, core, script, bar._move
        self.play = (bar._Played if played is None else played)(script.track, run_dir)
        self.factor, self.pose, self.load_lag, self.loads, self.yaw, self.force = 1.0, None, 0.0, None, 0.0, None
        self.c = float(t0)
        self.weight = float(p.m.body_subtreemass[p.trunk]) * 9.81
        mujoco.mj_resetData(p.m, p.d)
        self.set()
        core.begin(lambda: self.c, self.est, script, head)
        self.terms = []

    def set(self) -> None:
        p, d, c = self.p, self.p.d, self.c
        q = self.play.q(c)
        if self.pose is not None:
            q = self.pose(q, c)
        if self.yaw:
            q = self._move((self.yaw, 0.0, 0.0), q)
        d.qpos[:] = q
        mujoco.mj_kinematics(p.m, d)
        mujoco.mj_comPos(p.m, d)
        d.sensordata[p._load_adr] = (self.play.loads(max(c - self.load_lag, 0.0), self.weight) if self.loads is None
                                     else self.loads(c))
        d.actuator_force[:] = 0.0
        if self.force is not None:
            for i, f in self.force.items():
                d.actuator_force[i] = f

    def est(self) -> dict:
        p, d = self.p, self.p.d
        R = d.xmat[p.trunk].reshape(3, 3)
        return dict(q=d.qpos[p.qadr].copy(), qd=np.zeros(len(p.joints)), gravity=-R[2], gyro=np.zeros(3),
                    foot_load=d.sensordata[p._load_adr].copy(), yaw=control._yaw(R))

    def step(self, action=None) -> dict:
        core, p = self.core, self.p
        a_f = core.filter(np.zeros(core.nj + 1) if action is None else action)
        _, rf = core.offsets(a_f)
        c0 = self.c
        for _ in range(r5core.SUBSTEPS):
            self.c += core.dt * core.rate * rf * self.factor
            p.d.time += core.dt
            self.set()
            core.substep()
        core.interference()
        r, t = core.reward(c0)
        assert set(t) == set(r5core.REWARD_WEIGHTS) and all(0.0 <= v <= 1.0 for v in t.values()), t  # rule W1
        want = sum(core.weights[k] * t[k] for k in r5core.POSITIVE) - sum(core.weights[k] * t[k]
                                                                         for k in r5core.PENALTIES)
        assert r == pytest.approx(want, abs=1e-12)
        self.terms.append(t)
        return t

    def run(self, seconds: float, action=None) -> dict:
        """Play ``seconds`` (simulated); the smallest and the largest value of every term over them."""
        first = len(self.terms)
        for _ in range(int(round(seconds / 0.02))):
            self.step(action)
        rows = self.terms[first:]
        return {k: (min(t[k] for t in rows), max(t[k] for t in rows)) for k in rows[0]}


def playback(p, core, lib, task: str, t0: float = 0.0, head=None) -> Playback:
    pytest.importorskip("designlab.stage2.bar")
    return Playback(p, core, lib.script(task), lib.dir, t0, head)


def steady_time(script, strides: float = 1.5) -> float:
    """A reference time ``strides`` into the script's first periodic walking segment."""
    seg = next(s for s in script.track.segments if s.kind == "periodic")
    return seg.t0 + strides * seg.period


# ------------------------------------------------------------------------------------------------
# Zero is zero; the filter and the history
# ------------------------------------------------------------------------------------------------
@needs_run
def test_a_zero_filtered_action_gives_none_and_exactly_one(p, core, s1):
    pb = playback(p, core, s1, "walk")
    n = core.nj + 1
    off, rf = core.offsets(core.filter(np.zeros(n)))
    assert off is None and rf == 1.0 and type(rf) is float
    assert not core.a_f.any() and not core.hist.any()
    tiny = np.zeros(n)
    tiny[3] = 1e-6
    off, rf = core.offsets(core.filter(tiny))
    assert off is not None and off[3] == core.alpha * 1e-6 * core.scale[3] and np.count_nonzero(off) == 1 and rf == 1.0
    clock = np.zeros(n)
    clock[-1] = 1e-6
    pb = playback(p, core, s1, "walk")
    off, rf = core.offsets(core.filter(clock))
    assert off is None and rf == 1.0 + 0.25 * core.alpha * 1e-6 != 1.0
    assert pb.core.clock_stats()["steps"] == 1


@needs_run
def test_filter_clips_low_passes_and_keeps_three_actions_newest_first(p, core, s1):
    playback(p, core, s1, "walk")
    n = core.nj + 1
    assert core.alpha == 1.0 - math.exp(-2.0 * math.pi * 10.0 * 0.02) and core.dt_env == 0.02
    a1, a2, a3, a4 = (np.full(n, x) for x in (3.0, -0.5, 0.25, -7.0))
    f1 = core.filter(a1)
    np.testing.assert_array_equal(f1, np.full(n, core.alpha))  # clipped to 1 before the filter
    f2 = core.filter(a2)
    np.testing.assert_allclose(f2, core.alpha + core.alpha * (-0.5 - core.alpha))
    core.filter(a3)
    np.testing.assert_array_equal(core.hist, np.array([np.full(n, 0.25), np.full(n, -0.5), np.full(n, 1.0)]))
    core.filter(a4)
    np.testing.assert_array_equal(core.hist[:, 0], [-1.0, 0.25, -0.5])
    off, rf = core.offsets(core.a_f)
    legs_ = plant.leg_joints(p.joints)
    np.testing.assert_array_equal(off, core.a_f[:-1] * np.where(legs_, 0.35, 0.15))  # rule E1's scales, by joint
    assert rf == 1.0 + 0.25 * core.a_f[-1]
    assert not np.any(off < -0.35) and (off[~legs_] != 0).all()


# ------------------------------------------------------------------------------------------------
# The reward on a kinematic playback (rule W1)
# ------------------------------------------------------------------------------------------------
IMITATION = ("joint", "foot", "contact", "head")


def _check_reference_scores(rng: dict, tracking_floor: float = 0.99) -> None:
    for k in IMITATION:
        assert rng[k][0] > 1.0 - 1e-6, (k, rng[k])
    assert rng["tracking"][0] > tracking_floor, rng["tracking"]
    for k in r5core.PENALTIES:
        assert rng[k][1] < 1e-3, (k, rng[k])


@needs_run
@pytest.mark.parametrize("task", ["brisk", "stop"])
def test_a_playback_scores_imitation_one_and_every_penalty_zero(p, s1, task):
    """Stage 1's start, two walks, both changes, the stop and a hold, with the real leg-gap oracle."""
    core = r5core.Core(p, {}, legs.LegGap(BODY))
    pb = playback(p, core, s1, task)
    rng = pb.run(pb.script.t_end / RATE - 0.02)
    _check_reference_scores(rng)
    assert rng["interference"] == (0.0, 0.0) and rng["range"] == (0.0, 0.0) and rng["clearance"] == (0.0, 0.0)
    assert rng["duty"] == (0.0, 0.0) and rng["progress"][1] < 1e-4
    stats = core.clock_stats()
    assert stats["mean"] == stats["min"] == 1.0 and stats["reference_s"] == pytest.approx(pb.c)


# ------------------------------------------------------------------------------------------------
# The readings the lead ruled on (16 Nov 2026): each scores the reference's playback 0, and each differs from the
# wording it replaced, which is computed here beside it
# ------------------------------------------------------------------------------------------------
def _band(target_speed: float) -> tuple[float, float]:
    return (0.55, 0.72) if target_speed < 0.30 else (0.55, 0.65)  # rule W3


@needs_run
@pytest.mark.parametrize("task", ["walk", "brisk"])
def test_duty_reads_a_foot_only_where_the_references_own_share_is_inside_the_band(p, core, s1, task):
    """On the reference's own playback (Stage 1's start, walks, changes and stop) the duty term is 0 at every env
    step.  Without the condition on the reference's own share (each foot's share over the last stride outside
    rule W3's band, over 0.05, while the reference walks, and no more) the same contacts are charged in the start
    and in the stop, where the reference's own share is outside the band.  The term equals the reading, computed
    here from the playback's loads and the reference's support flags alone."""
    pb = playback(p, core, s1, task)
    s, weight = pb.script, pb.weight
    down, ref, orig = [], [], pb.set

    def recording():
        orig()
        down.append(p.d.sensordata[p._load_adr] > 0.02 * weight)
        ref.append(s.support(pb.c))

    pb.set = recording
    shares, ref_shares, coded, unconditional, kinds = [], [], [], [], []
    for _ in range(int(round((s.t_end / RATE - 0.02) / 0.02))):
        down.clear()
        ref.clear()
        t = pb.step()
        shares.append(np.mean(down, axis=0))
        ref_shares.append(np.mean(ref, axis=0))
        n = max(1, int(round(s.stride_s(pb.c) / RATE / 0.02)))
        reading = without = 0.0
        if s.walking(pb.c) and len(shares) >= n:
            lo, hi = _band(s.target_speed(pb.c))
            d, d_ref = np.mean(shares[-n:], axis=0), np.mean(ref_shares[-n:], axis=0)
            over = np.maximum(0.0, np.maximum(lo - d, d - hi))
            without = min(1.0, float(over.sum()) / 0.05)
            reading = min(1.0, float(over[(d_ref >= lo) & (d_ref <= hi)].sum()) / 0.05)
        assert t["duty"] == pytest.approx(reading, abs=1e-12)
        coded.append(t["duty"])
        unconditional.append(without)
        kinds.append(s.kind(pb.c))
    assert max(coded) == 0.0  # rule W1: the reference scores zero
    charged = [k for k, x in zip(kinds, unconditional, strict=True) if x > 0.0]
    assert len(charged) > 60 and set(charged) == {"finite"} and max(unconditional) == 1.0
    assert not any(x > 0.0 for k, x in zip(kinds, unconditional, strict=True) if k == "periodic")


@needs_run
def test_progress_is_the_shortfall_against_the_references_own_mean_speed(p, core, s1):
    """The CoM of a walking reference speeds up and slows down inside every stride.  Against the constant ``rate x
    v_cmd`` the bias-corrected mean of the playback's own speed falls short by up to a quarter of the term's scale
    and more; against the same mean of the reference's own speed it does not.  With the clock held 5 % slow the
    term is the reading's formula, computed here from the played CoM and the reference's stride table alone."""
    s = s1.script("walk")
    seg = s.track.segments[1]
    v_cmd, tau = s.command(seg.t0 + 0.1)[0], seg.period / RATE
    a = 1.0 - math.exp(-0.02 / tau)
    for factor in (1.0, 0.95):
        pb = playback(p, core, s1, "walk", seg.t0)
        pb.factor = factor
        com0, c0 = p.d.subtree_com[p.trunk][:2].copy(), pb.c
        heading = np.array([math.cos(core.yaw_home), math.sin(core.yaw_home)])  # a straight walk: the home heading
        v_bar = v_ref_bar = elapsed = 0.0
        coded, constant, reading = [], [], []
        while pb.c < seg.t0 + seg.duration - 0.05:
            t = pb.step()
            com = p.d.subtree_com[p.trunk][:2].copy()
            v_bar += a * (float((com - com0) @ heading) / 0.02 - v_bar)
            v_ref_bar += a * (RATE * s.walk_progress(c0, pb.c) / (pb.c - c0) - v_ref_bar)
            com0, c0, elapsed = com, pb.c, elapsed + 0.02
            if pb.c - seg.t0 >= seg.period:  # one stride after the segment's first frame
                grown = 1.0 - math.exp(-elapsed / tau)
                coded.append(t["progress"])
                constant.append(min(1.0, max(0.0, 1.0 - v_bar / grown / (RATE * v_cmd)) / 0.10))
                reading.append(min(1.0, max(0.0, (v_ref_bar - v_bar) / grown / (RATE * v_cmd)) / 0.10))
            else:
                assert t["progress"] == 0.0
        assert len(coded) > 200
        np.testing.assert_allclose(coded, reading, rtol=0, atol=1e-9)
        if factor == 1.0:
            assert max(coded) < 1e-4 and max(constant) > 0.25  # rule W1 holds for the reading only
            assert 0.0 < max(coded)  # ... to below 1e-4, not to exactly 0: REWARD_RULE says "progress below 1e-4"
        else:
            assert 0.42 < min(coded) and max(coded) < 0.58  # 5 % short of the reference: half the term


def _gait_swing_s(lib, script, c: float) -> float:
    """One swing time per gait: (1 - duty) x the stride period of the gait of the segment in force at ``c`` (a
    move: the gait it leads to, else the one it leaves), from the set's records."""
    item = script.plan[script.segment_index(c)]
    if item[0] == "ref":
        e = lib.refs_doc[item[1]]
        return (1.0 - float(script.track.segments[script.segment_index(c)].duty)) * float(e["period_s"])
    if item[0] == "transition":
        g = lib.tr_doc[item[1]]["gaits"]
        g = g["b"] if math.isfinite(float(g["b"]["period"])) and float(g["b"]["period"]) > 0.0 else g["a"]
        return (1.0 - float(g["duty"])) * float(g["period"])
    return 0.0


@needs_run
def test_clearance_a_times_a_swing_by_the_references_swing_in_progress(p, core, s1):
    """Inside a change between two walks every swing has a length of its own.  On the reference's playback
    (Stage 1's brisk task: both walks and both changes) clearance is 0 at every env step.  With one swing time per
    gait the same lift-offs put a foot "between 25 % and 75 % of its swing" while the reference has it near the
    floor, and the reference is charged inside the two changes."""
    pb = playback(p, core, s1, "brisk")
    s, weight, need = pb.script, pb.weight, 0.04 * core.H
    tables = env.foot_tables(p.m, p.feet)
    lift, was, orig = [0.0, 0.0], [True, True], pb.set
    names = sorted({seg.name for seg in s.track.segments if seg.kind == "finite"})
    assert names == ["brisk_to_nominal", "nominal_to_brisk", "start", "stop"]

    def recording():  # each foot's own lift-off, on the reference clock
        orig()
        for f in (0, 1):
            now = bool(p.d.sensordata[p._load_adr[f]] > 0.02 * weight)
            if was[f] and not now:
                lift[f] = pb.c
            was[f] = now

    pb.set = recording
    coded, per_gait, where = [], [], []
    for _ in range(int(round((s.t_end / RATE - 0.02) / 0.02))):
        t = pb.step()
        c = pb.c
        low = metrics.foot_lowest(p.m, p.d, *tables) - p.z_floor
        charge = {}
        for label, swing in (("progress", s.swing_s(c)), ("gait", _gait_swing_s(s1, s, c))):
            charge[label] = max([min(1.0, max(0.0, need - float(low[f])) / need) for f in (0, 1)
                                 if not was[f] and swing > 0.0 and 0.25 <= (c - lift[f]) / swing <= 0.75] or [0.0])
        assert t["clearance"] == pytest.approx(charge["progress"], abs=1e-12)
        coded.append(t["clearance"])
        per_gait.append(charge["gait"])
        where.append(s.track.segments[s.segment_index(c)].name)
    assert max(coded) == 0.0  # rule W1: the reference scores zero
    charged = sorted({n for n, x in zip(where, per_gait, strict=True) if x > 0.0})
    assert charged == ["brisk_to_nominal", "nominal_to_brisk"] and max(per_gait) > 0.5
    assert sum(1 for x in per_gait if x > 0.0) >= 5


@needs_run
@pytest.mark.parametrize("direction", ["left", "right"])
def test_a_playback_of_a_turn_in_place_scores_heading_zero(p, core, t0, direction):
    """The lead's ruling of 16 Nov 2026: in the turn-in-place tasks the commanded heading follows the reference's
    own yaw, so that the reference itself scores zero on the heading penalty over the whole task (the start from
    standing, three strides of the turn, the stop), and the actor's heading input stays at zero.  The same plan
    read by rule E4's integral (as a plan, not as the task) charges its own reference for most of the task."""
    pytest.importorskip("designlab.stage2.bar")
    s = t0.script(f"inplace_{direction}")
    assert s.heading_rule == "reference" and [x[0] for x in s.plan] == ["transition", "ref", "transition"]
    n = int(round((s.t_end / RATE - 0.02) / 0.02))
    at = core.layout["blocks"]["heading"][0]
    pb = Playback(p, core, s, t0.dir, played=played_turning)
    inputs = []
    for _ in range(n):
        assert pb.step()["heading"] == 0.0
        inputs.append(float(core.actor_obs()[at]))
    assert s.t_end - 0.05 < pb.c <= s.t_end and max(abs(x) for x in inputs) < 1e-6  # the whole task
    turned = control._wrap(control._yaw(p.d.xmat[p.trunk]) - core.yaw_home)  # the pose really turned: 260 degrees
    assert abs(control._wrap(turned - s.ref_yaw(pb.c))) < 1e-6 and abs(s.ref_yaw(pb.c)) > math.radians(250.0)
    rows = pb.terms
    _check_reference_scores({k: (min(t[k] for t in rows), max(t[k] for t in rows)) for k in rows[0]},
                            tracking_floor=0.95)
    # rule E4's integral on the same plan: the reference is charged, and the input sits at its clip
    e4 = t0.from_plan(s.plan)
    assert e4.heading_rule == "command"
    pb = Playback(p, core, e4, t0.dir, played=played_turning)
    charged, inputs = [], []
    for _ in range(n):
        charged.append(pb.step()["heading"])
        inputs.append(float(core.actor_obs()[at]))
    assert max(charged) == 1.0 and float(np.mean(charged)) > 0.7 and float(np.mean(np.array(charged) == 1.0)) > 0.65
    assert float(np.mean(np.abs(inputs) == 2.0)) > 0.5
    # the same plan given as a plan with the heading named, as a row that plays a turn in place gives it
    # (``from_plan(plan, heading="reference")``): read as the task is
    named = t0.from_plan(s.plan, heading=r5lib.HEADING_REFERENCE)
    assert named is not s and named is not e4 and named.heading_rule == "reference"
    pb = Playback(p, core, named, t0.dir, played=played_turning)
    for _ in range(n):
        assert pb.step()["heading"] == 0.0 and abs(float(core.actor_obs()[at])) < 1e-6


@needs_run
def test_a_playback_from_a_mid_task_start_scores_the_same(p, core, s1):
    s = s1.script("brisk")
    for t in (steady_time(s, 0.37), 8.4, 14.9, 20.3):
        _check_reference_scores(playback(p, core, s1, "brisk", s.start_state(t)[0]).run(4.0))


@needs_run
def test_observation_blocks_on_a_playback(p, core, s1):
    pb = playback(p, core, s1, "brisk", t0=s1.script("brisk").start_state(14.0)[0])
    pb.run(0.5, action=np.linspace(-1.0, 1.0, core.nj + 1))
    s, c, lay = pb.script, pb.c, core.layout["blocks"]
    obs = core.actor_obs()
    assert obs.dtype == np.float32 and obs.shape == (88,)

    def blk(x, name):
        return x[lay[name][0] : lay[name][1]]

    np.testing.assert_allclose(blk(obs, "q"), p.d.qpos[p.qadr], atol=1e-6)
    ph = 2 * math.pi * s.phase(c)
    np.testing.assert_allclose(blk(obs, "phase"), [math.sin(ph), math.cos(ph)], atol=1e-6)
    np.testing.assert_array_equal(blk(obs, "support"), s.support(c).astype(np.float32))
    v, w = s.command(c)
    np.testing.assert_allclose(blk(obs, "command"), [RATE * v / 0.5, RATE * w / 1.0], rtol=1e-6)
    assert abs(blk(obs, "heading")[0]) < 1e-3
    np.testing.assert_allclose(blk(obs, "foot_load"), p.d.sensordata[p._load_adr] / core.weight, rtol=1e-6)
    np.testing.assert_allclose(blk(obs, "prev_actions"), np.tile(np.linspace(-1.0, 1.0, 15), 3), atol=1e-7)
    factors = dict(friction=None, mass=1.05, kp=0.8, latency_steps=6, noise_scale=1.0, joint_damping=1.3,
                   joint_friction=0.7)
    crit = np.concatenate([obs, core.critic_obs(factors, None, "robot_floor_carpet")])
    assert crit.shape == (130,)
    np.testing.assert_allclose(blk(crit, "joint_error"), 0.0, atol=1e-6)
    np.testing.assert_allclose(blk(crit, "foot_error"), 0.0, atol=1e-5)
    np.testing.assert_allclose(blk(crit, "plant_factors"), [1.05, 0.8, 6 * 0.002 / 0.024, 1.3, 0.7], rtol=1e-6)
    np.testing.assert_array_equal(blk(crit, "floor"), [0.0, 1.0])
    np.testing.assert_array_equal(blk(crit, "push"), [0.0, 0.0, 0.0])
    assert blk(crit, "friction")[0] == np.float32(0.9) and abs(blk(crit, "heading_error")[0]) < 1e-6
    np.testing.assert_array_equal(blk(crit, "ref_contact"), s.support(c - 0.024 * RATE).astype(np.float32))
    assert blk(crit, "height")[0] == pytest.approx((p.d.qpos[2] - p.z_floor) / core.H, rel=1e-6)
    hard = core.critic_obs(dict(factors, friction=0.5), (0.0, 99.0, 12.0, -6.0), "hard")
    a = lay["friction"][0] - 88
    assert hard[a] == np.float32(0.5) and hard[a + 1 : a + 3].tolist() == [1.0, 0.0]
    mass = core.weight / 9.81
    np.testing.assert_allclose(hard[lay["push"][0] - 88 : lay["push"][1] - 88], [12.0 / mass, -6.0 / mass, 1.0],
                               rtol=1e-6)
    # a heading error of 0.2 rad reads 0.4; beyond 1 rad it is clipped at 2
    pb.yaw = 0.2
    pb.set()
    assert core.actor_obs()[lay["heading"][0]] == pytest.approx(0.4, abs=1e-3)
    pb.yaw = -1.5
    pb.set()
    assert core.actor_obs()[lay["heading"][0]] == -2.0


# ------------------------------------------------------------------------------------------------
# Each term moves when its cause is injected
# ------------------------------------------------------------------------------------------------
def _jq(p, name: str) -> int:
    return int(p.qadr[p.joints.index(name)])


@needs_run
def test_imitation_terms_move_with_their_cause(p, core, s1):
    t0 = s1.script("walk").start_state(steady_time(s1.script("walk")))[0]
    # a foot offset (a knee 0.15 rad off): the foot term and the joint term fall, the head's stays
    pb = playback(p, core, s1, "walk", t0)
    pb.pose = lambda q, c: _offset(q, _jq(p, "left_knee"), 0.15)
    rng = pb.run(1.0)
    assert rng["foot"][1] < 0.9 and rng["joint"][1] == pytest.approx(math.exp(-(0.15 / 0.1) ** 2 / 14), abs=1e-6)
    assert rng["head"][0] > 1.0 - 1e-6
    # a head pitch of 0.1 rad: exp(-(0.1 rad / 3 deg)^2), the feet's term untouched
    pb = playback(p, core, s1, "walk", t0)
    pb.pose = lambda q, c: _offset(q, _jq(p, "neck_pitch"), 0.1)
    rng = pb.run(1.0)
    assert rng["head"][1] == pytest.approx(math.exp(-(0.1 / math.radians(3.0)) ** 2), abs=2e-3)
    assert rng["foot"][0] > 1.0 - 1e-6
    # contacts late by 20 ms x the clock rate: inside the 24 ms allowance; by 60 ms: charged at every contact change
    pb = playback(p, core, s1, "walk", t0)
    pb.load_lag = 0.020 * RATE
    assert pb.run(2.5)["contact"] == (1.0, 1.0)
    pb = playback(p, core, s1, "walk", t0)
    pb.load_lag = 0.060 * RATE
    rng = pb.run(2.5)
    assert rng["contact"][0] < 0.95 and rng["contact"][1] == 1.0


def _offset(q, adr: int, by: float):
    q = np.array(q, float)
    q[adr] += by
    return q


@needs_run
def test_servo_term_reads_a_tail_servo_over_its_rating_from_the_first_step(p, core, s1):
    tail = next(i for i, d in enumerate(p.drives) if d.group == "tail")
    assert tail in core._servo and len(core._servo) == 2  # the neck and the tail: every drive that is no leg drive
    sustained = p.drives[tail].sustained
    for ratio, want in ((2.0, 1.0), (1.25, 0.5), (1.0, 0.0), (0.5, 0.0)):
        pb = playback(p, core, s1, "walk")
        pb.force = {tail: ratio * sustained}
        assert pb.step()["servo"] == pytest.approx(want, abs=1e-9)  # bias-corrected: the mean so far, at once
        assert pb.run(0.4)["servo"] == pytest.approx((want, want), abs=1e-9)
    # a leg drive far over its rating is not this term's
    pb = playback(p, core, s1, "walk")
    leg = next(i for i, d in enumerate(p.drives) if d.group not in ("neck", "tail"))
    pb.force = {leg: 50.0 * p.drives[leg].sustained}
    assert pb.step()["servo"] == 0.0


class _Gaps:
    """A stand-in for ``legs.LegGap``: a fixed pair and a moving pair at given distances."""

    cap = 0.025
    moving = np.array([False, True])

    def __init__(self, fixed: float, moving: float):
        self.d = np.array([fixed, moving])

    def at(self, qpos):
        return self.d.copy()


@needs_run
def test_interference_reads_the_moving_pairs_for_the_ramp_and_all_pairs_below_zero(p, s1):
    cases = (((0.0029, 0.002), (1.0 / 3.0, False)),  # a fixed pair at 2.9 mm adds nothing
             ((0.0029, 0.025), (0.0, False)), ((0.0029, 0.003), (0.0, False)), ((0.0040, 0.0015), (0.5, False)),
             ((0.0040, 0.0), (1.0, False)), ((0.0040, -0.001), (1.0, True)), ((-0.0005, 0.02), (0.0, True)))
    for gaps, (pen, below) in cases:
        core = r5core.Core(p, {}, _Gaps(*gaps))
        pb = playback(p, core, s1, "walk")
        got = core.interference()
        assert got == (pytest.approx(pen, abs=1e-9), below)
        assert pb.step()["interference"] == pytest.approx(pen, abs=1e-9)
    assert r5core.Core(p, {}, None).interference() == (0.0, False)
    assert r5core.REWARD_SCALES["interference_m"] == legs.clearance.TIGHT


@needs_run
def test_clearance_and_contact_charge_a_foot_loaded_through_the_references_mid_swing(p, core, s1):
    s = s1.script("walk")
    pb = playback(p, core, s1, "walk", s.start_state(steady_time(s))[0])
    pb.loads = lambda c: np.array([0.5 * pb.weight, 0.5 * pb.weight])  # both feet down, always: a dragged foot
    rows = []
    for _ in range(150):
        t = pb.step()
        both = any(s.mid_swing(pb.c) & s.mid_swing(pb.c - 0.024 * RATE))  # ... at the clock and 24 ms x 1.05 earlier
        rows.append((t["clearance"], both, any(s.mid_swing(pb.c)), t["contact"]))
    assert sum(1 for r in rows if r[1]) > 30 and all(r[0] == 1.0 for r in rows if r[1])
    assert all(r[0] == 0.0 for r in rows if not r[2] and not any(s.mid_swing(pb.c)))  # (b) only: the foot is never up
    assert any(r[0] == 0.0 for r in rows) and min(r[3] for r in rows) < 0.6
    # (a): a foot in the air through the middle of its own swing, but too low: the shortfall over 0.04 H
    need = 0.04 * core.H
    foot = p.feet[0]
    pb = playback(p, core, s1, "walk", s.start_state(steady_time(s))[0])
    low = []
    orig = pb.set

    def squat():  # the played pose lowered so that the swinging foot skims the floor
        orig()
        if not s.support(pb.c)[0]:
            lowest = env.metrics.foot_lowest(p.m, p.d, *core._foot_tables)[0] - p.z_floor
            drop = lowest - 0.25 * need
            for g in np.flatnonzero(p.m.geom_bodyid == foot):
                p.d.geom_xpos[g, 2] -= drop
            low.append(pb.c)

    pb.set = squat
    rng = pb.run(2.5)
    assert low and rng["clearance"][1] == pytest.approx(0.75, abs=1e-6) and rng["clearance"][0] == 0.0


@needs_run
def test_progress_charges_a_slow_clock_in_a_steady_walk(p, core, s1):
    s = s1.script("walk")
    seg = s.track.segments[1]
    for factor, lo, hi in ((0.90, 0.93, 1.0), (0.95, 0.42, 0.58), (1.0, 0.0, 1e-4), (1.1, 0.0, 0.0)):
        pb = playback(p, core, s1, "walk", seg.t0)
        pb.factor = factor
        vals, before = [], []
        while pb.c < seg.t0 + 5.5 * seg.period:
            t = pb.step()
            (vals if pb.c >= seg.t0 + 1.2 * seg.period else before).append(t["progress"])
        assert lo <= min(vals) and max(vals) <= hi, (factor, min(vals), max(vals))
        assert max(before[: int(0.9 * seg.period / RATE / 0.02)]) == 0.0  # not read before one stride
    # a body that stands still while the reference walks is charged in full; nothing in the start or in a stop
    pb = playback(p, core, s1, "walk", seg.t0)
    pb.pose = lambda q, c: pb.play.q(seg.t0)
    assert pb.run(3.0)["progress"][1] == 1.0
    pb = playback(p, core, s1, "walk")
    pb.factor = 0.8
    assert pb.run(4.5)["progress"] == (0.0, 0.0)


@needs_run
def test_heading_term_and_its_free_band(p, core, s1):
    s = s1.script("stop")
    for deg, want in ((10.0, 7.0 / 17.0), (2.9, 0.0), (20.0, 1.0), (-11.5, 0.5), (45.0, 1.0)):
        pb = playback(p, core, s1, "stop", s.start_state(steady_time(s, 0.4))[0])
        pb.yaw = math.radians(deg)
        rng = pb.run(0.6)
        assert rng["heading"] == pytest.approx((want, want), abs=2e-3), (deg, rng["heading"])
    hold = next(seg for seg in s.track.segments if seg.kind == "hold")
    pb = playback(p, core, s1, "stop")
    pb.c = hold.t0 + 0.5
    pb.yaw = math.radians(30.0)
    assert pb.run(0.4)["heading"] == (0.0, 0.0)  # zero in a hold
    # without the heading input the term is struck: its weight is zero and the reward does not move
    no = r5core.Core(p, dict(heading_input=False), None)
    assert no.weights["heading"] == 0.0 and no.layout["actor_dim"] == 87
    assert no.weights == {**r5core.REWARD_WEIGHTS, "heading": 0.0}


@needs_run
def test_range_term_reads_the_required_margin(p, core, s1):
    s = s1.script("walk")
    j = p.joints.index("right_hip_yaw")
    need = float(s.required_margin[j])
    assert need == 0.1
    cases = ((need + 0.01, 0.0), (need, 0.0), (need - 0.01, 0.5), (need - 0.02, 1.0), (0.0, 1.0), (-0.05, 1.0))
    for margin, want in cases:
        pb = playback(p, core, s1, "walk")
        pb.pose = lambda q, c, m=margin: _set(q, int(p.qadr[j]), core.hi[j] - m)
        assert pb.step()["range"] == pytest.approx(want, abs=1e-9)
        pb.pose = lambda q, c, m=margin: _set(q, int(p.qadr[j]), core.lo[j] + m)
        assert pb.step()["range"] == pytest.approx(want, abs=1e-9)


def _set(q, adr: int, value: float):
    q = np.array(q, float)
    q[adr] = value
    return q


@needs_run
def test_stage_1s_penalties_and_the_action_rate(p, core, s1):
    s = s1.script("walk")
    t0 = s.start_state(steady_time(s))[0]
    w = float(p.m.body_subtreemass[p.trunk]) * 9.81
    # flight: neither foot loaded; impact: 4 body weights on the floor
    pb = playback(p, core, s1, "walk", t0)
    pb.loads = lambda c: np.zeros(2)
    assert pb.step()["flight"] == 1.0
    pb = playback(p, core, s1, "walk", t0)
    pb.loads = lambda c: np.array([2.0 * w, 2.0 * w])
    t = pb.step()
    assert t["impact"] == pytest.approx(1.0) and t["flight"] == 0.0
    pb.loads = lambda c: np.array([1.2 * w, 1.2 * w])
    assert pb.step()["impact"] == pytest.approx((2.4 - 1.6) / 1.6)
    # tilt: a pitch of 6 degrees about home is half the term, 8 degrees all of it
    for deg, want in ((6.0, 0.5), (8.0, 1.0), (3.9, 0.0)):
        pb = playback(p, core, s1, "walk", t0)
        pb.pose = lambda q, c, a=math.radians(deg): _pitched(q, a)
        assert pb.step()["tilt"] == pytest.approx(want, abs=2e-2)
    # duty: both feet down for a stride of a steady walk; nothing while the reference itself stands or starts
    pb = playback(p, core, s1, "walk", t0)
    pb.loads = lambda c: np.array([0.5 * w, 0.5 * w])
    rng = pb.run(1.0)
    assert rng["duty"] == (0.0, 0.0)  # (no whole stride of history yet)
    assert pb.run(1.0)["duty"][1] == 1.0
    pb = playback(p, core, s1, "walk")
    pb.loads = lambda c: np.array([0.5 * w, 0.5 * w])
    assert pb.run(2.1)["duty"] == (0.0, 0.0)  # before the start's first lift-off its own shares are 1: not read
    assert pb.run(2.9)["duty"][1] == 1.0  # ... and once the reference steps, a body that does not is charged
    # action rate: min(1, mean_i (a_i - a_prev_i)^2 / 0.25) on the clipped actions
    pb = playback(p, core, s1, "walk", t0)
    n = core.nj + 1
    assert pb.step(np.full(n, 0.1))["action_rate"] == pytest.approx(0.01 / 0.25)
    assert pb.step(np.full(n, 0.1))["action_rate"] == 0.0
    assert pb.step(np.full(n, 5.0))["action_rate"] == 1.0  # clipped to 1: (0.9)^2 / 0.25, capped
    one = np.zeros(n)
    one[0] = 1.0
    pb.step(np.zeros(n))
    assert pb.step(one)["action_rate"] == pytest.approx(1.0 / n / 0.25)


@needs_run
def test_a_rung_at_0_30_m_s_keeps_the_usual_duty_band(p, core, t0):
    """Rule W3: the band is 0.55-0.72 while the target speed is below 0.30 m/s.  A rung named ``v0.3`` whose entry
    holds one ulp less than 0.30 is not below it (the bar takes the target from the name): a body whose feet are
    each down for 0.69 of the stride is charged under 0.55-0.65.  The same arrays as the nominal walk (0.243 m/s)
    are read under 0.55-0.72 and not charged."""
    w = float(p.m.body_subtreemass[p.trunk]) * 9.81
    assert t0.refs_doc["v0.3"]["speed_m_s"] == V03_BELOW < 0.3 == r5core.REWARD_SCALES["duty_slow_below_m_s"]
    for name, target, want in (("v0.3", 0.3, 1.0), ("nominal", t0.refs_doc["nominal"]["speed_m_s"], 0.0)):
        s = t0.from_plan([("transition", "start"), ("ref", name, 8)])
        seg = s.track.segments[1]
        pb = Playback(p, core, s, t0.dir, s.start_state(seg.t0 + 0.5 * seg.period)[0])
        # each foot in the air for the first 0.31 of the reference's own swing phase: a share of 0.69
        pb.loads = lambda c, s=s: np.array([0.0 if s.phase(c) < 0.31 else 0.5 * w,
                                            0.0 if 0.5 <= s.phase(c) < 0.81 else 0.5 * w])
        assert pb.run(1.0)["duty"] == (0.0, 0.0)  # (no whole stride of history yet)
        assert pb.run(2.0)["duty"][1] == want
        assert s.target_speed(pb.c) == target and s.command(pb.c)[0] == t0.refs_doc[name]["speed_m_s"]
        n = int(round(seg.period / RATE / 0.02))
        share = np.mean(core._shares[np.arange(core._shares_n - n, core._shares_n) % r5core.HISTORY_STEPS], axis=0)
        assert np.all(np.abs(share[:2] - 0.69) < 0.01) and np.all(np.abs(share[2:] - 0.60) < 0.02)


@needs_run
def test_dutys_stride_is_kept_between_1_and_256_env_steps(p, core, s1, monkeypatch):
    """``REWARD_RULE``: duty's n is ``round(the stride period / the clock rate / 20 ms)``, kept between 1 and 256
    env steps (``HISTORY_STEPS``: the history the core keeps).  With a stride period that would give 476 env steps
    the term is read from the 256th env step on, over the last 256; with one that would give none it is read from
    the first env step on, over that one step."""
    s = s1.script("walk")
    seg = s.track.segments[1]
    w = float(p.m.body_subtreemass[p.trunk]) * 9.81
    t0 = s.start_state(seg.t0 + 0.5 * seg.period)[0]
    assert seg.kind == "periodic" and seg.duration / RATE > 257 * 0.02 + 0.5 * seg.period
    # (the upper bound) a stride of 10 s: round(10 / 1.05 / 0.02) = 476 env steps, kept at 256
    monkeypatch.setattr(s, "stride_s", lambda c: 10.0)
    assert int(round(10.0 / RATE / 0.02)) == 476 > r5core.HISTORY_STEPS == 256
    pb = playback(p, core, s1, "walk", t0)
    pb.loads = lambda c: np.array([0.5 * w, 0.5 * w])  # both feet down, always
    read = [pb.step()["duty"] for _ in range(257)]
    assert read[:255] == [0.0] * 255 and read[255] == read[256] == 1.0  # read from the 256th env step, not the 476th
    share = np.mean(core._shares[np.arange(core._shares_n - 256, core._shares_n) % 256], axis=0)
    assert share[:2].tolist() == [1.0, 1.0] and np.all((share[2:] > 0.55) & (share[2:] < 0.65))
    # (the lower bound) a stride of 1 ms: round(0.001 / 1.05 / 0.02) = 0 env steps, kept at 1
    monkeypatch.setattr(s, "stride_s", lambda c: 0.001)
    assert int(round(0.001 / RATE / 0.02)) == 0
    pb = playback(p, core, s1, "walk", t0)
    pb.loads = lambda c: np.array([0.5 * w, 0.5 * w])
    ref, orig = [], pb.set

    def recording():
        orig()
        ref.append(s.support(pb.c))

    pb.set = recording
    coded, want = [], []
    for _ in range(int((seg.t0 + seg.duration - t0) / RATE / 0.02) - 2):  # to the end of the periodic walk
        ref.clear()
        coded.append(pb.step()["duty"])
        own = np.mean(ref, axis=0)  # the reference's own support share over this one env step
        want.append(min(1.0, float(np.sum(np.where((own >= 0.55) & (own <= 0.72), 1.0 - 0.72, 0.0))) / 0.05))
    assert len(coded) > 300 and coded == want and max(coded) == 1.0 and 0 < sum(1 for x in coded if x > 0.0) < 30
    assert s.target_speed(pb.c) < 0.30  # (Stage 1's nominal walk: the band is 0.55-0.72)


@needs_run
def test_clearance_a_is_not_read_without_a_swing_length(p, core, s1, monkeypatch):
    """``REWARD_RULE``: where T, the planned length of the reference's swing, is not above zero, clearance (a) is
    not read (no division by it).  A foot that skims the floor through the middle of its own swing is charged 0.75
    with the reference's swing length and nothing where the script gives none; (b) is read all the same."""
    s = s1.script("walk")
    need = 0.04 * core.H
    foot = p.feet[0]
    w = float(p.m.body_subtreemass[p.trunk]) * 9.81
    t0 = s.start_state(steady_time(s))[0]

    def skimming(swing):
        if swing is not None:
            monkeypatch.setattr(s, "swing_s", lambda c: swing)
        pb = playback(p, core, s1, "walk", t0)
        orig = pb.set

        def squat():  # the played pose lowered so that the left foot skims the floor while it swings
            orig()
            if not s.support(pb.c)[0]:
                lowest = metrics.foot_lowest(p.m, p.d, *core._foot_tables)[0] - p.z_floor
                for g in np.flatnonzero(p.m.geom_bodyid == foot):
                    p.d.geom_xpos[g, 2] -= lowest - 0.25 * need

        pb.set = squat
        return pb

    assert skimming(None).run(2.5)["clearance"] == (0.0, pytest.approx(0.75, abs=1e-6))
    for swing in (0.0, -0.3):
        assert skimming(swing).run(2.5)["clearance"] == (0.0, 0.0)  # (a) is not read; nothing divides by T
    pb = skimming(0.0)
    pb.loads = lambda c: np.array([0.5 * w, 0.5 * w])  # a foot loaded through the reference's mid-swing: (b)
    assert pb.run(2.5)["clearance"] == (0.0, 1.0)


def _pitched(q, angle: float):
    q = np.array(q, float)
    w, x, y, z = q[3:7]
    c, s = math.cos(0.5 * angle), math.sin(0.5 * angle)
    q[3:7] = (w * c - y * s, x * c + z * s, y * c + w * s, z * c - x * s)  # q x (rotation about the body's y)
    return q


@needs_run
def test_touchdown_term_reads_the_foots_speed_one_step_before_contact(p, core, s1):
    hold = next(seg for seg in s1.script("stop").track.segments if seg.kind == "hold")
    w = float(p.m.body_subtreemass[p.trunk]) * 9.81
    foot, adr = p.feet[0], p._load_adr[0]
    for per_step, want in ((0.0010, 1.0), (0.0005, (0.25 - 0.15) / 0.30), (0.0003, 0.0)):
        pb = playback(p, core, s1, "stop")
        pb.c = hold.t0 + 0.3
        pb.set()
        d = p.d
        d.sensordata[adr] = 0.0
        core.substep()  # the foot leaves the floor
        for _ in range(4):
            d.xpos[foot, 2] -= per_step  # coming down at per_step / 2 ms
            core.substep()
        d.xpos[foot, 2] -= 0.0  # it stops at the floor in the step its contact begins
        d.sensordata[adr] = 0.5 * w
        core.substep()
        assert core._td == pytest.approx(want, abs=1e-9)
        core.interference()
        assert core.reward(pb.c)[1]["touchdown"] == pytest.approx(want, abs=1e-9)
        assert core._td == 0.0  # read once per env step


# ------------------------------------------------------------------------------------------------
# R6: the span rule, the tracker under control.run
# ------------------------------------------------------------------------------------------------
@needs_run
def test_r6_span_targets_and_tracker(p, s1):
    tracks = [s1.script(t).track for t in ("stand", "walk", "brisk", "stop")]
    span = r5core.policy_span(p, tracks)
    q_home = p.q_settled[p.qadr]
    ex = np.max([np.max(np.abs(seg.S[:, :14] - q_home), axis=0) for tr in tracks for seg in tr.segments], axis=0)
    np.testing.assert_array_equal(span, np.maximum(1.25 * ex, 0.10))
    core = r5core.Core(p, dict(span=span), None)
    pb = playback(p, core, s1, "walk")
    q, rf = core.targets(np.zeros(15))
    np.testing.assert_array_equal(q, q_home)
    assert rf == 1.0
    a = np.r_[np.full(14, 0.5), -1.0]
    q, rf = core.targets(a)
    np.testing.assert_array_equal(q, np.clip(q_home + 0.5 * span, core.lo, core.hi))
    assert rf == 0.75 and np.all(core.targets(np.r_[np.full(14, 9.0), 0.0])[0] <= core.hi)
    # the actions that give the reference's joint targets at the three env steps before a start (design D8)
    t = pb.script.start_state(8.3)[0]
    acts = core.reference_actions(t)
    assert acts.shape == (3, 15) and not acts[:, -1].any() and np.max(np.abs(acts)) <= 0.8 + 1e-12
    for i in range(3):
        want = pb.script.track.joint(t - (i + 1) * 0.02 * RATE)[:14]
        np.testing.assert_allclose(q_home + acts[i, :14] * span, want, atol=1e-12)
    core.set_history(acts, acts[0])
    np.testing.assert_array_equal(core.hist, acts)
    np.testing.assert_allclose(core.targets(core.a_f)[0], pb.script.track.joint(t - 0.02 * RATE)[:14], atol=1e-12)
    # the tracker runs under control.run as any tracker does and holds what it is told
    tk = r5core.R6Tracker(p, s1.script("stand").track, RATE, model=None)
    out = control.run(tk, 0.3, start="stance", seed=1, noise_seed=2, joint_noise=0.01, root_velocity_noise=0.02)
    assert not out["fell"] and out["seconds"] == pytest.approx(0.3) and tk.tau_ref == pytest.approx(0.3 * RATE)
    np.testing.assert_array_equal(tk.q_target, q_home)
    assert tk.est is not None and r5core.tracker_estimate(tk)["q"].shape == (14,)
    with pytest.raises(ValueError, match="stance"):
        tk.reset("frame")


# ------------------------------------------------------------------------------------------------
# 16 joints: by name
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def px():
    return plant.build(BODY_X, "hard")


@needs_x
def test_x_core_addresses_joints_by_name(px):
    lib = r5lib.Library(px, BODY_X, "S1", RATE)
    core = r5core.Core(px, {}, None)
    assert px.joints == J16 and core.layout["blocks"] == LAYOUT_16
    assert (core.i_yaw, core.i_jaw) == (0, 2) and len(core._servo) == 4  # neck yaw, neck, jaw, tail
    np.testing.assert_array_equal(core.scale, [0.15] * 4 + [0.35] * 12)  # rule E1: the legs from index 4
    head = r5core.HeadScript([("gaze", 0.0, 1.0, 5.0, 0.4), ("jaw", 0.5, 0.5, 1.0, 0.3)])
    pb = playback(px, core, lib, "walk", head=head)
    # the overlay is added to the reference's neck-yaw and jaw targets; the played pose follows it
    pb.pose = lambda q, c: _offset(_offset(q, _jq(px, "neck_yaw"), head.value(px.d.time)[0]), _jq(px, "jaw"),
                                   head.value(px.d.time)[1])
    off, rf = core.offsets(core.filter(np.zeros(17)))
    assert off is None and rf == 1.0  # at rest at the first instant: R0L's targets exactly
    rng = pb.run(2.0)
    assert rng["joint"][0] > 1.0 - 1e-4 and px.d.time == pytest.approx(2.0)
    yaw, jaw, yaw_goal, jaw_goal = head.value(px.d.time)
    off, _ = core.offsets(core.filter(np.zeros(17)))
    assert np.flatnonzero(off).tolist() == [0, 2] and (off[0], off[2]) == (yaw, jaw) and yaw == 0.4 and jaw > 0.0
    obs = core.actor_obs()
    assert obs.shape == (102,)
    np.testing.assert_allclose(obs[98:102], np.array([yaw, jaw, yaw_goal, jaw_goal]) / 0.6, rtol=1e-6)
    crit = core.critic_obs(dict(friction=None), None, "hard")
    np.testing.assert_allclose(crit[110 - 102 : 126 - 102], 0.0, atol=1e-5)  # joint errors with the overlay
    # head neutral: the same playback without a script scores as the plain body's does
    pb = playback(px, core, lib, "walk")
    rng = pb.run(8.0)
    _check_reference_scores(rng)
    assert core.actor_obs()[98:102].tolist() == [0.0] * 4
    with pytest.raises(ValueError, match="neck yaw"):
        r5core.Core(plant.build(BODY, "hard"), {}, None).begin(lambda: 0.0, pb.est, lib.script("walk"), head)


def _script_extremes(h) -> tuple[float, float, float]:
    """(the lowest and the highest neck-yaw target, the widest jaw opening) a head script commands."""
    gaze = [e[4] for e in h.events if e[0] == "gaze"] or [0.0]
    return min(gaze), max(gaze), max([e[4] for e in h.events if e[0] == "jaw"] or [0.0])


@pytest.mark.parametrize("body", HEAD_BODIES)
def test_no_scripted_head_target_lies_within_0_15_rad_of_a_joint_limit(body, monkeypatch):
    """The lead's ruling of 16 Nov 2026: scripted head targets keep 0.15 rad from a joint limit; with the jaw's
    range of 0 to 0.6 rad a script opens it to at most 0.45 rad.  On every body with a head: the bar's script H1
    over 300 episode seeds, training's draws, and every value on the way between two targets."""
    px = plant.build(body, "hard")
    core = r5core.Core(px, {}, None)
    i_yaw, i_jaw = px.joints.index("neck_yaw"), px.joints.index("jaw")
    lo, hi, need = core.lo, core.hi, r5core.HEAD_TARGET_MARGIN_RAD
    assert (lo[i_yaw], hi[i_yaw], lo[i_jaw], hi[i_jaw]) == (-0.6, 0.6, 0.0, 0.6) and need == 0.15
    assert core.rest[i_jaw] == -1 and core.rest[i_yaw] == 0  # closed is the jaw's rest end: no limit there
    room = r5core.head_target_room(px.joints, lo, hi)
    assert room == dict(neck_yaw=pytest.approx(0.15, abs=1e-12), jaw=pytest.approx(0.15, abs=1e-12))
    rng = np.random.default_rng(0)
    scripts = [r5core.HeadScript.fixed(seed, 30.0) for seed in range(3042, 3342)]
    scripts += [r5core.HeadScript.drawn(rng, 30.0, 1.0, 1.0) for _ in range(300)]
    low, high, wide = (f(x) for f, x in zip((min, max, max), zip(*map(_script_extremes, scripts), strict=True),
                                            strict=True))
    assert low - lo[i_yaw] >= need - 1e-12 and hi[i_yaw] - high >= need - 1e-12 and hi[i_jaw] - wide >= need - 1e-12
    assert low < -0.44 and high > 0.44 and 0.44 < wide <= 0.45  # (the draws reach the ends of their ranges)
    for h in scripts[:20] + scripts[300:320]:  # a minimum-jerk move never passes its target
        val = np.array([h.value(float(t)) for t in np.arange(0.0, 30.0, 0.01)])
        assert hi[i_yaw] - np.max(np.abs(val[:, [0, 2]])) >= need - 1e-12
        assert hi[i_jaw] - np.max(val[:, [1, 3]]) >= need - 1e-12 and np.min(val[:, [1, 3]]) >= 0.0
    # the script as drafted (a jaw opened to 0.6 rad, a gaze to +-0.5 rad) is refused when a core is built
    monkeypatch.setattr(r5core, "JAW_OPEN_RAD", (0.3, 0.6))
    with pytest.raises(plant.PlantError, match="jaw 0.0000 rad"):
        r5core.Core(px, {}, None)
    monkeypatch.setattr(r5core, "JAW_OPEN_RAD", (0.3, 0.45))
    monkeypatch.setattr(r5core, "GAZE_YAW_RAD", 0.5)
    with pytest.raises(plant.PlantError, match="neck_yaw 0.1000 rad"):
        r5core.Core(px, {}, None)


@pytest.mark.parametrize("body", HEAD_BODIES)
def test_a_playback_obeying_the_head_command_scores_range_zero(body):
    """A body whose neck yaw and jaw sit exactly on the head command, at the script's extremes (a gaze held at
    either end, the jaw opened as far as a script opens it), is charged nothing by the ``range`` term and keeps
    the 0.10 rad of joint range the bar asks of a walk.  A jaw opened to 0.6 rad, as the script was drafted, sits
    on its limit and is charged in full."""
    pytest.importorskip("designlab.stage2.bar")
    px = plant.build(body, "hard")
    lib = r5lib.Library(px, body, "S1", RATE)
    core = r5core.Core(px, {}, None)
    i_yaw, i_jaw = core.i_yaw, core.i_jaw
    script = lib.script("walk")
    assert script.required_margin[i_yaw] == script.required_margin[i_jaw] == tune.WALK_RANGE_RAD == 0.10

    def obeying(head, seconds: float) -> tuple[dict, np.ndarray]:
        pb = playback(px, core, lib, "walk", head=head)
        pb.pose = lambda q, c: _offset(_offset(q, _jq(px, "neck_yaw"), head.value(px.d.time)[0]), _jq(px, "jaw"),
                                       head.value(px.d.time)[1])
        played, terms = [], []
        for _ in range(int(round(seconds / 0.02))):
            terms.append(pb.step())
            played.append(px.d.qpos[px.qadr].copy())
        margins = metrics.joint_margins(np.array(played), core.lo, core.hi, core.rest)
        return {k: (min(t[k] for t in terms), max(t[k] for t in terms)) for k in terms[0]}, margins

    for yaw in (r5core.GAZE_YAW_RAD, -r5core.GAZE_YAW_RAD):
        head = r5core.HeadScript([("gaze", 0.0, 0.8, 6.0, yaw), ("jaw", 1.0, 0.4, 1.5, r5core.JAW_OPEN_RAD[1])])
        rng, margins = obeying(head, 4.0)
        assert rng["range"] == (0.0, 0.0) and rng["joint"][0] > 1.0 - 1e-4
        assert margins[i_yaw] == pytest.approx(0.15, abs=1e-9) and margins[i_jaw] == pytest.approx(0.15, abs=1e-9)
        assert np.all(margins >= script.required_margin - 1e-12)  # every joint keeps what the bar asks
    rng, margins = obeying(r5core.HeadScript.fixed(3042, 12.0), 12.0)  # the bar's own script for one seed
    assert rng["range"] == (0.0, 0.0) and min(margins[i_yaw], margins[i_jaw]) >= 0.15 - 1e-9
    drafted = r5core.HeadScript([("jaw", 1.0, 0.4, 1.5, 0.6)])
    rng, margins = obeying(drafted, 3.0)
    assert rng["range"] == (0.0, 1.0) and margins[i_jaw] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("body", HEAD_BODIES)
def test_in_r6_the_head_overlay_is_added_to_the_settled_home_pose(body):
    """``HEAD_RULE``: in R5 the head command is added to the reference's neck-yaw and jaw targets; in R6 it is added
    to the settled home pose's angles (``Core.targets``).  The jaw settles a little open, so an R6 jaw target lies
    that much above the script's value: 0.4515 rad on ``75-M-H-10-X`` and 0.4507 rad on ``60-N-T-10-X`` for the
    widest opening a script commands, which the rule string states."""
    pytest.importorskip("designlab.stage2.bar")
    px = plant.build(body, "hard")
    lib = r5lib.Library(px, body, "S1", RATE)
    span = np.full(16, 0.2)
    core = r5core.Core(px, dict(span=span), None)
    i_yaw, i_jaw = core.i_yaw, core.i_jaw
    home = px.q_settled[px.qadr]
    assert abs(home[i_yaw]) < 1e-7 and 0.0005 < home[i_jaw] < 0.002  # the jaw rests a little open on its closed end
    wide, yaw = r5core.JAW_OPEN_RAD[1], -r5core.GAZE_YAW_RAD
    head = r5core.HeadScript([("gaze", 0.0, 0.8, 6.0, yaw), ("jaw", 0.5, 0.4, 1.5, wide)])
    playback(px, core, lib, "walk", head=head)
    px.d.time = 1.5  # both moves done, both held
    assert head.value(1.5) == (yaw, wide, yaw, wide)
    zero = np.zeros(17)
    q, rf = core.targets(zeros := zero.copy())  # R6: the settled home pose, the overlay, no action
    assert rf == 1.0 and not zeros.any()
    np.testing.assert_array_equal(np.delete(q, [i_yaw, i_jaw]), np.delete(home, [i_yaw, i_jaw]))
    assert q[i_yaw] == home[i_yaw] + yaw and q[i_jaw] == home[i_jaw] + wide > wide
    off, _ = core.offsets(zero.copy())  # R5: the offsets the tracker adds to the reference's targets
    assert np.flatnonzero(off).tolist() == [i_yaw, i_jaw] and (off[i_yaw], off[i_jaw]) == (yaw, wide)
    # the numbers the rule string gives, and what they leave to the jaw's limit
    said = {"75-M-H-10-X": ("settles 0.0015 rad open on 75-M-H-10-X", "0.4515"),
            "60-N-T-10-X": ("0.0007 rad on 60-N-T-10-X", "0.4507")}[body]
    rule = " ".join(r5core.HEAD_RULE.split())
    assert said[0] in rule and f"{home[i_jaw]:.4f}" in said[0] and f"{q[i_jaw]:.4f}" == said[1] and said[1] in rule
    assert core.hi[i_jaw] - q[i_jaw] == pytest.approx(0.15 - home[i_jaw], abs=1e-12)
    assert 0.148 < core.hi[i_jaw] - q[i_jaw] < 0.15  # just under the 0.15 rad a script's own value keeps
    assert core.hi[i_jaw] - q[i_jaw] > tune.WALK_RANGE_RAD  # ... and well over what the range term asks


@needs_x
def test_x_env_steps():
    """Stage 2's env takes a 16-joint plant; Stage 1's still refuses it (and the test that pins that stays true)."""
    from designlab.stage2 import r5env

    with pytest.raises(ValueError, match="16 hinges"):
        env.ResidualEnv(env.EnvConfig(body=BODY_X, run_dir=str(r5lib.s1_root(BODY_X) / BODY_X), randomize=False,
                                      gains="start"))
    e = r5env.R5Env(r5env.R5Config(body=BODY_X, set="S1", tasks=("walk",), interference="off"))
    assert e.observation_space.shape == (146,) and e.action_space.shape == (17,) and e.layout["head_x"]
    obs, info = e.reset(seed=5)
    assert obs.shape == (146,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))
    rng = np.random.default_rng(0)
    for _ in range(25):
        obs, r, term, trunc, info = e.step(0.1 * rng.standard_normal(17).astype(np.float32))
        assert np.all(np.isfinite(obs)) and math.isfinite(r)
        assert all(0.0 <= v <= 1.0 for v in info["reward_terms"].values())
        if term or trunc:
            break
    assert e.tk.residual is not None and e.tk.residual.shape == (16,)
    e.close()
