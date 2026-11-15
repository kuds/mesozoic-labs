"""Tests for the views of the bar's episodes (designlab/stage2/barviews.py; decision log P35).

Run from hardware/trex_form_study:  python -m pytest -p no:cacheprovider -q tests/test_stage2_barviews.py
(no test here runs an episode; the one test that renders is behind STAGE2_SLOW=1 and runs itself again, alone, in a
child process with MUJOCO_GL=osmesa, as the look clips' tests do)

1. The pose of a trace row: the attitude columns' convention; the round trip of a kinematic play through the bar's
   own trace file (every body within 1e-4 m and 1e-4 rad); the joint angles' one-step lag on a simulated trace
   (a cubic through the 50 Hz angles); and, behind STAGE2_SLOW=1, a Stage 1 smoke episode run in this process with
   the plant's own body positions kept at every row (the measured bound of ``bv.POSE_BOUND``).
2. A turn's readings on a synthetic circular path with a known radius, centre and heading: the function itself, the
   planned code path on the frozen set and the simulated code path on a made-up run directory (an episode that stops
   short, an episode that fell, a trace that is not the record's, a trace not kept, a record that does not parse).
3. The selection rule on synthetic records.
4. Labels, captions and the seconds of a clip; an episode that fell before its condition's own move began says
   so on every frame; the label bands of barviews' own.
5. Cameras: the top view holds its sweep, up is the incoming walking line; the robot is held in the free picture.
6. Nothing in a run directory outside its views/ is written (the commands run on a made-up run directory and the
   tree is compared); an output directory elsewhere inside it is refused.
7. One rendered frame (STAGE2_SLOW=1).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab.stage1 import control, metrics  # noqa: E402
from designlab.stage2 import bar, setio  # noqa: E402
from designlab.stage2 import barviews as bv  # noqa: E402

BODY = bv.BODY
SLOW = os.environ.get("STAGE2_SLOW") == "1"
CHILD = "BARVIEWS_RENDER_CHILD"
_SET = ROOT / "stage2" / "sets" / "V0" / BODY
needs_set = pytest.mark.skipif(not (_SET / "transitions" / "turn90_pi4_left.npz").exists(),
                               reason="the git-ignored arrays of the frozen set V0 are not here")
ROW = "R2/42@10ms"
BEFORE_ROW = "R2/43@10ms"  # the made-up run's record whose episodes fell before their command
BLOCK = bar.HEADLINE_BLOCK


@pytest.fixture(scope="module")
def model():
    return bv.body_model(BODY)


@pytest.fixture(scope="module")
def v0():
    return bv.set_view("V0", BODY)


# ------------------------------------------------------------------------------------------------
# Helpers: a trace of the home pose gliding along a path (kinematics only), a made-up record
# ------------------------------------------------------------------------------------------------
def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def glide(model, t, xy, psi, joints=None):
    """(trace arrays, qpos [n, nq]) of the home pose carried along ``xy`` with heading ``psi``: every column a
    recorder would write from kinematics, as ``bar.reference_recorder`` writes them."""
    p, m = model.p, model.m
    rows, Q, qs = [], [], []
    quat = np.zeros(4)
    for i in range(len(t)):
        q = model.q_home.copy()
        R = _rz(psi[i] - model.psi_cmd) @ p._R_home
        mujoco.mju_mat2Quat(quat, R.ravel())
        q[:2], q[3:7] = xy[i], quat
        if joints is not None:
            q[p.qadr] = joints[i]
        d = model.kin(q)
        low = metrics.foot_lowest(m, d, model._boxes, model._caps) - p.z_floor
        roll, pitch = control._roll_pitch(d.xmat[p.trunk].reshape(3, 3) @ p._R_home.T)
        u = dict.fromkeys(metrics.COLS, 0.0)
        u.update(t=t[i], t_ref=bar.RATE * t[i], com_x=d.subtree_com[p.trunk][0], com_y=d.subtree_com[p.trunk][1],
                 foot0_x=d.xpos[p.feet[0]][0], foot0_y=d.xpos[p.feet[0]][1], foot1_x=d.xpos[p.feet[1]][0],
                 foot1_y=d.xpos[p.feet[1]][1], low0=low[0], low1=low[1], roll=roll, pitch=pitch,
                 yaw=control._yaw(d.xmat[p.trunk]), root_x=q[0], root_y=q[1])
        rows.append([u[c] for c in metrics.COLS])
        Q.append(q[p.qadr].copy())
        qs.append(q)
    U = np.array(rows)
    return dict(t=U[:, 0].copy(), t_ref=U[:, 1].copy(), rows=U.astype(np.float32), cols=np.array(metrics.COLS),
                q=np.array(Q, np.float32), dt=np.array(model.dt)), np.array(qs)


def corner_path(t, t_c, v, R, sign, angle=math.pi / 2, start=(0.0, 0.0)):
    """A straight walk along +x, then from ``t_c`` an arc of ``angle`` with radius ``R`` to the ``sign`` side at
    the same speed, then straight on: (xy [n, 2], heading [n], centre)."""
    w = v / R
    o = np.array([start[0] + v * t_c, start[1]])
    c = o + sign * np.array([0.0, R])
    xy, psi = np.zeros((len(t), 2)), np.zeros(len(t))
    t_out = t_c + angle / w
    for i, x in enumerate(t):
        if x <= t_c:
            xy[i] = [start[0] + v * x, start[1]]
        else:
            a = w * (min(x, t_out) - t_c)
            xy[i] = c + R * np.array([math.sin(a), -sign * math.cos(a)])
            psi[i] = sign * a
            if x > t_out:
                xy[i] += v * (x - t_out) * np.array([math.cos(sign * a), math.sin(sign * a)])
    return xy, psi, c


def write_trace(run, set_name, row, ckey, seed, arrays, block=BLOCK) -> dict:
    behavior, _, name = ckey.partition("/")
    rel = f"{row}/{behavior}/{name}/{seed}.npz"
    path = Path(run) / set_name / BODY / block / "eval" / "traces" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    return dict(path=rel, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(arrays["t"])))


def episode(seed, fell=False, command_s=None, trace=None, **kw):
    e = dict(seed=seed, fell=fell, fall_time_s=kw.pop("fall_time_s", None), fall_cause="tilt" if fell else None,
             command_s=command_s, fell_before_command=bool(fell and kw.pop("before", False)), measures={})
    if trace is not None:
        e["trace"] = trace
    e.update(kw)
    return e


def record(sv, ckey, row, episodes, block=BLOCK, **kw) -> dict:
    cond = sv.condition(ckey)
    res = sv.planner.resolve(cond)
    listed = {e["file"]: e["sha256"] for doc in (sv.refs, sv.trs) for e in doc.values() if isinstance(e, dict) and e.get("file")}
    rec = dict(schema=bar.SCHEMA, status="done", set=sv.name, body=BODY, block=block, row=row, behavior=cond.behavior,
               condition=cond.name, key_name=ckey, gated=True, definition=cond.describe(),
               plan=[list(x) for x in res["plan"]], seeds=list(bar.BLOCKS[block]), episodes=episodes,
               inputs=dict(files={f: dict(sha256=listed[f]) for f in res["files"]}), key_sha256="0" * 64)
    rec.update(kw)
    return rec


def put_record(run, rec) -> Path:
    path = bv.record_path(run, rec["set"], BODY, rec["block"], rec["row"], rec["key_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return path


def tree(root) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------------------
# 1. The pose of a trace row
# ------------------------------------------------------------------------------------------------
def test_attitude_inverts_the_recorders_columns(model):
    rng = np.random.default_rng(3)
    Rh = model.p._R_home
    for _ in range(200):
        a, b, c = rng.uniform(-math.pi, math.pi), rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)
        cb, sb, cc, sc = math.cos(b), math.sin(b), math.cos(c), math.sin(c)
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
        R = _rz(a) @ Ry @ Rx @ Rh
        roll, pitch = control._roll_pitch(R @ Rh.T)  # what metrics.Recorder.update writes
        yaw = control._yaw(R)
        assert abs(roll - c) < 1e-12 and abs(pitch - b) < 1e-12
        assert np.allclose(bv.attitude(roll, pitch, yaw, Rh), R, atol=1e-12)


@needs_set
def test_pose_round_trip_of_a_kinematic_play_through_the_bars_trace_file(v0, model, tmp_path):
    """A pose written on the plant's model, recorded by ``metrics.Recorder`` (``bar.reference_recorder``), written
    by ``bar.write_trace`` and read back: every body within 1e-4 m and 1e-4 rad."""
    cond = v0.condition("turn/q90_left")
    res = v0.planner.resolve(cond)
    p = model.p
    track = setio.build_track(p, v0.dir, res["plan"], v0.omega(res["files"]), v0.trs)
    rec = bar.reference_recorder(p, track, res["seconds"], v0.dir, rate=bar.RATE)
    info = bar.write_trace(tmp_path, "t.npz", rec, [], None)
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    assert info["rows"] == len(trace["t"]) > 900 and trace["rows"].dtype == np.float32
    pose = bv.pose_of(trace, model, joint_lag_steps=0)
    play = bar._Played(track, v0.dir)
    d = mujoco.MjData(model.m)
    worst_m = worst_rad = 0.0
    for i in range(0, len(pose["t"]), 7):
        d.qpos[:] = play.q(min(bar.RATE * pose["t"][i], track.t_end - 1e-9))
        mujoco.mj_kinematics(model.m, d)
        want_pos, want_mat = d.xpos.copy(), d.xmat.copy()
        got = model.kin(pose["qpos"][i])
        worst_m = max(worst_m, float(np.max(np.linalg.norm(got.xpos - want_pos, axis=1))))
        for b in range(1, model.m.nbody):
            Rd = got.xmat[b].reshape(3, 3) @ want_mat[b].reshape(3, 3).T
            worst_rad = max(worst_rad, math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(Rd) - 1.0)))))
    assert worst_m < 1e-4 and worst_rad < 1e-4, (worst_m, worst_rad)
    assert max(pose["residual_m"].values()) < 1e-5  # the trace's own feet and CoM columns agree too
    # between rows: the row's pose at its own time, a unit quaternion and the held ends
    assert np.array_equal(bv.q_at(pose, pose["t"][40]), pose["qpos"][40])
    mid = bv.q_at(pose, 0.5 * (pose["t"][40] + pose["t"][41]))
    assert abs(np.linalg.norm(mid[3:7]) - 1.0) < 1e-12
    assert np.allclose(mid[7:], 0.5 * (pose["qpos"][40, 7:] + pose["qpos"][41, 7:]))
    assert np.array_equal(bv.q_at(pose, pose["t"][-1] + 5.0), pose["qpos"][-1])


def test_joint_angles_are_moved_back_one_physics_step(model):
    """``metrics.Recorder.update`` writes the joint angles of the state after the step beside the kinematics of the
    state before it: a trace built that way comes back within 0.02 mm with the shift (a cubic through the 50 Hz
    angles; a straight line between the rows left 0.2 mm) and more than a millimetre off without it."""
    p = model.p
    t = np.arange(150) * 0.02
    xy = np.column_stack([0.4 * t, np.zeros(len(t))])
    psi = np.zeros(len(t))
    j = np.flatnonzero([n in ("left_knee", "right_hip_pitch") for n in p.joints])

    def joints(at):
        Q = np.tile(model.q_home[p.qadr], (len(at), 1))
        Q[:, j[0]] += 0.25 * np.sin(2 * math.pi * 1.2 * at)
        Q[:, j[1]] += 0.20 * np.cos(2 * math.pi * 1.2 * at)
        return Q

    trace, qs = glide(model, t, xy, psi, joints(t))  # every column at the rows' own times
    late = dict(trace, q=joints(t + model.dt).astype(np.float32))  # ... but the joint angles one step later
    feet = [model.kin(q).xpos[p.feet].copy() for q in qs]
    err = {}
    for lag in (0, 1):
        pose = bv.pose_of(late, model, joint_lag_steps=lag)
        err[lag] = max(float(np.max(np.linalg.norm(model.kin(pose["qpos"][i]).xpos[p.feet] - feet[i], axis=1)))
                       for i in range(len(t)))
        assert pose["joint_lag_steps"] == lag
    assert err[1] < 2e-5 and err[0] > 1e-3, err
    assert bv.pose_of(late, model)["residual_m"]["feet_xy"] < 2e-5  # (the default is the simulated trace's shift)
    assert np.allclose(bv.joints_at_rows(t, joints(t + model.dt), model.dt, 1), joints(t), atol=2e-5)
    assert not np.allclose(joints(t + model.dt), joints(t), atol=2e-3)
    # fewer than four rows: a straight line between the rows; no lag: the angles as they are
    Q3 = joints(t[:3] + model.dt)
    assert np.allclose(bv.joints_at_rows(t[:3], Q3, model.dt, 1)[1:], Q3[1:] - 0.1 * np.diff(Q3, axis=0))
    assert np.array_equal(bv.joints_at_rows(t, Q3, model.dt, 0), Q3)


@pytest.mark.skipif(not SLOW, reason="slow: set STAGE2_SLOW=1 (one Stage 1 smoke episode, about 10 s)")
def test_pose_of_a_simulated_episode_against_the_plants_own_positions(model, monkeypatch):
    """A Stage 1 smoke episode (the pseudo-set S1: Stage 1's own run directory; never a Stage 2 set) run in this
    process, nothing written, with the plant's body positions kept at every recorder update: the pose rebuilt from
    the trace's arrays lies within ``bv.POSE_BOUND``'s measured bound of them, and a straight line between the
    rows (the rule before) does not."""
    from designlab.stage1 import panels
    from designlab.stage2 import s2io

    stage1_run = ROOT / "stage1" / "runs" / "2026-11-08"
    if not (stage1_run / BODY / "refs" / "nominal.npz").exists():
        pytest.skip("Stage 1's git-ignored arrays are not here")
    truth = []
    update = metrics.Recorder.update

    def keeping(self, *a):
        update(self, *a)
        truth.append(self._v.xpos.copy())  # (the positions the recorder's kinematic columns were read from)

    monkeypatch.setattr(metrics.Recorder, "update", keeping)
    inputs, docs = setio.stage1_view(stage1_run, BODY)
    ident = s2io.sha256_json(dict(files=inputs["files"], freeze=inputs.get("freeze")))
    ctx = bar.context_for(BODY, setio._anchored(stage1_run) / BODY, docs, None, None, ident)
    row = bar.rows(BODY, ["R2/42@0ms"], stage1_run)["R2/42@0ms"]
    cond = next(c for c in bar.stock_conditions() if c.key == "turn/walking_0.35_left")
    _, out, rec = row.episode(ctx, cond, 3042, ())
    monkeypatch.undo()
    every = panels.TRACE_EVERY
    U = rec.U[: rec.rows : every]
    trace = dict(t=U[:, metrics.C["t"]].copy(), t_ref=U[:, metrics.C["t_ref"]].copy(), rows=U.astype(np.float32),
                 cols=np.array(metrics.COLS), q=rec.Q[: rec.rows : every].astype(np.float32), dt=np.array(rec.dt))
    want = np.array(truth)[: rec.rows : every]
    assert len(want) == len(trace["t"]) > 1000 and not out.get("fell")

    def worst(pose):
        return np.array([np.max(np.linalg.norm(model.kin(pose["qpos"][i]).xpos - want[i], axis=1))
                         for i in range(len(want))])

    err = worst(bv.pose_of(trace, model))
    assert err[1:].max() < 1e-4 and err[0] < 1.5e-4, (err[1:].max(), err[0])  # (the first row: the start's transient)
    assert "0.08 mm" in bv.POSE_BOUND and err[1:].max() < 0.08e-3 and "0.11 mm" in bv.POSE_BOUND and err[0] < 0.11e-3
    monkeypatch.setattr(bv, "joints_at_rows", lambda t, Q, dt, lag: np.asarray(Q, float) - (lag * dt / 0.02) * np.vstack(
        [Q[1:2] - Q[0:1], np.diff(np.asarray(Q, float), axis=0)]))
    line = worst(bv.pose_of(trace, model))
    assert (line > 1e-4).sum() > 100 and line.max() > 2e-4, ((line > 1e-4).sum(), line.max())


# ------------------------------------------------------------------------------------------------
# 2. A turn's readings
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sign", (1.0, -1.0))
def test_turn_readings_on_a_known_circle(sign):
    t = np.arange(0.0, 9.0, 0.02)
    v, R, t_c, L = 0.42, 0.6, 2.0, 0.9
    xy, psi, c = corner_path(t, t_c, v, R, sign, start=(1.0, -0.3))
    t_out = t_c + (math.pi / 2) / (v / R)
    t_end = t_out + 1.0
    h = np.column_stack([np.cos(psi), np.sin(psi)])
    pts = np.stack([xy + L * h, xy - L * h, xy], axis=1)  # a snout ahead, a tail behind, the root
    labels = ["head: snout", "tail: tip", "trunk: root"]
    r = bv.turn_readings(t, xy, psi + 2 * math.pi, t_c, t_end, 0.0, sign, math.pi / 2, pts, labels)
    f = r["fit"]
    assert r["reaches_move_end"] and r["steady"]["complete"] and not r["notes"]
    assert abs(f["radius_fitted_m"] - R) < 1e-9 and f["fit_rms_m"] < 1e-9
    assert abs(f["radius_free_fit_m"] - R) < 1e-9  # (no sway on this path: the free fit agrees)
    assert np.allclose(r["centre_world"], c, atol=1e-9)
    assert abs(f["centre_along_m"]) < 1e-9 and abs(f["centre_sideways_m"] - R) < 1e-9
    assert abs(f["turn_rate_deg_s"] - math.degrees(v / R)) < 1e-6 and abs(f["path_speed_m_s"] - v) < 1e-6
    assert abs(r["steady"]["from_deg"] - 15) < 1e-9 and abs(r["steady"]["to_deg"] - 75) < 1e-9
    box = r["corner_box_root"]
    assert abs(box["advance_max_m"] - R) < 1e-6 and abs(box["advance_at_end_m"] - R) < 1e-6
    assert abs(box["sideways_at_end_m"] - (R + v * 1.0)) < 1e-6 and box["back_max_m"] == 0 and box["outside_max_m"] < 1e-12
    # the move ends between two rows: the end itself is in the largest (never 'largest' below 'at the end')
    assert 0.0 < (t_end / 0.02) % 1.0 < 1.0 and box["sideways_max_m"] == pytest.approx(box["sideways_at_end_m"], abs=1e-12)
    assert box["sideways_max_m"] > float(((xy - [1.0 + v * t_c, -0.3]) @ (sign * np.array([0.0, 1.0])))[t <= t_end].max()) + 1e-4
    assert box["seconds_after_command"] == pytest.approx(t_end - t_c)
    rob = r["corner_box_robot"]
    assert rob["sideways_max_m"] == pytest.approx(R + v * 1.0 + L, abs=1e-9)  # (the snout at the end itself)
    # the turn itself: to the first row at which the trunk has turned 90 degrees to within 0.5
    ti = r["turn_itself"]
    i1 = int(np.flatnonzero(np.degrees(sign * psi) >= 89.5)[0])
    assert ti["at_s"] == t[i1] and ti["seconds_after_command"] == pytest.approx(t[i1] - t_c) and ti["within_deg"] == 0.5
    assert 89.5 <= ti["turned_deg"] < 90.0 and t[i1] < t_out
    assert ti["root_advance_m"] == pytest.approx(R * math.sin(math.radians(ti["turned_deg"])), abs=1e-9)
    assert ti["root_sideways_m"] == pytest.approx(R * (1 - math.cos(math.radians(ti["turned_deg"]))), abs=1e-9)
    assert ti["root_to_move_end_m"] == pytest.approx(
        math.hypot(R - ti["root_advance_m"], R + v * 1.0 - ti["root_sideways_m"]), abs=1e-9)
    assert ti["robot"]["sideways_max_m"] < rob["sideways_max_m"] - 0.3 and ti["robot"]["along_max_m"] == pytest.approx(rob["along_max_m"])
    assert bv.reading(r, "turn_root_sideways") == ti["root_sideways_m"] and bv.reading(r, "walk_after_turn") == ti["root_to_move_end_m"]
    # on a circle every point keeps its distance from the centre: the second swept reading is the first
    st = r["swept_turn"]
    assert st["largest_distance_m"] == pytest.approx(math.hypot(R, L), abs=1e-9) and st["over"].startswith("from the command")
    # ... and where the tail stands further out at the command only, it is the second reading that says so
    out = pts.copy()
    k0 = int(np.searchsorted(t, t_c))
    out[k0 : k0 + 3, 1] += 0.2 * (out[k0 : k0 + 3, 1] - c) / np.linalg.norm(out[k0 : k0 + 3, 1] - c, axis=1)[:, None]
    r2 = bv.turn_readings(t, xy, psi, t_c, t_end, 0.0, sign, math.pi / 2, out, labels)
    assert r2["swept"]["swept_radius_m"] == pytest.approx(math.hypot(R, L), abs=1e-9)
    assert r2["swept_turn"]["largest_distance_m"] == pytest.approx(math.hypot(R, L) + 0.2, abs=1e-9)
    assert r2["swept_turn"]["set_by"] == "tail: tip" and r2["swept_turn"]["seconds_after_command"] < 0.06
    assert bv.reading(r2, "largest_from_centre_in_turn") > bv.reading(r2, "swept_radius") + 0.19
    sw = r["swept"]
    assert abs(sw["swept_radius_m"] - math.hypot(R, L)) < 1e-9 and abs(sw["turning_circle_m"] - 2 * math.hypot(R, L)) < 1e-9
    assert sw["set_by"] in ("head: snout", "tail: tip") and abs(sw["farthest_by_part_m"]["trunk"] - R) < 1e-9
    assert abs(sw["inner_radius_m"] - R) < 1e-9
    assert abs(r["turned_at_read_end_deg"] - 90) < 1e-9 and abs(r["turned_at_command_deg"]) < 1e-9
    # the corridor: every point of the sweep inside the L of that width, and not inside a narrower one
    co = r["corridor"]
    ex, ey = np.array([1.0, 0.0]), sign * np.array([0.0, 1.0])
    o = np.array([1.0 + v * t_c, -0.3])
    win = (t >= t_c - 1e-9) & (t <= t_end + 1e-9)
    ends = np.array([bv._between(t_c, t, pts), bv._between(t_end, t, pts)])  # (both ends of the span are held too)
    a, s = (np.concatenate([pts[win], ends]) - o) @ ex, (np.concatenate([pts[win], ends]) - o) @ ey

    def inside(w):
        X, Y0 = co["far_wall_along_m"], co["outer_wall_sideways_m"]
        return bool(np.all((a <= X + 1e-12) & (s >= Y0 - 1e-12) & ~((a < X - w - 1e-12) & (s > Y0 + w + 1e-12))))

    assert inside(co["width_m"]) and not inside(co["width_m"] - 1e-3)
    assert abs(co["far_wall_along_m"] - a.max()) < 1e-12 and abs(co["outer_wall_sideways_m"] - s.min()) < 1e-12
    assert co["inner_corner_set_by"] in labels and 0.3 < co["width_m"] < 2 * L


def test_turn_readings_of_a_path_that_stops_short_and_of_a_cut_trace():
    t = np.arange(0.0, 9.0, 0.02)
    xy, psi, _ = corner_path(t, 2.0, 0.42, 0.6, 1.0, angle=math.radians(40))
    r = bv.turn_readings(t, xy, psi, 2.0, 6.0, 0.0, 1.0, math.pi / 2)
    assert r["fit"] is None and not r["steady"]["complete"] and r["swept"] is None and r["corridor"] is None
    assert any(n.startswith("stops short: the trunk turned 40.0 deg") for n in r["notes"])
    assert r["turn_itself"] is None and any(n.startswith("the turn itself is not read: the trunk turned 40.0 deg") for n in r["notes"])
    assert bv.reading(r, "turn_seconds") is None and bv.reading(r, "turn_robot_along_max") is None
    assert abs(r["turned_at_read_end_deg"] - 40) < 1e-9 and r["corner_box_root"]["advance_max_m"] > 0.6
    # a trace that ends before the end of the move is read to its end and says so; before the command: nothing
    xy, psi, _ = corner_path(t, 2.0, 0.42, 0.6, 1.0)
    cut = t <= 5.0
    pts = np.stack([xy, xy], axis=1)
    r = bv.turn_readings(t[cut], xy[cut], psi[cut], 2.0, None, 0.0, 1.0, math.pi / 2, pts[cut], ["trunk: a", "trunk: b"])
    assert not r["reaches_move_end"] and r["read_to_s"] == pytest.approx(5.0) and r["corridor"] is None
    assert r["fit"] is not None and r["swept"] is not None and any("before the end of the move" in n for n in r["notes"])
    assert r["turn_itself"]["root_to_move_end_m"] is None and r["swept_turn"] is not None  # (turned 90 deg at 4.24 s)
    assert bv._between(5.0, t[cut], pts[cut]) is not None and bv._between(4.99, t, np.where(t[:, None, None] > 4.98, np.nan, pts)) is None
    r = bv.turn_readings(t[t < 1.5], xy[t < 1.5], psi[t < 1.5], 2.0, 6.0, 0.0, 1.0, math.pi / 2)
    assert r["reaches_command"] is False and "fit" not in r


def test_the_heading_keeps_sway_out_of_the_radius():
    """The root sways sideways each step; a circle fitted to the positions alone reads that as curvature."""
    t = np.arange(0.0, 9.0, 0.02)
    xy, psi, _ = corner_path(t, 2.0, 0.42, 0.5, 1.0)
    left = np.column_stack([-np.sin(psi), np.cos(psi)])
    sway = xy + (0.034 * np.sin(2 * math.pi * (t - 2.0) / 0.66 + 0.4))[:, None] * left
    r = bv.turn_readings(t, sway, psi, 2.0, 7.0, 0.0, 1.0, math.pi / 2)
    assert abs(r["fit"]["radius_fitted_m"] - 0.5) < 0.02 < abs(r["fit"]["radius_free_fit_m"] - 0.5)


@needs_set
def test_planned_turn_of_the_frozen_set(v0):
    cond = v0.condition("turn/q90_left")
    pl = bv.planned_turn(v0, cond)
    entry = v0.trs[cond.move]
    cmd, r = pl["commanded"], pl["readings"]
    assert pl["what"] == "planned" and pl["file_sha256"] == entry["sha256"] and pl["freeze"]["index"] == 0
    assert cmd["radius_m"] == pytest.approx(0.4 / (math.pi / 4)) and cmd["direction"] == "left"
    assert cmd["on_the_bars_clock"]["path_speed_m_s"] == pytest.approx(0.4 * bar.RATE) and "in_simulated_time" not in cmd
    assert abs(r["fit"]["radius_fitted_m"] - cmd["radius_m"]) < 0.05 * cmd["radius_m"]
    assert abs(r["fit"]["turn_rate_deg_s"] - bar.RATE * 45.0) < 0.5
    # the heading at the bar's judged instant is the one the set's own record holds for its reference
    assert r["turned_at_judged_instant_deg"] == pytest.approx(entry["reference"]["heading_at_bar_deg"], abs=0.05)
    assert r["turned_at_read_end_deg"] == pytest.approx(90.0, abs=0.01) and r["reaches_move_end"]
    assert r["swept"]["set_by"].startswith("tail:") and r["swept"]["turning_circle_m"] == 2 * r["swept"]["swept_radius_m"]
    assert r["swept"]["swept_radius_m"] > r["fit"]["radius_fitted_m"] + 0.3
    assert 0.3 < r["corridor"]["width_m"] < v0.model.length_m
    assert max(pl["pose_check"].values()) < 1e-9
    # the turn itself is done long before the end of the move's file, which holds the walk out of the turn
    ti, box = r["turn_itself"], r["corner_box_root"]
    assert 2.0 < ti["seconds_after_command"] < 2.3 < box["seconds_after_command"]
    assert ti["root_advance_m"] == pytest.approx(0.61, abs=0.01) and ti["root_sideways_m"] == pytest.approx(0.54, abs=0.01)
    assert ti["root_to_move_end_m"] > 0.7 and box["sideways_max_m"] > ti["root_sideways_m"] + 0.7
    assert box["sideways_max_m"] >= box["sideways_at_end_m"] and box["advance_max_m"] >= box["advance_at_end_m"]
    assert r["corner_box_robot"]["sideways_max_m"] > ti["robot"]["sideways_max_m"] + 0.7
    # at the command the tail reaches further from the turn's centre than on the steady part
    assert r["swept_turn"]["largest_distance_m"] > r["swept"]["swept_radius_m"] + 0.05 and r["swept_turn"]["set_by"].startswith("tail:")
    assert r["swept_turn"]["seconds_after_command"] < 0.1
    assert 1.5 < v0.model.length_m < 2.0 and "snout tip" in v0.model.length_info["front"]
    right = bv.planned_turn(v0, v0.condition("turn/q90_right"))  # the same room on the other side
    assert right["commanded"]["radius_m"] == pytest.approx(cmd["radius_m"]) and right["commanded"]["direction"] == "right"
    assert right["commanded"]["turn_rate_deg_s"] == pytest.approx(45.0)
    assert right["readings"]["fit"]["radius_fitted_m"] == pytest.approx(r["fit"]["radius_fitted_m"], abs=0.01)
    assert right["readings"]["fit"]["centre_sideways_m"] > 0.4 and right["readings"]["turned_at_read_end_deg"] == pytest.approx(90.0, abs=0.01)
    # the turn itself takes the same room to the right; the corner boxes to the ends of the two files do not
    tr = right["readings"]["turn_itself"]
    assert tr["root_advance_m"] == pytest.approx(ti["root_advance_m"], abs=0.01) and tr["root_sideways_m"] == pytest.approx(ti["root_sideways_m"], abs=0.01)
    assert abs(right["readings"]["corner_box_root"]["sideways_max_m"] - box["sideways_max_m"]) > 0.1
    have, missing = bv.turn_conditions(v0)
    assert [c.key for c in have][:2] == ["turn/q90_left", "turn/q90_right"]
    assert all(c.friction is None for c in have) and all(x["reason"] for x in missing)


@pytest.fixture(scope="module")
def fake_run(v0, model, tmp_path_factory):
    """A made-up run directory with one turn record of V0: seed 3042 turns on a known circle, 3043 fell, 3044 stops
    short, 3045's trace is not the record's, 3046's was not kept; a second record file does not parse; a walk
    record whose every episode fell in the start from standing; a stop record (another row) whose episodes fell
    before the stop was commanded.  The traces are kinematic glides of the home pose: no episode is run."""
    run = tmp_path_factory.mktemp("run") / "2026-11-15-75cm"
    ckey = "turn/q90_left"
    pl = v0.planned(v0.condition(ckey))
    t = np.arange(0.0, pl["t_end"] + 1.2, 0.02)
    t_c, v, R = 13.6, 0.42, 0.6
    eps = []
    xy, psi, centre = corner_path(t, t_c, v, R, 1.0)
    full, _ = glide(model, t, xy, psi)
    bar_ok = dict(at_s=2.0, angle_deg=90.0, sign=1.0, heading_error_at_deg=math.degrees(1.4) - 90.0)  # 0.7 rad/s x 2 s
    eps.append(episode(3042, command_s=t_c, trace=write_trace(run, "V0", ROW, ckey, 3042, full), measures=dict(bar=bar_ok)))
    eps.append(episode(3043, fell=True, fall_time_s=14.2, command_s=t_c, trace=write_trace(run, "V0", ROW, ckey, 3043, full)))
    xy2, psi2, _ = corner_path(t, t_c, v, R, 1.0, angle=math.radians(40))
    eps.append(episode(3044, command_s=t_c, trace=write_trace(run, "V0", ROW, ckey, 3044, glide(model, t, xy2, psi2)[0])))
    wrong = write_trace(run, "V0", ROW, ckey, 3045, full)
    eps.append(episode(3045, command_s=t_c, trace=dict(wrong, sha256="f" * 64)))
    eps.append(episode(3046, command_s=t_c, trace=dict(sha256="e" * 64, rows=5, kept=False, reason="disk")))
    eps.append(episode(3047, fell=True, fall_time_s=0.0, error="it broke", trace=None))
    put_record(run, record(v0, ckey, ROW, eps))
    torn = bv.record_path(run, "V0", BODY, BLOCK, "R2/43@10ms", ckey)
    torn.parent.mkdir(parents=True)
    torn.write_text('{"schema": "bar-1", "episodes": [')
    tw = np.arange(0.0, 4.72, 0.02)
    walk, _ = glide(model, tw, np.column_stack([0.3 * tw, np.zeros(len(tw))]), np.zeros(len(tw)))
    weps = [episode(s, fell=True, fall_time_s=4.718, trace=write_trace(run, "V0", ROW, "walk/v0.4", s, walk),
                    fell_before_command=None, segment_reached="start")  # (a walk has no command: bar._judge)
            for s in (3042, 3043)]
    put_record(run, record(v0, "walk/v0.4", ROW, weps))
    ts = np.arange(0.0, 8.24, 0.02)
    early, _ = glide(model, ts, np.column_stack([0.3 * ts, np.zeros(len(ts))]), np.zeros(len(ts)))
    seps = [episode(s, fell=True, fall_time_s=8.222, before=True, command_ref_s=14.8163, segment_reached="nominal_to_v0.4",
                    trace=write_trace(run, "V0", BEFORE_ROW, "stop/two_step", s, early)) for s in (3042, 3043)]
    put_record(run, record(v0, "stop/two_step", BEFORE_ROW, seps))
    (run / "budget.json").write_text("{}")
    return dict(run=run, t_c=t_c, R=R, v=v, centre=centre, t=t, xy=xy, psi=psi, ckey=ckey)


@needs_set
def test_simulated_turn_reads_what_stayed_up_and_lists_the_rest(v0, model, fake_run):
    run = fake_run["run"]
    files = bv.record_files(run, "V0", BODY, "turn")
    assert [f.relative_to(run).as_posix() for f in files] == [
        f"V0/{BODY}/{BLOCK}/eval/R2/42@10ms/turn/q90_left.json", f"V0/{BODY}/{BLOCK}/eval/R2/43@10ms/turn/q90_left.json"]
    s = bv.simulated_turn(v0, run, files[0])
    assert (s["row"], s["delay_ms"], s["floor_grip"], s["block"], s["move"]) == (ROW, 10, 0.9, BLOCK, "turn90_pi4_left")
    assert s["count"] == dict(episodes=6, stayed_up=4, read=2, fell=1)
    assert s["count_text"].startswith("4 of 6 episodes stayed up; 2 of those could be read")
    assert [x["seed"] for x in s["fell"]] == [3043] and s["fell"][0]["when"] == "after the command"
    why = {x["seed"]: x["reason"] for x in s["not_read"]}
    assert why[3045] == "the trace file's sha256 is not the record's" and "not kept (disk)" in why[3046]
    assert why[3047] == "failed to run: it broke"
    a, b = s["episodes"]
    r = a["readings"]
    assert a["seed"] == 3042 and abs(r["fit"]["radius_fitted_m"] - fake_run["R"]) < 1e-4
    assert np.allclose(r["centre_world"], fake_run["centre"], atol=1e-4)
    assert r["command_s"] == fake_run["t_c"] and r["reaches_move_end"]
    # the move's end is where the trace's reference clock reaches the end of the turn's own file
    assert r["move_end_s"] == pytest.approx(v0.planned(v0.condition(fake_run["ckey"]))["t_end"], abs=1e-6)
    # the swept radius, against the surface points carried by hand on the known circle
    t, xy, psi = fake_run["t"], fake_run["xy"], fake_run["psi"]
    rows = np.flatnonzero((t > fake_run["t_c"]) & (np.degrees(psi) >= 15.0) & (np.degrees(psi) < 75.0 + 0.9))
    want = 0.0
    for i in rows:
        q = model.q_home.copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (_rz(psi[i]) @ model.p._R_home).ravel())
        q[:2], q[3:7] = xy[i], quat
        want = max(want, float(np.max(np.linalg.norm(model.surface(q)[:, :2] - fake_run["centre"], axis=1))))
    assert r["swept"]["swept_radius_m"] == pytest.approx(want, abs=2e-3) and r["swept"]["set_by"].startswith("tail:")
    assert r["corridor"]["width_m"] > 0.3
    # the heading at the bar's judged instant is the record's (and the trace agrees with it)
    assert r["judged_instant_s"] == fake_run["t_c"] + 2.0
    assert r["turned_at_judged_instant_deg"] == pytest.approx(math.degrees(1.4))
    assert r["turned_at_judged_instant_from_trace_deg"] == pytest.approx(math.degrees(0.7 * 2.0), abs=1e-3)
    assert b["seed"] == 3044 and b["readings"]["fit"] is None
    assert any(n.startswith("stops short") for n in b["readings"]["notes"])
    sm = s["summary"]
    assert sm["radius_fitted"]["n"] == 1 and sm["advance_max"]["n"] == 2 and sm["turned_at_judged_instant"]["n"] == 1
    assert sm["advance_max"]["min"] <= sm["advance_max"]["median"] <= sm["advance_max"]["max"]
    torn = bv.simulated_turn(v0, run, files[1])
    assert torn["read"] is False and "does not parse" in torn["reason"]
    assert (torn["row"], torn["block"], torn["condition"], torn["move"]) == ("R2/43@10ms", BLOCK, "turn/q90_left", "turn90_pi4_left")
    # a record made from other files than this checkout's set is not read
    rec = json.loads(files[0].read_text())
    rec["inputs"]["files"]["refs/v0.4.npz"]["sha256"] = "0" * 64
    assert v0.inputs_differ(rec) == ["refs/v0.4.npz"]


def test_spread_and_body_lengths():
    assert bv.seed_runs([3044, 3042, 3043, 3050]) == "3042-3044, 3050" and bv.fell_words([]) == "0 fell"
    fell = [dict(seed=s, fall_time_s=4.5 + 0.1 * k, when="before the command") for k, s in enumerate((3042, 3043, 3045))]
    assert bv.fell_words(fell + [dict(seed=3050, fall_time_s=21.654, when="after the command")]) == (
        "4 fell: 3 before the command (seeds 3042-3043, 3045, between 4.50 and 4.70 s); 1 after the command (seed 3050, "
        "at 21.65 s)")
    assert bv.spread([None, 3.0, 1.0, 2.0]) == dict(n=3, median=2.0, min=1.0, max=3.0) and bv.spread([None]) is None
    x = bv.in_body_lengths(dict(a_m=2.0, b=dict(c_m=[1.0, 4.0], d_deg=3.0), body_length_m=2.0, e_m=None), 2.0)
    assert x == dict(a_m=2.0, a_bl=1.0, b=dict(c_m=[1.0, 4.0], c_bl=[0.5, 2.0], d_deg=3.0), body_length_m=2.0, e_m=None)


# ------------------------------------------------------------------------------------------------
# 3. The selection rule
# ------------------------------------------------------------------------------------------------
def _rec(outcomes, kept=None, seeds=None):
    """A record with one episode per entry of ``outcomes`` ("up", "fell" or "error") from seed 3042 on."""
    seeds = seeds or list(range(3042, 3042 + len(outcomes)))
    eps = []
    for s, o in zip(seeds, outcomes, strict=True):
        tr = dict(path=f"x/{s}.npz", sha256="a" * 64) if (kept is None or s in kept) else dict(sha256="a" * 64, kept=False, reason="R0, disk")
        eps.append(episode(s, fell=o != "up", fall_time_s=1.0, trace=tr, **(dict(error="boom") if o == "error" else {})))
    return dict(seeds=seeds, episodes=eps)


def _picked(sel):
    return [(p["seed"], p["selection"]) for p in sel["picks"]]


def test_selection_first_seed_is_the_majority():
    sel = bv.select_episodes(_rec(["up"] * 17 + ["fell"] * 3))
    assert _picked(sel) == [(3042, "first seed")] and not sel["skipped"] and sel["majority"] == "stayed up"
    assert sel["counts"] == {"episodes": 20, "stayed up": 17, "fell": 3, "failed to run": 0}
    sel = bv.select_episodes(_rec(["fell"] * 20))
    assert _picked(sel) == [(3042, "first seed")] and sel["majority"] == "fell"


def test_selection_first_seed_is_not_the_majority():
    sel = bv.select_episodes(_rec(["fell", "fell", "up"] + ["up"] * 17))
    assert _picked(sel) == [(3042, "first seed"), (3044, "majority outcome")]
    assert sel["picks"][1]["words"] == "the lowest seed with the majority outcome" and sel["picks"][1]["passed_over"] == []
    # the lowest majority seed whose trace is kept; the seeds passed over are named
    sel = bv.select_episodes(_rec(["fell", "up", "up", "up"], kept={3042, 3045}))
    assert _picked(sel) == [(3042, "first seed"), (3045, "majority outcome")] and sel["picks"][1]["passed_over"] == [3043, 3044]


def test_selection_no_majority_episode_kept():
    sel = bv.select_episodes(_rec(["fell", "up", "up"], kept={3042}))
    assert _picked(sel) == [(3042, "first seed")]
    assert sel["skipped"] == [dict(seed=3043, selection="majority outcome",
                                   reason="no episode with the majority outcome has a kept trace")]


def test_selection_trace_not_kept_failed_first_seed_and_a_tie():
    sel = bv.select_episodes(_rec(["up", "up", "fell"], kept=set()))
    assert not sel["picks"] and sel["skipped"][0]["seed"] == 3042 and "not kept (R0, disk)" in sel["skipped"][0]["reason"]
    sel = bv.select_episodes(_rec(["error", "fell", "up", "up"]))  # no outcome for the first seed: the majority's lowest
    assert _picked(sel) == [(3044, "majority outcome")] and "failed to run (boom)" in sel["skipped"][0]["reason"]
    assert sel["counts"]["failed to run"] == 1
    sel = bv.select_episodes(_rec(["fell", "up"]))  # a tie: the first seed's outcome is the majority
    assert _picked(sel) == [(3042, "first seed")] and sel["majority"] == "fell"
    assert bv.select_episodes(dict(seeds=[3042], episodes=[]))["skipped"][0]["reason"].startswith("the record holds no episode")


# ------------------------------------------------------------------------------------------------
# 4. Labels, captions, seconds
# ------------------------------------------------------------------------------------------------
DEFS = dict(
    walk=dict(behavior="walk", name="v0.4", walk="v0.4", move=None, friction=None),
    turn=dict(behavior="turn", name="q90_left", walk="v0.4", move="turn90_pi4_left", friction=None),
    slow=dict(behavior="turn", name="t90_0.35_right", walk="v0.4", move="turn90_0.35_right", friction=None),
    stop=dict(behavior="stop", name="two_step", walk="v0.4", move="stop2_v0.4", friction=None),
    nudge=dict(behavior="nudge", name="left_mu0.54", walk="v0.4", move=None, friction=0.54, direction="left", frac=0.25,
               recovery_s=3.0),
)


def test_every_top_band_says_simulation_or_planned_and_never_both():
    assert bv.SET_WORDS == {"V0": "today's walk", "F25F9": "the pull request's walk", "F25F9cl": "the candidate"}
    for set_name, words in bv.SET_WORDS.items():
        for name, definition in DEFS.items():
            rec = dict(row="R2/42@10ms", definition=definition, seeds=[3042, 3043, 3044],
                       episodes=[episode(3042, fell=True, fall_time_s=4.718), episode(3043), episode(3044)])
            pick = bv.select_episodes(dict(rec, episodes=[dict(e, trace=dict(path="p", sha256="s")) for e in rec["episodes"]]))["picks"]
            for view in bv.VIEWS_OF[definition["behavior"]]:
                top, bottom = bv.clip_labels("simulated", BODY, set_name, definition, view, rec, pick[0])
                band = " ".join(top)
                assert top[0].startswith(bv.SIM_LABEL) and bv.PLAN_LABEL not in band and "kinematic" not in band
                assert words in top[0] and set_name in top[0] and "75 cm Hybrid" in top[0]
                assert bv.condition_words(definition) in top[1] and "servo delay 10 ms" in top[1]
                # the fall's second is the episode's own time, and the label says so (the clock gives both times)
                assert bottom[0] == "seed 3042 (first seed): fell 4.72 s into the episode  |  stayed up in 2 of 3"
                assert bottom[1] == bv.VIEW_WORDS[view] == bv.view_words(view)
                top2, _ = bv.clip_labels("simulated", BODY, set_name, definition, view, rec, pick[1])
                assert _[0] == "seed 3043 (majority outcome): stayed up  |  stayed up in 2 of 3"
                assert bv.before_notice(definition, pick[0]["episode"]) is None  # (it fell after its command)
                if name != "nudge":
                    top, bottom = bv.clip_labels("planned", BODY, set_name, definition, view)
                    band = " ".join(top)
                    assert top[0].startswith(bv.PLAN_LABEL) and bv.SIM_LABEL not in band and "SIMULATION" not in band
                    assert words in top[0] and bottom == [bv.VIEW_WORDS[view]]
    with pytest.raises(ValueError):
        bv.clip_labels("both", BODY, "V0", DEFS["walk"], "side")
    assert bv.clock_words(4.0) == "clip  4.00 s   1x" and bv.clock_words(4.0, 8.222) == "clip  4.00 s = episode  8.22 s   1x"
    assert bv.view_words("visitor") == "visitor's view: eye 1.5 m high, 2.5 m away, three-quarter front"
    assert bv.view_words("side", (0.8, 3.75), True) == ("side view: eye 0.8 m high, 3.8 m away (pulled back to hold the "
                                                        "robot), on the robot's right")
    assert "landscape" in bv.CLIP_RULES["watch"] and "episode" in bv.CLIP_RULES["labels"]


def _before_episode(name: str) -> dict:
    """An episode record that fell before its condition's own move began, as ``bar._judge`` writes it."""
    e = episode(3042, fell=True, fall_time_s=8.222, command_ref_s=14.8163, segment_reached="nominal_to_v0.4",
                trace=dict(path="p", sha256="s"))
    if name == "walk":  # no command: fell_before_command is None; it fell in the start, before the 0.40 m/s walk
        e.update(fell_before_command=None, segment_reached="start", command_ref_s=None)
    elif name == "nudge":  # the record holds the push's planned onset, and that the episode fell before it
        e.update(fell_before_command=True, command_s=13.707,
                 measures=dict(push=dict(onset_s=13.707, end_s=13.807, force_N=[0.0, 21.9], fell_before_pulse=True)))
    else:
        e.update(fell_before_command=True)
    return e


def test_a_clip_of_an_episode_that_fell_before_its_move_says_so_on_the_frame():
    """The reviewer's finding: such a clip showed the start falling under a turn's, a stop's or a nudge's title
    with nothing on the frame to say so.  Every frame now carries two lines under the title, the bottom line and
    the fall's caption say 'before', and there is no command caption."""
    want = dict(
        turn=("THE TURN WAS NEVER COMMANDED: this episode fell before it",
              "it fell 8.22 s into the episode, in the change from the nominal walk to 0.40 m/s; the command was due "
              "at about 14.1 s", "before the command"),
        slow=("THE TURN WAS NEVER COMMANDED: this episode fell before it", None, "before the command"),
        stop=("THE STOP WAS NEVER COMMANDED: this episode fell before it", None, "before the command"),
        nudge=("THE PUSH NEVER CAME: this episode fell before it",
               "it fell 8.22 s into the episode, in the change from the nominal walk to 0.40 m/s; the push was due at "
               "13.7 s", "before the push"),
        walk=("THE WALK AT 0.40 m/s NEVER BEGAN: this episode fell before it",
              "it fell 8.22 s into the episode, in the start from standing", "before the walk at 0.40 m/s began"))
    assert 14.8163 / bar.RATE == pytest.approx(14.1, abs=0.05)
    for name, definition in DEFS.items():
        e = _before_episode(name)
        rec = dict(row="R2/42@10ms", definition=definition, seeds=[3042, 3043], episodes=[e, dict(e, seed=3043)])
        pick = bv.select_episodes(rec)["picks"][0]
        head, line, words = want[name]
        assert bv.fell_before(definition, e) and bv.before_words(definition) == words
        notice = bv.before_notice(definition, e)
        assert notice[0] == head and (line is None or notice[1] == line) and len(notice) == 2
        for view in bv.VIEWS_OF[definition["behavior"]]:
            top, bottom = bv.clip_labels("simulated", BODY, "V0", definition, view, rec, pick)
            assert bottom[0] == f"seed 3042 (first seed): fell 8.22 s into the episode, {words}  |  stayed up in 0 of 2"
            assert bv.condition_words(definition) in top[1]  # (the title is the record's; the notice is under it)
        assert bv.fall_caption(definition, e) == f"fell {words} (tilt); the trace ends here, the last pose is held"
        # the clip is the seconds before the fall, and it has no command instant: no command caption can show
        push = (e.get("measures") or {}).get("push") or {}
        span = bv.clip_span(definition, 8.22, push.get("onset_s") if name == "nudge" else e["command_s"], None, None, 8.222)
        assert span["command_s"] is None and span["t1"] == 8.222 and span["hold_s"] == 1.0
        assert all(k != "command" for tt in np.arange(span["t0"], 9.3, 0.05)
                   for k, _ in bv.frame_captions(tt, span, None, bv.fall_caption(definition, e)))
    # after the command, or an episode that stayed up: no notice, the plain words
    after = dict(_before_episode("turn"), fell_before_command=False, command_s=7.0, segment_reached="turn90_pi4_left")
    assert not bv.fell_before(DEFS["turn"], after) and bv.before_notice(DEFS["turn"], after) is None
    assert bv.fall_caption(DEFS["turn"], after) == "fell (tilt); the trace ends here, the last pose is held"
    assert not bv.fell_before(DEFS["turn"], episode(3042)) and bv.before_notice(DEFS["walk"], episode(3042)) is None
    walking = dict(_before_episode("walk"), segment_reached="v0.4")  # it fell in the 0.40 m/s walk itself
    assert not bv.fell_before(DEFS["walk"], walking)
    # a record without the flag: read from the command and the fall instants
    old = episode(3042, fell=True, fall_time_s=8.2, command_s=None)
    del old["fell_before_command"]
    assert bv.fell_before(DEFS["turn"], old) and not bv.fell_before(DEFS["turn"], dict(old, command_s=7.0))
    assert bv.fell_before(DEFS["nudge"], dict(old, command_s=13.7))
    for name, words in (("start", "the start from standing"), ("start_v0.4", "the start from standing into the walk at 0.40 m/s"),
                        ("nominal", "the nominal walk"), ("v0.45", "the walk at 0.45 m/s"), ("stand", "the stand after the stop"),
                        ("turn90_pi4_right", "the turn to the right"), ("stop2_v0.4", "the stop"),
                        (None, "a part of its plan the record does not name")):
        assert bv.segment_words(name) == words


def test_label_bands_of_barviews_own():
    """The lines a viewer needs are larger than ``clips.overlay``'s 13 px; the notice sits under the title beside
    a red bar; the picture between the bands is untouched."""
    top = [f"{bv.SIM_LABEL}  |  75-M-H-10 (75 cm Hybrid)  |  today's walk (V0)", "quick 90° turn to the right at 0.40 m/s  |  "
           "controller R2/43, servo delay 10 ms, floor grip 0.54"]
    bottom = ["seed 3043 (majority outcome): fell 8.22 s into the episode, before the command  |  stayed up in 1 of 20",
              bv.view_words("visitor")]
    notice = bv.before_notice(DEFS["turn"], _before_episode("turn"))
    assert bv.LABEL_PX["top"][1][0] == bv.LABEL_PX["bottom"][0][0] == 17 and bv.LABEL_PX["notice"][0][0] >= 16
    assert bv.band_heights(top, bottom) == (48, 44) and bv.band_heights(top, bottom, notice) == (48 + 4 + 21 + 18, 44)
    assert bv.free_top(48, 0) == 48 and bv.free_top(48, 1) == 48 + 8 + 30 and bv.free_top(91, 2) == 91 + 8 + 36 + 30
    img = np.full((480, 854, 3), 120, np.uint8)
    plain = bv.draw_labels(img, top, bottom, bv.clock_words(4.0, 8.222))
    noted = bv.draw_labels(img, top, bottom, bv.clock_words(4.0, 8.222), notice)
    assert plain.shape == noted.shape == img.shape and plain.dtype == np.uint8
    assert np.array_equal(plain[48:436], img[48:436]) and np.array_equal(noted[91:436], img[91:436])
    assert (plain[:48] != 120).all(axis=2).mean() > 0.99 and (plain[436:] != 120).all(axis=2).mean() > 0.99
    assert not np.array_equal(noted[48:91], img[48:91]) and np.abs(noted[60:80, 13:17].astype(int) - [162, 57, 42]).max() < 3
    dark = lambda a: (a.max(axis=2) < 110)  # noqa: E731  (ink)
    assert dark(plain[24:46, 12:800]).sum() > 1500 and dark(plain[440:458, 12:800]).sum() > 1500  # the 17 px lines
    assert dark(plain[460:476, 600:850]).sum() > 200 and dark(noted[52:90, 26:840]).sum() > 1500  # the clock, the notice
    cap = bv.draw_caption(noted, bv.fall_caption(DEFS["turn"], _before_episode("turn")), "fell", y=91 + 8)
    assert np.array_equal(cap[91 + 8 + 31 : 436], img[91 + 8 + 31 : 436]) and not np.array_equal(cap[99:129], noted[99:129])


def test_plain_words():
    assert bv.condition_words(DEFS["walk"]) == "walk at 0.40 m/s"
    assert bv.condition_words(DEFS["turn"]) == "quick 90° turn to the left at 0.40 m/s"
    assert bv.condition_words(DEFS["slow"]) == "90° turn to the right (0.35 rad/s) at 0.40 m/s"
    assert bv.condition_words(DEFS["stop"]) == "stop from 0.40 m/s within two steps"
    assert bv.condition_words(DEFS["nudge"]).startswith("nudge left (0.25 of the push bound)")
    assert bv.row_words(dict(row="R2/42@10ms", definition=DEFS["walk"])) == "controller R2/42, servo delay 10 ms, floor grip 0.9"
    assert bv.row_words(dict(row="R0@4ms", definition=DEFS["nudge"])) == "open loop (R0), servo delay 4 ms, floor grip 0.54"
    assert bv.command_words(DEFS["turn"]) == "command: turn left 90°" and bv.command_words(DEFS["stop"]) == "command: stop"
    assert bv.command_words(DEFS["walk"]) is None
    e = dict(measures=dict(push=dict(force_N=[0.0, 37.1], onset_s=5.0, end_s=5.1)))  # read from the record
    assert bv.command_words(DEFS["nudge"], e) == "command: push to its left (from its right), 37 newtons for 0.10 s"
    assert bv.push_words([-3.0, 1.0], 0.0) == "backward (from the front)" and bv.push_words([0.0, -1.0], 0.0).startswith("to its right")
    assert bv.push_words([0.0, 1.0], math.pi / 2) == "forward (from behind)"  # (in the commanded heading's frame)
    assert bv.count_words(dict(episodes=[episode(1), episode(2, fell=True), episode(3, fell=True, error="x")])) == \
        "stayed up in 1 of 3 (1 failed to run)"


def test_clip_seconds_and_captions():
    turn, walk = DEFS["turn"], DEFS["walk"]
    s = bv.clip_span(turn, 20.0, 13.7, 18.1, None, None)
    assert (s["t0"], s["t1"], s["hold_s"], s["notes"]) == (12.7, 19.1, 0.0, [])
    s = bv.clip_span(turn, 18.9, 13.7, 18.1, None, None)  # the trace ends first
    assert s["t1"] == 18.9 and "the clip ends there" in s["notes"][0]
    assert bv.clip_span(turn, 18.0, 13.7, None, None, None)["t1"] == 18.0
    s = bv.clip_span(turn, 15.2, 13.7, None, None, 15.21)  # fell after the command: to the fall, then the pose is held
    assert (s["t0"], s["t1"], s["hold_s"], s["command_s"], s["fall_s"]) == (12.7, 15.21, 1.0, 13.7, 15.21)
    s = bv.clip_span(turn, 4.7, None, None, None, 4.718)  # fell before the command
    assert (s["t0"], s["t1"], s["hold_s"], s["command_s"]) == (pytest.approx(0.718), 4.718, 1.0, None)
    assert s["notes"] == ["fell before the command"]
    s = bv.clip_span(walk, 24.4, None, None, 14.4, None)  # the judged window's 10 s
    assert (s["t0"], s["t1"], s["hold_s"]) == (14.4, 24.4, 0.0)
    s = bv.clip_span(walk, 4.7, None, None, None, 4.718)  # a walk that fell (it has no window): from the start
    assert (s["t0"], s["t1"], s["hold_s"], s["notes"]) == (0.0, 4.718, 1.0, ["fell: it has no judged window"])
    with pytest.raises(bv.ViewError):
        bv.clip_span(turn, 20.0, None, None, None, None)
    span = dict(t0=12.7, t1=15.21, hold_s=1.0, command_s=13.7, fall_s=15.21)
    cap = lambda t: bv.frame_captions(t, span, "command: turn left 90°", "fell (tilt)")  # noqa: E731
    assert cap(13.69) == [] and cap(13.7) == [("command", "command: turn left 90°")] and cap(15.19) == cap(13.7)
    assert cap(15.21) == [("fell", "fell (tilt)")] and cap(16.2) == cap(15.21)
    assert bv.frame_captions(15.3, dict(span, fall_s=None), "c", None) == []  # 1.5 s after the command: gone
    assert bv.parse_item("F25F9cl:walk/v0.4:R2/42@10ms") == ("F25F9cl", "walk/v0.4", "R2/42@10ms")
    assert bv.parse_item("V0:turn/q90_left", planned=True) == ("V0", "turn/q90_left")
    with pytest.raises(bv.ViewError):
        bv.parse_item("V0:turn/q90_left")


# ------------------------------------------------------------------------------------------------
# 5. Cameras
# ------------------------------------------------------------------------------------------------
def test_top_camera_holds_the_sweep_with_north_along_the_walking_line():
    rng = np.random.default_rng(5)
    for psi in (0.0, 0.7, -2.0):
        ex, ey = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
        pts = np.array([3.0, -1.0]) + np.outer(rng.uniform(-1.4, 1.1, 400), ex) + np.outer(rng.uniform(-0.6, 2.7, 400), ey)
        cam = bv.top_camera(pts, 0.35, psi)
        assert cam["elevation"] == -90.0 and cam["eye"][2] >= bv.TOP_MIN_HEIGHT_M - 1e-9
        uv = bv.project(np.column_stack([pts, np.zeros(len(pts))]), cam)
        assert uv[:, 0].min() > 0 and uv[:, 0].max() < 854 and uv[:, 1].min() > 46 and uv[:, 1].max() < 480 - 42
        o = np.array([3.0, -1.0, 0.0])
        ahead = bv.project([o, o + np.r_[ex, 0.0], o + np.r_[ey, 0.0]], cam)
        assert ahead[1, 1] < ahead[0, 1] and abs(ahead[1, 0] - ahead[0, 0]) < 1e-6  # further along: higher up
        assert ahead[2, 0] < ahead[0, 0] and abs(ahead[2, 1] - ahead[0, 1]) < 1e-6  # the robot's left: to the left
    cam = bv.top_camera(np.array([[0.0, 0.0], [0.1, 0.1]]), 0.35, 0.0)
    assert cam["eye"][2] == pytest.approx(bv.TOP_MIN_HEIGHT_M)


def test_follow_camera_and_the_smoothed_look_at_point():
    spec = dict(height_m=1.5, distance_m=2.5, bearing_deg=-45.0)
    cam = bv.follow_camera("visitor", (2.0, 1.0), 0.4, 0.0, spec)
    eye, f, r, u = bv.camera_vectors(cam)
    assert np.allclose(eye, cam["eye"]) and eye[2] == pytest.approx(1.5)
    assert np.hypot(*(eye[:2] - [2.0, 1.0])) == pytest.approx(2.5) and eye[1] < 1.0 < eye[0]  # front right
    assert np.allclose(bv.project([[2.0, 1.0, 0.4]], cam), [[427.0, 240.0]])
    cam = bv.follow_camera("visitor", (2.0, 1.0), 0.4, math.pi / 2, spec)  # the bearing turns with the incoming line
    assert cam["eye"][0] > 2.0 and cam["eye"][1] > 1.0
    # pulled back: the eye moves along its line of sight, the look-at point stays in the picture's middle
    near, far = bv.follow_camera("visitor", (2.0, 1.0), 0.4, 0.0, spec), bv.follow_camera("visitor", (2.0, 1.0), 0.4, 0.0, spec, 1.5)
    assert np.allclose(far["eye"] - far["lookat"], 1.5 * (near["eye"] - near["lookat"])) and far["distance"] == pytest.approx(1.5 * near["distance"])
    assert (far["azimuth"], far["elevation"]) == pytest.approx((near["azimuth"], near["elevation"]))
    assert np.allclose(bv.project([[2.0, 1.0, 0.4]], far), [[427.0, 240.0]])
    # the look-at point does not follow a sway of one stride's period
    t = np.arange(0.0, 6.0, 0.02)
    xy = np.column_stack([0.42 * t, 0.03 * np.sin(2 * math.pi * t / 0.8)])
    sm = bv.smooth_rows(xy, 40)
    mid = slice(40, -40)
    assert np.max(np.abs(sm[mid, 1])) < 1e-3 and np.max(np.abs(sm[mid, 0] - (xy[mid, 0] - 0.0042))) < 1e-9
    assert bv.smooth_rows(xy, 1).tolist() == xy.tolist()


def test_the_robot_is_held_in_the_free_picture():
    """``clip_framing`` measures how far the surface stays inside the free part of the picture (under the bands and
    a clip's captions); ``hold_in_picture`` takes the smallest pull-back, and for it the smallest raise of the
    look-at point, that keeps it inside."""
    spec = dict(height_m=1.5, distance_m=2.5, bearing_deg=-45.0)
    body = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.15, 0.15) for z in (0.1, 0.6)])
    tipped = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.15, 0.15) for z in (0.1, 0.85)])  # (it rears up)
    surfaces = [body, tipped, tipped]
    captions = [0, 0, 1]

    def margin(back, lift, bands=(48, 44), caps=captions):
        return bv.clip_framing(surfaces, lambda k: bv.follow_camera("visitor", (0.0, 0.0), 0.4 + lift, 0.0, spec, back), bands, caps)

    m0 = margin(1.0, 0.0)
    uv = bv.project(tipped, bv.follow_camera("visitor", (0.0, 0.0), 0.4, 0.0, spec))
    assert m0 == pytest.approx(min(uv[:, 0].min(), 854 - uv[:, 0].max(), uv[:, 1].min() - bv.free_top(48, 1), 480 - 44 - uv[:, 1].max()))
    assert m0 < 0 and margin(1.0, 0.0, caps=[0, 0, 0]) == pytest.approx(m0 + 38)  # (the caption's 38 px of the top)
    assert margin(1.0, 0.0, bands=(91, 44)) == pytest.approx(m0 - 43)
    back, lift, m = bv.hold_in_picture(margin)
    assert m >= 0 and m == pytest.approx(margin(back, lift)) and (back, lift) != (1.0, 0.0)
    assert back in bv.PULL_BACK and lift in bv.LOOK_LIFT_M
    assert all(margin(b, h) < 0 for b in bv.PULL_BACK if b < back for h in bv.LOOK_LIFT_M)  # no smaller factor holds it
    assert all(margin(back, h) < 0 for h in bv.LOOK_LIFT_M if h < lift)                      # ... and no smaller raise
    assert bv.hold_in_picture(lambda b, h: 5.0) == (1.0, 0.0, 5.0)  # a robot that is inside: nothing changes
    # none holds it: the largest factor and its best raise (the command then prints a warning)
    assert bv.hold_in_picture(lambda b, h: -100.0 + b + 10 * h - 30 * h * h) == (2.5, 0.15, pytest.approx(-96.675))
    assert bv.clip_log(dict(file="a.mp4", frames=3, bytes=2e6, render_s=1.0, encode_s=1.0, framing_margin_px=-12.4,
                            camera=dict(pulled_back=2.5, lookat_raised_m=0.15))).endswith(
        "margin -12 px, eye pulled back 2.5 x, look-at raised 0.15 m\n  WARNING: the robot leaves the free picture by 12 px in a.mp4")
    assert "WARNING" not in bv.clip_log(dict(file="a.mp4", frames=3, bytes=2e6, render_s=1.0, encode_s=1.0,
                                             framing_margin_px=3.0, camera=dict(kind="fixed")))


def test_the_renderer_is_told_not_to_keep_a_shader_cache():
    """Mesa's shader cache under the home directory was the one write outside the output directories."""
    env = bv.renderer_env({})
    assert env == {"MESA_SHADER_CACHE_DISABLE": "true", "MESA_GLSL_CACHE_DISABLE": "true"}
    assert bv.renderer_env({"MESA_SHADER_CACHE_DISABLE": "false"})["MESA_SHADER_CACHE_DISABLE"] == "false"  # (a set value stays)


# ------------------------------------------------------------------------------------------------
# 6. Nothing outside views/ is written
# ------------------------------------------------------------------------------------------------
@needs_set
def test_the_commands_write_nothing_in_the_run_directory_outside_views(v0, fake_run, capsys):
    run = fake_run["run"]
    before = tree(run)
    assert before and not any(k.startswith("views/") for k in before)
    assert bv.main(["turnpath", "--set", "V0", "--run", str(run), "--out", str(run / "views")]) == 0
    assert bv.main(["posecheck", f"V0:{fake_run['ckey']}:{ROW}", "V0:walk/v0.4:R0@10ms", "--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "V0: 6 planned turns, 2 simulated records, 1 not read" in out and "the record cannot be opened" in out
    after = tree(run)
    assert {k: v for k, v in after.items() if not k.startswith("views/")} == before
    made = sorted(k for k in after if k.startswith("views/"))
    assert "views/turnpath/V0/turnpath.json" in made and "views/turnpath/V0/TURNPATH.md" in made
    assert "views/turnpath/V0/turn_q90_left.svg" in made and "views/turnpath/V0/turn_t90_0.35_right.png" in made
    doc = json.loads((run / "views" / "turnpath" / "V0" / "turnpath.json").read_text())
    assert doc["schema"] == bv.SCHEMA_TURNPATH and doc["run"] == "2026-11-15-75cm" and doc["rules"] == bv.TURN_RULES
    q = doc["turns"][0]
    assert q["condition"] == "turn/q90_left" and q["planned"]["what"] == "planned" and len(q["simulated"]) == 2
    sim = q["simulated"][0]
    assert sim["record"] == f"V0/{BODY}/{BLOCK}/eval/R2/42@10ms/turn/q90_left.json" and "_paths" not in sim
    ep = sim["episodes"][0]  # the committed file holds each episode's readings as plain values
    assert ep["seed"] == 3042 and ep["values"]["radius_fitted"] == pytest.approx(0.6, abs=1e-4) and "readings" not in ep
    assert sorted(ep["values"]) == sorted(bv.READINGS) and doc["reading_units"]["radius_fitted"] == "m"
    assert ep["swept_radius_set_by"].startswith("tail:") and sim["episodes"][1]["values"]["radius_fitted"] is None
    assert any(n.startswith("stops short") for n in sim["episodes"][1]["notes"])
    assert sim["pose_check_worst"]["feet_xy"] < 1e-5 and "pose_check" not in ep
    assert q["planned"]["readings"]["fit"]["radius_fitted_bl"] == pytest.approx(
        q["planned"]["readings"]["fit"]["radius_fitted_m"] / v0.model.length_m)
    assert doc["turns"][1]["simulated"] == [] and doc["turns"][1]["simulated_note"] == "no record yet"
    assert [x["record"] for x in doc["records_not_read"]] == [f"V0/{BODY}/{BLOCK}/eval/R2/43@10ms/turn/q90_left.json"]
    md = (run / "views" / "turnpath" / "V0" / "TURNPATH.md").read_text()
    assert "### Planned" in md and "### Simulated" in md and "4 of 6 episodes stayed up" in md and "No record yet" in md
    blocks = md.split("### Planned")[1:]
    assert len(blocks) == 6
    for blk in blocks:  # the planned tables never use the word kept for an episode's readings
        planned_rows = [x for x in blk.split("### Simulated")[0].splitlines() if x.startswith("|")]
        assert len(planned_rows) >= 18 and not any("simulated" in x.lower() for x in planned_rows)
        assert sum("on the bar's clock" in x for x in planned_rows) == 2
        assert sum(x.startswith("| the turn itself") or x.startswith("| **the turn itself**") for x in planned_rows) == 4
        assert sum(x.startswith("| corner box to the end of the move's file") for x in planned_rows) == 4
        assert any("of walk after the turn" in x for x in planned_rows)
        assert any(x.startswith("| largest distance from the turn's centre, from the command until") for x in planned_rows)
    assert "| the turn itself, root: sideways when the trunk has turned 90° |" in md
    assert q["planned"]["commanded"]["on_the_bars_clock"]["path_speed_m_s"] == pytest.approx(0.42)
    assert doc["rules"]["pose"] == bv.POSE_BOUND and "turn_itself" in doc["rules"] and "ends" in doc["rules"]
    pr = q["planned"]["readings"]
    assert pr["turn_itself"]["root_sideways_bl"] == pytest.approx(pr["turn_itself"]["root_sideways_m"] / v0.model.length_m)
    assert ep["values"]["turn_root_advance"] == pytest.approx(0.6, abs=1e-3) and ep["values"]["turn_root_sideways"] == pytest.approx(0.6, abs=6e-3)
    assert ep["largest_from_centre_in_turn_set_by"].startswith("tail:") and sim["episodes"][1]["values"]["turn_seconds"] is None
    assert "dashed box is the turn itself" in q["figure"]["caption"] and "dotted box" in q["figure"]["caption"]
    assert "1 fell: 1 after the command (seed 3043, at 14.20 s)." in md and "| reading (simulated) |" in md
    assert "it held 2 turn records of this set when this file was written" in md
    assert q["figure"]["simulated_from"] == sim["record"] and "SIMULATED root paths of the 2 episodes" in q["figure"]["caption"]
    assert q["figure"]["rows_before_without_a_path"] == [] and "R2/42@10ms, R2/43@10ms, R0@10ms" in q["figure"]["rule"]
    # the figure's record: the first row in the stated order with a path to draw; the rows before it are named
    a = dict(row="R2/42@10ms", floor_grip=0.9, block=BLOCK, read=True, _paths=[], count_text="0 of 20 episodes stayed up")
    b = dict(row="R0@10ms", floor_grip=0.9, block=BLOCK, read=True, _paths=[np.zeros((2, 2))], count_text="20 of 20")
    # the judged delay's rows first, then each shorter delay the chain runs, the nearest first, down to none
    assert bv.FIGURE_ROWS[:3] == ("R2/42@10ms", "R2/43@10ms", "R0@10ms") and bv.FIGURE_ROWS[-3:] == ("R2/42@0ms", "R2/43@0ms", "R0@0ms")
    assert [bar.delay_ms(x) for x in bv.FIGURE_ROWS[::3]] == [10, 8, 6, 4, 2, 0] and len(bv.FIGURE_ROWS) == 18
    late = dict(b, row="R2/42@4ms")
    assert bv.figure_record([dict(b, row="R0@0ms"), late, a])[0] is late  # (4 ms before 0 ms)
    low = dict(b, row="R2/43@10ms", floor_grip=0.54)
    assert bv.figure_record([low, b, a]) == (b, ["R2/42@10ms (0 of 20 episodes stayed up)"])
    assert bv.figure_record([low, a]) == (a, []) and bv.figure_record([low]) == (None, [])
    assert bv.figure_record([a, b], ("R2/42@10ms",)) == (a, [])
    # an output directory elsewhere inside the run directory is refused, and nothing is written
    assert bv.main(["turnpath", "--set", "V0", "--run", str(run), "--out", str(run / "V0" / "views")]) == 1
    assert bv.main(["turnpath", "--set", "V0", "--run", str(run), "--out", str(run)]) == 1
    assert "refused:" in capsys.readouterr().err and tree(run) == after
    with pytest.raises(bv.ViewError):
        bv.check_out(run, run / "budget.json")
    assert bv.check_out(run, run / "views" / "x") == run / "views" / "x"


@needs_set
def test_clips_list_names_the_episodes_without_rendering(fake_run, capsys):
    run = fake_run["run"]
    before = tree(run)
    items = bv.gated_items(ROW, sets=("V0",))
    assert len(items) == 16 and items[0] == ("V0", "walk/v0.4", ROW) and ("V0", "turn/q90_right_mu0.54", ROW) in items
    assert bv.main(["clips", "--list", "--all-gated", "--row", ROW, "--set", "V0", "--run", str(run)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 16 and tree(run) == before
    assert out[0] == ("V0 walk/v0.4 R2/42@10ms: stayed up in 0 of 2; seed 3042 (first seed, fell before the walk at 0.40 m/s "
                      "began); views visitor, side")
    assert out[2] == ("V0 turn/q90_left R2/42@10ms: stayed up in 4 of 6 (1 failed to run); seed 3042 (first seed, stayed up); "
                      "views visitor, top")
    assert out[1] == "V0 walk/v0.4_mu0.54 R2/42@10ms: no clip (no record yet)"
    assert bv.main(["clips", "--run", str(run)]) == 1 and bv.main(["clips", "--all-gated", "--run", str(run)]) == 1
    assert "refused:" in capsys.readouterr().err
    # a record whose episodes fell before the stop was commanded: the list says so
    listed = bv.list_clips([("V0", "stop/two_step", BEFORE_ROW)], run, log=lambda *a: None)
    assert [(c["seed"], c["outcome"], c["fell_before_its_move"]) for c in listed[0]["clips"]] == [(3042, "fell before the command", True)]
    assert tree(run) == before


def test_run_label_and_missing_records(tmp_path):
    assert bv.run_label("/somewhere/else/hardware/trex_form_study/stage2/runs/2026-11-15-75cm") == "stage2/runs/2026-11-15-75cm"
    assert bv.record_files(tmp_path, "V0") == []
    assert bv.read_record(tmp_path / "none.json") == (None, "cannot be opened (FileNotFoundError)")
    (tmp_path / "other.json").write_text('{"schema": "panel-1"}')
    assert bv.read_record(tmp_path / "other.json")[1] == f"is not a {bar.SCHEMA} record"
    rec = dict(set="V0", body=BODY, block=BLOCK)
    assert bv.read_trace(tmp_path, rec, dict(seed=1))[1] == "the record names no trace for this episode"
    assert bv.read_trace(tmp_path, rec, dict(trace=dict(path="a/1.npz", sha256="0")))[1].startswith("the trace file is missing")


# ------------------------------------------------------------------------------------------------
# 7. One rendered frame (slow)
# ------------------------------------------------------------------------------------------------
def _render_in_child(request) -> bool:
    """Run the calling test again, alone, in a fresh process with MuJoCo's OSMesa backend; True in the parent
    (tests/test_stage2_clips.py's rule: the shared test process never loads a GL backend)."""
    if os.environ.get(CHILD) == "1":
        return False
    cache = request.getfixturevalue("tmp_path_factory").mktemp("xdg_cache")  # where Mesa would keep its shader cache
    env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="1", XDG_CACHE_HOME=str(cache),
               **{CHILD: "1"})
    for name in ("MESA_SHADER_CACHE_DISABLE", "MESA_GLSL_CACHE_DISABLE", "MESA_SHADER_CACHE_DIR"):
        env.pop(name, None)  # (the module's own default is what is tested)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", request.node.nodeid],
                       cwd=request.config.rootpath, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    return True


@pytest.mark.skipif(not SLOW, reason="slow: set STAGE2_SLOW=1")
@needs_set
def test_one_rendered_frame(request, v0, model):
    if _render_in_child(request):
        return
    from designlab.stage2 import clips

    cs = bv.scene(BODY)
    pl = v0.planned(v0.condition("turn/q90_left"))
    pose = pl["pose"]
    q = bv.q_at(pose, pl["t_c"] + 0.75)
    away = q.copy()
    away[0] += 200.0
    box, _ = cs.home_box(model.q_home)
    rows = np.flatnonzero((pose["t"] >= pl["t_c"] - 1) & (pose["t"] <= pl["t_end"] + 1))[::5]
    cams = dict(visitor=bv.follow_camera("visitor", q[:2], float(box[2]), 0.0, clips.CAMERAS["visitor"]),
                top=bv.top_camera(np.concatenate([model.surface(pose["qpos"][i])[:, :2] for i in rows]), float(box[2]), 0.0))
    for view, cam in cams.items():
        dim = bv.TOP_HEADLIGHT_DIFFUSE if view == "top" else None
        img, empty = bv.draw_frame(cs, q, cam, q[:2], dim), bv.draw_frame(cs, away, cam, q[:2], dim)
        assert img.shape == (480, 854, 3) and img.dtype == np.uint8
        robot = np.abs(img.astype(int) - empty.astype(int)).max(axis=2) > 30
        assert robot.sum() > (20000 if view == "visitor" else 1500), (view, int(robot.sum()))
        uv = bv.project(model.surface(q), cam)  # the robot is drawn where the surface points project (its shadow
        vs, us = np.nonzero(robot)              # differs too, so the drawn region reaches beyond them)
        assert us.min() < uv[:, 0].min() + 25 and us.max() > uv[:, 0].max() - 25
        assert vs.min() < uv[:, 1].min() + 25 and vs.max() > uv[:, 1].max() - 25
        inside = robot[int(uv[:, 1].min()) : int(uv[:, 1].max()) + 1, int(uv[:, 0].min()) : int(uv[:, 0].max()) + 1]
        assert inside.mean() > 0.15
        top, bottom = bv.clip_labels("planned", BODY, "V0", v0.condition("turn/q90_left").describe(), view)
        th, bh = bv.band_heights(top, bottom)
        out = bv.draw_caption(bv.draw_labels(img, top, bottom, bv.clock_words(1.75)), "command: turn left 90°", "command",
                              y=th + bv.CAPTION_GAP_PX)
        assert out.shape == img.shape and np.abs(out[60:80, 12:18].astype(int) - [42, 120, 214]).max() < 3
        assert np.array_equal(out[200:400], img[200:400])  # the picture's middle is untouched by the labels
    assert np.array_equal(cs.m.vis.headlight.diffuse, np.float32([0.55, 0.55, 0.55]))
    # the renderer wrote no shader cache (``bv.renderer_env``; this process's cache directory is the test's own)
    cache = Path(os.environ["XDG_CACHE_HOME"])
    assert os.environ["MESA_SHADER_CACHE_DISABLE"] == "true" and cache.is_dir() and not [p for p in cache.rglob("*") if p.is_file()]


@pytest.mark.skipif(not SLOW, reason="slow: set STAGE2_SLOW=1")
@needs_set
def test_clips_of_a_made_up_record_and_the_tree(request, v0, fake_run, tmp_path):
    """``barviews clips`` on the made-up run directory: the first seed's turn in both views (it is of the majority
    outcome, so no second episode), a walk that fell in its start, a stop whose episode fell before the command, a
    record that does not exist; nothing written outside views/."""
    if _render_in_child(request):
        return
    run = fake_run["run"]
    before = {k: v for k, v in tree(run).items() if not k.startswith("views/")}
    media = tmp_path / "media"
    with pytest.raises(bv.ViewError):
        bv.make_clips([("V0", fake_run["ckey"], ROW)], run=run, media=run / "media", out=run / "views")
    lines = []
    doc = bv.make_clips([("V0", fake_run["ckey"], ROW), ("V0", "walk/v0.4", ROW), ("V0", "stop/two_step", ROW),
                         ("V0", "stop/two_step", BEFORE_ROW)],
                        run=run, media=media, out=run / "views", log=lambda *a: lines.append(" ".join(str(x) for x in a)))
    assert {k: v for k, v in tree(run).items() if not k.startswith("views/")} == before
    assert json.loads((run / "views" / "clips.json").read_text())["schema"] == bv.SCHEMA_CLIPS
    assert [(c["condition"], c["row"], c["view"], c["seed"], c["selection"], c["outcome"]) for c in doc["clips"]] == [
        ("stop/two_step", BEFORE_ROW, "side", 3042, "first seed", "fell"), ("stop/two_step", BEFORE_ROW, "visitor", 3042, "first seed", "fell"),
        ("turn/q90_left", ROW, "top", 3042, "first seed", "stayed up"), ("turn/q90_left", ROW, "visitor", 3042, "first seed", "stayed up"),
        ("walk/v0.4", ROW, "side", 3042, "first seed", "fell"), ("walk/v0.4", ROW, "visitor", 3042, "first seed", "fell")]
    assert [(x["condition"], x["row"], x["reason"]) for x in doc["skipped"]] == [("stop/two_step", ROW, "no record yet")]
    assert doc["rules"] == json.loads(json.dumps(bv.CLIP_RULES)) and "landscape" in doc["rules"]["watch"]
    # every clip holds the robot in the free part of its picture, or the command warned about it
    for c in doc["clips"]:
        warned = any("WARNING: the robot leaves the free picture" in x and c["file"] in x for x in lines)
        assert warned == (c["framing_margin_px"] < 0), (c["file"], c["framing_margin_px"])
        assert c["bands_px"] == list(bv.band_heights(c["labels"]["top"], c["labels"]["bottom"], c["labels"]["notice"]))
    t_end = v0.planned(v0.condition(fake_run["ckey"]))["t_end"]
    for c in doc["clips"]:
        mp4 = media / c["file"]
        assert c["bytes"] == mp4.stat().st_size <= bv.MAX_BYTES and c["sha256"] == hashlib.sha256(mp4.read_bytes()).hexdigest()
        assert (media / c["poster"]).exists() and (media / c["sheet"]).exists()
        assert c["probe"]["codec"] == "h264" and (c["probe"]["width"], c["probe"]["height"]) == (854, 480)
        assert c["probe"]["fps"] == bv.FPS and c["probe"]["frames"] == c["frames"]
        assert c["labels"]["top"][0].startswith(bv.SIM_LABEL) and c["record"].endswith(".json") and len(c["trace_sha256"]) == 64
        if c["condition"] == "turn/q90_left":  # 1 s before the command to 1 s after the end of the move, real time
            assert c["span"]["t0"] == pytest.approx(fake_run["t_c"] - 1.0) and c["span"]["t1"] == pytest.approx(t_end + 1.0, abs=1e-6)
            assert c["seconds"] == pytest.approx(t_end + 2.0 - fake_run["t_c"], abs=1.0 / bv.FPS)
            assert c["captions"] == dict(command="command: turn left 90°", fell=None) and c["record_count"] == "stayed up in 4 of 6 (1 failed to run)"
            assert c["fell_before_its_move"] is False and c["labels"]["notice"] is None
            if c["view"] == "top":  # the fixed eye holds the whole sweep
                assert c["framing_margin_px"] > 0 and c["camera"]["kind"] == "fixed"
            else:  # the glide is far from the planned path: the constant shift of the clip keeps it in the picture
                assert np.linalg.norm(c["camera"]["shift_from_plan_m"]) > 1.0
                assert c["camera"]["largest_distance_box_centre_to_lookat_m"] < 0.5 and c["framing_margin_px"] >= 0
        elif c["condition"] == "walk/v0.4":  # a walk that fell in its start: from the start to the fall, the pose held 1 s
            assert (c["span"]["t0"], c["span"]["t1"], c["span"]["hold_s"]) == (0.0, 4.718, 1.0)
            assert c["captions"] == dict(command=None, fell="fell before the walk at 0.40 m/s began (tilt); the trace ends "
                                                              "here, the last pose is held")
            assert c["fell_before_its_move"] is True and c["labels"]["notice"] == [
                "THE WALK AT 0.40 m/s NEVER BEGAN: this episode fell before it",
                "it fell 4.72 s into the episode, in the start from standing"]
            assert c["labels"]["bottom"][0] == ("seed 3042 (first seed): fell 4.72 s into the episode, before the walk at "
                                                "0.40 m/s began  |  stayed up in 0 of 2")
            assert c["framing_margin_under_captions_px"] >= 0  # (the fall is held under its caption)
        else:  # the stop was never commanded: the frame says so, and no command caption exists
            assert c["row"] == BEFORE_ROW and c["span"]["command_s"] is None and c["span"]["notes"] == ["fell before the command"]
            assert (c["span"]["t0"], c["span"]["t1"], c["span"]["hold_s"]) == (pytest.approx(4.222), 8.222, 1.0)
            assert c["captions"] == dict(command=None, fell="fell before the command (tilt); the trace ends here, the last "
                                                              "pose is held")
            assert c["fell_before_its_move"] is True and c["segment_reached"] == "nominal_to_v0.4"
            assert c["labels"]["notice"] == ["THE STOP WAS NEVER COMMANDED: this episode fell before it",
                                             "it fell 8.22 s into the episode, in the change from the nominal walk to 0.40 m/s; "
                                             "the command was due at about 14.1 s"]
            assert c["labels"]["top"][1].startswith("stop from 0.40 m/s within two steps") and c["bands_px"][0] > 48
            assert c["labels"]["bottom"][0].startswith("seed 3042 (first seed): fell 8.22 s into the episode, before the command")
            assert c["framing_margin_under_captions_px"] >= 0
    # a second run of one item keeps the other clips' entries
    doc2 = bv.make_clips([("V0", "turn/q90_left")], planned=True, run=run, media=media, out=run / "views",
                         views=("top",), log=lambda *a: None)
    assert len(doc2["clips"]) == 7 and [c["kind"] for c in doc2["clips"]].count("planned") == 1
    planned = next(c for c in doc2["clips"] if c["kind"] == "planned")
    assert planned["labels"]["top"][0].startswith(bv.PLAN_LABEL) and planned["seed"] is None and planned["view"] == "top"
    assert planned["labels"]["notice"] is None and planned["captions"]["command"] == "command: turn left 90°"
    assert doc2["skipped"] == doc["skipped"]
