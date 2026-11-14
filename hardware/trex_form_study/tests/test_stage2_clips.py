"""Tests for the Stage 2 look clips (designlab/stage2/clips.py; STAGE2_PLAN.md section 4 A.4, research_look.md 4.1
item 3).

Run from hardware/trex_form_study:  python -m pytest tests/test_stage2_clips.py -q
(the two tests that render run themselves again, alone, in a child process with MUJOCO_GL=osmesa: ``_render_in_child``)

1. Playback: at a frame's own time the pose is that frame's, a window later the same pose moved by the advance; between
   frames the joints are linear and the root quaternion stays a unit quaternion; a turning window is refused.
2. Cameras: each eye at its height and floor distance from the look-at point, on its bearing, where MuJoCo's free
   camera puts it; the look-at point follows the mean forward motion, not the surge.
3. Labels: every clip carries "kinematic reference, not a simulation"; feature runs; the Stage 0 line where a rung
   is over a limit; the text fits the frame.
4. Encoding: an H.264 MP4 at the frame size and rate, under its size limit (the two-pass fallback included).
5. The review set, and (with the git-ignored arrays and an offscreen GL) a short real clip of each mode.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco is imported)
from designlab.stage2 import clips, look  # noqa: E402

LOOK = ROOT / "stage2" / "look"
LADDER = json.loads((LOOK / "look_refs.json").read_text()) if (LOOK / "look_refs.json").exists() else None
_V0 = None if LADDER is None else LOOK / LADDER["bodies"]["60-N-T-10"]["v040"]["V0"]["file"]
needs_arrays = pytest.mark.skipif(_V0 is None or not _V0.exists(), reason="the git-ignored ladder arrays are not here")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")

CHILD = "STAGE2_CLIPS_CHILD"


def _render_in_child(request) -> bool:
    """Run the calling rendering test again, alone, in a fresh process with MuJoCo's OSMesa backend; True in the parent.

    MuJoCo picks its OpenGL backend at its first import, so in a shared pytest process whether a render works depends
    on which test module imported mujoco first (GLFW, chosen without MUJOCO_GL, fails with no display and can abort
    the run). Setting MUJOCO_GL=osmesa for the whole session is no cure: with OSMesa loaded, torch segfaults in the
    PPO tests. So the rendering tests never render in the shared process.
    """
    if os.environ.get(CHILD) == "1":
        return False
    env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", **{CHILD: "1"})
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", request.node.nodeid],
                       cwd=request.config.rootpath, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    return True


def _quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _arrays(n=12, dt=0.01, adv=0.3, seed=1):
    rng = np.random.default_rng(seed)
    q = np.zeros((n, 10))
    q[:, 0] = np.linspace(0, adv, n, endpoint=False)
    q[:, 1:3] = rng.normal(0, 0.01, (n, 2))
    base = _quat(rng)
    for k in range(n):  # small rotations about a base attitude
        d = np.concatenate([[1.0], rng.normal(0, 0.02, 3)])
        q[k, 3:7] = d / np.linalg.norm(d) if k % 2 else base
    q[:, 7:] = rng.normal(0, 0.3, (n, 3))
    com = np.column_stack([np.linspace(0, adv, n, endpoint=False) + 0.01 * np.sin(np.arange(n)), np.zeros(n),
                           np.full(n, 0.4)])
    return dict(q=q, com=com, dt=np.array(dt), stride_motion=np.array([0.0, 0.0, 0.0, adv, 0.0]))


# ------------------------------------------------------------------------------------------------
# 1. Playback
# ------------------------------------------------------------------------------------------------
def test_playback_is_exact_at_frames_and_repeats_with_the_advance():
    a = _arrays()
    pb = clips.Playback(a)
    n, dt, adv = 12, 0.01, 0.3
    assert pb.window == pytest.approx(n * dt) and pb.speed == pytest.approx(adv / (n * dt))
    for k in (0, 1, 5, 11):
        assert np.array_equal(pb.q_at(k * dt), a["q"][k])
        later = a["q"][k].copy()
        later[0] += 2 * adv
        np.testing.assert_allclose(pb.q_at(2 * n * dt + k * dt), later, atol=1e-12)
        np.testing.assert_allclose(pb.com_at(n * dt + k * dt), a["com"][k] + [adv, 0, 0], atol=1e-12)


def test_playback_interpolates_between_frames_and_across_the_window_end():
    a = _arrays()
    pb = clips.Playback(a)
    dt, n, adv = 0.01, 12, 0.3
    mid = pb.q_at(3.25 * dt)
    np.testing.assert_allclose(mid[7:], 0.75 * a["q"][3, 7:] + 0.25 * a["q"][4, 7:], atol=1e-12)
    np.testing.assert_allclose(mid[:3], 0.75 * a["q"][3, :3] + 0.25 * a["q"][4, :3], atol=1e-12)
    assert np.linalg.norm(mid[3:7]) == pytest.approx(1.0, abs=1e-12)
    wrap = pb.q_at((n - 0.5) * dt)  # half way from the last frame to the next window's first
    assert wrap[0] == pytest.approx(0.5 * a["q"][-1, 0] + 0.5 * (a["q"][0, 0] + adv), abs=1e-12)
    np.testing.assert_allclose(wrap[7:], 0.5 * a["q"][-1, 7:] + 0.5 * a["q"][0, 7:], atol=1e-12)


def test_playback_refuses_a_turning_window():
    a = _arrays()
    a["stride_motion"] = np.array([0.1, 0.0, 0.0, 0.3, 0.0])
    with pytest.raises(ValueError):
        clips.Playback(a)


def test_clip_times_are_ten_seconds_at_thirty_fps():
    ts = clips.times()
    assert len(ts) == 300 and ts[0] == 0.0 and ts[-1] == pytest.approx(299 / 30) and clips.CLIP_S == 10.0


# ------------------------------------------------------------------------------------------------
# 2. Cameras
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("view", list(clips.CAMERAS))
def test_camera_eye_height_distance_and_bearing(view):
    spec = clips.CAMERAS[view]
    look_at = np.array([1.7, 0.0, 0.31])
    c = clips.camera_pose(look_at, spec)
    eye = c["eye"]
    assert eye[2] == pytest.approx(spec["height_m"])
    assert math.hypot(*(eye[:2] - look_at[:2])) == pytest.approx(spec["distance_m"])
    # MuJoCo's free camera: eye = lookat - distance (cos el cos az, cos el sin az, sin el)
    az, el = math.radians(c["azimuth"]), math.radians(c["elevation"])
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    np.testing.assert_allclose(look_at - c["distance"] * fwd, eye, atol=1e-12)
    assert eye[1] < look_at[1]  # on the robot's right
    if view == "visitor":  # three-quarter front: ahead of the look-at point and 45 deg off the path
        assert eye[0] > look_at[0] and (eye[0] - look_at[0]) == pytest.approx(look_at[1] - eye[1])
        assert c["elevation"] < 0
    else:  # square to the path
        assert eye[0] == pytest.approx(look_at[0])


def test_track_follows_the_mean_forward_motion_not_the_surge():
    a = _arrays()
    pb = clips.Playback(a)
    ts = clips.times(1.0)
    tr = clips.make_track(pb, box_centre=np.array([-0.1, 0.0, 0.3]), com_home=np.array([0.02, 0.0, 0.4]), ts=ts)
    assert tr.speed == pytest.approx(pb.speed)
    x = np.array([tr.lookat(t)[0] for t in ts])
    np.testing.assert_allclose(np.diff(x), pb.speed / 30.0, atol=1e-12)  # no surge
    assert tr.x0 == pytest.approx(np.mean([pb.com_at(t)[0] - pb.speed * t for t in ts]))
    assert tr.dx == pytest.approx(-0.12) and tr.z == pytest.approx(0.3)
    assert tr.travel(0.5) == pytest.approx(tr.x0 + 0.5 * pb.speed - 0.02)


# ------------------------------------------------------------------------------------------------
# 3. Labels
# ------------------------------------------------------------------------------------------------
def test_feature_text_joins_runs():
    assert clips.feature_text(["F1", "F2", "F3", "F4", "F5", "F6", "F9", "F10", "F14"]) == "F1-F6, F9, F10, F14"
    assert clips.feature_text(["F1", "F2"]) == "F1, F2"
    assert clips.feature_text(["F1", "F2", "F3"]) == "F1-F3"
    assert clips.feature_text(["F1", "F2", "F1 at 50 %"]) == "F1, F2, F1 at 50 %"
    assert clips.feature_text([]) == ""


@pytest.mark.parametrize("mode", clips.MODES)
def test_every_clip_says_kinematic_reference_not_a_simulation(mode):
    e = dict(features=["F1", "F2", "F3", "F4", "F5", "F6", "F9", "F10", "F13", "F14"], info=dict(speed=0.4961))
    over = [dict(limit="knee speed", value=1.0122964726272832)]
    top, bottom = clips.labels("75-M-H-10", "fr05", "V6", "visitor", mode, e, over)
    assert top[1].startswith(clips.LABEL) and clips.LABEL == "kinematic reference, not a simulation"
    assert "F1-F6, F9, F10, F13, F14" in top[0] and "0.50 m/s" in top[0]
    assert top[2].startswith("Stage 0: over a limit (knee speed 1.012)")  # at the top: the feet need the bottom
    assert bottom == [clips.CAMERAS["visitor"]["text"]]
    top0, bottom0 = clips.labels("60-N-T-10", "v040", "V0", "side", mode, dict(features=[], info=dict(speed=0.4)), [])
    assert "V0: today's walk" in top0[0] and len(top0) == 2 and len(bottom0) == 1
    img = np.full((clips.SIZE[1], clips.SIZE[0], 3), 0 if mode == "pointlight" else 128, np.uint8)
    out = clips.overlay(img, top, bottom, 3.25, dark=(mode == "pointlight"))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (out[:20] != img[:20]).any() and (out[-20:] != img[-20:]).any()  # top and bottom bands drawn
    mid = slice(clips.BANDS_PX[0] + clips.LINE_PX * (len(top) - 2), clips.SIZE[1] - clips.BANDS_PX[1])
    assert np.array_equal(out[mid], img[mid])  # the picture between the bands is untouched


def test_label_lines_fit_the_frame():
    from PIL import Image, ImageDraw

    dr = ImageDraw.Draw(Image.new("RGB", clips.SIZE))
    e = dict(features=["F1", "F2", "F3", "F4", "F5", "F6", "F9", "F10", "F13", "F14", "F7"], info=dict(speed=0.4961))
    top, bottom = clips.labels("75-M-H-10", "fr05", "V6S", "visitor", "pointlight", e,
                               [dict(limit="knee speed", value=1.009)])
    for k, s in enumerate(top + bottom):
        f = clips._fit(dr, s, 15 if k == 0 else 13, k == 0, clips.SIZE[0] - 24)
        assert dr.textlength(s, font=f) <= clips.SIZE[0] - 24
        assert f.size >= 12  # legible


# ------------------------------------------------------------------------------------------------
# 4. Encoding
# ------------------------------------------------------------------------------------------------
@needs_ffmpeg
def test_encode_makes_h264_under_the_limit(tmp_path):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (30, 48, 64, 3), dtype=np.uint8)  # noise: the CRF file is large
    first = clips.encode(frames, tmp_path / "a.mp4", fps=30, max_bytes=10**7)
    assert first["tries"][0]["crf"] == clips.CRF and len(first["tries"]) == 1
    limit = int(0.6 * first["bytes"])
    out = clips.encode(frames, tmp_path / "b.mp4", fps=30, max_bytes=limit, meta=dict(comment=clips.LABEL))
    assert out["bytes"] <= limit and "two_pass_kbit_s" in out["settings"]
    p = clips.probe(tmp_path / "b.mp4")
    assert p["codec"] == "h264" and (p["width"], p["height"]) == (64, 48) and p["frames"] == 30
    assert p["fps"] == 30 and p["pix_fmt"] == "yuv420p"


# ------------------------------------------------------------------------------------------------
# 5. The review set and real clips
# ------------------------------------------------------------------------------------------------
def test_review_set_is_the_tasks_list():
    rs = clips.review_set()
    assert len(rs) == 42 and len(set(rs)) == 42
    for b in look.BODIES:
        mine = [r for r in rs if r[0] == b]
        cov = {r[1:4] for r in mine if r[4] == "covered"}
        for s in look.SPEEDS:
            for v in ("V0", "V6"):
                assert {(s, v, "visitor"), (s, v, "side")} <= cov
        for v in ("V1", "V2", "V3", "V4", "V5"):
            assert (("v040", v, "visitor") in cov) and (("v040", v, "side") not in cov)
        pl = {r[1:4] for r in mine if r[4] == "pointlight"}
        assert pl == {("v040", v, view) for v in ("V0", "V6") for view in clips.CAMERAS}


@needs_arrays
@needs_ffmpeg
@pytest.mark.parametrize("mode", clips.MODES)
def test_a_short_real_clip(tmp_path, mode, request):
    if _render_in_child(request):
        return
    e = clips.render_clip("60-N-T-10", "v040", "V0", "side", mode, tmp_path, clip_s=0.1)
    assert e["probe"]["frames"] == 3 and e["probe"]["codec"] == "h264" and e["bytes"] <= clips.MAX_BYTES
    assert e["labels"][1].startswith(clips.LABEL)
    assert e["rung_sha256"] == LADDER["bodies"]["60-N-T-10"]["v040"]["V0"]["sha256"]
    from PIL import Image

    im = np.asarray(Image.open(tmp_path / e["poster"]))  # the poster time is past a 0.1 s clip: the last frame kept
    assert im.shape == (clips.SIZE[1], clips.SIZE[0], 3)
    body = im[60:420, 100:760].astype(int)
    if mode == "pointlight":  # white dots on black
        assert (body.min(axis=2) > 200).sum() > 50 and (body.max(axis=2) < 10).mean() > 0.95
    else:  # the robot's covers (green) are in the frame centre
        green = (body[..., 1] > body[..., 0] + 15) & (body[..., 1] > body[..., 2] + 15)
        assert green.sum() > 5000


@needs_arrays
def test_projection_matches_the_rendered_dots(tmp_path, request):
    """``project`` puts each marker on its rendered point-light dot (one frame, the side view)."""
    if _render_in_child(request):
        return
    from PIL import Image
    from scipy import ndimage

    e = clips.render_clip("60-N-T-10", "v040", "V6", "side", "pointlight", tmp_path, clip_s=0.1)
    cs = clips._scene("60-N-T-10", clips.OUT)
    _, arrs = clips.rung("60-N-T-10", "v040", "V6")
    t = 2 / 30.0  # the last frame, which the poster keeps
    cs.pose(clips.Playback(arrs).q_at(t))
    uv = clips.project(cs.points(), clips.camera_pose(clips.Track(**e["track"]).lookat(t), clips.CAMERAS["side"]))
    img = np.asarray(Image.open(tmp_path / e["poster"]).convert("L")) > 128
    img[: clips.BANDS_PX[0]] = img[-clips.BANDS_PX[1] :] = False  # the labels
    lab, n = ndimage.label(img)
    cents = np.array(ndimage.center_of_mass(img, lab, range(1, n + 1)))[:, ::-1]
    assert max(np.min(np.linalg.norm(cents - p, axis=1)) for p in uv) < 5.0  # px (overlapping dots merge)


CLIPS = json.loads((LOOK / "clips.json").read_text()) if (LOOK / "clips.json").exists() else None


@needs_arrays
@pytest.mark.skipif(CLIPS is None, reason="stage2/look/clips.json not generated")
def test_every_recorded_clip_keeps_the_robot_in_the_picture():
    """The committed manifest: every clip of the review set, each at 10 s, 30 fps, 854 x 480, H.264 under 1.5 MB, from
    the rung look_refs.json records, labelled; and on every frame all 13 markers inside the picture, clear of the
    label bands (``frame_margin``)."""
    assert {(e["body"], e["speed"], e["variant"], e["view"], e["mode"]) for e in CLIPS["clips"]} == set(clips.review_set())
    for e in CLIPS["clips"]:
        p = e["probe"]
        assert (p["codec"], p["width"], p["height"], p["frames"], p["fps"]) == ("h264", 854, 480, 300, 30.0)
        assert e["bytes"] < clips.MAX_BYTES and e["labels"][1].startswith(clips.LABEL)
        assert e["label_lines"] == [3 if e["stage0_over"] else 2, 1]
        assert e["rung_sha256"] == LADDER["bodies"][e["body"]][e["speed"]][e["variant"]]["sha256"]
        assert clips.frame_margin(e)["margin_px"] > 10, e["file"]
