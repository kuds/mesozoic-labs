"""Stage 2 Track A: kinematic clips of the look ladder (``STAGE2_PLAN.md`` section 4 A.4; ``stage2/research/
research_look.md`` 4.1 item 3).

Every clip is a kinematic playback of a rung's reference walk, not a simulation: the reference's poses drawn at 1x
real time with perfect tracking and no physics, and every frame says so (``LABEL``).  Nothing is adopted (P24).  The
rungs are the ladder's (``look.prepare``; arrays git-ignored), each file checked against the sha256 ``look_refs.json``
records; the media are written outside git (``--media``).

Playback (``Playback``)
-----------------------
* 10 s (``look.CLIP_S``) at 30 fps, frame k at t = k / 30 s.  A rung's window (one stride on V0-V5; F14's whole
  strides on V6, 10.2-10.6 s) repeats with its advance (``stride_motion``: a translation along x), so V0-V5 loop
  seamlessly and V6's window covers the clip without repeating.
* Between the reference's frames (4.5-7.1 ms apart) the root position and the joint angles are interpolated linearly
  and the root's quaternion by normalised linear interpolation; at a frame's own time the pose is that frame's.

Cameras (``CAMERAS``, ``Track``)
--------------------------------
* ``visitor``: the eye 1.5 m above the floor and 2.5 m from the look-at point along the floor, three-quarter front:
  on the robot's front right, 45 deg off its path (``render.VIEWS``' three-quarter view is front right too).
* ``side``: the eye 0.6 m above the floor, 2.5 m from the walking line on the robot's right, square to its path.
* Both: a 26 deg vertical field of view (45 deg across at 854 x 480, about a normal lens's).  The look-at point is the
  centre of the robot's bounding box at home (``render.Scene.bounds``: covers, legs and tail) carried along x at the
  walk's mean speed v (advance / window) from x0, the mean over the clip of (CoM x - v t), on the walking line y = 0.
  The cameras so follow the body's mean forward motion; its surge, sway and bob stay in the picture.
* The model's own scene: its checker floor, headlight and key light; the key light travels with the cameras, so the
  lighting and the shadow stay as they are at the start, while the floor stays in the world and a stance foot stands
  still on it.  Where the side view sees past the floor's far end (the far clipping plane), the sky is ``render.BG``.

Clips (``render_clip``)
-----------------------
* ``covered``: ``render.Scene``'s ``exterior`` style (covers on, hardware visible through the openings).
* ``pointlight`` (research 4.1 item 3; Johansson): every geom hidden, white dots on black at ``look.POINTS``' 13
  markers (snout, an eye, the neck base, hips, knees, MTPs (the foot bodies' origins), toe tips (the middle claws'
  tips), mid-tail and tail tip; ``look_refs.json``'s ``body_info.markers``), placed on the interpolated pose.
* On every frame: the body, the rung and its features, the speed; ``LABEL``; the view; the Stage 0 verdict where the
  rung is over a limit (``screen.json``); the clip time.
* 854 x 480 H.264 (libx264, yuv420p, faststart) through ffmpeg, each under ``MAX_BYTES``: CRF 23 with a capped rate,
  then two passes at a bitrate set from the size if that file is larger (``encode``).  Beside each MP4 a poster (the
  frame at 5 s) and a contact sheet (0, 2.5, 5 and 7.5 s), PNG; ``clips.json`` lists them (in the media directory
  and, without the media, in ``stage2/look/clips.json``).

Run from hardware/trex_form_study (MUJOCO_GL=osmesa; one llvmpipe thread per process):
    python -m designlab.stage2.clips render --media <dir> --procs 3          # the review set (``review_set``)
    python -m designlab.stage2.clips one --media <dir> --body 60-N-T-10 --speed v040 --variant V6 --view side
"""

from __future__ import annotations

import os

if __name__ == "__main__":  # the clips render in parallel processes: one llvmpipe thread each
    os.environ.setdefault("LP_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .. import render  # noqa: E402
from ..stage1 import plant  # noqa: E402
from . import look, screen  # noqa: E402

OUT = look.OUT
FPS = 30
CLIP_S = look.CLIP_S
SIZE = (854, 480)
#: On every frame (the task's wording).
LABEL = "kinematic reference, not a simulation"
#: Each MP4's size limit (bytes) and the encoder's settings (``encode``).
MAX_BYTES = 1_500_000
CRF = 23
MAXRATE_K = 1100
FOVY_DEG = 26.0
SKY = render.BG
#: The fixed cameras (module docstring): eye height and distance along the floor from the look-at point (m), and the
#: bearing of the eye from it (deg from +x, the walking direction; -y is the robot's right).
CAMERAS = {
    "visitor": dict(height_m=1.5, distance_m=2.5, bearing_deg=-45.0,
                    text="visitor's view: eye 1.5 m high, 2.5 m away, three-quarter front"),
    "side": dict(height_m=0.6, distance_m=2.5, bearing_deg=-90.0, text="side view: eye 0.6 m high, 2.5 m away"),
}
MODES = ("covered", "pointlight")
#: Point-light dots: radius (m) on black.
POINT_RADIUS_M = 0.011
#: The poster's and the contact sheet's times (s).
POSTER_T = 5.0
SHEET_T = (0.0, 2.5, 5.0, 7.5)
SPEED_TEXT = {"nominal": "nominal walk, Froude 0.012", "v040": "0.40 m/s, the agility bar's speed",
              "fr05": "animal pace, Froude 0.05"}
RUNG_TEXT = {"V0": "today's walk"}
#: The review set per body (``review_set``; the task's list): V0 and V6 at every speed from both views; V1-V5 at
#: 0.40 m/s from the visitor's view; point-light V0 and V6 at 0.40 m/s from both views.
REVIEW = dict(both=("V0", "V6"), ladder=("V1", "V2", "V3", "V4", "V5"), ladder_speed="v040", pointlight=("V0", "V6"),
              pointlight_speed="v040")


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------
_DOCS: dict = {}


def docs(out_dir: Path = OUT) -> tuple[dict, dict | None]:
    """``look_refs.json`` and ``screen.json`` (None if absent) of ``out_dir``, read once."""
    key = str(Path(out_dir).resolve())
    if key not in _DOCS:
        sp = Path(out_dir) / screen.SCREEN_JSON
        _DOCS[key] = (json.loads((Path(out_dir) / "look_refs.json").read_text()),
                      json.loads(sp.read_text()) if sp.exists() else None)
    return _DOCS[key]


def rung(body: str, speed: str, variant: str, out_dir: Path = OUT) -> tuple[dict, dict]:
    """(entry, arrays) of a rung: its ``look_refs.json`` entry and its arrays, checked against the recorded sha256."""
    doc, _ = docs(out_dir)
    e = doc["bodies"][body][speed][variant]
    return e, screen.load_arrays(e, out_dir)


def stage0_over(body: str, speed: str, variant: str, out_dir: Path = OUT) -> list:
    """The rung's Stage 0 limits over 1 (``screen.json``'s ``over``; empty if within or not screened)."""
    _, sdoc = docs(out_dir)
    if sdoc is None:
        return []
    return list(sdoc["rungs"].get(body, {}).get(speed, {}).get(variant, {}).get("over", []))


def feature_text(features) -> str:
    """'F1-F6, F9, F10, F14': runs of consecutive feature numbers joined (extras' words kept as they are)."""
    nums, words = [], []
    for f in features:
        if f[0] == "F" and f[1:].isdigit():
            nums.append(int(f[1:]))
        else:
            words.append(f)
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"F{nums[i]}" if j - i < 2 else f"F{nums[i]}-F{nums[j]}")
        if j - i == 1:
            parts.append(f"F{nums[j]}")
        i = j + 1
    return ", ".join(parts + words)


# ------------------------------------------------------------------------------------------------
# Playback
# ------------------------------------------------------------------------------------------------
class Playback:
    """A rung's window played at 1x and repeated with its advance (module docstring)."""

    def __init__(self, arrs: dict):
        self.q = np.asarray(arrs["q"], float)
        self.com = np.asarray(arrs["com"], float)
        self.n = len(self.q)
        self.dt = float(arrs["dt"])
        self.window = self.n * self.dt
        sm = np.asarray(arrs["stride_motion"], float)
        if np.any(np.abs(sm[[0, 1, 2, 4]]) > 1e-12):
            raise ValueError(f"stride_motion {sm.tolist()}: not a straight window (a pure advance along x)")
        self.advance = float(sm[3])
        self.speed = self.advance / self.window

    def _frames(self, tau: float) -> tuple[int, int, int, int, float]:
        s = tau / self.dt
        k0 = int(math.floor(s + 1e-9))
        a = max(s - k0, 0.0) if abs(s - round(s)) > 1e-9 else 0.0
        c0, i0 = divmod(k0, self.n)
        c1, i1 = divmod(k0 + 1, self.n)
        return c0, i0, c1, i1, a

    def q_at(self, tau: float) -> np.ndarray:
        """The pose (qpos) at time ``tau`` (s, from the window's start)."""
        c0, i0, c1, i1, a = self._frames(tau)
        q0 = self.q[i0].copy()
        q0[0] += c0 * self.advance
        if a == 0.0:
            return q0
        q1 = self.q[i1].copy()
        q1[0] += c1 * self.advance
        q = (1.0 - a) * q0 + a * q1
        r0, r1 = q0[3:7], q1[3:7]
        r = (1.0 - a) * r0 + a * (r1 if np.dot(r0, r1) >= 0 else -r1)
        q[3:7] = r / np.linalg.norm(r)
        return q

    def com_at(self, tau: float) -> np.ndarray:
        """The model CoM at ``tau`` (linear between frames, the advance added per window)."""
        c0, i0, c1, i1, a = self._frames(tau)
        p0 = self.com[i0] + np.array([c0 * self.advance, 0.0, 0.0])
        if a == 0.0:
            return p0
        p1 = self.com[i1] + np.array([c1 * self.advance, 0.0, 0.0])
        return (1.0 - a) * p0 + a * p1


def times(clip_s: float = CLIP_S, fps: int = FPS) -> np.ndarray:
    """The clip's frame times: k / fps for k < clip_s fps."""
    return np.arange(int(round(clip_s * fps))) / float(fps)


@dataclass
class Track:
    """The look-at point (module docstring): x(t) = x0 + speed t + dx, y = 0, z; ``com_home_x``: the CoM at home, from
    which the key light's travel is counted."""

    x0: float
    speed: float
    dx: float
    z: float
    com_home_x: float

    def lookat(self, tau: float) -> np.ndarray:
        return np.array([self.x0 + self.speed * tau + self.dx, 0.0, self.z])

    def travel(self, tau: float) -> float:
        """How far the cameras (and the key light) have moved along x from the home pose's place."""
        return self.x0 + self.speed * tau - self.com_home_x


def make_track(pb: Playback, box_centre: np.ndarray, com_home: np.ndarray, ts: np.ndarray) -> Track:
    """The cameras' track for a playback (``Track``): v the walk's mean speed, x0 the clip's mean of CoM x - v t, dx
    and z the home bounding box centre's offset from the home CoM and its height."""
    x0 = float(np.mean([pb.com_at(t)[0] - pb.speed * t for t in ts]))
    return Track(x0=x0, speed=pb.speed, dx=float(box_centre[0] - com_home[0]), z=float(box_centre[2]),
                 com_home_x=float(com_home[0]))


def camera_pose(lookat: np.ndarray, spec: dict) -> dict:
    """A free camera at ``spec``'s height and floor distance from ``lookat`` on its bearing, aimed at it: MuJoCo's
    lookat, distance, azimuth and elevation (deg), and the eye's position."""
    b = math.radians(spec["bearing_deg"])
    eye = np.array([lookat[0] + spec["distance_m"] * math.cos(b), lookat[1] + spec["distance_m"] * math.sin(b),
                    spec["height_m"]])
    d = lookat - eye
    return dict(lookat=np.asarray(lookat, float), distance=float(np.linalg.norm(d)),
                azimuth=math.degrees(math.atan2(d[1], d[0])),
                elevation=math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), eye=eye)


# ------------------------------------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------------------------------------
class ClipScene:
    """One body's compiled MJCF in ``render.Scene`` at ``SIZE``, with the point-light markers."""

    def __init__(self, body: str, out_dir: Path = OUT):
        xml, _ = plant.model_files(body)
        self.model_check = dict(plant.verify(body, xml), file=Path(xml).relative_to(look.STUDY).as_posix())
        self.body = body
        self.sc = render.Scene(str(xml), SIZE[0], SIZE[1])
        self.m, self.d = self.sc.m, self.sc.d
        self.light_pos0 = self.m.light_pos.copy()
        doc, _ = docs(out_dir)
        mk = doc["body_info"][body]["markers"]
        if list(mk) != list(look.POINTS):
            raise ValueError(f"{body}: markers {list(mk)} are not look.POINTS")
        self.markers = [(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, v["body"]), np.asarray(v["point"], float))
                        for v in mk.values()]
        if any(b < 0 for b, _ in self.markers):
            raise KeyError(f"{body}: a marker's body is not on the model")
        self.opt_covered = self.sc._style("exterior")
        self.opt_none = mujoco.MjvOption()
        self.opt_none.geomgroup[:] = 0
        self.opt_none.sitegroup[:] = 0
        self.far = float(self.m.stat.extent * self.m.vis.map.zfar)

    def close(self):
        self.m.light_pos[:] = self.light_pos0
        self.sc.close()

    def pose(self, q: np.ndarray) -> None:
        self.d.qpos[:] = q
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def home_box(self, q_home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(the bounding box centre, the CoM) at home."""
        self.pose(q_home)
        lo, hi = self.sc.bounds()
        return 0.5 * (lo + hi), self.d.subtree_com[0].copy()

    def points(self) -> np.ndarray:
        """The markers on the current pose (world)."""
        return np.array([self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ p for b, p in self.markers])

    def frame(self, q: np.ndarray, cam: dict, travel: float, mode: str) -> np.ndarray:
        """One frame at pose q from camera ``cam`` (``camera_pose``), the key light moved by ``travel`` along x."""
        m = self.m
        m.light_pos[:] = self.light_pos0
        m.light_pos[:, 0] += travel
        self.pose(q)
        m.vis.global_.orthographic = 0
        m.vis.global_.fovy = FOVY_DEG
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = cam["lookat"]
        c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        r = self.sc.r
        if mode == "covered":
            r.update_scene(self.d, c, self.opt_covered)
            img = r.render().copy()
            if cam["elevation"] + 0.5 * FOVY_DEG > 0.0:  # the top of the frame looks above the horizon: sky
                r.enable_depth_rendering()
                r.update_scene(self.d, c, self.opt_covered)
                depth = r.render()
                r.disable_depth_rendering()
                img[depth >= 0.999 * self.far] = SKY
            return img
        if mode == "pointlight":
            r.update_scene(self.d, c, self.opt_none)
            scn = r.scene
            for p in self.points():
                if scn.ngeom >= scn.maxgeom:
                    raise RuntimeError("scene full")
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([POINT_RADIUS_M, 0.0, 0.0]), p,
                                    np.eye(3).ravel(), np.array([1.0, 1.0, 1.0, 1.0], np.float32))
                g.emission = 1.0
                scn.ngeom += 1
            return r.render().copy()
        raise ValueError(mode)


def _band(width: int, height: int, alpha: int = 205) -> Image.Image:
    return Image.new("RGBA", (width, height), (252, 252, 251, alpha))


def _fit(dr: ImageDraw.ImageDraw, text: str, size: int, bold: bool, width: float):
    """The font at ``size`` or smaller (down to 10 px) that fits ``text`` in ``width`` pixels."""
    f = render.font(size, bold=bold)
    while dr.textlength(text, font=f) > width and size > 10:
        size -= 1
        f = render.font(size, bold=bold)
    return f


def overlay(img: np.ndarray, top: list[str], bottom: list[str], t: float, dark: bool) -> np.ndarray:
    """The labels (module docstring): ``top`` lines top left (the first bold), ``bottom`` lines bottom left and the
    clip time bottom right; on light bands over the covered scene, light grey on the point-light's black."""
    im = Image.fromarray(img).convert("RGBA")
    W, H = im.size
    ink = (215, 215, 210, 255) if dark else (11, 11, 11, 255)
    ink2 = (165, 165, 160, 255) if dark else (82, 81, 78, 255)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    ts = f"t {t:5.2f} s   1x"
    f13 = render.font(13)
    tw = dr.textlength(ts, font=f13)
    top_h = [21] + [17] * (len(top) - 1)
    bot_h = [17] * len(bottom)
    if not dark:
        im.alpha_composite(_band(W, 8 + sum(top_h)))
        im.alpha_composite(_band(W, 8 + max(sum(bot_h), 17)), (0, H - 8 - max(sum(bot_h), 17)))
    y = 5
    for k, (s, h) in enumerate(zip(top, top_h, strict=True)):
        f = _fit(dr, s, 15 if k == 0 else 13, k == 0, W - 24)
        dr.text((12, y), s, fill=ink if k == 0 else ink2, font=f)
        y += h
    y = H - 4 - sum(bot_h)
    for k, (s, h) in enumerate(zip(bottom, bot_h, strict=True)):
        room = W - 24 - (tw + 16 if k == len(bottom) - 1 else 0)  # the last line shares its row with the time
        dr.text((12, y), s, fill=ink2, font=_fit(dr, s, 13, False, room))
        y += h
    dr.text((W - tw - 10, H - 4 - 17), ts, fill=ink2, font=f13)
    im.alpha_composite(layer)
    return np.asarray(im.convert("RGB"))


def labels(body: str, speed: str, variant: str, view: str, mode: str, entry: dict, over: list) -> tuple[list, list]:
    """The label lines of a clip (module docstring): (top, bottom)."""
    feats = feature_text(entry["features"])
    v = float(entry["info"]["speed"])
    sp = SPEED_TEXT[speed] + ("" if speed == "v040" else f", {v:.2f} m/s")
    top = [f"{screen.BODY_LABEL.get(body, body)}  |  {variant}: {RUNG_TEXT.get(variant, feats)}  |  {sp}",
           f"{LABEL}: perfect tracking, no physics" + ("; point-light markers on black" if mode == "pointlight"
                                                        else "")]
    bottom = []
    if over:
        lim = "; ".join(f"{o['limit']} {o['value']:.3f}" if isinstance(o.get("value"), float) else str(o["limit"])
                        for o in over)
        bottom.append(f"Stage 0: over a limit ({lim}); the plan's rule drops this speed rung unless the owner "
                      "decides otherwise")
    bottom.append(CAMERAS[view]["text"])
    return top, bottom


# ------------------------------------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------------------------------------
def _ffmpeg(args: list[str], frames: np.ndarray | None) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg is not on PATH")
    p = subprocess.Popen([exe, "-hide_banner", "-loglevel", "error", "-y", *args], stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    try:
        if frames is not None:
            for f in frames:
                p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read().decode(errors="replace")
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()}")


def probe(path: Path) -> dict:
    """ffprobe's reading of an MP4's video stream: codec, size, frame rate, frames, duration."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return {}
    out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,profile,width,height,r_frame_rate,nb_read_frames,pix_fmt:format=duration,"
                          "size", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    s = d["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return dict(codec=s["codec_name"], profile=s.get("profile"), width=int(s["width"]), height=int(s["height"]),
                fps=num / den, frames=int(s["nb_read_frames"]), pix_fmt=s.get("pix_fmt"),
                duration_s=float(d["format"]["duration"]), bytes=int(d["format"]["size"]))


def encode(frames: np.ndarray, path: Path, fps: int = FPS, max_bytes: int = MAX_BYTES, meta: dict | None = None) -> dict:
    """``frames`` (n, h, w, 3) uint8 to an H.264 MP4 under ``max_bytes``: CRF ``CRF`` with the rate capped at
    ``MAXRATE_K`` kbit/s; if that file is larger, two passes at 93 % of the bitrate the size allows, lowered by 15 %
    a try until it fits (at most 4 tries).  Returns the settings that made the file and its size."""
    path = Path(path)
    n, h, w, _ = frames.shape
    src = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    md = []
    for k, v in (meta or {}).items():
        md += ["-metadata", f"{k}={v}"]
    common = ["-an", "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags",
              "+faststart", *md]
    tmp = path.with_suffix(".part.mp4")
    _ffmpeg([*src, *common, "-crf", str(CRF), "-maxrate", f"{MAXRATE_K}k", "-bufsize", f"{2 * MAXRATE_K}k",
             str(tmp)], frames)
    tries = [dict(crf=CRF, maxrate_kbit_s=MAXRATE_K, bytes=tmp.stat().st_size)]
    kbit = 0.93 * max_bytes * 8 / (n / fps) / 1000
    with tempfile.TemporaryDirectory() as td:
        log = str(Path(td) / "pass")
        while tries[-1]["bytes"] > max_bytes and len(tries) < 5:
            rate = f"{int(kbit)}k"
            _ffmpeg([*src, *common, "-b:v", rate, "-pass", "1", "-passlogfile", log, "-f", "mp4", os.devnull], frames)
            _ffmpeg([*src, *common, "-b:v", rate, "-pass", "2", "-passlogfile", log, str(tmp)], frames)
            tries.append(dict(two_pass_kbit_s=int(kbit), bytes=tmp.stat().st_size))
            kbit *= 0.85
    if tries[-1]["bytes"] > max_bytes:
        raise RuntimeError(f"{path.name}: {tries[-1]['bytes']} bytes after {len(tries)} tries (limit {max_bytes})")
    tmp.replace(path)
    return dict(settings=tries[-1], tries=tries, bytes=path.stat().st_size)


def contact_sheet(frames: dict, path: Path) -> None:
    """A 2 x 2 sheet of the frames at ``SHEET_T`` (half size each), PNG."""
    w, h = SIZE
    sheet = Image.new("RGB", (w, h), (0, 0, 0))
    for k, t in enumerate(SHEET_T):
        im = Image.fromarray(frames[t]).resize((w // 2, h // 2), Image.LANCZOS)
        sheet.paste(im, ((k % 2) * (w // 2), (k // 2) * (h // 2)))
    sheet.save(path)


# ------------------------------------------------------------------------------------------------
# Clips
# ------------------------------------------------------------------------------------------------
def clip_name(body: str, speed: str, variant: str, view: str, mode: str) -> str:
    return f"{body}_{speed}_{variant}_{view}" + ("_pointlight" if mode == "pointlight" else "")


_SCENES: dict = {}


def _scene(body: str, out_dir: Path) -> ClipScene:
    key = (body, str(Path(out_dir).resolve()))
    if key not in _SCENES:
        _SCENES[key] = ClipScene(body, out_dir)
    return _SCENES[key]


def render_clip(body: str, speed: str, variant: str, view: str, mode: str, media: Path, out_dir: Path = OUT,
                clip_s: float = CLIP_S, fps: int = FPS, max_bytes: int = MAX_BYTES) -> dict:
    """One clip (module docstring) to ``<media>/<body>/<name>.mp4`` with its poster and contact sheet; returns its
    manifest entry."""
    if view not in CAMERAS or mode not in MODES:
        raise KeyError((view, mode))
    t0 = time.perf_counter()
    entry, arrs = rung(body, speed, variant, out_dir)
    cs = _scene(body, out_dir)
    pb = Playback(arrs)
    ts = times(clip_s, fps)
    box, com_home = cs.home_box(np.asarray(arrs["q_home"], float))
    tr = make_track(pb, box, com_home, ts)
    over = stage0_over(body, speed, variant, out_dir)
    top, bottom = labels(body, speed, variant, view, mode, entry, over)
    frames = np.empty((len(ts), SIZE[1], SIZE[0], 3), np.uint8)
    try:
        for k, t in enumerate(ts):
            cam = camera_pose(tr.lookat(t), CAMERAS[view])
            img = cs.frame(pb.q_at(t), cam, tr.travel(t), mode)
            frames[k] = overlay(img, top, bottom, t, dark=(mode == "pointlight"))
    finally:
        cs.m.light_pos[:] = cs.light_pos0
    # the poster's and the sheet's frames (a clip shorter than their times gives its last frame)
    keep = {tt: frames[min(int(round(tt * fps)), len(ts) - 1)] for tt in (*SHEET_T, POSTER_T)}
    t_render = time.perf_counter() - t0
    name = clip_name(body, speed, variant, view, mode)
    d = Path(media) / body
    d.mkdir(parents=True, exist_ok=True)
    mp4 = d / f"{name}.mp4"
    t1 = time.perf_counter()
    enc = encode(frames, mp4, fps, max_bytes,
                 meta=dict(title=f"{body} {variant} {speed} {view} {mode}",
                           comment=f"{LABEL}; designlab/stage2/clips.py; arrays sha256 {entry['sha256']}"))
    t_enc = time.perf_counter() - t1
    Image.fromarray(keep[POSTER_T]).save(d / f"{name}.png")
    contact_sheet(keep, d / f"{name}_sheet.png")
    sha = hashlib.sha256(mp4.read_bytes()).hexdigest()
    return dict(file=f"{body}/{name}.mp4", poster=f"{body}/{name}.png", sheet=f"{body}/{name}_sheet.png",
                body=body, speed=speed, variant=variant, view=view, mode=mode, features=entry["features"],
                rung_file=entry["file"], rung_sha256=entry["sha256"], bytes=enc["bytes"], sha256=sha,
                encode=dict(enc["settings"], tries=enc["tries"]), probe=probe(mp4), frames=len(ts), fps=fps,
                seconds=clip_s, size=list(SIZE), labels=top + bottom, stage0_over=over,
                camera=dict(CAMERAS[view], fovy_deg=FOVY_DEG), track=asdict(tr),
                window=dict(frames=pb.n, dt_s=pb.dt, window_s=pb.window, advance_m=pb.advance, repeats=clip_s / pb.window),
                model=dict(file=cs.model_check["file"], sha256=cs.model_check["file_sha256"]),
                render_s=t_render, encode_s=t_enc)


def review_set(bodies=look.BODIES) -> list[tuple[str, str, str, str, str]]:
    """The clips the look review (L1) needs per body (``REVIEW``): (body, speed, variant, view, mode)."""
    out = []
    for b in bodies:
        for s in look.SPEEDS:
            for v in REVIEW["both"]:
                for view in CAMERAS:
                    out.append((b, s, v, view, "covered"))
        for v in REVIEW["ladder"]:
            out.append((b, REVIEW["ladder_speed"], v, "visitor", "covered"))
        for v in REVIEW["pointlight"]:
            for view in CAMERAS:
                out.append((b, REVIEW["pointlight_speed"], v, view, "pointlight"))
    return out


def _job(args) -> dict:
    job, media, out_dir = args
    e = render_clip(*job, media=Path(media), out_dir=Path(out_dir))
    print(f"  {e['file']}: {e['bytes'] / 1e6:.2f} MB, render {e['render_s']:.0f} s, encode {e['encode_s']:.0f} s",
          flush=True)
    return e


def _manifest(entries: list, media: Path, prov: dict, seconds: float) -> dict:
    return dict(what="Stage 2 look clips: kinematic playback of the ladder's references (not a simulation)",
                label=LABEL, fps=FPS, seconds=CLIP_S, size=list(SIZE), max_bytes=MAX_BYTES,
                cameras={k: dict(v, fovy_deg=FOVY_DEG) for k, v in CAMERAS.items()}, point_radius_m=POINT_RADIUS_M,
                points=list(look.POINTS), media_dir=str(media), method=(__doc__ or "").split("\n\n")[0],
                provenance=prov, versions=look._versions(), ffmpeg=_ffmpeg_version(), wall_s=seconds,
                render_cpu_s=sum(e["render_s"] + e["encode_s"] for e in entries),
                clips=sorted(entries, key=lambda e: e["file"]))


def _ffmpeg_version() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    return subprocess.run([exe, "-version"], capture_output=True, text=True).stdout.splitlines()[0]


def run(media: Path, jobs=None, procs: int = 1, out_dir: Path = OUT, record: Path | None = None) -> dict:
    """Render ``jobs`` (default ``review_set()``) into ``media``; write ``<media>/clips.json`` and, unless ``record``
    is None, the same manifest to ``record`` (``stage2/look/clips.json``)."""
    import multiprocessing as mp

    media = Path(media)
    media.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs or review_set())
    prov = plant.provenance()
    t0 = time.perf_counter()
    args = [(j, str(media), str(out_dir)) for j in jobs]
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            entries = list(pool.imap_unordered(_job, args, chunksize=1))
    else:
        entries = [_job(a) for a in args]
    doc = _manifest(entries, media, prov, time.perf_counter() - t0)
    path = media / "clips.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old is not None:  # keep the clips this run did not make
        mine = {e["file"] for e in entries}
        doc["clips"] = sorted([e for e in old.get("clips", []) if e["file"] not in mine] + doc["clips"],
                              key=lambda e: e["file"])
    path.write_text(json.dumps(look._plain(doc), indent=1))
    if record is not None:  # the media stay out of git: the record names them relative to the media directory
        rec = dict(doc, media_dir="outside git (the --media directory of the run; files relative to it)")
        Path(record).write_text(json.dumps(look._plain(rec), indent=1))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.clips", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render", help="render the review set (or the named bodies') into --media")
    p.add_argument("--media", required=True)
    p.add_argument("--body", action="append", choices=look.BODIES)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--no-record", action="store_true", help="do not copy the manifest to stage2/look/clips.json")
    o = sub.add_parser("one", help="render one clip into --media")
    o.add_argument("--media", required=True)
    o.add_argument("--body", required=True, choices=look.BODIES)
    o.add_argument("--speed", required=True, choices=look.SPEEDS)
    o.add_argument("--variant", required=True)
    o.add_argument("--view", required=True, choices=tuple(CAMERAS))
    o.add_argument("--mode", default="covered", choices=MODES)
    o.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == "render":
        out = Path(a.out)
        run(Path(a.media), review_set(tuple(a.body or look.BODIES)), a.procs, out,
            None if a.no_record else out / "clips.json")
    else:
        e = render_clip(a.body, a.speed, a.variant, a.view, a.mode, Path(a.media), Path(a.out))
        print(json.dumps(look._plain({k: e[k] for k in ("file", "bytes", "render_s", "encode_s", "encode", "probe")}),
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
