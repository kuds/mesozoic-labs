"""One closed-loop clip from a monitor trace (round C design 2.1, 3.9; the launcher starts it, the trainer never
waits for it).

    python -m designlab.stage2.clip2 --trace PATH --body B --set S --out clip_<n>.mp4 [--seconds 3] [--view visitor]
                                     [--strict]

The picture is a *view* of one simulated episode: the 50 Hz trace a monitor episode left (``panels.TRACE_RULE``,
``bar.RULES["trace"]``: the joint angles, the trunk's roll, pitch and yaw, the root's place and both feet's lowest
points), turned into poses by ``barviews.pose_of`` (its bound: ``barviews.POSE_BOUND``) and drawn with the covered
model from a camera that follows the root (``barviews.follow_camera``, ``clips.CAMERAS``).  It shows the last
``--seconds`` of the trace, so a fall is in the picture when there was one.  Every frame says that it is a
simulation drawn from a trace.  It judges nothing and changes no record.

Beside the clip the record ``<out stem>.json`` (``clip_<n>.json`` for ``clip_<n>.mp4``) is written in every case:
with the trace's and the clip's sha256, the modules that drew it and the span shown, or with the reason why no clip
was made (no trace, no renderer, no ``ffmpeg``).  The exit code is 0 either way unless ``--strict`` (then 1 when no
clip was made): a clip that fails must not fail a launcher.

This process runs with ``MUJOCO_GL=osmesa`` (assigned below, before any other import: a child of a ``disable``
process inherits ``disable`` otherwise) and imports no torch.  The encoder is the ``ffmpeg`` on ``PATH``, as
``designlab/stage2/clips.py`` uses it.
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("osmesa")

import os  # noqa: E402
import sys  # noqa: E402

os.environ.setdefault("GIT_OPTIONAL_LOCKS", "0")  # (as in train2: a killed ``git status`` leaves no index.lock)

if __name__ == "__main__":  # before numpy: one core; one llvmpipe thread; no shader cache under the home directory
    for _v in tbenv.THREAD_VARS + ("LP_NUM_THREADS",):
        os.environ.setdefault(_v, "1")
    for _v in ("MESA_SHADER_CACHE_DISABLE", "MESA_GLSL_CACHE_DISABLE"):
        os.environ.setdefault(_v, "true")

import argparse  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage2 import s2io  # noqa: E402

SCHEMA = "s2-clip-1"
FPS = 30
SECONDS = 3.0
VIEW = "visitor"
#: On every frame.
LABEL = "SIMULATION (physics): a monitor episode, drawn from its 50 Hz trace"
#: The look-at point follows the root's place averaged over this long (s), so that the stride's sway stays in the
#: picture.
LOOK_SMOOTH_S = 1.0
RULE = (
    "a clip is a view of one monitor episode: its trace's last --seconds, the poses rebuilt by barviews.pose_of "
    "(attitude and root place exact, the root's height from the feet's lowest points, the joint angles moved back "
    "by one physics step: barviews.POSE_BOUND), the covered model from clips.CAMERAS' camera on the root's place "
    f"averaged over {LOOK_SMOOTH_S:g} s, {FPS} frames a second, the label on every frame; it judges nothing")


def record_path(out) -> Path:
    """``clip_<n>.json`` beside ``clip_<n>.mp4``."""
    return Path(out).with_suffix(".json")


def _smooth(x: np.ndarray, rows: int) -> np.ndarray:
    """A centred moving average over ``rows`` rows (shorter at the ends)."""
    rows = max(1, int(rows))
    if rows == 1 or len(x) < 2:
        return x.copy()
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.empty_like(x, dtype=float)
    half = rows // 2
    for i in range(len(x)):
        a, b = max(0, i - half), min(len(x), i + half + 1)
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def render(trace, body: str, set_name: str, out, seconds: float = SECONDS, view: str = VIEW) -> dict:
    """Draw and encode the clip; what the record holds of it.  Raises on anything that stops it (``clip`` turns
    that into the record's reason)."""
    from designlab.stage2 import barviews, clips  # (late: they import mujoco, whose backend is chosen above)

    trace, out = Path(trace), Path(out)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH")
    if view not in clips.CAMERAS:
        raise ValueError(f"view {view!r}: choose from {', '.join(clips.CAMERAS)}")
    with np.load(trace, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    missing = [k for k in ("t", "t_ref", "rows", "cols", "q") if k not in arrays]
    if missing:
        raise ValueError(f"{trace.name} is no episode trace: it lacks {', '.join(missing)}")
    if len(arrays["t"]) < 2:
        raise ValueError(f"{trace.name} holds {len(arrays['t'])} rows: nothing to show")
    model = barviews.body_model(body)
    pose = barviews.pose_of(arrays, model)
    t = pose["t"]
    t1 = float(t[-1])
    t0 = max(float(t[0]), t1 - float(seconds))
    n = int(round((t1 - t0) * FPS)) + 1
    times = np.minimum(t0 + np.arange(n) / FPS, t1)
    cs = barviews.scene(body)
    box, _ = cs.home_box(model.q_home)
    z_look, psi = float(box[2]), model.psi_cmd
    every = float(np.median(np.diff(t)))
    look = _smooth(pose["root_xy"], round(LOOK_SMOOTH_S / every))
    lxy = np.column_stack([np.interp(times, t, look[:, 0]), np.interp(times, t, look[:, 1])])
    top = [LABEL, f"{body}, set {set_name}: {trace.parent.parent.name}/{trace.parent.name}, seed {trace.stem}"]
    bottom = [clips.CAMERAS[view]["text"] + "; the camera follows the root"]
    frames = np.empty((n, clips.SIZE[1], clips.SIZE[0], 3), np.uint8)
    t_draw = time.perf_counter()
    try:
        for k in range(n):
            cam = barviews.follow_camera(view, lxy[k], z_look, psi, clips.CAMERAS[view])
            raw = barviews.draw_frame(cs, barviews.q_at(pose, float(times[k])), cam, lxy[k] - box[:2])
            frames[k] = barviews.draw_labels(raw, top, bottom, "t = %.2f s" % times[k])
    finally:
        cs.m.light_pos[:] = cs.light_pos0
    render_s = time.perf_counter() - t_draw
    out.parent.mkdir(parents=True, exist_ok=True)
    t_enc = time.perf_counter()
    enc = clips.encode(frames, out, FPS, clips.MAX_BYTES, meta=dict(comment=LABEL))
    return dict(clip=dict(file=out.name, bytes=int(enc["bytes"]), sha256=s2io.sha256_file(out), frames=n, fps=FPS,
                          size=list(clips.SIZE), encode=enc["settings"]),
                span=dict(t0_s=t0, t1_s=t1, seconds=t1 - t0, rows=int(len(t)), trace_t0_s=float(t[0])),
                view=view, camera=dict(clips.CAMERAS[view], lookat_height_m=z_look, follows="the root's place, "
                                       f"averaged over {LOOK_SMOOTH_S:g} s"),
                labels=dict(top=top, bottom=bottom), pose_residual_m=pose["residual_m"],
                pose_bound=barviews.POSE_BOUND, model=cs.model_check, render_s=render_s,
                encode_s=time.perf_counter() - t_enc)


def clip(trace, body: str, set_name: str, out, seconds: float = SECONDS, view: str = VIEW) -> dict:
    """``render`` and the record beside the clip (returned): ``ok`` with the clip's and the trace's hashes, or
    ``ok`` False with ``reason``.  Never raises for a clip that cannot be made."""
    trace, out = Path(trace), Path(out)
    doc = dict(schema=SCHEMA, ok=False, reason=None, rule=RULE, label=LABEL, body=body, set=set_name,
               trace=dict(path=s2io.portable(str(trace.resolve())), name=trace.name,
                          sha256=s2io.sha256_file(trace) if trace.is_file() else None),
               seconds_asked=float(seconds), mujoco_gl=os.environ.get("MUJOCO_GL"),
               ffmpeg=shutil.which("ffmpeg"), torch_imported="torch" in sys.modules)
    try:
        if not trace.is_file():
            raise FileNotFoundError(f"no trace {trace}")
        doc.update(render(trace, body, set_name, out, seconds, view), ok=True)
    except Exception as exc:  # noqa: BLE001 - a clip that fails is recorded, never raised (the launcher goes on)
        doc.update(ok=False, reason=f"{type(exc).__name__}: {exc}")
        for stale in (out, out.with_suffix(".part.mp4")):
            if stale.is_file():
                stale.unlink()
    doc.update(modules={s2io.portable(str(s2io.module_path(m))): s2io.sha256_file(s2io.module_path(m))
                        for m in ("clip2.py", "barviews.py", "clips.py")},
               provenance=s2io.git_state(own_outputs=[str(out.parent.resolve())]), at=s2io.now_utc())
    doc = s2io.plain(doc)
    doc["sha256"] = s2io.sha256_json(doc)
    s2io.write_json(record_path(out), doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.clip2", description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", required=True, help="a monitor episode's trace (.npz)")
    ap.add_argument("--body", required=True)
    ap.add_argument("--set", required=True, help="the run's set (for the label and the record)")
    ap.add_argument("--out", required=True, help="the clip (.mp4); its record is written beside it (.json)")
    ap.add_argument("--seconds", type=float, default=SECONDS, help="the trace's last seconds to show")
    ap.add_argument("--view", default=VIEW)
    ap.add_argument("--strict", action="store_true", help="exit 1 when no clip was made")
    args = ap.parse_args(argv)
    doc = clip(args.trace, args.body, args.set, args.out, args.seconds, args.view)
    if doc["ok"]:
        print(f"clip {args.out}: {doc['clip']['frames']} frames, {doc['clip']['bytes']} bytes "
              f"({doc['span']['seconds']:.2f} s of the trace)", flush=True)
        return 0
    print(f"clip {args.out}: not made: {doc['reason']} (recorded in {record_path(args.out)})", flush=True)
    return 1 if args.strict else 0


__all__ = ["FPS", "LABEL", "RULE", "SCHEMA", "SECONDS", "clip", "main", "record_path", "render"]

if __name__ == "__main__":
    from designlab.stage2 import clip2 as _named  # the importable copy owns all module state

    sys.exit(_named.main())
