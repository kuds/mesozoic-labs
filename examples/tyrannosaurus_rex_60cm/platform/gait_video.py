"""Render comparable real-time videos from measured poses, including failed runs."""

from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from environments.shared.robot_rendering import camera_basis, font


def cached_geometry(model):
    from environments.shared.robot_rendering import _primitive

    # Static local meshes. Group 5 is installation space and is invisible here.
    meshes = []
    for g in range(model.ngeom):
        if model.geom_group[g] not in (1, 2, 3, 4):
            continue
        material = model.geom_matid[g]
        color = model.mat_rgba[material] if material >= 0 else model.geom_rgba[g]
        if color[3] < 0.5:
            continue
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            triangles = model.mesh_vert[model.mesh_vertadr[mesh] + model.mesh_face[start : start + count]]
        else:
            triangles, _ = _primitive(kind, model.geom_size[g])
        meshes.append((g, triangles, color[:3].copy()))
    return meshes


def geometry_frame(meshes, data):
    tris, colors = [], []
    for g, local, rgb in meshes:
        rot = data.geom_xmat[g].reshape(3, 3)
        faces = local @ rot.T + data.geom_xpos[g]
        tris.append(faces)
        normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normal /= np.maximum(1e-16, np.linalg.norm(normal, axis=1, keepdims=True))
        illumination = 0.46 + 0.54 * np.maximum(0.0, normal @ np.array([0.36, -0.48, 0.8]))
        colors.append((np.clip(np.sqrt(illumination[:, None] * rgb), 0, 1) * 255).astype(np.uint8))
    return np.concatenate(tris), np.concatenate(colors)


def camera_view(faces, colors, center, direction, width, height, scale):
    basis = camera_basis(direction)
    out = Image.new("RGB", (width, height), "#e8eceb")
    pen = ImageDraw.Draw(out)

    def project(points):
        coords = (points - center) @ basis.T
        coords[..., 0] = width * 0.5 + coords[..., 0] * scale
        coords[..., 1] = height * 0.54 - coords[..., 1] * scale
        return coords

    for i in range(-16, 24):
        for line in (((i * 0.1, -1.6, 0), (i * 0.1, 2.4, 0)), ((-1.6, i * 0.1, 0), (2.4, i * 0.1, 0))):
            pts = project(np.array(line))[:, :2].ravel().tolist()
            pen.line(pts, fill="#bec7c5")
    screen = project(faces)
    # Cull subpixel faces and offscreen triangles before painter sorting. This
    # schematic rendering uses the compiled shapes and poses, without OpenGL.
    delta1, delta2 = screen[:, 1, :2] - screen[:, 0, :2], screen[:, 2, :2] - screen[:, 0, :2]
    area = delta1[:, 0] * delta2[:, 1] - delta1[:, 1] * delta2[:, 0]
    indices = np.flatnonzero(np.abs(area) > 0.14)
    ordered = indices[np.argsort(np.mean(screen[indices, :, 2], axis=1))]
    pixels = screen[ordered, :, :2].round().astype("i4").reshape(-1, 6).tolist()
    palette = list(map(tuple, colors[ordered].tolist()))
    for polygon, color in zip(pixels, palette, strict=True):
        pen.polygon(polygon, fill=color)
    return out


def write_video(path, model, qpos, dt, surface, result, *, title="Learned policy"):
    import imageio_ffmpeg

    fps, width, height = 25, 880, 380
    data = mujoco.MjData(model)
    meshes = cached_geometry(model)
    duration = (len(qpos) - 1) * dt
    sink = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        output_params=["-crf", "22", "-movflags", "+faststart"],
        macro_block_size=1,
    )
    sink.send(None)
    try:
        for stamp in np.arange(0.0, duration, 1 / fps):
            data.qpos[:] = qpos[min(len(qpos) - 1, round(stamp / dt))]
            mujoco.mj_forward(model, data)
            faces, colors = geometry_frame(meshes, data)
            xy = data.body("pelvis").xpos[:2]
            center = np.array([xy[0] - 0.04, xy[1], 0.32])
            side = camera_view(faces, colors, center, (0, -1, 0.06), 530, 310, 395)
            front = camera_view(faces, colors, center, (1, -0.02, 0.10), 350, 310, 395)
            frame = Image.new("RGB", (width, height), "#172e2a")
            frame.paste(side, (0, 34))
            frame.paste(front, (530, 34))
            draw = ImageDraw.Draw(frame)
            draw.text(
                (12, 6), title + "   /   " + surface + "   /   seed " + str(result["seed"]), font=font(18), fill="white"
            )
            draw.text((15, 42), "Side", font=font(14), fill="#22332c")
            draw.text((541, 42), "Front", font=font(14), fill="#22332c")
            draw.line((529, 34, 529, 344), fill="#8a9d94")
            draw.text((12, 351), f"{stamp:.2f} s   ·   1×   ·   grid 0.10 m", font=font(16), fill="white")
            labels = {
                "early_end": "early termination",
                "max_transition_accel_p95_mps2": "start / stop acceleration",
                "min_forward_mps": "forward speed",
                "max_tilt_p95_deg": "body tilt",
            }
            first_failure = next(iter(result["failed_checks"]), "")
            status = (
                "PASS" if result["passed"] else "FAIL  /  " + labels.get(first_failure, first_failure.replace("_", " "))
            )
            draw.text((536, 351), status, font=font(16), fill="white")
            sink.send(np.asarray(frame))
    finally:
        sink.close()
