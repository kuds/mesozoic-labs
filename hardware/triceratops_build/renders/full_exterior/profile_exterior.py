"""Compare orthographic silhouettes at exactly the canonical model's home pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from build_exterior import SOURCE
from PIL import Image, ImageDraw, ImageFilter
from render_exterior import font
from scipy.spatial import ConvexHull

WIDTH, HEIGHT = 1650, 900
SCALE = 915
FLOOR = 695
X0 = 655
BG = (243, 242, 232)
INK = (40, 51, 44)
GREEN = (51, 90, 73)
GOLD = (174, 109, 47)


def pixel(xz):
    x, z = xz
    return (int(X0 + x * SCALE), int(FLOOR - z * SCALE))


def points(m, d, g):
    geom = m.geom_type[g]
    size = m.geom_size[g]
    if geom == mujoco.mjtGeom.mjGEOM_MESH:
        mi = m.geom_dataid[g]
        start, n = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
        verts = m.mesh_vert[start : start + n]
        f, n = m.mesh_faceadr[mi], m.mesh_facenum[mi]
        triangles = m.mesh_face[f : f + n]
    elif geom == mujoco.mjtGeom.mjGEOM_BOX:
        verts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
        triangles = None
    else:
        a, b = np.mgrid[0 : 2 * np.pi : 40j, -np.pi / 2 : np.pi / 2 : 19j]
        circ = np.stack([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)], axis=-1).reshape(-1, 3)
        if geom in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            r, h = size[:2]
            if geom == mujoco.mjtGeom.mjGEOM_CAPSULE:
                verts = circ * r
                verts[:, 2] += np.sign(verts[:, 2]) * h
            else:
                phi = np.linspace(0, 2 * np.pi, 41)
                verts = np.concatenate(
                    [np.column_stack([np.cos(phi) * r, np.sin(phi) * r, np.full_like(phi, z)]) for z in (-h, h)]
                )
        else:
            verts = circ * size
        triangles = None
    verts = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return verts, triangles


def load(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def masks(m, d, kind):
    img = Image.new("L", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    clouds = {}
    for i in range(m.ngeom):
        name = m.geom(i).name
        show = name.startswith("exterior_") if kind == "skin" else m.geom_group[i] in (1, 2) and name != "floor"
        if not show or m.geom_rgba[i, 3] <= 0:
            continue
        v, f = points(m, d, i)
        clouds[name] = v
        projected = v[:, [0, 2]]
        pix = np.asarray([pixel(p) for p in projected])
        if f is not None:
            for tri in f:
                draw.polygon([tuple(p) for p in pix[tri]], fill=255)
        else:
            hull = ConvexHull(projected)
            draw.polygon([tuple(pix[j]) for j in hull.vertices], fill=255)
    return img, clouds


def outline(mask):
    outer = np.asarray(mask.filter(ImageFilter.MaxFilter(5))).astype(int)
    inner = np.asarray(mask.filter(ImageFilter.MinFilter(3))).astype(int)
    return Image.fromarray(np.uint8(np.clip(outer - inner, 0, 255)))


def metrics(cloud):
    def ext(*names):
        v = np.concatenate([cloud["exterior_" + n] for n in names])
        return np.array([v.min(axis=0), v.max(axis=0)])

    frill = ext("frill")
    body = ext("torso")
    tail = ext("tail")
    beak = ext("upper_beak", "mandible_tip")
    horns = ext("left_horn", "right_horn")
    return {
        "torso_max_width_mm": 1000 * (body[1, 1] - body[0, 1]),
        "torso_max_vertical_extent_mm": 1000 * (body[1, 2] - body[0, 2]),
        "frill_max_width_mm": 1000 * (frill[1, 1] - frill[0, 1]),
        "frill_vertical_extent_mm": 1000 * (frill[1, 2] - frill[0, 2]),
        "frill_fore_aft_extent_mm": 1000 * (frill[1, 0] - frill[0, 0]),
        "frill_crown_floor_mm": 1000 * frill[1, 2],
        "beak_to_tail_horizontal_mm": 1000 * (beak[1, 0] - tail[0, 0]),
        "horn_tip_to_tail_horizontal_mm": 1000 * (horns[1, 0] - tail[0, 0]),
        "frill_back_to_beak_horizontal_mm": 1000 * (beak[1, 0] - frill[0, 0]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--before", type=Path, help="optional previous exterior, at the same home pose")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m, d = load(args.model)
    after, cloud = masks(m, d, "skin")
    stats = metrics(cloud)
    m0, d0 = load(args.source)
    before = None
    prior = {}
    if args.before:
        bm, bd = load(args.before)
        before, bc = masks(bm, bd, "skin")
        prior = metrics(bc)
    else:
        reference = Path(__file__).resolve().parent
        before = Image.open(reference / "initial_profile.png").convert("L")
        prior = json.loads((reference / "initial_profile_metrics.json").read_text())
        if before.size != (WIDTH, HEIGHT):
            raise ValueError("Reference silhouette scale differs")
    mechanical, _ = masks(m0, d0, "frame")
    plate = Image.new("RGB", (WIDTH, HEIGHT), BG)
    if before is not None:
        plate.paste((228, 210, 177), mask=before)
    plate.paste((193, 211, 189), mask=after)
    plate.paste((153, 170, 147), mask=mechanical.point(lambda v: int(v * 0.50)))
    if before is not None:
        plate.paste(GOLD, mask=outline(before))
    plate.paste(GREEN, mask=outline(after))
    draw = ImageDraw.Draw(plate)
    draw.text((43, 27), "Triceratops  /  Frame and outer profile", font=font(43, True), fill=INK)
    draw.text((46, 88), "60 cm Option C  •  Identical home pose  •  Orthographic side view", font=font(23), fill=INK)
    draw.line((43, 130, WIDTH - 43, 130), fill=(175, 183, 165), width=2)
    draw.line((43, FLOOR + 2, 1196, FLOOR + 2), fill=(148, 161, 146), width=2)

    def joint(tag):
        p = d0.xanchor[m0.joint(tag).id]
        return pixel(p[[0, 2]])

    for limb, pt, offset in (("hind", "Hip", (-24, -53)), ("fore", "Shoulder", (-50, -56))):
        tags = ["left_" + limb + "_pitch", "left_" + limb + "_flexion", "left_" + limb + "_ankle"]
        ps = [joint(tag) for tag in tags]
        draw.line(ps, fill=(51, 71, 64), width=6)
        for x, y in ps:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(244, 241, 222), outline=(51, 71, 64), width=3)
        x, y = ps[0]
        draw.text((x + offset[0], y + offset[1]), pt, font=font(18, True), fill=INK)
        draw.line((x, y - 9, x, y - 29), fill=INK, width=2)

    def leg(x, clr, txt):
        draw.line((x, 755, x + 40, 755), fill=clr, width=5)
        draw.text((x + 52, 742), txt, font=font(18), fill=INK)

    lx = 58
    if before is not None:
        leg(lx, GOLD, "First padded concept")
        lx += 265
    leg(lx, GREEN, "Revised cover")
    leg(lx + 237, (51, 71, 64), "Fixed leg axes")
    draw.line((1218, 150, 1218, 765), fill=(176, 186, 168), width=2)
    draw.text((1243, 169), "PROFILE MEASURES", font=font(20, True), fill=GREEN)
    y = 235
    for name, k in [
        ("Maximum torso width", "torso_max_width_mm"),
        ("Frill fore–aft depth", "frill_fore_aft_extent_mm"),
        ("Beak to tail", "beak_to_tail_horizontal_mm"),
        ("Frill width", "frill_max_width_mm"),
    ]:
        draw.text((1243, y), name, font=font(18), fill=INK)
        text = (f"{prior[k]:.0f} → " if prior else "") + f"{stats[k]:.0f} mm"
        draw.text((1243, y + 27), text, font=font(29, True), fill=GREEN)
        y += 94
    draw.text((1243, y + 8), "The limb pivots stay fixed.", font=font(17, True), fill=INK)
    draw.text(
        (43, 815),
        "The first covering widened the ribcage. The revised skin tapers at the waist and wraps each drive locally.",
        font=font(22),
        fill=INK,
    )
    draw.text(
        (43, 851),
        "The deep folded links and shoulder–hip spacing belong to the robot. A coat cannot move those pivots.",
        font=font(19),
        fill=(76, 89, 78),
    )
    plate.save(args.out / "frame-and-profile.png")
    axes = {}
    for j in ["left_fore_pitch", "left_hind_pitch", "left_fore_flexion", "left_fore_ankle"]:
        axes[j + "_mm"] = (d0.xanchor[m0.joint(j).id] * 1000).round(3).tolist()
    result = {
        "reference_pose": "MJCF home, forward kinematics, no settling",
        "initial": prior,
        "revised": stats,
        "axes": axes,
    }
    (args.out / "profile-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
