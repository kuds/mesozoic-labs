"""MJCF writer and component ledger for a built ``Design``."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .build import COVER_MATERIALS, Design
from .geometry import combine, grid_triangles, quat_from_matrix, shell_properties, thicken
from .params import size_class

INTEGRATION_RESERVE_FRACTION = 0.06  # PR #662 carried 0.60 kg on 9.90 kg (6 %)
FASTENER_MASS = {60: 0.0025, 75: 0.0030, 100: 0.0045, 120: 0.0055}  # per mount (screw, insert, boss)
SUPPORT_FRACTION = {60: 0.20, 75: 0.24, 100: 0.32, 120: 0.36}  # internal ribs/stringers as fraction of panel mass


def num(a, fmt="{:.6g}") -> str:
    return " ".join(fmt.format(float(v)) for v in np.ravel(a))


def panel_properties(design: Design, panel):
    c = design.cand
    mat = COVER_MATERIALS[panel.material]
    tris = grid_triangles(panel.grid, panel.close_ring)
    m, com, inertia, area = shell_properties(tris, panel.thickness, mat["density"])
    support = SUPPORT_FRACTION[size_class(c.height)] * m
    fasten = FASTENER_MASS[size_class(c.height)] * panel.mounts
    return dict(mass_shell=m, com=com, inertia=inertia, area=area, support=support, fasteners=fasten,
                cost_nom=(m + support) * mat["cost_per_kg"] + 6.0 * panel.mounts,
                cost_high=(m + support) * mat["cost_per_kg"] * 1.6 + 10.0 * panel.mounts)


def ledger(design: Design) -> list[dict]:
    """Flat ledger: one row per item and per panel (shell, support, fasteners)."""
    rows = []
    for b in design.bodies.values():
        for it in b.items:
            rows.append(dict(name=it.name, body=b.name, category=it.category, mass_kg=it.mass,
                             com_body=it.pos.tolist(), inertia_body=it.central_inertia().tolist(),
                             volume_class=it.volume_class, basis=it.basis, source=it.source,
                             cost_nom=it.cost_nom, cost_high=it.cost_high, power_nom_w=it.power_nom,
                             power_max_w=it.power_max, shape=it.shape, size=list(map(float, it.size)), meta=it.meta))
        for p in b.panels:
            pp = panel_properties(design, p)
            rows.append(dict(name=f"cover_{p.name}", body=b.name, category="cover", mass_kg=pp["mass_shell"],
                             com_body=pp["com"].tolist(), inertia_body=pp["inertia"].tolist(), volume_class="C",
                             basis="derived", source=f"{p.material} shell {p.thickness * 1000:.2f} mm, area "
                             f"{pp['area'] * 1e4:.0f} cm2", cost_nom=pp["cost_nom"], cost_high=pp["cost_high"],
                             power_nom_w=0, power_max_w=0, shape="mesh", size=[pp["area"]], meta=dict(region=p.region)))
            rows.append(dict(name=f"cover_{p.name}_support", body=b.name, category="cover_support",
                             mass_kg=pp["support"], com_body=pp["com"].tolist(),
                             inertia_body=(pp["inertia"] * (pp["support"] / max(pp["mass_shell"], 1e-9))).tolist(),
                             volume_class="B", basis="estimate",
                             source=f"internal ribs/stringers {100 * hw_support(design):.0f} % of shell mass",
                             cost_nom=0, cost_high=0, power_nom_w=0, power_max_w=0, shape="none", size=[0], meta={}))
            rows.append(dict(name=f"cover_{p.name}_fasteners", body=b.name, category="fasteners",
                             mass_kg=pp["fasteners"], com_body=pp["com"].tolist(), inertia_body=np.zeros((3, 3)).tolist(),
                             volume_class="B", basis="estimate", source=f"{p.mounts} mounts x captive screw/insert",
                             cost_nom=0, cost_high=0, power_nom_w=0, power_max_w=0, shape="none", size=[0], meta={}))
    return rows


def hw_support(design: Design) -> float:
    return SUPPORT_FRACTION[size_class(design.cand.height)]


def body_inertials(design: Design, rows: list[dict]):
    """Per-body mass, CoM and central inertia; adds the integration reserve to the trunk."""
    per = {}
    for r in rows:
        per.setdefault(r["body"], []).append((r["mass_kg"], np.array(r["com_body"]), np.array(r["inertia_body"])))
    total = sum(r["mass_kg"] for r in rows)
    reserve = INTEGRATION_RESERVE_FRACTION * total / (1 - INTEGRATION_RESERVE_FRACTION)
    out = {}
    for name, parts in per.items():
        out[name] = combine(parts)
    # reserve at the trunk centre of mass, as a 0.1 B box
    m, c, i = out["trunk"]
    B = design.B
    from .geometry import box_inertia

    rres = (reserve, c.copy(), box_inertia(reserve, (0.25 * B, 0.12 * B, 0.12 * B)))
    out["trunk"] = combine([(m, c, i), rres])
    rows.append(dict(name="integration_reserve", body="trunk", category="integration_reserve", mass_kg=reserve,
                     com_body=c.tolist(), inertia_body=rres[2].tolist(), volume_class="B", basis="estimate",
                     source="6 % of installed mass (PR #662 precedent: 0.60 of 9.90 kg)", cost_nom=0, cost_high=0,
                     power_nom_w=0, power_max_w=0, shape="none", size=[0], meta={}))
    return out


def write_mjcf(design: Design, path: Path, rows: list[dict] | None = None, home_qpos=None) -> dict:
    rows = rows if rows is not None else ledger(design)
    inert = body_inertials(design, rows)
    c = design.cand
    root = ET.Element("mujoco", model=f"trex_{c.id}")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="2400", offheight="1600")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.55 0.55 0.55", specular="0.1 0.1 0.1")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", condim="3", friction="0.9 0.005 0.0001")
    ET.SubElement(default, "joint", damping="0.2", frictionloss="0.15")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1=".93 .93 .91", rgb2=".88 .88 .86",
                  width="512", height="512")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="8 8", reflectance="0.0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="1.5 -2.5 4", dir="-0.3 0.5 -1", diffuse="0.6 0.6 0.6", castshadow="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="floor", contype="1",
                  conaffinity="1", group="0")

    elems = {}
    actuated = []
    for name, b in design.bodies.items():
        parent_el = world if b.parent is None else elems[b.parent]
        if b.parent is None:
            pos, R = b.pos_world, b.R_world
        else:
            pb = design.bodies[b.parent]
            pos = pb.R_world.T @ (b.pos_world - pb.pos_world)
            R = pb.R_world.T @ b.R_world
        el = ET.SubElement(parent_el, "body", name=name, pos=num(pos), quat=num(quat_from_matrix(R)))
        elems[name] = el
        m, com, inertia = inert.get(name, (1e-4, np.zeros(3), np.eye(3) * 1e-7))
        inertia = 0.5 * (inertia + inertia.T)
        inertia = inertia + np.eye(3) * 1e-9
        ET.SubElement(el, "inertial", pos=num(com), mass=f"{max(m, 1e-4):.6g}",
                      fullinertia=num([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2],
                                       inertia[1, 2]]))
        j = b.joint
        if j:
            if j["type"] == "free":
                ET.SubElement(el, "freejoint", name=j["name"])
            else:
                arm = joint_armature(design, j["name"])
                ET.SubElement(el, "joint", name=j["name"], type="hinge", axis=num(j["axis"]), range=num(j["range"]),
                              armature=f"{arm:.5g}")
                actuated.append(j["name"])
        for it in b.items:
            if not it.visual or it.shape == "none":
                continue
            q = num(quat_from_matrix(it.rot))
            if it.shape == "box":
                attrs = dict(type="box", size=num(np.array(it.size) / 2), pos=num(it.pos), quat=q)
            elif it.shape == "cylinder":
                attrs = dict(type="cylinder", size=num([it.size[0] / 2, it.size[1] / 2]), pos=num(it.pos), quat=q)
            elif it.shape == "capsule":
                attrs = dict(type="capsule", size=num([it.size[0], it.size[1]]), pos=num(it.pos), quat=q)
            else:
                continue
            contact = it.meta.get("floor_contact", False)
            g = ET.SubElement(el, "geom", name=f"hw_{it.name}", rgba=num(it.rgba),
                              group="0" if contact else "2", **attrs)
            if contact:
                g.set("contype", "1")
                g.set("conaffinity", "1")
                g.set("friction", "0.9 0.02 0.001")
                g.set("solref", "0.01 1")
            g.set("mass", "0")
        for p in b.panels:
            verts, faces = thicken(p.grid, p.thickness, p.close_ring)
            mname = f"mesh_{p.name}"
            ET.SubElement(asset, "mesh", name=mname, vertex=num(verts, "{:.5f}"), face=" ".join(map(str, faces.ravel())),
                          inertia="shell")
            ET.SubElement(el, "geom", name=f"cover_{p.name}", type="mesh", mesh=mname, rgba=num(p.rgba), group="1",
                          mass="0")
            if p.collide:
                from .geometry import convex_pieces

                for k, pts in enumerate(convex_pieces(p.grid, every=2, close_ring=p.close_ring)):
                    cname = f"cl_{p.name}_{k}"
                    ET.SubElement(asset, "mesh", name=cname, vertex=num(pts, "{:.5f}"), inertia="shell")
                    ET.SubElement(el, "geom", name=cname, type="mesh", mesh=cname, group="3", rgba="0 0 0 0", mass="0")
        for s in b.sites:
            attrs = dict(name=s["name"], pos=num(s["pos"]), size=f"{s.get('size', 0.005):.4g}")
            if "rot" in s:
                attrs["quat"] = num(quat_from_matrix(s["rot"]))
            ET.SubElement(el, "site", **attrs)
    # actuators: position servos with planning torque limits (screening only)
    act_el = ET.SubElement(root, "actuator")
    ranges = {b.joint["name"]: b.joint["range"] for b in design.bodies.values() if b.joint and "range" in b.joint}
    for jn in actuated:
        lim, kp = joint_limits(design, jn)
        ET.SubElement(act_el, "position", name=f"{jn}_servo", joint=jn, kp=f"{kp:.4g}", kv=f"{0.05 * kp:.4g}",
                      forcerange=num([-lim, lim]), ctrlrange=num(ranges[jn]))
    sens = ET.SubElement(root, "sensor")
    ET.SubElement(sens, "framequat", name="trunk_quat", objtype="body", objname="trunk")
    if home_qpos is not None:
        kf = ET.SubElement(root, "keyframe")
        ET.SubElement(kf, "key", name="home", qpos=num(home_qpos))
    ET.indent(root)
    text = ET.tostring(root, encoding="unicode")
    path.write_text(text)
    return dict(path=str(path), sha256=hashlib.sha256(text.encode()).hexdigest(), inertials=inert)


def joint_armature(design: Design, joint: str) -> float:
    role = joint_role(joint)
    if role is None:
        return 0.0
    act = design.act(role)
    return max(act.armature, 1e-5)


def joint_role(joint: str) -> str | None:
    if joint == "neck_pitch":
        return "neck"
    if joint == "tail_yaw":
        return "tail"
    for side in ("left_", "right_"):
        if joint.startswith(side):
            j = joint[len(side):]
            return {"hip_roll": "hip_roll", "hip_yaw": "hip_yaw", "hip_pitch": "hip_pitch", "knee": "knee",
                    "ankle_pitch": "ankle_a", "ankle_roll": "ankle_b"}[j]
    return None


def joint_limits(design: Design, joint: str):
    role = joint_role(joint)
    act = design.act(role)
    # Ankle (tandem/side layouts): the two motors drive pitch and roll through a parallel linkage, so the
    # joint-space capability is the diamond |tau_pitch| + |tau_roll| <= 2 x peak.  The per-joint box
    # used here (+-1 x peak on each joint) is the largest box inside that diamond: conservative for one
    # axis alone.  A controller study should model the linkage instead.  Serial layout: one motor each.
    lim = act.peak
    kp = max(5.0, 8.0 * act.rated_torque)
    return lim, kp


def write_ledger(design: Design, rows: list[dict], path: Path, extra: dict | None = None):
    total = sum(r["mass_kg"] for r in rows)
    by_cat, by_class, by_basis = {}, {}, {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["mass_kg"]
        by_class[r["volume_class"]] = by_class.get(r["volume_class"], 0) + r["mass_kg"]
        by_basis[r["basis"]] = by_basis.get(r["basis"], 0) + r["mass_kg"]
    doc = dict(id=design.cand.id, candidate=design.cand.__dict__ | {"overrides": list(design.cand.overrides)},
               B=design.B, H=design.sk.H, selection=design.sel, hip=design.hip, mass_kg=total,
               mass_by_category=by_cat, mass_by_volume_class=by_class, mass_by_basis=by_basis,
               geometry_checks=design.geometry_checks, notes=design.notes, rows=rows)
    if extra:
        doc.update(extra)
    path.write_text(json.dumps(doc, indent=1, default=_json))
    return doc


def _json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(type(x))
