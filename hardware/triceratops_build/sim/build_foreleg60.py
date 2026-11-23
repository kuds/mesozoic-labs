"""The 60 cm model with PR #697's repackaged forelegs, 128 or 160 mm upper link (decision D22). Simulation model.

PR #697 (the owner's; commit 65a44fad, not merged here) repackages the front elbow drive, whose housing overlaps the
upper strut in the 188 mm foreleg (F18), and offers two upper-link lengths: 128 mm (`refinement60/models/refined/
physics.xml`) and 160 mm (`models/mid160/physics.xml`). Both are this build's closed capsule model
(`generated/trike60_C_closed_capsule.xml`) with named elements changed or added:
- the fore-assembly: the roll yoke 40 mm lower on a drop bracket, the upper strut, limb covers, lower strut and foot
  re-placed so the neutral foot stays where it was, the X6 on an offset mount with a standoff, the cranks outboard;
- the open-rib chassis: rib 4 relieved, the shell seams shortened, the two ToF sensors on small arms;
- the reserve of small parts 0.012 kg lighter.

This module keeps that change set as data (`data/foreleg_pr697.json`, written by ``--extract`` from the PR's files,
with their sha256 and the base model's) and applies it to the generated model. Every change records the base
values the PR changed; if the base model no longer has them, ``edit`` refuses rather than overwrite a newer design.
``--check`` compiles each generated variant and the PR's file and compares them field by field.

Usage (from the repository root):
  python hardware/triceratops_build/sim/build_foreleg60.py --extract     # needs PR #697's commit fetched
  python hardware/triceratops_build/sim/build_foreleg60.py --check       # generated variants against the PR's files
The variants are written by ``build_model60.py --capsule-foot --foreleg 128 160``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "data" / "foreleg_pr697.json"
BASE = HERE / "generated" / "trike60_C_closed_capsule.xml"
PR_COMMIT = "65a44fadd72b3cde85d8c5c2516878bd7930435f"
PR_FILES = {128: "hardware/triceratops_build/refinement60/models/refined/physics.xml",
            160: "hardware/triceratops_build/refinement60/models/mid160/physics.xml"}
LENGTHS = (128, 160)


def _named(root) -> dict:
    """(tag, name) -> (element, parent) for every named element."""
    out = {}
    for parent in root.iter():
        for el in parent:
            if el.get("name"):
                out[(el.tag, el.get("name"))] = (el, parent)
    return out


def _pr_text(length: int) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{PR_COMMIT}:{PR_FILES[length]}"], capture_output=True,
                          check=True).stdout


def extract(base: Path = BASE, out: Path = DATA) -> dict:
    """The change set from the base model to each PR file: changed attributes (with the base's values), added named
    elements (with their parent and the named sibling before them). Refuses anything it cannot express."""
    base_bytes = base.read_bytes()
    b_root = ET.fromstring(base_bytes)
    b = _named(b_root)
    res = {"source": {"pr": 697, "commit": PR_COMMIT, "files": {}}, "base": {
        "file": str(base.relative_to(REPO)), "sha256": hashlib.sha256(base_bytes).hexdigest()}, "variants": {}}
    for length in LENGTHS:
        raw = _pr_text(length)
        res["source"]["files"][str(length)] = {"path": PR_FILES[length], "sha256": hashlib.sha256(raw).hexdigest()}
        p_root = ET.fromstring(raw)
        p = _named(p_root)
        missing = [k for k in b if k not in p]
        if missing:
            raise SystemExit(f"the PR's {length} mm model lacks named elements of the base: {missing[:5]}")
        changes, added = [], []
        for k, (el, parent) in p.items():
            if k not in b:
                sib = [c for c in parent if c.get("name")]
                i = sib.index(el)
                added.append({"tag": k[0], "name": k[1], "parent": [parent.tag, parent.get("name")],
                              "after": [sib[i - 1].tag, sib[i - 1].get("name")] if i else None,
                              "attrib": dict(el.attrib)})
                continue
            bel, bpar = b[k]
            if (bpar.tag, bpar.get("name")) != (parent.tag, parent.get("name")):
                raise SystemExit(f"{k} moved to another parent in the PR's {length} mm model")
            keys = sorted(set(bel.attrib) | set(el.attrib))
            diff = {a: [bel.get(a), el.get(a)] for a in keys if bel.get(a) != el.get(a)}
            if diff:
                changes.append({"tag": k[0], "name": k[1], "attrib": diff})
        # unnamed elements must be identical apart from the root's model name
        bu = [(e.tag, sorted(e.attrib.items())) for e in b_root.iter() if not e.get("name") and e is not b_root]
        pu = [(e.tag, sorted(e.attrib.items())) for e in p_root.iter() if not e.get("name") and e is not p_root]
        if bu != pu:
            raise SystemExit(f"unnamed elements differ in the PR's {length} mm model; the change set cannot express it")
        res["variants"][str(length)] = {"model_name": p_root.get("model"), "changed": changes, "added": added}
    out.write_text(json.dumps(res, indent=1) + "\n")
    return res


def edit(length: int, base: Path = BASE, data: Path = DATA) -> str:
    """The base model with the PR's fore-assembly of this length. Refuses if an attribute the PR changed no longer
    has its recorded base value (the base design moved on: re-extract and review)."""
    spec = json.loads(data.read_text())["variants"][str(length)]
    root = ET.parse(base).getroot()
    named = _named(root)
    stale = []
    for ch in spec["changed"]:
        el, _ = named[(ch["tag"], ch["name"])]
        for a, (old, new) in ch["attrib"].items():
            if el.get(a) != old:
                stale.append(f"{ch['name']}.{a}: base {el.get(a)!r}, recorded {old!r}")
    if stale:
        raise SystemExit("the base model differs from the one PR #697's change set was taken from:\n  "
                         + "\n  ".join(stale[:10]))
    for ch in spec["changed"]:
        el, _ = named[(ch["tag"], ch["name"])]
        for a, (_, new) in ch["attrib"].items():
            if new is None:
                el.attrib.pop(a, None)
            else:
                el.set(a, new)
    root.set("model", f"triceratops_60cm_C_closed_capsule_fore{length}")
    for ad in spec["added"]:
        if (ad["tag"], ad["name"]) in named:
            raise SystemExit(f"{ad['name']} already exists in the base model")
        parent = next(e for e in root.iter() if e.tag == ad["parent"][0] and e.get("name") == ad["parent"][1])
        new = ET.Element(ad["tag"], ad["attrib"])
        if ad["after"] is None:
            idx = 0
        else:
            prev = next(c for c in parent if c.tag == ad["after"][0] and c.get("name") == ad["after"][1])
            idx = list(parent).index(prev) + 1
        parent.insert(idx, new)
        new.tail = prev.tail if ad["after"] is not None else parent.text
        named[(ad["tag"], ad["name"])] = (new, parent)
    return ET.tostring(root, encoding="unicode") + "\n"


def mass_notes(length: int, data: Path = DATA) -> dict:
    """Ledger notes for the geoms the change set adds or whose mass it changes."""
    spec = json.loads(data.read_text())["variants"][str(length)]
    notes = {}
    for ad in spec["added"]:
        if "mass" in ad["attrib"]:
            notes[ad["name"]] = f"PR #697 fore-assembly / open-rib part, its mass {ad['attrib']['mass']} kg (EST)"
    for ch in spec["changed"]:
        if "mass" in ch["attrib"]:
            old, new = ch["attrib"]["mass"]
            notes[ch["name"]] = f"PR #697: {old} -> {new} kg (EST)"
    return notes


def check(generated: Path = HERE / "generated") -> list[str]:
    """Each generated variant against the PR's compiled model: every model array that describes the robot."""
    import mujoco
    import numpy as np

    problems = []
    fields = ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_pos", "jnt_axis",
              "jnt_range", "jnt_stiffness", "dof_damping", "dof_armature", "geom_type", "geom_size", "geom_pos",
              "geom_quat", "geom_contype", "geom_conaffinity", "geom_friction", "actuator_gainprm",
              "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange", "key_qpos", "site_pos", "site_quat")
    for length in LENGTHS:
        mine = mujoco.MjModel.from_xml_path(str(generated / f"trike60_C_closed_capsule_fore{length}.xml"))
        tmp = generated / f".pr697_{length}_check.xml"
        tmp.write_bytes(_pr_text(length))
        try:
            theirs = mujoco.MjModel.from_xml_path(str(tmp))
        finally:
            tmp.unlink()
        names = lambda m, kind, n: [getattr(m, kind)(i).name for i in range(n)]  # noqa: E731
        if names(mine, "geom", mine.ngeom) != names(theirs, "geom", theirs.ngeom):
            problems.append(f"{length} mm: geom order or names differ")
            continue
        for f in fields:
            a, b = getattr(mine, f), getattr(theirs, f)
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-9):
                problems.append(f"{length} mm: {f} differs (largest {np.max(np.abs(a - b)) if a.shape == b.shape else 'shape'})")
    return problems


def packaging(model_path: Path) -> dict:
    """Signed distances (m) from each front X6 housing to its upper strut (the two share a body, so one pose
    suffices) and, over the shoulder and elbow ranges, its least distance to any chassis geom."""
    import itertools

    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    out = {}
    ft = np.zeros(6)
    chassis = [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("chassis").id and m.geom_type[i] != 0]
    for c in ("left_fore", "right_fore"):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        x6, strut = m.geom(f"{c}_x6_flexion_stator").id, m.geom(f"{c}_upper_strut").id
        to_strut = float(mujoco.mj_geomDistance(m, d, x6, strut, 0.2, ft))
        joints = [m.joint(f"{c}_{j}") for j in ("roll", "pitch", "flexion")]
        grids = [np.linspace(j.range[0], j.range[1], 7) for j in joints]
        least = (np.inf, None)
        for q in itertools.product(*grids):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            for j, v in zip(joints, q):
                d.qpos[j.qposadr[0]] = v
            mujoco.mj_kinematics(m, d)
            for g in chassis:
                dist = float(mujoco.mj_geomDistance(m, d, x6, g, 0.2, ft))
                if dist < least[0]:
                    least = (dist, m.geom(g).name)
        out[c] = {"x6_to_upper_strut_m": to_strut, "x6_to_chassis_min_m": least[0], "nearest_chassis_geom": least[1],
                  "grid": "7 values over each of roll, pitch and flexion ranges"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--packaging", nargs="*", default=None, help="model files (default: the generated variants and "
                                                                 "the 188 mm capsule model)")
    a = ap.parse_args(argv)
    if a.extract:
        r = extract()
        for k, v in r["variants"].items():
            print(f"{k} mm: {len(v['changed'])} elements changed, {len(v['added'])} added")
        print("wrote", DATA)
    if a.check:
        p = check()
        print("check: " + ("the generated variants match PR #697's models in every compared field" if not p else
                           "PROBLEMS\n  " + "\n  ".join(p)))
        if p:
            return 1
    if a.packaging is not None:
        files = [Path(f) for f in a.packaging] or [BASE] + [HERE / "generated" / f"trike60_C_closed_capsule_fore{n}.xml"
                                                           for n in LENGTHS]
        for f in files:
            r = packaging(f)
            print(f.name, {c: {k.replace("_m", "_mm"): (round(v * 1000, 1) if isinstance(v, float) else v)
                               for k, v in x.items() if k != "grid"} for c, x in r.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
