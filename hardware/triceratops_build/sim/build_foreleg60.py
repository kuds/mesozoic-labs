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
        res["source"]["files"][str(length)] = {"path": PR_FILES[length], "sha256": hashlib.sha256(raw).hexdigest(),
                                               "canonical_sha256": _canonical_sha(raw)}
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


def _canonical_sha(text) -> str:
    """sha256 of the XML's canonical form (attribute order and whitespace between elements do not count)."""
    return hashlib.sha256(ET.canonicalize(xml_data=text, strip_text=True).encode()).hexdigest()


def _fromto_len(el) -> float:
    v = [float(x) for x in el.get("fromto").split()]
    return ((v[3] - v[0]) ** 2 + (v[4] - v[1]) ** 2 + (v[5] - v[2]) ** 2) ** 0.5


CONSISTENT = ("upper_strut", "upper_limb_cover", "pushrod")


def consistent_masses(root, base_root) -> list[dict]:
    """A recorded deviation from PR #697 (the review of 23 Nov): the 128 mm model keeps the 188 mm parts' masses for
    the upper strut, upper limb cover and pushrod, and the 160 mm model scales those by its own length ratios to
    128 mm, so its strut came out heavier than the 188 mm one. Here each of the three is the 188 mm part's mass times
    its length over the 188 mm part's length (same section). Edits ``root`` in place; returns what changed."""
    b = _named(base_root)
    out = []
    for c in ("left_fore", "right_fore"):
        for part in CONSISTENT:
            name = f"{c}_{part}"
            el, _ = _named(root)[("geom", name)]
            bel, _ = b[("geom", name)]
            ratio = _fromto_len(el) / _fromto_len(bel)
            mass = float(bel.get("mass")) * ratio
            out.append({"name": name, "pr_kg": float(el.get("mass")), "kg": mass, "length_ratio_to_188": ratio})
            el.set("mass", f"{mass:.6f}")
    return out


def _apply(spec, root, length):
    named = _named(root)
    for ch in spec["changed"]:
        el, _ = named[(ch["tag"], ch["name"])]
        for a, (_, new) in ch["attrib"].items():
            if new is None:
                el.attrib.pop(a, None)
            else:
                el.set(a, new)
    for ad in spec["added"]:
        if (ad["tag"], ad["name"]) in named:
            raise SystemExit(f"{ad['name']} already exists in the base model")
        parents = [e for e in root.iter() if e.tag == ad["parent"][0] and e.get("name") == ad["parent"][1]]
        if len(parents) != 1:
            raise SystemExit(f"{ad['name']}: its parent {ad['parent']} is not in the base model once")
        parent = parents[0]
        new = ET.Element(ad["tag"], ad["attrib"])
        if ad["after"] is None:
            idx, tail = 0, parent.text
        else:
            prev = [c for c in parent if c.tag == ad["after"][0] and c.get("name") == ad["after"][1]]
            if len(prev) != 1:
                raise SystemExit(f"{ad['name']}: its anchor {ad['after']} is not in {ad['parent']} once")
            idx, tail = list(parent).index(prev[0]) + 1, prev[0].tail
        parent.insert(idx, new)
        new.tail = tail
        named[(ad["tag"], ad["name"])] = (new, parent)


def edit(length: int, base: Path = BASE, data: Path = DATA, deviations: bool = True) -> str:
    """The base model with the PR's fore-assembly of this length, then (``deviations``) the consistent part masses.
    Refuses unless the base model is byte for byte the one the change set was taken from (its sha256 in the data):
    a base that moved on needs ``--extract`` again and a review of the new change set."""
    d = json.loads(data.read_text())
    spec = d["variants"][str(length)]
    raw = Path(base).read_bytes()
    if hashlib.sha256(raw).hexdigest() != d["base"]["sha256"]:
        raise SystemExit(f"the base model {Path(base).name} is not the one PR #697's change set was taken from "
                         f"(sha256 {d['base']['sha256'][:12]}...): re-run --extract and review the change set")
    root = ET.fromstring(raw)
    base_root = ET.fromstring(raw)
    _apply(spec, root, length)
    if deviations:
        consistent_masses(root, base_root)
        root.set("model", f"triceratops_60cm_C_closed_capsule_fore{length}")
    else:
        root.set("model", spec["model_name"])  # the PR's own name: reproduces its file (see check)
    return ET.tostring(root, encoding="unicode") + "\n"


def deviations(length: int, base: Path = BASE, data: Path = DATA) -> list[dict]:
    root = ET.fromstring(edit(length, base, data, deviations=False))
    return consistent_masses(root, ET.parse(base).getroot())


def mass_notes(length: int, data: Path = DATA) -> dict:
    """Ledger notes for the geoms the change set adds or whose mass it changes, and for the consistent masses."""
    spec = json.loads(data.read_text())["variants"][str(length)]
    notes = {}
    for ad in spec["added"]:
        if "mass" in ad["attrib"]:
            notes[ad["name"]] = (f"PR #697 part as its author estimated it, {ad['attrib']['mass']} kg (EST); not grown "
                                 "by the closed variant's reserve factor, unlike the generator's own EST parts")
    for ch in spec["changed"]:
        if "mass" in ch["attrib"]:
            old, new = ch["attrib"]["mass"]
            notes[ch["name"]] = f"PR #697 changed it from {old} to {new} kg (EST)"
    for dv in deviations(length, data=data):
        notes[dv["name"]] = (f"the 188 mm part's mass x its length ratio {dv['length_ratio_to_188']:.4f} (same section; "
                             f"a recorded deviation from PR #697's {dv['pr_kg']:.6f} kg, EST)")
    return notes


def _model_arrays(m):
    import numpy as np

    skip = ("names", "paths", "names_map", "text_data", "signature")
    for a in dir(m):
        if a.startswith("_") or a.startswith("name_") or a.startswith("path") or a in skip:
            continue
        v = getattr(m, a, None)
        if isinstance(v, np.ndarray):
            yield a, v


def check(generated: Path = HERE / "generated") -> list[str]:
    """Each generated variant against the PR's model with the same recorded deviation applied: every compiled array
    of the model (names and their addresses aside) and every option, so only the deviation may differ from the PR."""
    import mujoco
    import numpy as np

    problems = []
    base_root = ET.parse(BASE).getroot()
    for length in LENGTHS:
        mine = mujoco.MjModel.from_xml_path(str(generated / f"trike60_C_closed_capsule_fore{length}.xml"))
        pr_root = ET.fromstring(_pr_text(length))
        consistent_masses(pr_root, base_root)
        theirs = mujoco.MjModel.from_xml_string(ET.tostring(pr_root, encoding="unicode"))
        if [mine.geom(i).name for i in range(mine.ngeom)] != [theirs.geom(i).name for i in range(theirs.ngeom)]:
            problems.append(f"{length} mm: geom order or names differ")
            continue
        n = 0
        for a, v in _model_arrays(mine):
            w = getattr(theirs, a)
            n += 1
            if v.shape != w.shape or not np.array_equal(v, w):
                problems.append(f"{length} mm: {a} differs")
        for a in dir(mine.opt):
            if not a.startswith("_"):
                x, y = getattr(mine.opt, a), getattr(theirs.opt, a)
                if isinstance(x, (int, float, np.ndarray)) and not np.array_equal(x, y):
                    problems.append(f"{length} mm: option {a} differs")
        if n < 300:
            problems.append(f"{length} mm: only {n} arrays compared")
    return problems


def reproduces_pr(length: int, data: Path = DATA) -> bool:
    """Without the PR's commit: the change set, without the deviation, gives the PR's file (canonical XML sha256)."""
    d = json.loads(data.read_text())
    return _canonical_sha(edit(length, deviations=False, data=data)) == d["source"]["files"][str(length)][
        "canonical_sha256"]


LEG_BODIES = ("roll_yoke", "upper_link", "lower_link")
ZERO_RETRIES = []  # (pair, pose) where mj_geomDistance returned exactly 0 even after the retry (reported, never hidden)


def _dist(mujoco, m, d, g, h, distmax, ft, joints=None, pose=None):
    """mj_geomDistance, with its occasional exact 0 re-measured: on 23 Nov it read 0.0 for the X6 against the torso
    ellipsoid at isolated grid poses where every neighbour read about 40 mm and an implicit-surface test put the
    housing outside; a wider limit did not help. The retry nudges the leg's joints by 1e-4 rad (0.006 deg), which a
    real overlap survives; a 0 that survives the nudge is kept and recorded in ZERO_RETRIES."""
    v = float(mujoco.mj_geomDistance(m, d, g, h, distmax, ft))
    if v == 0.0 and joints is not None:
        for j in joints:
            d.qpos[j.qposadr[0]] += 1e-4
        mujoco.mj_kinematics(m, d)
        v = float(mujoco.mj_geomDistance(m, d, g, h, distmax, ft))
        for j in joints:
            d.qpos[j.qposadr[0]] -= 1e-4
        mujoco.mj_kinematics(m, d)
        if v == 0.0:
            ZERO_RETRIES.append((m.geom(g).name, m.geom(h).name, pose))
    return v


def packaging(model_path: Path, steps: int = 9) -> dict:
    """Signed distances (m), over a grid of ``steps`` values across each of the roll, pitch and flexion ranges:
    - ``pairs``: every geom on the front roll yoke, upper link and lower link against every chassis geom, the least
      distance of each pair (only pairs within 20 mm are kept);
    - the PR's named clearances that the physics model carries: X6 to upper strut (same body), X6 to any chassis
      geom, the offset motor plate and its standoff to any chassis geom, X6 to the output crank (across the elbow).
    Not here: the moving pushrod (the model's rod is a fixed capsule), the dressed meshes' clearances (rod to guard,
    X6 to the sculpted rib): PR #697's own evidence gives those."""
    import itertools

    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    ft = np.zeros(6)
    chassis = [i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("chassis").id and m.geom_type[i] != 0]
    out = {}
    for c in ("left_fore", "right_fore"):
        bodies = {m.body(f"{c}_{b}").id for b in LEG_BODIES}
        leg = [i for i in range(m.ngeom) if m.geom_bodyid[i] in bodies]
        joints = [m.joint(f"{c}_{j}") for j in ("roll", "pitch", "flexion")]
        grids = [np.linspace(j.range[0], j.range[1], steps) for j in joints]
        least = {}
        name = lambda i: m.geom(i).name  # noqa: E731
        x6 = m.geom(f"{c}_x6_flexion_stator").id
        crank = m.geom(f"{c}_output_crank").id
        named = {"x6_to_upper_strut": (x6, [m.geom(f"{c}_upper_strut").id]), "x6_to_chassis": (x6, chassis),
                 "x6_to_output_crank": (x6, [crank])}
        for part in ("offset_motor_mount", "mount_standoff"):
            if f"{c}_{part}" in [m.geom(i).name for i in range(m.ngeom)]:
                named[f"{part}_to_chassis"] = (m.geom(f"{c}_{part}").id, chassis)
        best = {k: (np.inf, None) for k in named}
        for q in itertools.product(*grids):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            for j, v in zip(joints, q):
                d.qpos[j.qposadr[0]] = v
            mujoco.mj_kinematics(m, d)
            for g in leg:
                for h in chassis:
                    dist = _dist(mujoco, m, d, g, h, 0.02, ft, joints, q)
                    if dist < 0.02 and dist < least.get((g, h), (np.inf,))[0]:
                        least[(g, h)] = (dist, [float(np.degrees(v)) for v in q])
            for k, (g, others) in named.items():
                for h in others:
                    dist = _dist(mujoco, m, d, g, h, 0.2, ft, joints, q)
                    if dist < best[k][0]:
                        best[k] = (dist, name(h))
        out[c] = {"pairs": {f"{name(g)} | {name(h)}": {"m": v[0], "at_roll_pitch_flexion_deg": v[1]}
                            for (g, h), v in sorted(least.items(), key=lambda kv: kv[1][0])},
                  "named": {k: {"m": v[0], "nearest": v[1]} for k, v in best.items()},
                  "grid": f"{steps} values over each of roll, pitch and flexion"}
    return out


def overlaps(model_path: Path, base_path: Path = BASE, steps: int = 9) -> dict:
    """Overlapping leg-chassis pairs of a variant, split into those the 188 mm base model has too (inherited) and
    those it does not (new)."""
    mine, base = packaging(model_path, steps), packaging(base_path, steps)
    res = {}
    for c in mine:
        bad = {k: v for k, v in mine[c]["pairs"].items() if v["m"] < 0}
        inherited = {k for k, v in base[c]["pairs"].items() if v["m"] < 0}
        res[c] = {"new": {k: v for k, v in bad.items() if k not in inherited},
                  "inherited": {k: v for k, v in bad.items() if k in inherited}}
    return res


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
        print("check: " + ("each variant equals PR #697's model with the recorded mass deviation, in every compiled "
                           "array and option" if not p else "PROBLEMS\n  " + "\n  ".join(p)))
        print("reproduces PR #697's files without the deviation (canonical XML): "
              + ", ".join(f"{n} mm {reproduces_pr(n)}" for n in LENGTHS))
        if p:
            return 1
    if a.packaging is not None:
        files = [Path(f) for f in a.packaging] or [HERE / "generated" / f"trike60_C_closed_capsule_fore{n}.xml"
                                                   for n in LENGTHS]
        for f in files:
            r = packaging(f)
            ov = overlaps(f)
            print(f.name)
            for c, x in r.items():
                print("  ", c, {k: (round(1000 * v["m"], 1), v["nearest"]) for k, v in x["named"].items()}, "(mm)")
                for kind in ("new", "inherited"):
                    print(f"     overlaps, {kind} against the 188 mm model:",
                          {k: (round(1000 * v["m"], 1), [round(a) for a in v["at_roll_pitch_flexion_deg"]])
                           for k, v in ov[c][kind].items()} or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
