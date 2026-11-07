"""Build an unregistered bracket/cover revision and repeat the r01 interference screen."""

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from environments.shared.robot_building import component_inertia, values

from .build import named
from .validate_platform import HERE, ROOT, clearance, dynamics, hash_file, single_support

OUT = HERE / "clearance_study"
REPORT = ROOT / "hardware/tyrannosaurus_rex_60cm/m02_integrated/clearance_study_r02.json"


def build():
    xml = ET.parse(HERE / "nominal/model.xml").getroot()
    ledger = json.loads((HERE / "nominal/components.json").read_text())
    parts = {part["name"]: part for part in ledger["components"]}
    xml.set("model", "rex_intermediate_clearance_study_r02")

    def geom(name):
        return named(xml, "geom", name)

    def add_bracket(name, body, first, last, radius, grams):
        # Gross bracket stock allowance includes fasteners and insert bosses.
        first, last = np.array(first), np.array(last)
        ET.SubElement(
            named(xml, "body", body),
            "geom",
            dict(
                name=name,
                type="capsule",
                fromto=values([*first, *last]),
                size=values(radius),
                group="2",
                material="metal",
            ),
        )
        ledger["components"].append(
            dict(
                name=name,
                body=body,
                category="clearance_revision_brackets",
                mass_kg=grams / 1000,
                pos_m=((first + last) / 2).tolist(),
                inertia_box_halfsize_m=(np.abs(first - last) / 2 + radius).tolist(),
            )
        )

    # Keep enclosure and maintenance reservation dimensions. Hang the disconnect
    # outboard on the right side, away from the rear battery tray and left exit.
    location = [-0.142, -0.089, 0.073]
    for name in ("drive_disconnect_reserve", "drive_disconnect_case"):
        geom(name).set("pos", values(location))
    parts["drive_disconnect"]["pos_m"] = location
    add_bracket("disconnect_bracket", "pelvis", (-0.142, -0.075, 0.130), (-0.142, -0.089, 0.092), 0.004, 18)

    # Move the front camera braces to the corners of the spine, bypassing
    # the Jetson on both sides. A bridge and short drop tabs carry the camera.
    for side, sign in (("left", 1), ("right", -1)):
        geom("camera_brace_" + side).set("fromto", values((0.169, sign * 0.075, 0.130, 0.211, sign * 0.071, 0.031)))
        add_bracket(
            "camera_drop_tab_" + side, "pelvis", (0.211, sign * 0.046, 0.031), (0.211, sign * 0.046, 0.014), 0.003, 4
        )
        # Raise the original thin torso skin locally away from both hip yaw cases.
        segment = geom(f"side_panel_{sign}")
        ends = np.fromstring(segment.get("fromto"), sep=" ").reshape((2, 3))
        ends[:, 1] += 0.014 * sign
        segment.set("fromto", values(ends.ravel()))
        parts[f"torso_side_panel_{sign}"]["pos_m"][1] += 0.014 * sign
        geom(f"rib_mount_{sign}").set("fromto", values((-0.08, sign * 0.075, 0.13, -0.08, sign * 0.101, 0.16)))
    add_bracket("camera_mount_bridge", "pelvis", (0.211, -0.071, 0.031), (0.211, 0.071, 0.031), 0.003, 18)

    # Pull only the near-pivot edges of the skin away from the stationary
    # bearing forks. The pivot, motor, bearings, distal tips and jaw stay put.
    # Account for the shifted skin mass by the same affine map as the cover.
    for mesh_name, near, far, part in (
        ("cranium", 0.034, 0.211, "skull_and_jaw_cover"),
        ("tail_shroud", -0.067, -0.292, "tail_spar_shell"),
    ):
        mesh = named(xml, "mesh", mesh_name)
        points = np.fromstring(mesh.get("vertex"), sep=" ").reshape((-1, 3))
        old_near = float(points[:, 0].min() if far > 0 else points[:, 0].max())
        factor = (far - near) / (far - old_near)
        points[:, 0] = near + (points[:, 0] - old_near) * factor
        mesh.set("vertex", values(points.ravel()))
        # Both stock allocations include a jaw or spar that did not move.
        # Move half their lumped center delta for that unresolved mass split.
        lump = parts[part]
        old_center = lump["pos_m"][0]
        lump["pos_m"][0] += 0.5 * (near + (old_center - old_near) * factor - old_center)
    # Move the crown marker to the revised first highest section.
    cranium = np.fromstring(named(xml, "mesh", "cranium").get("vertex"), sep=" ").reshape(-1, 3)
    crown = cranium[np.argmax(cranium[:, 2])]
    named(xml, "site", "head_crown").set("pos", values((crown[0], 0, crown[2])))
    add_bracket("neck_cover_spine", "head", (0, 0, 0), (0.070, 0, 0.012), 0.005, 20)
    # Incremental single-piece outsourced fabrication allowances in USD.
    # The r01 project budget already contains a frame/cover fabrication lot;
    # add these only if a supplier excludes these brackets from that lot.
    ledger["additional_fabrication_allowance_usd"] = {
        "basis": "engineering estimates for cut/bent sheet or machined stock and inserts, no quote",
        "disconnect_bracket": [25, 70],
        "camera_drop_tab_left": [10, 25],
        "camera_drop_tab_right": [10, 25],
        "camera_mount_bridge": [25, 65],
        "neck_cover_spine": [25, 80],
    }
    ledger["hardware_revision"] = "intermediate-clearance-study-r02"
    ledger["proportion_variant"] = "nominal_repack"
    categories = defaultdict(float)
    for component in ledger["components"]:
        categories[component["category"]] += component["mass_kg"]
    ledger["categories_kg"] = dict(categories)
    ledger["mass_kg"] = sum(categories.values())
    for body in ("pelvis", "head", "tail"):
        node = named(xml, "body", body).find("inertial")
        mass, center, inertia = component_inertia([c for c in ledger["components"] if c["body"] == body])
        node.set("pos", values(center))
        node.set("mass", values(mass))
        node.set("fullinertia", values(inertia[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]))
    ET.indent(xml, space="  ")
    OUT.mkdir(exist_ok=True)
    (OUT / "model.xml").write_text(ET.tostring(xml, encoding="unicode") + "\n")
    (OUT / "components.json").write_text(json.dumps(ledger, indent=2) + "\n")


def worst_by_pair(rows):
    by_pair = defaultdict(list)
    for witness in rows["failing_or_unresolved_witnesses"]:
        by_pair[tuple(sorted(witness["pair"]))].append(witness)
    return {pair: min(values, key=lambda x: x["distance_m"]) for pair, values in by_pair.items()}


def compare():
    build()
    # Reuse the existing r01 sample screen. Only use families generated in both.
    r01 = json.loads((REPORT.parent / "mechanical_report.json").read_text())
    old = r01["variants"]["nominal"]["clearance"]
    model = mujoco.MjModel.from_xml_path(str(OUT / "model.xml"))
    new = clearance(model)
    families = {r["family"] for r in new["families"]}

    def comparison_set(result):
        return worst_by_pair(
            {
                "failing_or_unresolved_witnesses": [
                    w for w in result["failing_or_unresolved_witnesses"] if w["family"] in families
                ]
            }
        )

    previous, current = comparison_set(old), comparison_set(new)

    def labels(source, keys):
        return [source[k] for k in sorted(keys)]

    motion = []
    for floor in ("hard", "carpet"):
        test_model = mujoco.MjModel.from_xml_path(str(OUT / "model.xml"))
        result, _ = dynamics(test_model, "hold_left", floor, 0.0, 22.2)
        motion.append(result)
    files = [
        Path(__file__),
        HERE / "validate_platform.py",
        HERE / "build.py",
        HERE / "nominal/model.xml",
        HERE / "nominal/components.json",
        OUT / "model.xml",
        OUT / "components.json",
        REPORT.parent / "mechanical_report.json",
        ROOT / "environments/shared/robot_building.py",
        ROOT / "environments/shared/robot_clearance.py",
        ROOT / "environments/shared/intermediate_drive.py",
        ROOT / "environments/shared/robot_floor.py",
    ]
    report = dict(
        schema="intermediate-clearance-remediation/v1",
        release_status="untrained engineering study",
        mass_kg=float(model.body_mass.sum()),
        versions=r01["versions"],
        source_sha256={p.relative_to(ROOT).as_posix(): hash_file(p) for p in files},
        comparison_families=sorted(families),
        initial_pair_count=old["pair_count"],
        current_pair_count=new["pair_count"],
        initial_unresolved_or_failed_pairs=len(previous),
        current_unresolved_or_failed_pairs=len(current),
        cleared_pairs=labels(previous, previous.keys() - current.keys()),
        newly_unresolved_or_failed_pairs=labels(current, current.keys() - previous.keys()),
        static_single_support=single_support(model),
        hold_left=motion,
        clearance=new,
    )
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    compare()
