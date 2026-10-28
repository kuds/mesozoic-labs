"""Build the hardware-sized juvenile candidate beside the reviewed mini."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from examples.mini_tyrannosaurus_rex.head_design import assign_inertia, box_component

from .appearance import apply_appearance
from .cavity import apply_cavities
from .hardware import apply_hardware
from .torso import apply_torso

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "model.xml"
REPO = HERE.parents[2]


def build(directory: Path = HERE) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(SOURCE).getroot()
    root.set("model", "mini_juvenile_hardware_candidate")
    root.find(".//text[@name='character_design']").set("data", "juvenile_hardware_candidate")
    for include in root.iter("include"):
        include.set("file", os.path.relpath(SOURCE.parent / include.attrib["file"], directory))
    appearance = apply_appearance(root, directory)
    hardware = apply_hardware(root)
    torso = apply_torso(root)
    bodies = {body.get("name"): body for body in root.iter("body")}

    # Reconstruct the 422 g source torso allocation explicitly. The old 43 g
    # skin allocation is replaced by the actual new surface-area estimate.
    rows = [
        box_component("hip_spine_frame_and_brackets", 0.082, (-0.010, 0, 0.022), (0.031, 0.030, 0.035)),
        box_component("unassigned_integration_reserve", 0.0279, (-0.025, 0, 0.025), (0.035, 0.035, 0.035)),
        torso["shell_mass_component"],
    ]
    for item in hardware["mass_components"]:
        if item["owner_body"] == "torso":
            rows.append(
                box_component(
                    item["name"], item["mass_kg"], tuple(item["pos_m"]), tuple(item["inertia_box_halfsize_m"])
                )
            )
    # Motor rotation is accounted for using its enclosing torso-frame box.
    for geom in bodies["torso"].findall("geom"):
        if geom.get("name", "").endswith("_servo_case"):
            quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, quat)
            half = abs(matrix.reshape(3, 3)) @ np.fromstring(geom.attrib["size"], sep=" ")
            rows.append(
                box_component(
                    geom.attrib["name"], 0.018, tuple(np.fromstring(geom.get("pos", "0 0 0"), sep=" ")), tuple(half)
                )
            )
    assign_inertia(bodies["torso"], rows)
    allocations = appearance["body_component_allocations"]
    allocations["torso"] = rows
    # Move the existing 3 g IMU allocation, then add the installed camera once.
    upper = allocations["head_pitch_body"]
    imu = next(item for item in hardware["mass_components"] if item["name"] == "head_imu")
    found = [i for i, row in enumerate(upper) if "imu" in row["name"].lower()]
    if len(found) != 1:
        raise ValueError("Expected exactly one retained head IMU allocation")
    upper[found[0]] = box_component(
        "head_imu", imu["mass_kg"], tuple(imu["pos_m"]), tuple(imu["inertia_box_halfsize_m"])
    )
    camera = next(item for item in hardware["mass_components"] if item["name"] == "camera")
    upper.append(
        box_component(
            "camera_installed_allowance",
            camera["mass_kg"],
            tuple(camera["pos_m"]),
            tuple(camera["inertia_box_halfsize_m"]),
        )
    )
    assign_inertia(bodies["head_pitch_body"], upper)
    used_meshes = {geom.get("mesh") for geom in root.iter("geom") if geom.get("mesh")}
    asset = root.find("asset")
    assert asset is not None
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used_meshes:
            asset.remove(mesh)
    ET.indent(root)
    path = directory / "model.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    mechanisms = [model.geom(i).name for i in range(model.ngeom) if model.geom_group[i] == 2]
    shells = torso["shell_geoms"] + [
        name for names in appearance["inspection_geoms_by_body"].values() for name in names
    ]
    pairs = []
    for shell in shells:
        for mechanism in mechanisms:
            if model.geom(shell).bodyid == model.geom(mechanism).bodyid:
                pairs.append(
                    dict(
                        first=shell,
                        second=mechanism,
                        clearance_m=0.002,
                        category="shell_vs_own_mechanism",
                    )
                )
    exclusions = []
    for interface in appearance.get("rigid_mount_interfaces", []):
        exclusions.append(dict(first=interface["first"], second=interface["second"], reason=interface["reason"]))
    for wall in torso["shell_geoms"]:
        if wall.startswith(("juv_torso_wall_00_", "juv_torso_wall_01_")):
            exclusions.append(
                dict(
                    first=wall,
                    second="juv_tail_shell_collision",
                    reason="Fixed tail root meets the rear torso mounting aperture; shared rigid assembly.",
                )
            )
    manifest = dict(
        schema_version=1,
        design="juvenile_hardware_candidate",
        source_pr=635,
        source_commit="398c02c13bdf82c907f1a75f34c30840ffba2d8e",
        clearance_m=0.002,
        hardware=hardware["hardware"],
        shell_geoms=shells,
        mechanism_geoms=mechanisms,
        clearance_pairs=pairs,
        excluded_pairs=exclusions,
        exterior_containment=apply_cavities(root, hardware, torso, appearance),
        mass_contract=dict(
            total_mass_kg=float(model.body_mass.sum()),
            body_masses_kg={model.body(i).name: float(model.body_mass[i]) for i in range(1, model.nbody)},
            expected_motor_count=14,
            motor_dimensions_m=[0.020, 0.034, 0.026],
        ),
        source_files=[
            str(file.relative_to(REPO))
            for file in (
                HERE / "build.py",
                HERE / "torso.py",
                HERE / "appearance.py",
                HERE / "hardware.py",
                HERE / "validate.py",
                HERE / "cavity.py",
                SOURCE,
            )
        ],
        omissions=hardware["hardware_limitations"] + appearance["limitations"] + torso["shell_limitations"],
        hardware_layout=hardware,
        torso_shell=torso,
        appearance=appearance,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (directory / "mass_ledger.json").write_text(
        json.dumps(
            dict(
                mass_kg=float(model.body_mass.sum()),
                body_components=allocations,
                basis="Explicit unweighed design allocations, full motor cases, provisional shell material. Not measured hardware mass.",
                retained_body_inertias=[
                    model.body(i).name for i in range(1, model.nbody) if model.body(i).name not in allocations
                ],
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(dict(model=str(path), mass_kg=float(model.body_mass.sum()), shell_patches=len(shells))))
    return path


if __name__ == "__main__":
    build()
