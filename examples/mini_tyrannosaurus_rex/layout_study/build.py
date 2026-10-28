"""Build three open-chassis layout studies with identical parts and payloads.

The rounded surface is a silhouette envelope, NOT a manufactured shell. It has
no contact or mass of its own; the ledger includes a common shell allocation.
Full motor cases and installation boxes are the packing-screen geometry.
"""

from __future__ import annotations

import itertools
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from environments.shared.robot_building import component_inertia, element, values

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "model.xml"


@dataclass(frozen=True)
class Part:
    name: str
    raw: tuple[float, float, float]
    installed: tuple[float, float, float]
    grams: float
    source: str
    basis: str


PARTS = (
    Part(
        "computer",
        (65, 30, 8),
        (85, 50, 22),
        35,
        "https://docs.radxa.com/en/zero/zero3",
        "65x30 PCB; 8 mm thickness and installed mass are allowances",
    ),
    Part(
        "battery",
        (70, 35, 22),
        (80, 43, 30),
        70,
        "../juvenile/hardware.py",
        "Unselected pack: maximum procurement envelope, not a verified battery SKU",
    ),
    Part(
        "u2d2",
        (48, 14.9, 18),
        (60, 20, 25),
        9,
        "https://docs.robotis.com/docs/parts/interface/u2d2/",
        "Body dimensions; connector reserve provisional",
    ),
    Part(
        "motor_regulator",
        (40.64, 20.32, 7.62),
        (50, 30, 15),
        4.8,
        "https://www.pololu.com/product/2866/specs",
        "D24V90F5 without terminals; heat rejection unqualified",
    ),
    Part(
        "logic_regulator",
        (17.78, 17.78, 7.874),
        (25, 25, 12),
        2.3,
        "https://www.pololu.com/product/2858/specs",
        "D24V22F5 without headers",
    ),
    Part(
        "power_distribution",
        (30, 20, 10),
        (35, 25, 15),
        28,
        "../juvenile/hardware.py",
        "Unselected protection and distribution assembly; procurement allowance",
    ),
    Part(
        "body_imu",
        (25.4, 17.78, 4.5),
        (30, 25, 8),
        3,
        "https://github.com/adafruit/Adafruit-LSM6DSOX-PCB",
        "ISM330DHCX board; thickness and plugs assumed",
    ),
    Part(
        "character_electronics",
        (20, 15, 10),
        (25, 20, 15),
        10,
        "../juvenile/hardware.py",
        "Unselected audio/LED/button allowance",
    ),
    Part(
        "harness_trunk",
        (32, 20, 6),
        (39, 25, 10),
        35,
        "../juvenile/hardware.py",
        "Whole-robot wire mass at a reserved gathering bay; not complete moving cable routes",
    ),
)

# Positions in mm from the pelvis origin. Permutations rotate the ORIGINAL
# part and its installed reserve together; no dimension is scaled or dropped.
LAYOUTS: dict[str, dict[str, Any]] = {
    "a_low_spine": {
        "label": "A · Low spine",
        "hip_yaw_half_spacing_mm": 18,
        "intent": "Lengthwise battery and computer; low oval torso; best initial dinosaur silhouette",
        "placements": {
            "battery": ((-10, 0, 58), (0, 1, 2)),
            "computer": ((-10, 0, 87), (0, 1, 2)),
            "u2d2": ((-10, 39, 65), (0, 1, 2)),
            "motor_regulator": ((-10, -40, 60), (0, 1, 2)),
            "logic_regulator": ((13, -40, 85), (0, 1, 2)),
            "power_distribution": ((-65, 0, 64), (2, 1, 0)),
            "body_imu": ((-19, -40, 84), (0, 1, 2)),
            "character_electronics": ((45, 0, 87), (2, 0, 1)),
            "harness_trunk": ((50, 0, 62), (2, 0, 1)),
        },
    },
    "b_upright_core": {
        "label": "B · Upright core",
        "hip_yaw_half_spacing_mm": 18,
        "intent": "Vertical battery and computer; shortest body; taller and less theropod-like",
        "placements": {
            "battery": ((-12, 0, 84), (1, 2, 0)),
            "computer": ((27, 0, 84), (2, 0, 1)),
            "u2d2": ((-12, 31, 82), (1, 2, 0)),
            "motor_regulator": ((-12, -28, 83), (1, 2, 0)),
            "logic_regulator": ((-48, 0, 109), (0, 1, 2)),
            "power_distribution": ((-47, 0, 82), (2, 1, 0)),
            "body_imu": ((0, -31, 117), (0, 1, 2)),
            "character_electronics": ((0, 28, 124), (0, 1, 2)),
            "harness_trunk": ((-48, 0, 52), (1, 0, 2)),
        },
    },
    "c_saddle": {
        "label": "C · Low saddle",
        "hip_yaw_half_spacing_mm": 39,
        "intent": "Battery between outboard hips; lower centre of mass; wider stance and hip structure",
        "placements": {
            "battery": ((-10, 0, 10), (0, 1, 2)),
            "computer": ((-10, 0, 42), (0, 1, 2)),
            "u2d2": ((-10, 39, 55), (0, 1, 2)),
            "motor_regulator": ((-10, -43, 50), (0, 1, 2)),
            "logic_regulator": ((13, -38, 68), (0, 1, 2)),
            "power_distribution": ((-65, 0, 40), (2, 1, 0)),
            "body_imu": ((-19, -38, 67), (0, 1, 2)),
            "character_electronics": ((45, 0, 46), (2, 0, 1)),
            "harness_trunk": ((50, 0, 20), (2, 0, 1)),
        },
    },
}

PAYLOADS = (
    ("future_head_and_neck", 90, (65, 0, 105), (45, 32, 30)),
    ("future_fixed_tail", 15, (-105, 0, 55), (65, 10, 10)),
    ("future_fixed_arms", 10, (35, 0, 55), (10, 45, 10)),
)
HEAD_BUDGET = {
    "two_full_XL330_motors_g": 36,
    "camera_and_mount_g": 8,
    "head_imu_g": 3,
    "neck_frame_shell_fasteners_and_cable_g": 43,
}


def corners(pos: Any, dims: Any, padding: float = 0) -> np.ndarray:
    return np.asarray(
        np.array(list(itertools.product((-1, 1), repeat=3))) * (np.array(dims) / 2 + padding) + np.asarray(pos)
    )


def fit_silhouette(parts: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Small enclosing ellipsoid of installed corners plus 3 mm axis padding.

    A convenient silhouette envelope only: this is not a wall-thickness or
    connector/assembly certificate. The eventual shell needs joint apertures.
    """
    pts = np.concatenate([corners(p["pos_mm"], p["installed_mm"], 3) for p in parts])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    initial = np.r_[(lo + hi) / 2, (hi - lo) * 0.9]

    def containment(x: np.ndarray) -> np.ndarray:
        return np.asarray(1 - np.sum(((pts - x[:3]) / x[3:]) ** 2, axis=1))

    result = minimize(
        lambda x: float(np.log(x[3:]).sum()),
        initial,
        method="SLSQP",
        bounds=[(lo[i], hi[i]) for i in range(3)] + [(10, 200)] * 3,
        constraints={"type": "ineq", "fun": containment},
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not result.success or containment(result.x).min() < -1e-7:
        raise RuntimeError("Silhouette fit failed")
    return result.x[:3], result.x[3:]


def _component(name: str, grams: float, pos_mm: Any, dims_mm: Any) -> dict[str, Any]:
    return dict(
        name=name,
        mass_kg=grams / 1000,
        pos_m=(np.array(pos_mm) / 1000).tolist(),
        inertia_box_halfsize_m=(np.array(dims_mm) / 2000).tolist(),
    )


def build(name: str, directory: Path | None = None) -> dict[str, Any]:
    pos: Any
    dims: Any
    layout = LAYOUTS[name]
    directory = directory or HERE / name
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.parse(BASE).getroot()
    root.set("model", name)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "include":
                parent.remove(child)
    # Keep only the complete reference leg assemblies and their original axes,
    # link lengths, cases, bearings, bracket proxies and foot contacts.
    torso = root.find("worldbody/body[@name='torso']")
    assert torso is not None
    torso.set("pos", "0 0 .113")
    retained = {"left_hip_yaw_body", "right_hip_yaw_body"}
    for child in list(torso):
        if child.tag == "freejoint" or (child.tag == "body" and child.get("name") in retained):
            continue
        if child.tag == "geom" and "hip_yaw_servo_case" in child.get("name", ""):
            continue
        torso.remove(child)
    for body in torso.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("type") == "mesh" or "fairing_fastener" in geom.get("name", ""):
                body.remove(geom)
        if body.get("name", "").endswith("_foot"):
            # Common 3 mm longer ankle pedestal: the original extreme ankle
            # poses left <1 mm between motor and sole. Keep full foot length
            # and ankle travel, and explicitly reallocate 25 g per foot.
            for geom in body.findall("geom"):
                n = geom.get("name", "")
                if "flat_pad" in n or "baseplate" in n:
                    pos = np.fromstring(geom.get("pos", ""), sep=" ")
                    pos[2] -= 0.003
                    geom.set("pos", values(pos))
                if "pedestal" in n:
                    pos = np.fromstring(geom.get("pos", ""), sep=" ")
                    size = np.fromstring(geom.get("size", ""), sep=" ")
                    pos[2] -= 0.0015
                    size[2] += 0.0015
                    geom.set("pos", values(pos))
                    geom.set("size", values(size))
            old = body.find("inertial")
            assert old is not None
            body.remove(old)
            foot_parts = [
                _component("sole_and_plate", 16, (7, 0, -25), (70, 34, 5)),
                _component("pedestal_bearings_brackets", 9, (0, -6, -13), (20, 20, 22)),
            ]
            fm, fc, fi = component_inertia(foot_parts)
            element(
                body,
                "inertial",
                mass=fm,
                pos=fc,
                fullinertia=(fi[0, 0], fi[1, 1], fi[2, 2], fi[0, 1], fi[0, 2], fi[1, 2]),
            )
            pad = body.find("geom[@class='collision'][@type='box']")
            assert pad is not None
            element(
                body,
                "geom",
                name=str(body.get("name")) + "_sole_visual",
                type="box",
                pos=pad.get("pos"),
                size=pad.get("size"),
                group=2,
                rgba=(0.24, 0.27, 0.26, 1),
                contype=0,
                conaffinity=0,
            )
    asset = root.find("asset")
    assert asset is not None
    for child in list(asset):
        if child.tag == "mesh":
            asset.remove(child)
    for tag in ("contact", "equality", "sensor", "keyframe", "custom"):
        old = root.find(tag)
        if old is not None:
            root.remove(old)
    actuator = root.find("actuator")
    assert actuator is not None
    for child in list(actuator):
        if child.get("joint") not in {j.get("name") for j in torso.iter("joint")}:
            actuator.remove(child)
        else:
            child.set("forcerange", "-0.10 0.10")

    delta = (layout["hip_yaw_half_spacing_mm"] - 18) / 1000
    for side, sign in (("left", 1), ("right", -1)):
        hip_body = torso.find(f"body[@name='{side}_hip_yaw_body']")
        assert hip_body is not None
        pos = np.fromstring(hip_body.get("pos", ""), sep=" ")
        pos[1] += sign * delta
        hip_body.set("pos", values(pos))
        for g in torso.findall("geom"):
            if g.get("name", "").startswith(f"{side}_hip_yaw_servo_case"):
                pos = np.fromstring(g.get("pos", ""), sep=" ")
                pos[1] += sign * delta
                g.set("pos", values(pos))

    parts, masses = [], []
    palette = {
        "battery": (0.88, 0.46, 0.16, 1),
        "computer": (0.12, 0.46, 0.39, 1),
        "harness_trunk": (0.50, 0.33, 0.56, 1),
    }
    for p in PARTS:
        pos, order = layout["placements"][p.name]
        raw = [p.raw[i] for i in order]
        installed = [p.installed[i] for i in order]
        row = dict(
            name=p.name,
            pos_mm=list(pos),
            raw_mm=raw,
            installed_mm=installed,
            mass_g=p.grams,
            source=p.source,
            basis=p.basis,
        )
        parts.append(row)
        color = palette.get(p.name, (0.24, 0.40, 0.58, 1))
        for suffix, dims, group, alpha in (("raw", raw, 4, 1), ("reserve", installed, 5, 0.15)):
            element(
                torso,
                "geom",
                name=f"part_{p.name}_{suffix}",
                type="box",
                pos=np.array(pos) / 1000,
                size=np.array(dims) / 2000,
                group=group,
                rgba=(*color[:3], alpha),
                contype=0,
                conaffinity=0,
            )
        # Raw part contacts for an open chassis; no invisible torso support shell.
        element(
            torso,
            "geom",
            name=f"part_{p.name}_collision",
            class_="collision",
            type="box",
            pos=np.array(pos) / 1000,
            size=np.array(raw) / 2000,
        )
        masses.append(_component(p.name, p.grams, pos, installed))

    for side, sign in (("left", 1), ("right", -1)):
        masses.append(
            _component(
                f"{side}_hip_yaw_motor", 18, (-7.5, sign * layout["hip_yaw_half_spacing_mm"], 26.5), (34, 20, 26)
            )
        )
    for label, grams, pos, dims in (
        ("torso_frame_fasteners", 80, (0, 0, 35), (90, 85, 50)),
        ("integration_reserve", 40, (0, 0, 35), (90, 85, 50)),
        ("future_torso_skin", 30, (0, 0, 60), (100, 90, 65)),
    ):
        masses.append(_component(label, grams, pos, dims))
    for label, grams, pos, half in PAYLOADS:
        body = element(torso, "body", name=label, pos=np.array(pos) / 1000)
        inertia = grams / 1000 / 3 * (np.sum((np.array(half) / 1000) ** 2) - (np.array(half) / 1000) ** 2)
        element(body, "inertial", mass=grams / 1000, pos=(0, 0, 0), diaginertia=inertia)
        element(
            body,
            "geom",
            name=f"{label}_mass_marker",
            type="ellipsoid",
            size=np.array(half) / 1000,
            rgba=(0.5, 0.55, 0.58, 0.2),
            group=5,
            contype=0,
            conaffinity=0,
        )
    mass, com, inertia = component_inertia(masses)
    element(
        torso,
        "inertial",
        mass=mass,
        pos=com,
        fullinertia=(inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]),
    )
    imu = next(p for p in parts if p["name"] == "body_imu")
    element(torso, "site", name="body_imu", pos=np.array(imu["pos_mm"]) / 1000, size=0.002)
    sensor = element(root, "sensor")
    element(sensor, "gyro", name="body_gyro", site="body_imu")
    element(sensor, "accelerometer", name="body_accelerometer", site="body_imu")
    for a in actuator:
        j = a.get("joint")
        element(sensor, "jointpos", name=f"{j}_position", joint=j)
        element(sensor, "jointvel", name=f"{j}_velocity", joint=j)
    center, radii = fit_silhouette(parts)
    element(
        torso,
        "geom",
        name="proposed_torso_silhouette",
        type="ellipsoid",
        pos=center / 1000,
        size=radii / 1000,
        rgba=(0.36, 0.53, 0.43, 1),
        group=1,
        contype=0,
        conaffinity=0,
    )
    # No head/tail/arm joints are created. Payloads are rigid budgeting fixtures,
    # not hardware designs or evidence for expression/recovery behaviours.
    key = element(root, "keyframe")
    element(key, "key", name="home", qpos=(0, 0, 0.113, 1, 0, 0, 0, *([0] * 10)), ctrl=[0] * 10)
    ET.indent(root, space="  ")
    (directory / "model.xml").write_text(ET.tostring(root, encoding="unicode") + "\n")
    record = dict(
        name=name,
        label=layout["label"],
        intent=layout["intent"],
        source_reference="28a6bcd52c07d8381e18e5947bdb40afbb7151a7",
        parts=parts,
        torso_mass_components=masses,
        future_payloads=PAYLOADS,
        future_head_budget_g=HEAD_BUDGET,
        hip_yaw_half_spacing_mm=layout["hip_yaw_half_spacing_mm"],
        foot_allocation_g=25,
        common_ankle_pedestal_extension_mm=3,
        silhouette_center_mm=center.tolist(),
        silhouette_radii_mm=radii.tolist(),
        silhouette_dimensions_mm=(2 * radii).tolist(),
        scope="Open-chassis packing study; rounded shape is a visual envelope, not shell CAD",
        omitted="Detailed mounting plates, horn/idler loads, moving cables, cooling, extraction paths and actual future head/tail geometry",
    )
    (directory / "layout.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    for key in LAYOUTS:
        record = build(key)
        print(key, "torso envelope mm", np.round(record["silhouette_dimensions_mm"], 1))
