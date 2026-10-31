"""Apply the candidate's sourced hardware bodies and estimated installation space.

The manifest owns placement and dimensions. Catalog parts consume existing mass
budgets; drawing additional hardware geometry never adds its mass a second time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .build_model import Builder
from .proportions import attrs, component, find

MANIFEST_PATH = Path(__file__).with_name("long_body_hardware.json")


def load_manifest(layout: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = dict(json.loads(MANIFEST_PATH.read_text()))
    if layout is not None:
        for assembly in manifest["assemblies"]:
            if assembly["id"] == "battery":
                assembly["position_m"][0] = float(layout["battery_x_m"])
    return manifest


def apply_hardware(builder: Builder, *, battery_x_m: float) -> None:
    """Keep catalog bodies visible in service views and reserve boxes hidden."""
    manifest = load_manifest({"battery_x_m": battery_x_m})
    for assembly in manifest["assemblies"]:
        pos = np.array(assembly["position_m"], dtype=float)
        halfsize = np.array(assembly["installed_size_m"]) / 2
        component(
            builder,
            assembly["component"],
            pos_m=pos.tolist(),
            mass_kg=assembly["mass_budget_kg"],
            inertia_box_halfsize_m=halfsize.tolist(),
        )
        attrs(builder, "geom", assembly["geometry"], pos=pos, size=halfsize, group=5)
        for part in assembly["contents"]:
            builder.geom(
                builder.bodies[assembly["body"]],
                part["geometry"],
                "box",
                np.array(part["size_m"]) / 2,
                pos + part["offset_m"],
                "graphite" if part["catalog"] == "battery" else "pcb",
                group=4,
            )
    # This is an inertia allocation, not a solid block occupying the hip bay.
    attrs(builder, "geom", "pelvis_structure_envelope", group=5)
    for item in manifest["mounted_hardware"]:
        catalog = manifest["catalog"][item["catalog"]]
        if catalog["shape"] == "cylinder" or item["catalog"] == "camera":
            continue
        part = next(c for c in builder.components if c["name"] == item["component"])
        dimensions = np.array(catalog["dimensions_m"])[item["catalog_axes"]]
        component(builder, item["component"], inertia_box_halfsize_m=(dimensions / 2).tolist())
        builder.geom(
            builder.bodies[part["body"]],
            item["geometry"],
            "box",
            dimensions / 2,
            part["pos_m"],
            "graphite",
            group=4,
        )
    # The visible yaw and shoulder cases must also contain their catalog bodies.
    for visible, actual in (
        ("neck_yaw_case", "hardware_head_yaw_motor"),
        ("neck_nod_case", "hardware_head_nod_motor"),
        ("left_shoulder_case", "hardware_left_shoulder_motor"),
        ("right_shoulder_case", "hardware_right_shoulder_motor"),
    ):
        source = find(builder, "geom", actual)
        target = find(builder, "geom", visible)
        target.attrib.pop("euler", None)
        target.set("type", "box")
        target.set("size", source.attrib["size"])
        target.set("pos", source.attrib["pos"])
