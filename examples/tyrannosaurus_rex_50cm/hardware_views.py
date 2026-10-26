"""Candidate-only service views and source stamps for compiled hardware geometry."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from environments.shared import robot_rendering

MANIFEST_PATH = Path(__file__).with_name("long_body_hardware.json")
SERVICE_GROUPS = (1, 2, 4)
VIEWS = {"right elevated": (0.35, -1.0, 0.45), "left elevated": (-0.35, 1.0, 0.45)}
HIDDEN_PANELS = re.compile(
    r"torso_(front|bridge|rear)_shell|(?:left|right)_torso_(?:cover|vent_\d+)"
    r"|pelvis_structure_envelope|head_shell|neck_(?:yaw|nod)_case|(?:left|right)_shoulder_case"
    r"|(?:left|right)_(?:eye_socket|eye_white|iris|pupil|eye_glint|nostril)"
)


def _manifest() -> dict[str, Any]:
    return dict(json.loads(MANIFEST_PATH.read_text()))


def required_hardware() -> tuple[str, ...]:
    """Every declared assembly part and the camera must be exposed by the views."""
    manifest = _manifest()
    return tuple(part["geometry"] for assembly in manifest["assemblies"] for part in assembly["contents"]) + tuple(
        part["geometry"] for part in manifest["mounted_hardware"] if part["catalog"] == "camera"
    )


def service_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Mask removable obstructions on a copy, retaining actual group-4 parts.

    Group-5 installation reserves remain hidden; the pelvis mass surrogate is
    also hidden because it is not a fabricated structure. No physical geometry,
    mass, actuator limit or original rendering state is changed.
    """
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_front_shell") < 0:
        raise ValueError("Hardware service views require the separate long-body candidate.")
    for name in required_hardware():
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0 or model.geom_group[geom] != 4:
            raise ValueError(f"Required hardware {name} must exist in visible group 4.")
    result = copy.copy(model)
    for geom in range(result.ngeom):
        if HIDDEN_PANELS.fullmatch(result.geom(geom).name):
            result.geom_group[geom] = 5
    return result


def _home(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def hardware_visibility(model: mujoco.MjModel) -> dict[str, Any]:
    """Ray-check the exact service groups from both orthographic view directions.

    A hit proves some hardware surface is visible at a sampled location, not
    that its complete surface or a service-removal path is unobstructed.
    """
    data = _home(model)
    groups = np.array([int(group in SERVICE_GROUPS) for group in range(6)], dtype=np.uint8)
    hit = np.array([-1], dtype=np.int32)
    points = np.array(list(itertools.product((-0.8, 0.0, 0.8), repeat=3)))
    rows = {}
    for name in required_hardware():
        geom = model.geom(name).id
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"Hardware visibility sampling requires a box body: {name}.")
        world = (points * model.geom_size[geom]) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        counts = {}
        for view, direction in VIEWS.items():
            eye = robot_rendering.camera_basis(direction)[2]
            counts[view] = int(
                sum(
                    mujoco.mj_ray(model, data, point + 3 * eye, -eye, groups, 1, -1, hit) >= 0 and hit[0] == geom
                    for point in world
                )
            )
        rows[name] = {"visible_in": [view for view, count in counts.items() if count], "sampled_ray_hits": counts}
    return {
        "passed": all(row["visible_in"] for row in rows.values()),
        "scope": "Sampled hardware visibility from two service views; not complete exposure or removal clearance.",
        "rays_per_part_per_view": len(points),
        "parts": rows,
    }


def source_stamp(source_paths: Sequence[Path], source_root: Path) -> dict[str, Any]:
    """Metadata shared by PNG text and GIF JSON comments, with source freshness."""
    root = source_root.resolve()
    paths = {*source_paths, Path(__file__), MANIFEST_PATH, Path(robot_rendering.__file__)}
    return {
        "schema_version": 1,
        "source_sha256": {
            path.resolve().relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
    }


def png_metadata(stamp: dict[str, Any]) -> PngImagePlugin.PngInfo:
    """Encode image evidence as PNG text without introducing a sidecar file."""
    metadata = PngImagePlugin.PngInfo()
    for name, value in stamp.items():
        metadata.add_text(name, json.dumps(value, sort_keys=True, allow_nan=False))
    return metadata


def write_service_view(
    model: mujoco.MjModel,
    *,
    output: Path,
    source_paths: Sequence[Path],
    source_root: Path,
) -> dict[str, Any]:
    """Save an annotated candidate hardware PNG and return visibility evidence."""
    service = service_model(model)
    visibility = hardware_visibility(service)
    if not visibility["passed"]:
        hidden = [name for name, row in visibility["parts"].items() if not row["visible_in"]]
        raise ValueError(f"Service views do not expose required hardware: {', '.join(hidden)}.")
    data = _home(service)
    manifest = _manifest()
    panels = Image.new("RGB", (1400, 690), (245, 247, 244))
    draw = ImageDraw.Draw(panels)
    draw.text((24, 15), "LONG-BODY CANDIDATE / HARDWARE SERVICE VIEW", font=robot_rendering.font(25), fill="#26372b")
    draw.text(
        (24, 51),
        "Torso/head shells removed; installation reserves and the pelvis mass surrogate hidden",
        font=robot_rendering.font(18),
        fill="#536158",
    )
    assemblies = [
        (assembly["id"], [service.geom(part["geometry"]).id for part in assembly["contents"]])
        for assembly in manifest["assemblies"]
    ]
    assemblies.append(("camera", [service.geom(name).id for name in required_hardware() if "camera" in name]))
    names = required_hardware()
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    world = np.concatenate(
        [
            (corners * service.geom_size[service.geom(name).id]) @ data.geom_xmat[service.geom(name).id].reshape(3, 3).T
            + data.geom_xpos[service.geom(name).id]
            for name in names
        ]
    )
    for column, (view, direction) in enumerate(VIEWS.items()):
        basis = robot_rendering.camera_basis(direction)
        projected = world @ basis.T
        low, high = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
        padding = 0.12 * (high - low)
        low, high = low - padding, high + padding
        panel = robot_rendering.render(
            service,
            data,
            direction=direction,
            groups=SERVICE_GROUPS,
            width=700,
            height=440,
            projection_bounds=(low, high),
        )
        local = ImageDraw.Draw(panel)
        scale = min(700 * 0.84 / (high[0] - low[0]), 440 * 0.82 / (high[1] - low[1]))
        for index, (_, geoms) in enumerate(assemblies, 1):
            center = data.geom_xpos[geoms].mean(axis=0) @ basis.T
            x, y = (center[:2] - (low + high) / 2) * [scale, -scale] + [350, 220]
            local.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#f7f6e9", outline="#344b3c", width=2)
            local.text((x, y), str(index), anchor="mm", font=robot_rendering.font(15), fill="#26372b")
        panels.paste(panel, (700 * column, 100))
        draw.text((24 + 700 * column, 80), view.upper(), font=robot_rendering.font(17), fill="#536158")
    draw.text(
        (24, 556),
        "1 Battery   2 Jetson compute   3 Control + U2D2/CAN   4 Five regulators   5 OAK-D Lite camera",
        font=robot_rendering.font(20),
        fill="#26372b",
    )
    draw.text(
        (24, 599),
        "Catalog-sized bodies and explicit planning allowances; mounts, cables and cooling remain unqualified.",
        font=robot_rendering.font(18),
        fill="#536158",
    )
    draw.text(
        (24, 630),
        "This cutaway exposes declared parts; it does not establish assembly or service access.",
        font=robot_rendering.font(18),
        fill="#536158",
    )
    stamp = source_stamp(source_paths, source_root)
    stamp["hardware_visibility"] = visibility
    output.parent.mkdir(parents=True, exist_ok=True)
    panels.save(output, pnginfo=png_metadata(stamp))
    return visibility
