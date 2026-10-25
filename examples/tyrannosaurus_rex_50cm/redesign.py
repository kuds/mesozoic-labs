"""Parameterized longer/lower medium concept with the existing leg mechanism.

This candidate is separate from silhouette C. Motor capacities and joint ranges
are preserved. Its component and collision geometry remains an unweighed
packaging allocation, not fabrication CAD or demonstrated self-righting.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from environments.shared.robot_building import element, loft

from . import build_model as base
from .proportions import attrs, component, find

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Layout:
    neck_x_m: float = 0.150
    neck_z_m: float = 0.188
    nod_reach_m: float = 0.060
    skull_length_m: float = 0.260
    tail_length_m: float = 0.440
    tail_root_x_m: float = -0.220
    battery_x_m: float = -0.145
    torso_shell_mass_g: float = 230
    pelvis_structure_mass_g: float = 380
    neck_support_mass_g: float = 80
    tail_mass_g: float = 120

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


TORSO_PROXIES = ("torso_front_collision", "torso_bridge_collision", "torso_rear_collision")
UPPER_PROXIES = (
    *TORSO_PROXIES,
    "neck_support_collision",
    "head_collision",
    "jaw_collision",
    "left_arm_collision",
    "right_arm_collision",
    "tail_segment_0_collision",
    "tail_segment_1_collision",
    "tail_segment_2_collision",
)
ATTACHMENT_EXCLUSIONS = {
    ("torso_front_collision", "torso_bridge_collision"): "Overlapping proxy volumes form one rigid torso envelope.",
    ("torso_bridge_collision", "torso_rear_collision"): "Overlapping proxy volumes form one rigid torso envelope.",
    ("torso_front_collision", "neck_support_collision"): "Neck support is rooted in the rigid front torso structure.",
    ("torso_bridge_collision", "neck_support_collision"): "Neck support crosses the rigid upper torso structure.",
    ("torso_rear_collision", "tail_segment_0_collision"): "Tail root meets its mounting aperture in the rear torso.",
    ("tail_segment_0_collision", "tail_segment_1_collision"): "Adjacent segments describe one continuous rigid tail.",
    ("tail_segment_0_collision", "tail_segment_2_collision"): "Both segments belong to the same rigid tail.",
    ("tail_segment_1_collision", "tail_segment_2_collision"): "Adjacent segments describe one continuous rigid tail.",
}


class Builder(base.Builder):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout
        self.xml.set("model", "tyrannosaurus_rex_medium_long_body_candidate")
        statistic = self.xml.find("statistic")
        assert statistic is not None
        statistic.set("center", "-0.10 0 0.27")
        statistic.set("extent", "1.15")

    def upper_body(self) -> None:
        super().upper_body()
        t, spec = self.torso, self.layout
        for geom in list(t.findall("geom")):
            name = geom.get("name", "")
            if name in ("torso_shell", "belly_panel", "back_access_panel", "torso_collision") or name.startswith(
                ("back_vent_", "back_screw_")
            ):
                t.remove(geom)
        # Three occupied lobes leave the hip-motor bay open below the bridge.
        # The visible surfaces follow these envelopes rather than covering an
        # opening with a cosmetic solid shell that the clearance screen ignores.
        for name, size, pos in (
            ("torso_front", (0.070, 0.100, 0.060), (0.110, 0, 0.130)),
            ("torso_bridge", (0.135, 0.095, 0.035), (-0.040, 0, 0.155)),
            ("torso_rear", (0.070, 0.090, 0.065), (-0.160, 0, 0.125)),
        ):
            self.geom(t, name + "_shell", "ellipsoid", size, pos)
            self.collision(t, name, "ellipsoid", size, pos)
        for side, sign in (("left", 1), ("right", -1)):
            self.geom(
                t,
                side + "_torso_cover",
                "ellipsoid",
                (0.087, 0.003, 0.029),
                (-0.030, sign * 0.093, 0.153),
                "dark_sage",
            )
            for i, x in enumerate((-0.085, -0.055, -0.025)):
                self.geom(
                    t, f"{side}_torso_vent_{i}", "box", (0.006, 0.001, 0.010), (x, sign * 0.097, 0.154), "graphite"
                )
            shoulder = (0.093, sign * 0.087, 0.108)
            attrs(self, "geom", side + "_shoulder_case", pos=shoulder)
            attrs(self, "body", side + "_arm", pos=(shoulder[0], sign * 0.119, shoulder[2]))
            # Enclose the bent upper arm, forearm and fingers rather than using
            # the inherited straight capsule through their end points.
            arm_proxy = find(self, "geom", side + "_arm_collision")
            arm_proxy.attrib.pop("fromto")
            attrs(
                self,
                "geom",
                side + "_arm_collision",
                type="box",
                size=(0.042, 0.015, 0.022),
                pos=(0.031, sign * 0.002, -0.013),
            )
            component(self, side + "_shoulder_motor", pos_m=list(shoulder))
            component(self, side + "_shoulder_mount", pos_m=list(shoulder))
            attrs(self, "mesh", side + "_thigh_fairing_mesh", scale=(0.037, 0.027, 0.048))
            component(self, side + "_leg_cover", mass_kg=0.035, inertia_box_halfsize_m=[0.037, 0.027, 0.048])
        locations = {
            "battery": (spec.battery_x_m, 0, 0.110),
            "compute_tray": (-0.050, 0, 0.155),
            "control_board": (0.075, 0, 0.140),
            "power_board": (-0.040, 0, 0.115),
            "pelvis_structure": (-0.010, 0, 0.050),
            "integration_reserve": (-0.018, 0, 0.065),
        }
        for name, pos in locations.items():
            component(self, name, pos_m=list(pos))
            if name != "integration_reserve":
                attrs(self, "geom", name + "_envelope", pos=pos)
        component(self, "pelvis_structure", mass_kg=spec.pelvis_structure_mass_g / 1000)
        component(
            self,
            "torso_shell_mass",
            mass_kg=spec.torso_shell_mass_g / 1000,
            pos_m=[-0.035, 0, 0.140],
            inertia_box_halfsize_m=[0.200, 0.100, 0.060],
        )
        self.component(
            t,
            "neck_support_allowance",
            "load_bearing_structure",
            spec.neck_support_mass_g,
            (spec.neck_x_m - 0.030, 0, spec.neck_z_m - 0.030),
            (0.045, 0.025, 0.030),
        )
        for collision in (False, True):
            kw = dict(
                name="neck_support" + ("_collision" if collision else ""),
                type="capsule",
                fromto=(spec.neck_x_m - 0.065, 0, 0.135, spec.neck_x_m, 0, spec.neck_z_m - 0.020),
                size=0.012,
            )
            element(t, "geom", **kw, **({"class_": "collision"} if collision else {"material": "graphite", "group": 2}))
        tail = self.bodies["tail"]
        attrs(self, "body", "tail", pos=(spec.tail_root_x_m, 0, 0.125))
        attrs(self, "geom", "tail_bearing", pos=(spec.tail_root_x_m + 0.009, 0, 0.125), size=(0.035, 0.009))
        component(self, "tail_motor", pos_m=[spec.tail_root_x_m + 0.016, 0, 0.125])
        component(
            self,
            "tail_shell_mass",
            mass_kg=spec.tail_mass_g / 1000,
            pos_m=[-0.43 * spec.tail_length_m, 0, 0.025],
            inertia_box_halfsize_m=[spec.tail_length_m / 2, 0.030, 0.035],
        )
        points = (
            (0, 0, 0),
            (-0.33 * spec.tail_length_m, 0, 0.010),
            (-0.70 * spec.tail_length_m, 0, 0.035),
            (-spec.tail_length_m, 0, 0.065),
        )
        for i, radius in enumerate((0.038, 0.030, 0.019)):
            attrs(self, "geom", f"tail_segment_{i}_collision", fromto=(*points[i], *points[i + 1]), size=radius)
        # The base builder has already attached tail_shell to this body.
        assert find(self, "geom", "tail_shell") in list(tail)

    def head(self) -> None:
        t, spec = self.torso, self.layout
        yaw_pos = (spec.neck_x_m, 0, spec.neck_z_m)
        self.component(t, "head_yaw_motor", "head_motors_2_x_82g", 82, yaw_pos, (0.01425, 0.017, 0.02325))
        self.geom(t, "neck_yaw_case", "cylinder", (0.025, 0.018), yaw_pos, "graphite", group=2)
        yaw = self.body(t, "head_yaw_body", yaw_pos)
        self.joint(yaw, "head_yaw", (0, 0, 1), (-0.65, 0.65), 0.6, 6, 0.25)
        nod_pos = (spec.nod_reach_m, 0, 0.030)
        self.component(yaw, "head_nod_motor", "head_motors_2_x_82g", 82, nod_pos, (0.01425, 0.02325, 0.017))
        self.geom(yaw, "neck_nod_case", "box", (0.01425, 0.02325, 0.017), nod_pos, "graphite", group=2)
        head = self.body(yaw, "head", nod_pos)
        self.joint(head, "head_nod", (0, 1, 0), (-0.25, 0.20), 0.6, 6, 0.25)
        length = spec.skull_length_m
        self.mesh(head, "head_shell", (1, 1, 1), (0, 0, 0), source="skull")
        self.collision(head, "head", "box", (length / 2, 0.080, 0.034), (length / 2 - 0.045, 0, 0.035))
        self.component(
            head, "head_upper_shell", "shells_and_guards", 220, (0.055, 0, 0.035), (length / 2, 0.080, 0.030)
        )
        self.component(head, "head_structure", "load_bearing_structure", 50, (0.025, 0, 0.015), (0.055, 0.035, 0.020))
        self.component(head, "head_bearings", "bearings_feet_fasteners", 39, (0, 0, 0), (0.020, 0.030, 0.020))
        jaw_pivot = (-0.025, 0, -0.015)
        self.component(head, "jaw_motor", "character_motors_4_x_18g", 18, jaw_pivot, (0.013, 0.010, 0.017))
        jaw = self.body(head, "jaw", jaw_pivot)
        self.joint(jaw, "jaw_pitch", (0, 1, 0), (0, 0.28), 0.1, 1.5, 0.04)
        self.mesh(jaw, "jaw_shell", (1, 1, 1), (0, 0, 0), "cream", source="jaw")
        self.collision(jaw, "jaw", "box", (0.1125, 0.072, 0.015), (0.1125, 0, -0.008))
        self.component(jaw, "jaw_shell_mass", "shells_and_guards", 60, (0.105, 0, -0.006), (0.115, 0.072, 0.015))
        # Lens origins lie beyond the closed front face of the skull loft.
        camera_x = length - 0.041
        self.component(head, "snout_camera", "camera", 61, (camera_x - 0.015, 0, 0.034), (0.00875, 0.0455, 0.014))
        self.geom(
            head,
            "camera_package_envelope",
            "box",
            (0.00875, 0.0455, 0.014),
            (camera_x - 0.015, 0, 0.034),
            "metal",
            group=4,
        )
        self.mesh(head, "snout_camera_panel", (0.004, 0.049, 0.020), (camera_x, 0, 0.034), "graphite", source="rounded")
        for label, y in (("left", 0.035), ("rgb", 0), ("right", -0.035)):
            for suffix, radius, material, offset in (("rim", 0.010, "metal", 0.004), ("lens", 0.0075, "glass", 0.006)):
                self.geom(
                    head,
                    f"snout_{label}_{suffix}",
                    "cylinder",
                    (radius, 0.002),
                    (camera_x + offset, y, 0.034),
                    material,
                    euler=(0, math.pi / 2, 0),
                )
            element(
                head,
                "camera",
                name="snout_" + label,
                pos=(camera_x + 0.010, y, 0.034),
                xyaxes=(0, -1, 0, 0, 0, 1),
                fovy=69,
            )
        for side, sign in (("left", 1), ("right", -1)):
            for name, size, pos, material in (
                ("eye_socket", (0.028, 0.008, 0.027), (0.015, sign * 0.076, 0.047), "dark_sage"),
                ("eye_white", (0.023, 0.006, 0.023), (0.017, sign * 0.083, 0.047), "cream"),
                ("iris", (0.019, 0.005, 0.020), (0.020, sign * 0.088, 0.047), "amber"),
                ("pupil", (0.012, 0.004, 0.017), (0.023, sign * 0.092, 0.047), "black"),
                ("eye_glint", (0.004, 0.002, 0.005), (0.029, sign * 0.096, 0.055), "white"),
                ("nostril", (0.008, 0.002, 0.004), (camera_x - 0.025, sign * 0.065, 0.048), "dark_sage"),
            ):
                self.geom(head, side + "_" + name, "ellipsoid", size, pos, material)


def build(directory: Path = HERE / "long_body", layout: Layout = Layout()) -> Path:
    """Write an isolated, deterministic candidate while retaining the old model."""
    if not all(math.isfinite(float(value)) for value in layout.as_dict().values()):
        raise ValueError("Layout values must be finite.")
    if not 0.20 <= layout.skull_length_m <= 0.35 or not 0.30 <= layout.tail_length_m <= 0.55:
        raise ValueError("Skull and tail lengths exceed this candidate's declared exploration range.")
    for name, minimum in (
        ("torso_shell_mass_g", 130),
        ("pelvis_structure_mass_g", 300),
        ("neck_support_mass_g", 50),
        ("tail_mass_g", 80),
    ):
        if getattr(layout, name) < minimum:
            raise ValueError(f"{name} must retain at least the baseline {minimum} g allocation.")
    directory.mkdir(parents=True, exist_ok=True)
    assets = directory / "assets"
    base.make_meshes(assets)
    loft(
        assets / "tail.obj",
        [
            (-layout.tail_length_m, 0.065, 0.002, 0.003),
            (-0.70 * layout.tail_length_m, 0.035, 0.014, 0.018),
            (-0.33 * layout.tail_length_m, 0.010, 0.025, 0.029),
            (0.015, 0, 0.037, 0.035),
        ],
    )
    length = layout.skull_length_m
    loft(
        assets / "skull.obj",
        [
            (-0.045, 0.032, 0.040, 0.020),
            (0.005, 0.036, 0.080, 0.033),
            (0.100, 0.032, 0.076, 0.028),
            (length - 0.045, 0.028, 0.055, 0.022),
        ],
        exponent=0.85,
    )
    loft(
        assets / "jaw.obj",
        [
            (0.0, -0.006, 0.040, 0.009),
            (0.050, -0.009, 0.072, 0.014),
            (0.160, -0.006, 0.067, 0.013),
            (0.225, -0.004, 0.047, 0.009),
        ],
        exponent=0.85,
    )
    b = Builder(layout)
    b.leg("left", 1)
    b.leg("right", -1)
    b.upper_body()
    compiler = b.xml.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", "assets")
    b.write(
        directory,
        imu_site="pelvis_imu",
        contact_sites=[
            (f"{side}_contact_force_{i}", f"{side}_contact_{i}") for side in ("left", "right") for i in range(4)
        ],
        limitations=[
            "Unweighed packaging candidate with the original medium leg mechanism, motor cases, torque caps and joint ranges.",
            "Existing component allocations are retained; longer torso, neck and tail receive explicit additional mass.",
            "The 600 g integration reserve is retained. Component box inertias use the parallel-axis theorem.",
            "Three torso lobes represent occupied envelopes around an open lower hip bay, not qualified fabrication CAD.",
            "Bearing capacities, wiring clearances, bracket strength and thermal performance are unverified.",
            "Local leg actuators are retained. Crouch-to-stand and self-righting require separate motion evidence.",
        ],
        metadata={
            "layout": layout.as_dict(),
            "reference_model": "examples/tyrannosaurus_rex_50cm/model.xml",
            "candidate": "long_body",
        },
    )
    return directory / "model.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "long_body")
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
