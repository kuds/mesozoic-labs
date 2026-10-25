"""Geometry, visibility and fixed-scale animation checks for the shared renderer."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import GifImagePlugin, Image

from environments.shared import robot_rendering as rendering


def scene(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize(
    "kind,size,extent",
    [
        ("box", "0.1 0.2 0.3", [0.1, 0.2, 0.3]),
        ("sphere", "0.1", [0.1, 0.1, 0.1]),
        ("ellipsoid", "0.1 0.2 0.3", [0.1, 0.2, 0.3]),
        ("cylinder", "0.1 0.3", [0.1, 0.1, 0.3]),
        ("capsule", "0.1 0.3", [0.1, 0.1, 0.4]),
    ],
)
def test_primitive_surfaces_preserve_bounds_and_unit_normals(kind, size, extent):
    model, data = scene(f'<mujoco><worldbody><geom type="{kind}" size="{size}" group="1"/></worldbody></mujoco>')
    triangles, normals, colors = rendering.geometry(model, data)
    assert triangles.max(axis=(0, 1)) == pytest.approx(extent)
    assert triangles.min(axis=(0, 1)) == pytest.approx(-np.array(extent))
    assert np.linalg.norm(normals, axis=2) == pytest.approx(np.ones(normals.shape[:2]))
    assert len(triangles) == len(colors)


def test_mesh_surfaces_follow_compiled_geometry_and_transforms():
    model, data = scene("""<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
      <worldbody><geom type="mesh" mesh="tetra" pos="2 3 4" euler="0 0 90" group="1"/></worldbody>
    </mujoco>""")
    triangles, normals, _ = rendering.geometry(model, data)
    assert triangles.min(axis=(0, 1)) == pytest.approx([1, 3, 4], abs=1e-6)
    assert triangles.max(axis=(0, 1)) == pytest.approx([2, 4, 5], abs=1e-6)
    assert np.linalg.norm(normals, axis=2) == pytest.approx(np.ones(normals.shape[:2]), abs=1e-6)


def test_visibility_filters_do_not_mutate_the_model():
    model, data = scene("""<mujoco><worldbody>
      <geom name="shell" type="box" size="1 1 1" group="1"/>
      <geom name="motor" type="box" size=".1 .1 .1" pos="3 0 0" group="2"/>
      <geom name="package" type="box" size=".1 .1 .1" pos="5 0 0" group="4"/>
      <geom name="transparent" type="box" size="1 1 1" pos="9 0 0" group="1" rgba="1 0 0 .2"/>
    </worldbody></mujoco>""")
    groups = model.geom_group.copy()
    renderer = rendering.GeometryRenderer(views={"isometric": (1, -1, 0.5)}, service_panels="shell")
    home = renderer.geometry(model, data)[0]
    service = renderer.geometry(model, data, service=True)[0]
    renderer.render(model, data, service=True, width=80, height=90)
    assert home.min(axis=(0, 1))[0] == pytest.approx(-1)
    assert home.max(axis=(0, 1))[0] == pytest.approx(3.1)
    assert service.min(axis=(0, 1))[0] == pytest.approx(2.9)
    assert service.max(axis=(0, 1))[0] == pytest.approx(5.1)
    assert np.array_equal(model.geom_group, groups)


def test_depth_buffer_keeps_the_front_surface_visible():
    model, data = scene("""<mujoco><worldbody>
      <geom type="box" size=".2 .2 .2" pos="1 0 0" rgba="1 0 0 1" group="1"/>
      <geom type="box" size=".2 .2 .2" pos="0 0 0" rgba="0 1 0 1" group="1"/>
    </worldbody></mujoco>""")
    basis = rendering.camera_basis((1, 0, 0))
    assert basis @ basis.T == pytest.approx(np.eye(3))
    image = rendering.render(model, data, direction=(1, 0, 0), width=80, height=90)
    pixel = image.getpixel((40, 45))
    assert pixel[0] > pixel[1] + 100


def test_gesture_uses_fixed_projection_bounds_across_moving_frames(tmp_path: Path):
    model, data = scene("""<mujoco><option gravity="0 0 0"/>
      <worldbody><body><joint type="hinge" axis="0 1 0"/>
      <geom type="box" size=".1 .1 .2" pos="0 0 -.5" group="1"/></body></worldbody>
    </mujoco>""")
    data.qvel[0] = 1
    frames = []

    def renderer(current_model, current_data, **kwargs):
        bounds = kwargs["projection_bounds"]
        frames.append((current_data.qpos.copy(), bounds))
        return rendering.render(current_model, current_data, direction=(1, -1, 0.5), **kwargs)

    output = tmp_path / "gesture.gif"
    rendering.gesture_gif(
        model,
        data,
        output,
        controls=lambda model, elapsed: np.zeros(model.nu),
        renderer=renderer,
        direction=(1, -1, 0.5),
        title="Free joint",
        footer=lambda elapsed: f"{elapsed:.1f}s",
        frames=3,
        duration=0.3,
    )
    assert len(frames) == 3
    assert frames[-1][0][0] > frames[0][0][0] + 0.1
    for _, bounds in frames[1:]:
        assert np.array_equal(bounds, frames[0][1])
    with Image.open(output) as image:
        assert isinstance(image, GifImagePlugin.GifImageFile)
        assert image.n_frames == 3
        assert image.info["duration"] == 100


def test_four_view_sheet_preserves_panel_order_and_layout():
    model, data = scene("<mujoco><worldbody/></mujoco>")
    calls = []
    colors = [(40, 60, 80), (80, 60, 40), (20, 40, 60), (60, 40, 20)]

    def renderer(model, data, *, view, service, width, height):
        calls.append((view, service, width, height))
        return Image.new("RGB", (width, height), colors[len(calls) - 1])

    panels = [("iso", False, "Home"), ("side", False, "Side"), ("front", False, "Front"), ("iso", True, "Service")]
    sheet = rendering.four_view_sheet(
        model,
        data,
        renderer=renderer,
        title="Title",
        subtitle="Dimensions",
        panels=panels,
        footnotes=("Scope", "Limits"),
    )
    assert sheet.size == (1600, 1580)
    assert calls == [(view, service, 790, 605) for view, service, _ in panels]
    for i, color in enumerate(colors):
        assert sheet.getpixel((i % 2 * 800 + 100, 145 + i // 2 * 660 + 100)) == color
