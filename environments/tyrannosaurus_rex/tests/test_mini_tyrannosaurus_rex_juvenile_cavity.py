"""Exterior enclosure must reject protrusions even when sampled points fit."""

import itertools
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from examples.mini_tyrannosaurus_rex.juvenile import cavity


def _model_and_manifest(profile, *, half=(0.1, 0.1, 0.1), position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
    def values(numbers):
        return " ".join(map(str, numbers))

    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body name="torso"><freejoint/>'
        f'<geom name="installation" type="box" size="{values(half)}" '
        f'pos="{values(position)}" quat="{values(quaternion)}"/>'
        f"</body></worldbody></mujoco>"
    )
    vertices, faces = cavity._torso_loft(profile)
    manifest = dict(
        hardware=[dict(envelope_geom="installation")],
        exterior_containment=dict(
            references=dict(torso=dict(owner_body="torso", vertices_m=vertices.tolist(), triangles=faces.tolist())),
            checks=[dict(hardware="part", envelope_geom="installation", reference="torso")],
        ),
    )
    return model, manifest


def test_installed_box_inside_actual_reference_and_moved_outside():
    model, manifest = _model_and_manifest([[-1, 0, 1, 1], [1, 0, 1, 1]])
    assert cavity.exterior_containment_screen(model, manifest)["passed"]
    model, manifest = _model_and_manifest([[-1, 0, 1, 1], [1, 0, 1, 1]], position=(0, 2, 0))
    result = cavity.exterior_containment_screen(model, manifest)
    assert not result["passed"]
    assert not result["checks"][0]["center_inside"]


def test_center_inside_does_not_hide_rotated_corner_protrusion():
    angle = np.pi / 4
    model, manifest = _model_and_manifest(
        [[-1, 0, 0.3, 1], [1, 0, 0.3, 1]],
        half=(0.4, 0.1, 0.1),
        quaternion=(np.cos(angle / 2), 0, 0, np.sin(angle / 2)),
    )
    result = cavity.exterior_containment_screen(model, manifest)
    assert result["checks"][0]["center_inside"]
    assert result["checks"][0]["intersecting_reference_triangles"] > 0
    assert not result["passed"]


def test_nonconvex_neck_rejects_box_even_when_every_corner_and_center_is_inside():
    model, manifest = _model_and_manifest([[-1, 0, 1, 1], [0, 0, 0.2, 1], [1, 0, 1, 1]], half=(0.9, 0.5, 0.1))
    reference = manifest["exterior_containment"]["references"]["torso"]
    vertices, faces = cavity._checked_mesh(reference["vertices_m"], reference["triangles"])
    half = model.geom_size[model.geom("installation").id]
    for signs in itertools.product((-1, 1), repeat=3):
        assert cavity._inside(np.asarray(signs) * half, vertices[faces])[0]
    result = cavity.exterior_containment_screen(model, manifest)
    assert result["checks"][0]["center_inside"]
    assert result["checks"][0]["intersecting_reference_triangles"] > 0
    assert not result["passed"]


def test_missing_hardware_reference_cannot_pass():
    model, manifest = _model_and_manifest([[-1, 0, 1, 1], [1, 0, 1, 1]])
    manifest["hardware"].append(dict(envelope_geom="uninspected_part"))
    result = cavity.exterior_containment_screen(model, manifest)
    assert result["missing_envelopes"] == ["uninspected_part"]
    assert not result["complete"]
    assert not result["passed"]


def test_open_reference_mesh_is_rejected():
    model, manifest = _model_and_manifest([[-1, 0, 1, 1], [1, 0, 1, 1]])
    manifest["exterior_containment"]["references"]["torso"]["triangles"].pop()
    with pytest.raises(ValueError, match="closed two-manifold"):
        cavity.exterior_containment_screen(model, manifest)


def test_reference_must_match_generated_torso_skin():
    profile = [[-1, 0, 1, 1], [1, 0, 1, 1]]
    vertices, faces = cavity._torso_loft(profile)
    root = ET.fromstring('<mujoco><asset/><worldbody><body name="torso"/></worldbody></mujoco>')
    mesh = ET.SubElement(
        root.find("asset"),
        "mesh",
        name="juv_torso_skin",
        vertex=" ".join(map(str, vertices.ravel())),
        face=" ".join(map(str, faces.ravel())),
    )
    hardware = dict(hardware=[dict(name="part", envelope_geom="installation", owner_body="torso")])
    assert cavity.apply_cavities(root, hardware, dict(profile_m=profile), {})["checks"]
    vertices[0, 0] += 0.001
    mesh.set("vertex", " ".join(map(str, vertices.ravel())))
    with pytest.raises(ValueError, match="no longer matches"):
        cavity.apply_cavities(root, hardware, dict(profile_m=profile), {})
