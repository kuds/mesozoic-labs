"""Physical invariants of the shared component and visual-surface builders."""

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from environments.shared.robot_building import component_inertia, loft, superellipsoid_mesh, tapered_tail_mesh


@pytest.mark.parametrize("origin", [(0, 0, 0), (3, 5, -2)])
def test_component_inertia_matches_two_offset_unit_cubes(origin):
    # Two 3 kg, one-metre cubes at opposite offsets have an analytical central
    # inertia. Translating the assembly must translate its COM, not its inertia.
    components = [
        dict(mass_kg=3.0, pos_m=(np.array(origin) + offset).tolist(), inertia_box_halfsize_m=[0.5] * 3)
        for offset in (np.array([-1, -2, 0]), np.array([1, 2, 0]))
    ]
    mass, center, inertia = component_inertia(components)
    assert mass == 6
    assert center == pytest.approx(origin)
    assert inertia == pytest.approx(np.array([[25, -12, 0], [-12, 7, 0], [0, 0, 31]]))


@pytest.mark.parametrize("shape", ["rounded", "taper", "loft"])
def test_generated_meshes_are_closed_indexed_surfaces(tmp_path: Path, shape: str):
    path = tmp_path / "shape.obj"
    if shape == "rounded":
        superellipsoid_mesh(path, exponent=0.7, columns=8, rings=4)
    elif shape == "taper":
        tapered_tail_mesh(path, length=0.5, rise=0.1, base_radius=0.05, tip_radius=0.005, taper=0.9, columns=8, rings=4)
    else:
        loft(path, [(-0.2, 0.0, 0.1, 0.1), (0.1, 0.2, 0.2, 0.1), (0.4, 0.0, 0.1, 0.1)])
    rows = [line.split() for line in path.read_text().splitlines()]
    vertices = np.array([[float(v) for v in row[1:]] for row in rows if row[0] == "v"])
    faces = [[int(v) - 1 for v in row[1:]] for row in rows if row[0] == "f"]
    assert np.isfinite(vertices).all()
    assert faces
    edges: Counter[tuple[int, int]] = Counter()
    for face in faces:
        assert len(set(face)) == 3
        assert all(0 <= i < len(vertices) for i in face)
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            edges[min(a, b), max(a, b)] += 1
    assert set(edges.values()) == {2}


@pytest.mark.parametrize("module", ["robot_building", "robot_geometry", "robot_rendering"])
def test_shared_robot_utilities_do_not_import_checkout_examples(module):
    """Keep dependencies directed from model adapters to reusable utilities."""
    path = Path(__file__).parents[1] / f"{module}.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "examples" for alias in node.names), path
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "examples", path
