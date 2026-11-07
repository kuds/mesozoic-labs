"""Regression tests for the round-one design study.

Run from hardware/trex_form_study:  python -m pytest tests -q
(Kept outside the repository's CI test paths: the study package is self-contained.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import designlab  # noqa: E402,F401  (selects the GL backend before mujoco)
from designlab import geometry, statics  # noqa: E402
from designlab.params import HEIGHT_CLASSES, parse  # noqa: E402
from designlab.pipeline import build_candidate  # noqa: E402

PR662 = ROOT / "r0_posture_study_2026-11-07" / "source" / "pr662" / "model.xml"


def test_shell_properties_match_closed_form_sphere():
    # thin spherical shell: m = 4 pi r^2 t rho, I = 2/3 m r^2
    r, t, rho = 0.1, 0.002, 1000.0
    u = np.linspace(0, np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 120)
    grid = np.stack([np.stack([r * np.sin(a) * np.cos(v), r * np.sin(a) * np.sin(v), np.full_like(v, r * np.cos(a))], -1)
                     for a in u])
    tris = geometry.grid_triangles(grid, close_ring=True)
    m, com, inertia, area = geometry.shell_properties(tris, t, rho)
    assert area == pytest.approx(4 * np.pi * r * r, rel=2e-3)
    assert m == pytest.approx(4 * np.pi * r * r * t * rho, rel=2e-3)
    assert np.allclose(com, 0, atol=1e-6)
    assert np.allclose(np.diag(inertia), 2 / 3 * m * r * r, rtol=5e-3)


@pytest.mark.parametrize("cid", ["60-M-S-10", "100-M-H-10"])
def test_crown_matches_height_class(tmp_path, cid):
    res = build_candidate(parse(cid), tmp_path)
    assert res["summary"]["crown"] == pytest.approx(HEIGHT_CLASSES[parse(cid).height], abs=0.003)
    assert res["summary"]["stance_ok"]


def test_build_is_deterministic(tmp_path):
    a = build_candidate(parse("75-N-T-20"), tmp_path / "a")["summary"]["xml_sha256"]
    b = build_candidate(parse("75-N-T-20"), tmp_path / "b")["summary"]["xml_sha256"]
    assert a == b


def test_feet_flat_on_floor_and_com_over_support(tmp_path):
    res = build_candidate(parse("120-W-S-00"), tmp_path)
    M = res["model"]
    M.forward(res["q_home"])
    for side in statics.SIDES:
        p, R = M.foot_pose(side)
        assert np.allclose(R, np.eye(3), atol=1e-4)
    poly = np.vstack([M.pad_polygon(s) for s in statics.SIDES])
    from scipy.spatial import ConvexHull

    hull = poly[ConvexHull(poly).vertices]
    assert statics.support_margin(M.com()[:2], hull) > 0.01


@pytest.mark.skipif(not PR662.exists(), reason="PR #662 nominal model not present")
def test_pr662_double_support_torques_reproduced():
    """designlab.statics reproduces PR #662's published double-support holds (hip 1.48, knee 3.35,
    ankle 1.19 N·m in the uploaded study's table)."""
    M = statics.Model.load(str(PR662))
    q = M.m.qpos0.copy()
    tau, _ = statics.contact_torques(M, q, support=("left", "right"))
    t = dict(zip(M.names, np.abs(tau), strict=True))
    assert t["left_hip_pitch"] == pytest.approx(1.48, abs=0.05)
    assert t["left_knee"] == pytest.approx(3.35, abs=0.08)
    assert t["left_ankle_pitch"] == pytest.approx(1.19, abs=0.05)
