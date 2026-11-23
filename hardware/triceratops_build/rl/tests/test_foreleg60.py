"""The 128 and 160 mm foreleg models (sim/build_foreleg60.py; F18, D22): reproducible from the generator, refused on a
stale base, packaged without the 188 mm overlap, loadable by the training env with run 3's interface."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from conftest import RL_DIR  # noqa: F401  (puts rl/ on sys.path)

SIM = RL_DIR.parent / "sim"
GEN = SIM / "generated"
sys.path.insert(0, str(SIM))

import build_foreleg60 as bf  # noqa: E402


def test_generator_reproduces_the_committed_variants(tmp_path):
    p = subprocess.run([sys.executable, str(SIM / "build_model60.py"), "--out", str(tmp_path), "--capsule-foot",
                        "--foreleg", "128", "160"], capture_output=True, text=True, cwd=str(SIM), timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    for n in ("trike60_C_closed_capsule", "trike60_C_closed_capsule_fore128", "trike60_C_closed_capsule_fore160"):
        assert (tmp_path / f"{n}.xml").read_bytes() == (GEN / f"{n}.xml").read_bytes(), n
        assert (tmp_path / f"mass_ledger_{n}.json").read_bytes() == (GEN / f"mass_ledger_{n}.json").read_bytes(), n


def test_edit_refuses_a_base_that_moved_on(tmp_path):
    root = ET.parse(bf.BASE).getroot()
    strut = next(e for e in root.iter("geom") if e.get("name") == "left_fore_upper_strut")
    strut.set("fromto", "0 0 0 -0.1 0 -0.15")  # a base design the change set was not taken from
    stale = tmp_path / "stale.xml"
    stale.write_text(ET.tostring(root, encoding="unicode"))
    with pytest.raises(SystemExit, match="differs from the one PR #697's change set was taken from"):
        bf.edit(128, base=stale)


def test_packaging_clears_at_128_and_160_and_not_at_188():
    base = bf.packaging(bf.BASE)
    assert all(v["x6_to_upper_strut_m"] < -0.04 for v in base.values())  # F18: the 188 mm X6 overlaps its strut
    for n in bf.LENGTHS:
        r = bf.packaging(GEN / f"trike60_C_closed_capsule_fore{n}.xml")
        for side in r.values():
            assert side["x6_to_upper_strut_m"] > 0.005 and side["x6_to_chassis_min_m"] > 0.005


def test_variants_keep_run3s_interface_and_geometry_where_it_should():
    import mujoco

    from trike_env import TrikeEnv, TrikeEnvConfig

    ref = mujoco.MjModel.from_xml_path(str(bf.BASE))
    shapes = None
    for n in bf.LENGTHS:
        path = GEN / f"trike60_C_closed_capsule_fore{n}.xml"
        m = mujoco.MjModel.from_xml_path(str(path))
        assert [m.joint(i).name for i in range(m.njnt)] == [ref.joint(i).name for i in range(ref.njnt)]
        assert [m.actuator(i).name for i in range(m.nu)] == [ref.actuator(i).name for i in range(ref.nu)]
        d, dr = mujoco.MjData(m), mujoco.MjData(ref)
        for mm, dd in ((m, d), (ref, dr)):
            mujoco.mj_resetDataKeyframe(mm, dd, 0)
            mujoco.mj_kinematics(mm, dd)
        for c in ("left_fore", "right_fore"):
            sole = lambda mm, dd: dd.geom_xpos[mm.geom(f"{c}_sole").id]  # noqa: E731
            assert np.allclose(sole(m, d), sole(ref, dr), atol=1e-6)  # the neutral foot stays put
            sh, el = (d.xanchor[m.joint(f"{c}_{j}").id] for j in ("pitch", "flexion"))
            assert np.linalg.norm(sh - el) == pytest.approx(n / 1000, abs=1e-6)
        for c in ("left_hind", "right_hind"):  # the hind legs are untouched
            assert np.allclose(d.xanchor[m.joint(f"{c}_pitch").id], dr.xanchor[ref.joint(f"{c}_pitch").id])
        env = TrikeEnv(TrikeEnvConfig(model_path=str(path)))
        obs, _ = env.reset(seed=0)
        shapes = shapes or (obs.shape, env.action_space.shape)
        assert (obs.shape, env.action_space.shape) == shapes == (
            TrikeEnv(TrikeEnvConfig(model_path=str(bf.BASE))).observation_space.shape, (12,))


def test_variants_match_the_prs_models_when_its_commit_is_here():
    have = subprocess.run(["git", "-C", str(bf.REPO), "cat-file", "-e", f"{bf.PR_COMMIT}^{{commit}}"],
                          capture_output=True).returncode == 0
    if not have:
        pytest.skip("PR #697's commit is not fetched in this clone")
    assert bf.check() == []
