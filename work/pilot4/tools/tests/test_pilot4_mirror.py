"""Tests of pilot4_mirror.py (pytest; walking pilot 4 scratch, never merged).

Pure functions on synthetic inputs (the signed permutation, the symmetrised statistics, the mirror-KL loss in numpy,
the per-episode pairing and its summary), the refusals (output places, cells, parents, the order of the checks), and
pins of the constants against their sources (v3's parent table; the veer diagnosis's recorded numbers) when those
files are on disk. Nothing here rolls an env, imports the repository or trains: the validation runs on pilot 3's 8M
cells are made by the CLI (out/readouts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot4_mirror as pm  # noqa: E402

SP = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad")
V3 = SP / "work/pilot4/postfix/fix/criteria_pilot4_draft_v3.txt"
VEER = SP / "work/veer"


def random_involution(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """A signed permutation M with M M = I: random disjoint swaps, the rest fixed; signs consistent over pairs."""
    perm = np.arange(n)
    idx = rng.permutation(n)
    for a, b in zip(idx[0::2][: n // 3], idx[1::2][: n // 3]):
        perm[a], perm[b] = b, a
    sign = np.ones(n)
    for i in range(n):
        j = perm[i]
        if j >= i:
            s = rng.choice([-1.0, 1.0])
            sign[i] = sign[j] = s
    return perm, sign


# ---------------------------------------------------------------- the maps and the loss


def test_signed_perm_apply_is_an_involution_and_keeps_float32():
    rng = np.random.default_rng(0)
    perm, sign = random_involution(63, rng)
    x = rng.normal(size=(5, 63)).astype(np.float32)
    y = pm.signed_perm_apply(x, perm, sign)
    assert y.dtype == np.float32
    assert np.array_equal(pm.signed_perm_apply(y, perm, sign), x)
    assert np.array_equal(y[:, 0], x[:, perm[0]] * sign[0])


def test_symmetrised_statistics_are_fixed_points_of_the_map():
    rng = np.random.default_rng(1)
    perm, sign = random_involution(63, rng)
    mean, var = rng.normal(size=63), rng.uniform(0.1, 2.0, size=63)
    ms, vs = pm.symmetrised_stats(mean, var, perm, sign)
    assert np.allclose(pm.signed_perm_apply(ms, perm, sign), ms)
    assert np.allclose(vs[perm], vs)
    # a channel whose mirror is itself with a sign flip has mean 0 under the symmetrised statistics
    k = int(np.flatnonzero((perm == np.arange(63)) & (sign == -1.0))[0]) if np.any(
        (perm == np.arange(63)) & (sign == -1.0)) else None
    if k is not None:
        assert ms[k] == 0.0


def test_normalize_is_the_probes():
    x = np.array([[1.0, -20.0, 3.0]])
    out = pm.normalize(x, np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 4.0]), 1e-8, 10.0)
    assert out.dtype == np.float32
    assert np.allclose(out, [[1.0, -10.0, 1.0]], atol=1e-6)


def test_lk_numpy_is_zero_for_an_equivariant_policy_and_matches_a_hand_value():
    rng = np.random.default_rng(2)
    perm, sign = random_involution(15, rng)
    mu = rng.normal(size=(7, 15))
    mu_m = mu[:, perm] * sign  # exactly M_a mu
    log_std = np.full(15, np.log(0.2))
    assert pm.lk_numpy(mu_m, mu, log_std, perm, sign) == (0.0, 0.0)
    # one row, one deviation of 0.1 on actuator 0 with sigma 0.2: 0.01 / (2 * 0.04) = 0.125; asymmetry 0.05
    mu1 = np.zeros((1, 15))
    mu1m = np.zeros((1, 15))
    mu1m[0, 0] = 0.1
    loss, asym = pm.lk_numpy(mu1m, mu1, log_std, np.arange(15), np.ones(15))
    assert loss == pytest.approx(0.125) and asym == pytest.approx(0.05)


def test_lk_numpy_clamps_sigma_at_the_floor():
    mu = np.zeros((1, 2))
    mu_m = np.array([[0.02, 0.0]])
    loss, _ = pm.lk_numpy(mu_m, mu, np.array([-20.0, -20.0]), np.arange(2), np.ones(2), sigma_floor=0.01)
    assert loss == pytest.approx(0.02 ** 2 / (2 * 0.01 ** 2))


# ---------------------------------------------------------------- pairing pi with pi_m


def row(seed, yaw, length=1000, terminated=False):
    return {"reset_seed": seed, "yaw_change_deg": yaw, "length": length, "terminated": terminated}


def test_pairing_gives_handedness_scene_part_and_flip():
    pairs = pm.pair_episodes([row(3042, -30.0), row(3043, -10.0)], [row(3042, 20.0), row(3043, -2.0)], 1000)
    assert pairs[0]["handedness_deg"] == -25.0 and pairs[0]["scene_part_deg"] == -5.0 and pairs[0]["flipped"]
    assert pairs[1]["handedness_deg"] == -4.0 and pairs[1]["scene_part_deg"] == -6.0 and not pairs[1]["flipped"]


def test_a_fall_on_either_side_leaves_the_episode_out_of_the_split():
    pairs = pm.pair_episodes([row(3042, -3.0, length=114, terminated=True)], [row(3042, 40.0)], 1000)
    assert pairs[0]["both_full"] is False and pairs[0]["handedness_deg"] is None
    s = pm.summarize_pairs(pairs)
    assert s["both_full"] == 0 and s["handedness_deg"] is None and s["flipped"] == 1


def test_an_episode_without_a_yaw_is_not_comparable():
    """A fall before the window (s52's reset 3050 at 90 steps in the pilot-4 env) has no yaw_change_deg."""
    pairs = pm.pair_episodes([row(3050, None, length=90, terminated=True), row(3051, -20.0)],
                             [row(3050, 30.0), row(3051, 25.0)], 1000)
    s = pm.summarize_pairs(pairs)
    assert pairs[0]["flipped"] is None and pairs[0]["handedness_deg"] is None
    assert s["comparable"] == 1 and s["flipped"] == 1 and s["episodes"] == 2 and s["pi_right"] == 1


def test_pairing_refuses_out_of_step_episodes():
    with pytest.raises(ValueError, match="out of step"):
        pm.pair_episodes([row(3042, 1.0)], [row(3043, 1.0)], 1000)


def test_summary_counts_turns_on_each_side():
    pi = [row(3042 + i, y) for i, y in enumerate([-30.0, -20.0, -10.0, 5.0])]
    pim = [row(3042 + i, y) for i, y in enumerate([25.0, 15.0, 12.0, -3.0])]
    s = pm.summarize_pairs(pm.pair_episodes(pi, pim, 1000))
    assert (s["pi_right"], s["pi_left"], s["pim_right"], s["pim_left"], s["flipped"]) == (3, 1, 1, 3, 4)
    assert s["handedness_deg"]["median"] == pytest.approx(np.median([-27.5, -17.5, -11.0, 4.0]))
    assert pm.min_med_max([None, float("nan")]) is None


# ---------------------------------------------------------------- refusals


def test_output_refusals(tmp_path):
    tree = tmp_path / "tree"
    (tree / "environments").mkdir(parents=True)
    run = tmp_path / "runs" / "pilot4_walk_s42"
    (run / "03_locomotion" / "models").mkdir(parents=True)
    (run / "provenance.json").write_text("{}")
    ckpt = run / "03_locomotion" / "models" / "stage2_2000000_steps.zip"
    existing = tmp_path / "old.json"
    existing.write_text("{}")
    ok = tmp_path / "out" / "m.json"
    assert pm.output_problems(ok, tree=tree, inputs=[ckpt], run_copy=str(run)) == []
    assert "exists" in pm.output_problems(existing, tree=tree, inputs=[], run_copy=None)[0]
    assert "inside the tree" in pm.output_problems(tree / "x.json", tree=tree, inputs=[], run_copy=None)[0]
    msgs = pm.output_problems(run / "x.json", tree=tree, inputs=[ckpt], run_copy=str(run))
    assert any("run copy" in m for m in msgs) and any("run directory" in m for m in msgs)
    # an input's run directory alone (no run copy named) is enough
    assert any("run directory" in m for m in pm.output_problems(run / "03_locomotion" / "y.json", tree=None,
                                                                    inputs=[ckpt], run_copy=None))


def binding(**over):
    b = {"schema": pm.PILOT4_CELL_SCHEMA, "head": "abc", "off_schedule": False, "validation_only": False}
    b.update(over)
    return b


def test_cell_refusals():
    assert pm.cell_problems(binding(), head="abc", dirty="", validation=False) == []
    assert "another" not in " ".join(pm.cell_problems(binding(), head="abc", dirty="", validation=False))
    assert any("judged at" in p for p in pm.cell_problems(binding(), head="def", dirty="", validation=False))
    assert any("local edits" in p for p in pm.cell_problems(binding(), head="abc", dirty=" M x", validation=False))
    assert any("off-schedule" in p for p in pm.cell_problems(binding(off_schedule=True, off_schedule_reason="r"),
                                                             head="abc", dirty="", validation=False))
    assert any("validation-only" in p for p in pm.cell_problems(binding(validation_only=True), head="abc", dirty="",
                                                                validation=False))
    assert pm.cell_problems(binding(validation_only=True), head="abc", dirty="", validation=True) == []
    p3 = binding(schema=pm.PILOT3_CELL_SCHEMA)
    assert any("pilot 3's" in p for p in pm.cell_problems(p3, head="abc", dirty="", validation=False))
    assert pm.cell_problems(p3, head="abc", dirty="", validation=True) == []
    assert any("not a judge cell" in p for p in pm.cell_problems(binding(schema="x"), head="abc", dirty="",
                                                                 validation=True))


def test_parent_refusals(tmp_path):
    z, p = pm.parent_paths(tmp_path, 42)
    assert z == tmp_path / "pilot3_walk_s42/03_locomotion/models/stage2_8000000_steps.zip"
    assert p.name == "stage2_vecnormalize_8000000_steps.pkl"
    assert "not one of the scored seeds" in pm.parent_problems(43, z, p)[0]
    assert all("missing" in m for m in pm.parent_problems(42, z, p))
    z.parent.mkdir(parents=True)
    z.write_bytes(b"not the parent")
    p.write_bytes(b"not the sidecar")
    assert all("is not s42's 8M parent" in m for m in pm.parent_problems(42, z, p))


def test_existing_json_is_refused_before_the_tree_is_imported(tmp_path, monkeypatch):
    out = tmp_path / "m.json"
    out.write_text("{}")
    monkeypatch.chdir(tmp_path)
    before = "environments" in sys.modules
    assert pm.main(["--parent", "42", "--json", str(out)]) == 2
    assert ("environments" in sys.modules) == before  # nothing imported
    assert pm.main(["--parent", "42", "--json", str(tmp_path / "n.json"), "--episodes", "0"]) == 2
    assert not (tmp_path / "n.json").exists()


def test_a_cell_is_refused_on_a_synthetic_tree_before_anything_rolls(tmp_path, monkeypatch, capsys):
    """The CLI on a stand-in tree (an empty `environments` package) and a cell whose binding names another head and
    a missing pair: every problem is listed, nothing is written, the exit status is 2."""
    tree = tmp_path / "tree"
    (tree / "environments").mkdir(parents=True)
    (tree / "environments" / "__init__.py").write_text("")
    cell = tmp_path / "OUT" / "s42" / "2000000"
    cell.mkdir(parents=True)
    (cell / "binding.json").write_text(json.dumps(binding(head="feed", seed=42, steps=2000000,
                                                          checkpoint=str(tmp_path / "no.zip"),
                                                          checkpoint_sha256="sha256:0", vecnorm=str(tmp_path / "no.pkl"),
                                                          vecnorm_sha256="sha256:0", run=str(tmp_path / "run"))))
    (cell / "probe.json").write_text(json.dumps({"meta": {"seed_scheme": "panel", "seed_base": 3042},
                                                 "episodes": [{}] * 40}))
    saved = sys.modules.pop("environments", None)
    monkeypatch.syspath_prepend(str(tree))
    monkeypatch.chdir(tree)
    monkeypatch.setattr(pm, "git", lambda t, *a: "abc" if a[0] == "rev-parse" else "")
    try:
        rc = pm.main(["--cell", str(cell), "--json", str(tmp_path / "res" / "m.json")])
    finally:
        sys.modules.pop("environments", None)
        if saved is not None:
            sys.modules["environments"] = saved
    text = capsys.readouterr().out
    assert rc == 2 and "judged at feed" in text and "no longer the judged file" in text
    assert not (tmp_path / "res").exists()


# ---------------------------------------------------------------- pins against the sources


@pytest.mark.skipif(not V3.is_file(), reason="v3 not on disk")
def test_the_parent_table_is_v3s():
    text = V3.read_text()
    for seed, (z, p) in pm.PARENTS.items():
        assert z.split(":", 1)[1] in text and p.split(":", 1)[1] in text, seed


@pytest.mark.skipif(not (VEER / "screen/out/lk_check.txt").is_file(), reason="veer screen not on disk")
def test_the_lk_references_are_the_screens_records():
    text = (VEER / "screen/out/lk_check.txt").read_text()
    for seed in (42, 52, 60):
        line = [ln for ln in text.splitlines() if ln.startswith(f"s{seed}_8M")][0]
        cols = [c.strip() for c in line.split("|")]
        assert float(cols[3]) == pm.PILOT3_REFERENCE["lk_own"][seed]
        assert float(cols[4]) == pm.PILOT3_REFERENCE["lk_symmetrised"][seed]


@pytest.mark.skipif(not (VEER / "policy/out/aggregate3.json").is_file(), reason="veer policy lane not on disk")
def test_the_handedness_references_are_the_policy_lanes_records():
    agg3 = json.loads((VEER / "policy/out/aggregate3.json").read_text())
    for seed in (42, 52, 60):
        assert round(float(np.median(agg3["8000000"][str(seed)]["bias"])), 1) == \
            pm.PILOT3_REFERENCE["handedness_median_deg"][seed]


def test_constants_are_v3s_and_the_lanes():
    assert pm.MIRROR_EPISODES == 10 and pm.LK_EPISODES == 20 and pm.PANEL_SEED_BASE == 3042
    assert pm.LK_SIGMA_FLOOR == 0.01 and pm.LK_KAPPA == 0.3 and pm.COMMAND_SLICE == slice(60, 63)
    assert "report only" in pm.REPORT_ONLY
