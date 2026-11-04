"""Tests of pilot3_vg.py, the report-only V - G readout (pytest; walking pilot 3 scratch, never merged; FIX2).

Pure tests check the buckets, G, the per-bucket statistics, the step-localised rule and the self-checks on synthetic
episodes. Two integration tests run the tool through iso.py on the clean tree (PILOT3_TREE, default ../tree): a short
measurement (1 episode per env) of the notebook smoke's 65,536 pair (skipped when the smoke's files are absent), and
a --cell refusal that writes nothing. FIX3 (the third review's surviving mutants): the frozen settings, the SE and
4-SE rule at its boundary, G's bootstrap after a truncation only (episode_rows, measure()'s own step), the seeding
(seed_all, and the short measurement's record), the --cell refusals (cell_problems) and no overwrite of --json.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot3_vg as vg  # noqa: E402

TREE = Path(os.environ.get("PILOT3_TREE", str(TOOLS / "tree"))).resolve()
SMOKE = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/"
             "smoke/clone/logs/tyrannosaurus_rex/anatomical/ppo_quick_test/pilot3_walk_s42/03_locomotion/models")
SMOKE_ZIP, SMOKE_PKL = SMOKE / "stage2_65536_steps.zip", SMOKE / "stage2_vecnormalize_65536_steps.pkl"


@pytest.mark.parametrize("k, b", [(0, 0), (99, 0), (100, 1), (199, 1), (200, 2), (499, 2), (500, 3), (799, 3),
                                  (800, 4), (999, 4), (1000, 4)])
def test_bucket_edges_are_the_criteria_s(k, b):
    assert vg.bucket_of(k) == b
    assert vg.BUCKET_NAMES == ("[0,100)", "[100,200)", "[200,500)", "[500,800)", "[800,1000]")


@pytest.mark.parametrize("k", [-1, 1001, 1.5, True])
def test_bucket_of_refuses_a_counter_outside_the_episode(k):
    with pytest.raises(ValueError):
        vg.bucket_of(k)


def test_returns_to_go_with_and_without_the_time_limit_bootstrap():
    assert np.allclose(vg.returns_to_go([1, 1, 1], 0.5), [1.75, 1.5, 1.0])
    assert np.allclose(vg.returns_to_go([1, 1, 1], 0.5, bootstrap=8.0), [2.75, 3.5, 5.0])
    g = vg.returns_to_go(np.ones(1000), 0.995, bootstrap=10.0)
    assert math.isclose(g[0], (1 - 0.995 ** 1000) / (1 - 0.995) + 0.995 ** 1000 * 10.0, rel_tol=1e-12)


def ep(err, length=1000, V=None):
    """A synthetic episode: V - G = err(k) per step (err a function of the step), G = 0."""
    k = np.arange(length)
    e = np.array([err(int(x)) for x in k], dtype=np.float64)
    return {"k": k, "V": e if V is None else V, "G": np.zeros(length), "trunc": length == 1000}


def test_bucket_stats_weight_episodes_equally_and_skip_buckets_an_episode_never_reaches():
    a = ep(lambda k: 1.0 if 200 <= k < 500 else 0.0)
    b = ep(lambda k: 0.5, length=300)
    rows = vg.bucket_stats([a, b])
    assert [r["n_episodes"] for r in rows] == [2, 2, 2, 1, 1]
    assert [r["n_states"] for r in rows] == [200, 200, 400, 300, 200]
    r2 = rows[2]
    assert math.isclose(r2["mean_err"], 0.75) and math.isclose(r2["se_err"], np.std([1.0, 0.5], ddof=1) / math.sqrt(2))
    assert rows[3]["se_err"] is None and rows[3]["mean_err"] == 0.0


def noisy(offsets: dict, n=32, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        level = {b: float(rng.normal(0.0, 0.05)) + offsets.get(b, 0.0) for b in range(5)}
        out.append(ep(lambda k, lv=level: lv[vg.bucket_of(k)]))
    return out


def test_a_step_localised_bucket_is_flagged_and_flat_buckets_are_not():
    flags = [r["step_localised"] for r in vg.localisation(noisy({2: 1.0}))]
    assert flags == [False, False, True, False, False]
    assert [r["step_localised"] for r in vg.localisation(noisy({}))] == [False] * 5


def test_the_truncation_bucket_can_be_flagged_against_its_one_neighbour():
    rows = vg.localisation(noisy({4: -0.8}))
    assert [r["step_localised"] for r in rows] == [False, False, False, False, True]
    assert len(rows[4]["comparisons"]) == 1 and rows[4]["comparisons"][0]["neighbour"] == "[500,800)"


def test_a_shift_that_spans_two_inner_buckets_flags_neither_of_them():
    """Each of [200,500) and [500,800) differs from one neighbour only; [800,1000] has one neighbour, which differs,
    so the rule flags it (the criteria's rule, as written)."""
    rows = vg.localisation(noisy({2: 1.0, 3: 1.0}))
    assert [r["step_localised"] for r in rows] == [False, False, False, False, True]


def test_a_comparison_with_fewer_than_2_episodes_is_n_a_and_flags_nothing():
    eps = noisy({}, n=5)
    eps[0] = ep(lambda k: 5.0 if k >= 800 else 0.0)
    for i in range(1, 5):
        eps[i] = ep(lambda k: 0.0, length=600)
    rows = vg.localisation(eps)
    c = rows[4]["comparisons"][0]
    assert c["n"] == 1 and c["se"] is None and c["over"] is None and rows[4]["step_localised"] is False


def test_episode_problems_catch_a_broken_counter_and_an_early_truncation():
    good = ep(lambda k: 0.0)
    bad_k = dict(good, k=np.r_[np.arange(10), np.arange(11, 1001)])
    early = dict(ep(lambda k: 0.0, length=400), trunc=True)
    assert vg.episode_problems([good, ep(lambda k: 0.0, length=400)]) == []
    got = vg.episode_problems([bad_k, early])
    assert len(got) == 2 and "not 0 ... 999" in got[0] and "truncated at 400 steps" in got[1]


def test_main_refuses_without_a_pair_and_writes_nothing(tmp_path):
    out = tmp_path / "x.json"
    assert vg.main(["--json", str(out)]) == 2 and not out.exists()
    assert vg.main(["--json", str(out), "--checkpoint", "a.zip"]) == 2 and not out.exists()
    assert vg.main(["--json", str(out), "--cell", "c", "--checkpoint", "a", "--vecnorm", "b"]) == 2


# ---------------------------------------------------------------- FIX3: the frozen settings and the untested branches


def test_the_frozen_settings():
    """Fixed at FIX2 before any pilot-3 result (criteria, the hidden-clock audit): seed base 6271, 8 complete episodes
    of each of 4 envs, 4 SE, gamma 0.995, the 1,000-step horizon; the CLI and measure() default to them."""
    assert (vg.SEED_BASE, vg.EPISODES_PER_ENV, vg.N_ENVS) == (6271, 8, 4)
    assert (vg.LOCALISED_SE, vg.GAMMA, vg.HORIZON, vg.DV_TOL) == (4.0, 0.995, 1000, 1e-4)
    a = vg.arg_parser().parse_args(["--json", "x.json"])
    assert (a.seed_base, a.episodes_per_env) == (6271, 8)
    sig = inspect.signature(vg.measure).parameters
    assert (sig["seed"].default, sig["n_per_env"].default) == (6271, 8)
    assert inspect.signature(vg.localisation).parameters["factor"].default == 4.0


def _at(level_2: list) -> list:
    """Episodes flat at 0 except bucket [200,500), whose per-episode level is given."""
    return [ep(lambda k, x=x: x if vg.bucket_of(k) == 2 else 0.0) for x in level_2]


def test_the_se_is_sd_over_root_n_and_the_flag_needs_more_than_4_se():
    """Per-episode differences m -/+ 1 (n 4): SD 2/sqrt(3), SE 1/sqrt(3); m = 3.5 SE flags at 3 SE, not at 4."""
    m = 3.5 / math.sqrt(3.0)
    eps = _at([m - 1, m + 1, m - 1, m + 1])
    rows = vg.localisation(eps)
    for c in rows[2]["comparisons"]:
        assert c["n"] == 4 and math.isclose(c["mean_diff"], m) and math.isclose(c["se"], 1 / math.sqrt(3.0))
    assert [r["step_localised"] for r in rows] == [False] * 5
    assert [r["step_localised"] for r in vg.localisation(eps, factor=3.0)] == [False, False, True, False, False]
    m = 4.5 / math.sqrt(3.0)
    assert vg.localisation(_at([m - 1, m + 1, m - 1, m + 1]))[2]["step_localised"] is True


def test_g_bootstraps_after_a_truncation_only():
    """measure()'s V, G and bootstrap step (episode_rows) on synthetic episodes: V = value_fn(X) = X[:, 0]; a truncated
    episode adds gamma^(T-k) V(terminal observation); a terminated one, whose terminal observation SB3 also hands
    over, adds nothing; a truncation without its terminal observation is a problem."""
    def epi(T, trunc, term):
        X = np.arange(T, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
        return {"env": 0, "k": np.arange(T), "X": X, "r_norm": np.ones(T, dtype=np.float32),
                "V_loop": X[:, 0].astype(np.float64), "trunc": trunc,
                "term_X": None if term is None else np.full(3, term, dtype=np.float32)}

    calls = []

    def value_fn(X):
        calls.append(len(X))
        return np.asarray(X, dtype=np.float64)[:, 0]

    eps = [epi(4, True, 8.0), epi(4, False, 8.0), epi(3, True, None)]
    rows, max_dv, problems = vg.episode_rows(eps, value_fn, 0.5)
    assert np.allclose(rows[0]["G"], [1.875 + 0.5 ** 4 * 8, 1.75 + 0.5 ** 3 * 8, 1.5 + 0.25 * 8, 1.0 + 0.5 * 8])
    assert np.allclose(rows[1]["G"], [1.875, 1.75, 1.5, 1.0])
    assert np.allclose(rows[2]["G"], [1.75, 1.5, 1.0])
    assert np.allclose(rows[0]["V"], [0, 1, 2, 3]) and max_dv == 0.0 and calls == [4, 1, 4, 3]
    assert problems == ["env 0: a truncated episode without its terminal observation"]
    assert [r["trunc"] for r in rows] == [True, False, True]
    eps[0]["V_loop"] = eps[0]["V_loop"] + 0.25
    assert vg.episode_rows(eps, value_fn, 0.5)[1] == 0.25


def test_seed_all_seeds_the_envs_torch_and_numpy():
    import torch

    class FakeVecNormalize:
        def __init__(self):
            self.seeds = []

        def seed(self, s):
            self.seeds.append(s)

    torch.manual_seed(1)
    np.random.seed(1)
    fake = FakeVecNormalize()
    got = vg.seed_all(fake, 6271)
    assert fake.seeds == [6271] and torch.initial_seed() == 6271
    assert got == {"vecenv_seed": 6271, "torch_initial_seed": 6271,
                   "numpy_mt19937_key0": int(np.random.RandomState(6271).get_state()[1][0])}
    assert np.random.random_sample() == np.random.RandomState(6271).random_sample()
    assert torch.equal(torch.rand(3), torch.rand(3, generator=torch.Generator().manual_seed(6271)))


def _cell(tmp_path, **over):
    zp, pk = tmp_path / "stage2_8000000_steps.zip", tmp_path / "stage2_vecnormalize_8000000_steps.pkl"
    zp.write_bytes(b"zip")
    pk.write_bytes(b"pkl")
    b = {"head": "a" * 40, "checkpoint": str(zp), "vecnorm": str(pk), "checkpoint_sha256": vg.sha256_file(zp),
         "vecnorm_sha256": vg.sha256_file(pk)}
    b.update(over)
    return b


def test_cell_refuses_a_dirty_tree_a_changed_or_missing_file_and_another_head(tmp_path):
    b = _cell(tmp_path)
    assert vg.cell_problems(b, head="a" * 40, dirty="") == []
    got = vg.cell_problems(b, head="a" * 40, dirty=" M environments/x.py")
    assert got == ["this tree has local edits"]
    got = vg.cell_problems(b, head="b" * 40, dirty="")
    assert len(got) == 1 and "this tree is " + "b" * 40 in got[0]
    Path(b["checkpoint"]).write_bytes(b"zip, changed")
    got = vg.cell_problems(b, head="a" * 40, dirty="")
    assert len(got) == 1 and "stage2_8000000_steps.zip is missing or no longer the judged file" in got[0]
    Path(b["vecnorm"]).unlink()
    got = vg.cell_problems(b, head="b" * 40, dirty="x")
    assert len(got) == 4 and "no longer the judged file" in got[1] and "vecnormalize" in got[2]


def test_main_never_overwrites_an_existing_json(tmp_path):
    target = tmp_path / "binding.json"
    target.write_text('{"judged": true}')
    assert vg.main(["--json", str(target), "--checkpoint", "a.zip", "--vecnorm", "b.pkl"]) == 2
    assert vg.main(["--json", str(target), "--cell", str(tmp_path)]) == 2
    assert target.read_text() == '{"judged": true}'


needs_tree = pytest.mark.skipif(not (TREE / "environments" / "__init__.py").is_file(), reason=f"no tree at {TREE}")


def iso_run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, ISO_WT=str(TREE), MUJOCO_GL="disable", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-I", str(TOOLS / "iso.py"), str(TOOLS / "pilot3_vg.py"), *args], cwd=TREE,
                          env=env, capture_output=True, text=True, timeout=1200)


@needs_tree
@pytest.mark.skipif(not SMOKE_ZIP.is_file(), reason="the notebook smoke's 65,536 pair is not on this machine")
def test_a_short_measurement_of_the_smoke_s_pair(tmp_path):
    out = tmp_path / "vg.json"
    r = iso_run("--checkpoint", str(SMOKE_ZIP), "--vecnorm", str(SMOKE_PKL), "--json", str(out),
                "--episodes-per-env", "1")
    assert r.returncode == 0, (r.stdout, r.stderr)
    rec = json.loads(out.read_text())
    assert rec["episodes"] == 4 and rec["problems"] == [] and rec["units"]["gamma"] == 0.995
    assert rec["seeding"] == {"vecenv_seed": 6271, "torch_initial_seed": 6271,  # FIX3: build() calls seed_all
                              "numpy_mt19937_key0": int(np.random.RandomState(6271).get_state()[1][0])}
    assert rec["settings"]["seed_base"] == 6271 and rec["settings"]["localised_se_factor"] == 4.0
    assert rec["units"]["num_timesteps"] == 65_536 and rec["max_abs_dV_loop"] <= vg.DV_TOL
    assert [b["bucket"] for b in rec["buckets"]] == list(vg.BUCKET_NAMES)
    assert rec["buckets"][0]["n_episodes"] == 4 and rec["report_only"] is True
    assert "STEP-LOCALISED" in r.stdout or " no " in r.stdout


@needs_tree
@pytest.mark.skipif(not SMOKE_ZIP.is_file(), reason="the notebook smoke's 65,536 pair is not on this machine")
def test_a_cell_judged_at_another_head_is_refused_and_nothing_is_written(tmp_path):
    cell = tmp_path / "OUT" / "s42" / "65536"
    cell.mkdir(parents=True)
    (cell / "binding.json").write_text(json.dumps({
        "seed": 42, "steps": 65536, "head": "0" * 40, "checkpoint": str(SMOKE_ZIP), "vecnorm": str(SMOKE_PKL),
        "checkpoint_sha256": vg.sha256_file(SMOKE_ZIP), "vecnorm_sha256": vg.sha256_file(SMOKE_PKL)}))
    out = tmp_path / "vg.json"
    r = iso_run("--cell", str(cell), "--json", str(out))
    assert r.returncode == 2 and "this tree is" in r.stdout and not out.exists(), (r.stdout, r.stderr)
