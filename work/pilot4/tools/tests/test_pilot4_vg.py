"""Tests of pilot3_vg.py on the pilot-4 tree, the report-only V - G readout (pytest; walking pilot 4 scratch).

Walking pilot 4 re-pins pilot 3's pilot3_vg.py as a byte copy (sha256 a58d5f4f...; the TOOL PLAN: "code unchanged,
re-pinned; G is the pilot-4 env's reward (price included)"). This file is pilot 3's tests/test_pilot3_vg.py (sha256
ead00c7b...) with only the paths adapted: the tool imported is the pilot-4 tools folder's byte copy (TOOLS, the
parent of this tests folder), and the integration tests' tree is the pilot-4 tree (PILOT4_TREE, default ../tree:
the clean clone at the pilot-4 head; pilot 3's variable name PILOT3_TREE would name pilot 3's tree). Pilot 3's
notebook smoke's 65,536 pair, still on this machine, stays the short measurement's pair (a check of the tool on any
locomotion pair; no pilot-4 checkpoint exists before launch). Pilot 3's docstring, kept: pure tests check the
buckets, G, the per-bucket statistics, the step-localised rule and the self-checks on synthetic episodes; two
integration tests run the tool through iso.py on the clean tree: a short measurement (1 episode per env) of the
notebook smoke's 65,536 pair (skipped when the smoke's files are absent), and a --cell refusal that writes nothing.
FIX3 (the third review's surviving mutants): the frozen settings, the SE and 4-SE rule at its boundary, G's bootstrap
after a truncation only (episode_rows, measure()'s own step), the seeding (seed_all, and the short measurement's
record), the --cell refusals (cell_problems) and no overwrite of --json.
New for pilot 4, at the end: the byte copy is pilot 3's pinned file; the short measurement reads the pilot-4 tree;
and G is the pilot-4 env's reward, price included (through iso.py on the pilot-4 tree: the tool's own build() and
collect() on the s42 parent pair, pilot 3's verified 8M run copy read only, with a price-off twin of the same env
stepped in lock step on the same actions).
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

TREE = Path(os.environ.get("PILOT4_TREE", str(TOOLS / "tree"))).resolve()
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


# ---------------------------------------------------------------- pilot 4: the byte copy, the tree, G with the price

SP = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad")
PILOT3_VG = SP / "work/pilot3/tools/pilot3_vg.py"
PINNED_SHA256 = "a58d5f4f24e0c3e70467c8aeea8df20188fc246d48b3127bef594061a6b44545"
PARENT = SP / "work/pilot3/runs/final/pilot3_walk_s42/03_locomotion/models"
PARENT_ZIP, PARENT_PKL = PARENT / "stage2_8000000_steps.zip", PARENT / "stage2_vecnormalize_8000000_steps.pkl"


def test_the_tool_is_a_byte_copy_of_pilot_3_s_pinned_file():
    import hashlib

    here = (TOOLS / "pilot3_vg.py").read_bytes()
    assert hashlib.sha256(here).hexdigest() == PINNED_SHA256
    if PILOT3_VG.is_file():
        assert PILOT3_VG.read_bytes() == here
    assert Path(vg.__file__).resolve() == (TOOLS / "pilot3_vg.py").resolve()
    assert TOOLS.name == "tools" and TOOLS.parent.name == "pilot4"


@needs_tree
@pytest.mark.skipif(not SMOKE_ZIP.is_file(), reason="the notebook smoke's 65,536 pair is not on this machine")
def test_the_short_measurement_reads_the_pilot_4_tree(tmp_path):
    out = tmp_path / "vg.json"
    r = iso_run("--checkpoint", str(SMOKE_ZIP), "--vecnorm", str(SMOKE_PKL), "--json", str(out),
                "--episodes-per-env", "1")
    assert r.returncode == 0, (r.stdout, r.stderr)
    rec = json.loads(out.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TREE, text=True, env=dict(
        os.environ, GIT_OPTIONAL_LOCKS="0")).strip()
    assert Path(rec["tree"]).resolve() == TREE and rec["head"] == head and rec["tree_clean"] is True
    assert rec["schema"] == "walking-pilot3-vg/v1" and rec["tool_sha256"] == "sha256:" + PINNED_SHA256


PRICE_SCRIPT = r'''
"""G includes the price: the tool's own build() and collect() on the pilot-4 tree, a price-off twin in lock step."""
import copy, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import pilot3_vg as vg
import environments
from environments.shared.species_names import require_model_id
from environments.shared.species_registry import get_species_config
from environments.shared.train_base import create_vec_env

zip_path, pkl_path = sys.argv[2], sys.argv[3]
cfg, gamma, vn, model, seeding = vg.build(zip_path, pkl_path, vg.SEED_BASE)
ek = dict(cfg["env_kwargs"])
off_kwargs = dict(copy.deepcopy(ek), walk_heading_price_weight=0.0)
mid = require_model_id("tyrannosaurus_rex", "anatomical")
twin = create_vec_env(get_species_config(mid), {2: {"env_kwargs": off_kwargs}}, 2, vg.N_ENVS, seed=vg.SEED_BASE,
                      use_subproc=False, algorithm="ppo", gamma=gamma).venv
twin.seed(vg.SEED_BASE)
inner = vn.venv
real_reset, real_step, real_norm = inner.reset, inner.step, vn.normalize_reward
log = {"reset_equal": [], "obs_equal": True, "dones_equal": True, "price_keys_off": 0, "steps": 0,
       "exact": 0, "max_rel_err": 0.0, "rew_is_total": True, "nonzero_price": 0, "price_sum": 0.0}
per_env = [{"on": [], "off": [], "price": []} for _ in range(vg.N_ENVS)]
first = [None] * vg.N_ENVS  # each env's first complete episode's (on, off, price) rewards


def reset():
    a = real_reset()
    b = twin.reset()
    log["reset_equal"].append(bool(np.array_equal(a, b)))
    return a


def step(actions):
    o, r, d, i = real_step(actions)
    o2, r2, d2, i2 = twin.step(actions)
    log["obs_equal"] &= bool(np.array_equal(o, o2))
    log["dones_equal"] &= bool(np.array_equal(d, d2))
    for k in range(len(r)):
        price = float(i[k]["reward_walk_heading_price"])
        on, off = float(i[k]["reward_total"]), float(i2[k]["reward_total"])
        log["price_keys_off"] += int("reward_walk_heading_price" in i2[k])
        log["steps"] += 1
        log["exact"] += int(on == off + price)
        log["max_rel_err"] = max(log["max_rel_err"], abs(on - (off + price)) / max(1.0, abs(on)))
        log["rew_is_total"] &= bool(np.float32(r[k]) == np.float32(on))
        log["nonzero_price"] += int(price != 0.0)
        log["price_sum"] += price
        e = per_env[k]
        e["on"].append(float(r[k]))
        e["off"].append(float(r2[k]))
        e["price"].append(price)
        if d[k]:
            if first[k] is None:
                first[k] = {key: list(v) for key, v in e.items()}
            per_env[k] = {"on": [], "off": [], "price": []}
    return o, r, d, i


fed = []


def norm(rew):
    fed.append(np.array(rew, dtype=np.float32, copy=True))
    return real_norm(rew)


inner.reset, inner.step, vn.normalize_reward = reset, step, norm
eps = vg.collect(vn, model, 1)
rows, max_dv, problems = vg.episode_rows(eps, lambda X: vg.critic_values(model.policy, vn, X), gamma)
g_diff, fed_equal = [], True
for ep, row in zip(eps, rows):
    f = first[ep["env"]]
    on = np.asarray(f["on"], dtype=np.float32)
    off = np.asarray(f["off"], dtype=np.float32)
    fed_equal &= bool(np.array_equal(ep["r_norm"], np.asarray(real_norm(on), dtype=np.float32)))
    boot = float(vg.critic_values(model.policy, vn, ep["term_X"][None, :])[0]) if ep["trunc"] else None
    g_on = vg.returns_to_go(np.asarray(real_norm(on), dtype=np.float32), gamma, boot)
    g_off = vg.returns_to_go(np.asarray(real_norm(off), dtype=np.float32), gamma, boot)
    g_diff.append({"env": ep["env"], "length": len(on), "trunc": ep["trunc"],
                   "tool_G_is_price_on_G": bool(np.array_equal(row["G"], g_on)),
                   "tool_G_minus_price_off_G_at_0": float(row["G"][0] - g_off[0]),
                   "max_abs_G_on_minus_off": float(np.max(np.abs(g_on - g_off))),
                   "episode_price": float(np.sum(f["price"]))})
vn.close()
twin.close()
print(json.dumps({"env_kwargs": {k: ek.get(k) for k in ("command_mode", "prey_lateral_range",
                                                         "walk_heading_price_weight", "walk_heading_price_scale_rad")},
                  "environments": environments.__file__, "seeding": seeding, "fed_calls": len(fed),
                  "fed_equal_first_episodes": fed_equal, "log": log, "episodes": g_diff, "problems": problems,
                  "max_dv": max_dv}))
'''


@needs_tree
@pytest.mark.skipif(not (PARENT_ZIP.is_file() and PARENT_PKL.is_file()), reason="pilot 3's s42 8M run copy is absent")
def test_g_is_the_pilot_4_env_s_reward_price_included(tmp_path):
    """The TOOL PLAN: "G is the pilot-4 env's reward (price included)". The tool's own build() on the pilot-4 tree
    builds the env with the heading price on (0.2 / 0.5, the heading command, the prey on the line); its collect()
    hands normalize_reward the env's step rewards, which are info reward_total; a twin with the price weight 0,
    stepped on the same actions, stays state-identical (same observations and dones) and its reward_total plus the
    price is the price-on reward_total on every step; so the tool's G (from those rewards) is below the price-off G.
    The pair is the s42 parent (pilot 3's 8M, unprepared: a check of the reward path, not a pilot-4 reading)."""
    script = tmp_path / "g_price.py"
    script.write_text(PRICE_SCRIPT)
    env = dict(os.environ, ISO_WT=str(TREE), MUJOCO_GL="disable", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               GIT_OPTIONAL_LOCKS="0")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-I", str(TOOLS / "iso.py"), str(script), str(TOOLS), str(PARENT_ZIP),
                        str(PARENT_PKL)], cwd=TREE, env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    got = json.loads(r.stdout.strip().splitlines()[-1])
    print(json.dumps(got, indent=1))  # the measured record, in the test log (-s)
    assert got["env_kwargs"] == {"command_mode": "heading", "prey_lateral_range": [0.0, 0.0],
                                 "walk_heading_price_weight": 0.2, "walk_heading_price_scale_rad": 0.5}
    assert Path(got["environments"]).resolve().is_relative_to(TREE)
    log = got["log"]
    assert log["reset_equal"] == [True] and log["obs_equal"] and log["dones_equal"]
    assert log["price_keys_off"] == 0 and log["rew_is_total"] and log["steps"] >= 4 * 100
    assert log["max_rel_err"] <= 1e-12 and log["nonzero_price"] > log["steps"] // 2 and log["price_sum"] < 0.0
    assert got["fed_calls"] * 4 == log["steps"] and got["fed_equal_first_episodes"] is True
    assert len(got["episodes"]) == 4 and got["problems"] == [] and got["max_dv"] <= vg.DV_TOL
    for e in got["episodes"]:
        assert e["tool_G_is_price_on_G"] is True and e["episode_price"] < 0.0
        assert e["tool_G_minus_price_off_G_at_0"] < 0.0 and e["max_abs_G_on_minus_off"] > 0.0
