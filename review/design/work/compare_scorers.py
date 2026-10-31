"""Design review: compare score_bs.score() with work/my_score.py on randomized synthetic PS-shaped roots.
Roots are built under <D>/work/roots/ with the scorer tests' own Screen helper (input generator only), then perturbed
here (8 distinct primary-row values, column 13 values, ties, missing / LOST / STOPPED / UNREADABLE / no final row /
no evaluation at E / NaN tail_2), scored by both, compared, and removed.  Usage: compare_scorers.py N SEED [MUTANT]
MUTANT (optional) names a copy of score_bs.py under <D>/work/mutants/ to load instead of the real one."""
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
PS = Path("/home/user/work/phase_screen")
D = PS / "review" / "design"
sys.path.insert(0, str(D / "work"))
import my_score as MY  # noqa: E402

spec = importlib.util.spec_from_file_location("test_score_bs_rd", str(PS / "tools/tests/test_score_bs.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
SB = T.SB
if len(sys.argv) > 3:
    mspec = importlib.util.spec_from_file_location("score_bs_mut", sys.argv[3])
    SB = importlib.util.module_from_spec(mspec)
    sys.modules["score_bs_mut"] = SB
    mspec.loader.exec_module(SB)
SB.code_seal_items = lambda d: []          # the code copy's seal: not part of the reading (as the tests' fixture)

N, SEED = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(SEED)
B = T.B_S
f32 = T.f32
ROOTS = D / "work" / "roots"
ROOTS.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def perturb(root, name, vals14, vals13):
    """Rewrite the run's probe file's primary rows (col 14 and 13) and the markers / check.json that bind it."""
    mk = root / "reads" / name / "markers.json"
    m = json.loads(mk.read_text())
    pf = Path(m["probe"]["file"])
    with np.load(pf, allow_pickle=False) as z:
        ut = np.asarray(z["update_timesteps"]).copy()
        mu = np.asarray(z["rollout_mu_mean"]).copy()
    for s, a, b in zip(SB.PRIMARY_STEPS, vals14, vals13):
        i = int(np.flatnonzero(ut == s)[0])
        mu[i, 14] = np.float32(a)
        mu[i, 13] = np.float32(b) if b is not None else np.nan
    with open(pf, "wb") as fh:
        np.savez(fh, update_timesteps=ut, rollout_mu_mean=mu)
    m["probe"]["rollout_mu14_mean"] = mu[:, 14].astype(np.float64).tolist()
    m["probe"]["sha256"] = sha(pf)
    mk.write_text(json.dumps(m, indent=1) + "\n")
    ck = root / "runs" / name / "check.json"
    c = json.loads(ck.read_text())
    c["sources"]["policy_probe.npz"] = [str(pf), m["probe"]["sha256"]]
    ck.write_text(json.dumps(c, indent=1) + "\n")


def scenario(i):
    root = ROOTS / f"r{SEED}_{i}"
    if root.exists():
        shutil.rmtree(root)
    s = T.Screen(root)
    mode = rng.choice(["climb", "edge", "ties", "null", "reduced", "reduced"])
    sep = rng.random() < 0.5
    grid = [k / 64 for k in range(-8, 64)]
    perturbs = {}
    for seed in SB.SEEDS:
        for arm in (T.CTRL, T.PHASE):
            if mode == "climb":
                v = B + (rng.uniform(0.15, 0.7) if arm == T.CTRL else rng.uniform(-0.1, 0.6))
            elif mode == "edge":
                v = B + rng.uniform(0.17, 0.23)
            elif mode == "ties":
                v = B + rng.choice([0.125, 0.25, 0.375])
            elif mode == "reduced":
                v = B + (rng.uniform(0.3, 0.6) if arm == T.CTRL else rng.uniform(-0.05, 0.3))
            else:
                v = B + rng.uniform(0.0, 0.6)
            kw = {"g": rng.choice(grid) - 0.5, "cmd": rng.choice(grid), "b": rng.choice(grid) + 0.5,
                  "mech": rng.choice(grid) / 10, "tail2": rng.choice(grid) / 4}
            if sep:   # separate the arms in each secondary's predicted direction (with overlap sometimes)
                up = 1 if arm == T.CTRL else -1
                kw["cmd"] = 0.5 + up * rng.choice([0.0625, 0.125, 0.25]) + rng.choice([0, 0, 0.1875])
                kw["mech"] = 0.03 + up * rng.choice([0.0078125, 0.015625, 0.03125])
                kw["b"] = 1.0 - up * rng.choice([0.125, 0.25, 0.5])
                kw["tail2"] = 0.03125 - up * rng.choice([0.0078125, 0.015625, 0.03125])
            kind = rng.random()
            if kind < 0.04:
                kw["stop"] = "collapse"
                kw["T"] = rng.choice([2_621_440, 3_145_728])
            elif kind < 0.08:
                kw["nan_at"] = rng.choice(SB.PRIMARY_STEPS)
            elif kind < 0.10:
                kw["final"] = False
            elif kind < 0.12:
                kw["drop_E"] = True
            elif kind < 0.14:
                kw["tail2_nan"] = True
            name = s.run(arm, seed, f32(v), **kw)
            if kind < 0.02:
                s.lost(name)
            elif kind >= 0.14 and rng.random() < 0.7:
                spread = rng.choice([0.0, 0.01, 0.05])
                v14 = [f32(v + rng.uniform(-spread, spread)) for _ in range(8)]
                v13 = [f32(rng.choice(grid) / 4 + rng.uniform(-spread, spread)) for _ in range(8)]
                perturbs[name] = (v14, v13)
    for name, (a, b) in perturbs.items():
        perturb(root, name, a, b)
    return root


def compare(root):
    res, _lines, code = SB.score(root)
    mine = MY.reading(MY.read_root(root))
    diffs = []
    if res.get("status") == "incomplete" or mine["status"] == "incomplete":
        if not (res.get("status") == "incomplete" and mine["status"] == "incomplete"):
            diffs.append(("status", res.get("status"), mine["status"]))
        return diffs, res
    if res["line"] != mine["line"]:
        diffs.append(("line", res["line"], mine["line"]))
    if (res["p_filled"]["count"], res["p_filled"]["total"]) != mine["filled"]:
        diffs.append(("p_filled", res["p_filled"], mine["filled"]))
    pa = res.get("p_as_counted")
    pa = None if pa is None else (pa["count"], pa["total"])
    if pa != mine["as_counted"]:
        diffs.append(("p_as_counted", pa, mine["as_counted"]))
    if bool(res["informative"]) != mine["informative"]:
        diffs.append(("informative", res["informative"], mine["informative"]))
    adds = []
    for a in res["additions"]:
        adds.append("as_counted" if a == SB.ADDITION_AS_COUNTED else
                    "not_informative_but_p" if a == SB.ADDITION_NOT_INFORMATIVE else a)
    if adds != mine["additions"]:
        diffs.append(("additions", adds, mine["additions"]))
    for sres in res["secondaries"]:
        my = mine["secs"][sres["key"]]
        if sres["p_filled"]["count"] != my["count"]:
            diffs.append((sres["key"] + " p", sres["p_filled"]["count"], my["count"]))
        if (sres["verdict"] == SB.SEC_SUPPORTED) != my["supported"]:
            diffs.append((sres["key"] + " verdict", sres["verdict"], my["supported"]))
        notes = ["bonferroni" if n == SB.SEC_BONFERRONI else "guard" if n == SB.SEC_GUARD else
                 "line1" if n == SB.SEC_UNDER_LINE1 else n for n in sres["notes"]]
        if notes != my["notes"]:
            diffs.append((sres["key"] + " notes", notes, my["notes"]))
    return diffs, res


lines_seen, n_diff, errors = {}, 0, 0
secsup = {}
for i in range(N):
    root = scenario(i)
    try:
        diffs, res = compare(root)
    except Exception as exc:  # noqa: BLE001
        errors += 1
        print(f"scenario {i}: EXCEPTION {exc.__class__.__name__}: {str(exc)[:300]}")
        shutil.rmtree(root)
        continue
    key = res.get("line", res.get("status"))
    lines_seen[key] = lines_seen.get(key, 0) + 1
    for sres in res.get("secondaries", []):
        secsup[(sres["key"], sres["verdict"])] = secsup.get((sres["key"], sres["verdict"]), 0) + 1
    if diffs:
        n_diff += 1
        print(f"scenario {i}: DIFF {diffs}")
    shutil.rmtree(root)
print(f"SEED {SEED}: {N} scenarios, {n_diff} with differences, {errors} exceptions; lines {lines_seen}")
print("secondary verdicts:", dict(sorted(secsup.items())))
