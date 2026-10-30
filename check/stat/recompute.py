"""Independent recomputation of the bias screen's statistic (lens: THE STATISTIC).

Reads only: each continuation's runs/<name>/01_stance/policy_probe.npz, the parent's F2 probe, reads/<name>/markers.json
(to compare the reader's probe columns), score/score.json (to compare). Writes nothing except stdout.
No import of the screen's tools.
"""
import hashlib
import itertools
import json
import math
import statistics
import sys
from fractions import Fraction

import numpy as np

BS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen"
PARENT = ("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/"
          "F2/stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1")
PARENT_SHA = "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11"
S = 1_048_576
E = 2_490_368
ROLL = 8_192
SEEDS = (2101, 2111, 2121, 2131)
ARMS = ("control", "critic_blind_tail3")
PRIMARY_STEPS = [2_424_832, 2_433_024, 2_441_216, 2_449_408, 2_457_600, 2_465_792, 2_473_984, 2_482_176]
BS_STEPS = [991_232 + ROLL * k for k in range(8)]
assert BS_STEPS[-1] == S
assert PRIMARY_STEPS == [E - ROLL * (8 - k) for k in range(8)]
A14 = 14


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in ("update_timesteps", "rollout_mu_mean", "rollout_rows")}


def mean_at(probe, steps, label):
    ts = probe["update_timesteps"]
    vals = []
    for s in steps:
        idx = np.flatnonzero(ts == s)
        assert idx.size == 1, (label, s, idx)
        v = probe["rollout_mu_mean"][idx[0], A14]
        assert v.dtype == np.float32
        v = float(v)  # float32 widened exactly to float64
        assert math.isfinite(v), (label, s)
        vals.append(v)
    exact = sum(Fraction(v) for v in vals) / len(vals)
    return exact, vals, [int(np.flatnonzero(ts == s)[0]) for s in steps]


out = {}
# ---------------------------------------------------------------- B_S
assert sha(PARENT) == PARENT_SHA, "parent probe sha"
par = load(PARENT)
bs_exact, bs_vals, bs_rows = mean_at(par, BS_STEPS, "parent")
print("parent probe sha256 ok; rows used", bs_rows)
print("B_S exact", bs_exact, "float", float(bs_exact))
out["B_S"] = float(bs_exact)
# also numpy float32 / float64 means, to see if the form matters
print("  B_S np.float64 mean", float(np.mean(np.array(bs_vals, dtype=np.float64))),
      "np.float32 mean", float(np.mean(par["rollout_mu_mean"][bs_rows, A14])))
p_E_exact, _, p_E_rows = mean_at(par, PRIMARY_STEPS, "parent@E")
print("parent primary-form at E", float(p_E_exact), "rise", float(p_E_exact - bs_exact), "rows", p_E_rows)

# ---------------------------------------------------------------- each continuation
prim = {}
for seed in SEEDS:
    for arm in ARMS:
        name = f"{arm}_{seed}"
        p = f"{BS}/runs/{name}/01_stance/policy_probe.npz"
        pr = load(p)
        ts = pr["update_timesteps"]
        # structure: the parent's first 128 rows then 175 window rows at S + 8192 k
        n = ts.size
        win = ts[ts > S]
        ok_win = np.array_equal(win, S + ROLL * np.arange(1, 176))
        pre = ts[ts <= S]
        ok_pre = np.array_equal(pre, par["update_timesteps"][:pre.size]) and \
            np.array_equal(pr["rollout_mu_mean"][:pre.size], par["rollout_mu_mean"][:pre.size])
        dup = n - np.unique(ts).size
        ex, vals, rows = mean_at(pr, PRIMARY_STEPS, name)
        # B_S from the continuation's own copy of the parent rows (should be identical)
        ex_bs_own, _, _ = mean_at(pr, BS_STEPS, name + "@S")
        prim[name] = ex
        print(f"{name:24s} rows {n} window {win.size} ok_win {ok_win} pre {pre.size} pre==parent {ok_pre} dup {dup} "
              f"primary rows {rows[0]}..{rows[-1]} rollout_rows {sorted(set(pr['rollout_rows'][rows].tolist()))}")
        print(f"    primary exact {ex} = {float(ex):+.17g}; rise {float(ex - bs_exact):+.17g}; own-copy B_S equal "
              f"{ex_bs_own == bs_exact}")
        # compare with the reader's markers.json probe columns
        mk = json.load(open(f"{BS}/reads/{name}/markers.json"))
        probe = mk.get("probe", {})
        msteps = probe.get("steps")
        mmu = probe.get("rollout_mu14_mean")
        if msteps is not None and mmu is not None:
            mvals = []
            for s in PRIMARY_STEPS:
                hits = [i for i, t in enumerate(msteps) if t == s]
                mvals.append([mmu[i] for i in hits])
            same = all(len(h) == 1 and h[0] == v for h, v in zip(mvals, vals))
            # whole column equality over the window
            wmask = ts > S
            col = pr["rollout_mu_mean"][:, A14].astype(np.float64)
            mk_map = dict(zip(msteps, mmu))
            all_same = len(msteps) == int(wmask.sum()) and all(mk_map.get(int(t)) == float(c)
                                                                  for t, c in zip(ts[wmask], col[wmask]))
            print(f"    markers.json probe: {len(msteps)} steps; 8 primary values identical {same}; whole window "
                  f"column identical {all_same}")
        else:
            print("    markers.json probe keys:", list(probe.keys())[:20])

# ---------------------------------------------------------------- summary statistics
ctrl = [prim[f"control_{s}"] for s in SEEDS]
blind = [prim[f"critic_blind_tail3_{s}"] for s in SEEDS]
cmed = (sorted(ctrl)[1] + sorted(ctrl)[2]) / 2
bmed = (sorted(blind)[1] + sorted(blind)[2]) / 2
rise = cmed - bs_exact
informative = rise >= Fraction(1, 10)
print("control primaries", [float(x) for x in ctrl])
print("blind primaries  ", [float(x) for x in blind])
print("control mean", float(sum(ctrl) / 4), "median", float(cmed), "SD", statistics.stdev([float(x) for x in ctrl]))
print("blind mean  ", float(sum(blind) / 4), "median", float(bmed), "SD", statistics.stdev([float(x) for x in blind]))
print("diff of means", float(sum(ctrl) / 4 - sum(blind) / 4))
print("control median rise", float(rise), "exact", rise, "informative (>= 1/10)", informative)
# sensitivity: informativeness on the mean instead of the median (not the rule)
print("  (not the rule) control MEAN rise", float(sum(ctrl) / 4 - bs_exact))

# exact permutation p: all C(8,4) labelings, statistic mean(control) - mean(blind), one-sided >= observed
allv = ctrl + blind
obs = sum(ctrl) / 4 - sum(blind) / 4
cnt = 0
tot = 0
stats_ = []
for comb in itertools.combinations(range(8), 4):
    c = [allv[i] for i in comb]
    b = [allv[i] for i in range(8) if i not in comb]
    st = sum(c) / 4 - sum(b) / 4
    stats_.append(st)
    tot += 1
    if st >= obs:
        cnt += 1
print(f"permutation p = {cnt}/{tot} = {cnt / tot}")
ties = sum(1 for st in stats_ if st == obs)
print("  ties with observed (incl. itself):", ties)
# sum-statistic equivalence
cnt2 = sum(1 for comb in itertools.combinations(range(8), 4) if sum(allv[i] for i in comb) >= sum(ctrl))
print("  control-sum statistic p", cnt2, "/70")
# sign-flip paired p
diffs = [c - b for c, b in zip(ctrl, blind)]
obs_d = sum(diffs)
cnt3 = sum(1 for sg in itertools.product((1, -1), repeat=4) if sum(s * d for s, d in zip(sg, diffs)) >= obs_d)
print("paired diffs", [float(d) for d in diffs], "sign-flip p", cnt3, "/16")
# the ranks
order = sorted(range(8), key=lambda i: allv[i])
print("ranks (lowest first):", [("C" if i < 4 else "B") + str(SEEDS[i % 4]) for i in order])

# ---------------------------------------------------------------- compare with score.json
sj = json.load(open(f"{BS}/score/score.json"))
cmp = []


def c(label, mine, theirs):
    eq = mine == theirs
    cmp.append((label, eq))
    print(f"  {label:40s} mine {mine!r:>28} score.json {theirs!r:>28} {'EQUAL' if eq else 'DIFFER'}")


print("COMPARE with score.json")
c("baseline_S", float(bs_exact), sj["baseline_S"])
c("control_median", float(cmed), sj["control_median"])
c("control_rise", float(rise), sj["control_rise"])
c("informative", bool(informative), sj["informative"])
c("arm_means.control", float(sum(ctrl) / 4), sj["arm_means"]["control"])
c("arm_means.blind", float(sum(blind) / 4), sj["arm_means"]["critic_blind_tail3"])
c("arm_medians.blind", float(bmed), sj["arm_medians"]["critic_blind_tail3"])
c("sd.control", statistics.stdev([float(x) for x in ctrl]), sj["arm_sd_report_only"]["control"])
c("sd.blind", statistics.stdev([float(x) for x in blind]), sj["arm_sd_report_only"]["critic_blind_tail3"])
c("p_filled", f"{cnt}/{tot}", f'{sj["p_filled"]["count"]}/{sj["p_filled"]["total"]}')
c("p_as_counted", f"{cnt}/{tot}", f'{sj["p_as_counted"]["count"]}/{sj["p_as_counted"]["total"]}')
c("sign_flip", f"{cnt3}/16", f'{sj["paired_sign_flip"]["count"]}/{sj["paired_sign_flip"]["total"]}')
c("primary_steps", PRIMARY_STEPS, sj["primary_steps"])
for cl in sj["classes"]:
    nm = cl["name"]
    c(f"primary {nm}", float(prim[nm]), cl["primary"])
    c(f"primary_exact {nm}", str(prim[nm]), cl["primary_exact"])
    c(f"cls {nm}", "COUNTED", cl["cls"])
for row in sj["paired"]:
    s = row["seed"]
    c(f"paired diff {s}", float(prim[f"control_{s}"] - prim[f"critic_blind_tail3_{s}"]), row["diff"])
c("line", 1, sj["line"])
print("ALL EQUAL" if all(e for _, e in cmp) else "SOME DIFFER: " + str([l for l, e in cmp if not e]))
