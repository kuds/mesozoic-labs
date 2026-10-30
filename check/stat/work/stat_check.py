"""BS2 statistic lens: an independent recomputation of the primary, B_S, the gate, the exact tests and the three
secondaries from the raw records, compared with score.json.  Own code: imports nothing from the screen's tools.

Reads only (numpy allow_pickle False); writes nothing (stdout is captured by the caller into check/stat/).
"""
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
from fractions import Fraction as Fr

import numpy as np
from pathlib import Path

_BS2 = Path(__file__).resolve().parents[3]                 # work -> stat -> check -> bias_screen2
S2 = str(_BS2.parents[1])                                   # bias_screen2 -> r10round -> the session scratchpad
S1 = str(_BS2.parents[3] / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad")
BS2 = str(_BS2)
F2_PROBE = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1"
F2_PROBE_SHA = "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11"
S = 2_097_152
E = 3_538_944
ROLL = 8192
PRIMARY_STEPS = [3_473_408, 3_481_600, 3_489_792, 3_497_984, 3_506_176, 3_514_368, 3_522_560, 3_530_752]
BS_STEPS = [2_039_808 + ROLL * j for j in range(8)]
B_S_CRITERIA = Fr(135930855, 1073741824)
ARMS = ("control", "critic_blind_tail3")
SEEDS = (2501, 2511, 2521, 2531)
EVAL_STEPS = [S + 131_072 * k for k in range(1, 12)]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fstr(x):
    return f"{x.numerator}/{x.denominator}" if isinstance(x, Fr) and x.denominator != 1 else str(x)


def load_probe(path):
    with open(path, "rb") as fh:
        z = np.load(fh, allow_pickle=False)
        u = np.array(z["update_timesteps"])
        m = np.array(z["rollout_mu_mean"])
    return u, m


def rows_at(u, m, steps, col=14):
    """Exact rationals of m[row, col] (float32 widened to float64) at the given steps; each step exactly once."""
    out = []
    for s in steps:
        idx = np.nonzero(u == s)[0]
        if len(idx) != 1:
            raise SystemExit(f"step {s}: {len(idx)} rows")
        v = float(m[idx[0], col])          # float32 -> float64 is exact
        if not math.isfinite(v):
            raise SystemExit(f"step {s}: not finite")
        out.append(Fr(v))
    return out


def perm_p(values_c, values_b, higher):
    """Exact one-sided unpaired permutation p over the C(8,4) = 70 relabelings, statistic = difference of arm means in
    the stated direction (higher arm minus the other), ties against; values are (inf_part, Fraction) pairs."""
    vals = list(values_c) + list(values_b)
    n = len(vals)
    nc = len(values_c)

    def stat(ic):
        c = [vals[i] for i in ic]
        b = [vals[i] for i in range(n) if i not in ic]
        sc = (sum(v[0] for v in c), sum((v[1] for v in c), Fr(0)))
        sb = (sum(v[0] for v in b), sum((v[1] for v in b), Fr(0)))
        # difference of means: with 4 + 4 the means' infinite parts carry the same sign as the sums'
        mc = (sc[0], sc[1] / len(c))
        mb = (sb[0], sb[1] / len(b))
        d = (mc[0] - mb[0], mc[1] - mb[1]) if higher == "control" else (mb[0] - mc[0], mb[1] - mc[1])
        return d
    obs = stat(tuple(range(nc)))
    count = 0
    total = 0
    for ic in itertools.combinations(range(n), nc):
        total += 1
        if stat(ic) >= obs:              # tuple comparison: the infinite part first
            count += 1
    return Fr(count, total), count, total


def sign_flip_p(diffs):
    obs = sum(diffs, Fr(0))
    count = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        if sum((s * d for s, d in zip(signs, diffs)), Fr(0)) >= obs:
            count += 1
    return Fr(count, 2 ** len(diffs)), count


def median_rule(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2


def sample_sd(xs):
    xs = [float(x) for x in xs]
    return statistics.stdev(xs)


def main():
    out = {}
    names = [(a, s, f"{a}_{s}") for s in SEEDS for a in ARMS]
    screen = json.load(open(BS2 + "/SCREEN.json"))
    q = [tuple(x) for x in screen["queue"]]
    print("SCREEN queue:", q)
    assert q == [(a, s) for (a, s, _) in names], "queue differs"

    # ---------------------------------------------------------------- B_S from the parent's F2 probe
    sha = sha256(F2_PROBE)
    print("F2 probe sha256", sha, "pin match", sha == F2_PROBE_SHA)
    pu, pm = load_probe(F2_PROBE)
    bs_rows = rows_at(pu, pm, BS_STEPS)
    idx = [int(np.nonzero(pu == s)[0][0]) for s in BS_STEPS]
    B_S = sum(bs_rows, Fr(0)) / 8
    print("B_S rows (0-based)", idx, "steps", BS_STEPS[0], "...", BS_STEPS[-1])
    print("B_S exact", fstr(B_S), "float", repr(float(B_S)), "== criteria", B_S == B_S_CRITERIA)
    parent_primary = sum(rows_at(pu, pm, PRIMARY_STEPS), Fr(0)) / 8
    print("parent primary at E (same 8 rows)", fstr(parent_primary), repr(float(parent_primary)),
          "rise", repr(float(parent_primary - B_S)))

    # ---------------------------------------------------------------- primaries from each continuation's probe
    prim = {}
    probe_sha = {}
    for a, s, nm in names:
        p = f"{BS2}/runs/{nm}/01_stance/policy_probe.npz"
        probe_sha[nm] = sha256(p)
        u, m = load_probe(p)
        win = u[u > S]
        exp = np.array([S + ROLL * k for k in range(1, 176)])
        ok_win = len(win) == 175 and np.array_equal(np.sort(win), exp) and len(set(win.tolist())) == 175
        pre = u[u <= S]
        pre_ok = np.array_equal(pre, np.array([ROLL * k for k in range(1, 257)]))
        # the pre-S rows equal the parent's (column 14)?
        same_parent = bool(np.array_equal(m[u <= S][:, 14], pm[:256][:, 14]))
        vals = rows_at(u, m, PRIMARY_STEPS)
        prim_idx = [int(np.nonzero(u == st)[0][0]) for st in PRIMARY_STEPS]
        win_rank = [int(np.nonzero(exp == st)[0][0]) + 1 for st in PRIMARY_STEPS]
        prim[nm] = sum(vals, Fr(0)) / 8
        # markers
        mk = json.load(open(f"{BS2}/reads/{nm}/markers.json"))
        msteps = mk["probe"]["steps"]
        mvals = mk["probe"]["rollout_mu14_mean"]
        npz_win = {int(st): float(m[i, 14]) for i, st in enumerate(u.tolist()) if st > S}
        mk_equal = all(npz_win[int(st)] == v for st, v in zip(msteps, mvals)) and len(msteps) == 175
        print(f"{nm:24s} probe rows {len(u)}, window rows 175 at S+8192k exact: {ok_win}; pre-S rows 256: {pre_ok}; "
              f"pre-S col14 == parent F2: {same_parent}; primary rows idx {prim_idx[0]}..{prim_idx[-1]} "
              f"(window rank {win_rank[0]}-{win_rank[-1]}); markers 175 values == npz: {mk_equal}; "
              f"markers probe sha == file: {mk['probe']['sha256'] == probe_sha[nm]}")
        print(f"    primary exact {fstr(prim[nm])}  float {float(prim[nm]):+.17g}  rise {float(prim[nm] - B_S):+.17g}")
    out["prim"] = prim

    ctrl = [prim[f"control_{s}"] for s in SEEDS]
    blind = [prim[f"critic_blind_tail3_{s}"] for s in SEEDS]
    med_c = median_rule(ctrl)
    rise = med_c - B_S
    print("control sorted", [f"{float(x):+.5f}" for x in sorted(ctrl)])
    print("control median (midpoint rule)", fstr(med_c), repr(float(med_c)), "rise", fstr(rise), repr(float(rise)))
    print("gate: rise >= 1/5 exact:", rise >= Fr(1, 5), "; rise >= Fraction(0.2 double):", rise >= Fr(0.2))
    mean_c = sum(ctrl, Fr(0)) / 4
    mean_b = sum(blind, Fr(0)) / 4
    print("arm means", repr(float(mean_c)), repr(float(mean_b)), "diff", repr(float(mean_c - mean_b)))
    print("arm medians", repr(float(med_c)), repr(float(median_rule(blind))))
    print("arm SDs (n-1)", repr(sample_sd(ctrl)), repr(sample_sd(blind)))

    pc = [(0, x) for x in ctrl]
    pb = [(0, x) for x in blind]
    p, cnt, tot = perm_p(pc, pb, "control")
    print(f"permutation p (filled == as counted; nothing missing): {cnt}/{tot} = {float(p):.6f}")
    # the statistic's rank among all 70 (ties?)
    allv = ctrl + blind
    stats = []
    for ic in itertools.combinations(range(8), 4):
        c = [allv[i] for i in ic]
        b = [allv[i] for i in range(8) if i not in ic]
        stats.append(sum(c, Fr(0)) / 4 - sum(b, Fr(0)) / 4)
    obs = mean_c - mean_b
    print("  relabelings with statistic >= observed:", sum(1 for x in stats if x >= obs),
          "; distinct values:", len(set(stats)), "; second largest statistic", repr(float(sorted(stats)[-2])))
    print("  complete separation (min control > max blind):", min(ctrl) > max(blind),
          f"min control {float(min(ctrl)):+.5f} max blind {float(max(blind)):+.5f}")
    # fill bounds sanity: one missing control
    for miss in ("control0", "blind0", "both"):
        fc = [(0, x) for x in ctrl]
        fb = [(0, x) for x in blind]
        if miss in ("control0", "both"):
            fc[0] = (-1, Fr(0))
        if miss in ("blind0", "both"):
            fb[0] = (1, Fr(0))
        pf, cf, tf = perm_p(fc, fb, "control")
        print(f"  fill check (hypothetical {miss} MISSING on these data): {cf}/{tf}")
    diffs = [prim[f"control_{s}"] - prim[f"critic_blind_tail3_{s}"] for s in SEEDS]
    psf, csf = sign_flip_p(diffs)
    print("paired diffs", [f"{float(d):+.5f}" for d in diffs], f"sign-flip p {csf}/16 = {float(psf):.4f}")

    # ---------------------------------------------------------------- secondaries
    sec = {}
    loops = {}
    for a, s, nm in names:
        mk = json.load(open(f"{BS2}/reads/{nm}/markers.json"))
        rowsE = [r for r in mk["loop"] if r.get("steps") == E]
        assert len(rowsE) == 1, (nm, len(rowsE))
        r = rowsE[0]
        loops[nm] = mk["loop"]
        G = r["OWN"]["P"]["G"]
        c0 = r["OWN"]["P"]["c0"]
        bstar = r["OWN"]["P"]["b_star"]
        cmd = mk["ledger"]["direct"]["per_evaluation"]["3538944"]["cmd_l"]
        n_e = mk["ledger"]["direct"]["per_evaluation"]["3538944"]["n"]
        sec.setdefault("SEC-G", {})[nm] = G
        sec.setdefault("SEC-CMD", {})[nm] = cmd
        sec.setdefault("SEC-MARGIN", {})[nm] = bstar
        zp = r["zip"]
        pk = r["pkl"]
        zsha = sha256(zp)
        psha = sha256(pk)
        form = r.get("critic_form")
        eff = r["critic_slope_obs6"]["ONPOLICY"].get("effective", {}).get("dV_dq3_per_deg")
        print(f"{nm:24s} E row: zip {zp.split('/')[-1]} sha ok {zsha == r['zip_sha256']}, pkl {pk.split('/')[-1]} sha ok "
              f"{psha == r['pkl_sha256']}, num_timesteps {r['num_timesteps']}, critic_form {form}, eff on-policy slope "
              f"{eff}; b*-(1-G-c0) = {bstar - (1 - G - c0):.3g}; CMD_L n {n_e}")
        # recompute CMD_L at E from the raw ledger
        with open(f"{BS2}/runs/{nm}/01_stance/eval_episodes.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        atE = [x for x in rows if float(x["timesteps"]) == E]
        cmds = [float(x["tail_3_pitch_act_command_mean"]) for x in atE]
        cmd_raw = sum(Fr(c) for c in cmds) / len(cmds)
        print(f"    CMD_L at E from eval_episodes.csv: n {len(atE)} mean(exact) {float(cmd_raw):+.17g}; float-sum mean "
              f"{sum(cmds) / len(cmds):+.17g}; markers {cmd:+.17g}; equal(float sum) {sum(cmds) / len(cmds) == cmd}")
    for key, higher in (("SEC-G", "critic_blind_tail3"), ("SEC-CMD", "control"), ("SEC-MARGIN", "critic_blind_tail3")):
        vc = [(0, Fr(sec[key][f"control_{s}"])) for s in SEEDS]
        vb = [(0, Fr(sec[key][f"critic_blind_tail3_{s}"])) for s in SEEDS]
        p, cnt, tot = perm_p(vc, vb, higher)
        verdict = "SUPPORTED (p <= 0.05, unadjusted)" if p <= Fr(1, 20) else "NOT SUPPORTED"
        bonf = p <= Fr(1, 20) / 3
        mc = float(sum((x[1] for x in vc), Fr(0)) / 4)
        mb = float(sum((x[1] for x in vb), Fr(0)) / 4)
        medc = float(median_rule([x[1] for x in vc]))
        medb = float(median_rule([x[1] for x in vb]))
        print(f"{key}: higher {higher}; p {cnt}/{tot}; {verdict}; survives Bonferroni (p <= 1/60): {bonf}; "
              f"means c {mc:+.5f} b {mb:+.5f}; medians c {medc:+.5f} b {medb:+.5f}; "
              f"min c {float(min(x[1] for x in vc)):+.5f} max c {float(max(x[1] for x in vc)):+.5f} "
              f"min b {float(min(x[1] for x in vb)):+.5f} max b {float(max(x[1] for x in vb)):+.5f}")
        sec[key]["_p"] = (cnt, tot)
        sec[key]["_verdict"] = verdict
        sec[key]["_bonf"] = bonf

    # ---------------------------------------------------------------- compare with score.json
    sc = json.load(open(BS2 + "/score/score.json"))
    diffs_found = []

    def cmp(label, mine, theirs):
        ok = mine == theirs
        if not ok:
            diffs_found.append((label, mine, theirs))
        print(f"  {'OK ' if ok else 'DIFF'} {label}: mine {mine!r} score {theirs!r}")
    print("COMPARE WITH score.json")
    cmp("status", "final", sc["status"])
    cmp("provisional", [], sc["provisional"])
    cmp("line", 2, sc["line"])
    cmp("line_key", "REDUCED", sc["line_key"])
    cmp("baseline_S", float(B_S), sc["baseline_S"])
    cmp("primary_steps", PRIMARY_STEPS, sc["primary_steps"])
    for c in sc["classes"]:
        cmp(f"{c['name']} cls", "COUNTED", c["cls"])
        cmp(f"{c['name']} T", E, c["T"])
        cmp(f"{c['name']} primary_exact", fstr(prim[c["name"]]), c["primary_exact"])
        cmp(f"{c['name']} primary", float(prim[c["name"]]), c["primary"])
    cmp("control_median", float(med_c), sc["control_median"])
    cmp("control_rise", float(rise), sc["control_rise"])
    cmp("informative", True, sc["informative"])
    cmp("rise_min", "1/5", sc["rise_min"])
    cmp("alpha", "1/20", sc["alpha"])
    cmp("p_filled", "1/70", sc["p_filled"]["p"])
    cmp("p_filled count", 1, sc["p_filled"]["count"])
    cmp("p_as_counted", "1/70", sc["p_as_counted"]["p"])
    cmp("arm mean control", float(mean_c), sc["arm_means"]["control"])
    cmp("arm mean blind", float(mean_b), sc["arm_means"]["critic_blind_tail3"])
    cmp("arm median control", float(med_c), sc["arm_medians"]["control"])
    cmp("arm median blind", float(median_rule(blind)), sc["arm_medians"]["critic_blind_tail3"])
    cmp("SD control", sample_sd(ctrl), sc["arm_sd_report_only"]["control"])
    cmp("SD blind", sample_sd(blind), sc["arm_sd_report_only"]["critic_blind_tail3"])
    cmp("n_missing", 0, sc["n_missing"])
    for prow, s, d in zip(sc["paired"], SEEDS, diffs):
        cmp(f"paired {s} seed", s, prow["seed"])
        cmp(f"paired {s} diff", float(d), prow["diff"])
    cmp("sign flip", f"{csf}/16", sc["paired_sign_flip"]["p"])
    for srec in sc["secondaries"]:
        key = srec["key"]
        cnt, tot = sec[key]["_p"]
        pstr = "1" if cnt == tot else str(Fr(cnt, tot))
        cmp(f"{key} p_filled", pstr, srec["p_filled"]["p"])
        cmp(f"{key} p_as_counted", pstr, srec["p_as_counted"]["p"])
        cmp(f"{key} verdict", sec[key]["_verdict"], srec["verdict"])
        cmp(f"{key} bonferroni note", (["(survives Bonferroni over the 3 secondaries)"] if sec[key]["_bonf"] else []),
            srec["notes"])
        for nm in [n for (_, _, n) in names]:
            cmp(f"{key} {nm} value_exact", fstr(Fr(sec[key][nm])), srec["values_exact"][nm])
        cmp(f"{key} higher", {"SEC-G": "critic_blind_tail3", "SEC-CMD": "control",
                              "SEC-MARGIN": "critic_blind_tail3"}[key], srec["higher"])
    # ---------------------------------------------------------------- the line text, verbatim against the criteria
    crit = open(BS2 + "/criteria.txt", encoding="utf-8").read()
    m = re.search(r'2\. REDUCED \(filled p <= 0\.05\): "(.*?)"', crit, re.S)
    crit_text = re.sub(r"\s+", " ", m.group(1)).strip()
    cmp("line_text verbatim (criteria lines 265-267, whitespace joined)", crit_text, sc["line_text"])
    txt = open(BS2 + "/score/score.txt", encoding="utf-8").read().splitlines()
    cmp("score.txt line 7", "  " + crit_text, txt[6])
    cmp("additions", [], sc["additions"])
    print("DIFFERENCES:", len(diffs_found))
    for d in diffs_found:
        print("   ", d)


if __name__ == "__main__":
    main()
