"""PS statistic check (independent): re-derive the reading of the phase screen from the raw records with own code.

Reads only. Imports nothing from <PS>/tools (score_bs.py / read_bs.py were read as text to learn file formats only).
Writes a JSON of its own results to the path given as argv[1]; everything else goes to stdout (captured by the caller
into this checker's own folder).

Sources: <PS>/SCREEN.json (queue), <PS>/criteria.txt (line texts), F2 policy_probe.npz.copy1 (B_S), each run's
runs/<name>/{run_meta.json, check.json, arm_evidence.jsonl, 01_stance/policy_probe.npz, 01_stance/eval_episodes.csv,
01_stance/models/stage1_final.zip, stage1_final_vecnorm.pkl}, reads/<name>/markers.json (loop rows at E: b*, slopes),
reads/_cache (cross-check of the loop rows), <PS>/score/score.json (the comparison only)."""
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

PS = Path("/home/user/work/phase_screen")
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/"
          "F2/stance6_r10LTSB_s42/copies/01_stance")
PARENT_PROBE = F2 / "policy_probe.npz.copy1"
PARENT_PROBE_SHA = "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11"

ROLL = 8192
S = 2_097_152
E = 3_538_944
assert E == S + 1_441_792 and (E - S) % ROLL == 0 and (E - S) // ROLL == 176
PRIMARY_STEPS = [E - ROLL * k for k in range(8, 0, -1)]
CRIT_PRIMARY_STEPS = [3_473_408, 3_481_600, 3_489_792, 3_497_984, 3_506_176, 3_514_368, 3_522_560, 3_530_752]
assert PRIMARY_STEPS == CRIT_PRIMARY_STEPS
BASE_STEPS = [S - ROLL * k for k in range(7, -1, -1)]          # 2,039,808 ... 2,097,152
assert BASE_STEPS[0] == 2_039_808 and BASE_STEPS[-1] == S
EVAL_STEPS = [S + 131_072 * k for k in range(1, 12)]           # 2,228,224 ... 3,538,944
assert EVAL_STEPS[0] == 2_228_224 and EVAL_STEPS[-1] == E
COL_T3, COL_T2 = 14, 13
GATE = Fraction(1, 5)
ALPHA = Fraction(1, 20)
INF = float("inf")

out = {"comparisons": []}
N_CMP = [0, 0]          # (compared, differing)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmp(label, mine, theirs, tol=0.0):
    """Record one comparison; exact for Fractions/ints/strings/bools, |d| <= tol for floats (tol 0: bitwise)."""
    N_CMP[0] += 1
    if isinstance(mine, Fraction) and isinstance(theirs, str) and "/" in theirs:
        ok = mine == Fraction(theirs)
        how = "exact rational"
    elif isinstance(mine, (Fraction, float, int)) and not isinstance(mine, bool) and isinstance(theirs, (float, int)) \
            and not isinstance(theirs, bool):
        d = abs(float(mine) - float(theirs))
        if isinstance(mine, Fraction) and tol == 0.0:
            ok = float(mine) == float(theirs)
            how = "float(mine) == theirs"
        else:
            ok = d <= tol
            how = f"|d| = {d:.3e} <= {tol:g}"
    else:
        ok = mine == theirs
        how = "equal"
    if not ok:
        N_CMP[1] += 1
    out["comparisons"].append({"label": label, "mine": str(mine), "theirs": str(theirs), "ok": ok, "how": how})
    flag = "OK  " if ok else "DIFF"
    print(f"  [{flag}] {label}: mine {fmt(mine)} | score {fmt(theirs)} ({how})")
    return ok


def fmt(x):
    if isinstance(x, Fraction):
        return f"{float(x):+.17g} ({x.numerator}/{x.denominator})" if x.denominator < 10 ** 30 else f"{float(x):+.17g}"
    if isinstance(x, float):
        return repr(x)
    return str(x)


def median_rule(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    if n % 2:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2


def sample_sd(vals):
    n = len(vals)
    if n < 2:
        return None
    m = sum(vals, Fraction(0)) / n
    var = sum(((x - m) ** 2 for x in vals), Fraction(0)) / (n - 1)
    return math.sqrt(var)


# ------------------------------------------------------------------ own permutation test (difference of arm means)
# A value is (inf, fin): inf in {-1, 0, +1} (a fill), fin a Fraction. Arm mean = (sum inf / n, sum fin / n), compared
# infinite part first. Statistic = mean(higher arm) - mean(other arm), computed explicitly (not the sum shortcut).
def arm_stat(hi_vals, lo_vals):
    nh, nl = len(hi_vals), len(lo_vals)
    inf = Fraction(sum(v[0] for v in hi_vals), nh) - Fraction(sum(v[0] for v in lo_vals), nl)
    fin = sum((v[1] for v in hi_vals), Fraction(0)) / nh - sum((v[1] for v in lo_vals), Fraction(0)) / nl
    return (inf, fin)


def perm_p(hi_vals, lo_vals):
    if not hi_vals or not lo_vals:
        return None
    pool = list(hi_vals) + list(lo_vals)
    n, k = len(pool), len(hi_vals)
    obs = arm_stat(hi_vals, lo_vals)
    count = total = 0
    stats = []
    for sub in itertools.combinations(range(n), k):
        sset = set(sub)
        st = arm_stat([pool[i] for i in sub], [pool[i] for i in range(n) if i not in sset])
        stats.append(st)
        total += 1
        if st >= obs:
            count += 1
    return Fraction(count, total), count, total, obs, sorted(stats, reverse=True)


def sign_flip_p(diffs):
    obs = sum(diffs, Fraction(0))
    count = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if sum((s * d for s, d in zip(signs, diffs)), Fraction(0)) >= obs:
            count += 1
    return Fraction(count, total), count, total


def fill(v, missing_inf):
    return (0, v) if v is not None else (missing_inf, Fraction(0))


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def npz_rows(path, steps, cols):
    """{col: [float64 values]} at the given update_timesteps; each step exactly once (else raises)."""
    with np.load(path, allow_pickle=False) as z:
        ut = np.asarray(z["update_timesteps"]).astype(np.int64)
        mu = np.asarray(z["rollout_mu_mean"])
    assert mu.dtype == np.float32, mu.dtype
    res = {c: [] for c in cols}
    for s in steps:
        idx = np.flatnonzero(ut == s)
        if len(idx) != 1:
            raise SystemExit(f"{path}: {len(idx)} rows at {s}")
        for c in cols:
            v = float(mu[int(idx[0]), c])      # float32 -> float64 exact
            if not math.isfinite(v):
                raise SystemExit(f"{path}: col {c} at {s} not finite")
            res[c].append(v)
    return res, ut, mu


def fr_mean(xs):
    return sum((Fraction(x) for x in xs), Fraction(0)) / len(xs)


def main():
    score = json.loads((PS / "score/score.json").read_text())
    screen = json.loads((PS / "SCREEN.json").read_text())
    queue = [(a, int(s)) for a, s in screen["queue"]]
    assert queue == [("control", 2901), ("critic_phase", 2901), ("control", 2911), ("critic_phase", 2911),
                     ("control", 2921), ("critic_phase", 2921), ("control", 2931), ("critic_phase", 2931)], queue
    names = [f"{a}_{s}" for a, s in queue]
    print(f"queue (SCREEN.json sha256 {sha(PS / 'SCREEN.json')}): {names}")

    # ------------------------------------------------------------------ B_S
    print("\n== B_S (parent F2 probe) ==")
    psha = sha(PARENT_PROBE)
    print(f"  parent probe sha256 {psha} (pin {PARENT_PROBE_SHA}: {'equal' if psha == PARENT_PROBE_SHA else 'DIFFERENT'})")
    pr, put, pmu = npz_rows(PARENT_PROBE, BASE_STEPS, [COL_T3, COL_T2])
    rows_idx = [int(np.flatnonzero(put == s)[0]) for s in BASE_STEPS]
    print(f"  parent rows used (0-based) {rows_idx} at update_timesteps {BASE_STEPS[0]:,} ... {BASE_STEPS[-1]:,}; "
          f"parent rows with update_timesteps <= S: {int((put <= S).sum())}; grid S+8192k check: "
          f"{bool(np.all(put[:256] == ROLL * np.arange(1, 257)))}")
    B = fr_mean(pr[COL_T3])
    print(f"  B_S = {B} = {float(B)!r}; criteria 135930855/1073741824: {B == Fraction(135930855, 1073741824)}; "
          f"float 0.1265954738482833: {float(B) == 0.1265954738482833}")
    parent_t2 = fr_mean(pr[COL_T2])
    print(f"  parent tail_2 at S (cols 13, same rows) {float(parent_t2):+.5f} (criteria +0.0233)")
    out["B_S"] = str(B)
    cmp("B_S (baseline_S)", B, score["baseline_S"])

    # ------------------------------------------------------------------ per run
    print("\n== per continuation: classification, sources, primary, tail_2, CMD_L, loop row at E ==")
    runs = {}
    for a, s in queue:
        name = f"{a}_{s}"
        rdir = PS / "runs" / name
        r = {"arm": a, "seed": s, "name": name}
        meta = json.loads((rdir / "run_meta.json").read_text())
        chk = json.loads((rdir / "check.json").read_text())
        mk = json.loads((PS / "reads" / name / "markers.json").read_text())
        r["meta"], r["check"], r["markers"] = meta, chk, mk
        # classification (criteria RUN HANDLING): COUNTED = CHECK passed, stop reason "end" at E
        counted = (chk.get("passed") is True and not chk.get("failures") and chk.get("mode") == "continuation"
                   and meta.get("status") == "done" and meta.get("exit") == 0 and meta.get("stop_reason") == "end"
                   and meta.get("T") == E and meta.get("num_timesteps_end") == E and chk.get("T") == E
                   and chk.get("stop_reason") == "end" and mk.get("T") == E and mk.get("stop_reason") == "end"
                   and meta.get("arm") == a and meta.get("seed") == s)
        r["cls"] = "COUNTED" if counted else "NOT COUNTED"
        # check.json's sources against the files now
        src_ok = {}
        for k, (p, h) in chk["sources"].items():
            src_ok[k] = sha(p) == h
        r["src_ok"] = src_ok
        probe_path = rdir / "01_stance/policy_probe.npz"
        r["probe_sha"] = sha(probe_path)
        probe_sha_ok = (r["probe_sha"] == chk["sources"]["policy_probe.npz"][1] == mk["probe"]["sha256"]
                        and Path(chk["sources"]["policy_probe.npz"][0]) == probe_path
                        and Path(mk["probe"]["file"]) == probe_path)
        r["probe_sha_ok"] = probe_sha_ok
        # probe: the window's rows and the primary
        rows, ut, mu = npz_rows(probe_path, PRIMARY_STEPS, list(range(15)))
        win = ut[(ut > S) & (ut < E)]
        grid_ok = (len(win) == 175 and bool(np.all(win == S + ROLL * np.arange(1, 176)))
                   and int((ut <= S).sum()) == 256 and int((ut >= E).sum()) == 0)
        # rows at or before S equal the parent's rows (report only; the probe was seeded from the parent's file)
        seeded_equal = bool(np.array_equal(mu[:256], pmu[:256])) and bool(np.array_equal(ut[:256], put[:256]))
        prim_idx = [int(np.flatnonzero(ut == st)[0]) for st in PRIMARY_STEPS]
        prim_rows_in_window = [int(np.flatnonzero(win == st)[0]) + 1 for st in PRIMARY_STEPS]
        r["primary"] = fr_mean(rows[COL_T3])
        r["tail2"] = fr_mean(rows[COL_T2])
        r["act_means"] = [fr_mean(rows[j]) for j in range(15)]
        # markers' rollout_mu14_mean at the same steps (exact equality)
        mst = mk["probe"]["steps"]
        mvals = []
        for st in PRIMARY_STEPS:
            ii = [i for i, x in enumerate(mst) if x == st]
            mvals.append(mk["probe"]["rollout_mu14_mean"][ii[0]] if len(ii) == 1 else None)
        r["markers_mu14_equal"] = all(mv is not None and Fraction(mv) == Fraction(v)
                                      for mv, v in zip(mvals, rows[COL_T3]))
        # early 8 rows (report only)
        e_steps = [S + ROLL * k for k in range(1, 9)]
        er, _, _ = npz_rows(probe_path, e_steps, [COL_T3])
        r["early8"] = fr_mean(er[COL_T3])
        # trajectory (8-row means ending at each evaluation step; at E the primary's rows), report only
        traj = {}
        for ev in EVAL_STEPS:
            end = ev if ev < E else E - ROLL          # rows ending AT ev (inclusive); at E (no row) the primary's rows
            st8 = [end - ROLL * k for k in range(7, -1, -1)]
            tr, _, _ = npz_rows(probe_path, st8, [COL_T3])
            traj[ev] = fr_mean(tr[COL_T3])
        r["traj"] = traj
        # CMD_L from the evaluation ledger (raw CSV)
        led = rdir / "01_stance/eval_episodes.csv"
        r["ledger_sha"] = sha(led)
        with open(led, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            allrows = list(rd)
        byt = {}
        for row in allrows:
            t = float(row["timesteps"])
            if t == int(t) and S < int(t) <= E:
                byt.setdefault(int(t), []).append(row)
        r["eval_steps_ok"] = sorted(byt) == EVAL_STEPS and all(len(v) == 10 for v in byt.values())
        atE = byt.get(E, [])
        txt = [row["tail_3_pitch_act_command_mean"] for row in atE]
        vals = [float(t) for t in txt]
        r["cmd_n"] = len(vals)
        r["cmd_finite"] = all(math.isfinite(v) for v in vals)
        r["cmd_l_exact_text"] = sum((Fraction(Decimal(t)) for t in txt), Fraction(0)) / len(txt)
        fsum = 0.0
        for v in vals:
            fsum += v
        r["cmd_l_float"] = fsum / len(vals)
        r["episodes_at_E"] = sorted(int(row["episode"]) for row in atE)
        r["cmd_l_per_eval"] = {t: sum(float(x["tail_3_pitch_act_command_mean"]) for x in v) / len(v)
                               for t, v in sorted(byt.items())}
        mk_cmd = get(mk, "ledger", "direct", "per_evaluation", str(E), "cmd_l")
        r["markers_cmd_l"] = mk_cmd
        r["ledger_sha_ok"] = r["ledger_sha"] == mk["ledger"]["sha256"] and Path(mk["ledger"]["file"]) == led
        # loop rows
        loop = mk.get("loop") or []
        atE_rows = [lr for lr in loop if lr.get("steps") == E]
        atS_rows = [lr for lr in loop if lr.get("steps") == S]
        r["n_loop"] = len(loop)
        r["loop_steps"] = [lr.get("steps") for lr in loop]
        assert len(atE_rows) == 1 and len(atS_rows) == 1, (name, len(atE_rows), len(atS_rows))
        lE, lS = atE_rows[0], atS_rows[0]
        r["loopE"] = lE
        fz, fp = rdir / "01_stance/models/stage1_final.zip", rdir / "01_stance/models/stage1_final_vecnorm.pkl"
        r["final_zip_ok"] = (Path(lE["zip"]) == fz and sha(fz) == lE["zip_sha256"] and Path(lE["pkl"]) == fp
                             and sha(fp) == lE["pkl_sha256"] and lE.get("num_timesteps") == E)
        G, c0, bstar = lE["OWN"]["P"]["G"], lE["OWN"]["P"]["c0"], lE["OWN"]["P"]["b_star"]
        r["b_star"] = bstar
        r["b_star_recomp"] = 1.0 - G - c0
        r["b_star_exact_err"] = float(Fraction(bstar) - (1 - Fraction(G) - Fraction(c0)))
        r["G_E"], r["c0_E"], r["G_S"] = G, c0, lS["OWN"]["P"]["G"]
        cs = lE["critic_slope_obs6"]["ONPOLICY"]
        r["slope_eff"] = cs["effective"]["dV_dq3_per_deg"]
        r["slope_raw"] = cs["raw"]["dV_dq3_per_deg"]
        r["slope_S"] = lS["critic_slope_obs6"]["ONPOLICY"]["effective"]["dV_dq3_per_deg"]
        r["critic_form"] = lE.get("critic_form")
        r["critic_forms_all"] = sorted({lr.get("critic_form") for lr in loop})
        # internal consistency of the slope record
        notes = []
        if a == "control":
            if cs["effective"] != cs["raw"]:
                notes.append("control: effective != raw")
            if "settled" in cs or lE.get("phase") is not None:
                notes.append("control: phase record present")
        else:
            ph = lE.get("phase") or {}
            fed = get(ph, "fed", "ONPOLICY") or {}
            if fed.get("n") != cs["effective"]["n"]:
                notes.append(f"phase fed n {fed.get('n')} != slope n {cs['effective']['n']}")
            if not (fed.get("k_min") == 0 and fed.get("k_max") is not None and fed.get("k_max") <= 999):
                notes.append(f"phase fed k range {fed.get('k_min')}..{fed.get('k_max')}")
            if cs["effective"]["dV_dq3_per_deg"] == cs["raw"]["dV_dq3_per_deg"]:
                notes.append("phase: effective == raw at E")
            if "settled" not in cs:
                notes.append("phase: no settled slope")
            r["phase_fed_onpolicy"] = fed
            r["swing"] = cs["settled"]["V_mean"] - cs["raw"]["V_mean"]
            r["swing_own"] = cs["effective"]["V_mean"] - cs["raw"]["V_mean"]
        if cs["effective"]["n"] != 16384 or cs["effective"].get("n_clipped") is None:
            notes.append(f"slope n {cs['effective']['n']} clipped {cs['effective'].get('n_clipped')}")
        r["slope_n"], r["slope_n_clipped"] = cs["effective"]["n"], cs["effective"].get("n_clipped")
        r["slope_notes"] = notes
        # the reader's loop cache: the entry for the final pair holds the same loop row
        zp, pp = lE["zip_sha256"][:16], lE["pkl_sha256"][:16]
        form = "plain" if a == "control" else "phase"
        cands = sorted((PS / "reads/_cache").glob(f"r{mk['tool_sha256'][:16]}_*_{zp}_{pp}_{form}_*.json"))
        cache_ok = None
        if len(cands) == 1:
            cj = json.loads(cands[0].read_text())
            row = cj.get("row", cj)
            keys_cmp = [("OWN", "P", "b_star"), ("OWN", "P", "G"), ("OWN", "P", "c0"),
                        ("critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")]
            cache_ok = all(get(row, *k) == get(lE, *k) for k in keys_cmp)
        r["cache"] = (len(cands), cache_ok)
        runs[name] = r
        print(f"  {name:<18} {r['cls']}; check passed {chk.get('passed')} failures {chk.get('failures')}; "
              f"sources now == check.json {src_ok}; probe sha {r['probe_sha'][:12]} == check == markers {probe_sha_ok}; "
              f"window rows 175 on grid, 256 seeded, none >= E: {grid_ok}; seeded rows == parent's: {seeded_equal}")
        print(f"      primary rows (0-based in file) {prim_idx} = window rows {prim_rows_in_window[0]}..{prim_rows_in_window[-1]} "
              f"of 175; primary {float(r['primary']):+.10f} ({r['primary']}); markers mu14 equal exactly "
              f"{r['markers_mu14_equal']}; tail_2 {float(r['tail2']):+.10f}")
        print(f"      ledger sha == markers {r['ledger_sha_ok']}; 11 evaluations x 10 rows in (S, E]: {r['eval_steps_ok']};"
              f" CMD_L at E n={r['cmd_n']} finite {r['cmd_finite']} episodes {r['episodes_at_E']}: exact(text) "
              f"{float(r['cmd_l_exact_text']):+.17g}, float sum/len {r['cmd_l_float']!r}, markers {mk_cmd!r}")
        print(f"      loop rows {r['n_loop']} at steps {r['loop_steps']}; critic forms {r['critic_forms_all']}; final "
              f"pair zip/pkl sha == files and num_timesteps == E: {r['final_zip_ok']}; b* {bstar!r} vs 1-G-c0 "
              f"{r['b_star_recomp']!r} (exact-rational error {r['b_star_exact_err']:.3e}); slope eff {r['slope_eff']!r} "
              f"raw {r['slope_raw']!r} (n {r['slope_n']}, clipped {r['slope_n_clipped']}); slope at S "
              f"{r['slope_S']!r}; consistency notes {notes or 'none'}; cache entries {r['cache']}")
    out["runs"] = {n: {k: (str(v) if isinstance(v, Fraction) else v) for k, v in r.items()
                       if k not in ("meta", "check", "markers", "loopE", "traj", "act_means", "cmd_l_per_eval")}
                   for n, r in runs.items()}

    # ------------------------------------------------------------------ the primary, gate, test
    print("\n== THE PRIMARY, THE GATE, THE TEST ==")
    ctrl = [runs[f"control_{s}"] for s in (2901, 2911, 2921, 2931)]
    phase = [runs[f"critic_phase_{s}"] for s in (2901, 2911, 2921, 2931)]
    cp = [r["primary"] if r["cls"] == "COUNTED" else None for r in ctrl]
    pp_ = [r["primary"] if r["cls"] == "COUNTED" else None for r in phase]
    for r in ctrl + phase:
        print(f"  {r['name']:<18} {r['cls']} primary {float(r['primary']):+.5f} rise {float(r['primary'] - B):+.5f}")
    cread = [v for v in cp if v is not None]
    pread = [v for v in pp_ if v is not None]
    med_c = median_rule(cread)
    rise = med_c - B
    informative = med_c is not None and rise >= GATE
    print(f"  control median (midpoint of x(2), x(3)) {float(med_c):+.17g} = {med_c}; rise {float(rise):+.17g} = {rise}; "
          f">= 1/5: {rise >= GATE} -> informative {informative}")
    mean_c, mean_p = sum(cread, Fraction(0)) / len(cread), sum(pread, Fraction(0)) / len(pread)
    sd_c, sd_p = sample_sd(cread), sample_sd(pread)
    med_p = median_rule(pread)
    print(f"  control mean {float(mean_c):+.6f} median {float(med_c):+.6f} SD {sd_c:.6f}; phase mean {float(mean_p):+.6f}"
          f" median {float(med_p):+.6f} SD {sd_p:.6f}; diff of means {float(mean_c - mean_p):+.6f}")
    pf = perm_p([fill(v, -1) for v in cp], [fill(v, +1) for v in pp_])
    pc = perm_p([(0, v) for v in cread], [(0, v) for v in pread])
    print(f"  p filled {pf[1]}/{pf[2]} = {float(pf[0]):.6f}; p as counted {pc[1]}/{pc[2]}; observed statistic "
          f"{float(pf[3][1]):+.6f}; the next statistic {float(pf[4][1][1]):+.6f}; distinct statistics "
          f"{len(set(pf[4]))}")
    print(f"  complete separation: min control {float(min(cread)):+.5f} > max phase {float(max(pread)):+.5f}: "
          f"{min(cread) > max(pread)}")
    if not informative:
        line = 1
    elif pf[0] <= ALPHA:
        line = 2
    else:
        line = 3
    adds = []
    if line == 1 and pf[0] <= ALPHA:
        adds.append("NOT INFORMATIVE addition")
    if line == 3 and pc[0] <= ALPHA and pf[0] > ALPHA:
        adds.append("as-counted addition")
    crit = (PS / "criteria.txt").read_text()
    texts = {}
    for k, key in ((1, "NOT INFORMATIVE"), (2, "REDUCED"), (3, "NO REDUCTION DETECTED")):
        m = re.search(rf"\n  {k}\. {re.escape(key)}[^\"]*\"(.*?)\"", crit, re.S)
        texts[k] = re.sub(r"\s+", " ", m.group(1)).strip()
    print(f"  LINE {line} ({ {1: 'NOT INFORMATIVE', 2: 'REDUCED', 3: 'NO REDUCTION DETECTED'}[line] }); additions "
          f"{adds or 'none'}")
    print(f"  criteria line {line} text: {texts[line]}")
    paired = []
    for s in (2901, 2911, 2921, 2931):
        c_, p_ = runs[f"control_{s}"]["primary"], runs[f"critic_phase_{s}"]["primary"]
        paired.append((s, c_, p_, c_ - p_))
    sf = sign_flip_p([d for *_, d in paired])
    for s, c_, p_, d in paired:
        print(f"  pair {s}: control {float(c_):+.5f} phase {float(p_):+.5f} diff {float(d):+.5f}")
    print(f"  sign-flip p {sf[1]}/{sf[2]}")

    print("\n  -- comparison with score.json (primary, gate, test) --")
    cmp("status", "final", score["status"])
    cmp("provisional items", [], score["provisional"])
    cmp("line", line, score["line"])
    cmp("line_key", "REDUCED" if line == 2 else str(line), score["line_key"])
    cmp("line_text (criteria verbatim, whitespace-normalised)", texts[line], score["line_text"])
    cmp("additions", [], score["additions"])
    cmp("p_filled", pf[0], score["p_filled"]["p"])
    cmp("p_filled count/total", (pf[1], pf[2]), (score["p_filled"]["count"], score["p_filled"]["total"]))
    cmp("p_as_counted", pc[0], score["p_as_counted"]["p"])
    cmp("alpha", ALPHA, score["alpha"])
    cmp("informative", informative, score["informative"])
    cmp("control_median", med_c, score["control_median"])
    cmp("control_rise", rise, score["control_rise"], tol=1e-16)
    cmp("rise_min", GATE, score["rise_min"])
    cmp("primary_steps", PRIMARY_STEPS, score["primary_steps"])
    cmp("arm mean control", mean_c, score["arm_means"]["control"], tol=1e-16)
    cmp("arm mean critic_phase", mean_p, score["arm_means"]["critic_phase"], tol=1e-16)
    cmp("arm median control", med_c, score["arm_medians"]["control"])
    cmp("arm median critic_phase", med_p, score["arm_medians"]["critic_phase"])
    cmp("arm SD control (n-1)", sd_c, score["arm_sd_report_only"]["control"], tol=1e-15)
    cmp("arm SD critic_phase (n-1)", sd_p, score["arm_sd_report_only"]["critic_phase"], tol=1e-15)
    cmp("n_missing", 0, score["n_missing"])
    cmp("missing lists", {"control": [], "critic_phase": []}, score["missing"])
    for (s, c_, p_, d), sp in zip(paired, score["paired"]):
        cmp(f"paired seed {s} seed", s, sp["seed"])
        cmp(f"paired seed {s} control", c_, sp["control"])
        cmp(f"paired seed {s} phase", p_, sp["phase"])
        cmp(f"paired seed {s} diff", d, sp["diff"], tol=1e-16)
    cmp("sign-flip p", sf[0], score["paired_sign_flip"]["p"])
    for c in score["classes"]:
        r = runs[c["name"]]
        cmp(f"{c['name']} class", r["cls"], c["cls"])
        cmp(f"{c['name']} T", E, c["T"])
        cmp(f"{c['name']} primary exact", r["primary"], c["primary_exact"])
        cmp(f"{c['name']} primary float", r["primary"], c["primary"])
    for n in names:
        cmp(f"{n} probe file sha256 (score probe_files)", runs[n]["probe_sha"], score["probe_files"][n]["sha256"])

    # ------------------------------------------------------------------ secondaries
    print("\n== THE SECONDARIES ==")
    sec_def = {
        # key: (value getter, higher arm, family)
        "SEC-CMD": (lambda r: Fraction(r["markers_cmd_l"]), "control", "efficacy"),
        "SEC-MARGIN": (lambda r: Fraction(r["b_star"]), "critic_phase", "efficacy"),
        "SEC-MECH": (lambda r: Fraction(r["slope_eff"]), "control", "efficacy"),
        "SEC-TAIL2": (lambda r: r["tail2"], "critic_phase", "guard"),
    }
    sec_by_key = {s["key"]: s for s in score["secondaries"]}
    cmp("secondaries listed", list(sec_def), [s["key"] for s in score["secondaries"]])
    mine_sec = {}
    for key, (getv, higher, fam) in sec_def.items():
        sc = sec_by_key[key]
        vals = {n: (getv(runs[n]) if runs[n]["cls"] == "COUNTED" else None) for n in names}
        hi = [vals[n] for n in names if runs[n]["arm"] == higher]
        lo = [vals[n] for n in names if runs[n]["arm"] != higher]
        # fill: a MISSING value counts as the infinity that works against the secondary's direction
        p_f = perm_p([fill(v, -1) for v in hi], [fill(v, +1) for v in lo])
        p_c = perm_p([(0, v) for v in hi if v is not None], [(0, v) for v in lo if v is not None])
        sup = p_f[0] <= ALPHA
        verdict = "SUPPORTED (p <= 0.05, unadjusted)" if sup else "NOT SUPPORTED"
        notes = []
        if fam == "efficacy" and p_f[0] <= ALPHA / 3:
            notes.append("(survives Bonferroni over the 3 efficacy secondaries)")
        if fam == "guard" and sup:
            notes.append("(guard: the habit may be moving to tail_2; a warning, not support for the phase input)")
        if fam == "guard" and any(v is None for v in vals.values()):
            notes.append("(with any MISSING value the guard cannot warn: read NOT SUPPORTED as not assessed)")
        cv = [vals[n] for n in names if runs[n]["arm"] == "control"]
        pv = [vals[n] for n in names if runs[n]["arm"] == "critic_phase"]
        means = {"control": sum(cv, Fraction(0)) / 4, "critic_phase": sum(pv, Fraction(0)) / 4}
        meds = {"control": median_rule(cv), "critic_phase": median_rule(pv)}
        mine_sec[key] = {"p_filled": str(p_f[0]), "p_as_counted": str(p_c[0]), "verdict": verdict, "notes": notes,
                         "values": {n: float(v) for n, v in vals.items()}, "higher": higher,
                         "min_hi_minus_max_lo": float(min(hi) - max(lo))}
        print(f"  {key} ({fam}; higher arm {higher}): p filled {p_f[1]}/{p_f[2]}, as counted {p_c[1]}/{p_c[2]} -> "
              f"{verdict} {' '.join(notes)}; arm means control {float(means['control']):+.5f} phase "
              f"{float(means['critic_phase']):+.5f}; medians {float(meds['control']):+.5f} / "
              f"{float(meds['critic_phase']):+.5f}; min(higher) - max(other) {float(min(hi) - max(lo)):+.5f}")
        for n in names:
            print(f"      {n:<18} {float(vals[n]):+.6f}")
        cmp(f"{key} higher arm", higher, sc["higher"])
        cmp(f"{key} family", fam, sc["family"])
        cmp(f"{key} verdict", verdict, sc["verdict"])
        cmp(f"{key} supported", sup, sc["supported"])
        cmp(f"{key} notes", notes, sc["notes"])
        cmp(f"{key} p_filled", p_f[0], sc["p_filled"]["p"])
        cmp(f"{key} p_as_counted", p_c[0], sc["p_as_counted"]["p"])
        for n in names:
            cmp(f"{key} {n} value (exact)", vals[n], sc["values_exact"][n])
            cmp(f"{key} {n} value (float)", vals[n], sc["values"][n])
            cmp(f"{key} {n} cause", None, sc["causes"][n])
        cmp(f"{key} n_read", {"control": 4, "critic_phase": 4}, sc["n_read"])
        for arm in ("control", "critic_phase"):
            cmp(f"{key} arm mean {arm}", means[arm], sc["arm_means"][arm], tol=1e-16)
            cmp(f"{key} arm median {arm}", meds[arm], sc["arm_medians"][arm], tol=1e-16)
    out["secondaries"] = mine_sec

    # SEC-CMD from the raw ledger: exact mean of the CSV text and float mean against the markers' value
    print("\n  -- SEC-CMD from the raw ledger (eval_episodes.csv at E) against the markers' cmd_l --")
    for n in names:
        r = runs[n]
        d_exact = float(r["cmd_l_exact_text"]) - r["markers_cmd_l"]
        print(f"  {n:<18} csv exact {float(r['cmd_l_exact_text']):+.17g} csv float {r['cmd_l_float']!r} markers "
              f"{r['markers_cmd_l']!r}; float == markers {r['cmd_l_float'] == r['markers_cmd_l']}; exact - markers "
              f"{d_exact:+.2e}")
        cmp(f"SEC-CMD {n} raw-ledger exact mean vs score value", r["cmd_l_exact_text"],
            sec_by_key["SEC-CMD"]["values"][n], tol=1e-15)
    # the verdict from the raw-ledger exact values (independent of the markers)
    hi = [runs[f"control_{s}"]["cmd_l_exact_text"] for s in (2901, 2911, 2921, 2931)]
    lo = [runs[f"critic_phase_{s}"]["cmd_l_exact_text"] for s in (2901, 2911, 2921, 2931)]
    p_raw = perm_p([(0, v) for v in hi], [(0, v) for v in lo])
    print(f"  SEC-CMD p from the raw-ledger exact values: {p_raw[1]}/{p_raw[2]}")
    out["sec_cmd_p_raw_ledger"] = str(p_raw[0])

    # ------------------------------------------------------------------ report-only cross-checks
    print("\n== REPORT-ONLY cross-checks (never tested) ==")
    for n in names:
        r = runs[n]
        ro = score["report_only"][n]
        cmp(f"{n} early8", r["early8"], ro["early8_rollout_mu14"])
        cmp(f"{n} last8 before T", r["primary"], ro["last8_rollout_mu14_before_T"])
        cmp(f"{n} cmd_l_last", r["markers_cmd_l"], ro["cmd_l_last"])
        cmp(f"{n} slope E", r["slope_eff"], ro["slope_end_vs_start"]["E"])
        cmp(f"{n} slope S", r["slope_S"], ro["slope_end_vs_start"]["S"])
        cmp(f"{n} slope E<=S", r["slope_eff"] <= r["slope_S"], ro["slope_end_vs_start"]["E_le_S"])
        cmp(f"{n} slope E<=0", r["slope_eff"] <= 0, ro["slope_end_vs_start"]["E_le_0"])
        cmp(f"{n} G(E) > G(S)", r["G_E"] > r["G_S"], ro["g_end_above_start"])
        for ev, v in r["traj"].items():
            t = ro["trajectory"]
            tv = None
            if isinstance(t, list):
                for item in t:
                    if isinstance(item, dict) and item.get("step") == ev:
                        tv = item.get("value", item.get("mean"))
                    elif isinstance(item, (list, tuple)) and item[0] == ev:
                        tv = item[1]
            if tv is not None:
                cmp(f"{n} trajectory {ev}", v, tv, tol=1e-15)
        for ev, v in r["cmd_l_per_eval"].items():
            tv = ro["cmd_l_per_evaluation"].get(str(ev), ro["cmd_l_per_evaluation"].get(ev))
            cmp(f"{n} CMD_L per evaluation {ev} (raw csv float)", v, tv, tol=1e-15)
    # per-arm counts
    for arm in ("control", "critic_phase"):
        rs = ctrl if arm == "control" else phase
        cmp(f"G(E)>G(S) count {arm}", sum(r["G_E"] > r["G_S"] for r in rs), score["g_end_above_start"][arm]["above"])
        cmp(f"slope E<=S count {arm}", sum(r["slope_eff"] <= r["slope_S"] for r in rs),
            score["slope_end_vs_start"][arm]["E_le_S"])
        cmp(f"slope E<=0 count {arm}", sum(r["slope_eff"] <= 0 for r in rs), score["slope_end_vs_start"][arm]["E_le_0"])
    # trajectory arm medians (report only) from the raw probes
    for ev in EVAL_STEPS:
        for arm in ("control", "critic_phase"):
            rs = ctrl if arm == "control" else phase
            mv = median_rule([r["traj"][ev] for r in rs])
            tv = score["trajectory_arm_medians"].get(str(ev), {}).get(arm)
            cmp(f"trajectory median {ev} {arm}", mv, tv, tol=1e-15)
    # parent trajectory from F2
    for ev in EVAL_STEPS:
        end = ev if ev < E else E - ROLL
        st8 = [end - ROLL * k for k in range(7, -1, -1)]
        tr, _, _ = npz_rows(PARENT_PROBE, st8, [COL_T3])
        cmp(f"parent trajectory {ev}", fr_mean(tr[COL_T3]), score["parent_trajectory"][str(ev)], tol=1e-15)
    # phase swing at E (report only)
    for r in phase:
        print(f"  {r['name']}: phase swing at E (V settled - V raw, on-policy) {r['swing']:+.4f}; own phase - raw "
              f"{r['swing_own']:+.4f}; fed ONPOLICY {r['phase_fed_onpolicy']}")
    # paired first-rollout digests (C2: the arms see the same first rollout)
    print("\n  -- pairing: rollout 1 digests equal between the arms of each seed (report only) --")
    for s in (2901, 2911, 2921, 2931):
        a1 = json.loads((PS / f"runs/control_{s}/arm_evidence.jsonl").read_text().splitlines()[0])
        b1 = json.loads((PS / f"runs/critic_phase_{s}/arm_evidence.jsonl").read_text().splitlines()[0])
        eq = {k: a1[k] == b1[k] for k in ("digest_env_side_no_phase", "digest_rewards", "digest_values",
                                           "digest_advantages")}
        a2 = json.loads((PS / f"runs/control_{s}/arm_evidence.jsonl").read_text().splitlines()[1])
        b2 = json.loads((PS / f"runs/critic_phase_{s}/arm_evidence.jsonl").read_text().splitlines()[1])
        print(f"  seed {s}: rollout 1 {eq}; rollout 2 env side equal {a2['digest_env_side_no_phase'] == b2['digest_env_side_no_phase']}")

    out["mine"] = {"line": line, "p_filled": str(pf[0]), "p_as_counted": str(pc[0]), "B_S": str(B),
                   "control_median": str(med_c), "control_rise": str(rise), "informative": informative,
                   "primaries": {n: str(runs[n]["primary"]) for n in names},
                   "primaries_float": {n: float(runs[n]["primary"]) for n in names},
                   "sd": {"control": sd_c, "critic_phase": sd_p},
                   "means": {"control": float(mean_c), "critic_phase": float(mean_p)},
                   "sign_flip": str(sf[0]), "paired": [(s, float(d)) for s, _, _, d in paired]}
    print(f"\nCOMPARISONS: {N_CMP[0]} made, {N_CMP[1]} differing")
    out["n_comparisons"], out["n_differing"] = N_CMP
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
