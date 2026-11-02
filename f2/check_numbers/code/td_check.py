"""Independent TD re-derivation (criteria.txt lines 211-218).

CMD_L(e) = mean over evaluation e's rows of tail_3_pitch_act_command_mean (blank / non-finite cells skipped);
TD = mean over the run's evaluations of |CMD_L(e)| (an evaluation with no finite cell is MISSING and skipped).
J window (report only): evaluations from 4,194,304.
Reads only eval_episodes.csv copies (text), checked against the round manifest's F2 COPY rows.
Prints each value several ways (sequential float, fsum, exact Fraction rounded once) so a last-digit
difference can be traced to the summation order.
"""
import csv
import hashlib
import json
import math
import sys
from fractions import Fraction

F2 = "/home/user/work/pc_study/round/fetch/F2"
MAN = "/home/user/work/pc_study/round/fetch/manifest.csv"
RUNS = [f"stance7_r10LTSB{a}_s{s}" for a in ("pc", "off") for s in (60, 42, 80)]
COL = "tail_3_pitch_act_command_mean"
J0 = 4194304


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_rows():
    out = {}
    with open(MAN, newline="") as f:
        for r in csv.DictReader(f):
            if r["fetch"] == "F2" and r["status"] == "COPY" and r["path"] == "01_stance/eval_episodes.csv":
                out.setdefault(r["run_id"], []).append(r)
    return out


def cmd_l(path):
    evals = {}  # evaluation index -> (timesteps, [cells])
    order = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            e = int(r["evaluation"])
            t = int(r["timesteps"])
            if e not in evals:
                evals[e] = (t, [])
                order.append(e)
            elif evals[e][0] != t:
                raise SystemExit(f"{path}: evaluation {e} has two timesteps")
            cell = (r.get(COL) or "").strip()
            try:
                v = float(cell)
            except ValueError:
                continue
            if math.isfinite(v):
                evals[e][1].append(v)
    rows = []
    for e in order:
        t, vals = evals[e]
        if not vals:
            rows.append((e, t, None, None, 0))
            continue
        seq = 0.0
        for v in vals:
            seq += v
        c_seq = seq / len(vals)
        c_exact = Fraction(sum(Fraction(v) for v in vals), len(vals))
        rows.append((e, t, c_seq, c_exact, len(vals)))
    return rows


def td_ways(rows, jonly=False):
    sel = [r for r in rows if r[2] is not None and (not jonly or r[1] >= J0)]
    n = len(sel)
    if n == 0:
        return None
    # way A: CMD_L as sequential float mean, |.|, sequential sum / n
    a = 0.0
    for r in sel:
        a += abs(r[2])
    a /= n
    # way B: CMD_L sequential float, fsum / n
    b = math.fsum(abs(r[2]) for r in sel) / n
    # way C: exact all the way (CMD_L exact rational), rounded once
    c = float(Fraction(sum(abs(r[3]) for r in sel), n))
    # way D: CMD_L sequential float, exact mean of those floats rounded once
    d = float(Fraction(sum(Fraction(abs(r[2])) for r in sel), n))
    return {"n": n, "seq": a, "fsum": b, "exact": c, "exact_of_float_cmd": d}


def main():
    man = manifest_rows()
    res = {}
    for run in RUNS:
        base = f"{F2}/{run}/copies/01_stance/eval_episodes.csv"
        p1, p2 = base + ".copy1", base + ".copy2"
        h1, h2 = sha(p1), sha(p2)
        mrows = man.get(run, [])
        mshas = sorted({r["sha256"] for r in mrows})
        rows = cmd_l(p1)
        missing = [(e, t) for e, t, c, _, _ in rows if c is None]
        cells = sum(r[4] for r in rows)
        all_td = td_ways(rows)
        j_td = td_ways(rows, jonly=True)
        n_ge_060 = sum(1 for r in rows if r[2] is not None and r[2] >= 0.60)
        res[run] = {
            "copy1_sha256": h1, "copy2_sha256": h2, "manifest_sha256": mshas,
            "copies_equal_and_match_manifest": h1 == h2 and mshas == [h1],
            "evaluations": len(rows), "finite_cells": cells, "missing_evaluations": missing,
            "TD": all_td, "TD_J": j_td, "cmd_l_ge_0.60_count": n_ge_060,
            "first_ts": rows[0][1], "last_ts": rows[-1][1],
        }
    json.dump(res, sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
