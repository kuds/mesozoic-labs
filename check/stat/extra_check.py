"""Extra checks (stat lens): robustness of the ANY input, borderline evaluations, score.txt report-only fields,
pins and status inputs.  Reads only; prints; writes extra_out.txt in its own folder.  Imports no ECS/M5 tool."""
import csv
import hashlib
import io
import json
import re
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ECS = Path(S2) / "r10round/ecs"
LANE = Path(S2) / "r10round/lean/build"
STARTS = {"s42": (3_145_728, 4_718_592), "s80": (4_194_304, 5_767_168)}
GRID = 131072
KEY = "max_actuator_saturation_fraction"
L = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fnum(c):
    try:
        v = float(c)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def rows_of(p):
    rd = csv.DictReader(io.StringIO(Path(p).read_bytes().decode("utf-8"), newline=""))
    return list(rd)


# 1. ANY input robustness across all 24 ledgers (every row, parent and window)
contra, listed_under, eval_col_bad, unmeasured = [], [], [], 0
alt_flags_diff = []
border = []
for d in sorted((ECS / "runs").iterdir()):
    name = d.name
    st = name[:3]
    S, E = STARTS[st]
    rows = rows_of(d / "01_stance/eval_episodes.csv")
    by_list, by_frac, by_frac_any = {}, {}, {}
    for r in rows:
        t = int(float(r["timesteps"]))
        listed = KEY in {x.split(":", 1)[0].strip() for x in r["screen_v2_reasons"].split(";") if x.strip()}
        clean = fnum(r["screen_v2_clean"])
        if clean is None:
            unmeasured += 1
        fr = fnum(r["saturated_actuator_fraction"])
        if listed and not (fr is not None and fr > 0.10):
            listed_under.append((name, t, r["episode"], r["saturated_actuator_fraction"]))
        if (not listed) and fr is not None and fr > 0.10:
            contra.append((name, t, r["episode"], fr))
        ev = int(float(r["evaluation"]))
        if (ev + 1) * GRID != t:          # the column is 0-indexed
            eval_col_bad.append((name, ev, t))
        by_list[t] = by_list.get(t, 0) + (1 if listed else 0)
        by_frac[t] = by_frac.get(t, 0) + (1 if (fr is not None and fr > 0.10) else 0)
        if S < t <= E and listed:
            by_frac_any.setdefault(t, []).append((r["episode"], r["saturated_actuator"], fr))
    fl_list = sorted(t for t, n in by_list.items() if n >= 5 and S < t <= E)
    fl_frac = sorted(t for t, n in by_frac.items() if n >= 5 and S < t <= E)
    if fl_list != fl_frac:
        alt_flags_diff.append((name, fl_list, fl_frac))
    for t in sorted(by_list):
        if S < t <= E and by_list[t] == 5:
            mn = min(x[2] for x in by_frac_any[t])
            border.append((name, t, by_list[t], round(mn, 4)))
    # evaluation column numbering
L.append(f"1. rows with screen_v2_clean unmeasured: {unmeasured}")
L.append(f"   rows listing the key with fraction not > 0.10: {len(listed_under)} {listed_under[:5]}")
L.append(f"   contradiction rows (measured screen, key not listed, fraction > 0.10): {len(contra)} {contra[:5]}")
L.append(f"   (evaluation column + 1) * 131072 != timesteps: {len(eval_col_bad)} {eval_col_bad[:5]}")
L.append(f"   window ANY-flag sets by 'listed' vs by 'fraction > 0.10' differ in: {alt_flags_diff}")
L.append(f"   window evaluations at exactly 5 of 10 (borderline), with the smallest over-bar fraction among the 5:")
for b in border:
    L.append(f"     {b}")

# 2. score.txt report-only per-continuation fields against my stat_out.json
mine = json.loads((HERE / "stat_out.json").read_text())["runs"]
txt = (ECS / "score/score.txt").read_text().splitlines()
pat = re.compile(r"^  (s\d\d_[AB]_\d{4})\s+(\S+)\s+(\S+)\s+T ([\d,]+)\s+first ANY flag (.*?); flagged (\d+); ANY ALARM "
                 r"(\[.*?\]); TAIL-flagged (\[.*?\]); TAIL ALARM (\[.*?\]); TAIL-input missing \(report only\) (\d+); "
                 r"mean std first/last row .*?; mean return at T ([\d.]+);")
seen = 0
bad = []
for ln in txt:
    m = pat.match(ln)
    if not m:
        continue
    seen += 1
    nm = m.group(1)
    r = mine[nm]
    first = None if m.group(5).startswith("none") else int(m.group(5).replace(",", ""))
    got = dict(cls=m.group(2), outcome=m.group(3), T=int(m.group(4).replace(",", "")), first=first,
               n=int(m.group(6)), any_alarm=json.loads(m.group(7)), tail_flagged=json.loads(m.group(8)),
               tail_alarm=json.loads(m.group(9)), tmiss=int(m.group(10)), ret=m.group(11))
    exp = dict(cls=r["cls"], outcome=r["outcome"], T=r["T"], first=r["first_any"], n=r["n_any_flagged"],
               any_alarm=r["any_alarms"], tail_flagged=r["tail_flagged"], tail_alarm=r["tail_alarms"],
               tmiss=len(r["tail_missing_window"]), ret=f"{r['mean_return_T']:.1f}")
    for k in got:
        if got[k] != exp[k]:
            bad.append((nm, k, got[k], exp[k]))
    if first is None and f"censored at T = {r['T']:,}" not in m.group(5):
        bad.append((nm, "censored text", m.group(5)))
L.append(f"2. score.txt per-continuation lines parsed {seen}; field mismatches with mine: {bad}")

# 3. pins / status inputs
appr = json.loads((ECS / "APPROVED").read_text())
L.append(f"3. criteria sha256 {sha(ECS / 'criteria.txt')} == APPROVED {appr['criteria_sha256'] == sha(ECS / 'criteria.txt')}")
for k, v in appr["tool_sha256"].items():
    L.append(f"   {k}: {sha(ECS / 'tools' / k) == v}")
crit = (ECS / "criteria.txt").read_text()
for m in re.finditer(r"^    ((?:tools|laneA|laneB)/\S+)\s+([0-9a-f]{64})$", crit, re.M):
    p = (ECS / m.group(1)) if m.group(1).startswith("tools") else (LANE / m.group(1))
    L.append(f"   pin {m.group(1)}: {'equal' if p.exists() and sha(p) == m.group(2) else 'DIFFERS/ABSENT'}")
for m in re.finditer(r"^    (v[12]/\S+)\s+([0-9a-f]{64})$", crit, re.M):
    p = ECS / m.group(1)
    L.append(f"   pin {m.group(1)}: {'equal' if p.exists() and sha(p) == m.group(2) else 'DIFFERS/ABSENT'}")
# m5 used by the reader
for nm in sorted(mine):
    rj = json.loads((ECS / "score/read" / nm / "read.json").read_text())["r1"]
    if rj.get("m5_sha256") != "0d89bcb858e19f780c3ce0ac7a5886fc01b33d78164d9ee37d2e50dc1d4b0511" or \
            rj.get("m5_common_sha256") != "ea2c5ea6984b8dc00c1c066a4441a59526c515ceb111da17eaf1688a009226c5" or \
            rj.get("m5_argv")[-3] != str(mine[nm]["T"]) or rj.get("m5_argv")[-4] != "--E":
        L.append(f"   {nm}: M5 pin or argv unexpected {rj.get('m5_sha256')} {rj.get('m5_argv')[-4:]}")
L.append("   M5 sha256 (both files) and argv --E T checked in all 24 read.json (only exceptions printed above)")

# 4. sensitivity: what each single flip of a continuation's class would read (the line's robustness)
def hyper(na, nb, m):
    t = comb(na + nb, m)
    return {k: Fraction(comb(na, k) * comb(nb, m - k), t) for k in range(max(0, m - nb), min(na, m) + 1)}


def cmh(tabs):
    dist = {0: Fraction(1)}
    x = 0
    for a, na, b, nb in tabs:
        h = hyper(na, nb, a + b)
        nd = {}
        for s, p in dist.items():
            for k, q in h.items():
                nd[s + k] = nd.get(s + k, 0) + p * q
        dist = nd
        x += a
    return (sum((p for s, p in dist.items() if s >= x), Fraction(0)),
            sum((p for s, p in dist.items() if s <= x), Fraction(0)))


base = {"s42": [6, 6], "s80": [1, 1]}
worst = []
for st in base:
    for arm in (0, 1):
        for d in (-1, 1):
            t = {k: list(v) for k, v in base.items()}
            t[st][arm] += d
            if not 0 <= t[st][arm] <= 6:
                continue
            u, dn = cmh([(t[s][0], 6, t[s][1], 6) for s in ("s42", "s80")])
            worst.append((st, "AB"[arm], d, str(u), str(dn), u <= Fraction(1, 40) or dn <= Fraction(1, 40)))
L.append("4. single-flip sensitivity (start, arm, change, P_UP, P_DOWN, would leave line 3):")
for w in worst:
    L.append(f"     {w}")
# the smallest number of s80 flips that would read line 1 or 2 with s42 at 6/6 vs 6/6
best = None
for a in range(7):
    for b in range(7):
        u, dn = cmh([(6, 6, 6, 6), (a, 6, b, 6)])
        if u <= Fraction(1, 40) or dn <= Fraction(1, 40):
            k = abs(a - 1) + abs(b - 1)
            if best is None or k < best[0]:
                best = (k, a, b, str(u), str(dn))
L.append(f"   fewest changed s80 classes to leave line 3 (s42 fixed 6/6 vs 6/6): {best}")
(HERE / "extra_out.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
