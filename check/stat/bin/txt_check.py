"""PS statistic check: every number in score.txt's reading, primary, gate, test, paired and secondaries sections against
this checker's own values (out/stat_check.json, out/spread.json), at the printed precision. Own code, reads only."""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
mine = json.load(open(sys.argv[1]))
txt = (PS / "score/score.txt").read_text()
L = txt.splitlines()
m = mine["mine"]
B = Fraction(m["B_S"])
prim = {k: Fraction(v) for k, v in m["primaries"].items()}
names = ["control_2901", "critic_phase_2901", "control_2911", "critic_phase_2911", "control_2921",
         "critic_phase_2921", "control_2931", "critic_phase_2931"]
n = [0, 0]
bad = []


def want(label, pattern_text):
    n[0] += 1
    if pattern_text in txt:
        return True
    n[1] += 1
    bad.append(f"{label}: '{pattern_text}' not in score.txt")
    print(f"  [DIFF] {label}: expected text '{pattern_text}'")
    return False


def f5(x):
    return f"{float(x):+.5f}"


want("status", "STATUS: final")
want("line header", "THE READING (line 2, REDUCED):")
crit = (PS / "criteria.txt").read_text()
mt = re.search(r"\n  2\. REDUCED[^\"]*\"(.*?)\"", crit, re.S)
want("line 2 text verbatim", "  " + re.sub(r"\s+", " ", mt.group(1)).strip())
want("B_S", f"B_S = {f5(B)}")
for nm in names:
    want(f"{nm} primary", f"{nm:<24} COUNTED  {f5(prim[nm])}  (rise from B_S {f5(prim[nm] - B)})")
ctrl = [prim[x] for x in names if x.startswith("control")]
ph = [prim[x] for x in names if x.startswith("critic")]
sd = mine["mine"]["sd"]
mc, mp = sum(ctrl, Fraction(0)) / 4, sum(ph, Fraction(0)) / 4
medc = Fraction(m["control_median"])
sp = sorted(ph)
medp = (sp[1] + sp[2]) / 2
want("control arm", f"control                  n counted 4, missing 0; mean {f5(mc)}, median {f5(medc)}, SD {sd['control']:.5f}")
want("phase arm", f"critic_phase             n counted 4, missing 0; mean {f5(mp)}, median {f5(medp)}, SD {sd['critic_phase']:.5f}")
want("diff of means", f"as counted: {f5(mc - mp)}")
par_rise = Fraction(0.7053788751363754) - B
want("informativeness", f"control median {f5(medc)} over 4 read, rise {f5(Fraction(m['control_rise']))} -> informative "
                        f"(the parent's own rise over the window {f5(par_rise)}")
want("p filled", "p (filled, decides) = 1/70 (0.0143)  <= 0.05")
want("p as counted", "p as counted (report only) = 1/70 (0.0143)")
for s, d in m["paired"]:
    c_, p_ = prim[f"control_{s}"], prim[f"critic_phase_{s}"]
    want(f"pair {s}", f"seed {s}: control {f5(c_)}  phase {f5(p_)}  diff {f5(Fraction(d))}")
want("sign flip", "= 1/16 (0.0625)")
# secondaries
sec = mine["secondaries"]
ps = {str(Fraction(k, 70)): f"{k}/70 ({k / 70:.4f})" for k in range(1, 71)}
for key, s in sec.items():
    block = txt.split(f"  {key} (")[1].split("\n  SEC-")[0]
    n[0] += 1
    v = s["verdict"] + (" " + " ".join(s["notes"]) if s["notes"] else "")
    if v not in block:
        n[1] += 1
        bad.append(f"{key} verdict")
    for nm in names:
        n[0] += 1
        if f"{nm:<24} {s['values'][nm]:+.5f}" not in block:
            n[1] += 1
            bad.append(f"{key} {nm} value")
    vals_c = [Fraction(s["values"][x]) for x in names if x.startswith("control")]
    vals_p = [Fraction(s["values"][x]) for x in names if x.startswith("critic")]

    def med(v):
        v = sorted(v)
        return (v[1] + v[2]) / 2
    for arm, vv in (("control", vals_c), ("critic_phase", vals_p)):
        n[0] += 1
        t = f"{arm:<24} n read 4, missing 0; mean {f5(sum(vv, Fraction(0)) / 4)}, median {f5(med(vv))}"
        if t not in block:
            n[1] += 1
            bad.append(f"{key} {arm} mean/median: {t}")
    n[0] += 1
    t = f"p (filled, decides SUPPORTED) = {ps[s['p_filled']]}; p as counted (report only) = {ps[s['p_as_counted']]}"
    if t not in block:
        n[1] += 1
        bad.append(f"{key} p line")
    print(f"  {key}: verdict, 8 values, arm means/medians, p line checked in score.txt")
# report-only slope table and counts
for nm in names:
    r = mine["runs"][nm]
    se, ss = r["slope_eff"], r["slope_S"]
    want(f"{nm} slope row", f"{nm:<24} S {ss:+.5f}  E {se:+.5f}  E - S {se - ss:+.5f}  E <= S {se <= ss}  E <= 0 {se <= 0}")
want("slope counts", "counts: control E <= S 0 of 4, E <= 0 0 of 4; critic_phase E <= S 4 of 4, E <= 0 2 of 4")
want("G count", "control 4 of 4; critic_phase 0 of 4")
want("missing none", "MISSING (STOPPED, LOST or primary unreadable; filled against REDUCED):\n  none")
print(f"score.txt checks: {n[0]} made, {n[1]} differing; {bad or 'no differences'}")
