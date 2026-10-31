"""PS statistic check, extras (own code, reads only): (a) the fill rule's bounds on the observed primaries with own
permutation code (criteria THE FILL: one MISSING >= 36/70, one per arm >= 51/70, two controls >= 56/70); (b) the guard
SEC-TAIL2 against the parent's tail_2 at S (rises per continuation) and which split(s) tie or beat the observed one;
(c) for each tested quantity, how many of the 70 splits reach the observed statistic."""
import itertools
import json
import sys
from fractions import Fraction

d = json.load(open(sys.argv[1]))
runs = d["runs"]
names = ["control_2901", "control_2911", "control_2921", "control_2931",
         "critic_phase_2901", "critic_phase_2911", "critic_phase_2921", "critic_phase_2931"]


def stat(hi, lo):
    return (Fraction(sum(v[0] for v in hi), len(hi)) - Fraction(sum(v[0] for v in lo), len(lo)),
            sum((v[1] for v in hi), Fraction(0)) / len(hi) - sum((v[1] for v in lo), Fraction(0)) / len(lo))


def perm(hi, lo):
    pool = hi + lo
    obs = stat(hi, lo)
    c = 0
    beat = []
    for sub in itertools.combinations(range(8), 4):
        st = stat([pool[i] for i in sub], [pool[i] for i in range(8) if i not in sub])
        if st >= obs:
            c += 1
            beat.append(sub)
    return Fraction(c, 70), beat


prim = {n: Fraction(runs[n]["primary"]) for n in names}
c = [(0, prim[n]) for n in names[:4]]
p = [(0, prim[n]) for n in names[4:]]
print("(a) fill bounds on the observed primaries (control MISSING -> -inf, phase MISSING -> +inf):")
for label, cc, pp in (
        ("none missing", c, p),
        ("control_2901 MISSING", [(-1, Fraction(0))] + c[1:], p),
        ("control_2931 MISSING (the highest phase-side gap)", c[:3] + [(-1, Fraction(0))], p),
        ("critic_phase_2921 MISSING", c, p[:2] + [(1, Fraction(0))] + p[3:]),
        ("one MISSING in each arm", [(-1, Fraction(0))] + c[1:], [(1, Fraction(0))] + p[1:]),
        ("two controls MISSING", [(-1, Fraction(0))] * 2 + c[2:], p)):
    pv, _ = perm(cc, pp)
    print(f"  {label}: p {pv.numerator * (70 // pv.denominator) if 70 % pv.denominator == 0 else pv} / 70 = {float(pv):.4f}")

print("\n(b) the guard SEC-TAIL2 (tail_2 command, on-policy, the primary's 8 rows):")
import numpy as np
F2P = ("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2/"
       "stance6_r10LTSB_s42/copies/01_stance/policy_probe.npz.copy1")
with np.load(F2P, allow_pickle=False) as z:
    ut = np.asarray(z["update_timesteps"]).astype(np.int64)
    mu = np.asarray(z["rollout_mu_mean"])
rows = [int(np.flatnonzero(ut == 2_097_152 - 8192 * k)[0]) for k in range(7, -1, -1)]
parent_t2 = sum((Fraction(float(mu[i, 13])) for i in rows), Fraction(0)) / 8
print(f"  the parent's tail_2 at S (rows {rows[0]}-{rows[-1]}): {float(parent_t2):+.5f}")
for n in names:
    v = Fraction(runs[n]['tail2'])
    print(f"  {n:<18} {float(v):+.5f}  rise from the parent's S value {float(v - parent_t2):+.5f}")
t2 = {n: Fraction(runs[n]["tail2"]) for n in names}
hi = [(0, t2[n]) for n in names[4:]]
lo = [(0, t2[n]) for n in names[:4]]
pv, beat = perm(hi, lo)
pool = names[4:] + names[:4]
print(f"  p {pv} ({len(beat)}/70); splits at or above the observed (as the 'phase' arm): "
      f"{[[pool[i] for i in b] for b in beat]}")
print(f"  lowest phase {float(min(t2[n] for n in names[4:])):+.5f}; highest control {float(max(t2[n] for n in names[:4])):+.5f}")

print("\n(c) the observed split's rank among the 70 for each tested quantity (count of splits >= observed):")
for key, getter, higher in (("primary", lambda n: Fraction(runs[n]["primary"]), "control"),
                            ("SEC-CMD", lambda n: Fraction(runs[n]["markers_cmd_l"]), "control"),
                            ("SEC-MARGIN", lambda n: Fraction(runs[n]["b_star"]), "critic_phase"),
                            ("SEC-MECH", lambda n: Fraction(runs[n]["slope_eff"]), "control"),
                            ("SEC-TAIL2", lambda n: Fraction(runs[n]["tail2"]), "critic_phase")):
    h = [(0, getter(n)) for n in names if n.startswith(higher)]
    o = [(0, getter(n)) for n in names if not n.startswith(higher)]
    pv, beat = perm(h, o)
    print(f"  {key}: {len(beat)}/70; complete separation {min(v[1] for v in h) > max(v[1] for v in o)}")
