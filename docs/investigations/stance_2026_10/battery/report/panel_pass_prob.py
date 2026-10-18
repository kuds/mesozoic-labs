# Probability that one 40-seed panel reaches >=37 clean seeds (the minimum count whose
# binomial LCB clears 0.80, per verify_skeptic/vfresh.out), if seeds were i.i.d. Bernoulli(p).
# p values are the pooled point estimates and 95% CI ends in fresh_seeds/fresh_seed_summary.json.
# Ignores the separate hop_or_fall rail (<=1 fall per heading). Illustration only.
import json, math, sys
def p_ge(k, n, p):
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))
src = json.load(open(sys.argv[1]))
out = {"note": "P(>=37/40 clean) under i.i.d. Bernoulli(p); p from fresh_seed_summary.json pooled.*; ignores hop_or_fall rail; illustration, not a gate statistic",
       "min_passing_count": 37, "n": 40, "rows": {}}
for key, v in src["pooled"].items():
    row = {}
    lo, hi = v["two_sided_95_clopper_pearson"]
    for name, p in (("rate", v["rate"]), ("ci95_lo", lo), ("ci95_hi", hi)):
        row[name] = {"p": round(p, 4), "P_pass_panel": round(p_ge(37, 40, p), 4)}
    row["k_n"] = f'{v["k"]}/{v["n"]}'
    out["rows"][key] = row
json.dump(out, open("panel_pass_prob.json", "w"), indent=1)
print(json.dumps(out, indent=1))
