import json, re
from fractions import Fraction
BS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen"
crit = open(BS + "/criteria.txt").read()
norm = re.sub(r"\s+", " ", crit)
sj = json.load(open(BS + "/score/score.json"))
txt = open(BS + "/score/score.txt").read()
lt = sj["line_text"]
# the criteria's quoted line-1 text
m = re.search(r'1\. NOT INFORMATIVE: "(.*?)"', norm)
q1 = m.group(1)
print("criteria line-1 quote:", q1)
print("score.json line_text == criteria quote:", lt == q1)
print("score.txt contains line_text verbatim:", ("  " + lt + "\n") in txt)
print("additions:", sj.get("additions"), "; p_filled", sj["p_filled"]["p"], "<= 1/20?", Fraction(sj["p_filled"]["p"]) <= Fraction(1, 20))
print("line-1 addition required? (filled p <= 0.05):", Fraction(sj["p_filled"]["p"]) <= Fraction(1, 20))
print("BASELINE_S float == exact -135608119/2**32:", Fraction(-0.03157372563146055) == Fraction(-135608119, 2**32))
for k in ("status", "provisional", "line", "line_key"):
    print(k, json.dumps(sj.get(k)))
print([k for k in sj.keys()])
