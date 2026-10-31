"""PS designer's self-check of the drafts (criteria_v1.txt, SCREEN_v1.json, DESIGN_NOTES.txt) against their sources.
Read only; prints PASS / FAIL per check and a total.  Not a substitute for the independent review.
Usage: cd /tmp && PY -I -B verify_drafts.py > verify_drafts.out 2> verify_drafts.err
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

CW = Path(__file__).resolve().parent
S2 = Path(sys.executable).parents[2]
S1 = S2.parent.parent / "0fdf74a4-2989-56c9-a773-92101004d7de" / "scratchpad"
assert (S2 / "r10round/bias_screen2").is_dir() and (S1 / "foot/stance6").is_dir(), "run with S2/venv/bin/python"
CRIT = (CW / "criteria_v1.txt").read_text()
NOTES = (CW / "DESIGN_NOTES.txt").read_text()
SCREEN = json.loads((CW / "SCREEN_v1.json").read_text())
POWER = (CW / "power_ps.out").read_text()
OTHER = (CW / "other_actuators_bs2.out").read_text()
RUNLOG = (S1 / "foot/stance4/live/run_log.txt").read_text().split("\n")
SCORE = (S2 / "r10round/bias_screen2/score/score.txt").read_text().split("\n")
PR655 = Path("/home/user/work/wt_pr655/docs/investigations/TREX_STANCE_POP_POSE_SATURATION_2026_10.md").read_text().split(
    "\n")
CODE = S2 / "r10round/bias_screen/code"
BASE = (CODE / "environments/shared/base_env.py").read_text().split("\n")
TREX = (CODE / "environments/tyrannosaurus_rex/base_env.py").read_text().split("\n")
PROBE = (CODE / "environments/shared/policy_probe.py").read_text().split("\n")
STANCE = (CODE / "configs/tyrannosaurus_rex/anatomical/stance.toml").read_text().split("\n")
WIDEN = Path("/home/user/work/repo/environments/shared/scripts/widen_checkpoint.py").read_text().split("\n")
PNOTES = (S2 / "com_revisit/notes/phase_probe/NOTES.txt").read_text().split("\n")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))


def line_has(lines, n, text):
    return text in lines[n - 1]


# 1. line texts and additions present exactly once each
for key in ('"NOT INFORMATIVE: THE CONTROL ARM DID NOT REPRODUCE THE CLIMB OF THE ON-POLICY TAIL_3 COMMAND',
            '"REDUCED: GIVING THE CRITIC THE EPISODE PHASE (the tail_3 angle still in its input;',
            '"NO REDUCTION DETECTED IN THE ON-POLICY TAIL_3 COMMAND OVER THE LAST 8 UPDATES BEFORE E',
            '"(the filled p is <= 0.05, but the control', '"(the as-counted p, report only, is <= 0.05; the filled p'):
    check(f"line text present once: {key[:50]}...", CRIT.count(key) == 1, CRIT.count(key))
# 2. SCREEN queue = criteria queue
q = [tuple(x) for x in SCREEN["queue"]]
order = re.search(r"in this launch order:\n\s+(control 2901.*?)\.\n", CRIT).group(1)
crit_q = [("control" if a == "control" else "critic_phase", int(s)) for a, s in re.findall(r"(control|phase) (\d+)", order)]
check("SCREEN_v1.json queue equals the criteria's launch order", q == crit_q, f"{q} vs {crit_q}")
check("SCREEN schema and intervention", SCREEN["schema"] == "bs-screen/1" and SCREEN["intervention"] == "critic_phase")
seeds = sorted({s for _, s in q})
derived = ", ".join(f"{s}-{s + 3} + {s + 1000}" for s in seeds)
check("derived seeds listed as in bs_common.derived_seeds", derived in CRIT, derived)
# 3. B_S exact
check("B_S = 135930855/2^30", Fraction(0.1265954738482833) == Fraction(135930855, 2 ** 30)
      and "0.1265954738482833 (exactly 135930855/1073741824)" in CRIT)
check("B_S re-measured equal (other_actuators_bs2.out)", "np.float64(0.1265954738482833)" in OTHER)
# 4. power numbers cited appear in power_ps.out
for tok in ("0.955/0.044", "0.509/0.490", "0.201/0.798", "0.991/0.007", "0.733/0.265", "0.341/0.657", "0.992/0.008",
            "0.657/0.343", "0.260/0.740", "0.712/0.265", "0.291/0.686", "0.127/0.850", "P2 REDUCED 0.8260",
            "P2 REDUCED 0.2510", "P(informative) = 0.9984", "P(informative) = 0.9766", "P(informative) = 0.6553",
            "P(p <= 0.05) 0.0602", "P(p <= 0.05) 0.0610", "given informative 0.0722", "= 0.927", "= 0.460",
            "P(SUPPORTED) = 0.4377", "= 19/70", "exact p = 2/35", "f1.00 0.001/0.991", "f1.00 0.037/0.823",
            "f1.00 0.166/0.583", "f0.50 0.001/0.305", "f0.50 0.166/0.198", "P(not informative) = 5/16"):
    check(f"power_ps.out has {tok}", tok in POWER)
# 5. BS2 score.txt lines cited
for n, tok in ((87, "+0.02143"), (93, "+0.06177"), (117, "+0.05324"), (141, "+0.02260"), (165, "+0.04147"),
               (181, "+0.1519 / +0.1200"), (184, "+0.2412"), (190, "+0.5904"), (18, "SD 0.16577"),
               (19, "SD 0.03586"), (47, "SUPPORTED"), (60, "SUPPORTED")):
    check(f"BS2 score.txt line {n} has {tok}", line_has(SCORE, n, tok), SCORE[n - 1][:100])
# 6. RUNLOG lines cited
for n, tok in ((788, "Launch with 0.20"), (789, "12:54:36"), (793, "15:43:28"), (794, "15:56:28"), (796, "BS2 VERIFIED"),
               (803, "TimeLimit.truncated"), (804, "WHY-PROBE"), (805, "Test the phase fix first"), (808, "RESUMED"),
               (809, "+0.31")):
    check(f"RUNLOG line {n} has {tok}", line_has(RUNLOG, n, tok), RUNLOG[n - 1][:100])
# 7. #655 lines cited
for n, tok in ((183, "155"), (229, "155"), (804, "stance_width"), (806, "AUC 0.71"), (808, "Unobserved action"),
               (864, "Widening precedent"), (873, "Constraints any phase context")):
    check(f"#655 line {n} has {tok}", line_has(PR655, n, tok), PR655[n - 1][:100])
# 8. code lines cited
for lines, n, tok, nm in ((BASE, 1335, "self._step_count += 1", "shared/base_env"),
                          (BASE, 1349, "obs = self._get_obs()", "shared/base_env"),
                          (BASE, 1352, "self._get_reward_info(action)", "shared/base_env"),
                          (TREX, 1270, "_step_count == self.stance_width_settle_steps", "trex/base_env"),
                          (TREX, 1278, "_step_count <= self.stance_width_settle_steps", "trex/base_env"),
                          (TREX, 1315, "_step_count > self.stance_width_settle_steps", "trex/base_env"),
                          (TREX, 1339, "_step_count > self.stance_width_settle_steps", "trex/base_env"),
                          (TREX, 1421, "_step_count > self.stance_width_settle_steps", "trex/base_env"),
                          (TREX, 974, "def _get_obs", "trex/base_env"),
                          (STANCE, 212, "stance_width_settle_steps = 200", "stance.toml"),
                          (STANCE, 400, "max_episode_steps = 1000", "stance.toml"),
                          (PROBE, 77, '("statue_settled", 3042, 0.0, 199)', "policy_probe"),
                          (PROBE, 252, "for step in range(max(wanted) + 1):", "policy_probe"),
                          (WIDEN, 172, "_MOMENT_KEYS", "widen_checkpoint"),
                          (WIDEN, 451, "def _insert_zero_columns", "widen_checkpoint"),
                          (WIDEN, 512, "for member, prefix in _OPTIMIZER_MEMBERS", "widen_checkpoint"),
                          (PNOTES, 19, "0.17-0.23", "phase_probe NOTES"),
                          (PNOTES, 56, "tail_2", "phase_probe NOTES")):
    check(f"{nm} line {n} has {tok}", line_has(lines, n, tok), lines[n - 1][:100])
# 9. format
for nm, txt in (("criteria_v1.txt", CRIT), ("DESIGN_NOTES.txt", NOTES)):
    long_lines = [i + 1 for i, l in enumerate(txt.split("\n")) if len(l) > 120]
    check(f"{nm}: no line over 120 columns", not long_lines, long_lines[:5])
    check(f"{nm}: ASCII only", all(ord(c) < 128 for c in txt))
print(f"\n{sum(results)} of {len(results)} checks pass")
