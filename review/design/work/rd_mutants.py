"""Design reviewer: mutate the text-to-code mapping of score_bs.py and see whether the independent scorer (rd_compare.py,
random synthetic roots) detects each mutant.  Copies go to <this folder>/mutants/ and are removed at the end.

    PY -I -B rd_mutants.py <n_roots> <seed>
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/"
           "bias_screen2/tools/score_bs.py")
PY = sys.executable
MUT = [
    ("gate 0.15", "RISE_MIN = Fraction(1, 5)", "RISE_MIN = Fraction(3, 20)"),
    ("gate 0.25", "RISE_MIN = Fraction(1, 5)", "RISE_MIN = Fraction(1, 4)"),
    ("gate strict >", "informative = rise is not None and rise >= RISE_MIN",
     "informative = rise is not None and rise > RISE_MIN"),
    ("median -> upper middle", "return statistics.median(values) if values else None",
     "return statistics.median_high(values) if values else None"),
    ("median -> mean", "return statistics.median(values) if values else None",
     "return statistics.mean(values) if values else None"),
    ("alpha strict <", "    elif filled[0] <= ALPHA:\n        line = 2", "    elif filled[0] < ALPHA:\n        line = 2"),
    ("alpha 4/70 allowed", "ALPHA = Fraction(1, 20)", "ALPHA = Fraction(4, 70)"),
    ("ties strict", "count = sum(1 for s in sums.values() if s >= obs)",
     "count = sum(1 for s in sums.values() if s > obs)"),
    ("decide on as-counted p", "    elif filled[0] <= ALPHA:\n        line = 2",
     "    elif (as_counted or filled)[0] <= ALPHA:\n        line = 2"),
    ("primary fill reversed", "vals[c[\"arm\"]].append(INF_CONTROL if c[\"arm\"] == CONTROL else INF_BLIND)",
     "vals[c[\"arm\"]].append(INF_BLIND if c[\"arm\"] == CONTROL else INF_CONTROL)"),
    ("secondary fill reversed", "vals[c[\"arm\"]].append(NEG_INF if c[\"arm\"] == higher else POS_INF)",
     "vals[c[\"arm\"]].append(POS_INF if c[\"arm\"] == higher else NEG_INF)"),
    ("SEC-CMD direction flipped", "{\"key\": \"SEC-CMD\", \"higher\": CONTROL,",
     "{\"key\": \"SEC-CMD\", \"higher\": INTERVENTION,"),
    ("SEC-G direction flipped", "{\"key\": \"SEC-G\", \"higher\": INTERVENTION,", "{\"key\": \"SEC-G\", \"higher\": CONTROL,"),
    ("SEC-MARGIN reads G", "field = \"G\" if key == \"SEC-G\" else \"b_star\"", "field = \"G\""),
    ("SEC-G/MARGIN reads the last loop row", "        rows = loop_rows_at(m, C.E)\n        if len(rows) != 1:",
     "        rows = (m.get(\"loop\") or [])[-1:]\n        if len(rows) != 1:"),
    ("secondary read for UNREADABLE", "    if primary_cause is not None:\n        return None, f\"UNREADABLE",
     "    if False:\n        return None, f\"UNREADABLE"),
    ("Bonferroni at alpha", "if supported and filled[0] <= ALPHA / N_SECONDARIES:", "if supported:"),
    ("Bonferroni at 2/70", "if supported and filled[0] <= ALPHA / N_SECONDARIES:",
     "if supported and filled[0] <= Fraction(2, 70):"),
    ("note under line 1 dropped", "    if line == 1:\n        notes.append(SEC_UNDER_LINE1)",
     "    if False:\n        notes.append(SEC_UNDER_LINE1)"),
    ("line-1 addition on as-counted", "    if line == 1 and filled[0] <= ALPHA:",
     "    if line == 1 and as_counted is not None and as_counted[0] <= ALPHA:"),
    ("line-3 addition dropped", "    if line == 3 and as_counted is not None and as_counted[0] <= ALPHA:",
     "    if False:"),
    ("primary rows shifted by one", "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K, 0, -1))",
     "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * (k + 1) for k in range(PRIMARY_K, 0, -1))"),
    ("B_S = parent first 8 window rows", "BASELINE_S = 0.1265954738482833", "BASELINE_S = 0.12512995954602957"),
    ("STOPPED counted", "    if c[\"cls\"] != COUNTED:\n        return None, c.get(\"cause\") or c[\"cls\"]\n    v, cause",
     "    if c[\"cls\"] not in (COUNTED, STOPPED):\n        return None, c.get(\"cause\") or c[\"cls\"]\n    v, cause"),
    ("SEC-CMD reads cmd_l_max", "        v, what = e.get(\"cmd_l\"), \"CMD_L at E\"",
     "        v, what = e.get(\"cmd_l_max\"), \"CMD_L at E\""),
]


def main():
    n, seed = sys.argv[1], sys.argv[2]
    src = SRC.read_text()
    out_dir = HERE / "mutants"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for i, (label, old, new) in enumerate(MUT):
        if src.count(old) != 1:
            rows.append((label, f"PATTERN COUNT {src.count(old)}"))
            continue
        p = out_dir / f"m{i:02d}" / "score_bs.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text(src.replace(old, new))
        r = subprocess.run([PY, "-I", "-B", str(HERE / "rd_compare.py"), n, seed, str(p)], capture_output=True,
                           text=True, cwd="/tmp", timeout=3000)
        detail = (r.stdout.strip().splitlines() or [""])
        bad = [x for x in detail if x.startswith("DISAGREE")]
        rows.append((label, ("DETECTED " + bad[0][:160]) if r.returncode == 1 else
                     ("SURVIVED" if r.returncode == 0 else f"ERROR rc {r.returncode} {r.stderr[-200:]}")))
    shutil.rmtree(out_dir)
    for label, verdict in rows:
        print(f"{label:<40} {verdict}")
    print("detected", sum(v.startswith("DETECTED") for _, v in rows), "of", len(rows))


if __name__ == "__main__":
    main()
