"""Read-only: per PHASE final, the 0-deg cell's per-episode criteria (fails, median, worst) from the M3 report text,
and the per-heading clean counts."""
import re
P = "/home/user/work/pc_study/round/panels/stance7_r10LTSBpc_%s/final/m3/stance_heading_report.txt"
pat = re.compile(r"^\s+(\w+)\s+(\S+ \S+|reach \d+ steps)\s+fails\s+(\d+)/(\d+)(?:\s+median (\S+), worst (\S+))?")
for s in ("s60", "s42", "s80"):
    lines = open(P % s).read().splitlines()
    heads = [l.split() for l in lines[14:20]]
    print(s, "headings", [(h[0], h[1], h[-1]) for h in heads], "|", lines[9].split("   ")[0].strip(), "|", lines[-1])
    for i, l in enumerate(lines, 1):
        m = pat.match(l)
        if m:
            print("   L%d %s %s fails %s/%s worst %s" % (i, m.group(1), m.group(2), m.group(3), m.group(4), m.group(6)))
