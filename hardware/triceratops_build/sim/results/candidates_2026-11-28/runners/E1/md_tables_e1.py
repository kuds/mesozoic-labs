"""Print the Markdown tables of E1's summary from sim/results/candidates_2026-11-28/E1/e1_summary.json (no computation
beyond formatting)."""
import json
import sys
from pathlib import Path

S = json.loads(Path("/home/user/mesozoic-labs/hardware/triceratops_build/sim/results/candidates_2026-11-28/E1/"
                    "e1_summary.json").read_text())
CT = S["controllers"]


def f(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def pct(x):
    return f"{100 * x:.3f} %"


lines = []
lines.append("| Controller | Episodes | Falls | Samples | Least #9, mm (either side) | #9 < 0 | #9 < 5 mm | Least #10, mm | "
             "#10 < 0 | #10 < 5 mm |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for c in CT:
    r = S["reported"][c]["all"]
    fa = S["falls"][c]
    lines.append(f"| {c} | {r['episodes']} | {fa['fell']} | {r['samples']} | {f(r['#9']['least']['least_mm'])} | "
                 f"{pct(r['#9']['under_0_share'])} | {pct(r['#9']['under_5mm_share'])} | "
                 f"{f(r['#10']['least']['least_mm'])} | {pct(r['#10']['under_0_share'])} | "
                 f"{pct(r['#10']['under_5mm_share'])} |")
lines.append("")
lines.append("Soles aside (supporting):")
lines.append("")
lines.append("| Controller | Least #9 soles aside, mm | < 0 | Least #10 soles aside, mm | < 0 |")
lines.append("|---|---|---|---|---|")
for c in CT:
    r = S["reported"][c]["all"]
    lines.append(f"| {c} | {f(r['#9_soles_aside']['least']['least_mm'])} | {pct(r['#9_soles_aside']['under_0_share'])}"
                 f" | {f(r['#10_soles_aside']['least']['least_mm'])} | {pct(r['#10_soles_aside']['under_0_share'])} |")
lines.append("")
lines.append("Least poses:")
lines.append("")
for c in CT:
    for k in ("#9", "#10"):
        L = S["reported"][c]["all"][k]["least"]
        lines.append(f"- {c} {k}: {L['least_mm']:.2f} mm, {L['parts'][0]} / {L['parts'][1]}, episode {L['episode']} "
                     f"at {L['t_s']:.2f} s")
lines.append("")
v = S["verdict"]
lines.append("| Clash | Controllers with a pose < 0 | REACHED |")
lines.append("|---|---|---|")
for k in ("9", "10"):
    lines.append(f"| #{k} | {', '.join(v['reached'][k]['controllers_overlapping']) or 'none'} | "
                 f"{v['reached'][k]['REACHED']} |")
lines.append("")
lines.append("| Stop | Record's least, mm | " + " | ".join(f"{c}: clipped / least after, mm" for c in CT) +
             " | ACCEPTABLE |")
lines.append("|---|---|" + "---|" * len(CT) + "---|")
pc = S["per_controller_judged"]
stops = {s: None for s in v["acceptable_stops"]}
labels = {}
import csv  # noqa: E402
for row in csv.DictReader(open(Path("/home/user/mesozoic-labs/hardware/triceratops_build/sim/results/"
                                    "candidates_2026-11-28/E1/e1_stops.csv"))):
    labels[row["stop"]] = (row["label"], row["record_least_mm_full_range"])
for s, a in v["acceptable_stops"].items():
    lab, rec = labels.get(s, (s, ""))
    cells = []
    for c in CT:
        e = pc[c]["stops"][s]
        cells.append(f"{100 * e['clipped_share']:.2f} % / {f(e['least_mm_after'])}")
    lines.append(f"| {s} ({lab}) | {rec or '-'} | " + " | ".join(cells) + f" | {a['ACCEPTABLE']} |")
print("\n".join(lines))
