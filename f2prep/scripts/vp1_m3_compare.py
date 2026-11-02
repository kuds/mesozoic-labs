"""VP1 M3 comparison (report only): the certificate (stance_heading_report.json, every leaf) and the selected panel CSV
of (a) the HEAD copy's M3 on STANCE6 s80's final F2 pair vs the round's panel (8737cd3d copy), and (b) the HEAD copy's
M3 on the zero-phase-column widened pair (saved as a zip by HEAD's own save path, review/fixer/vp2d1) vs (a).
Leaves and CSV columns that name the checkpoint file itself (its name and sha256: the widened zip is another file)
are listed apart as IDENTITY (v2: the CSV's checkpoint_sha256 column too; v1 compared the CSV bytes only); any other differing leaf makes the pair DIFFERENT.  Reads only; prints.

    PY -I -B vp1_m3_compare.py F2PREP
"""
import hashlib
import json
import re
import sys
from pathlib import Path

S1 = Path(re.search(r'^S1 = Path\("(.*)"\)$', Path("/home/user/work/pc_study/tools/pc/pc_common.py").read_text(),
                    re.M).group(1))
ROUND = S1 / "foot/stance6/panels/stance6_r10LTSB_s80/final/m3"
F = Path(sys.argv[1])
IDENTITY = {"/handoff/checkpoint", "/handoff/checkpoint_sha256", "/policy"}
CSV_IDENTITY = {"checkpoint_sha256"}          # v2: the panel CSV's column naming the certified file itself


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        yield (f"{p}#len", len(o))
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield (p, o)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmp(a, b, label):
    ra = json.loads((a / "stance_heading_report.json").read_text())
    rb = json.loads((b / "stance_heading_report.json").read_text())
    la, lb = dict(leaves(ra)), dict(leaves(rb))
    keys = sorted(set(la) | set(lb))
    diff = [k for k in keys if la.get(k, "<absent>") != lb.get(k, "<absent>")]
    ident = [k for k in diff if k in IDENTITY]
    other = [k for k in diff if k not in IDENTITY]
    print(f"{label}")
    print(f"  report leaves {len(la)} vs {len(lb)}; differing {len(diff)}: identity {ident}; other {other[:20]}")
    for k in ident:
        print(f"    IDENTITY {k}: {str(la.get(k))[:90]} | {str(lb.get(k))[:90]}")
    pa, pb = a / "stance_heading_panel_selected.csv", b / "stance_heading_panel_selected.csv"
    import csv
    ca, cb = list(csv.DictReader(open(pa, newline=""))), list(csv.DictReader(open(pb, newline="")))
    same_shape = len(ca) == len(cb) and (ca[0].keys() if ca else []) == (cb[0].keys() if cb else [])
    cols = sorted({k for x, y in zip(ca, cb) for k in x if x[k] != y.get(k)})
    other_cols = [c for c in cols if c not in CSV_IDENTITY]
    csv_eq = same_shape and not other_cols
    print(f"  stance_heading_panel_selected.csv sha256 {sha(pa)[:16]}... vs {sha(pb)[:16]}... (bytes equal "
          f"{sha(pa) == sha(pb)}); rows {len(ca)} vs {len(cb)}, columns {len(ca[0]) if ca else 0}; differing columns "
          f"{cols} (identity {[c for c in cols if c in CSV_IDENTITY]}); every other cell equal: {csv_eq}")
    print(f"  passed {ra.get('passed')} vs {rb.get('passed')}; failures {ra.get('failures')} vs {rb.get('failures')}; "
          f"clean_seed_lcb {ra['result'].get('clean_seed_lcb')} vs {rb['result'].get('clean_seed_lcb')}; "
          f"n_clean_seeds {ra['result'].get('n_clean_seeds')} vs {rb['result'].get('n_clean_seeds')}")
    sa = json.loads((a / "m3_summary.json").read_text())
    sb = json.loads((b / "m3_summary.json").read_text())
    print(f"  m3_summary verdict {sa.get('verdict')} vs {sb.get('verdict')}; headings equal {sa.get('headings') == sb.get('headings')}; "
          f"copy commits {sa['copy_seal']['commit'][:12]} vs {sb['copy_seal']['commit'][:12]}")
    ok = not other and csv_eq and sa.get("headings") == sb.get("headings") and sa.get("verdict") == sb.get("verdict")
    print(f"  {label}: {'CERTIFICATE EQUAL' if ok else 'DIFFERENT'}")
    return ok


r1 = cmp(ROUND, F / "vp1/m3_stock", "VP1 (b) M3 round panel (8737cd3d copy) vs HEAD copy, stock s80 final")
r2 = cmp(F / "vp1/m3_stock", F / "vp1/m3_widened", "VP1 (d) M3 HEAD copy stock vs widened zip (zero phase columns)")
print(f"VP1 M3 parts: {'PASS' if r1 and r2 else 'FAIL'}")
