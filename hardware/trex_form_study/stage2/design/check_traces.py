"""Compare regenerated eval traces (scratch) with the sha256 each committed episode record holds."""
import hashlib, json, sys
from pathlib import Path
scratch, committed, body = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
tot = same = diff = missing = 0
bad = []
for rec_path in sorted((committed / body / "eval").glob("*/*/*.json")) + sorted((committed / body / "eval").glob("*/*/*/*.json")):
    try:
        doc = json.loads(rec_path.read_text())
    except Exception:
        continue
    eps = doc.get("episodes")
    if not isinstance(eps, list):
        continue
    for e in eps:
        tr = e.get("trace") if isinstance(e, dict) else None
        if not tr or "sha256" not in tr:
            continue
        p = scratch / body / "eval" / "traces" / tr["path"] if not str(tr["path"]).startswith(body) else scratch / tr["path"]
        cands = [scratch / body / "eval" / "traces" / tr["path"], scratch / body / "eval" / tr["path"], scratch / body / tr["path"], scratch / tr["path"]]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            missing += 1
            continue
        tot += 1
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h == tr["sha256"]:
            same += 1
        else:
            diff += 1
            bad.append((str(rec_path.relative_to(committed)), e.get("seed"), tr["path"]))
print(f"compared {tot}: identical {same}, different {diff}; committed traces with no regenerated file: {missing}")
for b in bad[:20]:
    print("DIFF", b)
