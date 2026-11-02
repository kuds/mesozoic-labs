"""Read-only: per PHASE run, the notebook's gate_verdict.json (verdict, checkpoint, passed) and the in-run K3
panel (report only, not the K3 of record) fall counts per test at each panel step, from the F2 copies."""
import csv, json
B = "/home/user/work/pc_study/round/fetch/F2/stance7_r10LTSBpc_%s/copies/01_stance/"
for s in ("s60", "s42", "s80"):
    d = B % s
    gv = json.load(open(d + "gate_verdict.json.copy1"))
    keys = {k: gv.get(k) for k in ("passed", "verdict", "status", "checkpoint", "checkpoint_sha256", "failures")}
    print(s, "gate_verdict", json.dumps(keys))
    rows = list(csv.DictReader(open(d + "k3_panel.csv.copy1", newline="")))
    agg = {}
    for r in rows:
        k = (int(r["timesteps"]), r["kind"], r["test"])
        a = agg.setdefault(k, [0, 0])
        a[0] += int(r["fall"]); a[1] += 1
    for k in sorted(agg):
        if k[0] in (6291456, 12001280) or k[1] != "grid":
            print("   K3 panel", k, "falls %d of %d" % tuple(agg[k]))
    worst = {}
    for (t, kind, test), (f, n) in agg.items():
        worst.setdefault(test, []).append((f, t))
    for test, v in worst.items():
        v.sort(reverse=True)
        print("   max falls over all panels", test, v[:3])
