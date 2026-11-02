"""Read-only: range of the C8 manipulation-check readings over every PHASE checkpoint (manip_*.out files)."""
import json

BASE = "/home/user/work/pc_study/f2/measure"
lo, hi, mu, n = 9.0, 0.0, 0.0, 0
for r in ["s60", "s42", "s80"]:
    for c in ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]:
        t = open(f"{BASE}/stance7_r10LTSBpc_{r}/manip_{c}.out").read()
        j = json.loads(t[t.index("{"):t.rindex("}") + 1])
        d = j["checks"]["(i) phase in force"]["max_abs_dV_by_perturbation"]
        lo, hi = min(lo, min(d.values())), max(hi, max(d.values()))
        mu = max(mu, max(j["checks"]["(ii) actor independent"]["max_abs_dmu_by_perturbation"].values()))
        assert j["verdict"] == "IN FORCE" and j["checks"]["(iii) critic input from observation 6"]["holds"]
        n += 1
print(f"{n} checkpoints IN FORCE; critic |dV| under phase perturbation {lo:.3f}-{hi:.3f}; actor max |dmu| {mu}")
