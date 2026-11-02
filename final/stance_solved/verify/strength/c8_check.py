"""Read-only: recheck C8 ranges over the 18 PHASE manip outputs, plus the obs-6 report-only dV."""
import json
BASE = "/home/user/work/pc_study/f2/measure"
dv, mu, o6, o6w, verd = [], [], [], [], []
for r in ["s60", "s42", "s80"]:
    for c in ["4194304", "5242880", "6291456", "8388608", "10485760", "final"]:
        t = open(f"{BASE}/stance7_r10LTSBpc_{r}/manip_{c}.out").read()
        j = json.loads(t[t.index("{"):t.rindex("}") + 1])
        verd.append(j["verdict"])
        dv += list(j["checks"]["(i) phase in force"]["max_abs_dV_by_perturbation"].values())
        mu += list(j["checks"]["(ii) actor independent"]["max_abs_dmu_by_perturbation"].values())
        k = j["checks"]["(iii) critic input from observation 6"]
        o6.append(k.get("report_only_max_abs_dV_obs6_plus1"))
        o6w.append(k.get("obs6_weights_nonzero"))
print(len(verd), set(verd))
print("dV range", min(dv), max(dv))
print("dmu max", max(mu))
print("obs6 dV(+1) range", min(o6), max(o6))
print("obs6 nonzero weights", set(o6w))
