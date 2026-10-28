import json, sys
P = sys.argv[1]
labels = ["1048576","2097152","3145728","3670016","4194304","5242880","6291456","7340032","8388608","9437184","10485760","11534336","final"]
import os
for run in ("s60","s42","s80"):
    print("==", run)
    for lab in labels:
        f = f"{P}/stance6_r10LTSB_{run}/{lab}/m1/m1_summary.json"
        if not os.path.exists(f): continue
        d = json.load(open(f))
        hr = d["hip_roll"]
        if lab == "1048576" and run == "s60": print("hip_roll keys", {k: list(v.keys()) if isinstance(v, dict) else v for k, v in hr.items()})
        c = d["cmd_a"]
        def g(x):
            return x if not isinstance(x, dict) else x.get("mean", x)
        print(lab, "cmd11", round(g(c["11"]),4), "cmd14", round(g(c["14"]),4),
              "hr4", {k: (round(v,4) if isinstance(v,float) else v) for k, v in hr["4"].items()} if isinstance(hr["4"], dict) else hr["4"])
