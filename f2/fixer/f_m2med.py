import json, sys, statistics
P = sys.argv[1]
d = json.load(open(f"{P}/stance6_r10LTSB_s80/6291456/m2/panel.json"))
om = [e["gate_max_actuator_saturation_fraction"] for e in d["episodes"]]
print("s80 6.29M M2 gate_max_sat median over 40:", statistics.median(om), "max", max(om), "n>0.1", sum(x>0.1 for x in om))
s = d["summary"]
print({k: s[k] for k in s if any(w in k.lower() for w in ("heading","seed","yaw","task"))})
s42 = json.load(open(f"{P}/stance6_r10LTSB_s42/final/m2/panel.json"))["summary"]
print({k: s42[k] for k in s42 if any(w in k.lower() for w in ("heading","seed","yaw","task"))})
