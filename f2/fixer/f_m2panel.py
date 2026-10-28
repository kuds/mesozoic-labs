import json, sys, statistics
P = sys.argv[1]
def load(p):
    with open(p) as f: return json.load(f)
for run, label in (("s42","final"),("s80","6291456")):
    d = load(f"{P}/stance6_r10LTSB_{run}/{label}/m2/panel.json")
    eps = d["episodes"]
    print("==", run, label, "n", len(eps))
    over = []
    for e in eps:
        if (not e["gate_clean"]) or e["seed"] in (3079, 3080):
            print("  seed", e["seed"], "clean", e["gate_clean"], "rails", e["gate_failed_rails"], "gate_max_sat", e["gate_max_actuator_saturation_fraction"],
                  "tail3_frac99", e["tail3_frac99"], "tail3_frac99_abs", e["tail3_frac99_abs"], "tail3_cmd", round(e["tail3_cmd_mean"],4),
                  "others_max_sat", e["others_max_sat"], "act", e["others_max_sat_actuator"], "all_max_sat", e["all_max_sat"])
        if e["others_max_sat"] > 0.1: over.append(e)
    om = [e["others_max_sat"] for e in eps]
    acts = sorted(set(e["others_max_sat_actuator"] for e in over))
    print("  others_max_sat >0.1:", len(over), "actuators", acts, "mean all 40", round(statistics.mean(om),5),
          "median over", statistics.median([e["others_max_sat"] for e in over]) if over else None, "max", max(om))
    print("  tail3_frac99_abs max", max(e["tail3_frac99_abs"] for e in eps))
    s = load(f"{P}/stance6_r10LTSB_{run}/{label}/m2/m2_summary.json")
    print("  summary others_max_sat:", {k:v for k,v in s.items() if 'others' in k} if isinstance(s, dict) else None)
