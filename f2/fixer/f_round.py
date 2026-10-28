import json, sys
S2, P = sys.argv[1], sys.argv[2]
d = json.load(open(f"{S2}/r10round/f2/readouts/READOUTS.json"))
cols = ["head_tip_height_mean_m","neck_pitch_mean_deg","head_pitch_mean_deg","tail_3_pitch_act_command","neck_pitch_act_command",
        "head_pitch_act_command","tail_tip_chord_mean_deg","neck_head_work_abs_j","root_height_mean_m"]
for run in ("s60","s42","s80"):
    w = d["runs"][f"stance6_r10LTSB_{run}"]["D13"]["windows"]
    for c in cols:
        vals = []
        for k in w:
            x = w[k]
            v = x.get(c) if isinstance(x, dict) else None
            if isinstance(v, dict): v = v.get("mean", v)
            vals.append(v)
        print(run, c, [round(v,5) if isinstance(v,(int,float)) else v for v in vals])
print("statue", {c: d["statue_ledger_values"].get(c) for c in cols} if isinstance(d.get("statue_ledger_values"), dict) else None)
# D-4 cells
for run, lab in (("s60","4194304"),("s42","5242880"),("s80","4194304"),("s80","final")):
    s = json.load(open(f"{P}/stance6_r10LTSB_{run}/{lab}/m2/m2_summary.json"))
    ks = [k for k in s if "chord" in k]
    print(run, lab, {k: s[k] for k in ks})
