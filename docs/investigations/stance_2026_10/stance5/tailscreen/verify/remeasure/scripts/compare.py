import csv, json, hashlib, sys
from pathlib import Path
S = Path("scratchpad2/tailround/screen")
V = S / "verify" / "remeasure"
rows = {(r["name"], r["label"]): r for r in csv.DictReader(open(S / "panels.csv"))}
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def strip(o, keys):
    if isinstance(o, dict): return {k: strip(v, keys) for k, v in o.items() if k not in keys}
    if isinstance(o, list): return [strip(v, keys) for v in o]
    return o
out = {}
for name in sys.argv[1:]:
    label = "final"
    mine = json.loads((V / "out" / name / f"{label}_row.json").read_text())
    r = mine["row"]; ref = rows[(name, label)]
    diffs = {k: (r[k], ref[k]) for k in r if str(r[k]) != ref[k]}
    # flags
    f_mine = (V / "out" / name / f"{label}_flags.csv").read_text()
    f_ref = (S / "panels" / name / label / "flags.csv").read_text()
    # per-step CSVs vs recorded sha256
    rec = dict(reversed(l.split("  ")) for l in (S / "panels" / name / label / "csv_sha256.txt").read_text().splitlines())
    csv_mis = [n for n, s in rec.items() if sha(V / "out" / name / label / n) != s]
    # trace JSON
    jm = json.load(open(V / "out" / name / label / "episode_trace.json"))
    jr = json.load(open(S / "panels" / name / label / "episode_trace.json"))
    top_diff = [k for k in set(jm) | set(jr) if jm.get(k) != jr.get(k)]
    out[name] = {"row_fields_compared": len(r), "row_diffs": diffs, "problems": mine["problems"],
                 "gate_flagged_mine": mine["gate_flagged"],
                 "gate_flagged_ref": json.loads((S / "panels" / name / label / "panel_extra.json").read_text())["gate_flagged"],
                 "flags_csv_identical": f_mine == f_ref, "csv_count": len(rec), "csv_sha_mismatch": csv_mis,
                 "trace_json_top_level_diff_keys": top_diff,
                 "trace_json_episodes_equal": jm["episodes"] == jr["episodes"],
                 "trace_json_diff_values": {k: (jm.get(k), jr.get(k)) for k in top_diff if k != "episodes"},
                 "key_values": {k: r[k] for k in ("cmd", "frac99", "frac90", "tailup_flags", "other_flags", "other_reasons", "falls", "short_episodes", "return_mean", "class_single", "num_timesteps")},
                 "lengths_mine": [e["length"] for e in jm["episodes"]], "returns_equal": [e["return"] for e in jm["episodes"]] == [e["return"] for e in jr["episodes"]]}
print(json.dumps(out, indent=1, default=str))
