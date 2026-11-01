"""Fixer: pc_common.comparators recomputed with the fixed tools (the adapter's dry/comparators_recomputed.out format)."""
import importlib.util, json, sys
T = "/home/user/work/pc_study/tools/pc"
def load(n):
    s = importlib.util.spec_from_file_location(n, f"{T}/{n}.py"); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
C = load("r10round_common"); P = load("pc_common")
comp = P.comparators(C, C.load_pins())
for s, r in comp["runs"].items():
    print(f"{s} {r['run']} TD {r['td']['td']:.6f} TDJ {r['td']['td_j']:.6f} n {r['td']['n']} missing {r['td']['missing']}")
    e = r["early"]
    print(f"   early {json.dumps(e['firsts'])} n>=.60 {e['n_cmd_l_ge_060']} tail alarms m5 {len(e.get('m5_tail_alarms') or [])} flagged {len(e.get('m5_tail_flagged') or [])}")
    print(f"   classes {json.dumps(r['classes'])}")
    h = r["habit"]
    print(f"   habit {json.dumps(h['status'])} {h['saturating']} {h['drive_only']} {json.dumps(h['sites'])} no m1: {h['labels_without_m1']}")
    print(f"   probe {r['probe_tail3']} mean return {r['mean_return']:.1f} falls {r['falls']} of {r['rows']}")
print(f"score.json {comp['score_json_sha256']} fetch manifest {comp['fetch_manifest_sha256']}")
