import importlib.util, json, sys
sys.dont_write_bytecode = True
T = "/home/user/work/pc_study/tools/pc/"
def load(n):
    s = importlib.util.spec_from_file_location(n, T + n + ".py"); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
C = load("r10round_common")
notes, _o = C.parse_runlog(C.RUNLOG, C.RUN_IDS)
for rid in sorted(C.RUN_IDS):
    ns = C.notes_for(notes, rid)
    if not ns: continue
    v = dict(launches=[n["line"] for n in C.notes_for(notes, rid, ("LAUNCH","RELAUNCH"))], launch_check_stops=[], actions=[],
             lines={n["line"]: n["kind"] for n in ns})
    j = json.dumps(v, default=str)
    print(rid, len(ns), "notes; listing json", len(j), "chars (+2 F2-START/REFETCH-START entries about +60)")
