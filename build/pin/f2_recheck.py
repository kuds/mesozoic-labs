import csv, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs/tools")
import ecs_common as C
fman = {}
with open(C.FMAN, newline="") as fh:
    for r in csv.DictReader(fh):
        if r["fetch"] == "F2" and r["status"] == "COPY" and r["copy"] == "1":
            fman[(r["run_id"], r["path"])] = r["sha256"]
bad = 0
for start, st in C.STARTS.items():
    items = [(st["zip_rel"], st["zip_sha256"]), (st["pkl_rel"], st["pkl_sha256"])]
    items += [(f"{C.STAGE_DIR}/{n}", h) for n, h in st["records"].items()]
    for rel, pin in items:
        got = C.sha256_file(C.f2_file(start, rel))
        fm = fman.get((st["run_id"], rel))
        ok = got == pin == fm
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {start} {rel:55s} {got}")
print("ALL_EQUAL" if bad == 0 else f"MISMATCHES {bad}")
