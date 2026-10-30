"""The start pair, the parent records and the reference read's pairs: sha256 of the F2 .copy1 files against the fetch
manifest's COPY rows (FMAN) and, for the start pair and the records, against the pins in bs_common.START.  LB: BS2's
check_inputs.py with the reference read's F2 pairs (read_bs.REFERENCE_PAIRS: every 1,048,576 in [S, E] and the final
pair) added, FMAN only (read_bs checks them against FMAN again before reading).  Reads only.  Exit 0 iff every file
equals FMAN and every pinned one its pin."""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("bs_common", str(Path(__file__).resolve().parent / "bs_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

ST = C.START
RUN = ST["run_id"]
PINS = {ST["zip_rel"]: ST["zip_sha256"], ST["pkl_rel"]: ST["pkl_sha256"]}
PINS.update({f"{C.STAGE_DIR}/{n}": ST["records"][n] for n in (*C.PARENT_RECORDS, "stage_config.json")})
REF_STEPS = tuple(s for s in range(1_048_576, C.E + 1, 1_048_576) if s >= C.S) + (C.E,)     # = read_bs.REFERENCE_PAIRS
for s in REF_STEPS:
    names = (C.FINAL_ZIP, C.FINAL_PKL) if s == C.E else C.periodic_names(s)
    for n in names:
        PINS.setdefault(f"{C.STAGE_DIR}/models/{n}", None)                               # FMAN only
with open(C.FMAN, newline="") as fh:
    rows = list(csv.DictReader(fh))                 # data rows: the file's line number is index + 2
out, ok = {}, True
for rel, pin in PINS.items():
    path = f"{C.f2_copies()}/{rel}.copy1"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    got = h.hexdigest()
    man = [(i + 2, r["sha256"]) for i, r in enumerate(rows) if r["fetch"] == "F2" and r["run_id"] == RUN
           and r["path"] == rel and r["status"] == "COPY" and r["copy"] == "1"]
    eq_man = sorted({s for _, s in man}) == [got]
    eq_pin = pin is None or got == pin
    ok &= eq_man and eq_pin
    out[rel] = {"sha256": got, "fman_lines": [ln for ln, _ in man], "manifest_copy1": sorted({s for _, s in man}),
                "equal_fman": eq_man, "equal_bs_common_pin": eq_pin if pin is not None else "not pinned (FMAN only)"}
out["_reference_pairs"] = list(REF_STEPS)
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
