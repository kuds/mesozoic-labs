"""The start pair and the parent records: sha256 of the F2 .copy1 files against the fetch manifest's COPY rows (FMAN)
and against the pins in bs_common.START.  BS2: the paths and pins come from bs_common (BS's version listed the
1,048,576 pair literally); the FMAN line of each row is printed.  Reads only.  Exit 0 iff every file equals both."""
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
    eq_pin = got == pin
    ok &= eq_man and eq_pin
    out[rel] = {"sha256": got, "fman_lines": [ln for ln, _ in man], "manifest_copy1": sorted({s for _, s in man}),
                "equal_fman": eq_man, "equal_bs_common_pin": eq_pin}
print(json.dumps(out, indent=1))
sys.exit(0 if ok else 1)
