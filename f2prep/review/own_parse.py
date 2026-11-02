"""Reviewer's own reading of the RUNLOG parse rule (criteria RUN HANDLING / Records; r10round_common.parse_runlog's
documented rule), written independently: a line is a NOTE of the round when, stripped, it reads
'<YYYY-MM-DD HH:MM[:SS]> UTC <ws> STANCE7 <ws> <KIND> <ws> <RUN_ID>...' with KIND one of the Records kinds and RUN_ID a
round RUN_ID or a suffixed one; any other line holding a token 'stance7_r10LTSB(pc|off)_s<digits>...' that starts with a
round RUN_ID is an OTHER-FORM line.  Also lists lines naming a run only in an abbreviated form (pc_sNN / off_sNN).
Read only.  Usage: PY -I -B own_parse.py RUNLOG"""
import hashlib
import re
import sys

KINDS = {"LAUNCH", "CHECK", "STOP", "TRIGGER-a", "TRIGGER-b", "TRIGGER-c", "TRIGGER-d", "LISTING", "DELETED",
         "FOUND-GONE", "RELAUNCH", "ACTION", "F2-START", "REFETCH-START", "SNAPSHOT"}
RUNS = [f"stance7_r10LTSB{a}_s{s}" for a in ("pc", "off") for s in (60, 42, 80)]
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
notes, others, abbrev = [], [], []
for i, raw in enumerate(lines, 1):
    s = raw.strip()
    parts = s.split()
    is_note = False
    if (len(parts) >= 5 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[0]) and re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", parts[1])
            and parts[2] == "UTC" and parts[3] == "STANCE7" and parts[4] in KINDS):
        run = parts[5] if len(parts) > 5 else ""
        if any(run == r or run.startswith(r + "_") for r in RUNS):
            is_note = True
            notes.append((i, parts[4], run))
    toks = re.findall(r"stance7_r10LTSB(?:pc|off)_s\d+\w*", raw)
    if not is_note and any(any(t.startswith(r) for r in RUNS) for t in toks):
        others.append(i)
    elif not is_note and not toks and re.search(r"\b(pc|off)_s(60|42|80)\b", raw):
        abbrev.append(i)
print("sha256", hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(), "lines", len(lines))
print("notes", len(notes))
for n in notes:
    print("  NOTE", *n)
print("others", len(others), others)
print("abbreviated-name lines (no full RUN_ID token; not 'other' by the rule)", len(abbrev), abbrev)
