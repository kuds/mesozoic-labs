"""Run pytest in this process and record, as JSON, its exit status, its wall time and whether `environments` was
imported in this (the pytest) process, and from where (e2e lane helper; scratch).

usage: python -I -B run_pytest.py OUT.json [pytest args...]

The tests import repository code only in subprocesses through iso.py (each such test asserts the resolved path);
this records the in-process state, which should be "not imported".
"""

import json
import sys
import time

import pytest

out = sys.argv[1]
t0 = time.time()
rc = pytest.main(sys.argv[2:])
elapsed = time.time() - t0
mod = sys.modules.get("environments")
rec = {
    "pytest_args": sys.argv[2:],
    "exit_status": int(rc),
    "elapsed_s": round(elapsed, 2),
    "environments_imported_in_pytest_process": mod is not None,
    "environments_file_in_pytest_process": getattr(mod, "__file__", None) if mod is not None else None,
    "python": sys.executable,
    "dont_write_bytecode": sys.dont_write_bytecode,
}
with open(out, "w") as fh:
    json.dump(rec, fh, indent=1)
    fh.write("\n")
print(json.dumps(rec))
sys.exit(int(rc))
