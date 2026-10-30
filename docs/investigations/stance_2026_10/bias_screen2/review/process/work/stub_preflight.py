"""Stub RUN_CMD for run_waves_bs.sh in its test mode (BS_WAVES_TEST=1, BS_ROOT = the review's copy root): the COPY's
run_bs.py preflight with bs_common.BS pointed at BS_ROOT, then exit 2 (refused) or 0.  It NEVER calls run(): nothing
is created and nothing trains.  Process review of BS2 only."""
import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

root = os.path.realpath(os.environ["BS_ROOT"])
assert root.endswith("review/process/work/copyroot"), root
spec = importlib.util.spec_from_file_location("rb_copy", os.path.join(root, "tools", "run_bs.py"))
RB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RB)
RB.C.BS = root
ns = RB.parse(sys.argv[1:])
problems = RB.preflight(ns)
print(json.dumps({"stub": "the copy's run_bs.preflight only; run() never called", "argv": sys.argv[1:],
                  "problems": problems}), flush=True)
sys.exit(2 if problems else 0)
