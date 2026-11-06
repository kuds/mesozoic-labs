"""Summarise the e2e lane's test run: per file, the junit counts and time, the run wrapper's record, the score test's
summary line; written as JSON (scratch helper; reads only files in the lane folder).

usage: python -I -B junit_summary.py TESTS_FINAL_DIR OUT.json
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

d = Path(sys.argv[1])
out = {"files": {}}
for x in sorted(d.glob("*.xml")):
    root = ET.parse(x).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    tot = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time_s": 0.0}
    for s in suites:
        for k in ("tests", "failures", "errors", "skipped"):
            tot[k] += int(s.get(k, 0))
        tot["time_s"] += float(s.get("time", 0))
    cases = [c for s in suites for c in s.iter("testcase")]
    tot["testcases_listed"] = len(cases)
    tot["skipped_cases"] = [c.get("name") for c in cases if c.find("skipped") is not None]
    tot["failed_cases"] = [c.get("name") for c in cases if c.find("failure") is not None or c.find("error") is not None]
    run = d / (x.stem + ".run.json")
    if run.is_file():
        tot["wrapper"] = json.loads(run.read_text())
        tot["wrapper"].pop("pytest_args", None)
    log = (d / (x.stem + ".log.txt")).read_text()
    m = re.findall(r"^=+ (.*) =+$", log, re.M)
    tot["summary_line"] = m[-1] if m else None
    out["files"][f"tests/{x.stem}.py"] = tot
for mode in ("tree", "pure"):
    log = (d / f"test_pilot4_score.{mode}.log.txt").read_text()
    last = [ln for ln in log.splitlines() if re.match(r"^\d+/\d+ passed", ln)]
    env = [ln for ln in log.splitlines() if "environments resolves to" in ln]
    out["files"][f"test_pilot4_score.py ({mode})"] = {"summary_line": last[-1] if last else None,
                                                     "environments_line": env[0] if env else None,
                                                     "fail_lines": [ln for ln in log.splitlines()
                                                                    if ln.startswith("FAIL")]}
rcs = (d / "rcs.txt").read_text().splitlines()
out["rcs"] = rcs
out["total_pytest_tests"] = sum(v.get("tests", 0) for v in out["files"].values())
out["total_pytest_failed_or_error"] = sum(v.get("failures", 0) + v.get("errors", 0) for v in out["files"].values())
out["total_pytest_skipped"] = sum(v.get("skipped", 0) for v in out["files"].values())
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
for k, v in out["files"].items():
    print(k, {kk: v.get(kk) for kk in ("tests", "failures", "errors", "skipped", "time_s", "summary_line")})
print("total", out["total_pytest_tests"], "failed/error", out["total_pytest_failed_or_error"], "skipped",
      out["total_pytest_skipped"])
