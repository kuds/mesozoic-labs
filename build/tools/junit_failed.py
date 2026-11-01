"""Print failed/errored test ids and a count summary from a pytest junit xml (stdlib only)."""
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
bad = []
for case in root.iter("testcase"):
    name = case.get("classname", "").replace(".", "/") + "::" + case.get("name", "")
    kinds = [child.tag for child in case]
    if "failure" in kinds:
        counts["failed"] += 1
        bad.append("FAILED " + name)
    elif "error" in kinds:
        counts["error"] += 1
        bad.append("ERROR " + name)
    elif "skipped" in kinds:
        counts["skipped"] += 1
    else:
        counts["passed"] += 1
print(counts)
for line in sorted(bad):
    print(line)
