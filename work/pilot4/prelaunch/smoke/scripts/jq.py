"""Tiny read-only JSON helpers for the smoke lane (no shell quoting).

    python -I jq.py keys FILE.json KEY [KEY ...]          print {KEY: value} for top-level keys (dotted paths allowed)
    python -I jq.py edits FILE.json                       print a runner record's edits, one per line
    python -I jq.py params_diff A.json B.json             compare two runner params files but QUICK_TEST, RUN_LABEL,
                                                          path_map; print the differences (none: "equal")
"""

from __future__ import annotations

import json
import sys


def get(d, dotted):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return "<absent>"
    return cur


def main(argv):
    cmd = argv[1]
    if cmd == "keys":
        d = json.load(open(argv[2]))
        for k in argv[3:]:
            print(f"{k}: {json.dumps(get(d, k), default=str)}")
    elif cmd == "edits":
        for e in json.load(open(argv[2]))["edits"]:
            print(e)
    elif cmd == "params_diff":
        a, b = (json.load(open(p)) for p in argv[2:4])
        for d in (a, b):
            d["config"].pop("QUICK_TEST", None)
            d["config"].pop("RUN_LABEL", None)
            d.pop("path_map", None)
        if a == b:
            print("equal (but QUICK_TEST, RUN_LABEL and path_map)")
        else:
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    print(k, json.dumps(a.get(k)), "!=", json.dumps(b.get(k)))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
