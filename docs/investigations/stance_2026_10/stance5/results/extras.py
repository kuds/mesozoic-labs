"""Report-only extras on each stance5 final: the fresh panel (7042-7081), the forward-start tail panel (32 seeds) and
the common-noise panel (40 seeds), read from judge_stance5.py's files.  Written for the record on 2026-10-22.

    python3 -I extras.py [JUDGED_ROOT]

JUDGED_ROOT defaults to scratchpad/foot/stance5/judged (the session's scratch, relative to the working directory).
The references printed last are the criteria's (PRE-REGISTERED READOUTS).  Nothing here enters a verdict.
"""

from __future__ import annotations

import json
import os
import sys

RUNS = ("stance5_LTSB_s52", "stance5_LTSB_s60", "stance5_LTSB_s42")
REFERENCES = (
    "references (criteria): fresh panel, stance4 finals 39/40, 38/40 and 14/40 clean, stance3 Run 2 final 40/40; "
    "tail panel, statue 0, stance3 Run 2 final 0, stance4 finals 2, 6 and 5 (of 32); common noise, stance4 finals "
    "5, 1 and 30 (of 40)"
)


def load(root: str, run: str, mode: str) -> dict | None:
    path = os.path.join(root, run, f"judge_{mode}_final.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/foot/stance5/judged"
    for run in RUNS:
        out = [run]
        d = load(root, run, "fresh")
        if d:
            p = d["panel"]
            falls = sum(v for k, v in p["terminations"].items() if k != "truncated")
            out.append(
                f"fresh {p['verdict']} clean {p['n_clean']}/{p['episodes']} falls {falls} "
                f"hop-or-fall {p['hop_or_fall_episodes']} K2a {p['summary']['k2a_count']}"
            )
        else:
            out.append("fresh not judged")
        d = load(root, run, "tail")
        out.append(
            f"tail falls {d['tail']['falls']}/{d['tail']['n']} (statue {d['tail']['statue_falls']})"
            if d
            else "tail not judged"
        )
        d = load(root, run, "common_noise")
        c = d["common_noise"] if d else None
        out.append(f"common-noise falls {c['falls']}/{c['n']}" if c else "common-noise not judged")
        print(" | ".join(out))
    print(REFERENCES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
