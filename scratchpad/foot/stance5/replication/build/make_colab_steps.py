"""Write R/colab_steps.txt and R/colab_short.md from the templates beside this script, with the criteria text embedded
byte for byte and the current sha256s of criteria.txt, judge_stance5r.py, score_stance5r.py and sigma_s70_s80.json.

    python3 -I make_colab_steps.py [--force]    (--force replaces existing files: the normal path after a re-pin)
    python3 -I make_colab_steps.py --check      (rebuilds in memory, compares with the files on disk; writes nothing)
    python3 -I make_colab_steps.py --help       (this text; any other argument is refused, and a file whose content is
                                                 already the built text is not rewritten)

Re-run it whenever criteria.txt, the judge, the scorer or the sidecar changes before the first training step.  It
checks that the criteria block extracted the way the steps tell the maintainer to (awk between the BEGIN and END lines)
hashes to the criteria's sha256.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
R = os.path.dirname(HERE)
SOURCES = {"CRITERIA_SHA": os.path.join(R, "criteria.txt"), "JUDGE_SHA": os.path.join(R, "judge", "judge_stance5r.py"),
           "SCORER_SHA": os.path.join(R, "judge", "score_stance5r.py"), "SIDECAR_SHA": os.path.join(R, "sigma_s70_s80.json")}
OUT = {"colab_steps_template.txt": os.path.join(R, "colab_steps.txt"),
       "colab_short_template.md": os.path.join(R, "colab_short.md")}


def sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def extract(text: str) -> str:
    """awk '/^=== END criteria.txt/{f=0} f; /^=== BEGIN criteria.txt/{f=1}' in python."""
    out, f = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("=== END criteria.txt"):
            f = False
        if f:
            out.append(line)
        if line.startswith("=== BEGIN criteria.txt"):
            f = True
    return "".join(out)


def build() -> dict:
    vals = {k: sha(p) for k, p in SOURCES.items()}
    with open(SOURCES["CRITERIA_SHA"]) as fh:
        crit = fh.read()
    if not crit.endswith("\n"):
        raise SystemExit("criteria.txt must end with a newline (the END line follows it)")
    vals["CRITERIA_TEXT"] = crit
    out = {}
    for tpl, dst in OUT.items():
        with open(os.path.join(HERE, tpl)) as fh:
            text = fh.read()
        for k, v in vals.items():
            text = text.replace("{" + k + "}", v)
        if "{CRITERIA" in text or "{JUDGE_SHA}" in text or "{SCORER_SHA}" in text or "{SIDECAR_SHA}" in text:
            raise SystemExit(f"{tpl}: a placeholder is left")
        out[dst] = text
    steps = out[OUT["colab_steps_template.txt"]]
    got = hashlib.sha256(extract(steps).encode("utf-8")).hexdigest()
    if got != vals["CRITERIA_SHA"]:
        raise SystemExit(f"the embedded criteria extract to sha256 {got}, not {vals['CRITERIA_SHA']}")
    return {"files": out, "shas": {k: v for k, v in vals.items() if k != "CRITERIA_TEXT"}}


def main(argv: list[str]) -> int:
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    unknown = [a for a in argv if a not in ("--check", "--force")]
    if unknown or ("--check" in argv and "--force" in argv):
        raise SystemExit(f"usage: make_colab_steps.py [--force | --check | --help] (refused: {argv})")
    res = build()
    if "--check" in argv:
        bad = [p for p, t in res["files"].items() if not os.path.exists(p) or open(p).read() != t]
        print({"check": "ok" if not bad else "DIFFERS", "files": bad, "shas": res["shas"]})
        return 1 if bad else 0
    for p, t in res["files"].items():
        if os.path.exists(p) and open(p).read() != t and "--force" not in argv:
            raise SystemExit(f"{p} exists and differs (--force to replace)")
    wrote, same = [], []
    for p, t in res["files"].items():
        if os.path.exists(p) and open(p).read() == t:
            same.append(p)   # identical: not rewritten (the mtime stays)
            continue
        with open(p, "w") as fh:
            fh.write(t)
        wrote.append(p)
    print({"wrote": wrote, "unchanged": same, "shas": res["shas"]})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
