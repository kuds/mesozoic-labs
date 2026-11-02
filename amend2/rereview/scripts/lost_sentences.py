"""Re-review: sentences of a reviewed text (orig/) absent from its revision (whitespace normalised)."""
import re
import sys

def sents(p):
    t = re.sub(r"\s+", " ", open(p, encoding="utf-8").read())
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+(?=[A-Z(\-\"])", t) if s.strip()]

old, new = sents(sys.argv[1]), re.sub(r"\s+", " ", open(sys.argv[2], encoding="utf-8").read())
gone = [s for s in old if s not in new]
print(f"{len(old)} sentences in {sys.argv[1]}; {len(gone)} not found verbatim in {sys.argv[2]}:")
for s in gone:
    print(" -", s[:400])
