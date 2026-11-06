"""Stage 1 of the final text: parse the fix stage's 18 text corrections and apply them to v3, bottom-up.

Every OLD line is checked against v3's line of that number before anything is replaced; a mismatch refuses the whole
stage (nothing is written). C6 is applied in its alternative (a) form (maintainer decision (35)): its NEW text is
replaced by C6_ALT_A below, the rest of each correction's NEW text is used verbatim.
Imported by make_final.py; run alone it writes data/stage1_corrected.txt and data/stage1_report.json.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
P4 = os.path.dirname(os.path.dirname(LANE))
V3 = os.path.join(P4, "postfix", "fix", "criteria_pilot4_draft_v3.txt")
CORR = os.path.join(P4, "tools", "out", "fix", "text_corrections.txt")
V3_SHA256 = "1c52644dc96dfbe7188f42a7c81aa49c3743138781ff023da6186d528ffcc19d"

# C6 alternative (a), decision (35): v3 lines 740-743 with the tolerance 1e-4 deg and the tolerance stage's records.
C6_ALT_A = """\
      only). Tests: states 0..650 equal the panel episode's bitwise; a 0-deg nudge reproduces the panel episode
      bitwise; with the command columns zeroed (the entry baseline), the nudged episode equals the panel episode
      turned by the angle (heading + angle to 1e-4 deg at 10 s; on the 118 nudged entry episodes the largest deviation
      is 2.73e-5 deg, so none is flagged: s42 0 of 40, s52 0 of 38, s60 0 of 40, every baseline run exits 0; v3's
      1e-5, set from 15 episodes, flagged 3 s42 episodes at 2.73e-5, 1.94e-5 and 1.08e-5 deg and was loosened by
      maintainer decision (35) [measured: tolerance stage, SP/work/pilot4/prelaunch/tol/val_final/baseline_s42.json,
      _s52, _s60]) and the state-651 observation differs from the un-nudged one only in obs[60:63]; the pass check at
      state 650; NOT READ below 30 counting episodes.""".split("\n")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def parse_corrections(text):
    """Return a list of dicts: id, kind ('replace' or 'insert_after'), first, last, old (list), new (list)."""
    blocks = re.split(r"^={20,}\n", text, flags=re.M)
    out = []
    for blk in blocks[1:]:
        lines = blk.split("\n")
        head = lines[0]
        m = re.match(r"^(C\d+[ab]?)\s+v3 lines? (\d+)(?:-(\d+))?(\s+\(INSERT AFTER\))?", head)
        if not m:
            continue
        cid, first, last, ins = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2)), bool(m.group(4))
        mode = None
        old, new = [], []
        for ln in lines[1:]:
            if ln in ("OLD:", "AFTER:"):
                mode = "old"
                continue
            if ln in ("NEW:", "INSERT:"):
                mode = "new"
                continue
            if ln.startswith("Not changed in the text"):
                break
            if mode == "old":
                mm = re.match(r"^\s*(\d+) \|(?: (.*))?$", ln)
                if mm:
                    old.append((int(mm.group(1)), mm.group(2) or ""))
            elif mode == "new":
                new.append(ln)
        while new and new[-1] == "":
            new.pop()
        out.append({"id": cid, "kind": "insert_after" if ins else "replace", "first": first, "last": last,
                    "old": old, "new": new, "header": head, "block": blk})
    return out


def apply(v3_lines, corrections):
    report = []
    problems = []
    for c in corrections:
        nums = [n for n, _ in c["old"]]
        if nums != list(range(c["first"], c["last"] + 1)):
            problems.append(f"{c['id']}: OLD line numbers {nums} do not cover {c['first']}-{c['last']}")
            continue
        for n, t in c["old"]:
            if v3_lines[n - 1] != t:
                problems.append(f"{c['id']}: v3 line {n} does not match OLD: {v3_lines[n - 1]!r} != {t!r}")
        report.append({"id": c["id"], "kind": c["kind"], "lines": [c["first"], c["last"]], "old_matches": True})
    spans = sorted((c["first"], c["last"], c["id"]) for c in corrections if c["kind"] == "replace")
    for (a1, b1, i1), (a2, b2, i2) in zip(spans, spans[1:]):
        if a2 <= b1:
            problems.append(f"overlap {i1} {a1}-{b1} and {i2} {a2}-{b2}")
    if problems:
        return None, report, problems
    lines = list(v3_lines)
    for c in sorted(corrections, key=lambda c: c["first"], reverse=True):
        new = C6_ALT_A if c["id"] == "C6" else c["new"]
        if c["kind"] == "insert_after":
            lines[c["last"]:c["last"]] = new
        else:
            lines[c["first"] - 1:c["last"]] = new
        for r in report:
            if r["id"] == c["id"]:
                r["new_lines"] = len(new)
                r["new_is_alternative_a"] = c["id"] == "C6"
    return lines, report, problems


def stage1():
    with open(V3, "rb") as f:
        raw = f.read()
    if sha256_bytes(raw) != V3_SHA256:
        raise SystemExit(f"v3 sha256 {sha256_bytes(raw)} is not {V3_SHA256}: refusing")
    v3_lines = raw.decode("utf-8").split("\n")
    if v3_lines[-1] == "":
        v3_lines = v3_lines[:-1]
    with open(CORR, encoding="utf-8") as f:
        corr_text = f.read()
    corrections = parse_corrections(corr_text)
    ids = [c["id"] for c in corrections]
    expected = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9a", "C9b", "C10", "C11", "C12a", "C12b", "C13",
                "C14", "C15", "C16"]
    if ids != expected:
        raise SystemExit(f"parsed corrections {ids} are not the 18 expected: refusing")
    c6 = [c for c in corrections if c["id"] == "C6"][0]
    if "ALTERNATIVE (a)" not in c6["block"] or "1e-4 deg" not in c6["block"]:
        raise SystemExit("C6 does not name its alternative (a): refusing")
    lines, report, problems = apply(v3_lines, corrections)
    if problems:
        raise SystemExit("OLD text mismatch, nothing written:\n" + "\n".join(problems))
    return v3_lines, lines, report, {"v3_sha256": V3_SHA256, "v3_lines": len(v3_lines),
                                     "corrections_sha256": sha256_bytes(corr_text.encode("utf-8")),
                                     "n_corrections": len(corrections)}


if __name__ == "__main__":
    v3_lines, lines, report, meta = stage1()
    with open(os.path.join(LANE, "data", "stage1_corrected.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(LANE, "data", "stage1_report.json"), "w") as f:
        json.dump({"meta": meta, "report": report, "lines_out": len(lines)}, f, indent=1)
    print(json.dumps({"meta": meta, "lines_out": len(lines),
                      "applied": [(r["id"], r["lines"], r.get("new_lines")) for r in report]}))
    sys.exit(0)
