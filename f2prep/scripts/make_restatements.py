"""Draft (never write to RUNLOG) the restatements that rule 7 of ST/tools/pc/fetch_agent_brief.txt and COLAB_STEPS
line 239 ask the coordinator for before the F2-START lines: every RUNLOG line that names a round RUN_ID and is not a
STANCE7 note of a round RUN_ID, found by the pinned rule itself (r10round_common.parse_runlog with C.RUN_IDS, as
score_pc.py line 312 calls it).  For each line: its sha256 (so the coordinator can check it is unchanged), its RUN_ID
tokens, the Records KIND its own words name (ROUND l.411: "restate in this form, with its original time and the KIND
its own words name") with the KIND line(s) of the same RUN_ID that already record that event, and a STANCE7 NOTE draft
(original time; the line's own words, verbatim but for one substitution: the PC branch's name is written as
"<PC branch>", APPROVED's "branch"; line 881, APPROVED's 4,864 bytes, is restated by its pinned fields).
Checks (printed): every draft carries the original's time and exactly the original's RUN_ID tokens; the pinned parser
reads the drafts as no note at all (0 notes: no KIND line is added, so no rule's reading changes) and prints each as an
other-form line.  Reads RUNLOG read only; writes only the --out file.

    PY -I -B make_restatements.py --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

T = Path("/home/user/work/pc_study/tools/pc")
spec = importlib.util.spec_from_file_location("r10round_common", T / "r10round_common.py")
C = importlib.util.module_from_spec(spec)
sys.modules["r10round_common"] = C
spec.loader.exec_module(C)

BRANCH_RE = re.compile(r"\b[a-z]+/screen-trex-r10-phasecritic\b")
TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?) UTC\s+(?:STANCE7 NOTE\s+)?(.*)$")

#: line -> (the KIND its own words name, the KIND lines of the same RUN_ID(s) that already record it, remark)
CLASS = {
    874: ("none", "", "the maintainer's design answers (no run existed)"),
    875: ("none", "", "a workflow launch (no run existed)"),
    876: ("none", "", "the maintainer's waves decision (no run existed)"),
    881: ("none", "", "the pin record: APPROVED's exact bytes, following line 880 (03:07:17 UTC)"),
    884: ("LAUNCH", "LAUNCH lines 885, 886, 887", "the three PHASE launches"),
    889: ("CHECK (rows)", "CHECK line 893", "cell-9 configuration rows of the PHASE s60 launch check"),
    891: ("CHECK (rows)", "CHECK line 895", "cell-9 configuration rows of the PHASE s80 launch check"),
    906: ("LAUNCH", "LAUNCH line 907", "the OFF s60 launch"),
    909: ("CHECK (rows)", "CHECK line 915", "cell-6/9 rows of the OFF s60 launch check"),
    910: ("LAUNCH", "LAUNCH line 911 (OFF s80); OFF s42 launched later, LAUNCH line 923",
          "the OFF s80 launch; the order deviation names no KIND"),
    916: ("CHECK (result)", "CHECK line 915", "the OFF s60 launch-check result"),
    921: ("CHECK (result)", "CHECK line 920", "the OFF s80 launch-check result"),
    922: ("LAUNCH", "LAUNCH line 923; its CHECK line 932", "the OFF s42 launch and its rows"),
    924: ("none", "SNAPSHOT line 902 (the opening it cites is itself a KIND line)", "the blind period's end, interpreted"),
    925: ("CHECK (field new_run)", "CHECK line 915 (new_run=not-reported; no pinned tool reads new_run)", "cell-8 row"),
    926: ("CHECK (field new_run)", "CHECK line 920 (new_run=not-reported; no pinned tool reads new_run)", "cell-8 row"),
    927: ("CHECK (field new_run)", "CHECK line 932 (already new_run=\"new run\")", "cell-8 row"),
    928: ("none", "", "a notification request"),
    933: ("CHECK (result)", "CHECK line 932", "the OFF s42 launch-check result"),
    937: ("none", "", "liveness from Drive metadata only; no trigger"),
    938: ("none", "", "liveness from Drive metadata only; no trigger"),
    939: ("none", "", "liveness from Drive metadata only; no trigger; an inferred timetable"),
    940: ("none", "", "a schedule change of a check"),
}


def restate_881(line: str) -> str:
    import json

    d = json.loads(line)
    ids = d["pins"]["run_ids"]
    v = ", ".join(f"{k} {d['validations'][k]}" for k in sorted(d["validations"]))
    return (f"APPROVED's exact bytes as pinned (line 880: written once, 0444; its sha256 there): criteria_sha256 "
            f"{d['criteria_sha256']}, branch <PC branch>, branch_head {d['branch_head']}, run_ids phase "
            f"{', '.join(ids['phase'])} and off {', '.join(ids['off'])}, validations {v}, approved_at {d['approved_at']}, "
            f"pinned_at {d['pinned_at']}; the bytes themselves stand only on line 881")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runlog = Path(C.RUNLOG)
    raw = runlog.read_bytes()
    text = raw.decode("utf-8", "replace").splitlines()
    notes, others = C.parse_runlog(runlog, C.RUN_IDS)
    lines_other = [i for i, _ in others]
    out, drafts, problems = [], [], []
    shown = "S1/foot/stance4/live/run_log.txt"
    out.append("RESTATEMENT DRAFTS (rule 7 of the fetch brief; COLAB_STEPS line 239; ROUND l.403-412 Records) -- DRAFT ONLY, "
               "nothing here is written to RUNLOG; the coordinator decides and writes.")
    out.append(f"Read: RUNLOG {shown} sha256 {hashlib.sha256(raw).hexdigest()}, {len(text)} lines [measured]; "
               f"r10round_common.py sha256 {hashlib.sha256((T / 'r10round_common.py').read_bytes()).hexdigest()} (pinned).")
    out.append(f"The pinned rule (parse_runlog with C.RUN_IDS) reads {len(notes)} STANCE7 notes of round RUN_IDs and prints "
               f"{len(others)} other lines naming a PC RUN_ID: {lines_other} [measured].")
    out.append("")
    for i, line in others:
        kind, rec, remark = CLASS.get(i, ("UNCLASSIFIED", "", ""))
        if kind == "UNCLASSIFIED":
            problems.append(f"line {i} has no classification (a line added after this draft was made)")
        h = hashlib.sha256(line.encode()).hexdigest()
        toks = sorted(set(C.RUN_TOKEN_RE.findall(line)))
        if i == 881:
            t0 = TIME_RE.match(text[879].strip()).group(1)
            body = restate_881(line)
            when = f"no time of its own; line 880's {t0} UTC"
        else:
            m = TIME_RE.match(line.strip())
            if not m:
                problems.append(f"line {i}: no leading time")
                continue
            t0, body = m.group(1), m.group(2)
            when = f"{t0} UTC"
        body = BRANCH_RE.sub("<PC branch>", body)
        tag = f"KIND its own words name: {kind}" + (f"; already recorded by {rec}" if rec else "")
        draft = f"{t0} UTC  STANCE7 NOTE restates RUNLOG line {i} ({when}; other form; {tag}): {body}"
        dt = sorted(set(C.RUN_TOKEN_RE.findall(draft)))
        if dt != toks:
            problems.append(f"line {i}: draft RUN_ID tokens {dt} differ from the line's {toks}")
        if not draft.startswith(t0):
            problems.append(f"line {i}: draft time differs")
        drafts.append(draft)
        out.append(f"LINE {i}  time {when}  sha256 {h}  RUN_IDs {toks}")
        out.append(f"  {tag} ({remark})")
        out.append(f"  DRAFT: {draft}")
        out.append("")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "drafts.txt"
        p.write_text("\n".join(drafts) + "\n", encoding="utf-8")
        dn, do = C.parse_runlog(p, C.RUN_IDS)
    ok = not dn and len(do) == len(drafts) and not problems
    out.append(f"CHECKS [measured]: {len(drafts)} drafts for {len(others)} other lines; the pinned parser reads the drafts as "
               f"{len(dn)} notes and {len(do)} other lines (a STANCE7 NOTE line is no Records KIND, so a NOTE restatement "
               "adds no note and changes no rule's reading; each one is itself printed as an other-form line); every "
               f"draft carries its line's time and exactly its RUN_ID tokens: {not problems}; problems {problems}.")
    out.append(f"RESULT: {'ALL CHECKS PASS' if ok else 'CHECK FAILED'}")
    out.append("")
    ids = ", ".join(str(i) for i in lines_other)
    out.append("OPTION A (recommended [inferred], the stance6 precedent of RUNLOG line 706, which left its four commentary "
               "lines unrestated): no KIND restatement; one line naming no RUN_ID, e.g.")
    out.append(f"  <date -u> UTC  STANCE7 NOTE restatement review before the F2-START lines: RUNLOG lines {ids} name round "
               "RUN_IDs in another form; each is commentary (decisions, the pin record, liveness from metadata, a schedule) "
               "or names a LAUNCH or CHECK already recorded on that RUN_ID's own KIND line (LAUNCH 885, 886, 887, 907, 911, "
               "923; CHECK 893, 895, 915, 920, 932); none is restated in KIND form (a second LAUNCH line would make C1 read "
               "two launches); they stay printed and read by no rule.")
    out.append("OPTION B (as asked): the STANCE7 NOTE drafts above, appended before the F2-START lines; harmless to every "
               "rule (0 notes) but each is printed again among the scorer's other-form lines.")
    out.append("NEVER: a restatement as LAUNCH, CHECK or SNAPSHOT (C1 counts LAUNCH/RELAUNCH lines and needs exactly one; "
               "complete_r10round.py line 283 reads CHECK lines with a behavior field).")
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[-8:]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
