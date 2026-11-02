"""Reviewer check of ST/f2prep/restatements.draft.txt against RUNLOG (read only): for each DRAFT line, the text after
the draft's own header '...): ' must equal the RUNLOG line's own words (after its leading '<date> <time> UTC' and, for a
STANCE7 line, after 'STANCE7 NOTE '), with '<PC branch>' read back as the branch name; prints every difference.
Usage: PY -I -B check_restatements.py RUNLOG DRAFTS"""
import difflib
import json
import re
import sys

BR = json.load(open("/home/user/work/pc_study/round/APPROVED", encoding="utf-8"))["branch"]   # read, not typed
rl = open(sys.argv[1], encoding="utf-8").read().splitlines()
txt = open(sys.argv[2], encoding="utf-8").read().splitlines()
drafts = [l for l in txt if l.startswith("  DRAFT: ")]
print("drafts", len(drafts))
for d in drafts:
    m = re.match(r"  DRAFT: (\S+ \S+) UTC  STANCE7 NOTE restates RUNLOG line (\d+) \((.*)", d)
    n = int(m.group(2))
    orig = rl[n - 1]
    # header ends at the first '): ' after which the verbatim part begins
    rest = m.group(3)
    # find the split '): ' that closes the parenthesised header (balance parentheses)
    depth, cut = 1, None
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                cut = i
                break
    body = rest[cut + 3:]
    body_b = body.replace("<PC branch>", BR)
    om = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?) UTC\s+(.*)$", orig)
    otime = om.group(1) if om else None
    owords = om.group(2) if om else orig
    owords2 = re.sub(r"^STANCE7 NOTE ", "", owords)
    t_ok = (otime == m.group(1)) if otime else None
    same = body_b in (owords, owords2)
    print(f"line {n}: time {'equal' if t_ok else ('NO OWN TIME' if t_ok is None else 'DIFFERS ' + str(otime) + ' vs ' + m.group(1))}; "
          f"words {'VERBATIM' if same else 'DIFFER'}; branch placeholder {'yes' if '<PC branch>' in body else 'no'}")
    if not same:
        a, b = owords2, body_b
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                print(f"   {op}: orig[{i1}:{i2}]={a[i1:i2][:300]!r} -> draft[{j1}:{j2}]={b[j1:j2][:300]!r}")
