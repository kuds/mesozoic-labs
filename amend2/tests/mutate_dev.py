"""AMENDMENT 2: mutants of score_pc_dev.py; each must be killed by test_score_pc_dev.py (pytest exit != 0).

    cd ST/amend2/tests && PY -I -B mutate_dev.py ST/amend2/fix/mutants

Each mutant is one exact string replacement (found exactly once) in a copy of the tool written under the given folder;
the tests run against the copy (AMEND2_DEV_TOOL) with -x.  The unmutated tool runs first and must pass.  M1-M13 are the
build's mutants; R1-R8 the reviewer's (review/rt/own_mutants.py, adapted to the fixed code where their text moved);
N1-N9 mutants of the review fixes.  Disk: pytest keeps only a failed test's folder (tmp_path_retention_policy=failed);
after each mutant its log is kept and its tool copy (sha256 logged first), test folders and TMPDIR are deleted.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "score_pc_dev.py"

MUTANTS = [
    ("M1 drop every SNAPSHOT line (any arm, any time)",
     'n["kind"] == "SNAPSHOT" and C.arm_of(n["run"]) == "pc"\n            and last_off is not None and n["time"] <= last_off]',
     'n["kind"] == "SNAPSHOT"]'),
    ("M2 drop none (the deviation run reads RUNLOG as read; the copy's self-check skipped too)",
     'dev.write_bytes(build_deviation_copy(lines, drop_lines).encode("utf-8"))\n'
     '        check_deviation_copy(C, asread, dev, drop_lines, notes)',
     'dev.write_bytes(data)'),
    ("M3 drop the SNAPSHOT lines after the last OFF launch instead",
     'n["time"] <= last_off]', 'n["time"] > last_off]'),
    ("M4 drop only SNAPSHOT lines strictly earlier than the last OFF launch (a same-minute one kept)",
     'n["time"] <= last_off]', 'n["time"] < last_off]'),
    ("M5 skip the sha256 check of the dropped set",
     "if got != sorted(want):", "if False:"),
    ("M6 accept an empty dropped set under the default expectation",
     "if got != sorted(want):", "if got and got != sorted(want):"),
    ("M7 print the pinned run's lines as the deviation lines",
     'sel = deviation_lines(dv["stdout"])', 'sel = deviation_lines(pin["stdout"])'),
    ("M8 skip the pinned tools' sha256 verification",
     "verified = verify_tools(tools)", 'verified = dict(hashes={"score_pc.py": "not verified"}, closure=[])'),
    ("M9 delete the dropped line instead of blanking it (renumbers later lines; the copy's self-check skipped)",
     'out.append(ln[len(line_body(ln)):] if i in drop_lines else ln)', 'out.append("" if i in drop_lines else ln)'),
    ("M10 the deviation run reads the unfiltered copy",
     '{"pinned": asread, "deviation": dev}', '{"pinned": asread, "deviation": asread}'),
    ("M11 the deviation run's JSON written into --round",
     '"--json", str(out / f"{name}.json")]',
     '"--json", str((a.round if name == "deviation" else out) / f"{name}.json")]'),
    ("M12 the --out folder check skipped",
     "check_out(out, [a.round, tools, PINNED_TOOLS, PINNED_ROUND, runlog.parent])", "pass"),
    ("M13 the pinned run's exit code replaced by 0",
     'return pin["rc"]', "return 0"),
    # the reviewer's mutants (R1-R4 survived the build's suite; R5-R8 were defined there, results not on record)
    ("R1 OFF RELAUNCH lines ignored when finding the last OFF launch",
     'n["kind"] in ("LAUNCH", "RELAUNCH") and C.arm_of(n["run"]) == "off"]',
     'n["kind"] == "LAUNCH" and C.arm_of(n["run"]) == "off"]'),
    ("R2 no JSON difference is ever UNEXPECTED",
     '    return "UNEXPECTED"\n', '    return "blindness"\n'),
    ("R3 no output-line difference is ever UNEXPECTED",
     '    return "UNEXPECTED (not a blindness line)"', '    return "blindness: the seed\'s OFF run blind or not"'),
    ("R4 containment always reported unchanged",
     "changed = {d: sorted(set(before[d].items()) ^ set(after[d].items())) for d in before}",
     "changed = {d: [] for d in before}"),
    ("R5 the load-closure check skipped",
     '        if f"{m}.py" not in got:\n            raise Refused',
     '        if False:\n            raise Refused'),
    ("R6 the decision line's sha256 never checked",
     'ok = sha256_bytes(body.encode("utf-8")) == DECISION_LINE_SHA256', "ok = True"),
    ("R7 the deviation copy's self-check removed",
     "        check_deviation_copy(C, asread, dev, drop_lines, notes)\n", ""),
    ("R8 a SNAPSHOT of any arm dropped (arm test removed)",
     'n["kind"] == "SNAPSHOT" and C.arm_of(n["run"]) == "pc"', 'n["kind"] == "SNAPSHOT"'),
    # mutants of the review fixes
    ("N1 M1: --expect-dropped accepted on the round's RUNLOG and the pinned round",
     "        check_expect_dropped(a.expect_dropped, runlog_arg, round_arg, Path(C.RUNLOG))\n", ""),
    ("N2 M1: --expect-dropped accepted on a RUNLOG holding the decision line",
     "        check_no_decision_line(a.expect_dropped, lines)\n", ""),
    ("N3 M2: a deviation run that still reads an OFF run NOT BLIND is printed",
     '        check_deviation_blind(out / "deviation.json")\n', ""),
    ("N4 M3: PRESENCE p1's printed copy of C1's listing never accepted",
     """P1_LINE_RE = re.compile(r'\\s+p1: \\S+\\s+\\{"launches": ')""", "P1_LINE_RE = re.compile(r'(?!)')"),
    ("N5 M3: the PRESENCE / D-16 JSON copies of C1's listing never accepted",
     r'(completeness\.C1|presence\[\d+\]\.p1|d16\.presence\[\d+\]\.p1)', r'(completeness\.C1)'),
    ("N6 M3: any JSON value under a dropped line's listing path accepted (value check removed)",
     'int(m.group(2)) in dropped_lines and x == "SNAPSHOT" and y is ABSENT:', "int(m.group(2)) in dropped_lines:"),
    ("N7 M3: any change of a C1 / p1 line holding a dropped line's entry accepted (one-entry check removed)",
     'if e in x and x.replace(e, "", 1) == y:', "if e in x:"),
    ("N8 M5: a change under --round or --tools printed but not refused",
     "        if moved:\n", "        if False:\n"),
    ("N9 M3: an absent JSON key and a null value compared equal (as the reviewed code did)",
     "            if a.get(k, ABSENT) != b.get(k, ABSENT)]", "            if a.get(k) != b.get(k)]"),
]


def mutants():
    src = TOOL.read_text(encoding="utf-8")
    out = []
    for name, old, new in MUTANTS:
        assert src.count(old) == 1, (name, src.count(old))
        m = src.replace(old, new)
        if name.startswith("M9 "):
            c = "        check_deviation_copy(C, asread, dev, drop_lines, notes)\n"
            assert m.count(c) == 1
            m = m.replace(c, "")
        out.append((name, m))
    return out


def run(dev: Path, base: Path) -> tuple[int, str]:
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / "tmp"
    tmp.mkdir(exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp), "LANG": "C.UTF-8", "AMEND2_DEV_TOOL": str(dev)}
    p = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                        "-o", "tmp_path_retention_policy=failed", f"--basetemp={base / 'pt'}",
                        str(HERE / "test_score_pc_dev.py")],
                       capture_output=True, text=True, cwd=str(HERE), env=env)
    return p.returncode, p.stdout + p.stderr


def clean(base: Path, dev: Path | None) -> str:
    """Delete a mutant's tool copy, test folders and TMPDIR once its log is written; returns what was deleted."""
    gone = []
    if dev is not None and dev.exists():
        gone.append(f"{dev.name} (sha256 {hashlib.sha256(dev.read_bytes()).hexdigest()})")
        dev.unlink()
    for d in (base / "pt", base / "tmp"):
        if d.exists():
            n = sum(1 for _ in d.rglob("*"))
            shutil.rmtree(d)
            gone.append(f"{d.name}/ ({n} entries)")
    return "; ".join(gone) or "nothing"


def main(argv):
    root = Path(argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rc, log = run(TOOL, root / "M0_unmutated")
    (root / "M0_unmutated" / "pytest.log").write_text(log)
    tail = log.strip().splitlines()[-1] if log.strip() else ""
    print(f"M0 unmutated: pytest exit {rc} ({tail}); deleted: {clean(root / 'M0_unmutated', None)}", flush=True)
    if rc != 0:
        print("the unmutated tool does not pass: no mutant is judged")
        return 1
    survived = 0
    for name, text in mutants():
        d = root / name.split()[0]
        d.mkdir(parents=True, exist_ok=True)
        dev = d / "score_pc_dev.py"
        dev.write_text(text, encoding="utf-8")
        rc, log = run(dev, d)
        (d / "pytest.log").write_text(log)
        failed = [x for x in log.splitlines() if x.startswith(("FAILED ", "ERROR "))]
        killed = rc != 0
        survived += not killed
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} (pytest exit {rc}; "
              f"{failed[0].split(' ', 1)[1].split(' - ')[0] if failed else 'no FAILED line'}); deleted: {clean(d, dev)}",
              flush=True)
    print(f"{len(mutants())} mutants, {survived} survived")
    return 1 if survived else 0


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.exit(main(sys.argv))
