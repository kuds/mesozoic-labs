"""AMENDMENT 2: mutants of score_pc_dev.py; each must be killed by test_score_pc_dev.py (pytest exit != 0).

    cd ST/amend2/tests && PY -I -B mutate_dev.py ST/amend2/dry/mutants

Each mutant is one exact string replacement (found exactly once) in a copy of the tool written under the given folder;
the tests run against the copy (AMEND2_DEV_TOOL) with -x.  The unmutated tool runs first and must pass.
"""
from __future__ import annotations

import os
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
                        f"--basetemp={base / 'pt'}", str(HERE / "test_score_pc_dev.py")],
                       capture_output=True, text=True, cwd=str(HERE), env=env)
    return p.returncode, p.stdout + p.stderr


def main(argv):
    root = Path(argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rc, log = run(TOOL, root / "M0_unmutated")
    (root / "M0_unmutated" / "pytest.log").write_text(log)
    tail = log.strip().splitlines()[-1] if log.strip() else ""
    print(f"M0 unmutated: pytest exit {rc} ({tail})")
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
        failed = [x for x in log.splitlines() if x.startswith("FAILED ")]
        killed = rc != 0
        survived += not killed
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} (pytest exit {rc}; "
              f"{failed[0][len('FAILED '):].split(' - ')[0] if failed else 'no FAILED line'})")
    print(f"{len(mutants())} mutants, {survived} survived")
    return 1 if survived else 0


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.exit(main(sys.argv))
