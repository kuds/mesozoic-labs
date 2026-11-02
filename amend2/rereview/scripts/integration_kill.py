"""Re-review: the reviewer's surviving mutants R2, R3 (classifier never UNEXPECTED) and R4 (containment always
unchanged), each run against the end-to-end tests only (not the classifier unit tests): main() with stubbed scorer
outputs, and the two M5 tests.  Mutant copies are written under argv[1] and deleted after each run."""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

TOOL = Path("/home/user/work/pc_study/amend2/tools/score_pc_dev.py")
TESTS = Path("/home/user/work/pc_study/amend2/tests")
MUT = [
    ("R2", '    return "UNEXPECTED"\n', '    return "blindness"\n',
     ["test_r2_r3_unexpected_differences_are_flagged_and_counted"]),
    ("R3", '    return "UNEXPECTED (not a blindness line)"', '    return "blindness: the seed\'s OFF run blind or not"',
     ["test_r2_r3_unexpected_differences_are_flagged_and_counted"]),
    ("R2+R3", None, None, ["test_r2_r3_unexpected_differences_are_flagged_and_counted"]),
    ("R4", "changed = {d: sorted(set(before[d].items()) ^ set(after[d].items())) for d in before}",
     "changed = {d: [] for d in before}",
     ["test_m5_refuses_when_a_file_under_round_or_tools_changes_during_the_runs"]),
]
root = Path(sys.argv[1])
src = TOOL.read_text(encoding="utf-8")
for name, old, new, tests in MUT:
    if name == "R2+R3":
        m = src.replace(MUT[0][1], MUT[0][2], 1).replace(MUT[1][1], MUT[1][2], 1)
    else:
        assert src.count(old) == 1, name
        m = src.replace(old, new, 1)
    d = root / name
    d.mkdir(parents=True)
    dev = d / "score_pc_dev.py"
    dev.write_text(m, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(d), "LANG": "C.UTF-8", "AMEND2_DEV_TOOL": str(dev)}
    p = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
                        "tmp_path_retention_policy=none", f"--basetemp={d / 'pt'}", "-k", " or ".join(tests),
                        str(TESTS / "test_score_pc_dev.py")], capture_output=True, text=True, cwd=str(TESTS), env=env)
    lines = [x for x in (p.stdout + p.stderr).splitlines() if x.startswith(("FAILED", "PASSED", "ERROR")) or
             " passed" in x or " failed" in x]
    print(f"{name} (mutant sha256 {hashlib.sha256(m.encode()).hexdigest()[:16]}), tests {tests}: pytest exit "
          f"{p.returncode} -> {'KILLED' if p.returncode else 'SURVIVED'}; " + " | ".join(x[:150] for x in lines[-4:]))
    shutil.rmtree(d)
