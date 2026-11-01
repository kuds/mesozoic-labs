"""Second re-review: run mutate_pc.py's M45-M49 only.  Per mutant: (1) the new test that pins it, alone; (2) the full
test_score_pc.py with -x.  Copies under --work (inside ST/amend/rereview2, deleted after); also (0) each new test on an
unmutated copy.  Report only.   PY -I -B m45_49.py --work DIR --log FILE"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path("/home/user/work/pc_study/tools")
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"
OWN_TEST = {
    "M45": "test_root_blindness_an_off_snapshot_is_no_opening",
    "M46": "test_root_blindness_an_off_launch_in_the_opening_minute_is_not_blind",
    "M47": "test_c8_off_arm_reads_the_stock_critic_why",
    "M48": "test_launch_check_cli_refuses_an_unfilled_switch_pin_per_arm",
    "M49": "test_root_blindness_reads_the_off_seeds_relaunch",
}


def load_mutants():
    spec = importlib.util.spec_from_file_location("mutate_pc_copy", TOOLS / "mutate_pc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MUTANTS


def pytest(d: Path, sel: str | None, tag: str):
    args = [PY, "-I", "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
            str(d / "tests" / "test_score_pc.py"), f"--basetemp={d / ('tmp_' + tag)}"]
    if sel:
        args[-2] = str(d / "tests" / "test_score_pc.py") + "::" + sel
    p = subprocess.run(["nice", "-n", "10", *args], capture_output=True, text=True, cwd=str(d))
    last = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:] or [""]
    failed = [ln for ln in p.stdout.splitlines() if ln.startswith("FAILED")][:1]
    return p.returncode, last[0], failed[0] if failed else ""


def fresh(work: Path, name: str) -> Path:
    d = work / name
    shutil.copytree(TOOLS / "pc", d / "pc", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(TOOLS / "tests", d / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    if work.exists():
        print(f"REFUSED: {work} exists")
        return 2
    out = []
    muts = [m for m in load_mutants() if m[0].split()[0] in OWN_TEST]
    out.append(f"mutants selected: {[m[0].split()[0] for m in muts]}")
    d = fresh(work, "clean")
    for mid, t in OWN_TEST.items():
        rc, last, _f = pytest(d, t, "c_" + mid)
        out.append(f"UNMUTATED {mid} own test {t}: exit {rc} ({last})")
    shutil.rmtree(d)
    bad = 0
    for name, fname, old, new in muts:
        mid = name.split()[0]
        d = fresh(work, mid)
        f = d / "pc" / fname
        src = f.read_text()
        n = src.count(old)
        if n != 1:
            out.append(f"{name}: NOT APPLIED ({n} occurrences)")
            bad += 1
            shutil.rmtree(d)
            continue
        f.write_text(src.replace(old, new))
        rc1, last1, f1 = pytest(d, OWN_TEST[mid], "own")
        rc2, last2, f2 = pytest(d, None, "full")
        ok = rc1 != 0 and rc2 != 0
        bad += not ok
        out.append(f"{name} ({fname}): own test {'KILLS' if rc1 else 'MISSES'} (exit {rc1}; {last1}); "
                   f"full suite {'KILLED' if rc2 else 'SURVIVED'} (exit {rc2}; {last2}; {f2})")
        shutil.rmtree(d)
    out.append(f"RESULT: {len(muts) - bad} of {len(muts)} killed by their own new test and by the suite")
    Path(a.log).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    shutil.rmtree(work, ignore_errors=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
