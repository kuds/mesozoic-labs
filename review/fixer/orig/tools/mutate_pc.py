"""PC mutation tests: each mutant changes one PC rule in a copy of ST/tools/{pc,tests} (exact string, must occur once);
the PC test suite (tests/test_score_pc.py) must fail on every mutant.  Copies under --work (deleted after); the log
to --log.  Report only.

    PY -I -B mutate_pc.py --work DIR --log FILE
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

T = Path(__file__).resolve().parent
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"

MUTANTS = [
    ("M1 ties count for H1", "pc_common.py", "        k += tot <= obs\n", "        k += tot < obs\n"),
    ("M2 LOWER read at 2/20", "pc_common.py", 'if k == 1 else f"NOT SEPARATED', 'if k <= 2 else f"NOT SEPARATED'),
    ("M3 line 4 ignores hs", "pc_common.py", "    elif g == 3 and hs == 0:\n", "    elif g == 3:\n"),
    ("M4 line 2 needs u > 1", "pc_common.py", "    elif u >= 1:\n        line, head = 2", "    elif u > 1:\n        line, head = 2"),
    ("M5 own alarm needs 6 episodes", "pc_common.py", "flagged = sorted(s for s, k in ev.items() if k >= 5)",
     "flagged = sorted(s for s, k in ev.items() if k > 5)"),
    ("M6 T ignores ties", "pc_common.py",
     't_at = [lab for lab, v in m1.items() if (v["over_bar_full_plus"] + v["over_bar_full_tie"]) >= 2',
     't_at = [lab for lab, v in m1.items() if (v["over_bar_full_plus"]) >= 2'),
    ("M7 A does not saturate", "pc_common.py", 'status = ("SATURATING" if any(s in sites for s in ("L", "A", "T"))',
     'status = ("SATURATING" if any(s in sites for s in ("L", "T"))'),
    ("M8 TD signed", "pc_common.py", "used = [(s, abs(e[\"cmd_l\"])) for s, e in per.items()",
     "used = [(s, e[\"cmd_l\"]) for s, e in per.items()"),
    ("M9 first_t3_over counts tail_2", "pc_common.py",
     'n_t3_over=sum(1 for r in over if r.get("saturated_actuator") == T3)',
     'n_t3_over=sum(1 for r in over if r.get("saturated_actuator") == "tail_2_pitch_act")'),
    ("M10 D mean of R1/R2 dropped", "pc_common.py",
     "            if mean is not None and sg * mean >= HABIT_LINE:\n",
     "            if False:\n"),
    ("M11 C8 NOT IN FORCE not a failure", "complete_r10round.py", "    elif c8_failed:\n", "    elif False:\n"),
    ("M12 C8 binding not checked", "complete_r10round.py",
     "            bad = binding_problems(C.record_binding(rec), lab, files, E, run_id)\n",
     "            bad = []\n"),
    ("M13 switch read loosely", "complete_r10round.py",
     '        ok = isinstance(got, bool) and got is want["value"]\n', '        ok = bool(got) == want["value"]\n'),
    ("M14 C4 (p) dropped", "complete_r10round.py",
     '        out.append(critic_phase_fact("C4 (p) stage_config.json records the critic-phase switch ON", sc, pins))   # PC\n',
     ""),
    ("M15 fact (8) not a stop fault", "r10round_common.py", "STOP_FAULTS = (1, 2, 3, 4, 5, 7, 8)", "STOP_FAULTS = (1, 2, 3, 4, 5, 7)"),
    ("M16 a missing run enters the test at its TD", "score_pc.py",
     '                                    and x["td"]["td"] is not None) else math.inf) for x in per_seed]',
     '                                    and x["td"]["td"] is not None) else 0.0) for x in per_seed]'),
    ("M17 C8 seeds not checked", "complete_r10round.py",
     "        if list(rec.get(\"seeds\") or []) != list(C.MANIP_SEEDS) or rec.get(\"bank_limit\") != C.MANIP_BANK_LIMIT:\n",
     "        if False:\n"),
    ("M18 validations not PROVISIONAL", "score_pc.py", "        if val.get(v) != \"PASS\":\n", "        if False:\n"),
    ("M19 C8 absent counted as PASS", "complete_r10round.py",
     "        out[\"C8\"] = fact(c8_name, True if not c8_missing else None,",
     "        out[\"C8\"] = fact(c8_name, True,"),
    ("M21 F2_PINS sha256 not looked up in RUNLOG", "score_pc.py", "    elif f2p_sha not in runlog_text:\n",
     "    elif False:\n"),
    ("M22 VP1/VP4 not read", "score_pc.py", "    for v in PRE_F2_VALIDATIONS:\n", "    for v in ():\n"),
    ("M20 line-1 basis ignores the ledger", "pc_common.py", '        if "ledger" in basis:\n            b = BASIS_LEDGER\n',
     '        if False:\n            b = BASIS_LEDGER\n'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    if work.exists():
        print(f"REFUSED: {work} exists")
        return 2
    log = []
    killed = 0
    for i, (name, fname, old, new) in enumerate(MUTANTS, 1):
        d = work / f"m{i}"
        shutil.copytree(T / "pc", d / "pc")
        shutil.copytree(T / "tests", d / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        f = d / "pc" / fname
        src = f.read_text()
        n = src.count(old)
        if n != 1:
            log.append(f"{name}: NOT APPLIED (old text occurs {n} times)")
            continue
        f.write_text(src.replace(old, new))
        t0 = time.time()
        p = subprocess.run(["nice", "-n", "10", PY, "-I", "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            str(d / "tests" / "test_score_pc.py"), f"--basetemp={d / 'tmp'}"],
                           capture_output=True, text=True, cwd="/tmp")
        tail = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:] or [""]
        failed_test = [ln for ln in p.stdout.splitlines() if ln.startswith("FAILED")][:1]
        ok = p.returncode != 0
        killed += ok
        log.append(f"{name} ({fname}): {'KILLED' if ok else 'SURVIVED'} (exit {p.returncode}, {time.time() - t0:.1f} s; "
                   f"{tail[0]}; {failed_test[0] if failed_test else ''})")
        shutil.rmtree(d)
    log.append(f"KILLED {killed} of {len(MUTANTS)}")
    Path(a.log).write_text("\n".join(log) + "\n")
    print("\n".join(log))
    shutil.rmtree(work, ignore_errors=True)
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
