"""PC mutation tests: each mutant changes one PC rule in a copy of ST/tools/{pc,tests} (exact string, must occur once);
the PC test suite (tests/test_score_pc.py) must fail on every mutant.  Copies under --work (deleted after); the log
to --log.  Report only.
PC AMENDMENT (2026-11-01): M2, M14, M16, M28 and M31 re-pointed to the amended text of the same rule; M32-M44 the
amendment's rules (the one test against the concurrent switch-off runs, MISSING ranks, the switch-off arm's pins, C4 (p)
and C8, outcome blindness, the line read from the phase runs only).

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
    ("M2 LOWER read at 2/20", "pc_common.py", '(exact permutation p = 1/20)" if k == 1\n',
     '(exact permutation p = 1/20)" if k <= 2\n'),
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
     '        out.append(critic_phase_fact(switch_names(run_id)[1], sc, pins, run_id))   # PC (AMENDMENT: per arm)\n',
     ""),
    ("M15 fact (8) not a stop fault", "r10round_common.py", "STOP_FAULTS = (1, 2, 3, 4, 5, 7, 8)", "STOP_FAULTS = (1, 2, 3, 4, 5, 7)"),
    ("M16 a missing run enters the test at its TD", "score_pc.py",
     '        phase_td = [test_value(x, math.inf) for x in per_arm["pc"]]',
     '        phase_td = [test_value(x, 0.0) for x in per_arm["pc"]]'),
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
    # ---- fixer (2026-11-01): the review's surviving mutants R4, R7, R11, R12 and the fixed rules
    ("M23 (R4) line 3 checked before line 2", "pc_common.py",
     "    elif u >= 1:\n        line, head = 2, LINE_HEAD[2].format(read=3 - u, u=u)\n    elif p >= 1:\n"
     "        line, head = 3, LINE_HEAD[3].format(p=p)\n",
     "    elif p >= 1:\n        line, head = 3, LINE_HEAD[3].format(p=p)\n    elif u >= 1:\n"
     "        line, head = 2, LINE_HEAD[2].format(read=3 - u, u=u)\n"),
    ("M24 (R7) PRESENCE basis checked before the ledger", "pc_common.py",
     '        if "ledger" in basis:\n            b = BASIS_LEDGER\n        elif presence_rails:\n            b = BASIS_LOST\n',
     '        if presence_rails:\n            b = BASIS_LOST\n        elif "ledger" in basis:\n            b = BASIS_LEDGER\n'),
    ("M25 (R11) T read at R1/final only", "pc_common.py",
     '        t_at = [lab for lab, v in m1.items() if (v["over_bar_full_plus"]',
     '        t_at = [lab for lab, v in m1.items() if lab in ("final", str(r1)) and (v["over_bar_full_plus"]'),
    ("M26 (R12) a STOPPED COMPLETE run enters at its own TD", "score_pc.py",
     'x.get("verdict") == "COMPLETE" and x.get("finished") and x.get("td")',
     'x.get("verdict") == "COMPLETE" and x.get("td")'),
    ("M27 (D3) A without the 5-row threshold", "pc_common.py",
     '\n                     and x.get("step") is not None\n'
     '                     and max(ev.get(x["step"] - C.GRID, 0), ev.get(x["step"], 0)) >= A_SITE_MIN_ROWS]',
     ']'),
    ("M28 (I1) sums by math.fsum again", "pc_common.py",
     "        return (n_inf, sum((Fraction(v) for v in xs if math.isfinite(v)), Fraction(0)))",
     "        return (n_inf, math.fsum(v for v in xs if math.isfinite(v)))"),
    ("M29 (D4) a DRIVEN tail not printed", "pc_common.py",
     '    return f", WITH A DRIVEN TAIL ({\'; \'.join(driven)})" if driven else ""', '    return ""'),
    ("M30 (D5) line 4 without the gate label", "pc_common.py",
     '       "(GATE CERTIFICATE: final checkpoint, measured on a screen task; not certification), NO SATURATING HABIT ON "',
     '       "NO SATURATING HABIT ON "'),
    ("M31 (D5) the no-rail chance printed on lines 1-2", "pc_common.py",
     "no_rail_chance=_no_rail_chance(off) if line in (3, 4, 5) else None", "no_rail_chance=_no_rail_chance(off)"),
    # ---- PC AMENDMENT (2026-11-01): six runs, the one test against the concurrent switch-off runs
    ("M32 a missing switch-off run enters at 0, not the lowest rank", "score_pc.py",
     '        off_td = [(-math.inf if x["run"] in nb else test_value(x, -math.inf)) for x in per_arm["off"]]',
     '        off_td = [(-math.inf if x["run"] in nb else test_value(x, 0.0)) for x in per_arm["off"]]'),
    ("M33 the one test compares with STANCE6, not the switch-off runs", "score_pc.py",
     '        rd["one_test"] = P.one_test(phase_td, off_td)',
     '        rd["one_test"] = P.one_test(phase_td, [comparators["runs"][s]["td"]["td"] for s in C.SEEDS])'),
    ("M34 the switch-off C8 reads only the verdict", "complete_r10round.py",
     '        return ("spec" in rec and rec["spec"] is None and rec.get("verdict") == "NOT IN FORCE"',
     '        return (rec.get("verdict") == "NOT IN FORCE"'),
    ("M35 a switch-off run read against the phase pins", "complete_r10round.py",
     '    if C.arm_of(run_id) == "off":\n        blk = C.pin(pins, "off_arm")', '    if False:\n        blk = C.pin(pins, "off_arm")'),
    ("M36 the switch-off pin's absence not read", "complete_r10round.py",
     '    if isinstance(want, dict) and "path" in want and want.get("absent") is True:', '    if False:'),
    ("M37 outcome blindness not applied", "score_pc.py",
     '        blind = None if lo is None else (first is None or lo["time"] < first["time"])',
     '        blind = None if lo is None else True'),
    ("M38 a SNAPSHOT is no opening", "score_pc.py", 'OPENING_KINDS = ("F2-START", "REFETCH-START", "SNAPSHOT")',
     'OPENING_KINDS = ("F2-START", "REFETCH-START")'),
    ("M39 the line read from the switch-off runs", "score_pc.py",
     '        rd = P.reading(per_arm["pc"], presence_rails["pc"], per_arm["off"], presence_rails["off"])',
     '        rd = P.reading(per_arm["off"], presence_rails["off"], per_arm["off"], presence_rails["off"])'),
    ("M40 missing ranks dropped from the sums", "pc_common.py",
     '        n_inf = sum(1 for v in xs if v == math.inf) - sum(1 for v in xs if v == -math.inf)',
     '        n_inf = 0'),
    ("M41 the no-rail chance not at the switch-off rate", "pc_common.py",
     "no_rail_chance=_no_rail_chance(off) if line in (3, 4, 5) else None",
     "no_rail_chance=_no_rail_chance(None) if line in (3, 4, 5) else None"),
    ("M42 seed_run ignores the arm", "score_pc.py", '    first = f"{C.ARM_PREFIX[arm]}{seed}"',
     '    first = f"{C.RUN_ID_PREFIX}{seed}"'),
    ("M43 STANCE6 unreadable holds the reading open", "score_pc.py",
     '    out["comparators_error"] = comp_err\n',
     '    out["comparators_error"] = comp_err\n    if comparators is None:\n        out["incomplete"].append("comparators")\n'),
    ("M44 the switch-off arm's token not read in RUNLOG", "r10round_common.py",
     'RUN_TOKEN_RE = re.compile(r"stance7_r10LTSB(?:pc|off)_s\\d+\\w*")',
     'RUN_TOKEN_RE = re.compile(r"stance7_r10LTSBpc_s\\d+\\w*")'),
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
                           capture_output=True, text=True, cwd=str(d))
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
