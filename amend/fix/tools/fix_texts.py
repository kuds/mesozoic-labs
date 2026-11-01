"""FIXER of the PC amendment (2026-11-01, after the amendment's re-review): the text fixes for B1 (THE BLIND PERIOD),
M2 (DESIGN_NOTES section 9: the waves and the arm-wave confound), I1 (one SNAPSHOT line per RUN_ID) and I4 (the
reading formed once both arms' runs are decided), and the record of the fixer's re-runs.  Exact replacements, each
old text occurring exactly once (else STOP, nothing written).  Every application starts from the re-reviewed
amendment bytes in ST/amend/fix/rereviewed/ (their sha256 checked = ST/amend/after.txt), so a re-application
reproduces the file.  criteria.txt's PINS block is refilled afterwards by ST/amend/tools/refill_pins.py (from files).

    PY -I -B fix_texts.py colab | design | criteria [--check]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ST = Path("/home/user/work/pc_study")
SRC = ST / "amend/fix/rereviewed"
BEFORE = {  # the re-reviewed amendment's bytes (ST/amend/after.txt)
    "criteria.txt": "ed9d2f07762645b7e3a85c70108a6f7086c7b1f1bc9d2dcf959a477eca41478f",
    "COLAB_STEPS.txt": "2dc68023aadbcb924e7690f4785424d7bc1f5364a54f78b1bd38607e8bc9494e",
    "design/DESIGN_NOTES.txt": "89a9ad4681c8035733bf5a1178c11c9666f3f745c794781afaa158dc3fea0780",
}

COLAB = [
 # ---------------------------------------------------------------- header
 ("""switch-off runs), the order, fewer sessions, outcome blindness, the per-run _b rule, the SNAPSHOT line. Every change:
ST/amend/AMEND_NOTES.txt and ST/amend/diffs/COLAB_STEPS.txt.diff.
""",
  """switch-off runs), the order, fewer sessions, outcome blindness, the per-run _b rule, the SNAPSHOT line. Every change:
ST/amend/AMEND_NOTES.txt and ST/amend/diffs/COLAB_STEPS.txt.diff.
FIXED 2026-11-01 (date -u) by the amendment's fixer after its re-review (ST/amend/FIXES.txt): THE BLIND PERIOD (until
every OFF run's LAUNCH line, a PHASE run's liveness, end and stall from metadata only; re-review B1), the maintainer's
"Phase first" waves (RUNLOG line 876), one SNAPSHOT line per RUN_ID. Every change: ST/amend/fix/diffs/.
"""),
 # ---------------------------------------------------------------- THE ORDER: the waves (RUNLOG line 876)
 ("""   releases its runtime: AUTO_DISCONNECT), in seed order. With three sessions this is two waves: wave 1 the three PHASE
   runs, wave 2 the three OFF runs as wave 1's runtimes free (the arm is then confounded with the wave: criteria KNOWN
   LIMITS). Nothing about a run's outcome decides when or whether an OFF run starts: every OFF run is started.
""",
  """   releases its runtime: AUTO_DISCONNECT), in seed order. With three sessions this is two waves, the maintainer's plan
   (RUNLOG line 876, 01:41:53 UTC: "Phase first"): wave 1 the three PHASE runs, wave 2 the three OFF runs as wave 1's
   runtimes free (the arm is then confounded with the wave: criteria KNOWN LIMITS); wave 1 runs wholly in THE BLIND
   PERIOD. Nothing about a run's outcome decides when or whether an OFF run starts: every OFF run is started.
"""),
 # ---------------------------------------------------------------- THE ORDER: outcome blindness (B1)
 (""" - OUTCOME BLINDNESS: every OFF run starts before any PHASE run's results are opened. Until every OFF run's LAUNCH line
   is in RUNLOG: read no PHASE run's training output beyond the check rows of step 7 (liveness checks by Runtime >
   Manage sessions only), make no report-only snapshot of any run, and no F2 starts (it cannot: F2 waits for every run's
   end). After that, a report-only snapshot at the maintainer's request is noted at once as a SNAPSHOT line (RUNLOG
   NOTES). An OFF run whose LAUNCH line is not earlier than the first F2-START, REFETCH-START or SNAPSHOT line of a
   PHASE run is not blind: the scorer prints its classes and enters it in THE ONE TEST as missing (the lowest rank).
""",
  """ - OUTCOME BLINDNESS: every OFF run starts before any PHASE run's results are opened. Until every OFF run's LAUNCH line
   is in RUNLOG (THE BLIND PERIOD): read no PHASE run's training output beyond the check rows of step 7, make no
   report-only snapshot of any run, and no F2 starts (it cannot: F2 waits for every run's end). In THE BLIND PERIOD a
   PHASE run's liveness, end and stall rest on metadata only (THE ENDING PROCEDURE, IN THE BLIND PERIOD): Runtime >
   Manage sessions, and Drive listings' names, sizes and modifiedTime. Its progress.json and other run files (beyond
   the launch check's F0 copies, which only the launch-check tool reads), its notebook output and its downloaded
   notebook are not opened. Anything more is an opening: note it at once as a SNAPSHOT line of that PHASE RUN_ID;
   every OFF run launched after it is not blind. After THE BLIND PERIOD, a report-only snapshot at the maintainer's
   request is noted at once as SNAPSHOT lines (RUNLOG NOTES: one per RUN_ID looked at). An OFF run whose LAUNCH line
   is not earlier than the first F2-START, REFETCH-START or SNAPSHOT line of a PHASE run is not blind: the scorer
   prints its classes and enters it in THE ONE TEST as missing (the lowest rank).
"""),
 # ---------------------------------------------------------------- step 8
 (""" 8. Leave it running: about 14.1 h to "ended" and about 14.9 h to its manifest (stance6). It disconnects by itself
    at the end (AUTO_DISCONNECT True). PASS or FAIL is fine.
""",
  """ 8. Leave it running: about 14.1 h to "ended" and about 14.9 h to its manifest (stance6). It disconnects by itself
    at the end (AUTO_DISCONNECT True). PASS or FAIL is fine. A PHASE run in THE BLIND PERIOD: once its step-7 rows are
    recorded, its tab is not read again (it may stay open in the background).
"""),
 # ---------------------------------------------------------------- THE ENDING PROCEDURE (B1)
 ("""THE ENDING PROCEDURE (liveness or configuration only, never an outcome) -- as the stance6 steps, unchanged
""",
  """THE ENDING PROCEDURE (liveness or configuration only, never an outcome) -- as the stance6 steps, plus THE BLIND
PERIOD rule for a PHASE run (below)
"""),
 ("""     waited for at most 30 minutes; delete the runtime; the DELETED line.
 Suggested liveness check (not a rule): Runtime > Manage sessions every 30 min in a run's first 2 h, every 2 h after.
""",
  """     waited for at most 30 minutes; delete the runtime; the DELETED line.
 IN THE BLIND PERIOD (THE ORDER: until every OFF run's LAUNCH line is in RUNLOG), for a PHASE run, metadata only:
  - (a) from Runtime > Manage sessions, or from Drive (CHOICE ENDS) by names, sizes and modifiedTime only:
    artifact_manifest.json present, and 30 minutes with neither a progress.json rewrite (progress.json is rewritten
    after every evaluation, about every 9 minutes, and last at "ended") nor any other write in the run folder. Its
    progress.json is not opened, so "ended" is not read (a tool may print its status and timesteps fields only if no
    other field reaches a person or agent; a copy through the Drive connector does not meet this). The TRIGGER-a and
    FOUND-GONE notes say "established from metadata only (outcome blindness)".
  - (c) the STALL is 2 hours with no progress.json rewrite (its modifiedTime), not output lines. (b) needs the tab and
    is not looked for: a run stopped with its runtime still listed is found by (c) [inferred cost: a stop in the run's
    first 2 hours may then be noted after its REPLACEMENT window, so LOST instead of relaunched].
  - At any trigger, in the procedure's order: File > Download .ipynb (saved unopened until THE BLIND PERIOD ends),
    then Edit > Clear all outputs in the tab before the flush cell, reading nothing; after (a), the download likewise,
    then close the tab.
  - Anything more (a run file opened, the notebook output or the downloaded notebook read) is an opening: a SNAPSHOT
    line of that RUN_ID at once (THE ORDER).
 Suggested liveness check (not a rule): Runtime > Manage sessions every 30 min in a run's first 2 h, every 2 h after.
"""),
 # ---------------------------------------------------------------- RUNLOG NOTES: SNAPSHOT (I1)
 (""" SNAPSHOT <RUN_ID> note="<what was looked at>"   (a report-only look at a run's results, only after every OFF run's
                     LAUNCH line; read wherever it stands: outcome blindness)
""",
  """ SNAPSHOT <RUN_ID> note="<what was looked at>"   (a report-only look at a run's results, only after every OFF run's
                     LAUNCH line, or an opening in THE BLIND PERIOD noted at once; one line per RUN_ID looked at, the
                     RUN_ID exactly as THE RUNS name it: a SNAPSHOT line naming no RUN_ID is printed and read as no
                     opening; read wherever it stands: outcome blindness)
"""),
 # ---------------------------------------------------------------- WHEN TO FETCH (B1)
 (""" - TRIGGER: only at a trigger within 2 hours of a launch.
""",
  """ - TRIGGER: only at a trigger within 2 hours of a launch.
 - In THE BLIND PERIOD a PHASE run's files are fetched only by F0 (read by the launch-check tool) and listed only for
   names, sizes and modifiedTime (THE ENDING PROCEDURE; a TRIGGER LISTING is names and sizes).
"""),
]

DESIGN = [
 # ---------------------------------------------------------------- section 9: the waves (M2)
 ("""     SWITCH   as the builder made it (cell 6 CRITIC_PHASE; rows "Critic phase: on" / "Critic phase: off"; cell 9 "Critic
              phase on: ..." for a phase run only [measured, ST/check_code/nb/nb_on.log, nb_off.log]).
""",
  """     SWITCH   as the builder made it (cell 6 CRITIC_PHASE; rows "Critic phase: on" / "Critic phase: off"; cell 9 "Critic
              phase on: ..." for a phase run only [measured, ST/check_code/nb/nb_on.log, nb_off.log]).
     WAVES    answered 2026-11-01 01:41:53 UTC, after the amendment had started (RUNLOG line 876 [measured]): "I plan to
              run 3 runs at a time"; "Phase first": wave 1 the three phase runs, wave 2 the three switch-off runs when
              wave 1 ends; no phase result opened until every switch-off run has started; the arm confounded with the
              wave (for KNOWN LIMITS). THE ORDER's fewer-sessions rule is this plan; the fixer's blind-period rule (the
              amendment's re-review B1: until every switch-off LAUNCH line a phase run's liveness, end and stall rest
              on metadata only) makes "no phase result opened" followable (ST/amend/FIXES.txt).
"""),
 # ---------------------------------------------------------------- section 9: what concurrency does not remove (M2)
 ("""       differences between the test's arms. What it does not: 3 vs 3 still reaches only p = 1/20; GPU nondeterminism
       within today's image; image changes between staggered starts; seed effects (a shared start, A5) assumed to make
       the unpaired test conservative.
""",
  """       differences between the test's arms. What it does not: 3 vs 3 still reaches only p = 1/20; GPU nondeterminism
       within today's image; image changes between staggered starts; under the maintainer's "Phase first" waves
       (RUNLOG line 876) the arm is confounded with the wave, about one run's length apart (the start time, and the
       image if it changes between the waves), so the test also assumes no wave effect (criteria KNOWN LIMITS); seed
       effects (a shared start, A5) assumed to make the unpaired test conservative.
"""),
]

CRITERIA = [
 # ---------------------------------------------------------------- status
 ("""ST/amend/diffs/criteria.txt.diff. A draft for the quick re-review before the pin: nothing here is approved.
""",
  """ST/amend/diffs/criteria.txt.diff. A draft for the quick re-review before the pin: nothing here is approved.
FIXED 2026-11-01 (date -u) by the amendment's fixer after its re-review (ST/amend/FIXES.txt): the blind period (until
every OFF run's LAUNCH line, a PHASE run's liveness, end and stall from metadata only; RUN HANDLING), the maintainer's
"Phase first" waves (RUNLOG line 876), one SNAPSHOT line per RUN_ID, the reading formed once both arms' runs are
decided; five tests and five mutants for rules the suite did not hold. Every change: ST/amend/fix/diffs/.
"""),
 # ---------------------------------------------------------------- THE READING (I4)
 ("""  printed beside every line ("CONCURRENT SWITCH-OFF RUNS (printed beside; they decide no line): ...") and move none.
""",
  """  printed beside every line ("CONCURRENT SWITCH-OFF RUNS (printed beside; they decide no line): ...") and move none.
  The reading is formed only once every launched run of both arms is decided (WHEN THE READING IS MADE, RUN_IDs -> the
  six): an OFF run's open item (C8 PENDING, PRESENCE undecided, no F2 store, not ENDED) holds it INCOMPLETE as a PHASE
  run's does.
"""),
 # ---------------------------------------------------------------- CHOICE ENDS in the blind period (B1)
 ("""minutes with no write in the run folder) as the maintainer decided for STANCE6 (RUNLOG line 690) (CHOICE ENDS).
""",
  """minutes with no write in the run folder) as the maintainer decided for STANCE6 (RUNLOG line 690) (CHOICE ENDS; in the
blind period from metadata only, below).
"""),
 # ---------------------------------------------------------------- STARTS AND OUTCOME BLINDNESS (B1, I1)
 ("""run's launch line is not earlier is NOT BLIND, its classes are printed, and it enters THE ONE TEST as missing
(-infinity).
Report-only interim snapshots at the maintainer's request (as RUNLOG lines 681 and 689) are allowed only after every
OFF run's LAUNCH line, each noted at once as "<date -u> STANCE7 SNAPSHOT <RUN_ID> note=..." (a SNAPSHOT line is read
wherever it stands in RUNLOG): copies outside <PCR>, read by no other rule, never acting on a run.
""",
  """run's launch line is not earlier is NOT BLIND, its classes are printed, and it enters THE ONE TEST as missing
(-infinity).
THE BLIND PERIOD (fixer, the amendment's re-review B1; the maintainer's "Phase first" waves, RUNLOG line 876, put every
PHASE run in it): until every OFF run's LAUNCH line is in RUNLOG, a PHASE run's liveness, end and stall rest on
metadata only: Runtime > Manage sessions and Drive listings' names, sizes and modifiedTime. Its progress.json is not
opened (a tool may print its status and timesteps fields only if no other field reaches a person or agent); its end
from Drive (CHOICE ENDS) reads artifact_manifest.json present and 30 minutes with neither a progress.json rewrite nor
any other write in the run folder, and its TRIGGER-a / FOUND-GONE notes say they rest on metadata only; a STALL is 2
hours with no progress.json rewrite (its modifiedTime), not output lines; "no cell running" is not looked for in its
tab (a stop with the runtime still listed is found by the STALL, which in a run's first 2 hours may come after its
REPLACEMENT window [inferred]); its notebook is downloaded without its output being viewed (Colab steps). The launch
check's F0 copies are read by the launch-check tool only. Anything more (a PHASE run file opened, its notebook output
or downloaded notebook read) is an opening, noted at once as a SNAPSHOT line of that RUN_ID: every OFF run launched
after it is NOT BLIND. The scorer reads only the RUNLOG lines; what was looked at is the operator's record.
Report-only interim snapshots at the maintainer's request (as RUNLOG lines 681 and 689) are allowed only after every
OFF run's LAUNCH line, each noted at once as "<date -u> STANCE7 SNAPSHOT <RUN_ID> note=..." (one line per RUN_ID
looked at; a SNAPSHOT line is read wherever it stands in RUNLOG; one naming no round RUN_ID is printed among the other
lines and read as no opening): copies outside <PCR>, read by no other rule, never acting on a run.
"""),
 # ---------------------------------------------------------------- VP4: the fixer's re-runs
 ("""  manip_check.py unchanged). f2dry.py --arm off (C8 on a switch-OFF smoke) runs before F2 with VP4.
""",
  """  manip_check.py unchanged). f2dry.py --arm off (C8 on a switch-OFF smoke) runs before F2 with VP4.
  Fixer re-run 2026-11-01 after the amendment's re-review [measured, ST/amend/fix/dry/]: test_score_pc.py 65 tests
  pass (the 60 above and five for rules the suite did not hold: an OFF run's SNAPSHOT is no opening; an OFF launch in
  the opening's minute is not blind; the stock critic's why in C8; the launch-check CLI refusing an unfilled switch pin
  per arm; the not-blind rule reading an OFF seed's _b RELAUNCH); 49 of 49 mutants killed (the 44 above and M45-M49,
  the re-review's X1, X2, X3, X5, X10); VP3 and the stand-in both ways re-run with the same results (no ST/tools/pc
  file changed).
"""),
 # ---------------------------------------------------------------- KNOWN LIMITS: the waves (M2 / RUNLOG line 876)
 ("""  provenance.json records no torch, CUDA or GPU, and no notebook cell prints them); with the PHASE runs first and the
  OFF runs in a later wave (three sessions at a time), the arm is confounded with the wave, about one run's length
""",
  """  provenance.json records no torch, CUDA or GPU, and no notebook cell prints them); with the PHASE runs first and the
  OFF runs in a later wave (three sessions at a time: the maintainer's "Phase first", RUNLOG line 876), the arm is
  confounded with the wave, about one run's length
"""),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def apply(src: str, edits, name: str) -> str:
    for i, (old, new) in enumerate(edits, 1):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"STOP: {name} edit {i}: old text occurs {n} times: {old[:100]!r}")
        for ln in new.splitlines():
            if len(ln) > 120:
                raise SystemExit(f"STOP: {name} edit {i}: a new line is longer than 120: {ln!r}")
        src = src.replace(old, new)
    return src


def run(rel: str, edits, check: bool) -> int:
    b = (SRC / rel).read_bytes()
    if sha(b) != BEFORE[rel]:
        raise SystemExit(f"STOP: {SRC / rel} sha256 {sha(b)} is not the re-reviewed {BEFORE[rel]}")
    new = apply(b.decode("utf-8"), edits, rel)
    out = ST / rel
    if check:
        same = out.read_text(encoding="utf-8") == new
        print(f"{rel}: {'equals' if same else 'DIFFERS FROM'} the re-reviewed bytes + {len(edits)} edits")
        return 0 if same else 1
    out.write_text(new, encoding="utf-8")
    print(f"{rel}: {len(edits)} edits from the re-reviewed bytes; sha256 {sha(out.read_bytes())}")
    return 0


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    check = "--check" in sys.argv[2:]
    table = {"colab": ("COLAB_STEPS.txt", COLAB), "design": ("design/DESIGN_NOTES.txt", DESIGN),
             "criteria": ("criteria.txt", CRITERIA)}
    if which not in table:
        print(__doc__)
        return 2
    rel, edits = table[which]
    return run(rel, edits, check)


if __name__ == "__main__":
    raise SystemExit(main())
