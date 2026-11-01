"""PC AMENDMENT (2026-11-01): the criteria text's edits for the maintainer's CONTROL (b) answer (six runs; THE ONE TEST
against the concurrent switch-off runs; STANCE6 report only).  Exact replacements, each old text occurring exactly
once (else STOP, nothing written).  Applied to ST/criteria.txt as it stands after the re-review (its sha256 checked).
The PINS block is refilled afterwards by refill_pins.py from the files (never by hand).

    PY -I -B amend_texts.py criteria      (ST/criteria.txt from its re-reviewed bytes)
    PY -I -B amend_texts.py colab         (ST/COLAB_STEPS.txt from its re-reviewed bytes)
    PY -I -B amend_texts.py design        (ST/design/DESIGN_NOTES.txt from its re-reviewed bytes)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ST = Path("/home/user/work/pc_study")
CRIT_BEFORE = "7d6891fb114958fb166a8f959a4a440508808558820b95d6121e8dd1920efd25"


def apply(src: str, edits, name: str) -> str:
    for i, (old, new) in enumerate(edits, 1):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"STOP: {name} edit {i}: old text occurs {n} times: {old[:100]!r}")
        src = src.replace(old, new)
    return src


CRITERIA = [
 # ---------------------------------------------------------------- title and status
 ("""T. REX r10 PHASE-CRITIC ROUND (PC): THREE SEEDS FROM SCRATCH, 12M STEPS, COLAB L4, THE CRITIC GIVEN THE EPISODE PHASE
THROUGH A SEPARATE CRITIC-ONLY INPUT FROM STEP 0 -- CRITERIA v1, DRAFT FOR INDEPENDENT REVIEW. NOT APPROVED. NOTHING IS
LAUNCHED UNTIL <PCR>/APPROVED EXISTS AND ITS BYTES AND SHA256 ARE IN RUNLOG.
""",
  """T. REX r10 PHASE-CRITIC ROUND (PC): THREE SEEDS FROM SCRATCH, 12M STEPS, COLAB L4, THE CRITIC GIVEN THE EPISODE PHASE
THROUGH A SEPARATE CRITIC-ONLY INPUT FROM STEP 0, BESIDE THREE CONCURRENT SWITCH-OFF RUNS OF THE SAME SEEDS (SIX RUNS)
-- CRITERIA v1, AMENDED, DRAFT FOR THE QUICK RE-REVIEW. NOT APPROVED. NOTHING IS LAUNCHED UNTIL <PCR>/APPROVED EXISTS
AND ITS BYTES AND SHA256 ARE IN RUNLOG.
"""),
 ("""Still a draft for the re-review and the maintainer's confirmation: nothing here is approved.
STATUS:""",
  """Still a draft for the re-review and the maintainer's confirmation: nothing here is approved.
AMENDED 2026-11-01 (date -u) by the amendment editor (workflow wf_ab0237b7-d06, RUNLOG line 875) after the re-review
read READY and the maintainer answered the confirmation (RUNLOG line 874, 01:15:14 UTC): NAMES, PUSH, INIT (a), K3 ON,
RESUME (a), ENDS, TEST (keep the one test) as recommended, and CONTROL (b) "Add 3 switch-off runs" with "Make them the
test's control": SIX runs (the three phase runs and three concurrent switch-off runs, OFF below); THE ONE TEST compares
the phase runs with the OFF runs, decided before any data; STANCE6 is REPORT ONLY wherever it was a comparator. Every
change and why: ST/amend/AMEND_NOTES.txt; this text against the re-reviewed bytes (sha256 7d6891fb...) in
ST/amend/diffs/criteria.txt.diff. A draft for the quick re-review before the pin: nothing here is approved.
STATUS:"""),
 # ---------------------------------------------------------------- short names
 ("""  STANCE6 = the stance6 round's three runs stance6_r10LTSB_s60, _s42, _s80 (branch claude/screen-trex-r10-armltsb at
          8737cd3d86431de1bdd0fcdb49a0d956a45e084d), its F2 store S1/foot/stance6/fetch/ (manifest.csv = FMAN) and its
          scorer output S2/r10round/f2/score.json (sha256
          cb380a7e816e519cce267fd4a7c00a7d4528730a388e39b3678b95d235c01474).
""",
  """  STANCE6 = the stance6 round's three runs stance6_r10LTSB_s60, _s42, _s80 (branch claude/screen-trex-r10-armltsb at
          8737cd3d86431de1bdd0fcdb49a0d956a45e084d), its F2 store S1/foot/stance6/fetch/ (manifest.csv = FMAN) and its
          scorer output S2/r10round/f2/score.json (sha256
          cb380a7e816e519cce267fd4a7c00a7d4528730a388e39b3678b95d235c01474). Since the amendment: REPORT ONLY (printed
          beside, deciding nothing).
  PHASE = the three phase-critic runs stance7_r10LTSBpc_s60, _s42, _s80 (CRITIC_PHASE True); "PC run" below means a
          PHASE run.
  OFF   = the three concurrent switch-off runs stance7_r10LTSBoff_s60, _s42, _s80 (CRITIC_PHASE False, its default):
          the same HEAD, notebook path, seeds, schedule and every other setting; THE ONE TEST's control (amendment).
"""),
 # ---------------------------------------------------------------- the questions
 ("""  PREDICTED [inferred, DESIGN_NOTES section 1]:""",
  """  Q1-Q4 are asked of the PHASE runs; the same classes are read on the OFF runs and printed beside (they decide no line).
  PREDICTED [inferred, DESIGN_NOTES section 1]:"""),
 ("""  habit with no tail habit; PS's guard saw the phase arm's tail_2 and tail_1 pitch commands drift up slightly (+0.004
  median rise; +0.04 to +0.06) [measured, PS REPORT.md].
""",
  """  habit with no tail habit; PS's guard saw the phase arm's tail_2 and tail_1 pitch commands drift up slightly (+0.004
  median rise; +0.04 to +0.06) [measured, PS REPORT.md]. The OFF runs are STANCE6's configuration on today's image (A6):
  a habit in about 2 of 3 of them is the likelier result [inferred from STANCE6's 2 of 3; GPU nondeterminism can move
  any single run].
"""),
 # ---------------------------------------------------------------- what is trained
 ("""  panel per CHOICE K3), with ONE change: the switch below ON. Code: HEAD = 8737cd3d plus the builder's reviewed
  commits and the fixer's typing commit (no header commit: APPROVED and RUNLOG pin HEAD and this text; CHOICE HDR).
""",
  """  panel per CHOICE K3), in SIX runs (amendment): the three PHASE runs with ONE change, the switch below ON; the three
  OFF runs with the switch OFF (its default), everything else identical (same HEAD, notebook path, seeds, schedule,
  K3 panel; six Colab L4 runtimes, or fewer at once as RUN HANDLING allows). Code: HEAD = 8737cd3d plus the builder's
  reviewed commits and the fixer's typing commit (no header commit: APPROVED and RUNLOG pin HEAD and this text; CHOICE
  HDR); both arms run the same HEAD.
"""),
 ("""       and advantages all bitwise equal to the switch-OFF rollout; the phase columns in [0, 1]]
""",
  """       and advantages all bitwise equal to the switch-OFF rollout; the phase columns in [0, 1]] So a PHASE run and its
       same-seed OFF run start from the same networks and, on CPU, the same first rollout; they diverge from the first
       update (amendment: the 3 + 3 design is partly paired by seed; THE ONE TEST stays unpaired, KNOWN LIMITS).
"""),
 ("""       (compare_head_off_vs_head_on.txt)]
""",
  """       (compare_head_off_vs_head_on.txt)] So the OFF runs are STANCE6's configuration (8737cd3d) on today's Colab
       image: their classes beside STANCE6's are a report-only replication (REPORT ONLY).
"""),
 ("""       why).
- SEEDS AND RUNS (STANCE6's seeds, D2): seed 60 / 42 / 80; RUN_IDs stance7_r10LTSBpc_s60, _s42, _s80; RUN_LABEL
  "r10 LTSB phase-critic s60" (s42, s80); the in-run K3 sigma key per seed as ROUND l.78-79 (CHOICE NAMES). Seeds do
  not pair runs (GPU training is not bit-reproducible; the phase changes training from the first update): per-seed
  comparisons are report only.
""",
  """       why). With the switch OFF (the OFF runs) the same call reproduces ROUND's three pins (PINS: pins_pc.json
       off_arm, the switch key absent).
- SEEDS AND RUNS (STANCE6's seeds, D2): seed 60 / 42 / 80 in each arm; PHASE RUN_IDs stance7_r10LTSBpc_s60, _s42, _s80,
  RUN_LABEL "r10 LTSB phase-critic s60" (s42, s80); OFF RUN_IDs stance7_r10LTSBoff_s60, _s42, _s80, RUN_LABEL
  "r10 LTSB switch-off s60" (s42, s80) (amendment; no collision with RUNLOG, the stance6 Drive listing or the round's
  names [measured, ST/amend/names_check_off.out]); the in-run K3 sigma key per seed as ROUND l.78-79 in both arms
  (CHOICE NAMES). Seeds do not pair runs (GPU training is not bit-reproducible; the phase changes training from the
  first update): per-seed comparisons (PHASE beside OFF, beside STANCE6) are report only.
"""),
 # ---------------------------------------------------------------- SUBST
 ("""  <ROUND> -> <PCR>; RUN_IDs -> the three above (a relaunch: "_b" appended); RUNLOG KIND "STANCE6" -> "STANCE7" (same
  note form, ROUND l.403-415); the branch claude/screen-trex-r10-armltsb -> claude/screen-trex-r10-phasecritic; BR's
  HEAD -> HEAD; C4's pins (ROUND l.211-227) -> PINS below, plus C4 (p) and C8; the launch check (ROUND l.376-392)
  plus fact (8); "LTSB r9's 2 of 5" -> printed beside STANCE6's 2 of 3, report only; ROUND READING (l.307-333) ->""",
  """  <ROUND> -> <PCR>; RUN_IDs -> the six above, each run handled on its own (a relaunch: "_b" appended); RUNLOG KIND
  "STANCE6" -> "STANCE7" (same note form, ROUND l.403-415; one more kind, SNAPSHOT, RUN HANDLING); the branch
  claude/screen-trex-r10-armltsb -> claude/screen-trex-r10-phasecritic; BR's HEAD -> HEAD; C4's pins (ROUND l.211-227)
  -> PINS below, each run's arm's (an OFF run's are ROUND's own, the switch key absent), plus C4 (p) and C8; the launch
  check (ROUND l.376-392) plus fact (8); "LTSB r9's 2 of 5" -> printed beside the OFF runs' and STANCE6's 2 of 3, report
  only; ROUND READING (l.307-333) ->"""),
 # ---------------------------------------------------------------- what is read
 ("""- COMPLETE RUN C1-C7, LOST, PRESENCE: ROUND l.192-268, with SUBST, and two additions (COMPLETE = C1-C8):
  C4 (p) stage_config.json records the switch ON (PINS), read with C4's other facts (and by PRESENCE's p2);
  C8 THE MANIPULATION CHECK passes on R2 and on every J pair held, decided with C3, C5 and C7 after F2's step (5)
  fetches the pairs (a check that crashes is rerun on the same files; it never fails C8 by crashing).
""",
  """- COMPLETE RUN C1-C7, LOST, PRESENCE: ROUND l.192-268, with SUBST, and two additions (COMPLETE = C1-C8), for every
  run of both arms:
  C4 (p) stage_config.json records the switch as the run's arm sets it (PINS): ON for a PHASE run; OFF for an OFF run,
     no hyperparameters.critic_phase key at all (the code writes none with the switch OFF; a key holding any value,
     false included, fails); read with C4's other facts, each against its arm's pins (and by PRESENCE's p2);
  C8 THE MANIPULATION CHECK passes on R2 and on every J pair held, decided with C3, C5 and C7 after F2's step (5)
     fetches the pairs (a check that crashes is rerun on the same files; it never fails C8 by crashing): for a PHASE
     run it reads IN FORCE; for an OFF run it reads the stock critic, the record manip_check.py writes for a critic with
     no phase input (spec null; NOT IN FORCE failing (i) only, "the critic reads no phase (a stock critic)") [measured
     on STANCE6 s80's final pair, ST/tools/dry/vp4_widen_s80.out "STOCK"]; an OFF record reading a phase critic fails
     C8.
"""),
 ("""  (largest |dV| under a +1 shift of the normalised value) is printed, report only: it reads the mechanism this study
  is about (PS: the critic's tail slope went to about 0) and decides nothing. Printed per pair with the values read.
""",
  """  (largest |dV| under a +1 shift of the normalised value) is printed, report only: it reads the mechanism this study
  is about (PS: the critic's tail slope went to about 0) and decides nothing. Printed per pair with the values read.
  For an OFF run the same tool is run on the same pairs; it stops at (i) (no phase input) and C8 reads that record
  (above).
"""),
 ("""  finite cell at all has no TD and enters THE ONE TEST at +infinity). Read only from
  01_stance/eval_episodes.csv as C3 checks it against the manifest; no simulation.
""",
  """  finite cell at all has no TD and enters THE ONE TEST as missing: a PHASE run at +infinity, an OFF run at -infinity).
  Read only from 01_stance/eval_episodes.csv as C3 checks it against the manifest; no simulation; the same code for
  both arms (the evaluation rows are equal with the switch ON and OFF, VP0).
"""),
 # ---------------------------------------------------------------- the comparators -> report only
 ("""THE COMPARATORS (historical, same configuration but the switch; [measured, DESIGN/comparators.out unless named])
                         s60 (stance6)        s42 (stance6)               s80 (stance6)""",
  """STANCE6, REPORT ONLY (historical; the OFF runs' configuration; since the amendment it is printed beside the PHASE and
OFF runs and in the OFF-vs-STANCE6 replication block and decides no line, no class and no test; [measured,
DESIGN/comparators.out unless named])
                         s60 (stance6)        s42 (stance6)               s80 (stance6)"""),
 ("""  What differs besides the switch: HEAD's other bytes (the builder's commits; A6 shows OFF = 8737cd3d on CPU); GPU
  nondeterminism; the date (the Colab image; the notebook pins mujoco 3.10.0, gymnasium 1.3.0, SB3 2.9.0; torch and
  numpy are the image's: STANCE6 recorded numpy 2.1.3, torch not recorded) [measured, F2 provenance.json.copy1
  dependency_versions; notebook cell 3 at 8737cd3d]; the K3 panel if CHOICE K3 turns it off.
""",
  """  What differs between STANCE6 and the PHASE runs besides the switch: HEAD's other bytes (the builder's commits; A6
  shows OFF = 8737cd3d on CPU); GPU nondeterminism; the date (the Colab image; the notebook pins mujoco 3.10.0,
  gymnasium 1.3.0, SB3 2.9.0; torch and numpy are the image's: STANCE6 recorded numpy 2.1.3, torch not recorded)
  [measured, F2 provenance.json.copy1 dependency_versions; notebook cell 3 at 8737cd3d]. Between the PHASE and OFF
  runs only the switch differs, plus GPU nondeterminism and the start gaps (KNOWN LIMITS): the reason the OFF runs,
  not STANCE6, are the test's control.
"""),
 # ---------------------------------------------------------------- the reading
 ("""  r = COMPLETE runs with TAIL CLASS RAIL plus PRESENCE RAILS (ROUND l.253-268); u = seeds whose run is not COMPLETE or
  is UNREAD; p = PARKED; g = GATE PASS; hs = runs with a SATURATING habit elsewhere; hd = DRIVE ONLY.
""",
  """  The line is the PHASE runs' alone (their per-run classes as reviewed): r = COMPLETE PHASE runs with TAIL CLASS RAIL
  plus PHASE PRESENCE RAILS (ROUND l.253-268); u = PHASE seeds whose run is not COMPLETE or is UNREAD; p = PARKED;
  g = GATE PASS; hs = runs with a SATURATING habit elsewhere; hd = DRIVE ONLY. The OFF runs' r, p, g, hs, hd and u are
  printed beside every line ("CONCURRENT SWITCH-OFF RUNS (printed beside; they decide no line): ...") and move none.
"""),
 ("""     FROM STEP 0 DID NOT KEEP EVERY RUN OFF THE TAIL RAIL", each with its first TAIL ALARM beside STANCE6 s42's
     3,932,160.
""",
  """     FROM STEP 0 DID NOT KEEP EVERY RUN OFF THE TAIL RAIL", each with its first TAIL ALARM beside its same-seed OFF
     run's (STANCE6 s42's 3,932,160 stays in the report-only STANCE6 block).
"""),
 ("""  Lines 3-5 also print "[model, plug-in] if the phase critic changed nothing, no RAIL in 3 runs has chance (2/3)^3 =
  0.296 at STANCE6's 1 of 3 and (3/5)^3 = 0.216 at LTSB r9's 2 of 5" (ROUND printed such figures, TABLE E).
  Every line ends "(descriptive: 3 runs against the stance6 round's 3, a historical comparison; KNOWN LIMITS)" and
  prints: the per-run table (status, TAIL CLASS and kind, early signs, GATE CLASS labelled "(GATE CERTIFICATE: final
  checkpoint, measured on a screen task; not certification)" (ROUND l.280-283) and the notebook's own
  gate_verdict.json, RUN OUTCOME, habits elsewhere with DRIVE ONLY ones (hd), TD) beside THE COMPARATORS; Q1-Q3 counts;
  THE ONE TEST; the report-only comparisons; every LOST run with D-16 (ROUND l.346-354); any PROVISIONAL item.
""",
  """  Lines 3-5 also print "[model, plug-in] if the phase critic changed nothing, no RAIL in 3 runs has chance
  (1 - r_off/n_off)^3 at the concurrent switch-off runs' r_off of n_off (the OFF runs read to a class, plus OFF
  PRESENCE RAILS; "not formed" when none is read); report only beside it: (2/3)^3 = 0.296 at STANCE6's 1 of 3 and
  (3/5)^3 = 0.216 at LTSB r9's 2 of 5" (ROUND printed such figures, TABLE E).
  Every line ends "(descriptive: 3 phase-critic runs beside 3 concurrent switch-off runs; KNOWN LIMITS)" and prints:
  the OFF runs' counts beside; the per-run table (status, TAIL CLASS and kind, early signs, GATE CLASS labelled "(GATE
  CERTIFICATE: final checkpoint, measured on a screen task; not certification)" (ROUND l.280-283) and the notebook's
  own gate_verdict.json, RUN OUTCOME, habits elsewhere with DRIVE ONLY ones (hd), TD) for the PHASE runs, each beside
  its same-seed OFF run's classes and TD and STANCE6's same seed (report only); the OFF runs' own table; Q1-Q3 counts;
  THE ONE TEST; the OFF-vs-STANCE6 replication block and the other report-only comparisons; the outcome-blindness
  check and the start gaps (RUN HANDLING); every LOST run with D-16 (ROUND l.346-354); any PROVISIONAL item.
"""),
 # ---------------------------------------------------------------- the one test
 ("""THE ONE TEST (pre-registered; it never changes a class, a per-run verdict or the line)
- H1: the phase-critic runs' TAIL DRIVE is lower than STANCE6's. Statistic: the sum of the three PC runs' TD. Exact
  one-sided unpaired permutation test over the 20 ways to split the 6 TD values 3 + 3: p = (the number of splits whose
  3-sum is <= the PC sum) / 20, ties counting against H1. alpha 0.05. Because 1/20 is the smallest p, it reads
  "LOWER THAN EVERY STANCE6 RUN (exact permutation p = 1/20)" only on complete separation, which, the comparators being
  fixed, is the same as every PC run's TD below 0.154339 (STANCE6 s80's; [measured, DESIGN/comparators.out]); otherwise
  "NOT SEPARATED (p = <k>/20)". A PC run that is not COMPLETE and FINISHED enters with TD = +infinity (above every
  comparator): missing never helps H1 (the PC 3-sum is then infinite, every split ties or falls below it, and p =
  20/20). Sums are exact: rational arithmetic on the float values (fractions.Fraction of each float), compared
  exactly (math.fsum is correctly rounded, not exact, and could misorder a near tie). Only p = 1/20 depends on the
  placement alone (complete separation); every other p depends on the magnitudes as well. Printed with the six TDs,
  the line "equivalently, every PC TD < 0.154339 (STANCE6 s80's, the lowest comparator); the statistic was chosen
  with STANCE6's values in view (2 windows looked at: a strict bound of 2/20)", the J-window TD (evaluations from
  4,194,304) beside, report only (STANCE6: 0.577658 / 0.808751 / 0.217914), and the caveat line: "historical
  comparison: exchangeability of the 6 runs is assumed, not designed; each PC run starts from its STANCE6 seed's own
  network (A5), so seed effects make the unpaired test conservative; same-seed pairs printed, report only (KNOWN
  LIMITS)". A paired sign test cannot go below p = 1/8 with 3 pairs: the unpaired test stays.
""",
  """THE ONE TEST (pre-registered; amended before any data: the PHASE runs against the concurrent OFF runs, the
maintainer's "Make them the test's control", RUNLOG line 874; it never changes a class, a per-run verdict or the line)
- H1: the PHASE runs' TAIL DRIVE is lower than the concurrent OFF runs'. Statistic: the sum of the three PHASE runs'
  TD (TD unchanged, TAIL DRIVE above). Exact one-sided unpaired permutation test over the 20 ways to split the 6 TD
  values (3 PHASE, 3 OFF) 3 + 3: p = (the number of splits whose 3-sum is <= the PHASE sum) / 20, ties counting against
  H1. alpha 0.05. Because 1/20 is the smallest p, it reads "LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN (exact
  permutation p = 1/20)" only on complete separation (every PHASE TD below every OFF TD); otherwise "NOT SEPARATED (p =
  <k>/20)". Only p = 1/20 depends on the placement alone; every other p depends on the magnitudes as well.
- MISSING, against H1: a PHASE seed whose run is not COMPLETE and FINISHED (LOST, replaced and its _b not COMPLETE,
  STOPPED, never launched) enters at +infinity, the highest rank; an OFF seed's run likewise enters at -infinity, the
  lowest rank, and so does an OFF run launched after a PHASE run's results were opened (RUN HANDLING, outcome
  blindness). A 3-sum holding infinities is ordered by (the number of +infinity in it minus the number of -infinity,
  then the exact sum of its finite values): the limit of +M and -M for a large M, so a missing run moves its split by a
  whole rank and never ties by itself; any missing run gives p >= 11/20 (never LOWER). Finite sums are exact: rational
  arithmetic on the float values (fractions.Fraction of each float), compared exactly (math.fsum is correctly rounded,
  not exact, and could misorder a near tie).
- The statistic's history, stated honestly: TD's form (the whole-run mean of |CMD_L|; the J window looked at too and
  printed beside, report only) was chosen with STANCE6's values in view; the six runs' values, the OFF runs' included,
  are unseen at the pin, so the statistic and the test are fixed before their data.
- Printed with the six TDs (PHASE; OFF), the J-window TDs (evaluations from 4,194,304) beside, report only, STANCE6's
  TDs beside, report only (0.388227 / 0.635343 / 0.154339; J 0.577658 / 0.808751 / 0.217914), the line "p = 1/20 only
  on complete separation: every phase TD below every switch-off TD; TD's form (the whole-run mean; 2 windows looked at)
  was chosen with STANCE6's values in view, the six runs' values are unseen at the pin", and the caveat line:
  "concurrent control: the 6 runs share HEAD, the notebook, the seeds and the schedule, started within the gaps
  printed; their exchangeability under no effect is assumed up to seed effects: each phase run and its same-seed
  switch-off run start from the same network (A5), a positive within-seed dependence that makes the unpaired test
  conservative [inferred]; GPU nondeterminism within today's image remains; same-seed pairs printed, report only
  (KNOWN LIMITS)". A paired sign test cannot go below p = 1/8 with 3 pairs: the unpaired test stays.
"""),
 # ---------------------------------------------------------------- report only
 ("""- Statistic A as ROUND l.288-299 (ANY ALARM runs), printed beside STANCE6's 2 of 3 and LTSB r9's 2 of 5 with one-sided
  Fisher p (0 of 3 against 2 of 3: 0.20 [measured, DESIGN/comparators.out]).
- Per seed beside STANCE6's same seed; first TAIL ALARM and early signs beside STANCE6's; MEAN_a per tail actuator; the""",
  """- Statistic A as ROUND l.288-299 (ANY ALARM runs), PHASE beside the OFF runs' with one-sided Fisher p, and beside
  STANCE6's 2 of 3 and LTSB r9's 2 of 5 (0 of 3 against 2 of 3: 0.20 [measured, DESIGN/comparators.out]).
- THE OFF-vs-STANCE6 REPLICATION: with the switch OFF the code equals 8737cd3d bitwise (A6, measured), so the OFF runs
  replicate STANCE6's configuration on today's Colab image; per seed, each OFF run's TAIL CLASS (kind), first TAIL
  ALARM, GATE CLASS, RUN OUTCOME, HABIT ELSEWHERE, TD (and the J window) and CMD_L >= 0.60 count beside STANCE6's same
  seed, with the arm totals. It decides nothing.
- Per seed: each PHASE run beside its same-seed OFF run and STANCE6's same seed; first TAIL ALARM and early signs beside
  theirs; MEAN_a per tail actuator; the"""),
 ("""- The critic, phase fed (tools adapted, MEASUREMENT): its slope on the tail_3 angle per degree at the 1,048,576 grid
  pairs and the final pair, with the phase fed from k ("effective") and set to phi(1000) ("settled"); V - G by
  episode-step bucket [0,50) [50,150) [150,200) [200,250) [250,1000] at the final pair (PS's V-G tool form, RUNLOG
  line 861); the loop gain G, offset c0 and margin b* (actor-only, BS2 read_bs M-LOOP) beside STANCE6's""",
  """- The critic, phase fed (tools adapted, MEASUREMENT; the OFF runs' stock critic with its own feed, beside): its slope
  on the tail_3 angle per degree at the 1,048,576 grid pairs and the final pair, with the phase fed from k
  ("effective") and set to phi(1000) ("settled"); V - G by episode-step bucket [0,50) [50,150) [150,200) [200,250)
  [250,1000] at the final pair (PS's V-G tool form, RUNLOG line 861); the loop gain G, offset c0 and margin b*
  (actor-only, BS2 read_bs M-LOOP) beside the OFF runs' and STANCE6's"""),
 ("""- The guard after PS: tail_1 and tail_2 pitch commands (ledger and M1 CMD at J) beside STANCE6's.
""",
  """- The guard after PS: tail_1 and tail_2 pitch commands (ledger and M1 CMD at J) beside the OFF runs' and STANCE6's.
"""),
 # ---------------------------------------------------------------- run handling
 ("""RUN HANDLING, RECORDS, ENDING, REPLACEMENT: ROUND l.356-434 with SUBST (all or nothing; no run is resumed: CHOICE
RESUME), and: launch-check fact (8) the phase switch ON in 01_stance/stage_config.json (PINS) is a FAULT like (1)-(5)
and (7) (stop, LOST, relaunch _b by REPLACEMENT (ii)); if CHOICE RESUME (b) is ever taken, CRITIC_PHASE = True is
set again before the RESUME cell and fact (8) is read only after the corrected resume rewrote stage_config.json (a
refused load writes stage_config.json before it refuses, as at 8737cd3d); the notebook's own rows "Critic phase: on"
(cell 6) and "Critic phase on: [ppo]
critic_phase = true on ['stance']" (cell 9) (Colab steps) are recorded in the CHECK line. Run ends may be noted from
Drive (artifact_manifest.json present, progress.json ended, 30""",
  """RUN HANDLING, RECORDS, ENDING, REPLACEMENT: ROUND l.356-434 with SUBST (all or nothing; no run is resumed: CHOICE
RESUME), for each of the six runs on its own: its own launch, its launch check within 60 minutes of its own launch,
and at most one "_b" relaunch by REPLACEMENT within 2 h of its own launch (the window counts from that run's LAUNCH
line, whatever the other runs do); and: launch-check fact (8) the switch as the run's arm sets it in
01_stance/stage_config.json (PINS: ON for a PHASE run; for an OFF run no hyperparameters.critic_phase key) is a FAULT
like (1)-(5) and (7) (stop, LOST, relaunch _b by REPLACEMENT (ii)); if CHOICE RESUME (b) is ever taken (a PHASE run),
CRITIC_PHASE = True is set again before the RESUME cell and fact (8) is read only after the corrected resume rewrote
stage_config.json (a refused load writes stage_config.json before it refuses, as at 8737cd3d); the notebook's own rows
"Critic phase: on" (cell 6) and "Critic phase on: [ppo] critic_phase = true on ['stance']" (cell 9) for a PHASE run,
"Critic phase: off" and no cell-9 row for an OFF run (Colab steps), are recorded in the CHECK line. Run ends may be
noted from Drive (artifact_manifest.json present, progress.json ended, 30"""),
 ("""minutes with no write in the run folder) as the maintainer decided for STANCE6 (RUNLOG line 690) (CHOICE ENDS).
Report-only interim snapshots at the maintainer's request (as RUNLOG lines 681 and 689) are allowed: copies outside
<PCR>, read by no rule, never acting on a run.
""",
  """minutes with no write in the run folder) as the maintainer decided for STANCE6 (RUNLOG line 690) (CHOICE ENDS).
STARTS AND OUTCOME BLINDNESS (amendment): the six runs start interleaved by seed (PHASE s60, OFF s60, PHASE s42, OFF
s42, PHASE s80, OFF s80), all at once when six L4 sessions can run together. If fewer can, the PHASE runs start first
and the OFF runs may start later, in seed order, as sessions free (Colab steps), and every OFF run must start before
any PHASE run's results are opened: before any F2-START or REFETCH-START line of a PHASE run, before any report-only
snapshot of a PHASE run, and before anyone reads a PHASE run's training output beyond the launch check's rows. Each
run's launch time is its LAUNCH (or RELAUNCH) line; the scorer prints each seed's PHASE-OFF start gap, report only, and
reads the first opening as the earliest F2-START, REFETCH-START or SNAPSHOT line of a PHASE RUN_ID: an OFF seed whose
run's launch line is not earlier is NOT BLIND, its classes are printed, and it enters THE ONE TEST as missing
(-infinity).
Report-only interim snapshots at the maintainer's request (as RUNLOG lines 681 and 689) are allowed only after every
OFF run's LAUNCH line, each noted at once as "<date -u> STANCE7 SNAPSHOT <RUN_ID> note=..." (a SNAPSHOT line is read
wherever it stands in RUNLOG): copies outside <PCR>, read by no other rule, never acting on a run.
"""),
 # ---------------------------------------------------------------- fetches
 ("""FETCHES: ROUND l.436-485 with SUBST (F0 within 60 minutes of each launch; the trigger listing; F2 one fetch starting at
least 6 hours after RUNLOG's last ending note; the re-fetch), the transfers made by the exact-copy method the""",
  """FETCHES: ROUND l.436-485 with SUBST (F0 within 60 minutes of each of the six launches; the trigger listing; F2 one
fetch of every launched RUN_ID of both arms starting at least 6 hours after RUNLOG's last ending note; the re-fetch),
the transfers made by the exact-copy method the"""),
 # ---------------------------------------------------------------- measurement
 ("""- New: the manipulation check (above); TD (DESIGN/comparators.py's ledger reading, pinned); the phase-fed critic readers
  (report only).
""",
  """- New: the manipulation check (above); TD (DESIGN/comparators.py's ledger reading, pinned); the phase-fed critic readers
  (report only).
- Amended for the six runs (2026-11-01; every diff in ST/amend/diffs/, every change in ST/amend/AMEND_NOTES.txt): the
  arms (r10round_common: both RUN_ID prefixes, arm_of, the token pattern, the SNAPSHOT kind, the off_arm pin key);
  complete_r10round (each run against its arm's pins: launch-check facts (3) and (8), C4 and C4 (p); C8's arm form);
  the lane A runner's RUN_ID binding (measure_r10round.py, r10m_common.py: both prefixes); the scorer (score_pc.py,
  pc_common.py: both arms' seeds run-handled alike, THE READING from the PHASE runs with the OFF runs beside, THE ONE
  TEST against the OFF runs with the MISSING ranks, outcome blindness and the start gaps, the replication block,
  STANCE6 report only); pins_pc.json (off_arm from ST/tools/dry/pins_sc.out "off"; tool_sha256).
"""),
 # ---------------------------------------------------------------- validations
 ("""  Fisher 0 of 3 vs 2 of 3: p 0.20); the one
  test's smallest p is 1/20 and needs complete separation.""", None),   # placeholder (skipped below)
]
CRITERIA = [e for e in CRITERIA if e[1] is not None]

CRITERIA += [
 ("""  smoke folder); NEG faults (1), (3), (7), (8). Re-run 2026-11-01 by the fixer on e4d222bb's sealed measuring copy and a
  new 2-update smoke bundle [measured, ST/tools/dry/vp3.out]: POS no fault; NEG faults (1), (3), (7), (8). The pin step
  re-runs it on the pinned files.
""",
  """  smoke folder); NEG faults (1), (3), (7), (8). Re-run 2026-11-01 by the fixer on e4d222bb's sealed measuring copy and a
  new 2-update smoke bundle [measured, ST/tools/dry/vp3.out]: POS no fault; NEG faults (1), (3), (7), (8). The pin step
  re-runs it on the pinned files.
  Amendment, both arms (ST/tools/vp3_dryrun.py, each case under its arm's RUN_ID and LAUNCH fields, faults compared with
  the expected): POS-pc on the switch-ON bundle, none; POS-off on a switch-OFF bundle (smoke_pc.py bundle --switch
  off), none; the ON bundle under an OFF RUN_ID (3), (7), (8) and the OFF bundle under a PHASE RUN_ID (3), (7), (8);
  NEG-pc on STANCE6 s42's F0 files (1), (3), (7), (8); NEG-off on the same files under the OFF RUN_ID (1), (7) only
  (8737cd3d's switch-off files: only the commit and the run_id differ); SC-* the stage_config.json files HEAD's own
  save_stage_config wrote with the switch ON and OFF (ST/tools/dry/pins_sc) alone under each arm's RUN_ID: on-pc and
  off-off none, on-off and off-pc (3), (8). Dry run 2026-11-01 by the amendment editor with WT HEAD 1dfe2327 [measured,
  ST/amend/dry/vp3.out, SUMMARY]: NEG-pc, NEG-off and the four SC cases as expected; POS-pc, POS-off and the two
  crossed cases SKIPPED (the smoke bundles were deleted after recording and the amendment trains nothing): the pin step
  makes both bundles (one 2-update switch-ON smoke; bundle --switch on and --switch off) and runs all ten.
"""),
 ("""  widened s80 final (ST/tools/dry/vp4_widen_s80.out).
""",
  """  widened s80 final (ST/tools/dry/vp4_widen_s80.out).
  Amendment re-run 2026-11-01 by the amendment editor [measured, ST/amend/AMEND_NOTES.txt]: ST/tools/tests/
  test_score_pc.py 60 tests pass (ST/amend/dry/pytest_score_pc.log; six-run synthetic roots: LOWER at 1/20; not lower
  when one OFF TD is below the PHASE TDs; a missing PHASE run at +infinity and a missing OFF run at -infinity, k = 11;
  an OFF run reading a phase critic in C8 or carrying the switch key LOST; an OFF run started after a PHASE SNAPSHOT
  not blind, missing; STANCE6 unreadable holding nothing open; the OFF-vs-STANCE6 block; the MISSING rule against an
  independent +M/-M oracle on 400 random mixes); 44 of 44 mutants killed (ST/amend/dry/mutate_pc.log: the reviewed
  31, five of them re-pointed where the amended text moved, plus 13 for the amendment's rules); the stand-in both ways
  (ST/amend/dry/standin_pc:
  STANCE6 read as the PHASE arm, 25 of 25 class keys equal for each run, line 1, p 20/20 with the OFF arm not
  launched; ST/amend/dry/standin_off: STANCE6 read as the OFF arm against the off_arm pins with no substitution,
  C4 and C4 (p) PASS, 25 of 25 class keys equal, the replication block printing equal TDs, line 2 and p 20/20 with
  the PHASE arm not launched). test_manip_pc.py's 7 tests skip here (no measuring copy or smoke pair on disk;
  manip_check.py unchanged). f2dry.py --arm off (C8 on a switch-OFF smoke) runs before F2 with VP4.
"""),
 # ---------------------------------------------------------------- known limits
 ("""- Three runs measure no rate; binary outcomes cannot reach 0.05 against 3 (Fisher 0 of 3 vs 2 of 3: p 0.20); the one
  test's smallest p is 1/20 and needs complete separation.
- Historical control, not concurrent: one code commit, GPU nondeterminism and the date differ; exchangeability is
  assumed. The TD statistic and its window were chosen with the comparators' values in view (the simplest whole-run
  form; the J window printed beside). The seeds were chosen with their earlier outcomes known. Each PC run starts
  from its STANCE6 seed's own network (A5: the first rollout bitwise), so the 3 + 3 design is partly paired; with
  outcomes that persist by seed (r9's s42 and r10's s42 both first over the bar at 3,670,016), the unpaired test is
  conservative, not inflated; same-seed pairs are printed, report only. The date and the Colab image are not
  measured (provenance.json records no torch, CUDA or GPU, and no notebook cell prints them); a concurrent switch-OFF
  s42 run (about 15.5 L4-hours, report only) would test that confound directly: not in this design unless the
  maintainer adds it at the confirmation.
""",
  """- Three runs per arm measure no rate; binary outcomes cannot reach 0.05 with 3 against 3 (Fisher 0 of 3 vs 2 of 3:
  p 0.20); the one test's smallest p is still 1/20 with 3 PHASE against 3 OFF runs and needs complete separation.
- What the concurrent OFF runs remove (amendment): the code commit (both arms run HEAD), the date and the Colab image
  (within the start gaps), the notebook path and the K3 setting, as differences between the test's arms. What they do
  not remove: GPU nondeterminism within today's image (a seed fixes the start, not the run); 3 + 3 still reaches only
  p = 1/20; image or hardware changes between staggered starts (the gaps are printed; the image itself is not measured:
  provenance.json records no torch, CUDA or GPU, and no notebook cell prints them); with the PHASE runs first and the
  OFF runs in a later wave (three sessions at a time), the arm is confounded with the wave, about one run's length
  apart (the start time, and the Colab image if it changes between the waves): the test then also assumes no wave
  effect, and the gaps are printed beside it; exchangeability under no effect
  is assumed up to seed effects: each PHASE run and its same-seed OFF run start from the same network (A5: the first
  rollout bitwise), so the 3 + 3 design is partly paired, and with outcomes that persist by seed (r9's s42 and r10's
  s42 both first over the bar at 3,670,016) the unpaired test is conservative, not inflated [inferred]; same-seed pairs
  are printed, report only. The TD statistic and its window were chosen with STANCE6's values in view (the simplest
  whole-run form; the J window printed beside); the six runs' values are unseen at the pin. The seeds were chosen with
  their earlier outcomes known (both arms share them).
- STANCE6 as a historical control: report only now (its classes and TD printed beside, the OFF-vs-STANCE6 replication
  block); one code commit, GPU nondeterminism and the date differ from the PHASE runs; it decides nothing.
"""),
 ("""- LOST runs leave the classes (PRESENCE can still read line 1) and enter the test at +infinity; INCOMPLETE is possible
  (ROUND TABLE M: about 9.5% of rounds at REALISTIC loss rates [model]).
""",
  """- LOST runs leave the classes (PRESENCE can still read line 1) and enter the test as missing (a PHASE run at
  +infinity, an OFF run at -infinity: either way never LOWER); with six runs, losing any one of them already removes the
  chance of p = 1/20; INCOMPLETE is possible (ROUND TABLE M: about 9.5% of rounds at REALISTIC loss rates, for the three
  PHASE runs [model]).
"""),
 # ---------------------------------------------------------------- compute and time
 ("""COMPUTE AND TIME [inferred from ROUND l.677-683 and RUNLOG lines 693-699]: about 14.4-15.4 h per run on an L4, three in
parallel, about 46 L4-hours (about 62 with one _b); launched 01:00-02:00 UTC, the manifests land about 16:00-17:00 UTC
2026-11-01 (stance6: about 14.9 h after launch), F2 at least 6 h after the last ending note, then about 2 CPU-hours of
M1/M3 (M1 about 68 s per label, M3 about 33 min per final [measured, S2/r10round/long_blind/measure/validate/
classify_parents.out]). Disk: measure_r10round.py refuses below 3 GiB free; about 3.8 GB was free at 21:18 UTC
[measured, df -h /, DESIGN/cmdlog.txt], so the coordinator checks df before F2.
""",
  """COMPUTE AND TIME [inferred from ROUND l.677-683 and RUNLOG lines 693-699]: about 14.4-15.4 h per run on an L4, six
runs, about 92 L4-hours (about 108 with one _b); all six at once: the manifests land about 15 h after the launches; with
staggered OFF starts F2 waits for the last OFF run's ending note plus 6 h (up to about a day later); then about 4
CPU-hours of M1/M3 for six runs (M1 about 68 s per label, M3 about 33 min per final [measured, S2/r10round/long_blind/
measure/validate/classify_parents.out]) and the manipulation checks (about 7 s each). Disk: measure_r10round.py
refuses below 3 GiB free; 3.6 GB was free at 01:43 UTC 2026-11-01 [measured, df -h /, ST/amend/cmdlog.txt] and F2's
pairs for six runs need about 1.2 GB (about 0.19 GB per run, ROUND l.84-85), so the coordinator clears scratch and
checks df before F2.
"""),
 # ---------------------------------------------------------------- how the coordinator pins
 ("""   is frozen: no push to it until all three launch checks read no-fault; (c) the launch-check""",
  """   is frozen: no push to it until all six launch checks read no-fault; (c) the launch-check"""),
 ("""   port's VP3 dry runs re-run, exit codes as expected; (d) B1 compute balance and B2 fetch budget noted (Colab steps).
4. Only then the maintainer launches (ST/COLAB_STEPS.txt).""",
  """   port's VP3 dry runs re-run (both arms' cases, ST/tools/vp3_dryrun.py), every case as expected; (d) B1 compute
   balance and B2 fetch budget for six runs noted (Colab steps).
4. Only then the maintainer launches the six runs (ST/COLAB_STEPS.txt)."""),
 ("""4. Only then the maintainer launches the six runs (ST/COLAB_STEPS.txt). Nothing in criteria.txt or APPROVED is edited after
   the first training step; an error found later is recorded in RUNLOG and the reading prints it as PROVISIONAL.""",
  """4. Only then the maintainer launches the six runs (ST/COLAB_STEPS.txt). Nothing in criteria.txt or APPROVED is
   edited after the first training step; an error found later is recorded in RUNLOG and the reading prints it as
   PROVISIONAL."""),
]


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "colab":
        return main_colab()
    if which == "design":
        return main_design()
    if which != "criteria":
        raise SystemExit("usage: amend_texts.py criteria|colab")
    f = ST / "criteria.txt"
    got = hashlib.sha256(f.read_bytes()).hexdigest()
    if got != CRIT_BEFORE:
        raise SystemExit(f"STOP: criteria.txt sha256 {got} is not the re-reviewed {CRIT_BEFORE}")
    new = apply(f.read_text(encoding="utf-8"), CRITERIA, "criteria")
    f.write_text(new, encoding="utf-8")
    print(f"criteria.txt: {len(CRITERIA)} edits; sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")
    return 0



# ==================================================================================================== COLAB_STEPS.txt
COLAB_BEFORE = "00e30d4b107ba7427123435aab95e2504d1c7a92395665e3f383f05f7f13b38b"
COLAB = [
 ("""T. rex r10 PHASE-CRITIC ROUND (PC): THREE SEEDS FROM SCRATCH, 12M STEPS, THREE COLAB L4s -- COLAB STEPS v1 (DRAFT),
""",
  """T. rex r10 PHASE-CRITIC ROUND (PC): THREE SEEDS FROM SCRATCH IN TWO ARMS (THREE PHASE RUNS, THREE CONCURRENT
SWITCH-OFF RUNS), 12M STEPS, SIX COLAB L4 RUNS -- COLAB STEPS v1 (DRAFT, AMENDED),
"""),
 ("""this file: give the maintainer only this one.
=================================================================================================================""",
  """this file: give the maintainer only this one.
AMENDED 2026-11-01 (date -u) by the amendment editor after the maintainer's answers (RUNLOG line 874: CONTROL (b),
"Add 3 switch-off runs", "Make them the test's control"): six runs (the three phase runs and three concurrent
switch-off runs), the order, fewer sessions, outcome blindness, the per-run _b rule, the SNAPSHOT line. Every change:
ST/amend/AMEND_NOTES.txt and ST/amend/diffs/COLAB_STEPS.txt.diff.
================================================================================================================="""),
 ("""Three runs, one Colab L4 runtime each, started together. Nothing here is decided by what a run prints during training:
no run is stopped, changed, restarted, extended or rerun because of a readout, an alarm, a fetch, the in-run panel or
anything the notebook prints beyond the configuration the launch check reads. Fails stay fails. Nobody moves, edits or
deletes a file in a run folder; no fetch writes to Drive.""",
  """Six runs, one Colab L4 runtime each: three PHASE runs (CRITIC_PHASE True) and three OFF runs (CRITIC_PHASE False, the
switch-off control of THE ONE TEST), the same branch, HEAD, notebook link and every other setting. Started together
when six L4 sessions can run at once; else as THE ORDER says. Nothing here is decided by what a run prints during
training: no run is stopped, changed, restarted, extended or rerun because of a readout, an alarm, a fetch, the in-run
panel or anything the notebook prints beyond the configuration the launch check reads. Fails stay fails. Nobody moves,
edits or deletes a file in a run folder; no fetch writes to Drive."""),
 ("""   The branch: claude/screen-trex-r10-phasecritic, HEAD <HEAD> (as <PCR>/APPROVED records it; the tools and dry runs
   were last measured on e4d222bbde28800d7ee0851380123ff4ac9642a8). It is pushed only after the maintainer's OK so that
   REPO_REF can install it;""",
  """   The branch: claude/screen-trex-r10-phasecritic, HEAD <HEAD> (as <PCR>/APPROVED records it; the tools and dry runs
   were last measured on e4d222bbde28800d7ee0851380123ff4ac9642a8, the amendment's dry runs with the reworded HEAD
   1dfe2327f773007fa7fd68460dc0c53740936ff9, the same tree d834d833 [measured, ST/old_trees.txt, ST/new_trees.txt]).
   Both arms run this one HEAD. It is pushed only after the maintainer's OK so that REPO_REF can install it;"""),
 ("""THE RUNS
   Run    SEED   RUN_ID                   RUN_LABEL                    K3_PANEL_SIGMA_KEY
   s60    60     stance7_r10LTSBpc_s60    r10 LTSB phase-critic s60    source_vectors.60
   s42    42     stance7_r10LTSBpc_s42    r10 LTSB phase-critic s42    source_vectors.42
   s80    80     stance7_r10LTSBpc_s80    r10 LTSB phase-critic s80    control_sigma.80
   A relaunch (only as REPLACEMENT allows): the same RUN_ID with "_b" appended, every other setting the same. No "_c".
   [measured: no "stance7" or "r10LTSBpc" in RUNLOG (871 lines) and no "stance7*" folder in the ppo folder's last
   complete listing (2026-10-27T15:05:40Z, S1/foot/stance6/fetch/folders.json); DESIGN/names_check.out. Cell 8 must
   print "new run" (step 7): it is the check that counts.]""",
  """THE RUNS (in THE ORDER below: PHASE s60, OFF s60, PHASE s42, OFF s42, PHASE s80, OFF s80)
   Run        SEED  RUN_ID                    RUN_LABEL                    K3_PANEL_SIGMA_KEY  CRITIC_PHASE
   PHASE s60  60    stance7_r10LTSBpc_s60     r10 LTSB phase-critic s60    source_vectors.60   True
   OFF s60    60    stance7_r10LTSBoff_s60    r10 LTSB switch-off s60      source_vectors.60   False
   PHASE s42  42    stance7_r10LTSBpc_s42     r10 LTSB phase-critic s42    source_vectors.42   True
   OFF s42    42    stance7_r10LTSBoff_s42    r10 LTSB switch-off s42      source_vectors.42   False
   PHASE s80  80    stance7_r10LTSBpc_s80     r10 LTSB phase-critic s80    control_sigma.80    True
   OFF s80    80    stance7_r10LTSBoff_s80    r10 LTSB switch-off s80      control_sigma.80    False
   A relaunch (only as REPLACEMENT allows, for each run on its own): the same RUN_ID with "_b" appended, every other
   setting the same. No "_c".
   [measured: no "stance7" or "r10LTSBpc" in RUNLOG (871 lines) and no "stance7*" folder in the ppo folder's last
   complete listing (2026-10-27T15:05:40Z, S1/foot/stance6/fetch/folders.json); DESIGN/names_check.out. The OFF names:
   only RUNLOG line 875 (the coordinator's own note of this amendment, not a STANCE7 note) names them, no listed folder
   equals or prefixes them, no round name is a prefix of them or they of it [measured, ST/amend/names_check_off.out].
   Cell 8 must print "new run" (step 7): it is the check that counts.]"""),
 ("""   8737cd3d. FREEZE: after that push nobody pushes to the branch until all three launch checks read no-fault""",
  """   8737cd3d. FREEZE: after that push nobody pushes to the branch until all six launch checks read no-fault"""),
 (""" B1. Compute: read the L4's rate (Runtime > Change runtime type) and the account's balance. Needed: about 3 x 15.5
     L4-hours for the runs plus about 15.5 for one possible _b (about 62 L4-hours x the rate) [inferred from the
     stance6 runs: launched 14:57 UTC, progress "ended" 05:01-05:07 UTC (about 14.1 h), manifests 05:48-05:54 UTC
     (about 14.9 h) [measured, S2/r10round/f2/score.txt C4/C5 values; RUNLOG lines 693-698]].
 B2. Fetch budget: about 195 file copies through the Drive connector for F0, F2 and the readouts (as stance6) [model,
     ROUND TABLE M].""",
  """ B1. Compute: read the L4's rate (Runtime > Change runtime type), the account's balance and how many L4 sessions
     can run at once (note the number: THE ORDER uses it). Needed: about 6 x 15.5 L4-hours for the runs plus about
     15.5 for one possible _b (about 108 L4-hours x the rate) [inferred from the stance6 runs: launched 14:57 UTC,
     progress "ended" 05:01-05:07 UTC (about 14.1 h), manifests 05:48-05:54 UTC (about 14.9 h) [measured,
     S2/r10round/f2/score.txt C4/C5 values; RUNLOG lines 693-698]].
 B2. Fetch budget: about 390 file copies through the Drive connector for F0, F2 and the readouts (twice stance6's
     about 195) [model, ROUND TABLE M]."""),
 ("""FOR EACH RUN (all three at once, each in its own runtime)""",
  """THE ORDER (each run in its own runtime; the maintainer's sessions decide how many start at once)
 - Six L4 sessions at once: start all six in the order of THE RUNS, one after the other (PHASE s60, OFF s60, PHASE
   s42, OFF s42, PHASE s80, OFF s80), each with steps 1-8, each its own LAUNCH line.
 - Fewer than six: start the three PHASE runs first (s60, s42, s80), then the OFF runs in seed order (s60, s42, s80)
   in the sessions left; every OFF run that does not fit starts later, as soon as a session is free (a finished run
   releases its runtime: AUTO_DISCONNECT), in seed order. With three sessions this is two waves: wave 1 the three PHASE
   runs, wave 2 the three OFF runs as wave 1's runtimes free (the arm is then confounded with the wave: criteria KNOWN
   LIMITS). Nothing about a run's outcome decides when or whether an OFF run starts: every OFF run is started.
 - OUTCOME BLINDNESS: every OFF run starts before any PHASE run's results are opened. Until every OFF run's LAUNCH line
   is in RUNLOG: read no PHASE run's training output beyond the check rows of step 7 (liveness checks by Runtime >
   Manage sessions only), make no report-only snapshot of any run, and no F2 starts (it cannot: F2 waits for every run's
   end). After that, a report-only snapshot at the maintainer's request is noted at once as a SNAPSHOT line (RUNLOG
   NOTES). An OFF run whose LAUNCH line is not earlier than the first F2-START, REFETCH-START or SNAPSHOT line of a
   PHASE run is not blind: the scorer prints its classes and enters it in THE ONE TEST as missing (the lowest rank).
 - Each run's launch time is its own LAUNCH line (date -u to the minute, at step 6); the scorer prints each seed's
   PHASE-OFF start gap, report only. The 60-minute launch check and the 2-hour _b window count from each run's own
   LAUNCH line.

FOR EACH RUN (each in its own runtime, in THE ORDER)"""),
 ("""       SEED                = 60                          (s42: 42; s80: 80)""",
  """       SEED                = 60                          (s42: 42; s80: 80; the same for PHASE and OFF)"""),
 ("""       RUN_LABEL           = "r10 LTSB phase-critic s60" (s42 / s80 likewise)
       RUN_ID              = "stance7_r10LTSBpc_s60"     (s42 / s80 likewise)  <- the Drive folder name""",
  """       RUN_LABEL           = "r10 LTSB phase-critic s60" (PHASE; OFF: "r10 LTSB switch-off s60"; s42 / s80 likewise)
       RUN_ID              = "stance7_r10LTSBpc_s60"     (PHASE; OFF: "stance7_r10LTSBoff_s60"; s42 / s80 likewise)
                                                         <- the Drive folder name"""),
 ("""       CRITIC_PHASE        = True                        <- THE PHASE CRITIC (the 18th setting). Its default is
                                                         False: set it.
    Read SEED, RUN_ID, RUN_LABEL and K3_PANEL_SIGMA_KEY side by side (one seed), and CRITIC_PHASE once more.""",
  """       CRITIC_PHASE        = True                        <- PHASE runs: THE PHASE CRITIC (the 18th setting). Its
                                                         default is False: set it.
       CRITIC_PHASE        = False                       <- OFF runs: the switch-off control; False is the default:
                                                         leave it False and read it.
    Read SEED, RUN_ID, RUN_LABEL and K3_PANEL_SIGMA_KEY side by side (one seed), and CRITIC_PHASE once more against the
    run's row of THE RUNS (True with "pc" in the RUN_ID, False with "off")."""),
 ("""    - cell 6:  "Behavior recipe: stance", "In-run K3 panel: on" and "Critic phase: on" (the phase critic on;
               "Critic phase: off" means CRITIC_PHASE was left False: that run is not this round's
               configuration; record the row in the CHECK line: launch-check fact (8) decides, as L3 says)""",
  """    - cell 6:  "Behavior recipe: stance", "In-run K3 panel: on" and, for a PHASE run, "Critic phase: on"; for an
               OFF run, "Critic phase: off" (the other row in a run means CRITIC_PHASE was set wrong for its
               RUN_ID: that run is not of its arm's configuration; record the row in the CHECK line: launch-check
               fact (8) decides, as L3 says)"""),
 ("""    - cell 9:  "Critic phase on: [ppo] critic_phase = true on ['stance']" and
               "Behavior 'stance' -> deliverable 'stance'; chain: stance\"""",
  """    - cell 9:  for a PHASE run "Critic phase on: [ppo] critic_phase = true on ['stance']"; for an OFF run no
               "Critic phase on" row at all [measured, ST/check_code/nb/nb_off.log]; and, for both,
               "Behavior 'stance' -> deliverable 'stance'; chain: stance\""""),
 (""" L2. The ported launch-check (PY -I -B ST/tools/pc/complete_r10round.py launch-check --run <RUN_ID> --round <PCR>;
     pins ST/tools/pc/pins_pc.json) computes the stance6 round's facts (1)-(7) against the PC pins (HEAD; seed; the
     hyperparameters, task and plant pins; baseline not within 50 of 3766.6; #640's ledger columns; one sessions entry;
     run_id = RUN_ID) and (8) the phase switch ON in 01_stance/stage_config.json.""",
  """ L2. The ported launch-check (PY -I -B ST/tools/pc/complete_r10round.py launch-check --run <RUN_ID> --round <PCR>;
     pins ST/tools/pc/pins_pc.json) computes the stance6 round's facts (1)-(7) against the pins of the run's arm (HEAD;
     seed; the hyperparameters, task and plant pins -- an OFF run's hyperparameters digest is ROUND's own, pins_pc.json
     off_arm; baseline not within 50 of 3766.6; #640's ledger columns; one sessions entry; run_id = RUN_ID) and (8) the
     switch in 01_stance/stage_config.json as the arm sets it: ON for a PHASE run; for an OFF run no
     hyperparameters.critic_phase key."""),
 ("""AFTER A DISCONNECT: THERE IS NO RESUME (the PC criteria keep the stance6 round's all-or-nothing rule; CHOICE RESUME)
 - Within 2 hours of the launch and with no 01_stance/models/stage1_<steps>_steps.zip in the run's TRIGGER LISTING
   (the first periodic zip comes at 262,144 steps, about 20 minutes in [inferred: 12,001,280 steps in about 14.1 h]),""",
  """AFTER A DISCONNECT: THERE IS NO RESUME (the PC criteria keep the stance6 round's all-or-nothing rule; CHOICE RESUME),
FOR EACH OF THE SIX RUNS ON ITS OWN
 - Within 2 hours of that run's own launch and with no 01_stance/models/stage1_<steps>_steps.zip in the run's
   TRIGGER LISTING (the first periodic zip comes at 262,144 steps, about 20 minutes in [inferred: 12,001,280 steps in
   about 14.1 h]),"""),
 ("""REPLACEMENT (ONE RELAUNCH PER SEED) -- as stance6
 Only when: the trigger note is within 2 hours of the LAUNCH line; the _b starts within 1 hour after that note and after
 the first runtime's DELETED (or FOUND-GONE) line; and either (i) the first run ended by itself (TRIGGER-a or -b) and
 its TRIGGER LISTING holds no periodic zip, or (ii) the launch check stopped it for a fault (1)-(5), (7) or (8). Steps
 1-8 with RUN_ID "<RUN_ID>_b", every other setting the same; write its RELAUNCH line. The _b is then the seed's run.""",
  """REPLACEMENT (ONE RELAUNCH PER RUN: each of the six RUN_IDs at most one _b) -- as stance6
 Only when: the trigger note is within 2 hours of that run's own LAUNCH line; the _b starts within 1 hour after that
 note and after the first runtime's DELETED (or FOUND-GONE) line; and either (i) the first run ended by itself
 (TRIGGER-a or -b) and its TRIGGER LISTING holds no periodic zip, or (ii) the launch check stopped it for a fault
 (1)-(5), (7) or (8). Steps 1-8 with RUN_ID "<RUN_ID>_b", every other setting the same (CRITIC_PHASE as its arm's);
 write its RELAUNCH line. The _b is then that arm's run for the seed. An OFF run's _b is an OFF run: outcome blindness
 applies to its RELAUNCH line."""),
 (""" LAUNCH <RUN_ID>     seed=<n> runtime=<L4|...> head=<40-hex HEAD> REPO_REF=claude/screen-trex-r10-phasecritic""",
  """ LAUNCH <RUN_ID>     seed=<n> runtime=<L4|...> head=<40-hex HEAD> REPO_REF=claude/screen-trex-r10-phasecritic (the same
                     form for PHASE and OFF runs; CRITIC_PHASE=True or False as set)"""),
 (""" F2-START <RUN_ID>    REFETCH-START <RUN_ID>
 Any other line naming a RUN_ID is restated in this form before that RUN_ID's F2-START line (as stance6).""",
  """ F2-START <RUN_ID>    REFETCH-START <RUN_ID>
 SNAPSHOT <RUN_ID> note="<what was looked at>"   (a report-only look at a run's results, only after every OFF run's
                     LAUNCH line; read wherever it stands: outcome blindness)
 Any other line naming a RUN_ID is restated in this form before that RUN_ID's F2-START line (as stance6)."""),
 (""" - F0: within 60 minutes of each launch (the launch check).""",
  """ - F0: within 60 minutes of each of the six launches (the launch check; a staggered OFF run's F0 at its own launch)."""),
 (""" - No interim fetch for the reading. A report-only snapshot at the maintainer's request is allowed (copies outside
   <PCR>, read by no rule, acting on nothing), as RUNLOG lines 681 and 689 did for stance6.""",
  """ - No interim fetch for the reading. A report-only snapshot at the maintainer's request is allowed only after every
   OFF run's LAUNCH line and is noted as a SNAPSHOT line (copies outside <PCR>, read by no other rule, acting on
   nothing), as RUNLOG lines 681 and 689 did for stance6."""),
 (""" - F2: once every launched RUN_ID (each first run and each _b) has ENDED, starting at least 6 hours after RUNLOG's last
   ending note, and only after the F2 tools' sha256 are in RUNLOG (if they are not, F2 waits; the delay printed). Its
   F2-START lines go to RUNLOG before the first file. Per run, the stance6 order: (1) the listings of the run folder,
   01_stance/, 01_stance/models/, 01_stance/figures/training_history/, 01_stance/replays/; (2) artifact_manifest.json""",
  """ - F2: once every launched RUN_ID of both arms (each first run and each _b) has ENDED, starting at least 6 hours after
   RUNLOG's last ending note, and only after the F2 tools' sha256 are in RUNLOG (if they are not, F2 waits; the delay
   printed). Its F2-START lines go to RUNLOG before the first file. Per run, the stance6 order: (1) the listings of
   the run folder, 01_stance/, 01_stance/models/, 01_stance/figures/training_history/, 01_stance/replays/; (2)
   artifact_manifest.json"""),
 ("""   --run-id <RUN_ID> --label <step|final>, at every J label and final; about 7 s each [measured on CPU, smoke]),
   M1 at J/R1/RESOLUTION,
   M3 at each final (about 2 CPU-hours for the three), the completeness tool, the scorer; the reading to the maintainer
   at once.""",
  """   --run-id <RUN_ID> --label <step|final>, at every J label and final of all six runs (an OFF run's record reads the
   stock critic, C8); about 7 s each [measured on CPU, smoke]), M1 at J/R1/RESOLUTION of all six,
   M3 at each final (about 4 CPU-hours for the six), the completeness tool, the scorer; the reading to the maintainer
   at once."""),
 ("""TIMELINE IF LAUNCHED 01:00-02:00 UTC 2026-11-01 [inferred]
   launch + F0 + launch check            01:00-03:00
   progress "ended" / manifests          about 15:05-16:10 / 15:55-17:00   (AUTO_DISCONNECT releases each runtime)
   ending notes (from Drive, +30 min)    about 16:30-17:30
   F2 (>= 6 h after the last end note)   about 22:30-23:30, about 1.5 h of transfers (stance6: 12:25-13:55 UTC)
   measurements + scorer + checks        about 3-4 h after F2's last file""",
  """TIMELINE [inferred], with L the launch hour of the six runs (all at once)
   launch + F0 + launch check            L to L+2 h (six launch checks)
   progress "ended" / manifests          about L+14 h / L+15 h               (AUTO_DISCONNECT releases each runtime)
   ending notes (from Drive, +30 min)    about L+15.5 h
   F2 (>= 6 h after the last end note)   about L+21.5 h, about 3 h of transfers (stance6: 1.5 h for three runs)
   measurements + scorer + checks        about 5-6 h after F2's last file
   With fewer sessions the OFF runs' times move later by their start gap (up to about 15 h with three sessions)."""),
]


def main_colab() -> int:
    f = ST / "COLAB_STEPS.txt"
    got = hashlib.sha256(f.read_bytes()).hexdigest()
    if got != COLAB_BEFORE:
        raise SystemExit(f"STOP: COLAB_STEPS.txt sha256 {got} is not the re-reviewed {COLAB_BEFORE}")
    new = apply(f.read_text(encoding="utf-8"), COLAB, "COLAB_STEPS")
    f.write_text(new, encoding="utf-8")
    print(f"COLAB_STEPS.txt: {len(COLAB)} edits; sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")
    return 0


# ==================================================================================================== DESIGN_NOTES.txt
DESIGN_BEFORE = "ff89e0d4228f0add173e0ad5844d711692b5f4a5f4007b39c0dee590dc37c77a"
DESIGN = [
 ("""   the first time, so it needs the builder's tests and the independent review before the push. The comparison is
     historical, 3 against 3, not concurrent (section 2). Agents can only read Drive, and F2 cannot start before about
     22:30 UTC on 2026-11-01.""",
  """   the first time, so it needs the builder's tests and the independent review before the push. The comparison is
     historical, 3 against 3, not concurrent (section 2). [Superseded by the maintainer's CONTROL (b) answer, section
     9: three concurrent switch-off runs are added and are THE ONE TEST's control; STANCE6 is report only.] Agents can
     only read Drive, and F2 cannot start before about 22:30 UTC on 2026-11-01."""),
 ("""       - Exchangeability is assumed, not designed. The runs are not concurrent (2.1), and the caveat is printed with
         the result.""",
  """       - Exchangeability is assumed, not designed. The runs are not concurrent (2.1), and the caveat is printed with
         the result. [Superseded, section 9: the test now compares the PHASE runs with three concurrent switch-off
         runs (same HEAD, notebook, seeds, schedule); a missing PHASE run enters at +infinity, a missing OFF run at
         -infinity; STANCE6's values are printed beside, report only.]"""),
 ("""   - No concurrent control runs (CHOICE CONTROL).""",
  """   - No concurrent control runs (CHOICE CONTROL). [Superseded, section 9: the maintainer chose (b); the six-run
     design is in the amended criteria and ST/amend/AMEND_NOTES.txt.]"""),
 ("""   - Historical control: 2.1.""",
  """   - Historical control: 2.1 (now report only; what concurrency removes and does not: section 9 and the criteria's
     KNOWN LIMITS)."""),
 ("""9. THE CHOICES THE MAINTAINER CONFIRMS (with the answer time to the second; recommended first)""",
  """9. THE CHOICES THE MAINTAINER CONFIRMS (with the answer time to the second; recommended first)
   ANSWERED 2026-11-01 01:15:14 UTC (RUNLOG line 874: "MAINTAINER (AskUserQuestion, two dialogs answered just before
   this line)"; recorded here by the amendment editor, RUNLOG line 875) [measured, RUNLOG lines 874-875]:
     NAMES    confirmed: RUN_IDs stance7_r10LTSBpc_s60/_s42/_s80, tag STANCE7 ("Confirm all (Recommended)"). The
              CONTROL (b) answer adds the switch-off runs stance7_r10LTSBoff_s60/_s42/_s80, labels "r10 LTSB switch-off
              s<seed>" (the names the coordinator's line 875 gives; no collision [measured,
              ST/amend/names_check_off.out]).
     PUSH     confirmed: push claude/screen-trex-r10-phasecritic at 1dfe2327 (the reworded HEAD; trees equal to
              e4d222bb's [measured, ST/old_trees.txt, ST/new_trees.txt]) and a durable copy of the pre-registration.
     CONTROL  (b): "Add 3 switch-off runs" (6 L4 runs, about 92 L4-hours), with "Make them the test's control
              (Recommended)": THE ONE TEST compares the phase runs with the concurrent switch-off runs, decided before
              any data; STANCE6 report only; a criteria and scorer edit plus a quick re-review before pinning.
     TEST     (a) keep the one test (Recommended), now against the concurrent switch-off runs.
     INIT     (a) zero-appended critic phase weights ("Confirm all").
     K3       (a) the in-run K3 panel ON ("Confirm all"; VP2 passed on e4d222bb, the same tree).
     RESUME   (a) the round's rule, no resume ("Disconnects: Round's rule, no resume (Recommended)").
     ENDS     run ends noted from Drive ("Confirm all").
     TOOLS, HDR not in the two dialogs (RUNLOG line 874 names neither): the text keeps the recommended options (the
              F0 and launch-check port pinned before launch, the F2 tools before F2; no header commit); the pin step
              asks the maintainer to confirm them with the pin, or records that they stand as reviewed.
     SWITCH   as the builder made it (cell 6 CRITIC_PHASE; rows "Critic phase: on" / "Critic phase: off"; cell 9 "Critic
              phase on: ..." for a phase run only [measured, ST/check_code/nb/nb_on.log, nb_off.log]).
   What CONTROL (b) changes, in short (every change and why: ST/amend/AMEND_NOTES.txt):
     - six runs: the phase arm (CRITIC_PHASE True) and the switch-off arm (CRITIC_PHASE False, its default), same HEAD,
       notebook, seeds, schedule and settings; interleaved by seed; with fewer sessions the switch-off runs may start
       later, but each before any phase run's results are opened (outcome blindness; a late one enters the test as
       missing); start gaps printed, report only; each run handled on its own (all or nothing, one _b within 2 h of its
       own launch).
     - THE ONE TEST: phase vs concurrent switch-off, phase LOWER, TD unchanged, minimum p 1/20, alpha 0.05; a missing
       phase run at +infinity (highest rank), a missing switch-off run at -infinity (lowest). TD's form was chosen
       with STANCE6's values in view; the six runs' values are unseen at the pin.
     - THE READING: the phase runs' lines and per-run classes as reviewed; the switch-off runs' classes printed beside;
       the line texts that named STANCE6 now name the concurrent switch-off runs; STANCE6 report only (beside, and in
       an OFF-vs-STANCE6 replication block: with the switch OFF the code equals 8737cd3d bitwise, A6, so the
       switch-off runs replicate stance6 on today's Colab image).
     - What concurrency removes: the code-commit, date / Colab-image (within the start gaps), notebook and K3
       differences between the test's arms. What it does not: 3 vs 3 still reaches only p = 1/20; GPU nondeterminism
       within today's image; image changes between staggered starts; seed effects (a shared start, A5) assumed to make
       the unpaired test conservative.
   The options as offered (kept for the record):"""),
]


def main_design() -> int:
    f = ST / "design" / "DESIGN_NOTES.txt"
    got = hashlib.sha256(f.read_bytes()).hexdigest()
    if got != DESIGN_BEFORE:
        raise SystemExit(f"STOP: DESIGN_NOTES.txt sha256 {got} is not the re-reviewed {DESIGN_BEFORE}")
    new = apply(f.read_text(encoding="utf-8"), DESIGN, "DESIGN_NOTES")
    f.write_text(new, encoding="utf-8")
    print(f"DESIGN_NOTES.txt: {len(DESIGN)} edits; sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
