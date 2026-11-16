<!-- Working paper: the independent reviewer's report on section A of STAGE2_75_ADDON_PLAN.md and on run_addon.sh, as returned on 16 Nov 2026. Nothing here is a rule. The reviewer's proposed script change is review_A_run_addon_fix.diff. -->

# Independent review of section A (rules A-1 to A-10) and run_addon.sh

STUDY = /home/user/mesozoic-labs/hardware/trex_form_study. Scratch = /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/addon_review_A.

## Verdict

**Yes, after the listed fixes.** Section A must not be run as registered. One blocker is in the rules: the slow walk's reading (A-6) fits none of the three patterns in the outcome the plan itself expects. Each A1 unit makes slow-walk records, so this cannot be fixed after the first unit. Eight major points follow: four in the rules, four in the script. The script is right on a clean run with no kill, no failure and no cut. It is wrong or fails open on the other paths. The code state is right today, and the cost holds under the cap.

HEAD moved from 7282da2 to 876df7b during the review (three commits by another session: BUY_GATE_DRAFT.md, DECISION_LOG.md P39, STATUS.md). The plan, the script, designlab, the sets, stage1 and the registered run are identical at both commits (my `git diff --quiet`). I committed nothing.

## Findings

### Blocker

**1. A-6, plan line 92 ("The slow walk is read the same way").** Blocker.
- **Problem:** The window pattern needs "8 and 10 ms read down". For the slow walk, 8 ms is expected up, and its only record is unit A3, which is cut first. With 8 ms up or missing, 10 ms down and any later delay up, none of the three patterns applies.
- **Evidence:** Stage 1 ledger, latency.json, `walk/nominal|latency_steps=4` (8 ms): 3 of 3 up, failing toe clearance only. Registered V0 `walk/nominal`: up at 0, 4 and 6 ms; 2 and 1 of 20 at 10 ms; no record at 2 or 8 ms. A-2 itself calls 8 ms "the last delay before it falls".
- **Fix:** Replace the sentence with: "The slow walk has its own paragraph, from its two `R2` records per delay: a delay is up when both are up, down when both are down, otherwise mixed with both counts. Its registered delays read 0, 4 and 6 ms up and 10 ms down; 2 ms has no record; 8 ms is unit A3. 'Delay simply gets worse': every delay run above 10 ms reads down. 'A window': at least one delay run above 10 ms reads up; the 8 ms word is printed and is no condition. 'Neither': none reads up and at least one is mixed. The two paragraphs are not merged."

### Major, rules

**2. A-6 expectation 1 (lines 112-115) and A-1.** Major.
- **Problem:** An `R0` record under 17 means the `R2` cells are "reviewed before they are read". No rule says what the review is. That is decided after a result (review_AB finding 10, not addressed). Also, "as at 0 to 10 ms" has no source for `R0` on the slow walk, and that row is an undeclared addition (review_AB finding 12(5), not addressed).
- **Evidence:** No registered V0 `walk/nominal` record exists under `R0` at any delay (my table). STAGE2_75_PLAN.md line 775 lists "`R0` rows on the ladders" as not run.
- **Fix:** "An `R0` record of the 0.40 m/s walk under 17 is a flag. That delay then takes no word and enters no pattern; its cells are printed with the flag; nothing is run again. The check is fixed here: a no-physics test of `speedfit.servo_delay` at that k, and `controller.latency_steps` = ms / 2 in the records. `R0` on the slow walk is the lead's addition; its only source is Stage 1's `eval/R0/walk/nominal.json` (20 of 20, 0 ms, another plan); it is printed and raises no flag."

**3. A-6, the mirror estimate (lines 128-134).** Major.
- **Problem:** Five predictions are stated, with no statement of what refutes them. "The centre need not sit at exactly 10 ms" lets any result fit. Review_AB finding 10 asked for this.
- **Fix:** Add: "The report prints these five predictions beside the result whatever it is: 'met' when 12 ms reads down and 14, 16, 18 and 20 ms read up, on the delays that were run; otherwise 'not met', naming the delays that differ. No other centre is fitted after the result." Also remove the dangling "(2.5)".

**4. A-3 (lines 50-51) with script lines 102-109.** Major.
- **Problem:** The rule says a failed episode is run again "once". After two exits with code 4 the script stops the whole chain (`exit 1`) and leaves no marker. The next `all` runs the unit a third and fourth time. No rule says whether later units run, or what word a delay gets when one of its records is "not read".
- **Evidence:** Script line 109; no `.incomplete` marker exists (the registered run_75cm.sh has one).
- **Fix, rule:** "If the second run still holds a failed episode, the unit is marked `incomplete` in `ADDON_LEDGER.json`, is not started a third time, and the chain goes on. A delay with a record that is not read has no word, is printed 'not read', and enters no pattern." Script lines are in the diff named under finding 6.

**5. A-8 (lines 140-141), A-2, the item table (line 15).** Major.
- **Problem:** The label "delays above the judged 10 ms" is false for unit A3 (8 ms). The label "(P37)" and the table's "run first (P38)" put the lead's additions under the owner's decisions. A-2 states the A2 and A3 addition plainly. It does not say that the slow walk and the `R0` rows inside A1 go beyond the registered walk unit (`@walk` only). P38 puts the training path directly after the 14, 16 and 20 ms walks; A2 and A3 run before it.
- **Evidence:** P37 and P38 in DECISION_LOG.md lines 52-53. My estimate for A2 + A3 is 0.281 core-hours at committed rates.
- **Fix, A-8:** "…; a delay that is not the judged 10 ms; no verdict; added after the 75 cm results. Records of A1: '(P37: 14, 16 and 20 ms)'. Records of A2 and A3: 'the lead's addition, not among the delays the owner approved'."
- **Fix, A-2:** Add: "In A1 the slow walk and the `R0` rows are the lead's reading of 'today's walk'. A2 and A3 run before the training path of P38: about 0.28 core-hours."
- **Fix, table:** "A1 run first (P38); A2, A3: the lead's addition."

### Major, script

**6. Lines 90-99 and 110: a resumed unit.** Major.
- **Problem:** After a kill the script takes the dry run's "to run" figure and tests the cap again. A started unit can then be cut, against A-3 ("a started unit finishes"). The ledger row also gets the remaining estimate and a spent-before figure that includes the unit's own use.
- **Evidence:** Unchanged copy, `list`, fake budget.json, 6 of 9 records present:
  - Started at 0.86: `A2-V0-18ms [to run]: estimate 0.032 core-h, spent before 0.976000, cap 1.0, fits 0`. Line 98 then cuts it (read, not run).
  - All 9 records present: `estimate 0.000 core-h, spent before 0.960000`.
- **Fix:** A unit that A/budget.json already holds has started. Skip the cap test and take its first `estimate_core_h` and `spent_before_core_h` from budget.json. Use the dry run's Total, as A-3 says.
- **Corrected copy and diff:** `<scratch>/sandbox_fixed/stage2/runs/2026-11-16-75cm-addon/run_addon.sh` and `<scratch>/run_addon_fix.diff`. Its `list` on the same state prints `A2-V0-18ms [started before]: estimate 0.128000 core-h, spent before 0.860000 … fits 1`. Its `all` branches were not run.
- **Fix, A-3:** Add: "The estimate is the dry run's Total, taken at the unit's first start. A unit that `budget.json` holds is resumed without a new cap test. The total may end above 1.0 by the last started unit's overrun."

**7. Lines 92-93: the cap test fails open.** Major.
- **Problem:** If `spent` fails, SPENT is empty and the test becomes `+ EST <= 1.0`, which is true.
- **Evidence:** Copy with an unparsable budget.json: `A2-V0-18ms [to run]: … spent before , cap 1.0, fits 1`, exit 0. Direct: `python -c "print(1 if  + 0.128 <= 1.0 + 1e-9 else 0)"` prints 1. An empty FITS takes the cut branch at line 98.
- **Fix:**
  ```
  SPENT=$(spent) || fail "$NAME: $RUN/budget.json cannot be read"
  case "$SPENT" in ""|*[!0-9.]*) fail "$NAME: spent '$SPENT' is no number" ;; esac
  FITS=$(…) || fail …; [ "$FITS" = 1 ] || [ "$FITS" = 0 ] || fail …
  ```

**8. Lines 50-69, 87, 99, 109, 110: the ledger.** Major.
- **Problems:**
  - The status of `ledger` is never checked. An unparsable ledger gives status 1 and the chain goes on (harness test).
  - The write is not atomic (line 67).
  - The done marker is written before the row (line 110). A kill between the two leaves a finished unit with no row for good.
  - No row exists while a unit runs, so a killed and abandoned unit is absent. A-3 says "whether it started".
  - A unit cut after a cut unit gets `estimate_core_h: 0.0` (line 87; harness output), not its estimate.
- **Fix:** Write through a temporary file and `os.replace`. Use `ledger … || fail …`. Write a `started` row before the runner and the `done` row before the marker. Compute the estimate before the cut branch. All are in the diff; the helper was tested in a harness.

**9. Line 38: the dirty-tree check.** Major.
- **Problem:** It excludes all of `$ADD`, so an uncommitted edit of run_addon.sh itself passes. A-4 says "outside the add-on's run directory", which A-2 defines as `…/A/`. If `git` fails, the output is empty and the check passes.
- **Evidence:** Scratch git repository: the script modified gives "passes"; with `:(exclude)$ADD/A` it gives "REFUSES ( M …/run_addon.sh)".
- **Right as is:** The run's own outputs under A/ pass. A changed key module or plan refuses. MUJOCO_LOG.TXT and logs/ are ignored, so a resume does not wrongly fail.
- **Fix:**
  ```
  DIRTY=$(git status --short -- . ":(exclude)$RUN") || { say "FAILED code state: git status did not answer"; exit 1; }
  [ -z "$DIRTY" ] || { say "FAILED …"; echo "$DIRTY" | head; exit 1; }
  ```

### Minor

10. **Lines 94-97.** `list` does not show what `all` would do. With spent 0.90 it prints `A2-V0-18ms … fits 0` and then `A3-V0-nominal-8ms … fits 1`, but `all` cuts A3. Fix: the diff's list branch.
11. **Lines 90 and 104.** The runner inherits the loop's stdin, which is the unit list. I found no stdin read in the runner's sources (grep), and `list` showed all six units. Fix: add `< /dev/null` to both calls.
12. **Lines 26-37.** The code-state check runs once and covers the two hashes only. `key.context` (library versions, plant inputs) is not checked; it is equal today. `PY` can be overridden. Two `all` at once would write a `failed-exit-1` row for a running unit. Fix: compare `panels.key_context('75-M-H-10')` with a registered key, and add a chain lock.
13. **A-1.** B2 limits rows to k = 0..5; A-1 uses k up to 10 without saying so. No test runs `speedfit.servo_delay` above k = 5 (review_AB 12(2), not done). My stub check at k = 5 to 10 passes. Fix: "B2's range is widened here to k ≤ 10 (the runner's limit)", and add the test.
14. **Plan line 27, "meeting no limit there".** F26 says "none succeeding". The ledger shows each episode failed 5 terms: speed, cadence, stride, grf, drive_speed. Fix: "none meeting every limit".
15. **A-5.** The runner has three words; "one of two passes at <ms> ms" is missing, and "does not pass" is not H2's text (bar.py 2754-2756). Fix: list the three.
16. **A-6 line 83.** "Then nothing from 8 ms up holds" reads between delays if A2 is cut (B3). Fix: "no delay that was run from 8 ms up holds".
17. **A-6 line 103.** "`segment_reached` and fall times" names two inputs and no rule. `segment_reached` alone reproduces the report's table (8 ms: 5/39/36; 10 ms: 29/12/35, my recount). Fix: "by `segment_reached` alone; fall times are printed and classify nothing".
18. **A-6 second word.** Name the fields: `episodes[].fell`, `success.along`, `measures.stage1.success.speed`. The slow walk's own rule has `along` too (the record's `terms`). The two agree on every surviving registered episode. Use `along` for both, or say why not.
19. **A-6 expectations.** "From 6 ms up" holds at 6 ms only (20, 20, 19, 20); at 8 and 10 ms 0 to 3 stay up. "14 to 20 ms" leaves out 12. A mixed slow-walk record at 20 ms is neither "most stay up" nor a contradiction; say which. "(my dry run)" names nobody. "The toe clearance" and "the peak ground force" need a statistic (median).
20. **A-9.** "(P37, answer 3)" covers F25F9cl only, not F25F9.
21. **A-10.** `A/delays_table.py` does not exist. Commit it before the first unit, outside A/.
22. **A-2 / A-3.** The estimate is not stated. Add: 0.128 per delay, 0.025 for A3, 0.665 in all; about 0.79 at the registered records' rates.
23. **Line 16 of the plan.** "nine major points, on B": findings 8, 9 and 10 of review_AB are on A.

## Checked and found right

- **Code state.** All 594 registered records hold the pair bar 45719d03…528d and panels c0027992…e1c1. Today's values, the script's WANT_KEYS and A-4 are equal. The script calls the same two functions as `evaluate` (bar.py 2407-2408). Full keys recomputed for 14 registered V0 records equal the registered keys (no episode, no settle). The V0 freeze check passes.
- **Order and commands.** Six units in A-2's order. Set, body, rows, conditions, `--attribute none`, `--task 3`, `--cost`, `--stage1-run` and `--out` are as A-1; seeds 3042-3061 and 20 episodes by default. Dry run: 9 records and 180 episodes per delay; 40 for A3.
- **Cap.** The estimate is parsed from the Total line. Spent is read from tasks."3" (null-safe). A `.cut` marker cuts every later unit and survives a restart on the same machine. Without the marker the same arithmetic cuts again. The ledger is never read for a cut.
- **Pipe subshell.** CUT holds across iterations. `exit 1` reaches the script's exit. A final `continue` gives status 0. (Standalone snippets.)
- **Ledger on a clean path.** It holds the cap, both hashes, and per unit the estimate, spent-before, state and used.
- **Review_AB finding 8.** Addressed in A-3; in the script on the clean path only (findings 6 and 8).
- **Review_AB finding 9.** Addressed. `along` and `measures.stage1.success.speed` exist in `walk/v0.4` and `walk/nominal` records.
- **Counts.** 20 of 20 at 0 to 6 ms; 0 to 3 at 8 and 10 ms; registered words 0-6 up, 8 and 10 down; `R0` 18 to 20 on the 0.40 m/s walk; 76 of 80 fell at 10 ms, none in `v0.4`.
- **Stage 1 ledger.** 10 ms: 3 of 3 fall. 20 ms: 3 of 3 up, 0.240 m/s, planned 18.39 s. The bar's slow walk is 14.06 s.
- **Cost.**

| | Core-hours |
|---|---|
| Per delay, committed rates | 0.1281 |
| A3 | 0.0246 |
| A1 | 0.384 |
| All of A (940 episodes) | 0.665 |
| All of A, registered full-length rates | 0.785 |
| All of A at the ledger's worst ratio (1.42) | 0.945, all six units start |

## Not checked

- No `bar run` and no mode `all`, on the original or on any copy. The `all`-only lines (87, 98-112) were read; only their bash forms were tested. The corrected copy's `all` branches were not executed.
- No episode. So not checked: that the runner accepts the command at run time, that it leaves stdin alone, and the rates at 12 to 20 ms.
- That Stage 1's `ledger._latency` equals `speedfit.servo_delay` at k = 10.
- The test suite, sections B and C, and design_AB.md.
- A-2's S2-9 citation was read only as far as the line naming 14, 16 and 20 ms.

One note: A/logs/run_addon.log holds one FAILED line of 01:17:04Z from an earlier wording of the script, before the commit. No unit has started; A/ holds no budget.json and no record.

## Commands run

All from STUDY with the stated environment; longer ones through tools/lowprio.sh.

- `bash stage2/runs/2026-11-16-75cm-addon/run_addon.sh list` (exit 0, six units, 0.128 × 5 + 0.025).
- `bar dry-run` for unit A1-V0-14ms and unit A3.
- Read-only Python scripts in scratch: rec_shape.py, rec_read.py, rec_table.py, latency.py, latency2.py, key_check.py, full_key.py (plant.build with settle=False only), queue_check.py (stub, no physics), and a cost recomputation.
- The unchanged copy of the script in `<scratch>/sandbox`, `list` only, with fake budget.json, markers and records.
- funcs.sh and funcs_fixed.sh: the `ledger` and `spent` functions extracted verbatim and called directly.
- sem.sh: standalone bash snippets.
- A scratch git repository for the pathspec.
- make_fixed.py and `list` on the corrected copy.
- git status, log, diff, ls-files and check-ignore, read only. `git status --short` gives 0 lines at the end.
