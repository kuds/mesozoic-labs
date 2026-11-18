<!-- Working paper, 16 Nov 2026: the independent review of that integration. Nothing here is a rule. -->

## Verdict

The draft can be adopted as the pre-registration's text for these rules **after the listed fixes**. Rulings 1, 2, 3, 4, 5, 8, 9 and 10 are written in faithfully. Findings 1 to 3 need the lead's word (ruling 6's caps, ruling 7's resume). The rest are small wording changes.

I ran no episode, fit or training. Dry-run totals below are planning estimates printed by the bar runner at `COST.md`'s R2 rate (0.1574 wall s per simulated s, measured by this study on 15 Nov). Array readings are on the look package's unfrozen builds (prescribed motion, no record). Nothing here is a hardware measurement. Line numbers are the draft's at `d17afd6`.

## Findings, most serious first

**1. Q5, lines 983-985 and 1014-1025 (ruling 6): the `V6noF5` and 60 cm caps cannot work as written. For the lead.**
- My re-run of the dry run, same command and commit `82ba1ad`: stand-in `V0` 0.296 core-hours; the unfrozen `V6noF5` on `75-M-H-10` 0.567; the unfrozen `V6` on `60-N-T-10` 0.549 (my probe, not in the integrator's report).
- The bar's own ledger refuses a unit whose estimate does not fit: `bar.py` lines 1826-1851 and 2510-2511 (`spent + estimate > cap` raises `BudgetCut` before the unit starts). A 10 ms unit estimated at 0.567 against a cap of 0.444 is cut. That contradicts "the 10 ms records are never cut".
- "Until the 60 cm body's own rates exist" (line 1015) is not in the ruling, which names `COST.md`'s rates for every set and body.
- Options are the integrator's (a), (b), (c). I recommend (b). Proposed text: "A `V6noF5` cap is 1.5 times the total this dry run prints on the frozen set itself, at the same `COST.md`: `--set V6noF5 --body <body> --rows R2/42@10ms --conditions @gated,walk/nominal` (the three other delays added where S2-9 asks). It is made and committed with its printed total after the set's freeze and before the row's first panel record. The stand-in numbers are planning figures and cap nothing."

**2. Q5, lines 968-973 and 1029-1031 (ruling 6 as a rule): five cases are undecided. For the lead.**
- No run order is stated. The cut order only works if units run 10, 0, 4, 6 ms.
- "Reached" is not defined (the bar's rule is: a unit starts only if spent plus estimate fits; a started unit finishes).
- Nothing says what is recorded when the 10 ms unit alone passes the cap.
- No run directory or task key is named. The CLI's `--task` takes 3 or 4 only, with caps 16 and 12 (`bar.py` lines 140, 3468); another cap needs `bar.run(budget_cap_core_h=...)` (line 2328).
- The ledger also counts attribution's lifted episodes and C8's re-runs (`STAGE2_75_PLAN.md` H6), which the dry-run total leaves out. In the 75 cm round's committed `budget.json`, task 3's attribution units used 1.055 core-hours beside 3.191 for its panel units.
- Proposed text: "Each cap is spent by the bar's unit rule (H6). The 10 ms records are one unit and start first; then one unit per delay, in the order 0, 4, 6 ms. The first unit that does not fit is cut with every later one. If the 10 ms unit passes the cap, the excess is recorded and no further unit starts. Attribution of these records is [not run / run last and cut first]. Each cap has its own `budget.json`, in [directory], under one task key."
- The Check's "cut units: 6 ms, then 4 ms, then 0 ms" should read "the cut units are the tail of the order 10, 0, 4, 6 ms".

**3. K5, lines 1097-1103 (ruling 7): the resume discards the attempts that M1 counts. For the lead.** I confirm the integrator's point 3.
- K5 still says a resume "moves later entries to `discarded_after_resume`" and "runs any check that ... has no record naming that checkpoint". An attempt is always logged after its checkpoint's `run_state`.
- So the new sentence "A check that M1 has counted ... is not run again by a resume" cannot hold, and M4's "a function of the monitor records alone" fails for a counted check.
- Proposed: "... moves later entries to `discarded_after_resume`, but keeps the attempts (M1) logged for a check on the checkpoint it resumes from, ...".
- That last K5 sentence is the integrator's, yet it stands inside the lead's mark. Label it or have the lead confirm it.

**4. M1 and M4, lines 615-619 and 691-694 (ruling 7 as a rule): three cases are undecided. For the lead.**
- A check counted without a record has no per-condition counts, so M4's table cannot say which tasks are doubled. Proposed: "A check counted without a record counts each of its 18 policy episodes as an *error*."
- Never-closed attempts have no bound: a check that kills the session each time is run again by every resume and never counts. Proposed: "After two attempts at one check that were never closed, the run halts (`HALT.json`) and goes to the lead."
- "Left no record" is undefined when a file exists but is incomplete. Proposed: "A record is a `monitor_<n>.json` that names the checkpoint's hashes and holds an outcome for all 18 episodes."

**5. L8's Check, lines 1348-1350 (ruling 3): it now contradicts L10.** I confirm the integrator's point 4.
- It says a move "equals the resampled window" on the shared strides.
- In the unfrozen `V6` `nominal_to_v0.4`, `join_a` reads CoM 0.093 mm, ZMP 0.60 mm and joints 2.0e-4 rad against the walk: inside A7, not equal.
- Proposed: "on the shared strides a causal move equals the resampled window (join rules J1 and J2); a start or a change agrees with it within A7's limits (`join_a`, `join_b`), its foot pitch as L10 says."

**6. L10, lines 1358 and 1388-1389; L11, lines 1443-1444 (ruling 3 as a rule): one boundary and one test.**
- A stance that ends exactly at the instant: the new sentence gives it the walk's toe-off, while "ends between that instant and the end of the last ramp is flat", read inclusively, makes it flat. In the unfrozen change, both stances ending at phase 0.000 roll (0.299 rad at lift-off, 14 rolling frames). Proposed: "ends after that instant and no later than the end of the move's last ramp".
- "Follow the law of the walk it leaves" names no comparison. Equality with the walk's file cannot be the test: the entry's `variation_a_until_phase` is -1.0, so step times fade over the stride before the instant. Proposed Check: "every stance that ends at or before the instant has the lift-off `foot_pitch` and `rolling` frames that gait a's spec gives on the move's own step times".

**7. L12's Check, line 1502 (ruling 4): wider than the ruling.** "No set of a body is frozen while one of those two still fails" can be read as every set of the body; the ruling says "that body's set". Proposed: "a set in which `start` or `nominal_to_v0.4` still fails holds no freeze (A8)". Whether the body's other look sets may be frozen is the lead's. I recommend the failing set only, since the point goes to the lead anyway.

**8. U4, lines 794-798 and 808-810 (ruling 5 as a rule).**
- The Check prints the bar's count beside "each count", but the bar has no reading 1 for walks and turns (`counts_look: null`), and C7 is the stop item's rule. Proposed: "each stop item of a Track B row shows the companion's count and, beside it, the bar's own count headed "the bar's reading (C7)"".
- On other sets, say whether the companion's own counts carry the term. Proposed: "the companion's counts there equal the bar's".
- Line 797 widens the ruling's "the row `R2L-F25F9cl/42`" to "a Track B row" inside the lead's mark.

**9. Q5, line 958 (ruling 9).** "The share still holds" is undefined: 4.0 less everything spent, or the 0.4 part less the fit's excess. Proposed: "(4.0 core-hours less every second the fit has spent: blocks, decisive cell, earlier reported cells)".

**10. Y10, lines 1164-1165 (ruling 8).** The owner's answer lives in `QUEUE.json`, the file that queues jobs, while "until it passes no training job is queued"; and "with the date" has no field. Proposed: "(`QUEUE.json` `budget_by: owner` with `budget_set_on`, written when the queue is made and before its first job starts)". Y11's "(...)" alters a quotation; it is labelled; the lead's call.

**11. Small.**
- M5, line 720: "the Colab queue" is not defined. Add: "the jobs of a launch's `QUEUE.json`; not the smoke's jobs, N5's tiny jobs or K3's two runs".
- L13, line 1530: the gloss "a walk's moves (..., and its stop and turns from their command)" is the integrator's rewording inside the lead's mark, and the closing paragraph does not list it.
- L11, lines 1435 and 1453: "every move" and "every move entry" still take in the turn in place's start and stop. Add "but the turn in place's start and stop (L13)".
- L12, line 1474: "; no limit is changed for it" became ". No limit is changed for such a move", unflagged.
- Lines 275, 283, 1730, 1814: "the rulings are session notes, not in the repository" is false on the working branch. `TRACKB_RULINGS_15NOV.md` and `TRACKB_RULINGS2_15NOV.md` are tracked since `a0314fc`.
- Line 38: "the commit that adds the wording settled on 16 Nov" is five commits.

## Checked and found right

- **History.** The merge `287357f` did not change the draft. The five commits touch only the draft. `git status` is clean.
- **Nothing else changed.** A word-level diff ignoring wrapping gives 74 changes, all traced to a ruling or the frame.
- **Label.** 41 occurrences, none split over a line. Each replaced sentence is quoted once (M5, Y10, Q5, L10, L11, L12, L13).
- **Ruling 2.** `Builder.inplace` (`sets_look.py` lines 598-694) builds from `strip_f5`. The six files have `foot_pitch` 0 and `rolling` false, and their sha256 match the committed entries.
- **Ruling 6.** Every dry-run number reproduces: 0.296; 1.185; 0.344 + 0.128; 0.688 + 0.256; 0.567; and the alternatives 0.284 + 0.111 and 0.358 + 0.128. Episode counts, simulated hours and the 1.5 x arithmetic are right.
- **Ruling 10.** Amendment 13 of `STAGE2_75_PLAN.md` exists and says what the draft cites.
- **Other citations.** A5, A7, A8, A11, C7, H3 and the plan's section 6 and S2-9 are quoted correctly. No "208." is left. Both `set.json` files have `freezes` empty.

## Seen, for the look package's review (not the integration)

- `moves_look.py` line 599 still builds the committed L10 ("begins at or after the last ramp ends"). In the unfrozen change the right stance from +0.869 to +1.500 is flat; L10 as settled on 15 Nov says it rolls.
- `move.command_phase` is null for `start` and `nominal_to_v0.4`; the instant is in `look.move_begins_phase`. L10's and L11's Checks name `move.command_phase`.

## Not checked

- Track B's own code (`train2`, the monitor, the companion wrapper) and the look package's tests.
- `ROUND_C_DESIGN.md`: the draft does not cite it.
- `STAGE2_PLAN.md` beyond section 6 and S2-8 to S2-10.
- Rulings 11 to 13.
- The 60 cm dry run at all four delays.

Notes, logs and probes are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_trackb3/`: the word diff is `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_trackb3/wdiff.txt`, and the dry-run outputs are under `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_trackb3/dry/log/`.
