<!-- Working paper, 16 Nov 2026: the integrator's report on writing the lead's rulings of 16 Nov (TRACKB_RULINGS3_16NOV.md) into the Track B rules draft. Nothing here is a rule. -->

## Result

Nine of the ten rulings are written into `/home/user/mesozoic-wt/prereg-b/hardware/trex_form_study/STAGE2_TRACKB_PLAN.md` as ruled. Ruling 6 is written in too, but its two `V6noF5` caps need the lead: the stand-in estimate they rest on is too low.

I ran no episode, fit or training. Every number below is a rule's text or a planning estimate from the bar runner's dry run. None is simulation evidence, and the study has no hardware measurement.

Merge (step 1): `git merge claude/mesozoic-trex-stage2-refine-4jffre` had no conflict. The branch now holds the working branch up to `82ba1ad`, and only `STAGE2_TRACKB_PLAN.md` differs from it.

Marking: new text carries "[settled 16 Nov 2026, before any Track B run]", never split over two lines. Replaced sentences are quoted once as "in place of the committed sentence ... (settled 15 Nov)". Section L marks also say the look work package had built look sets by then (prescribed motion, none frozen). Line numbers are the file's at `d17afd6`.

## Per ruling

**1. M5 (lines 702-721); change list item 11.**
- Old: "Every run that passed G8a is judged on the panels at its last kept checkpoint, whatever ended it".
- New: "Every run of the Colab queue that passed G8a is judged ...".
- New paragraph: S2-4 (M3) and the local 60 cm run (M6) are not runs of the queue and are never judged on the panels.
- The brief already said "Every Colab run".

**2. L13 (1507-1548).**
- Old: "... and flat feet (L10)".
- New paragraph: "The turn in place has flat feet as its own law: no toe-off and no rolling entry in any stride, the start and the stop included." I worded the lead's gloss as "Rule L10 is about a walk's moves (its starts and changes, and its stop and turns from their command) and does not govern this gait", because L10 also covers causal moves.
- One Check sentence added (`foot_pitch` 0 and `rolling` false in every frame of the gaits, starts and stops).
- **Check, as the ruling assumes.** In `wt/wp-l` at `e2696b2`, `designlab/stage2/sets_look.py` `Builder.inplace` builds the gait with `strip_f5` (`toe_off` 0, `toe_reach` 0; defaults in `designlab/gait.py` lines 116 and 119). The foot law (`moves_look.py` lines 578-640) then writes no pitch for the long plan or the stop.
- On its unfrozen build of `V6` on `75-M-H-10` I read the six files (two gaits, two starts, two stops): `foot_pitch` max 0.0 and `rolling` all false in each.
- For the look work package, not changed by me: its rule texts `S2L-inplace` and `S2L-periodic` still cite `S2L-L10` for the flat feet.

**3. L10 (1352-1400), L11 (1433, 1448).**
- Old clause in L10: "the frames before the instant equal the look walk's, foot pitch included". L11 repeated it as "and the frames before it equal the look walk's, foot pitch included".
- New paragraph in L10: the lead's three sentences, word for word. L11's clause now follows it.
- L10's account now says the committed sentence on frames "before and at the command" stands for a causal move alone (A6).
- L10's Check: the one test is split in two (causal move; before a change's instant).

**4. L12 (1457-1505); the brief (line 181); change list item 30.**
- Old: "A move that still fails under the cuts open to it is "not available" with its numbers (A8 step 3; A11 for a turn)".
- New: a `start` or a `nominal_to_v0.4` that still fails stops that body's set, which is then not frozen, and goes to the lead (A8 stands). Any other move is "not available": a ladder change, the stop, a turn, the turn in place.
- The brief's "a move that still fails is recorded "not available"" gains the two exceptions. L12's Check gains the exception.

**5. U4 (771-810), Q6 (1033-1058); change list items 28 and 29.**
- U4, new paragraph: on `V6` and `V6noF5` reading 1 lives in the companion and is the verdict of a Track B row; the bar's own record is printed beside it as "the bar's reading (C7)"; where they differ on a stop item both are printed with the term's name. On any other set the bar's own readings judge, and `on_command_at_stop` is recorded and decides nothing.
- Q6, the one sentence: `R2L-F25F9cl/42` is judged by the bar's own readings, exactly as `R2/42` and `R2/43` beside it.
- Nothing was replaced. The 15 Nov sentence "An episode that fails it succeeds in no reading" is now scoped to the companion's readings on the look sets.

**6. Q5 (940-1031), pointer in Q6, the brief's Compute bullet (193), change list item 15.**
- Written as ruled: the cap formula, the bar's own `budget.json`, the cut order (6, 4, 0 ms), 10 ms never cut.
- The draft's "R2 re-runs" are `R2/42` and `R2/43` on `F25F9cl` from the R2L row's own commit (Q6), as the ruling assumes. Not handed back.
- Dry runs: made 16 Nov at `82ba1ad`, with `--cost stage2/runs/2026-11-15-75cm/COST.md` (R2 rate 0.1574 wall s per simulated s, measured by this study on 15 Nov). They ran no episode and wrote nothing. The commands are in the rule.

| Cap for | Dry run on | Episodes | Printed total, core-hours | Cap = 1.5 x (my arithmetic) |
|---|---|---|---|---|
| `R2L-V6noF5/42`, 75 cm, 17 conditions at 10 ms | `V0`, row `R2/42` (stand-in set and row) | 340 | 0.296 | 0.444 |
| the same with 0, 4, 6 ms | `V0`, row `R2/42` | 1,360 | 1.185 | 1.7775 |
| `R2L-V6noF5/42`, 60 cm | none: the 75 cm numbers "until its own rates exist" | - | 0.296; 1.185 | 0.444; 1.7775 |
| `R2L-F25F9cl/42`, Q6's 22 records | `F25F9cl` itself, row `R2/42` | 440 | 0.344 + 0.128 = 0.472 | 0.708 |
| `R2/42` and `R2/43` beside it, 44 records | `F25F9cl`, the rows themselves | 880 | 0.688 + 0.256 = 0.944 | 1.416 |

**7. M1 (594-627), K5 (1095-1105), pointer in M4.**
- M1, new paragraph: the lead's wording (attempt logged before the monitor starts with the file hashes, closed with the exit code; "once" counts closed attempts without a record; a never-closed attempt is an interruption).
- M1's Check rewritten to match. M4's "twice leaves no record" now reads "two closed attempts that left no record".
- K5 gains the interruption sentence, plus one sentence of mine: a check M1 has counted is not run again by a resume.

**8. Y10 (1162-1170), Y11 (1172-1189); change list item 21.**
- Old: "The smoke passes when Y1, Y3, Y4, Y6 and Y9 pass."
- New: the lead's sentence (Y4 passes or is answered by `QUEUE.json` `budget_by: owner`, with the date).
- Y11: "(r under 208.34)" became "(`r x 72,000 < 15,000,000`)".
- In Y11's quotation of the committed sentence, "(r under 208.33)" is now "(...)" with a note. That alters a quotation, as "both go" asks. No "208." is left in the file.

**9. Q5 (950-960).**
- "the only one that may pass the share" gains "without limit".
- Old: "The reported cells run in Q4's order while the share lasts; a started cell finishes."
- New paragraph: the lead's rule (estimate = the decisive cell's mean seconds per episode x 25 x 1.25; else "not run: budget"; excess recorded as `overrun_reported`; no further cell starts).

**10. X1 (1236-1243).** The first sentence gains "(lifted for the frozen 75 cm sets on 16 Nov 2026, after the panels)", with a pointer to the lead's amendment 13 of `STAGE2_75_PLAN.md`. Nothing else in X1 changed.

**The frame, mine and listed as the author's in the text:** a Status sentence, a bullet in "What had been run", the Tags entry, a brief bullet, and an "Amendment of 16 Nov 2026: the third list" closing the last section, so that "the last section" still points right.

## Handed back to the lead

1. **Ruling 6, the `V6noF5` caps.** The same dry run on the look work package's unfrozen `V6noF5` build of `75-M-H-10` (`--set-root`, planning only, no record) printed 0.567 core-hours for the 17 conditions. That is 1.9 times the stand-in's 0.296 and above its cap of 0.444. Episodes there are planned at 24.87 to 52.89 s, against 14.06 to 21.08 s on `V0`. The rule holds the stand-in numbers as instructed, labelled as estimates, with a note saying this. Options:
   - (a) keep the stand-in caps;
   - (b) take each cap from the dry run on the frozen `V6noF5` set, before its first R2L panel record;
   - (c) fix it now from the unfrozen build: 0.8505 at 10 ms and 3.402 with all four delays (my arithmetic).

   I recommend (b): it is what "per set and body" says.
2. **Ruling 6, smaller choices I made, each reversible.**
   - The candidate's dry runs are on `F25F9cl` itself, not `V0`. On `V0` they print 0.284 + 0.111, which gives a cap of 0.5925.
   - The candidate counts Q6's 22 records, without `walk/nominal`. With it the cap is 0.729.
   - "Where the plan asks" on `V6noF5` is read as S2-9's fallback only (R5 failing G8b on both seeds).
   - The ruling does not say what is recorded when 10 ms records alone pass a cap, nor which run directory's `budget.json` holds the cap.
3. **Ruling 7, a new conflict, not settled.** K5 says a resume "moves later entries to `discarded_after_resume`". An attempt entry is always later than its checkpoint's `run_state`, so the attempt log that M1 counts would be discarded. Proposed sentence: "the attempts logged for a check on the checkpoint resumed from are kept". My K5 sentence (a counted check is not run again) is a reading, not the lead's wording.
4. **Ruling 3, a place that still says it differently.** L8's Check reads "on the shared strides the move equals the resampled window (join rules J1 and J2)". For a start or change those frames are now "the move's own" within A7's gates. Not changed.
5. **Ruling 2.** L11 still says "Inside every move (start, change, stop, turn) the feet are flat by rule L10", which literally covers the turn in place's own start and stop. The outcome is the same; not changed.
6. **Ruling 4.** "That body's set" is singular, while L12 takes a cut for every look set of a body and its `-X` head. Written as ruled.
7. **Seen, not mine.** The draft says the 15 Nov rulings "are session notes, not in the repository". Files named `stage2/design/TRACKB_RULINGS_15NOV.md` and `TRACKB_RULINGS2_15NOV.md` are tracked on the working branch.

## Commits (branch `wt/prereg-b`, not pushed, no history rewritten)

- `287357f` merge of the working branch
- `500d73d` rulings 1 and 7
- `dbf5d25` rulings 5, 6 and 9
- `405c218` rulings 8 and 10
- `7ef3fc4` rulings 2, 3 and 4
- `d17afd6` the third list's frame

`git status --porcelain` is empty. The main checkout and `wt/wp-l` are untouched. Dry-run outputs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/dry6/log/`.
