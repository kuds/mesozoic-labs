<!-- Working paper, 16 Nov 2026: the fixer's report on applying that review. Nothing here is a rule. -->

## Result

The review is applied to `/home/user/mesozoic-wt/prereg-b/hardware/trex_form_study/STAGE2_TRACKB_PLAN.md`. Seven integration defects are fixed in two commits; fourteen points are left for the lead and nothing was changed for them.

I ran no episode, fit, training or dry run. Dry-run totals below are the reviewer's logs (planning estimates, no record). Nothing here is simulation evidence or a hardware measurement. Line numbers are the file's at `c4a0ff8`.

## Per finding

**1. Q5, the `V6noF5` and 60 cm caps (ruling 6): for the lead; one phrase fixed.**
- Fixed (Q5, table line 985 and lines 1015-1020): "until the 60 cm body's own rates exist" is gone. Ruling 6 names `COST.md`'s rates only. No number changed.
- Confirmed in `designlab/stage2/bar.py` lines 1826-1851: a unit is refused when spent plus estimate passes the cap. A 10 ms unit planned at 0.567 core-hours against a cap of 0.444 would be cut, against "never cut".
- The reviewer's logs print 0.567 (unfrozen `V6noF5`, 75 cm), 2.268 (the same, four delays) and 0.549 (unfrozen `V6`, 60 cm).
- Options: (a) keep the stand-in caps; (b) take each cap from the dry run on the frozen set itself, committed before the row's first panel record; (c) fix them now from the unfrozen build (0.8505 and 3.402, my arithmetic: 1.5 x 0.567 and 1.5 x 2.268). I recommend (b), with the reviewer's text.

**2. Q5, five undecided cases (ruling 6 as a rule): for the lead.** Not a defect of the integration: the draft holds the ruling's words.
- Confirmed: `--task` takes 3 or 4 only (`bar.py` lines 140, 3468); another cap needs `budget_cap_core_h` (line 2328).
- Confirmed in the 75 cm round's `budget.json`: task 3's attribution units used 1.055 core-hours beside 3.191 for its other units. The dry-run total leaves attribution out.
- I recommend the reviewer's text (units in the order 10, 0, 4, 6 ms; the bar's unit rule; one `budget.json` per cap), with attribution "run last and cut first". The Check's cut order then follows.

**3. K5, a resume discards the attempts M1 counts (ruling 7): for the lead; the label fixed.**
- Fixed (K5, lines 1106-1109): the integrator's sentence is now labelled as the author's reading, with what the ruling does not say.
- Options: (a) K5 keeps the attempts logged for a check on the checkpoint it resumes from; (b) the attempts are counted in the checkpoint's own record. I recommend (a), the reviewer's sentence.

**4. M1 and M4, three undecided cases (ruling 7 as a rule): for the lead.** I recommend the reviewer's three sentences (a counted check's 18 episodes are *errors*; a halt after two never-closed attempts; what "a record" is).

**5. L8's Check against L10 (ruling 3): for the lead.** Not changed: ruling 3 names L10 and its account only, and L8 is a design rule the lead let stand.
- Options: (a) the reviewer's wording (a causal move "equals"; a start or change "within A7's limits"); (b) say what is equal: on the shared strides the move's times and slow variations are the window's, and its frames are read by the join gates.
- I recommend (b). The look package's own test of L7 and L8 compares times and tail variation, not frames (`wt/wp-l` at `e2696b2`, `tests/test_stage2_moves_look.py` lines 420-459). Join rule J2 compares within `join_b`'s limits for a causal move too.

**6. L10 and L11, one boundary and one test (ruling 3 as a rule): for the lead.**
- Boundary (lines 1363, 1448): I recommend the reviewer's "ends after that instant and no later than the end of the move's last ramp". It agrees with the committed clause on the swing that lifts at the command, and with the look package's test (`ph > c + 1e-9`, line 478).
- Test: I recommend the reviewer's Check wording (lift-off `foot_pitch` and `rolling` as gait a's spec gives them on the move's own step times).

**7. L12's Check wider than ruling 4: fixed (L12, line 1509).** It now reads "a set in which one of those two still fails holds no freeze (A8)". Whether the body's other look sets may then be frozen is for the lead: the failing set only, or every look set of the body and its `-X` head. I recommend the failing set only.

**8. U4 (ruling 5).**
- Fixed (U4, lines 796-798): the sentence names the row `R2L-F25F9cl/42`, as the ruling does; it read "a Track B row".
- For the lead: what "printed beside" means where the bar has no reading 1 (walks and turns). Options: stop items only, or every item with the bar's readings 2 and 3. I recommend stop items only, the reviewer's Check.
- For the lead: whether the companion's counts carry the term on other sets. I recommend "the companion's counts there equal the bar's".

**9. Q5, "the share still holds" (ruling 9): for the lead.** Options: 4.0 core-hours less everything the fit has spent, or the 0.4 part less the fit's excess. I recommend the first; the same rule defines the share as 4.0 per fit.

**10. Y10 (ruling 8): for the lead.** I recommend the reviewer's text (`budget_set_on`, written when the queue is made and before its first job). Y11's "(...)" follows "both go"; I recommend keeping it.

**11. Small.**
- M5, "the Colab queue" undefined (line 720): for the lead; it is the ruling's own term. I recommend the reviewer's gloss.
- L13, the gloss on L10: fixed (line 1537 holds the ruling's sentence; the wider note stands in the author's account, lines 1541-1542).
- L11, "every move" (lines 1440, 1458): for the lead. Ruling 2 names L13 only, and L11 is the lead's look choice. I recommend adding "but the turn in place's start and stop (L13)" in both places.
- L12, the unflagged clause: fixed (lines 1477-1482). The mark quotes the whole committed sentence, and the clause again reads "a move that still fails".
- "Session notes, not in the repository" (lines 275, 283, 1737, 1821): older than this integration, so the four sentences are left as written. I added one factual note in the closing account (lines 1985-1988). For the lead: leave them with the note, or correct them with one dated sentence. I recommend the dated sentence.
- Line 38, "the commit": fixed. It now reads "the last of the commits that add the wording settled on 16 Nov 2026".

The closing account (lines 1976-1989) lists these changes as the author's.

## For the lead

1. Ruling 6: where the `V6noF5` caps come from, 60 cm included (finding 1). Recommend the dry run on the frozen set itself.
2. Ruling 6: run order, "reached", a 10 ms unit over its cap, the run directory and task key, attribution (finding 2).
3. Ruling 7: K5's resume and M1's attempt log (finding 3). The K5 sentence on a counted check is still the integrator's reading.
4. Ruling 7: three undecided cases in M1 and M4 (finding 4).
5. Ruling 3: L8's Check (finding 5).
6. Ruling 3: the stance that ends exactly at the instant, and the test of "follows the law" (finding 6).
7. Ruling 4: the body's other look sets when one set's start fails (finding 7).
8. Ruling 5: what is printed beside reading 1 on walks and turns; the companion's counts on other sets (finding 8).
9. Ruling 9: what "the share" is (finding 9).
10. Ruling 8: the field and time of the owner's budget answer; Y11's "(...)" (finding 10).
11. Ruling 1: a definition of "the Colab queue".
12. Ruling 2: L11's "every move" and the turn in place's start and stop.
13. The 15 Nov sentences on rulings "not in the repository". Lines 17 and 224 say the same of the designs, and `a0314fc` added design files too.
14. Seen, older than this integration: `stage2/design/TRACKB_RULINGS2_15NOV.md` is headed 03:50 UTC, the draft says that list was written in at 03:43 UTC (line 81), and the commit that wrote it in (`984f81d`) is dated 04:01:56 UTC.

For the look package's review, passed on from the reviewer and not checked by me: `moves_look.py` line 599 still builds L10 as committed, and `move.command_phase` is null for `start` and `nominal_to_v0.4`.

## Commits (branch `wt/prereg-b`, not pushed, no history rewritten)

- `8accf61` the review applied: U4, Q5, K5, L12, L13
- `c4a0ff8` the frame: the Status's Check and the closing account

Branch head: `c4a0ff8`. `git status --porcelain` is empty. Only `STAGE2_TRACKB_PLAN.md` changed; the main checkout and `wt/wp-l` are untouched.
