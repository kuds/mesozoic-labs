<!-- Working paper, 16 Nov 2026: the fixer's report on applying REVIEW.md to the design note. Nothing here is a rule. -->

## Corrected note

`/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_fix/SERVO_MODEL_DESIGN.md` (675 lines, sha256 `e84f9389…`).

- Only this copy was edited. The tracked note is byte-identical to before and `git status` is clean.
- Code lines are those of the merged files at commit `82ba1ad`. The head moved from `b57fdd0` during the work; only `stage2/design/checkpoint/STATUS.md` changed, and `designlab/` is unchanged since `26b057e`.
- Status line and header changed as asked. Sections 1 to 9 keep their structure and marks; one mark `[R]` (the reviewer's figures) is added.
- New section 10 is "Changes after the independent review"; new section 11 is "For the owner, in plain words" (23 lines).
- My arithmetic is in `calc_fix.py` in the same folder.

## Findings: 25 (2 blockers, 8 major, 14 minor, the owner's question)

- **Fixed, nothing left open: 17.** B1, M1, minors 1 to 14, the owner's question.
- **Fixed in the text, with a choice left to the lead or owner: 7.** B2, M2, M3, M4, M5, M6, M7.
- **Handed to the lead as a choice alone: 1.** M8.
- **Rejected: 0.** Every cited file, line and record was opened, and the reviewer's substance held each time.

Five details of the review are corrected or not taken:
- **Minor 4:** the worktree edits are now committed on `wt/wpc-ce` at `4523ed6` (still unmerged), not "uncommitted".
- **Minor 9:** in the merged file the lines are `r5env.py:475-476`, not 478-480.
- **B2:** the third-party unit being "worn" is not repeated; reader 3's summary does not say it and the paper was not opened.
- **B1:** a zero nominal for a signed encoder error does not contradict the maker's accuracy figure.
- **Owner text:** the reviewer's "anywhere in the table's ranges" is not used; 20 or 40 drawn runs do not show that.

## What changed in section 9 and the corner rule

- **Section 9** no longer says a maker's value is the nominal. By my count of the 38 table rows: 1 is the maker's own figure, 4 are a maker's figure for a related quantity, 6 are third-party, 27 are the study's.
- **Bench:** no sentence says the bench measures something the sheet lacks. N1 to N6 are named as proposed additions to `BENCH_SHEET.md`, and gate 2's "gear play" sentence has two proposed wordings.
- **Condition 2 (section 4):** the 0.30 m/s walk on the grippy floor, at 4 and 10 ms, both policies, seeds 3042 to 3061.
  - Seven judged settings: friction low and high, play 0 and 1.0° on the leg drives, trunk mass 0.9 and 1.1, and a seeded drawn panel.
  - That is 28 records and 560 runs.
  - A corner passes on 20 of 20 staying up; the count meeting every limit is recorded. The drawn panel must pass all of H1.
  - Reported, no verdict: high rotor inertia, low bus, gain limits, today's torque-speed line, the X6-50-L case, two combined corners, the slippery floor.
- **Statistics** (my arithmetic, runs taken as independent): at a true 99.9 % stay-up rate, all 560 stay up 57 % of the time (53 % for the first version's 640 runs). 20 of 20 at one setting bounds the rate above 86 %.

## Open choices for the lead (section 8), with my recommendation

1. **Maker questions (choice 1).** Ask all five before gate 1 is judged: inertia side, gain ranges and units, X6 at 24 V, which X6 ships, which encoder closes the loop.
2. **Which settings can stop a purchase (choice 9, the owner's question in section 11).** Recommended: (a) the six corners plus the drawn panel, with high inertia and low bus reported. Options: (b) all eight corners judged; (c) only nominal and the drawn panel judged.
3. **What "passes" at a corner (choice 22).** Stay-up only at corners, the bar's whole rule on the drawn panel.
4. **Route and names of gate 1's records (choice 18).** Recommended: (c) a runner outside `bar.py`. `bar.py` cannot run a served or policy row outside tasks 3 and 4. Options: (a) units of task 3 or 4; (b) a new bar task, which moves the bar's hash.
5. **Judged delay under the model (choice 19).** Keep the command lag and the 100 Hz hold in the nominal model and say so; a measured feedback age goes to `sense_steps`, not into T.
6. **X6 low end (choice 3).** Also draw the X6's zero-torque speed (8.01 to 9.215 rad/s) if the maker has not answered; the single corner at 2.15 does not reach the back-EMF line.
7. **Tail, condition 4 (choice 16).** Read it at the worse of nominal and high friction, with the tail's nominal inertia at the same fit's 0.028 kg·m². The neck's same asymmetry is kept and flagged until N5.
8. **Torque-scale draw (choice 20).** None now; say that A2's 0.70 is used up.
9. **Stand-in rows' ankle cap under drawn inertia (choice 21).** Accept.
10. **Steps below 20 V (choice 10).** Report in every record; do not fail judged runs yet.
11. **Mass corners (choice 11).** Policy rows need a nominal-model estimator before M- and M+ can be judged; condition 2 as proposed depends on this.
12. **Bench sheet (choice 15).** Add N1 to N6 and reword the gate draft's lines 59-61 and 67.

## Not verified

- Any maker page, and the maker's protocol document.
- The third-party paper behind the sibling's play and torque-constant figures (summary only).
- The reviewer's one-joint toy probe and the designer's timing probe (not rerun).
- Whether tendon friction acts as assumed once physics runs; only the forward-pass row counts were rerun (14, 18, 14).
- The cost of the 560 judged and 1,120 reported runs.

## Reruns and deviations

- Rerun by me, read-only: `probe_ref_accel.py`, which reproduced the reviewer's tail table, and `probe_runtime_arrays.py`.
- Two text-handling one-offs (splicing the copy, checking table columns) ran with the system python outside `tools/lowprio.sh`. They touched only the scratchpad copy.

## Noticed in passing, outside this job

`stage2/design/ROUND_C_DESIGN.md:759-760` says a look-set run needs no `--task`. `bar._check_rule` (`bar.py:2202-2205`) refuses any non-R0 row on a set that is not one of the three registered ones. I did not check this against later rulings.
