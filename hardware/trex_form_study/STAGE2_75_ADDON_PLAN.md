# The add-on to the 75 cm panels: rules written before its runs

16 Nov 2026. Body `75-M-H-10`, set `V0` (today's walk). **Simulation only.** Paths are relative to
`hardware/trex_form_study/`.

This file holds the rules of the runs the owner asked for after the 75 cm panels (`DECISION_LOG.md` P37 and P38;
`STAGE2_75_REPORT.md`). Each section is committed before its first episode. Everything here is **added after the
results of the 75 cm panels and labelled so**. The rules of `STAGE2_75_PLAN.md` hold unless a section says otherwise.
No verdict of the 75 cm panels changes.

**Where each item stands.**

| Item | What | State |
|---|---|---|
| A | Today's walk at more delays: 14, 16 and 20 ms, then 12 and 18 ms and the slow walk at 8 ms | **registered below**; run first (P38) |
| B | The carpet range, no verdict | **not registered.** A design and its independent review are on file (`stage2/design/addon/design_AB.md`, `review_AB.md`: one blocker and nine major points, on B). It is run only if there is room (P38), after its rules are rewritten, reviewed and committed here |
| C | The refit of the stand-in controller with delay, two fits | **not registered.** A design and its review are on file (`stage2/design/addon/design_C.md`, `review_C.md`: twelve major points). Run only if there is room (P38), after its rules are rewritten, reviewed and committed here |

The designer's and the reviewer's texts are working papers. Only this file is a rule.

## A. More delays (P37, answer 1, first part)

**The question.** Is 10 ms a special point for the stand-in controller (Stage 1's fitted tracker, rows `R2/42` and
`R2/43`, fitted with no delay), or does delay simply get worse? In the registered records the 0.40 m/s walk stays
up in 20 of 20 at 0 to 6 ms and in 0 to 3 of 20 at 8 and 10 ms under both fits on both floors. In Stage 1 this
body's slow walk fell at 10 ms and stayed up at 20 ms, meeting no limit there (three episodes, `R2/43`;
`DECISION_LOG.md` F26).

**A-1. What is run.** Set `V0`, body `75-M-H-10`, block `seeds3042` (seeds 3042 to 3061), 20 episodes per record,
nominal plant, commanded 5 % fast (B1). Conditions: `walk/v0.4`, `walk/v0.4_mu0.54` and `walk/nominal`. Rows:
`R2/42@<ms>ms`, `R2/43@<ms>ms`, `R0@<ms>ms` with the gains of `stage1/runs/2026-11-08/75-M-H-10/tune.json`
unchanged (B2). That is 9 records and 180 episodes per delay.

**A-2. Order and units.** One unit per delay, in this order: `A1-V0-14ms`, `A1-V0-16ms`, `A1-V0-20ms` (the three
delays the owner approved; `STAGE2_PLAN.md` S2-9 names them and the round listed them as not run); then
`A2-V0-12ms` and `A2-V0-18ms`, each as A-1; then `A3-V0-nominal-8ms`: `walk/nominal` under `R2/42@8ms` and
`R2/43@8ms`. **The A2 and A3 units are the lead's addition** to what the owner approved, inside the same budget and
placed last so that a cut removes them first: with them every 2 ms step from 0 to 20 ms has a record of the
0.40 m/s walk, and the slow walk has one at 8 ms, the last delay before it falls. The run directory is
`stage2/runs/2026-11-16-75cm-addon/A/`. The rates are those of the committed
`stage2/runs/2026-11-15-75cm/COST.md`.

**A-3. Budget and stop rule.** 1.0 core-hour for all of section A, on top of H6's 16 + 12 core-hours, counted as
the sum of episode wall times in `stage2/runs/2026-11-16-75cm-addon/A/budget.json`. A unit starts only if the spent
core-hours plus its estimate (the runner's dry-run total at the committed rates) fit 1.0. The first unit that does
not fit is cut with every unit after it. A started unit finishes. The runner's own ledger names its task "3" and
prints a cap of 16 core-hours: that is the stock runner's slot and constant. The cap of this section is held by the
chain script `stage2/runs/2026-11-16-75cm-addon/run_addon.sh`, which writes every unit's estimate, the core-hours
spent before it, and whether it started or was cut, into the committed file
`stage2/runs/2026-11-16-75cm-addon/A/ADDON_LEDGER.json`. An episode that failed to run is run again once by the
same command (C8); a record that still holds one is not read. After the first record of this section exists, no
delay, seed, row or condition is added or run again except as a new, labelled addition.

**A-4. Code state.** The units run with the runner's key sources as they stand in the registered records of
`stage2/runs/2026-11-15-75cm/` (code state A): the two hashes every record's key holds, `bar`
`45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d` and `panels`
`c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. The chain script computes them with the runner's own functions,
refuses to start on other values or on a checkout with uncommitted changes outside the add-on's run directory, and
writes them into `ADDON_LEDGER.json`. No module that enters a record key is changed before these units have run.

**A-5. What is judged.** Nothing. No verdict comes from this section and none changes. The bar stays judged at
10 ms (P30, P32, P37). A cell gives its counts and, by H2, "passes at <ms> ms" or "does not pass at <ms> ms",
never `shown` or `provisional`.

**A-6. What is reported, and how it is read** (stated before any result).

**Reading words.** They describe a count. They are not passes and not verdicts.

- A record is **up** when at least 17 of its 20 episodes stay up. It is **down** when at most 3 do. Otherwise it is
  **mixed**. The two numbers were chosen after the registered counts were seen: 3 is the most that stayed up in any
  registered `R2` record of this walk at 8 or 10 ms, and 17 borrows H1's number without being H1's rule (H1 asks for
  20 of 20 up and 17 that meet every limit).
- A record that is up carries a second word. It is **up and covering the distance** when at least 17 of its 20
  episodes stay up *and* meet the distance term of their rule (the bar's `along` for the 0.40 m/s walk; Stage 1's
  `speed` term, as recorded in `measures.stage1`, for the slow walk). Otherwise it is **up, short of the distance**.
  Staying up without walking as commanded is not walking.
- A delay is **up** when all four `R2` records of the 0.40 m/s walk are up (two fits, two floors). It is **down** when
  all four are down. Otherwise it is **mixed**, and the four counts are printed.
- By these words the registered delays read: 0, 2, 4 and 6 ms up; 8 and 10 ms down.

**The patterns.**

- **"Delay simply gets worse."** Every delay run above 10 ms reads down. Then nothing from 8 ms up holds, and 10 ms is
  not special.
- **"10 ms is a special point" (a window).** At least one delay run above 10 ms reads up, while 8 and 10 ms read down.
  The report then names the lowest delay above 10 ms that reads up, and says of each such delay whether its four
  records are up and covering the distance. Where they are not, the sentence reads "a window of staying up, not of
  walking".
- **"Neither."** No delay above 10 ms reads up and at least one is mixed. The report prints the counts and uses
  neither phrase.

The slow walk is read the same way from its two `R2` records, in its own paragraph. The two paragraphs are not merged.

**What a window would and would not mean.**

- It would not make any delay a verdict. The bar is judged at 10 ms (P30, P32, P37).
- It would not mean the build is fine at that delay. Stage 1's three 20 ms episodes met no limit set (above).
- It would describe the survival curve. It would not explain it. No mechanism is tested by these runs.
- It would bear on item C: a refit with delay in its mix would then be moving a narrow failure, not buying margin
  against a steady trend. That is the lead's reading to make, as an interpretation.

**Where before why.** Every count is printed with where the falls come (the start from standing, the slow stride,
the change of speed, the 0.40 m/s walk itself), from the episodes' `segment_reached` and fall times. At 10 ms 76 of
80 episodes fell and none fell in the 0.40 m/s walk itself (`STAGE2_75_REPORT.md` section 5). So a 0.40 m/s record
at a longer delay may say nothing about walking at 0.40 m/s.

**Of the episodes that stay up** the report prints, per record and not judged: the median speed, the toe clearance,
the failed terms with their counts, the largest drive speed ratio, and for the slow walk the peak ground force.

**Expectations, stated from results already known.**

1. `R0` rows: 18 to 20 of 20 stay up at every delay, as at 0 to 10 ms. `R0` has no feedback, so a delay only shifts
   its commands in time (design estimate). An `R0` record under 17 is a flag on the runner at long delays, not a
   finding about the controller. The `R2` cells of that delay are then printed with the flag and reviewed before
   they are read.
2. The slow walk under `R2/43` at 20 ms: most episodes stay up, from Stage 1's 3 of 3. The plans differ: Stage 1's
   walk is start, walk, stop over 18.4 s; the bar's is start then 8 strides over 14.06 s (my dry run). A record that
   reads down would contradict Stage 1's three episodes and is reported as a contradiction, with both records.
3. `R2/42` at 20 ms, every row at 14 and 16 ms, and the 0.40 m/s walk at 20 ms: no expectation can be stated. No
   record exists.
4. No record at 14 to 20 ms is expected to meet every limit in 17 of 20. From 6 ms up the registered records fail
   toe clearance in 19 or 20 of 20, and Stage 1's 20 ms episodes failed speed, cadence, stride, ground force and
   drive speed.

The registered records at 0 to 10 ms under `stage2/runs/2026-11-15-75cm/` are printed beside the new ones, each
with its path. Nothing is read between delays (B3).

**A design estimate, labelled as such, for why a window is possible at all.** A pure delay turns the phase of a
loop component at frequency f by 360° × f × delay, so the same phase returns when the delay grows by 1/f. The
tracker's feedback is computed at 100 Hz and ramped in over 10 ms (`designlab/stage1/control.py:20-22`). A component
that alternates from one update to the next has a 20 ms period: 10 ms of delay flips its sign and 20 ms restores it.
If that were what fells the walk, the curve would mirror about 10 ms: 12 like 8, 14 like 6, 16 like 4, 18 like 2,
20 like 0. This is arithmetic, not a result. The baseline loop already lags (sensors one step old, the ramp), so the
centre need not sit at exactly 10 ms. It is why 12 and 18 ms are worth their cost (2.5).

**A-7. Not attributed.** These records take no fall attribution (`--attribute none`). In the registered round the
attribution of records where 0 of 20 stayed up rested on a change of one physics step (`STAGE2_75_REPORT.md`
section 5).

**A-8. Labels.** Every table and sentence from these records carries: "simulation; stand-in controller fitted
with no delay; delays above the judged 10 ms; no verdict; added after the 75 cm results (P37)".

**A-9. Not run.** Turns, pushes and stops at these delays; the speed ladder's other rungs; `walk/nominal_mu0.54`;
the sets `F25F9` and `F25F9cl` (P37, answer 3); the block `seeds5042`; delays above 20 ms (the runner's limit);
any delay that is not a whole 2 ms step.

**A-10. Where the result goes.** A short section added to `STAGE2_75_REPORT.md`, labelled as A-8 says, with one
table made by script from the records (`stage2/runs/2026-11-16-75cm-addon/A/delays_table.py`, in no key), checked
by one independent reviewer against the records before the owner sees it.
