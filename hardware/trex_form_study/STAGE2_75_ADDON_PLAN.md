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
| A | Today's walk at more delays: 14, 16 and 20 ms, then 12 and 18 ms and the slow walk at 8 ms | **registered below.** A1 (14, 16, 20 ms) is run first (P38); A2 and A3 are the lead's addition |
| B | The carpet range, no verdict | **not registered.** A design and its independent review are on file (`stage2/design/addon/design_AB.md`, `review_AB.md`: one blocker and six major points on B; its three other major points, findings 8 to 10, are on A and are taken up below). It is run only if there is room (P38), after its rules are rewritten, reviewed and committed here |
| C | The refit of the stand-in controller with delay, two fits | **not registered.** A design and its review are on file (`stage2/design/addon/design_C.md`, `review_C.md`: twelve major points). Run only if there is room (P38), after its rules are rewritten, reviewed and committed here |

The designer's and the reviewers' texts are working papers. Only this file is a rule.

## A. More delays (P37, answer 1, first part)

**Amended on 16 Nov 2026 before any unit ran.** Section A was first committed at `7282da2`. An independent review
of it and of the chain script followed (`stage2/design/addon/review_A.md`: one blocker and eight major points). No
record of this section existed when the words below were changed, so no rule here was changed after a result. What
changed: the slow walk's own reading (A-6), the `R0` flag and its fixed check (A-6), what would refute the mirror
estimate (A-6), a unit whose episodes fail to run twice and a unit resumed after a kill (A-3), the labels (A-8),
and the wording the reviewer listed.

**The question.** Is 10 ms a special point for the stand-in controller (Stage 1's fitted tracker, rows `R2/42` and
`R2/43`, fitted with no delay), or does delay simply get worse? In the registered records the 0.40 m/s walk stays
up in 20 of 20 at 0 to 6 ms and in 0 to 3 of 20 at 8 and 10 ms under both fits on both floors. In Stage 1 this
body's slow walk fell at 10 ms and stayed up at 20 ms, none of the three episodes meeting every limit there (`R2/43`;
each failed the speed, cadence, stride, ground-force and drive-speed terms; `DECISION_LOG.md` F26).

**A-1. What is run.** Set `V0`, body `75-M-H-10`, block `seeds3042` (seeds 3042 to 3061), 20 episodes per record,
nominal plant, commanded 5 % fast (B1). Conditions: `walk/v0.4`, `walk/v0.4_mu0.54` and `walk/nominal`. Rows:
`R2/42@<ms>ms`, `R2/43@<ms>ms`, `R0@<ms>ms` with the gains of `stage1/runs/2026-11-08/75-M-H-10/tune.json`
unchanged (B2). That is 9 records and 180 episodes per delay. B2 names delays of k = 0 to 5 physics steps (0 to
10 ms); this section widens that range to k ≤ 10 (20 ms), the runner's limit. The delay queue is tested above
k = 5 without physics in `tests/test_stage2_addon_delay.py`.

**A-2. Order and units.** One unit per delay, in this order: `A1-V0-14ms`, `A1-V0-16ms`, `A1-V0-20ms` (the three
delays the owner approved; `STAGE2_PLAN.md` S2-9 names them and the round listed them as not run); then
`A2-V0-12ms` and `A2-V0-18ms`, each as A-1; then `A3-V0-nominal-8ms`: `walk/nominal` under `R2/42@8ms` and
`R2/43@8ms`. **The A2 and A3 units are the lead's addition** to what the owner approved, inside the same budget and
placed last so that a cut removes them first: with them every 2 ms step from 0 to 20 ms has a record of the
0.40 m/s walk, and the slow walk has one at 8 ms, the last delay before it falls. They run before the training path
of P38 and are estimated at about 0.28 core-hours. **In A1 the slow walk and the `R0` rows are the lead's reading**
of "today's walk": the registered walk unit of the panels held the 0.40 m/s walk on both floors only. The run
directory is `stage2/runs/2026-11-16-75cm-addon/A/`. The rates are those of the committed
`stage2/runs/2026-11-15-75cm/COST.md`. Estimates at those rates: 0.128 core-hours per delay, 0.025 for A3, 0.665
in all (about 0.79 at the rates of the registered records, which ran full length more often than the committed
rates assume).

**A-3. Budget and stop rule.** 1.0 core-hour for all of section A, on top of H6's 16 + 12 core-hours, counted as
the sum of episode wall times in `stage2/runs/2026-11-16-75cm-addon/A/budget.json`. A unit starts only if the spent
core-hours plus its estimate fit 1.0. The estimate is the runner's dry-run Total at the committed rates, taken at
the unit's first start. The first unit that does not fit is cut with every unit after it. A started unit finishes:
a unit that `budget.json` already holds is resumed after a kill without a new cap test, so the total may end above
1.0 by the last started unit's overrun. The runner's own ledger names its task "3" and prints a cap of 16
core-hours: that is the stock runner's slot and constant. The cap of this section is held by the chain script
`stage2/runs/2026-11-16-75cm-addon/run_addon.sh`, which writes every unit's estimate, the core-hours spent before
it, and whether it started, finished, was cut or is incomplete, into the committed file
`stage2/runs/2026-11-16-75cm-addon/A/ADDON_LEDGER.json`.

An episode that failed to run is run again once by the same command (C8). If the second run still holds a failed
episode, the unit is marked `incomplete` in `ADDON_LEDGER.json`, is not started a third time, and the chain goes
on. A record that holds a failed episode is not read. A delay with a record that is not read has no word, is
printed "not read", and enters no pattern (A-6). After the first record of this section exists, no delay, seed, row
or condition is added or run again except as a new, labelled addition.

**A-4. Code state.** The units run with the runner's key sources as they stand in the registered records of
`stage2/runs/2026-11-15-75cm/` (code state A): the two hashes every record's key holds, `bar`
`45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d` and `panels`
`c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`. The chain script computes them with the
runner's own functions, refuses to start on other values, or on a checkout with uncommitted changes outside the run
directory `stage2/runs/2026-11-16-75cm-addon/A/`, or when git does not answer, or while another chain holds its
lock, and writes the hashes into `ADDON_LEDGER.json`. No module that enters a record key is changed before these
units have run. The table script (A-10) reads a new record only if its `key.sources` and `key.context` (library
versions, plant inputs, environment constants) equal those of the registered records; otherwise the record is
printed "not read: another code state".

**A-5. What is judged.** Nothing. No verdict comes from this section and none changes. The bar stays judged at
10 ms (P30, P32, P37). The two fits of one condition at one delay give their counts and, by H2, the runner's word
for a delay that is not the judged one: "passes at <ms> ms", "one of two passes at <ms> ms" or "does not pass at
<ms> ms", in each of the two readings (every limit; but for A1). Never `shown` or `provisional`.

**A-6. What is reported, and how it is read** (stated before any result).

**Reading words.** They describe a count. They are not passes and not verdicts.

- A record is **up** when at least 17 of its 20 episodes stay up (`episodes[].fell` false). It is **down** when at
  most 3 do. Otherwise it is **mixed**. The two numbers were chosen after the registered counts were seen: 3 is the
  most that stayed up in any registered `R2` record of this walk at 8 or 10 ms, and 17 borrows H1's number without
  being H1's rule (H1 asks for 20 of 20 up and 17 that meet every limit).
- A record that is up carries a second word. It is **up and covering the distance** when at least 17 of its 20
  episodes stay up *and* meet the bar's distance term (`episodes[].success.along`), for the 0.40 m/s walk and for
  the slow walk alike. Otherwise it is **up, short of the distance**. Staying up without walking as commanded is
  not walking. For the slow walk, the count of Stage 1's `speed` term (`episodes[].measures.stage1.success.speed`)
  is printed beside it and decides nothing.
- **The 0.40 m/s walk.** A delay is **up** when all four `R2` records of the 0.40 m/s walk are up (two fits, two
  floors). It is **down** when all four are down. Otherwise it is **mixed**, and the four counts are printed. By
  these words the registered delays read: 0, 2, 4 and 6 ms up; 8 and 10 ms down.
- **The slow walk** has its own words, from its two `R2` records per delay: a delay is **up** when both are up,
  **down** when both are down, otherwise **mixed** with both counts. Its registered delays read 0, 4 and 6 ms up
  and 10 ms down; 2 ms has no record; 8 ms is unit A3.
- A delay that is flagged (expectation 1 below) or holds a record that is not read (A-3, A-4) takes **no word** in
  either walk's reading and enters no pattern.

**The patterns of the 0.40 m/s walk.**

- **"Delay simply gets worse."** Every delay run above 10 ms reads down. Then no delay that was run from 8 ms up
  holds, and 10 ms is not special.
- **"10 ms is a special point" (a window).** At least one delay run above 10 ms reads up, while 8 and 10 ms read
  down. The report then names the lowest delay above 10 ms that reads up, and says of each such delay whether its
  four records are up and covering the distance. Where they are not, the sentence reads "a window of staying up,
  not of walking".
- **"Neither."** No delay above 10 ms reads up and at least one is mixed. The report prints the counts and uses
  neither phrase.

**The patterns of the slow walk**, in its own paragraph. The two paragraphs are not merged.

- **"Delay simply gets worse."** Every delay run above 10 ms reads down.
- **"A window."** At least one delay run above 10 ms reads up. The 8 ms word is printed and is no condition. The
  second word is given as for the 0.40 m/s walk.
- **"Neither."** No delay above 10 ms reads up and at least one is mixed.

If no delay above 10 ms has a word, no pattern is named for that walk.

**What a window would and would not mean.**

- It would not make any delay a verdict. The bar is judged at 10 ms (P30, P32, P37).
- It would not mean the build is fine at that delay. None of Stage 1's three 20 ms episodes met every limit (above).
- It would describe the survival curve. It would not explain it. No mechanism is tested by these runs.
- It would bear on item C: a refit with delay in its mix would then be moving a narrow failure, not buying margin
  against a steady trend. That is the lead's reading to make, as an interpretation.

**Where before why.** Every count is printed with where the falls come (the start from standing, the slow stride,
the change of speed, the 0.40 m/s walk itself), classified by the episodes' `segment_reached` alone. Fall times are
printed and classify nothing. At 10 ms 76 of 80 episodes fell and none fell in the 0.40 m/s walk itself
(`STAGE2_75_REPORT.md` section 5). So a 0.40 m/s record at a longer delay may say nothing about walking at
0.40 m/s.

**Of the episodes that stay up** the report prints, per record and not judged: the median speed, the median toe
clearance, the failed terms with their counts, the largest drive speed ratio, and for the slow walk the median peak
ground force.

**Expectations, stated from results already known.**

1. `R0` rows of the 0.40 m/s walk: 18 to 20 of 20 stay up at every delay, as in the registered records at 0 to
   10 ms. `R0` has no feedback, so a delay only shifts its commands in time (design estimate). **An `R0` record of
   the 0.40 m/s walk under 17 is a flag** on the runner at long delays, not a finding about the controller. That
   delay then takes no word and enters no pattern; its cells are printed with the flag; nothing is run again. The
   check that goes with a flag is fixed here: the no-physics test of `speedfit.servo_delay` at that k
   (`tests/test_stage2_addon_delay.py`), and `controller.latency_steps` = ms / 2 in that delay's records. `R0` on
   the slow walk is the lead's addition: its only source is Stage 1's `eval/R0/walk/nominal.json` (20 of 20 at
   0 ms, under another plan). It is printed and raises no flag.
2. The slow walk under `R2/43` at 20 ms: the record reads up, from Stage 1's 3 of 3. The plans differ: Stage 1's
   walk is start, walk, stop over 18.4 s; the bar's is start then 8 strides over 14.06 s (the runner's dry run). A
   record that reads down would contradict Stage 1's three episodes and is reported as a contradiction, with both
   records. A record that reads mixed does not meet this expectation and does not contradict three episodes; its
   counts are printed.
3. `R2/42` at 20 ms, every row at 12, 14, 16 and 18 ms, and the 0.40 m/s walk at 20 ms: no expectation can be
   stated from results. No record exists. The design estimate below makes five predictions.
4. No record at 12 to 20 ms is expected to meet every limit in 17 of 20. At 6 ms the four registered `R2` records
   of the 0.40 m/s walk fail toe clearance in 19 or 20 of 20 episodes, at 8 and 10 ms 0 to 3 of 20 stay up, and
   Stage 1's 20 ms episodes failed speed, cadence, stride, ground force and drive speed.

The registered records at 0 to 10 ms under `stage2/runs/2026-11-15-75cm/` are printed beside the new ones, each
with its path. Nothing is read between delays (B3).

**A design estimate, labelled as such, for why a window is possible at all.** A pure delay turns the phase of a
loop component at frequency f by 360° × f × delay, so the same phase returns when the delay grows by 1/f. The
tracker's feedback is computed at 100 Hz and ramped in over 10 ms (`designlab/stage1/control.py:20-22`). A component
that alternates from one update to the next has a 20 ms period: 10 ms of delay flips its sign and 20 ms restores it.
If that were what fells the walk, the curve of the 0.40 m/s walk would mirror about 10 ms: 12 like 8, 14 like 6,
16 like 4, 18 like 2, 20 like 0. This is arithmetic, not a result. It is why 12 and 18 ms are worth their cost.

**What refutes it.** The report prints these five predictions beside the result whatever it is: "met" when 12 ms
reads down and 14, 16, 18 and 20 ms read up, on the delays that were run; otherwise "not met", naming the delays
that differ. The baseline loop already lags (sensors one step old, the ramp), so the true centre could sit
elsewhere; that is not used to rescue the estimate. No other centre is fitted after the result.

**A-7. Not attributed.** These records take no fall attribution (`--attribute none`). In the registered round the
attribution of records where 0 of 20 stayed up rested on a change of one physics step (`STAGE2_75_REPORT.md`
section 5).

**A-8. Labels.** Every table and sentence from these records carries: "simulation; stand-in controller fitted
with no delay; a delay that is not the judged 10 ms; no verdict; added after the 75 cm results". Records of A1 also
carry "(P37: 14, 16 and 20 ms)". Records of A2 and A3 also carry "the lead's addition, not among the delays the
owner approved".

**A-9. Not run.** Turns, pushes and stops at these delays; the speed ladder's other rungs; `walk/nominal_mu0.54`;
the set `F25F9` (the pull request's walk; nothing asks for it) and the set `F25F9cl` (P37, answer 3); the block
`seeds5042`; delays above 20 ms (the runner's limit); any delay that is not a whole 2 ms step.

**A-10. Where the result goes.** A short section added to `STAGE2_75_REPORT.md`, labelled as A-8 says, with its
tables made by script from the records: `stage2/runs/2026-11-16-75cm-addon/delays_table.py`, in no key, committed
before the first unit, which writes `DELAYS_TABLE.md` and `delays_table.json` beside itself. The section is checked
by one independent reviewer against the records before the owner sees it.
