# Stage 1b: the speed round (simulation)

**Owner request (P27, 10 Nov 2026, the evening after the Stage 1 report):** "continue with the next round of testing
and experimentation this evening." This round follows the Stage 1 report's speed findings (`stage1/runs/2026-11-08/
STAGE1_PROBE.md`) and the discussion after it. It answers how fast the 60 cm bodies can walk once the controller is
fitted for speed, with today's drives and with the faster knee. It is simulation only, like Stage 1. It changes no
body, and adopts nothing without the owner (P24).

## 1. Questions

1. **Controller.** Stage 1's tracker (R2) was fitted only on the nominal walk and the turns (decision 22). With the
   drives as picked, can the same tracker, re-fitted with the brisk walk and the animal-like walk in its mix, walk
   them on the 60 cm bodies?
   - The brisk walk is 0.315 / 0.314 m/s.
   - The animal-like walk is Froude 0.05: 0.446 / 0.444 m/s.
2. **Faster knee.** With round two's faster knee (sizing with 15 % knee headroom selects a RobStride 06 knee;
   `ROUND2_REVIEW.md` §7.3) applied to the narrow carpet-ready 60 cm picks, does the same speed fit reach them? In
   round two the faster knee took the knee from 0.95 to 0.56 of its speed limit at Froude 0.05, leaving the hip pitch
   (0.90) to bind next.
3. **Servo delay.** Stage 1 assumed no servo delay, yet the 60 cm nominal walk misses a limit at 6 ms and falls at
   8–10 ms. Do controllers fitted with delay in their episodes keep walking at 10 ms?

## 2. Arms

| Arm | Bodies | Controller |
|---|---|---|
| S | `60-N-T-10`, `60-N-H-10` (as picked) | R2 re-fitted for speed (R2s) |
| K | `60-N-T-10-K`, `60-N-H-10-K` (the picks with the faster knee) | R2s, fitted on the K bodies |

The 75 cm body is not in this round. Its brisk walk already stays up and misses only the tail servo's planning limit
(Stage 1 report).

## 3. Method

- **The K bodies.** Round two's sizing pipeline (`designlab/refine.py`) builds the narrow carpet-ready case with 15 %
  knee headroom (`narrow_knee_60`). Stage 1's `prepare` (plant, references, transitions) and gates G1–G4 then run on
  each K body, as Stage 1 ran them on the picks.
  - Their models' `xml_sha256` begin `85d98c24` (`60-N-T-10-K`) and `e0d4c6b7` (`60-N-H-10-K`), from the case's
    records of 10 Nov. G1 checks each model against these prefixes (`SHA256_PREFIX`), as it checks the picks'.
  - Registering them changes the panels' and the tuner's source hashes. With the round's code, Stage 1's panel and
    ledger records therefore read as made by other code, and none of them is run again (section 6).
  - The build's notes are in `stage1/runs/2026-11-10-speed/ARM_K.md`. They cover the git-ignored inputs a checkout
    needs, the open question on `60-N-H-10-K` (its body fails round two's clearance gate), and how the K bodies differ
    from their picks.
- **The speed fit (R2s).** This is Stage 1's CMA-ES fit of the tracker's gains (`stage1-tune-v2`, decision 22) with a
  wider episode mix. It is otherwise unchanged; capture margin held at 0.0.
  - Each evaluation runs the nominal walk, the brisk walk, the Froude 0.05 walk, the 0.35 rad/s walking turn and the
    0.5 rad/s in-place turn.
  - The brisk and Froude 0.05 walks are built and judged as the panels judge them (decision 31): they end in the gait
    and are judged over the last 10 s.
  - Each episode draws a servo delay between 0 and 10 ms.
  - The fit starts from each body's G5 selection (a K body starts from its pick's) and runs 360 evaluations (30
    generations of 12), seed 42.
  - Selection follows tune's G5 rule on 20 validation episodes per task: fewest falls, then the lower survival
    penalty, then the lower performance penalty.
- **Validation panels.** Each selected controller runs the panels' conditions (decision 31) on episodes 3042–3061:
  nominal walk, brisk walk, Froude 0.05 walk, both 0.35 rad/s walking turns and both 0.5 rad/s in-place turns. They run
  at 0 ms and at 10 ms of servo delay, all on hard floor at μ 0.9. Stage 1's R2 rows on the picks are the baseline.

## 4. Pass criteria (set before the runs)

- A condition is **shown** on a body if 20/20 episodes stay up and at least 17/20 meet every limit (Stage 1 §4).
- Q1 and Q2 are answered per arm and body by the brisk and Froude 0.05 walks at 0 ms: shown, survives only (20/20
  up, fewer than 17 meet every limit), or fails. The failing limits are named.
- Q3 is answered by the same conditions at 10 ms.
- A speed fit that loses the nominal walk or a turn at 0 ms is reported as a trade, not a success.

## 5. Budget

Compute runs from about 04:30 to 11:00 UTC on 10 Nov, on the session's 4 cores, with the report by about 12:30 UTC.
If the fits run long, every arm is judged at the last generation all of them reached. The round is recorded in
`stage1/runs/2026-11-10-speed/`.

## 6. Decisions taken

These were settled on 10 Nov, after the build's review and before any speed fit or row ran. They change no body
(P24) and no pass criterion of section 4. Each is recorded in the code as a rule that the records carry.

1. **The fit's step size is Stage 1's** (`speedfit.SIGMA0_RULE`). sigma0 is 0.1 of each gain's range, as section 3
   has it. Half of that (0.05) was built first and then dropped. Stage 1's six fits never raised their step size above
   1.2 times its start (0.119 at most), and their selections were found at 0.092, 0.054 and 0.052. A fit started at
   0.05 would therefore stay a local search about the start, and in 360 evaluations a negative answer to Q1 or Q2
   would be weaker evidence than "Stage 1's fit" implies.
2. **The speed rows' falls are attributed** (`speedfit.ATTRIBUTION_RULE`). Section 4 names the failing limits, and
   Stage 1's R2 baseline carries decision 20's labels. So the rows R2s, R2@10ms and R2@0ms are attributed the same
   way: `eval --attribute` lifts each relaxable candidate of a condition's falls over its 20 starts when the condition
   has at least 5 falls. The lifted runs keep the row's servo delay. A failing condition costs about 40–60 more
   episodes, about 0.1–0.4 core-h per body.
3. **How the rows are read** (`designlab/stage1/speedreport.py`). The panels' `eval/summary.json` gives Stage 1's
   verdicts (R2/42 with R2/43), which do not apply to the speed rows. `python -m designlab.stage1 speedreport` applies
   section 4 per row and condition: shown, survives only or fails, with the failing terms and their counts, and the
   falls' candidates and attribution.
   - Q1 (arm S) is read beside Stage 1's R2 row of the start's CMA-ES seed on the same body.
   - Q2 (arm K) is read beside R2@0ms on the K body: the pick's gains without a refit. R2@0ms is run on arm K for this,
     so that a trade can be told from the body's effect.
   - Q3 is read beside R2@10ms.
   - A trade is a nominal walk or target turn that the 0 ms baseline shows and R2s does not.
   - Speed G5's counts pool delays of 0–10 ms and answer no question.
4. **Judging the fits** (`speedfit.JUDGED_RULE`). A fit is judged at its last generation once it is done, or at
   `speedg5 --at-generation G` under section 5. `--partial` on an unfinished fit is a smoke check: its selection is
   recorded as not judged, and no row takes it. Speed G5 keeps each validation evaluation as it finishes, so a
   restart continues it (`speedfit.G5_RESUME_RULE`).
5. **The code is frozen from the merge until the rows have run** (`speedfit.SOURCES_RULE`). The speed fit's sources
   are 14 modules, and the faster-knee branch edits six of them (`__init__`, `control`, `gates`, `panels`, `plant`,
   `refs`). So this branch and the faster-knee branch are merged before any fit starts, arm S's included, and none of
   the 14 changes until the rows have run: a resumed fit and speed G5 refuse other sources. `speedreport.py` and the
   run script are outside them. env.py needs no edit at the merge: it reads an unlisted body's push bound from its
   screen by candidate id once `candidate_id` is in the registry.
6. **Stage 1's records after the merge.** The round edits env.py (the push bound of an unlisted body). The
   faster-knee branch edits the panels' sources (`__init__`, `panels`, `control`, `plant`, `refs`). So after the
   merge, Stage 1's panel and ledger records (panel sources 7516b927…) and its R3 and R4 rows (env.py beff6779…) name
   other sources than the code.
   - Stage 1's records and tables are kept as they are and read as Stage 1's. `STAGE1_TABLES.md` is regenerated only
     from Stage 1's commit (07a73f4 or earlier), and no Stage 1 record is rerun with the round's code.
   - The round cites Stage 1's R2 rows as its baseline, with this note.
   - The edits change no Stage 1 episode. The three listed push bounds are unchanged, and the round's code reproduces
     Stage 1's episodes: the review reran three brisk and Fr 0.05 episodes exactly, and `test_stage1_speedfit` reruns
     a recorded G5 episode. That test is run again on the merged code before any fit starts.
7. **Order of the runs** (`stage1/runs/2026-11-10-speed/run_speed_round.sh`). First the merge, then arm K's `prepare`
   and G1–G4. Then the four fits run at once, one worker each, with nothing else heavy on the machine. Speed G5 and
   the rows with `--attribute` follow, and `speedreport` last. If the fits cannot reach generation 30 in time for
   speed G5 and the rows before 11:00, every arm is judged with `speedg5 --at-generation` at the last generation all
   of them reached (section 5).
8. **Arm K is `60-N-T-10-K` alone** (decided at the merge, 07:50 UTC, from `ARM_K.md` section 4). `60-N-H-10-K` is
   not fitted, for two reasons:
   - **It fails round two's clearance gate.** Its hip-pitch extension stops at −0.509 rad against the
     power-distribution board, and its turn gaps are 0.84 and 0.28 mm against the case's 3 mm.
   - **It cannot pass the speed walks.** Its brisk and Froude 0.05 references keep only 0.07 and 0.04 rad of
     hip-pitch range, against the 0.10 rad every walk success needs (decision 14). Q2 could at best read "survives
     only" there, by construction.

   Q2 is therefore answered on the Tapered body. The freed core gives each of the three fits a core of its own. The K
   body also differs from its pick by more than the knee's speed (`ARM_K.md` section 7): +0.75 kg, hips +4 mm per
   side, and knee flexion 0.856 against 1.051 rad. Q2 compares the whole faster-knee body.
9. **The code at launch.** It is merge 3b0e948 of the round's two branches, and the speed sources are `68f5a80b`
   over 14 modules (decision 5). The K registration alone moves the panels' sources (7516b927 → 733f2b80) and the
   tuner's (e7b37918 → 6a33d3f9). Stage 1's fits name c8b112cf and were already refused by Stage 1's own final code
   (`ARM_K.md` section 3), so decision 6's rule stands: no eval, sweep, tune or g5 against 2026-11-08 with the round's
   code.
