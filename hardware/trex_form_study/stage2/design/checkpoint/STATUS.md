# Checkpoint of 15 Nov 2026: the second session paused for the usage limit

Written by the session's lead on 15 Nov 2026 at about 04:05 UTC and updated the same night (see the update below). The owner asked that the session stop before a tranche
of work that would take the account over its weekly allocation, with everything checked in and documented, and wait
for a go-ahead. This file is that record. It is a working paper: it sets no rule and reports no result.

## Update, 15 Nov, after the owner's answer

The owner answered: the workflows already running may continue; no new one is to be added that night. So the three
workflows that were running (round A's build, the look sets, round C's first wave) were left to run to their end,
queued agents included, and the lead launched nothing else. The workflow that was writing the Track B rules had
already been stopped after its first agent and stays stopped. Sections 3 to 5 below describe the state at 04:05 UTC;
`generated.json` and the bundle are refreshed when those workflows end, and a later commit of this file says what
they left.

The lead ruled on the reviewers' open points the same night, before each fixer started, and wrote the rulings into the
fixers' briefs: `../REVIEW_RULINGS_15NOV.md`. Of section 6's list, points 1 to 5 and 8 are ruled and written into
`STAGE2_75_PLAN.md` (its last section); points 6, 7 and 9 are still open.

## Update, 15 Nov, 04:25 UTC: the training-side workflow is paused until the owner's reset

The owner then said: another reset is available, to be triggered when the meter nears 99 %, but not for about eight
hours; no leaning down is wanted after that. A second meter reading (52 % at 04:14 UTC, after 46 % at 03:58)
calibrates the session's count: 1 % of the week is about 72,000 output tokens or 8.9 million cached tokens read, and
with five agents running the meter rose about 22 % an hour. `usage_watch.py` (in the session's scratch directory, its
calibration in its docstring) estimates the meter from the transcripts.

For those eight hours the lead holds a ceiling of 90 % by that estimate and puts the 75 cm path first:

- Round A's workflow (fix M, B, G, C; then S and T) keeps running. About 19 % of a week is left in it.
- Round C's first wave was stopped at 04:23 UTC with the estimate at 55 %. CP is merged; CE is built and reviewed,
  its fixer not started (the rulings are in its brief); the builders of CV and CK were stopped mid-build with their
  commits kept on `wt/wpc-cv` and `wt/wpc-ck`; CB is not started. It resumes after the reset
  (`Workflow({scriptPath, resumeFromRunId: "wf_bb1d63d5-059"})` while the session lives; from the bundle otherwise).
- The look sets' workflow is left to finish its build, then stopped before its two reviews, which wait for the reset.

Section 6's point 10, added the same night and ruled: the quick turn's hip roll came within 0.086 rad of its limit at
stride factor 0.8, under A12's 0.10 rad; A12 stands and the stride rule (A11) now reads it too, so V0's turn is
expected at factor 0.70 (`../hiproll_scan.json`).

## Owner's directions of 15 Nov, afternoon (decision log P34 to P36)

- P34: Stage 1's limits for the reference checks of moves; F25F9 starts straight into its 0.40 m/s walk. Written into
  `STAGE2_75_PLAN.md` (A7, A8, A11, A12), each labelled. The set builder was restarted with them in a follow-on
  workflow (`trex-round-a-finish`, run `wf_9bf696c4-f69`: build S, two reviews, fix S; fix T from its recorded
  review). Resuming the first workflow by run id was tried and stopped after two minutes: it ran finished reviews
  again, because every call after the first live one runs live.
- P35: the configuration stays as it is (no hip-range variant). The owner's page of the 75 cm results shows the
  turning circle at each turn rate, planned and measured apart, and video clips of the behaviours (the 90 degree turn
  among them).
- P36: after the main panels, a carpet side-check on a thin-carpet model set from published figures for commercial
  loop-pile carpet tile, checked independently, its rules committed before its first episode; reported rows only.
- Usage: the owner read 11 % at about 16:00 UTC, when the cost figure gave 12 % (76 USD since the reset; 1 % is about
  6.9 USD). The owner has one more reset this week.

## Update, 15 Nov, 14:15 UTC: the limit was hit at 07:36 UTC; the owner reset it; work resumed at 13:56

- **The limit.** The account's Fable limit was reached at 07:36 UTC, when the lead's highest estimate of the meter
  read 87 %: the estimate from transcripts was 13 points low, and the 90 % ceiling was not held. The owner confirmed
  100 % and reset the meter before 13:56 UTC.
- **The meter to use.** `get_session` reports the session's cost at API prices (`external_metadata.usage.cost_usd`):
  1,358 USD at 13:57 UTC. Against the owner's readings (46 % at 03:58, the limit at 07:36) one week of allocation is
  about 610 USD, so 1 % is about 6.1 USD and the session had used about 2.2 weeks. `usage_watch.py`'s token
  estimates are not to be trusted beyond a few minutes.
- **What was cut off.** Round A's set builder (S: V0 on 75-M-H-10 built, screened and verified, uncommitted; F25F9
  and F25F9cl not built), its table fixer (T: a first commit of fixes, the rest uncommitted, one test red) and the
  look sets' builder (L: V6 on both bodies and V6noF5 on 75-M-H-10 committed; V6noF5 on 60-N-T-10 and a head-body
  probe uncommitted). Neither S nor L has been reviewed. The workflows ended; they can be resumed by run id
  (`wf_4b831ec4-af2`, `wf_83a16354-c85`), which runs the cut-off agents again from the start.
- **What the lead does by hand from here** (CPU work without agents; agents only where independence is required):
  the builds of F25F9 and F25F9cl on 75-M-H-10 with the set builder's own commands, started at 14:05 UTC.
- **A result that goes to the owner (rule A11).** V0's turns from 0.40 m/s on 75-M-H-10, as built under the rules as
  amended: 0.35 rad/s has a reference (stride factor 1.0, every gate passed); 0.5 rad/s has none (factor 0.9;
  `join_command`: warm-up ZMP 2.76 mm against 2.0 mm); the quick turn at pi/4 rad/s has none (factor 0.70, hip-roll
  margin 0.106 rad, thigh twist 0.71; `join_command`: warm-up ZMP 2.26 mm against 2.0 mm; first-stride consistency
  2.63 mm against 2.5 mm after the sixth pass, still falling by a fifth per pass); 1.0 rad/s has none (no factor
  keeps 0.10 rad of hip roll). The warm-up ZMP reads 1.35 mm at factor 1.0, so it moves with the grid and not with
  the motion (the warm-up CoM is within 0.04 mm in every case).

## Update, 15 Nov, 06:30 UTC: round A's four fixed packages are merged

M, G, C and B came back from their fixers and are merged into the working branch (`4ef2674`, `2d9adef`, `2b12c13`,
`dd6b372`); their reports are in `reviews/`. The set builder (S) and the table builder (T) are building; the look
sets' build is still running. The usage estimate is 69 % (the ceiling until the owner's reset is 90 %).

- Known red until code state A: `test_legacy_check_record_is_current` (the set builder's edit to `moves.py` comes
  first) and `test_regress_record_is_current` (the record was made before the merges). Both records are made again at
  state A.
- The lead checked C3 as amended against an implementation of his own: 1,500 of 1,500 random traces equal
  (`../lead_c3_check.py`).
- Ruled for the set builder's fix round: A12's acceleration gate on a move reads the entry's value under A7 as
  amended, and the screen and the readings module give the same number (`../WP_S.md`, addendum 3).
- Sections 3 and 6 below are superseded where they name M, G, C or B as unmerged or points 1 to 5 and 8 as open.

## 1. Why the session stopped launching work

- The owner's usage meter read 46 % at 04:00 UTC on 15 Nov, after one reset from 99 % to 0 % during this session; the
  weekly allocation resets in 4 to 7 days.
- This session's own count at that time, from its transcripts: 80 agent and lead transcripts, 9.5 million output
  tokens, 1.27 billion cached-context tokens read, 34 million written.
- So the session had used at least 46 % of a week since the reset, and up to 145 % in all if the meter stood near zero
  when it began. On the larger figure one agent of this session costs about 1.8 % of a week (the lead had assumed
  0.4 %, from the first session's 271 agents). The agents ran at the session's own effort setting (`xhigh`), each with
  a design document of 150 to 190 kB in its context.
- The work already launched had about 21 agents still queued (about 38 % of a week at that rate), which with 46 % used
  comes to about the limit before the 75 cm panels are reported. That is the owner's stop condition.

What the lead did: let each agent that was in the middle of a task finish that task, stopped every workflow before it
started a queued agent, merged the two packages that were built, reviewed and fixed, and wrote everything else here.

## 2. What is on the branch (built, reviewed independently, fixed, merged)

| Package | What | Merge commit |
|---|---|---|
| Decision log, plans | P30 to P33, F26, F27, A17, A18; `STAGE2_PLAN.md` section 9 (S2-9 to S2-18); `STAGE2_75_PLAN.md` | `167dce4` to `7558b3b`, `89111c3` |
| Jetson documents | latency and bus plan, bench sheet, price record, time-of-flight governor (`stage2/jetson75/`) | `7e9ea0d`, `feb84ad` |
| Jetson body | 75-M-H-10-J (`stage2/j/`, `round2/refine/jetson_75/`) | `1aefbd0` |
| L1 freeze | `designlab/stage2/l1freeze.py`, `stage2/look/L1_FREEZE.json` | `90b0097` |
| Round A, R | the bar's readings, record helpers, set reader (`readings.py`, `s2io.py`, `setio.py`) | `dc5bb10` |
| Round C, CP | policy files, cross-version loader, picklable schedules (`s2load.py`, `schedules.py`, `tbenv.py`) | `c86e5e5` |

After the two merges of 15 Nov the six test files of R and CP gave exit 0 here (124 passed, 1 skipped). The full suite
was last run before them (679 passed, 107 skipped, on 14 Nov) and is owed again at the next merge.

## 3. What is in the bundle (built, not merged)

`worktrees.bundle` in this directory holds every branch below, commit for commit. None of it is on the working branch,
because none of it has finished review and fixing. Section 6 restores it.

| Package | Branch | Built | Reviewed | Fixed | What is owed |
|---|---|---|---|---|---|
| Round A, M (moves: starts, changes, stops, turns for any gait) | `wt/wp-m` | yes; 36 of 36 Stage 1 transitions regenerate bit for bit | two reviews done | no | ruling 1 below, then the fixer |
| Round A, B (the bar's runner) | `wt/wp-b` | yes | two reviews done | no | ruling 2 below, then the fixer (1 blocker, 7 major) |
| Round A, G (closed-loop regression record) | `wt/wp-g` | yes | see section 4 | no | the fixer |
| Round A, C (set screen) | `wt/wp-c` | yes | see section 4 | no | the fixer |
| Look sets, L (V6 and V6noF5 sets) | `wt/wp-l` | see section 4 | no | no | two reviews, the fixer |
| Round C, CE (R5's and R6's env) | `wt/wpc-ce` | yes | see section 4 | no | the fixer |
| Round C, CV (vec pool) | `wt/wpc-cv` | see section 4 | no | no | finish the build, review, fix |
| Track B rules | `wt/prereg-b` | draft `STAGE2_TRACKB_PLAN.md` with the lead's 25 rulings written in | no | no | ten conflicts between rulings (the integrator's report), then the independent check |

Every agent's final report, and every reviewer's findings in full, are in `reviews/`; `generated.json` lists them with
each branch's tip.

## 4. The agents that were finishing when this was written

- Review of G (regression record): done. No blocker or major finding; the Stage 1 edit leaves the default path
  bit-identical, 256 of 256 comparisons equal (248 distinct committed episodes). One point for the lead, ruled
  (the stride period of a K-stride window is made exact for every K).
- Review of CE (training env): done. One blocker and three major findings, all for the lead and all ruled: the
  reward's four readings stand as coded and the Track B rules text is amended to them; the commanded heading in a
  turn in place follows the reference's yaw; a later freeze of the same set is accepted (rule K4); scripted head
  targets keep 0.15 rad from a joint limit, so the jaw opens to at most 0.45 rad of its 0.6 rad in a script.
- Review of C (set screen), build of L (look sets), build of CV (vec pool): still running at the update.

## 5. Not started

- Round A: S (build and freeze the sets V0, F25F9, F25F9cl; restore group `j`) and T (tables, `COST.md`). Both wait for
  M, C and B.
- Round C: CK (look reading), CB (Drive bundle), then CT (trainer), CW (rows), CF (R2L fit), CG (gates G8a and G9),
  CN (notebook).
- Every run: no panel, no fit and no training run has been made on any Stage 2 set. No set is frozen. Code state A
  does not exist yet.

## 6. Open rulings for the lead, before the next builder starts

Each is a point a builder, reviewer or fixer handed up. None is settled.

1. **The V0 quick turn fails its consistency gate in both directions** on 75-M-H-10 (review of M, mechanics, finding
   1, blocker; bit-identity, finding 1). The reviewer traces it to the design's two causal ZMP limits contradicting
   each other as coded, not to the builder. The committed slow test `test_stop_and_turn_from_v0` is red. Needs a
   ruling before any set with a quick turn is built; the constant must not be moved to make it pass.
2. **Attribution units run out of the pre-registered order** (review of B, judge, finding 1, blocker;
   `STAGE2_75_PLAN.md`, confirmation 5 under H6). Either the chain is reordered or the rule is amended with a label,
   before task 3's first unit.
3. **The own-swing toe clearance (C3)** reads 0.010 to 0.017 H below Stage 1's reading on stock walks, with no servo
   delay, because a foot's own swing ends where its load crosses 2 % of the weight (review of R, finding 1; fix of R;
   review of B, judge, finding 15). Keep the rule with an expectation sentence, or redefine it with a label. The
   fixer measured one alternative (same swings, middle half of the planned duration counted from the own lift-off).
4. **A turn's window can cut the inside foot's first swing**, and then `count_ok` fails on a clean walk (fix of R,
   "new, mine"). Suggested: do not count a reference swing whose own swing the window cuts.
5. **With ramps fixed at 0.375 s the 1.0 rad/s ladder turn cannot pass the slowest-clock heading gate** on any set
   (review of M, mechanics, finding 3). Decide before the ladder is built.
6. **Whether a judging load reads the `.pkl`** (review of CP, finding 1; the fixer built rule K9's reading: it does
   not). Confirm, or revert one line and change K9's text.
7. **`actions_at_bound`** as a record key of the fingerprint (review of CP, finding 8). Accept or drop before
   `s2load.py` is frozen.
8. **Six major provenance points of the bar's runner** to fix before state A (review of B, provenance, findings 1 to
   6): the supersede label lost after a kill, a "not run" record replaced without a label, exit 0 when episodes failed
   to run, the dirty flag counting the run's own outputs, the crouch read from an unfrozen key, and R0 traces dropped
   under 1.5 GB free.
9. **Ten conflicts in the Track B rules draft** (`reviews/trackb_rules__integrate_trackb-rulings.md`).

Texts owed by the lead after these: `STAGE2_75_PLAN.md` C3 and C7 wording (five readings now stated only in
docstrings), design 4.4 and 2.1.

## 7. Restoring the bundle in a fresh clone

From the repository root, on the working branch:

```sh
B=hardware/trex_form_study/stage2/design/checkpoint/worktrees.bundle
sha256sum $B                                   # compare with bundle_sha256 in generated.json
git bundle verify $B
git fetch $B 'refs/heads/wt/*:refs/heads/wt/*'
mkdir -p ../mesozoic-wt
for p in wp-m wp-b wp-g wp-c wp-l wpc-ce wpc-cv prereg-b; do
  git rev-parse -q --verify wt/$p >/dev/null && git worktree add ../mesozoic-wt/$p wt/$p
done
```

A branch missing from `generated.json` had no commit of its own. The worktree branches may not be pushed (only the
session's designated branch may), which is why they travel as a bundle. Delete the bundle from the tree once every
branch in it is merged or dropped.

## 8. Order of work on resuming

The owner's stated focus is a 75 cm build that works in simulation (P33), so the 75 cm path comes first:

1. Rulings 1 to 5 and 8, written into the committed rules with labels where a rule changes.
2. Fix M, G, C and B; merge in the order M, G, C, B; full suite.
3. Build, review and fix S; freeze the three 75 cm sets; legacy check and regression record at state A.
4. Task 3's tiers, then task 4's (28 core-hours of CPU here; no agents needed while they run), with T's tables.
5. Report, independent review, owner page; push notification to the owner (asked for on 15 Nov).

The training side (round C, the Track B rules, the notebook, the Drive bundle) follows, and is the larger share of what
is left.

Ways to spend less of the allocation per package, for the owner to choose between (none taken yet): run builders and
fixers at a lower effort setting than the session's; one reviewer per package instead of two where the second lens is
not provenance or bit-identity; briefs that carry only the package's part of the design; builders and fixers on
another model with the lead and the independent reviewers unchanged.

## Update, 15 Nov 2026, 19:50 UTC: the sets are frozen and the 75 cm panels are running

- Round A is merged in full (M, R, G, C, B, T, S). The sets' fix round was reviewed independently by two reviewers
  before the freeze; their one blocker (the chain's reference-check step) is fixed in `run_75cm.sh`.
- Code state A is `c377ba1` (`STAGE2_75_PLAN.md` R4). Before it: legacy check 36 of 36, regression 256 of 256, full
  suite green but for two tables tests (a fixture older than the runner's lead-in refusal; being fixed in `wt/wp-t2`).
- The chain started at 19:44 UTC in the main checkout with `PROCS=3`: `stage2/runs/2026-11-15-75cm/run_75cm.sh all`,
  watched by `tools/keepalive.sh` with the watchdog in the session's scratch directory (`chain/watchdog_75cm.sh`:
  it starts the chain when it is neither done, stopped, failed nor running). After a container restart: restore the
  inputs, copy the sets' arrays back (`tools/restore_inputs.py`, group `sets`), and start the chain again; it resumes
  per episode.
- First judged tier (the walk at 0.40 m/s, 10 ms of servo delay, 20 seeds, the runner's own lines): under the two
  fitted controllers 0 to 3 of 20 episodes stay up on every set and floor; under the open-loop row 20 of 20 stay up
  and none meets every limit. The plan stated this as an expectation (the controllers were fitted with no delay).
- Running beside the chain (workflow `trex-75cm-report-tooling`): the tables' fix on real records (`wt/wp-t2`) and
  work package V, the turning path and video clips (`wt/wp-v`, brief `stage2/design/WP_V.md`).
- Unmerged and paused, in `worktrees.bundle` as of the 14:15 block: `wp-l`, `wpc-ce`, `wpc-cv`, `wpc-ck`, `wpc-cb`,
  `prereg-b`. Merged worktrees were removed.
- Usage: the session's cost stood at 1,493.5 USD at 17:55 UTC (about 20 % of the week since the owner's reset at a
  baseline of 1,357.5 USD; 1 % is about 6.9 USD).

## Update, 15 Nov 2026, 21:50 UTC: task 3 is done, task 4 is under way, the tables' fix is merged

- **Chain** (`stage2/runs/2026-11-15-75cm/run_75cm.sh all`, main checkout, PROCS=3): task 3 finished with 4.25 of 16
  core-hours and nothing cut. Task 4 is at tier (3) (`T4.3-V0-0ms`); tiers (4) to (7) follow, about 80 minutes by its
  own pace. To restart after a container restart: the keep-alive's watchdog does it
  (`<scratchpad>/chain/watchdog_75cm.sh`; it is safe to run by hand, it starts nothing while the chain runs or is done).
- **Merged since 19:50:** the tables' fix round (`wt/wp-t2`, merge `c9b6800`; 65 tests pass on the merged tree). The
  chain's last step now makes the tables with the fixed module.
- **Running:** the views' builder (`wt/wp-v`, workflow `wf_3ade5c27-6cd`; build, then an independent review, then a
  fixer). Not merged.
- **New in the run directory** (both in no record's key, standard library only): `rereads/term_aside.py` (the records
  with the joint-range term set aside: a labelled re-reading after a result, no verdict; plan, closing amendments,
  item 11) and `digest/digest.py` (a short digest for the written report). Run both again when the chain has ended.
- **Read so far (simulation, from the records):** at 10 ms nothing passes; under the two fitted controllers every walk
  falls at 8 and 10 ms on all three sets. V0 at 0.40 m/s passes but for A1 at 0 and 2 ms; V0 passes every limit at
  0.24 and 0.30 m/s at 0 ms. The candidate passes nothing as registered: its start fails the walk item's joint range
  by itself (item 11), and with any delay from 2 ms its toe clearance is under the limit in 20 of 20.
- **Next:** when the chain ends, run the digest and the re-reading again, commit, write `STAGE2_75_REPORT.md`, have it
  reviewed independently (records and code to the reviewer, not conclusions), merge the reviewed views, publish the
  owner page with the turning circle and the clips, send the push notification.
- **Usage:** the session's cost read 1,647 USD at 21:40 UTC, about 42 % of the week by the owner's 16:00 calibration
  (11 % at 1,433.50 USD).

## Update, 16 Nov 2026, 00:55 UTC: the 75 cm panels are reported; five questions wait for the owner

- **Chain:** ended 15 Nov 23:09 UTC. 592 records and 2 control records, all committed; 4.25 of 16 and 5.51 of 12
  core-hours used; nothing cut. Nothing is running.
- **Report:** `STAGE2_75_REPORT.md`, fourth version, after three rounds of independent checks (50, 24 and 12
  findings on the text; 10 on the page; no table cell was ever found wrong). Its sources and the page builder are in
  `stage2/runs/2026-11-15-75cm/report_src/` (README there); the report and the page regenerate byte for byte.
- **Owner page:** published as a private artifact, https://claude.ai/artifact/CeymkxuQuPpsoT8TTrbveZ (ten clips
  inline, the turning figure, the five questions). Checked at phone width, in dark mode and on its side.
- **Merged since 21:50:** the views (`wt/wp-v`, merge `4e7e9b5`: `designlab/stage2/barviews.py`, its tests, the turn
  files). 18 clips rendered in the main checkout (`logs/barviews/`, git-ignored; manifest `views/clips.json`).
- **Result in one line (simulation):** under Stage 1's fitted controller nothing passes at the judged 10 ms; today's
  walk stays up 20 of 20 to 6 ms and passes 4 of 16 conditions with no delay; the candidate passes nothing as
  registered.
- **Recorded errors of the lead:** the plan's A12 sentence contradicts C4 on the joint range (plan, closing
  amendments, items 11 and 12; report section 8); lesson 19 in `LESSONS_LEARNED.md`.
- **Waiting for the owner** (report section 3): the judged delay, the tail servo, whether to keep the candidate,
  pull request #668, carpet. Nothing more is run on the 75 cm track until they answer.
- **Not adopted:** the draft rows F28, F29 and E in report section 11.
- **Paused, as before:** round C training tooling, the look sets, Track B rules, the milestone items after the
  75 cm track (R5/G8a/G9, R2L, B.3, S2-4, the notebook, the Drive bundle, the steps sheet).

## Update, 16 Nov 2026, 01:10 UTC: the owner answered (P37); the add-on is being designed

- **Answers (`DECISION_LOG.md` P37):** run today's walk at 14, 16 and 20 ms, then refit the stand-in controller with
  delay on this body (two fits), 10 ms staying the judged delay; the counter-phased tail of look V6 as the tail
  servo's remedy (nothing run now); no more runs on `F25F9cl`; pull request #668 closed as superseded (closed on
  GitHub 16 Nov 00:33 UTC, not merged; a dated note stands on line 2 of its report here); carpet as a range with no
  verdict.
- **Running:** a design workflow (`wf_5204a798-a6e`): two designers (delays and carpet; the refit), each followed by
  an independent reviewer. They change no file and run no episode. Brief: `stage2/design/WP_ADDON_75.md`; their
  texts land in `<scratchpad>/addon/design_AB.md` and `design_C.md`.
- **Next, in order:** read the designs and reviews, rule on the findings, write `STAGE2_75_ADDON_PLAN.md` and commit
  it before any episode; item A (delays) first, since today's runner may already take it; then B and C, each with
  its build reviewed before its run. Report each as an addition to `STAGE2_75_REPORT.md`, labelled as made after the
  main panels' results.
- **Usage:** the session's cost read 1,777 USD at 00:18 UTC, about 61 % of the week by the owner's 16:00
  calibration (11 % at 1,433.50 USD; baseline 1,357.5; 1 % is about 6.9 USD).

## Update, 16 Nov 2026, 01:20 UTC: the owner's usage line

- The owner read 65 % at about 01:15 UTC; the session's cost then read 1,808 USD (00:53 UTC), which is 65 % by the
  16:00 calibration, so the meter agrees. 1 % is about 6.9 USD; 90 % is about 1,979 USD.
- Asked where to stop, the owner wrote: "Run up to about 90% and then tomorrow and I will rest it". Read as: work on
  up to about 90 %, then checkpoint and stop; the owner will use the week's remaining manual reset tomorrow. Read the
  cost with `get_session` before each tranche and stop before one that would pass 90 %.
- To spend less: one independent reviewer per build, and a short labelled section for the add-on's results in place
  of another full report cycle.

## Update, 16 Nov 2026, 01:35 UTC: the order of work (P38)

- The owner chose "Training path first" (`DECISION_LOG.md` P38). Order from here: (1) item A of the add-on, today's
  walk at 14, 16 and 20 ms, with its rules committed first; (2) the training path toward the day-one Colab smoke:
  sections 2 to 8 of this file say what is built, what is in the bundle and what is not started (round C tooling,
  the look sets, the Track B rules with ten conflicts, validation B.3 and S2-4, the notebook, the Drive bundle, the
  steps sheet); (3) the refit with delay and the carpet range only if there is room: their designs and reviews are
  being written by workflow `wf_5204a798-a6e` and stay on file (`<scratchpad>/addon/`; copy what returns into
  `stage2/design/` so that it survives a restart).
- The usage line is about 90 % (about 1,979 USD on `get_session`'s meter), then checkpoint and stop; the owner
  resets the allocation the next day.

## Update, 16 Nov 2026, 01:50 UTC: item A registered; what to do next, step by step

- **Committed with this update:** `STAGE2_75_ADDON_PLAN.md` (section A registered; B and C not registered), the chain
  script `stage2/runs/2026-11-16-75cm-addon/run_addon.sh` (list mode tried: six units, 0.665 core-hours estimated
  against the cap of 1.0), and the designers' and reviewers' working papers under `stage2/design/addon/`
  (`design_AB.md`, `review_AB.md`, `design_C.md`, `review_C.md`, `proto/`).
- **No episode of the add-on has run yet.**
- **Next, in this order:** (1) one independent reviewer reads section A and the chain script and tries `list`;
  fix what is found; commit. (2) `PY=.venv/bin/python PROCS=3 bash stage2/runs/2026-11-16-75cm-addon/run_addon.sh
  all` from `hardware/trex_form_study` in the main checkout with a clean tree (about 15 to 20 minutes); commit the
  records, `budget.json` and `ADDON_LEDGER.json` under `stage2/runs/2026-11-16-75cm-addon/A/`. (3) Write
  `A/delays_table.py` (standard library, in no key): per delay 0 to 20 ms, each record's count of episodes up, of
  episodes meeting the distance term, where the falls come, and the reading words of rule A-6; registered records
  at 0 to 10 ms from `stage2/runs/2026-11-15-75cm/`. (4) A short labelled section in `STAGE2_75_REPORT.md` (rule
  A-10), checked by one reviewer; tell the owner (push notification). (5) Then the training path (P38): start with
  the ten conflicts of the Track B rules draft (`reviews/trackb_rules__integrate_trackb-rulings.md` in this
  directory; branch `wt/prereg-b`) and the fixer for the training env (`wt/wpc-ce`; its review and the lead's four
  rulings are in section 4 above).
- **Usage line:** about 90 % (about 1,979 USD on `get_session`); read it before each step.

## Update, 16 Nov 2026, 02:10 UTC: the owner asked for a buy gate (P39)

- **New, by the owner (P39):** stronger evidence of robust walking before about 20,000 USD is spent. The lead drafts
  `BUY_GATE_DRAFT.md` for approval: stages (harsher simulation and a trained policy; a bench of one leg drive and
  the tail servo, about 1,300 to 2,000 USD; a legs-only prototype with the 12 leg drives, 3,546 USD, plus
  structure; the full build) and what must be shown before each purchase. A realistic servo model (friction, gear
  play, voltage sag, delay, all randomised; see `stage2/research/microduck_delay_15nov.md`) joins the training path.
- **The lead's stated confidence** (judgement, not a result): about 65 % that the real robot ends up walking steadily
  at 0.35 to 0.40 m/s on a flat hard floor; about 90 % for the simulated walk at 4 ms or less; about 40 % that a
  trained policy passes the whole bar at 10 ms.
- **In flight:** the independent review of section A's rules and chain script (an agent, background). Then steps (2)
  to (5) of the 01:50 update. The keep-alive ended at 01:36 UTC and is not re-armed while agents run.

## Update, 16 Nov 2026, 02:20 UTC: item A of the add-on is run and reported; the training env's fix round is running

**Item A (more delays), done.** Simulation; no verdict; the bar stays judged at 10 ms.

- Section A's rules were amended before any unit ran, after an independent review (one blocker, eight major points;
  `stage2/design/addon/review_A.md`). The chain script took the reviewer's corrections and a lock; its failure,
  cap, resume and dirty-tree paths were run against a stand-in runner in scratch. Commit `f4a8b7d`.
- All six units ran, none cut: 47 records, 940 runs, 0.495 of 1.0 core-hour. Commit `f35ed65`.
- Result and reading: `STAGE2_75_REPORT.md` section 13 and `stage2/runs/2026-11-16-75cm-addon/DELAYS_TABLE.md`.
  One independent check of the section found no wrong number and asked for changes of wording, which are in
  (`stage2/design/addon/review_sec13.md`). The owner page is republished with the section (version 2).
- Owed: a draft decision-log row for this result (proposed number F30), to be put to the owner with F28, F29 and E.
- Items B (carpet range) and C (refit with delay) are designed and reviewed, not registered, and wait for room (P38).

**Buy gate (P39).** `BUY_GATE_DRAFT.md` now has the section on when the form factor is reconsidered. The owner has
not yet been asked to approve the stages and numbers. The servo model for the training path (friction, gear play,
voltage sag, delay; drawn at random) is not designed yet; it is the next design item after the training env is
merged.

**Training path (P38).** The fix round of the training env (CE, `wt/wpc-ce`) was started at 01:53 UTC with the
lead's four rulings and the review's nine findings (`reviews/roundC__review_CE.json`). The lead lifted rule X1 for
the frozen 75 cm sets in the fixer's brief: the panels ended on 15 Nov, so the env may now be compared with the
bar's runner on set `V0`. After the fixer: one independent review, then the merge into the working branch.

**Usage.** The session's meter read $1,881.93 at 02:15 UTC, about 76 % by the calibration of the 01:20 block
(65 % at $1,808.26; 1 % is about $6.9). The owner's line is about 90 % (about $1,979). The account's rate-limit
status now reads "allowed_warning".

**Next, in order:** (1) review and merge CE; (2) the servo model's design for gate 1, with sources, reviewed;
(3) the ten conflicts of the Track B rules draft and section 6's open points 6 and 7; (4) CT (trainer), CB (Drive
bundle), CN (notebook) toward the day-one Colab smoke; (5) the look sets' review, CV, CK, CW, CF, CG.

## Update, 16 Nov 2026, 03:02 UTC: the training env (CE) is merged; the lead's rulings on what was handed up

**Merged.** `wt/wpc-ce` is merged into the working branch at `9ecefa5` (nine files: the env's four modules, their
four test files, `stage2/design/CE_RULE_WORDING.md`). The fixer's report and the independent review of the fix
round are `reviews/roundC__fix_CE.md` and `reviews/roundC__review_fix_CE.md`. The review found no blocker and no
major point and said to merge as it stands. Both record-key hashes are the registered ones after the merge. On the
merged tree the lead ran `tests/test_stage2_r5lib.py`, `test_stage2_r5core.py` and `test_stage2_tbenv.py`: all
passed. The reviewer's run of the env and comparison tests on the same tree: "36 passed, 1 skipped". On the frozen
set `V0` the env and the bar's runner were identical step for step in four comparisons (`walk/v0.4` at 0 and 6 ms,
`walk/nominal` at 0 and 10 ms; no feedback row). That is a check, not gate G8a's record. The bundle still holds
`wt/wpc-ce`; drop it from the bundle at the next refresh.

**Rule X1** is lifted for the frozen 75 cm sets as of 16 Nov 2026 (the panels ended on 15 Nov), by the lead, in the
fixer's brief and now here.

**The lead's rulings on the points handed up** (to be written into the Track B rules draft with the wording file):

1. *A playback does not score exactly zero in walking turns.* The reference's yaw lags the commanded heading by up
   to 9.0° in the quick turn, so a playback's heading penalty reaches 0.351 there. Ruled: rule E4 stays in walking
   turns. The bar judges the heading against the command, so the penalty points the same way as the bar. Rule W1's
   sentence is amended to say so, as the wording file's entry 1b proposes, with the reviewer's two corrections: the
   tracking figure is scoped to the two sets it was measured on, and "every penalty 0" becomes "below 1e-4".
2. *The gaze target* moves from ±0.5 to ±0.45 rad with the jaw's limit, as the 0.15 rad margin requires. Accepted.
3. *What a resumed run compares* (K4): the trainer compares the config's named freeze, the input files and the task
   names, not the env's whole description. The env configs should store the resolved task names, so that K4's
   accepted case can be reached. This goes into CT's brief.
4. *The eight minor points of the review of the fix round* are accepted as findings and owed as one small fix round
   before any run fixes the env's source hashes: the task table is compared when the named freeze is the latest;
   a set whose `crouch_frac` differs from a used walk entry's is refused; six unstated details join the rule
   strings; the wording file gains an entry for the mid-start rule (T3); `from_plan` gains a heading argument
   before any row plays a turn in place; three test gaps.

**Not done on the training path:** the Track B rules draft's ten conflicts and the wording file's integration; the
look sets' review; CV, CK, CB, CT, CW, CF, CG, CN. The servo model's design is being written (a workflow; its
review phase will be cut to hold the owner's usage line, and the note labelled "not yet reviewed").

**Usage.** $1,937.41 at 03:01 UTC, about 84 %. The owner read 81 % at 02:47 UTC, which agrees with the calibration.

## Update, 16 Nov 2026, 03:23 UTC: the servo model's design is not written; its three inputs are saved

`stage2/design/servo_model/README.md` says where it stands. Three readers' reports are saved there (the code, the
hardware facts in the repository, outside sources). The designer was stopped by the lead's mistake before it wrote
the note (a watcher's time-out was taken for the designer's result), and was not restarted, to hold the owner's
usage line. Finishing it (designer, then one independent reviewer) is the first job after the owner's reset.

**Usage.** $1,950.41 at 03:12 UTC (about 86 %). The lead's estimate at 03:23 UTC is about 87 to 88 %. No agent and
no run is active. The lead stops here except for the owner's answers on the buy gate and the notes they need.

**Order of work after the reset:** (1) the servo model's design note and its review; (2) the eight minor points of
the training env's review; (3) the Track B rules draft: its ten conflicts, the wording file `stage2/design/
CE_RULE_WORDING.md`, section 6's open points 6 and 7; (4) CT (trainer; see the ruling on what a resumed run
compares), CB (Drive bundle), CN (notebook), toward the day-one Colab smoke; (5) the look sets' review, CV, CK,
CW, CF, CG; (6) only if room: the add-on's items C and B, after their rules are rewritten and reviewed. Owed to
the owner: the draft decision-log rows F28, F29, E and a row for the add-on's result (proposed F30).

## Update, 16 Nov 2026, 03:29 UTC: the owner's answers on the buy gate (P40); the session stops for the usage line

The owner approved the buy gate's four stages, gate 1's speed (0.30 m/s must pass; 0.40 m/s reported), gate 1's
numbers as drafted, and hard floors only for gate 1 (`DECISION_LOG.md` P40; `BUY_GATE_DRAFT.md` marks what is
approved and what is still proposed). Nothing is running. The tree is clean and pushed. The order of work after
the reset is in the block above; gate 1's conditions 2 and 4 and the numbers of gates 2 and 3 go to the owner once
the servo model is designed.

## Update, 16 Nov 2026, 14:18 UTC: the owner reset the allocation; work resumed; what is running

**The reset.** The owner said to resume at 13:13 UTC and to work until about 99 %, then reset the allocation at about
14:15 UTC ("Reset trigger"). The session's meter read $2,042.71 at 14:16 UTC and the rate-limit status went from
"allowed_warning" to "allowed". **New calibration:** 0 % is about $2,040; 1 % is still about $6.9. The old
calibration (0 % at $1,357.5) agreed with the owner's own readings to within a point up to the reset.

**Done since the 03:29 block.**
- The servo model's design note and its independent review are written and saved (`stage2/design/servo_model/`:
  the note, `REVIEW.md`, the readers' reports, a README with the history). Verdict: usable after the listed fixes;
  its section for the owner and condition 2's corner rule are not to go to the owner as written.
- The lead's rulings on the Track B rules draft's ten conflicts and on two open points of the policy files:
  `stage2/design/TRACKB_RULINGS3_16NOV.md`. A closing note in `STAGE2_75_PLAN.md` (item 13) for ruling 10.

**Running now.**
- Wave 2a of the training tooling (workflow `wf_5b45aea3-b83`): the worker pool (CV), the Drive bundle tool (CB)
  and the env's eight minor points (CE), each built, independently reviewed, then fixed, in the worktrees
  `wpc-cv`, `wpc-cb`, `wpc-ce`. The lead merges each after reading its review and fix.
- A documents wave (workflow `wf_1552bd0f-86f`): the review applied to the servo model's note and re-checked;
  rulings 1 to 10 written into the Track B draft (`wt/prereg-b`), reviewed and fixed.

**Next, in order:** merge CV, CB and the env's fixes; the trainer (CT; its brief gains the ruling on what a resumed
run compares); the look reading (CK) and the look sets' review (L); the policy files' `actions_at_bound` (ruling
12); the reward wording into the Track B draft (ruling 13); the rows (CW), the refit tool (CF), the gates (CG) and
the notebook (CN); then the validation steps of `ROUND_C_DESIGN.md` section 6 toward the acceptance run and the
steps sheet for the day-one Colab smoke. Gate 1's condition 2 goes to the owner once the servo note's fixes are in.

## Update, 16 Nov 2026, 20:47 UTC: the second allocation is at about 90 %; what is merged, what is running, how to resume

**Usage.** After the owner's reset at about 14:15 UTC the meter's zero is $2,042.71; by the owner's reading at
18:45 UTC, 1 % is about $6.76. The reading at 20:46 UTC was $2,652.78, about 90 %. The owner was pushed at
85 %. Nothing new is launched. Near 98 % with no reset showing, the lead stops the two running workflows.

**Merged on the working branch since the reset** (each built, independently reviewed and fixed; reports under
`reviews/`; rulings in `../ROUND_C_RULINGS_16NOV.md`): the worker pool (CV), the Drive bundle tool (CB) and its
follow-ups, the env's small fix round (CE), the policy files' `actions_at_bound` (CP), the look reading (CK)
with the tail's amplitude band, the trainer (CT) with the monitor's refusal. The Track B rules draft is on the
branch as `STAGE2_TRACKB_PLAN.md` with five lists of the lead's rulings written in; the sixth list
(`../TRACKB_RULINGS6_16NOV.md`) is ruled and not written in. Tests seen on the merged tree at each merge; the two
record-key hashes are the registered ones throughout.

**Buy gate.** The servo model's design note went through two independent checks (`../servo_model/`). The owner
approved gate 1's condition 2 (`DECISION_LOG.md` P41). Five questions for the maker of the leg drives are in
`../servo_model/MAKER_QUESTIONS.md`, checked against the study's record only: the session could not read the
maker's pages (two attempts were refused by the permission check after the owner allowed the hosts). The owner
has not yet said whether to send them as they stand.

**Running at 20:47 UTC.**
- Workflow `wf_fbceb97c-fa4` (task `w17pnouw7`): the rows package (CW, `wt/wpc-cw`) is built and under review;
  the notebook and steps sheet (CN, `wt/wpc-cn`) is built and reviewed, its fixer is running.
- Workflow `wf_debdfcfc-4dd` (task `wuysq2rgi`): the look sets (L, `wt/wp-l`) are still in their build, about
  five hours in. An earlier report says its new build of set V6 is blocked by its start; the builder has not
  handed back.

**To resume after a stop.** Each workflow resumes from its saved script with its run id
(`round-c-wave-4-wf_fbceb97c-fa4.js`, `round-c-wave-3-builds-wf_debdfcfc-4dd.js` under the session's
`workflows/scripts/`): finished agents return from the journal, and an agent cut mid-run starts again and finds
its commits in its worktree. In a new session, start from the worktree branches and the briefs.

**Next after these land:** merge CW and CN; the look sets' review and the lead's decision on freezing them; the
refit tool (CF) and the gates (CG); the sixth list into the draft; then the validation steps of
`ROUND_C_DESIGN.md` section 6, the real bundle, the acceptance run and the steps sheet's final values for the
day-one Colab smoke.

## Update, 16 Nov 2026, 21:13 UTC: handover note before the allocation ends; the owner will switch the session's model

The owner said at about 21:12 UTC: once usage reaches 99 %, the session is switched to another model. The meter read
95.5 % at 21:11 UTC. The lead pauses the two running workflows at about 98 %, so that no agent is cut by the limit
in the middle of a turn, and they are resumed after the switch.

**State at 21:13 UTC.**
- Rows package (CW, `wt/wpc-cw`): built and independently reviewed; its fixer is running.
- Notebook and steps sheet (CN, `wt/wpc-cn`): built and independently reviewed; its fixer is running.
- Look sets (L, `wt/wp-l`): still in their build.
- Worktree heads at this time: wpc-cw 9952f5b ahead 8 dirty 3;wpc-cn 36347d4 ahead 11 dirty 0;wp-l b72c61c ahead 17 dirty 1;
- The reports of the builds and reviews that have finished are in the workflows' journals
  (`subagents/workflows/wf_fbceb97c-fa4/journal.jsonl`, `wf_debdfcfc-4dd/journal.jsonl` under the session's
  project directory) and are saved to `reviews/` when their package is brought in.

**For whoever continues, on any model.**
1. Resume the two workflows from their saved scripts with their run ids (the block of 20:46 UTC names them).
   Finished agents return from the journal; a fixer that was cut starts again and finds its commits in its
   worktree.
2. **The two attribution lines.** Every commit message ends with the two lines the session's instructions give
   at that time. The saved workflow scripts carry the lines that were in force when they were written, inside
   the agents' prompts. If the session's instructions give other lines after the switch, change them in the
   scripts before resuming, so that no commit is attributed wrongly. No model is named anywhere else in a commit
   or in code.
3. Before a branch is brought in: read its review and fix reports; check every commit that is not yet on the
   working branch for the two lines (a builder's merge commit without them stays out: bring the package in by
   cherry-pick, as was done for CB, CE and CK); compare the package's files with the reviewed tip; run the
   package's tests on the merged tree with `-o addopts=""` so that the summary line prints; confirm the two
   record-key hashes.
4. The standing rules are at the top of this file's history and in `LESSONS_LEARNED.md`: simulation only; rules
   fixed before runs; two seeds for any verdict; an independent review of every build, report and owner page;
   nothing adopted without the owner; push before a turn ends; never force-push.
5. Open with the owner: whether to send the five maker questions as they stand
   (`../servo_model/MAKER_QUESTIONS.md`); gate 1's condition 4 and the numbers of gates 2 and 3; the draft
   decision-log rows F28, F29, E and a row for the add-on's result (F30).

## Update, 16 Nov 2026, 21:20 UTC: both workflows paused at 97.3 % of the allocation

The meter read $2,700.74 at 21:19 UTC, 97.3 %, with no reset showing. The lead stopped the two running workflows
(tasks `w17pnouw7` and `wuysq2rgi`) so that no agent is cut by the limit mid-turn. No agent and no run is active.

- wf_debdfcfc-4dd finished: build:CK, build:CT, review:CK, review:CT, fix:CK, fix:CT | cut mid-run: build:L
- wf_fbceb97c-fa4 finished: build:CN, build:CW, review:CN, review:CW | cut mid-run: fix:CN, fix:CW
- Worktrees at the stop: wpc-cw at 9952f5b, 8 commits ahead, 4 uncommitted files; wpc-cn at b54c0aa, 12 commits ahead, 0 uncommitted files; wp-l at b72c61c, 17 commits ahead, 1 uncommitted files; 
- The build and review reports of the rows package and the notebook are saved under `reviews/`
  (`roundC__{build,review}_CW.md`, `roundC__{build,review}_CN.md`). Neither package is merged: each fix round was
  cut. An uncommitted file in a worktree is a cut agent's work in progress: look at it before resuming, and let
  the resumed fixer decide what to keep.

Resume as the 21:13 UTC block says.

## Update, 17 Nov 2026, 01:13 UTC: the session resumes on another model

The owner switched the session's model after the pause of 21:19 UTC on 16 Nov. The session's instructions now give
other attribution lines for commits; every commit from here ends with them. The two workflows of 16 Nov are not
resumed from their saved scripts, because their prompts carry the earlier lines: the stages that were cut (the
rows package's and the notebook's fix rounds, the look sets' build) run from new scripts, with the lead's rulings
of 17 Nov (`../ROUND_C_RULINGS_17NOV.md`). The worktrees hold the cut agents' work: `wt/wpc-cw` four modified
files, uncommitted; `wt/wpc-cn` one more commit than at the stop; `wt/wp-l` one untracked set directory.

**Usage.** The session's meter read $2,701.94 when work resumed. The owner's readings so far were per model; the
lead takes this reading as the zero of the new model's allocation until the owner gives a percentage.

## Update, 17 Nov 2026, 01:19 UTC: the usage calibration on the new model

The owner's screenshot at about 01:17 UTC: the week's bar 57 % used, resetting Saturday 3:00 AM (owner's time);
Fable capped at 50 % of the week and 97 % used of that. The new model draws on the week's remaining 43 %. With the
session meter at $2,710.92: Fable's $659 since the reset was 97 % of its cap, so 1 % of the week is about $13.6
on the meter (to be checked against the owner's next reading). The owner holds one reset. The lead pushes the owner
at 85 % and at 95 % of the week, and recommends using the reset at about 95 %.

## Update, 17 Nov 2026, 01:56 UTC: three jobs of the resumed round did no work; they run again

The owner's question at about 01:12 UTC (a summary of the last days and when training can start) reached the
agents of the resumed round as well. The rows package's fixer, the notebook's fixer and the trainer's follow-up
builder read it as their task and did read-only status work: nothing was fixed or built, and their checks said so
(`reviews/roundC__{fix,check}_{CW,CN}_17nov.md`, `reviews/roundC__{build,check}_CT3.md`; the cut fixer's
uncommitted diff in `wt/wpc-cw` is still there and the check found it right for findings 1 to 6). The look sets'
builder kept to its task and is still running. The three jobs run again from a new script whose prompts say that a
relayed message from the owner is addressed to the lead (workflow `wf_b61975bc-02b`). Lesson for the lead: tell
every agent that relayed owner messages are not its task.

## Update, 17 Nov 2026, 03:36 UTC: usage recalibrated; the owner says to work through the night

The owner's second reading: 59 % of the week at about 03:32 UTC, with the meter at $2,790.53; the first was 57 % at
$2,710.92. So 1 % of the week is about $40 on the meter (between $27 and $72, the readings being whole percents),
not the $13.6 first estimated. The owner said to keep working through the night without stopping; the owner holds one
reset. The lead pushes the owner at 85 % (about $3,830) and at 95 % (about $4,230).

## Update, 17 Nov 2026, 05:42 UTC: overnight handover (what is merged, what runs, what is owed)

**Merged overnight on the working branch** (each built, independently checked, fixed; reports in `reviews/` and
`../servo_model/`; the record-key hashes are the registered pair throughout): the rows (CW), the notebook and steps
sheet (CN), the trainer's third follow-up (CT3), the follow-up before the acceptance run (CN, CT, CW), the gates
(CG), the R2L refit tool (CF), the servo model's core layer (S1), and the Track B draft with the sixth list.

**Rulings of the night** are in `../ROUND_C_RULINGS_17NOV.md` (1 to 20). The two that matter most: ruling 18 (no look
set can be frozen as built: the look start is over the knee's speed limit, 1.126 at 75 cm and 1.234 at 60 cm; the
start now mixes the stand's law by the amplitude ramp, a change after a result; the sets are built again, 75 cm
first) and ruling 16 (the bar refuses Track B rows on a look set; `bar.py` is not edited; `lookbar` carries its own
narrow rule for the look sets, after a passed G9-ref).

**Running at 05:42 UTC.** Workflow `wf_daa54001-cce`: the look sets' fix round (its check said: merge by cherry-pick;
one major point outside the package for the lead: the env and the bar place a look set's turns and stop2 after
whole windows, not at J1's leave point). Workflow `wf_5fdcc198-2b2`: the lookbar look-set rule (16), the gates'
G9-ref fixes (14, 15), the trainer's owed points (17), the sheet's 5.3 (13), each with a check; and the
independent review of the join code (19).

**Owed next.** Bring in the look sets (cherry-pick) and launch their rebuild under ruling 18 with a check; rule on
the check's major point (turns and stop2 placement); a CF follow-up (its integration with G9-ref's record format,
and the chain's `--resume` on V10); the servo model's S1 low points and S2 (the env option and G8s); then, once a
75 cm look set is frozen: G9-ref, G8a, the validation steps, S2-4, the bundle and the acceptance run.

## Update, 17 Nov 2026, ~07:00 UTC

- **Merged:** the look sets package (WP L), squashed as 036ec9d6 (tree equal to the reviewed tip 628d60b7);
  merged-tree tests "244 passed, 3 skipped"; hashes the registered pair. Ruling 27: `sets.py` leaves D18's state-A
  test (lookbar state-A tests "2 passed").
- **Ruled:** 21–26 (the look sets check: turns and stop begin at J1's leave point, a code error; turn in place
  recorded as not available; minor points into the rebuild), 27, and 28–30 on the join-code review
  (`reviews/roundC__review_JR_17nov.md`): a causal move's preview grid comes from the frames up to the command on
  the look sets (A6; frozen sets keep their records, dated note in A6); cut 3 not taken where the review shows it
  fails at the boundary too (labelled, written into L12). Rules draft: "L1 and L4 in a start" (ruling 18) and
  L10's toes-up reach (ruling 25) written in.
- **Running:** wf_5fdcc198-2b2 (checks of LB16, CG2, CT4, CN3); wf_3ccbe5e7-793 (build:LS, the start's law, then
  its check).
- **Owed next:** merge LB16, CG2, CT4, CN3 as their checks allow; then the ruling-21 package (r5lib and lookbar)
  and the CF follow-up; after check:LS, the 75 cm look-set rebuild of ruling 30 with a check; then the 60 cm sets.
  For the owner later: the left turns of the look sets (by the review's scratch numbers 4.62 mm on the causal grid,
  against the owner's 4.0 mm limit of P34), with the rebuilt numbers.
- **Meter:** $3,012 at 06:44 UTC, about 64.5 % of the week.

## Update, 17 Nov 2026, ~10:00 UTC

- **Merged and pushed since 07:00:** LB16, CG2, CN3 (32b5ecc5; merged tests 60, 43, 66 passed); the start's law
  (ruling 18; cherry-picked c5544698..7cb824ec; merged tests 298 passed, 9 skipped): every look start now passes
  on a scratch build (75 cm knee 0.80, 0.85, 0.79; 60 cm 0.92, 0.98, 0.90 of the limit); wave 8, i.e. CT5, CN4,
  CF2, LT (96d38d2d; merged tests 47, 36, 94, 68, 86 passed; run_trackb.sh syntax ok). Hashes the registered pair
  throughout. Disk cleared from 99 % to 60 % (old scratch deleted).
- **Ruled:** 31–38 (the look sets' rule in lookbar; G9-ref; the trainer; the sheet; CF; the start's law; the
  placement wherever turns and stops are played; wave 8's points). Old texts updated: G9-ref (f) and (a) in the
  draft and the design, X1, M3.
- **Running:** wf_348ef162-856 (look-set rebuild: code of rulings 28, 32 A, 36 committed on wt/wp-l at 07347c77;
  75 cm tail phases rebuilt; the three 75 cm records building; then their check; then 60 cm and its check);
  wf_2b4a8be6-560 (LT2: placement-only context for G9-ref (a), G8a, the equivalence check and R2L's fit; F1 and F2;
  four commits so far on wt/wpc-cw).
- **Owed next:** merge LT2 and the rebuild as their checks allow; draft the owner's questions from the rebuilt
  numbers (the left turns against P34's 4.0 mm; any start or change still failing, the 60 cm change to 0.40 m/s
  most likely; the turn in place); then freeze (lead), G9-ref, the validation steps, S2-4, bundle, acceptance run.
- **Meter:** $3,162 at 09:50 UTC, about 68.3 % of the week.

## Update, 17 Nov 2026, ~13:00 UTC: usage

- The owner's reading at 12:56 UTC: **75 % of the week** at $3,303.08 on the session meter. The calibration is
  refitted: 1 % is about $32, so 85 % is about $3,623 and 95 % about $3,943. At today's pace, about 1.4 % an hour,
  85 % is about 7 hours away. The rate-limit status turned "allowed_warning" at the same time.
- Since 10:00 UTC: LT2, LT3, LT4 and servo S2 are merged (merged tests pass; hashes the registered pair); servo S3 is
  built and in a fix round (fifth list); the three 75 cm look sets are rebuilt (wt/wp-l a3ea12c0, not frozen), and
  their check is running, with the 60 cm rebuild after it.
- Priority while usage lasts: what brings training closest. That is the look sets merged and frozen, the owner's
  question on V6's left turns, G9-ref, the validation steps and S2-4. The servo work for gate 1 (S4) comes after.

## Update, 17 Nov 2026, ~17:15 UTC: usage

- The owner's second reading: **79 %** at $3,401.33 (17:14 UTC). Refitted: 1 % is about $25, so 85 % is about $3,551
  and 95 % about $3,801. Pace about 1 % an hour. Scheduled check-ins written before this carry the older figures
  ($32 a point); use these.
- State: the 75 cm look sets under P42 are built (wt/wp-p42 at 4ad729c5); their check is running. The 60 cm rebuild
  is on its last set (V6/60 and V6noF5/60 are stopped at nominal_to_v0.4, as before). After the check: freeze the
  75 cm sets, G9-ref, then the validation steps V2 to V7, then S2-4.

## Update, 17 Nov 2026, ~19:20 UTC: the 75 cm look sets frozen, validation done, S2-4 running

- **Owner's decisions today:**
  - P42: on the look sets the warm-up join reading is held to 5.0 mm, so V6's left turns are available.
  - P43: the 60 cm look sets are parked until the 75 cm training runs. Their records stay on wt/wp-l; their check
    said merge after a test-only fix, by cherry-pick.
- **75 cm look sets:** V6/75-M-H-10, V6noF5/75-M-H-10 and V6/75-M-H-10-X are frozen (freeze 0, mixed sources
  accepted by ruling 41). G9-ref passed on each.
- **Validation of 75-M-H-10 on V6 (merged 26674562, checked sound):**
  - G8a passed: 32 of 32 episodes identical.
  - R0L's records ran on 13 conditions; the 4 quick turns were not run.
  - G9-run passed after the code fix of ruling 42 (merged ea6c0a94).
  - Filter 10 Hz; learning rate 1e-5.
  - gl-start, r6-start and kill-resume (K3) passed.
  - Rates, provisional: about 536, 577 and 729 env steps/s at 2, 3 and 4 workers; 1 to 2 GB per worker.
- **Servo:** S1 to S3 are merged, S3 being gate 1's runner and the served rows SV0 and SV2. S4 (measuring) waits for
  the training path.
- **Running:** wf_febb2c1a-ba8, S2-4 in wt/wp-p42: the R0L baseline committed, then 2,000,000 steps of R5 on
  75-M-H-10 (V6, seed 42, 3 workers), collect, S2-10, then an independent check.
- **Next after a go:**
  - V11, the bundle;
  - V0b, the upload route (the owner approves each upload);
  - V12, the acceptance run;
  - V13, the steps sheet;
  - then the owner's day-one Colab smoke.
- **Meter:** 79 % at $3,401 (owner's reading, 17:14 UTC); 1 % is about $25; 85 % is about $3,551.
