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
