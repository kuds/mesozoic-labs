# Lead's notes for the Stage 2 build (second session, started 14 Nov 2026)

Scratch file, not committed. Paths relative to /home/user/mesozoic-labs/hardware/trex_form_study.

## Owner's decisions, 14 Nov 2026 (DECISION_LOG.md P30)

1. Scope: 75 cm + Jetson track first; 60-N-T-10 stays in the plan (P29 unchanged).
2. Compute: study the Jetson as a refine case; decide after its screens and the delay bench test; P14 stands.
3. Real-time loop: microcontroller baseline.
4. Judged delay: 10 ms; 0-8 ms reported beside it.
5. Bench tests: first, on both paths, before any compute purchase.
6. F25 tuned walk + look feature F9's tail + duty/lift margin: a labelled candidate reference beside the look ladder.
7. L1 pick: V6 (every look feature).
8. A look feature may cost a speed rung (character first).
9. Duty band 0.55-0.72 below 0.30 m/s; 0.55-0.65 from 0.30 m/s up.
10. Viewers: owner's pick decides L1 now; 3-5 viewers' scores later as a labelled late reading.
11. Budget: decided after the rebuild at catalog prices.
12. PR #668: kept open, unmerged.
13. Seven draft rows approved (F26, F27, A1/A15/A16 updates, A17, A18).
14. Colab plan: 24 h sessions, background execution, High-RAM CPU, 3 runtimes at once: all confirmed.

## Facts established this session

- Container: same CPU model (Xeon 2.30 GHz, 4 CPUs) and numpy BLAS (scipy-openblas 0.3.34 SkylakeX) as session 1.
- Venv: tools/setup_venv.sh failed at start because download.pytorch.org redirects wheels to download-r2.pytorch.org,
  which the environment's network policy blocked. The owner approved adding that host to the "mesozoic-labs"
  environment on 14 Nov. torch 2.15.0+cpu installed; versions as pinned.
- Drive route verified both ways on 14 Nov: create_file with base64Content "sha256:<hex>:<abs path>" uploaded a 10 KB
  tar and a 2,123,104-byte zip (sizes match); download_file_content of the 2 MB zip is saved by the harness as raw
  binary under /root/.claude/projects/.../tool-results/mcp-Google_Drive-blob-<ms>-<rand>.zip (path printed in the tool
  result), sha256 identical (3603...0baa). Small files come back inline as base64 JSON. Drive folder
  MyDrive/mesozoic-labs/stage2 created (id 1aAaOggKMoyG7qWDAO08XHMYOR6O8OZmk); parent mesozoic-labs id
  1njoU_1mF0k6_OTZvHUXbmFYL28efE4JQ. Test files there: drive_test_small.tar, drive_test_checkpoint_2MB.zip.
- An hourly scheduled-task backstop (create_trigger) was denied by the permission classifier; the keep-alive is the
  only re-invocation mechanism.
- Code maps (by subsystem) are in ../maps/ (01..09).

## Architecture intentions (to be confirmed by the design review)

- New Stage 2 code goes in designlab/stage2/ wherever possible; Stage 1 modules are edited only behind options that
  default to today's behaviour. Stage 1 and 1b records are cited, never rerun.
- "Reference sets": a directory in Stage 1's run-dir layout (<root>/<body>/refs/*.npz, transitions/*.npz, refs.json,
  transitions.json) per (set, body), so that Stage 1's Planner, Context and control load them unchanged.
  Sets: V0, F25F9, F25F9cl, V6 (frozen look), V6 without F5 for R2L.
- A generalised planner for starts, changes, two-step stops and finite 90 degree turns for any GaitSpec; for legacy
  inputs it must reproduce transitions.generate's arrays exactly (regression test).
- Bar readings and conditions in a Stage 2 module, hooked into the panels like --row-module.
- Code freeze points: state A (task 0) before the 75 cm panels; state B (R5, train extensions) before Track B runs.

## Rules to pre-register (drafted 14 Nov before any Stage 2 run; to be written into the repo and reviewed)

Stage 2 plan amendments (LESSONS 5.3), each "a decision taken before results":
- S2-9 whole-step delays: every delay is k x 2 ms. Judged 10 ms. Reported beside the bar: 0, 4, 6, 14, 16, 20 ms
  (each odd value of the plan replaced by its two neighbours; never read between delays). Ladders also at 0, 4, 6 ms.
  G8b fallback: 0, 4, 6, 10 ms. Training randomisation: k uniform integer 0..12 (0-24 ms). R2L mix k 0..5 as Stage 1b.
- S2-10 S2-4's reading point: the section 6 monitor (18 episodes at 10 ms) on the run's final checkpoint (>= 2.0M
  steps, actual count recorded) against R0L on the same 18 episodes, with section 6's gain rule; intermediate
  checkpoints reported only. A go/no-go, not a verdict (lesson 15).
- S2-11 G9's traces: Stage 1b's 10 ms turn traces of 60-N-T-10 regenerated from unchanged code into scratch, each
  checked against the sha256 in its committed episode record; for 75-M-H-10 (no Stage 1b rows) R0L's and R2's own
  10 ms in-place turn traces. Regeneration started 14 Nov 21:41 UTC (scratchpad/g9_traces).
- S2-12 judging Colab checkpoints here: flagged cross-version rule; versions trained/evaluated recorded in every
  policy-row record; torch build label ("+cpu", "+cu...") normalised on both sides; each kept checkpoint stores a
  policy fingerprint (fixed observations -> deterministic actions) that must reproduce here within a set tolerance.
- S2-13 resume across library versions: --allow-version-change, flagged in run.json sessions[] (as
  --allow-source-change).
- S2-14 picklable learning-rate schedule (module-level class), and loads through a safe loader.
- S2-15 L1: V6 frozen by hash per body and speed; 75 cm animal-pace rung lost (knee 1.012); duty band 0.55-0.72 below
  0.30 m/s.

75 cm track (tasks 2-4):
- Sets: V0 (control), F25F9 (control; PR's _tuned_f9_spec verbatim), F25F9cl (candidate; duty 0.59, lift 0.095 H at
  every speed; rungs other than 0.40-0.55 m/s labelled new; "margin set after results", on seeds 3042-3061).
- Speeds: nominal (Fr 0.012), 0.30, 0.35, 0.40, 0.45, 0.50 m/s.
- Task 3 rows: R2/42 and R2/43 unchanged at k = 0..5 steps (0-10 ms); mu 0.9 and 0.54; seeds 3042-3061; commanded
  5 % above; WALK_RULE's plan (start, 1 nominal stride, change, K strides; last 10 s judged).
- Readings: Stage 2's bar reading is the headline (own-swing toe clearance; >= 4.0 m along the heading within
  0.25 m); Stage 1's reading (tune.SUCCESS walk) beside it for comparison with JPS 2.3.
- Because F25F9cl's margin was set on seeds 3042-3061, any condition it passes is re-run on fresh seeds 5042-5061
  (labelled replication; the headline stays the study's seeds).
- Verdict: PASS_RULE per row; shown if both rows pass, provisional if one. Judged at 10 ms; others reported.
- Attribution: decision 20 on conditions with >= 5 falls.
- Task 4: on F25F9cl and V0: 90 degree turns left/right (pi/4), four nudges at 0.25 x bound, the two-touchdown stop;
  at 10, 0, 4 and 6 ms; both grips. Thigh twist screened first (map 03: the pi/4 turn from V0's 0.40 m/s gait
  saturates the +-0.30 rad twist on 75-M-H-10; first remedy: quicker steps inside the turn; else owner, lever 6).
- Budget: 12 core-hours for task 3 and 12 for task 4; order 10 ms, 0 ms, then 4, 6, then 2, 8.

## Kinematic twist screens (14 Nov, scratch; zmp_iters=0, steady turns from look.speed_spec's 0.40 m/s walk)

design/twist_ladder.json and twist_screen.py. Hip-yaw use of +-0.30 rad (worse leg), by stride scale at the same speed:
- 75-M-H-10, pi/4 rad/s: x1.0 (T 0.861 s) 1.000 with IK failures; x0.9 0.914; x0.8 (0.549 H, T 0.688 s) 0.805;
  x0.7 0.696; x0.6 0.588. 1.0 rad/s: pinned down to x0.8; x0.7 0.902; x0.6 0.759. 0.5 rad/s: 0.626 at x1.0.
  F25F9cl's 0.40 m/s gait (0.792 H, T 0.993 s): pinned on both legs at pi/4.
- 60-N-T-10, pi/4: x1.0 (T 0.751 s) 0.902; x0.9 0.804; x0.8 0.705. 1.0 rad/s: pinned to x0.9; x0.8 0.919; x0.7 0.787.
- Stage 1b's 10 ms turn traces of 60-N-T-10 regenerated into scratchpad/g9_traces: 160 of 160 byte-identical to the
  sha256 in the committed episode records (design/check_traces.py). This is also the closed-loop bit-identity
  baseline for any Stage 1 edit.
- Scratch Stage 0 pricing of steady pi/4 turns at 0.40 m/s on 75-M-H-10 with V0's tail (design/turn_price.py,
  turn_price_75-M-H-10.json; 6 ZMP passes, exact split): stride x0.9 (T 0.775 s): hip-yaw use 0.918, worst ratio
  0.906 (hip-pitch speed); x0.8 (T 0.688 s): use 0.807, worst 0.993 (tail peak; knee speed 0.919, hip-pitch speed
  0.924); x0.7: tail peak 1.20, knee speed 0.975 (over). Straight walk at x0.8: tail peak 0.969, knee 0.859.
  So quicker steps trade thigh twist against the tail's peak (it swings at the stride frequency) and leg speed.
- Same scratch pricing on 60-N-T-10 (turn_price_60-N-T-10.json): the steady pi/4 turn at 0.40 m/s is over the Stage 0
  SPEED limits (with the 1.15 margin) at every stride tried: thigh-twist drive (AK45-36) speed 1.135 (x0.9), 1.104
  (x0.8), 1.079 (x0.7); knee speed 1.044 / 1.075 / 1.133; hip-pitch speed 1.013 / 1.028 / 1.056. Hip-yaw RANGE use
  0.808 / 0.707 / 0.608. So at 60 cm the quick turn is limited by drive speed, not twist range (to be confirmed by the
  official screen; closed-loop LIMITS_RULE reads speed without the margin). A finding for the owner when confirmed.
- DECISION_LOG.md P30, F26, F27, A17, A18 committed as 167dce4 (14 review findings, all applied) and pushed.

## Lead's rulings on the round-A draft design (14 Nov, 23:10 UTC; the architect's final version must follow these)

Read after the draft's section 7 ("Risks and open questions for the lead"):
1. Tail and neck load in the quick turn's quicker steps: option (a). Inside a move whose stride period T(p) is
   shorter than the walk's, the tail's and the neck's swing amplitudes scale with (T(p) / T_walk)^2 (constant
   angular acceleration amplitude), for every set, look sets included. Pre-registered as a reference-design rule,
   labelled as a look choice; the owner is told (F11's head-first turn and tail flick stay deferred). If a turn is
   still over a neck or tail planning limit at Stage 0 after this rule, it is recorded and read under the plan's
   third reading ("shown but for A1"); if it is over a LEG drive limit, D7's "no reference frozen" applies.
2. Causal commanded moves (D5): accepted. Stops and turns are causal: the file begins at the command instant and
   nothing before it knows of the command. It is pre-registered as an S2 rule with the alternative (Stage 1's
   convention, command = start of the reference's ramp) named, and decision 18's reading kept in every record.
   Starts and speed changes stay Stage 1's construction.
3. The stop (D8): accepted as drafted (Stage 1's stop plan, causal, command at the right foot's lift-off; two
   touchdowns). The legacy one-step stop from 0.40 m/s is built as a labelled comparison.
4. A set or body with no quick-turn reference under D7 (leg drive over a Stage 0 limit, or no stride factor >= 0.60
   within the twist rule) gets none; the record says so with the numbers and it goes to the owner. Do not design
   around it. This is expected to matter for 60-N-T-10 (drive speed) and possibly for F25F9cl.
5. Definitions confirmed: the 0.25 m band is about the line along the commanded heading through the CoM's position
   at the start of the judged 10 s (the deviation from the episode-start line is reported beside it); a turn's
   "start heading" is the commanded heading at the command (the trunk's own heading there is reported); an air run
   shorter than half the shortest planned swing is an air gap, not a swing; ladder turn rates are read at bar time
   (pi/2)/rate; V0 above FAST's speed keeps FAST's stride, duty and lift (labelled new); turn ramps are half a
   stride; the replication block (seeds 5042-5061) runs on every condition the candidate passes in the first block.
6. Panel integration (D1, own runner in designlab/stage2/bar.py, no Stage 1 byte changed in round A): accepted,
   provided the runner's records carry Stage 1's rules they rely on, their own source hash and the panels' key
   context, and that R5's policy rows and Stage 1's twelve "no loss" conditions are shown to plug in.
7. Order of work stays: the 75 cm non-look sets and panels first; the look laws (WP L) never block them.
8. One more requirement the draft must cover: the Jetson body (75-M-H-10-J, a refine case built like the -X head)
   will be merged before code state A is frozen; its code edits (designlab/build.py, refine.py, power.py, the Stage 1
   registry) are outside round A's work packages. Round A must therefore not assume designlab/build.py, mjcf.py,
   stage1/__init__.py or stage1/refs.py stay byte-identical between today and the freeze; any regression baseline
   that depends on the panels' source hash must be taken at the freeze commit.

## Lead's rulings on the FINAL round-A design and PREREG_75_draft.md (14 Nov, 23:50 UTC)

These settle the final design's "open questions for the lead" and the draft's "For the lead before committing":
1. Air gaps (PREREG C3): the architect's amendment is ACCEPTED (an air run is a swing if any 100 Hz update inside it
   falls in the reference's swing of that foot, or if it lasts at least 0.5 x the shortest planned swing; otherwise an
   air gap). Turn ramps (A10): 0.375 s of reference time for every set, body and rate: ACCEPTED. Ruling 5 of 23:10 is
   amended accordingly.
2. Budgets: task 3 gets 16 core-hours, task 4 gets 12 core-hours, counted as the draft's H6 says; the tier order of
   H6 is CONFIRMED. One addition at the end of task 3's tier (3): a reported omega control, F25F9cl walk/v0.4 at mu 0.9
   under R2/42 at 10 ms and at 0 ms with Stage 1's 0.03 H crouch in omega (40 episodes), labelled a control.
3. Reading 2 ("shown": every limit) is the headline for these three sets, reading 3 ("shown but for A1") is printed
   beside it and never pooled; the expectation sentence of C2 stays. Either reading passing triggers replication.
4. A set with no quick turn under the stride rule gets "not run" records with the numbers; it goes to the owner.
5. The one Stage 1 edit (control.Track.add_periodic's optional window_strides key) lands BEFORE code state A, guarded
   by the committed regression record. The Jetson body's code (designlab/build.py, refine.py, power.py, the Stage 1
   registry) also lands before state A. After state A no module in a bar key changes until tasks 3-4 are done.
6. Shared lead-in and one seed slot for all eight items: ACCEPTED (labelled as differing from Stage 1).
7. Peak ground reaction reported, not judged, in the turn and the 0.40 m/s walk: ACCEPTED.
8. JPS cells with no counterpart print "not run".
9. Round C (R5, R6, R2L) must not edit a module that is in the bar's or the panels' key (control.py, plant.py,
   metrics.py, tune.py, refs.py, transitions.py, estimator.py, drive.py, gait.py, turning.py ...): subclass or wrap in
   designlab/stage2 instead. If an edit there proves unavoidable, R4's rule applies (affected baseline rows re-run
   into a new directory).

## Workflows running (resume with Workflow({scriptPath, resumeFromRunId}) after a restart)

Scripts under /root/.claude/projects/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/workflows/scripts/
- trex-round-a-build-wf_4b831ec4-af2.js   run wf_4b831ec4-af2   worktrees /home/user/mesozoic-wt/wp-{m,r,g,c,s,b,t}
  (branches wt/wp-*); merge order M, R, G, C, S, B, T; then full suite, moves legacy check, integration review.
- trex-prereg-75-wf_3f5b8bf4-8b0.js       run wf_3f5b8bf4-8b0   worktree /home/user/mesozoic-wt/prereg (wt/prereg):
  STAGE2_75_PLAN.md and STAGE2_PLAN.md section 9; merge and push as soon as it commits (it must precede any freeze).
- trex-jetson-body-wf_449c3010-848.js     run wf_449c3010-848   worktree /home/user/mesozoic-wt/jetson-body
  (wt/jetson-body): 75-M-H-10-J; merge before code state A; then add its restore group and rerun budget_75.py.
Done and merged: decision log (167dce4), hardware documents (stage2/jetson75: a55accb, 2b43361, feb84ad).
Still to do after these: restore check + full suite; G1 of -X and -J on an idle machine; owner pages (budget and
bench sheet); round C/D design (R5, R6, R2L, validation, S2-4, notebook, bundle).

## Owner, 14 Nov 23:48 UTC (DECISION_LOG.md P31, commit 4c94c68)

"Please remove any budget concerns at this point and just document the prices. I think explore the jetson nano
hardware is a good exercise even with the extra cost." Money budget only (P4 is not applied as a gate now; no budget
decision, no trim options; levers priced for the record). Compute budgets (core-hours, Colab hours) are unchanged.
Decision 11 of the 14 answers ("budget decided after the rebuild") is replaced by this.
- Workflow trex-budget-to-prices (run wf_04197411-307, worktree /home/user/mesozoic-wt/budget-prices, branch
  wt/budget-prices): BUDGET_75.md becomes a price record. Merge when it commits. The Jetson-body branch also
  regenerates budget_75.json (hash fields): on a merge conflict keep wt/budget-prices' script and rerun it.
- Owner page later: prices (not "budget") and the bench sheet.

## Environment verified (15 Nov 00:29 UTC)

- restore_inputs.py restore finished (about 3 h beside other jobs); `check`: 283 of 283 files match their sha256
  (picks-models 6, picks-arrays 84, k 60, x 60, look 73), exit 0.
- Full suite at e0ad284: "679 passed, 107 skipped in 737.97s", PYTEST_EXIT=0 = the fresh-clone 599 + PR #668's 9 +
  71 new tests of stage2/jetson75 (35 latency and bench, 17 prices, 19 ToF). Skips: 73 STAGE2_SLOW, 15 STAGE1_SLOW,
  18 policy rows, 1 LOOK_MEDIA, as tools/README.md lists.
- Budget document is a price record (P31), merged e0ad284.

## Pre-registration status (15 Nov 00:45 UTC)

STAGE2_75_PLAN.md and STAGE2_PLAN.md section 9 committed on wt/prereg as 0aea923 (43 review findings). Twelve points
were left for the lead; rulings in design/PREREG_RULINGS_15NOV.md; workflow trex-prereg-settle (run wf_a26a50bb-c5f)
writes them into the texts. Merge wt/prereg into the working branch and push as soon as it commits: it must precede
any frozen set and any Stage 2 closed-loop episode. WP_B, WP_S, WP_T and WP_C carry a lead addendum (top of each
brief) with what the rulings change for the build.

## 15 Nov 00:42 UTC

- L1 freeze record committed and pushed (90b0097): stage2/look/L1_FREEZE.json, designlab/stage2/l1freeze.py,
  tests/test_stage2_l1freeze.py (5 tests). V6 windows hold 9-14 strides (variation); 75 cm fr05 rung out of the ladder.
- Round C design workflow launched: trex-round-c-design, run wf_33e1ef4d-c03; outputs design/ROUND_C_DESIGN.md,
  design/WPC_*.md, design/PREREG_TRACKB_draft.md.
- Open hardware question carried by V6's flags: look feature F5 rolls onto the middle claw's tip, where the plant
  has no contact (look.CLAW_QUESTION): tell the owner with the V6 set's screen.

## Pre-registration merged (15 Nov 01:24 UTC, commit 89111c3, pushed)

STAGE2_75_PLAN.md and STAGE2_PLAN.md section 9 (S2-9 to S2-18) are on the working branch, with the twelve settled
points and nine confirmations (design/PREREG_RULINGS_15NOV.md; the plan's closing sections). No set.json and no
Stage 2 episode existed at that commit. WP_B.md's smoke now uses R0 rows on V0, one episode per condition.

## Jetson body merged (15 Nov 02:05 UTC, merge 1aefbd0)

75-M-H-10-J (refine case jetson_75): +105.3 g, CoM +1.64 mm forward; every round-two screen passes; runtime 32.9 /
29.5 min (15 / 25 W) against the Pi body's 36.7; G2, G3 pass; G1 PROVISIONAL (19,837 steps/s per process on 2 of 4
cores, under the 20,000 rule): run `PY=.venv/bin/python bash stage2/j/run_j.sh g1` on an idle machine, with -X's.
Inputs copied into the main checkout (30 files, check ok). budget_75.json regenerated: no change beyond the merge.
The merge moved the panel source hash (6a316fc8 -> 4583a187): baselines (legacy check, regression record) must be
taken at state A. Restore group `j`: handed to WP S (addendum). Draft A19 (mode power budgets) is in
stage2/j/JETSON_BODY.md section 10: propose to the owner. Test-order trap: tests/test_stage2_x_costs.py line 92
leaves the gait accuracy set; test_carpet's round-one test fails if run after it (fix: restore the accuracy).
Not done: paired-seed panels J against Pi (after round A; pre-register as an added tier).

## Round C launched (15 Nov 02:50 UTC)

Round C design final: design/ROUND_C_DESIGN.md (192 KB), briefs design/WPC_{CP,CV,CB,CK,CE,CT,CW,CF,CG,CN}.md, rule
draft design/PREREG_TRACKB_draft.md; the lead's rulings design/TRACKB_RULINGS_15NOV.md (also the look-in-moves
choices: F5 in periodic walks only, flat feet in every move; J1 with fallback order; sets and their order).
Round C edits NO existing file (trainer = designlab/stage2/train2.py; look reading = lookread.py + lookbar.py).
Workflows running (scripts under .../workflows/scripts/, resume with resumeFromRunId):
- trex-wp-l-look-sets       run wf_83a16354-c85   worktree /home/user/mesozoic-wt/wp-l (wt/wp-l): moves_look.py, V6 sets
- trex-prereg-trackb        run wf_67ae72ba-51b   worktree /home/user/mesozoic-wt/prereg-b: STAGE2_TRACKB_PLAN.md
- trex-round-c-build-1      run wf_bb1d63d5-059   worktrees /home/user/mesozoic-wt/wpc-{cp,cv,cb,ck,ce}
- trex-round-a-build        run wf_4b831ec4-af2   (still running: C build, M review; S, T queued)
- trex-owner-pages-jetson   run wf_a6d21355-4b4   pages under scratchpad/pages/ (bench_sheet.html, prices_75.html)
Still to launch: round C waves 3-4 (CT, CW, CF after CE + round A merged; CG, CN last).
Order of the remaining lead steps: merge round A (M, R, G, C, S, B, T) -> full suite -> legacy check + regression at
state A -> build + screen + freeze the three 75 cm sets -> COST.md -> task 3 tiers -> task 4 tiers -> report.
Publish owner pages when their workflow returns. G1 for -X and -J when the machine is idle.

## Owner pages published (15 Nov 03:10 UTC; private artifacts)

- Delay Bench Sheet: https://claude.ai/artifact/LePsAx8XM8q8hUTpZ3GqtJ  (source stage2/jetson75/BENCH_SHEET.md at 760b9f1;
  build: sh scratchpad/pages/bench/build.sh; file scratchpad/pages/bench_sheet.html, sha256 b6df2cd5...)
- 75 cm Build Prices: https://claude.ai/artifact/KDSuT1A1AsHad4KVVrzsfa  (source stage2/jetson75/BUDGET_75.md at 760b9f1;
  build: sh scratchpad/pages/prices/build.sh; file scratchpad/pages/prices_75.html, sha256 469534e4...)
Both reviewed independently (faithful to the reports word for word; contract; phone layout 320-1280 px; both themes).
To add to LESSONS_LEARNED.md section 8 at the end-of-session update. Kit follow-ups: page_kit.py's three literal
colours need tokens; shot_page.js omits the publisher's zero body margin.

## Owner, 15 Nov ~03:15 UTC (DECISION_LOG.md P32, commit f3a87c4, pushed)

"So full disclosure, I probably won't do the bench delay testing, not until I have a working 75 centimeter build
working in simulation." Reading: the bench test waits for a working 75 cm build in simulation; nothing waits on it;
the bar stays judged at the assumed 10 ms; the compute choice waits with the bench. No rule changed. Do NOT spend
effort on the bench sheet's two unwritten test programs. The bench page was rebuilt with the dated note and
republished at the same URL (scratchpad/pages/bench/make_copy.py patched for the extra paragraph).
Consequence for emphasis: the simulation result on the 75 cm build is now the gate to any hardware step; report the
0-8 ms sensitivity clearly, since it says what delay the hardware would have to reach.

## Owner, 15 Nov ~03:25 UTC (DECISION_LOG.md P33, pushed)

"My primary focus is finding the correct configuration, build size, parts, and policy that can handle the desired
behaviors I'm looking for in simulation. And then that's when I will start actually moving on to the physical building
of the robot." Consequences for the lead:
- The end product is a recommended build with evidence: size, body and head, parts, controller, against the bar.
- A limit that a screen or a panel exposes (tail servo peak in the quick turn; thigh-twist drive speed, knee and
  hip-pitch speed at 60 cm; thigh-twist range; knee speed at the animal pace) is a prompt for a model-only hardware
  lever (plan Track D), not a stopping point: plan the levers as soon as the 75 cm sets' screens and panels show
  which limits bind. Levers already in the plan: (1) hip-pitch and knee speed x1.25 / the faster knee; (2) STS3250
  tail servo; (3) bigger 60 cm feet; (4) joint damping and friction x0.25 on tail, neck yaw, jaw; (5) -X neck parts;
  (6) thigh-twist range beyond +-0.30 rad. Not gated on cost (P31).
- The pre-registered reference rules stay as they are (no constant changed to rescue a reference); a lever is a new,
  labelled body or part variant with its own screens and records.
