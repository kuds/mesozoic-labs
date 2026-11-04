# Walking pilot 3, FINALISE stage: every change against final/, and what stays open

2026-11-03 (UTC), 19:42-20:02 (FIX2 correction: the stage's last command ran at 20:01:56 and its files were last
written at 20:01-20:02; the end time "20:20" first written here was set in advance). Folder:
SP/work/pilot3/frozen_candidate ("FC"); "SP" is the session scratchpad.

FIX2 (a later stage, 2026-11-03 from 20:33 UTC; its record is FIXES2.md beside this file, a copy of
SP/work/pilot3/fix2/FIXES.md): open item 1 below is resolved (pilot3_vg.py and pilot3_noise.py built, tested and
pinned); pilot3_judge.py, pilot3_score.py, pilot3_habit.py and their tests were fixed and re-pinned, so the sha256s in
the table below for items 3, 4, 6 and 7 are the FINALISE stage's, not the current ones (the criteria give the current
ones); items 5 and 6 of the open items are recorded in the criteria's decision (e); the pending list was extended.
FIX3 (2026-11-03 from 21:40 UTC; its record is FIXES3.md beside this file, a copy of SP/work/pilot3/fix3/FIXES.md):
after a third code review, pilot3_habit.py, pilot3_score.py, pilot3_vg.py, pilot3_noise.py and their tests were
changed and re-pinned (the criteria give the current sha256s); no new open item.
AMEND (2026-11-04 from 16:01 UTC; its record is CHANGES2.md beside this file): the maintainer's decisions (8)-(11) of
2026-11-04 (the -2.0 reset, the walking reward R6, the gate and the inputs unchanged) and the branch's commits 4-6
(head a3f1b378) are written into the criteria, BRANCH_PLAN.md and NOTES.md; pilot3_judge.py, pilot3_noise.py and
their tests are re-pinned for -2.0. The head, the reset value and the recipe and task digests quoted below are the
FINALISE stage's (3ae1c959, -1.5, 8263f82f..., 5f835d65...); the criteria give the current ones.
This file is otherwise the FINALISE stage's record, unchanged.
No training, no simulation, no git write, no push, no PR, no Drive access; nothing written outside FC. Labels:
**[measured: source]** = read or computed from the named file; **[inferred: basis]** = reasoning. Commands are in
FC/CMDLOG.txt.

## Inputs and outputs

Inputs, copied byte for byte into FC (and into FC/base/ as an untouched copy) [measured: sha256sum, CMDLOG
19:43:55Z, FC/out/copy_sha.txt]:

| File | sha256 of final/ (input) |
|---|---|
| criteria_pilot3.txt (603 lines) | `ec74b4c31b1e7786a64caeed72489cf8514c12c23b1054c962e69dfc8bc2bfd0` |
| BRANCH_PLAN.md | `a812cbd5f1980dca4c5cc39dcfbe10ba0c183f63c8270154e5cd09b11e9455e3` |
| NOTES.md | `84af2a92a834a35edcf2a505b980fa73e1f9f97518d128ade94d8166bf5fcb2e` |

Outputs: FC/criteria_pilot3.txt, FC/BRANCH_PLAN.md, FC/NOTES.md (their sha256s and line counts are at the end of
CMDLOG.txt, written after the last edit); the diffs against final/ are FC/diffs/<file>.diff (`diff -u final/<file>
FC/<file>`). FC/bin holds the helpers used for the edits (lr, the command logger; reflow.py; edit_plan_table.py;
edit_notes.py).

Sources of the fills [measured, each read in this stage]:
- the worktree SP/wt_p3 (git log, rev-parse, status; read only);
- SP/work/pilot3/branch/ (out/c1, c2, c3: suites, quick checks, digest diff, deselect_ids.txt; commit messages);
- SP/work/pilot3/tools/ (SHA256SUMS.txt re-checked by `sha256sum -c`: 29 of 29 OK, FC/out/tools_sha256_check.txt;
  out/final/: pytest.log.txt, test_pilot3_score.log.txt, habit_pc_validation.txt, habit_trace_parent.txt, rcs.txt;
  out/test_pilot_score_pilot1_baseline.log.txt; CMDLOG.txt);
- SP/work/pilot3/smoke/ (out/*.txt and *.json, nb/, judge/logs/, probe_parent/, scripted/, replay/out2/, videos/);
- SP/work/run_log.txt lines 997, 1003 and 1005 (the Option B squash advice, PR #658's merge, the final decisions);
- notebooks/sb3_training.ipynb and environments/shared/evaluation.py at 3ae1c959 (git show / git grep, read only).

## The maintainer's decisions applied (all three files)

The decision line: SP/work/run_log.txt line 1005, "2026-11-03 14:53 UTC SESSION NOTE MAINTAINER DECISIONS
(AskUserQuestion) on pilot 3": budget "8M (Recommended)"; futility stop "No early stop" (F0); habit rule "Keep as
written (Recommended)"; remaining defaults "Accept all (Recommended)". It matches the decisions in this stage's task
text. Recorded in the criteria as decision (e).

D1. **The 4,194,304 variant deleted** from the criteria: the status paragraph names it as deleted; C6, C7 (pilot 1's
524,288 kept only as pilot 1's fact); "Code" (commit 3 "(4.2M only) the regenerated species catalog" removed, the
parent commit renumbered 4 -> 3); "What is trained" (`(524288 for 4.2M)`, `(16 at 8M)` -> `(16)`, the "4.2M only:
timesteps = 4194304 and learning_rate_end" sentence); "Budget" (the 4,194,304 paragraph; "not the 4.2M variant's
learning-rate path"); "How it is judged" (the 4.2M judged checkpoints; the fallback's "4.2M variant 4,194,304");
"Video review" ("at 8M"); "A seed's outcome" ("4.2M variant: 4,194,304"; pilot 1's 524,288-1,572,864 off-schedule
range); "Run handling" ("(or 4,194,304)", "(4.2M: 524288)"); "Habit" ("at 8M"); "Pre-launch checks" ("4,194,304 ->
4,194,304", "commits 2-4" -> "commits 2-3"). Kept, because they are facts about pilot 1, not this pilot's variant:
pilot 1's budget, pace and trajectory checkpoints (2,097,152, 3,145,728, 4,194,304) and "Cost against pilot 1's
4,194,304". In BRANCH_PLAN.md: the 4.2M keys in commit 2's table, section 3 (the catalog commit) replaced by a
measured note that no catalog commit exists, the parent commit renumbered 3, `PILOT_BUDGET 8,000,000 (or
4,194,304)`, "build commits 1, 2 and 4 (3 only for 4.2M)". In NOTES.md: I3, I4, I21 and decision item 1.

D2. **Futility stop F1 deleted; F0 stated as decided**: the criteria's status paragraph, C6, "Budget" (F0/F1 block
replaced by "Futility stop: none (F0, decision (e))"), "How it is judged" ("except 4,000,000 under futility stop F1"),
"Run handling" ("except futility stop F1 if the maintainer chooses it" -> "there is no futility stop (F0, decision
(e))"), "Habit" ("futility stop F1: through 4,000,000"), the exception text ("8,000,000 steps with no futility
stop"). NOTES.md: I26 emptied, decision item 3. BRANCH_PLAN.md: order of work step 5.

D3. **The other decisions marked**: the criteria's decision (e) lists each; "Run handling" now says the launch-fault
reading was confirmed by the maintainer (it said "the maintainer confirms this reading before freezing"); the habit
window cites decision (e). NOTES.md section 4 marks items 1-8 and 10-13 DECIDED, item 9 (the decision number) still
open. Item 12 (a judge refusal for a binding check restarts the seed) is not named in the run log's list; I read it
as covered by "remaining defaults: Accept all" [inferred], and NOTES says so.

## The <TO FILL> items (final/ had 27 by `grep -o "<TO FILL[^>]*>"`; 28 occurrences of "<TO FILL", because one item
## spans a line break)

Two occurrences are the status paragraph's own description of the rule (lines 4-5); they stay, reworded. The 26
items, each with what now stands in its place (the line numbers are final/'s):

| # | final/ line | Item | Filled with | Source [measured] |
|---|---|---|---|---|
| 1 | 86 | merge commit | `4d5f4e76232d6e19a4526e2945c28228d561f30d`, a merge (parents 4c9d5e9b, cab70394), tree dd9d6852 = cab70394's tree; the wording "the squash merge" corrected | git log, rev-parse in SP/wt_p3; run log line 1003 |
| 2 | 91 | head | `3ae1c95967547ada569ad71aebaa88999afa02cd`; commits 1-3 with full shas; no catalog commit | git log in SP/wt_p3; branch out/c{1,2,3}/quick.txt |
| 3 | 206 | pilot3_judge.py sha256 | `e962c289...` (+ tests/test_pilot3_judge.py, tests/_tree_facts.py) | tools SHA256SUMS.txt, re-checked |
| 4 | 214 | pilot3_score.py sha256 | `0529eaef...` | same |
| 5 | 216 | iso.py, pilot3_video.py | `5da55763...`, `43f72bfa...` (+ tests/test_pilot3_video.py) | same |
| 6 | 218 | adapted test and fixtures | test_pilot3_score.py `8a6645de...`, fixtures `24f51b51...`, make_gait_cases.py `8689bf02...`; 297/297 (pilot 1's 254 + 43 new); pilot 1's unchanged test 254/254 | same; tools out/final/test_pilot3_score.log.txt; out/test_pilot_score_pilot1_baseline.log.txt |
| 7 | 220 | pilot3_habit.py | `1ade1d9b...` (+ its tests and validate_pc_ledgers.py) | same |
| 8 | 222 | pilot3_vg.py, the noise readout | **OPEN** (open item 1) | no such file on this machine (find, CMDLOG) |
| 9 | 226 | nb_pilot_runner3.py sha256 | `7f533128...`; the smoke recorded the same | tools SHA256SUMS.txt; smoke out/smoke_evidence_launch_s42.json |
| 10 | 283 | final replay reset seed | seed + 2000 + stage 2: 2044 for s42 measured; 2054, 2062 for s52, s60 inferred from the code | smoke replay/out2/replay_check.json; evaluation.py replay_seed at 3ae1c959 |
| 11 | 336 | head in the "Repository ref" line | the full head; the line's format read from notebook cell 3 (only Colab prints it) | notebook at 3ae1c959 |
| 12 | 345 | hyperparameters_sha256 | `sha256:8263f82f6783cfcddd366a25ea884e1039516fd4c161e30689dceaa552136919` | branch out/c2/digest_snapshot.diff, hp_check.txt; smoke stage_config check |
| 13 | 345 | task_sha256 | `sha256:5f835d6588151a5d100491f1b255be1365883c6ca4913a144c9f037800d91c84` | same |
| 14 | 521 | digest lines and the generated-file checks | the four moved lines with old and new values; task and gate unchanged; checks current at each commit | branch out/c2/*, out/c{1,2,3}/quick.txt |
| 15 | 523 | the known failing set | the 14 ids, each with its cause; each passed at commit 1 and failed at commit 2 (re-verified in this stage) | branch out/c2/deselect_ids.txt, c1/shared.txt, c2/shared.txt; commit 2's message |
| 16 | 524 | commit 1 with nothing deselected | 6528 passed, 1 skipped, 0 failed (7 off-path tests passed); T. rex 174 passed, 4 xfailed; test_stage_cadence 15 passed; commit 3: 6514 passed, 14 deselected | branch out/c1, out/c3 |
| 17 | 532 | the smoke's run | host, times, clone, venv, runner sha256, cells, guard line | smoke nb/params_s42.json, nb/launch_s42/record.json |
| 18 | 547 | the smoke's expected outputs | Warm start lines, stage_config 17/17, TensorBoard, Eval line, ledger, final replay | smoke out/smoke_evidence_launch_s42.json etc. |
| 19 | 550 | smoke RESUME | stopped at 32,768 (exit 75); RESUME cell exit 0, no Warm start line, records kept, resume_load_path | smoke out/rtest_resume_stage_config_check.txt etc. |
| 20 | 558 | judge validation | J1-J10 with exit codes and messages | smoke judge/logs/ |
| 21 | 560 | judge unit tests for the on-schedule resume branch | 69 of 69, the resume and fallback cases listed | tools out/final/pytest.log.txt |
| 22 | 562 | probe on the parent | 40/40 STILL, L 0.8766 m, r10; launch not held | smoke probe_parent/, out/probe_parent_labels.txt |
| 23 | 565 | scripted r10 rollout | run; touchdowns and lift-offs on both feet, weak (1-2 events per episode, every episode fell) | smoke scripted/*.log.txt |
| 24 | 566 | video equality check | 40/40 match; the two named episodes rendered | smoke videos/s42_65536/videos.json |
| 25 | 575 | habit validation | 30/30 tests; PC ledgers 0 mismatches; parent trace (no PARK, no DRIVEN) | tools out/final/habit_*.txt |
| 26 | 576 | tool sha256s | pointer to the list; 29/29 OK in this stage; 17/17 at the smoke's start and end | FC/out/tools_sha256_check.txt; smoke out/tools_sha256_check_*.txt |

Item 26's line also lists, as "Pending", the steps after this stage (open item 2).

## Other edits to the criteria (not <TO FILL> items, made current by the measurements)

E1. Title and status: DRAFT -> FROZEN CANDIDATE; the status paragraph says which items are open and that the
unchosen options are deleted.
E2. Decision (d): "training waits for PR #658 to merge" now gives the merge (14:19 UTC, 4d5f4e7); C3 "once #658
merges" -> "since #658 merged (4d5f4e7)".
E3. "What is trained": the conditional "If #658's merge commit differs from cab70394 ... the two lines are
re-measured" is resolved: the merge's tree equals cab70394's, train_base.py and critic_phase_ppo.py are
byte-identical, and the smoke printed both lines string-equal; no amendment of the two lines.
E4. "What is trained", TensorBoard: the smoke's measurement added; "no override is planned" -> "No override is
added".
E5. "How it is judged": where the tools are and when they were pinned.
E6. "The probe and fixtures": "Not checkable here" and "What a real rollout has exercised ... nothing yet on r10"
rewritten in the past tense with the measured results (parent and child 40/40 STILL, L 0.8766 m; the scripted
rollout exercised the touchdown and lift-off path weakly).
E7. "Run handling": the resume's records cite the smoke RESUME too; the launch-fault reading cites decision (e).
E8. "Known limits": the WALK-path limit now states the scripted rollout's weak result; "whether a resume reproduces an
unbroken run is untested" -> the CPU measurement (bitwise equal at 32,768 before the stop, not equal at 65,536 after
the RESUME), with an inferred note that a resumed seed is likely not the unbroken run.
E9. "Pre-launch checks", last line: PR #658 merged and the branch's base measured; the pending steps listed in order.
E10. Exception text: the decision times add 14:53; `<budget>` -> 8,000,000 with no futility stop; `<head>` ->
3ae1c959 on 4d5f4e7. Its `(sha256 <...>)` stays: it is this file's own sha256 after the freeze, which cannot be
written inside the file; the later records PR fills it (not a <TO FILL> of this file). `D-D<next>` stays (NOTES item
9).
E11. Three paragraphs reflowed to 120 columns after the edits (bin/reflow.py: whitespace only). The only lines over
120 columns are the two frozen Warm start lines, as in final/. The file stays ASCII.

## Other edits to BRANCH_PLAN.md and NOTES.md

B1. BRANCH_PLAN: header and opening paragraph say what was built (the three shas); "Base" corrected from "the squash
merge" to the measured merge commit, with the consequence stated (the 13 PR commits, of which 4-9 fail one
identifier test alone, are now in main's history: a bisecting concern for main, none for the branch, whose base tree
is cab70394's); commit 1 notes how it was built; commit 2 adds the measured 14-id set (10 of the ids read in the
plan, plus 4 it did not name); "Possible 5" records that the smoke made it unnecessary; the tool table's Status
column now gives each tool's measured state and sha256 prefix (the V - G and noise readouts: not built).
N1. NOTES: a FINALISE paragraph at the top; "(FINALISE: measured since ...)" on I8, I20, I22, I24, I29, the "Not run"
paragraph of section 2 and two items of section 5; I21's basis restated without the deleted variant's range.

## Open items (visible in the text, for the maintainer)

1. (RESOLVED at FIX2: both built as pilot3_vg.py and pilot3_noise.py, tested and pinned before launch; FIXES2.md.)
   **pilot3_vg.py and the noise readout are not built** (criteria, "How it is judged", the one remaining
   `<TO FILL ...>`). The criteria pin these report-only tools "before the first judged checkpoint is read", which is
   after launch, while the status paragraph says no seed launches while any `<TO FILL>` remains, and a later fill
   would change the frozen sha256. The two rules cannot both hold as written. Ways out, for the maintainer (no
   recommendation): build and pin both tools before the freeze; or reword the item so their sha256s are recorded in a
   dated note beside the frozen file before the first judged checkpoint is read (the text then holds no `<TO FILL>`).
2. **Pending after this stage** (criteria, end of "Pre-launch checks"): the maintainer's approval of the frozen text
   (its sha256); the maintainer's OK to push and the push (the remote branch must read 3ae1c959...); the Colab steps
   (pilot 2's colab_steps.txt as the model); the launch approval. The head is local only; nothing was pushed.
3. **Not checkable locally** (written so in the criteria): the "Repository ref: ...; commit: ..." line (only the Colab
   branch of notebook cell 3 prints it), the /tmp/tb_buffer TensorBoard branch, real CUDA, cuDNN and driver values in
   provenance.json. The first hour of each run reads them.
4. **The scripted rollout is weak evidence**: the touchdown and lift-off path is exercised on r10, but alternation and
   the simultaneous fraction each rest on one event pair or one event, and no closed-loop or longer pattern was
   tried. The criteria say so in three places; the maintainer may want more before launch.
5. **The decision number** `D-D<next>` and the exception text's `(sha256 <...>)` are for the records PR.
6. **A wording difference found, not resolved**: the run log's line 1005 records the futility choice as "No early
   stop" "(against the recommendation; F0: always train to 8M)", while the criteria (FIX stage) presented F0 and F1
   with no recommendation. The decision itself (F0) is unambiguous; the criteria quote it without that parenthesis.
7. **PR #658 was merged as a merge commit, not the squash the Option B review advised** [measured: git log; run log
   line 997]. Nothing in the pilot depends on it (the base tree is cab70394's); recorded in BRANCH_PLAN for main's
   history.

## Checks run in this stage

- Tool sha256s: `sha256sum -c SP/work/pilot3/tools/SHA256SUMS.txt`, 29 of 29 OK (FC/out/tools_sha256_check.txt).
- The 14 deselected ids: each "PASSED" in branch out/c1/shared.txt and "FAILED" in out/c2/shared.txt (14 of 14
  each); 7 test_critic_phase_off_path tests passed at commit 1; the 4 T. rex xfails are test_com_stance_quality.py's.
- SP/wt_p3: `git status --porcelain --ignored` empty; HEAD 3ae1c959; commits chained 4d5f4e7 <- f2399c8 <- 76ceae9 <-
  3ae1c959; trees of 4d5f4e7 and cab70394 both dd9d6852.
- The criteria: no `<TO FILL` left but the one open item and the status paragraph's description; ASCII only; no
  line over 120 columns but the two frozen Warm start lines.
- Every line I added is in FC/diffs/; I read each diff in full after the last edit.

## Rule slips (this stage)

- CMDLOG is not fully literal before 19:43:29Z: the first commands (run before the logging helper existed) are copied
  by hand with their full scratchpad path shortened to ".../pilot3" or "SP" in the log, and only the first carries a
  measured time. Three helper scripts (bin/lr, bin/edit_plan_table.py, bin/edit_notes.py) were written with shell
  heredocs outside the logger, and two (bin/reflow.py, this file) with the file-writing tool; edits of the three
  output files were made with the file-editing tool. Each is listed in words in CMDLOG.
- A model-name or device-path search was not run on the outputs, because spelling either in a pattern is itself
  against the rules. Instead every added line was read in the diffs; none names a model or a device path.
- My first `git status --short --branch` in SP/wt_p3 (about 19:42 UTC) ran without GIT_OPTIONAL_LOCKS=0, so git was
  allowed to refresh the worktree's index, which lives under /home/user/mesozoic-labs/.git/worktrees/wt_p3/. It did
  not: that index's mtime is still 16:59:02 UTC, from the branch stage [measured: stat, CMDLOG 20:01:43Z]. The later
  status calls used GIT_OPTIONAL_LOCKS=0; every other git command was a read (log, rev-parse, show, grep, rev-list).
- No other slip known: no training or simulation, no write to SP/wt_p3 or /home/user/mesozoic-labs, no push, no PR,
  TMPDIR inside FC.
