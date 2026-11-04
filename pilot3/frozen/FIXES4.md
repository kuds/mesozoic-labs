# Walking pilot 3, FIX4 stage: what each finding of the R6 code and text reviews became

2026-11-04 (UTC), 17:27 to about 19:00. Stage folder: SP/work/pilot3/r6_fix ("RF"). "SP" is the session scratchpad
`/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad`; "FC2" is
SP/work/pilot3/frozen_candidate2 and "T" is SP/work/pilot3/tools. This file is also copied byte for byte to
FC2/FIXES4.md, which the criteria name ("beside this file").

Labels: **[measured: source]** means read or computed from the named file or a command of this stage.
**[inferred: basis]** means reasoning. The literal commands and their UTC times are in RF/CMDLOG.txt (the
exceptions are under "Slips").

## Summary

- **All three blocking findings are fixed in the text** (the HOPS reading and NOTES I33; the R6 limits list; the two
  wrong line citations). Every minor finding that was clearly right is applied; the two that are maintainer choices
  (the behavior recipes' inheritance of R6; the deliberate k+4 and touch-sensor departures) are now named explicitly
  for the approval rather than changed. Table below.
- **Two new pilot-only commits on the local branch** `claude/pilot-walk-r10-p3` (never pushed, no PR, no git config
  change):
  - 7. `99859224b41bf666f0b3587ea4b812ab638ff13d` "Test R6's touchdown event across episodes on one env (walking
    pilot 3, pilot-only, never merged)": tests only.
  - 8. `31d02e42d148511e50f9cba9a598d9a9d36c23af` "Credit the forward_vel_max cap finding to the config-only screen
    (walking pilot 3, pilot-only, never merged)": two comment lines of locomotion.toml.
  - **Branch head: `31d02e42d148511e50f9cba9a598d9a9d36c23af`** (tree `17ec0cb9`), parent chain 4d5f4e7 <- f2399c8 <-
    76ceae9 <- 3ae1c959 <- b7a146d <- 4dfc7cb <- a3f1b378 <- 99859224 <- 31d02e42; `status --porcelain --ignored`
    empty [measured: RF/out/wt_p3_state.txt].
- **Stance behaviour unchanged; no physics change.** Neither commit changes code or a config value: commit 7 touches
  only `environments/tyrannosaurus_rex/tests/test_anatomical_walk_r6.py`, commit 8 only lines 17-18 of
  `configs/tyrannosaurus_rex/anatomical/locomotion.toml` (parsed TOML equal to a3f1b378's, 204 lines before and
  after); digest_snapshot --check current with no line moved at both commits [measured: RF/out/c7/quick.txt,
  RF/out/c8/quick.txt].
- **Re-smoked at the new head.** The notebook smoke, the RESUME and the judge's ten validation steps re-ran at
  31d02e42; the training is bitwise the re-smoke's at a3f1b378 (policies, optimizers, VecNormalize, ledger,
  TensorBoard scalars, Eval lines, final replay mp4) [measured: RF/out/resmoke_checks.txt]. The tools that had never
  run in the head's R6 env (parent probe, scripted rollout, habit seed trace, V - G, video tool, replay byte check)
  ran too; only the held-out seeds and the e2e did not.
- **Re-pinned.** No tool file changed; T/SHA256SUMS.txt gains the FIX4 header, the two corrected section comments and
  16 evidence lines; `sha256sum -c` 105/105 OK [measured: RF/out/tools_sha256_check_after_repin.txt].

**Final values:**

| Item | Value |
|---|---|
| criteria_pilot3.txt sha256 | `853f49d6b7846e29e820de5e6572d36fc9f6b2c20b83c178e46f9a62eb291df4` (1438 lines; input `93f06851...`, 1284 lines) |
| T/SHA256SUMS.txt sha256 | `3c904ececf8280c7990f0bb6667a8ad5cf451b4978885902f505d15cccd22abd` (105 entries; input `f90a89b4...`, 89) |
| branch head | `31d02e42d148511e50f9cba9a598d9a9d36c23af` (local only) |
| NOTES.md sha256 | `7d6b09878d537e2cd94f67ccc523cfa6d30509fed51302db9d4cbd8ce0c41090` (268 lines) |
| BRANCH_PLAN.md sha256 | `5505f1a48cc01d7aca5aabcdd329777239cca63ea1103483cf5510ed496a1943` (347 lines) |

The inputs are kept in RF/before/ (fc2/criteria_pilot3.txt, NOTES.md, BRANCH_PLAN.md, CHANGES2.md; SHA256SUMS.txt;
the test file and locomotion.toml at a3f1b378); the diffs are RF/out/criteria_pilot3.txt.diff, NOTES.md.diff,
BRANCH_PLAN.md.diff and SHA256SUMS.txt.diff. CHANGES2.md is left as the AMEND stage's record.

## The findings

### Code review (r6_review_code; all minor)

| # | Finding | Disposition |
|---|---|---|
| C1 | Test gap: mutant T17 (no `_walk_touchdowns.reset()` in `_spawn_target`) survived all 26 R6 tests | **FIXED (commit 7).** New `test_a_reset_starts_the_touchdown_event_afresh_on_the_same_env`; see note 1. |
| C2 | Two deliberate departures from the approved rule and SCREEN's: pay at k+4, not k+3; down flag from the touch sensors, not floor truth; the approval should name both | **FIXED in the text:** criteria decision (9) now lists both as (i) and (ii) for explicit approval, with the agreement figures (0 of 639,346; 8 of 1,303,934); the pending list asks approval of "the three departures named under decision (9)". Code unchanged (both are deliberate and disclosed). |
| C3 | Commit b7a146d's message: the ledger sentence is imprecise; the down rule's wording hides that it replaces floor truth | **FIXED in commit 7's message** ("Corrections to b7a146d's message"), since history is not rewritten; the criteria already said the ledger has no reward-term column; NOTES' records-PR note points to the correction. |
| C4 | The 11 T. rex behavior recipes inherit R6; accept, or pin the four keys off in their [env] | **NOT CHANGED; named for the approval** as decision (9) (iii) (the maintainer's choice: accept, or a branch change and a new head). Pinning them off is a choice of scope, not clearly right, and the pilot trains none of them [inferred]. |
| C5 | `test_off_r6_is_the_branch_head_bitwise` skips outside the recording environment | **DISCLOSED in the criteria** (the R6 tests paragraph: it reports SKIPPED on Colab or CI, where only the head-against-head off-path tests run). Not changed: a bitwise golden binds its environment [inferred]. |
| C6 | Thin test coverage of the env's live R6 path; the integration test's exact counts (1 ALT pay, 0 HOP pays) are brittle | **FIXED in part (commit 7):** the new test rolls three live episodes on one env; the integration test's checks now assert each pattern's purpose (ALT: at least one paid alternating touchdown; HOP: a landing of both feet within 2 steps, neither foot paid) instead of exact counts. The criteria say what the repository's live coverage is and that VALIDATE's 320 and the review's 15 live episodes are outside it. |
| C7 | Two survivors (T22, T24) are equivalent mutants | **NO CHANGE NEEDED;** recorded in the criteria's tests paragraph. |
| C8 | Are pilot3_judge.py:85 and pilot3_noise.py:39 (now -2.0) re-pinned? | **CHECKED:** `sha256sum -c` 89/89 OK at the re-smoke's start and end and before the re-pin (df5ea214... and c191c72a... are the pinned values) [measured: RF/out/tools_sha256_check_start.txt, _end.txt, _before_repin.txt]. |

### Text and process review (r6_review_text)

| # | Finding | Disposition |
|---|---|---|
| B1 | **BLOCKING.** The HOPS reading (criteria lines 691-697) and NOTES I33 are false about R6 at forward weight 2.0 | **FIXED.** HOPS rewritten: R6 puts every hand-built hop below its matched walk at both ends of the ramp, below standing only at weight 0.2; at 2.0 (from 2,000,000 steps on) every hop at 1.0 m/s or faster out-earns standing, so such a HOPS agrees with SCREEN's ranking (PPO found no walk it kept); a slower hop would say the ranking did not carry over. I33 rewritten the same way. See note 2. |
| B2 | **BLOCKING.** The R6 limits list (lines 292-293) leaves this out | **FIXED.** A new first limit with the review's figures, recomputed here (H24 +135.3, 20/20; HF120 +299.0, 20/20; H20 +521.2, 19/20; H25 1491.8 and HF150 1519.9 against about 1092), the slow hops below standing, every hop 862-911 below standing at 0.2, the crossing weights; a second limit: H20 beats FINAL3 and V01, HF120 beats FINAL3 on 1 of 20 seeds (FINAL3 ahead by 33.4), H25 and HF150 out-earn FINAL3 (1424.9), and "a policy that finds no walk is pushed, from about 1.3-1.7M steps on, toward a fast hop rather than toward standing [inferred]". Also: the seed-outcome note and "Known limits" mention it. See note 2. |
| B3 | **BLOCKING.** Two wrong citations: the statue line (line 597) and the ramp lines (line 822) | **FIXED.** "locomotion.toml line 149 at a3f1b378 and at the head, line 134 at 3ae1c959, line 122 on main 4d5f4e7 and at PR #658's head cab70394"; "lines 6 and 188-189 at the head (173-174 at 3ae1c959)" [measured: git show of the four versions, RF/CMDLOG.txt]. |
| T1 | Heading "never trained" | **FIXED:** "untested in training: only the 65,536-step CPU smokes ran on it" (heading and "Known limits"). |
| T2 | R5p/R6p was a post-hoc SCREEN extra | **FIXED:** one clause in the SCREEN paragraph (added after its consistency checks found the causal detector missed landings after a one-step bounce, SCREEN slip 7; constants fixed beforehand; the pre-declared R6 also passed on all 40 seeds). |
| T3 | The k+4 and touch-sensor departures are not listed with decision (9) | **FIXED** (as C2). |
| T4 | Decision (8): 359/360 over all panels against 9/120 on one panel | **FIXED:** both bases given (120/120 against 9/120 on 3042-3081 in three streams; pooled 359/360 against 16/240, the morning report's table). |
| T5 | 1083.3 (40 seeds) against 1094.4 (20 seeds) | **FIXED:** "1094.4, a 20-seed mean on 3042-3061: the panels differ". |
| T6 | Code item 5 lists only locomotion.toml and the snapshot | **FIXED:** commit 5 also names the R6 test's first case and SOURCE.txt. |
| T7 | locomotion.toml lines 16-18 credit SCREEN with the config-only screen's inferred finding | **FIXED (commit 8):** "the config-only screen inferred (not measured) that a cap at or below 1.4 m/s stops paying at or below the cruise that gate needs" (configonly RESULT.md line 262, "[inferred]"). Same line count, so every line number cited holds. |
| T8 | SHA256SUMS.txt section comments stale since AMEND | **FIXED at the re-pin** (RF/scripts/repin_fix4.py): both comments now name AMEND for pilot3_judge.py, pilot3_noise.py, test_pilot3_judge.py, test_pilot3_noise.py and e2e_parent_cell.py. |
| T9 | The collapse reference's re-measure rule is not answered explicitly | **FIXED:** the collapse limit now says the rule is answered by VALIDATE's direct R6 statue (1091.40 at 2.0, 1094.43 at 0.2, within about 1.2 of 1092.6) and that the script's own run was not made. |
| T10 | Tools never ran at the head in the R6 env (habit seed trace, vg, video, scripted rollout, e2e) | **FIXED but the e2e:** all ran at 31d02e42 (RF/scripts/extras.sh): the scripted results equal the smoke's at 3ae1c959, the video tool's 40 re-rolls equal probe.json, the habit trace reads NO HABIT, V - G reports no problem, the replay is remade byte-identical. The e2e is not re-run (it deletes and rewrites T/e2e, whose summary is pinned); its parts ran. |
| T11 | The approval should name the recipes' inheritance | **FIXED** (decision (9) (iii); the pending list). |

### Note 1: commit 7's test (C1, C6)

`test_a_reset_starts_the_touchdown_event_afresh_on_the_same_env` rolls, on one instance of the committed locomotion
env (R6 on), the smoke's ALT_f1.5 pattern on seed 3042 to its end, then on seed 3045 cut one step after a
touchdown's registration (its pay still pending), then on seed 3045 again; a fresh instance rolls seed 3045 once.
After the last reset: the tracker's whole state equals a new tracker's (`vars()` equality); step 0 is accepted; the
episode's rewards and R6 info equal the fresh instance's; its pays equal SCREEN's events_probe (ported, delay 4) on
that episode alone; its first touchdown (the right foot at k = 2, the other foot than the left one registered last
before the reset) pays nothing; and the left foot's first touchdown (k = 90, 0.9 s, under the 1.2 s cap) pays
`min(k dt, 1.2) / (2 dt) x S` timed from the new reset.

Mutation check on a git-archive export of a3f1b378 with the new test file (sha256 `dae6b06b...`, equal to commit 7's;
the reviewer's harness, RF/scripts/mutate_reset.py) [measured: RF/out/mutants_reset.json, mutants_reset_stdout.txt]:
T17 KILLED; keeping across a reset the last foot, each foot's last touchdown, the pending pays, the down window, the
recent touchdowns or the leads: all KILLED; a first touchdown timed from k + 1: KILLED; keeping the stored output of
the last step: SURVIVED, equivalent (read only when the same step is scored again; a reset sets the last step to -1,
so step 0 overwrites it). A first version of the test (without the `vars()` check) let the recent-touchdowns and
leads mutants survive too (RF/out/mutants_reset_v1.json); the check was added before the commit. The export was
deleted after the run.

Checks at commit 7 (tree 5ca3e9b5) and commit 8 (tree 17ec0cb9) [measured: RF/out/c7, RF/out/c8]: ruff check and
format --check clean; mypy clean with and without SB3 and torch; notebook_cells parses; plant_contract,
species_catalog, digest_snapshot current; the R6 tests 27 passed; the T. rex suite 201 passed, 4 xfailed. At commit 8
the shared suite with 76ceae9's 14 ids deselected: 6514 passed, 1 skipped, 0 failed (28:40); the 14 alone: 14 failed,
the same ids and first-line reasons as at a3f1b378 (RF/out/c8/deselected14_failed_ids.txt, deselected14_reasons.txt).
The shared suite was not run at commit 7 alone (commit 7 changes only a T. rex test file, which the shared suite does
not collect; it ran at commit 8, which contains commit 7) [inferred].

### Note 2: the hop facts (B1, B2)

Recomputed from VALIDATE's per-episode records (r6_validate/out/v2/A_*.json, ret_w at 2.0 and 0.2, seeds 3042-3061)
with RF/scripts/hops_vs_standing.py [measured: RF/out/hops_vs_standing.txt], equal to the text review's file: best
standing per seed (statue, parent, W02) 1092.46 at 2.0 and 1095.79 at 0.2; at 2.0 H24 +135.3 (min +71.4, 20/20),
HF120 +299.0 (20/20), H20 +521.2 (19/20), H23 -160.4, HF050 -353.4, HF030 -546.0, HF020 -644.8; at 0.2 every hop
-862.5 to -911.2; FINAL3 - HF120 +33.4 (19/20), FINAL3 - H20 -188.8 (1/20), V01 - H20 -84.3 (1/20); every V05-V35 walk
ahead of every re-rolled hop on 20/20. H25 (1491.8) and HF150 (1519.9) are SCREEN's R6 means at 2.0 (SCREEN
RESULT.md, return table) [measured as recorded]. Each episode's return is linear in the forward weight, so the seed
means cross the best standing behaviour at weight 1.345 (H20), 1.537 (HF120) and 1.757 (H24), near ramp steps
1.27M, 1.49M and 1.73M [inferred: linearity; the ramp is linear, RewardRampCallback].

## Re-smoke at 31d02e42 and the tools' run

- Re-smoke (RF/scripts/resmoke.sh, 18:20:55-18:35:47 UTC) [measured: RF/out/resmoke_console.txt,
  smoke_evidence_launch_s42.*, resmoke_checks.txt, habit_ledger_smoke_s42.*; RF/nb/]: launch exit 0 (18:20:56-18:28:59),
  the two Warm start lines string-equal on stdout 91-92, 22 of 22 stage_config checks (git_commit 31d02e42), Eval
  1095.25 and 1092.31, 240 TensorBoard tags, ledger 192 columns with 0 alarms (TD 0.0093); rtest stopped after
  32,768 (exit 75), RESUME exit 0 with no Warm start line and both -2.0 records kept (Eval 1095.54). Everything
  that trained is bitwise the validate stage's re-smoke at a3f1b378.
- Tools (T/out/fix4_final_run.sh after `git merge --ff-only` of T/tree from a3f1b378 to 31d02e42, 18:36:25-18:43:10
  UTC) [measured: T/out/fix4_final/, RF/out/tree_ff_merge.txt]: pytest 194/194; test_pilot3_score.py 310/310; PC
  ledger validation byte-identical to AMEND's; J1-J10 equal to AMEND's (exit codes, nothing written, last lines;
  J1/J7/J10 probe episodes equal, 40/40 STILL); noise equal but for path fields; tool files unchanged; tree clean.
- Extras (RF/scripts/extras.sh, 18:43:23-18:50:16 UTC) [measured: RF/out/extras/, probe_parent/, scripted/,
  videos/, replay/]: parent probe 40/40 STILL, episodes equal to VALIDATE's; scripted grids 1 and 2 results equal to
  the smoke's (3ae1c959); habit seed (--validation) NO HABIT, 20/20 full-horizon; V - G no problem, [500,800) and
  [800,1000] flagged (report only; a 65,536-step child); video 40/40 match probe.json; replay byte-identical (0 of
  1,000 frames differ).

## Text changes (FC2)

criteria_pilot3.txt: title and Status (FIXES4); decision (8) both bases; decision (9) the three departures; the order
of work (commits 7-8; re-smoke at 31d02e42); Code (commit 5's full contents; commits 7 and 8; the head; "line numbers
cited at a3f1b378 hold at the head"); the recipes sentence; the Warm start pointer; the R6 heading; SCREEN's R6p
clause; the tests paragraph (27 cases, commit 7's test, the golden's skip, the mutants, live coverage); the two hop
limits; the noisy-start panels; the collapse re-measure sentence; the probe pointer; the statue and ramp citations;
the Repository ref line and the snapshot pointer; the collapse-stop note; HOPS; "Known limits" (untested in training;
the resume at 31d02e42); "How it is judged" (FIX4 re-pin; score and judge tests at FIX4); pre-launch (moved lines
none for 7-8; head; tree; commits 7-8 block; stance note; three smokes; judge pointer; "Not run at a3f1b378" pointer;
new "Re-smoke at 31d02e42" and "FIX4's run of the pinned tools" blocks; the FIX4 re-pin); the pending list (eight
commits; the departures; ls-remote at 31d02e42); the exception text's head. Placeholders for the new hashes were
filled by RF/scripts/fill.py (counted); paragraphs touched were rewrapped at 120 columns by RF/scripts/reflow_ranges.py
(whitespace only, words checked equal). ASCII only; the only lines over 120 columns are the two Warm start lines.
NOTES.md: a FIX4 paragraph; I31 (measured since); I33 rewritten; section 4 (the departures for the approval; the
not-run pointer); the records-PR note (commit 8 leaves the two known wordings; commit 7 corrects commit 4's).
BRANCH_PLAN.md: title; a FIX4 paragraph; commit 6 "the head until the R6 reviews"; sections for commits 7 and 8;
order of work step 7.

## Open items (for the maintainer; none blocks the text)

1. Approval of the text (sha256 `853f49d6...`), of the eight local commits, and of the three departures named under
   decision (9): touch-sensor down flag, pay at k+4, the 11 recipes inheriting R6 (or a branch change pinning them
   off, which would move the head again).
2. The push (remote must read `31d02e42d148511e50f9cba9a598d9a9d36c23af`), the durable copy of the 105 pinned
   entries, the Colab steps, the launch approval: unchanged in kind.
3. Not run at 31d02e42: the held-out seeds 3062-3081 in the R6 env; the full e2e.

## Slips (this stage)

1. **CMDLOG literalness before the logger.** The first six shell calls ran directly and are copied into CMDLOG.txt by
   hand, with paths shortened as "SP=..."/"P=..." and approximate times.
2. **Read-only calls outside the logger:** one `cat` of RF/out/mutants_reset_stdout.txt (about 17:40Z), one
   `tail -c 300` of RF/out/c8/shared.txt (17:54Z); both NOTE'd in CMDLOG.txt. Several heredoc file writes (scripts,
   the c7/c8 messages) and cp/sed copies ran in the shell tool just before a logged NOTE that names them, not inside
   the logger.
3. **The system python3 ran once:** a stray empty heredoc to `python3 -` (with `|| true`) at the start of the call
   that filled commit 8's message (about 18:20Z); it read and wrote nothing. NOTE'd.
4. **A pointless redirect target:** one logged read redirected a no-op `head -0` line to the standard-output device
   path (about 17:58Z); that is not the null or zero device, but it was needless.
5. **A failed one-line read:** the rtest launch record has no "ended" key (stopped run), so a `python -c` read raised
   KeyError; nothing else was affected. NOTE'd.
6. **Text work beside simulations.** While the shared suite, the re-smoke, the tools' run and the extras ran in the
   background, this stage edited the frozen texts and its own scripts and ran logged read-only commands; no two
   simulations or trainings overlapped.
7. **Writes outside RF and SP/wt_p3, all asked for by the task:** T/out/fix4_final_run.sh and T/out/fix4_final/
   (pinned evidence), T/SHA256SUMS.txt, T/tree's checkout and .git (the fast-forward), and FC2's three texts plus
   FIXES4.md. The two commits wrote their objects and the branch ref into the worktree's common git directory
   (/home/user/mesozoic-labs/.git, the store SP/wt_p3 shares), as commits 1-6 did; the --shared clones read objects
   from that store; nothing else there was touched. Xvfb (the private display for the final replay, the video tool and
   the replay check) recompiled a keymap in /var/lib/xkb, as at the earlier smokes; the system temp folder's own
   mtime moved during the launch while no entry in it (other than the session's) was newer than the stage marker.
8. **No forbidden-device check.** No search for the null or zero device was run on this folder's text, since the
   search would have to spell it; none was written knowingly; every discarded output went to a file in RF.
9. **Commit trailers.** Commits 7 and 8 end with the two lines the task prescribes ("Co-Authored-By: Claude
   <noreply@anthropic.com>" and the Claude-Session line), as commits 1-6 do; the harness's own attribution note asked
   for a product-and-version name in the first line, which the task's rule against model names and the branch's
   convention exclude.
10. **Model-family words in a logged pattern.** The final check for model names (about 19:00Z) ran through the
   logger, so its grep pattern (four model-family words) was written into CMDLOG.txt. That one line was masked with
   the file-editing tool right after ("<MASKED: four model-family words, alternated>"), a NOTE says so, and the
   check was re-run with the harness's search tool, which writes no file: no match in RF, FC2's texts or the two new
   commit messages (dumped to RF/tmp/last2_msgs.txt by one `git log` call outside the logger).
No push, PR, git-config or remote change; nothing trained but the pre-registered CPU smoke at the new head (65,536
steps, and its 32,768-step rtest and RESUME). Folder size at the end: about 300 MB (the clone about 175 MB).
