# Walking pilot 3, AMEND stage: every change against frozen_candidate/, and what stays open

2026-11-04 (UTC), 16:01:59-16:30 (the last command's time is at the end of CMDLOG.txt). Folder: SP/work/pilot3/frozen_candidate2 ("FC2"); "SP" is the session scratchpad
`/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad`. Labels: **[measured: source]**
= read or computed from the named file or a command this stage ran; **[measured as recorded: source]** = an earlier
stage's record; **[inferred: basis]** = reasoning. Every shell command is in FC2/CMDLOG.txt (UTC; literal from the
first "BEGIN" line on; the exceptions are under "Slips"). Writes: FC2 only, plus SP/work/pilot3/tools/ for the tool
change this stage was asked to make (below). Nothing was written in SP/wt_p3 or /home/user/mesozoic-labs; nothing was
committed, pushed or trained; the only simulations were the judge's three probe runs (J1, J7, J10) inside the tools'
final run, one at a time, CPU, `nice -n 10`, `OMP_NUM_THREADS=1`.

## Inputs and outputs

Inputs, copied byte for byte into FC2 and FC2/base/ [measured: FC2/out/copy_sha.txt]:

| File | sha256 (frozen_candidate/, input) | lines |
|---|---|---|
| criteria_pilot3.txt | `6daef03df8094df222a9e692485f2d5e40c218270c9573e736ca31d24b3ecbc7` | 965 |
| BRANCH_PLAN.md | `00906f09f54cb3b2f310926761bf78ac898727036e4c30bd496ac4531791c004` | 228 |
| NOTES.md | `53de4c3a7020fbb801bf6de63acd6d06bc9593c000c119684bdbeaae7a2e23b1` | 225 |
| CHANGES.md | `ffcddfa019a542501cceac74bc4334d05c22781d4ca73b25f66f7d6caab6f702` | 204 |

FIXES2.md and FIXES3.md were also copied, unchanged, because the criteria's title calls them "beside this file".

Outputs (sha256 and line counts after the last edit; also at the end of CMDLOG.txt):

| File | sha256 | lines | diff |
|---|---|---|---|
| criteria_pilot3.txt | `93f0685181d788fbba9a7a8c6b8c4495ab16f349b6712255f1d58dc493373138` | 1284 | diffs/criteria_pilot3.txt.diff |
| BRANCH_PLAN.md | `df6dc0a6f66bcef98ef9436bf1e285f82c39e2caa1c71c8144036b6881fe83c6` | 311 | diffs/BRANCH_PLAN.md.diff |
| NOTES.md | `c74a6fa287b7e3cb077732f0bbf597b074ccb454b17fcd00e1fa02f4bd65c71f` | 253 | diffs/NOTES.md.diff |
| CHANGES.md | `d03cff5f0c73be1094387e49f3a7f5052d21b9f97b9f18578c7fa996a233e100` | 209 | diffs/CHANGES.md.diff |
| SP/work/pilot3/tools/SHA256SUMS.txt | `f90a89b45fa5537f3ed5ad6e4c93a272e77947f2b004bfd710bd91fd4bdf86c5` | 89 entries | diffs/SHA256SUMS.txt.diff |
| the five changed tool and test files | below | | diffs/tools_amend.diff |

Each diff is `diff -u base/<file> <file>` (the tools': `diff -u tools_before/<file> tools/<file>`).

## The decisions applied

[measured: SP/work/run_log.txt lines 1034-1035, read in this stage]
- Line 1034, 2026-11-04 11:57 UTC, "MAINTAINER DECISIONS (AskUserQuestion)": (8) pilot 3 noise reset "-2.0
  (Recommended)"; (9) walking reward "R6, new code (Recommended)": support-gated alive/posture/height + airborne
  penalty + forward_vel_max 1.5 + small alternating-touchdown reward, pilot-only on the pilot branch, off by default in
  code, with tests; (10) gate "Fix later (Recommended)" (with the gait-quality plan; not for pilot 3); (11)
  inputs/plant "No change (Recommended)". It also set the order: build R6 and the -2.0 reset on the local branch,
  amend the frozen text, re-smoke, review; then approval of the text and commits, push, Colab steps, launch approval.
- Line 1035: the build workflow (design, implement, validate, this amendment, two reviews, fix).
These match the decisions in this stage's task text. In the criteria they are a new block, "Maintainer decisions of
2026-11-04", with their evidence lines; decision (c) is marked replaced by (8).

## Changes to criteria_pilot3.txt (each in diffs/criteria_pilot3.txt.diff)

| # | Section | Change | Source |
|---|---|---|---|
| A1 | Title, Status | "AMENDED 2026-11-04 ... (CHANGES2.md beside this file)"; the status says what the amendment records and that it opens no item | this file |
| A2 | Decisions | (c) marked "REPLACED by decision (8) of 2026-11-04: -2.0"; new block (8)-(11) with the run-log line, the log_std scan evidence for (8) (120/120 at -2.0 on 3042-3081, 359/360 over all panels, 9/120 at -1.5; micro-lifts, backward slide; no step), pointers for (9), (10), (11); decision (d)'s exception stands with a pilot-only reward; the order of work and the departure (commits 4-6 made locally before approval) | run log 1025, 1034-1035; overnight/logstd and verify_logstd RESULT.md |
| A3 | Changes against pilot 1 | C1: the launch check reads the branch's R6 task digest 782d95b6... (main's 5f835d65... kept as context); C3: R6's code added (commit 4) and turned on (commit 5); C5: value -2.0; new C16: the reward is R6, why | digest snapshot line 396 at a3f1b378; SCREEN RESULT.md |
| A4 | Code | commits 4-6 with full hashes and contents; head a3f1b378aede66b0ee8630197f36a7c374ec77c0; "it moved before launch, so this file was amended" | git log of SP/wt_p3 (read only); FC2/out/wt_p3_state.txt |
| A5 | What is trained | the reward is main's [env] with the four R6 lines; `warm_start_log_std = -2.0` (std 0.1353; x2.4, x19, x1.2 against the parent's printed std; written -2.0, not -2); a new block "The walking reward R6" stating R6 exactly as built, parts (a)-(d), with the code's line numbers at a3f1b378, the compsognathus rule (its lines 200 and 250), the threshold 33.636528 N on the summed per-foot substep-minimum touch, the R5p event rule as built (3 of 5 substeps; debounced touchdown set; alternation; +-2-step simultaneity; min(T_i, 1.2 s)/(2 dt) x clip(dx/0.1L, 0, 1); paid at k+4, not SCREEN's k+3; touch sensors rather than floor truth, with the measured agreement), the info keys, off-by-default mechanics, the 11 behavior recipes that inherit R6, and what does not change | tools/tree at a3f1b378 (base_env.py lines 110-223, 311-317, 1092-1102, 1124-1125, 1209-1210, 1251-1252, 1408-1437, 1531-1532; locomotion.toml lines 13-18, 39-51, 190-204); compsognathus_longipes/base_env.py lines 200, 250; messages of b7a146d, 4dfc7cb, a3f1b37; r6_design/DESIGN.md; r6_validate/RESULT.md |
| A6 | What is trained, Warm start lines | the second expected line replaced by the one VALIDATE measured: `Warm start: policy log_std set to -2 (std 0.135) on all 15 actuators (the parent's std: mean 0.057, min 0.007, max 0.111; Adam moments kept)`; the first unchanged; "No amendment of the two lines is needed" replaced by the two smokes' records (3ae1c959: Option B's lines; a3f1b378: the new pair, stdout 91-92); exactly two lines in the file start with "Warm start: ", and they equal VALIDATE's nb/expected_warm_start.txt (checked) | r6_validate/nb/launch_s42/runner.stdout.log lines 91-92; RESULT.md section 4 |
| A7 | What is trained, TensorBoard | the re-smoke's 240 tags (244 + reward_terms/eval_walk_touchdown - 5 fall/termination tags); the count is not fixed | r6_validate/out/tb_tags_vs_smoke.txt; RESULT.md finding 5 |
| A8 | New section "R6: the evidence and its limits" | SCREEN (setup, main's preference for the hop, R6p's verdict and margins on both panels, the verifier), the config-only screen, DESIGN's offline check, VALIDATE part 1 (env R6 = SCREEN R6p, ranking reproduced), the branch's 26 tests; limits: screened offline on hand-built behaviours only, untested in training; H20 out-earns FINAL3 and V01 at weight 2.0; the noisy -2.0 parent under R6 (40/40, 1083.3, below the statue); hidden history in the event; W02's small pay; posture forgiven in the air; tie order; SCREEN's untuned constants; the collapse backstop under R6 [inferred] | SCREEN, verify_screen, configonly RESULT.md; DESIGN sections 11-12; r6_validate RESULT.md parts 1 and 3 |
| A9 | How it is judged | the probe rolls the R6 env, no label reads a reward; the re-pin at AMEND; the judge's binding values -2.0; the judge's and its test's new sha256s and what changed; the noise tool's new sha256, what changed, and that its forward references stay on the 2.5 m/s scale; the e2e's new sha256 (not re-run since FIX2); the adapted score test passed again on the moved tree | tools/out/amend_final/; FC2/diffs/tools_amend.diff |
| A10 | The probe and fixtures | R6 enters no observation, action or dynamics; the parent re-probed 40/40 STILL at a3f1b378 | r6_validate probe_parent/ |
| A11 | A seed's outcome | collapse-stopped: under R6 a collapse stop is likelier [inferred] | SCREEN's R6 hop returns |
| A12 | Run handling | the Repository ref line's commit a3f1b378...; the first-Eval note beside both statues (main's 1092.60 at locomotion.toml line 149 at a3f1b378; R6's 1094.43 / 1091.40); first hour: -2.0 twice, hyperparameters_sha256 85881fc4..., task_sha256 782d95b6..., the four R6 reward_weights; the RESUME's records measured again at a3f1b378 | digest snapshot lines 396, 398 at a3f1b378; r6_validate RESULT.md sections 1 and 4 |
| A13 | What each outcome means | intro: reset 0.1353 on R6, "reward" among the differences from pilot 1; Q4 YES: R6, and that a walk on main's reward stays open; HOPS rewritten for R6 (the hop learned despite R6; SCREEN's ranking did not carry over); STILL: noise near 0.135, the forward term on R6's scale (0.8-5.3), R6 as a candidate cause; FALLS: x2.4 / x19 (x3.9 / x32 kept as -1.5's) | the R6 section; I17 |
| A14 | The locomotion gate | unchanged by decision (10) and R6 (02602cb0... at every commit); SCREEN's G0 finding | commit messages; SCREEN |
| A15 | Report only | noise against 0.1353; the forward term's 2.5 m/s references and their 5/3 scaling under R6 (the noise tool prints them unscaled); the hidden-clock audit: R6's event reads unobserved history and, for a foot's first touchdown, the episode step; structure in [0,100) and [100,200) possible [inferred]; G is R6's return | code reading of _WalkTouchdownEvents; DESIGN 10.2 |
| A16 | Known limits | one noise value -2.0 and one reward R6; R6 untested in training; the ledger has no reward-term columns (TensorBoard only); the re-smoke's train/std and value numbers beside the smoke's; the -2.0 jump smaller than pilot 2's; the re-smoke's children 40/40 STILL; the re-smoke's resume not bitwise either | r6_validate RESULT.md section 4 and finding 2 |
| A17 | Pre-launch checks | the head-level list per commit (2, 5, 6); "measured on the local commits up to a3f1b378"; the tree = 4d5f4e7 + commits 1-6, clean at a3f1b378; commits 4-6 with their moved digest lines and suite results; stance untouched (VALIDATE part 2); the notebook smoke marked as run twice; the 3ae1c959 record kept as history, with its Warm start line marked -1.5; the judge record marked as at 3ae1c959; 87 of 87 judge tests at AMEND; new blocks "Re-smoke at a3f1b378" and "AMEND's checks of the re-pinned tools"; the re-pin (89 entries); pending: six commits, the push reading a3f1b378, the Colab steps with the "-2" line | messages of b7a146d, 4dfc7cb, a3f1b37; r6_impl/out; r6_validate RESULT.md; tools/out/amend_final; FC2/judge; FC2/out |
| A18 | Recorded exception text | amended 2026-11-04; -2.0 (0.1353) replacing -1.5; the stage with the pilot-only reward R6 described in one sentence (screened offline, untested in training), the gate and inputs unchanged; a3f1b378 | the above |
| A19 | September table header | its forward term is on the 2.5 m/s scale (5/3 under R6's cap) | R6 (c) |
| A20 | Layout | paragraphs touched by the edits rewrapped at 120 columns (FC2/bin/reflow2.py: whitespace only, the words checked equal); one split path and one split quoted string rejoined (FC2/bin/edit_fix1.py, and the file-editing tool); ASCII only; the only lines over 120 columns are the two Warm start lines, as before | FC2/CMDLOG.txt |

Kept as history on purpose (they describe what was measured at 3ae1c959 or with the -1.5 tools): the smoke's record
of 2026-11-03 (its 17 checks with -1.5, 8263f82f..., 5f835d65..., train/std 0.2245-0.2252), commit 2's moved lines,
FIX2's and FIX3's checks, the earlier tool sha256s quoted in BRANCH_PLAN.md's table, and decision (e)'s departure.

## Changes to BRANCH_PLAN.md, NOTES.md and CHANGES.md

- BRANCH_PLAN.md: title and an "Amended 2026-11-04" paragraph (commits 4-6, head, the stages); commit 2's cell notes
  -2.0 since commit 6; new sections for commits 4, 5 and 6 (files, built hashes and trees, checks, moved lines, the 11
  recipes); "Possible 5" renamed "Possible extra commit (not made)" because a commit 5 now exists; an AMEND note on
  the judge and noise tools and the re-pin; the judge row notes -2.0; order of work step 6; two risks (R6's code on
  the stance class; tools re-pinned for the reset value).
- NOTES.md: an AMEND paragraph at the top; I6, I10, I17 annotated "(AMEND: ...)"; new inferred points I30 (collapse
  backstop under R6), I31 (scripted rollout and video unaffected at a3f1b378), I32 (V - G structure), I33 (what a
  STILL or HOPS under R6 says); section 4 records decisions (8)-(11); section 5 adds the validate stage's findings
  for the reviews and what was not run; the records-PR note says the two inaccurate wordings still hold at a3f1b378
  and names the recipes' inheritance.
- CHANGES.md: one AMEND paragraph at the top (the values quoted in it are FINALISE's); otherwise unchanged.

## The tool change (SP/work/pilot3/tools/)

Why: the validate stage measured that the pinned `pilot3_judge.py` refused every pilot-3 run at -2.0 because of
`WARM_START_LOG_STD = -1.5` (line 85, enforced at lines 206-211), and that `pilot3_noise.py:39` printed exp(-1.5) as its
reference [measured as recorded: r6_validate/RESULT.md section 4]. No tool hard-codes the head (`--expect-head` is an
argument) or the recipe digest (derived from the tree) [measured as recorded: same; and grep in this stage].

Backup first: FC2/tools_before/ holds the FIX3 bytes of `SHA256SUMS.txt`, `pilot3_judge.py`, `pilot3_noise.py`,
`tests/test_pilot3_judge.py`, `tests/test_pilot3_noise.py`, `tests/e2e_parent_cell.py` (BACKUP_SHA256.txt: their
sha256s equal the FIX3 pins), and `tree_state.txt` (tools/tree at 3ae1c959, tree 6d507481, clean). `sha256sum -c` of
all 73 FIX3 entries was OK before any change [measured: FC2/out/tools_sha256_check_start.txt].

The edits (FC2/bin/edit_tools.py, exact counted replacements; diffs/tools_amend.diff):

| File | Change | sha256 before | sha256 after |
|---|---|---|---|
| pilot3_judge.py | line 85 `WARM_START_LOG_STD = -1.5` -> `-2.0`; docstring line 39 -1.5 -> -2.0 | `66d238dd...` | `df5ea2148a948e5747576bf295510b1612ba2e3ccfdc02e890339d63062013ab` |
| pilot3_noise.py | line 39 `math.exp(-1.5)` -> `math.exp(-2.0)`; docstring line 8 "0.2231 ... (exp(-1.5))" -> "0.1353 ... (exp(-2.0))" | `95f05851...` | `c191c72ab1acfdd8e43f68f269fcce452d89b78cd13a2be4f74084580ebd01a0` |
| tests/test_pilot3_judge.py | -1.5 -> -2.0 at lines 39, 44, 65, 347, 377; the string case at 84 -> "-2.0" (still a type refusal); two added cases: run and curriculum -1.5 are refused ("is -1.5, not -2.0") | `21b26f14...` | `cc26edcdd8e7dc244b6ec4502055544c1e07435930cfcf853073b4bb4bfe65a3` |
| tests/test_pilot3_noise.py | line 130 reference 0.2231 -> 0.1353 | `6b25633d...` | `3a315964f6e1bdd3ee5c509f08ee8821496e762b8f35e1fbef30272be16ce146` |
| tests/e2e_parent_cell.py | -1.5 -> -2.0 at lines 4, 85, 105 | `be256afc...` | `3bafab00f86efbc91d23d8b31f0637f384255475386997d32046eca1098d9a0b` |

Left unchanged on purpose: `pilot3_noise.py`'s forward references (pilot 1's 0.5-3.2 and the September walkers'
39-54; they are on the 2.5 m/s scale and the criteria now say how to read them), the value-agnostic synthetic values
in `tests/test_nb_pilot_runner3.py:141`, `tests/test_pilot3_habit.py:305` and `tests/test_pilot3_judge.py:279-284`,
and `tests/test_pilot3_noise.py:102,121` (synthetic log_std values). The judge's code now equals VALIDATE's
diagnostic copy (sha256 bc483493...) except the docstring line 39 [measured: FC2/out/judge_vs_validate_diagnostic_copy.diff].

The tests' tree: `tools/tree` (a --shared clone of SP/wt_p3 on branch claude/pilot-walk-r10-p3, objects through the
main repository's store, read only) was fast-forwarded to a3f1b378 with `git merge --ff-only` (no fetch: the commit
was already reachable; nothing ran in SP/wt_p3); it is clean before and after (tree a08393c5, equal to commit 6's)
[measured: FC2/out/tree_ff_merge.txt, tree_state_after.txt]. Its remote-tracking ref still reads 3ae1c959 ("ahead
3"), which no test reads. To restore the FIX3 state: `git -C tools/tree reset --hard 3ae1c959` (not done).

The final run (tools/out/amend_final_run.sh; outputs tools/out/amend_final/; 16:08:03-16:15:14 UTC; TMPDIR, pytest
basetemp and caches in FC2/tmp) [measured]:
- pytest 194 of 194 passed (judge 87, habit 45, runner 13, video 9, V - G 31, noise 9); test_pilot3_score.py 310 of
  310; validate_pc_ledgers.py byte-identical to FIX3's; tool files equal before and after; tree clean after.
- pilot3_noise.py on the re-smoke's two runs: every value equal to VALIDATE's run of the FIX3 tool but the reference
  (0.1353 against 0.2231), tool_sha256, and three path fields (VALIDATE used relative paths) that name the same files
  [measured: compare_with_validate.txt; FC2/out/noise_path_check.txt].
- The judge's ten smoke validation steps on tools/tree (a3f1b378), the re-smoke's runs and VALIDATE's run copies,
  OUT in FC2/judge/: exit codes, "nothing written" and final messages equal to VALIDATE's diagnostic copy's in all
  ten; J1, J7, J10's 40 probe episodes equal (40/40 STILL each); bindings differ only in the tree path
  [measured: compare_with_validate.txt, FC2/judge/logs/].
- compare_amend.py exits 1 ("2 unexpected differences"): its allowed list for the noise records was too narrow; the
  two are the path fields above, resolved equal by FC2/bin/noise_path_check.py (exit 0). Recorded, not hidden: the
  rcs.txt line "compare_amend rc=1" stays in the pinned evidence.
- Not re-run: the e2e (it deletes and rewrites tools/e2e and starts Xvfb, which writes outside the stage folders; its
  FIX2 result stands for the unchanged tools); habit seed mode, V - G and the video tool (unchanged files).

Re-pin: FC2/bin/repin.py rewrote SHA256SUMS.txt: the five changed lines get their new sha256, a new header names this
stage and the kept FIX3 list (FC2/tools_before/SHA256SUMS.txt), and 16 lines pin this stage's run script and evidence;
every other line is byte for byte FIX3's. `sha256sum -c`: 89 of 89 OK [measured:
FC2/out/tools_sha256_check_after_repin.txt]. SHA256SUMS.txt sha256 `f90a89b45fa5537f3ed5ad6e4c93a272e77947f2b004bfd710bd91fd4bdf86c5`.

## Open items (for the maintainer and the reviews; none blocks the text)

1. **Pending, unchanged in kind:** the maintainer's approval of the amended text (its sha256) and of the six local
   commits; the OK to push and the push (remote must read a3f1b378...); the durable copy of the pinned tools (now 89
   entries); the Colab steps with the "-2 (std 0.135)" line; the launch approval.
2. **The 11 T. rex behavior recipes inherit R6** through locomotion's [env] (disclosed in 4dfc7cb; written in the
   criteria and NOTES). The maintainer may want them pinned off; that would be a branch change, so a new head and
   another amendment.
3. **Not run at a3f1b378** (written so in the criteria): the held-out seeds 3062-3081 in the head's R6 env, the
   scripted rollout, pilot3_video.py, the replay byte check, pilot3_vg.py, the full e2e. The reasons are inferred
   (the reward does not enter the dynamics; off, the env is 3ae1c959's).
4. **R6 is untested in training.** The text says so wherever it uses SCREEN's ranking.
5. **The collapse backstop under R6** (I30) is an inference, not measured; the collapse keys and the reference 1092.6
   are unchanged (the gate side was not decided).
6. **The decision number** `D-D<next>` and the exception text's `(sha256 <...>)` stay for the records PR.

## Checks run in this stage

- Inputs: sha256 equal to the task's (criteria `6daef03d...`, 965 lines) [FC2/out/copy_sha.txt].
- Tools: 73/73 OK before; 89/89 OK after the re-pin; the final run above.
- SP/wt_p3: HEAD a3f1b378, tree a08393c5, `status --porcelain --ignored` empty, the parent chain 4d5f4e7 <- f2399c8
  <- 76ceae9 <- 3ae1c959 <- b7a146d <- 4dfc7cb <- a3f1b378 [FC2/out/wt_p3_state.txt; read only, GIT_OPTIONAL_LOCKS=0].
- The criteria: the two "Warm start: " lines equal VALIDATE's measured pair; no `TO FILL`; ASCII only; no line over
  120 columns but the two Warm start lines; every remaining -1.5, 0.2231, 3ae1c959, 8263f82f..., 5f835d65... and 244
  read and kept only where it is history or a comparison (grep in CMDLOG at 16:21-16:22Z).
- Cited facts re-read in this stage: run log lines 1022-1035; the commit messages of b7a146d, 4dfc7cb, a3f1b37; the
  digest snapshot lines 395-401 and 630-633 and locomotion.toml at a3f1b378 (tools/tree); base_env.py's R6 lines;
  compsognathus_longipes/base_env.py lines 200, 202, 250; SOURCE.txt line 93 and locomotion.toml line 108 (the two
  wordings for the records PR); SCREEN, LOGSTD, verify_logstd and verify_screen RESULT.md (parts); DESIGN.md;
  r6_validate/RESULT.md; r6_validate/nb's Warm start lines.

## Slips (this stage)

1. **CMDLOG literalness, before the logger.** The first three shell calls ran directly in the shell tool and are
   copied into CMDLOG.txt by hand, with the scratchpad path written "SP=..." and only the first time measured.
2. **One read-only shell call outside the logger** (about 16:07Z): `ls` of SP/work/morning, overnight/, overnight/
   screen and overnight/logstd, and `wc -l` of the overnight RESULT.md files and the morning report. Recorded in a
   hand-written NOTE in CMDLOG.txt (16:13Z).
3. **The system python3 ran once.** A logged block at about 16:06Z ended with a stray `python3 -c "print(1)"` (it
   printed 1 and did nothing else); every other Python ran as the prescribed optB interpreter with `-I -B`, or the
   tools' own venv for the tools' run, as FIX3 did. Self-reported in the 16:13Z NOTE.
4. **The comparison script's allowed list was too narrow**, so the pinned rcs.txt carries "compare_amend rc=1"; the
   follow-up check (noise_path_check.py) first crashed on a dict field (TypeError) and was patched by a logged inline
   edit and re-run (exit 0).
5. **Read-only work beside a simulation.** While the final run's judge steps rolled the probe (J1 16:09:46-16:11:30,
   J7 16:11:43-16:13:27, J10 16:13:32-16:15:14), this stage edited text with the file-editing tool, read files with
   the file-reading and search tools, and ran logged read-only shell commands (`cat`/`tail` of the run's rcs files,
   one `sha256sum` of the five changed tool files). No second simulation or training ran at any time.
6. **Writes outside FC2, all in SP/work/pilot3/tools/ and asked for by the task:** the five edited files,
   SHA256SUMS.txt, out/amend_final_run.sh and out/amend_final/ (the run's evidence, pinned, as FIX2 and FIX3 kept
   theirs), and tools/tree's checkout and .git (the fast-forward). git read objects from the main repository's object
   store through the clone's alternates (read only), and one `git status` read SP/wt_p3 (GIT_OPTIONAL_LOCKS=0).
7. **File reads and writes with the harness's tools** are NOTE lines in CMDLOG.txt, in words (the criteria edits,
   BRANCH_PLAN.md, NOTES.md, CHANGES.md and this file were written with the file-editing and file-writing tools). The
   first NOTE's time was typed as "16:0xZ"; its BEGIN line carries the real time.
8. **No forbidden-device check.** No search for the null or zero device was run on this folder's text, because the
   search itself would have to spell it; none was written knowingly, and no command discarded output.
No model name was written; no training, push, commit, PR or git-config change; nothing in SP/wt_p3 or
/home/user/mesozoic-labs was written. Folder size at the end: about 10 MB (judge/ 5.9 MB, tmp/ 2.9 MB).
