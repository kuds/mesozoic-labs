# fix:CE-minor (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WP CE, third fix round: report (branch wt/wpc-ce, worktree /home/user/mesozoic-wt/wpc-ce, head cd32e19; nothing pushed, main checkout untouched). Simulation tooling only; nothing here is a result.

Three of the seven points were wording defects in CE_RULE_WORDING.md and are fixed, each with a test. No source file changed. Finding 1 (the merge commit without trailer lines) cannot be fixed without rewriting history and needs your decision.

THE MERGE
- The working branch had moved 23 commits (82ba1ad to cd68f2c). Merged first as 05150a2, automatic, no CE file touched; its message ends with the two lines.
- The branch is now 7 ahead, 0 behind. The working branch was still at cd68f2c when I finished.
- Record-key hashes before and after: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered ones).

1. PER FINDING
(1) Merge commit cdc4762 lacks the trailer lines: for the lead. Confirmed; it is on no remote. My merge 05150a2 sits on top of it, because the task said to merge first.
- Option a (recommended; the precedent of CB's 5aaf021 in ROUND_C_RULINGS_16NOV.md): merge d500d89 with your own merge commit, then cherry-pick cd32e19. cdc4762 and 05150a2 stay unmerged. `git merge-tree --write-tree cd68f2c d500d89` gives tree d7ef7ebd, which is exactly 05150a2's tree and cd32e19's parent tree; cd32e19 touches two CE files only. So the result is my head's tree.
- Option b: merge the tip cd32e19; cdc4762 enters the history without the two lines.
- Option c: amend; a rewrite, not done.

(2) Entry 9 (X1), "Old" quote no longer in the draft: fixed.
- Confirmed with my own check: 20 of 20 at 984f81d, 19 of 20 at wt/prereg-b's tip 0d10a94 (X1, since 405c218).
- Entry 9 is withdrawn ("No wording change is asked"); it cites the draft's own sentence. The header now names the draft commit the old sentences were compared with (c4a0ff8, the draft's last change; identical at 0d10a94).
- After the fix: "19 of 19 Old quotes verbatim" at both commits; 30 test names, 0 not defined.
- Test: test_the_wording_file_asks_no_paste_for_x1_which_the_draft_holds. The comparison with the draft itself stays a script, not a committed test, because the draft is on another branch.

(3) Entries 3a and 3b (W3) lack two clauses of REWARD_RULE: fixed. Confirmed against r5core.py (the clamp to 1..256, `swing <= 0.0`).
- 3b: "the last n env steps, n = round(stride period / 1.05 / 20 ms), kept between 1 and 256 env steps: the history the env keeps"; "until n env steps".
- 3a: "a track that plans no complete swing has its gait's, (1 - duty) x the stride period. Where T is not above zero, (a) is not read". The first half is one clause more than the finding named: REWARD_RULE holds it, and the second half reads oddly without it.
- Label A for both.
- Test: test_the_wording_for_w3_states_dutys_bounds_and_clearance_as_guard.

(4) Entry 8's K4 sentence lacks "as built": fixed in the wording file. Confirmed: `tasks` is in RESUME_KEYS and holds the weights in force; only describe()'s docstring had the caveat.
- The sentence now reads "of the description of the envs as built, `config`, ...", with the reason (the weights are run state by K5, changed by M4's fix).
- Test: test_of_what_a_resume_compares_a_change_of_the_task_weights_moves_the_weights_alone. It also pins the behaviour: set_task_weights changes `tasks` alone of the whole description, and in it the weights alone.
- The same sentence for CT's brief is yours (see "for the lead").

(5) env_configs names the tasks the records hold today: for the lead; code unchanged. Confirmed by reading (r5env.py:347-348 and 780).
- Option a: leave it. T2 allows an absent task for a new run, and `absent` lists it with its reason.
- Option b: env_configs refuses when the names differ from the latest manifest's table.
- I recommend a, as the reviewer does. T2 treats a reference that fails G9-ref part (a) as not available; if that is ever recorded by marking the entry after the freeze, option b would block it.

(6) The crouch check refuses the reviewer's case C: not a defect. It fails closed, and a set has one crouch. No change.

(7) CT and the design listing: for the lead; none of it is in CE's files.
- wpc-ct is at cd68f2c and does not hold this round.
- WPC_CT.md at cd68f2c has no line on the resume comparison, although ROUND_C_RULINGS_16NOV.md says it was added.
- The design's section 2.2 lacks `env_configs(..., root=)`, `from_plan(..., heading=)` and the new description keys.

All three wording tests fail on the file as it was ("FFF", in without_fix.log) and pass now ("3 passed, 36 deselected in 3.68s").

2. TESTS AND ACCEPTANCE (at cd32e19, clean tree, the brief's env, through tools/lowprio.sh)
- tests/test_stage2_r5lib.py tests/test_stage2_r5core.py: "62 passed in 49.39s"
- -rs tests/test_stage2_r5env.py tests/test_stage2_g8a.py: "46 passed, 1 skipped in 89.65s (0:01:29)"; the skip: "tests/test_stage2_r5env.py:675: slow: set STAGE2_SLOW=1"
- STAGE2_SLOW=1 -k walks_on: "1 passed, 38 deselected in 6.64s"
- tests/test_stage2_tbenv.py: "11 passed in 0.80s"
- ruff on the eight files (run before the commit; no .py file changed after it): "All checks passed!"
- `g8a.stock('75-M-H-10', 'walk/brisk', 3042, 5, 6.0)`: "True 3000", exit 0
- Throughput on the S1 view, 2000 env steps each, shared machine: hard floor 342.1 env steps/s (246.2 MB resident), carpet 261.9 env steps/s (318.2 MB).
- Not run: bar_form on V0. r5lib, r5core, r5env, g8a and three of the four test files are byte-identical to d500d89, so the reviewed run stands. No training, no policy episode, nothing on a look set, no suite beyond these files.

3. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` (tip cd68f2c):
 designlab/stage2/r5core.py | 66
 designlab/stage2/r5env.py | 148
 designlab/stage2/r5lib.py | 125
 stage2/design/CE_RULE_WORDING.md | 301
 tests/test_stage2_r5core.py | 159
 tests/test_stage2_r5env.py | 522
 tests/test_stage2_r5lib.py | 100
 7 files changed, 1261 insertions(+), 160 deletions(-)
This round's own change (cd32e19): CE_RULE_WORDING.md | 75, tests/test_stage2_r5env.py | 80; 2 files, 135 insertions, 20 deletions.
`git status --porcelain`: empty (also in the main checkout). No new output path in the worktree.

4. COMMITS (oldest first)
05150a2 merge of the working branch (cd68f2c), with the two lines
cd32e19 the wording file follows the review; a test for each point
Head: cd32e19a7201582fa097b11a07080128ae06cf3c

FOR THE LEAD
a. Finding 1: choose between options a, b and c above.
b. The integrator of ruling 13 should read CE_RULE_WORDING.md at cd32e19, not d500d89. Entries 3a, 3b and 8 differ, and entry 9 asks for nothing. The draft at 0d10a94 holds none of CE's new sentences yet.
c. Finding 4, an alternative I did not take: your ruling lists "the config's named freeze, the input files and the task names", and RESUME_KEYS also holds `tasks`. Keeping `tasks` with "as built" is K4's "resolved task table"; dropping it changes r5env.py and its source hash. I recommend keeping it, and adding "as built" to WPC_CT.md.
d. Finding 5: options above.
e. Finding 7: the three items above are yours.
f. The three wording tests read a working paper. They skip where the file is gone, but a later edit of those entries must keep the pinned phrases or drop the tests.
g. Seen while checking, not in the review, not changed: REWARD_RULE says in one place that a playback scores "progress below 1e-4" and in another that the speed ripple "charges nothing" (measured up to 2.1e-05). Wording entry 4a already says "below 1e-4". Aligning the string changes r5core.py's hash; I would do it with the next change to that file.
h. The wording file refers to this review without a path, because the review is not in the repository on this branch.

Logs and the quote check are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ce_fix3/ (quotes.py, without_fix.log, t_lib_core_head.log, t_env_g8a_head.log).
