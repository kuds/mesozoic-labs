Verdict: sound

Verdict: sound. The report matches the records and the tool output I produced myself. Every command is the one the chain names, with the right set, body and record path. The stop after G9-run is what the task's stop rule and rule Z2 require. The failure is correctly put down to the code, not to an episode, and goes to the lead. No code was changed, and the branch merges cleanly as a fast-forward.

Findings, most serious first (none needs a fix in this step):

1. G9-run (d) fails only because of how `gates2.part_d` treats conditions that were not run. This is confirmed, and it is the lead's to decide. I re-ran `gates2.part_d("stage2/tb/bar/r0l","75-M-H-10","V6","seeds3042")` in process, writing nothing:
   - It gives ok False. Its records, bar_records_10ms, problems, expected and missing are all equal to the committed g9_run.json.
   - The failing records are exactly turn/q90_left, turn/q90_left_mu0.54, turn/q90_right and turn/q90_right_mu0.54. Each reads "its bar record is 'not run', not done" and has 0 episodes.
   - All 13 expected records are ok, missing is [] and problems is []. The 7 ten-ms bar_walk/bar_turn records are ok: the three walks reached their window 20/20 with no missing toe readings, and the four q90 records have 0 episodes.
   - If the companions of conditions that are not expected are left out, (d) would read True. That is my reading, not the code's.
   - The cited lines check out: line 1335 globs every companion, and lines 1351-1352 flag any status other than "done".
   - The rest of the gate code treats a condition with no reference as "not run, ok": G9-ref (a) at gates2.py line 1019, G8a's not_run, and `_expected_conditions`. Rule Z2 also says that a reference marked not available leaves its conditions "not run" (H3). The test `test_g9_run_part_d_and_e` has no case for a "not run" companion.
   - So the report's diagnosis holds. The four q90 references are not available in set.json: turn90_pi4_left/right and turn90_1_left/right, over a leg drive's Stage 0 limit.
   - Advice for the lead: if `part_d` is changed, keep the failed g9_run.json (cf6d0492) beside the new one, and label the change as Z2 requires.

2. Minor side note for the same fix, not a defect of this step: the 4 "not run" companions carry `g9_ref: null`, while the 13 that were run cite af627ba90a65 (ruling 37 (c) says "the companion always holds g9_ref").

3. Minor wording: the report's G8a "delays 0 and 5" is the record's raw `delays` value of [0, 5]. The episodes' rows are R0@0ms and R0@10ms, as the commit message says.

My numbers beside the records':

- **V2 G8a** (log: `gates2 g8a --body 75-M-H-10 --set V6 --record stage2/tb/gates/V6/75-M-H-10/g8a.json`, exit 0, wall_s 263.49, "32 of 32 episodes identical, 2 conditions not run: PASS"):
  - File sha256 is a3e7e47d...413a and the record's own sha256 check is ok (b1642a4c...9a3e).
  - The record shows identical 32 of 32, compared_steps == reference_steps in 32 of 32 episodes, ends_alike in 32 of 32, first_difference None, differences [] and complete True.
  - It ran 8 conditions × seeds 3042, 3043 × k 0 and 5. not_run is q90_left and q90_right, which matches Z1's "conditions the set has references for".
  - It was made at commit 73d828e7, dirty False, freeze 0 (manifest 2eeaebe8...8def), and cites g9_ref.json af627ba9 (pass, 17:22:58Z). G8a started at 17:35:55Z, after G9-ref, so X1's order holds.
  - `require_g9_ref('75-M-H-10','V6')` passes on the latest freeze.
- **V3 R0L's records** (log: chain command with `--conditions @gated,walk/nominal --out stage2/tb/work/bar/r0l --procs 2`, exit 0, wall_s 409.02):
  - The session in eval/summary.json is at 2b6e3b2a, dirty False, procs 2.
  - Re-derived from the eval records, succeed counts: walk/v0.4 20, v0.4_mu0.54 13, nominal 0, stop/two_step 20 and its _mu0.54 20.
  - Nudges at mu 0.9 (forward/backward/left/right): 5/9/8/2. At mu 0.54: 7/10/8/6.
  - Read/fell from part (d): 12/8, 13/7, 12/8, 9/11 and 9/11, 10/10, 13/7, 8/12. No errors.
  - All of this matches the report's table.
- **nb collect**: exit 0, "296 files (296 copied, 0 already there)". COLLECTED.json file sha256 is 00c26615...354f and its own sha256 check is ok.
  - All 296 listed files match their sha256 on disk, and the live copies in work/ are identical.
  - It is dirty only because of its own untracked destination, "?? .../stage2/tb/bar/".
  - eval/summary.json is 6c4ecca7...be1a and look/summary.json is a77f42ad...d097.
- **V3 G9-run**: exit 1, wall_s 1.87. File sha256 is b9c02766...4239 and its own sha256 check is ok (899f7d7f...c3ae).
  - parts_ok: d False, e True. Part (e) does not apply ("only the 60 cm body and head...").
  - It was made at aaa042de, dirty False, on freeze 0.
  - followed: "no training run of this body and head starts; it goes to the lead".
- **Not run**: V5, V6, V7 and V4 did not run, which is correct after a failed gate. stage2/tb/dev_episodes.json and stage2/tb/validation/ do not exist, and there are no chain done-markers. None of my or the executor's processes are running.

Rules, hashes and merge readiness:
- The bit-identity check from the worktree prints `45719d03...528d c0027992...c1e1`, and the records' `sources` carry the same two hashes.
- The diff from 73d828e7 to HEAD is 299 added files only: g8a.json, g9_run.json, and COLLECTED.json plus 296 files under stage2/tb/bar/r0l/V6/75-M-H-10.
- Across the 12 commits on top of the working branch, nothing touches code, V0, F25F9 or F25F9cl.
- Every one of the 12 commits is single-parent and ends with exactly the two attribution lines. No other line names a model.
- The working tree is clean. The main checkout is at dd431e6c with a clean status, and wp-l is at 8bc47346.
- `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` gives f8c77b1d, which is HEAD's own tree: no conflict and a pure fast-forward. The tip to bring in is cf6d0492aacced55add01cccf34526e701c06312 (12 commits ahead, 0 behind).
- Disk: 15G free.

What I did not check:
- I did not re-run G8a or R0L's episodes, which would take about 11 minutes on shared cores.
- I did not run the g9-run CLI to write a record. I re-derived part (d) in process instead, writing nothing.
- I did not run the test suite or `sets verify`.
- I did not read every committed bar record's content beyond the sha256 checks and the deciding counts.
- I did not review the earlier freeze and G9-ref commits beyond their trailers and the G9-ref record's own sha256, pass and freeze.

My scripts are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/freeze75/check_validate/ (check_records.py, rederive_d.py).
