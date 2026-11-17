Verdict: sound

VERDICT: sound. G9-ref passed on all three 75 cm look sets, each on its freeze 0. I re-ran G9-ref on all three into scratch, and the records came out identical in every part (a), (b), (c) and (f), in the inputs and in the freeze cited. Only `at`, `commit`, `wall_s` and `sha256` differ. I found no rule broken and no wrong verdict. One decision is still the lead's (finding 1). The other findings are small wording errors and a code quirk.

FINDINGS, most serious first

1. Every freeze used `--allow-mixed`. This is a decision for the lead, not a fault. The report raises it openly.
   - All three freezes record `mixed_sources: true` and `dirty: false`.
   - Rule S2S-freeze allows a mixed freeze when it is recorded. The moves.py condition (A4), which no flag waives, held: the freeze did not refuse.
   - The P42 build and check reports had both told the lead that a freeze would need `--allow-mixed` or a full rebuild. The task did not say which.
   - Verify still notes that the walks, the start and nominal_to_v0.3/0.35/0.4/0.45/stop1 were "built by other code ... not compared with what today's code builds".
   - The report's fallback is incomplete. It says unmixed freezes would need "a full rebuild in an empty directory before anything runs on them". But a frozen entry can never be changed or rebuilt. That route therefore means dropping these 9 unmerged, unpushed commits (or making a new set version) first. The lead should confirm `--allow-mixed` before merging.

2. One claim about -X in part (b) is slightly wrong, in the report and in commit 73d828e7's message. Both say "all four terms differ by 0.0 on all six rungs".
   - In the record, bob, head and tail are exactly 0.0.
   - Toe-off differs by 1e-9 to 3e-9 rad (relative 9e-9 to 3.1e-8), against limits of 0.02 rad and 10 %.
   - This does not change the verdict. The commit message cannot be amended under the rules.

3. Part (a) wording is loose. The report puts walk/v0.5 among "references not available". The refcheck's own reason is "no transition into v0.5 (no transition within the gates)": walk v0.5 itself is frozen and available; only nominal_to_v0.5 is not.

4. A code quirk for the lead; the worker is not at fault. In part (b), the record's `comparand.files[*].entry_sha256` holds each file's sha256, not its entry hash. This comes from gates2.py near line 988, which reads `comparand.refs[n].get("sha256")`. For nominal it shows b3dd607d... where the manifest has 05c18268.... The record's `inputs.comparand` carries the correct entry hashes.

5. Not an error: the largest join ratio, 0.022659, is a four-way tie. The tied joins are turn_0.35_right, turn_0.5_right, stop and start_stop: all v0.4 into a move, at neck_pitch, 4.334e-05 against 0.001913 rad. The report names one of them.

MY CHECKS AGAINST THE RECORDS
- **Commands:**
  - The refcheck records sit at `stage2/sets/<S>/<B>/refcheck.json`, so `--out stage2/sets` was used.
  - The G9-ref records sit at `stage2/tb/gates/<S>/<B>/g9_ref.json`, matching the chain's V1 (lines 159 and 162) and the run plan's V1.
  - The order was V6/75, then V6noF5/75, then V6/75-X. Each set's refcheck was committed before its freeze, so each freeze's `commit` field holds the record it cites.
  - The reflog shows a fast-forward from 4ad729c5 to dd431e6c, then 9 plain commits: no amend and no reset.
- **Hashes (my sha256sum equals the report for all 9 files):**
  - refcheck: 5efd1045..., 37389fa9..., 1b7308e9...
  - set.json: a901500f..., 20a4c351..., 4fd9ceba...
  - g9_ref files: af627ba9..., e8387ba1..., 940701f4...
  - Each record's own field (862176970b..., d2bca1f1..., 8efa5e7e...) passes `record_sha_ok`.
  - Recomputed manifest sha256: 2eeaebe8..., ff2e79b3..., 77516680... (equal). Each freeze has 17 entries, index 0, `dropped` none, and cites its refcheck with pass_ true, failing {} and not_run [q90 left/right, t90_1 left/right, walk/v0.5].
  - In each set.json only `freezes` changed ([] became one freeze).
- **Verify and status now:** each set prints "verify ok: 17 references, 11 not available, 0 failures; freeze 0" and "freeze 0; lead-in start". The worker's own logs show the pre-freeze lines quoted in the report ("not frozen; lead-in start" and the 2-source-state note).
- **Refcheck re-run:** I re-ran the placed refcheck on V6/75-M-H-10 into scratch. It exited 0 with "pass", "placement: every turn and stop item ... placed". Its items and input files are identical to the committed record.
- **G9-ref re-run:** all three passed, "(a) True, (b) True, (c) True, (f) True: PASS; none". `require_g9_ref` accepts each committed record on the latest freeze.
- **Deciding numbers, re-derived from the records:**
  - **(a)** It cites the freeze's refcheck: files match the freeze, plans placed, no failing item, nothing still available, no item blocked.
  - **(b), V6:**
    - Nominal: bob_rel −0.000532, head 0.00486°, tail −0.000613 cycle, toe-off −0.00109 rad (−0.36 %).
    - v040: −0.000573, 0.00272°, 0.000771, −0.00153 rad (−2.74 %).
    - hash_ok true on both; fr05 is not in L1_FREEZE.
    - Six walks are finite with 18 to 25 steps. The played items have at least 6 steps and 7 lift-offs (the rule asks 4 and 4).
  - **(b), V6noF5:** bob, head and tail are exactly 0.0 on six walks; toe-off is "not read" and reads about −4e-8 rad.
  - **(c):** each degraded playback fails only its expected term; the unchanged and cut playbacks pass. The tail tenth reads 0.012496 against 0.124955 rad. There is no flat-feet playback on V6noF5.
  - **(f):** largest jump is 2.297e-4 rad (V6), 2.825e-4 (V6noF5) and 2.233e-4 (-X). No join exceeds its walk's step. The 2 s zero action is finite on all 12 available tasks. The v0.5 screen block hash equals the entry's.
- **Rule X1:** every step was kinematic (`kinematic_only` true). There are no chain markers, no `stage2/tb/work`, and no dev_episodes or bar records.
- **Round sets and code:** `git diff dd431e6c..HEAD` touches only the 9 record files: no code, tools or tests, and nothing under V0, F25F9 or F25F9cl. V0, F25F9 and F25F9cl still verify ok on freeze 0 with the same counts.
- **Bit-identity:** prints 45719d03...528d c0027992...c1e1 in both the worktree and the main checkout.
- **State of the other checkouts:** the main checkout is at dd431e6c with a clean `git status`; wp-l is at 8bc47346.

MERGE READINESS
- The working branch `claude/mesozoic-trex-stage2-refine-4jffre` is at dd431e6c, an ancestor of `wt/wp-p42`, so this is a fast-forward.
- `git merge-tree --write-tree` gives tree 8c242f94 with exit 0, equal to `wt/wp-p42`'s own tree.
- The tip to bring in is 73d828e7.
- All 9 commits end with exactly the two attribution lines and name no model elsewhere. The author "Claude <noreply@anthropic.com>" matches the repo config and earlier commits.

NOT CHECKED
- I did not re-run any freeze, since that would write set.json.
- I re-ran the refcheck only on V6/75-M-H-10, not on V6noF5 or -X.
- I ran no test suite.
- I did not check the worker's claims about disk use or process count during its run, or every per-item number in the turn items beyond steps and lift-offs.
- I did not hash the arrays directly; verify and G9-ref's inputs check did that.

My scripts, logs and scratch outputs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/freeze75/check_freeze/. The worktree is still clean at 73d828e7.
