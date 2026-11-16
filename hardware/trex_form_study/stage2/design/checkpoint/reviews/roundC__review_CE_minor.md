# review:CE-minor (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

REVIEW OF WP CE, SECOND FIX ROUND (wt/wpc-ce at cdc4762; working branch tip now cd68f2c). Simulation tooling only; nothing here is a result.

VERDICT: merge as it stands, but take commit d500d89, not the branch tip cdc4762 (finding 1). No blocker, no major. All eight findings are fixed as asked, each with a test that fails without the fix. Seven minor points follow; none needs a change to CE's source files.

FINDINGS (all minor)

1. Rules: the merge commit cdc4762 lacks the two trailer lines. Its message is only "Merge branch 'claude/mesozoic-trex-stage2-refine-4jffre' into wt/wpc-ce". The four build commits (011f6e2, b4c223e, 4523ed6, d500d89) end with the two lines and name no model elsewhere.
   - The working branch has moved 23 commits since (82ba1ad to cd68f2c), so cdc4762 no longer brings the branch up to date either.
   - `git merge-tree --write-tree cd68f2c d500d89` and `... cd68f2c cdc4762` both exit 0 with the same tree, d7ef7ebd. CE's nine files are identical at d500d89 and cdc4762.
   - Fix: merge d500d89 with a merge commit of the lead's own and leave cdc4762 unmerged, as was done for CB's 5aaf021 (ROUND_C_RULINGS_16NOV.md). cdc4762 is on no remote.

2. stage2/design/CE_RULE_WORDING.md:463-469 (entry 9, X1): its "Old" quote is no longer verbatim in the draft. The lead wrote X1's lift into the draft at 405c218 (14:38 UTC), after the builder's last commit (14:20 UTC).
   - My quotes.py: 20 of 20 Old quotes verbatim in the draft at 984f81d, 19 of 20 at wt/prereg-b's tip 0d10a94.
   - Fix: at integration, treat entry 9 as already in the draft. Not the builder's error.

3. CE_RULE_WORDING.md:152-180 (entries 3a, 3b): W3's new text lacks the two clauses that `REWARD_RULE` now holds: "kept between 1 and 256 env steps", and "where T is not above zero, (a) is not read". The finding asked for a and e only in the file, so this is within the brief (the builder's point j). Fix: add the two clauses at integration, Label A.

4. CE_RULE_WORDING.md:405-419 (entry 8's sentence for K4) and r5env.py:122 (`RESUME_KEYS`): `tasks` is among the keys compared on a resume, and it holds the task weights in force. K5 calls those run state.
   - The docstring gives the caveat ("compare the description of the env as built"); the sentence proposed for K4 does not.
   - Fix: add "of the envs as built" to the K4 sentence and to CT's brief.

5. r5env.py:769-783 (`env_configs`), for the lead. The configs name the tasks the records hold today. A run started after a record was edited behind the latest freeze is therefore accepted with one task fewer.
   - My probe_k4.py, case B (change marked `available: false`, no new freeze): a config made before the edit is refused ("task walk_v0.4 was in the table under freeze 0 and is absent now"); a config made after it is ACCEPTED with ['stand', 'walk_nominal']; a hand-made `tasks=()` is refused.
   - T2 allows this for a new run, and `absent` lists the task with its reason.
   - Options: leave it; or make `env_configs` refuse when the names differ from the table of the latest manifest. I would leave it.

6. r5lib.py:726-741 (`_check_crouch`), the builder's point c shown on a case. Probe case C: a later freeze adds a walk whose entry records crouch 0.06. The stored config names only stand and walk_nominal, and is refused: "set.json records crouch_frac 0.03, its walk entry 'v0.4' records 0.06". It fails closed; a set has one crouch by its own rule (S2S-crouch). No change.

7. For CT: its worktree (wpc-ct, at cd68f2c) does not hold this round, and `grep` finds no line on the resume comparison in stage2/design/WPC_CT.md. Merge CE before CT writes its resume code. The API additions (`env_configs(..., root=)`, `from_plan(..., heading=)`, the new description keys) are not in the design's section 2.2 listing.

THE EIGHT FINDINGS, EACH CHECKED BY A MUTATED COPY
Method: mutated copies of the three sources in my scratch directory, loaded under their module names by a pytest plugin; the worktree was not touched. 23 mutations, every one fails at least one test (muts.log). The unmutated run through the same plugin passes.

| Finding | Mutation | Tests that fail |
|---|---|---|
| 1 | early return when the named freeze is the latest | named_freeze_is_the_latest, set_freeze_started |
| 1 | table not compared when latest; manifest hash not checked | named_freeze_is_the_latest (each) |
| 2 | no crouch check | the r5lib crouch test, the env "refused" test |
| 2 | nominal walk only; an entry without a crouch accepted | the r5lib crouch test (each) |
| 6 | `from_plan` ignores heading; heading not in the cache key | plan_can_name_the_references_yaw, turn-in-place playback [left, right] (each) |
| 7 | `env_configs` names no task | task_weights_and_env_configs, env_configs_name_their_tasks |
| 7 | last index, not first | set_freeze_started |
| 7 | ValueError for a task that left (two forms) | named_freeze_is_the_latest, leaves_the_table |
| 7 | no `task_files`; no `task_names` | describe_holds, later_freeze_accepted, env_configs_name_their_tasks |
| 7 | `task_files` is every file read | env_configs_name_their_tasks |
| 7 | `task_names` gives the whole table | resolved_table (r5lib), env_configs_name_their_tasks |
| 8 | inline draw in reset | envs_own_draw_at_the_end_of_the_last_span |
| 3c | duty without the upper bound; without the lower | dutys_stride (each) |
| 3d | clearance (a) without the skip | clearance_a_is_not_read |
| 3f | R6 overlay not on the home angle | in_r6_the_head_overlay [both bodies] |
| 3e | flags read at the clock before the step | what_is_read_at_every_physics_step and 8 playback tests |
| 3a | first reading shifted by 0.1 mm | first_env_steps_displacement and 3 progress tests |

WHAT I CHECKED AND FOUND RIGHT
- Scope: `git diff claude/mesozoic-trex-stage2-refine-4jffre --stat` lists seven files, all CE's (r5core, r5env, r5lib, their three test files, CE_RULE_WORDING.md); "7 files changed, 1132 insertions(+), 146 deletions(-)". g8a.py and its test are unchanged. `git status --porcelain` is empty in the worktree and in the main checkout.
- Record-key hashes, printed in the worktree: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- r5core.py: only rule strings and docstrings changed. Each of the six sentences matches the code I read (the `_com0` read in `begin`, `substep`'s clock, the clamp at line 765, `swing <= 0.0` at line 789, `Core.targets`).
- `_check_freeze` (r5env.py:302-366): one path for the latest and for an earlier freeze. `sets.freeze` appends nothing when the manifest equals the previous one and refuses a dropped or changed entry, so "the first freeze that holds the manifest" is the start index on a real set.
- `crouch_frac` is in `setio.WALK_SUBSET`, so a freeze does cover each walk entry's value. V0 and F25F9cl pass the check (0.03, 0.05).
- Probes on synthetic sets (probe_k4.log):
  - R6 across a later freeze that adds a task: the changed keys are exactly the five `LATER_FREEZE_KEYS`; `RESUME_KEYS`, the span and `core` are equal.
  - An entry removed from the records behind the latest freeze is refused.
- The real V0 from `env_configs` (a build, no episode): 15 tasks named, the same as the library's table; freeze 0, 34a820e5...; the config survives a JSON round trip; no path in the description. Output is identical under PYTHONHASHSEED 0 and 7.
- Wording file numbers, measured again by me:
  - Kinematic playbacks of V0's 15 tasks and Stage 1's 4: heading 0.3511, 0.1395, 0.0447 on 98/99, 148/149, 207/208 steps; yaw gap 8.97, 5.37, 3.76 degrees; tracking minimum 0.9891 to 0.9938 on the walks, 0.9618 in the quick turn; progress up to 2.11e-05 (V0) and 8.32e-06 (Stage 1); every other penalty 0.
  - First env step: 0.3027 m/s read, 0.3384 over 20 ms, reference 0.3256, terms 0.9913 and 0.9970.
  - The reset's reading equals the reset pose's (difference 0.0) at stance and mid starts, with and without a delay.
  - Per-physics-step loads: 2.8e-09 body weights from the pass before, 1.20 from the pass after; heights 0.92 mm from the pass after.
  - Mid-start loads: at most 0.76 (k = 0) and 2.06 (k = 5), against 2.53 to 4.38 under the home targets.
- All 8 commit ids in the wording file are ancestors of HEAD; all 27 test names exist. Entry 10 (T3) matches `MID_START_RULE` and the reset's code; commit 225f4b3 did change both. Entry 1b follows the ruling (STATUS.md, 03:02 UTC, ruling 1).

TESTS (exact summary lines, at cdc4762)
- tests/test_stage2_r5lib.py tests/test_stage2_r5core.py: "62 passed in 46.90s"
- tests/test_stage2_r5env.py tests/test_stage2_g8a.py: "43 passed, 1 skipped in 83.32s (0:01:23)"; the skip: "tests/test_stage2_r5env.py:675: slow: set STAGE2_SLOW=1"
- tests/test_stage2_tbenv.py: "11 passed in 0.70s"
- STAGE2_SLOW=1, -k walks_on: "1 passed, 35 deselected in 6.92s"
- With PYTHONHASHSEED=3: "62 passed in 51.70s" and "43 passed, 1 skipped in 90.95s (0:01:30)"
- ruff on the eight files: "All checks passed!"

BAR FORM (my run; exit 0; main checkout status and the listing of stage2/sets unchanged before and after)
`g8a.bar_form("75-M-H-10", "V0", ("walk/nominal",), (3044,), (0, 5), root=<main checkout>/stage2/sets)`:
- R0@0ms: "reference_steps": 7030, "env_physics_steps": 7030, "compared_steps": 7030, "first_difference": null, "reference_fell_step": null, "env_terminated_step": null, "ends_alike": true, "identical": true
- R0@10ms: the same fields with the same values, "identical": true
- document: "identical": 2, "of": 2, "not_run": [], "passes": true; set_freeze index 0, 34a820e5...; "dirty": false
- track_b source hashes: r5lib 615e87cb..., r5core a021696a..., r5env 15e2adc0..., g8a b01013a3... (the builder's)

THE BUILDER'S POINTS FOR THE LEAD
- a. Merge commit: fast-forward to the tip (the commit enters the pushed history), merge d500d89 (same tree, no rewrite), or amend (a rewrite). Choose d500d89.
- b. Manifest hash also checked for the latest freeze: keep. `bar.check_manifest` does the same for every freeze, and it fails closed.
- c. Crouch for every walk entry the library reads, or this env's tasks only: keep the builder's. The library builds every held task's track with the one frequency.
- d. Index by the first freeze holding the manifest, or a config field: keep; no second freeze with the same manifest can come from `sets.freeze`.
- e. `config` stays as given; a hand-made `tasks=()` is still refused: keep. The design already stores `env_configs [8 x R5Config.describe()]`.
- f. InputError for any named task of the table that the set lacks: keep.
- g. Two body numbers in `HEAD_RULE`: keep them (a test pins them and fails loudly on a rebuilt model), or move them to the wording file. I would keep them.
- h. R6 jaw target 0.1485 and 0.1493 rad from the limit: accept and write entry 6c into E5. The alternatives (overlay on zero in R6, or a lower top of the jaw range) change code for 1.5 mrad of a 50 mrad allowance.
- i. Flags 2 ms after the contact reading; first step about 10 % low: accept both as stated. Dividing the first step by 18 ms would change the core's code and hash for 0.006 of one term on one step.
- j. See finding 3.
- k. No heading in the env's `options["plan"]`: leave until a turn-in-place condition exists; rows use `from_plan`.
- l. The extra runs were kinematic playbacks, a few env steps on the S1 view, and a V0 build. None was on a look set; I reproduced them.
- m. For CT: agreed; see findings 4 and 7.

NOT CHECKED
- Anything on a look set (V6 and its variants: not read at all).
- Any training run or policy episode.
- The suite beyond the five files.
- A later freeze made by `sets.freeze` itself (fixtures only).
- K4 on a -X body.
- The unchanged wording entries (2, 5, 7, 9's episode count).
- F25F9 cannot be read by the library at all (its nominal walk is not available: "refs/nominal.npz is not listed"); that predates this round.

My scripts and logs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_ceminor_wave2a/ (mutate.py, run_muts.sh, muts.log, probe_k4.py and .log, probe_real.py, probe_env.py, probe_playback.py, probe_substep.py, quotes.py, barform.out). I changed no tracked file and committed nothing.
