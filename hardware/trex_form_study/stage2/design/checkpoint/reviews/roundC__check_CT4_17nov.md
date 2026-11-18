VERDICT: merge after the listed fix (finding 1, major). Everything else ruling 17 asks of the trainer is done as the ruling says, and each change has a test that fails without it. Tip to merge after the fix: wt/wpc-ct, currently 1ecd32d121f2dbf7d8d4a993dc432238164601ad.

FINDINGS (most serious first)

1. MAJOR. The trainer halts S2-4 when its baseline equals the committed one, if any R0L condition is "not run". Files: monitor2.py:567 (`baseline_equal` digests_compared) and train2.py:1363 (`_expect_baseline`).
   - What happens: `digests_compared` is True only if every episode pair holds a `state_digest` (or `digest`) key. A not-run episode never holds one: lookbar.py:778 builds it without that key. So `_expect_baseline` adds "the state digests were not compared (a record holds none)" and raises Halt (exit 5), even when both records are identical.
   - Why it can happen: rule M1 (settled 16 Nov) explicitly allows a not-run condition in R0L's baseline. `baseline_problem` accepts it. Ruling 29 makes it likely for S2-4: the V6 left turns on 75-M-H-10 fail at J1's leave point and become "not available", and `bar.BarPlanner.missing` then makes turn/q90_left not run. This fails closed, but it would block the go/no-go run until the code is changed.
   - Evidence: I ran /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/check_ct4/notrun_digest.py. It compares two identical records (two read episodes with digests, plus one not-run episode shaped as lookbar makes it) and prints:
     - "lookbar not_run episode holds 'state_digest': False"
     - "baseline_equal: {'equal': True, 'episodes': 3, 'differ': [], 'digests_compared': False}"
     - "RESULT: Halt: j: the run's R0L baseline (monitor/r0l.json) is not the committed one (...): the state digests were not compared (a record holds none) ..."
   - The builder's stub (`stub_episodes`) never produces a not-run episode, so no test covers this.
   - Proposed fix (monitor2.py, within the package):
     - In `baseline_equal`, require digests only for pairs where at least one episode ran (outcome != NOT_RUN). Two not-run episodes are still compared on outcome and survival.
     - Count a digest as compared only when both sides hold a non-empty str. This also closes finding 2.
     - Keep `bool(...)` false when no episode ran.
     - Add "in every episode that ran" to EXPECT_BASELINE_RULE.
   - Tests to add:
     - test_stage2_train2: a committed baseline with one not-run condition, equal on both sides, lets the run train.
     - test_stage2_monitor2: the same case on `baseline_equal` directly.
   - `baseline_same` and `write_s2_10` only report `digests_compared`, so changing it there is safe.

2. MINOR. monitor2.py:567. A digest of None counts as compared.
   - lookbar's block sets `state_digest=None` when taking the digest raised. That fault does not make the episode an error, so a read or fell episode can carry None.
   - Evidence: none_digest.py prints "{'equal': True, 'episodes': 1, 'differ': [], 'digests_compared': True}" for two read episodes whose state_digest is None.
   - The strict reading the builder chose then passes without any digest. The str check in fix 1 closes it.

3. FOR THE LEAD (not a defect). On a mismatch the run halts (Halt, exit 5) rather than being marked `stopped`.
   - M5's five endings leave a halt as the only fit for M3's "the run stops". A resume halts again, because the r0l.json is kept and K4 pins the file.
   - I consider this defensible, as built.

CHECKED AND FOUND RIGHT
- survival_credit:
  - A clock within 1e-9 (relative) of 1.05 x the survival time is credited the survival time itself; below that, the product.
  - The guard `surv <= 0` comes first, so there is no division by zero.
  - S equals mean_survival_s, with error and not-run episodes counted 0 in both.
  - On real episodes: two open-loop R0L episodes on the Stage 1 view, tiny. This is what the lookbar unit test test_the_monitor_reads_these_episodes runs; output went to scratch and was then deleted. Results:
    - walk/nominal 6001: survival 18.38600000000069, credit 18.38600000000069, old product 18.38600000000034, gap -1.9e-14.
    - walk/brisk 6004: gap -2.6e-14.
    - "S == mean: True"; the old product gave "old S == mean: False".
    - So 1e-9 covers rounding with a margin of five orders of magnitude.
- Stand-in:
  - `_stand_in` and `_episodes_by` are gone, and `_episodes` has no fallback.
  - No other module or test references them (grep).
  - EPISODES_BY = "lookbar.episodes".
- monitor2 baseline without a job:
  - `--job` is optional for baseline only. Without `--out` it is an argparse error (exit 2), and `run` still requires `--job`.
  - The refusal note goes under `--out`, and the `train2 monitor` entry passes through.
- --expect-baseline:
  - It is checked after `_baseline()` and before `_learn()`, so before any rollout and at every session.
  - The result is recorded in run.json `baseline_expected` and in the session's `expect_baseline`.
  - Refused with exit 4, before anything is written: a missing, unreadable or non-baseline file, or the option given with --no-gates.
  - The file's sha256 is in the settings, so K4 refuses a resume without it or with another file.
  - Old run.json files without the key still resume (None == None).
  - gates2's K3 (`run_settings_problems`), nb.py and s2rows read named settings keys only.
- No timeout: `_child` uses `proc.wait()` with no arguments.
- Finding 3 of the FU check: the test now covers the `source_change` fallback.
- "Test fails without it":
  - Done on a scratch copy (git archive of HEAD, data directories symlinked; the worktree was never touched).
  - New tests against the working branch's code:
    - old monitor2 with the three new monitor tests: "3 failed, 24 deselected in 1.40s"
    - old train2 with the new tests (-k committed_one/expected_baseline/without_a_timeout/cli_hands): "4 failed, 55 deselected in 3.49s"
    - t6 against the old test file: "1 passed, 55 deselected in 1.66s", so it survived before the new test.
  - The unmutated selections pass in the copy: "4 passed, 23 deselected" and "5 passed, 54 deselected".
  - My 17 mutants are all caught: a_product, a_wide_tol (1e-3), b_fallback, c_job_required, c_no_out_check, d_no_call, d_after_learn, d_no_setting, d_no_digest_req, d_no_gates_ok, d_no_problem_check, d_warn_only, d_first_session_only, e_timeout (wait(timeout=3600)), f_t6.
- Only the package's files differ:
  - `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` exits 0 with tree 30f0711be092979044296162ffd73883d59b7817.
  - That tree differs from the working tip 81a395e1 only in monitor2.py, train2.py and their two tests (435+/170-).
  - The working branch gained two doc-only commits (29f8bb99, 81a395e1, rulings 28-30) after the builder's second merge. They do not conflict and touch nothing of the trainer's.
- Record-key hashes: they print the registered pair, 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1.
- Nothing ran that should not have:
  - `git status --porcelain` is empty in the worktree and the main checkout.
  - No file in the worktree is newer than my first scratch script.
  - No stage2/tb outputs exist.
  - ruff check --no-cache on the four files: "All checks passed!"

EXACT TEST SUMMARY LINES (mine; worktree head 1ecd32d1, standing environment, tools/lowprio.sh, basetemp in scratch)
- tests/test_stage2_train2.py: "56 passed, 3 skipped, 2 warnings in 262.56s (0:04:22)"
- tests/test_stage2_monitor2.py: "27 passed in 36.95s"

COMMITS / WHAT TO BRING IN
- Every commit not in the working branch ends with exactly the two given lines and names no model elsewhere: 1c00b280 (merge), 3047d418, aadebd08, fdc6bc27, b509c475, 983f9671, c8132d8b, 1ecd32d1 (merge). None are missing them.
- After the fix for finding 1 (with its tests) is committed on wt/wpc-ct: merge that branch tip into claude/mesozoic-trex-stage2-refine-4jffre (one merge; it already contains 1ecd32d1).
- Owed outside this package (gates package):
  - stage2/tb/run_trackb.sh V8-s2-4 should pass `--expect-baseline "$BL"`.
  - Its NOTE ("the trainer takes no committed baseline") is now stale.
  - V8-baseline can use `monitor2 baseline --body 75-M-H-10 --set V6 --out ...`.
  - Until then, S2-4 is still checked only after the run, by s2-10.

NOT CHECKED
- The test files of the other packages (lookbar, s2rows, gates2, nb) on this head.
- The STAGE2_SLOW suites (the 3 skipped trainer tests).
- Any real look-set or V6 episode (none is frozen), so the not-run case in finding 1 is shown on a synthetic record shaped as lookbar.episodes makes it, not on real V6 output.

HOUSEKEEPING
- The disk was nearly full during the review (down to 241 MB free; 0 MB once briefly). I deleted only my own temp dirs and scratch copy.
- Kept in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/check_ct4/:
  - Scripts: notrun_digest.py, none_digest.py, r0l_exact.py, tools/mut.py, tools/mkspec.py, tools/spec.json.
  - The working branch's files, under old/.
  - Logs, under logs/: train2.log, monitor2.log, mutants.log, mutants_d.log, mut_baseline.log.
